from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from proceedings_to_eee.evaluation.development_annotation import (
    DevelopmentAnnotationResponse,
    EvidenceAnchor,
    ResultBearing,
    ResultOrigin,
    ReviewConfidence,
    SelectionIntent,
    prepare_development_annotation_package,
    validate_completed_development_annotation_response,
    validate_initial_development_annotation_package,
)
from proceedings_to_eee.io import read_json, sha256_file, write_json, write_jsonl


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    public_repo = tmp_path / "public-repo"
    source_project = tmp_path / "historical-private"
    run_root = source_project / "runs" / "inspected-development"
    private = tmp_path / "private"
    public_repo.mkdir(parents=True)
    records = [
        (
            "paper-a",
            [
                ("System A", "System A 0.91", SelectionIntent.RESULT_PAPER_PRODUCED_CANDIDATE),
                ("column header", "DESCRIPTION SCORE", SelectionIntent.NON_RESULT_HEADER),
                ("sample total", "Total 1500", SelectionIntent.NON_RESULT_SAMPLE_COUNT),
            ],
            "We ran System A ourselves.",
        ),
        (
            "paper-b",
            [
                (
                    "copied baseline",
                    "Copied baseline 0.82",
                    SelectionIntent.RESULT_EXTERNAL_OR_COPIED_CANDIDATE,
                ),
                ("temperature", "temperature = 0", SelectionIntent.NON_RESULT_SETUP_VALUE),
                ("epochs", "epochs [10, 20]", SelectionIntent.NON_RESULT_PARAMETER),
            ],
            "Reported by Prior et al. (2020).",
        ),
        (
            "paper-c",
            [
                (
                    "ambiguous baseline",
                    "Ambiguous baseline 0.75",
                    SelectionIntent.RESULT_MIXED_OR_UNCERTAIN_ORIGIN_CANDIDATE,
                ),
                ("threshold", "threshold = 0.5", SelectionIntent.NON_RESULT_THRESHOLD),
                (
                    "caption",
                    "Table 1: Evaluation overview.",
                    SelectionIntent.NON_RESULT_CAPTION,
                ),
            ],
            "The paper does not state whether this baseline was rerun.",
        ),
    ]
    selected: list[dict] = []
    for paper_id, page_one_records, origin_line in records:
        pdf = f"%PDF-1.4\nprivate fixture {paper_id}\n".encode()
        source_sha = hashlib.sha256(pdf).hexdigest()
        cache_relpath = f"data/sources/{source_sha[:2]}/{source_sha}.pdf"
        source_path = source_project / cache_relpath
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(pdf)
        paper_root = run_root / paper_id
        (paper_root / "private").mkdir(parents=True)
        source_id = f"src_{paper_id.replace('-', '_')}"
        write_json(
            paper_root / "source-manifest.json",
            {
                "schema_version": "source-manifest/0.2",
                "paper_id": paper_id,
                "sources": [
                    {
                        "source_id": source_id,
                        "paper_id": paper_id,
                        "role": "paper",
                        "sha256": source_sha,
                        "byte_size": len(pdf),
                        "cache_relpath": cache_relpath,
                    }
                ],
            },
        )
        page_one = "\n".join(record[1] for record in page_one_records) + "\n"
        page_two = origin_line + "\n"
        write_json(
            paper_root / "private" / "layout.json",
            {
                "source_id": source_id,
                "pages": [
                    {
                        "page": 1,
                        "text": page_one,
                        "text_sha256": hashlib.sha256(page_one.encode()).hexdigest(),
                    },
                    {
                        "page": 2,
                        "text": page_two,
                        "text_sha256": hashlib.sha256(page_two.encode()).hexdigest(),
                    },
                ],
            },
        )
        selected.extend(
            {
                "paper_id": paper_id,
                "page": 1,
                "locator": locator,
                "selection_excerpt": excerpt,
                "selection_intent": intent.value,
            }
            for locator, excerpt, intent in page_one_records
        )
    selection = private / "selection.json"
    write_json(
        selection,
        {
            "schema_version": "mixed-development-annotation-selection/0.1",
            "source_scope": "development-and-error-analysis-only",
            "future_unseen_data_used": False,
            "source_run_id": run_root.name,
            "minimum_result_candidates": 3,
            "minimum_result_papers": 3,
            "required_intents": [intent.value for intent in SelectionIntent],
            "items": selected,
        },
    )
    return public_repo, source_project, run_root, selection


def _prepare(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    public_repo, source_project, run_root, selection = _fixture(tmp_path)
    package = tmp_path / "private" / "package"
    prepare_development_annotation_package(
        selection_path=selection,
        run_root=run_root,
        source_project_root=source_project,
        output_dir=package,
        public_repo_root=public_repo,
        reviewer="reviewer-expert",
    )
    return public_repo, source_project, run_root, selection, package


def test_generic_single_reviewer_package_is_mixed_blank_and_blinded(tmp_path: Path) -> None:
    public_repo, source_project, run_root, selection, package = _prepare(tmp_path)
    manifest = validate_initial_development_annotation_package(
        package_dir=package,
        selection_path=selection,
        run_root=run_root,
        source_project_root=source_project,
        public_repo_root=public_repo,
    )

    assert manifest.item_count == 9
    assert manifest.result_candidate_count == 3
    assert manifest.result_candidate_paper_count == 3
    assert manifest.paper_count == 3
    assert manifest.reviewer == "reviewer-expert"
    assert manifest.future_unseen_data_used is False
    responses = _read_jsonl(package / "response.jsonl")
    assert len(responses) == 9
    assert all(response["reviewer"] == "reviewer-expert" for response in responses)
    assert all(
        response[field] is None
        for response in responses
        for field in (
            "result_bearing",
            "result_evidence",
            "origin",
            "origin_evidence",
            "confidence",
            "notes",
        )
    )
    reviewer_projection = (package / "items.jsonl").read_text() + (
        package / "response.jsonl"
    ).read_text()
    assert "selection_intent" not in reviewer_projection
    assert "result_paper_produced_candidate" not in reviewer_projection
    assert str(source_project) not in package.joinpath("manifest.json").read_text()
    assert not any(path.is_symlink() for path in package.rglob("*"))
    assert len(list((package / "pdfs").glob("*.pdf"))) == 3


def test_completed_response_binds_result_and_origin_to_exact_declared_pages(
    tmp_path: Path,
) -> None:
    public_repo, source_project, run_root, selection, package = _prepare(tmp_path)
    items = _read_jsonl(package / "items.jsonl")
    origin_by_paper = {
        "paper-a": (ResultOrigin.PAPER_PRODUCED, "We ran System A ourselves."),
        "paper-b": (ResultOrigin.EXTERNALLY_SOURCED, "Reported by Prior et al. (2020)."),
        "paper-c": (ResultOrigin.UNCERTAIN, None),
    }
    result_locators = {"System A", "copied baseline", "ambiguous baseline"}
    responses: list[DevelopmentAnnotationResponse] = []
    for item in items:
        result_anchor = EvidenceAnchor(
            source_id=item["source_id"], page=1, exact_excerpt=item["source_excerpt"]
        )
        if item["locator"] in result_locators:
            origin, origin_excerpt = origin_by_paper[item["paper_id"]]
            responses.append(
                DevelopmentAnnotationResponse(
                    item_id=item["item_id"],
                    reviewer="reviewer-expert",
                    result_bearing=ResultBearing.YES,
                    result_evidence=result_anchor,
                    origin=origin,
                    origin_evidence=(
                        EvidenceAnchor(
                            source_id=item["source_id"], page=2, exact_excerpt=origin_excerpt
                        )
                        if origin_excerpt is not None
                        else None
                    ),
                    confidence=ReviewConfidence.HIGH,
                )
            )
        else:
            responses.append(
                DevelopmentAnnotationResponse(
                    item_id=item["item_id"],
                    reviewer="reviewer-expert",
                    result_bearing=ResultBearing.NO,
                    result_evidence=result_anchor,
                    confidence=ReviewConfidence.HIGH,
                )
            )
    write_jsonl(package / "response.jsonl", [row.model_dump(mode="json") for row in responses])

    receipt = validate_completed_development_annotation_response(
        package_dir=package,
        selection_path=selection,
        run_root=run_root,
        source_project_root=source_project,
        public_repo_root=public_repo,
        expected_manifest_sha256=sha256_file(package / "manifest.json"),
    )

    assert len(receipt.responses) == len(receipt.evidence_bindings) == 9
    assert receipt.reviewer == "reviewer-expert"
    assert len(receipt.response_sha256) == 64
    result_bindings = [binding for binding in receipt.evidence_bindings if binding.origin_evidence]
    assert len(result_bindings) == 2
    assert all(binding.result_evidence.page == 1 for binding in result_bindings)
    assert all(binding.origin_evidence.page == 2 for binding in result_bindings)
    uncertain = next(row for row in receipt.responses if row.origin is ResultOrigin.UNCERTAIN)
    assert uncertain.origin_evidence is None
    with pytest.raises(ValueError, match="preserved receipt"):
        validate_completed_development_annotation_response(
            package_dir=package,
            selection_path=selection,
            run_root=run_root,
            source_project_root=source_project,
            public_repo_root=public_repo,
            expected_manifest_sha256="0" * 64,
        )


def test_strict_evidence_rejects_whitespace_pooling_wrong_page_and_missing_origin(
    tmp_path: Path,
) -> None:
    public_repo, source_project, run_root, selection, package = _prepare(tmp_path)
    items = _read_jsonl(package / "items.jsonl")
    item = next(row for row in items if row["locator"] == "System A")
    base = dict(
        item_id=item["item_id"],
        reviewer="reviewer-expert",
        result_bearing=ResultBearing.YES,
        result_evidence=EvidenceAnchor(
            source_id=item["source_id"], page=1, exact_excerpt="System A 0.91"
        ),
        origin=ResultOrigin.PAPER_PRODUCED,
        origin_evidence=EvidenceAnchor(
            source_id=item["source_id"], page=2, exact_excerpt="We ran System A ourselves."
        ),
        confidence=ReviewConfidence.HIGH,
    )
    with pytest.raises(ValidationError, match="resolved origin decision requires"):
        DevelopmentAnnotationResponse(
            **{key: value for key, value in base.items() if key != "origin_evidence"}
        )

    rows = _read_jsonl(package / "response.jsonl")
    rows[0] = DevelopmentAnnotationResponse(
        **{
            **base,
            "result_evidence": EvidenceAnchor(
                source_id=item["source_id"], page=1, exact_excerpt="System A   0.91"
            ),
        }
    ).model_dump(mode="json")
    for index, other in enumerate(items[1:], start=1):
        rows[index] = DevelopmentAnnotationResponse(
            item_id=other["item_id"],
            reviewer="reviewer-expert",
            result_bearing=ResultBearing.NO,
            result_evidence=EvidenceAnchor(
                source_id=other["source_id"],
                page=1,
                exact_excerpt=other["source_excerpt"],
            ),
            confidence=ReviewConfidence.HIGH,
        ).model_dump(mode="json")
    write_jsonl(package / "response.jsonl", rows)
    with pytest.raises(ValueError, match="exact evidence is absent"):
        validate_completed_development_annotation_response(
            package_dir=package,
            selection_path=selection,
            run_root=run_root,
            source_project_root=source_project,
            public_repo_root=public_repo,
            expected_manifest_sha256=sha256_file(package / "manifest.json"),
        )

    rows[0] = DevelopmentAnnotationResponse(
        **{
            **base,
            "result_evidence": EvidenceAnchor(
                source_id=item["source_id"], page=2, exact_excerpt="We ran System A ourselves."
            ),
        }
    ).model_dump(mode="json")
    write_jsonl(package / "response.jsonl", rows)
    with pytest.raises(ValueError, match="result evidence must cite the selected item page"):
        validate_completed_development_annotation_response(
            package_dir=package,
            selection_path=selection,
            run_root=run_root,
            source_project_root=source_project,
            public_repo_root=public_repo,
            expected_manifest_sha256=sha256_file(package / "manifest.json"),
        )


def test_selection_contract_and_private_boundary_fail_closed(tmp_path: Path) -> None:
    public_repo, source_project, run_root, selection = _fixture(tmp_path)
    value = read_json(selection)
    value["items"] = [
        item for item in value["items"] if item["selection_intent"] != "non_result_caption"
    ]
    write_json(selection, value)
    with pytest.raises(ValidationError, match="selection has no item"):
        prepare_development_annotation_package(
            selection_path=selection,
            run_root=run_root,
            source_project_root=source_project,
            output_dir=tmp_path / "private" / "bad-package",
            public_repo_root=public_repo,
        )

    _, _, _, valid_selection = _fixture(tmp_path / "second")
    with pytest.raises(ValueError, match="outside the public repository"):
        prepare_development_annotation_package(
            selection_path=valid_selection,
            run_root=tmp_path / "second" / "historical-private" / "runs" / "inspected-development",
            source_project_root=tmp_path / "second" / "historical-private",
            output_dir=public_repo / "private-package",
            public_repo_root=public_repo,
        )
