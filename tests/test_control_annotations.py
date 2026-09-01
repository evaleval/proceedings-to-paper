from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from proceedings_to_eee.evaluation.control_annotations import (
    DEFAULT_ANNOTATORS,
    PROTOCOL_TEXT,
    AdjudicationRecord,
    AnnotationConfidence,
    AnnotationResponse,
    ResultBearing,
    ResultOrigin,
    measure_agreement,
    prepare_annotation_packet,
    validate_initial_packet,
)
from proceedings_to_eee.io import read_json, sha256_file, write_json


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _annotation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "sealed-fixture"
    paper_root = run_root / "fixture-paper"
    private = paper_root / "private"
    private.mkdir(parents=True)

    rows = [f"Model {index:02d}  0.{index:03d}" for index in range(53)]
    page_text = "Table 1. Fixture results.\n\n" + "\n".join(rows) + "\n"
    source_id = "src_fixture"
    source_sha256 = "a" * 64
    write_json(
        paper_root / "source-manifest.json",
        {
            "paper_id": "fixture-paper",
            "sources": [
                {
                    "source_id": source_id,
                    "sha256": source_sha256,
                    "role": "paper",
                    "cache_relpath": "/private/source-must-not-leak.pdf",
                }
            ],
        },
    )
    write_json(
        private / "layout.json",
        {
            "source_id": source_id,
            "pages": [
                {
                    "page": 1,
                    "text": page_text,
                    "text_sha256": hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
                }
            ],
        },
    )

    proposal_rows = [
        {
            "paper_id": "fixture-paper",
            "page": 1,
            "kind": "table",
            "label": "Table 1",
            "row": f"Model {index:02d}",
            "block_id": f"rblk_{index:02d}",
            "exact_quote": row,
            "confirmed": False,
            "expected_claim_type": None,
            "reason": "LEAKED_PROPOSAL_REASON",
            "rule_id": "sibling_row_needs_label",
        }
        for index, row in enumerate(rows)
    ]
    proposals_path = tmp_path / "control-proposals.json"
    write_json(
        proposals_path,
        {
            "schema_version": "control-proposal-worklist/0.1",
            "status": "proposed-unconfirmed",
            "rows_needing_a_human_label": proposal_rows,
        },
    )
    return proposals_path, run_root


def test_packet_has_53_unique_source_bound_items_and_no_fabricated_labels(
    tmp_path: Path,
) -> None:
    proposals_path, run_root = _annotation_fixture(tmp_path)
    output = tmp_path / "private-packet"

    manifest, manifest_sha = prepare_annotation_packet(
        proposals_path=proposals_path,
        run_root=run_root,
        output_dir=output,
    )

    assert manifest.item_count == 53
    assert manifest.paper_count == 1
    assert manifest.status == "prepared-unlabeled"
    assert manifest_sha == sha256_file(output / "manifest.json")
    assert validate_initial_packet(output) == manifest

    items = _read_jsonl(output / "items.jsonl")
    assert len(items) == len({item["item_id"] for item in items}) == 53
    assert all(item["item_id"].startswith("ann_") for item in items)
    assert all(item["source_id"] == "src_fixture" for item in items)
    assert all(item["source_sha256"] == "a" * 64 for item in items)
    assert all(item["page_text_sha256"] for item in items)
    assert all(
        item["exact_evidence_sha256"]
        == hashlib.sha256(item["exact_evidence"].encode("utf-8")).hexdigest()
        for item in items
    )

    for annotator in DEFAULT_ANNOTATORS:
        responses = _read_jsonl(output / "responses" / f"{annotator}.jsonl")
        assert len(responses) == 53
        assert all(response["annotator"] == annotator for response in responses)
        assert all(
            response[field] is None
            for response in responses
            for field in ("result_bearing", "origin", "exact_evidence", "confidence")
        )
    adjudications = _read_jsonl(output / "adjudication.jsonl")
    assert len(adjudications) == 53
    assert all(
        row[field] is None
        for row in adjudications
        for field in ("adjudicator", "result_bearing", "origin", "exact_evidence", "rationale")
    )


def test_annotator_facing_artifacts_strip_proposal_hints_paths_and_prior_labels(
    tmp_path: Path,
) -> None:
    proposals_path, run_root = _annotation_fixture(tmp_path)
    output = tmp_path / "private-packet"
    prepare_annotation_packet(
        proposals_path=proposals_path,
        run_root=run_root,
        output_dir=output,
    )

    private_rows = "\n".join(
        (output / relative).read_text(encoding="utf-8")
        for relative in (
            "items.jsonl",
            "responses/annotator-a.jsonl",
            "responses/annotator-b.jsonl",
            "adjudication.jsonl",
        )
    )
    for leaked in (
        "confirmed",
        "expected_claim_type",
        "LEAKED_PROPOSAL_REASON",
        "rule_id",
        "sibling_row_needs_label",
        "/private/source-must-not-leak.pdf",
    ):
        assert leaked not in private_rows
    assert read_json(output / "manifest.json")["privacy"] == {
        "contains_item_labels": False,
        "contains_local_paths": False,
        "contains_proposal_hints": False,
        "contains_source_text": True,
        "private_uncommitted_required": True,
    }


def test_packet_generation_is_deterministic_and_never_overwrites_human_work(
    tmp_path: Path,
) -> None:
    proposals_path, run_root = _annotation_fixture(tmp_path)
    first = tmp_path / "packet-one"
    second = tmp_path / "packet-two"
    prepare_annotation_packet(proposals_path=proposals_path, run_root=run_root, output_dir=first)
    prepare_annotation_packet(proposals_path=proposals_path, run_root=run_root, output_dir=second)

    relative_files = (
        "manifest.json",
        "protocol.md",
        "items.jsonl",
        "responses/annotator-a.jsonl",
        "responses/annotator-b.jsonl",
        "adjudication.jsonl",
    )
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes() for name in relative_files
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_annotation_packet(
            proposals_path=proposals_path,
            run_root=run_root,
            output_dir=first,
        )


def test_initial_packet_requires_exact_manifest_files_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    proposals_path, run_root = _annotation_fixture(tmp_path)
    missing_contract = tmp_path / "packet-missing-contract"
    prepare_annotation_packet(
        proposals_path=proposals_path,
        run_root=run_root,
        output_dir=missing_contract,
    )
    manifest_path = missing_contract / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["files"] = []
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="unexpected file contract"):
        validate_initial_packet(missing_contract)

    symlinked = tmp_path / "packet-symlinked"
    prepare_annotation_packet(
        proposals_path=proposals_path,
        run_root=run_root,
        output_dir=symlinked,
    )
    protocol_path = symlinked / "protocol.md"
    external_protocol = tmp_path / "identical-protocol.md"
    protocol_path.rename(external_protocol)
    protocol_path.symlink_to(external_protocol)
    with pytest.raises(ValueError, match="must not use symlinks"):
        validate_initial_packet(symlinked)


def test_response_and_adjudication_models_reject_partial_or_invalid_states() -> None:
    item_id = "ann_" + "a" * 20
    with pytest.raises(ValidationError, match="blank response"):
        AnnotationResponse(
            item_id=item_id,
            annotator="annotator-a",
            confidence=AnnotationConfidence.HIGH,
        )
    with pytest.raises(ValidationError, match="origin must be null"):
        AnnotationResponse(
            item_id=item_id,
            annotator="annotator-a",
            result_bearing=ResultBearing.NO,
            origin=ResultOrigin.PAPER_PRODUCED,
            exact_evidence="No result is reported.",
            confidence=AnnotationConfidence.HIGH,
        )
    with pytest.raises(ValidationError, match="origin is required"):
        AnnotationResponse(
            item_id=item_id,
            annotator="annotator-a",
            result_bearing=ResultBearing.YES,
            exact_evidence="A result is reported.",
            confidence=AnnotationConfidence.MEDIUM,
        )
    with pytest.raises(ValidationError, match="partial adjudication"):
        AdjudicationRecord(item_id=item_id, adjudicator="adjudicator-c")


def _completed_response(
    item_id: str,
    annotator: str,
    result_bearing: ResultBearing,
    origin: ResultOrigin | None,
) -> AnnotationResponse:
    return AnnotationResponse(
        item_id=item_id,
        annotator=annotator,
        result_bearing=result_bearing,
        origin=origin,
        exact_evidence="Exact fixture evidence.",
        confidence=AnnotationConfidence.HIGH,
    )


def test_agreement_is_joint_nominal_pre_adjudication_and_undefined_is_not_zero() -> None:
    first_id = "ann_" + "1" * 20
    second_id = "ann_" + "2" * 20
    left = [
        _completed_response(first_id, "annotator-a", ResultBearing.NO, None),
        _completed_response(
            second_id,
            "annotator-a",
            ResultBearing.YES,
            ResultOrigin.PAPER_PRODUCED,
        ),
    ]
    right = [
        _completed_response(first_id, "annotator-b", ResultBearing.NO, None),
        _completed_response(
            second_id,
            "annotator-b",
            ResultBearing.YES,
            ResultOrigin.EXTERNALLY_SOURCED,
        ),
    ]

    summary = measure_agreement(left, right)
    assert summary.denominator == 2
    assert summary.raw_agreement_count == 1
    assert summary.raw_agreement == 0.5
    assert summary.cohen_kappa == pytest.approx(1 / 3, abs=1e-6)
    assert summary.kappa_status == "defined"
    assert summary.confusion_matrix["result/paper_produced"]["result/externally_sourced"] == 1

    no_variance = measure_agreement(left[:1], right[:1])
    assert no_variance.raw_agreement == 1.0
    assert no_variance.cohen_kappa == "undefined"
    assert no_variance.kappa_status == "undefined_no_expected_variance"
    assert no_variance.categories == [
        "not_result",
        "uncertain_result_bearing",
        "result/paper_produced",
        "result/externally_sourced",
        "result/uncertain",
    ]
    assert no_variance.confusion_matrix_counts == [
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]


def test_protocol_states_blinding_preservation_adjudication_and_agreement_rules() -> None:
    normalized_protocol = " ".join(PROTOCOL_TEXT.split())
    required_phrases = (
        "Two annotators label every item independently",
        "must not inspect reference YAML",
        "Preserve both originals",
        "third person adjudicates",
        "short verbatim source excerpt",
        "unweighted Cohen's kappa",
        "before adjudication",
        "report `undefined`, never zero",
    )
    assert all(phrase in normalized_protocol for phrase in required_phrases)
