from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from proceedings_to_eee.cli import app
from proceedings_to_eee.evaluation.control_annotation_workflow import (
    OPEN_DEVELOPMENT_SCOPE,
    PRACTICE_EXAMPLE_ID,
    AdjudicationReason,
    AnnotationCompletionManifest,
    AnnotationWorkspaceManifest,
    lock_completed_responses,
    prepare_adjudication_workspace,
    prepare_annotation_workspace,
    validate_adjudication_workspace,
    validate_annotation_workspace,
    validate_completed_response,
    validate_completion_bundle,
    validate_workspace_response,
)
from proceedings_to_eee.evaluation.control_annotations import (
    AdjudicationRecord,
    AnnotationConfidence,
    AnnotationResponse,
    ResultBearing,
    ResultOrigin,
    prepare_annotation_packet,
)
from proceedings_to_eee.io import read_json, sha256_file, write_json, write_jsonl


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _workflow_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project_root = tmp_path
    run_root = project_root / "runs" / "sealed" / "frozen-seven"
    proposal_rows_by_paper: list[list[dict]] = []
    for index in range(7):
        paper_id = f"paper-{index}"
        paper_root = run_root / paper_id
        private = paper_root / "private"
        private.mkdir(parents=True)
        source_bytes = f"%PDF-1.4\nprivate fixture {index}\n".encode()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        source_path = project_root / "data" / "sources" / source_sha[:2] / f"{source_sha}.pdf"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(source_bytes)
        source_id = f"src_{index:020d}"
        row_count = 8 if index < 4 else 7
        paper_rows = [
            f"Fixture model {index}-{row_index}  0.{index + row_index + 1:03d}"
            for row_index in range(row_count)
        ]
        page_text = "Table 1. Private fixture results.\n" + "\n".join(paper_rows) + "\n"
        write_json(
            paper_root / "source-manifest.json",
            {
                "schema_version": "source-manifest/0.2",
                "paper_id": paper_id,
                "title": f"Fixture paper {index}",
                "sources": [
                    {
                        "source_id": source_id,
                        "paper_id": paper_id,
                        "role": "paper",
                        "original_uri": f"https://example.invalid/{paper_id}.pdf",
                        "resolved_uri": f"https://example.invalid/{paper_id}.pdf",
                        "retrieved_at": "2026-08-26T00:00:00Z",
                        "sha256": source_sha,
                        "byte_size": len(source_bytes),
                        "media_type": "application/pdf",
                        "cache_relpath": source_path.relative_to(project_root).as_posix(),
                        "access_status": "available",
                        "license_disposition": "private_use_only",
                        "notes": [],
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
                        "text_sha256": hashlib.sha256(page_text.encode()).hexdigest(),
                    },
                    {
                        "page": 2,
                        "text": "A quote that appears in this paper, but on the wrong page.\n",
                        "text_sha256": hashlib.sha256(
                            b"A quote that appears in this paper, but on the wrong page.\n"
                        ).hexdigest(),
                    },
                ],
            },
        )
        proposal_rows_by_paper.append(
            [
                {
                    "paper_id": paper_id,
                    "page": 1,
                    "kind": "table",
                    "label": "Table 1",
                    "row": f"Fixture model {index}-{row_index}",
                    "block_id": f"rblk_{index:02d}_{row_index:02d}",
                    "exact_quote": row,
                    "confirmed": False,
                    "expected_claim_type": None,
                    "reason": "must not reach human bundle",
                    "rule_id": "fixture_proposal_rule",
                }
                for row_index, row in enumerate(paper_rows)
            ]
        )
    proposal_rows = [
        paper_rows[row_index]
        for row_index in range(8)
        for paper_rows in proposal_rows_by_paper
        if row_index < len(paper_rows)
    ]
    proposals = project_root / "runs" / "control-proposals.json"
    write_json(
        proposals,
        {
            "schema_version": "control-proposal-worklist/0.1",
            "status": "proposed-unconfirmed",
            "rows_needing_a_human_label": proposal_rows,
        },
    )
    packet = project_root / "runs" / "private" / "packet"
    prepare_annotation_packet(
        proposals_path=proposals,
        run_root=run_root,
        output_dir=packet,
        expected_items=53,
    )
    workspace = project_root / "runs" / "private" / "workspace"
    prepare_annotation_workspace(
        packet_dir=packet,
        run_root=run_root,
        project_root=project_root,
        output_dir=workspace,
    )
    return project_root, run_root, packet, workspace


def _response(
    item: dict,
    annotator: str,
    result_bearing: ResultBearing,
    origin: ResultOrigin | None,
    confidence: AnnotationConfidence = AnnotationConfidence.HIGH,
) -> AnnotationResponse:
    return AnnotationResponse(
        item_id=item["item_id"],
        annotator=annotator,
        result_bearing=result_bearing,
        origin=origin,
        exact_evidence=item["exact_evidence"],
        confidence=confidence,
    )


def _complete_responses(workspace: Path) -> tuple[bytes, bytes]:
    items = _read_jsonl(workspace / "annotator-a" / "input" / "items.jsonl")
    left = [
        _response(items[0], "annotator-a", ResultBearing.NO, None),
        _response(
            items[1],
            "annotator-a",
            ResultBearing.YES,
            ResultOrigin.PAPER_PRODUCED,
        ),
        _response(
            items[2],
            "annotator-a",
            ResultBearing.YES,
            ResultOrigin.PAPER_PRODUCED,
            AnnotationConfidence.LOW,
        ),
        _response(
            items[3],
            "annotator-a",
            ResultBearing.UNCERTAIN,
            ResultOrigin.UNCERTAIN,
            AnnotationConfidence.MEDIUM,
        ),
        _response(
            items[4],
            "annotator-a",
            ResultBearing.YES,
            ResultOrigin.UNCERTAIN,
            AnnotationConfidence.MEDIUM,
        ),
        _response(
            items[5],
            "annotator-a",
            ResultBearing.YES,
            ResultOrigin.EXTERNALLY_SOURCED,
        ),
        _response(
            items[6],
            "annotator-a",
            ResultBearing.YES,
            ResultOrigin.PAPER_PRODUCED,
        ),
    ]
    right = [
        _response(items[0], "annotator-b", ResultBearing.NO, None),
        _response(
            items[1],
            "annotator-b",
            ResultBearing.YES,
            ResultOrigin.EXTERNALLY_SOURCED,
        ),
        _response(
            items[2],
            "annotator-b",
            ResultBearing.YES,
            ResultOrigin.PAPER_PRODUCED,
            AnnotationConfidence.LOW,
        ),
        _response(
            items[3],
            "annotator-b",
            ResultBearing.UNCERTAIN,
            ResultOrigin.PAPER_PRODUCED,
            AnnotationConfidence.MEDIUM,
        ),
        _response(
            items[4],
            "annotator-b",
            ResultBearing.YES,
            ResultOrigin.UNCERTAIN,
            AnnotationConfidence.MEDIUM,
        ),
        _response(
            items[5],
            "annotator-b",
            ResultBearing.YES,
            ResultOrigin.EXTERNALLY_SOURCED,
            AnnotationConfidence.MEDIUM,
        ),
        _response(
            items[6],
            "annotator-b",
            ResultBearing.YES,
            ResultOrigin.PAPER_PRODUCED,
        ),
    ]
    for item in items[7:]:
        left.append(
            _response(
                item,
                "annotator-a",
                ResultBearing.YES,
                ResultOrigin.PAPER_PRODUCED,
            )
        )
        right.append(
            _response(
                item,
                "annotator-b",
                ResultBearing.YES,
                ResultOrigin.PAPER_PRODUCED,
            )
        )
    left_path = workspace / "annotator-a" / "work" / "response.jsonl"
    right_path = workspace / "annotator-b" / "work" / "response.jsonl"
    write_jsonl(left_path, [item.model_dump(mode="json") for item in left])
    write_jsonl(right_path, [item.model_dump(mode="json") for item in right])
    return left_path.read_bytes(), right_path.read_bytes()


def test_working_copies_are_isolated_private_and_leave_packet_unchanged(
    tmp_path: Path,
) -> None:
    project, run_root, packet, workspace = _workflow_fixture(tmp_path)
    before = {
        path.relative_to(packet).as_posix(): sha256_file(path)
        for path in packet.rglob("*")
        if path.is_file()
    }

    manifest = validate_annotation_workspace(
        packet_dir=packet,
        workspace_dir=workspace,
        run_root=run_root,
        project_root=project,
    )

    assert isinstance(manifest, AnnotationWorkspaceManifest)
    assert manifest.item_count == manifest.evaluation_denominator == 53
    assert manifest.practice_example.practice_id == PRACTICE_EXAMPLE_ID
    assert not manifest.practice_example.included_in_denominator
    assert manifest.privacy.contains_human_labels is False
    for bundle in manifest.bundles:
        root = workspace / bundle.directory
        files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        assert files == {
            "input/protocol.md",
            "input/items.jsonl",
            "work/response.jsonl",
            *(f"input/pdfs/paper-{index}.pdf" for index in range(7)),
        }
        assert not any(other in files for other in ("manifest.json", "adjudication.jsonl"))
        assert all(
            not (root / f"input/pdfs/paper-{index}.pdf").stat().st_mode & 0o222
            for index in range(7)
        )
        protocol = (root / "input" / "protocol.md").read_text(encoding="utf-8")
        assert PRACTICE_EXAMPLE_ID in protocol
        assert "No decision or correct answer is supplied" in protocol
    readme = (workspace / "README.md").read_text(encoding="utf-8")
    assert OPEN_DEVELOPMENT_SCOPE in readme
    assert "lock-control-annotation-responses" in readme
    assert "measure-control-annotation-agreement" in readme
    assert "prepare-control-annotation-adjudication" in readme
    assert "validate-control-annotation-adjudication" in readme
    assert before == {
        path.relative_to(packet).as_posix(): sha256_file(path)
        for path in packet.rglob("*")
        if path.is_file()
    }


def test_completed_response_validation_ignores_blank_hash_but_fails_closed(
    tmp_path: Path,
) -> None:
    project, run_root, packet, workspace = _workflow_fixture(tmp_path)
    blank_hash = sha256_file(workspace / "annotator-a" / "work" / "response.jsonl")
    _complete_responses(workspace)
    receipt = validate_workspace_response(
        packet_dir=packet,
        workspace_dir=workspace,
        annotator="annotator-a",
        run_root=run_root,
        project_root=project,
    )
    assert receipt.sha256 != blank_hash
    assert len(receipt.responses) == 53

    path = workspace / "annotator-a" / "work" / "response.jsonl"
    good = path.read_bytes()
    rows = _read_jsonl(path)
    write_jsonl(path, rows[:-1])
    with pytest.raises(ValueError, match="exactly match packet order"):
        validate_workspace_response(
            packet_dir=packet,
            workspace_dir=workspace,
            annotator="annotator-a",
            run_root=run_root,
            project_root=project,
        )
    path.write_bytes(good)
    rows = _read_jsonl(path)
    rows[0]["annotator"] = "annotator-b"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="wrong or mixed annotator"):
        validate_workspace_response(
            packet_dir=packet,
            workspace_dir=workspace,
            annotator="annotator-a",
            run_root=run_root,
            project_root=project,
        )
    path.write_bytes(good)
    rows = _read_jsonl(path)
    rows[0]["exact_evidence"] = "text absent from every frozen page"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="absent from cited frozen page"):
        validate_workspace_response(
            packet_dir=packet,
            workspace_dir=workspace,
            annotator="annotator-a",
            run_root=run_root,
            project_root=project,
        )
    path.write_bytes(good)
    rows = _read_jsonl(path)
    rows[0]["exact_evidence"] = "A quote that appears in this paper, but on the wrong page."
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="absent from cited frozen page"):
        validate_workspace_response(
            packet_dir=packet,
            workspace_dir=workspace,
            annotator="annotator-a",
            run_root=run_root,
            project_root=project,
        )

    with pytest.raises(ValueError, match="blank template"):
        validate_completed_response(
            packet_dir=packet,
            response_path=packet / "responses" / "annotator-a.jsonl",
            expected_annotator="annotator-a",
            run_root=run_root,
            project_root=project,
        )


def test_response_schema_version_and_private_output_boundary_fail_closed(
    tmp_path: Path,
) -> None:
    project, run_root, packet, workspace = _workflow_fixture(tmp_path)
    _complete_responses(workspace)
    response_path = workspace / "annotator-a" / "work" / "response.jsonl"
    rows = _read_jsonl(response_path)
    rows[0]["schema_version"] = "control-row-annotation-response/9.9"
    write_jsonl(response_path, rows)
    with pytest.raises(ValueError):
        validate_workspace_response(
            packet_dir=packet,
            workspace_dir=workspace,
            annotator="annotator-a",
            run_root=run_root,
            project_root=project,
        )

    outside = project / "public-annotation-workspace"
    with pytest.raises(ValueError, match="below runs/private"):
        prepare_annotation_workspace(
            packet_dir=packet,
            run_root=run_root,
            project_root=project,
            output_dir=outside,
        )
    assert not outside.exists()


def test_malicious_paper_id_cannot_escape_packet_output(tmp_path: Path) -> None:
    proposals = tmp_path / "proposals.json"
    write_json(
        proposals,
        {
            "rows_needing_a_human_label": [
                {
                    "paper_id": "../../outside-sentinel",
                    "page": 1,
                    "kind": "table",
                    "label": "Table 1",
                    "row": "malicious",
                    "block_id": "rblk_malicious",
                    "exact_quote": "must never be written",
                }
            ]
        },
    )
    sentinel = tmp_path / "outside-sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")
    output = tmp_path / "runs" / "private" / "packet"

    with pytest.raises(ValueError, match="path-safe paper_id"):
        prepare_annotation_packet(
            proposals_path=proposals,
            run_root=tmp_path / "runs" / "sealed",
            output_dir=output,
            expected_items=1,
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not output.exists()


def test_lock_preserves_originals_then_computes_complete_fixed_agreement(
    tmp_path: Path,
) -> None:
    project, run_root, packet, workspace = _workflow_fixture(tmp_path)
    left_bytes, right_bytes = _complete_responses(workspace)
    completion_dir = project / "runs" / "private" / "completion"

    manifest, agreement, digest = lock_completed_responses(
        packet_dir=packet,
        workspace_dir=workspace,
        run_root=run_root,
        project_root=project,
        output_dir=completion_dir,
    )

    assert isinstance(manifest, AnnotationCompletionManifest)
    assert manifest.status == "independent-responses-complete"
    assert manifest.originals_preserved_before_comparison
    assert manifest.annotators == ["annotator-a", "annotator-b"]
    assert manifest.evaluation_denominator == agreement.denominator == 53
    assert digest == sha256_file(completion_dir / "completion-manifest.json")
    assert (completion_dir / "responses" / "annotator-a.jsonl").read_bytes() == left_bytes
    assert (completion_dir / "responses" / "annotator-b.jsonl").read_bytes() == right_bytes
    assert [response.file.sha256 for response in manifest.responses] == [
        hashlib.sha256(left_bytes).hexdigest(),
        hashlib.sha256(right_bytes).hexdigest(),
    ]
    assert agreement.categories == [
        "not_result",
        "uncertain_result_bearing",
        "result/paper_produced",
        "result/externally_sourced",
        "result/uncertain",
    ]
    assert len(agreement.confusion_matrix_counts) == 5
    assert all(len(row) == 5 for row in agreement.confusion_matrix_counts)
    assert sum(sum(row) for row in agreement.confusion_matrix_counts) == 53
    assert sum(agreement.annotator_category_counts["annotator-a"].values()) == 53
    assert sum(agreement.annotator_category_counts["annotator-b"].values()) == 53
    assert agreement.cohen_kappa != "undefined"
    assert manifest.adjudication_required_count == 4
    assert (
        manifest.adjudication_reason_counts[AdjudicationReason.JOINT_DISPOSITION_DISAGREEMENT.value]
        == 1
    )
    assert (
        manifest.adjudication_reason_counts[AdjudicationReason.ANNOTATOR_A_LOW_CONFIDENCE.value]
        == 1
    )
    assert (
        manifest.adjudication_reason_counts[AdjudicationReason.ANNOTATOR_B_LOW_CONFIDENCE.value]
        == 1
    )
    assert (
        manifest.adjudication_reason_counts[
            AdjudicationReason.ANNOTATOR_A_UNCERTAIN_RESULT_BEARING.value
        ]
        == 1
    )
    assert (
        manifest.adjudication_reason_counts[
            AdjudicationReason.ANNOTATOR_B_UNCERTAIN_RESULT_BEARING.value
        ]
        == 1
    )
    assert (
        manifest.adjudication_reason_counts[AdjudicationReason.ANNOTATOR_A_UNCERTAIN_ORIGIN.value]
        == 2
    )
    assert (
        manifest.adjudication_reason_counts[AdjudicationReason.ANNOTATOR_B_UNCERTAIN_ORIGIN.value]
        == 1
    )

    completion = validate_completion_bundle(
        packet_dir=packet,
        workspace_dir=workspace,
        completion_dir=completion_dir,
        run_root=run_root,
        project_root=project,
        expected_manifest_sha256=digest,
    )
    assert [selection.item_id for selection in completion.selections] == [
        _read_jsonl(workspace / "annotator-a" / "input" / "items.jsonl")[index]["item_id"]
        for index in (1, 2, 3, 4)
    ]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        lock_completed_responses(
            packet_dir=packet,
            workspace_dir=workspace,
            run_root=run_root,
            project_root=project,
            output_dir=completion_dir,
        )


def test_locked_response_mutation_is_detected_even_if_working_copy_changes(
    tmp_path: Path,
) -> None:
    project, run_root, packet, workspace = _workflow_fixture(tmp_path)
    _complete_responses(workspace)
    completion_dir = project / "runs" / "private" / "completion"
    _, _, digest = lock_completed_responses(
        packet_dir=packet,
        workspace_dir=workspace,
        run_root=run_root,
        project_root=project,
        output_dir=completion_dir,
    )
    working_path = workspace / "annotator-a" / "work" / "response.jsonl"
    working_path.write_text("changed after lock\n", encoding="utf-8")
    validate_completion_bundle(
        packet_dir=packet,
        workspace_dir=workspace,
        completion_dir=completion_dir,
        run_root=run_root,
        project_root=project,
        expected_manifest_sha256=digest,
    )
    locked_path = completion_dir / "responses" / "annotator-a.jsonl"
    locked_path.chmod(0o600)
    locked_path.write_bytes(locked_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_completion_bundle(
            packet_dir=packet,
            workspace_dir=workspace,
            completion_dir=completion_dir,
            run_root=run_root,
            project_root=project,
            expected_manifest_sha256=digest,
        )


def test_completion_manifest_requires_the_external_lock_receipt(tmp_path: Path) -> None:
    project, run_root, packet, workspace = _workflow_fixture(tmp_path)
    _complete_responses(workspace)
    completion_dir = project / "runs" / "private" / "completion"
    _, _, digest = lock_completed_responses(
        packet_dir=packet,
        workspace_dir=workspace,
        run_root=run_root,
        project_root=project,
        output_dir=completion_dir,
    )
    manifest_path = completion_dir / "completion-manifest.json"
    payload = read_json(manifest_path)
    payload["adjudication_reason_counts"]["annotator_a_low_confidence"] += 1
    write_json(manifest_path, payload)

    with pytest.raises(ValueError, match="external lock receipt"):
        validate_completion_bundle(
            packet_dir=packet,
            workspace_dir=workspace,
            completion_dir=completion_dir,
            run_root=run_root,
            project_root=project,
            expected_manifest_sha256=digest,
        )


def test_undefined_kappa_is_literal_undefined_and_never_zero(tmp_path: Path) -> None:
    project, run_root, packet, workspace = _workflow_fixture(tmp_path)
    items = _read_jsonl(workspace / "annotator-a" / "input" / "items.jsonl")
    for annotator in ("annotator-a", "annotator-b"):
        responses = [_response(item, annotator, ResultBearing.NO, None) for item in items]
        write_jsonl(
            workspace / annotator / "work" / "response.jsonl",
            [item.model_dump(mode="json") for item in responses],
        )
    completion_dir = project / "runs" / "private" / "completion"
    _, agreement, _ = lock_completed_responses(
        packet_dir=packet,
        workspace_dir=workspace,
        run_root=run_root,
        project_root=project,
        output_dir=completion_dir,
    )
    assert agreement.cohen_kappa == "undefined"
    assert agreement.kappa_status == "undefined_no_expected_variance"
    assert agreement.cohen_kappa != 0
    assert read_json(completion_dir / "agreement.json")["cohen_kappa"] == "undefined"


def test_separate_adjudication_subset_never_overwrites_primary_responses(
    tmp_path: Path,
) -> None:
    project, run_root, packet, workspace = _workflow_fixture(tmp_path)
    _complete_responses(workspace)
    completion_dir = project / "runs" / "private" / "completion"
    _, _, completion_digest = lock_completed_responses(
        packet_dir=packet,
        workspace_dir=workspace,
        run_root=run_root,
        project_root=project,
        output_dir=completion_dir,
    )
    before = {
        annotator: sha256_file(completion_dir / "responses" / f"{annotator}.jsonl")
        for annotator in ("annotator-a", "annotator-b")
    }
    adjudication_dir = project / "runs" / "private" / "adjudication"
    manifest, digest = prepare_adjudication_workspace(
        packet_dir=packet,
        workspace_dir=workspace,
        completion_dir=completion_dir,
        run_root=run_root,
        project_root=project,
        output_dir=adjudication_dir,
        expected_completion_manifest_sha256=completion_digest,
    )
    assert manifest.adjudication_required_count == 4
    assert len(manifest.pdfs) == 4
    assert digest == sha256_file(adjudication_dir / "adjudication-manifest.json")
    assert before == {
        annotator: sha256_file(completion_dir / "responses" / f"{annotator}.jsonl")
        for annotator in ("annotator-a", "annotator-b")
    }
    blank = validate_adjudication_workspace(
        packet_dir=packet,
        workspace_dir=workspace,
        completion_dir=completion_dir,
        adjudication_dir=adjudication_dir,
        run_root=run_root,
        project_root=project,
        expected_completion_manifest_sha256=completion_digest,
        require_complete=False,
    )
    assert not blank.complete
    with pytest.raises(ValueError, match="blank decisions"):
        validate_adjudication_workspace(
            packet_dir=packet,
            workspace_dir=workspace,
            completion_dir=completion_dir,
            adjudication_dir=adjudication_dir,
            run_root=run_root,
            project_root=project,
            expected_completion_manifest_sha256=completion_digest,
        )

    tasks = _read_jsonl(adjudication_dir / "tasks.jsonl")
    records = [
        AdjudicationRecord(
            item_id=task["item"]["item_id"],
            adjudicator="adjudicator-c",
            result_bearing=ResultBearing.YES,
            origin=ResultOrigin.PAPER_PRODUCED,
            exact_evidence=task["item"]["exact_evidence"],
            rationale="Resolved against the frozen paper and both original responses.",
        )
        for task in tasks
    ]
    write_jsonl(
        adjudication_dir / "response.jsonl",
        [record.model_dump(mode="json") for record in records],
    )
    receipt = validate_adjudication_workspace(
        packet_dir=packet,
        workspace_dir=workspace,
        completion_dir=completion_dir,
        adjudication_dir=adjudication_dir,
        run_root=run_root,
        project_root=project,
        expected_completion_manifest_sha256=completion_digest,
    )
    assert receipt.complete
    assert receipt.adjudicator_id == "adjudicator-c"
    assert receipt.record_count == 4
    assert before == {
        annotator: sha256_file(completion_dir / "responses" / f"{annotator}.jsonl")
        for annotator in ("annotator-a", "annotator-b")
    }


def test_adjudication_tasks_are_recomputed_not_trusted_from_their_manifest(
    tmp_path: Path,
) -> None:
    project, run_root, packet, workspace = _workflow_fixture(tmp_path)
    _complete_responses(workspace)
    completion_dir = project / "runs" / "private" / "completion"
    _, _, completion_digest = lock_completed_responses(
        packet_dir=packet,
        workspace_dir=workspace,
        run_root=run_root,
        project_root=project,
        output_dir=completion_dir,
    )
    adjudication_dir = project / "runs" / "private" / "adjudication"
    prepare_adjudication_workspace(
        packet_dir=packet,
        workspace_dir=workspace,
        completion_dir=completion_dir,
        run_root=run_root,
        project_root=project,
        output_dir=adjudication_dir,
        expected_completion_manifest_sha256=completion_digest,
    )
    tasks_path = adjudication_dir / "tasks.jsonl"
    tasks = _read_jsonl(tasks_path)
    tasks[0]["annotator_a_response"]["confidence"] = "medium"
    changed_tasks_sha = write_jsonl(tasks_path, tasks)
    manifest_path = adjudication_dir / "adjudication-manifest.json"
    manifest = read_json(manifest_path)
    manifest["tasks"]["sha256"] = changed_tasks_sha
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="tasks differ from completion selection"):
        validate_adjudication_workspace(
            packet_dir=packet,
            workspace_dir=workspace,
            completion_dir=completion_dir,
            adjudication_dir=adjudication_dir,
            run_root=run_root,
            project_root=project,
            expected_completion_manifest_sha256=completion_digest,
            require_complete=False,
        )


def test_cli_workspace_output_is_aggregate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, run_root, packet, _ = _workflow_fixture(tmp_path)
    output = project / "runs" / "private" / "cli-workspace"
    monkeypatch.chdir(project)
    result = CliRunner().invoke(
        app,
        [
            "prepare-control-annotation-workspace",
            str(packet),
            "--run-root",
            str(run_root),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "prepared-unlabeled-working-copies"
    assert payload["item_count"] == payload["evaluation_denominator"] == 53
    assert payload["frozen_pdfs_per_annotator"] == 7
    assert "Fixture model" not in result.output
    assert "annotator-a" not in result.output
    assert str(project) not in result.output


def test_cli_lock_measure_and_adjudication_round_trip_is_aggregate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, run_root, packet, workspace = _workflow_fixture(tmp_path)
    _complete_responses(workspace)
    completion = project / "runs" / "private" / "cli-completion"
    adjudication = project / "runs" / "private" / "cli-adjudication"
    monkeypatch.chdir(project)

    locked = CliRunner().invoke(
        app,
        [
            "lock-control-annotation-responses",
            str(workspace),
            "--packet",
            str(packet),
            "--run-root",
            str(run_root),
            "--output",
            str(completion),
        ],
    )
    assert locked.exit_code == 0, locked.output
    completion_sha = json.loads(locked.output)["completion_manifest_sha256"]

    measured = CliRunner().invoke(
        app,
        [
            "measure-control-annotation-agreement",
            str(completion),
            "--workspace",
            str(workspace),
            "--packet",
            str(packet),
            "--run-root",
            str(run_root),
            "--completion-manifest-sha256",
            completion_sha,
        ],
    )
    assert measured.exit_code == 0, measured.output
    agreement = json.loads(measured.output)
    assert agreement["denominator"] == 53
    assert len(agreement["confusion_matrix"]) == 5

    prepared = CliRunner().invoke(
        app,
        [
            "prepare-control-annotation-adjudication",
            str(completion),
            "--workspace",
            str(workspace),
            "--packet",
            str(packet),
            "--run-root",
            str(run_root),
            "--completion-manifest-sha256",
            completion_sha,
            "--output",
            str(adjudication),
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    tasks = _read_jsonl(adjudication / "tasks.jsonl")
    records = [
        AdjudicationRecord(
            item_id=task["item"]["item_id"],
            adjudicator="adjudicator-c",
            result_bearing=ResultBearing.YES,
            origin=ResultOrigin.PAPER_PRODUCED,
            exact_evidence=task["item"]["exact_evidence"],
            rationale="Resolved independently against the frozen cited page.",
        )
        for task in tasks
    ]
    write_jsonl(
        adjudication / "response.jsonl",
        [record.model_dump(mode="json") for record in records],
    )
    finalized = CliRunner().invoke(
        app,
        [
            "validate-control-annotation-adjudication",
            str(adjudication),
            "--completion",
            str(completion),
            "--workspace",
            str(workspace),
            "--packet",
            str(packet),
            "--run-root",
            str(run_root),
            "--completion-manifest-sha256",
            completion_sha,
        ],
    )
    assert finalized.exit_code == 0, finalized.output
    assert json.loads(finalized.output)["records"] == 4
    combined_output = locked.output + measured.output + prepared.output + finalized.output
    assert "Fixture model" not in combined_output
    assert "annotator-a" not in combined_output
    assert "adjudicator-c" not in combined_output
    assert str(project) not in combined_output


def test_cli_validation_failure_does_not_render_private_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, run_root, packet, workspace = _workflow_fixture(tmp_path)
    items_path = workspace / "annotator-a" / "input" / "items.jsonl"
    first_item_id = _read_jsonl(items_path)[0]["item_id"]
    monkeypatch.chdir(project)

    result = CliRunner().invoke(
        app,
        [
            "validate-control-annotation-response",
            str(workspace),
            "--packet",
            str(packet),
            "--run-root",
            str(run_root),
            "--annotator",
            "annotator-a",
        ],
    )

    assert result.exit_code == 1
    assert "private-annotation-validation-failed" in result.output
    assert "Traceback" not in result.output
    assert "AnnotationResponse" not in result.output
    assert first_item_id not in result.output
    assert str(project) not in result.output
