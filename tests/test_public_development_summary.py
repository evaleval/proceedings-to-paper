from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from proceedings_to_eee.cli import app
from proceedings_to_eee.domain.attribution import AttributionState, AttributionVerdict
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ExportStatus
from proceedings_to_eee.extraction.row_enumeration import EnumerationRow, make_row_batch
from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes, write_json, write_jsonl
from proceedings_to_eee.providers.openrouter import ProviderCall
from proceedings_to_eee.reporting.public_development_summary import (
    PublicDevelopmentSummaryError,
    build_public_development_summary,
    write_public_development_summary,
)
from proceedings_to_eee.resources import EEE_SCHEMA_SHA256

RUNNER = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _row(
    row_id: str,
    raw_text: str,
    *,
    line: int,
) -> dict[str, object]:
    return {
        "schema_version": "table-row-enumeration/0.1",
        "row_id": row_id,
        "input_sha256": row_id[-1] * 64,
        "source_id": "synthetic-source",
        "page": 1,
        "region_id": "region-1",
        "table_label": "Synthetic Table 1",
        "headers": [],
        "row_label": raw_text.split()[0],
        "raw_text": raw_text,
        "raw_cells": [{"raw": raw_text, "span": {"start_line": line, "end_line": line}}],
        "values": [{"raw": "0.5", "span": {"start_line": line, "end_line": line}}],
        "span": {"start_line": line, "end_line": line},
    }


def _row_plan() -> dict[str, object]:
    first = _row("row-1", "System-A 0.5", line=2)
    second = _row("row-2", "Header-like 0.5", line=3)
    third = _row("row-3", f"Oversized {'x' * 4_100} 0.5", line=4)
    config = {
        "min_dense_table_rows": 2,
        "max_rows_per_batch": 4,
        "max_value_tokens_per_batch": 24,
        "max_characters_per_batch": 4000,
        "max_recovery_depth": 1,
    }
    telemetry = {
        "tables_considered": 1,
        "dense_tables": 1,
        "rows_planned": 3,
        "unbatchable_rows": 1,
        "base_batches": 1,
        "expected_calls": 1,
        "maximum_calls": 3,
    }
    batch = make_row_batch(
        [EnumerationRow.model_validate(first), EnumerationRow.model_validate(second)]
    ).model_dump(mode="json")
    singleton = make_row_batch([EnumerationRow.model_validate(third)])
    return {
        "schema_version": "table-row-enumeration/0.1",
        "config": config,
        "rows": [first, second, third],
        "batches": [batch],
        "unbatchable_rows": [
            {
                "row_id": "row-3",
                "input_sha256": "3" * 64,
                "source_id": "synthetic-source",
                "page": 1,
                "region_id": "region-1",
                "character_count": singleton.character_count,
                "value_token_count": singleton.value_token_count,
                "max_characters_per_batch": 4000,
                "max_value_tokens_per_batch": 24,
                "reasons": ["max_characters_per_batch"],
            }
        ],
        "telemetry": telemetry,
    }


def _call(*, row: bool = False) -> ProviderCall:
    return ProviderCall(
        model_requested="fixture/model",
        model_returned="fixture/model",
        provider_returned="fixture-provider",
        prompt_sha256=("b" if row else "a") * 64,
        response_sha256=("d" if row else "c") * 64,
        temperature=0.0,
        reasoning_effort="minimal",
        max_tokens=16_000,
        seed=7,
        schema_name=("paper_table_row_dispositions" if row else "paper_evaluation_candidates"),
        schema_sha256=("f" if row else "e") * 64,
        latency_seconds=0.25 if row else 0.5,
        input_tokens=80 if row else 100,
        output_tokens=20 if row else 40,
        total_tokens=100 if row else 140,
        cost_usd=0.002 if row else 0.004,
        request_id="private-provider-request-id",
        attempts=2 if row else 1,
    )


def _run_root(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> Path:
    root = tmp_path / "development-run"
    paper = root / "fixture-paper"
    paper.mkdir(parents=True)

    exported = eligible_candidate.model_copy(deep=True)
    exported.paper_id = "fixture-paper"
    exported.export_status = ExportStatus.EXPORTED
    exported.attribution = AttributionVerdict(
        state=AttributionState.PAPER_PRODUCED,
        rule_id="explicit_test_fixture",
        lexicon_id="attribution-cues-v0",
        lexicon_sha256="9" * 64,
    )
    exported.observation_id = exported.stable_id()

    review_payload = exported.model_dump(mode="json")
    review_payload["value"]["numeric"] = 0.5
    review_payload["value"]["raw"] = "private source quotation 0.5"
    review_payload["evidence"][0]["quote"] = "private source quotation 0.5"
    review_payload["evidence"][0]["quote_sha256"] = None
    review_payload["export_status"] = ExportStatus.NEEDS_REVIEW
    review_payload["attribution"] = AttributionVerdict(
        state=AttributionState.UNRESOLVED,
        rule_id="fixture_unresolved",
        lexicon_id="attribution-cues-v0",
        lexicon_sha256="9" * 64,
    ).model_dump(mode="json")
    review_payload["notes"] = ["semantic safety: synthetic duplicate ambiguity"]
    review_payload["observation_id"] = None
    review = CandidateObservation.model_validate(review_payload)
    write_jsonl(paper / "observations.jsonl", [exported, review])

    request_contract = {
        "schema_version": "provider-request-contract/0.1",
        "privacy": {"data_collection": "deny", "zdr": True},
        "schema": {"schema_strict": True},
    }
    legacy_call = _call()
    row_call = _call(row=True)
    row_plan = _row_plan()
    private_root = paper / "private"
    private_root.mkdir()
    row_plan_sha256 = write_json(private_root / "row-enumeration-plan.json", row_plan)
    row_plan_summary = row_plan["telemetry"]
    row_outcome = {
        "rows_resolved": 2,
        "rows_unresolved": 0,
        "rows_unbatchable": 1,
        "dispositions": {"result": 1, "not_result": 1, "uncertain": 0},
        "calls": 1,
        "base_calls": 1,
        "recovery_calls": 0,
        "attempts": 1,
        "input_tokens": 80,
        "output_tokens": 20,
        "total_tokens": 100,
        "cost_usd": 0.002,
    }
    row_calls = [row_call.model_dump(mode="json", exclude_none=True)]
    row_attempts = [
        {
            "batch_id": row_plan["batches"][0]["batch_id"],
            "depth": 0,
            "row_ids": ["row-1", "row-2"],
            "status": "success",
            "resolved_row_ids": ["row-1", "row-2"],
            "unresolved_row_ids": [],
            "unknown_row_ids": [],
            "completed_provider_call": True,
        }
    ]
    write_json(
        private_root / "row-enumeration.json",
        {
            "schema_version": "row-enumeration-outcome/0.1",
            "plan_sha256": row_plan_sha256,
            "records": {
                "row-1": {
                    "row_id": "row-1",
                    "disposition": "result",
                    "candidates": [exported.model_dump(mode="json", exclude_none=True)],
                    "note": None,
                },
                "row-2": {
                    "row_id": "row-2",
                    "disposition": "not_result",
                    "candidates": [],
                    "note": None,
                },
            },
            "calls": row_calls,
            "attempts": row_attempts,
            "unresolved_row_ids": [],
            "unbatchable_row_ids": ["row-3"],
            "unknown_row_ids": [],
            "invalid_row_reasons": {},
            "warnings": [],
            "telemetry": row_outcome,
        },
    )
    counts = {
        "candidates": 2,
        "candidates_before_deduplication": 3,
        "duplicates_removed": 1,
        "candidates_needing_review": 1,
        "semantic_safety_reviews": 1,
        "exported": 1,
        "eee_records": 1,
        "eee_schema_issues": 0,
    }
    run = {
        "schema_version": "pipeline-run/0.2",
        "status": "partial_failure",
        "paper_id": "fixture-paper",
        "review_state": {
            "status": "needs_review",
            "reasons": [
                "row_enumeration_unbatchable",
                "candidate_review_required",
            ],
        },
        "extractor": {
            "provider": "openrouter",
            "model": "fixture/model",
            "temperature": 0.0,
            "reasoning_effort": "minimal",
            "max_tokens": 16_000,
            "seed": 7,
            "prompt_sha256": "a" * 64,
            "request_contract": request_contract,
            "calls": [legacy_call.model_dump(mode="json", exclude_none=True)],
            "execution": {
                "blocks_total": 1,
                "blocks_succeeded": 1,
                "blocks_failed": 0,
                "blocks_resumed": 0,
            },
            "block_attempts": [
                {
                    "status": "success",
                    "completed_provider_call": True,
                }
            ],
        },
        "row_enumeration": {
            "enabled": True,
            "provider": "openrouter",
            "model": "fixture/model",
            "temperature": 0.0,
            "reasoning_effort": "minimal",
            "max_tokens": 16_000,
            "seed": 7,
            "prompt_sha256": "b" * 64,
            "request_contract": request_contract,
            "limits": {
                "min_dense_table_rows": 2,
                "max_rows_per_batch": 4,
                "max_value_tokens_per_batch": 24,
                "max_characters_per_batch": 4000,
                "max_recovery_depth": 1,
            },
            "plan_sha256": row_plan_sha256,
            "plan": row_plan_summary,
            "outcome": row_outcome,
            "execution": {
                "batches_total": 1,
                "batches_resumed": 0,
                "batches_executed": 1,
                "invalid_rows_seen": 0,
                "unknown_row_ids_seen": 0,
            },
            "calls": row_calls,
            "attempts": row_attempts,
        },
        "verifier": {"enabled": False, "calls": []},
        "code": {
            "git_commit": "1" * 40,
            "git_dirty": True,
            "git_available": True,
            "source_tree_sha256": "2" * 64,
        },
        "eee_schema": {"version": "0.2.2", "sha256": EEE_SCHEMA_SHA256},
        "counts": counts,
    }
    write_json(paper / "run.json", run)
    eee_root = paper / "eee"
    eee_root.mkdir()
    eee_record = json.loads(
        (PROJECT_ROOT / "examples" / "quickstart" / "synthetic-eee.json").read_text(
            encoding="utf-8"
        )
    )
    eee_result = eee_record["evaluation_results"][0]
    eee_result["evaluation_result_id"] = exported.observation_id
    eee_result["score_details"]["details"]["candidate_observation_id"] = exported.observation_id
    write_json(eee_root / "fixture.json", eee_record)
    write_json(
        root / "corpus-run.json",
        {
            "schema_version": "corpus-run/0.2",
            "corpus_id": "fixture-development",
            "corpus_binding": {
                "schema_version": "pilot-corpus/0.2",
                "corpus_id": "fixture-development",
                "evaluation_split": "development",
                "corpus_spec_sha256": "4" * 64,
                "paper_ids_sha256": sha256_bytes(canonical_json_bytes(["fixture-paper"])),
            },
            "status": "error",
            "generated_at": "2026-09-01T12:00:00+00:00",
            "papers": 1,
            "papers_succeeded": 0,
            "papers_failed": 1,
            "papers_needing_review": 1,
            "totals": counts,
            "runs": [run],
            "operations": {"wall_clock_seconds": 2.0},
            "reference_evaluation": {
                "bases": {"reference_observations": 4},
                "detection": {
                    "true_positives": 3,
                    "false_negatives": 1,
                    "recall": 0.75,
                    "macro_recall": 0.7,
                    "precision": 0.5,
                },
                "coverage_statement": "private source quotation must never be copied",
            },
        },
    )
    return root


def test_public_development_summary_is_aggregate_bound_and_private_free(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)

    summary = build_public_development_summary(run_root)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["scope"]["holdout_included"] is False
    assert summary["row_enumeration"] == {
        "tables_considered": 1,
        "dense_tables": 1,
        "rows_planned": 3,
        "rows_resolved": 2,
        "rows_unresolved": 0,
        "rows_unbatchable": 1,
        "unknown_row_ids_seen": 0,
        "invalid_rows_seen": 0,
        "dispositions": {"not_result": 1, "result": 1, "uncertain": 0},
        "all_rows_accounted_for": True,
        "all_planned_rows_partitioned": True,
        "all_batchable_rows_resolved": True,
        "complete_extraction": False,
        "no_unknown_or_invalid_rows_seen": True,
    }
    assert summary["outputs"]["duplicates_removed"] == 1
    assert summary["outputs"]["candidate_proposal_removal_rate"] == pytest.approx(1 / 3)
    assert summary["numeric_export_provenance"]["all_complete"] is True
    assert summary["provider_usage_recorded"]["recorded_structured_invocations"] == 2
    assert summary["provider_usage_recorded"]["cost_usd_lower_bound"] == 0.006
    assert summary["provider_usage_recorded"]["retries_lower_bound"] == 1
    assert summary["reference_evaluation"]["micro_recall"] == 0.75
    assert summary["reference_evaluation"]["precision"] == {
        "status": "not_measured",
        "computed_slice_diagnostic_omitted": True,
        "reason": (
            "The available annotation frame does not establish current whole-pipeline "
            "precision or non-result-row specificity."
        ),
    }
    assert "private source quotation" not in serialized
    assert "private-provider-request-id" not in serialized
    assert str(tmp_path) not in serialized


def test_public_development_summary_refuses_non_positive_origin_export(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)
    path = run_root / "fixture-paper" / "observations.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["attribution"]["state"] = "unresolved"
    write_jsonl(path, rows)

    with pytest.raises(PublicDevelopmentSummaryError, match="positive paper-produced origin"):
        build_public_development_summary(run_root)


def test_public_development_summary_refuses_stale_adjacent_run(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)
    adjacent = run_root / "fixture-paper" / "run.json"
    payload = json.loads(adjacent.read_text(encoding="utf-8"))
    payload["counts"]["candidates"] = 99
    write_json(adjacent, payload)

    with pytest.raises(PublicDevelopmentSummaryError, match="does not match"):
        build_public_development_summary(run_root)


def test_public_development_summary_refuses_missing_attribution(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)
    observations = run_root / "fixture-paper" / "observations.jsonl"
    rows = [json.loads(line) for line in observations.read_text(encoding="utf-8").splitlines()]
    rows[1]["attribution"] = None
    write_jsonl(observations, rows)

    with pytest.raises(PublicDevelopmentSummaryError, match="require deterministic attribution"):
        build_public_development_summary(run_root)


def test_public_development_summary_refuses_unbound_holdout_split(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)
    corpus_path = run_root / "corpus-run.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["corpus_binding"]["evaluation_split"] = "holdout"
    write_json(corpus_path, corpus)

    with pytest.raises(PublicDevelopmentSummaryError, match="development corpus binding"):
        build_public_development_summary(run_root)


def test_public_development_summary_refuses_incomplete_private_row_ledger(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)
    outcome_path = run_root / "fixture-paper" / "private" / "row-enumeration.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["records"] = {}
    write_json(outcome_path, outcome)

    with pytest.raises(PublicDevelopmentSummaryError, match="partition"):
        build_public_development_summary(run_root)


def test_public_development_summary_refuses_manifest_plan_limit_mismatch(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)
    adjacent_path = run_root / "fixture-paper" / "run.json"
    adjacent = json.loads(adjacent_path.read_text(encoding="utf-8"))
    adjacent["row_enumeration"]["limits"]["max_characters_per_batch"] = 8_000
    write_json(adjacent_path, adjacent)
    corpus_path = run_root / "corpus-run.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["runs"][0] = adjacent
    write_json(corpus_path, corpus)

    with pytest.raises(PublicDevelopmentSummaryError, match="limits do not match"):
        build_public_development_summary(run_root)


def test_public_development_summary_refuses_batch_row_that_differs_from_plan_row(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)
    plan_path = run_root / "fixture-paper" / "private" / "row-enumeration-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    changed = dict(plan["batches"][0]["rows"][0])
    changed["raw_text"] = "Tampered-System 0.5"
    changed["raw_cells"] = [
        {
            "raw": "Tampered-System 0.5",
            "span": changed["raw_cells"][0]["span"],
        }
    ]
    batch_rows = [
        EnumerationRow.model_validate(changed),
        EnumerationRow.model_validate(plan["batches"][0]["rows"][1]),
    ]
    plan["batches"][0] = make_row_batch(batch_rows).model_dump(mode="json")
    new_plan_sha256 = write_json(plan_path, plan)

    outcome_path = run_root / "fixture-paper" / "private" / "row-enumeration.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["plan_sha256"] = new_plan_sha256
    write_json(outcome_path, outcome)

    adjacent_path = run_root / "fixture-paper" / "run.json"
    adjacent = json.loads(adjacent_path.read_text(encoding="utf-8"))
    adjacent["row_enumeration"]["plan_sha256"] = new_plan_sha256
    write_json(adjacent_path, adjacent)
    corpus_path = run_root / "corpus-run.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["runs"][0] = adjacent
    write_json(corpus_path, corpus)

    with pytest.raises(PublicDevelopmentSummaryError, match="batch does not match"):
        build_public_development_summary(run_root)


def test_cli_failure_does_not_render_private_locals(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)
    observations = run_root / "fixture-paper" / "observations.jsonl"
    rows = [json.loads(line) for line in observations.read_text(encoding="utf-8").splitlines()]
    rows[1]["attribution"] = None
    write_jsonl(observations, rows)

    result = RUNNER.invoke(
        app,
        [
            "build-public-development-summary",
            str(run_root),
            "--output",
            str(tmp_path / "public" / "summary.json"),
        ],
    )

    assert result.exit_code == 1
    assert "public-development-summary-not-written" in result.output
    assert "private source quotation" not in result.output
    assert str(tmp_path) not in result.output


def test_public_development_summary_writes_only_outside_private_run(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)
    output = tmp_path / "public" / "summary.json"

    digest = write_public_development_summary(run_root, output)

    assert len(digest) == 64
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == (
        "public-development-summary/0.1"
    )
    with pytest.raises(PublicDevelopmentSummaryError, match="outside the run root"):
        write_public_development_summary(run_root, run_root / "summary.json")
