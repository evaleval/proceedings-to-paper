from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_origin_pipeline import _origin_fixture, _OriginPipelineClient
from typer.testing import CliRunner

from proceedings_to_eee.cli import app
from proceedings_to_eee.corpus import CorpusSpec
from proceedings_to_eee.domain.attribution import AttributionState, AttributionVerdict
from proceedings_to_eee.domain.observation import CandidateObservation, EvidenceAnchor
from proceedings_to_eee.domain.status import EvidenceKind, ExportStatus
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.result_blocks import segment_page_result_blocks
from proceedings_to_eee.extraction.row_enumeration import (
    RowDisposition,
    RowDispositionRecord,
    RowEnumerationConfig,
    RowEnumerationPlan,
    RowTerminalCounts,
    RowTerminalLedger,
    RowTerminalRecord,
    RowTerminalState,
    build_row_enumeration_plan,
)
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    write_json,
    write_jsonl,
)
from proceedings_to_eee.pipeline import run_corpus
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderResponseValidationError,
)
from proceedings_to_eee.reporting.public_development_summary import (
    PublicDevelopmentSummaryError,
    build_public_development_summary,
    write_public_development_summary,
)
from proceedings_to_eee.resources import EEE_SCHEMA_SHA256
from proceedings_to_eee.reviewed_export.models import (
    DecisionAuthority,
    DecisionStatus,
    OriginDecision,
    ReviewAuthorityMode,
    ReviewedExportDecision,
    ReviewEvidenceAnchor,
    ReviewFieldAttestation,
    ReviewManifest,
    TupleDecision,
    model_payload,
)
from proceedings_to_eee.reviewed_export.workflow import (
    REVIEW_DECISIONS_NAME,
    REVIEW_MANIFEST_NAME,
    ReviewedExportError,
    compose_reviewed_eee,
    prepare_export_review,
    validate_export_review,
)
from proceedings_to_eee.run_seal import seal_run_tree
from proceedings_to_eee.sources.manifest import SourceManifest


class _FiveStageRowClient(_OriginPipelineClient):
    """Return a strict row wire payload while retaining the other local fixtures."""

    def structured_chat(self, **kwargs: Any):
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] != "paper_table_row_dispositions":
            return response
        row_ids = re.findall(r'"row_id": "(trow_[0-9a-f]+)"', kwargs["user"])
        assert len(row_ids) == 2
        observation = response.payload["observations"][0]
        observation = {
            **observation,
            # A row-stage proposal may quote only the selected physical row; its
            # verified row/header bindings provide the surrounding table context.
            "evidence": [
                {
                    **observation["evidence"][0],
                    "quote": "System A                 0.80",
                }
            ],
        }
        payload = {
            "dispositions": [
                {
                    "row_id": row_ids[0],
                    "disposition": "result",
                    "observations": [observation],
                    "note": None,
                },
                {
                    "row_id": row_ids[1],
                    "disposition": "not_result",
                    "observations": [],
                    "note": "local fixture row",
                },
            ],
            "warnings": [],
        }
        return replace(
            response,
            payload=payload,
            call=response.call.model_copy(
                update={
                    "response_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True).encode()
                    ).hexdigest()
                }
            ),
        )


class _FiveStageResponseFailureRowClient(_OriginPipelineClient):
    """Exercise paid row response failure followed by deterministic recovery."""

    def structured_chat(self, **kwargs: Any):
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] != "paper_table_row_dispositions":
            return response
        row_ids = re.findall(r'"row_id": "(trow_[0-9a-f]+)"', kwargs["user"])
        assert row_ids
        payload: dict[str, Any] = (
            {"dispositions": "invalid", "warnings": []}
            if len(row_ids) > 1
            else {
                "dispositions": [
                    {
                        "row_id": row_ids[0],
                        "disposition": "not_result",
                        "observations": [],
                        "note": "singleton recovery",
                    }
                ],
                "warnings": [],
            }
        )
        return replace(
            response,
            payload=payload,
            call=response.call.model_copy(
                update={
                    "response_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True).encode()
                    ).hexdigest()
                }
            ),
        )


class _TerminalExtractorFailureClient(_FiveStageRowClient):
    """Fail every extractor wire response while keeping downstream fixtures local."""

    def structured_chat(self, **kwargs: Any):
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] == "paper_evaluation_candidates":
            raise ProviderResponseValidationError(call=response.call, code="invalid_json")
        return response


class _RequestIdFiveStageRowClient(_FiveStageRowClient):
    """Attach private transport IDs to every completed fixture call."""

    def structured_chat(self, **kwargs: Any):
        response = super().structured_chat(**kwargs)
        return replace(
            response,
            call=response.call.model_copy(
                update={"request_id": f"private-{kwargs['schema_name']}-request-id"}
            ),
        )


RUNNER = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _five_stage_run_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    verifier_enabled: bool = True,
    sealed: bool = True,
    min_confidence: float = 0.8,
    client_mode: str = "positive",
    corpus_id: str = "five-stage-development",
    client: Any | None = None,
    resume_from_checkpoints: bool = False,
) -> Path:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    settings = replace(
        settings,
        row_enumeration_enabled=True,
        min_confidence=min_confidence,
        verifier_model=settings.verifier_model if verifier_enabled else None,
        origin_model=settings.origin_model if verifier_enabled else None,
    )
    corpus = CorpusSpec(
        corpus_id=corpus_id,
        evaluation_split="development",
        description="Provider-free five-stage reporting fixture.",
        papers=[spec],
    )
    selected_client = client or _FiveStageRowClient(client_mode)
    run_corpus(
        corpus=corpus,
        settings=settings,
        client=selected_client,
    )
    if resume_from_checkpoints:
        run_corpus(corpus=corpus, settings=settings, client=selected_client)
    if not sealed:
        return settings.output_root
    sealed_root = tmp_path / ("sealed-five-stage" if verifier_enabled else "sealed-tuple-review")
    seal_run_tree(settings.output_root, sealed_root)
    return sealed_root


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_mapping_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_mapping_keys(item)}
    return set()


def _complete_current_review(review_root: Path) -> None:
    manifest = ReviewManifest.model_validate(read_json(review_root / REVIEW_MANIFEST_NAME))
    paper = manifest.papers[0]
    source_manifest = SourceManifest.model_validate(
        read_json(review_root / paper.source_manifest.review_copy.path)
    )
    layout = PdfLayout.model_validate(read_json(review_root / paper.layout.review_copy.path))
    source = next(item for item in source_manifest.sources if item.source_id == layout.source_id)
    excerpt = "We evaluate System A on Dataset A using AUC."
    page = layout.pages[0]
    start = page.text.index(excerpt)
    origin_anchor = ReviewEvidenceAnchor(
        source_id=layout.source_id,
        source_sha256=str(source.sha256),
        source_manifest_sha256=paper.source_manifest.sha256,
        layout_sha256=paper.layout.sha256,
        parser=layout.parser,
        parser_version=layout.parser_version,
        page=page.page,
        page_text_sha256=page.text_sha256,
        exact_excerpt=excerpt,
        excerpt_sha256=sha256_bytes(excerpt.encode()),
        char_start=start,
        char_end=start + len(excerpt),
        start_line=page.text.count("\n", 0, start) + 1,
        end_line=page.text.count("\n", 0, start + len(excerpt) - 1) + 1,
        kind=EvidenceKind.PROSE,
    )
    result_page = layout.pages[1]
    decisions = [
        ReviewedExportDecision.model_validate(json.loads(line))
        for line in (review_root / REVIEW_DECISIONS_NAME).read_text().splitlines()
    ]
    completed: list[ReviewedExportDecision] = []
    for decision in decisions:
        payload = model_payload(decision)
        operationalization = decision.reviewed_tuple.operationalization
        result_excerpt = (
            "System A split test Dataset A AUC proportion 0.80"
            if operationalization is None
            else (f"System A Dataset A AUC proportion 0.80 operationalization {operationalization}")
        )
        result_start = result_page.text.index(result_excerpt)
        result_anchor = ReviewEvidenceAnchor(
            source_id=layout.source_id,
            source_sha256=str(source.sha256),
            source_manifest_sha256=paper.source_manifest.sha256,
            layout_sha256=paper.layout.sha256,
            parser=layout.parser,
            parser_version=layout.parser_version,
            page=result_page.page,
            page_text_sha256=result_page.text_sha256,
            exact_excerpt=result_excerpt,
            excerpt_sha256=sha256_bytes(result_excerpt.encode()),
            char_start=result_start,
            char_end=result_start + len(result_excerpt),
            start_line=result_page.text.count("\n", 0, result_start) + 1,
            end_line=result_page.text.count("\n", 0, result_start + len(result_excerpt) - 1) + 1,
            kind=EvidenceKind.TABLE,
        )
        value_hashes = decision.reviewed_tuple.field_value_sha256s()
        assert result_anchor.span_id is not None
        span_ids = [result_anchor.span_id]
        payload.update(
            {
                "status": DecisionStatus.COMPLETED.value,
                "tuple_decision": TupleDecision.CONFIRMED.value,
                "origin_decision": OriginDecision.PAPER_PRODUCED.value,
                "result_evidence": [model_payload(result_anchor)],
                "field_attestations": [
                    model_payload(
                        ReviewFieldAttestation(
                            field=field,
                            value_sha256=value_hashes[field],
                            span_ids=span_ids,
                        )
                    )
                    for field in sorted(
                        decision.reviewed_tuple.populated_fields(), key=lambda item: item.value
                    )
                ],
                "origin_evidence": [model_payload(origin_anchor)],
                "authority": model_payload(
                    DecisionAuthority(
                        mode=ReviewAuthorityMode.SINGLE_EXPERT,
                        reviewer_ids=["private-reviewer-a"],
                        protocol_sha256=manifest.protocol.sha256,
                    )
                ),
                "decided_at": "2026-09-02T12:00:00Z",
            }
        )
        completed.append(ReviewedExportDecision.model_validate(payload))
    write_jsonl(
        review_root / REVIEW_DECISIONS_NAME,
        [model_payload(decision) for decision in completed],
    )
    validate_export_review(review_root)


def _row_plan() -> dict[str, object]:
    text = (
        "Table 1. Synthetic values.\n"
        "System          Metric\n"
        "System-A        0.5\n"
        "Header-like     0.5\n"
        f"Oversized-{'x' * 300}  0.5\n"
    )
    page = PageFragment(
        fragment_id="frag_synthetic-source_0001",
        source_id="synthetic-source",
        page=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_count=len(text),
        numeric_token_count=4,
        result_signal_score=5.0,
    )
    layout = PdfLayout(
        source_id="synthetic-source",
        parser="fixture",
        parser_version="fixture",
        page_count=1,
        pages=[page],
    )
    return build_row_enumeration_plan(
        layout,
        segment_page_result_blocks(page),
        RowEnumerationConfig(
            min_dense_table_rows=2,
            max_rows_per_batch=4,
            max_value_tokens_per_batch=24,
            max_characters_per_batch=128,
            max_recovery_depth=1,
        ),
    ).model_dump(mode="json")


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
        completion_token_parameter="max_tokens",
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

    row_plan = RowEnumerationPlan.model_validate(_row_plan())
    first_row, second_row, unsupported_row = row_plan.rows
    first_value = first_row.values[0]
    exported_payload = eligible_candidate.model_dump(mode="json")
    exported_payload.update(
        {
            "paper_id": "fixture-paper",
            "value": {
                **exported_payload["value"],
                "numeric": 0.5,
                "raw": first_value.raw,
            },
            "evidence": [
                EvidenceAnchor(
                    source_id=first_row.source_id,
                    page=first_row.page,
                    kind=EvidenceKind.TABLE,
                    label=first_row.table_label,
                    row=first_row.row_label,
                    region_id=first_row.region_id,
                    planned_row_id=first_row.row_id,
                    cell_id=first_value.cell_id,
                    numeric_token_id=first_value.numeric_token_id,
                    header_ids=[
                        header.header_id
                        for binding in first_value.header_path
                        for header in binding.headers
                    ],
                    quote=first_row.raw_text,
                ).model_dump(mode="json")
            ],
            "export_status": ExportStatus.EXPORTED,
            "extraction_method": "openrouter:fixture/model:row-enumeration",
            "raw_payload_hash": "d" * 64,
            "attribution": AttributionVerdict(
                state=AttributionState.PAPER_PRODUCED,
                rule_id="explicit_test_fixture",
                lexicon_id="attribution-cues-v0",
                lexicon_sha256="9" * 64,
            ).model_dump(mode="json"),
            "observation_id": None,
        }
    )
    exported = CandidateObservation.model_validate(exported_payload)
    exported.attribution = AttributionVerdict(
        state=AttributionState.PAPER_PRODUCED,
        rule_id="explicit_test_fixture",
        lexicon_id="attribution-cues-v0",
        lexicon_sha256="9" * 64,
    )

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
    private_root = paper / "private"
    private_root.mkdir()
    row_plan_sha256 = write_json(private_root / "row-enumeration-plan.json", row_plan)
    row_plan_summary = row_plan.telemetry.model_dump(mode="json")
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
            "batch_id": row_plan.batches[0].batch_id,
            "depth": 0,
            "row_ids": [first_row.row_id, second_row.row_id],
            "status": "success",
            "resolved_row_ids": [first_row.row_id, second_row.row_id],
            "unresolved_row_ids": [],
            "unknown_row_ids": [],
            "completed_provider_call": True,
        }
    ]
    terminal_ledger = RowTerminalLedger(
        paper_id="fixture-paper",
        plan_sha256=row_plan_sha256,
        terminal_by_row={
            first_row.row_id: RowTerminalRecord(
                row_id=first_row.row_id,
                state=RowTerminalState.RESULT,
                disposition=RowDispositionRecord(
                    row_id=first_row.row_id,
                    disposition=RowDisposition.RESULT,
                    candidates=[exported],
                ),
            ),
            second_row.row_id: RowTerminalRecord(
                row_id=second_row.row_id,
                state=RowTerminalState.NOT_RESULT,
                disposition=RowDispositionRecord(
                    row_id=second_row.row_id,
                    disposition=RowDisposition.NOT_RESULT,
                    candidates=[],
                ),
            ),
            unsupported_row.row_id: RowTerminalRecord(
                row_id=unsupported_row.row_id,
                state=RowTerminalState.UNSUPPORTED,
                unsupported_reasons=row_plan.unbatchable_rows[0].reasons,
            ),
        },
        counts=RowTerminalCounts(
            planned=3,
            result=1,
            not_result=1,
            uncertain=0,
            unresolved=0,
            unsupported=1,
        ),
    )
    terminal_sha256 = write_json(
        private_root / "row-terminal-states.json",
        terminal_ledger,
    )
    write_json(
        private_root / "row-enumeration.json",
        {
            "schema_version": "row-enumeration-outcome/0.3",
            "plan_sha256": row_plan_sha256,
            "terminal_states_sha256": terminal_sha256,
            "calls": row_calls,
            "attempts": row_attempts,
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
                "max_characters_per_batch": 128,
                "max_recovery_depth": 1,
            },
            "plan_sha256": row_plan_sha256,
            "plan": row_plan_summary,
            "outcome": row_outcome,
            "terminal_states": {
                "schema_version": terminal_ledger.schema_version,
                "artifact_sha256": terminal_sha256,
                "path": "private/row-terminal-states.json",
                "counts": terminal_ledger.counts.model_dump(mode="json"),
            },
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
    terminal_path = run_root / "fixture-paper" / "private" / "row-terminal-states.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    omitted_row_id = next(
        row_id
        for row_id, record in terminal["terminal_by_row"].items()
        if record["state"] == "not_result"
    )
    terminal["terminal_by_row"].pop(omitted_row_id)
    terminal["counts"]["planned"] -= 1
    terminal["counts"]["not_result"] -= 1
    terminal_sha256 = write_json(terminal_path, terminal)

    outcome_path = run_root / "fixture-paper" / "private" / "row-enumeration.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["terminal_states_sha256"] = terminal_sha256
    write_json(outcome_path, outcome)

    adjacent_path = run_root / "fixture-paper" / "run.json"
    adjacent = json.loads(adjacent_path.read_text(encoding="utf-8"))
    adjacent["row_enumeration"]["terminal_states"]["artifact_sha256"] = terminal_sha256
    adjacent["row_enumeration"]["terminal_states"]["counts"] = terminal["counts"]
    write_json(adjacent_path, adjacent)
    corpus_path = run_root / "corpus-run.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["runs"][0] = adjacent
    write_json(corpus_path, corpus)

    with pytest.raises(PublicDevelopmentSummaryError, match="terminal ledger is invalid"):
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


def test_public_development_summary_refuses_tampered_physical_plan_row(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)
    plan_path = run_root / "fixture-paper" / "private" / "row-enumeration-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["rows"][0]["raw_text"] = "Tampered-System 0.5"
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

    with pytest.raises(PublicDevelopmentSummaryError, match="private row plan is invalid"):
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
        "public-development-summary/0.3"
    )
    historical = json.loads(output.read_text(encoding="utf-8"))
    assert historical["projection_mode"] == "historical_pipeline_0.2"
    assert historical["current_sealed_receipt"] is False
    with pytest.raises(PublicDevelopmentSummaryError, match="outside the run root"):
        write_public_development_summary(run_root, run_root / "summary.json")


def test_historical_projection_rejects_arbitrary_model_text(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root = _run_root(tmp_path, eligible_candidate)
    run_path = run_root / "fixture-paper" / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["extractor"]["model"] = "private source quotation"
    write_json(run_path, run)
    corpus_path = run_root / "corpus-run.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["runs"][0] = run
    write_json(corpus_path, corpus)

    with pytest.raises(PublicDevelopmentSummaryError, match="public-safe model ID"):
        build_public_development_summary(run_root)


def test_current_summary_validates_all_five_stages_without_private_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = _five_stage_run_root(monkeypatch, tmp_path)

    summary = build_public_development_summary(run_root)
    serialized = json.dumps(summary, sort_keys=True)
    stage_chain = summary["run_binding"]["stage_chain"]

    assert summary["schema_version"] == "public-development-summary/0.3"
    assert summary["projection_mode"] == "sealed_current_five_stage_receipt"
    assert summary["current_sealed_receipt"] is True
    assert summary["scope"]["independent_human_validation"] is False
    assert "corpus_spec_sha256" not in summary["run_binding"]
    assert len(summary["run_binding"]["recorded_corpus_spec_sha256"]) == 64
    assert summary["run_binding"]["corpus_spec_binding_status"] == ("recorded_hash_not_rederived")
    assert stage_chain["mode"] == "current_five_stage_contract"
    assert stage_chain["five_stage_gate_status"] == "validated"
    assert stage_chain["missing_gate_treated_as_passed"] is False
    assert set(stage_chain["checkpoint_validation"]) == {
        "extractor",
        "row_disposition",
        "tuple_resolution",
        "independent_verification",
        "origin_retrieval",
    }
    assert all(
        item["status"] == "validated" for item in stage_chain["checkpoint_validation"].values()
    )
    assert summary["export_provenance_modes"]["source_run"] == {
        "status": "measured",
        "eee_records": 0,
        "eee_observations": 0,
        "provenance_mode_counts": {
            "legacy_human_reviewed": 0,
            "legacy_manual": 0,
            "tuple_audited_human_reviewed": 0,
            "tuple_gated_production": 0,
            "tuple_gated_unverified": 0,
            "unclassified_legacy_source": 0,
        },
    }
    # One extractor, one row, two tuple, one verifier, and one origin call.
    assert summary["provider_usage_recorded"]["recorded_structured_invocations"] == 6
    assert {
        "proposal",
        "provider_call",
        "request_id",
        "observation_id",
        "exact_excerpt",
        "quote",
        "private",
        "path",
    }.isdisjoint(_all_mapping_keys(summary))
    assert "System A                 0.80" not in serialized
    assert "tuple_ev_" not in serialized
    assert "obs_" not in serialized
    assert str(tmp_path) not in serialized


def test_current_sealed_receipt_separates_successful_and_completed_row_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = _five_stage_run_root(
        monkeypatch,
        tmp_path,
        client=_FiveStageResponseFailureRowClient(),
    )
    run = read_json(run_root / "end-to-end-paper" / "run.json")

    assert run["row_enumeration"]["successful_call_telemetry"]["calls"] == 2
    assert run["row_enumeration"]["completed_call_telemetry"]["calls"] == 3

    summary = build_public_development_summary(run_root)
    row_receipt = summary["run_binding"]["stage_chain"]["checkpoint_validation"]["row_disposition"]
    assert row_receipt["status"] == "validated"
    assert row_receipt["completed_calls"] == 3
    assert summary["provider_usage_recorded"]["recorded_structured_invocations"] == 7


def test_current_sealed_receipt_accepts_exact_extractor_resume_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = _five_stage_run_root(
        monkeypatch,
        tmp_path,
        resume_from_checkpoints=True,
    )
    run = read_json(run_root / "end-to-end-paper" / "run.json")
    execution = run["extractor"]["execution"]

    assert execution["blocks_total"] == 1
    assert execution["blocks_succeeded"] == 0
    assert execution["blocks_resumed"] == 1
    assert execution["blocks_failed"] == 0
    assert (
        sum(execution[field] for field in ("blocks_succeeded", "blocks_resumed", "blocks_failed"))
        == execution["blocks_total"]
    )

    summary = build_public_development_summary(run_root)
    assert summary["current_sealed_receipt"] is True


def test_reviewed_export_accepts_historical_pipeline_verifier_schema_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root = _five_stage_run_root(monkeypatch, tmp_path, sealed=False)
    paper_root = raw_root / "end-to-end-paper"
    verifier_path = paper_root / "private" / "verifier-gates.json"
    verifier = read_json(verifier_path)
    verifier["schema_version"] = "independent-verifier-gates/0.1"
    verifier_sha256 = write_json(verifier_path, verifier)

    lineage_path = paper_root / "candidate-lineage.json"
    lineage = read_json(lineage_path)
    lineage["verifier_sidecar_sha256"] = verifier_sha256
    lineage_sha256 = write_json(lineage_path, lineage)

    run_path = paper_root / "run.json"
    run = read_json(run_path)
    run["schema_version"] = "pipeline-run/0.3"
    run["verifier"]["sidecar"]["schema_version"] = "independent-verifier-gates/0.1"
    run["verifier"]["sidecar"]["sha256"] = verifier_sha256
    run["candidate_lineage"]["artifact_sha256"] = lineage_sha256
    run["candidate_lineage"]["verifier_gate_sidecar_sha256"] = verifier_sha256
    write_json(run_path, run)
    corpus_path = raw_root / "corpus-run.json"
    corpus = read_json(corpus_path)
    corpus["runs"][0] = run
    write_json(corpus_path, corpus)

    sealed = tmp_path / "sealed-historical-verifier-pair"
    seal_run_tree(raw_root, sealed)
    review_root = tmp_path / "historical-review"
    prepare_export_review(run_root=sealed, output_root=review_root)

    manifest = ReviewManifest.model_validate(read_json(review_root / REVIEW_MANIFEST_NAME))
    assert manifest.papers[0].verifier_gates is not None


def test_reviewed_export_rejects_mismatched_pipeline_verifier_schema_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root = _five_stage_run_root(monkeypatch, tmp_path, sealed=False)
    run_path = raw_root / "end-to-end-paper" / "run.json"
    run = read_json(run_path)
    run["schema_version"] = "pipeline-run/0.3"
    write_json(run_path, run)
    corpus_path = raw_root / "corpus-run.json"
    corpus = read_json(corpus_path)
    corpus["runs"][0] = run
    write_json(corpus_path, corpus)
    sealed = tmp_path / "sealed-mismatched-verifier-pair"
    seal_run_tree(raw_root, sealed)

    with pytest.raises(ReviewedExportError, match="exact sidecar and lineage bindings"):
        prepare_export_review(run_root=sealed, output_root=tmp_path / "mismatched-review")


def test_current_summary_requires_a_valid_run_seal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = _five_stage_run_root(monkeypatch, tmp_path, sealed=False)

    with pytest.raises(PublicDevelopmentSummaryError, match="requires a valid run seal"):
        build_public_development_summary(run_root)


def test_current_summary_requires_paired_derived_and_private_review_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = _five_stage_run_root(monkeypatch, tmp_path)

    with pytest.raises(PublicDevelopmentSummaryError, match="must be supplied together"):
        build_public_development_summary(
            run_root,
            reviewed_derived_root=tmp_path,
        )


def test_tuple_only_current_summary_is_review_only_with_zero_canonical_eee(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = _five_stage_run_root(monkeypatch, tmp_path, verifier_enabled=False)

    summary = build_public_development_summary(run_root)

    assert summary["run_binding"]["stage_chain"]["five_stage_gate_status"] == "not_run"
    assert summary["export_provenance_modes"]["source_run"]["eee_observations"] == 0
    assert (
        summary["export_provenance_modes"]["source_run"]["provenance_mode_counts"][
            "tuple_gated_unverified"
        ]
        == 0
    )


def test_current_summary_rejects_missing_verifier_gate_before_reporting_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root = _five_stage_run_root(monkeypatch, tmp_path, sealed=False)
    gate_path = raw_root / "end-to-end-paper" / "private" / "verifier-gates.json"
    gate_path.unlink()
    run_root = tmp_path / "sealed-missing-verifier-gate"
    seal_run_tree(raw_root, run_root)

    with pytest.raises(PublicDevelopmentSummaryError, match="verifier sidecar binding"):
        build_public_development_summary(run_root)


def test_current_summary_replays_nondefault_candidate_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = _five_stage_run_root(monkeypatch, tmp_path, min_confidence=0.73)

    summary = build_public_development_summary(run_root)

    assert summary["run_binding"]["candidate_validation"] == {
        "schema_version": "candidate-validation/0.2",
        "min_confidence": 0.73,
        "origin_policy": "positive_only",
    }


def test_current_summary_retains_typed_terminal_failure_without_calling_it_passed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = _five_stage_run_root(monkeypatch, tmp_path, client_mode="invalid")

    summary = build_public_development_summary(run_root)
    stages = summary["run_binding"]["stage_chain"]

    assert stages["five_stage_gate_status"] == "partial_failure"
    origin = stages["checkpoint_validation"]["origin_retrieval"]
    assert origin["status"] == "partial_failure"
    assert origin["papers_partial_failure"] == 1
    assert origin["completed_calls"] == 1
    assert origin["selected_items"] == 1
    assert origin["failed_items"] == 1
    assert len(origin["checkpoint_contract_sha256s"]) == 1
    assert origin["sidecar_sha256s"] == []
    assert summary["technical_health"]["status"] == "error"
    assert summary["technical_health"]["release_ready"] is False


def test_current_summary_rejects_nonpublic_corpus_identifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = _five_stage_run_root(
        monkeypatch,
        tmp_path,
        corpus_id="private corpus title text",
    )

    with pytest.raises(PublicDevelopmentSummaryError, match="public-safe identifier"):
        build_public_development_summary(run_root)


def test_current_summary_rejects_fully_rehashed_public_count_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root = _five_stage_run_root(monkeypatch, tmp_path, sealed=False)
    run_path = raw_root / "end-to-end-paper" / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["counts"]["tuple_passed"] += 1
    write_json(run_path, run)
    corpus_path = raw_root / "corpus-run.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["runs"][0] = run
    corpus["totals"]["tuple_passed"] += 1
    write_json(corpus_path, corpus)
    sealed = tmp_path / "sealed-count-tamper"
    seal_run_tree(raw_root, sealed)

    with pytest.raises(PublicDevelopmentSummaryError, match="public counts differ"):
        build_public_development_summary(sealed)


def test_current_summary_rejects_fully_rehashed_ready_review_state_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root = _five_stage_run_root(monkeypatch, tmp_path, sealed=False)
    run_path = raw_root / "end-to-end-paper" / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["review_state"] = {"status": "ready", "reasons": []}
    write_json(run_path, run)
    corpus_path = raw_root / "corpus-run.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["runs"][0] = run
    corpus["papers_needing_review"] = 0
    write_json(corpus_path, corpus)
    sealed = tmp_path / "sealed-review-state-tamper"
    seal_run_tree(raw_root, sealed)

    with pytest.raises(PublicDevelopmentSummaryError, match="review state differs"):
        build_public_development_summary(sealed)


def test_current_summary_validates_partial_extractor_calls_and_rejects_extra_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root = _five_stage_run_root(
        monkeypatch,
        tmp_path,
        sealed=False,
        client=_TerminalExtractorFailureClient("positive"),
    )
    sealed = tmp_path / "sealed-terminal-extractor-failure"
    seal_run_tree(raw_root, sealed)
    summary = build_public_development_summary(sealed)
    assert summary["run_binding"]["stage_chain"]["five_stage_gate_status"] == ("partial_failure")

    run_path = raw_root / "end-to-end-paper" / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    assert run["extractor"]["calls"]
    run["extractor"]["calls"].append(run["extractor"]["calls"][0])
    write_json(run_path, run)
    corpus_path = raw_root / "corpus-run.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["runs"][0] = run
    write_json(corpus_path, corpus)
    tampered = tmp_path / "sealed-extra-extractor-call"
    seal_run_tree(raw_root, tampered)

    with pytest.raises(PublicDevelopmentSummaryError, match="call ledger differs"):
        build_public_development_summary(tampered)


def test_public_call_ledgers_omit_private_request_ids_but_still_validate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root = _five_stage_run_root(
        monkeypatch,
        tmp_path,
        sealed=False,
        client=_RequestIdFiveStageRowClient("positive"),
    )
    paper_root = raw_root / "end-to-end-paper"
    run = read_json(paper_root / "run.json")

    for stage_name in ("extractor", "row_enumeration", "verifier", "origin_retrieval"):
        calls = [
            *run[stage_name]["calls"],
            *run[stage_name].get("resumed_calls", []),
        ]
        assert calls
        for call in calls:
            assert "request_id" not in call
            assert "temperature" in call
            assert "seed" in call
            assert "reasoning_tokens" in call

    extractor_checkpoint = read_json(paper_root / "private" / "extractor-checkpoint.json")
    extractor_call = next(iter(extractor_checkpoint["blocks"].values()))["calls"][0]
    row_checkpoint = read_json(paper_root / "private" / "row-enumeration-checkpoint.json")
    row_call = next(iter(row_checkpoint["batches"].values()))["calls"][0]
    verifier_checkpoint = read_json(paper_root / "private" / "verifier-checkpoint.json")
    verifier_call = next(iter(verifier_checkpoint["candidates"].values()))["provider_call"]
    origin_checkpoint = read_json(paper_root / "private" / "origin-retrieval-checkpoint.json")
    origin_call = next(iter(origin_checkpoint["candidates"].values()))["provider_call"]
    for private_call in (extractor_call, row_call, verifier_call, origin_call):
        assert private_call["request_id"].startswith("private-")
        assert "temperature" in private_call
        assert "seed" in private_call
        assert "reasoning_tokens" in private_call

    sealed = tmp_path / "sealed-private-request-ids"
    seal_run_tree(raw_root, sealed)
    summary = build_public_development_summary(sealed)
    assert summary["run_binding"]["stage_chain"]["five_stage_gate_status"] == "validated"


def test_current_summary_rejects_fully_rehashed_final_candidate_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root = _five_stage_run_root(monkeypatch, tmp_path, sealed=False)
    paper_root = raw_root / "end-to-end-paper"
    observations_path = paper_root / "observations.jsonl"
    observations = [
        json.loads(line) for line in observations_path.read_text(encoding="utf-8").splitlines()
    ]
    observations[0]["notes"].append("adversarial fully rehashed mutation")
    observations_sha256 = write_jsonl(observations_path, observations)
    tampered_candidate = CandidateObservation.model_validate(observations[0])

    lineage_path = paper_root / "candidate-lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["observations_sha256"] = observations_sha256
    lineage_record = next(
        item
        for item in lineage["candidates"]
        if item["final_observation_id"] == tampered_candidate.observation_id
    )
    lineage_record["candidate_sha256"] = sha256_bytes(canonical_json_bytes(tampered_candidate))
    lineage_sha256 = write_json(lineage_path, lineage)

    run_path = paper_root / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["candidate_lineage"]["artifact_sha256"] = lineage_sha256
    write_json(run_path, run)
    corpus_path = raw_root / "corpus-run.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["runs"][0] = run
    write_json(corpus_path, corpus)

    sealed = tmp_path / "sealed-fully-rehashed-tamper"
    seal_run_tree(raw_root, sealed)
    with pytest.raises(PublicDevelopmentSummaryError, match="final observations differ"):
        build_public_development_summary(sealed)


def test_current_summary_rejects_resealed_verifier_ledger_disagreement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_root = _five_stage_run_root(monkeypatch, tmp_path, sealed=False)
    ledger_path = raw_root / "end-to-end-paper" / "verifications.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["provider_assessment"]["justification"] = "resealed ledger-only mutation"
    write_jsonl(ledger_path, rows)
    sealed = tmp_path / "sealed-verifier-ledger-tamper"
    seal_run_tree(raw_root, sealed)

    with pytest.raises(PublicDevelopmentSummaryError, match="verifier result ledger differs"):
        build_public_development_summary(sealed)


def test_current_summary_joins_verified_reviewed_result_without_combining_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = _five_stage_run_root(monkeypatch, tmp_path)
    review_root = tmp_path / "private-review"
    prepare_export_review(run_root=run_root, output_root=review_root)
    _complete_current_review(review_root)
    derived_root = tmp_path / "reviewed-derived"
    compose_reviewed_eee(
        run_root=run_root,
        decisions_path=review_root / REVIEW_DECISIONS_NAME,
        output_root=derived_root,
    )

    summary = build_public_development_summary(
        run_root,
        reviewed_derived_root=derived_root,
        review_root=review_root,
    )
    serialized = json.dumps(summary, sort_keys=True)
    provenance = summary["export_provenance_modes"]

    assert provenance["source_run"]["eee_observations"] == 0
    assert provenance["reviewed_derived"]["status"] == "contextually_verified"
    assert provenance["reviewed_derived"]["locked_review_context_verified"] is True
    assert provenance["reviewed_derived"]["independence_status"] == "not_measured"
    assert provenance["reviewed_derived"]["exported_observation_authority_mode_counts"] == {
        "adjudicated": 0,
        "dual_consensus": 0,
        "single_expert": provenance["reviewed_derived"]["eee_observations"],
    }
    assert provenance["reviewed_derived"]["provenance_mode_count_status"] == "measured"
    assert (
        sum(provenance["reviewed_derived"]["provenance_mode_counts"].values())
        == (provenance["reviewed_derived"]["eee_observations"])
    )
    assert provenance["reviewed_derived"]["eee_observations"] > 0
    assert provenance["counts_combined"] is False
    assert "private-reviewer-a" not in serialized
    assert "We evaluate System A" not in serialized
    assert "obs_" not in serialized
    cli_output = tmp_path / "public" / "contextual-summary.json"
    result = RUNNER.invoke(
        app,
        [
            "build-public-development-summary",
            str(run_root),
            "--reviewed-derived-root",
            str(derived_root),
            "--review-root",
            str(review_root),
            "--output",
            str(cli_output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        json.loads(cli_output.read_text(encoding="utf-8"))["export_provenance_modes"][
            "reviewed_derived"
        ]["locked_review_context_verified"]
        is True
    )
    with pytest.raises(PublicDevelopmentSummaryError, match="reviewed derived root"):
        write_public_development_summary(
            run_root,
            derived_root / "public-summary.json",
            reviewed_derived_root=derived_root,
            review_root=review_root,
        )


def test_current_summary_rejects_reviewed_result_from_another_sealed_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_root = _five_stage_run_root(monkeypatch, tmp_path / "first")
    review_root = tmp_path / "first-review"
    prepare_export_review(run_root=first_root, output_root=review_root)
    _complete_current_review(review_root)
    derived_root = tmp_path / "first-derived"
    compose_reviewed_eee(
        run_root=first_root,
        decisions_path=review_root / REVIEW_DECISIONS_NAME,
        output_root=derived_root,
    )
    second_root = _five_stage_run_root(monkeypatch, tmp_path / "second")

    with pytest.raises(PublicDevelopmentSummaryError, match="contextual verification"):
        build_public_development_summary(
            second_root,
            reviewed_derived_root=derived_root,
            review_root=review_root,
        )
