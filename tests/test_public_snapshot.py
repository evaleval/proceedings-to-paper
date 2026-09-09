from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from proceedings_to_eee import cli
from proceedings_to_eee.cli import app
from proceedings_to_eee.composition.eee import compose_eee_records
from proceedings_to_eee.domain.export_provenance import (
    legacy_export_provenance,
    tuple_gated_export_provenance,
)
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.evaluation.corpus_score import FIELD_NAMES, aggregate_reference_scores
from proceedings_to_eee.evaluation.human_review import build_human_review_template
from proceedings_to_eee.evaluation.reference_score import score_claim_type_pairs
from proceedings_to_eee.io import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from proceedings_to_eee.providers.openrouter import ProviderCall, public_provider_call
from proceedings_to_eee.public_snapshot import (
    PublicSnapshotError,
    _aggregate_private_calls,
    _project_composer_eee_record,
    _project_human_review_summary,
    _project_input_observability,
    _project_negative_safety,
    _project_paper_run,
    _project_reference_evaluation,
    _project_request_contract,
    _public_report_input,
    _validate_human_review_corpus_population,
    _validate_reference_score_details,
    build_public_snapshot,
)
from proceedings_to_eee.sources.manifest import SourceManifest
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = PROJECT_ROOT / "schemas" / "eee-0.2.2" / "eval.schema.json"
SCHEMA_SHA256 = "088fed8029d42fb3a607aa67e1a05c39e425241b5cd90803705b37562f402f2a"


def _provider_request_contract(
    *,
    schema_name: str,
    seed: int | None,
    require_parameters: bool,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, object]:
    materialized = model is not None
    result: dict[str, object] = {
        "schema_version": (
            "provider-request-contract/0.2" if materialized else "provider-request-contract/0.1"
        ),
        "privacy": {"data_collection": "deny", "zdr": True},
        "routing": {"require_parameters": require_parameters},
        "schema": {
            "response_format": "json_schema",
            "schema_name": schema_name,
            "schema_sha256": "a" * 64,
            "schema_strict": True,
        },
        "seed": seed,
    }
    if materialized:
        assert max_tokens is not None
        result["max_tokens"] = max_tokens
        result["completion_token_parameter"] = (
            "max_completion_tokens" if model.startswith("openai/") else "max_tokens"
        )
    return result


def _public_call(
    *,
    model: str,
    schema_name: str,
    seed: int | None,
    require_parameters: bool,
    max_tokens: int,
    temperature: float | None = 0.0,
    cost_usd: float | None = 0.001,
    input_tokens: int | None = 100,
    output_tokens: int | None = 20,
    reasoning_tokens: int | None = None,
    total_tokens: int | None = 120,
    request_id: str | None = None,
) -> dict[str, object]:
    return public_provider_call(
        ProviderCall(
            model_requested=model,
            model_returned=model,
            provider_returned="OpenAI",
            prompt_sha256="f" * 64,
            response_sha256="e" * 64,
            temperature=temperature,
            reasoning_effort="minimal",
            max_tokens=max_tokens,
            completion_token_parameter=(
                "max_completion_tokens" if model.startswith("openai/") else "max_tokens"
            ),
            seed=seed,
            schema_name=schema_name,
            schema_sha256="a" * 64,
            require_parameters=require_parameters,
            latency_seconds=0.1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            request_id=request_id,
            finish_reason="stop",
            attempts=1,
        )
    )


def _pipeline_counts(**overrides: int) -> dict[str, int]:
    fields = (
        "candidates",
        "candidates_before_deduplication",
        "duplicates_removed",
        "candidates_needing_review",
        "semantic_safety_reviews",
        "primary_results",
        "exported",
        "eee_records",
        "eee_schema_issues",
        "tuple_candidates",
        "tuple_passed",
        "tuple_review",
        "tuple_unsupported",
        "tuple_failed",
        "tuple_resumed",
        "verifications",
        "verifier_accepts",
        "verifier_rejects",
        "verifier_reviews",
        "verifier_failed",
        "verifier_resumed",
        "origin_candidates",
        "origin_resumed",
        "origin_failed",
        "origin_positive_review_only",
        "origin_external",
        "origin_unresolved",
        "origin_no_signal",
        "spot_checks",
        "spot_checks_exact",
        "reference_observations",
        "reference_true_positives",
        "reference_false_positives",
        "reference_false_negatives",
        "negative_control_false_primary",
    )
    counts = dict.fromkeys(fields, 0)
    counts.update(overrides)
    return counts


def _claim_type_classification(*, two_classes: bool = False) -> dict[str, object]:
    labels = (
        "primary_result",
        "secondary_claim",
        "illustration",
        "method_metadata",
        "uncertain",
    )
    per_class = {}
    for label in labels:
        supported = label == "primary_result" or (two_classes and label == "secondary_claim")
        per_class[label] = {
            "support": int(supported),
            "predicted": int(supported),
            "true_positives": int(supported),
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0 if supported else None,
            "recall": 1.0 if supported else None,
            "f1": 1.0 if supported else None,
        }
    return {
        "basis": 2 if two_classes else 1,
        "basis_scope": "matched reference observations and matched negative controls",
        "supported_classes": 2 if two_classes else 1,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "per_class": per_class,
    }


def _reference_score_detail_fixture(*, matches: int = 2) -> dict[str, object]:
    positive_matches = [
        {
            "reference_id": f"reference-{index}",
            "observation_id": f"observation-{index}",
            "expected_claim_type": "primary_result",
            "actual_claim_type": "primary_result",
            **dict.fromkeys(FIELD_NAMES, True),
        }
        for index in range(matches)
    ]
    return {
        "schema_version": "reference-score/0.7",
        "paper_id": "paper",
        "reference_observations": matches,
        "recall_basis": matches,
        "field_matching_basis": matches,
        "candidate_primary_results": matches,
        "primary_candidates_total": matches,
        "primary_candidates_in_coverage": matches,
        "primary_candidates_out_of_coverage": 0,
        "precision_basis": matches,
        "coverage": {
            "recall_scope": "annotated_reference_observations",
            "field_matching_scope": "annotated_reference_observations",
            "precision_scope": "fully_annotated_labels",
        },
        "detection": {
            "true_positives": matches,
            "precision_true_positives": matches,
            "precision_basis": matches,
            "recall_basis": matches,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0 if matches else None,
            "precision_defined": bool(matches),
            "recall": 1.0 if matches else 0.0,
            "f1": 1.0 if matches else None,
        },
        "input_observability": {
            "status": "not_assessed",
            "reference_observations": matches,
            "observable_reference_observations": None,
            "unobservable_reference_observations": None,
            "observation_coverage": None,
            "model_conditional_detection": {
                "true_positives": None,
                "false_negatives": None,
                "recall_basis": None,
                "recall": None,
            },
        },
        "field_accuracy": dict.fromkeys(FIELD_NAMES, 1.0),
        "matches": positive_matches,
        "unmatched_candidate_ids": [],
        "unmatched_candidate_ids_in_coverage": [],
        "unmatched_candidate_ids_out_of_coverage": [],
        "unmatched_primary_candidate_ids_in_coverage": [],
        "unmatched_primary_candidate_ids_out_of_coverage": [],
        "negative_control_safety": {
            "controls_total": 0,
            "control_ids": [],
            "matched_control_ids": [],
            "unmatched_control_ids": [],
            "control_status": {},
            "matched_candidate_ids": [],
            "false_primary_candidate_ids": [],
            "false_primary_export_candidate_ids": [],
            "matched_control_count": 0,
            "control_match_coverage": None,
            "control_match_coverage_defined": False,
            "matched_candidate_count": 0,
            "false_primary_count": 0,
            "false_primary_export_count": 0,
            "false_primary_rate_basis": 0,
            "false_primary_rate_defined": False,
            "false_primary_rate": 0.0,
            "examined_control_ids": [],
            "not_examined_control_ids": [],
            "control_examination_coverage": None,
            "passed_by_abstention_count": 0,
            "measurement_status": "not_measured",
            "zero_false_primary_gate_passed": None,
            "zero_false_primary_export_gate_passed": None,
            "matches": [],
        },
        "claim_type_classification": score_claim_type_pairs(
            [("primary_result", "primary_result")] * matches
        ),
    }


def _rewrite_corpus_and_paper_run(run_root: Path, corpus: dict[str, object]) -> None:
    write_json(run_root / "corpus-run.json", corpus)
    [run] = corpus["runs"]
    write_json(run_root / run["paper_id"] / "run.json", run)


def _refresh_human_review_audit(run_root: Path, summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["audit_id"] = build_human_review_template(
        run_root, sample_size=summary["sample"]["requested"]
    ).audit_id
    write_json(summary_path, summary)


def _private_inputs(
    root: Path,
    manifest: SourceManifest,
    candidate: CandidateObservation,
) -> tuple[Path, Path, Path, Path]:
    run_root = root / "run"
    paper_root = run_root / manifest.paper_id
    schema, authority = load_schema(SCHEMA, SCHEMA_SHA256)
    del schema
    record = compose_eee_records(
        manifest=manifest,
        candidates=[candidate],
        schema_version=authority.version,
        provenance=legacy_export_provenance(
            [candidate], reason="public_snapshot_fixture_manual_composition"
        ),
    )[0]
    write_json(paper_root / "source-manifest.json", manifest)
    write_json(paper_root / "eee" / "atlas-moderation-api.json", record)
    write_json(
        run_root / "corpus-run.json",
        {
            "schema_version": "corpus-run/0.3",
            "corpus_id": "pilot-corpus",
            "status": "success",
            "generated_at": "2026-08-04T12:00:00Z",
            "papers": 1,
            "papers_total": 1,
            "papers_not_started": 0,
            "papers_succeeded": 1,
            "papers_failed": 0,
            "papers_bounded_incomplete": 0,
            "papers_with_eee": 1,
            "papers_without_candidates": 0,
            "papers_without_eee": 0,
            "papers_needing_review": 0,
            "provider_budget": {
                "schema_version": "provider-budget-summary/0.2",
                "status": "available",
                "structured_calls_started": 1,
                "structured_calls_completed": 1,
                "structured_calls_pending": 0,
                "provider_call_telemetry_calls": 1,
                "provider_call_telemetry_missing_calls": 0,
                "provider_reported_cost_calls": 1,
                "provider_reported_cost_usd": 0.001,
                "provider_reported_input_tokens_lower_bound": 100,
                "provider_reported_output_tokens_lower_bound": 20,
                "provider_reported_reasoning_tokens_lower_bound": 0,
                "provider_reported_total_tokens_lower_bound": 120,
            },
            "totals": _pipeline_counts(
                candidates=1,
                candidates_before_deduplication=1,
                primary_results=1,
                exported=1,
                eee_records=1,
                spot_checks=1,
                spot_checks_exact=1,
            ),
            "runs": [
                {
                    "schema_version": "pipeline-run/0.4",
                    "paper_id": manifest.paper_id,
                    "title": manifest.title,
                    "status": "success",
                    "wall_clock_seconds": 3.25,
                    "source_manifest_sha256": sha256_file(paper_root / "source-manifest.json"),
                    "counts": _pipeline_counts(
                        candidates=1,
                        candidates_before_deduplication=1,
                        primary_results=1,
                        exported=1,
                        eee_records=1,
                        spot_checks=1,
                        spot_checks_exact=1,
                    ),
                    "review_state": {"status": "ready", "reasons": []},
                    "result_block_segmentation": {
                        "max_lines": 40,
                        "max_characters": 6_000,
                        "context_lines": 8,
                        "trailing_context_lines": 2,
                        "overlap_lines": 3,
                        "signal_gap_lines": 3,
                        "max_blank_gap": 1,
                        "max_data_rows": 6,
                        "min_signal_score": 1.5,
                        "max_blocks_per_page": 6,
                    },
                    "candidate_validation": {
                        "schema_version": "candidate-validation/0.1",
                        "min_confidence": 0.8,
                    },
                    "extractor": {
                        "provider": "openrouter",
                        "model": "example/extractor",
                        "temperature": None,
                        "reasoning_effort": "minimal",
                        "max_tokens": 16_000,
                        "seed": None,
                        "require_parameters": True,
                        "prompt_sha256": "b" * 64,
                        "request_contract": _provider_request_contract(
                            schema_name="extractor",
                            seed=None,
                            require_parameters=True,
                            model="example/extractor",
                            max_tokens=16_000,
                        ),
                        "warnings": ["DO_NOT_PUBLISH_WARNING " + "/" + "Users/example/private"],
                        "calls": [
                            {
                                **_public_call(
                                    model="example/extractor",
                                    schema_name="extractor",
                                    seed=None,
                                    require_parameters=True,
                                    max_tokens=16_000,
                                    temperature=None,
                                    request_id="DO_NOT_PUBLISH_REQUEST",
                                ),
                                "request_id": "DO_NOT_PUBLISH_REQUEST",
                                "exact_quote": "DO_NOT_PUBLISH_QUOTE",
                            }
                        ],
                        "resumed_calls": [],
                        "execution": {
                            "blocks_total": 1,
                            "blocks_succeeded": 1,
                            "blocks_failed": 0,
                            "blocks_resumed": 0,
                            "calls_succeeded": 1,
                            "calls_failed": 0,
                            "calls_resumed": 0,
                            "calls_resumed_succeeded": 0,
                            "calls_resumed_failed": 0,
                            "no_call_failures": 0,
                            "requests_rejected": 0,
                            "transport_failures": 0,
                            "local_failures": 0,
                        },
                    },
                    "row_enumeration": {
                        "enabled": False,
                        "provider": "openrouter",
                        "model": "example/extractor",
                        "temperature": 0,
                        "reasoning_effort": "minimal",
                        "max_tokens": 1_000,
                        "seed": 7,
                        "require_parameters": False,
                        "prompt_sha256": "c" * 64,
                        "request_contract": _provider_request_contract(
                            schema_name="row_extractor",
                            seed=7,
                            require_parameters=False,
                            model="example/extractor",
                            max_tokens=1_000,
                        ),
                        "calls": [],
                        "execution": {
                            "batches_total": 0,
                            "batches_resumed": 0,
                            "batches_executed": 0,
                            "invalid_rows_seen": 0,
                            "unknown_row_ids_seen": 0,
                        },
                    },
                    "tuple_resolution": {
                        "enabled": False,
                        "provider": "openrouter",
                        "model": None,
                        "temperature": 0,
                        "reasoning_effort": "minimal",
                        "max_tokens": 2_000,
                        "seed": 7,
                        "require_parameters": True,
                        "prompt_sha256": "d" * 64,
                        "request_contract": _provider_request_contract(
                            schema_name="tuple_resolution",
                            seed=7,
                            require_parameters=True,
                        ),
                        "calls": [],
                        "resumed_calls": [],
                        "execution": {
                            "candidates_selected": 0,
                            "candidates_passed": 0,
                            "candidates_routed_to_review": 0,
                            "candidates_unsupported": 0,
                            "candidates_failed": 0,
                            "candidates_mismatched": 0,
                            "candidates_resumed": 0,
                        },
                    },
                    "verifier": {
                        "enabled": False,
                        "model": None,
                        "temperature": None,
                        "reasoning_effort": "minimal",
                        "max_tokens": 2_000,
                        "seed": None,
                        "require_parameters": True,
                        "request_contract": _provider_request_contract(
                            schema_name="verifier",
                            seed=None,
                            require_parameters=True,
                        ),
                        "calls": [],
                        "resumed_calls": [],
                        "execution": {
                            "candidates_selected": 0,
                            "candidates_unbound": 0,
                            "candidates_verified": 0,
                            "candidates_failed": 0,
                            "candidates_resumed": 0,
                            "candidates_resumed_succeeded": 0,
                            "candidates_resumed_failed": 0,
                            "candidates_executed": 0,
                            "candidates_executed_succeeded": 0,
                            "candidates_executed_failed": 0,
                        },
                    },
                    "origin_retrieval": {
                        "enabled": False,
                        "requires_independent_verifier_accept": True,
                        "model": None,
                        "temperature": 0,
                        "reasoning_effort": "minimal",
                        "max_tokens": 2_000,
                        "seed": 7,
                        "require_parameters": True,
                        "prompt_sha256": "e" * 64,
                        "request_contract": _provider_request_contract(
                            schema_name="origin_retrieval",
                            seed=7,
                            require_parameters=True,
                        ),
                        "calls": [],
                        "resumed_calls": [],
                        "execution": {
                            "candidates_selected": 0,
                            "candidates_resumed": 0,
                            "candidates_failed": 0,
                            "candidates_deterministic_external": 0,
                        },
                    },
                    "eee_schema": {
                        "version": authority.version,
                        "sha256": authority.sha256,
                    },
                    "code": {
                        "git_commit": "fixture",
                        "git_dirty": False,
                        "git_available": False,
                        "source_tree_sha256": "8" * 64,
                    },
                    "reference_path": "/" + "Users/example/private/reference.json",
                }
            ],
        },
    )
    corpus_payload = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    write_json(paper_root / "run.json", corpus_payload["runs"][0])
    write_jsonl(paper_root / "observations.jsonl", [candidate])
    write_json(
        run_root / "reference-audit.json",
        {
            "schema_version": "corpus-reference-audit/0.1",
            "corpus_id": "pilot-corpus",
            "status": "passed",
            "papers": 1,
            "papers_passed": 0,
            "papers_failed": 0,
            "papers_skipped": 1,
            "text_verified": 0,
            "visual_verified": 0,
            "failed_evidence": 0,
            "results": [
                {
                    "paper_id": manifest.paper_id,
                    "status": "skipped_no_reference",
                    "warnings": ["DO_NOT_PUBLISH_AUDIT_WARNING"],
                }
            ],
        },
    )
    model_selection = root / "private-model-selection.json"
    selection_request_contract = _provider_request_contract(
        schema_name="extractor",
        seed=None,
        require_parameters=True,
    )
    write_json(
        model_selection,
        {
            "schema_version": "extractor-bakeoff-score/0.3",
            "bakeoff_id": "pilot-extractors",
            "configuration_sha256": "b" * 64,
            "run_contract_sha256": "c" * 64,
            "provider_phase_seal_sha256": "d" * 64,
            "checkpoint_sha256": "e" * 64,
            "stage_contract_sha256": "f" * 64,
            "execution_selection_sha256": "9" * 64,
            "code": {
                "git_commit": "fixture",
                "git_dirty": False,
                "git_available": False,
                "source_tree_sha256": "8" * 64,
            },
            "declared_models": ["example/extractor"],
            "executed_models": ["example/extractor"],
            "inputs": [
                {
                    "manifest_path": "/" + "Users/example/private/source-manifest.json",
                    "prompt": "DO_NOT_PUBLISH_PROMPT",
                }
            ],
            "determinism": {
                "seed": None,
                "temperature": None,
                "reasoning_effort": "minimal",
                "require_parameters": True,
                "fresh_repetitions": 1,
                "max_tokens": 16_000,
                "min_confidence": 0.8,
                "prompt_sha256": "b" * 64,
                "reference_prompt_isolation": True,
                "segmentation": {
                    "max_lines": 40,
                    "max_characters": 6_000,
                    "context_lines": 8,
                    "trailing_context_lines": 2,
                    "overlap_lines": 3,
                    "signal_gap_lines": 3,
                    "max_blank_gap": 1,
                    "max_data_rows": 6,
                    "min_signal_score": 1.5,
                    "max_blocks_per_page": 6,
                },
            },
            "request_contract": selection_request_contract,
            "request_contract_sha256": sha256_bytes(
                canonical_json_bytes(selection_request_contract)
            ),
            "models": [
                {
                    "model": "example/extractor",
                    "label": "Example extractor",
                    "contract_eligibility": "contract_eligible",
                    "eligibility_reason_codes": [],
                    "matched_quality_status": "executed",
                    "quality_status": "measured",
                    "aggregate": {
                        "execution": {
                            "cases_attempted": 1,
                            "cases_succeeded": 1,
                            "cases_partial_failure": 0,
                            "cases_failed": 0,
                            "case_success_rate": 1.0,
                            "cases_scored": 1,
                            "case_scored_rate": 1.0,
                            "repetitions_predeclared": 1,
                            "repetitions_attempted": 1,
                            "repetitions_succeeded": 1,
                            "repetitions_partial_failure": 0,
                            "repetitions_failed": 0,
                            "repetitions_scored": 1,
                            "calls_predeclared": 1,
                            "calls_attempted": 1,
                            "calls_succeeded": 1,
                            "calls_failed": 0,
                            "call_success_rate": 1.0,
                        },
                        "schema": {
                            "calls_in_end_to_end_denominator": 1,
                            "structured_responses_observed": 1,
                            "structured_responses_valid": 1,
                            "structured_responses_invalid": 0,
                            "structured_response_not_observed": 0,
                            "valid_rate_of_observed_responses": 1.0,
                            "end_to_end_schema_success_rate": 1.0,
                        },
                        "evidence": {
                            "candidate_text_support": {
                                "supported": 1,
                                "partially_supported": 0,
                                "unsupported": 0,
                                "unverified": 0,
                            },
                            "reference_evidence_supported_accuracy": {
                                "macro": 1.0,
                                "micro": 1.0,
                            },
                            "reference_page_anchor_accuracy": {
                                "macro": 1.0,
                                "micro": 1.0,
                            },
                        },
                        "quality_measurement": {
                            "status": "measured",
                            "scored_repetitions": 1,
                            "unmeasured_repetitions": 0,
                        },
                        "quality": {
                            "scored_cases": 1,
                            "scored_repetitions": 1,
                            "macro": {
                                "detection": {
                                    "precision": 1.0,
                                    "recall": 1.0,
                                    "f1": 1.0,
                                    "defined_cases": {
                                        "precision": 1,
                                        "recall": 1,
                                        "f1": 1,
                                    },
                                    "undefined_cases": {
                                        "precision": 0,
                                        "recall": 0,
                                        "f1": 0,
                                    },
                                },
                                "field_accuracy": dict.fromkeys(FIELD_NAMES, 1.0),
                            },
                            "micro": {
                                "reference_observations": 1,
                                "detection": {
                                    "true_positives": 1,
                                    "precision_true_positives": 1,
                                    "precision_basis": 1,
                                    "recall_basis": 1,
                                    "false_positives": 0,
                                    "false_negatives": 0,
                                    "precision": 1.0,
                                    "precision_defined": True,
                                    "recall": 1.0,
                                    "f1": 1.0,
                                },
                                "field_accuracy": dict.fromkeys(FIELD_NAMES, 1.0),
                                "input_observability": {
                                    "status": "measured",
                                    "cases_measured": 1,
                                    "cases_not_assessed": 0,
                                    "measured_reference_observations": 1,
                                    "reference_observations": 1,
                                    "observable_reference_observations": 1,
                                    "unobservable_reference_observations": 0,
                                    "observation_coverage": 1.0,
                                    "model_conditional_detection": {
                                        "true_positives": 1,
                                        "false_negatives": 0,
                                        "recall_basis": 1,
                                        "recall": 1.0,
                                    },
                                },
                            },
                        },
                        "usage": {
                            "input_tokens": {
                                "total": 10,
                                "reported_calls": 1,
                                "missing_calls": 0,
                            },
                            "output_tokens": {
                                "total": 2,
                                "reported_calls": 1,
                                "missing_calls": 0,
                            },
                            "reasoning_tokens": {
                                "total": 0,
                                "reported_calls": 0,
                                "missing_calls": 1,
                            },
                            "total_tokens": {
                                "total": 12,
                                "reported_calls": 1,
                                "missing_calls": 0,
                            },
                            "cost_usd": {
                                "total": 0.001,
                                "reported_calls": 1,
                                "missing_calls": 0,
                            },
                            "latency_seconds": {
                                "total": 0.1,
                                "mean": 0.1,
                                "p50": 0.1,
                                "p95": 0.1,
                                "max": 0.1,
                                "reported_calls": 1,
                                "missing_calls": 0,
                            },
                        },
                        "claim_type_classification": _claim_type_classification(two_classes=True),
                        "negative_control_safety": {
                            "controls_total": 1,
                            "controls_matched": 0,
                            "control_match_coverage": 0.0,
                            "control_match_coverage_defined": True,
                            "controls_examined": 1,
                            "controls_not_examined": 0,
                            "control_examination_coverage": 1.0,
                            "passed_by_abstention_count": 1,
                            "control_trials": {
                                "total": 1,
                                "matched": 0,
                                "examined": 1,
                                "passed_by_abstention": 1,
                            },
                            "measurement_status": "measured",
                            "matched_candidates": 0,
                            "false_primary_count": 0,
                            "false_primary_export_count": 0,
                            "false_primary_rate": 0.0,
                            "zero_false_primary_gate_passed": True,
                            "zero_false_primary_export_gate_passed": True,
                        },
                        "model_selection_gates": {
                            "claim_type_macro_f1": {
                                "status": "passed",
                                "value": 1.0,
                                "threshold": 0.9,
                                "direction": "at_least",
                            },
                            "false_primary_controls": {
                                "status": "passed",
                                "value": 0.0,
                                "threshold": 0.0,
                                "direction": "at_most",
                            },
                            "false_primary_exports": {
                                "status": "passed",
                                "value": 0.0,
                                "threshold": 0.0,
                                "direction": "at_most",
                            },
                        },
                    },
                    "cases": [
                        {
                            "request_id": "DO_NOT_PUBLISH_BAKEOFF_REQUEST",
                            "raw_response": "DO_NOT_PUBLISH_RESPONSE",
                        }
                    ],
                }
            ],
            "privacy": {
                "references_loaded_after_provider_phase_sealed": True,
                "reference_paths_in_score": False,
            },
        },
    )
    human_review = root / "private-human-review-summary.json"
    audit_id = build_human_review_template(run_root, sample_size=1).audit_id
    write_json(
        human_review,
        {
            "schema_version": "human-review-summary/0.1",
            "audit_id": audit_id,
            "sampling_policy": "risk-stratified-paper-coverage/0.1",
            "population": {
                "candidates": 1,
                "papers": 1,
                "papers_without_candidates": 0,
            },
            "sample": {
                "requested": 1,
                "reviewed": 1,
                "papers_reviewed": 1,
                "paper_coverage": 1.0,
                "risk_score_min": 10,
                "risk_score_max": 10,
                "risk_score_mean": 10.0,
                "risk_reason_counts": {"exported": 1},
                "item_type_counts": {
                    "candidate": 1,
                    "paper_without_candidates": 0,
                },
                "papers_without_candidates_reviewed": 0,
            },
            "decisions": {
                "completed": 1,
                "outcome_counts": {
                    "confirmed": 1,
                    "incorrect": 0,
                    "needs_followup": 0,
                },
                "issue_counts": {
                    "claim_type": 0,
                    "role": 0,
                    "version": 0,
                    "scope": 0,
                    "metric": 0,
                    "unit": 0,
                    "value": 0,
                    "evidence": 0,
                    "export_decision": 0,
                    "duplicate": 0,
                    "other": 0,
                },
            },
            "privacy": {
                "contains_evidence_quotes": False,
                "contains_candidate_payloads": False,
                "contains_provider_raw_data": False,
                "contains_local_paths": False,
                "contains_reviewer_notes": False,
            },
            "private_template_path": "/" + "Users/example/private/review-template.json",
            "private_quote": "DO_NOT_PUBLISH_REVIEW_QUOTE",
        },
    )
    return (
        run_root,
        model_selection,
        human_review,
        paper_root / "eee" / "atlas-moderation-api.json",
    )


def _build_snapshot(
    inputs: tuple[Path, Path, Path],
    output_root: Path,
    *,
    snapshot_id: str,
    selected_model: str | None = None,
) -> Path:
    run_root, model_selection, human_review = inputs
    return build_public_snapshot(
        snapshot_id=snapshot_id,
        corpus_run_root=run_root,
        model_selection_path=model_selection,
        human_review_summary_path=human_review,
        schema_path=SCHEMA,
        schema_sha256=SCHEMA_SHA256,
        output_root=output_root,
        selected_model=selected_model,
    )


def _json_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | {
            child_key for child in value.values() for child_key in _json_keys(child)
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return {child_key for child in value for child_key in _json_keys(child)}
    return set()


def test_public_snapshot_is_allowlist_only_and_reproducible(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    first = _build_snapshot(
        (run_root, model_selection, human_review),
        tmp_path / "public-a",
        snapshot_id="pilot-v1",
        selected_model="example/extractor",
    )
    second = _build_snapshot(
        (run_root, model_selection, human_review),
        tmp_path / "public-b",
        snapshot_id="pilot-v1",
        selected_model="example/extractor",
    )

    expected_files = {
        "README.md",
        "SHA256SUMS",
        "corpus-review.html",
        "eee/synthetic-audit-study/atlas-moderation-api.json",
        "human-review.json",
        "model-selection.json",
        "reference-audit.json",
        "snapshot.json",
        "sources.json",
    }
    first_files = {
        path.relative_to(first).as_posix() for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix() for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files == expected_files
    for relative in sorted(expected_files):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()

    public_bytes = b"\n".join((first / path).read_bytes() for path in sorted(expected_files))
    for private_marker in (
        b"DO_NOT_PUBLISH",
        b"/" + b"Users/",
        b"cache_relpath",
        b"request_id",
        b"raw_response",
    ):
        assert private_marker not in public_bytes
    public_keys = set()
    for path in first.rglob("*.json"):
        public_keys |= _json_keys(json.loads(path.read_text(encoding="utf-8")))
    assert not {"calls", "warnings", "quote", "exact_quote"} & public_keys

    selection = json.loads((first / "model-selection.json").read_text(encoding="utf-8"))
    assert selection["selection"] == {
        "selected_model": "example/extractor",
        "status": "selected",
    }
    assert "inputs" not in selection
    assert "cases" not in selection["models"][0]
    review = json.loads((first / "human-review.json").read_text(encoding="utf-8"))
    assert review["schema_version"] == "human-review-summary/0.1"
    assert review["sample"]["paper_coverage"] == 1.0
    assert review["decisions"]["completed"] == 1
    assert review["privacy"] == {
        "contains_candidate_payloads": False,
        "contains_evidence_quotes": False,
        "contains_local_paths": False,
        "contains_provider_raw_data": False,
        "contains_reviewer_notes": False,
    }
    assert "private_template_path" not in review
    assert "private_quote" not in review
    public_eee = json.loads(
        (first / "eee/synthetic-audit-study/atlas-moderation-api.json").read_text(encoding="utf-8")
    )
    provenance = public_eee["evaluation_results"][0]["score_details"]["details"]
    assert provenance["paper_id"] == "synthetic-audit-study"
    assert provenance["evidence_1_source_id"] == "src_paper"
    assert provenance["evidence_1_source_sha256"] == "a" * 64
    assert provenance["evidence_1_page"] == "7"
    assert provenance["evidence_1_kind"] == "table"
    assert provenance["evidence_1_quote_sha256"]
    assert "quote" not in provenance
    snapshot = json.loads((first / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["schema_version"] == "public-pilot-snapshot/0.3"
    private_review = json.loads(human_review.read_text(encoding="utf-8"))
    assert snapshot["human_review"] == {
        "audit_id": private_review["audit_id"],
        "paper_coverage": 1.0,
        "path": "human-review.json",
        "source_artifact_sha256": sha256_file(human_review),
    }
    paper = snapshot["corpora"][0]["papers_detail"][0]
    assert paper["extractor"]["usage"]["reasoning_tokens"] is None
    assert paper["extractor"]["usage"]["reasoning_tokens_reported_calls"] == 0
    assert paper["extractor"]["usage"]["reasoning_tokens_missing_calls"] == 1
    assert paper["candidate_validation"] == {
        "schema_version": "candidate-validation/0.1",
        "min_confidence": 0.8,
    }
    assert {
        "extractor",
        "row_enumeration",
        "tuple_resolution",
        "verifier",
        "origin_retrieval",
    } <= paper.keys()
    assert paper["extractor"]["request_contract"]["max_tokens"] == 16_000
    assert paper["extractor"]["request_contract"]["completion_token_parameter"] == "max_tokens"
    assert paper["verifier"]["enabled"] is False
    verifier_usage = paper["verifier"]["usage"]
    assert verifier_usage["calls_attempted"] == 0
    assert verifier_usage["calls_field_count"] == 0
    assert verifier_usage["resumed_calls_field_count"] == 0
    assert verifier_usage["call_accounting_basis"] == "calls_plus_resumed_calls"
    assert verifier_usage["cost_usd"] == 0.0
    assert verifier_usage["cost_usd_lower_bound"] == 0.0
    assert verifier_usage["input_tokens"] == 0
    assert verifier_usage["reasoning_tokens"] == 0
    assert verifier_usage["models_returned"] == []
    assert verifier_usage["providers_returned"] == []
    sources = json.loads((first / "sources.json").read_text(encoding="utf-8"))
    assert sources["papers"][0]["sources"][0]["sha256"] == "a" * 64
    assert "cache_relpath" not in sources["papers"][0]["sources"][0]

    checksum_lines = (first / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    checksum_paths = [line.split("  ", 1)[1] for line in checksum_lines]
    assert checksum_paths == sorted(expected_files - {"SHA256SUMS"})
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        assert digest == sha256_file(first / relative)


def test_current_model_selection_preserves_valid_contract_ineligible_skip(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    payload = json.loads(model_selection.read_text(encoding="utf-8"))
    payload["declared_models"].append("example/skipped")
    payload["models"].append(
        {
            "model": "example/skipped",
            "label": "Skipped extractor",
            "contract_eligibility": "contract_ineligible",
            "eligibility_reason_codes": ["wire_schema_below_gate"],
            "matched_quality_status": "not_run",
            "not_run_reason": "contract_ineligible_on_sealed_smoke",
            "quality_status": "unmeasured_contract_ineligible",
            "quality": None,
            "aggregate": None,
            "cases": [],
        }
    )
    write_json(model_selection, payload)

    public_root = _build_snapshot(
        (run_root, model_selection, human_review),
        tmp_path / "public",
        snapshot_id="skipped-contract-ineligible-model",
        selected_model="example/extractor",
    )

    selection = json.loads((public_root / "model-selection.json").read_text(encoding="utf-8"))
    skipped = selection["models"][1]
    assert skipped == {
        "model": "example/skipped",
        "label": "Skipped extractor",
        "contract_eligibility": "contract_ineligible",
        "eligibility_reason_codes": ["wire_schema_below_gate"],
        "matched_quality_status": "not_run",
        "not_run_reason": "contract_ineligible_on_sealed_smoke",
        "quality_status": "unmeasured_contract_ineligible",
        "aggregate": None,
    }


def test_current_model_selection_allows_honest_all_ineligible_campaign(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    payload = json.loads(model_selection.read_text(encoding="utf-8"))
    payload["executed_models"] = []
    payload["privacy"]["references_loaded_after_provider_phase_sealed"] = False
    payload["models"] = [
        {
            "model": "example/extractor",
            "label": "Example extractor",
            "contract_eligibility": "contract_ineligible",
            "eligibility_reason_codes": ["wire_schema_below_gate"],
            "matched_quality_status": "not_run",
            "not_run_reason": "contract_ineligible_on_sealed_smoke",
            "quality_status": "unmeasured_contract_ineligible",
            "quality": None,
            "aggregate": None,
            "cases": [],
        }
    ]
    write_json(model_selection, payload)

    public_root = _build_snapshot(
        (run_root, model_selection, human_review),
        tmp_path / "public",
        snapshot_id="all-models-contract-ineligible",
    )

    selection = json.loads((public_root / "model-selection.json").read_text(encoding="utf-8"))
    assert selection["selection"] == {"status": "pending", "selected_model": None}
    assert selection["models"][0]["aggregate"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "latency_order",
        "evidence_count",
        "evidence_duplicate",
        "macro_denominator",
        "claim_scope",
        "gate_contract",
        "observability_denominator",
    ],
)
def test_current_model_score_rejects_impossible_public_claims(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
    mutation: str,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    payload = json.loads(model_selection.read_text(encoding="utf-8"))
    aggregate = payload["models"][0]["aggregate"]
    if mutation == "latency_order":
        aggregate["usage"]["latency_seconds"]["p50"] = 2.0
    elif mutation == "evidence_count":
        aggregate["evidence"]["candidate_text_support"]["supported"] = -1
    elif mutation == "evidence_duplicate":
        aggregate["evidence"]["reference_evidence_supported_accuracy"]["macro"] = 0.5
    elif mutation == "macro_denominator":
        aggregate["quality"]["macro"]["detection"]["defined_cases"]["precision"] = 2
    elif mutation == "claim_scope":
        aggregate["claim_type_classification"]["basis_scope"] = "whole paper"
    elif mutation == "observability_denominator":
        aggregate["quality"]["micro"]["input_observability"]["cases_measured"] = 2
    else:
        aggregate["model_selection_gates"]["claim_type_macro_f1"]["threshold"] = 0.5
    write_json(model_selection, payload)

    with pytest.raises(PublicSnapshotError):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id=f"invalid-model-{mutation}",
        )


def test_selected_model_requires_every_frozen_gate_to_pass(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    payload = json.loads(model_selection.read_text(encoding="utf-8"))
    aggregate = payload["models"][0]["aggregate"]
    aggregate["claim_type_classification"] = _claim_type_classification()
    aggregate["model_selection_gates"]["claim_type_macro_f1"].update(
        {"status": "not_measured", "value": None}
    )
    write_json(model_selection, payload)

    with pytest.raises(PublicSnapshotError, match="does not pass every model-selection gate"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="selected-model-unmeasured-gate",
            selected_model="example/extractor",
        )


def test_selected_model_must_match_every_paper_run_extractor(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    extractor = corpus["runs"][0]["extractor"]
    extractor["model"] = "example/other"
    extractor["calls"][0] = {
        **_public_call(
            model="example/other",
            schema_name="extractor",
            seed=None,
            require_parameters=True,
            max_tokens=16_000,
            temperature=None,
            request_id="DO_NOT_PUBLISH_REQUEST",
        ),
        "request_id": "DO_NOT_PUBLISH_REQUEST",
        "exact_quote": "DO_NOT_PUBLISH_QUOTE",
    }
    _rewrite_corpus_and_paper_run(run_root, corpus)
    _refresh_human_review_audit(run_root, human_review)

    with pytest.raises(PublicSnapshotError, match="disagrees with a published paper-run"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="selected-model-run-mismatch",
            selected_model="example/extractor",
        )


def test_selected_model_requires_exact_returned_model_for_every_completed_call(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    call = corpus["runs"][0]["extractor"]["calls"][0]
    call["model_returned"] = None
    call["model_returned_disposition"] = "unrecognized_omitted"
    _rewrite_corpus_and_paper_run(run_root, corpus)
    _refresh_human_review_audit(run_root, human_review)

    with pytest.raises(PublicSnapshotError, match="exact returned-model evidence"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="selected-model-returned-mismatch",
            selected_model="example/extractor",
        )


@pytest.mark.parametrize(
    ("stage_name", "schema_name", "temperature", "seed", "require_parameters", "max_tokens"),
    [
        ("row_enumeration", "row_extractor", 0.0, 7, False, 1_000),
        ("tuple_resolution", "tuple_resolution", 0.0, 7, True, 2_000),
        ("verifier", "verifier", None, None, True, 2_000),
        ("origin_retrieval", "origin_retrieval", 0.0, 7, True, 2_000),
    ],
)
def test_finalized_paid_stage_requires_exact_returned_model_for_every_call(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
    stage_name: str,
    schema_name: str,
    temperature: float | None,
    seed: int | None,
    require_parameters: bool,
    max_tokens: int,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    stage = corpus["runs"][0][stage_name]
    model = f"example/{stage_name}"
    stage["enabled"] = True
    stage["model"] = model
    stage["request_contract"] = _provider_request_contract(
        schema_name=schema_name,
        seed=seed,
        require_parameters=require_parameters,
        model=model,
        max_tokens=max_tokens,
    )
    call = _public_call(
        model=model,
        schema_name=schema_name,
        seed=seed,
        require_parameters=require_parameters,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    call["model_returned"] = None
    call["model_returned_disposition"] = "unrecognized_omitted"
    stage["calls"] = [call]
    _rewrite_corpus_and_paper_run(run_root, corpus)

    with pytest.raises(
        PublicSnapshotError,
        match=rf"{stage_name} lacks exact returned-model evidence",
    ):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id=f"{stage_name}-returned-model-unverified",
        )


@pytest.mark.parametrize("run_status", ["bounded_incomplete", "error"])
def test_unfinished_run_preserves_unverified_returned_model_lower_bound(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
    run_status: str,
) -> None:
    run_root, _, _, _ = _private_inputs(tmp_path / "inputs", manifest, eligible_candidate)
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    run = corpus["runs"][0]
    run["status"] = run_status
    call = run["extractor"]["calls"][0]
    call["model_returned"] = None
    call["model_returned_disposition"] = "unrecognized_omitted"

    projected = _project_paper_run(run, 0)

    usage = projected["extractor"]["usage"]
    assert usage["calls_attempted"] == 1
    assert usage["model_returned_matches_requested_calls"] == 0
    assert usage["model_returned_unverified_calls"] == 1


def test_selected_model_must_match_production_confidence_policy(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    corpus["runs"][0]["candidate_validation"]["min_confidence"] = 0.7
    _rewrite_corpus_and_paper_run(run_root, corpus)
    _refresh_human_review_audit(run_root, human_review)

    with pytest.raises(PublicSnapshotError, match="confidence policy disagrees"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="selected-model-confidence-mismatch",
            selected_model="example/extractor",
        )


def test_current_model_selection_requires_execution_selection_provenance(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    payload = json.loads(model_selection.read_text(encoding="utf-8"))
    payload["execution_selection_sha256"] = None
    write_json(model_selection, payload)

    with pytest.raises(PublicSnapshotError, match="execution_selection_sha256"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="missing-execution-selection",
        )


@pytest.mark.parametrize("mutation", ["unmeasured_quality", "dropped_predeclared_call"])
def test_current_model_aggregate_rejects_denominator_and_measurement_contradictions(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
    mutation: str,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    payload = json.loads(model_selection.read_text(encoding="utf-8"))
    aggregate = payload["models"][0]["aggregate"]
    if mutation == "unmeasured_quality":
        aggregate["quality_measurement"].update(
            {"status": "unmeasured", "scored_repetitions": 0, "unmeasured_repetitions": 1}
        )
        aggregate["execution"]["repetitions_scored"] = 0
        aggregate["execution"]["cases_scored"] = 0
    else:
        aggregate["execution"]["calls_predeclared"] = 2
    write_json(model_selection, payload)

    with pytest.raises(PublicSnapshotError):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id=f"model-aggregate-{mutation}",
        )


def test_human_review_audit_id_binds_exact_run_artifacts(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    corpus["runs"][0]["wall_clock_seconds"] = 4.0
    _rewrite_corpus_and_paper_run(run_root, corpus)

    with pytest.raises(PublicSnapshotError, match="audit_id disagrees with current run artifacts"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="stale-human-review-audit",
        )


def test_invalid_eee_aborts_without_publishing(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, eee_path = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    invalid = json.loads(eee_path.read_text(encoding="utf-8"))
    invalid["schema_version"] = "wrong"
    write_json(eee_path, invalid)

    with pytest.raises(PublicSnapshotError, match="EEE validation failed"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="invalid-eee",
        )

    assert not (tmp_path / "public" / "invalid-eee").exists()


def test_paper_eee_schema_must_match_pinned_snapshot_schema(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    corpus["runs"][0]["eee_schema"] = {
        "version": "0.2.1",
        "sha256": "0" * 64,
    }
    _rewrite_corpus_and_paper_run(run_root, corpus)
    _refresh_human_review_audit(run_root, human_review)

    with pytest.raises(PublicSnapshotError, match="pinned snapshot schema"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="mismatched-paper-eee-schema",
        )


def test_public_snapshot_preserves_extractor_observability_and_control_examination(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    input_observability = {
        "status": "measured",
        "cases_measured": 1,
        "cases_not_assessed": 0,
        "measured_reference_observations": 21,
        "reference_observations": 21,
        "observable_reference_observations": 17,
        "unobservable_reference_observations": 4,
        "observation_coverage": 17 / 21,
        "model_conditional_detection": {
            "true_positives": 15,
            "false_negatives": 2,
            "recall_basis": 17,
            "recall": 15 / 17,
            "reference_ids": ["DO_NOT_PUBLISH_REFERENCE_ID"],
        },
        "reference_ids": ["DO_NOT_PUBLISH_REFERENCE_ID"],
    }
    reference_matches = [
        {
            "reference_id": f"reference-{index}",
            "observation_id": f"observation-{index}" if index < 15 else None,
            "expected_claim_type": "primary_result",
            "actual_claim_type": "primary_result" if index < 15 else None,
            **{field: index < 15 for field in FIELD_NAMES},
        }
        for index in range(21)
    ]
    claim_classification = _claim_type_classification()
    claim_classification.update({"basis": 15})
    claim_classification["per_class"]["primary_result"].update(
        {"support": 15, "predicted": 15, "true_positives": 15}
    )
    negative_control_safety = {
        "controls_total": 21,
        "controls_matched": 0,
        "control_match_coverage": 0.0,
        "control_match_coverage_defined": True,
        "controls_examined": 17,
        "controls_not_examined": 4,
        "control_examination_coverage": 17 / 21,
        "passed_by_abstention_count": 17,
        "control_trials": {
            "total": 42,
            "matched": 0,
            "examined": 34,
            "passed_by_abstention": 34,
            "control_ids": ["DO_NOT_PUBLISH_CONTROL_ID"],
        },
        "measurement_status": "partially_measured",
        "matched_candidates": 0,
        "false_primary_count": 0,
        "false_primary_export_count": 0,
        "false_primary_rate": 0.0,
        "zero_false_primary_gate_passed": None,
        "zero_false_primary_export_gate_passed": None,
        "control_ids": ["DO_NOT_PUBLISH_CONTROL_ID"],
    }

    selection_payload = json.loads(model_selection.read_text(encoding="utf-8"))
    aggregate = selection_payload["models"][0]["aggregate"]
    aggregate["negative_control_safety"] = negative_control_safety
    aggregate["claim_type_classification"] = claim_classification
    for gate_name in (
        "claim_type_macro_f1",
        "false_primary_controls",
        "false_primary_exports",
    ):
        aggregate["model_selection_gates"][gate_name].update(
            {"status": "not_measured", "value": None}
        )
    for field_group in (
        aggregate["quality"]["macro"]["field_accuracy"],
        aggregate["quality"]["micro"]["field_accuracy"],
    ):
        field_group.update(dict.fromkeys(FIELD_NAMES, 15 / 21))
    aggregate["evidence"]["reference_evidence_supported_accuracy"].update(
        {"macro": 15 / 21, "micro": 15 / 21}
    )
    aggregate["evidence"]["reference_page_anchor_accuracy"].update(
        {"macro": 15 / 21, "micro": 15 / 21}
    )
    aggregate["quality"]["micro"].update(
        {
            "reference_observations": 21,
            "detection": {
                "true_positives": 15,
                "precision_true_positives": 15,
                "precision_basis": 15,
                "recall_basis": 21,
                "false_positives": 0,
                "false_negatives": 6,
                "precision": 1.0,
                "precision_defined": True,
                "recall": 15 / 21,
                "f1": 5 / 6,
            },
            "input_observability": input_observability,
        }
    )
    write_json(model_selection, selection_payload)

    paper_score = {
        "schema_version": "reference-score/0.7",
        "paper_id": manifest.paper_id,
        "reference_observations": 21,
        "recall_basis": 21,
        "field_matching_basis": 21,
        "candidate_primary_results": 15,
        "primary_candidates_total": 15,
        "primary_candidates_in_coverage": 15,
        "primary_candidates_out_of_coverage": 0,
        "precision_basis": 15,
        "unmatched_candidate_ids": [],
        "unmatched_candidate_ids_in_coverage": [],
        "unmatched_candidate_ids_out_of_coverage": [],
        "unmatched_primary_candidate_ids_in_coverage": [],
        "unmatched_primary_candidate_ids_out_of_coverage": [],
        "coverage": {
            "recall_scope": "annotated_reference_observations",
            "field_matching_scope": "annotated_reference_observations",
            "precision_scope": "fully_annotated_labels",
            "fully_annotated_labels": [],
            "sampled_labels": [],
        },
        "detection": {
            "true_positives": 15,
            "precision_true_positives": 15,
            "precision_basis": 15,
            "recall_basis": 21,
            "false_positives": 0,
            "false_negatives": 6,
            "precision": 1.0,
            "precision_defined": True,
            "recall": 15 / 21,
            "f1": 5 / 6,
        },
        "input_observability": input_observability,
        "field_accuracy": dict.fromkeys(FIELD_NAMES, 15 / 21),
        "matches": reference_matches,
        "claim_type_classification": claim_classification,
        "negative_control_safety": {
            "controls_total": 21,
            "matched_control_count": 0,
            "matched_candidate_count": 0,
            "control_match_coverage": 0.0,
            "control_match_coverage_defined": True,
            "control_examination_coverage": 17 / 21,
            "measurement_status": "partially_measured",
            "passed_by_abstention_count": 17,
            "false_primary_count": 0,
            "false_primary_export_count": 0,
            "false_primary_rate": 0.0,
            "false_primary_rate_basis": 0,
            "false_primary_rate_defined": False,
            "zero_false_primary_gate_passed": None,
            "zero_false_primary_export_gate_passed": None,
            "control_ids": [f"control-{index}" for index in range(21)],
            "matched_control_ids": [],
            "unmatched_control_ids": [f"control-{index}" for index in range(21)],
            "control_status": {
                f"control-{index}": ("passed_by_abstention" if index < 17 else "not_examined")
                for index in range(21)
            },
            "examined_control_ids": [f"control-{index}" for index in range(17)],
            "not_examined_control_ids": [f"control-{index}" for index in range(17, 21)],
            "matched_candidate_ids": [],
            "false_primary_candidate_ids": [],
            "false_primary_export_candidate_ids": [],
            "matches": [],
        },
    }
    paper_root = run_root / manifest.paper_id
    score_path = paper_root / "reference-score.json"
    write_json(score_path, paper_score)
    corpus_evaluation = aggregate_reference_scores([paper_score])
    write_json(run_root / "corpus-evaluation.json", corpus_evaluation)
    corpus_path = run_root / "corpus-run.json"
    corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    run_payload = corpus_payload["runs"][0]
    run_payload["reference_evaluation"] = {
        "path": "private-reference.json",
        "score_path": "reference-score.json",
        "score_sha256": sha256_file(score_path),
        "schema_version": paper_score["schema_version"],
        "coverage": paper_score["coverage"],
        "detection": paper_score["detection"],
        "field_accuracy": paper_score["field_accuracy"],
        "negative_control_safety": paper_score["negative_control_safety"],
    }
    for counts in (run_payload["counts"], corpus_payload["totals"]):
        counts["reference_observations"] = 21
        counts["reference_true_positives"] = 15
        counts["reference_false_positives"] = 0
        counts["reference_false_negatives"] = 6
    corpus_payload["reference_evaluation"] = corpus_evaluation
    write_json(corpus_path, corpus_payload)
    write_json(paper_root / "run.json", run_payload)
    audit_path = run_root / "reference-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(
        {
            "papers_passed": 1,
            "papers_skipped": 0,
            "text_verified": 1,
        }
    )
    audit["results"][0].update(
        {
            "status": "passed",
            "page_count": 10,
            "source_hash_matches": True,
            "text_verified": 1,
            "visual_verified": 0,
            "failed_evidence": 0,
        }
    )
    write_json(audit_path, audit)
    _refresh_human_review_audit(run_root, human_review)

    public_root = _build_snapshot(
        (run_root, model_selection, human_review),
        tmp_path / "public",
        snapshot_id="extractor-diagnostics",
    )

    public_selection = json.loads(
        (public_root / "model-selection.json").read_text(encoding="utf-8")
    )
    public_aggregate = public_selection["models"][0]["aggregate"]
    public_snapshot = json.loads((public_root / "snapshot.json").read_text(encoding="utf-8"))
    public_reference = public_snapshot["corpora"][0]["reference_evaluation"]

    expected_safety = {
        key: value for key, value in negative_control_safety.items() if key != "control_ids"
    }
    expected_safety["control_trials"] = {
        key: value
        for key, value in negative_control_safety["control_trials"].items()
        if key != "control_ids"
    }
    expected_observability = {
        key: value for key, value in input_observability.items() if key != "reference_ids"
    }
    expected_observability["model_conditional_detection"] = {
        key: value
        for key, value in input_observability["model_conditional_detection"].items()
        if key != "reference_ids"
    }
    assert public_aggregate["negative_control_safety"] == expected_safety
    assert public_aggregate["quality"]["micro"]["input_observability"] == (expected_observability)
    assert public_reference["schema_version"] == "corpus-reference-score/0.4"
    assert public_reference["input_observability"] == corpus_evaluation["input_observability"]
    serialized = json.dumps([public_aggregate, public_reference], sort_keys=True)
    assert "DO_NOT_PUBLISH" not in serialized
    assert "control_ids" not in serialized
    assert "reference_ids" not in serialized

    paper_score["field_accuracy"]["metric"] = 1.0
    write_json(score_path, paper_score)
    tampered_corpus_evaluation = aggregate_reference_scores([paper_score])
    write_json(run_root / "corpus-evaluation.json", tampered_corpus_evaluation)
    run_payload["reference_evaluation"]["score_sha256"] = sha256_file(score_path)
    run_payload["reference_evaluation"]["field_accuracy"] = paper_score["field_accuracy"]
    corpus_payload["reference_evaluation"] = tampered_corpus_evaluation
    corpus_payload["runs"][0] = run_payload
    write_json(corpus_path, corpus_payload)
    write_json(paper_root / "run.json", run_payload)
    _refresh_human_review_audit(run_root, human_review)
    with pytest.raises(PublicSnapshotError, match="disagrees with detailed matches"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="tampered-reference-detail-summary",
        )

    paper_score["field_accuracy"]["metric"] = 15 / 21
    negative = paper_score["negative_control_safety"]
    negative["matches"] = [
        {
            "control_id": "control-0",
            "observation_id": "negative-primary",
            "expected_claim_type": "method_metadata",
            "actual_claim_type": "primary_result",
            "claim_type_matches": False,
            "false_primary": True,
            "export_status": "exported",
            "false_primary_export": True,
            "matched_evidence_ids": ["negative-evidence"],
        }
    ]
    negative.update(
        {
            "matched_control_ids": ["control-0"],
            "matched_candidate_ids": ["negative-primary"],
            "matched_control_count": 1,
            "matched_candidate_count": 1,
            "false_primary_rate_basis": 1,
            "false_primary_rate_defined": True,
            "false_primary_rate": 0.0,
        }
    )
    paper_score["claim_type_classification"] = score_claim_type_pairs(
        [("primary_result", "primary_result")] * 15 + [("method_metadata", "primary_result")]
    )
    write_json(score_path, paper_score)
    tampered_corpus_evaluation = aggregate_reference_scores([paper_score])
    write_json(run_root / "corpus-evaluation.json", tampered_corpus_evaluation)
    run_payload["reference_evaluation"]["score_sha256"] = sha256_file(score_path)
    run_payload["reference_evaluation"]["field_accuracy"] = paper_score["field_accuracy"]
    run_payload["reference_evaluation"]["negative_control_safety"] = negative
    corpus_payload["reference_evaluation"] = tampered_corpus_evaluation
    corpus_payload["runs"][0] = run_payload
    write_json(corpus_path, corpus_payload)
    write_json(paper_root / "run.json", run_payload)
    _refresh_human_review_audit(run_root, human_review)
    with pytest.raises(PublicSnapshotError, match="false_primary_candidate_ids disagrees"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="tampered-negative-control-summary",
        )


def test_reference_score_detail_fixture_is_internally_consistent() -> None:
    _validate_reference_score_details(_reference_score_detail_fixture(), "paper")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("empty_reference", "reference_id is invalid"),
        ("duplicate_reference", "reuses a reference ID"),
        ("missing_observation_only", "observation_id and actual_claim_type disagree"),
        ("unmatched_true_field", "cannot match without an observation"),
    ],
)
def test_reference_positive_details_reject_incoherent_matches(mutation: str, message: str) -> None:
    score = _reference_score_detail_fixture()
    matches = score["matches"]
    first = matches[0]
    second = matches[1]
    if mutation == "empty_reference":
        first["reference_id"] = ""
    elif mutation == "duplicate_reference":
        second["reference_id"] = first["reference_id"]
    elif mutation == "missing_observation_only":
        first["observation_id"] = None
    else:
        first["observation_id"] = None
        first["actual_claim_type"] = None

    with pytest.raises(PublicSnapshotError, match=message):
        _validate_reference_score_details(score, "paper")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("matched_overlap", "matched and unmatched candidate IDs overlap"),
        ("in_out_overlap", "unmatched candidate partitions disagree"),
        ("primary_not_subset", "unmatched candidate partitions disagree"),
    ],
)
def test_reference_unmatched_candidate_partitions_are_exact(mutation: str, message: str) -> None:
    score = _reference_score_detail_fixture()
    if mutation == "matched_overlap":
        score["unmatched_candidate_ids"] = ["observation-0"]
        score["unmatched_candidate_ids_in_coverage"] = ["observation-0"]
        score["unmatched_primary_candidate_ids_in_coverage"] = ["observation-0"]
    elif mutation == "in_out_overlap":
        score["unmatched_candidate_ids"] = ["unmatched"]
        score["unmatched_candidate_ids_in_coverage"] = ["unmatched"]
        score["unmatched_candidate_ids_out_of_coverage"] = ["unmatched"]
    else:
        score["unmatched_primary_candidate_ids_in_coverage"] = ["unmatched"]

    with pytest.raises(PublicSnapshotError, match=message):
        _validate_reference_score_details(score, "paper")


@pytest.mark.parametrize("mutation", ["inflate_matched_primary", "inflate_primary_out"])
def test_reference_primary_candidate_counts_bind_to_detailed_ids(mutation: str) -> None:
    score = _reference_score_detail_fixture()
    score["candidate_primary_results"] = 3
    score["primary_candidates_total"] = 3
    if mutation == "inflate_matched_primary":
        score["primary_candidates_in_coverage"] = 3
        score["precision_basis"] = 3
        detection = score["detection"]
        detection["precision_true_positives"] = 3
    else:
        score["primary_candidates_out_of_coverage"] = 1

    with pytest.raises(PublicSnapshotError, match="precision summary disagrees with detailed IDs"):
        _validate_reference_score_details(score, "paper")


def test_reference_observability_cannot_exceed_detailed_outcomes() -> None:
    score = _reference_score_detail_fixture()
    observability = score["input_observability"]
    observability.update(
        {
            "status": "measured",
            "observable_reference_observations": 2,
            "unobservable_reference_observations": 0,
            "observation_coverage": 1.0,
        }
    )
    conditional = observability["model_conditional_detection"]
    conditional["true_positives"] = 1
    conditional["false_negatives"] = 1
    conditional["recall_basis"] = conditional["true_positives"] + conditional["false_negatives"]
    conditional["recall"] = conditional["true_positives"] / conditional["recall_basis"]

    with pytest.raises(PublicSnapshotError, match="detection exceeds detailed outcomes"):
        _validate_reference_score_details(score, "paper")


def test_reference_negative_details_reject_duplicate_control_candidate_pairs() -> None:
    score = _reference_score_detail_fixture()
    match = {
        "control_id": "control-1",
        "observation_id": "negative-1",
        "expected_claim_type": "method_metadata",
        "actual_claim_type": "method_metadata",
        "claim_type_matches": True,
        "false_primary": False,
        "export_status": "not_eligible",
        "false_primary_export": False,
        "matched_evidence_ids": ["evidence-1"],
    }
    negative = score["negative_control_safety"]
    negative["matches"] = [match, dict(match)]

    with pytest.raises(PublicSnapshotError, match="contains a duplicate pair"):
        _validate_reference_score_details(score, "paper")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown_examined_id", "control ID partitions disagree"),
        ("duplicate_control_id", "control_ids is invalid"),
        ("status_disagrees", "control_status disagrees with ID lists"),
        ("inflated_coverage", "control_examination_coverage disagrees with IDs"),
    ],
)
def test_reference_negative_control_frame_is_exact(mutation: str, message: str) -> None:
    score = _reference_score_detail_fixture()
    negative = score["negative_control_safety"]
    negative.update(
        {
            "controls_total": 1,
            "control_ids": ["control-1"],
            "unmatched_control_ids": ["control-1"],
            "control_status": {"control-1": "passed_by_abstention"},
            "examined_control_ids": ["control-1"],
            "not_examined_control_ids": [],
            "control_match_coverage": 0.0,
            "control_match_coverage_defined": True,
            "control_examination_coverage": 1.0,
            "passed_by_abstention_count": 1,
            "measurement_status": "measured",
            "zero_false_primary_gate_passed": True,
            "zero_false_primary_export_gate_passed": True,
        }
    )
    if mutation == "unknown_examined_id":
        negative["examined_control_ids"] = ["fabricated-control"]
    elif mutation == "duplicate_control_id":
        negative["controls_total"] = 2
        negative["control_ids"] = ["control-1", "control-1"]
    elif mutation == "status_disagrees":
        negative["control_status"] = {"control-1": "not_examined"}
    else:
        negative["control_examination_coverage"] = 0.5

    with pytest.raises(PublicSnapshotError, match=message):
        _validate_reference_score_details(score, "paper")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"controls_examined": "17"}, "controls_examined must be an integer"),
        ({"control_examination_coverage": 1.2}, "must be between 0 and 1"),
        ({"measurement_status": "unknown"}, "measurement_status is unsupported"),
        ({"measurement_status": []}, "measurement_status is unsupported"),
        ({"control_trials": []}, "control_trials must be a JSON object"),
        (
            {"control_trials": {"examined": True}},
            "control_trials.examined must be an integer",
        ),
    ],
)
def test_negative_control_diagnostic_projection_rejects_malformed_types(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(PublicSnapshotError, match=message):
        _project_negative_safety(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"controls_total": 1, "controls_examined": 2},
            "examined count exceeds controls_total",
        ),
        (
            {
                "controls_total": 2,
                "controls_matched": 0,
                "controls_examined": 1,
                "controls_not_examined": 1,
                "passed_by_abstention_count": 0,
            },
            "examined outcomes are inconsistent",
        ),
        (
            {
                "controls_total": 2,
                "controls_examined": 1,
                "controls_not_examined": 1,
                "measurement_status": "measured",
            },
            "measurement_status is inconsistent",
        ),
        (
            {
                "controls_total": 2,
                "controls_examined": 1,
                "control_examination_coverage": 1.0,
            },
            "control_examination_coverage is inconsistent",
        ),
        (
            {
                "control_trials": {
                    "total": 4,
                    "matched": 1,
                    "examined": 3,
                    "passed_by_abstention": 1,
                }
            },
            "control_trials outcomes are inconsistent",
        ),
    ],
)
def test_negative_control_projection_rejects_inconsistent_algebra(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(PublicSnapshotError, match=message):
        _project_negative_safety(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"measurement_status": "measured"},
            "measurement_status requires reconstructable fields",
        ),
        (
            {"zero_false_primary_gate_passed": True},
            "zero_false_primary_gate_passed requires reconstructable fields",
        ),
        (
            {"false_primary_rate": 0.0, "false_primary_count": 0},
            "false_primary_rate requires reconstructable matched-candidate count",
        ),
        (
            {"control_trials": {"total": 1}},
            "control_trials requires reconstructable fields",
        ),
    ],
)
def test_negative_control_claims_require_complete_reconstructable_basis(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(PublicSnapshotError, match=message):
        _project_negative_safety(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"observable_reference_observations": True}, "must be an integer"),
        ({"observation_coverage": -0.1}, "must be between 0 and 1"),
        ({"status": "unknown"}, "status is unsupported"),
        ({"status": []}, "status is unsupported"),
        ({"model_conditional_detection": []}, "must be a JSON object"),
        (
            {"model_conditional_detection": {"recall": "0.9"}},
            "must be a finite number",
        ),
    ],
)
def test_input_observability_projection_rejects_malformed_types(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(PublicSnapshotError, match=message):
        _project_input_observability(payload, "input_observability")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "status": "measured",
                "reference_observations": 3,
                "observable_reference_observations": 2,
                "unobservable_reference_observations": 2,
                "observation_coverage": 2 / 3,
                "model_conditional_detection": {
                    "true_positives": 1,
                    "false_negatives": 1,
                    "recall_basis": 2,
                    "recall": 0.5,
                },
            },
            "observation partition is inconsistent",
        ),
        (
            {
                "status": "measured",
                "reference_observations": 3,
                "observable_reference_observations": 2,
                "unobservable_reference_observations": 1,
                "observation_coverage": 2 / 3,
                "model_conditional_detection": {
                    "true_positives": 2,
                    "false_negatives": 1,
                    "recall_basis": 2,
                    "recall": 1.0,
                },
            },
            "model-conditional detection is inconsistent",
        ),
    ],
)
def test_input_observability_projection_rejects_inconsistent_algebra(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(PublicSnapshotError, match=message):
        _project_input_observability(payload, "input_observability")


def test_not_assessed_observability_preserves_known_reference_denominator() -> None:
    projected = _project_input_observability(
        {
            "status": "not_assessed",
            "reference_observations": 3,
            "observable_reference_observations": None,
            "unobservable_reference_observations": None,
            "observation_coverage": None,
            "model_conditional_detection": {
                "true_positives": None,
                "false_negatives": None,
                "recall_basis": None,
                "recall": None,
            },
        },
        "input_observability",
    )

    assert projected["reference_observations"] == 3
    assert projected["observation_coverage"] is None


def test_partially_assessed_observability_is_bounded_and_nulls_global_claims() -> None:
    projected = _project_input_observability(
        {
            "status": "partially_assessed",
            "papers_measured": 2,
            "papers_not_assessed": 1,
            "measured_reference_observations": 8,
            "reference_observations": None,
            "observable_reference_observations": None,
            "unobservable_reference_observations": None,
            "observation_coverage": None,
            "model_conditional_detection": {
                "true_positives": None,
                "false_negatives": None,
                "recall_basis": None,
                "recall": None,
            },
        },
        "input_observability",
    )

    assert projected["status"] == "partially_assessed"
    assert projected["papers_measured"] == 2
    assert projected["papers_not_assessed"] == 1
    assert projected["observation_coverage"] is None
    assert all(value is None for value in projected["model_conditional_detection"].values())


def test_partially_assessed_observability_rejects_pooled_global_claims() -> None:
    with pytest.raises(PublicSnapshotError, match="aggregate fields must be null"):
        _project_input_observability(
            {
                "status": "partially_assessed",
                "papers_measured": 1,
                "papers_not_assessed": 1,
                "measured_reference_observations": 3,
                "reference_observations": 3,
                "observable_reference_observations": None,
                "unobservable_reference_observations": None,
                "observation_coverage": None,
                "model_conditional_detection": {
                    "true_positives": None,
                    "false_negatives": None,
                    "recall_basis": None,
                    "recall": None,
                },
            },
            "input_observability",
        )


def test_partially_assessed_case_observability_preserves_case_denominators() -> None:
    projected = _project_input_observability(
        {
            "status": "partially_assessed",
            "cases_measured": 2,
            "cases_not_assessed": 1,
            "measured_reference_observations": 7,
            "reference_observations": None,
            "observable_reference_observations": None,
            "unobservable_reference_observations": None,
            "observation_coverage": None,
            "model_conditional_detection": {
                "true_positives": None,
                "false_negatives": None,
                "recall_basis": None,
                "recall": None,
            },
        },
        "input_observability",
    )

    assert projected["cases_measured"] == 2
    assert projected["cases_not_assessed"] == 1
    assert "papers_measured" not in projected


def test_reference_score_detection_rejects_contradictory_rates() -> None:
    score = {
        "schema_version": "reference-score/0.7",
        "reference_observations": 1,
        "detection": {
            "true_positives": 1,
            "precision_true_positives": 1,
            "precision_basis": 1,
            "recall_basis": 1,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "precision_defined": True,
            "recall": 0.1,
            "f1": 1.0,
        },
        "field_accuracy": dict.fromkeys(FIELD_NAMES, 1.0),
        "claim_type_classification": _claim_type_classification(),
    }

    with pytest.raises(PublicSnapshotError, match="recall is inconsistent"):
        _project_reference_evaluation(score)


def test_corpus_reference_observability_binds_paper_denominator() -> None:
    aggregate = aggregate_reference_scores([_reference_score_detail_fixture()])
    aggregate["input_observability"]["papers_not_assessed"] = 2

    with pytest.raises(PublicSnapshotError, match="paper denominator disagrees"):
        _project_reference_evaluation(aggregate)


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        ({"calls": "not-a-list"}, "calls must be a JSON array"),
        ({"calls": ["not-an-object"]}, r"calls\[0\] must be a JSON object"),
    ],
)
def test_provider_call_aggregation_rejects_malformed_or_untrusted_values(
    stage: dict[str, object], message: str
) -> None:
    with pytest.raises(PublicSnapshotError, match=message):
        _aggregate_private_calls(stage)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cost_usd", -0.1, "not a typed ProviderCall"),
        ("cost_usd", float("nan"), "not a typed ProviderCall"),
        ("input_tokens", -1, "not a typed ProviderCall"),
        ("output_tokens", True, "not a typed ProviderCall"),
        ("model_returned", "private/mismatch", "model_returned is inconsistent"),
        ("provider_returned", "private-provider", "provider_returned is inconsistent"),
        ("finish_reason", "private-finish", "finish_reason is not public telemetry"),
    ],
)
def test_provider_call_aggregation_rejects_tampered_current_call(
    field: str, value: object, message: str
) -> None:
    call = _public_call(
        model="requested/model",
        schema_name="extractor",
        seed=7,
        require_parameters=False,
        max_tokens=100,
    )
    call[field] = value
    stage = {
        "model": "requested/model",
        "temperature": 0.0,
        "reasoning_effort": "minimal",
        "max_tokens": 100,
        "seed": 7,
        "require_parameters": False,
        "prompt_sha256": "b" * 64,
        "calls": [call],
    }

    with pytest.raises(PublicSnapshotError, match=message):
        _aggregate_private_calls(stage)


def test_provider_call_aggregation_counts_resumed_calls_without_ambiguity() -> None:
    request_contract = _provider_request_contract(
        schema_name="extractor",
        seed=7,
        require_parameters=False,
        model="requested/model",
        max_tokens=100,
    )
    stage = {
        "model": "requested/model",
        "temperature": 0.0,
        "reasoning_effort": "minimal",
        "max_tokens": 100,
        "seed": 7,
        "require_parameters": False,
        "prompt_sha256": "b" * 64,
        "request_contract": request_contract,
        "calls": [
            _public_call(
                model="requested/model",
                schema_name="extractor",
                seed=7,
                require_parameters=False,
                max_tokens=100,
                input_tokens=3,
                output_tokens=2,
                reasoning_tokens=1,
                total_tokens=5,
                cost_usd=0.1,
            )
        ],
        "resumed_calls": [
            _public_call(
                model="requested/model",
                schema_name="extractor",
                seed=7,
                require_parameters=False,
                max_tokens=100,
                input_tokens=4,
                output_tokens=1,
                reasoning_tokens=None,
                total_tokens=5,
                cost_usd=0.2,
            )
        ],
    }

    usage = _aggregate_private_calls(stage, expected_request_contract=request_contract)

    assert usage["calls_attempted"] == 2
    assert usage["calls_field_count"] == 1
    assert usage["resumed_calls_field_count"] == 1
    assert usage["cost_usd"] == 0.3
    assert usage["input_tokens"] == 7
    assert usage["reasoning_tokens"] is None
    assert usage["reasoning_tokens_lower_bound"] == 1
    assert usage["reasoning_tokens_missing_calls"] == 1


def test_provider_call_aggregation_accepts_safely_omitted_unrecognized_response_labels() -> None:
    model = "requested/model"
    request_contract = _provider_request_contract(
        schema_name="extractor",
        seed=7,
        require_parameters=False,
        model=model,
        max_tokens=100,
    )
    call = public_provider_call(
        ProviderCall(
            model_requested=model,
            model_returned="untrusted/model\nprivate-value",
            provider_returned="untrusted-provider/private-value",
            prompt_sha256="f" * 64,
            response_sha256="e" * 64,
            temperature=0.0,
            reasoning_effort="minimal",
            max_tokens=100,
            completion_token_parameter="max_tokens",
            seed=7,
            schema_name="extractor",
            schema_sha256="a" * 64,
            require_parameters=False,
            latency_seconds=0.1,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            cost_usd=0.01,
            finish_reason="stop",
            attempts=1,
        )
    )
    stage = {
        "model": model,
        "temperature": 0.0,
        "reasoning_effort": "minimal",
        "max_tokens": 100,
        "seed": 7,
        "require_parameters": False,
        "prompt_sha256": "b" * 64,
        "request_contract": request_contract,
        "calls": [call],
    }

    usage = _aggregate_private_calls(stage, expected_request_contract=request_contract)

    assert usage["models_returned"] == []
    assert usage["providers_returned"] == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("schema_version", "unknown/9"), "schema_version is unsupported"),
        (("max_tokens", 99), "max_tokens disagrees"),
        (("completion_token_parameter", "max_tokens"), "disagrees with outer stage model"),
        (("seed", 8), "seed disagrees"),
    ],
)
def test_materialized_request_contract_is_bound_to_outer_stage(
    mutation: tuple[str, object], message: str
) -> None:
    contract = _provider_request_contract(
        schema_name="extractor",
        seed=7,
        require_parameters=True,
        model="openai/gpt-5.5",
        max_tokens=100,
    )
    contract[mutation[0]] = mutation[1]
    outer = {
        "model": "openai/gpt-5.5",
        "max_tokens": 100,
        "seed": 7,
        "require_parameters": True,
    }

    with pytest.raises(PublicSnapshotError, match=message):
        _project_request_contract(contract, outer_stage=outer)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_privacy", "request_contract.privacy"),
        ("routing_drift", "require_parameters disagrees"),
    ],
)
def test_request_contract_rejects_incomplete_nested_shape_and_routing_drift(
    mutation: str, message: str
) -> None:
    outer = {"model": "example/model", "max_tokens": 100, "seed": 7, "require_parameters": True}
    contract = _provider_request_contract(schema_name="extractor", **outer)
    if mutation == "missing_privacy":
        del contract["privacy"]
    else:
        contract["routing"]["require_parameters"] = False

    with pytest.raises(PublicSnapshotError, match=message):
        _project_request_contract(contract, outer_stage=outer)


def test_public_report_cost_includes_every_paid_stage() -> None:
    stages = (
        "extractor",
        "row_enumeration",
        "tuple_resolution",
        "verifier",
        "origin_retrieval",
    )
    paper: dict[str, object] = {
        "paper_id": "paper",
        "title": "Paper",
        "status": "success",
        "counts": {},
        "wall_clock_seconds": 1.0,
    }
    for index, stage in enumerate(stages, start=1):
        paper[stage] = {
            "usage": {
                "cost_usd": float(index),
                "cost_usd_lower_bound": float(index),
                "calls_attempted": index,
                "total_tokens_lower_bound": index * 100,
                "retries_lower_bound": index - 1,
            }
        }

    report = _public_report_input(
        {"corpus_id": "corpus", "generated_at": None, "papers_detail": [paper]}
    )

    run = report["runs"][0]
    assert run["cost_usd"] == 15.0
    for index, stage in enumerate(stages, start=1):
        assert run[stage]["completed_call_telemetry"] == {
            "calls": index,
            "total_tokens_lower_bound": index * 100,
            "retries_lower_bound": index - 1,
        }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-provenance", "lacks evidence anchors"),
        ("quote-leak", "embeds an evidence quote"),
    ],
)
def test_eee_provenance_and_quote_failures_abort_without_publishing(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
    mutation: str,
    message: str,
) -> None:
    run_root, model_selection, human_review, eee_path = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    record = json.loads(eee_path.read_text(encoding="utf-8"))
    details = record["evaluation_results"][0]["score_details"]["details"]
    if mutation == "missing-provenance":
        details.pop("evidence_anchor_count")
    else:
        details["evidence_1_quote_text"] = "PRIVATE VERBATIM EVIDENCE"
    write_json(eee_path, record)

    with pytest.raises(PublicSnapshotError, match=message):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id=mutation,
        )

    assert not list((tmp_path / "public").glob("*"))


def test_schema_valid_nested_private_annotation_is_not_public_composer_content(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, eee_path = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    record = json.loads(eee_path.read_text(encoding="utf-8"))
    record["model_info"]["additional_details"]["reviewer_comment"] = (
        "PRIVATE ANNOTATION THAT IS VALID UNDER THE EEE SCHEMA"
    )
    schema, _ = load_schema(SCHEMA, SCHEMA_SHA256)
    assert validate_eee_record(record, schema) == []
    write_json(eee_path, record)

    with pytest.raises(PublicSnapshotError, match="unsupported composer fields"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="nested-private-annotation",
        )

    assert not (tmp_path / "public" / "nested-private-annotation").exists()


def test_current_tuple_gated_composer_shape_reprojects_exactly(
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    observation_id = eligible_candidate.observation_id or eligible_candidate.stable_id()
    provenance = tuple_gated_export_provenance(
        [eligible_candidate],
        tuple_sidecar_sha256="1" * 64,
        tuple_gates={observation_id: "2" * 64},
        verifier_sidecar_sha256="3" * 64,
        verifier_gates={observation_id: "4" * 64},
    )
    records = compose_eee_records(
        manifest=manifest,
        candidates=[eligible_candidate],
        schema_version="0.2.2",
        provenance=provenance,
    )
    assert len(records) == 1

    assert (
        _project_composer_eee_record(
            records[0], paper_id=manifest.paper_id, context="production EEE"
        )
        == records[0]
    )


@pytest.mark.parametrize(
    ("target", "unsafe_uri"),
    [
        ("resolved_uri", "file:///" + "Users/example/paper.pdf"),
        ("resolved_uri", "https://user:secret@example.org/paper.pdf"),
        ("resolved_uri", "https://example.org/paper.pdf?token=secret"),
        ("proceedings_url", "https://example.org/proceedings?token=secret"),
    ],
)
def test_source_urls_reject_local_paths_credentials_and_queries(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
    target: str,
    unsafe_uri: str,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    manifest_path = run_root / manifest.paper_id / "source-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if target == "proceedings_url":
        payload[target] = unsafe_uri
    else:
        payload["sources"][0][target] = unsafe_uri
    write_json(manifest_path, payload)
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    corpus["runs"][0]["source_manifest_sha256"] = sha256_file(manifest_path)
    _rewrite_corpus_and_paper_run(run_root, corpus)
    _refresh_human_review_audit(run_root, human_review)

    with pytest.raises(PublicSnapshotError, match=r"not a public HTTP\(S\) URI"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="unsafe-source-url",
        )


def test_symlinked_paper_directory_is_rejected(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    paper_root = run_root / manifest.paper_id
    relocated = tmp_path / "relocated-paper"
    paper_root.rename(relocated)
    paper_root.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(PublicSnapshotError, match="direct non-symlink paper directory"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="symlinked-paper",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("negative_count", "must be non-negative"),
        ("boolean_count", "must be an integer"),
        ("string_total", "must be an integer"),
        ("negative_wall_clock", "must be non-negative"),
        ("unknown_paper_status", "status is unsupported"),
        ("boolean_corpus_partition", "must be an integer"),
    ],
)
def test_corpus_and_paper_numeric_contract_rejects_malformed_values(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
    mutation: str,
    message: str,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    run = corpus["runs"][0]
    if mutation == "negative_count":
        run["counts"]["spot_checks"] = -1
    elif mutation == "boolean_count":
        run["counts"]["spot_checks"] = True
    elif mutation == "string_total":
        corpus["totals"]["spot_checks"] = "1"
    elif mutation == "negative_wall_clock":
        run["wall_clock_seconds"] = -0.1
    elif mutation == "unknown_paper_status":
        run["status"] = "looks_good"
    else:
        corpus["papers_succeeded"] = True
    _rewrite_corpus_and_paper_run(run_root, corpus)

    with pytest.raises(PublicSnapshotError, match=message):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id=f"malformed-{mutation}",
        )


def test_eee_file_count_must_match_run_and_corpus_partitions(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    corpus["runs"][0]["counts"]["eee_records"] = 0
    corpus["totals"]["eee_records"] = 0
    corpus["papers_with_eee"] = 0
    corpus["papers_without_eee"] = 1
    _rewrite_corpus_and_paper_run(run_root, corpus)

    with pytest.raises(PublicSnapshotError, match="eee_records disagrees with EEE files"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="eee-count-mismatch",
        )


def test_incomplete_global_budget_forces_lower_bound_usage(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    budget = corpus["provider_budget"]
    budget["structured_calls_started"] = 2
    budget["structured_calls_completed"] = 1
    budget["structured_calls_pending"] = 1
    _rewrite_corpus_and_paper_run(run_root, corpus)
    _refresh_human_review_audit(run_root, human_review)

    public_root = _build_snapshot(
        (run_root, model_selection, human_review),
        tmp_path / "public",
        snapshot_id="lower-bound-budget",
    )
    snapshot = json.loads((public_root / "snapshot.json").read_text(encoding="utf-8"))
    corpus_public = snapshot["corpora"][0]
    usage = corpus_public["papers_detail"][0]["extractor"]["usage"]

    assert corpus_public["provider_accounting"]["call_coverage"] == "lower_bound"
    assert corpus_public["provider_accounting"]["provider_reported_cost_usd"] is None
    assert corpus_public["provider_accounting"]["provider_reported_cost_usd_lower_bound"] == 0.001
    assert usage["cost_usd"] is None
    assert usage["cost_usd_lower_bound"] == 0.001
    assert usage["input_tokens"] is None


def test_execution_partition_mismatch_is_rejected(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    corpus["runs"][0]["extractor"]["execution"]["calls_succeeded"] = 0
    _rewrite_corpus_and_paper_run(run_root, corpus)

    with pytest.raises(PublicSnapshotError, match="executed-call partition disagrees"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="execution-mismatch",
        )


def test_finalized_pipeline_counts_must_match_downstream_stage_execution(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    for counts in (corpus["runs"][0]["counts"], corpus["totals"]):
        counts["tuple_candidates"] = 1
        counts["tuple_passed"] = 1
    _rewrite_corpus_and_paper_run(run_root, corpus)
    _refresh_human_review_audit(run_root, human_review)

    with pytest.raises(PublicSnapshotError, match="counts disagree with stage execution"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="stage-count-mismatch",
        )


def test_unbound_reference_score_is_rejected(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    write_json(
        run_root / manifest.paper_id / "reference-score.json",
        {"schema_version": "reference-score/0.7", "paper_id": manifest.paper_id},
    )

    with pytest.raises(PublicSnapshotError, match="reference_evaluation is missing"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="unbound-reference-score",
        )


@pytest.mark.parametrize("mutation", ["paper_id", "aggregate"])
def test_reference_audit_population_and_aggregates_are_bound(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
    mutation: str,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    audit_path = run_root / "reference-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if mutation == "paper_id":
        audit["results"][0]["paper_id"] = "stale-paper"
    else:
        audit["text_verified"] = 2
    write_json(audit_path, audit)

    with pytest.raises(PublicSnapshotError, match="paper population|aggregate counts"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id=f"audit-{mutation}",
        )


def test_positive_review_only_origin_is_an_overlapping_unresolved_subset(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    corpus = json.loads((run_root / "corpus-run.json").read_text(encoding="utf-8"))
    for counts in (corpus["runs"][0]["counts"], corpus["totals"]):
        counts["origin_candidates"] = 1
        counts["origin_unresolved"] = 1
        counts["origin_positive_review_only"] = 1
    corpus["runs"][0]["origin_retrieval"]["execution"]["candidates_selected"] = 1
    _rewrite_corpus_and_paper_run(run_root, corpus)
    _refresh_human_review_audit(run_root, human_review)

    public_root = _build_snapshot(
        (run_root, model_selection, human_review),
        tmp_path / "public",
        snapshot_id="positive-review-overlap",
    )
    snapshot = json.loads((public_root / "snapshot.json").read_text(encoding="utf-8"))
    counts = snapshot["corpora"][0]["papers_detail"][0]["counts"]
    assert counts["origin_positive_review_only"] == counts["origin_unresolved"] == 1


def test_incomplete_human_review_aborts_without_publishing(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    incomplete = json.loads(human_review.read_text(encoding="utf-8"))
    incomplete["decisions"]["completed"] = 0
    incomplete["decisions"]["outcome_counts"] = {
        "confirmed": 0,
        "incorrect": 0,
        "needs_followup": 0,
    }
    write_json(human_review, incomplete)

    with pytest.raises(PublicSnapshotError, match="not fully decided"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="incomplete-review",
        )

    assert not (tmp_path / "public" / "incomplete-review").exists()


def test_human_review_summary_cannot_overstate_paper_coverage(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    _, _, human_review, _ = _private_inputs(tmp_path / "inputs", manifest, eligible_candidate)
    overstated = json.loads(human_review.read_text(encoding="utf-8"))
    overstated["population"]["papers"] = 10
    overstated["sample"]["papers_reviewed"] = 10
    overstated["sample"]["paper_coverage"] = 1.0
    write_json(human_review, overstated)

    with pytest.raises(PublicSnapshotError, match="more papers than reviewed items"):
        _project_human_review_summary(human_review)


def test_human_review_full_coverage_requires_zero_candidate_review(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    _, _, human_review, _ = _private_inputs(tmp_path / "inputs", manifest, eligible_candidate)
    impossible = json.loads(human_review.read_text(encoding="utf-8"))
    impossible["population"]["candidates"] = 2
    impossible["population"]["papers"] = 2
    impossible["population"]["papers_without_candidates"] = 1
    impossible["sample"]["requested"] = 2
    impossible["sample"]["reviewed"] = 2
    impossible["sample"]["papers_reviewed"] = 2
    impossible["sample"]["paper_coverage"] = 1.0
    impossible["sample"]["item_type_counts"] = {
        "candidate": 2,
        "paper_without_candidates": 0,
    }
    impossible["sample"]["papers_without_candidates_reviewed"] = 0
    impossible["decisions"]["completed"] = 2
    impossible["decisions"]["outcome_counts"]["confirmed"] = 2
    write_json(human_review, impossible)

    with pytest.raises(PublicSnapshotError, match="papers with candidates"):
        _project_human_review_summary(human_review)


def test_human_review_named_counts_cannot_exceed_reviewed_items(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    _, _, human_review, _ = _private_inputs(tmp_path / "inputs", manifest, eligible_candidate)
    overstated = json.loads(human_review.read_text(encoding="utf-8"))
    overstated["sample"]["risk_reason_counts"]["exported"] = 999
    write_json(human_review, overstated)
    with pytest.raises(PublicSnapshotError, match="risk-reason count"):
        _project_human_review_summary(human_review)

    overstated["sample"]["risk_reason_counts"]["exported"] = 1
    overstated["decisions"]["issue_counts"]["evidence"] = 999
    write_json(human_review, overstated)
    with pytest.raises(PublicSnapshotError, match="issue count"):
        _project_human_review_summary(human_review)


def test_human_review_zero_candidate_population_must_match_corpus() -> None:
    human_review = {
        "population": {
            "papers": 10,
            "candidates": 658,
            "papers_without_candidates": 0,
        }
    }
    corpora = [
        {
            "papers": 10,
            "totals": {"candidates": 658},
            "papers_detail": [
                {"counts": {"candidates": 0}},
                *({"counts": {"candidates": 1}} for _ in range(9)),
            ],
        }
    ]

    with pytest.raises(PublicSnapshotError, match="zero-candidate paper population"):
        _validate_human_review_corpus_population(human_review, corpora)


def test_local_human_review_template_is_not_accepted_as_public_summary(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    write_json(
        human_review,
        {
            "schema_version": "human-review-template/0.1",
            "candidate": {"quote": "DO_NOT_PUBLISH_REVIEW_QUOTE"},
        },
    )

    with pytest.raises(PublicSnapshotError, match="not an aggregate review summary"):
        _build_snapshot(
            (run_root, model_selection, human_review),
            tmp_path / "public",
            snapshot_id="private-review-template",
        )

    assert not (tmp_path / "public" / "private-review-template").exists()


def test_public_snapshot_cli_forwards_repeated_run_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "primary"
    additional = tmp_path / "additional"
    primary.mkdir()
    additional.mkdir()
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")
    human_review = tmp_path / "human-review.json"
    human_review.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_build(**kwargs: object) -> Path:
        captured.update(kwargs)
        return tmp_path / "examples" / "pilot-cli"

    monkeypatch.setattr(cli, "build_public_snapshot", fake_build)
    result = CliRunner().invoke(
        app,
        [
            "export-public-snapshot",
            "pilot-cli",
            str(primary),
            str(selection),
            str(human_review),
            "--schema-path",
            str(SCHEMA),
            "--output-root",
            str(tmp_path / "examples"),
            "--additional-run-root",
            str(additional),
            "--selected-model",
            "example/extractor",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["snapshot_id"] == "pilot-cli"
    assert captured["corpus_run_root"] == primary
    assert captured["human_review_summary_path"] == human_review
    assert captured["additional_run_roots"] == [additional]
    assert captured["selected_model"] == "example/extractor"


def test_validate_eee_cli_uses_pinned_default_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    _, _, _, record_path = _private_inputs(tmp_path / "inputs", manifest, eligible_candidate)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["validate-eee", str(record_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"schema": "0.2.2", "issues": []}
