"""Deterministic, quote-free public preview for a sealed development corpus.

The preview is deliberately pre-human.  It reports model-proposed candidate-flow
counts, technical status, provenance-safe source identities, and provider usage,
but it never exports candidate records or claims canonical real-paper EEE.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from html import escape
from pathlib import Path
from urllib.parse import urlparse

from proceedings_to_eee.corpus import CorpusSpec, build_corpus_binding, load_corpus
from proceedings_to_eee.io import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_json,
)
from proceedings_to_eee.public_snapshot import PublicSnapshotError, _project_corpus
from proceedings_to_eee.reporting.public_development_summary import (
    PublicDevelopmentSummaryError,
    build_public_development_summary,
)
from proceedings_to_eee.reporting.validated_corpus import (
    CompletedCorpusValidationError,
    ValidatedCorpusReceipt,
    aggregate_stage_receipts,
    validate_completed_corpus_run,
)
from proceedings_to_eee.run_seal import RunSealVerificationError, VerifiedRunSeal, verify_run_seal
from proceedings_to_eee.sources.manifest import SourceManifest

PUBLIC_DEVELOPMENT_PREVIEW_SCHEMA_VERSION = "public-development-preview/0.1"
PUBLIC_SOURCE_INVENTORY_SCHEMA_VERSION = "public-source-inventory/0.1"
PUBLIC_USAGE_SCHEMA_VERSION = "public-development-usage/0.1"
PUBLIC_EVIDENCE_MAP_SCHEMA_VERSION = "public-candidate-evidence-map/0.1"
PUBLIC_PREVIEW_VERIFICATION_SCHEMA_VERSION = "public-development-preview-verification/0.1"

_ROUTE_SCHEMA_VERSION = "repair05-diagnostic-route-selection/0.1"
_PRECALL_SCHEMA_VERSION = "repair05-ten-paper-page-scoped-precall-seal/0.1"
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]*(?:/[A-Za-z0-9][A-Za-z0-9._:+-]*){1,3}$")
_MIN_PRIVATE_EVIDENCE_CHARS = 16
_MIN_EMBEDDED_PRIVATE_EVIDENCE_CHARS = 40
PRIVATE_EVIDENCE_SET_SCHEMA_VERSION = "private-evidence-set-binding/0.1"
PRIVATE_EVIDENCE_TEXT_CANONICALIZATION = "unicode-preserving-whitespace-collapse-strip/0.1"
_LOCAL_PATH = re.compile(
    r"(?:^|[\s\"'=:(])(?:/Users/|/home/|/private/|/tmp/|/var/folders/|"
    r"/Volumes/|/root/|/opt/|/etc/|/usr/|/Library/|/Applications/|/mnt/|"
    r"/workspace/|[A-Za-z]:\\|file://)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~-]{12,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|authorization)\s*[:=]\s*[\"'][^\"']{8,}[\"']", re.I),
    re.compile(r"\bOPENROUTER_API_KEY\b"),
)
_PRIVATE_IDENTIFIER_PATTERNS = (
    re.compile(r"\bobs_[0-9a-f]{8,}\b", re.I),
    re.compile(r"\bproposal_[0-9a-f]{8,}\b", re.I),
    re.compile(r"\btrow_[0-9a-f]{8,}\b", re.I),
    re.compile(r"\bblock_[0-9a-f]{8,}\b", re.I),
    re.compile(r"\btuple_ev_[0-9a-f]{8,}\b", re.I),
)
_FORBIDDEN_PUBLIC_KEYS = {
    "annotation",
    "annotations",
    "api_key",
    "authorization",
    "block_id",
    "cache_relpath",
    "calls",
    "candidate_id",
    "candidate_occurrence_id",
    "command",
    "completion",
    "cookie",
    "decision",
    "decisions",
    "exact_excerpt",
    "exact_quote",
    "manifest_path",
    "messages",
    "notes",
    "observation_id",
    "output_path",
    "pdf_path",
    "private_annotations",
    "prompt",
    "prompt_template",
    "proposal_id",
    "provider_request",
    "provider_response",
    "quote",
    "raw_payload",
    "raw_response",
    "reference_path",
    "request_id",
    "reviewer_id",
    "reviewer_ids",
    "reviewer_identity",
    "reviewer_notes",
    "row_id",
    "schema_path",
    "secret",
    "warnings",
}
_ROUTE_STAGES = (
    "block_candidate_extraction",
    "row_disposition",
    "tuple_resolution",
    "independent_verification",
    "origin_retrieval",
)
_ROUTE_TO_PUBLIC_STAGE = {
    "block_candidate_extraction": "extractor",
    "row_disposition": "row_enumeration",
    "tuple_resolution": "tuple_resolution",
    "independent_verification": "verifier",
    "origin_retrieval": "origin_retrieval",
}
_ROUTE_TO_RECEIPT_STAGE = {
    "block_candidate_extraction": "extractor",
    "row_disposition": "row_disposition",
    "tuple_resolution": "tuple_resolution",
    "independent_verification": "independent_verification",
    "origin_retrieval": "origin_retrieval",
}
_ROUTE_STAGE_FIELDS = {
    "execution_selection_sha256",
    "quality_source_artifact_sha256",
    "selected_model",
    "selection_basis",
}
_ROUTE_FIELDS = {
    "schema_version",
    "status",
    "experiment_id",
    "amendment_id",
    "code_commit",
    "code_tree",
    "runner_sha256",
    "campaign_seal_sha256",
    "plan_bundle_sha256",
    "stages",
}
_PRECALL_FIELDS = {
    "schema_version",
    "status",
    "sealed_at",
    "bindings",
    "declared_models",
    "selected_models",
    "command_argv",
    "budget",
    "privacy",
    "outputs",
    "execution_policy",
    "seal_sha256",
}
_PRECALL_BINDING_FIELDS = {
    "campaign_plan_sha256",
    "route_selection_sha256",
    "route_ranking_audit_sha256",
    "campaign_seal_sha256",
    "campaign_seal_file_sha256",
    "corpus_config_sha256",
    "corpus_spec_sha256",
    "paper_ids_sha256",
    "corpus_preflight_sha256",
    "corpus_freeze_sha256",
    "preflight_inventory_sha256",
    "schema_sha256",
    "provider_budget_contract_sha256",
    "morning_packet_manifest_sha256",
}
_PRECALL_BUDGET_FIELDS = {
    "max_structured_calls",
    "max_provider_cost_usd",
    "provider_call_cost_reservation_usd",
    "max_transport_attempts",
}
_PRECALL_PRIVACY_FIELDS = {
    "zdr",
    "data_collection",
    "label_blind_source_derived_scientific_payloads_only",
    (
        "human_labels_answers_scores_holdout_references_reviewer_ids_"
        "credentials_or_private_annotations_sent"
    ),
}
_PRECALL_OUTPUT_FIELDS = {
    "raw_root",
    "sealed_root",
    "raw_root_was_fresh_before_seal",
    "sealed_root_was_fresh_before_seal",
}
_PRECALL_EXECUTION_POLICY_FIELDS = {
    "working_directory",
    "command_argv",
    "seal_command_argv",
    "accepted_terminal_exit_codes",
    "completed_with_review_exit_code",
    "bounded_incomplete_exit_code",
    "generic_budget_doubling_forbidden",
    "copy_precall_seal_into_raw_root_before_execution",
    "seal_regardless_of_terminal_exit",
    "verify_seal_before_semantic_inspection",
}
_PREVIEW_FILES = {
    "README.md",
    "publication-manifest.json",
    "run-summary.json",
    "corpus.json",
    "route-selection.json",
    "sources.json",
    "usage.json",
    "evidence-map.json",
    "evidence-map.html",
    "verification.json",
    "SHA256SUMS",
}
_JSON_FILES = {name for name in _PREVIEW_FILES if name.endswith(".json")}
_MAX_PUBLIC_FILE_BYTES = 2_000_000

_CORPUS_TOP_FIELDS = {
    "schema_version",
    "corpus_id",
    "status",
    "generated_at",
    "papers",
    "papers_total",
    "papers_not_started",
    "papers_succeeded",
    "papers_failed",
    "papers_bounded_incomplete",
    "papers_with_eee",
    "papers_without_candidates",
    "papers_without_eee",
    "papers_needing_review",
    "source_artifacts",
    "totals",
    "provider_accounting",
    "reference_evaluation",
    "papers_detail",
}
_CORPUS_SOURCE_ARTIFACT_FIELDS = {"corpus_run_sha256", "corpus_evaluation_sha256"}
_CORPUS_PROVIDER_FIELDS = {
    "schema_version",
    "budget_status",
    "call_coverage",
    "structured_calls_started",
    "structured_calls_completed",
    "structured_calls_pending",
    "provider_call_telemetry_calls",
    "provider_call_telemetry_missing_calls",
    "provider_reported_cost_calls",
    "provider_reported_cost_usd",
    "provider_reported_cost_usd_lower_bound",
    "provider_reported_input_tokens_lower_bound",
    "provider_reported_output_tokens_lower_bound",
    "provider_reported_reasoning_tokens_lower_bound",
    "provider_reported_total_tokens_lower_bound",
}
_CORPUS_COUNT_FIELDS = {
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
}
_REFERENCE_COUNT_FIELDS = {
    "spot_checks",
    "spot_checks_exact",
    "reference_observations",
    "reference_true_positives",
    "reference_false_positives",
    "reference_false_negatives",
    "negative_control_false_primary",
}
_CORPUS_PAPER_FIELDS = {
    "paper_id",
    "title",
    "status",
    "wall_clock_seconds",
    "source_manifest_sha256",
    "selected_pages",
    "layout",
    "result_block_segmentation",
    "extractor",
    "row_enumeration",
    "candidate_validation",
    "tuple_resolution",
    "verifier",
    "origin_retrieval",
    "eee_schema",
    "code",
    "counts",
}
_CORPUS_LAYOUT_FIELDS = {"layout_parser", "layout_parser_version"}
_CORPUS_SEGMENTATION_FIELDS = {
    "max_lines",
    "max_characters",
    "context_lines",
    "trailing_context_lines",
    "overlap_lines",
    "signal_gap_lines",
    "max_blank_gap",
    "max_data_rows",
    "min_signal_score",
    "max_blocks_per_page",
}
_CORPUS_CANDIDATE_VALIDATION_FIELDS = {"schema_version", "min_confidence"}
_CORPUS_CANDIDATE_VALIDATION_VERSIONS = {
    "candidate-validation/0.1",
    "candidate-validation/0.2",
}


def _candidate_validation_fields(candidate_validation: Mapping[str, object]) -> set[str]:
    """0.2 names the producer-origin export policy the run used; 0.1 predates it."""

    if candidate_validation.get("schema_version") == "candidate-validation/0.2":
        return _CORPUS_CANDIDATE_VALIDATION_FIELDS | {"origin_policy"}
    return _CORPUS_CANDIDATE_VALIDATION_FIELDS


_CORPUS_EEE_SCHEMA_FIELDS = {"version", "sha256"}
_CORPUS_STAGE_FIELDS = {
    "extractor": {
        "provider",
        "model",
        "max_tokens",
        "temperature",
        "reasoning_effort",
        "seed",
        "prompt_sha256",
        "require_parameters",
        "request_contract",
        "usage",
        "execution",
    },
    "row_enumeration": {
        "enabled",
        "provider",
        "model",
        "max_tokens",
        "temperature",
        "reasoning_effort",
        "seed",
        "prompt_sha256",
        "limits",
        "require_parameters",
        "request_contract",
        "usage",
        "execution",
    },
    "tuple_resolution": {
        "enabled",
        "provider",
        "model",
        "max_tokens",
        "temperature",
        "reasoning_effort",
        "seed",
        "prompt_sha256",
        "mutates_candidate_tuple",
        "allows_origin_or_export",
        "requires_export_concordant_tuple",
        "require_parameters",
        "request_contract",
        "usage",
        "execution",
    },
    "verifier": {
        "enabled",
        "model",
        "max_tokens",
        "temperature",
        "reasoning_effort",
        "seed",
        "verification_schema_version",
        "grounding_schema_version",
        "require_parameters",
        "request_contract",
        "usage",
        "execution",
    },
    "origin_retrieval": {
        "enabled",
        "model",
        "max_tokens",
        "temperature",
        "reasoning_effort",
        "seed",
        "prompt_sha256",
        "requires_independent_verifier_accept",
        "require_parameters",
        "request_contract",
        "usage",
        "execution",
    },
}
_CORPUS_ROW_LIMIT_FIELDS = {
    "max_rows_per_batch",
    "max_characters_per_batch",
    "max_value_tokens_per_batch",
    "max_recovery_depth",
    "min_dense_table_rows",
}
_REQUEST_CONTRACT_FIELDS = {
    "schema_version",
    "schema",
    "privacy",
    "routing",
    "max_tokens",
    "seed",
    "completion_token_parameter",
}
_REQUEST_SCHEMA_FIELDS = {"response_format", "schema_name", "schema_sha256", "schema_strict"}
_REQUEST_PRIVACY_FIELDS = {"zdr", "data_collection"}
_REQUEST_ROUTING_FIELDS = {"require_parameters"}
_STAGE_USAGE_FIELDS = {
    "calls_attempted",
    "calls_field_count",
    "resumed_calls_field_count",
    "call_accounting_basis",
    "models_returned",
    "providers_returned",
    "model_returned_matches_requested_calls",
    "model_returned_unverified_calls",
    "cost_reported_calls",
    "cost_missing_calls",
    "cost_usd_lower_bound",
    "cost_usd",
    "input_tokens_reported_calls",
    "input_tokens_missing_calls",
    "input_tokens_lower_bound",
    "input_tokens",
    "output_tokens_reported_calls",
    "output_tokens_missing_calls",
    "output_tokens_lower_bound",
    "output_tokens",
    "reasoning_tokens_reported_calls",
    "reasoning_tokens_missing_calls",
    "reasoning_tokens_lower_bound",
    "reasoning_tokens",
    "total_tokens_reported_calls",
    "total_tokens_missing_calls",
    "total_tokens_lower_bound",
    "total_tokens",
    "retries_lower_bound",
    "attempts_lower_bound",
}
_STAGE_EXECUTION_FIELDS = {
    "block_candidate_extraction": {
        "blocks_total",
        "blocks_succeeded",
        "blocks_failed",
        "blocks_resumed",
        "calls_succeeded",
        "calls_failed",
        "calls_resumed",
        "calls_resumed_succeeded",
        "calls_resumed_failed",
        "requests_rejected",
        "transport_failures",
        "local_failures",
        "no_call_failures",
    },
    "row_disposition": {
        "batches_total",
        "batches_resumed",
        "batches_executed",
        "invalid_rows_seen",
        "unknown_row_ids_seen",
    },
    "tuple_resolution": {
        "candidates_selected",
        "candidates_passed",
        "candidates_routed_to_review",
        "candidates_unsupported",
        "candidates_failed",
        "candidates_mismatched",
        "candidates_resumed",
    },
    "independent_verification": {
        "candidates_selected",
        "candidates_unbound",
        "candidates_verified",
        "candidates_failed",
        "candidates_resumed",
        "candidates_resumed_succeeded",
        "candidates_resumed_failed",
        "candidates_executed",
        "candidates_executed_succeeded",
        "candidates_executed_failed",
    },
    "origin_retrieval": {
        "candidates_selected",
        "candidates_resumed",
        "candidates_failed",
        "candidates_deterministic_external",
    },
}
_PUBLIC_TO_CORPUS_STAGE = {
    "block_candidate_extraction": "extractor",
    "row_disposition": "row_enumeration",
    "tuple_resolution": "tuple_resolution",
    "independent_verification": "verifier",
    "origin_retrieval": "origin_retrieval",
}
_PUBLIC_REQUEST_SCHEMA_NAMES = {
    "block_candidate_extraction": "paper_evaluation_candidates",
    "row_disposition": "paper_table_row_dispositions",
    "tuple_resolution": "candidate_tuple_resolution",
    "independent_verification": "candidate_evidence_verification_v2",
    "origin_retrieval": "candidate_producer_origin_selection",
}
_STAGE_RECEIPT_FIELDS = {
    "status",
    "papers_validated",
    "papers_partial_failure",
    "papers_not_run",
    "completed_calls",
    "selected_items",
    "failed_items",
    "checkpoint_contract_sha256s",
    "sidecar_sha256s",
}
_TELEMETRY_FIELDS = {
    "basis",
    "recorded_structured_invocations",
    "cost_reported_calls",
    "cost_usd_lower_bound",
    "input_tokens_reported_calls",
    "input_tokens_lower_bound",
    "output_tokens_reported_calls",
    "output_tokens_lower_bound",
    "reasoning_tokens_reported_calls",
    "reasoning_tokens_lower_bound",
    "total_tokens_reported_calls",
    "total_tokens_lower_bound",
    "latency_seconds_total",
    "latency_seconds_max",
    "latency_seconds_mean",
    "retries_lower_bound",
    "attempts_lower_bound",
}
_SUMMARY_TOP_FIELDS = {
    "schema_version",
    "projection_mode",
    "current_sealed_receipt",
    "statement",
    "run_binding",
    "scope",
    "technical_health",
    "row_enumeration",
    "outputs",
    "canonical_eee",
    "numeric_export_provenance",
    "provider_usage_recorded",
    "reference_evaluation",
    "export_provenance_modes",
    "annotation_status",
    "limitations",
    "privacy",
}
_SUMMARY_SCOPE_FIELDS = {
    "split",
    "papers",
    "holdout_included",
    "private_human_annotations_included",
    "independent_human_validation",
}
_SUMMARY_CANONICAL_FIELDS = {
    "status",
    "records",
    "schema_issues",
    "safe_empty_output_is_valid",
    "positive_paper_produced_origin_required",
    "automatic_positive_origin_enabled",
}
_SUMMARY_NUMERIC_PROVENANCE_FIELDS = {
    "status",
    "exported_observations",
    "complete_observations",
    "all_complete",
    "evidence_quotations_included",
}
_SUMMARY_REFERENCE_FIELDS = {"status", "precision", "generalization_evidence"}
_SUMMARY_PROVENANCE_FIELDS = {"source_run", "reviewed_derived", "counts_combined"}
_SUMMARY_SOURCE_PROVENANCE_FIELDS = {
    "status",
    "eee_records",
    "eee_observations",
    "provenance_mode_counts",
}
_SUMMARY_REVIEWED_PROVENANCE_FIELDS = {
    "status",
    "eee_records",
    "eee_observations",
    "provenance_mode_counts",
    "exported_observation_authority_mode_counts",
    "independence_status",
    "locked_review_context_verified",
    "source_and_reviewed_counts_combined",
}
_SUMMARY_PROVENANCE_MODE_FIELDS = {
    "legacy_human_reviewed",
    "legacy_manual",
    "tuple_audited_human_reviewed",
    "tuple_gated_production",
    "tuple_gated_unverified",
    "unclassified_legacy_source",
}
_SUMMARY_ANNOTATION_FIELDS = {
    "source_run_human_annotations_included",
    "reviewed_derived_supplied",
    "reviewed_derived_authority_mode_counts",
    "inter_annotator_agreement_available",
}
_SUMMARY_PRIVACY_FIELDS = {
    "evidence_quotations_included",
    "paper_level_rows_or_labels_included",
    "private_annotations_included",
    "provider_traces_included",
    "request_identifiers_included",
    "credentials_included",
    "local_paths_included",
}
_SUMMARY_RUN_BINDING_FIELDS = {
    "corpus_id",
    "run_id",
    "generated_at",
    "corpus_run_sha256",
    "recorded_corpus_spec_sha256",
    "paper_ids_sha256",
    "corpus_spec_binding_status",
    "extractor",
    "row_extractor",
    "candidate_validation",
    "eee_schema",
    "code",
    "run_seal",
    "stage_chain",
}
_SUMMARY_MODEL_BINDING_FIELDS = {
    "provider",
    "model",
    "max_tokens",
    "temperature",
    "reasoning_effort",
    "seed",
    "prompt_sha256",
    "request_contract_sha256",
}
_SUMMARY_STAGE_CHAIN_FIELDS = {
    "mode",
    "five_stage_gate_status",
    "missing_gate_treated_as_passed",
    "checkpoint_validation",
}
_SUMMARY_OUTPUT_FIELDS = {
    "count_projection_basis",
    "candidates",
    "candidates_before_deduplication",
    "duplicates_removed",
    "candidate_proposal_removal_rate",
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
    "verifications",
    "verifier_accepts",
    "verifier_rejects",
    "verifier_reviews",
    "verifier_failed",
    "origin_candidates",
    "origin_failed",
    "origin_positive_review_only",
    "origin_external",
    "origin_unresolved",
    "origin_no_signal",
    "export_status_counts",
    "text_support_status_counts",
    "referential_status_counts",
    "attribution_state_counts",
    "candidate_lineage",
}
_SUMMARY_CANDIDATE_LINEAGE_FIELDS = {
    "proposals",
    "candidate_occurrences",
    "merged_candidates",
    "final_candidates",
    "runs_bound",
    "all_current_schema_runs_bound",
}
_SUMMARY_ROW_FIELDS = {
    "complete_extraction",
    "all_rows_accounted_for",
    "all_planned_rows_partitioned",
    "all_batchable_rows_resolved",
    "no_unknown_or_invalid_rows_seen",
    "tables_considered",
    "dense_tables",
    "rows_planned",
    "rows_resolved",
    "rows_unresolved",
    "rows_unbatchable",
    "unknown_row_ids_seen",
    "invalid_rows_seen",
    "dispositions",
}
_SUMMARY_ROW_DISPOSITION_FIELDS = {"result", "not_result", "uncertain"}
_SUMMARY_TECHNICAL_FIELDS = {
    "status",
    "run_completeness",
    "papers_succeeded",
    "papers_failed",
    "papers_needing_review",
    "wall_clock_seconds",
    "review_reason_counts",
    "release_ready",
}
_SUMMARY_REVIEW_REASON_FIELDS = {
    "candidate_review_required",
    "eee_schema_validation_failure",
    "paper_run_error",
    "row_enumeration_unbatchable",
    "row_enumeration_uncertain",
    "row_enumeration_unknown_ids",
    "row_enumeration_unresolved",
    "selected_result_blocks_produced_zero_candidates",
    "tuple_resolution_review_required",
    "zero_selected_result_blocks",
    "zero_valid_eee_records",
}

_OPERATIONAL_STAGE_EXTRA_FIELDS = {
    "block_candidate_extraction": {
        "blocks_total",
        "blocks_succeeded",
        "blocks_failed",
        "blocks_resumed",
    },
    "row_disposition": {
        "tables_considered",
        "dense_tables",
        "rows_planned",
        "unbatchable_rows",
        "base_batches",
        "maximum_calls",
        "batches_resumed",
    },
    "tuple_resolution": {
        "candidates_selected",
        "candidates_passed",
        "candidates_routed_to_review",
        "candidates_resumed",
    },
    "independent_verification": {
        "candidates_verified",
        "candidates_failed",
        "candidates_resumed",
        "candidates_executed",
    },
    "origin_retrieval": {
        "candidates_selected",
        "candidates_failed",
        "candidates_resumed",
    },
}

_MANIFEST_ARTIFACT_STATUS_FIELDS = {
    "classification",
    "release_ready",
    "human_review_status",
    "canonical_real_paper_eee_available",
}
_MANIFEST_SOURCE_RUN_FIELDS = {
    "corpus_id",
    "evaluation_split",
    "paper_count",
    "generated_at",
    "corpus_spec_sha256",
    "paper_ids_sha256",
    "corpus_run_sha256",
    "run_seal",
    "code",
}
_MANIFEST_RUN_SEAL_FIELDS = {
    "schema_version",
    "seal_sha256",
    "tree_sha256",
    "file_count",
    "total_bytes",
}
_CODE_FIELDS = {"git_available", "git_commit", "git_dirty", "source_tree_sha256"}
_MANIFEST_PROSPECTIVE_FIELDS = {
    "route_schema_version",
    "route_status",
    "route_selection_sha256",
    "precall_contract_sha256",
    "precall_file_sha256",
    "binding_status",
    "selected_models",
    "privacy",
    "budget",
}
_PUBLIC_PRECALL_PRIVACY_FIELDS = {
    "zdr",
    "data_collection",
    "label_blind_source_derived_scientific_payloads_only",
    "forbidden_private_or_labelled_payloads_sent",
}
_MANIFEST_OUTPUT_FIELDS = {"automatic_source_run", "reviewed_derived", "reviewed_summary"}
_AUTOMATIC_OUTPUT_FIELDS = {"status", "canonical_eee_records", "present"}
_REVIEWED_OUTPUT_FIELDS = {"status", "expected_relative_path", "records", "present"}
_REVIEWED_SUMMARY_FIELDS = {"status", "expected_relative_path", "present"}

_SOURCE_TOP_FIELDS = {"schema_version", "corpus_id", "evaluation_split", "papers"}
_SOURCE_PAPER_FIELDS = {
    "paper_id",
    "title",
    "year",
    "venue",
    "doi",
    "arxiv_id",
    "acm_url",
    "pdf_url",
    "supplement_urls",
    "repository_url",
    "repository_commit",
    "perspective_role",
    "page_scope",
    "source_manifest_sha256",
    "sources",
}
_SOURCE_PAGE_SCOPE_FIELDS = {
    "selection_mode",
    "configured_include_pages",
    "selected_pages",
    "max_result_pages",
}
_SOURCE_ITEM_FIELDS = {
    "source_id",
    "role",
    "public_original_uri",
    "public_resolved_uri",
    "retrieved_at",
    "sha256",
    "byte_size",
    "media_type",
    "git_commit",
    "access_status",
    "license_disposition",
}

_USAGE_TOP_FIELDS = {
    "schema_version",
    "accounting_scope",
    "statement",
    "provider_budget",
    "replayed_provider_usage",
    "stage_receipts",
    "operational_accounting",
    "paper_accounting",
    "qualification_campaign_usage",
}
_USAGE_OPERATIONAL_FIELDS = {"wall_clock_seconds", "stages"}
_USAGE_PAPER_FIELDS = {"paper_id", "status", "wall_clock_seconds", "stages"}
_USAGE_PAPER_STAGE_FIELDS = {"status", "selected_model", "execution", "usage"}
_USAGE_QUALIFICATION_FIELDS = {"status", "combined_with_source_run"}

_EVIDENCE_TOP_FIELDS = {"schema_version", "status", "statement", "scope", "papers"}
_EVIDENCE_SCOPE_FIELDS = {
    "corpus_id",
    "evaluation_split",
    "papers",
    "holdout",
    "whole_paper_evaluation",
    "independent_human_validation",
}
_EVIDENCE_PAPER_FIELDS = {
    "paper_id",
    "title",
    "year",
    "venue",
    "perspective_role",
    "page_scope",
    "technical",
    "candidate_layer",
    "tuple_gate",
    "independent_verifier",
    "producer_origin",
    "review_boundary",
}
_EVIDENCE_PAGE_SCOPE_FIELDS = {"selection_mode", "selected_pages"}
_EVIDENCE_TECHNICAL_FIELDS = {
    "paper_status",
    "needs_review",
    "review_reasons",
    "stage_status",
}
_EVIDENCE_CANDIDATE_FIELDS = {
    "model_proposals",
    "candidate_occurrences",
    "deduplicated_candidates",
    "duplicates_removed",
    "primary_result_candidates",
    "requires_human_review",
    "deterministically_not_eligible",
}
_EVIDENCE_TUPLE_FIELDS = {
    "selected",
    "passed",
    "routed_to_review",
    "unsupported_subset",
    "failed_subset",
}
_EVIDENCE_VERIFIER_FIELDS = {
    "selected",
    "verified",
    "accepts",
    "rejects",
    "reviews",
    "failed_subset",
}
_EVIDENCE_ORIGIN_FIELDS = {
    "selected",
    "external",
    "unresolved",
    "no_signal",
    "positive_review_only_subset",
    "failed_subset",
    "automatic_positive_promotion",
}
_EVIDENCE_REVIEW_FIELDS = {
    "accepted_by_model_gates",
    "not_accepted_by_complete_model_chain",
    "withheld_from_canonical_eee",
    "human_review_decisions_completed",
    "canonical_eee_records",
    "final_reviewed_records",
    "status",
}

_VERIFICATION_TOP_FIELDS = {"schema_version", "status", "checks", "bindings", "release_ready"}
_VERIFICATION_CHECK_FIELDS = {
    "source_run_seal_verified",
    "five_stage_receipt_replayed",
    "corpus_specification_rederived",
    "prospective_route_bound",
    "precall_privacy_contract_verified",
    "provider_accounting_reconciled",
    "automatic_canonical_eee_zero",
    "human_review_not_supplied",
    "allowlist_and_privacy_scan_passed_before_checksums",
}
_VERIFICATION_BINDING_FIELDS = {
    "source_run_seal_sha256",
    "source_run_tree_sha256",
    "corpus_run_sha256",
    "corpus_spec_sha256",
    "paper_ids_sha256",
    "route_selection_sha256",
    "precall_contract_sha256",
    "precall_file_sha256",
}


class PublicDevelopmentPreviewError(ValueError):
    """A sealed corpus cannot be projected into a safe pre-human preview."""


class _DuplicateJsonKeyError(ValueError):
    pass


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PublicDevelopmentPreviewError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise PublicDevelopmentPreviewError(f"{context} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise PublicDevelopmentPreviewError(f"{context} does not use the exact public schema")


def _safe_id(value: object, context: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PublicDevelopmentPreviewError(f"{context} is not a safe identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PublicDevelopmentPreviewError(f"{context} is not a lowercase SHA-256")
    return value


def _git_object(value: object, context: str) -> str:
    if not isinstance(value, str) or _GIT_OBJECT.fullmatch(value) is None:
        raise PublicDevelopmentPreviewError(f"{context} is not a full Git object ID")
    return value


def _model_id(value: object, context: str) -> str:
    if not isinstance(value, str) or _MODEL_ID.fullmatch(value) is None:
        raise PublicDevelopmentPreviewError(f"{context} is not a public-safe model ID")
    return value


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicDevelopmentPreviewError(f"{context} must be a non-negative integer")
    return value


def _nonnegative_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PublicDevelopmentPreviewError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise PublicDevelopmentPreviewError(f"{context} must be finite and non-negative")
    return result


def _same_number(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, int | float) or not isinstance(right, int | float):
        return False
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-10)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError
        result[key] = value
    return result


def _strict_json_bytes(content: bytes, context: str) -> object:
    try:
        return json.loads(content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKeyError,
        RecursionError,
        OverflowError,
        ValueError,
    ) as error:
        raise PublicDevelopmentPreviewError(f"{context} is not strict UTF-8 JSON") from error


def _strict_json_file(path: Path, context: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise PublicDevelopmentPreviewError(f"{context} must be one regular file")
    try:
        return _strict_json_bytes(path.read_bytes(), context)
    except OSError as error:
        raise PublicDevelopmentPreviewError(f"{context} could not be read") from error


def _compact_json_sha256(value: object, context: str) -> str:
    """Hash a provider-budget contract with the ledger's compact JSON encoding."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PublicDevelopmentPreviewError(f"{context} is not canonical JSON") from error
    return sha256_bytes(payload)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, context: str) -> Path:
    """Reject both live and dangling links in every existing path component."""

    absolute = _absolute_without_resolving(path)
    for candidate in (absolute, *absolute.parents):
        try:
            if candidate.is_symlink():
                raise PublicDevelopmentPreviewError(f"{context} must not traverse a symbolic link")
        except OSError as error:
            raise PublicDevelopmentPreviewError(
                f"{context} could not be inspected safely"
            ) from error
    return absolute


def _open_directory_fd(path: Path, context: str) -> int:
    absolute = _absolute_without_resolving(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.sep, flags)
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        if "descriptor" in locals():
            os.close(descriptor)
        raise PublicDevelopmentPreviewError(f"{context} could not be opened safely") from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise PublicDevelopmentPreviewError(f"{context} must be one regular directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_descriptor(descriptor: int, context: str) -> bytes:
    chunks: list[bytes] = []
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError as error:
        raise PublicDevelopmentPreviewError(f"{context} could not be read safely") from error
    return b"".join(chunks)


def _capture_directory_files(
    root: Path,
    *,
    expected_names: set[str],
    context: str,
) -> tuple[dict[str, bytes], tuple[int, int]]:
    """Capture an allowlisted directory through no-follow descriptors exactly once."""

    root_fd = _open_directory_fd(root, context)
    try:
        root_stat = os.fstat(root_fd)
        try:
            names = os.listdir(root_fd)
        except OSError as error:
            raise PublicDevelopmentPreviewError(f"{context} could not be listed safely") from error
        if set(names) != expected_names or len(names) != len(expected_names):
            raise PublicDevelopmentPreviewError("public preview file allowlist disagrees")
        contents: dict[str, bytes] = {}
        for name in sorted(expected_names):
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(name, flags, dir_fd=root_fd)
            except OSError as error:
                raise PublicDevelopmentPreviewError(
                    "public preview contains a non-regular entry"
                ) from error
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise PublicDevelopmentPreviewError(
                        "public preview contains a non-regular entry"
                    )
                content = _read_descriptor(descriptor, f"public preview file {name}")
                after = os.fstat(descriptor)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ) or len(content) != after.st_size:
                    raise PublicDevelopmentPreviewError(
                        "public preview changed while it was being captured"
                    )
                contents[name] = content
            finally:
                os.close(descriptor)
        if set(os.listdir(root_fd)) != expected_names:
            raise PublicDevelopmentPreviewError(
                "public preview changed while it was being captured"
            )
        return contents, (root_stat.st_dev, root_stat.st_ino)
    finally:
        os.close(root_fd)


def _assert_capture_still_current(
    root: Path,
    contents: Mapping[str, bytes],
    root_identity: tuple[int, int],
) -> None:
    """Re-open the published path and reject a swap after the captured validation."""

    current, current_identity = _capture_directory_files(
        root,
        expected_names=set(contents),
        context="public preview root",
    )
    if current_identity != root_identity or current != dict(contents):
        raise PublicDevelopmentPreviewError("public preview changed during verification")


def _scan_text(text: str, context: str) -> None:
    if _LOCAL_PATH.search(text):
        raise PublicDevelopmentPreviewError(f"absolute local path found in {context}")
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        raise PublicDevelopmentPreviewError(f"credential-like value found in {context}")
    if any(pattern.search(text) for pattern in _PRIVATE_IDENTIFIER_PATTERNS):
        raise PublicDevelopmentPreviewError(f"private record identifier found in {context}")


def _audit_public_value(value: object, context: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PublicDevelopmentPreviewError(f"{context} contains a non-string key")
            if key.casefold() in _FORBIDDEN_PUBLIC_KEYS:
                raise PublicDevelopmentPreviewError(f"forbidden public key at {context}.{key}")
            _audit_public_value(child, f"{context}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            _audit_public_value(child, f"{context}[{index}]")
        return
    if isinstance(value, str):
        _scan_text(value, context)
        return
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise PublicDevelopmentPreviewError(f"{context} contains an unsupported public value")


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _private_evidence_texts(run_root: Path) -> set[str]:
    """Collect exact evidence strings only for a final non-projection assertion."""

    result: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {
                    "quote",
                    "exact_quote",
                    "exact_excerpt",
                    "matched_text",
                } and isinstance(child, str):
                    normalized = _normalized_text(child)
                    if len(normalized) >= _MIN_PRIVATE_EVIDENCE_CHARS:
                        result.add(normalized)
                else:
                    collect(child)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            for child in value:
                collect(child)

    for path in sorted(run_root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix == ".json":
            try:
                collect(_strict_json_file(path, "sealed source JSON"))
            except PublicDevelopmentPreviewError:
                # The sealed tree may contain non-public schemas; validation of those files is
                # owned by the run receipt.  Only valid JSON can contribute evidence strings.
                continue
        elif path.suffix == ".jsonl":
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    collect(_strict_json_bytes(line.encode("utf-8"), "sealed source JSONL"))
                except PublicDevelopmentPreviewError:
                    continue
    return result


def collect_private_evidence_texts(run_root: Path) -> set[str]:
    """Collect normalized sealed-run evidence strings for nonreproduction checks."""

    return _private_evidence_texts(run_root)


def private_evidence_set_binding(evidence_texts: set[str]) -> dict[str, object]:
    """Bind a normalized evidence set without exposing any evidence string."""

    normalized = sorted({_normalized_text(item) for item in evidence_texts})
    if any(len(item) < _MIN_PRIVATE_EVIDENCE_CHARS for item in normalized):
        raise PublicDevelopmentPreviewError("private evidence set contains a short entry")
    return {
        "schema_version": PRIVATE_EVIDENCE_SET_SCHEMA_VERSION,
        "canonicalization": PRIVATE_EVIDENCE_TEXT_CANONICALIZATION,
        "minimum_exact_match_characters": _MIN_PRIVATE_EVIDENCE_CHARS,
        "minimum_embedded_match_characters": _MIN_EMBEDDED_PRIVATE_EVIDENCE_CHARS,
        "evidence_text_count": len(normalized),
        "evidence_set_sha256": sha256_bytes(canonical_json_bytes(normalized)),
    }


def _assert_no_evidence_text(value: object, evidence_texts: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_evidence_text(key, evidence_texts)
            _assert_no_evidence_text(child, evidence_texts)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for child in value:
            _assert_no_evidence_text(child, evidence_texts)
    elif isinstance(value, str):
        normalized = _normalized_text(value)
        if normalized in evidence_texts or any(
            len(evidence) >= _MIN_EMBEDDED_PRIVATE_EVIDENCE_CHARS and evidence in normalized
            for evidence in evidence_texts
        ):
            raise PublicDevelopmentPreviewError("public preview reproduces an evidence quotation")


def assert_no_private_evidence_text(value: object, evidence_texts: set[str]) -> None:
    """Reject exact or substantial embedded reproduction of sealed-run evidence."""

    _assert_no_evidence_text(value, evidence_texts)


def _verify_seal(run_root: Path) -> VerifiedRunSeal:
    if run_root.is_symlink():
        raise PublicDevelopmentPreviewError("sealed source run must not be a symbolic link")
    try:
        return verify_run_seal(run_root)
    except RunSealVerificationError as error:
        raise PublicDevelopmentPreviewError(
            f"sealed source run failed verification ({error.code})"
        ) from error


def _load_bound_corpus(path: Path) -> tuple[CorpusSpec, dict[str, str], str]:
    if path.is_symlink() or not path.is_file():
        raise PublicDevelopmentPreviewError("corpus specification must be one regular file")
    try:
        corpus = load_corpus(path)
    except (OSError, ValueError) as error:
        raise PublicDevelopmentPreviewError("corpus specification is invalid") from error
    if any(paper.reference_path or paper.expected_spot_checks for paper in corpus.papers):
        raise PublicDevelopmentPreviewError(
            "public pre-human preview forbids reference labels and evaluation material"
        )
    binding = build_corpus_binding(corpus)
    return corpus, binding, sha256_file(path)


def _validate_route_manifest(
    path: Path,
    *,
    receipt: ValidatedCorpusReceipt,
    corpus: Mapping[str, object],
) -> dict[str, object]:
    route = _mapping(_strict_json_file(path, "route manifest"), "route manifest")
    _exact_keys(route, _ROUTE_FIELDS, "route manifest")
    if route.get("schema_version") != _ROUTE_SCHEMA_VERSION:
        raise PublicDevelopmentPreviewError("route manifest schema is unsupported")
    if route.get("status") not in {"diagnostic_only", "quality_selected"}:
        raise PublicDevelopmentPreviewError("route manifest status is unsupported")
    _safe_id(route.get("experiment_id"), "route experiment id")
    _safe_id(route.get("amendment_id"), "route amendment id")
    route_commit = _git_object(route.get("code_commit"), "route code commit")
    _git_object(route.get("code_tree"), "route code tree")
    for field in ("runner_sha256", "campaign_seal_sha256", "plan_bundle_sha256"):
        _sha256(route.get(field), f"route {field}")

    code = _mapping(receipt.code, "sealed source code binding")
    if (
        code.get("git_available") is not True
        or code.get("git_dirty") is not False
        or code.get("git_commit") != route_commit
    ):
        raise PublicDevelopmentPreviewError("route and sealed clean-code binding disagree")
    _sha256(code.get("source_tree_sha256"), "sealed source tree SHA-256")

    stages = _mapping(route.get("stages"), "route stages")
    if set(stages) != set(_ROUTE_STAGES):
        raise PublicDevelopmentPreviewError("route must select every stage exactly once")
    any_unmeasured = False
    selected_models: dict[str, str] = {}
    for route_stage in _ROUTE_STAGES:
        stage = _mapping(stages[route_stage], f"route stage {route_stage}")
        _exact_keys(stage, _ROUTE_STAGE_FIELDS, f"route stage {route_stage}")
        _sha256(stage.get("execution_selection_sha256"), "route execution selection")
        quality_hash = stage.get("quality_source_artifact_sha256")
        if quality_hash is not None:
            _sha256(quality_hash, "route quality source")
        model = _model_id(stage.get("selected_model"), "route selected model")
        basis = stage.get("selection_basis")
        if basis not in {"measured_quality", "technical_only_unmeasured_quality"}:
            raise PublicDevelopmentPreviewError("route selection basis is unsupported")
        if basis == "measured_quality" and quality_hash is None:
            raise PublicDevelopmentPreviewError("measured route stage lacks a quality source")
        any_unmeasured = any_unmeasured or basis == "technical_only_unmeasured_quality"
        selected_models[route_stage] = model

    expected_status = "diagnostic_only" if any_unmeasured else "quality_selected"
    if route.get("status") != expected_status:
        raise PublicDevelopmentPreviewError("route status disagrees with its stage bases")

    papers = _sequence(corpus.get("papers_detail"), "projected corpus papers")
    for paper in papers:
        projected = _mapping(paper, "projected corpus paper")
        for route_stage, public_stage in _ROUTE_TO_PUBLIC_STAGE.items():
            stage = _mapping(projected.get(public_stage), f"sealed {public_stage} stage")
            if public_stage != "extractor" and stage.get("enabled") is not True:
                raise PublicDevelopmentPreviewError("sealed run did not enable every route stage")
            if stage.get("model") != selected_models[route_stage]:
                raise PublicDevelopmentPreviewError("sealed run model disagrees with route")
            contract = _mapping(stage.get("request_contract"), "sealed request contract")
            privacy = _mapping(contract.get("privacy"), "sealed request privacy")
            routing = _mapping(contract.get("routing"), "sealed request routing")
            if (
                privacy.get("zdr") is not True
                or privacy.get("data_collection") != "deny"
                or routing.get("require_parameters") is not True
            ):
                raise PublicDevelopmentPreviewError("sealed run lost its privacy/routing contract")
            usage = _mapping(stage.get("usage"), "sealed stage usage")
            returned = _sequence(usage.get("models_returned"), "returned models")
            if any(item != selected_models[route_stage] for item in returned):
                raise PublicDevelopmentPreviewError("returned model disagrees with route")
    return dict(route)


def _find_precall_seal(run_root: Path) -> tuple[Path, dict[str, object]]:
    matches: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(run_root.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            continue
        try:
            raw = _strict_json_file(path, "sealed pre-call candidate")
        except PublicDevelopmentPreviewError:
            continue
        if isinstance(raw, Mapping) and raw.get("schema_version") == _PRECALL_SCHEMA_VERSION:
            matches.append((path, dict(raw)))
    if len(matches) != 1:
        raise PublicDevelopmentPreviewError(
            "sealed run must contain exactly one prospective pre-call seal"
        )
    return matches[0]


def _validate_precall_seal(
    *,
    run_root: Path,
    route_path: Path,
    route: Mapping[str, object],
    corpus_path_sha256: str,
    corpus_binding: Mapping[str, str],
    receipt: ValidatedCorpusReceipt,
) -> dict[str, object]:
    path, precall = _find_precall_seal(run_root)
    _exact_keys(precall, _PRECALL_FIELDS, "prospective pre-call seal")
    if precall.get("status") != "sealed_before_page_scoped_provider_execution":
        raise PublicDevelopmentPreviewError("prospective pre-call seal status is unsupported")
    if not isinstance(precall.get("sealed_at"), str) or not precall["sealed_at"]:
        raise PublicDevelopmentPreviewError("prospective pre-call timestamp is invalid")
    expected_self_hash = _sha256(precall.get("seal_sha256"), "pre-call self seal")
    self_payload = {key: value for key, value in precall.items() if key != "seal_sha256"}
    if sha256_bytes(canonical_json_bytes(self_payload)) != expected_self_hash:
        raise PublicDevelopmentPreviewError("prospective pre-call self seal is invalid")

    bindings = _mapping(precall.get("bindings"), "pre-call bindings")
    _exact_keys(bindings, _PRECALL_BINDING_FIELDS, "pre-call bindings")
    for key, value in bindings.items():
        _sha256(value, f"pre-call binding {key}")
    if bindings.get("route_selection_sha256") != sha256_file(route_path):
        raise PublicDevelopmentPreviewError("pre-call seal binds another route file")
    if bindings.get("corpus_config_sha256") != corpus_path_sha256:
        raise PublicDevelopmentPreviewError("pre-call seal binds another corpus file")
    if any(
        bindings.get(field) != corpus_binding[field]
        for field in ("corpus_spec_sha256", "paper_ids_sha256")
    ):
        raise PublicDevelopmentPreviewError("pre-call seal binds another corpus specification")
    if bindings.get("schema_sha256") != receipt.eee_schema["sha256"]:
        raise PublicDevelopmentPreviewError("pre-call seal binds another EEE schema")
    if bindings.get("campaign_seal_sha256") != route.get("campaign_seal_sha256"):
        raise PublicDevelopmentPreviewError("pre-call seal and route campaign bindings disagree")

    selected = _mapping(precall.get("selected_models"), "pre-call selected models")
    if set(selected) != set(_ROUTE_STAGES):
        raise PublicDevelopmentPreviewError("pre-call seal must select every stage exactly once")
    route_stages = _mapping(route.get("stages"), "route stages")
    for stage in _ROUTE_STAGES:
        route_stage = _mapping(route_stages[stage], f"route stage {stage}")
        if selected.get(stage) != route_stage.get("selected_model"):
            raise PublicDevelopmentPreviewError("pre-call and route model selections disagree")
    declared = _sequence(precall.get("declared_models"), "pre-call declared models")
    if (
        not declared
        or any(not isinstance(item, str) or _MODEL_ID.fullmatch(item) is None for item in declared)
        or len(set(declared)) != len(declared)
        or any(value not in declared for value in selected.values())
    ):
        raise PublicDevelopmentPreviewError("pre-call declared model set is invalid")

    privacy = _mapping(precall.get("privacy"), "pre-call privacy")
    _exact_keys(privacy, _PRECALL_PRIVACY_FIELDS, "pre-call privacy")
    forbidden_sent_key = (
        "human_labels_answers_scores_holdout_references_reviewer_ids_"
        "credentials_or_private_annotations_sent"
    )
    if (
        privacy.get("zdr") is not True
        or privacy.get("data_collection") != "deny"
        or privacy.get("label_blind_source_derived_scientific_payloads_only") is not True
        or privacy.get(forbidden_sent_key) is not False
    ):
        raise PublicDevelopmentPreviewError("pre-call privacy assertions are not release-safe")

    budget = _mapping(precall.get("budget"), "pre-call budget")
    _exact_keys(budget, _PRECALL_BUDGET_FIELDS, "pre-call budget")
    max_calls = _nonnegative_int(budget.get("max_structured_calls"), "pre-call call ceiling")
    max_cost = _nonnegative_number(budget.get("max_provider_cost_usd"), "pre-call cost ceiling")
    reservation = _nonnegative_number(
        budget.get("provider_call_cost_reservation_usd"), "pre-call reservation"
    )
    max_attempts = _nonnegative_int(
        budget.get("max_transport_attempts"), "pre-call attempt ceiling"
    )
    if not max_calls or not max_cost or not reservation or max_attempts < max_calls:
        raise PublicDevelopmentPreviewError("pre-call budget ceilings are invalid")

    raw_corpus = _mapping(
        _strict_json_file(run_root / "corpus-run.json", "sealed corpus run"),
        "sealed corpus run",
    )
    provider_budget = _mapping(raw_corpus.get("provider_budget"), "sealed provider budget")
    ledger_identity = _sha256(
        provider_budget.get("contract_sha256"), "sealed provider budget identity"
    )
    if provider_budget.get("ledger") != "private/provider-budget-ledger.jsonl":
        raise PublicDevelopmentPreviewError("sealed provider budget ledger path is unsupported")
    ledger_path = run_root / "private" / "provider-budget-ledger.jsonl"
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise PublicDevelopmentPreviewError("sealed provider budget ledger is unavailable")
    try:
        first_line = ledger_path.read_bytes().splitlines()[0]
    except (OSError, IndexError) as error:
        raise PublicDevelopmentPreviewError(
            "sealed provider budget ledger is unavailable"
        ) from error
    contract_event = _mapping(
        _strict_json_bytes(first_line, "sealed provider budget contract event"),
        "sealed provider budget contract event",
    )
    if (
        contract_event.get("event_type") != "contract"
        or contract_event.get("sequence") != 0
        or contract_event.get("contract_sha256") != ledger_identity
    ):
        raise PublicDevelopmentPreviewError("sealed provider budget contract event is invalid")
    full_budget_contract = _mapping(
        contract_event.get("contract"), "sealed full provider budget contract"
    )
    if bindings.get("provider_budget_contract_sha256") != _compact_json_sha256(
        full_budget_contract, "sealed full provider budget contract"
    ):
        raise PublicDevelopmentPreviewError(
            "sealed provider budget contract disagrees with the pre-call seal"
        )
    contract_limits = _mapping(
        full_budget_contract.get("limits"), "sealed provider budget contract limits"
    )
    expected_budget = {
        "max_structured_calls": max_calls,
        "max_cost_usd": max_cost,
        "cost_reservation_per_call_usd": reservation,
    }
    for field, expected in expected_budget.items():
        actual = provider_budget.get(field)
        agrees = actual == expected if isinstance(expected, int) else _same_number(actual, expected)
        if not agrees:
            raise PublicDevelopmentPreviewError(
                "sealed provider budget disagrees with pre-call seal"
            )
        contracted = contract_limits.get(field)
        contract_agrees = (
            contracted == expected
            if isinstance(expected, int)
            else _same_number(contracted, expected)
        )
        if not contract_agrees:
            raise PublicDevelopmentPreviewError(
                "sealed provider budget contract limits disagree with pre-call seal"
            )
    completed = _nonnegative_int(
        provider_budget.get("structured_calls_completed"), "completed structured calls"
    )
    if completed > max_calls:
        raise PublicDevelopmentPreviewError("sealed provider calls exceed the prospective ceiling")
    attempts = _nonnegative_int(
        _mapping(receipt.provider_telemetry, "receipt provider telemetry").get(
            "attempts_lower_bound"
        ),
        "recorded provider attempts",
    )
    if attempts > max_attempts:
        raise PublicDevelopmentPreviewError(
            "recorded provider attempts exceed the prospective ceiling"
        )

    command_argv = precall.get("command_argv")
    if not isinstance(command_argv, Sequence) or isinstance(command_argv, str | bytes | bytearray):
        raise PublicDevelopmentPreviewError("pre-call command contract is invalid")
    command_argv = list(command_argv)
    if (
        not command_argv
        or any(not isinstance(item, str) or not item for item in command_argv)
        or "run-corpus" not in command_argv
    ):
        raise PublicDevelopmentPreviewError("pre-call command contract is invalid")
    if max_attempts != max_calls * 4:
        raise PublicDevelopmentPreviewError("pre-call transport-attempt ceiling is invalid")

    outputs = _mapping(precall.get("outputs"), "pre-call outputs")
    _exact_keys(outputs, _PRECALL_OUTPUT_FIELDS, "pre-call outputs")
    if (
        not isinstance(outputs.get("raw_root"), str)
        or not outputs.get("raw_root")
        or not isinstance(outputs.get("sealed_root"), str)
        or not outputs.get("sealed_root")
        or outputs.get("raw_root") == outputs.get("sealed_root")
        or outputs.get("raw_root_was_fresh_before_seal") is not True
        or outputs.get("sealed_root_was_fresh_before_seal") is not True
    ):
        raise PublicDevelopmentPreviewError("pre-call output freshness contract is invalid")

    execution = _mapping(precall.get("execution_policy"), "pre-call execution policy")
    _exact_keys(execution, _PRECALL_EXECUTION_POLICY_FIELDS, "pre-call execution policy")
    seal_argv = _sequence(execution.get("seal_command_argv"), "pre-call seal command")
    if (
        execution.get("command_argv") != command_argv
        or not isinstance(execution.get("working_directory"), str)
        or not execution.get("working_directory")
        or not seal_argv
        or any(not isinstance(item, str) or not item for item in seal_argv)
        or "seal-run" not in seal_argv
        or execution.get("accepted_terminal_exit_codes") != [0, 1, 2, 3]
        or execution.get("completed_with_review_exit_code") != 2
        or execution.get("bounded_incomplete_exit_code") != 3
        or execution.get("generic_budget_doubling_forbidden") is not True
        or execution.get("copy_precall_seal_into_raw_root_before_execution") is not True
        or execution.get("seal_regardless_of_terminal_exit") is not True
        or execution.get("verify_seal_before_semantic_inspection") is not True
    ):
        raise PublicDevelopmentPreviewError("pre-call execution policy is invalid")
    return {
        "schema_version": _PRECALL_SCHEMA_VERSION,
        "status": precall["status"],
        "contract_sha256": expected_self_hash,
        "file_sha256": sha256_file(path),
        "route_selection_sha256": bindings["route_selection_sha256"],
        "corpus_config_sha256": bindings["corpus_config_sha256"],
        "provider_budget_contract_sha256": bindings["provider_budget_contract_sha256"],
        "selected_models": {stage: selected[stage] for stage in _ROUTE_STAGES},
        "budget": {
            "max_structured_calls": max_calls,
            "max_provider_cost_usd": max_cost,
            "provider_call_cost_reservation_usd": reservation,
            "max_transport_attempts": max_attempts,
        },
        "privacy": {
            "zdr": True,
            "data_collection": "deny",
            "label_blind_source_derived_scientific_payloads_only": True,
            "forbidden_private_or_labelled_payloads_sent": False,
        },
    }


def _validate_corpus_projection(
    *,
    spec: CorpusSpec,
    binding: Mapping[str, str],
    receipt: ValidatedCorpusReceipt,
    projected: Mapping[str, object],
    summary: Mapping[str, object],
    expected_paper_count: int,
) -> None:
    if (
        isinstance(expected_paper_count, bool)
        or not isinstance(expected_paper_count, int)
        or expected_paper_count < 1
    ):
        raise PublicDevelopmentPreviewError("expected paper count must be positive")
    if spec.evaluation_split != "development" or binding.get("evaluation_split") != "development":
        raise PublicDevelopmentPreviewError("public preview requires a development corpus")
    if (
        binding.get("corpus_id") != receipt.corpus_id
        or binding.get("corpus_spec_sha256") != receipt.corpus_spec_sha256
        or binding.get("paper_ids_sha256") != receipt.paper_ids_sha256
    ):
        raise PublicDevelopmentPreviewError("corpus specification and sealed receipt disagree")
    if (
        projected.get("corpus_id") != receipt.corpus_id
        or projected.get("status") != receipt.corpus_status
    ):
        raise PublicDevelopmentPreviewError("projected corpus and sealed receipt disagree")
    source_artifacts = _mapping(projected.get("source_artifacts"), "corpus source artifacts")
    if source_artifacts.get("corpus_run_sha256") != receipt.corpus_run_sha256:
        raise PublicDevelopmentPreviewError("projected corpus-run hash disagrees with receipt")
    if (
        source_artifacts.get("corpus_evaluation_sha256") is not None
        or projected.get("reference_evaluation") is not None
    ):
        raise PublicDevelopmentPreviewError(
            "public pre-human preview forbids reference evaluation material"
        )

    papers = _sequence(projected.get("papers_detail"), "projected corpus papers")
    if (
        len(spec.papers) != expected_paper_count
        or len(receipt.papers) != expected_paper_count
        or len(papers) != expected_paper_count
        or projected.get("papers") != expected_paper_count
        or projected.get("papers_total") != expected_paper_count
        or projected.get("papers_not_started") != 0
    ):
        raise PublicDevelopmentPreviewError("sealed corpus is not the exact completed population")

    projected_by_id = {
        _safe_id(_mapping(item, "projected paper").get("paper_id"), "projected paper id"): _mapping(
            item, "projected paper"
        )
        for item in papers
    }
    ordered_ids = [paper.paper_id for paper in spec.papers]
    if ordered_ids != [paper.paper_id for paper in receipt.papers] or ordered_ids != list(
        projected_by_id
    ):
        raise PublicDevelopmentPreviewError("paper order differs across bound inputs")
    for paper_spec, paper_receipt in zip(spec.papers, receipt.papers, strict=True):
        paper = projected_by_id[paper_spec.paper_id]
        if paper.get("title") != paper_spec.title or paper.get("status") != paper_receipt.status:
            raise PublicDevelopmentPreviewError("paper metadata/status differs across bound inputs")
        counts = _mapping(paper.get("counts"), "projected paper counts")
        if any(counts.get(key) != value for key, value in paper_receipt.counts.items()):
            raise PublicDevelopmentPreviewError("paper counts differ from sealed receipt")
        if any(
            _nonnegative_int(counts.get(key), f"paper reference count {key}")
            for key in _REFERENCE_COUNT_FIELDS
        ):
            raise PublicDevelopmentPreviewError(
                "public pre-human preview forbids reference-derived counts"
            )
        selected_pages = list(_sequence(paper.get("selected_pages"), "selected pages"))
        if paper_spec.include_pages:
            if selected_pages != paper_spec.include_pages:
                raise PublicDevelopmentPreviewError("configured and executed page scopes disagree")
        elif len(selected_pages) > paper_spec.max_result_pages:
            raise PublicDevelopmentPreviewError("automatic page selection exceeded its limit")
        if paper.get("code") != dict(receipt.code):
            raise PublicDevelopmentPreviewError("paper code binding differs from sealed receipt")

    totals = _mapping(projected.get("totals"), "projected corpus totals")
    if any(totals.get(key) != value for key, value in receipt.totals.items()):
        raise PublicDevelopmentPreviewError("corpus totals differ from sealed receipt")
    if any(
        _nonnegative_int(totals.get(key), f"corpus reference count {key}")
        for key in _REFERENCE_COUNT_FIELDS
    ):
        raise PublicDevelopmentPreviewError(
            "public pre-human preview forbids reference-derived counts"
        )
    if any(
        _nonnegative_int(totals.get(key), f"corpus total {key}") != 0
        for key in ("exported", "eee_records", "eee_schema_issues")
    ):
        raise PublicDevelopmentPreviewError("pre-human source run must contain zero canonical EEE")

    if (
        summary.get("current_sealed_receipt") is not True
        or _mapping(summary.get("scope"), "public summary scope").get(
            "independent_human_validation"
        )
        is not False
        or _mapping(summary.get("technical_health"), "public summary health").get("release_ready")
        is not False
        or _mapping(summary.get("canonical_eee"), "public summary canonical EEE").get("records")
        != 0
        or _mapping(
            _mapping(summary.get("export_provenance_modes"), "public summary provenance").get(
                "reviewed_derived"
            ),
            "reviewed projection",
        ).get("status")
        != "not_supplied"
    ):
        raise PublicDevelopmentPreviewError("public summary is not an honest pre-human projection")


def _validate_usage_reconciliation(
    *, receipt: ValidatedCorpusReceipt, corpus: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    stage_receipts = aggregate_stage_receipts(receipt)
    papers = [
        _mapping(item, "projected paper") for item in _sequence(corpus["papers_detail"], "papers")
    ]
    total_calls = 0
    for route_stage, public_stage in _ROUTE_TO_PUBLIC_STAGE.items():
        receipt_stage = _ROUTE_TO_RECEIPT_STAGE[route_stage]
        projected_calls = sum(
            _nonnegative_int(
                _mapping(_mapping(paper[public_stage], public_stage)["usage"], "stage usage").get(
                    "calls_attempted"
                ),
                "projected stage calls",
            )
            for paper in papers
        )
        if projected_calls != stage_receipts[receipt_stage]["completed_calls"]:
            raise PublicDevelopmentPreviewError("stage call accounting disagrees with receipt")
        total_calls += projected_calls
    telemetry = _mapping(receipt.provider_telemetry, "receipt provider telemetry")
    replayed_calls = _nonnegative_int(
        telemetry.get("recorded_structured_invocations"), "receipt provider calls"
    )
    if total_calls != replayed_calls:
        raise PublicDevelopmentPreviewError("replayed provider calls disagree across projections")

    provider = _mapping(corpus.get("provider_accounting"), "public provider accounting")
    started = _nonnegative_int(provider.get("structured_calls_started"), "provider calls started")
    completed = _nonnegative_int(
        provider.get("structured_calls_completed"), "provider calls completed"
    )
    pending = _nonnegative_int(provider.get("structured_calls_pending"), "provider calls pending")
    if started != completed + pending or pending != 0 or completed < replayed_calls:
        raise PublicDevelopmentPreviewError("terminal provider budget partition is invalid")
    coverage = provider.get("call_coverage")
    if coverage not in {"exhaustive", "lower_bound"}:
        raise PublicDevelopmentPreviewError("provider call coverage is unsupported")
    if coverage == "exhaustive":
        if completed != replayed_calls:
            raise PublicDevelopmentPreviewError("exhaustive provider call accounting disagrees")
        comparisons = {
            "provider_reported_cost_usd_lower_bound": "cost_usd_lower_bound",
            "provider_reported_input_tokens_lower_bound": "input_tokens_lower_bound",
            "provider_reported_output_tokens_lower_bound": "output_tokens_lower_bound",
            "provider_reported_reasoning_tokens_lower_bound": "reasoning_tokens_lower_bound",
            "provider_reported_total_tokens_lower_bound": "total_tokens_lower_bound",
        }
        for provider_key, receipt_key in comparisons.items():
            if not _same_number(provider.get(provider_key), telemetry.get(receipt_key)):
                raise PublicDevelopmentPreviewError("exhaustive provider totals disagree")
    return stage_receipts


def _public_http_uri(value: object, context: str) -> str | None:
    if value is None:
        return None
    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        return None
    if (
        not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise PublicDevelopmentPreviewError(f"{context} is not a safe public URI")
    return text


def _project_sources_without_local_uris(
    *, run_root: Path, spec: CorpusSpec, corpus: Mapping[str, object]
) -> dict[str, object]:
    projected_papers = {
        str(_mapping(item, "projected paper").get("paper_id")): _mapping(item, "projected paper")
        for item in _sequence(corpus.get("papers_detail"), "projected papers")
    }
    papers: list[dict[str, object]] = []
    for paper in spec.papers:
        manifest_path = run_root / paper.paper_id / "source-manifest.json"
        try:
            manifest = SourceManifest.model_validate(
                _strict_json_file(manifest_path, "source manifest")
            )
        except ValueError as error:
            raise PublicDevelopmentPreviewError("source manifest is invalid") from error
        if manifest.schema_version != "source-manifest/0.2":
            raise PublicDevelopmentPreviewError("source manifest schema is unsupported")
        if manifest.paper_id != paper.paper_id or manifest.title != paper.title:
            raise PublicDevelopmentPreviewError("source manifest paper metadata disagrees")
        projected_run = projected_papers[paper.paper_id]
        if projected_run.get("source_manifest_sha256") != sha256_file(manifest_path):
            raise PublicDevelopmentPreviewError("source manifest hash disagrees with sealed run")
        if manifest.doi is not None and paper.doi is not None and manifest.doi != paper.doi:
            raise PublicDevelopmentPreviewError("source manifest DOI disagrees with corpus")
        if (
            manifest.arxiv_id is not None
            and paper.arxiv_id is not None
            and manifest.arxiv_id != paper.arxiv_id
        ):
            raise PublicDevelopmentPreviewError("source manifest arXiv ID disagrees with corpus")

        sources: list[dict[str, object]] = []
        for source in manifest.sources:
            source_id = _safe_id(source.source_id, "source id")
            if source.sha256 is not None:
                _sha256(source.sha256, "source SHA-256")
            if source.git_commit is not None:
                _git_object(source.git_commit.casefold(), "source repository commit")
            sources.append(
                {
                    "source_id": source_id,
                    "role": str(source.role),
                    "public_original_uri": _public_http_uri(
                        source.original_uri, "source original URI"
                    ),
                    "public_resolved_uri": _public_http_uri(
                        source.resolved_uri, "source resolved URI"
                    ),
                    "retrieved_at": source.retrieved_at.isoformat().replace("+00:00", "Z"),
                    "sha256": source.sha256,
                    "byte_size": source.byte_size,
                    "media_type": source.media_type,
                    "git_commit": source.git_commit,
                    "access_status": str(source.access_status),
                    "license_disposition": str(source.license_disposition),
                }
            )
        papers.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "venue": paper.venue,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "acm_url": _public_http_uri(paper.acm_url, "paper ACM URL"),
                "pdf_url": _public_http_uri(paper.pdf_url, "paper PDF URL"),
                "supplement_urls": [
                    _public_http_uri(item, "paper supplement URL") for item in paper.supplement_urls
                ],
                "repository_url": _public_http_uri(paper.repository_url, "paper repository URL"),
                "repository_commit": paper.repository_commit,
                "perspective_role": paper.perspective_role,
                "page_scope": {
                    "selection_mode": (
                        "configured_page_scope" if paper.include_pages else "automatic_selector"
                    ),
                    "configured_include_pages": list(paper.include_pages),
                    "selected_pages": list(
                        _sequence(projected_run.get("selected_pages"), "selected pages")
                    ),
                    "max_result_pages": paper.max_result_pages,
                },
                "source_manifest_sha256": projected_run["source_manifest_sha256"],
                "sources": sources,
            }
        )
    return {
        "schema_version": PUBLIC_SOURCE_INVENTORY_SCHEMA_VERSION,
        "corpus_id": spec.corpus_id,
        "evaluation_split": spec.evaluation_split,
        "papers": papers,
    }


def _build_usage(
    *,
    receipt: ValidatedCorpusReceipt,
    corpus: Mapping[str, object],
    stage_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    paper_receipts = {paper.paper_id: paper for paper in receipt.papers}
    paper_accounting: list[dict[str, object]] = []
    for raw_paper in _sequence(corpus.get("papers_detail"), "projected papers"):
        paper = _mapping(raw_paper, "projected paper")
        paper_id = str(paper["paper_id"])
        stages: dict[str, object] = {}
        for route_stage, public_stage in _ROUTE_TO_PUBLIC_STAGE.items():
            stage = _mapping(paper[public_stage], f"paper {public_stage}")
            stages[route_stage] = {
                "status": paper_receipts[paper_id]
                .stages[_ROUTE_TO_RECEIPT_STAGE[route_stage]]
                .status,
                "selected_model": stage.get("model"),
                "execution": stage.get("execution"),
                "usage": stage.get("usage"),
            }
        paper_accounting.append(
            {
                "paper_id": paper_id,
                "status": paper.get("status"),
                "wall_clock_seconds": paper.get("wall_clock_seconds"),
                "stages": stages,
            }
        )
    operational_accounting: dict[str, object] = {
        "wall_clock_seconds": receipt.operations.get("wall_clock_seconds"),
        "stages": {},
    }
    operational_stages = operational_accounting["stages"]
    assert isinstance(operational_stages, dict)
    for route_stage, public_stage in _ROUTE_TO_PUBLIC_STAGE.items():
        source = dict(_mapping(receipt.operations.get(public_stage), "stage operations"))
        source["completed_calls"] = source.pop("calls")
        operational_stages[route_stage] = source
    return {
        "schema_version": PUBLIC_USAGE_SCHEMA_VERSION,
        "accounting_scope": "sealed_development_source_run_only",
        "statement": (
            "Provider-reported cost and token values remain lower bounds whenever completed "
            "calls lack telemetry. Qualification-campaign and source-run spend are not combined."
        ),
        "provider_budget": corpus.get("provider_accounting"),
        "replayed_provider_usage": dict(receipt.provider_telemetry),
        "stage_receipts": {
            stage: dict(stage_receipts[_ROUTE_TO_RECEIPT_STAGE[stage]]) for stage in _ROUTE_STAGES
        },
        "operational_accounting": operational_accounting,
        "paper_accounting": paper_accounting,
        "qualification_campaign_usage": {
            "status": "separate_not_included",
            "combined_with_source_run": False,
        },
    }


def _count(counter: Mapping[str, int], key: str) -> int:
    value = counter.get(key, 0)
    return _nonnegative_int(value, f"count {key}")


def _build_evidence_map(
    *, receipt: ValidatedCorpusReceipt, corpus: Mapping[str, object], spec: CorpusSpec
) -> dict[str, object]:
    projected = {
        str(_mapping(item, "projected paper")["paper_id"]): _mapping(item, "projected paper")
        for item in _sequence(corpus.get("papers_detail"), "projected papers")
    }
    specs = {paper.paper_id: paper for paper in spec.papers}
    rows: list[dict[str, object]] = []
    for paper in receipt.papers:
        run = projected[paper.paper_id]
        paper_spec = specs[paper.paper_id]
        counts = _mapping(run.get("counts"), "paper counts")
        verifier_execution = _mapping(
            _mapping(run.get("verifier"), "paper verifier").get("execution"),
            "verifier execution",
        )
        accepted_by_model_gates = _nonnegative_int(
            counts.get("verifier_accepts"), "verifier accepts"
        )
        candidates = _nonnegative_int(counts.get("candidates"), "candidate count")
        rows.append(
            {
                "paper_id": paper.paper_id,
                "title": paper_spec.title,
                "year": paper_spec.year,
                "venue": paper_spec.venue,
                "perspective_role": paper_spec.perspective_role,
                "page_scope": {
                    "selection_mode": (
                        "configured_page_scope"
                        if paper_spec.include_pages
                        else "automatic_selector"
                    ),
                    "selected_pages": list(_sequence(run.get("selected_pages"), "selected pages")),
                },
                "technical": {
                    "paper_status": paper.status,
                    "needs_review": paper.needs_review,
                    "review_reasons": list(paper.review_reasons),
                    "stage_status": {
                        stage: paper.stages[_ROUTE_TO_RECEIPT_STAGE[stage]].status
                        for stage in _ROUTE_STAGES
                    },
                },
                "candidate_layer": {
                    "model_proposals": _count(paper.lineage_counts, "proposals"),
                    "candidate_occurrences": _count(paper.lineage_counts, "candidate_occurrences"),
                    "deduplicated_candidates": candidates,
                    "duplicates_removed": _nonnegative_int(
                        counts.get("duplicates_removed"), "duplicates removed"
                    ),
                    "primary_result_candidates": _nonnegative_int(
                        counts.get("primary_results"), "primary results"
                    ),
                    "requires_human_review": _count(paper.export_status_counts, "needs_review"),
                    "deterministically_not_eligible": _count(
                        paper.export_status_counts, "not_eligible"
                    ),
                },
                "tuple_gate": {
                    "selected": _nonnegative_int(
                        counts.get("tuple_candidates"), "tuple candidates"
                    ),
                    "passed": _nonnegative_int(counts.get("tuple_passed"), "tuple passed"),
                    "routed_to_review": _nonnegative_int(
                        counts.get("tuple_review"), "tuple review"
                    ),
                    "unsupported_subset": _nonnegative_int(
                        counts.get("tuple_unsupported"), "tuple unsupported"
                    ),
                    "failed_subset": _nonnegative_int(counts.get("tuple_failed"), "tuple failed"),
                },
                "independent_verifier": {
                    "selected": _nonnegative_int(
                        verifier_execution.get("candidates_selected"), "verifier selected"
                    ),
                    "verified": _nonnegative_int(counts.get("verifications"), "verifications"),
                    "accepts": accepted_by_model_gates,
                    "rejects": _nonnegative_int(counts.get("verifier_rejects"), "verifier rejects"),
                    "reviews": _nonnegative_int(counts.get("verifier_reviews"), "verifier reviews"),
                    "failed_subset": _nonnegative_int(
                        counts.get("verifier_failed"), "verifier failures"
                    ),
                },
                "producer_origin": {
                    "selected": _nonnegative_int(
                        counts.get("origin_candidates"), "origin candidates"
                    ),
                    "external": _nonnegative_int(counts.get("origin_external"), "external origins"),
                    "unresolved": _nonnegative_int(
                        counts.get("origin_unresolved"), "unresolved origins"
                    ),
                    "no_signal": _nonnegative_int(
                        counts.get("origin_no_signal"), "no-signal origins"
                    ),
                    "positive_review_only_subset": _nonnegative_int(
                        counts.get("origin_positive_review_only"), "positive review-only origins"
                    ),
                    "failed_subset": _nonnegative_int(
                        counts.get("origin_failed"), "origin failures"
                    ),
                    "automatic_positive_promotion": False,
                },
                "review_boundary": {
                    "accepted_by_model_gates": accepted_by_model_gates,
                    "not_accepted_by_complete_model_chain": candidates - accepted_by_model_gates,
                    "withheld_from_canonical_eee": candidates,
                    "human_review_decisions_completed": 0,
                    "canonical_eee_records": 0,
                    "final_reviewed_records": None,
                    "status": "model_proposed_human_review_pending",
                },
            }
        )
    return {
        "schema_version": PUBLIC_EVIDENCE_MAP_SCHEMA_VERSION,
        "status": "model_proposed_human_review_pending",
        "statement": (
            "Candidate-layer development preview. Accepted-by-model-gates means an independent "
            "verifier accept after tuple gating; it is not human acceptance or export authority."
        ),
        "scope": {
            "corpus_id": receipt.corpus_id,
            "evaluation_split": "development",
            "papers": len(rows),
            "holdout": False,
            "whole_paper_evaluation": False,
            "independent_human_validation": False,
        },
        "papers": rows,
    }


def _render_evidence_map(value: Mapping[str, object]) -> str:
    rows = []
    for raw in _sequence(value.get("papers"), "evidence-map papers"):
        paper = _mapping(raw, "evidence-map paper")
        scope = _mapping(paper["page_scope"], "evidence-map page scope")
        technical = _mapping(paper["technical"], "evidence-map technical status")
        candidate = _mapping(paper["candidate_layer"], "evidence-map candidate layer")
        tuple_gate = _mapping(paper["tuple_gate"], "evidence-map tuple gate")
        verifier = _mapping(paper["independent_verifier"], "evidence-map verifier")
        origin = _mapping(paper["producer_origin"], "evidence-map origin")
        review = _mapping(paper["review_boundary"], "evidence-map review boundary")
        pages = ", ".join(str(item) for item in _sequence(scope["selected_pages"], "pages"))
        rows.append(
            "<tr>"
            f"<th><span>{escape(str(paper['title']))}</span>"
            f"<small>{escape(str(paper['paper_id']))} · {escape(str(paper['perspective_role']))}"
            f" · pages {escape(pages)}</small></th>"
            f"<td>{escape(str(technical['paper_status']))}</td>"
            f"<td>{candidate['model_proposals']} → {candidate['deduplicated_candidates']}"
            f"<small>{candidate['requires_human_review']} review; "
            f"{candidate['deterministically_not_eligible']} not eligible</small></td>"
            f"<td>{tuple_gate['passed']} pass / {tuple_gate['routed_to_review']} review</td>"
            f"<td>{verifier['accepts']} accept / {verifier['rejects']} reject / "
            f"{verifier['reviews']} review</td>"
            f"<td>{origin['external']} external / {origin['unresolved']} unresolved / "
            f"{origin['no_signal']} no signal</td>"
            f"<td>{review['accepted_by_model_gates']} model-gate accepts"
            f"<small>0 canonical · human review pending</small></td>"
            "</tr>"
        )
    body = "".join(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Candidate-layer toxicity and Perspective evidence map</title>
<style>
:root {{ color-scheme: light; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 2rem; color: #172033; background: #f6f7fb; }}
main {{ max-width: 96rem; margin: auto; }}
h1 {{ margin-bottom: .35rem; }}
.notice {{ padding: 1rem; border-left: .4rem solid #a35d00; background: #fff4de; }}
.table-wrap {{ overflow-x: auto; margin-top: 1.25rem; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
th, td {{ padding: .75rem; border: 1px solid #d9deea; text-align: left; vertical-align: top; }}
thead th {{ background: #e9edf6; }}
tbody th {{ min-width: 18rem; }}
span, small {{ display: block; }}
small {{ color: #566078; margin-top: .25rem; }}
</style>
</head>
<body>
<main>
<h1>Candidate-layer toxicity and Perspective evidence map</h1>
<p class="notice"><strong>Model-proposed; human review pending.</strong> Model-gate accepts are
not canonical EEE and do not establish human acceptance or paper-produced origin.</p>
<div class="table-wrap"><table>
<thead><tr><th>Paper and scope</th><th>Technical</th><th>Candidates</th><th>Tuple gate</th>
<th>Independent verifier</th><th>Producer origin</th><th>Review boundary</th></tr></thead>
<tbody>{body}</tbody>
</table></div>
</main>
</body>
</html>
"""


def _build_manifest(
    *,
    bundle_id: str,
    receipt: ValidatedCorpusReceipt,
    route: Mapping[str, object],
    precall: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": PUBLIC_DEVELOPMENT_PREVIEW_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "artifact_status": {
            "classification": "pre_human_development_preview",
            "release_ready": False,
            "human_review_status": "pending",
            "canonical_real_paper_eee_available": False,
        },
        "statement": (
            "Quote-free inspected-development preview of a sealed source run. Model proposals "
            "and model-gate outcomes remain non-authoritative until candidate-bound human review."
        ),
        "source_run": {
            "corpus_id": receipt.corpus_id,
            "evaluation_split": "development",
            "paper_count": len(receipt.papers),
            "generated_at": receipt.generated_at,
            "corpus_spec_sha256": receipt.corpus_spec_sha256,
            "paper_ids_sha256": receipt.paper_ids_sha256,
            "corpus_run_sha256": receipt.corpus_run_sha256,
            "run_seal": {
                "schema_version": receipt.seal.schema_version,
                "seal_sha256": receipt.seal.seal_sha256,
                "tree_sha256": receipt.seal.tree_sha256,
                "file_count": receipt.seal.file_count,
                "total_bytes": receipt.seal.total_bytes,
            },
            "code": dict(receipt.code),
        },
        "prospective_execution": {
            "route_schema_version": route["schema_version"],
            "route_status": route["status"],
            "route_selection_sha256": precall["route_selection_sha256"],
            "precall_contract_sha256": precall["contract_sha256"],
            "precall_file_sha256": precall["file_sha256"],
            "binding_status": "prospectively_bound_to_sealed_source_run",
            "selected_models": precall["selected_models"],
            "privacy": precall["privacy"],
            "budget": precall["budget"],
        },
        "outputs": {
            "automatic_source_run": {
                "status": "model_proposed_human_review_pending",
                "canonical_eee_records": 0,
                "present": True,
            },
            "reviewed_derived": {
                "status": "pending_human_review",
                "expected_relative_path": "reviewed/eee",
                "records": None,
                "present": False,
            },
            "reviewed_summary": {
                "status": "pending_human_review",
                "expected_relative_path": "reviewed/public-summary.json",
                "present": False,
            },
        },
        "files": sorted(_PREVIEW_FILES - {"SHA256SUMS"}),
        "limitations": [
            "The papers are disclosed, previously inspected development material.",
            "The page-scoped run is not a whole-paper evaluation.",
            "Technical eligibility and model-gate acceptance are not quality validation.",
            "No independent human validation or canonical real-paper EEE is included.",
            "Qualification-campaign usage is separate from source-run usage.",
        ],
    }


def _build_verification(
    *, receipt: ValidatedCorpusReceipt, precall: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema_version": PUBLIC_PREVIEW_VERIFICATION_SCHEMA_VERSION,
        "status": "verified_pre_human_preview",
        "checks": {
            "source_run_seal_verified": True,
            "five_stage_receipt_replayed": True,
            "corpus_specification_rederived": True,
            "prospective_route_bound": True,
            "precall_privacy_contract_verified": True,
            "provider_accounting_reconciled": True,
            "automatic_canonical_eee_zero": True,
            "human_review_not_supplied": True,
            "allowlist_and_privacy_scan_passed_before_checksums": True,
        },
        "bindings": {
            "source_run_seal_sha256": receipt.seal.seal_sha256,
            "source_run_tree_sha256": receipt.seal.tree_sha256,
            "corpus_run_sha256": receipt.corpus_run_sha256,
            "corpus_spec_sha256": receipt.corpus_spec_sha256,
            "paper_ids_sha256": receipt.paper_ids_sha256,
            "route_selection_sha256": precall["route_selection_sha256"],
            "precall_contract_sha256": precall["contract_sha256"],
            "precall_file_sha256": precall["file_sha256"],
        },
        "release_ready": False,
    }


def _render_readme(manifest: Mapping[str, object], evidence_map: Mapping[str, object]) -> str:
    source = _mapping(manifest["source_run"], "manifest source run")
    status = _mapping(manifest["artifact_status"], "manifest artifact status")
    scope = _mapping(evidence_map["scope"], "evidence-map scope")
    return f"""# Proceedings-to-EEE inspected-development preview

This bundle is a quote-free, pre-human preview for **{source["paper_count"]}** disclosed,
previously inspected development papers. It is model-proposed and human review is pending.
It is not a holdout, whole-paper evaluation, independent validation, or proof of generalization.

The narrow community question is whether a privacy-constrained staged pipeline can surface
auditable evaluation-result candidates from Perspective/toxicity papers while preserving the
boundary between model proposals, producer-origin evidence, and later human authority.

The candidate-layer evidence map reports aggregate pipeline flow only. An
“accepted-by-model-gates” count means an independent verifier accepted a candidate after tuple
gating. It does not mean a human accepted it, and it is not authority for canonical EEE.

Current publication boundary:

- Release ready: `{str(status["release_ready"]).lower()}`
- Canonical real-paper EEE available: `{str(status["canonical_real_paper_eee_available"]).lower()}`
- Human review: `{status["human_review_status"]}`
- Evaluation split: `{scope["evaluation_split"]}`
- Paper count: `{scope["papers"]}`

## Contents

- `publication-manifest.json`: source-run, prospective route, budget, privacy, and
  pending-output bindings.
- `run-summary.json`: existing quote-free sealed-current development summary.
- `corpus.json`: allowlisted per-paper run, stage, model, and aggregate usage projection.
- `route-selection.json`: exact validated five-stage diagnostic route.
- `sources.json`: bibliographic identity, page scope, and content hashes without local paths.
- `usage.json`: source-run calls, models, providers, tokens, cost, latency, retries, and failures.
- `evidence-map.json` and `evidence-map.html`: static aggregate candidate-flow preview.
- `verification.json`: checks performed before publication.
- `SHA256SUMS`: deterministic checksum inventory for every other file.

No source PDFs, source-layout text, evidence quotations, candidate records, exact tuples, provider
requests or responses, request IDs, credentials, private annotations, reviewer identities, or human
decisions are included. Final reviewed outputs are represented as pending and are never fabricated.
"""


def _audit_preview_contents(contents: Mapping[str, bytes], *, expect_checksums: bool) -> None:
    expected = _PREVIEW_FILES if expect_checksums else _PREVIEW_FILES - {"SHA256SUMS"}
    if set(contents) != expected:
        raise PublicDevelopmentPreviewError("public preview file allowlist disagrees")
    for name, content in sorted(contents.items()):
        if len(content) > _MAX_PUBLIC_FILE_BYTES:
            raise PublicDevelopmentPreviewError("public preview file exceeds its size limit")
        if content.startswith(b"%PDF-") or content.startswith(b"PK\x03\x04") or b"\x00" in content:
            raise PublicDevelopmentPreviewError("public preview contains a binary/source payload")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PublicDevelopmentPreviewError("public preview contains non-UTF-8 data") from error
        _scan_text(text, name)
        if name in _JSON_FILES:
            parsed = _strict_json_bytes(content, name)
            _audit_public_value(parsed, name)
        if name.endswith(".html") and (
            re.search(r"<script\b", text, re.IGNORECASE)
            or re.search(r"<(?:img|link|iframe)\b", text, re.IGNORECASE)
        ):
            raise PublicDevelopmentPreviewError("public preview HTML is not self-contained/static")


def _write_checksums(root: Path) -> None:
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    content = "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths).encode("utf-8")
    atomic_write_bytes(root / "SHA256SUMS", content)


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        try:
            written = os.write(descriptor, content[offset:])
        except OSError as error:
            raise PublicDevelopmentPreviewError("public preview could not be published") from error
        if written < 1:
            raise PublicDevelopmentPreviewError("public preview write made no progress")
        offset += written


def _publish_contents_exclusive(
    *, parent: Path, bundle_id: str, contents: Mapping[str, bytes]
) -> Path:
    """Publish validated bytes without ever replacing an existing filesystem entry."""

    parent_fd = _open_directory_fd(parent, "public preview output root")
    parent_stat = os.fstat(parent_fd)
    parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
    destination_fd: int | None = None
    destination_identity: tuple[int, int] | None = None
    created = False
    try:
        try:
            os.mkdir(bundle_id, mode=0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError as error:
            raise PublicDevelopmentPreviewError(
                "public preview destination appeared during build"
            ) from error
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        destination_fd = os.open(bundle_id, flags, dir_fd=parent_fd)
        destination_stat = os.fstat(destination_fd)
        destination_identity = (destination_stat.st_dev, destination_stat.st_ino)
        path_stat = os.stat(bundle_id, dir_fd=parent_fd, follow_symlinks=False)
        if destination_identity != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise PublicDevelopmentPreviewError("public preview destination changed during build")
        for name, content in sorted(contents.items()):
            file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(name, file_flags, 0o600, dir_fd=destination_fd)
            try:
                _write_all(descriptor, content)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        if set(os.listdir(destination_fd)) != set(contents):
            raise PublicDevelopmentPreviewError(
                "public preview destination changed during publication"
            )
        for name, expected in sorted(contents.items()):
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=destination_fd,
            )
            try:
                before = os.fstat(descriptor)
                actual = _read_descriptor(descriptor, f"published public preview file {name}")
                after = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                    or actual != expected
                ):
                    raise PublicDevelopmentPreviewError(
                        "public preview destination changed during publication"
                    )
            finally:
                os.close(descriptor)
        os.fsync(destination_fd)
        final_stat = os.stat(bundle_id, dir_fd=parent_fd, follow_symlinks=False)
        if destination_identity != (final_stat.st_dev, final_stat.st_ino):
            raise PublicDevelopmentPreviewError("public preview destination changed during build")
        current_parent_fd = _open_directory_fd(parent, "public preview output root")
        try:
            current_parent_stat = os.fstat(current_parent_fd)
            if parent_identity != (current_parent_stat.st_dev, current_parent_stat.st_ino):
                raise PublicDevelopmentPreviewError(
                    "public preview output root changed during build"
                )
        finally:
            os.close(current_parent_fd)
        os.fchmod(destination_fd, 0o755)
        os.fsync(destination_fd)
        os.fsync(parent_fd)
    except BaseException:
        if destination_fd is not None:
            for name in contents:
                try:
                    os.unlink(name, dir_fd=destination_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
        if created and destination_identity is not None:
            with suppress(OSError):
                current = os.stat(bundle_id, dir_fd=parent_fd, follow_symlinks=False)
                if destination_identity == (current.st_dev, current.st_ino):
                    os.rmdir(bundle_id, dir_fd=parent_fd)
        raise
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(parent_fd)
    return parent / bundle_id


def _verify_checksums(contents: Mapping[str, bytes]) -> None:
    try:
        lines = contents["SHA256SUMS"].decode("utf-8").splitlines()
    except (KeyError, UnicodeDecodeError) as error:
        raise PublicDevelopmentPreviewError("public preview checksums are unreadable") from error
    expected_names = sorted(_PREVIEW_FILES - {"SHA256SUMS"})
    if len(lines) != len(expected_names):
        raise PublicDevelopmentPreviewError("public preview checksum inventory is incomplete")
    parsed: list[tuple[str, str]] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match is None:
            raise PublicDevelopmentPreviewError("public preview checksum line is invalid")
        parsed.append((match.group(1), match.group(2)))
    if [name for _, name in parsed] != expected_names:
        raise PublicDevelopmentPreviewError("public preview checksum order/allowlist disagrees")
    for digest, name in parsed:
        if sha256_bytes(contents[name]) != digest:
            raise PublicDevelopmentPreviewError("public preview checksum verification failed")
    canonical = "".join(f"{digest}  {name}\n" for digest, name in parsed).encode("utf-8")
    if contents["SHA256SUMS"] != canonical:
        raise PublicDevelopmentPreviewError("public preview checksum rendering is not canonical")


def _exact_nonnegative_counts(
    value: object,
    fields: set[str],
    context: str,
) -> dict[str, int]:
    counts = _mapping(value, context)
    _exact_keys(counts, fields, context)
    return {key: _nonnegative_int(counts[key], f"{context}.{key}") for key in fields}


def _validate_hash_list(value: object, context: str) -> list[str]:
    items = list(_sequence(value, context))
    for item in items:
        _sha256(item, context)
    if items != sorted(set(items)):
        raise PublicDevelopmentPreviewError(f"{context} must be unique and sorted")
    return [str(item) for item in items]


def _validate_stage_receipt(value: object, context: str, *, paper_count: int) -> None:
    receipt = _mapping(value, context)
    _exact_keys(receipt, _STAGE_RECEIPT_FIELDS, context)
    if receipt.get("status") not in {"validated", "partial_failure", "not_run"}:
        raise PublicDevelopmentPreviewError(f"{context} status is invalid")
    papers_validated = _nonnegative_int(receipt.get("papers_validated"), context)
    papers_partial = _nonnegative_int(receipt.get("papers_partial_failure"), context)
    papers_not_run = _nonnegative_int(receipt.get("papers_not_run"), context)
    if papers_validated + papers_partial + papers_not_run != paper_count:
        raise PublicDevelopmentPreviewError(f"{context} paper partition is invalid")
    for field in ("completed_calls", "selected_items", "failed_items"):
        _nonnegative_int(receipt.get(field), f"{context}.{field}")
    _validate_hash_list(receipt.get("checkpoint_contract_sha256s"), context)
    _validate_hash_list(receipt.get("sidecar_sha256s"), context)


def _validate_public_telemetry(value: object, context: str) -> dict[str, object]:
    telemetry = _mapping(value, context)
    _exact_keys(telemetry, _TELEMETRY_FIELDS, context)
    if not isinstance(telemetry.get("basis"), str) or not telemetry.get("basis"):
        raise PublicDevelopmentPreviewError(f"{context} basis is invalid")
    calls = _nonnegative_int(telemetry.get("recorded_structured_invocations"), f"{context} calls")
    for field in (
        "cost_reported_calls",
        "input_tokens_reported_calls",
        "output_tokens_reported_calls",
        "reasoning_tokens_reported_calls",
        "total_tokens_reported_calls",
    ):
        if _nonnegative_int(telemetry.get(field), f"{context}.{field}") > calls:
            raise PublicDevelopmentPreviewError(f"{context} reports telemetry for too many calls")
    for field in (
        "cost_usd_lower_bound",
        "input_tokens_lower_bound",
        "output_tokens_lower_bound",
        "reasoning_tokens_lower_bound",
        "total_tokens_lower_bound",
        "latency_seconds_total",
    ):
        _nonnegative_number(telemetry.get(field), f"{context}.{field}")
    for field in ("latency_seconds_max", "latency_seconds_mean"):
        item = telemetry.get(field)
        if item is None:
            if calls:
                raise PublicDevelopmentPreviewError(f"{context}.{field} must be numeric")
        else:
            _nonnegative_number(item, f"{context}.{field}")
    retries = _nonnegative_int(telemetry.get("retries_lower_bound"), f"{context} retries")
    attempts = _nonnegative_int(telemetry.get("attempts_lower_bound"), f"{context} attempts")
    if attempts != calls + retries:
        raise PublicDevelopmentPreviewError(f"{context} attempt accounting is invalid")
    latency_total = float(telemetry["latency_seconds_total"])
    latency_max = telemetry.get("latency_seconds_max")
    latency_mean = telemetry.get("latency_seconds_mean")
    if calls:
        if (
            not isinstance(latency_mean, int | float)
            or not math.isclose(
                float(latency_mean),
                latency_total / calls,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not isinstance(latency_max, int | float)
            or float(latency_max) > latency_total + 1e-6
        ):
            raise PublicDevelopmentPreviewError(f"{context} latency accounting is invalid")
    elif latency_total != 0 or latency_max is not None or latency_mean is not None:
        raise PublicDevelopmentPreviewError(f"{context} zero-call latency is invalid")
    return dict(telemetry)


def _validate_stage_usage(value: object, context: str, *, require_exact: bool) -> dict[str, object]:
    usage = _mapping(value, context)
    _exact_keys(usage, _STAGE_USAGE_FIELDS, context)
    calls = _nonnegative_int(usage.get("calls_attempted"), f"{context} calls")
    calls_field = _nonnegative_int(usage.get("calls_field_count"), f"{context} calls field")
    resumed = _nonnegative_int(usage.get("resumed_calls_field_count"), f"{context} resumed calls")
    if calls_field + resumed != calls or usage.get("call_accounting_basis") != (
        "calls_plus_resumed_calls"
    ):
        raise PublicDevelopmentPreviewError(f"{context} call partition is invalid")
    matches = _nonnegative_int(
        usage.get("model_returned_matches_requested_calls"), f"{context} model matches"
    )
    unverified = _nonnegative_int(
        usage.get("model_returned_unverified_calls"), f"{context} model unverified"
    )
    if matches + unverified != calls:
        raise PublicDevelopmentPreviewError(f"{context} returned-model partition is invalid")
    models = list(_sequence(usage.get("models_returned"), f"{context} models returned"))
    if models != sorted(set(models)):
        raise PublicDevelopmentPreviewError(f"{context} returned models are duplicated")
    for model in models:
        _model_id(model, f"{context} returned model")
    providers = list(_sequence(usage.get("providers_returned"), f"{context} providers returned"))
    if any(not isinstance(item, str) or not item for item in providers) or providers != sorted(
        set(providers)
    ):
        raise PublicDevelopmentPreviewError(f"{context} returned providers are invalid")
    if (not calls and (models or providers)) or (matches and not models):
        raise PublicDevelopmentPreviewError(f"{context} returned identities disagree with calls")

    for prefix in ("cost", "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
        reported = _nonnegative_int(usage.get(f"{prefix}_reported_calls"), context)
        missing = _nonnegative_int(usage.get(f"{prefix}_missing_calls"), context)
        if reported + missing != calls:
            raise PublicDevelopmentPreviewError(f"{context} {prefix} partition is invalid")
        lower = (
            _nonnegative_number(usage.get(f"{prefix}_usd_lower_bound"), context)
            if prefix == "cost"
            else _nonnegative_number(usage.get(f"{prefix}_lower_bound"), context)
        )
        exact_key = "cost_usd" if prefix == "cost" else prefix
        exact = usage.get(exact_key)
        if missing and exact is not None:
            raise PublicDevelopmentPreviewError(f"{context} incomplete {prefix} must be null")
        if require_exact and not missing and not _same_number(exact, lower):
            raise PublicDevelopmentPreviewError(f"{context} exact {prefix} disagrees")
        if not require_exact and exact is not None:
            raise PublicDevelopmentPreviewError(
                f"{context} lower-bound {prefix} must not be presented as exact"
            )
    attempts = _nonnegative_int(usage.get("attempts_lower_bound"), f"{context} attempts")
    retries = _nonnegative_int(usage.get("retries_lower_bound"), f"{context} retries")
    if attempts != calls + retries:
        raise PublicDevelopmentPreviewError(f"{context} attempt accounting is invalid")
    return dict(usage)


def _validate_request_contract(value: object, context: str) -> None:
    contract = _mapping(value, context)
    _exact_keys(contract, _REQUEST_CONTRACT_FIELDS, context)
    if contract.get("schema_version") != "provider-request-contract/0.2":
        raise PublicDevelopmentPreviewError(f"{context} schema is unsupported")
    _nonnegative_int(contract.get("max_tokens"), f"{context} max tokens")
    seed = contract.get("seed")
    if seed is not None:
        _nonnegative_int(seed, f"{context} seed")
    if contract.get("completion_token_parameter") not in {
        "max_tokens",
        "max_completion_tokens",
    }:
        raise PublicDevelopmentPreviewError(f"{context} completion-token parameter is invalid")
    schema = _mapping(contract.get("schema"), f"{context} schema")
    privacy = _mapping(contract.get("privacy"), f"{context} privacy")
    routing = _mapping(contract.get("routing"), f"{context} routing")
    _exact_keys(schema, _REQUEST_SCHEMA_FIELDS, f"{context} schema")
    _exact_keys(privacy, _REQUEST_PRIVACY_FIELDS, f"{context} privacy")
    _exact_keys(routing, _REQUEST_ROUTING_FIELDS, f"{context} routing")
    if (
        schema.get("response_format") != "json_schema"
        or schema.get("schema_strict") is not True
        or not isinstance(schema.get("schema_name"), str)
        or not schema.get("schema_name")
        or _SHA256.fullmatch(str(schema.get("schema_sha256"))) is None
        or privacy != {"zdr": True, "data_collection": "deny"}
        or routing != {"require_parameters": True}
    ):
        raise PublicDevelopmentPreviewError(f"{context} is not release-safe")


def _validate_public_corpus_schema(corpus: Mapping[str, object]) -> list[str]:
    _exact_keys(corpus, _CORPUS_TOP_FIELDS, "public corpus")
    if corpus.get("schema_version") != "corpus-run/0.3":
        raise PublicDevelopmentPreviewError("public corpus schema is unsupported")
    _safe_id(corpus.get("corpus_id"), "public corpus id")
    if corpus.get("status") not in {"success", "partial_failure", "error"}:
        raise PublicDevelopmentPreviewError("public corpus status is invalid")
    if not isinstance(corpus.get("generated_at"), str) or not corpus.get("generated_at"):
        raise PublicDevelopmentPreviewError("public corpus timestamp is invalid")
    partitions = {
        key: _nonnegative_int(corpus.get(key), f"public corpus {key}")
        for key in (
            "papers",
            "papers_total",
            "papers_not_started",
            "papers_succeeded",
            "papers_failed",
            "papers_bounded_incomplete",
            "papers_with_eee",
            "papers_without_candidates",
            "papers_without_eee",
            "papers_needing_review",
        )
    }
    if (
        partitions["papers"] != partitions["papers_total"]
        or partitions["papers_not_started"] != 0
        or partitions["papers_bounded_incomplete"] != 0
        or partitions["papers_succeeded"] + partitions["papers_failed"] != partitions["papers"]
    ):
        raise PublicDevelopmentPreviewError("public corpus paper partition is invalid")
    source = _mapping(corpus.get("source_artifacts"), "public corpus source artifacts")
    _exact_keys(source, _CORPUS_SOURCE_ARTIFACT_FIELDS, "public corpus source artifacts")
    _sha256(source.get("corpus_run_sha256"), "public corpus-run hash")
    if (
        source.get("corpus_evaluation_sha256") is not None
        or corpus.get("reference_evaluation") is not None
    ):
        raise PublicDevelopmentPreviewError(
            "public pre-human preview contains reference evaluation material"
        )
    totals = _exact_nonnegative_counts(corpus.get("totals"), _CORPUS_COUNT_FIELDS, "corpus totals")
    if any(totals[field] for field in _REFERENCE_COUNT_FIELDS):
        raise PublicDevelopmentPreviewError("public corpus contains reference-derived counts")
    if any(totals[field] for field in ("exported", "eee_records", "eee_schema_issues")):
        raise PublicDevelopmentPreviewError("public pre-human corpus contains canonical EEE")

    provider = _mapping(corpus.get("provider_accounting"), "public provider accounting")
    _exact_keys(provider, _CORPUS_PROVIDER_FIELDS, "public provider accounting")
    if (
        provider.get("schema_version") != "public-provider-accounting/0.1"
        or provider.get("budget_status") not in {"available", "exhausted", "actual_cost_overrun"}
        or provider.get("call_coverage") not in {"exhaustive", "lower_bound"}
    ):
        raise PublicDevelopmentPreviewError("public provider accounting status is invalid")
    started = _nonnegative_int(provider.get("structured_calls_started"), "provider calls started")
    completed = _nonnegative_int(
        provider.get("structured_calls_completed"), "provider calls completed"
    )
    pending = _nonnegative_int(provider.get("structured_calls_pending"), "provider calls pending")
    telemetry_calls = _nonnegative_int(
        provider.get("provider_call_telemetry_calls"), "provider telemetry calls"
    )
    telemetry_missing = _nonnegative_int(
        provider.get("provider_call_telemetry_missing_calls"), "provider missing telemetry"
    )
    cost_calls = _nonnegative_int(
        provider.get("provider_reported_cost_calls"), "provider cost calls"
    )
    if (
        started != completed + pending
        or pending != 0
        or telemetry_calls + telemetry_missing != completed
        or cost_calls > telemetry_calls
    ):
        raise PublicDevelopmentPreviewError("public provider accounting partition is invalid")
    for field in (
        "provider_reported_cost_usd_lower_bound",
        "provider_reported_input_tokens_lower_bound",
        "provider_reported_output_tokens_lower_bound",
        "provider_reported_reasoning_tokens_lower_bound",
        "provider_reported_total_tokens_lower_bound",
    ):
        _nonnegative_number(provider.get(field), f"public provider {field}")
    exact_cost = provider.get("provider_reported_cost_usd")
    if (cost_calls == completed) != (exact_cost is not None) or (
        exact_cost is not None
        and not _same_number(exact_cost, provider.get("provider_reported_cost_usd_lower_bound"))
    ):
        raise PublicDevelopmentPreviewError("public provider exact cost is invalid")

    papers = list(_sequence(corpus.get("papers_detail"), "public corpus papers"))
    if len(papers) != partitions["papers"]:
        raise PublicDevelopmentPreviewError("public corpus paper population is invalid")
    paper_ids: list[str] = []
    summed = {key: 0 for key in _CORPUS_COUNT_FIELDS}
    succeeded = 0
    failed = 0
    needing_review = 0
    successful_without_candidates = 0
    for raw_paper in papers:
        paper = _mapping(raw_paper, "public corpus paper")
        _exact_keys(paper, _CORPUS_PAPER_FIELDS, "public corpus paper")
        paper_id = _safe_id(paper.get("paper_id"), "public corpus paper id")
        paper_ids.append(paper_id)
        if not isinstance(paper.get("title"), str) or not paper.get("title"):
            raise PublicDevelopmentPreviewError("public corpus paper title is invalid")
        status = paper.get("status")
        if status not in {"success", "partial_failure"}:
            raise PublicDevelopmentPreviewError("public corpus paper status is invalid")
        succeeded += status == "success"
        failed += status != "success"
        _nonnegative_number(paper.get("wall_clock_seconds"), "paper wall clock")
        _sha256(paper.get("source_manifest_sha256"), "paper source manifest hash")
        pages = list(_sequence(paper.get("selected_pages"), "paper selected pages"))
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in pages
        ) or len(pages) != len(set(pages)):
            raise PublicDevelopmentPreviewError("paper selected pages are invalid")
        layout = _mapping(paper.get("layout"), "paper layout")
        segmentation = _mapping(paper.get("result_block_segmentation"), "paper segmentation")
        candidate_validation = _mapping(paper.get("candidate_validation"), "candidate validation")
        eee_schema = _mapping(paper.get("eee_schema"), "paper EEE schema")
        code = _mapping(paper.get("code"), "paper code")
        _exact_keys(layout, _CORPUS_LAYOUT_FIELDS, "paper layout")
        _exact_keys(segmentation, _CORPUS_SEGMENTATION_FIELDS, "paper segmentation")
        _exact_keys(
            candidate_validation,
            _candidate_validation_fields(candidate_validation),
            "candidate validation",
        )
        _exact_keys(eee_schema, _CORPUS_EEE_SCHEMA_FIELDS, "paper EEE schema")
        _exact_keys(code, _CODE_FIELDS, "paper code")
        if any(not isinstance(layout.get(field), str) or not layout.get(field) for field in layout):
            raise PublicDevelopmentPreviewError("public corpus paper layout is invalid")
        for field, value in segmentation.items():
            if field == "min_signal_score":
                _nonnegative_number(value, f"paper segmentation {field}")
            else:
                _nonnegative_int(value, f"paper segmentation {field}")
        if (
            candidate_validation.get("schema_version") not in _CORPUS_CANDIDATE_VALIDATION_VERSIONS
            or _nonnegative_number(
                candidate_validation.get("min_confidence"), "candidate minimum confidence"
            )
            > 1
            or not isinstance(eee_schema.get("version"), str)
            or not eee_schema.get("version")
        ):
            raise PublicDevelopmentPreviewError("public corpus validation contract is invalid")
        if code.get("git_available") is not True or code.get("git_dirty") is not False:
            raise PublicDevelopmentPreviewError("public corpus paper code is not clean Git")
        _git_object(code.get("git_commit"), "paper code commit")
        _sha256(code.get("source_tree_sha256"), "paper source tree")
        _sha256(eee_schema.get("sha256"), "paper EEE schema hash")
        counts = _exact_nonnegative_counts(
            paper.get("counts"), _CORPUS_COUNT_FIELDS, "public paper counts"
        )
        if any(counts[field] for field in _REFERENCE_COUNT_FIELDS):
            raise PublicDevelopmentPreviewError("public paper contains reference-derived counts")
        if any(counts[field] for field in ("exported", "eee_records", "eee_schema_issues")):
            raise PublicDevelopmentPreviewError("public pre-human paper contains canonical EEE")
        if (
            counts["candidates_before_deduplication"]
            != (counts["candidates"] + counts["duplicates_removed"])
            or counts["tuple_candidates"] != counts["tuple_passed"] + counts["tuple_review"]
        ):
            raise PublicDevelopmentPreviewError("public paper candidate-count algebra is invalid")
        if counts["verifications"] != (
            counts["verifier_accepts"] + counts["verifier_rejects"] + counts["verifier_reviews"]
        ) or counts["origin_candidates"] != (
            counts["origin_external"] + counts["origin_unresolved"] + counts["origin_no_signal"]
        ):
            raise PublicDevelopmentPreviewError("public paper stage-count algebra is invalid")
        for key, value in counts.items():
            summed[key] += value
        successful_without_candidates += status == "success" and counts["candidates"] == 0

        for public_stage, corpus_stage in _PUBLIC_TO_CORPUS_STAGE.items():
            stage = _mapping(paper.get(corpus_stage), f"public paper {corpus_stage}")
            _exact_keys(stage, _CORPUS_STAGE_FIELDS[corpus_stage], f"public paper {corpus_stage}")
            if public_stage != "block_candidate_extraction" and stage.get("enabled") is not True:
                raise PublicDevelopmentPreviewError("public corpus did not enable every stage")
            if "provider" in stage and stage.get("provider") != "openrouter":
                raise PublicDevelopmentPreviewError("public corpus provider is unsupported")
            _model_id(stage.get("model"), f"public paper {corpus_stage} model")
            max_tokens = _nonnegative_int(stage.get("max_tokens"), f"{corpus_stage} max tokens")
            temperature = stage.get("temperature")
            if temperature is not None:
                _nonnegative_number(temperature, f"{corpus_stage} temperature")
            if not isinstance(stage.get("reasoning_effort"), str) or not stage.get(
                "reasoning_effort"
            ):
                raise PublicDevelopmentPreviewError("public corpus reasoning effort is invalid")
            seed = stage.get("seed")
            if seed is not None:
                _nonnegative_int(seed, f"{corpus_stage} seed")
            if "prompt_sha256" in stage:
                _sha256(stage.get("prompt_sha256"), f"{corpus_stage} prompt hash")
            if max_tokens < 1 or stage.get("require_parameters") is not True:
                raise PublicDevelopmentPreviewError("public corpus stage contract is invalid")
            _validate_request_contract(stage.get("request_contract"), f"{corpus_stage} request")
            contract = _mapping(stage.get("request_contract"), f"{corpus_stage} request")
            if contract.get("max_tokens") != max_tokens or contract.get("seed") != stage.get(
                "seed"
            ):
                raise PublicDevelopmentPreviewError("public corpus request settings disagree")
            request_schema = _mapping(contract.get("schema"), f"{corpus_stage} request schema")
            if request_schema.get("schema_name") != _PUBLIC_REQUEST_SCHEMA_NAMES[public_stage]:
                raise PublicDevelopmentPreviewError("public corpus request API disagrees")
            if public_stage == "tuple_resolution" and (
                stage.get("mutates_candidate_tuple") is not False
                or stage.get("allows_origin_or_export") is not False
                or stage.get("requires_export_concordant_tuple") is not True
            ):
                raise PublicDevelopmentPreviewError("public tuple stage authority is invalid")
            if public_stage == "independent_verification" and (
                stage.get("verification_schema_version") != "candidate-verification/0.2"
                or stage.get("grounding_schema_version") != "candidate-verification-grounding/0.1"
            ):
                raise PublicDevelopmentPreviewError("public verifier API is invalid")
            if (
                public_stage == "origin_retrieval"
                and stage.get("requires_independent_verifier_accept") is not True
            ):
                raise PublicDevelopmentPreviewError("public origin stage authority is invalid")
            stage_usage = _validate_stage_usage(
                stage.get("usage"),
                f"{corpus_stage} usage",
                require_exact=provider.get("call_coverage") == "exhaustive",
            )
            execution = _mapping(stage.get("execution"), f"{corpus_stage} execution")
            _exact_keys(
                execution, _STAGE_EXECUTION_FIELDS[public_stage], f"{corpus_stage} execution"
            )
            for field, value in execution.items():
                _nonnegative_int(value, f"{corpus_stage} execution {field}")
            if public_stage == "row_disposition":
                limits = _mapping(stage.get("limits"), "row limits")
                _exact_keys(limits, _CORPUS_ROW_LIMIT_FIELDS, "row limits")
                for field, value in limits.items():
                    _nonnegative_int(value, f"row limit {field}")
            if public_stage == "block_candidate_extraction":
                if (
                    execution.get("blocks_total")
                    != execution.get("blocks_succeeded")
                    + execution.get("blocks_resumed")
                    + execution.get("blocks_failed")
                    or execution.get("calls_resumed")
                    != execution.get("calls_resumed_succeeded")
                    + execution.get("calls_resumed_failed")
                    or execution.get("no_call_failures")
                    != execution.get("requests_rejected")
                    + execution.get("transport_failures")
                    + execution.get("local_failures")
                    or execution.get("calls_succeeded") + execution.get("calls_failed")
                    != stage_usage["calls_field_count"]
                    or execution.get("calls_resumed") != stage_usage["resumed_calls_field_count"]
                ):
                    raise PublicDevelopmentPreviewError("extractor execution partition is invalid")
            elif public_stage == "row_disposition":
                if execution.get("batches_total") != execution.get(
                    "batches_resumed"
                ) + execution.get("batches_executed"):
                    raise PublicDevelopmentPreviewError("row execution partition is invalid")
            elif public_stage == "tuple_resolution":
                if (
                    execution.get("candidates_selected")
                    != execution.get("candidates_passed")
                    + execution.get("candidates_routed_to_review")
                    or any(
                        execution.get(field) > execution.get("candidates_routed_to_review")
                        for field in (
                            "candidates_unsupported",
                            "candidates_failed",
                            "candidates_mismatched",
                        )
                    )
                    or execution.get("candidates_resumed") > execution.get("candidates_selected")
                ):
                    raise PublicDevelopmentPreviewError("tuple execution partition is invalid")
            elif public_stage == "independent_verification":
                if (
                    execution.get("candidates_resumed")
                    != execution.get("candidates_resumed_succeeded")
                    + execution.get("candidates_resumed_failed")
                    or execution.get("candidates_executed")
                    != execution.get("candidates_executed_succeeded")
                    + execution.get("candidates_executed_failed")
                    or execution.get("candidates_selected")
                    != execution.get("candidates_unbound")
                    + execution.get("candidates_resumed")
                    + execution.get("candidates_executed")
                    or execution.get("candidates_verified")
                    != execution.get("candidates_resumed_succeeded")
                    + execution.get("candidates_executed_succeeded")
                    or execution.get("candidates_failed")
                    != execution.get("candidates_resumed_failed")
                    + execution.get("candidates_executed_failed")
                ):
                    raise PublicDevelopmentPreviewError("verifier execution partition is invalid")
            elif any(
                execution.get(field) > execution.get("candidates_selected")
                for field in (
                    "candidates_resumed",
                    "candidates_failed",
                    "candidates_deterministic_external",
                )
            ):
                raise PublicDevelopmentPreviewError("origin execution partition is invalid")
        needing_review += counts["candidates_needing_review"] > 0 or status != "success"
    if len(paper_ids) != len(set(paper_ids)) or summed != totals:
        raise PublicDevelopmentPreviewError("public corpus paper totals are invalid")
    if succeeded != partitions["papers_succeeded"] or failed != partitions["papers_failed"]:
        raise PublicDevelopmentPreviewError("public corpus success partition is invalid")
    expected_status = "success" if not failed else "error" if not succeeded else "partial_failure"
    if (
        corpus.get("status") != expected_status
        or partitions["papers_with_eee"] != 0
        or partitions["papers_without_eee"] != succeeded
        or partitions["papers_without_candidates"] != successful_without_candidates
    ):
        raise PublicDevelopmentPreviewError("public corpus terminal partition is invalid")
    if needing_review > partitions["papers_needing_review"]:
        raise PublicDevelopmentPreviewError("public corpus review partition is invalid")
    return paper_ids


def _validate_count_map(value: object, allowed: set[str], context: str) -> dict[str, int]:
    counts = _mapping(value, context)
    if not set(counts).issubset(allowed):
        raise PublicDevelopmentPreviewError(f"{context} has an unsupported category")
    return {key: _nonnegative_int(item, f"{context}.{key}") for key, item in counts.items()}


def _validate_summary_model_binding(value: object, context: str) -> None:
    binding = _mapping(value, context)
    _exact_keys(binding, _SUMMARY_MODEL_BINDING_FIELDS, context)
    if binding.get("provider") != "openrouter":
        raise PublicDevelopmentPreviewError(f"{context} provider is unsupported")
    _model_id(binding.get("model"), f"{context} model")
    if _nonnegative_int(binding.get("max_tokens"), f"{context} max tokens") < 1:
        raise PublicDevelopmentPreviewError(f"{context} max tokens is invalid")
    temperature = binding.get("temperature")
    if temperature is not None:
        _nonnegative_number(temperature, f"{context} temperature")
    if not isinstance(binding.get("reasoning_effort"), str):
        raise PublicDevelopmentPreviewError(f"{context} reasoning effort is invalid")
    seed = binding.get("seed")
    if seed is not None:
        _nonnegative_int(seed, f"{context} seed")
    _sha256(binding.get("prompt_sha256"), f"{context} prompt hash")
    _sha256(binding.get("request_contract_sha256"), f"{context} request hash")


def _validate_public_summary_schema(
    summary: Mapping[str, object], corpus: Mapping[str, object]
) -> dict[str, object]:
    _exact_keys(summary, _SUMMARY_TOP_FIELDS, "public summary")
    if (
        summary.get("schema_version") != "public-development-summary/0.3"
        or summary.get("projection_mode") != "sealed_current_five_stage_receipt"
        or summary.get("current_sealed_receipt") is not True
        or not isinstance(summary.get("statement"), str)
        or not summary.get("statement")
    ):
        raise PublicDevelopmentPreviewError("public summary schema/status is unsupported")

    paper_count = _nonnegative_int(corpus.get("papers"), "public corpus paper count")
    scope = _mapping(summary.get("scope"), "summary scope")
    _exact_keys(scope, _SUMMARY_SCOPE_FIELDS, "summary scope")
    if scope != {
        "split": "development",
        "papers": paper_count,
        "holdout_included": False,
        "private_human_annotations_included": False,
        "independent_human_validation": False,
    }:
        raise PublicDevelopmentPreviewError("public summary scope is invalid")

    binding = _mapping(summary.get("run_binding"), "summary run binding")
    _exact_keys(binding, _SUMMARY_RUN_BINDING_FIELDS, "summary run binding")
    _safe_id(binding.get("run_id"), "summary run id")
    _safe_id(binding.get("corpus_id"), "summary corpus id")
    if (
        binding.get("corpus_spec_binding_status") != "recorded_hash_not_rederived"
        or not isinstance(binding.get("generated_at"), str)
        or not binding.get("generated_at")
    ):
        raise PublicDevelopmentPreviewError("summary run binding status is invalid")
    for field in (
        "recorded_corpus_spec_sha256",
        "paper_ids_sha256",
        "corpus_run_sha256",
    ):
        _sha256(binding.get(field), f"summary binding {field}")
    _validate_summary_model_binding(binding.get("extractor"), "summary extractor binding")
    _validate_summary_model_binding(binding.get("row_extractor"), "summary row binding")
    candidate_validation = _mapping(
        binding.get("candidate_validation"), "summary candidate validation"
    )
    _exact_keys(
        candidate_validation,
        _candidate_validation_fields(candidate_validation),
        "summary candidate validation",
    )
    if candidate_validation.get("schema_version") not in _CORPUS_CANDIDATE_VALIDATION_VERSIONS:
        raise PublicDevelopmentPreviewError("summary candidate validation is unsupported")
    minimum_confidence = _nonnegative_number(
        candidate_validation.get("min_confidence"), "summary minimum confidence"
    )
    if minimum_confidence > 1:
        raise PublicDevelopmentPreviewError("summary minimum confidence is invalid")
    eee_schema = _mapping(binding.get("eee_schema"), "summary EEE schema")
    _exact_keys(eee_schema, _CORPUS_EEE_SCHEMA_FIELDS, "summary EEE schema")
    if not isinstance(eee_schema.get("version"), str) or not eee_schema.get("version"):
        raise PublicDevelopmentPreviewError("summary EEE schema version is invalid")
    _sha256(eee_schema.get("sha256"), "summary EEE schema hash")
    code = _mapping(binding.get("code"), "summary code")
    _exact_keys(code, _CODE_FIELDS, "summary code")
    if code.get("git_available") is not True or code.get("git_dirty") is not False:
        raise PublicDevelopmentPreviewError("summary code binding is not clean Git")
    _git_object(code.get("git_commit"), "summary code commit")
    _sha256(code.get("source_tree_sha256"), "summary source tree")
    run_seal = _mapping(binding.get("run_seal"), "summary run seal")
    _exact_keys(
        run_seal,
        {"schema_version", "seal_sha256", "tree_sha256", "file_count"},
        "summary run seal",
    )
    if run_seal.get("schema_version") != "run-tree-seal/0.1":
        raise PublicDevelopmentPreviewError("summary run-seal schema is unsupported")
    _sha256(run_seal.get("seal_sha256"), "summary seal hash")
    _sha256(run_seal.get("tree_sha256"), "summary tree hash")
    _nonnegative_int(run_seal.get("file_count"), "summary sealed file count")

    stage_chain = _mapping(binding.get("stage_chain"), "summary stage chain")
    _exact_keys(stage_chain, _SUMMARY_STAGE_CHAIN_FIELDS, "summary stage chain")
    stage_status = stage_chain.get("five_stage_gate_status")
    expected_mode = (
        "current_five_stage_contract"
        if stage_status == "validated"
        else "current_tuple_review_contract"
    )
    if (
        stage_status not in {"validated", "partial_failure", "not_run"}
        or stage_chain.get("mode") != expected_mode
        or stage_chain.get("missing_gate_treated_as_passed") is not False
    ):
        raise PublicDevelopmentPreviewError("summary stage-chain status is invalid")
    checkpoints = _mapping(
        stage_chain.get("checkpoint_validation"), "summary checkpoint validation"
    )
    if set(checkpoints) != set(_ROUTE_TO_RECEIPT_STAGE.values()):
        raise PublicDevelopmentPreviewError("summary checkpoint stage set is invalid")
    for stage_name, receipt in checkpoints.items():
        _validate_stage_receipt(
            receipt, f"summary checkpoint {stage_name}", paper_count=paper_count
        )

    technical = _mapping(summary.get("technical_health"), "summary technical health")
    _exact_keys(technical, _SUMMARY_TECHNICAL_FIELDS, "summary technical health")
    papers_succeeded = _nonnegative_int(
        technical.get("papers_succeeded"), "summary papers succeeded"
    )
    papers_failed = _nonnegative_int(technical.get("papers_failed"), "summary papers failed")
    papers_review = _nonnegative_int(
        technical.get("papers_needing_review"), "summary papers needing review"
    )
    if (
        technical.get("status") != corpus.get("status")
        or technical.get("run_completeness")
        != ("complete" if stage_status == "validated" else "typed_stage_incomplete")
        or technical.get("release_ready") is not False
        or papers_succeeded != corpus.get("papers_succeeded")
        or papers_failed != corpus.get("papers_failed")
        or papers_review != corpus.get("papers_needing_review")
    ):
        raise PublicDevelopmentPreviewError("summary technical accounting disagrees")
    _nonnegative_number(technical.get("wall_clock_seconds"), "summary wall clock")
    _validate_count_map(
        technical.get("review_reason_counts"),
        _SUMMARY_REVIEW_REASON_FIELDS,
        "summary review reasons",
    )

    row = _mapping(summary.get("row_enumeration"), "summary row enumeration")
    _exact_keys(row, _SUMMARY_ROW_FIELDS, "summary row enumeration")
    for field in (
        "complete_extraction",
        "all_rows_accounted_for",
        "all_planned_rows_partitioned",
        "all_batchable_rows_resolved",
        "no_unknown_or_invalid_rows_seen",
    ):
        if not isinstance(row.get(field), bool):
            raise PublicDevelopmentPreviewError("summary row completeness flag is invalid")
    row_counts = {
        field: _nonnegative_int(row.get(field), f"summary row {field}")
        for field in (
            "tables_considered",
            "dense_tables",
            "rows_planned",
            "rows_resolved",
            "rows_unresolved",
            "rows_unbatchable",
            "unknown_row_ids_seen",
            "invalid_rows_seen",
        )
    }
    dispositions = _exact_nonnegative_counts(
        row.get("dispositions"),
        _SUMMARY_ROW_DISPOSITION_FIELDS,
        "summary row dispositions",
    )
    if (
        sum(dispositions.values()) != row_counts["rows_resolved"]
        or row_counts["rows_resolved"]
        + row_counts["rows_unresolved"]
        + row_counts["rows_unbatchable"]
        != row_counts["rows_planned"]
        or row_counts["dense_tables"] > row_counts["tables_considered"]
        or row.get("all_rows_accounted_for") is not True
        or row.get("all_planned_rows_partitioned") is not True
        or row.get("all_batchable_rows_resolved") is not (row_counts["rows_unresolved"] == 0)
        or row.get("complete_extraction")
        is not (row_counts["rows_unresolved"] == 0 and row_counts["rows_unbatchable"] == 0)
        or row.get("no_unknown_or_invalid_rows_seen")
        is not (row_counts["unknown_row_ids_seen"] == 0 and row_counts["invalid_rows_seen"] == 0)
    ):
        raise PublicDevelopmentPreviewError("summary row accounting disagrees")

    outputs = _mapping(summary.get("outputs"), "summary outputs")
    _exact_keys(outputs, _SUMMARY_OUTPUT_FIELDS, "summary outputs")
    output_count_fields = {
        field
        for field in _SUMMARY_OUTPUT_FIELDS
        if field
        not in {
            "count_projection_basis",
            "candidate_proposal_removal_rate",
            "export_status_counts",
            "text_support_status_counts",
            "referential_status_counts",
            "attribution_state_counts",
            "candidate_lineage",
        }
    }
    output_counts = {
        field: _nonnegative_int(outputs.get(field), f"summary output {field}")
        for field in output_count_fields
    }
    if outputs.get("count_projection_basis") != "checkpoint_replayed_allowlist":
        raise PublicDevelopmentPreviewError("summary output count basis is invalid")
    denominator = output_counts["candidates_before_deduplication"]
    expected_rate = output_counts["duplicates_removed"] / denominator if denominator else None
    actual_rate = outputs.get("candidate_proposal_removal_rate")
    if (expected_rate is None and actual_rate is not None) or (
        expected_rate is not None and not _same_number(actual_rate, expected_rate)
    ):
        raise PublicDevelopmentPreviewError("summary candidate-removal rate disagrees")
    corpus_totals = _mapping(corpus.get("totals"), "corpus totals")
    if any(output_counts[field] != corpus_totals.get(field) for field in output_count_fields):
        raise PublicDevelopmentPreviewError("summary output totals disagree with corpus")
    category_maps = (
        (
            "export_status_counts",
            {"eligible", "needs_review", "not_eligible", "exported"},
        ),
        (
            "text_support_status_counts",
            {"unverified", "supported", "partially_supported", "unsupported"},
        ),
        (
            "referential_status_counts",
            {"unverified", "resolved", "unresolved", "wrong_scope"},
        ),
        (
            "attribution_state_counts",
            {"paper_produced", "externally_sourced", "unresolved", "no_signal"},
        ),
    )
    for field, allowed in category_maps:
        if (
            sum(_validate_count_map(outputs.get(field), allowed, f"summary {field}").values())
            != output_counts["candidates"]
        ):
            raise PublicDevelopmentPreviewError(f"summary {field} total disagrees")
    lineage = _mapping(outputs.get("candidate_lineage"), "summary candidate lineage")
    _exact_keys(lineage, _SUMMARY_CANDIDATE_LINEAGE_FIELDS, "summary candidate lineage")
    lineage_counts = {
        field: _nonnegative_int(lineage.get(field), f"summary lineage {field}")
        for field in _SUMMARY_CANDIDATE_LINEAGE_FIELDS - {"all_current_schema_runs_bound"}
    }
    if (
        lineage.get("all_current_schema_runs_bound") is not True
        or lineage_counts["runs_bound"] != paper_count
        or lineage_counts["final_candidates"] != output_counts["candidates"]
        or lineage_counts["candidate_occurrences"]
        != lineage_counts["final_candidates"] + lineage_counts["merged_candidates"]
    ):
        raise PublicDevelopmentPreviewError("summary candidate lineage disagrees")

    canonical = _mapping(summary.get("canonical_eee"), "summary canonical EEE")
    _exact_keys(canonical, _SUMMARY_CANONICAL_FIELDS, "summary canonical EEE")
    if canonical != {
        "status": "empty",
        "records": 0,
        "schema_issues": 0,
        "safe_empty_output_is_valid": True,
        "positive_paper_produced_origin_required": True,
        "automatic_positive_origin_enabled": False,
    }:
        raise PublicDevelopmentPreviewError("public summary contains canonical EEE")
    numeric = _mapping(summary.get("numeric_export_provenance"), "summary numeric provenance")
    _exact_keys(numeric, _SUMMARY_NUMERIC_PROVENANCE_FIELDS, "summary numeric provenance")
    if numeric != {
        "status": "not_applicable_empty_source_export",
        "exported_observations": 0,
        "complete_observations": 0,
        "all_complete": None,
        "evidence_quotations_included": False,
    }:
        raise PublicDevelopmentPreviewError("public summary numeric provenance is invalid")
    reference = _mapping(summary.get("reference_evaluation"), "summary reference evaluation")
    _exact_keys(reference, _SUMMARY_REFERENCE_FIELDS, "summary reference evaluation")
    precision = _mapping(reference.get("precision"), "summary reference precision")
    _exact_keys(precision, {"status"}, "summary reference precision")
    if reference != {
        "status": "not_projected_from_current_receipt",
        "precision": {"status": "not_measured"},
        "generalization_evidence": False,
    }:
        raise PublicDevelopmentPreviewError("public summary contains reference evaluation")

    provenance = _mapping(summary.get("export_provenance_modes"), "summary provenance")
    _exact_keys(provenance, _SUMMARY_PROVENANCE_FIELDS, "summary provenance")
    source = _mapping(provenance.get("source_run"), "summary source-run provenance")
    reviewed = _mapping(provenance.get("reviewed_derived"), "summary reviewed provenance")
    _exact_keys(source, _SUMMARY_SOURCE_PROVENANCE_FIELDS, "summary source-run provenance")
    _exact_keys(reviewed, _SUMMARY_REVIEWED_PROVENANCE_FIELDS, "summary reviewed provenance")
    modes = _exact_nonnegative_counts(
        source.get("provenance_mode_counts"),
        _SUMMARY_PROVENANCE_MODE_FIELDS,
        "summary source provenance modes",
    )
    if (
        provenance.get("counts_combined") is not False
        or source.get("status") != "measured"
        or source.get("eee_records") != 0
        or source.get("eee_observations") != 0
        or any(modes.values())
        or reviewed
        != {
            "status": "not_supplied",
            "locked_review_context_verified": False,
            "independence_status": "not_measured",
            "exported_observation_authority_mode_counts": None,
            "eee_records": None,
            "eee_observations": None,
            "provenance_mode_counts": None,
            "source_and_reviewed_counts_combined": False,
        }
    ):
        raise PublicDevelopmentPreviewError("public summary mixes reviewed-derived material")
    annotations = _mapping(summary.get("annotation_status"), "summary annotation status")
    _exact_keys(annotations, _SUMMARY_ANNOTATION_FIELDS, "summary annotation status")
    if annotations != {
        "source_run_human_annotations_included": False,
        "reviewed_derived_supplied": False,
        "reviewed_derived_authority_mode_counts": None,
        "inter_annotator_agreement_available": False,
    }:
        raise PublicDevelopmentPreviewError("public summary contains annotation material")
    privacy = _mapping(summary.get("privacy"), "summary privacy")
    _exact_keys(privacy, _SUMMARY_PRIVACY_FIELDS, "summary privacy")
    if any(value is not False for value in privacy.values()):
        raise PublicDevelopmentPreviewError("public summary privacy boundary is invalid")
    limitations = _sequence(summary.get("limitations"), "summary limitations")
    if not limitations or any(not isinstance(item, str) or not item for item in limitations):
        raise PublicDevelopmentPreviewError("public summary limitations are invalid")
    return _validate_public_telemetry(
        summary.get("provider_usage_recorded"), "summary provider usage"
    )


def _validate_public_route_schema(route: Mapping[str, object]) -> dict[str, str]:
    _exact_keys(route, _ROUTE_FIELDS, "public route")
    if route.get("schema_version") != _ROUTE_SCHEMA_VERSION:
        raise PublicDevelopmentPreviewError("public route schema is unsupported")
    _safe_id(route.get("experiment_id"), "public route experiment id")
    _safe_id(route.get("amendment_id"), "public route amendment id")
    _git_object(route.get("code_commit"), "public route code commit")
    _git_object(route.get("code_tree"), "public route code tree")
    for field in ("runner_sha256", "campaign_seal_sha256", "plan_bundle_sha256"):
        _sha256(route.get(field), f"public route {field}")
    stages = _mapping(route.get("stages"), "public route stages")
    if set(stages) != set(_ROUTE_STAGES):
        raise PublicDevelopmentPreviewError("public route stage set is invalid")
    selected: dict[str, str] = {}
    unmeasured = False
    for stage_name in _ROUTE_STAGES:
        stage = _mapping(stages[stage_name], f"public route stage {stage_name}")
        _exact_keys(stage, _ROUTE_STAGE_FIELDS, f"public route stage {stage_name}")
        _sha256(stage.get("execution_selection_sha256"), "public execution selection")
        quality_hash = stage.get("quality_source_artifact_sha256")
        if quality_hash is not None:
            _sha256(quality_hash, "public quality source")
        basis = stage.get("selection_basis")
        if basis not in {"measured_quality", "technical_only_unmeasured_quality"}:
            raise PublicDevelopmentPreviewError("public route selection basis is unsupported")
        if basis == "measured_quality" and quality_hash is None:
            raise PublicDevelopmentPreviewError("public measured route lacks a quality source")
        unmeasured = unmeasured or basis == "technical_only_unmeasured_quality"
        selected[stage_name] = _model_id(stage.get("selected_model"), "public route model")
    expected_status = "diagnostic_only" if unmeasured else "quality_selected"
    if route.get("status") != expected_status:
        raise PublicDevelopmentPreviewError("public route status disagrees with its stage bases")
    return selected


def _validate_public_manifest_schema(manifest: Mapping[str, object]) -> None:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "bundle_id",
            "artifact_status",
            "statement",
            "source_run",
            "prospective_execution",
            "outputs",
            "files",
            "limitations",
        },
        "publication manifest",
    )
    if manifest.get("schema_version") != PUBLIC_DEVELOPMENT_PREVIEW_SCHEMA_VERSION:
        raise PublicDevelopmentPreviewError("public preview manifest schema is unsupported")
    _safe_id(manifest.get("bundle_id"), "bundle id")
    if not isinstance(manifest.get("statement"), str) or not manifest.get("statement"):
        raise PublicDevelopmentPreviewError("public preview statement is invalid")
    limitations = _sequence(manifest.get("limitations"), "public preview limitations")
    if not limitations or any(not isinstance(item, str) or not item for item in limitations):
        raise PublicDevelopmentPreviewError("public preview limitations are invalid")
    if manifest.get("files") != sorted(_PREVIEW_FILES - {"SHA256SUMS"}):
        raise PublicDevelopmentPreviewError("public preview manifest file list disagrees")

    artifact_status = _mapping(manifest.get("artifact_status"), "artifact status")
    _exact_keys(artifact_status, _MANIFEST_ARTIFACT_STATUS_FIELDS, "artifact status")
    if artifact_status != {
        "classification": "pre_human_development_preview",
        "release_ready": False,
        "human_review_status": "pending",
        "canonical_real_paper_eee_available": False,
    }:
        raise PublicDevelopmentPreviewError("public preview overstates its release status")

    source_run = _mapping(manifest.get("source_run"), "manifest source run")
    _exact_keys(source_run, _MANIFEST_SOURCE_RUN_FIELDS, "manifest source run")
    _safe_id(source_run.get("corpus_id"), "manifest corpus id")
    if source_run.get("evaluation_split") != "development":
        raise PublicDevelopmentPreviewError("manifest source split is not development")
    if _nonnegative_int(source_run.get("paper_count"), "manifest paper count") < 1:
        raise PublicDevelopmentPreviewError("manifest paper count must be positive")
    for field in ("corpus_spec_sha256", "paper_ids_sha256", "corpus_run_sha256"):
        _sha256(source_run.get(field), f"manifest {field}")
    run_seal = _mapping(source_run.get("run_seal"), "manifest run seal")
    _exact_keys(run_seal, _MANIFEST_RUN_SEAL_FIELDS, "manifest run seal")
    if run_seal.get("schema_version") != "run-tree-seal/0.1":
        raise PublicDevelopmentPreviewError("manifest run-seal schema is unsupported")
    for field in ("seal_sha256", "tree_sha256"):
        _sha256(run_seal.get(field), f"manifest run seal {field}")
    _nonnegative_int(run_seal.get("file_count"), "manifest sealed file count")
    _nonnegative_int(run_seal.get("total_bytes"), "manifest sealed byte count")
    code = _mapping(source_run.get("code"), "manifest code")
    _exact_keys(code, _CODE_FIELDS, "manifest code")
    if code.get("git_available") is not True or code.get("git_dirty") is not False:
        raise PublicDevelopmentPreviewError("manifest source code was not a clean Git state")
    _git_object(code.get("git_commit"), "manifest code commit")
    _sha256(code.get("source_tree_sha256"), "manifest source tree")

    prospective = _mapping(manifest.get("prospective_execution"), "prospective execution")
    _exact_keys(prospective, _MANIFEST_PROSPECTIVE_FIELDS, "prospective execution")
    if (
        prospective.get("route_schema_version") != _ROUTE_SCHEMA_VERSION
        or prospective.get("route_status") not in {"diagnostic_only", "quality_selected"}
        or prospective.get("binding_status") != "prospectively_bound_to_sealed_source_run"
    ):
        raise PublicDevelopmentPreviewError("prospective route status is invalid")
    for field in ("route_selection_sha256", "precall_contract_sha256", "precall_file_sha256"):
        _sha256(prospective.get(field), f"prospective {field}")
    selected = _mapping(prospective.get("selected_models"), "prospective selected models")
    if set(selected) != set(_ROUTE_STAGES):
        raise PublicDevelopmentPreviewError("prospective stage selection is incomplete")
    for stage, model in selected.items():
        _model_id(model, f"prospective model {stage}")
    privacy = _mapping(prospective.get("privacy"), "prospective privacy")
    _exact_keys(privacy, _PUBLIC_PRECALL_PRIVACY_FIELDS, "prospective privacy")
    if privacy != {
        "zdr": True,
        "data_collection": "deny",
        "label_blind_source_derived_scientific_payloads_only": True,
        "forbidden_private_or_labelled_payloads_sent": False,
    }:
        raise PublicDevelopmentPreviewError("prospective privacy contract is invalid")
    budget = _mapping(prospective.get("budget"), "prospective budget")
    _exact_keys(budget, _PRECALL_BUDGET_FIELDS, "prospective budget")
    max_calls = _nonnegative_int(budget.get("max_structured_calls"), "prospective call ceiling")
    max_attempts = _nonnegative_int(
        budget.get("max_transport_attempts"), "prospective attempt ceiling"
    )
    if (
        max_calls < 1
        or max_attempts != max_calls * 4
        or _nonnegative_number(budget.get("max_provider_cost_usd"), "prospective cost ceiling") <= 0
        or _nonnegative_number(
            budget.get("provider_call_cost_reservation_usd"), "prospective reservation"
        )
        <= 0
    ):
        raise PublicDevelopmentPreviewError("prospective budget contract is invalid")

    outputs = _mapping(manifest.get("outputs"), "preview outputs")
    _exact_keys(outputs, _MANIFEST_OUTPUT_FIELDS, "preview outputs")
    source_output = _mapping(outputs.get("automatic_source_run"), "automatic source output")
    reviewed_output = _mapping(outputs.get("reviewed_derived"), "reviewed output")
    reviewed_summary = _mapping(outputs.get("reviewed_summary"), "reviewed summary")
    _exact_keys(source_output, _AUTOMATIC_OUTPUT_FIELDS, "automatic source output")
    _exact_keys(reviewed_output, _REVIEWED_OUTPUT_FIELDS, "reviewed output")
    _exact_keys(reviewed_summary, _REVIEWED_SUMMARY_FIELDS, "reviewed summary")
    if (
        source_output
        != {
            "status": "model_proposed_human_review_pending",
            "canonical_eee_records": 0,
            "present": True,
        }
        or reviewed_output
        != {
            "status": "pending_human_review",
            "expected_relative_path": "reviewed/eee",
            "records": None,
            "present": False,
        }
        or reviewed_summary
        != {
            "status": "pending_human_review",
            "expected_relative_path": "reviewed/public-summary.json",
            "present": False,
        }
    ):
        raise PublicDevelopmentPreviewError("public preview reviewed-output boundary is invalid")


def _validate_public_sources_schema(sources: Mapping[str, object]) -> list[str]:
    _exact_keys(sources, _SOURCE_TOP_FIELDS, "public sources")
    if sources.get("schema_version") != PUBLIC_SOURCE_INVENTORY_SCHEMA_VERSION:
        raise PublicDevelopmentPreviewError("public source inventory schema is unsupported")
    _safe_id(sources.get("corpus_id"), "public source corpus id")
    if sources.get("evaluation_split") != "development":
        raise PublicDevelopmentPreviewError("public source split is not development")
    paper_ids: list[str] = []
    for raw_paper in _sequence(sources.get("papers"), "public source papers"):
        paper = _mapping(raw_paper, "public source paper")
        _exact_keys(paper, _SOURCE_PAPER_FIELDS, "public source paper")
        paper_id = _safe_id(paper.get("paper_id"), "public source paper id")
        paper_ids.append(paper_id)
        if (
            not isinstance(paper.get("title"), str)
            or not paper.get("title")
            or not isinstance(paper.get("venue"), str)
            or not paper.get("venue")
            or not isinstance(paper.get("perspective_role"), str)
            or not paper.get("perspective_role")
            or isinstance(paper.get("year"), bool)
            or not isinstance(paper.get("year"), int)
            or int(paper["year"]) < 1
        ):
            raise PublicDevelopmentPreviewError("public source paper metadata is invalid")
        for identifier in ("doi", "arxiv_id"):
            item = paper.get(identifier)
            if item is not None and (not isinstance(item, str) or not item):
                raise PublicDevelopmentPreviewError("public source identifier is invalid")
        _sha256(paper.get("source_manifest_sha256"), "public source manifest hash")
        page_scope = _mapping(paper.get("page_scope"), "public source page scope")
        _exact_keys(page_scope, _SOURCE_PAGE_SCOPE_FIELDS, "public source page scope")
        configured = list(_sequence(page_scope.get("configured_include_pages"), "configured pages"))
        selected = list(_sequence(page_scope.get("selected_pages"), "selected pages"))
        if (
            any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in configured
            )
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in selected
            )
            or len(configured) != len(set(configured))
            or len(selected) != len(set(selected))
            or page_scope.get("selection_mode")
            != ("configured_page_scope" if configured else "automatic_selector")
            or (configured and configured != selected)
            or len(selected)
            > _nonnegative_int(page_scope.get("max_result_pages"), "maximum result pages")
        ):
            raise PublicDevelopmentPreviewError("public source page scope is invalid")
        for uri_field in ("acm_url", "pdf_url", "repository_url"):
            uri = paper.get(uri_field)
            if uri is not None and _public_http_uri(uri, f"public {uri_field}") != uri:
                raise PublicDevelopmentPreviewError("public source URI is invalid")
        supplements = _sequence(paper.get("supplement_urls"), "public supplement URLs")
        if any(
            _public_http_uri(item, "public supplement URL") != item for item in supplements
        ) or len(supplements) != len(set(supplements)):
            raise PublicDevelopmentPreviewError("public supplement URI is invalid")
        repository_commit = paper.get("repository_commit")
        if repository_commit is not None:
            _git_object(str(repository_commit).casefold(), "public repository commit")
        source_ids: list[str] = []
        paper_sources = _sequence(paper.get("sources"), "public source entries")
        for raw_source in paper_sources:
            source = _mapping(raw_source, "public source entry")
            _exact_keys(source, _SOURCE_ITEM_FIELDS, "public source entry")
            source_ids.append(_safe_id(source.get("source_id"), "public source id"))
            role = source.get("role")
            access = source.get("access_status")
            if role not in {"paper", "supplement", "repository", "proceedings_record"}:
                raise PublicDevelopmentPreviewError("public source role is invalid")
            if access not in {"available", "unavailable", "expired", "restricted"}:
                raise PublicDevelopmentPreviewError("public source access status is invalid")
            if source.get("license_disposition") not in {
                "private_use_only",
                "derived_metadata_only",
                "redistributable",
                "unknown",
            }:
                raise PublicDevelopmentPreviewError("public source license status is invalid")
            if not isinstance(source.get("retrieved_at"), str) or not source.get("retrieved_at"):
                raise PublicDevelopmentPreviewError("public source retrieval time is invalid")
            for uri_field in ("public_original_uri", "public_resolved_uri"):
                uri = source.get(uri_field)
                if uri is not None and _public_http_uri(uri, uri_field) != uri:
                    raise PublicDevelopmentPreviewError("public source URI is invalid")
            source_sha = source.get("sha256")
            if source_sha is not None:
                _sha256(source_sha, "public source hash")
            source_commit = source.get("git_commit")
            if source_commit is not None:
                _git_object(str(source_commit).casefold(), "public source commit")
            byte_size = source.get("byte_size")
            if byte_size is not None:
                _nonnegative_int(byte_size, "public source byte size")
            media_type = source.get("media_type")
            if media_type is not None and (not isinstance(media_type, str) or not media_type):
                raise PublicDevelopmentPreviewError("public source media type is invalid")
            if (
                access == "available"
                and role != "repository"
                and (source_sha is None or byte_size is None)
            ):
                raise PublicDevelopmentPreviewError("available public source lacks a byte binding")
            if access == "available" and role == "repository" and source_commit is None:
                raise PublicDevelopmentPreviewError(
                    "available public repository lacks a commit binding"
                )
        if (
            not source_ids
            or len(source_ids) != len(set(source_ids))
            or sum(
                _mapping(item, "public source entry").get("role") == "paper"
                for item in paper_sources
            )
            != 1
        ):
            raise PublicDevelopmentPreviewError("public source IDs are not unique")
    if not paper_ids or len(paper_ids) != len(set(paper_ids)):
        raise PublicDevelopmentPreviewError("public source paper IDs are invalid")
    return paper_ids


def _validate_operational_stage(value: object, stage_name: str, context: str) -> dict[str, object]:
    stage = _mapping(value, context)
    expected_fields = (
        (_TELEMETRY_FIELDS - {"recorded_structured_invocations"})
        | {"completed_calls"}
        | _OPERATIONAL_STAGE_EXTRA_FIELDS[stage_name]
    )
    _exact_keys(stage, expected_fields, context)
    if not isinstance(stage.get("basis"), str) or not stage.get("basis"):
        raise PublicDevelopmentPreviewError(f"{context} basis is invalid")
    calls = _nonnegative_int(stage.get("completed_calls"), f"{context} completed calls")
    for field in (
        "cost_reported_calls",
        "input_tokens_reported_calls",
        "output_tokens_reported_calls",
        "reasoning_tokens_reported_calls",
        "total_tokens_reported_calls",
    ):
        if _nonnegative_int(stage.get(field), f"{context}.{field}") > calls:
            raise PublicDevelopmentPreviewError(f"{context} reports telemetry for too many calls")
    for field in (
        "cost_usd_lower_bound",
        "input_tokens_lower_bound",
        "output_tokens_lower_bound",
        "reasoning_tokens_lower_bound",
        "total_tokens_lower_bound",
        "latency_seconds_total",
    ):
        _nonnegative_number(stage.get(field), f"{context}.{field}")
    for field in ("latency_seconds_max", "latency_seconds_mean"):
        item = stage.get(field)
        if item is None:
            if calls:
                raise PublicDevelopmentPreviewError(f"{context}.{field} must be numeric")
        else:
            _nonnegative_number(item, f"{context}.{field}")
    attempts = _nonnegative_int(stage.get("attempts_lower_bound"), f"{context} attempts")
    retries = _nonnegative_int(stage.get("retries_lower_bound"), f"{context} retries")
    if attempts != calls + retries:
        raise PublicDevelopmentPreviewError(f"{context} attempt accounting is invalid")
    latency_total = float(stage["latency_seconds_total"])
    latency_max = stage.get("latency_seconds_max")
    latency_mean = stage.get("latency_seconds_mean")
    if calls:
        if (
            not isinstance(latency_mean, int | float)
            or not math.isclose(
                float(latency_mean),
                latency_total / calls,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not isinstance(latency_max, int | float)
            or float(latency_max) > latency_total + 1e-6
        ):
            raise PublicDevelopmentPreviewError(f"{context} latency accounting is invalid")
    elif latency_total != 0 or latency_max is not None or latency_mean is not None:
        raise PublicDevelopmentPreviewError(f"{context} zero-call latency is invalid")
    for field in _OPERATIONAL_STAGE_EXTRA_FIELDS[stage_name]:
        _nonnegative_int(stage.get(field), f"{context}.{field}")
    return dict(stage)


def _validate_public_usage_schema(
    usage: Mapping[str, object], corpus: Mapping[str, object]
) -> list[str]:
    _exact_keys(usage, _USAGE_TOP_FIELDS, "public usage")
    if (
        usage.get("schema_version") != PUBLIC_USAGE_SCHEMA_VERSION
        or usage.get("accounting_scope") != "sealed_development_source_run_only"
    ):
        raise PublicDevelopmentPreviewError("public usage schema/scope is unsupported")
    if not isinstance(usage.get("statement"), str) or not usage.get("statement"):
        raise PublicDevelopmentPreviewError("public usage statement is invalid")
    provider_budget = _mapping(usage.get("provider_budget"), "public provider budget")
    _exact_keys(provider_budget, _CORPUS_PROVIDER_FIELDS, "public provider budget")
    replayed = _validate_public_telemetry(
        usage.get("replayed_provider_usage"), "public replayed usage"
    )
    paper_count = _nonnegative_int(corpus.get("papers"), "public corpus paper count")
    stage_receipts = _mapping(usage.get("stage_receipts"), "public stage receipts")
    if set(stage_receipts) != set(_ROUTE_STAGES):
        raise PublicDevelopmentPreviewError("public stage-receipt set is invalid")
    operational = _mapping(usage.get("operational_accounting"), "operational accounting")
    _exact_keys(operational, _USAGE_OPERATIONAL_FIELDS, "operational accounting")
    _nonnegative_number(operational.get("wall_clock_seconds"), "operational wall clock")
    operational_stages = _mapping(operational.get("stages"), "operational stages")
    if set(operational_stages) != set(_ROUTE_STAGES):
        raise PublicDevelopmentPreviewError("public operational stage set is invalid")
    validated_receipts: dict[str, Mapping[str, object]] = {}
    validated_operational: dict[str, Mapping[str, object]] = {}
    for stage in _ROUTE_STAGES:
        receipt = _mapping(stage_receipts[stage], f"public receipt stage {stage}")
        _validate_stage_receipt(receipt, f"public receipt stage {stage}", paper_count=paper_count)
        operation = _validate_operational_stage(
            operational_stages[stage], stage, f"public operational stage {stage}"
        )
        if receipt.get("completed_calls") != operation.get("completed_calls"):
            raise PublicDevelopmentPreviewError("public stage call accounting disagrees")
        validated_receipts[stage] = receipt
        validated_operational[stage] = operation
    qualification = _mapping(usage.get("qualification_campaign_usage"), "qualification usage")
    _exact_keys(qualification, _USAGE_QUALIFICATION_FIELDS, "qualification usage")
    if qualification != {
        "status": "separate_not_included",
        "combined_with_source_run": False,
    }:
        raise PublicDevelopmentPreviewError("qualification and source-run usage were combined")
    paper_ids: list[str] = []
    stage_calls = {stage: 0 for stage in _ROUTE_STAGES}
    stage_usage_totals = {
        stage: {
            key: 0
            for key in (
                "cost_reported_calls",
                "cost_usd_lower_bound",
                "input_tokens_reported_calls",
                "input_tokens_lower_bound",
                "output_tokens_reported_calls",
                "output_tokens_lower_bound",
                "reasoning_tokens_reported_calls",
                "reasoning_tokens_lower_bound",
                "total_tokens_reported_calls",
                "total_tokens_lower_bound",
                "retries_lower_bound",
                "attempts_lower_bound",
            )
        }
        for stage in _ROUTE_STAGES
    }
    stage_execution_totals = {
        stage: {field: 0 for field in _STAGE_EXECUTION_FIELDS[stage]} for stage in _ROUTE_STAGES
    }
    for raw_paper in _sequence(usage.get("paper_accounting"), "usage papers"):
        paper = _mapping(raw_paper, "usage paper")
        _exact_keys(paper, _USAGE_PAPER_FIELDS, "usage paper")
        paper_ids.append(_safe_id(paper.get("paper_id"), "usage paper id"))
        if paper.get("status") not in {"success", "partial_failure"}:
            raise PublicDevelopmentPreviewError("usage paper status is invalid")
        _nonnegative_number(paper.get("wall_clock_seconds"), "usage paper wall clock")
        stages = _mapping(paper.get("stages"), "usage paper stages")
        if set(stages) != set(_ROUTE_STAGES):
            raise PublicDevelopmentPreviewError("usage paper stage set is invalid")
        for stage_name in _ROUTE_STAGES:
            stage = _mapping(stages[stage_name], f"usage paper stage {stage_name}")
            _exact_keys(stage, _USAGE_PAPER_STAGE_FIELDS, f"usage paper stage {stage_name}")
            if stage.get("status") not in {"validated", "partial_failure", "not_run"}:
                raise PublicDevelopmentPreviewError("usage paper stage status is invalid")
            _model_id(stage.get("selected_model"), "usage selected model")
            execution = _mapping(stage.get("execution"), "usage stage execution")
            _exact_keys(
                execution,
                _STAGE_EXECUTION_FIELDS[stage_name],
                f"usage paper {stage_name} execution",
            )
            for field, item in execution.items():
                stage_execution_totals[stage_name][field] += _nonnegative_int(
                    item, f"usage paper {stage_name} execution {field}"
                )
            stage_usage = _validate_stage_usage(
                stage.get("usage"),
                f"usage paper {stage_name} telemetry",
                require_exact=provider_budget.get("call_coverage") == "exhaustive",
            )
            stage_calls[stage_name] += int(stage_usage["calls_attempted"])
            for field in stage_usage_totals[stage_name]:
                stage_usage_totals[stage_name][field] += stage_usage[field]  # type: ignore[operator]
    if not paper_ids or len(paper_ids) != len(set(paper_ids)) or len(paper_ids) != paper_count:
        raise PublicDevelopmentPreviewError("usage paper IDs are invalid")
    for stage_name in _ROUTE_STAGES:
        receipt = validated_receipts[stage_name]
        operation = validated_operational[stage_name]
        if stage_calls[stage_name] != receipt.get("completed_calls"):
            raise PublicDevelopmentPreviewError("usage paper calls disagree with stage receipt")
        for field, total in stage_usage_totals[stage_name].items():
            if not _same_number(operation.get(field), total):
                raise PublicDevelopmentPreviewError(
                    "usage operational telemetry disagrees with paper accounting"
                )
        for field in _OPERATIONAL_STAGE_EXTRA_FIELDS[stage_name] & set(
            stage_execution_totals[stage_name]
        ):
            if operation.get(field) != stage_execution_totals[stage_name][field]:
                raise PublicDevelopmentPreviewError(
                    "usage operational execution disagrees with paper accounting"
                )

    replayed_calls = _nonnegative_int(
        replayed.get("recorded_structured_invocations"), "public replayed calls"
    )
    if replayed_calls != sum(stage_calls.values()):
        raise PublicDevelopmentPreviewError("public replayed call total disagrees")
    provider_completed = _nonnegative_int(
        provider_budget.get("structured_calls_completed"), "public provider completed calls"
    )
    provider_telemetry_calls = _nonnegative_int(
        provider_budget.get("provider_call_telemetry_calls"), "public provider telemetry calls"
    )
    provider_cost_calls = _nonnegative_int(
        provider_budget.get("provider_reported_cost_calls"), "public provider cost calls"
    )
    if (
        replayed_calls > provider_completed
        or replayed_calls > provider_telemetry_calls
        or int(replayed["cost_reported_calls"]) > provider_cost_calls
        or float(replayed["cost_usd_lower_bound"])
        > float(provider_budget["provider_reported_cost_usd_lower_bound"]) + 1e-10
        or any(
            int(replayed[f"{name}_tokens_lower_bound"])
            > int(provider_budget[f"provider_reported_{name}_tokens_lower_bound"])
            for name in ("input", "output", "reasoning", "total")
        )
    ):
        raise PublicDevelopmentPreviewError("public provider accounting understates replayed usage")
    if provider_budget.get("call_coverage") == "exhaustive" and (
        provider_completed != replayed_calls
        or provider_telemetry_calls != replayed_calls
        or provider_cost_calls != replayed.get("cost_reported_calls")
        or not _same_number(
            provider_budget.get("provider_reported_cost_usd_lower_bound"),
            replayed.get("cost_usd_lower_bound"),
        )
        or any(
            provider_budget.get(f"provider_reported_{name}_tokens_lower_bound")
            != replayed.get(f"{name}_tokens_lower_bound")
            for name in ("input", "output", "reasoning", "total")
        )
    ):
        raise PublicDevelopmentPreviewError("exhaustive public provider accounting disagrees")
    additive_fields = (
        "cost_reported_calls",
        "cost_usd_lower_bound",
        "input_tokens_reported_calls",
        "input_tokens_lower_bound",
        "output_tokens_reported_calls",
        "output_tokens_lower_bound",
        "reasoning_tokens_reported_calls",
        "reasoning_tokens_lower_bound",
        "total_tokens_reported_calls",
        "total_tokens_lower_bound",
        "retries_lower_bound",
        "attempts_lower_bound",
    )
    for field in additive_fields:
        expected = sum(validated_operational[stage][field] for stage in _ROUTE_STAGES)  # type: ignore[misc]
        if not _same_number(replayed.get(field), expected):
            raise PublicDevelopmentPreviewError("public replayed telemetry total disagrees")
    expected_max = (
        max(
            float(validated_operational[stage]["latency_seconds_max"] or 0.0)
            for stage in _ROUTE_STAGES
        )
        if replayed_calls
        else None
    )
    expected_total = sum(
        float(validated_operational[stage]["latency_seconds_total"]) for stage in _ROUTE_STAGES
    )
    expected_mean = (
        round(float(replayed["latency_seconds_total"]) / replayed_calls, 6)
        if replayed_calls
        else None
    )
    if (
        not math.isclose(
            float(replayed["latency_seconds_total"]),
            expected_total,
            rel_tol=0.0,
            abs_tol=len(_ROUTE_STAGES) * 1e-6,
        )
        or (expected_max is None and replayed.get("latency_seconds_max") is not None)
        or (
            expected_max is not None
            and not _same_number(replayed.get("latency_seconds_max"), expected_max)
        )
        or (expected_mean is None and replayed.get("latency_seconds_mean") is not None)
        or (
            expected_mean is not None
            and not _same_number(replayed.get("latency_seconds_mean"), expected_mean)
        )
    ):
        raise PublicDevelopmentPreviewError("public replayed latency accounting disagrees")
    return paper_ids


def _validate_public_evidence_schema(evidence: Mapping[str, object]) -> list[str]:
    _exact_keys(evidence, _EVIDENCE_TOP_FIELDS, "public evidence map")
    if (
        evidence.get("schema_version") != PUBLIC_EVIDENCE_MAP_SCHEMA_VERSION
        or evidence.get("status") != "model_proposed_human_review_pending"
        or not isinstance(evidence.get("statement"), str)
        or not evidence.get("statement")
    ):
        raise PublicDevelopmentPreviewError("public evidence-map schema/status is invalid")
    scope = _mapping(evidence.get("scope"), "evidence-map scope")
    _exact_keys(scope, _EVIDENCE_SCOPE_FIELDS, "evidence-map scope")
    if (
        scope.get("evaluation_split") != "development"
        or scope.get("holdout") is not False
        or scope.get("whole_paper_evaluation") is not False
        or scope.get("independent_human_validation") is not False
    ):
        raise PublicDevelopmentPreviewError("public evidence-map scope is overstated")
    paper_ids: list[str] = []
    for raw_paper in _sequence(evidence.get("papers"), "evidence-map papers"):
        paper = _mapping(raw_paper, "evidence-map paper")
        _exact_keys(paper, _EVIDENCE_PAPER_FIELDS, "evidence-map paper")
        paper_ids.append(_safe_id(paper.get("paper_id"), "evidence-map paper id"))
        page_scope = _mapping(paper.get("page_scope"), "evidence-map page scope")
        technical = _mapping(paper.get("technical"), "evidence-map technical")
        candidate = _mapping(paper.get("candidate_layer"), "evidence-map candidates")
        tuple_gate = _mapping(paper.get("tuple_gate"), "evidence-map tuple gate")
        verifier = _mapping(paper.get("independent_verifier"), "evidence-map independent verifier")
        origin = _mapping(paper.get("producer_origin"), "evidence-map producer origin")
        review = _mapping(paper.get("review_boundary"), "evidence-map review boundary")
        _exact_keys(page_scope, _EVIDENCE_PAGE_SCOPE_FIELDS, "evidence-map page scope")
        _exact_keys(technical, _EVIDENCE_TECHNICAL_FIELDS, "evidence-map technical")
        _exact_keys(candidate, _EVIDENCE_CANDIDATE_FIELDS, "evidence-map candidates")
        _exact_keys(tuple_gate, _EVIDENCE_TUPLE_FIELDS, "evidence-map tuple gate")
        _exact_keys(verifier, _EVIDENCE_VERIFIER_FIELDS, "evidence-map verifier")
        _exact_keys(origin, _EVIDENCE_ORIGIN_FIELDS, "evidence-map origin")
        _exact_keys(review, _EVIDENCE_REVIEW_FIELDS, "evidence-map review boundary")
        if (
            not isinstance(paper.get("title"), str)
            or not paper.get("title")
            or not isinstance(paper.get("venue"), str)
            or not paper.get("venue")
            or not isinstance(paper.get("perspective_role"), str)
            or not paper.get("perspective_role")
            or isinstance(paper.get("year"), bool)
            or not isinstance(paper.get("year"), int)
            or int(paper["year"]) < 1
            or technical.get("paper_status") not in {"success", "partial_failure"}
            or not isinstance(technical.get("needs_review"), bool)
        ):
            raise PublicDevelopmentPreviewError("evidence-map paper metadata is invalid")
        reasons = list(_sequence(technical.get("review_reasons"), "evidence review reasons"))
        if any(item not in _SUMMARY_REVIEW_REASON_FIELDS for item in reasons) or len(
            reasons
        ) != len(set(reasons)):
            raise PublicDevelopmentPreviewError("evidence-map review reasons are invalid")
        stages = _mapping(technical.get("stage_status"), "evidence-map stage status")
        if set(stages) != set(_ROUTE_STAGES):
            raise PublicDevelopmentPreviewError("evidence-map stage status set is invalid")
        if any(
            value not in {"validated", "partial_failure", "not_run"} for value in stages.values()
        ):
            raise PublicDevelopmentPreviewError("evidence-map stage status is invalid")
        for group_name, group in (
            ("candidate", candidate),
            ("tuple", tuple_gate),
            ("verifier", verifier),
            ("origin", origin),
        ):
            for key, value in group.items():
                if key != "automatic_positive_promotion":
                    _nonnegative_int(value, f"evidence-map {group_name} {key}")
        selected_pages = list(_sequence(page_scope.get("selected_pages"), "evidence-map pages"))
        if (
            page_scope.get("selection_mode") not in {"configured_page_scope", "automatic_selector"}
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in selected_pages
            )
            or len(selected_pages) != len(set(selected_pages))
        ):
            raise PublicDevelopmentPreviewError("evidence-map page scope is invalid")
        candidates = _nonnegative_int(candidate.get("deduplicated_candidates"), "candidates")
        accepts = _nonnegative_int(review.get("accepted_by_model_gates"), "model-gate accepts")
        not_accepted = _nonnegative_int(
            review.get("not_accepted_by_complete_model_chain"), "non-accepts"
        )
        if (
            accepts != verifier.get("accepts")
            or accepts + not_accepted != candidates
            or review.get("withheld_from_canonical_eee") != candidates
            or review.get("human_review_decisions_completed") != 0
            or review.get("canonical_eee_records") != 0
            or review.get("final_reviewed_records") is not None
            or review.get("status") != "model_proposed_human_review_pending"
            or origin.get("automatic_positive_promotion") is not False
        ):
            raise PublicDevelopmentPreviewError("evidence-map review boundary is invalid")
    if (
        not paper_ids
        or len(paper_ids) != len(set(paper_ids))
        or scope.get("papers") != len(paper_ids)
    ):
        raise PublicDevelopmentPreviewError("evidence-map paper population is invalid")
    return paper_ids


def _validate_public_verification_schema(verification: Mapping[str, object]) -> None:
    _exact_keys(verification, _VERIFICATION_TOP_FIELDS, "public verification")
    if (
        verification.get("schema_version") != PUBLIC_PREVIEW_VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != "verified_pre_human_preview"
        or verification.get("release_ready") is not False
    ):
        raise PublicDevelopmentPreviewError("public preview verification status is invalid")
    checks = _mapping(verification.get("checks"), "verification checks")
    bindings = _mapping(verification.get("bindings"), "verification bindings")
    _exact_keys(checks, _VERIFICATION_CHECK_FIELDS, "verification checks")
    _exact_keys(bindings, _VERIFICATION_BINDING_FIELDS, "verification bindings")
    if any(value is not True for value in checks.values()):
        raise PublicDevelopmentPreviewError("public preview contains an unpassed verification")
    for field, value in bindings.items():
        _sha256(value, f"verification binding {field}")


def _validate_standalone_bundle(contents: Mapping[str, bytes]) -> dict[str, object]:
    manifest = _mapping(
        _strict_json_bytes(contents["publication-manifest.json"], "publication manifest"),
        "publication manifest",
    )
    _validate_public_manifest_schema(manifest)
    bundle_id = _safe_id(manifest.get("bundle_id"), "bundle id")

    corpus = _mapping(_strict_json_bytes(contents["corpus.json"], "public corpus"), "public corpus")
    summary = _mapping(
        _strict_json_bytes(contents["run-summary.json"], "public run summary"),
        "public run summary",
    )
    sources = _mapping(
        _strict_json_bytes(contents["sources.json"], "public sources"), "public sources"
    )
    usage = _mapping(_strict_json_bytes(contents["usage.json"], "public usage"), "public usage")
    evidence = _mapping(
        _strict_json_bytes(contents["evidence-map.json"], "public evidence map"),
        "public evidence map",
    )
    verification = _mapping(
        _strict_json_bytes(contents["verification.json"], "public verification"),
        "public verification",
    )
    route = _mapping(
        _strict_json_bytes(contents["route-selection.json"], "public route"), "public route"
    )
    corpus_paper_ids = _validate_public_corpus_schema(corpus)
    summary_telemetry = _validate_public_summary_schema(summary, corpus)
    route_models = _validate_public_route_schema(route)
    source_paper_ids = _validate_public_sources_schema(sources)
    usage_paper_ids = _validate_public_usage_schema(usage, corpus)
    evidence_paper_ids = _validate_public_evidence_schema(evidence)
    _validate_public_verification_schema(verification)

    source_run = _mapping(manifest.get("source_run"), "manifest source run")
    prospective = _mapping(manifest.get("prospective_execution"), "prospective execution")
    scope = _mapping(evidence.get("scope"), "evidence-map scope")
    verification_bindings = _mapping(verification.get("bindings"), "verification bindings")
    corpus_id = corpus.get("corpus_id")
    papers = corpus.get("papers")
    corpus_papers = [
        _mapping(item, "public corpus paper")
        for item in _sequence(corpus.get("papers_detail"), "public corpus papers")
    ]
    run_binding = _mapping(summary.get("run_binding"), "public summary run binding")
    if any(
        value != corpus_id
        for value in (source_run.get("corpus_id"), sources.get("corpus_id"), scope.get("corpus_id"))
    ) or any(
        value != papers
        for value in (
            source_run.get("paper_count"),
            scope.get("papers"),
            len(source_paper_ids),
            len(evidence_paper_ids),
            len(usage_paper_ids),
            len(corpus_paper_ids),
        )
    ):
        raise PublicDevelopmentPreviewError("public preview cross-file corpus binding disagrees")
    if not (corpus_paper_ids == source_paper_ids == usage_paper_ids == evidence_paper_ids):
        raise PublicDevelopmentPreviewError("public preview paper order/identity disagrees")
    source_by_id = {
        str(_mapping(item, "public source paper")["paper_id"]): _mapping(
            item, "public source paper"
        )
        for item in _sequence(sources.get("papers"), "public source papers")
    }
    evidence_by_id = {
        str(_mapping(item, "public evidence paper")["paper_id"]): _mapping(
            item, "public evidence paper"
        )
        for item in _sequence(evidence.get("papers"), "public evidence papers")
    }
    usage_by_id = {
        str(_mapping(item, "public usage paper")["paper_id"]): _mapping(item, "public usage paper")
        for item in _sequence(usage.get("paper_accounting"), "public usage papers")
    }
    evidence_lineage_totals = {
        "proposals": 0,
        "candidate_occurrences": 0,
        "final_candidates": 0,
        "merged_candidates": 0,
        "deterministically_not_eligible": 0,
    }
    evidence_review_reasons: dict[str, int] = {}
    for corpus_paper in corpus_papers:
        paper_id = str(corpus_paper["paper_id"])
        source_paper = source_by_id[paper_id]
        evidence_paper = evidence_by_id[paper_id]
        usage_paper = usage_by_id[paper_id]
        source_scope = _mapping(source_paper.get("page_scope"), "public source page scope")
        evidence_scope = _mapping(evidence_paper.get("page_scope"), "evidence page scope")
        evidence_technical = _mapping(evidence_paper.get("technical"), "evidence technical")
        evidence_reasons = list(
            _sequence(evidence_technical.get("review_reasons"), "evidence review reasons")
        )
        for reason in evidence_reasons:
            evidence_review_reasons[str(reason)] = evidence_review_reasons.get(str(reason), 0) + 1
        if (
            corpus_paper.get("title") != source_paper.get("title")
            or corpus_paper.get("title") != evidence_paper.get("title")
            or any(
                source_paper.get(field) != evidence_paper.get(field)
                for field in ("year", "venue", "perspective_role")
            )
            or corpus_paper.get("source_manifest_sha256")
            != source_paper.get("source_manifest_sha256")
            or corpus_paper.get("selected_pages") != source_scope.get("selected_pages")
            or corpus_paper.get("selected_pages") != evidence_scope.get("selected_pages")
            or corpus_paper.get("status") != usage_paper.get("status")
            or corpus_paper.get("status") != evidence_technical.get("paper_status")
            or evidence_technical.get("needs_review") is not bool(evidence_reasons)
            or not _same_number(
                corpus_paper.get("wall_clock_seconds"), usage_paper.get("wall_clock_seconds")
            )
        ):
            raise PublicDevelopmentPreviewError("public preview per-paper binding disagrees")
        usage_stages = _mapping(usage_paper.get("stages"), "usage paper stages")
        evidence_stages = _mapping(evidence_technical.get("stage_status"), "evidence stages")
        for route_stage, corpus_stage in _PUBLIC_TO_CORPUS_STAGE.items():
            corpus_stage_value = _mapping(
                corpus_paper.get(corpus_stage), f"corpus paper {corpus_stage}"
            )
            usage_stage = _mapping(usage_stages.get(route_stage), f"usage stage {route_stage}")
            if (
                usage_stage.get("selected_model") != corpus_stage_value.get("model")
                or usage_stage.get("selected_model") != route_models[route_stage]
                or any(
                    returned_model != route_models[route_stage]
                    for returned_model in _sequence(
                        _mapping(usage_stage.get("usage"), "usage stage telemetry").get(
                            "models_returned"
                        ),
                        "usage returned models",
                    )
                )
                or usage_stage.get("execution") != corpus_stage_value.get("execution")
                or usage_stage.get("usage") != corpus_stage_value.get("usage")
                or usage_stage.get("status") != evidence_stages.get(route_stage)
            ):
                raise PublicDevelopmentPreviewError("public preview stage projection disagrees")

        counts = _mapping(corpus_paper.get("counts"), "public paper counts")
        for summary_field, corpus_stage_name in (
            ("extractor", "extractor"),
            ("row_extractor", "row_enumeration"),
        ):
            corpus_stage_binding = _mapping(
                corpus_paper.get(corpus_stage_name), f"corpus {corpus_stage_name}"
            )
            expected_summary_binding = {
                field: corpus_stage_binding.get(field)
                for field in _SUMMARY_MODEL_BINDING_FIELDS - {"request_contract_sha256"}
            }
            expected_summary_binding["request_contract_sha256"] = sha256_bytes(
                canonical_json_bytes(corpus_stage_binding.get("request_contract"))
            )
            if run_binding.get(summary_field) != expected_summary_binding:
                raise PublicDevelopmentPreviewError(
                    "public summary stage binding disagrees with corpus"
                )
        if (
            run_binding.get("candidate_validation") != corpus_paper.get("candidate_validation")
            or run_binding.get("eee_schema") != corpus_paper.get("eee_schema")
            or run_binding.get("code") != corpus_paper.get("code")
        ):
            raise PublicDevelopmentPreviewError("public summary run binding disagrees with corpus")
        candidate_layer = _mapping(evidence_paper.get("candidate_layer"), "evidence candidates")
        tuple_gate = _mapping(evidence_paper.get("tuple_gate"), "evidence tuple gate")
        verifier = _mapping(evidence_paper.get("independent_verifier"), "evidence verifier")
        origin = _mapping(evidence_paper.get("producer_origin"), "evidence origin")
        review = _mapping(evidence_paper.get("review_boundary"), "evidence review boundary")
        expected_evidence_counts = {
            "deduplicated_candidates": counts.get("candidates"),
            "duplicates_removed": counts.get("duplicates_removed"),
            "primary_result_candidates": counts.get("primary_results"),
            "requires_human_review": counts.get("candidates_needing_review"),
        }
        if any(
            candidate_layer.get(key) != value for key, value in expected_evidence_counts.items()
        ):
            raise PublicDevelopmentPreviewError("public evidence candidate counts disagree")
        evidence_lineage_totals["proposals"] += int(candidate_layer["model_proposals"])
        evidence_lineage_totals["candidate_occurrences"] += int(
            candidate_layer["candidate_occurrences"]
        )
        evidence_lineage_totals["final_candidates"] += int(
            candidate_layer["deduplicated_candidates"]
        )
        evidence_lineage_totals["merged_candidates"] += int(candidate_layer["duplicates_removed"])
        evidence_lineage_totals["deterministically_not_eligible"] += int(
            candidate_layer["deterministically_not_eligible"]
        )
        expected_tuple_counts = {
            "selected": counts.get("tuple_candidates"),
            "passed": counts.get("tuple_passed"),
            "routed_to_review": counts.get("tuple_review"),
            "unsupported_subset": counts.get("tuple_unsupported"),
            "failed_subset": counts.get("tuple_failed"),
        }
        expected_verifier_counts = {
            "selected": _mapping(
                _mapping(corpus_paper.get("verifier"), "corpus verifier").get("execution"),
                "corpus verifier execution",
            ).get("candidates_selected"),
            "verified": counts.get("verifications"),
            "accepts": counts.get("verifier_accepts"),
            "rejects": counts.get("verifier_rejects"),
            "reviews": counts.get("verifier_reviews"),
            "failed_subset": counts.get("verifier_failed"),
        }
        expected_origin_counts = {
            "selected": counts.get("origin_candidates"),
            "external": counts.get("origin_external"),
            "unresolved": counts.get("origin_unresolved"),
            "no_signal": counts.get("origin_no_signal"),
            "positive_review_only_subset": counts.get("origin_positive_review_only"),
            "failed_subset": counts.get("origin_failed"),
        }
        if (
            any(tuple_gate.get(key) != value for key, value in expected_tuple_counts.items())
            or any(verifier.get(key) != value for key, value in expected_verifier_counts.items())
            or any(origin.get(key) != value for key, value in expected_origin_counts.items())
            or review.get("withheld_from_canonical_eee") != counts.get("candidates")
        ):
            raise PublicDevelopmentPreviewError("public evidence stage counts disagree")
    summary_outputs = _mapping(summary.get("outputs"), "summary outputs")
    summary_lineage = _mapping(
        summary_outputs.get("candidate_lineage"), "summary candidate lineage"
    )
    summary_health = _mapping(summary.get("technical_health"), "summary technical health")
    summary_review_reasons = _mapping(
        summary_health.get("review_reason_counts"), "summary review reasons"
    )
    summary_export_status = _mapping(
        summary_outputs.get("export_status_counts"), "summary export statuses"
    )
    if (
        any(
            evidence_lineage_totals[field] != summary_lineage.get(field)
            for field in (
                "proposals",
                "candidate_occurrences",
                "final_candidates",
                "merged_candidates",
            )
        )
        or evidence_lineage_totals["deterministically_not_eligible"]
        != summary_export_status.get("not_eligible", 0)
        or dict(sorted(evidence_review_reasons.items())) != dict(summary_review_reasons)
    ):
        raise PublicDevelopmentPreviewError("public evidence aggregate counts disagree")
    if (
        summary.get("current_sealed_receipt") is not True
        or _mapping(summary.get("canonical_eee"), "summary canonical EEE").get("records") != 0
        or evidence.get("status") != "model_proposed_human_review_pending"
        or prospective.get("route_selection_sha256")
        != sha256_bytes(contents["route-selection.json"])
    ):
        raise PublicDevelopmentPreviewError("public preview cross-file status binding disagrees")
    if sum(
        bool(_mapping(item, "evidence paper").get("technical", {}).get("needs_review"))
        for item in _sequence(evidence.get("papers"), "evidence papers")
    ) != corpus.get("papers_needing_review"):
        raise PublicDevelopmentPreviewError("public preview review population disagrees")
    source_artifacts = _mapping(corpus.get("source_artifacts"), "public corpus source artifacts")
    run_seal = _mapping(source_run.get("run_seal"), "manifest run seal")
    source_code = _mapping(source_run.get("code"), "manifest source code")
    prospective_budget = _mapping(prospective.get("budget"), "prospective execution budget")
    public_provider_budget = _mapping(usage.get("provider_budget"), "public provider budget")
    replayed_usage = _mapping(usage.get("replayed_provider_usage"), "public replayed usage")
    max_structured_calls = _nonnegative_int(
        prospective_budget.get("max_structured_calls"), "prospective structured-call ceiling"
    )
    max_transport_attempts = _nonnegative_int(
        prospective_budget.get("max_transport_attempts"), "prospective transport-attempt ceiling"
    )
    if (
        _nonnegative_int(
            public_provider_budget.get("structured_calls_started"),
            "public provider calls started",
        )
        > max_structured_calls
        or _nonnegative_int(
            public_provider_budget.get("structured_calls_completed"),
            "public provider calls completed",
        )
        > max_structured_calls
        or _nonnegative_int(
            replayed_usage.get("attempts_lower_bound"),
            "public replayed transport attempts",
        )
        > max_transport_attempts
    ):
        raise PublicDevelopmentPreviewError(
            "public provider usage exceeds a prospective execution ceiling"
        )
    if (
        prospective.get("selected_models") != route_models
        or prospective.get("route_status") != route.get("status")
        or source_code.get("git_commit") != route.get("code_commit")
        or usage.get("provider_budget") != corpus.get("provider_accounting")
        or source_artifacts.get("corpus_run_sha256") != source_run.get("corpus_run_sha256")
        or run_binding.get("corpus_id") != corpus_id
        or run_binding.get("generated_at") != corpus.get("generated_at")
        or run_binding.get("generated_at") != source_run.get("generated_at")
        or run_binding.get("corpus_run_sha256") != source_run.get("corpus_run_sha256")
        or run_binding.get("paper_ids_sha256") != source_run.get("paper_ids_sha256")
        or run_binding.get("recorded_corpus_spec_sha256") != source_run.get("corpus_spec_sha256")
        or run_binding.get("code") != source_code
        or any(
            run_binding.get("run_seal", {}).get(field) != run_seal.get(field)
            for field in ("schema_version", "seal_sha256", "tree_sha256", "file_count")
        )
        or summary_telemetry != usage.get("replayed_provider_usage")
        or _mapping(
            _mapping(run_binding.get("stage_chain"), "summary stage chain").get(
                "checkpoint_validation"
            ),
            "summary checkpoints",
        )
        != {
            _ROUTE_TO_RECEIPT_STAGE[stage]: _mapping(
                usage.get("stage_receipts"), "usage stage receipts"
            )[stage]
            for stage in _ROUTE_STAGES
        }
        or verification_bindings.get("source_run_seal_sha256") != run_seal.get("seal_sha256")
        or verification_bindings.get("source_run_tree_sha256") != run_seal.get("tree_sha256")
        or verification_bindings.get("corpus_run_sha256") != source_run.get("corpus_run_sha256")
        or verification_bindings.get("corpus_spec_sha256") != source_run.get("corpus_spec_sha256")
        or verification_bindings.get("paper_ids_sha256") != source_run.get("paper_ids_sha256")
        or verification_bindings.get("route_selection_sha256")
        != prospective.get("route_selection_sha256")
        or verification_bindings.get("precall_contract_sha256")
        != prospective.get("precall_contract_sha256")
        or verification_bindings.get("precall_file_sha256")
        != prospective.get("precall_file_sha256")
    ):
        raise PublicDevelopmentPreviewError("public preview cryptographic binding disagrees")
    expected_html = _render_evidence_map(evidence)
    if contents["evidence-map.html"].decode("utf-8") != expected_html:
        raise PublicDevelopmentPreviewError("public evidence-map HTML is not deterministic")
    expected_readme = _render_readme(manifest, evidence)
    if contents["README.md"].decode("utf-8") != expected_readme:
        raise PublicDevelopmentPreviewError("public preview README is not deterministic")
    return {
        "schema_version": PUBLIC_PREVIEW_VERIFICATION_SCHEMA_VERSION,
        "status": "verified",
        "bundle_id": bundle_id,
        "file_count": len(_PREVIEW_FILES),
        "checksums_sha256": sha256_bytes(contents["SHA256SUMS"]),
    }


def verify_public_development_preview(root: Path) -> dict[str, object]:
    """Verify one standalone public preview without reading its private source run."""

    absolute = _reject_symlink_components(root, "public preview root")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise PublicDevelopmentPreviewError("public preview root does not exist") from error
    contents, root_identity = _capture_directory_files(
        resolved,
        expected_names=_PREVIEW_FILES,
        context="public preview root",
    )
    _audit_preview_contents(contents, expect_checksums=True)
    _verify_checksums(contents)
    result = _validate_standalone_bundle(contents)
    _assert_capture_still_current(resolved, contents, root_identity)
    return result


def build_public_development_preview(
    *,
    bundle_id: str,
    run_root: Path,
    corpus_path: Path,
    route_manifest_path: Path,
    output_root: Path,
    expected_paper_count: int = 10,
) -> Path:
    """Build an immutable-shape pre-human preview from one sealed current run."""

    bundle_id = _safe_id(bundle_id, "bundle id")
    raw_run_root = _reject_symlink_components(run_root, "sealed run input")
    raw_corpus_path = _reject_symlink_components(corpus_path, "corpus input")
    raw_route_path = _reject_symlink_components(route_manifest_path, "route input")
    raw_output_root = _reject_symlink_components(output_root, "public preview output root")
    try:
        source_root = raw_run_root.resolve(strict=True)
        corpus_path = raw_corpus_path.resolve(strict=True)
        route_manifest_path = raw_route_path.resolve(strict=True)
    except OSError as error:
        raise PublicDevelopmentPreviewError("public preview input does not exist") from error
    raw_destination = raw_output_root / bundle_id
    if _is_within(raw_destination, source_root) or _is_within(source_root, raw_destination):
        raise PublicDevelopmentPreviewError("public preview output must be outside the sealed run")
    try:
        raw_output_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PublicDevelopmentPreviewError(
            "public preview output root could not be created"
        ) from error
    _reject_symlink_components(raw_output_root, "public preview output root")
    try:
        destination_parent = raw_output_root.resolve(strict=True)
    except OSError as error:
        raise PublicDevelopmentPreviewError("public preview output root is invalid") from error
    destination = destination_parent / bundle_id
    if _is_within(destination, source_root) or _is_within(source_root, destination):
        raise PublicDevelopmentPreviewError("public preview output must be outside the sealed run")
    if os.path.lexists(destination):
        raise PublicDevelopmentPreviewError("public preview destination already exists")

    seal_before = _verify_seal(source_root)
    try:
        receipt = validate_completed_corpus_run(source_root)
    except CompletedCorpusValidationError as error:
        raise PublicDevelopmentPreviewError("sealed five-stage receipt replay failed") from error
    try:
        projected_corpus = _project_corpus(source_root)
    except PublicSnapshotError as error:
        raise PublicDevelopmentPreviewError("sealed public-corpus projection failed") from error
    try:
        summary = build_public_development_summary(source_root)
    except PublicDevelopmentSummaryError as error:
        raise PublicDevelopmentPreviewError("sealed public-summary projection failed") from error
    spec, corpus_binding, corpus_path_sha256 = _load_bound_corpus(corpus_path)
    _validate_corpus_projection(
        spec=spec,
        binding=corpus_binding,
        receipt=receipt,
        projected=projected_corpus,
        summary=summary,
        expected_paper_count=expected_paper_count,
    )
    route = _validate_route_manifest(
        route_manifest_path,
        receipt=receipt,
        corpus=projected_corpus,
    )
    precall = _validate_precall_seal(
        run_root=source_root,
        route_path=route_manifest_path,
        route=route,
        corpus_path_sha256=corpus_path_sha256,
        corpus_binding=corpus_binding,
        receipt=receipt,
    )
    stage_receipts = _validate_usage_reconciliation(receipt=receipt, corpus=projected_corpus)
    sources = _project_sources_without_local_uris(
        run_root=source_root,
        spec=spec,
        corpus=projected_corpus,
    )
    usage = _build_usage(
        receipt=receipt,
        corpus=projected_corpus,
        stage_receipts=stage_receipts,
    )
    evidence_map = _build_evidence_map(receipt=receipt, corpus=projected_corpus, spec=spec)
    manifest = _build_manifest(
        bundle_id=bundle_id,
        receipt=receipt,
        route=route,
        precall=precall,
    )
    verification = _build_verification(receipt=receipt, precall=precall)
    payloads: dict[str, object] = {
        "publication-manifest.json": manifest,
        "run-summary.json": summary,
        "corpus.json": projected_corpus,
        "route-selection.json": route,
        "sources.json": sources,
        "usage.json": usage,
        "evidence-map.json": evidence_map,
        "verification.json": verification,
    }
    evidence_texts = _private_evidence_texts(source_root)
    for name, payload in payloads.items():
        _audit_public_value(payload, name)
        _assert_no_evidence_text(payload, evidence_texts)
    readme = _render_readme(manifest, evidence_map)
    html = _render_evidence_map(evidence_map)
    _scan_text(readme, "README.md")
    _scan_text(html, "evidence-map.html")

    staging = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.", dir=destination_parent))
    try:
        for name, payload in payloads.items():
            write_json(staging / name, payload)
        atomic_write_bytes(staging / "README.md", readme.encode("utf-8"))
        atomic_write_bytes(staging / "evidence-map.html", html.encode("utf-8"))
        partial_contents, partial_identity = _capture_directory_files(
            staging,
            expected_names=_PREVIEW_FILES - {"SHA256SUMS"},
            context="public preview staging root",
        )
        _audit_preview_contents(partial_contents, expect_checksums=False)
        _assert_capture_still_current(staging, partial_contents, partial_identity)
        _write_checksums(staging)
        contents, staging_identity = _capture_directory_files(
            staging,
            expected_names=_PREVIEW_FILES,
            context="public preview staging root",
        )
        _audit_preview_contents(contents, expect_checksums=True)
        _verify_checksums(contents)
        _validate_standalone_bundle(contents)
        seal_after = _verify_seal(source_root)
        if (
            seal_after.seal_sha256 != seal_before.seal_sha256
            or seal_after.tree_sha256 != seal_before.tree_sha256
            or seal_after.file_count != seal_before.file_count
            or seal_after.total_bytes != seal_before.total_bytes
        ):
            raise PublicDevelopmentPreviewError("sealed source run changed during projection")
        _assert_capture_still_current(staging, contents, staging_identity)
        destination = _publish_contents_exclusive(
            parent=destination_parent,
            bundle_id=bundle_id,
            contents=contents,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


__all__ = [
    "PRIVATE_EVIDENCE_SET_SCHEMA_VERSION",
    "PRIVATE_EVIDENCE_TEXT_CANONICALIZATION",
    "PUBLIC_DEVELOPMENT_PREVIEW_SCHEMA_VERSION",
    "PublicDevelopmentPreviewError",
    "assert_no_private_evidence_text",
    "build_public_development_preview",
    "collect_private_evidence_texts",
    "private_evidence_set_binding",
    "verify_public_development_preview",
]
