"""Build a deterministic, allowlist-only public snapshot from private run artifacts."""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from proceedings_to_eee.io import (
    atomic_write_bytes,
    read_json,
    sha256_file,
    write_json,
)
from proceedings_to_eee.reporting.corpus_html import render_corpus_html
from proceedings_to_eee.sources.manifest import SourceManifest
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record

SNAPSHOT_SCHEMA_VERSION = "public-pilot-snapshot/0.1"
MODEL_SELECTION_SCHEMA_VERSION = "public-model-selection/0.1"
SOURCES_SCHEMA_VERSION = "public-source-index/0.1"
REFERENCE_AUDIT_SCHEMA_VERSION = "public-reference-audit/0.1"
HUMAN_REVIEW_SCHEMA_VERSION = "human-review-summary/0.1"

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_LOCAL_PATH = re.compile(
    r"(?:^|[\s\"'=:(])(?:/Users/|/home/|/private/|/tmp/|/var/folders/|"
    r"[A-Za-z]:\\Users\\|file://)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~-]{12,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|authorization)\s*[:=]\s*[\"'][^\"']{8,}[\"']", re.I),
)
_FORBIDDEN_PUBLIC_KEYS = {
    "api_key",
    "authorization",
    "cache_relpath",
    "calls",
    "command",
    "completion",
    "cookie",
    "exact_quote",
    "manifest_path",
    "messages",
    "output_path",
    "prompt",
    "prompt_template",
    "provider_request",
    "provider_response",
    "quote",
    "raw_payload",
    "raw_response",
    "reference_path",
    "request_id",
    "schema_path",
    "secret",
    "warnings",
}
_TOP_LEVEL_FILES = {
    "README.md",
    "SHA256SUMS",
    "corpus-review.html",
    "human-review.json",
    "model-selection.json",
    "reference-audit.json",
    "snapshot.json",
    "sources.json",
}
_CLAIM_TYPES = (
    "primary_result",
    "secondary_claim",
    "illustration",
    "method_metadata",
    "uncertain",
)
_QUALITY_FIELDS = (
    "claim_type",
    "system",
    "dataset",
    "metric",
    "value",
    "unit",
    "slice",
    "page",
    "evidence_kind",
    "evidence_label",
    "evidence_row",
    "evidence_column",
    "evidence_structure",
    "evidence_supported",
    "missingness",
    "joint_semantics",
)
_COUNT_FIELDS = (
    "candidates",
    "primary_results",
    "exported",
    "eee_records",
    "eee_schema_issues",
    "verifications",
    "verifier_accepts",
    "verifier_rejects",
    "verifier_reviews",
    "spot_checks",
    "spot_checks_exact",
    "reference_observations",
    "reference_true_positives",
    "reference_false_positives",
    "reference_false_negatives",
    "negative_control_false_primary",
)
_SEGMENTATION_FIELDS = (
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
)
_REVIEW_RISK_REASONS = (
    "exported",
    "eligible",
    "needs_review",
    "text_support_risk",
    "referential_risk",
    "roles_missing",
    "role_version_missing",
    "role_confidence_low",
    "complex_role_assignment",
    "metric_unit_missing",
    "value_unit_missing",
    "unit_mismatch",
    "dataset_version_missing",
    "extraction_confidence_low",
    "table_anchor",
    "table_structure_incomplete",
    "non_exact_value",
    "no_candidates",
)
_REVIEW_OUTCOMES = ("confirmed", "incorrect", "needs_followup")
_REVIEW_ISSUES = (
    "claim_type",
    "role",
    "version",
    "scope",
    "metric",
    "unit",
    "value",
    "evidence",
    "export_decision",
    "duplicate",
    "other",
)
_REVIEW_ITEM_TYPES = ("candidate", "paper_without_candidates")


class PublicSnapshotError(ValueError):
    """The private input could not be represented by the public contract safely."""


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicSnapshotError(f"{context} must be a JSON object")
    return value


def _as_sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise PublicSnapshotError(f"{context} must be a JSON array")
    return value


def _safe_scalar(value: Any, context: str) -> str | int | float | bool | None:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise PublicSnapshotError(f"{context} must be a finite JSON scalar")


def _fields(
    value: Any,
    allowed: Sequence[str],
    context: str,
) -> dict[str, Any]:
    source = _as_mapping(value, context)
    return {key: _safe_scalar(source[key], f"{context}.{key}") for key in allowed if key in source}


def _nonnegative_int(value: Any, context: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PublicSnapshotError(f"{context} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise PublicSnapshotError(f"{context} must be {qualifier}")
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PublicSnapshotError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PublicSnapshotError(f"{context} must be a finite number")
    return result


def _project_named_counts(
    value: Any,
    allowed: Sequence[str],
    context: str,
    *,
    require_all: bool = False,
) -> dict[str, int]:
    source = _as_mapping(value, context)
    unexpected = set(source) - set(allowed)
    if unexpected:
        raise PublicSnapshotError(f"{context} contains unexpected labels")
    if require_all and set(source) != set(allowed):
        raise PublicSnapshotError(f"{context} must contain every required label")
    return {
        name: _nonnegative_int(source[name], f"{context}.{name}")
        for name in allowed
        if name in source
    }


def _project_human_review_summary(path: Path) -> dict[str, Any]:
    """Validate and allowlist one fully decided aggregate review summary."""

    raw = _as_mapping(read_json(path), "human review summary")
    if raw.get("schema_version") != HUMAN_REVIEW_SCHEMA_VERSION:
        raise PublicSnapshotError("human review input is not an aggregate review summary")
    audit_id = raw.get("audit_id")
    if not isinstance(audit_id, str) or not re.fullmatch(r"audit_[0-9a-f]{20}", audit_id):
        raise PublicSnapshotError("human review summary has an invalid audit_id")
    sampling_policy = raw.get("sampling_policy")
    if sampling_policy != "risk-stratified-paper-coverage/0.1":
        raise PublicSnapshotError("human review summary has an unsupported sampling policy")

    population_raw = _as_mapping(raw.get("population"), "human review population")
    candidates = _nonnegative_int(
        population_raw.get("candidates"), "human review population.candidates"
    )
    papers = _nonnegative_int(
        population_raw.get("papers"), "human review population.papers", positive=True
    )
    papers_without_candidates = _nonnegative_int(
        population_raw.get("papers_without_candidates", 0),
        "human review population.papers_without_candidates",
    )
    if papers_without_candidates > papers:
        raise PublicSnapshotError("papers_without_candidates exceeds paper population")

    sample_raw = _as_mapping(raw.get("sample"), "human review sample")
    requested = _nonnegative_int(
        sample_raw.get("requested"), "human review sample.requested", positive=True
    )
    reviewed = _nonnegative_int(
        sample_raw.get("reviewed"), "human review sample.reviewed", positive=True
    )
    if reviewed > requested:
        raise PublicSnapshotError("human review contains more items than requested")
    papers_reviewed = _nonnegative_int(
        sample_raw.get("papers_reviewed"),
        "human review sample.papers_reviewed",
        positive=True,
    )
    if papers_reviewed > papers:
        raise PublicSnapshotError("human review covers more papers than the population")
    if papers_reviewed > reviewed:
        raise PublicSnapshotError("human review covers more papers than reviewed items")
    if reviewed > candidates + papers_without_candidates:
        raise PublicSnapshotError("human review exceeds the reviewable population")
    paper_coverage = _finite_number(
        sample_raw.get("paper_coverage"), "human review sample.paper_coverage"
    )
    expected_coverage = round(papers_reviewed / papers, 6)
    if paper_coverage != expected_coverage:
        raise PublicSnapshotError("human review paper_coverage is inconsistent")
    risk_score_min = _finite_number(
        sample_raw.get("risk_score_min"), "human review sample.risk_score_min"
    )
    risk_score_max = _finite_number(
        sample_raw.get("risk_score_max"), "human review sample.risk_score_max"
    )
    risk_score_mean = _finite_number(
        sample_raw.get("risk_score_mean"), "human review sample.risk_score_mean"
    )
    if min(risk_score_min, risk_score_max, risk_score_mean) < 0:
        raise PublicSnapshotError("human review risk scores must be non-negative")
    if not risk_score_min <= risk_score_mean <= risk_score_max:
        raise PublicSnapshotError("human review risk score summary is inconsistent")
    risk_reason_counts = _project_named_counts(
        sample_raw.get("risk_reason_counts", {}),
        _REVIEW_RISK_REASONS,
        "human review sample.risk_reason_counts",
    )
    if any(count > reviewed for count in risk_reason_counts.values()):
        raise PublicSnapshotError("human review risk-reason count exceeds reviewed items")
    item_type_counts = _project_named_counts(
        sample_raw.get("item_type_counts", {"candidate": reviewed}),
        _REVIEW_ITEM_TYPES,
        "human review sample.item_type_counts",
    )
    if sum(item_type_counts.values()) != reviewed:
        raise PublicSnapshotError("human review item-type counts do not equal reviewed items")
    absence_reviewed = _nonnegative_int(
        sample_raw.get(
            "papers_without_candidates_reviewed",
            item_type_counts.get("paper_without_candidates", 0),
        ),
        "human review sample.papers_without_candidates_reviewed",
    )
    if absence_reviewed != item_type_counts.get("paper_without_candidates", 0):
        raise PublicSnapshotError("human review absence counts are inconsistent")
    if absence_reviewed > papers_without_candidates:
        raise PublicSnapshotError("reviewed candidate absences exceed the population")
    if absence_reviewed > papers_reviewed:
        raise PublicSnapshotError("reviewed candidate absences exceed reviewed papers")
    candidate_items = item_type_counts.get("candidate", 0)
    if candidate_items > candidates:
        raise PublicSnapshotError("reviewed candidates exceed the candidate population")
    candidate_papers_reviewed = papers_reviewed - absence_reviewed
    if candidate_papers_reviewed > candidate_items:
        raise PublicSnapshotError("candidate-paper coverage exceeds reviewed candidates")
    if candidate_papers_reviewed > papers - papers_without_candidates:
        raise PublicSnapshotError("candidate-paper coverage exceeds papers with candidates")

    decisions_raw = _as_mapping(raw.get("decisions"), "human review decisions")
    completed = _nonnegative_int(decisions_raw.get("completed"), "human review decisions.completed")
    if completed != reviewed:
        raise PublicSnapshotError("human review is not fully decided")
    outcome_counts = _project_named_counts(
        decisions_raw.get("outcome_counts"),
        _REVIEW_OUTCOMES,
        "human review decisions.outcome_counts",
        require_all=True,
    )
    if sum(outcome_counts.values()) != completed:
        raise PublicSnapshotError("human review outcome counts do not equal completed items")
    issue_counts = _project_named_counts(
        decisions_raw.get("issue_counts"),
        _REVIEW_ISSUES,
        "human review decisions.issue_counts",
        require_all=True,
    )
    if any(count > completed for count in issue_counts.values()):
        raise PublicSnapshotError("human review issue count exceeds completed items")

    privacy_raw = _as_mapping(raw.get("privacy"), "human review privacy")
    privacy_fields = (
        "contains_evidence_quotes",
        "contains_candidate_payloads",
        "contains_provider_raw_data",
        "contains_local_paths",
        "contains_reviewer_notes",
    )
    privacy: dict[str, bool] = {}
    for name in privacy_fields:
        value = privacy_raw.get(name)
        if value is not False:
            raise PublicSnapshotError(f"human review privacy flag {name} must be false")
        privacy[name] = False

    return {
        "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
        "source_artifact_sha256": sha256_file(path),
        "audit_id": audit_id,
        "sampling_policy": sampling_policy,
        "population": {
            "candidates": candidates,
            "papers": papers,
            "papers_without_candidates": papers_without_candidates,
        },
        "sample": {
            "requested": requested,
            "reviewed": reviewed,
            "papers_reviewed": papers_reviewed,
            "paper_coverage": paper_coverage,
            "risk_score_min": risk_score_min,
            "risk_score_max": risk_score_max,
            "risk_score_mean": risk_score_mean,
            "risk_reason_counts": risk_reason_counts,
            "item_type_counts": item_type_counts,
            "papers_without_candidates_reviewed": absence_reviewed,
        },
        "decisions": {
            "completed": completed,
            "outcome_counts": outcome_counts,
            "issue_counts": issue_counts,
        },
        "privacy": privacy,
    }


def _project_code(value: Any) -> dict[str, Any]:
    return _fields(
        value or {},
        ("git_commit", "git_dirty", "git_available", "source_tree_sha256"),
        "code",
    )


def _project_request_contract(value: Any) -> dict[str, Any]:
    source = _as_mapping(value or {}, "request_contract")
    return {
        "schema_version": _safe_scalar(
            source.get("schema_version"), "request_contract.schema_version"
        ),
        "privacy": _fields(
            source.get("privacy", {}), ("data_collection", "zdr"), "request_contract.privacy"
        ),
        "routing": _fields(
            source.get("routing", {}),
            ("require_parameters",),
            "request_contract.routing",
        ),
        "schema": _fields(
            source.get("schema", {}),
            ("response_format", "schema_name", "schema_sha256", "schema_strict"),
            "request_contract.schema",
        ),
        "seed": _safe_scalar(source.get("seed"), "request_contract.seed"),
    }


def _project_gate(value: Any, context: str) -> dict[str, Any]:
    return _fields(value or {}, ("status", "value", "threshold", "direction"), context)


def _project_claim_type_classification(value: Any) -> dict[str, Any]:
    source = _as_mapping(value or {}, "claim_type_classification")
    per_class = _as_mapping(source.get("per_class", {}), "claim_type_classification.per_class")
    return {
        **_fields(
            source,
            ("basis", "basis_scope", "supported_classes", "accuracy", "macro_f1"),
            "claim_type_classification",
        ),
        "per_class": {
            label: _fields(
                per_class.get(label, {}),
                (
                    "support",
                    "predicted",
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "precision",
                    "recall",
                    "f1",
                ),
                f"claim_type_classification.per_class.{label}",
            )
            for label in _CLAIM_TYPES
            if label in per_class
        },
    }


def _project_detection(value: Any, context: str) -> dict[str, Any]:
    return _fields(
        value or {},
        (
            "true_positives",
            "precision_true_positives",
            "precision_basis",
            "recall_basis",
            "false_positives",
            "false_negatives",
            "precision",
            "precision_defined",
            "recall",
            "f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "precision_defined_papers",
        ),
        context,
    )


def _project_negative_safety(value: Any) -> dict[str, Any]:
    return _fields(
        value or {},
        (
            "controls_total",
            "controls_matched",
            "matched_control_count",
            "control_match_coverage",
            "control_match_coverage_defined",
            "measurement_status",
            "matched_candidates",
            "matched_candidate_count",
            "false_primary_count",
            "false_primary_export_count",
            "false_primary_rate",
            "zero_false_primary_gate_passed",
            "zero_false_primary_export_gate_passed",
        ),
        "negative_control_safety",
    )


def _project_quality(value: Any) -> dict[str, Any]:
    source = _as_mapping(value or {}, "quality")
    macro = _as_mapping(source.get("macro", {}), "quality.macro")
    micro = _as_mapping(source.get("micro", {}), "quality.micro")
    macro_detection = _as_mapping(macro.get("detection", {}), "quality.macro.detection")
    macro_detection_public = _fields(
        macro_detection,
        ("precision", "recall", "f1"),
        "quality.macro.detection",
    )
    for basis_name in ("defined_cases", "undefined_cases"):
        if basis_name in macro_detection:
            macro_detection_public[basis_name] = _fields(
                macro_detection[basis_name],
                ("precision", "recall", "f1"),
                f"quality.macro.detection.{basis_name}",
            )
    return {
        **_fields(source, ("scored_cases",), "quality"),
        "macro": {
            "detection": macro_detection_public,
            "field_accuracy": _fields(
                macro.get("field_accuracy", {}), _QUALITY_FIELDS, "quality.macro.field_accuracy"
            ),
        },
        "micro": {
            **_fields(micro, ("reference_observations",), "quality.micro"),
            "detection": _project_detection(micro.get("detection", {}), "quality.micro.detection"),
            "field_accuracy": _fields(
                micro.get("field_accuracy", {}), _QUALITY_FIELDS, "quality.micro.field_accuracy"
            ),
        },
    }


def _project_usage(value: Any) -> dict[str, Any]:
    source = _as_mapping(value or {}, "usage")
    result: dict[str, Any] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens", "cost_usd"):
        if name in source:
            result[name] = _fields(
                source[name], ("total", "reported_calls", "missing_calls"), f"usage.{name}"
            )
    if "latency_seconds" in source:
        result["latency_seconds"] = _fields(
            source["latency_seconds"],
            ("total", "mean", "p50", "p95", "max", "reported_calls", "missing_calls"),
            "usage.latency_seconds",
        )
    return result


def _project_model_aggregate(value: Any) -> dict[str, Any]:
    source = _as_mapping(value or {}, "model.aggregate")
    evidence = _as_mapping(source.get("evidence", {}), "model.aggregate.evidence")
    selection_gates = _as_mapping(
        source.get("model_selection_gates", {}), "model.aggregate.model_selection_gates"
    )
    return {
        "execution": _fields(
            source.get("execution", {}),
            (
                "cases_attempted",
                "cases_succeeded",
                "cases_partial_failure",
                "cases_failed",
                "case_success_rate",
                "cases_scored",
                "case_scored_rate",
                "calls_attempted",
                "calls_succeeded",
                "calls_failed",
                "call_success_rate",
            ),
            "model.aggregate.execution",
        ),
        "schema": _fields(
            source.get("schema", {}),
            (
                "structured_responses_valid",
                "structured_responses_invalid",
                "structured_response_not_observed",
                "valid_rate_of_observed_responses",
                "end_to_end_schema_success_rate",
            ),
            "model.aggregate.schema",
        ),
        "quality": _project_quality(source.get("quality", {})),
        "evidence": {
            "candidate_text_support": _fields(
                evidence.get("candidate_text_support", {}),
                ("supported", "partially_supported", "unsupported", "unverified"),
                "model.aggregate.evidence.candidate_text_support",
            ),
            "reference_evidence_supported_accuracy": _fields(
                evidence.get("reference_evidence_supported_accuracy", {}),
                ("macro", "micro"),
                "model.aggregate.evidence.reference_evidence_supported_accuracy",
            ),
            "reference_page_anchor_accuracy": _fields(
                evidence.get("reference_page_anchor_accuracy", {}),
                ("macro", "micro"),
                "model.aggregate.evidence.reference_page_anchor_accuracy",
            ),
        },
        "negative_control_safety": _project_negative_safety(
            source.get("negative_control_safety", {})
        ),
        "claim_type_classification": _project_claim_type_classification(
            source.get("claim_type_classification", {})
        ),
        "model_selection_gates": {
            name: _project_gate(selection_gates.get(name, {}), f"model_selection_gates.{name}")
            for name in (
                "claim_type_macro_f1",
                "false_primary_controls",
                "false_primary_exports",
            )
            if name in selection_gates
        },
        "usage": _project_usage(source.get("usage", {})),
    }


def _project_model_selection(path: Path, selected_model: str | None) -> dict[str, Any]:
    raw = _as_mapping(read_json(path), "model selection")
    models = []
    model_ids: list[str] = []
    for index, item in enumerate(_as_sequence(raw.get("models", []), "model selection.models")):
        model = _as_mapping(item, f"model selection.models[{index}]")
        model_id = str(model.get("model", ""))
        if not model_id:
            raise PublicSnapshotError("every public model result requires a model ID")
        model_ids.append(model_id)
        models.append(
            {
                "model": model_id,
                "label": str(model.get("label", model_id)),
                "aggregate": _project_model_aggregate(model.get("aggregate", {})),
            }
        )
    if selected_model is not None and selected_model not in model_ids:
        raise PublicSnapshotError("selected model is absent from the model-selection artifact")
    determinism = _as_mapping(raw.get("determinism", {}), "model selection.determinism")
    segmentation = determinism.get("segmentation", {})
    return {
        "schema_version": MODEL_SELECTION_SCHEMA_VERSION,
        "source_artifact_sha256": sha256_file(path),
        "bakeoff_id": _safe_scalar(raw.get("bakeoff_id"), "model selection.bakeoff_id"),
        "configuration_sha256": _safe_scalar(
            raw.get("configuration_sha256"), "model selection.configuration_sha256"
        ),
        "code": _project_code(raw.get("code", {})),
        "determinism": {
            **_fields(
                determinism,
                (
                    "seed",
                    "temperature",
                    "reasoning_effort",
                    "max_tokens",
                    "min_confidence",
                    "prompt_sha256",
                    "reference_prompt_isolation",
                ),
                "model selection.determinism",
            ),
            "segmentation": _fields(
                segmentation, _SEGMENTATION_FIELDS, "model selection.determinism.segmentation"
            ),
        },
        "request_contract": _project_request_contract(raw.get("request_contract", {})),
        "selection": {
            "status": "selected" if selected_model is not None else "pending",
            "selected_model": selected_model,
        },
        "models": models,
    }


def _project_reference_evaluation(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    source = _as_mapping(value, "reference_evaluation")
    gates = _as_mapping(source.get("quality_gates", {}), "reference_evaluation.quality_gates")
    result = {
        **_fields(
            source,
            ("schema_version", "papers_scored", "coverage_statement"),
            "reference_evaluation",
        ),
        "bases": _fields(
            source.get("bases", {}),
            (
                "reference_observations",
                "field_matching",
                "precision_candidates_in_fully_annotated_regions",
            ),
            "reference_evaluation.bases",
        ),
        "detection": _project_detection(
            source.get("detection", {}), "reference_evaluation.detection"
        ),
        "field_accuracy": _fields(
            source.get("field_accuracy", {}),
            _QUALITY_FIELDS,
            "reference_evaluation.field_accuracy",
        ),
        "derived_accuracy": _fields(
            source.get("derived_accuracy", {}),
            ("exact_numeric_value_and_unit", "evidence_page_and_text_support"),
            "reference_evaluation.derived_accuracy",
        ),
        "negative_control_safety": _project_negative_safety(
            source.get("negative_control_safety", {})
        ),
        "claim_type_classification": _project_claim_type_classification(
            source.get("claim_type_classification", {})
        ),
        "quality_gates": {
            name: _project_gate(gates.get(name, {}), f"reference_evaluation.quality_gates.{name}")
            for name in (
                "candidate_detection_recall",
                "exact_numeric_value_and_unit",
                "joint_system_dataset_metric_value_slice",
                "evidence_page_and_text_support",
                "evidence_table_figure_row_column",
                "honest_missingness",
                "claim_type_macro_f1",
                "false_primary_controls",
                "false_primary_exports",
            )
            if name in gates
        },
    }
    return result


def _aggregate_private_calls(stage: Mapping[str, Any]) -> dict[str, Any]:
    calls = stage.get("calls", [])
    if not isinstance(calls, Sequence) or isinstance(calls, str | bytes | bytearray):
        calls = []
    costs: list[float] = []
    input_tokens = 0
    output_tokens = 0
    returned_models: set[str] = set()
    returned_providers: set[str] = set()
    for item in calls:
        if not isinstance(item, Mapping):
            continue
        cost = item.get("cost_usd")
        if isinstance(cost, int | float) and not isinstance(cost, bool) and math.isfinite(cost):
            costs.append(float(cost))
        for key, target in (("input_tokens", "input"), ("output_tokens", "output")):
            token_count = item.get(key)
            if isinstance(token_count, int) and not isinstance(token_count, bool):
                if target == "input":
                    input_tokens += token_count
                else:
                    output_tokens += token_count
        if item.get("model_returned"):
            returned_models.add(str(item["model_returned"]))
        if item.get("provider_returned"):
            returned_providers.add(str(item["provider_returned"]))
    return {
        "calls_attempted": len(calls),
        "cost_usd": round(sum(costs), 8),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "models_returned": sorted(returned_models),
        "providers_returned": sorted(returned_providers),
    }


def _project_stage(value: Any, *, verifier: bool = False) -> dict[str, Any]:
    source = _as_mapping(value or {}, "verifier" if verifier else "extractor")
    allowed = (
        ("enabled", "model", "max_tokens", "seed")
        if verifier
        else (
            "provider",
            "model",
            "temperature",
            "reasoning_effort",
            "max_tokens",
            "seed",
            "prompt_sha256",
        )
    )
    result = _fields(source, allowed, "verifier" if verifier else "extractor")
    if "request_contract" in source:
        result["request_contract"] = _project_request_contract(source["request_contract"])
    result["usage"] = _aggregate_private_calls(source)
    return result


def _project_paper_run(value: Any, index: int) -> dict[str, Any]:
    source = _as_mapping(value, f"corpus.runs[{index}]")
    paper_id = str(source.get("paper_id", ""))
    if not _SAFE_ID.fullmatch(paper_id):
        raise PublicSnapshotError(f"unsafe paper ID in corpus run: {paper_id!r}")
    counts = _fields(source.get("counts", {}), _COUNT_FIELDS, f"corpus.runs[{index}].counts")
    result: dict[str, Any] = {
        "paper_id": paper_id,
        "title": str(source.get("title", paper_id)),
        "status": str(source.get("status", "unknown")),
        "counts": counts,
        "wall_clock_seconds": _safe_scalar(
            source.get("wall_clock_seconds", 0.0),
            f"corpus.runs[{index}].wall_clock_seconds",
        ),
    }
    if "source_manifest_sha256" in source:
        result["source_manifest_sha256"] = _safe_scalar(
            source["source_manifest_sha256"], f"corpus.runs[{index}].source_manifest_sha256"
        )
    if "selected_pages" in source:
        pages = _as_sequence(source["selected_pages"], f"corpus.runs[{index}].selected_pages")
        result["selected_pages"] = [
            int(page) for page in pages if isinstance(page, int) and not isinstance(page, bool)
        ]
    if "result_block_segmentation" in source:
        result["result_block_segmentation"] = _fields(
            source["result_block_segmentation"],
            _SEGMENTATION_FIELDS,
            f"corpus.runs[{index}].result_block_segmentation",
        )
    result["layout"] = _fields(
        source,
        ("layout_parser", "layout_parser_version"),
        f"corpus.runs[{index}].layout",
    )
    result["extractor"] = _project_stage(source.get("extractor", {}))
    result["verifier"] = _project_stage(source.get("verifier", {}), verifier=True)
    result["eee_schema"] = _fields(
        source.get("eee_schema", {}), ("version", "sha256"), f"corpus.runs[{index}].eee_schema"
    )
    result["code"] = _project_code(source.get("code", {}))
    reference = _project_reference_evaluation(source.get("reference_evaluation"))
    if reference is not None:
        result["reference_evaluation"] = reference
    if source.get("error") and isinstance(source["error"], Mapping):
        result["error"] = _fields(source["error"], ("type", "code"), f"corpus.runs[{index}].error")
    return result


def _project_corpus(root: Path) -> dict[str, Any]:
    run_path = root / "corpus-run.json"
    if not run_path.is_file() or run_path.is_symlink():
        raise PublicSnapshotError(f"missing regular corpus-run.json under {root}")
    raw = _as_mapping(read_json(run_path), f"{root.name}.corpus-run")
    runs = [
        _project_paper_run(item, index)
        for index, item in enumerate(_as_sequence(raw.get("runs", []), "corpus.runs"))
    ]
    evaluation_value = raw.get("reference_evaluation")
    evaluation_path = root / "corpus-evaluation.json"
    evaluation_hash: str | None = None
    if evaluation_path.is_file() and not evaluation_path.is_symlink():
        evaluation_hash = sha256_file(evaluation_path)
        if evaluation_value is None:
            evaluation_value = read_json(evaluation_path)
    result = {
        **_fields(
            raw,
            (
                "schema_version",
                "corpus_id",
                "status",
                "generated_at",
                "papers",
                "papers_succeeded",
                "papers_failed",
                "papers_with_eee",
            ),
            "corpus",
        ),
        "source_artifacts": {
            "corpus_run_sha256": sha256_file(run_path),
            "corpus_evaluation_sha256": evaluation_hash,
        },
        "totals": _fields(raw.get("totals", {}), _COUNT_FIELDS, "corpus.totals"),
        "reference_evaluation": _project_reference_evaluation(evaluation_value),
        "papers_detail": runs,
    }
    return result


def _public_report_input(corpus: Mapping[str, Any]) -> dict[str, Any]:
    report_runs = []
    for paper in _as_sequence(corpus.get("papers_detail", []), "public corpus papers"):
        item = _as_mapping(paper, "public corpus paper")
        extractor = _as_mapping(item.get("extractor", {}), "public extractor")
        verifier = _as_mapping(item.get("verifier", {}), "public verifier")
        extractor_usage = _as_mapping(extractor.get("usage", {}), "public extractor usage")
        verifier_usage = _as_mapping(verifier.get("usage", {}), "public verifier usage")
        report_runs.append(
            {
                "paper_id": item["paper_id"],
                "title": item["title"],
                "status": item["status"],
                "counts": item["counts"],
                "cost_usd": float(extractor_usage.get("cost_usd", 0.0))
                + float(verifier_usage.get("cost_usd", 0.0)),
                "wall_clock_seconds": item["wall_clock_seconds"],
            }
        )
    return {
        "corpus_id": corpus.get("corpus_id"),
        "generated_at": corpus.get("generated_at"),
        "runs": report_runs,
        "reference_evaluation": corpus.get("reference_evaluation"),
    }


def _validate_public_uri(value: str | None, context: str) -> str | None:
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PublicSnapshotError(f"{context} is not a public HTTP(S) URI")
    return value


def _project_sources(run_roots: Sequence[Path]) -> tuple[dict[str, Any], set[str]]:
    papers: list[dict[str, Any]] = []
    paper_ids: set[str] = set()
    for root in run_roots:
        for manifest_path in sorted(root.glob("*/source-manifest.json")):
            if manifest_path.is_symlink():
                raise PublicSnapshotError(f"source manifest cannot be a symlink: {manifest_path}")
            manifest = SourceManifest.model_validate(read_json(manifest_path))
            if manifest.paper_id in paper_ids:
                raise PublicSnapshotError(
                    f"duplicate paper ID across run roots: {manifest.paper_id}"
                )
            if not _SAFE_ID.fullmatch(manifest.paper_id):
                raise PublicSnapshotError(f"unsafe paper ID: {manifest.paper_id!r}")
            paper_ids.add(manifest.paper_id)
            paper_sources = []
            for source in manifest.sources:
                paper_sources.append(
                    {
                        "source_id": source.source_id,
                        "role": str(source.role),
                        "original_uri": _validate_public_uri(
                            source.original_uri,
                            f"{manifest.paper_id}.{source.source_id}.original_uri",
                        ),
                        "resolved_uri": _validate_public_uri(
                            source.resolved_uri,
                            f"{manifest.paper_id}.{source.source_id}.resolved_uri",
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
                    "paper_id": manifest.paper_id,
                    "title": manifest.title,
                    "doi": manifest.doi,
                    "arxiv_id": manifest.arxiv_id,
                    "proceedings_url": (
                        str(manifest.proceedings_url) if manifest.proceedings_url else None
                    ),
                    "source_manifest_sha256": sha256_file(manifest_path),
                    "sources": paper_sources,
                }
            )
    return {"schema_version": SOURCES_SCHEMA_VERSION, "papers": papers}, paper_ids


def _project_reference_audits(run_roots: Sequence[Path]) -> dict[str, Any]:
    corpora = []
    for root in run_roots:
        path = root / "reference-audit.json"
        if not path.is_file() or path.is_symlink():
            raise PublicSnapshotError(f"missing regular reference-audit.json under {root}")
        raw = _as_mapping(read_json(path), f"{root.name}.reference-audit")
        results = []
        for index, item in enumerate(
            _as_sequence(raw.get("results", []), f"{root.name}.reference-audit.results")
        ):
            results.append(
                _fields(
                    item,
                    (
                        "paper_id",
                        "status",
                        "page_count",
                        "source_hash_matches",
                        "text_verified",
                        "visual_verified",
                        "failed_evidence",
                    ),
                    f"{root.name}.reference-audit.results[{index}]",
                )
            )
        corpora.append(
            {
                **_fields(
                    raw,
                    (
                        "schema_version",
                        "corpus_id",
                        "status",
                        "papers",
                        "papers_passed",
                        "papers_failed",
                        "papers_skipped",
                        "text_verified",
                        "visual_verified",
                        "failed_evidence",
                    ),
                    f"{root.name}.reference-audit",
                ),
                "source_artifact_sha256": sha256_file(path),
                "results": results,
            }
        )
    return {"schema_version": REFERENCE_AUDIT_SCHEMA_VERSION, "corpora": corpora}


def _manifest_evidence_sources(
    root: Path,
    paper_id: str,
) -> dict[str, tuple[str, str]]:
    manifest_path = root / paper_id / "source-manifest.json"
    manifest = _as_mapping(read_json(manifest_path), f"{paper_id}.source-manifest")
    if manifest.get("paper_id") != paper_id:
        raise PublicSnapshotError(f"source manifest paper_id mismatch for {paper_id}")
    sources: dict[str, tuple[str, str]] = {}
    for index, value in enumerate(
        _as_sequence(manifest.get("sources"), f"{paper_id}.source-manifest.sources")
    ):
        source = _as_mapping(value, f"{paper_id}.source-manifest.sources[{index}]")
        source_id = source.get("source_id")
        source_sha256 = source.get("sha256")
        source_role = source.get("role")
        if not isinstance(source_id, str) or not source_id:
            raise PublicSnapshotError(f"source manifest has invalid source_id for {paper_id}")
        if source_id in sources:
            raise PublicSnapshotError(f"source manifest has duplicate source_id for {paper_id}")
        if not isinstance(source_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha256):
            continue
        if not isinstance(source_role, str) or not source_role:
            raise PublicSnapshotError(f"source manifest has invalid source role for {paper_id}")
        sources[source_id] = (source_sha256, source_role)
    return sources


def _validate_eee_provenance(
    record: Mapping[str, Any],
    *,
    paper_id: str,
    sources: Mapping[str, tuple[str, str]],
    context: str,
) -> None:
    source_metadata = _as_mapping(record.get("source_metadata"), f"{context}.source_metadata")
    source_details = _as_mapping(
        source_metadata.get("additional_details"),
        f"{context}.source_metadata.additional_details",
    )
    if source_details.get("paper_id") != paper_id:
        raise PublicSnapshotError(f"EEE provenance paper_id mismatch in {context}")
    results = _as_sequence(record.get("evaluation_results"), f"{context}.evaluation_results")
    for result_index, value in enumerate(results):
        result_context = f"{context}.evaluation_results[{result_index}]"
        result = _as_mapping(value, result_context)
        score_details = _as_mapping(result.get("score_details"), f"{result_context}.score_details")
        details = _as_mapping(score_details.get("details"), f"{result_context}.details")
        if details.get("paper_id") != paper_id:
            raise PublicSnapshotError(
                f"EEE result lacks paper-bound provenance in {result_context}"
            )
        anchor_count_raw = details.get("evidence_anchor_count")
        if not isinstance(anchor_count_raw, str) or not re.fullmatch(
            r"[1-9][0-9]*", anchor_count_raw
        ):
            raise PublicSnapshotError(f"EEE result lacks evidence anchors in {result_context}")
        allowed_quote_keys = {
            f"evidence_{anchor_index}_quote_sha256"
            for anchor_index in range(1, int(anchor_count_raw) + 1)
        }
        if any("quote" in str(key).casefold() and key not in allowed_quote_keys for key in details):
            raise PublicSnapshotError(f"EEE result embeds an evidence quote in {result_context}")
        for anchor_index in range(1, int(anchor_count_raw) + 1):
            prefix = f"evidence_{anchor_index}"
            source_id = details.get(f"{prefix}_source_id")
            source_sha256 = details.get(f"{prefix}_source_sha256")
            source_role = details.get(f"{prefix}_source_role")
            expected_source = sources.get(source_id) if isinstance(source_id, str) else None
            if expected_source is None or (source_sha256, source_role) != expected_source:
                raise PublicSnapshotError(
                    f"EEE evidence source provenance is unbound in {result_context}"
                )
            page = details.get(f"{prefix}_page")
            if not isinstance(page, str) or not re.fullmatch(r"[1-9][0-9]*", page):
                raise PublicSnapshotError(f"EEE evidence page is invalid in {result_context}")
            kind = details.get(f"{prefix}_kind")
            if kind not in {"table", "figure", "prose", "appendix"}:
                raise PublicSnapshotError(f"EEE evidence kind is invalid in {result_context}")
            quote_sha256 = details.get(f"{prefix}_quote_sha256")
            if not isinstance(quote_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", quote_sha256):
                raise PublicSnapshotError(f"EEE quote hash is invalid in {result_context}")
            if kind in {"table", "figure"} and not any(
                details.get(f"{prefix}_{name}") for name in ("label", "row", "column")
            ):
                raise PublicSnapshotError(
                    f"EEE structured evidence lacks a structural anchor in {result_context}"
                )


def _copy_valid_eee(
    *,
    run_roots: Sequence[Path],
    paper_ids: set[str],
    destination: Path,
    schema: dict[str, Any],
) -> list[str]:
    copied: list[str] = []
    for root in run_roots:
        for paper_id in sorted(paper_ids):
            eee_dir = root / paper_id / "eee"
            if not eee_dir.exists():
                continue
            sources = _manifest_evidence_sources(root, paper_id)
            if not eee_dir.is_dir() or eee_dir.is_symlink():
                raise PublicSnapshotError(f"EEE input must be a regular directory: {eee_dir}")
            for source_path in sorted(eee_dir.glob("*.json")):
                if source_path.is_symlink() or not source_path.is_file():
                    raise PublicSnapshotError(f"EEE input must be a regular file: {source_path}")
                record = _as_mapping(read_json(source_path), f"EEE record {source_path.name}")
                issues = validate_eee_record(dict(record), schema)
                if issues:
                    details = "; ".join(f"{issue.path}: {issue.message}" for issue in issues[:3])
                    raise PublicSnapshotError(
                        f"EEE validation failed for {source_path.name}: {details}"
                    )
                _validate_eee_provenance(
                    record,
                    paper_id=paper_id,
                    sources=sources,
                    context=f"EEE record {source_path.name}",
                )
                target = destination / "eee" / paper_id / source_path.name
                if target.exists():
                    raise PublicSnapshotError(f"duplicate public EEE output: {target.name}")
                write_json(target, dict(record))
                copied.append(target.relative_to(destination).as_posix())
    return copied


def _scan_text(text: str, context: str) -> None:
    if _LOCAL_PATH.search(text):
        raise PublicSnapshotError(f"absolute local path found in {context}")
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        raise PublicSnapshotError(f"credential-like value found in {context}")


def _audit_json_value(value: Any, context: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_PUBLIC_KEYS:
                raise PublicSnapshotError(f"forbidden public key at {context}.{key}")
            _audit_json_value(child, f"{context}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            _audit_json_value(child, f"{context}[{index}]")
        return
    if isinstance(value, str):
        _scan_text(value, context)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise PublicSnapshotError(f"non-finite number found at {context}")


def _audit_snapshot_tree(root: Path, *, expect_checksums: bool) -> None:
    expected = _TOP_LEVEL_FILES if expect_checksums else _TOP_LEVEL_FILES - {"SHA256SUMS"}
    top_files = {path.name for path in root.iterdir() if path.is_file()}
    if top_files != expected:
        raise PublicSnapshotError(
            f"public snapshot top-level files differ: expected={sorted(expected)}, "
            f"actual={sorted(top_files)}"
        )
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PublicSnapshotError(f"public snapshot cannot contain symlinks: {path.name}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] == "eee":
            if len(relative.parts) != 3 or path.suffix != ".json":
                raise PublicSnapshotError(f"unexpected EEE snapshot path: {relative}")
        elif relative.name not in expected:
            raise PublicSnapshotError(f"unexpected public snapshot file: {relative}")
        content = path.read_bytes()
        if len(content) > 2_000_000:
            raise PublicSnapshotError(f"public snapshot file is unexpectedly large: {relative}")
        if content.startswith(b"%PDF-") or content.startswith(b"PK\x03\x04") or b"\x00" in content:
            raise PublicSnapshotError(f"binary/source payload found in public snapshot: {relative}")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PublicSnapshotError(f"non-UTF-8 public file: {relative}") from error
        _scan_text(text, relative.as_posix())
        if path.suffix == ".json":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as error:
                raise PublicSnapshotError(f"invalid public JSON: {relative}") from error
            _audit_json_value(parsed)


def _write_checksums(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    atomic_write_bytes(root / "SHA256SUMS", ("\n".join(lines) + "\n").encode("utf-8"))


def _readme(snapshot_id: str, corpora: Sequence[Mapping[str, Any]], eee_files: int) -> str:
    papers = sum(int(corpus.get("papers", 0) or 0) for corpus in corpora)
    return f"""# Public pilot snapshot: {snapshot_id}

This immutable snapshot contains derived metadata from {papers} paper run(s) and
{eee_files} schema-valid Every Eval Ever record file(s). It does not contain source
PDFs, source-layout text, raw provider responses, request IDs, per-call traces,
credentials, or evidence quotations.

Each numeric EEE result retains quote-free source provenance in
`score_details.details`: paper/source ID and hash, page, structure kind, optional
label/row/column, and the evidence-quote hash.

## Contents

- `snapshot.json`: allowlisted run summaries, aggregate evaluation results, and limitations.
- `model-selection.json`: aggregate model comparison without cases, calls, or provider traces.
- `human-review.json`: fully decided aggregate review counts without candidates,
  quotes, paths, or notes.
- `sources.json`: public source URLs and immutable hashes without cache paths.
- `reference-audit.json`: source-hash and annotation-audit counts.
- `corpus-review.html`: self-contained aggregate review generated from sanitized data.
- `eee/`: outputs validated against the pinned EEE schema before publication.
- `SHA256SUMS`: deterministic checksums for every other file in this directory.

This is a development pilot and error-analysis snapshot, not a universal benchmark
or an Evaluation Card. Paper copyrights remain with their respective owners; this
directory republishes derived factual metadata, not source documents or text snapshots.
"""


def _validate_human_review_corpus_population(
    human_review: Mapping[str, Any],
    corpora: Sequence[Mapping[str, Any]],
) -> None:
    corpus_papers = sum(int(corpus.get("papers", 0) or 0) for corpus in corpora)
    corpus_candidates = sum(
        int(_as_mapping(corpus.get("totals", {}), "corpus totals").get("candidates", 0) or 0)
        for corpus in corpora
    )
    corpus_papers_without_candidates = 0
    for corpus_index, corpus in enumerate(corpora):
        for paper_index, value in enumerate(
            _as_sequence(
                corpus.get("papers_detail", []),
                f"corpora[{corpus_index}].papers_detail",
            )
        ):
            paper = _as_mapping(value, f"corpora[{corpus_index}].papers_detail[{paper_index}]")
            counts = _as_mapping(
                paper.get("counts", {}),
                f"corpora[{corpus_index}].papers_detail[{paper_index}].counts",
            )
            if int(counts.get("candidates", 0) or 0) == 0:
                corpus_papers_without_candidates += 1
    population = _as_mapping(human_review.get("population"), "human review population")
    if population.get("papers") != corpus_papers:
        raise PublicSnapshotError("human review paper population does not match corpus runs")
    if population.get("candidates") != corpus_candidates:
        raise PublicSnapshotError("human review candidate population does not match corpus runs")
    if population.get("papers_without_candidates") != corpus_papers_without_candidates:
        raise PublicSnapshotError(
            "human review zero-candidate paper population does not match corpus runs"
        )


def build_public_snapshot(
    *,
    snapshot_id: str,
    corpus_run_root: Path,
    model_selection_path: Path,
    human_review_summary_path: Path,
    schema_path: Path,
    schema_sha256: str,
    output_root: Path,
    additional_run_roots: Sequence[Path] = (),
    selected_model: str | None = None,
) -> Path:
    """Build and atomically publish one deterministic public pilot snapshot."""

    if not _SAFE_ID.fullmatch(snapshot_id):
        raise PublicSnapshotError("snapshot_id must be a safe lowercase path component")
    run_roots = [corpus_run_root.resolve(), *(path.resolve() for path in additional_run_roots)]
    if any(not root.is_dir() for root in run_roots):
        raise PublicSnapshotError("every run root must be an existing directory")
    if not model_selection_path.is_file() or model_selection_path.is_symlink():
        raise PublicSnapshotError("model selection input must be a regular file")
    if not human_review_summary_path.is_file() or human_review_summary_path.is_symlink():
        raise PublicSnapshotError("human review summary input must be a regular file")
    human_review = _project_human_review_summary(human_review_summary_path)
    schema, authority = load_schema(schema_path, schema_sha256)
    destination_root = output_root.resolve()
    destination = destination_root / snapshot_id
    if destination.exists():
        raise FileExistsError(f"public snapshot already exists: {destination}")
    destination_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=destination_root))
    try:
        corpora = [_project_corpus(root) for root in run_roots]
        _validate_human_review_corpus_population(human_review, corpora)
        model_selection = _project_model_selection(model_selection_path, selected_model)
        sources, paper_ids = _project_sources(run_roots)
        audits = _project_reference_audits(run_roots)
        eee_files = _copy_valid_eee(
            run_roots=run_roots,
            paper_ids=paper_ids,
            destination=temporary,
            schema=schema,
        )
        snapshot = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "eee_schema": {"version": authority.version, "sha256": authority.sha256},
            "model_selection": {
                "path": "model-selection.json",
                "selected_model": selected_model,
                "source_artifact_sha256": model_selection["source_artifact_sha256"],
            },
            "human_review": {
                "path": "human-review.json",
                "audit_id": human_review["audit_id"],
                "source_artifact_sha256": human_review["source_artifact_sha256"],
                "paper_coverage": human_review["sample"]["paper_coverage"],
            },
            "corpora": corpora,
            "eee": {
                "files": len(eee_files),
                "papers_with_eee": len({Path(path).parts[1] for path in eee_files}),
            },
            "limitations": [
                "Development and error-analysis corpus; not a sealed generalization benchmark.",
                "Reference recall applies only to explicitly annotated observations.",
                "Precision applies only to explicitly fully annotated regions.",
                "Evaluation Cards are downstream and are not an output of this repository.",
            ],
            "sanitization": {
                "policy": "field-allowlist/0.1",
                "source_quotes_included": False,
                "raw_provider_traces_included": False,
                "absolute_local_paths_included": False,
            },
        }
        write_json(temporary / "snapshot.json", snapshot)
        write_json(temporary / "model-selection.json", model_selection)
        write_json(temporary / "human-review.json", human_review)
        write_json(temporary / "sources.json", sources)
        write_json(temporary / "reference-audit.json", audits)
        atomic_write_bytes(
            temporary / "corpus-review.html",
            render_corpus_html(_public_report_input(corpora[0])).encode("utf-8"),
        )
        atomic_write_bytes(
            temporary / "README.md",
            _readme(snapshot_id, corpora, len(eee_files)).encode("utf-8"),
        )
        _audit_snapshot_tree(temporary, expect_checksums=False)
        _write_checksums(temporary)
        _audit_snapshot_tree(temporary, expect_checksums=True)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
