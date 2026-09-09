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

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.evaluation.corpus_score import aggregate_reference_scores
from proceedings_to_eee.evaluation.reference_score import score_claim_type_pairs
from proceedings_to_eee.io import (
    atomic_write_bytes,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from proceedings_to_eee.providers.openrouter import (
    PUBLIC_PROVIDER_LABELS,
    ProviderCall,
    completion_token_parameter_for_model,
    public_provider_call,
    structured_request_contract_from_call,
)
from proceedings_to_eee.reporting.corpus_html import render_corpus_html
from proceedings_to_eee.sources.manifest import SourceManifest
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record

SNAPSHOT_SCHEMA_VERSION = "public-pilot-snapshot/0.3"
MODEL_SELECTION_SCHEMA_VERSION = "public-model-selection/0.1"
SOURCES_SCHEMA_VERSION = "public-source-index/0.1"
REFERENCE_AUDIT_SCHEMA_VERSION = "public-reference-audit/0.1"
HUMAN_REVIEW_SCHEMA_VERSION = "human-review-summary/0.1"

_CURRENT_GATE_CONTRACTS: dict[str, tuple[float, str]] = {
    "candidate_detection_recall": (0.90, "at_least"),
    "exact_numeric_value_and_unit": (0.98, "at_least"),
    "joint_system_dataset_metric_value_slice": (0.95, "at_least"),
    "evidence_page_and_text_support": (0.95, "at_least"),
    "evidence_table_figure_row_column": (0.95, "at_least"),
    "honest_missingness": (0.95, "at_least"),
    "claim_type_macro_f1": (0.90, "at_least"),
    "false_primary_controls": (0.0, "at_most"),
    "false_primary_exports": (0.0, "at_most"),
}
_CORPUS_REFERENCE_COVERAGE_STATEMENT = (
    "Recall and field accuracy cover annotated reference observations only. "
    "Precision covers only explicitly fully annotated labels. Sampled and "
    "excluded regions remain outside the precision basis and do not establish "
    "whole-paper gold."
)

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
_PAPER_RUN_STATUSES = frozenset(
    {"success", "partial_failure", "quality_failure", "bounded_incomplete", "error"}
)
_CORPUS_RUN_STATUSES = frozenset({"success", "partial_failure", "bounded_incomplete", "error"})
_CORPUS_PARTITION_FIELDS = (
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


def _nullable_nonnegative_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, context)


def _nullable_fraction(value: Any, context: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, context)
    if not 0.0 <= result <= 1.0:
        raise PublicSnapshotError(f"{context} must be between 0 and 1")
    return result


def _nullable_bool(value: Any, context: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise PublicSnapshotError(f"{context} must be a boolean or null")


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


def _project_request_contract(
    value: Any, *, outer_stage: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    context = "request_contract"
    source = _as_mapping(value, context)
    schema_version = source.get("schema_version")
    if schema_version not in {
        "provider-request-contract/0.1",
        "provider-request-contract/0.2",
    }:
        raise PublicSnapshotError(f"{context}.schema_version is unsupported")
    privacy = _as_mapping(source.get("privacy"), f"{context}.privacy")
    if privacy.get("data_collection") not in {"allow", "deny"} or not isinstance(
        privacy.get("zdr"), bool
    ):
        raise PublicSnapshotError(f"{context}.privacy is incomplete or unsupported")
    routing = _as_mapping(source.get("routing"), f"{context}.routing")
    if not isinstance(routing.get("require_parameters"), bool):
        raise PublicSnapshotError(f"{context}.routing.require_parameters must be a boolean")
    schema = _as_mapping(source.get("schema"), f"{context}.schema")
    if (
        schema.get("response_format") != "json_schema"
        or not isinstance(schema.get("schema_name"), str)
        or not schema["schema_name"]
        or not isinstance(schema.get("schema_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", schema["schema_sha256"]) is None
        or schema.get("schema_strict") is not True
    ):
        raise PublicSnapshotError(f"{context}.schema is incomplete or unsupported")
    if "seed" not in source:
        raise PublicSnapshotError(f"{context}.seed is required")
    seed = source["seed"]
    if seed is not None:
        seed = _nonnegative_int(seed, f"{context}.seed")
    result = {
        "schema_version": schema_version,
        "privacy": {
            "data_collection": privacy["data_collection"],
            "zdr": privacy["zdr"],
        },
        "routing": {"require_parameters": routing["require_parameters"]},
        "schema": {
            "response_format": schema["response_format"],
            "schema_name": schema["schema_name"],
            "schema_sha256": schema["schema_sha256"],
            "schema_strict": schema["schema_strict"],
        },
        "seed": seed,
    }
    if schema_version == "provider-request-contract/0.2":
        if "max_tokens" not in source or "completion_token_parameter" not in source:
            raise PublicSnapshotError(
                "request_contract/0.2 requires max_tokens and completion_token_parameter"
            )
        result["max_tokens"] = _nonnegative_int(
            source["max_tokens"], "request_contract.max_tokens", positive=True
        )
        parameter = source["completion_token_parameter"]
        if parameter not in {"max_tokens", "max_completion_tokens"}:
            raise PublicSnapshotError("request_contract.completion_token_parameter is unsupported")
        result["completion_token_parameter"] = parameter
    elif "max_tokens" in source or "completion_token_parameter" in source:
        raise PublicSnapshotError("materialized token fields require provider-request-contract/0.2")
    if outer_stage is not None:
        for field in ("seed", "require_parameters"):
            outer_field = (
                outer_stage.get(field) if field == "seed" else outer_stage.get("require_parameters")
            )
            contract_field = (
                result["seed"] if field == "seed" else result["routing"]["require_parameters"]
            )
            if field not in outer_stage or outer_field != contract_field:
                raise PublicSnapshotError(
                    f"request_contract.{field} disagrees with outer stage settings"
                )
        if schema_version == "provider-request-contract/0.2":
            model = outer_stage.get("model")
            if not isinstance(model, str) or not model:
                raise PublicSnapshotError(
                    "materialized request contract requires an outer stage model"
                )
            if outer_stage.get("max_tokens") != result["max_tokens"]:
                raise PublicSnapshotError(
                    "request_contract.max_tokens disagrees with outer stage settings"
                )
            if result["completion_token_parameter"] != completion_token_parameter_for_model(model):
                raise PublicSnapshotError(
                    "request_contract.completion_token_parameter disagrees with outer stage model"
                )
    elif schema_version == "provider-request-contract/0.2":
        raise PublicSnapshotError("materialized request contract requires outer stage settings")
    return result


def _project_gate(value: Any, context: str) -> dict[str, Any]:
    source = _as_mapping(value or {}, context)
    if not source:
        return {}
    status = source.get("status")
    direction = source.get("direction")
    if status not in {"passed", "failed", "not_measured"}:
        raise PublicSnapshotError(f"{context}.status is unsupported")
    if direction not in {"at_least", "at_most"}:
        raise PublicSnapshotError(f"{context}.direction is unsupported")
    threshold = _finite_number(source.get("threshold"), f"{context}.threshold")
    if threshold < 0:
        raise PublicSnapshotError(f"{context}.threshold must be non-negative")
    raw_value = source.get("value")
    metric = None if raw_value is None else _finite_number(raw_value, f"{context}.value")
    if metric is not None and metric < 0:
        raise PublicSnapshotError(f"{context}.value must be non-negative")
    expected_status = (
        "not_measured"
        if metric is None
        else "passed"
        if (metric >= threshold if direction == "at_least" else metric <= threshold)
        else "failed"
    )
    if status != expected_status:
        raise PublicSnapshotError(f"{context}.status disagrees with value and threshold")
    return {
        "status": status,
        "value": metric,
        "threshold": threshold,
        "direction": direction,
    }


def _require_gate_bindings(
    gates: Mapping[str, Mapping[str, Any]],
    expected_values: Mapping[str, int | float | None],
    context: str,
) -> None:
    if set(gates) != set(expected_values):
        raise PublicSnapshotError(f"{context} is incomplete")
    for name, expected in expected_values.items():
        threshold, direction = _CURRENT_GATE_CONTRACTS[name]
        gate = gates[name]
        if gate["direction"] != direction or not math.isclose(
            gate["threshold"], threshold, rel_tol=0.0, abs_tol=1e-12
        ):
            raise PublicSnapshotError(f"{context}.{name} contract is unsupported")
        actual = gate["value"]
        if (actual is None) != (expected is None) or (
            actual is not None
            and not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
        ):
            raise PublicSnapshotError(f"{context}.{name}.value disagrees with metric")


def _project_claim_type_classification(value: Any) -> dict[str, Any]:
    source = _as_mapping(value or {}, "claim_type_classification")
    if not source:
        return {}
    per_class = _as_mapping(source.get("per_class", {}), "claim_type_classification.per_class")
    if set(per_class) != set(_CLAIM_TYPES):
        raise PublicSnapshotError("claim_type_classification.per_class is incomplete")
    basis = _nonnegative_int(source.get("basis"), "claim_type_classification.basis")
    supported = _nonnegative_int(
        source.get("supported_classes"), "claim_type_classification.supported_classes"
    )
    scope = source.get("basis_scope")
    if scope != "matched reference observations and matched negative controls":
        raise PublicSnapshotError("claim_type_classification.basis_scope is invalid")
    accuracy = _nullable_fraction(source.get("accuracy"), "claim_type_classification.accuracy")
    macro_f1 = _nullable_fraction(source.get("macro_f1"), "claim_type_classification.macro_f1")
    projected_classes: dict[str, dict[str, Any]] = {}
    for label in _CLAIM_TYPES:
        context = f"claim_type_classification.per_class.{label}"
        item = _as_mapping(per_class[label], context)
        counts = {
            field: _nonnegative_int(item.get(field), f"{context}.{field}")
            for field in (
                "support",
                "predicted",
                "true_positives",
                "false_positives",
                "false_negatives",
            )
        }
        if counts["support"] != counts["true_positives"] + counts["false_negatives"]:
            raise PublicSnapshotError(f"{context} support counts disagree")
        if counts["predicted"] != counts["true_positives"] + counts["false_positives"]:
            raise PublicSnapshotError(f"{context} predicted counts disagree")
        precision = _nullable_fraction(item.get("precision"), f"{context}.precision")
        recall = _nullable_fraction(item.get("recall"), f"{context}.recall")
        f1 = _nullable_fraction(item.get("f1"), f"{context}.f1")
        if not _fraction_agrees(
            precision, counts["true_positives"], counts["predicted"]
        ) or not _fraction_agrees(recall, counts["true_positives"], counts["support"]):
            raise PublicSnapshotError(f"{context} precision/recall is inconsistent")
        expected_f1 = (
            None
            if counts["support"] == 0
            else 0.0
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        if (expected_f1 is None) != (f1 is None) or (
            expected_f1 is not None
            and not math.isclose(float(f1), expected_f1, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise PublicSnapshotError(f"{context}.f1 is inconsistent")
        projected_classes[label] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    if (
        sum(item["support"] for item in projected_classes.values()) != basis
        or sum(item["predicted"] for item in projected_classes.values()) != basis
    ):
        raise PublicSnapshotError("claim_type_classification basis disagrees with classes")
    supported_items = [item for item in projected_classes.values() if item["support"] > 0]
    if supported != len(supported_items):
        raise PublicSnapshotError("claim_type_classification supported_classes is inconsistent")
    true_positives = sum(item["true_positives"] for item in projected_classes.values())
    if not _fraction_agrees(accuracy, true_positives, basis):
        raise PublicSnapshotError("claim_type_classification.accuracy is inconsistent")
    expected_macro = (
        sum(float(item["f1"]) for item in supported_items) / len(supported_items)
        if supported_items
        else None
    )
    if (expected_macro is None) != (macro_f1 is None) or (
        expected_macro is not None
        and not math.isclose(float(macro_f1), expected_macro, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise PublicSnapshotError("claim_type_classification.macro_f1 is inconsistent")
    return {
        "basis": basis,
        "basis_scope": scope,
        "supported_classes": supported,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": projected_classes,
    }


def _project_detection(value: Any, context: str) -> dict[str, Any]:
    source = _as_mapping(value or {}, context)
    result: dict[str, Any] = {}
    for name in (
        "true_positives",
        "precision_true_positives",
        "precision_basis",
        "recall_basis",
        "false_positives",
        "false_negatives",
        "precision_defined_papers",
    ):
        if name in source:
            result[name] = _nonnegative_int(source[name], f"{context}.{name}")
    for name in (
        "precision",
        "recall",
        "f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        if name in source:
            result[name] = _nullable_fraction(source[name], f"{context}.{name}")
    if "precision_defined" in source:
        precision_defined = source["precision_defined"]
        if not isinstance(precision_defined, bool):
            raise PublicSnapshotError(f"{context}.precision_defined must be a boolean")
        result["precision_defined"] = precision_defined
    return result


def _validate_detection_algebra(
    detection: Mapping[str, Any], context: str, *, require_complete: bool
) -> None:
    count_fields = (
        "true_positives",
        "precision_true_positives",
        "precision_basis",
        "recall_basis",
        "false_positives",
        "false_negatives",
    )
    rate_fields = ("precision", "recall", "f1", "precision_defined")
    if require_complete and any(field not in detection for field in (*count_fields, *rate_fields)):
        raise PublicSnapshotError(f"{context} is incomplete")
    if not all(field in detection for field in count_fields):
        return
    if detection["true_positives"] + detection["false_negatives"] != detection["recall_basis"]:
        raise PublicSnapshotError(f"{context} recall counts disagree")
    if (
        detection["precision_true_positives"] + detection["false_positives"]
        != detection["precision_basis"]
    ):
        raise PublicSnapshotError(f"{context} precision counts disagree")
    if detection.get("precision_defined") is not (detection["precision_basis"] > 0):
        raise PublicSnapshotError(f"{context}.precision_defined is inconsistent")
    if not _fraction_agrees(
        detection.get("precision"),
        detection["precision_true_positives"],
        detection["precision_basis"],
    ):
        raise PublicSnapshotError(f"{context}.precision is inconsistent")
    if not _fraction_agrees(
        detection.get("recall"), detection["true_positives"], detection["recall_basis"]
    ):
        raise PublicSnapshotError(f"{context}.recall is inconsistent")
    precision = detection.get("precision")
    recall = detection.get("recall")
    expected_f1 = (
        None
        if precision is None or recall is None
        else 0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    actual_f1 = detection.get("f1")
    if (expected_f1 is None) != (actual_f1 is None) or (
        expected_f1 is not None
        and not math.isclose(float(actual_f1), expected_f1, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise PublicSnapshotError(f"{context}.f1 is inconsistent")


def _project_negative_safety(value: Any) -> dict[str, Any]:
    context = "negative_control_safety"
    source = _as_mapping({} if value is None else value, context)
    result: dict[str, Any] = {}
    for name in (
        "controls_total",
        "controls_matched",
        "matched_control_count",
        "controls_examined",
        "controls_not_examined",
        "passed_by_abstention_count",
        "matched_candidates",
        "matched_candidate_count",
        "false_primary_count",
        "false_primary_export_count",
    ):
        if name in source:
            result[name] = _nonnegative_int(source[name], f"{context}.{name}")
    if "examined_control_ids" in source and "controls_examined" not in result:
        examined_ids = list(
            _as_sequence(source["examined_control_ids"], f"{context}.examined_control_ids")
        )
        if any(not isinstance(item, str) or not item for item in examined_ids) or len(
            set(examined_ids)
        ) != len(examined_ids):
            raise PublicSnapshotError(f"{context}.examined_control_ids is invalid")
        result["controls_examined"] = len(examined_ids)
        if "controls_total" in result and "controls_not_examined" not in result:
            if len(examined_ids) > result["controls_total"]:
                raise PublicSnapshotError(f"{context}.examined_control_ids exceeds controls_total")
            result["controls_not_examined"] = result["controls_total"] - len(examined_ids)
    for name in (
        "control_match_coverage",
        "control_examination_coverage",
        "false_primary_rate",
    ):
        if name in source:
            result[name] = _nullable_fraction(source[name], f"{context}.{name}")
    for name in (
        "control_match_coverage_defined",
        "zero_false_primary_gate_passed",
        "zero_false_primary_export_gate_passed",
    ):
        if name in source:
            result[name] = _nullable_bool(source[name], f"{context}.{name}")
    if "measurement_status" in source:
        status = source["measurement_status"]
        if not isinstance(status, str) or status not in {
            "measured",
            "partially_measured",
            "not_measured",
        }:
            raise PublicSnapshotError(f"{context}.measurement_status is unsupported")
        result["measurement_status"] = status
    if "control_trials" in source:
        trials = _as_mapping(source["control_trials"], f"{context}.control_trials")
        result["control_trials"] = {
            name: _nonnegative_int(
                trials[name],
                f"{context}.control_trials.{name}",
            )
            for name in ("total", "matched", "examined", "passed_by_abstention")
            if name in trials
        }
    _validate_negative_safety_algebra(result, context)
    return result


def _fraction_agrees(actual: float | None, numerator: int, denominator: int) -> bool:
    expected = numerator / denominator if denominator else None
    if expected is None or actual is None:
        return expected is actual
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)


def _validate_negative_safety_algebra(result: Mapping[str, Any], context: str) -> None:
    """Reject internally impossible aggregate safety claims before publication."""

    def require(claim: str, fields: tuple[str, ...]) -> None:
        missing = [field for field in fields if field not in result]
        if missing:
            raise PublicSnapshotError(
                f"{context}.{claim} requires reconstructable fields: {', '.join(missing)}"
            )

    if "measurement_status" in result:
        require(
            "measurement_status",
            ("controls_total", "controls_examined", "controls_not_examined"),
        )
    if "control_match_coverage" in result:
        if "controls_matched" not in result and "matched_control_count" not in result:
            raise PublicSnapshotError(
                f"{context}.control_match_coverage requires reconstructable matched count"
            )
        require("control_match_coverage", ("controls_total",))
    if "control_examination_coverage" in result:
        require("control_examination_coverage", ("controls_total", "controls_examined"))
    if "control_match_coverage_defined" in result:
        require("control_match_coverage_defined", ("controls_total",))
    if "false_primary_rate" in result:
        if "matched_candidates" not in result and "matched_candidate_count" not in result:
            raise PublicSnapshotError(
                f"{context}.false_primary_rate requires reconstructable matched-candidate count"
            )
        require("false_primary_rate", ("false_primary_count",))
    for count_name, gate_name in (
        ("false_primary_count", "zero_false_primary_gate_passed"),
        ("false_primary_export_count", "zero_false_primary_export_gate_passed"),
    ):
        if gate_name in result:
            require(
                gate_name,
                (
                    "controls_total",
                    "controls_examined",
                    "controls_not_examined",
                    "measurement_status",
                    count_name,
                ),
            )
    if "control_trials" in result:
        trials = result["control_trials"]
        missing = [
            field
            for field in ("total", "matched", "examined", "passed_by_abstention")
            if field not in trials
        ]
        if missing:
            raise PublicSnapshotError(
                f"{context}.control_trials requires reconstructable fields: {', '.join(missing)}"
            )

    total = result.get("controls_total")
    matched = result.get("controls_matched", result.get("matched_control_count"))
    examined = result.get("controls_examined")
    not_examined = result.get("controls_not_examined")
    abstained = result.get("passed_by_abstention_count")
    if (
        "controls_matched" in result
        and "matched_control_count" in result
        and result["controls_matched"] != result["matched_control_count"]
    ):
        raise PublicSnapshotError(f"{context} matched-control counts disagree")
    if (
        "matched_candidates" in result
        and "matched_candidate_count" in result
        and result["matched_candidates"] != result["matched_candidate_count"]
    ):
        raise PublicSnapshotError(f"{context} matched-candidate counts disagree")
    if total is not None:
        for name, count in (("matched", matched), ("examined", examined)):
            if count is not None and count > total:
                raise PublicSnapshotError(f"{context}.{name} count exceeds controls_total")
    if matched is not None and examined is not None and matched > examined:
        raise PublicSnapshotError(f"{context}.controls_matched exceeds controls_examined")
    if (
        total is not None
        and examined is not None
        and not_examined is not None
        and examined + not_examined != total
    ):
        raise PublicSnapshotError(f"{context} examined partition is inconsistent")
    if (
        matched is not None
        and examined is not None
        and abstained is not None
        and matched + abstained != examined
    ):
        raise PublicSnapshotError(f"{context} examined outcomes are inconsistent")
    if (
        total is not None
        and matched is not None
        and "control_match_coverage" in result
        and not _fraction_agrees(result["control_match_coverage"], matched, total)
    ):
        raise PublicSnapshotError(f"{context}.control_match_coverage is inconsistent")
    if (
        total is not None
        and examined is not None
        and "control_examination_coverage" in result
        and not _fraction_agrees(result["control_examination_coverage"], examined, total)
    ):
        raise PublicSnapshotError(f"{context}.control_examination_coverage is inconsistent")
    if (
        total is not None
        and "control_match_coverage_defined" in result
        and result["control_match_coverage_defined"] is not (total > 0)
    ):
        raise PublicSnapshotError(f"{context}.control_match_coverage_defined is inconsistent")
    if total is not None and examined is not None and "measurement_status" in result:
        expected_status = (
            "measured"
            if total > 0 and examined == total
            else "partially_measured"
            if examined > 0
            else "not_measured"
        )
        if result["measurement_status"] != expected_status:
            raise PublicSnapshotError(f"{context}.measurement_status is inconsistent")
        fully_measured = total > 0 and examined == total
        for count_name, gate_name in (
            ("false_primary_count", "zero_false_primary_gate_passed"),
            ("false_primary_export_count", "zero_false_primary_export_gate_passed"),
        ):
            if count_name in result and gate_name in result:
                expected_gate = False if result[count_name] else True if fully_measured else None
                if result[gate_name] is not expected_gate:
                    raise PublicSnapshotError(f"{context}.{gate_name} is inconsistent")
    matched_candidates = result.get("matched_candidates", result.get("matched_candidate_count"))
    false_primary = result.get("false_primary_count")
    false_primary_export = result.get("false_primary_export_count")
    if matched_candidates is not None and false_primary is not None:
        if false_primary > matched_candidates:
            raise PublicSnapshotError(f"{context}.false_primary_count exceeds matched candidates")
        rate_disagrees = "false_primary_rate" in result and not _fraction_agrees(
            result["false_primary_rate"], false_primary, matched_candidates
        )
        # The scorer intentionally reports 0.0 rather than null for a zero basis.
        zero_basis_scorer_value = (
            matched_candidates == 0
            and false_primary == 0
            and result.get("false_primary_rate") == 0.0
        )
        if rate_disagrees and not zero_basis_scorer_value:
            raise PublicSnapshotError(f"{context}.false_primary_rate is inconsistent")
    if (
        false_primary is not None
        and false_primary_export is not None
        and false_primary_export > false_primary
    ):
        raise PublicSnapshotError(
            f"{context}.false_primary_export_count exceeds false_primary_count"
        )
    trials = result.get("control_trials")
    if isinstance(trials, Mapping):
        trial_total = trials.get("total")
        trial_matched = trials.get("matched")
        trial_examined = trials.get("examined")
        trial_abstained = trials.get("passed_by_abstention")
        if trial_total is not None:
            if trial_matched is not None and trial_matched > trial_total:
                raise PublicSnapshotError(f"{context}.control_trials.matched exceeds total")
            if trial_examined is not None and trial_examined > trial_total:
                raise PublicSnapshotError(f"{context}.control_trials.examined exceeds total")
        if trial_matched is not None and trial_examined is not None:
            if trial_matched > trial_examined:
                raise PublicSnapshotError(f"{context}.control_trials matched exceeds examined")
            if trial_abstained is not None and trial_matched + trial_abstained != trial_examined:
                raise PublicSnapshotError(f"{context}.control_trials outcomes are inconsistent")


def _project_input_observability(value: Any, context: str) -> dict[str, Any]:
    source = _as_mapping(value, context)
    result: dict[str, Any] = {}
    if "status" in source:
        status = source["status"]
        if not isinstance(status, str) or status not in {
            "measured",
            "partially_assessed",
            "not_assessed",
        }:
            raise PublicSnapshotError(f"{context}.status is unsupported")
        result["status"] = status
    for name in (
        "papers_measured",
        "papers_not_assessed",
        "cases_measured",
        "cases_not_assessed",
        "measured_reference_observations",
        "reference_observations",
        "observable_reference_observations",
        "unobservable_reference_observations",
    ):
        if name in source:
            result[name] = _nullable_nonnegative_int(source[name], f"{context}.{name}")
    if "observation_coverage" in source:
        result["observation_coverage"] = _nullable_fraction(
            source["observation_coverage"],
            f"{context}.observation_coverage",
        )
    if "model_conditional_detection" in source:
        detection_context = f"{context}.model_conditional_detection"
        detection = _as_mapping(source["model_conditional_detection"], detection_context)
        public_detection: dict[str, Any] = {}
        for name in ("true_positives", "false_negatives", "recall_basis"):
            if name in detection:
                public_detection[name] = _nullable_nonnegative_int(
                    detection[name],
                    f"{detection_context}.{name}",
                )
        if "recall" in detection:
            public_detection["recall"] = _nullable_fraction(
                detection["recall"],
                f"{detection_context}.recall",
            )
        result["model_conditional_detection"] = public_detection
    _validate_input_observability_algebra(result, context)
    return result


def _validate_input_observability_algebra(result: Mapping[str, Any], context: str) -> None:
    """Reject impossible observability partitions and conditional recall claims."""

    if not result:
        return
    status = result.get("status")
    if status is None:
        raise PublicSnapshotError(f"{context}.status is required")
    names = (
        "reference_observations",
        "observable_reference_observations",
        "unobservable_reference_observations",
        "observation_coverage",
    )
    detection = result.get("model_conditional_detection")
    assessment_dimensions = [
        ("papers", "papers_measured", "papers_not_assessed"),
        ("cases", "cases_measured", "cases_not_assessed"),
    ]
    present_dimensions = [
        item for item in assessment_dimensions if item[1] in result or item[2] in result
    ]
    if len(present_dimensions) > 1:
        raise PublicSnapshotError(f"{context} mixes paper and case assessment counts")
    has_assessment_accounting = bool(present_dimensions) or (
        "measured_reference_observations" in result
    )
    if has_assessment_accounting and not present_dimensions:
        raise PublicSnapshotError(f"{context} assessment counts must be complete")
    if present_dimensions:
        dimension, measured_name, not_assessed_name = present_dimensions[0]
        if any(
            name not in result
            for name in (measured_name, not_assessed_name, "measured_reference_observations")
        ):
            raise PublicSnapshotError(f"{context} {dimension}-assessment counts must be complete")
        assessed = result[measured_name]
        not_assessed = result[not_assessed_name]
        measured_references = result["measured_reference_observations"]
        if None in (assessed, not_assessed, measured_references):
            raise PublicSnapshotError(f"{context} {dimension}-assessment counts cannot be null")
        if status == "measured" and (assessed < 1 or not_assessed != 0):
            raise PublicSnapshotError(f"{context} measured {dimension} counts are inconsistent")
        if status == "partially_assessed" and (assessed < 1 or not_assessed < 1):
            raise PublicSnapshotError(
                f"{context} partially_assessed {dimension} counts are inconsistent"
            )
        if status == "not_assessed" and (assessed != 0 or measured_references != 0):
            raise PublicSnapshotError(f"{context} not_assessed {dimension} counts are inconsistent")
    if status == "partially_assessed":
        if not has_assessment_accounting:
            raise PublicSnapshotError(f"{context} partially_assessed counts are required")
        if any(result.get(name) is not None for name in names):
            raise PublicSnapshotError(f"{context} partially_assessed aggregate fields must be null")
        if not isinstance(detection, Mapping) or any(
            value is not None for value in detection.values()
        ):
            raise PublicSnapshotError(
                f"{context} partially_assessed detection must be complete and null"
            )
        required_detection = {"true_positives", "false_negatives", "recall_basis", "recall"}
        if set(detection) != required_detection:
            raise PublicSnapshotError(
                f"{context} partially_assessed detection must be complete and null"
            )
        return
    if status == "not_assessed":
        if any(
            result.get(name) is not None
            for name in (
                "observable_reference_observations",
                "unobservable_reference_observations",
                "observation_coverage",
            )
        ):
            raise PublicSnapshotError(f"{context} not_assessed observable counts must be null")
        if isinstance(detection, Mapping):
            required_detection = {"true_positives", "false_negatives", "recall_basis", "recall"}
            if set(detection) != required_detection or any(
                value is not None for value in detection.values()
            ):
                raise PublicSnapshotError(
                    f"{context} not_assessed detection must be complete and null"
                )
        return
    if any(name not in result or result[name] is None for name in names[:3]):
        raise PublicSnapshotError(f"{context} measured counts must be complete")
    reference = result["reference_observations"]
    observable = result["observable_reference_observations"]
    unobservable = result["unobservable_reference_observations"]
    if observable + unobservable != reference:
        raise PublicSnapshotError(f"{context} observation partition is inconsistent")
    if has_assessment_accounting and result["measured_reference_observations"] != reference:
        raise PublicSnapshotError(f"{context}.measured_reference_observations is inconsistent")
    if "observation_coverage" not in result or not _fraction_agrees(
        result["observation_coverage"], observable, reference
    ):
        raise PublicSnapshotError(f"{context}.observation_coverage is inconsistent")
    if not isinstance(detection, Mapping):
        raise PublicSnapshotError(f"{context}.model_conditional_detection is required")
    for name in ("true_positives", "false_negatives", "recall_basis"):
        if name not in detection or detection[name] is None:
            raise PublicSnapshotError(f"{context}.model_conditional_detection is incomplete")
    true_positives = detection["true_positives"]
    false_negatives = detection["false_negatives"]
    recall_basis = detection["recall_basis"]
    if true_positives + false_negatives != recall_basis or recall_basis != observable:
        raise PublicSnapshotError(f"{context} model-conditional detection is inconsistent")
    if "recall" not in detection or not _fraction_agrees(
        detection["recall"], true_positives, recall_basis
    ):
        raise PublicSnapshotError(f"{context}.model_conditional_detection.recall is inconsistent")


def _project_quality(
    value: Any, *, current: bool = False, execution: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    source = _as_mapping(value or {}, "quality")
    macro = _as_mapping(source.get("macro", {}), "quality.macro")
    micro = _as_mapping(source.get("micro", {}), "quality.micro")
    macro_detection = _as_mapping(macro.get("detection", {}), "quality.macro.detection")
    macro_detection_public = {
        name: _nullable_fraction(macro_detection[name], f"quality.macro.detection.{name}")
        for name in ("precision", "recall", "f1")
        if name in macro_detection
    }
    for basis_name in ("defined_cases", "undefined_cases"):
        if basis_name in macro_detection:
            basis = _as_mapping(
                macro_detection[basis_name],
                f"quality.macro.detection.{basis_name}",
            )
            macro_detection_public[basis_name] = {
                name: _nonnegative_int(basis[name], f"quality.macro.detection.{basis_name}.{name}")
                for name in ("precision", "recall", "f1")
                if name in basis
            }
    micro_detection = _project_detection(micro.get("detection", {}), "quality.micro.detection")
    micro_fields = _as_mapping(micro.get("field_accuracy", {}), "quality.micro.field_accuracy")
    micro_public = {
        "detection": micro_detection,
        "field_accuracy": {
            field: _nullable_fraction(micro_fields[field], f"quality.micro.field_accuracy.{field}")
            for field in _QUALITY_FIELDS
            if field in micro_fields
        },
    }
    if "reference_observations" in micro:
        micro_public["reference_observations"] = _nonnegative_int(
            micro["reference_observations"], "quality.micro.reference_observations"
        )
    if "input_observability" in micro:
        micro_public["input_observability"] = _project_input_observability(
            micro["input_observability"],
            "quality.micro.input_observability",
        )
    result = {
        "macro": {
            "detection": macro_detection_public,
            "field_accuracy": {
                field: _nullable_fraction(
                    _as_mapping(macro.get("field_accuracy", {}), "quality.macro.field_accuracy")[
                        field
                    ],
                    f"quality.macro.field_accuracy.{field}",
                )
                for field in _QUALITY_FIELDS
                if field
                in _as_mapping(macro.get("field_accuracy", {}), "quality.macro.field_accuracy")
            },
        },
        "micro": micro_public,
    }
    for field in ("scored_cases", "scored_repetitions"):
        if field in source:
            result[field] = _nonnegative_int(source[field], f"quality.{field}")
    if current:
        if any(field not in result for field in ("scored_cases", "scored_repetitions")):
            raise PublicSnapshotError("quality scoring denominators are incomplete")
        if execution is None:
            raise PublicSnapshotError("quality execution binding is missing")
        if (
            result["scored_cases"] != execution["cases_scored"]
            or result["scored_repetitions"] != execution["repetitions_scored"]
        ):
            raise PublicSnapshotError("quality scoring denominators disagree with execution")
        if "reference_observations" not in micro_public:
            raise PublicSnapshotError("quality.micro.reference_observations is required")
        _validate_detection_algebra(
            micro_detection, "quality.micro.detection", require_complete=True
        )
        if micro_detection["recall_basis"] != micro_public["reference_observations"]:
            raise PublicSnapshotError("quality micro recall basis disagrees with references")
        if set(micro_public["field_accuracy"]) != set(_QUALITY_FIELDS):
            raise PublicSnapshotError("quality.micro.field_accuracy is incomplete")
        macro_fields = result["macro"]["field_accuracy"]
        if set(macro_fields) != set(_QUALITY_FIELDS):
            raise PublicSnapshotError("quality.macro.field_accuracy is incomplete")
        no_references = micro_public["reference_observations"] == 0
        if any((value is None) != no_references for value in macro_fields.values()):
            raise PublicSnapshotError("quality.macro.field_accuracy disagrees with reference basis")
        if set(macro_detection_public) != {
            "precision",
            "recall",
            "f1",
            "defined_cases",
            "undefined_cases",
        }:
            raise PublicSnapshotError("quality.macro.detection is incomplete")
        for basis_name in ("defined_cases", "undefined_cases"):
            if set(macro_detection_public[basis_name]) != {"precision", "recall", "f1"}:
                raise PublicSnapshotError(f"quality.macro.detection.{basis_name} is incomplete")
        for metric in ("precision", "recall", "f1"):
            defined = macro_detection_public["defined_cases"][metric]
            undefined = macro_detection_public["undefined_cases"][metric]
            if defined + undefined != result["scored_repetitions"]:
                raise PublicSnapshotError(f"quality.macro.detection.{metric} denominators disagree")
            if (macro_detection_public[metric] is None) != (defined == 0):
                raise PublicSnapshotError(
                    f"quality.macro.detection.{metric} value disagrees with denominator"
                )
        if "input_observability" not in micro_public:
            raise PublicSnapshotError("quality.micro.input_observability is required")
        observability = micro_public["input_observability"]
        required_observability_counts = {
            "cases_measured",
            "cases_not_assessed",
            "measured_reference_observations",
        }
        if not required_observability_counts <= set(observability):
            raise PublicSnapshotError(
                "quality.micro.input_observability assessment counts are incomplete"
            )
        if (
            observability["cases_measured"] + observability["cases_not_assessed"]
            != result["scored_repetitions"]
        ):
            raise PublicSnapshotError(
                "quality.micro.input_observability repetition denominator disagrees"
            )
        if (
            observability["measured_reference_observations"]
            > micro_public["reference_observations"]
        ):
            raise PublicSnapshotError(
                "quality.micro.input_observability measured reference basis exceeds quality basis"
            )
    return result


def _project_usage(
    value: Any, *, calls_attempted: int | None = None, current: bool = False
) -> dict[str, Any]:
    source = _as_mapping(value or {}, "usage")
    result: dict[str, Any] = {}
    for name in (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cost_usd",
    ):
        if name in source:
            context = f"usage.{name}"
            item = _as_mapping(source[name], context)
            if "total" not in item:
                raise PublicSnapshotError(f"{context}.total is required")
            total = _finite_number(item["total"], f"{context}.total")
            if total < 0 or (name != "cost_usd" and not float(total).is_integer()):
                raise PublicSnapshotError(f"{context}.total must be non-negative")
            reported = _nonnegative_int(item.get("reported_calls"), f"{context}.reported_calls")
            missing = _nonnegative_int(item.get("missing_calls"), f"{context}.missing_calls")
            if calls_attempted is not None and reported + missing != calls_attempted:
                raise PublicSnapshotError(f"{context} call partition disagrees")
            result[name] = {
                "total": total if name == "cost_usd" else int(total),
                "reported_calls": reported,
                "missing_calls": missing,
            }
    if "latency_seconds" in source:
        context = "usage.latency_seconds"
        item = _as_mapping(source["latency_seconds"], context)
        latency: dict[str, Any] = {}
        for name in ("total", "mean", "p50", "p95", "max"):
            if name not in item:
                if current:
                    raise PublicSnapshotError(f"{context}.{name} is required")
                continue
            if item[name] is None and name in {"mean", "p50", "p95"}:
                latency[name] = None
                continue
            number = _finite_number(item[name], f"{context}.{name}")
            if number < 0:
                raise PublicSnapshotError(f"{context}.{name} must be non-negative")
            latency[name] = number
        reported = _nonnegative_int(item.get("reported_calls"), f"{context}.reported_calls")
        missing = _nonnegative_int(item.get("missing_calls"), f"{context}.missing_calls")
        if calls_attempted is not None and reported + missing != calls_attempted:
            raise PublicSnapshotError(f"{context} call partition disagrees")
        latency.update({"reported_calls": reported, "missing_calls": missing})
        if current:
            summary_values = [latency[name] for name in ("total", "mean", "p50", "p95", "max")]
            if reported == 0:
                if any(value != 0.0 for value in summary_values):
                    raise PublicSnapshotError(
                        f"{context} must contain zero summaries without reported calls"
                    )
            elif any(value is None for value in summary_values):
                raise PublicSnapshotError(f"{context} summaries require reported calls")
            elif not (
                latency["p50"] <= latency["p95"] <= latency["max"]
                and math.isclose(
                    latency["mean"] * reported,
                    latency["total"],
                    rel_tol=0.0,
                    abs_tol=(reported + 1) * 1e-6,
                )
            ):
                raise PublicSnapshotError(f"{context} summaries are inconsistent")
        result["latency_seconds"] = latency
    if current:
        expected = {
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
            "cost_usd",
            "latency_seconds",
        }
        if set(result) != expected:
            raise PublicSnapshotError("usage is incomplete for current model score")
    return result


def _project_model_evidence(value: Any, *, current: bool) -> dict[str, Any]:
    source = _as_mapping(value or {}, "model.aggregate.evidence")
    support_source = _as_mapping(
        source.get("candidate_text_support", {}),
        "model.aggregate.evidence.candidate_text_support",
    )
    support_fields = ("supported", "partially_supported", "unsupported", "unverified")
    support = {
        field: _nonnegative_int(
            support_source[field],
            f"model.aggregate.evidence.candidate_text_support.{field}",
        )
        for field in support_fields
        if field in support_source
    }

    def accuracy(name: str) -> dict[str, float | None]:
        context = f"model.aggregate.evidence.{name}"
        item = _as_mapping(source.get(name, {}), context)
        result = {
            field: _nullable_fraction(item[field], f"{context}.{field}")
            for field in ("macro", "micro")
            if field in item
        }
        if current and set(result) != {"macro", "micro"}:
            raise PublicSnapshotError(f"{context} is incomplete")
        return result

    if current and set(support) != set(support_fields):
        raise PublicSnapshotError("model.aggregate.evidence.candidate_text_support is incomplete")
    return {
        "candidate_text_support": support,
        "reference_evidence_supported_accuracy": accuracy("reference_evidence_supported_accuracy"),
        "reference_page_anchor_accuracy": accuracy("reference_page_anchor_accuracy"),
    }


def _project_model_aggregate(value: Any, *, current: bool = False) -> dict[str, Any]:
    source = _as_mapping(value or {}, "model.aggregate")
    selection_gates = _as_mapping(
        source.get("model_selection_gates", {}), "model.aggregate.model_selection_gates"
    )
    execution_source = _as_mapping(source.get("execution", {}), "model.aggregate.execution")
    execution_count_fields = (
        "cases_attempted",
        "cases_succeeded",
        "cases_partial_failure",
        "cases_failed",
        "cases_scored",
        "repetitions_predeclared",
        "repetitions_attempted",
        "repetitions_succeeded",
        "repetitions_partial_failure",
        "repetitions_failed",
        "repetitions_scored",
        "calls_predeclared",
        "calls_attempted",
        "calls_succeeded",
        "calls_failed",
    )
    execution = {
        field: _nonnegative_int(execution_source[field], f"model.aggregate.execution.{field}")
        for field in execution_count_fields
        if field in execution_source
    }
    for field in ("case_success_rate", "case_scored_rate", "call_success_rate"):
        if field in execution_source:
            execution[field] = _nullable_fraction(
                execution_source[field], f"model.aggregate.execution.{field}"
            )
    if current:
        if any(field not in execution for field in execution_count_fields) or any(
            field not in execution
            for field in ("case_success_rate", "case_scored_rate", "call_success_rate")
        ):
            raise PublicSnapshotError("model.aggregate.execution is incomplete")
        for prefix in ("cases", "repetitions", "calls"):
            attempted = execution[f"{prefix}_attempted"]
            parts = (
                execution[f"{prefix}_succeeded"] + execution[f"{prefix}_failed"]
                if prefix == "calls"
                else execution[f"{prefix}_succeeded"]
                + execution[f"{prefix}_partial_failure"]
                + execution[f"{prefix}_failed"]
            )
            if attempted != parts:
                raise PublicSnapshotError(f"model.aggregate.execution {prefix} partition disagrees")
        if (
            execution["cases_scored"] > execution["cases_attempted"]
            or execution["repetitions_scored"] > execution["repetitions_attempted"]
        ):
            raise PublicSnapshotError("model.aggregate.execution scored counts exceed attempts")
        if (
            execution["calls_predeclared"] != execution["calls_attempted"]
            or execution["repetitions_predeclared"] != execution["repetitions_attempted"]
        ):
            raise PublicSnapshotError("model.aggregate.execution omits predeclared attempts")
        if execution["repetitions_scored"] > execution["repetitions_succeeded"]:
            raise PublicSnapshotError(
                "model.aggregate.execution scored repetitions exceed successful repetitions"
            )
        for field, numerator, denominator in (
            ("case_success_rate", "cases_succeeded", "cases_attempted"),
            ("case_scored_rate", "cases_scored", "cases_attempted"),
            ("call_success_rate", "calls_succeeded", "calls_attempted"),
        ):
            if not _fraction_agrees(execution[field], execution[numerator], execution[denominator]):
                raise PublicSnapshotError(f"model.aggregate.execution.{field} is inconsistent")

    schema_source = _as_mapping(source.get("schema", {}), "model.aggregate.schema")
    schema_count_fields = (
        "calls_in_end_to_end_denominator",
        "structured_responses_observed",
        "structured_responses_valid",
        "structured_responses_invalid",
        "structured_response_not_observed",
    )
    schema = {
        field: _nonnegative_int(schema_source[field], f"model.aggregate.schema.{field}")
        for field in schema_count_fields
        if field in schema_source
    }
    for field in ("valid_rate_of_observed_responses", "end_to_end_schema_success_rate"):
        if field in schema_source:
            schema[field] = _nullable_fraction(
                schema_source[field], f"model.aggregate.schema.{field}"
            )
    if current:
        if any(field not in schema for field in schema_count_fields) or any(
            field not in schema
            for field in ("valid_rate_of_observed_responses", "end_to_end_schema_success_rate")
        ):
            raise PublicSnapshotError("model.aggregate.schema is incomplete")
        if schema["structured_responses_observed"] != (
            schema["structured_responses_valid"] + schema["structured_responses_invalid"]
        ) or schema["calls_in_end_to_end_denominator"] != (
            schema["structured_responses_observed"] + schema["structured_response_not_observed"]
        ):
            raise PublicSnapshotError("model.aggregate.schema response partition disagrees")
        if schema["calls_in_end_to_end_denominator"] != execution["calls_attempted"]:
            raise PublicSnapshotError("model.aggregate.schema denominator disagrees with execution")
        if execution["calls_succeeded"] > schema["structured_responses_valid"]:
            raise PublicSnapshotError(
                "model.aggregate successful calls exceed schema-valid responses"
            )
        if not _fraction_agrees(
            schema["valid_rate_of_observed_responses"],
            schema["structured_responses_valid"],
            schema["structured_responses_observed"],
        ) or not _fraction_agrees(
            schema["end_to_end_schema_success_rate"],
            schema["structured_responses_valid"],
            schema["calls_in_end_to_end_denominator"],
        ):
            raise PublicSnapshotError("model.aggregate.schema rates are inconsistent")

    quality_measurement_source = _as_mapping(
        source.get("quality_measurement", {}), "model.aggregate.quality_measurement"
    )
    quality_measurement: dict[str, Any] = {}
    if quality_measurement_source:
        measurement_status = quality_measurement_source.get("status")
        if measurement_status not in {"measured", "partially_measured", "unmeasured"}:
            raise PublicSnapshotError("model.aggregate.quality_measurement.status is unsupported")
        scored = _nonnegative_int(
            quality_measurement_source.get("scored_repetitions"),
            "model.aggregate.quality_measurement.scored_repetitions",
        )
        unmeasured = _nonnegative_int(
            quality_measurement_source.get("unmeasured_repetitions"),
            "model.aggregate.quality_measurement.unmeasured_repetitions",
        )
        if current and scored + unmeasured != execution["repetitions_attempted"]:
            raise PublicSnapshotError("model.aggregate.quality_measurement partition disagrees")
        if current and scored != execution["repetitions_scored"]:
            raise PublicSnapshotError(
                "model.aggregate.quality_measurement disagrees with scored repetitions"
            )
        expected_status = (
            "measured"
            if scored > 0 and unmeasured == 0
            else "partially_measured"
            if scored > 0
            else "unmeasured"
        )
        if measurement_status != expected_status:
            raise PublicSnapshotError("model.aggregate.quality_measurement.status is inconsistent")
        quality_measurement = {
            "status": measurement_status,
            "scored_repetitions": scored,
            "unmeasured_repetitions": unmeasured,
        }
    elif current:
        raise PublicSnapshotError("model.aggregate.quality_measurement is required")
    quality_value = source.get("quality")
    if current and quality_measurement.get("status") != "unmeasured" and quality_value is None:
        raise PublicSnapshotError("measured model aggregate requires quality")
    if current and quality_measurement.get("status") == "unmeasured" and quality_value is not None:
        raise PublicSnapshotError("unmeasured model aggregate cannot contain quality")
    claim_type_classification = _project_claim_type_classification(
        source.get("claim_type_classification", {})
    )
    if (
        current
        and quality_measurement.get("status") != "unmeasured"
        and not claim_type_classification
    ):
        raise PublicSnapshotError("measured model aggregate requires claim_type_classification")
    negative_control_safety = _project_negative_safety(source.get("negative_control_safety", {}))
    model_selection_gates = {
        name: _project_gate(selection_gates.get(name, {}), f"model_selection_gates.{name}")
        for name in (
            "claim_type_macro_f1",
            "false_primary_controls",
            "false_primary_exports",
        )
        if name in selection_gates
    }
    if current and quality_measurement.get("status") == "unmeasured":
        if claim_type_classification or model_selection_gates:
            raise PublicSnapshotError(
                "unmeasured model aggregate cannot contain measured classification or gates"
            )
        if execution["cases_scored"] != 0:
            raise PublicSnapshotError("unmeasured model aggregate cannot contain scored cases")
    if current and quality_measurement.get("status") != "unmeasured":
        fully_measured_controls = negative_control_safety.get("measurement_status") == "measured"
        false_primary = negative_control_safety.get("false_primary_count")
        false_primary_exports = negative_control_safety.get("false_primary_export_count")
        _require_gate_bindings(
            model_selection_gates,
            {
                "claim_type_macro_f1": (
                    claim_type_classification["macro_f1"]
                    if claim_type_classification["supported_classes"] >= 2
                    else None
                ),
                "false_primary_controls": (
                    float(false_primary) if false_primary or fully_measured_controls else None
                ),
                "false_primary_exports": (
                    float(false_primary_exports)
                    if false_primary_exports or fully_measured_controls
                    else None
                ),
            },
            "model_selection_gates",
        )
    result = {
        "execution": execution,
        "schema": schema,
        "quality_measurement": quality_measurement,
        "quality": (
            _project_quality(quality_value, current=current, execution=execution)
            if quality_value is not None
            else None
        ),
        "evidence": _project_model_evidence(source.get("evidence"), current=current),
        "negative_control_safety": negative_control_safety,
        "claim_type_classification": claim_type_classification,
        "model_selection_gates": model_selection_gates,
        "usage": _project_usage(
            source.get("usage", {}),
            calls_attempted=execution.get("calls_attempted"),
            current=current,
        ),
    }
    if current and result["quality"] is not None:
        for evidence_name, quality_field in (
            ("reference_evidence_supported_accuracy", "evidence_supported"),
            ("reference_page_anchor_accuracy", "page"),
        ):
            for aggregation in ("macro", "micro"):
                evidence_value = result["evidence"][evidence_name][aggregation]
                quality_value = result["quality"][aggregation]["field_accuracy"][quality_field]
                if (evidence_value is None) != (quality_value is None) or (
                    evidence_value is not None
                    and not math.isclose(
                        evidence_value,
                        quality_value,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                ):
                    raise PublicSnapshotError(
                        f"model.aggregate.evidence.{evidence_name}.{aggregation} "
                        f"disagrees with quality.{quality_field}"
                    )
    elif current:
        for evidence_name in (
            "reference_evidence_supported_accuracy",
            "reference_page_anchor_accuracy",
        ):
            if any(
                result["evidence"][evidence_name][key] is not None for key in ("macro", "micro")
            ):
                raise PublicSnapshotError(
                    "unmeasured model aggregate cannot contain measured evidence accuracy"
                )
    return result


def _project_model_selection(path: Path, selected_model: str | None) -> dict[str, Any]:
    raw = _as_mapping(read_json(path), "model selection")
    source_schema_version = raw.get("schema_version")
    sealed_score_versions = {
        "extractor-bakeoff-score/0.2",
        "extractor-bakeoff-score/0.3",
    }
    if source_schema_version in sealed_score_versions:
        if source_schema_version == "extractor-bakeoff-score/0.2" and selected_model is not None:
            raise PublicSnapshotError(
                "selected model requires the current sealed offline-scored artifact"
            )
        for field in (
            "configuration_sha256",
            "run_contract_sha256",
            "provider_phase_seal_sha256",
            "checkpoint_sha256",
            "stage_contract_sha256",
            "request_contract_sha256",
        ):
            value = raw.get(field)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise PublicSnapshotError(f"model selection {field} is missing or invalid")
        request_contract = _project_request_contract(raw.get("request_contract"))
        if raw["request_contract_sha256"] != sha256_bytes(
            canonical_json_bytes(raw["request_contract"])
        ):
            raise PublicSnapshotError("model selection request contract hash is invalid")
        privacy = _as_mapping(raw.get("privacy"), "model selection.privacy")
        declared = _as_sequence(raw.get("declared_models"), "model selection.declared_models")
        executed = _as_sequence(raw.get("executed_models"), "model selection.executed_models")
        if (
            any(not isinstance(item, str) or not item for item in [*declared, *executed])
            or len(set(declared)) != len(declared)
            or len(set(executed)) != len(executed)
            or not set(executed) <= set(declared)
        ):
            raise PublicSnapshotError("model selection model partitions are invalid")
        execution_selection_sha256 = raw.get("execution_selection_sha256")
        if source_schema_version == "extractor-bakeoff-score/0.3" and (
            not isinstance(execution_selection_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", execution_selection_sha256) is None
        ):
            raise PublicSnapshotError(
                "current model selection execution_selection_sha256 is missing or invalid"
            )
        references_loaded = privacy.get("references_loaded_after_provider_phase_sealed")
        expected_references_loaded = bool(executed)
        if privacy.get("reference_paths_in_score") is not False or references_loaded is not (
            expected_references_loaded
            if source_schema_version == "extractor-bakeoff-score/0.3"
            else True
        ):
            raise PublicSnapshotError("model selection is not a sealed scored artifact")
    elif source_schema_version == "extractor-bakeoff-result/0.3":
        if selected_model is not None:
            raise PublicSnapshotError(
                "selected model requires a sealed offline-scored selection artifact"
            )
        request_contract = _project_request_contract(raw.get("request_contract"))
        executed = None
    else:
        raise PublicSnapshotError("model selection schema_version is unsupported")
    models = []
    model_ids: list[str] = []
    current_score = source_schema_version == "extractor-bakeoff-score/0.3"
    allowed_eligibility_reasons = {
        "terminal_partition_incomplete",
        "wire_schema_below_gate",
        "request_contract_unsatisfied",
        "telemetry_incomplete",
    }
    for index, item in enumerate(_as_sequence(raw.get("models", []), "model selection.models")):
        model = _as_mapping(item, f"model selection.models[{index}]")
        model_id = model.get("model")
        if not isinstance(model_id, str) or not model_id:
            raise PublicSnapshotError("every public model result requires a model ID")
        if model_id in model_ids:
            raise PublicSnapshotError("model selection contains duplicate model IDs")
        contract_eligibility = model.get("contract_eligibility")
        matched_quality_status = model.get("matched_quality_status")
        quality_status = model.get("quality_status")
        public_model: dict[str, Any] = {
            "model": model_id,
            "label": str(model.get("label", model_id)),
        }
        if current_score:
            reason_codes = list(
                _as_sequence(
                    model.get("eligibility_reason_codes"),
                    f"model selection.models[{index}].eligibility_reason_codes",
                )
            )
            if any(reason not in allowed_eligibility_reasons for reason in reason_codes) or len(
                reason_codes
            ) != len(set(reason_codes)):
                raise PublicSnapshotError("current scored model has invalid eligibility reasons")
            was_executed = model_id in executed
            if was_executed:
                if (
                    contract_eligibility != "contract_eligible"
                    or reason_codes
                    or matched_quality_status != "executed"
                    or quality_status not in {"measured", "partially_measured", "unmeasured"}
                ):
                    raise PublicSnapshotError("current executed model has invalid status fields")
                aggregate = _project_model_aggregate(model.get("aggregate"), current=True)
                if quality_status != aggregate["quality_measurement"]["status"]:
                    raise PublicSnapshotError(
                        "model quality_status disagrees with aggregate measurement"
                    )
                public_model.update(
                    {
                        "contract_eligibility": contract_eligibility,
                        "eligibility_reason_codes": reason_codes,
                        "matched_quality_status": matched_quality_status,
                        "quality_status": quality_status,
                        "aggregate": aggregate,
                    }
                )
            else:
                if (
                    contract_eligibility != "contract_ineligible"
                    or not reason_codes
                    or matched_quality_status != "not_run"
                    or model.get("not_run_reason") != "contract_ineligible_on_sealed_smoke"
                    or quality_status != "unmeasured_contract_ineligible"
                    or model.get("quality") is not None
                    or model.get("aggregate") is not None
                    or list(
                        _as_sequence(
                            model.get("cases"),
                            f"model selection.models[{index}].cases",
                        )
                    )
                ):
                    raise PublicSnapshotError("current skipped model has invalid status fields")
                public_model.update(
                    {
                        "contract_eligibility": contract_eligibility,
                        "eligibility_reason_codes": reason_codes,
                        "matched_quality_status": matched_quality_status,
                        "not_run_reason": "contract_ineligible_on_sealed_smoke",
                        "quality_status": quality_status,
                        "aggregate": None,
                    }
                )
        else:
            public_model["aggregate"] = _project_model_aggregate(model.get("aggregate"))
        model_ids.append(model_id)
        models.append(public_model)
    if current_score and model_ids != list(declared):
        raise PublicSnapshotError("scored model list disagrees with declared_models")
    if source_schema_version == "extractor-bakeoff-score/0.2" and model_ids != list(executed):
        raise PublicSnapshotError("scored model list disagrees with executed_models")
    if selected_model is not None:
        if selected_model not in model_ids or selected_model not in executed:
            raise PublicSnapshotError("selected model is absent from the model-selection artifact")
        selected_source = next(
            _as_mapping(item, "selected model")
            for item in raw["models"]
            if item.get("model") == selected_model
        )
        if (
            selected_source.get("contract_eligibility") != "contract_eligible"
            or selected_source.get("matched_quality_status") != "executed"
            or selected_source.get("quality_status") != "measured"
        ):
            raise PublicSnapshotError(
                "selected model lacks sealed contract eligibility and measured quality"
            )
        selected_public = next(item for item in models if item["model"] == selected_model)
        selected_gates = selected_public["aggregate"]["model_selection_gates"]
        if not selected_gates or any(
            gate["status"] != "passed" for gate in selected_gates.values()
        ):
            raise PublicSnapshotError("selected model does not pass every model-selection gate")
    determinism = _as_mapping(raw.get("determinism", {}), "model selection.determinism")
    segmentation = _as_mapping(
        determinism.get("segmentation", {}), "model selection.determinism.segmentation"
    )
    determinism_public = {
        **_fields(
            determinism,
            (
                "seed",
                "temperature",
                "reasoning_effort",
                "require_parameters",
                "fresh_repetitions",
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
    }
    code = _project_code(raw.get("code", {}))
    if current_score:
        required_determinism = {
            "seed",
            "temperature",
            "reasoning_effort",
            "require_parameters",
            "fresh_repetitions",
            "max_tokens",
            "min_confidence",
            "prompt_sha256",
            "reference_prompt_isolation",
            "segmentation",
        }
        if set(determinism_public) != required_determinism:
            raise PublicSnapshotError("current model selection determinism is incomplete")
        if (
            determinism_public["seed"] is not None
            or determinism_public["temperature"] is not None
            or determinism_public["reasoning_effort"] != "minimal"
            or determinism_public["require_parameters"] is not True
            or _nonnegative_int(
                determinism_public["fresh_repetitions"],
                "model selection.determinism.fresh_repetitions",
                positive=True,
            )
            != determinism_public["fresh_repetitions"]
            or _nonnegative_int(
                determinism_public["max_tokens"],
                "model selection.determinism.max_tokens",
                positive=True,
            )
            != 16_000
            or _nullable_fraction(
                determinism_public["min_confidence"],
                "model selection.determinism.min_confidence",
            )
            is None
            or not isinstance(determinism_public["prompt_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", determinism_public["prompt_sha256"]) is None
            or determinism_public["reference_prompt_isolation"] is not True
            or set(determinism_public["segmentation"]) != set(_SEGMENTATION_FIELDS)
        ):
            raise PublicSnapshotError("current model selection determinism is unsupported")
        if set(code) != {"git_commit", "git_dirty", "git_available", "source_tree_sha256"} or (
            not isinstance(code["source_tree_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", code["source_tree_sha256"]) is None
        ):
            raise PublicSnapshotError("current model selection code provenance is incomplete")
    return {
        "schema_version": MODEL_SELECTION_SCHEMA_VERSION,
        "source_artifact_sha256": sha256_file(path),
        "bakeoff_id": _safe_scalar(raw.get("bakeoff_id"), "model selection.bakeoff_id"),
        "configuration_sha256": _safe_scalar(
            raw.get("configuration_sha256"), "model selection.configuration_sha256"
        ),
        "execution_selection_sha256": (
            raw.get("execution_selection_sha256") if current_score else None
        ),
        "code": code,
        "determinism": determinism_public,
        "request_contract": request_contract,
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
    schema_version = source.get("schema_version")
    if schema_version not in {"reference-score/0.7", "corpus-reference-score/0.4"}:
        raise PublicSnapshotError("reference_evaluation.schema_version is unsupported")
    gates = _as_mapping(source.get("quality_gates", {}), "reference_evaluation.quality_gates")
    result = {
        "schema_version": schema_version,
        "bases": {
            name: _nonnegative_int(
                _as_mapping(source.get("bases", {}), "reference_evaluation.bases")[name],
                f"reference_evaluation.bases.{name}",
            )
            for name in (
                "reference_observations",
                "field_matching",
                "precision_candidates_in_fully_annotated_regions",
            )
            if name in _as_mapping(source.get("bases", {}), "reference_evaluation.bases")
        },
        "detection": _project_detection(
            source.get("detection", {}), "reference_evaluation.detection"
        ),
        "field_accuracy": {
            name: _nullable_fraction(
                _as_mapping(
                    source.get("field_accuracy", {}),
                    "reference_evaluation.field_accuracy",
                )[name],
                f"reference_evaluation.field_accuracy.{name}",
            )
            for name in _QUALITY_FIELDS
            if name
            in _as_mapping(
                source.get("field_accuracy", {}),
                "reference_evaluation.field_accuracy",
            )
        },
        "derived_accuracy": {
            name: _nullable_fraction(
                _as_mapping(
                    source.get("derived_accuracy", {}),
                    "reference_evaluation.derived_accuracy",
                )[name],
                f"reference_evaluation.derived_accuracy.{name}",
            )
            for name in ("exact_numeric_value_and_unit", "evidence_page_and_text_support")
            if name
            in _as_mapping(
                source.get("derived_accuracy", {}),
                "reference_evaluation.derived_accuracy",
            )
        },
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
    if "papers_scored" in source:
        result["papers_scored"] = _nonnegative_int(
            source["papers_scored"], "reference_evaluation.papers_scored"
        )
    if "reference_observations" in source:
        result["reference_observations"] = _nonnegative_int(
            source["reference_observations"],
            "reference_evaluation.reference_observations",
        )
    if "coverage_statement" in source:
        coverage_statement = source["coverage_statement"]
        if coverage_statement != _CORPUS_REFERENCE_COVERAGE_STATEMENT:
            raise PublicSnapshotError("reference_evaluation.coverage_statement is unsupported")
        result["coverage_statement"] = coverage_statement
    detection = result["detection"]
    if not result["claim_type_classification"]:
        raise PublicSnapshotError(f"{schema_version} requires claim_type_classification")
    if schema_version == "reference-score/0.7":
        if "reference_observations" not in result:
            raise PublicSnapshotError("reference-score/0.7 requires reference_observations")
        _validate_detection_algebra(
            detection, "reference_evaluation.detection", require_complete=True
        )
        if detection["recall_basis"] != result["reference_observations"]:
            raise PublicSnapshotError(
                "reference_evaluation detection basis disagrees with references"
            )
        if set(result["field_accuracy"]) != set(_QUALITY_FIELDS):
            raise PublicSnapshotError("reference-score/0.7 field_accuracy is incomplete")
    else:
        required_bases = {
            "reference_observations",
            "field_matching",
            "precision_candidates_in_fully_annotated_regions",
        }
        required_detection = {
            "true_positives",
            "precision_true_positives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
        }
        if set(result["bases"]) != required_bases or not required_detection <= set(detection):
            raise PublicSnapshotError("corpus-reference-score/0.4 is incomplete")
        reference_basis = result["bases"]["reference_observations"]
        precision_basis = result["bases"]["precision_candidates_in_fully_annotated_regions"]
        if detection["true_positives"] + detection["false_negatives"] != reference_basis:
            raise PublicSnapshotError("reference_evaluation recall counts disagree")
        if detection["precision_true_positives"] + detection["false_positives"] != precision_basis:
            raise PublicSnapshotError("reference_evaluation precision counts disagree")
        if not _fraction_agrees(
            detection["precision"], detection["precision_true_positives"], precision_basis
        ) or not _fraction_agrees(
            detection["recall"], detection["true_positives"], reference_basis
        ):
            raise PublicSnapshotError("reference_evaluation detection rates disagree")
        precision = detection["precision"]
        recall = detection["recall"]
        expected_f1 = (
            None
            if precision is None or recall is None
            else 0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        if (expected_f1 is None) != (detection["f1"] is None) or (
            expected_f1 is not None
            and not math.isclose(detection["f1"], expected_f1, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise PublicSnapshotError("reference_evaluation.detection.f1 is inconsistent")
        negative = result["negative_control_safety"]
        fully_measured_controls = negative.get("measurement_status") == "measured"
        false_primary = negative.get("false_primary_count")
        false_primary_exports = negative.get("false_primary_export_count")
        _require_gate_bindings(
            result["quality_gates"],
            {
                "candidate_detection_recall": detection["recall"],
                "exact_numeric_value_and_unit": result["derived_accuracy"].get(
                    "exact_numeric_value_and_unit"
                ),
                "joint_system_dataset_metric_value_slice": result["field_accuracy"].get(
                    "joint_semantics"
                ),
                "evidence_page_and_text_support": result["derived_accuracy"].get(
                    "evidence_page_and_text_support"
                ),
                "evidence_table_figure_row_column": result["field_accuracy"].get(
                    "evidence_structure"
                ),
                "honest_missingness": result["field_accuracy"].get("missingness"),
                "claim_type_macro_f1": (
                    result["claim_type_classification"]["macro_f1"]
                    if result["claim_type_classification"]["supported_classes"] >= 2
                    else None
                ),
                "false_primary_controls": (
                    float(false_primary) if false_primary or fully_measured_controls else None
                ),
                "false_primary_exports": (
                    float(false_primary_exports)
                    if false_primary_exports or fully_measured_controls
                    else None
                ),
            },
            "reference_evaluation.quality_gates",
        )
    if "input_observability" in source:
        result["input_observability"] = _project_input_observability(
            source["input_observability"],
            "reference_evaluation.input_observability",
        )
        observed_reference_basis = result["input_observability"].get("reference_observations")
        expected_reference_basis = (
            result.get("reference_observations")
            if schema_version == "reference-score/0.7"
            else result["bases"].get("reference_observations")
        )
        if (
            observed_reference_basis is not None
            and observed_reference_basis != expected_reference_basis
        ):
            raise PublicSnapshotError(
                "reference_evaluation.input_observability.reference_observations "
                "disagrees with reference basis"
            )
        if schema_version == "corpus-reference-score/0.4":
            observability = result["input_observability"]
            required_observability_counts = {
                "papers_measured",
                "papers_not_assessed",
                "measured_reference_observations",
            }
            if "papers_scored" not in result or not required_observability_counts <= set(
                observability
            ):
                raise PublicSnapshotError(
                    "corpus reference input_observability assessment counts are incomplete"
                )
            if (
                observability["papers_measured"] + observability["papers_not_assessed"]
                != result["papers_scored"]
            ):
                raise PublicSnapshotError(
                    "corpus reference input_observability paper denominator disagrees"
                )
            if observability["measured_reference_observations"] > expected_reference_basis:
                raise PublicSnapshotError(
                    "corpus reference input_observability measured reference basis exceeds total"
                )
    elif schema_version == "corpus-reference-score/0.4":
        raise PublicSnapshotError("corpus-reference-score/0.4 requires input_observability")
    return result


def _aggregate_completed_call_telemetry(value: Any, *, context: str) -> dict[str, Any]:
    """Project a legacy stage that persisted only an aggregate completed-call ledger."""

    source = _as_mapping(value, context)
    calls = _nonnegative_int(source.get("calls"), f"{context}.calls")
    cost = _finite_number(source.get("cost_usd_lower_bound"), f"{context}.cost_usd_lower_bound")
    if cost < 0:
        raise PublicSnapshotError(f"{context}.cost_usd_lower_bound must be non-negative")

    totals: dict[str, int] = {}
    reported: dict[str, int] = {}
    for name in ("input", "output", "reasoning", "total"):
        totals[name] = _nonnegative_int(
            source.get(f"{name}_tokens_lower_bound"),
            f"{context}.{name}_tokens_lower_bound",
        )
        reported[name] = _nonnegative_int(
            source.get(f"{name}_tokens_reported_calls"),
            f"{context}.{name}_tokens_reported_calls",
        )
        if reported[name] > calls:
            raise PublicSnapshotError(f"{context}.{name}_tokens_reported_calls exceeds calls")
    cost_reported = _nonnegative_int(
        source.get("cost_reported_calls"), f"{context}.cost_reported_calls"
    )
    if cost_reported > calls:
        raise PublicSnapshotError(f"{context}.cost_reported_calls exceeds calls")
    attempts = _nonnegative_int(
        source.get("attempts_lower_bound"), f"{context}.attempts_lower_bound"
    )
    retries = _nonnegative_int(source.get("retries_lower_bound"), f"{context}.retries_lower_bound")
    if attempts < calls or retries != attempts - calls:
        raise PublicSnapshotError(f"{context} attempt/retry accounting disagrees")
    result = {
        "calls_attempted": calls,
        "calls_field_count": None,
        "resumed_calls_field_count": None,
        "call_accounting_basis": "completed_call_telemetry",
        "model_returned_matches_requested_calls": 0,
        "model_returned_unverified_calls": calls,
        "models_returned": [],
        "providers_returned": [],
        "attempts_lower_bound": attempts,
        "retries_lower_bound": retries,
    }
    result.update(
        {
            "cost_usd": round(cost, 8) if cost_reported == calls else None,
            "cost_usd_lower_bound": round(cost, 8),
            "cost_reported_calls": cost_reported,
            "cost_missing_calls": calls - cost_reported,
        }
    )
    for name in ("input", "output", "reasoning", "total"):
        result[f"{name}_tokens"] = totals[name] if reported[name] == calls else None
        result[f"{name}_tokens_lower_bound"] = totals[name]
        result[f"{name}_tokens_reported_calls"] = reported[name]
        result[f"{name}_tokens_missing_calls"] = calls - reported[name]
    return result


def _typed_aggregated_call(
    item: Mapping[str, Any],
    *,
    stage: Mapping[str, Any],
    context: str,
    expected_request_contract: Mapping[str, Any] | None,
) -> tuple[ProviderCall, dict[str, Any]]:
    """Rehydrate and request-bind one current or explicitly legacy call record."""

    version = item.get("schema_version")
    if version == "public-provider-call/0.1":
        finish_category = item.get("finish_category")
        if finish_category not in {"stop", "length", "error", "other", "missing"}:
            raise PublicSnapshotError(f"{context}.finish_category is unsupported")
        if not isinstance(item.get("request_id_observed"), bool):
            raise PublicSnapshotError(f"{context}.request_id_observed must be a boolean")
        for forbidden in (
            "finish_reason",
            "model_returned_sha256",
            "provider_returned_sha256",
            "request_id_sha256",
        ):
            if forbidden in item:
                raise PublicSnapshotError(f"{context}.{forbidden} is not public telemetry")
        model_disposition = item.get("model_returned_disposition")
        model_returned = item.get("model_returned")
        valid_model_disposition = (
            model_disposition == "matches_requested"
            and model_returned == item.get("model_requested")
        ) or (model_disposition in {"missing", "unrecognized_omitted"} and model_returned is None)
        if not valid_model_disposition:
            raise PublicSnapshotError(f"{context}.model_returned is inconsistent with disposition")
        provider_disposition = item.get("provider_returned_disposition")
        provider_returned = item.get("provider_returned")
        valid_provider_disposition = (
            provider_disposition == "known_label" and provider_returned in PUBLIC_PROVIDER_LABELS
        ) or (
            provider_disposition in {"missing", "unrecognized_omitted"}
            and provider_returned is None
        )
        if not valid_provider_disposition:
            raise PublicSnapshotError(
                f"{context}.provider_returned is inconsistent with disposition"
            )
        typed_payload = {
            "provider": item.get("provider_requested"),
            "model_requested": item.get("model_requested"),
            "model_returned": item.get("model_returned"),
            "provider_returned": item.get("provider_returned"),
            "prompt_sha256": item.get("prompt_sha256"),
            "response_sha256": item.get("response_sha256"),
            "temperature": item.get("temperature"),
            "reasoning_effort": item.get("reasoning_effort"),
            "max_tokens": item.get("max_tokens"),
            "completion_token_parameter": item.get("completion_token_parameter"),
            "seed": item.get("seed"),
            "response_format": item.get("response_format"),
            "schema_name": item.get("schema_name"),
            "schema_sha256": item.get("schema_sha256"),
            "schema_strict": item.get("schema_strict"),
            "data_collection": item.get("data_collection"),
            "require_parameters": item.get("require_parameters"),
            "zdr": item.get("zdr"),
            "latency_seconds": item.get("latency_seconds"),
            "input_tokens": item.get("input_tokens"),
            "output_tokens": item.get("output_tokens"),
            "reasoning_tokens": item.get("reasoning_tokens"),
            "total_tokens": item.get("total_tokens"),
            "cost_usd": item.get("cost_usd"),
            "request_id": None,
            "finish_reason": None if finish_category == "missing" else finish_category,
            "attempts": item.get("attempts"),
        }
        try:
            typed = ProviderCall.model_validate(typed_payload)
        except ValueError as error:
            raise PublicSnapshotError(f"{context} is not a typed ProviderCall") from error
        canonical_public = public_provider_call(typed)
        for key, expected in canonical_public.items():
            if (
                key
                not in {
                    "request_id_observed",
                    "model_returned",
                    "model_returned_disposition",
                    "provider_returned",
                    "provider_returned_disposition",
                }
                and item.get(key) != expected
            ):
                raise PublicSnapshotError(f"{context}.{key} is inconsistent")
        canonical_public.update(
            {
                "model_returned": model_returned,
                "model_returned_disposition": model_disposition,
                "provider_returned": provider_returned,
                "provider_returned_disposition": provider_disposition,
                "request_id_observed": item["request_id_observed"],
            }
        )
    elif version is None:
        try:
            typed = ProviderCall.model_validate(dict(item))
        except ValueError as error:
            raise PublicSnapshotError(
                f"{context} legacy call is not a complete typed ProviderCall"
            ) from error
        canonical_public = public_provider_call(typed)
    else:
        raise PublicSnapshotError(f"{context}.schema_version is unsupported")

    typed_contract = structured_request_contract_from_call(typed)
    if expected_request_contract is not None and typed_contract != dict(expected_request_contract):
        raise PublicSnapshotError(f"{context}.request_contract disagrees with stage contract")
    for field in (
        "model_requested",
        "temperature",
        "reasoning_effort",
        "max_tokens",
        "seed",
        "require_parameters",
    ):
        stage_field = "model" if field == "model_requested" else field
        if stage_field not in stage or getattr(typed, field) != stage[stage_field]:
            raise PublicSnapshotError(f"{context}.{field} disagrees with stage settings")
    stage_prompt = stage.get("prompt_sha256")
    if stage_prompt is not None and (
        not isinstance(stage_prompt, str) or re.fullmatch(r"[0-9a-f]{64}", stage_prompt) is None
    ):
        raise PublicSnapshotError("stage prompt_sha256 is invalid")
    return typed, canonical_public


def _aggregate_private_calls(
    stage: Mapping[str, Any],
    *,
    expected_request_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        "calls" not in stage
        and "resumed_calls" not in stage
        and "completed_call_telemetry" in stage
    ):
        return _aggregate_completed_call_telemetry(
            stage["completed_call_telemetry"], context="completed_call_telemetry"
        )
    calls = stage.get("calls", [])
    resumed_calls = stage.get("resumed_calls", [])
    for field, values in (("calls", calls), ("resumed_calls", resumed_calls)):
        if not isinstance(values, Sequence) or isinstance(values, str | bytes | bytearray):
            raise PublicSnapshotError(f"provider {field} must be a JSON array")
    call_records = [
        (field, index, item)
        for field, values in (("calls", calls), ("resumed_calls", resumed_calls))
        for index, item in enumerate(values)
    ]
    costs: list[float] = []
    token_totals = {name: 0 for name in ("input", "output", "reasoning", "total")}
    token_reported = {name: 0 for name in ("input", "output", "reasoning", "total")}
    returned_models: set[str] = set()
    returned_providers: set[str] = set()
    model_returned_matches_requested = 0
    attempts_lower_bound = 0
    for field, index, item in call_records:
        if not isinstance(item, Mapping):
            raise PublicSnapshotError(f"provider {field}[{index}] must be a JSON object")
        context = f"provider {field}[{index}]"
        typed, safe = _typed_aggregated_call(
            item,
            stage=stage,
            context=context,
            expected_request_contract=expected_request_contract,
        )
        if typed.cost_usd is not None:
            costs.append(typed.cost_usd)
        for name in ("input", "output", "reasoning", "total"):
            token_count = getattr(typed, f"{name}_tokens")
            if token_count is not None:
                token_totals[name] += token_count
                token_reported[name] += 1
        if safe["model_returned"] is not None:
            returned_models.add(safe["model_returned"])
        if safe["model_returned_disposition"] == "matches_requested":
            model_returned_matches_requested += 1
        if safe["provider_returned"] is not None:
            returned_providers.add(safe["provider_returned"])
        attempts_lower_bound += typed.attempts
    total_cost = sum(costs)
    if not math.isfinite(total_cost):
        raise PublicSnapshotError("provider call cost total must be finite")
    call_count = len(call_records)
    result = {
        "calls_attempted": len(call_records),
        "calls_field_count": len(calls),
        "resumed_calls_field_count": len(resumed_calls),
        "call_accounting_basis": "calls_plus_resumed_calls",
        "model_returned_matches_requested_calls": model_returned_matches_requested,
        "model_returned_unverified_calls": call_count - model_returned_matches_requested,
        "models_returned": sorted(returned_models),
        "providers_returned": sorted(returned_providers),
        "attempts_lower_bound": attempts_lower_bound,
        "retries_lower_bound": attempts_lower_bound - call_count,
    }
    result.update(
        {
            "cost_usd": round(total_cost, 8) if len(costs) == call_count else None,
            "cost_usd_lower_bound": round(total_cost, 8),
            "cost_reported_calls": len(costs),
            "cost_missing_calls": call_count - len(costs),
        }
    )
    for name in ("input", "output", "reasoning", "total"):
        result[f"{name}_tokens"] = (
            token_totals[name] if token_reported[name] == call_count else None
        )
        result[f"{name}_tokens_lower_bound"] = token_totals[name]
        result[f"{name}_tokens_reported_calls"] = token_reported[name]
        result[f"{name}_tokens_missing_calls"] = call_count - token_reported[name]
    return result


_STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "extractor": (
        "provider",
        "model",
        "temperature",
        "reasoning_effort",
        "max_tokens",
        "seed",
        "require_parameters",
        "prompt_sha256",
    ),
    "row_enumeration": (
        "enabled",
        "provider",
        "model",
        "temperature",
        "reasoning_effort",
        "max_tokens",
        "seed",
        "require_parameters",
        "prompt_sha256",
    ),
    "tuple_resolution": (
        "enabled",
        "provider",
        "model",
        "temperature",
        "reasoning_effort",
        "max_tokens",
        "seed",
        "require_parameters",
        "prompt_sha256",
        "requires_export_concordant_tuple",
        "mutates_candidate_tuple",
        "allows_origin_or_export",
    ),
    "verifier": (
        "enabled",
        "model",
        "temperature",
        "reasoning_effort",
        "max_tokens",
        "seed",
        "require_parameters",
        "verification_schema_version",
        "grounding_schema_version",
    ),
    "origin_retrieval": (
        "enabled",
        "requires_independent_verifier_accept",
        "model",
        "temperature",
        "reasoning_effort",
        "max_tokens",
        "seed",
        "require_parameters",
        "prompt_sha256",
    ),
}
_STAGE_EXECUTION_FIELDS: dict[str, tuple[str, ...]] = {
    "extractor": (
        "blocks_total",
        "blocks_succeeded",
        "blocks_failed",
        "blocks_resumed",
        "calls_succeeded",
        "calls_failed",
        "calls_resumed",
        "calls_resumed_succeeded",
        "calls_resumed_failed",
        "no_call_failures",
        "requests_rejected",
        "transport_failures",
        "local_failures",
        "blocks_checkpointed",
        "calls_checkpointed",
    ),
    "row_enumeration": (
        "batches_total",
        "batches_resumed",
        "batches_executed",
        "batches_checkpointed",
        "calls_checkpointed",
        "invalid_rows_seen",
        "unknown_row_ids_seen",
    ),
    "tuple_resolution": (
        "candidates_selected",
        "candidates_passed",
        "candidates_routed_to_review",
        "candidates_unsupported",
        "candidates_failed",
        "candidates_mismatched",
        "candidates_resumed",
        "candidates_checkpointed",
        "calls_checkpointed",
    ),
    "verifier": (
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
        "candidates_checkpointed",
        "calls_checkpointed",
    ),
    "origin_retrieval": (
        "candidates_selected",
        "candidates_resumed",
        "candidates_failed",
        "candidates_deterministic_external",
        "candidates_checkpointed",
        "calls_checkpointed",
    ),
}
_STAGE_REQUIRED_EXECUTION_FIELDS: dict[str, tuple[str, ...]] = {
    "extractor": (
        "blocks_total",
        "blocks_succeeded",
        "blocks_failed",
        "blocks_resumed",
        "calls_succeeded",
        "calls_failed",
        "calls_resumed",
        "calls_resumed_succeeded",
        "calls_resumed_failed",
        "no_call_failures",
        "requests_rejected",
        "transport_failures",
        "local_failures",
    ),
    "row_enumeration": (
        "batches_total",
        "batches_resumed",
        "batches_executed",
        "invalid_rows_seen",
        "unknown_row_ids_seen",
    ),
    "tuple_resolution": (
        "candidates_selected",
        "candidates_passed",
        "candidates_routed_to_review",
        "candidates_unsupported",
        "candidates_failed",
        "candidates_mismatched",
        "candidates_resumed",
    ),
    "verifier": (
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
    ),
    "origin_retrieval": (
        "candidates_selected",
        "candidates_resumed",
        "candidates_failed",
        "candidates_deterministic_external",
    ),
}
_ROW_LIMIT_FIELDS = (
    "min_dense_table_rows",
    "max_rows_per_batch",
    "max_value_tokens_per_batch",
    "max_characters_per_batch",
    "max_recovery_depth",
)


def _project_candidate_validation(value: Any) -> dict[str, Any]:
    context = "candidate_validation"
    source = _as_mapping(value or {}, context)
    schema_version = source.get("schema_version")
    # 0.1 predates the origin policy and carries no such key; 0.2 must name it. Both
    # project, so artifacts sealed under 0.1 keep reprojecting byte-identically.
    if schema_version not in {"candidate-validation/0.1", "candidate-validation/0.2"}:
        raise PublicSnapshotError(f"{context}.schema_version is unsupported")
    if "min_confidence" not in source:
        raise PublicSnapshotError(f"{context}.min_confidence is required")
    projected = {
        "schema_version": schema_version,
        "min_confidence": _nullable_fraction(source["min_confidence"], f"{context}.min_confidence"),
    }
    if schema_version == "candidate-validation/0.2":
        policy = source.get("origin_policy")
        if policy not in {"positive_only", "tiered"}:
            raise PublicSnapshotError(f"{context}.origin_policy is unsupported")
        projected["origin_policy"] = policy
    elif "origin_policy" in source:
        raise PublicSnapshotError(f"{context}.origin_policy is not part of this version")
    return projected


def _project_stage_execution(
    value: Any,
    *,
    stage: str,
    run_status: str,
    usage: Mapping[str, Any],
) -> dict[str, int]:
    context = f"{stage}.execution"
    source = _as_mapping(value, context)
    missing = set(_STAGE_REQUIRED_EXECUTION_FIELDS[stage]) - set(source)
    if missing:
        raise PublicSnapshotError(f"{context} is missing required execution counts")
    result = {
        field: _nonnegative_int(source[field], f"{context}.{field}")
        for field in _STAGE_EXECUTION_FIELDS[stage]
        if field in source
    }
    finalized = run_status not in {"bounded_incomplete", "error"}
    if stage == "extractor":
        if result["calls_resumed"] != (
            result["calls_resumed_succeeded"] + result["calls_resumed_failed"]
        ):
            raise PublicSnapshotError(f"{context} resumed-call partition disagrees")
        if result["no_call_failures"] != (
            result["requests_rejected"] + result["transport_failures"] + result["local_failures"]
        ):
            raise PublicSnapshotError(f"{context} no-call failure partition disagrees")
        if finalized:
            if result["blocks_total"] != (
                result["blocks_succeeded"] + result["blocks_resumed"] + result["blocks_failed"]
            ):
                raise PublicSnapshotError(f"{context} block partition disagrees")
            if result["calls_succeeded"] + result["calls_failed"] != usage["calls_field_count"]:
                raise PublicSnapshotError(f"{context} executed-call partition disagrees")
            if result["calls_resumed"] != usage["resumed_calls_field_count"]:
                raise PublicSnapshotError(f"{context} resumed-call count disagrees")
    elif stage == "row_enumeration":
        if finalized and result["batches_total"] != (
            result["batches_resumed"] + result["batches_executed"]
        ):
            raise PublicSnapshotError(f"{context} batch partition disagrees")
    elif stage == "tuple_resolution":
        if finalized and result["candidates_selected"] != (
            result["candidates_passed"] + result["candidates_routed_to_review"]
        ):
            raise PublicSnapshotError(f"{context} candidate outcome partition disagrees")
        for field in (
            "candidates_unsupported",
            "candidates_failed",
            "candidates_mismatched",
        ):
            if result[field] > result["candidates_routed_to_review"]:
                raise PublicSnapshotError(f"{context}.{field} exceeds routed candidates")
        if result["candidates_resumed"] > result["candidates_selected"]:
            raise PublicSnapshotError(f"{context}.candidates_resumed exceeds selected candidates")
    elif stage == "verifier":
        if result["candidates_resumed"] != (
            result["candidates_resumed_succeeded"] + result["candidates_resumed_failed"]
        ):
            raise PublicSnapshotError(f"{context} resumed-candidate partition disagrees")
        if result["candidates_executed"] != (
            result["candidates_executed_succeeded"] + result["candidates_executed_failed"]
        ):
            raise PublicSnapshotError(f"{context} executed-candidate partition disagrees")
        if finalized:
            if result["candidates_selected"] != (
                result["candidates_unbound"]
                + result["candidates_resumed"]
                + result["candidates_executed"]
            ):
                raise PublicSnapshotError(f"{context} selected-candidate partition disagrees")
            if result["candidates_verified"] != (
                result["candidates_resumed_succeeded"] + result["candidates_executed_succeeded"]
            ):
                raise PublicSnapshotError(f"{context} verified-candidate count disagrees")
            if result["candidates_failed"] != (
                result["candidates_resumed_failed"] + result["candidates_executed_failed"]
            ):
                raise PublicSnapshotError(f"{context} failed-candidate count disagrees")
    else:
        for field in (
            "candidates_resumed",
            "candidates_failed",
            "candidates_deterministic_external",
        ):
            if result[field] > result["candidates_selected"]:
                raise PublicSnapshotError(f"{context}.{field} exceeds selected candidates")
    if "calls_checkpointed" in result and result["calls_checkpointed"] != usage["calls_attempted"]:
        raise PublicSnapshotError(f"{context}.calls_checkpointed disagrees with call ledger")
    return result


def _project_stage(value: Any, *, stage: str, run_status: str) -> dict[str, Any]:
    source = _as_mapping(value or {}, stage)
    result = _fields(source, _STAGE_FIELDS[stage], stage)
    if "request_contract" not in source:
        raise PublicSnapshotError(f"{stage}.request_contract is required")
    result["request_contract"] = _project_request_contract(
        source["request_contract"], outer_stage=source
    )
    if stage == "row_enumeration" and "limits" in source:
        limits = _as_mapping(source["limits"], f"{stage}.limits")
        result["limits"] = {
            field: _nonnegative_int(limits[field], f"{stage}.limits.{field}")
            for field in _ROW_LIMIT_FIELDS
            if field in limits
        }
    usage = _aggregate_private_calls(source, expected_request_contract=result["request_contract"])
    if "execution" not in source:
        raise PublicSnapshotError(f"{stage}.execution is required")
    result["execution"] = _project_stage_execution(
        source["execution"], stage=stage, run_status=run_status, usage=usage
    )
    result["usage"] = usage
    return result


def _project_pipeline_counts(value: Any, context: str) -> dict[str, int]:
    source = _as_mapping(value, context)
    missing = set(_COUNT_FIELDS) - set(source)
    if missing:
        raise PublicSnapshotError(f"{context} is missing required pipeline counts")
    return {field: _nonnegative_int(source[field], f"{context}.{field}") for field in _COUNT_FIELDS}


def _validate_paper_count_algebra(counts: Mapping[str, int], context: str) -> None:
    if (
        counts["candidates"] + counts["duplicates_removed"]
        != counts["candidates_before_deduplication"]
    ):
        raise PublicSnapshotError(f"{context} candidate deduplication counts disagree")
    if counts["primary_results"] > counts["candidates"]:
        raise PublicSnapshotError(f"{context}.primary_results exceeds candidates")
    if counts["exported"] > counts["candidates"]:
        raise PublicSnapshotError(f"{context}.exported exceeds candidates")
    if counts["tuple_passed"] + counts["tuple_review"] != counts["tuple_candidates"]:
        raise PublicSnapshotError(f"{context} tuple outcome counts disagree")
    if (
        counts["verifier_accepts"] + counts["verifier_rejects"] + counts["verifier_reviews"]
        != counts["verifications"]
    ):
        raise PublicSnapshotError(f"{context} verifier decision counts disagree")
    if counts["spot_checks_exact"] > counts["spot_checks"]:
        raise PublicSnapshotError(f"{context}.spot_checks_exact exceeds spot_checks")
    if (
        counts["origin_external"] + counts["origin_unresolved"] + counts["origin_no_signal"]
        != counts["origin_candidates"]
    ):
        raise PublicSnapshotError(f"{context} origin outcome counts disagree")
    if counts["origin_positive_review_only"] > counts["origin_unresolved"]:
        raise PublicSnapshotError(
            f"{context}.origin_positive_review_only exceeds unresolved origins"
        )


def _project_paper_run(value: Any, index: int) -> dict[str, Any]:
    source = _as_mapping(value, f"corpus.runs[{index}]")
    if source.get("schema_version") != "pipeline-run/0.4":
        raise PublicSnapshotError(f"corpus.runs[{index}].schema_version is unsupported")
    paper_id = source.get("paper_id")
    if not isinstance(paper_id, str) or not _SAFE_ID.fullmatch(paper_id):
        raise PublicSnapshotError(f"unsafe paper ID in corpus run: {paper_id!r}")
    title = source.get("title", paper_id)
    if not isinstance(title, str):
        raise PublicSnapshotError(f"corpus.runs[{index}].title must be a string")
    status = source.get("status")
    if status not in _PAPER_RUN_STATUSES:
        raise PublicSnapshotError(f"corpus.runs[{index}].status is unsupported")
    counts = _project_pipeline_counts(source.get("counts"), f"corpus.runs[{index}].counts")
    _validate_paper_count_algebra(counts, f"corpus.runs[{index}].counts")
    wall_clock_seconds = _finite_number(
        source.get("wall_clock_seconds"),
        f"corpus.runs[{index}].wall_clock_seconds",
    )
    if wall_clock_seconds < 0:
        raise PublicSnapshotError(f"corpus.runs[{index}].wall_clock_seconds must be non-negative")
    result: dict[str, Any] = {
        "paper_id": paper_id,
        "title": title,
        "status": status,
        "counts": counts,
        "wall_clock_seconds": wall_clock_seconds,
    }
    source_manifest_sha256 = source.get("source_manifest_sha256")
    if (
        not isinstance(source_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256) is None
    ):
        raise PublicSnapshotError(f"corpus.runs[{index}].source_manifest_sha256 is invalid")
    result["source_manifest_sha256"] = source_manifest_sha256
    if "selected_pages" in source:
        pages = _as_sequence(source["selected_pages"], f"corpus.runs[{index}].selected_pages")
        result["selected_pages"] = [
            _nonnegative_int(
                page,
                f"corpus.runs[{index}].selected_pages[{page_index}]",
                positive=True,
            )
            for page_index, page in enumerate(pages)
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
    if "candidate_validation" not in source:
        raise PublicSnapshotError(f"corpus.runs[{index}].candidate_validation is required")
    result["candidate_validation"] = _project_candidate_validation(source["candidate_validation"])
    for stage in (
        "extractor",
        "row_enumeration",
        "tuple_resolution",
        "verifier",
        "origin_retrieval",
    ):
        if stage not in source:
            raise PublicSnapshotError(f"corpus.runs[{index}].{stage} is required")
        result[stage] = _project_stage(source[stage], stage=stage, run_status=status)
    if status not in {"bounded_incomplete", "error"}:
        for stage in (
            "extractor",
            "row_enumeration",
            "tuple_resolution",
            "verifier",
            "origin_retrieval",
        ):
            usage = result[stage]["usage"]
            if usage["model_returned_matches_requested_calls"] != usage["calls_attempted"]:
                raise PublicSnapshotError(
                    f"corpus.runs[{index}].{stage} lacks exact returned-model evidence "
                    "for every completed call"
                )
        execution_bindings = {
            "tuple_candidates": result["tuple_resolution"]["execution"]["candidates_selected"],
            "tuple_passed": result["tuple_resolution"]["execution"]["candidates_passed"],
            "tuple_review": result["tuple_resolution"]["execution"]["candidates_routed_to_review"],
            "tuple_unsupported": result["tuple_resolution"]["execution"]["candidates_unsupported"],
            "tuple_failed": result["tuple_resolution"]["execution"]["candidates_failed"],
            "tuple_resumed": result["tuple_resolution"]["execution"]["candidates_resumed"],
            "verifications": result["verifier"]["execution"]["candidates_verified"],
            "verifier_failed": result["verifier"]["execution"]["candidates_failed"],
            "verifier_resumed": result["verifier"]["execution"]["candidates_resumed"],
            "origin_candidates": result["origin_retrieval"]["execution"]["candidates_selected"],
            "origin_failed": result["origin_retrieval"]["execution"]["candidates_failed"],
            "origin_resumed": result["origin_retrieval"]["execution"]["candidates_resumed"],
        }
        if any(counts[field] != expected for field, expected in execution_bindings.items()):
            raise PublicSnapshotError(f"corpus.runs[{index}] counts disagree with stage execution")
    eee_schema = _as_mapping(source.get("eee_schema"), f"corpus.runs[{index}].eee_schema")
    eee_version = eee_schema.get("version")
    eee_sha256 = eee_schema.get("sha256")
    if not isinstance(eee_version, str) or not eee_version:
        raise PublicSnapshotError(f"corpus.runs[{index}].eee_schema.version is invalid")
    if not isinstance(eee_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", eee_sha256) is None:
        raise PublicSnapshotError(f"corpus.runs[{index}].eee_schema.sha256 is invalid")
    result["eee_schema"] = {"version": eee_version, "sha256": eee_sha256}
    result["code"] = _project_code(source.get("code", {}))
    if source.get("error") and isinstance(source["error"], Mapping):
        result["error"] = _fields(source["error"], ("type", "code"), f"corpus.runs[{index}].error")
    return result


def _paper_directory(root: Path, paper_id: str, context: str) -> Path:
    paper_root = root / paper_id
    if (
        paper_root.parent != root
        or paper_root.is_symlink()
        or not paper_root.is_dir()
        or paper_root.resolve() != paper_root
    ):
        raise PublicSnapshotError(f"{context} must be a direct non-symlink paper directory")
    return paper_root


def _project_corpus_provider_accounting(
    value: Any, runs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    context = "corpus.provider_budget"
    source = _as_mapping(value, context)
    if source.get("schema_version") != "provider-budget-summary/0.2":
        raise PublicSnapshotError(f"{context}.schema_version is unsupported")
    budget_status = source.get("status")
    if budget_status not in {"available", "exhausted", "actual_cost_overrun"}:
        raise PublicSnapshotError(f"{context}.status is unsupported")
    counts = {
        field: _nonnegative_int(source.get(field), f"{context}.{field}")
        for field in (
            "structured_calls_started",
            "structured_calls_completed",
            "structured_calls_pending",
            "provider_call_telemetry_calls",
            "provider_call_telemetry_missing_calls",
            "provider_reported_cost_calls",
        )
    }
    if counts["structured_calls_started"] != (
        counts["structured_calls_completed"] + counts["structured_calls_pending"]
    ):
        raise PublicSnapshotError(f"{context} started/completed call partition disagrees")
    if (
        counts["provider_call_telemetry_calls"] + counts["provider_call_telemetry_missing_calls"]
        != counts["structured_calls_completed"]
    ):
        raise PublicSnapshotError(f"{context} provider-call telemetry partition disagrees")
    if counts["provider_reported_cost_calls"] > counts["provider_call_telemetry_calls"]:
        raise PublicSnapshotError(f"{context}.provider_reported_cost_calls exceeds telemetry")
    reported_cost = _finite_number(
        source.get("provider_reported_cost_usd"),
        f"{context}.provider_reported_cost_usd",
    )
    if reported_cost < 0:
        raise PublicSnapshotError(f"{context}.provider_reported_cost_usd must be non-negative")
    token_totals = {
        name: _nonnegative_int(
            source.get(f"provider_reported_{name}_tokens_lower_bound"),
            f"{context}.provider_reported_{name}_tokens_lower_bound",
        )
        for name in ("input", "output", "reasoning", "total")
    }
    stage_usages = [
        _as_mapping(run[stage]["usage"], f"{run['paper_id']}.{stage}.usage")
        for run in runs
        for stage in (
            "extractor",
            "row_enumeration",
            "tuple_resolution",
            "verifier",
            "origin_retrieval",
        )
    ]
    listed_calls = sum(usage["calls_attempted"] for usage in stage_usages)
    listed_cost_calls = sum(usage["cost_reported_calls"] for usage in stage_usages)
    listed_cost = sum(float(usage["cost_usd_lower_bound"]) for usage in stage_usages)
    listed_tokens = {
        name: sum(int(usage[f"{name}_tokens_lower_bound"]) for usage in stage_usages)
        for name in ("input", "output", "reasoning", "total")
    }
    if listed_calls > counts["provider_call_telemetry_calls"]:
        raise PublicSnapshotError(f"{context} lists more calls than budget telemetry")
    if listed_cost_calls > counts["provider_reported_cost_calls"] or listed_cost > (
        reported_cost + 1e-10
    ):
        raise PublicSnapshotError(f"{context} provider cost is smaller than listed calls")
    if any(listed_tokens[name] > token_totals[name] for name in listed_tokens):
        raise PublicSnapshotError(f"{context} token totals are smaller than listed calls")
    exhaustive = bool(
        counts["structured_calls_pending"] == 0
        and counts["provider_call_telemetry_missing_calls"] == 0
        and listed_calls == counts["structured_calls_completed"]
        and listed_calls == counts["provider_call_telemetry_calls"]
        and listed_cost_calls == counts["provider_reported_cost_calls"]
        and math.isclose(listed_cost, reported_cost, rel_tol=0.0, abs_tol=1e-10)
        and all(listed_tokens[name] == token_totals[name] for name in listed_tokens)
    )
    if not exhaustive:
        for usage in stage_usages:
            usage["cost_usd"] = None
            for name in ("input", "output", "reasoning", "total"):
                usage[f"{name}_tokens"] = None
    return {
        "schema_version": "public-provider-accounting/0.1",
        "budget_status": budget_status,
        "call_coverage": "exhaustive" if exhaustive else "lower_bound",
        **counts,
        "provider_reported_cost_usd": (
            round(reported_cost, 8)
            if counts["structured_calls_pending"] == 0
            and counts["provider_reported_cost_calls"] == counts["structured_calls_completed"]
            else None
        ),
        "provider_reported_cost_usd_lower_bound": round(reported_cost, 8),
        **{
            f"provider_reported_{name}_tokens_lower_bound": total
            for name, total in token_totals.items()
        },
    }


def _bound_reference_score(
    *, root: Path, paper_id: str, run: Mapping[str, Any], index: int
) -> dict[str, Any] | None:
    context = f"corpus.runs[{index}].reference_evaluation"
    paper_root = _paper_directory(root, paper_id, f"corpus.runs[{index}]")
    score_path = paper_root / "reference-score.json"
    reference = run.get("reference_evaluation")
    if reference is None:
        if score_path.exists():
            raise PublicSnapshotError(
                f"{context} is missing despite an existing reference-score.json"
            )
        return None
    source = _as_mapping(reference, context)
    if source.get("score_path") != "reference-score.json":
        raise PublicSnapshotError(f"{context}.score_path is unsupported")
    if not score_path.is_file() or score_path.is_symlink():
        raise PublicSnapshotError(f"missing regular reference-score.json for {paper_id}")
    score_sha256 = source.get("score_sha256")
    if (
        not isinstance(score_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", score_sha256) is None
        or score_sha256 != sha256_file(score_path)
    ):
        raise PublicSnapshotError(f"{context}.score_sha256 disagrees with reference score")
    score = _as_mapping(read_json(score_path), f"{paper_id}.reference-score")
    if score.get("schema_version") != "reference-score/0.7":
        raise PublicSnapshotError(f"{paper_id}.reference-score schema_version is unsupported")
    if score.get("paper_id") != paper_id:
        raise PublicSnapshotError(f"{paper_id}.reference-score paper_id disagrees")
    if source.get("schema_version") != score["schema_version"]:
        raise PublicSnapshotError(f"{context}.schema_version disagrees with reference score")
    _validate_reference_score_details(score, paper_id)
    for field in (
        "coverage",
        "detection",
        "field_accuracy",
        "negative_control_safety",
    ):
        if source.get(field) != score.get(field):
            raise PublicSnapshotError(f"{context}.{field} disagrees with reference score")
    return dict(score)


def _validate_reference_score_details(score: Mapping[str, Any], paper_id: str) -> None:
    """Bind current per-paper score summaries to their detailed match population."""

    context = f"{paper_id}.reference-score"
    matches = list(_as_sequence(score.get("matches"), f"{context}.matches"))
    reference_observations = _nonnegative_int(
        score.get("reference_observations"), f"{context}.reference_observations"
    )
    for field in ("recall_basis", "field_matching_basis"):
        if _nonnegative_int(score.get(field), f"{context}.{field}") != len(matches):
            raise PublicSnapshotError(f"{context}.{field} disagrees with detailed matches")
    if reference_observations != len(matches):
        raise PublicSnapshotError(f"{context}.reference_observations disagrees with matches")

    field_totals = dict.fromkeys(_QUALITY_FIELDS, 0)
    positive_pairs: list[tuple[str, str]] = []
    reference_ids: set[str] = set()
    matched_observation_ids: set[str] = set()
    matched_primary_observation_ids: set[str] = set()
    true_positives = 0
    for index, raw_match in enumerate(matches):
        item_context = f"{context}.matches[{index}]"
        match = _as_mapping(raw_match, item_context)
        reference_id = match.get("reference_id")
        if not isinstance(reference_id, str) or not reference_id:
            raise PublicSnapshotError(f"{item_context}.reference_id is invalid")
        if reference_id in reference_ids:
            raise PublicSnapshotError(f"{context} reuses a reference ID")
        reference_ids.add(reference_id)
        expected_claim_type = match.get("expected_claim_type")
        actual_claim_type = match.get("actual_claim_type")
        if expected_claim_type not in _CLAIM_TYPES or (
            actual_claim_type is not None and actual_claim_type not in _CLAIM_TYPES
        ):
            raise PublicSnapshotError(f"{item_context} has an unsupported claim type")
        observation_id = match.get("observation_id")
        if observation_id is not None:
            if not isinstance(observation_id, str) or not observation_id:
                raise PublicSnapshotError(f"{item_context}.observation_id is invalid")
            if observation_id in matched_observation_ids:
                raise PublicSnapshotError(f"{context} reuses a matched observation ID")
            matched_observation_ids.add(observation_id)
            if actual_claim_type == "primary_result":
                matched_primary_observation_ids.add(observation_id)
            true_positives += 1
        if (observation_id is None) != (actual_claim_type is None):
            raise PublicSnapshotError(
                f"{item_context} observation_id and actual_claim_type disagree"
            )
        if actual_claim_type is not None:
            positive_pairs.append((expected_claim_type, actual_claim_type))
        for field in _QUALITY_FIELDS:
            value = match.get(field)
            if not isinstance(value, bool):
                raise PublicSnapshotError(f"{item_context}.{field} must be a boolean")
            if observation_id is None and value:
                raise PublicSnapshotError(
                    f"{item_context}.{field} cannot match without an observation"
                )
            field_totals[field] += int(value)

    detection = _as_mapping(score.get("detection"), f"{context}.detection")
    if (
        _nonnegative_int(detection.get("true_positives"), f"{context}.detection.true_positives")
        != true_positives
        or _nonnegative_int(
            detection.get("false_negatives"), f"{context}.detection.false_negatives"
        )
        != len(matches) - true_positives
    ):
        raise PublicSnapshotError(f"{context}.detection disagrees with detailed matches")
    observability = _project_input_observability(
        score.get("input_observability"), f"{context}.input_observability"
    )
    if observability.get("reference_observations") != reference_observations:
        raise PublicSnapshotError(
            f"{context}.input_observability reference basis disagrees with matches"
        )
    if observability.get("status") not in {"measured", "not_assessed"}:
        raise PublicSnapshotError(
            f"{context}.input_observability status is unsupported for a paper score"
        )
    if observability.get("status") == "measured":
        conditional = _as_mapping(
            observability.get("model_conditional_detection"),
            f"{context}.input_observability.model_conditional_detection",
        )
        if (
            conditional["true_positives"] > true_positives
            or conditional["false_negatives"] > len(matches) - true_positives
        ):
            raise PublicSnapshotError(
                f"{context}.input_observability detection exceeds detailed outcomes"
            )
    primary_total = _nonnegative_int(
        score.get("primary_candidates_total"), f"{context}.primary_candidates_total"
    )
    candidate_primary_results = _nonnegative_int(
        score.get("candidate_primary_results"), f"{context}.candidate_primary_results"
    )
    primary_in_coverage = _nonnegative_int(
        score.get("primary_candidates_in_coverage"),
        f"{context}.primary_candidates_in_coverage",
    )
    primary_out_of_coverage = _nonnegative_int(
        score.get("primary_candidates_out_of_coverage"),
        f"{context}.primary_candidates_out_of_coverage",
    )
    precision_basis = _nonnegative_int(score.get("precision_basis"), f"{context}.precision_basis")
    unmatched_primary_in = list(
        _as_sequence(
            score.get("unmatched_primary_candidate_ids_in_coverage"),
            f"{context}.unmatched_primary_candidate_ids_in_coverage",
        )
    )
    unmatched_primary_out = list(
        _as_sequence(
            score.get("unmatched_primary_candidate_ids_out_of_coverage"),
            f"{context}.unmatched_primary_candidate_ids_out_of_coverage",
        )
    )
    unmatched_all = list(
        _as_sequence(score.get("unmatched_candidate_ids"), f"{context}.unmatched_candidate_ids")
    )
    unmatched_in = list(
        _as_sequence(
            score.get("unmatched_candidate_ids_in_coverage"),
            f"{context}.unmatched_candidate_ids_in_coverage",
        )
    )
    unmatched_out = list(
        _as_sequence(
            score.get("unmatched_candidate_ids_out_of_coverage"),
            f"{context}.unmatched_candidate_ids_out_of_coverage",
        )
    )
    unmatched_lists = {
        "unmatched_candidate_ids": unmatched_all,
        "unmatched_candidate_ids_in_coverage": unmatched_in,
        "unmatched_candidate_ids_out_of_coverage": unmatched_out,
        "unmatched_primary_candidate_ids_in_coverage": unmatched_primary_in,
        "unmatched_primary_candidate_ids_out_of_coverage": unmatched_primary_out,
    }
    for field, ids in unmatched_lists.items():
        if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
            raise PublicSnapshotError(f"{context}.{field} is invalid")
    unmatched_all_set = set(unmatched_all)
    unmatched_in_set = set(unmatched_in)
    unmatched_out_set = set(unmatched_out)
    unmatched_primary_in_set = set(unmatched_primary_in)
    unmatched_primary_out_set = set(unmatched_primary_out)
    if (
        unmatched_in_set & unmatched_out_set
        or unmatched_all_set != unmatched_in_set | unmatched_out_set
        or not unmatched_primary_in_set <= unmatched_in_set
        or not unmatched_primary_out_set <= unmatched_out_set
    ):
        raise PublicSnapshotError(f"{context} unmatched candidate partitions disagree")
    if (
        unmatched_all_set & matched_observation_ids
        or unmatched_primary_in_set & unmatched_primary_out_set
        or (unmatched_primary_in_set | unmatched_primary_out_set) & matched_observation_ids
    ):
        raise PublicSnapshotError(f"{context} matched and unmatched candidate IDs overlap")
    if (
        candidate_primary_results != primary_total
        or primary_total != primary_in_coverage + primary_out_of_coverage
        or precision_basis != primary_in_coverage
        or primary_in_coverage != len(matched_primary_observation_ids) + len(unmatched_primary_in)
        or primary_out_of_coverage != len(unmatched_primary_out)
        or _nonnegative_int(
            detection.get("false_positives"), f"{context}.detection.false_positives"
        )
        != len(unmatched_primary_in)
        or _nonnegative_int(
            detection.get("precision_true_positives"),
            f"{context}.detection.precision_true_positives",
        )
        != len(matched_primary_observation_ids)
    ):
        raise PublicSnapshotError(f"{context} precision summary disagrees with detailed IDs")
    coverage = _as_mapping(score.get("coverage"), f"{context}.coverage")
    if (
        coverage.get("recall_scope") != "annotated_reference_observations"
        or coverage.get("field_matching_scope") != "annotated_reference_observations"
        or coverage.get("precision_scope") != "fully_annotated_labels"
    ):
        raise PublicSnapshotError(f"{context}.coverage scopes are unsupported")
    field_accuracy = _as_mapping(score.get("field_accuracy"), f"{context}.field_accuracy")
    if set(field_accuracy) != set(_QUALITY_FIELDS):
        raise PublicSnapshotError(f"{context}.field_accuracy is incomplete")
    for field, total in field_totals.items():
        actual = _nullable_fraction(field_accuracy[field], f"{context}.field_accuracy.{field}")
        expected = total / len(matches) if matches else 0.0
        if actual is None or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise PublicSnapshotError(
                f"{context}.field_accuracy.{field} disagrees with detailed matches"
            )

    negative = _as_mapping(
        score.get("negative_control_safety"), f"{context}.negative_control_safety"
    )
    negative_matches = _as_sequence(
        negative.get("matches"), f"{context}.negative_control_safety.matches"
    )
    matched_control_ids: list[str] = []
    matched_candidate_ids: list[str] = []
    false_primary_ids: list[str] = []
    false_primary_export_ids: list[str] = []
    matched_control_candidate_pairs: set[tuple[str, str]] = set()
    for index, raw_match in enumerate(negative_matches):
        match_context = f"{context}.negative_control_safety.matches[{index}]"
        match = _as_mapping(raw_match, match_context)
        control_id = match.get("control_id")
        observation_id = match.get("observation_id")
        if (
            not isinstance(control_id, str)
            or not control_id
            or not isinstance(observation_id, str)
            or not observation_id
        ):
            raise PublicSnapshotError(f"{match_context} has invalid IDs")
        pair = (control_id, observation_id)
        if pair in matched_control_candidate_pairs:
            raise PublicSnapshotError(
                f"{context}.negative_control_safety.matches contains a duplicate pair"
            )
        matched_control_candidate_pairs.add(pair)
        expected_claim_type = match.get("expected_claim_type")
        actual_claim_type = match.get("actual_claim_type")
        if expected_claim_type not in _CLAIM_TYPES or actual_claim_type not in _CLAIM_TYPES:
            raise PublicSnapshotError(f"{match_context} has unsupported claim type")
        export_status = match.get("export_status")
        if export_status not in {"eligible", "needs_review", "not_eligible", "exported"}:
            raise PublicSnapshotError(f"{match_context}.export_status is unsupported")
        expected_claim_match = expected_claim_type == actual_claim_type
        expected_false_primary = actual_claim_type == "primary_result"
        expected_false_primary_export = expected_false_primary and export_status in {
            "eligible",
            "exported",
        }
        for field, expected in (
            ("claim_type_matches", expected_claim_match),
            ("false_primary", expected_false_primary),
            ("false_primary_export", expected_false_primary_export),
        ):
            if match.get(field) is not expected:
                raise PublicSnapshotError(f"{match_context}.{field} disagrees with match")
        matched_evidence_ids = list(
            _as_sequence(match.get("matched_evidence_ids"), f"{match_context}.matched_evidence_ids")
        )
        if (
            not matched_evidence_ids
            or any(not isinstance(item, str) or not item for item in matched_evidence_ids)
            or len(set(matched_evidence_ids)) != len(matched_evidence_ids)
        ):
            raise PublicSnapshotError(f"{match_context}.matched_evidence_ids is invalid")
        if control_id not in matched_control_ids:
            matched_control_ids.append(control_id)
        if observation_id not in matched_candidate_ids:
            matched_candidate_ids.append(observation_id)
        if expected_false_primary and observation_id not in false_primary_ids:
            false_primary_ids.append(observation_id)
        if expected_false_primary_export and observation_id not in false_primary_export_ids:
            false_primary_export_ids.append(observation_id)
        positive_pairs.append((expected_claim_type, actual_claim_type))

    def exact_id_summary(field: str, expected: list[str]) -> None:
        actual = list(
            _as_sequence(negative.get(field), f"{context}.negative_control_safety.{field}")
        )
        if actual != expected:
            raise PublicSnapshotError(
                f"{context}.negative_control_safety.{field} disagrees with detailed matches"
            )

    for field, expected in (
        ("matched_control_ids", matched_control_ids),
        ("matched_candidate_ids", matched_candidate_ids),
        ("false_primary_candidate_ids", false_primary_ids),
        ("false_primary_export_candidate_ids", false_primary_export_ids),
    ):
        exact_id_summary(field, expected)
    for field, expected in (
        ("matched_control_count", len(matched_control_ids)),
        ("matched_candidate_count", len(matched_candidate_ids)),
        ("false_primary_count", len(false_primary_ids)),
        ("false_primary_export_count", len(false_primary_export_ids)),
        ("false_primary_rate_basis", len(matched_candidate_ids)),
    ):
        if (
            _nonnegative_int(negative.get(field), f"{context}.negative_control_safety.{field}")
            != expected
        ):
            raise PublicSnapshotError(
                f"{context}.negative_control_safety.{field} disagrees with detailed matches"
            )
    false_primary_rate_defined = negative.get("false_primary_rate_defined")
    if false_primary_rate_defined is not bool(matched_candidate_ids):
        raise PublicSnapshotError(
            f"{context}.negative_control_safety.false_primary_rate_defined disagrees"
        )
    false_primary_rate = _nullable_fraction(
        negative.get("false_primary_rate"),
        f"{context}.negative_control_safety.false_primary_rate",
    )
    expected_false_primary_rate = (
        len(false_primary_ids) / len(matched_candidate_ids) if matched_candidate_ids else 0.0
    )
    if false_primary_rate is None or not math.isclose(
        false_primary_rate, expected_false_primary_rate, rel_tol=0.0, abs_tol=1e-9
    ):
        raise PublicSnapshotError(
            f"{context}.negative_control_safety.false_primary_rate disagrees with matches"
        )
    controls_total = _nonnegative_int(
        negative.get("controls_total"), f"{context}.negative_control_safety.controls_total"
    )
    negative_context = f"{context}.negative_control_safety"

    def control_id_list(field: str) -> list[str]:
        values = list(_as_sequence(negative.get(field), f"{negative_context}.{field}"))
        if any(not isinstance(item, str) or not item for item in values) or len(set(values)) != len(
            values
        ):
            raise PublicSnapshotError(f"{negative_context}.{field} is invalid")
        return values

    control_ids = control_id_list("control_ids")
    unmatched_control_ids = control_id_list("unmatched_control_ids")
    examined_ids = control_id_list("examined_control_ids")
    not_examined_ids = control_id_list("not_examined_control_ids")
    control_id_set = set(control_ids)
    matched_control_id_set = set(matched_control_ids)
    examined_id_set = set(examined_ids)
    not_examined_id_set = set(not_examined_ids)
    if len(control_ids) != controls_total:
        raise PublicSnapshotError(f"{negative_context}.control_ids disagrees with controls_total")
    if (
        not matched_control_id_set <= control_id_set
        or set(unmatched_control_ids) != control_id_set - matched_control_id_set
        or examined_id_set - control_id_set
        or not_examined_id_set - control_id_set
        or not matched_control_id_set <= examined_id_set
        or examined_id_set & not_examined_id_set
    ):
        raise PublicSnapshotError(f"{negative_context} control ID partitions disagree")

    control_status = _as_mapping(
        negative.get("control_status"), f"{negative_context}.control_status"
    )
    if set(control_status) != control_id_set:
        raise PublicSnapshotError(f"{negative_context}.control_status has the wrong control IDs")
    supported_statuses = {
        "matched",
        "passed_by_abstention",
        "not_examined",
        "examination_unknown",
    }
    if any(status not in supported_statuses for status in control_status.values()):
        raise PublicSnapshotError(f"{negative_context}.control_status is invalid")
    expected_matched = {
        control_id for control_id, status in control_status.items() if status == "matched"
    }
    expected_examined = {
        control_id
        for control_id, status in control_status.items()
        if status in {"matched", "passed_by_abstention"}
    }
    expected_not_examined = {
        control_id for control_id, status in control_status.items() if status == "not_examined"
    }
    if (
        expected_matched != matched_control_id_set
        or expected_examined != examined_id_set
        or expected_not_examined != not_examined_id_set
    ):
        raise PublicSnapshotError(f"{negative_context}.control_status disagrees with ID lists")

    controls_examined = len(examined_ids)
    controls_fully_examined = controls_total > 0 and controls_examined == controls_total
    expected_measurement_status = (
        "measured"
        if controls_fully_examined
        else "partially_measured"
        if controls_examined
        else "not_measured"
    )
    if negative.get("measurement_status") != expected_measurement_status:
        raise PublicSnapshotError(f"{negative_context}.measurement_status disagrees with IDs")
    expected_abstentions = sum(
        status == "passed_by_abstention" for status in control_status.values()
    )
    if (
        _nonnegative_int(
            negative.get("passed_by_abstention_count"),
            f"{negative_context}.passed_by_abstention_count",
        )
        != expected_abstentions
    ):
        raise PublicSnapshotError(
            f"{negative_context}.passed_by_abstention_count disagrees with control_status"
        )
    control_match_coverage = _nullable_fraction(
        negative.get("control_match_coverage"), f"{negative_context}.control_match_coverage"
    )
    control_examination_coverage = _nullable_fraction(
        negative.get("control_examination_coverage"),
        f"{negative_context}.control_examination_coverage",
    )
    if not _fraction_agrees(control_match_coverage, len(matched_control_ids), controls_total):
        raise PublicSnapshotError(f"{negative_context}.control_match_coverage disagrees with IDs")
    if not _fraction_agrees(control_examination_coverage, controls_examined, controls_total):
        raise PublicSnapshotError(
            f"{negative_context}.control_examination_coverage disagrees with IDs"
        )
    if negative.get("control_match_coverage_defined") is not (controls_total > 0):
        raise PublicSnapshotError(
            f"{negative_context}.control_match_coverage_defined disagrees with controls_total"
        )
    for field, ids in (
        ("zero_false_primary_gate_passed", false_primary_ids),
        ("zero_false_primary_export_gate_passed", false_primary_export_ids),
    ):
        expected_gate = False if ids else True if controls_fully_examined else None
        if negative.get(field) is not expected_gate:
            raise PublicSnapshotError(
                f"{context}.negative_control_safety.{field} disagrees with detailed matches"
            )
    expected_classification = score_claim_type_pairs(positive_pairs)
    if score.get("claim_type_classification") != expected_classification:
        raise PublicSnapshotError(
            f"{context}.claim_type_classification disagrees with detailed matches"
        )


def _project_corpus(root: Path) -> dict[str, Any]:
    run_path = root / "corpus-run.json"
    if not run_path.is_file() or run_path.is_symlink():
        raise PublicSnapshotError(f"missing regular corpus-run.json under {root}")
    raw = _as_mapping(read_json(run_path), f"{root.name}.corpus-run")
    if raw.get("schema_version") != "corpus-run/0.3":
        raise PublicSnapshotError(f"{root.name}.corpus-run schema_version is unsupported")
    embedded_runs = list(_as_sequence(raw.get("runs", []), "corpus.runs"))
    embedded_ids: list[str] = []
    for index, value in enumerate(embedded_runs):
        embedded = _as_mapping(value, f"corpus.runs[{index}]")
        paper_id = embedded.get("paper_id")
        if not isinstance(paper_id, str) or not _SAFE_ID.fullmatch(paper_id):
            raise PublicSnapshotError(f"corpus.runs[{index}].paper_id is unsafe")
        if paper_id in embedded_ids:
            raise PublicSnapshotError("corpus run contains duplicate paper IDs")
        embedded_ids.append(paper_id)
        paper_root = _paper_directory(root, paper_id, f"corpus.runs[{index}]")
        paper_run_path = paper_root / "run.json"
        if not paper_run_path.is_file() or paper_run_path.is_symlink():
            raise PublicSnapshotError(f"missing regular run.json for {paper_id}")
        if read_json(paper_run_path) != dict(embedded):
            raise PublicSnapshotError(f"embedded corpus run disagrees with {paper_id}/run.json")
        manifest_path = paper_root / "source-manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise PublicSnapshotError(f"missing regular source manifest for {paper_id}")
        if embedded.get("source_manifest_sha256") != sha256_file(manifest_path):
            raise PublicSnapshotError(
                f"source_manifest_sha256 disagrees with source manifest for {paper_id}"
            )
    source_ids = {
        path.parent.name
        for path in root.glob("*/source-manifest.json")
        if path.is_file() and not path.is_symlink()
    }
    if set(embedded_ids) != source_ids:
        raise PublicSnapshotError("corpus runs and source manifests have different paper-ID sets")
    runs = [_project_paper_run(item, index) for index, item in enumerate(embedded_runs)]
    reference_scores: list[dict[str, Any]] = []
    for index, (embedded, run) in enumerate(zip(embedded_runs, runs, strict=True)):
        score = _bound_reference_score(
            root=root,
            paper_id=run["paper_id"],
            run=_as_mapping(embedded, f"corpus.runs[{index}]"),
            index=index,
        )
        if score is not None:
            reference_scores.append(score)
            projected_score = _project_reference_evaluation(score)
            assert projected_score is not None
            run["reference_evaluation"] = {
                **projected_score,
                "source_artifact_sha256": sha256_file(
                    root / run["paper_id"] / "reference-score.json"
                ),
            }
            expected_reference_counts = {
                "reference_observations": score.get("reference_observations"),
                "reference_true_positives": _as_mapping(
                    score.get("detection"), f"{run['paper_id']}.reference-score.detection"
                ).get("true_positives"),
                "reference_false_positives": _as_mapping(
                    score.get("detection"), f"{run['paper_id']}.reference-score.detection"
                ).get("false_positives"),
                "reference_false_negatives": _as_mapping(
                    score.get("detection"), f"{run['paper_id']}.reference-score.detection"
                ).get("false_negatives"),
                "negative_control_false_primary": _as_mapping(
                    score.get("negative_control_safety", {}),
                    f"{run['paper_id']}.reference-score.negative_control_safety",
                ).get("false_primary_count", 0),
            }
            if any(
                run["counts"][field]
                != _nonnegative_int(value, f"{run['paper_id']}.reference-score.{field}")
                for field, value in expected_reference_counts.items()
            ):
                raise PublicSnapshotError(
                    f"{run['paper_id']} reference-score counts disagree with paper run"
                )
        elif any(
            run["counts"][field]
            for field in (
                "reference_observations",
                "reference_true_positives",
                "reference_false_positives",
                "reference_false_negatives",
                "negative_control_false_primary",
            )
        ):
            raise PublicSnapshotError(
                f"{run['paper_id']} has reference counts without a bound reference score"
            )
    corpus_id = raw.get("corpus_id")
    if not isinstance(corpus_id, str) or not _SAFE_ID.fullmatch(corpus_id):
        raise PublicSnapshotError("corpus.corpus_id is unsafe")
    status = raw.get("status")
    if status not in _CORPUS_RUN_STATUSES:
        raise PublicSnapshotError("corpus.status is unsupported")
    partitions = {
        field: _nonnegative_int(raw.get(field), f"corpus.{field}")
        for field in _CORPUS_PARTITION_FIELDS
    }
    if partitions["papers"] != len(runs):
        raise PublicSnapshotError("corpus paper count disagrees with embedded runs")
    if partitions["papers_total"] < partitions["papers"]:
        raise PublicSnapshotError("corpus.papers_total is smaller than papers")
    if partitions["papers_total"] - partitions["papers"] != partitions["papers_not_started"]:
        raise PublicSnapshotError("corpus started/not-started paper partition disagrees")

    successful_runs = [run for run in runs if run["status"] == "success"]
    failed_runs = [run for run in runs if run["status"] != "success"]
    expected_partitions = {
        "papers_succeeded": len(successful_runs),
        "papers_failed": len(failed_runs),
        "papers_bounded_incomplete": sum(run["status"] == "bounded_incomplete" for run in runs),
        "papers_with_eee": sum(run["counts"]["eee_records"] > 0 for run in successful_runs),
        "papers_without_candidates": sum(
            run["counts"]["candidates"] == 0 for run in successful_runs
        ),
        "papers_without_eee": sum(run["counts"]["eee_records"] == 0 for run in successful_runs),
    }
    for field, expected in expected_partitions.items():
        if partitions[field] != expected:
            raise PublicSnapshotError(f"corpus.{field} disagrees with paper runs")
    if partitions["papers_succeeded"] + partitions["papers_failed"] != partitions["papers"]:
        raise PublicSnapshotError("corpus succeeded/failed paper partition disagrees")
    if (
        partitions["papers_with_eee"] + partitions["papers_without_eee"]
        != partitions["papers_succeeded"]
    ):
        raise PublicSnapshotError("corpus successful-paper EEE partition disagrees")

    review_states = []
    for index, embedded in enumerate(embedded_runs):
        review_state = _as_mapping(
            _as_mapping(embedded, f"corpus.runs[{index}]").get("review_state"),
            f"corpus.runs[{index}].review_state",
        )
        review_status = review_state.get("status")
        if review_status not in {"ready", "needs_review", "blocked"}:
            raise PublicSnapshotError(f"corpus.runs[{index}].review_state.status is unsupported")
        review_states.append(review_status)
    if partitions["papers_needing_review"] != sum(
        review_status != "ready" for review_status in review_states
    ):
        raise PublicSnapshotError("corpus.papers_needing_review disagrees with paper runs")

    bounded = bool(partitions["papers_not_started"] or partitions["papers_bounded_incomplete"])
    expected_status = (
        "bounded_incomplete"
        if bounded
        else "success"
        if not failed_runs
        else "error"
        if not successful_runs
        else "partial_failure"
    )
    if status != expected_status:
        raise PublicSnapshotError("corpus.status disagrees with paper-run partition")

    totals = _project_pipeline_counts(raw.get("totals"), "corpus.totals")
    for field in _COUNT_FIELDS:
        if totals[field] != sum(run["counts"][field] for run in runs):
            raise PublicSnapshotError(f"corpus.totals.{field} disagrees with paper runs")

    for index, run in enumerate(runs):
        eee_dir = root / run["paper_id"] / "eee"
        if eee_dir.exists() and (not eee_dir.is_dir() or eee_dir.is_symlink()):
            raise PublicSnapshotError(f"EEE input must be a regular directory: {eee_dir}")
        eee_paths = list(eee_dir.glob("*.json")) if eee_dir.is_dir() else []
        if any(path.is_symlink() or not path.is_file() for path in eee_paths):
            raise PublicSnapshotError(
                f"EEE input must contain regular files for corpus.runs[{index}]"
            )
        if len(eee_paths) != run["counts"]["eee_records"]:
            raise PublicSnapshotError(
                f"corpus.runs[{index}].counts.eee_records disagrees with EEE files"
            )
    provider_accounting = _project_corpus_provider_accounting(raw.get("provider_budget"), runs)

    evaluation_value = raw.get("reference_evaluation")
    evaluation_path = root / "corpus-evaluation.json"
    evaluation_hash: str | None = None
    if evaluation_path.exists() and (not evaluation_path.is_file() or evaluation_path.is_symlink()):
        raise PublicSnapshotError("corpus-evaluation.json must be a regular file")
    if evaluation_path.is_file():
        evaluation_hash = sha256_file(evaluation_path)
        file_evaluation = read_json(evaluation_path)
        if evaluation_value is not None and evaluation_value != file_evaluation:
            raise PublicSnapshotError(
                "embedded reference_evaluation disagrees with corpus-evaluation.json"
            )
        evaluation_value = file_evaluation
    if reference_scores:
        if not evaluation_path.is_file():
            raise PublicSnapshotError(
                "corpus-evaluation.json is required for bound reference scores"
            )
        try:
            expected_evaluation = aggregate_reference_scores(reference_scores)
        except (KeyError, TypeError, ValueError) as error:
            raise PublicSnapshotError("reference scores cannot be aggregated safely") from error
        if evaluation_value != expected_evaluation:
            raise PublicSnapshotError(
                "corpus reference_evaluation disagrees with bound reference scores"
            )
    elif evaluation_value is not None or evaluation_path.exists():
        raise PublicSnapshotError(
            "corpus reference_evaluation has no bound per-paper reference scores"
        )
    result = {
        "schema_version": raw["schema_version"],
        "corpus_id": corpus_id,
        "status": status,
        "generated_at": _safe_scalar(raw.get("generated_at"), "corpus.generated_at"),
        **partitions,
        "source_artifacts": {
            "corpus_run_sha256": sha256_file(run_path),
            "corpus_evaluation_sha256": evaluation_hash,
        },
        "totals": totals,
        "provider_accounting": provider_accounting,
        "reference_evaluation": _project_reference_evaluation(evaluation_value),
        "papers_detail": runs,
    }
    return result


def _public_report_input(corpus: Mapping[str, Any]) -> dict[str, Any]:
    report_runs = []
    paid_stages = (
        "extractor",
        "row_enumeration",
        "tuple_resolution",
        "verifier",
        "origin_retrieval",
    )
    for paper in _as_sequence(corpus.get("papers_detail", []), "public corpus papers"):
        item = _as_mapping(paper, "public corpus paper")
        stage_usages = {
            stage: _as_mapping(
                _as_mapping(item.get(stage, {}), f"public {stage}").get("usage", {}),
                f"public {stage} usage",
            )
            for stage in paid_stages
        }
        cost_lower_bound = round(
            sum(float(usage.get("cost_usd_lower_bound", 0.0)) for usage in stage_usages.values()),
            8,
        )
        exact_costs = [usage.get("cost_usd") for usage in stage_usages.values()]
        report_run = {
            "paper_id": item["paper_id"],
            "title": item["title"],
            "status": item["status"],
            "counts": item["counts"],
            "cost_usd": (
                round(sum(float(cost) for cost in exact_costs), 8)
                if all(cost is not None for cost in exact_costs)
                else None
            ),
            "cost_usd_lower_bound": cost_lower_bound,
            "cost_accounting": (
                "complete" if all(cost is not None for cost in exact_costs) else "lower_bound"
            ),
            "wall_clock_seconds": item["wall_clock_seconds"],
        }
        for stage, usage in stage_usages.items():
            report_run[stage] = {
                "completed_call_telemetry": {
                    "calls": usage.get("calls_attempted", 0),
                    "total_tokens_lower_bound": usage.get("total_tokens_lower_bound", 0),
                    "retries_lower_bound": usage.get("retries_lower_bound", 0),
                }
            }
        report_runs.append(report_run)
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
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
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
            paper_root = _paper_directory(
                root, manifest.paper_id, f"source manifest {manifest.paper_id}"
            )
            if manifest_path.parent != paper_root:
                raise PublicSnapshotError(
                    f"source manifest is outside its paper directory: {manifest_path}"
                )
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
                    "proceedings_url": _validate_public_uri(
                        str(manifest.proceedings_url) if manifest.proceedings_url else None,
                        f"{manifest.paper_id}.proceedings_url",
                    ),
                    "source_manifest_sha256": sha256_file(manifest_path),
                    "sources": paper_sources,
                }
            )
    return {"schema_version": SOURCES_SCHEMA_VERSION, "papers": papers}, paper_ids


def _project_reference_audits(
    run_roots: Sequence[Path], corpora: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(run_roots) != len(corpora):
        raise PublicSnapshotError("reference-audit corpus roots are inconsistent")
    projected_corpora = []
    for root, corpus in zip(run_roots, corpora, strict=True):
        path = root / "reference-audit.json"
        if not path.is_file() or path.is_symlink():
            raise PublicSnapshotError(f"missing regular reference-audit.json under {root}")
        context = f"{root.name}.reference-audit"
        raw = _as_mapping(read_json(path), context)
        if raw.get("schema_version") != "corpus-reference-audit/0.1":
            raise PublicSnapshotError(f"{context}.schema_version is unsupported")
        if raw.get("corpus_id") != corpus.get("corpus_id"):
            raise PublicSnapshotError(f"{context}.corpus_id disagrees with corpus run")
        expected_ids = {
            paper["paper_id"]
            for paper in _as_sequence(corpus.get("papers_detail"), f"{context}.papers")
        }
        scored_ids = {
            paper["paper_id"]
            for paper in _as_sequence(corpus.get("papers_detail"), f"{context}.papers")
            if paper.get("reference_evaluation") is not None
        }
        results = []
        result_ids: set[str] = set()
        for index, value in enumerate(_as_sequence(raw.get("results"), f"{context}.results")):
            item_context = f"{context}.results[{index}]"
            item = _as_mapping(value, item_context)
            paper_id = item.get("paper_id")
            if not isinstance(paper_id, str) or not _SAFE_ID.fullmatch(paper_id):
                raise PublicSnapshotError(f"{item_context}.paper_id is unsafe")
            if paper_id in result_ids:
                raise PublicSnapshotError(f"{context} contains duplicate paper IDs")
            result_ids.add(paper_id)
            status = item.get("status")
            if status not in {"passed", "failed", "error", "skipped_no_reference"}:
                raise PublicSnapshotError(f"{item_context}.status is unsupported")
            if (paper_id in scored_ids) != (status == "passed"):
                raise PublicSnapshotError(
                    f"{item_context}.status disagrees with bound reference-score presence"
                )
            result: dict[str, Any] = {"paper_id": paper_id, "status": status}
            if status in {"passed", "failed"}:
                source_hash_matches = item.get("source_hash_matches")
                if not isinstance(source_hash_matches, bool):
                    raise PublicSnapshotError(
                        f"{item_context}.source_hash_matches must be a boolean"
                    )
                result.update(
                    {
                        "page_count": _nonnegative_int(
                            item.get("page_count"),
                            f"{item_context}.page_count",
                            positive=True,
                        ),
                        "source_hash_matches": source_hash_matches,
                        "text_verified": _nonnegative_int(
                            item.get("text_verified"), f"{item_context}.text_verified"
                        ),
                        "visual_verified": _nonnegative_int(
                            item.get("visual_verified"), f"{item_context}.visual_verified"
                        ),
                        "failed_evidence": _nonnegative_int(
                            item.get("failed_evidence"), f"{item_context}.failed_evidence"
                        ),
                    }
                )
                expected_status = (
                    "passed" if source_hash_matches and result["failed_evidence"] == 0 else "failed"
                )
                if status != expected_status:
                    raise PublicSnapshotError(f"{item_context}.status disagrees with evidence")
            results.append(result)
        if result_ids != expected_ids:
            raise PublicSnapshotError(f"{context} paper population disagrees with corpus")
        aggregate_counts = {
            field: _nonnegative_int(raw.get(field), f"{context}.{field}")
            for field in (
                "papers",
                "papers_passed",
                "papers_failed",
                "papers_skipped",
                "text_verified",
                "visual_verified",
                "failed_evidence",
            )
        }
        expected_counts = {
            "papers": len(results),
            "papers_passed": sum(item["status"] == "passed" for item in results),
            "papers_failed": sum(item["status"] in {"failed", "error"} for item in results),
            "papers_skipped": sum(item["status"] == "skipped_no_reference" for item in results),
            "text_verified": sum(int(item.get("text_verified", 0)) for item in results),
            "visual_verified": sum(int(item.get("visual_verified", 0)) for item in results),
            "failed_evidence": sum(int(item.get("failed_evidence", 0)) for item in results),
        }
        if aggregate_counts != expected_counts:
            raise PublicSnapshotError(f"{context} aggregate counts disagree with results")
        status = raw.get("status")
        expected_status = "passed" if aggregate_counts["papers_failed"] == 0 else "failed"
        if status != expected_status:
            raise PublicSnapshotError(f"{context}.status disagrees with results")
        projected_corpora.append(
            {
                "schema_version": raw["schema_version"],
                "corpus_id": raw["corpus_id"],
                "status": status,
                **aggregate_counts,
                "source_artifact_sha256": sha256_file(path),
                "results": results,
            }
        )
    return {
        "schema_version": REFERENCE_AUDIT_SCHEMA_VERSION,
        "corpora": projected_corpora,
    }


def _manifest_evidence_sources(
    root: Path,
    paper_id: str,
) -> dict[str, tuple[str, str]]:
    manifest_path = (
        _paper_directory(root, paper_id, f"EEE paper {paper_id}") / "source-manifest.json"
    )
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


def _composer_object(
    value: Any,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> Mapping[str, Any]:
    """Require one exact nested object shape emitted by ``compose_eee_records``."""

    source = _as_mapping(value, context)
    keys = set(source)
    missing = required - keys
    extra = keys - required - optional
    if missing or extra:
        raise PublicSnapshotError(
            f"{context} has unsupported composer fields "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    return source


def _composer_string_map(value: Any, *, allowed: set[str], context: str) -> dict[str, str]:
    source = _as_mapping(value, context)
    if set(source) - allowed or any(not isinstance(item, str) for item in source.values()):
        raise PublicSnapshotError(f"{context} has unsupported composer fields")
    return {key: source[key] for key in sorted(source)}


def _composer_source_provenance(value: Any, *, paper_id: str, context: str) -> dict[str, str]:
    source = _as_mapping(value, context)
    mode = source.get("export_provenance_mode")
    modes = {
        "tuple_gated_production",
        "tuple_audited_human_reviewed",
        "legacy_manual",
        "legacy_human_reviewed",
    }
    if mode not in modes:
        raise PublicSnapshotError(f"{context}.export_provenance_mode is unsupported")
    required = {
        "paper_id",
        "export_provenance_mode",
        "export_composition_sha256",
        "verifier_gate_required",
    }
    # Tier labels the composer writes on every record. Optional, because records
    # composed before the tiered policy existed do not carry them, and those records
    # must keep projecting byte-identically.
    optional = {
        "doi",
        "arxiv_id",
        "review_tier",
        "producer_origin_basis",
        "origin_export_policy",
        "pipeline_git_commit",
    }
    tuple_audited = mode in {"tuple_gated_production", "tuple_audited_human_reviewed"}
    human_reviewed = mode in {"tuple_audited_human_reviewed", "legacy_human_reviewed"}
    if tuple_audited:
        required.add("tuple_sidecar_sha256")
    else:
        required.add("legacy_provenance_reason")
    if human_reviewed:
        required.add("review_manifest_sha256")
    if mode in {"tuple_gated_production", "tuple_audited_human_reviewed"} and (
        mode == "tuple_gated_production" or "verifier_sidecar_sha256" in source
    ):
        required.add("verifier_sidecar_sha256")
    projected = _composer_object(
        source,
        required=required,
        optional=optional,
        context=context,
    )
    if any(not isinstance(item, str) for item in projected.values()):
        raise PublicSnapshotError(f"{context} values must be strings")
    if projected["paper_id"] != paper_id:
        raise PublicSnapshotError(f"{context}.paper_id disagrees with its paper")
    for field in (
        "export_composition_sha256",
        "tuple_sidecar_sha256",
        "review_manifest_sha256",
        "verifier_sidecar_sha256",
    ):
        if field in projected and re.fullmatch(r"[a-f0-9]{64}", projected[field]) is None:
            raise PublicSnapshotError(f"{context}.{field} is invalid")
    expected_verifier_required = "true" if mode == "tuple_gated_production" else "false"
    if projected["verifier_gate_required"] != expected_verifier_required:
        raise PublicSnapshotError(f"{context}.verifier_gate_required disagrees with mode")
    return {key: projected[key] for key in sorted(projected)}


def _project_composer_score_details(
    value: Any,
    *,
    paper_id: str,
    observation_id: str,
    source_provenance: Mapping[str, str],
    context: str,
) -> dict[str, str]:
    source = _as_mapping(value, context)
    if any(not isinstance(item, str) for item in source.values()):
        raise PublicSnapshotError(f"{context} values must be strings")
    anchor_count = source.get("evidence_anchor_count")
    if not isinstance(anchor_count, str) or re.fullmatch(r"[1-9][0-9]*", anchor_count) is None:
        raise PublicSnapshotError(f"{context}.evidence_anchor_count is invalid")
    required = {
        "raw_reported_value",
        "value_comparator",
        "candidate_observation_id",
        "paper_id",
        "evidence_anchor_count",
        "export_provenance_mode",
        "export_composition_sha256",
        "export_candidate_sha256",
        "verifier_gate_required",
    }
    allowed = set(required) | {
        "proposal_ids",
        "candidate_occurrence_ids",
        # Tier labels, optional for the same reason as on the record above.
        "producer_origin_basis",
        "attribution_state",
        "attribution_rule_id",
        "review_tier",
        "referential_status",
        "field_provenance_gate",
        "text_support",
        "evidence_page",
        "origin_export_policy",
    }
    mode = source_provenance["export_provenance_mode"]
    if "tuple_sidecar_sha256" in source_provenance:
        required.update({"tuple_gate_sha256", "tuple_sidecar_sha256"})
        allowed.update({"tuple_gate_sha256", "tuple_sidecar_sha256"})
    else:
        required.add("legacy_provenance_reason")
        allowed.add("legacy_provenance_reason")
    if "review_manifest_sha256" in source_provenance:
        required.add("review_manifest_sha256")
        allowed.add("review_manifest_sha256")
    if "verifier_sidecar_sha256" in source_provenance:
        required.update({"source_verifier_gate_passed", "verifier_sidecar_sha256"})
        allowed.update(
            {"source_verifier_gate_passed", "verifier_sidecar_sha256", "verifier_gate_sha256"}
        )
        if mode == "tuple_gated_production":
            required.add("verifier_gate_sha256")
    for index in range(1, int(anchor_count) + 1):
        prefix = f"evidence_{index}"
        anchor_required = {
            f"{prefix}_source_id",
            f"{prefix}_source_role",
            f"{prefix}_page",
            f"{prefix}_kind",
            f"{prefix}_quote_sha256",
        }
        required.update(anchor_required)
        allowed.update(
            anchor_required
            | {
                f"{prefix}_source_sha256",
                f"{prefix}_source_git_commit",
                f"{prefix}_label",
                f"{prefix}_row",
                f"{prefix}_column",
                f"{prefix}_region_id",
                f"{prefix}_planned_row_id",
                f"{prefix}_cell_id",
                f"{prefix}_numeric_token_id",
                f"{prefix}_header_ids",
            }
        )
    field_names = {"system", "dataset_scope", "metric", "setting", "value", "unit"}
    field_source_suffixes = {
        "kind",
        "source_id",
        "page",
        "region_id",
        "planned_row_id",
        "row_label_cell_id",
        "physical_cell_id",
        "numeric_token_id",
        "header_ids",
        "quote_sha256",
    }
    for field in field_names:
        prefix = f"field_{field}"
        if not any(key.startswith(f"{prefix}_") for key in source):
            continue
        field_required = {
            f"{prefix}_value_sha256",
            f"{prefix}_status",
            f"{prefix}_source_count",
        }
        required.update(field_required)
        allowed.update(field_required | {f"{prefix}_reason"})
        source_count = source.get(f"{prefix}_source_count")
        if (
            not isinstance(source_count, str)
            or re.fullmatch(r"0|[1-9][0-9]*", source_count) is None
        ):
            raise PublicSnapshotError(f"{context}.{prefix}_source_count is invalid")
        for index in range(1, int(source_count) + 1):
            source_prefix = f"{prefix}_source_{index}"
            source_required = {
                f"{source_prefix}_kind",
                f"{source_prefix}_source_id",
                f"{source_prefix}_page",
            }
            required.update(source_required)
            allowed.update(
                source_required | {f"{source_prefix}_{suffix}" for suffix in field_source_suffixes}
            )
    missing = required - set(source)
    extra = set(source) - allowed
    if missing or extra:
        raise PublicSnapshotError(
            f"{context} has unsupported composer fields "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    if source["candidate_observation_id"] != observation_id or source["paper_id"] != paper_id:
        raise PublicSnapshotError(f"{context} candidate or paper binding disagrees")
    for field in (
        "export_provenance_mode",
        "export_composition_sha256",
        "tuple_sidecar_sha256",
        "legacy_provenance_reason",
        "review_manifest_sha256",
        "verifier_gate_required",
        "verifier_sidecar_sha256",
    ):
        if field in source_provenance and source.get(field) != source_provenance[field]:
            raise PublicSnapshotError(f"{context}.{field} disagrees with record provenance")
    for field in ("export_candidate_sha256", "tuple_gate_sha256", "verifier_gate_sha256"):
        if field in source and re.fullmatch(r"[a-f0-9]{64}", source[field]) is None:
            raise PublicSnapshotError(f"{context}.{field} is invalid")
    if "source_verifier_gate_passed" in source and source["source_verifier_gate_passed"] not in {
        "true",
        "false",
    }:
        raise PublicSnapshotError(f"{context}.source_verifier_gate_passed is invalid")
    if mode == "tuple_gated_production" and source.get("source_verifier_gate_passed") != "true":
        raise PublicSnapshotError(f"{context} production verifier gate did not pass")
    if ("proposal_ids" in source) != ("candidate_occurrence_ids" in source):
        raise PublicSnapshotError(f"{context} proposal trace fields are incomplete")
    return {key: source[key] for key in sorted(source)}


def _project_composer_uncertainty(value: Any, context: str) -> dict[str, Any]:
    source = _composer_object(
        value,
        required=set(),
        optional={
            "standard_error",
            "confidence_interval",
            "standard_deviation",
            "num_samples",
        },
        context=context,
    )
    if not source:
        raise PublicSnapshotError(f"{context} cannot be empty")
    result: dict[str, Any] = {}
    if "standard_error" in source:
        item = _composer_object(
            source["standard_error"],
            required={"value"},
            optional={"method"},
            context=f"{context}.standard_error",
        )
        result["standard_error"] = dict(item)
    if "confidence_interval" in source:
        item = _composer_object(
            source["confidence_interval"],
            required={"lower", "upper"},
            optional={"confidence_level", "method"},
            context=f"{context}.confidence_interval",
        )
        result["confidence_interval"] = dict(item)
    for field in ("standard_deviation", "num_samples"):
        if field in source:
            result[field] = source[field]
    return result


def _project_composer_eee_record(
    record: Mapping[str, Any], *, paper_id: str, context: str
) -> dict[str, Any]:
    """Reproject only the exact nested EEE shape emitted by the local composer."""

    source = _composer_object(
        record,
        required={
            "schema_version",
            "evaluation_id",
            "retrieved_timestamp",
            "source_metadata",
            "model_info",
            "eval_library",
            "evaluation_results",
        },
        context=context,
    )
    if not isinstance(source["evaluation_id"], str) or not source["evaluation_id"].startswith(
        f"paper/{paper_id}/"
    ):
        raise PublicSnapshotError(f"{context}.evaluation_id is not composer-bound")
    metadata = _composer_object(
        source["source_metadata"],
        required={
            "source_name",
            "source_type",
            "source_organization_name",
            "evaluator_relationship",
            "additional_details",
        },
        context=f"{context}.source_metadata",
    )
    if (
        metadata["source_type"] != "documentation"
        or metadata["source_organization_name"] != "paper authors"
        or metadata["evaluator_relationship"] != "other"
    ):
        raise PublicSnapshotError(f"{context}.source_metadata is not composer-emitted")
    source_provenance = _composer_source_provenance(
        metadata["additional_details"],
        paper_id=paper_id,
        context=f"{context}.source_metadata.additional_details",
    )
    model = _composer_object(
        source["model_info"],
        required={"name", "id", "additional_details"},
        optional={"developer"},
        context=f"{context}.model_info",
    )
    model_details = _composer_object(
        model["additional_details"],
        required={"identity_status"},
        optional={"reported_version"},
        context=f"{context}.model_info.additional_details",
    )
    if model_details["identity_status"] not in {"canonical", "raw_name"}:
        raise PublicSnapshotError(f"{context}.model_info.identity_status is unsupported")
    library = _composer_object(
        source["eval_library"],
        required={"name", "version", "additional_details"},
        context=f"{context}.eval_library",
    )
    library_details = _composer_object(
        library["additional_details"],
        required={"ingestion_method"},
        context=f"{context}.eval_library.additional_details",
    )
    if (
        library["name"] != "paper-reported"
        or library["version"] != "unknown"
        or library_details["ingestion_method"] != "proceedings-to-eee"
    ):
        raise PublicSnapshotError(f"{context}.eval_library is not composer-emitted")
    projected_results: list[dict[str, Any]] = []
    for index, raw_result in enumerate(
        _as_sequence(source["evaluation_results"], f"{context}.evaluation_results")
    ):
        result_context = f"{context}.evaluation_results[{index}]"
        result = _composer_object(
            raw_result,
            required={
                "evaluation_result_id",
                "evaluation_name",
                "source_data",
                "metric_config",
                "score_details",
            },
            optional={"evaluation_timestamp"},
            context=result_context,
        )
        observation_id = result["evaluation_result_id"]
        if not isinstance(observation_id, str) or not observation_id:
            raise PublicSnapshotError(f"{result_context}.evaluation_result_id is invalid")
        source_data = _as_mapping(result["source_data"], f"{result_context}.source_data")
        source_type = source_data.get("source_type")
        source_required = {"dataset_name", "source_type", "additional_details"}
        if source_type == "url":
            source_required.add("url")
        elif source_type != "other":
            raise PublicSnapshotError(f"{result_context}.source_data is not composer-emitted")
        source_data = _composer_object(
            source_data,
            required=source_required,
            context=f"{result_context}.source_data",
        )
        scope_details = _composer_string_map(
            source_data["additional_details"],
            allowed={
                "dataset_version",
                "split",
                "subset",
                "group",
                "language",
                "aggregation",
                "raw_scope",
                "samples_number_reported",
            },
            context=f"{result_context}.source_data.additional_details",
        )
        metric = _composer_object(
            result["metric_config"],
            required={"metric_name", "metric_parameters", "lower_is_better"},
            optional={
                "metric_id",
                "metric_kind",
                "metric_unit",
                "score_type",
                "min_score",
                "max_score",
                "additional_details",
            },
            context=f"{result_context}.metric_config",
        )
        bounded_metric = {"score_type", "min_score", "max_score"} & set(metric)
        if bounded_metric and (
            bounded_metric != {"score_type", "min_score", "max_score"}
            or metric["score_type"] != "continuous"
        ):
            raise PublicSnapshotError(f"{result_context}.metric_config bounds are unsupported")
        metric_details = None
        if "additional_details" in metric:
            metric_details = _composer_string_map(
                metric["additional_details"],
                allowed={
                    "construct",
                    "operationalization",
                    "decision_rule",
                    "evaluation_instrument",
                },
                context=f"{result_context}.metric_config.additional_details",
            )
            if not metric_details:
                raise PublicSnapshotError(
                    f"{result_context}.metric_config.additional_details cannot be empty"
                )
        score = _composer_object(
            result["score_details"],
            required={"score", "details"},
            optional={"uncertainty"},
            context=f"{result_context}.score_details",
        )
        details = _project_composer_score_details(
            score["details"],
            paper_id=paper_id,
            observation_id=observation_id,
            source_provenance=source_provenance,
            context=f"{result_context}.score_details.details",
        )
        projected_score: dict[str, Any] = {"score": score["score"], "details": details}
        if "uncertainty" in score:
            projected_score["uncertainty"] = _project_composer_uncertainty(
                score["uncertainty"], f"{result_context}.score_details.uncertainty"
            )
        projected_source_data = {
            "dataset_name": source_data["dataset_name"],
            "source_type": source_data["source_type"],
            **({"url": list(source_data["url"])} if "url" in source_data else {}),
            "additional_details": scope_details,
        }
        projected_metric = {
            key: (
                dict(metric[key])
                if key == "metric_parameters"
                else metric_details
                if key == "additional_details"
                else metric[key]
            )
            for key in metric
        }
        projected_results.append(
            {
                "evaluation_result_id": observation_id,
                "evaluation_name": result["evaluation_name"],
                "source_data": projected_source_data,
                "metric_config": projected_metric,
                "score_details": projected_score,
                **(
                    {"evaluation_timestamp": result["evaluation_timestamp"]}
                    if "evaluation_timestamp" in result
                    else {}
                ),
            }
        )
    return {
        "schema_version": source["schema_version"],
        "evaluation_id": source["evaluation_id"],
        "retrieved_timestamp": source["retrieved_timestamp"],
        "source_metadata": {
            "source_name": metadata["source_name"],
            "source_type": metadata["source_type"],
            "source_organization_name": metadata["source_organization_name"],
            "evaluator_relationship": metadata["evaluator_relationship"],
            "additional_details": source_provenance,
        },
        "model_info": {
            "name": model["name"],
            "id": model["id"],
            **({"developer": model["developer"]} if "developer" in model else {}),
            "additional_details": dict(model_details),
        },
        "eval_library": {
            "name": library["name"],
            "version": library["version"],
            "additional_details": dict(library_details),
        },
        "evaluation_results": projected_results,
    }


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
            paper_root = root / paper_id
            if not paper_root.exists():
                continue
            eee_dir = _paper_directory(root, paper_id, f"EEE paper {paper_id}") / "eee"
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
                public_record = _project_composer_eee_record(
                    record,
                    paper_id=paper_id,
                    context=f"EEE record {source_path.name}",
                )
                public_issues = validate_eee_record(public_record, schema)
                if public_issues:
                    details = "; ".join(
                        f"{issue.path}: {issue.message}" for issue in public_issues[:3]
                    )
                    raise PublicSnapshotError(
                        f"reprojected EEE validation failed for {source_path.name}: {details}"
                    )
                target = destination / "eee" / paper_id / source_path.name
                if target.exists():
                    raise PublicSnapshotError(f"duplicate public EEE output: {target.name}")
                write_json(target, public_record)
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


def _validate_human_review_provenance(
    human_review: Mapping[str, Any],
    run_roots: Sequence[Path],
    corpora: Sequence[Mapping[str, Any]],
) -> None:
    """Reconstruct the private audit fingerprint without publishing its input hashes."""

    artifacts: list[dict[str, Any]] = []
    for root, corpus in zip(run_roots, corpora, strict=True):
        for paper in _as_sequence(corpus.get("papers_detail"), "corpus.papers_detail"):
            paper_id = paper["paper_id"]
            paper_root = _paper_directory(root, paper_id, f"human-review.{paper_id}")
            run_path = paper_root / "run.json"
            observations_path = paper_root / "observations.jsonl"
            if (
                not run_path.is_file()
                or run_path.is_symlink()
                or not observations_path.is_file()
                or observations_path.is_symlink()
            ):
                raise PublicSnapshotError(
                    "human review provenance requires regular run and observations "
                    f"files for {paper_id}"
                )
            try:
                observations = [
                    CandidateObservation.model_validate_json(line)
                    for line in observations_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (TypeError, ValueError) as error:
                raise PublicSnapshotError(
                    f"human review provenance has invalid observations for {paper_id}"
                ) from error
            if any(item.paper_id != paper_id for item in observations):
                raise PublicSnapshotError(
                    f"human review observation paper_id disagrees for {paper_id}"
                )
            if len(observations) != paper["counts"]["candidates"]:
                raise PublicSnapshotError(
                    f"human review observation count disagrees for {paper_id}"
                )
            artifacts.append(
                {
                    "paper_id": paper_id,
                    "run_status": paper["status"],
                    "run_sha256": sha256_file(run_path),
                    "observations_sha256": sha256_file(observations_path),
                    "candidate_count": len(observations),
                    "eee_record_count": paper["counts"]["eee_records"],
                }
            )
    artifacts.sort(key=lambda item: item["paper_id"])
    basis = {
        "sampling_policy": "risk-stratified-paper-coverage/0.1",
        "sample_requested": human_review["sample"]["requested"],
        "source_artifacts": artifacts,
    }
    expected_audit_id = f"audit_{sha256_bytes(canonical_json_bytes(basis))[:20]}"
    if human_review.get("audit_id") != expected_audit_id:
        raise PublicSnapshotError("human review audit_id disagrees with current run artifacts")


def _validate_selected_model_production_binding(
    selected_model: str,
    model_selection: Mapping[str, Any],
    corpora: Sequence[Mapping[str, Any]],
) -> None:
    determinism = _as_mapping(model_selection.get("determinism"), "model selection.determinism")
    selection_contract = _as_mapping(
        model_selection.get("request_contract"), "model selection.request_contract"
    )
    expected_request_contract = dict(selection_contract)
    expected_request_contract.update(
        {
            "schema_version": "provider-request-contract/0.2",
            "max_tokens": determinism["max_tokens"],
            "completion_token_parameter": completion_token_parameter_for_model(selected_model),
        }
    )
    extractor_fields = (
        "temperature",
        "reasoning_effort",
        "max_tokens",
        "seed",
        "require_parameters",
        "prompt_sha256",
    )
    for corpus in corpora:
        for paper in _as_sequence(corpus.get("papers_detail"), "corpus.papers_detail"):
            extractor = _as_mapping(paper.get("extractor"), "paper.extractor")
            if extractor.get("model") != selected_model:
                raise PublicSnapshotError(
                    "selected model disagrees with a published paper-run extractor model"
                )
            if any(extractor.get(field) != determinism.get(field) for field in extractor_fields):
                raise PublicSnapshotError(
                    "selected model determinism disagrees with paper-run extractor settings"
                )
            if extractor.get("request_contract") != expected_request_contract:
                raise PublicSnapshotError(
                    "selected model request contract disagrees with paper-run extractor contract"
                )
            usage = _as_mapping(extractor.get("usage"), "paper.extractor.usage")
            if usage.get("model_returned_matches_requested_calls") != usage.get("calls_attempted"):
                raise PublicSnapshotError(
                    "selected model lacks exact returned-model evidence for every completed call"
                )
            if paper.get("candidate_validation", {}).get("min_confidence") != determinism.get(
                "min_confidence"
            ):
                raise PublicSnapshotError(
                    "selected model confidence policy disagrees with paper-run validation"
                )
            if paper.get("result_block_segmentation") != determinism.get("segmentation"):
                raise PublicSnapshotError(
                    "selected model segmentation disagrees with paper-run segmentation"
                )
            if paper.get("code") != model_selection.get("code"):
                raise PublicSnapshotError(
                    "selected model code provenance disagrees with paper-run code"
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
        expected_eee_schema = {"version": authority.version, "sha256": authority.sha256}
        if any(
            paper["eee_schema"] != expected_eee_schema
            for corpus in corpora
            for paper in corpus["papers_detail"]
        ):
            raise PublicSnapshotError(
                "paper-run EEE schema disagrees with the pinned snapshot schema"
            )
        _validate_human_review_corpus_population(human_review, corpora)
        _validate_human_review_provenance(human_review, run_roots, corpora)
        model_selection = _project_model_selection(model_selection_path, selected_model)
        if selected_model is not None:
            _validate_selected_model_production_binding(selected_model, model_selection, corpora)
        sources, paper_ids = _project_sources(run_roots)
        audits = _project_reference_audits(run_roots, corpora)
        eee_files = _copy_valid_eee(
            run_roots=run_roots,
            paper_ids=paper_ids,
            destination=temporary,
            schema=schema,
        )
        expected_eee_files = sum(
            paper["counts"]["eee_records"]
            for corpus in corpora
            for paper in corpus["papers_detail"]
        )
        expected_eee_papers = sum(
            paper["counts"]["eee_records"] > 0
            for corpus in corpora
            for paper in corpus["papers_detail"]
        )
        copied_eee_papers = len({Path(path).parts[1] for path in eee_files})
        if len(eee_files) != expected_eee_files or copied_eee_papers != expected_eee_papers:
            raise PublicSnapshotError("copied EEE files disagree with paper-run counts")
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
                "papers_with_eee": copied_eee_papers,
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
