"""End-to-end paper pipeline with content-addressed intermediate artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from proceedings_to_eee.composition.eee import compose_eee_records
from proceedings_to_eee.corpus import CorpusSpec, PaperSpec, build_corpus_binding
from proceedings_to_eee.domain.attribution import (
    AttributionState,
    AttributionVerdict,
    OriginExportPolicy,
)
from proceedings_to_eee.domain.export_provenance import (
    legacy_export_provenance,
    tuple_gated_export_provenance,
    tuple_unverified_review_provenance,
)
from proceedings_to_eee.domain.lineage import build_candidate_lineage
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.provenance import ProposalStage
from proceedings_to_eee.domain.status import ClaimType, ExportStatus
from proceedings_to_eee.evaluation.control_coverage import (
    control_examination,
    observation_examination,
)
from proceedings_to_eee.evaluation.corpus_score import aggregate_reference_scores
from proceedings_to_eee.evaluation.reference_score import score_reference
from proceedings_to_eee.evaluation.spot_checks import score_spot_checks
from proceedings_to_eee.extraction.llm import (
    EXTRACTOR_REASONING_EFFORT,
    EXTRACTOR_SEED,
    EXTRACTOR_TEMPERATURE,
    RowEnumerationOutcome,
    enumerate_row_batch,
    extract_page_candidates,
    extractor_request_contract,
    row_extractor_request_contract,
)
from proceedings_to_eee.extraction.pdf_layout import (
    PageFragment,
    PdfLayout,
    extract_pdf_layout,
    select_result_pages,
)
from proceedings_to_eee.extraction.prompt import (
    SYSTEM_PROMPT,
    page_prompt,
    prompt_hash,
    row_prompt_hash,
)
from proceedings_to_eee.extraction.result_blocks import (
    LEGACY_RECOVERY_MAX_DEPTH,
    ResultBlock,
    ResultBlockConfig,
    segment_page_result_blocks,
    split_result_block,
)
from proceedings_to_eee.extraction.row_enumeration import (
    RowAttemptTelemetry,
    RowBatch,
    RowDisposition,
    RowDispositionRecord,
    RowEnumerationConfig,
    RowEnumerationPlan,
    RowTerminalLedger,
    build_row_enumeration_plan,
)
from proceedings_to_eee.extraction.row_validation import (
    partition_row_provider_calls,
    validate_batch_outcome_against,
    validate_batch_outcome_prefix,
    validate_outcome_against,
)
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from proceedings_to_eee.providers.budget import (
    BudgetedProviderClient,
    ProviderBudgetContractError,
    ProviderBudgetError,
    ProviderBudgetExhausted,
    ProviderBudgetLimits,
    provider_budget_contract,
)
from proceedings_to_eee.providers.openrouter import (
    OpenRouterClient,
    ProviderCall,
    ProviderRequestRejectedError,
    ProviderResponseValidationError,
    completion_token_parameter_for_model,
    public_provider_call,
)
from proceedings_to_eee.reference import load_reference
from proceedings_to_eee.reporting.corpus_html import render_corpus_html_file
from proceedings_to_eee.reporting.html import render_review_report
from proceedings_to_eee.resolution.origin_retrieval import (
    DEFAULT_MAX_TOKENS as DEFAULT_ORIGIN_MAX_TOKENS,
)
from proceedings_to_eee.resolution.origin_retrieval import (
    ORIGIN_REASONING_EFFORT,
    ORIGIN_SCHEMA_NAME,
    ORIGIN_SEED,
    ORIGIN_SYSTEM_PROMPT,
    ORIGIN_TEMPERATURE,
    OriginRetrievalBundle,
    ProducerOriginAssessment,
    ProducerOriginProposal,
    candidate_origin_binding_sha256,
    layout_binding_sha256,
    producer_origin_prompt,
    producer_origin_prompt_hash,
    producer_origin_request_contract,
    producer_origin_request_fingerprint,
    producer_origin_wire_response_sha256,
    propose_producer_origin,
    retrieve_origin_context,
    verify_producer_origin_proposal,
)
from proceedings_to_eee.resolution.tuple_resolution import (
    DEFAULT_MAX_TOKENS as DEFAULT_TUPLE_MAX_TOKENS,
)
from proceedings_to_eee.resolution.tuple_resolution import (
    TUPLE_REASONING_EFFORT,
    TUPLE_SCHEMA_NAME,
    TUPLE_SEED,
    TUPLE_SYSTEM_PROMPT,
    TUPLE_TEMPERATURE,
    TupleResolutionAssessment,
    TupleResolutionDecision,
    TupleResolutionInput,
    TupleWireProposal,
    build_tuple_resolution_input,
    propose_tuple_resolution,
    tuple_assessment_is_export_concordant,
    tuple_candidate_binding_sha256,
    tuple_resolution_prompt,
    tuple_resolution_prompt_hash,
    tuple_resolution_request_contract,
    tuple_wire_response_sha256,
    verify_tuple_resolution_proposal,
)
from proceedings_to_eee.sources.manifest import (
    HostRateLimiter,
    LicenseDisposition,
    SourceManifest,
    SourceRole,
    download_and_freeze_source,
    freeze_local_source,
    freeze_repository_source,
    resolve_cached_path,
)
from proceedings_to_eee.sources.processing import build_source_processing_artifact
from proceedings_to_eee.validation.candidates import (
    deduplicate_candidates_with_lineage,
    route_candidate_attribution,
    validate_non_origin_candidates,
)
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record
from proceedings_to_eee.verification.binding import (
    bind_candidate_block,
    frozen_evidence_block,
)
from proceedings_to_eee.verification.independent import (
    DEFAULT_MAX_TOKENS as DEFAULT_VERIFIER_MAX_TOKENS,
)
from proceedings_to_eee.verification.independent import (
    VERIFIER_REQUEST_SETTINGS,
    VERIFIER_SCHEMA_NAME,
    VERIFIER_SYSTEM_PROMPT,
    CandidateVerification,
    FrozenEvidenceBlock,
    IndependentDecision,
    VerificationRequest,
    contextualize_verification,
    verification_prompt,
    verifier_assessment_sha256,
    verifier_evidence_block_sha256,
    verifier_request_contract,
    verify_candidate,
)


@dataclass(frozen=True)
class PipelineSettings:
    project_root: Path
    schema_path: Path
    schema_sha256: str
    output_root: Path
    model: str
    min_confidence: float = 0.8
    max_tokens: int = 16_000
    temperature: float | None = EXTRACTOR_TEMPERATURE
    reasoning_effort: str | None = EXTRACTOR_REASONING_EFFORT
    #: Which producer-origin bases this run will export. The pipeline itself always runs
    #: the default; the tiered policy is reached through `ere recompose-census`.
    origin_policy: OriginExportPolicy = OriginExportPolicy.POSITIVE_ONLY
    seed: int | None = EXTRACTOR_SEED
    max_blocks_per_page: int = 6
    row_enumeration_enabled: bool = False
    row_model: str | None = None
    row_enumeration_config: RowEnumerationConfig = dataclass_field(
        default_factory=RowEnumerationConfig
    )
    row_estimated_call_cost_usd: float | None = None
    tuple_model: str | None = None
    tuple_max_tokens: int = DEFAULT_TUPLE_MAX_TOKENS
    verifier_model: str | None = None
    verifier_max_tokens: int = DEFAULT_VERIFIER_MAX_TOKENS
    origin_model: str | None = None
    origin_max_tokens: int = DEFAULT_ORIGIN_MAX_TOKENS
    provider_max_structured_calls: int = 10_000
    provider_max_cost_usd: float = 1_000.0
    provider_cost_reservation_per_call_usd: float = 0.25


@dataclass(frozen=True)
class LegacyRecoveryFailure:
    """Secret-free terminal state for one bounded recovery subtree."""

    block_id: str
    page: int
    depth: int
    error_code: str
    completed_provider_call: bool
    terminal_reason: str
    safe_details: dict[str, int] = dataclass_field(default_factory=dict)


@dataclass
class LegacyRecoveryOutcome:
    """All usable work and typed failures from one bounded split tree."""

    candidates: list[CandidateObservation] = dataclass_field(default_factory=list)
    calls: list[ProviderCall] = dataclass_field(default_factory=list)
    successful_calls: list[ProviderCall] = dataclass_field(default_factory=list)
    new_calls: list[ProviderCall] = dataclass_field(default_factory=list)
    new_successful_calls: list[ProviderCall] = dataclass_field(default_factory=list)
    resumed_calls: list[ProviderCall] = dataclass_field(default_factory=list)
    resumed_successful_calls: list[ProviderCall] = dataclass_field(default_factory=list)
    warnings: list[str] = dataclass_field(default_factory=list)
    terminal_failures: list[LegacyRecoveryFailure] = dataclass_field(default_factory=list)
    max_depth_reached: int = 0

    @property
    def succeeded(self) -> bool:
        return not self.terminal_failures


_EXTRACTOR_CHECKPOINT_SCHEMA_VERSION = "extractor-block-checkpoint/0.3"
_EXTRACTOR_CHECKPOINT_CONTRACT_VERSION = "extractor-block-checkpoint-contract/0.3"
_EXTRACTOR_RECOVERY_ENTRY_VERSION = "extractor-recovery-progress/0.2"
# 0.2 adds origin_policy: the export gate can now be told which producer-origin bases a
# run is willing to export, so the run must say which one it used.
_CANDIDATE_VALIDATION_SCHEMA_VERSION = "candidate-validation/0.2"
_ROW_CHECKPOINT_SCHEMA_VERSION = "row-enumeration-checkpoint/0.4"
# This version also names the provider-response projection contract. Bump it when
# row responses can no longer be rehydrated with the current typed domain models.
_ROW_CHECKPOINT_CONTRACT_VERSION = "row-enumeration-checkpoint-contract/0.4"
_TUPLE_CHECKPOINT_SCHEMA_VERSION = "tuple-resolution-checkpoint/0.2"
_TUPLE_CHECKPOINT_CONTRACT_VERSION = "tuple-resolution-checkpoint-contract/0.2"
_TUPLE_CHECKPOINT_ENTRY_VERSION = "tuple-resolution-checkpoint-entry/0.2"
_VERIFIER_CHECKPOINT_SCHEMA_VERSION = "independent-verifier-checkpoint/0.4"
_VERIFIER_CHECKPOINT_CONTRACT_VERSION = "independent-verifier-checkpoint-contract/0.4"
_VERIFIER_CHECKPOINT_ENTRY_VERSION = "independent-verifier-checkpoint-entry/0.4"
_ORIGIN_CHECKPOINT_SCHEMA_VERSION = "producer-origin-checkpoint/0.3"
_ORIGIN_CHECKPOINT_CONTRACT_VERSION = "producer-origin-checkpoint-contract/0.3"
_ORIGIN_CHECKPOINT_ENTRY_VERSION = "producer-origin-checkpoint-entry/0.3"

_RETRYABLE_NO_CALL_FAILURE_CODES = frozenset(
    {"provider_request_rejected", "provider_transport_failed"}
)
_PIPELINE_RUN_SCHEMA_VERSION = "pipeline-run/0.4"
_CORPUS_RUN_SCHEMA_VERSION = "corpus-run/0.3"
_RETURNED_MODEL_MISMATCH = "provider_response_returned_model_mismatch"
_PAPER_RUN_OUTPUTS = (
    "run.json",
    "reference-score.json",
    "observations.jsonl",
    "candidate-lineage.json",
    "verifications.jsonl",
    "spot-checks.json",
    "source-processing.json",
    "review.html",
)
_CORPUS_RUN_OUTPUTS = ("corpus-run.json", "corpus-evaluation.json", "corpus-review.html")


def _extractor_run_configuration(settings: PipelineSettings) -> dict[str, Any]:
    """Return reproducibility metadata available before any extractor call."""

    request_contract = extractor_request_contract(
        seed=settings.seed,
        model=settings.model,
        max_tokens=settings.max_tokens,
    )
    return {
        "provider": "openrouter",
        "model": settings.model,
        "temperature": settings.temperature,
        "reasoning_effort": settings.reasoning_effort,
        "max_tokens": settings.max_tokens,
        "seed": settings.seed,
        "require_parameters": request_contract["routing"]["require_parameters"],
        "prompt_sha256": prompt_hash(),
        "request_contract": request_contract,
    }


def _candidate_validation_run_configuration(settings: PipelineSettings) -> dict[str, Any]:
    """Return the versioned deterministic candidate-export policy.

    ``min_confidence`` affects which observations can survive local validation, so it
    is part of the run identity even though it never enters a provider prompt.
    """

    return {
        "schema_version": _CANDIDATE_VALIDATION_SCHEMA_VERSION,
        "min_confidence": settings.min_confidence,
        "origin_policy": settings.origin_policy.value,
    }


def _effective_row_model(settings: PipelineSettings) -> str:
    """Return the explicitly selected row model or the extractor fallback."""

    return settings.row_model or settings.model


def _validate_pipeline_settings(settings: PipelineSettings) -> None:
    """Reject cross-stage configurations before any source or provider work."""

    if not isinstance(settings.model, str) or not settings.model.strip():
        raise ValueError("extractor model is required")
    if (
        isinstance(settings.min_confidence, bool)
        or not isinstance(settings.min_confidence, int | float)
        or not math.isfinite(float(settings.min_confidence))
        or not 0.0 <= settings.min_confidence <= 1.0
    ):
        raise ValueError("min_confidence must be a finite number between 0 and 1")
    if settings.temperature is not None and (
        isinstance(settings.temperature, bool)
        or not isinstance(settings.temperature, int | float)
        or not math.isfinite(float(settings.temperature))
        or not 0.0 <= float(settings.temperature) <= 2.0
    ):
        raise ValueError("temperature must be null or a finite number between 0 and 2")
    if settings.reasoning_effort is not None and (
        not isinstance(settings.reasoning_effort, str) or not settings.reasoning_effort.strip()
    ):
        raise ValueError("reasoning_effort must be null or a non-empty string")
    if settings.seed is not None and (
        isinstance(settings.seed, bool) or not isinstance(settings.seed, int) or settings.seed < 0
    ):
        raise ValueError("seed must be null or a non-negative integer")
    for name, value in (
        ("max_tokens", settings.max_tokens),
        ("tuple_max_tokens", settings.tuple_max_tokens),
        ("verifier_max_tokens", settings.verifier_max_tokens),
        ("origin_max_tokens", settings.origin_max_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if settings.row_model is not None:
        if not isinstance(settings.row_model, str) or not settings.row_model.strip():
            raise ValueError("row model must be non-empty when supplied")
        if not settings.row_enumeration_enabled:
            raise ValueError("row_model requires row_enumeration_enabled")
    for name, value in (
        ("tuple", settings.tuple_model),
        ("verifier", settings.verifier_model),
        ("origin", settings.origin_model),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} model must be non-empty when supplied")
    if settings.verifier_model is not None and settings.tuple_model is None:
        raise ValueError("verifier_model requires tuple_model")
    if settings.origin_model is not None and settings.verifier_model is None:
        raise ValueError("origin_model requires verifier_model")


def _row_enumeration_run_configuration(settings: PipelineSettings) -> dict[str, Any]:
    """Return the independent row-stage contract, even when the stage is disabled."""

    request_contract = row_extractor_request_contract(
        seed=settings.seed,
        model=_effective_row_model(settings),
        max_tokens=settings.max_tokens,
    )
    return {
        "enabled": settings.row_enumeration_enabled,
        "provider": "openrouter",
        "model": _effective_row_model(settings),
        "temperature": settings.temperature,
        "reasoning_effort": settings.reasoning_effort,
        "max_tokens": settings.max_tokens,
        "seed": settings.seed,
        "require_parameters": request_contract["routing"]["require_parameters"],
        "prompt_sha256": row_prompt_hash(),
        "request_contract": request_contract,
        "limits": settings.row_enumeration_config.model_dump(mode="json"),
    }


def _row_preflight(
    *,
    settings: PipelineSettings,
    block_count: int,
    plan: RowEnumerationPlan,
) -> dict[str, Any]:
    """Bound row-stage calls and label any cost projection as an estimate."""

    if (
        settings.row_estimated_call_cost_usd is not None
        and settings.row_estimated_call_cost_usd < 0
    ):
        raise ValueError("row estimated call cost must be non-negative")
    expected_total = block_count + plan.telemetry.expected_calls
    maximum_total = block_count + plan.telemetry.maximum_calls
    per_call = settings.row_estimated_call_cost_usd
    return {
        "basis": (
            "block count plus deterministic row-plan batches before provider execution; "
            "maximum permits one unresolved-only split level and is a hard call bound"
        ),
        "baseline_block_calls": block_count,
        "planned_row_base_calls": plan.telemetry.expected_calls,
        "maximum_row_calls": plan.telemetry.maximum_calls,
        "expected_total_calls": expected_total,
        "maximum_total_calls": maximum_total,
        "expected_call_multiplier": expected_total / block_count if block_count else None,
        "maximum_call_multiplier": maximum_total / block_count if block_count else None,
        "estimated_cost_basis": (
            "user-supplied historical mean cost per successful call; not a provider quote"
            if per_call is not None
            else None
        ),
        "estimated_cost_per_call_usd": per_call,
        "estimated_row_cost_usd": (
            round(plan.telemetry.expected_calls * per_call, 12) if per_call is not None else None
        ),
        "estimated_expected_total_cost_usd": (
            round(expected_total * per_call, 12) if per_call is not None else None
        ),
        "estimated_maximum_total_cost_usd": (
            round(maximum_total * per_call, 12) if per_call is not None else None
        ),
    }


def _tuple_run_configuration(settings: PipelineSettings) -> dict[str, Any]:
    """Return the quote-free, demotion-only tuple-stage contract."""

    request_contract = tuple_resolution_request_contract(
        require_parameters=True,
        model=settings.tuple_model,
        max_tokens=settings.tuple_max_tokens if settings.tuple_model is not None else None,
    )
    return {
        "enabled": settings.tuple_model is not None,
        "provider": "openrouter",
        "model": settings.tuple_model,
        "max_tokens": settings.tuple_max_tokens,
        "temperature": TUPLE_TEMPERATURE,
        "reasoning_effort": TUPLE_REASONING_EFFORT,
        "seed": TUPLE_SEED,
        "require_parameters": request_contract["routing"]["require_parameters"],
        "prompt_sha256": tuple_resolution_prompt_hash(),
        "request_contract": request_contract,
        "requires_export_concordant_tuple": True,
        "optional_unclaimed_fields_may_remain_unresolved": [
            "system_version",
            "dataset_version",
            "setting",
        ],
        "mutates_candidate_tuple": False,
        "allows_origin_or_export": False,
    }


def _verifier_run_configuration(settings: PipelineSettings) -> dict[str, Any]:
    """Return reproducibility metadata even when verification is disabled or fails."""

    return {
        "enabled": settings.verifier_model is not None,
        "model": settings.verifier_model,
        "max_tokens": settings.verifier_max_tokens,
        **VERIFIER_REQUEST_SETTINGS.as_dict(),
        "verification_schema_version": "candidate-verification/0.2",
        "grounding_schema_version": "candidate-verification-grounding/0.1",
        "request_contract": verifier_request_contract(
            model=settings.verifier_model,
            max_tokens=(
                settings.verifier_max_tokens if settings.verifier_model is not None else None
            ),
        ),
    }


def _origin_run_configuration(settings: PipelineSettings) -> dict[str, Any]:
    """Return the complete quote-free origin-provider contract."""

    request_contract = producer_origin_request_contract(
        model=settings.origin_model,
        max_tokens=settings.origin_max_tokens if settings.origin_model is not None else None,
    )
    return {
        "enabled": settings.origin_model is not None,
        "requires_independent_verifier_accept": True,
        "model": settings.origin_model,
        "max_tokens": settings.origin_max_tokens,
        "temperature": ORIGIN_TEMPERATURE,
        "reasoning_effort": ORIGIN_REASONING_EFFORT,
        "seed": ORIGIN_SEED,
        "require_parameters": request_contract["routing"]["require_parameters"],
        "prompt_sha256": producer_origin_prompt_hash(),
        "request_contract": request_contract,
        "automatic_paper_produced_forbidden": True,
    }


def _provider_call_telemetry(
    calls: list[ProviderCall],
    *,
    basis: str | None = None,
) -> dict[str, Any]:
    """Aggregate only secret-free successful-call metadata as honest lower bounds."""

    costs: list[float] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    reasoning_tokens: list[int] = []
    total_tokens: list[int] = []
    latencies: list[float] = []
    attempts_total = 0
    for call in calls:
        if not math.isfinite(call.latency_seconds) or call.latency_seconds < 0:
            raise ValueError("provider latency must be non-negative and finite")
        latencies.append(float(call.latency_seconds))
        attempts_total += call.attempts
        if call.cost_usd is not None:
            if not math.isfinite(call.cost_usd) or call.cost_usd < 0:
                raise ValueError("provider cost must be non-negative and finite")
            costs.append(float(call.cost_usd))
        for value, output, field in (
            (call.input_tokens, input_tokens, "input_tokens"),
            (call.output_tokens, output_tokens, "output_tokens"),
            (call.reasoning_tokens, reasoning_tokens, "reasoning_tokens"),
            (call.total_tokens, total_tokens, "total_tokens"),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"provider {field} must be a non-negative integer")
            output.append(value)
    call_count = len(calls)
    return {
        "basis": basis
        or (
            "successful final block calls; cost, token, retry, and attempt totals are lower "
            "bounds when provider metadata or superseded/failed attempts are unavailable"
        ),
        "calls": call_count,
        "cost_usd_lower_bound": round(sum(costs), 12),
        "cost_reported_calls": len(costs),
        "input_tokens_lower_bound": sum(input_tokens),
        "input_tokens_reported_calls": len(input_tokens),
        "output_tokens_lower_bound": sum(output_tokens),
        "output_tokens_reported_calls": len(output_tokens),
        "reasoning_tokens_lower_bound": sum(reasoning_tokens),
        "reasoning_tokens_reported_calls": len(reasoning_tokens),
        "total_tokens_lower_bound": sum(total_tokens),
        "total_tokens_reported_calls": len(total_tokens),
        "latency_seconds_total": round(sum(latencies), 6),
        "latency_seconds_mean": round(sum(latencies) / call_count, 6) if calls else None,
        "latency_seconds_max": round(max(latencies), 6) if calls else None,
        "attempts_lower_bound": attempts_total,
        "retries_lower_bound": attempts_total - call_count,
    }


def _public_provider_call(call: ProviderCall) -> dict[str, Any]:
    return public_provider_call(call)


def _corpus_operational_metrics(
    summaries: list[dict[str, Any]], *, wall_clock_seconds: float
) -> dict[str, Any]:
    telemetry = []
    execution = []
    row_telemetry = []
    row_execution = []
    row_plans = []
    tuple_telemetry = []
    tuple_execution = []
    verifier_telemetry = []
    verifier_execution = []
    origin_telemetry = []
    origin_execution = []
    for summary in summaries:
        extractor = summary.get("extractor")
        if not isinstance(extractor, dict):
            continue
        item = extractor.get("completed_call_telemetry")
        if not isinstance(item, dict):
            item = extractor.get("successful_call_telemetry")
        if isinstance(item, dict):
            telemetry.append(item)
        item = extractor.get("execution")
        if isinstance(item, dict):
            execution.append(item)
        row_stage = summary.get("row_enumeration")
        if not isinstance(row_stage, dict):
            continue
        item = row_stage.get("completed_call_telemetry")
        if not isinstance(item, dict):
            item = row_stage.get("successful_call_telemetry")
        if isinstance(item, dict):
            row_telemetry.append(item)
        item = row_stage.get("execution")
        if isinstance(item, dict):
            row_execution.append(item)
        item = row_stage.get("plan")
        if isinstance(item, dict):
            row_plans.append(item)
        tuple_stage = summary.get("tuple_resolution")
        if isinstance(tuple_stage, dict):
            item = tuple_stage.get("completed_call_telemetry")
            if isinstance(item, dict):
                tuple_telemetry.append(item)
            item = tuple_stage.get("execution")
            if isinstance(item, dict):
                tuple_execution.append(item)
        verifier_stage = summary.get("verifier")
        if isinstance(verifier_stage, dict):
            item = verifier_stage.get("completed_call_telemetry")
            if not isinstance(item, dict):
                item = verifier_stage.get("successful_call_telemetry")
            if isinstance(item, dict):
                verifier_telemetry.append(item)
            item = verifier_stage.get("execution")
            if isinstance(item, dict):
                verifier_execution.append(item)
        origin_stage = summary.get("origin_retrieval")
        if isinstance(origin_stage, dict):
            item = origin_stage.get("completed_call_telemetry")
            if isinstance(item, dict):
                origin_telemetry.append(item)
            item = origin_stage.get("execution")
            if isinstance(item, dict):
                origin_execution.append(item)
    calls = sum(int(item.get("calls", 0)) for item in telemetry)
    latency_total = sum(float(item.get("latency_seconds_total", 0.0)) for item in telemetry)
    latency_max_values = [
        float(item["latency_seconds_max"])
        for item in telemetry
        if item.get("latency_seconds_max") is not None
    ]
    row_calls = sum(int(item.get("calls", 0)) for item in row_telemetry)
    row_latency_total = sum(float(item.get("latency_seconds_total", 0.0)) for item in row_telemetry)
    row_latency_max_values = [
        float(item["latency_seconds_max"])
        for item in row_telemetry
        if item.get("latency_seconds_max") is not None
    ]
    tuple_calls = sum(int(item.get("calls", 0)) for item in tuple_telemetry)
    tuple_latency_total = sum(
        float(item.get("latency_seconds_total", 0.0)) for item in tuple_telemetry
    )
    tuple_latency_max_values = [
        float(item["latency_seconds_max"])
        for item in tuple_telemetry
        if item.get("latency_seconds_max") is not None
    ]
    verifier_calls = sum(int(item.get("calls", 0)) for item in verifier_telemetry)
    verifier_latency_total = sum(
        float(item.get("latency_seconds_total", 0.0)) for item in verifier_telemetry
    )
    verifier_latency_max_values = [
        float(item["latency_seconds_max"])
        for item in verifier_telemetry
        if item.get("latency_seconds_max") is not None
    ]
    origin_calls = sum(int(item.get("calls", 0)) for item in origin_telemetry)
    origin_latency_total = sum(
        float(item.get("latency_seconds_total", 0.0)) for item in origin_telemetry
    )
    origin_latency_max_values = [
        float(item["latency_seconds_max"])
        for item in origin_telemetry
        if item.get("latency_seconds_max") is not None
    ]
    return {
        "wall_clock_seconds": round(wall_clock_seconds, 6),
        "extractor": {
            "basis": (
                "successful final block calls across the split; monetary, token, retry, and "
                "attempt totals are lower bounds"
            ),
            "calls": calls,
            "cost_usd_lower_bound": round(
                sum(float(item.get("cost_usd_lower_bound", 0.0)) for item in telemetry), 12
            ),
            "cost_reported_calls": sum(
                int(item.get("cost_reported_calls", 0)) for item in telemetry
            ),
            "input_tokens_lower_bound": sum(
                int(item.get("input_tokens_lower_bound", 0)) for item in telemetry
            ),
            "input_tokens_reported_calls": sum(
                int(item.get("input_tokens_reported_calls", 0)) for item in telemetry
            ),
            "output_tokens_lower_bound": sum(
                int(item.get("output_tokens_lower_bound", 0)) for item in telemetry
            ),
            "output_tokens_reported_calls": sum(
                int(item.get("output_tokens_reported_calls", 0)) for item in telemetry
            ),
            "reasoning_tokens_lower_bound": sum(
                int(item.get("reasoning_tokens_lower_bound", 0)) for item in telemetry
            ),
            "reasoning_tokens_reported_calls": sum(
                int(item.get("reasoning_tokens_reported_calls", 0)) for item in telemetry
            ),
            "total_tokens_lower_bound": sum(
                int(item.get("total_tokens_lower_bound", 0)) for item in telemetry
            ),
            "total_tokens_reported_calls": sum(
                int(item.get("total_tokens_reported_calls", 0)) for item in telemetry
            ),
            "latency_seconds_total": round(latency_total, 6),
            "latency_seconds_mean": round(latency_total / calls, 6) if calls else None,
            "latency_seconds_max": round(max(latency_max_values), 6)
            if latency_max_values
            else None,
            "attempts_lower_bound": sum(
                int(item.get("attempts_lower_bound", 0)) for item in telemetry
            ),
            "retries_lower_bound": sum(
                int(item.get("retries_lower_bound", 0)) for item in telemetry
            ),
            "blocks_total": sum(int(item.get("blocks_total", 0)) for item in execution),
            "blocks_succeeded": sum(int(item.get("blocks_succeeded", 0)) for item in execution),
            "blocks_resumed": sum(int(item.get("blocks_resumed", 0)) for item in execution),
            "blocks_failed": sum(int(item.get("blocks_failed", 0)) for item in execution),
        },
        "row_enumeration": {
            "basis": (
                "bounded staged row calls across the split; monetary, token, retry, and "
                "attempt totals are lower bounds"
            ),
            "calls": row_calls,
            "cost_usd_lower_bound": round(
                sum(float(item.get("cost_usd_lower_bound", 0.0)) for item in row_telemetry),
                12,
            ),
            "cost_reported_calls": sum(
                int(item.get("cost_reported_calls", 0)) for item in row_telemetry
            ),
            "input_tokens_lower_bound": sum(
                int(item.get("input_tokens_lower_bound", 0)) for item in row_telemetry
            ),
            "input_tokens_reported_calls": sum(
                int(item.get("input_tokens_reported_calls", 0)) for item in row_telemetry
            ),
            "output_tokens_lower_bound": sum(
                int(item.get("output_tokens_lower_bound", 0)) for item in row_telemetry
            ),
            "output_tokens_reported_calls": sum(
                int(item.get("output_tokens_reported_calls", 0)) for item in row_telemetry
            ),
            "reasoning_tokens_lower_bound": sum(
                int(item.get("reasoning_tokens_lower_bound", 0)) for item in row_telemetry
            ),
            "reasoning_tokens_reported_calls": sum(
                int(item.get("reasoning_tokens_reported_calls", 0)) for item in row_telemetry
            ),
            "total_tokens_lower_bound": sum(
                int(item.get("total_tokens_lower_bound", 0)) for item in row_telemetry
            ),
            "total_tokens_reported_calls": sum(
                int(item.get("total_tokens_reported_calls", 0)) for item in row_telemetry
            ),
            "latency_seconds_total": round(row_latency_total, 6),
            "latency_seconds_mean": (
                round(row_latency_total / row_calls, 6) if row_calls else None
            ),
            "latency_seconds_max": (
                round(max(row_latency_max_values), 6) if row_latency_max_values else None
            ),
            "attempts_lower_bound": sum(
                int(item.get("attempts_lower_bound", 0)) for item in row_telemetry
            ),
            "retries_lower_bound": sum(
                int(item.get("retries_lower_bound", 0)) for item in row_telemetry
            ),
            "tables_considered": sum(int(item.get("tables_considered", 0)) for item in row_plans),
            "dense_tables": sum(int(item.get("dense_tables", 0)) for item in row_plans),
            "rows_planned": sum(int(item.get("rows_planned", 0)) for item in row_plans),
            "unbatchable_rows": sum(int(item.get("unbatchable_rows", 0)) for item in row_plans),
            "base_batches": sum(int(item.get("base_batches", 0)) for item in row_plans),
            "maximum_calls": sum(int(item.get("maximum_calls", 0)) for item in row_plans),
            "batches_resumed": sum(int(item.get("batches_resumed", 0)) for item in row_execution),
        },
        "tuple_resolution": {
            "basis": (
                "completed tuple-resolution calls across the split, including validated "
                "checkpoint reuse and response-validation failures"
            ),
            "calls": tuple_calls,
            "cost_usd_lower_bound": round(
                sum(float(item.get("cost_usd_lower_bound", 0.0)) for item in tuple_telemetry),
                12,
            ),
            "cost_reported_calls": sum(
                int(item.get("cost_reported_calls", 0)) for item in tuple_telemetry
            ),
            "input_tokens_lower_bound": sum(
                int(item.get("input_tokens_lower_bound", 0)) for item in tuple_telemetry
            ),
            "input_tokens_reported_calls": sum(
                int(item.get("input_tokens_reported_calls", 0)) for item in tuple_telemetry
            ),
            "output_tokens_lower_bound": sum(
                int(item.get("output_tokens_lower_bound", 0)) for item in tuple_telemetry
            ),
            "output_tokens_reported_calls": sum(
                int(item.get("output_tokens_reported_calls", 0)) for item in tuple_telemetry
            ),
            "reasoning_tokens_lower_bound": sum(
                int(item.get("reasoning_tokens_lower_bound", 0)) for item in tuple_telemetry
            ),
            "reasoning_tokens_reported_calls": sum(
                int(item.get("reasoning_tokens_reported_calls", 0)) for item in tuple_telemetry
            ),
            "total_tokens_lower_bound": sum(
                int(item.get("total_tokens_lower_bound", 0)) for item in tuple_telemetry
            ),
            "total_tokens_reported_calls": sum(
                int(item.get("total_tokens_reported_calls", 0)) for item in tuple_telemetry
            ),
            "latency_seconds_total": round(tuple_latency_total, 6),
            "latency_seconds_mean": (
                round(tuple_latency_total / tuple_calls, 6) if tuple_calls else None
            ),
            "latency_seconds_max": (
                round(max(tuple_latency_max_values), 6) if tuple_latency_max_values else None
            ),
            "attempts_lower_bound": sum(
                int(item.get("attempts_lower_bound", 0)) for item in tuple_telemetry
            ),
            "retries_lower_bound": sum(
                int(item.get("retries_lower_bound", 0)) for item in tuple_telemetry
            ),
            "candidates_selected": sum(
                int(item.get("candidates_selected", 0)) for item in tuple_execution
            ),
            "candidates_passed": sum(
                int(item.get("candidates_passed", 0)) for item in tuple_execution
            ),
            "candidates_routed_to_review": sum(
                int(item.get("candidates_routed_to_review", 0)) for item in tuple_execution
            ),
            "candidates_resumed": sum(
                int(item.get("candidates_resumed", 0)) for item in tuple_execution
            ),
        },
        "verifier": {
            "basis": (
                "completed independent-verifier calls across the split, including validated "
                "checkpoint reuse and response-validation failures; monetary, token, retry, "
                "and attempt totals are lower bounds"
            ),
            "calls": verifier_calls,
            "cost_usd_lower_bound": round(
                sum(float(item.get("cost_usd_lower_bound", 0.0)) for item in verifier_telemetry),
                12,
            ),
            "cost_reported_calls": sum(
                int(item.get("cost_reported_calls", 0)) for item in verifier_telemetry
            ),
            "input_tokens_lower_bound": sum(
                int(item.get("input_tokens_lower_bound", 0)) for item in verifier_telemetry
            ),
            "input_tokens_reported_calls": sum(
                int(item.get("input_tokens_reported_calls", 0)) for item in verifier_telemetry
            ),
            "output_tokens_lower_bound": sum(
                int(item.get("output_tokens_lower_bound", 0)) for item in verifier_telemetry
            ),
            "output_tokens_reported_calls": sum(
                int(item.get("output_tokens_reported_calls", 0)) for item in verifier_telemetry
            ),
            "reasoning_tokens_lower_bound": sum(
                int(item.get("reasoning_tokens_lower_bound", 0)) for item in verifier_telemetry
            ),
            "reasoning_tokens_reported_calls": sum(
                int(item.get("reasoning_tokens_reported_calls", 0)) for item in verifier_telemetry
            ),
            "total_tokens_lower_bound": sum(
                int(item.get("total_tokens_lower_bound", 0)) for item in verifier_telemetry
            ),
            "total_tokens_reported_calls": sum(
                int(item.get("total_tokens_reported_calls", 0)) for item in verifier_telemetry
            ),
            "latency_seconds_total": round(verifier_latency_total, 6),
            "latency_seconds_mean": (
                round(verifier_latency_total / verifier_calls, 6) if verifier_calls else None
            ),
            "latency_seconds_max": (
                round(max(verifier_latency_max_values), 6) if verifier_latency_max_values else None
            ),
            "attempts_lower_bound": sum(
                int(item.get("attempts_lower_bound", 0)) for item in verifier_telemetry
            ),
            "retries_lower_bound": sum(
                int(item.get("retries_lower_bound", 0)) for item in verifier_telemetry
            ),
            "candidates_verified": sum(
                int(item.get("candidates_verified", 0)) for item in verifier_execution
            ),
            "candidates_failed": sum(
                int(item.get("candidates_failed", 0)) for item in verifier_execution
            ),
            "candidates_resumed": sum(
                int(item.get("candidates_resumed", 0)) for item in verifier_execution
            ),
            "candidates_executed": sum(
                int(item.get("candidates_executed", 0)) for item in verifier_execution
            ),
        },
        "origin_retrieval": {
            "basis": (
                "completed producer-origin calls across the split, including validated "
                "checkpoint reuse and response-validation failures"
            ),
            "calls": origin_calls,
            "cost_usd_lower_bound": round(
                sum(float(item.get("cost_usd_lower_bound", 0.0)) for item in origin_telemetry),
                12,
            ),
            "cost_reported_calls": sum(
                int(item.get("cost_reported_calls", 0)) for item in origin_telemetry
            ),
            "input_tokens_lower_bound": sum(
                int(item.get("input_tokens_lower_bound", 0)) for item in origin_telemetry
            ),
            "input_tokens_reported_calls": sum(
                int(item.get("input_tokens_reported_calls", 0)) for item in origin_telemetry
            ),
            "output_tokens_lower_bound": sum(
                int(item.get("output_tokens_lower_bound", 0)) for item in origin_telemetry
            ),
            "output_tokens_reported_calls": sum(
                int(item.get("output_tokens_reported_calls", 0)) for item in origin_telemetry
            ),
            "reasoning_tokens_lower_bound": sum(
                int(item.get("reasoning_tokens_lower_bound", 0)) for item in origin_telemetry
            ),
            "reasoning_tokens_reported_calls": sum(
                int(item.get("reasoning_tokens_reported_calls", 0)) for item in origin_telemetry
            ),
            "total_tokens_lower_bound": sum(
                int(item.get("total_tokens_lower_bound", 0)) for item in origin_telemetry
            ),
            "total_tokens_reported_calls": sum(
                int(item.get("total_tokens_reported_calls", 0)) for item in origin_telemetry
            ),
            "latency_seconds_total": round(origin_latency_total, 6),
            "latency_seconds_mean": (
                round(origin_latency_total / origin_calls, 6) if origin_calls else None
            ),
            "latency_seconds_max": (
                round(max(origin_latency_max_values), 6) if origin_latency_max_values else None
            ),
            "attempts_lower_bound": sum(
                int(item.get("attempts_lower_bound", 0)) for item in origin_telemetry
            ),
            "retries_lower_bound": sum(
                int(item.get("retries_lower_bound", 0)) for item in origin_telemetry
            ),
            "candidates_selected": sum(
                int(item.get("candidates_selected", 0)) for item in origin_execution
            ),
            "candidates_resumed": sum(
                int(item.get("candidates_resumed", 0)) for item in origin_execution
            ),
            "candidates_failed": sum(
                int(item.get("candidates_failed", 0)) for item in origin_execution
            ),
        },
    }


def _manifest_path(settings: PipelineSettings, paper_id: str) -> Path:
    return settings.output_root / paper_id / "source-manifest.json"


def freeze_paper(
    spec: PaperSpec,
    settings: PipelineSettings,
    *,
    before_request: Callable[[], None] | None = None,
) -> SourceManifest:
    """Reuse an already pinned manifest or download exact source bytes once.

    `before_request` is called immediately before each outbound download, so a bulk
    freeze can pace itself against one origin.
    """

    path = _manifest_path(settings, spec.paper_id)
    paper_location = str(spec.pdf_url) if spec.pdf_url is not None else str(spec.pdf_path)
    configured_sources: list[tuple[SourceRole, str, str | None]] = [
        (SourceRole.PAPER, paper_location, None),
        *[(SourceRole.SUPPLEMENT, str(url), None) for url in spec.supplement_urls],
    ]
    if spec.repository_url and spec.repository_commit:
        configured_sources.append(
            (SourceRole.REPOSITORY, str(spec.repository_url), spec.repository_commit.casefold())
        )
    if path.exists():
        manifest = SourceManifest.model_validate(read_json(path))
        frozen_contract = [
            (source.role, source.original_uri, source.git_commit) for source in manifest.sources
        ]
        if frozen_contract != configured_sources:
            raise ValueError(
                f"configured source bundle changed for {spec.paper_id}; "
                f"frozen={frozen_contract!r}, configured={configured_sources!r}"
            )
        for source in manifest.sources:
            if source.role == SourceRole.REPOSITORY:
                if source.git_commit is None:
                    raise ValueError(f"repository source is not commit-pinned for {spec.paper_id}")
                continue
            resolve_cached_path(source, settings.project_root)
        return manifest
    cache_root = settings.project_root / "data" / "sources"
    if spec.pdf_path is not None:
        paper_source = freeze_local_source(
            paper_id=spec.paper_id,
            role=SourceRole.PAPER,
            path=Path(spec.pdf_path),
            cache_root=cache_root,
            original_uri=spec.pdf_path,
            license_disposition=LicenseDisposition.DERIVED_METADATA_ONLY,
        )
    else:
        paper_source = download_and_freeze_source(
            paper_id=spec.paper_id,
            role=SourceRole.PAPER,
            url=str(spec.pdf_url),
            cache_root=cache_root,
            license_disposition=LicenseDisposition.DERIVED_METADATA_ONLY,
            before_request=before_request,
        )
    sources = [paper_source]
    for supplement_url in spec.supplement_urls:
        sources.append(
            download_and_freeze_source(
                paper_id=spec.paper_id,
                role=SourceRole.SUPPLEMENT,
                url=str(supplement_url),
                cache_root=cache_root,
                license_disposition=LicenseDisposition.DERIVED_METADATA_ONLY,
                before_request=before_request,
            )
        )
    if spec.repository_url and spec.repository_commit:
        sources.append(
            freeze_repository_source(
                paper_id=spec.paper_id,
                url=str(spec.repository_url),
                git_commit=spec.repository_commit,
                license_disposition=LicenseDisposition.UNKNOWN,
            )
        )
    manifest = SourceManifest(
        paper_id=spec.paper_id,
        title=spec.title,
        doi=spec.doi,
        arxiv_id=spec.arxiv_id,
        proceedings_url=spec.acm_url,
        sources=sources,
    )
    write_json(path, manifest)
    return manifest


def _code_state(project_root: Path) -> dict[str, str | bool]:
    git_available = True
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "uncommitted"
        git_available = False
    try:
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "src",
                "configs",
                "schemas",
                "pyproject.toml",
                "uv.lock",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        status = ""
        git_available = False
    semantic_paths: list[tuple[str, Path]] = []
    if git_available:
        try:
            listed = subprocess.run(
                [
                    "git",
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "--",
                    "src",
                    "configs",
                ],
                cwd=project_root,
                check=True,
                capture_output=True,
                timeout=15,
            ).stdout
            semantic_paths = [
                (raw.decode("utf-8"), project_root / raw.decode("utf-8"))
                for raw in listed.split(b"\0")
                if raw
            ]
        except (subprocess.CalledProcessError, FileNotFoundError, UnicodeDecodeError):
            git_available = False
    if not semantic_paths:
        for base in (project_root / "src", project_root / "configs"):
            if not base.exists():
                continue
            semantic_paths.extend(
                (path.relative_to(project_root).as_posix(), path)
                for path in base.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            )
    if not semantic_paths:
        package_root = Path(__file__).resolve().parent
        semantic_paths.extend(
            (f"installed-package/{path.relative_to(package_root).as_posix()}", path)
            for path in package_root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    source_hash = hashlib.sha256()
    for label, path in sorted(set(semantic_paths)):
        if not path.is_file() or path.is_symlink():
            continue
        source_hash.update(label.encode())
        source_hash.update(path.read_bytes())
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_available": git_available,
        "source_tree_sha256": source_hash.hexdigest(),
    }


def _select_pages(layout: PdfLayout, spec: PaperSpec):
    if spec.include_pages:
        missing = [page for page in spec.include_pages if page < 1 or page > layout.page_count]
        if missing:
            raise ValueError(f"configured pages outside PDF for {spec.paper_id}: {missing}")
        return [layout.pages[page - 1] for page in spec.include_pages]
    return select_result_pages(layout, limit=spec.max_result_pages)


def _block_fragment(block: ResultBlock):
    """Adapt a bounded block to the existing source-fragment extraction contract."""

    text = block.prompt_text()
    return PageFragment(
        fragment_id=block.block_id,
        source_id=block.source_id,
        page=block.page,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=block.numeric_token_count,
        result_signal_score=block.result_signal_score,
    )


def _safe_error_message(error: Exception) -> str:
    """Return an allowlisted diagnostic without echoing arbitrary exception text."""

    if isinstance(error, ProviderBudgetExhausted):
        return f"provider budget stopped before dispatch: {error.reason}"
    if isinstance(error, ProviderBudgetContractError):
        return "provider budget ledger or contract validation failed"
    if isinstance(error, ProviderResponseValidationError):
        return f"provider response validation failed: {error.code}"
    if isinstance(error, ProviderRequestRejectedError):
        return f"provider request rejected with HTTP {error.status_code}"
    if isinstance(error, FileNotFoundError) and Path(error.filename or "").name == "pdftotext":
        return "required local PDF parser 'pdftotext' is unavailable"
    if isinstance(error, subprocess.TimeoutExpired):
        return "local PDF parser timed out"
    if isinstance(error, subprocess.CalledProcessError):
        return "local PDF parser failed"
    fixed_messages = {
        "OPENROUTER_API_KEY must be supplied in the runtime environment",
        "extractor model is required",
        "tuple model must be non-empty",
        "tuple model must be non-empty when supplied",
        "verifier model must be non-empty when supplied",
        "origin model must be non-empty when supplied",
        "verifier_model requires tuple_model",
        "origin_model requires verifier_model",
        "row_model requires row_enumeration_enabled",
        "row_model requires --row-enumeration",
        "row model must be non-empty",
        "row model must be non-empty when supplied",
    }
    if str(error) in fixed_messages:
        return str(error)
    safe_categories = {
        ValueError: "invalid or inconsistent input",
        FileNotFoundError: "required local input is unavailable",
        PermissionError: "required local input is not readable",
        OSError: "local input/output operation failed",
        RuntimeError: "runtime operation failed",
    }
    for error_type, message in safe_categories.items():
        if isinstance(error, error_type):
            return message
    return f"{type(error).__name__}: operation failed; details suppressed"


def _reference_path(project_root: Path, configured_path: str) -> Path:
    root = project_root.resolve()
    path = (root / configured_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("reference path escaped project root")
    return path


def _score_reference_after_paper_error(
    *,
    spec: PaperSpec,
    settings: PipelineSettings,
    manifest_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Retain frozen reference denominators when a paper fails after source freeze."""

    if spec.reference_path is None:
        return None, None
    manifest = SourceManifest.model_validate(read_json(manifest_path))
    paper_source = next(source for source in manifest.sources if source.role == SourceRole.PAPER)
    reference = load_reference(_reference_path(settings.project_root, spec.reference_path))
    if reference.paper_id != spec.paper_id:
        raise ValueError("reference paper_id does not match the failed paper")
    if reference.source_sha256 != paper_source.sha256:
        raise ValueError("reference source hash does not match the failed paper")
    score = score_reference(reference, [])
    return score, write_json(output_path, score)


def _clear_paper_run_outputs(output_dir: Path) -> None:
    """Remove only known generated outputs before rebuilding one paper run."""

    for name in _PAPER_RUN_OUTPUTS:
        (output_dir / name).unlink(missing_ok=True)
    # These describe the current invocation, unlike the contract-bound checkpoint
    # retained for a later opt-in resume. Leaving them behind when the row stage is
    # disabled makes offline coverage mistake a prior ledger for the current run.
    for name in (
        "row-enumeration-plan.json",
        "row-enumeration-preflight.json",
        "row-enumeration.json",
        "row-terminal-states.json",
        "tuple-resolution.json",
        "verifier-gates.json",
        "origin-retrieval.json",
    ):
        (output_dir / "private" / name).unlink(missing_ok=True)
    eee_dir = output_dir / "eee"
    if eee_dir.is_symlink():
        raise RuntimeError("refusing to clean a symlinked EEE output directory")
    if eee_dir.is_dir():
        for record_path in eee_dir.glob("*.json"):
            if record_path.is_file():
                record_path.unlink()
    (output_dir / "private" / "invalid-eee.json").unlink(missing_ok=True)


def _clear_corpus_run_outputs(output_root: Path) -> None:
    """Remove only corpus aggregates that belong to a previous invocation."""

    for name in _CORPUS_RUN_OUTPUTS:
        (output_root / name).unlink(missing_ok=True)


def _extractor_checkpoint_contract(
    *,
    spec: PaperSpec,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    block_config: ResultBlockConfig,
    blocks: list[ResultBlock],
    code_state: dict[str, str | bool],
) -> dict[str, Any]:
    """Bind reusable block results to every input that can change extraction."""

    return {
        "schema_version": _EXTRACTOR_CHECKPOINT_CONTRACT_VERSION,
        "paper_id": spec.paper_id,
        "paper_title": spec.title,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_parser": layout.parser,
        "layout_parser_version": layout.parser_version,
        "result_block_segmentation": asdict(block_config),
        "blocks": [
            {
                "block_id": block.block_id,
                "source_id": block.source_id,
                "page": block.page,
                "text_sha256": block.text_sha256,
            }
            for block in blocks
        ],
        "extractor": _extractor_run_configuration(settings),
        "code": code_state,
    }


def _new_extractor_checkpoint(contract: dict[str, Any], contract_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": _EXTRACTOR_CHECKPOINT_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "blocks": {},
        "recoveries": {},
    }


def _checkpoint_reuse_envelope(contract: dict[str, Any]) -> dict[str, Any]:
    """Return inputs that must match before any prior block can be reused.

    The complete block set may change while an individual block request remains
    byte-for-byte equivalent, but repair-04 deliberately binds reuse to the exact
    code state. The block text hash is checked separately during rehydration.
    """

    keys = (
        "schema_version",
        "paper_id",
        "paper_title",
        "source_manifest_sha256",
        "layout_parser",
        "layout_parser_version",
        "extractor",
        "code",
    )
    return {key: contract.get(key) for key in keys}


def _migrate_compatible_checkpoint_blocks(
    payload: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    """Carry forward only exact block results under an equivalent request envelope."""

    previous_contract = payload.get("contract")
    previous_blocks = payload.get("blocks")
    previous_recoveries = payload.get("recoveries", {})
    if not isinstance(previous_contract, dict) or not isinstance(previous_blocks, dict):
        return _new_extractor_checkpoint(contract, contract_sha256)
    if not isinstance(previous_recoveries, dict):
        previous_recoveries = {}
    if _checkpoint_reuse_envelope(previous_contract) != _checkpoint_reuse_envelope(contract):
        return _new_extractor_checkpoint(contract, contract_sha256)
    current_hashes = {
        block["block_id"]: block["text_sha256"]
        for block in contract.get("blocks", [])
        if isinstance(block, dict)
        and isinstance(block.get("block_id"), str)
        and isinstance(block.get("text_sha256"), str)
    }
    checkpoint = _new_extractor_checkpoint(contract, contract_sha256)
    checkpoint["blocks"] = {
        block_id: entry
        for block_id, entry in previous_blocks.items()
        if block_id in current_hashes
        and isinstance(entry, dict)
        and entry.get("block_text_sha256") == current_hashes[block_id]
    }
    checkpoint["recoveries"] = {
        block_id: entry
        for block_id, entry in previous_recoveries.items()
        if block_id in current_hashes
        and isinstance(entry, dict)
        and entry.get("block_text_sha256") == current_hashes[block_id]
    }
    return checkpoint


def _load_extractor_checkpoint(
    path: Path,
    *,
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    """Load exact state or migrate exact blocks under a compatible request envelope."""

    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return _new_extractor_checkpoint(contract, contract_sha256)
    if not isinstance(payload, dict):
        return _new_extractor_checkpoint(contract, contract_sha256)
    if payload.get("schema_version") != _EXTRACTOR_CHECKPOINT_SCHEMA_VERSION:
        return _new_extractor_checkpoint(contract, contract_sha256)
    if (
        payload.get("contract_sha256") == contract_sha256
        and payload.get("contract") == contract
        and isinstance(payload.get("blocks"), dict)
    ):
        if not isinstance(payload.get("recoveries"), dict):
            payload["recoveries"] = {}
        return payload
    return _migrate_compatible_checkpoint_blocks(
        payload,
        contract=contract,
        contract_sha256=contract_sha256,
    )


def _legacy_candidate_matches_attempt(
    candidate: CandidateObservation,
    *,
    attempted_block: ResultBlock,
    call: ProviderCall,
    spec: PaperSpec,
    settings: PipelineSettings,
) -> bool:
    """Bind one retained legacy candidate to its exact producing request and block."""

    if (
        candidate.paper_id != spec.paper_id
        or candidate.extraction_method != f"openrouter:{settings.model}"
        or candidate.raw_payload_hash != call.response_sha256
        or len(candidate.proposal_traces) != 1
        or not candidate.evidence
    ):
        return False
    trace = candidate.proposal_traces[0]
    if (
        trace.stage is not ProposalStage.LEGACY_BLOCK
        or trace.paper_id != spec.paper_id
        or trace.source_id != attempted_block.source_id
        or trace.page != attempted_block.page
        or trace.fragment_id != attempted_block.block_id
        or trace.batch_id is not None
        or trace.planned_row_id is not None
    ):
        return False
    return all(
        anchor.source_id == attempted_block.source_id and anchor.page == attempted_block.page
        for anchor in candidate.evidence
    )


def _validated_checkpoint_entry(
    entry: Any,
    *,
    block: ResultBlock,
    spec: PaperSpec,
    settings: PipelineSettings,
    contract: dict[str, Any],
) -> (
    tuple[
        list[CandidateObservation],
        list[ProviderCall],
        list[ProviderCall],
        list[str],
    ]
    | None
):
    """Rehydrate one successful block result without trusting private JSON."""

    expected_block_contract = {
        "block_id": block.block_id,
        "source_id": block.source_id,
        "page": block.page,
        "text_sha256": block.text_sha256,
    }
    if (
        not isinstance(entry, dict)
        or entry.get("block_text_sha256") != block.text_sha256
        or contract.get("paper_id") != spec.paper_id
        or contract.get("paper_title") != spec.title
        or contract.get("extractor") != _extractor_run_configuration(settings)
        or expected_block_contract not in contract.get("blocks", [])
    ):
        return None
    raw_candidates = entry.get("candidates")
    raw_warnings = entry.get("warnings")
    raw_attempts = entry.get("attempts")
    raw_successful_indexes = entry.get("successful_call_indexes")
    if (
        not isinstance(raw_candidates, list)
        or not isinstance(raw_warnings, list)
        or not isinstance(raw_attempts, list)
        or not raw_attempts
        or not isinstance(raw_successful_indexes, list)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in raw_successful_indexes
        )
        or len(raw_successful_indexes) != len(set(raw_successful_indexes))
    ):
        return None
    if not all(isinstance(warning, str) for warning in raw_warnings):
        return None
    try:
        candidates = [CandidateObservation.model_validate(item) for item in raw_candidates]
        calls = [ProviderCall.model_validate(item) for item in entry.get("calls", [])]
    except (TypeError, ValueError):
        return None
    if any(index >= len(calls) for index in raw_successful_indexes) or not all(
        isinstance(warning, str) for warning in raw_warnings
    ):
        return None

    parsed_attempts: list[dict[str, Any]]
    if (
        len(raw_attempts) == 1
        and isinstance(raw_attempts[0], dict)
        and raw_attempts[0].get("status") == "success"
    ):
        raw_attempt = raw_attempts[0]
        try:
            direct_call = ProviderCall.model_validate(raw_attempt.get("call"))
            direct_candidates = [
                CandidateObservation.model_validate(item)
                for item in raw_attempt.get("candidates", [])
            ]
        except (AttributeError, TypeError, ValueError):
            return None
        direct_warnings = raw_attempt.get("warnings")
        if (
            raw_attempt.get("block_id") != block.block_id
            or raw_attempt.get("block_text_sha256") != block.text_sha256
            or raw_attempt.get("depth") != 0
            or raw_attempt.get("completed_provider_call") is not True
            or raw_attempt.get("safe_details") != {}
            or not isinstance(direct_warnings, list)
            or not all(isinstance(warning, str) for warning in direct_warnings)
            or not _legacy_call_matches(
                direct_call,
                block=block,
                spec=spec,
                settings=settings,
            )
            or any(
                not _legacy_candidate_matches_attempt(
                    candidate,
                    attempted_block=block,
                    call=direct_call,
                    spec=spec,
                    settings=settings,
                )
                for candidate in direct_candidates
            )
        ):
            return None
        parsed_attempts = [
            {
                "block": block,
                "status": "success",
                "call": direct_call,
                "candidates": direct_candidates,
                "warnings": direct_warnings,
            }
        ]
    else:
        validated_recovery = _validated_legacy_recovery_entry(
            {
                "schema_version": _EXTRACTOR_RECOVERY_ENTRY_VERSION,
                "block_text_sha256": block.text_sha256,
                "max_depth": LEGACY_RECOVERY_MAX_DEPTH,
                "attempts": raw_attempts,
            },
            block=block,
            spec=spec,
            settings=settings,
        )
        if validated_recovery is None or not validated_recovery[1]:
            return None
        parsed_attempts = validated_recovery[0]
        if any(
            attempt["status"] != "success"
            and (
                not attempt["status"].startswith("provider_response_")
                or attempt["depth"] >= LEGACY_RECOVERY_MAX_DEPTH
                or not split_result_block(attempt["block"])
            )
            for attempt in parsed_attempts
        ):
            return None

    expected_calls = [attempt["call"] for attempt in parsed_attempts if attempt["call"] is not None]
    expected_successful_indexes = [
        index
        for index, attempt in enumerate(
            attempt for attempt in parsed_attempts if attempt["call"] is not None
        )
        if attempt["status"] == "success"
    ]
    expected_candidates = [
        candidate
        for attempt in parsed_attempts
        if attempt["status"] == "success"
        for candidate in attempt["candidates"]
    ]
    expected_warnings = [
        warning
        for attempt in parsed_attempts
        if attempt["status"] == "success"
        for warning in attempt["warnings"]
    ]
    if (
        calls != expected_calls
        or raw_successful_indexes != expected_successful_indexes
        or candidates != expected_candidates
        or raw_warnings != expected_warnings
        or not expected_successful_indexes
    ):
        return None
    successful_calls = [calls[index] for index in raw_successful_indexes]
    return candidates, calls, successful_calls, raw_warnings


def _legacy_call_matches(
    call: ProviderCall,
    *,
    block: ResultBlock,
    spec: PaperSpec,
    settings: PipelineSettings,
    require_returned_model_match: bool = True,
) -> bool:
    request = extractor_request_contract(
        seed=settings.seed,
        model=settings.model,
        max_tokens=settings.max_tokens,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": page_prompt(
                paper_title=spec.title,
                paper_id=spec.paper_id,
                fragment=_block_fragment(block),
            ),
        },
    ]
    expected_prompt_sha256 = hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    schema = request["schema"]
    privacy = request["privacy"]
    routing = request["routing"]
    return all(
        (
            call.provider == "openrouter",
            call.model_requested == settings.model,
            not require_returned_model_match or call.model_returned == settings.model,
            call.prompt_sha256 == expected_prompt_sha256,
            call.temperature == settings.temperature,
            call.reasoning_effort == settings.reasoning_effort,
            call.max_tokens == settings.max_tokens,
            call.completion_token_parameter
            == request["completion_token_parameter"]
            == completion_token_parameter_for_model(settings.model),
            call.seed == request["seed"],
            call.response_format == schema["response_format"],
            call.schema_name == schema["schema_name"],
            call.schema_sha256 == schema["schema_sha256"],
            call.schema_strict == schema["schema_strict"],
            call.data_collection == privacy["data_collection"],
            call.require_parameters == routing["require_parameters"],
            call.zdr == privacy["zdr"],
        )
    )


def _legacy_recovery_attempt_payload(
    *,
    block: ResultBlock,
    depth: int,
    status: str,
    call: ProviderCall | None,
    candidates: list[CandidateObservation] | None = None,
    warnings: list[str] | None = None,
    safe_details: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "block_text_sha256": block.text_sha256,
        "depth": depth,
        "status": status,
        "completed_provider_call": call is not None,
        "call": call.model_dump(mode="json") if call is not None else None,
        "candidates": [
            candidate.model_dump(mode="json", by_alias=True, exclude_none=True)
            for candidate in (candidates or [])
        ],
        "warnings": list(warnings or []),
        "safe_details": dict(safe_details or {}),
    }


def _validated_legacy_recovery_entry(
    entry: Any,
    *,
    block: ResultBlock,
    spec: PaperSpec,
    settings: PipelineSettings,
    max_depth: int = LEGACY_RECOVERY_MAX_DEPTH,
) -> tuple[list[dict[str, Any]], bool] | None:
    """Validate one exact depth-first prefix of a legacy split-recovery tree."""

    if (
        not isinstance(entry, dict)
        or entry.get("schema_version") != _EXTRACTOR_RECOVERY_ENTRY_VERSION
        or entry.get("block_text_sha256") != block.text_sha256
        or entry.get("max_depth") != max_depth
        or not isinstance(entry.get("attempts"), list)
        or not entry["attempts"]
    ):
        return None
    raw_attempts = entry["attempts"]
    maximum_attempts = 2 ** (max_depth + 1) - 1
    if len(raw_attempts) > maximum_attempts:
        return None
    validated: list[dict[str, Any]] = []

    def parse_attempt(raw: Any, *, expected: ResultBlock, depth: int) -> dict[str, Any]:
        if (
            not isinstance(raw, dict)
            or raw.get("block_id") != expected.block_id
            or raw.get("block_text_sha256") != expected.text_sha256
            or raw.get("depth") != depth
            or not isinstance(raw.get("status"), str)
            or not isinstance(raw.get("completed_provider_call"), bool)
            or not isinstance(raw.get("candidates"), list)
            or not isinstance(raw.get("warnings"), list)
            or not all(isinstance(item, str) for item in raw["warnings"])
            or not isinstance(raw.get("safe_details"), dict)
            or any(
                not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int)
                for key, value in raw["safe_details"].items()
            )
        ):
            raise ValueError("legacy recovery attempt shape mismatch")
        status = raw["status"]
        completed = status == "success" or status.startswith("provider_response_")
        if raw["completed_provider_call"] is not completed:
            raise ValueError("legacy recovery completion flag mismatch")
        raw_call = raw.get("call")
        call = ProviderCall.model_validate(raw_call) if completed else None
        returned_model_mismatch = status == _RETURNED_MODEL_MISMATCH
        if (
            (raw_call is not None) != completed
            or (
                call is not None
                and not _legacy_call_matches(
                    call,
                    block=expected,
                    spec=spec,
                    settings=settings,
                    require_returned_model_match=not returned_model_mismatch,
                )
            )
            or (returned_model_mismatch and (call is None or call.model_returned == settings.model))
        ):
            raise ValueError("legacy recovery call does not match its exact input")
        candidates = [CandidateObservation.model_validate(item) for item in raw["candidates"]]
        warnings = list(raw["warnings"])
        safe_details = dict(raw["safe_details"])
        if status == "success":
            assert call is not None
            if safe_details:
                raise ValueError("successful legacy recovery cannot contain failure details")
            if any(
                not _legacy_candidate_matches_attempt(
                    candidate,
                    attempted_block=expected,
                    call=call,
                    spec=spec,
                    settings=settings,
                )
                for candidate in candidates
            ):
                raise ValueError("legacy recovery candidate is not bound to its response")
        elif status in {
            "provider_response_invalid_json",
            "provider_response_schema_validation",
            "provider_response_wire_validation",
            _RETURNED_MODEL_MISMATCH,
        }:
            if candidates or warnings or safe_details:
                raise ValueError("validation failure contains noncanonical recovery payload")
        elif status == "provider_request_rejected":
            if completed or candidates or warnings or set(safe_details) != {"http_status"}:
                raise ValueError("request rejection contains noncanonical recovery payload")
        elif status == "extractor_block_failed":
            if completed or candidates or warnings or safe_details:
                raise ValueError("transport failure contains noncanonical recovery payload")
        else:
            raise ValueError("unknown legacy recovery attempt status")
        return {
            "block": expected,
            "depth": depth,
            "status": status,
            "call": call,
            "candidates": candidates,
            "warnings": warnings,
            "safe_details": safe_details,
        }

    try:
        base = parse_attempt(raw_attempts[0], expected=block, depth=0)
        if not base["status"].startswith("provider_response_"):
            return None
        validated.append(base)
        index = 1

        def visit(expected: ResultBlock, depth: int) -> bool:
            nonlocal index
            if index >= len(raw_attempts):
                return False
            parsed = parse_attempt(raw_attempts[index], expected=expected, depth=depth)
            validated.append(parsed)
            index += 1
            descendants = split_result_block(expected)
            if (
                parsed["status"].startswith("provider_response_")
                and depth < max_depth
                and descendants
            ):
                for descendant in descendants:
                    if not visit(descendant, depth + 1):
                        return False
            return True

        tree_complete = True
        for child in split_result_block(block):
            if not visit(child, 1):
                tree_complete = False
                break
        if index != len(raw_attempts):
            return None
    except (AssertionError, TypeError, ValueError):
        return None
    return validated, tree_complete


def _row_checkpoint_contract(
    *,
    spec: PaperSpec,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    plan_sha256: str,
    code_state: dict[str, str | bool],
) -> dict[str, Any]:
    """Bind reusable row outcomes to source, plan, provider contract, and code."""

    return {
        "schema_version": _ROW_CHECKPOINT_CONTRACT_VERSION,
        "paper_id": spec.paper_id,
        "paper_title": spec.title,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_parser": layout.parser,
        "layout_parser_version": layout.parser_version,
        "plan_sha256": plan_sha256,
        "row_enumeration": _row_enumeration_run_configuration(settings),
        "code": code_state,
    }


def _new_row_checkpoint(contract: dict[str, Any], contract_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": _ROW_CHECKPOINT_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "batches": {},
    }


def _row_checkpoint_reuse_envelope(contract: dict[str, Any]) -> dict[str, Any]:
    """Return request inputs that must match before a row batch can be reused.

    Repair-04 binds reuse to the exact code state as well as the exact plan, batch
    hashes, request envelope, and typed terminal outcome.
    """

    keys = (
        "schema_version",
        "paper_id",
        "paper_title",
        "source_manifest_sha256",
        "layout_parser",
        "layout_parser_version",
        "plan_sha256",
        "row_enumeration",
        "code",
    )
    return {key: contract.get(key) for key in keys}


def _migrate_compatible_row_checkpoint_batches(
    payload: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_sha256: str,
    batches: list[RowBatch],
) -> dict[str, Any]:
    """Carry forward only exact, typed row batches under one request envelope."""

    previous_contract = payload.get("contract")
    previous_batches = payload.get("batches")
    if not isinstance(previous_contract, dict) or not isinstance(previous_batches, dict):
        return _new_row_checkpoint(contract, contract_sha256)
    if _row_checkpoint_reuse_envelope(previous_contract) != _row_checkpoint_reuse_envelope(
        contract
    ):
        return _new_row_checkpoint(contract, contract_sha256)
    current_batches = {batch.batch_id: batch for batch in batches}
    checkpoint = _new_row_checkpoint(contract, contract_sha256)
    checkpoint["batches"] = {
        batch_id: entry
        for batch_id, entry in previous_batches.items()
        if (batch := current_batches.get(batch_id)) is not None
        and (
            _validated_row_checkpoint_entry(entry, batch=batch, contract=contract) is not None
            or _validated_row_checkpoint_prefix(entry, batch=batch, contract=contract) is not None
        )
    }
    return checkpoint


def _load_row_checkpoint(
    path: Path,
    *,
    contract: dict[str, Any],
    contract_sha256: str,
    batches: list[RowBatch],
) -> dict[str, Any]:
    """Load exact state or migrate typed batches under a compatible request envelope."""

    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return _new_row_checkpoint(contract, contract_sha256)
    if not isinstance(payload, dict):
        return _new_row_checkpoint(contract, contract_sha256)
    if payload.get("schema_version") != _ROW_CHECKPOINT_SCHEMA_VERSION:
        return _new_row_checkpoint(contract, contract_sha256)
    if (
        payload.get("contract_sha256") == contract_sha256
        and payload.get("contract") == contract
        and isinstance(payload.get("batches"), dict)
    ):
        return payload
    return _migrate_compatible_row_checkpoint_batches(
        payload,
        contract=contract,
        contract_sha256=contract_sha256,
        batches=batches,
    )


def _row_outcome_payload(outcome: RowEnumerationOutcome) -> dict[str, Any]:
    return {
        "records": {
            row_id: record.model_dump(mode="json", by_alias=True, exclude_none=True)
            for row_id, record in sorted(outcome.records.items())
        },
        "calls": [call.model_dump(mode="json") for call in outcome.calls],
        "attempts": [attempt.model_dump(mode="json") for attempt in outcome.attempts],
        "unresolved_row_ids": outcome.unresolved_row_ids,
        "unbatchable_row_ids": outcome.unbatchable_row_ids,
        "unknown_row_ids": outcome.unknown_row_ids,
        "invalid_row_reasons": outcome.invalid_row_reasons,
        "warnings": outcome.warnings,
        "telemetry": outcome.telemetry,
    }


def _rehydrated_row_checkpoint_entry(
    entry: Any,
    *,
    batch: RowBatch,
    contract: dict[str, Any],
) -> tuple[RowEnumerationOutcome, RowEnumerationConfig, dict[str, Any]] | None:
    """Parse and bind the common structure of complete or prefix row state."""

    batch_sha256 = sha256_bytes(canonical_json_bytes(batch))
    if not isinstance(entry, dict) or entry.get("batch_sha256") != batch_sha256:
        return None
    row_ids = {row.row_id for row in batch.rows}
    raw_records = entry.get("records")
    list_fields = (
        "calls",
        "attempts",
        "unresolved_row_ids",
        "unbatchable_row_ids",
        "unknown_row_ids",
        "warnings",
    )
    if not isinstance(raw_records, dict) or any(
        not isinstance(entry.get(name), list) for name in list_fields
    ):
        return None
    if not isinstance(entry.get("invalid_row_reasons"), dict):
        return None
    row_configuration = contract.get("row_enumeration")
    if not isinstance(row_configuration, dict):
        return None
    try:
        config = RowEnumerationConfig.model_validate(row_configuration.get("limits"))
        outcome = RowEnumerationOutcome(
            records={
                row_id: RowDispositionRecord.model_validate(record)
                for row_id, record in raw_records.items()
            },
            calls=[ProviderCall.model_validate(call) for call in entry["calls"]],
            attempts=[RowAttemptTelemetry.model_validate(item) for item in entry["attempts"]],
            unresolved_row_ids=list(entry["unresolved_row_ids"]),
            unbatchable_row_ids=list(entry["unbatchable_row_ids"]),
            unknown_row_ids=list(entry["unknown_row_ids"]),
            invalid_row_reasons=dict(entry["invalid_row_reasons"]),
            warnings=list(entry["warnings"]),
        )
    except (TypeError, ValueError):
        return None
    record_ids = set(outcome.records)
    unresolved_ids = set(outcome.unresolved_row_ids)
    if (
        len(unresolved_ids) != len(outcome.unresolved_row_ids)
        or record_ids & unresolved_ids
        or record_ids | unresolved_ids != row_ids
        or outcome.unbatchable_row_ids
        or len(outcome.attempts) > 3
        or len(outcome.calls) > 3
        or entry.get("telemetry") != outcome.telemetry
        or any(
            not isinstance(row_id, str) or not isinstance(reason, str)
            for row_id, reason in outcome.invalid_row_reasons.items()
        )
        or set(outcome.invalid_row_reasons) - row_ids
        or any(
            not isinstance(item, str)
            for name in (
                "unresolved_row_ids",
                "unbatchable_row_ids",
                "unknown_row_ids",
                "warnings",
            )
            for item in entry[name]
        )
    ):
        return None
    if any(record.row_id != row_id for row_id, record in outcome.records.items()):
        return None
    expected_unknown = [
        row_id for attempt in outcome.attempts for row_id in attempt.unknown_row_ids
    ]
    if outcome.unknown_row_ids != expected_unknown:
        return None
    return outcome, config, row_configuration


def _validated_row_checkpoint_entry(
    entry: Any,
    *,
    batch: RowBatch,
    contract: dict[str, Any],
) -> RowEnumerationOutcome | None:
    """Rehydrate one complete bounded batch without trusting private JSON."""

    rehydrated = _rehydrated_row_checkpoint_entry(entry, batch=batch, contract=contract)
    if rehydrated is None:
        return None
    outcome, config, row_configuration = rehydrated
    # A request/transport failure has no durable provider response.  Keep it in the
    # just-finished invocation's checkpoint for typed diagnostics, but do not turn
    # that transient no-call outcome into a terminal cache hit on the next run.
    if any(not attempt.completed_provider_call for attempt in outcome.attempts):
        return None
    try:
        validate_batch_outcome_against(
            batch,
            outcome,
            config=config,
            paper_id=str(contract["paper_id"]),
            paper_title=str(contract["paper_title"]),
            model=str(row_configuration["model"]),
            max_tokens=int(row_configuration["max_tokens"]),
            temperature=row_configuration.get("temperature"),
            reasoning_effort=row_configuration.get("reasoning_effort"),
            seed=(
                int(row_configuration["seed"])
                if row_configuration.get("seed") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return outcome


def _validated_row_checkpoint_prefix(
    entry: Any,
    *,
    batch: RowBatch,
    contract: dict[str, Any],
) -> RowEnumerationOutcome | None:
    """Rehydrate a strict completed-attempt prefix for one exact base batch."""

    rehydrated = _rehydrated_row_checkpoint_entry(entry, batch=batch, contract=contract)
    if rehydrated is None:
        return None
    outcome, config, row_configuration = rehydrated
    try:
        partition_row_provider_calls(outcome)
    except ValueError:
        return None
    first_no_call = next(
        (
            index
            for index, attempt in enumerate(outcome.attempts)
            if not attempt.completed_provider_call
        ),
        None,
    )
    if first_no_call is not None:
        # Only a contiguous completed prefix is durable. A trailing request or
        # transport failure has no response-derived state, so discard that suffix
        # while preserving every paid attempt before it. A gap followed by another
        # completed response is not a prefix and remains fail-closed.
        if first_no_call == 0 or any(
            attempt.completed_provider_call for attempt in outcome.attempts[first_no_call + 1 :]
        ):
            return None
        outcome = RowEnumerationOutcome(
            records=dict(outcome.records),
            calls=list(outcome.calls),
            attempts=list(outcome.attempts[:first_no_call]),
            unresolved_row_ids=list(outcome.unresolved_row_ids),
            unbatchable_row_ids=list(outcome.unbatchable_row_ids),
            unknown_row_ids=list(outcome.unknown_row_ids),
            invalid_row_reasons=dict(outcome.invalid_row_reasons),
            warnings=list(outcome.warnings),
        )
    try:
        validate_batch_outcome_prefix(
            batch,
            outcome,
            config=config,
            paper_id=str(contract["paper_id"]),
            paper_title=str(contract["paper_title"]),
            model=str(row_configuration["model"]),
            max_tokens=int(row_configuration["max_tokens"]),
            temperature=row_configuration.get("temperature"),
            reasoning_effort=row_configuration.get("reasoning_effort"),
            seed=(
                int(row_configuration["seed"])
                if row_configuration.get("seed") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return outcome


def _tuple_checkpoint_contract(
    *,
    spec: PaperSpec,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    code_state: dict[str, str | bool],
) -> dict[str, Any]:
    """Bind tuple decisions to the exact immutable production inputs."""

    return {
        "schema_version": _TUPLE_CHECKPOINT_CONTRACT_VERSION,
        "paper_id": spec.paper_id,
        "paper_title": spec.title,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_sha256": layout_binding_sha256(layout),
        "tuple_resolution": _tuple_run_configuration(settings),
        "code": code_state,
    }


def _new_tuple_checkpoint(contract: dict[str, Any], contract_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": _TUPLE_CHECKPOINT_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "candidates": {},
    }


def _load_tuple_checkpoint(
    path: Path,
    *,
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return _new_tuple_checkpoint(contract, contract_sha256)
    if not isinstance(payload, dict):
        return _new_tuple_checkpoint(contract, contract_sha256)
    if (
        payload.get("schema_version") != _TUPLE_CHECKPOINT_SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256") != contract_sha256
        or not isinstance(payload.get("candidates"), dict)
    ):
        return _new_tuple_checkpoint(contract, contract_sha256)
    return payload


def _tuple_wire_prompt_sha256(request: TupleResolutionInput) -> str:
    messages = [
        {"role": "system", "content": TUPLE_SYSTEM_PROMPT},
        {"role": "user", "content": tuple_resolution_prompt(request)},
    ]
    encoded = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tuple_input_failure_code(error: Exception) -> str:
    message = str(error)
    if "physical numeric cell" in message:
        return "missing_physical_cell_evidence"
    if "occur exactly once" in message:
        return "evidence_not_uniquely_bound"
    if "no exact evidence" in message:
        return "evidence_not_in_layout"
    return "input_binding_failed"


def _tuple_provider_failure(error: Exception) -> tuple[str, ProviderCall | None]:
    if isinstance(error, ProviderResponseValidationError):
        if error.validation_keyword == "returned_model_mismatch":
            return _RETURNED_MODEL_MISMATCH, error.call
        allowed_codes = {"invalid_json", "schema_validation", "wire_validation"}
        code = error.code if error.code in allowed_codes else "unknown"
        return f"provider_response_{code}", error.call
    if isinstance(error, ProviderRequestRejectedError):
        return "provider_request_rejected", None
    if isinstance(error, RuntimeError):
        return "provider_transport_failed", None
    if isinstance(error, ValueError):
        return "local_validation_failed", None
    return "tuple_stage_failed", None


def _tuple_entry_sha256(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _tuple_entry_bindings(
    *,
    candidate: CandidateObservation,
    request: TupleResolutionInput | None,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    code_state: dict[str, str | bool],
    tuple_contract_sha256: str,
) -> dict[str, Any]:
    assert settings.tuple_model is not None
    request_contract = tuple_resolution_request_contract(
        require_parameters=True,
        model=settings.tuple_model,
        max_tokens=settings.tuple_max_tokens,
    )
    return {
        "candidate_sha256": tuple_candidate_binding_sha256(candidate),
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_sha256": layout_binding_sha256(layout),
        "code": code_state,
        "tuple_checkpoint_contract_sha256": tuple_contract_sha256,
        "model": settings.tuple_model,
        "max_tokens": settings.tuple_max_tokens,
        "completion_token_parameter": request_contract["completion_token_parameter"],
        "temperature": TUPLE_TEMPERATURE,
        "reasoning_effort": TUPLE_REASONING_EFFORT,
        "seed": TUPLE_SEED,
        "prompt_sha256": tuple_resolution_prompt_hash(),
        "wire_prompt_sha256": _tuple_wire_prompt_sha256(request) if request is not None else None,
        "input_sha256": request.input_sha256 if request is not None else None,
        "schema_name": TUPLE_SCHEMA_NAME,
        "schema_sha256": request_contract["schema"]["schema_sha256"],
        "request_contract_sha256": sha256_bytes(canonical_json_bytes(request_contract)),
    }


def _new_tuple_checkpoint_entry(
    *,
    status: str,
    candidate: CandidateObservation,
    request: TupleResolutionInput | None,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    code_state: dict[str, str | bool],
    tuple_contract_sha256: str,
    proposal: TupleWireProposal | None = None,
    assessment: TupleResolutionAssessment | None = None,
    call: ProviderCall | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "schema_version": _TUPLE_CHECKPOINT_ENTRY_VERSION,
        "status": status,
        "bindings": _tuple_entry_bindings(
            candidate=candidate,
            request=request,
            settings=settings,
            manifest=manifest,
            layout=layout,
            code_state=code_state,
            tuple_contract_sha256=tuple_contract_sha256,
        ),
        "input": request.model_dump(mode="json") if request is not None else None,
        "proposal": proposal.model_dump(mode="json") if proposal is not None else None,
        "assessment": assessment.model_dump(mode="json") if assessment is not None else None,
        # ProviderCall requires temperature/seed keys even when their exact shared values are
        # null. Preserve those nulls so a checkpoint remains rehydratable on resume.
        "provider_call": call.model_dump(mode="json") if call else None,
        "error_code": error_code,
    }
    entry["entry_sha256"] = _tuple_entry_sha256(entry)
    return entry


def _tuple_call_matches(
    call: ProviderCall,
    *,
    settings: PipelineSettings,
    bindings: dict[str, Any],
    require_returned_model_match: bool = True,
) -> bool:
    return not any(
        (
            call.model_requested != settings.tuple_model,
            require_returned_model_match and call.model_returned != settings.tuple_model,
            call.prompt_sha256 != bindings["wire_prompt_sha256"],
            call.temperature != TUPLE_TEMPERATURE,
            call.reasoning_effort != TUPLE_REASONING_EFFORT,
            call.max_tokens != settings.tuple_max_tokens,
            call.completion_token_parameter != bindings["completion_token_parameter"],
            call.seed != TUPLE_SEED,
            call.schema_name != TUPLE_SCHEMA_NAME,
            call.schema_sha256 != bindings["schema_sha256"],
            not call.schema_strict,
            call.data_collection != "deny",
            not call.zdr,
            not call.require_parameters,
        )
    )


def _validated_tuple_checkpoint_entry(
    entry: Any,
    *,
    candidate: CandidateObservation,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    code_state: dict[str, str | bool],
    tuple_contract_sha256: str,
) -> tuple[str, TupleResolutionAssessment | None, ProviderCall | None] | None:
    """Recompute an exact terminal tuple outcome; never trust stored proposals alone."""

    if not isinstance(entry, dict) or entry.get("schema_version") != (
        _TUPLE_CHECKPOINT_ENTRY_VERSION
    ):
        return None
    recorded_sha256 = entry.get("entry_sha256")
    unsigned = {key: value for key, value in entry.items() if key != "entry_sha256"}
    if not isinstance(recorded_sha256, str) or recorded_sha256 != _tuple_entry_sha256(unsigned):
        return None
    status = entry.get("status")
    if status not in {"success", "unsupported", "response_failure", "provider_failure"}:
        return None
    try:
        request = build_tuple_resolution_input(candidate, layout)
    except ValueError as error:
        if (
            status != "unsupported"
            or entry.get("error_code") != _tuple_input_failure_code(error)
            or entry.get("input") is not None
            or entry.get("proposal") is not None
            or entry.get("assessment") is not None
            or entry.get("provider_call") is not None
        ):
            return None
        expected = _tuple_entry_bindings(
            candidate=candidate,
            request=None,
            settings=settings,
            manifest=manifest,
            layout=layout,
            code_state=code_state,
            tuple_contract_sha256=tuple_contract_sha256,
        )
        return (status, None, None) if entry.get("bindings") == expected else None
    if status == "unsupported":
        return None
    try:
        stored_request = TupleResolutionInput.model_validate(entry.get("input"))
    except (TypeError, ValueError):
        return None
    if stored_request.model_dump(mode="json") != request.model_dump(mode="json"):
        return None
    expected = _tuple_entry_bindings(
        candidate=candidate,
        request=request,
        settings=settings,
        manifest=manifest,
        layout=layout,
        code_state=code_state,
        tuple_contract_sha256=tuple_contract_sha256,
    )
    if entry.get("bindings") != expected:
        return None
    returned_model_mismatch = (
        status == "response_failure" and entry.get("error_code") == _RETURNED_MODEL_MISMATCH
    )
    call: ProviderCall | None = None
    if entry.get("provider_call") is not None:
        try:
            call = ProviderCall.model_validate(entry["provider_call"])
        except (TypeError, ValueError):
            return None
        if not _tuple_call_matches(
            call,
            settings=settings,
            bindings=expected,
            require_returned_model_match=not returned_model_mismatch,
        ) or (returned_model_mismatch and call.model_returned == settings.tuple_model):
            return None
    if status == "success":
        if call is None or entry.get("error_code") is not None:
            return None
        try:
            proposal = TupleWireProposal.model_validate(entry.get("proposal"))
            assessment = TupleResolutionAssessment.model_validate(entry.get("assessment"))
            expected_assessment = verify_tuple_resolution_proposal(
                request=request,
                proposal=proposal,
            )
        except (TypeError, ValueError):
            return None
        if assessment.model_dump(mode="json") != expected_assessment.model_dump(mode="json"):
            return None
        if call.response_sha256 != tuple_wire_response_sha256(proposal):
            return None
        return status, assessment, call
    if entry.get("proposal") is not None or entry.get("assessment") is not None:
        return None
    if not isinstance(entry.get("error_code"), str):
        return None
    if status == "response_failure" and call is None:
        return None
    if status == "provider_failure" and call is not None:
        return None
    if status == "provider_failure" and entry.get("error_code") in (
        _RETRYABLE_NO_CALL_FAILURE_CODES
    ):
        return None
    return status, None, call


def _tuple_gate_sha256(
    *,
    candidate: CandidateObservation,
    tuple_contract_sha256: str,
    entry_sha256: str | None,
    status: str,
    semantic_match: bool,
    passed: bool,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "tuple-resolution-gate/0.1",
                "candidate_sha256": tuple_candidate_binding_sha256(candidate),
                "tuple_checkpoint_contract_sha256": tuple_contract_sha256,
                "tuple_checkpoint_entry_sha256": entry_sha256,
                "status": status,
                "semantic_match": semantic_match,
                "passed": passed,
            }
        )
    )


def _verifier_candidate_sha256(candidate: CandidateObservation) -> str:
    return sha256_bytes(canonical_json_bytes(candidate))


def _verifier_evidence_sha256(evidence_block: FrozenEvidenceBlock) -> str:
    return sha256_bytes(canonical_json_bytes(evidence_block))


def _verifier_checkpoint_contract(
    *,
    spec: PaperSpec,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    code_state: dict[str, str | bool],
    tuple_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": _VERIFIER_CHECKPOINT_CONTRACT_VERSION,
        "paper_id": spec.paper_id,
        "paper_title": spec.title,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_sha256": layout_binding_sha256(layout),
        "tuple_checkpoint_contract_sha256": tuple_contract_sha256,
        "verifier": _verifier_run_configuration(settings),
        "code": code_state,
    }


def _new_verifier_checkpoint(contract: dict[str, Any], contract_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": _VERIFIER_CHECKPOINT_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "candidates": {},
    }


def _load_verifier_checkpoint(
    path: Path,
    *,
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return _new_verifier_checkpoint(contract, contract_sha256)
    if not isinstance(payload, dict):
        return _new_verifier_checkpoint(contract, contract_sha256)
    if (
        payload.get("schema_version") != _VERIFIER_CHECKPOINT_SCHEMA_VERSION
        or payload.get("contract_sha256") != contract_sha256
        or payload.get("contract") != contract
        or not isinstance(payload.get("candidates"), dict)
    ):
        return _new_verifier_checkpoint(contract, contract_sha256)
    return payload


def _verifier_wire_prompt_sha256(
    candidate: CandidateObservation,
    evidence_block: FrozenEvidenceBlock,
) -> str:
    request = VerificationRequest(candidate=candidate, evidence_block=evidence_block)
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": verification_prompt(request)},
    ]
    prompt_bytes = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(prompt_bytes).hexdigest()


def _verifier_response_sha256(verification: CandidateVerification) -> str:
    return verifier_assessment_sha256(verification.provider_assessment)


def _verifier_gate_sha256(
    *,
    candidate: CandidateObservation,
    tuple_gate_sha256: str,
    verifier_contract_sha256: str,
    verifier_entry_sha256: str,
    verification: CandidateVerification,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "independent-verifier-gate/0.2",
                "candidate_sha256": _verifier_candidate_sha256(candidate),
                "tuple_gate_sha256": tuple_gate_sha256,
                "verifier_checkpoint_contract_sha256": verifier_contract_sha256,
                "verifier_checkpoint_entry_sha256": verifier_entry_sha256,
                "provider_decision": verification.provider_assessment.decision.value,
                "grounding": verification.grounding,
                "effective_decision": verification.effective_decision.value,
                "passed": verification.effective_decision is IndependentDecision.ACCEPT,
            }
        )
    )


def _new_verifier_checkpoint_entry(
    *,
    status: str,
    candidate: CandidateObservation,
    evidence_block: FrozenEvidenceBlock,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    code_state: dict[str, str | bool],
    tuple_gate_sha256: str,
    verification: CandidateVerification | None = None,
    call: ProviderCall | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    assert settings.verifier_model is not None
    request_contract = verifier_request_contract(
        model=settings.verifier_model,
        max_tokens=settings.verifier_max_tokens,
    )
    entry = {
        "schema_version": _VERIFIER_CHECKPOINT_ENTRY_VERSION,
        "status": status,
        "bindings": {
            "candidate_sha256": _verifier_candidate_sha256(candidate),
            "evidence_block_sha256": _verifier_evidence_sha256(evidence_block),
            "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
            "layout_sha256": layout_binding_sha256(layout),
            "tuple_gate_sha256": tuple_gate_sha256,
            "code": code_state,
            "model": settings.verifier_model,
            "max_tokens": settings.verifier_max_tokens,
            "completion_token_parameter": request_contract["completion_token_parameter"],
            **VERIFIER_REQUEST_SETTINGS.as_dict(),
            "wire_prompt_sha256": _verifier_wire_prompt_sha256(candidate, evidence_block),
            "schema_name": VERIFIER_SCHEMA_NAME,
            "schema_sha256": request_contract["schema"]["schema_sha256"],
            "request_contract_sha256": sha256_bytes(canonical_json_bytes(request_contract)),
        },
        "verification": verification.model_dump(mode="json") if verification else None,
        # Preserve required null request controls for exact checkpoint rehydration.
        "provider_call": call.model_dump(mode="json") if call else None,
        "error_code": error_code,
    }
    entry["entry_sha256"] = _origin_checkpoint_entry_sha256(entry)
    return entry


def _validated_verifier_checkpoint_entry(
    entry: Any,
    *,
    candidate: CandidateObservation,
    evidence_block: FrozenEvidenceBlock,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    code_state: dict[str, str | bool],
    tuple_gate_sha256: str,
) -> tuple[str, CandidateVerification | None, ProviderCall | None] | None:
    if not isinstance(entry, dict) or entry.get("schema_version") != (
        _VERIFIER_CHECKPOINT_ENTRY_VERSION
    ):
        return None
    recorded_sha256 = entry.get("entry_sha256")
    unsigned = {key: value for key, value in entry.items() if key != "entry_sha256"}
    if not isinstance(recorded_sha256, str) or recorded_sha256 != (
        _origin_checkpoint_entry_sha256(unsigned)
    ):
        return None
    if settings.verifier_model is None:
        return None
    status = entry.get("status")
    if status not in {"success", "response_failure", "provider_failure"}:
        return None
    request_contract = verifier_request_contract(
        model=settings.verifier_model,
        max_tokens=settings.verifier_max_tokens,
    )
    expected_bindings = {
        "candidate_sha256": _verifier_candidate_sha256(candidate),
        "evidence_block_sha256": _verifier_evidence_sha256(evidence_block),
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_sha256": layout_binding_sha256(layout),
        "tuple_gate_sha256": tuple_gate_sha256,
        "code": code_state,
        "model": settings.verifier_model,
        "max_tokens": settings.verifier_max_tokens,
        "completion_token_parameter": request_contract["completion_token_parameter"],
        **VERIFIER_REQUEST_SETTINGS.as_dict(),
        "wire_prompt_sha256": _verifier_wire_prompt_sha256(candidate, evidence_block),
        "schema_name": VERIFIER_SCHEMA_NAME,
        "schema_sha256": request_contract["schema"]["schema_sha256"],
        "request_contract_sha256": sha256_bytes(canonical_json_bytes(request_contract)),
    }
    if entry.get("bindings") != expected_bindings:
        return None
    returned_model_mismatch = (
        status == "response_failure" and entry.get("error_code") == _RETURNED_MODEL_MISMATCH
    )
    call: ProviderCall | None = None
    if entry.get("provider_call") is not None:
        try:
            call = ProviderCall.model_validate(entry["provider_call"])
        except (TypeError, ValueError):
            return None
        if any(
            (
                call.model_requested != settings.verifier_model,
                (not returned_model_mismatch and call.model_returned != settings.verifier_model),
                call.prompt_sha256 != expected_bindings["wire_prompt_sha256"],
                call.temperature != VERIFIER_REQUEST_SETTINGS.temperature,
                call.reasoning_effort != VERIFIER_REQUEST_SETTINGS.reasoning_effort,
                call.max_tokens != settings.verifier_max_tokens,
                call.completion_token_parameter != expected_bindings["completion_token_parameter"],
                call.seed != VERIFIER_REQUEST_SETTINGS.seed,
                call.schema_name != VERIFIER_SCHEMA_NAME,
                call.schema_sha256 != expected_bindings["schema_sha256"],
                not call.schema_strict,
                call.data_collection != "deny",
                not call.zdr,
                call.require_parameters != VERIFIER_REQUEST_SETTINGS.require_parameters,
            )
        ):
            return None
        if returned_model_mismatch and call.model_returned == settings.verifier_model:
            return None
    if status == "success":
        if call is None or entry.get("error_code") is not None:
            return None
        try:
            verification = CandidateVerification.model_validate(entry.get("verification"))
        except (TypeError, ValueError):
            return None
        expected_verification = contextualize_verification(
            candidate=candidate,
            evidence_block=evidence_block,
            provider_assessment=verification.provider_assessment,
        )
        if any(
            (
                verification != expected_verification,
                verification.evidence_block_sha256
                != verifier_evidence_block_sha256(evidence_block),
                call.response_sha256 != _verifier_response_sha256(verification),
            )
        ):
            return None
        return status, verification, call
    if entry.get("verification") is not None or not isinstance(entry.get("error_code"), str):
        return None
    if status == "response_failure" and call is None:
        return None
    if status == "provider_failure" and call is not None:
        return None
    if status == "provider_failure" and entry.get("error_code") in (
        _RETRYABLE_NO_CALL_FAILURE_CODES
    ):
        return None
    return status, None, call


def _verifier_failure(error: Exception) -> tuple[str, ProviderCall | None]:
    """Reduce one verifier failure to a quote-free, fail-closed outcome."""

    if isinstance(error, ProviderResponseValidationError):
        if error.validation_keyword == "returned_model_mismatch":
            return _RETURNED_MODEL_MISMATCH, error.call
        allowed_codes = {"invalid_json", "schema_validation", "wire_validation"}
        code = error.code if error.code in allowed_codes else "unknown"
        return f"provider_response_{code}", error.call
    if isinstance(error, ProviderRequestRejectedError):
        return "provider_request_rejected", None
    if isinstance(error, RuntimeError):
        return "provider_transport_failed", None
    if isinstance(error, ValueError):
        return "local_validation_failed", None
    return "verifier_stage_failed", None


def _origin_checkpoint_contract(
    *,
    spec: PaperSpec,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    code_state: dict[str, str | bool],
    tuple_contract_sha256: str,
    verifier_contract_sha256: str,
) -> dict[str, Any]:
    """Bind private origin proposals to source, layout, provider, and code."""

    return {
        "schema_version": _ORIGIN_CHECKPOINT_CONTRACT_VERSION,
        "paper_id": spec.paper_id,
        "paper_title": spec.title,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_sha256": layout_binding_sha256(layout),
        "tuple_checkpoint_contract_sha256": tuple_contract_sha256,
        "verifier_checkpoint_contract_sha256": verifier_contract_sha256,
        "origin": _origin_run_configuration(settings),
        "code": code_state,
    }


def _new_origin_checkpoint(contract: dict[str, Any], contract_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": _ORIGIN_CHECKPOINT_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "candidates": {},
    }


def _load_origin_checkpoint(
    path: Path,
    *,
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    """Reuse only an exact stage contract; stale or malformed state is rebuilt."""

    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return _new_origin_checkpoint(contract, contract_sha256)
    if not isinstance(payload, dict):
        return _new_origin_checkpoint(contract, contract_sha256)
    if (
        payload.get("schema_version") != _ORIGIN_CHECKPOINT_SCHEMA_VERSION
        or payload.get("contract_sha256") != contract_sha256
        or payload.get("contract") != contract
        or not isinstance(payload.get("candidates"), dict)
    ):
        return _new_origin_checkpoint(contract, contract_sha256)
    return payload


def _origin_wire_prompt_sha256(
    candidate: CandidateObservation,
    bundle: OriginRetrievalBundle,
) -> str:
    messages = [
        {"role": "system", "content": ORIGIN_SYSTEM_PROMPT},
        {"role": "user", "content": producer_origin_prompt(candidate, bundle)},
    ]
    prompt_bytes = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(prompt_bytes).hexdigest()


def _origin_checkpoint_entry_sha256(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _new_origin_checkpoint_entry(
    *,
    status: str,
    candidate: CandidateObservation,
    layout: PdfLayout,
    bundle: OriginRetrievalBundle | None,
    settings: PipelineSettings,
    code_state: dict[str, str | bool],
    tuple_gate_sha256: str,
    verifier_gate_sha256: str,
    proposal: ProducerOriginProposal | None = None,
    assessment: ProducerOriginAssessment | None = None,
    call: ProviderCall | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    assert settings.origin_model is not None
    request_contract = producer_origin_request_contract(
        model=settings.origin_model,
        max_tokens=settings.origin_max_tokens,
    )
    entry = {
        "schema_version": _ORIGIN_CHECKPOINT_ENTRY_VERSION,
        "status": status,
        "bindings": {
            "candidate_sha256": candidate_origin_binding_sha256(candidate),
            "layout_sha256": layout_binding_sha256(layout),
            "tuple_gate_sha256": tuple_gate_sha256,
            "verifier_gate_sha256": verifier_gate_sha256,
            "code": code_state,
            "model": settings.origin_model,
            "max_tokens": settings.origin_max_tokens,
            "completion_token_parameter": request_contract["completion_token_parameter"],
            "temperature": ORIGIN_TEMPERATURE,
            "reasoning_effort": ORIGIN_REASONING_EFFORT,
            "require_parameters": request_contract["routing"]["require_parameters"],
            "seed": ORIGIN_SEED,
            "prompt_sha256": producer_origin_prompt_hash(),
            "wire_prompt_sha256": (
                _origin_wire_prompt_sha256(candidate, bundle) if bundle is not None else None
            ),
            "schema_name": ORIGIN_SCHEMA_NAME,
            "schema_sha256": request_contract["schema"]["schema_sha256"],
            "request_contract_sha256": sha256_bytes(canonical_json_bytes(request_contract)),
            "request_fingerprint_sha256": (
                producer_origin_request_fingerprint(
                    model=settings.origin_model,
                    candidate=candidate,
                    bundle=bundle,
                    max_tokens=settings.origin_max_tokens,
                )
                if bundle is not None
                else None
            ),
        },
        "retrieval_bundle": bundle.model_dump(mode="json") if bundle else None,
        "proposal": proposal.model_dump(mode="json") if proposal else None,
        "assessment": assessment.model_dump(mode="json") if assessment else None,
        # Preserve required null request controls for exact checkpoint rehydration.
        "provider_call": call.model_dump(mode="json") if call else None,
        "error_code": error_code,
    }
    entry["entry_sha256"] = _origin_checkpoint_entry_sha256(entry)
    return entry


def _validated_origin_checkpoint_entry(
    entry: Any,
    *,
    candidate: CandidateObservation,
    layout: PdfLayout,
    settings: PipelineSettings,
    code_state: dict[str, str | bool],
    tuple_gate_sha256: str,
    verifier_gate_sha256: str,
) -> (
    tuple[
        str,
        OriginRetrievalBundle | None,
        ProducerOriginProposal | None,
        ProducerOriginAssessment | None,
        ProviderCall | None,
        str | None,
    ]
    | None
):
    """Rehydrate and independently recompute every private origin artifact."""

    if not isinstance(entry, dict) or entry.get("schema_version") != (
        _ORIGIN_CHECKPOINT_ENTRY_VERSION
    ):
        return None
    recorded_sha256 = entry.get("entry_sha256")
    unsigned = {key: value for key, value in entry.items() if key != "entry_sha256"}
    if not isinstance(recorded_sha256, str) or recorded_sha256 != (
        _origin_checkpoint_entry_sha256(unsigned)
    ):
        return None
    if settings.origin_model is None:
        return None
    status = entry.get("status", "success")
    if status not in {"success", "response_failure", "provider_failure"}:
        return None
    bundle: OriginRetrievalBundle | None = None
    proposal: ProducerOriginProposal | None = None
    assessment: ProducerOriginAssessment | None = None
    call: ProviderCall | None = None
    try:
        if entry.get("retrieval_bundle") is not None:
            bundle = OriginRetrievalBundle.model_validate(entry["retrieval_bundle"])
        if entry.get("proposal") is not None:
            proposal = ProducerOriginProposal.model_validate(entry["proposal"])
        if entry.get("assessment") is not None:
            assessment = ProducerOriginAssessment.model_validate(entry["assessment"])
        if entry.get("provider_call") is not None:
            call = ProviderCall.model_validate(entry["provider_call"])
    except (TypeError, ValueError):
        return None
    try:
        expected_bundle = retrieve_origin_context(candidate, layout)
    except ValueError as error:
        expected_bundle = None
        expected_local_code, _ = _origin_failure(error)
    else:
        expected_local_code = None
    if (bundle is None) != (expected_bundle is None) or (
        bundle is not None
        and expected_bundle is not None
        and bundle.model_dump(mode="json") != expected_bundle.model_dump(mode="json")
    ):
        return None
    request_contract = producer_origin_request_contract(
        model=settings.origin_model,
        max_tokens=settings.origin_max_tokens,
    )
    expected_bindings = {
        "candidate_sha256": candidate_origin_binding_sha256(candidate),
        "layout_sha256": layout_binding_sha256(layout),
        "tuple_gate_sha256": tuple_gate_sha256,
        "verifier_gate_sha256": verifier_gate_sha256,
        "code": code_state,
        "model": settings.origin_model,
        "max_tokens": settings.origin_max_tokens,
        "completion_token_parameter": request_contract["completion_token_parameter"],
        "temperature": ORIGIN_TEMPERATURE,
        "reasoning_effort": ORIGIN_REASONING_EFFORT,
        "require_parameters": request_contract["routing"]["require_parameters"],
        "seed": ORIGIN_SEED,
        "prompt_sha256": producer_origin_prompt_hash(),
        "wire_prompt_sha256": (
            _origin_wire_prompt_sha256(candidate, bundle) if bundle is not None else None
        ),
        "schema_name": ORIGIN_SCHEMA_NAME,
        "schema_sha256": request_contract["schema"]["schema_sha256"],
        "request_contract_sha256": sha256_bytes(canonical_json_bytes(request_contract)),
        "request_fingerprint_sha256": (
            producer_origin_request_fingerprint(
                model=settings.origin_model,
                candidate=candidate,
                bundle=bundle,
                max_tokens=settings.origin_max_tokens,
            )
            if bundle is not None
            else None
        ),
    }
    if entry.get("bindings") != expected_bindings:
        return None
    returned_model_mismatch = (
        status == "response_failure" and entry.get("error_code") == _RETURNED_MODEL_MISMATCH
    )
    if call is not None and any(
        (
            call.provider != "openrouter",
            call.model_requested != settings.origin_model,
            not returned_model_mismatch and call.model_returned != settings.origin_model,
            call.prompt_sha256 != expected_bindings["wire_prompt_sha256"],
            call.temperature != ORIGIN_TEMPERATURE,
            call.reasoning_effort != ORIGIN_REASONING_EFFORT,
            call.max_tokens != settings.origin_max_tokens,
            call.completion_token_parameter != expected_bindings["completion_token_parameter"],
            call.seed != ORIGIN_SEED,
            call.response_format != request_contract["schema"]["response_format"],
            call.schema_name != ORIGIN_SCHEMA_NAME,
            call.schema_sha256 != expected_bindings["schema_sha256"],
            call.schema_strict != request_contract["schema"]["schema_strict"],
            call.data_collection != request_contract["privacy"]["data_collection"],
            call.zdr != request_contract["privacy"]["zdr"],
            call.require_parameters != request_contract["routing"]["require_parameters"],
        )
    ):
        return None
    if returned_model_mismatch and (call is None or call.model_returned == settings.origin_model):
        return None
    error_code = entry.get("error_code")
    if status == "success":
        if (
            any(value is None for value in (bundle, proposal, assessment, call))
            or error_code is not None
        ):
            return None
        assert (
            bundle is not None
            and proposal is not None
            and assessment is not None
            and call is not None
        )
        expected_assessment = verify_producer_origin_proposal(
            candidate=candidate,
            layout=layout,
            bundle=bundle,
            proposal=proposal,
        )
        if assessment.model_dump(mode="json") != expected_assessment.model_dump(mode="json"):
            return None
        if call.response_sha256 != producer_origin_wire_response_sha256(proposal):
            return None
    elif status == "response_failure":
        if bundle is None or call is None or proposal is not None or assessment is not None:
            return None
        if not isinstance(error_code, str) or not (
            error_code.startswith("provider_response_") or error_code == "local_validation_failed"
        ):
            return None
    else:
        if call is not None or proposal is not None or assessment is not None:
            return None
        if not isinstance(error_code, str) or (
            bundle is None and error_code != expected_local_code
        ):
            return None
        if error_code in _RETRYABLE_NO_CALL_FAILURE_CODES:
            return None
    return status, bundle, proposal, assessment, call, error_code


def _origin_failure(error: Exception) -> tuple[str, ProviderCall | None]:
    """Reduce an origin-stage failure to one quote-free candidate outcome."""

    if isinstance(error, ProviderResponseValidationError):
        if error.validation_keyword == "returned_model_mismatch":
            return _RETURNED_MODEL_MISMATCH, error.call
        allowed_codes = {"invalid_json", "schema_validation", "wire_validation"}
        code = error.code if error.code in allowed_codes else "unknown"
        return f"provider_response_{code}", error.call
    if isinstance(error, ProviderRequestRejectedError):
        return "provider_request_rejected", None
    if isinstance(error, RuntimeError):
        return "provider_transport_failed", None
    if isinstance(error, ValueError):
        return "local_validation_failed", None
    return "origin_stage_failed", None


def _apply_origin_assessment(
    candidate: CandidateObservation,
    assessment: ProducerOriginAssessment,
) -> None:
    """Apply only review or external demotion; never grant positive authority."""

    candidate.export_status = ExportStatus.NEEDS_REVIEW
    if assessment.effective_state is AttributionState.EXTERNALLY_SOURCED:
        candidate.attribution = AttributionVerdict(
            state=AttributionState.EXTERNALLY_SOURCED,
            rule_id="whole_paper_origin_verified_external",
        )
        candidate.export_reason = "origin_retrieval=verified_external"
    elif assessment.effective_state is AttributionState.NO_SIGNAL:
        candidate.attribution = AttributionVerdict(
            state=AttributionState.NO_SIGNAL,
            rule_id="whole_paper_origin_no_signal",
        )
        candidate.export_reason = "origin_retrieval=no_signal"
    else:
        candidate.attribution = AttributionVerdict(
            state=AttributionState.UNRESOLVED,
            rule_id="whole_paper_origin_review_only",
        )
        candidate.export_reason = (
            "origin_retrieval=positive_evidence_review_only"
            if assessment.positive_evidence_verified
            else "origin_retrieval=unresolved"
        )


def _merge_row_outcome(
    target: RowEnumerationOutcome,
    source: RowEnumerationOutcome,
) -> None:
    overlap = set(target.records) & set(source.records)
    if overlap:
        raise ValueError(f"row outcome contains duplicate planned IDs: {sorted(overlap)!r}")
    target.records.update(source.records)
    target.calls.extend(source.calls)
    target.attempts.extend(source.attempts)
    target.unresolved_row_ids.extend(source.unresolved_row_ids)
    target.unbatchable_row_ids.extend(source.unbatchable_row_ids)
    target.unknown_row_ids.extend(source.unknown_row_ids)
    target.invalid_row_reasons.update(source.invalid_row_reasons)
    target.warnings.extend(source.warnings)


def _extractor_failure(
    error: Exception,
) -> tuple[str, ProviderCall | None, dict[str, int]]:
    """Reduce an extractor exception to bounded, secret-free run telemetry."""

    if isinstance(error, ProviderResponseValidationError):
        if error.validation_keyword == "returned_model_mismatch":
            return _RETURNED_MODEL_MISMATCH, error.call, {}
        allowed_codes = {"invalid_json", "schema_validation", "wire_validation"}
        validation_code = error.code if error.code in allowed_codes else "unknown"
        return f"provider_response_{validation_code}", error.call, {}
    if isinstance(error, ProviderRequestRejectedError):
        return "provider_request_rejected", None, {"http_status": error.status_code}
    if isinstance(error, RuntimeError):
        return "provider_transport_failed", None, {}
    if isinstance(error, ValueError):
        return "local_validation_failed", None, {}
    return "extractor_block_failed", None, {}


def _recover_split_block(
    *,
    client: OpenRouterClient,
    settings: PipelineSettings,
    spec: PaperSpec,
    block: ResultBlock,
    max_depth: int = LEGACY_RECOVERY_MAX_DEPTH,
    resume_attempts: list[dict[str, Any]] | None = None,
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> LegacyRecoveryOutcome:
    """Recover a content-invalid block through a bounded changed-input split tree.

    Only a completed response-validation failure may recurse. Request rejection and
    transport failure are terminal, and exception text is never copied into the run.
    Successful siblings remain usable even when another subtree terminates.
    """

    if max_depth < 1:
        raise ValueError("legacy recovery max depth must be positive")

    outcome = LegacyRecoveryOutcome()
    resumed = list(resume_attempts or [])
    resume_index = 0

    def terminal_failure(
        *,
        failed_block: ResultBlock,
        depth: int,
        error_code: str,
        completed_provider_call: bool,
        terminal_reason: str,
        safe_details: dict[str, int],
    ) -> None:
        outcome.terminal_failures.append(
            LegacyRecoveryFailure(
                block_id=failed_block.block_id,
                page=failed_block.page,
                depth=depth,
                error_code=error_code,
                completed_provider_call=completed_provider_call,
                terminal_reason=terminal_reason,
                safe_details=safe_details,
            )
        )

    def apply_resumed(current: ResultBlock, depth: int, attempt: dict[str, Any]) -> None:
        call = attempt["call"]
        status = attempt["status"]
        if call is not None:
            outcome.calls.append(call)
            outcome.resumed_calls.append(call)
        if status == "success":
            outcome.candidates.extend(attempt["candidates"])
            outcome.successful_calls.append(call)
            outcome.resumed_successful_calls.append(call)
            outcome.warnings.extend(attempt["warnings"])
            return
        descendants = split_result_block(current)
        if status.startswith("provider_response_") and depth < max_depth and descendants:
            for descendant in descendants:
                visit(descendant, depth + 1)
            return
        if status.startswith("provider_response_"):
            terminal_reason = "unsplittable" if not descendants else "max_depth_reached"
        elif status == "provider_request_rejected":
            terminal_reason = "request_rejected"
        else:
            terminal_reason = "transport_failure"
        terminal_failure(
            failed_block=current,
            depth=depth,
            error_code=status,
            completed_provider_call=call is not None,
            terminal_reason=terminal_reason,
            safe_details=attempt["safe_details"],
        )

    def visit(child: ResultBlock, depth: int) -> None:
        nonlocal resume_index
        outcome.max_depth_reached = max(outcome.max_depth_reached, depth)
        if resume_index < len(resumed):
            cached = resumed[resume_index]
            if cached["block"] != child or cached["depth"] != depth:
                raise ValueError("legacy recovery resume prefix does not match traversal")
            resume_index += 1
            apply_resumed(child, depth, cached)
            return
        try:
            child_candidates, call, child_warnings = extract_page_candidates(
                client=client,
                model=settings.model,
                paper_id=spec.paper_id,
                paper_title=spec.title,
                fragment=_block_fragment(child),
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                reasoning_effort=settings.reasoning_effort,
                seed=settings.seed,
            )
        except ProviderBudgetError:
            raise
        except ProviderResponseValidationError as error:
            error_code, completed_call, safe_details = _extractor_failure(error)
            assert completed_call is not None
            outcome.calls.append(completed_call)
            outcome.new_calls.append(completed_call)
            if on_attempt is not None:
                on_attempt(
                    _legacy_recovery_attempt_payload(
                        block=child,
                        depth=depth,
                        status=error_code,
                        call=completed_call,
                        safe_details=safe_details,
                    )
                )
            descendants = split_result_block(child)
            if depth < max_depth and descendants:
                for descendant in descendants:
                    visit(descendant, depth + 1)
                return
            terminal_failure(
                failed_block=child,
                depth=depth,
                error_code=error_code,
                completed_provider_call=True,
                terminal_reason=("unsplittable" if not descendants else "max_depth_reached"),
                safe_details=safe_details,
            )
        except ProviderRequestRejectedError as error:
            error_code, completed_call, safe_details = _extractor_failure(error)
            assert completed_call is None
            if on_attempt is not None:
                on_attempt(
                    _legacy_recovery_attempt_payload(
                        block=child,
                        depth=depth,
                        status=error_code,
                        call=None,
                        safe_details=safe_details,
                    )
                )
            terminal_failure(
                failed_block=child,
                depth=depth,
                error_code=error_code,
                completed_provider_call=False,
                terminal_reason="request_rejected",
                safe_details=safe_details,
            )
        except RuntimeError as error:
            error_code, completed_call, safe_details = _extractor_failure(error)
            assert completed_call is None
            if on_attempt is not None:
                on_attempt(
                    _legacy_recovery_attempt_payload(
                        block=child,
                        depth=depth,
                        status=error_code,
                        call=None,
                        safe_details=safe_details,
                    )
                )
            terminal_failure(
                failed_block=child,
                depth=depth,
                error_code=error_code,
                completed_provider_call=False,
                terminal_reason="transport_failure",
                safe_details=safe_details,
            )
        else:
            outcome.candidates.extend(child_candidates)
            outcome.calls.append(call)
            outcome.successful_calls.append(call)
            outcome.new_calls.append(call)
            outcome.new_successful_calls.append(call)
            outcome.warnings.extend(child_warnings)
            if on_attempt is not None:
                on_attempt(
                    _legacy_recovery_attempt_payload(
                        block=child,
                        depth=depth,
                        status="success",
                        call=call,
                        candidates=child_candidates,
                        warnings=child_warnings,
                    )
                )

    children = split_result_block(block)
    if not children:
        outcome.terminal_failures.append(
            LegacyRecoveryFailure(
                block_id=block.block_id,
                page=block.page,
                depth=0,
                error_code="recovery_unsplittable",
                completed_provider_call=False,
                terminal_reason="unsplittable",
            )
        )
        return outcome
    for child in children:
        visit(child, 1)
    if resume_index != len(resumed):
        raise ValueError("legacy recovery resume prefix contains unreachable attempts")
    return outcome


def run_paper(
    *,
    spec: PaperSpec,
    settings: PipelineSettings,
    client: OpenRouterClient,
) -> dict[str, Any]:
    """Run every stage for one paper and return a compact public-safe summary."""

    _validate_pipeline_settings(settings)
    started = time.monotonic()
    output_dir = settings.output_root / spec.paper_id
    _clear_paper_run_outputs(output_dir)
    manifest = freeze_paper(spec, settings)
    paper_source = next(source for source in manifest.sources if source.role == SourceRole.PAPER)
    pdf_path = resolve_cached_path(paper_source, settings.project_root)
    layout = extract_pdf_layout(pdf_path, paper_source.source_id)
    layout_sha256 = write_json(output_dir / "private" / "layout.json", layout)
    source_processing = build_source_processing_artifact(
        manifest=manifest,
        layout=layout,
        layout_sha256=layout_sha256,
    )
    source_processing_sha256 = write_json(
        output_dir / "source-processing.json",
        source_processing,
    )
    selected_pages = _select_pages(layout, spec)
    block_config = ResultBlockConfig(max_blocks_per_page=settings.max_blocks_per_page)
    blocks = [
        block
        for page in selected_pages
        for block in segment_page_result_blocks(
            page,
            config=block_config,
        )
    ]
    write_json(output_dir / "private" / "result-blocks.json", blocks)
    code_state = _code_state(settings.project_root)
    tuple_contract = _tuple_checkpoint_contract(
        spec=spec,
        settings=settings,
        manifest=manifest,
        layout=layout,
        code_state=code_state,
    )
    tuple_checkpoint_contract_sha256 = sha256_bytes(canonical_json_bytes(tuple_contract))
    tuple_checkpoint_path = output_dir / "private" / "tuple-resolution-checkpoint.json"
    tuple_checkpoint: dict[str, Any] | None = None
    tuple_checkpoint_candidates: dict[str, Any] = {}
    if settings.tuple_model is not None:
        tuple_checkpoint = _load_tuple_checkpoint(
            tuple_checkpoint_path,
            contract=tuple_contract,
            contract_sha256=tuple_checkpoint_contract_sha256,
        )
        write_json(tuple_checkpoint_path, tuple_checkpoint)
        tuple_checkpoint_candidates = tuple_checkpoint["candidates"]
    verifier_checkpoint_path = output_dir / "private" / "verifier-checkpoint.json"
    verifier_checkpoint: dict[str, Any] | None = None
    verifier_checkpoint_contract_sha256: str | None = None
    verifier_checkpoint_candidates: dict[str, Any] = {}
    if settings.verifier_model is not None:
        verifier_contract = _verifier_checkpoint_contract(
            spec=spec,
            settings=settings,
            manifest=manifest,
            layout=layout,
            code_state=code_state,
            tuple_contract_sha256=tuple_checkpoint_contract_sha256,
        )
        verifier_checkpoint_contract_sha256 = sha256_bytes(canonical_json_bytes(verifier_contract))
        verifier_checkpoint = _load_verifier_checkpoint(
            verifier_checkpoint_path,
            contract=verifier_contract,
            contract_sha256=verifier_checkpoint_contract_sha256,
        )
        write_json(verifier_checkpoint_path, verifier_checkpoint)
        verifier_checkpoint_candidates = verifier_checkpoint["candidates"]
    origin_checkpoint_path = output_dir / "private" / "origin-retrieval-checkpoint.json"
    origin_checkpoint: dict[str, Any] | None = None
    origin_checkpoint_contract_sha256: str | None = None
    origin_checkpoint_candidates: dict[str, Any] = {}
    if settings.origin_model is not None:
        assert verifier_checkpoint_contract_sha256 is not None
        origin_contract = _origin_checkpoint_contract(
            spec=spec,
            settings=settings,
            manifest=manifest,
            layout=layout,
            code_state=code_state,
            tuple_contract_sha256=tuple_checkpoint_contract_sha256,
            verifier_contract_sha256=verifier_checkpoint_contract_sha256,
        )
        origin_checkpoint_contract_sha256 = sha256_bytes(canonical_json_bytes(origin_contract))
        origin_checkpoint = _load_origin_checkpoint(
            origin_checkpoint_path,
            contract=origin_contract,
            contract_sha256=origin_checkpoint_contract_sha256,
        )
        write_json(origin_checkpoint_path, origin_checkpoint)
        origin_checkpoint_candidates = origin_checkpoint["candidates"]
    row_plan: RowEnumerationPlan | None = None
    row_preflight: dict[str, Any] | None = None
    row_plan_sha256: str | None = None
    row_checkpoint_contract_sha256: str | None = None
    row_checkpoint_path = output_dir / "private" / "row-enumeration-checkpoint.json"
    row_checkpoint: dict[str, Any] | None = None
    row_checkpoint_batches: dict[str, Any] = {}
    row_outcome = RowEnumerationOutcome()
    row_terminal_ledger: RowTerminalLedger | None = None
    row_terminal_sha256: str | None = None
    row_batches_resumed = 0
    if settings.row_enumeration_enabled:
        row_plan = build_row_enumeration_plan(
            layout,
            blocks,
            config=settings.row_enumeration_config,
        )
        row_plan_path = output_dir / "private" / "row-enumeration-plan.json"
        row_plan_sha256 = write_json(row_plan_path, row_plan)
        row_preflight = _row_preflight(
            settings=settings,
            block_count=len(blocks),
            plan=row_plan,
        )
        write_json(output_dir / "private" / "row-enumeration-preflight.json", row_preflight)
        contract = _row_checkpoint_contract(
            spec=spec,
            settings=settings,
            manifest=manifest,
            layout=layout,
            plan_sha256=row_plan_sha256,
            code_state=code_state,
        )
        row_checkpoint_contract_sha256 = sha256_bytes(canonical_json_bytes(contract))
        row_checkpoint = _load_row_checkpoint(
            row_checkpoint_path,
            contract=contract,
            contract_sha256=row_checkpoint_contract_sha256,
            batches=row_plan.batches,
        )
        write_json(row_checkpoint_path, row_checkpoint)
        row_checkpoint_batches = row_checkpoint["batches"]
        row_outcome.unbatchable_row_ids = [item.row_id for item in row_plan.unbatchable_rows]
    checkpoint_contract = _extractor_checkpoint_contract(
        spec=spec,
        settings=settings,
        manifest=manifest,
        layout=layout,
        block_config=block_config,
        blocks=blocks,
        code_state=code_state,
    )
    checkpoint_contract_sha256 = sha256_bytes(canonical_json_bytes(checkpoint_contract))
    checkpoint_path = output_dir / "private" / "extractor-checkpoint.json"
    checkpoint = _load_extractor_checkpoint(
        checkpoint_path,
        contract=checkpoint_contract,
        contract_sha256=checkpoint_contract_sha256,
    )
    write_json(checkpoint_path, checkpoint)
    checkpoint_blocks: dict[str, Any] = checkpoint["blocks"]
    checkpoint_recoveries: dict[str, Any] = checkpoint["recoveries"]
    candidates: list[CandidateObservation] = []
    verifications: list[CandidateVerification] = []
    calls: list[ProviderCall] = []
    successful_new_calls: list[ProviderCall] = []
    resumed_calls: list[ProviderCall] = []
    successful_resumed_calls: list[ProviderCall] = []
    tuple_calls: list[ProviderCall] = []
    tuple_resumed_calls: list[ProviderCall] = []
    tuple_outcomes: list[dict[str, Any]] = []
    tuple_candidates_resumed = 0
    tuple_candidates_failed = 0
    tuple_candidates_unsupported = 0
    tuple_pass_ids: set[str] = set()
    tuple_gate_by_observation_id: dict[str, str] = {}
    tuple_sidecar_sha256: str | None = None
    verifier_calls: list[ProviderCall] = []
    verifier_successful_calls: list[ProviderCall] = []
    verifier_resumed_calls: list[ProviderCall] = []
    verifier_successful_resumed_calls: list[ProviderCall] = []
    verifier_candidates_resumed = 0
    verifier_candidates_selected = 0
    verifier_candidates_unbound = 0
    verifier_candidates_executed = 0
    verifier_candidates_failed = 0
    verifier_gate_by_observation_id: dict[str, str] = {}
    verifier_sidecar_sha256: str | None = None
    origin_calls: list[ProviderCall] = []
    origin_resumed_calls: list[ProviderCall] = []
    origin_outcomes: list[dict[str, Any]] = []
    origin_candidates_resumed = 0
    origin_candidates_failed = 0
    block_attempts: list[dict[str, Any]] = []
    warnings: list[str] = []
    blocks_succeeded = 0
    blocks_failed = 0
    blocks_resumed = 0
    successfully_examined_blocks: list[ResultBlock] = []
    for block in blocks:
        fragment = _block_fragment(block)
        cached = _validated_checkpoint_entry(
            checkpoint_blocks.get(block.block_id),
            block=block,
            spec=spec,
            settings=settings,
            contract=checkpoint["contract"],
        )
        if cached is not None:
            page_candidates, cached_calls, cached_successful_calls, page_warnings = cached
            if block.block_id in checkpoint_recoveries:
                checkpoint_recoveries.pop(block.block_id, None)
                write_json(checkpoint_path, checkpoint)
            candidates.extend(page_candidates)
            resumed_calls.extend(cached_calls)
            successful_resumed_calls.extend(cached_successful_calls)
            blocks_resumed += 1
            successfully_examined_blocks.append(block)
            block_attempts.append(
                {
                    "block_id": block.block_id,
                    "page": block.page,
                    "status": "resumed",
                    "completed_provider_call": True,
                }
            )
            warnings.extend(
                f"page {fragment.page} block {block.block_id}: {warning}"
                for warning in page_warnings
            )
            continue
        recovery_entry = checkpoint_recoveries.get(block.block_id)
        validated_recovery = _validated_legacy_recovery_entry(
            recovery_entry,
            block=block,
            spec=spec,
            settings=settings,
        )
        if recovery_entry is not None and validated_recovery is None:
            checkpoint_recoveries.pop(block.block_id, None)
            write_json(checkpoint_path, checkpoint)
        recovery_payloads = (
            list(recovery_entry["attempts"])
            if validated_recovery is not None and isinstance(recovery_entry, dict)
            else []
        )

        def checkpoint_recovery_attempt(
            attempt: dict[str, Any],
            current_block: ResultBlock = block,
            current_payloads: list[dict[str, Any]] = recovery_payloads,
        ) -> None:
            current_payloads.append(attempt)
            checkpoint_recoveries[current_block.block_id] = {
                "schema_version": _EXTRACTOR_RECOVERY_ENTRY_VERSION,
                "block_text_sha256": current_block.text_sha256,
                "max_depth": LEGACY_RECOVERY_MAX_DEPTH,
                "attempts": current_payloads,
            }
            write_json(checkpoint_path, checkpoint)

        page_result: tuple[list[CandidateObservation], ProviderCall, list[str]] | None = None
        extraction_error: Exception | None = None
        base_attempt_resumed = validated_recovery is not None
        if validated_recovery is not None:
            recovery_attempts, _ = validated_recovery
            base_attempt = recovery_attempts[0]
            base_call = base_attempt["call"]
            assert isinstance(base_call, ProviderCall)
            returned_model_mismatch = base_attempt["status"] == _RETURNED_MODEL_MISMATCH
            validation_code = (
                "wire_validation"
                if returned_model_mismatch
                else base_attempt["status"].removeprefix("provider_response_")
            )
            extraction_error = ProviderResponseValidationError(
                call=base_call,
                code=validation_code,
                validation_path=(("model",) if returned_model_mismatch else ()),
                validation_keyword=("returned_model_mismatch" if returned_model_mismatch else None),
            )
            resumed_calls.append(base_call)
        else:
            try:
                page_result = extract_page_candidates(
                    client=client,
                    model=settings.model,
                    paper_id=spec.paper_id,
                    paper_title=spec.title,
                    fragment=fragment,
                    max_tokens=settings.max_tokens,
                    temperature=settings.temperature,
                    reasoning_effort=settings.reasoning_effort,
                    seed=settings.seed,
                )
            except ProviderBudgetError:
                raise
            except Exception as error:
                extraction_error = error

        if extraction_error is not None:
            error = extraction_error
            error_code, completed_call, safe_details = _extractor_failure(error)
            if completed_call is not None and not base_attempt_resumed:
                calls.append(completed_call)
            recovery = None
            if isinstance(error, ProviderResponseValidationError):
                if not base_attempt_resumed:
                    checkpoint_recovery_attempt(
                        _legacy_recovery_attempt_payload(
                            block=block,
                            depth=0,
                            status=error_code,
                            call=completed_call,
                            safe_details=safe_details,
                        )
                    )
                recovery = _recover_split_block(
                    client=client,
                    settings=settings,
                    spec=spec,
                    block=block,
                    resume_attempts=(
                        validated_recovery[0][1:] if validated_recovery is not None else None
                    ),
                    on_attempt=checkpoint_recovery_attempt,
                )
            recovery_metadata: dict[str, Any] = {}
            if recovery is not None:
                candidates.extend(recovery.candidates)
                calls.extend(recovery.new_calls)
                resumed_calls.extend(recovery.resumed_calls)
                successful_new_calls.extend(recovery.new_successful_calls)
                successful_resumed_calls.extend(recovery.resumed_successful_calls)
                warnings.extend(
                    f"page {fragment.page} block {block.block_id}: {warning}"
                    for warning in recovery.warnings
                )
                recovery_metadata = {
                    "recovery_calls": len(recovery.calls),
                    "recovery_successful_calls": len(recovery.successful_calls),
                    "recovery_validation_failed_calls": (
                        len(recovery.calls) - len(recovery.successful_calls)
                    ),
                    "recovery_max_depth_reached": recovery.max_depth_reached,
                    "recovery_terminal_failures": [
                        asdict(failure) for failure in recovery.terminal_failures
                    ],
                }
            if recovery is None or not recovery.succeeded:
                if recovery is None and block.block_id in checkpoint_recoveries:
                    checkpoint_recoveries.pop(block.block_id, None)
                    write_json(checkpoint_path, checkpoint)
                blocks_failed += 1
                block_attempts.append(
                    {
                        "block_id": block.block_id,
                        "page": block.page,
                        "status": "failed",
                        "error_code": error_code,
                        "completed_provider_call": completed_call is not None,
                        **recovery_metadata,
                        **safe_details,
                    }
                )
                warnings.append(
                    f"page {fragment.page} block {block.block_id}: extractor_error={error_code}"
                )
                continue
            blocks_succeeded += 1
            successfully_examined_blocks.append(block)
            block_attempts.append(
                {
                    "block_id": block.block_id,
                    "page": block.page,
                    "status": "recovered_by_split",
                    "error_code": error_code,
                    "completed_provider_call": True,
                    **recovery_metadata,
                    **safe_details,
                }
            )
            warnings.append(
                f"page {fragment.page} block {block.block_id}: "
                f"extractor_error={error_code} recovered_by_split"
            )
            checkpoint_blocks[block.block_id] = {
                "block_text_sha256": block.text_sha256,
                "attempts": list(recovery_payloads),
                "candidates": [
                    candidate.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for candidate in recovery.candidates
                ],
                "calls": [
                    attempt["call"]
                    for attempt in recovery_payloads
                    if isinstance(attempt.get("call"), dict)
                ],
                "successful_call_indexes": [
                    index
                    for index, attempt in enumerate(
                        item for item in recovery_payloads if isinstance(item.get("call"), dict)
                    )
                    if attempt.get("status") == "success"
                ],
                "warnings": recovery.warnings,
            }
            checkpoint_recoveries.pop(block.block_id, None)
            write_json(checkpoint_path, checkpoint)
            continue
        assert page_result is not None
        page_candidates, call, page_warnings = page_result
        candidates.extend(page_candidates)
        calls.append(call)
        successful_new_calls.append(call)
        blocks_succeeded += 1
        successfully_examined_blocks.append(block)
        block_attempts.append(
            {
                "block_id": block.block_id,
                "page": block.page,
                "status": "success",
                "completed_provider_call": True,
            }
        )
        warnings.extend(
            f"page {fragment.page} block {block.block_id}: {warning}" for warning in page_warnings
        )
        checkpoint_blocks[block.block_id] = {
            "block_text_sha256": block.text_sha256,
            "attempts": [
                _legacy_recovery_attempt_payload(
                    block=block,
                    depth=0,
                    status="success",
                    call=call,
                    candidates=page_candidates,
                    warnings=page_warnings,
                )
            ],
            "candidates": [
                candidate.model_dump(mode="json", by_alias=True, exclude_none=True)
                for candidate in page_candidates
            ],
            "calls": [call.model_dump(mode="json")],
            "successful_call_indexes": [0],
            "warnings": page_warnings,
        }
        write_json(checkpoint_path, checkpoint)

    if row_plan is not None:
        assert row_checkpoint is not None
        for batch in row_plan.batches:
            checkpoint_entry = row_checkpoint_batches.get(batch.batch_id)
            cached_row = _validated_row_checkpoint_entry(
                checkpoint_entry,
                batch=batch,
                contract=row_checkpoint["contract"],
            )
            if cached_row is not None:
                _merge_row_outcome(row_outcome, cached_row)
                row_batches_resumed += 1
                continue
            resumed_prefix = _validated_row_checkpoint_prefix(
                checkpoint_entry,
                batch=batch,
                contract=row_checkpoint["contract"],
            )
            if checkpoint_entry is not None and resumed_prefix is None:
                row_checkpoint_batches.pop(batch.batch_id, None)
                write_json(row_checkpoint_path, row_checkpoint)

            def checkpoint_row_progress(
                progress: RowEnumerationOutcome,
                current_batch: RowBatch = batch,
            ) -> None:
                row_checkpoint_batches[current_batch.batch_id] = {
                    "batch_sha256": sha256_bytes(canonical_json_bytes(current_batch)),
                    **_row_outcome_payload(progress),
                }
                write_json(row_checkpoint_path, row_checkpoint)

            batch_outcome = enumerate_row_batch(
                client=client,
                model=_effective_row_model(settings),
                paper_id=spec.paper_id,
                paper_title=spec.title,
                batch=batch,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                reasoning_effort=settings.reasoning_effort,
                seed=settings.seed,
                max_recovery_depth=row_plan.config.max_recovery_depth,
                resume_outcome=resumed_prefix,
                on_progress=checkpoint_row_progress,
            )
            _merge_row_outcome(row_outcome, batch_outcome)
            row_checkpoint_batches[batch.batch_id] = {
                "batch_sha256": sha256_bytes(canonical_json_bytes(batch)),
                **_row_outcome_payload(batch_outcome),
            }
            write_json(row_checkpoint_path, row_checkpoint)
        row_terminal_ledger = validate_outcome_against(
            row_plan,
            row_outcome,
            paper_id=spec.paper_id,
            paper_title=spec.title,
            model=_effective_row_model(settings),
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            reasoning_effort=settings.reasoning_effort,
            seed=settings.seed,
        )
        row_terminal_sha256 = write_json(
            output_dir / "private" / "row-terminal-states.json",
            row_terminal_ledger,
        )
        candidates.extend(row_outcome.candidates)
        write_json(
            output_dir / "private" / "row-enumeration.json",
            {
                "schema_version": "row-enumeration-outcome/0.3",
                "plan_sha256": row_plan_sha256,
                "terminal_states_sha256": row_terminal_sha256,
                "calls": [call.model_dump(mode="json") for call in row_outcome.calls],
                "attempts": [attempt.model_dump(mode="json") for attempt in row_outcome.attempts],
                "warnings": row_outcome.warnings,
                "telemetry": row_outcome.telemetry,
            },
        )
        if row_outcome.unresolved_row_ids:
            warnings.append(f"row_enumeration_unresolved={len(row_outcome.unresolved_row_ids)}")
        if row_outcome.unbatchable_row_ids:
            warnings.append(f"row_enumeration_unbatchable={len(row_outcome.unbatchable_row_ids)}")
        if row_terminal_ledger.protocol_events:
            warnings.append(
                f"row_enumeration_protocol_events={len(row_terminal_ledger.protocol_events)}"
            )

    candidates_before_deduplication = len(candidates)
    candidates = validate_non_origin_candidates(
        candidates,
        {layout.source_id: layout},
        min_confidence=settings.min_confidence,
        processed_source_ids=source_processing.processed_source_ids,
    )
    deduplication = deduplicate_candidates_with_lineage(
        candidates,
        {layout.source_id: layout},
    )
    candidates = deduplication.candidates
    duplicates_removed = candidates_before_deduplication - len(candidates)
    candidates = validate_non_origin_candidates(
        candidates,
        {layout.source_id: layout},
        min_confidence=settings.min_confidence,
        processed_source_ids=source_processing.processed_source_ids,
    )
    if settings.tuple_model is None:
        for candidate in candidates:
            observation_id = candidate.observation_id or candidate.stable_id()
            tuple_pass_ids.add(observation_id)
            tuple_gate_by_observation_id[observation_id] = _tuple_gate_sha256(
                candidate=candidate,
                tuple_contract_sha256=tuple_checkpoint_contract_sha256,
                entry_sha256=None,
                status="disabled",
                semantic_match=True,
                passed=True,
            )
    else:
        assert tuple_checkpoint is not None
        selected_tuple_keys = {
            tuple_candidate_binding_sha256(candidate)
            for candidate in candidates
            if candidate.claim_type is ClaimType.PRIMARY_RESULT
            and candidate.export_status is ExportStatus.ELIGIBLE
        }
        if set(tuple_checkpoint_candidates) - selected_tuple_keys:
            for stale_key in set(tuple_checkpoint_candidates) - selected_tuple_keys:
                tuple_checkpoint_candidates.pop(stale_key, None)
            write_json(tuple_checkpoint_path, tuple_checkpoint)
        for candidate in candidates:
            observation_id = candidate.observation_id or candidate.stable_id()
            if (
                candidate.claim_type is not ClaimType.PRIMARY_RESULT
                or candidate.export_status is not ExportStatus.ELIGIBLE
            ):
                tuple_gate_by_observation_id[observation_id] = _tuple_gate_sha256(
                    candidate=candidate,
                    tuple_contract_sha256=tuple_checkpoint_contract_sha256,
                    entry_sha256=None,
                    status="not_selected",
                    semantic_match=False,
                    passed=False,
                )
                continue
            pre_gate_candidate = candidate.model_copy(deep=True)
            tuple_key = tuple_candidate_binding_sha256(candidate)
            cached_tuple = _validated_tuple_checkpoint_entry(
                tuple_checkpoint_candidates.get(tuple_key),
                candidate=candidate,
                settings=settings,
                manifest=manifest,
                layout=layout,
                code_state=code_state,
                tuple_contract_sha256=tuple_checkpoint_contract_sha256,
            )
            resumed = cached_tuple is not None
            if cached_tuple is not None:
                tuple_status, tuple_assessment, tuple_call = cached_tuple
                tuple_candidates_resumed += 1
                if tuple_call is not None:
                    tuple_resumed_calls.append(tuple_call)
            else:
                if tuple_key in tuple_checkpoint_candidates:
                    tuple_checkpoint_candidates.pop(tuple_key, None)
                    write_json(tuple_checkpoint_path, tuple_checkpoint)
                try:
                    tuple_input = build_tuple_resolution_input(candidate, layout)
                except ValueError as error:
                    tuple_status = "unsupported"
                    tuple_assessment = None
                    tuple_call = None
                    error_code = _tuple_input_failure_code(error)
                    tuple_checkpoint_candidates[tuple_key] = _new_tuple_checkpoint_entry(
                        status=tuple_status,
                        candidate=candidate,
                        request=None,
                        settings=settings,
                        manifest=manifest,
                        layout=layout,
                        code_state=code_state,
                        tuple_contract_sha256=tuple_checkpoint_contract_sha256,
                        error_code=error_code,
                    )
                    write_json(tuple_checkpoint_path, tuple_checkpoint)
                else:
                    try:
                        tuple_proposal, tuple_assessment, tuple_call = propose_tuple_resolution(
                            client=client,
                            model=settings.tuple_model,
                            request=tuple_input,
                            max_tokens=settings.tuple_max_tokens,
                            temperature=TUPLE_TEMPERATURE,
                            reasoning_effort=TUPLE_REASONING_EFFORT,
                            seed=TUPLE_SEED,
                            require_parameters=True,
                        )
                    except ProviderBudgetError:
                        raise
                    except Exception as error:
                        error_code, tuple_call = _tuple_provider_failure(error)
                        tuple_status = (
                            "response_failure" if tuple_call is not None else "provider_failure"
                        )
                        tuple_assessment = None
                        if tuple_call is not None:
                            tuple_calls.append(tuple_call)
                        tuple_checkpoint_candidates[tuple_key] = _new_tuple_checkpoint_entry(
                            status=tuple_status,
                            candidate=candidate,
                            request=tuple_input,
                            settings=settings,
                            manifest=manifest,
                            layout=layout,
                            code_state=code_state,
                            tuple_contract_sha256=tuple_checkpoint_contract_sha256,
                            call=tuple_call,
                            error_code=error_code,
                        )
                        write_json(tuple_checkpoint_path, tuple_checkpoint)
                    else:
                        tuple_status = "success"
                        tuple_calls.append(tuple_call)
                        tuple_checkpoint_candidates[tuple_key] = _new_tuple_checkpoint_entry(
                            status=tuple_status,
                            candidate=candidate,
                            request=tuple_input,
                            settings=settings,
                            manifest=manifest,
                            layout=layout,
                            code_state=code_state,
                            tuple_contract_sha256=tuple_checkpoint_contract_sha256,
                            proposal=tuple_proposal,
                            assessment=tuple_assessment,
                            call=tuple_call,
                        )
                        write_json(tuple_checkpoint_path, tuple_checkpoint)
            tuple_entry = tuple_checkpoint_candidates[tuple_key]
            semantic_match = bool(
                tuple_status == "success"
                and tuple_assessment is not None
                and tuple_assessment_is_export_concordant(candidate, tuple_assessment)
            )
            passed = bool(
                semantic_match
                and tuple_assessment is not None
                and tuple_assessment.decision
                in {TupleResolutionDecision.VERIFIED, TupleResolutionDecision.REVIEW}
            )
            if passed:
                tuple_pass_ids.add(observation_id)
            else:
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                if tuple_status == "unsupported":
                    candidate.export_reason = "tuple_resolution=unsupported_physical_evidence"
                    tuple_candidates_unsupported += 1
                elif tuple_status != "success":
                    candidate.export_reason = f"tuple_resolution={tuple_status}"
                    tuple_candidates_failed += 1
                elif not semantic_match:
                    candidate.export_reason = "tuple_resolution=candidate_mismatch"
                elif tuple_assessment is not None and (
                    tuple_assessment.decision is not TupleResolutionDecision.VERIFIED
                ):
                    candidate.export_reason = f"tuple_resolution={tuple_assessment.decision.value}"
                else:
                    candidate.export_reason = "tuple_resolution=candidate_mismatch"
            gate_sha256 = _tuple_gate_sha256(
                candidate=pre_gate_candidate,
                tuple_contract_sha256=tuple_checkpoint_contract_sha256,
                entry_sha256=tuple_entry["entry_sha256"],
                status=tuple_status,
                semantic_match=semantic_match,
                passed=passed,
            )
            tuple_gate_by_observation_id[observation_id] = gate_sha256
            tuple_outcomes.append(
                {
                    "observation_id": observation_id,
                    "status": tuple_status,
                    "resumed": resumed,
                    "semantic_match": semantic_match,
                    "passed": passed,
                    "gate_sha256": gate_sha256,
                    "checkpoint_entry": tuple_entry,
                }
            )
        tuple_sidecar_sha256 = write_json(
            output_dir / "private" / "tuple-resolution.json",
            {
                "schema_version": "tuple-resolution-run/0.1",
                "configuration": _tuple_run_configuration(settings),
                "checkpoint_contract_sha256": tuple_checkpoint_contract_sha256,
                "candidate_gates": dict(sorted(tuple_gate_by_observation_id.items())),
                "outcomes": tuple_outcomes,
            },
        )
    verifier_accept_ids: set[str] = set()
    if settings.verifier_model:
        assert verifier_checkpoint is not None
        for candidate in candidates:
            observation_id = candidate.observation_id or candidate.stable_id()
            if (
                observation_id not in tuple_pass_ids
                or observation_id not in tuple_gate_by_observation_id
                or candidate.claim_type != ClaimType.PRIMARY_RESULT
                or candidate.export_status != ExportStatus.ELIGIBLE
            ):
                continue
            verifier_candidates_selected += 1
            support = bind_candidate_block(candidate, blocks)
            if support is None:
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                candidate.export_reason = "no frozen result block contains the evidence quote"
                verifier_candidates_unbound += 1
                continue
            block, anchor = support
            evidence_block = frozen_evidence_block(
                paper_id=spec.paper_id, block=block, anchor=anchor
            )
            verifier_key = _verifier_candidate_sha256(candidate)
            cached_verification = _validated_verifier_checkpoint_entry(
                verifier_checkpoint_candidates.get(verifier_key),
                candidate=candidate,
                evidence_block=evidence_block,
                settings=settings,
                manifest=manifest,
                layout=layout,
                code_state=code_state,
                tuple_gate_sha256=tuple_gate_by_observation_id[observation_id],
            )
            if cached_verification is not None:
                verifier_status, verification, verification_call = cached_verification
                if verification_call is not None:
                    verifier_resumed_calls.append(verification_call)
                    if verifier_status == "success":
                        verifier_successful_resumed_calls.append(verification_call)
                verifier_candidates_resumed += 1
            else:
                verifier_candidates_executed += 1
                if verifier_key in verifier_checkpoint_candidates:
                    verifier_checkpoint_candidates.pop(verifier_key)
                    write_json(verifier_checkpoint_path, verifier_checkpoint)
                try:
                    verification, verification_call = verify_candidate(
                        client=client,
                        model=settings.verifier_model,
                        candidate=candidate,
                        evidence_block=evidence_block,
                        max_tokens=settings.verifier_max_tokens,
                    )
                except ProviderBudgetError:
                    raise
                except Exception as error:
                    error_code, verification_call = _verifier_failure(error)
                    verifier_status = (
                        "response_failure" if verification_call is not None else "provider_failure"
                    )
                    verification = None
                    if verification_call is not None:
                        verifier_calls.append(verification_call)
                    verifier_checkpoint_candidates[verifier_key] = _new_verifier_checkpoint_entry(
                        status=verifier_status,
                        candidate=candidate,
                        evidence_block=evidence_block,
                        call=verification_call,
                        error_code=error_code,
                        settings=settings,
                        manifest=manifest,
                        layout=layout,
                        code_state=code_state,
                        tuple_gate_sha256=tuple_gate_by_observation_id[observation_id],
                    )
                    write_json(verifier_checkpoint_path, verifier_checkpoint)
                else:
                    verifier_status = "success"
                    verifier_calls.append(verification_call)
                    verifier_successful_calls.append(verification_call)
                    verifier_checkpoint_candidates[verifier_key] = _new_verifier_checkpoint_entry(
                        status=verifier_status,
                        candidate=candidate,
                        evidence_block=evidence_block,
                        verification=verification,
                        call=verification_call,
                        settings=settings,
                        manifest=manifest,
                        layout=layout,
                        code_state=code_state,
                        tuple_gate_sha256=tuple_gate_by_observation_id[observation_id],
                    )
                    write_json(verifier_checkpoint_path, verifier_checkpoint)
            if verifier_status != "success" or verification is None:
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                error_code = str(verifier_checkpoint_candidates[verifier_key]["error_code"])
                candidate.export_reason = f"independent_verifier={error_code}"
                verifier_candidates_failed += 1
                warnings.append(f"independent_verifier_failed={error_code}")
                continue
            assert verifier_checkpoint_contract_sha256 is not None
            verifier_entry = verifier_checkpoint_candidates[verifier_key]
            verifier_gate_by_observation_id[observation_id] = _verifier_gate_sha256(
                candidate=candidate,
                tuple_gate_sha256=tuple_gate_by_observation_id[observation_id],
                verifier_contract_sha256=verifier_checkpoint_contract_sha256,
                verifier_entry_sha256=verifier_entry["entry_sha256"],
                verification=verification,
            )
            verifications.append(verification)
            if verification.effective_decision == IndependentDecision.ACCEPT:
                verifier_accept_ids.add(observation_id)
            else:
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                candidate.export_reason = (
                    f"independent_verifier={verification.effective_decision.value}: "
                    f"{verification.provider_assessment.justification}"
                )
    if settings.verifier_model is not None:
        verifier_sidecar_sha256 = write_json(
            output_dir / "private" / "verifier-gates.json",
            {
                "schema_version": "independent-verifier-gates/0.2",
                "configuration": _verifier_run_configuration(settings),
                "checkpoint_contract_sha256": verifier_checkpoint_contract_sha256,
                "candidate_gates": dict(sorted(verifier_gate_by_observation_id.items())),
                "passed_candidate_ids": sorted(verifier_accept_ids),
            },
        )
    candidates = route_candidate_attribution(candidates, {layout.source_id: layout})
    if settings.origin_model is not None:
        assert origin_checkpoint is not None
        if settings.verifier_model is None:
            warnings.append("origin_retrieval_skipped=independent_verifier_required")
        for candidate in candidates:
            observation_id = candidate.observation_id or candidate.stable_id()
            if observation_id not in verifier_accept_ids:
                continue
            tuple_gate_sha256 = tuple_gate_by_observation_id.get(observation_id)
            if tuple_gate_sha256 is None or observation_id not in tuple_pass_ids:
                continue
            verifier_gate_sha256 = verifier_gate_by_observation_id.get(observation_id)
            if verifier_gate_sha256 is None:
                continue
            if (
                candidate.attribution is not None
                and candidate.attribution.state is AttributionState.EXTERNALLY_SOURCED
            ):
                origin_outcomes.append(
                    {
                        "observation_id": observation_id,
                        "status": "deterministic_external",
                        "resumed": False,
                        "effective_state": AttributionState.EXTERNALLY_SOURCED.value,
                        "route": "demote",
                        "reason_codes": ["deterministic_external_preserved"],
                    }
                )
                continue
            origin_key = candidate_origin_binding_sha256(candidate)
            cached_origin = _validated_origin_checkpoint_entry(
                origin_checkpoint_candidates.get(origin_key),
                candidate=candidate,
                layout=layout,
                settings=settings,
                code_state=code_state,
                tuple_gate_sha256=tuple_gate_sha256,
                verifier_gate_sha256=verifier_gate_sha256,
            )
            resumed = cached_origin is not None
            if cached_origin is not None:
                (
                    origin_status,
                    bundle,
                    proposal,
                    assessment,
                    origin_call,
                    origin_error_code,
                ) = cached_origin
                del bundle, proposal
                if origin_call is not None:
                    origin_resumed_calls.append(origin_call)
                origin_candidates_resumed += 1
                if origin_status != "success":
                    assert origin_error_code is not None
                    candidate.export_status = ExportStatus.NEEDS_REVIEW
                    candidate.export_reason = f"origin_retrieval={origin_error_code}"
                    origin_candidates_failed += 1
                    origin_outcomes.append(
                        {
                            "observation_id": observation_id,
                            "status": origin_status,
                            "resumed": True,
                            "effective_state": (
                                candidate.attribution.state.value
                                if candidate.attribution is not None
                                else AttributionState.UNRESOLVED.value
                            ),
                            "route": "review",
                            "error_code": origin_error_code,
                            "completed_provider_call": origin_call is not None,
                        }
                    )
                    warnings.append(f"origin_retrieval_failed={origin_error_code}")
                    continue
                assert assessment is not None and origin_call is not None
            else:
                if origin_key in origin_checkpoint_candidates:
                    origin_checkpoint_candidates.pop(origin_key)
                    write_json(origin_checkpoint_path, origin_checkpoint)
                bundle = None
                origin_call = None
                try:
                    bundle = retrieve_origin_context(candidate, layout)
                    proposal, origin_call = propose_producer_origin(
                        client=client,
                        model=settings.origin_model,
                        candidate=candidate,
                        bundle=bundle,
                        max_tokens=settings.origin_max_tokens,
                    )
                    assessment = verify_producer_origin_proposal(
                        candidate=candidate,
                        layout=layout,
                        bundle=bundle,
                        proposal=proposal,
                    )
                except ProviderBudgetError:
                    raise
                except Exception as error:
                    error_code, completed_call = _origin_failure(error)
                    if completed_call is None and origin_call is not None:
                        completed_call = origin_call
                    origin_status = (
                        "response_failure" if completed_call is not None else "provider_failure"
                    )
                    if completed_call is not None:
                        origin_calls.append(completed_call)
                    origin_checkpoint_candidates[origin_key] = _new_origin_checkpoint_entry(
                        status=origin_status,
                        candidate=candidate,
                        layout=layout,
                        bundle=bundle,
                        settings=settings,
                        code_state=code_state,
                        tuple_gate_sha256=tuple_gate_sha256,
                        verifier_gate_sha256=verifier_gate_sha256,
                        call=completed_call,
                        error_code=error_code,
                    )
                    write_json(origin_checkpoint_path, origin_checkpoint)
                    candidate.export_status = ExportStatus.NEEDS_REVIEW
                    candidate.export_reason = f"origin_retrieval={error_code}"
                    origin_candidates_failed += 1
                    origin_outcomes.append(
                        {
                            "observation_id": observation_id,
                            "status": origin_status,
                            "resumed": False,
                            "effective_state": (
                                candidate.attribution.state.value
                                if candidate.attribution is not None
                                else AttributionState.UNRESOLVED.value
                            ),
                            "route": "review",
                            "error_code": error_code,
                            "completed_provider_call": completed_call is not None,
                        }
                    )
                    warnings.append(f"origin_retrieval_failed={error_code}")
                    continue
                origin_calls.append(origin_call)
                origin_checkpoint_candidates[origin_key] = _new_origin_checkpoint_entry(
                    status="success",
                    candidate=candidate,
                    layout=layout,
                    bundle=bundle,
                    proposal=proposal,
                    assessment=assessment,
                    call=origin_call,
                    settings=settings,
                    code_state=code_state,
                    tuple_gate_sha256=tuple_gate_sha256,
                    verifier_gate_sha256=verifier_gate_sha256,
                )
                write_json(origin_checkpoint_path, origin_checkpoint)
            _apply_origin_assessment(candidate, assessment)
            origin_outcomes.append(
                {
                    "observation_id": observation_id,
                    "status": "success",
                    "resumed": resumed,
                    "effective_state": assessment.effective_state.value,
                    "route": assessment.route.value,
                    "reason_codes": assessment.reason_codes,
                    "candidate_binding_sha256": assessment.candidate_binding_sha256,
                    "retrieval_checkpoint_sha256": assessment.retrieval_checkpoint_sha256,
                    "proposal_sha256": assessment.proposal_sha256,
                    "positive_evidence_verified": assessment.positive_evidence_verified,
                    "automatic_export": assessment.allows_automatic_export,
                }
            )
        write_json(
            output_dir / "private" / "origin-retrieval.json",
            {
                "schema_version": "producer-origin-run/0.1",
                "configuration": _origin_run_configuration(settings),
                "outcomes": origin_outcomes,
            },
        )
    if settings.tuple_model is not None:
        for candidate in candidates:
            observation_id = candidate.observation_id or candidate.stable_id()
            if candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED} and (
                observation_id not in tuple_pass_ids
                or observation_id not in tuple_gate_by_observation_id
            ):
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                candidate.export_reason = "tuple_resolution=missing_verified_concordant_gate"
    if settings.tuple_model is not None and settings.verifier_model is None:
        for candidate in candidates:
            if candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}:
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                candidate.export_reason = "independent_verifier=disabled_review_only"
    schema, authority = load_schema(settings.schema_path, settings.schema_sha256)
    composition_candidates = [
        candidate
        for candidate in candidates
        if candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}
        and candidate.attribution is not None
        and candidate.attribution.allows_canonical_export
    ]
    if settings.tuple_model is not None and settings.verifier_model is not None:
        if tuple_sidecar_sha256 is None:
            raise ValueError("tuple-gated composition lacks its sealed private sidecar")
        export_provenance = tuple_gated_export_provenance(
            composition_candidates,
            tuple_sidecar_sha256=tuple_sidecar_sha256,
            tuple_gates={
                str(candidate.observation_id): tuple_gate_by_observation_id[
                    str(candidate.observation_id)
                ]
                for candidate in composition_candidates
            },
            verifier_sidecar_sha256=verifier_sidecar_sha256,
            verifier_gates=(
                {
                    str(candidate.observation_id): verifier_gate_by_observation_id[
                        str(candidate.observation_id)
                    ]
                    for candidate in composition_candidates
                }
                if settings.verifier_model is not None
                else None
            ),
        )
    elif settings.tuple_model is not None:
        if tuple_sidecar_sha256 is None:
            raise ValueError("tuple-only review mode lacks its sealed private sidecar")
        export_provenance = tuple_unverified_review_provenance(
            tuple_sidecar_sha256=tuple_sidecar_sha256
        )
    else:
        export_provenance = legacy_export_provenance(
            composition_candidates,
            reason="pipeline_tuple_stage_disabled",
        )
    records = compose_eee_records(
        manifest=manifest,
        candidates=candidates,
        schema_version=authority.version,
        provenance=export_provenance,
    )
    validation_errors: dict[str, list[str]] = {}
    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    eee_dir = output_dir / "eee"
    eee_dir.mkdir(parents=True, exist_ok=True)
    for stale_record in eee_dir.glob("*.json"):
        stale_record.unlink()
    invalid_output_path = output_dir / "private" / "invalid-eee.json"
    if invalid_output_path.exists():
        invalid_output_path.unlink()
    for record in records:
        issues = validate_eee_record(record, schema)
        evaluation_id = record["evaluation_id"]
        validation_errors[evaluation_id] = [f"{issue.path}: {issue.message}" for issue in issues]
        if issues:
            invalid_records.append(record)
            observation_ids = {
                result["evaluation_result_id"] for result in record["evaluation_results"]
            }
            for candidate in candidates:
                if candidate.observation_id in observation_ids:
                    candidate.export_status = ExportStatus.NEEDS_REVIEW
                    candidate.export_reason = "projected EEE record failed schema validation"
            continue
        valid_records.append(record)
        observation_ids = {
            result["evaluation_result_id"] for result in record["evaluation_results"]
        }
        for candidate in candidates:
            if (
                candidate.observation_id in observation_ids
                and candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}
                and candidate.attribution is not None
                and candidate.attribution.allows_canonical_export
            ):
                candidate.export_status = ExportStatus.EXPORTED
        filename = evaluation_id.rsplit("/", 1)[-1] + ".json"
        write_json(eee_dir / filename, record)
    if invalid_records:
        write_json(invalid_output_path, invalid_records)
    spot_checks = score_spot_checks(spec.expected_spot_checks, candidates)
    reference_score: dict[str, Any] | None = None
    reference_score_sha256: str | None = None
    if spec.reference_path:
        reference = load_reference(_reference_path(settings.project_root, spec.reference_path))
        if reference.paper_id != spec.paper_id:
            raise ValueError(
                f"reference paper_id mismatch: {reference.paper_id!r} != {spec.paper_id!r}"
            )
        if reference.source_sha256 != paper_source.sha256:
            raise ValueError("reference source hash does not match frozen paper")
        reference_score = score_reference(
            reference,
            candidates,
            control_examination=control_examination(
                reference,
                layout,
                successfully_examined_blocks,
            ),
            observation_examination=observation_examination(
                reference,
                layout,
                successfully_examined_blocks,
            ),
        )
        reference_score_sha256 = write_json(output_dir / "reference-score.json", reference_score)
    observations_sha256 = write_jsonl(output_dir / "observations.jsonl", candidates)
    candidate_lineage = build_candidate_lineage(
        paper_id=spec.paper_id,
        candidates=candidates,
        merge_kinds={
            observation_id: kind.value
            for observation_id, kind in deduplication.kind_by_observation_id.items()
        },
        observations_sha256=observations_sha256,
        valid_records=valid_records,
        invalid_records=invalid_records,
        export_provenance=export_provenance,
        tuple_gates=(tuple_gate_by_observation_id if settings.tuple_model is not None else {}),
        verifier_gates=(
            verifier_gate_by_observation_id if settings.verifier_model is not None else {}
        ),
    )
    candidate_lineage_sha256 = write_json(
        output_dir / "candidate-lineage.json",
        candidate_lineage,
    )
    write_jsonl(output_dir / "verifications.jsonl", verifications)
    write_json(
        output_dir / "spot-checks.json",
        [
            {
                "expected": item.expected.model_dump(mode="json"),
                "matched_observation_id": item.matched_observation_id,
                "exact_value": item.exact_value,
                "exact_page": item.exact_page,
                "notes": item.notes,
            }
            for item in spot_checks
        ],
    )
    render_review_report(
        manifest=manifest,
        candidates=candidates,
        eee_records=valid_records,
        validation_errors=validation_errors,
        output_path=output_dir / "review.html",
    )
    review_reasons: list[str] = []
    if not blocks:
        review_reasons.append("zero_selected_result_blocks")
    elif not candidates:
        review_reasons.append("selected_result_blocks_produced_zero_candidates")
    if not valid_records:
        review_reasons.append("zero_valid_eee_records")
    if invalid_records:
        review_reasons.append("eee_schema_validation_failure")
    if row_plan is not None:
        if row_outcome.unresolved_row_ids:
            review_reasons.append("row_enumeration_unresolved")
        if row_outcome.unbatchable_row_ids:
            review_reasons.append("row_enumeration_unbatchable")
        if row_terminal_ledger is not None and row_terminal_ledger.protocol_events:
            review_reasons.append("row_enumeration_unknown_ids")
        if any(
            record.disposition is RowDisposition.UNCERTAIN
            for record in row_outcome.records.values()
        ):
            review_reasons.append("row_enumeration_uncertain")
    candidates_needing_review = sum(
        candidate.export_status == ExportStatus.NEEDS_REVIEW for candidate in candidates
    )
    semantic_safety_reviews = sum(
        any(note.startswith("semantic safety:") for note in candidate.notes)
        for candidate in candidates
        if candidate.export_status == ExportStatus.NEEDS_REVIEW
    )
    if candidates_needing_review:
        review_reasons.append("candidate_review_required")
    if settings.tuple_model is not None and any(not item["passed"] for item in tuple_outcomes):
        review_reasons.append("tuple_resolution_review_required")
    if review_reasons:
        warnings.extend(f"paper_review_required={reason}" for reason in review_reasons)
    row_stage_incomplete = bool(
        row_terminal_ledger is not None
        and (row_terminal_ledger.counts.unresolved or row_terminal_ledger.counts.unsupported)
    )
    downstream_technical_failure = bool(
        tuple_candidates_failed or verifier_candidates_failed or origin_candidates_failed
    )
    extractor_failure_events: list[dict[str, Any]] = []
    for attempt in block_attempts:
        if isinstance(attempt.get("error_code"), str):
            extractor_failure_events.append(
                {
                    "error_code": attempt["error_code"],
                    "completed_provider_call": attempt.get("completed_provider_call") is True,
                }
            )
        terminal_failures = attempt.get("recovery_terminal_failures")
        if isinstance(terminal_failures, list):
            extractor_failure_events.extend(
                {
                    "error_code": terminal["error_code"],
                    "completed_provider_call": terminal.get("completed_provider_call") is True,
                }
                for terminal in terminal_failures
                if isinstance(terminal, dict) and isinstance(terminal.get("error_code"), str)
            )
    extractor_no_call_failures = [
        event for event in extractor_failure_events if not event["completed_provider_call"]
    ]
    extractor_requests_rejected = sum(
        event["error_code"] == "provider_request_rejected" for event in extractor_no_call_failures
    )
    extractor_transport_failures = sum(
        event["error_code"] == "provider_transport_failed" for event in extractor_no_call_failures
    )
    extractor_local_failures = (
        len(extractor_no_call_failures) - extractor_requests_rejected - extractor_transport_failures
    )
    extractor_calls_succeeded = len(successful_new_calls)
    extractor_calls_failed = len(calls) - extractor_calls_succeeded
    extractor_resumed_succeeded = len(successful_resumed_calls)
    extractor_resumed_failed = len(resumed_calls) - extractor_resumed_succeeded
    row_completed_calls, row_successful_calls = partition_row_provider_calls(row_outcome)
    if (
        min(
            extractor_calls_failed,
            extractor_resumed_failed,
            extractor_local_failures,
        )
        < 0
    ):
        raise RuntimeError("extractor execution accounting invariant failed")
    verifier_resumed_succeeded = len(verifier_successful_resumed_calls)
    verifier_resumed_failed = verifier_candidates_resumed - verifier_resumed_succeeded
    verifier_executed_succeeded = len(verifier_successful_calls)
    verifier_executed_failed = verifier_candidates_executed - verifier_executed_succeeded
    if any(
        (
            verifier_candidates_selected
            != (
                verifier_candidates_unbound
                + verifier_candidates_resumed
                + verifier_candidates_executed
            ),
            len(verifications) != verifier_resumed_succeeded + verifier_executed_succeeded,
            verifier_candidates_failed != verifier_resumed_failed + verifier_executed_failed,
            verifier_resumed_failed < 0,
            verifier_executed_failed < 0,
        )
    ):
        raise RuntimeError("independent verifier execution accounting invariant failed")
    run_status = (
        "partial_failure"
        if blocks_failed or row_stage_incomplete or downstream_technical_failure
        else "quality_failure"
        if invalid_records
        else "success"
    )
    run_manifest = {
        "schema_version": _PIPELINE_RUN_SCHEMA_VERSION,
        "status": run_status,
        "paper_id": spec.paper_id,
        "title": spec.title,
        "candidate_validation": _candidate_validation_run_configuration(settings),
        "review_state": {
            "status": "needs_review" if review_reasons else "ready",
            "reasons": review_reasons,
        },
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "source_processing": {
            "schema_version": source_processing.schema_version,
            "artifact_sha256": source_processing_sha256,
            "path": "source-processing.json",
            "counts": source_processing.counts.model_dump(mode="json"),
        },
        "candidate_lineage": {
            "schema_version": candidate_lineage.schema_version,
            "artifact_sha256": candidate_lineage_sha256,
            "path": "candidate-lineage.json",
            "counts": candidate_lineage.counts.model_dump(mode="json"),
            "tuple_gate_sidecar_sha256": tuple_sidecar_sha256,
            "export_provenance_mode": export_provenance.mode.value,
            "export_composition_sha256": export_provenance.sha256,
            "verifier_gate_required": export_provenance.verifier_gate_required,
            "verifier_gate_sidecar_sha256": verifier_sidecar_sha256,
        },
        "selected_pages": [fragment.page for fragment in selected_pages],
        "result_block_segmentation": asdict(block_config),
        "selected_blocks": [
            {
                "block_id": block.block_id,
                "page": block.page,
                "body_lines": [block.body_start_line, block.body_end_line],
                "context_lines": (
                    [block.context_start_line, block.context_end_line]
                    if block.context_start_line is not None
                    else None
                ),
            }
            for block in blocks
        ],
        "layout_parser": layout.parser,
        "layout_parser_version": layout.parser_version,
        "extractor": {
            **_extractor_run_configuration(settings),
            "calls": [_public_provider_call(call) for call in calls],
            "resumed_calls": [_public_provider_call(call) for call in resumed_calls],
            "execution": {
                "blocks_total": len(blocks),
                "blocks_succeeded": blocks_succeeded,
                "blocks_failed": blocks_failed,
                "blocks_resumed": blocks_resumed,
                "calls_succeeded": extractor_calls_succeeded,
                "calls_failed": extractor_calls_failed,
                "calls_resumed": len(resumed_calls),
                "calls_resumed_succeeded": extractor_resumed_succeeded,
                "calls_resumed_failed": extractor_resumed_failed,
                "no_call_failures": len(extractor_no_call_failures),
                "requests_rejected": extractor_requests_rejected,
                "transport_failures": extractor_transport_failures,
                "local_failures": extractor_local_failures,
            },
            "successful_call_telemetry": _provider_call_telemetry(
                [*successful_new_calls, *successful_resumed_calls]
            ),
            "completed_call_telemetry": _provider_call_telemetry(
                [*calls, *resumed_calls],
                basis=(
                    "all completed extractor calls, including response-validation failures and "
                    "validated checkpoint reuse; monetary and token totals are lower bounds"
                ),
            ),
            "block_attempts": block_attempts,
            "checkpoint": {
                "schema_version": _EXTRACTOR_CHECKPOINT_SCHEMA_VERSION,
                "contract_sha256": checkpoint_contract_sha256,
                "path": "private/extractor-checkpoint.json",
            },
        },
        "row_enumeration": {
            **_row_enumeration_run_configuration(settings),
            "plan_sha256": row_plan_sha256,
            "plan": row_plan.telemetry.model_dump(mode="json") if row_plan else None,
            "preflight": row_preflight,
            "outcome": row_outcome.telemetry,
            "terminal_states": (
                {
                    "schema_version": row_terminal_ledger.schema_version,
                    "artifact_sha256": row_terminal_sha256,
                    "path": "private/row-terminal-states.json",
                    "counts": row_terminal_ledger.counts.model_dump(mode="json"),
                }
                if row_terminal_ledger is not None
                else None
            ),
            "calls": [_public_provider_call(call) for call in row_completed_calls],
            "successful_call_telemetry": _provider_call_telemetry(
                row_successful_calls,
                basis=(
                    "response-valid bounded row-enumeration calls; cost, token, retry, and "
                    "attempt totals are lower bounds when provider metadata is unavailable"
                ),
            ),
            "completed_call_telemetry": _provider_call_telemetry(
                row_completed_calls,
                basis=(
                    "all completed bounded row-enumeration calls, including response-validation "
                    "failures; monetary and token totals are lower bounds"
                ),
            ),
            "attempts": [attempt.model_dump(mode="json") for attempt in row_outcome.attempts],
            "execution": {
                "batches_total": len(row_plan.batches) if row_plan else 0,
                "batches_resumed": row_batches_resumed,
                "batches_executed": (
                    len(row_plan.batches) - row_batches_resumed if row_plan else 0
                ),
                "invalid_rows_seen": len(row_outcome.invalid_row_reasons),
                "unknown_row_ids_seen": (
                    len(row_terminal_ledger.protocol_events)
                    if row_terminal_ledger is not None
                    else 0
                ),
            },
            "checkpoint": (
                {
                    "schema_version": _ROW_CHECKPOINT_SCHEMA_VERSION,
                    "contract_sha256": row_checkpoint_contract_sha256,
                    "path": "private/row-enumeration-checkpoint.json",
                }
                if row_plan is not None
                else None
            ),
        },
        "tuple_resolution": {
            **_tuple_run_configuration(settings),
            "calls": [_public_provider_call(call) for call in tuple_calls],
            "resumed_calls": [_public_provider_call(call) for call in tuple_resumed_calls],
            "completed_call_telemetry": _provider_call_telemetry(
                [*tuple_calls, *tuple_resumed_calls],
                basis=(
                    "completed tuple-resolution calls, including validated checkpoint reuse and "
                    "response-validation failures; monetary and token totals are lower bounds"
                ),
            ),
            "execution": {
                "candidates_selected": len(tuple_outcomes),
                "candidates_passed": sum(item["passed"] for item in tuple_outcomes),
                "candidates_routed_to_review": sum(not item["passed"] for item in tuple_outcomes),
                "candidates_unsupported": tuple_candidates_unsupported,
                "candidates_failed": tuple_candidates_failed,
                "candidates_mismatched": sum(
                    item["status"] == "success" and not item["semantic_match"]
                    for item in tuple_outcomes
                ),
                "candidates_resumed": tuple_candidates_resumed,
            },
            "sidecar": (
                {
                    "schema_version": "tuple-resolution-run/0.1",
                    "sha256": tuple_sidecar_sha256,
                    "path": "private/tuple-resolution.json",
                }
                if tuple_sidecar_sha256 is not None
                else None
            ),
            "checkpoint": (
                {
                    "schema_version": _TUPLE_CHECKPOINT_SCHEMA_VERSION,
                    "contract_sha256": tuple_checkpoint_contract_sha256,
                    "path": "private/tuple-resolution-checkpoint.json",
                }
                if settings.tuple_model is not None
                else None
            ),
        },
        "verifier": {
            **_verifier_run_configuration(settings),
            # Preserve required nullable request controls so the public call ledger can be
            # rehydrated under the same exact request contract as the private checkpoint.
            "calls": [_public_provider_call(call) for call in verifier_calls],
            "resumed_calls": [_public_provider_call(call) for call in verifier_resumed_calls],
            "successful_call_telemetry": _provider_call_telemetry(
                [*verifier_successful_calls, *verifier_successful_resumed_calls],
                basis=(
                    "successful independent-verifier calls, including validated checkpoint "
                    "reuse; monetary and token totals are lower bounds"
                ),
            ),
            "completed_call_telemetry": _provider_call_telemetry(
                [*verifier_calls, *verifier_resumed_calls],
                basis=(
                    "completed independent-verifier calls, including validated checkpoint "
                    "reuse and response-validation failures; monetary, token, retry, and "
                    "attempt totals are lower bounds"
                ),
            ),
            "execution": {
                "candidates_selected": verifier_candidates_selected,
                "candidates_unbound": verifier_candidates_unbound,
                "candidates_verified": len(verifications),
                "candidates_failed": verifier_candidates_failed,
                "candidates_resumed": verifier_candidates_resumed,
                "candidates_resumed_succeeded": verifier_resumed_succeeded,
                "candidates_resumed_failed": verifier_resumed_failed,
                "candidates_executed": verifier_candidates_executed,
                "candidates_executed_succeeded": verifier_executed_succeeded,
                "candidates_executed_failed": verifier_executed_failed,
            },
            "sidecar": (
                {
                    "schema_version": "independent-verifier-gates/0.2",
                    "sha256": verifier_sidecar_sha256,
                    "path": "private/verifier-gates.json",
                }
                if verifier_sidecar_sha256 is not None
                else None
            ),
            "checkpoint": (
                {
                    "schema_version": _VERIFIER_CHECKPOINT_SCHEMA_VERSION,
                    "contract_sha256": verifier_checkpoint_contract_sha256,
                    "path": "private/verifier-checkpoint.json",
                }
                if settings.verifier_model is not None
                else None
            ),
        },
        "origin_retrieval": {
            **_origin_run_configuration(settings),
            "calls": [_public_provider_call(call) for call in origin_calls],
            "resumed_calls": [_public_provider_call(call) for call in origin_resumed_calls],
            "completed_call_telemetry": _provider_call_telemetry(
                [*origin_calls, *origin_resumed_calls],
                basis=(
                    "completed producer-origin calls, including validated checkpoint reuse "
                    "and response-validation failures; monetary and token totals are lower bounds"
                ),
            ),
            "execution": {
                "candidates_selected": len(origin_outcomes),
                "candidates_resumed": origin_candidates_resumed,
                "candidates_failed": origin_candidates_failed,
                "candidates_deterministic_external": sum(
                    item.get("status") == "deterministic_external" for item in origin_outcomes
                ),
            },
            "checkpoint": (
                {
                    "schema_version": _ORIGIN_CHECKPOINT_SCHEMA_VERSION,
                    "contract_sha256": origin_checkpoint_contract_sha256,
                    "path": "private/origin-retrieval-checkpoint.json",
                }
                if settings.origin_model is not None
                else None
            ),
        },
        "eee_schema": {"version": authority.version, "sha256": authority.sha256},
        "code": code_state,
        "counts": {
            "candidates": len(candidates),
            "candidates_before_deduplication": candidates_before_deduplication,
            "duplicates_removed": duplicates_removed,
            "candidates_needing_review": candidates_needing_review,
            "semantic_safety_reviews": semantic_safety_reviews,
            "primary_results": sum(c.claim_type == "primary_result" for c in candidates),
            "exported": sum(c.export_status == "exported" for c in candidates),
            "eee_records": len(valid_records),
            "eee_schema_issues": sum(len(items) for items in validation_errors.values()),
            "tuple_candidates": len(tuple_outcomes),
            "tuple_passed": sum(item["passed"] for item in tuple_outcomes),
            "tuple_review": sum(not item["passed"] for item in tuple_outcomes),
            "tuple_unsupported": tuple_candidates_unsupported,
            "tuple_failed": tuple_candidates_failed,
            "tuple_resumed": tuple_candidates_resumed,
            "verifications": len(verifications),
            "verifier_accepts": sum(
                item.effective_decision == IndependentDecision.ACCEPT for item in verifications
            ),
            "verifier_rejects": sum(
                item.effective_decision == IndependentDecision.REJECT for item in verifications
            ),
            "verifier_reviews": sum(
                item.effective_decision == IndependentDecision.REVIEW for item in verifications
            ),
            "verifier_failed": verifier_candidates_failed,
            "verifier_resumed": verifier_candidates_resumed,
            "origin_candidates": len(origin_outcomes),
            "origin_resumed": origin_candidates_resumed,
            "origin_failed": origin_candidates_failed,
            "origin_positive_review_only": sum(
                item.get("status") == "success" and item.get("positive_evidence_verified") is True
                for item in origin_outcomes
            ),
            "origin_external": sum(
                item.get("effective_state") == AttributionState.EXTERNALLY_SOURCED.value
                for item in origin_outcomes
            ),
            "origin_unresolved": sum(
                item.get("effective_state") == AttributionState.UNRESOLVED.value
                for item in origin_outcomes
            ),
            "origin_no_signal": sum(
                item.get("effective_state") == AttributionState.NO_SIGNAL.value
                for item in origin_outcomes
            ),
            "spot_checks": len(spot_checks),
            "spot_checks_exact": sum(item.exact_value for item in spot_checks),
            "reference_observations": (
                reference_score["reference_observations"] if reference_score else 0
            ),
            "reference_true_positives": (
                reference_score["detection"]["true_positives"] if reference_score else 0
            ),
            "reference_false_positives": (
                reference_score["detection"]["false_positives"] if reference_score else 0
            ),
            "reference_false_negatives": (
                reference_score["detection"]["false_negatives"] if reference_score else 0
            ),
            "negative_control_false_primary": (
                reference_score.get("negative_control_safety", {}).get("false_primary_count", 0)
                if reference_score
                else 0
            ),
        },
        "reference_evaluation": (
            {
                "path": spec.reference_path,
                "score_path": "reference-score.json",
                "score_sha256": reference_score_sha256,
                "schema_version": reference_score["schema_version"],
                "coverage": reference_score["coverage"],
                "detection": reference_score["detection"],
                "field_accuracy": reference_score["field_accuracy"],
                "negative_control_safety": reference_score.get("negative_control_safety"),
            }
            if reference_score
            else None
        ),
        "warnings": warnings,
        "wall_clock_seconds": round(time.monotonic() - started, 6),
    }
    if isinstance(client, BudgetedProviderClient):
        run_manifest["provider_budget"] = client.summary
    write_json(output_dir / "run.json", run_manifest)
    return run_manifest


def _error_checkpoint_call_accounting(paper_root: Path) -> dict[str, Any]:
    """Recover typed completed-call lower bounds from durable stage checkpoints.

    This is an error-path projection, not checkpoint reuse.  It deliberately keeps
    every parseable completed ``ProviderCall`` while marking malformed or missing
    checkpoint state as incomplete; the corpus budget ledger remains authoritative
    for calls that completed before their stage checkpoint could be written.
    """

    specs = {
        "extractor": {
            "path": "private/extractor-checkpoint.json",
            "schema_version": _EXTRACTOR_CHECKPOINT_SCHEMA_VERSION,
            "collection": "blocks",
            "call_list": "calls",
            "progress_collection": "recoveries",
        },
        "row_enumeration": {
            "path": "private/row-enumeration-checkpoint.json",
            "schema_version": _ROW_CHECKPOINT_SCHEMA_VERSION,
            "collection": "batches",
            "call_list": "calls",
        },
        "tuple_resolution": {
            "path": "private/tuple-resolution-checkpoint.json",
            "schema_version": _TUPLE_CHECKPOINT_SCHEMA_VERSION,
            "collection": "candidates",
            "provider_call": "provider_call",
            "entry_hash": True,
        },
        "verifier": {
            "path": "private/verifier-checkpoint.json",
            "schema_version": _VERIFIER_CHECKPOINT_SCHEMA_VERSION,
            "collection": "candidates",
            "provider_call": "provider_call",
            "entry_hash": True,
        },
        "origin_retrieval": {
            "path": "private/origin-retrieval-checkpoint.json",
            "schema_version": _ORIGIN_CHECKPOINT_SCHEMA_VERSION,
            "collection": "candidates",
            "provider_call": "provider_call",
            "entry_hash": True,
        },
    }

    stages: dict[str, dict[str, Any]] = {}
    all_calls: list[ProviderCall] = []
    for stage, spec in specs.items():
        relative = str(spec["path"])
        path = paper_root / relative
        checkpoint: dict[str, Any] = {
            "path": relative,
            "schema_version": spec["schema_version"],
            "sha256": None,
            "parse_status": "missing",
            "records_examined": 0,
            "invalid_records": 0,
        }
        calls: list[ProviderCall] = []
        if path.is_symlink():
            checkpoint["parse_status"] = "invalid"
            checkpoint["invalid_records"] = 1
        elif path.is_file():
            try:
                raw = read_json(path)
                checkpoint["sha256"] = sha256_file(path)
            except (OSError, ValueError):
                checkpoint["parse_status"] = "invalid"
                checkpoint["invalid_records"] = 1
            else:
                contract = raw.get("contract") if isinstance(raw, dict) else None
                entries = raw.get(spec["collection"]) if isinstance(raw, dict) else None
                contract_sha256 = (
                    sha256_bytes(canonical_json_bytes(contract))
                    if isinstance(contract, dict)
                    else None
                )
                if (
                    not isinstance(raw, dict)
                    or raw.get("schema_version") != spec["schema_version"]
                    or raw.get("contract_sha256") != contract_sha256
                    or not isinstance(entries, dict)
                ):
                    checkpoint["parse_status"] = "invalid"
                    checkpoint["invalid_records"] = 1
                else:
                    invalid_records = 0

                    def recover_call(
                        value: Any,
                        recovered_calls: list[ProviderCall] = calls,
                    ) -> None:
                        nonlocal invalid_records
                        if value is None:
                            return
                        try:
                            call = ProviderCall.model_validate(value)
                            _provider_call_telemetry([call])
                            _public_provider_call(call)
                        except (TypeError, ValueError):
                            invalid_records += 1
                            return
                        recovered_calls.append(call)

                    for entry in entries.values():
                        checkpoint["records_examined"] += 1
                        if not isinstance(entry, dict):
                            invalid_records += 1
                            continue
                        if spec.get("entry_hash"):
                            recorded_sha256 = entry.get("entry_sha256")
                            unsigned = {
                                key: value for key, value in entry.items() if key != "entry_sha256"
                            }
                            if not isinstance(
                                recorded_sha256, str
                            ) or recorded_sha256 != sha256_bytes(canonical_json_bytes(unsigned)):
                                invalid_records += 1
                                continue
                        call_list_key = spec.get("call_list")
                        if isinstance(call_list_key, str):
                            raw_calls = entry.get(call_list_key)
                            if not isinstance(raw_calls, list):
                                invalid_records += 1
                                continue
                            for raw_call in raw_calls:
                                recover_call(raw_call)
                        provider_call_key = spec.get("provider_call")
                        if isinstance(provider_call_key, str):
                            recover_call(entry.get(provider_call_key))

                    progress_collection = spec.get("progress_collection")
                    progress_entries = raw.get(progress_collection)
                    if progress_collection is not None:
                        if not isinstance(progress_entries, dict):
                            invalid_records += 1
                        else:
                            for entry in progress_entries.values():
                                checkpoint["records_examined"] += 1
                                if not isinstance(entry, dict) or not isinstance(
                                    entry.get("attempts"), list
                                ):
                                    invalid_records += 1
                                    continue
                                for attempt in entry["attempts"]:
                                    if not isinstance(attempt, dict):
                                        invalid_records += 1
                                        continue
                                    raw_call = attempt.get("call")
                                    completed = attempt.get("completed_provider_call")
                                    if not isinstance(completed, bool) or (
                                        (raw_call is not None) != completed
                                    ):
                                        invalid_records += 1
                                    recover_call(raw_call)
                    checkpoint["invalid_records"] = invalid_records
                    checkpoint["parse_status"] = "complete" if invalid_records == 0 else "partial"
        telemetry = _provider_call_telemetry(
            calls,
            basis=(
                "typed completed calls recovered from the durable stage checkpoint after a "
                "paper error; values are lower bounds and the corpus budget ledger is "
                "authoritative for pre-checkpoint completions"
            ),
        )
        stages[stage] = {
            "calls": calls,
            "checkpoint": checkpoint,
            "telemetry": telemetry,
        }
        all_calls.extend(calls)
    return {
        "schema_version": "paper-error-provider-accounting/0.1",
        "basis": (
            "stage values are parseable checkpoint lower bounds; the corpus provider-budget "
            "ledger is authoritative for all reservations, completions, and committed cost"
        ),
        "checkpointed_completed_calls": len(all_calls),
        "checkpoint_parse_complete": all(
            stage["checkpoint"]["parse_status"] in {"complete", "missing"}
            for stage in stages.values()
        ),
        "stages": stages,
    }


def _bounded_checkpoint_progress(paper_root: Path) -> dict[str, Any]:
    """Summarize parseable partial artifacts without exposing their source text."""

    artifacts: dict[str, dict[str, Any]] = {}

    def load(relative: str) -> Any | None:
        path = paper_root / relative
        if path.is_symlink() or not path.is_file():
            return None
        try:
            value = read_json(path)
            digest = sha256_file(path)
            size_bytes = path.stat().st_size
        except (OSError, ValueError):
            return None
        artifacts[relative] = {"sha256": digest, "size_bytes": size_bytes}
        return value

    selected_blocks: list[dict[str, Any]] = []
    selected_pages: list[int] = []
    raw_blocks = load("private/result-blocks.json")
    if isinstance(raw_blocks, list):
        try:
            blocks = [ResultBlock.model_validate(item) for item in raw_blocks]
        except (TypeError, ValueError):
            blocks = []
        selected_pages = sorted({block.page for block in blocks})
        selected_blocks = [
            {
                "block_id": block.block_id,
                "page": block.page,
                "body_lines": [block.body_start_line, block.body_end_line],
                "context_lines": (
                    [block.context_start_line, block.context_end_line]
                    if block.context_start_line is not None
                    else None
                ),
            }
            for block in blocks
        ]

    layout_summary: dict[str, Any] | None = None
    raw_layout = load("private/layout.json")
    if isinstance(raw_layout, dict):
        try:
            layout = PdfLayout.model_validate(raw_layout)
        except ValueError:
            pass
        else:
            layout_summary = {
                "parser": layout.parser,
                "parser_version": layout.parser_version,
                "page_count": layout.page_count,
            }

    row_plan_summary: dict[str, Any] | None = None
    raw_plan = load("private/row-enumeration-plan.json")
    if isinstance(raw_plan, dict):
        try:
            plan = RowEnumerationPlan.model_validate(raw_plan)
        except ValueError:
            pass
        else:
            row_plan_summary = {
                "sha256": artifacts["private/row-enumeration-plan.json"]["sha256"],
                "telemetry": plan.telemetry.model_dump(mode="json"),
            }

    def checkpoint_count(
        relative: str,
        collection: str,
        *,
        progress_collection: str | None = None,
    ) -> tuple[int, int]:
        raw = load(relative)
        if not isinstance(raw, dict) or not isinstance(raw.get(collection), dict):
            return 0, 0
        entries = raw[collection]
        completed_calls = 0
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            calls = entry.get("calls")
            if isinstance(calls, list):
                completed_calls += len(calls)
            elif isinstance(entry.get("provider_call"), dict):
                completed_calls += 1
        progress_entries = raw.get(progress_collection) if progress_collection is not None else None
        if isinstance(progress_entries, dict):
            for entry in progress_entries.values():
                if not isinstance(entry, dict) or not isinstance(entry.get("attempts"), list):
                    continue
                completed_calls += sum(
                    isinstance(attempt, dict) and isinstance(attempt.get("call"), dict)
                    for attempt in entry["attempts"]
                )
            return len(entries) + len(progress_entries), completed_calls
        return len(entries), completed_calls

    extractor_entries, extractor_calls = checkpoint_count(
        "private/extractor-checkpoint.json",
        "blocks",
        progress_collection="recoveries",
    )
    row_entries, row_calls = checkpoint_count("private/row-enumeration-checkpoint.json", "batches")
    tuple_entries, tuple_calls = checkpoint_count(
        "private/tuple-resolution-checkpoint.json", "candidates"
    )
    verifier_entries, verifier_calls = checkpoint_count(
        "private/verifier-checkpoint.json", "candidates"
    )
    origin_entries, origin_calls = checkpoint_count(
        "private/origin-retrieval-checkpoint.json", "candidates"
    )
    source_processing = load("source-processing.json")
    source_processing_summary = None
    if isinstance(source_processing, dict):
        counts = source_processing.get("counts")
        source_processing_summary = {
            "sha256": artifacts["source-processing.json"]["sha256"],
            "counts": counts if isinstance(counts, dict) else None,
        }
    return {
        "accounting_basis": (
            "regular parseable artifacts present at the typed pre-dispatch stop; entry counts "
            "describe recorded checkpoint structure, not finalized or quality-scored candidates; "
            "the provider_budget ledger remains authoritative for provider-call accounting"
        ),
        "finalized_candidate_counts_available": False,
        "selected_pages": selected_pages,
        "selected_blocks": selected_blocks,
        "layout": layout_summary,
        "row_plan": row_plan_summary,
        "source_processing": source_processing_summary,
        "checkpointed_entries": {
            "extractor_blocks": extractor_entries,
            "row_batches": row_entries,
            "tuple_candidates": tuple_entries,
            "verifier_candidates": verifier_entries,
            "origin_candidates": origin_entries,
        },
        "checkpointed_completed_call_records": (
            extractor_calls + row_calls + tuple_calls + verifier_calls + origin_calls
        ),
        "artifacts": artifacts,
    }


def _bounded_paper_summary(
    *,
    paper: PaperSpec,
    settings: PipelineSettings,
    schema_version: str,
    schema_sha256: str,
    code_state: dict[str, str | bool],
    block_config: dict[str, Any],
    error: ProviderBudgetExhausted,
    started: float,
) -> dict[str, Any]:
    """Describe an intentional budget stop without misclassifying it as a run error."""

    paper_root = settings.output_root / paper.paper_id
    progress = _bounded_checkpoint_progress(paper_root)
    call_accounting = _error_checkpoint_call_accounting(paper_root)

    def stage_calls(name: str) -> dict[str, Any]:
        return call_accounting["stages"][name]

    zero_counts = {
        key: 0
        for key in (
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
    }
    return {
        "schema_version": _PIPELINE_RUN_SCHEMA_VERSION,
        "status": "bounded_incomplete",
        "paper_id": paper.paper_id,
        "title": paper.title,
        "candidate_validation": _candidate_validation_run_configuration(settings),
        "source_processing": progress["source_processing"],
        "candidate_lineage": None,
        "selected_pages": progress["selected_pages"],
        "selected_blocks": progress["selected_blocks"],
        "result_block_segmentation": block_config,
        "layout_parser": (progress["layout"]["parser"] if progress["layout"] is not None else None),
        "layout_parser_version": (
            progress["layout"]["parser_version"] if progress["layout"] is not None else None
        ),
        "extractor": {
            **_extractor_run_configuration(settings),
            "calls": [_public_provider_call(call) for call in stage_calls("extractor")["calls"]],
            "resumed_calls": [],
            "execution": {
                "blocks_total": len(progress["selected_blocks"]),
                "blocks_succeeded": 0,
                "blocks_failed": 0,
                "blocks_resumed": 0,
                "calls_succeeded": 0,
                "calls_failed": 0,
                "calls_resumed": 0,
                "calls_resumed_succeeded": 0,
                "calls_resumed_failed": 0,
                "no_call_failures": 0,
                "requests_rejected": 0,
                "transport_failures": 0,
                "local_failures": 0,
                "blocks_checkpointed": progress["checkpointed_entries"]["extractor_blocks"],
                "calls_checkpointed": stage_calls("extractor")["telemetry"]["calls"],
            },
            "successful_call_telemetry": _provider_call_telemetry([]),
            "completed_call_telemetry": stage_calls("extractor")["telemetry"],
            "block_attempts": [],
            "checkpoint": stage_calls("extractor")["checkpoint"],
        },
        "row_enumeration": {
            **_row_enumeration_run_configuration(settings),
            "plan_sha256": (
                progress["row_plan"]["sha256"] if progress["row_plan"] is not None else None
            ),
            "plan": (
                progress["row_plan"]["telemetry"] if progress["row_plan"] is not None else None
            ),
            "preflight": None,
            "outcome": RowEnumerationOutcome().telemetry,
            "calls": [
                _public_provider_call(call) for call in stage_calls("row_enumeration")["calls"]
            ],
            "successful_call_telemetry": _provider_call_telemetry([]),
            "completed_call_telemetry": stage_calls("row_enumeration")["telemetry"],
            "attempts": [],
            "execution": {
                "batches_total": (
                    progress["row_plan"]["telemetry"]["base_batches"]
                    if progress["row_plan"] is not None
                    else 0
                ),
                "batches_resumed": 0,
                "batches_executed": 0,
                "batches_checkpointed": progress["checkpointed_entries"]["row_batches"],
                "calls_checkpointed": stage_calls("row_enumeration")["telemetry"]["calls"],
                "invalid_rows_seen": 0,
                "unknown_row_ids_seen": 0,
            },
            "checkpoint": stage_calls("row_enumeration")["checkpoint"],
        },
        "tuple_resolution": {
            **_tuple_run_configuration(settings),
            "calls": [
                _public_provider_call(call) for call in stage_calls("tuple_resolution")["calls"]
            ],
            "resumed_calls": [],
            "completed_call_telemetry": stage_calls("tuple_resolution")["telemetry"],
            "execution": {
                "candidates_selected": progress["checkpointed_entries"]["tuple_candidates"],
                "candidates_passed": 0,
                "candidates_routed_to_review": 0,
                "candidates_unsupported": 0,
                "candidates_failed": 0,
                "candidates_mismatched": 0,
                "candidates_resumed": 0,
                "candidates_checkpointed": progress["checkpointed_entries"]["tuple_candidates"],
                "calls_checkpointed": stage_calls("tuple_resolution")["telemetry"]["calls"],
            },
            "sidecar": None,
            "checkpoint": stage_calls("tuple_resolution")["checkpoint"],
        },
        "verifier": {
            **_verifier_run_configuration(settings),
            "calls": [_public_provider_call(call) for call in stage_calls("verifier")["calls"]],
            "resumed_calls": [],
            "successful_call_telemetry": _provider_call_telemetry([]),
            "completed_call_telemetry": stage_calls("verifier")["telemetry"],
            "execution": {
                "candidates_selected": progress["checkpointed_entries"]["verifier_candidates"],
                "candidates_unbound": 0,
                "candidates_verified": 0,
                "candidates_failed": 0,
                "candidates_resumed": 0,
                "candidates_resumed_succeeded": 0,
                "candidates_resumed_failed": 0,
                "candidates_executed": 0,
                "candidates_executed_succeeded": 0,
                "candidates_executed_failed": 0,
                "candidates_checkpointed": progress["checkpointed_entries"]["verifier_candidates"],
                "calls_checkpointed": stage_calls("verifier")["telemetry"]["calls"],
            },
            "checkpoint": stage_calls("verifier")["checkpoint"],
        },
        "origin_retrieval": {
            **_origin_run_configuration(settings),
            "calls": [
                _public_provider_call(call) for call in stage_calls("origin_retrieval")["calls"]
            ],
            "resumed_calls": [],
            "completed_call_telemetry": stage_calls("origin_retrieval")["telemetry"],
            "execution": {
                "candidates_selected": progress["checkpointed_entries"]["origin_candidates"],
                "candidates_resumed": 0,
                "candidates_failed": 0,
                "candidates_deterministic_external": 0,
                "candidates_checkpointed": progress["checkpointed_entries"]["origin_candidates"],
                "calls_checkpointed": stage_calls("origin_retrieval")["telemetry"]["calls"],
            },
            "checkpoint": stage_calls("origin_retrieval")["checkpoint"],
        },
        "eee_schema": {"version": schema_version, "sha256": schema_sha256},
        "code": code_state,
        "counts": zero_counts,
        "counts_status": "not_finalized_due_to_bounded_stop",
        "partial_progress": progress,
        "failure_provider_accounting": {
            "schema_version": call_accounting["schema_version"],
            "basis": call_accounting["basis"],
            "checkpointed_completed_calls": call_accounting["checkpointed_completed_calls"],
            "checkpoint_parse_complete": call_accounting["checkpoint_parse_complete"],
            "stages": {
                name: {
                    "checkpoint": stage["checkpoint"],
                    "completed_call_telemetry": stage["telemetry"],
                }
                for name, stage in call_accounting["stages"].items()
            },
        },
        "bounded_stop": {
            "code": error.code,
            "reason": error.reason,
            "provider_call_dispatched": False,
        },
        "provider_budget": error.summary,
        "review_state": {
            "status": "blocked",
            "reasons": ["provider_budget_exhausted"],
        },
        "reference_evaluation": None,
        "warnings": ["paper_review_required=provider_budget_exhausted"],
        "wall_clock_seconds": round(time.monotonic() - started, 6),
    }


def run_corpus(
    *,
    corpus: CorpusSpec,
    settings: PipelineSettings,
    client: OpenRouterClient,
) -> dict[str, Any]:
    _validate_pipeline_settings(settings)
    corpus_started = time.monotonic()
    _, schema_authority = load_schema(settings.schema_path, settings.schema_sha256)
    corpus_binding = build_corpus_binding(corpus)
    limits = ProviderBudgetLimits(
        max_structured_calls=settings.provider_max_structured_calls,
        max_cost_usd=settings.provider_max_cost_usd,
        cost_reservation_per_call_usd=settings.provider_cost_reservation_per_call_usd,
    )
    budget_contract = provider_budget_contract(
        corpus_binding=corpus_binding,
        provider_run_contract={
            "candidate_validation": _candidate_validation_run_configuration(settings),
            "extractor": _extractor_run_configuration(settings),
            "row_enumeration": _row_enumeration_run_configuration(settings),
            "tuple_resolution": _tuple_run_configuration(settings),
            "verifier": _verifier_run_configuration(settings),
            "origin_retrieval": _origin_run_configuration(settings),
        },
        limits=limits,
    )
    budget_client = BudgetedProviderClient(
        client=client,
        ledger_path=settings.output_root / "private" / "provider-budget-ledger.jsonl",
        contract=budget_contract,
        limits=limits,
    )
    failure_code_state = _code_state(settings.project_root)
    failure_block_config = asdict(
        ResultBlockConfig(max_blocks_per_page=settings.max_blocks_per_page)
    )
    _clear_corpus_run_outputs(settings.output_root)
    summaries: list[dict[str, Any]] = []
    bounded_stop: ProviderBudgetExhausted | None = None
    for paper in corpus.papers:
        started = time.monotonic()
        try:
            summary = run_paper(spec=paper, settings=settings, client=budget_client)
        except ProviderBudgetExhausted as error:
            paper_root = settings.output_root / paper.paper_id
            summary = _bounded_paper_summary(
                paper=paper,
                settings=settings,
                schema_version=schema_authority.version,
                schema_sha256=schema_authority.sha256,
                code_state=failure_code_state,
                block_config=failure_block_config,
                error=error,
                started=started,
            )
            source_manifest_path = paper_root / "source-manifest.json"
            if source_manifest_path.is_file() and not source_manifest_path.is_symlink():
                summary["source_manifest_sha256"] = sha256_file(source_manifest_path)
            write_json(paper_root / "run.json", summary)
            summaries.append(summary)
            bounded_stop = error
            break
        except ProviderBudgetError:
            # Ledger corruption, contract drift, and other budget-integrity failures are
            # corpus-fatal. They must never be rewritten as an ordinary paper failure.
            raise
        except Exception as error:
            paper_root = settings.output_root / paper.paper_id
            failure_call_accounting = _error_checkpoint_call_accounting(paper_root)
            failure_budget = budget_client.summary

            def failure_stage(
                name: str,
                accounting: dict[str, Any] = failure_call_accounting,
            ) -> dict[str, Any]:
                return accounting["stages"][name]

            _clear_paper_run_outputs(paper_root)
            source_manifest_path = paper_root / "source-manifest.json"
            failure_warnings = ["paper_review_required=paper_run_error"]
            reference_score: dict[str, Any] | None = None
            reference_score_sha256: str | None = None
            if source_manifest_path.is_file() and not source_manifest_path.is_symlink():
                try:
                    reference_score, reference_score_sha256 = _score_reference_after_paper_error(
                        spec=paper,
                        settings=settings,
                        manifest_path=source_manifest_path,
                        output_path=paper_root / "reference-score.json",
                    )
                except (OSError, TypeError, ValueError, StopIteration):
                    failure_warnings.append("reference_score_unavailable_after_paper_error")
            summary = {
                "schema_version": _PIPELINE_RUN_SCHEMA_VERSION,
                "status": "error",
                "paper_id": paper.paper_id,
                "title": paper.title,
                "candidate_validation": _candidate_validation_run_configuration(settings),
                "selected_pages": [],
                "selected_blocks": [],
                "result_block_segmentation": failure_block_config,
                "layout_parser": None,
                "layout_parser_version": None,
                "extractor": {
                    **_extractor_run_configuration(settings),
                    "calls": [
                        _public_provider_call(call) for call in failure_stage("extractor")["calls"]
                    ],
                    "resumed_calls": [],
                    "execution": {
                        "blocks_total": 0,
                        "blocks_succeeded": 0,
                        "blocks_failed": 0,
                        "blocks_resumed": 0,
                        "calls_succeeded": 0,
                        "calls_failed": 0,
                        "calls_resumed": 0,
                        "calls_resumed_succeeded": 0,
                        "calls_resumed_failed": 0,
                        "no_call_failures": 0,
                        "requests_rejected": 0,
                        "transport_failures": 0,
                        "local_failures": 0,
                        "calls_checkpointed": failure_stage("extractor")["telemetry"]["calls"],
                    },
                    "successful_call_telemetry": _provider_call_telemetry([]),
                    "completed_call_telemetry": failure_stage("extractor")["telemetry"],
                    "block_attempts": [],
                    "checkpoint": failure_stage("extractor")["checkpoint"],
                },
                "row_enumeration": {
                    **_row_enumeration_run_configuration(settings),
                    "plan_sha256": None,
                    "plan": None,
                    "preflight": None,
                    "outcome": RowEnumerationOutcome().telemetry,
                    "calls": [
                        _public_provider_call(call)
                        for call in failure_stage("row_enumeration")["calls"]
                    ],
                    "successful_call_telemetry": _provider_call_telemetry(
                        [],
                        basis=(
                            "completed bounded row-enumeration calls; cost, token, retry, "
                            "and attempt totals are lower bounds"
                        ),
                    ),
                    "completed_call_telemetry": failure_stage("row_enumeration")["telemetry"],
                    "attempts": [],
                    "execution": {
                        "batches_total": 0,
                        "batches_resumed": 0,
                        "batches_executed": 0,
                        "calls_checkpointed": failure_stage("row_enumeration")["telemetry"][
                            "calls"
                        ],
                        "invalid_rows_seen": 0,
                        "unknown_row_ids_seen": 0,
                    },
                    "checkpoint": failure_stage("row_enumeration")["checkpoint"],
                },
                "tuple_resolution": {
                    **_tuple_run_configuration(settings),
                    "calls": [
                        _public_provider_call(call)
                        for call in failure_stage("tuple_resolution")["calls"]
                    ],
                    "resumed_calls": [],
                    "completed_call_telemetry": failure_stage("tuple_resolution")["telemetry"],
                    "execution": {
                        "candidates_selected": 0,
                        "candidates_passed": 0,
                        "candidates_routed_to_review": 0,
                        "candidates_unsupported": 0,
                        "candidates_failed": 0,
                        "candidates_mismatched": 0,
                        "candidates_resumed": 0,
                        "calls_checkpointed": failure_stage("tuple_resolution")["telemetry"][
                            "calls"
                        ],
                    },
                    "sidecar": None,
                    "checkpoint": failure_stage("tuple_resolution")["checkpoint"],
                },
                "verifier": {
                    **_verifier_run_configuration(settings),
                    "calls": [
                        _public_provider_call(call) for call in failure_stage("verifier")["calls"]
                    ],
                    "resumed_calls": [],
                    "successful_call_telemetry": _provider_call_telemetry([]),
                    "completed_call_telemetry": failure_stage("verifier")["telemetry"],
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
                        "calls_checkpointed": failure_stage("verifier")["telemetry"]["calls"],
                    },
                    "checkpoint": failure_stage("verifier")["checkpoint"],
                },
                "origin_retrieval": {
                    **_origin_run_configuration(settings),
                    "calls": [
                        _public_provider_call(call)
                        for call in failure_stage("origin_retrieval")["calls"]
                    ],
                    "resumed_calls": [],
                    "completed_call_telemetry": failure_stage("origin_retrieval")["telemetry"],
                    "execution": {
                        "candidates_selected": 0,
                        "candidates_resumed": 0,
                        "candidates_failed": 0,
                        "candidates_deterministic_external": 0,
                        "calls_checkpointed": failure_stage("origin_retrieval")["telemetry"][
                            "calls"
                        ],
                    },
                    "checkpoint": failure_stage("origin_retrieval")["checkpoint"],
                },
                "eee_schema": {
                    "version": schema_authority.version,
                    "sha256": schema_authority.sha256,
                },
                "code": failure_code_state,
                "counts": {
                    "candidates": 0,
                    "candidates_before_deduplication": 0,
                    "duplicates_removed": 0,
                    "candidates_needing_review": 0,
                    "semantic_safety_reviews": 0,
                    "primary_results": 0,
                    "exported": 0,
                    "eee_records": 0,
                    "eee_schema_issues": 0,
                    "tuple_candidates": 0,
                    "tuple_passed": 0,
                    "tuple_review": 0,
                    "tuple_unsupported": 0,
                    "tuple_failed": 0,
                    "tuple_resumed": 0,
                    "verifications": 0,
                    "verifier_accepts": 0,
                    "verifier_rejects": 0,
                    "verifier_reviews": 0,
                    "verifier_failed": 0,
                    "verifier_resumed": 0,
                    "origin_candidates": 0,
                    "origin_resumed": 0,
                    "origin_failed": 0,
                    "origin_positive_review_only": 0,
                    "origin_external": 0,
                    "origin_unresolved": 0,
                    "origin_no_signal": 0,
                    "spot_checks": 0,
                    "spot_checks_exact": 0,
                    "reference_observations": (
                        reference_score["reference_observations"] if reference_score else 0
                    ),
                    "reference_true_positives": (
                        reference_score["detection"]["true_positives"] if reference_score else 0
                    ),
                    "reference_false_positives": (
                        reference_score["detection"]["false_positives"] if reference_score else 0
                    ),
                    "reference_false_negatives": (
                        reference_score["detection"]["false_negatives"] if reference_score else 0
                    ),
                    "negative_control_false_primary": (
                        reference_score.get("negative_control_safety", {}).get(
                            "false_primary_count", 0
                        )
                        if reference_score
                        else 0
                    ),
                },
                "error": {
                    "type": type(error).__name__,
                    "message": _safe_error_message(error),
                },
                "failure_provider_accounting": {
                    "schema_version": failure_call_accounting["schema_version"],
                    "basis": failure_call_accounting["basis"],
                    "checkpointed_completed_calls": failure_call_accounting[
                        "checkpointed_completed_calls"
                    ],
                    "checkpoint_parse_complete": failure_call_accounting[
                        "checkpoint_parse_complete"
                    ],
                    "stages": {
                        name: {
                            "checkpoint": stage["checkpoint"],
                            "completed_call_telemetry": stage["telemetry"],
                        }
                        for name, stage in failure_call_accounting["stages"].items()
                    },
                },
                "provider_budget": failure_budget,
                "wall_clock_seconds": round(time.monotonic() - started, 6),
                "review_state": {
                    "status": "blocked",
                    "reasons": ["paper_run_error"],
                },
                "reference_evaluation": (
                    {
                        "path": paper.reference_path,
                        "score_path": "reference-score.json",
                        "score_sha256": reference_score_sha256,
                        "schema_version": reference_score["schema_version"],
                        "coverage": reference_score["coverage"],
                        "detection": reference_score["detection"],
                        "field_accuracy": reference_score["field_accuracy"],
                        "negative_control_safety": reference_score.get("negative_control_safety"),
                    }
                    if reference_score
                    else None
                ),
                "warnings": failure_warnings,
            }
            if source_manifest_path.is_file() and not source_manifest_path.is_symlink():
                summary["source_manifest_sha256"] = sha256_file(source_manifest_path)
            write_jsonl(paper_root / "observations.jsonl", [])
            write_jsonl(paper_root / "verifications.jsonl", [])
            write_json(paper_root / "spot-checks.json", [])
            write_json(paper_root / "run.json", summary)
        summaries.append(summary)
    succeeded = [summary for summary in summaries if summary.get("status") == "success"]
    failed = [summary for summary in summaries if summary.get("status") != "success"]
    reference_scores: list[dict[str, Any]] = []
    for summary in summaries:
        reference_evaluation = summary.get("reference_evaluation")
        if not isinstance(reference_evaluation, dict):
            continue
        expected_sha256 = reference_evaluation.get("score_sha256")
        if not isinstance(expected_sha256, str):
            continue
        score_path = settings.output_root / summary["paper_id"] / "reference-score.json"
        try:
            if sha256_file(score_path) != expected_sha256:
                continue
            score = read_json(score_path)
        except (OSError, ValueError):
            continue
        if isinstance(score, dict):
            reference_scores.append(score)
    corpus_evaluation = aggregate_reference_scores(reference_scores) if reference_scores else None
    if corpus_evaluation:
        write_json(settings.output_root / "corpus-evaluation.json", corpus_evaluation)
    report = {
        "schema_version": _CORPUS_RUN_SCHEMA_VERSION,
        "corpus_id": corpus.corpus_id,
        "corpus_binding": corpus_binding,
        "status": (
            "bounded_incomplete"
            if bounded_stop is not None
            else "success"
            if not failed
            else "error"
            if not succeeded
            else "partial_failure"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "papers": len(summaries),
        "papers_total": len(corpus.papers),
        "papers_not_started": len(corpus.papers) - len(summaries),
        "papers_succeeded": len(succeeded),
        "papers_failed": len(failed),
        "papers_bounded_incomplete": sum(
            summary.get("status") == "bounded_incomplete" for summary in summaries
        ),
        "papers_with_eee": sum(summary["counts"]["eee_records"] > 0 for summary in succeeded),
        "papers_without_candidates": sum(
            summary["counts"].get("candidates", 0) == 0 for summary in succeeded
        ),
        "papers_without_eee": sum(
            summary["counts"].get("eee_records", 0) == 0 for summary in succeeded
        ),
        "papers_needing_review": sum(
            summary.get("review_state", {}).get("status") != "ready" for summary in summaries
        ),
        "reference_evaluation": corpus_evaluation,
        "totals": {
            key: sum(summary["counts"].get(key, 0) for summary in summaries)
            for key in (
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
        },
        "runs": summaries,
        "provider_budget": {
            **budget_client.summary,
            "bounded_stop": (
                {
                    "code": bounded_stop.code,
                    "reason": bounded_stop.reason,
                    "provider_call_dispatched": False,
                }
                if bounded_stop is not None
                else None
            ),
        },
    }
    report["operations"] = _corpus_operational_metrics(
        summaries,
        wall_clock_seconds=time.monotonic() - corpus_started,
    )
    write_json(settings.output_root / "corpus-run.json", report)
    render_corpus_html_file(
        settings.output_root / "corpus-run.json",
        settings.output_root / "corpus-review.html",
    )
    return report


def freeze_corpus(
    corpus: CorpusSpec,
    settings: PipelineSettings,
    *,
    workers: int = 1,
    requests_per_second: float = 2.0,
    progress_every: int = 0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Acquire and pin a corpus without contacting a model provider.

    Papers already pinned under this output root are reused rather than re-downloaded,
    so an interrupted freeze resumes for free. Downloads run on `workers` threads paced
    by one shared limiter, and a paper that cannot be frozen is recorded with a typed
    reason instead of ending the corpus.
    """

    if workers < 1:
        raise ValueError("workers must be at least 1")

    limiter = HostRateLimiter(requests_per_second)
    lock = threading.Lock()
    manifests: list[SourceManifest] = []
    results: list[dict[str, Any]] = []
    counters = {"frozen": 0, "reused": 0, "failed": 0}

    def freeze_one(paper: PaperSpec) -> None:
        already_pinned = _manifest_path(settings, paper.paper_id).exists()
        try:
            manifest = freeze_paper(paper, settings, before_request=limiter.acquire)
        except Exception as error:
            with lock:
                counters["failed"] += 1
                results.append(
                    {
                        "paper_id": paper.paper_id,
                        "status": "error",
                        "error": {
                            "type": type(error).__name__,
                            "message": _safe_error_message(error),
                        },
                    }
                )
            return
        with lock:
            counters["reused" if already_pinned else "frozen"] += 1
            manifests.append(manifest)
            results.append(
                {
                    "paper_id": manifest.paper_id,
                    # A reused manifest is still a successful freeze. Downstream code
                    # gates on this exact value (preflight refuses to plan a paper whose
                    # freeze did not report "success"), so reuse is reported beside the
                    # status rather than by replacing it.
                    "status": "success",
                    "reused": already_pinned,
                    "source_ids": [source.source_id for source in manifest.sources],
                    "sha256": [source.sha256 for source in manifest.sources],
                }
            )
            done = counters["frozen"] + counters["reused"] + counters["failed"]
        if progress is not None and progress_every > 0 and done % progress_every == 0:
            progress(
                f"{done}/{len(corpus.papers)} frozen={counters['frozen']} "
                f"reused={counters['reused']} failed={counters['failed']}"
            )

    if workers == 1:
        for paper in corpus.papers:
            freeze_one(paper)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(freeze_one, corpus.papers):
                pass

    order = {paper.paper_id: index for index, paper in enumerate(corpus.papers)}
    results.sort(key=lambda item: order[item["paper_id"]])
    manifests.sort(key=lambda item: order[item.paper_id])

    failures = counters["failed"]
    status = "success" if failures == 0 else "error" if not manifests else "partial_failure"
    summary = {
        "schema_version": "corpus-freeze/0.2",
        "corpus_id": corpus.corpus_id,
        "status": status,
        "papers": len(corpus.papers),
        "papers_succeeded": len(manifests),
        "papers_failed": failures,
        "papers_downloaded": counters["frozen"],
        "papers_reused": counters["reused"],
        "workers": workers,
        "requests_per_second": requests_per_second,
        "sources": sum(len(manifest.sources) for manifest in manifests),
        "failure_reasons": dict(
            sorted(
                Counter(
                    item["error"]["type"] for item in results if item["status"] == "error"
                ).items()
            )
        ),
        "manifests": [
            {
                "paper_id": manifest.paper_id,
                "source_ids": [source.source_id for source in manifest.sources],
                "sha256": [source.sha256 for source in manifest.sources],
            }
            for manifest in manifests
        ],
        "results": results,
    }
    write_json(settings.output_root / "corpus-freeze.json", summary)
    return summary


def runtime_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY must be supplied in the runtime environment")
    return key
