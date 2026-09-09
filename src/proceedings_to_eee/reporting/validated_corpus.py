"""Provider-free verification receipts for sealed current pipeline runs.

This module is deliberately not a second implementation of extraction.  It
rehydrates the production checkpoints with the same validators used by
``run_paper`` and only retains quote-free aggregate facts for reporting.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from proceedings_to_eee.corpus import PaperSpec
from proceedings_to_eee.domain.attribution import AttributionState
from proceedings_to_eee.domain.export_provenance import (
    ExportProvenanceMode,
    tuple_gated_export_provenance,
    tuple_unverified_review_provenance,
)
from proceedings_to_eee.domain.lineage import build_candidate_lineage
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ClaimType, ExportStatus
from proceedings_to_eee.extraction.llm import RowEnumerationOutcome
from proceedings_to_eee.extraction.result_blocks import (
    ResultBlock,
    ResultBlockConfig,
    segment_page_result_blocks,
)
from proceedings_to_eee.extraction.row_enumeration import (
    RowEnumerationConfig,
    RowEnumerationPlan,
    RowTerminalLedger,
    build_row_enumeration_plan,
)
from proceedings_to_eee.extraction.row_validation import (
    partition_row_provider_calls,
    validate_outcome_against,
)
from proceedings_to_eee.io import canonical_json_bytes, read_json, sha256_bytes, sha256_file
from proceedings_to_eee.pipeline import (
    PipelineSettings,
    _apply_origin_assessment,
    _candidate_validation_run_configuration,
    _corpus_operational_metrics,
    _extractor_checkpoint_contract,
    _extractor_run_configuration,
    _merge_row_outcome,
    _origin_checkpoint_contract,
    _origin_run_configuration,
    _provider_call_telemetry,
    _row_checkpoint_contract,
    _row_enumeration_run_configuration,
    _tuple_checkpoint_contract,
    _tuple_gate_sha256,
    _tuple_run_configuration,
    _validate_pipeline_settings,
    _validated_checkpoint_entry,
    _validated_legacy_recovery_entry,
    _validated_origin_checkpoint_entry,
    _validated_row_checkpoint_entry,
    _validated_tuple_checkpoint_entry,
    _validated_verifier_checkpoint_entry,
    _verifier_candidate_sha256,
    _verifier_checkpoint_contract,
    _verifier_gate_sha256,
    _verifier_run_configuration,
)
from proceedings_to_eee.providers.openrouter import ProviderCall, public_provider_call
from proceedings_to_eee.resolution.origin_retrieval import candidate_origin_binding_sha256
from proceedings_to_eee.resolution.tuple_resolution import (
    TupleResolutionDecision,
    tuple_assessment_is_export_concordant,
    tuple_candidate_binding_sha256,
)
from proceedings_to_eee.resources import (
    DEFAULT_EEE_SCHEMA_PATH,
    EEE_SCHEMA_SHA256,
    EEE_SCHEMA_VERSION,
)
from proceedings_to_eee.reviewed_export.workflow import ReviewedExportError, _load_run_papers
from proceedings_to_eee.run_seal import (
    RunSealVerificationError,
    VerifiedRunSeal,
    verify_run_seal,
)
from proceedings_to_eee.sources.processing import (
    SourceProcessingArtifact,
    validate_source_processing_artifact,
)
from proceedings_to_eee.validation.candidates import (
    deduplicate_candidates_with_lineage,
    route_candidate_attribution,
    validate_non_origin_candidates,
)
from proceedings_to_eee.verification.binding import bind_candidate_block, frozen_evidence_block
from proceedings_to_eee.verification.independent import IndependentDecision

StageStatus = Literal["validated", "partial_failure", "not_run"]
_STAGES = (
    "extractor",
    "row_disposition",
    "tuple_resolution",
    "independent_verification",
    "origin_retrieval",
)
_PROVENANCE_MODES = (
    *(item.value for item in ExportProvenanceMode),
    "unclassified_legacy_source",
)
_CODE_KEYS = {"git_commit", "git_dirty", "git_available", "source_tree_sha256"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]*(?:/[A-Za-z0-9][A-Za-z0-9._:+-]*){1,3}$")
_REASONING_EFFORTS = {None, "none", "minimal", "low", "medium", "high", "xhigh"}
_COUNT_KEYS = {
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
_PUBLIC_COUNT_KEYS = _COUNT_KEYS - {
    "tuple_resumed",
    "verifier_resumed",
    "origin_resumed",
    "spot_checks",
    "spot_checks_exact",
    "reference_observations",
    "reference_true_positives",
    "reference_false_positives",
    "reference_false_negatives",
    "negative_control_false_primary",
}
_REVIEW_REASONS = {
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


class CompletedCorpusValidationError(ValueError):
    """A sealed current run cannot produce a trustworthy reporting receipt."""


@dataclass(frozen=True, slots=True)
class ValidatedStageReceipt:
    """Quote-free terminal accounting for one production stage."""

    status: StageStatus
    completed_calls: int
    selected_items: int
    failed_items: int
    checkpoint_contract_sha256: str | None
    sidecar_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ValidatedPaperReceipt:
    """Private-free facts reconstructed from one exact paper run."""

    paper_id: str
    status: str
    needs_review: bool
    review_reasons: tuple[str, ...]
    counts: Mapping[str, int]
    stages: Mapping[str, ValidatedStageReceipt]
    export_status_counts: Mapping[str, int]
    text_support_counts: Mapping[str, int]
    referential_status_counts: Mapping[str, int]
    attribution_state_counts: Mapping[str, int]
    row_metrics: Mapping[str, Any]
    lineage_counts: Mapping[str, int]
    tuple_sidecar_sha256: str
    verifier_sidecar_sha256: str | None


@dataclass(frozen=True, slots=True)
class ValidatedCorpusReceipt:
    """Allowlisted reporting state for one verified sealed current corpus run."""

    seal: VerifiedRunSeal
    corpus_id: str
    corpus_run_sha256: str
    corpus_spec_sha256: str
    paper_ids_sha256: str
    generated_at: str
    code: Mapping[str, Any]
    eee_schema: Mapping[str, str]
    extractor_binding: Mapping[str, Any]
    row_binding: Mapping[str, Any]
    candidate_validation: Mapping[str, Any]
    corpus_status: str
    papers: tuple[ValidatedPaperReceipt, ...]
    totals: Mapping[str, int]
    operations: Mapping[str, Any]
    provider_telemetry: Mapping[str, Any]


def _fail(message: str) -> None:
    raise CompletedCorpusValidationError(message)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{context} is invalid")
    return value


def _sequence(value: object, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{context} is invalid")
    return value


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{context} is invalid")
    return value


def _model_id(value: object, context: str) -> str:
    if not isinstance(value, str) or len(value) > 255 or _MODEL_ID.fullmatch(value) is None:
        _fail(f"{context} is invalid")
    return value


def _configuration_temperature(value: object) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 2
    ):
        _fail("temperature binding is invalid")
    return float(value)


def _regular_json(path: Path, context: str) -> Any:
    if path.is_symlink() or not path.is_file():
        _fail(f"{context} is missing")
    try:
        return read_json(path)
    except (OSError, ValueError) as error:
        raise CompletedCorpusValidationError(f"{context} is invalid") from error


def _regular_jsonl(path: Path, context: str) -> list[Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{context} is missing")
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError) as error:
        raise CompletedCorpusValidationError(f"{context} is invalid") from error


def _public_calls(value: object, context: str) -> list[Mapping[str, Any]]:
    items = _sequence(value, context)
    if any(not isinstance(item, Mapping) for item in items):
        _fail(f"{context} is invalid")
    calls = [item for item in items if isinstance(item, Mapping)]
    if any("request_id" in item or "request_id_sha256" in item for item in calls):
        _fail(f"{context} exposes a private provider request identifier")
    return calls


def _call_counter(calls: Sequence[ProviderCall]) -> Counter[bytes]:
    """Project private calls into the exact public representation for comparison.

    Provider request IDs are transport identifiers, not request or response bindings.  The
    centralized projection preserves request-ID presence only and bounds response metadata.
    """

    return Counter(canonical_json_bytes(public_provider_call(call)) for call in calls)


def _recorded_call_counter(value: object, context: str) -> Counter[bytes]:
    return Counter(canonical_json_bytes(dict(call)) for call in _public_calls(value, context))


def _telemetry_matches(recorded: object, calls: list[ProviderCall]) -> None:
    recorded_mapping = _mapping(recorded, "stage provider telemetry")
    expected = _provider_call_telemetry(calls, basis=str(recorded_mapping.get("basis", "")))
    if recorded_mapping != expected:
        _fail("stage provider telemetry does not match validated calls")


def _configuration_binding(configuration: Mapping[str, Any]) -> dict[str, Any]:
    request_contract = _mapping(configuration.get("request_contract"), "request contract")
    provider = configuration.get("provider", "openrouter")
    if provider != "openrouter":
        _fail("provider binding is invalid")
    model = configuration.get("model")
    if model is not None:
        model = _model_id(model, "model binding")
    reasoning_effort = configuration.get("reasoning_effort")
    if reasoning_effort not in _REASONING_EFFORTS:
        _fail("reasoning-effort binding is invalid")
    max_tokens = configuration.get("max_tokens")
    seed = configuration.get("seed")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        _fail("max-token binding is invalid")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        _fail("seed binding is invalid")
    prompt_sha256 = configuration.get("prompt_sha256")
    if not isinstance(prompt_sha256, str) or _SHA256.fullmatch(prompt_sha256) is None:
        _fail("prompt binding is invalid")
    return {
        "provider": provider,
        "model": model,
        "temperature": _configuration_temperature(configuration.get("temperature")),
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "seed": seed,
        "prompt_sha256": prompt_sha256,
        "request_contract_sha256": sha256_bytes(canonical_json_bytes(request_contract)),
    }


def _paper_settings(run: Mapping[str, Any], paper_root: Path) -> PipelineSettings:
    extractor = _mapping(run.get("extractor"), "extractor configuration")
    row = _mapping(run.get("row_enumeration"), "row configuration")
    tuple_stage = _mapping(run.get("tuple_resolution"), "tuple configuration")
    verifier = _mapping(run.get("verifier"), "verifier configuration")
    origin = _mapping(run.get("origin_retrieval"), "origin configuration")
    schema = _mapping(run.get("eee_schema"), "EEE schema binding")
    candidate_validation = _mapping(
        run.get("candidate_validation"), "candidate validation configuration"
    )
    if candidate_validation.get("schema_version") not in {
        "candidate-validation/0.1",
        "candidate-validation/0.2",
    }:
        _fail("candidate validation configuration is unsupported")
    min_confidence = candidate_validation.get("min_confidence")
    if (
        isinstance(min_confidence, bool)
        or not isinstance(min_confidence, int | float)
        or not 0 <= float(min_confidence) <= 1
    ):
        _fail("candidate validation min_confidence is invalid")
    try:
        extractor_model = _model_id(extractor.get("model"), "extractor model")
        row_model = _model_id(row.get("model"), "row model") if row.get("enabled") is True else None
        tuple_model = (
            _model_id(tuple_stage.get("model"), "tuple model")
            if tuple_stage.get("enabled") is True
            else None
        )
        verifier_model = (
            _model_id(verifier.get("model"), "verifier model")
            if verifier.get("enabled") is True
            else None
        )
        origin_model = (
            _model_id(origin.get("model"), "origin model")
            if origin.get("enabled") is True
            else None
        )
        settings = PipelineSettings(
            project_root=paper_root,
            schema_path=DEFAULT_EEE_SCHEMA_PATH,
            schema_sha256=str(schema.get("sha256")),
            output_root=paper_root,
            model=extractor_model,
            min_confidence=float(min_confidence),
            max_tokens=int(extractor.get("max_tokens")),
            temperature=extractor.get("temperature"),
            reasoning_effort=extractor.get("reasoning_effort"),
            seed=(int(extractor["seed"]) if extractor.get("seed") is not None else None),
            max_blocks_per_page=int(
                _mapping(run.get("result_block_segmentation"), "block segmentation").get(
                    "max_blocks_per_page"
                )
            ),
            row_enumeration_enabled=row.get("enabled") is True,
            row_model=row_model,
            row_enumeration_config=RowEnumerationConfig.model_validate(row.get("limits")),
            tuple_model=tuple_model,
            tuple_max_tokens=int(tuple_stage.get("max_tokens")),
            verifier_model=verifier_model,
            verifier_max_tokens=int(verifier.get("max_tokens")),
            origin_model=origin_model,
            origin_max_tokens=int(origin.get("max_tokens")),
        )
        _validate_pipeline_settings(settings)
    except (TypeError, ValueError) as error:
        raise CompletedCorpusValidationError("recorded pipeline settings are invalid") from error
    expected_configurations = (
        (extractor, _extractor_run_configuration(settings)),
        (row, _row_enumeration_run_configuration(settings)),
        (tuple_stage, _tuple_run_configuration(settings)),
        (verifier, _verifier_run_configuration(settings)),
        (origin, _origin_run_configuration(settings)),
    )
    for recorded, expected in expected_configurations:
        if any(recorded.get(key) != value for key, value in expected.items()):
            _fail("recorded stage configuration differs from the production contract")
    if candidate_validation != _candidate_validation_run_configuration(settings):
        _fail("recorded candidate validation differs from the production contract")
    return settings


def _paper_spec(run: Mapping[str, Any]) -> PaperSpec:
    try:
        return PaperSpec(
            paper_id=str(run.get("paper_id")),
            title=str(run.get("title")),
            year=0,
            venue="sealed-run-replay",
            pdf_path="sealed-run-replay.pdf",
            perspective_role="sealed-run-replay",
        )
    except ValueError as error:
        raise CompletedCorpusValidationError("paper identity is invalid") from error


def _checkpoint(
    paper_root: Path,
    binding_value: object,
    *,
    expected_path: str,
    expected_schema: str,
    expected_contract: Mapping[str, Any],
    collection_key: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    binding = _mapping(binding_value, "checkpoint binding")
    expected_digest = sha256_bytes(canonical_json_bytes(expected_contract))
    if (
        binding.get("path") != expected_path
        or binding.get("schema_version") != expected_schema
        or binding.get("contract_sha256") != expected_digest
    ):
        _fail("checkpoint binding differs from its production contract")
    payload = _mapping(_regular_json(paper_root / expected_path, "checkpoint"), "checkpoint")
    collection = _mapping(payload.get(collection_key), "checkpoint entries")
    if (
        payload.get("schema_version") != expected_schema
        or payload.get("contract") != expected_contract
        or payload.get("contract_sha256") != expected_digest
    ):
        _fail("checkpoint contract is stale or tampered")
    return payload, collection, expected_digest


def _replayed_blocks(paper: Any, run: Mapping[str, Any]) -> list[ResultBlock]:
    try:
        persisted = [
            ResultBlock.model_validate(item)
            for item in _sequence(
                _regular_json(paper.result_blocks_path, "result blocks"), "result blocks"
            )
        ]
        config = ResultBlockConfig(
            **dict(_mapping(run.get("result_block_segmentation"), "block segmentation"))
        )
    except (TypeError, ValueError) as error:
        raise CompletedCorpusValidationError("result block artifact is invalid") from error
    selected_pages = list(_sequence(run.get("selected_pages"), "selected pages"))
    if any(isinstance(page, bool) or not isinstance(page, int) for page in selected_pages):
        _fail("selected pages are invalid")
    pages_by_number = {page.page: page for page in paper.layout.pages}
    if len(selected_pages) != len(set(selected_pages)) or set(selected_pages) - set(
        pages_by_number
    ):
        _fail("selected pages do not belong to the frozen layout")
    replayed = [
        block
        for page_number in selected_pages
        for block in segment_page_result_blocks(pages_by_number[page_number], config=config)
    ]
    if [item.model_dump(mode="json") for item in persisted] != [
        item.model_dump(mode="json") for item in replayed
    ]:
        _fail("result blocks do not match deterministic segmentation")
    selected_summary = [
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
        for block in replayed
    ]
    if run.get("selected_blocks") != selected_summary:
        _fail("run block inventory differs from deterministic segmentation")
    return replayed


def _source_processing(paper: Any, run: Mapping[str, Any]) -> SourceProcessingArtifact:
    binding = _mapping(run.get("source_processing"), "source-processing binding")
    path = paper.run_path.parent / "source-processing.json"
    if binding.get("path") != "source-processing.json" or binding.get(
        "artifact_sha256"
    ) != sha256_file(path):
        _fail("source-processing binding is invalid")
    try:
        artifact = SourceProcessingArtifact.model_validate(
            _regular_json(path, "source-processing artifact")
        )
        validate_source_processing_artifact(
            manifest=paper.source_manifest,
            artifact=artifact,
            layout=paper.layout,
        )
    except ValueError as error:
        raise CompletedCorpusValidationError("source-processing artifact is invalid") from error
    if binding.get("counts") != artifact.counts.model_dump(mode="json"):
        _fail("source-processing counts are invalid")
    return artifact


def _replay_paper(paper: Any) -> tuple[ValidatedPaperReceipt, tuple[ProviderCall, ...]]:
    run = _mapping(paper.run_manifest, "paper run")
    if run.get("schema_version") != "pipeline-run/0.4":
        _fail("current receipt requires pipeline-run/0.4")
    paper_root = paper.run_path.parent
    settings = _paper_settings(run, paper_root)
    spec = _paper_spec(run)
    code_state = dict(_mapping(run.get("code"), "code binding"))
    if (
        set(code_state) != _CODE_KEYS
        or not isinstance(code_state.get("git_commit"), str)
        or not isinstance(code_state.get("git_dirty"), bool)
        or not isinstance(code_state.get("git_available"), bool)
        or not isinstance(code_state.get("source_tree_sha256"), str)
        or _SHA256.fullmatch(str(code_state["source_tree_sha256"])) is None
        or (
            code_state["git_commit"] != "uncommitted"
            and _GIT_COMMIT.fullmatch(str(code_state["git_commit"])) is None
        )
        or (code_state["git_available"] is True and code_state["git_commit"] == "uncommitted")
    ):
        _fail("code binding is invalid")
    source_sha256 = sha256_bytes(canonical_json_bytes(paper.source_manifest))
    if run.get("source_manifest_sha256") != source_sha256:
        _fail("paper run does not bind its source manifest")
    processing = _source_processing(paper, run)
    blocks = _replayed_blocks(paper, run)
    block_config = ResultBlockConfig(
        **dict(_mapping(run.get("result_block_segmentation"), "block segmentation"))
    )

    extractor_stage = _mapping(run.get("extractor"), "extractor stage")
    extractor_contract = _extractor_checkpoint_contract(
        spec=spec,
        settings=settings,
        manifest=paper.source_manifest,
        layout=paper.layout,
        block_config=block_config,
        blocks=blocks,
        code_state=code_state,
    )
    extractor_checkpoint, extractor_entries, extractor_contract_sha256 = _checkpoint(
        paper_root,
        extractor_stage.get("checkpoint"),
        expected_path="private/extractor-checkpoint.json",
        expected_schema="extractor-block-checkpoint/0.3",
        expected_contract=extractor_contract,
        collection_key="blocks",
    )
    recoveries = _mapping(extractor_checkpoint.get("recoveries"), "extractor recoveries")
    if set(extractor_entries) - {block.block_id for block in blocks} or set(recoveries) - {
        block.block_id for block in blocks
    }:
        _fail("extractor checkpoint contains stale blocks")
    candidates: list[CandidateObservation] = []
    extractor_completed_calls: list[ProviderCall] = []
    extractor_successful_calls: list[ProviderCall] = []
    extractor_failures = 0
    for block in blocks:
        validated = _validated_checkpoint_entry(
            extractor_entries.get(block.block_id),
            block=block,
            spec=spec,
            settings=settings,
            contract=extractor_contract,
        )
        if validated is not None:
            block_candidates, block_calls, block_successful_calls, _ = validated
            candidates.extend(block_candidates)
            extractor_completed_calls.extend(block_calls)
            extractor_successful_calls.extend(block_successful_calls)
            if block.block_id in recoveries:
                _fail("complete extractor block retains stale recovery state")
            continue
        recovery = _validated_legacy_recovery_entry(
            recoveries.get(block.block_id),
            block=block,
            spec=spec,
            settings=settings,
        )
        if recovery is None or recovery[1] is not True:
            _fail("extractor checkpoint lacks a valid terminal block outcome")
        extractor_failures += 1
        for attempt in recovery[0]:
            call = attempt["call"]
            if isinstance(call, ProviderCall):
                extractor_completed_calls.append(call)
            if attempt["status"] == "success":
                candidates.extend(attempt["candidates"])
                if isinstance(call, ProviderCall):
                    extractor_successful_calls.append(call)
    extractor_execution = _mapping(extractor_stage.get("execution"), "extractor execution")
    blocks_total = _integer(extractor_execution.get("blocks_total"), "extractor blocks")
    blocks_succeeded = _integer(extractor_execution.get("blocks_succeeded"), "extractor successes")
    blocks_failed = _integer(extractor_execution.get("blocks_failed"), "extractor failures")
    blocks_resumed = _integer(extractor_execution.get("blocks_resumed"), "extractor resumes")
    if (
        blocks_total != len(blocks)
        or blocks_failed != extractor_failures
        or blocks_succeeded + blocks_resumed + blocks_failed != blocks_total
    ):
        _fail("extractor execution telemetry differs from its checkpoint")
    extractor_run_calls = [
        *_public_calls(extractor_stage.get("calls"), "extractor calls"),
        *_public_calls(extractor_stage.get("resumed_calls"), "extractor resumed calls"),
    ]
    if Counter(canonical_json_bytes(dict(call)) for call in extractor_run_calls) != _call_counter(
        extractor_completed_calls
    ):
        _fail("extractor call ledger differs from terminal checkpoints")
    _telemetry_matches(extractor_stage.get("successful_call_telemetry"), extractor_successful_calls)
    _telemetry_matches(extractor_stage.get("completed_call_telemetry"), extractor_completed_calls)

    row_stage = _mapping(run.get("row_enumeration"), "row stage")
    row_calls: list[ProviderCall] = []
    row_successful_calls: list[ProviderCall] = []
    row_contract_sha256: str | None = None
    row_metrics: dict[str, Any] = {
        "tables_considered": 0,
        "dense_tables": 0,
        "rows_planned": 0,
        "rows_resolved": 0,
        "rows_unresolved": 0,
        "rows_unbatchable": 0,
        "unknown_row_ids_seen": 0,
        "invalid_rows_seen": 0,
        "dispositions": {"not_result": 0, "result": 0, "uncertain": 0},
    }
    row_failures = 0
    if settings.row_enumeration_enabled:
        plan_path = paper_root / "private" / "row-enumeration-plan.json"
        try:
            plan = RowEnumerationPlan.model_validate(_regular_json(plan_path, "row plan"))
        except ValueError as error:
            raise CompletedCorpusValidationError("row plan is invalid") from error
        expected_plan = build_row_enumeration_plan(
            paper.layout,
            blocks,
            config=settings.row_enumeration_config,
        )
        if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
            _fail("row plan differs from deterministic production planning")
        plan_sha256 = sha256_file(plan_path)
        if row_stage.get("plan_sha256") != plan_sha256:
            _fail("row stage does not bind its deterministic plan")
        row_contract = _row_checkpoint_contract(
            spec=spec,
            settings=settings,
            manifest=paper.source_manifest,
            layout=paper.layout,
            plan_sha256=plan_sha256,
            code_state=code_state,
        )
        row_checkpoint, row_entries, row_contract_sha256 = _checkpoint(
            paper_root,
            row_stage.get("checkpoint"),
            expected_path="private/row-enumeration-checkpoint.json",
            expected_schema="row-enumeration-checkpoint/0.4",
            expected_contract=row_contract,
            collection_key="batches",
        )
        del row_checkpoint
        if set(row_entries) != {batch.batch_id for batch in plan.batches}:
            _fail("row checkpoint does not exactly cover its base batches")
        row_outcome = RowEnumerationOutcome(
            unbatchable_row_ids=[item.row_id for item in plan.unbatchable_rows]
        )
        for batch in plan.batches:
            batch_outcome = _validated_row_checkpoint_entry(
                row_entries.get(batch.batch_id),
                batch=batch,
                contract=row_contract,
            )
            if batch_outcome is None:
                _fail("row checkpoint contains an invalid terminal batch")
            _merge_row_outcome(row_outcome, batch_outcome)
        try:
            replayed_ledger = validate_outcome_against(
                plan,
                row_outcome,
                paper_id=paper.paper_id,
                paper_title=spec.title,
                model=settings.row_model or settings.model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                reasoning_effort=settings.reasoning_effort,
                seed=settings.seed,
            )
            persisted_ledger = RowTerminalLedger.model_validate(
                _regular_json(
                    paper_root / "private" / "row-terminal-states.json",
                    "row terminal ledger",
                )
            )
        except ValueError as error:
            raise CompletedCorpusValidationError("row terminal ledger is invalid") from error
        if replayed_ledger.model_dump(mode="json") != persisted_ledger.model_dump(mode="json"):
            _fail("row terminal ledger differs from checkpoint replay")
        terminal_binding = _mapping(row_stage.get("terminal_states"), "row terminal binding")
        terminal_path = paper_root / "private" / "row-terminal-states.json"
        if (
            terminal_binding.get("path") != "private/row-terminal-states.json"
            or terminal_binding.get("artifact_sha256") != sha256_file(terminal_path)
            or terminal_binding.get("counts") != replayed_ledger.counts.model_dump(mode="json")
        ):
            _fail("row terminal-state binding is invalid")
        if row_stage.get("plan") != plan.telemetry.model_dump(mode="json"):
            _fail("row plan telemetry is invalid")
        if row_stage.get("outcome") != row_outcome.telemetry:
            _fail("row outcome telemetry is invalid")
        row_calls, row_successful_calls = partition_row_provider_calls(row_outcome)
        if _call_counter(row_calls) != _recorded_call_counter(row_stage.get("calls"), "row calls"):
            _fail("row call ledger differs from its checkpoint")
        candidates.extend(row_outcome.candidates)
        terminal_counts = replayed_ledger.counts
        row_failures = terminal_counts.unresolved + terminal_counts.unsupported
        row_metrics = {
            "tables_considered": plan.telemetry.tables_considered,
            "dense_tables": plan.telemetry.dense_tables,
            "rows_planned": terminal_counts.planned,
            "rows_resolved": (
                terminal_counts.result + terminal_counts.not_result + terminal_counts.uncertain
            ),
            "rows_unresolved": terminal_counts.unresolved,
            "rows_unbatchable": terminal_counts.unsupported,
            "unknown_row_ids_seen": len(replayed_ledger.protocol_events),
            "invalid_rows_seen": len(replayed_ledger.invalid_row_reasons),
            "dispositions": {
                "not_result": terminal_counts.not_result,
                "result": terminal_counts.result,
                "uncertain": terminal_counts.uncertain,
            },
        }
    elif any(
        row_stage.get(key) is not None
        for key in ("plan_sha256", "plan", "preflight", "terminal_states", "checkpoint")
    ) or _sequence(row_stage.get("calls"), "disabled row calls"):
        _fail("disabled row stage retains execution artifacts")
    _telemetry_matches(row_stage.get("successful_call_telemetry"), row_successful_calls)
    _telemetry_matches(row_stage.get("completed_call_telemetry"), row_calls)

    candidates_before_dedup = len(candidates)
    candidates = validate_non_origin_candidates(
        candidates,
        {paper.layout.source_id: paper.layout},
        min_confidence=settings.min_confidence,
        processed_source_ids=processing.processed_source_ids,
    )
    deduplication = deduplicate_candidates_with_lineage(
        candidates, {paper.layout.source_id: paper.layout}
    )
    candidates = validate_non_origin_candidates(
        deduplication.candidates,
        {paper.layout.source_id: paper.layout},
        min_confidence=settings.min_confidence,
        processed_source_ids=processing.processed_source_ids,
    )

    tuple_stage = _mapping(run.get("tuple_resolution"), "tuple stage")
    tuple_contract = _tuple_checkpoint_contract(
        spec=spec,
        settings=settings,
        manifest=paper.source_manifest,
        layout=paper.layout,
        code_state=code_state,
    )
    tuple_contract_sha256 = sha256_bytes(canonical_json_bytes(tuple_contract))
    tuple_entries: Mapping[str, Any] = {}
    tuple_calls: list[ProviderCall] = []
    tuple_failures = 0
    tuple_selected = 0
    tuple_pass_ids: set[str] = set()
    tuple_gates: dict[str, str] = {}
    if settings.tuple_model is None:
        _fail("current public reporting requires the tuple stage")
    _, tuple_entries, tuple_contract_sha256 = _checkpoint(
        paper_root,
        tuple_stage.get("checkpoint"),
        expected_path="private/tuple-resolution-checkpoint.json",
        expected_schema="tuple-resolution-checkpoint/0.2",
        expected_contract=tuple_contract,
        collection_key="candidates",
    )
    selected_tuple_keys: set[str] = set()
    selected_tuple_ids: set[str] = set()
    tuple_outcome_by_id: dict[str, Mapping[str, Any]] = {}
    tuple_sidecar = _mapping(
        _regular_json(paper_root / "private" / "tuple-resolution.json", "tuple sidecar"),
        "tuple sidecar",
    )
    for outcome in _sequence(tuple_sidecar.get("outcomes"), "tuple outcomes"):
        outcome = _mapping(outcome, "tuple outcome")
        observation_id = outcome.get("observation_id")
        if not isinstance(observation_id, str) or observation_id in tuple_outcome_by_id:
            _fail("tuple outcome identity is invalid")
        tuple_outcome_by_id[observation_id] = outcome
    for candidate in candidates:
        observation_id = str(candidate.observation_id or candidate.stable_id())
        if candidate.claim_type is not ClaimType.PRIMARY_RESULT or (
            candidate.export_status is not ExportStatus.ELIGIBLE
        ):
            tuple_gates[observation_id] = _tuple_gate_sha256(
                candidate=candidate,
                tuple_contract_sha256=tuple_contract_sha256,
                entry_sha256=None,
                status="not_selected",
                semantic_match=False,
                passed=False,
            )
            continue
        pre_gate_candidate = candidate.model_copy(deep=True)
        tuple_key = tuple_candidate_binding_sha256(candidate)
        selected_tuple_keys.add(tuple_key)
        selected_tuple_ids.add(observation_id)
        validated = _validated_tuple_checkpoint_entry(
            tuple_entries.get(tuple_key),
            candidate=candidate,
            settings=settings,
            manifest=paper.source_manifest,
            layout=paper.layout,
            code_state=code_state,
            tuple_contract_sha256=tuple_contract_sha256,
        )
        if validated is None:
            _fail("tuple checkpoint contains an invalid terminal candidate")
        status, assessment, call = validated
        tuple_selected += 1
        if call is not None:
            tuple_calls.append(call)
        semantic_match = bool(
            status == "success"
            and assessment is not None
            and tuple_assessment_is_export_concordant(candidate, assessment)
        )
        passed = bool(
            semantic_match
            and assessment is not None
            and assessment.decision
            in {TupleResolutionDecision.VERIFIED, TupleResolutionDecision.REVIEW}
        )
        if passed:
            tuple_pass_ids.add(observation_id)
        else:
            candidate.export_status = ExportStatus.NEEDS_REVIEW
            if status == "unsupported":
                candidate.export_reason = "tuple_resolution=unsupported_physical_evidence"
            elif status != "success":
                candidate.export_reason = f"tuple_resolution={status}"
                tuple_failures += 1
            elif not semantic_match:
                candidate.export_reason = "tuple_resolution=candidate_mismatch"
            elif assessment is not None:
                candidate.export_reason = f"tuple_resolution={assessment.decision.value}"
            else:
                candidate.export_reason = "tuple_resolution=candidate_mismatch"
        entry = _mapping(tuple_entries[tuple_key], "tuple checkpoint entry")
        gate = _tuple_gate_sha256(
            candidate=pre_gate_candidate,
            tuple_contract_sha256=tuple_contract_sha256,
            entry_sha256=str(entry.get("entry_sha256")),
            status=status,
            semantic_match=semantic_match,
            passed=passed,
        )
        tuple_gates[observation_id] = gate
        outcome = tuple_outcome_by_id.get(observation_id)
        if outcome is None or any(
            (
                outcome.get("status") != status,
                outcome.get("semantic_match") is not semantic_match,
                outcome.get("passed") is not passed,
                outcome.get("gate_sha256") != gate,
                outcome.get("checkpoint_entry") != entry,
            )
        ):
            _fail("tuple sidecar differs from checkpoint replay")
    if set(tuple_entries) != selected_tuple_keys:
        _fail("tuple checkpoint contains stale candidate entries")
    if tuple_sidecar.get("candidate_gates") != dict(sorted(tuple_gates.items())) or (
        tuple_sidecar.get("checkpoint_contract_sha256") != tuple_contract_sha256
        or tuple_sidecar.get("configuration") != _tuple_run_configuration(settings)
    ):
        _fail("tuple sidecar binding is invalid")
    if set(tuple_outcome_by_id) != selected_tuple_ids:
        _fail("tuple sidecar outcome partition is invalid")
    tuple_run_calls = [
        *_public_calls(tuple_stage.get("calls"), "tuple calls"),
        *_public_calls(tuple_stage.get("resumed_calls"), "tuple resumed calls"),
    ]
    if Counter(canonical_json_bytes(dict(call)) for call in tuple_run_calls) != _call_counter(
        tuple_calls
    ):
        _fail("tuple call ledger differs from its checkpoint")
    _telemetry_matches(tuple_stage.get("completed_call_telemetry"), tuple_calls)

    verifier_stage = _mapping(run.get("verifier"), "verifier stage")
    verifier_calls: list[ProviderCall] = []
    verifier_failures = 0
    verifier_selected = 0
    verifier_accept_ids: set[str] = set()
    verifier_decisions: Counter[str] = Counter()
    verifier_gates: dict[str, str] = {}
    replayed_verifications: list[dict[str, Any]] = []
    verifier_contract_sha256: str | None = None
    tuple_sidecar_binding = _mapping(tuple_stage.get("sidecar"), "tuple sidecar")
    tuple_sidecar_sha256 = str(tuple_sidecar_binding.get("sha256"))
    if settings.verifier_model is not None:
        verifier_contract = _verifier_checkpoint_contract(
            spec=spec,
            settings=settings,
            manifest=paper.source_manifest,
            layout=paper.layout,
            code_state=code_state,
            tuple_contract_sha256=tuple_contract_sha256,
        )
        _, verifier_entries, verifier_contract_sha256 = _checkpoint(
            paper_root,
            verifier_stage.get("checkpoint"),
            expected_path="private/verifier-checkpoint.json",
            expected_schema="independent-verifier-checkpoint/0.4",
            expected_contract=verifier_contract,
            collection_key="candidates",
        )
        selected_verifier_keys: set[str] = set()
        for candidate in candidates:
            observation_id = str(candidate.observation_id or candidate.stable_id())
            if (
                observation_id not in tuple_pass_ids
                or observation_id not in tuple_gates
                or candidate.claim_type is not ClaimType.PRIMARY_RESULT
                or candidate.export_status is not ExportStatus.ELIGIBLE
            ):
                continue
            verifier_selected += 1
            support = bind_candidate_block(candidate, blocks)
            if support is None:
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                candidate.export_reason = "no frozen result block contains the evidence quote"
                continue
            block, anchor = support
            evidence_block = frozen_evidence_block(
                paper_id=paper.paper_id,
                block=block,
                anchor=anchor,
            )
            verifier_key = _verifier_candidate_sha256(candidate)
            selected_verifier_keys.add(verifier_key)
            validated = _validated_verifier_checkpoint_entry(
                verifier_entries.get(verifier_key),
                candidate=candidate,
                evidence_block=evidence_block,
                settings=settings,
                manifest=paper.source_manifest,
                layout=paper.layout,
                code_state=code_state,
                tuple_gate_sha256=tuple_gates[observation_id],
            )
            if validated is None:
                _fail("verifier checkpoint contains an invalid terminal candidate")
            status, verification, call = validated
            if call is not None:
                verifier_calls.append(call)
            if status != "success" or verification is None:
                entry = _mapping(verifier_entries[verifier_key], "verifier checkpoint entry")
                error_code = entry.get("error_code")
                if not isinstance(error_code, str):
                    _fail("verifier failure lacks a typed error code")
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                candidate.export_reason = f"independent_verifier={error_code}"
                verifier_failures += 1
                continue
            assert verifier_contract_sha256 is not None
            entry = _mapping(verifier_entries[verifier_key], "verifier checkpoint entry")
            gate = _verifier_gate_sha256(
                candidate=candidate,
                tuple_gate_sha256=tuple_gates[observation_id],
                verifier_contract_sha256=verifier_contract_sha256,
                verifier_entry_sha256=str(entry.get("entry_sha256")),
                verification=verification,
            )
            verifier_gates[observation_id] = gate
            replayed_verifications.append(verification.model_dump(mode="json", exclude_none=True))
            verifier_decisions[verification.effective_decision.value] += 1
            if verification.effective_decision is IndependentDecision.ACCEPT:
                verifier_accept_ids.add(observation_id)
            else:
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                candidate.export_reason = (
                    f"independent_verifier={verification.effective_decision.value}: "
                    f"{verification.provider_assessment.justification}"
                )
        if set(verifier_entries) != selected_verifier_keys:
            _fail("verifier checkpoint contains stale candidate entries")
        if (
            _regular_jsonl(paper_root / "verifications.jsonl", "verifier result ledger")
            != replayed_verifications
        ):
            _fail("verifier result ledger differs from checkpoint replay")
        sidecar = _mapping(
            _regular_json(paper_root / "private" / "verifier-gates.json", "verifier sidecar"),
            "verifier sidecar",
        )
        if (
            sidecar.get("schema_version") != "independent-verifier-gates/0.2"
            or sidecar.get("configuration") != _verifier_run_configuration(settings)
            or sidecar.get("checkpoint_contract_sha256") != verifier_contract_sha256
            or sidecar.get("candidate_gates") != dict(sorted(verifier_gates.items()))
            or sidecar.get("passed_candidate_ids") != sorted(verifier_accept_ids)
        ):
            _fail("verifier sidecar binding is invalid")
        verifier_run_calls = [
            *_public_calls(verifier_stage.get("calls"), "verifier calls"),
            *_public_calls(verifier_stage.get("resumed_calls"), "verifier resumed calls"),
        ]
        if Counter(
            canonical_json_bytes(dict(call)) for call in verifier_run_calls
        ) != _call_counter(verifier_calls):
            _fail("verifier call ledger differs from its checkpoint")
        _telemetry_matches(verifier_stage.get("completed_call_telemetry"), verifier_calls)
    elif (
        any(verifier_stage.get(key) is not None for key in ("sidecar", "checkpoint"))
        or _sequence(verifier_stage.get("calls"), "disabled verifier calls")
        or _sequence(verifier_stage.get("resumed_calls"), "disabled verifier resumed calls")
    ):
        _fail("disabled verifier stage retains execution artifacts")
    elif _regular_jsonl(paper_root / "verifications.jsonl", "disabled verifier result ledger"):
        _fail("disabled verifier stage retains verification results")

    candidates = route_candidate_attribution(candidates, {paper.layout.source_id: paper.layout})
    origin_stage = _mapping(run.get("origin_retrieval"), "origin stage")
    origin_calls: list[ProviderCall] = []
    origin_failures = 0
    origin_selected = 0
    origin_positive_review_only = 0
    origin_states: Counter[str] = Counter()
    origin_contract_sha256: str | None = None
    if settings.origin_model is not None:
        if verifier_contract_sha256 is None:
            _fail("origin stage lacks the required verifier contract")
        origin_contract = _origin_checkpoint_contract(
            spec=spec,
            settings=settings,
            manifest=paper.source_manifest,
            layout=paper.layout,
            code_state=code_state,
            tuple_contract_sha256=tuple_contract_sha256,
            verifier_contract_sha256=verifier_contract_sha256,
        )
        _, origin_entries, origin_contract_sha256 = _checkpoint(
            paper_root,
            origin_stage.get("checkpoint"),
            expected_path="private/origin-retrieval-checkpoint.json",
            expected_schema="producer-origin-checkpoint/0.3",
            expected_contract=origin_contract,
            collection_key="candidates",
        )
        selected_origin_keys: set[str] = set()
        for candidate in candidates:
            observation_id = str(candidate.observation_id or candidate.stable_id())
            if (
                observation_id not in verifier_accept_ids
                or observation_id not in tuple_pass_ids
                or observation_id not in verifier_gates
            ):
                continue
            origin_selected += 1
            if (
                candidate.attribution is not None
                and candidate.attribution.state is AttributionState.EXTERNALLY_SOURCED
            ):
                origin_states[AttributionState.EXTERNALLY_SOURCED.value] += 1
                continue
            origin_key = candidate_origin_binding_sha256(candidate)
            selected_origin_keys.add(origin_key)
            validated = _validated_origin_checkpoint_entry(
                origin_entries.get(origin_key),
                candidate=candidate,
                layout=paper.layout,
                settings=settings,
                code_state=code_state,
                tuple_gate_sha256=tuple_gates[observation_id],
                verifier_gate_sha256=verifier_gates[observation_id],
            )
            if validated is None:
                _fail("origin checkpoint contains an invalid terminal candidate")
            status, _, _, assessment, call, error_code = validated
            if call is not None:
                origin_calls.append(call)
            if status != "success":
                if error_code is None:
                    _fail("origin failure lacks a typed error code")
                origin_states[
                    (
                        candidate.attribution.state.value
                        if candidate.attribution is not None
                        else AttributionState.UNRESOLVED.value
                    )
                ] += 1
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                candidate.export_reason = f"origin_retrieval={error_code}"
                origin_failures += 1
                continue
            if assessment is None:
                _fail("successful origin checkpoint lacks its assessment")
            origin_states[assessment.effective_state.value] += 1
            origin_positive_review_only += assessment.positive_evidence_verified
            _apply_origin_assessment(candidate, assessment)
        if set(origin_entries) != selected_origin_keys:
            _fail("origin checkpoint contains stale candidate entries")
        origin_run_calls = [
            *_public_calls(origin_stage.get("calls"), "origin calls"),
            *_public_calls(origin_stage.get("resumed_calls"), "origin resumed calls"),
        ]
        if Counter(canonical_json_bytes(dict(call)) for call in origin_run_calls) != _call_counter(
            origin_calls
        ):
            _fail("origin call ledger differs from its checkpoint")
        _telemetry_matches(origin_stage.get("completed_call_telemetry"), origin_calls)
    elif (
        origin_stage.get("checkpoint") is not None
        or _sequence(origin_stage.get("calls"), "disabled origin calls")
        or _sequence(origin_stage.get("resumed_calls"), "disabled origin resumed calls")
    ):
        _fail("disabled origin stage retains execution artifacts")

    for candidate in candidates:
        observation_id = str(candidate.observation_id or candidate.stable_id())
        if candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED} and (
            observation_id not in tuple_pass_ids or observation_id not in tuple_gates
        ):
            candidate.export_status = ExportStatus.NEEDS_REVIEW
            candidate.export_reason = "tuple_resolution=missing_verified_concordant_gate"
    if settings.verifier_model is None:
        for candidate in candidates:
            if candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}:
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                candidate.export_reason = "independent_verifier=disabled_review_only"

    if [item.model_dump(mode="json") for item in candidates] != [
        item.model_dump(mode="json") for item in paper.candidates
    ]:
        _fail("final observations differ from production checkpoint replay")
    recorded_counts = _mapping(run.get("counts"), "paper counts")
    if set(recorded_counts) != _COUNT_KEYS:
        _fail("paper count vocabulary is invalid")
    typed_recorded_counts = {
        key: _integer(value, f"paper count {key}") for key, value in recorded_counts.items()
    }
    candidates_needing_review = sum(
        candidate.export_status is ExportStatus.NEEDS_REVIEW for candidate in candidates
    )
    expected_public_counts = {
        "candidates": len(candidates),
        "candidates_before_deduplication": candidates_before_dedup,
        "duplicates_removed": candidates_before_dedup - len(candidates),
        "candidates_needing_review": candidates_needing_review,
        "semantic_safety_reviews": sum(
            any(note.startswith("semantic safety:") for note in candidate.notes)
            for candidate in candidates
            if candidate.export_status is ExportStatus.NEEDS_REVIEW
        ),
        "primary_results": sum(
            candidate.claim_type is ClaimType.PRIMARY_RESULT for candidate in candidates
        ),
        "exported": 0,
        "eee_records": 0,
        "eee_schema_issues": 0,
        "tuple_candidates": tuple_selected,
        "tuple_passed": len(tuple_pass_ids),
        "tuple_review": tuple_selected - len(tuple_pass_ids),
        "tuple_unsupported": sum(
            outcome.get("status") == "unsupported" for outcome in tuple_outcome_by_id.values()
        ),
        "tuple_failed": tuple_failures,
        "verifications": sum(verifier_decisions.values()),
        "verifier_accepts": verifier_decisions[IndependentDecision.ACCEPT.value],
        "verifier_rejects": verifier_decisions[IndependentDecision.REJECT.value],
        "verifier_reviews": verifier_decisions[IndependentDecision.REVIEW.value],
        "verifier_failed": verifier_failures,
        "origin_candidates": origin_selected,
        "origin_failed": origin_failures,
        "origin_positive_review_only": origin_positive_review_only,
        "origin_external": origin_states[AttributionState.EXTERNALLY_SOURCED.value],
        "origin_unresolved": origin_states[AttributionState.UNRESOLVED.value],
        "origin_no_signal": origin_states[AttributionState.NO_SIGNAL.value],
    }
    if set(expected_public_counts) != _PUBLIC_COUNT_KEYS or any(
        typed_recorded_counts[key] != value for key, value in expected_public_counts.items()
    ):
        _fail("paper public counts differ from checkpoint replay")
    eee_root = paper_root / "eee"
    if (
        not eee_root.is_dir()
        or eee_root.is_symlink()
        or any(path.is_file() for path in eee_root.glob("*.json"))
    ):
        _fail("current automatic source run must have zero canonical EEE records")

    if settings.verifier_model is not None:
        verifier_sidecar_binding = _mapping(verifier_stage.get("sidecar"), "verifier sidecar")
        verifier_sidecar_sha256 = str(verifier_sidecar_binding.get("sha256"))
        provenance = tuple_gated_export_provenance(
            [],
            tuple_sidecar_sha256=tuple_sidecar_sha256,
            tuple_gates={},
            verifier_sidecar_sha256=verifier_sidecar_sha256,
            verifier_gates={},
        )
    else:
        verifier_sidecar_sha256 = None
        provenance = tuple_unverified_review_provenance(tuple_sidecar_sha256=tuple_sidecar_sha256)
    if tuple_sidecar_sha256 != sha256_file(paper_root / "private" / "tuple-resolution.json"):
        _fail("tuple sidecar digest is invalid")
    if verifier_sidecar_sha256 is not None and verifier_sidecar_sha256 != sha256_file(
        paper_root / "private" / "verifier-gates.json"
    ):
        _fail("verifier sidecar digest is invalid")
    try:
        expected_lineage = build_candidate_lineage(
            paper_id=paper.paper_id,
            candidates=candidates,
            merge_kinds={
                observation_id: kind.value
                for observation_id, kind in deduplication.kind_by_observation_id.items()
            },
            observations_sha256=sha256_file(paper.observations_path),
            valid_records=[],
            invalid_records=[],
            export_provenance=provenance,
            tuple_gates=tuple_gates,
            verifier_gates=verifier_gates,
        )
    except ValueError as error:
        raise CompletedCorpusValidationError("candidate lineage replay failed") from error
    if paper.candidate_lineage is None or paper.candidate_lineage.model_dump(
        mode="json"
    ) != expected_lineage.model_dump(mode="json"):
        _fail("candidate lineage differs from production replay")

    stage_receipts = {
        "extractor": ValidatedStageReceipt(
            status="partial_failure" if extractor_failures else "validated",
            completed_calls=len(extractor_completed_calls),
            selected_items=len(blocks),
            failed_items=extractor_failures,
            checkpoint_contract_sha256=extractor_contract_sha256,
        ),
        "row_disposition": ValidatedStageReceipt(
            status=(
                "not_run"
                if not settings.row_enumeration_enabled
                else "partial_failure"
                if row_failures
                else "validated"
            ),
            completed_calls=len(row_calls),
            selected_items=int(row_metrics["rows_planned"]),
            failed_items=row_failures,
            checkpoint_contract_sha256=row_contract_sha256,
        ),
        "tuple_resolution": ValidatedStageReceipt(
            status="partial_failure" if tuple_failures else "validated",
            completed_calls=len(tuple_calls),
            selected_items=tuple_selected,
            failed_items=tuple_failures,
            checkpoint_contract_sha256=tuple_contract_sha256,
            sidecar_sha256=tuple_sidecar_sha256,
        ),
        "independent_verification": ValidatedStageReceipt(
            status=(
                "not_run"
                if settings.verifier_model is None
                else "partial_failure"
                if verifier_failures
                else "validated"
            ),
            completed_calls=len(verifier_calls),
            selected_items=verifier_selected,
            failed_items=verifier_failures,
            checkpoint_contract_sha256=verifier_contract_sha256,
            sidecar_sha256=verifier_sidecar_sha256,
        ),
        "origin_retrieval": ValidatedStageReceipt(
            status=(
                "not_run"
                if settings.origin_model is None
                else "partial_failure"
                if origin_failures
                else "validated"
            ),
            completed_calls=len(origin_calls),
            selected_items=origin_selected,
            failed_items=origin_failures,
            checkpoint_contract_sha256=origin_contract_sha256,
        ),
    }
    technical_failure = any(item.status == "partial_failure" for item in stage_receipts.values())
    expected_status = "partial_failure" if technical_failure else "success"
    if run.get("status") != expected_status:
        _fail("paper status differs from validated terminal stage state")
    review_state = _mapping(run.get("review_state"), "paper review state")
    expected_review_reasons: list[str] = []
    if not blocks:
        expected_review_reasons.append("zero_selected_result_blocks")
    elif not candidates:
        expected_review_reasons.append("selected_result_blocks_produced_zero_candidates")
    expected_review_reasons.append("zero_valid_eee_records")
    if row_metrics["rows_unresolved"]:
        expected_review_reasons.append("row_enumeration_unresolved")
    if row_metrics["rows_unbatchable"]:
        expected_review_reasons.append("row_enumeration_unbatchable")
    if row_metrics["unknown_row_ids_seen"]:
        expected_review_reasons.append("row_enumeration_unknown_ids")
    if row_metrics["dispositions"]["uncertain"]:
        expected_review_reasons.append("row_enumeration_uncertain")
    if candidates_needing_review:
        expected_review_reasons.append("candidate_review_required")
    if tuple_selected != len(tuple_pass_ids):
        expected_review_reasons.append("tuple_resolution_review_required")
    expected_review_state = {
        "status": "needs_review" if expected_review_reasons else "ready",
        "reasons": expected_review_reasons,
    }
    if review_state != expected_review_state or any(
        reason not in _REVIEW_REASONS for reason in expected_review_reasons
    ):
        _fail("review state differs from checkpoint replay")

    receipt = ValidatedPaperReceipt(
        paper_id=paper.paper_id,
        status=expected_status,
        needs_review=bool(expected_review_reasons),
        review_reasons=tuple(expected_review_reasons),
        counts=expected_public_counts,
        stages=stage_receipts,
        export_status_counts=Counter(item.export_status.value for item in candidates),
        text_support_counts=Counter(item.text_support.value for item in candidates),
        referential_status_counts=Counter(item.referential_status.value for item in candidates),
        attribution_state_counts=Counter(
            item.attribution.state.value for item in candidates if item.attribution is not None
        ),
        row_metrics=row_metrics,
        lineage_counts=expected_lineage.counts.model_dump(mode="json"),
        tuple_sidecar_sha256=tuple_sidecar_sha256,
        verifier_sidecar_sha256=verifier_sidecar_sha256,
    )
    return receipt, tuple(
        [*extractor_completed_calls, *row_calls, *tuple_calls, *verifier_calls, *origin_calls]
    )


def _shared(values: Sequence[Any], context: str) -> Any:
    if not values or len({canonical_json_bytes(item) for item in values}) != 1:
        _fail(f"{context} differs across paper runs")
    return values[0]


def validate_completed_corpus_run(run_root: Path) -> ValidatedCorpusReceipt:
    """Verify one sealed current corpus and return an internal reporting receipt.

    No provider is available on this path.  Every current paper is reconstructed
    from sealed production checkpoints before any aggregate is returned.
    """

    root = run_root.resolve()
    try:
        seal = verify_run_seal(root)
    except RunSealVerificationError as error:
        raise CompletedCorpusValidationError(
            "current public summary requires a valid run seal"
        ) from error
    corpus_path = root / "corpus-run.json"
    corpus = _mapping(_regular_json(corpus_path, "corpus run"), "corpus run")
    if corpus.get("schema_version") != "corpus-run/0.3":
        _fail("current corpus schema is unsupported")
    binding = _mapping(corpus.get("corpus_binding"), "corpus binding")
    if (
        binding.get("schema_version") != "pilot-corpus/0.2"
        or binding.get("evaluation_split") != "development"
        or binding.get("corpus_id") != corpus.get("corpus_id")
    ):
        _fail("public summary requires an exact development corpus binding")
    try:
        run_kind, loaded_papers, loaded_corpus_path = _load_run_papers(root)
    except ReviewedExportError as error:
        if "verifier" in error.detail.casefold():
            raise CompletedCorpusValidationError("verifier sidecar binding is invalid") from error
        if "tuple" in error.detail.casefold():
            raise CompletedCorpusValidationError("tuple sidecar binding is invalid") from error
        raise CompletedCorpusValidationError(
            "sealed run membership or gate binding is invalid"
        ) from error
    if run_kind != "corpus" or loaded_corpus_path != corpus_path:
        _fail("public current summary requires a sealed corpus run")
    replayed = tuple(_replay_paper(paper) for paper in loaded_papers)
    papers = tuple(item[0] for item in replayed)
    completed_calls = [call for _, calls in replayed for call in calls]
    if not papers:
        _fail("sealed corpus contains no paper runs")
    paper_ids = [paper.paper_id for paper in papers]
    if binding.get("paper_ids_sha256") != sha256_bytes(canonical_json_bytes(paper_ids)):
        _fail("corpus paper order differs from its binding")
    if _integer(corpus.get("papers"), "corpus papers") != len(papers):
        _fail("corpus paper count is invalid")

    runs = [paper.run_manifest for paper in loaded_papers]
    count_maps = [_mapping(run.get("counts"), "paper counts") for run in runs]
    corpus_totals = _mapping(corpus.get("totals"), "corpus totals")
    for key, value in corpus_totals.items():
        total = sum(_integer(counts.get(key), f"paper count {key}") for counts in count_maps)
        if _integer(value, f"corpus total {key}") != total:
            _fail("corpus totals differ from paper receipts")
    if set(corpus_totals) != _COUNT_KEYS or any(
        set(counts) != _COUNT_KEYS for counts in count_maps
    ):
        # Every current run uses one exact count vocabulary.  Reject silent omissions.
        _fail("corpus total vocabulary differs from paper runs")
    totals = {key: sum(paper.counts[key] for paper in papers) for key in sorted(_PUBLIC_COUNT_KEYS)}

    succeeded = sum(paper.status == "success" for paper in papers)
    failed = len(papers) - succeeded
    expected_status = "success" if not failed else "error" if not succeeded else "partial_failure"
    if (
        corpus.get("status") != expected_status
        or _integer(corpus.get("papers_succeeded"), "papers succeeded") != succeeded
        or _integer(corpus.get("papers_failed"), "papers failed") != failed
        or _integer(corpus.get("papers_needing_review"), "papers needing review")
        != sum(paper.needs_review for paper in papers)
    ):
        _fail("corpus status accounting differs from validated paper state")
    operations = _mapping(corpus.get("operations"), "corpus operations")
    wall_clock = operations.get("wall_clock_seconds")
    if isinstance(wall_clock, bool) or not isinstance(wall_clock, int | float) or wall_clock < 0:
        _fail("corpus wall-clock telemetry is invalid")
    expected_operations = _corpus_operational_metrics(
        [dict(run) for run in runs], wall_clock_seconds=float(wall_clock)
    )
    if operations != expected_operations:
        _fail("corpus operational telemetry differs from paper stages")

    provider_telemetry = _provider_call_telemetry(
        completed_calls,
        basis=(
            "validated completed calls reconstructed from sealed production checkpoints; "
            "cost, token, retry, and attempt totals remain lower bounds when provider "
            "metadata is unavailable"
        ),
    )
    provider_telemetry = {
        "recorded_structured_invocations": provider_telemetry.pop("calls"),
        **provider_telemetry,
    }

    code = _mapping(_shared([run.get("code") for run in runs], "code binding"), "code")
    schema = _mapping(
        _shared([run.get("eee_schema") for run in runs], "EEE schema binding"), "EEE schema"
    )
    if schema.get("version") != EEE_SCHEMA_VERSION or schema.get("sha256") != EEE_SCHEMA_SHA256:
        _fail("current summary requires the pinned EEE schema")
    extractor = _mapping(
        _shared(
            [_extractor_run_configuration(_paper_settings(run, root)) for run in runs],
            "extractor configuration",
        ),
        "extractor configuration",
    )
    row = _mapping(
        _shared(
            [_row_enumeration_run_configuration(_paper_settings(run, root)) for run in runs],
            "row configuration",
        ),
        "row configuration",
    )
    candidate_validation = _mapping(
        _shared([run.get("candidate_validation") for run in runs], "candidate validation"),
        "candidate validation",
    )
    generated_at = corpus.get("generated_at")
    if not isinstance(generated_at, str):
        _fail("corpus generation timestamp is invalid")
    corpus_id = corpus.get("corpus_id")
    if not isinstance(corpus_id, str) or not corpus_id:
        _fail("corpus id is invalid")
    corpus_spec_sha256 = binding.get("corpus_spec_sha256")
    paper_ids_sha256 = binding.get("paper_ids_sha256")
    if not all(
        isinstance(item, str) and _SHA256.fullmatch(item) is not None
        for item in (corpus_spec_sha256, paper_ids_sha256)
    ):
        _fail("corpus hash binding is invalid")
    return ValidatedCorpusReceipt(
        seal=seal,
        corpus_id=corpus_id,
        corpus_run_sha256=sha256_file(corpus_path),
        corpus_spec_sha256=str(corpus_spec_sha256),
        paper_ids_sha256=str(paper_ids_sha256),
        generated_at=generated_at,
        code=dict(code),
        eee_schema={"version": str(schema.get("version")), "sha256": str(schema.get("sha256"))},
        extractor_binding=_configuration_binding(extractor),
        row_binding=_configuration_binding(row),
        candidate_validation=dict(candidate_validation),
        corpus_status=expected_status,
        papers=papers,
        totals=totals,
        operations=dict(operations),
        provider_telemetry=provider_telemetry,
    )


def aggregate_stage_receipts(receipt: ValidatedCorpusReceipt) -> dict[str, dict[str, Any]]:
    """Aggregate paper receipts without retaining paper or candidate identifiers."""

    result: dict[str, dict[str, Any]] = {}
    for stage in _STAGES:
        paper_stages = [paper.stages[stage] for paper in receipt.papers]
        statuses = {item.status for item in paper_stages}
        status: StageStatus = (
            "partial_failure"
            if "partial_failure" in statuses
            else "validated"
            if statuses == {"validated"}
            else "not_run"
            if statuses == {"not_run"}
            else "partial_failure"
        )
        result[stage] = {
            "status": status,
            "papers_validated": sum(item.status == "validated" for item in paper_stages),
            "papers_partial_failure": sum(
                item.status == "partial_failure" for item in paper_stages
            ),
            "papers_not_run": sum(item.status == "not_run" for item in paper_stages),
            "completed_calls": sum(item.completed_calls for item in paper_stages),
            "selected_items": sum(item.selected_items for item in paper_stages),
            "failed_items": sum(item.failed_items for item in paper_stages),
            "checkpoint_contract_sha256s": sorted(
                {
                    item.checkpoint_contract_sha256
                    for item in paper_stages
                    if item.checkpoint_contract_sha256 is not None
                }
            ),
            "sidecar_sha256s": sorted(
                {item.sidecar_sha256 for item in paper_stages if item.sidecar_sha256 is not None}
            ),
        }
    return result


def aggregate_provider_telemetry(receipt: ValidatedCorpusReceipt) -> dict[str, Any]:
    return dict(receipt.provider_telemetry)


def empty_source_provenance_counts() -> dict[str, int]:
    return {mode: 0 for mode in _PROVENANCE_MODES}
