"""Deterministic aggregate projection of one development run for publication."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from proceedings_to_eee.domain.attribution import AttributionState
from proceedings_to_eee.domain.lineage import CandidateLineageArtifact
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ExportStatus
from proceedings_to_eee.extraction.row_enumeration import (
    RowDisposition,
    RowEnumerationConfig,
    RowEnumerationPlan,
    RowTerminalLedger,
    RowTerminalState,
)
from proceedings_to_eee.extraction.row_validation import validate_terminal_ledger_against
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from proceedings_to_eee.providers.openrouter import ProviderCall
from proceedings_to_eee.reporting.extraction_review_cards import (
    ExtractionReviewCardError,
    _assert_public_payload,
)
from proceedings_to_eee.reporting.validated_corpus import (
    CompletedCorpusValidationError,
    ValidatedCorpusReceipt,
    aggregate_provider_telemetry,
    aggregate_stage_receipts,
    empty_source_provenance_counts,
    validate_completed_corpus_run,
)
from proceedings_to_eee.resources import DEFAULT_EEE_SCHEMA_PATH
from proceedings_to_eee.reviewed_export.models import DerivedRunManifest
from proceedings_to_eee.reviewed_export.workflow import (
    DERIVED_MANIFEST_NAME,
    ReviewedExportError,
    verify_contextual_derived_run,
)
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record

PUBLIC_DEVELOPMENT_SUMMARY_SCHEMA_VERSION = "public-development-summary/0.3"
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]*(?:/[A-Za-z0-9][A-Za-z0-9._:+-]*){1,3}$")
_REASONING_EFFORTS = {None, "none", "minimal", "low", "medium", "high", "xhigh"}
_REVIEW_REASONS = {
    "candidate_review_required",
    "eee_schema_validation_failure",
    "paper_run_error",
    "row_enumeration_unbatchable",
    "row_enumeration_uncertain",
    "row_enumeration_unknown_ids",
    "row_enumeration_unresolved",
    "selected_result_blocks_produced_zero_candidates",
    "zero_selected_result_blocks",
    "zero_valid_eee_records",
}
_ROW_DISPOSITIONS = {item.value for item in RowDisposition}


class PublicDevelopmentSummaryError(ValueError):
    """A run cannot be projected into a trustworthy public aggregate."""


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicDevelopmentSummaryError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise PublicDevelopmentSummaryError(f"{context} must be an array")
    return value


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicDevelopmentSummaryError(f"{context} must be a non-negative integer")
    return value


def _optional_nonnegative_number(value: object, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PublicDevelopmentSummaryError(f"{context} must be numeric or null")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise PublicDevelopmentSummaryError(f"{context} must be finite and non-negative")
    return number


def _optional_rate(value: object, context: str) -> float | None:
    number = _optional_nonnegative_number(value, context)
    if number is not None and number > 1:
        raise PublicDevelopmentSummaryError(f"{context} must be at most one")
    return number


def _optional_temperature(value: object, context: str) -> float | None:
    number = _optional_nonnegative_number(value, context)
    if number is not None and number > 2:
        raise PublicDevelopmentSummaryError(f"{context} must be at most two")
    return number


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PublicDevelopmentSummaryError(f"{context} must be a lowercase SHA-256")
    return value


def _safe_identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PublicDevelopmentSummaryError(f"{context} is not a public-safe identifier")
    return value


def _unique_strings(value: object, context: str) -> list[str]:
    items = list(_sequence(value, context))
    if any(not isinstance(item, str) or not item for item in items):
        raise PublicDevelopmentSummaryError(f"{context} must contain non-empty strings")
    if len(items) != len(set(items)):
        raise PublicDevelopmentSummaryError(f"{context} must not contain duplicates")
    return items


def _generated_at(value: object) -> str:
    if not isinstance(value, str):
        raise PublicDevelopmentSummaryError("generated_at must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PublicDevelopmentSummaryError("generated_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise PublicDevelopmentSummaryError("generated_at must include a timezone")
    return value


def _shared(values: Sequence[Any], context: str) -> Any:
    if not values:
        raise PublicDevelopmentSummaryError(f"{context} is missing")
    encoded = {canonical_json_bytes(value) for value in values}
    if len(encoded) != 1:
        raise PublicDevelopmentSummaryError(f"{context} differs across paper runs")
    return values[0]


def _stage_binding(stage: Mapping[str, Any], context: str) -> dict[str, Any]:
    request_contract = _mapping(stage.get("request_contract"), f"{context}.request_contract")
    prompt_sha256 = _sha256(stage.get("prompt_sha256"), f"{context}.prompt_sha256")
    provider = stage.get("provider")
    model = stage.get("model")
    if not isinstance(provider, str) or not provider:
        raise PublicDevelopmentSummaryError(f"{context}.provider must be a non-empty string")
    if not isinstance(model, str) or len(model) > 255 or _MODEL_ID.fullmatch(model) is None:
        raise PublicDevelopmentSummaryError(f"{context}.model is not a public-safe model ID")
    reasoning_effort = stage.get("reasoning_effort")
    if reasoning_effort not in _REASONING_EFFORTS:
        raise PublicDevelopmentSummaryError(f"{context}.reasoning_effort is unsupported")
    seed = stage.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise PublicDevelopmentSummaryError(
            f"{context}.seed must be null or a non-negative integer"
        )
    return {
        "provider": _safe_identifier(provider, f"{context}.provider"),
        "model": model,
        "temperature": _optional_temperature(stage.get("temperature"), f"{context}.temperature"),
        "reasoning_effort": reasoning_effort,
        "max_tokens": _nonnegative_int(stage.get("max_tokens"), f"{context}.max_tokens"),
        "seed": seed,
        "prompt_sha256": prompt_sha256,
        "request_contract_sha256": sha256_bytes(canonical_json_bytes(request_contract)),
    }


def _validate_eee_outputs(
    run_root: Path,
    runs: Sequence[Mapping[str, Any]],
    exported_ids: set[str],
    expected_records: int,
    schema_sha256: str,
) -> None:
    schema, _ = load_schema(DEFAULT_EEE_SCHEMA_PATH, schema_sha256)
    result_ids: list[str] = []
    record_count = 0
    for run in runs:
        paper_id = str(run["paper_id"])
        eee_root = run_root / paper_id / "eee"
        if not eee_root.is_dir() or eee_root.is_symlink():
            raise PublicDevelopmentSummaryError("each paper run requires a regular EEE directory")
        for path in sorted(eee_root.glob("*.json")):
            if not path.is_file() or path.is_symlink():
                raise PublicDevelopmentSummaryError("EEE outputs must be regular JSON files")
            record = _mapping(read_json(path), "EEE record")
            issues = validate_eee_record(record, schema)
            if issues:
                raise PublicDevelopmentSummaryError("a retained EEE record fails the pinned schema")
            results = _sequence(record.get("evaluation_results"), "EEE evaluation_results")
            for result in results:
                result = _mapping(result, "EEE evaluation result")
                result_id = result.get("evaluation_result_id")
                if not isinstance(result_id, str):
                    raise PublicDevelopmentSummaryError("EEE evaluation_result_id must be a string")
                result_ids.append(result_id)
            record_count += 1
    if record_count != expected_records:
        raise PublicDevelopmentSummaryError("EEE file count does not match the run report")
    if len(result_ids) != len(set(result_ids)):
        raise PublicDevelopmentSummaryError("EEE result identifiers must be unique")
    if set(result_ids) != exported_ids:
        raise PublicDevelopmentSummaryError(
            "EEE result identifiers do not match exported observations"
        )


def _load_candidates(path: Path, expected_count: int) -> list[CandidateObservation]:
    if not path.is_file() or path.is_symlink():
        raise PublicDevelopmentSummaryError("each paper run requires a regular observations file")
    candidates: list[CandidateObservation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            candidates.append(CandidateObservation.model_validate(json.loads(line)))
        except (ValueError, json.JSONDecodeError) as error:
            raise PublicDevelopmentSummaryError(
                f"observations record {line_number} is invalid"
            ) from error
    if len(candidates) != expected_count:
        raise PublicDevelopmentSummaryError("candidate count does not match observations file")
    return candidates


def _recorded_call_telemetry(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    legacy_calls: list[Mapping[str, Any]] = []
    row_calls: list[Mapping[str, Any]] = []
    uncompleted_attempts = 0
    row_batches_resumed = 0
    for run in runs:
        extractor = _mapping(run.get("extractor"), "run.extractor")
        row_stage = _mapping(run.get("row_enumeration"), "run.row_enumeration")
        legacy_calls.extend(
            _mapping(item, "extractor call")
            for item in _sequence(extractor.get("calls"), "extractor.calls")
        )
        row_calls.extend(
            _mapping(item, "row call")
            for item in _sequence(row_stage.get("calls"), "row_enumeration.calls")
        )
        execution = _mapping(row_stage.get("execution"), "row_enumeration.execution")
        row_batches_resumed += _nonnegative_int(
            execution.get("batches_resumed"), "row execution.batches_resumed"
        )
        for attempts, context in (
            (extractor.get("block_attempts"), "extractor.block_attempts"),
            (row_stage.get("attempts"), "row_enumeration.attempts"),
        ):
            for attempt in _sequence(attempts, context):
                attempt = _mapping(attempt, f"{context} item")
                if attempt.get("completed_provider_call") is False:
                    uncompleted_attempts += 1

    calls = [*legacy_calls, *row_calls]
    costs = [_optional_nonnegative_number(call.get("cost_usd"), "call.cost_usd") for call in calls]
    input_tokens = [
        call.get("input_tokens") for call in calls if call.get("input_tokens") is not None
    ]
    output_tokens = [
        call.get("output_tokens") for call in calls if call.get("output_tokens") is not None
    ]
    reasoning_tokens = [
        call.get("reasoning_tokens") for call in calls if call.get("reasoning_tokens") is not None
    ]
    total_tokens = [
        call.get("total_tokens") for call in calls if call.get("total_tokens") is not None
    ]
    latencies = [
        _optional_nonnegative_number(call.get("latency_seconds"), "call.latency_seconds")
        for call in calls
    ]
    attempts = [_nonnegative_int(call.get("attempts"), "call.attempts") for call in calls]
    if any(value < 1 for value in attempts):
        raise PublicDevelopmentSummaryError("completed call attempts must be positive")
    numeric_costs = [value for value in costs if value is not None]
    numeric_latencies = [value for value in latencies if value is not None]
    return {
        "basis": (
            "Completed provider invocations retained in the final development-run artifact. "
            "Legacy checkpoint calls are excluded. Row calls restored from a row checkpoint "
            "remain included because the final artifact cannot distinguish process-local spend. "
            "Cost, token, retry, and attempt totals are lower bounds when provider metadata or "
            "failed transport attempts are unavailable."
        ),
        "recorded_legacy_structured_invocations": len(legacy_calls),
        "recorded_row_structured_invocations": len(row_calls),
        "recorded_structured_invocations": len(calls),
        "row_batches_resumed": row_batches_resumed,
        "cost_usd_lower_bound": round(sum(numeric_costs), 12),
        "cost_reported_invocations": len(numeric_costs),
        "cost_missing_invocations": len(calls) - len(numeric_costs),
        "input_tokens_lower_bound": sum(
            _nonnegative_int(value, "call.input_tokens") for value in input_tokens
        ),
        "input_tokens_reported_invocations": len(input_tokens),
        "input_tokens_missing_invocations": len(calls) - len(input_tokens),
        "output_tokens_lower_bound": sum(
            _nonnegative_int(value, "call.output_tokens") for value in output_tokens
        ),
        "output_tokens_reported_invocations": len(output_tokens),
        "output_tokens_missing_invocations": len(calls) - len(output_tokens),
        "reasoning_tokens_lower_bound": sum(
            _nonnegative_int(value, "call.reasoning_tokens") for value in reasoning_tokens
        ),
        "reasoning_tokens_reported_invocations": len(reasoning_tokens),
        "reasoning_tokens_missing_invocations": len(calls) - len(reasoning_tokens),
        "total_tokens_lower_bound": sum(
            _nonnegative_int(value, "call.total_tokens") for value in total_tokens
        ),
        "total_tokens_reported_invocations": len(total_tokens),
        "total_tokens_missing_invocations": len(calls) - len(total_tokens),
        "latency_seconds_total": round(sum(numeric_latencies), 6),
        "latency_seconds_max": round(max(numeric_latencies), 6) if numeric_latencies else None,
        "transport_attempts_lower_bound": sum(attempts),
        "retries_lower_bound": sum(attempts) - len(calls),
        "attempt_records_without_completed_provider_call": uncompleted_attempts,
    }


def _build_legacy_public_development_summary(run_root: Path) -> dict[str, Any]:
    """Build the explicit historical projection for a pipeline-run/0.2 corpus."""

    run_root = run_root.resolve()
    corpus_path = run_root / "corpus-run.json"
    if not corpus_path.is_file() or corpus_path.is_symlink():
        raise PublicDevelopmentSummaryError("run root requires a regular corpus-run.json")
    corpus = _mapping(read_json(corpus_path), "corpus-run.json")
    if corpus.get("schema_version") != "corpus-run/0.2":
        raise PublicDevelopmentSummaryError("unsupported corpus-run schema")
    corpus_id = _safe_identifier(corpus.get("corpus_id"), "corpus id")
    run_id = _safe_identifier(run_root.name, "run id")

    runs = [
        _mapping(item, "corpus run item")
        for item in _sequence(corpus.get("runs"), "corpus-run.runs")
    ]
    if not runs:
        raise PublicDevelopmentSummaryError("corpus run contains no paper runs")
    corpus_binding = _mapping(corpus.get("corpus_binding"), "corpus binding")
    if corpus_binding.get("schema_version") != "pilot-corpus/0.2":
        raise PublicDevelopmentSummaryError("unsupported corpus binding schema")
    if corpus_binding.get("evaluation_split") != "development":
        raise PublicDevelopmentSummaryError(
            "public development summary requires an explicit development corpus binding"
        )
    if corpus_binding.get("corpus_id") != corpus_id:
        raise PublicDevelopmentSummaryError("corpus binding id does not match corpus-run")
    corpus_spec_sha256 = _sha256(corpus_binding.get("corpus_spec_sha256"), "corpus spec SHA-256")
    paper_ids_sha256 = _sha256(corpus_binding.get("paper_ids_sha256"), "corpus paper ids SHA-256")
    run_statuses = [run.get("status") for run in runs]
    if any(status not in {"success", "partial_failure"} for status in run_statuses):
        raise PublicDevelopmentSummaryError(
            "public current summary rejects paper errors and quality failures"
        )
    succeeded_count = run_statuses.count("success")
    incomplete_count = len(runs) - succeeded_count
    expected_corpus_status = (
        "success"
        if incomplete_count == 0
        else "error"
        if succeeded_count == 0
        else "partial_failure"
    )
    if corpus.get("status") != expected_corpus_status:
        raise PublicDevelopmentSummaryError("corpus status does not match paper statuses")
    if (
        _nonnegative_int(corpus.get("papers_succeeded"), "papers succeeded") != succeeded_count
        or _nonnegative_int(corpus.get("papers_failed"), "papers failed") != incomplete_count
    ):
        raise PublicDevelopmentSummaryError("corpus paper status counts do not match its runs")
    paper_ids: list[str] = []
    candidate_sets: list[list[CandidateObservation]] = []
    lineage_counts: list[Mapping[str, Any]] = []
    observation_ids: set[str] = set()
    for run in runs:
        paper_id = _safe_identifier(run.get("paper_id"), "paper id")
        if paper_id in paper_ids:
            raise PublicDevelopmentSummaryError("paper ids must be unique")
        paper_ids.append(paper_id)
        adjacent_path = run_root / paper_id / "run.json"
        if not adjacent_path.is_file() or adjacent_path.is_symlink():
            raise PublicDevelopmentSummaryError("each paper requires a regular adjacent run.json")
        adjacent = _mapping(read_json(adjacent_path), f"{paper_id}/run.json")
        pipeline_schema = adjacent.get("schema_version")
        if pipeline_schema not in {"pipeline-run/0.2", "pipeline-run/0.3"} or adjacent != run:
            raise PublicDevelopmentSummaryError(
                "adjacent paper run does not match the corpus-run manifest"
            )
        counts = _mapping(run.get("counts"), f"{paper_id}.counts")
        expected_candidates = _nonnegative_int(
            counts.get("candidates"), f"{paper_id}.counts.candidates"
        )
        candidates = _load_candidates(
            run_root / paper_id / "observations.jsonl", expected_candidates
        )
        for candidate in candidates:
            if candidate.paper_id != paper_id:
                raise PublicDevelopmentSummaryError(
                    "candidate paper id does not match its paper run"
                )
            if candidate.observation_id != candidate.stable_id():
                raise PublicDevelopmentSummaryError("candidate observation id is not stable")
            if candidate.observation_id in observation_ids:
                raise PublicDevelopmentSummaryError("candidate observation ids must be unique")
            observation_ids.add(candidate.observation_id)
            attribution = candidate.attribution
            if attribution is None:
                raise PublicDevelopmentSummaryError(
                    "current candidates require deterministic attribution"
                )
            if attribution.schema_version != "attribution-verdict/0.2":
                raise PublicDevelopmentSummaryError("unsupported attribution schema")
            _safe_identifier(attribution.lexicon_id, "attribution lexicon id")
            _sha256(attribution.lexicon_sha256, "attribution lexicon SHA-256")
            if (
                candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}
                and attribution.state is not AttributionState.PAPER_PRODUCED
            ):
                raise PublicDevelopmentSummaryError(
                    "export gate contains a candidate without positive paper-produced origin"
                )
        before_dedup = _nonnegative_int(
            counts.get("candidates_before_deduplication"),
            f"{paper_id}.counts.candidates_before_deduplication",
        )
        duplicates = _nonnegative_int(
            counts.get("duplicates_removed"), f"{paper_id}.counts.duplicates_removed"
        )
        if before_dedup != expected_candidates + duplicates:
            raise PublicDevelopmentSummaryError(
                "pre-deduplication count does not equal candidates plus removals"
            )
        if pipeline_schema == "pipeline-run/0.3":
            binding = _mapping(run.get("candidate_lineage"), "candidate lineage binding")
            lineage_path = run_root / paper_id / "candidate-lineage.json"
            if (
                binding.get("path") != "candidate-lineage.json"
                or not lineage_path.is_file()
                or lineage_path.is_symlink()
                or _sha256(
                    binding.get("artifact_sha256"),
                    "candidate lineage artifact SHA-256",
                )
                != sha256_file(lineage_path)
            ):
                raise PublicDevelopmentSummaryError(
                    "candidate lineage does not match the paper run"
                )
            try:
                lineage = CandidateLineageArtifact.model_validate(read_json(lineage_path))
            except ValueError as error:
                raise PublicDevelopmentSummaryError("candidate lineage is invalid") from error
            reported_lineage_counts = _mapping(binding.get("counts"), "candidate lineage counts")
            if (
                lineage.paper_id != paper_id
                or lineage.observations_sha256
                != sha256_file(run_root / paper_id / "observations.jsonl")
                or reported_lineage_counts != lineage.counts.model_dump(mode="json")
                or lineage.counts.final_candidates != expected_candidates
                or lineage.counts.candidate_occurrences != before_dedup
                or {item.final_observation_id for item in lineage.candidates}
                != {str(candidate.observation_id) for candidate in candidates}
            ):
                raise PublicDevelopmentSummaryError(
                    "candidate lineage does not exactly account the observation ledger"
                )
            lineage_counts.append(reported_lineage_counts)
        candidate_sets.append(candidates)

    if sha256_bytes(canonical_json_bytes(paper_ids)) != paper_ids_sha256:
        raise PublicDevelopmentSummaryError("paper ids do not match the bound corpus order")

    if _nonnegative_int(corpus.get("papers"), "papers") != len(runs):
        raise PublicDevelopmentSummaryError("paper total does not match corpus runs")
    flat_candidates = [candidate for values in candidate_sets for candidate in values]
    counts_by_run = [_mapping(run.get("counts"), "run.counts") for run in runs]
    count_names = (
        "candidates",
        "candidates_before_deduplication",
        "duplicates_removed",
        "candidates_needing_review",
        "semantic_safety_reviews",
        "exported",
        "eee_records",
        "eee_schema_issues",
    )
    totals = {
        name: sum(_nonnegative_int(counts.get(name), f"counts.{name}") for counts in counts_by_run)
        for name in count_names
    }
    corpus_totals = _mapping(corpus.get("totals"), "corpus-run.totals")
    for name, total in totals.items():
        if _nonnegative_int(corpus_totals.get(name), f"corpus totals.{name}") != total:
            raise PublicDevelopmentSummaryError(f"corpus total {name} does not match paper runs")
    if totals["candidates"] != len(flat_candidates):
        raise PublicDevelopmentSummaryError("corpus candidate total does not match observations")

    export_status_counts = Counter(candidate.export_status.value for candidate in flat_candidates)
    text_support_counts = Counter(candidate.text_support.value for candidate in flat_candidates)
    referential_counts = Counter(
        candidate.referential_status.value for candidate in flat_candidates
    )
    attribution_counts = Counter(
        candidate.attribution.state.value
        for candidate in flat_candidates
        if candidate.attribution is not None
    )
    attribution_bindings = [
        {
            "schema_version": candidate.attribution.schema_version,
            "lexicon_id": candidate.attribution.lexicon_id,
            "lexicon_sha256": candidate.attribution.lexicon_sha256,
        }
        for candidate in flat_candidates
        if candidate.attribution is not None
    ]
    attribution_binding = _mapping(
        _shared(attribution_bindings, "attribution binding"), "attribution binding"
    )
    exported = [
        candidate
        for candidate in flat_candidates
        if candidate.export_status is ExportStatus.EXPORTED
    ]
    if len(exported) != totals["exported"]:
        raise PublicDevelopmentSummaryError("exported count does not match observations")
    if export_status_counts[ExportStatus.NEEDS_REVIEW.value] != totals["candidates_needing_review"]:
        raise PublicDevelopmentSummaryError("review count does not match observations")
    semantic_safety_reviews = sum(
        candidate.export_status is ExportStatus.NEEDS_REVIEW
        and any(note.startswith("semantic safety:") for note in candidate.notes)
        for candidate in flat_candidates
    )
    if semantic_safety_reviews != totals["semantic_safety_reviews"]:
        raise PublicDevelopmentSummaryError(
            "semantic-safety review count does not match observations"
        )

    complete_exported = sum(
        bool(candidate.evidence)
        and all(
            anchor.source_id
            and anchor.page >= 1
            and anchor.kind.value
            and isinstance(anchor.quote_sha256, str)
            and _SHA256.fullmatch(anchor.quote_sha256) is not None
            for anchor in candidate.evidence
        )
        for candidate in exported
    )

    row_plans: list[Mapping[str, Any]] = []
    full_row_plans: list[RowEnumerationPlan] = []
    row_outcomes: list[Mapping[str, Any]] = []
    row_executions: list[Mapping[str, Any]] = []
    row_call_counts: list[int] = []
    row_attempt_counts: list[int] = []
    row_bindings: list[dict[str, Any]] = []
    legacy_bindings: list[dict[str, Any]] = []
    code_bindings: list[Mapping[str, Any]] = []
    eee_bindings: list[Mapping[str, Any]] = []
    review_reasons: Counter[str] = Counter()
    for run in runs:
        paper_id = str(run["paper_id"])
        extractor = _mapping(run.get("extractor"), "run.extractor")
        extractor_execution = _mapping(extractor.get("execution"), "extractor execution")
        blocks_total = _nonnegative_int(
            extractor_execution.get("blocks_total"), "extractor blocks total"
        )
        blocks_succeeded = _nonnegative_int(
            extractor_execution.get("blocks_succeeded"), "extractor blocks succeeded"
        )
        blocks_resumed = _nonnegative_int(
            extractor_execution.get("blocks_resumed"), "extractor blocks resumed"
        )
        blocks_failed = _nonnegative_int(
            extractor_execution.get("blocks_failed"), "extractor blocks failed"
        )
        if blocks_succeeded + blocks_resumed + blocks_failed != blocks_total:
            raise PublicDevelopmentSummaryError("extractor block accounting is inconsistent")
        if blocks_failed:
            raise PublicDevelopmentSummaryError(
                "public current summary rejects failed legacy extraction blocks"
            )
        row_stage = _mapping(run.get("row_enumeration"), "run.row_enumeration")
        if row_stage.get("enabled") is not True:
            raise PublicDevelopmentSummaryError("public current summary requires the row stage")
        legacy_bindings.append(_stage_binding(extractor, "run.extractor"))
        row_binding = _stage_binding(row_stage, "run.row_enumeration")
        try:
            row_limits = RowEnumerationConfig.model_validate(row_stage.get("limits"))
        except ValueError as error:
            raise PublicDevelopmentSummaryError("row limits are invalid") from error
        row_binding["limits"] = row_limits.model_dump(mode="json")
        row_bindings.append(row_binding)
        row_plan_summary = _mapping(row_stage.get("plan"), "row plan")
        row_outcome = _mapping(row_stage.get("outcome"), "row outcome")
        row_calls = _sequence(row_stage.get("calls"), "row calls")
        row_attempts = _sequence(row_stage.get("attempts"), "row attempts")
        row_execution = _mapping(row_stage.get("execution"), "row execution")
        plan_sha256 = _sha256(row_stage.get("plan_sha256"), "row plan SHA-256")
        plan_path = run_root / paper_id / "private" / "row-enumeration-plan.json"
        if not plan_path.is_file() or plan_path.is_symlink():
            raise PublicDevelopmentSummaryError("each paper requires a regular private row plan")
        if sha256_file(plan_path) != plan_sha256:
            raise PublicDevelopmentSummaryError("private row plan hash does not match run.json")
        try:
            full_plan = RowEnumerationPlan.model_validate(read_json(plan_path))
        except ValueError as error:
            raise PublicDevelopmentSummaryError("private row plan is invalid") from error
        if full_plan.config != row_limits:
            raise PublicDevelopmentSummaryError(
                "reported row limits do not match the hash-bound private plan"
            )
        if full_plan.telemetry.model_dump(mode="json") != row_plan_summary:
            raise PublicDevelopmentSummaryError("row plan telemetry does not match private plan")
        outcome_path = run_root / paper_id / "private" / "row-enumeration.json"
        if not outcome_path.is_file() or outcome_path.is_symlink():
            raise PublicDevelopmentSummaryError("each paper requires a regular private row outcome")
        private_outcome = _mapping(read_json(outcome_path), "private row outcome")
        terminal_binding = _mapping(row_stage.get("terminal_states"), "row terminal binding")
        terminal_path = run_root / paper_id / "private" / "row-terminal-states.json"
        if not terminal_path.is_file() or terminal_path.is_symlink():
            raise PublicDevelopmentSummaryError(
                "each paper requires a regular private row terminal ledger"
            )
        terminal_sha256 = _sha256(
            terminal_binding.get("artifact_sha256"),
            "row terminal ledger SHA-256",
        )
        if (
            private_outcome.get("schema_version") != "row-enumeration-outcome/0.3"
            or private_outcome.get("plan_sha256") != plan_sha256
            or private_outcome.get("terminal_states_sha256") != terminal_sha256
            or private_outcome.get("telemetry") != row_outcome
            or private_outcome.get("calls") != row_calls
            or private_outcome.get("attempts") != row_attempts
            or sha256_file(terminal_path) != terminal_sha256
        ):
            raise PublicDevelopmentSummaryError("private row outcome does not match the paper run")
        try:
            terminal_ledger = RowTerminalLedger.model_validate(read_json(terminal_path))
            typed_calls = [ProviderCall.model_validate(item) for item in row_calls]
            validate_terminal_ledger_against(
                full_plan,
                terminal_ledger,
                paper_id=paper_id,
                model=str(row_stage.get("model")),
                retained_calls=typed_calls,
            )
        except ValueError as error:
            raise PublicDevelopmentSummaryError("private row terminal ledger is invalid") from error
        terminal_counts = terminal_ledger.counts
        if terminal_binding.get("counts") != terminal_counts.model_dump(mode="json"):
            raise PublicDevelopmentSummaryError(
                "row terminal counts do not match the hash-bound private ledger"
            )
        unresolved_ids = [
            row_id
            for row_id, terminal in terminal_ledger.terminal_by_row.items()
            if terminal.state is RowTerminalState.UNRESOLVED
        ]
        unbatchable_ids = [
            row_id
            for row_id, terminal in terminal_ledger.terminal_by_row.items()
            if terminal.state is RowTerminalState.UNSUPPORTED
        ]
        actual_dispositions = {
            "result": terminal_counts.result,
            "not_result": terminal_counts.not_result,
            "uncertain": terminal_counts.uncertain,
        }
        reported_dispositions = _mapping(row_outcome.get("dispositions"), "row dispositions")
        if (
            set(reported_dispositions) != _ROW_DISPOSITIONS
            or {
                name: _nonnegative_int(value, f"row disposition {name}")
                for name, value in reported_dispositions.items()
            }
            != actual_dispositions
        ):
            raise PublicDevelopmentSummaryError(
                "row disposition telemetry does not match private terminal records"
            )
        if (
            _nonnegative_int(row_outcome.get("rows_resolved"), "rows resolved")
            != terminal_counts.result + terminal_counts.not_result + terminal_counts.uncertain
            or _nonnegative_int(row_outcome.get("rows_unresolved"), "rows unresolved")
            != terminal_counts.unresolved
            or _nonnegative_int(row_outcome.get("rows_unbatchable"), "rows unbatchable")
            != terminal_counts.unsupported
            or _nonnegative_int(row_execution.get("unknown_row_ids_seen"), "unknown row ids seen")
            != len(terminal_ledger.protocol_events)
            or _nonnegative_int(row_execution.get("invalid_rows_seen"), "invalid rows seen")
            != len(terminal_ledger.invalid_row_reasons)
        ):
            raise PublicDevelopmentSummaryError(
                "row telemetry does not match the private row terminal ledger"
            )
        row_incomplete = bool(unresolved_ids or unbatchable_ids)
        expected_run_status = "partial_failure" if row_incomplete else "success"
        if run.get("status") != expected_run_status:
            raise PublicDevelopmentSummaryError(
                "paper status does not match its bounded row-stage completion"
            )
        row_plans.append(row_plan_summary)
        full_row_plans.append(full_plan)
        row_outcomes.append(row_outcome)
        row_executions.append(row_execution)
        row_call_counts.append(len(row_calls))
        row_attempt_counts.append(len(row_attempts))
        code_bindings.append(_mapping(run.get("code"), "run.code"))
        eee_bindings.append(_mapping(run.get("eee_schema"), "run.eee_schema"))
        verifier = _mapping(run.get("verifier"), "run.verifier")
        if verifier.get("enabled") is not False or _sequence(
            verifier.get("calls"), "verifier.calls"
        ):
            raise PublicDevelopmentSummaryError(
                "public current summary requires the independent verifier to be disabled"
            )
        review = _mapping(run.get("review_state"), "run.review_state")
        reasons = _unique_strings(review.get("reasons"), "review reasons")
        reason_set = set(reasons)
        if ("row_enumeration_unbatchable" in reason_set) != bool(unbatchable_ids) or (
            "row_enumeration_unresolved" in reason_set
        ) != bool(unresolved_ids):
            raise PublicDevelopmentSummaryError(
                "row-stage review reasons do not match the private row-id ledger"
            )
        for reason in reasons:
            if reason not in _REVIEW_REASONS:
                raise PublicDevelopmentSummaryError("run contains an unknown review reason")
            review_reasons[reason] += 1

    legacy_binding = _shared(legacy_bindings, "extractor contract")
    row_binding = _shared(row_bindings, "row extractor contract")
    code_binding = _mapping(_shared(code_bindings, "code binding"), "code binding")
    eee_binding = _mapping(_shared(eee_bindings, "EEE schema binding"), "EEE binding")
    git_commit = code_binding.get("git_commit")
    if git_commit is not None and (
        not isinstance(git_commit, str) or _GIT_COMMIT.fullmatch(git_commit) is None
    ):
        raise PublicDevelopmentSummaryError("git commit is not a lowercase commit hash")
    if not isinstance(code_binding.get("git_dirty"), bool):
        raise PublicDevelopmentSummaryError("git_dirty must be boolean")
    eee_schema_sha256 = _sha256(eee_binding.get("sha256"), "EEE schema SHA-256")
    if eee_binding.get("version") != "0.2.2":
        raise PublicDevelopmentSummaryError("public summary requires EEE schema 0.2.2")
    if totals["eee_schema_issues"] != 0:
        raise PublicDevelopmentSummaryError("successful public run has EEE schema issues")
    _validate_eee_outputs(
        run_root,
        runs,
        {candidate.observation_id for candidate in exported if candidate.observation_id},
        totals["eee_records"],
        eee_schema_sha256,
    )
    row_dispositions: Counter[str] = Counter()
    for full_plan, outcome, execution, call_count, attempt_count in zip(
        full_row_plans,
        row_outcomes,
        row_executions,
        row_call_counts,
        row_attempt_counts,
        strict=True,
    ):
        dispositions = _mapping(outcome.get("dispositions"), "row dispositions")
        if set(dispositions) != _ROW_DISPOSITIONS:
            raise PublicDevelopmentSummaryError("row dispositions must use the exact enum labels")
        for name, value in dispositions.items():
            row_dispositions[name] += _nonnegative_int(value, f"row disposition {name}")
        plan_telemetry = full_plan.telemetry
        if _nonnegative_int(outcome.get("rows_unbatchable"), "rows unbatchable") != (
            plan_telemetry.unbatchable_rows
        ):
            raise PublicDevelopmentSummaryError("row plan and outcome disagree on unbatchable rows")
        batches_total = _nonnegative_int(
            execution.get("batches_total"), "row execution.batches_total"
        )
        batches_resumed = _nonnegative_int(
            execution.get("batches_resumed"), "row execution.batches_resumed"
        )
        batches_executed = _nonnegative_int(
            execution.get("batches_executed"), "row execution.batches_executed"
        )
        if (
            batches_total != plan_telemetry.base_batches
            or batches_resumed + batches_executed != batches_total
        ):
            raise PublicDevelopmentSummaryError("row batch execution does not match the plan")
        if _nonnegative_int(outcome.get("calls"), "row outcome.calls") != call_count:
            raise PublicDevelopmentSummaryError("row call count does not match retained calls")
        if _nonnegative_int(outcome.get("attempts"), "row outcome.attempts") != attempt_count:
            raise PublicDevelopmentSummaryError(
                "row attempt count does not match retained attempts"
            )
        if call_count > attempt_count or attempt_count > plan_telemetry.maximum_calls:
            raise PublicDevelopmentSummaryError("row calls or attempts exceed the bounded plan")

    rows_planned = sum(
        _nonnegative_int(item.get("rows_planned"), "rows planned") for item in row_plans
    )
    rows_resolved = sum(
        _nonnegative_int(item.get("rows_resolved"), "rows resolved") for item in row_outcomes
    )
    rows_unresolved = sum(
        _nonnegative_int(item.get("rows_unresolved"), "rows unresolved") for item in row_outcomes
    )
    rows_unbatchable = sum(
        _nonnegative_int(item.get("rows_unbatchable"), "rows unbatchable") for item in row_outcomes
    )
    unknown_rows = sum(
        _nonnegative_int(item.get("unknown_row_ids_seen"), "unknown row ids")
        for item in row_executions
    )
    invalid_rows = sum(
        _nonnegative_int(item.get("invalid_rows_seen"), "invalid rows") for item in row_executions
    )
    if sum(row_dispositions.values()) != rows_resolved:
        raise PublicDevelopmentSummaryError("row dispositions do not match resolved rows")
    if rows_resolved + rows_unresolved + rows_unbatchable != rows_planned:
        raise PublicDevelopmentSummaryError("row outcomes do not account for every planned row")

    reference = corpus.get("reference_evaluation")
    reference_summary: dict[str, Any]
    if isinstance(reference, Mapping):
        detection = _mapping(reference.get("detection"), "reference detection")
        bases = _mapping(reference.get("bases"), "reference bases")
        reference_observations = _nonnegative_int(
            bases.get("reference_observations"), "reference observations"
        )
        true_positives = _nonnegative_int(
            detection.get("true_positives"), "reference true positives"
        )
        false_negatives = _nonnegative_int(
            detection.get("false_negatives"), "reference false negatives"
        )
        if true_positives + false_negatives != reference_observations:
            raise PublicDevelopmentSummaryError(
                "reference recall basis does not equal true positives plus false negatives"
            )
        recomputed_recall = (
            true_positives / reference_observations if reference_observations else None
        )
        reported_recall = _optional_rate(detection.get("recall"), "reference recall")
        if (
            recomputed_recall is None
            and reported_recall is not None
            or recomputed_recall is not None
            and (
                reported_recall is None
                or not math.isclose(recomputed_recall, reported_recall, abs_tol=5e-7)
            )
        ):
            raise PublicDevelopmentSummaryError("reported reference recall does not match counts")
        reference_summary = {
            "status": "candidate_layer_annotated_reference_recall_measured",
            "reference_observations": reference_observations,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "micro_recall": (
                round(recomputed_recall, 6) if recomputed_recall is not None else None
            ),
            "macro_recall": _optional_rate(detection.get("macro_recall"), "reference macro recall"),
            "coverage_statement": (
                "Candidate-layer recall over pre-existing annotated reference observations in "
                "the open development corpus; not canonical-EEE recall, whole-paper recall, "
                "holdout evidence, or generalization evidence."
            ),
            "precision": {
                "status": "not_measured",
                "computed_slice_diagnostic_omitted": detection.get("precision") is not None,
                "reason": (
                    "The available annotation frame does not establish current whole-pipeline "
                    "precision or non-result-row specificity."
                ),
            },
            "generalization_evidence": False,
        }
    else:
        reference_summary = {
            "status": "not_available",
            "precision": {"status": "not_measured"},
            "generalization_evidence": False,
        }

    duplicate_basis = totals["candidates_before_deduplication"]
    operations = _mapping(corpus.get("operations"), "corpus operations")
    summary = {
        "schema_version": PUBLIC_DEVELOPMENT_SUMMARY_SCHEMA_VERSION,
        "projection_mode": "historical_pipeline_0.2",
        "current_sealed_receipt": False,
        "statement": (
            "Development-only aggregate for an evidence-first research prototype; human review "
            "is required and this is not validation for unattended extraction."
        ),
        "run_binding": {
            "run_id": run_id,
            "corpus_id": corpus_id,
            "corpus_spec_sha256": corpus_spec_sha256,
            "paper_ids_sha256": paper_ids_sha256,
            "corpus_run_sha256": sha256_file(corpus_path),
            "generated_at": _generated_at(corpus.get("generated_at")),
            "code": {
                "git_commit": code_binding.get("git_commit"),
                "git_dirty": code_binding.get("git_dirty"),
                "source_tree_sha256": _sha256(
                    code_binding.get("source_tree_sha256"), "source tree SHA-256"
                ),
            },
            "extractor": legacy_binding,
            "row_extractor": row_binding,
            "eee_schema": {
                "version": eee_binding.get("version"),
                "sha256": eee_schema_sha256,
            },
            "attribution": attribution_binding,
        },
        "scope": {
            "split": "development",
            "papers": len(runs),
            "holdout_included": False,
            "private_human_annotations_included": False,
            "independent_human_validation": False,
        },
        "technical_health": {
            "status": corpus.get("status"),
            "run_completeness": (
                "complete" if incomplete_count == 0 else "bounded_row_stage_incomplete"
            ),
            "papers_succeeded": _nonnegative_int(
                corpus.get("papers_succeeded"), "papers succeeded"
            ),
            "papers_failed": _nonnegative_int(corpus.get("papers_failed"), "papers failed"),
            "papers_needing_review": _nonnegative_int(
                corpus.get("papers_needing_review"), "papers needing review"
            ),
            "wall_clock_seconds": _optional_nonnegative_number(
                operations.get("wall_clock_seconds"), "wall clock seconds"
            ),
            "review_reason_counts": dict(sorted(review_reasons.items())),
        },
        "row_enumeration": {
            "tables_considered": sum(
                _nonnegative_int(item.get("tables_considered"), "tables considered")
                for item in row_plans
            ),
            "dense_tables": sum(
                _nonnegative_int(item.get("dense_tables"), "dense tables") for item in row_plans
            ),
            "rows_planned": rows_planned,
            "rows_resolved": rows_resolved,
            "rows_unresolved": rows_unresolved,
            "rows_unbatchable": rows_unbatchable,
            "unknown_row_ids_seen": unknown_rows,
            "invalid_rows_seen": invalid_rows,
            "dispositions": dict(sorted(row_dispositions.items())),
            "all_rows_accounted_for": True,
            "all_planned_rows_partitioned": True,
            "all_batchable_rows_resolved": rows_unresolved == 0,
            "complete_extraction": rows_unresolved == 0 and rows_unbatchable == 0,
            "no_unknown_or_invalid_rows_seen": unknown_rows == 0 and invalid_rows == 0,
        },
        "outputs": {
            **totals,
            "candidate_proposal_removal_rate": (
                totals["duplicates_removed"] / duplicate_basis if duplicate_basis else None
            ),
            "export_status_counts": dict(sorted(export_status_counts.items())),
            "text_support_status_counts": dict(sorted(text_support_counts.items())),
            "referential_status_counts": dict(sorted(referential_counts.items())),
            "attribution_state_counts": dict(sorted(attribution_counts.items())),
        },
        "canonical_eee": {
            "status": "produced" if totals["eee_records"] else "empty",
            "records": totals["eee_records"],
            "schema_issues": totals["eee_schema_issues"],
            "positive_paper_produced_origin_required": True,
            "safe_empty_output_is_valid": True,
        },
        "numeric_export_provenance": {
            "status": "measured" if exported else "not_applicable_empty_export",
            "exported_observations": len(exported),
            "complete_observations": complete_exported,
            "all_complete": complete_exported == len(exported) if exported else None,
            "evidence_quotations_included": False,
        },
        "provider_usage_recorded": _recorded_call_telemetry(runs),
        "reference_evaluation": reference_summary,
        "annotation_status": {
            "single_annotator_aggregate_included": False,
            "inter_annotator_agreement_available": False,
            "adjudication_available": False,
        },
        "limitations": [
            "This is an open-development result, not holdout or generalization evidence.",
            (
                "Current precision, full-tuple correctness, and non-result-row specificity "
                "remain unmeasured."
            ),
            "Canonical EEE can be empty when positive paper-produced origin is not established.",
            (
                "Provider usage is artifact-basis telemetry and totals are lower bounds when "
                "failed or superseded attempt telemetry is unavailable."
            ),
            "Private human responses and individual labels are not included.",
        ],
        "privacy": {
            "evidence_quotations_included": False,
            "paper_level_rows_or_labels_included": False,
            "provider_traces_included": False,
            "request_identifiers_included": False,
            "credentials_included": False,
            "local_paths_included": False,
            "private_annotations_included": False,
        },
    }
    if lineage_counts:
        summary["outputs"]["candidate_lineage"] = {
            "runs_bound": len(lineage_counts),
            "all_current_schema_runs_bound": len(lineage_counts)
            == sum(run.get("schema_version") == "pipeline-run/0.3" for run in runs),
            "proposals": sum(
                _nonnegative_int(item.get("proposals"), "lineage proposals")
                for item in lineage_counts
            ),
            "candidate_occurrences": sum(
                _nonnegative_int(
                    item.get("candidate_occurrences"),
                    "lineage candidate occurrences",
                )
                for item in lineage_counts
            ),
            "final_candidates": sum(
                _nonnegative_int(
                    item.get("final_candidates"),
                    "lineage final candidates",
                )
                for item in lineage_counts
            ),
            "merged_candidates": sum(
                _nonnegative_int(
                    item.get("merged_candidates"),
                    "lineage merged candidates",
                )
                for item in lineage_counts
            ),
        }
    try:
        _assert_public_payload(summary, "public development summary")
    except ExtractionReviewCardError as error:
        raise PublicDevelopmentSummaryError(
            "projected summary failed the public-payload audit"
        ) from error
    return summary


def _counter_sum(receipt: ValidatedCorpusReceipt, field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for paper in receipt.papers:
        counter.update(getattr(paper, field))
    return dict(sorted(counter.items()))


def _row_summary(receipt: ValidatedCorpusReceipt) -> dict[str, Any]:
    dispositions: Counter[str] = Counter()
    names = (
        "tables_considered",
        "dense_tables",
        "rows_planned",
        "rows_resolved",
        "rows_unresolved",
        "rows_unbatchable",
        "unknown_row_ids_seen",
        "invalid_rows_seen",
    )
    values = {name: sum(int(paper.row_metrics[name]) for paper in receipt.papers) for name in names}
    for paper in receipt.papers:
        dispositions.update(paper.row_metrics["dispositions"])
    return {
        **values,
        "dispositions": dict(sorted(dispositions.items())),
        "all_rows_accounted_for": True,
        "all_planned_rows_partitioned": True,
        "all_batchable_rows_resolved": values["rows_unresolved"] == 0,
        "complete_extraction": (values["rows_unresolved"] == 0 and values["rows_unbatchable"] == 0),
        "no_unknown_or_invalid_rows_seen": (
            values["unknown_row_ids_seen"] == 0 and values["invalid_rows_seen"] == 0
        ),
    }


def _reviewed_derived_projection(
    receipt: ValidatedCorpusReceipt,
    source_run_root: Path,
    review_root: Path,
    reviewed_derived_root: Path,
) -> dict[str, Any]:
    try:
        contextual = verify_contextual_derived_run(
            reviewed_derived_root.resolve(),
            run_root=source_run_root.resolve(),
            review_root=review_root.resolve(),
            schema_sha256=str(receipt.eee_schema["sha256"]),
        )
        verification = contextual.verification
        manifest = DerivedRunManifest.model_validate(
            read_json(reviewed_derived_root.resolve() / DERIVED_MANIFEST_NAME)
        )
    except (OSError, ValueError, ReviewedExportError) as error:
        raise PublicDevelopmentSummaryError(
            "reviewed derived result failed contextual verification"
        ) from error
    if (
        manifest.source_run_name != receipt.seal.source_run_name
        or manifest.source_run_seal_sha256 != receipt.seal.seal_sha256
        or manifest.source_run_tree_sha256 != receipt.seal.tree_sha256
        or manifest.eee_schema_version != receipt.eee_schema["version"]
        or manifest.eee_schema_sha256 != receipt.eee_schema["sha256"]
    ):
        raise PublicDevelopmentSummaryError(
            "reviewed derived result belongs to another sealed source run"
        )
    source_tuple_sidecars = {paper.tuple_sidecar_sha256 for paper in receipt.papers}
    source_verifier_sidecars = {
        paper.verifier_sidecar_sha256
        for paper in receipt.papers
        if paper.verifier_sidecar_sha256 is not None
    }
    if not set(manifest.tuple_sidecar_sha256s).issubset(source_tuple_sidecars) or not set(
        manifest.verifier_sidecar_sha256s
    ).issubset(source_verifier_sidecars):
        raise PublicDevelopmentSummaryError(
            "reviewed derived result cites a gate outside the sealed source run"
        )
    modes = [item.value for item in manifest.export_provenance_modes]
    mode_counts = empty_source_provenance_counts()
    mode_counts.update(contextual.provenance_mode_counts)
    return {
        "status": "contextually_verified",
        "locked_review_context_verified": True,
        "independence_status": "not_measured",
        "exported_observation_authority_mode_counts": dict(
            sorted(contextual.authority_mode_counts.items())
        ),
        "source_run_binding": {
            "seal_sha256": receipt.seal.seal_sha256,
            "tree_sha256": receipt.seal.tree_sha256,
        },
        "derived_run_sha256": verification.derived_run_sha256,
        "eee_records": manifest.counts["eee_records"],
        "eee_observations": manifest.counts["eee_observations"],
        "outcomes_exported": manifest.counts["outcomes_exported"],
        "outcomes_withheld": manifest.counts["outcomes_withheld"],
        "outcomes_failed": manifest.counts["outcomes_failed"],
        "provenance_modes_present": modes,
        "provenance_mode_counts": mode_counts,
        "provenance_mode_count_status": "measured",
        "source_and_reviewed_counts_combined": False,
    }


def _build_current_public_development_summary(
    receipt: ValidatedCorpusReceipt,
    *,
    source_run_root: Path,
    review_root: Path | None,
    reviewed_derived_root: Path | None,
) -> dict[str, Any]:
    stages = aggregate_stage_receipts(receipt)
    stage_statuses = {stage: item["status"] for stage, item in stages.items()}
    five_stage_status = (
        "validated"
        if set(stage_statuses.values()) == {"validated"}
        else "partial_failure"
        if "partial_failure" in stage_statuses.values()
        else "not_run"
    )
    counts = dict(receipt.totals)
    proposals = sum(paper.lineage_counts["proposals"] for paper in receipt.papers)
    occurrences = sum(paper.lineage_counts["candidate_occurrences"] for paper in receipt.papers)
    final_candidates = sum(paper.lineage_counts["final_candidates"] for paper in receipt.papers)
    merged_candidates = sum(paper.lineage_counts["merged_candidates"] for paper in receipt.papers)
    review_reasons = Counter(reason for paper in receipt.papers for reason in paper.review_reasons)
    papers_succeeded = sum(paper.status == "success" for paper in receipt.papers)
    papers_failed = len(receipt.papers) - papers_succeeded
    source_modes = {
        "status": "measured",
        "eee_records": 0,
        "eee_observations": 0,
        "provenance_mode_counts": empty_source_provenance_counts(),
    }
    reviewed = (
        _reviewed_derived_projection(
            receipt,
            source_run_root,
            review_root,
            reviewed_derived_root,
        )
        if reviewed_derived_root is not None and review_root is not None
        else {
            "status": "not_supplied",
            "locked_review_context_verified": False,
            "independence_status": "not_measured",
            "exported_observation_authority_mode_counts": None,
            "eee_records": None,
            "eee_observations": None,
            "provenance_mode_counts": None,
            "source_and_reviewed_counts_combined": False,
        }
    )
    summary = {
        "schema_version": PUBLIC_DEVELOPMENT_SUMMARY_SCHEMA_VERSION,
        "projection_mode": "sealed_current_five_stage_receipt",
        "current_sealed_receipt": True,
        "statement": (
            "Development-only aggregate for an evidence-first research prototype; source-run "
            "and reviewed-derived results are verified and reported separately."
        ),
        "run_binding": {
            "run_id": _safe_identifier(receipt.seal.source_run_name, "source run name"),
            "corpus_id": _safe_identifier(receipt.corpus_id, "corpus id"),
            "recorded_corpus_spec_sha256": receipt.corpus_spec_sha256,
            "corpus_spec_binding_status": "recorded_hash_not_rederived",
            "paper_ids_sha256": receipt.paper_ids_sha256,
            "corpus_run_sha256": receipt.corpus_run_sha256,
            "generated_at": _generated_at(receipt.generated_at),
            "run_seal": {
                "schema_version": receipt.seal.schema_version,
                "seal_sha256": receipt.seal.seal_sha256,
                "tree_sha256": receipt.seal.tree_sha256,
                "file_count": receipt.seal.file_count,
            },
            "code": dict(receipt.code),
            "extractor": dict(receipt.extractor_binding),
            "row_extractor": dict(receipt.row_binding),
            "candidate_validation": dict(receipt.candidate_validation),
            "eee_schema": dict(receipt.eee_schema),
            "stage_chain": {
                "mode": (
                    "current_five_stage_contract"
                    if five_stage_status == "validated"
                    else "current_tuple_review_contract"
                ),
                "five_stage_gate_status": five_stage_status,
                "missing_gate_treated_as_passed": False,
                "checkpoint_validation": stages,
            },
        },
        "scope": {
            "split": "development",
            "papers": len(receipt.papers),
            "holdout_included": False,
            "private_human_annotations_included": False,
            "independent_human_validation": False,
        },
        "technical_health": {
            "status": receipt.corpus_status,
            "run_completeness": (
                "complete" if five_stage_status == "validated" else "typed_stage_incomplete"
            ),
            "release_ready": False,
            "papers_succeeded": papers_succeeded,
            "papers_failed": papers_failed,
            "papers_needing_review": sum(paper.needs_review for paper in receipt.papers),
            "wall_clock_seconds": receipt.operations.get("wall_clock_seconds"),
            "review_reason_counts": dict(sorted(review_reasons.items())),
        },
        "row_enumeration": _row_summary(receipt),
        "outputs": {
            **counts,
            "count_projection_basis": "checkpoint_replayed_allowlist",
            "candidate_proposal_removal_rate": (
                counts["duplicates_removed"] / counts["candidates_before_deduplication"]
                if counts["candidates_before_deduplication"]
                else None
            ),
            "export_status_counts": _counter_sum(receipt, "export_status_counts"),
            "text_support_status_counts": _counter_sum(receipt, "text_support_counts"),
            "referential_status_counts": _counter_sum(receipt, "referential_status_counts"),
            "attribution_state_counts": _counter_sum(receipt, "attribution_state_counts"),
            "candidate_lineage": {
                "runs_bound": len(receipt.papers),
                "all_current_schema_runs_bound": True,
                "proposals": proposals,
                "candidate_occurrences": occurrences,
                "final_candidates": final_candidates,
                "merged_candidates": merged_candidates,
            },
        },
        "canonical_eee": {
            "status": "empty",
            "records": 0,
            "schema_issues": 0,
            "positive_paper_produced_origin_required": True,
            "automatic_positive_origin_enabled": False,
            "safe_empty_output_is_valid": True,
        },
        "numeric_export_provenance": {
            "status": "not_applicable_empty_source_export",
            "exported_observations": 0,
            "complete_observations": 0,
            "all_complete": None,
            "evidence_quotations_included": False,
        },
        "provider_usage_recorded": aggregate_provider_telemetry(receipt),
        "reference_evaluation": {
            "status": "not_projected_from_current_receipt",
            "precision": {"status": "not_measured"},
            "generalization_evidence": False,
        },
        "export_provenance_modes": {
            "source_run": source_modes,
            "reviewed_derived": reviewed,
            "counts_combined": False,
        },
        "annotation_status": {
            "source_run_human_annotations_included": False,
            "reviewed_derived_supplied": reviewed["status"] == "contextually_verified",
            "reviewed_derived_authority_mode_counts": reviewed[
                "exported_observation_authority_mode_counts"
            ],
            "inter_annotator_agreement_available": False,
        },
        "limitations": [
            "This is open-development evidence, not holdout or generalization evidence.",
            "Automatic positive producer-origin authority remains disabled.",
            "Source-run and human-reviewed derived outcomes are never combined.",
            "Private excerpts, labels, proposals, calls, and identifiers are not projected.",
        ],
        "privacy": {
            "evidence_quotations_included": False,
            "paper_level_rows_or_labels_included": False,
            "provider_traces_included": False,
            "request_identifiers_included": False,
            "credentials_included": False,
            "local_paths_included": False,
            "private_annotations_included": False,
        },
    }
    try:
        _assert_public_payload(summary, "public development summary")
    except ExtractionReviewCardError as error:
        raise PublicDevelopmentSummaryError(
            "projected summary failed the public-payload audit"
        ) from error
    return summary


def build_public_development_summary(
    run_root: Path,
    *,
    reviewed_derived_root: Path | None = None,
    review_root: Path | None = None,
) -> dict[str, Any]:
    """Build a quote-free aggregate from a historical or sealed current run."""

    root = run_root.resolve()
    corpus = _mapping(_regular_summary_json(root / "corpus-run.json"), "corpus run")
    runs = [_mapping(item, "corpus run item") for item in _sequence(corpus.get("runs"), "runs")]
    schemas = {run.get("schema_version") for run in runs}
    if (reviewed_derived_root is None) != (review_root is None):
        raise PublicDevelopmentSummaryError(
            "reviewed derived and private review roots must be supplied together"
        )
    if schemas == {"pipeline-run/0.2"}:
        if reviewed_derived_root is not None or review_root is not None:
            raise PublicDevelopmentSummaryError(
                "reviewed derived joins require a sealed current pipeline run"
            )
        return _build_legacy_public_development_summary(root)
    if schemas != {"pipeline-run/0.4"}:
        raise PublicDevelopmentSummaryError("corpus mixes unsupported pipeline schemas")
    try:
        receipt = validate_completed_corpus_run(root)
    except CompletedCorpusValidationError as error:
        raise PublicDevelopmentSummaryError(str(error)) from error
    return _build_current_public_development_summary(
        receipt,
        source_run_root=root,
        review_root=review_root,
        reviewed_derived_root=reviewed_derived_root,
    )


def _regular_summary_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise PublicDevelopmentSummaryError("run root requires a regular corpus-run.json")
    try:
        return read_json(path)
    except (OSError, ValueError) as error:
        raise PublicDevelopmentSummaryError("corpus-run.json is invalid") from error


def write_public_development_summary(
    run_root: Path,
    output_path: Path,
    *,
    reviewed_derived_root: Path | None = None,
    review_root: Path | None = None,
) -> str:
    """Build and atomically write one aggregate outside the private run tree."""

    run_root = run_root.resolve()
    output_path = output_path.resolve()
    input_roots = [(run_root, "run")]
    if reviewed_derived_root is not None:
        input_roots.append((reviewed_derived_root.resolve(), "reviewed derived"))
    if review_root is not None:
        input_roots.append((review_root.resolve(), "private review"))
    for input_root, label in input_roots:
        try:
            output_path.relative_to(input_root)
        except ValueError:
            continue
        raise PublicDevelopmentSummaryError(
            f"public summary output must be outside the {label} root"
        )
    return write_json(
        output_path,
        build_public_development_summary(
            run_root,
            reviewed_derived_root=reviewed_derived_root,
            review_root=review_root,
        ),
    )


__all__ = [
    "PUBLIC_DEVELOPMENT_SUMMARY_SCHEMA_VERSION",
    "PublicDevelopmentSummaryError",
    "build_public_development_summary",
    "write_public_development_summary",
]
