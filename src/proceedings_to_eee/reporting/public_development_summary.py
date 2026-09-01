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
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ExportStatus
from proceedings_to_eee.extraction.row_enumeration import (
    RowDisposition,
    RowDispositionRecord,
    RowEnumerationConfig,
    RowEnumerationPlan,
    UnbatchableReason,
    make_row_batch,
)
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from proceedings_to_eee.reporting.extraction_review_cards import (
    ExtractionReviewCardError,
    _assert_public_payload,
)
from proceedings_to_eee.resources import DEFAULT_EEE_SCHEMA_PATH
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record

PUBLIC_DEVELOPMENT_SUMMARY_SCHEMA_VERSION = "public-development-summary/0.1"
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
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
    if not isinstance(model, str) or not model:
        raise PublicDevelopmentSummaryError(f"{context}.model must be a non-empty string")
    return {
        "provider": _safe_identifier(provider, f"{context}.provider"),
        "model": model,
        "temperature": stage.get("temperature"),
        "reasoning_effort": stage.get("reasoning_effort"),
        "max_tokens": _nonnegative_int(stage.get("max_tokens"), f"{context}.max_tokens"),
        "seed": stage.get("seed"),
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


def build_public_development_summary(run_root: Path) -> dict[str, Any]:
    """Build a path-free aggregate from one completed development run. Offline."""

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
        if adjacent.get("schema_version") != "pipeline-run/0.2" or adjacent != run:
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
        planned_ids = [row.row_id for row in full_plan.rows]
        if len(planned_ids) != len(set(planned_ids)):
            raise PublicDevelopmentSummaryError("private row plan contains duplicate row ids")
        planned_id_set = set(planned_ids)
        plan_unbatchable_ids = [item.row_id for item in full_plan.unbatchable_rows]
        if len(plan_unbatchable_ids) != len(set(plan_unbatchable_ids)):
            raise PublicDevelopmentSummaryError(
                "private row plan contains duplicate unbatchable row ids"
            )
        plan_unbatchable_set = set(plan_unbatchable_ids)
        batch_ids = [row.row_id for batch in full_plan.batches for row in batch.rows]
        if len(batch_ids) != len(set(batch_ids)):
            raise PublicDevelopmentSummaryError("private row plan batches repeat row ids")
        if (
            plan_unbatchable_set - planned_id_set
            or set(batch_ids) != planned_id_set - plan_unbatchable_set
            or full_plan.telemetry.rows_planned != len(planned_ids)
            or full_plan.telemetry.unbatchable_rows != len(plan_unbatchable_ids)
            or full_plan.telemetry.base_batches != len(full_plan.batches)
            or full_plan.telemetry.expected_calls != len(full_plan.batches)
            or full_plan.telemetry.maximum_calls
            != len(full_plan.batches) * (1 + 2 * row_limits.max_recovery_depth)
        ):
            raise PublicDevelopmentSummaryError("private row plan does not partition planned rows")
        planned_by_id = {row.row_id: row for row in full_plan.rows}
        for batch in full_plan.batches:
            if (
                make_row_batch(batch.rows) != batch
                or any(planned_by_id.get(row.row_id) != row for row in batch.rows)
                or len(batch.rows) > row_limits.max_rows_per_batch
                or batch.character_count > row_limits.max_characters_per_batch
                or batch.value_token_count > row_limits.max_value_tokens_per_batch
            ):
                raise PublicDevelopmentSummaryError(
                    "private row batch does not match its configured hard limits"
                )
        for item in full_plan.unbatchable_rows:
            row = planned_by_id[item.row_id]
            singleton = make_row_batch([row])
            reasons: list[UnbatchableReason] = []
            if singleton.character_count > row_limits.max_characters_per_batch:
                reasons.append(UnbatchableReason.CHARACTER_LIMIT)
            if singleton.value_token_count > row_limits.max_value_tokens_per_batch:
                reasons.append(UnbatchableReason.VALUE_TOKEN_LIMIT)
            if (
                item.input_sha256 != row.input_sha256
                or item.source_id != row.source_id
                or item.page != row.page
                or item.region_id != row.region_id
                or item.character_count != singleton.character_count
                or item.value_token_count != singleton.value_token_count
                or item.max_characters_per_batch != row_limits.max_characters_per_batch
                or item.max_value_tokens_per_batch != row_limits.max_value_tokens_per_batch
                or item.reasons != reasons
            ):
                raise PublicDevelopmentSummaryError(
                    "typed unbatchable row does not match its planned row and limits"
                )
        outcome_path = run_root / paper_id / "private" / "row-enumeration.json"
        if not outcome_path.is_file() or outcome_path.is_symlink():
            raise PublicDevelopmentSummaryError("each paper requires a regular private row outcome")
        private_outcome = _mapping(read_json(outcome_path), "private row outcome")
        if (
            private_outcome.get("schema_version") != "row-enumeration-outcome/0.1"
            or private_outcome.get("plan_sha256") != plan_sha256
            or private_outcome.get("telemetry") != row_outcome
            or private_outcome.get("calls") != row_calls
            or private_outcome.get("attempts") != row_attempts
        ):
            raise PublicDevelopmentSummaryError("private row outcome does not match the paper run")
        raw_records = _mapping(private_outcome.get("records"), "private row records")
        records: dict[str, RowDispositionRecord] = {}
        try:
            for row_id, raw_record in raw_records.items():
                if not isinstance(row_id, str) or not row_id:
                    raise PublicDevelopmentSummaryError(
                        "private row record keys must be non-empty strings"
                    )
                record = RowDispositionRecord.model_validate(raw_record)
                if record.row_id != row_id:
                    raise PublicDevelopmentSummaryError(
                        "private row record key does not match its row id"
                    )
                if any(candidate.paper_id != paper_id for candidate in record.candidates):
                    raise PublicDevelopmentSummaryError(
                        "private row record candidate belongs to another paper"
                    )
                records[row_id] = record
        except ValueError as error:
            raise PublicDevelopmentSummaryError("private row records are invalid") from error
        unresolved_ids = _unique_strings(
            private_outcome.get("unresolved_row_ids"), "unresolved row ids"
        )
        unbatchable_ids = _unique_strings(
            private_outcome.get("unbatchable_row_ids"), "unbatchable row ids"
        )
        unknown_ids = _unique_strings(private_outcome.get("unknown_row_ids"), "unknown row ids")
        invalid_reasons = _mapping(
            private_outcome.get("invalid_row_reasons"), "invalid row reasons"
        )
        if any(
            not isinstance(row_id, str) or not isinstance(reason, str) or not reason
            for row_id, reason in invalid_reasons.items()
        ):
            raise PublicDevelopmentSummaryError("invalid row reasons are malformed")
        record_ids = set(records)
        unresolved_set = set(unresolved_ids)
        unbatchable_set = set(unbatchable_ids)
        if (
            record_ids & unresolved_set
            or record_ids & unbatchable_set
            or unresolved_set & unbatchable_set
            or record_ids | unresolved_set | unbatchable_set != planned_id_set
            or unbatchable_set != plan_unbatchable_set
            or set(unknown_ids) & planned_id_set
            or set(invalid_reasons) - planned_id_set
        ):
            raise PublicDevelopmentSummaryError(
                "private row outcome does not exactly partition the planned row ids"
            )
        actual_dispositions = Counter(record.disposition.value for record in records.values())
        actual_dispositions.update(
            {disposition: 0 for disposition in _ROW_DISPOSITIONS - actual_dispositions.keys()}
        )
        reported_dispositions = _mapping(row_outcome.get("dispositions"), "row dispositions")
        if set(reported_dispositions) != _ROW_DISPOSITIONS or {
            name: _nonnegative_int(value, f"row disposition {name}")
            for name, value in reported_dispositions.items()
        } != dict(actual_dispositions):
            raise PublicDevelopmentSummaryError(
                "row disposition telemetry does not match private row records"
            )
        if (
            _nonnegative_int(row_outcome.get("rows_resolved"), "rows resolved") != len(record_ids)
            or _nonnegative_int(row_outcome.get("rows_unresolved"), "rows unresolved")
            != len(unresolved_ids)
            or _nonnegative_int(row_outcome.get("rows_unbatchable"), "rows unbatchable")
            != len(unbatchable_ids)
            or _nonnegative_int(row_execution.get("unknown_row_ids_seen"), "unknown row ids seen")
            != len(unknown_ids)
            or _nonnegative_int(row_execution.get("invalid_rows_seen"), "invalid rows seen")
            != len(invalid_reasons)
        ):
            raise PublicDevelopmentSummaryError(
                "row telemetry does not match the private row-id ledger"
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
    try:
        _assert_public_payload(summary, "public development summary")
    except ExtractionReviewCardError as error:
        raise PublicDevelopmentSummaryError(
            "projected summary failed the public-payload audit"
        ) from error
    return summary


def write_public_development_summary(run_root: Path, output_path: Path) -> str:
    """Build and atomically write one aggregate outside the private run tree."""

    run_root = run_root.resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(run_root)
    except ValueError:
        pass
    else:
        raise PublicDevelopmentSummaryError("public summary output must be outside the run root")
    return write_json(output_path, build_public_development_summary(run_root))


__all__ = [
    "PUBLIC_DEVELOPMENT_SUMMARY_SCHEMA_VERSION",
    "PublicDevelopmentSummaryError",
    "build_public_development_summary",
    "write_public_development_summary",
]
