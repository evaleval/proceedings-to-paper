"""Convert strict LLM proposals into typed Candidate Observations."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from proceedings_to_eee.domain.observation import (
    CandidateObservation,
    EvidenceAnchor,
    MetricSpec,
    ObservationScope,
    ReportedValue,
    RoleAssignment,
    Uncertainty,
)
from proceedings_to_eee.domain.status import ValueComparator
from proceedings_to_eee.extraction.llm_schema import (
    WireExtraction,
    WireObservation,
    WireRowExtraction,
    provider_json_schema,
    row_provider_json_schema,
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment
from proceedings_to_eee.extraction.prompt import (
    ROW_ENUMERATION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    page_prompt,
    row_batch_prompt,
)
from proceedings_to_eee.extraction.row_enumeration import (
    EnumerationRow,
    RowAttemptTelemetry,
    RowBatch,
    RowDisposition,
    RowDispositionRecord,
    RowEnumerationPlan,
    recovery_batches,
)
from proceedings_to_eee.providers.openrouter import (
    OpenRouterClient,
    ProviderCall,
    ProviderRequestRejectedError,
    ProviderResponseValidationError,
    structured_request_contract,
)

EXTRACTOR_SEED = 7
EXTRACTOR_SCHEMA_NAME = "paper_evaluation_candidates"
ROW_EXTRACTOR_SCHEMA_NAME = "paper_table_row_dispositions"


@dataclass(frozen=True)
class RowBatchAttemptResult:
    records: dict[str, RowDispositionRecord]
    unresolved_row_ids: list[str]
    unknown_row_ids: list[str]
    invalid_row_reasons: dict[str, str]
    call: ProviderCall
    warnings: list[str]


@dataclass
class RowEnumerationOutcome:
    records: dict[str, RowDispositionRecord] = field(default_factory=dict)
    calls: list[ProviderCall] = field(default_factory=list)
    attempts: list[RowAttemptTelemetry] = field(default_factory=list)
    unresolved_row_ids: list[str] = field(default_factory=list)
    unbatchable_row_ids: list[str] = field(default_factory=list)
    unknown_row_ids: list[str] = field(default_factory=list)
    invalid_row_reasons: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def candidates(self) -> list[CandidateObservation]:
        return [candidate for record in self.records.values() for candidate in record.candidates]

    @property
    def telemetry(self) -> dict[str, Any]:
        dispositions = {item.value: 0 for item in RowDisposition}
        for record in self.records.values():
            dispositions[record.disposition.value] += 1
        recovery_calls = sum(attempt.depth == 1 for attempt in self.attempts)
        return {
            "rows_resolved": len(self.records),
            "rows_unresolved": len(self.unresolved_row_ids),
            "rows_unbatchable": len(self.unbatchable_row_ids),
            "dispositions": dispositions,
            "calls": len(self.calls),
            "base_calls": len(self.attempts) - recovery_calls,
            "recovery_calls": recovery_calls,
            "attempts": len(self.attempts),
            "input_tokens": sum(call.input_tokens or 0 for call in self.calls),
            "output_tokens": sum(call.output_tokens or 0 for call in self.calls),
            "total_tokens": sum(call.total_tokens or 0 for call in self.calls),
            "cost_usd": sum(call.cost_usd or 0.0 for call in self.calls),
        }


def extractor_request_contract(*, seed: int = EXTRACTOR_SEED) -> dict[str, Any]:
    """Return the call-independent provider contract recorded for extraction runs."""

    return structured_request_contract(
        schema_name=EXTRACTOR_SCHEMA_NAME,
        schema=provider_json_schema(),
        seed=seed,
        require_parameters=False,
    )


def row_extractor_request_contract(*, seed: int = EXTRACTOR_SEED) -> dict[str, Any]:
    """Return the independent strict-schema contract for row dispositions."""

    return structured_request_contract(
        schema_name=ROW_EXTRACTOR_SCHEMA_NAME,
        schema=row_provider_json_schema(),
        seed=seed,
        require_parameters=False,
    )


def _candidate_from_wire(
    *,
    proposed: WireObservation,
    paper_id: str,
    model: str,
    fragment: PageFragment,
    payload_hash: str,
    evidence_row: EnumerationRow | None = None,
    extraction_method: str | None = None,
) -> CandidateObservation:
    """Apply stricter cross-field domain invariants to one wire proposal."""

    roles = [
        RoleAssignment(
            role=role.role,
            raw_name=role.raw_name,
            version=role.version,
            provider=role.provider,
            confidence=role.confidence,
        )
        for role in proposed.roles
    ]
    scope = ObservationScope(**proposed.scope.model_dump()) if proposed.scope else None
    metric = MetricSpec(**proposed.metric.model_dump()) if proposed.metric else None
    value = None
    if proposed.value:
        uncertainty = (
            Uncertainty(**proposed.value.uncertainty.model_dump())
            if proposed.value.uncertainty
            else None
        )
        value = ReportedValue(
            raw=proposed.value.raw,
            numeric=proposed.value.numeric,
            unit=proposed.value.unit,
            comparator=proposed.value.comparator,
            uncertainty=uncertainty,
        )
    evidence = [
        EvidenceAnchor(
            source_id=fragment.source_id,
            page=fragment.page,
            kind=anchor.kind,
            label=evidence_row.table_label if evidence_row is not None else anchor.label,
            row=evidence_row.row_label if evidence_row is not None else anchor.row,
            column=anchor.column,
            quote=anchor.quote,
        )
        for anchor in proposed.evidence
    ]
    return CandidateObservation(
        paper_id=paper_id,
        claim_type=proposed.claim_type,
        roles=roles,
        scope=scope,
        metric=metric,
        value=value,
        evidence=evidence,
        extraction_method=extraction_method or f"openrouter:{model}",
        extraction_confidence=proposed.extraction_confidence,
        evaluation_construct=proposed.evaluation_construct,
        operationalization=proposed.operationalization,
        decision_rule=proposed.decision_rule,
        evaluation_date=proposed.evaluation_date,
        notes=proposed.notes,
        raw_payload_hash=payload_hash,
    )


def extract_page_candidates(
    *,
    client: OpenRouterClient,
    model: str,
    paper_id: str,
    paper_title: str,
    fragment: PageFragment,
    max_tokens: int = 16_000,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = "minimal",
    seed: int = EXTRACTOR_SEED,
) -> tuple[list[CandidateObservation], ProviderCall, list[str]]:
    """Make one source-scoped proposal call for one page."""

    response = client.structured_chat(
        model=model,
        system=SYSTEM_PROMPT,
        user=page_prompt(paper_title=paper_title, paper_id=paper_id, fragment=fragment),
        schema_name=EXTRACTOR_SCHEMA_NAME,
        schema=provider_json_schema(),
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        seed=seed,
        require_parameters=False,
    )
    try:
        wire = WireExtraction.model_validate(response.payload)
    except PydanticValidationError:
        raise ProviderResponseValidationError(
            call=response.call,
            code="wire_validation",
        ) from None
    payload_hash = hashlib.sha256(
        json.dumps(response.payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    candidates: list[CandidateObservation] = []
    rejected_candidates = 0
    for proposed in wire.observations:
        try:
            candidate = _candidate_from_wire(
                proposed=proposed,
                paper_id=paper_id,
                model=model,
                fragment=fragment,
                payload_hash=payload_hash,
            )
        except PydanticValidationError:
            rejected_candidates += 1
            continue
        candidates.append(candidate)
    warnings: list[str] = []
    if wire.warnings:
        warnings.append(f"provider_reported_warnings={len(wire.warnings)}")
    if rejected_candidates:
        warnings.append(f"local_candidate_validation_rejected={rejected_candidates}")
    return candidates, response.call, warnings


def _normalize_row_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\u2212", "-")
    normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _batch_fragment(batch: RowBatch) -> PageFragment:
    text = "\n".join(row.raw_text for row in batch.rows) + "\n"
    return PageFragment(
        fragment_id=batch.batch_id,
        source_id=batch.source_id,
        page=batch.page,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=batch.value_token_count,
        result_signal_score=float(batch.value_token_count),
    )


def _observation_supported_by_row(proposed: WireObservation, row: EnumerationRow) -> bool:
    row_text = _normalize_row_text(row.raw_text)
    if not proposed.evidence:
        return False
    if any(anchor.kind.value != "table" for anchor in proposed.evidence):
        return False
    if any(_normalize_row_text(anchor.quote) not in row_text for anchor in proposed.evidence):
        return False
    if proposed.value is None:
        return True
    raw_value = _normalize_row_text(proposed.value.raw)
    printed_values = {_normalize_row_text(value.raw) for value in row.values}
    if raw_value in printed_values:
        return True
    if proposed.value.comparator is ValueComparator.EXACT:
        return False
    # The structural value tokenizer records the numeric token while the proposal's
    # exact raw text may retain a printed comparator such as ``<0.001``.
    without_comparator = re.sub(r"^(?:<=|>=|<|>|≤|≥|≈|~)\s*", "", raw_value)
    return without_comparator in printed_values


def extract_row_batch_candidates(
    *,
    client: OpenRouterClient,
    model: str,
    paper_id: str,
    paper_title: str,
    batch: RowBatch,
    max_tokens: int = 16_000,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = "minimal",
    seed: int = EXTRACTOR_SEED,
) -> RowBatchAttemptResult:
    """Make one strict row call and reconcile it against the exact input IDs."""

    response = client.structured_chat(
        model=model,
        system=ROW_ENUMERATION_SYSTEM_PROMPT,
        user=row_batch_prompt(paper_title=paper_title, paper_id=paper_id, batch=batch),
        schema_name=ROW_EXTRACTOR_SCHEMA_NAME,
        schema=row_provider_json_schema(),
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        seed=seed,
        require_parameters=False,
    )
    try:
        wire = WireRowExtraction.model_validate(response.payload)
    except PydanticValidationError:
        raise ProviderResponseValidationError(
            call=response.call,
            code="wire_validation",
        ) from None

    payload_hash = hashlib.sha256(
        json.dumps(response.payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    expected = {row.row_id: row for row in batch.rows}
    returned: dict[str, list[Any]] = {}
    unknown: list[str] = []
    for item in wire.dispositions:
        if item.row_id not in expected:
            unknown.append(item.row_id)
            continue
        returned.setdefault(item.row_id, []).append(item)

    records: dict[str, RowDispositionRecord] = {}
    invalid: dict[str, str] = {}
    fragment = _batch_fragment(batch)
    for row_id, row in expected.items():
        items = returned.get(row_id, [])
        if not items:
            invalid[row_id] = "missing_disposition"
            continue
        if len(items) != 1:
            invalid[row_id] = "duplicate_disposition"
            continue
        item = items[0]
        if item.disposition is RowDisposition.RESULT and not item.observations:
            invalid[row_id] = "result_without_observation"
            continue
        if item.disposition is not RowDisposition.RESULT and item.observations:
            invalid[row_id] = "abstention_with_observation"
            continue
        if item.disposition is not RowDisposition.RESULT:
            records[row_id] = RowDispositionRecord(
                row_id=row_id,
                disposition=item.disposition,
                candidates=[],
                note=item.note,
            )
            continue
        if any(not _observation_supported_by_row(proposed, row) for proposed in item.observations):
            invalid[row_id] = "observation_not_bound_to_row"
            continue
        row_candidates: list[CandidateObservation] = []
        try:
            for proposed in item.observations:
                row_candidates.append(
                    _candidate_from_wire(
                        proposed=proposed,
                        paper_id=paper_id,
                        model=model,
                        fragment=fragment,
                        payload_hash=payload_hash,
                        evidence_row=row,
                        extraction_method=f"openrouter:{model}:row-enumeration",
                    )
                )
        except PydanticValidationError:
            invalid[row_id] = "domain_candidate_validation"
            continue
        records[row_id] = RowDispositionRecord(
            row_id=row_id,
            disposition=item.disposition,
            candidates=row_candidates,
            note=item.note,
        )

    warnings: list[str] = []
    if wire.warnings:
        warnings.append(f"provider_reported_warnings={len(wire.warnings)}")
    if unknown:
        warnings.append(f"unknown_row_dispositions={len(unknown)}")
    if invalid:
        warnings.append(f"invalid_or_missing_row_dispositions={len(invalid)}")
    return RowBatchAttemptResult(
        records=records,
        unresolved_row_ids=[row.row_id for row in batch.rows if row.row_id not in records],
        unknown_row_ids=list(dict.fromkeys(unknown)),
        invalid_row_reasons=invalid,
        call=response.call,
        warnings=warnings,
    )


def enumerate_row_batch(
    *,
    client: OpenRouterClient,
    model: str,
    paper_id: str,
    paper_title: str,
    batch: RowBatch,
    max_tokens: int = 16_000,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = "minimal",
    seed: int = EXTRACTOR_SEED,
    max_recovery_depth: int = 1,
) -> RowEnumerationOutcome:
    """Resolve one base batch with at most one unresolved-only split level."""

    if max_recovery_depth not in {0, 1}:
        raise ValueError("row recovery depth must be zero or one")
    outcome = RowEnumerationOutcome()

    def attempt(current: RowBatch, depth: int) -> list[str]:
        try:
            result = extract_row_batch_candidates(
                client=client,
                model=model,
                paper_id=paper_id,
                paper_title=paper_title,
                batch=current,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                seed=seed,
            )
        except ProviderResponseValidationError as error:
            outcome.calls.append(error.call)
            unresolved = [row.row_id for row in current.rows]
            outcome.attempts.append(
                RowAttemptTelemetry(
                    batch_id=current.batch_id,
                    depth=depth,
                    row_ids=unresolved,
                    status=f"provider_response_{error.code}",
                    unresolved_row_ids=unresolved,
                    completed_provider_call=True,
                )
            )
            return unresolved
        except (ProviderRequestRejectedError, RuntimeError):
            unresolved = [row.row_id for row in current.rows]
            outcome.attempts.append(
                RowAttemptTelemetry(
                    batch_id=current.batch_id,
                    depth=depth,
                    row_ids=unresolved,
                    status="provider_request_failed",
                    unresolved_row_ids=unresolved,
                    completed_provider_call=False,
                )
            )
            return unresolved
        outcome.calls.append(result.call)
        outcome.records.update(result.records)
        outcome.unknown_row_ids.extend(result.unknown_row_ids)
        outcome.invalid_row_reasons.update(result.invalid_row_reasons)
        outcome.warnings.extend(result.warnings)
        outcome.attempts.append(
            RowAttemptTelemetry(
                batch_id=current.batch_id,
                depth=depth,
                row_ids=[row.row_id for row in current.rows],
                status="success" if not result.unresolved_row_ids else "partial_invalid",
                resolved_row_ids=list(result.records),
                unresolved_row_ids=result.unresolved_row_ids,
                unknown_row_ids=result.unknown_row_ids,
                completed_provider_call=True,
            )
        )
        return result.unresolved_row_ids

    unresolved = attempt(batch, 0)
    if unresolved and max_recovery_depth:
        for child in recovery_batches(batch, unresolved):
            attempt(child, 1)
    outcome.unresolved_row_ids = [
        row.row_id for row in batch.rows if row.row_id not in outcome.records
    ]
    outcome.unknown_row_ids = list(dict.fromkeys(outcome.unknown_row_ids))
    if len(outcome.attempts) > 3:
        raise AssertionError("row batch exceeded its three-call recovery bound")
    return outcome


def enumerate_row_plan(
    *,
    client: OpenRouterClient,
    model: str,
    paper_id: str,
    paper_title: str,
    plan: RowEnumerationPlan,
    max_tokens: int = 16_000,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = "minimal",
    seed: int = EXTRACTOR_SEED,
) -> RowEnumerationOutcome:
    """Run a deterministic plan while retaining per-batch attempt telemetry."""

    combined = RowEnumerationOutcome()
    combined.unbatchable_row_ids = [row.row_id for row in plan.unbatchable_rows]
    for batch in plan.batches:
        result = enumerate_row_batch(
            client=client,
            model=model,
            paper_id=paper_id,
            paper_title=paper_title,
            batch=batch,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            seed=seed,
            max_recovery_depth=plan.config.max_recovery_depth,
        )
        combined.records.update(result.records)
        combined.calls.extend(result.calls)
        combined.attempts.extend(result.attempts)
        combined.unresolved_row_ids.extend(result.unresolved_row_ids)
        combined.unknown_row_ids.extend(result.unknown_row_ids)
        combined.invalid_row_reasons.update(result.invalid_row_reasons)
        combined.warnings.extend(result.warnings)
    combined.unresolved_row_ids = list(dict.fromkeys(combined.unresolved_row_ids))
    combined.unknown_row_ids = list(dict.fromkeys(combined.unknown_row_ids))
    if len(combined.attempts) > plan.telemetry.maximum_calls:
        raise AssertionError("row plan exceeded its deterministic recovery bound")
    return combined
