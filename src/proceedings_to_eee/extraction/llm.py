"""Convert strict LLM proposals into typed Candidate Observations."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
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
from proceedings_to_eee.domain.provenance import ProposalStage, make_proposal_trace
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
from proceedings_to_eee.extraction.row_validation import (
    matching_row_value,
    validate_outcome_against,
    validate_row_record_binding,
)
from proceedings_to_eee.providers.openrouter import (
    OpenRouterClient,
    ProviderCall,
    ProviderRequestRejectedError,
    ProviderResponseValidationError,
    require_exact_returned_model,
    structured_request_contract,
)
from proceedings_to_eee.validation.field_provenance import (
    quote_field_provenance,
    row_field_provenance,
)

EXTRACTOR_TEMPERATURE: None = None
EXTRACTOR_REASONING_EFFORT = "minimal"
EXTRACTOR_SEED: None = None
EXTRACTOR_REQUIRE_PARAMETERS = True
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
            "reasoning_tokens": sum(call.reasoning_tokens or 0 for call in self.calls),
            "total_tokens": sum(call.total_tokens or 0 for call in self.calls),
            "cost_usd": sum(call.cost_usd or 0.0 for call in self.calls),
        }


def _merge_row_records_exact(
    target: dict[str, RowDispositionRecord],
    incoming: dict[str, RowDispositionRecord],
) -> None:
    overlap = set(target) & set(incoming)
    if overlap:
        raise ValueError("row disposition merge would assign duplicate terminal ownership")
    target.update(incoming)


def extractor_request_contract(
    *,
    seed: int | None = EXTRACTOR_SEED,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Return a schema-only or fully materialized extraction request contract."""

    return structured_request_contract(
        schema_name=EXTRACTOR_SCHEMA_NAME,
        schema=provider_json_schema(),
        seed=seed,
        require_parameters=EXTRACTOR_REQUIRE_PARAMETERS,
        model=model,
        max_tokens=max_tokens,
    )


def row_extractor_request_contract(
    *,
    seed: int | None = EXTRACTOR_SEED,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Return a schema-only or fully materialized row request contract."""

    return structured_request_contract(
        schema_name=ROW_EXTRACTOR_SCHEMA_NAME,
        schema=row_provider_json_schema(),
        seed=seed,
        require_parameters=EXTRACTOR_REQUIRE_PARAMETERS,
        model=model,
        max_tokens=max_tokens,
    )


def _candidate_from_wire(
    *,
    proposed: WireObservation,
    paper_id: str,
    model: str,
    fragment: PageFragment,
    payload_hash: str,
    proposal_ordinal: int,
    evidence_row: EnumerationRow | None = None,
    batch_id: str | None = None,
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
    evidence_cell = (
        matching_row_value(
            evidence_row,
            proposed.value.raw,
            proposed.value.comparator,
        )
        if evidence_row is not None and proposed.value is not None
        else None
    )
    evidence = [
        EvidenceAnchor(
            source_id=fragment.source_id,
            page=fragment.page,
            kind=anchor.kind,
            label=evidence_row.table_label if evidence_row is not None else anchor.label,
            row=evidence_row.row_label if evidence_row is not None else anchor.row,
            column=anchor.column,
            region_id=evidence_row.region_id if evidence_row is not None else None,
            planned_row_id=evidence_row.row_id if evidence_row is not None else None,
            cell_id=evidence_cell.cell_id if evidence_cell is not None else None,
            numeric_token_id=(
                evidence_cell.numeric_token_id if evidence_cell is not None else None
            ),
            header_ids=(
                [
                    header.header_id
                    for binding in evidence_cell.header_path
                    for header in binding.headers
                ]
                if evidence_cell is not None
                else []
            ),
            quote=anchor.quote,
        )
        for anchor in proposed.evidence
    ]
    candidate = CandidateObservation(
        schema_version="candidate-observation/0.3",
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
    proposal_payload_sha256 = hashlib.sha256(
        json.dumps(
            proposed.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    stage = (
        ProposalStage.ROW_ENUMERATION if evidence_row is not None else ProposalStage.LEGACY_BLOCK
    )
    trace = make_proposal_trace(
        stage=stage,
        paper_id=paper_id,
        source_id=fragment.source_id,
        page=fragment.page,
        fragment_id=fragment.fragment_id if evidence_row is None else None,
        batch_id=batch_id,
        planned_row_id=evidence_row.row_id if evidence_row is not None else None,
        raw_payload_sha256=proposal_payload_sha256,
        proposal_ordinal=proposal_ordinal,
        input_candidate_id=str(candidate.observation_id),
    )
    provenance = (
        row_field_provenance(candidate, evidence_row, evidence_cell)
        if evidence_row is not None and evidence_cell is not None
        else quote_field_provenance(candidate)
    )
    payload = candidate.model_dump(mode="json")
    payload["observation_id"] = None
    payload["proposal_traces"] = [trace.model_dump(mode="json")]
    payload["field_provenance"] = [item.model_dump(mode="json") for item in provenance]
    return CandidateObservation.model_validate(payload)


def extract_page_candidates(
    *,
    client: OpenRouterClient,
    model: str,
    paper_id: str,
    paper_title: str,
    fragment: PageFragment,
    max_tokens: int = 16_000,
    temperature: float | None = EXTRACTOR_TEMPERATURE,
    reasoning_effort: str | None = EXTRACTOR_REASONING_EFFORT,
    seed: int | None = EXTRACTOR_SEED,
) -> tuple[list[CandidateObservation], ProviderCall, list[str]]:
    """Make one source-scoped proposal call for one page."""

    response = require_exact_returned_model(
        client.structured_chat(
            model=model,
            system=SYSTEM_PROMPT,
            user=page_prompt(paper_title=paper_title, paper_id=paper_id, fragment=fragment),
            schema_name=EXTRACTOR_SCHEMA_NAME,
            schema=provider_json_schema(),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            seed=seed,
            require_parameters=EXTRACTOR_REQUIRE_PARAMETERS,
        ),
        requested_model=model,
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
    for proposal_ordinal, proposed in enumerate(wire.observations, start=1):
        try:
            candidate = _candidate_from_wire(
                proposed=proposed,
                paper_id=paper_id,
                model=model,
                fragment=fragment,
                payload_hash=payload_hash,
                proposal_ordinal=proposal_ordinal,
            )
        except (PydanticValidationError, ValueError):
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
    return matching_row_value(row, proposed.value.raw, proposed.value.comparator) is not None


def extract_row_batch_candidates(
    *,
    client: OpenRouterClient,
    model: str,
    paper_id: str,
    paper_title: str,
    batch: RowBatch,
    max_tokens: int = 16_000,
    temperature: float | None = EXTRACTOR_TEMPERATURE,
    reasoning_effort: str | None = EXTRACTOR_REASONING_EFFORT,
    seed: int | None = EXTRACTOR_SEED,
) -> RowBatchAttemptResult:
    """Make one strict row call and reconcile it against the exact input IDs."""

    response = require_exact_returned_model(
        client.structured_chat(
            model=model,
            system=ROW_ENUMERATION_SYSTEM_PROMPT,
            user=row_batch_prompt(paper_title=paper_title, paper_id=paper_id, batch=batch),
            schema_name=ROW_EXTRACTOR_SCHEMA_NAME,
            schema=row_provider_json_schema(),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            seed=seed,
            require_parameters=EXTRACTOR_REQUIRE_PARAMETERS,
        ),
        requested_model=model,
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
    proposal_ordinal = 0
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
                proposal_ordinal += 1
                row_candidates.append(
                    _candidate_from_wire(
                        proposed=proposed,
                        paper_id=paper_id,
                        model=model,
                        fragment=fragment,
                        payload_hash=payload_hash,
                        proposal_ordinal=proposal_ordinal,
                        evidence_row=row,
                        batch_id=batch.batch_id,
                        extraction_method=f"openrouter:{model}:row-enumeration",
                    )
                )
            record = RowDispositionRecord(
                row_id=row_id,
                disposition=item.disposition,
                candidates=row_candidates,
                note=item.note,
            )
            validate_row_record_binding(
                row,
                record,
                paper_id=paper_id,
                model=model,
                retained_calls=[response.call],
            )
        except (PydanticValidationError, ValueError):
            invalid[row_id] = "domain_candidate_validation"
            continue
        records[row_id] = record

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
        unknown_row_ids=unknown,
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
    temperature: float | None = EXTRACTOR_TEMPERATURE,
    reasoning_effort: str | None = EXTRACTOR_REASONING_EFFORT,
    seed: int | None = EXTRACTOR_SEED,
    max_recovery_depth: int = 1,
    resume_outcome: RowEnumerationOutcome | None = None,
    on_progress: Callable[[RowEnumerationOutcome], None] | None = None,
) -> RowEnumerationOutcome:
    """Resolve one base batch with durable progress after every bounded attempt.

    ``resume_outcome`` must be a contract-bound, validated prefix produced for this
    exact base batch.  The pipeline owns that fail-closed validation; this function
    additionally rejects a prefix whose attempt order no longer matches the
    deterministic base/recovery schedule.
    """

    if max_recovery_depth not in {0, 1}:
        raise ValueError("row recovery depth must be zero or one")
    outcome = resume_outcome or RowEnumerationOutcome()
    planned_row_ids = {row.row_id for row in batch.rows}
    if (
        set(outcome.records) - planned_row_ids
        or set(outcome.unresolved_row_ids) - planned_row_ids
        or outcome.unbatchable_row_ids
    ):
        raise ValueError("row resume outcome does not belong to the exact base batch")

    def refresh_unresolved() -> None:
        outcome.unresolved_row_ids = [
            row.row_id for row in batch.rows if row.row_id not in outcome.records
        ]

    def persist_progress() -> None:
        refresh_unresolved()
        if on_progress is not None:
            on_progress(outcome)

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
                    status=(
                        "provider_response_returned_model_mismatch"
                        if error.validation_keyword == "returned_model_mismatch"
                        else f"provider_response_{error.code}"
                    ),
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
        _merge_row_records_exact(outcome.records, result.records)
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

    base_row_ids = [row.row_id for row in batch.rows]
    if outcome.attempts:
        base_attempt = outcome.attempts[0]
        if (
            base_attempt.depth != 0
            or base_attempt.batch_id != batch.batch_id
            or base_attempt.row_ids != base_row_ids
        ):
            raise ValueError("row resume prefix does not start with the exact base batch")
        unresolved = list(base_attempt.unresolved_row_ids)
    else:
        unresolved = attempt(batch, 0)
        persist_progress()

    children = recovery_batches(batch, unresolved) if unresolved and max_recovery_depth else []
    expected_attempts = 1 + len(children)
    if len(outcome.attempts) > expected_attempts:
        raise ValueError("row resume prefix exceeds the deterministic recovery schedule")
    for index, child in enumerate(children, start=1):
        child_row_ids = [row.row_id for row in child.rows]
        if len(outcome.attempts) > index:
            resumed_attempt = outcome.attempts[index]
            if (
                resumed_attempt.depth != 1
                or resumed_attempt.batch_id != child.batch_id
                or resumed_attempt.row_ids != child_row_ids
            ):
                raise ValueError("row resume prefix does not match deterministic recovery")
            continue
        attempt(child, 1)
        persist_progress()

    refresh_unresolved()
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
    temperature: float | None = EXTRACTOR_TEMPERATURE,
    reasoning_effort: str | None = EXTRACTOR_REASONING_EFFORT,
    seed: int | None = EXTRACTOR_SEED,
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
        _merge_row_records_exact(combined.records, result.records)
        combined.calls.extend(result.calls)
        combined.attempts.extend(result.attempts)
        combined.unresolved_row_ids.extend(result.unresolved_row_ids)
        combined.unknown_row_ids.extend(result.unknown_row_ids)
        combined.invalid_row_reasons.update(result.invalid_row_reasons)
        combined.warnings.extend(result.warnings)
    if len(combined.attempts) > plan.telemetry.maximum_calls:
        raise AssertionError("row plan exceeded its deterministic recovery bound")
    validate_outcome_against(
        plan,
        combined,
        paper_id=paper_id,
        paper_title=paper_title,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        seed=seed,
    )
    return combined
