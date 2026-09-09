"""Shared validation for dense-row results, checkpoints, and offline ledgers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from typing import Protocol

from proceedings_to_eee.domain.provenance import ProposalStage
from proceedings_to_eee.domain.status import EvidenceKind, ValueComparator
from proceedings_to_eee.extraction.llm_schema import row_provider_json_schema
from proceedings_to_eee.extraction.prompt import ROW_ENUMERATION_SYSTEM_PROMPT, row_batch_prompt
from proceedings_to_eee.extraction.row_enumeration import (
    EnumerationRow,
    EnumerationValue,
    RowAttemptTelemetry,
    RowBatch,
    RowDispositionRecord,
    RowEnumerationConfig,
    RowEnumerationPlan,
    RowPlanTelemetry,
    RowProtocolEvent,
    RowTablePlan,
    RowTerminalCounts,
    RowTerminalLedger,
    RowTerminalRecord,
    RowTerminalState,
    make_row_batch,
    recovery_batches,
)
from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    completion_token_parameter_for_model,
    structured_request_contract,
)
from proceedings_to_eee.validation.field_provenance import row_field_provenance

ROW_EXTRACTOR_SCHEMA_NAME = "paper_table_row_dispositions"


class RowOutcomeLike(Protocol):
    records: dict[str, RowDispositionRecord]
    calls: list[ProviderCall]
    attempts: list[RowAttemptTelemetry]
    unresolved_row_ids: list[str]
    unbatchable_row_ids: list[str]
    unknown_row_ids: list[str]
    invalid_row_reasons: dict[str, str]


_RESPONSE_VALID_ATTEMPT_STATUSES = frozenset({"success", "partial_invalid"})
_RESPONSE_FAILURE_ATTEMPT_STATUSES = frozenset(
    {
        "provider_response_invalid_json",
        "provider_response_schema_validation",
        "provider_response_wire_validation",
        "provider_response_returned_model_mismatch",
    }
)


def partition_row_provider_calls(
    outcome: RowOutcomeLike,
) -> tuple[list[ProviderCall], list[ProviderCall]]:
    """Return completed and response-valid calls from the ordered attempt ledger."""

    completed_attempts = [
        attempt for attempt in outcome.attempts if attempt.completed_provider_call
    ]
    if len(completed_attempts) != len(outcome.calls):
        raise ValueError("row calls do not match completed attempt telemetry")
    successful: list[ProviderCall] = []
    for attempt, call in zip(completed_attempts, outcome.calls, strict=True):
        if attempt.status in _RESPONSE_VALID_ATTEMPT_STATUSES:
            successful.append(call)
        elif attempt.status not in _RESPONSE_FAILURE_ATTEMPT_STATUSES:
            raise ValueError("completed row attempt has an unsupported terminal status")
    if any(
        not attempt.completed_provider_call and attempt.status != "provider_request_failed"
        for attempt in outcome.attempts
    ):
        raise ValueError("no-call row attempt has an unsupported terminal status")
    return list(outcome.calls), successful


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\u2212", "-")
    return re.sub(r"\s+", " ", value).strip()


def matching_row_value(
    row: EnumerationRow,
    raw: str,
    comparator: ValueComparator = ValueComparator.EXACT,
) -> EnumerationValue | None:
    """Return the unique physical value token supporting a proposed raw value."""

    normalized = _normalize(raw)
    if comparator is not ValueComparator.EXACT:
        normalized = re.sub(r"^(?:<=|>=|<|>|≤|≥|≈|~)\s*", "", normalized)
    matches = [value for value in row.values if _normalize(value.raw) == normalized]
    return matches[0] if len(matches) == 1 else None


def _expected_prompt_sha256(*, paper_id: str, paper_title: str, batch: RowBatch) -> str:
    messages = [
        {"role": "system", "content": ROW_ENUMERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": row_batch_prompt(
                paper_title=paper_title,
                paper_id=paper_id,
                batch=batch,
            ),
        },
    ]
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_row_provider_call(
    call: ProviderCall,
    *,
    batch: RowBatch,
    paper_id: str,
    paper_title: str,
    model: str,
    max_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    seed: int | None,
    require_returned_model_match: bool = True,
) -> None:
    """Bind retained call telemetry to the exact row request that produced it."""

    request = structured_request_contract(
        schema_name=ROW_EXTRACTOR_SCHEMA_NAME,
        schema=row_provider_json_schema(),
        seed=seed,
        require_parameters=True,
        model=model,
        max_tokens=max_tokens,
    )
    schema = request["schema"]
    privacy = request["privacy"]
    routing = request["routing"]
    expected = (
        call.model_requested == model,
        not require_returned_model_match or call.model_returned == model,
        call.prompt_sha256
        == _expected_prompt_sha256(paper_id=paper_id, paper_title=paper_title, batch=batch),
        call.temperature == temperature,
        call.reasoning_effort == reasoning_effort,
        call.max_tokens == max_tokens,
        call.completion_token_parameter
        == request["completion_token_parameter"]
        == completion_token_parameter_for_model(model),
        call.seed == request["seed"],
        call.response_format == schema["response_format"],
        call.schema_name == schema["schema_name"],
        call.schema_sha256 == schema["schema_sha256"],
        call.schema_strict == schema["schema_strict"],
        call.data_collection == privacy["data_collection"],
        call.require_parameters == routing["require_parameters"],
        call.zdr == privacy["zdr"],
    )
    if not all(expected):
        raise ValueError("retained provider call does not match the exact row request")


def validate_row_record_binding(
    row: EnumerationRow,
    record: RowDispositionRecord,
    *,
    paper_id: str,
    model: str,
    retained_calls: Iterable[ProviderCall],
) -> None:
    """Bind one typed row record to its paper, physical row, cell, and response."""

    if record.row_id != row.row_id:
        raise ValueError("row record does not match the planned row id")
    response_hashes = {call.response_sha256 for call in retained_calls}
    expected_method = f"openrouter:{model}:row-enumeration"
    for candidate in record.candidates:
        if candidate.paper_id != paper_id:
            raise ValueError("row candidate belongs to another paper")
        if candidate.extraction_method != expected_method:
            raise ValueError("row candidate extraction method/model does not match the run")
        if candidate.raw_payload_hash not in response_hashes:
            raise ValueError("row candidate is not linked to a retained provider response")
        if not candidate.evidence:
            raise ValueError("row candidate has no evidence")
        if candidate.value is None:
            raise ValueError("row result candidate has no raw value")
        cell = matching_row_value(row, candidate.value.raw, candidate.value.comparator)
        if cell is None:
            raise ValueError("row candidate value does not identify one exact planned cell")
        expected_header_ids = [
            header.header_id for binding in cell.header_path for header in binding.headers
        ]
        for anchor in candidate.evidence:
            if (
                anchor.kind is not EvidenceKind.TABLE
                or anchor.source_id != row.source_id
                or anchor.page != row.page
                or anchor.region_id != row.region_id
                or anchor.planned_row_id != row.row_id
                or anchor.label != row.table_label
                or anchor.row != row.row_label
                or _normalize(anchor.quote) not in _normalize(row.raw_text)
                or _normalize(cell.raw) not in _normalize(anchor.quote)
                or anchor.cell_id != cell.cell_id
                or anchor.numeric_token_id != cell.numeric_token_id
                or anchor.header_ids != expected_header_ids
            ):
                raise ValueError("row candidate evidence is not bound to the exact planned row")
        if candidate.schema_version == "candidate-observation/0.3":
            expected_provenance = row_field_provenance(candidate, row, cell)
            if candidate.field_provenance != expected_provenance:
                raise ValueError(
                    "row candidate field provenance does not match its physical sources"
                )
            if len(candidate.proposal_traces) != 1:
                raise ValueError("row candidate must retain exactly one producing proposal")
            trace = candidate.proposal_traces[0]
            if (
                trace.stage is not ProposalStage.ROW_ENUMERATION
                or trace.paper_id != paper_id
                or trace.source_id != row.source_id
                or trace.page != row.page
                or trace.planned_row_id != row.row_id
                or trace.batch_id is None
            ):
                raise ValueError("row candidate proposal lineage does not match the planned row")


def _attempt_batch(
    attempt: RowAttemptTelemetry,
    *,
    plan: RowEnumerationPlan,
) -> RowBatch:
    planned = {row.row_id: row for row in plan.rows}
    if len(attempt.row_ids) != len(set(attempt.row_ids)):
        raise ValueError("row attempt repeats a planned row id")
    try:
        rows = [planned[row_id] for row_id in attempt.row_ids]
    except KeyError as error:
        raise ValueError("row attempt contains an unknown planned row id") from error
    batch = make_row_batch(rows)
    if batch.batch_id != attempt.batch_id:
        raise ValueError("row attempt batch id does not match its exact row inputs")
    if attempt.depth == 0 and batch not in plan.batches:
        raise ValueError("base row attempt is not one of the canonical planned batches")
    if attempt.depth == 1 and not any(
        set(attempt.row_ids) < {row.row_id for row in base.rows} for base in plan.batches
    ):
        raise ValueError("recovery row attempt is not a strict subset of one base batch")
    owned = set(attempt.row_ids)
    resolved = set(attempt.resolved_row_ids)
    unresolved = set(attempt.unresolved_row_ids)
    if (
        len(resolved) != len(attempt.resolved_row_ids)
        or len(unresolved) != len(attempt.unresolved_row_ids)
        or resolved & unresolved
        or resolved | unresolved != owned
    ):
        raise ValueError("row attempt resolved/unresolved ids do not exactly partition its input")
    if any(not isinstance(row_id, str) or not row_id for row_id in attempt.unknown_row_ids):
        raise ValueError("row attempt unknown ids are malformed")
    return batch


def validate_terminal_ledger_against(
    plan: RowEnumerationPlan,
    ledger: RowTerminalLedger,
    *,
    paper_id: str,
    model: str,
    retained_calls: Iterable[ProviderCall],
) -> None:
    """Validate a persisted terminal ledger against its exact plan and calls."""

    if ledger.paper_id != paper_id:
        raise ValueError("row terminal ledger belongs to another paper")
    if ledger.plan_sha256 != sha256_bytes(canonical_json_bytes(plan)):
        raise ValueError("row terminal ledger plan hash mismatch")
    planned = {row.row_id: row for row in plan.rows}
    if set(ledger.terminal_by_row) != set(planned):
        raise ValueError("row terminal ledger does not own every planned row exactly once")
    unbatchable = {row.row_id: row for row in plan.unbatchable_rows}
    calls = list(retained_calls)
    if any(
        call.model_requested != model or call.schema_name != ROW_EXTRACTOR_SCHEMA_NAME
        for call in calls
    ):
        raise ValueError("row terminal ledger retained calls use another model or schema")
    for row_id, terminal in ledger.terminal_by_row.items():
        if row_id in unbatchable:
            expected = unbatchable[row_id]
            if (
                terminal.state is not RowTerminalState.UNSUPPORTED
                or terminal.unsupported_reasons != expected.reasons
            ):
                raise ValueError("unbatchable row is not typed as unsupported with exact reasons")
        elif terminal.state is RowTerminalState.UNSUPPORTED:
            raise ValueError("batchable row cannot be owned by unsupported")
        if terminal.disposition is not None:
            validate_row_record_binding(
                planned[row_id],
                terminal.disposition,
                paper_id=paper_id,
                model=model,
                retained_calls=calls,
            )
    if set(ledger.invalid_row_reasons) - set(planned):
        raise ValueError("row invalid-reason ledger contains unknown planned ids")


def validate_outcome_against(
    plan: RowEnumerationPlan,
    outcome: RowOutcomeLike,
    *,
    paper_id: str,
    paper_title: str,
    model: str,
    max_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    seed: int | None,
) -> RowTerminalLedger:
    """Reject overlaps/omissions and project one terminal owner per planned row."""

    planned = {row.row_id: row for row in plan.rows}
    record_ids = list(outcome.records)
    unresolved_ids = outcome.unresolved_row_ids
    unsupported_ids = outcome.unbatchable_row_ids
    for name, values in (
        ("record", record_ids),
        ("unresolved", unresolved_ids),
        ("unsupported", unsupported_ids),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"row outcome repeats a {name} terminal owner")
    record_set = set(record_ids)
    unresolved_set = set(unresolved_ids)
    unsupported_set = set(unsupported_ids)
    expected_unsupported = {row.row_id for row in plan.unbatchable_rows}
    if (
        record_set & unresolved_set
        or record_set & unsupported_set
        or unresolved_set & unsupported_set
        or record_set | unresolved_set | unsupported_set != set(planned)
        or unsupported_set != expected_unsupported
    ):
        raise ValueError("row outcome does not exactly partition all planned row ids")
    if any(key != record.row_id for key, record in outcome.records.items()):
        raise ValueError("row outcome record key does not match record row_id")
    if set(outcome.invalid_row_reasons) - set(planned):
        raise ValueError("row outcome invalid reasons contain unknown planned ids")

    partition_row_provider_calls(outcome)
    completed_attempts: list[tuple[RowAttemptTelemetry, RowBatch]] = []
    protocol_events: list[RowProtocolEvent] = []
    for attempt in outcome.attempts:
        batch = _attempt_batch(attempt, plan=plan)
        if attempt.completed_provider_call:
            completed_attempts.append((attempt, batch))
        protocol_events.extend(
            RowProtocolEvent(
                batch_id=attempt.batch_id,
                depth=attempt.depth,
                row_id=row_id,
            )
            for row_id in attempt.unknown_row_ids
        )
    if outcome.unknown_row_ids != [event.row_id for event in protocol_events]:
        raise ValueError("row outcome unknown-id telemetry does not match attempt events")

    attempt_index = 0
    for base in plan.batches:
        if attempt_index >= len(outcome.attempts):
            raise ValueError("row outcome omits a canonical base-batch attempt")
        base_attempt = outcome.attempts[attempt_index]
        if (
            base_attempt.depth != 0
            or base_attempt.batch_id != base.batch_id
            or base_attempt.row_ids != [row.row_id for row in base.rows]
        ):
            raise ValueError("row outcome base attempts do not follow the canonical plan")
        attempt_index += 1
        expected_recovery = (
            recovery_batches(base, base_attempt.unresolved_row_ids)
            if plan.config.max_recovery_depth
            else []
        )
        for child in expected_recovery:
            if attempt_index >= len(outcome.attempts):
                raise ValueError("row outcome omits a deterministic recovery attempt")
            recovery = outcome.attempts[attempt_index]
            if (
                recovery.depth != 1
                or recovery.batch_id != child.batch_id
                or recovery.row_ids != [row.row_id for row in child.rows]
            ):
                raise ValueError("row outcome recovery attempts do not match the base result")
            attempt_index += 1
    if attempt_index != len(outcome.attempts):
        raise ValueError("row outcome contains duplicate or noncanonical attempts")
    if len(completed_attempts) != len(outcome.calls):
        raise ValueError("row calls do not match completed attempt telemetry")
    for (attempt, batch), call in zip(completed_attempts, outcome.calls, strict=True):
        returned_model_mismatch = attempt.status == "provider_response_returned_model_mismatch"
        validate_row_provider_call(
            call,
            batch=batch,
            paper_id=paper_id,
            paper_title=paper_title,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            seed=seed,
            require_returned_model_match=not returned_model_mismatch,
        )
        if returned_model_mismatch and call.model_returned == model:
            raise ValueError("row returned-model mismatch status has matching telemetry")

    terminals: dict[str, RowTerminalRecord] = {}
    for row_id, record in outcome.records.items():
        state = RowTerminalState(record.disposition.value)
        terminals[row_id] = RowTerminalRecord(
            row_id=row_id,
            state=state,
            disposition=record,
        )
    for row_id in unresolved_ids:
        terminals[row_id] = RowTerminalRecord(
            row_id=row_id,
            state=RowTerminalState.UNRESOLVED,
            unresolved_reason=outcome.invalid_row_reasons.get(
                row_id,
                "provider_or_protocol_failure",
            ),
        )
    unbatchable_by_id = {row.row_id: row for row in plan.unbatchable_rows}
    for row_id in unsupported_ids:
        terminals[row_id] = RowTerminalRecord(
            row_id=row_id,
            state=RowTerminalState.UNSUPPORTED,
            unsupported_reasons=unbatchable_by_id[row_id].reasons,
        )
    counts_by_state = {state: 0 for state in RowTerminalState}
    for terminal in terminals.values():
        counts_by_state[terminal.state] += 1
    ledger = RowTerminalLedger(
        paper_id=paper_id,
        plan_sha256=sha256_bytes(canonical_json_bytes(plan)),
        terminal_by_row=terminals,
        protocol_events=protocol_events,
        invalid_row_reasons=outcome.invalid_row_reasons,
        counts=RowTerminalCounts(
            planned=len(terminals),
            result=counts_by_state[RowTerminalState.RESULT],
            not_result=counts_by_state[RowTerminalState.NOT_RESULT],
            uncertain=counts_by_state[RowTerminalState.UNCERTAIN],
            unresolved=counts_by_state[RowTerminalState.UNRESOLVED],
            unsupported=counts_by_state[RowTerminalState.UNSUPPORTED],
        ),
    )
    validate_terminal_ledger_against(
        plan,
        ledger,
        paper_id=paper_id,
        model=model,
        retained_calls=outcome.calls,
    )
    return ledger


def validate_batch_outcome_against(
    batch: RowBatch,
    outcome: RowOutcomeLike,
    *,
    config: RowEnumerationConfig,
    paper_id: str,
    paper_title: str,
    model: str,
    max_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    seed: int | None,
) -> RowTerminalLedger:
    """Apply the complete shared invariant to one cacheable base-batch outcome."""

    plan = _single_batch_plan(batch, config=config)
    return validate_outcome_against(
        plan,
        outcome,
        paper_id=paper_id,
        paper_title=paper_title,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        seed=seed,
    )


def _single_batch_plan(
    batch: RowBatch,
    *,
    config: RowEnumerationConfig,
) -> RowEnumerationPlan:
    first = batch.rows[0]
    return RowEnumerationPlan(
        config=config,
        tables=[
            RowTablePlan(
                source_id=first.source_id,
                page=first.page,
                region_id=first.region_id,
                region_span=first.region_span,
                table_label=first.table_label,
                dense=True,
            )
        ],
        rows=batch.rows,
        batches=[batch],
        unbatchable_rows=[],
        telemetry=RowPlanTelemetry(
            tables_considered=1,
            dense_tables=1,
            rows_planned=len(batch.rows),
            unbatchable_rows=0,
            base_batches=1,
            expected_calls=1,
            maximum_calls=1 + 2 * config.max_recovery_depth,
        ),
    )


def validate_batch_outcome_prefix(
    batch: RowBatch,
    outcome: RowOutcomeLike,
    *,
    config: RowEnumerationConfig,
    paper_id: str,
    paper_title: str,
    model: str,
    max_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    seed: int | None,
) -> None:
    """Validate a durably checkpointed prefix of one deterministic row schedule."""

    plan = _single_batch_plan(batch, config=config)
    planned = {row.row_id: row for row in batch.rows}
    record_ids = list(outcome.records)
    unresolved_ids = outcome.unresolved_row_ids
    if (
        not outcome.attempts
        or len(record_ids) != len(set(record_ids))
        or len(unresolved_ids) != len(set(unresolved_ids))
        or set(record_ids) & set(unresolved_ids)
        or set(record_ids) | set(unresolved_ids) != set(planned)
        or outcome.unbatchable_row_ids
        or any(key != record.row_id for key, record in outcome.records.items())
        or set(outcome.invalid_row_reasons) - set(planned)
    ):
        raise ValueError("row checkpoint prefix does not partition its exact base batch")

    attempt_batches = [_attempt_batch(attempt, plan=plan) for attempt in outcome.attempts]
    base_attempt = outcome.attempts[0]
    if (
        base_attempt.depth != 0
        or base_attempt.batch_id != batch.batch_id
        or base_attempt.row_ids != [row.row_id for row in batch.rows]
    ):
        raise ValueError("row checkpoint prefix does not start with its canonical base batch")
    children = (
        recovery_batches(batch, base_attempt.unresolved_row_ids)
        if config.max_recovery_depth
        else []
    )
    expected_batches = [batch, *children]
    if len(outcome.attempts) > len(expected_batches):
        raise ValueError("row checkpoint prefix exceeds its deterministic recovery schedule")
    for attempt, actual_batch, expected_batch in zip(
        outcome.attempts,
        attempt_batches,
        expected_batches,
        strict=False,
    ):
        expected_depth = 0 if expected_batch is batch else 1
        if (
            attempt.depth != expected_depth
            or actual_batch != expected_batch
            or attempt.row_ids != [row.row_id for row in expected_batch.rows]
        ):
            raise ValueError("row checkpoint prefix attempts are not a canonical prefix")

    completed = [
        (attempt, attempt_batch)
        for attempt, attempt_batch in zip(outcome.attempts, attempt_batches, strict=True)
        if attempt.completed_provider_call
    ]
    if len(completed) != len(outcome.calls):
        raise ValueError("row checkpoint prefix calls do not match completed attempts")
    for (attempt, attempt_batch), call in zip(completed, outcome.calls, strict=True):
        returned_model_mismatch = attempt.status == "provider_response_returned_model_mismatch"
        validate_row_provider_call(
            call,
            batch=attempt_batch,
            paper_id=paper_id,
            paper_title=paper_title,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            seed=seed,
            require_returned_model_match=not returned_model_mismatch,
        )
        if returned_model_mismatch and call.model_returned == model:
            raise ValueError("row returned-model mismatch status has matching telemetry")
    expected_unknown = [
        row_id for attempt in outcome.attempts for row_id in attempt.unknown_row_ids
    ]
    if outcome.unknown_row_ids != expected_unknown:
        raise ValueError("row checkpoint prefix unknown-id telemetry does not match attempts")
    for row_id, record in outcome.records.items():
        validate_row_record_binding(
            planned[row_id],
            record,
            paper_id=paper_id,
            model=model,
            retained_calls=outcome.calls,
        )
