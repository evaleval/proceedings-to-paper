"""Deterministic dense-table row planning for staged candidate extraction.

The region index owns table geometry.  This module only projects indexed rows that
were already inside selected result-block bodies into stable, bounded provider inputs.
It makes no claim that a row is result-bearing; that is the disposition stage's job.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from itertools import groupby
from typing import Literal

from pydantic import Field, model_validator

from proceedings_to_eee.domain.observation import CandidateObservation, StrictModel
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.region_index import (
    Caption,
    GridSpan,
    Region,
    RegionKind,
    TableRow,
    build_region_index,
)
from proceedings_to_eee.extraction.result_blocks import ResultBlock

ROW_ENUMERATION_SCHEMA_VERSION = "table-row-enumeration/0.3"
ROW_TERMINAL_SCHEMA_VERSION = "row-terminal-states/0.1"

_CELL = re.compile(r"\S(?:(?!\s{2,}).)*?(?=\s{2,}|$)")
_VALUE = re.compile(r"(?<![\w@-])[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:\s*%)?(?!\w)")


class RowDisposition(StrEnum):
    RESULT = "result"
    NOT_RESULT = "not_result"
    UNCERTAIN = "uncertain"


class UnbatchableReason(StrEnum):
    CHARACTER_LIMIT = "max_characters_per_batch"
    VALUE_TOKEN_LIMIT = "max_value_tokens_per_batch"


class PhysicalCellState(StrEnum):
    EXACT = "exact"
    SPLIT_VALUE_RUN = "split_value_run"
    AMBIGUOUS = "ambiguous"


class HeaderBindingState(StrEnum):
    RESOLVED = "resolved"
    MERGED = "merged"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"


class RowLabelBindingState(StrEnum):
    DIRECT = "direct"
    INHERITED = "inherited"


class RowTerminalState(StrEnum):
    """Exactly one final accounting state for every planned row."""

    RESULT = "result"
    NOT_RESULT = "not_result"
    UNCERTAIN = "uncertain"
    UNRESOLVED = "unresolved"
    UNSUPPORTED = "unsupported"


class RowEnumerationConfig(StrictModel):
    """Hard input and recovery limits for dense-table enumeration."""

    min_dense_table_rows: int = Field(default=2, ge=2)
    max_rows_per_batch: int = Field(default=4, ge=1)
    max_value_tokens_per_batch: int = Field(default=24, ge=1)
    max_characters_per_batch: int = Field(default=4_000, ge=128)
    max_recovery_depth: int = Field(default=1, ge=0, le=1)


class EnumerationCell(StrictModel):
    raw: str = Field(min_length=1)
    span: GridSpan


class EnumerationHeaderCell(EnumerationCell):
    header_id: str = Field(pattern=r"^theader_[0-9a-f]{20}$")
    level: int = Field(ge=1)

    @model_validator(mode="after")
    def header_identity_is_canonical(self) -> EnumerationHeaderCell:
        if self.header_id != _header_id(self.level, self.raw, self.span):
            raise ValueError("header cell id does not match its physical span")
        return self


class EnumerationHeaderBinding(StrictModel):
    level: int = Field(ge=1)
    state: HeaderBindingState
    headers: list[EnumerationHeaderCell] = Field(default_factory=list)

    @model_validator(mode="after")
    def state_matches_headers(self) -> EnumerationHeaderBinding:
        if self.state in {HeaderBindingState.RESOLVED, HeaderBindingState.MERGED}:
            if len(self.headers) != 1:
                raise ValueError("resolved or merged header binding requires one header cell")
        elif self.state is HeaderBindingState.AMBIGUOUS:
            if not self.headers:
                raise ValueError("ambiguous header binding requires candidate header cells")
        elif self.headers:
            raise ValueError("missing header binding cannot contain header cells")
        if any(header.level != self.level for header in self.headers):
            raise ValueError("header binding level does not match its cells")
        return self


class EnumerationValue(StrictModel):
    cell_id: str = Field(pattern=r"^tcell_[0-9a-f]{20}$")
    cell_raw: str = Field(min_length=1)
    cell_span: GridSpan
    cell_state: PhysicalCellState
    numeric_token_id: str = Field(pattern=r"^ttoken_[0-9a-f]{20}$")
    ordinal: int = Field(ge=1)
    raw: str = Field(min_length=1)
    span: GridSpan
    header_path: list[EnumerationHeaderBinding]

    @model_validator(mode="after")
    def token_is_inside_cell(self) -> EnumerationValue:
        if (
            self.cell_span.start_line > self.span.start_line
            or self.cell_span.end_line < self.span.end_line
            or (
                self.cell_span.column_start is not None
                and self.span.column_start is not None
                and self.cell_span.column_start > self.span.column_start
            )
            or (
                self.cell_span.column_end is not None
                and self.span.column_end is not None
                and self.cell_span.column_end < self.span.column_end
            )
        ):
            raise ValueError("numeric token span is outside its physical cell")
        levels = [item.level for item in self.header_path]
        if levels != sorted(levels) or len(levels) != len(set(levels)):
            raise ValueError("numeric token header path levels are not canonical")
        return self


class EnumerationHeader(StrictModel):
    level: int = Field(ge=1)
    raw_text: str = Field(min_length=1)
    columns: list[EnumerationHeaderCell]
    span: GridSpan

    @model_validator(mode="after")
    def column_levels_match(self) -> EnumerationHeader:
        if any(column.level != self.level for column in self.columns):
            raise ValueError("header column level does not match its row")
        return self


class EnumerationRowLabel(StrictModel):
    cell_id: str = Field(pattern=r"^tlabel_[0-9a-f]{20}$")
    raw: str = Field(min_length=1)
    span: GridSpan
    state: RowLabelBindingState

    @model_validator(mode="after")
    def identity_is_canonical(self) -> EnumerationRowLabel:
        if self.cell_id != _row_label_cell_id(self.raw, self.span):
            raise ValueError("row-label cell id does not match its physical span")
        return self


class EnumerationRow(StrictModel):
    """One physical indexed table row, bound to exact layout-grid coordinates."""

    schema_version: Literal["table-row-enumeration/0.3"] = ROW_ENUMERATION_SCHEMA_VERSION
    row_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_id: str
    page: int = Field(ge=1)
    region_id: str
    region_span: GridSpan
    table_label: str | None = None
    caption: str | None = None
    caption_span: GridSpan | None = None
    headers: list[EnumerationHeader]
    row_label: str | None = None
    row_label_binding: EnumerationRowLabel | None = None
    raw_text: str = Field(min_length=1)
    raw_cells: list[EnumerationCell]
    values: list[EnumerationValue]
    span: GridSpan

    @model_validator(mode="after")
    def identity_and_cells_are_canonical(self) -> EnumerationRow:
        expected_id = _row_id(
            source_id=self.source_id,
            page=self.page,
            region_id=self.region_id,
            region_span=self.region_span,
            table_label=self.table_label,
            span=self.span,
            raw_text=self.raw_text,
        )
        if self.row_id != expected_id:
            raise ValueError("row_id does not match the physical row identity")
        expected_values = _bind_values(
            row_id=self.row_id,
            raw_cells=self.raw_cells,
            headers=self.headers,
            tokens=[(value.raw, value.span) for value in self.values],
        )
        if self.values != expected_values:
            raise ValueError("row values do not match canonical cell/token/header bindings")
        if (self.row_label is None) != (self.row_label_binding is None):
            raise ValueError("row label and its physical binding must be present together")
        if self.row_label_binding is not None:
            if not _normalized_label_matches(self.row_label, self.row_label_binding.raw):
                raise ValueError("row label does not match its bound source cell")
            if (
                self.row_label_binding.state is RowLabelBindingState.DIRECT
                and self.row_label_binding.span.start_line != self.span.start_line
            ):
                raise ValueError("direct row label does not originate on the planned row")
            if (
                self.row_label_binding.state is RowLabelBindingState.INHERITED
                and self.row_label_binding.span.end_line >= self.span.start_line
            ):
                raise ValueError("inherited row label does not originate above the planned row")
        if self.input_sha256 != _row_input_sha256(self):
            raise ValueError("row input_sha256 does not match the complete row input")
        return self


class RowBatch(StrictModel):
    """A bounded set of rows from exactly one physical table."""

    schema_version: Literal["table-row-enumeration/0.3"] = ROW_ENUMERATION_SCHEMA_VERSION
    batch_id: str
    source_id: str
    page: int = Field(ge=1)
    region_id: str
    table_label: str | None = None
    caption: str | None = None
    caption_span: GridSpan | None = None
    headers: list[EnumerationHeader]
    rows: list[EnumerationRow] = Field(min_length=1)
    character_count: int = Field(ge=1)
    value_token_count: int = Field(ge=0)

    @model_validator(mode="after")
    def one_table_only(self) -> RowBatch:
        identities = {(row.source_id, row.page, row.region_id) for row in self.rows}
        if identities != {(self.source_id, self.page, self.region_id)}:
            raise ValueError("a row batch must contain rows from exactly one table")
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("a row batch must not repeat a planned row")
        return self


class RowTablePlan(StrictModel):
    """One table intersecting a selected result block, dense or otherwise."""

    source_id: str
    page: int = Field(ge=1)
    region_id: str
    region_span: GridSpan
    table_label: str | None = None
    dense: bool


class RowPlanTelemetry(StrictModel):
    tables_considered: int = Field(ge=0)
    dense_tables: int = Field(ge=0)
    rows_planned: int = Field(ge=0)
    unbatchable_rows: int = Field(ge=0)
    base_batches: int = Field(ge=0)
    expected_calls: int = Field(ge=0)
    maximum_calls: int = Field(ge=0)


class UnbatchableRow(StrictModel):
    """A planned row retained for review but never sent above an input limit."""

    row_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_id: str
    page: int = Field(ge=1)
    region_id: str
    character_count: int = Field(ge=1)
    value_token_count: int = Field(ge=0)
    max_characters_per_batch: int = Field(ge=128)
    max_value_tokens_per_batch: int = Field(ge=1)
    reasons: list[UnbatchableReason] = Field(min_length=1)


class RowEnumerationPlan(StrictModel):
    schema_version: Literal["table-row-enumeration/0.3"] = ROW_ENUMERATION_SCHEMA_VERSION
    config: RowEnumerationConfig
    tables: list[RowTablePlan]
    rows: list[EnumerationRow]
    batches: list[RowBatch]
    unbatchable_rows: list[UnbatchableRow]
    telemetry: RowPlanTelemetry

    @model_validator(mode="after")
    def exact_canonical_partition(self) -> RowEnumerationPlan:
        table_keys = [(item.source_id, item.page, item.region_id) for item in self.tables]
        if len(table_keys) != len(set(table_keys)):
            raise ValueError("row plan contains duplicate table identities")
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("row plan contains duplicate row ids")
        batch_ids = [batch.batch_id for batch in self.batches]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("row plan contains duplicate batch ids")

        dense_tables = {
            (item.source_id, item.page, item.region_id) for item in self.tables if item.dense
        }
        row_tables = {(row.source_id, row.page, row.region_id) for row in self.rows}
        if row_tables != dense_tables:
            raise ValueError("dense table identities do not exactly match planned rows")
        table_by_key = {(item.source_id, item.page, item.region_id): item for item in self.tables}
        if any(
            row.region_span != table_by_key[(row.source_id, row.page, row.region_id)].region_span
            or row.table_label != table_by_key[(row.source_id, row.page, row.region_id)].table_label
            for row in self.rows
        ):
            raise ValueError("planned row table geometry does not match its table identity")

        canonical_batches, canonical_unbatchable = _canonical_partition(self.rows, self.config)
        if self.batches != canonical_batches or self.unbatchable_rows != canonical_unbatchable:
            raise ValueError("row plan batches do not form the exact canonical partition")

        expected_telemetry = RowPlanTelemetry(
            tables_considered=len(self.tables),
            dense_tables=len(dense_tables),
            rows_planned=len(self.rows),
            unbatchable_rows=len(self.unbatchable_rows),
            base_batches=len(self.batches),
            expected_calls=len(self.batches),
            maximum_calls=len(self.batches) * (1 + 2 * self.config.max_recovery_depth),
        )
        if self.telemetry != expected_telemetry:
            raise ValueError("row plan telemetry does not match the canonical plan")
        return self


class RowDispositionRecord(StrictModel):
    row_id: str
    disposition: RowDisposition
    candidates: list[CandidateObservation]
    note: str | None = None

    @model_validator(mode="after")
    def candidate_count_matches_disposition(self) -> RowDispositionRecord:
        if self.disposition is RowDisposition.RESULT and not self.candidates:
            raise ValueError("result disposition requires at least one candidate")
        if self.disposition is not RowDisposition.RESULT and self.candidates:
            raise ValueError("abstention dispositions must not contain candidates")
        return self


class RowTerminalRecord(StrictModel):
    """The sole final owner of one planned row."""

    row_id: str
    state: RowTerminalState
    disposition: RowDispositionRecord | None = None
    unsupported_reasons: list[UnbatchableReason] = Field(default_factory=list)
    unresolved_reason: str | None = None

    @model_validator(mode="after")
    def state_matches_payload(self) -> RowTerminalRecord:
        disposition_states = {
            RowTerminalState.RESULT: RowDisposition.RESULT,
            RowTerminalState.NOT_RESULT: RowDisposition.NOT_RESULT,
            RowTerminalState.UNCERTAIN: RowDisposition.UNCERTAIN,
        }
        expected_disposition = disposition_states.get(self.state)
        if expected_disposition is not None:
            if (
                self.disposition is None
                or self.disposition.row_id != self.row_id
                or self.disposition.disposition is not expected_disposition
                or self.unsupported_reasons
                or self.unresolved_reason is not None
            ):
                raise ValueError("terminal disposition payload does not match its state")
        elif self.state is RowTerminalState.UNRESOLVED:
            if self.disposition is not None or self.unsupported_reasons:
                raise ValueError("unresolved terminal rows cannot carry disposition/support data")
        elif self.state is RowTerminalState.UNSUPPORTED and (
            self.disposition is not None or not self.unsupported_reasons
        ):
            raise ValueError("unsupported terminal rows require typed unsupported reasons")
        return self


class RowProtocolEvent(StrictModel):
    """Attempt-scoped provider protocol output that never owns a planned row."""

    event: Literal["unknown_row_id"] = "unknown_row_id"
    batch_id: str
    depth: int = Field(ge=0, le=1)
    row_id: str


class RowTerminalCounts(StrictModel):
    planned: int = Field(ge=0)
    result: int = Field(ge=0)
    not_result: int = Field(ge=0)
    uncertain: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    unsupported: int = Field(ge=0)

    @model_validator(mode="after")
    def states_sum_to_plan(self) -> RowTerminalCounts:
        if (
            self.result + self.not_result + self.uncertain + self.unresolved + self.unsupported
            != self.planned
        ):
            raise ValueError("row terminal counts do not sum to planned rows")
        return self


class RowTerminalLedger(StrictModel):
    schema_version: Literal["row-terminal-states/0.1"] = ROW_TERMINAL_SCHEMA_VERSION
    paper_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_by_row: dict[str, RowTerminalRecord]
    protocol_events: list[RowProtocolEvent] = Field(default_factory=list)
    invalid_row_reasons: dict[str, str] = Field(default_factory=dict)
    counts: RowTerminalCounts

    @model_validator(mode="after")
    def keys_and_counts_match(self) -> RowTerminalLedger:
        if any(key != value.row_id for key, value in self.terminal_by_row.items()):
            raise ValueError("row terminal ledger key does not match terminal row_id")
        actual = {state: 0 for state in RowTerminalState}
        for terminal in self.terminal_by_row.values():
            actual[terminal.state] += 1
        expected = RowTerminalCounts(
            planned=len(self.terminal_by_row),
            result=actual[RowTerminalState.RESULT],
            not_result=actual[RowTerminalState.NOT_RESULT],
            uncertain=actual[RowTerminalState.UNCERTAIN],
            unresolved=actual[RowTerminalState.UNRESOLVED],
            unsupported=actual[RowTerminalState.UNSUPPORTED],
        )
        if self.counts != expected:
            raise ValueError("row terminal counts do not match terminal records")
        return self


class RowAttemptTelemetry(StrictModel):
    batch_id: str
    depth: int = Field(ge=0, le=1)
    row_ids: list[str]
    status: str
    resolved_row_ids: list[str] = Field(default_factory=list)
    unresolved_row_ids: list[str] = Field(default_factory=list)
    unknown_row_ids: list[str] = Field(default_factory=list)
    completed_provider_call: bool = False


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_identity_payload(
    *,
    source_id: str,
    page: int,
    region_id: str,
    region_span: GridSpan,
    table_label: str | None,
    span: GridSpan,
    raw_text: str,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "page": page,
        "region_id": region_id,
        "region_span": region_span.model_dump(mode="json"),
        "table_label": table_label,
        "row_span": span.model_dump(mode="json"),
        "raw_text": raw_text,
    }


def _row_id(
    *,
    source_id: str,
    page: int,
    region_id: str,
    region_span: GridSpan,
    table_label: str | None,
    span: GridSpan,
    raw_text: str,
) -> str:
    return (
        "trow_"
        + _canonical_hash(
            _row_identity_payload(
                source_id=source_id,
                page=page,
                region_id=region_id,
                region_span=region_span,
                table_label=table_label,
                span=span,
                raw_text=raw_text,
            )
        )[:20]
    )


def _header_id(level: int, raw: str, span: GridSpan) -> str:
    return (
        "theader_"
        + _canonical_hash(
            {
                "level": level,
                "raw": raw,
                "span": span.model_dump(mode="json"),
            }
        )[:20]
    )


def _row_label_cell_id(raw: str, span: GridSpan) -> str:
    return (
        "tlabel_"
        + _canonical_hash(
            {
                "raw": raw,
                "span": span.model_dump(mode="json"),
            }
        )[:20]
    )


def _normalized_label_matches(label: str | None, raw: str) -> bool:
    if label is None:
        return False
    normalized_label = re.sub(r"\W+", " ", label.casefold()).strip()
    normalized_raw = re.sub(r"\W+", " ", raw.casefold()).strip()
    return bool(normalized_label) and (
        normalized_label == normalized_raw or normalized_label in normalized_raw
    )


def _numeric_token_id(row_id: str, raw: str, span: GridSpan) -> str:
    return (
        "ttoken_"
        + _canonical_hash(
            {
                "row_id": row_id,
                "raw": raw,
                "span": span.model_dump(mode="json"),
            }
        )[:20]
    )


def _physical_cell_id(
    row_id: str,
    raw: str,
    span: GridSpan,
) -> str:
    return (
        "tcell_"
        + _canonical_hash(
            {
                "row_id": row_id,
                "raw": raw,
                "span": span.model_dump(mode="json"),
            }
        )[:20]
    )


def _row_input_payload(
    *,
    source_id: str,
    page: int,
    region_id: str,
    region_span: GridSpan,
    table_label: str | None,
    caption: str | None,
    caption_span: GridSpan | None,
    headers: list[EnumerationHeader],
    row_label: str | None,
    row_label_binding: EnumerationRowLabel | None,
    raw_text: str,
    raw_cells: list[EnumerationCell],
    values: list[EnumerationValue],
    span: GridSpan,
) -> dict[str, object]:
    return {
        **_row_identity_payload(
            source_id=source_id,
            page=page,
            region_id=region_id,
            region_span=region_span,
            table_label=table_label,
            span=span,
            raw_text=raw_text,
        ),
        "caption": caption,
        "caption_span": caption_span.model_dump(mode="json") if caption_span else None,
        "headers": [item.model_dump(mode="json") for item in headers],
        "row_label": row_label,
        "row_label_binding": (
            row_label_binding.model_dump(mode="json") if row_label_binding else None
        ),
        "raw_cells": [item.model_dump(mode="json") for item in raw_cells],
        "values": [item.model_dump(mode="json") for item in values],
    }


def _row_input_sha256(row: EnumerationRow) -> str:
    return _canonical_hash(
        _row_input_payload(
            source_id=row.source_id,
            page=row.page,
            region_id=row.region_id,
            region_span=row.region_span,
            table_label=row.table_label,
            caption=row.caption,
            caption_span=row.caption_span,
            headers=row.headers,
            row_label=row.row_label,
            row_label_binding=row.row_label_binding,
            raw_text=row.raw_text,
            raw_cells=row.raw_cells,
            values=row.values,
            span=row.span,
        )
    )


def _tight_row_text(page: PageFragment, region: Region, line: int) -> tuple[str, GridSpan]:
    lines = page.text.splitlines()
    if line > len(lines):
        raise ValueError(f"indexed row line {line} is outside page {page.page}")
    page_line = lines[line - 1]
    lower = (region.span.column_start or 1) - 1
    upper = min(len(page_line), region.span.column_end or len(page_line))
    window = page_line[lower:upper]
    leading = len(window) - len(window.lstrip())
    trailing = len(window.rstrip())
    if trailing <= leading:
        raise ValueError(f"indexed row line {line} contains no table text")
    raw_text = window[leading:trailing]
    column_start = lower + leading + 1
    column_end = lower + trailing
    return raw_text, GridSpan(
        start_line=line,
        end_line=line,
        column_start=column_start,
        column_end=column_end,
    )


def _row_intersects_block(
    page: PageFragment,
    region: Region,
    row: TableRow,
    block: ResultBlock,
) -> bool:
    if not block.body_start_line <= row.line <= block.body_end_line:
        return False
    if block.source_column_start is None or block.source_column_end is None:
        return True
    _, row_span = _tight_row_text(page, region, row.line)
    assert row_span.column_start is not None and row_span.column_end is not None
    return not (
        row_span.column_end < block.source_column_start
        or row_span.column_start > block.source_column_end
    )


def _spanned_items(
    raw_text: str,
    span: GridSpan,
    pattern: re.Pattern[str],
) -> list[tuple[str, GridSpan]]:
    assert span.column_start is not None
    return [
        (
            match.group(0),
            GridSpan(
                start_line=span.start_line,
                end_line=span.end_line,
                column_start=span.column_start + match.start(),
                column_end=span.column_start + match.end() - 1,
            ),
        )
        for match in pattern.finditer(raw_text)
    ]


def _spans_overlap(left: GridSpan, right: GridSpan) -> bool:
    if left.column_start is None or left.column_end is None:
        return False
    if right.column_start is None or right.column_end is None:
        return False
    return not (left.column_end < right.column_start or right.column_end < left.column_start)


def _header_path(
    *,
    headers: list[EnumerationHeader],
    raw_cells: list[EnumerationCell],
    tokens: list[tuple[str, GridSpan]],
    value_ordinal: int,
) -> list[EnumerationHeaderBinding]:
    value_count = len(tokens)
    first_value_span = tokens[0][1]
    label_cell = next(
        (
            cell
            for cell in raw_cells
            if cell.span.column_end is not None
            and first_value_span.column_start is not None
            and cell.span.column_end < first_value_span.column_start
        ),
        None,
    )
    bindings: list[EnumerationHeaderBinding] = []
    for header in headers:
        columns = [
            column
            for column in header.columns
            if label_cell is None or not _spans_overlap(column.span, label_cell.span)
        ]
        if not columns:
            bindings.append(
                EnumerationHeaderBinding(
                    level=header.level,
                    state=HeaderBindingState.MISSING,
                    headers=[],
                )
            )
            continue
        if len(columns) == value_count:
            bindings.append(
                EnumerationHeaderBinding(
                    level=header.level,
                    state=HeaderBindingState.RESOLVED,
                    headers=[columns[value_ordinal - 1]],
                )
            )
            continue
        if len(columns) < value_count and value_count % len(columns) == 0:
            values_per_header = value_count // len(columns)
            index = min((value_ordinal - 1) // values_per_header, len(columns) - 1)
            bindings.append(
                EnumerationHeaderBinding(
                    level=header.level,
                    state=HeaderBindingState.MERGED,
                    headers=[columns[index]],
                )
            )
            continue
        token_span = tokens[value_ordinal - 1][1]
        overlapping = [column for column in columns if _spans_overlap(column.span, token_span)]
        bindings.append(
            EnumerationHeaderBinding(
                level=header.level,
                state=HeaderBindingState.AMBIGUOUS,
                headers=overlapping or columns,
            )
        )
    return bindings


def _bind_values(
    *,
    row_id: str,
    raw_cells: list[EnumerationCell],
    headers: list[EnumerationHeader],
    tokens: list[tuple[str, GridSpan]],
) -> list[EnumerationValue]:
    values: list[EnumerationValue] = []
    for ordinal, (raw, span) in enumerate(tokens, start=1):
        containers = [
            cell
            for cell in raw_cells
            if cell.span.start_line <= span.start_line <= cell.span.end_line
            and cell.span.column_start is not None
            and cell.span.column_end is not None
            and span.column_start is not None
            and span.column_end is not None
            and cell.span.column_start <= span.column_start
            and cell.span.column_end >= span.column_end
        ]
        if len(containers) == 1:
            container = containers[0]
            contained_tokens = [
                item
                for item in tokens
                if item[1].column_start is not None
                and item[1].column_end is not None
                and container.span.column_start is not None
                and container.span.column_end is not None
                and container.span.column_start <= item[1].column_start
                and container.span.column_end >= item[1].column_end
            ]
            cell_state = (
                PhysicalCellState.EXACT
                if len(contained_tokens) == 1
                else PhysicalCellState.SPLIT_VALUE_RUN
            )
        else:
            container = EnumerationCell(raw=raw, span=span)
            cell_state = PhysicalCellState.AMBIGUOUS
        values.append(
            EnumerationValue(
                cell_id=_physical_cell_id(
                    row_id,
                    container.raw,
                    container.span,
                ),
                cell_raw=container.raw,
                cell_span=container.span,
                cell_state=cell_state,
                numeric_token_id=_numeric_token_id(row_id, raw, span),
                ordinal=ordinal,
                raw=raw,
                span=span,
                header_path=_header_path(
                    headers=headers,
                    raw_cells=raw_cells,
                    tokens=tokens,
                    value_ordinal=ordinal,
                ),
            )
        )
    return values


def _header(
    page: PageFragment,
    region: Region,
    row: TableRow,
    *,
    level: int,
) -> EnumerationHeader:
    raw_text, span = _tight_row_text(page, region, row.line)
    return EnumerationHeader(
        level=level,
        raw_text=raw_text,
        columns=[
            EnumerationHeaderCell(
                header_id=_header_id(level, raw, item_span),
                level=level,
                raw=raw,
                span=item_span,
            )
            for raw, item_span in _spanned_items(raw_text, span, _CELL)
        ],
        span=span,
    )


def _row_label_binding(
    page: PageFragment,
    region: Region,
    indexed: TableRow,
) -> EnumerationRowLabel | None:
    """Locate the exact direct or inherited cell supporting a row's system label."""

    label = indexed.effective_row_label
    if label is None:
        return None
    if indexed.row_label is not None:
        source = indexed
        state = RowLabelBindingState.DIRECT
    else:
        prior = [
            row
            for row in region.rows
            if not row.is_header
            and row.line < indexed.line
            and row.row_label is not None
            and _normalized_label_matches(label, row.row_label)
        ]
        if not prior:
            raise ValueError("inherited row label has no physical source row")
        source = max(prior, key=lambda row: row.line)
        state = RowLabelBindingState.INHERITED

    source_text, source_span = _tight_row_text(page, region, source.line)
    matches = [
        (raw, span)
        for raw, span in _spanned_items(source_text, source_span, _CELL)
        if _normalized_label_matches(label, raw)
    ]
    if not matches:
        raise ValueError("effective row label has no matching physical source cell")
    raw, span = min(matches, key=lambda item: (len(item[0]), item[1].column_start or 0))
    return EnumerationRowLabel(
        cell_id=_row_label_cell_id(raw, span),
        raw=raw,
        span=span,
        state=state,
    )


def _row(
    *,
    page: PageFragment,
    region: Region,
    indexed: TableRow,
    headers: list[EnumerationHeader],
) -> EnumerationRow:
    raw_text, span = _tight_row_text(page, region, indexed.line)
    row_label_binding = _row_label_binding(page, region, indexed)
    raw_cells = [
        EnumerationCell(raw=raw, span=item_span)
        for raw, item_span in _spanned_items(raw_text, span, _CELL)
    ]
    row_id = _row_id(
        source_id=page.source_id,
        page=page.page,
        region_id=region.region_id,
        region_span=region.span,
        table_label=region.table_label,
        span=span,
        raw_text=raw_text,
    )
    tokens = _spanned_items(raw_text, span, _VALUE)
    if (
        row_label_binding is not None
        and row_label_binding.state is RowLabelBindingState.DIRECT
        and row_label_binding.span.column_end is not None
    ):
        tokens = [
            token
            for token in tokens
            if token[1].column_start is None
            or token[1].column_start > row_label_binding.span.column_end
        ]
    values = _bind_values(
        row_id=row_id,
        raw_cells=raw_cells,
        headers=headers,
        tokens=tokens,
    )
    caption: Caption | None = region.caption
    input_sha256 = _canonical_hash(
        _row_input_payload(
            source_id=page.source_id,
            page=page.page,
            region_id=region.region_id,
            region_span=region.span,
            table_label=region.table_label,
            caption=caption.text if caption else None,
            caption_span=caption.span if caption else None,
            headers=headers,
            row_label=indexed.effective_row_label,
            row_label_binding=row_label_binding,
            raw_text=raw_text,
            raw_cells=raw_cells,
            values=values,
            span=span,
        )
    )
    return EnumerationRow(
        row_id=row_id,
        input_sha256=input_sha256,
        source_id=page.source_id,
        page=page.page,
        region_id=region.region_id,
        region_span=region.span,
        table_label=region.table_label,
        caption=caption.text if caption else None,
        caption_span=caption.span if caption else None,
        headers=headers,
        row_label=indexed.effective_row_label,
        row_label_binding=row_label_binding,
        raw_text=raw_text,
        raw_cells=raw_cells,
        values=values,
        span=span,
    )


def _context_characters(rows: list[EnumerationRow]) -> int:
    first = rows[0]
    return len(first.caption or "") + sum(len(header.raw_text) for header in first.headers)


def make_row_batch(rows: Iterable[EnumerationRow]) -> RowBatch:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot create an empty row batch")
    first = materialized[0]
    identity = {
        "source_id": first.source_id,
        "page": first.page,
        "region_id": first.region_id,
        "row_ids": [row.row_id for row in materialized],
        "row_input_sha256": [row.input_sha256 for row in materialized],
    }
    return RowBatch(
        batch_id="rbatch_" + _canonical_hash(identity)[:20],
        source_id=first.source_id,
        page=first.page,
        region_id=first.region_id,
        table_label=first.table_label,
        caption=first.caption,
        caption_span=first.caption_span,
        headers=first.headers,
        rows=materialized,
        character_count=_context_characters(materialized)
        + sum(len(row.raw_text) for row in materialized),
        value_token_count=sum(len(row.values) for row in materialized),
    )


def _unbatchable_row(
    row: EnumerationRow,
    config: RowEnumerationConfig,
) -> UnbatchableRow | None:
    singleton = make_row_batch([row])
    reasons: list[UnbatchableReason] = []
    if singleton.character_count > config.max_characters_per_batch:
        reasons.append(UnbatchableReason.CHARACTER_LIMIT)
    if singleton.value_token_count > config.max_value_tokens_per_batch:
        reasons.append(UnbatchableReason.VALUE_TOKEN_LIMIT)
    if not reasons:
        return None
    return UnbatchableRow(
        row_id=row.row_id,
        input_sha256=row.input_sha256,
        source_id=row.source_id,
        page=row.page,
        region_id=row.region_id,
        character_count=singleton.character_count,
        value_token_count=singleton.value_token_count,
        max_characters_per_batch=config.max_characters_per_batch,
        max_value_tokens_per_batch=config.max_value_tokens_per_batch,
        reasons=reasons,
    )


def _bounded_batches(
    rows: list[EnumerationRow],
    config: RowEnumerationConfig,
) -> tuple[list[RowBatch], list[UnbatchableRow]]:
    batches: list[RowBatch] = []
    unbatchable: list[UnbatchableRow] = []
    pending: list[EnumerationRow] = []
    for row in rows:
        skipped = _unbatchable_row(row, config)
        if skipped is not None:
            if pending:
                batches.append(make_row_batch(pending))
                pending = []
            unbatchable.append(skipped)
            continue
        proposed = [*pending, row]
        proposed_characters = _context_characters(proposed) + sum(
            len(item.raw_text) for item in proposed
        )
        proposed_values = sum(len(item.values) for item in proposed)
        exceeds = (
            len(proposed) > config.max_rows_per_batch
            or proposed_values > config.max_value_tokens_per_batch
            or proposed_characters > config.max_characters_per_batch
        )
        if pending and exceeds:
            batches.append(make_row_batch(pending))
            pending = []
        pending.append(row)
    if pending:
        batches.append(make_row_batch(pending))
    return batches, unbatchable


def _canonical_partition(
    rows: list[EnumerationRow],
    config: RowEnumerationConfig,
) -> tuple[list[RowBatch], list[UnbatchableRow]]:
    batches: list[RowBatch] = []
    unbatchable: list[UnbatchableRow] = []
    for _, grouped in groupby(
        rows,
        key=lambda row: (row.source_id, row.page, row.region_id),
    ):
        table_batches, table_unbatchable = _bounded_batches(list(grouped), config)
        batches.extend(table_batches)
        unbatchable.extend(table_unbatchable)
    return batches, unbatchable


def build_row_enumeration_plan(
    layout: PdfLayout,
    blocks: Iterable[ResultBlock],
    config: RowEnumerationConfig | None = None,
) -> RowEnumerationPlan:
    """Plan stable dense-table rows that selected block bodies actually exposed."""

    config = config or RowEnumerationConfig()
    blocks_by_page: dict[int, list[ResultBlock]] = defaultdict(list)
    for block in blocks:
        blocks_by_page[block.page].append(block)
    pages = {page.page: page for page in layout.pages}
    indexes = build_region_index(layout)
    tables: list[RowTablePlan] = []
    rows: list[EnumerationRow] = []
    batches: list[RowBatch] = []
    unbatchable_rows: list[UnbatchableRow] = []
    tables_considered = 0
    dense_tables = 0
    for page_number, page_index in indexes.items():
        page = pages[page_number]
        for region in page_index.regions:
            if region.kind is not RegionKind.TABLE or region.in_references:
                continue
            shown = [
                row
                for row in region.rows
                if not row.is_header
                and any(
                    _row_intersects_block(page, region, row, block)
                    for block in blocks_by_page.get(page_number, [])
                )
            ]
            if not shown:
                continue
            tables_considered += 1
            dense = len(shown) >= config.min_dense_table_rows
            tables.append(
                RowTablePlan(
                    source_id=page.source_id,
                    page=page.page,
                    region_id=region.region_id,
                    region_span=region.span,
                    table_label=region.table_label,
                    dense=dense,
                )
            )
            if not dense:
                continue
            dense_tables += 1
            headers = [
                _header(page, region, row, level=level)
                for level, row in enumerate(
                    (row for row in region.rows if row.is_header),
                    start=1,
                )
            ]
            table_rows = [
                _row(page=page, region=region, indexed=row, headers=headers) for row in shown
            ]
            rows.extend(table_rows)
            table_batches, table_unbatchable = _bounded_batches(table_rows, config)
            batches.extend(table_batches)
            unbatchable_rows.extend(table_unbatchable)
    return RowEnumerationPlan(
        config=config,
        tables=tables,
        rows=rows,
        batches=batches,
        unbatchable_rows=unbatchable_rows,
        telemetry=RowPlanTelemetry(
            tables_considered=tables_considered,
            dense_tables=dense_tables,
            rows_planned=len(rows),
            unbatchable_rows=len(unbatchable_rows),
            base_batches=len(batches),
            expected_calls=len(batches),
            maximum_calls=len(batches) * (1 + 2 * config.max_recovery_depth),
        ),
    )


def recovery_batches(batch: RowBatch, unresolved_row_ids: Iterable[str]) -> list[RowBatch]:
    """Return at most two changed-input child batches for unresolved rows."""

    wanted = set(unresolved_row_ids)
    unresolved = [row for row in batch.rows if row.row_id in wanted]
    if not unresolved or len(batch.rows) == 1:
        return []
    if len(unresolved) == 1:
        return [make_row_batch(unresolved)]
    midpoint = len(unresolved) // 2
    return [make_row_batch(unresolved[:midpoint]), make_row_batch(unresolved[midpoint:])]
