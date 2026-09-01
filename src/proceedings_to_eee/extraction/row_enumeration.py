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

ROW_ENUMERATION_SCHEMA_VERSION = "table-row-enumeration/0.1"

_CELL = re.compile(r"\S(?:(?!\s{2,}).)*?(?=\s{2,}|$)")
_VALUE = re.compile(r"(?<![\w@-])[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:\s*%)?(?!\w)")


class RowDisposition(StrEnum):
    RESULT = "result"
    NOT_RESULT = "not_result"
    UNCERTAIN = "uncertain"


class UnbatchableReason(StrEnum):
    CHARACTER_LIMIT = "max_characters_per_batch"
    VALUE_TOKEN_LIMIT = "max_value_tokens_per_batch"


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


class EnumerationValue(StrictModel):
    raw: str = Field(min_length=1)
    span: GridSpan


class EnumerationHeader(StrictModel):
    raw_text: str = Field(min_length=1)
    columns: list[EnumerationCell]
    span: GridSpan


class EnumerationRow(StrictModel):
    """One physical indexed table row, bound to exact layout-grid coordinates."""

    schema_version: str = ROW_ENUMERATION_SCHEMA_VERSION
    row_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_id: str
    page: int = Field(ge=1)
    region_id: str
    table_label: str | None = None
    caption: str | None = None
    caption_span: GridSpan | None = None
    headers: list[EnumerationHeader]
    row_label: str | None = None
    raw_text: str = Field(min_length=1)
    raw_cells: list[EnumerationCell]
    values: list[EnumerationValue]
    span: GridSpan


class RowBatch(StrictModel):
    """A bounded set of rows from exactly one physical table."""

    schema_version: str = ROW_ENUMERATION_SCHEMA_VERSION
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
        return self


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
    schema_version: str = ROW_ENUMERATION_SCHEMA_VERSION
    config: RowEnumerationConfig
    rows: list[EnumerationRow]
    batches: list[RowBatch]
    unbatchable_rows: list[UnbatchableRow]
    telemetry: RowPlanTelemetry


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


def _header(page: PageFragment, region: Region, row: TableRow) -> EnumerationHeader:
    raw_text, span = _tight_row_text(page, region, row.line)
    return EnumerationHeader(
        raw_text=raw_text,
        columns=[
            EnumerationCell(raw=raw, span=item_span)
            for raw, item_span in _spanned_items(raw_text, span, _CELL)
        ],
        span=span,
    )


def _row(
    *,
    page: PageFragment,
    region: Region,
    indexed: TableRow,
    headers: list[EnumerationHeader],
) -> EnumerationRow:
    raw_text, span = _tight_row_text(page, region, indexed.line)
    raw_cells = [
        EnumerationCell(raw=raw, span=item_span)
        for raw, item_span in _spanned_items(raw_text, span, _CELL)
    ]
    values = [
        EnumerationValue(raw=raw, span=item_span)
        for raw, item_span in _spanned_items(raw_text, span, _VALUE)
    ]
    identity_payload = {
        "source_id": page.source_id,
        "page": page.page,
        "line": indexed.line,
        "column_start": span.column_start,
        "column_end": span.column_end,
        "raw_text": raw_text,
    }
    row_id = "trow_" + _canonical_hash(identity_payload)[:20]
    caption: Caption | None = region.caption
    input_payload = {
        **identity_payload,
        "region_id": region.region_id,
        "table_label": region.table_label,
        "caption": caption.text if caption else None,
        "caption_span": caption.span.model_dump(mode="json") if caption else None,
        "headers": [item.model_dump(mode="json") for item in headers],
        "row_label": indexed.effective_row_label,
        "raw_cells": [item.model_dump(mode="json") for item in raw_cells],
        "values": [item.model_dump(mode="json") for item in values],
    }
    return EnumerationRow(
        row_id=row_id,
        input_sha256=_canonical_hash(input_payload),
        source_id=page.source_id,
        page=page.page,
        region_id=region.region_id,
        table_label=region.table_label,
        caption=caption.text if caption else None,
        caption_span=caption.span if caption else None,
        headers=headers,
        row_label=indexed.effective_row_label,
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
            if len(shown) < config.min_dense_table_rows:
                continue
            dense_tables += 1
            headers = [_header(page, region, row) for row in region.rows if row.is_header]
            table_rows = [
                _row(page=page, region=region, indexed=row, headers=headers) for row in shown
            ]
            rows.extend(table_rows)
            table_batches, table_unbatchable = _bounded_batches(table_rows, config)
            batches.extend(table_batches)
            unbatchable_rows.extend(table_unbatchable)
    return RowEnumerationPlan(
        config=config,
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
