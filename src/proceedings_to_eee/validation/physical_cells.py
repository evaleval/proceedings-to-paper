"""Conservative binding of table candidates to exact layout-grid value cells.

The binding in this module is deliberately narrower than semantic resolution.  It
answers only whether a candidate's exact printed raw value can be located uniquely in
one physical table row named by its evidence.  In particular, a repeated raw token in
the same row is ambiguous: model-proposed column wording is not treated as geometry.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from proceedings_to_eee.domain.observation import CandidateObservation, EvidenceAnchor
from proceedings_to_eee.domain.status import EvidenceKind
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.region_index import (
    PageRegionIndex,
    Region,
    RegionKind,
    TableRow,
    build_region_index,
)

_VALUE_TOKEN = re.compile(r"(?<![\w@-])[+\-\u2212]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:\s*%)?(?!\w)")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


class PhysicalCellBindingStatus(StrEnum):
    """Whether structural evidence identifies exactly one physical value cell."""

    BOUND = "bound"
    UNLOCATED = "unlocated"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class PhysicalCellIdentity:
    """Exact identity of one printed numeric token in a layout-indexed table."""

    paper_id: str
    source_id: str
    page: int
    page_text_sha256: str
    region_id: str
    table_start_line: int
    table_end_line: int
    table_column_start: int | None
    table_column_end: int | None
    row_line: int
    row_ordinal: int
    row_label: str | None
    value_ordinal: int
    value_column_start: int
    value_column_end: int
    raw: str


@dataclass(frozen=True)
class PhysicalCellBinding:
    """A conservative binding result, including why a candidate was not bound."""

    status: PhysicalCellBindingStatus
    identity: PhysicalCellIdentity | None = None
    reason: str | None = None


@dataclass(frozen=True)
class _LocatedRow:
    page: PageFragment
    region: Region
    row: TableRow
    row_ordinal: int
    raw_text: str
    column_start: int


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\u2212", "-")
    return re.sub(r"\s+", " ", normalized).strip()


def _normalized_label(value: str | None) -> str:
    if value is None:
        return ""
    return _NON_ALPHANUMERIC.sub(" ", _normalized_text(value).casefold()).strip()


def _normalized_raw_value(value: str) -> str:
    normalized = _normalized_text(value)
    return re.sub(r"\s+%$", "%", normalized)


def _row_text(page: PageFragment, region: Region, line: int) -> tuple[str, int] | None:
    lines = page.text.splitlines()
    if line > len(lines):
        return None
    page_line = lines[line - 1]
    lower = (region.span.column_start or 1) - 1
    upper = min(len(page_line), region.span.column_end or len(page_line))
    window = page_line[lower:upper]
    leading = len(window) - len(window.lstrip())
    trailing = len(window.rstrip())
    if trailing <= leading:
        return None
    return window[leading:trailing], lower + leading + 1


def _quote_matches_row(quote: str, raw_text: str) -> bool:
    normalized_quote = _normalized_text(quote).casefold()
    normalized_row = _normalized_text(raw_text).casefold()
    if not normalized_quote or not normalized_row:
        return False
    return normalized_quote in normalized_row or normalized_row in normalized_quote


def _labels_compatible(anchor_label: str, table_label: str) -> bool:
    anchor = _normalized_label(anchor_label)
    table = _normalized_label(table_label)
    if not anchor or not table:
        return False
    anchor_tokens = set(anchor.split())
    table_tokens = set(table.split())
    return anchor == table or anchor_tokens <= table_tokens or table_tokens <= anchor_tokens


def _matching_rows(
    anchor: EvidenceAnchor,
    page: PageFragment,
    index: PageRegionIndex,
) -> tuple[list[_LocatedRow], str | None]:
    rows: list[_LocatedRow] = []
    for region in index.regions:
        if region.kind is not RegionKind.TABLE or region.in_references:
            continue
        for row_ordinal, row in enumerate(region.rows, start=1):
            if row.is_header:
                continue
            positioned = _row_text(page, region, row.line)
            if positioned is None:
                continue
            raw_text, column_start = positioned
            if _quote_matches_row(anchor.quote, raw_text):
                rows.append(
                    _LocatedRow(
                        page=page,
                        region=region,
                        row=row,
                        row_ordinal=row_ordinal,
                        raw_text=raw_text,
                        column_start=column_start,
                    )
                )

    if not rows:
        return [], None

    if anchor.row:
        wanted_row = _normalized_label(anchor.row)
        matching_labels = [
            located
            for located in rows
            if _normalized_label(located.row.effective_row_label) == wanted_row
        ]
        if not matching_labels:
            return [], "evidence row label disagrees with the located table row"
        rows = matching_labels

    labelled_regions = [located for located in rows if located.region.table_label]
    if anchor.label and labelled_regions:
        matching_tables = [
            located
            for located in labelled_regions
            if located.region.table_label
            and _labels_compatible(anchor.label, located.region.table_label)
        ]
        if not matching_tables:
            return [], "evidence table label disagrees with the located table"
        rows = matching_tables

    return rows, None


def _label_end(row: _LocatedRow) -> int:
    """Return the local offset after a printed row label, excluding its digits."""

    label = row.row.row_label
    if not label:
        return 0
    match = re.match(re.escape(label), row.raw_text, re.IGNORECASE)
    return match.end() if match else 0


def _bind_anchor(
    *,
    paper_id: str,
    raw_value: str,
    anchor: EvidenceAnchor,
    page: PageFragment,
    index: PageRegionIndex,
) -> PhysicalCellBinding:
    rows, mismatch = _matching_rows(anchor, page, index)
    if mismatch:
        return PhysicalCellBinding(PhysicalCellBindingStatus.AMBIGUOUS, reason=mismatch)
    if not rows:
        return PhysicalCellBinding(
            PhysicalCellBindingStatus.UNLOCATED,
            reason="evidence quote did not locate a table row",
        )
    if len(rows) > 1:
        return PhysicalCellBinding(
            PhysicalCellBindingStatus.AMBIGUOUS,
            reason="evidence quote locates more than one physical table row",
        )

    located = rows[0]
    label_end = _label_end(located)
    tokens = [match for match in _VALUE_TOKEN.finditer(located.raw_text) if match.end() > label_end]
    matches = [
        (ordinal, token)
        for ordinal, token in enumerate(tokens, start=1)
        if _normalized_raw_value(token.group(0)) == _normalized_raw_value(raw_value)
    ]
    if not matches:
        return PhysicalCellBinding(
            PhysicalCellBindingStatus.UNLOCATED,
            reason="candidate raw value is absent from the located table row",
        )
    if len(matches) > 1:
        return PhysicalCellBinding(
            PhysicalCellBindingStatus.AMBIGUOUS,
            reason="candidate raw value occurs in more than one cell of the located row",
        )

    value_ordinal, token = matches[0]
    region = located.region
    page_hash = hashlib.sha256(page.text.encode("utf-8")).hexdigest()
    return PhysicalCellBinding(
        PhysicalCellBindingStatus.BOUND,
        identity=PhysicalCellIdentity(
            paper_id=paper_id,
            source_id=anchor.source_id,
            page=anchor.page,
            page_text_sha256=page_hash,
            region_id=region.region_id,
            table_start_line=region.span.start_line,
            table_end_line=region.span.end_line,
            table_column_start=region.span.column_start,
            table_column_end=region.span.column_end,
            row_line=located.row.line,
            row_ordinal=located.row_ordinal,
            row_label=located.row.effective_row_label,
            value_ordinal=value_ordinal,
            value_column_start=located.column_start + token.start(),
            value_column_end=located.column_start + token.end() - 1,
            raw=token.group(0),
        ),
    )


class PhysicalCellLocator:
    """Lazily index layouts and bind candidates without semantic guessing."""

    def __init__(self, layouts: dict[str, PdfLayout]) -> None:
        self._layouts = layouts
        self._indexes: dict[str, dict[int, PageRegionIndex]] = {}
        self._pages = {
            source_id: {page.page: page for page in layout.pages}
            for source_id, layout in layouts.items()
        }

    def _index(self, source_id: str) -> dict[int, PageRegionIndex] | None:
        layout = self._layouts.get(source_id)
        if layout is None:
            return None
        if source_id not in self._indexes:
            self._indexes[source_id] = build_region_index(layout)
        return self._indexes[source_id]

    def bind(self, candidate: CandidateObservation) -> PhysicalCellBinding:
        """Bind all table evidence for a candidate to one exact physical cell."""

        if candidate.value is None:
            return PhysicalCellBinding(
                PhysicalCellBindingStatus.UNLOCATED,
                reason="candidate has no reported value",
            )

        bound: set[PhysicalCellIdentity] = set()
        for anchor in candidate.evidence:
            if anchor.kind is not EvidenceKind.TABLE:
                continue
            index = self._index(anchor.source_id)
            page = self._pages.get(anchor.source_id, {}).get(anchor.page)
            page_index = index.get(anchor.page) if index is not None else None
            if page is None or page_index is None:
                continue
            result = _bind_anchor(
                paper_id=candidate.paper_id,
                raw_value=candidate.value.raw,
                anchor=anchor,
                page=page,
                index=page_index,
            )
            if result.status is PhysicalCellBindingStatus.AMBIGUOUS:
                return result
            if result.identity is not None:
                bound.add(result.identity)

        if not bound:
            return PhysicalCellBinding(
                PhysicalCellBindingStatus.UNLOCATED,
                reason="no table evidence uniquely binds the candidate value",
            )
        if len(bound) > 1:
            return PhysicalCellBinding(
                PhysicalCellBindingStatus.AMBIGUOUS,
                reason="candidate evidence points to more than one physical value cell",
            )
        return PhysicalCellBinding(PhysicalCellBindingStatus.BOUND, identity=bound.pop())
