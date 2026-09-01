"""Deterministic caption and region index over ``pdftotext -layout`` page text.

An evidence quote alone says what a number is; it does not say where the number sits.
Whether a value stands in a table body or in a caption, which table it belongs to, which
row label governs it, and which section encloses it are all structural facts that the
layout text already contains and that nothing downstream could previously ask for.

This module answers those questions and nothing else. It classifies, it never decides:
no export policy, no attribution judgment, no notion of a target entity. It is
paper-agnostic, uses only local typography and lexical form, and reuses the segmenter's
own column-gutter rule via :func:`page_panels` so page geometry has one definition.

Coordinates are the layout grid the quotes themselves live in: one-based page line
ordinals and one-based inclusive page columns. A quote can therefore be located and the
answer handed back in the same coordinate system as the evidence anchors.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import Field, model_validator

from proceedings_to_eee.domain.observation import StrictModel
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.result_blocks import PagePanel, ResultBlockConfig, page_panels

REGION_INDEX_SCHEMA_VERSION = "page-region-index/0.1"

_MAX_QUOTE_WINDOW_LINES = 8

_CELL_SPLIT = re.compile(r"\s{2,}")
_VALUE_TOKEN = r"[\[\(<]?[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*%?[\]\)>*\u2020\u2021]?"
# A cell can hold a run of values the layout only single-spaced apart, as in the
# invented fixture ``Gauge Lumen .53 .64 .58``, so a value cell is one or more value
# tokens and nothing else.
_VALUE_CELL = re.compile(rf"^(?:{_VALUE_TOKEN})(?:\s+(?:{_VALUE_TOKEN}))*$")
_CONTAINS_NUMBER = re.compile(r"(?<![\w@])[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*%?(?!\w)")

_CAPTION_LABEL = re.compile(
    r"^\s*(?P<kind>table|figure|fig\.?|chart|plot|algorithm|listing)\s*"
    r"(?P<number>[A-Z]?\d+(?:[.\-]\d+)*|[IVXLCDM]+)\s*(?::|\.|[-–—])",
    re.IGNORECASE,
)

# "6.3.1 Fixture Assembly. During a toy run, we ..."; a numbered run-in
# heading whose body text continues on the same line.
_RUN_IN_HEADING = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)+)\s+(?P<title>[A-Z][^.]{1,80})\.\s+(?=[A-Z(])"
)
_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<number>(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*))[.)]?\s+(?P<title>\S.{0,90})$"
)
# Words that stay lowercase inside a genuine title. Any other lowercase-initial word
# means the line reads as a sentence, so it is body text or a numbered list item.
_TITLE_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "between",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "per",
        "the",
        "to",
        "under",
        "via",
        "vs",
        "with",
        "within",
    }
)

_SOLE_WORD_HEADING = re.compile(r"^[A-Z][A-Za-z-]{3,}\.?$")
_REFERENCE_HEADING = re.compile(r"^(references|bibliography|works cited)\b", re.IGNORECASE)
_APPENDIX_HEADING = re.compile(r"^(appendix|appendices|supplement)", re.IGNORECASE)

# Page furniture only when it also sits at the very top or bottom of the page.
_FURNITURE = re.compile(
    r"(?:^\s*\d{1,4}\s*$)"
    r"|(?:\b\d+\s*:\s*\d+\s*$)"
    r"|(?:publication date\s*:)"
    r"|(?:^\s*proc\.\s)"
    r"|(?:\barxiv\s*:\s*\d{4}\.\d{4,5})"
    r"|(?:\b(?:acm|ieee)\s+(?:isbn|reference format)\b)"
    r"|(?:^\s*(?:[ivxlcdm]{1,7})\s*$)",
    re.IGNORECASE,
)


class RegionKind(StrEnum):
    """Structural role of one contiguous run of lines inside one column panel."""

    HEADING = "heading"
    CAPTION = "caption"
    TABLE = "table"
    PROSE = "prose"
    REFERENCES = "references"
    PAGE_FURNITURE = "page_furniture"


class GridSpan(StrictModel):
    """A rectangle in the layout grid: page line ordinals and page columns."""

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    column_start: int | None = Field(default=None, ge=1)
    column_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_rectangle(self) -> GridSpan:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be at or after start_line")
        if (self.column_start is None) != (self.column_end is None):
            raise ValueError("a column window must provide both endpoints")
        if (
            self.column_start is not None
            and self.column_end is not None
            and self.column_end < self.column_start
        ):
            raise ValueError("column_end must be at or after column_start")
        return self


class TableRow(StrictModel):
    """One physical line of a table body, split on the layout's own column gutters."""

    line: int = Field(ge=1)
    cells: list[str]
    row_label: str | None = None
    effective_row_label: str | None = None
    is_header: bool = False
    numeric_cells: int = Field(ge=0)


class Caption(StrictModel):
    """A table or figure caption, joined across its wrapped lines."""

    label: str | None = None
    label_kind: str | None = None
    text: str = Field(min_length=1)
    span: GridSpan


class Region(StrictModel):
    """One contiguous, single-kind run of lines inside one column panel."""

    region_id: str
    kind: RegionKind
    span: GridSpan
    text: str
    section_path: list[str] = Field(default_factory=list)
    section_number: str | None = None
    in_references: bool = False
    in_appendix: bool = False
    caption: Caption | None = None
    caption_position: str | None = None
    table_label: str | None = None
    rows: list[TableRow] = Field(default_factory=list)


class PageRegionIndex(StrictModel):
    """Every region on one page, bound to the exact layout text it was built from."""

    schema_version: str = REGION_INDEX_SCHEMA_VERSION
    source_id: str
    page: int = Field(ge=1)
    page_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    panel_columns: list[tuple[int | None, int | None]] = Field(default_factory=list)
    regions: list[Region] = Field(default_factory=list)


class QuoteLocation(StrictModel):
    """Where one evidence quote sits, in the quote's own coordinate system."""

    region_id: str
    kind: RegionKind
    span: GridSpan
    section_path: list[str] = Field(default_factory=list)
    in_references: bool = False
    in_appendix: bool = False
    table_label: str | None = None
    caption_text: str | None = None
    caption_position: str | None = None
    row_label: str | None = None
    matched_lines: list[int] = Field(default_factory=list)


# ----------------------------------------------------------------------------------
# Line classification
# ----------------------------------------------------------------------------------


def _cells(text: str) -> list[str]:
    return [cell for cell in _CELL_SPLIT.split(text.strip()) if cell]


def _numeric_cell_count(cells: list[str]) -> int:
    return sum(bool(_VALUE_CELL.match(cell)) for cell in cells)


def _is_table_line(text: str) -> bool:
    """A line is tabular when the layout gutters split it into columns holding values.

    The value must stand in a column other than the first. Otherwise a numbered heading
    such as ``5   EVALUATION`` reads as a two-cell row whose first cell is a number.
    """

    stripped = text.strip()
    if not stripped or _CAPTION_LABEL.match(stripped):
        return False
    cells = _cells(stripped)
    if len(cells) < 2:
        return False
    return any(position > 0 for position, cell in enumerate(cells) if _VALUE_CELL.match(cell))


def _is_table_header_line(text: str) -> bool:
    """A column-aligned, value-free line: a header row rather than a sentence."""

    stripped = text.strip()
    if not stripped or _CAPTION_LABEL.match(stripped):
        return False
    cells = _cells(stripped)
    if len(cells) < 2 or _numeric_cell_count(cells) > 0:
        return False
    # A wrapped sentence also contains several words but no wide internal gutters.
    return all(len(cell.split()) <= 6 for cell in cells)


def _title_cased(title: str) -> bool:
    """True when every lowercase-initial word in the title is an ordinary function word.

    ``Aggregate Fixture Scores for Two Slices`` is a heading. ``Additionally, if each
    item uses a fixed seed`` is a numbered list item that only looks like one.
    """

    words = [word for word in re.findall(r"[A-Za-z][\w'-]*", title) if word]
    if not words:
        return False
    if all(word.isupper() for word in words):
        return True
    return all(word[0].isupper() or word.casefold() in _TITLE_FUNCTION_WORDS for word in words)


def _heading_of(text: str) -> tuple[str, str | None] | None:
    """Return (title, number) when a line opens a section, including run-in headings."""

    stripped = text.strip()
    if not stripped or len(stripped) > 200:
        return None
    if _CAPTION_LABEL.match(stripped):
        return None
    run_in = _RUN_IN_HEADING.match(stripped)
    if run_in:
        return run_in.group("title").strip(), run_in.group("number")
    # Trailing figures usually mean a data row or a running head, not a heading.
    long_or_numeric = len(stripped.split()) > 16 or _CONTAINS_NUMBER.search(stripped[-12:])
    if long_or_numeric and not _NUMBERED_HEADING.match(stripped):
        return None
    numbered = _NUMBERED_HEADING.match(stripped)
    if numbered and len(numbered.group("title").split()) <= 12:
        title = numbered.group("title").strip().rstrip(".")
        if title and title[0].isupper() and _title_cased(title):
            return title, numbered.group("number")
    letters = [character for character in stripped if character.isalpha()]
    if letters and len(stripped.split()) <= 12 and all(char.isupper() for char in letters):
        return stripped.rstrip("."), None
    # A line holding one capitalized word and nothing else is a heading in paper
    # layout. "References" is the case that matters most: missing it leaves a whole
    # bibliography classified as prose and its author lists reading as headings.
    if _SOLE_WORD_HEADING.match(stripped):
        return stripped.rstrip("."), None
    if (
        len(stripped.split()) in range(2, 9)
        and _title_cased(stripped)
        and stripped[0].isupper()
        and not _CONTAINS_NUMBER.search(stripped)
    ):
        return stripped.rstrip("."), None
    return None


def _exits_references(title: str, number: str | None) -> bool:
    """True only for a numbered section heading, which a bibliography does not contain.

    Deliberately narrow. An all-caps test looks reasonable and is wrong: a synthetic
    running head such as ``FIXTURE-RUN`` can repeat on every page, including reference
    pages, and would end the bibliography on its first continuation page. A four-digit
    publication year opening a reference entry can also look like a section number, so
    the numeric form is bounded to plausible section numbers.
    """

    del title
    if not number or not re.fullmatch(r"\d{1,2}(?:\.\d{1,2})*", number):
        return False
    return int(number.split(".")[0]) <= 30


def _is_furniture(text: str, *, first_or_last: bool) -> bool:
    stripped = text.strip()
    if not stripped or not first_or_last:
        return False
    if len(stripped) > 160:
        return False
    return bool(_FURNITURE.search(stripped))


# ----------------------------------------------------------------------------------
# Region construction
# ----------------------------------------------------------------------------------


def _panel_column(panel: PagePanel, offset: int) -> int:
    """Map a zero-based offset inside a panel line to a one-based page column."""

    return (panel.source_column_start or 1) + offset


def _line_span(panel: PagePanel, start: int, end: int, lines: list[str]) -> GridSpan:
    """Column window that tightly encloses the text of the given panel lines."""

    starts = [len(line) - len(line.lstrip()) for line in lines[start : end + 1] if line.strip()]
    ends = [len(line.rstrip()) for line in lines[start : end + 1] if line.strip()]
    if starts and ends:
        column_start = _panel_column(panel, min(starts))
        column_end = _panel_column(panel, max(ends) - 1)
    else:
        column_start = panel.source_column_start
        column_end = panel.source_column_end
    return GridSpan(
        start_line=start + 1,
        end_line=end + 1,
        column_start=column_start,
        column_end=column_end,
    )


def _classify_panel_lines(panel: PagePanel) -> list[RegionKind | None]:
    """Assign one kind per non-blank panel line. ``None`` marks a blank line."""

    lines = panel.lines
    non_blank = [index for index, line in enumerate(lines) if line.strip()]
    first = non_blank[0] if non_blank else None
    last = non_blank[-1] if non_blank else None
    kinds: list[RegionKind | None] = []
    for index, line in enumerate(lines):
        if not line.strip():
            kinds.append(None)
            continue
        if _is_furniture(line, first_or_last=index in {first, last}):
            kinds.append(RegionKind.PAGE_FURNITURE)
        elif _CAPTION_LABEL.match(line.strip()):
            kinds.append(RegionKind.CAPTION)
        elif _is_table_line(line):
            kinds.append(RegionKind.TABLE)
        elif _heading_of(line) is not None:
            kinds.append(RegionKind.HEADING)
        else:
            kinds.append(RegionKind.PROSE)
    _absorb_table_headers(lines, kinds)
    _absorb_wrapped_captions(lines, kinds)
    return kinds


def _absorb_table_headers(lines: list[str], kinds: list[RegionKind | None]) -> None:
    """Pull value-free aligned lines into the table they head or close.

    Stacked headers are common: a spanning group row, a sub-group row, and a column row
    can sit above the first data line. Absorption therefore runs to a fixpoint so each
    newly absorbed row lets the one above it absorb in turn.
    """

    while _absorb_table_headers_once(lines, kinds):
        pass


def _absorb_table_headers_once(lines: list[str], kinds: list[RegionKind | None]) -> bool:
    changed = False
    for index, kind in enumerate(kinds):
        if kind not in {RegionKind.PROSE, RegionKind.HEADING}:
            continue
        if not _is_table_header_line(lines[index]):
            continue
        following = next(
            (kinds[after] for after in range(index + 1, len(kinds)) if kinds[after] is not None),
            None,
        )
        preceding = next(
            (kinds[before] for before in range(index - 1, -1, -1) if kinds[before] is not None),
            None,
        )
        if RegionKind.TABLE in {following, preceding}:
            kinds[index] = RegionKind.TABLE
            changed = True
    return changed


def _absorb_wrapped_captions(lines: list[str], kinds: list[RegionKind | None]) -> None:
    """Continue a caption onto its immediately following wrapped line."""

    for index, kind in enumerate(kinds):
        if kind is not RegionKind.CAPTION:
            continue
        follower = index + 1
        if follower >= len(kinds) or kinds[follower] is not RegionKind.PROSE:
            continue
        if not lines[index].rstrip().endswith("."):
            kinds[follower] = RegionKind.CAPTION


def _group_regions(kinds: list[RegionKind | None], max_blank_gap: int) -> list[tuple[int, int]]:
    """Group same-kind lines, bridging short blank gaps inside tables only."""

    groups: list[tuple[int, int]] = []
    index = 0
    while index < len(kinds):
        if kinds[index] is None:
            index += 1
            continue
        kind = kinds[index]
        start = end = index
        probe = index + 1
        while probe < len(kinds):
            if kinds[probe] == kind:
                end = probe
                probe += 1
                continue
            if kinds[probe] is None and kind is RegionKind.TABLE:
                gap = probe
                while gap < len(kinds) and kinds[gap] is None:
                    gap += 1
                if gap - probe <= max_blank_gap and gap < len(kinds) and kinds[gap] == kind:
                    probe = gap
                    continue
            break
        groups.append((start, end))
        index = end + 1
    return groups


_TRAILING_VALUES = re.compile(r"(?:\s+[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*%?){2,}$")


def _row_label_from_cell(cell: str) -> str:
    """Separate a row label from values the layout only single-spaced away from it.

    ``Gauge Lumen .53 .64 .58`` is one gutter-delimited cell but two things. Two or more
    trailing value tokens are required before any are stripped, so a label that merely
    ends in a number, such as ``Model 1``, is left intact.
    """

    stripped = _TRAILING_VALUES.sub("", cell).strip()
    return stripped or cell


_ORDINAL_CELL = re.compile(r"^\d{1,4}$")
_LABEL_COLUMN_TOLERANCE = 2


def _has_ordinal_first_column(cells: list[str]) -> bool:
    """True when the first cell is a bare row number and the second names the row.

    ``7  Packets archived  -2.7%  0.008`` indexes its rows. Requiring the second cell
    to be non-numeric keeps a genuine numeric first column, such as a threshold, intact.
    """

    return (
        len(cells) >= 3 and bool(_ORDINAL_CELL.match(cells[0])) and not _VALUE_CELL.match(cells[1])
    )


def _table_shape(rows: list[tuple[int, list[str], int, bool]]) -> tuple[int, int]:
    """Return the modal cell count of the data rows and their leftmost start column.

    Row labels are often centred rather than left aligned, so indentation alone cannot
    mark a row as unlabelled. A continuation row is instead recognised by carrying one
    fewer cell than a full row while starting to the right of where labels begin.
    """

    counts: dict[int, int] = {}
    for _, cells, _, header in rows:
        if header:
            continue
        counts[len(cells)] = counts.get(len(cells), 0) + 1
    if not counts:
        return 0, 0
    modal = min(counts, key=lambda count: (-counts[count], -count))
    indents = [indent for _, cells, indent, header in rows if not header and len(cells) == modal]
    return modal, min(indents) if indents else 0


def _table_rows(panel: PagePanel, start: int, end: int) -> list[TableRow]:
    raw: list[tuple[int, list[str], int, bool]] = []
    for index in range(start, end + 1):
        text = panel.lines[index]
        if not text.strip():
            continue
        cells = _cells(text)
        raw.append((index, cells, len(text) - len(text.lstrip()), _numeric_cell_count(cells) == 0))
    modal_cells, label_indent = _table_shape(raw)

    rows: list[TableRow] = []
    inherited: str | None = None
    for index, cells, indent, header in raw:
        # A data row missing a cell and starting to the right of the label column has an
        # empty label column. It belongs to the most recent labelled row ABOVE it, not
        # below; the synthetic continuation-row regression fixes this reading direction.
        continuation = (
            not header
            and modal_cells > 0
            and len(cells) < modal_cells
            and indent > label_indent + _LABEL_COLUMN_TOLERANCE
        )
        label: str | None = None
        if not continuation:
            if cells and not _VALUE_CELL.match(cells[0]):
                label = _row_label_from_cell(cells[0])
            elif _has_ordinal_first_column(cells):
                # A leading row-number column is an index, not the row's identity.
                label = _row_label_from_cell(cells[1])
        if not header and label is not None:
            inherited = label
        effective = label if label is not None else (None if header else inherited)
        rows.append(
            TableRow(
                line=index + 1,
                cells=cells,
                row_label=label,
                effective_row_label=effective,
                is_header=header,
                numeric_cells=_numeric_cell_count(cells),
            )
        )
    return rows


def _caption_of(panel: PagePanel, start: int, end: int) -> Caption:
    text = " ".join(
        panel.lines[index].strip() for index in range(start, end + 1) if panel.lines[index].strip()
    )
    match = _CAPTION_LABEL.match(text)
    label = None
    label_kind = None
    if match:
        kind = match.group("kind").lower().rstrip(".")
        label_kind = "figure" if kind in {"fig", "figure"} else kind
        label = f"{match.group('kind').strip().title()} {match.group('number')}"
        if label_kind == "figure":
            label = f"Figure {match.group('number')}"
    return Caption(
        label=label,
        label_kind=label_kind,
        text=text,
        span=_line_span(panel, start, end, panel.lines),
    )


def _attach_captions(regions: list[Region], captions: list[tuple[int, int, Caption]]) -> None:
    """Bind each caption to the adjacent table, preferring the closer side."""

    tables = [region for region in regions if region.kind is RegionKind.TABLE]
    for start, end, caption in captions:
        best: tuple[int, str, Region] | None = None
        for table in tables:
            if table.span.end_line < start + 1:
                distance = start + 1 - table.span.end_line
                position = "trailing"
            elif table.span.start_line > end + 1:
                distance = table.span.start_line - (end + 1)
                position = "leading"
            else:
                continue
            if distance > 3:
                continue
            if best is None or distance < best[0]:
                best = (distance, position, table)
        if best is None:
            continue
        _, position, table = best
        if table.caption is not None:
            continue
        table.caption = caption
        table.caption_position = position
        table.table_label = caption.label


@dataclass
class SectionState:
    """Section context carried across page boundaries by the document-level index."""

    section_path: list[str] = field(default_factory=list)
    section_number: str | None = None
    in_references: bool = False
    in_appendix: bool = False


def build_page_region_index(
    page: PageFragment,
    config: ResultBlockConfig | None = None,
    state: SectionState | None = None,
    *,
    single_panel: bool = False,
) -> PageRegionIndex:
    """Index one page. Deterministic and offline; no model is consulted.

    ``state`` carries the enclosing section across a page break and is mutated in place
    so a caller can thread it through a whole document. Omitting it starts the page with
    no known enclosing section, which is honest rather than guessed.
    """

    config = config or ResultBlockConfig()
    panels = (
        [PagePanel(lines=page.text.splitlines())] if single_panel else page_panels(page, config)
    )
    regions: list[Region] = []
    ordinal = 0
    carried = state if state is not None else SectionState()
    for panel_index, panel in enumerate(panels, start=1):
        kinds = _classify_panel_lines(panel)
        section_path = list(carried.section_path)
        section_number = carried.section_number
        in_references = carried.in_references
        in_appendix = carried.in_appendix
        captions: list[tuple[int, int, Caption]] = []
        panel_regions: list[Region] = []
        for start, end in _group_regions(kinds, config.max_blank_gap):
            kind = kinds[start]
            assert kind is not None
            if kind is RegionKind.HEADING:
                heading = _heading_of(panel.lines[start])
                if heading is not None:
                    title, number = heading
                    if _REFERENCE_HEADING.match(title):
                        in_references = True
                        in_appendix = False
                    elif _APPENDIX_HEADING.match(title):
                        in_appendix = True
                        in_references = False
                    elif in_references and _exits_references(title, number):
                        # A bibliography contains no numbered section headings, so one
                        # marks the end of it. Without this the flag never clears and
                        # every later page reads as reference text.
                        in_references = False
                    section_path = [title]
                    section_number = number
            if in_references and kind in {
                RegionKind.PROSE,
                RegionKind.TABLE,
                RegionKind.HEADING,
            }:
                # Inside a bibliography every line is a bibliography line. Author lists
                # and work titles otherwise read as headings and pollute section state.
                kind = RegionKind.REFERENCES
            ordinal += 1
            text = "\n".join(panel.lines[start : end + 1])
            region = Region(
                region_id=f"reg_{page.source_id}_p{page.page:04d}_c{panel_index}_{ordinal:03d}",
                kind=kind,
                span=_line_span(panel, start, end, panel.lines),
                text=text,
                section_path=list(section_path),
                section_number=section_number,
                in_references=in_references,
                in_appendix=in_appendix,
                rows=_table_rows(panel, start, end) if kind is RegionKind.TABLE else [],
            )
            if kind is RegionKind.CAPTION:
                caption = _caption_of(panel, start, end)
                captions.append((start, end, caption))
                # A caption is itself evidence. It carries its own label so that a quote
                # landing in the caption resolves without depending on table attachment.
                region.caption = caption
                region.table_label = caption.label
            panel_regions.append(region)
        _attach_captions(panel_regions, captions)
        regions.extend(panel_regions)
        carried.section_path = list(section_path)
        carried.section_number = section_number
        carried.in_references = in_references
        carried.in_appendix = in_appendix
    return PageRegionIndex(
        source_id=page.source_id,
        page=page.page,
        page_text_sha256=hashlib.sha256(page.text.encode("utf-8")).hexdigest(),
        panel_columns=[(panel.source_column_start, panel.source_column_end) for panel in panels],
        regions=regions,
    )


def build_region_index(
    layout: PdfLayout, config: ResultBlockConfig | None = None
) -> dict[int, PageRegionIndex]:
    """Index a whole document, threading section context across page breaks."""

    config = config or ResultBlockConfig()
    state = SectionState()
    return {
        page.page: build_page_region_index(page, config, state)
        for page in sorted(layout.pages, key=lambda item: item.page)
    }


# ----------------------------------------------------------------------------------
# Locating a quote in the same coordinates
# ----------------------------------------------------------------------------------


def _normalize(value: str) -> str:
    """Relax layout whitespace and line-end hyphenation only; never wording or digits."""

    normalized = unicodedata.normalize("NFKC", value).replace("\u2212", "-")
    normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def locate_quote(
    page: PageFragment, quote: str, config: ResultBlockConfig | None = None
) -> QuoteLocation | None:
    """Find the smallest line window containing ``quote`` and report its structure.

    Matching is containment after layout-whitespace normalization only, the same
    relaxation the pipeline's evidence binding uses. Wording, numbers, and punctuation
    are never relaxed. Returns ``None`` rather than guessing when the quote is absent.
    """

    index = build_page_region_index(page, config)
    return locate_quote_in_index(index, page, quote, config)


def locate_quote_in_index(
    index: PageRegionIndex,
    page: PageFragment,
    quote: str,
    config: ResultBlockConfig | None = None,
) -> QuoteLocation | None:
    """Locate a quote against an index that was already built for this page."""

    needle = _normalize(quote)
    if not needle:
        return None
    config = config or ResultBlockConfig()
    panels = page_panels(page, config)
    best = _search_panels(panels, needle)
    if best is None and len(panels) > 1:
        # A full-width table or a centred caption on an otherwise two-column page is
        # severed by the gutter and therefore exists in no panel. The physical page
        # line is always ground truth for what stands on that line, so fall back to it.
        best = _search_panels([PagePanel(lines=page.text.splitlines())], needle)
        if best is not None:
            index = build_page_region_index(page, config, single_panel=True)
    if best is None:
        return None
    start, end, _, panel = best
    return _location(index, panel, start, end)


def _search_panels(panels: list[PagePanel], needle: str) -> tuple[int, int, int, PagePanel] | None:
    """Smallest line window in any panel whose normalized text contains the quote."""

    best: tuple[int, int, int, PagePanel] | None = None
    for panel in panels:
        for start in range(len(panel.lines)):
            if not panel.lines[start].strip():
                continue
            for width in range(1, _MAX_QUOTE_WINDOW_LINES + 1):
                end = start + width - 1
                if end >= len(panel.lines):
                    break
                window = _normalize(" ".join(panel.lines[start : end + 1]))
                if needle in window:
                    if best is None or width < best[2]:
                        best = (start, end, width, panel)
                    break
    return best


def _location(
    index: PageRegionIndex, panel: PagePanel, start: int, end: int
) -> QuoteLocation | None:
    region = _region_covering(index, start + 1, end + 1, panel)
    if region is None:
        return None
    return QuoteLocation(
        region_id=region.region_id,
        kind=region.kind,
        span=_line_span(panel, start, end, panel.lines),
        section_path=list(region.section_path),
        in_references=region.in_references,
        in_appendix=region.in_appendix,
        table_label=region.table_label,
        caption_text=region.caption.text if region.caption else None,
        caption_position=region.caption_position,
        row_label=_row_label_for(region, start + 1, end + 1),
        matched_lines=list(range(start + 1, end + 2)),
    )


def _region_covering(
    index: PageRegionIndex, start_line: int, end_line: int, panel: PagePanel
) -> Region | None:
    """Prefer the region containing the first matched line inside the same panel."""

    column_start = panel.source_column_start
    candidates = [
        region
        for region in index.regions
        if region.span.start_line <= start_line <= region.span.end_line
        and (
            column_start is None
            or region.span.column_start is None
            or region.span.column_start >= column_start
        )
    ]
    if not candidates:
        candidates = [
            region
            for region in index.regions
            if region.span.start_line <= end_line and region.span.end_line >= start_line
        ]
    if not candidates:
        return None
    # A table beats overlapping prose from the other panel: the tighter structure wins.
    candidates.sort(
        key=lambda region: (region.kind is not RegionKind.TABLE, region.span.start_line)
    )
    return candidates[0]


def _row_label_for(region: Region, start_line: int, end_line: int) -> str | None:
    if region.kind is not RegionKind.TABLE:
        return None
    for row in region.rows:
        if start_line <= row.line <= end_line:
            return row.effective_row_label
    return None
