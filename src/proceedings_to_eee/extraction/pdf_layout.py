"""Layout-preserving PDF text extraction using the pinned local Poppler binary."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from pydantic import Field

from proceedings_to_eee.domain.observation import StrictModel


class PageFragment(StrictModel):
    """One page of layout text with deterministic identity."""

    fragment_id: str
    source_id: str
    page: int = Field(ge=1)
    text: str
    text_sha256: str
    character_count: int = Field(ge=0)
    numeric_token_count: int = Field(ge=0)
    result_signal_score: float = Field(ge=0.0)


class PdfLayout(StrictModel):
    schema_version: str = "pdf-layout/0.2"
    source_id: str
    parser: str
    parser_version: str
    page_count: int = Field(ge=1)
    pages: list[PageFragment] = Field(min_length=1)


_NUMERIC = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*%?(?!\w)")
_RESULT_TERMS = re.compile(
    r"\b(?:results?|accuracy|auc|auroc|f1|precision|recall|error|rate|score|"
    r"performance|evaluation|table|figure|mean|median|significant)\b",
    re.IGNORECASE,
)
_REFERENCE_SECTION_HEADING = re.compile(
    r"^\s*(?:references|bibliography|works cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _version(binary: str) -> str:
    completed = subprocess.run(
        [binary, "-v"], check=True, text=True, capture_output=True, timeout=15
    )
    first_line = (completed.stderr or completed.stdout).splitlines()[0]
    return first_line.strip()


def extract_pdf_layout(path: Path, source_id: str, binary: str = "pdftotext") -> PdfLayout:
    """Extract form-feed-separated pages while preserving horizontal whitespace."""

    completed = subprocess.run(
        [binary, "-layout", "-enc", "UTF-8", str(path), "-"],
        check=True,
        capture_output=True,
        timeout=180,
    )
    decoded = completed.stdout.decode("utf-8", errors="replace")
    page_texts = decoded.split("\f")
    while page_texts and not page_texts[-1].strip():
        page_texts.pop()
    if not page_texts:
        raise ValueError(f"no pages extracted from {path.name}")
    pages: list[PageFragment] = []
    for page_number, raw_text in enumerate(page_texts, start=1):
        text = raw_text.rstrip() + "\n"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        numeric_count = len(_NUMERIC.findall(text))
        term_count = len(_RESULT_TERMS.findall(text))
        density = numeric_count / max(len(text) / 1000.0, 1.0)
        score = round(density + min(term_count, 20) * 0.35, 6)
        pages.append(
            PageFragment(
                fragment_id=f"frag_{source_id}_{page_number:04d}",
                source_id=source_id,
                page=page_number,
                text=text,
                text_sha256=digest,
                character_count=len(text),
                numeric_token_count=numeric_count,
                result_signal_score=score,
            )
        )
    return PdfLayout(
        source_id=source_id,
        parser="poppler-pdftotext-layout",
        parser_version=_version(binary),
        page_count=len(pages),
        pages=pages,
    )


def _reference_fraction(page_index: object, region_kind: type) -> float:
    """Share of a page's content lines that the region index places in a bibliography."""

    if page_index is None:
        return 0.0
    reference_lines = 0
    content_lines = 0
    for region in page_index.regions:
        if region.kind is region_kind.PAGE_FURNITURE:
            continue
        lines = region.span.end_line - region.span.start_line + 1
        content_lines += lines
        if region.in_references:
            reference_lines += lines
    if content_lines == 0:
        return 0.0
    return reference_lines / content_lines


def select_result_pages(layout: PdfLayout, limit: int = 12) -> list[PageFragment]:
    """Select result-rich pages using the same structural signals as extraction.

    Raw numeric density over-ranks reference lists and can miss qualitative or
    graphically rendered tables. Result blocks add aligned-row, metric, caption,
    and table-context signals while remaining paper-agnostic.
    """

    if limit < 1:
        raise ValueError("page selection limit must be positive")
    from proceedings_to_eee.extraction.region_index import RegionKind, build_region_index
    from proceedings_to_eee.extraction.result_blocks import segment_page_result_blocks

    region_indexes = build_region_index(layout)

    ranked_pages: list[tuple[float, int, int, PageFragment]] = []
    for page in layout.pages:
        blocks = segment_page_result_blocks(page)
        data_rows = sum(block.data_row_count for block in blocks)
        caption_blocks = sum("caption" in block.signal_kinds for block in blocks)
        result_heading_blocks = sum("result_heading" in block.signal_kinds for block in blocks)
        caption_only_blocks = sum(
            "caption" in block.signal_kinds and block.data_row_count == 0 for block in blocks
        )
        structural_score = (
            sum(block.result_signal_score for block in blocks)
            + data_rows * 0.75
            + caption_blocks * 3.0
            + caption_only_blocks * 3.0
        )
        # Bibliographies contain dense years and page ranges that resemble sparse
        # result tables, and a continuation page of a reference list carries no
        # "References" heading of its own for a page-local regex to find. The region
        # index tracks the bibliography across page breaks, so a page whose content is
        # mostly reference lines is discounted whether or not the heading is on it.
        #
        # In development diagnostics, a continuation bibliography page could outrank
        # the actual results pages under the page-local rule. Cross-page region state
        # prevents dense citation lines from dominating page selection.
        if _reference_fraction(region_indexes.get(page.page), RegionKind) >= 0.6 or (
            _REFERENCE_SECTION_HEADING.search(page.text)
            and caption_blocks == 0
            and result_heading_blocks == 0
        ):
            structural_score *= 0.1
        ranked_pages.append((structural_score, data_rows, caption_blocks, page))

    ranked = sorted(
        ranked_pages,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            item[3].result_signal_score,
            item[3].numeric_token_count,
            -item[3].page,
        ),
        reverse=True,
    )[:limit]
    return sorted((item[3] for item in ranked), key=lambda page: page.page)
