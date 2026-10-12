"""Bibliography filtering and chunk telemetry over selected pages."""

from __future__ import annotations

import hashlib

import pytest

from proceedings_to_eee.extraction.page_blocks import (
    block_in_references,
    body_reads_like_bibliography,
    count_chunked_tables,
    segment_result_pages,
)
from proceedings_to_eee.extraction.pdf_layout import (
    PageFragment,
    PdfLayout,
    reference_heading_line,
)
from proceedings_to_eee.extraction.result_blocks import (
    ResultBlockConfig,
    segment_page_result_blocks,
)

RESULTS_THEN_REFERENCES = """5 Results

Table 2. Detection performance per model.

     Model            Precision   Recall    F1
     System Cedar        0.42       0.81   0.55
     System Juniper      0.57       0.63   0.60
     System Maple        0.67       0.49   0.57

References

[1] A. Author. 2019. A synthetic title. In Proc. Conf., pages 11-22.
[2] B. Author and C. Author. 2020. Another synthetic title. Journal 12(3):45-67.
[3] D. Author. 2021. A third synthetic title. In Proc. Conf., pages 100-110.
"""

REFERENCES_CONTINUED = """[4] E. Author. 2018. A fourth synthetic title. In Proc. Conf., pages 200-210.
[5] F. Author and G. Author. 2017. A fifth synthetic title. Journal 7(1):1-20.
[6] H. Author. 2016. A sixth synthetic title. In Proc. Conf., pages 300-333.
"""  # noqa: E501 - layout line widths are part of the fixture

APPENDIX_AFTER_REFERENCES = """References

[1] A. Author. 2019. A synthetic title. In Proc. Conf., pages 11-22.
[2] B. Author and C. Author. 2020. Another synthetic title. Journal 12(3):45-67.

Table 9. Appendix detection performance.

     Model            Precision   Recall    F1
     System Cedar        0.42       0.81   0.55
     System Juniper      0.57       0.63   0.60
"""

REFERENCES_THEN_GAP = """References

[1] A. Author. 2019. A synthetic title. In Proc. Conf., pages 11-22.
[2] B. Author and C. Author. 2020. Another synthetic title. Journal 12(3):45-67.


"""

# A letter-numbered appendix heading does not end the bibliography for the region
# index, and the page-local heading rule covers everything below ``References``.
PROSE_RESULT_AFTER_REFERENCES = (
    REFERENCES_THEN_GAP
    + """A Additional Material

On the held-out split, System Cedar reaches an F1 score of 0.71 and a recall of 0.65,
while System Juniper reaches an F1 score of 0.60 on the same split.
"""
)

# The second column inherits the bibliography state of the first.
BIBLIOGRAPHY_BESIDE_RESULTS = """References                                                  The best-performing system is System Cedar
[1] A. Author. 2019. A synthetic title. In Proc.           (0.71 F1 score), followed by System Juniper
    Conf., pages 11-22.                                     (0.60 F1 score) and System Maple (0.57 F1
[2] B. Author and C. Author. 2020. Another title.          score). Annotator agreement reached 0.41.
    Journal 12(3):45-67.                                    Future work should extend the evaluation.
[3] D. Author. 2021. A third title. In Proc. Conf.,
    pages 100-110.
"""  # noqa: E501 - layout line widths are part of the fixture

# Rows that name a system with its year read like bibliography fields on their own.
YEARED_SYSTEM_TABLE = """     System              Precision   Recall    F1
     Cedar (2019)           0.42       0.81   0.55
     Juniper (2020)         0.57       0.63   0.60
"""
TABLE_CAPTION = "Table 9. Appendix detection performance per system.\n"


def _page(page: int, text: str) -> PageFragment:
    return PageFragment(
        fragment_id=f"frag_src_{page:04d}",
        source_id="src",
        page=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=sum(character.isdigit() for character in text),
        result_signal_score=1.0,
    )


def _layout(*texts: str) -> PdfLayout:
    pages = [_page(number, text) for number, text in enumerate(texts, start=1)]
    return PdfLayout(
        source_id="src",
        parser="fixture",
        parser_version="fixture/1",
        page_count=len(pages),
        pages=pages,
    )


def _long_table(rows: int) -> str:
    body = "\n".join(
        f"     System {index:02d}        0.{index:02d}   0.{index + 1:02d}   0.{index + 2:02d}"
        for index in range(1, rows + 1)
    )
    header = "     System             Accuracy  Recall  F1"
    return f"Table 3. Synthetic long matrix.\n\n{header}\n{body}\n"


def test_reference_blocks_are_dropped_on_the_heading_page_and_its_continuation() -> None:
    layout = _layout(RESULTS_THEN_REFERENCES, REFERENCES_CONTINUED)

    segmentation = segment_result_pages(layout, layout.pages)

    by_page = {page.page: page for page in segmentation.pages}
    assert segmentation.page_count == 2
    assert all("System Cedar" in block.body_text for block in by_page[1].blocks)
    assert by_page[1].blocks_dropped_in_references >= 1
    assert by_page[2].blocks == ()
    assert by_page[2].blocks_dropped_in_references >= 1
    assert segmentation.blocks_dropped_in_references == (
        by_page[1].blocks_dropped_in_references + by_page[2].blocks_dropped_in_references
    )
    assert not any("[1]" in block.body_text for block in segmentation.blocks)


def test_page_local_heading_is_a_second_criterion_without_a_region_index() -> None:
    page = _page(1, RESULTS_THEN_REFERENCES)
    heading_line = reference_heading_line(page.text)
    page_lines = page.text.splitlines()
    blocks = segment_page_result_blocks(page)
    table = next(block for block in blocks if "System Cedar" in block.body_text)
    references = next(block for block in blocks if "[1]" in block.body_text)

    assert heading_line is not None
    assert table.body_start_line < heading_line < references.body_start_line
    assert not block_in_references(
        table, index=None, heading_line=heading_line, page_lines=page_lines
    )
    assert block_in_references(
        references, index=None, heading_line=heading_line, page_lines=page_lines
    )
    assert not block_in_references(references, index=None, heading_line=None, page_lines=page_lines)


def test_a_captioned_table_after_the_references_is_kept() -> None:
    layout = _layout(APPENDIX_AFTER_REFERENCES)

    segmentation = segment_result_pages(layout, layout.pages)

    kept = segmentation.blocks
    assert any("System Cedar" in block.body_text for block in kept)
    assert not any("[1]" in block.body_text for block in kept)
    assert segmentation.blocks_dropped_in_references >= 1


def test_segmentation_respects_the_page_selection_and_reports_chunked_tables() -> None:
    layout = _layout(_long_table(20), RESULTS_THEN_REFERENCES)

    segmentation = segment_result_pages(layout, [layout.pages[0]], ResultBlockConfig())

    assert [page.page for page in segmentation.pages] == [1]
    assert segmentation.tables_chunked == 1
    assert segmentation.telemetry() == {
        "page_count": 2,
        "pages_selected": 1,
        "blocks_total": len(segmentation.blocks),
        "blocks_dropped_in_references": 0,
        "tables_chunked": 1,
    }


def test_chunked_table_count_ignores_single_chunk_blocks_and_prose_chunks() -> None:
    blocks = segment_page_result_blocks(_page(1, _long_table(20)))
    unsplit = segment_page_result_blocks(_page(2, _long_table(3)))

    assert count_chunked_tables(blocks) == 1
    assert count_chunked_tables(unsplit) == 0
    assert count_chunked_tables([*blocks, *unsplit]) == 1


def test_result_prose_after_the_references_is_kept_while_the_entries_are_dropped() -> None:
    layout = _layout(PROSE_RESULT_AFTER_REFERENCES)

    segmentation = segment_result_pages(layout, layout.pages)

    kept = segmentation.blocks
    assert any("0.71" in block.body_text for block in kept)
    assert not any("[1]" in block.body_text for block in kept)
    assert segmentation.blocks_dropped_in_references == 1


def test_a_result_column_beside_a_bibliography_column_is_kept() -> None:
    layout = _layout(BIBLIOGRAPHY_BESIDE_RESULTS)

    segmentation = segment_result_pages(layout, layout.pages)

    kept = segmentation.blocks
    assert [block.source_column_start is not None for block in kept] == [True]
    assert "F1 score" in kept[0].body_text
    assert not any("[2]" in block.body_text for block in kept)
    assert segmentation.blocks_dropped_in_references == 1


@pytest.mark.parametrize(
    ("page_text", "kept"),
    [
        pytest.param(
            REFERENCES_THEN_GAP + TABLE_CAPTION + "\n\n" + YEARED_SYSTEM_TABLE,
            True,
            id="caption above the block within the lookback window",
        ),
        pytest.param(
            REFERENCES_THEN_GAP + YEARED_SYSTEM_TABLE + TABLE_CAPTION,
            True,
            id="caption below the rows",
        ),
        pytest.param(
            REFERENCES_THEN_GAP + "A Additional Results\n" + YEARED_SYSTEM_TABLE,
            True,
            id="result heading in the leading context",
        ),
        pytest.param(REFERENCES_THEN_GAP + YEARED_SYSTEM_TABLE, False, id="no marker"),
    ],
)
def test_a_result_marker_keeps_a_bibliography_like_table_after_the_references(
    page_text: str, kept: bool
) -> None:
    layout = _layout(page_text)

    segmentation = segment_result_pages(layout, layout.pages)

    assert any("Cedar (2019)" in block.body_text for block in segmentation.blocks) is kept
    assert not any("[1]" in block.body_text for block in segmentation.blocks)
    assert segmentation.blocks_dropped_in_references == (1 if kept else 2)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        pytest.param(
            [
                "[7] A. Author and B. Author. 2021. Error rate estimation at scale.",
                "    In Proc. Conf., pages 11-22.",
            ],
            True,
            id="numbered entries whose title names a metric",
        ),
        pytest.param(
            ["recognition letters 27(8):861-874.", "", "10.1145/3599696.3612895."],
            True,
            id="entry fragments with volume, pages and a DOI",
        ),
        pytest.param(
            [
                "Alex Example and Casey Example. 2024. A Synthetic Reference.",
                "Robin Example and Taylor Example. 2025. Another Fixture Study.",
            ],
            True,
            id="author-year entries without numbers",
        ),
        pytest.param(
            [
                "comments on a 0-4 toxicity scale with a score lower than 2",
                "were removed; fine-tuning takes about 3-4 days to complete.",
            ],
            False,
            id="prose with ranges inside sentences",
        ),
        pytest.param(
            [
                "[7] A. Author. 2021. A synthetic title. In Proc. Conf.",
                "we observed a significant effect, F(2, 57) = 3.21, p < .05.",
            ],
            False,
            id="a statistic outweighs the entry beside it",
        ),
        pytest.param(
            [
                "[7] A. Author. 2021. Title.         English reaches 0.71 F1 score,",
                "    In Proc. Conf., 11-22.          followed by German (0.55 F1 score).",
            ],
            False,
            id="a result column beside a bibliography column",
        ),
        pytest.param(["", "   "], False, id="no content lines"),
    ],
)
def test_bibliography_likeness_of_a_block_body(lines: list[str], expected: bool) -> None:
    assert body_reads_like_bibliography(lines) is expected
