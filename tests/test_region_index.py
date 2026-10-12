from __future__ import annotations

import hashlib

import pytest

from proceedings_to_eee.extraction.pdf_layout import PageFragment
from proceedings_to_eee.extraction.region_index import (
    Region,
    RegionKind,
    build_page_region_index,
    build_region_index,
    locate_quote,
)
from proceedings_to_eee.extraction.result_blocks import ResultBlockConfig

# Invented page with an external leaderboard comparator and a paper-owned system under
# one deliberately ambiguous caption.
MIXED_ORIGIN_PAGE = """Synthetic Study                                                             000:01


                              Model                                  ROC-AUC
                              Linear Baseline                             0.611
                              Paper System                                0.742
                              Example Ensemble (Leaderboard Entry)         0.781
Table 2. Reported benchmark performance alongside the system developed in this study.


6.3.1 Fixture Assembly. During the toy experiment, we configured a compact
synthetic encoder that maps each example through a fixed deterministic transform.

5   EVALUATION
The final analysis compares the compact variant against a fixed reference.
"""

# A full-width table and a centred caption on a page whose body is two columns. The
# gutter severs both, so neither exists inside a single column panel.
SPLIT_PAGE = """       Asset                   Measure-A  ScaleB     Index-C        Measure-D  ScaleE   Index-F
       System Cedar              0.42       0.81      0.55            0.37      0.69      0.48
       System Juniper            0.57       0.63      0.60            0.49      0.62      0.55

                     Table 2. Invented scoring grid across two fixture categories.

Two reviewers covered three categories and we               The right column discusses tradeoffs across
report aggregate scores for every system.                   invented model variants in this fixture.
All systems used the same synthetic split                   System Juniper prioritizes recall over
and the same deterministic scoring script.                  precision in the illustrative values.
Results average three invented trials and                   Combining the variants can support
require no additional calibration.                          decisions in a synthetic deployment.
"""

LIST_AND_REFERENCES_PAGE = """3   RESULTS
We report three findings.

2. We introduce a synthetic multimodal component
3. Additionally, we evaluate a compact variant

References
Alex Example and Casey Example. 2024. A Synthetic Reference.
Robin Example and Taylor Example. 2025. Another Fixture Study.
"""

ORDINAL_TABLE_PAGE = """Table 7. Synthetic changes for indexed fixture outcomes.

        #    Fixture outcome         Shift                       Interval       Adjusted
        2    Packets queued          +6.4%                       [3.1%, 8.8%]    0.025
        7    Packets archived        -2.7%                       [-4.5%, -1.0%]  0.008
"""

# Invented two-column fixture whose row label is only single-spaced from its values.
SINGLE_SPACED_PAGE = """Table 4. Synthetic measurements for paired fixture gauges.

       Engine          X      Y     Z                         U     V    W
       GaugeKite  .42 .81 .55                              .37 .69 .48
      GaugeLumen  .53 .64 .58                              .49 .62 .55
"""

PUNCTUATION_LABEL_PAGE = """Table 10. Synthetic grid with an unlabeled numeric panel.

       —               Slice A       Slice B       0.50
       0.11            0.12          0.13          0.14
       0.21            0.22          0.23          0.24
       α-model         0.31          0.32          0.33
"""


def _page(text: str, page: int = 13, source_id: str = "src_paper") -> PageFragment:
    return PageFragment(
        fragment_id=f"frag_{source_id}_{page:04d}",
        source_id=source_id,
        page=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=0,
        result_signal_score=1.0,
    )


def test_caption_test_locates_both_mixed_origin_rows() -> None:
    """The comparator and paper row resolve to the same ambiguous caption."""

    location = locate_quote(_page(MIXED_ORIGIN_PAGE), "Example Ensemble (Leaderboard Entry) 0.781")
    assert location is not None
    assert location.kind is RegionKind.TABLE
    assert location.table_label == "Table 2"
    assert location.row_label == "Example Ensemble (Leaderboard Entry)"
    assert location.caption_position == "trailing"
    assert location.caption_text is not None
    assert "Reported benchmark" in location.caption_text
    # The paper's own row sits in the same table under the same caption, so the caption
    # alone can never be the whole attribution decision.
    own = locate_quote(_page(MIXED_ORIGIN_PAGE), "Paper System 0.742")
    assert own is not None
    assert own.table_label == location.table_label
    assert own.caption_text == location.caption_text
    assert own.row_label == "Paper System"


def test_numbered_heading_is_not_read_as_a_table_row() -> None:
    index = build_page_region_index(_page(MIXED_ORIGIN_PAGE))
    evaluation = [
        region
        for region in index.regions
        if region.kind is RegionKind.HEADING and "EVALUATION" in region.text
    ]
    assert len(evaluation) == 1


def test_run_in_heading_sets_the_enclosing_section() -> None:
    index = build_page_region_index(_page(MIXED_ORIGIN_PAGE))
    prose = [
        region
        for region in index.regions
        if region.kind is RegionKind.PROSE and "synthetic encoder" in region.text
    ]
    assert prose
    assert prose[0].section_path == ["Fixture Assembly"]
    assert prose[0].section_number == "6.3.1"


def test_page_furniture_is_separated_from_content() -> None:
    index = build_page_region_index(_page(MIXED_ORIGIN_PAGE))
    furniture = [region for region in index.regions if region.kind is RegionKind.PAGE_FURNITURE]
    assert len(furniture) == 1
    assert "000:01" in furniture[0].text


def test_unlabelled_continuation_row_inherits_the_label_above_it() -> None:
    """An empty label cell inherits the nearest labelled row above it."""

    text = """Table 8. Synthetic scores for paired fixture modalities.

  Model                  Modality     Alpha   Beta    Gamma   Total
  Fixture-Atlas              Text       0.42    0.57    0.63    0.54
                        Multimodal    0.46    0.61    0.68    0.58
  Fixture-Birch              Text       0.39    0.52    0.60    0.49
"""
    location = locate_quote(_page(text, page=6), "Multimodal 0.46 0.61 0.68 0.58")
    assert location is not None
    assert location.kind is RegionKind.TABLE
    assert location.row_label == "Fixture-Atlas"


def test_full_width_table_severed_by_the_column_gutter_is_still_located() -> None:
    page = _page(SPLIT_PAGE, page=6)
    row = locate_quote(page, "System Juniper 0.57 0.63 0.60 0.49 0.62 0.55")
    assert row is not None
    assert row.kind is RegionKind.TABLE
    assert row.row_label == "System Juniper"
    caption = locate_quote(page, "Table 2. Invented scoring grid across two fixture categories.")
    assert caption is not None
    assert caption.kind is RegionKind.CAPTION
    assert caption.table_label == "Table 2"


def test_two_column_prose_is_indexed_per_column() -> None:
    location = locate_quote(
        _page(SPLIT_PAGE, page=6),
        "report aggregate scores for every system. All systems used",
    )
    assert location is not None
    assert location.kind is RegionKind.PROSE


def test_numbered_list_items_are_not_headings() -> None:
    index = build_page_region_index(_page(LIST_AND_REFERENCES_PAGE, page=2))
    headings = [
        region.text.strip() for region in index.regions if region.kind is RegionKind.HEADING
    ]
    assert any("RESULTS" in text for text in headings)
    assert not any("introduce a synthetic" in text for text in headings)
    assert not any("Additionally, we evaluate" in text for text in headings)


def test_bibliography_lines_are_references_not_headings() -> None:
    index = build_page_region_index(_page(LIST_AND_REFERENCES_PAGE, page=2))
    bibliography = [region for region in index.regions if region.kind is RegionKind.REFERENCES]
    assert bibliography
    assert any("Alex Example" in region.text for region in bibliography)
    assert all(region.in_references for region in bibliography)
    assert not any(
        "Alex Example" in region.text
        for region in index.regions
        if region.kind is RegionKind.HEADING
    )


def test_leading_row_number_column_is_not_the_row_label() -> None:
    location = locate_quote(_page(ORDINAL_TABLE_PAGE, page=12), "0.008")
    assert location is not None
    assert location.row_label == "Packets archived"


def test_single_spaced_values_are_split_off_the_row_label() -> None:
    location = locate_quote(_page(SINGLE_SPACED_PAGE, page=7), "GaugeLumen .53 .64 .58")
    assert location is not None
    assert location.row_label == "GaugeLumen"


def test_a_label_ending_in_one_number_is_left_intact() -> None:
    text = """Table 1. Ablation results.

     System         Accuracy   F1
     Model 2          0.91     0.88
"""
    location = locate_quote(_page(text, page=4), "Model 2 0.91 0.88")
    assert location is not None
    assert location.row_label == "Model 2"


@pytest.mark.parametrize("content_free_label", ["—", "___"])
def test_punctuation_only_row_label_is_not_direct_or_inherited(
    content_free_label: str,
) -> None:
    text = PUNCTUATION_LABEL_PAGE.replace("—", content_free_label)
    index = build_page_region_index(_page(text, page=5))
    table = next(region for region in index.regions if region.kind is RegionKind.TABLE)

    assert [row.row_label for row in table.rows] == [None, None, None, "α-model"]
    assert [row.effective_row_label for row in table.rows] == [
        None,
        None,
        None,
        "α-model",
    ]


def test_index_is_deterministic_and_bound_to_the_page_text() -> None:
    page = _page(MIXED_ORIGIN_PAGE)
    first = build_page_region_index(page)
    second = build_page_region_index(page)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.page_text_sha256 == page.text_sha256


def test_absent_quote_returns_none_rather_than_a_guess() -> None:
    assert locate_quote(_page(MIXED_ORIGIN_PAGE), "Nonexistent Model 0.111") is None
    assert locate_quote(_page(MIXED_ORIGIN_PAGE), "   ") is None


def test_document_index_carries_sections_across_a_page_break() -> None:
    from proceedings_to_eee.extraction.pdf_layout import PdfLayout

    first_text = "4.7.1 Fixture Inputs. A generator emits deterministic token bundles.\n"
    second_text = "Additional bundles are reserved for the scoring pass.\n"
    layout = PdfLayout(
        source_id="src_paper",
        parser="poppler-pdftotext-layout",
        parser_version="test",
        page_count=2,
        pages=[_page(first_text, page=1), _page(second_text, page=2)],
    )
    index = build_region_index(layout)
    trailing = [region for region in index[2].regions if region.kind is RegionKind.PROSE]
    assert trailing
    assert trailing[0].section_path == ["Fixture Inputs"]


@pytest.mark.parametrize("page_number", [1, 7, 13])
def test_spans_stay_inside_the_page(page_number: int) -> None:
    page = _page(MIXED_ORIGIN_PAGE, page=page_number)
    index = build_page_region_index(page)
    line_count = len(page.text.splitlines())
    assert index.regions
    for region in index.regions:
        assert 1 <= region.span.start_line <= region.span.end_line <= line_count
        if region.span.column_start is not None:
            assert region.span.column_start >= 1
            assert region.span.column_end >= region.span.column_start


# ----------------------------------------------------------------------------------
# Caption attachment
# ----------------------------------------------------------------------------------

# A caption whose explanatory note and vertical spacing put the grid seven lines below it.
DISTANT_CAPTION_PAGE = """Table 5. Synthetic scores with a long note between caption and grid.
The note explains the abbreviations used below.
It continues for a second line.
And a third line that ends the note.



     System         Accuracy   F1
     Model Cedar      0.91     0.88
     Model Juniper    0.87     0.84
"""

# Two captions compete for one grid: the farther one comes first in reading order.
COMPETING_CAPTIONS_PAGE = """Table 7. Caption of a figure that left no text on this page.


Table 8. Synthetic scores for the grid below.
     System         Accuracy   F1
     Model Cedar      0.91     0.88
"""

# One caption sits exactly between two grids.
SHARED_CAPTION_PAGE = """     System         Accuracy   F1
     Model Cedar      0.91     0.88


Table 9. Synthetic scores printed below the first grid.


     System         Accuracy   F1
     Model Juniper    0.87     0.84
"""

# Side-by-side column tables with captions below them. The right caption is closer to
# the left table than the left caption is, but captions do not change column.
SIDE_BY_SIDE_PAGE = """     System       Precision  Recall  F1               System          Model-A  Model-B
     Cedar           0.42     0.81  0.55             Baseline          0.37     0.69
     Juniper         0.57     0.63  0.60             Tuned             0.49     0.62
     Maple           0.61     0.59  0.66
     Willow          0.52     0.71  0.58
                                                     Table 4. Right-hand synthetic grid.

Table 3. Left-hand synthetic grid.

Two reviewers covered three categories and we        The right column discusses tradeoffs across
report aggregate scores for every system.            invented model variants in this fixture.
All systems used the same synthetic split            System Juniper prioritizes recall over
and the same deterministic scoring script.           precision in the illustrative values.
"""

# A full-width grid whose centred caption is too far below to join the spanning run,
# so the gutter cuts the caption into two column fragments.
SEVERED_CAPTION_PAGE = """       Asset                   Measure-A  ScaleB     Index-C        Measure-D  ScaleE   Index-F
       System Cedar              0.42       0.81      0.55            0.37      0.69      0.48
       System Juniper            0.57       0.63      0.60            0.49      0.62      0.55
       System Maple              0.61       0.59      0.66            0.52      0.64      0.51


                     Table 2. Invented scoring grid across two fixture categories.

Two reviewers covered three categories and we               The right column discusses tradeoffs across
report aggregate scores for every system.                   invented model variants in this fixture.
All systems used the same synthetic split                   System Juniper prioritizes recall over
and the same deterministic scoring script.                  precision in the illustrative values.
Results average three invented trials and                   Combining the variants can support
require no additional calibration.                          decisions in a synthetic deployment.
"""

# A left-column caption whose text exactly fills its column. It stands six lines below
# the left grid and one line above the right grid, but nothing of it is in the gutter.
COLUMN_FILLING_CAPTION_PAGE = """     System       Precision  Recall  F1               The right column discusses tradeoffs across
     Cedar           0.42     0.81  0.55             invented model variants in this fixture.
     Juniper         0.57     0.63  0.60             System Juniper prioritizes recall over
                                                     precision in the illustrative values.
Two reviewers covered three categories and we        Combining the variants can support
report aggregate scores for every system.            decisions in a synthetic deployment.
All systems used the same synthetic split            Every value in this column is invented
and the same deterministic scoring script.           and serves only the fixture geometry.
Table 6. Left-hand grid that fills its column
                                                          System         Model-A  Model-B
Results average three invented trials and                 Baseline         0.37     0.69
require no additional calibration.                        Tuned            0.49     0.62
"""

FIGURE_ABOVE_PAGE = """Figure 3. Synthetic plot whose axis labels did not survive extraction.

     System         Accuracy   F1
     Model Cedar      0.91     0.88
"""

FIGURE_BELOW_PAGE = """     0.0        0.1         0.2         0.3
     0.42       0.55        0.61        0.70

Figure 4. Synthetic curve rendered as aligned tick labels.
"""


def _tables(text: str, config: ResultBlockConfig | None = None) -> list[Region]:
    index = build_page_region_index(_page(text, page=3), config)
    return [region for region in index.regions if region.kind is RegionKind.TABLE]


def test_caption_beyond_three_lines_is_attached_within_the_default_distance() -> None:
    [table] = _tables(DISTANT_CAPTION_PAGE)
    assert table.table_label == "Table 5"
    assert table.caption_position == "leading"


def test_caption_distance_is_configurable() -> None:
    [table] = _tables(DISTANT_CAPTION_PAGE, ResultBlockConfig(caption_distance_lines=3))
    assert table.caption is None


def test_nearest_caption_wins_over_the_first_caption_in_reading_order() -> None:
    [table] = _tables(COMPETING_CAPTIONS_PAGE)
    assert table.table_label == "Table 8"


def test_a_caption_between_two_grids_is_claimed_once() -> None:
    tables = _tables(SHARED_CAPTION_PAGE)
    assert len(tables) == 2
    claimed = [table for table in tables if table.caption is not None]
    assert len(claimed) == 1
    assert claimed[0].caption_position == "trailing"


def test_side_by_side_tables_keep_their_own_column_captions() -> None:
    tables = _tables(SIDE_BY_SIDE_PAGE)
    labels = {table.span.column_start: table.table_label for table in tables}
    assert len(tables) == 2
    assert labels[min(labels)] == "Table 3"
    assert labels[max(labels)] == "Table 4"


def test_a_caption_severed_by_the_gutter_attaches_to_the_full_width_table() -> None:
    index = build_page_region_index(_page(SEVERED_CAPTION_PAGE, page=3))
    assert len(index.panel_columns) == 3
    [table] = [region for region in index.regions if region.kind is RegionKind.TABLE]
    assert table.span.start_line == 1
    assert table.table_label == "Table 2"
    assert table.caption_position == "trailing"


def test_a_caption_that_merely_fills_its_column_stays_in_its_panel() -> None:
    index = build_page_region_index(_page(COLUMN_FILLING_CAPTION_PAGE, page=3))
    [left_end] = [end for start, end in index.panel_columns if start == 1]
    [caption] = [region for region in index.regions if region.kind is RegionKind.CAPTION]
    assert caption.span.column_end == left_end
    labels = {
        region.span.column_start <= left_end: region.table_label
        for region in index.regions
        if region.kind is RegionKind.TABLE
    }
    assert labels == {True: "Table 6", False: None}


@pytest.mark.parametrize(
    ("text", "label"),
    [(FIGURE_ABOVE_PAGE, None), (FIGURE_BELOW_PAGE, "Figure 4")],
)
def test_a_figure_caption_attaches_only_from_below(text: str, label: str | None) -> None:
    [table] = _tables(text)
    assert table.table_label == label


# ----------------------------------------------------------------------------------
# Heading heuristics
# ----------------------------------------------------------------------------------


def _heading_texts(line: str) -> list[str]:
    text = f"An opening sentence for the fixture.\n{line}\nThe fixture sentence continues here.\n"
    index = build_page_region_index(_page(text, page=2))
    return [region.text.strip() for region in index.regions if region.kind is RegionKind.HEADING]


@pytest.mark.parametrize("line", ["F1", "ACM", "AUC F1", "ROC AUC"])
def test_a_short_all_caps_token_is_not_a_heading(line: str) -> None:
    assert _heading_texts(line) == []


@pytest.mark.parametrize(
    "line", ["RELATED WORK", "EXPERIMENTAL RESULTS", "ABSTRACT", "Overview", "Übersicht"]
)
def test_titles_and_single_capitalized_words_remain_headings(line: str) -> None:
    assert _heading_texts(line) == [line]


@pytest.mark.parametrize(
    "heading",
    [
        "References",
        "REFERENCES",
        "Bibliography",
        "Works Cited",
        "Bibliographie",
        "Références",
        "Riferimenti bibliografici",
        "7 Riferimenti bibliografici",
        "Bibliografia",
        "Literatur",
        "Literaturverzeichnis",
        "Referenzen",
        "Quellen",
    ],
)
def test_bibliography_headings_in_corpus_languages_open_the_reference_section(
    heading: str,
) -> None:
    text = f"{heading}\nAlex Example and Casey Example. 2024. A Synthetic Reference.\n"
    index = build_page_region_index(_page(text, page=9))
    entries = [region for region in index.regions if "Alex Example" in region.text]
    assert entries
    assert all(region.kind is RegionKind.REFERENCES for region in entries)
    assert all(region.in_references for region in entries)
