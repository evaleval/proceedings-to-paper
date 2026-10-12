from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import pytest

from proceedings_to_eee.corpus import PaperSpec
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout, select_result_pages
from proceedings_to_eee.extraction.region_index import _normalize, build_region_index
from proceedings_to_eee.extraction.result_blocks import (
    LEGACY_RECOVERY_MAX_DEPTH,
    ResultBlockConfig,
    maximum_legacy_block_invocations,
    segment_page_result_blocks,
    split_result_block,
)
from proceedings_to_eee.pipeline import PipelineSettings, _recover_split_block
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderRequestRejectedError,
    ProviderResponseValidationError,
    StructuredResponse,
    completion_token_parameter_for_model,
)

TABLE_PAGE = """Table 8. Synthetic indicator grid.

     Unit               Measure-A   ScaleB   Index-C
     System Cedar          0.42       0.81   0.55
     System Juniper        0.57       0.63   0.60
     System Maple          0.67       0.49   0.57
     System Willow         0.38       0.73   0.50
     System Aspen          0.71       0.44   0.54
"""


def _page(text: str, page: int = 8, source_id: str = "src_paper") -> PageFragment:
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


def _table_block():
    blocks = segment_page_result_blocks(_page(TABLE_PAGE))
    block = next((item for item in blocks if item.data_row_count >= 4), None)
    assert block is not None, "fixture must segment into one multi-row table block"
    return block


def test_split_result_block_halves_the_body_and_keeps_context() -> None:
    block = _table_block()
    halves = split_result_block(block)
    assert len(halves) == 2
    first, second = halves
    # Line ranges must tile the original exactly, so every anchor stays recoverable.
    assert first.body_start_line == block.body_start_line
    assert second.body_end_line == block.body_end_line
    assert second.body_start_line == first.body_end_line + 1
    assert first.body_text + second.body_text == block.body_text
    # Context is repeated on both halves; the block contract already allows that.
    assert first.context_text == block.context_text
    assert second.context_text == block.context_text
    assert {half.block_id for half in halves} == {
        f"{block.block_id}_s1",
        f"{block.block_id}_s2",
    }
    for half in halves:
        assert half.text_sha256 != block.text_sha256


def test_split_result_block_refuses_a_single_line_body() -> None:
    block = _table_block()
    one_line = block.body_text.splitlines()[0] + "\n"
    single = block.model_copy(
        update={
            "body_text": one_line,
            "body_end_line": block.body_start_line,
            "line_count": 1,
            "character_count": len(block.context_text)
            + len(one_line)
            + len(block.trailing_context_text),
            "text_sha256": hashlib.sha256(
                (block.context_text + one_line + block.trailing_context_text).encode("utf-8")
            ).hexdigest(),
        }
    )
    assert split_result_block(single) == []


class _RecordingClient:
    """Return an empty but valid extraction, and count the requests it received."""

    def __init__(
        self,
        fail_on: set[int] | None = None,
        reject_on: set[int] | None = None,
        transport_fail_on: set[int] | None = None,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail_on = fail_on or set()
        self.reject_on = reject_on or set()
        self.transport_fail_on = transport_fail_on or set()

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        self.requests.append(kwargs)
        call = ProviderCall(
            model_requested=kwargs["model"],
            model_returned=kwargs["model"],
            prompt_sha256="a" * 64,
            response_sha256="b" * 64,
            temperature=kwargs["temperature"],
            reasoning_effort=kwargs["reasoning_effort"],
            max_tokens=kwargs["max_tokens"],
            completion_token_parameter=completion_token_parameter_for_model(kwargs["model"]),
            seed=kwargs["seed"],
            schema_name=kwargs["schema_name"],
            schema_sha256="c" * 64,
            latency_seconds=0.01,
            attempts=1,
        )
        request_number = len(self.requests)
        if request_number in self.reject_on:
            raise ProviderRequestRejectedError(status_code=429)
        if request_number in self.transport_fail_on:
            raise RuntimeError("raw transport detail must not enter recovery telemetry")
        if request_number in self.fail_on:
            raise ProviderResponseValidationError(call=call, code="invalid_json")
        return StructuredResponse(
            payload={"observations": [], "page_summary": "none", "warnings": []},
            call=call,
        )


def _spec_and_settings(tmp_path: Path) -> tuple[PaperSpec, PipelineSettings]:
    spec = PaperSpec(
        paper_id="fixture-paper",
        title="Fixture Paper",
        year=2024,
        venue="Fixture Symposium",
        pdf_url="https://example.org/paper.pdf",
        perspective_role="subject",
    )
    settings = PipelineSettings(
        project_root=tmp_path,
        schema_path=tmp_path / "schema.json",
        schema_sha256="d" * 64,
        output_root=tmp_path / "runs",
        model="fixture/extractor",
    )
    return spec, settings


def test_recovery_re_extracts_from_the_halves(tmp_path: Path) -> None:
    """An identical retry of a temperature-zero call cannot help; the input must change."""

    spec, settings = _spec_and_settings(tmp_path)
    client = _RecordingClient()
    recovered = _recover_split_block(
        client=client, settings=settings, spec=spec, block=_table_block()
    )
    assert recovered.succeeded is True
    assert recovered.candidates == []
    assert len(recovered.calls) == 2
    assert len(recovered.successful_calls) == 2
    assert len(client.requests) == 2
    # The two retries must not be the request that just failed.
    sent = [request["user"] for request in client.requests]
    assert sent[0] != sent[1]


def test_recovery_recursively_splits_a_validation_failed_half(tmp_path: Path) -> None:
    spec, settings = _spec_and_settings(tmp_path)
    client = _RecordingClient(fail_on={2})
    recovered = _recover_split_block(
        client=client, settings=settings, spec=spec, block=_table_block()
    )

    assert recovered.succeeded is True
    assert len(client.requests) == 4
    assert len(recovered.calls) == 4
    assert len(recovered.successful_calls) == 3
    assert recovered.max_depth_reached == 2


def test_recovery_preserves_successful_siblings_at_a_typed_terminal_leaf(
    tmp_path: Path,
) -> None:
    spec, settings = _spec_and_settings(tmp_path)
    client = _RecordingClient(fail_on={2, 3})

    recovered = _recover_split_block(
        client=client, settings=settings, spec=spec, block=_table_block()
    )

    assert recovered.succeeded is False
    assert len(client.requests) == 4
    assert len(recovered.calls) == 4
    assert len(recovered.successful_calls) == 2
    assert len(recovered.terminal_failures) == 1
    failure = recovered.terminal_failures[0]
    assert failure.error_code == "provider_response_invalid_json"
    assert failure.completed_provider_call is True
    assert failure.terminal_reason == "unsplittable"
    serialized = json.dumps(asdict(failure))
    assert "OpenRouter response" not in serialized
    assert "raw transport detail" not in serialized


def test_recovery_stops_at_the_configured_depth_bound(tmp_path: Path) -> None:
    spec, settings = _spec_and_settings(tmp_path)
    client = _RecordingClient(fail_on={2})

    recovered = _recover_split_block(
        client=client,
        settings=settings,
        spec=spec,
        block=_table_block(),
        max_depth=1,
    )

    assert recovered.succeeded is False
    assert len(client.requests) == 2
    assert recovered.max_depth_reached == 1
    assert recovered.terminal_failures[0].terminal_reason == "max_depth_reached"


@pytest.mark.parametrize(
    ("client", "error_code", "terminal_reason"),
    [
        (
            _RecordingClient(reject_on={2}),
            "provider_request_rejected",
            "request_rejected",
        ),
        (
            _RecordingClient(transport_fail_on={2}),
            "provider_transport_failed",
            "transport_failure",
        ),
    ],
)
def test_recovery_never_recurses_request_or_transport_failures(
    tmp_path: Path,
    client: _RecordingClient,
    error_code: str,
    terminal_reason: str,
) -> None:
    spec, settings = _spec_and_settings(tmp_path)

    recovered = _recover_split_block(
        client=client, settings=settings, spec=spec, block=_table_block()
    )

    assert recovered.succeeded is False
    assert len(client.requests) == 2
    assert len(recovered.calls) == 1
    assert len(recovered.successful_calls) == 1
    assert recovered.terminal_failures[0].error_code == error_code
    assert recovered.terminal_failures[0].terminal_reason == terminal_reason
    assert "raw transport detail" not in json.dumps(asdict(recovered.terminal_failures[0]))


def test_five_line_legacy_block_has_a_nine_invocation_hard_bound() -> None:
    block = _table_block()

    assert len(block.body_text.splitlines()) == 5
    assert LEGACY_RECOVERY_MAX_DEPTH == 3
    assert maximum_legacy_block_invocations(block) == 9


# ----------------------------------------------------------------------------------
# Reference-aware page selection
# ----------------------------------------------------------------------------------


def _layout(pages: list[tuple[int, str]]) -> PdfLayout:
    return PdfLayout(
        source_id="src_paper",
        parser="fixture",
        parser_version="fixture/1",
        page_count=len(pages),
        pages=[_page(text, number) for number, text in pages],
    )


BIBLIOGRAPHY_START = """References
Alex Example and Casey Example. 2018. A Synthetic Language Study. In Venue A, 1877-1901.
Robin Example and Taylor Example. 2022. Fixture Scaling. In Venue B, 234-267.
"""

# A continuation page of the same bibliography. It carries no heading of its own, and
# its years and page ranges look exactly like a sparse result table.
BIBLIOGRAPHY_CONTINUATION = """Morgan Example and River Example. 2020. Scaling Fixtures. In Venue C, 55-79.
Sky Example and Drew Example. 2021. Controlled Decoding. In Venue D, 6691-6706.
Lee Example and Quinn Example. 2023. Retrieval Fixtures. In Venue E, 5108-5125.
Sam Example and Jules Example. 2015. A Synthetic Corpus. In Venue F, 632-642.
"""

RESULTS_PAGE = """5   RESULTS
The held-out F1 score was 0.82, exceeding the 0.71 baseline by a wide margin.
"""


def test_bibliography_continuation_page_loses_to_a_results_page() -> None:
    """The page-local heading rule cannot see a continuation page; the index can."""

    layout = _layout([(1, BIBLIOGRAPHY_START), (2, BIBLIOGRAPHY_CONTINUATION), (3, RESULTS_PAGE)])
    index = build_region_index(layout)
    assert all(region.in_references for region in index[2].regions)

    selected = [page.page for page in select_result_pages(layout, limit=1)]
    assert selected == [3]


def test_a_numbered_section_ends_the_bibliography() -> None:
    layout = _layout([(1, BIBLIOGRAPHY_START), (2, RESULTS_PAGE)])
    index = build_region_index(layout)
    assert all(region.in_references for region in index[1].regions)
    assert not any(region.in_references for region in index[2].regions)


def test_a_running_head_does_not_end_the_bibliography() -> None:
    """A short synthetic running head must not terminate the reference section."""

    layout = _layout([(1, BIBLIOGRAPHY_START), (2, "FIXTURE\n" + BIBLIOGRAPHY_CONTINUATION)])
    index = build_region_index(layout)
    continuation = [region for region in index[2].regions if "Morgan Example" in region.text]
    assert continuation
    assert all(region.in_references for region in continuation)


@pytest.mark.parametrize("limit", [1, 2])
def test_selection_is_deterministic(limit: int) -> None:
    layout = _layout([(1, BIBLIOGRAPHY_START), (2, BIBLIOGRAPHY_CONTINUATION), (3, RESULTS_PAGE)])
    first = [page.page for page in select_result_pages(layout, limit=limit)]
    second = [page.page for page in select_result_pages(layout, limit=limit)]
    assert first == second


# ----------------------------------------------------------------------------------
# Full-width tables on an otherwise two-column page
# ----------------------------------------------------------------------------------

# Rows 1-5 span the page: their numeric columns straddle the gutter on a stable grid.
# Below the caption the body is genuinely two columns, and the right column holds its own
# table whose values sit entirely to the right of the gutter.
FULL_WIDTH_PAGE = """                          Slice Amber                   Slice Indigo                       Combined
 Model                   Measure-A ScaleB Index-C      Measure-A ScaleB  Index-C     Measure-A ScaleB Q3
 System Cedar              0.41     0.83      0.55       0.62      0.48     0.54       0.50    0.69   0.58
 System Juniper            0.57     0.66      0.61       0.44      0.76     0.56       0.51    0.71   0.60
 Human Reviewer            0.69     0.63      0.66       0.73      0.59     0.65       0.71    0.61   0.66

                     Table 9. Synthetic scores across two fixture slices.

Two reviewers covered three categories and we                Model            Modality      C1     All
report aggregate scores for every system.                    System Cedar       Text        0.54   0.50
All systems used the same synthetic split                                     Multimodal    0.61   0.57
and the same deterministic scoring script.                   System Juniper     Text        0.49   0.53
Results average three invented trials and                                     Multimodal    0.58   0.62
require no additional calibration.                           System Maple       Text        0.52   0.55
"""


def _full_width_blocks():
    return segment_page_result_blocks(_page(FULL_WIDTH_PAGE, page=6))


def test_a_full_width_row_is_not_severed_by_the_gutter() -> None:
    """The gutter split used to put half of every full-width row in each panel."""

    row = "System Juniper            0.57     0.66      0.61       0.44      0.76     0.56       0.51    0.71   0.60"
    intact = [
        block
        for block in _full_width_blocks()
        if _normalize(row) in _normalize(block.prompt_text())
    ]
    assert intact, "a full-width table row must survive inside one block"
    assert intact[0].source_column_start == 1


def test_a_centred_caption_survives_the_gutter() -> None:
    caption = "Table 9. Synthetic scores across two fixture slices."
    assert any(
        _normalize(caption) in _normalize(block.prompt_text()) for block in _full_width_blocks()
    )


def test_the_two_column_table_below_stays_column_scoped() -> None:
    """Only the straddling grid is full width; ordinary two-column content is not."""

    blocks = _full_width_blocks()
    column_scoped = [block for block in blocks if block.source_column_start not in (1, None)]
    assert column_scoped, "the right-column table must still be segmented per column"
    assert any(
        _normalize("System Juniper     Text        0.49   0.53") in _normalize(block.prompt_text())
        for block in blocks
    )


def test_left_column_prose_is_not_swallowed_by_the_full_width_run() -> None:
    """Extension crosses headers and captions only, never a two-column paragraph."""

    blocks = _full_width_blocks()
    prose = "report aggregate scores for every system."
    swallowed = [
        block
        for block in blocks
        if block.source_column_start == 1
        and block.source_column_end is not None
        and block.source_column_end > 80
        and _normalize(prose) in _normalize(block.prompt_text())
    ]
    assert not swallowed


def test_a_single_column_page_is_unaffected() -> None:
    """No gutter means no full-width carve-out and no change in behaviour."""

    blocks = segment_page_result_blocks(_page(TABLE_PAGE))
    assert blocks
    assert all(block.source_column_start is None for block in blocks)


PARALLEL_TABLES_PAGE = """        System       Precision  Recall   F1                 Condition       Model-A  Model-B  Model-C
        Audit API         0.53      0.64 0.58                 Plain           0.37     0.80     2.93
        Classifier One    0.78      0.27 0.41                 Clustered       0.37     1.07     2.90
        Classifier Two    0.43      0.65 0.52                 Prefixed        0.50     1.23     3.83
Table 14. Direct audit against human labels.                  Table 17. Independent attack results.
"""

PARALLEL_HEADER_TABLES_PAGE = """Model              Precision  Recall  F1                 Model              Accuracy  Error  Score
Audit API               0.53    0.64 0.58                 Attack One             0.37   0.80   2.93
Classifier One          0.78    0.27 0.41                 Attack Two             0.37   1.07   2.90
Classifier Two          0.43    0.65 0.52                 Attack Three           0.50   1.23   3.83
Classifier Three        0.61    0.48 0.54                 Attack Four            0.62   0.77   2.41
"""


def test_parallel_numeric_tables_are_not_mistaken_for_one_full_width_table() -> None:
    blocks = segment_page_result_blocks(_page(PARALLEL_TABLES_PAGE, page=11))
    left_target = [
        block
        for block in blocks
        if _normalize("Audit API 0.53 0.64 0.58") in _normalize(block.prompt_text())
    ]
    right_target = [
        block
        for block in blocks
        if _normalize("Prefixed 0.50 1.23 3.83") in _normalize(block.prompt_text())
    ]

    assert left_target and right_target
    assert left_target[0].source_column_end is not None
    assert right_target[0].source_column_start is not None
    assert left_target[0].source_column_end < right_target[0].source_column_start
    assert "Independent attack" not in left_target[0].prompt_text()
    assert "Direct audit" not in right_target[0].prompt_text()


def test_matching_panel_headers_veto_a_false_full_width_table_without_captions() -> None:
    blocks = segment_page_result_blocks(_page(PARALLEL_HEADER_TABLES_PAGE, page=12))
    left_target = [
        block
        for block in blocks
        if _normalize("Classifier One 0.78 0.27 0.41") in _normalize(block.prompt_text())
    ]
    right_target = [
        block
        for block in blocks
        if _normalize("Attack Two 0.37 1.07 2.90") in _normalize(block.prompt_text())
    ]

    assert left_target and right_target
    assert left_target[0].source_column_end is not None
    assert right_target[0].source_column_start is not None
    assert left_target[0].source_column_end < right_target[0].source_column_start


def test_dense_table_rows_never_overlap_between_result_block_bodies() -> None:
    rows = "\n".join(
        f"System {index:02d}        0.{index:02d}   0.{index + 1:02d}   0.{index + 2:02d}"
        for index in range(1, 21)
    )
    page = _page(
        f"Table 12. Dense synthetic matrix.\n\nSystem             Accuracy  Recall  F1\n{rows}\n",
        page=12,
    )
    blocks = segment_page_result_blocks(
        page,
        ResultBlockConfig(max_data_rows=6, overlap_lines=3, max_blocks_per_page=None),
    )
    table_blocks = sorted(
        (block for block in blocks if block.data_row_count),
        key=lambda block: block.body_start_line,
    )

    assert len(table_blocks) > 1
    assert all(block.overlap_with_previous_lines == 0 for block in table_blocks)
    assert all(
        left.body_end_line < right.body_start_line
        for left, right in zip(table_blocks, table_blocks[1:], strict=False)
    )
    for index in range(1, 21):
        row = _normalize(f"System {index:02d}")
        assert sum(row in _normalize(block.body_text) for block in table_blocks) == 1


def test_prose_result_blocks_keep_the_configured_overlap() -> None:
    conditions = (
        "amber",
        "indigo",
        "cedar",
        "juniper",
        "maple",
        "willow",
        "aspen",
        "birch",
        "elm",
    )
    lines = "\n".join(
        f"Accuracy was 0.{index + 21} for the synthetic {condition} condition."
        for index, condition in enumerate(conditions)
    )
    blocks = segment_page_result_blocks(
        _page(lines + "\n", page=13),
        ResultBlockConfig(
            max_lines=4,
            context_lines=0,
            trailing_context_lines=0,
            overlap_lines=2,
            max_data_rows=None,
            max_blocks_per_page=None,
        ),
    )

    assert len(blocks) > 1
    assert any(block.overlap_with_previous_lines > 0 for block in blocks[1:])


def test_mixed_table_prose_chunks_overlap_only_the_trailing_prose_suffix() -> None:
    conditions = ("amber", "indigo", "cedar", "juniper", "maple", "willow", "aspen")
    prose = "\n".join(
        f"Accuracy was 0.{index + 72} for the synthetic {condition} condition."
        for index, condition in enumerate(conditions)
    )
    page = _page(
        "Table 14. Mixed result evidence.\n"
        "System             Accuracy\n"
        "Atlas                  0.91\n"
        f"{prose}\n",
        page=14,
    )
    blocks = segment_page_result_blocks(
        page,
        ResultBlockConfig(
            max_lines=4,
            context_lines=2,
            trailing_context_lines=0,
            overlap_lines=2,
            max_data_rows=None,
            max_blocks_per_page=None,
        ),
    )

    assert len(blocks) > 1
    assert sum("Atlas                  0.91" in block.body_text for block in blocks) == 1
    assert any(block.overlap_with_previous_lines > 0 for block in blocks[1:])
    for left, right in zip(blocks, blocks[1:], strict=False):
        shared = set(left.body_text.splitlines()) & set(right.body_text.splitlines())
        assert "Atlas                  0.91" not in shared


# ----------------------------------------------------------------------------------
# Captions beyond the context windows
# ----------------------------------------------------------------------------------

# A note and a double blank line separate the caption from the grid, so the leading
# context window stops at the header row.
DISTANT_LEADING_CAPTION_PAGE = """Table 11. Synthetic indicator grid with a note between caption and grid.
The note explains the abbreviations used in the grid.
It also names the synthetic split that produced the values.


     Unit               Measure-A   ScaleB   Index-C
     System Cedar          0.42       0.81   0.55
     System Juniper        0.57       0.63   0.60
     System Maple          0.67       0.49   0.57
"""

# Notes and a double blank line separate the grid from the caption printed below it.
DISTANT_TRAILING_CAPTION_PAGE = """     Unit               Measure-A   ScaleB   Index-C
     System Cedar          0.42       0.81   0.55
     System Juniper        0.57       0.63   0.60
     System Maple          0.67       0.49   0.57
Values are averaged over three synthetic trials.
Bold marks the best synthetic value per column.


Table 12. Synthetic indicator grid with its caption printed below the notes.
"""

# One caption between two grids, adjacent to the second one.
TWO_GRIDS_ONE_CAPTION_PAGE = """     Unit               Measure-A   ScaleB   Index-C
     System Cedar          0.42       0.81   0.55
     System Juniper        0.57       0.63   0.60


Table 13. Synthetic grid printed directly above its rows.
     Unit               Measure-A   ScaleB   Index-C
     System Maple          0.67       0.49   0.57
     System Willow         0.38       0.73   0.50
"""


def _data_blocks(text: str, config: ResultBlockConfig | None = None):
    blocks = segment_page_result_blocks(_page(text, page=11), config)
    return [block for block in blocks if block.data_row_count > 0]


def test_a_caption_beyond_the_leading_window_joins_the_leading_context() -> None:
    [block] = _data_blocks(DISTANT_LEADING_CAPTION_PAGE)
    assert block.context_start_line == 1
    assert block.context_text.startswith("Table 11.")
    assert block.body_start_line == 7
    assert "Table 11." not in block.body_text


def test_a_caption_beyond_the_trailing_window_becomes_the_trailing_context() -> None:
    [block] = _data_blocks(DISTANT_TRAILING_CAPTION_PAGE)
    assert block.trailing_context_start_line == 9
    assert block.trailing_context_text.startswith("Table 12.")
    assert block.body_end_line < block.trailing_context_start_line
    assert "Table 12." not in block.body_text


@pytest.mark.parametrize(
    ("text", "span"),
    [
        (DISTANT_LEADING_CAPTION_PAGE, "context_text"),
        (DISTANT_TRAILING_CAPTION_PAGE, "trailing_context_text"),
    ],
)
def test_the_caption_distance_bounds_the_block_context(text: str, span: str) -> None:
    [block] = _data_blocks(text, ResultBlockConfig(caption_distance_lines=3))
    assert "Table 1" not in getattr(block, span)


def test_a_caption_between_two_grids_goes_to_the_nearer_grid_only() -> None:
    blocks = _data_blocks(TWO_GRIDS_ONE_CAPTION_PAGE)
    assert len(blocks) == 2
    with_caption = [block for block in blocks if "Table 13." in block.prompt_text()]
    assert len(with_caption) == 1
    assert "System Maple" in with_caption[0].body_text


def _caption_above_header_run(notes: int, headers: int) -> str:
    """A caption, explanatory notes, value-free aligned header rows, then data rows."""

    lines = ["Table 14. Synthetic grid printed below a run of value-free header rows."]
    lines += [
        f"Note {letter} explains one abbreviation used in the grid."
        for letter in ascii_uppercase[:notes]
    ]
    lines += [
        f"     Header-{letter}{letter}          Alpha   Beta   Gamma"
        for letter in ascii_uppercase[:headers]
    ]
    lines += [
        f"     System {name:<8}          0.4{index}    0.8{index}   0.5{index}"
        for index, name in enumerate(("Cedar", "Juniper", "Maple", "Willow"))
    ]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(("notes", "headers", "carries_caption"), [(3, 5, True), (10, 18, False)])
def test_the_extended_leading_context_is_bounded_by_the_caption_reach(
    notes: int, headers: int, carries_caption: bool
) -> None:
    config = ResultBlockConfig()
    blocks = _data_blocks(_caption_above_header_run(notes, headers), config)
    assert blocks
    for block in blocks:
        assert block.context_start_line is not None
        context_lines = block.context_end_line - block.context_start_line + 1
        assert context_lines <= config.context_lines + config.caption_distance_lines
        assert ("Table 14." in block.context_text) is carries_caption


def test_a_caption_inside_the_window_leaves_the_window_unchanged() -> None:
    block = _table_block()
    assert block.context_start_line == 1
    assert block.context_end_line == 3
    assert block.body_start_line == 4


def test_caption_distance_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        ResultBlockConfig(caption_distance_lines=-1)


def _long_table_page(rows: int) -> PageFragment:
    body = "\n".join(
        f"System {index:02d}        0.{index:02d}   0.{index + 1:02d}   0.{index + 2:02d}"
        for index in range(1, rows + 1)
    )
    return _page(
        f"Table 15. Chunked synthetic matrix.\n\nSystem             Accuracy  Recall  F1\n{body}\n",
        page=15,
    )


def test_long_table_is_chunked_by_data_rows_with_caption_and_header_repeated() -> None:
    config = ResultBlockConfig()
    chunks = sorted(
        (
            block
            for block in segment_page_result_blocks(_long_table_page(30))
            if block.data_row_count
        ),
        key=lambda block: block.body_start_line,
    )

    assert [block.chunk_index for block in chunks] == list(range(1, len(chunks) + 1))
    assert {block.chunk_count for block in chunks} == {len(chunks)}
    assert len(chunks) > 1
    assert all(block.data_row_count <= config.max_data_rows for block in chunks)
    assert sum(block.data_row_count for block in chunks) == 30
    for block in chunks:
        assert "Table 15." in block.context_text
        assert "Accuracy  Recall  F1" in block.context_text
    assert chunks[0].header_repeated is False
    assert all(block.header_repeated for block in chunks[1:])


def _wide_table_page(rows: int, columns: int) -> PageFragment:
    header = "System             " + "  ".join(f"M{index}" for index in range(1, columns + 1))
    # Row labels carry no digits, so every numeric token in the body is a value cell.
    body = "\n".join(
        f"System {ascii_uppercase[row % 26]}{ascii_uppercase[row // 26]}        "
        + "   ".join(f"0.{(row + column) % 100:02d}" for column in range(columns))
        for row in range(rows)
    )
    return _page(f"Table 16. Wide synthetic matrix.\n\n{header}\n{body}\n", page=16)


@pytest.mark.parametrize(("rows", "columns"), [(12, 8), (14, 3), (6, 24)])
def test_chunks_are_bounded_by_numeric_cells_not_only_rows(rows: int, columns: int) -> None:
    config = ResultBlockConfig()
    chunks = sorted(
        (
            block
            for block in segment_page_result_blocks(_wide_table_page(rows, columns))
            if block.data_row_count
        ),
        key=lambda block: block.body_start_line,
    )

    assert sum(block.data_row_count for block in chunks) == rows
    assert all(block.data_row_count <= config.max_data_rows for block in chunks)
    # Every chunk body stays inside the cell bound unless a single row alone exceeds it.
    assert all(
        len(re.findall(r"\d+\.\d+", block.body_text)) <= config.max_numeric_cells
        or block.data_row_count == 1
        for block in chunks
    )
    # A wide table is cut into more chunks than its row count alone would require.
    expected_at_least = -(-rows * columns // config.max_numeric_cells)
    assert len(chunks) >= min(expected_at_least, rows)
    for block in chunks[1:]:
        assert block.header_repeated is True
        assert "Table 16." in block.context_text


def test_the_cell_bound_can_be_switched_off_and_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_numeric_cells"):
        ResultBlockConfig(max_numeric_cells=0)
    config = ResultBlockConfig(max_numeric_cells=None)
    chunks = [
        block
        for block in segment_page_result_blocks(_wide_table_page(12, 8), config=config)
        if block.data_row_count
    ]
    assert len(chunks) == 1
    assert len(re.findall(r"\d+\.\d+", chunks[0].body_text)) == 96


def test_a_table_within_the_row_cap_is_a_single_unrepeated_chunk() -> None:
    blocks = [
        block for block in segment_page_result_blocks(_long_table_page(5)) if block.data_row_count
    ]

    assert len(blocks) == 1
    assert (blocks[0].chunk_index, blocks[0].chunk_count) == (1, 1)
    assert blocks[0].header_repeated is False


def test_chunk_metadata_must_be_consistent() -> None:
    block = _table_block()

    with pytest.raises(ValueError, match="chunk_index"):
        block.model_copy(update={"chunk_index": 3, "chunk_count": 2}).model_validate(
            block.model_copy(update={"chunk_index": 3, "chunk_count": 2}).model_dump()
        )
    with pytest.raises(ValueError, match="header_repeated"):
        type(block).model_validate(block.model_copy(update={"header_repeated": True}).model_dump())


def test_line_budget_must_fit_the_row_cap_behind_repeated_context() -> None:
    with pytest.raises(ValueError, match="max_lines"):
        ResultBlockConfig(max_lines=20, context_lines=8, max_data_rows=14)
    assert ResultBlockConfig(max_lines=20, context_lines=8, max_data_rows=None).max_lines == 20
    assert ResultBlockConfig(max_lines=30, context_lines=8, max_data_rows=14).max_data_rows == 14
