from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from proceedings_to_eee.corpus import PaperSpec
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout, select_result_pages
from proceedings_to_eee.extraction.region_index import _normalize, build_region_index
from proceedings_to_eee.extraction.result_blocks import (
    LEGACY_RECOVERY_MAX_DEPTH,
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
            "extractor_block_failed",
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
