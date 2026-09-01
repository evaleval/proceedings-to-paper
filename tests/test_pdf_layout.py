from __future__ import annotations

import hashlib

from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout, select_result_pages


def _page(page: int, text: str, *, legacy_score: float) -> PageFragment:
    normalized = text.rstrip() + "\n"
    return PageFragment(
        fragment_id=f"page-{page}",
        source_id="source",
        page=page,
        text=normalized,
        text_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        character_count=len(normalized),
        numeric_token_count=sum(character.isdigit() for character in normalized),
        result_signal_score=legacy_score,
    )


def test_page_selection_retains_caption_only_graphical_result_over_numeric_noise() -> None:
    layout = PdfLayout(
        source_id="source",
        parser="fixture",
        parser_version="fixture/1",
        page_count=3,
        pages=[
            _page(
                1,
                "References\nSmith 2020 12 34 56 78 90\nJones 2021 11 22 33 44 55",
                legacy_score=50.0,
            ),
            _page(
                2,
                "Synthetic Test Set\nAUCs AEGs\nTable 3: Comparison of model results.",
                legacy_score=2.0,
            ),
            _page(
                3,
                "Results\nTable 4: Accuracy\nSystem A       0.82\nSystem B       0.75",
                legacy_score=10.0,
            ),
        ],
    )

    selected = select_result_pages(layout, limit=2)

    assert [page.page for page in selected] == [2, 3]


def test_page_selection_retains_result_value_wrapped_onto_its_own_line() -> None:
    layout = PdfLayout(
        source_id="source",
        parser="fixture",
        parser_version="fixture/1",
        page_count=3,
        pages=[
            _page(
                1,
                "References\nSmith 2020 12 34 56 78\nJones 2021 11 22 33 44",
                legacy_score=50.0,
            ),
            _page(
                2,
                "6 Results\nThe held-out F1 score was\n0.82, exceeding the baseline.",
                legacy_score=1.0,
            ),
            _page(
                3,
                "Table 3: Visual comparison of the systems.",
                legacy_score=0.5,
            ),
        ],
    )

    selected = select_result_pages(layout, limit=2)

    assert [page.page for page in selected] == [2, 3]
