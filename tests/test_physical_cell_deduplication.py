from __future__ import annotations

import hashlib
import json

import pytest

from proceedings_to_eee.domain.observation import (
    CandidateObservation,
    EvidenceAnchor,
    MetricSpec,
    ObservationScope,
    ReportedValue,
    RoleAssignment,
)
from proceedings_to_eee.domain.status import (
    ActorRole,
    ClaimType,
    EvidenceKind,
    ExportStatus,
    ReferentialStatus,
    TextSupportStatus,
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.validation.candidates import deduplicate_candidates, validate_candidates
from proceedings_to_eee.validation.physical_cells import (
    PhysicalCellBindingStatus,
    PhysicalCellLocator,
)

SOURCE_ID = "src_synthetic_fixture"
PAPER_ID = "synthetic-physical-cells"

TABLE_ROWS = {
    "SysA": "                SysA                    4.28            1.49            2.77         1.17           6.73            1.31             2.41        1.19",  # noqa: E501
    "SysBee": "                SysBee                  5.12            1.89            1.72         0.82           5.29            2.17             1.68        0.79",  # noqa: E501
    "SysCee": "                SysCee                  6.40            1.65            2.43         1.46           7.23            1.28             3.42        1.88",  # noqa: E501
}
EVIDENCE_ROWS = {
    "SysA": "SysA                    4.28            1.49            2.77",
    "SysBee": "SysBee                  5.12            1.89            1.72",
    "SysCee": "SysCee                  6.40            1.65            2.43",
}

TABLE_PAGE_TEXT = (
    "\n".join(
        [
            "Synthetic Proceedings, 2026, Example City",
            "",
            "",
            "                                                               CategoryA Scores                                            CategoryB Score",  # noqa: E501
            "",
            "                                               Synthetic GroupA           Synthetic Controls A             Synthetic GroupB           Synthetic Controls B",  # noqa: E501
            "",
            "                Unit                    Avg.            Disp.           Avg.         Disp.          Avg.            Disp.            Avg.        Disp.",  # noqa: E501
            "",
            "                EvaluatorA              3.14            1.87            1.70         1.14                                        -",  # noqa: E501
            "                ToolOne                 3.29            2.07            1.24         0.86                                        -",  # noqa: E501
            "                ToolAI                  4.13            1.87            2.11         0.79                                        -",  # noqa: E501
            TABLE_ROWS["SysA"],
            TABLE_ROWS["SysBee"],
            TABLE_ROWS["SysCee"],
            "                Model                   6.81            2.18            2.96         1.57           7.24            2.16             3.11        1.56",  # noqa: E501
            "                Grp                     7.17            1.83            2.45         1.52           7.05            2.12             2.13        1.39",  # noqa: E501
            "                Non-Grp                 5.91            2.18            2.34         1.43           6.09            2.32             2.15        1.28",  # noqa: E501
            "",
            "                                 Table 1: Summary of synthetic ratings by tools and reviewer groups",  # noqa: E501
            "",
        ]
    )
    + "\n"
)


def _fragment(page: int, text: str) -> PageFragment:
    return PageFragment(
        fragment_id=f"fragment-{page}",
        source_id=SOURCE_ID,
        page=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_count=len(text),
        numeric_token_count=0,
        result_signal_score=0,
    )


def _table_layout() -> PdfLayout:
    return PdfLayout(
        source_id=SOURCE_ID,
        parser="fixture",
        parser_version="fixture-1",
        page_count=6,
        pages=[
            *[_fragment(page, "empty\n") for page in range(1, 6)],
            _fragment(6, TABLE_PAGE_TEXT),
        ],
    )


def _candidate(
    row: str,
    raw: str,
    column: str,
    *,
    aggregation: str | None = "mean",
    dataset: str = "Synthetic GroupA",
) -> CandidateObservation:
    return CandidateObservation(
        paper_id=PAPER_ID,
        claim_type=ClaimType.PRIMARY_RESULT,
        roles=[
            RoleAssignment(
                role=ActorRole.EVALUATED_SYSTEM,
                raw_name=row,
                confidence=1.0,
            )
        ],
        scope=ObservationScope(
            dataset_raw=dataset,
            subset=dataset,
            aggregation=aggregation,
        ),
        metric=MetricSpec(raw_name="Mean"),
        value=ReportedValue(raw=raw, numeric=float(raw)),
        evidence=[
            EvidenceAnchor(
                source_id=SOURCE_ID,
                page=6,
                kind=EvidenceKind.TABLE,
                label="Table 1",
                row=row,
                column=column,
                quote=EVIDENCE_ROWS[row],
            )
        ],
        text_support=TextSupportStatus.SUPPORTED,
        referential_status=ReferentialStatus.UNRESOLVED,
        export_status=ExportStatus.NEEDS_REVIEW,
        extraction_method=f"fixture:{column}",
        extraction_confidence=1.0,
        notes=[f"source note: {column}"],
        raw_payload_hash=hashlib.sha256(column.encode()).hexdigest(),
    )


@pytest.mark.parametrize(
    ("row", "raw", "line"),
    [("SysA", "4.28", 13), ("SysBee", "5.12", 14), ("SysCee", "6.40", 15)],
)
def test_target_values_bind_to_their_exact_printed_cells(
    row: str,
    raw: str,
    line: int,
) -> None:
    layout = _table_layout()
    candidate = _candidate(row, raw, "Synthetic GroupA Mean")

    binding = PhysicalCellLocator({SOURCE_ID: layout}).bind(candidate)

    assert binding.status is PhysicalCellBindingStatus.BOUND
    assert binding.identity is not None
    assert binding.identity.page_text_sha256 == hashlib.sha256(TABLE_PAGE_TEXT.encode()).hexdigest()
    # This single-panel fixture absorbs the spanning title on line 4. The frozen
    # two-column page indexes the table at lines 6-18; either way the full region span
    # is part of the identity rather than inferred from the model's label.
    assert binding.identity.table_start_line == 4
    assert binding.identity.table_end_line == 18
    assert binding.identity.row_line == line
    assert binding.identity.row_label == row
    assert binding.identity.value_ordinal == 1
    assert binding.identity.value_column_start == 41
    assert binding.identity.value_column_end == 44
    assert binding.identity.raw == raw


def test_overlapping_target_proposals_merge_by_physical_cell() -> None:
    layout = _table_layout()
    candidates = [
        proposal
        for row, raw in (("SysCee", "6.40"), ("SysBee", "5.12"), ("SysA", "4.28"))
        for proposal in (
            _candidate(row, raw, "Synthetic GroupA Mean"),
            _candidate(row, raw, "Mean (Synthetic GroupA)", aggregation=None),
        )
    ]

    merged = deduplicate_candidates(candidates, {SOURCE_ID: layout})

    assert len(merged) == 3
    assert {candidate.value.raw for candidate in merged if candidate.value} == {
        "6.40",
        "5.12",
        "4.28",
    }
    for candidate in merged:
        assert candidate.scope is not None
        assert candidate.scope.aggregation == "mean"
        assert {anchor.column for anchor in candidate.evidence} == {
            "Synthetic GroupA Mean",
            "Mean (Synthetic GroupA)",
        }
        assert {
            "source note: Synthetic GroupA Mean",
            "source note: Mean (Synthetic GroupA)",
        } <= set(candidate.notes)
        snapshot_note = next(
            note for note in candidate.notes if note.startswith("physical-cell source proposals:")
        )
        snapshots = json.loads(snapshot_note.removeprefix("physical-cell source proposals: "))
        assert len(snapshots) == 2
        assert {snapshot["scope"].get("aggregation") for snapshot in snapshots} == {None, "mean"}
        assert {snapshot["raw_payload_hash"] for snapshot in snapshots} == {
            hashlib.sha256(b"Synthetic GroupA Mean").hexdigest(),
            hashlib.sha256(b"Mean (Synthetic GroupA)").hexdigest(),
        }
        assert any(
            note.startswith("merged 2 proposals for physical cell") for note in candidate.notes
        )
        assert candidate.export_status is ExportStatus.NEEDS_REVIEW

    reverse_merged = deduplicate_candidates(list(reversed(candidates)), {SOURCE_ID: layout})

    def canonical(item: CandidateObservation) -> str:
        return item.model_dump_json(exclude_none=False)

    assert sorted(map(canonical, merged)) == sorted(map(canonical, reverse_merged))


def test_semantic_conflict_on_one_physical_cell_is_routed_to_review() -> None:
    layout = _table_layout()
    group_a = _candidate("SysCee", "6.40", "Synthetic GroupA Mean")
    conflicting = _candidate(
        "SysCee",
        "6.40",
        "Mean (Synthetic GroupA)",
        dataset="Synthetic GroupB",
    )

    [merged] = deduplicate_candidates([group_a, conflicting], {SOURCE_ID: layout})

    assert merged.referential_status is ReferentialStatus.WRONG_SCOPE
    assert merged.export_status is ExportStatus.NEEDS_REVIEW
    assert "physical-cell conflict" in (merged.export_reason or "")
    assert len(merged.evidence) == 2
    assert any("incompatible proposals" in note for note in merged.notes)


def test_physical_conflict_survives_full_revalidation() -> None:
    layout = _table_layout()
    layouts = {SOURCE_ID: layout}
    group_a = _candidate("SysCee", "6.40", "Synthetic GroupA Mean")
    conflicting = _candidate(
        "SysCee",
        "6.40",
        "Mean (Synthetic GroupA)",
        dataset="Synthetic GroupB",
    )

    first_pass = validate_candidates([group_a, conflicting], layouts)
    [merged] = deduplicate_candidates(first_pass, layouts)
    [revalidated] = validate_candidates([merged], layouts)

    assert revalidated.referential_status is ReferentialStatus.WRONG_SCOPE
    assert revalidated.export_status is ExportStatus.NEEDS_REVIEW
    assert revalidated.export_reason == (
        "physical-cell conflict: proposals for the same printed value "
        "have incompatible essential semantics"
    )
    assert any("incompatible proposals" in note for note in revalidated.notes)


def test_repeated_equal_values_in_distinct_columns_are_not_merged() -> None:
    repeated_row = "System                  1.00            1.00"
    text = "Metric A                Metric B\n" + repeated_row + "\nTable 1: Scores\n"
    layout = PdfLayout(
        source_id=SOURCE_ID,
        parser="fixture",
        parser_version="fixture-1",
        page_count=1,
        pages=[_fragment(1, text)],
    )
    first = _candidate("SysA", "4.28", "unused").model_copy(deep=True)
    first.roles[0].raw_name = "System"
    first.value = ReportedValue(raw="1.00", numeric=1.0)
    first.evidence = [
        EvidenceAnchor(
            source_id=SOURCE_ID,
            page=1,
            kind=EvidenceKind.TABLE,
            label="Table 1",
            row="System",
            column="Metric A",
            quote=repeated_row,
        )
    ]
    first.observation_id = first.stable_id()
    second = first.model_copy(deep=True)
    second.evidence[0].column = "Metric B"
    second.observation_id = second.stable_id()

    locator = PhysicalCellLocator({SOURCE_ID: layout})
    assert locator.bind(first).status is PhysicalCellBindingStatus.AMBIGUOUS
    assert locator.bind(second).status is PhysicalCellBindingStatus.AMBIGUOUS

    merged = deduplicate_candidates([first, second], {SOURCE_ID: layout})

    assert len(merged) == 2
    assert {candidate.evidence[0].column for candidate in merged} == {"Metric A", "Metric B"}


def test_exact_ambiguous_table_proposals_are_merged() -> None:
    repeated_row = "System                  1.00            1.00"
    text = "Metric A                Metric B\n" + repeated_row + "\nTable 1: Scores\n"
    layout = PdfLayout(
        source_id=SOURCE_ID,
        parser="fixture",
        parser_version="fixture-1",
        page_count=1,
        pages=[_fragment(1, text)],
    )
    first = _candidate("SysA", "4.28", "unused").model_copy(deep=True)
    first.roles[0].raw_name = "System"
    first.value = ReportedValue(raw="1.00", numeric=1.0)
    first.evidence = [
        EvidenceAnchor(
            source_id=SOURCE_ID,
            page=1,
            kind=EvidenceKind.TABLE,
            label="Table 1",
            row="System",
            column="Metric A",
            quote=repeated_row,
        )
    ]
    first.observation_id = first.stable_id()
    duplicate = first.model_copy(deep=True)

    locator = PhysicalCellLocator({SOURCE_ID: layout})
    assert locator.bind(first).status is PhysicalCellBindingStatus.AMBIGUOUS
    assert locator.bind(duplicate).status is PhysicalCellBindingStatus.AMBIGUOUS

    [merged] = deduplicate_candidates([first, duplicate], {SOURCE_ID: layout})

    assert merged.observation_id == merged.stable_id()
    assert "merged 2 duplicate proposals" in merged.notes


def test_unlocated_table_values_do_not_fall_back_to_semantic_merging() -> None:
    layout = _table_layout()
    first = _candidate("SysA", "4.28", "Model column A")
    first.evidence[0] = first.evidence[0].model_copy(
        update={"quote": "unlocatable evidence A", "quote_sha256": None}
    )
    first.observation_id = first.stable_id()
    second = first.model_copy(deep=True)
    second.evidence[0] = second.evidence[0].model_copy(
        update={
            "column": "Model column B",
            "quote": "unlocatable evidence B",
            "quote_sha256": None,
        }
    )
    second.observation_id = second.stable_id()

    locator = PhysicalCellLocator({SOURCE_ID: layout})
    assert locator.bind(first).status is PhysicalCellBindingStatus.UNLOCATED
    assert locator.bind(second).status is PhysicalCellBindingStatus.UNLOCATED

    merged = deduplicate_candidates([first, second], {SOURCE_ID: layout})

    assert len(merged) == 2
    assert {candidate.evidence[0].column for candidate in merged} == {
        "Model column A",
        "Model column B",
    }
