from __future__ import annotations

import hashlib
import json

import pytest

from proceedings_to_eee.domain.attribution import (
    AttributionState,
    AttributionVerdict,
    OriginBasis,
    OriginExportPolicy,
    PaperContext,
    PaperKind,
)
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ClaimType, ExportStatus
from proceedings_to_eee.extraction.llm_schema import provider_json_schema
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.region_index import build_page_region_index
from proceedings_to_eee.resolution.attribution import (
    PAPER_CONTEXT_CUE_SCOPE,
    attribute_candidate,
    has_table_level_cue,
    load_lexicon,
    load_recorded_lexicon,
    paper_context_from_title,
    row_names_team,
)
from proceedings_to_eee.resources import ATTRIBUTION_LEXICON_PATHS, DEFAULT_ATTRIBUTION_LEXICON_PATH
from proceedings_to_eee.validation.candidates import _route_attribution, validate_candidates

# Invented table with one external comparator, one paper-owned result, and a caption
# that contains both external-reporting and first-party language.
MIXED_ORIGIN_PAGE = """                              Model                                  ROC-AUC
                              Linear Baseline                             0.611
                              Paper System                                0.742
                              Example Ensemble (Leaderboard Entry)         0.781
Table 2. Reported benchmark performance alongside the system developed in this study.
"""

# Every row carries the marker, so there is no contrast to decide on.
ALL_FOREIGN_PAGE = """Table 4. Published leaderboard entries.

     System                          Score
     Alpha Leaderboard Entry          0.91
     Beta Leaderboard Entry           0.88
     Gamma Leaderboard Entry          0.87
"""

# Citations here name invented datasets, not the origin of the numbers.
DATASET_CITATION_PAGE = """Table 1. Detector performance per dataset.

     Dataset                  AUC
     Dataset Alpha [11]      0.68
     Dataset Beta [90]       0.59
     Dataset Gamma [40]      0.63
"""


def _page(text: str, page: int = 13) -> PageFragment:
    return PageFragment(
        fragment_id=f"frag_src_paper_{page:04d}",
        source_id="src_paper",
        page=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=0,
        result_signal_score=1.0,
    )


def _candidate(
    quote: str,
    row: str,
    value: str,
    *,
    page: int = 13,
    export_status: ExportStatus = ExportStatus.ELIGIBLE,
) -> CandidateObservation:
    return CandidateObservation.model_validate(
        {
            "paper_id": "fixture",
            "claim_type": ClaimType.PRIMARY_RESULT,
            "roles": [{"role": "evaluated_system", "raw_name": row, "confidence": 0.9}],
            "scope": {"dataset_raw": "Synthetic Benchmark"},
            "metric": {"raw_name": "ROC-AUC", "canonical_id": "auroc", "unit": "proportion"},
            "value": {"raw": value, "numeric": float(value)},
            "evidence": [
                {
                    "source_id": "src_paper",
                    "page": page,
                    "kind": "table",
                    "label": "Table 2",
                    "row": row,
                    "column": "ROC-AUC",
                    "quote": quote,
                }
            ],
            "export_status": export_status,
            "extraction_method": "fixture",
            "extraction_confidence": 0.95,
        }
    )


def _attribute(page_text: str, quote: str, row: str, value: str, page: int = 13):
    fragment = _page(page_text, page)
    return attribute_candidate(
        _candidate(quote, row, value, page=page),
        build_page_region_index(fragment),
        fragment,
        load_lexicon(),
    )


def test_external_comparator_is_attributed_to_its_own_source() -> None:
    verdict = _attribute(
        MIXED_ORIGIN_PAGE,
        "Example Ensemble (Leaderboard Entry) 0.781",
        "Example Ensemble (Leaderboard Entry)",
        "0.781",
    )
    assert verdict.state is AttributionState.EXTERNALLY_SOURCED
    assert verdict.rule_id == "row_scoped_foreign_cue"
    # The decisive row cue leads; the caption cue is recorded behind it, never deciding.
    assert [cue.cue_id for cue in verdict.cues] == ["leaderboard", "caption_reported"]
    assert verdict.demotes


def test_the_papers_own_row_in_an_ambiguous_caption_stays_unresolved() -> None:
    """Caption-scoped evidence may neither decide external nor positive origin."""

    verdict = _attribute(MIXED_ORIGIN_PAGE, "Paper System 0.742", "Paper System", "0.742")
    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.demotes
    # The caption cue is recorded, so the evidence stays visible in review.
    assert [cue.cue_id for cue in verdict.cues] == ["caption_reported"]
    assert verdict.rule_id == "weak_cue_recorded"


def test_a_dataset_citation_abstains_without_claiming_external_origin() -> None:
    verdict = _attribute(
        DATASET_CITATION_PAGE,
        "Dataset Beta [90] 0.59",
        "Dataset Beta [90]",
        "0.59",
        page=4,
    )
    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.demotes
    assert "citation_bracket" in {cue.cue_id for cue in verdict.cues}


def test_a_marker_on_every_row_abstains_rather_than_demoting_the_table() -> None:
    verdict = _attribute(
        ALL_FOREIGN_PAGE, "Beta Leaderboard Entry 0.88", "Beta Leaderboard Entry", "0.88", page=9
    )
    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.rule_id == "decisive_cue_without_contrast"
    assert verdict.contrast_rows_total == verdict.contrast_rows_matched


def test_missing_structure_is_unresolved_not_no_signal() -> None:
    candidate = _candidate("System A 0.80", "System A", "0.80", page=1)
    verdict = attribute_candidate(candidate, None, None, load_lexicon())

    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.rule_id == "no_page_index"


def test_unlocatable_quote_is_unresolved_not_no_signal() -> None:
    fragment = _page("Table 1. Results.\nSystem    AUC\nSystem A  0.81\n", page=1)
    candidate = _candidate("System A  0.80", "System A", "0.80", page=1)
    verdict = attribute_candidate(
        candidate,
        build_page_region_index(fragment),
        fragment,
        load_lexicon(),
    )

    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.rule_id == "unlocatable"


def test_non_table_quote_is_unresolved_not_no_signal() -> None:
    text = "Results\nSystem A achieved an AUC of 0.80 on Dataset A.\n"
    verdict = _attribute(text, "System A achieved an AUC of 0.80", "System A", "0.80", page=1)

    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.rule_id == "not_a_table_row"


def test_paper_produced_exists_as_a_trusted_state_but_is_not_inferred() -> None:
    trusted = AttributionVerdict(
        state=AttributionState.PAPER_PRODUCED,
        rule_id="explicit_test_fixture",
    )
    inferred = _attribute(
        """Table 1. Results.\nSystem    AUC\nSystem A  0.80\n""",
        "System A  0.80",
        "System A",
        "0.80",
        page=1,
    )

    assert trusted.allows_canonical_export
    assert not trusted.demotes
    assert inferred.state is AttributionState.NO_SIGNAL
    assert inferred.rule_id == "no_cue"
    assert inferred.state is not AttributionState.PAPER_PRODUCED
    assert inferred.demotes


def test_verdict_records_the_lexicon_it_used() -> None:
    verdict = _attribute(
        MIXED_ORIGIN_PAGE,
        "Example Ensemble (Leaderboard Entry) 0.781",
        "Example Ensemble (Leaderboard Entry)",
        "0.781",
    )
    lexicon = load_lexicon()
    assert verdict.lexicon_id == lexicon.lexicon_id
    assert verdict.lexicon_sha256 == lexicon.sha256


def _layout(text: str, page: int = 13) -> dict[str, PdfLayout]:
    """Pad to the requested page so page numbers line up with the real pipeline."""

    pages = [_page("filler\n", number) for number in range(1, page)]
    pages.append(_page(text, page))
    return {
        "src_paper": PdfLayout(
            source_id="src_paper",
            parser="poppler-pdftotext-layout",
            parser_version="test",
            page_count=page,
            pages=pages,
        )
    }


def test_gate_demotes_the_comparator_to_review_not_to_a_silent_drop() -> None:
    comparator = _candidate(
        "                              Example Ensemble (Leaderboard Entry)         0.781",
        "Example Ensemble (Leaderboard Entry)",
        "0.781",
    )
    own = _candidate(
        "                              Paper System                                0.742",
        "Paper System",
        "0.742",
    )
    validate_candidates([comparator, own], _layout(MIXED_ORIGIN_PAGE), min_confidence=0.0)

    assert comparator.export_status is ExportStatus.NEEDS_REVIEW
    assert comparator.export_reason is not None
    assert "attribution=externally_sourced" in comparator.export_reason
    # NOT_ELIGIBLE carries no risk reason in the human review lane, so it would drop the
    # candidate out of sight. Review keeps it inspectable and still clears the export gate.
    assert comparator.export_status is not ExportStatus.NOT_ELIGIBLE
    assert comparator.attribution is not None
    assert comparator.attribution.state is AttributionState.EXTERNALLY_SOURCED
    assert own.attribution is not None
    assert own.attribution.state is AttributionState.UNRESOLVED
    assert own.export_status is ExportStatus.NEEDS_REVIEW
    assert "attribution=unresolved" in (own.export_reason or "")


def test_attribution_is_demote_only() -> None:
    """The attribution step alone may only worsen an export status, never improve it."""

    layouts = _layout(MIXED_ORIGIN_PAGE)
    held = _candidate(
        "Paper System 0.742",
        "Paper System",
        "0.742",
        export_status=ExportStatus.NEEDS_REVIEW,
    )
    held.export_reason = "held for another reason"
    comparator = _candidate(
        "Example Ensemble (Leaderboard Entry) 0.781",
        "Example Ensemble (Leaderboard Entry)",
        "0.781",
        export_status=ExportStatus.NEEDS_REVIEW,
    )
    _route_attribution([held, comparator], layouts)

    # A no_signal verdict changes nothing at all.
    assert held.export_status is ExportStatus.NEEDS_REVIEW
    assert held.export_reason == "held for another reason"
    # A demoting verdict on an already-held candidate does not resurrect it either.
    assert comparator.export_status is ExportStatus.NEEDS_REVIEW
    assert comparator.attribution is not None
    assert comparator.attribution.state is AttributionState.EXTERNALLY_SOURCED


def test_many_systems_with_no_cue_all_stay_in_review() -> None:
    """No rule may treat one entity as the target and the rest as distractors."""

    text = """Table 5. Aggregate performance.

     System                 AUC
     System Cedar          0.68
     System Juniper        0.59
     System Maple          0.63
     System Willow         0.71
"""
    layouts = _layout(text, page=5)
    rows = [
        ("System Cedar 0.68", "System Cedar", "0.68"),
        ("System Juniper 0.59", "System Juniper", "0.59"),
        ("System Maple 0.63", "System Maple", "0.63"),
        ("System Willow 0.71", "System Willow", "0.71"),
    ]
    candidates = [_candidate(quote, row, value, page=5) for quote, row, value in rows]
    validate_candidates(candidates, layouts, min_confidence=0.0)
    assert all(item.export_status is ExportStatus.NEEDS_REVIEW for item in candidates)
    assert all(
        item.attribution is not None and item.attribution.state is AttributionState.NO_SIGNAL
        for item in candidates
    )


def test_absent_layout_demotes_an_otherwise_exportable_candidate() -> None:
    candidate = _candidate("System A 0.80", "System A", "0.80", page=1)

    _route_attribution([candidate], {})

    assert candidate.attribution is not None
    assert candidate.attribution.state is AttributionState.UNRESOLVED
    assert candidate.export_status is ExportStatus.NEEDS_REVIEW


def test_attribution_never_enters_the_provider_wire_schema() -> None:
    """Standing guard: the extraction schema is hash-pinned in the holdout seal."""

    assert "attribution" not in json.dumps(provider_json_schema())


def test_attribution_never_becomes_a_claim_type_member() -> None:
    """Standing guard: ClaimType types the wire model, so adding a member voids the seal."""

    assert {member.value for member in ClaimType} == {
        "primary_result",
        "secondary_claim",
        "illustration",
        "method_metadata",
        "uncertain",
    }


def test_attribution_does_not_move_observation_ids() -> None:
    """stable_id() builds from an explicit whitelist, so the new field must be inert."""

    candidate = _candidate(
        "Example Ensemble (Leaderboard Entry) 0.781",
        "Example Ensemble (Leaderboard Entry)",
        "0.781",
    )
    before = candidate.stable_id()
    _route_attribution([candidate], _layout(MIXED_ORIGIN_PAGE))
    assert candidate.attribution is not None
    assert candidate.attribution.state is AttributionState.EXTERNALLY_SOURCED
    assert candidate.stable_id() == before


# ----------------------------------------------------------------------------------
# Lexicon v1: table-level abstaining cues and the paper-context prior
# ----------------------------------------------------------------------------------

# A ranked results table. The header carries a rank column and the caption describes the
# official standings; neither says which row is the paper's own.
RANKED_PAGE = """Table 3: Official results of the shared task, top-5 systems.

     Rank    Team                     F1
     1       Alpha Team              0.83
     2       Beta Team               0.81
     3       Ours                    0.79
     4       Run 1                   0.75
"""

# The same systems without a rank column or a standings caption: no table-level cue.
PLAIN_PAGE = """Table 3: Results per system.

     Alpha Team              0.83
     Beta Team               0.81
     Ours                    0.79
"""

SHARED_TASK_CONTEXT = paper_context_from_title(
    "Alpha Team at SemEval-2030 Task 9: Detecting Invented Phenomena"
)
OVERVIEW_CONTEXT = paper_context_from_title("Overview of the Invented Phenomena Shared Task")


def _attribute_in_context(
    page_text: str, quote: str, row: str, value: str, context: PaperContext | None
) -> AttributionVerdict:
    fragment = _page(page_text, 3)
    return attribute_candidate(
        _candidate(quote, row, value, page=3),
        build_page_region_index(fragment),
        fragment,
        load_lexicon(),
        context,
    )


def test_table_level_cues_are_recorded_on_a_trusted_row_and_abstain() -> None:
    verdict = _attribute_in_context(RANKED_PAGE, "Beta Team 0.81", "Beta Team", "0.81", None)

    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.rule_id == "weak_cue_recorded"
    assert {cue.cue_id for cue in verdict.cues} == {"rank_column", "leaderboard_caption"}
    assert {cue.scope for cue in verdict.cues} == {"table_header", "caption"}
    assert has_table_level_cue(verdict, load_lexicon())
    # A cue that fired is doubt, not silence: no basis under either policy.
    assert verdict.origin_basis(ClaimType.PRIMARY_RESULT) is OriginBasis.NONE
    assert not OriginExportPolicy.TIERED.permits(verdict.origin_basis(ClaimType.PRIMARY_RESULT))


def test_table_level_cues_are_recorded_on_the_untrusted_row_path_without_changing_state() -> None:
    verdict = _attribute_in_context(RANKED_PAGE, "Beta Team 0.81", "Gamma Team", "0.81", None)

    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.rule_id == "row_unresolved"
    assert {cue.cue_id for cue in verdict.cues} == {"rank_column", "leaderboard_caption"}


def test_a_decisive_row_cue_still_decides_inside_a_ranked_table() -> None:
    page = RANKED_PAGE.replace("Beta Team   ", "Beta et al. ")
    verdict = _attribute_in_context(page, "Beta et al. 0.81", "Beta et al.", "0.81", None)

    assert verdict.state is AttributionState.EXTERNALLY_SOURCED
    assert verdict.cues[0].cue_id == "attributed_author"
    assert {cue.cue_id for cue in verdict.cues[1:]} == {"rank_column", "leaderboard_caption"}


@pytest.mark.parametrize(
    ("row", "fires"),
    [
        ("Beta Team", True),
        ("Alpha Team", False),
        ("Ours", False),
        ("Run 1", False),
    ],
)
def test_foreign_team_row_spares_the_titles_team_and_self_references(row: str, fires: bool) -> None:
    quote = next(line.strip() for line in RANKED_PAGE.splitlines() if row in line)
    verdict = _attribute_in_context(RANKED_PAGE, quote, row, quote.split()[-1], SHARED_TASK_CONTEXT)

    # A self-referential label is the one row that the table cues do not hold back.
    if row == "Ours":
        assert verdict.state is AttributionState.NO_SIGNAL
        assert verdict.rule_id == "self_reference_row"
    else:
        assert verdict.state is AttributionState.UNRESOLVED
    assert ("foreign_team_row" in {cue.cue_id for cue in verdict.cues}) is fires
    assert row_names_team(SHARED_TASK_CONTEXT, row) is (row == "Alpha Team")
    # The prior is read from the title and the table, so its cue never wears the one
    # scope that may decide.
    assert {cue.scope for cue in verdict.cues if cue.cue_id == "foreign_team_row"} <= {
        PAPER_CONTEXT_CUE_SCOPE
    }
    assert all(cue.scope != "row_label" for cue in verdict.cues if cue.cue_id != "self_reference")


def test_foreign_team_row_needs_a_table_level_cue_and_a_trusted_row() -> None:
    plain = _attribute_in_context(
        PLAIN_PAGE, "Beta Team 0.81", "Beta Team", "0.81", SHARED_TASK_CONTEXT
    )
    untrusted = _attribute_in_context(
        RANKED_PAGE, "Beta Team 0.81", "Gamma", "0.81", SHARED_TASK_CONTEXT
    )

    assert plain.state is AttributionState.NO_SIGNAL
    assert plain.cues == []
    assert untrusted.rule_id == "row_unresolved"
    assert "foreign_team_row" not in {cue.cue_id for cue in untrusted.cues}


def test_a_caption_cue_outside_the_table_level_set_never_triggers_the_prior() -> None:
    """`caption_reported` is recorded but is not a table-level cue, so the team's
    absence from the row label says nothing and `foreign_team_row` stays silent."""

    verdict = _attribute_in_context(
        MIXED_ORIGIN_PAGE, "Paper System 0.742", "Paper System", "0.742", SHARED_TASK_CONTEXT
    )

    assert verdict.rule_id == "weak_cue_recorded"
    assert [cue.cue_id for cue in verdict.cues] == ["caption_reported"]
    assert not has_table_level_cue(verdict, load_lexicon())


def test_overview_paper_rows_fires_on_every_row_of_a_cue_bearing_table() -> None:
    trusted = _attribute_in_context(RANKED_PAGE, "Ours 0.79", "Ours", "0.79", OVERVIEW_CONTEXT)
    untrusted = _attribute_in_context(
        RANKED_PAGE, "Beta Team 0.81", "Gamma", "0.81", OVERVIEW_CONTEXT
    )
    plain = _attribute_in_context(
        PLAIN_PAGE, "Beta Team 0.81", "Beta Team", "0.81", OVERVIEW_CONTEXT
    )

    assert "overview_paper_rows" in {cue.cue_id for cue in trusted.cues}
    assert "overview_paper_rows" in {cue.cue_id for cue in untrusted.cues}
    assert {
        cue.scope for cue in (*trusted.cues, *untrusted.cues) if cue.cue_id == "overview_paper_rows"
    } == {PAPER_CONTEXT_CUE_SCOPE}
    # "Ours" names the paper itself, so the overview cue is recorded but does not
    # hold that row back; any other trusted row in the table stays unresolved.
    assert trusted.state is AttributionState.NO_SIGNAL
    assert trusted.rule_id == "self_reference_row"
    other = _attribute_in_context(
        RANKED_PAGE, "Beta Team 0.81", "Beta Team", "0.81", OVERVIEW_CONTEXT
    )
    assert other.state is AttributionState.UNRESOLVED
    assert "overview_paper_rows" in {cue.cue_id for cue in other.cues}
    assert plain.cues == []


@pytest.mark.parametrize(
    "label",
    ["Ours", "Our system", "This work", "Proposed model", "BERT (ours)"],
)
def test_a_self_referential_row_label_keeps_its_row_out_of_the_table_cues(label: str) -> None:
    page = RANKED_PAGE.replace("Ours        ", f"{label:<12}")
    quote = next(line.strip() for line in page.splitlines() if label in line)
    verdict = _attribute_in_context(page, quote, label, "0.79", SHARED_TASK_CONTEXT)

    assert verdict.state is AttributionState.NO_SIGNAL
    assert verdict.rule_id == "self_reference_row"
    assert verdict.cues[0].cue_id == "self_reference"
    assert verdict.cues[0].scope == "row_label"
    assert {cue.cue_id for cue in verdict.cues[1:]} >= {"rank_column", "leaderboard_caption"}
    assert verdict.origin_basis(ClaimType.PRIMARY_RESULT) is (
        OriginBasis.MODEL_ASSERTED_PRIMARY_NO_EXTERNAL_CUE
    )
    assert OriginExportPolicy.TIERED.permits(verdict.origin_basis(ClaimType.PRIMARY_RESULT))
    assert not OriginExportPolicy.POSITIVE_ONLY.permits(
        verdict.origin_basis(ClaimType.PRIMARY_RESULT)
    )


def test_a_self_referential_label_with_a_citation_bracket_still_abstains() -> None:
    page = RANKED_PAGE.replace("Ours        ", "Ours [3]    ")
    verdict = _attribute_in_context(page, "Ours [3] 0.79", "Ours [3]", "0.79", None)

    assert verdict.state is AttributionState.UNRESOLVED
    assert verdict.rule_id == "weak_cue_recorded"
    assert "citation_bracket" in {cue.cue_id for cue in verdict.cues}


def test_without_a_paper_context_the_prior_stays_silent() -> None:
    verdict = _attribute_in_context(RANKED_PAGE, "Beta Team 0.81", "Beta Team", "0.81", None)

    assert {cue.cue_id for cue in verdict.cues} == {"rank_column", "leaderboard_caption"}


@pytest.mark.parametrize(
    ("title", "kind", "team"),
    [
        (
            "Alpha Team at SemEval-2030 Task 9: Detecting Invented Phenomena",
            "shared_task_system_paper",
            "Alpha Team",
        ),
        (
            "CoLab at UdX at SemEval-2030 Task 9: Ensembles",
            "shared_task_system_paper",
            "CoLab at UdX",
        ),
        (
            "Maple Leaf@CASE 2031: Task-Specific Adaptation",
            "shared_task_system_paper",
            "Maple Leaf",
        ),
        (
            "Team Juniper at the ExampleEval 2030 Shared Task 1 on Invented Detection",
            "shared_task_system_paper",
            "Team Juniper",
        ),
        ("ABC-NLP at MultiThing: Continual Learning", "shared_task_system_paper", "ABC-NLP"),
        ("Overview of the Invented Phenomena Shared Task", "overview_paper", None),
        ("Findings of the 2030 Invented Shared Task", "overview_paper", None),
        ("GRUPPO at MultiPRIDE: Reclamatory Intent", "shared_task_system_paper", "GRUPPO"),
        ("A Look at Invented Detection: Lessons", "other", None),
        ("Looking at BERT: What It Knows", "other", None),
        ("Hate at First Sight: A Study", "other", None),
        ("Detecting Invented Compounds - An Unsupervised Approach", "other", None),
        ("", "other", None),
        (None, "other", None),
    ],
)
def test_paper_context_is_derived_from_the_title_shape(
    title: str | None, kind: str, team: str | None
) -> None:
    context = paper_context_from_title(title)

    assert context.paper_kind is PaperKind(kind)
    assert context.team_name == team
    assert context.title == (title or None)


@pytest.mark.parametrize(
    ("team", "label", "names"),
    [
        ("Team Juniper", "Juniper run 2", True),
        ("Alpha Team", "alpha-team (ours)", True),
        ("NS", "XLM-R ensemble", False),
        ("Alpha Team", "Team Alpha", False),
        ("Alpha Team", None, False),
    ],
)
def test_row_names_team_matches_whole_words_only(team: str, label: str | None, names: bool) -> None:
    context = PaperContext(paper_kind=PaperKind.SHARED_TASK_SYSTEM_PAPER, team_name=team)

    assert row_names_team(context, label) is names
    assert row_names_team(None, label) is False


def test_the_default_lexicon_is_v2_and_the_earlier_ones_still_load() -> None:
    current = load_lexicon()
    previous = load_lexicon(ATTRIBUTION_LEXICON_PATHS["attribution-cues-v0"])
    v1 = load_lexicon(ATTRIBUTION_LEXICON_PATHS["attribution-cues-v1"])

    assert current.lexicon_id == "attribution-cues-v2"
    assert {cue.cue_id for cue in current.self_reference_row} == {"self_reference"}
    assert v1.lexicon_id == "attribution-cues-v1"
    assert v1.self_reference_row == ()
    assert {cue.cue_id for cue in current.abstaining_table_header} == {"rank_column"}
    assert {cue.cue_id for cue in current.abstaining_paper_context} == {
        "foreign_team_row",
        "overview_paper_rows",
    }
    assert previous.lexicon_id == "attribution-cues-v0"
    assert previous.abstaining_table_header == ()
    assert previous.abstaining_paper_context == ()
    assert current.sha256 != previous.sha256


def test_a_recorded_lexicon_loads_from_the_shipped_copy_it_names() -> None:
    """A replay runs under the lexicon the run recorded, even when the default moved on."""

    previous = load_lexicon(ATTRIBUTION_LEXICON_PATHS["attribution-cues-v0"])

    recorded = load_recorded_lexicon(previous.lexicon_id, previous.sha256)

    assert (recorded.lexicon_id, recorded.sha256) == (previous.lexicon_id, previous.sha256)
    assert recorded.lexicon_id != load_lexicon().lexicon_id


@pytest.mark.parametrize(
    ("lexicon_id", "sha256"),
    [
        ("attribution-cues-v9", "0" * 64),
        ("attribution-cues-v1", "0" * 64),
        (None, None),
    ],
)
def test_a_recorded_lexicon_that_is_missing_or_changed_is_refused(
    lexicon_id: str | None, sha256: str | None
) -> None:
    with pytest.raises(ValueError):
        load_recorded_lexicon(lexicon_id, sha256)


def test_a_lexicon_whose_table_level_cues_name_no_table_cue_is_refused(tmp_path) -> None:
    text = DEFAULT_ATTRIBUTION_LEXICON_PATH.read_text(encoding="utf-8")
    assert "table_level_cues:\n  - rank_column\n" in text
    path = tmp_path / "attribution-cues-bad.yaml"
    path.write_text(
        text.replace(
            "table_level_cues:\n  - rank_column\n", "table_level_cues:\n  - citation_bracket\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="citation_bracket"):
        load_lexicon(path)
