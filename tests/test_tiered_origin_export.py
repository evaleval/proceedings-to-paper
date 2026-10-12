"""Tiered producer-origin export, descriptive field provenance, and offline replay.

Canonical export means positively established origin, and the deterministic resolver
never establishes it. These tests pin the two things that make a tier honest: the default
policy behaves exactly as before, and a record granted export under the tiered policy
carries the basis it was granted on.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proceedings_to_eee.census_recompose import (
    PaperRecomposition,
    recompose_census,
    recompose_paper,
)
from proceedings_to_eee.census_review import candidate_payload_sha256
from proceedings_to_eee.domain.attribution import (
    AttributionState,
    AttributionVerdict,
    OriginBasis,
    OriginExportPolicy,
)
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
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.resolution.metrics import resolve_metric
from proceedings_to_eee.resources import LEGACY_EEE_SCHEMA_PATH
from proceedings_to_eee.validation.candidates import (
    candidate_origin_basis,
    route_candidate_attribution,
    validate_candidates,
)
from proceedings_to_eee.validation.eee_schema import load_schema
from proceedings_to_eee.validation.field_provenance import quote_field_provenance

SOURCE_ID = "src_tiered_fixture"
PAPER_ID = "tiered-fixture"
QUOTE = "PerspectiveAPI reaches 0.81 macro F1 on the HateCheck test split."
PAGE_TEXT = f"Results\n\n{QUOTE}\n\nTable 1: Scores.\n"

# A ranked results table: the header has a rank column and the caption names the official
# standings, so the deterministic verdict of every row in it records a table-level cue.
RANKED_PAGE_TEXT = """Table 2: Official results of the shared task.

     Rank    Team                 macro F1
     1       Alpha Team           0.83
     2       PerspectiveAPI       0.81
"""
RANKED_ANCHOR = dict(
    label="Table 2", row="PerspectiveAPI", column="macro F1", quote="PerspectiveAPI 0.81"
)

# A row-scoped decisive cue next to a paper-owned row: externally sourced.
LEADERBOARD_ROW_PAGE_TEXT = """Table 2: Results.

     System                                   macro F1
     Local Baseline                           0.62
     PerspectiveAPI (Leaderboard Entry)       0.81
"""
LEADERBOARD_ROW_ANCHOR = dict(
    label="Table 2",
    row="PerspectiveAPI (Leaderboard Entry)",
    column="macro F1",
    quote="PerspectiveAPI (Leaderboard Entry) 0.81",
)


def _manifest(title: str = "Tiered fixture") -> str:
    return json.dumps(
        {
            "schema_version": "source-manifest/0.1",
            "paper_id": PAPER_ID,
            "title": title,
            "sources": [
                {
                    "source_id": SOURCE_ID,
                    "paper_id": PAPER_ID,
                    "role": "paper",
                    "original_uri": "https://example.invalid/paper.pdf",
                    "resolved_uri": "https://example.invalid/paper.pdf",
                    "retrieved_at": "2026-09-04T00:00:00+00:00",
                    "sha256": "0" * 64,
                    "byte_size": 10,
                    "media_type": "application/pdf",
                    "cache_relpath": "data/sources/fixture.pdf",
                    "access_status": "available",
                    "license_disposition": "unknown",
                }
            ],
        }
    )


_MANIFEST = _manifest()


def _layout(text: str = PAGE_TEXT) -> PdfLayout:
    return PdfLayout(
        source_id=SOURCE_ID,
        parser="fixture",
        parser_version="fixture-1",
        page_count=1,
        pages=[
            PageFragment(
                fragment_id="fragment-1",
                source_id=SOURCE_ID,
                page=1,
                text=text,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                character_count=len(text),
                numeric_token_count=1,
                result_signal_score=1,
            )
        ],
    )


def _candidate(**overrides: object) -> CandidateObservation:
    payload: dict[str, object] = {
        # 0.3 is the version the field-provenance gate applies to.
        "schema_version": "candidate-observation/0.3",
        "paper_id": PAPER_ID,
        "claim_type": ClaimType.PRIMARY_RESULT,
        "extraction_confidence": 0.95,
        "roles": [RoleAssignment(role=ActorRole.EVALUATED_SYSTEM, raw_name="PerspectiveAPI")],
        "scope": ObservationScope(dataset_raw="HateCheck", split="test"),
        "metric": MetricSpec(raw_name="macro F1", canonical_id="macro_f1", unit="proportion"),
        "value": ReportedValue(raw="0.81", numeric=0.81, unit="proportion"),
        "evidence": [
            EvidenceAnchor(
                source_id=SOURCE_ID,
                page=1,
                kind=EvidenceKind.PROSE,
                quote=QUOTE,
            )
        ],
    }
    payload.update(overrides)
    candidate = CandidateObservation(**payload)
    # Provenance is attached at extraction time, not by validation, so a fixture that
    # wants the field-provenance gate to run has to bind it the same way.
    return candidate.model_copy(update={"field_provenance": quote_field_provenance(candidate)})


def test_the_default_policy_still_demotes_everything_the_resolver_cannot_establish() -> None:
    candidate = _candidate()
    candidate.export_status = ExportStatus.ELIGIBLE

    route_candidate_attribution([candidate], {})

    assert candidate.export_status is ExportStatus.NEEDS_REVIEW
    assert candidate.export_reason is not None
    assert candidate.export_reason.startswith("attribution=")
    assert candidate.attribution is not None
    assert candidate.attribution.state is AttributionState.UNRESOLVED


def test_the_tiered_policy_keeps_it_eligible_and_names_the_basis() -> None:
    candidate = _candidate()
    candidate.export_status = ExportStatus.ELIGIBLE

    route_candidate_attribution([candidate], {}, OriginExportPolicy.TIERED)

    assert candidate.export_status is ExportStatus.ELIGIBLE
    assert candidate_origin_basis(candidate) is OriginBasis.MODEL_ASSERTED_PRIMARY_UNCHECKED
    # The verdict itself is untouched: nothing here calls the number paper-produced.
    assert candidate.attribution is not None
    assert candidate.attribution.state is AttributionState.UNRESOLVED


def test_an_externally_sourced_candidate_never_exports_under_any_policy() -> None:
    candidate = _candidate()
    candidate.export_status = ExportStatus.ELIGIBLE
    candidate.attribution = AttributionVerdict(
        state=AttributionState.EXTERNALLY_SOURCED, rule_id="row_scoped_foreign_cue"
    )

    assert candidate_origin_basis(candidate) is OriginBasis.NONE
    assert not OriginExportPolicy.TIERED.permits(OriginBasis.NONE)
    assert not OriginExportPolicy.POSITIVE_ONLY.permits(OriginBasis.NONE)


def test_descriptive_setting_prose_no_longer_gates_but_a_literal_still_does() -> None:
    """`operationalization` is descriptive prose; `language` requires literal binding."""

    described = _candidate(operationalization="threshold 0.7 on the toxicity score")
    literal = _candidate(scope=ObservationScope(dataset_raw="HateCheck", language="Portuguese"))
    layouts = {SOURCE_ID: _layout()}

    validate_candidates([described], layouts, min_confidence=0.8)
    validate_candidates([literal], layouts, min_confidence=0.8)

    # The prose one clears field provenance and stops at attribution instead.
    assert described.export_reason is not None
    assert described.export_reason.startswith("attribution=")
    # The literal one never gets that far.
    assert literal.export_reason == "field_provenance=ambiguous_or_unsupported"


def test_recompose_writes_tiered_records_and_leaves_observations_untouched(
    tmp_path: Path,
) -> None:
    paper_dir = tmp_path / PAPER_ID
    (paper_dir / "private").mkdir(parents=True)
    candidate = _candidate()
    validate_candidates([candidate], {SOURCE_ID: _layout()}, min_confidence=0.8)
    observations = paper_dir / "observations.jsonl"
    observations.write_text(json.dumps(candidate.model_dump(mode="json")) + "\n", encoding="utf-8")
    before = observations.read_bytes()
    (paper_dir / "private" / "layout.json").write_text(
        json.dumps(_layout().model_dump(mode="json")), encoding="utf-8"
    )
    (paper_dir / "source-manifest.json").write_text(_MANIFEST, encoding="utf-8")
    schema, authority = load_schema()

    positive = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.POSITIVE_ONLY,
        schema=schema,
        schema_version=authority.version,
    )
    tiered = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )

    assert positive.records == []
    assert len(tiered.records) == 1
    details = tiered.records[0]["source_metadata"]["additional_details"]
    assert details["review_tier"] == "deterministic"
    assert details["producer_origin_basis"] == "model_asserted_primary_unchecked"
    assert details["origin_export_policy"] == "tiered"
    result_details = tiered.records[0]["evaluation_results"][0]["score_details"]["details"]
    assert result_details["producer_origin_basis"] == "model_asserted_primary_unchecked"
    assert result_details["attribution_state"] == "unresolved"
    assert result_details["review_tier"] == "deterministic"
    assert result_details["evidence_page"] == "1"
    # The cues that fired are visible on every result; none fired on this prose anchor.
    assert result_details["attribution_cues"] == ""
    # The replay reads the run; it never rewrites it.
    assert observations.read_bytes() == before
    # And it is idempotent.
    again = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )
    assert again.records == tiered.records


def test_a_reviewed_candidate_composes_at_the_model_reviewed_tier(tmp_path: Path) -> None:
    """This tier records a failed deterministic gate without treating it as a blocker."""

    from proceedings_to_eee.census_review import candidate_payload_sha256
    from proceedings_to_eee.validation.eee_schema import validate_eee_record

    paper_dir = tmp_path / PAPER_ID
    (paper_dir / "private").mkdir(parents=True)
    # A literal setting component that is not in the quote, so field provenance fails.
    candidate = _candidate(
        scope=ObservationScope(dataset_raw="HateCheck", split="test", language="Portuguese")
    )
    validate_candidates([candidate], {SOURCE_ID: _layout()}, min_confidence=0.8)
    assert candidate.export_status is not ExportStatus.ELIGIBLE
    record = candidate.model_dump(mode="json")
    (paper_dir / "observations.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (paper_dir / "private" / "layout.json").write_text(
        json.dumps(_layout().model_dump(mode="json")), encoding="utf-8"
    )
    (paper_dir / "source-manifest.json").write_text(_MANIFEST, encoding="utf-8")
    (paper_dir / "model-review-gates.json").write_text(
        json.dumps(
            {
                "schema_version": "census-model-review-gates/0.1",
                "model": "fixture/model",
                "gates": {
                    str(record["observation_id"]): {
                        "decision": "accept",
                        "candidate_payload_sha256": candidate_payload_sha256(record),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    schema, authority = load_schema()

    tiered = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )

    assert len(tiered.records) == 1
    details = tiered.records[0]["source_metadata"]["additional_details"]
    assert details["review_tier"] == "model_reviewed"
    assert details["producer_origin_basis"] == "model_reviewed_origin_quote"
    result_details = tiered.records[0]["evaluation_results"][0]["score_details"]["details"]
    # The deterministic failure is carried in the record rather than hidden by the tier.
    assert result_details["field_provenance_gate"] == "unbound"
    assert validate_eee_record(tiered.records[0], schema) == []

    # A gate written for a different tuple is ignored, not trusted.
    (paper_dir / "model-review-gates.json").write_text(
        json.dumps(
            {
                "schema_version": "census-model-review-gates/0.1",
                "model": "fixture/model",
                "gates": {
                    str(record["observation_id"]): {
                        "decision": "accept",
                        "candidate_payload_sha256": "0" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    stale = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )
    assert stale.records == []
    assert stale.stale_review_gates == 1


def _reviewed_paper(tmp_path: Path, candidate: CandidateObservation, decision: str) -> Path:
    """Write one validated candidate as a run paper with a bare reviewer gate over it:
    no proposed decision and no origin quote, as the earliest gate files were written."""

    return _write_paper(tmp_path, candidate, gate={"decision": decision})


def _unbounded_candidate() -> CandidateObservation:
    # chrf is identity-only: the registry gives it no unit and no bounds.
    chrf = resolve_metric(MetricSpec(raw_name="ChrF"))
    assert chrf.min_score is None and chrf.max_score is None
    return _candidate(metric=chrf, value=ReportedValue(raw="0.62", numeric=0.62))


def test_a_metric_with_no_known_scale_composes_with_explicit_null_bounds(
    tmp_path: Path,
) -> None:
    """Unknown bounds must not be invented. The default authority can say "unknown", so
    the candidate composes with explicit nulls, labelled, and is counted as such."""

    from proceedings_to_eee.validation.eee_schema import validate_eee_record

    # A reviewer accept is what carried these past the reference gate in production.
    paper_dir = _reviewed_paper(tmp_path, _unbounded_candidate(), "accept_tuple_only")
    schema, authority = load_schema()

    tiered = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )

    assert len(tiered.records) == 1
    assert tiered.composed_with_unresolved_bounds == 1
    assert tiered.skipped_unbounded_metric == 0
    assert tiered.invalid_records == []
    [result] = tiered.records[0]["evaluation_results"]
    assert result["metric_config"]["score_type"] == "continuous"
    assert result["metric_config"]["min_score"] is None
    assert result["metric_config"]["max_score"] is None
    assert result["score_details"]["details"]["metric_bounds"] == "unresolved"
    assert result["score_details"]["details"]["metric_identity_source"] == "registry"
    assert validate_eee_record(tiered.records[0], schema) == []


def test_a_metric_with_no_known_scale_is_left_out_and_counted_under_the_legacy_schema(
    tmp_path: Path,
) -> None:
    """EEE 0.2.2 requires min and max for a continuous score, and its metric_config `if`
    matches vacuously when score_type is absent, so such a record is rejected either
    way. There the candidate is left out and counted instead of producing an invalid
    record."""

    paper_dir = _reviewed_paper(tmp_path, _unbounded_candidate(), "accept_tuple_only")
    schema, authority = load_schema(LEGACY_EEE_SCHEMA_PATH)
    assert not authority.allows_null_bounds

    tiered = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )

    assert tiered.records == []
    assert tiered.skipped_unbounded_metric == 1
    assert tiered.composed_with_unresolved_bounds == 0
    # Nothing invalid was written; the exclusion is counted, not a failed validation.
    assert tiered.invalid_records == []


@pytest.mark.parametrize(
    ("value", "notes", "blocker"),
    [
        (
            ReportedValue(raw="+6.7", numeric=6.7, unit="percent"),
            ["Few-shot percentual change in macro F1."],
            "signed_difference_value",
        ),
        (
            ReportedValue(raw="146%", numeric=146.0, unit="percent"),
            [],
            "value_outside_registry_metric_bounds",
        ),
    ],
)
def test_a_reviewer_accept_cannot_export_a_value_that_is_not_a_score(
    tmp_path: Path, value: ReportedValue, notes: list[str], blocker: str
) -> None:
    """The model-reviewed tier may override deterministic tuple failures, but a signed
    difference or an out-of-range number is not a score on the metric's scale, and no
    reviewer decision about the tuple changes that."""

    candidate = _candidate(
        metric=MetricSpec(raw_name="macro F1", canonical_id="macro_f1", unit="percent"),
        value=value,
        notes=notes,
    )
    paper_dir = _reviewed_paper(tmp_path, candidate, "accept")
    schema, authority = load_schema()

    tiered = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )

    assert tiered.records == []
    assert tiered.skipped_value_blockers == {blocker: 1}
    assert tiered.skipped_no_metric_id == 0


# ----------------------------------------------------------------------------------
# The model_reviewed promotion rule: the cue wins, and withheld origin grants nothing
# ----------------------------------------------------------------------------------

ACCEPT_GATE = {"decision": "accept", "proposed_decision": "accept", "origin_quote_sha256": "a" * 64}


def _write_paper(
    tmp_path: Path,
    candidate: CandidateObservation,
    *,
    page_text: str = PAGE_TEXT,
    title: str = "Tiered fixture",
    gate: dict[str, Any] | None = None,
    review_log: list[dict[str, Any]] | None = None,
) -> Path:
    """Lay out one paper the way a census run leaves it, with an optional gate and log."""

    paper_dir = tmp_path / PAPER_ID
    (paper_dir / "private").mkdir(parents=True, exist_ok=True)
    layout = _layout(page_text)
    validate_candidates([candidate], {SOURCE_ID: layout}, min_confidence=0.8)
    record = candidate.model_dump(mode="json")
    payload_sha256 = candidate_payload_sha256(record)
    (paper_dir / "observations.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (paper_dir / "private" / "layout.json").write_text(
        json.dumps(layout.model_dump(mode="json")), encoding="utf-8"
    )
    (paper_dir / "source-manifest.json").write_text(_manifest(title), encoding="utf-8")
    if gate is not None:
        (paper_dir / "model-review-gates.json").write_text(
            json.dumps(
                {
                    "schema_version": "census-model-review-gates/0.1",
                    "model": "fixture/model",
                    "gates": {
                        str(record["observation_id"]): {
                            "candidate_payload_sha256": payload_sha256,
                            **gate,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
    if review_log is not None:
        lines = [
            {
                "schema_version": "candidate-page-review/0.1",
                "observation_id": str(record["observation_id"]),
                "candidate_payload_sha256": payload_sha256,
                "decision": (gate or {}).get("decision"),
                "rationale": None,
                **entry,
            }
            for entry in review_log
        ]
        (paper_dir / "private" / "model-review.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
        )
    return paper_dir


def _recompose(paper_dir: Path, **options: Any) -> PaperRecomposition:
    schema, authority = load_schema()
    return recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
        **options,
    )


def _reviewed_candidate() -> CandidateObservation:
    """A prose candidate only the reviewer can carry past the gates: its literal
    `language` is not in the quote, so field provenance holds it in review."""

    return _candidate(
        scope=ObservationScope(dataset_raw="HateCheck", split="test", language="Portuguese")
    )


def _table_candidate(anchor: dict[str, str]) -> CandidateObservation:
    return _candidate(
        evidence=[EvidenceAnchor(source_id=SOURCE_ID, page=1, kind=EvidenceKind.TABLE, **anchor)]
    )


@pytest.mark.parametrize(
    ("page_text", "anchor", "state"),
    [
        (RANKED_PAGE_TEXT, RANKED_ANCHOR, "unresolved"),
        (LEADERBOARD_ROW_PAGE_TEXT, LEADERBOARD_ROW_ANCHOR, "externally_sourced"),
    ],
)
def test_an_accept_gate_on_a_cue_bearing_verdict_composes_nothing_without_crashing(
    tmp_path: Path, page_text: str, anchor: dict[str, str], state: str
) -> None:
    candidate = _table_candidate(anchor)
    paper_dir = _write_paper(tmp_path, candidate, page_text=page_text, gate=ACCEPT_GATE)
    assert candidate.attribution is not None
    assert candidate.attribution.state.value == state
    assert candidate.attribution.cues

    tiered = _recompose(paper_dir)

    assert tiered.records == []
    assert tiered.invalid_records == []
    assert tiered.skipped_cue_overrides_review == 1
    assert tiered.skipped_no_origin_basis == 0
    (reason,) = tiered.review_demotions
    assert reason.startswith(f"accept -> none: cue_overrides_review:{state}:")
    assert all(cue.cue_id in reason for cue in candidate.attribution.cues)
    assert not (paper_dir / "eee-tiered").exists()


def test_an_accept_tuple_only_gate_on_a_cue_bearing_verdict_names_the_cues(tmp_path: Path) -> None:
    """A tuple-only accept keeps the verdict's own basis, and when that is none the
    reason says which cues held it back, not only the rule."""

    candidate = _table_candidate(RANKED_ANCHOR)
    gate = {
        "decision": "accept_tuple_only",
        "proposed_decision": "accept_tuple_only",
        "origin_quote_sha256": "b" * 64,
    }
    paper_dir = _write_paper(tmp_path, candidate, page_text=RANKED_PAGE_TEXT, gate=gate)
    assert candidate.attribution is not None
    assert candidate.attribution.cues

    tiered = _recompose(paper_dir)

    assert tiered.records == []
    assert tiered.skipped_no_origin_basis == 1
    assert tiered.skipped_cue_overrides_review == 0
    assert tiered.skipped_reviewer_withheld_origin == 0
    (reason,) = tiered.review_demotions
    assert reason.startswith(
        f"accept_tuple_only -> none: no_origin_basis:{candidate.attribution.state.value}:"
        f"{candidate.attribution.rule_id}:"
    )
    assert all(cue.cue_id in reason for cue in candidate.attribution.cues)


def test_an_accept_gate_on_a_cue_free_verdict_still_earns_the_reviewed_origin_basis(
    tmp_path: Path,
) -> None:
    paper_dir = _write_paper(tmp_path, _reviewed_candidate(), gate=ACCEPT_GATE)

    tiered = _recompose(paper_dir)

    assert len(tiered.records) == 1
    assert tiered.review_demotions == Counter()
    details = tiered.records[0]["evaluation_results"][0]["score_details"]["details"]
    assert details["producer_origin_basis"] == "model_reviewed_origin_quote"
    assert details["review_tier"] == "model_reviewed"


@pytest.mark.parametrize(
    ("proposed", "origin_quote_sha256", "withheld"),
    [
        ("accept_tuple_only", None, True),
        ("accept_tuple_only", "b" * 64, False),
        ("accept", None, False),
    ],
)
def test_a_reviewer_that_withheld_origin_grants_no_basis(
    tmp_path: Path, proposed: str, origin_quote_sha256: str | None, withheld: bool
) -> None:
    gate = {
        "decision": "accept_tuple_only",
        "proposed_decision": proposed,
        "origin_quote_sha256": origin_quote_sha256,
    }
    paper_dir = _write_paper(tmp_path, _reviewed_candidate(), gate=gate)

    tiered = _recompose(paper_dir)

    assert tiered.skipped_reviewer_withheld_origin == (1 if withheld else 0)
    assert len(tiered.records) == (0 if withheld else 1)
    if withheld:
        assert tiered.review_demotions == Counter(
            {"accept_tuple_only -> none: reviewer_withheld_origin": 1}
        )
    else:
        details = tiered.records[0]["evaluation_results"][0]["score_details"]["details"]
        assert details["producer_origin_basis"] == "model_asserted_primary_unchecked"
        assert details["review_tier"] == "model_reviewed"


def test_the_paper_context_is_read_from_the_manifest_title_and_counted(tmp_path: Path) -> None:
    """The team's own row in a ranked table is held back by the table cue, never by
    `foreign_team_row`, and the summary says how many such rows there were."""

    candidate = _table_candidate(RANKED_ANCHOR)
    paper_dir = _write_paper(
        tmp_path,
        candidate,
        page_text=RANKED_PAGE_TEXT,
        title="PerspectiveAPI at SemEval-2030 Task 9: Invented Detection",
        gate=ACCEPT_GATE,
    )

    tiered = _recompose(paper_dir)
    summary = recompose_census(run_root=tmp_path, origin_policy=OriginExportPolicy.TIERED)

    assert tiered.paper_kind == "shared_task_system_paper"
    assert tiered.attribution_cue_counts == Counter({"rank_column": 1, "leaderboard_caption": 1})
    assert tiered.team_rows_in_cue_bearing_tables == 1
    assert tiered.team_rows_with_foreign_team_cue == 0
    assert tiered.skipped_cue_overrides_review == 1
    assert summary["paper_kinds"] == {"shared_task_system_paper": 1}
    assert summary["attribution_cue_counts"] == {"leaderboard_caption": 1, "rank_column": 1}
    assert summary["team_rows_in_cue_bearing_tables"] == 1
    assert summary["team_rows_with_foreign_team_cue"] == 0
    assert summary["skipped_cue_overrides_review"] == 1
    assert summary["attribution_lexicon_id"] == "attribution-cues-v2"


# The same ranked table with the paper's own row labelled as such.
OWN_ROW_RANKED_PAGE_TEXT = """Table 2: Official results of the shared task.

     Rank    Team                 macro F1
     1       Alpha Team           0.83
     2       Ours                 0.81
"""
OWN_ROW_RANKED_ANCHOR = dict(label="Table 2", row="Ours", column="macro F1", quote="Ours 0.81")


@pytest.mark.parametrize(
    ("gate", "basis"),
    [
        (
            {
                "decision": "accept_tuple_only",
                "proposed_decision": "accept_tuple_only",
                "origin_quote_sha256": "b" * 64,
            },
            "model_asserted_primary_no_external_cue",
        ),
        (ACCEPT_GATE, "model_reviewed_origin_quote"),
    ],
)
def test_a_row_that_names_the_paper_itself_is_not_held_back_by_table_cues(
    tmp_path: Path, gate: dict[str, object], basis: str
) -> None:
    """The rank column still describes the table, and the record says so, but the
    paper's own row keeps the basis its review earned."""

    candidate = _table_candidate(OWN_ROW_RANKED_ANCHOR)
    paper_dir = _write_paper(
        tmp_path,
        candidate,
        page_text=OWN_ROW_RANKED_PAGE_TEXT,
        title="Alpha Team at SemEval-2030 Task 9: Invented Detection",
        gate=gate,
    )
    assert candidate.attribution is not None
    assert candidate.attribution.rule_id == "self_reference_row"
    assert {cue.cue_id for cue in candidate.attribution.cues} >= {"rank_column", "self_reference"}

    tiered = _recompose(paper_dir)

    assert len(tiered.records) == 1
    assert tiered.skipped_cue_overrides_review == 0
    assert tiered.skipped_no_origin_basis == 0
    (record,) = tiered.records
    (result,) = record["evaluation_results"]
    details = result["score_details"]["details"]
    assert details["producer_origin_basis"] == basis
    assert "rank_column" in details["attribution_cues"]
    assert "self_reference" in details["attribution_cues"]


@pytest.mark.parametrize(
    ("rationale", "flagged"),
    [
        ("Row is copied from the organisers' leaderboard table.", True),
        ("Score belongs to another team's submission.", True),
        ("Participants are listed with their official ranks.", True),
        ("The value and metric match the quoted table cell.", False),
    ],
)
def test_opt_in_rationale_demotion_reads_the_private_log_and_never_copies_it(
    tmp_path: Path, rationale: str, flagged: bool
) -> None:
    paper_dir = _write_paper(
        tmp_path, _reviewed_candidate(), gate=ACCEPT_GATE, review_log=[{"rationale": rationale}]
    )

    default = _recompose(paper_dir)
    opted = _recompose(paper_dir, demote_reviewer_flagged_origin=True)
    summary = recompose_census(
        run_root=tmp_path,
        origin_policy=OriginExportPolicy.TIERED,
        demote_reviewer_flagged_origin=True,
    )

    assert len(default.records) == 1
    assert default.skipped_reviewer_rationale_flagged == 0
    assert len(opted.records) == (0 if flagged else 1)
    assert opted.skipped_reviewer_rationale_flagged == (1 if flagged else 0)
    assert opted.reviewer_rationale_log_missing is False
    if flagged:
        assert opted.review_demotions == Counter({"accept -> none: reviewer_rationale_flagged": 1})
    assert summary["demote_reviewer_flagged_origin"] is True
    assert summary["skipped_reviewer_rationale_flagged"] == (1 if flagged else 0)
    assert summary["reviewer_rationale_logs_missing"] == []
    written = (tmp_path / "eee" / "census-eee-summary.json").read_text(encoding="utf-8")
    written += (tmp_path / "eee" / "census-eee.jsonl").read_text(encoding="utf-8")
    assert rationale not in written
    assert rationale not in json.dumps(summary)


def test_only_the_log_record_bound_to_the_gate_decides_the_rationale_flag(tmp_path: Path) -> None:
    """A re-review supersedes its predecessor, and another candidate's record is not ours."""

    superseded = [
        {"rationale": "copied from the leaderboard"},
        {"rationale": "the quoted cell carries the value"},
    ]
    foreign = [{"rationale": "copied from the leaderboard", "candidate_payload_sha256": "f" * 64}]
    for log in (superseded, foreign):
        paper_dir = _write_paper(tmp_path, _reviewed_candidate(), gate=ACCEPT_GATE, review_log=log)

        opted = _recompose(paper_dir, demote_reviewer_flagged_origin=True)

        assert len(opted.records) == 1
        assert opted.skipped_reviewer_rationale_flagged == 0


def test_the_opt_in_flag_reports_a_paper_without_a_private_log(tmp_path: Path) -> None:
    paper_dir = _write_paper(tmp_path, _reviewed_candidate(), gate=ACCEPT_GATE)

    opted = _recompose(paper_dir, demote_reviewer_flagged_origin=True)
    summary = recompose_census(
        run_root=tmp_path,
        origin_policy=OriginExportPolicy.TIERED,
        demote_reviewer_flagged_origin=True,
    )

    assert opted.reviewer_rationale_log_missing is True
    assert len(opted.records) == 1
    assert summary["reviewer_rationale_logs_missing"] == [PAPER_ID]


def test_recompose_census_cli_keeps_the_rationale_flag_off_unless_asked(tmp_path: Path) -> None:
    from proceedings_to_eee.cli import app

    _write_paper(
        tmp_path,
        _reviewed_candidate(),
        gate=ACCEPT_GATE,
        review_log=[{"rationale": "taken from the shared task results table"}],
    )
    runner = CliRunner()

    default = runner.invoke(app, ["recompose-census", str(tmp_path), "--origin-policy", "tiered"])
    opted = runner.invoke(
        app,
        [
            "recompose-census",
            str(tmp_path),
            "--origin-policy",
            "tiered",
            "--demote-reviewer-flagged-origin",
        ],
    )

    assert default.exit_code == 0, default.output
    assert opted.exit_code == 0, opted.output
    assert json.loads(default.output)["demote_reviewer_flagged_origin"] is False
    assert json.loads(default.output)["records"] == 1
    assert json.loads(opted.output)["demote_reviewer_flagged_origin"] is True
    assert json.loads(opted.output)["records"] == 0
    assert json.loads(opted.output)["skipped_reviewer_rationale_flagged"] == 1
