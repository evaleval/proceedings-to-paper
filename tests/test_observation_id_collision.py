"""Deduplication must guarantee unique observation IDs, which grouping alone cannot.

Two `method_metadata` candidates can hash to the same observation ID. The lineage
builder must resolve that collision instead of rejecting the whole paper's artifact.

The divergence is precise. `_semantic_key` hashes the SETTING projection, which includes
`operationalization`, `construct` and `decision_rule`; `stable_id` includes none of them.
Two candidates differing only in those fields therefore land in different semantic groups
and then collide on one ID.
"""

from __future__ import annotations

import hashlib

from proceedings_to_eee.domain.observation import CandidateObservation, EvidenceAnchor
from proceedings_to_eee.domain.status import (
    ClaimType,
    EvidenceKind,
    ExportStatus,
    ReferentialStatus,
    TextSupportStatus,
)
from proceedings_to_eee.validation.candidates import (
    _semantic_key,
    deduplicate_candidates_with_lineage,
)

PAPER_ID = "collision-fixture"
SOURCE_ID = "src_collision_fixture"


def _anchor(quote: str, page: int = 4) -> EvidenceAnchor:
    return EvidenceAnchor(
        source_id=SOURCE_ID,
        page=page,
        kind=EvidenceKind.PROSE,
        quote=quote,
        quote_sha256=hashlib.sha256(quote.encode()).hexdigest(),
    )


def _method_metadata(
    quote: str,
    *,
    operationalization: str | None = None,
    support: TextSupportStatus = TextSupportStatus.SUPPORTED,
    page: int = 4,
    note: str = "note",
) -> CandidateObservation:
    """A candidate with nothing in it that `stable_id` can see.

    Null value, metric and scope, no roles, and an anchor with no label, row or cell.
    """

    return CandidateObservation(
        paper_id=PAPER_ID,
        claim_type=ClaimType.METHOD_METADATA,
        roles=[],
        scope=None,
        metric=None,
        value=None,
        evidence=[_anchor(quote, page)],
        extraction_confidence=0.9,
        text_support=support,
        referential_status=ReferentialStatus.UNVERIFIED,
        export_status=ExportStatus.NOT_ELIGIBLE,
        operationalization=operationalization,
        notes=[note],
    )


def _colliding_pair() -> tuple[CandidateObservation, CandidateObservation]:
    return (
        _method_metadata(
            "with a participation rate of 51%.",
            operationalization="participation rate",
            support=TextSupportStatus.SUPPORTED,
            note="p1",
        ),
        _method_metadata(
            "replacement test was completed by 1,045 panelists,",
            operationalization="panel size",
            support=TextSupportStatus.UNSUPPORTED,
            note="p2",
        ),
    )


def test_deduplication_emits_one_candidate_per_observation_id() -> None:
    first, second = _colliding_pair()
    # Exercise an ID collision that ordinary semantic grouping cannot resolve.
    assert str(first.observation_id) == str(second.observation_id)
    assert _semantic_key(first) != _semantic_key(second)

    result = deduplicate_candidates_with_lineage([first, second])

    ids = [str(candidate.observation_id) for candidate in result.candidates]
    assert len(ids) == len(set(ids)), "deduplication must not emit a repeated observation ID"
    assert len(result.candidates) == 1


def test_collapsing_keeps_every_quotation_and_note() -> None:
    """Collapsing is a merge, not a discard."""

    first, second = _colliding_pair()

    merged = deduplicate_candidates_with_lineage([first, second]).candidates[0]

    assert {anchor.quote for anchor in merged.evidence} == {
        "with a participation rate of 51%.",
        "replacement test was completed by 1,045 panelists,",
    }
    assert {"p1", "p2"} <= set(merged.notes)
    assert any("collapsed 2 proposals" in note for note in merged.notes)


def test_the_supported_member_represents_the_collapsed_candidate() -> None:
    first, second = _colliding_pair()

    merged = deduplicate_candidates_with_lineage([second, first]).candidates[0]

    assert merged.text_support is TextSupportStatus.SUPPORTED


def test_collapsing_does_not_raise_on_disagreeing_provenance() -> None:
    """The reason `_merge_semantic_duplicates` cannot be reused here.

    It raises a disagreeing field to CONFLICT while the source set is empty, and
    `CandidateFieldProvenance` rejects a conflict with no source binding. On the real
    records that is exactly what happened, and the exception cost the whole paper.
    """

    first, second = _colliding_pair()

    merged = deduplicate_candidates_with_lineage([first, second]).candidates[0]

    assert merged.field_provenance == []


def test_candidates_with_distinct_identities_are_left_alone() -> None:
    """The invariant pass must not touch anything that does not actually collide.

    These two differ on both axes: a different page gives them different observation IDs,
    and a different operationalization puts them in different semantic groups.
    """

    first = _method_metadata("shared quotation", page=4, operationalization="rate", note="a")
    other = _method_metadata("shared quotation", page=9, operationalization="size", note="b")
    assert str(first.observation_id) != str(other.observation_id)
    assert _semantic_key(first) != _semantic_key(other)

    result = deduplicate_candidates_with_lineage([first, other])

    ids = [str(candidate.observation_id) for candidate in result.candidates]
    assert len(ids) == len(set(ids))
    assert len(result.candidates) == 2
