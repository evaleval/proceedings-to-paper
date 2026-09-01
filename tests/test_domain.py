from __future__ import annotations

import pytest
from pydantic import ValidationError

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
    ValueComparator,
)


def test_candidate_id_is_stable(eligible_candidate: CandidateObservation) -> None:
    payload = eligible_candidate.model_dump(mode="json")
    payload["observation_id"] = None
    rebuilt = CandidateObservation.model_validate(payload)
    assert rebuilt.observation_id == eligible_candidate.observation_id
    assert rebuilt.observation_id.startswith("obs_")


def test_primary_result_requires_exactly_one_evaluated_system() -> None:
    with pytest.raises(ValidationError, match="exactly one evaluated_system"):
        CandidateObservation(
            paper_id="paper",
            claim_type=ClaimType.PRIMARY_RESULT,
            roles=[
                RoleAssignment(
                    role=ActorRole.EVALUATION_INSTRUMENT,
                    raw_name="Atlas Moderation API",
                    confidence=1,
                )
            ],
            scope=ObservationScope(dataset_raw="Dataset"),
            metric=MetricSpec(raw_name="Accuracy"),
            value=ReportedValue(raw="0.8", numeric=0.8),
            evidence=[
                EvidenceAnchor(
                    source_id="source",
                    page=1,
                    kind=EvidenceKind.PROSE,
                    quote="Accuracy was 0.8.",
                )
            ],
        )


def test_raw_percent_is_not_silently_rescaled(eligible_candidate: CandidateObservation) -> None:
    assert eligible_candidate.value.numeric == 74.6
    assert eligible_candidate.value.unit == "percent"
    assert eligible_candidate.metric.max_score == 100


@pytest.mark.parametrize("alias", ["%", "pct", "percent", "percentage"])
def test_explicit_percent_unit_aliases_are_canonical_without_rescaling(alias: str) -> None:
    metric = MetricSpec(raw_name="F1", unit=alias)
    value = ReportedValue(raw="73.4%", numeric=73.4, unit=alias)

    assert metric.unit == "percent"
    assert value.unit == "percent"
    assert value.numeric == 73.4


def test_percentage_points_remain_distinct_from_percent() -> None:
    value = ReportedValue(raw="3 percentage points", numeric=3, unit="percentage points")

    assert value.unit == "percentage_points"
    assert value.numeric == 3


def test_inequality_semantics_are_preserved() -> None:
    value = ReportedValue(
        raw="<0.001***",
        numeric=0.001,
        comparator=ValueComparator.LESS_THAN,
    )
    assert value.numeric == 0.001
    assert value.comparator is ValueComparator.LESS_THAN
