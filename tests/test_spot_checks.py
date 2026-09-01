from __future__ import annotations

from proceedings_to_eee.corpus import ExpectedSpotCheck
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.evaluation.spot_checks import score_spot_checks


def test_exact_spot_check_matches_after_extraction(
    eligible_candidate: CandidateObservation,
) -> None:
    expected = ExpectedSpotCheck(
        system="Atlas Moderation API",
        dataset="Synthetic Speech Set",
        metric="AUC",
        raw_value="74.6%",
        page=7,
        label="Table 2",
    )
    result = score_spot_checks([expected], [eligible_candidate])[0]
    assert result.exact_value is True
    assert result.exact_page is True
    assert result.matched_observation_id == eligible_candidate.observation_id
