"""Conservative matching of extracted candidates to sealed-from-prompt spot checks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from proceedings_to_eee.corpus import ExpectedSpotCheck
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ActorRole


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", value.casefold()).strip()


@dataclass(frozen=True)
class SpotCheckResult:
    expected: ExpectedSpotCheck
    matched_observation_id: str | None
    exact_value: bool
    exact_page: bool | None
    notes: tuple[str, ...]


def score_spot_checks(
    expected: Iterable[ExpectedSpotCheck], candidates: Iterable[CandidateObservation]
) -> list[SpotCheckResult]:
    """Score reference values after extraction; references never enter the prompt path."""

    candidates = list(candidates)
    results: list[SpotCheckResult] = []
    for target in expected:
        compatible: list[CandidateObservation] = []
        for candidate in candidates:
            system = next(
                (
                    role.raw_name
                    for role in candidate.roles
                    if role.role == ActorRole.EVALUATED_SYSTEM
                ),
                "",
            )
            if not candidate.scope or not candidate.metric or not candidate.value:
                continue
            target_system = _norm(target.system)
            actual_system = _norm(system)
            if target_system not in actual_system and actual_system not in target_system:
                continue
            if _norm(target.dataset) not in _norm(candidate.scope.dataset_raw):
                continue
            if _norm(target.metric) not in {
                _norm(candidate.metric.raw_name),
                _norm(candidate.metric.canonical_id or ""),
            }:
                continue
            compatible.append(candidate)
        exact = [item for item in compatible if _norm(item.value.raw) == _norm(target.raw_value)]
        winner = exact[0] if exact else (compatible[0] if compatible else None)
        page_match = None
        if target.page is not None:
            page_match = bool(winner and any(a.page == target.page for a in winner.evidence))
        notes: list[str] = []
        if len(compatible) > 1:
            notes.append(f"{len(compatible)} compatible candidates")
        if winner is None:
            notes.append("no compatible candidate")
        elif not exact:
            notes.append(f"value mismatch: extracted {winner.value.raw!r}")
        results.append(
            SpotCheckResult(
                expected=target,
                matched_observation_id=winner.observation_id if winner else None,
                exact_value=bool(exact),
                exact_page=page_match,
                notes=tuple(notes),
            )
        )
    return results
