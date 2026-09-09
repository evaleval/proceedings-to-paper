"""Census driver orchestration: skip key, resume, budget guard, drain, and ledger.

The per-paper pipeline is stubbed here on purpose. What these tests cover is the part
that is new and easy to get wrong: deciding what to run, what to charge, when to stop,
and what to record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from proceedings_to_eee import census_run
from proceedings_to_eee.census_run import (
    DONE,
    FAILED,
    NOT_STARTED,
    RETRY_CAPPED,
    SKIPPED,
    SOURCE_UNAVAILABLE,
    CensusLimits,
    extraction_contract_key,
    run_census,
)
from proceedings_to_eee.corpus import CorpusSpec, PaperSpec
from proceedings_to_eee.pipeline import PipelineSettings
from proceedings_to_eee.providers.budget import ProviderBudgetContractError


def _settings(tmp_path: Path, **overrides: Any) -> PipelineSettings:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    base = {
        "project_root": tmp_path,
        "schema_path": schema,
        "schema_sha256": "0" * 64,
        "output_root": tmp_path / "runs" / "census",
        "model": "test/model-a",
    }
    base.update(overrides)
    return PipelineSettings(**base)


def _corpus(count: int) -> CorpusSpec:
    return CorpusSpec(
        corpus_id="census",
        evaluation_split="development",
        description="d",
        papers=[
            PaperSpec(
                paper_id=f"paper-{index}",
                title=f"Paper {index}",
                year=2024,
                venue="acl",
                pdf_url=f"https://example.org/{index}.pdf",
                perspective_role="Primary paper.",
            )
            for index in range(count)
        ],
    )


def _freeze(settings: PipelineSettings, slug: str, *, body: str = "manifest") -> Path:
    directory = settings.output_root / slug
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "source-manifest.json"
    path.write_text(json.dumps({"paper_id": slug, "body": body}), encoding="utf-8")
    return path


def _complete(settings: PipelineSettings, slug: str, key: str, *, status: str = "success") -> None:
    directory = settings.output_root / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(
        json.dumps(
            {
                "paper_id": slug,
                "status": status,
                "counts": {"candidates": 7},
                "census_contract_key": key,
            }
        ),
        encoding="utf-8",
    )


class _StubBudget:
    """Stands in for BudgetedProviderClient, reporting a fixed cost."""

    def __init__(self, cost: float = 0.10, **_kwargs: Any) -> None:
        self._cost = cost

    @property
    def summary(self) -> dict[str, Any]:
        return {"provider_reported_cost_usd": self._cost}


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch):
    """Replace the pipeline with a recorder so orchestration can be tested alone."""

    calls: list[str] = []
    cost = {"value": 0.10}
    failures: set[str] = set()

    def fake_run_paper(*, spec: PaperSpec, settings: PipelineSettings, client: Any) -> dict:
        calls.append(spec.paper_id)
        if spec.paper_id in failures:
            raise RuntimeError("stubbed pipeline failure")
        directory = settings.output_root / spec.paper_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.json").write_text(
            json.dumps(
                {"paper_id": spec.paper_id, "status": "success", "counts": {"candidates": 3}}
            ),
            encoding="utf-8",
        )
        return {"status": "success", "counts": {"candidates": 3}, "wall_clock_seconds": 1.0}

    monkeypatch.setattr(census_run, "run_paper", fake_run_paper)
    # The driver builds a real provider client for each paper; tests must not need a key.
    monkeypatch.setattr(census_run, "_default_client_factory", lambda: object())
    monkeypatch.setattr(
        census_run, "BudgetedProviderClient", lambda **kw: _StubBudget(cost["value"])
    )
    monkeypatch.setattr(census_run, "_budget_contract", lambda *a, **k: {"stub": True})
    return {"calls": calls, "cost": cost, "failures": failures}


def _ledger(settings: PipelineSettings) -> list[dict[str, Any]]:
    path = settings.output_root / "census-ledger.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_contract_key_ignores_code_state_but_tracks_the_real_inputs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manifest = _freeze(settings, "paper-0")

    base = extraction_contract_key(settings, manifest)
    assert base == extraction_contract_key(settings, manifest)

    other_model = extraction_contract_key(_settings(tmp_path, model="test/model-b"), manifest)
    assert other_model != base

    rows_on = extraction_contract_key(_settings(tmp_path, row_enumeration_enabled=True), manifest)
    assert rows_on != base

    manifest.write_text(json.dumps({"paper_id": "paper-0", "body": "changed"}), encoding="utf-8")
    assert extraction_contract_key(settings, manifest) != base


def test_a_paper_without_a_frozen_manifest_is_never_downloaded(tmp_path: Path, stub) -> None:
    settings = _settings(tmp_path)
    corpus = _corpus(3)
    _freeze(settings, "paper-0")
    # paper-1 and paper-2 have no manifest.

    result = run_census(
        corpus=corpus,
        settings=settings,
        order=[p.paper_id for p in corpus.papers],
        limits=CensusLimits(workers=1, max_cost_usd=10.0),
    )

    assert result["counts"][SOURCE_UNAVAILABLE] == 2
    assert result["counts"][DONE] == 1
    assert stub["calls"] == ["paper-0"]


def test_dry_run_walks_the_order_without_spending_or_creating_a_ledger(
    tmp_path: Path, stub
) -> None:
    settings = _settings(tmp_path)
    corpus = _corpus(4)
    for paper in corpus.papers:
        _freeze(settings, paper.paper_id)

    result = run_census(
        corpus=corpus,
        settings=settings,
        order=[p.paper_id for p in corpus.papers],
        limits=CensusLimits(workers=4, max_cost_usd=10.0),
        dry_run=True,
    )

    assert result["counts"][NOT_STARTED] == 4
    assert result["reported_cost_usd"] == 0.0
    assert stub["calls"] == []
    assert {row["reason"] for row in _ledger(settings)} == {"dry_run"}
    assert not (settings.output_root / "paper-0" / "private").exists()


def test_a_completed_paper_under_the_same_contract_is_skipped(tmp_path: Path, stub) -> None:
    settings = _settings(tmp_path)
    corpus = _corpus(2)
    for paper in corpus.papers:
        _freeze(settings, paper.paper_id)
    # The key must come from this paper's own manifest; each manifest hashes differently.
    own_manifest = settings.output_root / "paper-0" / "source-manifest.json"
    _complete(settings, "paper-0", extraction_contract_key(settings, own_manifest))

    result = run_census(
        corpus=corpus,
        settings=settings,
        order=["paper-0", "paper-1"],
        limits=CensusLimits(workers=1, max_cost_usd=10.0),
    )

    assert result["counts"][SKIPPED] == 1
    assert stub["calls"] == ["paper-1"]


def test_a_completed_paper_under_a_different_contract_reruns_and_supersedes(
    tmp_path: Path, stub
) -> None:
    settings = _settings(tmp_path)
    corpus = _corpus(1)
    _freeze(settings, "paper-0")
    _complete(settings, "paper-0", "a-stale-key-from-another-model")

    run_census(
        corpus=corpus,
        settings=settings,
        order=["paper-0"],
        limits=CensusLimits(workers=1, max_cost_usd=10.0),
    )

    assert stub["calls"] == ["paper-0"]
    assert (settings.output_root / "paper-0" / "run.superseded-1.json").is_file()
    fresh = json.loads((settings.output_root / "paper-0" / "run.json").read_text())
    assert fresh["census_contract_key"]


def test_the_budget_guard_stops_dispatch_before_the_ceiling_is_exceeded(
    tmp_path: Path, stub
) -> None:
    settings = _settings(tmp_path)
    corpus = _corpus(10)
    for paper in corpus.papers:
        _freeze(settings, paper.paper_id)
    stub["cost"]["value"] = 0.5

    result = run_census(
        corpus=corpus,
        settings=settings,
        order=[p.paper_id for p in corpus.papers],
        limits=CensusLimits(workers=1, max_cost_usd=2.0, per_paper_max_cost_usd=0.5),
    )

    # The guard authorises work up to and including the ceiling, so four papers at 0.5
    # each exactly reach 2.0 and the fifth is refused.
    assert result["stopped_for_budget"] is True
    assert result["counts"][DONE] == 4
    assert result["reported_cost_usd"] == pytest.approx(2.0)
    assert result["reported_cost_usd"] <= 2.0
    assert result["counts"][NOT_STARTED] == 6
    assert {row.get("reason") for row in _ledger(settings) if row["state"] == NOT_STARTED} == {
        "budget_ceiling"
    }


def test_the_guard_adapts_to_observed_cost_instead_of_reserving_the_ceiling(
    tmp_path: Path, stub
) -> None:
    """Reserve using observed costs once enough papers have reported.

    Reserving the per-paper ceiling indefinitely can stop affordable work while
    much of the actual budget remains unspent.
    """

    settings = _settings(tmp_path)
    corpus = _corpus(30)
    for paper in corpus.papers:
        _freeze(settings, paper.paper_id)
    stub["cost"]["value"] = 0.2

    result = run_census(
        corpus=corpus,
        settings=settings,
        order=[p.paper_id for p in corpus.papers],
        limits=CensusLimits(workers=1, max_cost_usd=4.0, per_paper_max_cost_usd=1.0),
    )

    # Charging the ceiling would have stopped after 3 papers (3 x 1.0 + 1.0 > 4.0).
    # Adapting to the observed 0.2 lets the run use the budget it actually has.
    assert result["counts"][DONE] >= 15
    assert result["reported_cost_usd"] <= 4.0


def test_the_estimator_uses_the_ceiling_until_enough_papers_have_reported() -> None:
    guard = census_run._BudgetGuard(
        CensusLimits(workers=4, max_cost_usd=10.0, per_paper_max_cost_usd=1.0)
    )

    # With nothing observed, an in-flight paper must be charged at the ceiling.
    assert guard._estimate_locked() == 1.0

    for _ in range(4):
        guard.settle(0.2)
    assert guard._estimate_locked() == 1.0, "four reports is still too few to trust"

    guard.settle(0.2)
    # Five reports of 0.2: p95 is 0.2 and twice the mean is 0.4, so the larger wins.
    assert guard._estimate_locked() == pytest.approx(0.4)
    assert guard._estimate_locked() < 1.0


def test_the_estimate_never_exceeds_the_per_paper_ceiling() -> None:
    guard = census_run._BudgetGuard(
        CensusLimits(workers=4, max_cost_usd=10.0, per_paper_max_cost_usd=0.5)
    )
    for _ in range(6):
        guard.settle(0.45)

    # Twice the mean would be 0.9, but the ledger caps any single paper at 0.5.
    assert guard._estimate_locked() == 0.5


def test_worst_case_reports_the_hard_bound_not_the_estimate(tmp_path: Path, stub) -> None:
    guard = census_run._BudgetGuard(
        CensusLimits(workers=4, max_cost_usd=10.0, per_paper_max_cost_usd=1.0)
    )
    guard.try_reserve()
    guard.try_reserve()

    assert guard.worst_case == pytest.approx(2.0)


def test_a_failing_paper_still_reports_the_money_it_spent(tmp_path: Path, stub) -> None:
    """Charge actual spend even when a paper fails before producing a run summary."""

    settings = _settings(tmp_path)
    corpus = _corpus(2)
    for paper in corpus.papers:
        _freeze(settings, paper.paper_id)
    stub["cost"]["value"] = 0.3
    stub["failures"].add("paper-0")

    result = run_census(
        corpus=corpus,
        settings=settings,
        order=[p.paper_id for p in corpus.papers],
        limits=CensusLimits(workers=1, max_cost_usd=10.0),
    )

    rows = {row["paper_id"]: row for row in _ledger(settings)}
    assert rows["paper-0"]["state"] == FAILED
    assert rows["paper-0"]["reported_cost_usd"] == pytest.approx(0.3)
    # Both papers spent 0.3, so the guard must have seen 0.6, not 0.3.
    assert result["reported_cost_usd"] == pytest.approx(0.6)


def test_one_failing_paper_does_not_end_the_census(tmp_path: Path, stub) -> None:
    settings = _settings(tmp_path)
    corpus = _corpus(4)
    for paper in corpus.papers:
        _freeze(settings, paper.paper_id)
    stub["failures"].add("paper-1")

    result = run_census(
        corpus=corpus,
        settings=settings,
        order=[p.paper_id for p in corpus.papers],
        limits=CensusLimits(workers=2, max_cost_usd=10.0),
    )

    assert result["counts"][DONE] == 3
    assert result["counts"][FAILED] == 1
    failed = [row for row in _ledger(settings) if row["state"] == FAILED]
    assert failed[0]["error_type"] == "RuntimeError"


def test_every_ordered_paper_reaches_exactly_one_terminal_state(tmp_path: Path, stub) -> None:
    settings = _settings(tmp_path)
    corpus = _corpus(6)
    for paper in corpus.papers[:5]:
        _freeze(settings, paper.paper_id)
    stub["failures"].add("paper-2")

    result = run_census(
        corpus=corpus,
        settings=settings,
        order=[p.paper_id for p in corpus.papers],
        limits=CensusLimits(workers=3, max_cost_usd=10.0),
    )

    rows = _ledger(settings)
    assert len(rows) == 6
    assert len({row["paper_id"] for row in rows}) == 6
    assert sum(result["counts"].values()) == 6


def test_progress_file_is_written_and_final(tmp_path: Path, stub) -> None:
    settings = _settings(tmp_path)
    corpus = _corpus(2)
    for paper in corpus.papers:
        _freeze(settings, paper.paper_id)

    run_census(
        corpus=corpus,
        settings=settings,
        order=[p.paper_id for p in corpus.papers],
        limits=CensusLimits(workers=1, max_cost_usd=10.0),
    )

    progress = json.loads((settings.output_root / "census-progress.json").read_text())
    assert progress["final"] is True
    assert progress["ordered"] == 2
    assert progress["finished"] == 2
    assert progress["remaining"] == 0


def test_limits_reject_a_per_paper_ceiling_below_the_reservation() -> None:
    with pytest.raises(ValueError, match="must exceed the per-call reservation"):
        CensusLimits(per_paper_max_cost_usd=0.05, reservation_usd=0.05)
    with pytest.raises(ValueError, match="must not be below the per-paper ceiling"):
        CensusLimits(max_cost_usd=0.5, per_paper_max_cost_usd=1.5)


def test_a_paper_that_keeps_failing_is_capped_instead_of_retried_forever(
    tmp_path: Path, stub
) -> None:
    """Enforce the retry cap even when failures leave no run.json.

    Resume must count prior attempts without charging their spend again.
    """

    settings = _settings(tmp_path)
    corpus = _corpus(1)
    _freeze(settings, "paper-0")
    stub["cost"]["value"] = 0.3
    stub["failures"].add("paper-0")
    order = ["paper-0"]
    limits = CensusLimits(workers=1, max_cost_usd=10.0, max_technical_retries=2)

    for _ in range(2):
        run_census(corpus=corpus, settings=settings, order=order, limits=limits)
    third = run_census(corpus=corpus, settings=settings, order=order, limits=limits)

    assert third["counts"][RETRY_CAPPED] == 1
    assert third["counts"][FAILED] == 0
    # The capped leg spent nothing at all.
    assert third["reported_cost_usd"] == pytest.approx(0.0)
    capped = [row for row in _ledger(settings) if row["state"] == RETRY_CAPPED]
    assert capped[-1]["attempts"] == 2
    assert capped[-1]["reason"] == "max_technical_retries"


def test_a_paper_that_succeeds_starts_its_retry_budget_again(tmp_path: Path, stub) -> None:
    """The cap counts consecutive unproductive attempts, not attempts ever made."""

    settings = _settings(tmp_path)
    corpus = _corpus(1)
    _freeze(settings, "paper-0")
    order = ["paper-0"]
    limits = CensusLimits(workers=1, max_cost_usd=10.0, max_technical_retries=2)

    stub["failures"].add("paper-0")
    run_census(corpus=corpus, settings=settings, order=order, limits=limits)
    stub["failures"].discard("paper-0")
    result = run_census(corpus=corpus, settings=settings, order=order, limits=limits)

    assert result["counts"][DONE] == 1
    assert census_run.technical_attempts(settings.output_root / "paper-0") == 0


def test_superseded_runs_from_an_older_driver_still_count_toward_the_cap(
    tmp_path: Path, stub
) -> None:
    settings = _settings(tmp_path)
    corpus = _corpus(1)
    directory = _freeze(settings, "paper-0").parent
    for index in (1, 2):
        (directory / f"run.superseded-{index}.json").write_text("{}", encoding="utf-8")

    result = run_census(
        corpus=corpus,
        settings=settings,
        order=["paper-0"],
        limits=CensusLimits(workers=1, max_cost_usd=10.0, max_technical_retries=2),
    )

    assert result["counts"][RETRY_CAPPED] == 1


def test_a_paper_whose_run_contract_moved_gets_a_fresh_ledger_beside_the_old_one(
    tmp_path: Path, stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ledger binds its contract, so a paper that never finished cannot be resumed
    after the contract legitimately changes. Retire the ledger, never overwrite it."""

    settings = _settings(tmp_path)
    corpus = _corpus(1)
    directory = _freeze(settings, "paper-0").parent
    ledger = directory / "private" / "provider-budget-ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    stale = json.dumps(
        {
            "schema_version": "provider-budget-ledger-event/0.3",
            "event_type": "completion",
            "sequence": 1,
            "contract_sha256": "0" * 64,
            "outcome": "success",
            "provider_call": {"cost_usd": 0.4},
        }
    )
    ledger.write_text(stale + "\n", encoding="utf-8")
    (directory / "private" / "provider-budget-ledger.jsonl.head.json").write_text(
        "{}", encoding="utf-8"
    )

    # The stub client stands in for the budgeted one, so make it refuse the stale ledger
    # exactly as the real client does when a contract identity no longer matches.
    attempts = {"count": 0}
    real_stub = census_run.BudgetedProviderClient

    def refuse_once(**kwargs: Any):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ProviderBudgetContractError("contract identity changed")
        return real_stub(**kwargs)

    monkeypatch.setattr(census_run, "BudgetedProviderClient", refuse_once)

    result = run_census(
        corpus=corpus,
        settings=settings,
        order=["paper-0"],
        limits=CensusLimits(workers=1, max_cost_usd=10.0),
    )

    assert attempts["count"] == 2
    assert result["counts"][DONE] == 1
    retired = directory / "private" / "provider-budget-ledger.superseded-1.jsonl"
    assert retired.is_file()
    # Retired, not overwritten: the old accounting survives byte for byte. The stub
    # client writes no replacement ledger; the real one does, at the original path.
    assert retired.read_text(encoding="utf-8").strip() == stale
    assert not ledger.exists()
    row = next(row for row in _ledger(settings) if row["paper_id"] == "paper-0")
    assert row["superseded_ledger"] == "provider-budget-ledger.superseded-1.jsonl"


def test_a_retired_ledger_still_counts_toward_reported_cost(tmp_path: Path) -> None:
    from proceedings_to_eee.census_atlas import paper_ledger_cost

    directory = tmp_path / "paper-0"
    (directory / "private").mkdir(parents=True)
    for name, cost in (
        ("provider-budget-ledger.jsonl", 0.2),
        ("provider-budget-ledger.superseded-1.jsonl", 0.4),
    ):
        (directory / "private" / name).write_text(
            json.dumps({"event_type": "completion", "provider_call": {"cost_usd": cost}}) + "\n",
            encoding="utf-8",
        )

    cost, calls = paper_ledger_cost(directory)

    assert cost == pytest.approx(0.6)
    assert calls == 2
