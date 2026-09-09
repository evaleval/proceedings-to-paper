"""Atlas builder: flattening, missingness, census joins, and claim framing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from proceedings_to_eee.census_atlas import (
    _OBSERVATION_COLUMNS,
    build_atlas,
    census_instrument_match,
    dataset_key,
    language_code,
    paper_ledger_cost,
)


def _paper(
    root: Path,
    slug: str,
    *,
    candidates: list[dict[str, Any]],
    rows: dict[str, Any] | None = None,
    ledger_costs: list[float] | None = None,
    status: str = "success",
) -> None:
    directory = root / slug
    (directory / "private").mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(
        json.dumps(
            {
                "paper_id": slug,
                "status": status,
                "selected_pages": [{"page": 3}],
                "counts": {"eee_records": 0},
                "wall_clock_seconds": 12.0,
                "code": {"source_tree_sha256": "abc"},
                "extractor": {"execution": {"blocks_total": 2, "blocks_succeeded": 2}},
            }
        ),
        encoding="utf-8",
    )
    (directory / "observations.jsonl").write_text(
        "\n".join(json.dumps(item) for item in candidates) + "\n", encoding="utf-8"
    )
    if rows is not None:
        (directory / "private" / "row-terminal-states.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )
    if ledger_costs is not None:
        lines = [
            json.dumps({"event_type": "completion", "provider_call": {"cost_usd": cost}})
            for cost in ledger_costs
        ]
        (directory / "private" / "provider-budget-ledger.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def _candidate(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "observation_id": "obs_1",
        "paper_id": "p",
        "claim_type": "primary_result",
        "text_support": "supported",
        "referential_status": "resolved",
        "export_status": "needs_review",
        "extraction_method": "openrouter:model",
        "extraction_confidence": 0.9,
        "roles": [{"role": "evaluated_system", "raw_name": "Perspective API"}],
        "scope": {"dataset_raw": "HateCheck (subset)", "language": "English"},
        "metric": {"raw_name": "Macro F1", "canonical_id": "macro_f1"},
        "value": {"raw": "0.81", "numeric": 0.81, "unit": "proportion"},
        "evidence": [{"page": 3, "kind": "table", "quote": "q", "quote_sha256": "h"}],
        "operationalization": "toxicity threshold 0.7",
        "notes": ["a note"],
    }
    base.update(overrides)
    return base


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_language_codes_map_names_and_abstain_otherwise() -> None:
    assert language_code("English") == "en"
    assert language_code("en") == "en"
    assert language_code("Polish") == "pl"
    assert language_code("Klingon") is None
    assert language_code(None) is None


def test_dataset_key_groups_one_benchmark_without_guessing_aliases() -> None:
    assert dataset_key("KLEJ benchmark (CBD)") == dataset_key("KLEJ benchmark") == "klej"
    assert dataset_key("HateCheck") == "hatecheck"
    # Two genuinely different names must not collapse.
    assert dataset_key("OLID") != dataset_key("HatEval")
    assert dataset_key(None) is None


def test_instrument_match_prefers_exact_and_reports_its_kind() -> None:
    exact = census_instrument_match(
        "Perspective API; Detoxify", {"system": "Perspective API", "dataset": None}
    )
    assert exact == ("Perspective API", "system", "exact")

    partial = census_instrument_match(
        "Perspective API", {"system": "Google Perspective API scorer", "dataset": None}
    )
    assert partial[2] == "substring"

    assert census_instrument_match("Detoxify", {"system": "BERT"})[2] == "none"
    assert census_instrument_match(None, {"system": "BERT"})[2] == "none"


def test_ledger_cost_counts_completions_only(tmp_path: Path) -> None:
    directory = tmp_path / "paper"
    (directory / "private").mkdir(parents=True)
    (directory / "private" / "provider-budget-ledger.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event_type": "reservation"}),
                json.dumps({"event_type": "completion", "provider_call": {"cost_usd": 0.2}}),
                json.dumps({"event_type": "completion", "provider_call": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cost, calls = paper_ledger_cost(directory)

    assert cost == 0.2
    assert calls == 2


def test_every_observation_column_is_written(tmp_path: Path) -> None:
    _paper(tmp_path, "paper-a", candidates=[_candidate()])

    build_atlas(run_root=tmp_path, output=tmp_path / "atlas")

    rows = _read(tmp_path / "atlas" / "observations.csv")
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(_OBSERVATION_COLUMNS)
    assert rows[0]["operationalization"] == "toxicity threshold 0.7"
    assert rows[0]["language_iso639"] == "en"
    assert rows[0]["dataset_key"] == "hatecheck"
    assert rows[0]["extraction_stage"] == "block"


def test_row_stage_candidates_are_labelled(tmp_path: Path) -> None:
    _paper(
        tmp_path,
        "paper-a",
        candidates=[
            _candidate(observation_id="obs_1"),
            _candidate(
                observation_id="obs_2",
                extraction_method="openrouter:model:row-enumeration",
            ),
        ],
    )

    summary = build_atlas(run_root=tmp_path, output=tmp_path / "atlas")

    assert summary["by_extraction_stage"] == {"block": 1, "row": 1}


def test_planned_rows_are_all_represented_so_missingness_is_visible(tmp_path: Path) -> None:
    _paper(
        tmp_path,
        "paper-a",
        candidates=[_candidate()],
        rows={
            "counts": {"planned": 3, "unresolved": 1},
            "invalid_row_reasons": {"r3": "domain_candidate_validation"},
            "terminal_by_row": {
                "r1": {
                    "state": "result",
                    "disposition": {"disposition": "result", "candidates": [{}]},
                },
                "r2": {
                    "state": "not_result",
                    "disposition": {"disposition": "not_result", "candidates": []},
                },
                "r3": {"state": "unresolved", "disposition": {}},
            },
        },
    )

    summary = build_atlas(run_root=tmp_path, output=tmp_path / "atlas")

    rows = _read(tmp_path / "atlas" / "rows.csv")
    assert len(rows) == 3
    assert summary["row_states"] == {"result": 1, "not_result": 1, "unresolved": 1}
    unresolved = next(row for row in rows if row["row_id"] == "r3")
    assert unresolved["invalid_reason"] == "domain_candidate_validation"


def test_cost_comes_from_the_paper_ledgers_not_the_census_ledger(tmp_path: Path) -> None:
    """The census ledger undercounted failed papers; the per-paper ledgers did not."""

    _paper(tmp_path, "paper-a", candidates=[_candidate()], ledger_costs=[0.3, 0.2])
    (tmp_path / "census-ledger.jsonl").write_text(
        json.dumps({"paper_id": "paper-a", "state": "done", "reported_cost_usd": 0.0}) + "\n",
        encoding="utf-8",
    )

    summary = build_atlas(run_root=tmp_path, output=tmp_path / "atlas")

    assert summary["provider_reported_cost_usd"] == 0.5
    assert summary["structured_calls"] == 2


def test_census_metadata_is_joined_under_census_names_and_flagged_unvalidated(
    tmp_path: Path,
) -> None:
    _paper(tmp_path, "paper-a", candidates=[_candidate()])
    metadata = {
        "by_slug": {
            "paper-a": {
                "census_paper_id": "2024.acl-main.1",
                "census_year": 2024,
                "census_venue": "acl",
                "census_measurement_role": "applies_existing_instrument",
                "census_instrument_mentions": "Perspective API",
            }
        }
    }

    summary = build_atlas(
        run_root=tmp_path,
        output=tmp_path / "atlas",
        census_metadata=metadata,
        census_provenance={"producer": "unknown", "validation_status": "none"},
    )

    rows = _read(tmp_path / "atlas" / "observations.csv")
    assert rows[0]["census_paper_id"] == "2024.acl-main.1"
    assert rows[0]["census_match_kind"] == "exact"
    assert summary["census_provenance"]["producer"] == "unknown"
    assert summary["by_measurement_role"]["applies_existing_instrument"]["papers"] == 1


def test_the_index_carries_the_claim_boundary(tmp_path: Path) -> None:
    _paper(tmp_path, "paper-a", candidates=[_candidate()])

    summary = build_atlas(run_root=tmp_path, output=tmp_path / "atlas")
    html = (tmp_path / "atlas" / "index.html").read_text(encoding="utf-8")

    assert "unreviewed candidate layer" in html
    assert "Canonical EEE is empty by design" in html
    assert len(summary["claim_boundary"]) >= 5


def test_a_candidate_without_a_metric_does_not_break_the_build(tmp_path: Path) -> None:
    """Candidates may lack a metric; MetricSpec rejects an empty name."""

    _paper(tmp_path, "paper-a", candidates=[_candidate(metric=None, value=None)])

    summary = build_atlas(run_root=tmp_path, output=tmp_path / "atlas")

    rows = _read(tmp_path / "atlas" / "observations.csv")
    assert rows[0]["metric_canonical_source"] == "none"
    assert summary["observations"] == 1


def test_every_census_paper_is_accounted_for_not_only_the_ones_with_run_json(
    tmp_path: Path,
) -> None:
    """Include every census paper in papers.csv, even when no run.json was written."""

    _paper(tmp_path, "ran-a", candidates=[_candidate()], ledger_costs=[0.2])
    for slug in ("frozen-not-started", "failed-a", "stopped-a", "no-source"):
        (tmp_path / slug).mkdir(parents=True, exist_ok=True)
    for slug in ("frozen-not-started", "failed-a", "stopped-a"):
        (tmp_path / slug / "source-manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "failed-a" / "private").mkdir(parents=True, exist_ok=True)
    (tmp_path / "failed-a" / "private" / "provider-budget-ledger.jsonl").write_text(
        json.dumps({"event_type": "completion", "provider_call": {"cost_usd": 0.4}}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "census-ledger.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"paper_id": "ran-a", "state": "done"},
                {"paper_id": "failed-a", "state": "failed", "error": "ValidationError"},
                {"paper_id": "stopped-a", "state": "budget_stopped"},
                {"paper_id": "no-source", "state": "source_unavailable"},
                {"paper_id": "never-run", "state": "not_started"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "by_slug": {
            slug: {
                "census_paper_id": slug,
                "census_measurement_role": "applies_existing_instrument",
            }
            for slug in (
                "ran-a",
                "frozen-not-started",
                "failed-a",
                "stopped-a",
                "no-source",
                "never-run",
                "held-back",
            )
        }
    }

    summary = build_atlas(run_root=tmp_path, output=tmp_path / "atlas", census_metadata=metadata)

    papers = {row["paper_id"]: row for row in _read(tmp_path / "atlas" / "papers.csv")}
    assert len(papers) == 7
    assert papers["ran-a"]["status"] == "success"
    assert papers["failed-a"]["status"] == "failed"
    assert papers["stopped-a"]["status"] == "budget_stopped"
    assert papers["no-source"]["status"] == "unavailable"
    assert papers["never-run"]["status"] == "not_started"
    assert papers["frozen-not-started"]["status"] == "not_started"
    # Never opened the sealed reserve: a census row with no directory and no ledger entry.
    assert papers["held-back"]["status"] == "reserve"
    assert papers["held-back"]["status_basis"] == "census row absent from the run"
    assert summary["papers_total"] == 7
    assert summary["papers_ran"] == 1
    assert summary["papers_with_results"] == 1
    assert summary["papers_by_status"]["reserve"] == 1
    # Only papers that ran count toward the role breakdown.
    assert summary["by_measurement_role"]["applies_existing_instrument"]["papers"] == 1


def test_cost_counts_ledgers_of_papers_that_wrote_no_run_json(tmp_path: Path) -> None:
    """The $3.90 the old guard never read belonged to papers without a run.json."""

    _paper(tmp_path, "ran-a", candidates=[_candidate()], ledger_costs=[0.2])
    (tmp_path / "failed-a" / "private").mkdir(parents=True, exist_ok=True)
    (tmp_path / "failed-a" / "source-manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "failed-a" / "private" / "provider-budget-ledger.jsonl").write_text(
        json.dumps({"event_type": "completion", "provider_call": {"cost_usd": 0.4}}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "census-ledger.jsonl").write_text(
        json.dumps({"paper_id": "failed-a", "state": "failed"}) + "\n", encoding="utf-8"
    )

    summary = build_atlas(run_root=tmp_path, output=tmp_path / "atlas")

    assert summary["provider_reported_cost_usd"] == 0.6
    assert "under the run root" in summary["provider_reported_cost_basis"]
    assert summary["structured_calls"] == 2
