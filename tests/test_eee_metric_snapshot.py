"""The vendored EEE registry snapshot: a curated fallback behind the local registry.

The snapshot is package data pinned by hash, so these tests may read it directly. What
they pin is the resolver's behaviour over it: exact normalised names only, the local
registry first, curated entries only, and nothing invented for a metric without a
direction.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from proceedings_to_eee.domain.observation import MetricSpec, ReportedValue
from proceedings_to_eee.resolution.metrics import (
    EEE_REGISTRY_SOURCE,
    load_eee_metric_snapshot,
    metric_canonical_source,
    metric_registry_resolved,
    resolve_metric,
    resolve_metric_value,
)
from proceedings_to_eee.resources import DEFAULT_EEE_METRIC_SNAPSHOT_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFRESH = runpy.run_path(str(PROJECT_ROOT / "scripts/refresh_eee_metric_snapshot.py"))
build_snapshot = REFRESH["build_snapshot"]
render_snapshot = REFRESH["render_snapshot"]


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("EM", "exact-match"),
        ("Exact Match", "exact-match"),
        ("exact_match", "exact-match"),
        ("EXACT   MATCH", "exact-match"),
        ("Win rate (%)", "win-rate"),
        ("Pass@1", "pass-at-1"),
    ],
)
def test_exact_normalised_names_resolve_through_the_snapshot(raw_name: str, expected: str) -> None:
    metric = MetricSpec(raw_name=raw_name)

    assert metric_canonical_source(metric) == EEE_REGISTRY_SOURCE
    resolved = resolve_metric(metric)
    assert resolved.canonical_id == expected
    assert resolved.kind == expected
    assert resolved.lower_is_better is False
    # A snapshot hit is registry-backed, but not a local plain hit.
    assert metric_registry_resolved(metric) is False


def test_the_local_registry_wins_over_a_snapshot_entry_of_the_same_name() -> None:
    # Both registries know "Accuracy"; the local definition decides and says so.
    assert metric_canonical_source(MetricSpec(raw_name="Accuracy")) == "registry"
    assert resolve_metric(MetricSpec(raw_name="F1 Score")).canonical_id == "f1"


@pytest.mark.parametrize("raw_name", ["performance", "Performance", "Overall", "Coverage"])
def test_draft_registry_entries_are_not_vendored_and_do_not_resolve(raw_name: str) -> None:
    assert metric_canonical_source(MetricSpec(raw_name=raw_name)) == "none"
    assert resolve_metric(MetricSpec(raw_name=raw_name)).canonical_id is None


@pytest.mark.parametrize("raw_name", ["score", "Score", "Mean Score", "mean_score", "Rank", "rank"])
def test_generic_placeholder_entities_are_not_vendored_and_do_not_resolve(raw_name: str) -> None:
    """A column headed "Score" is an F1 in one paper and a loss in the next."""

    assert metric_canonical_source(MetricSpec(raw_name=raw_name)) == "none"
    resolved = resolve_metric(MetricSpec(raw_name=raw_name))
    assert resolved.canonical_id is None
    assert resolved.lower_is_better is None


def test_a_vendored_metric_without_a_direction_is_identity_nobody_can_use() -> None:
    snapshot = load_eee_metric_snapshot()
    vendored = {
        row["id"]: row
        for row in json.loads(DEFAULT_EEE_METRIC_SNAPSHOT_PATH.read_text(encoding="utf-8"))[
            "metrics"
        ]
    }
    directionless = [key for key, row in vendored.items() if row["lower_is_better"] is None]

    assert directionless
    assert all(key not in snapshot.definitions for key in directionless)
    assert all(metric_canonical_source(MetricSpec(raw_name=key)) == "none" for key in directionless)


@pytest.mark.parametrize("raw_name", ["EM", "Win Rate"])
def test_rate_bounds_in_the_snapshot_take_the_unit_dependent_scale(raw_name: str) -> None:
    """Whether the registry wrote the rate as [0, 1] or [0, 100], the page's unit decides."""

    decimal = resolve_metric_value(
        MetricSpec(raw_name=raw_name), ReportedValue(raw="0.85", numeric=0.85)
    )
    printed_percent = resolve_metric_value(
        MetricSpec(raw_name=raw_name), ReportedValue(raw="85.3%", numeric=85.3)
    )
    unmarked = resolve_metric_value(
        MetricSpec(raw_name=raw_name), ReportedValue(raw="85.3", numeric=85.3)
    )

    assert decimal[0].unit == "proportion"
    assert (decimal[0].min_score, decimal[0].max_score) == (0.0, 1.0)
    assert printed_percent[0].unit == "percent"
    assert (printed_percent[0].min_score, printed_percent[0].max_score) == (0.0, 100.0)
    # The unmarked-percent inference is reserved for the local registry.
    assert unmarked[0].unit is None
    assert (unmarked[0].min_score, unmarked[0].max_score) == (None, None)


def test_a_snapshot_whose_bytes_drifted_from_the_pin_is_refused(tmp_path: Path) -> None:
    edited = tmp_path / "snapshot.json"
    payload = json.loads(DEFAULT_EEE_METRIC_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    payload["aliases"].append(
        {"canonical_id": "exact-match", "raw_value": "Avg", "status": "confirmed"}
    )
    edited.write_text(json.dumps(payload), encoding="utf-8")
    load_eee_metric_snapshot.cache_clear()
    try:
        with pytest.raises(ValueError, match="hash mismatch"):
            load_eee_metric_snapshot(edited, "0" * 64)
        unpinned = load_eee_metric_snapshot(edited, None)
        assert unpinned.lookup("Avg") is not None
    finally:
        load_eee_metric_snapshot.cache_clear()


def _registry_rows() -> tuple[list[dict], list[dict]]:
    metrics = [
        {
            "id": "exact-match",
            "display_name": "Exact Match",
            "score_type": "continuous",
            "lower_is_better": False,
            "min_score": 0.0,
            "max_score": 1.0,
            "review_status": "reviewed",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "performance",
            "display_name": "Performance",
            "score_type": None,
            "lower_is_better": False,
            "min_score": None,
            "max_score": None,
            "review_status": "draft",
        },
        # Reviewed, but a placeholder: it names the form of the number, not a measurement.
        {
            "id": "score",
            "display_name": "Score",
            "score_type": "continuous",
            "lower_is_better": False,
            "min_score": None,
            "max_score": None,
            "review_status": "reviewed",
        },
    ]
    aliases = [
        {"raw_value": "EM", "canonical_id": "exact-match", "status": "confirmed", "id": "a"},
        {"raw_value": "perf", "canonical_id": "performance", "status": "confirmed", "id": "b"},
        {"raw_value": "Exact", "canonical_id": "exact-match", "status": "uncertain", "id": "c"},
        {"raw_value": "EM", "canonical_id": "exact-match", "status": "confirmed", "id": "d"},
        {"raw_value": "Score", "canonical_id": "score", "status": "confirmed", "id": "e"},
    ]
    return metrics, aliases


def test_the_refresh_script_vendors_only_curated_entries_and_counts_the_rest() -> None:
    metrics, aliases = _registry_rows()

    snapshot = build_snapshot(metrics, aliases, retrieved_date="2026-10-12")

    assert [row["id"] for row in snapshot["metrics"]] == ["exact-match"]
    assert set(snapshot["metrics"][0]) == {
        "id",
        "display_name",
        "score_type",
        "lower_is_better",
        "min_score",
        "max_score",
        "review_status",
    }
    assert snapshot["aliases"] == [
        {"raw_value": "EM", "canonical_id": "exact-match", "status": "confirmed"}
    ]
    assert snapshot["_meta"]["counts"] == {
        "metrics_total": 3,
        "metrics_vendored": 1,
        "aliases_total": 5,
        "aliases_vendored": 1,
    }
    assert snapshot["_meta"]["retrieved_date"] == "2026-10-12"
    assert "resolve" not in json.dumps(snapshot["_meta"]["source"])


def test_the_refresh_script_renders_deterministically_regardless_of_input_order() -> None:
    metrics, aliases = _registry_rows()
    forward = render_snapshot(build_snapshot(metrics, aliases, retrieved_date="2026-10-12"))
    backward = render_snapshot(
        build_snapshot(metrics[::-1], aliases[::-1], retrieved_date="2026-10-12")
    )

    assert forward == backward
    assert forward.endswith("\n")
    assert json.loads(forward)["metrics"]


@pytest.mark.parametrize(
    ("metrics", "aliases"),
    [
        ({"not": "a list"}, []),
        (
            [
                {
                    "id": "x",
                    "display_name": "X",
                    "lower_is_better": "no",
                    "review_status": "reviewed",
                }
            ],
            [],
        ),
        (
            [
                {
                    "id": "x",
                    "display_name": "X",
                    "lower_is_better": False,
                    "review_status": "reviewed",
                    "min_score": "0",
                }
            ],
            [],
        ),
    ],
)
def test_the_refresh_script_refuses_malformed_registry_payloads(
    metrics: object, aliases: object
) -> None:
    with pytest.raises(ValueError):
        build_snapshot(metrics, aliases, retrieved_date="2026-10-12")


def test_the_vendored_snapshot_is_what_the_script_produces_from_its_meta() -> None:
    payload = json.loads(DEFAULT_EEE_METRIC_SNAPSHOT_PATH.read_text(encoding="utf-8"))

    assert payload["_meta"]["schema_version"] == REFRESH["SNAPSHOT_SCHEMA_VERSION"]
    assert payload["_meta"]["policy"] == REFRESH["VENDORING_POLICY"]
    assert render_snapshot(payload) == DEFAULT_EEE_METRIC_SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert all(row["review_status"] == "reviewed" for row in payload["metrics"])
    vendored_ids = {row["id"] for row in payload["metrics"]}
    assert not vendored_ids & REFRESH["GENERIC_PLACEHOLDER_IDS"]
    assert all(
        row["status"] == "confirmed" and row["canonical_id"] in vendored_ids
        for row in payload["aliases"]
    )
