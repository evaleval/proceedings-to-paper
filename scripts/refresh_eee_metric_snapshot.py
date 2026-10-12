#!/usr/bin/env python3
"""Regenerate the vendored EEE metric registry snapshot from its read-only endpoints.

The pipeline resolves metric names offline against a snapshot shipped as package data.
This script is the only place that talks to the registry, and it only reads: it fetches
the metric list and the metric aliases with plain GET requests. It never calls the
registry's resolve endpoint, which creates draft entities as a side effect.

Only entries the registry itself has curated are vendored: metrics whose review status is
``reviewed`` and confirmed aliases that point at one of them. Draft metrics and uncertain
aliases are auto-created guesses; vendoring them would let a generic column header such as
"performance" acquire a canonical identity, and a dataset-specific draft its direction.
Reviewed placeholder entities whose name only says that a number is a score or a rank are
left out as well: a column headed "Score" is an F1 in one paper and a loss in the next, so
resolving it would guess an identity and assert a direction the page never stated.

Usage:
  python scripts/refresh_eee_metric_snapshot.py --output <snapshot.json>
      [--metrics-file <metrics.json> --aliases-file <aliases.json>] [--retrieved-date DATE]

With ``--metrics-file`` and ``--aliases-file`` the script reads previously downloaded
endpoint payloads instead of the network.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REGISTRY_BASE = "https://evaleval-entity-registry.hf.space/api/v1"
METRICS_URL = f"{REGISTRY_BASE}/metrics"
ALIASES_URL = f"{REGISTRY_BASE}/aliases?entity_type=metric"
SNAPSHOT_SCHEMA_VERSION = "eee-metric-snapshot/0.1"
VENDORING_POLICY = (
    "metrics with review_status 'reviewed' except generic placeholder ids; aliases with "
    "status 'confirmed' whose canonical_id names a vendored metric"
)
#: Reviewed registry entities that name the form of a number, not what was measured.
GENERIC_PLACEHOLDER_IDS = frozenset({"score", "mean-score", "rank"})
_METRIC_FIELDS = (
    "id",
    "display_name",
    "score_type",
    "lower_is_better",
    "min_score",
    "max_score",
    "review_status",
)
_ALIAS_FIELDS = ("raw_value", "canonical_id", "status")


def _rows(payload: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"{name} payload is not a list of objects")
    return payload


def _bound(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} is not a number or null")
    return float(value)


def build_snapshot(
    metrics: Any,
    aliases: Any,
    *,
    retrieved_date: str,
) -> dict[str, Any]:
    """Project endpoint payloads onto the vendored shape, keeping only curated entries."""

    metric_rows = _rows(metrics, "metrics")
    alias_rows = _rows(aliases, "aliases")
    vendored_metrics: dict[str, dict[str, Any]] = {}
    for row in metric_rows:
        if row.get("review_status") != "reviewed":
            continue
        metric_id = row.get("id")
        if not isinstance(metric_id, str) or not metric_id:
            raise ValueError("reviewed metric without a string id")
        if metric_id in GENERIC_PLACEHOLDER_IDS:
            continue
        if metric_id in vendored_metrics:
            raise ValueError(f"metric id {metric_id!r} is listed twice")
        display_name = row.get("display_name")
        lower_is_better = row.get("lower_is_better")
        if not isinstance(display_name, str) or not display_name:
            raise ValueError(f"metric {metric_id!r} has no display name")
        if lower_is_better is not None and not isinstance(lower_is_better, bool):
            raise ValueError(f"metric {metric_id!r} has a non-boolean direction")
        vendored_metrics[metric_id] = {
            "id": metric_id,
            "display_name": display_name,
            "score_type": row.get("score_type"),
            "lower_is_better": lower_is_better,
            "min_score": _bound(row.get("min_score"), f"metric {metric_id!r} min_score"),
            "max_score": _bound(row.get("max_score"), f"metric {metric_id!r} max_score"),
            "review_status": "reviewed",
        }
    vendored_aliases: dict[tuple[str, str], dict[str, Any]] = {}
    for row in alias_rows:
        if row.get("status") != "confirmed" or row.get("canonical_id") not in vendored_metrics:
            continue
        raw_value = row.get("raw_value")
        if not isinstance(raw_value, str) or not raw_value:
            raise ValueError("confirmed alias without a raw value")
        vendored_aliases[(raw_value, row["canonical_id"])] = {
            "raw_value": raw_value,
            "canonical_id": row["canonical_id"],
            "status": "confirmed",
        }
    return {
        "_meta": {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "source": {"metrics": METRICS_URL, "aliases": ALIASES_URL},
            "retrieved_date": retrieved_date,
            "policy": VENDORING_POLICY,
            "counts": {
                "metrics_total": len(metric_rows),
                "metrics_vendored": len(vendored_metrics),
                "aliases_total": len(alias_rows),
                "aliases_vendored": len(vendored_aliases),
            },
        },
        "metrics": [vendored_metrics[key] for key in sorted(vendored_metrics)],
        "aliases": [vendored_aliases[key] for key in sorted(vendored_aliases)],
    }


def render_snapshot(snapshot: dict[str, Any]) -> str:
    """Serialize deterministically: sorted keys, two-space indent, trailing newline."""

    return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _fetch(url: str) -> Any:
    import httpx

    response = httpx.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--aliases-file", type=Path, default=None)
    parser.add_argument(
        "--retrieved-date",
        default=dt.datetime.now(dt.UTC).date().isoformat(),
        help="ISO date recorded in _meta; defaults to today (UTC).",
    )
    args = parser.parse_args(argv)
    if (args.metrics_file is None) != (args.aliases_file is None):
        parser.error("--metrics-file and --aliases-file must be given together")
    if args.metrics_file is not None:
        metrics = json.loads(args.metrics_file.read_text(encoding="utf-8"))
        aliases = json.loads(args.aliases_file.read_text(encoding="utf-8"))
    else:
        metrics = _fetch(METRICS_URL)
        aliases = _fetch(ALIASES_URL)
    snapshot = build_snapshot(metrics, aliases, retrieved_date=args.retrieved_date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_snapshot(snapshot), encoding="utf-8")
    counts = snapshot["_meta"]["counts"]
    print(
        f"wrote {args.output}: {counts['metrics_vendored']}/{counts['metrics_total']} metrics, "
        f"{counts['aliases_vendored']}/{counts['aliases_total']} aliases"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
