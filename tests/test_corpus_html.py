from __future__ import annotations

import json

import pytest

from proceedings_to_eee.reporting.corpus_html import (
    render_corpus_html,
    render_corpus_html_file,
)


def _ten_paper_run() -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for index in range(1, 11):
        runs.append(
            {
                "paper_id": f"paper-{index:02d}",
                "title": "<script>alert(1)</script>" if index == 1 else f"Paper {index}",
                "counts": {
                    "candidates": index,
                    "exported": index - 1,
                    "eee_records": 1,
                    "eee_schema_issues": 1 if index == 10 else 0,
                    "spot_checks": 2,
                    "spot_checks_exact": 1 if index == 10 else 2,
                },
                "extractor": {"calls": [{"cost_usd": index / 100}]},
                "verifier": {"calls": [{"cost_usd": index / 1000}]},
                "wall_clock_seconds": index * 10,
            }
        )
    return {
        "schema_version": "corpus-run/0.2",
        "corpus_id": "demo-10-papers",
        "papers": 10,
        "totals": {"candidates": 999_999},
        "reference_evaluation": {
            "detection": {"precision": 0.95, "recall": 0.9},
            "field_accuracy": {"joint_semantics": 0.96},
            "negative_control_safety": {"false_primary_count": 0},
        },
        "runs": runs,
    }


def test_render_corpus_html_has_ten_rows_and_recomputed_totals() -> None:
    run = _ten_paper_run()

    first = render_corpus_html(run)
    second = render_corpus_html(run)

    assert first == second
    assert first.count('<tr class="paper-row') == 10
    assert "demo-10-papers · 10 papers" in first
    assert "<span>Candidates</span><strong>55</strong>" in first
    assert "<span>Exported</span><strong>45</strong>" in first
    assert "<span>EEE records</span><strong>10</strong>" in first
    assert "<strong>1</strong>" in first
    assert "<span>Spot-checks</span><strong>19/20</strong>" in first
    assert "<span>Cost</span><strong>$0.6050</strong>" in first
    assert "<span>Runtime</span>" in first
    assert "<strong>9m 10s</strong>" in first
    assert "<span>Reference recall</span><strong>90.0%</strong>" in first
    assert "<span>Covered precision</span><strong>95.0%</strong>" in first
    assert "<span>Joint semantics</span><strong>96.0%</strong>" in first
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in first
    assert "<script>alert(1)</script>" not in first
    assert 'data-paper-id="paper-10"' in first
    assert 'badge-error">Schema errors' in first
    assert "999,999" not in first


def test_render_corpus_html_file_accepts_legacy_collection_aliases(tmp_path) -> None:
    source = tmp_path / "corpus-run.json"
    destination = tmp_path / "report" / "index.html"
    source.write_text(
        json.dumps(
            {
                "id": "alias-run",
                "papers": [
                    {
                        "id": "arxiv:example",
                        "paper_title": "Alias paper",
                        "candidates": [{}, {}, {}],
                        "exports": [{}, {}],
                        "eee_records": [{}],
                        "schema_errors": [],
                        "spot_checks": ["passed", {"status": "failed"}],
                        "total_cost_usd": 0.0025,
                        "runtime_seconds": 61,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    returned = render_corpus_html_file(source, destination)

    assert returned == destination
    report = destination.read_text(encoding="utf-8")
    assert "alias-run · 1 paper" in report
    assert "Alias paper" in report
    assert "1/2" in report
    assert "$0.0025" in report
    assert "1m 01s" in report


def test_render_corpus_html_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        render_corpus_html(
            {
                "runs": [
                    {
                        "paper_id": "bad",
                        "counts": {"candidates": -1},
                    }
                ]
            }
        )


def test_render_corpus_html_uses_successful_telemetry_including_resumed_calls() -> None:
    report = render_corpus_html(
        {
            "corpus_id": "resumed-cost",
            "runs": [
                {
                    "paper_id": "paper-a",
                    "counts": {},
                    "extractor": {
                        "calls": [],
                        "resumed_calls": [{"cost_usd": 99}],
                        "successful_call_telemetry": {
                            "calls": 3,
                            "cost_usd_lower_bound": 0.123456,
                            "total_tokens_lower_bound": 4567,
                            "retries_lower_bound": 2,
                        },
                        "execution": {"blocks_failed": 1, "blocks_resumed": 3},
                    },
                }
            ],
        }
    )

    assert "<span>Cost</span><strong>$0.1235</strong>" in report
    assert "<span>Successful calls</span>" in report
    assert "<strong>3</strong>" in report
    assert "<span>Tokens</span><strong>4,567+</strong>" in report
    assert "<span>Retries / failed blocks</span>" in report
    assert "<strong>2 / 1</strong>" in report
    assert "<span>Resumed blocks</span>" in report
