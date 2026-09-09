from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proceedings_to_eee.cli import (
    DEFAULT_EEE_SCHEMA_PATH,
    DEFAULT_SCHEMA_SHA256,
    app,
)


def _corpus(tmp_path: Path) -> Path:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"fixture")
    path = tmp_path / "corpus.yaml"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pilot-corpus/0.2",
                "corpus_id": "cli-status-fixture",
                "evaluation_split": "development",
                "description": "fixture",
                "papers": [
                    {
                        "paper_id": "fixture-paper",
                        "title": "Fixture paper",
                        "year": 2026,
                        "venue": "Fixture",
                        "pdf_path": str(pdf),
                        "perspective_role": "evaluated_system",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _materialize(output: Path, *, paper_status: str, eee_records: int) -> None:
    paper = output / "fixture-paper"
    (output / "private").mkdir(parents=True)
    paper.mkdir(parents=True)
    (output / "corpus-run.json").write_text("{}\n", encoding="utf-8")
    (output / "corpus-review.html").write_text("review\n", encoding="utf-8")
    (output / "private" / "provider-budget-ledger.jsonl").write_text("{}\n", encoding="utf-8")
    for name in ("run.json", "review.html", "observations.jsonl", "verifications.jsonl"):
        (paper / name).write_text("{}\n", encoding="utf-8")
    if eee_records:
        (paper / "eee").mkdir()
        (paper / "eee" / "fixture.json").write_text("{}\n", encoding="utf-8")
    if paper_status != "success":
        (paper / "run.json").write_text('{"error":{"code":"fixture"}}\n', encoding="utf-8")


def _summary(*, status: str, eee_records: int, review: int, failed: int) -> dict[str, Any]:
    paper_status = (
        "bounded_incomplete"
        if status == "bounded_incomplete"
        else ("error" if failed else "success")
    )
    return {
        "schema_version": "corpus-run/0.2",
        "status": status,
        "papers": 1,
        "papers_total": 1,
        "papers_not_started": 0,
        "papers_succeeded": 0 if failed else 1,
        "papers_failed": failed,
        "papers_needing_review": review,
        "totals": {"eee_records": eee_records, "candidates": 1},
        "runs": [
            {
                "paper_id": "fixture-paper",
                "status": paper_status,
                "counts": {"eee_records": eee_records},
            }
        ],
        "provider_budget": {
            "structured_calls_started": 1,
            "bounded_stop": (
                {
                    "code": "provider_budget_exhausted",
                    "reason": "structured_call_limit",
                    "provider_call_dispatched": False,
                }
                if status == "bounded_incomplete"
                else None
            ),
        },
    }


@pytest.mark.parametrize(
    ("pipeline_status", "eee_records", "review", "failed", "terminal", "exit_code"),
    [
        ("success", 1, 0, 0, "successful-export", 0),
        ("success", 0, 1, 0, "completed-with-review", 2),
        ("partial_failure", 0, 1, 1, "technical-failure", 1),
        ("bounded_incomplete", 0, 1, 1, "bounded-incomplete", 3),
    ],
)
def test_run_corpus_has_distinct_terminal_status_exit_artifacts_and_next_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pipeline_status: str,
    eee_records: int,
    review: int,
    failed: int,
    terminal: str,
    exit_code: int,
) -> None:
    corpus = _corpus(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr("proceedings_to_eee.cli.runtime_key", lambda: "fixture-key")
    monkeypatch.setattr("proceedings_to_eee.cli.OpenRouterClient", lambda **kwargs: object())

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        _materialize(
            output,
            paper_status=("error" if failed else "success"),
            eee_records=eee_records,
        )
        return _summary(
            status=pipeline_status,
            eee_records=eee_records,
            review=review,
            failed=failed,
        )

    monkeypatch.setattr("proceedings_to_eee.cli.run_corpus", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "run-corpus",
            str(corpus),
            "--model",
            "fixture/model",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == exit_code, result.output
    payload = json.loads(result.output)
    assert payload["status"] == terminal
    assert payload["exit_code"] == exit_code
    assert payload["artifacts"]["corpus_run"]["exists"]
    assert payload["artifacts"]["corpus_review"]["exists"]
    assert payload["artifacts"]["papers"][0]["observations"]["exists"]
    assert payload["artifacts"]["papers"][0]["paper_review"]["exists"]
    assert payload["artifacts"]["papers"][0]["eee"]["exists"] is bool(eee_records)
    if terminal in {"successful-export", "completed-with-review"}:
        assert "seal-run" in payload["next_command"]
    elif terminal == "bounded-incomplete":
        assert "--max-structured-calls 20000" in payload["next_command"]
    else:
        assert "run-corpus" in payload["next_command"]
        assert payload["artifacts"]["errors"]


def test_quiet_still_prints_status_artifacts_and_next_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus = _corpus(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr("proceedings_to_eee.cli.runtime_key", lambda: "fixture-key")
    monkeypatch.setattr("proceedings_to_eee.cli.OpenRouterClient", lambda **kwargs: object())

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        _materialize(output, paper_status="success", eee_records=0)
        return _summary(status="success", eee_records=0, review=1, failed=0)

    monkeypatch.setattr("proceedings_to_eee.cli.run_corpus", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "run-corpus",
            str(corpus),
            "--model",
            "fixture/model",
            "--output",
            str(output),
            "--quiet",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["status"] == "completed-with-review"
    assert "artifacts" in payload and "next_command" in payload
    assert "provider_budget" not in payload


@pytest.mark.parametrize(
    ("model", "invalid_options", "explanation"),
    [
        pytest.param(
            "fixture/model",
            [
                "--max-provider-cost-usd",
                "0.1",
                "--provider-call-cost-reservation-usd",
                "0.2",
            ],
            "invalid or inconsistent input",
            id="reservation-exceeds-budget",
        ),
        pytest.param(
            "fixture/extractor",
            ["--origin-model", "fixture/origin"],
            "origin_model requires verifier_model",
            id="origin-requires-verifier",
        ),
        pytest.param(
            "fixture/extractor",
            ["--verifier-model", "fixture/verifier"],
            "verifier_model requires tuple_model",
            id="verifier-requires-tuple",
        ),
        pytest.param(
            "fixture/extractor",
            ["--row-model", "fixture/row"],
            "row_model requires --row-enumeration",
            id="row-model-requires-row-stage",
        ),
    ],
)
def test_invalid_local_configuration_fails_before_credentials_or_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: str,
    invalid_options: list[str],
    explanation: str,
) -> None:
    corpus = _corpus(tmp_path)
    output = tmp_path / "must-not-exist"
    credential_reads = 0

    def forbidden_key() -> str:
        nonlocal credential_reads
        credential_reads += 1
        raise AssertionError("credential access must not happen")

    monkeypatch.setattr("proceedings_to_eee.cli.runtime_key", forbidden_key)
    result = CliRunner().invoke(
        app,
        [
            "run-corpus",
            str(corpus),
            "--model",
            model,
            "--output",
            str(output),
            *invalid_options,
        ],
    )

    assert result.exit_code == 1
    assert credential_reads == 0
    assert not output.exists()
    payload = json.loads(result.output)
    assert payload["phase"] == "local_configuration"
    assert payload["error"]["message"] == explanation
    assert "Traceback" not in result.output


def test_run_help_exposes_separate_stage_models() -> None:
    result = CliRunner().invoke(app, ["run-corpus", "--help"], env={"COLUMNS": "240"})

    assert result.exit_code == 0, result.output
    assert "--row-model" in result.output
    assert "--tuple-model" in result.output
    assert "--tuple-max-tokens" in result.output
    assert "--verifier-model" in result.output
    assert "--verifier-max-tokens" in result.output
    assert "--origin-model" in result.output
    assert "--origin-max-tokens" in result.output


def test_reasoning_effort_reaches_the_pipeline_settings() -> None:
    """`--reasoning-effort none` must arrive in the settings that write run.json."""

    from proceedings_to_eee.cli import _settings
    from proceedings_to_eee.extraction.llm import EXTRACTOR_REASONING_EFFORT
    from proceedings_to_eee.pipeline import _extractor_run_configuration

    default = _settings(
        schema_path=DEFAULT_EEE_SCHEMA_PATH,
        schema_sha256=DEFAULT_SCHEMA_SHA256,
        output=Path("runs/unused"),
        model="vendor/model",
    )
    disabled = _settings(
        schema_path=DEFAULT_EEE_SCHEMA_PATH,
        schema_sha256=DEFAULT_SCHEMA_SHA256,
        output=Path("runs/unused"),
        model="vendor/model",
        reasoning_effort="none",
    )

    assert default.reasoning_effort == EXTRACTOR_REASONING_EFFORT
    assert disabled.reasoning_effort == "none"
    assert _extractor_run_configuration(disabled)["reasoning_effort"] == "none"
    # The extractor contract gates checkpoint reuse, so the two runs cannot share one.
    assert _extractor_run_configuration(default) != _extractor_run_configuration(disabled)
