from __future__ import annotations

import json
from pathlib import Path

from test_reviewed_export import (
    ORIGIN_TEXT,
    _build_sealed_run,
    _complete_decision,
    _compose_reviewed_fixture,
    _prepare,
)
from typer.testing import CliRunner

from proceedings_to_eee.cli import app
from proceedings_to_eee.io import read_json, write_json
from proceedings_to_eee.reviewed_export.workflow import (
    REVIEW_DECISIONS_NAME,
    validate_export_review,
)

RUNNER = CliRunner()
PRIVATE_SENTINEL = "DO_NOT_PRINT_PRIVATE_REVIEW_EVIDENCE"


def _invoke(*arguments: str):
    return RUNNER.invoke(app, list(arguments))


def _payload(output: str) -> dict[str, object]:
    parsed = json.loads(output)
    assert isinstance(parsed, dict)
    return parsed


def _assert_safe_typed_failure(*, output: str, tmp_path: Path, expected_code: str) -> None:
    payload = _payload(output)
    assert payload == {
        "status": "reviewed-export-failed",
        "code": expected_code,
        "detail": payload["detail"],
    }
    assert isinstance(payload["detail"], str)
    assert payload["detail"]
    assert "Traceback" not in output
    assert PRIVATE_SENTINEL not in output
    assert ORIGIN_TEXT.strip() not in output
    assert "reviewer-a" not in output
    assert str(tmp_path) not in output


def test_reviewed_export_cli_round_trip(tmp_path: Path) -> None:
    sealed = _build_sealed_run(tmp_path)
    review = tmp_path / "private-review"
    derived = tmp_path / "derived"

    prepared = _invoke(
        "prepare-export-review",
        str(sealed),
        "--output",
        str(review),
    )
    assert prepared.exit_code == 0, prepared.output
    prepared_payload = _payload(prepared.output)
    assert prepared_payload["status"] == "prepared"
    assert prepared_payload["items"] == 1
    assert Path(str(prepared_payload["decisions"])) == review / REVIEW_DECISIONS_NAME

    _complete_decision(review)
    locked = _invoke("validate-export-review", str(review))
    assert locked.exit_code == 0, locked.output
    locked_payload = _payload(locked.output)
    assert locked_payload["status"] == "locked"
    assert locked_payload["completed"] == 1
    assert locked_payload["pending"] == 0
    assert len(str(locked_payload["review_lock_sha256"])) == 64

    composed = _invoke(
        "compose-reviewed-eee",
        str(sealed),
        "--decisions",
        str(review / REVIEW_DECISIONS_NAME),
        "--output",
        str(derived),
    )
    assert composed.exit_code == 0, composed.output
    composed_payload = _payload(composed.output)
    assert composed_payload["status"] == "verified-reviewed-export"
    assert composed_payload["counts"] == {
        "review_items": 1,
        "decisions_completed": 1,
        "decisions_pending": 0,
        "outcomes_exported": 1,
        "outcomes_withheld": 0,
        "outcomes_failed": 0,
        "eee_records": 1,
        "eee_observations": 1,
    }

    verified = _invoke("verify-run", str(derived))
    assert verified.exit_code == 0, verified.output
    verification = _payload(verified.output)
    assert verification["status"] == "verified"
    assert len(str(verification["derived_run_sha256"])) == 64
    assert verification["payload_file_count"] >= 3
    assert verification["eee_record_count"] == 1
    assert verification["exported_observation_count"] == 1

    combined_output = prepared.output + locked.output + composed.output + verified.output
    assert ORIGIN_TEXT.strip() not in combined_output
    assert "reviewer-a" not in combined_output


def test_prepare_export_review_cli_reports_safe_typed_unsealed_failure(
    tmp_path: Path,
) -> None:
    unsealed = tmp_path / "private-user-run"
    unsealed.mkdir()
    (unsealed / "private-input.txt").write_text(
        f"{PRIVATE_SENTINEL}\n{ORIGIN_TEXT}", encoding="utf-8"
    )

    result = _invoke(
        "prepare-export-review",
        str(unsealed),
        "--output",
        str(tmp_path / "review"),
    )

    assert result.exit_code == 1
    _assert_safe_typed_failure(
        output=result.output,
        tmp_path=tmp_path,
        expected_code="RUN_NOT_SEALED",
    )


def test_verify_run_cli_reports_safe_typed_tamper_failure(tmp_path: Path) -> None:
    sealed, review = _prepare(tmp_path)
    derived = tmp_path / "derived"
    _complete_decision(review)
    validate_export_review(review)
    _compose_reviewed_fixture(sealed, review, derived)

    [eee_path] = derived.glob("*/eee/*.json")
    eee = read_json(eee_path)
    eee["private_evidence"] = PRIVATE_SENTINEL
    write_json(eee_path, eee)

    result = _invoke("verify-run", str(derived))

    assert result.exit_code == 1
    _assert_safe_typed_failure(
        output=result.output,
        tmp_path=tmp_path,
        expected_code="ARTIFACT_HASH_MISMATCH",
    )
