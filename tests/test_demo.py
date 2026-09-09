from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from proceedings_to_eee.cli import app
from proceedings_to_eee.demo import (
    DEMO_MANIFEST_NAME,
    DEMO_PDF_NAME,
    DEMO_REVIEWER_ID,
    DEMO_SHA256SUMS_NAME,
    generate_demo_pdf,
    run_offline_demo,
    verify_demo_bundle,
)
from proceedings_to_eee.io import read_json, sha256_file
from proceedings_to_eee.sources.manifest import SourceManifest

RUNNER = CliRunner()


def test_generated_demo_pdf_is_byte_deterministic_and_poppler_readable(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "nested" / "second.pdf"

    first_sha = generate_demo_pdf(first)
    second_sha = generate_demo_pdf(second)

    assert first_sha == second_sha
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().startswith(b"%PDF-1.4")
    info = subprocess.run(
        ["pdfinfo", str(first)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    assert re.search(r"^Pages:\s+2$", info, re.MULTILINE)
    text = subprocess.run(
        ["pdftotext", "-layout", str(first), "-"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    assert "Our team trained DemoNet and evaluated DemoNet on DemoSet in this study." in text
    assert "ExternalBaseline (Smith et al., 2024)" in text
    assert "DemoNet" in text and "0.91" in text


def test_demo_cli_runs_full_pipeline_without_credentials_or_network(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "offline-demo"
    secret = "must-not-appear-in-demo-output"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setenv("ERE_OPENROUTER_API_KEY", secret)

    def reject_network_client(*args, **kwargs):
        del args, kwargs
        raise AssertionError("offline demo must not construct the OpenRouter client")

    monkeypatch.setattr(
        "proceedings_to_eee.providers.openrouter.OpenRouterClient.__init__",
        reject_network_client,
    )
    monkeypatch.setattr("httpx.Client", reject_network_client)

    result = RUNNER.invoke(app, ["demo", "--output", str(output)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "verified-offline-demo"
    assert payload["counts"] == {
        "review_items": 2,
        "decisions_completed": 2,
        "decisions_pending": 0,
        "outcomes_exported": 1,
        "outcomes_withheld": 1,
        "outcomes_failed": 0,
        "eee_records": 1,
        "eee_observations": 1,
    }
    assert not (output / "work-run").exists()
    assert (output / DEMO_PDF_NAME).is_file()
    assert (output / "sealed-run" / "review.html").is_file()
    assert (output / "review" / "review-lock.json").is_file()
    assert (output / "derived" / "verification.json").is_file()
    assert (output / DEMO_MANIFEST_NAME).is_file()
    assert (output / DEMO_SHA256SUMS_NAME).is_file()

    audit = read_json(output / DEMO_MANIFEST_NAME)
    assert audit["network_used"] is False
    assert audit["credentials_required"] is False
    assert audit["synthetic_fixture"] is True
    assert audit["empirical_annotation"] is False
    assert audit["source"]["license"] == "CC0-1.0"
    assert audit["source"]["sha256"] == sha256_file(output / DEMO_PDF_NAME)
    assert audit["counts"]["verified_eee_records"] == 1
    assert audit["pipeline"]["normal_extraction_api"] is True

    source_manifest = SourceManifest.model_validate(
        read_json(output / "sealed-run" / "source-manifest.json")
    )
    assert source_manifest.sources[0].license_disposition.value == "redistributable"
    run = read_json(output / "sealed-run" / "run.json")
    assert run["counts"]["candidates"] == 2
    assert run["counts"]["eee_records"] == 0
    assert run["extractor"]["calls"]
    assert all(
        call["provider_returned"] is None
        and call["provider_returned_disposition"] == "unrecognized_omitted"
        and call["data_collection"] == "deny"
        and call["zdr"] is True
        for call in run["extractor"]["calls"]
    )
    observations = [
        json.loads(line)
        for line in (output / "sealed-run" / "observations.jsonl").read_text().splitlines()
    ]
    external = next(
        observation
        for observation in observations
        if observation["roles"][0]["raw_name"] == "ExternalBaseline"
    )
    assert external["attribution"]["state"] == "externally_sourced"

    decisions = [
        json.loads(line)
        for line in (output / "review" / "decisions.jsonl").read_text().splitlines()
    ]
    assert all(decision["status"] == "completed" for decision in decisions)
    assert all(decision["authority"]["mode"] == "single_expert" for decision in decisions)
    assert sorted(decision["origin_decision"] for decision in decisions) == [
        "externally_sourced",
        "paper_produced",
    ]

    outcomes = [
        json.loads(line)
        for line in (output / "derived" / "export-outcomes.jsonl").read_text().splitlines()
    ]
    assert sorted(outcome["state"] for outcome in outcomes) == ["exported", "withheld"]
    [withheld] = [outcome for outcome in outcomes if outcome["state"] == "withheld"]
    assert withheld["failure_codes"] == ["ORIGIN_EXTERNAL"]
    eee_files = list((output / "derived").glob("*/eee/*.json"))
    assert len(eee_files) == 1
    eee = read_json(eee_files[0])
    provenance = eee["evaluation_results"][0]["score_details"]["details"]
    assert provenance["origin_decision"] == "paper_produced"
    assert provenance["decision_authority"] == "single_expert"

    verification = verify_demo_bundle(output)
    assert verification["eee_record_count"] == 1
    assert verification["exported_observation_count"] == 1
    all_bytes = b"".join(path.read_bytes() for path in output.rglob("*") if path.is_file())
    assert secret.encode() not in all_bytes
    assert str(tmp_path).encode() not in all_bytes
    assert DEMO_REVIEWER_ID.encode() not in b"".join(
        path.read_bytes() for path in (output / "derived").rglob("*") if path.is_file()
    )


def test_demo_cli_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    result = RUNNER.invoke(app, ["demo", "--output", str(output)])

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "status": "offline-demo-failed",
        "code": "OUTPUT_EXISTS",
        "detail": "demo output already exists; choose a new directory",
    }
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert "Traceback" not in result.output


def test_complete_demo_bundle_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first-demo"
    second = tmp_path / "second-demo"

    run_offline_demo(first)
    run_offline_demo(second)

    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
