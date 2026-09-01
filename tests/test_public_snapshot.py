from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from proceedings_to_eee import cli
from proceedings_to_eee.cli import app
from proceedings_to_eee.composition.eee import compose_eee_records
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.io import sha256_file, write_json
from proceedings_to_eee.public_snapshot import (
    PublicSnapshotError,
    _project_human_review_summary,
    _validate_human_review_corpus_population,
    build_public_snapshot,
)
from proceedings_to_eee.sources.manifest import SourceManifest
from proceedings_to_eee.validation.eee_schema import load_schema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = PROJECT_ROOT / "schemas" / "eee-0.2.2" / "eval.schema.json"
SCHEMA_SHA256 = "088fed8029d42fb3a607aa67e1a05c39e425241b5cd90803705b37562f402f2a"


def _private_inputs(
    root: Path,
    manifest: SourceManifest,
    candidate: CandidateObservation,
) -> tuple[Path, Path, Path, Path]:
    run_root = root / "run"
    paper_root = run_root / manifest.paper_id
    schema, authority = load_schema(SCHEMA, SCHEMA_SHA256)
    del schema
    record = compose_eee_records(
        manifest=manifest,
        candidates=[candidate],
        schema_version=authority.version,
    )[0]
    write_json(paper_root / "source-manifest.json", manifest)
    write_json(paper_root / "eee" / "atlas-moderation-api.json", record)
    write_json(
        run_root / "corpus-run.json",
        {
            "schema_version": "corpus-run/0.2",
            "corpus_id": "pilot-corpus",
            "status": "success",
            "generated_at": "2026-08-04T12:00:00Z",
            "papers": 1,
            "papers_succeeded": 1,
            "papers_failed": 0,
            "papers_with_eee": 1,
            "totals": {
                "candidates": 1,
                "primary_results": 1,
                "exported": 1,
                "eee_records": 1,
                "eee_schema_issues": 0,
                "spot_checks": 1,
                "spot_checks_exact": 1,
            },
            "runs": [
                {
                    "paper_id": manifest.paper_id,
                    "title": manifest.title,
                    "status": "success",
                    "wall_clock_seconds": 3.25,
                    "counts": {
                        "candidates": 1,
                        "primary_results": 1,
                        "exported": 1,
                        "eee_records": 1,
                        "eee_schema_issues": 0,
                        "spot_checks": 1,
                        "spot_checks_exact": 1,
                    },
                    "extractor": {
                        "provider": "openrouter",
                        "model": "example/extractor",
                        "temperature": 0,
                        "seed": 7,
                        "warnings": ["DO_NOT_PUBLISH_WARNING " + "/" + "Users/example/private"],
                        "calls": [
                            {
                                "request_id": "DO_NOT_PUBLISH_REQUEST",
                                "exact_quote": "DO_NOT_PUBLISH_QUOTE",
                                "input_tokens": 100,
                                "output_tokens": 20,
                                "cost_usd": 0.001,
                                "model_returned": "example/extractor",
                                "provider_returned": "example-provider",
                            }
                        ],
                    },
                    "verifier": {"enabled": False, "calls": []},
                    "eee_schema": {
                        "version": authority.version,
                        "sha256": authority.sha256,
                    },
                    "reference_path": "/" + "Users/example/private/reference.json",
                }
            ],
        },
    )
    write_json(
        run_root / "reference-audit.json",
        {
            "schema_version": "reference-corpus-audit/0.1",
            "corpus_id": "pilot-corpus",
            "status": "success",
            "papers": 1,
            "papers_passed": 1,
            "papers_failed": 0,
            "papers_skipped": 0,
            "text_verified": 1,
            "visual_verified": 0,
            "failed_evidence": 0,
            "results": [
                {
                    "paper_id": manifest.paper_id,
                    "status": "success",
                    "page_count": 10,
                    "source_hash_matches": True,
                    "text_verified": 1,
                    "visual_verified": 0,
                    "failed_evidence": 0,
                    "warnings": ["DO_NOT_PUBLISH_AUDIT_WARNING"],
                }
            ],
        },
    )
    model_selection = root / "private-model-selection.json"
    write_json(
        model_selection,
        {
            "schema_version": "extractor-bakeoff/0.2",
            "bakeoff_id": "pilot-extractors",
            "configuration_sha256": "b" * 64,
            "inputs": [
                {
                    "manifest_path": "/" + "Users/example/private/source-manifest.json",
                    "prompt": "DO_NOT_PUBLISH_PROMPT",
                }
            ],
            "determinism": {
                "seed": 7,
                "temperature": 0,
                "prompt_sha256": "c" * 64,
                "reference_prompt_isolation": True,
                "segmentation": {"max_lines": 40, "max_blocks_per_page": 6},
            },
            "models": [
                {
                    "model": "example/extractor",
                    "label": "Example extractor",
                    "aggregate": {
                        "execution": {
                            "cases_attempted": 1,
                            "cases_succeeded": 1,
                            "calls_attempted": 1,
                            "calls_succeeded": 1,
                        },
                        "quality": {"scored_cases": 1},
                    },
                    "cases": [
                        {
                            "request_id": "DO_NOT_PUBLISH_BAKEOFF_REQUEST",
                            "raw_response": "DO_NOT_PUBLISH_RESPONSE",
                        }
                    ],
                }
            ],
        },
    )
    human_review = root / "private-human-review-summary.json"
    write_json(
        human_review,
        {
            "schema_version": "human-review-summary/0.1",
            "audit_id": "audit_" + "d" * 20,
            "sampling_policy": "risk-stratified-paper-coverage/0.1",
            "population": {
                "candidates": 1,
                "papers": 1,
                "papers_without_candidates": 0,
            },
            "sample": {
                "requested": 1,
                "reviewed": 1,
                "papers_reviewed": 1,
                "paper_coverage": 1.0,
                "risk_score_min": 10,
                "risk_score_max": 10,
                "risk_score_mean": 10.0,
                "risk_reason_counts": {"exported": 1},
                "item_type_counts": {
                    "candidate": 1,
                    "paper_without_candidates": 0,
                },
                "papers_without_candidates_reviewed": 0,
            },
            "decisions": {
                "completed": 1,
                "outcome_counts": {
                    "confirmed": 1,
                    "incorrect": 0,
                    "needs_followup": 0,
                },
                "issue_counts": {
                    "claim_type": 0,
                    "role": 0,
                    "version": 0,
                    "scope": 0,
                    "metric": 0,
                    "unit": 0,
                    "value": 0,
                    "evidence": 0,
                    "export_decision": 0,
                    "duplicate": 0,
                    "other": 0,
                },
            },
            "privacy": {
                "contains_evidence_quotes": False,
                "contains_candidate_payloads": False,
                "contains_provider_raw_data": False,
                "contains_local_paths": False,
                "contains_reviewer_notes": False,
            },
            "private_template_path": "/" + "Users/example/private/review-template.json",
            "private_quote": "DO_NOT_PUBLISH_REVIEW_QUOTE",
        },
    )
    return (
        run_root,
        model_selection,
        human_review,
        paper_root / "eee" / "atlas-moderation-api.json",
    )


def _json_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | {
            child_key for child in value.values() for child_key in _json_keys(child)
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return {child_key for child in value for child_key in _json_keys(child)}
    return set()


def test_public_snapshot_is_allowlist_only_and_reproducible(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    first = build_public_snapshot(
        snapshot_id="pilot-v1",
        corpus_run_root=run_root,
        model_selection_path=model_selection,
        human_review_summary_path=human_review,
        schema_path=SCHEMA,
        schema_sha256=SCHEMA_SHA256,
        output_root=tmp_path / "public-a",
        selected_model="example/extractor",
    )
    second = build_public_snapshot(
        snapshot_id="pilot-v1",
        corpus_run_root=run_root,
        model_selection_path=model_selection,
        human_review_summary_path=human_review,
        schema_path=SCHEMA,
        schema_sha256=SCHEMA_SHA256,
        output_root=tmp_path / "public-b",
        selected_model="example/extractor",
    )

    expected_files = {
        "README.md",
        "SHA256SUMS",
        "corpus-review.html",
        "eee/synthetic-audit-study/atlas-moderation-api.json",
        "human-review.json",
        "model-selection.json",
        "reference-audit.json",
        "snapshot.json",
        "sources.json",
    }
    first_files = {
        path.relative_to(first).as_posix() for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix() for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files == expected_files
    for relative in sorted(expected_files):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()

    public_bytes = b"\n".join((first / path).read_bytes() for path in sorted(expected_files))
    for private_marker in (
        b"DO_NOT_PUBLISH",
        b"/" + b"Users/",
        b"cache_relpath",
        b"request_id",
        b"raw_response",
    ):
        assert private_marker not in public_bytes
    public_keys = set()
    for path in first.rglob("*.json"):
        public_keys |= _json_keys(json.loads(path.read_text(encoding="utf-8")))
    assert not {"calls", "warnings", "quote", "exact_quote"} & public_keys

    selection = json.loads((first / "model-selection.json").read_text(encoding="utf-8"))
    assert selection["selection"] == {
        "selected_model": "example/extractor",
        "status": "selected",
    }
    assert "inputs" not in selection
    assert "cases" not in selection["models"][0]
    review = json.loads((first / "human-review.json").read_text(encoding="utf-8"))
    assert review["schema_version"] == "human-review-summary/0.1"
    assert review["sample"]["paper_coverage"] == 1.0
    assert review["decisions"]["completed"] == 1
    assert review["privacy"] == {
        "contains_candidate_payloads": False,
        "contains_evidence_quotes": False,
        "contains_local_paths": False,
        "contains_provider_raw_data": False,
        "contains_reviewer_notes": False,
    }
    assert "private_template_path" not in review
    assert "private_quote" not in review
    public_eee = json.loads(
        (first / "eee/synthetic-audit-study/atlas-moderation-api.json").read_text(encoding="utf-8")
    )
    provenance = public_eee["evaluation_results"][0]["score_details"]["details"]
    assert provenance["paper_id"] == "synthetic-audit-study"
    assert provenance["evidence_1_source_id"] == "src_paper"
    assert provenance["evidence_1_source_sha256"] == "a" * 64
    assert provenance["evidence_1_page"] == "7"
    assert provenance["evidence_1_kind"] == "table"
    assert provenance["evidence_1_quote_sha256"]
    assert "quote" not in provenance
    snapshot = json.loads((first / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["human_review"] == {
        "audit_id": "audit_" + "d" * 20,
        "paper_coverage": 1.0,
        "path": "human-review.json",
        "source_artifact_sha256": sha256_file(human_review),
    }
    sources = json.loads((first / "sources.json").read_text(encoding="utf-8"))
    assert sources["papers"][0]["sources"][0]["sha256"] == "a" * 64
    assert "cache_relpath" not in sources["papers"][0]["sources"][0]

    checksum_lines = (first / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    checksum_paths = [line.split("  ", 1)[1] for line in checksum_lines]
    assert checksum_paths == sorted(expected_files - {"SHA256SUMS"})
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        assert digest == sha256_file(first / relative)


def test_invalid_eee_aborts_without_publishing(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, eee_path = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    invalid = json.loads(eee_path.read_text(encoding="utf-8"))
    invalid["schema_version"] = "wrong"
    write_json(eee_path, invalid)

    with pytest.raises(PublicSnapshotError, match="EEE validation failed"):
        build_public_snapshot(
            snapshot_id="invalid-eee",
            corpus_run_root=run_root,
            model_selection_path=model_selection,
            human_review_summary_path=human_review,
            schema_path=SCHEMA,
            schema_sha256=SCHEMA_SHA256,
            output_root=tmp_path / "public",
        )

    assert not (tmp_path / "public" / "invalid-eee").exists()


def test_eee_without_evidence_provenance_aborts_without_publishing(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, eee_path = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    record = json.loads(eee_path.read_text(encoding="utf-8"))
    details = record["evaluation_results"][0]["score_details"]["details"]
    details.pop("evidence_anchor_count")
    write_json(eee_path, record)

    with pytest.raises(PublicSnapshotError, match="lacks evidence anchors"):
        build_public_snapshot(
            snapshot_id="missing-provenance",
            corpus_run_root=run_root,
            model_selection_path=model_selection,
            human_review_summary_path=human_review,
            schema_path=SCHEMA,
            schema_sha256=SCHEMA_SHA256,
            output_root=tmp_path / "public",
        )

    assert not (tmp_path / "public" / "missing-provenance").exists()

    details["evidence_anchor_count"] = "1"
    details["evidence_1_quote_text"] = "PRIVATE VERBATIM EVIDENCE"
    write_json(eee_path, record)
    with pytest.raises(PublicSnapshotError, match="embeds an evidence quote"):
        build_public_snapshot(
            snapshot_id="quote-leak",
            corpus_run_root=run_root,
            model_selection_path=model_selection,
            human_review_summary_path=human_review,
            schema_path=SCHEMA,
            schema_sha256=SCHEMA_SHA256,
            output_root=tmp_path / "public",
        )

    assert not (tmp_path / "public" / "quote-leak").exists()
    assert not list((tmp_path / "public").glob(".invalid-eee.*"))


def test_local_source_uri_is_rejected(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    manifest_path = run_root / manifest.paper_id / "source-manifest.json"
    local_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    local_manifest["sources"][0]["resolved_uri"] = "file:///" + "Users/example/paper.pdf"
    write_json(manifest_path, local_manifest)

    with pytest.raises(PublicSnapshotError, match=r"not a public HTTP\(S\) URI"):
        build_public_snapshot(
            snapshot_id="local-source",
            corpus_run_root=run_root,
            model_selection_path=model_selection,
            human_review_summary_path=human_review,
            schema_path=SCHEMA,
            schema_sha256=SCHEMA_SHA256,
            output_root=tmp_path / "public",
        )


def test_incomplete_human_review_aborts_without_publishing(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    incomplete = json.loads(human_review.read_text(encoding="utf-8"))
    incomplete["decisions"]["completed"] = 0
    incomplete["decisions"]["outcome_counts"] = {
        "confirmed": 0,
        "incorrect": 0,
        "needs_followup": 0,
    }
    write_json(human_review, incomplete)

    with pytest.raises(PublicSnapshotError, match="not fully decided"):
        build_public_snapshot(
            snapshot_id="incomplete-review",
            corpus_run_root=run_root,
            model_selection_path=model_selection,
            human_review_summary_path=human_review,
            schema_path=SCHEMA,
            schema_sha256=SCHEMA_SHA256,
            output_root=tmp_path / "public",
        )

    assert not (tmp_path / "public" / "incomplete-review").exists()


def test_human_review_summary_cannot_overstate_paper_coverage(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    _, _, human_review, _ = _private_inputs(tmp_path / "inputs", manifest, eligible_candidate)
    overstated = json.loads(human_review.read_text(encoding="utf-8"))
    overstated["population"]["papers"] = 10
    overstated["sample"]["papers_reviewed"] = 10
    overstated["sample"]["paper_coverage"] = 1.0
    write_json(human_review, overstated)

    with pytest.raises(PublicSnapshotError, match="more papers than reviewed items"):
        _project_human_review_summary(human_review)


def test_human_review_full_coverage_requires_zero_candidate_review(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    _, _, human_review, _ = _private_inputs(tmp_path / "inputs", manifest, eligible_candidate)
    impossible = json.loads(human_review.read_text(encoding="utf-8"))
    impossible["population"]["candidates"] = 2
    impossible["population"]["papers"] = 2
    impossible["population"]["papers_without_candidates"] = 1
    impossible["sample"]["requested"] = 2
    impossible["sample"]["reviewed"] = 2
    impossible["sample"]["papers_reviewed"] = 2
    impossible["sample"]["paper_coverage"] = 1.0
    impossible["sample"]["item_type_counts"] = {
        "candidate": 2,
        "paper_without_candidates": 0,
    }
    impossible["sample"]["papers_without_candidates_reviewed"] = 0
    impossible["decisions"]["completed"] = 2
    impossible["decisions"]["outcome_counts"]["confirmed"] = 2
    write_json(human_review, impossible)

    with pytest.raises(PublicSnapshotError, match="papers with candidates"):
        _project_human_review_summary(human_review)


def test_human_review_named_counts_cannot_exceed_reviewed_items(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    _, _, human_review, _ = _private_inputs(tmp_path / "inputs", manifest, eligible_candidate)
    overstated = json.loads(human_review.read_text(encoding="utf-8"))
    overstated["sample"]["risk_reason_counts"]["exported"] = 999
    write_json(human_review, overstated)
    with pytest.raises(PublicSnapshotError, match="risk-reason count"):
        _project_human_review_summary(human_review)

    overstated["sample"]["risk_reason_counts"]["exported"] = 1
    overstated["decisions"]["issue_counts"]["evidence"] = 999
    write_json(human_review, overstated)
    with pytest.raises(PublicSnapshotError, match="issue count"):
        _project_human_review_summary(human_review)


def test_human_review_zero_candidate_population_must_match_corpus() -> None:
    human_review = {
        "population": {
            "papers": 10,
            "candidates": 658,
            "papers_without_candidates": 0,
        }
    }
    corpora = [
        {
            "papers": 10,
            "totals": {"candidates": 658},
            "papers_detail": [
                {"counts": {"candidates": 0}},
                *({"counts": {"candidates": 1}} for _ in range(9)),
            ],
        }
    ]

    with pytest.raises(PublicSnapshotError, match="zero-candidate paper population"):
        _validate_human_review_corpus_population(human_review, corpora)


def test_local_human_review_template_is_not_accepted_as_public_summary(
    tmp_path: Path,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    run_root, model_selection, human_review, _ = _private_inputs(
        tmp_path / "inputs", manifest, eligible_candidate
    )
    write_json(
        human_review,
        {
            "schema_version": "human-review-template/0.1",
            "candidate": {"quote": "DO_NOT_PUBLISH_REVIEW_QUOTE"},
        },
    )

    with pytest.raises(PublicSnapshotError, match="not an aggregate review summary"):
        build_public_snapshot(
            snapshot_id="private-review-template",
            corpus_run_root=run_root,
            model_selection_path=model_selection,
            human_review_summary_path=human_review,
            schema_path=SCHEMA,
            schema_sha256=SCHEMA_SHA256,
            output_root=tmp_path / "public",
        )

    assert not (tmp_path / "public" / "private-review-template").exists()


def test_public_snapshot_cli_forwards_repeated_run_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "primary"
    additional = tmp_path / "additional"
    primary.mkdir()
    additional.mkdir()
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")
    human_review = tmp_path / "human-review.json"
    human_review.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_build(**kwargs: object) -> Path:
        captured.update(kwargs)
        return tmp_path / "examples" / "pilot-cli"

    monkeypatch.setattr(cli, "build_public_snapshot", fake_build)
    result = CliRunner().invoke(
        app,
        [
            "export-public-snapshot",
            "pilot-cli",
            str(primary),
            str(selection),
            str(human_review),
            "--schema-path",
            str(SCHEMA),
            "--output-root",
            str(tmp_path / "examples"),
            "--additional-run-root",
            str(additional),
            "--selected-model",
            "example/extractor",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["snapshot_id"] == "pilot-cli"
    assert captured["corpus_run_root"] == primary
    assert captured["human_review_summary_path"] == human_review
    assert captured["additional_run_roots"] == [additional]
    assert captured["selected_model"] == "example/extractor"


def test_validate_eee_cli_uses_pinned_default_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
) -> None:
    _, _, _, record_path = _private_inputs(tmp_path / "inputs", manifest, eligible_candidate)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["validate-eee", str(record_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"schema": "0.2.2", "issues": []}
