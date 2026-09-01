from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from proceedings_to_eee.cli import app
from proceedings_to_eee.composition.eee import compose_eee_records
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.evaluation.human_review import build_human_review_template
from proceedings_to_eee.io import read_json, sha256_file, write_json
from proceedings_to_eee.reporting.extraction_review_cards import (
    CARD_STATEMENT,
    PINNED_EEE_SCHEMA_SHA256,
    PINNED_EEE_SCHEMA_VERSION,
    POST_HOC_CORRECTED_HOLDOUT_LIMITATION,
    CorpusCardInput,
    ExtractionReviewCardError,
    build_paper_extraction_review_card,
    render_paper_extraction_review_card,
    write_extraction_review_bundle,
)
from proceedings_to_eee.sources.manifest import SourceManifest

PRIVATE_QUOTE = "DO_NOT_PUBLISH_EVIDENCE_QUOTE"


def _manifest(paper_id: str, title: str) -> dict[str, object]:
    return {
        "schema_version": "source-manifest/0.2",
        "paper_id": paper_id,
        "title": title,
        "proceedings_url": f"https://example.org/proceedings/{paper_id}",
        "sources": [
            {
                "source_id": f"src_{paper_id.replace('-', '_')}",
                "paper_id": paper_id,
                "role": "paper",
                "original_uri": f"https://example.org/papers/{paper_id}.pdf",
                "resolved_uri": f"https://cdn.example.org/{paper_id}.pdf",
                "retrieved_at": "2026-08-04T10:00:00Z",
                "sha256": "a" * 64,
                "byte_size": 1234,
                "media_type": "application/pdf",
                "cache_relpath": "/Users/private/source.pdf",
                "license_disposition": "derived_metadata_only",
            }
        ],
    }


def _candidate(
    paper_id: str,
    *,
    exported: bool = True,
    supported: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": "candidate-observation/0.2",
        "observation_id": f"obs_{paper_id.replace('-', '')}",
        "paper_id": paper_id,
        "claim_type": "primary_result",
        "reporting_status": "present",
        "roles": [
            {
                "role": "evaluated_system",
                "raw_name": "System A",
                "confidence": 0.99,
            }
        ],
        "scope": {"dataset_raw": "Dataset A"},
        "metric": {
            "raw_name": "F1",
            "canonical_id": "f1",
            "unit": "proportion",
            "lower_is_better": False,
            "min_score": 0.0,
            "max_score": 1.0,
            "parameters": {},
        },
        "value": {"raw": "0.8", "numeric": 0.8, "unit": "proportion"},
        "evidence": [
            {
                "source_id": f"src_{paper_id.replace('-', '_')}",
                "page": 7,
                "kind": "table",
                "label": "Table 2",
                "row": "System A",
                "column": "F1",
                "quote": PRIVATE_QUOTE,
                "quote_sha256": hashlib.sha256(PRIVATE_QUOTE.encode()).hexdigest(),
            }
        ],
        "text_support": "supported" if supported else "unsupported",
        "referential_status": "resolved" if exported else "unresolved",
        "export_status": "exported" if exported else "needs_review",
        "export_reason": (
            "passed primary-result gates" if exported else "referential_status=unresolved"
        ),
        "attribution": {
            "state": "paper_produced",
            "rule_id": "explicit_exported_card_fixture",
        },
        "extraction_method": "openrouter:google/gemini-3.5-flash-lite",
        "extraction_confidence": 0.99,
    }


def _run(
    paper_id: str,
    title: str,
    manifest_sha256: str,
    *,
    candidates: int,
    exported: int,
    eee_records: int,
    review_state: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "paper-run/0.3",
        "paper_id": paper_id,
        "title": title,
        "status": "success",
        "source_manifest_sha256": manifest_sha256,
        "code": {
            "git_commit": "1" * 40,
            "git_available": True,
            "git_dirty": False,
            "source_tree_sha256": "c" * 64,
        },
        "layout_parser": "pdftotext-layout",
        "layout_parser_version": "25.0",
        "selected_pages": [7],
        "selected_blocks": [{"block_id": "rblk_a", "page": 7}],
        "result_block_segmentation": {"max_lines": 80, "overlap_lines": 6},
        "extractor": {
            "model": "google/gemini-3.5-flash-lite",
            "provider": "openrouter",
            "prompt_sha256": "d" * 64,
            "temperature": 0.0,
            "seed": 7,
            "max_tokens": 16000,
            "reasoning_effort": "minimal",
            "request_contract": {
                "privacy": {"data_collection": "deny", "zdr": True},
                "schema": {"response_format": "json_schema", "schema_strict": True},
            },
            "execution": {
                "blocks_total": 1,
                "blocks_succeeded": 1,
                "blocks_failed": 0,
                "blocks_resumed": 0,
            },
            "calls": [
                {
                    "request_id": "private-request-id",
                    "provider_response": PRIVATE_QUOTE,
                }
            ],
        },
        "verifier": {
            "enabled": False,
            "model": None,
            "request_contract": {"privacy": {"data_collection": "deny", "zdr": True}},
            "calls": [],
        },
        "eee_schema": {
            "version": PINNED_EEE_SCHEMA_VERSION,
            "sha256": PINNED_EEE_SCHEMA_SHA256,
        },
        "counts": {
            "candidates": candidates,
            "exported": exported,
            "eee_records": eee_records,
            "eee_schema_issues": 0,
            "negative_control_false_primary": 0,
        },
        "warnings": [f"private {PRIVATE_QUOTE} /Users/private/run"],
        "reference_evaluation": {
            "coverage": {"fully_annotated_labels": ["Table 2"], "sampled_labels": []},
            "detection": {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "precision_basis": 1,
                "recall_basis": 1,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 0,
            },
            "field_accuracy": {"joint_semantics": 1.0, "missingness": 1.0},
            "negative_control_safety": {
                "measurement_status": "measured",
                "controls_total": 1,
                "matched_control_count": 1,
                "false_primary_count": 0,
                "false_primary_export_count": 0,
            },
        },
    }
    if review_state is not None:
        payload["review_state"] = review_state
    return payload


def _write_paper(
    root: Path,
    paper_id: str,
    *,
    title: str | None = None,
    candidate: dict[str, object] | None = None,
    with_eee: bool = False,
    review_state: dict[str, object] | None = None,
) -> Path:
    paper_root = root / paper_id
    paper_root.mkdir(parents=True)
    title = title or f"Paper {paper_id}"
    manifest_path = paper_root / "source-manifest.json"
    write_json(manifest_path, _manifest(paper_id, title))
    observations = [] if candidate is None else [candidate]
    (paper_root / "observations.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in observations),
        encoding="utf-8",
    )
    if with_eee:
        if candidate is None:
            raise ValueError("with_eee requires one exported fixture candidate")
        eee_root = paper_root / "eee"
        eee_root.mkdir()
        records = compose_eee_records(
            manifest=SourceManifest.model_validate(_manifest(paper_id, title)),
            candidates=[CandidateObservation.model_validate(candidate)],
            schema_version=PINNED_EEE_SCHEMA_VERSION,
        )
        if len(records) != 1:
            raise AssertionError("fixture must compose exactly one EEE record")
        write_json(eee_root / "system-a.json", records[0])
    write_json(
        paper_root / "run.json",
        _run(
            paper_id,
            title,
            sha256_file(manifest_path),
            candidates=len(observations),
            exported=int(candidate is not None and candidate.get("export_status") == "exported"),
            eee_records=int(with_eee),
            review_state=review_state,
        ),
    )
    return paper_root


def _aggregate_evaluation(recall: float) -> dict[str, object]:
    return {
        "bases": {
            "reference_observations": 10,
            "field_matching": 10,
            "precision_candidates_in_fully_annotated_regions": 10,
        },
        "detection": {
            "recall": recall,
            "macro_recall": recall,
            "precision": recall,
            "f1": recall,
        },
        "field_accuracy": {
            "joint_semantics": recall,
            "evidence_structure": recall,
            "missingness": recall,
        },
        "derived_accuracy": {
            "exact_numeric_value_and_unit": recall,
            "evidence_page_and_text_support": recall,
        },
        "claim_type_classification": {"macro_f1": recall},
        "negative_control_safety": {
            "measurement_status": "measured",
            "controls_total": 10,
            "controls_matched": 5,
            "control_match_coverage": 0.5,
            "false_primary_count": 1,
            "false_primary_export_count": 0,
        },
        "quality_gates": {
            "candidate_detection_recall": {
                "direction": "at_least",
                "status": "passed" if recall >= 0.8 else "failed",
                "threshold": 0.8,
                "value": recall,
            }
        },
    }


def _write_corpus(root: Path, split: str, recall: float, paper_count: int = 10) -> Path:
    runs: list[dict[str, object]] = []
    for index in range(paper_count):
        paper_id = f"{split}-paper-{index:02d}"
        paper_root = _write_paper(
            root,
            paper_id,
            review_state={
                "status": "needs_review",
                "reasons": [
                    "selected_result_blocks_produced_zero_candidates",
                    "zero_valid_eee_records",
                ],
            },
        )
        runs.append(json.loads((paper_root / "run.json").read_text(encoding="utf-8")))
    write_json(
        root / "corpus-run.json",
        {
            "schema_version": "corpus-run/0.2",
            "corpus_id": f"{split}-corpus",
            "runs": runs,
            "reference_evaluation": _aggregate_evaluation(recall),
            "operations": {
                "wall_clock_seconds": 12.5,
                "extractor": {
                    "calls": paper_count,
                    "cost_usd_lower_bound": 0.1,
                    "input_tokens_lower_bound": 1000,
                    "output_tokens_lower_bound": 500,
                    "total_tokens_lower_bound": 1500,
                    "latency_seconds_total": 10.0,
                    "retries_lower_bound": 1,
                    "blocks_total": paper_count,
                    "blocks_resumed": 2,
                    "blocks_failed": 0,
                },
            },
        },
    )
    return root


def test_card_is_quote_free_and_retains_hashed_provenance(tmp_path: Path) -> None:
    paper_id = "paper-safe"
    paper_root = _write_paper(
        tmp_path,
        paper_id,
        title="<script>alert(1)</script>",
        candidate=_candidate(paper_id),
        with_eee=True,
        review_state={"status": "ready", "reasons": []},
    )
    aggregate_review = {
        "schema_version": "human-review-summary/0.1",
        "audit_id": "audit_1234567890abcdef1234",
        "sample": {"papers_reviewed": 1, "paper_coverage": 1.0},
        "decisions": {
            "completed": 1,
            "outcome_counts": {"confirmed": 1, "incorrect": 0, "needs_followup": 0},
            "issue_counts": {"scope": 0},
        },
    }

    card = build_paper_extraction_review_card(
        paper_root,
        split="development",
        aggregate_evaluation=_aggregate_evaluation(1.0),
        aggregate_review=aggregate_review,
        paper_review={
            "audit_id": aggregate_review["audit_id"],
            "outcome": "included_in_analyst_review",
            "decision": "withheld_in_public_artifacts",
        },
    )
    html = render_paper_extraction_review_card(card)
    serialized = json.dumps(card, sort_keys=True)

    assert card["statement"] == CARD_STATEMENT
    assert card["models"] == {
        "extractor": "google/gemini-3.5-flash-lite",
        "provider": "openrouter",
        "verifier_enabled": False,
        "verifier_model": None,
    }
    assert card["processing"]["review_state"] == {
        "status": "ready",
        "reasons": [],
        "source": "run_manifest",
    }
    assert card["review"]["paper"] == {
        "scope": "paper",
        "audit_id": aggregate_review["audit_id"],
        "outcome": "included_in_analyst_review",
        "decision": "withheld_in_public_artifacts",
    }
    assert "individual decision is withheld" in html
    assert "confirmed" not in json.dumps(card["review"]["paper"])
    assert card["counts"]["eee_records"] == 1
    assert card["eee"]["reported_count_matches_files"] is True
    assert card["provenance"]["complete"] is True
    assert (
        card["provenance"]["anchors"][0]["quote_sha256"]
        == hashlib.sha256(PRIVATE_QUOTE.encode()).hexdigest()
    )
    assert PRIVATE_QUOTE not in serialized
    assert "private-request-id" not in serialized
    assert '"request_id":' not in serialized
    assert "provider_response" not in serialized
    assert "/Users/" not in serialized
    assert CARD_STATEMENT in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert PRIVATE_QUOTE not in html

    with pytest.raises(ExtractionReviewCardError, match="different audits"):
        build_paper_extraction_review_card(
            paper_root,
            split="development",
            aggregate_review=aggregate_review,
            paper_review={
                "audit_id": "audit_different1234567890",
                "outcome": "included_in_analyst_review",
                "decision": "withheld_in_public_artifacts",
            },
        )


def test_zero_candidate_and_zero_eee_paper_has_explicit_card(tmp_path: Path) -> None:
    paper_root = _write_paper(
        tmp_path,
        "paper-empty",
        review_state={
            "status": "needs_review",
            "reasons": [
                "selected_result_blocks_produced_zero_candidates",
                "zero_valid_eee_records",
            ],
        },
    )

    card = build_paper_extraction_review_card(paper_root, split="holdout")

    assert card["counts"] == {
        "candidates": 0,
        "exported_observations": 0,
        "eee_records": 0,
        "eee_files": 0,
        "eee_schema_issues": 0,
    }
    assert card["eee"]["links"] == []
    assert card["abstention"]["applies"] is True
    assert card["abstention"]["primary_reason"] == "no_candidates"
    assert card["abstention"]["reason_counts"]["no_candidates"] == 1
    assert card["provenance"]["complete"] is False
    assert card["provenance"]["quote_hashes_present"] is False
    assert "No EEE record was produced" in " ".join(card["known_limitations"])


def test_card_rejects_private_per_paper_review_decision(tmp_path: Path) -> None:
    paper_root = _write_paper(tmp_path, "paper-private-review")

    with pytest.raises(ExtractionReviewCardError, match="paper review outcome is invalid"):
        build_paper_extraction_review_card(
            paper_root,
            split="development",
            paper_review={"outcome": "incorrect", "issue_codes": ["role"]},
        )


def test_unresolved_candidate_reasons_are_counted_without_payload(tmp_path: Path) -> None:
    paper_id = "paper-unresolved"
    paper_root = _write_paper(
        tmp_path,
        paper_id,
        candidate=_candidate(paper_id, exported=False, supported=False),
        review_state={"status": "needs_review", "reasons": ["zero_valid_eee_records"]},
    )

    card = build_paper_extraction_review_card(paper_root, split="development")
    reasons = card["abstention"]["reason_counts"]

    assert reasons["unsupported_evidence"] == 1
    assert reasons["unresolved_role"] == 1
    assert reasons["unresolved_scope"] == 1
    assert reasons["not_export_eligible"] == 1
    assert PRIVATE_QUOTE not in json.dumps(card)


def test_twenty_paper_bundle_is_deterministic_and_compares_splits(tmp_path: Path) -> None:
    development = _write_corpus(tmp_path / "development", "development", 0.9)
    holdout = _write_corpus(tmp_path / "holdout", "holdout", 0.7)
    inputs = [
        CorpusCardInput(split="development", run_root=development),
        CorpusCardInput(split="holdout", run_root=holdout),
    ]

    first = write_extraction_review_bundle(inputs, tmp_path / "public-first")
    second = write_extraction_review_bundle(inputs, tmp_path / "public-second")
    index = json.loads((first / "extraction-review-index.json").read_text(encoding="utf-8"))

    assert index["card_count"] == 20
    assert index["split_counts"] == {"development": 10, "holdout": 10}
    comparison = index["development_holdout_comparison"]
    assert comparison["status"] == "available"
    assert comparison["reference_metric_deltas"]["micro_recall"] == -0.2
    development_summary = comparison["development"]
    assert development_summary["reference_bases"]["reference_observations"] == 10
    assert development_summary["reference_metrics"]["exact_numeric_value_and_unit"] == 0.9
    assert development_summary["reference_metrics"]["page_and_text_support"] == 0.9
    assert development_summary["reference_metrics"]["claim_type_macro_f1"] == 0.9
    assert development_summary["negative_control_safety"] == {
        "measurement_status": "measured",
        "controls_total": 10,
        "controls_matched": 5,
        "control_match_coverage": 0.5,
        "false_primary_candidates": 1,
        "false_primary_exports": 0,
    }
    assert development_summary["eee_schema_validity"]["all_valid"] is True
    assert development_summary["numeric_provenance"]["all_complete"] is True
    assert development_summary["operations"]["cost_usd_lower_bound"] == 0.1
    assert development_summary["operations"]["total_tokens_lower_bound"] == 1500
    assert development_summary["quality_gate_status_counts"] == {
        "passed": 1,
        "failed": 0,
        "not_measured": 0,
    }
    assert (first / "extraction-review-index.html").read_text().count("<tr>") == 21
    assert len(list((first / "cards" / "development").glob("*.json"))) == 10
    assert len(list((first / "cards" / "holdout").glob("*.html"))) == 10
    assert (first / "SHA256SUMS").read_bytes() == (second / "SHA256SUMS").read_bytes()
    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_files == second_files
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_posthoc_corrected_holdout_is_labeled_in_cards_index_and_html(
    tmp_path: Path,
) -> None:
    development = _write_corpus(tmp_path / "development", "development", 0.9, paper_count=1)
    holdout = _write_corpus(tmp_path / "holdout", "holdout", 0.7, paper_count=1)

    output = write_extraction_review_bundle(
        [
            CorpusCardInput(split="development", run_root=development),
            CorpusCardInput(
                split="holdout",
                run_root=holdout,
                evaluation_status="post_hoc_corrected",
            ),
        ],
        tmp_path / "public-posthoc",
    )

    development_card = read_json(output / "cards" / "development" / "development-paper-00.json")
    holdout_card = read_json(output / "cards" / "holdout" / "holdout-paper-00.json")
    index = read_json(output / "extraction-review-index.json")

    assert development_card["evaluation_status"] == "development"
    assert POST_HOC_CORRECTED_HOLDOUT_LIMITATION not in development_card["known_limitations"]
    assert holdout_card["evaluation_status"] == "post_hoc_corrected"
    assert holdout_card["known_limitations"].count(POST_HOC_CORRECTED_HOLDOUT_LIMITATION) == 1
    assert index["split_summaries"]["development"]["evaluation_status"] == "development"
    assert index["split_summaries"]["holdout"]["evaluation_status"] == "post_hoc_corrected"
    assert {item["evaluation_status"] for item in index["cards"] if item["split"] == "holdout"} == {
        "post_hoc_corrected"
    }
    assert "holdout · post_hoc_corrected" in (
        output / "cards" / "holdout" / "holdout-paper-00.html"
    ).read_text(encoding="utf-8")
    assert "Evaluation status: post_hoc_corrected" in (
        output / "extraction-review-index.html"
    ).read_text(encoding="utf-8")


def test_bundle_copies_only_card_listed_eee_with_verified_hash(tmp_path: Path) -> None:
    run_root = tmp_path / "development"
    paper_id = "development-exported"
    paper_root = _write_paper(
        run_root,
        paper_id,
        candidate=_candidate(paper_id),
        with_eee=True,
        review_state={"status": "ready", "reasons": []},
    )
    run = json.loads((paper_root / "run.json").read_text(encoding="utf-8"))
    write_json(
        run_root / "corpus-run.json",
        {
            "schema_version": "corpus-run/0.2",
            "corpus_id": "development-exported",
            "runs": [run],
            "reference_evaluation": _aggregate_evaluation(1.0),
        },
    )
    output = write_extraction_review_bundle(
        [CorpusCardInput(split="development", run_root=run_root)],
        tmp_path / "public-exported",
    )
    card_path = output / "cards" / "development" / f"{paper_id}.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    link = card["eee"]["links"][0]
    copied = output / "eee" / "development" / paper_id / "system-a.json"

    assert link["href"] == f"../../eee/development/{paper_id}/system-a.json"
    assert copied.is_file()
    assert sha256_file(copied) == link["sha256"]
    assert (card_path.parent / link["href"]).resolve() == copied.resolve()
    assert f"eee/development/{paper_id}/system-a.json" in (output / "SHA256SUMS").read_text(
        encoding="utf-8"
    )


def test_private_eee_payload_aborts_bundle_atomically(tmp_path: Path) -> None:
    run_root = tmp_path / "development"
    paper_id = "development-private-eee"
    paper_root = _write_paper(
        run_root,
        paper_id,
        candidate=_candidate(paper_id),
        with_eee=True,
        review_state={"status": "ready", "reasons": []},
    )
    write_json(paper_root / "eee" / "system-a.json", {"quote": PRIVATE_QUOTE})
    run = json.loads((paper_root / "run.json").read_text(encoding="utf-8"))
    write_json(
        run_root / "corpus-run.json",
        {
            "schema_version": "corpus-run/0.2",
            "corpus_id": "development-private-eee",
            "runs": [run],
        },
    )
    output = tmp_path / "public-private-eee"

    with pytest.raises(ExtractionReviewCardError, match="forbidden key quote"):
        write_extraction_review_bundle(
            [CorpusCardInput(split="development", run_root=run_root)],
            output,
        )

    assert not output.exists()


def test_schema_invalid_public_eee_aborts_bundle_atomically(tmp_path: Path) -> None:
    run_root = tmp_path / "development"
    paper_id = "development-invalid-schema"
    paper_root = _write_paper(
        run_root,
        paper_id,
        candidate=_candidate(paper_id),
        with_eee=True,
        review_state={"status": "ready", "reasons": []},
    )
    eee_path = paper_root / "eee" / "system-a.json"
    eee = json.loads(eee_path.read_text(encoding="utf-8"))
    eee["schema_version"] = "wrong-version"
    write_json(eee_path, eee)
    run = json.loads((paper_root / "run.json").read_text(encoding="utf-8"))
    write_json(
        run_root / "corpus-run.json",
        {"schema_version": "corpus-run/0.2", "corpus_id": paper_id, "runs": [run]},
    )
    output = tmp_path / "public-invalid-schema"

    with pytest.raises(ExtractionReviewCardError, match="pinned-schema validation"):
        write_extraction_review_bundle(
            [CorpusCardInput(split="development", run_root=run_root)],
            output,
        )

    assert not output.exists()


def test_eee_count_and_flattened_provenance_mismatches_are_fatal(tmp_path: Path) -> None:
    paper_id = "paper-integrity"
    paper_root = _write_paper(
        tmp_path,
        paper_id,
        candidate=_candidate(paper_id),
        with_eee=True,
        review_state={"status": "ready", "reasons": []},
    )
    run_path = paper_root / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["counts"]["eee_records"] = 0
    write_json(run_path, run)
    with pytest.raises(ExtractionReviewCardError, match="record count does not match"):
        build_paper_extraction_review_card(paper_root, split="development")

    run["counts"]["eee_records"] = 1
    write_json(run_path, run)
    eee_path = paper_root / "eee" / "system-a.json"
    eee = json.loads(eee_path.read_text(encoding="utf-8"))
    eee["evaluation_results"][0]["score_details"]["details"]["evidence_1_source_sha256"] = "f" * 64
    write_json(eee_path, eee)
    with pytest.raises(ExtractionReviewCardError, match="source/evidence provenance mismatch"):
        build_paper_extraction_review_card(paper_root, split="development")


def test_manifest_and_quote_hash_tampering_are_rejected(tmp_path: Path) -> None:
    manifest_root = _write_paper(tmp_path, "paper-manifest-tamper")
    manifest_path = manifest_root / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["media_type"] = "application/x-tampered-pdf"
    write_json(manifest_path, manifest)
    with pytest.raises(ExtractionReviewCardError, match="source manifest SHA-256"):
        build_paper_extraction_review_card(manifest_root, split="development")

    paper_id = "paper-quote-tamper"
    quote_root = _write_paper(tmp_path, paper_id, candidate=_candidate(paper_id))
    observations_path = quote_root / "observations.jsonl"
    observation = json.loads(observations_path.read_text(encoding="utf-8"))
    observation["evidence"][0]["quote_sha256"] = "0" * 64
    observations_path.write_text(json.dumps(observation) + "\n", encoding="utf-8")
    with pytest.raises(ExtractionReviewCardError, match="invalid candidate observation"):
        build_paper_extraction_review_card(quote_root, split="development")


def test_public_text_and_run_review_state_are_strictly_allowlisted(tmp_path: Path) -> None:
    paper_root = _write_paper(
        tmp_path,
        "paper-private",
        review_state={"status": "needs_review", "reasons": ["invented_private_reason"]},
    )
    with pytest.raises(ExtractionReviewCardError, match="unsupported reasons"):
        build_paper_extraction_review_card(paper_root, split="development")

    safe_root = _write_paper(tmp_path, "paper-limit")
    with pytest.raises(ExtractionReviewCardError, match="private or secret"):
        build_paper_extraction_review_card(
            safe_root,
            split="development",
            known_limitations=["Private annotation at /Users/reviewer/notes.txt"],
        )


def test_cli_reproducibly_builds_one_or_two_splits(tmp_path: Path) -> None:
    development = _write_corpus(tmp_path / "development", "development", 0.9, paper_count=1)
    holdout = _write_corpus(tmp_path / "holdout", "holdout", 0.7, paper_count=1)
    review_template_path = tmp_path / "private-development-review.json"
    review_template = build_human_review_template(development, sample_size=1).model_dump(
        mode="json"
    )
    review_template["items"][0]["decision"] = {
        "outcome": "incorrect",
        "issue_codes": ["role"],
        "notes": "private decision details",
    }
    write_json(review_template_path, review_template)
    output = tmp_path / "public-cards"

    result = CliRunner().invoke(
        app,
        [
            "build-extraction-review-cards",
            "--development-run-root",
            str(development),
            "--holdout-run-root",
            str(holdout),
            "--development-human-review-template",
            str(review_template_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads((output / "extraction-review-index.json").read_text())["card_count"] == 2
    development_card = json.loads(
        (output / "cards" / "development" / "development-paper-00.json").read_text()
    )
    assert development_card["review"]["paper"] == {
        "scope": "paper",
        "audit_id": review_template["audit_id"],
        "outcome": "included_in_analyst_review",
        "decision": "withheld_in_public_artifacts",
    }
    assert "incorrect" not in json.dumps(development_card["review"]["paper"])
    assert "role" not in json.dumps(development_card["review"]["paper"])
    assert "private decision details" not in json.dumps(development_card)
    assert (output / "SHA256SUMS").is_file()


def test_cli_posthoc_flag_labels_holdout_and_requires_holdout_root(tmp_path: Path) -> None:
    holdout = _write_corpus(tmp_path / "holdout", "holdout", 0.7, paper_count=1)
    output = tmp_path / "public-posthoc-cli"

    result = CliRunner().invoke(
        app,
        [
            "build-extraction-review-cards",
            "--holdout-run-root",
            str(holdout),
            "--holdout-posthoc-corrected",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    card = read_json(output / "cards" / "holdout" / "holdout-paper-00.json")
    assert card["evaluation_status"] == "post_hoc_corrected"
    assert POST_HOC_CORRECTED_HOLDOUT_LIMITATION in card["known_limitations"]

    missing_holdout = CliRunner().invoke(
        app,
        [
            "build-extraction-review-cards",
            "--development-run-root",
            str(holdout),
            "--holdout-posthoc-corrected",
            "--output",
            str(tmp_path / "invalid-posthoc-cli"),
        ],
    )
    assert missing_holdout.exit_code != 0
    assert "post-hoc correction label requires a holdout run root" in missing_holdout.output
