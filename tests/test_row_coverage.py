from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from proceedings_to_eee.cli import app
from proceedings_to_eee.corpus import CorpusSpec, PaperSpec
from proceedings_to_eee.evaluation import row_plan as row_plan_module
from proceedings_to_eee.evaluation.row_coverage import (
    score_paper_row_coverage,
    score_run_row_coverage,
)
from proceedings_to_eee.evaluation.row_plan import (
    build_next_run_row_plan_report,
    build_run_row_plan_report,
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.result_blocks import (
    ResultBlock,
    ResultBlockConfig,
    segment_page_result_blocks,
)
from proceedings_to_eee.extraction.row_enumeration import build_row_enumeration_plan
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from proceedings_to_eee.providers.openrouter import ProviderCall
from proceedings_to_eee.sources.manifest import FrozenSource, SourceManifest, SourceRole

PAGE = """Table 2. Detection performance per model.

     Model            Precision   Recall    F1
     System Cedar        0.42       0.81   0.55
     System Juniper      0.57       0.63   0.60
     System Maple        0.67       0.49   0.57
"""


def _page(text: str = PAGE, page: int = 7) -> PageFragment:
    return PageFragment(
        fragment_id=f"frag_src_paper_{page:04d}",
        source_id="src_paper",
        page=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=0,
        result_signal_score=1.0,
    )


def _candidate(quote: str, row: str, value: str, page: int = 7) -> dict:
    return {
        "paper_id": "fixture",
        "claim_type": "primary_result",
        "roles": [{"role": "evaluated_system", "raw_name": row, "confidence": 0.9}],
        "scope": {"dataset_raw": "Fixture"},
        "metric": {"raw_name": "F1", "unit": "proportion"},
        "value": {"raw": value, "numeric": float(value), "unit": "proportion"},
        "evidence": [
            {
                "source_id": "src_paper",
                "page": page,
                "kind": "table",
                "label": "Table 2",
                "row": row,
                "column": "F1",
                "quote": quote,
            }
        ],
        "export_status": "exported",
        "extraction_method": "fixture",
        "extraction_confidence": 0.95,
    }


def _paper(tmp_path: Path, candidates: list[dict]) -> Path:
    paper_dir = tmp_path / "fixture-paper"
    (paper_dir / "private").mkdir(parents=True, exist_ok=True)
    page = _page()
    write_json(paper_dir / "run.json", {"paper_id": "fixture-paper", "status": "success"})
    write_json(
        paper_dir / "private" / "layout.json",
        PdfLayout(
            source_id="src_paper",
            parser="fixture",
            parser_version="fixture/1",
            page_count=7,
            pages=[_page("filler\n", number) for number in range(1, 7)] + [page],
        ),
    )
    write_json(paper_dir / "private" / "result-blocks.json", segment_page_result_blocks(page))
    write_jsonl(paper_dir / "observations.jsonl", candidates)
    return paper_dir


def _plan(paper_dir: Path):
    layout = PdfLayout.model_validate(read_json(paper_dir / "private" / "layout.json"))
    blocks = [
        ResultBlock.model_validate(item)
        for item in read_json(paper_dir / "private" / "result-blocks.json")
    ]
    return build_row_enumeration_plan(layout, blocks)


def _write_ledger(
    paper_dir: Path,
    dispositions: list[tuple[str, str, list[dict]]],
    *,
    unresolved_row_ids: list[str] | None = None,
) -> None:
    plan = _plan(paper_dir)
    unresolved = unresolved_row_ids or []
    records = {
        row_id: {
            "row_id": row_id,
            "disposition": disposition,
            "candidates": candidates,
            "note": None,
        }
        for row_id, disposition, candidates in dispositions
    }
    counts = {name: 0 for name in ("result", "not_result", "uncertain")}
    for _, disposition, _ in dispositions:
        counts[disposition] += 1
    plan_sha256 = write_json(paper_dir / "private" / "row-enumeration-plan.json", plan)
    write_json(
        paper_dir / "private" / "row-enumeration.json",
        {
            "schema_version": "row-enumeration-outcome/0.1",
            "plan_sha256": plan_sha256,
            "records": records,
            "unresolved_row_ids": unresolved,
            "unbatchable_row_ids": [row.row_id for row in plan.unbatchable_rows],
            "telemetry": {
                "rows_resolved": len(records),
                "rows_unresolved": len(unresolved),
                "rows_unbatchable": len(plan.unbatchable_rows),
                "dispositions": counts,
            },
        },
    )


def _next_run_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, CorpusSpec, str]:
    paper = _paper(tmp_path, [])
    write_json(
        paper / "run.json",
        {
            "paper_id": "fixture-paper",
            "status": "success",
            "extractor": {
                "successful_call_telemetry": {
                    "calls": 2,
                    "cost_reported_calls": 2,
                    "cost_usd_lower_bound": 0.02,
                }
            },
        },
    )
    spec = PaperSpec(
        paper_id="fixture-paper",
        title="Fixture Paper",
        year=2026,
        venue="Fixture Venue",
        pdf_url="https://example.test/fixture.pdf",
        perspective_role="evaluated_system",
        include_pages=[7],
    )
    corpus = CorpusSpec(corpus_id="fixture-corpus", description="fixture", papers=[spec])
    source = FrozenSource(
        source_id="src_paper",
        paper_id="fixture-paper",
        role=SourceRole.PAPER,
        original_uri=str(spec.pdf_url),
        retrieved_at=datetime(2026, 8, 26, tzinfo=UTC),
        sha256="a" * 64,
        byte_size=1,
        cache_relpath="data/sources/aa/fixture.pdf",
    )
    source_manifest = SourceManifest(
        paper_id="fixture-paper",
        title=spec.title,
        sources=[source],
    )
    layout = PdfLayout.model_validate(read_json(paper / "private" / "layout.json"))
    monkeypatch.setattr(
        row_plan_module,
        "_reconstruct_frozen_layout",
        lambda **_: (source_manifest, layout),
    )

    current_blocks = segment_page_result_blocks(
        layout.pages[6],
        config=ResultBlockConfig(max_blocks_per_page=6),
    )
    assert current_blocks
    model = "fixture/model"
    checkpoint_contract = {
        "schema_version": "extractor-block-checkpoint-contract/0.1",
        "paper_id": spec.paper_id,
        "paper_title": spec.title,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(source_manifest)),
        "layout_parser": layout.parser,
        "layout_parser_version": layout.parser_version,
        "extractor": row_plan_module._current_extractor_configuration(model),
    }
    call = ProviderCall(
        model_requested=model,
        model_returned=model,
        prompt_sha256="b" * 64,
        response_sha256="c" * 64,
        temperature=0.0,
        reasoning_effort="minimal",
        max_tokens=16_000,
        seed=7,
        schema_name="paper_evaluation_candidates",
        schema_sha256="d" * 64,
        latency_seconds=0.01,
        attempts=1,
    )
    write_json(
        paper / "private" / "extractor-checkpoint.json",
        {
            "schema_version": "extractor-block-checkpoint/0.1",
            "contract": checkpoint_contract,
            "contract_sha256": "e" * 64,
            "blocks": {
                block.block_id: {
                    "block_text_sha256": block.text_sha256,
                    "candidates": [],
                    "call": call,
                    "warnings": [],
                }
                for block in current_blocks
            },
        },
    )
    write_json(paper / "private" / "result-blocks.json", [])
    return tmp_path, corpus, model


def _write_compatible_row_checkpoint(
    run_root: Path,
    corpus: CorpusSpec,
    model: str,
) -> Path:
    spec = corpus.papers[0]
    paper_dir = run_root / spec.paper_id
    manifest, layout = row_plan_module._reconstruct_frozen_layout(
        paper_dir=paper_dir,
        spec=spec,
        project_root=run_root,
    )
    blocks = segment_page_result_blocks(
        layout.pages[spec.include_pages[0] - 1],
        config=ResultBlockConfig(max_blocks_per_page=6),
    )
    plan = build_row_enumeration_plan(layout, blocks)
    row_configuration = {
        "enabled": True,
        **row_plan_module._current_row_extractor_configuration(model),
        "limits": plan.config.model_dump(mode="json"),
    }
    contract = {
        "schema_version": "row-enumeration-checkpoint-contract/0.1",
        "paper_id": spec.paper_id,
        "paper_title": spec.title,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_parser": layout.parser,
        "layout_parser_version": layout.parser_version,
        "plan_sha256": sha256_bytes(canonical_json_bytes(plan)),
        "row_enumeration": row_configuration,
        "code": {
            "git_commit": "old-code",
            "git_dirty": True,
            "git_available": True,
            "source_tree_sha256": "9" * 64,
        },
    }
    call = ProviderCall(
        model_requested=model,
        model_returned=model,
        prompt_sha256="1" * 64,
        response_sha256="2" * 64,
        temperature=0.0,
        reasoning_effort="minimal",
        max_tokens=16_000,
        seed=7,
        schema_name="paper_table_row_dispositions",
        schema_sha256="3" * 64,
        latency_seconds=0.01,
        attempts=1,
    )
    entries = {}
    for batch in plan.batches:
        entries[batch.batch_id] = {
            "batch_sha256": sha256_bytes(canonical_json_bytes(batch)),
            "records": {
                row.row_id: {
                    "row_id": row.row_id,
                    "disposition": "not_result",
                    "candidates": [],
                    "note": "fixture",
                }
                for row in batch.rows
            },
            "calls": [call.model_dump(mode="json")],
            "attempts": [
                {
                    "batch_id": batch.batch_id,
                    "depth": 0,
                    "row_ids": [row.row_id for row in batch.rows],
                    "status": "success",
                    "resolved_row_ids": [row.row_id for row in batch.rows],
                    "unresolved_row_ids": [],
                    "unknown_row_ids": [],
                    "completed_provider_call": True,
                }
            ],
            "unresolved_row_ids": [],
            "unbatchable_row_ids": [],
            "unknown_row_ids": [],
            "invalid_row_reasons": {},
            "warnings": [],
            "telemetry": {},
        }
    checkpoint_path = paper_dir / "private" / "row-enumeration-checkpoint.json"
    write_json(
        checkpoint_path,
        {
            "schema_version": "row-enumeration-checkpoint/0.1",
            "contract": contract,
            "contract_sha256": sha256_bytes(canonical_json_bytes(contract)),
            "batches": entries,
        },
    )
    return checkpoint_path


def test_one_row_of_three_gives_a_third(tmp_path: Path) -> None:
    paper = _paper(
        tmp_path,
        [_candidate("System Cedar        0.42       0.81   0.55", "System Cedar", "0.55")],
    )
    score = score_paper_row_coverage(paper)
    assert score["rows_shown"] == 3
    assert score["rows_with_a_candidate"] == 1
    assert score["row_coverage"] == pytest.approx(1 / 3)
    assert "row_disposition_coverage" not in score


def test_every_row_covered_gives_one(tmp_path: Path) -> None:
    paper = _paper(
        tmp_path,
        [
            _candidate("System Cedar        0.42       0.81   0.55", "System Cedar", "0.55"),
            _candidate("System Juniper      0.57       0.63   0.60", "System Juniper", "0.60"),
            _candidate("System Maple        0.67       0.49   0.57", "System Maple", "0.57"),
        ],
    )
    assert score_paper_row_coverage(paper)["row_coverage"] == pytest.approx(1.0)


def test_a_paper_with_no_table_anchors_is_named(tmp_path: Path) -> None:
    _paper(tmp_path, [])
    summary = score_run_row_coverage(tmp_path)
    assert summary["papers_with_zero_table_anchors"] == ["fixture-paper"]
    assert summary["row_coverage"] == pytest.approx(0.0)
    empty = summary["papers"][0]["tables_with_no_candidate"]
    assert empty and empty[0]["table_label"] == "Table 2"


def test_rows_never_shown_to_the_extractor_are_not_counted(tmp_path: Path) -> None:
    """The denominator is enumeration, not page selection."""

    paper = _paper(tmp_path, [])
    write_json(paper / "private" / "result-blocks.json", [])
    score = score_paper_row_coverage(paper)
    assert score["rows_shown"] == 0
    assert score["row_coverage"] is None


def test_valid_ledger_reports_dispositions_separately_from_candidate_coverage(
    tmp_path: Path,
) -> None:
    candidate = _candidate("System Cedar        0.42       0.81   0.55", "System Cedar", "0.55")
    paper = _paper(tmp_path, [candidate])
    rows = _plan(paper).rows
    _write_ledger(
        paper,
        [
            (rows[0].row_id, "result", [candidate]),
            (rows[1].row_id, "not_result", []),
            (rows[2].row_id, "uncertain", []),
        ],
    )

    score = score_paper_row_coverage(paper)
    assert score["rows_with_a_candidate"] == 1
    assert score["row_coverage"] == pytest.approx(1 / 3)
    assert score["row_disposition_coverage"] == {
        "ledger_status": "validated",
        "rows_planned": 3,
        "rows_resolved": 3,
        "resolution_coverage": 1.0,
        "result_rows": 1,
        "not_result_rows": 1,
        "uncertain_rows": 1,
        "unresolved_rows": 0,
        "unbatchable_rows": 0,
        "basis": (
            "Resolved/planned uses the validated row-disposition ledger. result rows carry "
            "one or more candidates; not_result and uncertain are resolved abstentions and "
            "are never counted as candidate-bearing rows. Unbatchable rows are unresolved."
        ),
    }
    aggregate = score_run_row_coverage(tmp_path)["row_disposition_coverage"]
    assert aggregate["rows_resolved"] == 3
    assert aggregate["not_result_rows"] == 1
    assert aggregate["complete_for_scored_papers"] is True


def test_partial_ledger_counts_unresolved_rows_without_inventing_candidates(
    tmp_path: Path,
) -> None:
    paper = _paper(tmp_path, [])
    rows = _plan(paper).rows
    _write_ledger(
        paper,
        [(rows[0].row_id, "not_result", [])],
        unresolved_row_ids=[rows[1].row_id, rows[2].row_id],
    )

    score = score_paper_row_coverage(paper)
    disposition = score["row_disposition_coverage"]
    assert disposition["rows_resolved"] == 1
    assert disposition["resolution_coverage"] == pytest.approx(1 / 3)
    assert disposition["not_result_rows"] == 1
    assert disposition["unresolved_rows"] == 2
    assert score["rows_with_a_candidate"] == 0


def test_malformed_or_mismatched_ledger_fails_closed(tmp_path: Path) -> None:
    paper = _paper(tmp_path, [])
    rows = _plan(paper).rows
    _write_ledger(
        paper,
        [(rows[0].row_id, "not_result", [])],
        unresolved_row_ids=[rows[1].row_id, "trow_unknown"],
    )

    with pytest.raises(ValueError, match="unknown row ids"):
        score_paper_row_coverage(paper)


def test_incomplete_plan_ledger_pair_fails_closed(tmp_path: Path) -> None:
    paper = _paper(tmp_path, [])
    write_json(paper / "private" / "row-enumeration-plan.json", _plan(paper))

    with pytest.raises(ValueError, match="incomplete row ledger pair"):
        score_paper_row_coverage(paper)


def test_not_result_with_a_candidate_is_rejected_by_typed_ledger(tmp_path: Path) -> None:
    paper = _paper(tmp_path, [])
    rows = _plan(paper).rows
    _write_ledger(
        paper,
        [
            (
                rows[0].row_id,
                "not_result",
                [_candidate(rows[0].raw_text, "System Cedar", "0.55")],
            )
        ],
        unresolved_row_ids=[rows[1].row_id, rows[2].row_id],
    )

    with pytest.raises(ValueError, match="malformed disposition record"):
        score_paper_row_coverage(paper)


def test_offline_run_plan_aggregates_bounds_and_explicit_cost_basis(tmp_path: Path) -> None:
    paper = _paper(tmp_path, [])
    write_json(
        paper / "run.json",
        {
            "paper_id": "fixture-paper",
            "status": "success",
            "extractor": {
                "successful_call_telemetry": {
                    "calls": 2,
                    "cost_reported_calls": 2,
                    "cost_usd_lower_bound": 0.02,
                }
            },
        },
    )

    report = build_run_row_plan_report(tmp_path)

    assert report["mode"] == "offline_stored_artifact_diagnostic"
    assert report["legacy_block_baseline"]["successful_calls"] == 2
    assert report["row_plan"] == {
        "tables_considered": 1,
        "dense_tables": 1,
        "rows_planned": 3,
        "unbatchable_rows": 0,
        "base_batches": 1,
        "expected_row_calls": 1,
        "hard_max_row_calls": 3,
        "basis": (
            "Expected row calls assume one call per base batch. Hard maximum applies the "
            "configured deterministic recovery bound to every base batch."
        ),
    }
    assert report["call_estimates"]["expected_total_calls"] == 3
    assert report["call_estimates"]["hard_max_total_calls"] == 5
    assert report["call_estimates"]["expected_total_call_multiplier"] == pytest.approx(1.5)
    assert report["call_estimates"]["hard_max_total_call_multiplier"] == pytest.approx(2.5)
    costs = report["cost_estimates"]
    assert costs["historical_mean_cost_per_successful_call_usd"] == pytest.approx(0.01)
    assert costs["estimated_expected_additional_row_cost_usd"] == pytest.approx(0.01)
    assert costs["estimated_expected_combined_cost_usd"] == pytest.approx(0.03)
    assert costs["estimated_hard_max_combined_cost_usd"] == pytest.approx(0.05)


def test_offline_run_plan_omits_cost_estimates_without_complete_basis(tmp_path: Path) -> None:
    paper = _paper(tmp_path, [])
    write_json(
        paper / "run.json",
        {
            "paper_id": "fixture-paper",
            "status": "success",
            "extractor": {
                "successful_call_telemetry": {
                    "calls": 2,
                    "cost_reported_calls": 1,
                    "cost_usd_lower_bound": 0.01,
                }
            },
        },
    )

    report = build_run_row_plan_report(tmp_path)

    assert report["cost_estimates"] is None
    assert report["cost_estimate_status"] == {
        "available": False,
        "reason": "requires explicit cost for every historical successful legacy call",
    }


def test_next_run_plan_reconstructs_current_inputs_and_validates_checkpoint_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, corpus, model = _next_run_fixture(monkeypatch, tmp_path)

    report = build_next_run_row_plan_report(
        run_root,
        corpus=corpus,
        project_root=tmp_path,
        extractor_model=model,
    )

    assert report["mode"] == "offline_current_code_frozen_source_preflight"
    assert report["provider_or_network_calls"] == 0
    assert report["stored_artifact_comparison"]["result_blocks"] == 0
    assert report["next_run_blocks"] == {
        "current_result_blocks": 1,
        "reusable_checkpoint_blocks": 1,
        "new_block_calls": 0,
        "reuse_fraction": 1.0,
        "basis": (
            "Current local layout extraction, current corpus page selection, current block "
            "segmentation, and fully validated exact historical checkpoint entries."
        ),
    }
    assert report["row_plan"]["dense_tables"] == 1
    assert report["row_plan"]["rows_planned"] == 3
    assert report["row_plan"]["base_batches"] == 1
    assert report["row_plan"]["hard_max_row_calls"] == 3
    assert report["next_run_rows"] == {
        "current_base_batches": 1,
        "reusable_checkpoint_batches": 0,
        "new_base_batches": 1,
        "reuse_fraction": 0.0,
        "expected_new_structured_chat_invocations": 1,
        "hard_max_new_structured_chat_invocations": 3,
        "basis": (
            "Current exact row plan and fully validated checkpoint entries. Each new base "
            "batch requires one expected invocation and permits one bounded split level."
        ),
    }
    assert report["next_run_preflight"]["expected_logical_calls"] == 2
    assert report["next_run_preflight"]["hard_max_logical_calls"] == 4
    assert report["next_run_preflight"]["expected_new_structured_chat_invocations"] == 1
    assert report["next_run_preflight"]["hard_max_new_structured_chat_invocations"] == 3
    assert (
        report["next_run_preflight"]["max_transport_attempts_per_structured_chat_invocation"] == 4
    )
    assert report["next_run_preflight"]["hard_max_new_transport_attempts"] == 12
    assert report["cost_estimates"]["estimated_expected_new_run_cost_usd"] == pytest.approx(0.01)
    assert report["cost_estimates"]["estimated_hard_max_new_run_cost_usd"] == pytest.approx(0.03)


def test_next_run_plan_counts_all_blocks_new_when_model_envelope_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, corpus, model = _next_run_fixture(monkeypatch, tmp_path)
    _write_compatible_row_checkpoint(run_root, corpus, model)

    report = build_next_run_row_plan_report(
        run_root,
        corpus=corpus,
        project_root=tmp_path,
        extractor_model="different/model",
    )

    paper = report["papers"][0]
    assert paper["checkpoint_reuse"]["status"] == "incompatible_request_envelope"
    assert paper["checkpoint_reuse"]["mismatched_request_envelope_fields"] == ["extractor"]
    assert paper["row_checkpoint_reuse"]["status"] == "incompatible_request_envelope"
    assert paper["row_checkpoint_reuse"]["mismatched_request_envelope_fields"] == [
        "row_enumeration"
    ]
    assert report["next_run_blocks"]["reusable_checkpoint_blocks"] == 0
    assert report["next_run_blocks"]["new_block_calls"] == 1
    assert paper["checkpoint_reuse"]["hard_max_new_structured_chat_invocations"] == 5
    assert report["next_run_preflight"]["expected_logical_calls"] == 2
    assert report["next_run_preflight"]["hard_max_logical_calls"] == 8
    assert report["next_run_preflight"]["expected_new_structured_chat_invocations"] == 2
    assert report["next_run_preflight"]["hard_max_new_structured_chat_invocations"] == 8
    assert report["next_run_preflight"]["hard_max_new_transport_attempts"] == 32
    assert report["cost_estimates"]["estimated_hard_max_new_run_cost_usd"] == pytest.approx(0.08)


def test_next_run_plan_reuses_typed_row_checkpoint_despite_code_provenance_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, corpus, model = _next_run_fixture(monkeypatch, tmp_path)
    _write_compatible_row_checkpoint(run_root, corpus, model)

    first = build_next_run_row_plan_report(
        run_root,
        corpus=corpus,
        project_root=tmp_path,
        extractor_model=model,
    )
    second = build_next_run_row_plan_report(
        run_root,
        corpus=corpus,
        project_root=tmp_path,
        extractor_model=model,
    )

    assert first == second
    row_reuse = first["papers"][0]["row_checkpoint_reuse"]
    assert row_reuse["status"] == "compatible"
    assert row_reuse["checkpoint_entries"] == 1
    assert row_reuse["reusable_checkpoint_batches"] == 1
    assert row_reuse["new_base_batches"] == 0
    assert first["next_run_rows"]["reusable_checkpoint_batches"] == 1
    assert first["next_run_rows"]["new_base_batches"] == 0
    preflight = first["next_run_preflight"]
    assert preflight["expected_logical_calls"] == 2
    assert preflight["hard_max_logical_calls"] == 2
    assert preflight["reused_legacy_block_calls"] == 1
    assert preflight["reused_row_base_batches"] == 1
    assert preflight["expected_new_structured_chat_invocations"] == 0
    assert preflight["hard_max_new_structured_chat_invocations"] == 0
    assert preflight["hard_max_new_transport_attempts"] == 0
    assert first["cost_estimates"]["estimated_expected_new_run_cost_usd"] == 0.0
    assert first["cost_estimates"]["estimated_hard_max_new_run_cost_usd"] == 0.0


def test_next_run_plan_treats_malformed_row_checkpoint_as_no_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, corpus, model = _next_run_fixture(monkeypatch, tmp_path)
    write_json(
        run_root / "fixture-paper" / "private" / "row-enumeration-checkpoint.json",
        {},
    )

    report = build_next_run_row_plan_report(
        run_root,
        corpus=corpus,
        project_root=tmp_path,
        extractor_model=model,
    )

    row_reuse = report["papers"][0]["row_checkpoint_reuse"]
    assert row_reuse["status"] == "malformed"
    assert row_reuse["reusable_checkpoint_batches"] == 0
    assert report["next_run_rows"]["new_base_batches"] == 1


def test_next_run_plan_rejects_tampered_row_checkpoint_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, corpus, model = _next_run_fixture(monkeypatch, tmp_path)
    checkpoint_path = _write_compatible_row_checkpoint(run_root, corpus, model)
    checkpoint = read_json(checkpoint_path)
    entry = next(iter(checkpoint["batches"].values()))
    entry["records"].pop(next(iter(entry["records"])))
    write_json(checkpoint_path, checkpoint)

    report = build_next_run_row_plan_report(
        run_root,
        corpus=corpus,
        project_root=tmp_path,
        extractor_model=model,
    )

    row_reuse = report["papers"][0]["row_checkpoint_reuse"]
    assert row_reuse["status"] == "compatible"
    assert row_reuse["checkpoint_entries"] == 1
    assert row_reuse["reusable_checkpoint_batches"] == 0
    assert report["next_run_rows"]["new_base_batches"] == 1


def test_plan_row_enumeration_cli_requires_corpus_and_model_for_exact_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, corpus, model = _next_run_fixture(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr("proceedings_to_eee.cli.load_corpus", lambda _: corpus)

    def fake_report(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "run_root": "fixture",
            "corpus_id": "fixture-corpus",
            "mode": "offline_current_code_frozen_source_preflight",
            "provider_or_network_calls": 0,
            "extractor_model": model,
            "extractor_contract": {},
            "row_extractor_contract": {},
            "row_config": {},
            "stored_artifact_comparison": {},
            "next_run_blocks": {},
            "row_plan": {},
            "next_run_preflight": {},
            "cost_estimate_status": {"available": False},
            "cost_estimates": None,
        }

    monkeypatch.setattr(
        "proceedings_to_eee.cli.build_next_run_row_plan_report",
        fake_report,
    )
    result = CliRunner().invoke(
        app,
        [
            "plan-row-enumeration",
            str(run_root),
            "--corpus",
            str(run_root / "fixture-paper" / "run.json"),
            "--model",
            model,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"] == (run_root.resolve(),)
    assert captured["kwargs"]["corpus"] is corpus
    assert captured["kwargs"]["extractor_model"] == model
    assert captured["kwargs"]["output_path"] is None


def test_plan_row_enumeration_cli_persists_corpus_and_code_bindings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, corpus, model = _next_run_fixture(monkeypatch, tmp_path)
    corpus_path = run_root / "fixture-paper" / "run.json"
    output = tmp_path / "preflight.json"
    monkeypatch.setattr("proceedings_to_eee.cli.load_corpus", lambda _: corpus)
    monkeypatch.setattr(
        "proceedings_to_eee.cli._code_state",
        lambda _: {
            "git_commit": "uncommitted",
            "git_dirty": True,
            "git_available": False,
            "source_tree_sha256": "a" * 64,
        },
    )

    result = CliRunner().invoke(
        app,
        [
            "plan-row-enumeration",
            str(run_root),
            "--corpus",
            str(corpus_path),
            "--model",
            model,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    persisted = read_json(output)
    assert persisted["corpus_file_sha256"] == sha256_file(corpus_path)
    assert persisted["corpus_binding"]["evaluation_split"] == "unspecified"
    assert persisted["code"]["source_tree_sha256"] == "a" * 64
