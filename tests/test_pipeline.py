from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
import yaml

from proceedings_to_eee.cli import _paper_subset
from proceedings_to_eee.corpus import CorpusSpec, PaperSpec
from proceedings_to_eee.domain.attribution import AttributionState, AttributionVerdict
from proceedings_to_eee.domain.observation import MetricSpec, ObservationScope, ReportedValue
from proceedings_to_eee.domain.status import ActorRole, ClaimType, EvidenceKind
from proceedings_to_eee.extraction.llm import extractor_request_contract
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.result_blocks import segment_page_result_blocks
from proceedings_to_eee.extraction.row_enumeration import RowEnumerationPlan
from proceedings_to_eee.io import read_json, write_json
from proceedings_to_eee.pipeline import (
    PipelineSettings,
    _code_state,
    _validated_row_checkpoint_entry,
    run_corpus,
    run_paper,
)
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderResponseValidationError,
    StructuredResponse,
    structured_request_contract,
)
from proceedings_to_eee.reference import (
    AnnotationCoverage,
    EvidencePurpose,
    PaperReference,
    ReferenceActor,
    ReferenceEvidence,
    ReferenceObservation,
)
from proceedings_to_eee.reporting.extraction_review_cards import (
    build_paper_extraction_review_card,
)
from proceedings_to_eee.sources.manifest import FrozenSource, SourceManifest, SourceRole
from proceedings_to_eee.verification.independent import verifier_request_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _paper(paper_id: str) -> PaperSpec:
    return PaperSpec(
        paper_id=paper_id,
        title=f"Title {paper_id}",
        year=2025,
        venue="ACM Test",
        pdf_url=f"https://example.org/{paper_id}.pdf",
        perspective_role="evaluated_system",
    )


def _settings(tmp_path: Path) -> PipelineSettings:
    return PipelineSettings(
        project_root=tmp_path,
        schema_path=PROJECT_ROOT / "schemas" / "eee-0.2.2" / "eval.schema.json",
        schema_sha256="088fed8029d42fb3a607aa67e1a05c39e425241b5cd90803705b37562f402f2a",
        output_root=tmp_path / "runs",
        model="extractor",
    )


def test_pipeline_settings_default_to_reproducible_extractor_seed(tmp_path: Path) -> None:
    assert _settings(tmp_path).seed == 7


def test_code_state_ignores_runtime_bytecode_but_hashes_semantic_sources(tmp_path: Path) -> None:
    source = tmp_path / "src" / "package" / "module.py"
    config = tmp_path / "configs" / "corpus.yaml"
    bytecode = tmp_path / "src" / "package" / "__pycache__" / "module.cpython-312.pyc"
    source.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    bytecode.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config.write_text("corpus: fixture\n", encoding="utf-8")
    bytecode.write_bytes(b"runtime-one")

    first = _code_state(tmp_path)["source_tree_sha256"]
    bytecode.write_bytes(b"runtime-two")
    second = _code_state(tmp_path)["source_tree_sha256"]
    source.write_text("VALUE = 2\n", encoding="utf-8")
    third = _code_state(tmp_path)["source_tree_sha256"]

    assert first == second
    assert third != second


def test_code_state_hashes_installed_package_when_checkout_is_absent(tmp_path: Path) -> None:
    state = _code_state(tmp_path)

    assert state["git_commit"] == "uncommitted"
    assert state["git_available"] is False
    assert state["source_tree_sha256"] != hashlib.sha256().hexdigest()
    assert state["source_tree_sha256"] == _code_state(tmp_path)["source_tree_sha256"]


def test_paper_subset_supports_checkpointed_single_paper_smoke() -> None:
    first, second = _paper("first"), _paper("second")
    corpus = CorpusSpec(
        corpus_id="full",
        evaluation_split="development",
        description="fixture",
        papers=[first, second],
    )

    selected = _paper_subset(corpus, "second")

    assert selected.corpus_id == "full--paper-second"
    assert selected.evaluation_split == "development"
    assert [paper.paper_id for paper in selected.papers] == ["second"]
    assert _paper_subset(corpus, None) is corpus
    with pytest.raises(typer.BadParameter, match="not present"):
        _paper_subset(corpus, "missing")


def _successful_summary(paper: PaperSpec) -> dict[str, object]:
    return {
        "schema_version": "pipeline-run/0.2",
        "status": "success",
        "paper_id": paper.paper_id,
        "title": paper.title,
        "counts": {
            "candidates": 2,
            "primary_results": 1,
            "exported": 1,
            "eee_records": 1,
            "eee_schema_issues": 0,
            "verifications": 1,
            "verifier_accepts": 1,
            "verifier_rejects": 0,
            "verifier_reviews": 0,
            "spot_checks": 1,
            "spot_checks_exact": 1,
        },
        "wall_clock_seconds": 0.1,
    }


def test_run_corpus_aggregates_errors_without_aborting(monkeypatch, tmp_path: Path) -> None:
    first, second = _paper("first"), _paper("second")
    corpus = CorpusSpec(
        corpus_id="resilient",
        description="fixture",
        papers=[first, second],
    )

    def fake_run_paper(*, spec, settings, client):
        del client
        if spec.paper_id == "second":
            source_bytes = b"frozen"
            source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            write_json(
                settings.output_root / spec.paper_id / "source-manifest.json",
                SourceManifest(
                    paper_id=spec.paper_id,
                    title=spec.title,
                    sources=[
                        FrozenSource(
                            source_id="src_second",
                            paper_id=spec.paper_id,
                            role=SourceRole.PAPER,
                            original_uri=str(spec.pdf_url),
                            resolved_uri=str(spec.pdf_url),
                            retrieved_at=datetime(2026, 8, 4, tzinfo=UTC),
                            sha256=source_sha256,
                            byte_size=len(source_bytes),
                            media_type="application/pdf",
                            cache_relpath="data/sources/frozen.pdf",
                        )
                    ],
                ),
            )
            fake_openrouter_token = "sk" + "-or-example"
            fake_auth_header = "Bear" + "er secret-token"
            raise RuntimeError(f"{fake_auth_header} {fake_openrouter_token} API key=also-secret")
        return _successful_summary(spec)

    monkeypatch.setattr("proceedings_to_eee.pipeline.run_paper", fake_run_paper)

    write_json(tmp_path / "runs" / "second" / "reference-score.json", {"stale": True})
    write_json(tmp_path / "runs" / "corpus-evaluation.json", {"stale": True})

    result = run_corpus(corpus=corpus, settings=_settings(tmp_path), client=object())

    assert result["status"] == "partial_failure"
    assert result["corpus_binding"]["evaluation_split"] == "unspecified"
    assert len(result["corpus_binding"]["corpus_spec_sha256"]) == 64
    assert len(result["corpus_binding"]["paper_ids_sha256"]) == 64
    assert result["papers_succeeded"] == 1
    assert result["papers_failed"] == 1
    assert result["papers_with_eee"] == 1
    assert result["reference_evaluation"] is None
    assert result["totals"]["eee_records"] == 1
    assert not (tmp_path / "runs" / "corpus-evaluation.json").exists()
    error = result["runs"][1]["error"]
    assert "secret-token" not in error["message"]
    assert "sk" + "-or-example" not in error["message"]
    assert "also-secret" not in error["message"]
    assert "[REDACTED]" in error["message"]

    saved = json.loads((tmp_path / "runs" / "corpus-run.json").read_text())
    assert saved["papers_failed"] == 1
    failed_run = json.loads((tmp_path / "runs" / "second" / "run.json").read_text())
    assert failed_run["status"] == "error"
    assert (tmp_path / "runs" / "second" / "observations.jsonl").read_text() == ""
    assert (tmp_path / "runs" / "second" / "verifications.jsonl").read_text() == ""
    assert json.loads((tmp_path / "runs" / "second" / "spot-checks.json").read_text()) == []
    assert failed_run["extractor"]["seed"] == 7
    assert failed_run["extractor"]["request_contract"] == extractor_request_contract(seed=7)
    assert failed_run["verifier"]["enabled"] is False
    assert failed_run["verifier"]["seed"] == 7
    assert failed_run["verifier"]["request_contract"] == verifier_request_contract()
    assert failed_run["eee_schema"] == {
        "version": "0.2.2",
        "sha256": "088fed8029d42fb3a607aa67e1a05c39e425241b5cd90803705b37562f402f2a",
    }
    assert failed_run["extractor"]["execution"]["blocks_total"] == 0
    card = build_paper_extraction_review_card(
        tmp_path / "runs" / "second",
        split="development",
    )
    assert card["processing"]["status"] == "error"
    assert card["abstention"]["primary_reason"] == "paper_run_error"
    report = (tmp_path / "runs" / "corpus-review.html").read_text()
    assert "resilient · 2 papers" in report
    assert "Run failed" in report


class _EndToEndClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        self.requests.append(kwargs)
        payload = {
            "observations": [
                {
                    "claim_type": "primary_result",
                    "roles": [
                        {
                            "role": "evaluated_system",
                            "raw_name": "System A",
                            "version": None,
                            "provider": None,
                            "confidence": 0.99,
                        }
                    ],
                    "scope": {
                        "dataset_raw": "Dataset A",
                        "dataset_id": None,
                        "dataset_url": None,
                        "dataset_version": None,
                        "split": None,
                        "subset": None,
                        "group": None,
                        "language": None,
                        "sample_count": None,
                        "aggregation": None,
                        "raw_scope": None,
                    },
                    "metric": {
                        "raw_name": "AUC",
                        "canonical_id": None,
                        "kind": None,
                        "unit": "proportion",
                        "lower_is_better": None,
                        "min_score": None,
                        "max_score": None,
                        "parameters": {},
                    },
                    "value": {
                        "raw": "0.80",
                        "numeric": 0.8,
                        "unit": "proportion",
                        "comparator": "exact",
                        "uncertainty": None,
                    },
                    "evidence": [
                        {
                            "kind": "table",
                            "label": "Table 1",
                            "row": "System A",
                            "column": "AUC",
                            "quote": "System A                 0.80",
                        }
                    ],
                    "extraction_confidence": 0.99,
                    "construct": None,
                    "operationalization": None,
                    "decision_rule": None,
                    "evaluation_date": None,
                    "notes": [],
                }
            ],
            "page_summary": "one result",
            "warnings": [],
        }
        digest = hashlib.sha256(kwargs["user"].encode()).hexdigest()
        contract = structured_request_contract(
            schema_name=kwargs["schema_name"],
            schema=kwargs["schema"],
            seed=kwargs.get("seed"),
            require_parameters=kwargs.get("require_parameters", False),
        )
        schema_contract = contract["schema"]
        return StructuredResponse(
            payload=payload,
            call=ProviderCall(
                model_requested=kwargs["model"],
                model_returned=kwargs["model"],
                provider_returned="fixture",
                prompt_sha256=digest,
                response_sha256=hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()
                ).hexdigest(),
                temperature=kwargs["temperature"],
                reasoning_effort=kwargs["reasoning_effort"],
                max_tokens=kwargs["max_tokens"],
                seed=contract["seed"],
                response_format=schema_contract["response_format"],
                schema_name=schema_contract["schema_name"],
                schema_sha256=schema_contract["schema_sha256"],
                schema_strict=schema_contract["schema_strict"],
                latency_seconds=0.01,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost_usd=0.001,
                finish_reason="stop",
                attempts=1,
            ),
        )


class _NeverCallClient:
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        del kwargs
        raise AssertionError("a compatible successful checkpoint must not call the provider")


class _EmptyEndToEndClient(_EndToEndClient):
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        return replace(
            response,
            payload={
                "observations": [],
                "page_summary": "no extractable observations",
                "warnings": [],
            },
        )


class _EndToEndRowClient(_EndToEndClient):
    """Return one row result and one valid zero-candidate abstention."""

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] != "paper_table_row_dispositions":
            return response
        row_ids = re.findall(r'"row_id": "(trow_[0-9a-f]+)"', kwargs["user"])
        assert len(row_ids) == 2
        observation = response.payload["observations"][0]
        payload = {
            "dispositions": [
                {
                    "row_id": row_ids[0],
                    "disposition": "result",
                    "observations": [observation],
                    "note": None,
                },
                {
                    "row_id": row_ids[1],
                    "disposition": "not_result",
                    "observations": [],
                    "note": "descriptive fixture row",
                },
            ],
            "warnings": [],
        }
        return replace(
            response,
            payload=payload,
            call=response.call.model_copy(
                update={
                    "response_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True).encode()
                    ).hexdigest()
                }
            ),
        )


class _FailFirstValidationClient:
    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self._delegate = _EndToEndClient()
        self.fail_on = fail_on or {1}

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        self.requests.append(kwargs)
        response = self._delegate.structured_chat(**kwargs)
        if len(self.requests) in self.fail_on:
            raise ProviderResponseValidationError(call=response.call, code="invalid_json")
        return response


def _end_to_end_fixture(monkeypatch, tmp_path: Path) -> tuple[PaperSpec, PipelineSettings]:
    paper_id = "end-to-end-paper"
    source_bytes = b"%PDF-end-to-end-fixture"
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_id = "src_end_to_end"
    cache_path = tmp_path / "data" / "sources" / source_sha256[:2] / f"{source_sha256}.pdf"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(source_bytes)
    output_root = tmp_path / "runs" / "pilot"
    manifest = SourceManifest(
        paper_id=paper_id,
        title="End-to-end Paper",
        sources=[
            FrozenSource(
                source_id=source_id,
                paper_id=paper_id,
                role=SourceRole.PAPER,
                original_uri="https://example.org/paper.pdf",
                resolved_uri="https://example.org/paper.pdf",
                retrieved_at=datetime(2026, 8, 4, tzinfo=UTC),
                sha256=source_sha256,
                byte_size=len(source_bytes),
                media_type="application/pdf",
                cache_relpath=cache_path.relative_to(tmp_path).as_posix(),
            )
        ],
    )
    write_json(output_root / paper_id / "source-manifest.json", manifest)
    page_text = """Results
Table 1: AUC on Dataset A
System                    AUC
System A                 0.80
System B                 0.70
"""
    layout = PdfLayout(
        source_id=source_id,
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[
            PageFragment(
                fragment_id="page-1",
                source_id=source_id,
                page=1,
                text=page_text,
                text_sha256=hashlib.sha256(page_text.encode()).hexdigest(),
                character_count=len(page_text),
                numeric_token_count=2,
                result_signal_score=10.0,
            )
        ],
    )
    monkeypatch.setattr(
        "proceedings_to_eee.pipeline.extract_pdf_layout", lambda path, source: layout
    )
    reference = PaperReference(
        paper_id=paper_id,
        source_sha256=source_sha256,
        annotation_protocol="fixture/0.1",
        annotation_status="checked",
        coverage=AnnotationCoverage(
            fully_annotated_labels=["Table 1"],
            inclusion_rule="The only result cell.",
            exclusion_rule="Headers are context.",
        ),
        evidence=[
            ReferenceEvidence(
                evidence_id="result",
                purpose=EvidencePurpose.RESULT,
                page=1,
                kind=EvidenceKind.TABLE,
                label="Table 1",
                row="System A",
                column="AUC",
                exact_quote="System A                 0.80",
            )
        ],
        observations=[
            ReferenceObservation(
                reference_id="ref-result",
                claim_type=ClaimType.PRIMARY_RESULT,
                actors=[ReferenceActor(role=ActorRole.EVALUATED_SYSTEM, raw_name="System A")],
                scope=ObservationScope(dataset_raw="Dataset A"),
                metric=MetricSpec(
                    raw_name="AUC",
                    canonical_id="auroc",
                    unit="proportion",
                    lower_is_better=False,
                ),
                value=ReportedValue(raw="0.80", numeric=0.8, unit="proportion"),
                result_evidence_ids=["result"],
                expected_missing_fields=["evaluated_system.version", "evaluation_date"],
            )
        ],
    )
    reference_path = tmp_path / "references" / "paper.yaml"
    reference_path.parent.mkdir()
    reference_path.write_text(
        yaml.safe_dump(reference.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    spec = PaperSpec(
        paper_id=paper_id,
        title="End-to-end Paper",
        year=2026,
        venue="ACM Test",
        pdf_url="https://example.org/paper.pdf",
        perspective_role="evaluated_system",
        include_pages=[1],
        reference_path="references/paper.yaml",
    )
    settings = PipelineSettings(
        project_root=tmp_path,
        schema_path=PROJECT_ROOT / "schemas" / "eee-0.2.2" / "eval.schema.json",
        schema_sha256="088fed8029d42fb3a607aa67e1a05c39e425241b5cd90803705b37562f402f2a",
        output_root=output_root,
        model="fixture/model",
        seed=19,
    )

    return spec, settings


def _trust_current_paper_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install an explicit positive-origin fixture; production v0 never infers this."""

    monkeypatch.setattr(
        "proceedings_to_eee.validation.candidates.attribute_candidate",
        lambda *args, **kwargs: AttributionVerdict(
            state=AttributionState.PAPER_PRODUCED,
            rule_id="explicit_pipeline_test_fixture",
        ),
    )


def test_run_paper_with_explicit_trusted_origin_writes_valid_eee_and_reference_score(
    monkeypatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    _trust_current_paper_origin(monkeypatch)
    output_root = settings.output_root
    paper_id = spec.paper_id

    client = _EndToEndClient()
    result = run_paper(spec=spec, settings=settings, client=client)

    assert result["status"] == "success"
    assert result["counts"]["candidates"] == 1
    assert result["counts"]["exported"] == 1
    assert result["counts"]["eee_records"] == 1
    assert result["counts"]["eee_schema_issues"] == 0
    assert result["counts"]["reference_true_positives"] == 1
    assert result["review_state"] == {"status": "ready", "reasons": []}
    assert len(client.requests) == 1
    assert client.requests[0]["seed"] == 19
    assert client.requests[0]["require_parameters"] is False
    assert result["extractor"]["seed"] == 19
    assert result["extractor"]["request_contract"] == extractor_request_contract(seed=19)
    assert result["extractor"]["request_contract"]["schema"] == {
        "response_format": "json_schema",
        "schema_name": "paper_evaluation_candidates",
        "schema_sha256": result["extractor"]["calls"][0]["schema_sha256"],
        "schema_strict": True,
    }
    assert result["extractor"]["successful_call_telemetry"] == {
        "basis": (
            "successful final block calls; cost, token, retry, and attempt totals are lower "
            "bounds when provider metadata or superseded/failed attempts are unavailable"
        ),
        "calls": 1,
        "cost_usd_lower_bound": 0.001,
        "cost_reported_calls": 1,
        "input_tokens_lower_bound": 100,
        "input_tokens_reported_calls": 1,
        "output_tokens_lower_bound": 50,
        "output_tokens_reported_calls": 1,
        "total_tokens_lower_bound": 150,
        "total_tokens_reported_calls": 1,
        "latency_seconds_total": 0.01,
        "latency_seconds_mean": 0.01,
        "latency_seconds_max": 0.01,
        "attempts_lower_bound": 1,
        "retries_lower_bound": 0,
    }
    assert result["verifier"]["enabled"] is False
    assert result["verifier"]["seed"] == 7
    assert result["verifier"]["request_contract"] == verifier_request_contract()
    eee_files = list((output_root / paper_id / "eee").glob("*.json"))
    assert len(eee_files) == 1
    record = json.loads(eee_files[0].read_text())
    assert record["evaluation_results"][0]["score_details"]["score"] == 0.8
    assert result["result_block_segmentation"] == {
        "max_lines": 40,
        "max_characters": 6_000,
        "context_lines": 8,
        "trailing_context_lines": 2,
        "overlap_lines": 3,
        "signal_gap_lines": 3,
        "max_blank_gap": 1,
        "max_data_rows": 6,
        "min_signal_score": 1.5,
        "max_blocks_per_page": 6,
        "detect_parallel_columns": True,
        "min_column_gutter_width": 3,
        "min_parallel_lines": 4,
        "max_column_analysis_width": 240,
    }
    assert (output_root / paper_id / "reference-score.json").exists()
    assert (output_root / paper_id / "review.html").exists()


def test_default_v0_keeps_no_signal_candidate_in_review_without_canonical_eee(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)

    result = run_paper(spec=spec, settings=settings, client=_EndToEndClient())

    assert result["status"] == "success"
    assert result["counts"]["candidates"] == 1
    assert result["counts"]["exported"] == 0
    assert result["counts"]["eee_records"] == 0
    # Reference scoring remains candidate-based; the export gate does not erase recall.
    assert result["counts"]["reference_true_positives"] == 1
    assert result["review_state"] == {
        "status": "needs_review",
        "reasons": ["zero_valid_eee_records", "candidate_review_required"],
    }
    observations = (settings.output_root / spec.paper_id / "observations.jsonl").read_text()
    assert '"state":"no_signal"' in observations
    assert not list((settings.output_root / spec.paper_id / "eee").glob("*.json"))


def test_opt_in_row_stage_is_bounded_deduplicated_and_checkpointed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(
        settings,
        row_enumeration_enabled=True,
        row_estimated_call_cost_usd=0.001,
    )
    client = _EndToEndRowClient()

    first = run_paper(spec=spec, settings=settings, client=client)

    assert len(client.requests) == 2
    assert first["row_enumeration"]["plan"] == {
        "tables_considered": 1,
        "dense_tables": 1,
        "rows_planned": 2,
        "unbatchable_rows": 0,
        "base_batches": 1,
        "expected_calls": 1,
        "maximum_calls": 3,
    }
    assert "historical mean" in first["row_enumeration"]["preflight"]["estimated_cost_basis"]
    assert first["row_enumeration"]["preflight"]["baseline_block_calls"] == 1
    assert first["row_enumeration"]["preflight"]["expected_total_calls"] == 2
    assert first["row_enumeration"]["preflight"]["maximum_total_calls"] == 4
    assert first["row_enumeration"]["preflight"]["estimated_row_cost_usd"] == 0.001
    assert first["row_enumeration"]["outcome"]["dispositions"] == {
        "result": 1,
        "not_result": 1,
        "uncertain": 0,
    }
    assert first["counts"]["candidates_before_deduplication"] == 2
    assert first["counts"]["duplicates_removed"] == 1
    assert first["counts"]["candidates"] == 1
    private = settings.output_root / spec.paper_id / "private"
    assert (private / "row-enumeration-plan.json").is_file()
    assert (private / "row-enumeration-preflight.json").is_file()
    assert (private / "row-enumeration-checkpoint.json").is_file()
    assert (private / "row-enumeration.json").is_file()

    plan = RowEnumerationPlan.model_validate(read_json(private / "row-enumeration-plan.json"))
    checkpoint = read_json(private / "row-enumeration-checkpoint.json")
    batch = plan.batches[0]
    entry = checkpoint["batches"][batch.batch_id]
    assert _validated_row_checkpoint_entry(entry, batch=batch) is not None
    incomplete_entry = json.loads(json.dumps(entry))
    incomplete_entry["records"].pop(batch.rows[-1].row_id)
    assert _validated_row_checkpoint_entry(incomplete_entry, batch=batch) is None

    second = run_paper(spec=spec, settings=settings, client=_NeverCallClient())

    assert second["extractor"]["execution"]["blocks_resumed"] == 1
    assert second["row_enumeration"]["execution"]["batches_resumed"] == 1
    assert second["counts"]["duplicates_removed"] == 1

    disabled = run_paper(
        spec=spec,
        settings=replace(settings, row_enumeration_enabled=False),
        client=_NeverCallClient(),
    )

    assert disabled["row_enumeration"]["enabled"] is False
    assert not (private / "row-enumeration-plan.json").exists()
    assert not (private / "row-enumeration-preflight.json").exists()
    assert not (private / "row-enumeration.json").exists()
    assert (private / "row-enumeration-checkpoint.json").is_file()


def test_row_checkpoint_survives_unrelated_code_state_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(settings, row_enumeration_enabled=True)
    source_hash = ["a" * 64]

    monkeypatch.setattr(
        "proceedings_to_eee.pipeline._code_state",
        lambda _: {
            "git_commit": "fixture",
            "git_dirty": True,
            "git_available": True,
            "source_tree_sha256": source_hash[0],
        },
    )
    first = run_paper(spec=spec, settings=settings, client=_EndToEndRowClient())
    source_hash[0] = "b" * 64

    second = run_paper(spec=spec, settings=settings, client=_NeverCallClient())

    assert (
        first["row_enumeration"]["checkpoint"]["contract_sha256"]
        != (second["row_enumeration"]["checkpoint"]["contract_sha256"])
    )
    assert second["extractor"]["execution"]["blocks_resumed"] == 1
    assert second["row_enumeration"]["execution"] == {
        "batches_total": 1,
        "batches_resumed": 1,
        "batches_executed": 0,
        "invalid_rows_seen": 0,
        "unknown_row_ids_seen": 0,
    }
    checkpoint = read_json(
        settings.output_root / spec.paper_id / "private" / "row-enumeration-checkpoint.json"
    )
    assert checkpoint["contract"]["code"]["source_tree_sha256"] == "b" * 64


def test_row_checkpoint_migration_rejects_a_tampered_typed_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(settings, row_enumeration_enabled=True)
    source_hash = ["a" * 64]
    monkeypatch.setattr(
        "proceedings_to_eee.pipeline._code_state",
        lambda _: {
            "git_commit": "fixture",
            "git_dirty": True,
            "git_available": True,
            "source_tree_sha256": source_hash[0],
        },
    )
    run_paper(spec=spec, settings=settings, client=_EndToEndRowClient())
    checkpoint_path = (
        settings.output_root / spec.paper_id / "private" / "row-enumeration-checkpoint.json"
    )
    checkpoint = read_json(checkpoint_path)
    entry = next(iter(checkpoint["batches"].values()))
    entry["records"].pop(next(iter(entry["records"])))
    write_json(checkpoint_path, checkpoint)
    source_hash[0] = "b" * 64
    client = _EndToEndRowClient()

    resumed = run_paper(spec=spec, settings=settings, client=client)

    assert len(client.requests) == 1
    assert client.requests[0]["schema_name"] == "paper_table_row_dispositions"
    assert resumed["extractor"]["execution"]["blocks_resumed"] == 1
    assert resumed["row_enumeration"]["execution"] == {
        "batches_total": 1,
        "batches_resumed": 0,
        "batches_executed": 1,
        "invalid_rows_seen": 0,
        "unknown_row_ids_seen": 0,
    }


def test_zero_candidate_and_zero_eee_are_explicit_review_states(
    monkeypatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)

    result = run_paper(spec=spec, settings=settings, client=_EmptyEndToEndClient())

    assert result["status"] == "success"
    assert result["counts"]["candidates"] == 0
    assert result["counts"]["eee_records"] == 0
    assert result["review_state"] == {
        "status": "needs_review",
        "reasons": [
            "selected_result_blocks_produced_zero_candidates",
            "zero_valid_eee_records",
        ],
    }
    assert "paper_review_required=zero_valid_eee_records" in result["warnings"]


def test_paper_error_retains_frozen_reference_denominator(monkeypatch, tmp_path: Path) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    corpus = CorpusSpec(corpus_id="error-denominator", description="fixture", papers=[spec])

    def fail_after_source_freeze(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise RuntimeError("fatal deterministic composition failure")

    monkeypatch.setattr("proceedings_to_eee.pipeline.run_paper", fail_after_source_freeze)

    result = run_corpus(corpus=corpus, settings=settings, client=object())

    assert result["status"] == "error"
    assert result["reference_evaluation"]["papers_scored"] == 1
    assert result["reference_evaluation"]["detection"]["true_positives"] == 0
    assert result["reference_evaluation"]["detection"]["false_negatives"] == 1
    assert result["reference_evaluation"]["detection"]["recall"] == 0.0
    failed = result["runs"][0]
    assert failed["counts"]["reference_false_negatives"] == 1
    assert failed["reference_evaluation"]["score_sha256"]


def test_successful_blocks_resume_from_contract_bound_private_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    _trust_current_paper_origin(monkeypatch)

    first = run_paper(spec=spec, settings=settings, client=_EndToEndClient())
    checkpoint_path = settings.output_root / spec.paper_id / "private" / "extractor-checkpoint.json"
    checkpoint_text = checkpoint_path.read_text(encoding="utf-8")

    assert first["extractor"]["execution"] == {
        "blocks_total": 1,
        "blocks_succeeded": 1,
        "blocks_failed": 0,
        "blocks_resumed": 0,
        "calls_succeeded": 1,
        "calls_failed": 0,
        "calls_resumed": 0,
    }
    assert "page_summary" not in checkpoint_text
    assert "one result" not in checkpoint_text

    second = run_paper(spec=spec, settings=settings, client=_NeverCallClient())

    assert second["status"] == "success"
    assert second["counts"]["eee_records"] == 1
    assert second["extractor"]["calls"] == []
    assert len(second["extractor"]["resumed_calls"]) == 1
    assert second["extractor"]["execution"] == {
        "blocks_total": 1,
        "blocks_succeeded": 0,
        "blocks_failed": 0,
        "blocks_resumed": 1,
        "calls_succeeded": 0,
        "calls_failed": 0,
        "calls_resumed": 1,
    }
    assert second["extractor"]["successful_call_telemetry"]["calls"] == 1
    assert second["extractor"]["successful_call_telemetry"]["cost_usd_lower_bound"] == 0.001
    assert second["extractor"]["successful_call_telemetry"]["total_tokens_lower_bound"] == 150


def test_failed_block_isolated_completed_call_preserved_and_retried(
    monkeypatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)

    def two_blocks(page: PageFragment, *, config):
        blocks = segment_page_result_blocks(page, config=config)
        assert len(blocks) == 1
        second = blocks[0].model_copy(
            update={"block_id": f"{blocks[0].block_id}-second", "page_ordinal": 2}
        )
        return [blocks[0], second]

    monkeypatch.setattr(
        "proceedings_to_eee.pipeline.segment_page_result_blocks",
        two_blocks,
    )
    monkeypatch.setattr(
        "proceedings_to_eee.pipeline._recover_split_block",
        lambda **kwargs: None,
    )
    paper_dir = settings.output_root / spec.paper_id
    write_json(paper_dir / "eee" / "stale.json", {"stale": True})
    write_json(paper_dir / "reference-score.json", {"stale": True})
    (paper_dir / "review.html").write_text("STALE REVIEW", encoding="utf-8")

    failing_client = _FailFirstValidationClient()
    first = run_paper(spec=spec, settings=settings, client=failing_client)

    assert first["status"] == "partial_failure"
    assert len(failing_client.requests) == 2
    assert first["extractor"]["execution"] == {
        "blocks_total": 2,
        "blocks_succeeded": 1,
        "blocks_failed": 1,
        "blocks_resumed": 0,
        "calls_succeeded": 1,
        "calls_failed": 1,
        "calls_resumed": 0,
    }
    assert len(first["extractor"]["calls"]) == 2
    failed_attempt = first["extractor"]["block_attempts"][0]
    assert failed_attempt["status"] == "failed"
    assert failed_attempt["error_code"] == "provider_response_invalid_json"
    assert failed_attempt["completed_provider_call"] is True
    assert "OpenRouter response" not in json.dumps(first)
    checkpoint = json.loads(
        (paper_dir / "private" / "extractor-checkpoint.json").read_text(encoding="utf-8")
    )
    assert list(checkpoint["blocks"]) == [first["selected_blocks"][1]["block_id"]]
    assert not (paper_dir / "eee" / "stale.json").exists()
    assert "STALE REVIEW" not in (paper_dir / "review.html").read_text(encoding="utf-8")
    assert json.loads((paper_dir / "reference-score.json").read_text()) != {"stale": True}

    retry_client = _EndToEndClient()
    second = run_paper(spec=spec, settings=settings, client=retry_client)

    assert second["status"] == "success"
    assert len(retry_client.requests) == 1
    assert second["extractor"]["execution"]["blocks_succeeded"] == 1
    assert second["extractor"]["execution"]["blocks_resumed"] == 1
    assert second["extractor"]["execution"]["blocks_failed"] == 0


def test_recursive_recovery_retains_failed_call_telemetry_and_checkpoints_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    client = _FailFirstValidationClient()

    result = run_paper(spec=spec, settings=settings, client=client)

    assert result["status"] == "success"
    assert len(client.requests) == 3
    assert len(result["extractor"]["calls"]) == 3
    assert result["extractor"]["successful_call_telemetry"]["calls"] == 2
    attempt = result["extractor"]["block_attempts"][0]
    assert attempt["status"] == "recovered_by_split"
    assert attempt["recovery_calls"] == 2
    assert attempt["recovery_successful_calls"] == 2
    assert attempt["recovery_validation_failed_calls"] == 0
    assert attempt["recovery_max_depth_reached"] == 1
    assert attempt["recovery_terminal_failures"] == []
    checkpoint = read_json(
        settings.output_root / spec.paper_id / "private" / "extractor-checkpoint.json"
    )
    assert list(checkpoint["blocks"]) == [result["selected_blocks"][0]["block_id"]]


def test_terminal_recovery_preserves_successful_sibling_candidates_and_safe_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    client = _FailFirstValidationClient(fail_on={1, 3})

    result = run_paper(spec=spec, settings=settings, client=client)

    assert result["status"] == "partial_failure"
    assert len(client.requests) == 3
    assert len(result["extractor"]["calls"]) == 3
    assert result["extractor"]["successful_call_telemetry"]["calls"] == 1
    assert result["counts"]["candidates"] == 1
    attempt = result["extractor"]["block_attempts"][0]
    assert attempt["status"] == "failed"
    assert attempt["recovery_calls"] == 2
    assert attempt["recovery_successful_calls"] == 1
    assert attempt["recovery_validation_failed_calls"] == 1
    assert attempt["recovery_max_depth_reached"] == 1
    assert attempt["recovery_terminal_failures"] == [
        {
            "block_id": f"{result['selected_blocks'][0]['block_id']}_s2",
            "page": 1,
            "depth": 1,
            "error_code": "provider_response_invalid_json",
            "completed_provider_call": True,
            "terminal_reason": "unsplittable",
            "safe_details": {},
        }
    ]
    assert "OpenRouter response" not in json.dumps(result)
    checkpoint = read_json(
        settings.output_root / spec.paper_id / "private" / "extractor-checkpoint.json"
    )
    assert checkpoint["blocks"] == {}


def test_checkpoint_is_not_reused_when_extractor_contract_changes(
    monkeypatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    first = run_paper(spec=spec, settings=settings, client=_EndToEndClient())

    changed_client = _EndToEndClient()
    second = run_paper(
        spec=spec,
        settings=replace(settings, seed=settings.seed + 1),
        client=changed_client,
    )

    assert len(changed_client.requests) == 1
    assert second["extractor"]["execution"]["blocks_succeeded"] == 1
    assert second["extractor"]["execution"]["blocks_resumed"] == 0
    assert (
        first["extractor"]["checkpoint"]["contract_sha256"]
        != second["extractor"]["checkpoint"]["contract_sha256"]
    )


def test_exact_block_checkpoint_survives_unrelated_segmentation_contract_change(
    monkeypatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    first = run_paper(spec=spec, settings=settings, client=_EndToEndClient())

    second = run_paper(
        spec=spec,
        settings=replace(settings, max_blocks_per_page=settings.max_blocks_per_page - 1),
        client=_NeverCallClient(),
    )

    assert (
        first["extractor"]["checkpoint"]["contract_sha256"]
        != second["extractor"]["checkpoint"]["contract_sha256"]
    )
    assert second["extractor"]["execution"]["blocks_resumed"] == 1
    assert second["extractor"]["execution"]["blocks_succeeded"] == 0


def test_corpus_uses_hash_bound_partial_failure_score_and_rejects_tampering(
    monkeypatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    paper_summary = run_paper(spec=spec, settings=settings, client=_EndToEndClient())
    paper_summary = {**paper_summary, "status": "partial_failure"}
    corpus = CorpusSpec(
        corpus_id="hash-bound-reference-score",
        description="fixture",
        papers=[spec],
    )
    monkeypatch.setattr(
        "proceedings_to_eee.pipeline.run_paper",
        lambda **kwargs: paper_summary,
    )

    current = run_corpus(corpus=corpus, settings=settings, client=object())

    assert current["reference_evaluation"]["papers_scored"] == 1

    write_json(settings.output_root / spec.paper_id / "reference-score.json", {"stale": True})
    tampered = run_corpus(corpus=corpus, settings=settings, client=object())

    assert tampered["reference_evaluation"] is None
    assert not (settings.output_root / "corpus-evaluation.json").exists()


def test_review_omits_schema_invalid_projection(monkeypatch, tmp_path: Path) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    _trust_current_paper_origin(monkeypatch)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "proceedings_to_eee.pipeline.validate_eee_record",
        lambda record, schema: [SimpleNamespace(path="$", message="forced invalid")],
    )

    def capture_review_report(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "proceedings_to_eee.pipeline.render_review_report",
        capture_review_report,
    )

    result = run_paper(spec=spec, settings=settings, client=_EndToEndClient())

    paper_dir = settings.output_root / spec.paper_id
    assert result["status"] == "quality_failure"
    assert result["counts"]["eee_records"] == 0
    assert result["counts"]["eee_schema_issues"] == 1
    assert result["review_state"] == {
        "status": "needs_review",
        "reasons": [
            "zero_valid_eee_records",
            "eee_schema_validation_failure",
            "candidate_review_required",
        ],
    }
    assert result["counts"]["candidates_needing_review"] == 1
    assert result["counts"]["semantic_safety_reviews"] == 0
    assert captured["eee_records"] == []
    assert len(captured["validation_errors"]) == 1
    assert not list((paper_dir / "eee").glob("*.json"))
    assert (paper_dir / "private" / "invalid-eee.json").exists()
