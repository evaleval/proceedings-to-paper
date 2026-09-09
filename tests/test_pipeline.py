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
from proceedings_to_eee.evaluation.control_coverage import (
    control_examination as compute_control_examination,
)
from proceedings_to_eee.evaluation.control_coverage import (
    observation_examination as compute_observation_examination,
)
from proceedings_to_eee.extraction.llm import extractor_request_contract
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.result_blocks import segment_page_result_blocks
from proceedings_to_eee.extraction.row_enumeration import RowEnumerationPlan
from proceedings_to_eee.io import canonical_json_bytes, read_json, sha256_bytes, write_json
from proceedings_to_eee.pipeline import (
    PipelineSettings,
    _bounded_paper_summary,
    _candidate_validation_run_configuration,
    _code_state,
    _corpus_operational_metrics,
    _extractor_run_configuration,
    _origin_run_configuration,
    _row_enumeration_run_configuration,
    _tuple_run_configuration,
    _validated_row_checkpoint_entry,
    _verifier_run_configuration,
    run_corpus,
    run_paper,
)
from proceedings_to_eee.providers.budget import (
    BudgetedProviderClient,
    ProviderBudgetExhausted,
    ProviderBudgetLimits,
    provider_budget_contract,
)
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderRequestRejectedError,
    ProviderResponseValidationError,
    StructuredResponse,
    completion_token_parameter_for_model,
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
from proceedings_to_eee.verification.independent import (
    VERIFIER_SCHEMA_NAME,
    verifier_request_contract,
)

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
        model="fixture/extractor",
    )


def test_pipeline_settings_default_to_common_capability_controls(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.temperature is None
    assert settings.reasoning_effort == "minimal"
    assert settings.seed is None


def test_all_enabled_stage_manifests_bind_materialized_openai_token_field(
    tmp_path: Path,
) -> None:
    settings = replace(
        _settings(tmp_path),
        model="openai/gpt-5.5",
        row_enumeration_enabled=True,
        row_model="openai/gpt-5.5",
        tuple_model="openai/gpt-5.5",
        verifier_model="openai/gpt-5.5",
        origin_model="openai/gpt-5.5",
    )
    stages = (
        (_extractor_run_configuration(settings), settings.max_tokens),
        (_row_enumeration_run_configuration(settings), settings.max_tokens),
        (_tuple_run_configuration(settings), settings.tuple_max_tokens),
        (_verifier_run_configuration(settings), settings.verifier_max_tokens),
        (_origin_run_configuration(settings), settings.origin_max_tokens),
    )

    for stage, max_tokens in stages:
        contract = stage["request_contract"]
        assert contract["schema_version"] == "provider-request-contract/0.2"
        assert contract["max_tokens"] == max_tokens
        assert contract["completion_token_parameter"] == "max_completion_tokens"


def test_candidate_validation_policy_is_versioned_and_budget_bound(tmp_path: Path) -> None:
    settings = replace(_settings(tmp_path), min_confidence=0.61)

    policy = _candidate_validation_run_configuration(settings)
    assert policy == {
        "schema_version": "candidate-validation/0.2",
        "min_confidence": 0.61,
        "origin_policy": "positive_only",
    }

    limits = ProviderBudgetLimits(
        max_structured_calls=2,
        max_cost_usd=2.0,
        cost_reservation_per_call_usd=0.5,
    )
    corpus_binding = {
        "schema_version": "fixture/0.1",
        "corpus_id": "fixture",
    }
    first = provider_budget_contract(
        corpus_binding=corpus_binding,
        provider_run_contract={"candidate_validation": policy},
        limits=limits,
    )
    second = provider_budget_contract(
        corpus_binding=corpus_binding,
        provider_run_contract={
            "candidate_validation": _candidate_validation_run_configuration(
                replace(settings, min_confidence=0.8)
            )
        },
        limits=limits,
    )
    assert first["provider_run_contract_sha256"] != second["provider_run_contract_sha256"]


def test_bounded_paper_manifest_persists_candidate_validation_policy(tmp_path: Path) -> None:
    settings = replace(_settings(tmp_path), min_confidence=0.61)
    error = ProviderBudgetExhausted(
        reason="structured_call_limit",
        summary={"structured_calls_started": 2},
    )

    summary = _bounded_paper_summary(
        paper=_paper("bounded"),
        settings=settings,
        schema_version="0.2.2",
        schema_sha256="a" * 64,
        code_state={"git_commit": "uncommitted", "git_dirty": False, "git_available": False},
        block_config={"max_blocks_per_page": 6},
        error=error,
        started=0.0,
    )

    assert summary["candidate_validation"] == {
        "schema_version": "candidate-validation/0.2",
        "min_confidence": 0.61,
        "origin_policy": "positive_only",
    }


def test_corpus_operations_preserve_stage_usage_completeness_denominators() -> None:
    telemetry = {
        "calls": 2,
        "cost_usd_lower_bound": 0.3,
        "cost_reported_calls": 1,
        "input_tokens_lower_bound": 11,
        "input_tokens_reported_calls": 1,
        "output_tokens_lower_bound": 7,
        "output_tokens_reported_calls": 2,
        "total_tokens_lower_bound": 18,
        "total_tokens_reported_calls": 1,
        "latency_seconds_total": 0.6,
        "latency_seconds_max": 0.4,
        "attempts_lower_bound": 3,
        "retries_lower_bound": 1,
    }
    summary = {
        "extractor": {
            "successful_call_telemetry": telemetry,
            "execution": {},
        },
        "row_enumeration": {
            "successful_call_telemetry": telemetry,
            "execution": {},
            "plan": {},
        },
        "tuple_resolution": {
            "completed_call_telemetry": telemetry,
            "execution": {
                "candidates_selected": 3,
                "candidates_passed": 2,
                "candidates_routed_to_review": 1,
                "candidates_resumed": 1,
            },
        },
        "verifier": {
            "completed_call_telemetry": telemetry,
            "execution": {
                "candidates_verified": 2,
                "candidates_failed": 1,
                "candidates_resumed": 1,
                "candidates_executed": 1,
            },
        },
        "origin_retrieval": {
            "completed_call_telemetry": telemetry,
            "execution": {
                "candidates_selected": 2,
                "candidates_resumed": 1,
                "candidates_failed": 0,
            },
        },
    }

    operations = _corpus_operational_metrics([summary], wall_clock_seconds=1.0)

    for stage_name in ("row_enumeration", "tuple_resolution", "verifier", "origin_retrieval"):
        stage = operations[stage_name]
        assert stage["calls"] == 2
        assert stage["cost_reported_calls"] == 1
        assert stage["input_tokens_reported_calls"] == 1
        assert stage["output_tokens_reported_calls"] == 2
        assert stage["total_tokens_reported_calls"] == 1
        assert stage["attempts_lower_bound"] == 3
        assert stage["retries_lower_bound"] == 1
    assert operations["verifier"]["candidates_verified"] == 2
    assert operations["verifier"]["candidates_failed"] == 1
    assert operations["verifier"]["candidates_resumed"] == 1
    assert operations["verifier"]["candidates_executed"] == 1


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
    assert error["message"] == "runtime operation failed"

    saved = json.loads((tmp_path / "runs" / "corpus-run.json").read_text())
    assert saved["papers_failed"] == 1
    failed_run = json.loads((tmp_path / "runs" / "second" / "run.json").read_text())
    assert failed_run["status"] == "error"
    assert failed_run["candidate_validation"] == {
        "schema_version": "candidate-validation/0.2",
        "min_confidence": 0.8,
        "origin_policy": "positive_only",
    }
    assert (tmp_path / "runs" / "second" / "observations.jsonl").read_text() == ""
    assert (tmp_path / "runs" / "second" / "verifications.jsonl").read_text() == ""
    assert json.loads((tmp_path / "runs" / "second" / "spot-checks.json").read_text()) == []
    assert failed_run["extractor"]["seed"] is None
    assert failed_run["extractor"]["request_contract"] == extractor_request_contract(
        seed=None,
        model=_settings(tmp_path).model,
        max_tokens=_settings(tmp_path).max_tokens,
    )
    assert failed_run["verifier"]["enabled"] is False
    assert failed_run["verifier"]["seed"] is None
    assert failed_run["verifier"]["temperature"] is None
    assert failed_run["verifier"]["reasoning_effort"] == "minimal"
    assert failed_run["verifier"]["require_parameters"] is True
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


def test_paper_error_preserves_completed_checkpointed_call_accounting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paper = _paper("paid-then-failed")
    corpus = CorpusSpec(corpus_id="paid-error", description="fixture", papers=[paper])
    settings = _settings(tmp_path)

    def fail_after_checkpointed_call(*, spec: PaperSpec, settings: PipelineSettings, client: Any):
        response = client.structured_chat(
            model=settings.model,
            system="fixture system",
            user="fixture user",
            schema_name="fixture_error_accounting",
            schema={"type": "object", "properties": {}, "additionalProperties": False},
            temperature=settings.temperature,
            reasoning_effort=settings.reasoning_effort,
            max_tokens=settings.max_tokens,
            seed=settings.seed,
            require_parameters=False,
        )
        contract = {"paper_id": spec.paper_id, "stage": "verifier"}
        entry = {
            "schema_version": "independent-verifier-checkpoint-entry/0.4",
            "status": "success",
            "provider_call": response.call.model_dump(mode="json"),
        }
        entry["entry_sha256"] = sha256_bytes(canonical_json_bytes(entry))
        write_json(
            settings.output_root / spec.paper_id / "private" / "verifier-checkpoint.json",
            {
                "schema_version": "independent-verifier-checkpoint/0.4",
                "contract": contract,
                "contract_sha256": sha256_bytes(canonical_json_bytes(contract)),
                "candidates": {"candidate": entry},
            },
        )
        raise RuntimeError("failure after durable paid call")

    monkeypatch.setattr("proceedings_to_eee.pipeline.run_paper", fail_after_checkpointed_call)

    result = run_corpus(corpus=corpus, settings=settings, client=_EndToEndClient())

    failed = result["runs"][0]
    assert failed["status"] == "error"
    assert failed["provider_budget"]["structured_calls_completed"] == 1
    assert failed["failure_provider_accounting"]["checkpointed_completed_calls"] == 1
    assert failed["failure_provider_accounting"]["checkpoint_parse_complete"] is True
    assert failed["verifier"]["completed_call_telemetry"]["calls"] == 1
    assert failed["verifier"]["completed_call_telemetry"]["cost_usd_lower_bound"] == 0.001
    assert len(failed["verifier"]["calls"]) == 1
    assert "request_id" not in failed["verifier"]["calls"][0]
    assert result["operations"]["verifier"]["calls"] == 1
    assert result["operations"]["verifier"]["cost_usd_lower_bound"] == 0.001
    assert result["provider_budget"]["structured_calls_completed"] == 1


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
                        "unit": None,
                        "lower_is_better": None,
                        "min_score": None,
                        "max_score": None,
                        "parameters": {},
                    },
                    "value": {
                        "raw": "0.80",
                        "numeric": 0.8,
                        "unit": None,
                        "comparator": "exact",
                        "uncertainty": None,
                    },
                    "evidence": [
                        {
                            "kind": "table",
                            "label": "Table 1",
                            "row": "System A (Dataset A AUC)",
                            "column": "AUC",
                            "quote": "System A (Dataset A AUC)  0.80",
                        },
                        {
                            "kind": "table",
                            "label": "Table 1",
                            "row": None,
                            "column": "AUC",
                            "quote": "Table 1: AUC on Dataset A",
                        },
                        {
                            "kind": "prose",
                            "label": None,
                            "row": None,
                            "column": None,
                            "quote": "System A (Dataset A AUC)  0.80",
                        },
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
        digest = hashlib.sha256(
            json.dumps(
                [
                    {"role": "system", "content": kwargs["system"]},
                    {"role": "user", "content": kwargs["user"]},
                ],
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
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
                completion_token_parameter=completion_token_parameter_for_model(kwargs["model"]),
                seed=contract["seed"],
                response_format=schema_contract["response_format"],
                schema_name=schema_contract["schema_name"],
                schema_sha256=schema_contract["schema_sha256"],
                schema_strict=schema_contract["schema_strict"],
                require_parameters=kwargs.get("require_parameters", False),
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
        observation = {
            **observation,
            # Row extraction is constrained to exact text from this one row; the
            # structural row/header plan supplies the surrounding table context.
            "evidence": [{**observation["evidence"][0], "kind": "table"}],
        }
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


class _NoCallRowFailureClient(_EndToEndClient):
    """Fail every row attempt before a completed provider response exists."""

    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        if kwargs["schema_name"] != "paper_table_row_dispositions":
            return super().structured_chat(**kwargs)
        self.requests.append(kwargs)
        if self.failure == "request_rejected":
            raise ProviderRequestRejectedError(status_code=400)
        raise RuntimeError("fixture transport failure")


class _RecoveringRowClient(_EndToEndClient):
    """Force one two-row base miss followed by valid singleton recoveries."""

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] != "paper_table_row_dispositions":
            return response
        row_ids = re.findall(r'"row_id": "(trow_[0-9a-f]+)"', kwargs["user"])
        assert row_ids
        payload = {
            "dispositions": (
                []
                if len(row_ids) > 1
                else [
                    {
                        "row_id": row_ids[0],
                        "disposition": "not_result",
                        "observations": [],
                        "note": "singleton recovery",
                    }
                ]
            ),
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


class _ResponseFailureRecoveringRowClient(_EndToEndClient):
    """Return one paid invalid base response followed by valid singleton recoveries."""

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] != "paper_table_row_dispositions":
            return response
        row_ids = re.findall(r'"row_id": "(trow_[0-9a-f]+)"', kwargs["user"])
        assert row_ids
        payload: dict[str, Any] = (
            {"dispositions": "invalid", "warnings": []}
            if len(row_ids) > 1
            else {
                "dispositions": [
                    {
                        "row_id": row_ids[0],
                        "disposition": "not_result",
                        "observations": [],
                        "note": "singleton recovery",
                    }
                ],
                "warnings": [],
            }
        )
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


class _TrailingNoCallRecoveringRowClient(_RecoveringRowClient):
    """Complete a base and first child, then fail the final child before a call."""

    def __init__(self) -> None:
        super().__init__()
        self.row_requests = 0

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        if kwargs["schema_name"] == "paper_table_row_dispositions":
            self.row_requests += 1
            if self.row_requests == 3:
                self.requests.append(kwargs)
                raise RuntimeError("fixture transport failure")
        return super().structured_chat(**kwargs)


class _VerifierAcceptClient(_EndToEndClient):
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] != VERIFIER_SCHEMA_NAME:
            return response
        raw = (
            kwargs["user"]
            .split("<VERIFICATION_INPUT>\n", 1)[1]
            .split("\n</VERIFICATION_INPUT>", 1)[0]
        )
        verifier_input = json.loads(raw)
        candidate = verifier_input["candidate_claim_untrusted"]
        anchor = verifier_input["candidate_claimed_anchor_untrusted"]
        lines = verifier_input["trusted_frozen_source_block"]["lines"]

        def evidence_ids(*claims: object) -> list[str]:
            selected: list[str] = []
            for claim in claims:
                if claim is None or claim == "":
                    continue
                match = next(
                    (
                        line["line_id"]
                        for line in lines
                        if str(claim).casefold() in line["text"].casefold()
                    ),
                    None,
                )
                if match is not None and match not in selected:
                    selected.append(match)
            return selected

        scope = candidate["scope"] or {}
        metric = candidate["metric"] or {}
        value = candidate["value"] or {}
        payload = {
            "support": "supported",
            "support_evidence_line_ids": evidence_ids(anchor["quote"]),
            "role": "supported",
            "role_evidence_line_ids": evidence_ids(
                *(role["raw_name"] for role in candidate["roles"])
            ),
            "scope": "supported",
            "scope_evidence_line_ids": evidence_ids(
                scope.get("dataset_raw"),
                scope.get("dataset_version"),
                scope.get("split"),
                scope.get("subset"),
                scope.get("group"),
                scope.get("language"),
                scope.get("sample_count"),
                scope.get("aggregation"),
                scope.get("raw_scope"),
            ),
            "value": "supported",
            "value_evidence_line_ids": evidence_ids(value.get("raw"), value.get("unit")),
            "metric": "supported",
            "metric_evidence_line_ids": evidence_ids(
                metric.get("raw_name"),
                metric.get("unit"),
                *(metric.get("parameters") or {}).values(),
            ),
            "decision": "accept",
            "justification": "Every candidate field is supported by the frozen block.",
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


def _budgeted_pipeline_client(
    tmp_path: Path,
    raw_client: Any,
    *,
    calls: int,
) -> BudgetedProviderClient:
    limits = ProviderBudgetLimits(
        max_structured_calls=calls,
        max_cost_usd=10.0,
        cost_reservation_per_call_usd=0.01,
    )
    contract = provider_budget_contract(
        corpus_binding={
            "schema_version": "pipeline-budget-fixture/0.1",
            "corpus_id": "pipeline-budget-fixture",
            "evaluation_split": "development",
            "corpus_spec_sha256": "a" * 64,
            "paper_ids_sha256": "b" * 64,
        },
        provider_run_contract={"model": "fixture/model", "seed": 19},
        limits=limits,
    )
    return BudgetedProviderClient(
        client=raw_client,
        ledger_path=tmp_path / "private" / "provider-budget-ledger.jsonl",
        contract=contract,
        limits=limits,
    )


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
System A (Dataset A AUC)  0.80
System B (Dataset A AUC)  0.70
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
                row="System A (Dataset A AUC)",
                column="AUC",
                exact_quote="System A (Dataset A AUC)  0.80",
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


def test_programmatic_origin_configuration_fails_before_source_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(settings, origin_model="fixture/origin", verifier_model=None)
    source_work_started = False

    def forbidden_freeze(*args: Any, **kwargs: Any) -> Any:
        nonlocal source_work_started
        source_work_started = True
        raise AssertionError("source work must not start")

    monkeypatch.setattr("proceedings_to_eee.pipeline.freeze_paper", forbidden_freeze)

    with pytest.raises(ValueError, match="origin_model requires verifier_model"):
        run_paper(spec=spec, settings=settings, client=_NeverCallClient())

    assert source_work_started is False


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
    settings = replace(settings, min_confidence=0.61)
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
    assert client.requests[0]["require_parameters"] is True
    assert result["extractor"]["seed"] == 19
    assert result["extractor"]["request_contract"] == extractor_request_contract(
        seed=19,
        model=settings.model,
        max_tokens=settings.max_tokens,
    )
    assert result["candidate_validation"] == {
        "schema_version": "candidate-validation/0.2",
        "min_confidence": 0.61,
        "origin_policy": "positive_only",
    }
    assert read_json(output_root / paper_id / "run.json")["candidate_validation"] == {
        "schema_version": "candidate-validation/0.2",
        "min_confidence": 0.61,
        "origin_policy": "positive_only",
    }
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
        "reasoning_tokens_lower_bound": 0,
        "reasoning_tokens_reported_calls": 0,
        "total_tokens_lower_bound": 150,
        "total_tokens_reported_calls": 1,
        "latency_seconds_total": 0.01,
        "latency_seconds_mean": 0.01,
        "latency_seconds_max": 0.01,
        "attempts_lower_bound": 1,
        "retries_lower_bound": 0,
    }
    assert result["verifier"]["enabled"] is False
    assert result["verifier"]["seed"] is None
    assert result["verifier"]["temperature"] is None
    assert result["verifier"]["reasoning_effort"] == "minimal"
    assert result["verifier"]["require_parameters"] is True
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


def test_verifier_requires_tuple_gate_before_any_source_or_provider_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(settings, verifier_model="fixture/verifier")
    client = _VerifierAcceptClient()
    source_work_started = False

    def forbidden_freeze(*args: Any, **kwargs: Any) -> Any:
        nonlocal source_work_started
        source_work_started = True
        raise AssertionError("source work must not start")

    monkeypatch.setattr("proceedings_to_eee.pipeline.freeze_paper", forbidden_freeze)

    with pytest.raises(ValueError, match="verifier_model requires tuple_model"):
        run_paper(spec=spec, settings=settings, client=client)

    assert source_work_started is False
    assert client.requests == []


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        ({"model": ""}, "extractor model is required"),
        (
            {"row_enumeration_enabled": True, "row_model": 123},
            "row model must be non-empty when supplied",
        ),
        ({"tuple_model": []}, "tuple model must be non-empty when supplied"),
        (
            {"tuple_model": "fixture/tuple", "verifier_model": "  "},
            "verifier model must be non-empty when supplied",
        ),
        (
            {
                "tuple_model": "fixture/tuple",
                "verifier_model": "fixture/verifier",
                "origin_model": 123,
            },
            "origin model must be non-empty when supplied",
        ),
    ],
)
def test_all_enabled_stage_models_are_strictly_validated_before_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, Any],
    expected_error: str,
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(settings, **overrides)
    client = _EndToEndClient()
    source_work_started = False

    def forbidden_freeze(*args: Any, **kwargs: Any) -> Any:
        nonlocal source_work_started
        source_work_started = True
        raise AssertionError("source work must not start")

    monkeypatch.setattr("proceedings_to_eee.pipeline.freeze_paper", forbidden_freeze)

    with pytest.raises(ValueError, match=expected_error):
        run_paper(spec=spec, settings=settings, client=client)

    assert source_work_started is False
    assert client.requests == []


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
        seed=None,
        row_enumeration_enabled=True,
        row_model="fixture/row-model",
        row_estimated_call_cost_usd=0.001,
    )
    client = _EndToEndRowClient()

    first = run_paper(spec=spec, settings=settings, client=client)

    assert len(client.requests) == 2
    assert [request["model"] for request in client.requests] == [
        "fixture/model",
        "fixture/row-model",
    ]
    assert first["row_enumeration"]["model"] == "fixture/row-model"
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
    assert first["row_enumeration"]["successful_call_telemetry"]["calls"] == 1
    assert first["row_enumeration"]["completed_call_telemetry"]["calls"] == 1
    assert first["counts"]["candidates_before_deduplication"] == 2
    assert first["counts"]["duplicates_removed"] == 1
    assert first["counts"]["candidates"] == 1
    private = settings.output_root / spec.paper_id / "private"
    assert (private / "row-enumeration-plan.json").is_file()
    assert (private / "row-enumeration-preflight.json").is_file()
    assert (private / "row-enumeration-checkpoint.json").is_file()
    assert (private / "row-enumeration.json").is_file()
    assert (private / "row-terminal-states.json").is_file()
    lineage_path = settings.output_root / spec.paper_id / "candidate-lineage.json"
    lineage = read_json(lineage_path)
    assert lineage["schema_version"] == "candidate-lineage/0.1"
    assert lineage["counts"] == {
        "proposals": 2,
        "candidate_occurrences": 2,
        "final_candidates": 1,
        "singleton_candidates": 0,
        "merged_candidates": 1,
        "exported": 0,
        "needs_review": 1,
        "not_eligible": 0,
        "invalid_eee": 0,
        "eligible_no_eee": 0,
    }
    assert lineage["candidates"][0]["merge_kind"] == "physical_cell"
    assert '"quote":' not in lineage_path.read_text(encoding="utf-8")
    assert (settings.output_root / spec.paper_id / "source-processing.json").is_file()
    assert first["row_enumeration"]["terminal_states"]["counts"] == {
        "planned": 2,
        "result": 1,
        "not_result": 1,
        "uncertain": 0,
        "unresolved": 0,
        "unsupported": 0,
    }
    operational = read_json(private / "row-enumeration.json")
    assert operational["schema_version"] == "row-enumeration-outcome/0.3"
    assert "records" not in operational

    plan = RowEnumerationPlan.model_validate(read_json(private / "row-enumeration-plan.json"))
    checkpoint = read_json(private / "row-enumeration-checkpoint.json")
    batch = plan.batches[0]
    entry = checkpoint["batches"][batch.batch_id]
    private_call = entry["calls"][0]
    assert private_call["temperature"] is None
    assert private_call["seed"] is None
    assert private_call["reasoning_tokens"] is None
    assert (
        _validated_row_checkpoint_entry(
            entry,
            batch=batch,
            contract=checkpoint["contract"],
        )
        is not None
    )
    incomplete_entry = json.loads(json.dumps(entry))
    incomplete_entry["records"].pop(batch.rows[-1].row_id)
    assert (
        _validated_row_checkpoint_entry(
            incomplete_entry,
            batch=batch,
            contract=checkpoint["contract"],
        )
        is None
    )
    poisoned_entry = json.loads(json.dumps(entry))
    result_row_id = next(
        row_id
        for row_id, record in poisoned_entry["records"].items()
        if record["disposition"] == "result"
    )
    sibling = next(row for row in batch.rows if row.row_id != result_row_id)
    poisoned_candidate = poisoned_entry["records"][result_row_id]["candidates"][0]
    poisoned_candidate["evidence"][0]["planned_row_id"] = sibling.row_id
    poisoned_candidate["evidence"][0]["region_id"] = sibling.region_id
    assert (
        _validated_row_checkpoint_entry(
            poisoned_entry,
            batch=batch,
            contract=checkpoint["contract"],
        )
        is None
    )
    telemetry_drift = json.loads(json.dumps(entry))
    telemetry_drift["calls"] = []
    assert (
        _validated_row_checkpoint_entry(
            telemetry_drift,
            batch=batch,
            contract=checkpoint["contract"],
        )
        is None
    )

    second = run_paper(spec=spec, settings=settings, client=_NeverCallClient())

    assert second["extractor"]["execution"]["blocks_resumed"] == 1
    assert second["row_enumeration"]["execution"]["batches_resumed"] == 1
    assert second["counts"]["duplicates_removed"] == 1
    resumed_public_call = second["row_enumeration"]["calls"][0]
    assert resumed_public_call["temperature"] is None
    assert resumed_public_call["seed"] is None
    assert resumed_public_call["reasoning_tokens"] is None

    disabled = run_paper(
        spec=spec,
        settings=replace(settings, row_enumeration_enabled=False, row_model=None),
        client=_NeverCallClient(),
    )

    assert disabled["row_enumeration"]["enabled"] is False
    assert not (private / "row-enumeration-plan.json").exists()
    assert not (private / "row-enumeration-preflight.json").exists()
    assert not (private / "row-enumeration.json").exists()
    assert (private / "row-enumeration-checkpoint.json").is_file()


def test_row_manifest_separates_response_valid_from_all_completed_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(
        settings,
        row_enumeration_enabled=True,
        row_model="fixture/row-model",
        row_estimated_call_cost_usd=0.001,
    )
    client = _ResponseFailureRecoveringRowClient()

    result = run_paper(spec=spec, settings=settings, client=client)

    row = result["row_enumeration"]
    assert [attempt["status"] for attempt in row["attempts"]] == [
        "provider_response_wire_validation",
        "success",
        "success",
    ]
    assert len(row["calls"]) == 3
    assert row["successful_call_telemetry"]["calls"] == 2
    assert row["successful_call_telemetry"]["cost_usd_lower_bound"] == 0.002
    assert row["completed_call_telemetry"]["calls"] == 3
    assert row["completed_call_telemetry"]["cost_usd_lower_bound"] == 0.003

    resumed = run_paper(spec=spec, settings=settings, client=_NeverCallClient())
    assert resumed["row_enumeration"]["execution"]["batches_resumed"] == 1


@pytest.mark.parametrize(
    ("failure", "ledger_outcome"),
    [
        ("transport", "technical_failure"),
        ("request_rejected", "provider_request_rejected"),
    ],
)
def test_no_call_row_failures_are_typed_then_retried_without_resetting_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    ledger_outcome: str,
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(
        settings,
        row_enumeration_enabled=True,
        row_model="fixture/row-model",
        row_estimated_call_cost_usd=0.001,
    )
    first_raw = _NoCallRowFailureClient(failure)
    first_client = _budgeted_pipeline_client(tmp_path, first_raw, calls=5)

    first = run_paper(spec=spec, settings=settings, client=first_client)

    assert first["status"] == "partial_failure"
    assert [attempt["status"] for attempt in first["row_enumeration"]["attempts"]] == [
        "provider_request_failed",
        "provider_request_failed",
        "provider_request_failed",
    ]
    assert first["row_enumeration"]["completed_call_telemetry"]["calls"] == 0
    assert first_client.summary["structured_calls_started"] == 4
    assert first_client.summary["completion_outcomes"][ledger_outcome] == 3
    checkpoint = read_json(
        settings.output_root / spec.paper_id / "private" / "row-enumeration-checkpoint.json"
    )
    entry = next(iter(checkpoint["batches"].values()))
    assert entry["calls"] == []
    assert all(not attempt["completed_provider_call"] for attempt in entry["attempts"])

    resumed_raw = _EndToEndRowClient()
    resumed_client = _budgeted_pipeline_client(tmp_path, resumed_raw, calls=5)
    resumed = run_paper(spec=spec, settings=settings, client=resumed_client)

    assert [request["schema_name"] for request in resumed_raw.requests] == [
        "paper_table_row_dispositions"
    ]
    assert resumed["extractor"]["execution"]["blocks_resumed"] == 1
    assert resumed["row_enumeration"]["execution"]["batches_resumed"] == 0
    assert resumed["row_enumeration"]["outcome"]["rows_unresolved"] == 0
    assert resumed_client.summary["structured_calls_started"] == 5
    assert resumed_client.summary["structured_calls_failed"] == 3


def test_trailing_no_call_row_failure_preserves_completed_checkpoint_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(
        settings,
        row_enumeration_enabled=True,
        row_model="fixture/row-model",
        row_estimated_call_cost_usd=0.001,
    )
    first_client = _TrailingNoCallRecoveringRowClient()

    first = run_paper(spec=spec, settings=settings, client=first_client)

    assert first["status"] == "partial_failure"
    assert [attempt["status"] for attempt in first["row_enumeration"]["attempts"]] == [
        "partial_invalid",
        "success",
        "provider_request_failed",
    ]
    completed_calls = first["row_enumeration"]["calls"]
    assert len(completed_calls) == 2

    resumed_client = _RecoveringRowClient()
    resumed = run_paper(spec=spec, settings=settings, client=resumed_client)

    row_requests = [
        request
        for request in resumed_client.requests
        if request["schema_name"] == "paper_table_row_dispositions"
    ]
    assert len(row_requests) == 1
    assert resumed["extractor"]["execution"]["blocks_resumed"] == 1
    assert resumed["row_enumeration"]["execution"]["batches_resumed"] == 0
    assert [attempt["status"] for attempt in resumed["row_enumeration"]["attempts"]] == [
        "partial_invalid",
        "success",
        "success",
    ]
    assert resumed["row_enumeration"]["calls"][:2] == completed_calls
    assert resumed["row_enumeration"]["outcome"]["rows_unresolved"] == 0


def test_budget_stop_mid_row_recovery_resumes_without_repeating_charged_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(
        settings,
        row_enumeration_enabled=True,
        row_model="fixture/row-model",
        row_estimated_call_cost_usd=0.001,
    )
    first_raw = _RecoveringRowClient()
    first_client = _budgeted_pipeline_client(tmp_path, first_raw, calls=3)

    with pytest.raises(ProviderBudgetExhausted, match="structured_call_limit"):
        run_paper(spec=spec, settings=settings, client=first_client)

    assert [request["schema_name"] for request in first_raw.requests] == [
        "paper_evaluation_candidates",
        "paper_table_row_dispositions",
        "paper_table_row_dispositions",
    ]
    assert first_client.summary["structured_calls_started"] == 3
    private = settings.output_root / spec.paper_id / "private"
    interrupted = read_json(private / "row-enumeration-checkpoint.json")
    batch_id = next(iter(interrupted["batches"]))
    partial = interrupted["batches"][batch_id]
    assert [attempt["depth"] for attempt in partial["attempts"]] == [0, 1]
    assert len(partial["calls"]) == 2
    assert len(partial["unresolved_row_ids"]) == 1

    resumed_raw = _RecoveringRowClient()
    resumed_client = _budgeted_pipeline_client(tmp_path, resumed_raw, calls=4)
    resumed = run_paper(spec=spec, settings=settings, client=resumed_client)

    assert len(resumed_raw.requests) == 1
    assert resumed_raw.requests[0]["schema_name"] == "paper_table_row_dispositions"
    assert resumed_client.summary["structured_calls_started"] == 4
    assert resumed["extractor"]["execution"]["blocks_resumed"] == 1
    assert resumed["row_enumeration"]["execution"]["batches_resumed"] == 0
    assert [attempt["depth"] for attempt in resumed["row_enumeration"]["attempts"]] == [
        0,
        1,
        1,
    ]
    assert resumed["row_enumeration"]["outcome"]["dispositions"] == {
        "result": 0,
        "not_result": 2,
        "uncertain": 0,
    }
    completed = read_json(private / "row-enumeration-checkpoint.json")
    assert len(completed["batches"][batch_id]["calls"]) == 3

    events = [
        json.loads(line)
        for line in resumed_client.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    request_hashes = [
        event["request"]["request_sha256"]
        for event in events
        if event["event_type"] == "reservation"
    ]
    assert len(request_hashes) == len(set(request_hashes)) == 4

    final_client = _budgeted_pipeline_client(tmp_path, _NeverCallClient(), calls=4)
    final = run_paper(spec=spec, settings=settings, client=final_client)
    assert final["row_enumeration"]["execution"]["batches_resumed"] == 1
    assert final_client.summary["structured_calls_started"] == 4


def test_extractor_and_row_checkpoints_fail_closed_on_code_state_change(
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

    second_client = _EndToEndRowClient()
    second = run_paper(spec=spec, settings=settings, client=second_client)

    assert (
        first["row_enumeration"]["checkpoint"]["contract_sha256"]
        != (second["row_enumeration"]["checkpoint"]["contract_sha256"])
    )
    assert [request["schema_name"] for request in second_client.requests] == [
        "paper_evaluation_candidates",
        "paper_table_row_dispositions",
    ]
    assert second["extractor"]["execution"]["blocks_resumed"] == 0
    assert second["extractor"]["execution"]["blocks_succeeded"] == 1
    assert second["row_enumeration"]["execution"] == {
        "batches_total": 1,
        "batches_resumed": 0,
        "batches_executed": 1,
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


def test_row_checkpoint_rejects_returned_model_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(settings, row_enumeration_enabled=True)
    run_paper(spec=spec, settings=settings, client=_EndToEndRowClient())
    checkpoint_path = (
        settings.output_root / spec.paper_id / "private" / "row-enumeration-checkpoint.json"
    )
    checkpoint = read_json(checkpoint_path)
    entry = next(iter(checkpoint["batches"].values()))
    entry["calls"][0]["model_returned"] = "fixture/wrong-model"
    write_json(checkpoint_path, checkpoint)
    client = _EndToEndRowClient()

    resumed = run_paper(spec=spec, settings=settings, client=client)

    assert [request["schema_name"] for request in client.requests] == [
        "paper_table_row_dispositions"
    ]
    assert resumed["row_enumeration"]["execution"]["batches_resumed"] == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_requested", "fixture/other-model"),
        ("model_returned", "fixture/other-model"),
        ("model_returned", None),
        ("prompt_sha256", "a" * 64),
        ("max_tokens", 15_999),
        ("schema_sha256", "b" * 64),
        ("data_collection", "allow"),
        ("require_parameters", False),
        ("zdr", False),
    ],
)
def test_extractor_checkpoint_rejects_mutated_exact_call_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    replacement: Any,
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_EndToEndClient())
    checkpoint_path = settings.output_root / spec.paper_id / "private" / "extractor-checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    entry = next(iter(checkpoint["blocks"].values()))
    entry["calls"][0][field] = replacement
    entry["attempts"][0]["call"][field] = replacement
    write_json(checkpoint_path, checkpoint)
    client = _EndToEndClient()

    resumed = run_paper(spec=spec, settings=settings, client=client)

    assert len(client.requests) == 1
    assert resumed["extractor"]["execution"]["blocks_resumed"] == 0
    assert resumed["extractor"]["execution"]["blocks_succeeded"] == 1


@pytest.mark.parametrize("mutation", ["extraction_method", "evidence_page"])
def test_extractor_checkpoint_rejects_mutated_candidate_block_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_EndToEndClient())
    checkpoint_path = settings.output_root / spec.paper_id / "private" / "extractor-checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    entry = next(iter(checkpoint["blocks"].values()))
    for candidate in (entry["candidates"][0], entry["attempts"][0]["candidates"][0]):
        if mutation == "extraction_method":
            candidate["extraction_method"] = "openrouter:fixture/other-model"
        else:
            candidate["evidence"][0]["page"] = 2
    write_json(checkpoint_path, checkpoint)
    client = _EndToEndClient()

    resumed = run_paper(spec=spec, settings=settings, client=client)

    assert len(client.requests) == 1
    assert resumed["extractor"]["execution"]["blocks_resumed"] == 0


def test_extractor_checkpoint_rejects_mutated_attempt_status_and_success_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_EndToEndClient())
    checkpoint_path = settings.output_root / spec.paper_id / "private" / "extractor-checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    entry = next(iter(checkpoint["blocks"].values()))
    entry["attempts"][0]["status"] = "provider_response_wire_validation"
    entry["successful_call_indexes"] = []
    write_json(checkpoint_path, checkpoint)
    client = _EndToEndClient()

    resumed = run_paper(spec=spec, settings=settings, client=client)

    assert len(client.requests) == 1
    assert resumed["extractor"]["execution"]["blocks_resumed"] == 0


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
    settings = replace(settings, seed=None)
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
        "calls_resumed_succeeded": 0,
        "calls_resumed_failed": 0,
        "no_call_failures": 0,
        "requests_rejected": 0,
        "transport_failures": 0,
        "local_failures": 0,
    }
    assert "page_summary" not in checkpoint_text
    assert "one result" not in checkpoint_text
    checkpoint = read_json(checkpoint_path)
    private_call = next(iter(checkpoint["blocks"].values()))["calls"][0]
    assert private_call["temperature"] is None
    assert private_call["seed"] is None
    assert private_call["reasoning_tokens"] is None

    second = run_paper(spec=spec, settings=settings, client=_NeverCallClient())

    assert second["status"] == "success"
    assert second["counts"]["eee_records"] == 1
    assert second["extractor"]["calls"] == []
    assert len(second["extractor"]["resumed_calls"]) == 1
    resumed_public_call = second["extractor"]["resumed_calls"][0]
    assert resumed_public_call["temperature"] is None
    assert resumed_public_call["seed"] is None
    assert resumed_public_call["reasoning_tokens"] is None
    assert second["extractor"]["execution"] == {
        "blocks_total": 1,
        "blocks_succeeded": 0,
        "blocks_failed": 0,
        "blocks_resumed": 1,
        "calls_succeeded": 0,
        "calls_failed": 0,
        "calls_resumed": 1,
        "calls_resumed_succeeded": 1,
        "calls_resumed_failed": 0,
        "no_call_failures": 0,
        "requests_rejected": 0,
        "transport_failures": 0,
        "local_failures": 0,
    }
    assert second["extractor"]["successful_call_telemetry"]["calls"] == 1
    assert second["extractor"]["successful_call_telemetry"]["cost_usd_lower_bound"] == 0.001
    assert second["extractor"]["successful_call_telemetry"]["total_tokens_lower_bound"] == 150


def test_failed_block_isolated_completed_call_preserved_and_retried(
    monkeypatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    control_block_ids: list[list[str]] = []
    observation_block_ids: list[list[str]] = []

    def tracked_control_examination(reference, layout, examined_blocks):
        control_block_ids.append([block.block_id for block in examined_blocks])
        return compute_control_examination(reference, layout, examined_blocks)

    def tracked_observation_examination(reference, layout, examined_blocks):
        observation_block_ids.append([block.block_id for block in examined_blocks])
        return compute_observation_examination(reference, layout, examined_blocks)

    monkeypatch.setattr(
        "proceedings_to_eee.pipeline.control_examination",
        tracked_control_examination,
    )
    monkeypatch.setattr(
        "proceedings_to_eee.pipeline.observation_examination",
        tracked_observation_examination,
    )

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
        "calls_resumed_succeeded": 0,
        "calls_resumed_failed": 0,
        "no_call_failures": 0,
        "requests_rejected": 0,
        "transport_failures": 0,
        "local_failures": 0,
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
    assert control_block_ids == [[first["selected_blocks"][1]["block_id"]]]
    assert observation_block_ids == [[first["selected_blocks"][1]["block_id"]]]
    reference_score = read_json(paper_dir / "reference-score.json")
    assert reference_score["input_observability"]["status"] == "measured"
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
    assert result["extractor"]["completed_call_telemetry"]["calls"] == 3
    assert result["extractor"]["execution"]["blocks_succeeded"] == 1
    assert result["extractor"]["execution"]["blocks_failed"] == 0
    assert result["extractor"]["execution"]["calls_succeeded"] == 2
    assert result["extractor"]["execution"]["calls_failed"] == 1
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


def test_recovered_extractor_checkpoint_revalidates_the_failed_parent_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_FailFirstValidationClient())
    checkpoint_path = settings.output_root / spec.paper_id / "private" / "extractor-checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    entry = next(iter(checkpoint["blocks"].values()))
    assert [attempt["status"] for attempt in entry["attempts"]] == [
        "provider_response_invalid_json",
        "success",
        "success",
    ]
    entry["calls"][0]["prompt_sha256"] = "a" * 64
    entry["attempts"][0]["call"]["prompt_sha256"] = "a" * 64
    write_json(checkpoint_path, checkpoint)
    client = _FailFirstValidationClient()

    resumed = run_paper(spec=spec, settings=settings, client=client)

    assert len(client.requests) == 3
    assert resumed["extractor"]["execution"]["blocks_resumed"] == 0
    assert resumed["extractor"]["execution"]["blocks_succeeded"] == 1


def test_budget_stop_mid_legacy_split_resumes_without_repeating_charged_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    first_raw = _FailFirstValidationClient()
    first_client = _budgeted_pipeline_client(tmp_path, first_raw, calls=2)

    with pytest.raises(ProviderBudgetExhausted, match="structured_call_limit"):
        run_paper(spec=spec, settings=settings, client=first_client)

    assert len(first_raw.requests) == 2
    assert first_client.summary["structured_calls_started"] == 2
    checkpoint_path = settings.output_root / spec.paper_id / "private" / "extractor-checkpoint.json"
    interrupted = read_json(checkpoint_path)
    root_id = next(iter(interrupted["recoveries"]))
    assert interrupted["blocks"] == {}
    assert [item["depth"] for item in interrupted["recoveries"][root_id]["attempts"]] == [
        0,
        1,
    ]

    resumed_raw = _EndToEndClient()
    resumed_client = _budgeted_pipeline_client(tmp_path, resumed_raw, calls=3)
    resumed = run_paper(spec=spec, settings=settings, client=resumed_client)

    assert len(resumed_raw.requests) == 1
    assert resumed_raw.requests[0]["schema_name"] == "paper_evaluation_candidates"
    assert resumed_client.summary["structured_calls_started"] == 3
    assert len(resumed["extractor"]["calls"]) == 1
    assert len(resumed["extractor"]["resumed_calls"]) == 2
    assert resumed["extractor"]["execution"]["calls_resumed"] == 2
    assert resumed["extractor"]["successful_call_telemetry"]["calls"] == 2
    completed = read_json(checkpoint_path)
    assert completed["recoveries"] == {}
    assert len(completed["blocks"][root_id]["calls"]) == 3
    assert completed["blocks"][root_id]["successful_call_indexes"] == [1, 2]

    events = [
        json.loads(line)
        for line in resumed_client.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    request_hashes = [
        event["request"]["request_sha256"]
        for event in events
        if event["event_type"] == "reservation"
    ]
    assert len(request_hashes) == len(set(request_hashes)) == 3

    final_client = _budgeted_pipeline_client(tmp_path, _NeverCallClient(), calls=3)
    final = run_paper(spec=spec, settings=settings, client=final_client)
    assert final["extractor"]["execution"]["blocks_resumed"] == 1
    assert len(final["extractor"]["resumed_calls"]) == 3
    assert final["extractor"]["successful_call_telemetry"]["calls"] == 2
    assert final_client.summary["structured_calls_started"] == 3


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
    observation = json.loads((paper_dir / "observations.jsonl").read_text(encoding="utf-8").strip())
    assert observation["export_status"] == "needs_review"
    assert observation["export_reason"] == "projected EEE record failed schema validation"
    assert captured["eee_records"] == []
    assert len(captured["validation_errors"]) == 1
    assert not list((paper_dir / "eee").glob("*.json"))
    assert (paper_dir / "private" / "invalid-eee.json").exists()
