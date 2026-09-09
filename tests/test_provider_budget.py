from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proceedings_to_eee.cli import app
from proceedings_to_eee.corpus import CorpusSpec, PaperSpec
from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes, write_json
from proceedings_to_eee.pipeline import PipelineSettings, run_corpus
from proceedings_to_eee.providers.budget import (
    BudgetedProviderClient,
    ProviderBudgetContractError,
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _call(
    *,
    model: str = "fixture/requested",
    prompt_sha256: str = "1" * 64,
    schema_name: str = "fixture_schema",
    schema_sha256: str = "3" * 64,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = "minimal",
    max_tokens: int = 100,
    seed: int | None = 7,
    require_parameters: bool = False,
    cost: float | None = 0.01,
    reasoning_tokens: int | None = 2,
    attempts: int = 2,
) -> ProviderCall:
    return ProviderCall(
        model_requested=model,
        model_returned=model,
        provider_returned="OpenAI",
        prompt_sha256=prompt_sha256,
        response_sha256="2" * 64,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        completion_token_parameter=completion_token_parameter_for_model(model),
        seed=seed,
        schema_name=schema_name,
        schema_sha256=schema_sha256,
        require_parameters=require_parameters,
        latency_seconds=0.25,
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=reasoning_tokens,
        total_tokens=15,
        cost_usd=cost,
        request_id="private-request-id",
        finish_reason="stop",
        attempts=attempts,
    )


class _SuccessClient:
    def __init__(
        self,
        *,
        cost: float | None = 0.01,
        reasoning_tokens: int | None = 2,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self.cost = cost
        self.reasoning_tokens = reasoning_tokens

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        self.requests.append(kwargs)
        contract = structured_request_contract(
            schema_name=kwargs["schema_name"],
            schema=kwargs["schema"],
            seed=kwargs["seed"],
            require_parameters=kwargs["require_parameters"],
        )
        messages = [
            {"role": "system", "content": kwargs["system"]},
            {"role": "user", "content": kwargs["user"]},
        ]
        prompt_sha256 = hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        return StructuredResponse(
            payload={"ok": True},
            call=_call(
                model=kwargs["model"],
                prompt_sha256=prompt_sha256,
                schema_name=kwargs["schema_name"],
                schema_sha256=contract["schema"]["schema_sha256"],
                temperature=kwargs["temperature"],
                reasoning_effort=kwargs["reasoning_effort"],
                max_tokens=kwargs["max_tokens"],
                seed=kwargs["seed"],
                require_parameters=kwargs["require_parameters"],
                cost=self.cost,
                reasoning_tokens=self.reasoning_tokens,
            ),
        )


class _FailingClient:
    def __init__(self) -> None:
        self.requests = 0

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        del kwargs
        self.requests += 1
        raise RuntimeError("Bearer raw-provider-secret and raw prompt payload")


class _NeverCallClient:
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        del kwargs
        raise AssertionError("budget or checkpoint should stop before provider dispatch")


def _limits(
    *, calls: int = 2, max_cost: float = 2.0, reservation: float = 0.5
) -> ProviderBudgetLimits:
    return ProviderBudgetLimits(
        max_structured_calls=calls,
        max_cost_usd=max_cost,
        cost_reservation_per_call_usd=reservation,
    )


def _contract(limits: ProviderBudgetLimits) -> dict[str, Any]:
    return provider_budget_contract(
        corpus_binding={
            "schema_version": "pilot-corpus/0.2",
            "corpus_id": "fixture",
            "evaluation_split": "development",
            "corpus_spec_sha256": "a" * 64,
            "paper_ids_sha256": "b" * 64,
        },
        provider_run_contract={"model": "fixture/requested", "seed": 7},
        limits=limits,
    )


def _request(secret: str = "raw source excerpt") -> dict[str, Any]:
    return {
        "model": "fixture/requested",
        "system": f"system {secret}",
        "user": f"user {secret}",
        "schema_name": "fixture_schema",
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        "temperature": 0.0,
        "reasoning_effort": "minimal",
        "max_tokens": 100,
        "seed": 7,
        "require_parameters": False,
    }


def _budgeted(
    tmp_path: Path, client: Any, *, limits: ProviderBudgetLimits
) -> BudgetedProviderClient:
    return BudgetedProviderClient(
        client=client,
        ledger_path=tmp_path / "private" / "provider-budget-ledger.jsonl",
        contract=_contract(limits),
        limits=limits,
    )


def test_call_ceiling_stops_before_dispatch_and_ledger_has_full_safe_telemetry(
    tmp_path: Path,
) -> None:
    raw = _SuccessClient()
    client = _budgeted(tmp_path, raw, limits=_limits(calls=2))

    client.structured_chat(**_request())
    client.structured_chat(**_request())
    with pytest.raises(ProviderBudgetExhausted) as exc_info:
        client.structured_chat(**_request())

    assert exc_info.value.reason == "structured_call_limit"
    assert len(raw.requests) == 2
    assert client.summary["structured_calls_started"] == 2
    assert client.summary["structured_calls_succeeded"] == 2
    assert client.summary["reserved_cost_usd"] == 1.0
    assert client.summary["committed_cost_usd"] == 0.02
    assert client.summary["provider_reported_cost_usd"] == 0.02
    assert client.summary["provider_reported_reasoning_tokens_lower_bound"] == 4
    assert client.summary["provider_reported_reasoning_tokens_calls"] == 2
    assert client.summary["provider_reported_reasoning_tokens_missing_calls"] == 0
    assert client.summary["transport_attempts_lower_bound"] == 4
    assert client.summary["transport_retries_lower_bound"] == 2

    events = [
        json.loads(line) for line in client.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_type"] for event in events] == [
        "contract",
        "reservation",
        "completion",
        "reservation",
        "completion",
    ]
    reservation = events[1]
    completion = events[2]
    assert all(event["schema_version"] == "provider-budget-ledger-event/0.3" for event in events)
    expected_contract = structured_request_contract(
        schema_name="fixture_schema",
        schema=_request()["schema"],
        seed=7,
        require_parameters=False,
        model="fixture/requested",
        max_tokens=100,
    )
    assert reservation["request"]["request_contract"] == expected_contract
    assert (
        reservation["request"]["provider_schema_sha256"]
        == expected_contract["schema"]["schema_sha256"]
    )
    assert len(reservation["request"]["request_sha256"]) == 64
    assert reservation["request"]["max_tokens"] == 100
    assert reservation["request"]["completion_token_parameter"] == "max_tokens"
    assert len(reservation["request"]["system_sha256"]) == 64
    assert len(reservation["request"]["user_sha256"]) == 64
    assert completion["provider_call"] == {
        "attempts": 2,
        "cost_usd": 0.01,
        "data_collection": "deny",
        "finish_reason": "stop",
        "input_tokens": 10,
        "latency_seconds": 0.25,
        "max_tokens": 100,
        "completion_token_parameter": "max_tokens",
        "model_requested": "fixture/requested",
        "model_returned": "fixture/requested",
        "model_returned_disposition": "matches_requested",
        "model_returned_sha256": hashlib.sha256(b"fixture/requested").hexdigest(),
        "output_tokens": 5,
        "reasoning_tokens": 2,
        "prompt_sha256": reservation["request"]["prompt_sha256"],
        "provider_returned": "OpenAI",
        "provider_returned_disposition": "known_label",
        "provider_returned_sha256": hashlib.sha256(b"OpenAI").hexdigest(),
        "reasoning_effort": "minimal",
        "require_parameters": False,
        "response_format": "json_schema",
        "response_sha256": "2" * 64,
        "retries": 1,
        "schema_name": "fixture_schema",
        "schema_sha256": reservation["request"]["provider_schema_sha256"],
        "schema_strict": True,
        "seed": 7,
        "temperature": 0.0,
        "total_tokens": 15,
        "zdr": True,
    }
    assert completion["failure"] is None
    assert "timestamp" in reservation
    assert "timestamp" in completion
    assert client.head_path.is_file()


def test_openai_budget_reservation_and_completion_bind_materialized_token_field(
    tmp_path: Path,
) -> None:
    client = _budgeted(tmp_path, _SuccessClient(), limits=_limits())
    request = _request()
    request["model"] = "openai/gpt-5.5"

    client.structured_chat(**request)

    events = [json.loads(line) for line in client.ledger_path.read_text().splitlines()]
    reservation = events[1]["request"]
    completion = events[2]["provider_call"]
    assert reservation["max_tokens"] == 100
    assert reservation["completion_token_parameter"] == "max_completion_tokens"
    assert reservation["request_contract"]["max_tokens"] == 100
    assert reservation["request_contract"]["completion_token_parameter"] == (
        "max_completion_tokens"
    )
    assert completion["max_tokens"] == 100
    assert completion["completion_token_parameter"] == "max_completion_tokens"


def test_missing_reasoning_usage_is_explicit_and_summary_does_not_derive_it(
    tmp_path: Path,
) -> None:
    client = _budgeted(
        tmp_path,
        _SuccessClient(reasoning_tokens=None),
        limits=_limits(),
    )

    client.structured_chat(**_request())

    completion = json.loads(client.ledger_path.read_text().splitlines()[-1])
    assert completion["provider_call"]["reasoning_tokens"] is None
    resumed = _budgeted(tmp_path, _NeverCallClient(), limits=_limits())
    assert resumed.summary["provider_reported_reasoning_tokens_lower_bound"] == 0
    assert resumed.summary["provider_reported_reasoning_tokens_calls"] == 0
    assert resumed.summary["provider_reported_reasoning_tokens_missing_calls"] == 1
    assert resumed.summary["provider_reported_output_tokens_lower_bound"] == 5
    assert resumed.summary["provider_reported_total_tokens_lower_bound"] == 15


def test_remote_controlled_metadata_is_safely_enumerated_or_hashed(tmp_path: Path) -> None:
    class HostileMetadataClient(_SuccessClient):
        def structured_chat(self, **kwargs: Any) -> StructuredResponse:
            response = super().structured_chat(**kwargs)
            return StructuredResponse(
                payload=response.payload,
                call=response.call.model_copy(
                    update={
                        "model_returned": "Bearer remote-model-secret",
                        "provider_returned": "api_key=remote-provider-secret",
                        "finish_reason": "raw prompt payload from remote",
                    }
                ),
            )

    client = _budgeted(tmp_path, HostileMetadataClient(), limits=_limits())
    client.structured_chat(**_request())
    ledger = client.ledger_path.read_text(encoding="utf-8")
    for forbidden in (
        "remote-model-secret",
        "remote-provider-secret",
        "raw prompt payload from remote",
    ):
        assert forbidden not in ledger
    completion = json.loads(ledger.splitlines()[-1])
    call = completion["provider_call"]
    assert call["model_returned"] is None
    assert call["model_returned_disposition"] == "hashed_untrusted"
    assert call["provider_returned"] is None
    assert call["provider_returned_disposition"] == "hashed_untrusted"
    assert call["finish_reason"] == "other"


def test_remote_lone_surrogate_is_hashed_without_leaking_or_breaking_completion(
    tmp_path: Path,
) -> None:
    class SurrogateMetadataClient(_SuccessClient):
        def structured_chat(self, **kwargs: Any) -> StructuredResponse:
            response = super().structured_chat(**kwargs)
            return StructuredResponse(
                payload=response.payload,
                call=response.call.model_copy(update={"model_returned": "\ud800"}),
            )

    client = _budgeted(tmp_path, SurrogateMetadataClient(), limits=_limits())
    client.structured_chat(**_request())

    assert client.summary["structured_calls_completed"] == 1
    completion = json.loads(client.ledger_path.read_text(encoding="utf-8").splitlines()[-1])
    assert completion["provider_call"]["model_returned"] is None
    assert completion["provider_call"]["model_returned_disposition"] == "hashed_untrusted"


def test_cost_ceiling_uses_non_refundable_reservation_and_never_overruns(
    tmp_path: Path,
) -> None:
    raw = _SuccessClient(cost=None)
    limits = _limits(calls=50, max_cost=0.5, reservation=0.3)
    client = _budgeted(tmp_path, raw, limits=limits)

    client.structured_chat(**_request())
    with pytest.raises(ProviderBudgetExhausted) as exc_info:
        client.structured_chat(**_request())

    assert exc_info.value.reason == "cost_limit"
    assert len(raw.requests) == 1
    assert client.summary["reserved_cost_usd"] == 0.3
    assert client.summary["reservable_cost_remaining_usd"] == 0.2
    assert client.summary["reserved_cost_usd"] <= client.summary["max_cost_usd"]


def test_failed_call_is_charged_and_resume_cannot_reset_or_hide_it(tmp_path: Path) -> None:
    limits = _limits(calls=2, max_cost=1.0, reservation=0.5)
    first_raw = _FailingClient()
    first = _budgeted(tmp_path, first_raw, limits=limits)

    with pytest.raises(RuntimeError, match="raw-provider-secret"):
        first.structured_chat(**_request("private evidence quote"))

    assert first.summary["structured_calls_started"] == 1
    assert first.summary["structured_calls_failed"] == 1
    assert first.summary["reserved_cost_usd"] == 0.5
    first_events = [
        json.loads(line) for line in first.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert first_events[-1]["failure"] == {
        "http_status": None,
        "known_attempts": None,
        "terminal_class": "transport_or_client_failure",
    }
    assert first.summary["provider_call_telemetry_missing_calls"] == 1
    assert first.summary["completion_outcomes"]["technical_failure"] == 1

    second_raw = _SuccessClient()
    resumed = _budgeted(tmp_path, second_raw, limits=limits)
    resumed.structured_chat(**_request("another private quote"))
    with pytest.raises(ProviderBudgetExhausted):
        resumed.structured_chat(**_request())

    assert len(second_raw.requests) == 1
    assert resumed.summary["structured_calls_started"] == 2
    assert resumed.summary["structured_calls_failed"] == 1
    assert resumed.summary["structured_calls_succeeded"] == 1

    ledger = resumed.ledger_path.read_text(encoding="utf-8")
    for forbidden in (
        "raw-provider-secret",
        "private evidence quote",
        "another private quote",
        "raw source excerpt",
        "raw prompt payload",
        "private-request-id",
        "system private",
        "user private",
    ):
        assert forbidden not in ledger


def test_contract_binds_caller_metadata_without_persisting_it(tmp_path: Path) -> None:
    limits = _limits()
    corpus_binding = {"corpus_id": "private-corpus-name", "raw_note": "corpus secret"}
    run_contract = {"api_key": "sk-plain-secret", "system_prompt": "raw prompt"}
    contract = provider_budget_contract(
        corpus_binding=corpus_binding,
        provider_run_contract=run_contract,
        limits=limits,
    )
    client = BudgetedProviderClient(
        client=_NeverCallClient(),
        ledger_path=tmp_path / "private" / "provider-budget-ledger.jsonl",
        contract=contract,
        limits=limits,
    )

    ledger = client.ledger_path.read_text(encoding="utf-8")
    assert "private-corpus-name" not in ledger
    assert "corpus secret" not in ledger
    assert "sk-plain-secret" not in ledger
    assert "raw prompt" not in ledger
    assert (
        contract["corpus_binding_sha256"]
        == hashlib.sha256(_compact_json(corpus_binding)).hexdigest()
    )
    assert (
        contract["provider_run_contract_sha256"]
        == hashlib.sha256(_compact_json(run_contract)).hexdigest()
    )


def test_contract_hash_changes_with_candidate_validation_policy() -> None:
    limits = _limits()
    corpus_binding = {"corpus_id": "fixture"}
    baseline = {
        "candidate_validation": {
            "schema_version": "candidate-validation/0.1",
            "min_confidence": 0.8,
        }
    }
    changed = {
        "candidate_validation": {
            "schema_version": "candidate-validation/0.1",
            "min_confidence": 0.61,
        }
    }

    first = provider_budget_contract(
        corpus_binding=corpus_binding,
        provider_run_contract=baseline,
        limits=limits,
    )
    second = provider_budget_contract(
        corpus_binding=corpus_binding,
        provider_run_contract=changed,
        limits=limits,
    )

    assert first["provider_run_contract_sha256"] != second["provider_run_contract_sha256"]


def test_rejected_request_retains_only_allowlisted_failure_telemetry(tmp_path: Path) -> None:
    class RejectedClient:
        def structured_chat(self, **kwargs: Any) -> StructuredResponse:
            del kwargs
            raise ProviderRequestRejectedError(status_code=403)

    client = _budgeted(tmp_path, RejectedClient(), limits=_limits())
    with pytest.raises(ProviderRequestRejectedError):
        client.structured_chat(**_request())

    completion = json.loads(client.ledger_path.read_text().splitlines()[-1])
    assert completion["provider_call"] is None
    assert completion["failure"] == {
        "http_status": 403,
        "known_attempts": 1,
        "terminal_class": "request_rejected",
    }
    assert completion["outcome"] == "provider_request_rejected"
    assert client.summary["transport_attempts_lower_bound"] == 1
    assert client.summary["failure_attempt_telemetry_calls"] == 1
    assert client.summary["transport_attempt_telemetry_missing_calls"] == 0


def test_paid_refusal_failure_retains_provider_call_telemetry(tmp_path: Path) -> None:
    class RefusalClient:
        def structured_chat(self, **kwargs: Any) -> StructuredResponse:
            call = (
                _SuccessClient(cost=0.02)
                .structured_chat(**kwargs)
                .call.model_copy(update={"attempts": 1})
            )
            raise ProviderResponseValidationError(
                call=call,
                code="invalid_json",
                validation_keyword="structured_content_refusal",
            )

    client = _budgeted(tmp_path, RefusalClient(), limits=_limits())
    with pytest.raises(ProviderResponseValidationError) as exc_info:
        client.structured_chat(**_request())

    assert exc_info.value.validation_keyword == "structured_content_refusal"
    completion = json.loads(client.ledger_path.read_text().splitlines()[-1])
    assert completion["outcome"] == "provider_response_invalid_json"
    assert completion["provider_call"]["cost_usd"] == 0.02
    assert completion["provider_call"]["completion_token_parameter"] == "max_tokens"
    assert completion["failure"] == {
        "http_status": None,
        "known_attempts": 1,
        "terminal_class": "response_validation",
    }


def test_resume_allows_bound_ceiling_increase_but_rejects_accounting_change(
    tmp_path: Path,
) -> None:
    first_limits = _limits(calls=2)
    first = _budgeted(tmp_path, _SuccessClient(), limits=first_limits)
    first.structured_chat(**_request())

    increased_limits = _limits(calls=3, max_cost=3.0)
    resumed = _budgeted(tmp_path, _SuccessClient(), limits=increased_limits)
    assert resumed.summary["structured_calls_started"] == 1
    assert resumed.summary["max_structured_calls"] == 3
    assert "budget_amendment" in resumed.ledger_path.read_text(encoding="utf-8")

    changed_accounting = _limits(calls=3, max_cost=3.0, reservation=0.25)
    with pytest.raises(ProviderBudgetContractError, match="ledger contract changed"):
        _budgeted(tmp_path, _SuccessClient(), limits=changed_accounting)


def _paper() -> PaperSpec:
    return PaperSpec(
        paper_id="budget-paper",
        title="Budget paper",
        year=2026,
        venue="Fixture",
        pdf_url="https://example.org/paper.pdf",
        perspective_role="evaluated_system",
    )


def _settings(tmp_path: Path, *, calls: int = 2) -> PipelineSettings:
    return PipelineSettings(
        project_root=tmp_path,
        schema_path=PROJECT_ROOT / "schemas" / "eee-0.2.2" / "eval.schema.json",
        schema_sha256="088fed8029d42fb3a607aa67e1a05c39e425241b5cd90803705b37562f402f2a",
        output_root=tmp_path / "runs",
        model="fixture/requested",
        provider_max_structured_calls=calls,
        provider_max_cost_usd=10.0,
        provider_cost_reservation_per_call_usd=0.5,
    )


def _successful_paper_summary(paper: PaperSpec) -> dict[str, Any]:
    return {
        "schema_version": "pipeline-run/0.2",
        "status": "success",
        "paper_id": paper.paper_id,
        "title": paper.title,
        "counts": {
            "candidates": 0,
            "exported": 0,
            "eee_records": 0,
            "eee_schema_issues": 0,
            "spot_checks": 0,
            "spot_checks_exact": 0,
        },
        "wall_clock_seconds": 0.0,
        "review_state": {"status": "needs_review", "reasons": ["fixture"]},
    }


def test_corpus_bounded_stop_is_typed_and_checkpoint_resume_does_not_repeat_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paper = _paper()
    corpus = CorpusSpec(corpus_id="budget-corpus", description="fixture", papers=[paper])
    checkpoint = tmp_path / "runs" / paper.paper_id / "private" / "fixture-checkpoint"

    def checkpointed_run(*, spec: PaperSpec, settings: PipelineSettings, client: Any):
        del settings
        if checkpoint.exists():
            return _successful_paper_summary(spec)
        client.structured_chat(**_request("checkpointed private payload"))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("complete\n", encoding="utf-8")
        return _successful_paper_summary(spec)

    monkeypatch.setattr("proceedings_to_eee.pipeline.run_paper", checkpointed_run)
    first_raw = _SuccessClient()
    first = run_corpus(
        corpus=corpus,
        settings=_settings(tmp_path, calls=1),
        client=first_raw,
    )
    assert first["status"] == "success"
    assert first["provider_budget"]["structured_calls_started"] == 1
    assert len(first_raw.requests) == 1

    second = run_corpus(
        corpus=corpus,
        settings=_settings(tmp_path, calls=1),
        client=_NeverCallClient(),
    )
    assert second["status"] == "success"
    assert second["provider_budget"]["structured_calls_started"] == 1

    checkpoint.unlink()
    bounded = run_corpus(
        corpus=corpus,
        settings=_settings(tmp_path, calls=1),
        client=_NeverCallClient(),
    )
    assert bounded["status"] == "bounded_incomplete"
    assert bounded["papers_bounded_incomplete"] == 1
    assert bounded["runs"][0]["status"] == "bounded_incomplete"
    assert bounded["runs"][0]["bounded_stop"] == {
        "code": "provider_budget_exhausted",
        "reason": "structured_call_limit",
        "provider_call_dispatched": False,
    }
    assert bounded["provider_budget"]["bounded_stop"]["reason"] == "structured_call_limit"


def test_budget_contract_failure_is_corpus_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paper = _paper()
    corpus = CorpusSpec(corpus_id="budget-contract", description="fixture", papers=[paper])

    def corrupt_ledger(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise ProviderBudgetContractError("private source text must not be summarized")

    monkeypatch.setattr("proceedings_to_eee.pipeline.run_paper", corrupt_ledger)

    with pytest.raises(ProviderBudgetContractError, match="private source text"):
        run_corpus(
            corpus=corpus,
            settings=_settings(tmp_path, calls=2),
            client=_NeverCallClient(),
        )

    assert not (tmp_path / "runs" / paper.paper_id / "run.json").exists()


def test_bounded_summary_reports_checkpoint_progress_without_false_empty_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paper = _paper()
    corpus = CorpusSpec(corpus_id="budget-partial", description="fixture", papers=[paper])

    def partial_run(*, spec: PaperSpec, settings: PipelineSettings, client: Any) -> dict[str, Any]:
        checkpoint_path = (
            settings.output_root / spec.paper_id / "private" / "extractor-checkpoint.json"
        )
        if checkpoint_path.is_file():
            return _successful_paper_summary(spec)
        first = client.structured_chat(**_request("completed private payload"))
        contract = {"paper_id": spec.paper_id, "stage": "extractor"}
        write_json(
            checkpoint_path,
            {
                "schema_version": "extractor-block-checkpoint/0.3",
                "contract": contract,
                "contract_sha256": sha256_bytes(canonical_json_bytes(contract)),
                "blocks": {
                    "fixture-block": {
                        "block_text_sha256": "a" * 64,
                        "candidates": [],
                        "calls": [first.call.model_dump(mode="json", exclude_none=False)],
                        "successful_call_indexes": [0],
                        "warnings": [],
                    }
                },
                "recoveries": {},
            },
        )
        client.structured_chat(**_request("must stop before dispatch"))
        raise AssertionError("budget stop must interrupt before a second dispatch")

    monkeypatch.setattr("proceedings_to_eee.pipeline.run_paper", partial_run)
    summary = run_corpus(
        corpus=corpus,
        settings=_settings(tmp_path, calls=1),
        client=_SuccessClient(),
    )

    paper_summary = summary["runs"][0]
    assert paper_summary["status"] == "bounded_incomplete"
    assert paper_summary["counts_status"] == "not_finalized_due_to_bounded_stop"
    assert paper_summary["partial_progress"]["finalized_candidate_counts_available"] is False
    assert paper_summary["partial_progress"]["checkpointed_entries"] == {
        "extractor_blocks": 1,
        "row_batches": 0,
        "tuple_candidates": 0,
        "verifier_candidates": 0,
        "origin_candidates": 0,
    }
    assert paper_summary["partial_progress"]["checkpointed_completed_call_records"] == 1
    assert paper_summary["failure_provider_accounting"]["checkpointed_completed_calls"] == 1
    assert paper_summary["failure_provider_accounting"]["checkpoint_parse_complete"] is True
    assert paper_summary["extractor"]["completed_call_telemetry"]["calls"] == 1
    assert paper_summary["extractor"]["completed_call_telemetry"]["cost_usd_lower_bound"] == 0.01
    assert len(paper_summary["extractor"]["calls"]) == 1
    assert "request_id" not in paper_summary["extractor"]["calls"][0]
    assert summary["operations"]["extractor"]["calls"] == 1
    assert summary["provider_budget"]["structured_calls_completed"] == 1
    paper_root = tmp_path / "runs" / paper.paper_id
    assert not (paper_root / "observations.jsonl").exists()
    assert not (paper_root / "verifications.jsonl").exists()
    assert not (paper_root / "spot-checks.json").exists()

    resumed = run_corpus(
        corpus=corpus,
        settings=_settings(tmp_path, calls=1),
        client=_NeverCallClient(),
    )
    assert resumed["status"] == "success"
    assert resumed["provider_budget"]["structured_calls_completed"] == 1


def test_run_corpus_cli_exposes_finite_budget_options() -> None:
    result = CliRunner().invoke(app, ["run-corpus", "--help"], env={"COLUMNS": "240"})

    assert result.exit_code == 0, result.output
    assert "--max-structured-calls" in result.output
    assert "--max-provider-cost-usd" in result.output
    assert "--provider-call-cost-reservation-usd" in result.output


def _write_cli_corpus(tmp_path: Path, *, paper_id: str = "cli-paper") -> Path:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"fixture")
    corpus = tmp_path / "corpus.yaml"
    corpus.write_text(
        json.dumps(
            {
                "schema_version": "pilot-corpus/0.2",
                "corpus_id": "cli-corpus",
                "description": "fixture",
                "papers": [
                    {
                        "paper_id": paper_id,
                        "title": "CLI paper",
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
    return corpus


def test_cli_validates_corpus_before_reading_credentials_or_creating_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "proceedings_to_eee.cli._run_command",
        lambda **kwargs: "ere run-corpus fixture --max-structured-calls 10001",
    )
    corpus = _write_cli_corpus(tmp_path, paper_id="Unsafe_ID")
    output = tmp_path / "must-not-exist"
    credential_reads = 0

    def forbidden_runtime_key() -> str:
        nonlocal credential_reads
        credential_reads += 1
        raise AssertionError("credentials must not be read")

    monkeypatch.setattr("proceedings_to_eee.cli.runtime_key", forbidden_runtime_key)
    result = CliRunner().invoke(
        app,
        ["run-corpus", str(corpus), "--model", "fixture/model", "--output", str(output)],
    )

    assert result.exit_code == 1
    assert credential_reads == 0
    assert not output.exists()
    assert '"phase": "corpus_validation"' in result.output
    assert "Traceback" not in result.output


def test_cli_catches_runtime_failure_without_traceback_or_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "proceedings_to_eee.cli._run_command", lambda **kwargs: "ere run-corpus fixture"
    )
    corpus = _write_cli_corpus(tmp_path)
    output = tmp_path / "runs"

    def failed_runtime_key() -> str:
        raise RuntimeError("Bearer runtime-secret api_key=also-secret")

    monkeypatch.setattr("proceedings_to_eee.cli.runtime_key", failed_runtime_key)
    result = CliRunner().invoke(
        app,
        ["run-corpus", str(corpus), "--model", "fixture/model", "--output", str(output)],
    )

    assert result.exit_code == 1
    assert '"status": "run-corpus-technical-failure"' in result.output
    assert '"phase": "runtime"' in result.output
    assert "runtime-secret" not in result.output
    assert "also-secret" not in result.output
    assert '"message": "runtime operation failed"' in result.output
    assert "Traceback" not in result.output
    assert "provider-budget-ledger.jsonl" in result.output


def test_cli_returns_distinct_bounded_stop_exit_and_resume_instruction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "proceedings_to_eee.cli._run_command",
        lambda **kwargs: "ere run-corpus fixture --max-structured-calls 10001",
    )
    corpus_path = _write_cli_corpus(tmp_path)
    output = tmp_path / "runs"
    monkeypatch.setattr("proceedings_to_eee.cli.runtime_key", lambda: "fixture-key")
    monkeypatch.setattr("proceedings_to_eee.cli.OpenRouterClient", lambda **kwargs: object())
    monkeypatch.setattr(
        "proceedings_to_eee.cli.run_corpus",
        lambda **kwargs: {
            "status": "bounded_incomplete",
            "totals": {"eee_records": 0},
            "papers_failed": 1,
            "provider_budget": {
                "structured_calls_started": 2,
                "bounded_stop": {
                    "code": "provider_budget_exhausted",
                    "reason": "structured_call_limit",
                    "provider_call_dispatched": False,
                },
            },
        },
    )

    result = CliRunner().invoke(
        app,
        [
            "run-corpus",
            str(corpus_path),
            "--model",
            "fixture/model",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 3
    assert '"status": "bounded-incomplete"' in result.output
    assert "--max-structured-calls" in result.output
    assert "provider-budget-ledger.jsonl" in result.output
    assert "Traceback" not in result.output


def test_ledger_hash_chain_detects_tampering(tmp_path: Path) -> None:
    limits = _limits()
    client = _budgeted(tmp_path, _SuccessClient(), limits=limits)
    client.structured_chat(**_request())
    ledger = client.ledger_path
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(
        text.replace(
            '"model_returned":"fixture/requested"',
            '"model_returned":"tampered/returned"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProviderBudgetContractError, match="hash chain"):
        _budgeted(tmp_path, _NeverCallClient(), limits=limits)


def test_request_hash_is_bound_to_safe_fingerprint_not_raw_payload(tmp_path: Path) -> None:
    limits = _limits()
    client = _budgeted(tmp_path, _SuccessClient(), limits=limits)
    client.structured_chat(**_request("highly sensitive excerpt"))
    reservation = json.loads(client.ledger_path.read_text().splitlines()[1])
    safe_request = dict(reservation["request"])
    digest = safe_request.pop("request_sha256")

    assert (
        digest
        == hashlib.sha256(
            json.dumps(
                safe_request,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    assert "highly sensitive excerpt" not in json.dumps(reservation)
    without_materialization = dict(safe_request)
    without_materialization.pop("completion_token_parameter")
    assert hashlib.sha256(_compact_json(without_materialization)).hexdigest() != digest


def test_provider_budget_limits_require_an_actual_integer_call_ceiling() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ProviderBudgetLimits(  # type: ignore[arg-type]
            max_structured_calls=1.5,
            max_cost_usd=1.0,
            cost_reservation_per_call_usd=0.1,
        )

    with pytest.raises(ValueError, match="positive and finite"):
        ProviderBudgetLimits(
            max_structured_calls=1,
            max_cost_usd=10**400,
            cost_reservation_per_call_usd=0.1,
        )


def test_unicode_json_characters_do_not_split_ledger_records(tmp_path: Path) -> None:
    limits = _limits()
    client = _budgeted(tmp_path, _SuccessClient(), limits=limits)
    request = _request()
    request["user"] = "fixture user\u2028with JSON separators\u0085preserved"
    client.structured_chat(**request)

    assert client.summary["structured_calls_started"] == 1


def test_two_live_clients_cannot_both_dispatch_against_one_call_ceiling(
    tmp_path: Path,
) -> None:
    limits = _limits(calls=1, max_cost=1.0, reservation=0.1)
    first_raw = _SuccessClient()
    second_raw = _SuccessClient()
    first = _budgeted(tmp_path, first_raw, limits=limits)
    second = _budgeted(tmp_path, second_raw, limits=limits)
    start = threading.Barrier(2)

    def invoke(client: BudgetedProviderClient) -> str:
        start.wait(timeout=5)
        try:
            client.structured_chat(**_request())
        except ProviderBudgetExhausted:
            return "bounded"
        return "dispatched"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(invoke, (first, second)))

    assert sorted(outcomes) == ["bounded", "dispatched"]
    assert len(first_raw.requests) + len(second_raw.requests) == 1
    assert first.summary["structured_calls_started"] == 1
    assert second.summary["structured_calls_completed"] == 1
    assert len(first.ledger_path.read_text(encoding="utf-8").splitlines()) == 3


def test_advisory_lock_is_not_held_during_network_dispatch(tmp_path: Path) -> None:
    dispatch_barrier = threading.Barrier(2)

    class BarrierClient(_SuccessClient):
        def structured_chat(self, **kwargs: Any) -> StructuredResponse:
            dispatch_barrier.wait(timeout=5)
            return super().structured_chat(**kwargs)

    limits = _limits(calls=2, max_cost=1.0, reservation=0.1)
    first = _budgeted(tmp_path, BarrierClient(), limits=limits)
    second = _budgeted(tmp_path, BarrierClient(), limits=limits)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(client.structured_chat, **_request()) for client in (first, second)
        ]
        assert all(future.result(timeout=6).payload == {"ok": True} for future in futures)

    assert first.summary["structured_calls_completed"] == 2


def test_ledger_only_suffix_rollback_is_detected_by_separate_head(tmp_path: Path) -> None:
    limits = _limits(calls=2)
    client = _budgeted(tmp_path, _SuccessClient(), limits=limits)
    client.structured_chat(**_request())
    lines = client.ledger_path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines) == 3

    client.ledger_path.write_text("".join(lines[:-1]), encoding="utf-8")

    with pytest.raises(ProviderBudgetContractError, match="head does not match ledger"):
        _budgeted(tmp_path, _NeverCallClient(), limits=limits)


def test_missing_corrupt_and_symlinked_head_fail_conservatively(tmp_path: Path) -> None:
    limits = _limits()
    missing_root = tmp_path / "missing"
    missing = _budgeted(missing_root, _SuccessClient(), limits=limits)
    missing.head_path.unlink()
    with pytest.raises(ProviderBudgetContractError, match="exist together"):
        _budgeted(missing_root, _NeverCallClient(), limits=limits)

    corrupt_root = tmp_path / "corrupt"
    corrupt = _budgeted(corrupt_root, _SuccessClient(), limits=limits)
    corrupt.head_path.write_text('{"not":"a valid head"}\n', encoding="utf-8")
    with pytest.raises(ProviderBudgetContractError, match="head does not match ledger"):
        _budgeted(corrupt_root, _NeverCallClient(), limits=limits)

    symlink_root = tmp_path / "symlink"
    symlinked = _budgeted(symlink_root, _SuccessClient(), limits=limits)
    target = symlink_root / "target.json"
    target.write_text(symlinked.head_path.read_text(encoding="utf-8"), encoding="utf-8")
    symlinked.head_path.unlink()
    symlinked.head_path.symlink_to(target)
    with pytest.raises(ProviderBudgetContractError, match="head.*opened safely"):
        _budgeted(symlink_root, _NeverCallClient(), limits=limits)


def test_head_rejects_integral_float_fields_and_poisons_live_client(tmp_path: Path) -> None:
    limits = _limits()
    client = _budgeted(tmp_path, _SuccessClient(), limits=limits)
    client.structured_chat(**_request())
    head = json.loads(client.head_path.read_text(encoding="utf-8"))
    head["final_sequence"] = float(head["final_sequence"])
    client.head_path.write_bytes(_compact_json(head) + b"\n")

    with pytest.raises(ProviderBudgetContractError, match="head does not match ledger"):
        _ = client.summary
    with pytest.raises(ProviderBudgetContractError, match="poisoned"):
        _ = client.summary


def test_symlinked_lock_is_rejected_without_touching_target(tmp_path: Path) -> None:
    ledger = tmp_path / "private" / "provider-budget-ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    target = tmp_path / "lock-target"
    target.write_text("do-not-touch\n", encoding="utf-8")
    Path(f"{ledger}.lock").symlink_to(target)
    limits = _limits()

    with pytest.raises(ProviderBudgetContractError, match="lock could not be opened safely"):
        BudgetedProviderClient(
            client=_NeverCallClient(),
            ledger_path=ledger,
            contract=_contract(limits),
            limits=limits,
        )

    assert target.read_text(encoding="utf-8") == "do-not-touch\n"


def test_multiply_linked_ledger_is_rejected_and_poisons_live_client(tmp_path: Path) -> None:
    limits = _limits()
    client = _budgeted(tmp_path, _NeverCallClient(), limits=limits)
    os.link(client.ledger_path, tmp_path / "ledger-hard-link.jsonl")

    with pytest.raises(ProviderBudgetContractError, match="regular file"):
        _ = client.summary
    with pytest.raises(ProviderBudgetContractError, match="poisoned"):
        _ = client.summary


def test_crash_between_ledger_append_and_head_update_poisons_client_and_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    limits = _limits()
    client = _budgeted(tmp_path, _NeverCallClient(), limits=limits)

    def fail_head(*, event: dict[str, Any]) -> None:
        del event
        raise ProviderBudgetContractError("simulated head durability failure")

    monkeypatch.setattr(client, "_write_head_atomic_locked", fail_head)
    with pytest.raises(ProviderBudgetContractError, match="simulated head"):
        client.structured_chat(**_request())
    with pytest.raises(ProviderBudgetContractError, match="poisoned"):
        client.structured_chat(**_request())
    with pytest.raises(ProviderBudgetContractError, match="head does not match ledger"):
        _budgeted(tmp_path, _NeverCallClient(), limits=limits)


def test_complete_write_loop_handles_short_os_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import proceedings_to_eee.providers.budget as budget_module

    real_write = os.write

    def short_write(descriptor: int, payload: bytes) -> int:
        return real_write(descriptor, payload[: max(1, len(payload) // 3)])

    monkeypatch.setattr(budget_module.os, "write", short_write)
    limits = _limits()
    client = _budgeted(tmp_path, _SuccessClient(), limits=limits)
    client.structured_chat(**_request())

    assert (
        _budgeted(tmp_path, _NeverCallClient(), limits=limits).summary["structured_calls_completed"]
        == 1
    )


def test_fresh_nested_output_persists_each_created_directory_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_fsync_directory = BudgetedProviderClient._fsync_directory
    synced: list[Path] = []

    def recording_fsync(path: Path) -> None:
        synced.append(path)
        real_fsync_directory(path)

    monkeypatch.setattr(
        BudgetedProviderClient,
        "_fsync_directory",
        staticmethod(recording_fsync),
    )
    limits = _limits()
    output = tmp_path / "new-output" / "nested" / "private"
    BudgetedProviderClient(
        client=_NeverCallClient(),
        ledger_path=output / "provider-budget-ledger.jsonl",
        contract=_contract(limits),
        limits=limits,
    )

    assert tmp_path in synced
    assert tmp_path / "new-output" in synced
    assert tmp_path / "new-output" / "nested" in synced


def test_reported_actual_cost_replaces_reservation_and_overrun_blocks_future_calls(
    tmp_path: Path,
) -> None:
    raw = _SuccessClient(cost=1.2)
    limits = _limits(calls=5, max_cost=1.0, reservation=0.3)
    client = _budgeted(tmp_path, raw, limits=limits)

    client.structured_chat(**_request())
    summary = client.summary
    assert summary["reserved_authorization_usd"] == 0.3
    assert summary["committed_cost_usd"] == 1.2
    assert summary["provider_reported_cost_usd"] == 1.2
    assert summary["reservation_underestimated_calls"] == 1
    assert summary["actual_cost_overrun"] is True
    assert summary["actual_cost_overrun_usd"] == pytest.approx(0.2)
    assert summary["status"] == "actual_cost_overrun"

    with pytest.raises(ProviderBudgetExhausted) as exc_info:
        client.structured_chat(**_request())
    assert exc_info.value.reason == "cost_limit"
    assert len(raw.requests) == 1


def test_cost_gate_uses_exact_decimal_total_without_summary_float_round_trip(
    tmp_path: Path,
) -> None:
    class SequenceCostClient(_SuccessClient):
        def __init__(self) -> None:
            super().__init__()
            self.costs = iter((24.56929554586598, 1.387731481966084e-06))

        def structured_chat(self, **kwargs: Any) -> StructuredResponse:
            self.cost = next(self.costs)
            return super().structured_chat(**kwargs)

    raw = SequenceCostClient()
    limits = _limits(
        calls=4,
        max_cost=24.66929693359746,
        reservation=0.1,
    )
    client = _budgeted(tmp_path, raw, limits=limits)
    client.structured_chat(**_request())
    client.structured_chat(**_request())

    with pytest.raises(ProviderBudgetExhausted, match="cost_limit"):
        client.structured_chat(**_request())
    assert len(raw.requests) == 2


def test_summary_never_reintroduces_nonfinite_json_numbers(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)

    class HugeCostClient(_SuccessClient):
        def structured_chat(self, **kwargs: Any) -> StructuredResponse:
            barrier.wait(timeout=5)
            return super().structured_chat(**kwargs)

    limits = _limits(calls=2, max_cost=1e308, reservation=0.1)
    clients = [
        _budgeted(tmp_path, HugeCostClient(cost=1e308), limits=limits),
        _budgeted(tmp_path, HugeCostClient(cost=1e308), limits=limits),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(client.structured_chat, **_request()) for client in clients]
        assert all(future.result(timeout=6).payload == {"ok": True} for future in futures)

    summary = clients[0].summary
    assert summary["committed_cost_usd"] == "2" + ("0" * 308)
    assert isinstance(summary["provider_reported_cost_usd"], str)
    json.dumps(summary, allow_nan=False)


def _compact_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _rewrite_bound_ledger(
    client: BudgetedProviderClient,
    events: list[dict[str, Any]],
    *,
    normalize_sequences: bool = True,
) -> None:
    previous = None
    for sequence, event in enumerate(events):
        if normalize_sequences:
            event["sequence"] = sequence
        event["previous_event_sha256"] = previous
        event.pop("event_sha256", None)
        event["event_sha256"] = hashlib.sha256(_compact_json(event)).hexdigest()
        previous = event["event_sha256"]
    ledger_bytes = b"".join(_compact_json(event) + b"\n" for event in events)
    client.ledger_path.write_bytes(ledger_bytes)
    client.head_path.write_bytes(
        _compact_json(
            {
                "schema_version": "provider-budget-head/0.1",
                "contract_sha256": client.contract_sha256,
                "final_sequence": events[-1]["sequence"],
                "final_event_sha256": events[-1]["event_sha256"],
                "ledger_size_bytes": len(ledger_bytes),
                "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            }
        )
        + b"\n"
    )


@pytest.mark.parametrize(
    ("request_field", "tampered_value"),
    [
        ("max_tokens", 101),
        ("completion_token_parameter", "max_completion_tokens"),
    ],
)
def test_hash_valid_request_materialization_tampering_is_rejected_on_replay(
    request_field: str,
    tampered_value: Any,
    tmp_path: Path,
) -> None:
    limits = _limits()
    client = _budgeted(tmp_path, _SuccessClient(), limits=limits)
    client.structured_chat(**_request())
    events = [
        json.loads(line) for line in client.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    request = events[1]["request"]
    request[request_field] = tampered_value
    request["request_contract"][request_field] = tampered_value
    request["request_contract_sha256"] = hashlib.sha256(
        _compact_json(request["request_contract"])
    ).hexdigest()
    request_without_digest = {
        key: value for key, value in request.items() if key != "request_sha256"
    }
    request["request_sha256"] = hashlib.sha256(_compact_json(request_without_digest)).hexdigest()
    _rewrite_bound_ledger(client, events)

    with pytest.raises(ProviderBudgetContractError):
        _budgeted(tmp_path, _NeverCallClient(), limits=limits)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda event: event.update({"unexpected": True}), "event shape"),
        (lambda event: event["provider_call"].update({"cost_usd": -1}), "telemetry values"),
        (
            lambda event: event["provider_call"].update({"cost_usd": 10**400}),
            "telemetry values",
        ),
        (lambda event: event["provider_call"].update({"retries": True}), "telemetry values"),
        (
            lambda event: event["provider_call"].update({"reasoning_tokens": -1}),
            "telemetry values",
        ),
        (lambda event: event["provider_call"].pop("reasoning_tokens"), "telemetry shape"),
        (
            lambda event: event["provider_call"].update(
                {"completion_token_parameter": "max_completion_tokens"}
            ),
            "request-bound",
        ),
        (
            lambda event: event["provider_call"].pop("completion_token_parameter"),
            "telemetry shape",
        ),
        (lambda event: event.update({"outcome": "invented"}), "outcome"),
        (lambda event: event.update({"outcome": []}), "outcome"),
    ],
)
def test_hash_valid_malformed_events_are_rejected_semantically(
    mutation: Any,
    message: str,
    tmp_path: Path,
) -> None:
    limits = _limits()
    client = _budgeted(tmp_path, _SuccessClient(), limits=limits)
    client.structured_chat(**_request())
    events = [
        json.loads(line) for line in client.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    mutation(events[-1])
    _rewrite_bound_ledger(client, events)

    with pytest.raises(ProviderBudgetContractError, match=message):
        _budgeted(tmp_path, _NeverCallClient(), limits=limits)


def test_hash_valid_non_integer_sequence_and_boolean_reservation_are_rejected(
    tmp_path: Path,
) -> None:
    sequence_root = tmp_path / "sequence"
    limits = _limits()
    sequence_client = _budgeted(sequence_root, _SuccessClient(), limits=limits)
    sequence_client.structured_chat(**_request())
    sequence_events = [
        json.loads(line)
        for line in sequence_client.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    sequence_events[1]["sequence"] = 1.0
    _rewrite_bound_ledger(
        sequence_client,
        sequence_events,
        normalize_sequences=False,
    )
    with pytest.raises(ProviderBudgetContractError, match="hash chain"):
        _budgeted(sequence_root, _NeverCallClient(), limits=limits)

    reservation_root = tmp_path / "reservation"
    unit_limits = _limits(calls=2, max_cost=2.0, reservation=1.0)
    reservation_client = _budgeted(
        reservation_root,
        _SuccessClient(),
        limits=unit_limits,
    )
    reservation_client.structured_chat(**_request())
    reservation_events = [
        json.loads(line)
        for line in reservation_client.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    reservation_events[1]["reserved_cost_usd"] = True
    _rewrite_bound_ledger(reservation_client, reservation_events)
    with pytest.raises(ProviderBudgetContractError, match="reservation changed"):
        _budgeted(reservation_root, _NeverCallClient(), limits=unit_limits)


def test_nonfinite_json_telemetry_is_rejected_before_hash_validation(tmp_path: Path) -> None:
    limits = _limits()
    client = _budgeted(tmp_path, _SuccessClient(), limits=limits)
    client.structured_chat(**_request())
    ledger = client.ledger_path.read_text(encoding="utf-8")
    client.ledger_path.write_text(
        ledger.replace('"cost_usd":0.01', '"cost_usd":NaN'), encoding="utf-8"
    )

    with pytest.raises(ProviderBudgetContractError, match="invalid numeric constant"):
        _budgeted(tmp_path, _NeverCallClient(), limits=limits)


def test_oversized_json_integer_is_typed_and_poisons_live_client(tmp_path: Path) -> None:
    limits = _limits()
    client = _budgeted(tmp_path, _NeverCallClient(), limits=limits)
    ledger = client.ledger_path.read_text(encoding="utf-8")
    client.ledger_path.write_text(
        ledger.replace('"sequence":0', f'"sequence":{"9" * 5000}'),
        encoding="utf-8",
    )

    with pytest.raises(ProviderBudgetContractError, match="not valid JSONL"):
        _ = client.summary
    with pytest.raises(ProviderBudgetContractError, match="poisoned"):
        _ = client.summary
