from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import EvidenceKind
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    StructuredResponse,
    openrouter_structural_schema,
    structured_request_contract,
)
from proceedings_to_eee.verification.independent import (
    VERIFIER_SCHEMA_NAME,
    VERIFIER_SEED,
    CandidateVerificationAssessment,
    FrozenEvidenceBlock,
    IndependentDecision,
    VerificationFinding,
    verification_provider_json_schema,
    verifier_request_contract,
    verify_candidate,
)


class FakeStructuredClient:
    def __init__(self, payload: dict[str, Any], api_key: str = "secret-test-key") -> None:
        self.payload = payload
        self.api_key = api_key
        self.calls: list[dict[str, Any]] = []

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        self.calls.append(kwargs)
        response_sha256 = hashlib.sha256(
            json.dumps(self.payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        contract = structured_request_contract(
            schema_name=kwargs["schema_name"],
            schema=kwargs["schema"],
            seed=kwargs.get("seed"),
            require_parameters=kwargs.get("require_parameters", False),
        )
        schema_contract = contract["schema"]
        return StructuredResponse(
            payload=self.payload,
            call=ProviderCall(
                model_requested=kwargs["model"],
                model_returned=kwargs["model"],
                prompt_sha256="a" * 64,
                response_sha256=response_sha256,
                temperature=kwargs["temperature"],
                reasoning_effort=kwargs["reasoning_effort"],
                max_tokens=kwargs["max_tokens"],
                seed=contract["seed"],
                response_format=schema_contract["response_format"],
                schema_name=schema_contract["schema_name"],
                schema_sha256=schema_contract["schema_sha256"],
                schema_strict=schema_contract["schema_strict"],
                latency_seconds=0.01,
                attempts=1,
            ),
        )


def _block(candidate: CandidateObservation) -> FrozenEvidenceBlock:
    text = "Table 2\nAtlas Moderation API  61.3  74.6%  58.2\n"
    return FrozenEvidenceBlock(
        block_id="src_paper:p7:table2",
        paper_id=candidate.paper_id,
        source_id="src_paper",
        page=7,
        kind=EvidenceKind.TABLE,
        label="Table 2",
        row="Atlas Moderation API · Synthetic Speech Set",
        column="AUC",
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _payload(
    *,
    scope: str = "supported",
    support: str = "supported",
    decision: str = "accept",
) -> dict[str, str]:
    return {
        "support": support,
        "role": "supported",
        "scope": scope,
        "value": "supported",
        "metric": "supported",
        "decision": decision,
        "justification": (
            "Table 2 binds Atlas Moderation API, Synthetic Speech Set, AUC, and 74.6% in one row."
        ),
    }


def test_provider_schema_is_strict_and_complete() -> None:
    schema = verification_provider_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "support",
        "role",
        "scope",
        "value",
        "metric",
        "decision",
        "justification",
    }
    assert CandidateVerificationAssessment.model_config["extra"] == "forbid"
    expected_schema_sha256 = hashlib.sha256(
        json.dumps(
            openrouter_structural_schema(schema),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert verifier_request_contract() == {
        "schema_version": "provider-request-contract/0.1",
        "privacy": {"data_collection": "deny", "zdr": True},
        "routing": {"require_parameters": False},
        "schema": {
            "response_format": "json_schema",
            "schema_name": VERIFIER_SCHEMA_NAME,
            "schema_sha256": expected_schema_sha256,
            "schema_strict": True,
        },
        "seed": VERIFIER_SEED,
    }


def test_verifier_uses_separate_model_and_deterministic_settings(
    eligible_candidate: CandidateObservation,
) -> None:
    client = FakeStructuredClient(_payload())
    block = _block(eligible_candidate)

    result, call = verify_candidate(
        client=client,  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=eligible_candidate,
        evidence_block=block,
    )

    assert result.decision == IndependentDecision.ACCEPT
    assert result.support == VerificationFinding.SUPPORTED
    assert result.observation_id == eligible_candidate.observation_id
    assert result.evidence_block_sha256 == block.text_sha256
    assert call.model_requested == "independent/verifier-model"
    assert len(client.calls) == 1
    provider_args = client.calls[0]
    assert provider_args["temperature"] == 0.0
    assert provider_args["reasoning_effort"] == "minimal"
    assert provider_args["seed"] == 7
    assert provider_args["require_parameters"] is False
    assert provider_args["max_tokens"] == 2_000
    assert "export_status" not in provider_args["user"]
    assert "text_support" not in provider_args["user"]
    assert "Atlas Moderation API" in provider_args["user"]

    persisted = json.dumps(
        {
            "result": result.model_dump(mode="json"),
            "call": call.model_dump(mode="json"),
        }
    )
    assert client.api_key not in persisted
    assert client.api_key not in json.dumps(client.calls)


def test_insufficient_dimension_requires_review(
    eligible_candidate: CandidateObservation,
) -> None:
    client = FakeStructuredClient(_payload(scope="insufficient_evidence", decision="review"))
    result, _ = verify_candidate(
        client=client,  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=eligible_candidate,
        evidence_block=_block(eligible_candidate),
    )
    assert result.scope == VerificationFinding.INSUFFICIENT_EVIDENCE
    assert result.decision == IndependentDecision.REVIEW


def test_contradicted_dimension_requires_rejection(
    eligible_candidate: CandidateObservation,
) -> None:
    client = FakeStructuredClient(_payload(support="contradicted", decision="reject"))
    result, _ = verify_candidate(
        client=client,  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=eligible_candidate,
        evidence_block=_block(eligible_candidate),
    )
    assert result.support == VerificationFinding.CONTRADICTED
    assert result.decision == IndependentDecision.REJECT


def test_inconsistent_provider_decision_is_rejected(
    eligible_candidate: CandidateObservation,
) -> None:
    client = FakeStructuredClient(_payload(support="contradicted", decision="accept"))
    with pytest.raises(ValidationError, match="inconsistent with dimension findings"):
        verify_candidate(
            client=client,  # type: ignore[arg-type]
            model="independent/verifier-model",
            candidate=eligible_candidate,
            evidence_block=_block(eligible_candidate),
        )


def test_source_mismatch_fails_before_provider_call(
    eligible_candidate: CandidateObservation,
) -> None:
    client = FakeStructuredClient(_payload())
    block = _block(eligible_candidate).model_copy(update={"source_id": "different_source"})
    with pytest.raises(ValueError, match="no evidence anchor"):
        verify_candidate(
            client=client,  # type: ignore[arg-type]
            model="independent/verifier-model",
            candidate=eligible_candidate,
            evidence_block=block,
        )
    assert client.calls == []


def test_frozen_evidence_block_rejects_hash_mismatch(
    eligible_candidate: CandidateObservation,
) -> None:
    with pytest.raises(ValidationError, match="text_sha256 does not match"):
        FrozenEvidenceBlock(
            block_id="bad-block",
            paper_id=eligible_candidate.paper_id,
            source_id="src_paper",
            page=7,
            kind=EvidenceKind.TABLE,
            text="changed text",
            text_sha256="0" * 64,
        )
