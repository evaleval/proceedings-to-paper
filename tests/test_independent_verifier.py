from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderResponseValidationError,
    StructuredResponse,
    completion_token_parameter_for_model,
    openrouter_structural_schema,
    structured_request_contract,
)
from proceedings_to_eee.verification.independent import (
    VERIFIER_REQUEST_SETTINGS,
    VERIFIER_SCHEMA_NAME,
    CandidateVerificationAssessment,
    FrozenEvidenceBlock,
    FrozenEvidenceLine,
    GroundingStatus,
    IndependentDecision,
    UntrustedClaimedAnchor,
    VerificationRequest,
    contextualize_verification,
    verification_prompt,
    verification_provider_json_schema,
    verifier_evidence_block_sha256,
    verifier_request_contract,
    verify_candidate,
)


class FakeStructuredClient:
    def __init__(
        self,
        payload: dict[str, Any],
        api_key: str = "secret-test-key",
        *,
        response_sha256: str | None = None,
    ) -> None:
        self.payload = payload
        self.api_key = api_key
        self.response_sha256 = response_sha256
        self.calls: list[dict[str, Any]] = []

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        self.calls.append(kwargs)
        response_sha256 = (
            self.response_sha256
            or hashlib.sha256(
                json.dumps(self.payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
        )
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
                completion_token_parameter=completion_token_parameter_for_model(kwargs["model"]),
                seed=contract["seed"],
                response_format=schema_contract["response_format"],
                schema_name=schema_contract["schema_name"],
                schema_sha256=schema_contract["schema_sha256"],
                schema_strict=schema_contract["schema_strict"],
                require_parameters=contract["routing"]["require_parameters"],
                latency_seconds=0.01,
                attempts=1,
            ),
        )


def _block(
    candidate: CandidateObservation,
    *,
    lines: tuple[str, ...] = (
        "Table 2: Synthetic Speech Set, test split, n=6400; AUC percent; "
        "reference: Synthetic Speech Set labels",
        "Atlas Moderation API  61.3  74.6%  58.2",
    ),
) -> FrozenEvidenceBlock:
    frozen_lines = [
        FrozenEvidenceLine(
            line_id=f"L{index:04d}",
            section="result_block",
            source_line=20 + index,
            text=text,
        )
        for index, text in enumerate(lines, start=1)
    ]
    text = "\n".join(lines)
    return FrozenEvidenceBlock(
        block_id="src_paper:p7:table2",
        paper_id=candidate.paper_id,
        source_id="src_paper",
        page=7,
        lines=frozen_lines,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        claimed_anchor_untrusted=UntrustedClaimedAnchor.from_anchor(candidate.evidence[0]),
    )


def _payload(
    *,
    scope: str = "supported",
    support: str = "supported",
    decision: str = "accept",
    support_ids: list[str] | None = None,
    role_ids: list[str] | None = None,
    scope_ids: list[str] | None = None,
    value_ids: list[str] | None = None,
    metric_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "support": support,
        "support_evidence_line_ids": support_ids if support_ids is not None else ["L0002"],
        "role": "supported",
        "role_evidence_line_ids": role_ids if role_ids is not None else ["L0001", "L0002"],
        "scope": scope,
        "scope_evidence_line_ids": scope_ids if scope_ids is not None else ["L0001"],
        "value": "supported",
        "value_evidence_line_ids": value_ids if value_ids is not None else ["L0001", "L0002"],
        "metric": "supported",
        "metric_evidence_line_ids": metric_ids if metric_ids is not None else ["L0001"],
        "decision": decision,
        "justification": "The cited source lines establish the claimed result tuple.",
    }


def _candidate_with_dataset(
    candidate: CandidateObservation,
    dataset_raw: str,
) -> CandidateObservation:
    payload = candidate.model_dump(mode="python")
    payload["observation_id"] = None
    payload["scope"]["dataset_raw"] = dataset_raw
    return CandidateObservation.model_validate(payload)


def test_provider_schema_is_strict_complete_and_bounded() -> None:
    schema = verification_provider_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "support",
        "support_evidence_line_ids",
        "role",
        "role_evidence_line_ids",
        "scope",
        "scope_evidence_line_ids",
        "value",
        "value_evidence_line_ids",
        "metric",
        "metric_evidence_line_ids",
        "decision",
        "justification",
    }
    assert all(
        schema["properties"][name]["maxItems"] == 4
        for name in schema["properties"]
        if name.endswith("_evidence_line_ids")
    )
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
        "routing": {"require_parameters": True},
        "schema": {
            "response_format": "json_schema",
            "schema_name": VERIFIER_SCHEMA_NAME,
            "schema_sha256": expected_schema_sha256,
            "schema_strict": True,
        },
        "seed": None,
    }


def test_verifier_uses_shared_settings_and_grounded_effective_accept(
    eligible_candidate: CandidateObservation,
) -> None:
    payload = _payload()
    client = FakeStructuredClient(payload)
    block = _block(eligible_candidate)

    result, call = verify_candidate(
        client=client,  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=eligible_candidate,
        evidence_block=block,
    )

    assert result.provider_assessment == CandidateVerificationAssessment.model_validate(payload)
    assert result.provider_assessment.decision is IndependentDecision.ACCEPT
    assert result.grounding.passed is True
    assert result.effective_decision is IndependentDecision.ACCEPT
    assert result.observation_id == eligible_candidate.observation_id
    assert result.evidence_block_sha256 == verifier_evidence_block_sha256(block)
    assert call.model_requested == "independent/verifier-model"
    provider_args = client.calls[0]
    assert {
        key: provider_args[key] for key in VERIFIER_REQUEST_SETTINGS.as_dict()
    } == VERIFIER_REQUEST_SETTINGS.as_dict()
    assert provider_args["max_tokens"] == 2_000


def test_prompt_separates_trusted_lines_from_untrusted_anchor_and_review_state(
    eligible_candidate: CandidateObservation,
) -> None:
    prompt = verification_prompt(
        VerificationRequest(candidate=eligible_candidate, evidence_block=_block(eligible_candidate))
    )
    serialized = prompt.split("<VERIFICATION_INPUT>\n", 1)[1].split("\n</VERIFICATION_INPUT>", 1)[0]
    payload = json.loads(serialized)

    assert set(payload) == {
        "candidate_claim_untrusted",
        "candidate_claimed_anchor_untrusted",
        "trusted_frozen_source_block",
    }
    assert payload["candidate_claimed_anchor_untrusted"]["trust"] == (
        "candidate_supplied_untrusted"
    )
    trusted = payload["trusted_frozen_source_block"]
    assert "row" not in trusted and "column" not in trusted and "label" not in trusted
    assert trusted["lines"][0]["line_id"] == "L0001"
    for forbidden in (
        eligible_candidate.observation_id,
        "export_status",
        "text_support",
        "field_provenance",
        "proposal_traces",
        "example/atlas-moderation-api",
    ):
        assert forbidden not in prompt


def test_absent_dataset_demotes_accept_without_mutating_provider_assessment(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate = _candidate_with_dataset(eligible_candidate, "Dataset Not In Block")
    payload = _payload()
    result, _ = verify_candidate(
        client=FakeStructuredClient(payload),  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=candidate,
        evidence_block=_block(candidate),
    )

    assert result.provider_assessment.decision is IndependentDecision.ACCEPT
    assert result.grounding.scope.status is GroundingStatus.FAILED
    assert "scope:claim_not_present" in result.grounding.failure_codes
    assert result.effective_decision is IndependentDecision.REVIEW


def test_numeric_substring_does_not_ground_a_different_value(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate = eligible_candidate.model_copy(deep=True)
    assert candidate.value is not None
    candidate.value.raw = "5%"
    candidate.value.numeric = 5.0
    quote = "Atlas Moderation API  15%"
    candidate.evidence[0] = candidate.evidence[0].model_copy(
        update={"quote": quote, "quote_sha256": hashlib.sha256(quote.encode()).hexdigest()}
    )
    block = _block(
        candidate,
        lines=(
            "Table 2: Synthetic Speech Set, test split; AUC percent",
            "Atlas Moderation API  15%",
        ),
    )

    result, _ = verify_candidate(
        client=FakeStructuredClient(_payload()),  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=candidate,
        evidence_block=block,
    )

    assert result.grounding.value.status is GroundingStatus.FAILED
    assert "value:claim_not_present" in result.grounding.failure_codes
    assert result.effective_decision is IndependentDecision.REVIEW


def test_support_quote_must_be_exact_on_one_cited_source_line(
    eligible_candidate: CandidateObservation,
) -> None:
    result, _ = verify_candidate(
        client=FakeStructuredClient(_payload(support_ids=["L0001", "L0002"])),  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=eligible_candidate,
        evidence_block=_block(
            eligible_candidate,
            lines=(
                "Table 2: Synthetic Speech Set, test split, n=6400; AUC percent; "
                "Atlas Moderation API  61.3",
                "74.6%  58.2",
            ),
        ),
    )

    assert "support:claim_not_present" in result.grounding.failure_codes
    assert result.effective_decision is IndependentDecision.REVIEW


def test_adjacent_group_line_after_atomic_result_does_not_ground_scope(
    eligible_candidate: CandidateObservation,
) -> None:
    block = _block(
        eligible_candidate,
        lines=(
            "Table 2: AUC percent; reference: Synthetic Speech Set labels",
            "Atlas Moderation API  61.3  74.6%  58.2",
            "Synthetic Speech Set, test split, n=6400 (adjacent group)",
        ),
    )
    result, _ = verify_candidate(
        client=FakeStructuredClient(_payload(scope_ids=["L0003"])),  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=eligible_candidate,
        evidence_block=block,
    )

    assert "scope:scope_only_after_atomic_result" in result.grounding.failure_codes
    assert result.effective_decision is IndependentDecision.REVIEW


def test_irrelevant_pre_row_scope_citation_cannot_mask_post_row_only_claims(
    eligible_candidate: CandidateObservation,
) -> None:
    block = _block(
        eligible_candidate,
        lines=(
            "Table 2: AUC percent",
            "Atlas Moderation API  61.3  74.6%  58.2",
            "Synthetic Speech Set, test split, n=6400 (adjacent group)",
        ),
    )
    result, _ = verify_candidate(
        client=FakeStructuredClient(_payload(scope_ids=["L0001", "L0003"])),  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=eligible_candidate,
        evidence_block=block,
    )

    assert "scope:scope_only_after_atomic_result" in result.grounding.failure_codes
    assert result.effective_decision is IndependentDecision.REVIEW


@pytest.mark.parametrize(
    ("payload", "failure"),
    [
        (_payload(scope_ids=["L9999"]), "scope:unknown_line_id"),
        (_payload(scope_ids=["L0001", "L0001"]), "scope:duplicate_line_id"),
        (_payload(metric_ids=["L0002"]), "metric:claim_not_present"),
    ],
)
def test_unknown_duplicate_and_dimension_incompatible_ids_demote(
    eligible_candidate: CandidateObservation,
    payload: dict[str, Any],
    failure: str,
) -> None:
    result, _ = verify_candidate(
        client=FakeStructuredClient(payload),  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=eligible_candidate,
        evidence_block=_block(eligible_candidate),
    )
    assert failure in result.grounding.failure_codes
    assert result.provider_assessment.decision is IndependentDecision.ACCEPT
    assert result.effective_decision is IndependentDecision.REVIEW


def test_provider_review_and_reject_are_never_promoted(
    eligible_candidate: CandidateObservation,
) -> None:
    review_payload = _payload(scope="insufficient_evidence", decision="review", scope_ids=[])
    review, _ = verify_candidate(
        client=FakeStructuredClient(review_payload),  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=eligible_candidate,
        evidence_block=_block(eligible_candidate),
    )
    reject_payload = _payload(support="contradicted", decision="reject")
    reject, _ = verify_candidate(
        client=FakeStructuredClient(reject_payload),  # type: ignore[arg-type]
        model="independent/verifier-model",
        candidate=eligible_candidate,
        evidence_block=_block(eligible_candidate),
    )
    assert review.effective_decision is IndependentDecision.REVIEW
    assert reject.effective_decision is IndependentDecision.REJECT


def test_contextualization_recomputes_local_result(
    eligible_candidate: CandidateObservation,
) -> None:
    assessment = CandidateVerificationAssessment.model_validate(_payload())
    block = _block(eligible_candidate)
    expected = contextualize_verification(
        candidate=eligible_candidate,
        evidence_block=block,
        provider_assessment=assessment,
    )
    tampered = expected.model_dump(mode="python")
    tampered["grounding"]["passed"] = False
    with pytest.raises(ValidationError):
        type(expected).model_validate(tampered)


def test_inconsistent_provider_decision_is_rejected(
    eligible_candidate: CandidateObservation,
) -> None:
    with pytest.raises(ProviderResponseValidationError) as captured:
        verify_candidate(
            client=FakeStructuredClient(_payload(support="contradicted", decision="accept")),  # type: ignore[arg-type]
            model="independent/verifier-model",
            candidate=eligible_candidate,
            evidence_block=_block(eligible_candidate),
        )
    assert captured.value.code == "wire_validation"
    assert captured.value.validation_path == ()
    assert captured.value.validation_keyword == "candidate_verification_assessment"
    assert captured.value.call.model_requested == "independent/verifier-model"


def test_provider_assessment_must_match_completed_call_response_hash(
    eligible_candidate: CandidateObservation,
) -> None:
    with pytest.raises(ProviderResponseValidationError) as captured:
        verify_candidate(
            client=FakeStructuredClient(_payload(), response_sha256="f" * 64),  # type: ignore[arg-type]
            model="independent/verifier-model",
            candidate=eligible_candidate,
            evidence_block=_block(eligible_candidate),
        )

    assert captured.value.code == "wire_validation"
    assert captured.value.validation_keyword == "response_hash_mismatch"
    assert captured.value.call.response_sha256 == "f" * 64


def test_exact_anchor_mismatch_fails_before_provider_call(
    eligible_candidate: CandidateObservation,
) -> None:
    client = FakeStructuredClient(_payload())
    payload = eligible_candidate.model_dump(mode="python")
    payload["observation_id"] = None
    payload["evidence"][0]["row"] = "different claimed row"
    candidate = CandidateObservation.model_validate(payload)
    with pytest.raises(ValueError, match="no exact evidence anchor"):
        verify_candidate(
            client=client,  # type: ignore[arg-type]
            model="independent/verifier-model",
            candidate=candidate,
            evidence_block=_block(eligible_candidate),
        )
    assert client.calls == []


def test_frozen_evidence_block_rejects_hash_mismatch(
    eligible_candidate: CandidateObservation,
) -> None:
    block = _block(eligible_candidate)
    payload = block.model_dump(mode="python")
    payload["text_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="text_sha256 does not match"):
        FrozenEvidenceBlock.model_validate(payload)
