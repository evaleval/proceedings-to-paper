from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderResponseValidationError,
    StructuredResponse,
    completion_token_parameter_for_model,
    structured_request_contract,
)
from proceedings_to_eee.resolution.tuple_resolution import (
    TUPLE_SCHEMA_NAME,
    TupleDataset,
    TupleDirection,
    TupleFieldEvidence,
    TupleMetric,
    TupleResolutionDecision,
    TupleScale,
    TupleScope,
    TupleSetting,
    TupleSystem,
    TupleValue,
    TupleWireProposal,
    build_tuple_resolution_input,
    materialize_tuple_wire_proposal,
    propose_tuple_resolution,
    tuple_assessment_is_export_concordant,
    tuple_resolution_prompt,
    tuple_resolution_provider_json_schema,
    tuple_wire_response_sha256,
    verify_tuple_resolution_proposal,
)

_ROW = (
    "Atlas Moderation API v2  Synthetic Speech Set v1  test  n=6400  "
    "zero-shot  AUC  higher is better  range 0 100  74.6%"
)


def _fixture(
    eligible_candidate: CandidateObservation,
) -> tuple[CandidateObservation, PdfLayout]:
    raw = eligible_candidate.model_dump(mode="json")
    raw["evidence"][0].update(
        {
            "row": (
                "Atlas Moderation API v2 · Synthetic Speech Set v1 · test · n=6400 · zero-shot"
            ),
            "column": "AUC percent higher is better range 0 100",
            "quote": _ROW,
            "quote_sha256": hashlib.sha256(_ROW.encode("utf-8")).hexdigest(),
            "region_id": "tregion_fixture",
            "planned_row_id": "trow_fixture",
            "cell_id": "tcell_" + "a" * 20,
            "numeric_token_id": "ttoken_" + "b" * 20,
            "header_ids": ["theader_" + "c" * 20],
        }
    )
    raw["observation_id"] = None
    candidate = CandidateObservation.model_validate(raw)
    text = "Table 2: Moderation results\n" + _ROW + "\n"
    page = PageFragment(
        fragment_id="frag-src-paper-0007",
        source_id="src_paper",
        page=7,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=4,
        result_signal_score=8.0,
    )
    return candidate, PdfLayout(
        source_id="src_paper",
        parser="test-layout",
        parser_version="1",
        page_count=1,
        pages=[page],
    )


def _proposal(candidate: CandidateObservation, layout: PdfLayout) -> TupleWireProposal:
    request = build_tuple_resolution_input(candidate, layout)
    evidence_id = request.result_evidence[0].evidence_id
    return TupleWireProposal(
        schema_version="tuple-resolution-wire-proposal/0.1",
        candidate_binding_sha256=request.candidate_binding_sha256,
        result_evidence_binding_sha256=request.result_evidence_binding_sha256,
        result_evidence_id=evidence_id,
        evaluated_system=TupleSystem(raw_name="Atlas Moderation API"),
        system_version="v2",
        dataset=TupleDataset(dataset_raw="Synthetic Speech Set"),
        dataset_version="v1",
        metric=TupleMetric(raw_name="AUC"),
        direction=TupleDirection(raw_direction="higher is better", lower_is_better=False),
        scale=TupleScale(raw_scale="range 0 100", min_score=0, max_score=100),
        value=TupleValue(raw="74.6%", numeric=74.6, comparator="exact"),
        uncertainty=None,
        unit="percent",
        setting=TupleSetting(raw_setting="zero-shot", parameters={}),
        scope=TupleScope(
            split="test",
            subset=None,
            group=None,
            language=None,
            sample_count=6400,
            aggregation=None,
            raw_scope=None,
        ),
        field_evidence=TupleFieldEvidence(
            system_evidence_id=evidence_id,
            system_version_evidence_id=evidence_id,
            dataset_evidence_id=evidence_id,
            dataset_version_evidence_id=evidence_id,
            metric_evidence_id=evidence_id,
            direction_evidence_id=evidence_id,
            scale_evidence_id=evidence_id,
            value_evidence_id=evidence_id,
            uncertainty_evidence_id=evidence_id,
            unit_evidence_id=evidence_id,
            setting_evidence_id=evidence_id,
            scope_evidence_id=evidence_id,
        ),
        unresolved_fields=[],
        not_applicable_fields=["uncertainty"],
        summary="Every field is printed in the selected result row and its headers.",
    )


def test_input_withholds_prior_tuple_fields_and_binds_exact_page_and_cell(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate, layout = _fixture(eligible_candidate)
    request = build_tuple_resolution_input(candidate, layout)

    assert request.candidate_tuple_fields == "withheld_untrusted"
    assert request.result_evidence[0].cell_id == "tcell_" + "a" * 20
    assert request.result_evidence[0].numeric_token_id == "ttoken_" + "b" * 20
    prompt_payload = json.loads(
        tuple_resolution_prompt(request)
        .split("<TUPLE_INPUT>\n", maxsplit=1)[1]
        .split("\n</TUPLE_INPUT>", maxsplit=1)[0]
    )
    assert not {"roles", "scope", "metric", "value"}.intersection(prompt_payload)
    assert "observation_id" not in prompt_payload


def test_provider_schema_omits_local_version_and_materialization_preserves_exact_hash(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate, layout = _fixture(eligible_candidate)
    proposal = _proposal(candidate, layout)
    provider_payload = proposal.model_dump(
        mode="json", exclude={"schema_version"}, exclude_none=False
    )
    schema = tuple_resolution_provider_json_schema()

    assert "schema_version" not in schema["properties"]
    assert "schema_version" not in schema["required"]
    assert "schema_version" not in json.dumps(schema, sort_keys=True)
    assert materialize_tuple_wire_proposal(provider_payload) == proposal
    assert (
        tuple_wire_response_sha256(proposal)
        == hashlib.sha256(
            json.dumps(provider_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    )
    assert "Omit schema_version from the response" in tuple_resolution_prompt(
        build_tuple_resolution_input(candidate, layout)
    )
    with pytest.raises(ValueError, match="must omit schema_version"):
        materialize_tuple_wire_proposal(
            provider_payload | {"schema_version": "provider-owned-version"}
        )


def test_provider_absence_lists_are_order_insensitive_but_duplicate_free(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate, layout = _fixture(eligible_candidate)
    provider_payload = _proposal(candidate, layout).model_dump(
        mode="json", exclude={"schema_version"}, exclude_none=False
    )
    for field in ("system_version", "dataset_version", "setting"):
        provider_payload[field] = None
        provider_payload["field_evidence"][f"{field}_evidence_id"] = None
    provider_payload["unresolved_fields"] = [
        "system_version",
        "dataset_version",
        "setting",
    ]
    original_payload = json.loads(json.dumps(provider_payload))
    provider_payload_sha256 = hashlib.sha256(
        json.dumps(provider_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    proposal = materialize_tuple_wire_proposal(provider_payload)

    assert provider_payload == original_payload
    assert proposal.unresolved_fields == [
        "system_version",
        "dataset_version",
        "setting",
    ]
    assert tuple_wire_response_sha256(proposal) == provider_payload_sha256
    duplicate_payload = json.loads(json.dumps(provider_payload))
    duplicate_payload["unresolved_fields"].append("setting")
    with pytest.raises(ValueError, match="unresolved_fields must be unique"):
        materialize_tuple_wire_proposal(duplicate_payload)

    duplicate_not_applicable = json.loads(json.dumps(provider_payload))
    duplicate_not_applicable["not_applicable_fields"].append("uncertainty")
    with pytest.raises(ValueError, match="not_applicable_fields must be unique"):
        materialize_tuple_wire_proposal(duplicate_not_applicable)

    overlapping_payload = json.loads(json.dumps(provider_payload))
    overlapping_payload["unresolved_fields"].append("uncertainty")
    with pytest.raises(ValueError, match="must be disjoint"):
        materialize_tuple_wire_proposal(overlapping_payload)

    invalid_not_applicable = json.loads(json.dumps(provider_payload))
    invalid_not_applicable["dataset_version"] = None
    invalid_not_applicable["field_evidence"]["dataset_version_evidence_id"] = (
        invalid_not_applicable["result_evidence_id"]
    )
    invalid_not_applicable["unresolved_fields"].remove("dataset_version")
    invalid_not_applicable["not_applicable_fields"].append("dataset_version")
    with pytest.raises(ValueError, match="only tuple uncertainty"):
        materialize_tuple_wire_proposal(invalid_not_applicable)


def test_local_verifier_accepts_only_evidence_supported_full_tuple(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate, layout = _fixture(eligible_candidate)
    request = build_tuple_resolution_input(candidate, layout)
    assessment = verify_tuple_resolution_proposal(
        request=request,
        proposal=_proposal(candidate, layout),
    )

    assert assessment.decision is TupleResolutionDecision.VERIFIED
    assert assessment.unresolved_fields == []
    assert assessment.unsafe_value_or_scope is False
    assert assessment.accepted_tuple.value is not None
    assert assessment.accepted_tuple.value.raw == "74.6%"
    assert assessment.allows_origin_or_export is False


def test_production_concordance_allows_only_unclaimed_optional_omissions(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate, layout = _fixture(eligible_candidate)
    request = build_tuple_resolution_input(candidate, layout)
    raw = _proposal(candidate, layout).model_dump(mode="json")
    for field in ("system_version", "dataset_version", "setting"):
        raw[field] = None
        raw["field_evidence"][f"{field}_evidence_id"] = None
    raw["unresolved_fields"] = ["dataset_version", "setting", "system_version"]
    proposal = TupleWireProposal.model_validate(raw)

    assessment = verify_tuple_resolution_proposal(request=request, proposal=proposal)

    assert assessment.decision is TupleResolutionDecision.REVIEW
    assert tuple_assessment_is_export_concordant(candidate, assessment) is True


def test_claimed_setting_without_lossless_projection_fails_concordance(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate, layout = _fixture(eligible_candidate)
    raw_candidate = candidate.model_dump(mode="json", by_alias=True)
    raw_candidate["observation_id"] = None
    raw_candidate["construct"] = "toxicity"
    candidate = CandidateObservation.model_validate(raw_candidate)
    request = build_tuple_resolution_input(candidate, layout)
    assessment = verify_tuple_resolution_proposal(
        request=request,
        proposal=_proposal(candidate, layout),
    )

    assert assessment.decision is TupleResolutionDecision.VERIFIED
    assert tuple_assessment_is_export_concordant(candidate, assessment) is False


def test_unresolved_scope_never_passes_production_concordance(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate, layout = _fixture(eligible_candidate)
    request = build_tuple_resolution_input(candidate, layout)
    raw = _proposal(candidate, layout).model_dump(mode="json")
    raw["scope"] = None
    raw["field_evidence"]["scope_evidence_id"] = None
    raw["unresolved_fields"] = ["scope"]
    proposal = TupleWireProposal.model_validate(raw)

    assessment = verify_tuple_resolution_proposal(request=request, proposal=proposal)

    assert assessment.decision is TupleResolutionDecision.REVIEW
    assert tuple_assessment_is_export_concordant(candidate, assessment) is False


def test_exact_arrow_and_percent_scale_are_structurally_supported(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate, layout = _fixture(eligible_candidate)
    candidate.evidence[0].column = "AUC ↑ %"
    request = build_tuple_resolution_input(candidate, layout)
    raw = _proposal(candidate, layout).model_dump(mode="json")
    raw["candidate_binding_sha256"] = request.candidate_binding_sha256
    raw["result_evidence_binding_sha256"] = request.result_evidence_binding_sha256
    raw["result_evidence_id"] = request.result_evidence[0].evidence_id
    for name, value in raw["field_evidence"].items():
        if value is not None:
            raw["field_evidence"][name] = request.result_evidence[0].evidence_id
    raw["direction"] = {"raw_direction": "↑", "lower_is_better": False}
    raw["scale"] = {"raw_scale": "%", "min_score": 0, "max_score": 100}
    proposal = TupleWireProposal.model_validate(raw)

    assessment = verify_tuple_resolution_proposal(request=request, proposal=proposal)

    assert assessment.field_states.direction.value == "verified"
    assert assessment.field_states.scale.value == "verified"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("value", TupleValue(raw="99.9%", numeric=99.9, comparator="exact")),
        (
            "scope",
            TupleScope(
                split=None,
                subset="invented holdout",
                group=None,
                language=None,
                sample_count=None,
                aggregation=None,
                raw_scope=None,
            ),
        ),
    ],
)
def test_invented_value_or_scope_is_rejected_and_removed(
    eligible_candidate: CandidateObservation,
    field: str,
    replacement: Any,
) -> None:
    candidate, layout = _fixture(eligible_candidate)
    request = build_tuple_resolution_input(candidate, layout)
    proposal = _proposal(candidate, layout).model_copy(update={field: replacement})
    proposal = TupleWireProposal.model_validate(proposal.model_dump(mode="json"))

    assessment = verify_tuple_resolution_proposal(request=request, proposal=proposal)

    assert assessment.decision is TupleResolutionDecision.REJECT
    assert assessment.unsafe_value_or_scope is True
    assert getattr(assessment.accepted_tuple, field) is None
    assert field in assessment.unresolved_fields


@pytest.mark.parametrize(
    "field",
    ["system_version", "dataset_version", "direction", "scale", "setting", "scope"],
)
def test_missing_required_semantic_prevents_verified_full_tuple(
    eligible_candidate: CandidateObservation,
    field: str,
) -> None:
    candidate, layout = _fixture(eligible_candidate)
    request = build_tuple_resolution_input(candidate, layout)
    raw = _proposal(candidate, layout).model_dump(mode="json")
    raw[field] = None
    raw["field_evidence"][f"{field}_evidence_id"] = None
    raw["unresolved_fields"] = [field]
    proposal = TupleWireProposal.model_validate(raw)

    assessment = verify_tuple_resolution_proposal(request=request, proposal=proposal)

    assert assessment.decision is TupleResolutionDecision.REVIEW
    assert getattr(assessment.field_states, field).value == "unresolved"


class _TupleClient:
    def __init__(self, payload: dict[str, Any], *, wrong_response_hash: bool = False) -> None:
        self.payload = payload
        self.wrong_response_hash = wrong_response_hash
        self.requests: list[dict[str, Any]] = []

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        self.requests.append(dict(kwargs))
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
        response_hash = hashlib.sha256(
            json.dumps(self.payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if self.wrong_response_hash:
            response_hash = "0" * 64
        return StructuredResponse(
            payload=self.payload,
            call=ProviderCall(
                model_requested=kwargs["model"],
                model_returned=kwargs["model"],
                provider_returned="test-provider",
                prompt_sha256=hashlib.sha256(
                    json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
                response_sha256=response_hash,
                temperature=kwargs["temperature"],
                reasoning_effort=kwargs["reasoning_effort"],
                max_tokens=kwargs["max_tokens"],
                completion_token_parameter=completion_token_parameter_for_model(kwargs["model"]),
                seed=kwargs["seed"],
                schema_name=kwargs["schema_name"],
                schema_sha256=contract["schema"]["schema_sha256"],
                require_parameters=kwargs["require_parameters"],
                latency_seconds=0.1,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cost_usd=0.001,
                attempts=1,
            ),
        )


def test_provider_proposal_is_response_hash_bound_and_never_exports(
    eligible_candidate: CandidateObservation,
) -> None:
    candidate, layout = _fixture(eligible_candidate)
    request = build_tuple_resolution_input(candidate, layout)
    proposal = _proposal(candidate, layout)
    provider_payload = proposal.model_dump(
        mode="json", exclude={"schema_version"}, exclude_none=False
    )
    client = _TupleClient(provider_payload)

    returned, assessment, call = propose_tuple_resolution(
        client=client,
        model="vendor/tuple-model",
        request=request,
        require_parameters=True,
    )

    assert returned == proposal
    assert call.response_sha256 == tuple_wire_response_sha256(proposal)
    assert assessment.allows_origin_or_export is False
    assert client.requests[0]["schema_name"] == TUPLE_SCHEMA_NAME

    with pytest.raises(
        ProviderResponseValidationError, match="WireExtraction validation"
    ) as generic_error:
        propose_tuple_resolution(
            client=_TupleClient(
                provider_payload,
                wrong_response_hash=True,
            ),
            model="vendor/tuple-model",
            request=request,
            require_parameters=True,
        )
    assert generic_error.value.validation_path == ()
    assert generic_error.value.validation_keyword is None

    pydantic_invalid_payload = json.loads(json.dumps(provider_payload))
    pydantic_invalid_payload["scale"]["min_score"] = 100
    pydantic_invalid_payload["scale"]["max_score"] = 0
    with pytest.raises(ProviderResponseValidationError) as pydantic_error:
        propose_tuple_resolution(
            client=_TupleClient(pydantic_invalid_payload),
            model="vendor/tuple-model",
            request=request,
            require_parameters=True,
        )
    assert pydantic_error.value.code == "wire_validation"
    assert pydantic_error.value.validation_path[0] == "scale"
    assert pydantic_error.value.validation_keyword == "value_error"
    assert (
        pydantic_error.value.call.response_sha256
        == hashlib.sha256(
            json.dumps(pydantic_invalid_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    )

    private_looking_key = "/Users/example/private/source.pdf"
    provider_controlled_key_payload = json.loads(json.dumps(provider_payload))
    provider_controlled_key_payload[private_looking_key] = "do not retain this key"
    with pytest.raises(ProviderResponseValidationError) as sanitized_error:
        propose_tuple_resolution(
            client=_TupleClient(provider_controlled_key_payload),
            model="vendor/tuple-model",
            request=request,
            require_parameters=True,
        )
    assert sanitized_error.value.validation_path == ("<unknown_field>",)
    assert private_looking_key not in str(sanitized_error.value)
    assert private_looking_key not in repr(sanitized_error.value.validation_path)
