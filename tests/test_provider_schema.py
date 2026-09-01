from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from proceedings_to_eee.extraction.llm import extract_page_candidates
from proceedings_to_eee.extraction.llm_schema import WireExtraction, provider_json_schema
from proceedings_to_eee.extraction.pdf_layout import PageFragment
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderResponseValidationError,
    StructuredResponse,
    structured_request_contract,
)


def test_provider_schema_rejects_unknown_fields() -> None:
    schema = provider_json_schema()
    assert schema["additionalProperties"] is False
    assert "observations" in schema["required"]
    assert WireExtraction.model_config["extra"] == "forbid"


class _InvalidWireClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.call: ProviderCall | None = None

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        contract = structured_request_contract(
            schema_name=kwargs["schema_name"],
            schema=kwargs["schema"],
            seed=kwargs["seed"],
            require_parameters=kwargs["require_parameters"],
        )
        schema_contract = contract["schema"]
        response_bytes = json.dumps(self.payload, sort_keys=True).encode()
        self.call = ProviderCall(
            model_requested=kwargs["model"],
            model_returned="fixture/model",
            provider_returned="fixture-provider",
            prompt_sha256="a" * 64,
            response_sha256=hashlib.sha256(response_bytes).hexdigest(),
            temperature=kwargs["temperature"],
            reasoning_effort=kwargs["reasoning_effort"],
            max_tokens=kwargs["max_tokens"],
            seed=contract["seed"],
            response_format=schema_contract["response_format"],
            schema_name=schema_contract["schema_name"],
            schema_sha256=schema_contract["schema_sha256"],
            schema_strict=schema_contract["schema_strict"],
            latency_seconds=0.25,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_usd=0.0005,
            request_id="wire-invalid-request",
            finish_reason="stop",
            attempts=1,
        )
        return StructuredResponse(payload=self.payload, call=self.call)


def test_wire_pydantic_failure_retains_provider_call_without_raw_payload() -> None:
    secret = "raw-provider-secret"
    client = _InvalidWireClient(
        {
            "observations": [],
            "warnings": [secret],
        }
    )
    text = "Results table 0.80\n"
    fragment = PageFragment(
        fragment_id="fixture-fragment",
        source_id="fixture-source",
        page=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_count=len(text),
        numeric_token_count=1,
        result_signal_score=1.0,
    )

    with pytest.raises(ProviderResponseValidationError) as exc_info:
        extract_page_candidates(
            client=client,  # type: ignore[arg-type]
            model="fixture/model",
            paper_id="fixture-paper",
            paper_title="Fixture Paper",
            fragment=fragment,
        )

    assert exc_info.value.code == "wire_validation"
    assert exc_info.value.call is client.call
    assert exc_info.value.call.cost_usd == 0.0005
    assert secret not in str(exc_info.value)
    assert secret not in exc_info.value.call.model_dump_json()


def test_domain_invalid_proposal_is_dropped_without_losing_valid_siblings_or_raw_warning() -> None:
    secret = "provider-warning-source-text"
    base = {
        "roles": [],
        "scope": None,
        "metric": None,
        "value": None,
        "evidence": [
            {
                "kind": "prose",
                "label": "Methods",
                "row": None,
                "column": None,
                "quote": "The threshold is 0.5.",
            }
        ],
        "extraction_confidence": 0.9,
        "construct": None,
        "operationalization": None,
        "decision_rule": None,
        "evaluation_date": None,
        "notes": [],
    }
    client = _InvalidWireClient(
        {
            "observations": [
                base | {"claim_type": "primary_result"},
                base | {"claim_type": "method_metadata"},
            ],
            "page_summary": "fixture",
            "warnings": [secret],
        }
    )
    text = "The threshold is 0.5.\n"
    fragment = PageFragment(
        fragment_id="fixture-fragment",
        source_id="fixture-source",
        page=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_count=len(text),
        numeric_token_count=1,
        result_signal_score=1.0,
    )

    candidates, call, warnings = extract_page_candidates(
        client=client,  # type: ignore[arg-type]
        model="fixture/model",
        paper_id="fixture-paper",
        paper_title="Fixture Paper",
        fragment=fragment,
    )

    assert call is client.call
    assert [str(candidate.claim_type) for candidate in candidates] == ["method_metadata"]
    assert warnings == [
        "provider_reported_warnings=1",
        "local_candidate_validation_rejected=1",
    ]
    assert secret not in json.dumps(warnings)
