from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import httpx
import pytest

import proceedings_to_eee.providers.openrouter as openrouter_module
from proceedings_to_eee.providers.openrouter import (
    OpenRouterClient,
    ProviderRequestRejectedError,
    ProviderResponseValidationError,
    openrouter_structural_schema,
    structured_request_contract,
    structured_request_contract_from_call,
)

API_KEY = "sk" + "-or-test-secret"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _envelope(
    content: Any,
    *,
    request_id: str = "generation-id",
    model: str = "resolved/model",
    provider: str = "Mock Provider",
) -> dict[str, Any]:
    return {
        "id": request_id,
        "model": model,
        "provider": provider,
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "cost": 0.00125,
        },
    }


def _response(
    status_code: int,
    *,
    envelope: dict[str, Any] | None = None,
    text: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    request = httpx.Request("POST", "https://openrouter.test/chat/completions")
    if envelope is not None:
        return httpx.Response(
            status_code,
            json=envelope,
            headers=headers,
            request=request,
        )
    return httpx.Response(
        status_code,
        text=text or "",
        headers=headers,
        request=request,
    )


class _FakeHttpClient:
    def __init__(
        self,
        responses: list[httpx.Response | httpx.HTTPError],
        calls: list[dict[str, Any]],
        timeout: float,
    ) -> None:
        self._responses = responses
        self._calls = calls
        self._timeout = timeout

    def __enter__(self) -> _FakeHttpClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> httpx.Response:
        self._calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "json": json,
                "timeout": self._timeout,
            }
        )
        if not self._responses:
            raise AssertionError("unexpected provider call")
        response = self._responses.pop(0)
        if isinstance(response, httpx.HTTPError):
            raise response
        return response


def _install_fake_http(
    monkeypatch: pytest.MonkeyPatch,
    responses: Iterable[httpx.Response | httpx.HTTPError],
) -> list[dict[str, Any]]:
    queued_responses = list(responses)
    calls: list[dict[str, Any]] = []

    def client_factory(*, timeout: float) -> _FakeHttpClient:
        return _FakeHttpClient(queued_responses, calls, timeout)

    monkeypatch.setattr(openrouter_module.httpx, "Client", client_factory)
    return calls


def _structured_chat(client: OpenRouterClient, **overrides: Any):
    request: dict[str, Any] = {
        "model": "requested/model",
        "system": "Follow the schema.",
        "user": "Return the answer.",
        "schema_name": "answer_schema",
        "schema": SCHEMA,
        "temperature": 0.0,
        "reasoning_effort": "minimal",
        "max_tokens": 321,
        "seed": 17,
    }
    request.update(overrides)
    return client.structured_chat(**request)


def _exception_chain_text(error: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = error
    while current is not None:
        parts.extend((str(current), repr(current)))
        current = current.__cause__
    return "\n".join(parts)


def test_structured_request_contract_is_available_before_provider_call() -> None:
    canonical_schema = json.dumps(
        SCHEMA,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()

    contract = structured_request_contract(
        schema_name="answer_schema",
        schema=SCHEMA,
        seed=17,
        require_parameters=False,
    )

    assert contract == {
        "schema_version": "provider-request-contract/0.1",
        "privacy": {"data_collection": "deny", "zdr": True},
        "routing": {"require_parameters": False},
        "schema": {
            "response_format": "json_schema",
            "schema_name": "answer_schema",
            "schema_sha256": hashlib.sha256(canonical_schema).hexdigest(),
            "schema_strict": True,
        },
        "seed": 17,
    }


def test_strict_json_success_records_complete_secret_free_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"answer": 42}
    calls = _install_fake_http(
        monkeypatch,
        [
            _response(
                200,
                envelope=_envelope(json.dumps(payload)),
                headers={"x-request-id": "request-header-id"},
            )
        ],
    )
    clock = iter((100.0, 100.375))
    monkeypatch.setattr(openrouter_module.time, "monotonic", lambda: next(clock))

    result = _structured_chat(
        OpenRouterClient(
            api_key=API_KEY,
            base_url="https://openrouter.test/",
            timeout_seconds=9.5,
            max_attempts=2,
        )
    )

    assert result.payload == payload
    assert len(calls) == 1
    request = calls[0]
    assert request["url"] == "https://openrouter.test/chat/completions"
    assert request["timeout"] == 9.5
    assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert request["json"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_schema",
            "strict": True,
            "schema": SCHEMA,
        },
    }
    assert request["json"]["provider"] == {
        "data_collection": "deny",
        "require_parameters": False,
        "zdr": True,
    }
    assert request["json"]["temperature"] == 0.0
    assert request["json"]["reasoning"] == {"effort": "minimal", "exclude": True}
    assert request["json"]["seed"] == 17

    prompt = json.dumps(
        [
            {"role": "system", "content": "Follow the schema."},
            {"role": "user", "content": "Return the answer."},
        ],
        sort_keys=True,
        ensure_ascii=False,
    ).encode()
    raw_response = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    canonical_schema = json.dumps(
        SCHEMA,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert result.call.model_requested == "requested/model"
    assert result.call.model_returned == "resolved/model"
    assert result.call.provider_returned == "Mock Provider"
    assert result.call.prompt_sha256 == hashlib.sha256(prompt).hexdigest()
    assert result.call.response_sha256 == hashlib.sha256(raw_response).hexdigest()
    assert result.call.temperature == 0.0
    assert result.call.reasoning_effort == "minimal"
    assert result.call.max_tokens == 321
    assert result.call.seed == 17
    assert result.call.response_format == "json_schema"
    assert result.call.schema_name == "answer_schema"
    assert result.call.schema_sha256 == hashlib.sha256(canonical_schema).hexdigest()
    assert result.call.schema_strict is True
    assert result.call.data_collection == "deny"
    assert result.call.require_parameters is False
    assert result.call.zdr is True
    assert result.call.latency_seconds == 0.375
    assert result.call.input_tokens == 11
    assert result.call.output_tokens == 7
    assert result.call.total_tokens == 18
    assert result.call.cost_usd == 0.00125
    assert result.call.request_id == "request-header-id"
    assert result.call.finish_reason == "stop"
    assert result.call.attempts == 1
    assert API_KEY not in result.call.model_dump_json()
    contract = structured_request_contract_from_call(result.call)
    assert contract == {
        "schema_version": "provider-request-contract/0.1",
        "privacy": {"data_collection": "deny", "zdr": True},
        "routing": {"require_parameters": False},
        "schema": {
            "response_format": "json_schema",
            "schema_name": "answer_schema",
            "schema_sha256": hashlib.sha256(canonical_schema).hexdigest(),
            "schema_strict": True,
        },
        "seed": 17,
    }
    assert API_KEY not in json.dumps(contract)


def test_require_parameters_client_default_can_be_overridden_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_http(
        monkeypatch,
        [
            _response(200, envelope=_envelope('{"answer": 1}')),
            _response(200, envelope=_envelope('{"answer": 2}')),
        ],
    )
    client = OpenRouterClient(
        api_key=API_KEY,
        max_attempts=1,
        require_parameters=True,
    )

    client_default = _structured_chat(client)
    call_override = _structured_chat(client, require_parameters=False)

    assert calls[0]["json"]["provider"]["require_parameters"] is True
    assert client_default.call.require_parameters is True
    assert calls[1]["json"]["provider"]["require_parameters"] is False
    assert call_override.call.require_parameters is False
    assert call_override.call.data_collection == "deny"
    assert call_override.call.zdr is True


def test_content_list_text_fragments_are_joined(monkeypatch: pytest.MonkeyPatch) -> None:
    content = [
        {"type": "text", "text": '{"answer":'},
        {"type": "image_url", "image_url": {"url": "ignored"}},
        "ignored non-object fragment",
        {"type": "text", "text": " 42}"},
    ]
    _install_fake_http(monkeypatch, [_response(200, envelope=_envelope(content))])

    result = _structured_chat(OpenRouterClient(api_key=API_KEY, max_attempts=1))

    assert result.payload == {"answer": 42}


def test_numeric_bounds_are_removed_only_from_openrouter_schema_and_enforced_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bounded_schema = {
        "title": "Bounded score response",
        "description": "Annotations are not part of the provider grammar.",
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "note": {
                "title": "Optional note",
                "anyOf": [{"type": "string"}, {"type": "null"}],
            },
            "ratings": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1, "maximum": 5},
                        {"type": "null"},
                    ]
                },
            },
            "minimum": {"type": "string"},
        },
        "required": ["score"],
        "additionalProperties": False,
    }
    calls = _install_fake_http(
        monkeypatch,
        [_response(200, envelope=_envelope('{"score": 2.0}'))],
    )

    with pytest.raises(ProviderResponseValidationError) as exc_info:
        OpenRouterClient(api_key=API_KEY, max_attempts=1).structured_chat(
            model="requested/model",
            system="Follow the schema.",
            user="Return a score.",
            schema_name="bounded_score",
            schema=bounded_schema,
            seed=17,
        )

    provider_schema = openrouter_structural_schema(bounded_schema)
    assert bounded_schema["properties"]["score"]["minimum"] == 0.0
    assert bounded_schema["properties"]["score"]["maximum"] == 1.0
    assert "minimum" not in provider_schema["properties"]["score"]
    assert "maximum" not in provider_schema["properties"]["score"]
    assert "title" not in provider_schema
    assert "description" not in provider_schema
    assert provider_schema["properties"]["note"] == {"type": ["string", "null"]}
    numeric_items = provider_schema["properties"]["ratings"]["items"]
    assert numeric_items == {"type": ["integer", "null"]}
    assert provider_schema["properties"]["minimum"] == {"type": "string"}
    assert calls[0]["json"]["response_format"]["json_schema"]["schema"] == provider_schema
    canonical_provider_schema = json.dumps(
        provider_schema,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert exc_info.value.code == "schema_validation"
    assert (
        exc_info.value.call.schema_sha256 == hashlib.sha256(canonical_provider_schema).hexdigest()
    )


def test_transient_status_is_retried_and_attempt_count_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_http(
        monkeypatch,
        [
            _response(429, text="rate limited"),
            _response(200, envelope=_envelope('{"answer": 42}')),
        ],
    )
    delays: list[float] = []
    monkeypatch.setattr(openrouter_module.random, "random", lambda: 0.0)
    monkeypatch.setattr(openrouter_module.time, "sleep", delays.append)

    result = _structured_chat(OpenRouterClient(api_key=API_KEY, max_attempts=3))

    assert len(calls) == 2
    assert calls[0]["json"] == calls[1]["json"]
    assert delays == [1.0]
    assert result.call.attempts == 2


def test_truncated_structured_content_retains_secret_free_call_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truncated = f'{{"answer": 4, "credential": "{API_KEY}'
    _install_fake_http(
        monkeypatch,
        [
            _response(
                200,
                envelope=_envelope(truncated),
                headers={"x-request-id": "truncated-request"},
            )
        ],
    )

    with pytest.raises(
        ProviderResponseValidationError,
        match="did not contain valid structured JSON",
    ) as exc_info:
        _structured_chat(OpenRouterClient(api_key=API_KEY, max_attempts=1))

    call = exc_info.value.call
    assert exc_info.value.code == "invalid_json"
    assert call.response_sha256 == hashlib.sha256(truncated.encode()).hexdigest()
    assert call.input_tokens == 11
    assert call.output_tokens == 7
    assert call.total_tokens == 18
    assert call.cost_usd == 0.00125
    assert call.request_id == "truncated-request"
    assert API_KEY not in _exception_chain_text(exc_info.value)
    assert API_KEY not in call.model_dump_json()


def test_invalid_schema_payload_retains_paid_call_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"answer": "not an integer"}
    _install_fake_http(
        monkeypatch,
        [
            _response(
                200,
                envelope=_envelope(json.dumps(payload)),
                headers={"x-request-id": "invalid-payload-request"},
            )
        ],
    )

    with pytest.raises(
        ProviderResponseValidationError,
        match="did not satisfy requested JSON schema",
    ) as exc_info:
        _structured_chat(OpenRouterClient(api_key=API_KEY, max_attempts=1))

    call = exc_info.value.call
    raw_response = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    assert call.response_sha256 == hashlib.sha256(raw_response).hexdigest()
    assert call.seed == 17
    assert call.schema_name == "answer_schema"
    assert call.input_tokens == 11
    assert call.output_tokens == 7
    assert call.total_tokens == 18
    assert call.cost_usd == 0.00125
    assert call.request_id == "invalid-payload-request"
    assert call.finish_reason == "stop"
    assert call.attempts == 1
    assert exc_info.value.code == "schema_validation"
    assert API_KEY not in _exception_chain_text(exc_info.value)
    assert API_KEY not in call.model_dump_json()


def test_invalid_outer_json_raises_stable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_response = "not an API JSON envelope"
    _install_fake_http(
        monkeypatch,
        [_response(200, text=raw_response, headers={"x-request-id": "outer-json-request"})],
    )

    with pytest.raises(
        ProviderResponseValidationError,
        match="did not contain valid structured JSON",
    ) as exc_info:
        _structured_chat(OpenRouterClient(api_key=API_KEY, max_attempts=1))

    assert exc_info.value.code == "invalid_json"
    assert exc_info.value.call.response_sha256 == hashlib.sha256(raw_response.encode()).hexdigest()
    assert exc_info.value.call.request_id == "outer-json-request"


def test_refusal_is_reported_without_echoing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = _envelope("")
    envelope["choices"][0]["message"]["refusal"] = f"policy refusal: {API_KEY}"
    _install_fake_http(monkeypatch, [_response(200, envelope=envelope)])

    with pytest.raises(ValueError, match="model refused structured extraction") as exc_info:
        _structured_chat(OpenRouterClient(api_key=API_KEY, max_attempts=1))

    error_text = _exception_chain_text(exc_info.value)
    assert API_KEY not in error_text
    assert "[REDACTED]" in error_text


def test_http_error_body_is_redacted_through_exception_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_http(
        monkeypatch,
        [
            _response(
                400,
                text=(
                    f'{{"error":"invalid credential {API_KEY}",'
                    '"user_id":"private-user","request_id":"private-request"}'
                ),
            )
        ],
    )

    with pytest.raises(
        ProviderRequestRejectedError,
        match="request rejected with HTTP 400",
    ) as exc_info:
        _structured_chat(OpenRouterClient(api_key=API_KEY, max_attempts=3))

    error_text = _exception_chain_text(exc_info.value)
    assert API_KEY not in error_text
    assert "private-user" not in error_text
    assert "private-request" not in error_text
    assert exc_info.value.status_code == 400


def test_transport_error_is_sanitized_before_exception_chaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request(
        "POST",
        "https://openrouter.test/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    transport_error = httpx.ConnectError(
        f"transport rejected credential {API_KEY}",
        request=request,
    )
    _install_fake_http(monkeypatch, [transport_error])

    with pytest.raises(RuntimeError, match="request failed after retries") as exc_info:
        _structured_chat(OpenRouterClient(api_key=API_KEY, max_attempts=1))

    error_text = _exception_chain_text(exc_info.value)
    assert API_KEY not in error_text
    assert "[REDACTED]" in error_text
    assert not isinstance(exc_info.value.__cause__, httpx.HTTPError)
