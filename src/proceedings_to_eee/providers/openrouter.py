"""Minimal OpenRouter client with structured-output and complete run metadata."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from jsonschema import ValidationError, validate
from pydantic import BaseModel, ConfigDict, Field

ProviderValidationCode = Literal["invalid_json", "schema_validation", "wire_validation"]


class ProviderCall(BaseModel):
    """Secret-free metadata for one provider request."""

    model_config = ConfigDict(extra="forbid")
    provider: str = "openrouter"
    model_requested: str
    model_returned: str | None = None
    provider_returned: str | None = None
    prompt_sha256: str
    response_sha256: str
    temperature: float | None
    reasoning_effort: str | None = None
    max_tokens: int
    seed: int | None
    response_format: Literal["json_schema"] = "json_schema"
    schema_name: str
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_strict: Literal[True] = True
    data_collection: Literal["allow", "deny"] = "deny"
    require_parameters: bool = False
    zdr: bool = True
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    attempts: int = Field(ge=1)


class ProviderResponseValidationError(ValueError):
    """A completed provider call whose response failed secret-free local validation."""

    def __init__(
        self,
        *,
        call: ProviderCall,
        code: ProviderValidationCode = "schema_validation",
    ) -> None:
        messages = {
            "invalid_json": "OpenRouter response did not contain valid structured JSON",
            "schema_validation": "OpenRouter response did not satisfy requested JSON schema",
            "wire_validation": "Provider response did not satisfy local WireExtraction validation",
        }
        super().__init__(messages[code])
        self.call = call
        self.code = code


class ProviderRequestRejectedError(RuntimeError):
    """A non-retryable HTTP rejection whose response body is intentionally discarded."""

    def __init__(self, *, status_code: int) -> None:
        super().__init__(f"OpenRouter request rejected with HTTP {status_code}")
        self.status_code = status_code


@dataclass(frozen=True)
class StructuredResponse:
    payload: dict[str, Any]
    call: ProviderCall


def openrouter_structural_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a compact structural schema while retaining stricter local validation.

    Provider grammar compilers do not need JSON Schema annotation keywords and
    some reject numeric bounds. Simple nullable primitive unions are equivalent
    as ``type`` arrays but compile to materially smaller grammars.
    """

    annotations = {
        "$comment",
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
    }

    def transform(value: Any) -> Any:
        if isinstance(value, list):
            return [transform(item) for item in value]
        if not isinstance(value, dict):
            return value
        transformed = {
            key: transform(item) for key, item in value.items() if key not in annotations
        }
        node_type = value.get("type")
        numeric = (isinstance(node_type, str) and node_type in {"integer", "number"}) or (
            isinstance(node_type, list) and any(item in {"integer", "number"} for item in node_type)
        )
        if numeric:
            transformed.pop("minimum", None)
            transformed.pop("maximum", None)
        branches = transformed.get("anyOf")
        if (
            isinstance(branches, list)
            and branches
            and all(
                isinstance(branch, dict)
                and set(branch) == {"type"}
                and isinstance(branch["type"], str)
                for branch in branches
            )
        ):
            transformed.pop("anyOf")
            transformed["type"] = list(dict.fromkeys(branch["type"] for branch in branches))
        return transformed

    return transform(schema)


def _fingerprint_bytes(value: Any, *, fallback: bytes) -> bytes:
    """Serialize only for hashing; callers never retain or expose the returned raw bytes."""

    if isinstance(value, str):
        return value.encode("utf-8")
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        return fallback


def _request_contract(
    *,
    data_collection: Literal["allow", "deny"],
    zdr: bool,
    response_format: Literal["json_schema"],
    schema_name: str,
    schema_sha256: str,
    schema_strict: Literal[True],
    seed: int | None,
    require_parameters: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "provider-request-contract/0.1",
        "privacy": {
            "data_collection": data_collection,
            "zdr": zdr,
        },
        "routing": {"require_parameters": require_parameters},
        "schema": {
            "response_format": response_format,
            "schema_name": schema_name,
            "schema_sha256": schema_sha256,
            "schema_strict": schema_strict,
        },
        "seed": seed,
    }


def structured_request_contract(
    *,
    schema_name: str,
    schema: dict[str, Any],
    seed: int | None,
    require_parameters: bool = False,
) -> dict[str, Any]:
    """Build the versioned, secret-free contract before any provider call."""

    provider_schema = openrouter_structural_schema(schema)
    schema_bytes = json.dumps(
        provider_schema,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return _request_contract(
        data_collection="deny",
        zdr=True,
        response_format="json_schema",
        schema_name=schema_name,
        schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
        schema_strict=True,
        seed=seed,
        require_parameters=require_parameters,
    )


def structured_request_contract_from_call(call: ProviderCall) -> dict[str, Any]:
    """Reconstruct a request contract from completed-call telemetry."""

    return _request_contract(
        data_collection=call.data_collection,
        zdr=call.zdr,
        response_format=call.response_format,
        schema_name=call.schema_name,
        schema_sha256=call.schema_sha256,
        schema_strict=call.schema_strict,
        seed=call.seed,
        require_parameters=call.require_parameters,
    )


class OpenRouterClient:
    """Call the OpenAI-compatible OpenRouter API without persisting credentials."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 180.0,
        max_attempts: int = 4,
        require_parameters: bool = False,
    ) -> None:
        if not api_key or api_key.isspace():
            raise ValueError("OpenRouter API key is required at runtime")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._require_parameters = require_parameters

    def structured_chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
        temperature: float | None = 0.0,
        reasoning_effort: str | None = "minimal",
        max_tokens: int = 16_000,
        seed: int | None = 7,
        require_parameters: bool | None = None,
    ) -> StructuredResponse:
        """Request one strict JSON object and retain only secret-free telemetry."""

        effective_require_parameters = (
            self._require_parameters if require_parameters is None else require_parameters
        )
        provider_schema = openrouter_structural_schema(schema)
        contract = structured_request_contract(
            schema_name=schema_name,
            schema=schema,
            seed=seed,
            require_parameters=effective_require_parameters,
        )
        privacy_contract = contract["privacy"]
        routing_contract = contract["routing"]
        schema_contract = contract["schema"]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_bytes = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": {
                "type": schema_contract["response_format"],
                "json_schema": {
                    "name": schema_contract["schema_name"],
                    "strict": schema_contract["schema_strict"],
                    "schema": provider_schema,
                },
            },
            "provider": {
                "data_collection": privacy_contract["data_collection"],
                "require_parameters": routing_contract["require_parameters"],
                "zdr": privacy_contract["zdr"],
            },
        }
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
        if contract["seed"] is not None:
            body["seed"] = contract["seed"]
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/evaleval/proceedings-to-eee",
            "X-Title": "Proceedings to EEE",
        }
        started = time.monotonic()
        last_error: Exception | None = None
        response: httpx.Response | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.post(
                        f"{self._base_url}/chat/completions", headers=headers, json=body
                    )
                if response.status_code in {408, 409, 429} or response.status_code >= 500:
                    response.raise_for_status()
                if response.is_error:
                    raise ProviderRequestRejectedError(status_code=response.status_code)
                break
            except ProviderRequestRejectedError:
                raise
            except (httpx.HTTPError, RuntimeError) as error:
                safe_error = RuntimeError(
                    (str(error) or type(error).__name__).replace(self._api_key, "[REDACTED]")
                )
                last_error = safe_error
                if attempt == self._max_attempts:
                    raise RuntimeError("OpenRouter request failed after retries") from safe_error
                delay = min(12.0, (2 ** (attempt - 1)) + random.random())
                time.sleep(delay)
        if response is None:
            raise RuntimeError("OpenRouter request produced no response") from last_error

        def completed_call(
            *,
            response_bytes: bytes,
            envelope: dict[str, Any] | None = None,
            choice: dict[str, Any] | None = None,
        ) -> ProviderCall:
            safe_envelope = envelope or {}
            usage = safe_envelope.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            safe_choice = choice or {}
            return ProviderCall(
                model_requested=model,
                model_returned=safe_envelope.get("model"),
                provider_returned=safe_envelope.get("provider"),
                prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
                response_sha256=hashlib.sha256(response_bytes).hexdigest(),
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
                seed=contract["seed"],
                response_format=schema_contract["response_format"],
                schema_name=schema_contract["schema_name"],
                schema_sha256=schema_contract["schema_sha256"],
                schema_strict=schema_contract["schema_strict"],
                data_collection=privacy_contract["data_collection"],
                require_parameters=routing_contract["require_parameters"],
                zdr=privacy_contract["zdr"],
                latency_seconds=round(time.monotonic() - started, 6),
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                cost_usd=usage.get("cost"),
                request_id=response.headers.get("x-request-id") or safe_envelope.get("id"),
                finish_reason=safe_choice.get("finish_reason"),
                attempts=attempt,
            )

        try:
            envelope = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            call = completed_call(response_bytes=response.content)
            raise ProviderResponseValidationError(call=call, code="invalid_json") from None
        if not isinstance(envelope, dict):
            call = completed_call(response_bytes=response.content)
            raise ProviderResponseValidationError(call=call, code="invalid_json") from None
        choice: dict[str, Any] | None = None
        content: Any = envelope
        try:
            choice = envelope["choices"][0]
            if not isinstance(choice, dict):
                raise TypeError("choice is not an object")
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
            if message.get("refusal"):
                safe_refusal = str(message["refusal"]).replace(self._api_key, "[REDACTED]")
                raise ValueError(f"model refused structured extraction: {safe_refusal}")
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            payload = json.loads(content) if isinstance(content, str) else content
            if not isinstance(payload, dict):
                raise TypeError("structured response is not a JSON object")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            call = completed_call(
                response_bytes=_fingerprint_bytes(content, fallback=response.content),
                envelope=envelope,
                choice=choice,
            )
            raise ProviderResponseValidationError(call=call, code="invalid_json") from None
        raw_response = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        call = completed_call(
            response_bytes=raw_response,
            envelope=envelope,
            choice=choice,
        )
        try:
            validate(instance=payload, schema=schema)
        except ValidationError:
            raise ProviderResponseValidationError(call=call) from None
        return StructuredResponse(payload=payload, call=call)
