"""Minimal OpenRouter client with structured-output and complete run metadata."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import httpx
from jsonschema import ValidationError, validate
from pydantic import BaseModel, ConfigDict, Field, model_validator

ProviderValidationCode = Literal["invalid_json", "schema_validation", "wire_validation"]
CompletionTokenParameter = Literal["max_tokens", "max_completion_tokens"]

# OpenRouter's response-side provider label is untrusted text.  Public artifacts may
# retain only this deliberately frozen vocabulary; anything else is represented by a
# bounded disposition and omitted.  Keep this set independent of the live catalogue so
# rebuilding an artifact cannot silently change its disclosure boundary.
PUBLIC_PROVIDER_LABELS: frozenset[str] = frozenset(
    {
        "Amazon Bedrock",
        "Anthropic",
        "Azure",
        "Cerebras",
        "Chutes",
        "Cloudflare",
        "DeepInfra",
        "Fireworks",
        "Google",
        "Google AI Studio",
        "Groq",
        "Hyperbolic",
        "Lambda",
        "Lepton",
        "Mistral",
        "Nebius",
        "NVIDIA",
        "Novita",
        "OpenAI",
        "SambaNova",
        "Together",
        "xAI",
    }
)


class ProviderPayloadValidationKeyword(StrEnum):
    """Stable, payload-blind diagnostics for invalid structured responses."""

    OUTER_ENVELOPE_INVALID_JSON = "outer_envelope_invalid_json"
    OUTER_ENVELOPE_NON_OBJECT = "outer_envelope_non_object"
    OUTER_ENVELOPE_MALFORMED = "outer_envelope_malformed"
    STRUCTURED_CONTENT_REFUSAL = "structured_content_refusal"
    STRUCTURED_CONTENT_MISSING = "structured_content_missing"
    STRUCTURED_CONTENT_NULL = "structured_content_null"
    STRUCTURED_CONTENT_MALFORMED_LIST = "structured_content_malformed_list"
    STRUCTURED_CONTENT_INVALID_JSON = "structured_content_invalid_json"
    STRUCTURED_CONTENT_NON_OBJECT = "structured_content_non_object"


def completion_token_parameter_for_model(model: str) -> CompletionTokenParameter:
    """Return the OpenRouter token-limit field supported by the model family.

    OpenRouter's ZDR endpoint catalogue advertises OpenAI endpoints with
    ``max_completion_tokens`` rather than the legacy ``max_tokens`` field.
    With ``require_parameters=true``, sending the legacy field makes an
    otherwise compatible OpenAI route ineligible.  The mapping is derived from
    the exact persisted model ID, so call metadata plus the bound code version
    reconstructs the wire request without mutable catalogue state.
    """

    normalized = model.strip().removeprefix("~")
    if not normalized or "/" not in normalized:
        raise ValueError("OpenRouter model must be a non-empty author/model ID")
    author, _, slug = normalized.partition("/")
    if not author or not slug:
        raise ValueError("OpenRouter model must be a non-empty author/model ID")
    return "max_completion_tokens" if author.casefold() == "openai" else "max_tokens"


REASONING_DISABLED = "none"
"""`reasoning_effort` value that turns provider reasoning off instead of lowering it."""


class ProviderCall(BaseModel):
    """Secret-free metadata for one provider request."""

    model_config = ConfigDict(extra="forbid")
    provider: Literal["openrouter"] = "openrouter"
    model_requested: str = Field(pattern=r"^~?[^/\s]+/[^/\s]+$", strict=True)
    model_returned: str | None = None
    provider_returned: str | None = None
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temperature: float | None = Field(allow_inf_nan=False, strict=True)
    reasoning_effort: str | None = Field(default=None, min_length=1, strict=True)
    max_tokens: int = Field(ge=1, strict=True)
    completion_token_parameter: CompletionTokenParameter
    seed: int | None = Field(ge=0, strict=True)
    response_format: Literal["json_schema"] = "json_schema"
    schema_name: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, strict=True)
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_strict: Literal[True] = True
    data_collection: Literal["allow", "deny"] = "deny"
    require_parameters: bool = Field(default=False, strict=True)
    zdr: bool = Field(default=True, strict=True)
    latency_seconds: float = Field(ge=0, allow_inf_nan=False, strict=True)
    input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)
    reasoning_tokens: int | None = Field(default=None, ge=0, strict=True)
    total_tokens: int | None = Field(default=None, ge=0, strict=True)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)
    request_id: str | None = None
    finish_reason: str | None = None
    attempts: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def _validate_completion_token_parameter(self) -> ProviderCall:
        """Reject telemetry whose persisted wire field disagrees with its model."""

        expected = completion_token_parameter_for_model(self.model_requested)
        if self.completion_token_parameter != expected:
            raise ValueError("completion_token_parameter does not match model_requested")
        return self


class ProviderResponseValidationError(ValueError):
    """A completed provider call whose response failed secret-free local validation."""

    def __init__(
        self,
        *,
        call: ProviderCall,
        code: ProviderValidationCode = "schema_validation",
        validation_path: tuple[str | int, ...] = (),
        validation_keyword: str | None = None,
    ) -> None:
        messages = {
            "invalid_json": "OpenRouter response did not contain valid structured JSON",
            "schema_validation": "OpenRouter response did not satisfy requested JSON schema",
            "wire_validation": "Provider response did not satisfy local WireExtraction validation",
        }
        super().__init__(messages[code])
        self.call = call
        self.code = code
        self.validation_path = validation_path
        self.validation_keyword = validation_keyword


class ProviderRequestRejectedError(RuntimeError):
    """A non-retryable HTTP rejection whose response body is intentionally discarded."""

    def __init__(self, *, status_code: int) -> None:
        super().__init__(f"OpenRouter request rejected with HTTP {status_code}")
        self.status_code = status_code


@dataclass(frozen=True)
class StructuredResponse:
    payload: dict[str, Any]
    call: ProviderCall


def require_exact_returned_model(
    response: StructuredResponse,
    *,
    requested_model: str,
) -> StructuredResponse:
    """Reject a completed response routed to any model other than the exact request."""

    if response.call.model_returned != requested_model:
        raise ProviderResponseValidationError(
            call=response.call,
            code="wire_validation",
            validation_path=("model",),
            validation_keyword="returned_model_mismatch",
        )
    return response


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
    max_tokens: int | None = None,
    completion_token_parameter: CompletionTokenParameter | None = None,
) -> dict[str, Any]:
    token_fields_bound = max_tokens is not None or completion_token_parameter is not None
    if token_fields_bound and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
        or completion_token_parameter not in {"max_tokens", "max_completion_tokens"}
    ):
        raise ValueError("max_tokens and completion_token_parameter must be bound together")
    contract: dict[str, Any] = {
        "schema_version": (
            "provider-request-contract/0.2"
            if token_fields_bound
            else "provider-request-contract/0.1"
        ),
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
    if token_fields_bound:
        contract["max_tokens"] = max_tokens
        contract["completion_token_parameter"] = completion_token_parameter
    return contract


def structured_request_contract(
    *,
    schema_name: str,
    schema: dict[str, Any],
    seed: int | None,
    require_parameters: bool = False,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build a versioned, secret-free schema or materialized request contract.

    Call-independent schema helpers may omit both ``model`` and ``max_tokens``
    and retain the 0.1 contract.  Actual provider dispatch and budget
    reservation paths must provide both; their 0.2 contract binds the logical
    limit to the exact model-specific OpenRouter wire parameter.
    """

    if (model is None) != (max_tokens is None):
        raise ValueError("model and max_tokens must be provided together")
    completion_token_parameter = (
        None if model is None else completion_token_parameter_for_model(model)
    )

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
        max_tokens=max_tokens,
        completion_token_parameter=completion_token_parameter,
    )


def structured_request_contract_from_call(call: ProviderCall) -> dict[str, Any]:
    """Reconstruct a request contract from completed-call telemetry."""

    expected_parameter = completion_token_parameter_for_model(call.model_requested)
    if call.completion_token_parameter != expected_parameter:
        raise ValueError("provider call completion-token materialization is invalid")
    return _request_contract(
        data_collection=call.data_collection,
        zdr=call.zdr,
        response_format=call.response_format,
        schema_name=call.schema_name,
        schema_sha256=call.schema_sha256,
        schema_strict=call.schema_strict,
        seed=call.seed,
        require_parameters=call.require_parameters,
        max_tokens=call.max_tokens,
        completion_token_parameter=call.completion_token_parameter,
    )


def public_provider_call(call: ProviderCall) -> dict[str, Any]:
    """Project one completed call into the sole public telemetry representation.

    Request IDs are operational correlators and are never published, including as a
    stable digest.  Response-side model, provider, and finish metadata are untrusted:
    only an exact requested-model echo and frozen provider labels survive, while all
    other values collapse to bounded dispositions/categories.
    """

    if not isinstance(call, ProviderCall):
        raise TypeError("public provider-call projection requires typed ProviderCall")
    if call.provider != "openrouter":
        raise ValueError("public provider-call projection supports only openrouter")

    if call.model_returned is None:
        model_returned = None
        model_returned_disposition = "missing"
    elif call.model_returned == call.model_requested:
        model_returned = call.model_returned
        model_returned_disposition = "matches_requested"
    else:
        model_returned = None
        model_returned_disposition = "unrecognized_omitted"

    if call.provider_returned is None:
        provider_returned = None
        provider_returned_disposition = "missing"
    elif call.provider_returned in PUBLIC_PROVIDER_LABELS:
        provider_returned = call.provider_returned
        provider_returned_disposition = "known_label"
    else:
        provider_returned = None
        provider_returned_disposition = "unrecognized_omitted"

    finish_category = (
        "missing"
        if call.finish_reason is None
        else call.finish_reason
        if call.finish_reason in {"stop", "length", "error"}
        else "other"
    )
    return {
        "schema_version": "public-provider-call/0.1",
        "provider_requested": "openrouter",
        "model_requested": call.model_requested,
        "model_returned": model_returned,
        "model_returned_disposition": model_returned_disposition,
        "provider_returned": provider_returned,
        "provider_returned_disposition": provider_returned_disposition,
        "prompt_sha256": call.prompt_sha256,
        "response_sha256": call.response_sha256,
        "temperature": call.temperature,
        "reasoning_effort": call.reasoning_effort,
        "max_tokens": call.max_tokens,
        "completion_token_parameter": call.completion_token_parameter,
        "seed": call.seed,
        "response_format": call.response_format,
        "schema_name": call.schema_name,
        "schema_sha256": call.schema_sha256,
        "schema_strict": call.schema_strict,
        "request_contract": structured_request_contract_from_call(call),
        "data_collection": call.data_collection,
        "require_parameters": call.require_parameters,
        "zdr": call.zdr,
        "latency_seconds": call.latency_seconds,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "reasoning_tokens": call.reasoning_tokens,
        "total_tokens": call.total_tokens,
        "cost_usd": call.cost_usd,
        "finish_category": finish_category,
        "attempts": call.attempts,
        "request_id_observed": call.request_id is not None,
    }


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
        completion_token_parameter = completion_token_parameter_for_model(model)
        contract = structured_request_contract(
            schema_name=schema_name,
            schema=schema,
            seed=seed,
            require_parameters=effective_require_parameters,
            model=model,
            max_tokens=max_tokens,
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
        body[completion_token_parameter] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort == REASONING_DISABLED:
            # Requesting minimal reasoning effort does not disable reasoning. Use the
            # explicit switch when the caller requests no reasoning.
            body["reasoning"] = {"enabled": False, "exclude": True}
        elif reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
        if contract["seed"] is not None:
            body["seed"] = contract["seed"]
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/evaleval/proceedings-to-paper",
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
            completion_details = usage.get("completion_tokens_details") or {}
            if not isinstance(completion_details, dict):
                completion_details = {}

            def optional_nonnegative_int(value: Any) -> int | None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    return None
                return value

            def optional_nonnegative_number(value: Any) -> float | None:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    return None
                try:
                    parsed = float(value)
                except OverflowError:
                    return None
                return parsed if parsed >= 0 and math.isfinite(parsed) else None

            reasoning_tokens = optional_nonnegative_int(completion_details.get("reasoning_tokens"))
            safe_choice = choice or {}
            model_returned = safe_envelope.get("model")
            if not isinstance(model_returned, str):
                model_returned = None
            provider_returned = safe_envelope.get("provider")
            if not isinstance(provider_returned, str):
                provider_returned = None
            finish_reason = safe_choice.get("finish_reason")
            if not isinstance(finish_reason, str):
                finish_reason = None
            request_id = response.headers.get("x-request-id") or safe_envelope.get("id")
            if not isinstance(request_id, str):
                request_id = None
            return ProviderCall(
                model_requested=model,
                model_returned=model_returned,
                provider_returned=provider_returned,
                prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
                response_sha256=hashlib.sha256(response_bytes).hexdigest(),
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
                completion_token_parameter=completion_token_parameter,
                seed=contract["seed"],
                response_format=schema_contract["response_format"],
                schema_name=schema_contract["schema_name"],
                schema_sha256=schema_contract["schema_sha256"],
                schema_strict=schema_contract["schema_strict"],
                data_collection=privacy_contract["data_collection"],
                require_parameters=routing_contract["require_parameters"],
                zdr=privacy_contract["zdr"],
                latency_seconds=round(time.monotonic() - started, 6),
                input_tokens=optional_nonnegative_int(usage.get("prompt_tokens")),
                output_tokens=optional_nonnegative_int(usage.get("completion_tokens")),
                reasoning_tokens=reasoning_tokens,
                total_tokens=optional_nonnegative_int(usage.get("total_tokens")),
                cost_usd=optional_nonnegative_number(usage.get("cost")),
                request_id=request_id,
                finish_reason=finish_reason,
                attempts=attempt,
            )

        try:
            envelope = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            call = completed_call(response_bytes=response.content)
            raise ProviderResponseValidationError(
                call=call,
                code="invalid_json",
                validation_keyword=ProviderPayloadValidationKeyword.OUTER_ENVELOPE_INVALID_JSON,
            ) from None
        if not isinstance(envelope, dict):
            call = completed_call(response_bytes=response.content)
            raise ProviderResponseValidationError(
                call=call,
                code="invalid_json",
                validation_keyword=ProviderPayloadValidationKeyword.OUTER_ENVELOPE_NON_OBJECT,
            ) from None
        choice: dict[str, Any] | None = None
        content: Any = None
        try:
            choice = envelope["choices"][0]
            if not isinstance(choice, dict):
                raise TypeError("choice is not an object")
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
        except (KeyError, IndexError, TypeError):
            call = completed_call(
                response_bytes=response.content,
                envelope=envelope,
                choice=choice,
            )
            raise ProviderResponseValidationError(
                call=call,
                code="invalid_json",
                validation_keyword=ProviderPayloadValidationKeyword.OUTER_ENVELOPE_MALFORMED,
            ) from None
        if message.get("refusal"):
            call = completed_call(
                response_bytes=response.content,
                envelope=envelope,
                choice=choice,
            )
            raise ProviderResponseValidationError(
                call=call,
                code="invalid_json",
                validation_keyword=ProviderPayloadValidationKeyword.STRUCTURED_CONTENT_REFUSAL,
            )
        if "content" not in message:
            call = completed_call(
                response_bytes=response.content,
                envelope=envelope,
                choice=choice,
            )
            raise ProviderResponseValidationError(
                call=call,
                code="invalid_json",
                validation_keyword=ProviderPayloadValidationKeyword.STRUCTURED_CONTENT_MISSING,
            )
        content = message["content"]
        if content is None:
            call = completed_call(
                response_bytes=b"null",
                envelope=envelope,
                choice=choice,
            )
            raise ProviderResponseValidationError(
                call=call,
                code="invalid_json",
                validation_keyword=ProviderPayloadValidationKeyword.STRUCTURED_CONTENT_NULL,
            )
        if isinstance(content, list):
            if not content or any(
                not isinstance(item, dict) or not isinstance(item.get("text"), str)
                for item in content
            ):
                call = completed_call(
                    response_bytes=_fingerprint_bytes(content, fallback=response.content),
                    envelope=envelope,
                    choice=choice,
                )
                raise ProviderResponseValidationError(
                    call=call,
                    code="invalid_json",
                    validation_keyword=(
                        ProviderPayloadValidationKeyword.STRUCTURED_CONTENT_MALFORMED_LIST
                    ),
                )
            fragments = [item["text"] for item in content]
            content = "".join(fragments)
        if isinstance(content, str):
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                call = completed_call(
                    response_bytes=content.encode("utf-8"),
                    envelope=envelope,
                    choice=choice,
                )
                raise ProviderResponseValidationError(
                    call=call,
                    code="invalid_json",
                    validation_keyword=(
                        ProviderPayloadValidationKeyword.STRUCTURED_CONTENT_INVALID_JSON
                    ),
                ) from None
        else:
            payload = content
        if not isinstance(payload, dict):
            call = completed_call(
                response_bytes=_fingerprint_bytes(payload, fallback=response.content),
                envelope=envelope,
                choice=choice,
            )
            raise ProviderResponseValidationError(
                call=call,
                code="invalid_json",
                validation_keyword=ProviderPayloadValidationKeyword.STRUCTURED_CONTENT_NON_OBJECT,
            )
        raw_response = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        call = completed_call(
            response_bytes=raw_response,
            envelope=envelope,
            choice=choice,
        )
        try:
            validate(instance=payload, schema=schema)
        except ValidationError as error:
            raise ProviderResponseValidationError(
                call=call,
                validation_path=tuple(error.absolute_path),
                validation_keyword=(str(error.validator) if error.validator is not None else None),
            ) from None
        return StructuredResponse(payload=payload, call=call)
