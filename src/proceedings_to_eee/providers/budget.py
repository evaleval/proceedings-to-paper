"""Append-only, resumable budgets for structured provider invocations.

The budget is deliberately enforced outside the OpenRouter transport.  Every pipeline
stage receives the same duck-typed client, so one wrapper covers legacy extraction,
row enumeration, and independent verification without letting a stage maintain its own
counter.  A reservation is durably appended before dispatch.  Provider-reported actual
cost replaces it after a completed call; failed calls, interrupted processes, and calls
without cost telemetry retain the reservation and therefore cannot reset committed spend
on resume.  A separately replaced and fsynced head record detects ledger-only truncation
or suffix rollback.  It is not an authenticity mechanism: an attacker able to roll back
both the ledger and its head can evade detection.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from weakref import WeakValueDictionary

from proceedings_to_eee.providers.openrouter import (
    PUBLIC_PROVIDER_LABELS,
    ProviderCall,
    ProviderRequestRejectedError,
    ProviderResponseValidationError,
    StructuredResponse,
    completion_token_parameter_for_model,
    structured_request_contract,
    structured_request_contract_from_call,
)

_EVENT_SCHEMA_VERSION = "provider-budget-ledger-event/0.3"
_CONTRACT_SCHEMA_VERSION = "provider-budget-contract/0.3"
_HEAD_SCHEMA_VERSION = "provider-budget-head/0.1"

_LOCAL_LOCKS_GUARD = threading.Lock()
_LOCAL_LOCKS: WeakValueDictionary[str, threading.RLock] = WeakValueDictionary()

_LIMIT_FIELDS = {
    "max_structured_calls",
    "max_cost_usd",
    "cost_reservation_per_call_usd",
    "accounting",
    "telemetry_semantics",
}
_BASE_EVENT_FIELDS = {
    "schema_version",
    "sequence",
    "event_type",
    "previous_event_sha256",
    "event_sha256",
    "contract_sha256",
    "timestamp",
}
_EVENT_FIELDS = {
    "contract": _BASE_EVENT_FIELDS | {"contract"},
    "budget_amendment": _BASE_EVENT_FIELDS | {"limits", "previous_limits_sha256", "limits_sha256"},
    "reservation": _BASE_EVENT_FIELDS | {"invocation_id", "request", "reserved_cost_usd"},
    "completion": _BASE_EVENT_FIELDS | {"invocation_id", "outcome", "provider_call", "failure"},
}
_REQUEST_FIELDS = {
    "model_requested",
    "schema_name",
    "local_schema_sha256",
    "provider_schema_sha256",
    "request_contract",
    "request_contract_sha256",
    "request_sha256",
    "prompt_sha256",
    "system_sha256",
    "user_sha256",
    "temperature",
    "reasoning_effort",
    "max_tokens",
    "completion_token_parameter",
    "seed",
    "require_parameters",
}
_PROVIDER_CALL_FIELDS = {
    "model_requested",
    "model_returned",
    "model_returned_disposition",
    "model_returned_sha256",
    "provider_returned",
    "provider_returned_disposition",
    "provider_returned_sha256",
    "prompt_sha256",
    "response_sha256",
    "temperature",
    "reasoning_effort",
    "max_tokens",
    "completion_token_parameter",
    "seed",
    "response_format",
    "schema_name",
    "schema_sha256",
    "schema_strict",
    "data_collection",
    "require_parameters",
    "zdr",
    "latency_seconds",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "attempts",
    "retries",
    "finish_reason",
}
_FAILURE_FIELDS = {"terminal_class", "http_status", "known_attempts"}
_MODEL_RETURNED_DISPOSITIONS = {"missing", "matches_requested", "hashed_untrusted"}
_PROVIDER_RETURNED_DISPOSITIONS = {"missing", "known_label", "hashed_untrusted"}
_SAFE_FINISH_REASONS = {"stop", "length", "content_filter", "tool_calls", "error", "other"}
_OUTCOMES = {
    "success",
    "provider_response_invalid_json",
    "provider_response_schema_validation",
    "provider_response_wire_validation",
    "provider_request_rejected",
    "technical_failure",
}


class ProviderBudgetError(Exception):
    """Base class for typed budget failures."""


class ProviderBudgetContractError(ProviderBudgetError):
    """An existing ledger is malformed or belongs to a different run contract."""


class ProviderBudgetExhausted(ProviderBudgetError):
    """A provider call was not dispatched because a hard corpus limit was reached."""

    def __init__(
        self, *, reason: Literal["structured_call_limit", "cost_limit"], summary: dict[str, Any]
    ):
        super().__init__(f"provider budget stopped before dispatch: {reason}")
        self.code = "provider_budget_exhausted"
        self.reason = reason
        self.summary = summary


@dataclass(frozen=True)
class ProviderBudgetLimits:
    """Finite corpus limits and the conservative amount charged before every call."""

    max_structured_calls: int
    max_cost_usd: float
    cost_reservation_per_call_usd: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_structured_calls, bool)
            or not isinstance(self.max_structured_calls, int)
            or self.max_structured_calls < 1
        ):
            raise ValueError("provider max structured calls must be a positive integer")
        for name, value in (
            ("max_cost_usd", self.max_cost_usd),
            ("cost_reservation_per_call_usd", self.cost_reservation_per_call_usd),
        ):
            if not _is_non_negative_number(value) or value <= 0:
                raise ValueError(f"provider {name} must be positive and finite")
        if self.cost_reservation_per_call_usd > self.max_cost_usd:
            raise ValueError("provider per-call reservation cannot exceed the corpus cost limit")

    def contract_payload(self) -> dict[str, Any]:
        return {
            "max_structured_calls": self.max_structured_calls,
            "max_cost_usd": self.max_cost_usd,
            "cost_reservation_per_call_usd": self.cost_reservation_per_call_usd,
            "accounting": (
                "each invocation reserves cost before dispatch; committed cost is provider-"
                "reported actual cost when available and otherwise the non-refundable reservation"
            ),
            "telemetry_semantics": (
                "completed responses expose provider-reported tokens and cost plus transport "
                "attempts; rejected, interrupted, and transport-failed requests may lack that "
                "metadata, so reported actual cost and transport-attempt aggregates are lower "
                "bounds while committed cost remains conservative for missing telemetry"
            ),
        }


def provider_budget_contract(
    *,
    corpus_binding: dict[str, str],
    provider_run_contract: dict[str, Any],
    limits: ProviderBudgetLimits,
) -> dict[str, Any]:
    """Build a secret-free contract whose digests bind caller-supplied run metadata."""

    try:
        corpus_binding_sha256 = hashlib.sha256(_canonical_bytes(corpus_binding)).hexdigest()
        provider_run_contract_sha256 = hashlib.sha256(
            _canonical_bytes(provider_run_contract)
        ).hexdigest()
    except (TypeError, ValueError) as error:
        raise ValueError("provider budget run metadata must be finite canonical JSON") from error

    return {
        "schema_version": _CONTRACT_SCHEMA_VERSION,
        "corpus_binding_sha256": corpus_binding_sha256,
        "provider_run_contract_sha256": provider_run_contract_sha256,
        "limits": limits.contract_payload(),
    }


def _contract_identity(contract: dict[str, Any]) -> dict[str, Any]:
    """Return immutable run/accounting fields while allowing monotonic ceiling increases."""

    if (
        set(contract)
        != {
            "schema_version",
            "corpus_binding_sha256",
            "provider_run_contract_sha256",
            "limits",
        }
        or contract.get("schema_version") != _CONTRACT_SCHEMA_VERSION
    ):
        raise ProviderBudgetContractError("provider budget contract is invalid")
    if not _is_sha256(contract.get("corpus_binding_sha256")) or not _is_sha256(
        contract.get("provider_run_contract_sha256")
    ):
        raise ProviderBudgetContractError("provider budget run binding is invalid")
    limits = contract.get("limits")
    if not isinstance(limits, dict):
        raise ProviderBudgetContractError("provider budget contract limits are invalid")
    return {
        "schema_version": contract.get("schema_version"),
        "corpus_binding_sha256": contract.get("corpus_binding_sha256"),
        "provider_run_contract_sha256": contract.get("provider_run_contract_sha256"),
        "accounting": {
            key: limits.get(key)
            for key in (
                "cost_reservation_per_call_usd",
                "accounting",
                "telemetry_semantics",
            )
        },
    }


def _limits_from_contract(contract: dict[str, Any]) -> ProviderBudgetLimits:
    limits = contract.get("limits")
    if not isinstance(limits, dict) or set(limits) != _LIMIT_FIELDS:
        raise ProviderBudgetContractError("provider budget contract limits are invalid")
    try:
        parsed = ProviderBudgetLimits(
            max_structured_calls=limits["max_structured_calls"],
            max_cost_usd=limits["max_cost_usd"],
            cost_reservation_per_call_usd=limits["cost_reservation_per_call_usd"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProviderBudgetContractError("provider budget contract limits are invalid") from error
    if limits != parsed.contract_payload():
        raise ProviderBudgetContractError("provider budget accounting contract is invalid")
    return parsed


def _decimal(value: float | str | int) -> Decimal:
    return Decimal(str(value))


def _json_decimal_number(value: Decimal, *, round_digits: int | None = None) -> float | str:
    """Return a finite JSON number, or an exact decimal string beyond float range."""

    converted = float(value)
    if math.isfinite(converted):
        return round(converted, round_digits) if round_digits is not None else converted
    return format(value, "f")


def _canonical_bytes(value: Any) -> bytes:
    """Serialize one compact ledger record without accepting non-JSON floats."""

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_without_duplicate_keys(raw: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ProviderBudgetContractError("provider budget ledger has duplicate keys")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ProviderBudgetContractError(
            f"provider budget JSON contains invalid numeric constant {value}"
        )

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except ProviderBudgetContractError:
        raise
    except (ValueError, UnicodeDecodeError) as error:
        raise ProviderBudgetContractError("provider budget ledger is not valid JSONL") from error


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _is_non_negative_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_optional_non_negative_int(value: Any) -> bool:
    return value is None or (not isinstance(value, bool) and isinstance(value, int) and value >= 0)


def _hashed_event(payload: dict[str, Any]) -> dict[str, Any]:
    event = dict(payload)
    event["event_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return event


def _event_hash_is_valid(event: dict[str, Any]) -> bool:
    digest = event.get("event_sha256")
    payload = {key: value for key, value in event.items() if key != "event_sha256"}
    try:
        expected = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    except (TypeError, ValueError):
        return False
    return isinstance(digest, str) and digest == expected


class BudgetedProviderClient:
    """Structured-chat proxy backed by a hash-chained append-only corpus ledger."""

    def __init__(
        self,
        *,
        client: Any,
        ledger_path: Path,
        contract: dict[str, Any],
        limits: ProviderBudgetLimits,
    ) -> None:
        self._client = client
        self._path = ledger_path
        self._head_path = Path(f"{ledger_path}.head.json")
        self._lock_path = Path(f"{ledger_path}.lock")
        self._contract = contract
        self._limits = limits
        if contract.get("limits") != limits.contract_payload():
            raise ValueError("provider budget limits do not match the supplied contract")
        self._contract_sha256 = hashlib.sha256(
            _canonical_bytes(_contract_identity(contract))
        ).hexdigest()
        self._events: list[dict[str, Any]] = []
        self._reservations: dict[int, dict[str, Any]] = {}
        self._completions: dict[int, dict[str, Any]] = {}
        self._ledger_limits: ProviderBudgetLimits | None = None
        self._poisoned = False
        try:
            self._initialize_or_resume()
        except ProviderBudgetContractError:
            self._poisoned = True
            raise

    @property
    def contract_sha256(self) -> str:
        return self._contract_sha256

    @property
    def ledger_path(self) -> Path:
        return self._path

    @property
    def head_path(self) -> Path:
        return self._head_path

    def _ensure_usable(self) -> None:
        if self._poisoned:
            raise ProviderBudgetContractError(
                "provider budget client is poisoned after an integrity or durability failure"
            )

    def _poison(self, error: ProviderBudgetContractError) -> None:
        self._poisoned = True
        raise error

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short provider budget write")
            offset += written

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError("provider budget path is not a directory")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _ensure_directory_tree_durable(cls, path: Path) -> None:
        """Create each missing directory and persist every new parent entry."""

        missing: list[Path] = []
        cursor = path
        while not cursor.exists():
            if cursor.is_symlink():
                raise ProviderBudgetContractError(
                    "provider budget directory tree must not contain symbolic links"
                )
            missing.append(cursor)
            parent = cursor.parent
            if parent == cursor:
                raise ProviderBudgetContractError(
                    "provider budget directory tree has no existing ancestor"
                )
            cursor = parent
        if cursor.is_symlink() or not cursor.is_dir():
            raise ProviderBudgetContractError(
                "provider budget directory tree must contain only directories"
            )
        for directory in reversed(missing):
            try:
                os.mkdir(directory, 0o700)
            except FileExistsError:
                if directory.is_symlink() or not directory.is_dir():
                    raise ProviderBudgetContractError(
                        "provider budget directory tree must contain only directories"
                    ) from None
            cls._fsync_directory(directory.parent)

    @staticmethod
    def _read_regular_bytes(path: Path, *, label: str) -> bytes:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise ProviderBudgetContractError(
                f"provider budget {label} could not be opened safely"
            ) from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ProviderBudgetContractError(f"provider budget {label} must be a regular file")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        except OSError as error:
            raise ProviderBudgetContractError(
                f"provider budget {label} could not be read"
            ) from error
        finally:
            os.close(descriptor)

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        """Serialize threads and processes without spanning provider dispatch."""

        try:
            self._ensure_directory_tree_durable(self._path.parent)
        except OSError as error:
            raise ProviderBudgetContractError(
                "provider budget directory creation could not be persisted"
            ) from error
        lock_key = str(self._lock_path.resolve(strict=False))
        with _LOCAL_LOCKS_GUARD:
            local_lock = _LOCAL_LOCKS.setdefault(lock_key, threading.RLock())
        with local_lock:
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            lock_existed = self._lock_path.exists() or self._lock_path.is_symlink()
            try:
                descriptor = os.open(self._lock_path, flags, 0o600)
            except OSError as error:
                raise ProviderBudgetContractError(
                    "provider budget lock could not be opened safely"
                ) from error
            locked = False
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ProviderBudgetContractError(
                        "provider budget lock must be a singly linked regular file"
                    )
                if not lock_existed:
                    self._fsync_directory(self._lock_path.parent)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
                yield
            except OSError as error:
                raise ProviderBudgetContractError("provider budget lock failed") from error
            finally:
                cleanup_error: OSError | None = None
                try:
                    if locked:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError as error:
                    cleanup_error = error
                try:
                    os.close(descriptor)
                except OSError as error:
                    cleanup_error = cleanup_error or error
                if cleanup_error is not None:
                    raise ProviderBudgetContractError(
                        "provider budget lock release failed"
                    ) from cleanup_error

    def _initialize_or_resume(self) -> None:
        with self._exclusive_lock():
            ledger_exists = self._path.exists() or self._path.is_symlink()
            head_exists = self._head_path.exists() or self._head_path.is_symlink()
            if ledger_exists != head_exists:
                raise ProviderBudgetContractError(
                    "provider budget ledger and head must exist together"
                )
            if not ledger_exists:
                header = self._next_event(
                    event_type="contract",
                    contract_sha256=self._contract_sha256,
                    contract=self._contract,
                )
                self._append_locked(header)
            self._load_and_validate_locked()
            self._amend_limits_if_needed_locked()

    @staticmethod
    def _validate_timestamp(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return False
        return parsed.tzinfo is not None

    @staticmethod
    def _validate_request(request: Any) -> dict[str, Any]:
        if not isinstance(request, dict) or set(request) != _REQUEST_FIELDS:
            raise ProviderBudgetContractError("provider budget request fingerprint is invalid")
        hashes = (
            "local_schema_sha256",
            "provider_schema_sha256",
            "request_contract_sha256",
            "request_sha256",
            "prompt_sha256",
            "system_sha256",
            "user_sha256",
        )
        if any(not _is_sha256(request.get(name)) for name in hashes):
            raise ProviderBudgetContractError("provider budget request hash is invalid")
        if not isinstance(request.get("model_requested"), str) or not request["model_requested"]:
            raise ProviderBudgetContractError("provider budget request model is invalid")
        try:
            expected_completion_token_parameter = completion_token_parameter_for_model(
                request["model_requested"]
            )
        except ValueError as error:
            raise ProviderBudgetContractError("provider budget request model is invalid") from error
        if not isinstance(request.get("schema_name"), str) or not request["schema_name"]:
            raise ProviderBudgetContractError("provider budget request schema name is invalid")
        if (
            isinstance(request.get("max_tokens"), bool)
            or not isinstance(request.get("max_tokens"), int)
            or request["max_tokens"] < 1
            or request.get("completion_token_parameter")
            not in {"max_tokens", "max_completion_tokens"}
            or request["completion_token_parameter"] != expected_completion_token_parameter
            or (
                request.get("temperature") is not None
                and not _is_non_negative_number(request["temperature"])
            )
            or (
                request.get("reasoning_effort") is not None
                and not isinstance(request["reasoning_effort"], str)
            )
            or not _is_optional_non_negative_int(request.get("seed"))
            or not isinstance(request.get("require_parameters"), bool)
        ):
            raise ProviderBudgetContractError("provider budget request settings are invalid")
        contract = request.get("request_contract")
        if not isinstance(contract, dict) or set(contract) != {
            "schema_version",
            "privacy",
            "routing",
            "schema",
            "seed",
            "max_tokens",
            "completion_token_parameter",
        }:
            raise ProviderBudgetContractError("provider request contract is invalid")
        if (
            contract.get("schema_version") != "provider-request-contract/0.2"
            or contract.get("privacy") != {"data_collection": "deny", "zdr": True}
            or contract.get("routing") != {"require_parameters": request["require_parameters"]}
            or contract.get("schema")
            != {
                "response_format": "json_schema",
                "schema_name": request["schema_name"],
                "schema_sha256": request["provider_schema_sha256"],
                "schema_strict": True,
            }
            or contract.get("seed") != request["seed"]
            or contract.get("max_tokens") != request["max_tokens"]
            or contract.get("completion_token_parameter") != request["completion_token_parameter"]
        ):
            raise ProviderBudgetContractError("provider request contract does not match request")
        if (
            request["request_contract_sha256"]
            != hashlib.sha256(_canonical_bytes(contract)).hexdigest()
        ):
            raise ProviderBudgetContractError("provider request contract hash is invalid")
        request_payload = {key: value for key, value in request.items() if key != "request_sha256"}
        if (
            request["request_sha256"]
            != hashlib.sha256(_canonical_bytes(request_payload)).hexdigest()
        ):
            raise ProviderBudgetContractError("provider budget request hash is invalid")
        return request

    @staticmethod
    def _validate_provider_call(call: Any, *, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(call, dict) or set(call) != _PROVIDER_CALL_FIELDS:
            raise ProviderBudgetContractError("provider call telemetry shape is invalid")
        if (
            not isinstance(call.get("model_requested"), str)
            or (
                call.get("model_returned") is not None
                and not isinstance(call["model_returned"], str)
            )
            or (
                call.get("provider_returned") is not None
                and not isinstance(call["provider_returned"], str)
            )
            or call.get("model_returned_disposition") not in _MODEL_RETURNED_DISPOSITIONS
            or (
                call.get("model_returned_sha256") is not None
                and not _is_sha256(call["model_returned_sha256"])
            )
            or call.get("provider_returned_disposition") not in _PROVIDER_RETURNED_DISPOSITIONS
            or (
                call.get("provider_returned_sha256") is not None
                and not _is_sha256(call["provider_returned_sha256"])
            )
            or not _is_sha256(call.get("prompt_sha256"))
            or not _is_sha256(call.get("response_sha256"))
            or not _is_sha256(call.get("schema_sha256"))
            or not isinstance(call.get("schema_name"), str)
            or call.get("response_format") != "json_schema"
            or call.get("schema_strict") is not True
            or call.get("data_collection") != "deny"
            or not isinstance(call.get("require_parameters"), bool)
            or call.get("zdr") is not True
            or not _is_non_negative_number(call.get("latency_seconds"))
            or (call.get("cost_usd") is not None and not _is_non_negative_number(call["cost_usd"]))
            or any(
                not _is_optional_non_negative_int(call.get(name))
                for name in (
                    "input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                )
            )
            or isinstance(call.get("attempts"), bool)
            or not isinstance(call.get("attempts"), int)
            or call["attempts"] < 1
            or isinstance(call.get("retries"), bool)
            or not isinstance(call.get("retries"), int)
            or call["retries"] < 0
            or call.get("retries") != call["attempts"] - 1
            or isinstance(call.get("max_tokens"), bool)
            or not isinstance(call.get("max_tokens"), int)
            or call["max_tokens"] < 1
            or call.get("completion_token_parameter") not in {"max_tokens", "max_completion_tokens"}
            or (
                call.get("temperature") is not None
                and not _is_non_negative_number(call["temperature"])
            )
            or (
                call.get("reasoning_effort") is not None
                and not isinstance(call["reasoning_effort"], str)
            )
            or not _is_optional_non_negative_int(call.get("seed"))
            or call.get("finish_reason") not in _SAFE_FINISH_REASONS | {None}
        ):
            raise ProviderBudgetContractError("provider call telemetry values are invalid")
        model_disposition = call["model_returned_disposition"]
        if (
            (
                model_disposition == "missing"
                and (
                    call["model_returned"] is not None or call["model_returned_sha256"] is not None
                )
            )
            or (
                model_disposition == "matches_requested"
                and (
                    call["model_returned"] != request["model_requested"]
                    or call["model_returned_sha256"] != _text_sha256(call["model_returned"])
                )
            )
            or (
                model_disposition == "hashed_untrusted"
                and (call["model_returned"] is not None or not call["model_returned_sha256"])
            )
        ):
            raise ProviderBudgetContractError("returned-model telemetry is not secret-safe")
        provider_disposition = call["provider_returned_disposition"]
        if (
            (
                provider_disposition == "missing"
                and (
                    call["provider_returned"] is not None
                    or call["provider_returned_sha256"] is not None
                )
            )
            or (
                provider_disposition == "known_label"
                and (
                    call["provider_returned"] not in PUBLIC_PROVIDER_LABELS
                    or call["provider_returned_sha256"] != _text_sha256(call["provider_returned"])
                )
            )
            or (
                provider_disposition == "hashed_untrusted"
                and (call["provider_returned"] is not None or not call["provider_returned_sha256"])
            )
        ):
            raise ProviderBudgetContractError("returned-provider telemetry is not secret-safe")
        if (
            call["model_requested"] != request["model_requested"]
            or call["prompt_sha256"] != request["prompt_sha256"]
            or call["temperature"] != request["temperature"]
            or call["reasoning_effort"] != request["reasoning_effort"]
            or call["max_tokens"] != request["max_tokens"]
            or call["completion_token_parameter"] != request["completion_token_parameter"]
            or call["seed"] != request["seed"]
            or call["schema_name"] != request["schema_name"]
            or call["schema_sha256"] != request["provider_schema_sha256"]
            or call["require_parameters"] != request["require_parameters"]
        ):
            raise ProviderBudgetContractError("provider call telemetry is not request-bound")
        try:
            typed_call = ProviderCall.model_validate(
                {
                    key: value
                    for key, value in call.items()
                    if key
                    not in {
                        "retries",
                        "model_returned_disposition",
                        "model_returned_sha256",
                        "provider_returned_disposition",
                        "provider_returned_sha256",
                    }
                }
            )
        except ValueError as error:
            raise ProviderBudgetContractError("provider call telemetry is invalid") from error
        if structured_request_contract_from_call(typed_call) != request["request_contract"]:
            raise ProviderBudgetContractError("provider call contract is not request-bound")
        return call

    @staticmethod
    def _validate_failure(failure: Any, *, outcome: str, call: dict[str, Any] | None) -> None:
        if outcome == "success":
            if failure is not None or call is None:
                raise ProviderBudgetContractError("successful provider completion is invalid")
            return
        if not isinstance(failure, dict) or set(failure) != _FAILURE_FIELDS:
            raise ProviderBudgetContractError("provider failure telemetry shape is invalid")
        terminal = failure.get("terminal_class")
        status = failure.get("http_status")
        attempts = failure.get("known_attempts")
        if not _is_optional_non_negative_int(attempts) or attempts == 0:
            raise ProviderBudgetContractError("provider failure attempt telemetry is invalid")
        if outcome.startswith("provider_response_"):
            valid = terminal == "response_validation" and status is None and call is not None
        elif outcome == "provider_request_rejected":
            valid = (
                terminal == "request_rejected"
                and call is None
                and not isinstance(status, bool)
                and isinstance(status, int)
                and 400 <= status <= 599
            )
        else:
            valid = terminal == "transport_or_client_failure" and status is None and call is None
        if not valid:
            raise ProviderBudgetContractError("provider failure telemetry is inconsistent")
        if call is not None and attempts != call["attempts"]:
            raise ProviderBudgetContractError("provider failure attempts do not match call")

    def _validate_head_locked(self, *, ledger_bytes: bytes, final_event: dict[str, Any]) -> None:
        head_bytes = self._read_regular_bytes(self._head_path, label="head")
        if not head_bytes.endswith(b"\n") or head_bytes.count(b"\n") != 1:
            raise ProviderBudgetContractError("provider budget head is incomplete")
        try:
            head_text = head_bytes[:-1].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProviderBudgetContractError("provider budget head is not UTF-8") from error
        head = _json_without_duplicate_keys(head_text)
        expected_fields = {
            "schema_version",
            "contract_sha256",
            "final_sequence",
            "final_event_sha256",
            "ledger_size_bytes",
            "ledger_sha256",
        }
        if (
            not isinstance(head, dict)
            or set(head) != expected_fields
            or head.get("schema_version") != _HEAD_SCHEMA_VERSION
            or not _is_sha256(head.get("contract_sha256"))
            or head.get("contract_sha256") != self._contract_sha256
            or isinstance(head.get("final_sequence"), bool)
            or not isinstance(head.get("final_sequence"), int)
            or head["final_sequence"] < 0
            or head.get("final_sequence") != final_event["sequence"]
            or not _is_sha256(head.get("final_event_sha256"))
            or head.get("final_event_sha256") != final_event["event_sha256"]
            or isinstance(head.get("ledger_size_bytes"), bool)
            or not isinstance(head.get("ledger_size_bytes"), int)
            or head["ledger_size_bytes"] < 1
            or head.get("ledger_size_bytes") != len(ledger_bytes)
            or not _is_sha256(head.get("ledger_sha256"))
            or head.get("ledger_sha256") != hashlib.sha256(ledger_bytes).hexdigest()
        ):
            raise ProviderBudgetContractError("provider budget head does not match ledger")

    def _load_and_validate_locked(self) -> None:
        ledger_exists = self._path.exists() or self._path.is_symlink()
        head_exists = self._head_path.exists() or self._head_path.is_symlink()
        if not ledger_exists or not head_exists:
            raise ProviderBudgetContractError("provider budget ledger and head must exist together")
        ledger_bytes = self._read_regular_bytes(self._path, label="ledger")
        if not ledger_bytes or not ledger_bytes.endswith(b"\n"):
            raise ProviderBudgetContractError("provider budget ledger is incomplete")
        try:
            text = ledger_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProviderBudgetContractError("provider budget ledger is not UTF-8") from error
        events: list[dict[str, Any]] = []
        previous: str | None = None
        for expected_sequence, line in enumerate(text[:-1].split("\n")):
            parsed = _json_without_duplicate_keys(line)
            if not isinstance(parsed, dict):
                raise ProviderBudgetContractError("provider budget ledger event is not an object")
            event_type = parsed.get("event_type")
            expected_fields = _EVENT_FIELDS.get(event_type) if isinstance(event_type, str) else None
            if expected_fields is None or set(parsed) != expected_fields:
                raise ProviderBudgetContractError("provider budget event shape is invalid")
            if (
                parsed.get("schema_version") != _EVENT_SCHEMA_VERSION
                or isinstance(parsed.get("sequence"), bool)
                or not isinstance(parsed.get("sequence"), int)
                or parsed.get("sequence") != expected_sequence
                or parsed.get("previous_event_sha256") != previous
                or not _is_sha256(parsed.get("event_sha256"))
                or not self._validate_timestamp(parsed.get("timestamp"))
                or not _event_hash_is_valid(parsed)
            ):
                raise ProviderBudgetContractError("provider budget ledger hash chain is invalid")
            if parsed.get("contract_sha256") != self._contract_sha256:
                raise ProviderBudgetContractError("provider budget ledger contract changed")
            previous = parsed["event_sha256"]
            events.append(parsed)
        header = events[0]
        if (
            header["event_type"] != "contract"
            or header["previous_event_sha256"] is not None
            or not isinstance(header.get("contract"), dict)
            or _contract_identity(header["contract"]) != _contract_identity(self._contract)
        ):
            raise ProviderBudgetContractError("provider budget ledger header does not match")
        active_limits = _limits_from_contract(header["contract"])
        reservations: dict[int, dict[str, Any]] = {}
        completions: dict[int, dict[str, Any]] = {}
        for event in events[1:]:
            if event["event_type"] == "budget_amendment":
                raw_limits = event["limits"]
                if not isinstance(raw_limits, dict):
                    raise ProviderBudgetContractError("provider budget amendment is invalid")
                amended_contract = {**self._contract, "limits": raw_limits}
                new_limits = _limits_from_contract(amended_contract)
                if (
                    new_limits.cost_reservation_per_call_usd
                    != active_limits.cost_reservation_per_call_usd
                    or new_limits.max_structured_calls < active_limits.max_structured_calls
                    or new_limits.max_cost_usd < active_limits.max_cost_usd
                    or event["previous_limits_sha256"]
                    != hashlib.sha256(
                        _canonical_bytes(active_limits.contract_payload())
                    ).hexdigest()
                    or event["limits_sha256"]
                    != hashlib.sha256(_canonical_bytes(raw_limits)).hexdigest()
                ):
                    raise ProviderBudgetContractError(
                        "provider budget amendment is not a bound monotonic increase"
                    )
                active_limits = new_limits
                continue
            invocation_id = event["invocation_id"]
            if isinstance(invocation_id, bool) or not isinstance(invocation_id, int):
                raise ProviderBudgetContractError("provider budget invocation id is invalid")
            if event["event_type"] == "reservation":
                if invocation_id != len(reservations) + 1 or invocation_id in reservations:
                    raise ProviderBudgetContractError(
                        "provider budget reservations are not ordered"
                    )
                if not _is_non_negative_number(event["reserved_cost_usd"]) or _decimal(
                    event["reserved_cost_usd"]
                ) != _decimal(active_limits.cost_reservation_per_call_usd):
                    raise ProviderBudgetContractError("provider budget reservation changed")
                self._validate_request(event["request"])
                reservations[invocation_id] = event
                continue
            if invocation_id not in reservations or invocation_id in completions:
                raise ProviderBudgetContractError("provider budget completion is unbound")
            outcome = event["outcome"]
            if not isinstance(outcome, str) or outcome not in _OUTCOMES:
                raise ProviderBudgetContractError("provider budget outcome is invalid")
            request = reservations[invocation_id]["request"]
            call = event["provider_call"]
            if call is not None:
                call = self._validate_provider_call(call, request=request)
            self._validate_failure(event["failure"], outcome=outcome, call=call)
            completions[invocation_id] = event
        self._validate_head_locked(ledger_bytes=ledger_bytes, final_event=events[-1])
        self._events = events
        self._reservations = reservations
        self._completions = completions
        self._ledger_limits = active_limits

    def _amend_limits_if_needed_locked(self) -> None:
        active = self._ledger_limits
        if active is None:
            raise ProviderBudgetContractError("provider budget ledger has no active limits")
        if self._limits.cost_reservation_per_call_usd != active.cost_reservation_per_call_usd:
            raise ProviderBudgetContractError("provider budget reservation contract changed")
        if (
            self._limits.max_structured_calls < active.max_structured_calls
            or self._limits.max_cost_usd < active.max_cost_usd
        ):
            raise ProviderBudgetContractError("provider budget ceilings cannot decrease on resume")
        if self._limits == active:
            return
        old_payload = active.contract_payload()
        new_payload = self._limits.contract_payload()
        event = self._next_event(
            event_type="budget_amendment",
            contract_sha256=self._contract_sha256,
            previous_limits_sha256=hashlib.sha256(_canonical_bytes(old_payload)).hexdigest(),
            limits_sha256=hashlib.sha256(_canonical_bytes(new_payload)).hexdigest(),
            limits=new_payload,
        )
        self._append_locked(event)
        self._ledger_limits = self._limits

    def _next_event(self, *, event_type: str, **payload: Any) -> dict[str, Any]:
        previous = self._events[-1]["event_sha256"] if self._events else None
        return _hashed_event(
            {
                "schema_version": _EVENT_SCHEMA_VERSION,
                "sequence": len(self._events),
                "event_type": event_type,
                "previous_event_sha256": previous,
                "timestamp": datetime.now(UTC).isoformat(),
                **payload,
            }
        )

    def _head_payload(self, *, event: dict[str, Any], ledger_bytes: bytes) -> dict[str, Any]:
        return {
            "schema_version": _HEAD_SCHEMA_VERSION,
            "contract_sha256": self._contract_sha256,
            "final_sequence": event["sequence"],
            "final_event_sha256": event["event_sha256"],
            "ledger_size_bytes": len(ledger_bytes),
            "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        }

    def _write_head_atomic_locked(self, *, event: dict[str, Any]) -> None:
        ledger_bytes = self._read_regular_bytes(self._path, label="ledger")
        payload = (
            _canonical_bytes(self._head_payload(event=event, ledger_bytes=ledger_bytes)) + b"\n"
        )
        if self._head_path.is_symlink():
            raise ProviderBudgetContractError("provider budget head must not be a symlink")
        if self._head_path.exists() and not self._head_path.is_file():
            raise ProviderBudgetContractError("provider budget head must be a regular file")
        descriptor: int | None = None
        temporary_name: str | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._head_path.name}.",
                suffix=".tmp",
                dir=self._head_path.parent,
            )
            os.fchmod(descriptor, 0o600)
            self._write_all(descriptor, payload)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            if self._head_path.is_symlink():
                raise ProviderBudgetContractError("provider budget head must not be a symlink")
            os.replace(temporary_name, self._head_path)
            temporary_name = None
            self._fsync_directory(self._head_path.parent)
        except ProviderBudgetContractError:
            raise
        except OSError as error:
            raise ProviderBudgetContractError("provider budget head update failed") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_name is not None:
                with suppress(FileNotFoundError):
                    os.unlink(temporary_name)

    def _append_locked(self, event: dict[str, Any]) -> None:
        if self._path.is_symlink():
            raise ProviderBudgetContractError("provider budget ledger must not be a symlink")
        line = _canonical_bytes(event) + b"\n"
        ledger_existed = self._path.exists()
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._path, flags, 0o600)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ProviderBudgetContractError(
                        "provider budget ledger must be a singly linked regular file"
                    )
                self._write_all(descriptor, line)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if not ledger_existed:
                self._fsync_directory(self._path.parent)
            self._events.append(event)
            self._write_head_atomic_locked(event=event)
        except ProviderBudgetContractError:
            raise
        except OSError as error:
            raise ProviderBudgetContractError("provider budget ledger append failed") from error

    @property
    def summary(self) -> dict[str, Any]:
        self._ensure_usable()
        try:
            with self._exclusive_lock():
                self._load_and_validate_locked()
                return self._summary_locked()
        except ProviderBudgetContractError as error:
            self._poison(error)

    def _summary_locked(self) -> dict[str, Any]:
        limits = self._ledger_limits
        if limits is None:
            raise ProviderBudgetContractError("provider budget ledger has no active limits")
        reservations = len(self._reservations)
        completions = len(self._completions)
        successes = sum(event.get("outcome") == "success" for event in self._completions.values())
        actual_costs = [
            event["provider_call"]["cost_usd"]
            for event in self._completions.values()
            if isinstance(event.get("provider_call"), dict)
            and event["provider_call"].get("cost_usd") is not None
        ]
        reported_calls = [
            event["provider_call"]
            for event in self._completions.values()
            if isinstance(event.get("provider_call"), dict)
        ]
        reasoning_token_calls = [
            call["reasoning_tokens"]
            for call in reported_calls
            if call.get("reasoning_tokens") is not None
        ]
        failure_attempts = [
            event["failure"]["known_attempts"]
            for event in self._completions.values()
            if event.get("provider_call") is None
            and isinstance(event.get("failure"), dict)
            and event["failure"].get("known_attempts") is not None
        ]
        reservation = _decimal(limits.cost_reservation_per_call_usd)
        reserved_authorization = reservation * reservations
        committed_cost, committed_without_actual, reservation_underestimated = (
            self._committed_cost_locked(limits=limits)
        )
        max_cost = _decimal(limits.max_cost_usd)
        actual_overrun = max(Decimal("0"), committed_cost - max_cost)
        provider_reported_cost = sum((_decimal(value) for value in actual_costs), Decimal("0"))
        provider_latency = sum(
            (_decimal(call.get("latency_seconds") or 0) for call in reported_calls),
            Decimal("0"),
        )
        serialized_totals = {
            "reserved_authorization": _json_decimal_number(reserved_authorization),
            "committed_cost": _json_decimal_number(committed_cost),
            "committed_remaining": _json_decimal_number(max_cost - committed_cost),
            "actual_overrun": _json_decimal_number(actual_overrun),
            "reported_cost": _json_decimal_number(provider_reported_cost, round_digits=12),
            "provider_latency": _json_decimal_number(provider_latency, round_digits=6),
        }
        outcome_counts = {outcome: 0 for outcome in sorted(_OUTCOMES)}
        for event in self._completions.values():
            outcome_counts[event["outcome"]] += 1
        no_next_call = committed_cost + reservation > max_cost
        return {
            "schema_version": "provider-budget-summary/0.2",
            "status": (
                "actual_cost_overrun"
                if actual_overrun > 0
                else "exhausted"
                if reservations >= limits.max_structured_calls or no_next_call
                else "available"
            ),
            "contract_sha256": self._contract_sha256,
            "ledger": "private/provider-budget-ledger.jsonl",
            "head": "private/provider-budget-ledger.jsonl.head.json",
            "max_structured_calls": limits.max_structured_calls,
            "structured_calls_started": reservations,
            "structured_calls_completed": completions,
            "structured_calls_pending": reservations - completions,
            "structured_calls_succeeded": successes,
            "structured_calls_failed": completions - successes,
            "structured_calls_remaining": max(0, limits.max_structured_calls - reservations),
            "completion_outcomes": outcome_counts,
            "max_cost_usd": limits.max_cost_usd,
            "cost_reservation_per_call_usd": limits.cost_reservation_per_call_usd,
            "reserved_authorization_usd": serialized_totals["reserved_authorization"],
            "reserved_cost_usd": serialized_totals["reserved_authorization"],
            "committed_cost_usd": serialized_totals["committed_cost"],
            "committed_cost_remaining_usd": serialized_totals["committed_remaining"],
            "reservable_cost_remaining_usd": serialized_totals["committed_remaining"],
            "committed_cost_without_actual_calls": committed_without_actual,
            "reservation_underestimated_calls": reservation_underestimated,
            "actual_cost_overrun": actual_overrun > 0,
            "actual_cost_overrun_usd": serialized_totals["actual_overrun"],
            "provider_reported_cost_usd": serialized_totals["reported_cost"],
            "provider_reported_cost_calls": len(actual_costs),
            "provider_call_telemetry_calls": len(reported_calls),
            "provider_call_telemetry_missing_calls": completions - len(reported_calls),
            "failure_attempt_telemetry_calls": len(failure_attempts),
            "transport_attempt_telemetry_calls": len(reported_calls) + len(failure_attempts),
            "transport_attempt_telemetry_missing_calls": (
                completions - len(reported_calls) - len(failure_attempts)
            ),
            "provider_reported_input_tokens_lower_bound": sum(
                int(call.get("input_tokens") or 0) for call in reported_calls
            ),
            "provider_reported_output_tokens_lower_bound": sum(
                int(call.get("output_tokens") or 0) for call in reported_calls
            ),
            "provider_reported_reasoning_tokens_lower_bound": sum(reasoning_token_calls),
            "provider_reported_reasoning_tokens_calls": len(reasoning_token_calls),
            "provider_reported_reasoning_tokens_missing_calls": (
                completions - len(reasoning_token_calls)
            ),
            "provider_reported_total_tokens_lower_bound": sum(
                int(call.get("total_tokens") or 0) for call in reported_calls
            ),
            "provider_latency_seconds_lower_bound": serialized_totals["provider_latency"],
            "transport_attempts_lower_bound": sum(
                int(call.get("attempts") or 0) for call in reported_calls
            )
            + sum(failure_attempts),
            "transport_retries_lower_bound": sum(
                int(call.get("retries") or 0) for call in reported_calls
            )
            + sum(max(0, attempts - 1) for attempts in failure_attempts),
            "cost_accounting_basis": (
                "committed cost uses provider-reported actual cost when available and the "
                "non-refundable reservation otherwise; reserved_authorization_usd separately "
                "reports all pre-dispatch reservations; provider-reported actual cost is a "
                "lower bound when failed calls lack telemetry"
            ),
            "numeric_serialization": (
                "finite totals are JSON numbers; totals beyond the IEEE-754 finite range are "
                "exact decimal strings"
            ),
        }

    def _committed_cost_locked(self, *, limits: ProviderBudgetLimits) -> tuple[Decimal, int, int]:
        """Return exact committed cost, missing-actual count, and underestimates."""

        reservation = _decimal(limits.cost_reservation_per_call_usd)
        committed_cost = Decimal("0")
        committed_without_actual = 0
        reservation_underestimated = 0
        for invocation_id in self._reservations:
            completion = self._completions.get(invocation_id)
            call = completion.get("provider_call") if completion is not None else None
            actual = call.get("cost_usd") if isinstance(call, dict) else None
            if actual is None:
                committed_cost += reservation
                committed_without_actual += 1
                continue
            actual_decimal = _decimal(actual)
            committed_cost += actual_decimal
            if actual_decimal > reservation:
                reservation_underestimated += 1
        return committed_cost, committed_without_actual, reservation_underestimated

    def _reserve(self, request: dict[str, Any]) -> int:
        self._ensure_usable()
        try:
            with self._exclusive_lock():
                self._load_and_validate_locked()
                summary = self._summary_locked()
                if summary["structured_calls_remaining"] < 1:
                    raise ProviderBudgetExhausted(reason="structured_call_limit", summary=summary)
                limits = self._ledger_limits
                if limits is None:
                    raise ProviderBudgetContractError("provider budget ledger has no active limits")
                reservation = _decimal(limits.cost_reservation_per_call_usd)
                committed_cost, _, _ = self._committed_cost_locked(limits=limits)
                if committed_cost > _decimal(
                    limits.max_cost_usd
                ) or committed_cost + reservation > _decimal(limits.max_cost_usd):
                    raise ProviderBudgetExhausted(reason="cost_limit", summary=summary)
                invocation_id = len(self._reservations) + 1
                event = self._next_event(
                    event_type="reservation",
                    contract_sha256=self._contract_sha256,
                    invocation_id=invocation_id,
                    request=request,
                    reserved_cost_usd=limits.cost_reservation_per_call_usd,
                )
                self._append_locked(event)
                self._reservations[invocation_id] = event
                return invocation_id
        except ProviderBudgetContractError as error:
            self._poison(error)

    def _complete(
        self,
        *,
        invocation_id: int,
        outcome: str,
        call: ProviderCall | None,
        failure: dict[str, Any] | None,
    ) -> None:
        self._ensure_usable()
        try:
            with self._exclusive_lock():
                self._load_and_validate_locked()
                if invocation_id not in self._reservations or invocation_id in self._completions:
                    raise ProviderBudgetContractError(
                        "provider completion reservation is not authoritative"
                    )
                provider_call = self._project_provider_call(call) if call is not None else None
                request = self._reservations[invocation_id]["request"]
                if provider_call is not None:
                    self._validate_provider_call(provider_call, request=request)
                if outcome not in _OUTCOMES:
                    raise ProviderBudgetContractError("provider budget outcome is invalid")
                self._validate_failure(
                    failure,
                    outcome=outcome,
                    call=provider_call,
                )
                event = self._next_event(
                    event_type="completion",
                    contract_sha256=self._contract_sha256,
                    invocation_id=invocation_id,
                    outcome=outcome,
                    provider_call=provider_call,
                    failure=failure,
                )
                self._append_locked(event)
                self._completions[invocation_id] = event
        except ProviderBudgetContractError as error:
            self._poison(error)

    @staticmethod
    def _project_provider_call(call: ProviderCall) -> dict[str, Any]:
        if not isinstance(call, ProviderCall):
            raise ProviderBudgetContractError("provider call telemetry is not typed")
        model_returned_sha256 = (
            _text_sha256(call.model_returned) if call.model_returned is not None else None
        )
        if call.model_returned is None:
            model_returned = None
            model_returned_disposition = "missing"
        elif call.model_returned == call.model_requested:
            model_returned = call.model_returned
            model_returned_disposition = "matches_requested"
        else:
            model_returned = None
            model_returned_disposition = "hashed_untrusted"
        provider_returned_sha256 = (
            _text_sha256(call.provider_returned) if call.provider_returned is not None else None
        )
        if call.provider_returned is None:
            provider_returned = None
            provider_returned_disposition = "missing"
        elif call.provider_returned in PUBLIC_PROVIDER_LABELS:
            provider_returned = call.provider_returned
            provider_returned_disposition = "known_label"
        else:
            provider_returned = None
            provider_returned_disposition = "hashed_untrusted"
        finish_reason = (
            None
            if call.finish_reason is None
            else call.finish_reason
            if call.finish_reason in _SAFE_FINISH_REASONS
            else "other"
        )
        return {
            "model_requested": call.model_requested,
            "model_returned": model_returned,
            "model_returned_disposition": model_returned_disposition,
            "model_returned_sha256": model_returned_sha256,
            "provider_returned": provider_returned,
            "provider_returned_disposition": provider_returned_disposition,
            "provider_returned_sha256": provider_returned_sha256,
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
            "data_collection": call.data_collection,
            "require_parameters": call.require_parameters,
            "zdr": call.zdr,
            "latency_seconds": call.latency_seconds,
            "cost_usd": call.cost_usd,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "reasoning_tokens": call.reasoning_tokens,
            "total_tokens": call.total_tokens,
            "attempts": call.attempts,
            "retries": call.attempts - 1,
            "finish_reason": finish_reason,
        }

    def _request_fingerprint(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        schema = kwargs.get("schema")
        schema_name = kwargs.get("schema_name")
        model = kwargs.get("model")
        system = kwargs.get("system")
        user = kwargs.get("user")
        if (
            not isinstance(schema, dict)
            or not isinstance(schema_name, str)
            or not schema_name
            or not isinstance(model, str)
            or not model
            or not isinstance(system, str)
            or not isinstance(user, str)
        ):
            raise ValueError("structured provider request fields are invalid")
        configured_require = kwargs.get("require_parameters")
        effective_require = (
            bool(getattr(self._client, "_require_parameters", False))
            if configured_require is None
            else configured_require
        )
        if not isinstance(effective_require, bool):
            raise ValueError("require_parameters must be a boolean")
        request_contract = structured_request_contract(
            schema_name=schema_name,
            schema=schema,
            seed=kwargs.get("seed"),
            require_parameters=effective_require,
            model=model,
            max_tokens=kwargs.get("max_tokens"),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_bytes = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        request = {
            "model_requested": model,
            "schema_name": schema_name,
            "local_schema_sha256": hashlib.sha256(_canonical_bytes(schema)).hexdigest(),
            "provider_schema_sha256": request_contract["schema"]["schema_sha256"],
            "request_contract": request_contract,
            "request_contract_sha256": hashlib.sha256(
                _canonical_bytes(request_contract)
            ).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
            "temperature": kwargs.get("temperature"),
            "reasoning_effort": kwargs.get("reasoning_effort"),
            "max_tokens": kwargs.get("max_tokens"),
            "completion_token_parameter": request_contract["completion_token_parameter"],
            "seed": kwargs.get("seed"),
            "require_parameters": effective_require,
        }
        request["request_sha256"] = hashlib.sha256(_canonical_bytes(request)).hexdigest()
        return self._validate_request(request)

    @staticmethod
    def _failure_payload(
        *,
        terminal_class: Literal[
            "response_validation",
            "request_rejected",
            "transport_or_client_failure",
        ],
        http_status: int | None = None,
        known_attempts: Any = None,
    ) -> dict[str, Any]:
        if (
            isinstance(known_attempts, bool)
            or not isinstance(known_attempts, int)
            or known_attempts < 1
        ):
            known_attempts = None
        return {
            "terminal_class": terminal_class,
            "http_status": http_status,
            "known_attempts": known_attempts,
        }

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        """Reserve, dispatch exactly once, and append a secret-free outcome."""

        request = self._request_fingerprint(kwargs)
        invocation_id = self._reserve(request)
        try:
            response = self._client.structured_chat(**kwargs)
        except ProviderResponseValidationError as error:
            self._complete(
                invocation_id=invocation_id,
                outcome=f"provider_response_{error.code}",
                call=error.call,
                failure=self._failure_payload(
                    terminal_class="response_validation",
                    known_attempts=error.call.attempts,
                ),
            )
            raise
        except ProviderRequestRejectedError as error:
            self._complete(
                invocation_id=invocation_id,
                outcome="provider_request_rejected",
                call=None,
                failure=self._failure_payload(
                    terminal_class="request_rejected",
                    http_status=error.status_code,
                    known_attempts=getattr(error, "attempts", 1),
                ),
            )
            raise
        except Exception as error:
            self._complete(
                invocation_id=invocation_id,
                outcome="technical_failure",
                call=None,
                failure=self._failure_payload(
                    terminal_class="transport_or_client_failure",
                    known_attempts=getattr(error, "attempts", None),
                ),
            )
            raise
        if not isinstance(response, StructuredResponse):
            error = TypeError("provider client returned an invalid structured response")
            self._complete(
                invocation_id=invocation_id,
                outcome="technical_failure",
                call=None,
                failure=self._failure_payload(terminal_class="transport_or_client_failure"),
            )
            raise error
        self._complete(
            invocation_id=invocation_id,
            outcome="success",
            call=response.call,
            failure=None,
        )
        return response
