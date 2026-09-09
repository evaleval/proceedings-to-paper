"""Private, resumable model bake-offs for whole-paper producer-origin proposals.

The provider phase accepts only candidate locators in a verified sealed development
run.  It has no label-path argument and writes exact retrieval/proposal evidence only to
its caller-designated private checkpoint.  The returned report is quote-free.  A
separate offline function joins a strict single-reviewer label file after the provider
phase is sealed.

Provider output remains a proposal.  Every successful proposal is passed through the
production deterministic verifier, whose assessment never permits automatic positive
promotion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from proceedings_to_eee.domain.attribution import AttributionState
from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.evaluation.artifact_safety import assert_artifact_paths_safe
from proceedings_to_eee.evaluation.staged_eligibility import (
    StageExecutionSelection,
    StageExecutionSelectionExpectation,
    derive_stage_execution_selection,
    selected_models_for_execution,
)
from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from proceedings_to_eee.pipeline import _code_state
from proceedings_to_eee.providers.budget import ProviderBudgetError
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderResponseValidationError,
    public_provider_call,
    structured_request_contract,
    structured_request_contract_from_call,
)
from proceedings_to_eee.resolution.origin_retrieval import (
    ORIGIN_SCHEMA_NAME,
    ORIGIN_SYSTEM_PROMPT,
    OriginRetrievalBundle,
    OriginRetrievalContract,
    ProducerOriginAssessment,
    ProducerOriginProposal,
    candidate_origin_binding_sha256,
    layout_binding_sha256,
    producer_origin_prompt,
    producer_origin_prompt_hash,
    producer_origin_provider_json_schema,
    producer_origin_wire_response_sha256,
    propose_producer_origin,
    retrieve_origin_context,
    verify_producer_origin_proposal,
)
from proceedings_to_eee.run_seal import VerifiedRunSeal, verify_run_seal

HEX_64 = r"^[0-9a-f]{64}$"
_CHECKPOINT_SCHEMA_VERSION = "origin-bakeoff-checkpoint/0.2"
_ENTRY_SCHEMA_VERSION = "origin-bakeoff-entry/0.2"
_RESULT_SCHEMA_VERSION = "origin-bakeoff-provider-result/0.2"
_SCORE_SCHEMA_VERSION = "origin-bakeoff-score/0.3"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=False)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(_jsonable(value)))


def _strict_relative_path(value: str, *, label: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a canonical relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative path")
    return value


class OriginBakeoffModelSpec(StrictModel):
    model: str = Field(min_length=1)
    label: str = Field(min_length=1)


class FrozenOriginCandidateLocator(StrictModel):
    """One candidate and layout pinned inside the configured sealed run."""

    case_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    observations_path: str = Field(min_length=1)
    layout_path: str = Field(min_length=1)
    candidate_binding_sha256: str = Field(pattern=HEX_64)
    layout_binding_sha256: str = Field(pattern=HEX_64)

    @field_validator("observations_path", "layout_path")
    @classmethod
    def artifact_paths_are_canonical(cls, value: str) -> str:
        return _strict_relative_path(value, label="candidate artifact path")


class OriginBakeoffConfig(StrictModel):
    """Strict, predeclared provider-phase experiment contract."""

    schema_version: Literal["origin-bakeoff/0.1", "origin-bakeoff/0.2"] = "origin-bakeoff/0.1"
    bakeoff_id: str = Field(min_length=1)
    sealed_run_path: str = Field(min_length=1)
    sealed_run_seal_sha256: str = Field(pattern=HEX_64)
    sealed_run_tree_sha256: str = Field(pattern=HEX_64)
    models: list[OriginBakeoffModelSpec] = Field(min_length=2)
    candidates: list[FrozenOriginCandidateLocator] = Field(min_length=1)
    retrieval: OriginRetrievalContract = Field(default_factory=OriginRetrievalContract)
    max_tokens: int = Field(default=2_500, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_effort: str | None = Field(default=None, min_length=1)
    seed: int | None = None
    require_parameters: Literal[True] = True
    fresh_repetitions: int = Field(default=2, ge=1)

    @model_validator(mode="before")
    @classmethod
    def repair04_defaults_use_the_production_contract(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("schema_version") == "origin-bakeoff/0.2":
            value = dict(value)
            value.setdefault("max_tokens", 16_000)
            value.setdefault("temperature", None)
            value.setdefault("reasoning_effort", "minimal")
            value.setdefault("seed", None)
            value.setdefault("require_parameters", True)
        return value

    @field_validator("sealed_run_path")
    @classmethod
    def sealed_path_is_canonical(cls, value: str) -> str:
        return _strict_relative_path(value, label="sealed_run_path")

    @field_validator("seed", mode="before")
    @classmethod
    def seed_is_exact_integer_or_null(cls, value: Any) -> Any:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("seed must be an integer or null")
        return value

    @field_validator("fresh_repetitions", mode="before")
    @classmethod
    def repetitions_are_exact_integer(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("fresh_repetitions must be an integer")
        return value

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> OriginBakeoffConfig:
        if self.schema_version == "origin-bakeoff/0.2" and (
            self.max_tokens != 16_000
            or self.temperature is not None
            or self.reasoning_effort != "minimal"
            or self.seed is not None
            or not self.require_parameters
        ):
            raise ValueError("origin-bakeoff/0.2 requires the production request contract")
        models = [item.model for item in self.models]
        cases = [item.case_id for item in self.candidates]
        observations = [(item.observations_path, item.observation_id) for item in self.candidates]
        if len(models) != len(set(models)):
            raise ValueError("origin bake-off model IDs must be unique")
        if len(cases) != len(set(cases)):
            raise ValueError("origin bake-off case IDs must be unique")
        if len(observations) != len(set(observations)):
            raise ValueError("origin bake-off candidate locators must be unique")
        return self


class OriginResultLabel(StrictModel):
    """Private single-reviewer label; never accepted by the provider phase."""

    schema_version: Literal["origin-bakeoff-label/0.1"] = "origin-bakeoff-label/0.1"
    case_id: str = Field(min_length=1)
    candidate_binding_sha256: str = Field(pattern=HEX_64)
    annotator: str = Field(min_length=1)
    result_label: Literal["result", "not_result", "uncertain"]
    origin_label: AttributionState | None

    @model_validator(mode="after")
    def origin_matches_result_disposition(self) -> OriginResultLabel:
        if self.result_label == "result" and self.origin_label is None:
            raise ValueError("result labels require an origin_label")
        if self.result_label == "not_result" and self.origin_label is not None:
            raise ValueError("not_result labels cannot carry an origin_label")
        return self


def load_origin_bakeoff_config(path: Path) -> OriginBakeoffConfig:
    """Load one strict YAML or JSON-compatible provider-phase definition."""

    return OriginBakeoffConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class _SafeError(StrictModel):
    stage: str
    type: str
    code: str
    validation_path: list[str | int] = Field(default_factory=list)
    validation_keyword: str | None = None


class _CheckpointEntry(StrictModel):
    schema_version: Literal["origin-bakeoff-entry/0.2"] = _ENTRY_SCHEMA_VERSION
    slot_id: str = Field(min_length=1)
    slot_contract_sha256: str = Field(pattern=HEX_64)
    case_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    repetition_index: int = Field(ge=1)
    candidate_binding_sha256: str = Field(pattern=HEX_64)
    layout_binding_sha256: str = Field(pattern=HEX_64)
    retrieval_checkpoint_sha256: str = Field(pattern=HEX_64)
    request: dict[str, Any]
    status: Literal[
        "success",
        "contract_failure",
        "response_validation_failure",
        "provider_failure",
        "local_verification_failure",
    ]
    schema_status: Literal["valid", "invalid", "not_observed"]
    contract_status: Literal["satisfied", "failed", "not_observed"]
    contract_failure_codes: list[str] = Field(default_factory=list)
    proposal: ProducerOriginProposal | None = None
    assessment: ProducerOriginAssessment | None = None
    call: ProviderCall | None = None
    error: _SafeError | None = None
    entry_sha256: str = Field(pattern=HEX_64)

    @model_validator(mode="after")
    def validate_entry(self) -> _CheckpointEntry:
        if self.request.get("request_sha256") != _hash(
            {key: value for key, value in self.request.items() if key != "request_sha256"}
        ):
            raise ValueError("checkpoint request_sha256 is invalid")
        payload = self.model_dump(mode="json", exclude={"entry_sha256"})
        if self.entry_sha256 != _hash(payload):
            raise ValueError("checkpoint entry_sha256 is invalid")
        if self.proposal is not None:
            if self.call is None:
                raise ValueError("materialized origin proposal lacks its provider call")
            if self.call.response_sha256 != producer_origin_wire_response_sha256(self.proposal):
                raise ValueError("materialized origin proposal does not match provider response")
        if self.status in {"success", "contract_failure"}:
            if self.proposal is None or self.assessment is None or self.call is None:
                raise ValueError("completed proposal entry is incomplete")
            if self.schema_status != "valid":
                raise ValueError("completed proposal must have valid wire schema")
        if self.status == "success" and self.contract_status != "satisfied":
            raise ValueError("successful entry must satisfy the request contract")
        if self.status == "contract_failure" and self.contract_status != "failed":
            raise ValueError("contract failure entry must record failed contract")
        if self.status == "response_validation_failure" and self.schema_status != "invalid":
            raise ValueError("response validation failure must record invalid schema")
        if self.status == "provider_failure" and self.schema_status != "not_observed":
            raise ValueError("provider failure must record an unobserved schema")
        if self.assessment is not None and self.assessment.allows_automatic_export:
            raise ValueError("origin bake-off entry cannot allow automatic export")
        return self


class _Checkpoint(StrictModel):
    schema_version: Literal["origin-bakeoff-checkpoint/0.2"] = _CHECKPOINT_SCHEMA_VERSION
    bakeoff_id: str = Field(min_length=1)
    configuration_sha256: str = Field(pattern=HEX_64)
    run_contract_sha256: str = Field(pattern=HEX_64)
    status: Literal["in_progress", "sealed"]
    entries: dict[str, _CheckpointEntry]
    provider_phase_seal_sha256: str | None = Field(default=None, pattern=HEX_64)
    checkpoint_sha256: str = Field(pattern=HEX_64)

    @model_validator(mode="after")
    def validate_checkpoint(self) -> _Checkpoint:
        if any(key != entry.slot_id for key, entry in self.entries.items()):
            raise ValueError("checkpoint entry key does not match slot_id")
        if self.status == "sealed":
            expected_seal = _hash(
                {
                    "schema_version": "origin-bakeoff-provider-seal/0.1",
                    "configuration_sha256": self.configuration_sha256,
                    "run_contract_sha256": self.run_contract_sha256,
                    "entry_sha256s": {
                        key: entry.entry_sha256 for key, entry in sorted(self.entries.items())
                    },
                }
            )
            if self.provider_phase_seal_sha256 != expected_seal:
                raise ValueError("provider phase seal is invalid")
        elif self.provider_phase_seal_sha256 is not None:
            raise ValueError("in-progress checkpoint cannot carry a phase seal")
        payload = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if self.checkpoint_sha256 != _hash(payload):
            raise ValueError("checkpoint_sha256 is invalid")
        return self


@dataclass(frozen=True, slots=True)
class _PreparedCase:
    locator: FrozenOriginCandidateLocator
    candidate: CandidateObservation
    layout: PdfLayout
    retrieval: OriginRetrievalBundle
    observations_file_sha256: str
    layout_file_sha256: str


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    config: OriginBakeoffConfig
    configuration_sha256: str
    verified_seal: VerifiedRunSeal
    code: dict[str, Any]
    code_sha256: str
    stage_contract_sha256: str
    execution_selection: StageExecutionSelection | None
    models: tuple[OriginBakeoffModelSpec, ...]
    run_contract_sha256: str
    cases: tuple[_PreparedCase, ...]


@dataclass(frozen=True, slots=True)
class _ConfiguredClient:
    client: Any
    temperature: float | None
    reasoning_effort: str | None
    seed: int | None
    require_parameters: bool

    def structured_chat(self, **kwargs: Any) -> Any:
        forwarded = dict(kwargs)
        forwarded["temperature"] = self.temperature
        forwarded["reasoning_effort"] = self.reasoning_effort
        forwarded["seed"] = self.seed
        forwarded["require_parameters"] = self.require_parameters
        return self.client.structured_chat(**forwarded)


def _safe_error(stage: str, error: Exception) -> _SafeError:
    validation_path: list[str | int] = []
    validation_keyword: str | None = None
    if isinstance(error, ProviderResponseValidationError):
        validation_path = list(error.validation_path)
        validation_keyword = error.validation_keyword
    return _SafeError(
        stage=stage,
        type=type(error).__name__,
        code=(
            error.code if isinstance(error, ProviderResponseValidationError) else "unexpected_error"
        ),
        validation_path=validation_path,
        validation_keyword=validation_keyword,
    )


def _project_path(project_root: Path, configured: str) -> Path:
    root = project_root.resolve()
    path = (root / configured).resolve()
    if not path.is_relative_to(root):
        raise ValueError("configured path escaped project root")
    return path


def _inventory(verified: VerifiedRunSeal) -> dict[str, dict[str, Any]]:
    return {str(item["path"]): item for item in verified.files}


def _load_candidate(path: Path, observation_id: str) -> CandidateObservation:
    matches: list[CandidateObservation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        candidate = CandidateObservation.model_validate_json(line)
        if candidate.observation_id == observation_id:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError("candidate locator did not resolve exactly one observation")
    return matches[0]


def origin_stage_contract_sha256(config: OriginBakeoffConfig) -> str:
    """Bind the route-relevant contract shared by smoke and quality phases."""

    schema = producer_origin_provider_json_schema()
    request_contract = structured_request_contract(
        schema_name=ORIGIN_SCHEMA_NAME,
        schema=schema,
        seed=config.seed,
        require_parameters=True,
    )
    return _hash(
        {
            "schema_version": "origin-bakeoff-stage-contract/0.1",
            "models": [item.model for item in config.models],
            "retrieval": config.retrieval,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "seed": config.seed,
            "require_parameters": True,
            "prompt_template_sha256": producer_origin_prompt_hash(),
            "provider_schema_sha256": request_contract["schema"]["schema_sha256"],
            "request_contract_sha256": _hash(request_contract),
        }
    )


def _prepare_execution(
    config: OriginBakeoffConfig,
    *,
    project_root: Path,
    code_root: Path | None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> _ExecutionContext:
    sealed_root = _project_path(project_root, config.sealed_run_path)
    verified = verify_run_seal(sealed_root)
    if verified.seal_sha256 != config.sealed_run_seal_sha256:
        raise ValueError("sealed run seal hash does not match configuration")
    if verified.tree_sha256 != config.sealed_run_tree_sha256:
        raise ValueError("sealed run tree hash does not match configuration")
    inventory = _inventory(verified)
    cases: list[_PreparedCase] = []
    for locator in config.candidates:
        if locator.observations_path not in inventory or locator.layout_path not in inventory:
            raise ValueError("candidate locator artifact is absent from sealed inventory")
        observations_path = sealed_root / locator.observations_path
        layout_path = sealed_root / locator.layout_path
        observations_digest = str(inventory[locator.observations_path]["sha256"])
        layout_digest = str(inventory[locator.layout_path]["sha256"])
        if sha256_file(observations_path) != observations_digest:
            raise ValueError("sealed observations changed after verification")
        if sha256_file(layout_path) != layout_digest:
            raise ValueError("sealed layout changed after verification")
        candidate = _load_candidate(observations_path, locator.observation_id)
        layout = PdfLayout.model_validate(read_json(layout_path))
        if sha256_file(observations_path) != observations_digest:
            raise ValueError("sealed observations changed while loading")
        if sha256_file(layout_path) != layout_digest:
            raise ValueError("sealed layout changed while loading")
        if candidate.paper_id != locator.paper_id:
            raise ValueError("candidate paper_id does not match locator")
        candidate_binding = candidate_origin_binding_sha256(candidate)
        if candidate_binding != locator.candidate_binding_sha256:
            raise ValueError("candidate binding does not match locator")
        layout_binding = layout_binding_sha256(layout)
        if layout_binding != locator.layout_binding_sha256:
            raise ValueError("layout binding does not match locator")
        retrieval = retrieve_origin_context(candidate, layout, contract=config.retrieval)
        cases.append(
            _PreparedCase(
                locator=locator,
                candidate=candidate,
                layout=layout,
                retrieval=retrieval,
                observations_file_sha256=observations_digest,
                layout_file_sha256=layout_digest,
            )
        )
    verified_after = verify_run_seal(sealed_root)
    if (
        verified_after.seal_sha256 != verified.seal_sha256
        or verified_after.tree_sha256 != verified.tree_sha256
        or verified_after.files != verified.files
    ):
        raise ValueError("sealed run changed while preparing origin bake-off")
    canonical_config = config.model_dump(mode="json", exclude_none=False)
    configuration_sha256 = _hash(canonical_config)
    code = _code_state(code_root if code_root is not None else project_root)
    code_sha256 = _hash(code)
    stage_contract_sha256 = origin_stage_contract_sha256(config)
    selected_model_ids = selected_models_for_execution(
        execution_selection,
        expected_stage="origin_retrieval",
        declared_models=[item.model for item in config.models],
        stage_contract_sha256=stage_contract_sha256,
        code_sha256=code_sha256,
        expectation=execution_selection_expectation,
    )
    selected = set(selected_model_ids)
    models = tuple(item for item in config.models if item.model in selected)
    if tuple(item.model for item in models) != selected_model_ids:
        raise ValueError("origin execution selection does not preserve declared model order")
    run_contract_sha256 = _hash(
        {
            "schema_version": "origin-bakeoff-run-contract/0.1",
            "configuration_sha256": configuration_sha256,
            "sealed_run_seal_sha256": verified.seal_sha256,
            "sealed_run_tree_sha256": verified.tree_sha256,
            "code_sha256": code_sha256,
            "stage_contract_sha256": stage_contract_sha256,
            "execution_selection_sha256": (
                execution_selection.selection_sha256 if execution_selection is not None else None
            ),
            "executed_models": list(selected_model_ids),
            "case_bindings": [
                {
                    "case_id": case.locator.case_id,
                    "candidate_binding_sha256": case.locator.candidate_binding_sha256,
                    "layout_binding_sha256": case.locator.layout_binding_sha256,
                    "retrieval_checkpoint_sha256": case.retrieval.checkpoint_sha256,
                    "observations_file_sha256": case.observations_file_sha256,
                    "layout_file_sha256": case.layout_file_sha256,
                }
                for case in cases
            ],
        }
    )
    return _ExecutionContext(
        config=config,
        configuration_sha256=configuration_sha256,
        verified_seal=verified,
        code=code,
        code_sha256=code_sha256,
        stage_contract_sha256=stage_contract_sha256,
        execution_selection=execution_selection,
        models=models,
        run_contract_sha256=run_contract_sha256,
        cases=tuple(cases),
    )


def _request_binding(
    context: _ExecutionContext,
    case: _PreparedCase,
    model: OriginBakeoffModelSpec,
    repetition_index: int,
) -> dict[str, Any]:
    prompt = producer_origin_prompt(case.candidate, case.retrieval)
    messages = [
        {"role": "system", "content": ORIGIN_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    prompt_sha256 = hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    schema = producer_origin_provider_json_schema()
    provider_contract = structured_request_contract(
        schema_name=ORIGIN_SCHEMA_NAME,
        schema=schema,
        seed=context.config.seed,
        require_parameters=True,
        model=model.model,
        max_tokens=context.config.max_tokens,
    )
    request: dict[str, Any] = {
        "schema_version": "origin-bakeoff-request/0.1",
        "case_id": case.locator.case_id,
        "model_requested": model.model,
        "provider_requested": "openrouter",
        "repetition_index": repetition_index,
        "candidate_binding_sha256": case.locator.candidate_binding_sha256,
        "layout_binding_sha256": case.locator.layout_binding_sha256,
        "retrieval_checkpoint_sha256": case.retrieval.checkpoint_sha256,
        "origin_prompt_template_sha256": producer_origin_prompt_hash(),
        "prompt_sha256": prompt_sha256,
        "system_sha256": hashlib.sha256(ORIGIN_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "user_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "local_schema_sha256": _hash(schema),
        "provider_schema_sha256": provider_contract["schema"]["schema_sha256"],
        "request_contract": provider_contract,
        "request_contract_sha256": _hash(provider_contract),
        "settings": {
            "temperature": context.config.temperature,
            "reasoning_effort": context.config.reasoning_effort,
            "max_tokens": context.config.max_tokens,
            "seed": context.config.seed,
            "require_parameters": True,
        },
        "configuration_sha256": context.configuration_sha256,
        "code_sha256": context.code_sha256,
        "run_contract_sha256": context.run_contract_sha256,
    }
    request["request_sha256"] = _hash(request)
    return request


def _slot_id(request: dict[str, Any]) -> str:
    return (
        "origin_slot_"
        + _hash(
            {
                "case_id": request["case_id"],
                "model_requested": request["model_requested"],
                "repetition_index": request["repetition_index"],
                "request_sha256": request["request_sha256"],
            }
        )[:24]
    )


def _contract_assessment(call: ProviderCall, request: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    settings = request["settings"]
    comparisons = (
        ("provider_requested", call.provider, request["provider_requested"]),
        ("model_requested", call.model_requested, request["model_requested"]),
        ("model_returned", call.model_returned, request["model_requested"]),
        ("prompt_sha256", call.prompt_sha256, request["prompt_sha256"]),
        ("temperature", call.temperature, settings["temperature"]),
        ("reasoning_effort", call.reasoning_effort, settings["reasoning_effort"]),
        ("max_tokens", call.max_tokens, settings["max_tokens"]),
        ("seed", call.seed, settings["seed"]),
        ("require_parameters", call.require_parameters, True),
    )
    for name, actual, expected in comparisons:
        if actual != expected:
            failures.append(f"{name}_mismatch")
    observed_contract = structured_request_contract_from_call(call)
    if _hash(observed_contract) != request["request_contract_sha256"]:
        failures.append("request_contract_mismatch")
    return ("failed", failures) if failures else ("satisfied", [])


def _entry_payload(
    *,
    request: dict[str, Any],
    case: _PreparedCase,
    model: OriginBakeoffModelSpec,
    repetition_index: int,
    status: str,
    schema_status: str,
    contract_status: str,
    contract_failure_codes: list[str] | None = None,
    proposal: ProducerOriginProposal | None = None,
    assessment: ProducerOriginAssessment | None = None,
    call: ProviderCall | None = None,
    error: _SafeError | None = None,
) -> _CheckpointEntry:
    slot_id = _slot_id(request)
    payload: dict[str, Any] = {
        "schema_version": _ENTRY_SCHEMA_VERSION,
        "slot_id": slot_id,
        "slot_contract_sha256": _hash(
            {
                "slot_id": slot_id,
                "request_sha256": request["request_sha256"],
                "candidate_binding_sha256": case.locator.candidate_binding_sha256,
                "layout_binding_sha256": case.locator.layout_binding_sha256,
                "retrieval_checkpoint_sha256": case.retrieval.checkpoint_sha256,
            }
        ),
        "case_id": case.locator.case_id,
        "model": model.model,
        "repetition_index": repetition_index,
        "candidate_binding_sha256": case.locator.candidate_binding_sha256,
        "layout_binding_sha256": case.locator.layout_binding_sha256,
        "retrieval_checkpoint_sha256": case.retrieval.checkpoint_sha256,
        "request": request,
        "status": status,
        "schema_status": schema_status,
        "contract_status": contract_status,
        "contract_failure_codes": contract_failure_codes or [],
        "proposal": proposal,
        "assessment": assessment,
        "call": call,
        "error": error,
    }
    payload["entry_sha256"] = _hash(payload)
    return _CheckpointEntry.model_validate(payload)


def _phase_seal(
    configuration_sha256: str,
    run_contract_sha256: str,
    entries: dict[str, _CheckpointEntry],
) -> str:
    return _hash(
        {
            "schema_version": "origin-bakeoff-provider-seal/0.1",
            "configuration_sha256": configuration_sha256,
            "run_contract_sha256": run_contract_sha256,
            "entry_sha256s": {key: entry.entry_sha256 for key, entry in sorted(entries.items())},
        }
    )


def _checkpoint(
    context: _ExecutionContext,
    entries: dict[str, _CheckpointEntry],
    *,
    sealed: bool,
) -> _Checkpoint:
    payload: dict[str, Any] = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "bakeoff_id": context.config.bakeoff_id,
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "status": "sealed" if sealed else "in_progress",
        "entries": entries,
        "provider_phase_seal_sha256": (
            _phase_seal(
                context.configuration_sha256,
                context.run_contract_sha256,
                entries,
            )
            if sealed
            else None
        ),
    }
    payload["checkpoint_sha256"] = _hash(payload)
    return _Checkpoint.model_validate(payload)


def _write_checkpoint(path: Path, checkpoint: _Checkpoint) -> None:
    # Nullable request settings are semantically meaningful and ProviderCall declares
    # them as required fields, so the private checkpoint must retain explicit nulls.
    write_json(path, checkpoint.model_dump(mode="json", exclude_none=False))


def _load_checkpoint(
    path: Path,
    context: _ExecutionContext,
) -> tuple[dict[str, _CheckpointEntry], list[str]]:
    if not path.exists():
        return {}, []
    try:
        checkpoint = _Checkpoint.model_validate(read_json(path))
    except Exception:
        return {}, ["checkpoint_invalid"]
    if checkpoint.bakeoff_id != context.config.bakeoff_id:
        return {}, ["checkpoint_bakeoff_id_mismatch"]
    if checkpoint.configuration_sha256 != context.configuration_sha256:
        return {}, ["checkpoint_configuration_stale"]
    if checkpoint.run_contract_sha256 != context.run_contract_sha256:
        return {}, ["checkpoint_run_contract_stale"]
    return dict(checkpoint.entries), []


def _entry_reusable(
    entry: _CheckpointEntry,
    *,
    request: dict[str, Any],
    case: _PreparedCase,
    model: OriginBakeoffModelSpec,
    repetition_index: int,
) -> bool:
    expected_slot = _slot_id(request)
    expected_slot_contract = _hash(
        {
            "slot_id": expected_slot,
            "request_sha256": request["request_sha256"],
            "candidate_binding_sha256": case.locator.candidate_binding_sha256,
            "layout_binding_sha256": case.locator.layout_binding_sha256,
            "retrieval_checkpoint_sha256": case.retrieval.checkpoint_sha256,
        }
    )
    if (
        entry.slot_id != expected_slot
        or entry.slot_contract_sha256 != expected_slot_contract
        or entry.case_id != case.locator.case_id
        or entry.model != model.model
        or entry.repetition_index != repetition_index
        or entry.request != request
        or entry.candidate_binding_sha256 != case.locator.candidate_binding_sha256
        or entry.layout_binding_sha256 != case.locator.layout_binding_sha256
        or entry.retrieval_checkpoint_sha256 != case.retrieval.checkpoint_sha256
    ):
        return False
    if entry.call is not None:
        contract_status, failures = _contract_assessment(entry.call, request)
        if contract_status != entry.contract_status or failures != entry.contract_failure_codes:
            return False
    elif entry.contract_status != "not_observed":
        return False
    if entry.proposal is not None:
        if entry.call is None or entry.call.response_sha256 != producer_origin_wire_response_sha256(
            entry.proposal
        ):
            return False
        try:
            assessment = verify_producer_origin_proposal(
                candidate=case.candidate,
                layout=case.layout,
                bundle=case.retrieval,
                proposal=entry.proposal,
            )
        except Exception:
            return entry.status == "local_verification_failure" and entry.assessment is None
        if entry.assessment != assessment:
            return False
    return True


def _public_telemetry(call: ProviderCall) -> dict[str, Any]:
    return public_provider_call(call)


def _public_entry(entry: _CheckpointEntry, *, resumed: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "slot_id": entry.slot_id,
        "slot_contract_sha256": entry.slot_contract_sha256,
        "case_id": entry.case_id,
        "model": entry.model,
        "repetition_index": entry.repetition_index,
        "status": entry.status,
        "schema_status": entry.schema_status,
        "contract_status": entry.contract_status,
        "contract_failure_codes": entry.contract_failure_codes,
        "candidate_binding_sha256": entry.candidate_binding_sha256,
        "layout_binding_sha256": entry.layout_binding_sha256,
        "retrieval_checkpoint_sha256": entry.retrieval_checkpoint_sha256,
        "request": entry.request,
        "entry_sha256": entry.entry_sha256,
        "resumed": resumed,
    }
    if entry.proposal is not None:
        result["proposal"] = {
            "proposal_sha256": _hash(entry.proposal),
            "proposed_state": entry.proposal.proposed_state.value,
            "evidence_relation": entry.proposal.evidence_relation.value,
            "origin_anchor_supplied": entry.proposal.origin_anchor is not None,
            "summary_sha256": hashlib.sha256(entry.proposal.summary.encode("utf-8")).hexdigest(),
        }
    if entry.assessment is not None:
        result["assessment"] = entry.assessment.model_dump(mode="json")
    if entry.call is not None:
        result["telemetry"] = _public_telemetry(entry.call)
    if entry.error is not None:
        result["error"] = entry.error.model_dump(mode="json")
    return result


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6) if denominator else None,
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _aggregate_entries(entries: list[_CheckpointEntry]) -> dict[str, Any]:
    schema_valid = sum(entry.schema_status == "valid" for entry in entries)
    schema_invalid = sum(entry.schema_status == "invalid" for entry in entries)
    schema_observed = schema_valid + schema_invalid
    contracts_satisfied = sum(entry.contract_status == "satisfied" for entry in entries)
    calls = [entry.call for entry in entries if entry.call is not None]
    latencies = [float(call.latency_seconds) for call in calls]
    first_pass = [entry for entry in entries if entry.repetition_index == 1]

    def schema_report(values: list[_CheckpointEntry]) -> dict[str, Any]:
        valid = sum(item.schema_status == "valid" for item in values)
        invalid = sum(item.schema_status == "invalid" for item in values)
        observed = valid + invalid
        return {
            "calls": len(values),
            "valid": valid,
            "invalid": invalid,
            "not_observed": len(values) - observed,
            "valid_of_observed": _rate(valid, observed),
            "valid_end_to_end": _rate(valid, len(values)),
        }

    usage: dict[str, Any] = {}
    for field in (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cost_usd",
    ):
        values = [getattr(call, field) for call in calls if getattr(call, field) is not None]
        usage[field] = {
            "total": round(sum(values), 8),
            "reported_calls": len(values),
            "missing_calls": len(entries) - len(values),
        }
    usage["latency_seconds"] = {
        "total": round(sum(latencies), 6),
        "mean": round(sum(latencies) / len(latencies), 6) if latencies else None,
        "p50": _percentile(latencies, 0.5),
        "p95": _percentile(latencies, 0.95),
        "max": round(max(latencies), 6) if latencies else None,
        "reported_calls": len(latencies),
        "missing_calls": len(entries) - len(latencies),
    }
    return {
        "execution": {
            "logical_calls_predeclared": len(entries),
            "terminal_entries": len(entries),
            "success": sum(entry.status == "success" for entry in entries),
            "contract_failure": sum(entry.status == "contract_failure" for entry in entries),
            "response_validation_failure": sum(
                entry.status == "response_validation_failure" for entry in entries
            ),
            "provider_failure": sum(entry.status == "provider_failure" for entry in entries),
            "local_verification_failure": sum(
                entry.status == "local_verification_failure" for entry in entries
            ),
        },
        "schema": {
            "all_repetitions": {
                "calls": len(entries),
                "valid": schema_valid,
                "invalid": schema_invalid,
                "not_observed": len(entries) - schema_observed,
                "valid_of_observed": _rate(schema_valid, schema_observed),
                "valid_end_to_end": _rate(schema_valid, len(entries)),
            },
            "first_pass": schema_report(first_pass),
        },
        "contract": {
            "satisfied": contracts_satisfied,
            "failed": sum(entry.contract_status == "failed" for entry in entries),
            "not_observed": sum(entry.contract_status == "not_observed" for entry in entries),
            "satisfaction_rate": _rate(contracts_satisfied, len(entries)),
        },
        "usage": usage,
    }


def run_origin_bakeoff_provider_phase(
    config: OriginBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    client: Any,
    code_root: Path | None = None,
    output_path: Path | None = None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> dict[str, Any]:
    """Run or exactly resume all predeclared label-blind provider calls."""

    sealed_root = _project_path(project_root, config.sealed_run_path)
    assert_artifact_paths_safe(
        [checkpoint_path, *([output_path] if output_path is not None else [])],
        sealed_root=sealed_root,
    )
    context = _prepare_execution(
        config,
        project_root=project_root,
        code_root=code_root,
        execution_selection=execution_selection,
        execution_selection_expectation=execution_selection_expectation,
    )
    entries, checkpoint_rejections = _load_checkpoint(checkpoint_path, context)
    configured_client = _ConfiguredClient(
        client=client,
        temperature=config.temperature,
        reasoning_effort=config.reasoning_effort,
        seed=config.seed,
        require_parameters=True,
    )
    # Establish a valid resumable state before the first potentially paid call.
    # Budget exhaustion/contract poisoning must stop the run, not manufacture
    # provider-failure entries for calls that were never dispatched.
    _write_checkpoint(checkpoint_path, _checkpoint(context, entries, sealed=False))
    expected_slots: list[str] = []
    resumed_slots: set[str] = set()
    executed_slots: set[str] = set()
    for model in context.models:
        for case in context.cases:
            for repetition_index in range(1, config.fresh_repetitions + 1):
                request = _request_binding(context, case, model, repetition_index)
                slot_id = _slot_id(request)
                expected_slots.append(slot_id)
                cached = entries.get(slot_id)
                if cached is not None and _entry_reusable(
                    cached,
                    request=request,
                    case=case,
                    model=model,
                    repetition_index=repetition_index,
                ):
                    resumed_slots.add(slot_id)
                    continue
                if cached is not None:
                    checkpoint_rejections.append("checkpoint_entry_stale_or_tampered")
                    entries.pop(slot_id, None)
                    _write_checkpoint(checkpoint_path, _checkpoint(context, entries, sealed=False))
                try:
                    proposal, call = propose_producer_origin(
                        client=configured_client,
                        model=model.model,
                        candidate=case.candidate,
                        bundle=case.retrieval,
                        max_tokens=config.max_tokens,
                        seed=config.seed,
                    )
                except ProviderResponseValidationError as error:
                    contract_status, failures = _contract_assessment(error.call, request)
                    entry = _entry_payload(
                        request=request,
                        case=case,
                        model=model,
                        repetition_index=repetition_index,
                        status="response_validation_failure",
                        schema_status="invalid",
                        contract_status=contract_status,
                        contract_failure_codes=failures,
                        call=error.call,
                        error=_safe_error("provider_response_validation", error),
                    )
                except ProviderBudgetError:
                    raise
                except Exception as error:
                    entry = _entry_payload(
                        request=request,
                        case=case,
                        model=model,
                        repetition_index=repetition_index,
                        status="provider_failure",
                        schema_status="not_observed",
                        contract_status="not_observed",
                        error=_safe_error("provider_call", error),
                    )
                else:
                    contract_status, failures = _contract_assessment(call, request)
                    try:
                        assessment = verify_producer_origin_proposal(
                            candidate=case.candidate,
                            layout=case.layout,
                            bundle=case.retrieval,
                            proposal=proposal,
                        )
                    except Exception as error:
                        entry = _entry_payload(
                            request=request,
                            case=case,
                            model=model,
                            repetition_index=repetition_index,
                            status="local_verification_failure",
                            schema_status="valid",
                            contract_status=contract_status,
                            contract_failure_codes=failures,
                            proposal=proposal,
                            call=call,
                            error=_safe_error("deterministic_origin_verification", error),
                        )
                    else:
                        entry = _entry_payload(
                            request=request,
                            case=case,
                            model=model,
                            repetition_index=repetition_index,
                            status=(
                                "success" if contract_status == "satisfied" else "contract_failure"
                            ),
                            schema_status="valid",
                            contract_status=contract_status,
                            contract_failure_codes=failures,
                            proposal=proposal,
                            assessment=assessment,
                            call=call,
                        )
                entries[slot_id] = entry
                executed_slots.add(slot_id)
                # Persist this exact terminal result before starting another paid call.
                _write_checkpoint(checkpoint_path, _checkpoint(context, entries, sealed=False))

    expected_set = set(expected_slots)
    if set(entries) != expected_set:
        raise RuntimeError("origin bake-off checkpoint does not partition predeclared slots")
    sealed_checkpoint = _checkpoint(context, entries, sealed=True)
    _write_checkpoint(checkpoint_path, sealed_checkpoint)
    ordered_entries = [entries[slot_id] for slot_id in expected_slots]
    public_entries = [
        _public_entry(entry, resumed=entry.slot_id in resumed_slots) for entry in ordered_entries
    ]
    model_reports = []
    executed_models = {item.model for item in context.models}
    for model in config.models:
        model_entries = [entry for entry in ordered_entries if entry.model == model.model]
        if model.model in executed_models:
            decision = (
                execution_selection.decision_for(model.model)
                if execution_selection is not None
                else None
            )
            eligibility = decision.status if decision is not None else "not_assessed"
            model_reports.append(
                {
                    "model": model.model,
                    "label": model.label,
                    "contract_eligibility": eligibility,
                    "eligibility_reason_codes": (
                        decision.reason_codes if decision is not None else []
                    ),
                    "matched_quality_status": "executed",
                    "quality_status": "pending_offline_scoring",
                    "quality": None,
                    "aggregate": _aggregate_entries(model_entries),
                    "stability": _model_stability(model_entries),
                }
            )
            continue
        if execution_selection is None:
            raise RuntimeError("origin model was skipped without an execution selection")
        decision = execution_selection.decision_for(model.model)
        model_reports.append(
            {
                "model": model.model,
                "label": model.label,
                "contract_eligibility": decision.status,
                "eligibility_reason_codes": decision.reason_codes,
                "matched_quality_status": "not_run",
                "not_run_reason": "contract_ineligible_on_sealed_smoke",
                "quality_status": "unmeasured_contract_ineligible",
                "quality": None,
                "aggregate": None,
                "stability": None,
            }
        )
    result = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "bakeoff_id": config.bakeoff_id,
        "status": "sealed",
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "provider_phase_seal_sha256": sealed_checkpoint.provider_phase_seal_sha256,
        "checkpoint_sha256": sealed_checkpoint.checkpoint_sha256,
        "source": {
            "sealed_run_seal_sha256": context.verified_seal.seal_sha256,
            "sealed_run_tree_sha256": context.verified_seal.tree_sha256,
            "cases": [
                {
                    "case_id": case.locator.case_id,
                    "source_id": case.layout.source_id,
                    "observations_file_sha256": case.observations_file_sha256,
                    "layout_file_sha256": case.layout_file_sha256,
                    "candidate_binding_sha256": case.locator.candidate_binding_sha256,
                    "layout_binding_sha256": case.locator.layout_binding_sha256,
                    "retrieval_checkpoint_sha256": case.retrieval.checkpoint_sha256,
                    "retrieval_hits_sha256": case.retrieval.hits_sha256,
                    "retrieval_hit_count": len(case.retrieval.hits),
                }
                for case in context.cases
            ],
        },
        "code": context.code,
        "code_sha256": context.code_sha256,
        "stage_contract_sha256": context.stage_contract_sha256,
        "execution_selection": (
            {
                "selection_sha256": execution_selection.selection_sha256,
                "smoke_provider_phase_seal_sha256": (
                    execution_selection.smoke_provider_phase_seal_sha256
                ),
                "smoke_checkpoint_sha256": execution_selection.smoke_checkpoint_sha256,
                "eligible_models": list(execution_selection.eligible_models),
            }
            if execution_selection is not None
            else None
        ),
        "declared_models": [item.model for item in config.models],
        "executed_models": [item.model for item in context.models],
        "requested_settings": {
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
            "require_parameters": True,
            "fresh_repetitions": config.fresh_repetitions,
        },
        "resume": {
            "checkpoint_rejections": list(dict.fromkeys(checkpoint_rejections)),
            "logical_calls_reused": len(resumed_slots),
            "logical_calls_executed": len(executed_slots),
        },
        "aggregate": _aggregate_entries(ordered_entries),
        "models": model_reports,
        "entries": public_entries,
        "privacy": {
            "labels_loaded_during_provider_phase": False,
            "exact_evidence_in_public_result": False,
            "exact_evidence_checkpoint_only": True,
        },
        "automatic_positive_promotion": False,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def _load_sealed_checkpoint_for_scoring(
    checkpoint_path: Path,
    context: _ExecutionContext,
) -> tuple[_Checkpoint, list[_CheckpointEntry]]:
    checkpoint = _Checkpoint.model_validate(read_json(checkpoint_path))
    if checkpoint.status != "sealed":
        raise ValueError("provider phase checkpoint is not sealed")
    if (
        checkpoint.configuration_sha256 != context.configuration_sha256
        or checkpoint.run_contract_sha256 != context.run_contract_sha256
    ):
        raise ValueError("provider phase checkpoint is stale")
    expected: list[_CheckpointEntry] = []
    for model in context.models:
        for case in context.cases:
            for repetition_index in range(1, context.config.fresh_repetitions + 1):
                request = _request_binding(context, case, model, repetition_index)
                slot_id = _slot_id(request)
                entry = checkpoint.entries.get(slot_id)
                if entry is None or not _entry_reusable(
                    entry,
                    request=request,
                    case=case,
                    model=model,
                    repetition_index=repetition_index,
                ):
                    raise ValueError("provider phase checkpoint has an invalid slot")
                expected.append(entry)
    if len(expected) != len(checkpoint.entries):
        raise ValueError("provider phase checkpoint contains unexpected slots")
    return checkpoint, expected


def derive_origin_execution_selection(
    config: OriginBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    experiment_id: str,
    public_manifest_sha256: str,
    wire_schema_gate: str = "0.99",
    code_root: Path | None = None,
) -> StageExecutionSelection:
    """Derive matched-quality eligibility from a sealed one-pass origin smoke."""

    if config.fresh_repetitions != 1:
        raise ValueError("origin contract smoke must declare exactly one repetition")
    context = _prepare_execution(config, project_root=project_root, code_root=code_root)
    checkpoint, entries = _load_sealed_checkpoint_for_scoring(checkpoint_path, context)
    if checkpoint.provider_phase_seal_sha256 is None:
        raise ValueError("origin contract smoke is missing its provider phase seal")
    return derive_stage_execution_selection(
        experiment_id=experiment_id,
        stage="origin_retrieval",
        public_manifest_sha256=public_manifest_sha256,
        stage_contract_sha256=context.stage_contract_sha256,
        smoke_bakeoff_id=config.bakeoff_id,
        smoke_configuration_sha256=context.configuration_sha256,
        smoke_run_contract_sha256=context.run_contract_sha256,
        smoke_code_sha256=context.code_sha256,
        smoke_provider_phase_seal_sha256=checkpoint.provider_phase_seal_sha256,
        smoke_checkpoint_sha256=checkpoint.checkpoint_sha256,
        wire_schema_gate=wire_schema_gate,
        declared_models=[item.model for item in config.models],
        expected_slots_per_model=len(context.cases),
        entries=entries,
    )


def _load_labels(
    path: Path,
    context: _ExecutionContext,
) -> tuple[dict[str, OriginResultLabel], str]:
    labels: list[OriginResultLabel] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            labels.append(OriginResultLabel.model_validate_json(line))
    by_case = {label.case_id: label for label in labels}
    if len(by_case) != len(labels):
        raise ValueError("origin label case IDs must be unique")
    expected = {case.locator.case_id: case for case in context.cases}
    if set(by_case) != set(expected):
        raise ValueError("origin labels do not exactly cover configured cases")
    for case_id, case in expected.items():
        if by_case[case_id].candidate_binding_sha256 != case.locator.candidate_binding_sha256:
            raise ValueError("origin label candidate binding mismatch")
    annotators = {label.annotator for label in labels}
    if len(annotators) != 1:
        raise ValueError("origin bake-off requires exactly one annotator")
    return by_case, hashlib.sha256(next(iter(annotators)).encode("utf-8")).hexdigest()


def _model_stability(entries: list[_CheckpointEntry]) -> dict[str, Any]:
    by_case: dict[str, list[_CheckpointEntry]] = {}
    for entry in entries:
        by_case.setdefault(entry.case_id, []).append(entry)
    exact_matches = 0
    state_matches = 0
    comparable_pairs = 0
    for case_entries in by_case.values():
        successful = [entry for entry in case_entries if entry.status == "success"]
        for left, right in combinations(successful, 2):
            comparable_pairs += 1
            exact_matches += int(_hash(left.proposal) == _hash(right.proposal))
            state_matches += int(
                left.proposal is not None
                and right.proposal is not None
                and left.proposal.proposed_state == right.proposal.proposed_state
            )
    return {
        "comparable_pair_denominator": comparable_pairs,
        "exact_proposal_matches": exact_matches,
        "exact_proposal_match_rate": _rate(exact_matches, comparable_pairs),
        "proposed_state_matches": state_matches,
        "proposed_state_match_rate": _rate(state_matches, comparable_pairs),
    }


def _score_model(
    entries: list[_CheckpointEntry],
    labels: dict[str, OriginResultLabel],
) -> dict[str, Any]:
    contract_eligible = [entry for entry in entries if entry.contract_status != "failed"]
    proposals = [entry for entry in contract_eligible if entry.proposal is not None]
    unsafe_positive = 0
    positive_proposals = 0
    externally_labeled = 0
    external_correct = 0
    evidence_bound = 0
    review_routes = 0
    unresolved_routes = 0
    automatic_promotion_violations = 0
    for entry in contract_eligible:
        label = labels[entry.case_id]
        expected_external = (
            label.result_label == "result"
            and label.origin_label is AttributionState.EXTERNALLY_SOURCED
        )
        externally_labeled += int(expected_external)
        if entry.proposal is None or entry.assessment is None:
            review_routes += 1
            unresolved_routes += 1
            continue
        expected_positive = (
            label.result_label == "result" and label.origin_label is AttributionState.PAPER_PRODUCED
        )
        proposed_positive = entry.proposal.proposed_state is AttributionState.PAPER_PRODUCED
        positive_proposals += int(proposed_positive)
        unsafe_positive += int(proposed_positive and not expected_positive)
        external_correct += int(
            expected_external
            and entry.proposal.proposed_state is AttributionState.EXTERNALLY_SOURCED
        )
        evidence_bound += int(
            entry.assessment.result_anchor_verified
            and (entry.proposal.origin_anchor is None or entry.assessment.origin_anchor_verified)
        )
        review_routes += int(entry.assessment.route.value == "review")
        unresolved_routes += int(
            entry.assessment.effective_state
            in {AttributionState.UNRESOLVED, AttributionState.NO_SIGNAL}
        )
        automatic_promotion_violations += int(
            entry.assessment.allows_automatic_export
            or entry.assessment.effective_state is AttributionState.PAPER_PRODUCED
        )
    return {
        # Safety is deliberately first in the model-quality artifact.
        "unsafe_positive_proposals": {
            "count": unsafe_positive,
            "paper_produced_proposals": positive_proposals,
            "rate_of_all_contract_eligible_calls": _rate(unsafe_positive, len(contract_eligible)),
            "rate_of_paper_produced_proposals": _rate(unsafe_positive, positive_proposals),
        },
        "scoring_denominators": {
            "logical_calls_predeclared": len(entries),
            "contract_eligible_calls": len(contract_eligible),
            "valid_proposals": len(proposals),
            "externally_sourced_label_instances": externally_labeled,
        },
        "evidence_binding_accuracy": _rate(evidence_bound, len(proposals)),
        "external_recall": _rate(external_correct, externally_labeled),
        "review_burden": _rate(review_routes, len(contract_eligible)),
        "unresolved_or_no_signal": _rate(unresolved_routes, len(contract_eligible)),
        "automatic_positive_promotion_violations": automatic_promotion_violations,
        "schema": _aggregate_entries(entries)["schema"],
        "stability": _model_stability(entries),
        "usage": _aggregate_entries(entries)["usage"],
    }


def _quality_measurement(entries: list[_CheckpointEntry]) -> dict[str, Any]:
    """Separate matched-quality evidence from terminal execution telemetry.

    A schema, provider, request-contract, or local-verification failure is not a
    semantic origin prediction.  Treating such a slot as an unresolved/review
    prediction would silently turn technical failure into apparent model quality.
    """

    aggregate = _aggregate_entries(entries)
    status_counts = aggregate["execution"]
    successful = status_counts["success"]
    complete = bool(entries) and successful == len(entries)
    return {
        "status": "measured" if complete else "unmeasured",
        "reason": (
            None
            if complete
            else "one_or_more_calls_lacked_a_valid_contract_complete_origin_assessment"
        ),
        "logical_calls_predeclared": len(entries),
        "valid_contract_complete_assessments": successful,
        "incomplete_or_failed_calls": len(entries) - successful,
        "terminal_status_counts": {
            key: status_counts[key]
            for key in (
                "success",
                "contract_failure",
                "response_validation_failure",
                "provider_failure",
                "local_verification_failure",
            )
        },
        "schema": aggregate["schema"],
        "contract": aggregate["contract"],
    }


def score_origin_bakeoff_offline(
    config: OriginBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    labels_path: Path,
    code_root: Path | None = None,
    output_path: Path | None = None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> dict[str, Any]:
    """Join private labels only after validating a sealed provider-phase checkpoint."""

    sealed_root = _project_path(project_root, config.sealed_run_path)
    if output_path is not None:
        assert_artifact_paths_safe(
            [output_path],
            sealed_root=sealed_root,
            protected_paths=[checkpoint_path, labels_path],
        )
    context = _prepare_execution(
        config,
        project_root=project_root,
        code_root=code_root,
        execution_selection=execution_selection,
        execution_selection_expectation=execution_selection_expectation,
    )
    checkpoint, entries = _load_sealed_checkpoint_for_scoring(checkpoint_path, context)
    # Deliberately last: provider output and every request binding are already sealed.
    labels, annotator_sha256 = _load_labels(labels_path, context)
    models: list[dict[str, Any]] = []
    executed_models = {item.model for item in context.models}
    for model in config.models:
        model_entries = [entry for entry in entries if entry.model == model.model]
        if model.model in executed_models:
            decision = (
                execution_selection.decision_for(model.model)
                if execution_selection is not None
                else None
            )
            eligibility = decision.status if decision is not None else "not_assessed"
            measurement = _quality_measurement(model_entries)
            measured = measurement["status"] == "measured"
            models.append(
                {
                    "model": model.model,
                    "label": model.label,
                    "contract_eligibility": eligibility,
                    "eligibility_reason_codes": (
                        decision.reason_codes if decision is not None else []
                    ),
                    "matched_quality_status": "executed",
                    "quality_status": (
                        "measured" if measured else "unmeasured_incomplete_provider_contract"
                    ),
                    "quality_measurement": measurement,
                    "quality": _score_model(model_entries, labels) if measured else None,
                    "terminal_telemetry": _aggregate_entries(model_entries),
                }
            )
            continue
        if execution_selection is None:
            raise RuntimeError("origin model was skipped without an execution selection")
        decision = execution_selection.decision_for(model.model)
        models.append(
            {
                "model": model.model,
                "label": model.label,
                "contract_eligibility": decision.status,
                "eligibility_reason_codes": decision.reason_codes,
                "matched_quality_status": "not_run",
                "not_run_reason": "contract_ineligible_on_sealed_smoke",
                "quality_status": "unmeasured_contract_ineligible",
                "quality_measurement": {
                    "status": "unmeasured",
                    "reason": "contract_ineligible_on_sealed_smoke",
                    "logical_calls_predeclared": 0,
                    "valid_contract_complete_assessments": 0,
                    "incomplete_or_failed_calls": 0,
                },
                "quality": None,
                "terminal_telemetry": None,
            }
        )
    result = {
        "schema_version": _SCORE_SCHEMA_VERSION,
        "bakeoff_id": config.bakeoff_id,
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "provider_phase_seal_sha256": checkpoint.provider_phase_seal_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "stage_contract_sha256": context.stage_contract_sha256,
        "execution_selection_sha256": (
            execution_selection.selection_sha256 if execution_selection is not None else None
        ),
        "private_labels_sha256": sha256_file(labels_path),
        "single_annotator_sha256": annotator_sha256,
        "models": models,
        "policy": {
            "provider_output_is_proposal_only": True,
            "automatic_positive_promotion": False,
        },
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
