"""Private, resumable comparisons of evidence-bound tuple resolvers.

The provider phase is deliberately label-blind.  It reads only candidates and layouts
from a verified sealed run, converts them into immutable tuple-resolution inputs, and
checkpoints each predeclared logical call immediately.  Candidate tuple fields are
withheld from the provider and provider output remains an untrusted proposal.  A
separate offline scorer joins exact-cover reference-authority records only after the
provider checkpoint is complete and sealed.

This module never resolves producer origin, eligibility, or export status.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

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
from proceedings_to_eee.resolution.origin_retrieval import layout_binding_sha256
from proceedings_to_eee.resolution.tuple_resolution import (
    TUPLE_SCHEMA_NAME,
    TUPLE_SYSTEM_PROMPT,
    TupleField,
    TupleFields,
    TupleResolutionAssessment,
    TupleResolutionInput,
    TupleWireProposal,
    build_tuple_resolution_input,
    propose_tuple_resolution,
    tuple_candidate_binding_sha256,
    tuple_resolution_prompt,
    tuple_resolution_provider_json_schema,
    tuple_wire_response_sha256,
    verify_tuple_resolution_proposal,
)
from proceedings_to_eee.run_seal import VerifiedRunSeal, verify_run_seal

HEX_64 = r"^[0-9a-f]{64}$"
_CHECKPOINT_VERSION = "tuple-bakeoff-checkpoint/0.2"
_ENTRY_VERSION = "tuple-bakeoff-entry/0.2"
_RESULT_VERSION = "tuple-bakeoff-provider-result/0.3"
_SCORE_VERSION = "tuple-bakeoff-score/0.4"


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


class TupleBakeoffModelSpec(StrictModel):
    model: str = Field(min_length=1)
    label: str = Field(min_length=1)


class FrozenTupleCaseLocator(StrictModel):
    """One exact candidate/layout/result-evidence binding in a sealed run."""

    case_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    observations_path: str = Field(min_length=1)
    layout_path: str = Field(min_length=1)
    candidate_binding_sha256: str = Field(pattern=HEX_64)
    layout_binding_sha256: str = Field(pattern=HEX_64)
    tuple_input_sha256: str = Field(pattern=HEX_64)
    result_evidence_binding_sha256: str = Field(pattern=HEX_64)

    @field_validator("observations_path", "layout_path")
    @classmethod
    def artifact_paths_are_canonical(cls, value: str) -> str:
        return _strict_relative_path(value, label="tuple artifact path")


class TupleBakeoffConfig(StrictModel):
    """Strict predeclared common-capability tuple-resolution experiment."""

    schema_version: Literal["tuple-bakeoff/0.1", "tuple-bakeoff/0.2"] = "tuple-bakeoff/0.1"
    bakeoff_id: str = Field(min_length=1)
    sealed_run_path: str = Field(min_length=1)
    sealed_run_seal_sha256: str = Field(pattern=HEX_64)
    sealed_run_tree_sha256: str = Field(pattern=HEX_64)
    models: list[TupleBakeoffModelSpec] = Field(min_length=2)
    cases: list[FrozenTupleCaseLocator] = Field(min_length=1)
    max_tokens: int = Field(default=3_000, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_effort: str | None = Field(default=None, min_length=1)
    seed: int | None = None
    require_parameters: Literal[True] = True
    fresh_repetitions: int = Field(default=2, ge=1)

    @model_validator(mode="before")
    @classmethod
    def repair04_defaults_use_the_production_contract(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("schema_version") == "tuple-bakeoff/0.2":
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
    def identifiers_are_unique(self) -> TupleBakeoffConfig:
        if self.schema_version == "tuple-bakeoff/0.2" and (
            self.max_tokens != 16_000
            or self.temperature is not None
            or self.reasoning_effort != "minimal"
            or self.seed is not None
            or not self.require_parameters
        ):
            raise ValueError("tuple-bakeoff/0.2 requires the production request contract")
        models = [item.model for item in self.models]
        cases = [item.case_id for item in self.cases]
        observations = [(item.observations_path, item.observation_id) for item in self.cases]
        if len(models) != len(set(models)):
            raise ValueError("tuple bake-off model IDs must be unique")
        if len(cases) != len(set(cases)):
            raise ValueError("tuple bake-off case IDs must be unique")
        if len(observations) != len(set(observations)):
            raise ValueError("tuple candidate locators must be unique")
        return self


class TupleReferenceAuthority(StrictModel):
    """Explicit provenance contract for one private tuple reference source."""

    schema_version: Literal["tuple-reference-authority/0.1"] = "tuple-reference-authority/0.1"
    authority_id: str = Field(min_length=1)
    authority_type: Literal["human_annotation", "development_reference_derivation"]
    human_annotation: bool

    @model_validator(mode="after")
    def authority_type_matches_human_flag(self) -> TupleReferenceAuthority:
        expected_human = self.authority_type == "human_annotation"
        if self.human_annotation is not expected_human:
            raise ValueError("tuple reference authority type is inconsistent with human_annotation")
        return self


class TupleReferenceLabel(StrictModel):
    """Private reference-authority tuple, loaded only by the scorer."""

    schema_version: Literal["tuple-bakeoff-label/0.2"] = "tuple-bakeoff-label/0.2"
    case_id: str = Field(min_length=1)
    tuple_input_sha256: str = Field(pattern=HEX_64)
    reference_authority: TupleReferenceAuthority
    reference_tuple: TupleFields
    unresolved_fields: list[TupleField]
    not_applicable_fields: list[TupleField]

    @model_validator(mode="after")
    def reference_and_unresolved_partition_fields(self) -> TupleReferenceLabel:
        if self.unresolved_fields != sorted(set(self.unresolved_fields), key=str):
            raise ValueError("tuple reference unresolved_fields must be sorted and unique")
        if self.not_applicable_fields != sorted(set(self.not_applicable_fields), key=str):
            raise ValueError("tuple reference not_applicable_fields must be sorted and unique")
        if set(self.unresolved_fields).intersection(self.not_applicable_fields):
            raise ValueError("tuple reference absence states overlap")
        if any(field is not TupleField.UNCERTAINTY for field in self.not_applicable_fields):
            raise ValueError("only tuple uncertainty has a deterministic not-applicable label")
        unresolved = set(self.unresolved_fields)
        not_applicable = set(self.not_applicable_fields)
        values = {
            TupleField.SYSTEM: self.reference_tuple.evaluated_system,
            TupleField.SYSTEM_VERSION: self.reference_tuple.system_version,
            TupleField.DATASET: self.reference_tuple.dataset,
            TupleField.DATASET_VERSION: self.reference_tuple.dataset_version,
            TupleField.METRIC: self.reference_tuple.metric,
            TupleField.DIRECTION: self.reference_tuple.direction,
            TupleField.SCALE: self.reference_tuple.scale,
            TupleField.VALUE: self.reference_tuple.value,
            TupleField.UNCERTAINTY: self.reference_tuple.uncertainty,
            TupleField.UNIT: self.reference_tuple.unit,
            TupleField.SETTING: self.reference_tuple.setting,
            TupleField.SCOPE: self.reference_tuple.scope,
        }
        for field, value in values.items():
            absent = field in unresolved or field in not_applicable
            if (value is None) != absent or (field in unresolved and field in not_applicable):
                raise ValueError("tuple reference does not partition semantic field states")
        return self


def load_tuple_bakeoff_config(path: Path) -> TupleBakeoffConfig:
    """Load one strict YAML or JSON-compatible tuple comparison."""

    return TupleBakeoffConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class _SafeError(StrictModel):
    stage: str
    type: str
    code: str
    validation_path: list[str | int] = Field(default_factory=list)
    validation_keyword: str | None = None


class _CheckpointEntry(StrictModel):
    schema_version: Literal["tuple-bakeoff-entry/0.2"] = _ENTRY_VERSION
    slot_id: str = Field(min_length=1)
    slot_contract_sha256: str = Field(pattern=HEX_64)
    case_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    repetition_index: int = Field(ge=1)
    candidate_binding_sha256: str = Field(pattern=HEX_64)
    tuple_input_sha256: str = Field(pattern=HEX_64)
    result_evidence_binding_sha256: str = Field(pattern=HEX_64)
    request: dict[str, Any]
    status: Literal[
        "success",
        "contract_failure",
        "response_validation_failure",
        "local_validation_failure",
        "provider_failure",
    ]
    schema_status: Literal["valid", "invalid", "not_observed"]
    contract_status: Literal["satisfied", "failed", "not_observed"]
    contract_failure_codes: list[str] = Field(default_factory=list)
    proposal: TupleWireProposal | None = None
    assessment: TupleResolutionAssessment | None = None
    call: ProviderCall | None = None
    error: _SafeError | None = None
    entry_sha256: str = Field(pattern=HEX_64)

    @model_validator(mode="after")
    def exact_terminal_shape(self) -> _CheckpointEntry:
        if len(self.contract_failure_codes) != len(set(self.contract_failure_codes)):
            raise ValueError("tuple contract failure codes must be unique")
        request_payload = {
            key: value for key, value in self.request.items() if key != "request_sha256"
        }
        if self.request.get("request_sha256") != _hash(request_payload):
            raise ValueError("tuple request_sha256 is invalid")
        successful_shape = self.status in {"success", "contract_failure"}
        if successful_shape and (
            self.proposal is None
            or self.assessment is None
            or self.call is None
            or self.schema_status != "valid"
        ):
            raise ValueError("schema-valid tuple entry is incomplete")
        if self.status == "success" and self.contract_status != "satisfied":
            raise ValueError("successful tuple entry must satisfy request contract")
        if self.status == "contract_failure" and self.contract_status != "failed":
            raise ValueError("tuple contract failure must retain failed contract status")
        if self.status in {"response_validation_failure", "local_validation_failure"} and (
            self.call is None or self.schema_status != "invalid"
        ):
            raise ValueError("tuple validation failure must retain its completed call")
        if self.status == "provider_failure" and (
            self.call is not None
            or self.proposal is not None
            or self.assessment is not None
            or self.schema_status != "not_observed"
            or self.contract_status != "not_observed"
        ):
            raise ValueError("tuple provider failure has an invalid shape")
        if self.proposal is not None:
            if (
                self.call is None
                or tuple_wire_response_sha256(self.proposal) != self.call.response_sha256
            ):
                raise ValueError("tuple proposal is not bound to provider response hash")
            if self.proposal.candidate_binding_sha256 != self.candidate_binding_sha256:
                raise ValueError("tuple proposal candidate binding is invalid")
            if self.proposal.result_evidence_binding_sha256 != self.result_evidence_binding_sha256:
                raise ValueError("tuple proposal evidence binding is invalid")
        if self.assessment is not None and (
            self.assessment.candidate_binding_sha256 != self.candidate_binding_sha256
            or self.assessment.input_sha256 != self.tuple_input_sha256
            or self.assessment.allows_origin_or_export is not False
        ):
            raise ValueError("tuple assessment binding is invalid")
        payload = self.model_dump(mode="json", exclude={"entry_sha256"})
        if self.entry_sha256 != _hash(payload):
            raise ValueError("tuple entry_sha256 is invalid")
        return self


class _Checkpoint(StrictModel):
    schema_version: Literal["tuple-bakeoff-checkpoint/0.2"] = _CHECKPOINT_VERSION
    bakeoff_id: str = Field(min_length=1)
    configuration_sha256: str = Field(pattern=HEX_64)
    run_contract_sha256: str = Field(pattern=HEX_64)
    status: Literal["in_progress", "sealed"]
    entries: dict[str, _CheckpointEntry]
    provider_phase_seal_sha256: str | None = Field(default=None, pattern=HEX_64)
    checkpoint_sha256: str = Field(pattern=HEX_64)

    @model_validator(mode="after")
    def exact_checkpoint_shape(self) -> _Checkpoint:
        if any(key != value.slot_id for key, value in self.entries.items()):
            raise ValueError("tuple checkpoint entry key mismatch")
        if self.status == "sealed":
            expected = _phase_seal(
                self.configuration_sha256, self.run_contract_sha256, self.entries
            )
            if self.provider_phase_seal_sha256 != expected:
                raise ValueError("tuple provider phase seal is invalid")
        elif self.provider_phase_seal_sha256 is not None:
            raise ValueError("in-progress tuple checkpoint cannot carry a seal")
        payload = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if self.checkpoint_sha256 != _hash(payload):
            raise ValueError("tuple checkpoint_sha256 is invalid")
        return self


@dataclass(frozen=True, slots=True)
class _PreparedCase:
    locator: FrozenTupleCaseLocator
    candidate: CandidateObservation
    layout: PdfLayout
    tuple_input: TupleResolutionInput
    observations_file_sha256: str
    layout_file_sha256: str


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    config: TupleBakeoffConfig
    configuration_sha256: str
    verified_seal: VerifiedRunSeal
    code: dict[str, Any]
    code_sha256: str
    stage_contract_sha256: str
    execution_selection: StageExecutionSelection | None
    models: tuple[TupleBakeoffModelSpec, ...]
    run_contract_sha256: str
    cases: tuple[_PreparedCase, ...]


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
        raise ValueError("tuple candidate locator did not resolve exactly one observation")
    return matches[0]


def tuple_stage_contract_sha256(config: TupleBakeoffConfig) -> str:
    """Bind the route-relevant contract shared by smoke and quality phases."""

    schema = tuple_resolution_provider_json_schema()
    contract = structured_request_contract(
        schema_name=TUPLE_SCHEMA_NAME,
        schema=schema,
        seed=config.seed,
        require_parameters=True,
    )
    return _hash(
        {
            "schema_version": "tuple-bakeoff-stage-contract/0.1",
            "models": [item.model for item in config.models],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "seed": config.seed,
            "require_parameters": True,
            "system_prompt_sha256": hashlib.sha256(TUPLE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "provider_schema_sha256": contract["schema"]["schema_sha256"],
            "request_contract_sha256": _hash(contract),
        }
    )


def _prepare_execution(
    config: TupleBakeoffConfig,
    *,
    project_root: Path,
    code_root: Path | None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> _ExecutionContext:
    sealed_root = _project_path(project_root, config.sealed_run_path)
    verified = verify_run_seal(sealed_root)
    if verified.seal_sha256 != config.sealed_run_seal_sha256:
        raise ValueError("sealed run seal hash does not match tuple configuration")
    if verified.tree_sha256 != config.sealed_run_tree_sha256:
        raise ValueError("sealed run tree hash does not match tuple configuration")
    inventory = _inventory(verified)
    prepared: list[_PreparedCase] = []
    for locator in config.cases:
        if locator.observations_path not in inventory or locator.layout_path not in inventory:
            raise ValueError("tuple artifact is absent from sealed inventory")
        observations_path = sealed_root / locator.observations_path
        layout_path = sealed_root / locator.layout_path
        observations_file_sha256 = str(inventory[locator.observations_path]["sha256"])
        layout_file_sha256 = str(inventory[locator.layout_path]["sha256"])
        if (
            sha256_file(observations_path) != observations_file_sha256
            or sha256_file(layout_path) != layout_file_sha256
        ):
            raise ValueError("sealed tuple artifact changed after verification")
        candidate = _load_candidate(observations_path, locator.observation_id)
        layout = PdfLayout.model_validate(read_json(layout_path))
        tuple_input = build_tuple_resolution_input(candidate, layout)
        if (
            sha256_file(observations_path) != observations_file_sha256
            or sha256_file(layout_path) != layout_file_sha256
        ):
            raise ValueError("sealed tuple artifact changed while loading")
        if candidate.paper_id != locator.paper_id:
            raise ValueError("tuple candidate paper_id does not match locator")
        actual = {
            "candidate_binding_sha256": tuple_candidate_binding_sha256(candidate),
            "layout_binding_sha256": layout_binding_sha256(layout),
            "tuple_input_sha256": tuple_input.input_sha256,
            "result_evidence_binding_sha256": tuple_input.result_evidence_binding_sha256,
        }
        expected = {
            key: getattr(locator, key)
            for key in (
                "candidate_binding_sha256",
                "layout_binding_sha256",
                "tuple_input_sha256",
                "result_evidence_binding_sha256",
            )
        }
        if actual != expected:
            raise ValueError("frozen tuple candidate/evidence binding is stale")
        prepared.append(
            _PreparedCase(
                locator=locator,
                candidate=candidate,
                layout=layout,
                tuple_input=tuple_input,
                observations_file_sha256=observations_file_sha256,
                layout_file_sha256=layout_file_sha256,
            )
        )
    verified_after = verify_run_seal(sealed_root)
    if (
        verified_after.seal_sha256 != verified.seal_sha256
        or verified_after.tree_sha256 != verified.tree_sha256
        or verified_after.files != verified.files
    ):
        raise ValueError("sealed run changed while preparing tuple bake-off")
    configuration_sha256 = _hash(config.model_dump(mode="json", exclude_none=False))
    code = _code_state(code_root if code_root is not None else project_root)
    code_sha256 = _hash(code)
    stage_contract_sha256 = tuple_stage_contract_sha256(config)
    selected_model_ids = selected_models_for_execution(
        execution_selection,
        expected_stage="tuple_resolution",
        declared_models=[item.model for item in config.models],
        stage_contract_sha256=stage_contract_sha256,
        code_sha256=code_sha256,
        expectation=execution_selection_expectation,
    )
    selected = set(selected_model_ids)
    models = tuple(item for item in config.models if item.model in selected)
    if tuple(item.model for item in models) != selected_model_ids:
        raise ValueError("tuple execution selection does not preserve declared model order")
    run_contract_sha256 = _hash(
        {
            "schema_version": "tuple-bakeoff-run-contract/0.1",
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
                    "tuple_input_sha256": case.locator.tuple_input_sha256,
                    "result_evidence_binding_sha256": (case.locator.result_evidence_binding_sha256),
                    "observations_file_sha256": case.observations_file_sha256,
                    "layout_file_sha256": case.layout_file_sha256,
                }
                for case in prepared
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
        cases=tuple(prepared),
    )


def _request_binding(
    context: _ExecutionContext,
    case: _PreparedCase,
    model: TupleBakeoffModelSpec,
    repetition_index: int,
) -> dict[str, Any]:
    user = tuple_resolution_prompt(case.tuple_input)
    messages = [
        {"role": "system", "content": TUPLE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    schema = tuple_resolution_provider_json_schema()
    contract = structured_request_contract(
        schema_name=TUPLE_SCHEMA_NAME,
        schema=schema,
        seed=context.config.seed,
        require_parameters=True,
        model=model.model,
        max_tokens=context.config.max_tokens,
    )
    request: dict[str, Any] = {
        "schema_version": "tuple-bakeoff-request/0.1",
        "case_id": case.locator.case_id,
        "model_requested": model.model,
        "provider_requested": "openrouter",
        "repetition_index": repetition_index,
        "candidate_binding_sha256": case.locator.candidate_binding_sha256,
        "layout_binding_sha256": case.locator.layout_binding_sha256,
        "tuple_input_sha256": case.locator.tuple_input_sha256,
        "result_evidence_binding_sha256": case.locator.result_evidence_binding_sha256,
        "prompt_sha256": hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "system_sha256": hashlib.sha256(TUPLE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
        "local_schema_sha256": _hash(schema),
        "provider_schema_sha256": contract["schema"]["schema_sha256"],
        "request_contract": contract,
        "request_contract_sha256": _hash(contract),
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
        "tuple_slot_"
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
    settings = request["settings"]
    failures: list[str] = []
    for name, actual, expected in (
        ("provider_requested", call.provider, request["provider_requested"]),
        ("model_requested", call.model_requested, request["model_requested"]),
        ("model_returned", call.model_returned, request["model_requested"]),
        ("prompt_sha256", call.prompt_sha256, request["prompt_sha256"]),
        ("temperature", call.temperature, settings["temperature"]),
        ("reasoning_effort", call.reasoning_effort, settings["reasoning_effort"]),
        ("max_tokens", call.max_tokens, settings["max_tokens"]),
        ("seed", call.seed, settings["seed"]),
        ("require_parameters", call.require_parameters, True),
    ):
        if actual != expected:
            failures.append(f"{name}_mismatch")
    if _hash(structured_request_contract_from_call(call)) != request["request_contract_sha256"]:
        failures.append("request_contract_mismatch")
    return ("failed", failures) if failures else ("satisfied", [])


def _verify_result_binding(
    proposal: TupleWireProposal,
    assessment: TupleResolutionAssessment,
    call: ProviderCall,
    case: _PreparedCase,
) -> None:
    expected = verify_tuple_resolution_proposal(request=case.tuple_input, proposal=proposal)
    if any(
        (
            tuple_wire_response_sha256(proposal) != call.response_sha256,
            proposal.candidate_binding_sha256 != case.locator.candidate_binding_sha256,
            proposal.result_evidence_binding_sha256 != case.locator.result_evidence_binding_sha256,
            assessment != expected,
            assessment.input_sha256 != case.locator.tuple_input_sha256,
            assessment.allows_origin_or_export is not False,
        )
    ):
        raise ValueError("tuple result is not bound to its candidate/evidence request")


def _entry_payload(
    *,
    request: dict[str, Any],
    case: _PreparedCase,
    model: TupleBakeoffModelSpec,
    repetition_index: int,
    status: str,
    schema_status: str,
    contract_status: str,
    contract_failure_codes: list[str] | None = None,
    proposal: TupleWireProposal | None = None,
    assessment: TupleResolutionAssessment | None = None,
    call: ProviderCall | None = None,
    error: _SafeError | None = None,
) -> _CheckpointEntry:
    slot_id = _slot_id(request)
    payload: dict[str, Any] = {
        "schema_version": _ENTRY_VERSION,
        "slot_id": slot_id,
        "slot_contract_sha256": _hash(
            {
                "slot_id": slot_id,
                "request_sha256": request["request_sha256"],
                "candidate_binding_sha256": case.locator.candidate_binding_sha256,
                "tuple_input_sha256": case.locator.tuple_input_sha256,
                "result_evidence_binding_sha256": (case.locator.result_evidence_binding_sha256),
            }
        ),
        "case_id": case.locator.case_id,
        "model": model.model,
        "repetition_index": repetition_index,
        "candidate_binding_sha256": case.locator.candidate_binding_sha256,
        "tuple_input_sha256": case.locator.tuple_input_sha256,
        "result_evidence_binding_sha256": case.locator.result_evidence_binding_sha256,
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


def _entry_reusable(
    entry: _CheckpointEntry,
    *,
    request: dict[str, Any],
    case: _PreparedCase,
    model: TupleBakeoffModelSpec,
    repetition_index: int,
) -> bool:
    slot_id = _slot_id(request)
    expected_contract = _hash(
        {
            "slot_id": slot_id,
            "request_sha256": request["request_sha256"],
            "candidate_binding_sha256": case.locator.candidate_binding_sha256,
            "tuple_input_sha256": case.locator.tuple_input_sha256,
            "result_evidence_binding_sha256": case.locator.result_evidence_binding_sha256,
        }
    )
    if (
        entry.slot_id != slot_id
        or entry.slot_contract_sha256 != expected_contract
        or entry.case_id != case.locator.case_id
        or entry.model != model.model
        or entry.repetition_index != repetition_index
        or entry.candidate_binding_sha256 != case.locator.candidate_binding_sha256
        or entry.tuple_input_sha256 != case.locator.tuple_input_sha256
        or entry.result_evidence_binding_sha256 != case.locator.result_evidence_binding_sha256
        or entry.request != request
    ):
        return False
    if entry.call is not None:
        status, failures = _contract_assessment(entry.call, request)
        if status != entry.contract_status or failures != entry.contract_failure_codes:
            return False
    elif entry.contract_status != "not_observed":
        return False
    if entry.proposal is not None or entry.assessment is not None:
        if entry.proposal is None or entry.assessment is None or entry.call is None:
            return False
        try:
            _verify_result_binding(entry.proposal, entry.assessment, entry.call, case)
        except ValueError:
            return False
    return True


def _phase_seal(
    configuration_sha256: str,
    run_contract_sha256: str,
    entries: dict[str, _CheckpointEntry],
) -> str:
    return _hash(
        {
            "schema_version": "tuple-bakeoff-provider-seal/0.1",
            "configuration_sha256": configuration_sha256,
            "run_contract_sha256": run_contract_sha256,
            "entry_sha256s": {key: value.entry_sha256 for key, value in sorted(entries.items())},
        }
    )


def _checkpoint(
    context: _ExecutionContext,
    entries: dict[str, _CheckpointEntry],
    *,
    sealed: bool,
) -> _Checkpoint:
    payload: dict[str, Any] = {
        "schema_version": _CHECKPOINT_VERSION,
        "bakeoff_id": context.config.bakeoff_id,
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "status": "sealed" if sealed else "in_progress",
        "entries": entries,
        "provider_phase_seal_sha256": (
            _phase_seal(context.configuration_sha256, context.run_contract_sha256, entries)
            if sealed
            else None
        ),
    }
    payload["checkpoint_sha256"] = _hash(payload)
    return _Checkpoint.model_validate(payload)


def _write_checkpoint(
    path: Path,
    context: _ExecutionContext,
    entries: dict[str, _CheckpointEntry],
    *,
    sealed: bool,
) -> _Checkpoint:
    checkpoint = _checkpoint(context, entries, sealed=sealed)
    write_json(path, checkpoint.model_dump(mode="json", exclude_none=False))
    return checkpoint


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


def _run_slot(
    context: _ExecutionContext,
    case: _PreparedCase,
    model: TupleBakeoffModelSpec,
    repetition_index: int,
    client: Any,
) -> _CheckpointEntry:
    request = _request_binding(context, case, model, repetition_index)
    candidate_before = tuple_candidate_binding_sha256(case.candidate)
    try:
        proposal, assessment, call = propose_tuple_resolution(
            client=client,
            model=model.model,
            request=case.tuple_input,
            max_tokens=context.config.max_tokens,
            temperature=context.config.temperature,
            reasoning_effort=context.config.reasoning_effort,
            seed=context.config.seed,
            require_parameters=True,
        )
    except ProviderBudgetError:
        raise
    except ProviderResponseValidationError as error:
        contract_status, failures = _contract_assessment(error.call, request)
        return _entry_payload(
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
    except (ValidationError, ValueError) as error:
        return _entry_payload(
            request=request,
            case=case,
            model=model,
            repetition_index=repetition_index,
            status="provider_failure",
            schema_status="not_observed",
            contract_status="not_observed",
            error=_safe_error("local_tuple_validation", error),
        )
    except Exception as error:
        return _entry_payload(
            request=request,
            case=case,
            model=model,
            repetition_index=repetition_index,
            status="provider_failure",
            schema_status="not_observed",
            contract_status="not_observed",
            error=_safe_error("provider_call", error),
        )
    if tuple_candidate_binding_sha256(case.candidate) != candidate_before:
        raise RuntimeError("production tuple resolver mutated its candidate input")
    contract_status, failures = _contract_assessment(call, request)
    try:
        _verify_result_binding(proposal, assessment, call, case)
    except ValueError as error:
        return _entry_payload(
            request=request,
            case=case,
            model=model,
            repetition_index=repetition_index,
            status="local_validation_failure",
            schema_status="invalid",
            contract_status=contract_status,
            contract_failure_codes=failures,
            call=call,
            error=_safe_error("tuple_result_binding", error),
        )
    return _entry_payload(
        request=request,
        case=case,
        model=model,
        repetition_index=repetition_index,
        status="success" if contract_status == "satisfied" else "contract_failure",
        schema_status="valid",
        contract_status=contract_status,
        contract_failure_codes=failures,
        proposal=proposal,
        assessment=assessment,
        call=call,
    )


def _public_call(call: ProviderCall) -> dict[str, Any]:
    return public_provider_call(call)


def _public_entry(entry: _CheckpointEntry, *, resumed: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "slot_id": entry.slot_id,
        "slot_contract_sha256": entry.slot_contract_sha256,
        "case_id": entry.case_id,
        "model": entry.model,
        "repetition_index": entry.repetition_index,
        "candidate_binding_sha256": entry.candidate_binding_sha256,
        "tuple_input_sha256": entry.tuple_input_sha256,
        "result_evidence_binding_sha256": entry.result_evidence_binding_sha256,
        "request": entry.request,
        "status": entry.status,
        "schema_status": entry.schema_status,
        "contract_status": entry.contract_status,
        "contract_failure_codes": entry.contract_failure_codes,
        "entry_sha256": entry.entry_sha256,
        "resumed": resumed,
    }
    if entry.proposal is not None and entry.assessment is not None:
        result["tuple_review"] = {
            "proposal_sha256": _hash(entry.proposal),
            "assessment_sha256": _hash(entry.assessment),
            "decision": entry.assessment.decision.value,
            "field_states": entry.assessment.field_states.model_dump(mode="json"),
            "unresolved_fields": [field.value for field in entry.assessment.unresolved_fields],
            "not_applicable_fields": [
                field.value for field in entry.assessment.not_applicable_fields
            ],
            "unsafe_value_or_scope": entry.assessment.unsafe_value_or_scope,
            "allows_origin_or_export": False,
        }
    if entry.call is not None:
        result["call"] = _public_call(entry.call)
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


def _aggregate(entries: list[_CheckpointEntry]) -> dict[str, Any]:
    first = [entry for entry in entries if entry.repetition_index == 1]

    def schema_report(values: list[_CheckpointEntry]) -> dict[str, Any]:
        valid = sum(entry.schema_status == "valid" for entry in values)
        invalid = sum(entry.schema_status == "invalid" for entry in values)
        observed = valid + invalid
        return {
            "logical_calls": len(values),
            "valid": valid,
            "invalid": invalid,
            "not_observed": len(values) - observed,
            "valid_of_observed": _rate(valid, observed),
            "valid_end_to_end": _rate(valid, len(values)),
        }

    calls = [entry.call for entry in entries if entry.call is not None]
    usage: dict[str, Any] = {}
    for name in (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cost_usd",
    ):
        values = [getattr(call, name) for call in calls if getattr(call, name) is not None]
        usage[name] = {
            "total": round(sum(values), 8),
            "reported_calls": len(values),
            "missing_calls": len(entries) - len(values),
        }
    latencies = [float(call.latency_seconds) for call in calls]
    usage["latency_seconds"] = {
        "total": round(sum(latencies), 6),
        "mean": round(sum(latencies) / len(latencies), 6) if latencies else None,
        "p50": _percentile(latencies, 0.5),
        "p95": _percentile(latencies, 0.95),
        "max": round(max(latencies), 6) if latencies else None,
        "reported_calls": len(latencies),
        "missing_calls": len(entries) - len(latencies),
    }
    satisfied = sum(entry.contract_status == "satisfied" for entry in entries)
    return {
        "execution": {
            "logical_calls_predeclared": len(entries),
            "terminal_entries": len(entries),
            "success": sum(entry.status == "success" for entry in entries),
            "contract_failure": sum(entry.status == "contract_failure" for entry in entries),
            "response_validation_failure": sum(
                entry.status == "response_validation_failure" for entry in entries
            ),
            "local_validation_failure": sum(
                entry.status == "local_validation_failure" for entry in entries
            ),
            "provider_failure": sum(entry.status == "provider_failure" for entry in entries),
        },
        "wire_schema": {
            "all_repetitions": schema_report(entries),
            "first_pass": schema_report(first),
        },
        "contract": {
            "satisfied": satisfied,
            "failed": sum(entry.contract_status == "failed" for entry in entries),
            "not_observed": sum(entry.contract_status == "not_observed" for entry in entries),
            "satisfaction_rate": _rate(satisfied, len(entries)),
        },
        "failure_diagnostics": _failure_diagnostics(entries),
        "usage": usage,
    }


def _invalid_json_finish_reason(entry: _CheckpointEntry) -> str:
    if entry.call is None or entry.call.finish_reason is None:
        return "missing"
    if entry.call.finish_reason in {"length", "error"}:
        return entry.call.finish_reason
    return "other"


def _failure_diagnostics(entries: list[_CheckpointEntry]) -> dict[str, Any]:
    failures = [entry for entry in entries if entry.status != "success"]
    errors = [entry for entry in failures if entry.error is not None]
    error_code_counts = Counter(entry.error.code for entry in errors if entry.error is not None)
    invalid_json = [
        entry for entry in errors if entry.error is not None and entry.error.code == "invalid_json"
    ]
    finish_reason_counts = Counter(_invalid_json_finish_reason(entry) for entry in invalid_json)
    wire_validation = [
        entry
        for entry in errors
        if entry.error is not None and entry.error.code == "wire_validation"
    ]
    validation_path_counts = Counter(
        tuple(entry.error.validation_path) for entry in wire_validation if entry.error is not None
    )
    validation_keyword_counts = Counter(
        entry.error.validation_keyword for entry in wire_validation if entry.error is not None
    )
    return {
        "technical_failure_entries": len(failures),
        "entries_with_error_metadata": len(errors),
        "error_code_counts": [
            {"code": code, "count": count} for code, count in sorted(error_code_counts.items())
        ],
        "invalid_json": {
            "count": len(invalid_json),
            "finish_reason_counts": {
                category: finish_reason_counts[category]
                for category in ("length", "error", "missing", "other")
            },
        },
        "wire_validation": {
            "count": len(wire_validation),
            "validation_path_counts": [
                {"validation_path": list(path), "count": count}
                for path, count in sorted(
                    validation_path_counts.items(),
                    key=lambda item: (
                        len(item[0]),
                        canonical_json_bytes(list(item[0])),
                    ),
                )
            ],
            "validation_keyword_counts": [
                {"validation_keyword": keyword, "count": count}
                for keyword, count in sorted(
                    validation_keyword_counts.items(),
                    key=lambda item: (item[0] is not None, item[0] or ""),
                )
            ],
        },
    }


def _technical_outcome_payload(entry: _CheckpointEntry) -> dict[str, Any]:
    error = entry.error
    return {
        "status": entry.status,
        "schema_status": entry.schema_status,
        "contract_status": entry.contract_status,
        "contract_failure_codes": entry.contract_failure_codes,
        "error": (
            {
                "stage": error.stage,
                "type": error.type,
                "code": error.code,
                "validation_path": error.validation_path,
                "validation_keyword": error.validation_keyword,
                "invalid_json_finish_reason": (
                    _invalid_json_finish_reason(entry) if error.code == "invalid_json" else None
                ),
            }
            if error is not None
            else None
        ),
    }


def _semantic_assessment_payload(assessment: TupleResolutionAssessment) -> dict[str, Any]:
    return {
        "decision": assessment.decision,
        "accepted_tuple": assessment.accepted_tuple,
        "field_states": assessment.field_states,
        "unresolved_fields": assessment.unresolved_fields,
        "not_applicable_fields": assessment.not_applicable_fields,
        "unsafe_value_or_scope": assessment.unsafe_value_or_scope,
    }


def _stability(entries: list[_CheckpointEntry]) -> dict[str, Any]:
    by_case: dict[str, list[_CheckpointEntry]] = {}
    for entry in entries:
        by_case.setdefault(entry.case_id, []).append(entry)
    technical_pairs = 0
    matching_technical_outcomes = 0
    semantic_pairs = 0
    matching_semantic_decisions = 0
    exact_assessments = 0
    for case_entries in by_case.values():
        for left, right in combinations(case_entries, 2):
            technical_pairs += 1
            matching_technical_outcomes += int(
                _hash(_technical_outcome_payload(left)) == _hash(_technical_outcome_payload(right))
            )
            if (
                left.status == "success"
                and right.status == "success"
                and left.assessment is not None
                and right.assessment is not None
            ):
                semantic_pairs += 1
                matching_semantic_decisions += int(
                    left.assessment.decision == right.assessment.decision
                )
                exact_assessments += int(
                    _hash(_semantic_assessment_payload(left.assessment))
                    == _hash(_semantic_assessment_payload(right.assessment))
                )
    return {
        "repetition_pair_denominator": technical_pairs,
        "technical_outcome_pair_denominator": technical_pairs,
        "matching_technical_outcomes": matching_technical_outcomes,
        "technical_outcome_match_rate": _rate(matching_technical_outcomes, technical_pairs),
        "semantic_decision_pair_denominator": semantic_pairs,
        "matching_semantic_decisions": matching_semantic_decisions,
        "semantic_decision_match_rate": _rate(matching_semantic_decisions, semantic_pairs),
        "decision_pair_denominator": semantic_pairs,
        "matching_decisions": matching_semantic_decisions,
        "decision_match_rate": _rate(matching_semantic_decisions, semantic_pairs),
        "semantic_assessment_pair_denominator": semantic_pairs,
        "assessment_pair_denominator": semantic_pairs,
        "exact_semantic_assessment_matches": exact_assessments,
        "exact_semantic_assessment_match_rate": _rate(exact_assessments, semantic_pairs),
    }


def run_tuple_bakeoff_provider_phase(
    config: TupleBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    client: Any,
    code_root: Path | None = None,
    output_path: Path | None = None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> dict[str, Any]:
    """Run or exactly resume every predeclared label-blind tuple call."""

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
    _write_checkpoint(checkpoint_path, context, entries, sealed=False)
    expected_slots: list[str] = []
    reused: set[str] = set()
    executed: set[str] = set()
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
                    reused.add(slot_id)
                    continue
                if cached is not None:
                    checkpoint_rejections.append("checkpoint_entry_stale_or_tampered")
                    entries.pop(slot_id, None)
                    _write_checkpoint(checkpoint_path, context, entries, sealed=False)
                entry = _run_slot(context, case, model, repetition_index, client)
                entries[slot_id] = entry
                executed.add(slot_id)
                _write_checkpoint(checkpoint_path, context, entries, sealed=False)
    expected_set = set(expected_slots)
    unexpected = set(entries) - expected_set
    if unexpected:
        checkpoint_rejections.append("checkpoint_unexpected_entries_removed")
        entries = {key: value for key, value in entries.items() if key in expected_set}
    if set(entries) != expected_set:
        raise RuntimeError("tuple checkpoint does not partition predeclared slots")
    sealed = _write_checkpoint(checkpoint_path, context, entries, sealed=True)
    ordered = [entries[key] for key in expected_slots]
    models: list[dict[str, Any]] = []
    executed_models = {item.model for item in context.models}
    for model in config.models:
        model_entries = [entry for entry in ordered if entry.model == model.model]
        if model.model in executed_models:
            decision = (
                execution_selection.decision_for(model.model)
                if execution_selection is not None
                else None
            )
            models.append(
                {
                    "model": model.model,
                    "label": model.label,
                    "contract_eligibility": (
                        decision.status if decision is not None else "not_assessed"
                    ),
                    "eligibility_reason_codes": (
                        decision.reason_codes if decision is not None else []
                    ),
                    "matched_quality_status": "executed",
                    "quality_status": "pending_offline_scoring",
                    "quality": None,
                    "aggregate": _aggregate(model_entries),
                    "stability": _stability(model_entries),
                }
            )
            continue
        if execution_selection is None:
            raise RuntimeError("tuple model was skipped without an execution selection")
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
                "quality": None,
                "aggregate": None,
                "stability": None,
            }
        )
    result = {
        "schema_version": _RESULT_VERSION,
        "bakeoff_id": config.bakeoff_id,
        "status": "sealed",
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "provider_phase_seal_sha256": sealed.provider_phase_seal_sha256,
        "checkpoint_sha256": sealed.checkpoint_sha256,
        "source": {
            "sealed_run_seal_sha256": context.verified_seal.seal_sha256,
            "sealed_run_tree_sha256": context.verified_seal.tree_sha256,
            "cases": [
                {
                    "case_id": case.locator.case_id,
                    "candidate_binding_sha256": case.locator.candidate_binding_sha256,
                    "layout_binding_sha256": case.locator.layout_binding_sha256,
                    "tuple_input_sha256": case.locator.tuple_input_sha256,
                    "result_evidence_binding_sha256": (case.locator.result_evidence_binding_sha256),
                    "observations_file_sha256": case.observations_file_sha256,
                    "layout_file_sha256": case.layout_file_sha256,
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
            "logical_calls_reused": len(reused),
            "logical_calls_executed": len(executed),
        },
        "aggregate": _aggregate(ordered),
        "models": models,
        "entries": [_public_entry(entry, resumed=entry.slot_id in reused) for entry in ordered],
        "privacy": {
            "labels_loaded_during_provider_phase": False,
            "candidate_tuple_fields_withheld": True,
            "observation_ids_in_public_result": False,
            "evidence_or_row_ids_in_public_result": False,
            "raw_quotes_or_evidence_in_public_result": False,
            "exact_inputs_checkpoint_and_sealed_run_only": True,
        },
        "origin_or_export_decisions": False,
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
        raise ValueError("tuple provider phase checkpoint is not sealed")
    if (
        checkpoint.configuration_sha256 != context.configuration_sha256
        or checkpoint.run_contract_sha256 != context.run_contract_sha256
    ):
        raise ValueError("tuple provider phase checkpoint is stale")
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
                    raise ValueError("tuple checkpoint contains an invalid slot")
                expected.append(entry)
    if len(expected) != len(checkpoint.entries):
        raise ValueError("tuple checkpoint contains unexpected slots")
    return checkpoint, expected


def derive_tuple_execution_selection(
    config: TupleBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    experiment_id: str,
    public_manifest_sha256: str,
    expected_fresh_repetitions: int,
    wire_schema_gate: str = "0.99",
    code_root: Path | None = None,
) -> StageExecutionSelection:
    """Derive eligibility from every call in the frozen repeated tuple qualification."""

    if expected_fresh_repetitions < 1 or config.fresh_repetitions != expected_fresh_repetitions:
        raise ValueError("tuple qualification repetition count differs from its frozen contract")
    context = _prepare_execution(config, project_root=project_root, code_root=code_root)
    checkpoint, entries = _load_sealed_checkpoint_for_scoring(checkpoint_path, context)
    if checkpoint.provider_phase_seal_sha256 is None:
        raise ValueError("tuple contract smoke is missing its provider phase seal")
    return derive_stage_execution_selection(
        experiment_id=experiment_id,
        stage="tuple_resolution",
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
        expected_slots_per_model=len(context.cases) * expected_fresh_repetitions,
        entries=entries,
        expected_repetitions=expected_fresh_repetitions,
    )


def _load_labels(
    path: Path,
    context: _ExecutionContext,
) -> tuple[dict[str, TupleReferenceLabel], TupleReferenceAuthority]:
    labels = [
        TupleReferenceLabel.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_case = {label.case_id: label for label in labels}
    if len(by_case) != len(labels):
        raise ValueError("tuple label case IDs must be unique")
    cases = {case.locator.case_id: case for case in context.cases}
    if set(by_case) != set(cases):
        raise ValueError("tuple labels do not exactly cover configured cases")
    for case_id, case in cases.items():
        if by_case[case_id].tuple_input_sha256 != case.locator.tuple_input_sha256:
            raise ValueError("tuple label input binding mismatch")
    authority_contracts = {
        (
            label.reference_authority.authority_id,
            label.reference_authority.authority_type,
            label.reference_authority.human_annotation,
        )
        for label in labels
    }
    if len(authority_contracts) != 1:
        raise ValueError(
            "tuple bake-off requires exactly one consistent reference-authority contract"
        )
    authority = next(iter(labels)).reference_authority
    return by_case, authority


def _field_value(fields: TupleFields, field: TupleField) -> Any:
    return getattr(
        fields,
        {
            TupleField.SYSTEM: "evaluated_system",
            TupleField.SYSTEM_VERSION: "system_version",
            TupleField.DATASET: "dataset",
            TupleField.DATASET_VERSION: "dataset_version",
            TupleField.METRIC: "metric",
            TupleField.DIRECTION: "direction",
            TupleField.SCALE: "scale",
            TupleField.VALUE: "value",
            TupleField.UNCERTAINTY: "uncertainty",
            TupleField.UNIT: "unit",
            TupleField.SETTING: "setting",
            TupleField.SCOPE: "scope",
        }[field],
    )


def _score_model(
    entries: list[_CheckpointEntry],
    labels: dict[str, TupleReferenceLabel],
) -> dict[str, Any]:
    total = len(entries)
    unsafe = sum(
        entry.assessment is not None and entry.assessment.unsafe_value_or_scope for entry in entries
    )
    field_stats: dict[str, dict[str, Any]] = {}
    exact_joint = 0
    joint_denominator = 0
    unresolved_proposals = 0
    review_routes = 0
    for field in TupleField:
        reference_resolved = 0
        predicted_resolved = 0
        exact_values = 0
        exact_states = 0
        exact_semantics = 0
        reference_states = {"resolved": 0, "not_applicable": 0, "unresolved": 0}
        predicted_states = {
            "verified": 0,
            "not_applicable": 0,
            "unresolved": 0,
            "unsupported": 0,
            "unobserved": 0,
        }
        for entry in entries:
            label = labels[entry.case_id]
            expected = _field_value(label.reference_tuple, field)
            actual = (
                _field_value(entry.assessment.accepted_tuple, field)
                if entry.status == "success" and entry.assessment is not None
                else None
            )
            if field in label.not_applicable_fields:
                reference_state = "not_applicable"
                expected_assessment_state = "not_applicable"
            elif field in label.unresolved_fields:
                reference_state = "unresolved"
                expected_assessment_state = "unresolved"
            else:
                reference_state = "resolved"
                expected_assessment_state = "verified"
            reference_states[reference_state] += 1
            if entry.status == "success" and entry.assessment is not None:
                predicted_state = entry.assessment.field_states.for_field(field).value
            else:
                predicted_state = "unobserved"
            predicted_states[predicted_state] += 1
            state_exact = predicted_state == expected_assessment_state
            value_exact = expected is not None and actual == expected
            reference_resolved += int(expected is not None)
            predicted_resolved += int(actual is not None)
            exact_values += int(value_exact)
            exact_states += int(state_exact)
            exact_semantics += int(state_exact and (expected is None or value_exact))
        field_stats[field.value] = {
            "reference_states": reference_states,
            "predicted_states": predicted_states,
            "reference_resolved": reference_resolved,
            "predicted_resolved": predicted_resolved,
            "exact_value_matches": exact_values,
            "exact_value_accuracy_on_reference_resolved": _rate(exact_values, reference_resolved),
            "exact_state_matches": exact_states,
            "exact_state_accuracy": _rate(exact_states, total),
            "exact_semantic_matches": exact_semantics,
            "exact_semantic_accuracy": _rate(exact_semantics, total),
        }
    for entry in entries:
        label = labels[entry.case_id]
        if entry.status == "success" and entry.assessment is not None:
            joint_denominator += 1
            exact_joint += int(
                entry.assessment.accepted_tuple == label.reference_tuple
                and entry.assessment.unresolved_fields == label.unresolved_fields
                and entry.assessment.not_applicable_fields == label.not_applicable_fields
            )
            unresolved_proposals += int(bool(entry.assessment.unresolved_fields))
            review_routes += int(entry.assessment.decision.value in {"review", "reject"})
        else:
            unresolved_proposals += 1
            review_routes += 1
    return {
        "unsafe_value_or_scope_proposals": {
            "count": unsafe,
            "rate_of_all_predeclared_calls": _rate(unsafe, total),
        },
        "evidence_bound_field_accuracy": field_stats,
        "exact_joint_tuple_accuracy": _rate(exact_joint, joint_denominator),
        "exact_denominators": {
            "logical_calls_predeclared": total,
            "terminal_entries": len(entries),
            "contract_satisfied_entries": sum(
                entry.contract_status == "satisfied" for entry in entries
            ),
            "locally_verified_tuple_assessments": joint_denominator,
            "joint_reference_comparisons": joint_denominator,
        },
        "unresolved_burden": {
            "count": unresolved_proposals,
            "rate": _rate(unresolved_proposals, total),
        },
        "review_burden": {
            "count": review_routes,
            "rate": _rate(review_routes, total),
            "includes": ["review", "reject", "unresolved_call"],
        },
        "wire_schema": _aggregate(entries)["wire_schema"],
        "contract": _aggregate(entries)["contract"],
        "stability": _stability(entries),
        "usage": _aggregate(entries)["usage"],
        "origin_or_export_decisions": 0,
    }


def _quality_partition_accounting(
    entries: list[_CheckpointEntry],
    *,
    expected_calls: int,
) -> dict[str, Any]:
    terminal = len(entries)
    schema_valid = sum(entry.schema_status == "valid" for entry in entries)
    contract_satisfied = sum(entry.contract_status == "satisfied" for entry in entries)
    assessments = sum(entry.assessment is not None for entry in entries)
    reasons: list[str] = []
    if terminal != expected_calls:
        reasons.append("terminal_partition_incomplete")
    if schema_valid != expected_calls:
        reasons.append("wire_schema_invalid_or_unobserved")
    if contract_satisfied != expected_calls:
        reasons.append("request_contract_unsatisfied_or_unobserved")
    if assessments != expected_calls:
        reasons.append("semantic_assessment_missing")
    return {
        "logical_calls_predeclared": expected_calls,
        "terminal_entries": terminal,
        "wire_schema_valid": schema_valid,
        "request_contract_satisfied": contract_satisfied,
        "deterministic_semantic_assessments": assessments,
        "reason_codes": reasons,
        "complete": not reasons,
    }


def score_tuple_bakeoff_offline(
    config: TupleBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    labels_path: Path,
    code_root: Path | None = None,
    output_path: Path | None = None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> dict[str, Any]:
    """Load private reference tuples only after validating the provider-phase seal."""

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
    # Deliberately last: all provider outputs and request bindings are already sealed.
    labels, reference_authority = _load_labels(labels_path, context)
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
            expected_calls = len(context.cases) * config.fresh_repetitions
            accounting = _quality_partition_accounting(
                model_entries,
                expected_calls=expected_calls,
            )
            measured = bool(accounting["complete"])
            models.append(
                {
                    "model": model.model,
                    "label": model.label,
                    "contract_eligibility": (
                        decision.status if decision is not None else "not_assessed"
                    ),
                    "eligibility_reason_codes": (
                        decision.reason_codes if decision is not None else []
                    ),
                    "matched_quality_status": "executed",
                    "quality_status": (
                        "measured"
                        if measured
                        else "unmeasured_incomplete_matched_quality_partition"
                    ),
                    "quality_accounting": accounting,
                    "failure_diagnostics": _failure_diagnostics(model_entries),
                    "stability": _stability(model_entries),
                    "quality": _score_model(model_entries, labels) if measured else None,
                }
            )
            continue
        if execution_selection is None:
            raise RuntimeError("tuple model was skipped without an execution selection")
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
                "quality_accounting": None,
                "failure_diagnostics": None,
                "stability": None,
                "quality": None,
            }
        )
    result = {
        "schema_version": _SCORE_VERSION,
        "bakeoff_id": config.bakeoff_id,
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "provider_phase_seal_sha256": checkpoint.provider_phase_seal_sha256,
        "stage_contract_sha256": context.stage_contract_sha256,
        "execution_selection_sha256": (
            execution_selection.selection_sha256 if execution_selection is not None else None
        ),
        "labels_sha256": sha256_file(labels_path),
        "reference_authority": {
            "authority_type": reference_authority.authority_type,
            "human_annotation": reference_authority.human_annotation,
        },
        "models": models,
        "privacy": {
            "labels_loaded_after_provider_phase_sealed": True,
            "observation_ids_in_score": False,
            "evidence_or_row_ids_in_score": False,
            "raw_quotes_or_evidence_in_score": False,
        },
        "origin_or_export_decisions": False,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
