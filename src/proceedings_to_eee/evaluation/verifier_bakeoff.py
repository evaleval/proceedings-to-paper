"""Private, resumable comparisons of independent candidate verifiers.

The provider phase is label-blind and accepts only candidates whose deterministic
binding to an exact result block and evidence anchor was frozen in a verified sealed
run.  It checkpoints every logical call before another call begins.  A separate
offline scorer joins private single-reviewer labels only after the provider phase is
complete and sealed.

Verifier output is always a review sidecar.  This module never mutates a candidate or
promotes it into an export path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.domain.observation import CandidateObservation, EvidenceAnchor
from proceedings_to_eee.evaluation.artifact_safety import assert_artifact_paths_safe
from proceedings_to_eee.evaluation.staged_eligibility import (
    StageExecutionSelection,
    StageExecutionSelectionExpectation,
    derive_stage_execution_selection,
    selected_models_for_execution,
)
from proceedings_to_eee.extraction.result_blocks import ResultBlock
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
from proceedings_to_eee.run_seal import VerifiedRunSeal, verify_run_seal
from proceedings_to_eee.verification.binding import (
    bind_candidate_block,
    frozen_evidence_block,
)
from proceedings_to_eee.verification.independent import (
    VERIFIER_REQUEST_SETTINGS,
    VERIFIER_SCHEMA_NAME,
    VERIFIER_SYSTEM_PROMPT,
    CandidateVerification,
    FrozenEvidenceBlock,
    VerificationRequest,
    contextualize_verification,
    verification_prompt,
    verification_provider_json_schema,
    verifier_evidence_block_sha256,
    verifier_request_contract,
    verify_candidate,
)

HEX_64 = r"^[0-9a-f]{64}$"
_CHECKPOINT_VERSION = "verifier-bakeoff-checkpoint/0.3"
_ENTRY_VERSION = "verifier-bakeoff-entry/0.3"
_RESULT_VERSION = "verifier-bakeoff-provider-result/0.3"
_SCORE_VERSION = "verifier-bakeoff-score/0.4"
_LABEL_CLASSES = ("good_candidate", "bad_candidate", "uncertain")
_PREDICTIONS = ("accept", "reject", "review", "unresolved")


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


def verifier_candidate_binding_sha256(candidate: CandidateObservation) -> str:
    """Return the complete logical binding for one frozen candidate."""

    return _hash(candidate)


def verifier_result_block_binding_sha256(block: ResultBlock) -> str:
    """Return the complete logical binding for one frozen result block."""

    return _hash(block)


def verifier_evidence_anchor_binding_sha256(anchor: EvidenceAnchor) -> str:
    """Return the complete logical binding for the selected evidence anchor."""

    return _hash(anchor)


def verifier_evidence_block_binding_sha256(block: FrozenEvidenceBlock) -> str:
    """Return the complete verifier-input binding, including exact block text."""

    return _hash(block)


def verifier_candidate_block_binding_sha256(
    *,
    candidate: CandidateObservation,
    result_block: ResultBlock,
    evidence_anchor: EvidenceAnchor,
    evidence_block: FrozenEvidenceBlock,
) -> str:
    """Bind the candidate and deterministic result-block/anchor projection together."""

    return _hash(
        {
            "schema_version": "verifier-candidate-block-binding/0.2",
            "candidate_sha256": verifier_candidate_binding_sha256(candidate),
            "result_block_sha256": verifier_result_block_binding_sha256(result_block),
            "evidence_anchor_sha256": verifier_evidence_anchor_binding_sha256(evidence_anchor),
            "evidence_block_sha256": verifier_evidence_block_binding_sha256(evidence_block),
        }
    )


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


class VerifierBakeoffModelSpec(StrictModel):
    model: str = Field(min_length=1)
    label: str = Field(min_length=1)


class FrozenVerifierCaseLocator(StrictModel):
    """One exact candidate/result-block/evidence binding in a sealed run."""

    case_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    observations_path: str = Field(min_length=1)
    result_blocks_path: str = Field(min_length=1)
    result_block_id: str = Field(min_length=1)
    candidate_binding_sha256: str = Field(pattern=HEX_64)
    result_block_sha256: str = Field(pattern=HEX_64)
    evidence_anchor_sha256: str = Field(pattern=HEX_64)
    evidence_block_sha256: str = Field(pattern=HEX_64)
    candidate_block_binding_sha256: str = Field(pattern=HEX_64)

    @field_validator("observations_path", "result_blocks_path")
    @classmethod
    def artifact_paths_are_canonical(cls, value: str) -> str:
        return _strict_relative_path(value, label="verifier artifact path")


class VerifierBakeoffConfig(StrictModel):
    """Strict, predeclared common-capability verifier experiment."""

    schema_version: Literal["verifier-bakeoff/0.2"] = "verifier-bakeoff/0.2"
    bakeoff_id: str = Field(min_length=1)
    sealed_run_path: str = Field(min_length=1)
    sealed_run_seal_sha256: str = Field(pattern=HEX_64)
    sealed_run_tree_sha256: str = Field(pattern=HEX_64)
    models: list[VerifierBakeoffModelSpec] = Field(min_length=2)
    cases: list[FrozenVerifierCaseLocator] = Field(min_length=1)
    max_tokens: int = Field(default=2_000, ge=1)
    temperature: None = None
    reasoning_effort: Literal["minimal"] = "minimal"
    seed: None = None
    require_parameters: Literal[True] = True
    fresh_repetitions: int = Field(default=2, ge=1)

    @field_validator("sealed_run_path")
    @classmethod
    def sealed_path_is_canonical(cls, value: str) -> str:
        return _strict_relative_path(value, label="sealed_run_path")

    @field_validator("fresh_repetitions", mode="before")
    @classmethod
    def repetitions_are_exact_integer(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("fresh_repetitions must be an integer")
        return value

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> VerifierBakeoffConfig:
        models = [item.model for item in self.models]
        cases = [item.case_id for item in self.cases]
        observations = [(item.observations_path, item.observation_id) for item in self.cases]
        if len(models) != len(set(models)):
            raise ValueError("verifier bake-off model IDs must be unique")
        if len(cases) != len(set(cases)):
            raise ValueError("verifier bake-off case IDs must be unique")
        if len(observations) != len(set(observations)):
            raise ValueError("verifier candidate locators must be unique")
        return self


class VerifierCandidateLabel(StrictModel):
    """Private single-reviewer candidate label, loaded only by the scorer."""

    schema_version: Literal["verifier-bakeoff-label/0.1"] = "verifier-bakeoff-label/0.1"
    case_id: str = Field(min_length=1)
    candidate_binding_sha256: str = Field(pattern=HEX_64)
    candidate_block_binding_sha256: str = Field(pattern=HEX_64)
    annotator: str = Field(min_length=1)
    label: Literal["good_candidate", "bad_candidate", "uncertain"]


def load_verifier_bakeoff_config(path: Path) -> VerifierBakeoffConfig:
    """Load one strict YAML or JSON-compatible verifier comparison."""

    return VerifierBakeoffConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class _SafeError(StrictModel):
    stage: str
    type: str
    code: str


class _CheckpointEntry(StrictModel):
    schema_version: Literal["verifier-bakeoff-entry/0.3"] = _ENTRY_VERSION
    slot_id: str = Field(min_length=1)
    slot_contract_sha256: str = Field(pattern=HEX_64)
    case_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    repetition_index: int = Field(ge=1)
    candidate_binding_sha256: str = Field(pattern=HEX_64)
    candidate_block_binding_sha256: str = Field(pattern=HEX_64)
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
    verification: CandidateVerification | None = None
    call: ProviderCall | None = None
    error: _SafeError | None = None
    entry_sha256: str = Field(pattern=HEX_64)

    @model_validator(mode="after")
    def exact_terminal_shape(self) -> _CheckpointEntry:
        if self.request.get("request_sha256") != _hash(
            {key: value for key, value in self.request.items() if key != "request_sha256"}
        ):
            raise ValueError("verifier request_sha256 is invalid")
        if self.status in {"success", "contract_failure"} and (
            self.verification is None or self.call is None or self.schema_status != "valid"
        ):
            raise ValueError("schema-valid verifier entry is incomplete")
        if self.status == "success" and self.contract_status != "satisfied":
            raise ValueError("successful verifier entry must satisfy the request contract")
        if self.status == "contract_failure" and self.contract_status != "failed":
            raise ValueError("verifier contract failure must retain failed contract status")
        if self.status in {"response_validation_failure", "local_validation_failure"} and (
            self.call is None or self.schema_status != "invalid"
        ):
            raise ValueError("verifier validation failure must retain its completed call")
        if self.status == "provider_failure" and (
            self.call is not None
            or self.verification is not None
            or self.schema_status != "not_observed"
            or self.contract_status != "not_observed"
        ):
            raise ValueError("verifier provider failure has an invalid shape")
        payload = self.model_dump(mode="json", exclude={"entry_sha256"})
        if self.entry_sha256 != _hash(payload):
            raise ValueError("verifier entry_sha256 is invalid")
        return self


class _Checkpoint(StrictModel):
    schema_version: Literal["verifier-bakeoff-checkpoint/0.3"] = _CHECKPOINT_VERSION
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
            raise ValueError("verifier checkpoint entry key mismatch")
        if self.status == "sealed":
            expected = _phase_seal(
                self.configuration_sha256, self.run_contract_sha256, self.entries
            )
            if self.provider_phase_seal_sha256 != expected:
                raise ValueError("verifier provider phase seal is invalid")
        elif self.provider_phase_seal_sha256 is not None:
            raise ValueError("in-progress verifier checkpoint cannot carry a seal")
        payload = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if self.checkpoint_sha256 != _hash(payload):
            raise ValueError("verifier checkpoint_sha256 is invalid")
        return self


@dataclass(frozen=True, slots=True)
class _PreparedCase:
    locator: FrozenVerifierCaseLocator
    candidate: CandidateObservation
    result_block: ResultBlock
    evidence_anchor: EvidenceAnchor
    evidence_block: FrozenEvidenceBlock
    observations_file_sha256: str
    result_blocks_file_sha256: str


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    config: VerifierBakeoffConfig
    configuration_sha256: str
    verified_seal: VerifiedRunSeal
    code: dict[str, Any]
    code_sha256: str
    stage_contract_sha256: str
    execution_selection: StageExecutionSelection | None
    models: tuple[VerifierBakeoffModelSpec, ...]
    run_contract_sha256: str
    cases: tuple[_PreparedCase, ...]


@dataclass(slots=True)
class _ConfiguredClient:
    client: Any
    last_call: ProviderCall | None = field(default=None, init=False)

    def structured_chat(self, **kwargs: Any) -> Any:
        forwarded = dict(kwargs)
        forwarded.update(VERIFIER_REQUEST_SETTINGS.as_dict())
        self.last_call = None
        try:
            response = self.client.structured_chat(**forwarded)
        except ProviderResponseValidationError as error:
            self.last_call = error.call
            raise
        self.last_call = response.call
        return response


def _safe_error(stage: str, error: Exception) -> _SafeError:
    return _SafeError(
        stage=stage,
        type=type(error).__name__,
        code=(
            error.code if isinstance(error, ProviderResponseValidationError) else "unexpected_error"
        ),
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
        raise ValueError("verifier candidate locator did not resolve exactly one observation")
    return matches[0]


def verifier_stage_contract_sha256(config: VerifierBakeoffConfig) -> str:
    """Bind the route-relevant contract shared by smoke and quality phases."""

    request_contract = verifier_request_contract()
    return _hash(
        {
            "schema_version": "verifier-bakeoff-stage-contract/0.2",
            "models": [item.model for item in config.models],
            "max_tokens": config.max_tokens,
            **VERIFIER_REQUEST_SETTINGS.as_dict(),
            "system_prompt_sha256": hashlib.sha256(
                VERIFIER_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "provider_schema_sha256": request_contract["schema"]["schema_sha256"],
            "request_contract_sha256": _hash(request_contract),
        }
    )


def _prepare_execution(
    config: VerifierBakeoffConfig,
    *,
    project_root: Path,
    code_root: Path | None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> _ExecutionContext:
    sealed_root = _project_path(project_root, config.sealed_run_path)
    verified = verify_run_seal(sealed_root)
    if verified.seal_sha256 != config.sealed_run_seal_sha256:
        raise ValueError("sealed run seal hash does not match verifier configuration")
    if verified.tree_sha256 != config.sealed_run_tree_sha256:
        raise ValueError("sealed run tree hash does not match verifier configuration")
    inventory = _inventory(verified)
    prepared: list[_PreparedCase] = []
    for locator in config.cases:
        if (
            locator.observations_path not in inventory
            or locator.result_blocks_path not in inventory
        ):
            raise ValueError("verifier artifact is absent from sealed inventory")
        observations_path = sealed_root / locator.observations_path
        blocks_path = sealed_root / locator.result_blocks_path
        observations_file_sha256 = str(inventory[locator.observations_path]["sha256"])
        blocks_file_sha256 = str(inventory[locator.result_blocks_path]["sha256"])
        if (
            sha256_file(observations_path) != observations_file_sha256
            or sha256_file(blocks_path) != blocks_file_sha256
        ):
            raise ValueError("sealed verifier artifact changed after verification")
        candidate = _load_candidate(observations_path, locator.observation_id)
        blocks = [ResultBlock.model_validate(item) for item in read_json(blocks_path)]
        if (
            sha256_file(observations_path) != observations_file_sha256
            or sha256_file(blocks_path) != blocks_file_sha256
        ):
            raise ValueError("sealed verifier artifact changed while loading")
        if candidate.paper_id != locator.paper_id:
            raise ValueError("verifier candidate paper_id does not match locator")
        support = bind_candidate_block(candidate, blocks)
        if support is None:
            raise ValueError("frozen candidate does not bind to a result block")
        result_block, anchor = support
        evidence_block = frozen_evidence_block(
            paper_id=locator.paper_id,
            block=result_block,
            anchor=anchor,
        )
        actual = {
            "result_block_id": result_block.block_id,
            "candidate_binding_sha256": verifier_candidate_binding_sha256(candidate),
            "result_block_sha256": verifier_result_block_binding_sha256(result_block),
            "evidence_anchor_sha256": verifier_evidence_anchor_binding_sha256(anchor),
            "evidence_block_sha256": verifier_evidence_block_binding_sha256(evidence_block),
            "candidate_block_binding_sha256": verifier_candidate_block_binding_sha256(
                candidate=candidate,
                result_block=result_block,
                evidence_anchor=anchor,
                evidence_block=evidence_block,
            ),
        }
        expected = {
            key: getattr(locator, key)
            for key in (
                "result_block_id",
                "candidate_binding_sha256",
                "result_block_sha256",
                "evidence_anchor_sha256",
                "evidence_block_sha256",
                "candidate_block_binding_sha256",
            )
        }
        if actual != expected:
            raise ValueError("frozen verifier candidate/result-block binding is stale")
        prepared.append(
            _PreparedCase(
                locator=locator,
                candidate=candidate,
                result_block=result_block,
                evidence_anchor=anchor,
                evidence_block=evidence_block,
                observations_file_sha256=observations_file_sha256,
                result_blocks_file_sha256=blocks_file_sha256,
            )
        )
    verified_after = verify_run_seal(sealed_root)
    if (
        verified_after.seal_sha256 != verified.seal_sha256
        or verified_after.tree_sha256 != verified.tree_sha256
        or verified_after.files != verified.files
    ):
        raise ValueError("sealed run changed while preparing verifier bake-off")
    configuration_sha256 = _hash(config.model_dump(mode="json", exclude_none=False))
    code = _code_state(code_root if code_root is not None else project_root)
    code_sha256 = _hash(code)
    stage_contract_sha256 = verifier_stage_contract_sha256(config)
    selected_model_ids = selected_models_for_execution(
        execution_selection,
        expected_stage="independent_verification",
        declared_models=[item.model for item in config.models],
        stage_contract_sha256=stage_contract_sha256,
        code_sha256=code_sha256,
        expectation=execution_selection_expectation,
    )
    selected = set(selected_model_ids)
    models = tuple(item for item in config.models if item.model in selected)
    if tuple(item.model for item in models) != selected_model_ids:
        raise ValueError("verifier execution selection does not preserve declared model order")
    run_contract_sha256 = _hash(
        {
            "schema_version": "verifier-bakeoff-run-contract/0.2",
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
                    "result_block_sha256": case.locator.result_block_sha256,
                    "evidence_anchor_sha256": case.locator.evidence_anchor_sha256,
                    "evidence_block_sha256": case.locator.evidence_block_sha256,
                    "candidate_block_binding_sha256": (case.locator.candidate_block_binding_sha256),
                    "observations_file_sha256": case.observations_file_sha256,
                    "result_blocks_file_sha256": case.result_blocks_file_sha256,
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
    model: VerifierBakeoffModelSpec,
    repetition_index: int,
) -> dict[str, Any]:
    verification_request = VerificationRequest(
        candidate=case.candidate,
        evidence_block=case.evidence_block,
    )
    user = verification_prompt(verification_request)
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    schema = verification_provider_json_schema()
    contract = structured_request_contract(
        schema_name=VERIFIER_SCHEMA_NAME,
        schema=schema,
        seed=VERIFIER_REQUEST_SETTINGS.seed,
        require_parameters=VERIFIER_REQUEST_SETTINGS.require_parameters,
        model=model.model,
        max_tokens=context.config.max_tokens,
    )
    request: dict[str, Any] = {
        "schema_version": "verifier-bakeoff-request/0.2",
        "case_id": case.locator.case_id,
        "model_requested": model.model,
        "provider_requested": "openrouter",
        "repetition_index": repetition_index,
        "candidate_binding_sha256": case.locator.candidate_binding_sha256,
        "result_block_sha256": case.locator.result_block_sha256,
        "evidence_anchor_sha256": case.locator.evidence_anchor_sha256,
        "evidence_block_sha256": case.locator.evidence_block_sha256,
        "candidate_block_binding_sha256": case.locator.candidate_block_binding_sha256,
        "prompt_sha256": hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "system_sha256": hashlib.sha256(VERIFIER_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
        "local_schema_sha256": _hash(schema),
        "provider_schema_sha256": contract["schema"]["schema_sha256"],
        "request_contract": contract,
        "request_contract_sha256": _hash(contract),
        "settings": {
            **VERIFIER_REQUEST_SETTINGS.as_dict(),
            "max_tokens": context.config.max_tokens,
        },
        "configuration_sha256": context.configuration_sha256,
        "code_sha256": context.code_sha256,
        "run_contract_sha256": context.run_contract_sha256,
    }
    request["request_sha256"] = _hash(request)
    return request


def _slot_id(request: dict[str, Any]) -> str:
    return (
        "verifier_slot_"
        + _hash(
            {
                "case_id": request["case_id"],
                "model_requested": request["model_requested"],
                "repetition_index": request["repetition_index"],
                "request_sha256": request["request_sha256"],
            }
        )[:24]
    )


def _assessment_payload(verification: CandidateVerification) -> dict[str, Any]:
    return verification.provider_assessment.model_dump(mode="json")


def _response_sha256(verification: CandidateVerification) -> str:
    return hashlib.sha256(
        json.dumps(_assessment_payload(verification), sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


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
        (
            "require_parameters",
            call.require_parameters,
            VERIFIER_REQUEST_SETTINGS.require_parameters,
        ),
    ):
        if actual != expected:
            failures.append(f"{name}_mismatch")
    if _hash(structured_request_contract_from_call(call)) != request["request_contract_sha256"]:
        failures.append("request_contract_mismatch")
    return ("failed", failures) if failures else ("satisfied", [])


def _verify_result_binding(
    verification: CandidateVerification,
    call: ProviderCall,
    case: _PreparedCase,
) -> None:
    expected = contextualize_verification(
        candidate=case.candidate,
        evidence_block=case.evidence_block,
        provider_assessment=verification.provider_assessment,
    )
    if any(
        (
            verification != expected,
            verification.evidence_block_sha256
            != verifier_evidence_block_sha256(case.evidence_block),
            call.response_sha256 != _response_sha256(verification),
        )
    ):
        raise ValueError("verifier result is not bound to its candidate/evidence request")


def _entry_payload(
    *,
    request: dict[str, Any],
    case: _PreparedCase,
    model: VerifierBakeoffModelSpec,
    repetition_index: int,
    status: str,
    schema_status: str,
    contract_status: str,
    contract_failure_codes: list[str] | None = None,
    verification: CandidateVerification | None = None,
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
                "candidate_block_binding_sha256": (case.locator.candidate_block_binding_sha256),
            }
        ),
        "case_id": case.locator.case_id,
        "model": model.model,
        "repetition_index": repetition_index,
        "candidate_binding_sha256": case.locator.candidate_binding_sha256,
        "candidate_block_binding_sha256": case.locator.candidate_block_binding_sha256,
        "request": request,
        "status": status,
        "schema_status": schema_status,
        "contract_status": contract_status,
        "contract_failure_codes": contract_failure_codes or [],
        "verification": verification,
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
    model: VerifierBakeoffModelSpec,
    repetition_index: int,
) -> bool:
    slot_id = _slot_id(request)
    expected_contract = _hash(
        {
            "slot_id": slot_id,
            "request_sha256": request["request_sha256"],
            "candidate_binding_sha256": case.locator.candidate_binding_sha256,
            "candidate_block_binding_sha256": case.locator.candidate_block_binding_sha256,
        }
    )
    if (
        entry.slot_id != slot_id
        or entry.slot_contract_sha256 != expected_contract
        or entry.case_id != case.locator.case_id
        or entry.model != model.model
        or entry.repetition_index != repetition_index
        or entry.candidate_binding_sha256 != case.locator.candidate_binding_sha256
        or entry.candidate_block_binding_sha256 != case.locator.candidate_block_binding_sha256
        or entry.request != request
    ):
        return False
    if entry.call is not None:
        status, failures = _contract_assessment(entry.call, request)
        if status != entry.contract_status or failures != entry.contract_failure_codes:
            return False
    elif entry.contract_status != "not_observed":
        return False
    if entry.verification is not None:
        try:
            assert entry.call is not None
            _verify_result_binding(entry.verification, entry.call, case)
        except (AssertionError, ValueError):
            return False
    return True


def _phase_seal(
    configuration_sha256: str,
    run_contract_sha256: str,
    entries: dict[str, _CheckpointEntry],
) -> str:
    return _hash(
        {
            "schema_version": "verifier-bakeoff-provider-seal/0.2",
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
    model: VerifierBakeoffModelSpec,
    repetition_index: int,
    client: _ConfiguredClient,
) -> _CheckpointEntry:
    request = _request_binding(context, case, model, repetition_index)
    candidate_before = verifier_candidate_binding_sha256(case.candidate)
    try:
        verification, call = verify_candidate(
            client=client,  # type: ignore[arg-type]
            model=model.model,
            candidate=case.candidate,
            evidence_block=case.evidence_block,
            max_tokens=context.config.max_tokens,
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
        if client.last_call is None:
            return _entry_payload(
                request=request,
                case=case,
                model=model,
                repetition_index=repetition_index,
                status="provider_failure",
                schema_status="not_observed",
                contract_status="not_observed",
                error=_safe_error("local_verifier_validation", error),
            )
        contract_status, failures = _contract_assessment(client.last_call, request)
        return _entry_payload(
            request=request,
            case=case,
            model=model,
            repetition_index=repetition_index,
            status="local_validation_failure",
            schema_status="invalid",
            contract_status=contract_status,
            contract_failure_codes=failures,
            call=client.last_call,
            error=_safe_error("local_verifier_validation", error),
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
    if verifier_candidate_binding_sha256(case.candidate) != candidate_before:
        raise RuntimeError("production verifier mutated its candidate input")
    contract_status, failures = _contract_assessment(call, request)
    try:
        _verify_result_binding(verification, call, case)
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
            error=_safe_error("verifier_result_binding", error),
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
        verification=verification,
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
        "candidate_block_binding_sha256": entry.candidate_block_binding_sha256,
        "request": entry.request,
        "status": entry.status,
        "schema_status": entry.schema_status,
        "contract_status": entry.contract_status,
        "contract_failure_codes": entry.contract_failure_codes,
        "entry_sha256": entry.entry_sha256,
        "resumed": resumed,
    }
    if entry.verification is not None:
        assessment = entry.verification.provider_assessment
        result["verification"] = {
            "assessment_sha256": _hash(_assessment_payload(entry.verification)),
            "provider_decision": assessment.decision.value,
            "effective_decision": entry.verification.effective_decision.value,
            "findings": {
                name: getattr(assessment, name).value
                for name in ("support", "role", "scope", "value", "metric")
            },
            "grounding": {
                "passed": entry.verification.grounding.passed,
                "failure_codes": entry.verification.grounding.failure_codes,
            },
            "justification_sha256": hashlib.sha256(
                assessment.justification.encode("utf-8")
            ).hexdigest(),
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
        "usage": usage,
    }


def _prediction(entry: _CheckpointEntry) -> str:
    if (
        entry.status == "success"
        and entry.contract_status == "satisfied"
        and entry.verification is not None
    ):
        return entry.verification.effective_decision.value
    return "unresolved"


def _stability(entries: list[_CheckpointEntry]) -> dict[str, Any]:
    by_case: dict[str, list[_CheckpointEntry]] = {}
    for entry in entries:
        by_case.setdefault(entry.case_id, []).append(entry)
    pairs = 0
    matching_decisions = 0
    assessment_pairs = 0
    exact_assessments = 0
    for case_entries in by_case.values():
        for left, right in combinations(case_entries, 2):
            pairs += 1
            matching_decisions += int(_prediction(left) == _prediction(right))
            if left.verification is not None and right.verification is not None:
                assessment_pairs += 1
                exact_assessments += int(
                    _hash(_assessment_payload(left.verification))
                    == _hash(_assessment_payload(right.verification))
                )
    return {
        "repetition_pair_denominator": pairs,
        "matching_decisions": matching_decisions,
        "decision_match_rate": _rate(matching_decisions, pairs),
        "assessment_pair_denominator": assessment_pairs,
        "exact_assessment_matches": exact_assessments,
        "exact_assessment_match_rate": _rate(exact_assessments, assessment_pairs),
    }


def run_verifier_bakeoff_provider_phase(
    config: VerifierBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    client: Any,
    code_root: Path | None = None,
    output_path: Path | None = None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> dict[str, Any]:
    """Run or exactly resume every predeclared label-blind verifier call."""

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
    configured_client = _ConfiguredClient(client=client)
    # A bounded stop before dispatch must still leave an exact resumable checkpoint.
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
                entry = _run_slot(
                    context,
                    case,
                    model,
                    repetition_index,
                    configured_client,
                )
                entries[slot_id] = entry
                executed.add(slot_id)
                _write_checkpoint(checkpoint_path, context, entries, sealed=False)
    expected_set = set(expected_slots)
    unexpected = set(entries) - expected_set
    if unexpected:
        checkpoint_rejections.append("checkpoint_unexpected_entries_removed")
        entries = {key: value for key, value in entries.items() if key in expected_set}
    if set(entries) != expected_set:
        raise RuntimeError("verifier checkpoint does not partition predeclared slots")
    sealed = _write_checkpoint(checkpoint_path, context, entries, sealed=True)
    ordered = [entries[key] for key in expected_slots]
    models = []
    executed_models = {item.model for item in context.models}
    for model in config.models:
        model_entries = [entry for entry in ordered if entry.model == model.model]
        if model.model in executed_models:
            decision = (
                execution_selection.decision_for(model.model)
                if execution_selection is not None
                else None
            )
            eligibility = decision.status if decision is not None else "not_assessed"
            models.append(
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
                    "aggregate": _aggregate(model_entries),
                    "stability": _stability(model_entries),
                }
            )
            continue
        if execution_selection is None:
            raise RuntimeError("verifier model was skipped without an execution selection")
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
                    "result_block_sha256": case.locator.result_block_sha256,
                    "evidence_anchor_sha256": case.locator.evidence_anchor_sha256,
                    "evidence_block_sha256": case.locator.evidence_block_sha256,
                    "candidate_block_binding_sha256": (case.locator.candidate_block_binding_sha256),
                    "observations_file_sha256": case.observations_file_sha256,
                    "result_blocks_file_sha256": case.result_blocks_file_sha256,
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
            **VERIFIER_REQUEST_SETTINGS.as_dict(),
            "max_tokens": config.max_tokens,
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
            "observation_ids_in_public_result": False,
            "result_block_ids_in_public_result": False,
            "row_ids_in_public_result": False,
            "raw_quotes_or_evidence_in_public_result": False,
            "exact_inputs_checkpoint_and_sealed_run_only": True,
        },
        "candidate_mutation_or_promotion": False,
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
        raise ValueError("verifier provider phase checkpoint is not sealed")
    if (
        checkpoint.configuration_sha256 != context.configuration_sha256
        or checkpoint.run_contract_sha256 != context.run_contract_sha256
    ):
        raise ValueError("verifier provider phase checkpoint is stale")
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
                    raise ValueError("verifier checkpoint contains an invalid slot")
                expected.append(entry)
    if len(expected) != len(checkpoint.entries):
        raise ValueError("verifier checkpoint contains unexpected slots")
    return checkpoint, expected


def derive_verifier_execution_selection(
    config: VerifierBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    experiment_id: str,
    public_manifest_sha256: str,
    wire_schema_gate: str = "0.99",
    code_root: Path | None = None,
) -> StageExecutionSelection:
    """Derive matched-quality eligibility from a sealed one-pass verifier smoke."""

    if config.fresh_repetitions != 1:
        raise ValueError("verifier contract smoke must declare exactly one repetition")
    context = _prepare_execution(config, project_root=project_root, code_root=code_root)
    checkpoint, entries = _load_sealed_checkpoint_for_scoring(checkpoint_path, context)
    if checkpoint.provider_phase_seal_sha256 is None:
        raise ValueError("verifier contract smoke is missing its provider phase seal")
    return derive_stage_execution_selection(
        experiment_id=experiment_id,
        stage="independent_verification",
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
) -> tuple[dict[str, VerifierCandidateLabel], str]:
    labels = [
        VerifierCandidateLabel.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_case = {label.case_id: label for label in labels}
    if len(by_case) != len(labels):
        raise ValueError("verifier label case IDs must be unique")
    cases = {case.locator.case_id: case for case in context.cases}
    if set(by_case) != set(cases):
        raise ValueError("verifier labels do not exactly cover configured cases")
    for case_id, case in cases.items():
        label = by_case[case_id]
        if (
            label.candidate_binding_sha256 != case.locator.candidate_binding_sha256
            or label.candidate_block_binding_sha256 != case.locator.candidate_block_binding_sha256
        ):
            raise ValueError("verifier label candidate/block binding mismatch")
    annotators = {label.annotator for label in labels}
    if len(annotators) != 1:
        raise ValueError("verifier bake-off requires exactly one annotator")
    annotator = next(iter(annotators))
    return by_case, hashlib.sha256(annotator.encode("utf-8")).hexdigest()


def _score_model(
    entries: list[_CheckpointEntry],
    labels: dict[str, VerifierCandidateLabel],
) -> dict[str, Any]:
    confusion = {
        expected: {predicted: 0 for predicted in _PREDICTIONS} for expected in _LABEL_CLASSES
    }
    unsafe_accepts = 0
    accepts = 0
    bad = 0
    bad_reject = 0
    bad_review = 0
    good = 0
    good_accept = 0
    review_routes = 0
    for entry in entries:
        expected = labels[entry.case_id].label
        predicted = _prediction(entry)
        confusion[expected][predicted] += 1
        accepts += int(predicted == "accept")
        unsafe_accepts += int(predicted == "accept" and expected != "good_candidate")
        bad += int(expected == "bad_candidate")
        bad_reject += int(expected == "bad_candidate" and predicted == "reject")
        bad_review += int(expected == "bad_candidate" and predicted == "review")
        good += int(expected == "good_candidate")
        good_accept += int(expected == "good_candidate" and predicted == "accept")
        review_routes += int(predicted in {"review", "unresolved"})
    total = len(entries)
    return {
        "unsafe_accepts": {
            "count": unsafe_accepts,
            "accept_predictions": accepts,
            "rate_of_all_predeclared_calls": _rate(unsafe_accepts, total),
            "rate_of_accept_predictions": _rate(unsafe_accepts, accepts),
        },
        "bad_candidate_reject_or_review_recall": _rate(bad_reject + bad_review, bad),
        "bad_candidate_reject_recall": _rate(bad_reject, bad),
        "bad_candidate_review_recall": _rate(bad_review, bad),
        "good_candidate_accept_recall": _rate(good_accept, good),
        "confusion": confusion,
        "exact_denominators": {
            "logical_calls_predeclared": total,
            "terminal_predictions": sum(sum(values.values()) for values in confusion.values()),
            "good_candidate_instances": good,
            "bad_candidate_instances": bad,
            "contract_eligible_assessments": sum(
                _prediction(entry) != "unresolved" for entry in entries
            ),
        },
        "review_burden": {
            "count": review_routes,
            "rate": _rate(review_routes, total),
            "includes": ["review", "unresolved"],
        },
        "wire_schema": _aggregate(entries)["wire_schema"],
        "contract": _aggregate(entries)["contract"],
        "stability": _stability(entries),
        "usage": _aggregate(entries)["usage"],
        "candidate_mutation_or_promotion_violations": 0,
    }


def _quality_measurement(entries: list[_CheckpointEntry]) -> dict[str, Any]:
    """Do not convert provider/contract failures into semantic predictions."""

    aggregate = _aggregate(entries)
    status_counts = aggregate["execution"]
    successful = status_counts["success"]
    complete = bool(entries) and successful == len(entries)
    return {
        "status": "measured" if complete else "unmeasured",
        "reason": (
            None if complete else "one_or_more_calls_lacked_a_valid_contract_complete_verification"
        ),
        "logical_calls_predeclared": len(entries),
        "valid_contract_complete_verifications": successful,
        "incomplete_or_failed_calls": len(entries) - successful,
        "terminal_status_counts": {
            key: status_counts[key]
            for key in (
                "success",
                "contract_failure",
                "response_validation_failure",
                "local_validation_failure",
                "provider_failure",
            )
        },
        "wire_schema": aggregate["wire_schema"],
        "contract": aggregate["contract"],
    }


def score_verifier_bakeoff_offline(
    config: VerifierBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    labels_path: Path,
    code_root: Path | None = None,
    output_path: Path | None = None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> dict[str, Any]:
    """Load private labels only after validating a complete provider-phase seal."""

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
    labels, annotator_sha256 = _load_labels(labels_path, context)
    models = []
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
                    "terminal_telemetry": _aggregate(model_entries),
                }
            )
            continue
        if execution_selection is None:
            raise RuntimeError("verifier model was skipped without an execution selection")
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
                    "valid_contract_complete_verifications": 0,
                    "incomplete_or_failed_calls": 0,
                },
                "quality": None,
                "terminal_telemetry": None,
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
        "annotator_sha256": annotator_sha256,
        "models": models,
        "privacy": {
            "labels_loaded_after_provider_phase_sealed": True,
            "observation_ids_in_score": False,
            "result_block_ids_in_score": False,
            "raw_quotes_or_evidence_in_score": False,
        },
        "candidate_mutation_or_promotion": False,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
