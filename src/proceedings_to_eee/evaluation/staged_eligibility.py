"""Hashed model-selection contracts derived from sealed contract smokes.

The contract-smoke phase answers only whether a configured route can satisfy a
stage's request and wire-schema contract.  It does not measure model quality.
This module turns the terminal entries of a *validated sealed checkpoint* into
an immutable execution selection for the later matched-quality phase.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes

HEX_64 = r"^[0-9a-f]{64}$"
StageName = Literal[
    "block_candidate_extraction",
    "origin_retrieval",
    "independent_verification",
    "row_disposition",
    "tuple_resolution",
]
EligibilityStatus = Literal["contract_eligible", "contract_ineligible"]
EligibilityReason = Literal[
    "terminal_partition_incomplete",
    "wire_schema_below_gate",
    "request_contract_unsatisfied",
    "telemetry_incomplete",
]


def _hash(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    return sha256_bytes(canonical_json_bytes(value))


class SmokeModelEligibility(StrictModel):
    """Exact contract-smoke accounting for one declared model."""

    model: str = Field(min_length=1)
    status: EligibilityStatus
    reason_codes: list[EligibilityReason]
    logical_calls_predeclared: int = Field(ge=1)
    terminal_entries: int = Field(ge=0)
    first_pass_schema_valid: int = Field(ge=0)
    first_pass_schema_invalid: int = Field(ge=0)
    first_pass_schema_not_observed: int = Field(ge=0)
    contract_satisfied: int = Field(ge=0)
    contract_failed: int = Field(ge=0)
    contract_not_observed: int = Field(ge=0)
    telemetry_complete: int = Field(ge=0)

    @model_validator(mode="after")
    def status_matches_reasons_and_denominators(self) -> SmokeModelEligibility:
        terminal = self.terminal_entries
        if (
            self.first_pass_schema_valid
            + self.first_pass_schema_invalid
            + self.first_pass_schema_not_observed
            != terminal
        ):
            raise ValueError("smoke schema counts do not partition terminal entries")
        if self.contract_satisfied + self.contract_failed + self.contract_not_observed != terminal:
            raise ValueError("smoke contract counts do not partition terminal entries")
        if self.telemetry_complete > terminal:
            raise ValueError("smoke telemetry count exceeds terminal entries")
        if self.status == "contract_eligible" and self.reason_codes:
            raise ValueError("eligible smoke decision cannot carry failure reasons")
        if self.status == "contract_ineligible" and not self.reason_codes:
            raise ValueError("ineligible smoke decision requires a failure reason")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("smoke eligibility reason codes must be unique")
        return self


class StageExecutionSelectionExpectation(StrictModel):
    """Caller-owned experiment/amendment binding required at selection consumption."""

    experiment_id: str = Field(min_length=1)
    public_manifest_sha256: str = Field(pattern=HEX_64)
    wire_schema_gate: str

    @field_validator("wire_schema_gate")
    @classmethod
    def gate_is_canonical_decimal(cls, value: str) -> str:
        return _canonical_gate(value)


def selection_expectation_from_public_manifest(
    *,
    experiment_id: str,
    expected_amendment_id: str,
    public_manifest_path: Path,
) -> StageExecutionSelectionExpectation:
    """Bind an invocation to the exact named amendment bytes and frozen schema gate."""

    if not expected_amendment_id:
        raise ValueError("selection amendment ID must be non-empty")

    if public_manifest_path.is_symlink() or not public_manifest_path.is_file():
        raise ValueError("selection public manifest must be one regular non-symlink file")
    try:
        raw = public_manifest_path.read_bytes()
        payload = json.loads(raw)
        amendment_id = payload["amendment_id"]
        qualification = payload["technical_qualification"]
        gate = qualification["wire_schema_gate"]
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError("selection public manifest lacks its technical schema gate") from error
    if not isinstance(gate, str):
        raise ValueError("selection public manifest technical schema gate is not a string")
    if amendment_id != expected_amendment_id:
        raise ValueError("selection public manifest amendment ID is unexpected")
    return StageExecutionSelectionExpectation(
        experiment_id=experiment_id,
        public_manifest_sha256=sha256_bytes(raw),
        wire_schema_gate=gate,
    )


def _canonical_gate(value: str) -> str:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("wire-schema gate must be a decimal string") from error
    if parsed < 0 or parsed > 1 or value != format(parsed, "f"):
        raise ValueError("wire-schema gate must be canonical and between zero and one")
    return value


def _expected_reason_codes(
    decision: SmokeModelEligibility,
    gate: Decimal,
) -> list[EligibilityReason]:
    logical = decision.logical_calls_predeclared
    reasons: list[EligibilityReason] = []
    if decision.terminal_entries != logical:
        reasons.append("terminal_partition_incomplete")
    if decision.terminal_entries != logical or Decimal(
        decision.first_pass_schema_valid
    ) < gate * Decimal(logical):
        reasons.append("wire_schema_below_gate")
    if (
        decision.contract_satisfied != logical
        or decision.contract_failed
        or decision.contract_not_observed
    ):
        reasons.append("request_contract_unsatisfied")
    if decision.telemetry_complete != logical:
        reasons.append("telemetry_incomplete")
    return reasons


class StageExecutionSelection(StrictModel):
    """Self-hashed, stage-specific matched-quality execution selection."""

    schema_version: Literal["stage-execution-selection/0.1"] = "stage-execution-selection/0.1"
    experiment_id: str = Field(min_length=1)
    stage: StageName
    public_manifest_sha256: str = Field(pattern=HEX_64)
    stage_contract_sha256: str = Field(pattern=HEX_64)
    smoke_bakeoff_id: str = Field(min_length=1)
    smoke_configuration_sha256: str = Field(pattern=HEX_64)
    smoke_run_contract_sha256: str = Field(pattern=HEX_64)
    smoke_code_sha256: str = Field(pattern=HEX_64)
    smoke_provider_phase_seal_sha256: str = Field(pattern=HEX_64)
    smoke_checkpoint_sha256: str = Field(pattern=HEX_64)
    wire_schema_gate: str
    declared_models: list[str] = Field(min_length=1)
    models: list[SmokeModelEligibility] = Field(min_length=1)
    selection_sha256: str = Field(pattern=HEX_64)

    @field_validator("wire_schema_gate")
    @classmethod
    def gate_is_canonical_decimal(cls, value: str) -> str:
        return _canonical_gate(value)

    @model_validator(mode="after")
    def exact_model_partition_and_hash(self) -> StageExecutionSelection:
        if len(self.declared_models) != len(set(self.declared_models)):
            raise ValueError("declared selection models must be unique")
        if [item.model for item in self.models] != self.declared_models:
            raise ValueError("smoke decisions must exactly preserve declared model order")
        gate = Decimal(self.wire_schema_gate)
        for decision in self.models:
            expected_reasons = _expected_reason_codes(decision, gate)
            expected_status = "contract_ineligible" if expected_reasons else "contract_eligible"
            if decision.reason_codes != expected_reasons or decision.status != expected_status:
                raise ValueError(
                    "smoke eligibility decision differs from its exact counts and gate"
                )
        payload = self.model_dump(mode="json", exclude={"selection_sha256"})
        if self.selection_sha256 != _hash(payload):
            raise ValueError("stage execution selection hash is invalid")
        return self

    @property
    def eligible_models(self) -> tuple[str, ...]:
        return tuple(item.model for item in self.models if item.status == "contract_eligible")

    def decision_for(self, model: str) -> SmokeModelEligibility:
        for decision in self.models:
            if decision.model == model:
                return decision
        raise KeyError(model)


def _entry_field(entry: Any, name: str) -> Any:
    if isinstance(entry, dict):
        return entry.get(name)
    return getattr(entry, name)


def _telemetry_is_complete(entry: Any, model: str) -> bool:
    call = _entry_field(entry, "call")
    if call is None:
        return False

    def field(name: str) -> Any:
        return call.get(name) if isinstance(call, dict) else getattr(call, name)

    returned_model = field("model_returned")
    returned_provider = field("provider_returned")
    return bool(
        field("model_requested") == model
        and returned_model == model
        and isinstance(returned_provider, str)
        and returned_provider.strip()
        and field("schema_strict") is True
        and field("require_parameters") is True
        and field("data_collection") == "deny"
        and field("zdr") is True
    )


def derive_stage_execution_selection(
    *,
    experiment_id: str,
    stage: StageName,
    public_manifest_sha256: str,
    stage_contract_sha256: str,
    smoke_bakeoff_id: str,
    smoke_configuration_sha256: str,
    smoke_run_contract_sha256: str,
    smoke_code_sha256: str,
    smoke_provider_phase_seal_sha256: str,
    smoke_checkpoint_sha256: str,
    wire_schema_gate: str,
    declared_models: Sequence[str],
    expected_slots_per_model: int,
    entries: Sequence[Any],
    expected_repetitions: int = 1,
) -> StageExecutionSelection:
    """Derive an exact selection from already validated sealed smoke entries.

    The caller remains responsible for validating the checkpoint's internal seal
    and exact slot set before passing entries here.  Aggregate report fields are
    deliberately ignored.
    """

    if expected_slots_per_model < 1:
        raise ValueError("smoke selection requires at least one slot per model")
    if expected_repetitions < 1:
        raise ValueError("smoke selection requires at least one repetition")
    declared = list(declared_models)
    if not declared or len(declared) != len(set(declared)):
        raise ValueError("declared smoke models must be non-empty and unique")
    if any(_entry_field(entry, "model") not in declared for entry in entries):
        raise ValueError("sealed smoke contains an undeclared model")
    if any(
        _entry_field(entry, "repetition_index") not in range(1, expected_repetitions + 1)
        for entry in entries
    ):
        raise ValueError("contract smoke entry has an unexpected repetition index")
    gate = Decimal(wire_schema_gate)
    decisions: list[SmokeModelEligibility] = []
    for model in declared:
        model_entries = [entry for entry in entries if _entry_field(entry, "model") == model]
        terminal = len(model_entries)
        valid = sum(_entry_field(entry, "schema_status") == "valid" for entry in model_entries)
        invalid = sum(_entry_field(entry, "schema_status") == "invalid" for entry in model_entries)
        schema_not_observed = terminal - valid - invalid
        satisfied = sum(
            _entry_field(entry, "contract_status") == "satisfied" for entry in model_entries
        )
        failed = sum(_entry_field(entry, "contract_status") == "failed" for entry in model_entries)
        contract_not_observed = terminal - satisfied - failed
        telemetry_complete = sum(_telemetry_is_complete(entry, model) for entry in model_entries)
        reasons: list[EligibilityReason] = []
        if terminal != expected_slots_per_model:
            reasons.append("terminal_partition_incomplete")
        if terminal != expected_slots_per_model or Decimal(valid) < gate * Decimal(
            expected_slots_per_model
        ):
            reasons.append("wire_schema_below_gate")
        if satisfied != expected_slots_per_model or failed or contract_not_observed:
            reasons.append("request_contract_unsatisfied")
        if telemetry_complete != expected_slots_per_model:
            reasons.append("telemetry_incomplete")
        decisions.append(
            SmokeModelEligibility(
                model=model,
                status="contract_eligible" if not reasons else "contract_ineligible",
                reason_codes=reasons,
                logical_calls_predeclared=expected_slots_per_model,
                terminal_entries=terminal,
                first_pass_schema_valid=valid,
                first_pass_schema_invalid=invalid,
                first_pass_schema_not_observed=schema_not_observed,
                contract_satisfied=satisfied,
                contract_failed=failed,
                contract_not_observed=contract_not_observed,
                telemetry_complete=telemetry_complete,
            )
        )
    payload = {
        "schema_version": "stage-execution-selection/0.1",
        "experiment_id": experiment_id,
        "stage": stage,
        "public_manifest_sha256": public_manifest_sha256,
        "stage_contract_sha256": stage_contract_sha256,
        "smoke_bakeoff_id": smoke_bakeoff_id,
        "smoke_configuration_sha256": smoke_configuration_sha256,
        "smoke_run_contract_sha256": smoke_run_contract_sha256,
        "smoke_code_sha256": smoke_code_sha256,
        "smoke_provider_phase_seal_sha256": smoke_provider_phase_seal_sha256,
        "smoke_checkpoint_sha256": smoke_checkpoint_sha256,
        "wire_schema_gate": wire_schema_gate,
        "declared_models": declared,
        "models": [item.model_dump(mode="json") for item in decisions],
    }
    return StageExecutionSelection.model_validate(payload | {"selection_sha256": _hash(payload)})


def selected_models_for_execution(
    selection: StageExecutionSelection | None,
    *,
    expected_stage: StageName,
    declared_models: Sequence[str],
    stage_contract_sha256: str,
    code_sha256: str,
    expectation: StageExecutionSelectionExpectation | None,
) -> tuple[str, ...]:
    """Validate a selection against the current stage before any provider call."""

    declared = tuple(declared_models)
    if selection is None:
        if expectation is not None:
            raise ValueError("selection expectation was supplied without an execution selection")
        return declared
    if expectation is None:
        raise ValueError("stage execution selection lacks an expected amendment binding")
    selection = StageExecutionSelection.model_validate(
        selection.model_dump(mode="json", exclude_none=False)
    )
    expectation = StageExecutionSelectionExpectation.model_validate(
        expectation.model_dump(mode="json", exclude_none=False)
    )
    if selection.experiment_id != expectation.experiment_id:
        raise ValueError("stage execution selection targets another experiment")
    if selection.public_manifest_sha256 != expectation.public_manifest_sha256:
        raise ValueError("stage execution selection targets another public manifest")
    if selection.wire_schema_gate != expectation.wire_schema_gate:
        raise ValueError("stage execution selection uses another wire-schema gate")
    if selection.stage != expected_stage:
        raise ValueError("stage execution selection targets another stage")
    if tuple(selection.declared_models) != declared:
        raise ValueError("stage execution selection has a stale model declaration")
    if selection.stage_contract_sha256 != stage_contract_sha256:
        raise ValueError("stage execution selection has a stale stage contract")
    if selection.smoke_code_sha256 != code_sha256:
        raise ValueError("stage execution selection was derived under different code")
    return selection.eligible_models
