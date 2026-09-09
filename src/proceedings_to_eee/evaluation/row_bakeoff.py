"""Private, resumable model bake-offs for dense-table row dispositions.

The provider phase is deliberately label-blind.  It reads only a verified sealed
layout/row plan and a predeclared set of private row IDs, then checkpoints every
terminal provider attempt before another call is made.  Exact row text, candidate
evidence, and row IDs remain in that caller-designated private checkpoint.  A
separate offline scorer joins a strict single-reviewer label file only after the
provider phase has been sealed.

Provider output remains proposal-only: this module never exports or promotes a row
candidate.  Production row parsing and terminal-ledger validation remain the source
of truth for every live and resumed outcome.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import Field, JsonValue, field_validator, model_validator

from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.evaluation.artifact_safety import assert_artifact_paths_safe
from proceedings_to_eee.evaluation.staged_eligibility import (
    StageExecutionSelection,
    StageExecutionSelectionExpectation,
    derive_stage_execution_selection,
    selected_models_for_execution,
)
from proceedings_to_eee.extraction.llm import (
    ROW_EXTRACTOR_SCHEMA_NAME,
    RowEnumerationOutcome,
    extract_row_batch_candidates,
)
from proceedings_to_eee.extraction.llm_schema import WireRowExtraction, row_provider_json_schema
from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.extraction.prompt import (
    ROW_ENUMERATION_SYSTEM_PROMPT,
    row_batch_prompt,
    row_prompt_hash,
)
from proceedings_to_eee.extraction.row_enumeration import (
    RowAttemptTelemetry,
    RowBatch,
    RowDispositionRecord,
    RowEnumerationPlan,
    RowTerminalCounts,
    RowTerminalRecord,
    RowTerminalState,
    make_row_batch,
    recovery_batches,
)
from proceedings_to_eee.extraction.row_validation import validate_batch_outcome_against
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
    StructuredResponse,
    public_provider_call,
    structured_request_contract,
    structured_request_contract_from_call,
)
from proceedings_to_eee.run_seal import VerifiedRunSeal, verify_run_seal

HEX_64 = r"^[0-9a-f]{64}$"
_CHECKPOINT_VERSION = "row-bakeoff-checkpoint/0.3"
_ATTEMPT_VERSION = "row-bakeoff-attempt/0.3"
_REPETITION_VERSION = "row-bakeoff-repetition/0.1"
_RESULT_VERSION = "row-bakeoff-provider-result/0.2"
_SCORE_VERSION = "row-bakeoff-score/0.2"
_LABEL_CLASSES = ("result", "not_result", "uncertain", "unsupported")
_PREDICTION_CLASSES = (*_LABEL_CLASSES[:3], "unresolved", "unsupported")


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


def row_layout_binding_sha256(layout: PdfLayout) -> str:
    """Return the complete logical binding used for a frozen layout."""

    return _hash(layout)


def row_plan_binding_sha256(plan: RowEnumerationPlan) -> str:
    """Return the complete logical binding used for a frozen row plan."""

    return _hash(plan)


def row_source_binding_sha256(layout: PdfLayout) -> str:
    """Bind the source identity and every exact page-text digest."""

    return _hash(
        {
            "source_id": layout.source_id,
            "page_count": layout.page_count,
            "pages": [
                {
                    "page": page.page,
                    "text_sha256": page.text_sha256,
                    "character_count": page.character_count,
                }
                for page in layout.pages
            ],
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


class RowBakeoffModelSpec(StrictModel):
    model: str = Field(min_length=1)
    label: str = Field(min_length=1)


class FrozenRowCaseLocator(StrictModel):
    """One layout, row plan, and balanced private selection in a sealed run."""

    case_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    paper_title: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    layout_path: str = Field(min_length=1)
    row_plan_path: str = Field(min_length=1)
    source_binding_sha256: str = Field(pattern=HEX_64)
    layout_binding_sha256: str = Field(pattern=HEX_64)
    row_plan_sha256: str = Field(pattern=HEX_64)
    selected_row_ids: list[str] = Field(min_length=1)
    balanced_rows_per_class: int | None = Field(default=None, ge=1)

    @field_validator("layout_path", "row_plan_path")
    @classmethod
    def artifact_paths_are_canonical(cls, value: str) -> str:
        return _strict_relative_path(value, label="row artifact path")

    @field_validator("balanced_rows_per_class", mode="before")
    @classmethod
    def balance_is_exact_integer(cls, value: Any) -> Any:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("balanced_rows_per_class must be an integer")
        return value

    @model_validator(mode="after")
    def selection_is_unique_and_balanced(self) -> FrozenRowCaseLocator:
        if len(self.selected_row_ids) != len(set(self.selected_row_ids)):
            raise ValueError("selected row IDs must be unique")
        if self.balanced_rows_per_class is not None and len(self.selected_row_ids) != (
            len(_LABEL_CLASSES) * self.balanced_rows_per_class
        ):
            raise ValueError("selected rows do not match the predeclared balanced class size")
        return self


class RowBakeoffConfig(StrictModel):
    """Strict, predeclared common-capability experiment contract."""

    schema_version: Literal[
        "row-bakeoff/0.1",
        "row-bakeoff/0.2",
        "row-bakeoff/0.3",
    ] = "row-bakeoff/0.1"
    frame_mode: Literal["contract_smoke", "balanced_quality"] = "balanced_quality"
    bakeoff_id: str = Field(min_length=1)
    sealed_run_path: str = Field(min_length=1)
    sealed_run_seal_sha256: str = Field(pattern=HEX_64)
    sealed_run_tree_sha256: str = Field(pattern=HEX_64)
    models: list[RowBakeoffModelSpec] = Field(min_length=2)
    cases: list[FrozenRowCaseLocator] = Field(min_length=1)
    max_tokens: int = Field(default=4_000, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_effort: str | None = Field(default="minimal", min_length=1)
    seed: int | None = None
    require_parameters: Literal[True] = True
    fresh_repetitions: int = Field(default=2, ge=1)

    @model_validator(mode="before")
    @classmethod
    def repair04_defaults_use_the_production_contract(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("schema_version") == "row-bakeoff/0.3":
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
    def identifiers_are_unique(self) -> RowBakeoffConfig:
        if self.schema_version == "row-bakeoff/0.3" and (
            self.max_tokens != 16_000
            or self.temperature is not None
            or self.reasoning_effort != "minimal"
            or self.seed is not None
            or not self.require_parameters
        ):
            raise ValueError("row-bakeoff/0.3 requires the production request contract")
        model_ids = [item.model for item in self.models]
        case_ids = [item.case_id for item in self.cases]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("row bake-off model IDs must be unique")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("row bake-off case IDs must be unique")
        balances = [item.balanced_rows_per_class for item in self.cases]
        if self.frame_mode == "contract_smoke":
            if self.schema_version not in {"row-bakeoff/0.2", "row-bakeoff/0.3"}:
                raise ValueError("contract-smoke row frames require a current row contract")
            if self.fresh_repetitions != 1:
                raise ValueError("contract-smoke row frames require exactly one repetition")
            if any(value is not None for value in balances):
                raise ValueError("contract-smoke row frames cannot claim class balance")
        elif any(value is None for value in balances):
            raise ValueError("balanced-quality row frames require a class-balance declaration")
        return self


class RowDispositionLabel(StrictModel):
    """Private single-reviewer label; never accepted by the provider phase."""

    schema_version: Literal["row-bakeoff-label/0.1"] = "row-bakeoff-label/0.1"
    case_id: str = Field(min_length=1)
    row_id: str = Field(min_length=1)
    row_plan_sha256: str = Field(pattern=HEX_64)
    annotator: str = Field(min_length=1)
    label: Literal["result", "not_result", "uncertain", "unsupported"]


def load_row_bakeoff_config(path: Path) -> RowBakeoffConfig:
    """Load one strict YAML or JSON-compatible row bake-off definition."""

    return RowBakeoffConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class _SafeError(StrictModel):
    stage: str
    type: str
    code: str


def _provider_payload_sha256(payload: dict[str, JsonValue]) -> str:
    """Mirror the production OpenRouter response fingerprint exactly."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class _NormalizedRowWireProposal(StrictModel):
    """Typed structured content only; never an HTTP envelope or raw JSON string."""

    structured_content: dict[str, JsonValue]

    @model_validator(mode="after")
    def content_is_the_exact_row_wire_type(self) -> _NormalizedRowWireProposal:
        try:
            WireRowExtraction.model_validate(self.structured_content)
        except Exception as error:
            raise ValueError("stored row wire proposal is invalid") from error
        return self


class _AttemptEntry(StrictModel):
    schema_version: Literal["row-bakeoff-attempt/0.3"] = _ATTEMPT_VERSION
    attempt_id: str = Field(min_length=1)
    attempt_contract_sha256: str = Field(pattern=HEX_64)
    case_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    repetition_index: int = Field(ge=1)
    base_batch_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    depth: int = Field(ge=0, le=1)
    row_ids: list[str] = Field(min_length=1)
    request: dict[str, Any]
    status: Literal[
        "success",
        "partial_invalid",
        "contract_failure",
        "response_validation_failure",
        "provider_failure",
    ]
    schema_status: Literal["valid", "invalid", "not_observed"]
    contract_status: Literal["satisfied", "failed", "not_observed"]
    contract_failure_codes: list[str] = Field(default_factory=list)
    records: dict[str, RowDispositionRecord] = Field(default_factory=dict)
    unresolved_row_ids: list[str]
    unknown_row_ids: list[str] = Field(default_factory=list)
    invalid_row_reasons: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    call: ProviderCall | None = None
    normalized_wire_proposal: _NormalizedRowWireProposal | None = None
    error: _SafeError | None = None
    entry_sha256: str = Field(pattern=HEX_64)

    @model_validator(mode="after")
    def exact_attempt_accounting(self) -> _AttemptEntry:
        if len(self.row_ids) != len(set(self.row_ids)):
            raise ValueError("row attempt repeats input IDs")
        if any(key != value.row_id for key, value in self.records.items()):
            raise ValueError("row attempt record key mismatch")
        resolved = set(self.records)
        unresolved = set(self.unresolved_row_ids)
        owned = set(self.row_ids)
        if (
            resolved & unresolved
            or resolved | unresolved != owned
            or len(self.unresolved_row_ids) != len(unresolved)
        ):
            raise ValueError("row attempt records and unresolved IDs must exactly partition input")
        if set(self.invalid_row_reasons) - owned:
            raise ValueError("row attempt invalid reasons contain non-input IDs")
        if self.request.get("request_sha256") != _hash(
            {key: value for key, value in self.request.items() if key != "request_sha256"}
        ):
            raise ValueError("row attempt request_sha256 is invalid")
        if self.status in {"success", "partial_invalid", "contract_failure"} and (
            self.call is None or self.schema_status != "valid"
        ):
            raise ValueError("schema-valid row attempt must retain its provider call")
        if self.schema_status == "valid":
            if self.call is None or self.normalized_wire_proposal is None:
                raise ValueError("schema-valid row attempt must retain its wire response")
            if (
                _provider_payload_sha256(self.normalized_wire_proposal.structured_content)
                != self.call.response_sha256
            ):
                raise ValueError("row attempt wire response does not match its provider call")
        elif self.normalized_wire_proposal is not None:
            raise ValueError("schema-invalid row attempt cannot retain a usable wire response")
        if self.status == "success" and self.unresolved_row_ids:
            raise ValueError("successful row attempt cannot retain unresolved rows")
        if self.status == "partial_invalid" and not self.unresolved_row_ids:
            raise ValueError("partial row attempt requires unresolved rows")
        if self.status == "contract_failure" and self.contract_status != "failed":
            raise ValueError("contract failure must retain failed contract status")
        if self.status == "response_validation_failure" and (
            self.call is None or self.schema_status != "invalid"
        ):
            raise ValueError("response validation failure must retain its completed call")
        if self.status == "provider_failure" and (
            self.call is not None or self.schema_status != "not_observed"
        ):
            raise ValueError("provider failure cannot claim a completed response")
        if self.status in {"success", "partial_invalid"} and self.contract_status != "satisfied":
            raise ValueError("eligible row attempt must satisfy its request contract")
        payload = self.model_dump(mode="json", exclude={"entry_sha256"})
        if self.entry_sha256 != _hash(payload):
            raise ValueError("row attempt entry_sha256 is invalid")
        return self


class _RepetitionResult(StrictModel):
    schema_version: Literal["row-bakeoff-repetition/0.1"] = _REPETITION_VERSION
    repetition_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    repetition_index: int = Field(ge=1)
    selected_rows_sha256: str = Field(pattern=HEX_64)
    row_plan_sha256: str = Field(pattern=HEX_64)
    attempt_ids: list[str]
    terminal_by_row: dict[str, RowTerminalRecord]
    counts: RowTerminalCounts
    repetition_sha256: str = Field(pattern=HEX_64)

    @model_validator(mode="after")
    def exact_terminal_partition(self) -> _RepetitionResult:
        if any(key != value.row_id for key, value in self.terminal_by_row.items()):
            raise ValueError("repetition terminal key mismatch")
        actual = {state: 0 for state in RowTerminalState}
        for terminal in self.terminal_by_row.values():
            actual[terminal.state] += 1
        expected = RowTerminalCounts(
            planned=len(self.terminal_by_row),
            result=actual[RowTerminalState.RESULT],
            not_result=actual[RowTerminalState.NOT_RESULT],
            uncertain=actual[RowTerminalState.UNCERTAIN],
            unresolved=actual[RowTerminalState.UNRESOLVED],
            unsupported=actual[RowTerminalState.UNSUPPORTED],
        )
        if self.counts != expected:
            raise ValueError("repetition terminal counts are invalid")
        payload = self.model_dump(mode="json", exclude={"repetition_sha256"})
        if self.repetition_sha256 != _hash(payload):
            raise ValueError("repetition_sha256 is invalid")
        return self


class _Checkpoint(StrictModel):
    schema_version: Literal["row-bakeoff-checkpoint/0.3"] = _CHECKPOINT_VERSION
    bakeoff_id: str = Field(min_length=1)
    configuration_sha256: str = Field(pattern=HEX_64)
    run_contract_sha256: str = Field(pattern=HEX_64)
    status: Literal["in_progress", "sealed"]
    attempts: dict[str, _AttemptEntry]
    repetitions: dict[str, _RepetitionResult]
    provider_phase_seal_sha256: str | None = Field(default=None, pattern=HEX_64)
    checkpoint_sha256: str = Field(pattern=HEX_64)

    @model_validator(mode="after")
    def hashes_and_seal_are_valid(self) -> _Checkpoint:
        if any(key != entry.attempt_id for key, entry in self.attempts.items()):
            raise ValueError("checkpoint attempt key mismatch")
        if any(key != item.repetition_id for key, item in self.repetitions.items()):
            raise ValueError("checkpoint repetition key mismatch")
        if self.status == "sealed":
            expected = _phase_seal(
                self.configuration_sha256,
                self.run_contract_sha256,
                self.attempts,
                self.repetitions,
            )
            if self.provider_phase_seal_sha256 != expected:
                raise ValueError("row provider phase seal is invalid")
        elif self.provider_phase_seal_sha256 is not None:
            raise ValueError("in-progress row checkpoint cannot carry a seal")
        payload = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if self.checkpoint_sha256 != _hash(payload):
            raise ValueError("row checkpoint hash is invalid")
        return self


@dataclass(frozen=True, slots=True)
class _PreparedCase:
    locator: FrozenRowCaseLocator
    layout: PdfLayout
    plan: RowEnumerationPlan
    selected_batches: tuple[RowBatch, ...]
    selected_unbatchable: tuple[Any, ...]
    layout_file_sha256: str
    row_plan_file_sha256: str
    selected_rows_sha256: str


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    config: RowBakeoffConfig
    configuration_sha256: str
    verified_seal: VerifiedRunSeal
    code: dict[str, Any]
    code_sha256: str
    stage_contract_sha256: str
    execution_selection: StageExecutionSelection | None
    models: tuple[RowBakeoffModelSpec, ...]
    run_contract_sha256: str
    cases: tuple[_PreparedCase, ...]


@dataclass(frozen=True, slots=True)
class _ConfiguredClient:
    client: Any
    temperature: float | None
    reasoning_effort: str | None
    seed: int | None
    captured_structured_content: dict[str, dict[str, JsonValue]] = dataclass_field(
        default_factory=dict
    )

    def structured_chat(self, **kwargs: Any) -> Any:
        forwarded = dict(kwargs)
        forwarded["temperature"] = self.temperature
        forwarded["reasoning_effort"] = self.reasoning_effort
        forwarded["seed"] = self.seed
        forwarded["require_parameters"] = True
        response = self.client.structured_chat(**forwarded)
        payload = json.loads(json.dumps(response.payload, ensure_ascii=False))
        response_sha256 = response.call.response_sha256
        previous = self.captured_structured_content.get(response_sha256)
        if previous is not None and previous != payload:
            raise ValueError("one provider response hash maps to conflicting row payloads")
        self.captured_structured_content[response_sha256] = payload
        return response

    def normalized_wire_proposal_for(self, call: ProviderCall) -> _NormalizedRowWireProposal:
        try:
            content = self.captured_structured_content[call.response_sha256]
        except KeyError as error:
            raise ValueError("completed row call has no captured wire response") from error
        return _NormalizedRowWireProposal(structured_content=content)


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


def row_stage_contract_sha256(config: RowBakeoffConfig) -> str:
    """Bind the provider-route contract shared by row smoke and quality frames."""

    schema = row_provider_json_schema()
    request_contract = structured_request_contract(
        schema_name=ROW_EXTRACTOR_SCHEMA_NAME,
        schema=schema,
        seed=config.seed,
        require_parameters=True,
    )
    return _hash(
        {
            "schema_version": "row-bakeoff-stage-contract/0.1",
            "models": [item.model for item in config.models],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "seed": config.seed,
            "require_parameters": True,
            "row_prompt_template_sha256": row_prompt_hash(),
            "system_sha256": hashlib.sha256(
                ROW_ENUMERATION_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "local_schema_sha256": _hash(schema),
            "provider_schema_sha256": request_contract["schema"]["schema_sha256"],
            "request_contract_sha256": _hash(request_contract),
        }
    )


def _prepare_execution(
    config: RowBakeoffConfig,
    *,
    project_root: Path,
    code_root: Path | None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> _ExecutionContext:
    validated_selection = (
        StageExecutionSelection.model_validate(
            execution_selection.model_dump(mode="json", exclude_none=False)
        )
        if execution_selection is not None
        else None
    )
    configuration_sha256 = _hash(config.model_dump(mode="json", exclude_none=False))
    code = _code_state(code_root if code_root is not None else project_root)
    code_sha256 = _hash(code)
    stage_contract_sha256 = row_stage_contract_sha256(config)
    selected_model_ids = selected_models_for_execution(
        validated_selection,
        expected_stage="row_disposition",
        declared_models=[item.model for item in config.models],
        stage_contract_sha256=stage_contract_sha256,
        code_sha256=code_sha256,
        expectation=execution_selection_expectation,
    )
    selected = set(selected_model_ids)
    models = tuple(item for item in config.models if item.model in selected)
    if tuple(item.model for item in models) != selected_model_ids:
        raise ValueError("row execution selection does not preserve declared model order")
    sealed_root = _project_path(project_root, config.sealed_run_path)
    verified = verify_run_seal(sealed_root)
    if verified.seal_sha256 != config.sealed_run_seal_sha256:
        raise ValueError("sealed run seal hash does not match row bake-off configuration")
    if verified.tree_sha256 != config.sealed_run_tree_sha256:
        raise ValueError("sealed run tree hash does not match row bake-off configuration")
    inventory = _inventory(verified)
    prepared: list[_PreparedCase] = []
    for locator in config.cases:
        if locator.layout_path not in inventory or locator.row_plan_path not in inventory:
            raise ValueError("row bake-off artifact is absent from sealed inventory")
        layout_path = sealed_root / locator.layout_path
        plan_path = sealed_root / locator.row_plan_path
        layout_file_sha256 = str(inventory[locator.layout_path]["sha256"])
        plan_file_sha256 = str(inventory[locator.row_plan_path]["sha256"])
        if (
            sha256_file(layout_path) != layout_file_sha256
            or sha256_file(plan_path) != plan_file_sha256
        ):
            raise ValueError("sealed row artifact changed after verification")
        layout = PdfLayout.model_validate(read_json(layout_path))
        plan = RowEnumerationPlan.model_validate(read_json(plan_path))
        if (
            sha256_file(layout_path) != layout_file_sha256
            or sha256_file(plan_path) != plan_file_sha256
        ):
            raise ValueError("sealed row artifact changed while loading")
        if layout.source_id != locator.source_id:
            raise ValueError("row layout source_id does not match locator")
        if any(row.source_id != locator.source_id for row in plan.rows):
            raise ValueError("row plan contains another source")
        if row_source_binding_sha256(layout) != locator.source_binding_sha256:
            raise ValueError("row source binding does not match locator")
        if row_layout_binding_sha256(layout) != locator.layout_binding_sha256:
            raise ValueError("row layout binding does not match locator")
        if row_plan_binding_sha256(plan) != locator.row_plan_sha256:
            raise ValueError("row plan binding does not match locator")
        planned = {row.row_id: row for row in plan.rows}
        if not set(locator.selected_row_ids).issubset(planned):
            raise ValueError("selected row ID is absent from frozen row plan")
        selected = set(locator.selected_row_ids)
        selected_batches = tuple(
            make_row_batch(rows)
            for batch in plan.batches
            if (rows := [row for row in batch.rows if row.row_id in selected])
        )
        selected_unbatchable = tuple(row for row in plan.unbatchable_rows if row.row_id in selected)
        owned = {row.row_id for batch in selected_batches for row in batch.rows} | {
            row.row_id for row in selected_unbatchable
        }
        if owned != selected:
            raise ValueError("selected rows do not have exactly one frozen plan owner")
        if (
            locator.balanced_rows_per_class is not None
            and len(selected_unbatchable) != locator.balanced_rows_per_class
        ):
            raise ValueError(
                "predeclared unsupported selection must equal one balanced label class"
            )
        if config.frame_mode == "contract_smoke" and not selected_batches:
            raise ValueError("contract-smoke row frames require a nonempty provider batch")
        prepared.append(
            _PreparedCase(
                locator=locator,
                layout=layout,
                plan=plan,
                selected_batches=selected_batches,
                selected_unbatchable=selected_unbatchable,
                layout_file_sha256=layout_file_sha256,
                row_plan_file_sha256=plan_file_sha256,
                selected_rows_sha256=_hash(locator.selected_row_ids),
            )
        )
    verified_after = verify_run_seal(sealed_root)
    if (
        verified_after.seal_sha256 != verified.seal_sha256
        or verified_after.tree_sha256 != verified.tree_sha256
        or verified_after.files != verified.files
    ):
        raise ValueError("sealed run changed while preparing row bake-off")
    run_contract_sha256 = _hash(
        {
            "schema_version": "row-bakeoff-run-contract/0.1",
            "configuration_sha256": configuration_sha256,
            "sealed_run_seal_sha256": verified.seal_sha256,
            "sealed_run_tree_sha256": verified.tree_sha256,
            "code_sha256": code_sha256,
            "stage_contract_sha256": stage_contract_sha256,
            "execution_selection_sha256": (
                validated_selection.selection_sha256 if validated_selection is not None else None
            ),
            "executed_models": list(selected_model_ids),
            "case_bindings": [
                {
                    "case_id": case.locator.case_id,
                    "source_binding_sha256": case.locator.source_binding_sha256,
                    "layout_binding_sha256": case.locator.layout_binding_sha256,
                    "row_plan_sha256": case.locator.row_plan_sha256,
                    "layout_file_sha256": case.layout_file_sha256,
                    "row_plan_file_sha256": case.row_plan_file_sha256,
                    "selected_rows_sha256": case.selected_rows_sha256,
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
        execution_selection=validated_selection,
        models=models,
        run_contract_sha256=run_contract_sha256,
        cases=tuple(prepared),
    )


def _request_binding(
    context: _ExecutionContext,
    case: _PreparedCase,
    model: RowBakeoffModelSpec,
    repetition_index: int,
    base_batch: RowBatch,
    batch: RowBatch,
    depth: int,
) -> dict[str, Any]:
    user = row_batch_prompt(
        paper_title=case.locator.paper_title,
        paper_id=case.locator.paper_id,
        batch=batch,
    )
    messages = [
        {"role": "system", "content": ROW_ENUMERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    schema = row_provider_json_schema()
    contract = structured_request_contract(
        schema_name=ROW_EXTRACTOR_SCHEMA_NAME,
        schema=schema,
        seed=context.config.seed,
        require_parameters=True,
        model=model.model,
        max_tokens=context.config.max_tokens,
    )
    request: dict[str, Any] = {
        "schema_version": "row-bakeoff-request/0.1",
        "case_id": case.locator.case_id,
        "model_requested": model.model,
        "provider_requested": "openrouter",
        "repetition_index": repetition_index,
        "base_batch_id": base_batch.batch_id,
        "batch_id": batch.batch_id,
        "depth": depth,
        "batch_binding_sha256": _hash(batch),
        "selected_rows_sha256": case.selected_rows_sha256,
        "source_binding_sha256": case.locator.source_binding_sha256,
        "layout_binding_sha256": case.locator.layout_binding_sha256,
        "row_plan_sha256": case.locator.row_plan_sha256,
        "row_prompt_template_sha256": row_prompt_hash(),
        "prompt_sha256": hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "system_sha256": hashlib.sha256(ROW_ENUMERATION_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
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


def _attempt_id(request: dict[str, Any]) -> str:
    return (
        "row_attempt_"
        + _hash(
            {
                "case_id": request["case_id"],
                "model_requested": request["model_requested"],
                "repetition_index": request["repetition_index"],
                "base_batch_id": request["base_batch_id"],
                "batch_id": request["batch_id"],
                "depth": request["depth"],
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


def _entry_payload(
    *,
    request: dict[str, Any],
    case: _PreparedCase,
    model: RowBakeoffModelSpec,
    repetition_index: int,
    base_batch: RowBatch,
    batch: RowBatch,
    depth: int,
    status: str,
    schema_status: str,
    contract_status: str,
    contract_failure_codes: list[str] | None = None,
    records: dict[str, RowDispositionRecord] | None = None,
    unresolved_row_ids: list[str] | None = None,
    unknown_row_ids: list[str] | None = None,
    invalid_row_reasons: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    call: ProviderCall | None = None,
    normalized_wire_proposal: _NormalizedRowWireProposal | None = None,
    error: _SafeError | None = None,
) -> _AttemptEntry:
    attempt_id = _attempt_id(request)
    payload: dict[str, Any] = {
        "schema_version": _ATTEMPT_VERSION,
        "attempt_id": attempt_id,
        "attempt_contract_sha256": _hash(
            {
                "attempt_id": attempt_id,
                "request_sha256": request["request_sha256"],
                "batch_binding_sha256": request["batch_binding_sha256"],
                "row_plan_sha256": case.locator.row_plan_sha256,
            }
        ),
        "case_id": case.locator.case_id,
        "model": model.model,
        "repetition_index": repetition_index,
        "base_batch_id": base_batch.batch_id,
        "batch_id": batch.batch_id,
        "depth": depth,
        "row_ids": [row.row_id for row in batch.rows],
        "request": request,
        "status": status,
        "schema_status": schema_status,
        "contract_status": contract_status,
        "contract_failure_codes": contract_failure_codes or [],
        "records": records or {},
        "unresolved_row_ids": (
            unresolved_row_ids
            if unresolved_row_ids is not None
            else [row.row_id for row in batch.rows]
        ),
        "unknown_row_ids": unknown_row_ids or [],
        "invalid_row_reasons": invalid_row_reasons or {},
        "warnings": warnings or [],
        "call": call,
        "normalized_wire_proposal": normalized_wire_proposal,
        "error": error,
    }
    payload["entry_sha256"] = _hash(payload)
    return _AttemptEntry.model_validate(payload)


@dataclass(frozen=True, slots=True)
class _ReplayClient:
    proposal: _NormalizedRowWireProposal
    call: ProviderCall

    def structured_chat(self, **_kwargs: Any) -> StructuredResponse:
        return StructuredResponse(
            payload=self.proposal.structured_content,
            call=self.call,
        )


def _entry_response_binding_matches(
    entry: _AttemptEntry,
    *,
    context: _ExecutionContext,
    case: _PreparedCase,
    model: RowBakeoffModelSpec,
    batch: RowBatch,
) -> bool:
    """Replay the exact wire payload through production conversion before reuse."""

    if entry.schema_status != "valid":
        return entry.normalized_wire_proposal is None and not entry.records
    if entry.call is None or entry.normalized_wire_proposal is None:
        return False
    try:
        result = extract_row_batch_candidates(
            client=_ReplayClient(  # type: ignore[arg-type]
                entry.normalized_wire_proposal,
                entry.call,
            ),
            model=model.model,
            paper_id=case.locator.paper_id,
            paper_title=case.locator.paper_title,
            batch=batch,
            max_tokens=context.config.max_tokens,
            temperature=context.config.temperature,
            reasoning_effort=context.config.reasoning_effort,
            seed=context.config.seed,  # type: ignore[arg-type]
        )
    except Exception:
        return False
    contract_eligible = entry.contract_status == "satisfied"
    expected_status = (
        "contract_failure"
        if not contract_eligible
        else ("partial_invalid" if result.unresolved_row_ids else "success")
    )
    expected_records = result.records if contract_eligible else {}
    expected_unresolved = (
        result.unresolved_row_ids if contract_eligible else [row.row_id for row in batch.rows]
    )
    expected_invalid = (
        result.invalid_row_reasons
        if contract_eligible
        else {row.row_id: "request_contract_failure" for row in batch.rows}
    )
    return bool(
        result.call == entry.call
        and entry.status == expected_status
        and entry.records == expected_records
        and entry.unresolved_row_ids == expected_unresolved
        and entry.unknown_row_ids == result.unknown_row_ids
        and entry.invalid_row_reasons == expected_invalid
        and entry.warnings == result.warnings
    )


def _entry_reusable(
    entry: _AttemptEntry,
    *,
    context: _ExecutionContext,
    request: dict[str, Any],
    case: _PreparedCase,
    model: RowBakeoffModelSpec,
    repetition_index: int,
    base_batch: RowBatch,
    batch: RowBatch,
    depth: int,
) -> bool:
    expected_id = _attempt_id(request)
    expected_contract = _hash(
        {
            "attempt_id": expected_id,
            "request_sha256": request["request_sha256"],
            "batch_binding_sha256": request["batch_binding_sha256"],
            "row_plan_sha256": case.locator.row_plan_sha256,
        }
    )
    if (
        entry.attempt_id != expected_id
        or entry.attempt_contract_sha256 != expected_contract
        or entry.case_id != case.locator.case_id
        or entry.model != model.model
        or entry.repetition_index != repetition_index
        or entry.base_batch_id != base_batch.batch_id
        or entry.batch_id != batch.batch_id
        or entry.depth != depth
        or entry.row_ids != [row.row_id for row in batch.rows]
        or entry.request != request
    ):
        return False
    if entry.call is not None:
        status, failures = _contract_assessment(entry.call, request)
        if status != entry.contract_status or failures != entry.contract_failure_codes:
            return False
    elif entry.contract_status != "not_observed":
        return False
    return _entry_response_binding_matches(
        entry,
        context=context,
        case=case,
        model=model,
        batch=batch,
    )


def _phase_seal(
    configuration_sha256: str,
    run_contract_sha256: str,
    attempts: dict[str, _AttemptEntry],
    repetitions: dict[str, _RepetitionResult],
) -> str:
    return _hash(
        {
            "schema_version": "row-bakeoff-provider-seal/0.1",
            "configuration_sha256": configuration_sha256,
            "run_contract_sha256": run_contract_sha256,
            "attempt_sha256s": {key: value.entry_sha256 for key, value in sorted(attempts.items())},
            "repetition_sha256s": {
                key: value.repetition_sha256 for key, value in sorted(repetitions.items())
            },
        }
    )


def _checkpoint(
    context: _ExecutionContext,
    attempts: dict[str, _AttemptEntry],
    repetitions: dict[str, _RepetitionResult],
    *,
    sealed: bool,
) -> _Checkpoint:
    payload: dict[str, Any] = {
        "schema_version": _CHECKPOINT_VERSION,
        "bakeoff_id": context.config.bakeoff_id,
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "status": "sealed" if sealed else "in_progress",
        "attempts": attempts,
        "repetitions": repetitions,
        "provider_phase_seal_sha256": (
            _phase_seal(
                context.configuration_sha256,
                context.run_contract_sha256,
                attempts,
                repetitions,
            )
            if sealed
            else None
        ),
    }
    payload["checkpoint_sha256"] = _hash(payload)
    return _Checkpoint.model_validate(payload)


def _write_checkpoint(
    path: Path,
    context: _ExecutionContext,
    attempts: dict[str, _AttemptEntry],
    repetitions: dict[str, _RepetitionResult],
    *,
    sealed: bool,
) -> _Checkpoint:
    checkpoint = _checkpoint(context, attempts, repetitions, sealed=sealed)
    write_json(path, checkpoint.model_dump(mode="json", exclude_none=False))
    return checkpoint


def _load_checkpoint(
    path: Path,
    context: _ExecutionContext,
) -> tuple[dict[str, _AttemptEntry], dict[str, _RepetitionResult], list[str]]:
    if not path.exists():
        return {}, {}, []
    try:
        checkpoint = _Checkpoint.model_validate(read_json(path))
    except Exception:
        return {}, {}, ["checkpoint_invalid"]
    if checkpoint.bakeoff_id != context.config.bakeoff_id:
        return {}, {}, ["checkpoint_bakeoff_id_mismatch"]
    if checkpoint.configuration_sha256 != context.configuration_sha256:
        return {}, {}, ["checkpoint_configuration_stale"]
    if checkpoint.run_contract_sha256 != context.run_contract_sha256:
        return {}, {}, ["checkpoint_run_contract_stale"]
    return dict(checkpoint.attempts), dict(checkpoint.repetitions), []


def _validation_view(call: ProviderCall) -> ProviderCall:
    """Return the exact call now that production and bakeoff routing are aligned."""

    return call


def _production_attempt_status(entry: _AttemptEntry, *, eligible: bool) -> str:
    """Map bakeoff diagnostics into the narrower production outcome vocabulary."""

    if not eligible:
        return "provider_request_failed"
    if entry.status in {"success", "partial_invalid"}:
        return entry.status
    if entry.status == "response_validation_failure":
        if entry.error is None or entry.error.code not in {
            "invalid_json",
            "schema_validation",
            "wire_validation",
        }:
            raise ValueError("bakeoff response failure lacks a production validation code")
        return f"provider_response_{entry.error.code}"
    raise ValueError("contract-eligible bakeoff attempt has no production status mapping")


def _attempt_telemetry(entry: _AttemptEntry, *, eligible: bool) -> RowAttemptTelemetry:
    return RowAttemptTelemetry(
        batch_id=entry.batch_id,
        depth=entry.depth,
        row_ids=entry.row_ids,
        status=_production_attempt_status(entry, eligible=eligible),
        resolved_row_ids=list(entry.records) if eligible and entry.schema_status == "valid" else [],
        unresolved_row_ids=(
            entry.unresolved_row_ids
            if eligible and entry.schema_status == "valid"
            else entry.row_ids
        ),
        unknown_row_ids=entry.unknown_row_ids if entry.schema_status == "valid" else [],
        completed_provider_call=eligible and entry.call is not None,
    )


def _batch_ledger(
    context: _ExecutionContext,
    case: _PreparedCase,
    model: RowBakeoffModelSpec,
    base_batch: RowBatch,
    entries: list[_AttemptEntry],
):
    validation_config = case.plan.config
    if context.config.frame_mode == "contract_smoke":
        validation_config = validation_config.model_copy(update={"max_recovery_depth": 0})
    outcome = RowEnumerationOutcome()
    for entry in entries:
        eligible = entry.contract_status == "satisfied" and entry.call is not None
        outcome.attempts.append(_attempt_telemetry(entry, eligible=eligible))
        if eligible:
            outcome.calls.append(_validation_view(entry.call))
        if eligible and entry.schema_status == "valid":
            overlap = set(outcome.records) & set(entry.records)
            if overlap:
                raise ValueError("cached row attempts assign duplicate terminal ownership")
            outcome.records.update(entry.records)
            outcome.unknown_row_ids.extend(entry.unknown_row_ids)
            outcome.invalid_row_reasons.update(entry.invalid_row_reasons)
        else:
            for row_id in entry.row_ids:
                outcome.invalid_row_reasons[row_id] = entry.status
    outcome.unresolved_row_ids = [
        row.row_id for row in base_batch.rows if row.row_id not in outcome.records
    ]
    for row_id in outcome.records:
        outcome.invalid_row_reasons.pop(row_id, None)
    return validate_batch_outcome_against(
        base_batch,
        outcome,
        config=validation_config,
        paper_id=case.locator.paper_id,
        paper_title=case.locator.paper_title,
        model=model.model,
        max_tokens=context.config.max_tokens,
        temperature=context.config.temperature,
        reasoning_effort=context.config.reasoning_effort,
        seed=context.config.seed,
    )


def _repetition_id(
    context: _ExecutionContext,
    case: _PreparedCase,
    model: RowBakeoffModelSpec,
    repetition_index: int,
) -> str:
    return (
        "row_repetition_"
        + _hash(
            {
                "run_contract_sha256": context.run_contract_sha256,
                "case_id": case.locator.case_id,
                "model": model.model,
                "repetition_index": repetition_index,
                "selected_rows_sha256": case.selected_rows_sha256,
            }
        )[:24]
    )


def _repetition_result(
    context: _ExecutionContext,
    case: _PreparedCase,
    model: RowBakeoffModelSpec,
    repetition_index: int,
    attempt_ids: list[str],
    batch_ledgers: list[Any],
) -> _RepetitionResult:
    terminal_by_row: dict[str, RowTerminalRecord] = {}
    for ledger in batch_ledgers:
        overlap = set(terminal_by_row) & set(ledger.terminal_by_row)
        if overlap:
            raise ValueError("row batch ledgers overlap")
        terminal_by_row.update(ledger.terminal_by_row)
    selected_unbatchable = {row.row_id: row for row in case.selected_unbatchable}
    for row_id, row in selected_unbatchable.items():
        if row_id in terminal_by_row:
            raise ValueError("unbatchable row already has a batch terminal owner")
        terminal_by_row[row_id] = RowTerminalRecord(
            row_id=row_id,
            state=RowTerminalState.UNSUPPORTED,
            unsupported_reasons=row.reasons,
        )
    if set(terminal_by_row) != set(case.locator.selected_row_ids):
        raise ValueError("repetition does not own every selected row exactly once")
    counts_by_state = {state: 0 for state in RowTerminalState}
    for terminal in terminal_by_row.values():
        counts_by_state[terminal.state] += 1
    payload: dict[str, Any] = {
        "schema_version": _REPETITION_VERSION,
        "repetition_id": _repetition_id(context, case, model, repetition_index),
        "case_id": case.locator.case_id,
        "model": model.model,
        "repetition_index": repetition_index,
        "selected_rows_sha256": case.selected_rows_sha256,
        "row_plan_sha256": case.locator.row_plan_sha256,
        "attempt_ids": attempt_ids,
        "terminal_by_row": terminal_by_row,
        "counts": RowTerminalCounts(
            planned=len(terminal_by_row),
            result=counts_by_state[RowTerminalState.RESULT],
            not_result=counts_by_state[RowTerminalState.NOT_RESULT],
            uncertain=counts_by_state[RowTerminalState.UNCERTAIN],
            unresolved=counts_by_state[RowTerminalState.UNRESOLVED],
            unsupported=counts_by_state[RowTerminalState.UNSUPPORTED],
        ),
    }
    payload["repetition_sha256"] = _hash(payload)
    return _RepetitionResult.model_validate(payload)


def _run_attempt(
    context: _ExecutionContext,
    case: _PreparedCase,
    model: RowBakeoffModelSpec,
    repetition_index: int,
    base_batch: RowBatch,
    batch: RowBatch,
    depth: int,
    client: _ConfiguredClient,
) -> tuple[dict[str, Any], _AttemptEntry]:
    request = _request_binding(context, case, model, repetition_index, base_batch, batch, depth)
    try:
        result = extract_row_batch_candidates(
            client=client,
            model=model.model,
            paper_id=case.locator.paper_id,
            paper_title=case.locator.paper_title,
            batch=batch,
            max_tokens=context.config.max_tokens,
            temperature=context.config.temperature,
            reasoning_effort=context.config.reasoning_effort,
            seed=context.config.seed,  # type: ignore[arg-type]
        )
    except ProviderBudgetError:
        raise
    except ProviderResponseValidationError as error:
        contract_status, failures = _contract_assessment(error.call, request)
        entry = _entry_payload(
            request=request,
            case=case,
            model=model,
            repetition_index=repetition_index,
            base_batch=base_batch,
            batch=batch,
            depth=depth,
            status="response_validation_failure",
            schema_status="invalid",
            contract_status=contract_status,
            contract_failure_codes=failures,
            call=error.call,
            error=_safe_error("provider_response_validation", error),
        )
    except Exception as error:
        entry = _entry_payload(
            request=request,
            case=case,
            model=model,
            repetition_index=repetition_index,
            base_batch=base_batch,
            batch=batch,
            depth=depth,
            status="provider_failure",
            schema_status="not_observed",
            contract_status="not_observed",
            error=_safe_error("provider_call", error),
        )
    else:
        contract_status, failures = _contract_assessment(result.call, request)
        status = (
            "contract_failure"
            if contract_status == "failed"
            else ("partial_invalid" if result.unresolved_row_ids else "success")
        )
        contract_eligible = contract_status == "satisfied"
        entry = _entry_payload(
            request=request,
            case=case,
            model=model,
            repetition_index=repetition_index,
            base_batch=base_batch,
            batch=batch,
            depth=depth,
            status=status,
            schema_status="valid",
            contract_status=contract_status,
            contract_failure_codes=failures,
            # A schema-valid response obtained under a different capability/request
            # contract is retained as telemetry but never merged into row ownership.
            records=result.records if contract_eligible else {},
            unresolved_row_ids=(
                result.unresolved_row_ids
                if contract_eligible
                else [row.row_id for row in batch.rows]
            ),
            unknown_row_ids=result.unknown_row_ids,
            invalid_row_reasons=(
                result.invalid_row_reasons
                if contract_eligible
                else {row.row_id: "request_contract_failure" for row in batch.rows}
            ),
            warnings=result.warnings,
            call=result.call,
            normalized_wire_proposal=client.normalized_wire_proposal_for(result.call),
        )
    return request, entry


def _public_call(call: ProviderCall) -> dict[str, Any]:
    return public_provider_call(call)


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


def _attempt_aggregate(entries: list[_AttemptEntry]) -> dict[str, Any]:
    first = [entry for entry in entries if entry.repetition_index == 1]

    def schema_report(values: list[_AttemptEntry]) -> dict[str, Any]:
        valid = sum(entry.schema_status == "valid" for entry in values)
        invalid = sum(entry.schema_status == "invalid" for entry in values)
        observed = valid + invalid
        return {
            "logical_attempts": len(values),
            "valid": valid,
            "invalid": invalid,
            "not_observed": len(values) - observed,
            "valid_of_observed": _rate(valid, observed),
            "valid_end_to_end": _rate(valid, len(values)),
        }

    calls = [entry.call for entry in entries if entry.call is not None]
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
    return {
        "execution": {
            "logical_attempts_terminal": len(entries),
            "success": sum(entry.status == "success" for entry in entries),
            "partial_invalid": sum(entry.status == "partial_invalid" for entry in entries),
            "contract_failure": sum(entry.status == "contract_failure" for entry in entries),
            "response_validation_failure": sum(
                entry.status == "response_validation_failure" for entry in entries
            ),
            "provider_failure": sum(entry.status == "provider_failure" for entry in entries),
        },
        "schema": {"all_repetitions": schema_report(entries), "first_pass": schema_report(first)},
        "contract": {
            "satisfied": sum(entry.contract_status == "satisfied" for entry in entries),
            "failed": sum(entry.contract_status == "failed" for entry in entries),
            "not_observed": sum(entry.contract_status == "not_observed" for entry in entries),
            "satisfaction_rate": _rate(
                sum(entry.contract_status == "satisfied" for entry in entries), len(entries)
            ),
        },
        "usage": usage,
    }


def _stability(repetitions: list[_RepetitionResult]) -> dict[str, Any]:
    by_case: dict[str, list[_RepetitionResult]] = {}
    for repetition in repetitions:
        by_case.setdefault(repetition.case_id, []).append(repetition)
    pairs = 0
    row_comparisons = 0
    matches = 0
    for case_repetitions in by_case.values():
        for left, right in combinations(case_repetitions, 2):
            pairs += 1
            if set(left.terminal_by_row) != set(right.terminal_by_row):
                raise ValueError("repetition stability rows do not match")
            for row_id in left.terminal_by_row:
                row_comparisons += 1
                matches += int(
                    left.terminal_by_row[row_id].state is right.terminal_by_row[row_id].state
                )
    return {
        "repetition_pair_denominator": pairs,
        "row_state_comparison_denominator": row_comparisons,
        "matching_row_states": matches,
        "row_state_match_rate": _rate(matches, row_comparisons),
    }


def run_row_bakeoff_provider_phase(
    config: RowBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    client: Any,
    code_root: Path | None = None,
    output_path: Path | None = None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> dict[str, Any]:
    """Run or exactly resume the predeclared label-blind row experiment."""

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
    attempts, repetitions, checkpoint_rejections = _load_checkpoint(checkpoint_path, context)
    configured_client = _ConfiguredClient(
        client=client,
        temperature=config.temperature,
        reasoning_effort=config.reasoning_effort,
        seed=config.seed,
    )
    # A bounded stop before the first dispatch must still leave a valid resumable state.
    _write_checkpoint(checkpoint_path, context, attempts, repetitions, sealed=False)
    expected_attempt_ids: list[str] = []
    expected_repetition_ids: list[str] = []
    reused: set[str] = set()
    executed: set[str] = set()
    for model in context.models:
        for case in context.cases:
            for repetition_index in range(1, config.fresh_repetitions + 1):
                repetition_attempt_ids: list[str] = []
                batch_ledgers: list[Any] = []
                for base_batch in case.selected_batches:
                    base_request = _request_binding(
                        context,
                        case,
                        model,
                        repetition_index,
                        base_batch,
                        base_batch,
                        0,
                    )
                    base_id = _attempt_id(base_request)
                    cached = attempts.get(base_id)
                    if cached is not None and _entry_reusable(
                        cached,
                        context=context,
                        request=base_request,
                        case=case,
                        model=model,
                        repetition_index=repetition_index,
                        base_batch=base_batch,
                        batch=base_batch,
                        depth=0,
                    ):
                        base_entry = cached
                        reused.add(base_id)
                    else:
                        if cached is not None:
                            checkpoint_rejections.append("checkpoint_attempt_stale_or_tampered")
                            attempts.pop(base_id, None)
                            _write_checkpoint(
                                checkpoint_path, context, attempts, repetitions, sealed=False
                            )
                        _, base_entry = _run_attempt(
                            context,
                            case,
                            model,
                            repetition_index,
                            base_batch,
                            base_batch,
                            0,
                            configured_client,
                        )
                        attempts[base_id] = base_entry
                        executed.add(base_id)
                        _write_checkpoint(
                            checkpoint_path, context, attempts, repetitions, sealed=False
                        )
                    expected_attempt_ids.append(base_id)
                    repetition_attempt_ids.append(base_id)
                    batch_entries = [base_entry]
                    # A contract smoke has a frozen scientific denominator of one call per
                    # predeclared base batch. Recovery children are useful for the balanced
                    # quality frame, but dispatching them here would make technical
                    # qualification adaptively exceed its declared call count.
                    if (
                        config.frame_mode != "contract_smoke"
                        and case.plan.config.max_recovery_depth
                        and base_entry.unresolved_row_ids
                    ):
                        children = recovery_batches(base_batch, base_entry.unresolved_row_ids)
                        for child in children:
                            child_request = _request_binding(
                                context,
                                case,
                                model,
                                repetition_index,
                                base_batch,
                                child,
                                1,
                            )
                            child_id = _attempt_id(child_request)
                            cached_child = attempts.get(child_id)
                            if cached_child is not None and _entry_reusable(
                                cached_child,
                                context=context,
                                request=child_request,
                                case=case,
                                model=model,
                                repetition_index=repetition_index,
                                base_batch=base_batch,
                                batch=child,
                                depth=1,
                            ):
                                child_entry = cached_child
                                reused.add(child_id)
                            else:
                                if cached_child is not None:
                                    checkpoint_rejections.append(
                                        "checkpoint_attempt_stale_or_tampered"
                                    )
                                    attempts.pop(child_id, None)
                                    _write_checkpoint(
                                        checkpoint_path,
                                        context,
                                        attempts,
                                        repetitions,
                                        sealed=False,
                                    )
                                _, child_entry = _run_attempt(
                                    context,
                                    case,
                                    model,
                                    repetition_index,
                                    base_batch,
                                    child,
                                    1,
                                    configured_client,
                                )
                                attempts[child_id] = child_entry
                                executed.add(child_id)
                                _write_checkpoint(
                                    checkpoint_path,
                                    context,
                                    attempts,
                                    repetitions,
                                    sealed=False,
                                )
                            expected_attempt_ids.append(child_id)
                            repetition_attempt_ids.append(child_id)
                            batch_entries.append(child_entry)
                    batch_ledgers.append(
                        _batch_ledger(context, case, model, base_batch, batch_entries)
                    )
                repetition = _repetition_result(
                    context,
                    case,
                    model,
                    repetition_index,
                    repetition_attempt_ids,
                    batch_ledgers,
                )
                repetitions[repetition.repetition_id] = repetition
                expected_repetition_ids.append(repetition.repetition_id)
                _write_checkpoint(checkpoint_path, context, attempts, repetitions, sealed=False)
    expected_attempt_set = set(expected_attempt_ids)
    expected_repetition_set = set(expected_repetition_ids)
    unexpected_attempts = set(attempts) - expected_attempt_set
    unexpected_repetitions = set(repetitions) - expected_repetition_set
    if unexpected_attempts or unexpected_repetitions:
        checkpoint_rejections.append("checkpoint_unexpected_entries_removed")
        attempts = {key: value for key, value in attempts.items() if key in expected_attempt_set}
        repetitions = {
            key: value for key, value in repetitions.items() if key in expected_repetition_set
        }
    if set(attempts) != expected_attempt_set or set(repetitions) != expected_repetition_set:
        raise RuntimeError("row checkpoint does not exactly partition the frozen experiment")
    sealed_checkpoint = _write_checkpoint(
        checkpoint_path, context, attempts, repetitions, sealed=True
    )
    ordered_attempts = [attempts[key] for key in expected_attempt_ids]
    ordered_repetitions = [repetitions[key] for key in expected_repetition_ids]
    models: list[dict[str, Any]] = []
    executed_models = {item.model for item in context.models}
    for model in config.models:
        if model.model not in executed_models:
            if context.execution_selection is None:
                raise RuntimeError("row model was skipped without an execution selection")
            decision = context.execution_selection.decision_for(model.model)
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
            continue
        model_attempts = [entry for entry in ordered_attempts if entry.model == model.model]
        model_repetitions = [item for item in ordered_repetitions if item.model == model.model]
        decision = (
            context.execution_selection.decision_for(model.model)
            if context.execution_selection is not None
            else None
        )
        models.append(
            {
                "model": model.model,
                "label": model.label,
                "contract_eligibility": (
                    decision.status if decision is not None else "not_assessed"
                ),
                "eligibility_reason_codes": (decision.reason_codes if decision is not None else []),
                "matched_quality_status": "executed",
                "quality_status": "pending_offline_scoring",
                "quality": None,
                "aggregate": _attempt_aggregate(model_attempts),
                "stability": _stability(model_repetitions),
            }
        )
    result = {
        "schema_version": _RESULT_VERSION,
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
                    "source_binding_sha256": case.locator.source_binding_sha256,
                    "layout_binding_sha256": case.locator.layout_binding_sha256,
                    "row_plan_sha256": case.locator.row_plan_sha256,
                    "layout_file_sha256": case.layout_file_sha256,
                    "row_plan_file_sha256": case.row_plan_file_sha256,
                    "selected_rows_sha256": case.selected_rows_sha256,
                    "selected_row_count": len(case.locator.selected_row_ids),
                    "selected_unbatchable_count": len(case.selected_unbatchable),
                    "base_batches_per_repetition": len(case.selected_batches),
                    "maximum_attempts_per_repetition": len(case.selected_batches)
                    * (
                        1
                        + (
                            2
                            if config.frame_mode != "contract_smoke"
                            and case.plan.config.max_recovery_depth
                            else 0
                        )
                    ),
                }
                for case in context.cases
            ],
        },
        "code": context.code,
        "code_sha256": context.code_sha256,
        "stage_contract_sha256": context.stage_contract_sha256,
        "execution_selection": (
            {
                "selection_sha256": context.execution_selection.selection_sha256,
                "smoke_provider_phase_seal_sha256": (
                    context.execution_selection.smoke_provider_phase_seal_sha256
                ),
                "smoke_checkpoint_sha256": context.execution_selection.smoke_checkpoint_sha256,
                "eligible_models": list(context.execution_selection.eligible_models),
            }
            if context.execution_selection is not None
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
            "logical_attempts_reused": len(reused),
            "logical_attempts_executed": len(executed),
        },
        "accounting": {
            "repetitions_predeclared": len(context.models)
            * len(config.cases)
            * config.fresh_repetitions,
            "repetitions_terminal": len(ordered_repetitions),
            "selected_row_decisions": sum(item.counts.planned for item in ordered_repetitions),
            "logical_attempts_terminal": len(ordered_attempts),
        },
        "aggregate": _attempt_aggregate(ordered_attempts),
        "models": models,
        "attempts": [
            {
                "attempt_id": entry.attempt_id,
                "attempt_contract_sha256": entry.attempt_contract_sha256,
                "case_id": entry.case_id,
                "model": entry.model,
                "repetition_index": entry.repetition_index,
                "depth": entry.depth,
                "request": entry.request,
                "status": entry.status,
                "schema_status": entry.schema_status,
                "contract_status": entry.contract_status,
                "contract_failure_codes": entry.contract_failure_codes,
                "input_row_count": len(entry.row_ids),
                "resolved_row_count": len(entry.records),
                "unresolved_row_count": len(entry.unresolved_row_ids),
                "unknown_row_id_count": len(entry.unknown_row_ids),
                "invalid_row_count": len(entry.invalid_row_reasons),
                "call": _public_call(entry.call) if entry.call is not None else None,
                "error": entry.error.model_dump(mode="json") if entry.error else None,
                "entry_sha256": entry.entry_sha256,
                "resumed": entry.attempt_id in reused,
            }
            for entry in ordered_attempts
        ],
        "privacy": {
            "labels_loaded_during_provider_phase": False,
            "row_ids_in_public_result": False,
            "exact_evidence_in_public_result": False,
            "exact_rows_and_evidence_checkpoint_only": True,
            "typed_wire_proposals_checkpoint_only": True,
        },
        "automatic_positive_promotion": False,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def _load_sealed_checkpoint_for_scoring(
    checkpoint_path: Path,
    context: _ExecutionContext,
) -> tuple[_Checkpoint, list[_AttemptEntry], list[_RepetitionResult]]:
    checkpoint = _Checkpoint.model_validate(read_json(checkpoint_path))
    if checkpoint.status != "sealed":
        raise ValueError("row provider phase checkpoint is not sealed")
    if (
        checkpoint.configuration_sha256 != context.configuration_sha256
        or checkpoint.run_contract_sha256 != context.run_contract_sha256
    ):
        raise ValueError("row provider phase checkpoint is stale")
    expected_repetitions: list[_RepetitionResult] = []
    expected_attempt_ids: list[str] = []
    ordered_attempts: list[_AttemptEntry] = []
    for model in context.models:
        for case in context.cases:
            for repetition_index in range(1, context.config.fresh_repetitions + 1):
                repetition_attempt_ids: list[str] = []
                batch_ledgers: list[Any] = []
                for base_batch in case.selected_batches:
                    base_request = _request_binding(
                        context,
                        case,
                        model,
                        repetition_index,
                        base_batch,
                        base_batch,
                        0,
                    )
                    base_id = _attempt_id(base_request)
                    base_entry = checkpoint.attempts.get(base_id)
                    if base_entry is None or not _entry_reusable(
                        base_entry,
                        context=context,
                        request=base_request,
                        case=case,
                        model=model,
                        repetition_index=repetition_index,
                        base_batch=base_batch,
                        batch=base_batch,
                        depth=0,
                    ):
                        raise ValueError("row checkpoint contains an invalid base attempt")
                    expected_attempt_ids.append(base_id)
                    ordered_attempts.append(base_entry)
                    repetition_attempt_ids.append(base_id)
                    batch_entries = [base_entry]
                    if (
                        context.config.frame_mode != "contract_smoke"
                        and case.plan.config.max_recovery_depth
                        and base_entry.unresolved_row_ids
                    ):
                        for child in recovery_batches(
                            base_batch,
                            base_entry.unresolved_row_ids,
                        ):
                            child_request = _request_binding(
                                context,
                                case,
                                model,
                                repetition_index,
                                base_batch,
                                child,
                                1,
                            )
                            child_id = _attempt_id(child_request)
                            child_entry = checkpoint.attempts.get(child_id)
                            if child_entry is None or not _entry_reusable(
                                child_entry,
                                context=context,
                                request=child_request,
                                case=case,
                                model=model,
                                repetition_index=repetition_index,
                                base_batch=base_batch,
                                batch=child,
                                depth=1,
                            ):
                                raise ValueError(
                                    "row checkpoint contains an invalid recovery attempt"
                                )
                            expected_attempt_ids.append(child_id)
                            ordered_attempts.append(child_entry)
                            repetition_attempt_ids.append(child_id)
                            batch_entries.append(child_entry)
                    batch_ledgers.append(
                        _batch_ledger(context, case, model, base_batch, batch_entries)
                    )
                expected_repetition = _repetition_result(
                    context,
                    case,
                    model,
                    repetition_index,
                    repetition_attempt_ids,
                    batch_ledgers,
                )
                repetition = checkpoint.repetitions.get(expected_repetition.repetition_id)
                if repetition is None:
                    raise ValueError("row checkpoint omits a predeclared repetition")
                if repetition != expected_repetition:
                    raise ValueError("row checkpoint repetition binding is invalid")
                expected_repetitions.append(repetition)
    if len(expected_repetitions) != len(checkpoint.repetitions):
        raise ValueError("row checkpoint contains unexpected repetitions")
    if len(expected_attempt_ids) != len(set(expected_attempt_ids)):
        raise ValueError("row checkpoint repeats an expected attempt")
    if set(expected_attempt_ids) != set(checkpoint.attempts):
        raise ValueError("row checkpoint attempts do not match repetition ledgers")
    return checkpoint, ordered_attempts, expected_repetitions


def derive_row_execution_selection(
    config: RowBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    experiment_id: str,
    public_manifest_sha256: str,
    wire_schema_gate: str = "0.99",
    code_root: Path | None = None,
) -> StageExecutionSelection:
    """Derive row-route eligibility from sealed first-pass base smoke attempts."""

    if config.frame_mode != "contract_smoke" or config.fresh_repetitions != 1:
        raise ValueError("row eligibility requires a one-repetition contract-smoke frame")
    context = _prepare_execution(config, project_root=project_root, code_root=code_root)
    checkpoint, attempts, _ = _load_sealed_checkpoint_for_scoring(checkpoint_path, context)
    if checkpoint.provider_phase_seal_sha256 is None:
        raise ValueError("row contract smoke is missing its provider phase seal")
    base_attempts = [entry for entry in attempts if entry.depth == 0]
    expected_base_attempts_per_model = sum(len(case.selected_batches) for case in context.cases)
    return derive_stage_execution_selection(
        experiment_id=experiment_id,
        stage="row_disposition",
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
        expected_slots_per_model=expected_base_attempts_per_model,
        entries=base_attempts,
    )


def _load_labels(
    path: Path,
    context: _ExecutionContext,
) -> tuple[dict[tuple[str, str], RowDispositionLabel], str]:
    labels = [
        RowDispositionLabel.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_key = {(label.case_id, label.row_id): label for label in labels}
    if len(by_key) != len(labels):
        raise ValueError("row label keys must be unique")
    expected = {
        (case.locator.case_id, row_id): case
        for case in context.cases
        for row_id in case.locator.selected_row_ids
    }
    if set(by_key) != set(expected):
        raise ValueError("row labels do not exactly cover selected rows")
    for key, case in expected.items():
        label = by_key[key]
        if label.row_plan_sha256 != case.locator.row_plan_sha256:
            raise ValueError("row label plan binding mismatch")
    annotators = {label.annotator for label in labels}
    if len(annotators) != 1:
        raise ValueError("row bake-off requires exactly one annotator")
    for case in context.cases:
        balance = case.locator.balanced_rows_per_class
        if balance is None:
            raise ValueError("row quality labels require a declared balanced frame")
        case_labels = [
            by_key[(case.locator.case_id, row_id)] for row_id in case.locator.selected_row_ids
        ]
        counts = {label: 0 for label in _LABEL_CLASSES}
        for item in case_labels:
            counts[item.label] += 1
        if set(counts.values()) != {balance}:
            raise ValueError("private row labels do not match the predeclared balanced sample")
        unbatchable = {row.row_id for row in case.selected_unbatchable}
        unsupported = {item.row_id for item in case_labels if item.label == "unsupported"}
        if unsupported != unbatchable:
            raise ValueError("unsupported labels do not match frozen unbatchable rows")
    annotator = next(iter(annotators))
    return by_key, hashlib.sha256(annotator.encode("utf-8")).hexdigest()


def _score_model(
    repetitions: list[_RepetitionResult],
    labels: dict[tuple[str, str], RowDispositionLabel],
) -> dict[str, Any]:
    confusion = {
        expected: {predicted: 0 for predicted in _PREDICTION_CLASSES} for expected in _LABEL_CLASSES
    }
    unsafe_result_false_positives = 0
    result_predictions = 0
    review = 0
    planned = 0
    for repetition in repetitions:
        for row_id, terminal in repetition.terminal_by_row.items():
            expected = labels[(repetition.case_id, row_id)].label
            predicted = terminal.state.value
            confusion[expected][predicted] += 1
            planned += 1
            result_predictions += int(predicted == "result")
            unsafe_result_false_positives += int(predicted == "result" and expected != "result")
            review += int(predicted in {"uncertain", "unresolved", "unsupported"})
    per_class: dict[str, Any] = {}
    for label in _LABEL_CLASSES:
        true_positive = confusion[label][label]
        predicted_count = sum(confusion[expected][label] for expected in _LABEL_CLASSES)
        expected_count = sum(confusion[label].values())
        per_class[label] = {
            "true_positive": true_positive,
            "predicted": predicted_count,
            "expected": expected_count,
            "precision": _rate(true_positive, predicted_count),
            "recall": _rate(true_positive, expected_count),
        }
    state_counts = {state: 0 for state in _PREDICTION_CLASSES}
    for repetition in repetitions:
        for terminal in repetition.terminal_by_row.values():
            state_counts[terminal.state.value] += 1
    return {
        "unsafe_result_false_positives": {
            "count": unsafe_result_false_positives,
            "result_predictions": result_predictions,
            "rate_of_all_selected_row_decisions": _rate(unsafe_result_false_positives, planned),
            "rate_of_result_predictions": _rate(unsafe_result_false_positives, result_predictions),
        },
        "exact_accounting": {
            "selected_row_decisions": planned,
            "terminal_row_decisions": sum(state_counts.values()),
            "terminal_state_counts": state_counts,
            "expected_class_counts": {
                label: sum(confusion[label].values()) for label in _LABEL_CLASSES
            },
        },
        "confusion": confusion,
        "per_class": per_class,
        "review_burden": {
            "count": review,
            "rate": _rate(review, planned),
            "includes": ["uncertain", "unresolved", "unsupported"],
        },
        "stability": _stability(repetitions),
        "automatic_positive_promotion_violations": 0,
    }


def _model_terminal_telemetry(
    attempts: list[_AttemptEntry],
    repetitions: list[_RepetitionResult],
) -> tuple[dict[str, Any], list[_RepetitionResult]]:
    by_attempt_id = {entry.attempt_id: entry for entry in attempts}
    contract_complete: list[_RepetitionResult] = []
    for repetition in repetitions:
        repetition_attempts = [by_attempt_id[attempt_id] for attempt_id in repetition.attempt_ids]
        if repetition_attempts and all(
            entry.call is not None
            and entry.schema_status == "valid"
            and entry.contract_status == "satisfied"
            for entry in repetition_attempts
        ):
            contract_complete.append(repetition)
    state_counts = {state.value: 0 for state in RowTerminalState}
    for repetition in repetitions:
        for terminal in repetition.terminal_by_row.values():
            state_counts[terminal.state.value] += 1
    return (
        {
            "repetitions_predeclared": len(repetitions),
            "repetitions_contract_complete": len(contract_complete),
            "repetitions_contract_incomplete": len(repetitions) - len(contract_complete),
            "selected_row_decisions": sum(item.counts.planned for item in repetitions),
            "terminal_state_counts": state_counts,
            "attempts": _attempt_aggregate(attempts),
            "stability": _stability(repetitions),
        },
        contract_complete,
    )


def score_row_bakeoff_offline(
    config: RowBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    labels_path: Path,
    code_root: Path | None = None,
    output_path: Path | None = None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> dict[str, Any]:
    """Join private labels only after validating the sealed provider checkpoint."""

    if config.frame_mode != "balanced_quality":
        raise ValueError("row offline scoring requires a predeclared balanced-quality frame")
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
    checkpoint, attempts, repetitions = _load_sealed_checkpoint_for_scoring(
        checkpoint_path, context
    )
    # Deliberately last: every provider request/result and source binding is sealed.
    labels, annotator_sha256 = _load_labels(labels_path, context)
    models: list[dict[str, Any]] = []
    executed_models = {item.model for item in context.models}
    for model in config.models:
        if model.model not in executed_models:
            if context.execution_selection is None:
                raise RuntimeError("row model was skipped without an execution selection")
            decision = context.execution_selection.decision_for(model.model)
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
                        "repetitions_predeclared": 0,
                        "repetitions_contract_complete": 0,
                        "repetitions_contract_incomplete": 0,
                    },
                    "quality": None,
                    "terminal_telemetry": None,
                }
            )
            continue
        model_attempts = [entry for entry in attempts if entry.model == model.model]
        model_repetitions = [item for item in repetitions if item.model == model.model]
        terminal_telemetry, complete_repetitions = _model_terminal_telemetry(
            model_attempts,
            model_repetitions,
        )
        fully_measured = len(complete_repetitions) == len(model_repetitions)
        quality = _score_model(model_repetitions, labels) if fully_measured else None
        if quality is not None:
            quality["schema"] = terminal_telemetry["attempts"]["schema"]
            quality["contract"] = terminal_telemetry["attempts"]["contract"]
            quality["usage"] = terminal_telemetry["attempts"]["usage"]
        decision = (
            context.execution_selection.decision_for(model.model)
            if context.execution_selection is not None
            else None
        )
        models.append(
            {
                "model": model.model,
                "label": model.label,
                "contract_eligibility": (
                    decision.status if decision is not None else "not_assessed"
                ),
                "eligibility_reason_codes": (decision.reason_codes if decision is not None else []),
                "matched_quality_status": "executed",
                "quality_status": (
                    "measured" if fully_measured else "unmeasured_incomplete_provider_contract"
                ),
                "quality_measurement": {
                    "status": "measured" if fully_measured else "unmeasured",
                    "reason": (
                        None
                        if fully_measured
                        else "one_or_more_repetitions_lacked_valid_contract_observation"
                    ),
                    "repetitions_predeclared": len(model_repetitions),
                    "repetitions_contract_complete": len(complete_repetitions),
                    "repetitions_contract_incomplete": (
                        len(model_repetitions) - len(complete_repetitions)
                    ),
                },
                "quality": quality,
                "terminal_telemetry": terminal_telemetry,
            }
        )
    result = {
        "schema_version": _SCORE_VERSION,
        "bakeoff_id": config.bakeoff_id,
        "provider_phase_seal_sha256": checkpoint.provider_phase_seal_sha256,
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "stage_contract_sha256": context.stage_contract_sha256,
        "execution_selection_sha256": (
            context.execution_selection.selection_sha256
            if context.execution_selection is not None
            else None
        ),
        "declared_models": [item.model for item in config.models],
        "executed_models": [item.model for item in context.models],
        "labels_sha256": sha256_file(labels_path),
        "annotator_sha256": annotator_sha256,
        "models": models,
        "privacy": {
            "labels_loaded_after_provider_phase_sealed": True,
            "row_ids_in_score": False,
            "exact_evidence_in_score": False,
        },
        "automatic_positive_promotion": False,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
