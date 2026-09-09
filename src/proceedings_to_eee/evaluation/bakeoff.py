"""Prompt-isolated, failure-tolerant extractor model bake-offs.

The scientific API checkpoints each provider call, seals the complete label-blind
phase, and exposes offline scoring as a separate operation.  References are not
resolved, opened, or hashed until that seal has been validated.  A historical
combined entry point remains for compatibility, but it is not the sealed experiment
boundary.  Provider payloads, source text, credentials, and exception messages are
excluded from public reports.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from proceedings_to_eee.domain.observation import CandidateObservation, StrictModel
from proceedings_to_eee.domain.status import ClaimType
from proceedings_to_eee.evaluation.artifact_safety import assert_artifact_paths_safe
from proceedings_to_eee.evaluation.control_coverage import (
    control_examination,
    observation_examination,
)
from proceedings_to_eee.evaluation.corpus_score import aggregate_reference_scores
from proceedings_to_eee.evaluation.reference_score import score_reference
from proceedings_to_eee.evaluation.staged_eligibility import (
    StageExecutionSelection,
    StageExecutionSelectionExpectation,
    derive_stage_execution_selection,
    selected_models_for_execution,
)
from proceedings_to_eee.extraction.llm import EXTRACTOR_SCHEMA_NAME, extract_page_candidates
from proceedings_to_eee.extraction.llm_schema import provider_json_schema
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout, extract_pdf_layout
from proceedings_to_eee.extraction.prompt import SYSTEM_PROMPT, page_prompt, prompt_hash
from proceedings_to_eee.extraction.result_blocks import (
    ResultBlock,
    ResultBlockConfig,
    segment_page_result_blocks,
)
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
    OpenRouterClient,
    ProviderCall,
    ProviderResponseValidationError,
    public_provider_call,
    structured_request_contract,
    structured_request_contract_from_call,
)
from proceedings_to_eee.reference import PaperReference, load_reference
from proceedings_to_eee.sources.manifest import SourceManifest, SourceRole, resolve_cached_path
from proceedings_to_eee.validation.candidates import deduplicate_candidates, validate_candidates

EXTRACTOR_SEED = 7
QUALITY_FIELDS = (
    "claim_type",
    "system",
    "dataset",
    "metric",
    "value",
    "unit",
    "slice",
    "page",
    "evidence_kind",
    "evidence_label",
    "evidence_row",
    "evidence_column",
    "evidence_structure",
    "evidence_supported",
    "missingness",
    "joint_semantics",
)
_SCHEMA_ERROR_TYPES = {"JSONDecodeError", "TypeError", "ValidationError", "ValueError"}


class BakeoffModelSpec(StrictModel):
    """One OpenRouter model candidate."""

    model: str = Field(min_length=1)
    label: str = Field(min_length=1)


class BakeoffCaseSpec(StrictModel):
    """One frozen paper page and its prompt-isolated reference annotation."""

    case_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    manifest_path: str = Field(min_length=1)
    reference_path: str = Field(min_length=1)

    @field_validator("manifest_path", "reference_path")
    @classmethod
    def paths_are_project_relative(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("bake-off paths must stay project-relative")
        return path.as_posix()


class BakeoffSegmentation(StrictModel):
    """Serialized form of the production result-block settings."""

    max_lines: int = Field(default=40, ge=2)
    max_characters: int = Field(default=6_000, ge=128)
    context_lines: int = Field(default=8, ge=0)
    trailing_context_lines: int = Field(default=2, ge=0)
    overlap_lines: int = Field(default=3, ge=0)
    signal_gap_lines: int = Field(default=3, ge=0)
    max_blank_gap: int = Field(default=1, ge=0)
    max_data_rows: int | None = Field(default=6, ge=1)
    min_signal_score: float = Field(default=1.5, gt=0.0)
    max_blocks_per_page: int | None = Field(default=6, ge=1)

    def production_config(self) -> ResultBlockConfig:
        return ResultBlockConfig(**self.model_dump())


class ExtractorBakeoffConfig(StrictModel):
    """Versioned, deterministic model-by-case experiment definition."""

    schema_version: Literal[
        "extractor-bakeoff/0.1",
        "extractor-bakeoff/0.2",
        "extractor-bakeoff/0.3",
    ] = "extractor-bakeoff/0.1"
    bakeoff_id: str = Field(min_length=1)
    models: list[BakeoffModelSpec] = Field(min_length=2)
    cases: list[BakeoffCaseSpec] = Field(min_length=1)
    segmentation: BakeoffSegmentation = Field(default_factory=BakeoffSegmentation)
    min_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    max_tokens: int = Field(default=16_000, ge=1)
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    reasoning_effort: str | None = Field(default="minimal", min_length=1)
    seed: int | None = 7
    require_parameters: bool | None = None
    fresh_repetitions: int = Field(default=1, ge=1)

    @model_validator(mode="before")
    @classmethod
    def repair04_defaults_use_the_production_contract(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("schema_version") == "extractor-bakeoff/0.3":
            value = dict(value)
            value.setdefault("temperature", None)
            value.setdefault("reasoning_effort", "minimal")
            value.setdefault("seed", None)
            value.setdefault("require_parameters", True)
            value.setdefault("max_tokens", 16_000)
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def seed_is_an_exact_integer_or_null(cls, value: Any) -> Any:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("seed must be an integer or null")
        return value

    @field_validator("fresh_repetitions", mode="before")
    @classmethod
    def repetitions_are_an_exact_integer(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("fresh_repetitions must be an integer")
        return value

    @field_validator("require_parameters", mode="before")
    @classmethod
    def routing_flag_is_an_exact_boolean_or_omitted(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, bool):
            raise ValueError("require_parameters must be a boolean")
        return value

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> ExtractorBakeoffConfig:
        # Version 0.1 configurations predate the explicit common-capability flag and
        # retain their historical false default. Version 0.2 makes the frozen public
        # experiment's route-capability contract mandatory.
        if self.require_parameters is None:
            self.require_parameters = self.schema_version != "extractor-bakeoff/0.1"
        if self.schema_version in {"extractor-bakeoff/0.2", "extractor-bakeoff/0.3"} and not (
            self.require_parameters
        ):
            raise ValueError(f"{self.schema_version} requires require_parameters=true")
        if self.schema_version == "extractor-bakeoff/0.3" and (
            self.temperature is not None
            or self.reasoning_effort != "minimal"
            or self.seed is not None
            or self.max_tokens != 16_000
        ):
            raise ValueError("extractor-bakeoff/0.3 requires the production request contract")
        models = [item.model for item in self.models]
        cases = [item.case_id for item in self.cases]
        if len(models) != len(set(models)):
            raise ValueError("bake-off model IDs must be unique")
        if len(cases) != len(set(cases)):
            raise ValueError("bake-off case IDs must be unique")
        return self


@dataclass(frozen=True, slots=True)
class _PreparedCase:
    spec: BakeoffCaseSpec
    manifest: SourceManifest
    reference_path: Path | None
    layout: PdfLayout
    blocks: tuple[ResultBlock, ...]
    source_id: str
    source_sha256: str
    manifest_sha256: str
    reference_sha256: str | None


@dataclass(slots=True)
class _ExtractionRun:
    """One fresh logical repetition, retained privately until offline scoring."""

    public_result: dict[str, Any]
    candidates: tuple[Any, ...]
    candidate_fingerprints: tuple[str, ...]
    scoring_eligible: bool


@dataclass(frozen=True, slots=True)
class _Preparation:
    spec: BakeoffCaseSpec
    prepared: _PreparedCase | None
    public_result: dict[str, Any]


class _InputError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def load_bakeoff_config(path: Path) -> ExtractorBakeoffConfig:
    """Load a strict YAML bake-off definition."""

    return ExtractorBakeoffConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _safe_error(stage: str, error: Exception) -> dict[str, str]:
    """Return diagnostics which cannot persist a URL, source excerpt, or credential."""

    return {
        "stage": stage,
        "type": type(error).__name__,
        "code": error.code if isinstance(error, _InputError) else "unexpected_error",
    }


def _project_path(project_root: Path, configured_path: str) -> Path:
    root = project_root.resolve()
    path = (root / configured_path).resolve()
    if not path.is_relative_to(root):
        raise _InputError("path_escaped_project_root")
    return path


def _fragment_for_block(block: ResultBlock) -> PageFragment:
    """Use the same bounded-fragment adapter as the production pipeline."""

    text = block.prompt_text()
    return PageFragment(
        fragment_id=block.block_id,
        source_id=block.source_id,
        page=block.page,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=block.numeric_token_count,
        result_signal_score=block.result_signal_score,
    )


def _normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _coverage_label_is_on_page(value: str, evidence_labels: set[str]) -> bool:
    region = _normalized_label(re.split(r"[·|]", value, maxsplit=1)[0])
    return bool(
        region
        and any(
            region == label or region.startswith(label + " ") or label.startswith(region + " ")
            for label in evidence_labels
        )
    )


def _scope_reference_to_page(
    reference: PaperReference, page: int
) -> tuple[PaperReference, dict[str, int | float | None]]:
    """Create a graph-consistent reference slice owned by one bake-off page."""

    page_evidence_ids = {item.evidence_id for item in reference.evidence if item.page == page}
    observations = []
    referenced_evidence_ids: set[str] = set()
    for observation in reference.observations:
        result_ids = [
            evidence_id
            for evidence_id in observation.result_evidence_ids
            if evidence_id in page_evidence_ids
        ]
        if not result_ids:
            continue
        context_ids = [
            evidence_id
            for evidence_id in observation.context_evidence_ids
            if evidence_id in page_evidence_ids
        ]
        observations.append(
            observation.model_copy(
                update={
                    "result_evidence_ids": result_ids,
                    "context_evidence_ids": context_ids,
                }
            )
        )
        referenced_evidence_ids.update(result_ids)
        referenced_evidence_ids.update(context_ids)

    negative_controls = []
    for control in reference.negative_controls:
        evidence_ids = [
            evidence_id for evidence_id in control.evidence_ids if evidence_id in page_evidence_ids
        ]
        if not evidence_ids:
            continue
        negative_controls.append(control.model_copy(update={"evidence_ids": evidence_ids}))
        referenced_evidence_ids.update(evidence_ids)

    evidence = [item for item in reference.evidence if item.evidence_id in referenced_evidence_ids]
    evidence_labels = {_normalized_label(item.label) for item in evidence if item.label is not None}
    coverage = reference.coverage.model_copy(
        update={
            "fully_annotated_labels": [
                label
                for label in reference.coverage.fully_annotated_labels
                if _coverage_label_is_on_page(label, evidence_labels)
            ],
            "sampled_labels": [
                label
                for label in reference.coverage.sampled_labels
                if _coverage_label_is_on_page(label, evidence_labels)
            ],
        }
    )
    scoped = PaperReference.model_validate(
        reference.model_copy(
            update={
                "coverage": coverage,
                "evidence": evidence,
                "observations": observations,
                "negative_controls": negative_controls,
            }
        ).model_dump(mode="python")
    )

    def share(scoped_count: int, paper_count: int) -> float | None:
        return round(scoped_count / paper_count, 6) if paper_count else None

    scope = {
        "target_page": page,
        "paper_reference_observations": len(reference.observations),
        "scoped_reference_observations": len(observations),
        "observation_reference_share": share(len(observations), len(reference.observations)),
        "paper_negative_controls": len(reference.negative_controls),
        "scoped_negative_controls": len(negative_controls),
        "negative_control_reference_share": share(
            len(negative_controls), len(reference.negative_controls)
        ),
        "paper_evidence_anchors": len(reference.evidence),
        "scoped_evidence_anchors": len(evidence),
        "evidence_reference_share": share(len(evidence), len(reference.evidence)),
    }
    return scoped, scope


def _prepare_case(
    spec: BakeoffCaseSpec,
    *,
    project_root: Path,
    segmentation: ResultBlockConfig,
    bind_reference: bool = True,
) -> _PreparedCase:
    manifest_path = _project_path(project_root, spec.manifest_path)
    reference_path = _project_path(project_root, spec.reference_path) if bind_reference else None
    manifest = SourceManifest.model_validate(read_json(manifest_path))
    if manifest.paper_id != spec.paper_id:
        raise _InputError("manifest_paper_id_mismatch")
    paper_sources = [source for source in manifest.sources if source.role == SourceRole.PAPER]
    if len(paper_sources) != 1:
        raise _InputError("manifest_requires_one_paper_source")
    source = paper_sources[0]
    if source.sha256 is None:
        raise _InputError("paper_source_has_no_sha256")
    pdf_path = resolve_cached_path(source, project_root)
    layout = extract_pdf_layout(pdf_path, source.source_id)
    pages = {page.page: page for page in layout.pages}
    if spec.page not in pages:
        raise _InputError("configured_page_out_of_range")
    blocks = tuple(segment_page_result_blocks(pages[spec.page], config=segmentation))
    if not blocks:
        raise _InputError("configured_page_has_no_result_blocks")

    return _PreparedCase(
        spec=spec,
        manifest=manifest,
        reference_path=reference_path,
        layout=layout,
        blocks=blocks,
        source_id=source.source_id,
        source_sha256=source.sha256,
        manifest_sha256=sha256_file(manifest_path),
        reference_sha256=sha256_file(reference_path) if reference_path is not None else None,
    )


def _load_scoring_reference(
    prepared: _PreparedCase,
) -> tuple[PaperReference, dict[str, int | float | None]]:
    """Join private reference data only after every provider call has completed."""

    if prepared.reference_path is None:
        raise _InputError("reference_not_bound")
    paper_reference = load_reference(prepared.reference_path)
    if paper_reference.paper_id != prepared.spec.paper_id:
        raise _InputError("reference_paper_id_mismatch")
    if paper_reference.source_sha256 != prepared.source_sha256:
        raise _InputError("reference_source_hash_mismatch")
    return _scope_reference_to_page(paper_reference, prepared.spec.page)


def _prepare_cases(
    config: ExtractorBakeoffConfig,
    project_root: Path,
    *,
    bind_references: bool = True,
) -> list[_Preparation]:
    preparations: list[_Preparation] = []
    segmentation = config.segmentation.production_config()
    for spec in config.cases:
        try:
            prepared = _prepare_case(
                spec,
                project_root=project_root,
                segmentation=segmentation,
                bind_reference=bind_references,
            )
        except Exception as error:
            preparations.append(
                _Preparation(
                    spec=spec,
                    prepared=None,
                    public_result={
                        "case_id": spec.case_id,
                        "paper_id": spec.paper_id,
                        "page": spec.page,
                        "status": "error",
                        "error": _safe_error("input_preparation", error),
                    },
                )
            )
            continue
        preparations.append(
            _Preparation(
                spec=spec,
                prepared=prepared,
                public_result={
                    "case_id": spec.case_id,
                    "paper_id": spec.paper_id,
                    "page": spec.page,
                    "status": "success",
                    "manifest_path": spec.manifest_path,
                    "manifest_sha256": prepared.manifest_sha256,
                    "source_id": prepared.source_id,
                    "source_sha256": prepared.source_sha256,
                    "layout": {
                        "parser": prepared.layout.parser,
                        "parser_version": prepared.layout.parser_version,
                        "page_text_sha256": next(
                            page.text_sha256
                            for page in prepared.layout.pages
                            if page.page == spec.page
                        ),
                    },
                    "blocks": [
                        {
                            "block_id": block.block_id,
                            "text_sha256": block.text_sha256,
                            "body_lines": [block.body_start_line, block.body_end_line],
                            "context_lines": (
                                [block.context_start_line, block.context_end_line]
                                if block.context_start_line is not None
                                else None
                            ),
                        }
                        for block in prepared.blocks
                    ],
                },
            )
        )
        if bind_references:
            preparations[-1].public_result.update(
                {
                    "reference_path": spec.reference_path,
                    "reference_sha256": prepared.reference_sha256,
                }
            )
    return preparations


def _public_telemetry(call: ProviderCall) -> dict[str, Any]:
    return public_provider_call(call)


def _json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _request_binding(
    *,
    prepared: _PreparedCase,
    model: BakeoffModelSpec,
    config: ExtractorBakeoffConfig,
    block: ResultBlock,
) -> dict[str, Any]:
    """Build the exact secret-free request fingerprint before dispatch."""

    fragment = _fragment_for_block(block)
    user = page_prompt(
        paper_title=prepared.manifest.title,
        paper_id=prepared.spec.paper_id,
        fragment=fragment,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    prompt_sha256 = hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    schema = provider_json_schema()
    require_parameters = bool(config.require_parameters)
    request_contract = structured_request_contract(
        schema_name=EXTRACTOR_SCHEMA_NAME,
        schema=schema,
        seed=config.seed,
        require_parameters=require_parameters,
        model=model.model,
        max_tokens=config.max_tokens,
    )
    binding: dict[str, Any] = {
        "model_requested": model.model,
        "provider_requested": "openrouter",
        "prompt_sha256": prompt_sha256,
        "system_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
        "local_schema_sha256": _json_sha256(schema),
        "provider_schema_sha256": request_contract["schema"]["schema_sha256"],
        "request_contract": request_contract,
        "request_contract_sha256": _json_sha256(request_contract),
        "settings": {
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
            "require_parameters": require_parameters,
        },
    }
    binding["request_sha256"] = _json_sha256(binding)
    return binding


def _contract_assessment(call: ProviderCall, requested: dict[str, Any]) -> dict[str, Any]:
    """Compare completed telemetry with the predeclared request contract."""

    failures: list[str] = []
    settings = requested["settings"]
    comparisons = (
        ("provider_requested", call.provider, requested["provider_requested"]),
        ("model_requested", call.model_requested, requested["model_requested"]),
        ("model_returned", call.model_returned, requested["model_requested"]),
        ("prompt_sha256", call.prompt_sha256, requested["prompt_sha256"]),
        ("temperature", call.temperature, settings["temperature"]),
        ("reasoning_effort", call.reasoning_effort, settings["reasoning_effort"]),
        ("max_tokens", call.max_tokens, settings["max_tokens"]),
        ("seed", call.seed, settings["seed"]),
        ("require_parameters", call.require_parameters, settings["require_parameters"]),
    )
    for name, actual, expected in comparisons:
        if actual != expected:
            failures.append(f"{name}_mismatch")
    actual_contract = structured_request_contract_from_call(call)
    if _json_sha256(actual_contract) != requested["request_contract_sha256"]:
        failures.append("request_contract_mismatch")
    return {
        "status": "satisfied" if not failures else "failed",
        "failure_codes": failures,
        "requested_contract_sha256": requested["request_contract_sha256"],
        "observed_contract_sha256": _json_sha256(actual_contract),
    }


@dataclass(frozen=True, slots=True)
class _ConfiguredClient:
    """Override the production helper's legacy routing flag for this experiment."""

    client: Any
    require_parameters: bool

    def structured_chat(self, **kwargs: Any) -> Any:
        forwarded = dict(kwargs)
        forwarded["require_parameters"] = self.require_parameters
        return self.client.structured_chat(**forwarded)


def _candidate_fingerprint(candidate: Any) -> str:
    """Hash semantic output without response-specific payload metadata."""

    payload = candidate.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=False,
        exclude={"raw_payload_hash", "extraction_method"},
    )
    return _json_sha256(payload)


def _run_case_repetition(
    *,
    prepared: _PreparedCase,
    model: BakeoffModelSpec,
    config: ExtractorBakeoffConfig,
    client: Any,
    repetition_index: int,
) -> _ExtractionRun:
    candidates = []
    call_results: list[dict[str, Any]] = []
    warning_count = 0
    for block in prepared.blocks:
        requested = _request_binding(
            prepared=prepared,
            model=model,
            config=config,
            block=block,
        )
        logical_call_id = (
            f"{model.model}:{prepared.spec.case_id}:{block.block_id}:rep-{repetition_index}"
        )
        try:
            proposed, call, warnings = extract_page_candidates(
                client=client,
                model=model.model,
                paper_id=prepared.spec.paper_id,
                paper_title=prepared.manifest.title,
                fragment=_fragment_for_block(block),
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                reasoning_effort=config.reasoning_effort,
                seed=config.seed,
            )
        except ProviderResponseValidationError as error:
            assessment = _contract_assessment(error.call, requested)
            call_results.append(
                {
                    "logical_call_id": logical_call_id,
                    "repetition_index": repetition_index,
                    "block_id": block.block_id,
                    "page": block.page,
                    "status": ("contract_failure" if assessment["status"] == "failed" else "error"),
                    "schema_status": "invalid",
                    "error": _safe_error("extractor_call", error),
                    "requested": requested,
                    "contract": assessment,
                    "telemetry": _public_telemetry(error.call),
                }
            )
            continue
        except ProviderBudgetError:
            raise
        except Exception as error:
            call_results.append(
                {
                    "logical_call_id": logical_call_id,
                    "repetition_index": repetition_index,
                    "block_id": block.block_id,
                    "page": block.page,
                    "status": "error",
                    "schema_status": (
                        "invalid" if type(error).__name__ in _SCHEMA_ERROR_TYPES else "not_observed"
                    ),
                    "error": _safe_error("extractor_call", error),
                    "requested": requested,
                    "contract": {
                        "status": "not_observed",
                        "failure_codes": [],
                        "requested_contract_sha256": requested["request_contract_sha256"],
                        "observed_contract_sha256": None,
                    },
                }
            )
            continue
        assessment = _contract_assessment(call, requested)
        warning_count += len(warnings)
        contract_satisfied = assessment["status"] == "satisfied"
        if contract_satisfied:
            candidates.extend(proposed)
        call_results.append(
            {
                "logical_call_id": logical_call_id,
                "repetition_index": repetition_index,
                "block_id": block.block_id,
                "page": block.page,
                "status": "success" if contract_satisfied else "contract_failure",
                "schema_status": "valid",
                "requested": requested,
                "contract": assessment,
                "telemetry": _public_telemetry(call),
            }
        )

    successful_calls = sum(item["status"] == "success" for item in call_results)
    contract_failures = sum(item["status"] == "contract_failure" for item in call_results)
    if successful_calls == len(call_results):
        status = "success"
    elif successful_calls:
        status = "partial_failure"
    elif contract_failures:
        status = "contract_failure"
    else:
        status = "error"
    result: dict[str, Any] = {
        "case_id": prepared.spec.case_id,
        "paper_id": prepared.spec.paper_id,
        "page": prepared.spec.page,
        "repetition_index": repetition_index,
        "status": status,
        "calls": call_results,
        "warning_count": warning_count,
        "contract_failures": contract_failures,
    }
    try:
        candidates = validate_candidates(
            candidates,
            {prepared.layout.source_id: prepared.layout},
            min_confidence=config.min_confidence,
        )
        candidates = deduplicate_candidates(
            candidates,
            {prepared.layout.source_id: prepared.layout},
        )
        candidates = validate_candidates(
            candidates,
            {prepared.layout.source_id: prepared.layout},
            min_confidence=config.min_confidence,
        )
    except Exception as error:
        result["status"] = "error"
        result["error"] = _safe_error("candidate_validation", error)
        return _ExtractionRun(
            public_result=result,
            candidates=(),
            candidate_fingerprints=(),
            scoring_eligible=False,
        )
    fingerprints = tuple(sorted(_candidate_fingerprint(candidate) for candidate in candidates))
    result["candidate_count"] = len(candidates)
    result["candidate_set_sha256"] = _json_sha256(list(fingerprints))
    result["candidate_validation"] = {
        "primary_results": sum(
            candidate.claim_type == ClaimType.PRIMARY_RESULT for candidate in candidates
        ),
        "text_support": {
            status: sum(candidate.text_support.value == status for candidate in candidates)
            for status in ("supported", "partially_supported", "unsupported", "unverified")
        },
    }
    # Quality is observed only when every predeclared block produced a locally
    # valid structured response under the exact request contract.  Treating a
    # request rejection or schema failure as an empty prediction would impute a
    # zero quality score for a model whose quality was never observed.
    scoring_eligible = bool(call_results) and successful_calls == len(call_results)
    result["scoring_eligibility"] = "eligible" if scoring_eligible else "contract_ineligible"
    return _ExtractionRun(
        public_result=result,
        candidates=tuple(candidates),
        candidate_fingerprints=fingerprints,
        scoring_eligible=scoring_eligible,
    )


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _run_reference_examination(
    reference: PaperReference,
    prepared: _PreparedCase,
    run: _ExtractionRun,
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Bind control and positive-target examination to successful provider blocks."""

    successful_block_ids = {
        str(call["block_id"])
        for call in run.public_result.get("calls", [])
        if call.get("status") == "success"
    }
    successful_blocks = [
        block for block in prepared.blocks if block.block_id in successful_block_ids
    ]
    return (
        control_examination(reference, prepared.layout, successful_blocks),
        observation_examination(reference, prepared.layout, successful_blocks),
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _optional_mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 6)


def _aggregate_quality(scores: list[dict[str, Any]]) -> dict[str, Any]:
    defined_detection = {
        metric: [
            float(score["detection"][metric])
            for score in scores
            if score["detection"][metric] is not None
            and (
                metric not in {"recall", "f1"} or int(score["detection"].get("recall_basis", 0)) > 0
            )
        ]
        for metric in ("precision", "recall", "f1")
    }
    macro_detection = {
        metric: _optional_mean(defined_detection[metric])
        for metric in ("precision", "recall", "f1")
    }
    macro_detection["defined_cases"] = {
        metric: len(defined_detection[metric]) for metric in ("precision", "recall", "f1")
    }
    macro_detection["undefined_cases"] = {
        metric: len(scores) - len(defined_detection[metric])
        for metric in ("precision", "recall", "f1")
    }
    field_scores = [score for score in scores if int(score.get("field_matching_basis", 0)) > 0]
    macro_fields = {
        field: _optional_mean([float(score["field_accuracy"][field]) for score in field_scores])
        for field in QUALITY_FIELDS
    }
    reference_observations = sum(int(score["reference_observations"]) for score in scores)
    true_positives = sum(int(score["detection"]["true_positives"]) for score in scores)
    precision_true_positives = sum(
        int(score["detection"].get("precision_true_positives", 0)) for score in scores
    )
    precision_basis = sum(int(score["detection"].get("precision_basis", 0)) for score in scores)
    recall_basis = sum(
        int(score["detection"].get("recall_basis", score["reference_observations"]))
        for score in scores
    )
    false_positives = sum(int(score["detection"]["false_positives"]) for score in scores)
    false_negatives = sum(int(score["detection"]["false_negatives"]) for score in scores)
    micro_precision = _optional_ratio(precision_true_positives, precision_basis)
    micro_recall = _optional_ratio(true_positives, recall_basis)
    micro_fields = {
        field: _optional_ratio(
            sum(
                float(score["field_accuracy"][field]) * int(score["reference_observations"])
                for score in scores
            ),
            reference_observations,
        )
        for field in QUALITY_FIELDS
    }
    micro_f1 = None
    if micro_precision is not None and micro_recall is not None:
        micro_f1 = (
            _ratio(
                2.0 * micro_precision * micro_recall,
                micro_precision + micro_recall,
            )
            if micro_precision + micro_recall
            else 0.0
        )
    observability = [
        score.get("input_observability", {})
        for score in scores
        if score.get("input_observability", {}).get("status") == "measured"
    ]
    observability_complete = bool(scores) and len(observability) == len(scores)
    observability_partial = bool(observability) and not observability_complete
    observable_basis = sum(
        int(item.get("observable_reference_observations") or 0) for item in observability
    )
    unobservable_basis = sum(
        int(item.get("unobservable_reference_observations") or 0) for item in observability
    )
    observable_true_positives = sum(
        int(item.get("model_conditional_detection", {}).get("true_positives") or 0)
        for item in observability
    )
    observable_false_negatives = sum(
        int(item.get("model_conditional_detection", {}).get("false_negatives") or 0)
        for item in observability
    )
    observable_total = observable_basis + unobservable_basis
    return {
        "scored_cases": len(scores),
        "macro": {"detection": macro_detection, "field_accuracy": macro_fields},
        "micro": {
            "reference_observations": reference_observations,
            "detection": {
                "true_positives": true_positives,
                "precision_true_positives": precision_true_positives,
                "precision_basis": precision_basis,
                "recall_basis": recall_basis,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": micro_precision,
                "precision_defined": precision_basis > 0,
                "recall": micro_recall,
                "f1": micro_f1,
            },
            "field_accuracy": micro_fields,
            "input_observability": {
                "status": (
                    "measured"
                    if observability_complete
                    else "partially_assessed"
                    if observability_partial
                    else "not_assessed"
                ),
                "cases_measured": len(observability),
                "cases_not_assessed": len(scores) - len(observability),
                "measured_reference_observations": observable_total,
                "reference_observations": (observable_total if observability_complete else None),
                "observable_reference_observations": (
                    observable_basis if observability_complete else None
                ),
                "unobservable_reference_observations": (
                    unobservable_basis if observability_complete else None
                ),
                "observation_coverage": (
                    _optional_ratio(observable_basis, observable_total)
                    if observability_complete
                    else None
                ),
                "model_conditional_detection": {
                    "true_positives": (
                        observable_true_positives if observability_complete else None
                    ),
                    "false_negatives": (
                        observable_false_negatives if observability_complete else None
                    ),
                    "recall_basis": observable_basis if observability_complete else None,
                    "recall": (
                        _optional_ratio(observable_true_positives, observable_basis)
                        if observability_complete
                        else None
                    ),
                },
            },
        },
    }


def _aggregate_usage(call_results: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [item for item in call_results if "telemetry" in item]
    usage: dict[str, Any] = {}
    for field in (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cost_usd",
    ):
        values = [
            item["telemetry"][field] for item in observed if item["telemetry"][field] is not None
        ]
        usage[field] = {
            "total": round(sum(values), 8),
            "reported_calls": len(values),
            "missing_calls": len(call_results) - len(values),
        }
    latencies = [float(item["telemetry"]["latency_seconds"]) for item in observed]
    usage["latency_seconds"] = {
        "total": round(sum(latencies), 6),
        "mean": _mean(latencies),
        "p50": _percentile(latencies, 0.5),
        "p95": _percentile(latencies, 0.95),
        "max": round(max(latencies), 6) if latencies else 0.0,
        "reported_calls": len(latencies),
        "missing_calls": len(call_results) - len(latencies),
    }
    return usage


def _rate_record(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": _optional_ratio(numerator, denominator),
    }


def _schema_report(calls: list[dict[str, Any]]) -> dict[str, Any]:
    valid = sum(call.get("schema_status") == "valid" for call in calls)
    invalid = sum(call.get("schema_status") == "invalid" for call in calls)
    not_observed = len(calls) - valid - invalid
    observed = valid + invalid
    return {
        "calls_in_end_to_end_denominator": len(calls),
        "structured_responses_observed": observed,
        "structured_responses_valid": valid,
        "structured_responses_invalid": invalid,
        "structured_response_not_observed": not_observed,
        "valid_rate_of_observed_responses": _optional_ratio(valid, observed),
        "end_to_end_schema_success_rate": _optional_ratio(valid, len(calls)),
        "valid_of_observed": _rate_record(valid, observed),
        "valid_end_to_end": _rate_record(valid, len(calls)),
    }


def _contract_report(calls: list[dict[str, Any]]) -> dict[str, Any]:
    satisfied = sum(call.get("contract", {}).get("status") == "satisfied" for call in calls)
    failed = sum(call.get("contract", {}).get("status") == "failed" for call in calls)
    not_observed = len(calls) - satisfied - failed
    return {
        "calls_in_denominator": len(calls),
        "satisfied": satisfied,
        "failed": failed,
        "not_observed": not_observed,
        "satisfaction_rate": _rate_record(satisfied, len(calls)),
    }


def _case_stability(runs: list[_ExtractionRun], repetitions_predeclared: int) -> dict[str, Any]:
    comparable = [
        run for run in runs if run.scoring_eligible and "candidate_set_sha256" in run.public_result
    ]
    pairwise: list[dict[str, Any]] = []
    exact_matches = 0
    for left, right in combinations(comparable, 2):
        left_set = set(left.candidate_fingerprints)
        right_set = set(right.candidate_fingerprints)
        union = left_set | right_set
        intersection = left_set & right_set
        exact = left_set == right_set
        exact_matches += int(exact)
        pairwise.append(
            {
                "left_repetition": left.public_result["repetition_index"],
                "right_repetition": right.public_result["repetition_index"],
                "candidate_sets_exact_match": exact,
                "candidate_set_jaccard": (
                    round(len(intersection) / len(union), 6) if union else 1.0
                ),
            }
        )
    planned_pairs = repetitions_predeclared * (repetitions_predeclared - 1) // 2
    comparable_pairs = len(pairwise)
    return {
        "repetitions_predeclared": repetitions_predeclared,
        "repetitions_completed": len(runs),
        "repetitions_comparable": len(comparable),
        "repetitions_contract_ineligible": sum(not run.scoring_eligible for run in runs),
        "planned_pair_denominator": planned_pairs,
        "comparable_pair_denominator": comparable_pairs,
        "exact_candidate_set_matches": exact_matches,
        "exact_candidate_set_match_rate": _rate_record(exact_matches, comparable_pairs),
        "per_repetition": [
            {
                "repetition_index": run.public_result["repetition_index"],
                "status": run.public_result["status"],
                "scoring_eligibility": run.public_result.get("scoring_eligibility"),
                "calls": len(run.public_result.get("calls", [])),
                "candidate_count": run.public_result.get("candidate_count"),
                "candidate_set_sha256": run.public_result.get("candidate_set_sha256"),
                "schema": _schema_report(run.public_result.get("calls", [])),
                "contract": _contract_report(run.public_result.get("calls", [])),
                "score_observed": "score" in run.public_result,
            }
            for run in runs
        ],
        "pairwise": pairwise,
    }


def _combine_case_runs(
    preparation: _Preparation,
    runs: list[_ExtractionRun],
    *,
    repetitions_predeclared: int,
) -> dict[str, Any]:
    if preparation.prepared is None:
        return {
            "case_id": preparation.spec.case_id,
            "paper_id": preparation.spec.paper_id,
            "page": preparation.spec.page,
            "status": "error",
            "calls": [],
            "calls_predeclared": 0,
            "repetitions_predeclared": repetitions_predeclared,
            "repetitions": [],
            "error": preparation.public_result["error"],
            "stability": _case_stability([], repetitions_predeclared),
        }
    repetitions = [run.public_result for run in runs]
    calls = [call for repetition in repetitions for call in repetition.get("calls", [])]
    successful = sum(repetition["status"] == "success" for repetition in repetitions)
    contract_ineligible = sum(
        repetition.get("scoring_eligibility") == "contract_ineligible" for repetition in repetitions
    )
    scoring_errors = sum(
        repetition.get("scoring", {}).get("status") == "error" for repetition in repetitions
    )
    if scoring_errors:
        status = "error"
    elif successful == len(repetitions):
        status = "success"
    elif successful:
        status = "partial_failure"
    elif contract_ineligible:
        status = "contract_failure"
    else:
        status = "error"
    result: dict[str, Any] = {
        "case_id": preparation.spec.case_id,
        "paper_id": preparation.spec.paper_id,
        "page": preparation.spec.page,
        "status": status,
        "calls_predeclared": len(preparation.prepared.blocks) * repetitions_predeclared,
        "calls": calls,
        "repetitions_predeclared": repetitions_predeclared,
        "repetitions": repetitions,
        "stability": _case_stability(runs, repetitions_predeclared),
    }
    if repetitions and "reference_scope" in repetitions[0]:
        result["reference_scope"] = repetitions[0]["reference_scope"]
    # Preserve the historical one-repetition conveniences without pretending that
    # repeated scores or candidate counts can be merged into one case output.
    if repetitions_predeclared == 1 and repetitions:
        for key in ("candidate_count", "candidate_validation", "score"):
            if key in repetitions[0]:
                result[key] = repetitions[0][key]
    return result


def _aggregate_model(cases: list[dict[str, Any]]) -> dict[str, Any]:
    repetitions = [repetition for case in cases for repetition in case.get("repetitions", [])]
    calls = [call for repetition in repetitions for call in repetition.get("calls", [])]
    successful_calls = sum(call["status"] == "success" for call in calls)
    successful_cases = sum(case["status"] == "success" for case in cases)
    partial_cases = sum(case["status"] == "partial_failure" for case in cases)
    successful_repetitions = sum(repetition["status"] == "success" for repetition in repetitions)
    partial_repetitions = sum(
        repetition["status"] == "partial_failure" for repetition in repetitions
    )
    scores = [repetition["score"] for repetition in repetitions if "score" in repetition]
    scored_cases = sum(
        any("score" in item for item in case.get("repetitions", [])) for case in cases
    )
    quality = _aggregate_quality(scores) if scores else None
    if quality is not None:
        quality["scored_cases"] = scored_cases
        quality["scored_repetitions"] = len(scores)
    reference_evaluation = aggregate_reference_scores(scores) if scores else None
    schema = _schema_report(calls)
    first_pass_calls = [call for call in calls if call.get("repetition_index") == 1]
    support_counts = {
        status: sum(
            repetition.get("candidate_validation", {}).get("text_support", {}).get(status, 0)
            for repetition in repetitions
        )
        for status in ("supported", "partially_supported", "unsupported", "unverified")
    }
    calls_predeclared = sum(int(case.get("calls_predeclared", 0)) for case in cases)
    comparable_pairs = sum(
        int(case.get("stability", {}).get("comparable_pair_denominator", 0)) for case in cases
    )
    exact_pairs = sum(
        int(case.get("stability", {}).get("exact_candidate_set_matches", 0)) for case in cases
    )
    return {
        "execution": {
            "cases_attempted": len(cases),
            "cases_succeeded": successful_cases,
            "cases_partial_failure": partial_cases,
            "cases_failed": len(cases) - successful_cases - partial_cases,
            "case_success_rate": _ratio(successful_cases, len(cases)),
            "cases_scored": scored_cases,
            "case_scored_rate": _ratio(scored_cases, len(cases)),
            "repetitions_predeclared": sum(
                int(case.get("repetitions_predeclared", 0)) for case in cases
            ),
            "repetitions_attempted": len(repetitions),
            "repetitions_succeeded": successful_repetitions,
            "repetitions_partial_failure": partial_repetitions,
            "repetitions_failed": (len(repetitions) - successful_repetitions - partial_repetitions),
            "repetitions_scored": len(scores),
            "calls_predeclared": calls_predeclared,
            "calls_attempted": len(calls),
            "calls_succeeded": successful_calls,
            "calls_failed": len(calls) - successful_calls,
            "call_success_rate": _ratio(successful_calls, len(calls)),
            "call_success": _rate_record(successful_calls, len(calls)),
        },
        "schema": schema,
        "first_pass_wire_schema": {
            "repetition_index": 1,
            **_schema_report(first_pass_calls),
        },
        "contract": _contract_report(calls),
        "stability": {
            "cases_with_repetition_data": sum(bool(case.get("repetitions")) for case in cases),
            "comparable_pair_denominator": comparable_pairs,
            "exact_candidate_set_matches": exact_pairs,
            "exact_candidate_set_match_rate": _rate_record(exact_pairs, comparable_pairs),
        },
        "usage": _aggregate_usage(calls),
        "evidence": {
            "candidate_text_support": support_counts,
            "reference_evidence_supported_accuracy": {
                "macro": (
                    quality["macro"]["field_accuracy"]["evidence_supported"]
                    if quality is not None
                    else None
                ),
                "micro": (
                    quality["micro"]["field_accuracy"]["evidence_supported"]
                    if quality is not None
                    else None
                ),
            },
            "reference_page_anchor_accuracy": {
                "macro": (
                    quality["macro"]["field_accuracy"]["page"] if quality is not None else None
                ),
                "micro": (
                    quality["micro"]["field_accuracy"]["page"] if quality is not None else None
                ),
            },
        },
        "negative_control_safety": (
            reference_evaluation["negative_control_safety"]
            if reference_evaluation is not None
            else None
        ),
        "claim_type_classification": (
            reference_evaluation["claim_type_classification"]
            if reference_evaluation is not None
            else None
        ),
        "model_selection_gates": (
            {
                key: reference_evaluation["quality_gates"][key]
                for key in (
                    "claim_type_macro_f1",
                    "false_primary_controls",
                    "false_primary_exports",
                )
            }
            if reference_evaluation is not None
            else None
        ),
        "quality_measurement": {
            "status": (
                "measured"
                if len(scores) == len(repetitions) and scores
                else "partially_measured"
                if scores
                else "unmeasured"
            ),
            "scored_repetitions": len(scores),
            "unmeasured_repetitions": len(repetitions) - len(scores),
            "reason": None if scores else "no_fully_successful_contract_eligible_repetition",
        },
        "quality": quality,
    }


def run_extractor_bakeoff(
    config: ExtractorBakeoffConfig,
    *,
    project_root: Path,
    client: OpenRouterClient,
    output_path: Path | None = None,
    code_root: Path | None = None,
) -> dict[str, Any]:
    """Run all fresh calls first, then join references for offline scoring."""

    preparations = _prepare_cases(config, project_root)
    configured_client = _ConfiguredClient(
        client=client,
        require_parameters=bool(config.require_parameters),
    )
    executions: list[tuple[BakeoffModelSpec, list[tuple[_Preparation, list[_ExtractionRun]]]]] = []
    for model in config.models:
        case_runs: list[tuple[_Preparation, list[_ExtractionRun]]] = []
        for preparation in preparations:
            if preparation.prepared is None:
                case_runs.append((preparation, []))
                continue
            runs = [
                _run_case_repetition(
                    prepared=preparation.prepared,
                    model=model,
                    config=config,
                    client=configured_client,
                    repetition_index=repetition_index,
                )
                for repetition_index in range(1, config.fresh_repetitions + 1)
            ]
            case_runs.append((preparation, runs))
        executions.append((model, case_runs))

    # This is the only point at which reference annotations enter computation. Every
    # declared provider call across every model has already returned or failed.
    scoring_references: dict[
        str,
        tuple[PaperReference, dict[str, int | float | None]] | dict[str, str],
    ] = {}
    for preparation in preparations:
        if preparation.prepared is None:
            continue
        try:
            reference, scope = _load_scoring_reference(preparation.prepared)
        except Exception as error:
            scoring_references[preparation.spec.case_id] = _safe_error(
                "reference_scoring_input", error
            )
            preparation.public_result["reference_scoring"] = {
                "status": "error",
                "error": scoring_references[preparation.spec.case_id],
            }
            continue
        scoring_references[preparation.spec.case_id] = (reference, scope)
        preparation.public_result["reference_scope"] = scope
        preparation.public_result["reference_scoring"] = {"status": "ready"}

    model_results: list[dict[str, Any]] = []
    for model, case_runs in executions:
        cases: list[dict[str, Any]] = []
        for preparation, runs in case_runs:
            scoring_reference = scoring_references.get(preparation.spec.case_id)
            for run in runs:
                if isinstance(scoring_reference, dict):
                    run.public_result["scoring"] = {
                        "status": "error",
                        "error": scoring_reference,
                    }
                    continue
                if scoring_reference is None:
                    run.public_result["scoring"] = {
                        "status": "not_available",
                        "reason": "input_preparation_failed",
                    }
                    continue
                reference, scope = scoring_reference
                run.public_result["reference_scope"] = scope
                if not run.scoring_eligible:
                    run.public_result["scoring"] = {
                        "status": "not_scored",
                        "reason": "request_contract_ineligible",
                    }
                    continue
                try:
                    assert preparation.prepared is not None
                    control_examined, observations_examined = _run_reference_examination(
                        reference,
                        preparation.prepared,
                        run,
                    )
                    run.public_result["score"] = score_reference(
                        reference,
                        list(run.candidates),
                        control_examined,
                        observations_examined,
                    )
                except Exception as error:
                    run.public_result["scoring"] = {
                        "status": "error",
                        "error": _safe_error("reference_scoring", error),
                    }
                    continue
                run.public_result["scoring"] = {"status": "scored"}
            cases.append(
                _combine_case_runs(
                    preparation,
                    runs,
                    repetitions_predeclared=config.fresh_repetitions,
                )
            )
        model_results.append(
            {
                "model": model.model,
                "label": model.label,
                "aggregate": _aggregate_model(cases),
                "cases": cases,
            }
        )

    request_contract = structured_request_contract(
        schema_name=EXTRACTOR_SCHEMA_NAME,
        schema=provider_json_schema(),
        seed=config.seed,
        require_parameters=bool(config.require_parameters),
    )
    canonical_config = config.model_dump(mode="json", by_alias=True, exclude_none=False)

    result = {
        "schema_version": "extractor-bakeoff-result/0.3",
        "bakeoff_id": config.bakeoff_id,
        "configuration_schema_version": config.schema_version,
        "configuration_sha256": _json_sha256(canonical_config),
        "configuration_binding": {
            "models_sha256": _json_sha256(canonical_config["models"]),
            "cases_sha256": _json_sha256(canonical_config["cases"]),
            "segmentation_sha256": _json_sha256(canonical_config["segmentation"]),
            "requested_settings_sha256": _json_sha256(
                {
                    key: canonical_config[key]
                    for key in (
                        "temperature",
                        "reasoning_effort",
                        "max_tokens",
                        "seed",
                        "require_parameters",
                        "fresh_repetitions",
                    )
                }
            ),
        },
        "determinism": {
            "seed": config.seed,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "require_parameters": bool(config.require_parameters),
            "fresh_repetitions": config.fresh_repetitions,
            "max_tokens": config.max_tokens,
            "min_confidence": config.min_confidence,
            "prompt_sha256": prompt_hash(),
            "segmentation": asdict(config.segmentation.production_config()),
            "reference_prompt_isolation": True,
            "reference_join_phase": "after_all_provider_calls",
        },
        "request_contract": request_contract,
        "request_contract_sha256": _json_sha256(request_contract),
        "wire_schema_sha256": _json_sha256(provider_json_schema()),
        "code": _code_state(code_root if code_root is not None else project_root),
        "inputs": [preparation.public_result for preparation in preparations],
        "models": model_results,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def run_extractor_bakeoff_file(
    config_path: Path,
    *,
    project_root: Path,
    client: OpenRouterClient,
    output_path: Path | None = None,
    code_root: Path | None = None,
) -> dict[str, Any]:
    """Load and run one YAML bake-off definition."""

    return run_extractor_bakeoff(
        load_bakeoff_config(config_path),
        project_root=project_root,
        client=client,
        output_path=output_path,
        code_root=code_root,
    )


# The split provider/scoring API below is the scientific path for v0.2
# experiments.  The combined API above remains only for compatibility with
# historical callers and fixtures.
_EXTRACTOR_PROVIDER_ENTRY_VERSION = "extractor-bakeoff-provider-entry/0.3"
_EXTRACTOR_PROVIDER_CHECKPOINT_VERSION = "extractor-bakeoff-provider-checkpoint/0.3"
_EXTRACTOR_PROVIDER_RESULT_VERSION = "extractor-bakeoff-provider-result/0.2"
_EXTRACTOR_SCORE_VERSION = "extractor-bakeoff-score/0.3"


def _extractor_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, dict):
        return {key: _extractor_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_extractor_jsonable(item) for item in value]
    return value


def _extractor_hash(value: Any) -> str:
    return _json_sha256(_extractor_jsonable(value))


def _extractor_candidates_bound_to_call(
    candidates: list[CandidateObservation],
    call: ProviderCall | None,
) -> bool:
    """Match the production extractor's response and proposal-lineage binding."""

    return not candidates or (
        call is not None
        and all(
            candidate.raw_payload_hash == call.response_sha256 and bool(candidate.proposal_traces)
            for candidate in candidates
        )
    )


class _ExtractorProviderEntry(StrictModel):
    schema_version: Literal["extractor-bakeoff-provider-entry/0.3"] = (
        _EXTRACTOR_PROVIDER_ENTRY_VERSION
    )
    slot_id: str = Field(min_length=1)
    slot_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    repetition_index: int = Field(ge=1)
    block_id: str = Field(min_length=1)
    request: dict[str, Any]
    status: Literal[
        "success",
        "contract_failure",
        "response_validation_failure",
        "provider_failure",
    ]
    schema_status: Literal["valid", "invalid", "not_observed"]
    contract_status: Literal["satisfied", "failed", "not_observed"]
    contract_failure_codes: list[str] = Field(default_factory=list)
    candidates: list[CandidateObservation] = Field(default_factory=list)
    candidate_fingerprints: list[str] = Field(default_factory=list)
    warning_count: int = Field(default=0, ge=0)
    call: ProviderCall | None = None
    error: dict[str, str] | None = None
    entry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_entry_is_self_consistent(self) -> _ExtractorProviderEntry:
        request_without_hash = {
            key: value for key, value in self.request.items() if key != "request_sha256"
        }
        if self.request.get("request_sha256") != _json_sha256(request_without_hash):
            raise ValueError("extractor checkpoint request hash is invalid")
        payload = self.model_dump(mode="json", exclude={"entry_sha256"}, exclude_none=False)
        if self.entry_sha256 != _extractor_hash(payload):
            raise ValueError("extractor checkpoint entry hash is invalid")
        expected_fingerprints = sorted(_candidate_fingerprint(item) for item in self.candidates)
        if self.candidate_fingerprints != expected_fingerprints:
            raise ValueError("extractor checkpoint candidate fingerprints are invalid")
        if self.status in {"success", "contract_failure"} and (
            self.call is None or self.schema_status != "valid"
        ):
            raise ValueError("completed extractor response is incomplete")
        if self.status == "success" and self.contract_status != "satisfied":
            raise ValueError("successful extractor entry must satisfy its request contract")
        if self.status == "contract_failure" and self.contract_status != "failed":
            raise ValueError("extractor contract failure must record a failed contract")
        if self.status == "response_validation_failure" and (
            self.call is None or self.schema_status != "invalid"
        ):
            raise ValueError("extractor response-validation failure is malformed")
        if self.status == "provider_failure" and (
            self.call is not None or self.schema_status != "not_observed"
        ):
            raise ValueError("extractor provider failure is malformed")
        if self.status != "success" and self.candidates:
            raise ValueError("ineligible extractor entries cannot retain usable candidates")
        if self.status == "success" and not _extractor_candidates_bound_to_call(
            self.candidates,
            self.call,
        ):
            raise ValueError("extractor candidate is not bound to its provider response")
        return self


class _ExtractorProviderCheckpoint(StrictModel):
    schema_version: Literal["extractor-bakeoff-provider-checkpoint/0.3"] = (
        _EXTRACTOR_PROVIDER_CHECKPOINT_VERSION
    )
    bakeoff_id: str = Field(min_length=1)
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["in_progress", "sealed"]
    entries: dict[str, _ExtractorProviderEntry]
    provider_phase_seal_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_checkpoint_is_self_consistent(self) -> _ExtractorProviderCheckpoint:
        if any(slot_id != entry.slot_id for slot_id, entry in self.entries.items()):
            raise ValueError("extractor checkpoint entry key does not match slot")
        expected_seal = _extractor_provider_phase_seal(
            self.configuration_sha256,
            self.run_contract_sha256,
            self.entries,
        )
        if self.status == "sealed":
            if self.provider_phase_seal_sha256 != expected_seal:
                raise ValueError("extractor provider-phase seal is invalid")
        elif self.provider_phase_seal_sha256 is not None:
            raise ValueError("in-progress extractor checkpoint cannot carry a seal")
        payload = self.model_dump(mode="json", exclude={"checkpoint_sha256"}, exclude_none=False)
        if self.checkpoint_sha256 != _extractor_hash(payload):
            raise ValueError("extractor checkpoint hash is invalid")
        return self


@dataclass(frozen=True, slots=True)
class _ExtractorProviderContext:
    config: ExtractorBakeoffConfig
    configuration_sha256: str
    code: dict[str, Any]
    code_sha256: str
    stage_contract_sha256: str
    execution_selection: StageExecutionSelection | None
    models: tuple[BakeoffModelSpec, ...]
    run_contract_sha256: str
    preparations: tuple[_Preparation, ...]


def _extractor_provider_phase_seal(
    configuration_sha256: str,
    run_contract_sha256: str,
    entries: dict[str, _ExtractorProviderEntry],
) -> str:
    return _json_sha256(
        {
            "schema_version": "extractor-bakeoff-provider-seal/0.1",
            "configuration_sha256": configuration_sha256,
            "run_contract_sha256": run_contract_sha256,
            "entry_sha256s": {
                slot_id: entry.entry_sha256 for slot_id, entry in sorted(entries.items())
            },
        }
    )


def _extractor_provider_checkpoint(
    context: _ExtractorProviderContext,
    entries: dict[str, _ExtractorProviderEntry],
    *,
    sealed: bool,
) -> _ExtractorProviderCheckpoint:
    payload: dict[str, Any] = {
        "schema_version": _EXTRACTOR_PROVIDER_CHECKPOINT_VERSION,
        "bakeoff_id": context.config.bakeoff_id,
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "status": "sealed" if sealed else "in_progress",
        "entries": entries,
        "provider_phase_seal_sha256": (
            _extractor_provider_phase_seal(
                context.configuration_sha256,
                context.run_contract_sha256,
                entries,
            )
            if sealed
            else None
        ),
    }
    payload["checkpoint_sha256"] = _extractor_hash(payload)
    return _ExtractorProviderCheckpoint.model_validate(payload)


def _write_extractor_provider_checkpoint(
    path: Path,
    checkpoint: _ExtractorProviderCheckpoint,
) -> None:
    write_json(path, checkpoint.model_dump(mode="json", exclude_none=False))


def extractor_stage_contract_sha256(config: ExtractorBakeoffConfig) -> str:
    """Bind the route-relevant extractor contract shared by smoke and full phases."""

    schema = provider_json_schema()
    request_contract = structured_request_contract(
        schema_name=EXTRACTOR_SCHEMA_NAME,
        schema=schema,
        seed=config.seed,
        require_parameters=bool(config.require_parameters),
    )
    return _json_sha256(
        {
            "schema_version": "extractor-bakeoff-stage-contract/0.1",
            "models": [item.model for item in config.models],
            "segmentation": asdict(config.segmentation.production_config()),
            "min_confidence": config.min_confidence,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "seed": config.seed,
            "require_parameters": bool(config.require_parameters),
            "prompt_sha256": prompt_hash(),
            "provider_schema_sha256": request_contract["schema"]["schema_sha256"],
            "request_contract_sha256": _json_sha256(request_contract),
        }
    )


def _prepare_extractor_provider_context(
    config: ExtractorBakeoffConfig,
    *,
    project_root: Path,
    code_root: Path | None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> _ExtractorProviderContext:
    validated_selection = (
        StageExecutionSelection.model_validate(
            execution_selection.model_dump(mode="json", exclude_none=False)
        )
        if execution_selection is not None
        else None
    )
    canonical_config = config.model_dump(mode="json", by_alias=True, exclude_none=False)
    configuration_sha256 = _json_sha256(canonical_config)
    code = _code_state(code_root if code_root is not None else project_root)
    code_sha256 = _json_sha256(code)
    stage_contract_sha256 = extractor_stage_contract_sha256(config)
    selected_model_ids = selected_models_for_execution(
        validated_selection,
        expected_stage="block_candidate_extraction",
        declared_models=[item.model for item in config.models],
        stage_contract_sha256=stage_contract_sha256,
        code_sha256=code_sha256,
        expectation=execution_selection_expectation,
    )
    selected = set(selected_model_ids)
    models = tuple(item for item in config.models if item.model in selected)
    if tuple(item.model for item in models) != selected_model_ids:
        raise ValueError("extractor execution selection does not preserve declared model order")
    preparations = _prepare_cases(config, project_root, bind_references=False)
    if any(item.prepared is None for item in preparations):
        raise ValueError("extractor provider input preparation failed")
    input_binding = [
        {
            key: value
            for key, value in preparation.public_result.items()
            if key not in {"manifest_path", "reference_path", "reference_sha256"}
        }
        for preparation in preparations
    ]
    request_contract = structured_request_contract(
        schema_name=EXTRACTOR_SCHEMA_NAME,
        schema=provider_json_schema(),
        seed=config.seed,
        require_parameters=bool(config.require_parameters),
    )
    run_contract_sha256 = _json_sha256(
        {
            "schema_version": "extractor-bakeoff-provider-run-contract/0.1",
            "bakeoff_id": config.bakeoff_id,
            "configuration_sha256": configuration_sha256,
            "code_sha256": code_sha256,
            "stage_contract_sha256": stage_contract_sha256,
            "execution_selection_sha256": (
                validated_selection.selection_sha256 if validated_selection is not None else None
            ),
            "executed_models": list(selected_model_ids),
            "input_binding_sha256": _json_sha256(input_binding),
            "prompt_sha256": prompt_hash(),
            "wire_schema_sha256": _json_sha256(provider_json_schema()),
            "request_contract_sha256": _json_sha256(request_contract),
        }
    )
    return _ExtractorProviderContext(
        config=config,
        configuration_sha256=configuration_sha256,
        code=code,
        code_sha256=code_sha256,
        stage_contract_sha256=stage_contract_sha256,
        execution_selection=validated_selection,
        models=models,
        run_contract_sha256=run_contract_sha256,
        preparations=tuple(preparations),
    )


def _extractor_slot_contract(
    context: _ExtractorProviderContext,
    preparation: _Preparation,
    model: BakeoffModelSpec,
    repetition_index: int,
    block: ResultBlock,
) -> tuple[str, str, dict[str, Any]]:
    assert preparation.prepared is not None
    request = _request_binding(
        prepared=preparation.prepared,
        model=model,
        config=context.config,
        block=block,
    )
    contract = {
        "schema_version": "extractor-bakeoff-provider-slot/0.1",
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "model": model.model,
        "case_id": preparation.spec.case_id,
        "repetition_index": repetition_index,
        "block_id": block.block_id,
        "block_text_sha256": block.text_sha256,
        "request_sha256": request["request_sha256"],
    }
    slot_contract_sha256 = _json_sha256(contract)
    slot_id = f"extractor-slot-{slot_contract_sha256}"
    return slot_id, slot_contract_sha256, request


def _extractor_expected_slots(
    context: _ExtractorProviderContext,
) -> list[tuple[str, str, dict[str, Any], _Preparation, BakeoffModelSpec, int, ResultBlock]]:
    slots = []
    for model in context.models:
        for preparation in context.preparations:
            assert preparation.prepared is not None
            for repetition_index in range(1, context.config.fresh_repetitions + 1):
                for block in preparation.prepared.blocks:
                    slot_id, slot_contract_sha256, request = _extractor_slot_contract(
                        context,
                        preparation,
                        model,
                        repetition_index,
                        block,
                    )
                    slots.append(
                        (
                            slot_id,
                            slot_contract_sha256,
                            request,
                            preparation,
                            model,
                            repetition_index,
                            block,
                        )
                    )
    return slots


def _extractor_entry_payload(
    *,
    slot_id: str,
    slot_contract_sha256: str,
    request: dict[str, Any],
    preparation: _Preparation,
    model: BakeoffModelSpec,
    repetition_index: int,
    block: ResultBlock,
    status: str,
    schema_status: str,
    contract_status: str,
    contract_failure_codes: list[str] | None = None,
    candidates: list[CandidateObservation] | None = None,
    warning_count: int = 0,
    call: ProviderCall | None = None,
    error: dict[str, str] | None = None,
) -> _ExtractorProviderEntry:
    retained_candidates = list(candidates or []) if status == "success" else []
    payload: dict[str, Any] = {
        "schema_version": _EXTRACTOR_PROVIDER_ENTRY_VERSION,
        "slot_id": slot_id,
        "slot_contract_sha256": slot_contract_sha256,
        "model": model.model,
        "case_id": preparation.spec.case_id,
        "repetition_index": repetition_index,
        "block_id": block.block_id,
        "request": request,
        "status": status,
        "schema_status": schema_status,
        "contract_status": contract_status,
        "contract_failure_codes": list(contract_failure_codes or []),
        "candidates": retained_candidates,
        "candidate_fingerprints": sorted(
            _candidate_fingerprint(candidate) for candidate in retained_candidates
        ),
        "warning_count": warning_count,
        "call": call,
        "error": error,
    }
    payload["entry_sha256"] = _extractor_hash(payload)
    return _ExtractorProviderEntry.model_validate(payload)


def _extractor_entry_reusable(
    entry: _ExtractorProviderEntry,
    *,
    slot_id: str,
    slot_contract_sha256: str,
    request: dict[str, Any],
    preparation: _Preparation,
    model: BakeoffModelSpec,
    repetition_index: int,
    block: ResultBlock,
) -> bool:
    if not _extractor_candidates_bound_to_call(entry.candidates, entry.call):
        return False
    if (
        entry.slot_id != slot_id
        or entry.slot_contract_sha256 != slot_contract_sha256
        or entry.request != request
        or entry.model != model.model
        or entry.case_id != preparation.spec.case_id
        or entry.repetition_index != repetition_index
        or entry.block_id != block.block_id
    ):
        return False
    if entry.call is not None:
        assessment = _contract_assessment(entry.call, request)
        if (
            entry.contract_status != assessment["status"]
            or entry.contract_failure_codes != assessment["failure_codes"]
        ):
            return False
    elif entry.contract_status != "not_observed":
        return False
    return True


def _load_extractor_provider_checkpoint(
    path: Path,
    context: _ExtractorProviderContext,
) -> _ExtractorProviderCheckpoint | None:
    if not path.exists():
        return None
    try:
        checkpoint = _ExtractorProviderCheckpoint.model_validate(read_json(path))
    except Exception as error:
        raise ValueError("extractor provider checkpoint is invalid") from error
    if checkpoint.bakeoff_id != context.config.bakeoff_id:
        raise ValueError("extractor provider checkpoint bake-off ID is stale")
    if checkpoint.configuration_sha256 != context.configuration_sha256:
        raise ValueError("extractor provider checkpoint configuration is stale")
    if checkpoint.run_contract_sha256 != context.run_contract_sha256:
        raise ValueError("extractor provider checkpoint run contract is stale")
    return checkpoint


def _validated_extractor_provider_entries(
    checkpoint: _ExtractorProviderCheckpoint,
    context: _ExtractorProviderContext,
    *,
    require_sealed: bool,
) -> tuple[list[_ExtractorProviderEntry], list[Any]]:
    if require_sealed and checkpoint.status != "sealed":
        raise ValueError("extractor provider phase checkpoint is not sealed")
    slots = _extractor_expected_slots(context)
    expected_ids = {slot[0] for slot in slots}
    if not set(checkpoint.entries).issubset(expected_ids):
        raise ValueError("extractor provider checkpoint contains unexpected slots")
    ordered: list[_ExtractorProviderEntry] = []
    for (
        slot_id,
        slot_contract_sha256,
        request,
        preparation,
        model,
        repetition_index,
        block,
    ) in slots:
        entry = checkpoint.entries.get(slot_id)
        if entry is None:
            if require_sealed:
                raise ValueError("sealed extractor provider checkpoint is incomplete")
            continue
        if not _extractor_entry_reusable(
            entry,
            slot_id=slot_id,
            slot_contract_sha256=slot_contract_sha256,
            request=request,
            preparation=preparation,
            model=model,
            repetition_index=repetition_index,
            block=block,
        ):
            raise ValueError("extractor provider checkpoint contains a stale slot")
        ordered.append(entry)
    return ordered, slots


def _public_extractor_provider_entry(entry: _ExtractorProviderEntry) -> dict[str, Any]:
    result: dict[str, Any] = {
        "slot_id": entry.slot_id,
        "slot_contract_sha256": entry.slot_contract_sha256,
        "model": entry.model,
        "case_id": entry.case_id,
        "repetition_index": entry.repetition_index,
        "block_id": entry.block_id,
        "status": entry.status,
        "schema_status": entry.schema_status,
        "contract": {
            "status": entry.contract_status,
            "failure_codes": entry.contract_failure_codes,
            "requested_contract_sha256": entry.request["request_contract_sha256"],
            "observed_contract_sha256": (
                _json_sha256(structured_request_contract_from_call(entry.call))
                if entry.call is not None
                else None
            ),
        },
        "requested": entry.request,
        "candidate_count": len(entry.candidates),
        "candidate_fingerprints": entry.candidate_fingerprints,
        "warning_count": entry.warning_count,
    }
    if entry.call is not None:
        result["telemetry"] = _public_telemetry(entry.call)
    if entry.error is not None:
        result["error"] = entry.error
    return result


def _extractor_runs_from_entries(
    context: _ExtractorProviderContext,
    entries: list[_ExtractorProviderEntry],
    *,
    model: BakeoffModelSpec,
    preparation: _Preparation,
) -> list[_ExtractionRun]:
    assert preparation.prepared is not None
    by_slot = {entry.slot_id: entry for entry in entries}
    runs: list[_ExtractionRun] = []
    for repetition_index in range(1, context.config.fresh_repetitions + 1):
        repetition_entries = []
        for block in preparation.prepared.blocks:
            slot_id, _, _ = _extractor_slot_contract(
                context,
                preparation,
                model,
                repetition_index,
                block,
            )
            repetition_entries.append(by_slot[slot_id])
        if any(
            not _extractor_candidates_bound_to_call(entry.candidates, entry.call)
            for entry in repetition_entries
        ):
            raise ValueError("extractor candidate is not bound to its provider response")
        calls = [_public_extractor_provider_entry(entry) for entry in repetition_entries]
        candidates = [candidate for entry in repetition_entries for candidate in entry.candidates]
        fully_eligible = all(entry.status == "success" for entry in repetition_entries)
        successful_calls = sum(entry.status == "success" for entry in repetition_entries)
        if fully_eligible:
            status = "success"
        elif successful_calls:
            status = "partial_failure"
        elif any(entry.status == "contract_failure" for entry in repetition_entries):
            status = "contract_failure"
        else:
            status = "error"
        public_result: dict[str, Any] = {
            "case_id": preparation.spec.case_id,
            "paper_id": preparation.spec.paper_id,
            "page": preparation.spec.page,
            "repetition_index": repetition_index,
            "status": status,
            "calls": calls,
            "warning_count": sum(entry.warning_count for entry in repetition_entries),
            "contract_failures": sum(
                entry.status == "contract_failure" for entry in repetition_entries
            ),
        }
        try:
            validated = validate_candidates(
                candidates,
                {preparation.prepared.layout.source_id: preparation.prepared.layout},
                min_confidence=context.config.min_confidence,
            )
            validated = deduplicate_candidates(
                validated,
                {preparation.prepared.layout.source_id: preparation.prepared.layout},
            )
            validated = validate_candidates(
                validated,
                {preparation.prepared.layout.source_id: preparation.prepared.layout},
                min_confidence=context.config.min_confidence,
            )
        except Exception as error:
            public_result["status"] = "error"
            public_result["error"] = _safe_error("candidate_validation", error)
            fully_eligible = False
            validated = []
        fingerprints = tuple(sorted(_candidate_fingerprint(item) for item in validated))
        public_result["candidate_count"] = len(validated)
        public_result["candidate_set_sha256"] = _json_sha256(list(fingerprints))
        public_result["candidate_validation"] = {
            "primary_results": sum(
                item.claim_type == ClaimType.PRIMARY_RESULT for item in validated
            ),
            "text_support": {
                support: sum(item.text_support.value == support for item in validated)
                for support in ("supported", "partially_supported", "unsupported", "unverified")
            },
        }
        public_result["scoring_eligibility"] = (
            "eligible" if fully_eligible else "unmeasured_provider_or_contract_failure"
        )
        runs.append(
            _ExtractionRun(
                public_result=public_result,
                candidates=tuple(validated),
                candidate_fingerprints=fingerprints,
                scoring_eligible=fully_eligible,
            )
        )
    return runs


def _extractor_provider_models(
    context: _ExtractorProviderContext,
    entries: list[_ExtractorProviderEntry],
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    executed_models = {item.model for item in context.models}
    for model in context.config.models:
        if model.model not in executed_models:
            if context.execution_selection is None:
                raise RuntimeError("extractor model was skipped without an execution selection")
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
                    "cases": [],
                }
            )
            continue
        cases = []
        for preparation in context.preparations:
            runs = _extractor_runs_from_entries(
                context,
                entries,
                model=model,
                preparation=preparation,
            )
            cases.append(
                _combine_case_runs(
                    preparation,
                    runs,
                    repetitions_predeclared=context.config.fresh_repetitions,
                )
            )
        aggregate = _aggregate_model(cases)
        aggregate.pop("quality", None)
        aggregate.pop("quality_measurement", None)
        aggregate.pop("negative_control_safety", None)
        aggregate.pop("claim_type_classification", None)
        aggregate.pop("model_selection_gates", None)
        aggregate["evidence"] = {
            "candidate_text_support": aggregate["evidence"]["candidate_text_support"]
        }
        for field in ("cases_scored", "case_scored_rate", "repetitions_scored"):
            aggregate["execution"].pop(field, None)
        models.append(
            {
                "model": model.model,
                "label": model.label,
                "contract_eligibility": (
                    context.execution_selection.decision_for(model.model).status
                    if context.execution_selection is not None
                    else "not_assessed"
                ),
                "eligibility_reason_codes": (
                    context.execution_selection.decision_for(model.model).reason_codes
                    if context.execution_selection is not None
                    else []
                ),
                "matched_quality_status": "executed",
                "quality_status": "pending_offline_scoring",
                "quality": None,
                "aggregate": aggregate,
                "cases": cases,
            }
        )
    return models


def _write_immutable_json(path: Path, value: Any) -> None:
    content = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError("existing immutable artifact has different content")
        return
    write_json(path, value)


def _assert_extractor_artifact_paths_do_not_overwrite_inputs(
    config: ExtractorBakeoffConfig,
    *,
    project_root: Path,
    artifact_paths: list[Path],
    context: _ExtractorProviderContext,
) -> None:
    root = project_root.resolve()
    protected = [
        _project_path(root, configured)
        for case in config.cases
        for configured in (case.manifest_path, case.reference_path)
    ]
    protected.extend(
        (root / source.cache_relpath).resolve(strict=False)
        for preparation in context.preparations
        if preparation.prepared is not None
        for source in preparation.prepared.manifest.sources
        if source.cache_relpath is not None
    )
    assert_artifact_paths_safe(artifact_paths, protected_paths=protected)


def _extractor_provider_result(
    context: _ExtractorProviderContext,
    checkpoint: _ExtractorProviderCheckpoint,
    entries: list[_ExtractorProviderEntry],
) -> dict[str, Any]:
    request_contract = structured_request_contract(
        schema_name=EXTRACTOR_SCHEMA_NAME,
        schema=provider_json_schema(),
        seed=context.config.seed,
        require_parameters=bool(context.config.require_parameters),
    )
    return {
        "schema_version": _EXTRACTOR_PROVIDER_RESULT_VERSION,
        "bakeoff_id": context.config.bakeoff_id,
        "status": "sealed",
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "provider_phase_seal_sha256": checkpoint.provider_phase_seal_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "request_contract": request_contract,
        "request_contract_sha256": _json_sha256(request_contract),
        "wire_schema_sha256": _json_sha256(provider_json_schema()),
        "code": context.code,
        "code_sha256": context.code_sha256,
        "stage_contract_sha256": context.stage_contract_sha256,
        "execution_selection": (
            {
                "selection_sha256": context.execution_selection.selection_sha256,
                "smoke_provider_phase_seal_sha256": (
                    context.execution_selection.smoke_provider_phase_seal_sha256
                ),
                "smoke_checkpoint_sha256": (context.execution_selection.smoke_checkpoint_sha256),
                "eligible_models": list(context.execution_selection.eligible_models),
            }
            if context.execution_selection is not None
            else None
        ),
        "declared_models": [item.model for item in context.config.models],
        "executed_models": [item.model for item in context.models],
        "inputs": [
            {
                key: value
                for key, value in preparation.public_result.items()
                if key not in {"manifest_path", "reference_path", "reference_sha256"}
            }
            for preparation in context.preparations
        ],
        "models": _extractor_provider_models(context, entries),
        "privacy": {
            "references_loaded_during_provider_phase": False,
            "reference_paths_or_hashes_in_provider_result": False,
            "exact_candidate_evidence_checkpoint_only": True,
        },
    }


def run_extractor_bakeoff_provider_phase(
    config: ExtractorBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    client: Any,
    output_path: Path | None = None,
    code_root: Path | None = None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> dict[str, Any]:
    """Run or exactly resume every label-blind per-block extractor call."""

    context = _prepare_extractor_provider_context(
        config,
        project_root=project_root,
        code_root=code_root,
        execution_selection=execution_selection,
        execution_selection_expectation=execution_selection_expectation,
    )
    _assert_extractor_artifact_paths_do_not_overwrite_inputs(
        config,
        project_root=project_root,
        artifact_paths=[checkpoint_path, *([output_path] if output_path is not None else [])],
        context=context,
    )
    existing = _load_extractor_provider_checkpoint(checkpoint_path, context)
    if existing is None:
        entries: dict[str, _ExtractorProviderEntry] = {}
        _write_extractor_provider_checkpoint(
            checkpoint_path,
            _extractor_provider_checkpoint(context, entries, sealed=False),
        )
    else:
        _validated_extractor_provider_entries(
            existing,
            context,
            require_sealed=existing.status == "sealed",
        )
        entries = dict(existing.entries)
        if existing.status == "sealed":
            ordered, _ = _validated_extractor_provider_entries(
                existing,
                context,
                require_sealed=True,
            )
            result = _extractor_provider_result(context, existing, ordered)
            if output_path is not None:
                _write_immutable_json(output_path, result)
            return result

    configured_client = _ConfiguredClient(
        client=client,
        require_parameters=bool(config.require_parameters),
    )
    slots = _extractor_expected_slots(context)
    for (
        slot_id,
        slot_contract_sha256,
        request,
        preparation,
        model,
        repetition_index,
        block,
    ) in slots:
        cached = entries.get(slot_id)
        if cached is not None:
            if not _extractor_entry_reusable(
                cached,
                slot_id=slot_id,
                slot_contract_sha256=slot_contract_sha256,
                request=request,
                preparation=preparation,
                model=model,
                repetition_index=repetition_index,
                block=block,
            ):
                raise ValueError("extractor provider checkpoint contains a stale slot")
            continue
        assert preparation.prepared is not None
        try:
            proposed, call, warnings = extract_page_candidates(
                client=configured_client,
                model=model.model,
                paper_id=preparation.spec.paper_id,
                paper_title=preparation.prepared.manifest.title,
                fragment=_fragment_for_block(block),
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                reasoning_effort=config.reasoning_effort,
                seed=config.seed,
            )
        except ProviderResponseValidationError as error:
            assessment = _contract_assessment(error.call, request)
            entry = _extractor_entry_payload(
                slot_id=slot_id,
                slot_contract_sha256=slot_contract_sha256,
                request=request,
                preparation=preparation,
                model=model,
                repetition_index=repetition_index,
                block=block,
                status="response_validation_failure",
                schema_status="invalid",
                contract_status=assessment["status"],
                contract_failure_codes=assessment["failure_codes"],
                call=error.call,
                error={
                    "stage": "extractor_call",
                    "type": type(error).__name__,
                    "code": error.code,
                },
            )
        except ProviderBudgetError:
            raise
        except Exception as error:
            entry = _extractor_entry_payload(
                slot_id=slot_id,
                slot_contract_sha256=slot_contract_sha256,
                request=request,
                preparation=preparation,
                model=model,
                repetition_index=repetition_index,
                block=block,
                status="provider_failure",
                schema_status="not_observed",
                contract_status="not_observed",
                error=_safe_error("extractor_call", error),
            )
        else:
            assessment = _contract_assessment(call, request)
            entry = _extractor_entry_payload(
                slot_id=slot_id,
                slot_contract_sha256=slot_contract_sha256,
                request=request,
                preparation=preparation,
                model=model,
                repetition_index=repetition_index,
                block=block,
                status=("success" if assessment["status"] == "satisfied" else "contract_failure"),
                schema_status="valid",
                contract_status=assessment["status"],
                contract_failure_codes=assessment["failure_codes"],
                candidates=proposed,
                warning_count=len(warnings),
                call=call,
            )
        entries[slot_id] = entry
        _write_extractor_provider_checkpoint(
            checkpoint_path,
            _extractor_provider_checkpoint(context, entries, sealed=False),
        )

    if set(entries) != {slot[0] for slot in slots}:
        raise RuntimeError("extractor checkpoint does not partition predeclared slots")
    sealed_checkpoint = _extractor_provider_checkpoint(context, entries, sealed=True)
    _write_extractor_provider_checkpoint(checkpoint_path, sealed_checkpoint)
    ordered, _ = _validated_extractor_provider_entries(
        sealed_checkpoint,
        context,
        require_sealed=True,
    )
    result = _extractor_provider_result(context, sealed_checkpoint, ordered)
    if output_path is not None:
        _write_immutable_json(output_path, result)
    return result


def _load_extractor_offline_references(
    context: _ExtractorProviderContext,
    *,
    project_root: Path,
) -> dict[str, tuple[PaperReference, dict[str, int | float | None], str]]:
    references = {}
    for preparation in context.preparations:
        assert preparation.prepared is not None
        reference_path = _project_path(project_root, preparation.spec.reference_path)
        reference_sha256 = sha256_file(reference_path)
        reference = load_reference(reference_path)
        if reference.paper_id != preparation.spec.paper_id:
            raise _InputError("reference_paper_id_mismatch")
        if reference.source_sha256 != preparation.prepared.source_sha256:
            raise _InputError("reference_source_hash_mismatch")
        scoped, scope = _scope_reference_to_page(reference, preparation.spec.page)
        references[preparation.spec.case_id] = (scoped, scope, reference_sha256)
    return references


def derive_extractor_execution_selection(
    config: ExtractorBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    experiment_id: str,
    public_manifest_sha256: str,
    wire_schema_gate: str = "0.99",
    code_root: Path | None = None,
) -> StageExecutionSelection:
    """Derive matched-quality eligibility from a sealed one-pass extractor smoke."""

    if config.fresh_repetitions != 1:
        raise ValueError("extractor contract smoke must declare exactly one repetition")
    context = _prepare_extractor_provider_context(
        config,
        project_root=project_root,
        code_root=code_root,
    )
    checkpoint = _load_extractor_provider_checkpoint(checkpoint_path, context)
    if checkpoint is None:
        raise ValueError("extractor contract smoke checkpoint is missing")
    entries, _ = _validated_extractor_provider_entries(
        checkpoint,
        context,
        require_sealed=True,
    )
    if checkpoint.provider_phase_seal_sha256 is None:
        raise ValueError("extractor contract smoke is missing its provider phase seal")
    expected_slots_per_model = sum(
        len(preparation.prepared.blocks)
        for preparation in context.preparations
        if preparation.prepared is not None
    )
    return derive_stage_execution_selection(
        experiment_id=experiment_id,
        stage="block_candidate_extraction",
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
        expected_slots_per_model=expected_slots_per_model,
        entries=entries,
    )


def score_extractor_bakeoff_offline(
    config: ExtractorBakeoffConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    output_path: Path | None = None,
    code_root: Path | None = None,
    execution_selection: StageExecutionSelection | None = None,
    execution_selection_expectation: StageExecutionSelectionExpectation | None = None,
) -> dict[str, Any]:
    """Join private references only after validating a complete provider seal."""

    context = _prepare_extractor_provider_context(
        config,
        project_root=project_root,
        code_root=code_root,
        execution_selection=execution_selection,
        execution_selection_expectation=execution_selection_expectation,
    )
    _assert_extractor_artifact_paths_do_not_overwrite_inputs(
        config,
        project_root=project_root,
        artifact_paths=[checkpoint_path, *([output_path] if output_path is not None else [])],
        context=context,
    )
    checkpoint = _load_extractor_provider_checkpoint(checkpoint_path, context)
    if checkpoint is None:
        raise ValueError("extractor provider phase checkpoint is missing")
    entries, _ = _validated_extractor_provider_entries(
        checkpoint,
        context,
        require_sealed=True,
    )
    # Deliberately last: no reference path is resolved, opened, or hashed until
    # the provider checkpoint and every expected slot have passed validation.
    references = (
        _load_extractor_offline_references(context, project_root=project_root)
        if context.models
        else {}
    )

    models: list[dict[str, Any]] = []
    executed_models = {item.model for item in context.models}
    for model in context.config.models:
        if model.model not in executed_models:
            if context.execution_selection is None:
                raise RuntimeError("extractor model was skipped without an execution selection")
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
                    "cases": [],
                }
            )
            continue
        cases = []
        for preparation in context.preparations:
            reference, scope, _ = references[preparation.spec.case_id]
            runs = _extractor_runs_from_entries(
                context,
                entries,
                model=model,
                preparation=preparation,
            )
            for run in runs:
                run.public_result["reference_scope"] = scope
                if not run.scoring_eligible:
                    run.public_result["scoring"] = {
                        "status": "not_scored",
                        "reason": "provider_or_contract_ineligible",
                    }
                    continue
                assert preparation.prepared is not None
                control_examined, observations_examined = _run_reference_examination(
                    reference,
                    preparation.prepared,
                    run,
                )
                run.public_result["score"] = score_reference(
                    reference,
                    list(run.candidates),
                    control_examined,
                    observations_examined,
                )
                run.public_result["scoring"] = {"status": "scored"}
            cases.append(
                _combine_case_runs(
                    preparation,
                    runs,
                    repetitions_predeclared=context.config.fresh_repetitions,
                )
            )
        aggregate = _aggregate_model(cases)
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
                "quality_status": aggregate["quality_measurement"]["status"],
                "quality": aggregate["quality"],
                "aggregate": aggregate,
                "cases": cases,
            }
        )

    request_contract = structured_request_contract(
        schema_name=EXTRACTOR_SCHEMA_NAME,
        schema=provider_json_schema(),
        seed=config.seed,
        require_parameters=bool(config.require_parameters),
    )
    result = {
        "schema_version": _EXTRACTOR_SCORE_VERSION,
        "bakeoff_id": config.bakeoff_id,
        "configuration_sha256": context.configuration_sha256,
        "run_contract_sha256": context.run_contract_sha256,
        "provider_phase_seal_sha256": checkpoint.provider_phase_seal_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "stage_contract_sha256": context.stage_contract_sha256,
        "determinism": {
            "seed": config.seed,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "require_parameters": bool(config.require_parameters),
            "fresh_repetitions": config.fresh_repetitions,
            "max_tokens": config.max_tokens,
            "min_confidence": config.min_confidence,
            "prompt_sha256": prompt_hash(),
            "segmentation": asdict(config.segmentation.production_config()),
            "reference_prompt_isolation": True,
        },
        "request_contract": request_contract,
        "request_contract_sha256": _json_sha256(request_contract),
        "code": context.code,
        "execution_selection_sha256": (
            context.execution_selection.selection_sha256
            if context.execution_selection is not None
            else None
        ),
        "declared_models": [item.model for item in context.config.models],
        "executed_models": [item.model for item in context.models],
        "private_reference_set_sha256": _json_sha256(
            {
                case_id: reference_sha256
                for case_id, (_, _, reference_sha256) in sorted(references.items())
            }
        ),
        "references": [
            {
                "case_id": preparation.spec.case_id,
                "reference_sha256": references[preparation.spec.case_id][2],
                "scope": references[preparation.spec.case_id][1],
            }
            for preparation in context.preparations
        ]
        if context.models
        else [],
        "models": models,
        "privacy": {
            "references_loaded_after_provider_phase_sealed": bool(context.models),
            "reference_paths_in_score": False,
        },
    }
    if output_path is not None:
        _write_immutable_json(output_path, result)
    return result
