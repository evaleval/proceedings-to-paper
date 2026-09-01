"""Prompt-isolated, failure-tolerant extractor model bake-offs.

The harness deliberately reuses the production segmenter, extractor, candidate
validation, and reference scorer. References are loaded for scoring but are never
passed to the extractor. Only aggregate telemetry and derived metadata are returned;
provider payloads, source text, credentials, and exception messages are excluded.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from proceedings_to_eee.domain.observation import StrictModel
from proceedings_to_eee.domain.status import ClaimType
from proceedings_to_eee.evaluation.corpus_score import aggregate_reference_scores
from proceedings_to_eee.evaluation.reference_score import score_reference
from proceedings_to_eee.extraction.llm import EXTRACTOR_SCHEMA_NAME, extract_page_candidates
from proceedings_to_eee.extraction.llm_schema import provider_json_schema
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout, extract_pdf_layout
from proceedings_to_eee.extraction.prompt import prompt_hash
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
from proceedings_to_eee.providers.openrouter import (
    OpenRouterClient,
    ProviderCall,
    ProviderResponseValidationError,
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

    schema_version: Literal["extractor-bakeoff/0.1"] = "extractor-bakeoff/0.1"
    bakeoff_id: str = Field(min_length=1)
    models: list[BakeoffModelSpec] = Field(min_length=2)
    cases: list[BakeoffCaseSpec] = Field(min_length=1)
    segmentation: BakeoffSegmentation = Field(default_factory=BakeoffSegmentation)
    min_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    max_tokens: int = Field(default=16_000, ge=1)
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    reasoning_effort: str | None = "minimal"
    seed: Literal[7] = EXTRACTOR_SEED

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> ExtractorBakeoffConfig:
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
    reference: PaperReference
    layout: PdfLayout
    blocks: tuple[ResultBlock, ...]
    source_id: str
    source_sha256: str
    manifest_sha256: str
    reference_sha256: str
    reference_scope: dict[str, int | float | None]


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
) -> _PreparedCase:
    manifest_path = _project_path(project_root, spec.manifest_path)
    reference_path = _project_path(project_root, spec.reference_path)
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

    # This object is retained only for post-extraction scoring. It never enters a prompt.
    paper_reference = load_reference(reference_path)
    if paper_reference.paper_id != spec.paper_id:
        raise _InputError("reference_paper_id_mismatch")
    if paper_reference.source_sha256 != source.sha256:
        raise _InputError("reference_source_hash_mismatch")
    reference, reference_scope = _scope_reference_to_page(paper_reference, spec.page)
    return _PreparedCase(
        spec=spec,
        manifest=manifest,
        reference=reference,
        layout=layout,
        blocks=blocks,
        source_id=source.source_id,
        source_sha256=source.sha256,
        manifest_sha256=sha256_file(manifest_path),
        reference_sha256=sha256_file(reference_path),
        reference_scope=reference_scope,
    )


def _prepare_cases(config: ExtractorBakeoffConfig, project_root: Path) -> list[_Preparation]:
    preparations: list[_Preparation] = []
    segmentation = config.segmentation.production_config()
    for spec in config.cases:
        try:
            prepared = _prepare_case(
                spec,
                project_root=project_root,
                segmentation=segmentation,
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
                    "reference_path": spec.reference_path,
                    "manifest_sha256": prepared.manifest_sha256,
                    "reference_sha256": prepared.reference_sha256,
                    "source_id": prepared.source_id,
                    "source_sha256": prepared.source_sha256,
                    "reference_scope": prepared.reference_scope,
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
    return preparations


def _public_telemetry(call: ProviderCall) -> dict[str, Any]:
    """Select only secret-free, non-trace provider metadata."""

    return {
        "model_requested": call.model_requested,
        "model_returned": call.model_returned,
        "provider_returned": call.provider_returned,
        "prompt_sha256": call.prompt_sha256,
        "response_sha256": call.response_sha256,
        "finish_reason": call.finish_reason,
        "attempts": call.attempts,
        "temperature": call.temperature,
        "reasoning_effort": call.reasoning_effort,
        "max_tokens": call.max_tokens,
        "request_contract": structured_request_contract_from_call(call),
        "data_collection": call.data_collection,
        "require_parameters": call.require_parameters,
        "zdr": call.zdr,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "total_tokens": call.total_tokens,
        "cost_usd": call.cost_usd,
        "latency_seconds": call.latency_seconds,
    }


def _run_case_model(
    *,
    prepared: _PreparedCase,
    model: BakeoffModelSpec,
    config: ExtractorBakeoffConfig,
    client: OpenRouterClient,
) -> dict[str, Any]:
    candidates = []
    call_results: list[dict[str, Any]] = []
    warning_count = 0
    for block in prepared.blocks:
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
            call_results.append(
                {
                    "block_id": block.block_id,
                    "page": block.page,
                    "status": "error",
                    "schema_status": "invalid",
                    "error": _safe_error("extractor_call", error),
                    "telemetry": _public_telemetry(error.call),
                }
            )
            continue
        except Exception as error:
            call_results.append(
                {
                    "block_id": block.block_id,
                    "page": block.page,
                    "status": "error",
                    "schema_status": (
                        "invalid" if type(error).__name__ in _SCHEMA_ERROR_TYPES else "not_observed"
                    ),
                    "error": _safe_error("extractor_call", error),
                }
            )
            continue
        candidates.extend(proposed)
        warning_count += len(warnings)
        call_results.append(
            {
                "block_id": block.block_id,
                "page": block.page,
                "status": "success",
                "schema_status": "valid",
                "telemetry": _public_telemetry(call),
            }
        )

    successful_calls = sum(item["status"] == "success" for item in call_results)
    failed_calls = len(call_results) - successful_calls
    status = (
        "success" if failed_calls == 0 else "error" if successful_calls == 0 else "partial_failure"
    )
    result: dict[str, Any] = {
        "case_id": prepared.spec.case_id,
        "paper_id": prepared.spec.paper_id,
        "page": prepared.spec.page,
        "status": status,
        "calls": call_results,
        "warning_count": warning_count,
        "reference_scope": prepared.reference_scope,
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
        return result
    result["candidate_count"] = len(candidates)
    result["candidate_validation"] = {
        "primary_results": sum(
            candidate.claim_type == ClaimType.PRIMARY_RESULT for candidate in candidates
        ),
        "text_support": {
            status: sum(candidate.text_support.value == status for candidate in candidates)
            for status in ("supported", "partially_supported", "unsupported", "unverified")
        },
    }
    try:
        # Reference data first enters the flow here, after all extraction calls completed.
        result["score"] = score_reference(prepared.reference, candidates)
    except Exception as error:
        result["status"] = "error"
        result["error"] = _safe_error("reference_scoring", error)
    return result


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


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
        },
    }


def _aggregate_usage(call_results: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [item for item in call_results if "telemetry" in item]
    usage: dict[str, Any] = {}
    for field in ("input_tokens", "output_tokens", "total_tokens", "cost_usd"):
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


def _aggregate_model(cases: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [call for case in cases for call in case.get("calls", [])]
    successful_calls = sum(call["status"] == "success" for call in calls)
    successful_cases = sum(case["status"] == "success" for case in cases)
    partial_cases = sum(case["status"] == "partial_failure" for case in cases)
    scores = [case["score"] for case in cases if "score" in case]
    quality = _aggregate_quality(scores)
    reference_evaluation = aggregate_reference_scores(scores) if scores else None
    schema_valid = sum(call.get("schema_status") == "valid" for call in calls)
    schema_invalid = sum(call.get("schema_status") == "invalid" for call in calls)
    schema_not_observed = len(calls) - schema_valid - schema_invalid
    support_counts = {
        status: sum(
            case.get("candidate_validation", {}).get("text_support", {}).get(status, 0)
            for case in cases
        )
        for status in ("supported", "partially_supported", "unsupported", "unverified")
    }
    return {
        "execution": {
            "cases_attempted": len(cases),
            "cases_succeeded": successful_cases,
            "cases_partial_failure": partial_cases,
            "cases_failed": len(cases) - successful_cases - partial_cases,
            "case_success_rate": _ratio(successful_cases, len(cases)),
            "cases_scored": len(scores),
            "case_scored_rate": _ratio(len(scores), len(cases)),
            "calls_attempted": len(calls),
            "calls_succeeded": successful_calls,
            "calls_failed": len(calls) - successful_calls,
            "call_success_rate": _ratio(successful_calls, len(calls)),
        },
        "schema": {
            "structured_responses_valid": schema_valid,
            "structured_responses_invalid": schema_invalid,
            "structured_response_not_observed": schema_not_observed,
            "valid_rate_of_observed_responses": _ratio(schema_valid, schema_valid + schema_invalid),
            "end_to_end_schema_success_rate": _ratio(schema_valid, len(calls)),
        },
        "usage": _aggregate_usage(calls),
        "evidence": {
            "candidate_text_support": support_counts,
            "reference_evidence_supported_accuracy": {
                "macro": quality["macro"]["field_accuracy"]["evidence_supported"],
                "micro": quality["micro"]["field_accuracy"]["evidence_supported"],
            },
            "reference_page_anchor_accuracy": {
                "macro": quality["macro"]["field_accuracy"]["page"],
                "micro": quality["micro"]["field_accuracy"]["page"],
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
        "quality": quality,
    }


def run_extractor_bakeoff(
    config: ExtractorBakeoffConfig,
    *,
    project_root: Path,
    client: OpenRouterClient,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run every model by every case while isolating and aggregating failures."""

    preparations = _prepare_cases(config, project_root)
    model_results: list[dict[str, Any]] = []
    for model in config.models:
        cases: list[dict[str, Any]] = []
        for preparation in preparations:
            if preparation.prepared is None:
                cases.append(
                    {
                        "case_id": preparation.spec.case_id,
                        "paper_id": preparation.spec.paper_id,
                        "page": preparation.spec.page,
                        "status": "error",
                        "calls": [],
                        "error": preparation.public_result["error"],
                    }
                )
                continue
            cases.append(
                _run_case_model(
                    prepared=preparation.prepared,
                    model=model,
                    config=config,
                    client=client,
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

    result = {
        "schema_version": "extractor-bakeoff-result/0.1",
        "bakeoff_id": config.bakeoff_id,
        "configuration_sha256": sha256_bytes(canonical_json_bytes(config)),
        "determinism": {
            "seed": config.seed,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "max_tokens": config.max_tokens,
            "min_confidence": config.min_confidence,
            "prompt_sha256": prompt_hash(),
            "segmentation": asdict(config.segmentation.production_config()),
            "reference_prompt_isolation": True,
        },
        "request_contract": structured_request_contract(
            schema_name=EXTRACTOR_SCHEMA_NAME,
            schema=provider_json_schema(),
            seed=config.seed,
            require_parameters=False,
        ),
        "code": _code_state(project_root),
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
) -> dict[str, Any]:
    """Load and run one YAML bake-off definition."""

    return run_extractor_bakeoff(
        load_bakeoff_config(config_path),
        project_root=project_root,
        client=client,
        output_path=output_path,
    )
