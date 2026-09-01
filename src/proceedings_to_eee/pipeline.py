"""End-to-end paper pipeline with content-addressed intermediate artifacts."""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from proceedings_to_eee.composition.eee import compose_eee_records
from proceedings_to_eee.corpus import CorpusSpec, PaperSpec, build_corpus_binding
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ClaimType, ExportStatus
from proceedings_to_eee.evaluation.control_coverage import control_examination
from proceedings_to_eee.evaluation.corpus_score import aggregate_reference_scores
from proceedings_to_eee.evaluation.reference_score import score_reference
from proceedings_to_eee.evaluation.spot_checks import score_spot_checks
from proceedings_to_eee.extraction.llm import (
    EXTRACTOR_SEED,
    RowEnumerationOutcome,
    enumerate_row_batch,
    extract_page_candidates,
    extractor_request_contract,
    row_extractor_request_contract,
)
from proceedings_to_eee.extraction.pdf_layout import (
    PageFragment,
    PdfLayout,
    extract_pdf_layout,
    select_result_pages,
)
from proceedings_to_eee.extraction.prompt import prompt_hash, row_prompt_hash
from proceedings_to_eee.extraction.result_blocks import (
    LEGACY_RECOVERY_MAX_DEPTH,
    ResultBlock,
    ResultBlockConfig,
    segment_page_result_blocks,
    split_result_block,
)
from proceedings_to_eee.extraction.row_enumeration import (
    RowAttemptTelemetry,
    RowBatch,
    RowDisposition,
    RowDispositionRecord,
    RowEnumerationConfig,
    RowEnumerationPlan,
    build_row_enumeration_plan,
)
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from proceedings_to_eee.providers.openrouter import (
    OpenRouterClient,
    ProviderCall,
    ProviderRequestRejectedError,
    ProviderResponseValidationError,
)
from proceedings_to_eee.reference import load_reference
from proceedings_to_eee.reporting.corpus_html import render_corpus_html_file
from proceedings_to_eee.reporting.html import render_review_report
from proceedings_to_eee.sources.manifest import (
    LicenseDisposition,
    SourceManifest,
    SourceRole,
    download_and_freeze_source,
    freeze_repository_source,
    resolve_cached_path,
)
from proceedings_to_eee.validation.candidates import (
    deduplicate_candidates,
    validate_candidates,
)
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record
from proceedings_to_eee.verification.binding import (
    bind_candidate_block,
    frozen_evidence_block,
)
from proceedings_to_eee.verification.independent import (
    VERIFIER_SEED,
    CandidateVerification,
    IndependentDecision,
    verifier_request_contract,
    verify_candidate,
)


@dataclass(frozen=True)
class PipelineSettings:
    project_root: Path
    schema_path: Path
    schema_sha256: str
    output_root: Path
    model: str
    min_confidence: float = 0.8
    max_tokens: int = 16_000
    temperature: float | None = 0.0
    reasoning_effort: str | None = "minimal"
    seed: int = EXTRACTOR_SEED
    max_blocks_per_page: int = 6
    row_enumeration_enabled: bool = False
    row_enumeration_config: RowEnumerationConfig = dataclass_field(
        default_factory=RowEnumerationConfig
    )
    row_estimated_call_cost_usd: float | None = None
    verifier_model: str | None = None
    verifier_max_tokens: int = 2_000


@dataclass(frozen=True)
class LegacyRecoveryFailure:
    """Secret-free terminal state for one bounded recovery subtree."""

    block_id: str
    page: int
    depth: int
    error_code: str
    completed_provider_call: bool
    terminal_reason: str
    safe_details: dict[str, int] = dataclass_field(default_factory=dict)


@dataclass
class LegacyRecoveryOutcome:
    """All usable work and typed failures from one bounded split tree."""

    candidates: list[CandidateObservation] = dataclass_field(default_factory=list)
    calls: list[ProviderCall] = dataclass_field(default_factory=list)
    successful_calls: list[ProviderCall] = dataclass_field(default_factory=list)
    warnings: list[str] = dataclass_field(default_factory=list)
    terminal_failures: list[LegacyRecoveryFailure] = dataclass_field(default_factory=list)
    max_depth_reached: int = 0

    @property
    def succeeded(self) -> bool:
        return not self.terminal_failures


_EXTRACTOR_CHECKPOINT_SCHEMA_VERSION = "extractor-block-checkpoint/0.1"
_EXTRACTOR_CHECKPOINT_CONTRACT_VERSION = "extractor-block-checkpoint-contract/0.1"
_ROW_CHECKPOINT_SCHEMA_VERSION = "row-enumeration-checkpoint/0.1"
# This version also names the provider-response projection contract. Bump it when
# row responses can no longer be rehydrated with the current typed domain models.
_ROW_CHECKPOINT_CONTRACT_VERSION = "row-enumeration-checkpoint-contract/0.1"
_PAPER_RUN_OUTPUTS = (
    "run.json",
    "reference-score.json",
    "observations.jsonl",
    "verifications.jsonl",
    "spot-checks.json",
    "review.html",
)
_CORPUS_RUN_OUTPUTS = ("corpus-run.json", "corpus-evaluation.json", "corpus-review.html")


def _extractor_run_configuration(settings: PipelineSettings) -> dict[str, Any]:
    """Return reproducibility metadata available before any extractor call."""

    return {
        "provider": "openrouter",
        "model": settings.model,
        "temperature": settings.temperature,
        "reasoning_effort": settings.reasoning_effort,
        "max_tokens": settings.max_tokens,
        "seed": settings.seed,
        "prompt_sha256": prompt_hash(),
        "request_contract": extractor_request_contract(seed=settings.seed),
    }


def _row_enumeration_run_configuration(settings: PipelineSettings) -> dict[str, Any]:
    """Return the independent row-stage contract, even when the stage is disabled."""

    return {
        "enabled": settings.row_enumeration_enabled,
        "provider": "openrouter",
        "model": settings.model,
        "temperature": settings.temperature,
        "reasoning_effort": settings.reasoning_effort,
        "max_tokens": settings.max_tokens,
        "seed": settings.seed,
        "prompt_sha256": row_prompt_hash(),
        "request_contract": row_extractor_request_contract(seed=settings.seed),
        "limits": settings.row_enumeration_config.model_dump(mode="json"),
    }


def _row_preflight(
    *,
    settings: PipelineSettings,
    block_count: int,
    plan: RowEnumerationPlan,
) -> dict[str, Any]:
    """Bound row-stage calls and label any cost projection as an estimate."""

    if (
        settings.row_estimated_call_cost_usd is not None
        and settings.row_estimated_call_cost_usd < 0
    ):
        raise ValueError("row estimated call cost must be non-negative")
    expected_total = block_count + plan.telemetry.expected_calls
    maximum_total = block_count + plan.telemetry.maximum_calls
    per_call = settings.row_estimated_call_cost_usd
    return {
        "basis": (
            "block count plus deterministic row-plan batches before provider execution; "
            "maximum permits one unresolved-only split level and is a hard call bound"
        ),
        "baseline_block_calls": block_count,
        "planned_row_base_calls": plan.telemetry.expected_calls,
        "maximum_row_calls": plan.telemetry.maximum_calls,
        "expected_total_calls": expected_total,
        "maximum_total_calls": maximum_total,
        "expected_call_multiplier": expected_total / block_count if block_count else None,
        "maximum_call_multiplier": maximum_total / block_count if block_count else None,
        "estimated_cost_basis": (
            "user-supplied historical mean cost per successful call; not a provider quote"
            if per_call is not None
            else None
        ),
        "estimated_cost_per_call_usd": per_call,
        "estimated_row_cost_usd": (
            round(plan.telemetry.expected_calls * per_call, 12) if per_call is not None else None
        ),
        "estimated_expected_total_cost_usd": (
            round(expected_total * per_call, 12) if per_call is not None else None
        ),
        "estimated_maximum_total_cost_usd": (
            round(maximum_total * per_call, 12) if per_call is not None else None
        ),
    }


def _verifier_run_configuration(settings: PipelineSettings) -> dict[str, Any]:
    """Return reproducibility metadata even when verification is disabled or fails."""

    return {
        "enabled": settings.verifier_model is not None,
        "model": settings.verifier_model,
        "max_tokens": settings.verifier_max_tokens,
        "seed": VERIFIER_SEED,
        "request_contract": verifier_request_contract(),
    }


def _provider_call_telemetry(
    calls: list[ProviderCall],
    *,
    basis: str | None = None,
) -> dict[str, Any]:
    """Aggregate only secret-free successful-call metadata as honest lower bounds."""

    costs: list[float] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    total_tokens: list[int] = []
    latencies: list[float] = []
    attempts_total = 0
    for call in calls:
        if not math.isfinite(call.latency_seconds) or call.latency_seconds < 0:
            raise ValueError("provider latency must be non-negative and finite")
        latencies.append(float(call.latency_seconds))
        attempts_total += call.attempts
        if call.cost_usd is not None:
            if not math.isfinite(call.cost_usd) or call.cost_usd < 0:
                raise ValueError("provider cost must be non-negative and finite")
            costs.append(float(call.cost_usd))
        for value, output, field in (
            (call.input_tokens, input_tokens, "input_tokens"),
            (call.output_tokens, output_tokens, "output_tokens"),
            (call.total_tokens, total_tokens, "total_tokens"),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"provider {field} must be a non-negative integer")
            output.append(value)
    call_count = len(calls)
    return {
        "basis": basis
        or (
            "successful final block calls; cost, token, retry, and attempt totals are lower "
            "bounds when provider metadata or superseded/failed attempts are unavailable"
        ),
        "calls": call_count,
        "cost_usd_lower_bound": round(sum(costs), 12),
        "cost_reported_calls": len(costs),
        "input_tokens_lower_bound": sum(input_tokens),
        "input_tokens_reported_calls": len(input_tokens),
        "output_tokens_lower_bound": sum(output_tokens),
        "output_tokens_reported_calls": len(output_tokens),
        "total_tokens_lower_bound": sum(total_tokens),
        "total_tokens_reported_calls": len(total_tokens),
        "latency_seconds_total": round(sum(latencies), 6),
        "latency_seconds_mean": round(sum(latencies) / call_count, 6) if calls else None,
        "latency_seconds_max": round(max(latencies), 6) if calls else None,
        "attempts_lower_bound": attempts_total,
        "retries_lower_bound": attempts_total - call_count,
    }


def _corpus_operational_metrics(
    summaries: list[dict[str, Any]], *, wall_clock_seconds: float
) -> dict[str, Any]:
    telemetry = []
    execution = []
    row_telemetry = []
    row_execution = []
    row_plans = []
    for summary in summaries:
        extractor = summary.get("extractor")
        if not isinstance(extractor, dict):
            continue
        item = extractor.get("successful_call_telemetry")
        if isinstance(item, dict):
            telemetry.append(item)
        item = extractor.get("execution")
        if isinstance(item, dict):
            execution.append(item)
        row_stage = summary.get("row_enumeration")
        if not isinstance(row_stage, dict):
            continue
        item = row_stage.get("successful_call_telemetry")
        if isinstance(item, dict):
            row_telemetry.append(item)
        item = row_stage.get("execution")
        if isinstance(item, dict):
            row_execution.append(item)
        item = row_stage.get("plan")
        if isinstance(item, dict):
            row_plans.append(item)
    calls = sum(int(item.get("calls", 0)) for item in telemetry)
    latency_total = sum(float(item.get("latency_seconds_total", 0.0)) for item in telemetry)
    latency_max_values = [
        float(item["latency_seconds_max"])
        for item in telemetry
        if item.get("latency_seconds_max") is not None
    ]
    row_calls = sum(int(item.get("calls", 0)) for item in row_telemetry)
    row_latency_total = sum(float(item.get("latency_seconds_total", 0.0)) for item in row_telemetry)
    row_latency_max_values = [
        float(item["latency_seconds_max"])
        for item in row_telemetry
        if item.get("latency_seconds_max") is not None
    ]
    return {
        "wall_clock_seconds": round(wall_clock_seconds, 6),
        "extractor": {
            "basis": (
                "successful final block calls across the split; monetary, token, retry, and "
                "attempt totals are lower bounds"
            ),
            "calls": calls,
            "cost_usd_lower_bound": round(
                sum(float(item.get("cost_usd_lower_bound", 0.0)) for item in telemetry), 12
            ),
            "cost_reported_calls": sum(
                int(item.get("cost_reported_calls", 0)) for item in telemetry
            ),
            "input_tokens_lower_bound": sum(
                int(item.get("input_tokens_lower_bound", 0)) for item in telemetry
            ),
            "input_tokens_reported_calls": sum(
                int(item.get("input_tokens_reported_calls", 0)) for item in telemetry
            ),
            "output_tokens_lower_bound": sum(
                int(item.get("output_tokens_lower_bound", 0)) for item in telemetry
            ),
            "output_tokens_reported_calls": sum(
                int(item.get("output_tokens_reported_calls", 0)) for item in telemetry
            ),
            "total_tokens_lower_bound": sum(
                int(item.get("total_tokens_lower_bound", 0)) for item in telemetry
            ),
            "total_tokens_reported_calls": sum(
                int(item.get("total_tokens_reported_calls", 0)) for item in telemetry
            ),
            "latency_seconds_total": round(latency_total, 6),
            "latency_seconds_mean": round(latency_total / calls, 6) if calls else None,
            "latency_seconds_max": round(max(latency_max_values), 6)
            if latency_max_values
            else None,
            "attempts_lower_bound": sum(
                int(item.get("attempts_lower_bound", 0)) for item in telemetry
            ),
            "retries_lower_bound": sum(
                int(item.get("retries_lower_bound", 0)) for item in telemetry
            ),
            "blocks_total": sum(int(item.get("blocks_total", 0)) for item in execution),
            "blocks_succeeded": sum(int(item.get("blocks_succeeded", 0)) for item in execution),
            "blocks_resumed": sum(int(item.get("blocks_resumed", 0)) for item in execution),
            "blocks_failed": sum(int(item.get("blocks_failed", 0)) for item in execution),
        },
        "row_enumeration": {
            "basis": (
                "bounded staged row calls across the split; monetary, token, retry, and "
                "attempt totals are lower bounds"
            ),
            "calls": row_calls,
            "cost_usd_lower_bound": round(
                sum(float(item.get("cost_usd_lower_bound", 0.0)) for item in row_telemetry),
                12,
            ),
            "cost_reported_calls": sum(
                int(item.get("cost_reported_calls", 0)) for item in row_telemetry
            ),
            "input_tokens_lower_bound": sum(
                int(item.get("input_tokens_lower_bound", 0)) for item in row_telemetry
            ),
            "output_tokens_lower_bound": sum(
                int(item.get("output_tokens_lower_bound", 0)) for item in row_telemetry
            ),
            "total_tokens_lower_bound": sum(
                int(item.get("total_tokens_lower_bound", 0)) for item in row_telemetry
            ),
            "latency_seconds_total": round(row_latency_total, 6),
            "latency_seconds_mean": (
                round(row_latency_total / row_calls, 6) if row_calls else None
            ),
            "latency_seconds_max": (
                round(max(row_latency_max_values), 6) if row_latency_max_values else None
            ),
            "attempts_lower_bound": sum(
                int(item.get("attempts_lower_bound", 0)) for item in row_telemetry
            ),
            "retries_lower_bound": sum(
                int(item.get("retries_lower_bound", 0)) for item in row_telemetry
            ),
            "tables_considered": sum(int(item.get("tables_considered", 0)) for item in row_plans),
            "dense_tables": sum(int(item.get("dense_tables", 0)) for item in row_plans),
            "rows_planned": sum(int(item.get("rows_planned", 0)) for item in row_plans),
            "unbatchable_rows": sum(int(item.get("unbatchable_rows", 0)) for item in row_plans),
            "base_batches": sum(int(item.get("base_batches", 0)) for item in row_plans),
            "maximum_calls": sum(int(item.get("maximum_calls", 0)) for item in row_plans),
            "batches_resumed": sum(int(item.get("batches_resumed", 0)) for item in row_execution),
        },
    }


def _manifest_path(settings: PipelineSettings, paper_id: str) -> Path:
    return settings.output_root / paper_id / "source-manifest.json"


def freeze_paper(spec: PaperSpec, settings: PipelineSettings) -> SourceManifest:
    """Reuse an already pinned manifest or download exact source bytes once."""

    path = _manifest_path(settings, spec.paper_id)
    configured_sources: list[tuple[SourceRole, str, str | None]] = [
        (SourceRole.PAPER, str(spec.pdf_url), None),
        *[(SourceRole.SUPPLEMENT, str(url), None) for url in spec.supplement_urls],
    ]
    if spec.repository_url and spec.repository_commit:
        configured_sources.append(
            (SourceRole.REPOSITORY, str(spec.repository_url), spec.repository_commit.casefold())
        )
    if path.exists():
        manifest = SourceManifest.model_validate(read_json(path))
        frozen_contract = [
            (source.role, source.original_uri, source.git_commit) for source in manifest.sources
        ]
        if frozen_contract != configured_sources:
            raise ValueError(
                f"configured source bundle changed for {spec.paper_id}; "
                f"frozen={frozen_contract!r}, configured={configured_sources!r}"
            )
        for source in manifest.sources:
            if source.role == SourceRole.REPOSITORY:
                if source.git_commit is None:
                    raise ValueError(f"repository source is not commit-pinned for {spec.paper_id}")
                continue
            resolve_cached_path(source, settings.project_root)
        return manifest
    cache_root = settings.project_root / "data" / "sources"
    paper_source = download_and_freeze_source(
        paper_id=spec.paper_id,
        role=SourceRole.PAPER,
        url=str(spec.pdf_url),
        cache_root=cache_root,
        license_disposition=LicenseDisposition.DERIVED_METADATA_ONLY,
    )
    sources = [paper_source]
    for supplement_url in spec.supplement_urls:
        sources.append(
            download_and_freeze_source(
                paper_id=spec.paper_id,
                role=SourceRole.SUPPLEMENT,
                url=str(supplement_url),
                cache_root=cache_root,
                license_disposition=LicenseDisposition.DERIVED_METADATA_ONLY,
            )
        )
    if spec.repository_url and spec.repository_commit:
        sources.append(
            freeze_repository_source(
                paper_id=spec.paper_id,
                url=str(spec.repository_url),
                git_commit=spec.repository_commit,
                license_disposition=LicenseDisposition.UNKNOWN,
            )
        )
    manifest = SourceManifest(
        paper_id=spec.paper_id,
        title=spec.title,
        doi=spec.doi,
        arxiv_id=spec.arxiv_id,
        proceedings_url=spec.acm_url,
        sources=sources,
    )
    write_json(path, manifest)
    return manifest


def _code_state(project_root: Path) -> dict[str, str | bool]:
    git_available = True
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "uncommitted"
        git_available = False
    try:
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "src",
                "configs",
                "schemas",
                "pyproject.toml",
                "uv.lock",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        status = ""
        git_available = False
    semantic_paths: list[tuple[str, Path]] = []
    if git_available:
        try:
            listed = subprocess.run(
                [
                    "git",
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "--",
                    "src",
                    "configs",
                ],
                cwd=project_root,
                check=True,
                capture_output=True,
                timeout=15,
            ).stdout
            semantic_paths = [
                (raw.decode("utf-8"), project_root / raw.decode("utf-8"))
                for raw in listed.split(b"\0")
                if raw
            ]
        except (subprocess.CalledProcessError, FileNotFoundError, UnicodeDecodeError):
            git_available = False
    if not semantic_paths:
        for base in (project_root / "src", project_root / "configs"):
            if not base.exists():
                continue
            semantic_paths.extend(
                (path.relative_to(project_root).as_posix(), path)
                for path in base.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            )
    if not semantic_paths:
        package_root = Path(__file__).resolve().parent
        semantic_paths.extend(
            (f"installed-package/{path.relative_to(package_root).as_posix()}", path)
            for path in package_root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    source_hash = hashlib.sha256()
    for label, path in sorted(set(semantic_paths)):
        if not path.is_file() or path.is_symlink():
            continue
        source_hash.update(label.encode())
        source_hash.update(path.read_bytes())
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_available": git_available,
        "source_tree_sha256": source_hash.hexdigest(),
    }


def _select_pages(layout: PdfLayout, spec: PaperSpec):
    if spec.include_pages:
        missing = [page for page in spec.include_pages if page < 1 or page > layout.page_count]
        if missing:
            raise ValueError(f"configured pages outside PDF for {spec.paper_id}: {missing}")
        return [layout.pages[page - 1] for page in spec.include_pages]
    return select_result_pages(layout, limit=spec.max_result_pages)


def _block_fragment(block: ResultBlock):
    """Adapt a bounded block to the existing source-fragment extraction contract."""

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


def _safe_error_message(error: Exception) -> str:
    """Return bounded diagnostic text without credentials or authorization headers."""

    message = str(error) or type(error).__name__
    message = re.sub(
        r"(?i)(https?://[^\s?#]+)[?#][^\s]*",
        r"\1?[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?<![A-Za-z0-9_:/])/(?:Users|home|private|tmp|var)/[^\s\"']+",
        "[LOCAL_PATH]",
        message,
    )
    message = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"sk-or-[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = re.sub(r"(?i)(api[_ -]?key\s*[=:]\s*)\S+", r"\1[REDACTED]", message)
    return message[:1000]


def _reference_path(project_root: Path, configured_path: str) -> Path:
    root = project_root.resolve()
    path = (root / configured_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("reference path escaped project root")
    return path


def _score_reference_after_paper_error(
    *,
    spec: PaperSpec,
    settings: PipelineSettings,
    manifest_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Retain frozen reference denominators when a paper fails after source freeze."""

    if spec.reference_path is None:
        return None, None
    manifest = SourceManifest.model_validate(read_json(manifest_path))
    paper_source = next(source for source in manifest.sources if source.role == SourceRole.PAPER)
    reference = load_reference(_reference_path(settings.project_root, spec.reference_path))
    if reference.paper_id != spec.paper_id:
        raise ValueError("reference paper_id does not match the failed paper")
    if reference.source_sha256 != paper_source.sha256:
        raise ValueError("reference source hash does not match the failed paper")
    score = score_reference(reference, [])
    return score, write_json(output_path, score)


def _clear_paper_run_outputs(output_dir: Path) -> None:
    """Remove only known generated outputs before rebuilding one paper run."""

    for name in _PAPER_RUN_OUTPUTS:
        (output_dir / name).unlink(missing_ok=True)
    # These describe the current invocation, unlike the contract-bound checkpoint
    # retained for a later opt-in resume. Leaving them behind when the row stage is
    # disabled makes offline coverage mistake a prior ledger for the current run.
    for name in (
        "row-enumeration-plan.json",
        "row-enumeration-preflight.json",
        "row-enumeration.json",
    ):
        (output_dir / "private" / name).unlink(missing_ok=True)
    eee_dir = output_dir / "eee"
    if eee_dir.is_symlink():
        raise RuntimeError("refusing to clean a symlinked EEE output directory")
    if eee_dir.is_dir():
        for record_path in eee_dir.glob("*.json"):
            if record_path.is_file():
                record_path.unlink()
    (output_dir / "private" / "invalid-eee.json").unlink(missing_ok=True)


def _clear_corpus_run_outputs(output_root: Path) -> None:
    """Remove only corpus aggregates that belong to a previous invocation."""

    for name in _CORPUS_RUN_OUTPUTS:
        (output_root / name).unlink(missing_ok=True)


def _extractor_checkpoint_contract(
    *,
    spec: PaperSpec,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    block_config: ResultBlockConfig,
    blocks: list[ResultBlock],
    code_state: dict[str, str | bool],
) -> dict[str, Any]:
    """Bind reusable block results to every input that can change extraction."""

    return {
        "schema_version": _EXTRACTOR_CHECKPOINT_CONTRACT_VERSION,
        "paper_id": spec.paper_id,
        "paper_title": spec.title,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_parser": layout.parser,
        "layout_parser_version": layout.parser_version,
        "result_block_segmentation": asdict(block_config),
        "blocks": [
            {
                "block_id": block.block_id,
                "source_id": block.source_id,
                "page": block.page,
                "text_sha256": block.text_sha256,
            }
            for block in blocks
        ],
        "extractor": _extractor_run_configuration(settings),
        "code": code_state,
    }


def _new_extractor_checkpoint(contract: dict[str, Any], contract_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": _EXTRACTOR_CHECKPOINT_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "blocks": {},
    }


def _checkpoint_reuse_envelope(contract: dict[str, Any]) -> dict[str, Any]:
    """Return inputs that must match before any prior block can be reused.

    Segmentation, the complete block set, and unrelated code may change while an
    individual block's provider request remains byte-for-byte equivalent. The
    exact block text hash is checked separately during migration and rehydration.
    """

    keys = (
        "schema_version",
        "paper_id",
        "paper_title",
        "source_manifest_sha256",
        "layout_parser",
        "layout_parser_version",
        "extractor",
    )
    return {key: contract.get(key) for key in keys}


def _migrate_compatible_checkpoint_blocks(
    payload: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    """Carry forward only exact block results under an equivalent request envelope."""

    previous_contract = payload.get("contract")
    previous_blocks = payload.get("blocks")
    if not isinstance(previous_contract, dict) or not isinstance(previous_blocks, dict):
        return _new_extractor_checkpoint(contract, contract_sha256)
    if _checkpoint_reuse_envelope(previous_contract) != _checkpoint_reuse_envelope(contract):
        return _new_extractor_checkpoint(contract, contract_sha256)
    current_hashes = {
        block["block_id"]: block["text_sha256"]
        for block in contract.get("blocks", [])
        if isinstance(block, dict)
        and isinstance(block.get("block_id"), str)
        and isinstance(block.get("text_sha256"), str)
    }
    checkpoint = _new_extractor_checkpoint(contract, contract_sha256)
    checkpoint["blocks"] = {
        block_id: entry
        for block_id, entry in previous_blocks.items()
        if block_id in current_hashes
        and isinstance(entry, dict)
        and entry.get("block_text_sha256") == current_hashes[block_id]
    }
    return checkpoint


def _load_extractor_checkpoint(
    path: Path,
    *,
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    """Load exact state or migrate exact blocks under a compatible request envelope."""

    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return _new_extractor_checkpoint(contract, contract_sha256)
    if not isinstance(payload, dict):
        return _new_extractor_checkpoint(contract, contract_sha256)
    if payload.get("schema_version") != _EXTRACTOR_CHECKPOINT_SCHEMA_VERSION:
        return _new_extractor_checkpoint(contract, contract_sha256)
    if (
        payload.get("contract_sha256") == contract_sha256
        and payload.get("contract") == contract
        and isinstance(payload.get("blocks"), dict)
    ):
        return payload
    return _migrate_compatible_checkpoint_blocks(
        payload,
        contract=contract,
        contract_sha256=contract_sha256,
    )


def _validated_checkpoint_entry(
    entry: Any,
    *,
    block: ResultBlock,
) -> tuple[list[CandidateObservation], ProviderCall, list[str]] | None:
    """Rehydrate one successful block result without trusting private JSON."""

    if not isinstance(entry, dict) or entry.get("block_text_sha256") != block.text_sha256:
        return None
    raw_candidates = entry.get("candidates")
    raw_warnings = entry.get("warnings")
    if not isinstance(raw_candidates, list) or not isinstance(raw_warnings, list):
        return None
    if not all(isinstance(warning, str) for warning in raw_warnings):
        return None
    try:
        candidates = [CandidateObservation.model_validate(item) for item in raw_candidates]
        call = ProviderCall.model_validate(entry.get("call"))
    except (TypeError, ValueError):
        return None
    return candidates, call, raw_warnings


def _row_checkpoint_contract(
    *,
    spec: PaperSpec,
    settings: PipelineSettings,
    manifest: SourceManifest,
    layout: PdfLayout,
    plan_sha256: str,
    code_state: dict[str, str | bool],
) -> dict[str, Any]:
    """Bind reusable row outcomes to source, plan, provider contract, and code."""

    return {
        "schema_version": _ROW_CHECKPOINT_CONTRACT_VERSION,
        "paper_id": spec.paper_id,
        "paper_title": spec.title,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_parser": layout.parser,
        "layout_parser_version": layout.parser_version,
        "plan_sha256": plan_sha256,
        "row_enumeration": _row_enumeration_run_configuration(settings),
        "code": code_state,
    }


def _new_row_checkpoint(contract: dict[str, Any], contract_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": _ROW_CHECKPOINT_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "batches": {},
    }


def _row_checkpoint_reuse_envelope(contract: dict[str, Any]) -> dict[str, Any]:
    """Return request inputs that must match before a row batch can be reused.

    The full code state remains in the contract for provenance. Downstream code may
    change without changing a provider request, while the exact plan and batch hashes
    plus typed entry validation bind reusable outcomes to their original inputs.
    """

    keys = (
        "schema_version",
        "paper_id",
        "paper_title",
        "source_manifest_sha256",
        "layout_parser",
        "layout_parser_version",
        "plan_sha256",
        "row_enumeration",
    )
    return {key: contract.get(key) for key in keys}


def _migrate_compatible_row_checkpoint_batches(
    payload: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_sha256: str,
    batches: list[RowBatch],
) -> dict[str, Any]:
    """Carry forward only exact, typed row batches under one request envelope."""

    previous_contract = payload.get("contract")
    previous_batches = payload.get("batches")
    if not isinstance(previous_contract, dict) or not isinstance(previous_batches, dict):
        return _new_row_checkpoint(contract, contract_sha256)
    if _row_checkpoint_reuse_envelope(previous_contract) != _row_checkpoint_reuse_envelope(
        contract
    ):
        return _new_row_checkpoint(contract, contract_sha256)
    current_batches = {batch.batch_id: batch for batch in batches}
    checkpoint = _new_row_checkpoint(contract, contract_sha256)
    checkpoint["batches"] = {
        batch_id: entry
        for batch_id, entry in previous_batches.items()
        if (batch := current_batches.get(batch_id)) is not None
        and _validated_row_checkpoint_entry(entry, batch=batch) is not None
    }
    return checkpoint


def _load_row_checkpoint(
    path: Path,
    *,
    contract: dict[str, Any],
    contract_sha256: str,
    batches: list[RowBatch],
) -> dict[str, Any]:
    """Load exact state or migrate typed batches under a compatible request envelope."""

    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return _new_row_checkpoint(contract, contract_sha256)
    if not isinstance(payload, dict):
        return _new_row_checkpoint(contract, contract_sha256)
    if payload.get("schema_version") != _ROW_CHECKPOINT_SCHEMA_VERSION:
        return _new_row_checkpoint(contract, contract_sha256)
    if (
        payload.get("contract_sha256") == contract_sha256
        and payload.get("contract") == contract
        and isinstance(payload.get("batches"), dict)
    ):
        return payload
    return _migrate_compatible_row_checkpoint_batches(
        payload,
        contract=contract,
        contract_sha256=contract_sha256,
        batches=batches,
    )


def _row_outcome_payload(outcome: RowEnumerationOutcome) -> dict[str, Any]:
    return {
        "records": {
            row_id: record.model_dump(mode="json", by_alias=True, exclude_none=True)
            for row_id, record in sorted(outcome.records.items())
        },
        "calls": [call.model_dump(mode="json", exclude_none=True) for call in outcome.calls],
        "attempts": [attempt.model_dump(mode="json") for attempt in outcome.attempts],
        "unresolved_row_ids": outcome.unresolved_row_ids,
        "unbatchable_row_ids": outcome.unbatchable_row_ids,
        "unknown_row_ids": outcome.unknown_row_ids,
        "invalid_row_reasons": outcome.invalid_row_reasons,
        "warnings": outcome.warnings,
        "telemetry": outcome.telemetry,
    }


def _validated_row_checkpoint_entry(
    entry: Any,
    *,
    batch: RowBatch,
) -> RowEnumerationOutcome | None:
    """Rehydrate one bounded batch without trusting its private checkpoint JSON."""

    batch_sha256 = sha256_bytes(canonical_json_bytes(batch))
    if not isinstance(entry, dict) or entry.get("batch_sha256") != batch_sha256:
        return None
    row_ids = {row.row_id for row in batch.rows}
    raw_records = entry.get("records")
    list_fields = (
        "calls",
        "attempts",
        "unresolved_row_ids",
        "unbatchable_row_ids",
        "unknown_row_ids",
        "warnings",
    )
    if not isinstance(raw_records, dict) or any(
        not isinstance(entry.get(name), list) for name in list_fields
    ):
        return None
    if not isinstance(entry.get("invalid_row_reasons"), dict):
        return None
    try:
        outcome = RowEnumerationOutcome(
            records={
                row_id: RowDispositionRecord.model_validate(record)
                for row_id, record in raw_records.items()
            },
            calls=[ProviderCall.model_validate(call) for call in entry["calls"]],
            attempts=[RowAttemptTelemetry.model_validate(item) for item in entry["attempts"]],
            unresolved_row_ids=list(entry["unresolved_row_ids"]),
            unbatchable_row_ids=list(entry["unbatchable_row_ids"]),
            unknown_row_ids=list(entry["unknown_row_ids"]),
            invalid_row_reasons=dict(entry["invalid_row_reasons"]),
            warnings=list(entry["warnings"]),
        )
    except (TypeError, ValueError):
        return None
    record_ids = set(outcome.records)
    unresolved_ids = set(outcome.unresolved_row_ids)
    if (
        len(unresolved_ids) != len(outcome.unresolved_row_ids)
        or record_ids & unresolved_ids
        or record_ids | unresolved_ids != row_ids
        or outcome.unbatchable_row_ids
        or len(outcome.attempts) > 3
        or len(outcome.calls) > 3
        or any(
            not isinstance(row_id, str) or not isinstance(reason, str)
            for row_id, reason in outcome.invalid_row_reasons.items()
        )
        or set(outcome.invalid_row_reasons) - row_ids
    ):
        return None
    if any(record.row_id != row_id for row_id, record in outcome.records.items()):
        return None
    return outcome


def _merge_row_outcome(
    target: RowEnumerationOutcome,
    source: RowEnumerationOutcome,
) -> None:
    overlap = set(target.records) & set(source.records)
    if overlap:
        raise ValueError(f"row outcome contains duplicate planned IDs: {sorted(overlap)!r}")
    target.records.update(source.records)
    target.calls.extend(source.calls)
    target.attempts.extend(source.attempts)
    target.unresolved_row_ids.extend(source.unresolved_row_ids)
    target.unbatchable_row_ids.extend(source.unbatchable_row_ids)
    target.unknown_row_ids.extend(source.unknown_row_ids)
    target.invalid_row_reasons.update(source.invalid_row_reasons)
    target.warnings.extend(source.warnings)


def _extractor_failure(
    error: Exception,
) -> tuple[str, ProviderCall | None, dict[str, int]]:
    """Reduce an extractor exception to bounded, secret-free run telemetry."""

    if isinstance(error, ProviderResponseValidationError):
        allowed_codes = {"invalid_json", "schema_validation", "wire_validation"}
        validation_code = error.code if error.code in allowed_codes else "unknown"
        return f"provider_response_{validation_code}", error.call, {}
    if isinstance(error, ProviderRequestRejectedError):
        return "provider_request_rejected", None, {"http_status": error.status_code}
    return "extractor_block_failed", None, {}


def _recover_split_block(
    *,
    client: OpenRouterClient,
    settings: PipelineSettings,
    spec: PaperSpec,
    block: ResultBlock,
    max_depth: int = LEGACY_RECOVERY_MAX_DEPTH,
) -> LegacyRecoveryOutcome:
    """Recover a content-invalid block through a bounded changed-input split tree.

    Only a completed response-validation failure may recurse. Request rejection and
    transport failure are terminal, and exception text is never copied into the run.
    Successful siblings remain usable even when another subtree terminates.
    """

    if max_depth < 1:
        raise ValueError("legacy recovery max depth must be positive")

    outcome = LegacyRecoveryOutcome()

    def terminal_failure(
        *,
        failed_block: ResultBlock,
        depth: int,
        error: Exception,
        terminal_reason: str,
    ) -> None:
        error_code, completed_call, safe_details = _extractor_failure(error)
        if completed_call is not None:
            outcome.calls.append(completed_call)
        outcome.terminal_failures.append(
            LegacyRecoveryFailure(
                block_id=failed_block.block_id,
                page=failed_block.page,
                depth=depth,
                error_code=error_code,
                completed_provider_call=completed_call is not None,
                terminal_reason=terminal_reason,
                safe_details=safe_details,
            )
        )

    def visit(child: ResultBlock, depth: int) -> None:
        outcome.max_depth_reached = max(outcome.max_depth_reached, depth)
        try:
            child_candidates, call, child_warnings = extract_page_candidates(
                client=client,
                model=settings.model,
                paper_id=spec.paper_id,
                paper_title=spec.title,
                fragment=_block_fragment(child),
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                reasoning_effort=settings.reasoning_effort,
                seed=settings.seed,
            )
        except ProviderResponseValidationError as error:
            # The exception carries a completed, secret-free ProviderCall. Retain it
            # even though its response did not validate locally.
            descendants = split_result_block(child)
            if depth < max_depth and descendants:
                outcome.calls.append(error.call)
                for descendant in descendants:
                    visit(descendant, depth + 1)
                return
            terminal_failure(
                failed_block=child,
                depth=depth,
                error=error,
                terminal_reason=("unsplittable" if not descendants else "max_depth_reached"),
            )
        except ProviderRequestRejectedError as error:
            terminal_failure(
                failed_block=child,
                depth=depth,
                error=error,
                terminal_reason="request_rejected",
            )
        except RuntimeError as error:
            terminal_failure(
                failed_block=child,
                depth=depth,
                error=error,
                terminal_reason="transport_failure",
            )
        else:
            outcome.candidates.extend(child_candidates)
            outcome.calls.append(call)
            outcome.successful_calls.append(call)
            outcome.warnings.extend(child_warnings)

    children = split_result_block(block)
    if not children:
        outcome.terminal_failures.append(
            LegacyRecoveryFailure(
                block_id=block.block_id,
                page=block.page,
                depth=0,
                error_code="recovery_unsplittable",
                completed_provider_call=False,
                terminal_reason="unsplittable",
            )
        )
        return outcome
    for child in children:
        visit(child, 1)
    return outcome


def run_paper(
    *,
    spec: PaperSpec,
    settings: PipelineSettings,
    client: OpenRouterClient,
) -> dict[str, Any]:
    """Run every stage for one paper and return a compact public-safe summary."""

    started = time.monotonic()
    output_dir = settings.output_root / spec.paper_id
    _clear_paper_run_outputs(output_dir)
    manifest = freeze_paper(spec, settings)
    paper_source = next(source for source in manifest.sources if source.role == SourceRole.PAPER)
    pdf_path = resolve_cached_path(paper_source, settings.project_root)
    layout = extract_pdf_layout(pdf_path, paper_source.source_id)
    write_json(output_dir / "private" / "layout.json", layout)
    selected_pages = _select_pages(layout, spec)
    block_config = ResultBlockConfig(max_blocks_per_page=settings.max_blocks_per_page)
    blocks = [
        block
        for page in selected_pages
        for block in segment_page_result_blocks(
            page,
            config=block_config,
        )
    ]
    write_json(output_dir / "private" / "result-blocks.json", blocks)
    code_state = _code_state(settings.project_root)
    row_plan: RowEnumerationPlan | None = None
    row_preflight: dict[str, Any] | None = None
    row_plan_sha256: str | None = None
    row_checkpoint_contract_sha256: str | None = None
    row_checkpoint_path = output_dir / "private" / "row-enumeration-checkpoint.json"
    row_checkpoint: dict[str, Any] | None = None
    row_checkpoint_batches: dict[str, Any] = {}
    row_outcome = RowEnumerationOutcome()
    row_batches_resumed = 0
    if settings.row_enumeration_enabled:
        row_plan = build_row_enumeration_plan(
            layout,
            blocks,
            config=settings.row_enumeration_config,
        )
        row_plan_path = output_dir / "private" / "row-enumeration-plan.json"
        row_plan_sha256 = write_json(row_plan_path, row_plan)
        row_preflight = _row_preflight(
            settings=settings,
            block_count=len(blocks),
            plan=row_plan,
        )
        write_json(output_dir / "private" / "row-enumeration-preflight.json", row_preflight)
        contract = _row_checkpoint_contract(
            spec=spec,
            settings=settings,
            manifest=manifest,
            layout=layout,
            plan_sha256=row_plan_sha256,
            code_state=code_state,
        )
        row_checkpoint_contract_sha256 = sha256_bytes(canonical_json_bytes(contract))
        row_checkpoint = _load_row_checkpoint(
            row_checkpoint_path,
            contract=contract,
            contract_sha256=row_checkpoint_contract_sha256,
            batches=row_plan.batches,
        )
        write_json(row_checkpoint_path, row_checkpoint)
        row_checkpoint_batches = row_checkpoint["batches"]
        row_outcome.unbatchable_row_ids = [item.row_id for item in row_plan.unbatchable_rows]
    checkpoint_contract = _extractor_checkpoint_contract(
        spec=spec,
        settings=settings,
        manifest=manifest,
        layout=layout,
        block_config=block_config,
        blocks=blocks,
        code_state=code_state,
    )
    checkpoint_contract_sha256 = sha256_bytes(canonical_json_bytes(checkpoint_contract))
    checkpoint_path = output_dir / "private" / "extractor-checkpoint.json"
    checkpoint = _load_extractor_checkpoint(
        checkpoint_path,
        contract=checkpoint_contract,
        contract_sha256=checkpoint_contract_sha256,
    )
    write_json(checkpoint_path, checkpoint)
    checkpoint_blocks: dict[str, Any] = checkpoint["blocks"]
    candidates: list[CandidateObservation] = []
    verifications: list[CandidateVerification] = []
    calls: list[ProviderCall] = []
    successful_new_calls: list[ProviderCall] = []
    resumed_calls: list[ProviderCall] = []
    verifier_calls: list[ProviderCall] = []
    block_attempts: list[dict[str, Any]] = []
    warnings: list[str] = []
    blocks_succeeded = 0
    blocks_failed = 0
    blocks_resumed = 0
    for block in blocks:
        fragment = _block_fragment(block)
        cached = _validated_checkpoint_entry(
            checkpoint_blocks.get(block.block_id),
            block=block,
        )
        if cached is not None:
            page_candidates, call, page_warnings = cached
            candidates.extend(page_candidates)
            resumed_calls.append(call)
            blocks_resumed += 1
            block_attempts.append(
                {
                    "block_id": block.block_id,
                    "page": block.page,
                    "status": "resumed",
                    "completed_provider_call": True,
                }
            )
            warnings.extend(
                f"page {fragment.page} block {block.block_id}: {warning}"
                for warning in page_warnings
            )
            continue
        try:
            page_candidates, call, page_warnings = extract_page_candidates(
                client=client,
                model=settings.model,
                paper_id=spec.paper_id,
                paper_title=spec.title,
                fragment=fragment,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                reasoning_effort=settings.reasoning_effort,
                seed=settings.seed,
            )
        except Exception as error:
            error_code, completed_call, safe_details = _extractor_failure(error)
            if completed_call is not None:
                calls.append(completed_call)
            recovery = None
            if isinstance(error, ProviderResponseValidationError):
                recovery = _recover_split_block(
                    client=client,
                    settings=settings,
                    spec=spec,
                    block=block,
                )
            recovery_metadata: dict[str, Any] = {}
            if recovery is not None:
                candidates.extend(recovery.candidates)
                calls.extend(recovery.calls)
                successful_new_calls.extend(recovery.successful_calls)
                warnings.extend(
                    f"page {fragment.page} block {block.block_id}: {warning}"
                    for warning in recovery.warnings
                )
                recovery_metadata = {
                    "recovery_calls": len(recovery.calls),
                    "recovery_successful_calls": len(recovery.successful_calls),
                    "recovery_validation_failed_calls": (
                        len(recovery.calls) - len(recovery.successful_calls)
                    ),
                    "recovery_max_depth_reached": recovery.max_depth_reached,
                    "recovery_terminal_failures": [
                        asdict(failure) for failure in recovery.terminal_failures
                    ],
                }
            if recovery is None or not recovery.succeeded:
                blocks_failed += 1
                block_attempts.append(
                    {
                        "block_id": block.block_id,
                        "page": block.page,
                        "status": "failed",
                        "error_code": error_code,
                        "completed_provider_call": completed_call is not None,
                        **recovery_metadata,
                        **safe_details,
                    }
                )
                warnings.append(
                    f"page {fragment.page} block {block.block_id}: extractor_error={error_code}"
                )
                continue
            blocks_succeeded += 1
            block_attempts.append(
                {
                    "block_id": block.block_id,
                    "page": block.page,
                    "status": "recovered_by_split",
                    "error_code": error_code,
                    "completed_provider_call": True,
                    **recovery_metadata,
                    **safe_details,
                }
            )
            warnings.append(
                f"page {fragment.page} block {block.block_id}: "
                f"extractor_error={error_code} recovered_by_split"
            )
            checkpoint_blocks[block.block_id] = {
                "block_text_sha256": block.text_sha256,
                "candidates": [
                    candidate.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for candidate in recovery.candidates
                ],
                "call": recovery.successful_calls[-1].model_dump(mode="json"),
                "warnings": recovery.warnings,
            }
            write_json(checkpoint_path, checkpoint)
            continue
        candidates.extend(page_candidates)
        calls.append(call)
        successful_new_calls.append(call)
        blocks_succeeded += 1
        block_attempts.append(
            {
                "block_id": block.block_id,
                "page": block.page,
                "status": "success",
                "completed_provider_call": True,
            }
        )
        warnings.extend(
            f"page {fragment.page} block {block.block_id}: {warning}" for warning in page_warnings
        )
        checkpoint_blocks[block.block_id] = {
            "block_text_sha256": block.text_sha256,
            "candidates": [
                candidate.model_dump(mode="json", by_alias=True, exclude_none=True)
                for candidate in page_candidates
            ],
            "call": call.model_dump(mode="json"),
            "warnings": page_warnings,
        }
        write_json(checkpoint_path, checkpoint)

    if row_plan is not None:
        assert row_checkpoint is not None
        for batch in row_plan.batches:
            cached_row = _validated_row_checkpoint_entry(
                row_checkpoint_batches.get(batch.batch_id),
                batch=batch,
            )
            if cached_row is not None:
                _merge_row_outcome(row_outcome, cached_row)
                row_batches_resumed += 1
                continue
            batch_outcome = enumerate_row_batch(
                client=client,
                model=settings.model,
                paper_id=spec.paper_id,
                paper_title=spec.title,
                batch=batch,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                reasoning_effort=settings.reasoning_effort,
                seed=settings.seed,
                max_recovery_depth=row_plan.config.max_recovery_depth,
            )
            _merge_row_outcome(row_outcome, batch_outcome)
            row_checkpoint_batches[batch.batch_id] = {
                "batch_sha256": sha256_bytes(canonical_json_bytes(batch)),
                **_row_outcome_payload(batch_outcome),
            }
            write_json(row_checkpoint_path, row_checkpoint)
        row_outcome.unresolved_row_ids = list(dict.fromkeys(row_outcome.unresolved_row_ids))
        row_outcome.unbatchable_row_ids = list(dict.fromkeys(row_outcome.unbatchable_row_ids))
        row_outcome.unknown_row_ids = list(dict.fromkeys(row_outcome.unknown_row_ids))
        candidates.extend(row_outcome.candidates)
        write_json(
            output_dir / "private" / "row-enumeration.json",
            {
                "schema_version": "row-enumeration-outcome/0.1",
                "plan_sha256": row_plan_sha256,
                **_row_outcome_payload(row_outcome),
            },
        )
        if row_outcome.unresolved_row_ids:
            warnings.append(f"row_enumeration_unresolved={len(row_outcome.unresolved_row_ids)}")
        if row_outcome.unbatchable_row_ids:
            warnings.append(f"row_enumeration_unbatchable={len(row_outcome.unbatchable_row_ids)}")
        if row_outcome.unknown_row_ids:
            warnings.append(f"row_enumeration_unknown_ids={len(row_outcome.unknown_row_ids)}")

    candidates_before_deduplication = len(candidates)
    candidates = validate_candidates(
        candidates,
        {layout.source_id: layout},
        min_confidence=settings.min_confidence,
    )
    candidates = deduplicate_candidates(candidates, {layout.source_id: layout})
    duplicates_removed = candidates_before_deduplication - len(candidates)
    candidates = validate_candidates(
        candidates,
        {layout.source_id: layout},
        min_confidence=settings.min_confidence,
    )
    if settings.verifier_model:
        for candidate in candidates:
            if (
                candidate.claim_type != ClaimType.PRIMARY_RESULT
                or candidate.export_status != ExportStatus.ELIGIBLE
            ):
                continue
            support = bind_candidate_block(candidate, blocks)
            if support is None:
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                candidate.export_reason = "no frozen result block contains the evidence quote"
                continue
            block, anchor = support
            evidence_block = frozen_evidence_block(
                paper_id=spec.paper_id, block=block, anchor=anchor
            )
            verification, verification_call = verify_candidate(
                client=client,
                model=settings.verifier_model,
                candidate=candidate,
                evidence_block=evidence_block,
                max_tokens=settings.verifier_max_tokens,
            )
            verifications.append(verification)
            verifier_calls.append(verification_call)
            if verification.decision != IndependentDecision.ACCEPT:
                candidate.export_status = ExportStatus.NEEDS_REVIEW
                candidate.export_reason = (
                    f"independent_verifier={verification.decision.value}: "
                    f"{verification.justification}"
                )
    schema, authority = load_schema(settings.schema_path, settings.schema_sha256)
    records = compose_eee_records(
        manifest=manifest,
        candidates=candidates,
        schema_version=authority.version,
    )
    validation_errors: dict[str, list[str]] = {}
    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    eee_dir = output_dir / "eee"
    eee_dir.mkdir(parents=True, exist_ok=True)
    for stale_record in eee_dir.glob("*.json"):
        stale_record.unlink()
    invalid_output_path = output_dir / "private" / "invalid-eee.json"
    if invalid_output_path.exists():
        invalid_output_path.unlink()
    for record in records:
        issues = validate_eee_record(record, schema)
        evaluation_id = record["evaluation_id"]
        validation_errors[evaluation_id] = [f"{issue.path}: {issue.message}" for issue in issues]
        if issues:
            invalid_records.append(record)
            observation_ids = {
                result["evaluation_result_id"] for result in record["evaluation_results"]
            }
            for candidate in candidates:
                if candidate.observation_id in observation_ids:
                    candidate.export_status = ExportStatus.NEEDS_REVIEW
                    candidate.export_reason = "projected EEE record failed schema validation"
            continue
        valid_records.append(record)
        filename = evaluation_id.rsplit("/", 1)[-1] + ".json"
        write_json(eee_dir / filename, record)
    if invalid_records:
        write_json(invalid_output_path, invalid_records)
    spot_checks = score_spot_checks(spec.expected_spot_checks, candidates)
    reference_score: dict[str, Any] | None = None
    reference_score_sha256: str | None = None
    if spec.reference_path:
        reference = load_reference(_reference_path(settings.project_root, spec.reference_path))
        if reference.paper_id != spec.paper_id:
            raise ValueError(
                f"reference paper_id mismatch: {reference.paper_id!r} != {spec.paper_id!r}"
            )
        if reference.source_sha256 != paper_source.sha256:
            raise ValueError("reference source hash does not match frozen paper")
        reference_score = score_reference(
            reference,
            candidates,
            control_examination(reference, layout, blocks),
        )
        reference_score_sha256 = write_json(output_dir / "reference-score.json", reference_score)
    write_jsonl(output_dir / "observations.jsonl", candidates)
    write_jsonl(output_dir / "verifications.jsonl", verifications)
    write_json(
        output_dir / "spot-checks.json",
        [
            {
                "expected": item.expected.model_dump(mode="json"),
                "matched_observation_id": item.matched_observation_id,
                "exact_value": item.exact_value,
                "exact_page": item.exact_page,
                "notes": item.notes,
            }
            for item in spot_checks
        ],
    )
    render_review_report(
        manifest=manifest,
        candidates=candidates,
        eee_records=valid_records,
        validation_errors=validation_errors,
        output_path=output_dir / "review.html",
    )
    review_reasons: list[str] = []
    if not blocks:
        review_reasons.append("zero_selected_result_blocks")
    elif not candidates:
        review_reasons.append("selected_result_blocks_produced_zero_candidates")
    if not valid_records:
        review_reasons.append("zero_valid_eee_records")
    if invalid_records:
        review_reasons.append("eee_schema_validation_failure")
    if row_plan is not None:
        if row_outcome.unresolved_row_ids:
            review_reasons.append("row_enumeration_unresolved")
        if row_outcome.unbatchable_row_ids:
            review_reasons.append("row_enumeration_unbatchable")
        if row_outcome.unknown_row_ids:
            review_reasons.append("row_enumeration_unknown_ids")
        if any(
            record.disposition is RowDisposition.UNCERTAIN
            for record in row_outcome.records.values()
        ):
            review_reasons.append("row_enumeration_uncertain")
    candidates_needing_review = sum(
        candidate.export_status == ExportStatus.NEEDS_REVIEW for candidate in candidates
    )
    semantic_safety_reviews = sum(
        any(note.startswith("semantic safety:") for note in candidate.notes)
        for candidate in candidates
        if candidate.export_status == ExportStatus.NEEDS_REVIEW
    )
    if candidates_needing_review:
        review_reasons.append("candidate_review_required")
    if review_reasons:
        warnings.extend(f"paper_review_required={reason}" for reason in review_reasons)
    row_stage_incomplete = bool(row_outcome.unresolved_row_ids or row_outcome.unbatchable_row_ids)
    run_status = (
        "partial_failure"
        if blocks_failed or row_stage_incomplete
        else "quality_failure"
        if invalid_records
        else "success"
    )
    run_manifest = {
        "schema_version": "pipeline-run/0.2",
        "status": run_status,
        "paper_id": spec.paper_id,
        "title": spec.title,
        "review_state": {
            "status": "needs_review" if review_reasons else "ready",
            "reasons": review_reasons,
        },
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "selected_pages": [fragment.page for fragment in selected_pages],
        "result_block_segmentation": asdict(block_config),
        "selected_blocks": [
            {
                "block_id": block.block_id,
                "page": block.page,
                "body_lines": [block.body_start_line, block.body_end_line],
                "context_lines": (
                    [block.context_start_line, block.context_end_line]
                    if block.context_start_line is not None
                    else None
                ),
            }
            for block in blocks
        ],
        "layout_parser": layout.parser,
        "layout_parser_version": layout.parser_version,
        "extractor": {
            **_extractor_run_configuration(settings),
            "calls": [call.model_dump(mode="json", exclude_none=True) for call in calls],
            "resumed_calls": [
                call.model_dump(mode="json", exclude_none=True) for call in resumed_calls
            ],
            "execution": {
                "blocks_total": len(blocks),
                "blocks_succeeded": blocks_succeeded,
                "blocks_failed": blocks_failed,
                "blocks_resumed": blocks_resumed,
                "calls_succeeded": blocks_succeeded,
                "calls_failed": blocks_failed,
                "calls_resumed": blocks_resumed,
            },
            "successful_call_telemetry": _provider_call_telemetry(
                [*successful_new_calls, *resumed_calls]
            ),
            "block_attempts": block_attempts,
            "checkpoint": {
                "schema_version": _EXTRACTOR_CHECKPOINT_SCHEMA_VERSION,
                "contract_sha256": checkpoint_contract_sha256,
                "path": "private/extractor-checkpoint.json",
            },
        },
        "row_enumeration": {
            **_row_enumeration_run_configuration(settings),
            "plan_sha256": row_plan_sha256,
            "plan": row_plan.telemetry.model_dump(mode="json") if row_plan else None,
            "preflight": row_preflight,
            "outcome": row_outcome.telemetry,
            "calls": [
                call.model_dump(mode="json", exclude_none=True) for call in row_outcome.calls
            ],
            "successful_call_telemetry": _provider_call_telemetry(
                row_outcome.calls,
                basis=(
                    "completed bounded row-enumeration calls; cost, token, retry, and "
                    "attempt totals are lower bounds when provider metadata is unavailable"
                ),
            ),
            "attempts": [attempt.model_dump(mode="json") for attempt in row_outcome.attempts],
            "execution": {
                "batches_total": len(row_plan.batches) if row_plan else 0,
                "batches_resumed": row_batches_resumed,
                "batches_executed": (
                    len(row_plan.batches) - row_batches_resumed if row_plan else 0
                ),
                "invalid_rows_seen": len(row_outcome.invalid_row_reasons),
                "unknown_row_ids_seen": len(row_outcome.unknown_row_ids),
            },
            "checkpoint": (
                {
                    "schema_version": _ROW_CHECKPOINT_SCHEMA_VERSION,
                    "contract_sha256": row_checkpoint_contract_sha256,
                    "path": "private/row-enumeration-checkpoint.json",
                }
                if row_plan is not None
                else None
            ),
        },
        "verifier": {
            **_verifier_run_configuration(settings),
            "calls": [call.model_dump(mode="json", exclude_none=True) for call in verifier_calls],
        },
        "eee_schema": {"version": authority.version, "sha256": authority.sha256},
        "code": code_state,
        "counts": {
            "candidates": len(candidates),
            "candidates_before_deduplication": candidates_before_deduplication,
            "duplicates_removed": duplicates_removed,
            "candidates_needing_review": candidates_needing_review,
            "semantic_safety_reviews": semantic_safety_reviews,
            "primary_results": sum(c.claim_type == "primary_result" for c in candidates),
            "exported": sum(c.export_status == "exported" for c in candidates),
            "eee_records": len(valid_records),
            "eee_schema_issues": sum(len(items) for items in validation_errors.values()),
            "verifications": len(verifications),
            "verifier_accepts": sum(
                item.decision == IndependentDecision.ACCEPT for item in verifications
            ),
            "verifier_rejects": sum(
                item.decision == IndependentDecision.REJECT for item in verifications
            ),
            "verifier_reviews": sum(
                item.decision == IndependentDecision.REVIEW for item in verifications
            ),
            "spot_checks": len(spot_checks),
            "spot_checks_exact": sum(item.exact_value for item in spot_checks),
            "reference_observations": (
                reference_score["reference_observations"] if reference_score else 0
            ),
            "reference_true_positives": (
                reference_score["detection"]["true_positives"] if reference_score else 0
            ),
            "reference_false_positives": (
                reference_score["detection"]["false_positives"] if reference_score else 0
            ),
            "reference_false_negatives": (
                reference_score["detection"]["false_negatives"] if reference_score else 0
            ),
            "negative_control_false_primary": (
                reference_score.get("negative_control_safety", {}).get("false_primary_count", 0)
                if reference_score
                else 0
            ),
        },
        "reference_evaluation": (
            {
                "path": spec.reference_path,
                "score_path": "reference-score.json",
                "score_sha256": reference_score_sha256,
                "schema_version": reference_score["schema_version"],
                "coverage": reference_score["coverage"],
                "detection": reference_score["detection"],
                "field_accuracy": reference_score["field_accuracy"],
                "negative_control_safety": reference_score.get("negative_control_safety"),
            }
            if reference_score
            else None
        ),
        "warnings": warnings,
        "wall_clock_seconds": round(time.monotonic() - started, 6),
    }
    write_json(output_dir / "run.json", run_manifest)
    return run_manifest


def run_corpus(
    *,
    corpus: CorpusSpec,
    settings: PipelineSettings,
    client: OpenRouterClient,
) -> dict[str, Any]:
    corpus_started = time.monotonic()
    _, schema_authority = load_schema(settings.schema_path, settings.schema_sha256)
    failure_code_state = _code_state(settings.project_root)
    failure_block_config = asdict(
        ResultBlockConfig(max_blocks_per_page=settings.max_blocks_per_page)
    )
    _clear_corpus_run_outputs(settings.output_root)
    summaries: list[dict[str, Any]] = []
    for paper in corpus.papers:
        started = time.monotonic()
        try:
            summary = run_paper(spec=paper, settings=settings, client=client)
        except Exception as error:
            paper_root = settings.output_root / paper.paper_id
            _clear_paper_run_outputs(paper_root)
            source_manifest_path = paper_root / "source-manifest.json"
            failure_warnings = ["paper_review_required=paper_run_error"]
            reference_score: dict[str, Any] | None = None
            reference_score_sha256: str | None = None
            if source_manifest_path.is_file() and not source_manifest_path.is_symlink():
                try:
                    reference_score, reference_score_sha256 = _score_reference_after_paper_error(
                        spec=paper,
                        settings=settings,
                        manifest_path=source_manifest_path,
                        output_path=paper_root / "reference-score.json",
                    )
                except (OSError, TypeError, ValueError, StopIteration):
                    failure_warnings.append("reference_score_unavailable_after_paper_error")
            summary = {
                "schema_version": "pipeline-run/0.2",
                "status": "error",
                "paper_id": paper.paper_id,
                "title": paper.title,
                "selected_pages": [],
                "selected_blocks": [],
                "result_block_segmentation": failure_block_config,
                "layout_parser": None,
                "layout_parser_version": None,
                "extractor": {
                    **_extractor_run_configuration(settings),
                    "calls": [],
                    "resumed_calls": [],
                    "execution": {
                        "blocks_total": 0,
                        "blocks_succeeded": 0,
                        "blocks_failed": 0,
                        "blocks_resumed": 0,
                        "calls_succeeded": 0,
                        "calls_failed": 0,
                        "calls_resumed": 0,
                    },
                    "successful_call_telemetry": _provider_call_telemetry([]),
                    "block_attempts": [],
                },
                "row_enumeration": {
                    **_row_enumeration_run_configuration(settings),
                    "plan_sha256": None,
                    "plan": None,
                    "preflight": None,
                    "outcome": RowEnumerationOutcome().telemetry,
                    "calls": [],
                    "successful_call_telemetry": _provider_call_telemetry(
                        [],
                        basis=(
                            "completed bounded row-enumeration calls; cost, token, retry, "
                            "and attempt totals are lower bounds"
                        ),
                    ),
                    "attempts": [],
                    "execution": {
                        "batches_total": 0,
                        "batches_resumed": 0,
                        "batches_executed": 0,
                        "invalid_rows_seen": 0,
                        "unknown_row_ids_seen": 0,
                    },
                    "checkpoint": None,
                },
                "verifier": {**_verifier_run_configuration(settings), "calls": []},
                "eee_schema": {
                    "version": schema_authority.version,
                    "sha256": schema_authority.sha256,
                },
                "code": failure_code_state,
                "counts": {
                    "candidates": 0,
                    "candidates_before_deduplication": 0,
                    "duplicates_removed": 0,
                    "candidates_needing_review": 0,
                    "semantic_safety_reviews": 0,
                    "primary_results": 0,
                    "exported": 0,
                    "eee_records": 0,
                    "eee_schema_issues": 0,
                    "verifications": 0,
                    "verifier_accepts": 0,
                    "verifier_rejects": 0,
                    "verifier_reviews": 0,
                    "spot_checks": 0,
                    "spot_checks_exact": 0,
                    "reference_observations": (
                        reference_score["reference_observations"] if reference_score else 0
                    ),
                    "reference_true_positives": (
                        reference_score["detection"]["true_positives"] if reference_score else 0
                    ),
                    "reference_false_positives": (
                        reference_score["detection"]["false_positives"] if reference_score else 0
                    ),
                    "reference_false_negatives": (
                        reference_score["detection"]["false_negatives"] if reference_score else 0
                    ),
                    "negative_control_false_primary": (
                        reference_score.get("negative_control_safety", {}).get(
                            "false_primary_count", 0
                        )
                        if reference_score
                        else 0
                    ),
                },
                "error": {
                    "type": type(error).__name__,
                    "message": _safe_error_message(error),
                },
                "wall_clock_seconds": round(time.monotonic() - started, 6),
                "review_state": {
                    "status": "blocked",
                    "reasons": ["paper_run_error"],
                },
                "reference_evaluation": (
                    {
                        "path": paper.reference_path,
                        "score_path": "reference-score.json",
                        "score_sha256": reference_score_sha256,
                        "schema_version": reference_score["schema_version"],
                        "coverage": reference_score["coverage"],
                        "detection": reference_score["detection"],
                        "field_accuracy": reference_score["field_accuracy"],
                        "negative_control_safety": reference_score.get("negative_control_safety"),
                    }
                    if reference_score
                    else None
                ),
                "warnings": failure_warnings,
            }
            if source_manifest_path.is_file() and not source_manifest_path.is_symlink():
                summary["source_manifest_sha256"] = sha256_file(source_manifest_path)
            write_jsonl(paper_root / "observations.jsonl", [])
            write_jsonl(paper_root / "verifications.jsonl", [])
            write_json(paper_root / "spot-checks.json", [])
            write_json(paper_root / "run.json", summary)
        summaries.append(summary)
    succeeded = [summary for summary in summaries if summary.get("status") == "success"]
    failed = [summary for summary in summaries if summary.get("status") != "success"]
    reference_scores: list[dict[str, Any]] = []
    for summary in summaries:
        reference_evaluation = summary.get("reference_evaluation")
        if not isinstance(reference_evaluation, dict):
            continue
        expected_sha256 = reference_evaluation.get("score_sha256")
        if not isinstance(expected_sha256, str):
            continue
        score_path = settings.output_root / summary["paper_id"] / "reference-score.json"
        try:
            if sha256_file(score_path) != expected_sha256:
                continue
            score = read_json(score_path)
        except (OSError, ValueError):
            continue
        if isinstance(score, dict):
            reference_scores.append(score)
    corpus_evaluation = aggregate_reference_scores(reference_scores) if reference_scores else None
    if corpus_evaluation:
        write_json(settings.output_root / "corpus-evaluation.json", corpus_evaluation)
    report = {
        "schema_version": "corpus-run/0.2",
        "corpus_id": corpus.corpus_id,
        "corpus_binding": build_corpus_binding(corpus),
        "status": "success" if not failed else "error" if not succeeded else "partial_failure",
        "generated_at": datetime.now(UTC).isoformat(),
        "papers": len(summaries),
        "papers_succeeded": len(succeeded),
        "papers_failed": len(failed),
        "papers_with_eee": sum(summary["counts"]["eee_records"] > 0 for summary in succeeded),
        "papers_without_candidates": sum(
            summary["counts"].get("candidates", 0) == 0 for summary in succeeded
        ),
        "papers_without_eee": sum(
            summary["counts"].get("eee_records", 0) == 0 for summary in succeeded
        ),
        "papers_needing_review": sum(
            summary.get("review_state", {}).get("status") != "ready" for summary in summaries
        ),
        "reference_evaluation": corpus_evaluation,
        "totals": {
            key: sum(summary["counts"].get(key, 0) for summary in summaries)
            for key in (
                "candidates",
                "candidates_before_deduplication",
                "duplicates_removed",
                "candidates_needing_review",
                "semantic_safety_reviews",
                "primary_results",
                "exported",
                "eee_records",
                "eee_schema_issues",
                "verifications",
                "verifier_accepts",
                "verifier_rejects",
                "verifier_reviews",
                "spot_checks",
                "spot_checks_exact",
                "reference_observations",
                "reference_true_positives",
                "reference_false_positives",
                "reference_false_negatives",
                "negative_control_false_primary",
            )
        },
        "runs": summaries,
    }
    report["operations"] = _corpus_operational_metrics(
        summaries,
        wall_clock_seconds=time.monotonic() - corpus_started,
    )
    write_json(settings.output_root / "corpus-run.json", report)
    render_corpus_html_file(
        settings.output_root / "corpus-run.json",
        settings.output_root / "corpus-review.html",
    )
    return report


def freeze_corpus(corpus: CorpusSpec, settings: PipelineSettings) -> dict[str, Any]:
    """Acquire and pin a corpus without contacting a model provider."""

    manifests: list[SourceManifest] = []
    results: list[dict[str, Any]] = []
    for paper in corpus.papers:
        try:
            manifest = freeze_paper(paper, settings)
        except Exception as error:
            results.append(
                {
                    "paper_id": paper.paper_id,
                    "status": "error",
                    "error": {
                        "type": type(error).__name__,
                        "message": _safe_error_message(error),
                    },
                }
            )
            continue
        manifests.append(manifest)
        results.append(
            {
                "paper_id": manifest.paper_id,
                "status": "success",
                "source_ids": [source.source_id for source in manifest.sources],
                "sha256": [source.sha256 for source in manifest.sources],
            }
        )

    failures = len(corpus.papers) - len(manifests)
    status = "success" if failures == 0 else "error" if not manifests else "partial_failure"
    summary = {
        "schema_version": "corpus-freeze/0.2",
        "corpus_id": corpus.corpus_id,
        "status": status,
        "papers": len(corpus.papers),
        "papers_succeeded": len(manifests),
        "papers_failed": failures,
        "sources": sum(len(manifest.sources) for manifest in manifests),
        "manifests": [
            {
                "paper_id": manifest.paper_id,
                "source_ids": [source.source_id for source in manifest.sources],
                "sha256": [source.sha256 for source in manifest.sources],
            }
            for manifest in manifests
        ],
        "results": results,
    }
    write_json(settings.output_root / "corpus-freeze.json", summary)
    return summary


def runtime_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY must be supplied in the runtime environment")
    return key
