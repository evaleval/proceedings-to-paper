"""Provider-free first-run planning over newly frozen corpus sources."""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from proceedings_to_eee.corpus import CorpusSpec, build_corpus_binding
from proceedings_to_eee.extraction.pdf_layout import extract_pdf_layout
from proceedings_to_eee.extraction.result_blocks import (
    LEGACY_RECOVERY_MAX_DEPTH,
    ResultBlock,
    ResultBlockConfig,
    maximum_legacy_block_invocations,
    segment_page_result_blocks,
)
from proceedings_to_eee.extraction.row_enumeration import (
    RowEnumerationPlan,
    build_row_enumeration_plan,
)
from proceedings_to_eee.io import read_json, sha256_file, write_json
from proceedings_to_eee.pipeline import (
    PipelineSettings,
    _code_state,
    _safe_error_message,
    _select_pages,
    freeze_corpus,
)
from proceedings_to_eee.sources.manifest import (
    SourceManifest,
    SourceRole,
    resolve_cached_path,
)

CORPUS_PREFLIGHT_SCHEMA_VERSION = "corpus-preflight/0.1"
PAPER_PREFLIGHT_SCHEMA_VERSION = "paper-preflight/0.1"


@dataclass(frozen=True, slots=True)
class PreflightCostAssumptions:
    """Optional user-supplied planning inputs, never provider prices."""

    legacy_call_usd: float | None = None
    row_call_usd: float | None = None
    tuple_call_usd: float | None = None
    verifier_call_usd: float | None = None
    origin_call_usd: float | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


def _clear_planning_artifacts(paper_root: Path) -> None:
    """Remove only preflight-owned files so a failed rerun cannot expose stale plans."""

    private_root = paper_root / "private"
    if private_root.is_symlink():
        raise ValueError("refusing to use a symlinked private preflight directory")
    for path in (
        paper_root / "preflight.json",
        private_root / "layout.json",
        private_root / "result-blocks.json",
        private_root / "row-enumeration-plan.json",
    ):
        path.unlink(missing_ok=True)


def _error_record(error: Exception, *, stage: str) -> dict[str, str]:
    if isinstance(error, FileNotFoundError) and (
        error.filename == "pdftotext" or "pdftotext" in str(error)
    ):
        return {
            "code": "missing_poppler",
            "stage": stage,
            "type": type(error).__name__,
            "message": "pdftotext is required for PDF layout extraction but was not found.",
        }
    if isinstance(error, subprocess.CalledProcessError):
        return {
            "code": "pdf_layout_failed",
            "stage": stage,
            "type": type(error).__name__,
            "message": "pdftotext could not parse the frozen PDF.",
        }
    return {
        "code": "source_freeze_failed" if stage == "source_freeze" else "planning_failed",
        "stage": stage,
        "type": type(error).__name__,
        "message": _safe_error_message(error),
    }


def _selected_page_record(page: Any, blocks: list[ResultBlock]) -> dict[str, Any]:
    page_blocks = [block for block in blocks if block.page == page.page]
    return {
        "page": page.page,
        "fragment_id": page.fragment_id,
        "text_sha256": page.text_sha256,
        "result_signal_score": page.result_signal_score,
        "numeric_token_count": page.numeric_token_count,
        "result_blocks": len(page_blocks),
        "block_ids": [block.block_id for block in page_blocks],
    }


def _block_record(block: ResultBlock) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "source_id": block.source_id,
        "page": block.page,
        "text_sha256": block.text_sha256,
        "body_start_line": block.body_start_line,
        "body_end_line": block.body_end_line,
        "source_column_start": block.source_column_start,
        "source_column_end": block.source_column_end,
        "character_count": block.character_count,
        "numeric_token_count": block.numeric_token_count,
        "data_row_count": block.data_row_count,
    }


def _row_record(row: Any) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "input_sha256": row.input_sha256,
        "source_id": row.source_id,
        "page": row.page,
        "region_id": row.region_id,
        "table_label": row.table_label,
        "value_tokens": len(row.values),
    }


def _paper_call_bound(
    *,
    blocks: list[ResultBlock],
    row_plan: RowEnumerationPlan,
    row_enumeration_enabled: bool,
) -> dict[str, Any]:
    legacy_initial = len(blocks)
    legacy_maximum = sum(maximum_legacy_block_invocations(block) for block in blocks)
    planned_row_base = row_plan.telemetry.expected_calls
    planned_row_maximum = row_plan.telemetry.maximum_calls
    active_row_base = planned_row_base if row_enumeration_enabled else 0
    active_row_maximum = planned_row_maximum if row_enumeration_enabled else 0
    return {
        "legacy": {
            "initial_calls": legacy_initial,
            "maximum_calls": legacy_maximum,
            "maximum_recovery_calls": legacy_maximum - legacy_initial,
            "max_recovery_depth": LEGACY_RECOVERY_MAX_DEPTH,
            "basis": (
                "One initial call per selected block plus the exact bounded binary split "
                "tree permitted by that block's body-line geometry."
            ),
        },
        "row": {
            "enabled_for_planned_run": row_enumeration_enabled,
            "planned_base_calls": planned_row_base,
            "planned_maximum_calls": planned_row_maximum,
            "active_base_calls": active_row_base,
            "active_maximum_calls": active_row_maximum,
            "max_recovery_depth": row_plan.config.max_recovery_depth,
            "basis": (
                "The row plan is always materialized. Its calls enter the configured bound "
                "only when row enumeration is enabled."
            ),
        },
        "configured_without_verifier": {
            "base_calls": legacy_initial + active_row_base,
            "maximum_calls": legacy_maximum + active_row_maximum,
        },
    }


def _stage_cost(
    *,
    calls: int,
    assumption: float | None,
) -> dict[str, Any]:
    return {
        "maximum_calls": calls,
        "user_supplied_cost_per_call_usd": assumption,
        "estimated_maximum_cost_usd": (
            round(calls * assumption, 12) if assumption is not None else None
        ),
    }


def _cost_report(
    *,
    planning_complete: bool,
    legacy_calls: int,
    row_calls: int,
    tuple_calls: int,
    verifier_calls: int,
    origin_calls: int,
    assumptions: PreflightCostAssumptions,
) -> dict[str, Any]:
    values = {
        "legacy": assumptions.legacy_call_usd,
        "row": assumptions.row_call_usd,
        "tuple": assumptions.tuple_call_usd,
        "verifier": assumptions.verifier_call_usd,
        "origin": assumptions.origin_call_usd,
    }
    calls = {
        "legacy": legacy_calls,
        "row": row_calls,
        "tuple": tuple_calls,
        "verifier": verifier_calls,
        "origin": origin_calls,
    }
    missing = [name for name, count in calls.items() if count and values[name] is None]
    complete = planning_complete and not missing
    stage_reports = {
        name: _stage_cost(calls=count, assumption=values[name]) for name, count in calls.items()
    }
    return {
        "available": complete,
        "planning_complete": planning_complete,
        "missing_user_assumptions": missing,
        "basis": (
            "Only explicit user-supplied per-structured-call assumptions are multiplied by "
            "the configured hard call bounds. These are planning estimates, not provider "
            "quotes or guarantees."
        ),
        "stages": stage_reports,
        "estimated_maximum_total_cost_usd": (
            round(
                sum(
                    report["estimated_maximum_cost_usd"]
                    for report in stage_reports.values()
                    if report["estimated_maximum_cost_usd"] is not None
                ),
                12,
            )
            if complete
            else None
        ),
    }


def preflight_corpus(
    *,
    corpus: CorpusSpec,
    settings: PipelineSettings,
    tuple_call_ceiling: int = 0,
    verifier_call_ceiling: int = 0,
    origin_call_ceiling: int = 0,
    cost_assumptions: PreflightCostAssumptions | None = None,
) -> dict[str, Any]:
    """Freeze and plan a new corpus without constructing or calling a model provider.

    Remote corpus sources can require ordinary source-download network access. This
    function never reads provider credentials and never creates an OpenRouter client.
    """

    for name, value in (
        ("tuple", tuple_call_ceiling),
        ("verifier", verifier_call_ceiling),
        ("origin", origin_call_ceiling),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} call ceiling must be a non-negative integer")
    if verifier_call_ceiling > tuple_call_ceiling:
        raise ValueError("verifier call ceiling cannot exceed tuple call ceiling")
    if origin_call_ceiling > verifier_call_ceiling:
        raise ValueError("origin call ceiling cannot exceed verifier call ceiling")
    assumptions = cost_assumptions or PreflightCostAssumptions()
    started = time.monotonic()
    settings.output_root.mkdir(parents=True, exist_ok=True)
    freeze_summary = freeze_corpus(corpus, settings)
    freeze_results = {item["paper_id"]: item for item in freeze_summary["results"]}
    block_config = ResultBlockConfig(max_blocks_per_page=settings.max_blocks_per_page)
    code_state = _code_state(settings.project_root)
    paper_reports: list[dict[str, Any]] = []

    for spec in corpus.papers:
        paper_root = settings.output_root / spec.paper_id
        freeze_result = freeze_results[spec.paper_id]
        try:
            _clear_planning_artifacts(paper_root)
        except Exception as error:
            report = {
                "schema_version": PAPER_PREFLIGHT_SCHEMA_VERSION,
                "status": "error",
                "paper_id": spec.paper_id,
                "error": _error_record(error, stage="artifact_preparation"),
                "provider_calls_made": 0,
            }
            write_json(paper_root / "preflight.json", report)
            paper_reports.append(report)
            continue
        if freeze_result["status"] != "success":
            freeze_error = freeze_result.get("error", {})
            report = {
                "schema_version": PAPER_PREFLIGHT_SCHEMA_VERSION,
                "status": "error",
                "paper_id": spec.paper_id,
                "error": {
                    "code": "source_freeze_failed",
                    "stage": "source_freeze",
                    "type": freeze_error.get("type", "SourceFreezeError"),
                    "message": freeze_error.get("message", "Source freeze failed."),
                },
                "provider_calls_made": 0,
            }
            write_json(paper_root / "preflight.json", report)
            paper_reports.append(report)
            continue

        try:
            manifest_path = paper_root / "source-manifest.json"
            manifest = SourceManifest.model_validate(read_json(manifest_path))
            paper_sources = [
                source for source in manifest.sources if source.role is SourceRole.PAPER
            ]
            if len(paper_sources) != 1:
                raise ValueError("expected exactly one frozen paper source")
            paper_source = paper_sources[0]
            pdf_path = resolve_cached_path(paper_source, settings.project_root)
            layout = extract_pdf_layout(pdf_path, paper_source.source_id)
            layout_path = paper_root / "private" / "layout.json"
            layout_sha256 = write_json(layout_path, layout)
            selected_pages = _select_pages(layout, spec)
            blocks = [
                block
                for page in selected_pages
                for block in segment_page_result_blocks(page, config=block_config)
            ]
            block_ids = [block.block_id for block in blocks]
            if len(block_ids) != len(set(block_ids)):
                raise ValueError("result-block segmentation produced duplicate block ids")
            blocks_path = paper_root / "private" / "result-blocks.json"
            blocks_sha256 = write_json(blocks_path, blocks)
            row_plan = build_row_enumeration_plan(
                layout,
                blocks,
                config=settings.row_enumeration_config,
            )
            row_plan_path = paper_root / "private" / "row-enumeration-plan.json"
            row_plan_sha256 = write_json(row_plan_path, row_plan)
            call_bound = _paper_call_bound(
                blocks=blocks,
                row_plan=row_plan,
                row_enumeration_enabled=settings.row_enumeration_enabled,
            )
            warnings = []
            if not selected_pages:
                warnings.append("zero_selected_pages")
            if not blocks:
                warnings.append("zero_selected_result_blocks")
            if not row_plan.rows:
                warnings.append("zero_dense_table_rows_planned")
            report = {
                "schema_version": PAPER_PREFLIGHT_SCHEMA_VERSION,
                "status": "success",
                "paper_id": spec.paper_id,
                "title": spec.title,
                "source_manifest_sha256": sha256_file(manifest_path),
                "layout_parser": layout.parser,
                "layout_parser_version": layout.parser_version,
                "page_count": layout.page_count,
                "selection": {
                    "configured_include_pages": spec.include_pages,
                    "max_result_pages": spec.max_result_pages,
                    "selected_pages": [
                        _selected_page_record(page, blocks) for page in selected_pages
                    ],
                },
                "result_block_segmentation": asdict(block_config),
                "selected_blocks": [_block_record(block) for block in blocks],
                "row_plan": {
                    "config": row_plan.config.model_dump(mode="json"),
                    "telemetry": row_plan.telemetry.model_dump(mode="json"),
                    "rows": [_row_record(row) for row in row_plan.rows],
                    "unbatchable_row_ids": [item.row_id for item in row_plan.unbatchable_rows],
                },
                "structured_call_bound": call_bound,
                "artifacts": {
                    "source_manifest": {
                        "path": "source-manifest.json",
                        "sha256": sha256_file(manifest_path),
                    },
                    "layout": {"path": "private/layout.json", "sha256": layout_sha256},
                    "result_blocks": {
                        "path": "private/result-blocks.json",
                        "sha256": blocks_sha256,
                    },
                    "row_plan": {
                        "path": "private/row-enumeration-plan.json",
                        "sha256": row_plan_sha256,
                    },
                },
                "provider_calls_made": 0,
                "warnings": warnings,
            }
        except Exception as error:
            report = {
                "schema_version": PAPER_PREFLIGHT_SCHEMA_VERSION,
                "status": "error",
                "paper_id": spec.paper_id,
                "error": _error_record(error, stage="pdf_planning"),
                "provider_calls_made": 0,
            }
        write_json(paper_root / "preflight.json", report)
        paper_reports.append(report)

    successful = [report for report in paper_reports if report["status"] == "success"]
    failed = [report for report in paper_reports if report["status"] != "success"]
    legacy_initial = sum(
        report["structured_call_bound"]["legacy"]["initial_calls"] for report in successful
    )
    legacy_maximum = sum(
        report["structured_call_bound"]["legacy"]["maximum_calls"] for report in successful
    )
    row_base = sum(
        report["structured_call_bound"]["row"]["active_base_calls"] for report in successful
    )
    row_maximum = sum(
        report["structured_call_bound"]["row"]["active_maximum_calls"] for report in successful
    )
    planned_row_maximum = sum(
        report["structured_call_bound"]["row"]["planned_maximum_calls"] for report in successful
    )
    planning_complete = not failed
    subset_base = (
        legacy_initial + row_base + tuple_call_ceiling + verifier_call_ceiling + origin_call_ceiling
    )
    subset_maximum = (
        legacy_maximum
        + row_maximum
        + tuple_call_ceiling
        + verifier_call_ceiling
        + origin_call_ceiling
    )
    call_bound = {
        "complete_for_corpus": planning_complete,
        "legacy_initial_calls": legacy_initial,
        "legacy_maximum_calls": legacy_maximum,
        "row_enumeration_enabled": settings.row_enumeration_enabled,
        "row_base_calls": row_base,
        "row_maximum_calls": row_maximum,
        "row_maximum_calls_if_enabled": planned_row_maximum,
        "tuple_call_ceiling": tuple_call_ceiling,
        "verifier_call_ceiling": verifier_call_ceiling,
        "origin_call_ceiling": origin_call_ceiling,
        "configured_base_structured_calls": subset_base if planning_complete else None,
        "maximum_structured_calls": subset_maximum if planning_complete else None,
        "planned_subset_base_structured_calls": subset_base,
        "planned_subset_maximum_structured_calls": subset_maximum,
        "basis": (
            "Legacy maximums include the exact bounded response-validation split tree for "
            "each selected block. Row maximums include the configured single split level "
            "only when row enumeration is enabled. Tuple, verifier, and origin ceilings are "
            "supplied by the user because candidates do not exist before extraction; the "
            "verifier stage is a subset of tuple-passed candidates and the origin stage is a "
            "subset of verifier-accepted candidates. Set the subsequent "
            "run's global --max-structured-calls no higher than this reported total when "
            "this is the intended campaign bound."
        ),
    }
    cost_report = _cost_report(
        planning_complete=planning_complete,
        legacy_calls=legacy_maximum,
        row_calls=row_maximum,
        tuple_calls=tuple_call_ceiling,
        verifier_calls=verifier_call_ceiling,
        origin_calls=origin_call_ceiling,
        assumptions=assumptions,
    )
    status = "success" if not failed else "error" if not successful else "partial_failure"
    report = {
        "schema_version": CORPUS_PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "corpus_id": corpus.corpus_id,
        "corpus_binding": build_corpus_binding(corpus),
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "new_corpus_provider_free_preflight",
        "provider_calls_made": 0,
        "provider_credentials_required": False,
        "source_network_access": (
            "Remote PDFs, supplements, or repository verification can use network access "
            "during source freeze; local pdf_path inputs do not."
        ),
        "papers": len(paper_reports),
        "papers_succeeded": len(successful),
        "papers_failed": len(failed),
        "counts": {
            "selected_pages": sum(len(item["selection"]["selected_pages"]) for item in successful),
            "result_blocks": sum(len(item["selected_blocks"]) for item in successful),
            "rows_planned": sum(
                item["row_plan"]["telemetry"]["rows_planned"] for item in successful
            ),
            "unbatchable_rows": sum(
                item["row_plan"]["telemetry"]["unbatchable_rows"] for item in successful
            ),
        },
        "structured_call_bound": call_bound,
        "cost_estimates": cost_report,
        "freeze_artifact": {
            "path": "corpus-freeze.json",
            "sha256": sha256_file(settings.output_root / "corpus-freeze.json"),
        },
        "code": code_state,
        "paper_reports": paper_reports,
        "wall_clock_seconds": round(time.monotonic() - started, 6),
    }
    write_json(settings.output_root / "corpus-preflight.json", report)
    return report
