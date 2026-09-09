"""Small, explicit command-line surface for reproducible pilot runs."""

from __future__ import annotations

import json
import shlex
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from proceedings_to_eee.census import CensusImportError, import_census
from proceedings_to_eee.census_atlas import build_atlas
from proceedings_to_eee.census_audit import (
    AUDIT_SCHEMA_NAME,
    audit_response_schema,
    run_audit,
)
from proceedings_to_eee.census_recompose import recompose_census
from proceedings_to_eee.census_review import (
    REVIEW_REASONING_EFFORT,
    REVIEW_SCHEMA_NAME,
    REVIEW_SEED,
    REVIEW_TEMPERATURE,
    build_label_packet,
    collect_review_items,
    load_gates,
    prepare_blinded_label_packet,
    review_prompt_sha256,
    review_response_schema,
    review_schema_sha256,
    run_review,
    score_review_labels,
)
from proceedings_to_eee.census_run import CensusLimits, run_census
from proceedings_to_eee.corpus import CorpusSpec, load_corpus
from proceedings_to_eee.demo import OfflineDemoError, run_offline_demo
from proceedings_to_eee.domain.attribution import OriginExportPolicy
from proceedings_to_eee.evaluation.attribution_score import (
    render_attribution_summary,
    score_run_attribution,
)
from proceedings_to_eee.evaluation.bakeoff import load_bakeoff_config, run_extractor_bakeoff
from proceedings_to_eee.evaluation.control_annotation_workflow import (
    lock_completed_responses,
    prepare_adjudication_workspace,
    prepare_annotation_workspace,
    validate_adjudication_workspace,
    validate_completion_bundle,
    validate_workspace_response,
)
from proceedings_to_eee.evaluation.control_proposals import propose_controls, worklist
from proceedings_to_eee.evaluation.corpus_audit import audit_corpus_references
from proceedings_to_eee.evaluation.development_annotation import (
    prepare_development_annotation_package,
    validate_completed_development_annotation_response,
)
from proceedings_to_eee.evaluation.human_review import (
    project_paper_review_outcomes,
    summarize_human_review,
    write_human_review_artifacts,
)
from proceedings_to_eee.evaluation.reference_audit import audit_reference_pdf
from proceedings_to_eee.evaluation.reference_score import score_reference_files
from proceedings_to_eee.evaluation.row_coverage import score_run_row_coverage
from proceedings_to_eee.evaluation.row_plan import build_next_run_row_plan_report
from proceedings_to_eee.extraction.llm import EXTRACTOR_REASONING_EFFORT
from proceedings_to_eee.extraction.pdf_layout import extract_pdf_layout, select_result_pages
from proceedings_to_eee.extraction.region_index import build_page_region_index, locate_quote
from proceedings_to_eee.io import sha256_file, write_json
from proceedings_to_eee.pipeline import (
    PipelineSettings,
    _code_state,
    _safe_error_message,
    freeze_corpus,
    run_corpus,
    runtime_key,
)
from proceedings_to_eee.preflight import PreflightCostAssumptions, preflight_corpus
from proceedings_to_eee.providers.budget import (
    BudgetedProviderClient,
    ProviderBudgetLimits,
    provider_budget_contract,
)
from proceedings_to_eee.providers.openrouter import OpenRouterClient
from proceedings_to_eee.public_snapshot import build_public_snapshot
from proceedings_to_eee.reporting.extraction_review_cards import (
    CorpusCardInput,
    write_extraction_review_bundle,
)
from proceedings_to_eee.reporting.public_development_preview import (
    PublicDevelopmentPreviewError,
    build_public_development_preview,
    verify_public_development_preview,
)
from proceedings_to_eee.reporting.public_development_summary import (
    PublicDevelopmentSummaryError,
    write_public_development_summary,
)
from proceedings_to_eee.reporting.public_reviewed_development_bundle import (
    PublicReviewedDevelopmentBundleError,
    build_public_reviewed_development_bundle,
    verify_public_reviewed_development_bundle,
)
from proceedings_to_eee.resources import DEFAULT_EEE_SCHEMA_PATH, EEE_SCHEMA_SHA256
from proceedings_to_eee.reviewed_export.workflow import (
    ReviewedExportError,
    compose_reviewed_eee,
    prepare_export_review,
    validate_export_review,
    verify_derived_run,
)
from proceedings_to_eee.run_seal import seal_run_tree
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record
from proceedings_to_eee.verification.replay import (
    ReplayScope,
    ReplaySettings,
    measure_replay,
    replay_run,
)

app = typer.Typer(no_args_is_help=True, help="Evidence-bound Proceedings -> EEE pipeline")

DEFAULT_SCHEMA_SHA256 = EEE_SCHEMA_SHA256


def _private_annotation_call[AnnotationResult](
    operation: Callable[..., AnnotationResult], /, **kwargs: Any
) -> AnnotationResult:
    """Keep private annotation data out of rich tracebacks and command output."""

    try:
        return operation(**kwargs)
    except Exception:
        typer.echo(
            json.dumps(
                {
                    "status": "private-annotation-validation-failed",
                    "detail": "Validation failed without displaying private record details.",
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None


def _reviewed_export_call[ReviewResult](
    operation: Callable[..., ReviewResult], /, **kwargs: Any
) -> ReviewResult:
    """Render stable review/export failures without excerpts, identities, or tracebacks."""

    try:
        return operation(**kwargs)
    except ReviewedExportError as error:
        typer.echo(
            json.dumps(
                {
                    "status": "reviewed-export-failed",
                    "code": error.code.value,
                    "detail": error.detail,
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None


def _offline_demo_call(output: Path) -> dict[str, Any]:
    """Render path-free demo failures without a traceback."""

    try:
        return run_offline_demo(output)
    except OfflineDemoError as error:
        typer.echo(
            json.dumps(
                {
                    "status": "offline-demo-failed",
                    "code": error.code.value,
                    "detail": error.detail,
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    except Exception:
        typer.echo(
            json.dumps(
                {
                    "status": "offline-demo-failed",
                    "code": "DEMO_FAILED",
                    "detail": "The offline demo failed without displaying local data.",
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None


def _preflight_call[PreflightResult](
    operation: Callable[..., PreflightResult], /, **kwargs: Any
) -> PreflightResult:
    """Render bounded provider-free preflight failures without a traceback."""

    try:
        return operation(**kwargs)
    except Exception as error:
        typer.echo(
            json.dumps(
                {
                    "status": "preflight-failed",
                    "error": {
                        "type": type(error).__name__,
                        "message": _safe_error_message(error),
                    },
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None


def _run_corpus_failure(*, error: Exception, phase: str, output: Path, next_command: str) -> None:
    """Emit a bounded secret-safe run failure without Rich traceback locals."""

    typer.echo(
        json.dumps(
            {
                "status": "run-corpus-technical-failure",
                "phase": phase,
                "error": {
                    "type": type(error).__name__,
                    "message": _safe_error_message(error),
                },
                "artifacts": {
                    "corpus_run": str(output.resolve() / "corpus-run.json"),
                    "provider_budget_ledger": str(
                        output.resolve() / "private" / "provider-budget-ledger.jsonl"
                    ),
                    "provider_budget_head": str(
                        output.resolve() / "private" / "provider-budget-ledger.jsonl.head.json"
                    ),
                },
                "next": (
                    "Fix the reported corpus, configuration, credential, or runtime issue, "
                    "then run next_command. Existing compatible checkpoints and the budget "
                    "ledger are retained."
                ),
                "next_command": next_command,
            },
            indent=2,
        ),
        err=True,
    )
    raise typer.Exit(code=1) from None


def _paper_subset(corpus: CorpusSpec, paper_id: str | None) -> CorpusSpec:
    if paper_id is None:
        return corpus
    matches = [paper for paper in corpus.papers if paper.paper_id == paper_id]
    if not matches:
        raise typer.BadParameter(f"paper_id is not present in the corpus: {paper_id}")
    return CorpusSpec(
        schema_version=corpus.schema_version,
        corpus_id=f"{corpus.corpus_id}--paper-{paper_id}",
        evaluation_split=corpus.evaluation_split,
        description=f"Single-paper technical smoke from {corpus.corpus_id}.",
        papers=matches,
    )


def _artifact_entry(path: Path, *, kind: str) -> dict[str, Any]:
    """Describe one local run artifact without claiming a missing path exists."""

    resolved = path.resolve()
    exists = (
        resolved.is_file() if kind == "file" else resolved.is_dir() and not resolved.is_symlink()
    )
    return {"path": str(resolved), "exists": exists}


def _run_artifacts(*, summary: dict[str, Any], corpus: CorpusSpec, output: Path) -> dict[str, Any]:
    """Return the complete, explicit local handoff for one corpus invocation."""

    output = output.resolve()
    paper_summaries = {
        item.get("paper_id"): item
        for item in summary.get("runs", [])
        if isinstance(item, dict) and isinstance(item.get("paper_id"), str)
    }
    papers: list[dict[str, Any]] = []
    errors: list[str] = []
    for paper in corpus.papers:
        paper_root = output / paper.paper_id
        paper_summary = paper_summaries.get(paper.paper_id)
        status = paper_summary.get("status") if paper_summary is not None else "not_started"
        run_path = paper_root / "run.json"
        error_record = str(run_path.resolve()) if status not in {"success", "not_started"} else None
        if error_record is not None:
            errors.append(error_record)
        papers.append(
            {
                "paper_id": paper.paper_id,
                "status": status,
                "paper_run": _artifact_entry(run_path, kind="file"),
                "paper_review": _artifact_entry(paper_root / "review.html", kind="file"),
                "observations": _artifact_entry(paper_root / "observations.jsonl", kind="file"),
                "verifications": _artifact_entry(paper_root / "verifications.jsonl", kind="file"),
                "tuple_resolution": _artifact_entry(
                    paper_root / "private" / "tuple-resolution.json", kind="file"
                ),
                "tuple_checkpoint": _artifact_entry(
                    paper_root / "private" / "tuple-resolution-checkpoint.json", kind="file"
                ),
                "verifier_gates": _artifact_entry(
                    paper_root / "private" / "verifier-gates.json", kind="file"
                ),
                "verifier_checkpoint": _artifact_entry(
                    paper_root / "private" / "verifier-checkpoint.json", kind="file"
                ),
                "origin_checkpoint": _artifact_entry(
                    paper_root / "private" / "origin-retrieval-checkpoint.json", kind="file"
                ),
                "eee": _artifact_entry(paper_root / "eee", kind="directory"),
                "invalid_eee": _artifact_entry(
                    paper_root / "private" / "invalid-eee.json", kind="file"
                ),
                "error_record": error_record,
            }
        )
    return {
        "corpus_run": _artifact_entry(output / "corpus-run.json", kind="file"),
        "corpus_review": _artifact_entry(output / "corpus-review.html", kind="file"),
        "corpus_evaluation": _artifact_entry(output / "corpus-evaluation.json", kind="file"),
        "provider_budget_ledger": _artifact_entry(
            output / "private" / "provider-budget-ledger.jsonl", kind="file"
        ),
        "provider_budget_head": _artifact_entry(
            output / "private" / "provider-budget-ledger.jsonl.head.json", kind="file"
        ),
        "papers": papers,
        "errors": errors,
    }


def _run_terminal_status(summary: dict[str, Any]) -> tuple[str, int]:
    """Map internal run health to distinct user-facing statuses and exits."""

    if summary.get("status") == "bounded_incomplete":
        return "bounded-incomplete", 3
    if summary.get("status") in {"error", "partial_failure"} or summary.get("papers_failed", 0):
        return "technical-failure", 1
    totals = summary.get("totals") if isinstance(summary.get("totals"), dict) else {}
    if summary.get("papers_needing_review", 0) or not totals.get("eee_records", 0):
        return "completed-with-review", 2
    return "successful-export", 0


def _run_command(
    *,
    corpus_path: Path,
    model: str,
    schema_path: Path,
    schema_sha256: str,
    tuple_model: str | None,
    tuple_max_tokens: int,
    verifier_model: str | None,
    verifier_max_tokens: int,
    origin_model: str | None,
    origin_max_tokens: int,
    output: Path,
    min_confidence: float,
    row_enumeration: bool,
    row_model: str | None,
    row_estimated_call_cost_usd: float | None,
    max_structured_calls: int,
    max_provider_cost_usd: float,
    provider_call_cost_reservation_usd: float,
    paper_id: str | None,
) -> str:
    """Render a shell-safe, credential-free reproduction/resume command."""

    parts = [
        "ere",
        "run-corpus",
        str(corpus_path.resolve()),
        "--model",
        model,
        "--schema-path",
        str(schema_path.resolve()),
        "--schema-sha256",
        schema_sha256,
        "--output",
        str(output.resolve()),
        "--min-confidence",
        str(min_confidence),
        "--max-structured-calls",
        str(max_structured_calls),
        "--max-provider-cost-usd",
        str(max_provider_cost_usd),
        "--provider-call-cost-reservation-usd",
        str(provider_call_cost_reservation_usd),
    ]
    if tuple_model is not None:
        parts.extend(
            [
                "--tuple-model",
                tuple_model,
                "--tuple-max-tokens",
                str(tuple_max_tokens),
            ]
        )
    if verifier_model is not None:
        parts.extend(
            [
                "--verifier-model",
                verifier_model,
                "--verifier-max-tokens",
                str(verifier_max_tokens),
            ]
        )
    if origin_model is not None:
        parts.extend(
            [
                "--origin-model",
                origin_model,
                "--origin-max-tokens",
                str(origin_max_tokens),
            ]
        )
    if row_enumeration:
        parts.append("--row-enumeration")
    if row_model is not None:
        parts.extend(["--row-model", row_model])
    if row_estimated_call_cost_usd is not None:
        parts.extend(["--row-estimated-call-cost-usd", str(row_estimated_call_cost_usd)])
    if paper_id is not None:
        parts.extend(["--paper-id", paper_id])
    return shlex.join(parts)


def _settings(
    *,
    schema_path: Path,
    schema_sha256: str,
    output: Path,
    model: str,
    min_confidence: float = 0.8,
    reasoning_effort: str = EXTRACTOR_REASONING_EFFORT,
    row_enumeration_enabled: bool = False,
    row_model: str | None = None,
    row_estimated_call_cost_usd: float | None = None,
    tuple_model: str | None = None,
    tuple_max_tokens: int = 16_000,
    verifier_model: str | None = None,
    verifier_max_tokens: int = 2_000,
    origin_model: str | None = None,
    origin_max_tokens: int = 16_000,
    provider_max_structured_calls: int = 10_000,
    provider_max_cost_usd: float = 1_000.0,
    provider_cost_reservation_per_call_usd: float = 0.25,
) -> PipelineSettings:
    return PipelineSettings(
        project_root=Path.cwd().resolve(),
        schema_path=schema_path.resolve(),
        schema_sha256=schema_sha256,
        output_root=output.resolve(),
        model=model,
        min_confidence=min_confidence,
        reasoning_effort=reasoning_effort,
        row_enumeration_enabled=row_enumeration_enabled,
        row_model=row_model,
        row_estimated_call_cost_usd=row_estimated_call_cost_usd,
        tuple_model=tuple_model,
        tuple_max_tokens=tuple_max_tokens,
        verifier_model=verifier_model,
        verifier_max_tokens=verifier_max_tokens,
        origin_model=origin_model,
        origin_max_tokens=origin_max_tokens,
        provider_max_structured_calls=provider_max_structured_calls,
        provider_max_cost_usd=provider_max_cost_usd,
        provider_cost_reservation_per_call_usd=provider_cost_reservation_per_call_usd,
    )


@app.command("demo")
def demo_command(
    output: Annotated[Path, typer.Option()] = Path("runs/demo"),
) -> None:
    """Run a complete deterministic synthetic example without credentials or network."""

    result = _offline_demo_call(output.resolve())
    typer.echo(json.dumps(result, indent=2))


@app.command("inspect-pdf")
def inspect_pdf(
    pdf: Annotated[Path, typer.Argument(exists=True, readable=True)],
    source_id: Annotated[str, typer.Option()] = "inspection",
    limit: Annotated[int, typer.Option(min=1, max=30)] = 12,
) -> None:
    """Show result-rich pages selected by the deterministic layout stage."""

    layout = extract_pdf_layout(pdf, source_id)
    selected = select_result_pages(layout, limit=limit)
    typer.echo(
        json.dumps(
            {
                "pages": layout.page_count,
                "parser": layout.parser_version,
                "selected": [
                    {
                        "page": page.page,
                        "score": page.result_signal_score,
                        "numbers": page.numeric_token_count,
                    }
                    for page in selected
                ],
            },
            indent=2,
        )
    )


@app.command("run-corpus")
def run_corpus_command(
    corpus_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    model: Annotated[str, typer.Option(envvar="ERE_EXTRACTOR_MODEL")],
    schema_path: Annotated[
        Path, typer.Option(exists=True, readable=True)
    ] = DEFAULT_EEE_SCHEMA_PATH,
    tuple_model: Annotated[str | None, typer.Option(envvar="ERE_TUPLE_MODEL")] = None,
    tuple_max_tokens: Annotated[int, typer.Option(min=1)] = 16_000,
    verifier_model: Annotated[str | None, typer.Option(envvar="ERE_VERIFIER_MODEL")] = None,
    verifier_max_tokens: Annotated[int, typer.Option(min=1)] = 2_000,
    origin_model: Annotated[str | None, typer.Option(envvar="ERE_ORIGIN_MODEL")] = None,
    origin_max_tokens: Annotated[int, typer.Option(min=1)] = 16_000,
    output: Annotated[Path, typer.Option()] = Path("runs/latest"),
    schema_sha256: Annotated[str, typer.Option()] = DEFAULT_SCHEMA_SHA256,
    min_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
    reasoning_effort: Annotated[
        str,
        typer.Option(
            help=(
                "Reasoning effort asked of the extractor model, recorded in run.json. "
                "Pass 'none' to turn provider reasoning off rather than lower it: some "
                "models bill reasoning tokens even at the lowest effort."
            )
        ),
    ] = EXTRACTOR_REASONING_EFFORT,
    row_enumeration: Annotated[
        bool,
        typer.Option(
            help=(
                "Run the bounded dense-table row stage in addition to legacy block "
                "extraction. This incurs additional provider calls."
            )
        ),
    ] = False,
    row_model: Annotated[
        str | None,
        typer.Option(
            envvar="ERE_ROW_MODEL",
            help=(
                "Optional row-disposition model. Defaults to --model when the row stage is enabled."
            ),
        ),
    ] = None,
    row_estimated_call_cost_usd: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help=(
                "Optional historical per-call mean used only for the preflight estimate; "
                "it is not treated as a provider quote."
            ),
        ),
    ] = None,
    max_structured_calls: Annotated[
        int,
        typer.Option(
            min=1,
            help="Hard per-corpus ceiling on structured provider invocations across all stages.",
        ),
    ] = 10_000,
    max_provider_cost_usd: Annotated[
        float,
        typer.Option(
            min=0.000001,
            help="Hard per-corpus ceiling for conservative provider cost reservations.",
        ),
    ] = 1_000.0,
    provider_call_cost_reservation_usd: Annotated[
        float,
        typer.Option(
            min=0.000001,
            help=(
                "Conservative non-refundable cost reservation charged before every structured call."
            ),
        ),
    ] = 0.25,
    paper_id: Annotated[str | None, typer.Option()] = None,
    quiet: Annotated[bool, typer.Option()] = False,
) -> None:
    """Freeze, extract, validate, compose, and report a corpus."""

    current_command = _run_command(
        corpus_path=corpus_path,
        model=model,
        schema_path=schema_path,
        schema_sha256=schema_sha256,
        tuple_model=tuple_model,
        tuple_max_tokens=tuple_max_tokens,
        verifier_model=verifier_model,
        verifier_max_tokens=verifier_max_tokens,
        origin_model=origin_model,
        origin_max_tokens=origin_max_tokens,
        output=output,
        min_confidence=min_confidence,
        row_enumeration=row_enumeration,
        row_model=row_model,
        row_estimated_call_cost_usd=row_estimated_call_cost_usd,
        max_structured_calls=max_structured_calls,
        max_provider_cost_usd=max_provider_cost_usd,
        provider_call_cost_reservation_usd=provider_call_cost_reservation_usd,
        paper_id=paper_id,
    )
    try:
        corpus = _paper_subset(load_corpus(corpus_path), paper_id)
    except Exception as error:
        _run_corpus_failure(
            error=error,
            phase="corpus_validation",
            output=output,
            next_command=current_command,
        )
    settings = _settings(
        schema_path=schema_path.resolve(),
        schema_sha256=schema_sha256,
        output=output.resolve(),
        model=model,
        min_confidence=min_confidence,
        reasoning_effort=reasoning_effort,
        row_enumeration_enabled=row_enumeration,
        row_model=row_model,
        row_estimated_call_cost_usd=row_estimated_call_cost_usd,
        tuple_model=tuple_model,
        tuple_max_tokens=tuple_max_tokens,
        verifier_model=verifier_model,
        verifier_max_tokens=verifier_max_tokens,
        origin_model=origin_model,
        origin_max_tokens=origin_max_tokens,
        provider_max_structured_calls=max_structured_calls,
        provider_max_cost_usd=max_provider_cost_usd,
        provider_cost_reservation_per_call_usd=provider_call_cost_reservation_usd,
    )
    try:
        if settings.tuple_model is not None and not settings.tuple_model.strip():
            raise ValueError("tuple model must be non-empty")
        if settings.verifier_model is not None and settings.tuple_model is None:
            raise ValueError("verifier_model requires tuple_model")
        if settings.origin_model is not None and settings.verifier_model is None:
            raise ValueError("origin_model requires verifier_model")
        if settings.row_model is not None and not settings.row_enumeration_enabled:
            raise ValueError("row_model requires --row-enumeration")
        if settings.row_model is not None and not settings.row_model.strip():
            raise ValueError("row model must be non-empty")
        load_schema(settings.schema_path, settings.schema_sha256)
        ProviderBudgetLimits(
            max_structured_calls=settings.provider_max_structured_calls,
            max_cost_usd=settings.provider_max_cost_usd,
            cost_reservation_per_call_usd=settings.provider_cost_reservation_per_call_usd,
        )
    except Exception as error:
        _run_corpus_failure(
            error=error,
            phase="local_configuration",
            output=output,
            next_command=current_command,
        )
    try:
        client = OpenRouterClient(api_key=runtime_key())
        summary = run_corpus(corpus=corpus, settings=settings, client=client)
    except Exception as error:
        _run_corpus_failure(
            error=error,
            phase="runtime",
            output=output,
            next_command=current_command,
        )
    terminal_status, exit_code = _run_terminal_status(summary)
    artifacts = _run_artifacts(summary=summary, corpus=corpus, output=output)
    sealed_destination = output.resolve().with_name(f"{output.resolve().name}-sealed")
    if terminal_status == "bounded-incomplete":
        next_command = _run_command(
            corpus_path=corpus_path,
            model=model,
            schema_path=schema_path,
            schema_sha256=schema_sha256,
            tuple_model=tuple_model,
            tuple_max_tokens=tuple_max_tokens,
            verifier_model=verifier_model,
            verifier_max_tokens=verifier_max_tokens,
            origin_model=origin_model,
            origin_max_tokens=origin_max_tokens,
            output=output,
            min_confidence=min_confidence,
            row_enumeration=row_enumeration,
            row_model=row_model,
            row_estimated_call_cost_usd=row_estimated_call_cost_usd,
            max_structured_calls=max(max_structured_calls + 1, max_structured_calls * 2),
            max_provider_cost_usd=max_provider_cost_usd * 2,
            provider_call_cost_reservation_usd=provider_call_cost_reservation_usd,
            paper_id=paper_id,
        )
        next_detail = (
            "Resume with monotonically higher ceilings. Compatible extractor, row, tuple, "
            "verifier, and origin checkpoints are reused; the prior ledger remains charged."
        )
    elif terminal_status == "technical-failure":
        next_command = current_command
        next_detail = (
            "Inspect each listed error_record, fix the typed technical cause, then rerun "
            "the exact command. Compatible checkpoints remain available."
        )
    else:
        next_command = shlex.join(
            [
                "ere",
                "seal-run",
                str(output.resolve()),
                "--destination",
                str(sealed_destination),
            ]
        )
        next_detail = (
            "Seal the completed run before semantic inspection or review. Then prepare a "
            "candidate-bound export review from the sealed tree."
        )
    payload: dict[str, Any] = {
        "status": terminal_status,
        "pipeline_status": summary.get("status"),
        "exit_code": exit_code,
        "totals": summary.get("totals", {}),
        "artifacts": artifacts,
        "next": next_detail,
        "next_command": next_command,
    }
    if not quiet:
        payload["provider_budget"] = summary.get("provider_budget")
        payload["papers"] = {
            "total": summary.get("papers_total", summary.get("papers")),
            "succeeded": summary.get("papers_succeeded"),
            "failed": summary.get("papers_failed"),
            "not_started": summary.get("papers_not_started"),
            "needing_review": summary.get("papers_needing_review"),
        }
    typer.echo(json.dumps(payload, indent=2), err=exit_code != 0)
    if exit_code:
        raise typer.Exit(exit_code)


@app.command("bakeoff-extractors")
def bakeoff_extractors_command(
    config_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()] = Path("runs/bakeoff/extractor-pilot.json"),
) -> None:
    """Compare configured OpenRouter extractors on frozen, prompt-isolated cases."""

    client = OpenRouterClient(api_key=runtime_key())
    result = run_extractor_bakeoff(
        load_bakeoff_config(config_path),
        project_root=Path.cwd().resolve(),
        client=client,
        output_path=output.resolve(),
    )
    summary = [
        {
            "model": item["model"],
            "execution": item["aggregate"]["execution"],
            "schema": item["aggregate"]["schema"],
            "quality": item["aggregate"]["quality"],
            "negative_control_safety": item["aggregate"]["negative_control_safety"],
            "claim_type_classification": item["aggregate"]["claim_type_classification"],
            "model_selection_gates": item["aggregate"]["model_selection_gates"],
            "usage": item["aggregate"]["usage"],
        }
        for item in result["models"]
    ]
    typer.echo(json.dumps(summary, indent=2))


@app.command("freeze-corpus")
def freeze_corpus_command(
    corpus_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()] = Path("runs/frozen"),
    schema_path: Annotated[
        Path, typer.Option(exists=True, readable=True)
    ] = DEFAULT_EEE_SCHEMA_PATH,
    schema_sha256: Annotated[str, typer.Option()] = DEFAULT_SCHEMA_SHA256,
    workers: Annotated[
        int,
        typer.Option(min=1, max=32, help="Concurrent download threads sharing one pacer."),
    ] = 4,
    requests_per_second: Annotated[
        float,
        typer.Option(min=0.0, help="Ceiling on outbound requests, shared by all workers."),
    ] = 2.0,
    progress_every: Annotated[
        int,
        typer.Option(min=0, help="Emit a progress line every N papers; 0 disables it."),
    ] = 0,
) -> None:
    """Download and content-address all paper sources without using OpenRouter."""

    settings = _settings(
        schema_path=schema_path,
        schema_sha256=schema_sha256,
        output=output,
        model="not-used",
    )
    summary = freeze_corpus(
        load_corpus(corpus_path),
        settings,
        workers=workers,
        requests_per_second=requests_per_second,
        progress_every=progress_every,
        progress=lambda line: typer.echo(line, err=True),
    )
    compact = {key: value for key, value in summary.items() if key not in ("manifests", "results")}
    typer.echo(json.dumps(compact, indent=2))
    if summary["papers_failed"]:
        raise typer.Exit(1)


@app.command("build-census-atlas")
def build_census_atlas_command(
    run_root: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()] = Path("runs/census-2026-09/atlas"),
    census_metadata: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    census_provenance: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    route_validated: Annotated[str, typer.Option()] = "partial",
) -> None:
    """Build the evidence atlas from a completed census run. Offline."""

    summary = build_atlas(
        run_root=run_root,
        output=output,
        census_metadata=(
            json.loads(census_metadata.read_text(encoding="utf-8")) if census_metadata else None
        ),
        census_provenance=(
            json.loads(census_provenance.read_text(encoding="utf-8")) if census_provenance else None
        ),
        route_validated=route_validated,
    )
    typer.echo(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "papers_total",
                    "papers_by_status",
                    "papers_ran",
                    "papers_with_results",
                    "observations",
                    "supported_observations",
                    "by_extraction_stage",
                    "planned_rows",
                    "row_states",
                    "provider_reported_cost_usd",
                )
            },
            indent=2,
        )
    )


@app.command("recompose-census")
def recompose_census_command(
    run_root: Annotated[Path, typer.Argument(exists=True, readable=True)],
    origin_policy: Annotated[
        str,
        typer.Option(
            help=(
                "Which producer-origin bases may export. 'positive_only' is the "
                "historical policy and must reproduce the run exactly; 'tiered' exports "
                "any basis but none, carrying the basis in every record."
            )
        ),
    ] = "positive_only",
    output_dir: Annotated[str, typer.Option()] = "eee",
    min_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
    schema_path: Annotated[
        Path, typer.Option(exists=True, readable=True)
    ] = DEFAULT_EEE_SCHEMA_PATH,
    paper_ids: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
) -> None:
    """Replay validation and EEE composition over a finished census run. Offline."""

    try:
        policy = OriginExportPolicy(origin_policy)
    except ValueError:
        typer.echo(f"unsupported origin policy: {origin_policy}", err=True)
        raise typer.Exit(code=2) from None
    wanted: set[str] | None = None
    if paper_ids is not None:
        payload = json.loads(paper_ids.read_text(encoding="utf-8"))
        slugs = payload.get("slugs") if isinstance(payload, dict) else payload
        wanted = {str(slug) for slug in slugs or []}
    summary = recompose_census(
        run_root=run_root,
        origin_policy=policy,
        output_dir=output_dir,
        min_confidence=min_confidence,
        schema_path=schema_path,
        pipeline_git_commit=str(_code_state(Path.cwd().resolve()).get("git_commit") or "") or None,
        paper_ids=wanted,
    )
    typer.echo(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "origin_policy",
                    "papers_read",
                    "candidates",
                    "records",
                    "records_by_tier",
                    "records_by_producer_origin_basis",
                )
            },
            indent=2,
        )
    )


@app.command("review-candidates")
def review_candidates_command(
    run_root: Annotated[Path, typer.Argument(exists=True, readable=True)],
    model: Annotated[str, typer.Option(envvar="ERE_REVIEW_MODEL")],
    max_cost_usd: Annotated[float, typer.Option(min=1e-6)] = 8.0,
    workers: Annotated[int, typer.Option(min=1, max=64)] = 8,
    max_tokens: Annotated[int, typer.Option(min=1)] = 4_000,
    provider_call_cost_reservation_usd: Annotated[float, typer.Option(min=1e-6)] = 0.05,
    max_structured_calls: Annotated[int, typer.Option(min=1)] = 20_000,
    paper_ids: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Print the selection size and spend nothing.")
    ] = False,
) -> None:
    """Review selected candidates against their own evidence page, inside a cost ceiling.

    Model judgements, checked mechanically against the page. Never a human label, and
    never a measured error rate.
    """

    wanted: set[str] | None = None
    if paper_ids is not None:
        payload = json.loads(paper_ids.read_text(encoding="utf-8"))
        slugs = payload.get("slugs") if isinstance(payload, dict) else payload
        wanted = {str(slug) for slug in slugs or []}
    items = collect_review_items(run_root, paper_ids=wanted)
    if dry_run:
        already = 0
        for paper_id in {item.paper_id for item in items}:
            already += len(load_gates(run_root / paper_id))
        typer.echo(
            json.dumps(
                {
                    "selected": len(items),
                    "papers": len({item.paper_id for item in items}),
                    "already_gated": already,
                    "provider_calls": 0,
                },
                indent=2,
            )
        )
        return

    limits = ProviderBudgetLimits(
        max_structured_calls=max_structured_calls,
        max_cost_usd=max_cost_usd,
        cost_reservation_per_call_usd=provider_call_cost_reservation_usd,
    )
    contract = provider_budget_contract(
        corpus_binding={"run_root": run_root.name, "stage": "candidate_page_review"},
        provider_run_contract={
            "review": {
                "provider": "openrouter",
                "model": model,
                "max_tokens": max_tokens,
                "schema_name": REVIEW_SCHEMA_NAME,
                "schema_sha256": review_schema_sha256(),
                "prompt_sha256": review_prompt_sha256(),
                "temperature": REVIEW_TEMPERATURE,
                "reasoning_effort": REVIEW_REASONING_EFFORT,
                "seed": REVIEW_SEED,
            }
        },
        limits=limits,
    )
    client = BudgetedProviderClient(
        client=OpenRouterClient(api_key=runtime_key()),
        ledger_path=run_root / "private" / "review-budget-ledger.jsonl",
        contract=contract,
        limits=limits,
    )
    spent = 0.0
    spend_lock = threading.Lock()

    def call(system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal spent
        response = client.structured_chat(
            model=model,
            system=system,
            user=user,
            schema_name=REVIEW_SCHEMA_NAME,
            schema=review_response_schema(),
            max_tokens=max_tokens,
            temperature=REVIEW_TEMPERATURE,
            reasoning_effort=REVIEW_REASONING_EFFORT,
            seed=REVIEW_SEED,
        )
        provider_call = response.call
        with spend_lock:
            spent += provider_call.cost_usd or 0.0
        telemetry = {
            "request_sha256": provider_call.prompt_sha256,
            "response_sha256": provider_call.response_sha256,
            "input_tokens": provider_call.input_tokens,
            "output_tokens": provider_call.output_tokens,
            "cost_usd": provider_call.cost_usd,
            "latency_seconds": provider_call.latency_seconds,
            "attempts": provider_call.attempts,
        }
        return response.payload, telemetry

    def stop() -> bool:
        with spend_lock:
            return spent >= max_cost_usd - provider_call_cost_reservation_usd

    summary = run_review(
        run_root=run_root,
        model=model,
        call=call,
        paper_ids=wanted,
        workers=workers,
        items=items,
        stop=stop,
    )
    summary["provider_reported_cost_usd"] = round(spent, 6)
    summary["max_cost_usd"] = max_cost_usd
    write_json(output or (run_root / "review-summary.json"), summary)
    typer.echo(json.dumps(summary, indent=2))


@app.command("review-label-packet")
def review_label_packet_command(
    run_root: Annotated[Path, typer.Argument(exists=True, readable=True)],
    size: Annotated[int, typer.Option(min=1, help="Accepted items to sample.")] = 100,
    negatives: Annotated[
        int, typer.Option(min=0, help="Rejected or abstained items to sample.")
    ] = 50,
    seed: Annotated[str, typer.Option()] = "20260905",
    output: Annotated[Path | None, typer.Option()] = None,
    paper_ids: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
) -> None:
    """Write a blank label packet for a human to fill in. Offline, and never filled here."""

    wanted: set[str] | None = None
    if paper_ids is not None:
        payload = json.loads(paper_ids.read_text(encoding="utf-8"))
        slugs = payload.get("slugs") if isinstance(payload, dict) else payload
        wanted = {str(slug) for slug in slugs or []}
    packet = build_label_packet(
        run_root=run_root, size=size, negatives=negatives, seed=seed, paper_ids=wanted
    )
    destination = output or (run_root / "private" / "label-packet" / "review-label-packet.json")
    write_json(destination, packet)
    typer.echo(
        json.dumps(
            {
                "written": str(destination),
                "entries": len(packet["entries"]),
                "available": packet["available"],
                "labels_filled": 0,
            },
            indent=2,
        )
    )


@app.command("blind-review-label-packet")
def blind_review_label_packet_command(
    packet_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option(help="New private reviewer JSON file.")],
    seed: Annotated[str, typer.Option()] = "review-label-shuffle-v1",
) -> None:
    """Shuffle a frozen packet and hide model decisions without filling any labels."""

    if output.exists():
        raise typer.BadParameter("output already exists; choose a new reviewer file")
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        blinded = prepare_blinded_label_packet(packet, seed=seed)
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    write_json(output, blinded)
    typer.echo(
        json.dumps(
            {"written": str(output), "entries": len(blinded["entries"]), "labels_filled": 0},
            indent=2,
        )
    )


@app.command("score-review-labels")
def score_review_labels_command(
    packet_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    labels_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    confidence: Annotated[float, typer.Option(min=0.5, max=0.999)] = 0.95,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Score a filled label file against the model's decisions, with exact bounds."""

    try:
        summary = score_review_labels(
            packet=json.loads(packet_path.read_text(encoding="utf-8")),
            labels=json.loads(labels_path.read_text(encoding="utf-8")),
            confidence=confidence,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if output is not None:
        write_json(output, summary)
    typer.echo(json.dumps(summary, indent=2))


@app.command("audit-candidates")
def audit_candidates_command(
    run_root: Annotated[Path, typer.Argument(exists=True, readable=True)],
    model: Annotated[str, typer.Option(envvar="ERE_AUDIT_MODEL")],
    output: Annotated[Path, typer.Option()] = Path("audit.json"),
    sample: Annotated[int, typer.Option(min=1)] = 120,
    seed: Annotated[str, typer.Option()] = "20260903",
    census_metadata: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    order: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    max_tokens: Annotated[int, typer.Option(min=1)] = 4_000,
) -> None:
    """Audit sampled candidates against their full source page. AI analyst QA only.

    This is a diagnostic. It is never precision, recall, accuracy, or independent human
    validation, and every artifact it writes says so.
    """

    client = OpenRouterClient(api_key=runtime_key())

    def call(system: str, user: str) -> dict[str, Any]:
        response = client.structured_chat(
            model=model,
            system=system,
            user=user,
            schema_name=AUDIT_SCHEMA_NAME,
            schema=audit_response_schema(),
            max_tokens=max_tokens,
        )
        return response.payload

    metadata = json.loads(census_metadata.read_text(encoding="utf-8")) if census_metadata else None
    paper_ids = None
    if order is not None:
        payload = json.loads(order.read_text(encoding="utf-8"))
        paper_ids = payload["slugs"] if isinstance(payload, dict) else list(payload)

    summary = run_audit(
        run_root=run_root,
        output=output,
        model=model,
        sample=sample,
        seed=seed,
        call=call,
        paper_ids=paper_ids,
        census_metadata=metadata,
    )
    typer.echo(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "candidates_audited",
                    "verdicts_kept",
                    "verdicts_dropped_unquotable",
                    "per_field",
                    "sampling_frame",
                )
            },
            indent=2,
        )
    )


@app.command("run-census")
def run_census_command(
    corpus_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    model: Annotated[str, typer.Option(envvar="ERE_EXTRACTOR_MODEL")],
    order: Annotated[Path, typer.Option(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()] = Path("runs/census-2026-09"),
    schema_path: Annotated[
        Path, typer.Option(exists=True, readable=True)
    ] = DEFAULT_EEE_SCHEMA_PATH,
    schema_sha256: Annotated[str, typer.Option()] = DEFAULT_SCHEMA_SHA256,
    min_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
    reasoning_effort: Annotated[
        str,
        typer.Option(
            help=(
                "Reasoning effort asked of the extractor model, recorded in run.json. "
                "Pass 'none' to turn provider reasoning off rather than lower it."
            )
        ),
    ] = EXTRACTOR_REASONING_EFFORT,
    row_enumeration: Annotated[bool, typer.Option()] = False,
    workers: Annotated[int, typer.Option(min=1, max=64)] = 16,
    max_cost_usd: Annotated[
        float, typer.Option(min=1e-6, help="Hard ceiling on provider-reported cost.")
    ] = 25.0,
    per_paper_max_cost_usd: Annotated[float, typer.Option(min=1e-6)] = 1.5,
    per_paper_max_calls: Annotated[int, typer.Option(min=1)] = 400,
    provider_call_cost_reservation_usd: Annotated[float, typer.Option(min=1e-6)] = 0.05,
    dry_run: Annotated[
        bool,
        typer.Option(help="Walk the order and write a ledger without any provider call."),
    ] = False,
) -> None:
    """Run an ordered census resumably, one paper per worker, inside a cost ceiling."""

    settings = _settings(
        schema_path=schema_path,
        schema_sha256=schema_sha256,
        output=output,
        model=model,
        min_confidence=min_confidence,
        reasoning_effort=reasoning_effort,
        row_enumeration_enabled=row_enumeration,
        provider_max_structured_calls=per_paper_max_calls,
        provider_max_cost_usd=per_paper_max_cost_usd,
        provider_cost_reservation_per_call_usd=provider_call_cost_reservation_usd,
    )
    try:
        limits = CensusLimits(
            workers=workers,
            max_cost_usd=max_cost_usd,
            per_paper_max_cost_usd=per_paper_max_cost_usd,
            per_paper_max_calls=per_paper_max_calls,
            reservation_usd=provider_call_cost_reservation_usd,
        )
    except ValueError as error:
        typer.echo(f"census limits are inconsistent: {error}", err=True)
        raise typer.Exit(2) from error

    order_payload = json.loads(order.read_text(encoding="utf-8"))
    slugs = order_payload["slugs"] if isinstance(order_payload, dict) else list(order_payload)
    summary = run_census(
        corpus=load_corpus(corpus_path),
        settings=settings,
        order=slugs,
        limits=limits,
        dry_run=dry_run,
    )
    typer.echo(json.dumps(summary, indent=2))
    if summary["counts"].get("failed"):
        raise typer.Exit(3)


@app.command("import-census")
def import_census_command(
    census_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()] = Path("runs/census-2026-09"),
    corpus_dir: Annotated[Path, typer.Option()] = Path("configs/corpora"),
    corpus_id: Annotated[str, typer.Option()] = "census-toxicity-2026-09",
    seed: Annotated[
        str, typer.Option(help="Deterministic seed for the reserve draw and the run order.")
    ] = "20260903",
    reserve: Annotated[
        int,
        typer.Option(
            min=0,
            help=(
                "Papers held back unseen so a future sealed evaluation stays possible. They "
                "are excluded from the corpus, every shard, and every later phase."
            ),
        ),
    ] = 60,
    shard_size: Annotated[int, typer.Option(min=1)] = 250,
    as_of: Annotated[str, typer.Option(help="Import date recorded in the provenance file.")] = "",
    known_corpus: Annotated[
        list[Path] | None,
        typer.Option(
            "--known-corpus",
            help="Earlier corpus YAML whose papers count as already inspected.",
        ),
    ] = None,
    known_paper_id: Annotated[
        list[str] | None,
        typer.Option(
            "--known-paper-id",
            help="Census identifier to flag as already inspected or reserved elsewhere.",
        ),
    ] = None,
) -> None:
    """Turn a paper census CSV into a corpus, a sealed reserve, and a seeded order."""

    stamp = as_of.strip() or datetime.now(UTC).date().isoformat()
    try:
        summary = import_census(
            csv_path=census_path,
            output_dir=output,
            corpus_dir=corpus_dir,
            corpus_id=corpus_id,
            seed=seed,
            reserve_size=reserve,
            shard_size=shard_size,
            as_of=stamp,
            known_corpus_paths=tuple(known_corpus or ()),
            known_paper_ids=tuple(known_paper_id or ()),
        )
    except CensusImportError as error:
        typer.echo(f"census import failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(summary, indent=2))


@app.command("preflight-corpus")
def preflight_corpus_command(
    corpus_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()] = Path("runs/preflight"),
    schema_path: Annotated[
        Path, typer.Option(exists=True, readable=True)
    ] = DEFAULT_EEE_SCHEMA_PATH,
    schema_sha256: Annotated[str, typer.Option()] = DEFAULT_SCHEMA_SHA256,
    row_enumeration: Annotated[
        bool,
        typer.Option(
            help=(
                "Include the already-materialized bounded row plan in the configured "
                "structured-call maximum."
            )
        ),
    ] = False,
    tuple_call_ceiling: Annotated[
        int,
        typer.Option(
            min=0,
            help=("Tuple-resolution call allowance in the global structured-call maximum."),
        ),
    ] = 0,
    verifier_call_ceiling: Annotated[
        int,
        typer.Option(
            min=0,
            help=("Verifier-call allowance to include in the global structured-call maximum."),
        ),
    ] = 0,
    origin_call_ceiling: Annotated[
        int,
        typer.Option(
            min=0,
            help=(
                "Whole-paper origin-call allowance to include in the global structured-call "
                "maximum; origin calls can only follow verifier acceptance."
            ),
        ),
    ] = 0,
    legacy_call_cost_usd: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Optional user-supplied legacy structured-call estimate; not a quote.",
        ),
    ] = None,
    row_call_cost_usd: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Optional user-supplied row structured-call estimate; not a quote.",
        ),
    ] = None,
    tuple_call_cost_usd: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Optional user-supplied tuple-call estimate; not a quote.",
        ),
    ] = None,
    verifier_call_cost_usd: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Optional user-supplied verifier-call estimate; not a quote.",
        ),
    ] = None,
    origin_call_cost_usd: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Optional user-supplied origin-call estimate; not a quote.",
        ),
    ] = None,
    paper_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Freeze and plan a new corpus without a provider client or API key."""

    resolved_corpus_path = corpus_path.resolve()
    corpus = _preflight_call(load_corpus, path=resolved_corpus_path)
    corpus = _paper_subset(corpus, paper_id)
    settings = _settings(
        schema_path=schema_path,
        schema_sha256=schema_sha256,
        output=output,
        model="not-used-by-preflight",
        row_enumeration_enabled=row_enumeration,
    )
    report = _preflight_call(
        preflight_corpus,
        corpus=corpus,
        settings=settings,
        tuple_call_ceiling=tuple_call_ceiling,
        verifier_call_ceiling=verifier_call_ceiling,
        origin_call_ceiling=origin_call_ceiling,
        cost_assumptions=PreflightCostAssumptions(
            legacy_call_usd=legacy_call_cost_usd,
            row_call_usd=row_call_cost_usd,
            tuple_call_usd=tuple_call_cost_usd,
            verifier_call_usd=verifier_call_cost_usd,
            origin_call_usd=origin_call_cost_usd,
        ),
    )
    paper_summaries = []
    for paper in report["paper_reports"]:
        summary: dict[str, Any] = {
            "paper_id": paper["paper_id"],
            "status": paper["status"],
            "preflight": str(output.resolve() / paper["paper_id"] / "preflight.json"),
        }
        if paper["status"] == "success":
            summary.update(
                {
                    "selected_pages": [
                        item["page"] for item in paper["selection"]["selected_pages"]
                    ],
                    "result_blocks": len(paper["selected_blocks"]),
                    "rows_planned": paper["row_plan"]["telemetry"]["rows_planned"],
                    "maximum_extraction_calls_before_verifier_or_origin": paper[
                        "structured_call_bound"
                    ]["configured_without_verifier"]["maximum_calls"],
                }
            )
        else:
            summary["error"] = paper["error"]
        paper_summaries.append(summary)
    typer.echo(
        json.dumps(
            {
                "status": report["status"],
                "mode": report["mode"],
                "output": str(output.resolve()),
                "corpus_preflight": str(output.resolve() / "corpus-preflight.json"),
                "provider_calls_made": report["provider_calls_made"],
                "provider_credentials_required": report["provider_credentials_required"],
                "papers_succeeded": report["papers_succeeded"],
                "papers_failed": report["papers_failed"],
                "counts": report["counts"],
                "structured_call_bound": report["structured_call_bound"],
                "cost_estimates": report["cost_estimates"],
                "papers": paper_summaries,
            },
            indent=2,
        )
    )
    if report["papers_failed"]:
        raise typer.Exit(1)


@app.command("validate-eee")
def validate_eee(
    record_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    schema_path: Annotated[
        Path, typer.Option(exists=True, readable=True)
    ] = DEFAULT_EEE_SCHEMA_PATH,
    schema_sha256: Annotated[str | None, typer.Option()] = DEFAULT_SCHEMA_SHA256,
) -> None:
    """Validate one EEE record and enforce schema-version equality."""

    schema, authority = load_schema(schema_path, schema_sha256)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    issues = validate_eee_record(record, schema)
    typer.echo(
        json.dumps(
            {"schema": authority.version, "issues": [issue.__dict__ for issue in issues]},
            indent=2,
        )
    )
    if issues:
        raise typer.Exit(1)


@app.command("seal-run")
def seal_run_command(
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    destination: Annotated[Path, typer.Option()],
) -> None:
    """Preserve and checksum a first run, including partial/error runs, before inspection."""

    manifest = seal_run_tree(run_root, destination)
    typer.echo(
        json.dumps(
            {
                "destination": destination.name,
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "tree_sha256": manifest["tree_sha256"],
                "manifest": "RUN-SEAL.json",
            },
            indent=2,
        )
    )


@app.command("prepare-export-review")
def prepare_export_review_command(
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option()],
    min_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
) -> None:
    """Prepare a private candidate-bound origin/export decision packet."""

    manifest = _reviewed_export_call(
        prepare_export_review,
        run_root=run_root.resolve(),
        output_root=output.resolve(),
        min_confidence=min_confidence,
    )
    typer.echo(
        json.dumps(
            {
                "status": "prepared",
                "review_root": str(output.resolve()),
                "items": manifest.item_count,
                "decisions": str((output / "decisions.jsonl").resolve()),
                "next_command": f"ere validate-export-review {output.resolve()}",
            },
            indent=2,
        )
    )


@app.command("validate-export-review")
def validate_export_review_command(
    review_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
) -> None:
    """Strictly validate and lock exact private export decisions."""

    lock, lock_sha = _reviewed_export_call(
        validate_export_review,
        review_root=review_root.resolve(),
    )
    typer.echo(
        json.dumps(
            {
                "status": "locked",
                "review_lock_sha256": lock_sha,
                "completed": lock.completed_count,
                "pending": lock.pending_count,
                "lock": str((review_root / "review-lock.json").resolve()),
            },
            indent=2,
        )
    )


@app.command("compose-reviewed-eee")
def compose_reviewed_eee_command(
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    decisions: Annotated[Path, typer.Option(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    schema_path: Annotated[
        Path, typer.Option(exists=True, readable=True)
    ] = DEFAULT_EEE_SCHEMA_PATH,
    schema_sha256: Annotated[str, typer.Option()] = DEFAULT_SCHEMA_SHA256,
) -> None:
    """Reopen a sealed run and compose a new verified reviewed EEE tree."""

    manifest = _reviewed_export_call(
        compose_reviewed_eee,
        run_root=run_root.resolve(),
        decisions_path=decisions.resolve(),
        output_root=output.resolve(),
        schema_path=schema_path.resolve(),
        schema_sha256=schema_sha256,
    )
    typer.echo(
        json.dumps(
            {
                "status": manifest.status,
                "derived_root": str(output.resolve()),
                "counts": manifest.counts,
                "next_command": f"ere verify-run {output.resolve()}",
            },
            indent=2,
        )
    )


@app.command("verify-run")
def verify_run_command(
    derived_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    schema_path: Annotated[
        Path, typer.Option(exists=True, readable=True)
    ] = DEFAULT_EEE_SCHEMA_PATH,
    schema_sha256: Annotated[str, typer.Option()] = DEFAULT_SCHEMA_SHA256,
) -> None:
    """Verify a standalone reviewed derived run, checksums, schema, and provenance."""

    verification = _reviewed_export_call(
        verify_derived_run,
        root=derived_root.resolve(),
        schema_path=schema_path.resolve(),
        schema_sha256=schema_sha256,
    )
    typer.echo(json.dumps(verification.model_dump(mode="json"), indent=2))


@app.command("evaluate-reference")
def evaluate_reference(
    reference_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    observations_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()] = Path("reference-score.json"),
) -> None:
    """Score already-extracted observations against prompt-isolated annotations."""

    result = score_reference_files(reference_path, observations_path, output)
    typer.echo(json.dumps(result, indent=2))


@app.command("audit-reference")
def audit_reference_command(
    reference_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    pdf_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Verify reference anchors against a content-addressed PDF."""

    result = audit_reference_pdf(reference_path, pdf_path)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
    if not result.passed:
        raise typer.Exit(1)


@app.command("audit-corpus-references")
def audit_corpus_references_command(
    corpus_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    frozen_run_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("runs/reference-audit.json"),
) -> None:
    """Audit every configured paper reference against its frozen PDF hash and anchors."""

    result = audit_corpus_references(
        load_corpus(corpus_path),
        project_root=Path.cwd().resolve(),
        frozen_run_root=frozen_run_root.resolve(),
        output_path=output.resolve(),
    )
    typer.echo(json.dumps(result, indent=2))
    if result["papers_failed"]:
        raise typer.Exit(1)


@app.command("prepare-human-review")
def prepare_human_review_command(
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    template: Annotated[Path, typer.Option()] = Path("runs/human-review/template.json"),
    report: Annotated[Path, typer.Option()] = Path("runs/human-review/review.html"),
    sample_size: Annotated[int, typer.Option(min=1, max=1000)] = 20,
) -> None:
    """Create a deterministic local review sample and quote-bearing HTML report."""

    result = write_human_review_artifacts(
        run_root,
        template_path=template,
        report_path=report,
        sample_size=sample_size,
    )
    typer.echo(
        json.dumps(
            {
                "audit_id": result.audit_id,
                "population_candidates": result.population_candidates,
                "sampled": len(result.items),
                "template": template.name,
                "report": report.name,
            },
            indent=2,
        )
    )


@app.command("summarize-human-review")
def summarize_human_review_command(
    template: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()] = Path("runs/human-review/public-summary.json"),
) -> None:
    """Publish only aggregate decisions from a fully completed local review template."""

    summary = summarize_human_review(template, output_path=output)
    typer.echo(json.dumps(summary, indent=2))


@app.command("build-extraction-review-cards")
def build_extraction_review_cards_command(
    development_run_root: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, readable=True),
    ] = None,
    holdout_run_root: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, readable=True),
    ] = None,
    development_human_review: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = None,
    holdout_human_review: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = None,
    development_human_review_template: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = None,
    holdout_human_review_template: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = None,
    development_eee_link_prefix: Annotated[str, typer.Option()] = "../../eee/development",
    holdout_eee_link_prefix: Annotated[str, typer.Option()] = "../../eee/holdout",
    holdout_posthoc_corrected: Annotated[
        bool,
        typer.Option(
            help=(
                "Label the holdout cards and split summary as post-hoc corrected; the "
                "immutable first run must be preserved separately."
            )
        ),
    ] = False,
    output: Annotated[Path, typer.Option()] = Path("runs/extraction-review-cards"),
) -> None:
    """Build deterministic public Extraction Review Cards for one or both splits."""

    if development_run_root is None and holdout_run_root is None:
        raise typer.BadParameter("provide at least one split run root")
    if development_human_review is not None and development_run_root is None:
        raise typer.BadParameter("development review requires a development run root")
    if holdout_human_review is not None and holdout_run_root is None:
        raise typer.BadParameter("holdout review requires a holdout run root")
    if development_human_review_template is not None and development_run_root is None:
        raise typer.BadParameter("development review template requires a development run root")
    if holdout_human_review_template is not None and holdout_run_root is None:
        raise typer.BadParameter("holdout review template requires a holdout run root")
    if holdout_posthoc_corrected and holdout_run_root is None:
        raise typer.BadParameter("post-hoc correction label requires a holdout run root")
    corpora: list[CorpusCardInput] = []
    if development_run_root is not None:
        corpora.append(
            CorpusCardInput(
                split="development",
                run_root=development_run_root,
                human_review_summary_path=development_human_review,
                paper_review_outcomes=(
                    project_paper_review_outcomes(
                        development_human_review_template,
                        run_root=development_run_root,
                    )
                    if development_human_review_template is not None
                    else {}
                ),
                eee_link_prefix=development_eee_link_prefix,
            )
        )
    if holdout_run_root is not None:
        corpora.append(
            CorpusCardInput(
                split="holdout",
                run_root=holdout_run_root,
                evaluation_status=("post_hoc_corrected" if holdout_posthoc_corrected else None),
                human_review_summary_path=holdout_human_review,
                paper_review_outcomes=(
                    project_paper_review_outcomes(
                        holdout_human_review_template,
                        run_root=holdout_run_root,
                    )
                    if holdout_human_review_template is not None
                    else {}
                ),
                eee_link_prefix=holdout_eee_link_prefix,
            )
        )
    destination = write_extraction_review_bundle(corpora, output)
    typer.echo(
        json.dumps(
            {
                "output": destination.as_posix(),
                "splits": [corpus.split for corpus in corpora],
                "index": "extraction-review-index.html",
                "checksums": "SHA256SUMS",
            },
            indent=2,
        )
    )


@app.command("export-public-snapshot")
def export_public_snapshot_command(
    snapshot_id: Annotated[str, typer.Argument()],
    corpus_run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    model_selection_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True)
    ],
    human_review_summary_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True)
    ],
    schema_path: Annotated[
        Path, typer.Option(exists=True, readable=True)
    ] = DEFAULT_EEE_SCHEMA_PATH,
    schema_sha256: Annotated[str, typer.Option()] = DEFAULT_SCHEMA_SHA256,
    output_root: Annotated[Path, typer.Option()] = Path("examples"),
    additional_run_root: Annotated[
        list[Path] | None,
        typer.Option("--additional-run-root", exists=True, file_okay=False, readable=True),
    ] = None,
    selected_model: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Export a deterministic, allowlist-only public pilot snapshot."""

    destination = build_public_snapshot(
        snapshot_id=snapshot_id,
        corpus_run_root=corpus_run_root,
        model_selection_path=model_selection_path,
        human_review_summary_path=human_review_summary_path,
        schema_path=schema_path,
        schema_sha256=schema_sha256,
        output_root=output_root,
        additional_run_roots=additional_run_root or (),
        selected_model=selected_model,
    )
    typer.echo(destination)


@app.command("build-public-development-summary")
def build_public_development_summary_command(
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option()] = Path("results/current-development-summary.json"),
    reviewed_derived_root: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, readable=True),
    ] = None,
    review_root: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, readable=True),
    ] = None,
) -> None:
    """Write a quote-free source aggregate and context-verified reviewed join. Offline."""

    try:
        digest = write_public_development_summary(
            run_root.resolve(),
            output.resolve(),
            reviewed_derived_root=(
                reviewed_derived_root.resolve() if reviewed_derived_root is not None else None
            ),
            review_root=review_root.resolve() if review_root is not None else None,
        )
    except PublicDevelopmentSummaryError as error:
        typer.echo(
            json.dumps(
                {
                    "status": "public-development-summary-not-written",
                    "detail": str(error),
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    except Exception:
        typer.echo(
            json.dumps(
                {
                    "status": "public-development-summary-not-written",
                    "detail": "Summary construction failed without displaying private data.",
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    typer.echo(
        json.dumps(
            {"status": "public-development-summary-written", "sha256": digest},
            indent=2,
        )
    )


@app.command("build-public-development-preview")
def build_public_development_preview_command(
    bundle_id: Annotated[str, typer.Argument()],
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    corpus_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    route_manifest_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True)
    ],
    expected_paper_count: Annotated[int, typer.Option(min=1)] = 10,
    output_root: Annotated[Path, typer.Option()] = Path("results"),
) -> None:
    """Build a sealed, quote-free, pre-human development preview. Offline."""

    try:
        destination = build_public_development_preview(
            bundle_id=bundle_id,
            run_root=run_root,
            corpus_path=corpus_path,
            route_manifest_path=route_manifest_path,
            output_root=output_root,
            expected_paper_count=expected_paper_count,
        )
        verification = verify_public_development_preview(destination)
    except PublicDevelopmentPreviewError as error:
        typer.echo(
            json.dumps(
                {
                    "status": "public-development-preview-not-written",
                    "detail": str(error),
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    except Exception:
        typer.echo(
            json.dumps(
                {
                    "status": "public-development-preview-not-written",
                    "detail": "Preview construction failed without displaying private data.",
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    typer.echo(
        json.dumps(
            {
                "status": "public-development-preview-written",
                "output": destination.as_posix(),
                "checksums_sha256": verification["checksums_sha256"],
            },
            indent=2,
        )
    )


@app.command("verify-public-development-preview")
def verify_public_development_preview_command(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
) -> None:
    """Verify a standalone pre-human development preview. Offline."""

    try:
        result = verify_public_development_preview(root)
    except PublicDevelopmentPreviewError as error:
        typer.echo(
            json.dumps(
                {
                    "status": "public-development-preview-verification-failed",
                    "detail": str(error),
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    except Exception:
        typer.echo(
            json.dumps(
                {
                    "status": "public-development-preview-verification-failed",
                    "detail": "Preview verification failed without displaying private data.",
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(result, indent=2))


@app.command("build-public-reviewed-development-bundle")
def build_public_reviewed_development_bundle_command(
    bundle_id: Annotated[str, typer.Argument()],
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    preview_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    reviewed_derived_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True),
    ],
    review_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True),
    ],
    output_root: Annotated[Path, typer.Option()] = Path("results"),
) -> None:
    """Close a verified preview with context-bound reviewed EEE. Offline."""

    try:
        destination = build_public_reviewed_development_bundle(
            bundle_id=bundle_id,
            run_root=run_root,
            preview_root=preview_root,
            reviewed_derived_root=reviewed_derived_root,
            review_root=review_root,
            output_root=output_root,
        )
        verification = verify_public_reviewed_development_bundle(destination)
    except PublicReviewedDevelopmentBundleError as error:
        typer.echo(
            json.dumps(
                {
                    "status": "public-reviewed-development-bundle-not-written",
                    "detail": str(error),
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    except Exception:
        typer.echo(
            json.dumps(
                {
                    "status": "public-reviewed-development-bundle-not-written",
                    "detail": (
                        "Reviewed bundle construction failed without displaying private data."
                    ),
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    typer.echo(
        json.dumps(
            {
                "status": "public-reviewed-development-bundle-written",
                "output": destination.as_posix(),
                "checksums_sha256": verification["checksums_sha256"],
            },
            indent=2,
        )
    )


@app.command("verify-public-reviewed-development-bundle")
def verify_public_reviewed_development_bundle_command(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
) -> None:
    """Verify a public reviewed development bundle without private inputs. Offline."""

    try:
        result = verify_public_reviewed_development_bundle(root)
    except PublicReviewedDevelopmentBundleError as error:
        typer.echo(
            json.dumps(
                {
                    "status": "public-reviewed-development-bundle-verification-failed",
                    "detail": str(error),
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    except Exception:
        typer.echo(
            json.dumps(
                {
                    "status": "public-reviewed-development-bundle-verification-failed",
                    "detail": "Bundle verification failed without displaying private data.",
                },
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(result, indent=2))


@app.command("score-row-coverage")
def score_row_coverage_command(
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Measure how much of each shown table the extractor enumerated. Offline."""

    summary = score_run_row_coverage(run_root.resolve(), output.resolve() if output else None)
    compact = {
        key: summary[key]
        for key in (
            "run_root",
            "papers_scored",
            "tables_shown",
            "rows_shown",
            "rows_with_a_candidate",
            "row_coverage",
            "papers_with_zero_table_anchors",
        )
    }
    if "row_disposition_coverage" in summary:
        compact["row_disposition_coverage"] = summary["row_disposition_coverage"]
    typer.echo(json.dumps(compact, indent=2))


@app.command("plan-row-enumeration")
def plan_row_enumeration_command(
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    corpus_path: Annotated[
        Path,
        typer.Option("--corpus", exists=True, dir_okay=False, readable=True),
    ],
    model: Annotated[str, typer.Option(envvar="ERE_EXTRACTOR_MODEL")],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Reconstruct exact next-run block reuse and bounded row calls. Offline."""

    resolved_corpus_path = corpus_path.resolve()
    project_root = Path.cwd().resolve()
    report = build_next_run_row_plan_report(
        run_root.resolve(),
        corpus=load_corpus(resolved_corpus_path),
        project_root=project_root,
        extractor_model=model,
        output_path=None,
    )
    report["corpus_file_sha256"] = sha256_file(resolved_corpus_path)
    report["code"] = _code_state(project_root)
    if output is not None:
        write_json(output.resolve(), report)
    typer.echo(
        json.dumps(
            {
                "run_root": report["run_root"],
                "corpus_id": report["corpus_id"],
                "corpus_sha256": report["corpus_file_sha256"],
                "mode": report["mode"],
                "provider_or_network_calls": report["provider_or_network_calls"],
                "extractor_model": report["extractor_model"],
                "extractor_contract": report["extractor_contract"],
                "row_extractor_contract": report["row_extractor_contract"],
                "row_config": report["row_config"],
                "code": report["code"],
                "stored_artifact_comparison": report["stored_artifact_comparison"],
                "next_run_blocks": report["next_run_blocks"],
                "row_plan": report["row_plan"],
                "next_run_preflight": report["next_run_preflight"],
                "cost_estimate_status": report["cost_estimate_status"],
                "cost_estimates": report["cost_estimates"],
            },
            indent=2,
        )
    )


@app.command("propose-controls")
def propose_controls_command(
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    reference_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option()] = Path("runs/control-proposals.json"),
) -> None:
    """Propose negative-control regions from inside blocks extraction actually saw.

    Offline and read-only. The output is a worklist of unconfirmed proposals that no
    scorer reads; a human confirms each one before it may reach `references/`.
    """

    from proceedings_to_eee.extraction.pdf_layout import PdfLayout
    from proceedings_to_eee.extraction.result_blocks import ResultBlock
    from proceedings_to_eee.io import read_json
    from proceedings_to_eee.reference import load_reference

    proposals = []
    for paper_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        layout_path = paper_dir / "private" / "layout.json"
        blocks_path = paper_dir / "private" / "result-blocks.json"
        if not (layout_path.is_file() and blocks_path.is_file()):
            continue
        reference_path = reference_dir / f"{paper_dir.name}.yaml"
        proposals.extend(
            propose_controls(
                paper_id=paper_dir.name,
                layout=PdfLayout.model_validate(read_json(layout_path)),
                blocks=[ResultBlock.model_validate(item) for item in read_json(blocks_path)],
                reference=load_reference(reference_path) if reference_path.is_file() else None,
            )
        )
    document = worklist(proposals)
    write_json(output.resolve(), document)
    typer.echo(json.dumps({"output": output.as_posix(), **document["counts"]}, indent=2))


@app.command("score-attribution")
def score_attribution_command(
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Score deterministic attribution against a completed run. Offline and read-only."""

    summary = score_run_attribution(run_root.resolve(), output.resolve() if output else None)
    typer.echo(render_attribution_summary(summary))


@app.command("inspect-regions")
def inspect_regions_command(
    pdf: Annotated[Path, typer.Argument(exists=True, readable=True)],
    page: Annotated[int, typer.Option(min=1)],
    source_id: Annotated[str, typer.Option()] = "inspection",
    quote: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Show the deterministic region index for one page, and optionally locate a quote.

    Offline: the page is parsed locally and no model is consulted.
    """

    layout = extract_pdf_layout(pdf, source_id)
    if page > layout.page_count:
        raise typer.BadParameter(f"page {page} is beyond the {layout.page_count}-page PDF")
    fragment = layout.pages[page - 1]
    index = build_page_region_index(fragment)
    payload: dict[str, object] = {
        "page": page,
        "panel_columns": index.panel_columns,
        "regions": [
            {
                "kind": region.kind.value,
                "lines": [region.span.start_line, region.span.end_line],
                "columns": [region.span.column_start, region.span.column_end],
                "section": region.section_path,
                "table_label": region.table_label,
                "caption": region.caption.text if region.caption else None,
                "rows": [
                    {"line": row.line, "label": row.effective_row_label, "header": row.is_header}
                    for row in region.rows
                ],
            }
            for region in index.regions
        ],
    }
    if quote is not None:
        location = locate_quote(fragment, quote)
        payload["located"] = (
            location.model_dump(mode="json", exclude_none=True) if location else None
        )
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command("replay-verifier")
def replay_verifier_command(
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    verifier_model: Annotated[str, typer.Option(envvar="ERE_VERIFIER_MODEL")],
    output: Annotated[Path, typer.Option()] = Path("runs/verifier-replay"),
    scope: Annotated[ReplayScope, typer.Option()] = ReplayScope.EXPORT_GATE,
    max_tokens: Annotated[int, typer.Option(min=1)] = 2_000,
    concurrency: Annotated[int, typer.Option(min=1, max=16)] = 4,
    paper_id: Annotated[list[str] | None, typer.Option("--paper-id")] = None,
    max_candidates_per_paper: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Replay the independent verifier over a completed run without re-extracting.

    The run tree is read-only. Verdicts, secret-free call telemetry, and the binding
    ledger are written under ``output``. The replay is resumable and never reads a
    reference annotation.
    """

    settings = ReplaySettings(
        run_root=run_root.resolve(),
        output_root=output.resolve(),
        verifier_model=verifier_model,
        scope=scope,
        max_tokens=max_tokens,
        concurrency=concurrency,
        paper_ids=tuple(paper_id or ()),
        max_candidates_per_paper=max_candidates_per_paper,
    )
    client = OpenRouterClient(api_key=runtime_key())
    summary = replay_run(client=client, settings=settings)
    typer.echo(json.dumps({"totals": summary["totals"], "cost": summary["cost"]}, indent=2))


@app.command("measure-verifier-replay")
def measure_verifier_replay_command(
    run_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    replay_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
) -> None:
    """Join replay verdicts to the frozen reference score. Offline; makes no calls."""

    report = measure_replay(run_root=run_root.resolve(), replay_root=replay_root.resolve())
    typer.echo(json.dumps({"classes": report["classes"], "headline": report["headline"]}, indent=2))


@app.command("prepare-mixed-development-annotation")
def prepare_mixed_development_annotation_command(
    selection_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    run_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    source_project_root: Annotated[
        Path,
        typer.Option("--source-project-root", exists=True, file_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option()],
    reviewer: Annotated[str, typer.Option()] = "reviewer-primary",
) -> None:
    """Create one private, label-free mixed development-review bundle. Offline."""

    manifest, digest = _private_annotation_call(
        prepare_development_annotation_package,
        selection_path=selection_path.resolve(),
        run_root=run_root.resolve(),
        source_project_root=source_project_root.resolve(),
        output_dir=output.resolve(),
        public_repo_root=Path.cwd().resolve(),
        reviewer=reviewer,
    )
    typer.echo(
        json.dumps(
            {
                "status": manifest.status,
                "item_count": manifest.item_count,
                "result_candidate_count": manifest.result_candidate_count,
                "result_candidate_paper_count": manifest.result_candidate_paper_count,
                "paper_count": manifest.paper_count,
                "single_reviewer_development_only": True,
                "future_unseen_data_used": False,
                "package_manifest_sha256": digest,
            },
            indent=2,
        )
    )


@app.command("validate-mixed-development-annotation-response")
def validate_mixed_development_annotation_response_command(
    package_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    selection_path: Annotated[
        Path,
        typer.Option("--selection", exists=True, dir_okay=False, readable=True),
    ],
    run_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    source_project_root: Annotated[
        Path,
        typer.Option("--source-project-root", exists=True, file_okay=False, readable=True),
    ],
    package_manifest_sha256: Annotated[str, typer.Option("--package-manifest-sha256")],
) -> None:
    """Validate one completed development response and both evidence anchors."""

    receipt = _private_annotation_call(
        validate_completed_development_annotation_response,
        package_dir=package_dir.resolve(),
        selection_path=selection_path.resolve(),
        run_root=run_root.resolve(),
        source_project_root=source_project_root.resolve(),
        public_repo_root=Path.cwd().resolve(),
        expected_manifest_sha256=package_manifest_sha256,
    )
    typer.echo(
        json.dumps(
            {
                "status": "complete-valid-development-response",
                "records": len(receipt.responses),
                "response_sha256": receipt.response_sha256,
                "single_reviewer_development_only": True,
                "agreement_measured": False,
            },
            indent=2,
        )
    )


@app.command("prepare-control-annotation-workspace")
def prepare_control_annotation_workspace_command(
    packet_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    run_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Create two isolated private human-annotation working copies. Offline."""

    manifest, digest = _private_annotation_call(
        prepare_annotation_workspace,
        packet_dir=packet_dir.resolve(),
        run_root=run_root.resolve(),
        project_root=Path.cwd().resolve(),
        output_dir=output.resolve(),
    )
    typer.echo(
        json.dumps(
            {
                "status": manifest.status,
                "item_count": manifest.item_count,
                "evaluation_denominator": manifest.evaluation_denominator,
                "frozen_pdfs_per_annotator": len(manifest.bundles[0].pdfs),
                "practice_example_in_denominator": False,
                "workspace_manifest_sha256": digest,
            },
            indent=2,
        )
    )


@app.command("validate-control-annotation-response")
def validate_control_annotation_response_command(
    workspace_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    packet_dir: Annotated[
        Path,
        typer.Option("--packet", exists=True, file_okay=False, readable=True),
    ],
    run_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    annotator: Annotated[str, typer.Option()],
) -> None:
    """Validate one completed mutable response without expecting its blank hash."""

    receipt = _private_annotation_call(
        validate_workspace_response,
        packet_dir=packet_dir.resolve(),
        workspace_dir=workspace_dir.resolve(),
        annotator=annotator,
        run_root=run_root.resolve(),
        project_root=Path.cwd().resolve(),
    )
    typer.echo(
        json.dumps(
            {
                "status": "complete-valid",
                "records": len(receipt.responses),
                "response_sha256": receipt.sha256,
            },
            indent=2,
        )
    )


@app.command("lock-control-annotation-responses")
def lock_control_annotation_responses_command(
    workspace_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    packet_dir: Annotated[
        Path,
        typer.Option("--packet", exists=True, file_okay=False, readable=True),
    ],
    run_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Lock both completed originals, then compute pre-adjudication agreement."""

    manifest, agreement, digest = _private_annotation_call(
        lock_completed_responses,
        packet_dir=packet_dir.resolve(),
        workspace_dir=workspace_dir.resolve(),
        run_root=run_root.resolve(),
        project_root=Path.cwd().resolve(),
        output_dir=output.resolve(),
    )
    typer.echo(
        json.dumps(
            {
                "status": manifest.status,
                "completion_manifest_sha256": digest,
                "denominator": agreement.denominator,
                "raw_agreement_count": agreement.raw_agreement_count,
                "raw_agreement": agreement.raw_agreement,
                "cohen_kappa": agreement.cohen_kappa,
                "kappa_status": agreement.kappa_status,
                "adjudication_required_count": manifest.adjudication_required_count,
            },
            indent=2,
        )
    )


@app.command("measure-control-annotation-agreement")
def measure_control_annotation_agreement_command(
    completion_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    workspace_dir: Annotated[
        Path,
        typer.Option("--workspace", exists=True, file_okay=False, readable=True),
    ],
    packet_dir: Annotated[
        Path,
        typer.Option("--packet", exists=True, file_okay=False, readable=True),
    ],
    run_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    completion_manifest_sha256: Annotated[str, typer.Option("--completion-manifest-sha256")],
) -> None:
    """Revalidate locked originals and print aggregate pre-adjudication agreement."""

    completion = _private_annotation_call(
        validate_completion_bundle,
        packet_dir=packet_dir.resolve(),
        workspace_dir=workspace_dir.resolve(),
        completion_dir=completion_dir.resolve(),
        run_root=run_root.resolve(),
        project_root=Path.cwd().resolve(),
        expected_manifest_sha256=completion_manifest_sha256,
    )
    agreement = completion.agreement
    typer.echo(
        json.dumps(
            {
                "status": "pre-adjudication-agreement-valid",
                "denominator": agreement.denominator,
                "category_order": agreement.categories,
                "category_counts": {
                    "annotator_a": agreement.annotator_category_counts[agreement.row_annotator],
                    "annotator_b": agreement.annotator_category_counts[agreement.column_annotator],
                },
                "confusion_matrix": agreement.confusion_matrix_counts,
                "raw_agreement_count": agreement.raw_agreement_count,
                "raw_agreement": agreement.raw_agreement,
                "expected_agreement": agreement.expected_agreement,
                "cohen_kappa": agreement.cohen_kappa,
                "kappa_status": agreement.kappa_status,
            },
            indent=2,
        )
    )


@app.command("prepare-control-annotation-adjudication")
def prepare_control_annotation_adjudication_command(
    completion_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    workspace_dir: Annotated[
        Path,
        typer.Option("--workspace", exists=True, file_okay=False, readable=True),
    ],
    packet_dir: Annotated[
        Path,
        typer.Option("--packet", exists=True, file_okay=False, readable=True),
    ],
    run_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    completion_manifest_sha256: Annotated[str, typer.Option("--completion-manifest-sha256")],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Prepare a separate private adjudication subset from locked responses."""

    manifest, digest = _private_annotation_call(
        prepare_adjudication_workspace,
        packet_dir=packet_dir.resolve(),
        workspace_dir=workspace_dir.resolve(),
        completion_dir=completion_dir.resolve(),
        run_root=run_root.resolve(),
        project_root=Path.cwd().resolve(),
        output_dir=output.resolve(),
        expected_completion_manifest_sha256=completion_manifest_sha256,
    )
    typer.echo(
        json.dumps(
            {
                "status": manifest.status,
                "adjudication_required_count": manifest.adjudication_required_count,
                "adjudication_manifest_sha256": digest,
                "primary_responses_overwritten": False,
            },
            indent=2,
        )
    )


@app.command("validate-control-annotation-adjudication")
def validate_control_annotation_adjudication_command(
    adjudication_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    completion_dir: Annotated[
        Path,
        typer.Option("--completion", exists=True, file_okay=False, readable=True),
    ],
    workspace_dir: Annotated[
        Path,
        typer.Option("--workspace", exists=True, file_okay=False, readable=True),
    ],
    packet_dir: Annotated[
        Path,
        typer.Option("--packet", exists=True, file_okay=False, readable=True),
    ],
    run_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    completion_manifest_sha256: Annotated[str, typer.Option("--completion-manifest-sha256")],
) -> None:
    """Validate a completed adjudication without changing either primary response."""

    receipt = _private_annotation_call(
        validate_adjudication_workspace,
        packet_dir=packet_dir.resolve(),
        workspace_dir=workspace_dir.resolve(),
        completion_dir=completion_dir.resolve(),
        adjudication_dir=adjudication_dir.resolve(),
        run_root=run_root.resolve(),
        project_root=Path.cwd().resolve(),
        expected_completion_manifest_sha256=completion_manifest_sha256,
        require_complete=True,
    )
    typer.echo(
        json.dumps(
            {
                "status": "adjudication-complete-valid",
                "records": receipt.record_count,
                "response_sha256": receipt.sha256,
                "primary_responses_overwritten": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
