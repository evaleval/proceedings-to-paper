"""Small, explicit command-line surface for reproducible pilot runs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from proceedings_to_eee.corpus import CorpusSpec, load_corpus
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
from proceedings_to_eee.evaluation.human_review import (
    project_paper_review_outcomes,
    summarize_human_review,
    write_human_review_artifacts,
)
from proceedings_to_eee.evaluation.reference_audit import audit_reference_pdf
from proceedings_to_eee.evaluation.reference_score import score_reference_files
from proceedings_to_eee.evaluation.row_coverage import score_run_row_coverage
from proceedings_to_eee.evaluation.row_plan import build_next_run_row_plan_report
from proceedings_to_eee.extraction.pdf_layout import extract_pdf_layout, select_result_pages
from proceedings_to_eee.extraction.region_index import build_page_region_index, locate_quote
from proceedings_to_eee.io import sha256_file, write_json
from proceedings_to_eee.pipeline import (
    PipelineSettings,
    _code_state,
    freeze_corpus,
    run_corpus,
    runtime_key,
)
from proceedings_to_eee.providers.openrouter import OpenRouterClient
from proceedings_to_eee.public_snapshot import build_public_snapshot
from proceedings_to_eee.reporting.extraction_review_cards import (
    CorpusCardInput,
    write_extraction_review_bundle,
)
from proceedings_to_eee.reporting.public_development_summary import (
    PublicDevelopmentSummaryError,
    write_public_development_summary,
)
from proceedings_to_eee.resources import DEFAULT_EEE_SCHEMA_PATH, EEE_SCHEMA_SHA256
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


def _settings(
    *,
    schema_path: Path,
    schema_sha256: str,
    output: Path,
    model: str,
    min_confidence: float = 0.8,
    row_enumeration_enabled: bool = False,
    row_estimated_call_cost_usd: float | None = None,
    verifier_model: str | None = None,
) -> PipelineSettings:
    return PipelineSettings(
        project_root=Path.cwd().resolve(),
        schema_path=schema_path.resolve(),
        schema_sha256=schema_sha256,
        output_root=output.resolve(),
        model=model,
        min_confidence=min_confidence,
        row_enumeration_enabled=row_enumeration_enabled,
        row_estimated_call_cost_usd=row_estimated_call_cost_usd,
        verifier_model=verifier_model,
    )


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
    verifier_model: Annotated[str | None, typer.Option(envvar="ERE_VERIFIER_MODEL")] = None,
    output: Annotated[Path, typer.Option()] = Path("runs/latest"),
    schema_sha256: Annotated[str, typer.Option()] = DEFAULT_SCHEMA_SHA256,
    min_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
    row_enumeration: Annotated[
        bool,
        typer.Option(
            help=(
                "Run the bounded dense-table row stage in addition to legacy block "
                "extraction. This incurs additional provider calls."
            )
        ),
    ] = False,
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
    paper_id: Annotated[str | None, typer.Option()] = None,
    quiet: Annotated[bool, typer.Option()] = False,
) -> None:
    """Freeze, extract, validate, compose, and report a corpus."""

    settings = _settings(
        schema_path=schema_path.resolve(),
        schema_sha256=schema_sha256,
        output=output.resolve(),
        model=model,
        min_confidence=min_confidence,
        row_enumeration_enabled=row_enumeration,
        row_estimated_call_cost_usd=row_estimated_call_cost_usd,
        verifier_model=verifier_model,
    )
    client = OpenRouterClient(api_key=runtime_key())
    corpus = _paper_subset(load_corpus(corpus_path), paper_id)
    summary = run_corpus(corpus=corpus, settings=settings, client=client)
    if not quiet:
        typer.echo(json.dumps(summary["totals"], indent=2))
    if summary["papers_failed"]:
        raise typer.Exit(1)


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
) -> None:
    """Download and content-address all paper sources without using OpenRouter."""

    settings = _settings(
        schema_path=schema_path,
        schema_sha256=schema_sha256,
        output=output,
        model="not-used",
    )
    summary = freeze_corpus(load_corpus(corpus_path), settings)
    typer.echo(json.dumps(summary, indent=2))
    if summary["papers_failed"]:
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
) -> None:
    """Write a quote-free, path-free aggregate for one development run. Offline."""

    try:
        digest = write_public_development_summary(run_root.resolve(), output.resolve())
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
