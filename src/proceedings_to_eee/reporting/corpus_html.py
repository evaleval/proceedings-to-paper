"""Render a deterministic, self-contained HTML summary for a corpus run."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

__all__ = ["render_corpus_html", "render_corpus_html_file"]


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _non_negative_int(value: Any, field: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer, not a boolean")
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return len(value)
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, int | float) and math.isfinite(value) and value >= 0:
        integer = int(value)
        if integer == value:
            return integer
    raise ValueError(f"{field} must be a non-negative integer or a collection")


def _non_negative_float(value: Any, field: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a non-negative finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a non-negative finite number")
    return number


def _status_is_pass(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.casefold() in {"ok", "pass", "passed", "success", "valid"}
    if isinstance(value, Mapping):
        return _status_is_pass(_first(value, "status", "result", "passed"))
    return False


def _spot_checks(run: Mapping[str, Any], counts: Mapping[str, Any]) -> tuple[int, int]:
    if "spot_checks" in counts or "spot_checks_exact" in counts:
        passed = _non_negative_int(counts.get("spot_checks_exact"), "spot_checks_exact")
        total = _non_negative_int(counts.get("spot_checks"), "spot_checks")
    else:
        raw = run.get("spot_checks")
        if isinstance(raw, Mapping):
            checks = raw.get("checks")
            if isinstance(checks, Sequence) and not isinstance(checks, str | bytes | bytearray):
                return sum(_status_is_pass(check) for check in checks), len(checks)
            passed = _non_negative_int(
                _first(raw, "passed", "pass_count", default=0), "spot_checks.passed"
            )
            total_value = _first(raw, "total", "count")
            if total_value is None and "failed" in raw:
                failed = _non_negative_int(raw["failed"], "spot_checks.failed")
                total_value = passed + failed
            total = _non_negative_int(total_value, "spot_checks.total")
        elif isinstance(raw, Sequence) and not isinstance(raw, str | bytes | bytearray):
            passed, total = sum(_status_is_pass(check) for check in raw), len(raw)
        elif raw is None:
            passed = _non_negative_int(
                _first(run, "spot_checks_passed", "spot_check_passed", default=0),
                "spot_checks_passed",
            )
            total = _non_negative_int(
                _first(run, "spot_checks_total", "spot_check_total", default=0),
                "spot_checks_total",
            )
        else:
            raise ValueError("spot_checks must be an object or a list")
    if passed > total:
        raise ValueError("spot-check passes cannot exceed the total")
    return passed, total


def _count(run: Mapping[str, Any], counts: Mapping[str, Any], field: str, *aliases: str) -> int:
    value = _first(counts, field, *aliases)
    if value is None:
        value = _first(run, field, *aliases)
    return _non_negative_int(value, field)


def _provider_cost(run: Mapping[str, Any]) -> float:
    explicit = _first(run, "cost_usd", "total_cost_usd", "cost")
    if explicit is not None:
        return _non_negative_float(explicit, "cost_usd")
    total = 0.0
    for stage_name in ("extractor", "verifier"):
        stage = run.get(stage_name)
        if not isinstance(stage, Mapping):
            continue
        telemetry = stage.get("successful_call_telemetry")
        if isinstance(telemetry, Mapping):
            total += _non_negative_float(
                telemetry.get("cost_usd_lower_bound"),
                f"{stage_name}.successful_call_telemetry.cost_usd_lower_bound",
            )
            continue
        calls: list[Any] = []
        for field in ("calls", "resumed_calls"):
            raw_calls = stage.get(field)
            if isinstance(raw_calls, Sequence) and not isinstance(
                raw_calls, str | bytes | bytearray
            ):
                calls.extend(raw_calls)
        for index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                raise ValueError(f"{stage_name}.calls items must be objects")
            total += _non_negative_float(
                call.get("cost_usd"), f"{stage_name}.calls[{index}].cost_usd"
            )
    return total


def _paper_projection(run: Mapping[str, Any], index: int) -> dict[str, Any]:
    raw_counts = run.get("counts", {})
    if not isinstance(raw_counts, Mapping):
        raise ValueError("run counts must be an object")
    paper_id = str(_first(run, "paper_id", "id", "source_id", default=f"paper-{index}"))
    title = str(_first(run, "title", "paper_title", default=paper_id))
    candidates = _count(run, raw_counts, "candidates", "candidate_count", "candidates_count")
    exports = _count(
        run,
        raw_counts,
        "exported",
        "export_count",
        "export_eligible_count",
        "eligible_count",
        "exports",
    )
    eee_records = _count(
        run, raw_counts, "eee_records", "eee_record_count", "eee_count", "eee_files"
    )
    schema_errors = _count(
        run,
        raw_counts,
        "eee_schema_issues",
        "schema_error_count",
        "schema_errors_count",
        "schema_errors",
    )
    spot_passed, spot_total = _spot_checks(run, raw_counts)
    runtime = _non_negative_float(
        _first(
            run,
            "wall_clock_seconds",
            "runtime_seconds",
            "duration_seconds",
            "runtime",
            default=0.0,
        ),
        "wall_clock_seconds",
    )
    extractor = run.get("extractor") if isinstance(run.get("extractor"), Mapping) else {}
    telemetry = (
        extractor.get("successful_call_telemetry") if isinstance(extractor, Mapping) else None
    )
    execution = extractor.get("execution") if isinstance(extractor, Mapping) else None
    if not isinstance(telemetry, Mapping):
        telemetry = {}
    if not isinstance(execution, Mapping):
        execution = {}
    return {
        "paper_id": paper_id,
        "title": title,
        "run_status": str(run.get("status", "success")),
        "candidates": candidates,
        "exports": exports,
        "eee_records": eee_records,
        "schema_errors": schema_errors,
        "spot_passed": spot_passed,
        "spot_total": spot_total,
        "cost": _provider_cost(run),
        "runtime": runtime,
        "successful_calls": _non_negative_int(telemetry.get("calls"), "successful calls"),
        "total_tokens": _non_negative_int(
            telemetry.get("total_tokens_lower_bound"), "total tokens"
        ),
        "retries": _non_negative_int(telemetry.get("retries_lower_bound"), "retries"),
        "failed_blocks": _non_negative_int(execution.get("blocks_failed"), "failed blocks"),
        "resumed_blocks": _non_negative_int(execution.get("blocks_resumed"), "resumed blocks"),
    }


def _runs(corpus_run: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    raw_runs = corpus_run.get("runs")
    if raw_runs is None:
        raw_runs = corpus_run.get("papers")
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, str | bytes | bytearray):
        raise ValueError("corpus run must contain a runs array")
    if any(not isinstance(run, Mapping) for run in raw_runs):
        raise ValueError("every runs item must be a JSON object")
    return raw_runs


def _format_cost(value: float) -> str:
    if value == 0:
        return "$0.0000"
    if value < 0.0001:
        return "&lt;$0.0001"
    return f"${value:,.4f}"


def _format_runtime(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _format_rate(value: Any) -> str:
    if value is None:
        return "Not measured"
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("quality rate must be numeric or null")
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > 1:
        raise ValueError("quality rate must be between zero and one")
    return f"{number * 100:.1f}%"


def _spot_display(passed: int, total: int) -> str:
    return "Not run" if total == 0 else f"{passed}/{total}"


def _row_status(paper: Mapping[str, Any]) -> tuple[str, str]:
    if paper["run_status"] != "success":
        return "error", "Run failed"
    if paper["schema_errors"]:
        return "error", "Schema errors"
    if paper["spot_total"] == 0:
        return "pending", "Spot-check pending"
    if paper["spot_passed"] < paper["spot_total"]:
        return "warning", "Review needed"
    return "ok", "Ready"


def render_corpus_html(corpus_run: Mapping[str, Any]) -> str:
    """Return a standalone aggregate report for a parsed ``corpus-run.json``.

    Run order is preserved. Aggregate values are recomputed from individual
    runs rather than trusting the precomputed ``totals`` object.
    """

    if not isinstance(corpus_run, Mapping):
        raise ValueError("corpus run must be a JSON object")
    papers = [_paper_projection(run, index) for index, run in enumerate(_runs(corpus_run), start=1)]
    totals = {
        "candidates": sum(paper["candidates"] for paper in papers),
        "exports": sum(paper["exports"] for paper in papers),
        "eee_records": sum(paper["eee_records"] for paper in papers),
        "schema_errors": sum(paper["schema_errors"] for paper in papers),
        "spot_passed": sum(paper["spot_passed"] for paper in papers),
        "spot_total": sum(paper["spot_total"] for paper in papers),
        "cost": sum(paper["cost"] for paper in papers),
        "runtime": sum(paper["runtime"] for paper in papers),
        "successful_calls": sum(paper["successful_calls"] for paper in papers),
        "total_tokens": sum(paper["total_tokens"] for paper in papers),
        "retries": sum(paper["retries"] for paper in papers),
        "failed_blocks": sum(paper["failed_blocks"] for paper in papers),
        "resumed_blocks": sum(paper["resumed_blocks"] for paper in papers),
    }
    run_id = escape(
        str(_first(corpus_run, "corpus_id", "run_id", "corpus_run_id", "id", default="Corpus run"))
    )
    generated_value = _first(corpus_run, "generated_at", "created_at", "retrieved_at")
    generated_at = escape(str(generated_value)) if generated_value is not None else ""
    paper_word = "paper" if len(papers) == 1 else "papers"

    rows: list[str] = []
    for index, paper in enumerate(papers, start=1):
        status_class, status_label = _row_status(paper)
        escaped_id = escape(paper["paper_id"], quote=True)
        escaped_title = escape(paper["title"])
        schema_class = "has-errors" if paper["schema_errors"] else ""
        spot_checks = _spot_display(paper["spot_passed"], paper["spot_total"])
        rows.append(
            f"""          <tr class="paper-row status-{status_class}"
              data-paper-id="{escaped_id}">
            <td class="row-number">{index}</td>
            <td class="paper-cell"><strong>{escaped_title}</strong><span>{escaped_id}</span></td>
            <td><span class="badge badge-{status_class}">{status_label}</span></td>
            <td class="number">{paper["candidates"]:,}</td>
            <td class="number">{paper["exports"]:,}</td>
            <td class="number">{paper["eee_records"]:,}</td>
            <td class="number {schema_class}">{paper["schema_errors"]:,}</td>
            <td class="number">{spot_checks}</td>
            <td class="number">{_format_cost(paper["cost"])}</td>
            <td class="number">{_format_runtime(paper["runtime"])}</td>
          </tr>"""
        )

    generated_meta = f"<span>Generated {generated_at}</span>" if generated_at else ""
    table_body = "\n".join(rows) or (
        '          <tr class="empty-row"><td colspan="10">No papers in this run.</td></tr>'
    )
    spot_total = _spot_display(totals["spot_passed"], totals["spot_total"])
    schema_card_class = "error" if totals["schema_errors"] else ""
    reference_evaluation = corpus_run.get("reference_evaluation")
    if not isinstance(reference_evaluation, Mapping):
        reference_evaluation = {}
    detection = reference_evaluation.get("detection")
    if not isinstance(detection, Mapping):
        detection = {}
    fields = reference_evaluation.get("field_accuracy")
    if not isinstance(fields, Mapping):
        fields = {}
    negative_safety = reference_evaluation.get("negative_control_safety")
    if not isinstance(negative_safety, Mapping):
        negative_safety = {}
    reference_recall = _format_rate(detection.get("recall"))
    reference_precision = _format_rate(detection.get("precision"))
    joint_accuracy = _format_rate(fields.get("joint_semantics"))
    false_primary_controls = _non_negative_int(
        negative_safety.get("false_primary_count"), "false_primary_count"
    )
    false_primary_card_class = "error" if false_primary_controls else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{run_id} · Proceedings to EEE</title>
  <style>
    :root {{ color-scheme:light; --ink:#172033; --muted:#667085; --line:#e4e7ec;
      --paper:#fff; --canvas:#f5f7fb; --accent:#4058d6; --good:#067647;
      --warn:#b54708; --bad:#b42318; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--canvas); color:var(--ink); font:14px/1.45 Inter,
      ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1440px,calc(100% - 40px)); margin:36px auto 48px; }}
    header {{ display:flex; align-items:end; justify-content:space-between; gap:24px;
      margin-bottom:22px; }}
    h1 {{ margin:0 0 5px; font-size:30px; letter-spacing:-.025em; }}
    .eyebrow {{ color:var(--accent); font-size:12px; font-weight:750; letter-spacing:.09em;
      text-transform:uppercase; }}
    .subtitle,.meta {{ color:var(--muted); }} .meta {{ white-space:nowrap; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:12px;
      margin-bottom:18px; }}
    .card {{ background:var(--paper); border:1px solid var(--line); border-radius:12px;
      padding:15px 16px; box-shadow:0 1px 2px rgb(16 24 40 / 4%); }}
    .card span {{ display:block; color:var(--muted); font-size:12px; margin-bottom:5px; }}
    .card strong {{ font-size:22px; letter-spacing:-.02em; }}
    .card.error strong {{ color:var(--bad); }}
    .table-shell {{ overflow:auto; background:var(--paper); border:1px solid var(--line);
      border-radius:14px; box-shadow:0 1px 3px rgb(16 24 40 / 5%); }}
    table {{ width:100%; border-collapse:collapse; min-width:1050px; }}
    th {{ padding:12px 13px; background:#fafbfc; color:#475467; border-bottom:1px solid var(--line);
      font-size:11px; letter-spacing:.045em; text-align:right; text-transform:uppercase; }}
    th:nth-child(-n+3) {{ text-align:left; }}
    td {{ padding:14px 13px; border-bottom:1px solid #eef0f4; vertical-align:middle; }}
    tbody tr:last-child td {{ border-bottom:0; }} tbody tr:hover {{ background:#fafbff; }}
    .row-number {{ width:38px; color:#98a2b3; }} .paper-cell {{ min-width:290px; }}
    .paper-cell strong,.paper-cell span {{ display:block; }}
    .paper-cell span {{ margin-top:2px; color:var(--muted); font-size:12px; }}
    .number {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .has-errors {{ color:var(--bad); font-weight:750; }}
    .badge {{ display:inline-block; border-radius:999px; padding:4px 8px; font-size:11px;
      font-weight:700; }}
    .badge-ok {{ background:#ecfdf3; color:var(--good); }}
    .badge-warning {{ background:#fffaeb; color:var(--warn); }}
    .badge-error {{ background:#fef3f2; color:var(--bad); }}
    .badge-pending {{ background:#f2f4f7; color:#475467; }}
    .empty-row td {{ padding:34px; color:var(--muted); text-align:center; }}
    footer {{ color:var(--muted); font-size:12px; margin-top:12px; }}
    @media (max-width:1000px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }}
      header {{ align-items:start; flex-direction:column; }} .meta {{ white-space:normal; }} }}
    @media print {{ body {{ background:#fff; }} main {{ width:100%; margin:0; }}
      .table-shell,.card {{ box-shadow:none; }} tbody tr:hover {{ background:transparent; }} }}
  </style>
</head>
<body><main>
  <header><div>
    <div class="eyebrow">Evidence-bound extraction</div>
    <h1>Proceedings → EEE corpus run</h1>
    <div class="subtitle">{run_id} · {len(papers)} {paper_word}</div>
  </div><div class="meta">{generated_meta}</div></header>
  <section class="cards" aria-label="Corpus totals">
    <div class="card"><span>Candidates</span><strong>{totals["candidates"]:,}</strong></div>
    <div class="card"><span>Exported</span><strong>{totals["exports"]:,}</strong></div>
    <div class="card"><span>EEE records</span><strong>{totals["eee_records"]:,}</strong></div>
    <div class="card {schema_card_class}"><span>Schema errors</span>
      <strong>{totals["schema_errors"]:,}</strong></div>
    <div class="card"><span>Spot-checks</span><strong>{spot_total}</strong></div>
    <div class="card"><span>Cost</span><strong>{_format_cost(totals["cost"])}</strong></div>
    <div class="card"><span>Successful calls</span>
      <strong>{totals["successful_calls"]:,}</strong></div>
    <div class="card"><span>Tokens</span><strong>{totals["total_tokens"]:,}+</strong></div>
    <div class="card"><span>Retries / failed blocks</span>
      <strong>{totals["retries"]:,} / {totals["failed_blocks"]:,}</strong></div>
    <div class="card"><span>Resumed blocks</span>
      <strong>{totals["resumed_blocks"]:,}</strong></div>
    <div class="card"><span>Runtime</span>
      <strong>{_format_runtime(totals["runtime"])}</strong></div>
    <div class="card"><span>Reference recall</span><strong>{reference_recall}</strong></div>
    <div class="card"><span>Covered precision</span><strong>{reference_precision}</strong></div>
    <div class="card"><span>Joint semantics</span><strong>{joint_accuracy}</strong></div>
    <div class="card {false_primary_card_class}"><span>False-primary controls</span>
      <strong>{false_primary_controls}</strong></div>
  </section>
  <section class="table-shell" aria-label="Paper results"><table>
    <thead><tr><th>#</th><th>Paper</th><th>Status</th><th>Candidates</th><th>Export</th>
      <th>EEE</th><th>Schema errors</th><th>Spot-checks</th><th>Cost</th>
      <th>Runtime</th></tr></thead>
    <tbody>
{table_body}
    </tbody>
  </table></section>
  <footer>Aggregate values are recomputed from paper runs. Cost, token, retry, and
    attempt totals are lower bounds when failed or superseded provider calls are unavailable.
    Raw source text is not embedded.</footer>
</main></body>
</html>
"""


def render_corpus_html_file(corpus_run_path: str | Path, output_path: str | Path) -> Path:
    """Load ``corpus-run.json`` and write its standalone HTML report."""

    parsed = json.loads(Path(corpus_run_path).read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError("corpus-run.json must contain a JSON object")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_corpus_html(parsed), encoding="utf-8")
    return destination
