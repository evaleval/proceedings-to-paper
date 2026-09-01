"""Dependency-free static extraction review report."""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.io import atomic_write_bytes
from proceedings_to_eee.sources.manifest import SourceManifest


def _e(value: object) -> str:
    return html.escape("" if value is None else str(value))


def render_review_report(
    *,
    manifest: SourceManifest,
    candidates: Iterable[CandidateObservation],
    eee_records: list[dict],
    validation_errors: dict[str, list[str]],
    output_path: Path,
) -> None:
    """Write a readable provenance/review view, explicitly not a card."""

    candidates = list(candidates)
    claim_counts = Counter(candidate.claim_type for candidate in candidates)
    export_counts = Counter(candidate.export_status for candidate in candidates)
    rows: list[str] = []
    for candidate in candidates:
        system = next(
            (role.raw_name for role in candidate.roles if role.role == "evaluated_system"),
            "-",
        )
        scope = candidate.scope.dataset_raw if candidate.scope else "-"
        if candidate.scope and candidate.scope.group:
            scope += f" · group={candidate.scope.group}"
        metric = candidate.metric.raw_name if candidate.metric else "-"
        value = candidate.value.raw if candidate.value else "-"
        evidence = "<br>".join(
            f"p.{anchor.page} {_e(anchor.label or anchor.kind)}: <code>{_e(anchor.quote)}</code>"
            for anchor in candidate.evidence
        )
        origin = "-"
        if candidate.attribution is not None:
            cue_ids = ", ".join(cue.cue_id for cue in candidate.attribution.cues) or "no cues"
            origin = (
                f"<strong>{_e(candidate.attribution.state)}</strong><br>"
                f"{_e(candidate.attribution.rule_id)} · {_e(cue_ids)}"
            )
        rows.append(
            "<tr>"
            f"<td><code>{_e(candidate.observation_id)}</code></td>"
            f"<td>{_e(candidate.claim_type)}</td><td>{_e(system)}</td>"
            f"<td>{_e(scope)}</td><td>{_e(metric)}</td><td>{_e(value)}</td>"
            f"<td>{_e(candidate.text_support)} / {_e(candidate.referential_status)}</td>"
            f"<td>{origin}</td>"
            f"<td><strong>{_e(candidate.export_status)}</strong><br>{_e(candidate.export_reason)}</td>"
            f"<td>{evidence}</td></tr>"
        )
    source_rows = "".join(
        "<tr>"
        f"<td>{_e(source.role)}</td><td><code>{_e(source.source_id)}</code></td>"
        f"<td><code>{_e(source.sha256)}</code></td><td>{_e(source.retrieved_at)}</td>"
        f"<td>{_e(source.original_uri)}</td></tr>"
        for source in manifest.sources
    )
    errors = sum(len(items) for items in validation_errors.values())
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Extraction review · {_e(manifest.title)}</title>
<style>
:root{{--ink:#162033;--muted:#637086;--paper:#fff;--line:#dce2ea;--accent:#155eef;--bg:#f4f7fb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
main{{max-width:1500px;margin:32px auto;padding:0 24px}}h1{{font-size:28px;margin-bottom:4px}}h2{{margin-top:32px}}
.notice{{background:#e9f0ff;border-left:4px solid var(--accent);padding:12px 16px;margin:18px 0}}
.metrics{{display:flex;gap:12px;flex-wrap:wrap}}.metric{{background:var(--paper);border:1px solid var(--line);padding:14px 18px;border-radius:10px;min-width:150px}}
.metric b{{display:block;font-size:24px}}table{{border-collapse:collapse;width:100%;background:var(--paper)}}th,td{{border:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}}th{{background:#edf2f8;position:sticky;top:0}}code{{white-space:pre-wrap;font-size:12px}}.scroll{{overflow:auto;max-height:68vh;border-radius:10px}}
</style></head><body><main>
<h1>{_e(manifest.title)}</h1><div>Paper ID: <code>{_e(manifest.paper_id)}</code></div>
<div class="notice"><strong>Extraction Review</strong>: diagnostic evidence and decisions. This is not an Evaluation Card and not the EEE dataset itself.</div>
<div class="metrics">
<div class="metric"><b>{len(candidates)}</b>candidates</div>
<div class="metric"><b>{sum(1 for c in candidates if str(c.export_status) == "exported")}</b>exported</div>
<div class="metric"><b>{len(eee_records)}</b>EEE records</div>
<div class="metric"><b>{errors}</b>schema issues</div>
</div>
<p>Claim classes: {_e(dict(claim_counts))}<br>Export states: {_e(dict(export_counts))}</p>
<h2>Frozen sources</h2><div class="scroll"><table><thead><tr><th>Role</th><th>ID</th><th>SHA-256</th><th>Retrieved</th><th>Origin</th></tr></thead><tbody>{source_rows}</tbody></table></div>
<h2>Atomic observations</h2><div class="scroll"><table><thead><tr><th>ID</th><th>Claim</th><th>Evaluated system</th><th>Scope</th><th>Metric</th><th>Value</th><th>Checks</th><th>Origin</th><th>Export</th><th>Evidence</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
</main></body></html>"""
    atomic_write_bytes(output_path, document.encode("utf-8"))
