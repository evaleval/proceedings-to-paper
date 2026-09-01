"""Render local-only, evidence-bearing human review templates."""

from __future__ import annotations

import html
import json

from proceedings_to_eee.evaluation.human_review import (
    HumanReviewTemplate,
    PaperWithoutCandidatesReviewItem,
    PaperWithoutEEEReviewItem,
)


def _json_view(value: object) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def render_human_review_html(
    template: HumanReviewTemplate,
    *,
    decision_file_name: str = "human-review-template.json",
) -> str:
    """Return a standalone local report; evidence quotes are intentionally visible."""

    cards: list[str] = []
    for item in template.items:
        reasons = ", ".join(html.escape(reason.value) for reason in item.risk_reasons) or "none"
        issue_codes = ", ".join(code.value for code in item.decision.issue_codes) or "none"
        if isinstance(item, PaperWithoutCandidatesReviewItem):
            cards.append(
                "<article class='card absence'>"
                f"<h2>{html.escape(item.review_id)} · risk {item.risk_score}</h2>"
                f"<p><b>Paper:</b> {html.escape(item.paper_id)} · "
                "<b>Review item:</b> paper without candidates</p>"
                f"<p><b>Risk reasons:</b> {reasons}</p>"
                "<p><b>No candidate observations were produced for this paper.</b> "
                "Review whether this is a correct abstention or a missed extraction.</p>"
                "<h3>Decision template</h3>"
                f"<p>Outcome: <b>{html.escape(item.decision.outcome.value)}</b> · "
                f"issue codes: {html.escape(issue_codes)}</p>"
                "</article>"
            )
            continue
        candidate = item.candidate
        zero_eee = isinstance(item, PaperWithoutEEEReviewItem)
        output_absence = (
            "<p><b>This paper produced candidate observations but no EEE record.</b> "
            "Review whether this is a correct abstention, an unresolved export, or a missed "
            "composition.</p>"
            if zero_eee
            else ""
        )
        roles = (
            "".join(
                "<li>"
                f"{html.escape(role.role.value)}: {html.escape(role.raw_name)}; "
                f"version={html.escape(role.version or 'missing')}; "
                f"confidence={role.confidence:.3f}"
                "</li>"
                for role in candidate.roles
            )
            or "<li>no roles</li>"
        )
        evidence = "".join(
            "<section class='evidence'>"
            f"<b>page {anchor.page} · {html.escape(anchor.kind.value)}</b> "
            f"label={html.escape(anchor.label or 'missing')} · "
            f"row={html.escape(anchor.row or 'missing')} · "
            f"column={html.escape(anchor.column or 'missing')}"
            f"<blockquote>{html.escape(anchor.quote)}</blockquote>"
            "</section>"
            for anchor in candidate.evidence
        )
        scope_view = _json_view(
            candidate.scope.model_dump(mode="json") if candidate.scope else None
        )
        metric_view = _json_view(
            candidate.metric.model_dump(mode="json") if candidate.metric else None
        )
        value_view = _json_view(
            candidate.value.model_dump(mode="json") if candidate.value else None
        )
        cards.append(
            f"<article class='card{' absence' if zero_eee else ''}'>"
            f"<h2>{html.escape(item.review_id)} · risk {item.risk_score}</h2>"
            f"<p><b>Paper:</b> {html.escape(candidate.paper_id)} · "
            f"<b>Observation:</b> {html.escape(candidate.observation_id or 'missing')} · "
            f"<b>Claim:</b> {html.escape(candidate.claim_type.value)}</p>"
            f"<p><b>Review item:</b> {'paper without EEE output' if zero_eee else 'candidate'}</p>"
            f"{output_absence}"
            f"<p><b>Risk reasons:</b> {reasons}</p>"
            f"<p><b>Status:</b> text={html.escape(candidate.text_support.value)}, "
            f"reference={html.escape(candidate.referential_status.value)}, "
            f"export={html.escape(candidate.export_status.value)}, "
            f"confidence={candidate.extraction_confidence:.3f}</p>"
            f"<h3>Roles</h3><ul>{roles}</ul>"
            f"<h3>Scope</h3><pre>{scope_view}</pre>"
            f"<h3>Metric</h3><pre>{metric_view}</pre>"
            f"<h3>Value</h3><pre>{value_view}</pre>"
            f"<h3>Evidence</h3>{evidence}"
            "<h3>Decision template</h3>"
            f"<p>Outcome: <b>{html.escape(item.decision.outcome.value)}</b> · "
            f"issue codes: {html.escape(issue_codes)}</p>"
            "</article>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Local analyst review</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#f5f7fa;color:#17202a}}
.warning{{background:#fff3cd;border:1px solid #ffda6a;padding:1rem;border-radius:.5rem}}
.card{{background:white;border:1px solid #dfe4ea;border-radius:.6rem;padding:1.2rem;margin:1rem 0}}
.absence{{border-left:5px solid #c77d00}}
.evidence{{border-left:4px solid #6c7ae0;padding:.5rem 1rem;margin:.7rem 0}}
blockquote{{white-space:pre-wrap;background:#f6f7f9;padding:.7rem;margin:.5rem 0}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f7f9;padding:.7rem}}
</style></head><body>
<h1>Local analyst review</h1>
<div class="warning"><b>Local/private artifact; not independent human validation.</b>
This report contains source evidence quotes.
Do not publish it. Fill each <code>decision</code> object in
<code>{html.escape(decision_file_name)}</code>, then generate the aggregate public summary.</div>
<p>Audit {html.escape(template.audit_id)} · {len(template.items)} sampled from
{template.population_candidates} candidates across {template.population_papers} papers.</p>
{"".join(cards)}
</body></html>"""
