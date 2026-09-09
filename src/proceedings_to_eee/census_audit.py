"""AI analyst QA over extracted candidates, judged against the whole source page.

This is a diagnostic, not validation. It says where to look next; it never establishes
precision, recall, or correctness, and every artifact it writes says so.

The auditor receives the entire frozen page, rather than only the bounded block used
for extraction. It returns a verdict per field so the diagnostic can distinguish
value, system, dataset, and metric errors using the available page context.

Two local defences apply to everything it returns. Every supporting quotation must occur
on the page the auditor was given, and a verdict whose quotation does not is dropped
rather than counted, so the auditor cannot assert from nothing. And no census metadata,
reference annotation, gate outcome, or sibling candidate is ever sent, so it cannot infer
the answer from the frame.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from proceedings_to_eee.io import write_json

AUDIT_SCHEMA_VERSION = "census-candidate-audit/0.1"
AUDIT_SCHEMA_NAME = "candidate_page_audit"

#: Fields the auditor judges. Each is answerable from a single page of a paper.
AUDITED_FIELDS = (
    "value",
    "system_identity",
    "system_role",
    "dataset",
    "scope",
    "metric",
    "unit",
    "producer_origin",
)
VERDICTS = ("correct", "incorrect", "unsupported", "cannot_tell")

SYSTEM_PROMPT = """You audit one extracted evaluation result against the full page it came from.

You are given the complete text of one page of a scientific paper, and one extracted
record claiming that this page reports a particular value for a particular system,
dataset, metric and scope.

Judge each field independently:
- correct: the page supports the extracted field as stated.
- incorrect: the page contradicts it, for example the value belongs to a different row,
  system, dataset or metric than the one claimed.
- unsupported: the page does not contain enough to support the field either way.
- cannot_tell: the page is ambiguous, or the field is genuinely undecidable from this page.

Rules:
1. Judge only from the supplied page. Do not use outside knowledge about the systems,
   datasets or papers involved.
2. For every field you do not mark correct, quote the exact span of the supplied page
   that justifies your verdict. Copy it verbatim from the page. Never invent a quotation.
3. producer_origin asks whether this page shows the current paper produced this number,
   as opposed to quoting it from another work. Absence of evidence is unsupported or
   cannot_tell, never incorrect.
4. Prefer cannot_tell over guessing. An honest abstention is more useful than a
   confident error.
"""


def audit_response_schema() -> dict[str, Any]:
    """Strict wire schema. Every audited field is present and typed."""

    field_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdict", "reason", "page_quote"],
        "properties": {
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "reason": {"type": ["string", "null"]},
            "page_quote": {"type": ["string", "null"]},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(AUDITED_FIELDS),
        "properties": {field: field_schema for field in AUDITED_FIELDS},
    }


@dataclass(frozen=True)
class AuditItem:
    """One sampled candidate, with the page it will be judged against."""

    paper_id: str
    observation_id: str
    page: int
    page_text: str
    raw_value: str
    evidence_quote: str
    system: str | None
    dataset: str | None
    metric: str | None
    unit: str | None
    claim_type: str | None
    stratum: tuple[str, str, str]


def _normalize(text: str) -> str:
    """Collapse whitespace only. Nothing else is repaired, so a match stays exact."""

    return re.sub(r"\s+", " ", text).strip()


def quotation_is_on_page(quote: str | None, page_text: str) -> bool:
    """A verdict may only stand on a quotation that really occurs on the page shown."""

    if not quote or not quote.strip():
        return False
    return _normalize(quote) in _normalize(page_text)


def _stratum_key(metadata: dict[str, Any] | None) -> tuple[str, str, str]:
    if not metadata:
        return ("unknown", "unknown", "unknown")
    return (
        str(metadata.get("census_year", "unknown")),
        str(metadata.get("census_venue", "unknown")),
        str(metadata.get("census_measurement_role", "unknown")),
    )


def _selection_key(seed: str, observation_id: str) -> str:
    return hashlib.sha256(f"{seed}|{observation_id}".encode()).hexdigest()


def stratified_sample(items: Sequence[AuditItem], *, seed: str, size: int) -> list[AuditItem]:
    """Spread the sample across strata so no year, venue or role dominates it."""

    if size >= len(items):
        return sorted(items, key=lambda item: item.observation_id)
    groups: dict[tuple[str, str, str], list[AuditItem]] = defaultdict(list)
    for item in items:
        groups[item.stratum].append(item)
    ranked: list[tuple[float, str, AuditItem]] = []
    for members in groups.values():
        ordered = sorted(members, key=lambda item: _selection_key(seed, item.observation_id))
        for index, item in enumerate(ordered):
            ranked.append(
                ((index + 0.5) / len(ordered), _selection_key(seed, item.observation_id), item)
            )
    ranked.sort(key=lambda entry: (entry[0], entry[1]))
    return [item for _, _, item in ranked[:size]]


def collect_audit_items(
    run_root: Path,
    *,
    paper_ids: Iterable[str] | None = None,
    census_metadata: dict[str, Any] | None = None,
) -> list[AuditItem]:
    """Gather supported candidates and pair each with its own evidence page."""

    metadata = (census_metadata or {}).get("by_slug", {})
    wanted = set(paper_ids) if paper_ids is not None else None
    items: list[AuditItem] = []
    for observations in sorted(run_root.glob("*/observations.jsonl")):
        paper_id = observations.parent.name
        if wanted is not None and paper_id not in wanted:
            continue
        layout_path = observations.parent / "private" / "layout.json"
        if not layout_path.is_file():
            continue
        try:
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pages = {int(page["page"]): page.get("text", "") for page in layout.get("pages", [])}
        for line in observations.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("text_support") != "supported":
                continue
            evidence = (record.get("evidence") or [{}])[0]
            page = evidence.get("page")
            if page is None or int(page) not in pages:
                continue
            value = record.get("value") or {}
            roles = record.get("roles") or []
            scope = record.get("scope") or {}
            metric = record.get("metric") or {}
            items.append(
                AuditItem(
                    paper_id=paper_id,
                    observation_id=record["observation_id"],
                    page=int(page),
                    page_text=pages[int(page)],
                    raw_value=str(value.get("raw", "")),
                    evidence_quote=str(evidence.get("quote", "")),
                    system=next(
                        (role.get("raw_name") for role in roles if role.get("raw_name")), None
                    ),
                    dataset=scope.get("dataset_raw"),
                    metric=metric.get("raw_name"),
                    unit=value.get("unit"),
                    claim_type=record.get("claim_type"),
                    stratum=_stratum_key(metadata.get(paper_id)),
                )
            )
    return items


def audit_prompt(item: AuditItem) -> str:
    """Build the user message. Carries the page and the claim, and nothing else."""

    # `claim_type` is deliberately absent. It is the extractor's own label for the
    # record, and telling an auditor what a previous model concluded invites agreement
    # rather than judgement.
    claim = {
        "reported_value": item.raw_value,
        "evidence_quote": item.evidence_quote,
        "claimed_system": item.system,
        "claimed_dataset": item.dataset,
        "claimed_metric": item.metric,
        "claimed_unit": item.unit,
    }
    return f"""PAGE {item.page} OF THE PAPER

<PAGE_TEXT>
{item.page_text}
</PAGE_TEXT>

EXTRACTED RECORD TO AUDIT

{json.dumps(claim, indent=2, ensure_ascii=False)}

Judge every field against the page above and quote the page for each field you do not
mark correct."""


def _verdict_rows(item: AuditItem, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Keep verdicts whose quotation is really on the page; count the ones dropped."""

    rows: list[dict[str, Any]] = []
    dropped = 0
    for field in AUDITED_FIELDS:
        answer = payload.get(field) or {}
        verdict = answer.get("verdict")
        if verdict not in VERDICTS:
            dropped += 1
            continue
        quote = answer.get("page_quote")
        if verdict != "correct" and not quotation_is_on_page(quote, item.page_text):
            # The auditor asserted a defect it could not point to on the page.
            dropped += 1
            continue
        rows.append(
            {
                "paper_id": item.paper_id,
                "observation_id": item.observation_id,
                "field": field,
                "verdict": verdict,
                "reason": answer.get("reason"),
                "stratum_year": item.stratum[0],
                "stratum_venue": item.stratum[1],
                "stratum_role": item.stratum[2],
            }
        )
    return rows, dropped


def summarize(rows: Sequence[dict[str, Any]], *, audited: int, dropped: int) -> dict[str, Any]:
    """Per-field rates with explicit numerators, plus the instrument's own abstention rate."""

    per_field: dict[str, Any] = {}
    for field in AUDITED_FIELDS:
        field_rows = [row for row in rows if row["field"] == field]
        counts = Counter(row["verdict"] for row in field_rows)
        decided = counts["correct"] + counts["incorrect"] + counts["unsupported"]
        per_field[field] = {
            "counts": dict(counts),
            "denominator_all": len(field_rows),
            "denominator_decided": decided,
            "correct_rate_over_decided": (
                round(counts["correct"] / decided, 6) if decided else None
            ),
            "cannot_tell_rate": (
                round(counts["cannot_tell"] / len(field_rows), 6) if field_rows else None
            ),
        }
    by_role: dict[str, Any] = {}
    for role in sorted({row["stratum_role"] for row in rows}):
        subset = [row for row in rows if row["stratum_role"] == role and row["field"] == "value"]
        decided = [row for row in subset if row["verdict"] != "cannot_tell"]
        by_role[role] = {
            "value_denominator_decided": len(decided),
            "value_correct_rate": (
                round(sum(row["verdict"] == "correct" for row in decided) / len(decided), 6)
                if decided
                else None
            ),
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "review_type": "ai_analyst_qa",
        "independent_human_validation": False,
        "claim_boundary": (
            "AI analyst QA on a sample, judged by a model that can itself be wrong. Not "
            "precision, recall, accuracy, or independent validation, and not a substitute "
            "for human annotation."
        ),
        "candidates_audited": audited,
        "verdicts_kept": len(rows),
        "verdicts_dropped_unquotable": dropped,
        "per_field": per_field,
        "value_field_by_measurement_role": by_role,
        "frequent_findings": [
            {"field": field, "verdict": verdict, "count": count}
            for (field, verdict), count in Counter(
                (row["field"], row["verdict"]) for row in rows if row["verdict"] != "correct"
            ).most_common(10)
        ],
    }


def run_audit(
    *,
    run_root: Path,
    output: Path,
    model: str,
    sample: int,
    seed: str,
    call: Callable[[str, str], dict[str, Any]],
    paper_ids: Iterable[str] | None = None,
    census_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample candidates, audit each against its page, and write a quote-free summary."""

    items = collect_audit_items(run_root, paper_ids=paper_ids, census_metadata=census_metadata)
    chosen = stratified_sample(items, seed=seed, size=sample)
    rows: list[dict[str, Any]] = []
    dropped = 0
    failures = 0
    for item in chosen:
        try:
            payload = call(SYSTEM_PROMPT, audit_prompt(item))
        except Exception:  # noqa: BLE001 - one bad audit must not end the audit
            failures += 1
            continue
        kept, lost = _verdict_rows(item, payload)
        rows.extend(kept)
        dropped += lost
    summary = summarize(rows, audited=len(chosen) - failures, dropped=dropped)
    summary["model"] = model
    summary["seed"] = seed
    summary["sampling_frame"] = {
        "supported_candidates_available": len(items),
        "requested_sample": sample,
        "sampled": len(chosen),
        "audit_calls_failed": failures,
        "strata": "census year, venue and measurement role; metadata is never sent to the provider",
    }
    write_json(output, summary)
    return summary
