"""Versioned extraction prompt: propose candidates, never final EEE."""

from __future__ import annotations

import hashlib
import json

from proceedings_to_eee.extraction.pdf_layout import PageFragment
from proceedings_to_eee.extraction.row_enumeration import RowBatch

SYSTEM_PROMPT = """You extract evaluation-result candidates from scientific papers.
Your output is an evidence-bound proposal, not final Every Eval Ever data.

Definitions:
- primary_result: a result produced by the current paper's own experiment or analysis.
- secondary_claim: a result attributed to another paper, prior study, vendor, or external source.
- illustration: an example, hypothetical score, schematic value, or explanatory mock-up.
- method_metadata: versions, thresholds, sample sizes, requirements, setup, or definitions rather than a performance result.
- uncertain: evidence does not safely distinguish the above.

Keep roles separate:
- evaluated_system: the model/API/system whose performance or behavior is measured.
- evaluation_instrument: a scorer, judge, classifier, or API used to measure another system.
- label_generator: a system that generated reference labels.
- human_reference: human annotations or experts used as reference.

Rules:
1. Extract only claims visibly supported on the supplied page. Do not use outside knowledge.
2. Copy evidence quotes verbatim from the page text. The quote must contain the raw value and enough row or sentence context to identify the system, dataset, and metric.
3. A single observation contains one value, one metric, one exact scope, and exactly one evaluated system for primary results. Split table rows into atomic cells.
4. Keep printed units and scales. If the paper prints 73.4%, numeric is 73.4 and unit is percent; do not convert to 0.734. Preserve inequality or approximation semantics in comparator; for <0.001, numeric is 0.001 and comparator is less_than.
5. A confidence interval belongs to its point estimate, not a second observation.
6. Do not turn table headings, dataset sizes, years, citations, equation constants, thresholds, significance markers, or illustrative examples into primary results.
7. If prose repeats a table result, represent it once with multiple evidence anchors only when both anchors are present on this page. Do not duplicate it.
8. If a role, version, scope, unit, direction, or canonical identifier is not explicit, return null rather than guessing.
9. For metric IDs, use a conventional ID only when unambiguous: accuracy, auroc, f1, precision, recall, fpr, fnr, toxicity_score. Otherwise null.
10. Prefer precise abstention over plausible completion. Preserve conflicts in notes.
"""

ROW_ENUMERATION_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

Dense-table row enumeration contract:
- The user supplies stable input row IDs from a deterministic layout index.
- Return exactly one disposition for every supplied row ID and never invent a row ID.
- result means the row contains one or more independently reportable evaluation results;
  emit one atomic observation per physical result cell.
- not_result means the row is descriptive, configurational, metadata, a header-like row,
  or otherwise contains no reportable evaluation result; emit no observations.
- uncertain means the supplied row and its table context are insufficient to decide safely;
  emit no observations so the row reaches review instead of becoming a guessed candidate.
- Every observation quote must be copied from that row's raw_text, and every value must
  occur in that same raw_text. Use the supplied caption and all header levels only as
  context for interpreting columns, metrics, and scope.
"""
)

_ROW_PROMPT_TEMPLATE_VERSION = "table-row-prompt/0.1"


def page_prompt(*, paper_title: str, paper_id: str, fragment: PageFragment) -> str:
    return f"""CURRENT PAPER
paper_id: {paper_id}
title: {paper_title}

SOURCE PAGE FRAGMENT
source_id: {fragment.source_id}
page: {fragment.page}
fragment_id: {fragment.fragment_id}

Extract every independently reportable evaluation observation in this bounded source fragment. A fragment can contain repeated table/header context from the same page; do not emit that context as a result. Fragments can contain no observations; return an empty list when appropriate. Preserve layout spaces in evidence quotes.

<PAGE_TEXT>
{fragment.text}
</PAGE_TEXT>
"""


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def row_batch_prompt(*, paper_title: str, paper_id: str, batch: RowBatch) -> str:
    payload = {
        "schema_version": _ROW_PROMPT_TEMPLATE_VERSION,
        "source_id": batch.source_id,
        "page": batch.page,
        "table_region_id": batch.region_id,
        "table_label": batch.table_label,
        "caption": batch.caption,
        "caption_coordinates": (
            batch.caption_span.model_dump(mode="json") if batch.caption_span else None
        ),
        "headers": [header.model_dump(mode="json") for header in batch.headers],
        "rows": [
            {
                "row_id": row.row_id,
                "row_label": row.row_label,
                "raw_text": row.raw_text,
                "raw_cells": [cell.model_dump(mode="json") for cell in row.raw_cells],
                "value_positions": [value.model_dump(mode="json") for value in row.values],
                "evidence_coordinates": row.span.model_dump(mode="json"),
            }
            for row in batch.rows
        ],
    }
    return f"""CURRENT PAPER
paper_id: {paper_id}
title: {paper_title}

Return exactly one result, not_result, or uncertain disposition for every row_id in
INPUT_TABLE_ROWS. Preserve row IDs exactly. Valid not_result and uncertain rows have an
empty observations list. A result row has at least one atomic observation.

<INPUT_TABLE_ROWS>
{json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)}
</INPUT_TABLE_ROWS>
"""


def row_prompt_hash() -> str:
    payload = ROW_ENUMERATION_SYSTEM_PROMPT + "\0" + _ROW_PROMPT_TEMPLATE_VERSION
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
