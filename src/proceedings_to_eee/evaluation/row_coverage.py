"""How much of a table the extractor actually enumerates.

A recurring failure mode produces per-cell observations for an early row of a multi-row
table but never reaches later rows. Development diagnostics showed that this is a general
enumeration problem rather than an isolated layout accident, and a major recall gap in
the pipeline.

The denominator is deliberately narrow: only table data rows that lay inside the body of a
block actually sent to the extractor. Rows on unselected pages and rows in bibliographies
are excluded, so the number reflects enumeration rather than page selection.

**This is not a target of 1.0.** Papers carry descriptive and qualitative tables whose rows
should never become results, and nothing here can tell those apart from a genuine miss. A
low coverage figure is a signal to look, not a defect count. What makes it credible as a
signal is the annotated cases inside it: at least one missed row is an annotated target.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.extraction.region_index import RegionKind, build_region_index
from proceedings_to_eee.extraction.result_blocks import ResultBlock
from proceedings_to_eee.extraction.row_enumeration import (
    RowDisposition,
    RowDispositionRecord,
    RowEnumerationPlan,
    build_row_enumeration_plan,
)
from proceedings_to_eee.io import read_json, sha256_file, write_json
from proceedings_to_eee.validation.candidates import normalize_evidence_text

ROW_COVERAGE_SCHEMA_VERSION = "row-coverage/0.2"

_MIN_QUOTE_OVERLAP_CHARS = 8


def _extracted_lines(blocks: list[ResultBlock]) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = defaultdict(set)
    for block in blocks:
        seen[block.page].update(range(block.body_start_line, block.body_end_line + 1))
    return seen


def _quotes_by_page(candidates: list[CandidateObservation]) -> dict[int, list[str]]:
    quotes: dict[int, list[str]] = defaultdict(list)
    for candidate in candidates:
        for anchor in candidate.evidence:
            normalized = normalize_evidence_text(anchor.quote)
            if len(normalized) >= _MIN_QUOTE_OVERLAP_CHARS:
                quotes[anchor.page].append(normalized)
    return quotes


def _validated_row_ledger(
    *,
    paper_dir: Path,
    layout: PdfLayout,
    blocks: list[ResultBlock],
) -> dict[str, Any] | None:
    """Validate an optional plan/ledger pair and summarize its dispositions.

    A stale, partial, or malformed pair raises instead of silently falling back to
    quote overlap.  Quote overlap remains a separate diagnostic below; it is not used
    to manufacture row dispositions.
    """

    private = paper_dir / "private"
    plan_path = private / "row-enumeration-plan.json"
    ledger_path = private / "row-enumeration.json"
    if not plan_path.exists() and not ledger_path.exists():
        return None
    if not plan_path.is_file() or not ledger_path.is_file():
        missing = plan_path.name if not plan_path.is_file() else ledger_path.name
        raise ValueError(f"{paper_dir.name}: incomplete row ledger pair; missing {missing}")

    try:
        plan = RowEnumerationPlan.model_validate(read_json(plan_path))
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError(f"{paper_dir.name}: malformed row enumeration plan") from exc

    rebuilt = build_row_enumeration_plan(layout, blocks, config=plan.config)
    if plan.model_dump(mode="json") != rebuilt.model_dump(mode="json"):
        raise ValueError(f"{paper_dir.name}: row enumeration plan does not match layout/blocks")

    planned_ids = [row.row_id for row in plan.rows]
    if len(set(planned_ids)) != len(planned_ids):
        raise ValueError(f"{paper_dir.name}: duplicate row ids in row enumeration plan")
    planned = set(planned_ids)
    unbatchable_ids = [row.row_id for row in plan.unbatchable_rows]
    unbatchable = set(unbatchable_ids)
    batch_ids = [row.row_id for batch in plan.batches for row in batch.rows]
    if len(unbatchable) != len(unbatchable_ids) or len(set(batch_ids)) != len(batch_ids):
        raise ValueError(f"{paper_dir.name}: duplicate row assignment in row enumeration plan")
    if set(batch_ids) != planned - unbatchable:
        raise ValueError(f"{paper_dir.name}: row enumeration batches do not partition planned rows")

    try:
        raw = read_json(ledger_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{paper_dir.name}: malformed row enumeration ledger") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"{paper_dir.name}: row enumeration ledger must be an object")
    ledger_schema = raw.get("schema_version")
    if ledger_schema is not None and ledger_schema != "row-enumeration-outcome/0.1":
        raise ValueError(f"{paper_dir.name}: unsupported row enumeration ledger schema")
    ledger_plan_sha256 = raw.get("plan_sha256")
    if ledger_plan_sha256 is not None and ledger_plan_sha256 != sha256_file(plan_path):
        raise ValueError(f"{paper_dir.name}: row enumeration ledger plan hash mismatch")
    raw_records = raw.get("records")
    raw_unresolved = raw.get("unresolved_row_ids")
    raw_unbatchable = raw.get("unbatchable_row_ids")
    telemetry = raw.get("telemetry")
    if not isinstance(raw_records, Mapping):
        raise ValueError(f"{paper_dir.name}: row enumeration records must be keyed by row_id")
    if not isinstance(raw_unresolved, list) or not all(
        isinstance(item, str) for item in raw_unresolved
    ):
        raise ValueError(f"{paper_dir.name}: unresolved_row_ids must be a list of strings")
    if raw_unbatchable is not None and (
        not isinstance(raw_unbatchable, list)
        or not all(isinstance(item, str) for item in raw_unbatchable)
    ):
        raise ValueError(f"{paper_dir.name}: unbatchable_row_ids must be a list of strings")
    if not isinstance(telemetry, Mapping):
        raise ValueError(f"{paper_dir.name}: row enumeration telemetry must be an object")

    records: dict[str, RowDispositionRecord] = {}
    for key, payload in raw_records.items():
        if not isinstance(key, str):
            raise ValueError(f"{paper_dir.name}: row enumeration record key must be a string")
        try:
            record = RowDispositionRecord.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            raise ValueError(f"{paper_dir.name}: malformed disposition record for {key}") from exc
        if record.row_id != key:
            raise ValueError(f"{paper_dir.name}: disposition record key/row_id mismatch for {key}")
        records[key] = record

    record_ids = set(records)
    unresolved_ids = set(raw_unresolved)
    if len(unresolved_ids) != len(raw_unresolved):
        raise ValueError(f"{paper_dir.name}: duplicate unresolved row ids")
    if raw_unbatchable is not None and (
        len(set(raw_unbatchable)) != len(raw_unbatchable) or set(raw_unbatchable) != unbatchable
    ):
        raise ValueError(f"{paper_dir.name}: row ledger unbatchable ids mismatch plan")
    unknown = (record_ids | unresolved_ids | unbatchable) - planned
    if unknown:
        raise ValueError(f"{paper_dir.name}: row ledger contains unknown row ids")
    if record_ids & (unresolved_ids | unbatchable) or unresolved_ids & unbatchable:
        raise ValueError(f"{paper_dir.name}: row ledger assigns a row more than once")
    if record_ids | unresolved_ids | unbatchable != planned:
        raise ValueError(f"{paper_dir.name}: row ledger does not account for every planned row")

    dispositions = Counter(record.disposition.value for record in records.values())
    expected_telemetry: dict[str, object] = {
        "rows_resolved": len(records),
        "rows_unresolved": len(unresolved_ids),
        "rows_unbatchable": len(unbatchable),
        "dispositions": {item.value: dispositions[item.value] for item in RowDisposition},
    }
    for key, expected in expected_telemetry.items():
        if telemetry.get(key) != expected:
            raise ValueError(f"{paper_dir.name}: row ledger telemetry mismatch for {key}")

    resolved = len(records)
    unresolved = len(unresolved_ids) + len(unbatchable)
    return {
        "ledger_status": "validated",
        "rows_planned": len(planned),
        "rows_resolved": resolved,
        "resolution_coverage": resolved / len(planned) if planned else None,
        "result_rows": dispositions[RowDisposition.RESULT.value],
        "not_result_rows": dispositions[RowDisposition.NOT_RESULT.value],
        "uncertain_rows": dispositions[RowDisposition.UNCERTAIN.value],
        "unresolved_rows": unresolved,
        "unbatchable_rows": len(unbatchable),
        "basis": (
            "Resolved/planned uses the validated row-disposition ledger. result rows carry "
            "one or more candidates; not_result and uncertain are resolved abstentions and "
            "are never counted as candidate-bearing rows. Unbatchable rows are unresolved."
        ),
    }


def score_paper_row_coverage(paper_dir: Path) -> dict[str, Any]:
    """Enumeration coverage for one completed paper run. Offline and read-only."""

    observations = paper_dir / "observations.jsonl"
    layout_path = paper_dir / "private" / "layout.json"
    blocks_path = paper_dir / "private" / "result-blocks.json"
    if not (layout_path.is_file() and blocks_path.is_file()):
        return {"paper_id": paper_dir.name, "scored": False}

    candidates = (
        [
            CandidateObservation.model_validate_json(line)
            for line in observations.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if observations.is_file()
        else []
    )
    layout = PdfLayout.model_validate(read_json(layout_path))
    blocks = [ResultBlock.model_validate(item) for item in read_json(blocks_path)]
    disposition_coverage = _validated_row_ledger(
        paper_dir=paper_dir,
        layout=layout,
        blocks=blocks,
    )

    seen = _extracted_lines(blocks)
    quotes = _quotes_by_page(candidates)
    indexes = build_region_index(layout)

    tables = 0
    rows = 0
    covered = 0
    empty_tables: list[dict[str, Any]] = []
    for page, page_index in indexes.items():
        for region in page_index.regions:
            if region.kind is not RegionKind.TABLE or region.in_references:
                continue
            data_rows = [
                row
                for row in region.rows
                if not row.is_header and row.line in seen.get(page, set())
            ]
            if not data_rows:
                continue
            tables += 1
            hits = 0
            for row in data_rows:
                line = normalize_evidence_text(" ".join(row.cells))
                if any(line in quote or quote in line for quote in quotes.get(page, ())):
                    hits += 1
            rows += len(data_rows)
            covered += hits
            if hits == 0:
                empty_tables.append(
                    {
                        "page": page,
                        "table_label": region.table_label,
                        "rows_shown": len(data_rows),
                    }
                )
    result = {
        "paper_id": paper_dir.name,
        "scored": True,
        "candidates": len(candidates),
        "table_anchors": sum(
            anchor.kind.value == "table" for item in candidates for anchor in item.evidence
        ),
        "tables_shown": tables,
        "rows_shown": rows,
        "rows_with_a_candidate": covered,
        "row_coverage": covered / rows if rows else None,
        "tables_with_no_candidate": empty_tables,
    }
    if disposition_coverage is not None:
        result["row_disposition_coverage"] = disposition_coverage
    return result


def score_run_row_coverage(run_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Aggregate enumeration coverage over a completed run."""

    papers = [
        score_paper_row_coverage(path)
        for path in sorted(run_root.iterdir())
        if path.is_dir() and (path / "run.json").is_file()
    ]
    scored = [paper for paper in papers if paper.get("scored")]
    rows = sum(paper["rows_shown"] for paper in scored)
    covered = sum(paper["rows_with_a_candidate"] for paper in scored)
    summary = {
        "schema_version": ROW_COVERAGE_SCHEMA_VERSION,
        "run_root": run_root.name,
        "papers_scored": len(scored),
        "tables_shown": sum(paper["tables_shown"] for paper in scored),
        "rows_shown": rows,
        "rows_with_a_candidate": covered,
        "row_coverage": covered / rows if rows else None,
        "papers_with_zero_table_anchors": [
            paper["paper_id"] for paper in scored if paper["table_anchors"] == 0
        ],
        "basis": (
            "Denominator is table data rows inside the body of an extracted block, "
            "excluding bibliographies. Not a target of 1.0: descriptive and qualitative "
            "table rows should never become results and cannot be told apart here."
        ),
        "papers": papers,
    }
    ledgers = [
        paper["row_disposition_coverage"] for paper in scored if "row_disposition_coverage" in paper
    ]
    if ledgers:
        planned = sum(item["rows_planned"] for item in ledgers)
        resolved = sum(item["rows_resolved"] for item in ledgers)
        summary["row_disposition_coverage"] = {
            "papers_with_ledger": len(ledgers),
            "papers_scored": len(scored),
            "complete_for_scored_papers": len(ledgers) == len(scored),
            "papers_without_ledger": [
                paper["paper_id"] for paper in scored if "row_disposition_coverage" not in paper
            ],
            "rows_planned": planned,
            "rows_resolved": resolved,
            "resolution_coverage": resolved / planned if planned else None,
            "result_rows": sum(item["result_rows"] for item in ledgers),
            "not_result_rows": sum(item["not_result_rows"] for item in ledgers),
            "uncertain_rows": sum(item["uncertain_rows"] for item in ledgers),
            "unresolved_rows": sum(item["unresolved_rows"] for item in ledgers),
            "unbatchable_rows": sum(item["unbatchable_rows"] for item in ledgers),
            "basis": (
                "Disposition counts cover only papers with a validated plan/ledger pair; "
                "candidate-bearing quote coverage remains the separate historical diagnostic."
            ),
        }
    if output_path is not None:
        write_json(output_path, summary)
    return summary
