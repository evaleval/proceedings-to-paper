"""Score deterministic attribution against a completed run, without touching it.

The scorer reads a completed run read-only, recomputes attribution from the stored
candidates and page layouts, and reports what the strict origin gate would change. No
provider call is made and no stored run artifact is modified.

Three endpoints, declared before the first measurement:

1. ``false_primary_export`` on an annotated negative control must flip true to false.
2. Every candidate the scorer matched to an annotated reference observation must survive.
3. The demotion rate among exported candidates is reported so an abstain budget can be
   set from evidence rather than from taste.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from proceedings_to_eee.domain.attribution import AttributionState
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ClaimType, ExportStatus
from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.io import read_json, write_json
from proceedings_to_eee.resolution.attribution import (
    AttributionLexicon,
    attribute_candidates,
    load_lexicon,
)

ATTRIBUTION_SCORE_SCHEMA_VERSION = "attribution-score/0.1"

_EXPORTABLE = frozenset({ExportStatus.ELIGIBLE, ExportStatus.EXPORTED})


def _read_candidates(paper_dir: Path) -> list[CandidateObservation]:
    path = paper_dir / "observations.jsonl"
    if not path.is_file():
        return []
    return [
        CandidateObservation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_layout(paper_dir: Path) -> dict[str, PdfLayout]:
    path = paper_dir / "private" / "layout.json"
    if not path.is_file():
        return {}
    layout = PdfLayout.model_validate(read_json(path))
    return {layout.source_id: layout}


def _reference_ids(paper_dir: Path) -> tuple[set[str], set[str], set[str]]:
    """Annotated true positives, matched controls, and known false-primary exports."""

    path = paper_dir / "reference-score.json"
    if not path.is_file():
        return set(), set(), set()
    score = read_json(path)
    safety = score.get("negative_control_safety") or {}
    return (
        {match["observation_id"] for match in score.get("matches") or []},
        set(safety.get("matched_candidate_ids") or []),
        set(safety.get("false_primary_export_candidate_ids") or []),
    )


def score_paper_attribution(
    paper_dir: Path, lexicon: AttributionLexicon | None = None
) -> dict[str, Any]:
    """Recompute attribution for one completed paper run and report the deltas."""

    candidates = _read_candidates(paper_dir)
    layouts = _read_layout(paper_dir)
    if not candidates or not layouts:
        return {
            "paper_id": paper_dir.name,
            "candidates": len(candidates),
            "scored": False,
            "reason": "no candidates or no stored layout",
        }
    verdicts = attribute_candidates(candidates, layouts, lexicon or load_lexicon())
    matched, controls, false_primary_exports = _reference_ids(paper_dir)

    states = {state.value: 0 for state in AttributionState}
    exported_states = {state.value: 0 for state in AttributionState}
    rules: dict[str, int] = {}
    demoted_exports: list[dict[str, Any]] = []
    harmed_targets: list[dict[str, Any]] = []
    control_flips: list[dict[str, Any]] = []

    for candidate in candidates:
        key = candidate.observation_id or candidate.stable_id()
        verdict = verdicts[key]
        states[verdict.state.value] += 1
        rules[verdict.rule_id] = rules.get(verdict.rule_id, 0) + 1
        exportable = (
            candidate.claim_type == ClaimType.PRIMARY_RESULT
            and candidate.export_status in _EXPORTABLE
        )
        if exportable:
            exported_states[verdict.state.value] += 1
        if not verdict.demotes:
            continue
        record = {
            "observation_id": key,
            "state": verdict.state.value,
            "rule_id": verdict.rule_id,
            "row_label": verdict.row_label,
            "table_label": verdict.table_label,
            "page": verdict.page,
            "cues": [cue.model_dump(mode="json") for cue in verdict.cues],
            "was_exported": exportable,
        }
        if exportable:
            demoted_exports.append(record)
        if key in matched:
            harmed_targets.append(record)
        if key in false_primary_exports:
            control_flips.append(record)

    exported_total = sum(exported_states.values())
    return {
        "paper_id": paper_dir.name,
        "scored": True,
        "candidates": len(candidates),
        "exported_primary": exported_total,
        "states": states,
        "states_among_exported": exported_states,
        "rules": dict(sorted(rules.items())),
        "demoted_exports": demoted_exports,
        "harmed_reference_targets": harmed_targets,
        "false_primary_exports_caught": control_flips,
        "false_primary_exports_known": sorted(false_primary_exports),
        "matched_reference_targets": len(matched),
        "matched_controls": len(controls),
    }


def score_run_attribution(
    run_root: Path,
    output_path: Path | None = None,
    lexicon: AttributionLexicon | None = None,
) -> dict[str, Any]:
    """Score every paper of a completed run and aggregate the three endpoints."""

    lexicon = lexicon or load_lexicon()
    paper_dirs = sorted(
        path for path in run_root.iterdir() if path.is_dir() and (path / "run.json").is_file()
    )
    papers = [score_paper_attribution(path, lexicon) for path in paper_dirs]
    scored = [paper for paper in papers if paper.get("scored")]

    states = {state.value: 0 for state in AttributionState}
    exported_states = {state.value: 0 for state in AttributionState}
    for paper in scored:
        for key, count in paper["states"].items():
            states[key] += count
        for key, count in paper["states_among_exported"].items():
            exported_states[key] += count
    exported_total = sum(exported_states.values())
    # Canonical EEE requires positive paper-produced origin. ``NO_SIGNAL`` is an
    # abstention, not evidence that the current paper produced the result.
    demoted = exported_total - exported_states[AttributionState.PAPER_PRODUCED.value]
    known_false_primary = sum(len(paper["false_primary_exports_known"]) for paper in scored)
    caught = sum(len(paper["false_primary_exports_caught"]) for paper in scored)
    harmed = sum(len(paper["harmed_reference_targets"]) for paper in scored)
    matched = sum(paper["matched_reference_targets"] for paper in scored)

    summary = {
        "schema_version": ATTRIBUTION_SCORE_SCHEMA_VERSION,
        "run_root": run_root.name,
        "lexicon_id": lexicon.lexicon_id,
        "lexicon_sha256": lexicon.sha256,
        "papers_scored": len(scored),
        "candidates": sum(paper["candidates"] for paper in scored),
        "states": states,
        "exported_primary": exported_total,
        "states_among_exported": exported_states,
        "endpoints": {
            "false_primary_exports_known": known_false_primary,
            "false_primary_exports_caught": caught,
            "reference_targets_matched": matched,
            "reference_targets_demoted": harmed,
            "demoted_exports": demoted,
            "demotion_rate_among_exports": (demoted / exported_total if exported_total else None),
            "externally_sourced_among_exports": exported_states[
                AttributionState.EXTERNALLY_SOURCED.value
            ],
            "unresolved_among_exports": exported_states[AttributionState.UNRESOLVED.value],
        },
        "papers": papers,
    }
    if output_path is not None:
        write_json(output_path, summary)
    return summary


def render_attribution_summary(summary: dict[str, Any]) -> str:
    """Compact human-readable form for the command line."""

    endpoints = summary["endpoints"]
    lines = [
        f"run                        {summary['run_root']}",
        f"lexicon                    {summary['lexicon_id']} {summary['lexicon_sha256'][:16]}",
        f"papers scored              {summary['papers_scored']}",
        f"candidates                 {summary['candidates']}",
        f"exported primary results   {summary['exported_primary']}",
        "",
        "attribution over exported primary results:",
    ]
    for state, count in summary["states_among_exported"].items():
        lines.append(f"   {state:20s} {count}")
    lines += [
        "",
        f"known false-primary exports      {endpoints['false_primary_exports_known']}",
        f"   caught                        {endpoints['false_primary_exports_caught']}",
        f"annotated targets matched        {endpoints['reference_targets_matched']}",
        f"   demoted (must be 0)           {endpoints['reference_targets_demoted']}",
        f"demotion rate among exports      {endpoints['demotion_rate_among_exports']}",
    ]
    return "\n".join(lines)


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
