"""Replay validation and EEE composition over a finished census run, offline.

Nothing here calls a provider and nothing here rewrites `observations.jsonl`. The run's
candidates, its frozen page layout and its source manifest are enough to replay every
deterministic gate, so a change to the export policy can be applied to work already paid
for. The replay under the default `positive_only` policy is a no-op by construction: it
must reproduce the run's own export reasons exactly, which is what the phase gate checks.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proceedings_to_eee.census_review import candidate_payload_sha256, load_gates
from proceedings_to_eee.composition.eee import compose_eee_records
from proceedings_to_eee.domain.attribution import OriginBasis, OriginExportPolicy, ReviewTier
from proceedings_to_eee.domain.export_provenance import legacy_export_provenance
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ExportStatus
from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.io import write_json
from proceedings_to_eee.resources import DEFAULT_EEE_SCHEMA_PATH
from proceedings_to_eee.sources.manifest import SourceManifest
from proceedings_to_eee.validation.candidates import (
    _validate_non_origin_candidates_in_place,
    candidate_origin_basis,
    route_candidate_attribution,
)
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record

RECOMPOSE_SCHEMA_VERSION = "census-recompose/0.1"
RECORD_INDEX_SCHEMA_VERSION = "census-eee-record-index/0.1"

#: Directory written under each paper. Deliberately not `eee/`: the run's own canonical
#: directory is the pipeline's output and this replay must never overwrite it.
TIERED_RECORD_DIRNAME = "eee-tiered"


@dataclass
class PaperRecomposition:
    """What one paper produced on the replay, and how it differs from its own run."""

    paper_id: str
    candidates: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    export_status_counts: Counter[str] = field(default_factory=Counter)
    export_reason_counts: Counter[str] = field(default_factory=Counter)
    reason_transitions: Counter[str] = field(default_factory=Counter)
    basis_counts: Counter[str] = field(default_factory=Counter)
    tier_counts: Counter[str] = field(default_factory=Counter)
    invalid_records: list[dict[str, Any]] = field(default_factory=list)
    #: Candidates whose observation id changed because the replay resolved their tuple
    #: further than the original run did. The id is derived from the tuple, so this is
    #: expected whenever the registry learns a name; it is counted, never hidden.
    reidentified: int = 0
    #: Review gates whose candidate payload hash no longer matches the candidate. The
    #: page said something about a tuple that has since changed, so the gate is ignored.
    stale_review_gates: int = 0
    #: Candidates left out because the registry knows no scale for their metric, so no
    #: schema-valid record can be written without inventing one.
    skipped_unbounded_metric: int = 0
    skipped_reason: str | None = None


def load_paper_candidates(paper_dir: Path) -> list[CandidateObservation]:
    """Read one paper's candidates exactly as the run wrote them."""

    path = paper_dir / "observations.jsonl"
    if not path.is_file():
        return []
    candidates: list[CandidateObservation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            candidates.append(CandidateObservation.model_validate(json.loads(line)))
    return candidates


def _paper_layouts(paper_dir: Path, manifest: SourceManifest) -> dict[str, PdfLayout]:
    """Bind the frozen layout to the source it was extracted from.

    A census paper freezes exactly one source, and `private/layout.json` is that source's
    layout. Anything else is refused rather than guessed at, because binding a layout to
    the wrong source would silently invalidate every quote check below.
    """

    layout_path = paper_dir / "private" / "layout.json"
    if not layout_path.is_file():
        return {}
    layout = PdfLayout.model_validate(json.loads(layout_path.read_text(encoding="utf-8")))
    if len(manifest.sources) != 1:
        return {}
    return {manifest.sources[0].source_id: layout}


def _promote_reviewed_candidates(
    *,
    original: list[CandidateObservation],
    validated: list[CandidateObservation],
    paper_dir: Path,
    already: set[str],
) -> tuple[list[tuple[CandidateObservation, OriginBasis]], int]:
    """Admit candidates a page-scoped reviewer accepted, on the basis it earned.

    Gates are keyed by the observation id the run wrote, and validation re-stamps that id
    from the resolved tuple, so the lookup pairs by position and the payload hash decides
    whether the gate still describes this candidate. A referential or field-provenance
    failure does not block this tier; it travels into the record instead, which is the
    whole point of naming the tier.
    """

    gates = load_gates(paper_dir)
    if not gates:
        return [], 0
    promoted: list[tuple[CandidateObservation, OriginBasis]] = []
    stale = 0
    for was, candidate in zip(original, validated, strict=True):
        gate = gates.get(str(was.observation_id))
        if not gate or gate.get("decision") not in {"accept", "accept_tuple_only"}:
            continue
        if gate.get("candidate_payload_sha256") != candidate_payload_sha256(
            was.model_dump(mode="json")
        ):
            stale += 1
            continue
        observation_id = candidate.observation_id or candidate.stable_id()
        if observation_id in already:
            continue
        if candidate.metric is None or not candidate.metric.canonical_id:
            continue
        if candidate.value is None or candidate.scope is None:
            continue
        if candidate.attribution is None:
            continue
        basis = (
            OriginBasis.MODEL_REVIEWED_ORIGIN_QUOTE
            if gate["decision"] == "accept"
            else candidate_origin_basis(candidate)
        )
        if basis is OriginBasis.NONE:
            continue
        candidate.export_status = ExportStatus.ELIGIBLE
        candidate.export_reason = (
            f"model review {gate['decision']} over the evidence page; "
            f"deterministic gate said {candidate.export_reason}"
        )
        promoted.append((candidate, basis))
    return promoted, stale


def recompose_paper(
    paper_dir: Path,
    *,
    origin_policy: OriginExportPolicy,
    schema: dict[str, Any],
    schema_version: str,
    min_confidence: float = 0.8,
    pipeline_git_commit: str | None = None,
) -> PaperRecomposition:
    """Replay one paper's gates under a policy and compose whatever it authorizes."""

    result = PaperRecomposition(paper_id=paper_dir.name)
    manifest_path = paper_dir / "source-manifest.json"
    if not manifest_path.is_file():
        result.skipped_reason = "no source manifest"
        return result
    original = load_paper_candidates(paper_dir)
    result.candidates = len(original)
    if not original:
        result.skipped_reason = "no candidates"
        return result
    manifest = SourceManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    layouts = _paper_layouts(paper_dir, manifest)
    if not layouts:
        result.skipped_reason = "no frozen layout for exactly one source"
        return result

    replayed = [candidate.model_copy(deep=True) for candidate in original]
    validated = _validate_non_origin_candidates_in_place(
        replayed, layouts, min_confidence=min_confidence
    )
    route_candidate_attribution(validated, layouts, origin_policy)

    as_run = {
        str(candidate.observation_id or candidate.stable_id()): str(
            was.observation_id or was.stable_id()
        )
        for was, candidate in zip(original, validated, strict=True)
    }

    # Paired by position, not by observation id. Validation re-stamps the id from the
    # resolved tuple, so a candidate whose metric newly resolves comes back with a
    # different id; keying the comparison on the id would report every such candidate as
    # a new one and hide the transition it actually made.
    for was, candidate in zip(original, validated, strict=True):
        result.export_status_counts[candidate.export_status.value] += 1
        result.export_reason_counts[str(candidate.export_reason)] += 1
        if (was.export_status.value, was.export_reason) != (
            candidate.export_status.value,
            candidate.export_reason,
        ):
            result.reason_transitions[
                f"{was.export_status.value}:{was.export_reason} -> "
                f"{candidate.export_status.value}:{candidate.export_reason}"
            ] += 1
        if candidate.observation_id != was.observation_id:
            result.reidentified += 1

    eligible = [
        candidate
        for candidate in validated
        if candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}
        and candidate.attribution is not None
        and origin_policy.permits(candidate_origin_basis(candidate))
    ]
    tiers = {
        candidate.observation_id or candidate.stable_id(): ReviewTier.DETERMINISTIC
        for candidate in eligible
    }
    reviewed_bases: dict[str, OriginBasis] = {}
    if origin_policy is OriginExportPolicy.TIERED:
        promoted, stale = _promote_reviewed_candidates(
            original=original, validated=validated, paper_dir=paper_dir, already=set(tiers)
        )
        result.stale_review_gates = stale
        for candidate, basis in promoted:
            eligible.append(candidate)
            observation_id = candidate.observation_id or candidate.stable_id()
            tiers[observation_id] = ReviewTier.MODEL_REVIEWED
            reviewed_bases[observation_id] = basis
    # EEE 0.2.2 requires min_score and max_score for a continuous score, and its
    # metric_config `if` matches vacuously when score_type is absent, so a metric whose
    # scale the registry does not know cannot be expressed in this schema at all. The
    # honest move is to leave it out and count it, not to invent a scale for it, which
    # the no-guessed-values contract forbids.
    unbounded = [
        candidate
        for candidate in eligible
        if candidate.metric is None
        or candidate.metric.min_score is None
        or candidate.metric.max_score is None
    ]
    if unbounded:
        result.skipped_unbounded_metric = len(unbounded)
        excluded = {id(candidate) for candidate in unbounded}
        eligible = [candidate for candidate in eligible if id(candidate) not in excluded]
    if not eligible:
        return result

    provenance = legacy_export_provenance(
        eligible,
        reason=(
            "offline recompose of a finished census run under the "
            f"{origin_policy.value} producer-origin export policy"
        ),
    )
    bases = {
        candidate.observation_id or candidate.stable_id(): reviewed_bases.get(
            candidate.observation_id or candidate.stable_id()
        )
        or candidate_origin_basis(candidate)
        for candidate in eligible
    }
    provenance = provenance.model_copy(
        update={
            "origin_policy": origin_policy,
            "pipeline_git_commit": pipeline_git_commit,
            "authorizations": [
                item.model_copy(
                    update={
                        "origin_basis": bases[item.observation_id],
                        "review_tier": (
                            ReviewTier.HUMAN_CONFIRMED
                            if bases[item.observation_id] is OriginBasis.HUMAN_CONFIRMED
                            else tiers[item.observation_id]
                        ),
                    }
                )
                for item in provenance.authorizations
            ],
        }
    )
    records = compose_eee_records(
        manifest=manifest,
        candidates=eligible,
        schema_version=schema_version,
        provenance=provenance,
    )
    record_dir = paper_dir / TIERED_RECORD_DIRNAME
    record_dir.mkdir(parents=True, exist_ok=True)
    for stale in record_dir.glob("*.json"):
        stale.unlink()
    index: dict[str, dict[str, str]] = {}
    for record in records:
        issues = validate_eee_record(record, schema)
        if issues:
            result.invalid_records.append(
                {
                    "evaluation_id": record["evaluation_id"],
                    "issues": [f"{issue.path}: {issue.message}" for issue in issues],
                }
            )
            continue
        result.records.append(record)
        details = record["source_metadata"]["additional_details"]
        result.tier_counts[details["review_tier"]] += 1
        result.basis_counts[details["producer_origin_basis"]] += 1
        write_json(record_dir / (record["evaluation_id"].rsplit("/", 1)[-1] + ".json"), record)
        for entry in record["evaluation_results"]:
            replayed_id = entry["score_details"]["details"]["candidate_observation_id"]
            index[str(as_run.get(replayed_id, replayed_id))] = {
                "eee_record_id": record["evaluation_id"],
                "review_tier": entry["score_details"]["details"]["review_tier"],
                "producer_origin_basis": entry["score_details"]["details"]["producer_origin_basis"],
            }
    # Keyed by the observation id the run wrote, not the replayed one, so the atlas can
    # join without knowing that validation re-stamps ids from the resolved tuple.
    write_json(
        record_dir / "record-index.json",
        {"schema_version": RECORD_INDEX_SCHEMA_VERSION, "by_observation_id": index},
    )
    return result


def recompose_census(
    *,
    run_root: Path,
    origin_policy: OriginExportPolicy,
    output_dir: str = "eee",
    min_confidence: float = 0.8,
    schema_path: Path = DEFAULT_EEE_SCHEMA_PATH,
    pipeline_git_commit: str | None = None,
    paper_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Replay every paper in a run root and write the census-wide EEE artifacts."""

    schema, authority = load_schema(schema_path)
    papers: list[PaperRecomposition] = []
    for observations in sorted(run_root.glob("*/observations.jsonl")):
        paper_dir = observations.parent
        if paper_ids is not None and paper_dir.name not in paper_ids:
            continue
        papers.append(
            recompose_paper(
                paper_dir,
                origin_policy=origin_policy,
                schema=schema,
                schema_version=authority.version,
                min_confidence=min_confidence,
                pipeline_git_commit=pipeline_git_commit,
            )
        )

    export_status = Counter()
    export_reasons = Counter()
    transitions = Counter()
    tiers = Counter()
    bases = Counter()
    for paper in papers:
        export_status.update(paper.export_status_counts)
        export_reasons.update(paper.export_reason_counts)
        transitions.update(paper.reason_transitions)
        tiers.update(paper.tier_counts)
        bases.update(paper.basis_counts)

    records = [record for paper in papers for record in paper.records]
    destination = run_root / output_dir
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "census-eee.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "schema_version": RECOMPOSE_SCHEMA_VERSION,
        "origin_policy": origin_policy.value,
        "eee_schema_version": authority.version,
        "eee_schema_sha256": authority.sha256,
        "min_confidence": min_confidence,
        "pipeline_git_commit": pipeline_git_commit,
        "papers_read": len(papers),
        "papers_skipped": {
            paper.paper_id: paper.skipped_reason for paper in papers if paper.skipped_reason
        },
        "candidates": sum(paper.candidates for paper in papers),
        "records": len(records),
        "records_by_tier": dict(tiers),
        "records_by_producer_origin_basis": dict(bases),
        "export_status_counts": dict(export_status),
        "export_reason_counts": dict(export_reasons),
        "export_reason_transitions": dict(transitions),
        "candidates_reidentified": sum(paper.reidentified for paper in papers),
        "stale_review_gates": sum(paper.stale_review_gates for paper in papers),
        "skipped_unbounded_metric": sum(paper.skipped_unbounded_metric for paper in papers),
        "skipped_unbounded_metric_basis": (
            "eligible candidates whose metric has no registry-known min and max score. "
            "EEE 0.2.2 requires both for a continuous score, and inventing a scale is "
            "forbidden by the no-guessed-values contract, so these are left out and counted"
        ),
        "records_per_paper": {
            paper.paper_id: len(paper.records) for paper in papers if paper.records
        },
        "invalid_records": {
            paper.paper_id: paper.invalid_records for paper in papers if paper.invalid_records
        },
        "claim_boundary": [
            "Every record here was composed offline from candidates a model proposed and "
            "deterministic code checked. No record claims paper_produced attribution.",
            "Each record carries the review tier and producer-origin basis it was granted "
            "on, in source_metadata.additional_details and in every result's score_details.",
            "The model_reviewed tier's error rate is unmeasured until a human label file exists.",
        ],
    }
    write_json(destination / "census-eee-summary.json", summary)
    return summary
