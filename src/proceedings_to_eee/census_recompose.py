"""Replay validation and EEE composition over a finished census run, offline.

Nothing here calls a provider and nothing here rewrites `observations.jsonl`. The run's
candidates, its frozen page layout and its source manifest are enough to replay every
deterministic gate, so a change to the export policy can be applied to work already paid
for. The replay under the default `positive_only` policy is a no-op by construction only
under the lexicon the run itself recorded; this replay deliberately attributes under the
current default lexicon, so a newer lexicon can add cues the run never saw. The phase
gate therefore checks the run's own export reasons against the replayed ones and reports
every transition, and the summary names the lexicon it used.

The `model_reviewed` tier admits a candidate a page-scoped reviewer accepted, on one
rule: the cue wins. A reviewer's accept yields the `model_reviewed_origin_quote` basis
only when the deterministic verdict recorded no cue and is not externally sourced; a
reviewer that withheld origin itself grants nothing; and every candidate held back is
counted under the reason it was held back.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proceedings_to_eee.census_review import candidate_payload_sha256, load_gates
from proceedings_to_eee.composition.eee import (
    UNRESOLVED_BOUNDS,
    compose_eee_records,
    metric_bounds_basis,
)
from proceedings_to_eee.domain.attribution import (
    SELF_REFERENCE_RULE_ID,
    AttributionState,
    OriginBasis,
    OriginExportPolicy,
    PaperContext,
    ReviewTier,
)
from proceedings_to_eee.domain.export_provenance import legacy_export_provenance
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ExportStatus
from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.io import write_json
from proceedings_to_eee.resolution.attribution import (
    AttributionLexicon,
    has_table_level_cue,
    load_lexicon,
    paper_context_from_title,
    row_names_team,
)
from proceedings_to_eee.resolution.dataset_binding import bind_paper_candidates
from proceedings_to_eee.resolution.dataset_inventory import (
    inventory_content_sha256,
    load_inventory,
)
from proceedings_to_eee.resources import DEFAULT_EEE_SCHEMA_PATH
from proceedings_to_eee.sources.manifest import SourceManifest
from proceedings_to_eee.validation.candidates import (
    _validate_non_origin_candidates_in_place,
    candidate_origin_basis,
    route_candidate_attribution,
    value_export_blocker,
)
from proceedings_to_eee.validation.eee_schema import (
    load_schema,
    schema_version_allows_null_bounds,
    validate_eee_record,
)

RECOMPOSE_SCHEMA_VERSION = "census-recompose/0.4"
RECORD_INDEX_SCHEMA_VERSION = "census-eee-record-index/0.1"

#: Directory written under each paper. Deliberately not `eee/`: the run's own canonical
#: directory is the pipeline's output and this replay must never overwrite it.
TIERED_RECORD_DIRNAME = "eee-tiered"

#: Reviewer rationale phrasing that describes another party's result. Applied only on
#: opt-in, to the private review log, and the matched text never leaves this module.
REVIEWER_RATIONALE_FLAG = re.compile(
    r"\b(other|another|different|rival)\s+(team|system|group|submission|participant|model)s?\b"
    r"|\bleaderboard\b|\bparticipat|\bparticipants?\b|\bcompetit"
    r"|\bshared[- ]task (results|table)|\b(reported|taken|copied) from\b",
    re.IGNORECASE,
)

#: Reasons a reviewer-accepted candidate is held back, as recorded per paper and summary.
CUE_OVERRIDES_REVIEW = "cue_overrides_review"
REVIEWER_WITHHELD_ORIGIN = "reviewer_withheld_origin"
REVIEWER_RATIONALE_FLAGGED = "reviewer_rationale_flagged"
NO_ORIGIN_BASIS = "no_origin_basis"


@dataclass
class PaperRecomposition:
    """What one paper produced on the replay, and how it differs from its own run."""

    paper_id: str
    candidates: int = 0
    paper_kind: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    export_status_counts: Counter[str] = field(default_factory=Counter)
    export_reason_counts: Counter[str] = field(default_factory=Counter)
    reason_transitions: Counter[str] = field(default_factory=Counter)
    basis_counts: Counter[str] = field(default_factory=Counter)
    tier_counts: Counter[str] = field(default_factory=Counter)
    #: Cue firings by cue id over every replayed candidate, exported or not.
    attribution_cue_counts: Counter[str] = field(default_factory=Counter)
    #: Rows whose label names the title's team inside a table that carries a table-level
    #: cue. They are held back by that cue like every other row; `foreign_team_row` must
    #: never be what holds them back, which the second counter checks.
    team_rows_in_cue_bearing_tables: int = 0
    team_rows_with_foreign_team_cue: int = 0
    invalid_records: list[dict[str, Any]] = field(default_factory=list)
    #: Candidates whose observation id changed because the replay resolved their tuple
    #: further than the original run did. The id is derived from the tuple, so this is
    #: expected whenever the registry learns a name; it is counted, never hidden.
    reidentified: int = 0
    #: Review gates whose candidate payload hash no longer matches the candidate. The
    #: page said something about a tuple that has since changed, so the gate is ignored.
    stale_review_gates: int = 0
    #: Candidates left out because no registry established a scale for their metric and
    #: the schema authority cannot express an unknown scale, so no schema-valid record
    #: can be written without inventing one.
    skipped_unbounded_metric: int = 0
    #: Candidates composed with explicit null bounds under an authority that allows
    #: them. The record says the scale is unknown instead of the candidate being dropped.
    composed_with_unresolved_bounds: int = 0
    #: Reviewer-accepted candidates left out because no canonical metric id was
    #: established. Counted so the drop is visible instead of silent.
    skipped_no_metric_id: int = 0
    #: Reviewer-accepted candidates left out because their producer-origin basis is
    #: none under the export policy.
    skipped_no_origin_basis: int = 0
    #: Reviewer-accepted candidates left out because their value can never export as a
    #: score: a signed difference or a value outside its metric's declared bounds. A
    #: reviewer's accept speaks to the tuple, not to the number being a score. Keyed by
    #: the blocker's reason id.
    skipped_value_blockers: Counter[str] = field(default_factory=Counter)
    #: Reviewer accepts whose deterministic verdict recorded a cue or external origin.
    skipped_cue_overrides_review: int = 0
    #: Reviewer-proposed accept_tuple_only without an origin quote: origin withheld.
    skipped_reviewer_withheld_origin: int = 0
    #: Opt-in only: the reviewer's private rationale described another party's result.
    skipped_reviewer_rationale_flagged: int = 0
    #: Every held-back reviewer-accepted candidate, by the reason it was held back.
    review_demotions: Counter[str] = field(default_factory=Counter)
    #: Opt-in only: the private review log was requested but the paper has none.
    reviewer_rationale_log_missing: bool = False
    #: Dataset binding, applied only when the paper has a verified inventory gate. The
    #: counts cover every validated candidate with a scope, exported or not.
    dataset_inventory_present: bool = False
    dataset_inventory_entries: int = 0
    dataset_binding_counts: Counter[str] = field(default_factory=Counter)
    dataset_binding_rules: Counter[str] = field(default_factory=Counter)
    split_derived_counts: Counter[str] = field(default_factory=Counter)
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


def flagged_review_rationales(paper_dir: Path) -> set[tuple[str, str | None]] | None:
    """Keys of private review records whose rationale matches the flag pattern.

    Keyed by observation id and candidate payload hash, which is how a gate binds its
    candidate. When one candidate was reviewed more than once the last record is the one
    the gate came from. Returns None when the paper has no private log. The rationale is
    read here and nowhere else; only the match outcome leaves this function.
    """

    path = paper_dir / "private" / "model-review.jsonl"
    if not path.is_file():
        return None
    latest: dict[tuple[str, str | None], Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record.get("observation_id")), record.get("candidate_payload_sha256"))
        latest[key] = record.get("rationale")
    return {
        key
        for key, rationale in latest.items()
        if isinstance(rationale, str) and REVIEWER_RATIONALE_FLAG.search(rationale)
    }


@dataclass
class _Promotion:
    """What the page-scoped reviewer's gates admitted, and what they could not."""

    promoted: list[tuple[CandidateObservation, OriginBasis]] = field(default_factory=list)
    stale: int = 0
    no_metric_id: int = 0
    no_origin_basis: int = 0
    value_blockers: Counter[str] = field(default_factory=Counter)
    cue_overrides_review: int = 0
    reviewer_withheld_origin: int = 0
    reviewer_rationale_flagged: int = 0
    demotions: Counter[str] = field(default_factory=Counter)
    rationale_log_missing: bool = False


def _reviewed_basis(
    candidate: CandidateObservation, gate: dict[str, Any]
) -> tuple[OriginBasis, str | None]:
    """Apply the one promotion rule and say why a candidate earns no basis.

    An accept earns `model_reviewed_origin_quote` only when the deterministic verdict
    recorded no cue and is not externally sourced: a cue is evidence the page gave, and
    a reviewer's quote cannot talk over it. The one exception is a row whose label names
    the paper itself: the table-scoped cues recorded beside it describe the table, not
    that row, so they do not block it. An accept_tuple_only the reviewer itself
    proposed without an origin quote is the reviewer withholding origin, which grants
    nothing either. Otherwise the candidate keeps the basis its verdict allows, and a
    verdict that allows none is reported with its state, rule and the cues that fired,
    so the summary can say which cue held an `accept_tuple_only` back.
    """

    verdict = candidate.attribution
    assert verdict is not None
    deterministic = candidate_origin_basis(candidate)
    cue_ids = ";".join(dict.fromkeys(cue.cue_id for cue in verdict.cues))
    no_basis = f"{NO_ORIGIN_BASIS}:{verdict.state.value}:{verdict.rule_id}" + (
        f":{cue_ids}" if cue_ids else ""
    )
    if gate["decision"] == "accept":
        blocking = verdict.cues and verdict.rule_id != SELF_REFERENCE_RULE_ID
        if blocking or verdict.state is AttributionState.EXTERNALLY_SOURCED:
            cues = cue_ids or verdict.rule_id
            return OriginBasis.NONE, f"{CUE_OVERRIDES_REVIEW}:{verdict.state.value}:{cues}"
        if deterministic is OriginBasis.NONE:
            return OriginBasis.NONE, no_basis
        return OriginBasis.MODEL_REVIEWED_ORIGIN_QUOTE, None
    if (
        gate.get("proposed_decision") == "accept_tuple_only"
        and gate.get("origin_quote_sha256") is None
    ):
        return OriginBasis.NONE, REVIEWER_WITHHELD_ORIGIN
    if deterministic is OriginBasis.NONE:
        return OriginBasis.NONE, no_basis
    return deterministic, None


def _promote_reviewed_candidates(
    *,
    original: list[CandidateObservation],
    validated: list[CandidateObservation],
    paper_dir: Path,
    already: set[str],
    demote_reviewer_flagged_origin: bool = False,
) -> _Promotion:
    """Admit candidates a page-scoped reviewer accepted, on the basis they earned.

    Gates are keyed by the observation id the run wrote, and validation re-stamps that id
    from the resolved tuple, so the lookup pairs by position and the payload hash decides
    whether the gate still describes this candidate. A referential or field-provenance
    failure does not block this tier; it travels into the record instead, which is the
    whole point of naming the tier. A value that can never be a score does block it: the
    reviewer confirmed what the tuple says, not that a signed difference or an
    out-of-range number is a score on the metric's scale. The basis itself follows
    `_reviewed_basis`, and the opt-in rationale flag can only take a basis away, never
    grant one.
    """

    outcome = _Promotion()
    gates = load_gates(paper_dir)
    if not gates:
        return outcome
    flagged: set[tuple[str, str | None]] = set()
    if demote_reviewer_flagged_origin:
        found = flagged_review_rationales(paper_dir)
        outcome.rationale_log_missing = found is None
        flagged = found or set()
    for was, candidate in zip(original, validated, strict=True):
        as_run = str(was.observation_id)
        gate = gates.get(as_run)
        if not gate or gate.get("decision") not in {"accept", "accept_tuple_only"}:
            continue
        if gate.get("candidate_payload_sha256") != candidate_payload_sha256(
            was.model_dump(mode="json")
        ):
            outcome.stale += 1
            continue
        observation_id = candidate.observation_id or candidate.stable_id()
        if observation_id in already:
            continue
        if candidate.metric is None or not candidate.metric.canonical_id:
            outcome.no_metric_id += 1
            continue
        if candidate.value is None or candidate.scope is None:
            continue
        blocker = value_export_blocker(candidate)
        if blocker is not None:
            outcome.value_blockers[blocker] += 1
            continue
        if candidate.attribution is None:
            continue
        basis, reason = _reviewed_basis(candidate, gate)
        log_key = (as_run, gate.get("candidate_payload_sha256"))
        if basis is not OriginBasis.NONE and log_key in flagged:
            basis, reason = OriginBasis.NONE, REVIEWER_RATIONALE_FLAGGED
        if basis is OriginBasis.NONE:
            assert reason is not None
            outcome.demotions[f"{gate['decision']} -> none: {reason}"] += 1
            if reason.startswith(CUE_OVERRIDES_REVIEW):
                outcome.cue_overrides_review += 1
            elif reason == REVIEWER_WITHHELD_ORIGIN:
                outcome.reviewer_withheld_origin += 1
            elif reason == REVIEWER_RATIONALE_FLAGGED:
                outcome.reviewer_rationale_flagged += 1
            else:
                outcome.no_origin_basis += 1
            continue
        candidate.export_status = ExportStatus.ELIGIBLE
        candidate.export_reason = (
            f"model review {gate['decision']} over the evidence page; "
            f"deterministic gate said {candidate.export_reason}"
        )
        outcome.promoted.append((candidate, basis))
    return outcome


def _count_paper_context(
    result: PaperRecomposition,
    validated: list[CandidateObservation],
    context: PaperContext,
    lexicon: AttributionLexicon,
) -> None:
    for candidate in validated:
        verdict = candidate.attribution
        if verdict is None:
            continue
        for cue in verdict.cues:
            result.attribution_cue_counts[cue.cue_id] += 1
        if has_table_level_cue(verdict, lexicon) and row_names_team(context, verdict.row_label):
            result.team_rows_in_cue_bearing_tables += 1
            if any(cue.cue_id == "foreign_team_row" for cue in verdict.cues):
                result.team_rows_with_foreign_team_cue += 1


def recompose_paper(
    paper_dir: Path,
    *,
    origin_policy: OriginExportPolicy,
    schema: dict[str, Any],
    schema_version: str,
    min_confidence: float = 0.8,
    pipeline_git_commit: str | None = None,
    demote_reviewer_flagged_origin: bool = False,
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
    context = paper_context_from_title(manifest.title)
    result.paper_kind = context.paper_kind.value

    replayed = [candidate.model_copy(deep=True) for candidate in original]
    validated = _validate_non_origin_candidates_in_place(
        replayed, layouts, min_confidence=min_confidence
    )
    # Binding runs after the ids are stamped and before attribution: it attaches a
    # sidecar and nothing else, so no gate below can see a different candidate.
    inventory = load_inventory(paper_dir)
    if inventory is not None:
        inventory_sha256 = inventory_content_sha256(inventory)
        result.dataset_inventory_present = True
        result.dataset_inventory_entries = len(inventory.get("entries") or [])
        bound = bind_paper_candidates(
            validated,
            inventory=inventory,
            inventory_sha256=inventory_sha256,
            layout=next(iter(layouts.values())),
        )
        result.dataset_binding_counts = bound.classes
        result.dataset_binding_rules = bound.rules
        result.split_derived_counts = bound.split_sources
    lexicon = load_lexicon()
    route_candidate_attribution(validated, layouts, origin_policy, context, lexicon)
    _count_paper_context(result, validated, context, lexicon)

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
        promotion = _promote_reviewed_candidates(
            original=original,
            validated=validated,
            paper_dir=paper_dir,
            already=set(tiers),
            demote_reviewer_flagged_origin=demote_reviewer_flagged_origin,
        )
        result.stale_review_gates = promotion.stale
        result.skipped_no_metric_id = promotion.no_metric_id
        result.skipped_no_origin_basis = promotion.no_origin_basis
        result.skipped_value_blockers = promotion.value_blockers
        result.skipped_cue_overrides_review = promotion.cue_overrides_review
        result.skipped_reviewer_withheld_origin = promotion.reviewer_withheld_origin
        result.skipped_reviewer_rationale_flagged = promotion.reviewer_rationale_flagged
        result.review_demotions = promotion.demotions
        result.reviewer_rationale_log_missing = promotion.rationale_log_missing
        for candidate, basis in promotion.promoted:
            eligible.append(candidate)
            observation_id = candidate.observation_id or candidate.stable_id()
            tiers[observation_id] = ReviewTier.MODEL_REVIEWED
            reviewed_bases[observation_id] = basis
    # A metric whose scale no registry established gets no bounds. EEE 0.3.0 can say so
    # with explicit nulls, so the candidate composes and is counted. EEE 0.2.2 requires
    # min_score and max_score for a continuous score, and its metric_config `if` matches
    # vacuously when score_type is absent, so there the honest move is to leave the
    # candidate out and count it, not to invent a scale for it, which the
    # no-guessed-values contract forbids.
    unbounded = [
        candidate for candidate in eligible if metric_bounds_basis(candidate) == UNRESOLVED_BOUNDS
    ]
    if schema_version_allows_null_bounds(schema_version):
        result.composed_with_unresolved_bounds = len(unbounded)
    elif unbounded:
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
    demote_reviewer_flagged_origin: bool = False,
) -> dict[str, Any]:
    """Replay every paper in a run root and write the census-wide EEE artifacts."""

    schema, authority = load_schema(schema_path)
    lexicon = load_lexicon()
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
                demote_reviewer_flagged_origin=demote_reviewer_flagged_origin,
            )
        )

    export_status = Counter()
    export_reasons = Counter()
    transitions = Counter()
    tiers = Counter()
    bases = Counter()
    value_blockers = Counter()
    cue_counts = Counter()
    demotions = Counter()
    paper_kinds = Counter()
    binding_classes = Counter()
    binding_rules = Counter()
    split_sources = Counter()
    for paper in papers:
        export_status.update(paper.export_status_counts)
        export_reasons.update(paper.export_reason_counts)
        transitions.update(paper.reason_transitions)
        tiers.update(paper.tier_counts)
        bases.update(paper.basis_counts)
        value_blockers.update(paper.skipped_value_blockers)
        cue_counts.update(paper.attribution_cue_counts)
        demotions.update(paper.review_demotions)
        binding_classes.update(paper.dataset_binding_counts)
        binding_rules.update(paper.dataset_binding_rules)
        split_sources.update(paper.split_derived_counts)
        if paper.paper_kind is not None:
            paper_kinds[paper.paper_kind] += 1

    records = [record for paper in papers for record in paper.records]
    destination = run_root / output_dir
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "census-eee.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "schema_version": RECOMPOSE_SCHEMA_VERSION,
        "origin_policy": origin_policy.value,
        "demote_reviewer_flagged_origin": demote_reviewer_flagged_origin,
        "eee_schema_version": authority.version,
        "eee_schema_sha256": authority.sha256,
        "attribution_lexicon_id": lexicon.lexicon_id,
        "attribution_lexicon_sha256": lexicon.sha256,
        "min_confidence": min_confidence,
        "pipeline_git_commit": pipeline_git_commit,
        "papers_read": len(papers),
        "papers_skipped": {
            paper.paper_id: paper.skipped_reason for paper in papers if paper.skipped_reason
        },
        "paper_kinds": dict(sorted(paper_kinds.items())),
        "candidates": sum(paper.candidates for paper in papers),
        "records": len(records),
        "records_by_tier": dict(tiers),
        "records_by_producer_origin_basis": dict(bases),
        "export_status_counts": dict(export_status),
        "export_reason_counts": dict(export_reasons),
        "export_reason_transitions": dict(transitions),
        "attribution_cue_counts": dict(sorted(cue_counts.items())),
        "team_rows_in_cue_bearing_tables": sum(
            paper.team_rows_in_cue_bearing_tables for paper in papers
        ),
        "team_rows_with_foreign_team_cue": sum(
            paper.team_rows_with_foreign_team_cue for paper in papers
        ),
        "candidates_reidentified": sum(paper.reidentified for paper in papers),
        "stale_review_gates": sum(paper.stale_review_gates for paper in papers),
        "skipped_no_metric_id": sum(paper.skipped_no_metric_id for paper in papers),
        "skipped_no_origin_basis": sum(paper.skipped_no_origin_basis for paper in papers),
        "skipped_cue_overrides_review": sum(paper.skipped_cue_overrides_review for paper in papers),
        "skipped_reviewer_withheld_origin": sum(
            paper.skipped_reviewer_withheld_origin for paper in papers
        ),
        "skipped_reviewer_rationale_flagged": sum(
            paper.skipped_reviewer_rationale_flagged for paper in papers
        ),
        "review_demotion_counts": dict(sorted(demotions.items())),
        "reviewer_rationale_logs_missing": sorted(
            paper.paper_id for paper in papers if paper.reviewer_rationale_log_missing
        ),
        "skipped_value_blockers": dict(value_blockers),
        "skipped_value_blockers_basis": (
            "reviewer-accepted candidates whose value can never export as a score: a "
            "signed difference, or a value outside the bounds its metric declares"
        ),
        "skipped_unbounded_metric": sum(paper.skipped_unbounded_metric for paper in papers),
        "composed_with_unresolved_bounds": sum(
            paper.composed_with_unresolved_bounds for paper in papers
        ),
        "unresolved_bounds_basis": (
            "eligible candidates whose metric has no registry-established min and max "
            "score. Inventing a scale is forbidden by the no-guessed-values contract. "
            "Under an EEE schema that accepts null bounds (0.3.0 and later) they compose "
            "with explicit nulls and metric_bounds=unresolved and are counted as "
            "composed_with_unresolved_bounds; under 0.2.2, which requires both numbers, "
            "they are left out and counted as skipped_unbounded_metric"
        ),
        "papers_with_dataset_inventory": sum(paper.dataset_inventory_present for paper in papers),
        "dataset_inventory_entries": sum(paper.dataset_inventory_entries for paper in papers),
        "dataset_binding_counts": dict(sorted(binding_classes.items())),
        "dataset_binding_rule_counts": dict(sorted(binding_rules.items())),
        "split_derived_counts": dict(sorted(split_sources.items())),
        "dataset_binding_basis": (
            "candidates of papers with a verified dataset inventory, bound by the exact, "
            "caption or unique rule or left unbound (none); the raw dataset string is "
            "never rewritten and a paper without an inventory is not bound at all"
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
            "on, in source_metadata.additional_details and in every result's score_details, "
            "together with the ids of the attribution cues that fired on it.",
            "A reviewer accept earns its basis only where the deterministic verdict "
            "recorded no cue; a cue or a reviewer that withheld origin holds the "
            "candidate back, counted under review_demotion_counts.",
            "The model_reviewed tier's error rate is unmeasured until a human label file exists.",
            "Where a paper has a dataset inventory, source_data.dataset_name is the bound "
            "entry's name and additional_details keeps the raw string with the binding "
            "class and rule; the inventory's entries are model proposals whose quotes "
            "were re-found on the frozen layout, and their error rate is unmeasured.",
        ],
    }
    write_json(destination / "census-eee-summary.json", summary)
    return summary
