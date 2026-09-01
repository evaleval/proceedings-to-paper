"""Deterministic candidate verification and export gating."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict

from proceedings_to_eee.domain.observation import CandidateObservation, EvidenceAnchor
from proceedings_to_eee.domain.status import (
    ActorRole,
    ClaimType,
    EvidenceKind,
    ExportStatus,
    ReferentialStatus,
    TextSupportStatus,
)
from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.extraction.region_index import build_region_index
from proceedings_to_eee.resolution.attribution import attribute_candidate, load_lexicon
from proceedings_to_eee.resolution.metrics import resolve_metric_value
from proceedings_to_eee.validation.physical_cells import (
    PhysicalCellBindingStatus,
    PhysicalCellIdentity,
    PhysicalCellLocator,
)

_PHYSICAL_CELL_CONFLICT_NOTE = (
    "semantic safety: incompatible proposals share one physical value cell"
)
_PHYSICAL_CELL_CONFLICT_REASON = (
    "physical-cell conflict: proposals for the same printed value "
    "have incompatible essential semantics"
)


def normalize_evidence_text(value: str) -> str:
    """Normalize layout whitespace without altering numbers or punctuation."""

    value = value.replace("\u00a0", " ").replace("\u2212", "-")
    return re.sub(r"\s+", " ", value).strip()


def verify_text_support(candidate: CandidateObservation, layouts: dict[str, PdfLayout]) -> None:
    """Check quote presence and raw-value presence, independently of scope semantics."""

    outcomes: list[TextSupportStatus] = []
    for anchor in candidate.evidence:
        layout = layouts.get(anchor.source_id)
        if layout is None or anchor.page > layout.page_count:
            outcomes.append(TextSupportStatus.UNSUPPORTED)
            continue
        page = layout.pages[anchor.page - 1]
        page_normalized = normalize_evidence_text(page.text)
        quote_normalized = normalize_evidence_text(anchor.quote)
        if not quote_normalized or quote_normalized not in page_normalized:
            outcomes.append(TextSupportStatus.UNSUPPORTED)
            continue
        if candidate.value is not None:
            raw_normalized = normalize_evidence_text(candidate.value.raw)
            raw_plain = raw_normalized.rstrip("%").strip()
            quote_plain = quote_normalized.replace(",", ".")
            if raw_normalized not in quote_normalized and raw_plain not in quote_plain:
                outcomes.append(TextSupportStatus.PARTIALLY_SUPPORTED)
                continue
        occurrences = page_normalized.count(quote_normalized)
        if occurrences > 1:
            note = f"evidence quote occurs {occurrences} times on page {anchor.page}"
            if note not in candidate.notes:
                candidate.notes.append(note)
        outcomes.append(TextSupportStatus.SUPPORTED)
    if outcomes and all(outcome == TextSupportStatus.SUPPORTED for outcome in outcomes):
        candidate.text_support = TextSupportStatus.SUPPORTED
    elif (
        TextSupportStatus.SUPPORTED in outcomes or TextSupportStatus.PARTIALLY_SUPPORTED in outcomes
    ):
        candidate.text_support = TextSupportStatus.PARTIALLY_SUPPORTED
    else:
        candidate.text_support = TextSupportStatus.UNSUPPORTED


def resolve_references(candidate: CandidateObservation) -> None:
    """Resolve only facts backed by the typed candidate or the explicit metric registry."""

    if candidate.metric and candidate.value:
        candidate.metric, candidate.value, resolution_note = resolve_metric_value(
            candidate.metric,
            candidate.value,
            (anchor.quote for anchor in candidate.evidence),
        )
        if resolution_note and resolution_note not in candidate.notes:
            candidate.notes.append(resolution_note)
    if candidate.claim_type != ClaimType.PRIMARY_RESULT:
        candidate.referential_status = ReferentialStatus.UNVERIFIED
        return
    evaluated = [role for role in candidate.roles if role.role == ActorRole.EVALUATED_SYSTEM]
    required_present = all(
        (
            len(evaluated) == 1,
            candidate.scope is not None,
            candidate.metric is not None,
            candidate.value is not None,
        )
    )
    if not required_present:
        candidate.referential_status = ReferentialStatus.UNRESOLVED
        return
    assert candidate.metric is not None
    assert candidate.value is not None
    if candidate.metric.canonical_id is None or candidate.metric.lower_is_better is None:
        candidate.referential_status = ReferentialStatus.UNRESOLVED
        return
    if candidate.metric.unit is None or candidate.value.unit is None:
        candidate.referential_status = ReferentialStatus.UNRESOLVED
        return
    if candidate.metric.unit != candidate.value.unit:
        candidate.referential_status = ReferentialStatus.UNRESOLVED
        return
    candidate.referential_status = ReferentialStatus.RESOLVED


def apply_export_policy(candidate: CandidateObservation, min_confidence: float = 0.8) -> None:
    """Apply a conservative, explainable EEE eligibility gate."""

    if candidate.claim_type != ClaimType.PRIMARY_RESULT:
        candidate.export_status = ExportStatus.NOT_ELIGIBLE
        candidate.export_reason = f"claim_type={candidate.claim_type} is not a primary result"
    elif candidate.text_support != TextSupportStatus.SUPPORTED:
        candidate.export_status = ExportStatus.NEEDS_REVIEW
        candidate.export_reason = f"text_support={candidate.text_support}"
    elif candidate.referential_status != ReferentialStatus.RESOLVED:
        candidate.export_status = ExportStatus.NEEDS_REVIEW
        candidate.export_reason = f"referential_status={candidate.referential_status}"
    elif candidate.extraction_confidence < min_confidence:
        candidate.export_status = ExportStatus.NEEDS_REVIEW
        candidate.export_reason = (
            f"extraction_confidence={candidate.extraction_confidence:.3f} "
            f"below {min_confidence:.3f}"
        )
    else:
        candidate.export_status = ExportStatus.ELIGIBLE
        candidate.export_reason = "passed primary-result, evidence, reference, and confidence gates"


def _normalized_semantic_name(value: str | None) -> str:
    """Normalize a semantic label for conservative identity comparisons."""

    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _evaluated_system_name(candidate: CandidateObservation) -> str:
    return next(
        (
            _normalized_semantic_name(role.raw_name)
            for role in candidate.roles
            if role.role == ActorRole.EVALUATED_SYSTEM
        ),
        "",
    )


def _semantic_safety_issues(candidate: CandidateObservation) -> list[str]:
    """Find structural contradictions without trying to infer paper semantics.

    These checks intentionally abstain only on identities that cannot all be true
    for one atomic primary result. They do not promote or rewrite any field.
    """

    if candidate.claim_type != ClaimType.PRIMARY_RESULT:
        return []
    system = _evaluated_system_name(candidate)
    dataset = _normalized_semantic_name(
        candidate.scope.dataset_raw if candidate.scope is not None else None
    )
    metric = _normalized_semantic_name(
        candidate.metric.raw_name if candidate.metric is not None else None
    )
    language = _normalized_semantic_name(
        candidate.scope.language if candidate.scope is not None else None
    )
    issues: list[str] = []
    if system and dataset and system == dataset:
        issues.append("evaluated system and dataset have the same identity")
    if metric and dataset and metric == dataset:
        issues.append("metric and dataset have the same identity")
    if dataset and language and dataset == language:
        issues.append("dataset identity is only the language slice")
    if candidate.scope is not None:
        subset = _normalized_semantic_name(candidate.scope.subset)
        group = _normalized_semantic_name(candidate.scope.group)
        slice_names = {
            " ".join(parts) for parts in ((subset, group), (group, subset)) if all(parts)
        }
        if dataset and dataset in slice_names:
            issues.append("dataset identity contains only subset and group labels")
    if re.fullmatch(r"prompt(?: \d+)?", dataset):
        issues.append("dataset identity is only a prompt condition")

    role_names: dict[str, set[ActorRole]] = defaultdict(set)
    for role in candidate.roles:
        role_names[_normalized_semantic_name(role.raw_name)].add(role.role)
    if system and len(role_names[system]) > 1:
        issues.append("the evaluated system has multiple roles in one observation")

    non_system_suffixes = (
        " annotations",
        " annotators",
        " corpus",
        " dataset",
        " ground truth",
        " human labels",
        " human reference",
        " labels",
        " test set",
        " training set",
    )
    if system and any(
        system == suffix.strip() or system.endswith(suffix) for suffix in non_system_suffixes
    ):
        issues.append("evaluated-system identity appears to name reference data")

    non_metric_suffixes = (" corpus", " dataset", " data set", " test set", " training set")
    if metric and any(
        metric == suffix.strip() or metric.endswith(suffix) for suffix in non_metric_suffixes
    ):
        issues.append("metric identity appears to name a dataset")

    non_dataset_suffixes = (
        " api",
        " classifier",
        " classifiers",
        " model",
        " models",
        " system",
    )
    if dataset and any(
        dataset == suffix.strip() or dataset.endswith(suffix) for suffix in non_dataset_suffixes
    ):
        issues.append("dataset identity appears to name a system or model family")
    return issues


def _route_semantic_safety_issues(candidate: CandidateObservation) -> None:
    issues = _semantic_safety_issues(candidate)
    if not issues:
        return
    candidate.referential_status = ReferentialStatus.WRONG_SCOPE
    candidate.export_status = ExportStatus.NEEDS_REVIEW
    candidate.export_reason = "semantic safety check: " + "; ".join(issues)
    for issue in issues:
        note = f"semantic safety: {issue}"
        if note not in candidate.notes:
            candidate.notes.append(note)


def _ambiguity_value_key(candidate: CandidateObservation) -> tuple[float, str] | None:
    if candidate.value is None:
        return None
    return candidate.value.numeric, _normalized_semantic_name(candidate.value.unit)


def _compatible_semantic_name(left: str, right: str) -> bool:
    """Treat exact names and explicit longer aliases as compatible identities."""

    if not left or not right or left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    return left_tokens <= right_tokens or right_tokens <= left_tokens


def _candidate_semantic_fields(candidate: CandidateObservation) -> dict[str, str]:
    scope = candidate.scope
    metric = candidate.metric
    return {
        "system": _evaluated_system_name(candidate),
        "dataset": _normalized_semantic_name(scope.dataset_raw if scope else None),
        "split": _normalized_semantic_name(scope.split if scope else None),
        "subset": _normalized_semantic_name(scope.subset if scope else None),
        "group": _normalized_semantic_name(scope.group if scope else None),
        "language": _normalized_semantic_name(scope.language if scope else None),
        "aggregation": _normalized_semantic_name(scope.aggregation if scope else None),
        "metric": _normalized_semantic_name(
            (metric.canonical_id or metric.raw_name) if metric else None
        ),
    }


def _semantics_conflict(
    left: CandidateObservation,
    right: CandidateObservation,
) -> bool:
    left_fields = _candidate_semantic_fields(left)
    right_fields = _candidate_semantic_fields(right)
    return any(
        left_fields[name]
        and right_fields[name]
        and not _compatible_semantic_name(left_fields[name], right_fields[name])
        for name in left_fields
    )


def _anchor_structure_compatible(left: EvidenceAnchor, right: EvidenceAnchor) -> bool:
    left_quote = normalize_evidence_text(left.quote).casefold()
    right_quote = normalize_evidence_text(right.quote).casefold()
    if not left_quote or not right_quote:
        return False
    if left_quote not in right_quote and right_quote not in left_quote:
        return False
    return all(
        _compatible_semantic_name(
            _normalized_semantic_name(getattr(left, field)),
            _normalized_semantic_name(getattr(right, field)),
        )
        for field in ("label", "row", "column")
    )


def _candidates_share_ambiguous_evidence(
    left: CandidateObservation,
    right: CandidateObservation,
) -> bool:
    return any(
        left_anchor.source_id == right_anchor.source_id
        and left_anchor.page == right_anchor.page
        and _anchor_structure_compatible(left_anchor, right_anchor)
        for left_anchor in left.evidence
        for right_anchor in right.evidence
    )


def _route_ambiguous_evidence(candidates: list[CandidateObservation]) -> None:
    """Abstain when the same quoted value receives incompatible semantics.

    Overlapping extraction blocks can propose several meanings for one printed
    value. The resolver must not silently choose one. Every conflicting proposal
    remains in the review artifact, while none is eligible for EEE export.
    """

    groups: dict[tuple[str, tuple[float, str]], list[CandidateObservation]] = defaultdict(list)
    for candidate in candidates:
        if candidate.claim_type != ClaimType.PRIMARY_RESULT:
            continue
        value_key = _ambiguity_value_key(candidate)
        if value_key is None:
            continue
        groups[(candidate.paper_id, value_key)].append(candidate)

    ambiguous_ids: set[int] = set()
    for group in groups.values():
        unique = {id(candidate): candidate for candidate in group}
        if len(unique) < 2:
            continue
        items = list(unique.values())
        for index, left in enumerate(items):
            for right in items[index + 1 :]:
                if not _semantics_conflict(left, right):
                    continue
                if not _candidates_share_ambiguous_evidence(left, right):
                    continue
                ambiguous_ids.update((id(left), id(right)))

    for candidate in candidates:
        if id(candidate) not in ambiguous_ids:
            continue
        candidate.referential_status = ReferentialStatus.WRONG_SCOPE
        candidate.export_status = ExportStatus.NEEDS_REVIEW
        candidate.export_reason = (
            "ambiguous evidence: the same quoted value has incompatible essential semantics"
        )
        note = "semantic safety: incompatible proposals share the same quoted value"
        if note not in candidate.notes:
            candidate.notes.append(note)


def _route_attribution(
    candidates: list[CandidateObservation], layouts: dict[str, PdfLayout]
) -> None:
    """Attach a deterministic attribution verdict and demote what it places elsewhere.

    Demote-only: only a positive PAPER_PRODUCED verdict may remain in the export gate.
    The current resolver never produces that verdict. Routing is to NEEDS_REVIEW rather
    than NOT_ELIGIBLE because NOT_ELIGIBLE carries no risk reason in the human review
    lane and would silently drop the candidate, and because false_primary_export keys on
    {ELIGIBLE, EXPORTED} so NEEDS_REVIEW already clears that gate.
    """

    lexicon = load_lexicon()
    indexes = {source_id: build_region_index(layout) for source_id, layout in layouts.items()}
    pages = {
        source_id: {fragment.page: fragment for fragment in layout.pages}
        for source_id, layout in layouts.items()
    }
    for candidate in candidates:
        anchor = candidate.evidence[0]
        verdict = attribute_candidate(
            candidate,
            indexes.get(anchor.source_id, {}).get(anchor.page),
            pages.get(anchor.source_id, {}).get(anchor.page),
            lexicon,
        )
        candidate.attribution = verdict
        if verdict.allows_canonical_export:
            continue
        if candidate.export_status not in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}:
            continue
        cues = ", ".join(cue.cue_id for cue in verdict.cues) or verdict.rule_id
        candidate.export_status = ExportStatus.NEEDS_REVIEW
        candidate.export_reason = f"attribution={verdict.state.value}: {cues}"
        note = f"attribution: {verdict.state.value} via {verdict.rule_id} ({cues})"
        if note not in candidate.notes:
            candidate.notes.append(note)


def _restore_physical_cell_conflict(candidate: CandidateObservation) -> None:
    """Keep a deterministic merge conflict demoted across validation replays.

    Reference resolution necessarily recomputes its status from the representative
    candidate, which cannot itself carry every alternate proposal. The merge retains
    those alternates in provenance notes and writes this marker, so replay restores the
    conservative state after all ordinary semantic routing has run.
    """

    if _PHYSICAL_CELL_CONFLICT_NOTE not in candidate.notes:
        return
    candidate.referential_status = ReferentialStatus.WRONG_SCOPE
    candidate.export_status = ExportStatus.NEEDS_REVIEW
    candidate.export_reason = _PHYSICAL_CELL_CONFLICT_REASON


def validate_candidates(
    candidates: Iterable[CandidateObservation],
    layouts: dict[str, PdfLayout],
    min_confidence: float = 0.8,
) -> list[CandidateObservation]:
    validated: list[CandidateObservation] = []
    for candidate in candidates:
        verify_text_support(candidate, layouts)
        resolve_references(candidate)
        apply_export_policy(candidate, min_confidence=min_confidence)
        _route_semantic_safety_issues(candidate)
        candidate.observation_id = candidate.stable_id()
        validated.append(candidate)
    _route_ambiguous_evidence(validated)
    for candidate in validated:
        _restore_physical_cell_conflict(candidate)
    _route_attribution(validated, layouts)
    return validated


def _semantic_key(candidate: CandidateObservation) -> str:
    payload = {
        "paper_id": candidate.paper_id,
        "claim_type": candidate.claim_type,
        "roles": sorted(
            (role.role, role.raw_name.casefold(), role.version) for role in candidate.roles
        ),
        "scope": candidate.scope.model_dump(mode="json") if candidate.scope else None,
        "metric": candidate.metric.model_dump(mode="json") if candidate.metric else None,
        "value": candidate.value.model_dump(mode="json") if candidate.value else None,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _merge_semantic_duplicates(
    group: list[CandidateObservation],
) -> CandidateObservation:
    """Preserve the historical semantic-only merge exactly."""

    first = group[0]
    anchors = {anchor.quote_sha256: anchor for item in group for anchor in item.evidence}
    notes = list(dict.fromkeys(note for item in group for note in item.notes))
    notes.append(f"merged {len(group)} duplicate proposals")
    combined = first.model_copy(
        update={
            "observation_id": None,
            "evidence": list(anchors.values()),
            "notes": notes,
            "extraction_confidence": max(item.extraction_confidence for item in group),
        }
    )
    combined.observation_id = combined.stable_id()
    return combined


def _information_count(value: object) -> int:
    if value is None or value == "" or value == [] or value == {}:
        return 0
    if isinstance(value, dict):
        return sum(_information_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_information_count(item) for item in value)
    return 1


def _proposal_snapshot(candidate: CandidateObservation) -> str:
    """Serialize fields that cannot all fit in one representative candidate."""

    payload = candidate.model_dump(mode="json")
    # Evidence and notes are unioned as their native typed fields below. Keeping them
    # out of the snapshot avoids recursive growth if a merged artifact is reloaded.
    payload.pop("evidence", None)
    payload.pop("notes", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _exact_candidate_key(candidate: CandidateObservation) -> str:
    """Serialize every substantive field of one candidate proposal.

    ``observation_id`` is derived from the other fields and therefore cannot make
    otherwise identical proposals distinct. This key is deliberately stricter than
    the semantic key: it lets us collapse literal duplicate model proposals without
    treating an unbound repeated table value as one physical cell.
    """

    payload = candidate.model_dump(mode="json")
    payload.pop("observation_id", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _physical_representative(group: list[CandidateObservation]) -> CandidateObservation:
    """Choose the richest candidate with a canonical, input-order-independent tie break."""

    def key(candidate: CandidateObservation) -> tuple[int, float, str]:
        semantic_payload = json.loads(_semantic_key(candidate))
        return (
            -_information_count(semantic_payload),
            -candidate.extraction_confidence,
            _proposal_snapshot(candidate),
        )

    return min(group, key=key)


def _value_interpretation_conflicts(
    left: CandidateObservation,
    right: CandidateObservation,
) -> bool:
    if left.value is None or right.value is None:
        return left.value is not right.value
    if left.value.numeric != right.value.numeric:
        return True
    if left.value.comparator != right.value.comparator:
        return True
    if left.value.unit and right.value.unit and left.value.unit != right.value.unit:
        return True
    return bool(
        left.metric is not None
        and right.metric is not None
        and left.metric.unit
        and right.metric.unit
        and left.metric.unit != right.metric.unit
    )


def _physical_semantics_conflict(
    left: CandidateObservation,
    right: CandidateObservation,
) -> bool:
    return (
        left.claim_type != right.claim_type
        or left.reporting_status != right.reporting_status
        or _semantics_conflict(left, right)
        or _value_interpretation_conflicts(left, right)
    )


def _canonical_anchor(anchor: EvidenceAnchor) -> str:
    return json.dumps(anchor.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _table_value_requires_structural_identity(candidate: CandidateObservation) -> bool:
    return candidate.value is not None and any(
        anchor.kind is EvidenceKind.TABLE for anchor in candidate.evidence
    )


def _merge_physical_duplicates(
    group: list[CandidateObservation],
    identity: PhysicalCellIdentity,
) -> CandidateObservation:
    """Merge proposals for one exact cell while retaining all alternate information."""

    representative = _physical_representative(group)
    anchor_payloads = {
        _canonical_anchor(anchor): anchor for item in group for anchor in item.evidence
    }
    anchors = [anchor_payloads[key] for key in sorted(anchor_payloads)]
    notes = sorted({note for item in group for note in item.notes})
    snapshots = sorted(_proposal_snapshot(item) for item in group)
    notes.append("physical-cell source proposals: [" + ",".join(snapshots) + "]")
    identity_payload = json.dumps(asdict(identity), sort_keys=True, separators=(",", ":"))
    notes.append(f"merged {len(group)} proposals for physical cell {identity_payload}")

    conflict = any(
        _physical_semantics_conflict(left, right)
        for position, left in enumerate(group)
        for right in group[position + 1 :]
    )
    review_state_disagrees = len({item.export_status for item in group}) > 1
    update: dict[str, object] = {
        "observation_id": None,
        "evidence": anchors,
        "notes": notes,
        "extraction_confidence": max(item.extraction_confidence for item in group),
    }
    if conflict:
        update.update(
            {
                "referential_status": ReferentialStatus.WRONG_SCOPE,
                "export_status": ExportStatus.NEEDS_REVIEW,
                "export_reason": _PHYSICAL_CELL_CONFLICT_REASON,
            }
        )
        notes.append(_PHYSICAL_CELL_CONFLICT_NOTE)
    elif review_state_disagrees:
        update.update(
            {
                "export_status": ExportStatus.NEEDS_REVIEW,
                "export_reason": (
                    "physical-cell conflict: duplicate proposals disagree on review state"
                ),
            }
        )
        notes.append("semantic safety: physical duplicates disagree on review state")

    combined = representative.model_copy(update=update)
    combined.observation_id = combined.stable_id()
    return combined


def deduplicate_candidates(
    candidates: Iterable[CandidateObservation],
    layouts: dict[str, PdfLayout] | None = None,
) -> list[CandidateObservation]:
    """Merge duplicate proposals, preferring exact physical cells when available.

    Without layouts this retains the historical semantic-only behavior. With layouts,
    uniquely bound table values use their source geometry as identity. A repeated raw
    value that cannot be assigned to one cell is deliberately left unmerged.
    """

    materialized = list(candidates)
    if layouts is None:
        semantic_groups: dict[str, list[CandidateObservation]] = defaultdict(list)
        for candidate in materialized:
            semantic_groups[_semantic_key(candidate)].append(candidate)
        semantic_merged: list[CandidateObservation] = []
        for key in sorted(semantic_groups):
            group = semantic_groups[key]
            semantic_merged.append(
                group[0] if len(group) == 1 else _merge_semantic_duplicates(group)
            )
        return semantic_merged

    locator = PhysicalCellLocator(layouts)
    groups: dict[tuple[str, object], list[CandidateObservation]] = defaultdict(list)
    physical_identities: dict[tuple[str, object], PhysicalCellIdentity] = {}
    for candidate in materialized:
        binding = locator.bind(candidate)
        if binding.status is PhysicalCellBindingStatus.BOUND:
            assert binding.identity is not None
            key = ("physical", binding.identity)
            physical_identities[key] = binding.identity
        elif (
            binding.status is PhysicalCellBindingStatus.AMBIGUOUS
            or _table_value_requires_structural_identity(candidate)
        ):
            # Exact proposal copies contain no evidence of distinct printed cells and
            # must collapse so their derived observation IDs stay unique. Any
            # substantive difference keeps a separate key: ambiguous structure still
            # means abstain from semantic deduplication, and UNLOCATED table values do
            # not fall back to semantic identity either.
            key = ("unbound_table", _exact_candidate_key(candidate))
        else:
            key = ("semantic", _semantic_key(candidate))
        groups[key].append(candidate)

    merged: list[CandidateObservation] = []
    for key in sorted(groups, key=repr):
        group = groups[key]
        if len(group) == 1:
            merged.append(group[0])
            continue
        if key[0] == "physical":
            merged.append(_merge_physical_duplicates(group, physical_identities[key]))
        else:
            merged.append(_merge_semantic_duplicates(group))
    return merged
