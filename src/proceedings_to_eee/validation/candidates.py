"""Deterministic candidate verification and export gating."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum

from proceedings_to_eee.domain.attribution import OriginBasis, OriginExportPolicy
from proceedings_to_eee.domain.observation import CandidateObservation, EvidenceAnchor
from proceedings_to_eee.domain.provenance import (
    CandidateField,
    CandidateFieldProvenance,
    FieldBindingStatus,
    FieldSourceKind,
    FieldSourceRef,
)
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
from proceedings_to_eee.resolution.metrics import (
    metric_unit_is_compatible,
    registry_value_range_issue,
    resolve_metric_value,
)
from proceedings_to_eee.validation.field_provenance import (
    literal_setting_components_are_bound,
    quote_field_provenance,
)
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
_UNTRUSTED_REPLAY_SOURCE_KINDS = frozenset(
    {FieldSourceKind.EVIDENCE_QUOTE, FieldSourceKind.REVIEW_SPAN}
)


def normalize_evidence_text(value: str) -> str:
    """Normalize layout whitespace without altering numbers or punctuation."""

    value = value.replace("\u00a0", " ").replace("\u2212", "-")
    return re.sub(r"\s+", " ", value).strip()


def bounded_claim_present(text: str, claim: str) -> bool:
    """Match a literal claim without accepting a substring of another number.

    Textual claims retain ordinary literal containment. Claims containing digits use
    numeric-token boundaries so, for example, ``0.9`` cannot be grounded by ``10.90``
    and ``5%`` cannot be grounded by ``15%``. Decimal separators and explicit signs are
    treated as part of the adjacent numeric token.
    """

    if not claim:
        return True
    if not any(character.isdigit() for character in claim):
        return claim in text
    # Checking only the adjacent digit and decimal characters is not enough: the
    # mantissa ``5`` is not evidence for ``5e3``, nor is ``1`` evidence for
    # ``1×10^3``.  Inspect every literal occurrence and reject one that touches a
    # larger numeric token, including spaced scientific-notation continuations.
    scientific_suffix = re.compile(
        r"^(?:[eE]\s*[+\-−]?\s*\d|\s*[xX×]\s*10\s*(?:\^\s*)?[+\-−]?\s*\d)"
    )
    scientific_prefix = re.compile(r"\d(?:[\d.,]*\d)?[eE]\s*[+\-−]?$")
    start = 0
    while (index := text.find(claim, start)) >= 0:
        end = index + len(claim)
        left = text[:index]
        right = text[end:]
        left_neighbor = bool(
            left
            and (
                left[-1].isdigit()
                or left[-1] in "+-−^"
                or (left[-1] in ".," and len(left) > 1 and left[-2].isdigit())
            )
        )
        right_neighbor = bool(
            right
            and (
                right[0].isdigit()
                or right[0] == "^"
                or (right[0] in ".,+-−" and len(right) > 1 and right[1].isdigit())
            )
        )
        touches_numeric_neighbor = (
            left_neighbor
            or right_neighbor
            or bool(scientific_prefix.search(left))
            or bool(scientific_suffix.match(right))
        )
        if not touches_numeric_neighbor:
            return True
        start = index + 1
    return False


def verify_text_support(candidate: CandidateObservation, layouts: dict[str, PdfLayout]) -> None:
    """Check contiguous-line quotes and aggregate value support across all anchors.

    Table context and a physical result value often occupy different layout lines.  A
    candidate may therefore carry several exact anchors, but at least one of them must
    contain the asserted raw value.  Checking each page line independently prevents a
    model-created string from passing merely because whitespace normalization stitches
    adjacent source lines together.
    """

    quote_support: list[bool] = []
    value_support: list[bool] = []
    for anchor in candidate.evidence:
        layout = layouts.get(anchor.source_id)
        if layout is None or anchor.page > layout.page_count:
            quote_support.append(False)
            value_support.append(False)
            continue
        page = layout.pages[anchor.page - 1]
        # The public contract calls this an exact quote, so do not repair whitespace or
        # hyphenation here. A quote is supported only when its original bytes occur in one
        # source-native layout line on the cited page.
        quote = anchor.quote
        occurrences = sum(line.count(quote) for line in page.text.splitlines())
        quote_is_supported = bool(quote and quote.strip() and occurrences)
        quote_support.append(quote_is_supported)
        if not quote_is_supported:
            value_support.append(False)
            continue
        if candidate.value is not None:
            raw_value = candidate.value.raw
            value_support.append(bool(raw_value and bounded_claim_present(quote, raw_value)))
        else:
            value_support.append(True)
        if occurrences > 1:
            note = f"evidence quote occurs {occurrences} times on page {anchor.page}"
            if note not in candidate.notes:
                candidate.notes.append(note)

    all_quotes_supported = bool(quote_support) and all(quote_support)
    any_quote_supported = any(quote_support)
    has_value_support = candidate.value is None or any(value_support)
    if all_quotes_supported and has_value_support:
        candidate.text_support = TextSupportStatus.SUPPORTED
    elif any_quote_supported:
        candidate.text_support = TextSupportStatus.PARTIALLY_SUPPORTED
    else:
        candidate.text_support = TextSupportStatus.UNSUPPORTED


def _resolution_reason(note: str) -> str:
    return "deterministic_reference_resolution=" + re.sub(
        r"[^a-z0-9]+", "_", note.casefold()
    ).strip("_")


def _deterministic_quote_derivations(
    candidate: CandidateObservation,
) -> dict[CandidateField, str]:
    """Reconstruct registry derivations from quote-supported primitive inputs.

    This reconstruction deliberately starts without provider-supplied semantic metadata.
    It therefore remains stable on validation replay and cannot bless a provider value
    merely because an earlier validation pass already copied it into the candidate.
    """

    if candidate.metric is None or candidate.value is None:
        return {}
    primitive_metric = candidate.metric.model_copy(
        update={
            "canonical_id": None,
            "kind": None,
            "unit": None,
            "lower_is_better": None,
            "min_score": None,
            "max_score": None,
        }
    )
    primitive_value = candidate.value.model_copy(update={"unit": None})
    inferred_metric, inferred_value, note = resolve_metric_value(
        primitive_metric,
        primitive_value,
        (anchor.quote for anchor in candidate.evidence),
    )
    reasons: dict[CandidateField, str] = {}
    metric_fields = ("canonical_id", "kind", "lower_is_better", "min_score", "max_score")
    if any(getattr(inferred_metric, name) is not None for name in metric_fields) and all(
        getattr(candidate.metric, name) == getattr(inferred_metric, name) for name in metric_fields
    ):
        reasons[CandidateField.METRIC] = "deterministic_reference_resolution=metric_registry"
    if (
        note
        and candidate.metric.unit == inferred_metric.unit
        and candidate.value.unit == inferred_value.unit
    ):
        reasons[CandidateField.UNIT] = _resolution_reason(note)
    return reasons


def _canonical_field_sources(sources: Iterable[FieldSourceRef]) -> list[FieldSourceRef]:
    """Return deterministic, duplicate-free field sources without changing their type."""

    by_identity = {
        json.dumps(
            source.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ): source
        for source in sources
    }
    return [by_identity[key] for key in sorted(by_identity)]


def _refresh_resolved_field_provenance(
    candidate: CandidateObservation,
    provenance: list[CandidateFieldProvenance],
    *,
    previous_hashes: dict[CandidateField, str],
    resolution_note: str | None,
) -> list[CandidateFieldProvenance]:
    """Refresh quote bindings per field while retaining independent structure.

    A candidate can mix row/header bindings with legacy quote bindings.  A structural
    source on one field must never exempt quote-only bindings on other fields from fresh
    support checks.  Existing quote sources are therefore discarded and regenerated;
    only nonquote sources and their conservative status survive from the prior binding.
    """

    hashes = candidate.field_value_sha256s()
    quote_bindings = {
        item.field: item
        for item in quote_field_provenance(
            candidate,
            derived_reasons=_deterministic_quote_derivations(candidate),
        )
    }
    prior_bindings = {item.field: item for item in provenance}
    refreshed: dict[CandidateField, CandidateFieldProvenance] = {}

    metric_changed = hashes[CandidateField.METRIC] != previous_hashes[CandidateField.METRIC]
    unit_changed = hashes[CandidateField.UNIT] != previous_hashes[CandidateField.UNIT]

    for field in CandidateField:
        prior = prior_bindings[field]
        quote = quote_bindings[field]
        nonquote_sources = [
            source for source in prior.sources if source.kind not in _UNTRUSTED_REPLAY_SOURCE_KINDS
        ]
        fresh_quote_sources = list(quote.sources)

        if prior.status is FieldBindingStatus.CONFLICT:
            # Reference normalization must never silently adjudicate a pre-existing
            # disagreement.  If the resolved value equals an old alternate, the old
            # primary becomes the alternate so the conflict remains representable.
            alternates = sorted(
                {
                    *prior.alternate_value_sha256s,
                    prior.value_sha256,
                }
                - {hashes[field]}
            )
            conflict_sources = _canonical_field_sources([*nonquote_sources, *fresh_quote_sources])
            if not conflict_sources:
                # Conflict sources identify the disagreement rather than claim current
                # field support.  Retain them only when no refreshed/structural pointer
                # exists, because the provenance model requires a conflict source.
                conflict_sources = [
                    source
                    for source in prior.sources
                    if source.kind is FieldSourceKind.EVIDENCE_QUOTE
                ]
            if not conflict_sources:
                raise ValueError(
                    "conflicting field provenance has no admissible source after "
                    "review-span rejection"
                )
            refreshed[field] = prior.model_copy(
                update={
                    "value_sha256": hashes[field],
                    "status": FieldBindingStatus.CONFLICT,
                    "sources": conflict_sources,
                    "alternate_value_sha256s": alternates,
                }
            )
            continue

        if field is CandidateField.SETTING and prior.value_sha256 != previous_hashes[field]:
            # The only accepted stale SETTING hash is the sealed pre-role projection.
            # Its structural sources cannot attest roles that did not exist in that
            # projection, so re-prove the complete current setting from exact quotes.
            refreshed[field] = quote
            continue

        if nonquote_sources:
            # The prior status belongs to independently verified structure for this
            # field.  Fresh quote sources may supplement it, but can neither cure nor
            # weaken that structural status.
            refreshed[field] = prior.model_copy(
                update={
                    "value_sha256": hashes[field],
                    "sources": _canonical_field_sources([*nonquote_sources, *fresh_quote_sources]),
                    "alternate_value_sha256s": [],
                }
            )
        else:
            # Quote-only (including empty/unsupported) bindings are replaced outright;
            # no provider-supplied quote status or source survives validation replay.
            refreshed[field] = quote

    metric_binding = refreshed[CandidateField.METRIC]
    if (
        metric_changed
        and metric_binding.status is FieldBindingStatus.BOUND
        and any(
            source.kind not in _UNTRUSTED_REPLAY_SOURCE_KINDS for source in metric_binding.sources
        )
    ):
        refreshed[CandidateField.METRIC] = metric_binding.model_copy(
            update={"reason": "deterministic_reference_resolution=metric_registry"}
        )

    unit_binding = refreshed[CandidateField.UNIT]
    if unit_changed and unit_binding.status is not FieldBindingStatus.CONFLICT:
        metric_binding = refreshed[CandidateField.METRIC]
        value_binding = refreshed[CandidateField.VALUE]
        derivation_sources = _canonical_field_sources(
            [*metric_binding.sources, *value_binding.sources]
        )
        if (
            resolution_note
            and metric_binding.status is FieldBindingStatus.BOUND
            and value_binding.status is FieldBindingStatus.BOUND
            and derivation_sources
        ):
            refreshed[CandidateField.UNIT] = unit_binding.model_copy(
                update={
                    "status": FieldBindingStatus.BOUND,
                    "sources": derivation_sources,
                    "reason": _resolution_reason(resolution_note),
                }
            )
        else:
            refreshed[CandidateField.UNIT] = unit_binding.model_copy(
                update={
                    "status": FieldBindingStatus.UNSUPPORTED,
                    "sources": [],
                    "reason": "reference_resolution_changed_unbound_unit",
                }
            )

    unit_binding = refreshed[CandidateField.UNIT]
    if (
        candidate.metric is not None
        and candidate.metric.unit is not None
        and not metric_unit_is_compatible(candidate.metric, candidate.metric.unit)
        and unit_binding.status is not FieldBindingStatus.CONFLICT
    ):
        refreshed[CandidateField.UNIT] = unit_binding.model_copy(
            update={
                "status": FieldBindingStatus.UNSUPPORTED,
                "alternate_value_sha256s": [],
                "reason": "incompatible_unit_for_registry_bounded_metric",
            }
        )

    if candidate.metric is not None and candidate.value is not None:
        range_issue = registry_value_range_issue(candidate.metric, candidate.value)
        value_binding = refreshed[CandidateField.VALUE]
        if range_issue and value_binding.status is not FieldBindingStatus.CONFLICT:
            refreshed[CandidateField.VALUE] = value_binding.model_copy(
                update={
                    "status": FieldBindingStatus.UNSUPPORTED,
                    "alternate_value_sha256s": [],
                    "reason": range_issue,
                }
            )

    return [refreshed[field] for field in CandidateField]


def resolve_references(candidate: CandidateObservation) -> None:
    """Resolve only facts backed by the typed candidate or the explicit metric registry."""

    if candidate.metric and candidate.value:
        previous_hashes = candidate.field_value_sha256s()
        metric, value, resolution_note = resolve_metric_value(
            candidate.metric,
            candidate.value,
            (anchor.quote for anchor in candidate.evidence),
        )
        provenance = candidate.field_provenance
        if provenance:
            candidate.field_provenance = []
        candidate.metric = metric
        candidate.value = value
        if provenance:
            candidate.field_provenance = _refresh_resolved_field_provenance(
                candidate,
                provenance,
                previous_hashes=previous_hashes,
                resolution_note=resolution_note,
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
    if not metric_unit_is_compatible(candidate.metric, candidate.metric.unit):
        candidate.referential_status = ReferentialStatus.UNRESOLVED
        return
    if registry_value_range_issue(candidate.metric, candidate.value):
        candidate.referential_status = ReferentialStatus.UNRESOLVED
        return
    candidate.referential_status = ReferentialStatus.RESOLVED


def _provenance_item_gates(candidate: CandidateObservation, item: CandidateFieldProvenance) -> bool:
    """Free-text descriptive setting components never gate export.

    `construct`, `operationalization`, `decision_rule` and `evaluation_date` are prose a
    paper writes about its measurement, not literals of it. Requiring them verbatim in
    the result quote blocked candidates whose split, subset, language, sample count,
    aggregation, raw scope, metric parameters and non-evaluated roles were all bound. A
    merge conflict still gates, because that is a real disagreement rather than absent
    prose, and the binding status still records description as unbound.
    """

    if item.field is not CandidateField.SETTING:
        return True
    if item.status is FieldBindingStatus.CONFLICT:
        return True
    return not literal_setting_components_are_bound(candidate)


def apply_export_policy(candidate: CandidateObservation, min_confidence: float = 0.8) -> None:
    """Apply a conservative, explainable EEE eligibility gate."""

    populated_fields = candidate.populated_fields()

    if candidate.claim_type != ClaimType.PRIMARY_RESULT:
        candidate.export_status = ExportStatus.NOT_ELIGIBLE
        candidate.export_reason = f"claim_type={candidate.claim_type} is not a primary result"
    elif candidate.text_support != TextSupportStatus.SUPPORTED:
        candidate.export_status = ExportStatus.NEEDS_REVIEW
        candidate.export_reason = f"text_support={candidate.text_support}"
    elif candidate.referential_status != ReferentialStatus.RESOLVED:
        candidate.export_status = ExportStatus.NEEDS_REVIEW
        candidate.export_reason = f"referential_status={candidate.referential_status}"
    elif candidate.schema_version == "candidate-observation/0.3" and not candidate.field_provenance:
        candidate.export_status = ExportStatus.NEEDS_REVIEW
        candidate.export_reason = "field_provenance=missing"
    elif candidate.schema_version == "candidate-observation/0.3" and any(
        item.field in populated_fields and item.status is FieldBindingStatus.CONFLICT
        for item in candidate.field_provenance
    ):
        candidate.export_status = ExportStatus.NEEDS_REVIEW
        candidate.export_reason = "field_provenance=conflict"
    elif candidate.schema_version == "candidate-observation/0.3" and any(
        item.field in populated_fields
        and item.status is not FieldBindingStatus.BOUND
        and _provenance_item_gates(candidate, item)
        for item in candidate.field_provenance
    ):
        candidate.export_status = ExportStatus.NEEDS_REVIEW
        candidate.export_reason = "field_provenance=ambiguous_or_unsupported"
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


def _ambiguity_value_key(candidate: CandidateObservation) -> str | None:
    if candidate.value is None:
        return None
    return normalize_evidence_text(candidate.value.raw).casefold()


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
    if any(
        left_fields[name]
        and right_fields[name]
        and not _compatible_semantic_name(left_fields[name], right_fields[name])
        for name in left_fields
    ):
        return True
    if left.value is None or right.value is None:
        return left.value is not right.value
    return left.value.model_dump(mode="json") != right.value.model_dump(mode="json")


def _absolute_raw_value_offsets(
    anchor: EvidenceAnchor,
    raw_value: str,
    layouts: dict[str, PdfLayout],
) -> set[int]:
    """Resolve exact quote-relative value offsets into immutable page coordinates."""

    layout = layouts.get(anchor.source_id)
    if layout is None or anchor.page > layout.page_count or anchor.quote.count(raw_value) != 1:
        return set()
    relative = anchor.quote.index(raw_value)
    page_text = layout.pages[anchor.page - 1].text
    offsets: set[int] = set()
    start = 0
    while (quote_start := page_text.find(anchor.quote, start)) >= 0:
        offsets.add(quote_start + relative)
        start = quote_start + 1
    return offsets


def _anchor_structure_compatible(
    left: EvidenceAnchor,
    right: EvidenceAnchor,
    *,
    raw_value: str,
    layouts: dict[str, PdfLayout],
) -> bool:
    left_quote = normalize_evidence_text(left.quote).casefold()
    right_quote = normalize_evidence_text(right.quote).casefold()
    if not left_quote or not right_quote:
        return False
    left_offsets = _absolute_raw_value_offsets(left, raw_value, layouts)
    right_offsets = _absolute_raw_value_offsets(right, raw_value, layouts)
    if left_offsets and right_offsets and left_offsets & right_offsets:
        # Frozen page coordinates, not proposal-supplied labels, define one printed
        # occurrence even when two exact spans have different widths.
        return True
    if left_quote == right_quote:
        # Descriptive anchor metadata is proposal-supplied.  It cannot split two
        # proposals grounded in the same immutable source/page/exact occurrence.
        return True
    return left_quote in right_quote or right_quote in left_quote


def _candidates_share_ambiguous_evidence(
    left: CandidateObservation,
    right: CandidateObservation,
    layouts: dict[str, PdfLayout],
) -> bool:
    if left.value is None or right.value is None:
        return False
    raw_value = left.value.raw
    if normalize_evidence_text(raw_value).casefold() != (
        normalize_evidence_text(right.value.raw).casefold()
    ):
        return False
    return any(
        left_anchor.source_id == right_anchor.source_id
        and left_anchor.page == right_anchor.page
        and _anchor_structure_compatible(
            left_anchor,
            right_anchor,
            raw_value=raw_value,
            layouts=layouts,
        )
        for left_anchor in left.evidence
        for right_anchor in right.evidence
    )


def _route_ambiguous_evidence(
    candidates: list[CandidateObservation],
    layouts: dict[str, PdfLayout],
) -> None:
    """Abstain when the same quoted value receives incompatible semantics.

    Overlapping extraction blocks can propose several meanings for one printed
    value. The resolver must not silently choose one. Every conflicting proposal
    remains in the review artifact, while none is eligible for EEE export.
    """

    groups: dict[tuple[str, str], list[CandidateObservation]] = defaultdict(list)
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
                if not _candidates_share_ambiguous_evidence(left, right, layouts):
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
    candidates: list[CandidateObservation],
    layouts: dict[str, PdfLayout],
    origin_policy: OriginExportPolicy = OriginExportPolicy.POSITIVE_ONLY,
) -> None:
    """Attach a deterministic attribution verdict and demote what it places elsewhere.

    Under the default `positive_only` policy this is demote-only: only a positive
    PAPER_PRODUCED verdict may remain in the export gate, and the current resolver never
    produces that verdict. Routing is to NEEDS_REVIEW rather than NOT_ELIGIBLE because
    NOT_ELIGIBLE carries no risk reason in the human review lane and would silently drop
    the candidate, and because false_primary_export keys on {ELIGIBLE, EXPORTED} so
    NEEDS_REVIEW already clears that gate.

    Under `tiered` the same verdict is attached and the same candidates are demoted,
    except that a basis the policy permits survives. No candidate is ever labelled
    `paper_produced` by this: the verdict is untouched, and what changes is only which
    bases a composition run is willing to export, which every record then carries.
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
        if origin_policy.permits(verdict.origin_basis(candidate.claim_type)):
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


def _validate_non_origin_candidates_in_place(
    candidates: Iterable[CandidateObservation],
    layouts: dict[str, PdfLayout],
    min_confidence: float = 0.8,
    processed_source_ids: set[str] | None = None,
) -> list[CandidateObservation]:
    """Run every deterministic candidate gate except producer attribution."""

    processed = set(layouts) if processed_source_ids is None else processed_source_ids
    validated: list[CandidateObservation] = []
    for candidate in candidates:
        verify_text_support(candidate, layouts)
        resolve_references(candidate)
        apply_export_policy(candidate, min_confidence=min_confidence)
        _route_semantic_safety_issues(candidate)
        candidate.observation_id = candidate.stable_id()
        validated.append(candidate)
    _route_ambiguous_evidence(validated, layouts)
    for candidate in validated:
        _restore_physical_cell_conflict(candidate)
        if any(anchor.source_id not in processed for anchor in candidate.evidence):
            candidate.export_status = ExportStatus.NEEDS_REVIEW
            candidate.export_reason = "source_not_processed"
    return validated


def validate_non_origin_candidates(
    candidates: Iterable[CandidateObservation],
    layouts: dict[str, PdfLayout],
    min_confidence: float = 0.8,
    processed_source_ids: set[str] | None = None,
) -> list[CandidateObservation]:
    """Recompute all non-origin gates on deep copies of the supplied candidates.

    The returned observations expose whether result evidence and tuple semantics pass
    the export gate before attribution is considered. Inputs are never changed, which
    makes this entry point safe for replaying validation over a frozen observation
    ledger. Existing attribution fields are carried through but are neither recomputed
    nor used to change eligibility.
    """

    copies = [candidate.model_copy(deep=True) for candidate in candidates]
    return _validate_non_origin_candidates_in_place(
        copies,
        layouts,
        min_confidence=min_confidence,
        processed_source_ids=processed_source_ids,
    )


def route_candidate_attribution(
    candidates: list[CandidateObservation],
    layouts: dict[str, PdfLayout],
    origin_policy: OriginExportPolicy = OriginExportPolicy.POSITIVE_ONLY,
) -> list[CandidateObservation]:
    """Attach conservative producer-origin verdicts after independent checks.

    This step is intentionally in-place and demote-only. Keeping it separate from
    non-origin validation lets an independent verifier inspect candidates that pass
    the result, tuple, scope, confidence, and conflict gates before unresolved origin
    routes them to review.
    """

    _route_attribution(candidates, layouts, origin_policy)
    return candidates


def validate_candidates(
    candidates: Iterable[CandidateObservation],
    layouts: dict[str, PdfLayout],
    min_confidence: float = 0.8,
    processed_source_ids: set[str] | None = None,
    origin_policy: OriginExportPolicy = OriginExportPolicy.POSITIVE_ONLY,
) -> list[CandidateObservation]:
    """Run all candidate gates in place, including conservative attribution routing."""

    validated = _validate_non_origin_candidates_in_place(
        candidates,
        layouts,
        min_confidence=min_confidence,
        processed_source_ids=processed_source_ids,
    )
    _route_attribution(validated, layouts, origin_policy)
    return validated


def candidate_origin_basis(candidate: CandidateObservation) -> OriginBasis:
    """Return the origin basis a composed record would have to carry for this candidate.

    Deliberately derived rather than stored: the candidate payload is what a model
    proposes, and a basis written into it could be mistaken for something the model
    established. Composition records it in the provenance and in the record instead.
    """

    if candidate.attribution is None:
        return OriginBasis.NONE
    return candidate.attribution.origin_basis(candidate.claim_type)


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
        "field_values": candidate.field_value_sha256s(),
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
    traces = sorted(
        (trace for item in group for trace in item.proposal_traces),
        key=lambda trace: trace.proposal_id,
    )
    combined = first.model_copy(
        update={
            "observation_id": None,
            "evidence": list(anchors.values()),
            "notes": notes,
            "extraction_confidence": max(item.extraction_confidence for item in group),
            "proposal_traces": traces,
            "field_provenance": _merge_matching_field_provenance(group),
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
    payload.pop("proposal_traces", None)
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
    payload.pop("proposal_traces", None)
    payload.pop("raw_payload_hash", None)
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


def _merge_matching_field_provenance(
    group: list[CandidateObservation],
    representative: CandidateObservation | None = None,
) -> list[CandidateFieldProvenance]:
    """Union source bindings when duplicate candidates make the same field claims."""

    if not all(candidate.field_provenance for candidate in group):
        return []
    merged: list[CandidateFieldProvenance] = []
    for field in CandidateField:
        items = [
            next(item for item in candidate.field_provenance if item.field is field)
            for candidate in group
        ]
        primary = next(
            item for item in (representative or group[0]).field_provenance if item.field is field
        )
        value_hashes = {
            value_hash
            for item in items
            for value_hash in (item.value_sha256, *item.alternate_value_sha256s)
        }
        sources = {
            json.dumps(
                source.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ): source
            for item in items
            for source in item.sources
        }
        statuses = {item.status for item in items}
        if len(value_hashes) > 1 or FieldBindingStatus.CONFLICT in statuses:
            status = FieldBindingStatus.CONFLICT
        elif FieldBindingStatus.AMBIGUOUS in statuses:
            status = FieldBindingStatus.AMBIGUOUS
        elif FieldBindingStatus.BOUND in statuses:
            status = FieldBindingStatus.BOUND
        else:
            status = FieldBindingStatus.UNSUPPORTED
        reason = (
            "deduplicated proposals conflict on this field"
            if len(value_hashes) > 1
            else next((item.reason for item in items if item.reason), None)
        )
        bound_sources = [sources[key] for key in sorted(sources)]
        if not bound_sources and status is not FieldBindingStatus.UNSUPPORTED:
            # Two proposals can disagree on a field that neither of them supports: the
            # values differ, the quotes bind nothing, and there is no source to attach a
            # conflict to. `CandidateFieldProvenance` refuses that shape, and rightly,
            # because BOUND, AMBIGUOUS and CONFLICT all assert evidence.
            # The disagreement is not lost: it is recorded in the reason.
            status = FieldBindingStatus.UNSUPPORTED
            reason = (
                "deduplicated proposals disagree on this field and no quote supports any of them"
                if len(value_hashes) > 1
                else reason
            )
        merged.append(
            CandidateFieldProvenance(
                field=field,
                value_sha256=primary.value_sha256,
                status=status,
                sources=bound_sources,
                alternate_value_sha256s=(
                    sorted(value_hashes - {primary.value_sha256})
                    if status is FieldBindingStatus.CONFLICT
                    else []
                ),
                reason=reason,
            )
        )
    return merged


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
        "proposal_traces": sorted(
            (trace for item in group for trace in item.proposal_traces),
            key=lambda trace: trace.proposal_id,
        ),
        "field_provenance": _merge_matching_field_provenance(group, representative),
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


class DeduplicationKind(StrEnum):
    SINGLETON = "singleton"
    SEMANTIC = "semantic"
    PHYSICAL_CELL = "physical_cell"


@dataclass(frozen=True)
class DeduplicationResult:
    candidates: list[CandidateObservation]
    kind_by_observation_id: dict[str, DeduplicationKind]


def deduplicate_candidates_with_lineage(
    candidates: Iterable[CandidateObservation],
    layouts: dict[str, PdfLayout] | None = None,
) -> DeduplicationResult:
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
        kinds = {
            str(candidate.observation_id): (
                DeduplicationKind.SEMANTIC
                if len(semantic_groups[_semantic_key(candidate)]) > 1
                else DeduplicationKind.SINGLETON
            )
            for candidate in semantic_merged
        }
        semantic_merged, kinds = _collapse_residual_id_collisions(semantic_merged, kinds)
        return DeduplicationResult(semantic_merged, kinds)

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
    kinds: dict[str, DeduplicationKind] = {}
    for key in sorted(groups, key=repr):
        group = groups[key]
        if len(group) == 1:
            candidate = group[0]
            merged.append(candidate)
            kinds[str(candidate.observation_id)] = DeduplicationKind.SINGLETON
            continue
        if key[0] == "physical":
            candidate = _merge_physical_duplicates(group, physical_identities[key])
            kind = DeduplicationKind.PHYSICAL_CELL
        else:
            candidate = _merge_semantic_duplicates(group)
            kind = DeduplicationKind.SEMANTIC
        merged.append(candidate)
        kinds[str(candidate.observation_id)] = kind
    merged, kinds = _collapse_residual_id_collisions(merged, kinds)
    return DeduplicationResult(merged, kinds)


_MAX_ID_COLLISION_PASSES = 4
"""Bound on the re-stamp passes below. Re-stamping is idempotent, so this converges."""

_DESCRIPTIVE_COLLAPSE_FIELDS = (
    "evaluation_construct",
    "operationalization",
    "decision_rule",
    "evaluation_date",
)
"""Free-text fields outside the `stable_id()` payload; two candidates can share one
identity and still disagree here, so a collapse records what it discards."""


def _collapse_residual_id_collisions(
    merged: list[CandidateObservation],
    kinds: dict[str, DeduplicationKind],
) -> tuple[list[CandidateObservation], dict[str, DeduplicationKind]]:
    """Guarantee the invariant every downstream stage assumes: unique observation IDs.

    Grouping alone cannot guarantee it. `_semantic_key` includes the evidence quote
    hashes, while `stable_id` deliberately excludes quote text, so two candidates whose
    only difference is their quotation are kept apart here and then collide downstream.
    For example, two `method_metadata` candidates with null value, metric and scope,
    no roles and the same unlabeled page anchor can share an ID despite different
    quotations. Unresolved duplicates cause `build_candidate_lineage` to reject the
    artifact.

    By the project's own identity rule those two are one observation, so they are merged,
    which unions their evidence anchors and proposal traces rather than discarding either
    quotation. The alternative, widening `stable_id`, would move every observation ID in
    the project and void comparability with the sealed runs.
    """

    for _ in range(_MAX_ID_COLLISION_PASSES):
        by_id: dict[str, list[CandidateObservation]] = defaultdict(list)
        for candidate in merged:
            by_id[str(candidate.observation_id)].append(candidate)
        if all(len(group) == 1 for group in by_id.values()):
            return merged, kinds
        merged, kinds = _resolve_collision_pass(by_id, kinds)
    return merged, kinds


def _resolve_collision_pass(
    by_id: dict[str, list[CandidateObservation]],
    kinds: dict[str, DeduplicationKind],
) -> tuple[list[CandidateObservation], dict[str, DeduplicationKind]]:
    """Resolve every colliding ID once. Re-stamping can create a new collision, so the
    caller repeats this until the IDs are unique; re-stamping is idempotent, so it
    converges."""

    collapsed: list[CandidateObservation] = []
    resolved: dict[str, DeduplicationKind] = {}
    for observation_id in sorted(by_id):
        group = by_id[observation_id]
        if len(group) == 1:
            candidate = group[0]
            collapsed.append(candidate)
            resolved[str(candidate.observation_id)] = kinds.get(
                observation_id, DeduplicationKind.SINGLETON
            )
            continue
        previous = kinds.get(observation_id, DeduplicationKind.SINGLETON)
        for candidate, was_collapsed in _resolve_one_identity(group):
            collapsed.append(candidate)
            resolved[str(candidate.observation_id)] = (
                DeduplicationKind.SEMANTIC if was_collapsed else previous
            )
    return collapsed, resolved


def _resolve_one_identity(
    group: list[CandidateObservation],
) -> list[tuple[CandidateObservation, bool]]:
    """Resolve one shared ID without ever merging observations that differ.

    Members that disagree on anything `stable_id()` covers, which includes value, metric,
    scope, roles, claim type and the evidence anchors, are not one observation: the shared
    ID is stale, so each is re-stamped with its own `stable_id()` and kept. Only members
    whose semantic identity really is the same are collapsed. Returns each surviving
    candidate with whether it is the product of a collapse.
    """

    by_identity: dict[str, list[CandidateObservation]] = defaultdict(list)
    for candidate in group:
        by_identity[candidate.stable_id()].append(candidate)
    resolved: list[tuple[CandidateObservation, bool]] = []
    for identity in sorted(by_identity):
        members = by_identity[identity]
        if len(members) == 1:
            member = members[0]
            if str(member.observation_id) != identity:
                member = member.model_copy(update={"observation_id": identity})
            resolved.append((member, False))
            continue
        resolved.append((_collapse_one_identity(members), True))
    return resolved


def _collapse_one_identity(group: list[CandidateObservation]) -> CandidateObservation:
    """Merge candidates that share an observation ID, without inventing provenance.

    `_merge_semantic_duplicates` cannot be reused here. It unions field provenance, and
    when two members disagree on a field's value hash it raises the status to CONFLICT
    while their source sets are empty, which `CandidateFieldProvenance` rejects because a
    conflict needs a source binding. That is exactly the shape of a collision: the members
    differ only in quote-derived provenance for fields that are themselves null.

    Field provenance is therefore kept only where the whole group already agrees, and
    dropped otherwise with an explicit note. Every member shares one `stable_id()`, which
    `_resolve_one_identity` guarantees, so value, metric, scope, roles and anchors already
    agree and no field value is lost. What can still differ are the descriptive fields
    outside that payload, and each discarded variant is written into the notes rather than
    dropped silently. Unioning the anchors changes the payload, so the merged candidate is
    re-stamped with its own `stable_id()`.
    """

    ordered = sorted(
        group,
        key=lambda candidate: (
            candidate.text_support is not TextSupportStatus.SUPPORTED,
            str(candidate.observation_id),
        ),
    )
    representative = ordered[0]
    anchors = {anchor.quote_sha256: anchor for item in ordered for anchor in item.evidence}
    notes = list(dict.fromkeys(note for item in ordered for note in item.notes))
    notes.append(f"collapsed {len(ordered)} proposals sharing one observation id")
    for field in _DESCRIPTIVE_COLLAPSE_FIELDS:
        kept = getattr(representative, field)
        discarded = [
            value
            for value in dict.fromkeys(getattr(item, field) for item in ordered)
            if value is not None and value != kept
        ]
        if discarded:
            variants = ", ".join(repr(value) for value in discarded)
            notes.append(
                f"collapsed proposals disagreed on {field}: kept {kept!r}, dropped {variants}"
            )
    provenance = representative.field_provenance
    if any(item.field_provenance != provenance for item in ordered):
        provenance = []
        notes.append("field provenance dropped: collapsed proposals disagreed")
    traces = sorted(
        {trace.proposal_id: trace for item in ordered for trace in item.proposal_traces}.values(),
        key=lambda trace: trace.proposal_id,
    )
    combined = representative.model_copy(
        update={
            "observation_id": None,
            "evidence": [anchors[key] for key in sorted(anchors)],
            "notes": notes,
            "field_provenance": provenance,
            "proposal_traces": traces,
        }
    )
    combined.observation_id = combined.stable_id()
    return combined


def deduplicate_candidates(
    candidates: Iterable[CandidateObservation],
    layouts: dict[str, PdfLayout] | None = None,
) -> list[CandidateObservation]:
    """Compatibility wrapper returning only the deduplicated candidates."""

    return deduplicate_candidates_with_lineage(candidates, layouts).candidates
