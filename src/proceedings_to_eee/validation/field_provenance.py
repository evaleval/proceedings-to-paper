"""Deterministic per-field provenance for legacy and row proposals."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import product

from proceedings_to_eee.domain.observation import (
    CandidateObservation,
    EvidenceAnchor,
    ReportedValue,
    Uncertainty,
)
from proceedings_to_eee.domain.provenance import (
    CandidateField,
    CandidateFieldProvenance,
    FieldBindingStatus,
    FieldSourceKind,
    FieldSourceRef,
)
from proceedings_to_eee.domain.status import ActorRole, EvidenceKind, ValueComparator
from proceedings_to_eee.extraction.row_enumeration import (
    EnumerationHeaderCell,
    EnumerationRow,
    EnumerationValue,
    HeaderBindingState,
)
from proceedings_to_eee.resolution.metrics import resolve_metric_value


def _normalized(value: str | None) -> str:
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", value).replace("\u2212", "-")
    return re.sub(r"[^a-z0-9%]+", " ", value.casefold()).strip()


def _compatible(left: str | None, right: str | None) -> bool:
    first = _normalized(left)
    second = _normalized(right)
    if not first or not second:
        return False
    left_tokens = set(first.split())
    right_tokens = set(second.split())
    return first == second or left_tokens <= right_tokens or right_tokens <= left_tokens


_NUMBER = re.compile(
    r"(?<![\w.])[-+\N{MINUS SIGN}]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)"
    r"(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?(?![\w.])"
)


def _phrase_present(quote: str, claim: str) -> bool:
    expected = _normalized(claim)
    actual = _normalized(quote)
    return bool(expected) and f" {expected} " in f" {actual} "


def _bounded_literal_present(quote: str, claim: str) -> bool:
    """Match digit-bearing claims without erasing signs or decimal punctuation."""

    text = unicodedata.normalize("NFKC", quote).replace("\N{MINUS SIGN}", "-").casefold()
    expected = unicodedata.normalize("NFKC", claim).replace("\N{MINUS SIGN}", "-").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    expected = re.sub(r"\s+", " ", expected).strip()
    if not expected:
        return False
    pattern = re.compile(rf"(?<![\w.,]){re.escape(expected)}(?![\w.,])")
    return bool(pattern.search(text))


def _number_present(quote: str, claim: int | float) -> bool:
    try:
        expected = Decimal(str(claim))
    except InvalidOperation:
        return False
    for match in _NUMBER.finditer(quote):
        token = match.group().replace("\N{MINUS SIGN}", "-").replace(",", "")
        try:
            if Decimal(token) == expected:
                return True
        except InvalidOperation:
            continue
    return False


def _claim_present(quote: str, claim: object) -> bool:
    if isinstance(claim, bool):
        return _phrase_present(quote, str(claim))
    if isinstance(claim, int | float):
        return _number_present(quote, claim)
    if not isinstance(claim, str):
        return False
    if any(character.isdigit() for character in claim):
        return _bounded_literal_present(quote, claim)
    return bool(_claim_spans(quote, claim))


_CLAUSE_SEPARATOR = re.compile(r"(?:[\r\n]+|[;；]+|\s+\|\s+)")


def _quote_clauses(quote: str) -> list[str]:
    """Split result prose at boundaries that cannot express one atomic tuple."""

    return [part.strip() for part in _CLAUSE_SEPARATOR.split(quote) if part.strip()]


def _source_for_anchor(anchor: EvidenceAnchor) -> FieldSourceRef:
    return FieldSourceRef(
        kind=FieldSourceKind.EVIDENCE_QUOTE,
        source_id=anchor.source_id,
        page=anchor.page,
        region_id=anchor.region_id,
        planned_row_id=anchor.planned_row_id,
        physical_cell_id=anchor.cell_id,
        numeric_token_id=anchor.numeric_token_id,
        header_ids=anchor.header_ids,
        quote_sha256=anchor.quote_sha256,
    )


def _canonical_sources(sources: Iterable[FieldSourceRef]) -> list[FieldSourceRef]:
    by_identity = {
        json.dumps(source.model_dump(mode="json"), sort_keys=True, separators=(",", ":")): source
        for source in sources
    }
    return [by_identity[key] for key in sorted(by_identity)]


def _claim_sources(candidate: CandidateObservation, claim: object) -> list[FieldSourceRef]:
    return _canonical_sources(
        _source_for_anchor(anchor)
        for anchor in candidate.evidence
        if _claim_present(anchor.quote, claim)
    )


def _claims_binding(
    candidate: CandidateObservation,
    claims: Iterable[object],
) -> tuple[FieldBindingStatus, list[FieldSourceRef], str | None]:
    material = [claim for claim in claims if claim is not None and claim != ""]
    supported = [(claim, _claim_sources(candidate, claim)) for claim in material]
    sources = _canonical_sources(source for _, matches in supported for source in matches)
    supported_count = sum(bool(matches) for _, matches in supported)
    if material and supported_count == len(material):
        return FieldBindingStatus.BOUND, sources, None
    if supported_count:
        return (
            FieldBindingStatus.AMBIGUOUS,
            sources,
            "field_partially_supported_by_evidence_quotes",
        )
    return FieldBindingStatus.UNSUPPORTED, [], "field_not_supported_by_evidence_quotes"


def _identity_claims_binding(
    candidate: CandidateObservation,
    identity_claims: Iterable[object],
    supporting_claims: Iterable[tuple[str, object]] = (),
) -> tuple[FieldBindingStatus, list[FieldSourceRef], str | None]:
    """Require the load-bearing field identity before accepting partial support."""

    identities = [claim for claim in identity_claims if claim is not None and claim != ""]
    identity_status, identity_sources, _ = _claims_binding(candidate, identities)
    if not identities or identity_status is not FieldBindingStatus.BOUND:
        return FieldBindingStatus.UNSUPPORTED, [], "field_identity_not_supported_by_evidence_quotes"
    extras = [
        (label, claim) for label, claim in supporting_claims if claim is not None and claim != ""
    ]
    if not extras:
        return FieldBindingStatus.BOUND, identity_sources, None
    supported_extras = [
        (
            (label, claim),
            _canonical_sources(
                _source_for_anchor(anchor)
                for anchor in candidate.evidence
                if any(
                    _entity_metadata_present(
                        anchor.quote,
                        identity=identity,
                        label=label,
                        claim=claim,
                    )
                    for identity in identities
                )
            ),
        )
        for label, claim in extras
    ]
    extra_sources = _canonical_sources(
        source for _, sources in supported_extras for source in sources
    )
    sources = _canonical_sources([*identity_sources, *extra_sources])
    if all(matches for _, matches in supported_extras):
        return FieldBindingStatus.BOUND, sources, None
    return FieldBindingStatus.AMBIGUOUS, sources, "field_partially_supported_by_evidence_quotes"


def _entity_metadata_present(
    quote: str,
    *,
    identity: object,
    label: str,
    claim: object,
    allow_identity_equivalence: bool = True,
) -> bool:
    """Require metadata to be explicitly attached to its entity identity."""

    return any(
        _entity_metadata_present_in_clause(
            clause,
            identity=identity,
            label=label,
            claim=claim,
            allow_identity_equivalence=allow_identity_equivalence,
        )
        for clause in _quote_clauses(quote)
    )


def _entity_metadata_present_in_clause(
    clause: str,
    *,
    identity: object,
    label: str,
    claim: object,
    allow_identity_equivalence: bool = True,
) -> bool:
    if not _claim_present(clause, identity) or not _claim_present(clause, claim):
        return False
    if (
        allow_identity_equivalence
        and isinstance(identity, str)
        and isinstance(claim, str)
        and _normalized(identity) == _normalized(claim)
    ):
        # Exact self-identification is deterministic (for example, raw system
        # ``Microsoft`` and provider ``Microsoft``).  Do not extend this to token-set
        # compatibility: a shorter actor name must not bless another entity's ID.
        return True
    labels = _metadata_labels(label)
    protected = [
        *_object_claim_spans(clause, identity),
        *_object_claim_spans(clause, claim),
    ]
    boundaries = [
        match for match in _NUMBER.finditer(clause) if not _span_is_within(match.span(), protected)
    ]
    starts = [0, *(match.end() for match in boundaries)]
    ends = [*(match.start() for match in boundaries), len(clause)]
    for start, end in zip(starts, ends, strict=True):
        segment = clause[start:end]
        if not _claim_present(segment, identity) or not _claim_present(segment, claim):
            continue
        identity_spans = _object_claim_spans(segment, identity)
        claim_spans = _object_claim_spans(segment, claim)
        for marker in labels:
            for identity_span, marker_span, claim_span in product(
                identity_spans,
                _metadata_marker_spans(segment, marker),
                claim_spans,
            ):
                forward = identity_span[1] <= marker_span[0] <= marker_span[1] <= claim_span[0]
                reverse = claim_span[1] <= marker_span[0] <= marker_span[1] <= identity_span[0]
                forward_suffix = (
                    identity_span[1] <= claim_span[0] <= claim_span[1] <= marker_span[0]
                )
                if not (forward or reverse or forward_suffix):
                    continue
                association_start = min(identity_span[0], claim_span[0])
                association_end = max(identity_span[1], claim_span[1])
                if not _has_unclaimed_relation_boundary(
                    segment,
                    start=association_start,
                    end=association_end,
                    claimed_spans=(identity_span, marker_span, claim_span),
                ):
                    return True
    return False


def _metadata_labels(label: str) -> tuple[str, ...]:
    return {
        "canonical_id": ("canonical id", "canonical"),
        "dataset_id": ("dataset id", "id"),
        "dataset_url": ("dataset url", "url"),
        "dataset_version": ("dataset version", "version", "v"),
        "version": ("version", "v"),
        "provider": ("provider", "by", "from"),
        "kind": ("metric kind", "kind"),
        "lower_is_better": ("lower is better", "higher is better"),
        "min_score": ("minimum", "min", "range"),
        "max_score": ("maximum", "max", "range"),
    }.get(label, (label.replace("_", " "),))


def _metadata_marker_spans(text: str, marker: str) -> list[tuple[int, int]]:
    """Return marker spans including an adjacent assignment delimiter."""

    spans: list[tuple[int, int]] = []
    for start, end in _claim_spans(text, marker):
        suffix = re.match(r"\s*(?:(?:is|of|at)\b\s*|[:=]\s*)?", text[end:], re.IGNORECASE)
        spans.append((start, end + (suffix.end() if suffix is not None else 0)))
    return spans


def _labeled_claim_present(quote: str, label: str, claim: object) -> bool:
    return any(
        _labeled_claim_present_in_clause(clause, label, claim) for clause in _quote_clauses(quote)
    )


def _labeled_claim_present_in_clause(
    clause: str,
    label: str,
    claim: object,
) -> bool:
    if not _claim_present(clause, claim):
        return False
    normalized_quote = _normalized(clause)
    normalized_label = _normalized(label)
    normalized_claim = _normalized(str(claim))
    if not normalized_label or not normalized_claim:
        return False
    label_then_value = (
        rf"\b{re.escape(normalized_label)}\b "
        rf"(?:is |of |at )?{re.escape(normalized_claim)}\b"
    )
    value_then_label = rf"\b{re.escape(normalized_claim)}\b {re.escape(normalized_label)}\b"
    return bool(
        re.search(label_then_value, normalized_quote)
        or re.search(value_then_label, normalized_quote)
    )


def _setting_owner_claims(candidate: CandidateObservation) -> list[object]:
    owners: list[object] = [
        role.raw_name for role in candidate.roles if role.role is ActorRole.EVALUATED_SYSTEM
    ]
    if candidate.scope is not None:
        owners.append(candidate.scope.dataset_raw)
    if candidate.metric is not None:
        owners.append(candidate.metric.raw_name)
    if candidate.value is not None:
        owners.append(candidate.value.raw)
    return owners


def _associated_setting_sources(
    candidate: CandidateObservation,
    *,
    labels: Iterable[str],
    value: object,
) -> list[FieldSourceRef]:
    """Bind a setting only when its typed literal is local to this observation."""

    owners = _setting_owner_claims(candidate)
    return _canonical_sources(
        _source_for_anchor(anchor)
        for anchor in candidate.evidence
        if any(
            _entity_metadata_present(
                anchor.quote,
                identity=owner,
                label=label,
                claim=value,
                allow_identity_equivalence=False,
            )
            for owner in owners
            for label in labels
        )
    )


def _scope_setting_sources(
    candidate: CandidateObservation,
    *,
    label: str,
    value: object,
) -> list[FieldSourceRef]:
    return _associated_setting_sources(candidate, labels=(label,), value=value)


def _sample_count_setting_sources(
    candidate: CandidateObservation,
    sample_count: int,
) -> list[FieldSourceRef]:
    return _associated_setting_sources(
        candidate,
        labels=("sample count", "samples", "sample", "n"),
        value=sample_count,
    )


def _parameter_sources(
    candidate: CandidateObservation,
    key: str,
    value: object,
) -> list[FieldSourceRef]:
    if value is None:
        return []
    return _associated_setting_sources(
        candidate,
        labels=(key,),
        value=value,
    )


def _setting_components(
    candidate: CandidateObservation,
) -> list[tuple[str, list[FieldSourceRef]]]:
    components: list[tuple[str, list[FieldSourceRef]]] = []
    scope = candidate.scope
    if scope is not None:
        for label, value in (
            ("split", scope.split),
            ("subset", scope.subset),
            ("group", scope.group),
            ("language", scope.language),
            ("aggregation", scope.aggregation),
        ):
            if value is not None and value != "":
                components.append(
                    (label, _scope_setting_sources(candidate, label=label, value=value))
                )
        if scope.sample_count is not None:
            components.append(
                (
                    "sample_count",
                    _sample_count_setting_sources(candidate, scope.sample_count),
                )
            )
        if scope.raw_scope:
            components.append(
                (
                    "raw_scope",
                    _associated_setting_sources(
                        candidate,
                        labels=("raw scope", "scope"),
                        value=scope.raw_scope,
                    ),
                )
            )
    if candidate.metric is not None:
        components.extend(
            (f"parameter:{key}", _parameter_sources(candidate, key, value))
            for key, value in candidate.metric.parameters.items()
        )
    for label, value in (
        ("construct", candidate.evaluation_construct),
        ("operationalization", candidate.operationalization),
        ("decision_rule", candidate.decision_rule),
        ("evaluation_date", candidate.evaluation_date),
    ):
        if value is not None and value != "":
            components.append(
                (
                    label,
                    _associated_setting_sources(
                        candidate,
                        labels=(label.replace("_", " "),),
                        value=value,
                    ),
                )
            )

    for role in candidate.roles:
        if role.role is ActorRole.EVALUATED_SYSTEM:
            continue
        identity_sources = _claim_sources(candidate, role.raw_name)
        components.append((f"role:{role.role.value}:raw_name", identity_sources))
        for label, value in (
            ("canonical_id", role.canonical_id),
            ("version", role.version),
            ("provider", role.provider),
        ):
            if value is None or value == "":
                continue
            components.append(
                (
                    f"role:{role.role.value}:{label}",
                    _canonical_sources(
                        _source_for_anchor(anchor)
                        for anchor in candidate.evidence
                        if _entity_metadata_present(
                            anchor.quote,
                            identity=role.raw_name,
                            label=label,
                            claim=value,
                        )
                    ),
                )
            )

    return components


#: Setting components that are free-text description rather than a literal of the
#: measurement. A paper states them in prose, if at all, so requiring them to appear
#: verbatim inside the result quote blocks candidates whose measurement is fully bound.
DESCRIPTIVE_SETTING_COMPONENTS = frozenset(
    {"construct", "operationalization", "decision_rule", "evaluation_date"}
)


def literal_setting_components_are_bound(candidate: CandidateObservation) -> bool:
    """True when every non-descriptive setting component is bound to evidence.

    The binding status itself is unchanged and still records description as unbound. This
    answers a different question, asked only by the export gate: is the part of the
    setting that a reader could check against the quote actually bound.
    """

    components = [
        (label, sources)
        for label, sources in _setting_components(candidate)
        if label not in DESCRIPTIVE_SETTING_COMPONENTS
    ]
    if not components:
        return True
    return all(bool(sources) for _, sources in components)


def _setting_binding(
    candidate: CandidateObservation,
) -> tuple[FieldBindingStatus, list[FieldSourceRef], str | None]:
    """Bind each setting member to typed, entity-local evidence."""

    components = _setting_components(candidate)

    if not components:
        return FieldBindingStatus.UNSUPPORTED, [], "field_not_present"
    sources = _canonical_sources(
        source for _, component_sources in components for source in component_sources
    )
    supported_count = sum(bool(component_sources) for _, component_sources in components)
    if supported_count == len(components):
        return FieldBindingStatus.BOUND, sources, None
    if supported_count:
        return (
            FieldBindingStatus.AMBIGUOUS,
            sources,
            "field_partially_supported_by_evidence_quotes",
        )
    return FieldBindingStatus.UNSUPPORTED, [], "field_not_supported_by_evidence_quotes"


def _unit_sources(candidate: CandidateObservation) -> list[FieldSourceRef]:
    units = {
        unit
        for unit in (
            candidate.metric.unit if candidate.metric is not None else None,
            candidate.value.unit if candidate.value is not None else None,
        )
        if unit
    }
    if len(units) != 1 or candidate.value is None:
        return []
    unit = next(iter(units))
    raw_match = _NUMBER.search(candidate.value.raw)
    if raw_match is None:
        return []
    raw_token = raw_match.group().replace("\N{MINUS SIGN}", "-").replace(",", "")
    try:
        raw_numeric = Decimal(raw_token)
    except InvalidOperation:
        return []
    sources: list[FieldSourceRef] = []
    for anchor in candidate.evidence:
        quote = anchor.quote
        occurrences = _matching_number_occurrences(quote, raw_numeric)
        if not occurrences:
            continue
        if unit == "percent":
            notation = [bool(re.match(r"^\s*%", quote[match.end() :])) for match in occurrences]
            supports = bool(notation) and all(notation)
        else:
            supports = any(
                _phrase_present(clause, unit) and _claim_present(clause, candidate.value.raw)
                for clause in _quote_clauses(quote)
            )
        if supports:
            sources.append(_source_for_anchor(anchor))
    return _canonical_sources(sources)


_LOAD_BEARING_FIELDS = (
    CandidateField.SYSTEM,
    CandidateField.DATASET_SCOPE,
    CandidateField.METRIC,
    CandidateField.VALUE,
)


def _populated_association_fields(
    candidate: CandidateObservation,
) -> tuple[CandidateField, ...]:
    fields = tuple(field for field in _LOAD_BEARING_FIELDS if _field_present(candidate, field))
    if _field_present(candidate, CandidateField.SETTING):
        fields = (*fields, CandidateField.SETTING)
    if _field_present(candidate, CandidateField.UNIT):
        fields = (*fields, CandidateField.UNIT)
    return fields


def _claim_spans(text: str, claim: str) -> list[tuple[int, int]]:
    tokens = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", claim).casefold())
    if not tokens:
        return []

    def token_pattern(token: str) -> str:
        wrapped = [
            re.escape(token[:index]) + r"[-\u2010-\u2015\u2212]\s+" + re.escape(token[index:])
            for index in range(1, len(token))
        ]
        return "(?:" + "|".join([re.escape(token), *wrapped]) + ")"

    pattern = re.compile(
        r"(?<!\w)" + r"\W+".join(token_pattern(token) for token in tokens) + r"(?!\w)",
        re.IGNORECASE,
    )
    return [match.span() for match in pattern.finditer(text)]


def _object_claim_spans(text: str, claim: object) -> list[tuple[int, int]]:
    if isinstance(claim, bool):
        return _claim_spans(text, str(claim))
    if isinstance(claim, int | float | Decimal):
        return [match.span() for match in _matching_number_occurrences(text, claim)]
    return _claim_spans(text, str(claim))


def _span_is_within(span: tuple[int, int], containers: Iterable[tuple[int, int]]) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in containers)


_RELATION_BOUNDARY = re.compile(
    r"[;,/:]|(?<!\d)\.(?!\d)|\b(?:and|baseline|compared|versus|vs|whereas|while)\b",
    re.IGNORECASE,
)
_CAPITALIZED_ENTITY_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*\b")


@dataclass(frozen=True)
class _SelectedCoreClaim:
    field: CandidateField
    value: str
    span: tuple[int, int]


@dataclass(frozen=True)
class _ResultOccurrence:
    window: tuple[int, int]
    raw_span: tuple[int, int]
    claims: tuple[_SelectedCoreClaim, ...]


def _has_unclaimed_relation_boundary(
    quote: str,
    *,
    start: int,
    end: int,
    claimed_spans: Iterable[tuple[int, int]],
) -> bool:
    claimed = list(claimed_spans)
    for pattern in (_RELATION_BOUNDARY, _CAPITALIZED_ENTITY_TOKEN):
        for match in pattern.finditer(quote, start, end):
            if not _span_is_within(match.span(), claimed):
                return True
    return False


def _association_auxiliary_claim_spans(
    candidate: CandidateObservation,
    quote: str,
) -> list[tuple[int, int]]:
    """Return only explicitly labeled metadata/setting literal occurrences."""

    items: list[tuple[tuple[str, ...], object]] = []
    for role in candidate.roles:
        items.extend(
            (_metadata_labels(label), value)
            for label, value in (
                ("canonical_id", role.canonical_id),
                ("version", role.version),
                ("provider", role.provider),
            )
        )
    scope = candidate.scope
    if scope is not None:
        items.extend(
            (_metadata_labels(label), value)
            for label, value in (
                ("dataset_id", scope.dataset_id),
                ("dataset_url", scope.dataset_url),
                ("dataset_version", scope.dataset_version),
                ("split", scope.split),
                ("subset", scope.subset),
                ("group", scope.group),
                ("language", scope.language),
                ("aggregation", scope.aggregation),
                ("raw scope", scope.raw_scope),
            )
        )
        if scope.sample_count is not None:
            items.append((("sample count", "samples", "sample", "n"), scope.sample_count))
    metric = candidate.metric
    if metric is not None:
        items.extend(
            (_metadata_labels(label), value)
            for label, value in (
                ("canonical_id", metric.canonical_id),
                ("kind", metric.kind),
                ("lower_is_better", metric.lower_is_better),
                ("min_score", metric.min_score),
                ("max_score", metric.max_score),
            )
        )
        items.extend(((key,), value) for key, value in metric.parameters.items())
    items.extend(
        ((label,), value)
        for label, value in (
            ("construct", candidate.evaluation_construct),
            ("operationalization", candidate.operationalization),
            ("decision rule", candidate.decision_rule),
            ("evaluation date", candidate.evaluation_date),
        )
    )
    spans = [
        claim_span
        for labels, claim in items
        if claim is not None and claim != ""
        for _, claim_span in _metadata_pair_options(
            quote,
            labels=labels,
            claim=claim,
            window=(0, len(quote)),
        )
    ]
    if candidate.value is not None:
        uncertainty = candidate.value.uncertainty
        if uncertainty is not None:
            spans.extend(
                span
                for claim in (
                    uncertainty.standard_error,
                    uncertainty.standard_deviation,
                    uncertainty.confidence_interval_lower,
                    uncertainty.confidence_interval_upper,
                    uncertainty.confidence_level,
                    uncertainty.num_samples,
                )
                if claim is not None
                for span in _object_claim_spans(quote, claim)
            )
    return sorted(set(spans))


def _atomic_result_window(
    quote: str,
    *,
    association_start: int,
    association_end: int,
    claimed_spans: Iterable[tuple[int, int]],
    hard_barrier_spans: Iterable[tuple[int, int]] = (),
) -> tuple[int, int]:
    """Bound one result occurrence away from unrelated actors and values.

    Field-level predicates may legitimately consume metadata immediately before or
    after the core system/dataset/metric/value expression.  They must not, however,
    cross an unclaimed actor/relation boundary or another numeric result.  Return the
    largest surrounding slice that observes those boundaries.
    """

    claimed = list(claimed_spans)
    barriers = [
        match.span()
        for pattern in (_RELATION_BOUNDARY, _CAPITALIZED_ENTITY_TOKEN, _NUMBER)
        for match in pattern.finditer(quote)
        if not _span_is_within(match.span(), claimed)
    ]
    barriers.extend(hard_barrier_spans)
    starts = [end for start, end in barriers if end <= association_start]
    ends = [start for start, end in barriers if start >= association_end]
    return max(starts, default=0), min(ends, default=len(quote))


def _prose_load_bearing_tuple_windows(
    candidate: CandidateObservation,
    anchor: EvidenceAnchor,
) -> list[_ResultOccurrence]:
    """Return boundary-isolated slices containing one core result occurrence."""

    quote = anchor.quote
    evaluated = [role for role in candidate.roles if role.role is ActorRole.EVALUATED_SYSTEM]
    if not evaluated or candidate.metric is None or candidate.value is None:
        return []
    if value_quote_support_issue(candidate.value, quote) is not None:
        return []
    claim_entries = [(CandidateField.SYSTEM, role.raw_name) for role in evaluated]
    if candidate.scope is not None:
        claim_entries.append((CandidateField.DATASET_SCOPE, candidate.scope.dataset_raw))
    claim_entries.append((CandidateField.METRIC, candidate.metric.raw_name))
    claim_entries.extend(
        (CandidateField.SETTING, role.raw_name)
        for role in candidate.roles
        if role.role is not ActorRole.EVALUATED_SYSTEM
    )
    spans_by_claim = [_claim_spans(quote, claim) for _, claim in claim_entries]
    if any(not spans for spans in spans_by_claim):
        return []
    identity_spans = [span for spans in spans_by_claim for span in spans]
    auxiliary_spans = _association_auxiliary_claim_spans(candidate, quote)
    raw_match = _NUMBER.search(candidate.value.raw)
    if raw_match is None:
        return []
    raw_token = raw_match.group().replace("\N{MINUS SIGN}", "-").replace(",", "")
    try:
        raw_numeric = Decimal(raw_token)
    except InvalidOperation:
        return []
    all_numbers = list(_NUMBER.finditer(quote))
    occurrences: dict[
        tuple[tuple[int, int], tuple[int, int], tuple[tuple[int, int], ...]],
        _ResultOccurrence,
    ] = {}
    for raw_occurrence in _matching_number_occurrences(quote, raw_numeric):
        preceding = [
            number
            for number in all_numbers
            if number.end() <= raw_occurrence.start()
            and not _span_is_within(number.span(), [*identity_spans, *auxiliary_spans])
        ]
        following = [
            number
            for number in all_numbers
            if number.start() >= raw_occurrence.end()
            and not _span_is_within(number.span(), [*identity_spans, *auxiliary_spans])
        ]
        local_start = preceding[-1].end() if preceding else 0
        local_end = following[0].start() if following else len(quote)
        local_claim_spans = [
            [span for span in spans if local_start <= span[0] and span[1] <= local_end]
            for spans in spans_by_claim
        ]
        if any(not spans for spans in local_claim_spans):
            continue
        for selected_spans in product(*local_claim_spans):
            claimed_spans = (*selected_spans, raw_occurrence.span(), *auxiliary_spans)
            selected_set = set(selected_spans)
            hard_barriers = [
                span
                for span in identity_spans
                if span not in selected_set and not _span_is_within(span, auxiliary_spans)
            ]
            hard_barriers.extend(
                match.span()
                for match in _matching_number_occurrences(quote, raw_numeric)
                if match.span() != raw_occurrence.span()
                and not _span_is_within(match.span(), auxiliary_spans)
            )
            association_start = min(raw_occurrence.start(), *(start for start, _ in selected_spans))
            association_end = max(raw_occurrence.end(), *(end for _, end in selected_spans))
            if any(
                association_start <= start and end <= association_end
                for start, end in hard_barriers
            ):
                continue
            if not _has_unclaimed_relation_boundary(
                quote,
                start=association_start,
                end=association_end,
                claimed_spans=claimed_spans,
            ):
                window = _atomic_result_window(
                    quote,
                    association_start=association_start,
                    association_end=association_end,
                    claimed_spans=claimed_spans,
                    hard_barrier_spans=hard_barriers,
                )
                selected_claims = tuple(
                    _SelectedCoreClaim(field=field, value=claim, span=span)
                    for (field, claim), span in zip(claim_entries, selected_spans, strict=True)
                )
                occurrence = _ResultOccurrence(
                    window=window,
                    raw_span=raw_occurrence.span(),
                    claims=selected_claims,
                )
                occurrences[(window, raw_occurrence.span(), selected_spans)] = occurrence
    return [occurrences[key] for key in sorted(occurrences)]


def _candidate_for_atomic_quote(
    candidate: CandidateObservation,
    anchor: EvidenceAnchor,
    *,
    start: int,
    end: int,
) -> CandidateObservation:
    quote = anchor.quote[start:end].strip()
    local_anchor = EvidenceAnchor(
        source_id=anchor.source_id,
        page=anchor.page,
        kind=EvidenceKind.PROSE,
        quote=quote,
    )
    return candidate.model_copy(deep=True, update={"evidence": [local_anchor]})


def _atomic_quote_derivations(
    candidate: CandidateObservation,
) -> dict[CandidateField, str]:
    """Re-prove deterministic metric facts using only one atomic quote slice."""

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
        reasons[CandidateField.UNIT] = "deterministic_reference_resolution=atomic_quote_unit"
    return reasons


_LOCAL_RELATION_GAP = re.compile(r"^[\s=:\-\N{EN DASH}\N{EM DASH}()\[\]{}]*$")


def _locally_joined_spans(
    quote: str,
    spans: Iterable[tuple[int, int]],
) -> bool:
    ordered = sorted(set(spans))
    return all(
        left[1] <= right[0] and bool(_LOCAL_RELATION_GAP.fullmatch(quote[left[1] : right[0]]))
        for left, right in zip(ordered, ordered[1:], strict=False)
    )


def _metadata_pair_options(
    quote: str,
    *,
    labels: Iterable[str],
    claim: object,
    window: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    start, end = window
    claim_spans = [
        span for span in _object_claim_spans(quote, claim) if start <= span[0] and span[1] <= end
    ]
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for label in labels:
        marker_spans = [
            span
            for span in _metadata_marker_spans(quote, label)
            if start <= span[0] and span[1] <= end
        ]
        for marker_span, claim_span in product(marker_spans, claim_spans):
            if _locally_joined_spans(quote, (marker_span, claim_span)):
                options.append((marker_span, claim_span))
    return options


def _metadata_chain_supported_at_identity(
    quote: str,
    *,
    identity: str,
    identity_span: tuple[int, int],
    items: Iterable[tuple[Iterable[str], object]],
    window: tuple[int, int],
    allow_identity_equivalence: bool,
) -> bool:
    """Require every metadata literal in one contiguous chain at this identity."""

    unresolved: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []
    for labels, claim in items:
        if claim is None or claim == "":
            continue
        if (
            allow_identity_equivalence
            and isinstance(claim, str)
            and _normalized(identity) == _normalized(claim)
        ):
            continue
        options = _metadata_pair_options(
            quote,
            labels=labels,
            claim=claim,
            window=window,
        )
        if not options:
            return False
        unresolved.append(options)
    if not unresolved:
        return True
    return any(
        _locally_joined_spans(
            quote,
            (
                identity_span,
                *(span for option in selected for span in option),
            ),
        )
        for selected in product(*unresolved)
    )


def _occurrence_claim_spans(
    occurrence: _ResultOccurrence,
    *,
    field: CandidateField,
    value: str,
) -> list[tuple[int, int]]:
    normalized = _normalized(value)
    return [
        claim.span
        for claim in occurrence.claims
        if claim.field is field and _normalized(claim.value) == normalized
    ]


def _identity_fields_supported_at_occurrence(
    candidate: CandidateObservation,
    quote: str,
    occurrence: _ResultOccurrence,
    derived: dict[CandidateField, str],
) -> bool:
    evaluated = [role for role in candidate.roles if role.role is ActorRole.EVALUATED_SYSTEM]
    for role in evaluated:
        spans = _occurrence_claim_spans(
            occurrence,
            field=CandidateField.SYSTEM,
            value=role.raw_name,
        )
        items = [
            (_metadata_labels(label), value)
            for label, value in (
                ("canonical_id", role.canonical_id),
                ("version", role.version),
                ("provider", role.provider),
            )
        ]
        if not any(
            _metadata_chain_supported_at_identity(
                quote,
                identity=role.raw_name,
                identity_span=span,
                items=items,
                window=occurrence.window,
                allow_identity_equivalence=True,
            )
            for span in spans
        ):
            return False

    scope = candidate.scope
    if scope is not None:
        spans = _occurrence_claim_spans(
            occurrence,
            field=CandidateField.DATASET_SCOPE,
            value=scope.dataset_raw,
        )
        items = [
            (_metadata_labels(label), value)
            for label, value in (
                ("dataset_id", scope.dataset_id),
                ("dataset_url", scope.dataset_url),
                ("dataset_version", scope.dataset_version),
            )
        ]
        if not any(
            _metadata_chain_supported_at_identity(
                quote,
                identity=scope.dataset_raw,
                identity_span=span,
                items=items,
                window=occurrence.window,
                allow_identity_equivalence=True,
            )
            for span in spans
        ):
            return False

    metric = candidate.metric
    if metric is not None and CandidateField.METRIC not in derived:
        spans = _occurrence_claim_spans(
            occurrence,
            field=CandidateField.METRIC,
            value=metric.raw_name,
        )
        items = [
            (_metadata_labels(label), value)
            for label, value in (
                ("canonical_id", metric.canonical_id),
                ("kind", metric.kind),
                ("lower_is_better", metric.lower_is_better),
                ("min_score", metric.min_score),
                ("max_score", metric.max_score),
            )
        ]
        if not any(
            _metadata_chain_supported_at_identity(
                quote,
                identity=metric.raw_name,
                identity_span=span,
                items=items,
                window=occurrence.window,
                allow_identity_equivalence=True,
            )
            for span in spans
        ):
            return False
    return True


def _ordinary_setting_items(
    candidate: CandidateObservation,
) -> list[tuple[tuple[str, ...], object]]:
    items: list[tuple[tuple[str, ...], object]] = []
    scope = candidate.scope
    if scope is not None:
        for label, value in (
            ("split", scope.split),
            ("subset", scope.subset),
            ("group", scope.group),
            ("language", scope.language),
            ("aggregation", scope.aggregation),
            ("raw scope", scope.raw_scope),
        ):
            if value is not None and value != "":
                items.append(((label,), value))
        if scope.sample_count is not None:
            items.append((("sample count", "samples", "sample", "n"), scope.sample_count))
    if candidate.metric is not None:
        items.extend(((key,), value) for key, value in candidate.metric.parameters.items())
    for label, value in (
        ("construct", candidate.evaluation_construct),
        ("operationalization", candidate.operationalization),
        ("decision rule", candidate.decision_rule),
        ("evaluation date", candidate.evaluation_date),
    ):
        if value is not None and value != "":
            items.append(((label,), value))
    return items


def _setting_supported_at_occurrence(
    candidate: CandidateObservation,
    quote: str,
    occurrence: _ResultOccurrence,
) -> bool:
    items = _ordinary_setting_items(candidate)
    owners = [
        (claim.value, claim.span)
        for claim in occurrence.claims
        if claim.field
        in {
            CandidateField.SYSTEM,
            CandidateField.DATASET_SCOPE,
            CandidateField.METRIC,
        }
    ]
    if candidate.value is not None:
        owners.append((candidate.value.raw, occurrence.raw_span))
    if items and not any(
        _metadata_chain_supported_at_identity(
            quote,
            identity=identity,
            identity_span=span,
            items=items,
            window=occurrence.window,
            allow_identity_equivalence=False,
        )
        for identity, span in owners
    ):
        return False

    for role in candidate.roles:
        if role.role is ActorRole.EVALUATED_SYSTEM:
            continue
        spans = _occurrence_claim_spans(
            occurrence,
            field=CandidateField.SETTING,
            value=role.raw_name,
        )
        metadata = [
            (_metadata_labels(label), value)
            for label, value in (
                ("canonical_id", role.canonical_id),
                ("version", role.version),
                ("provider", role.provider),
            )
        ]
        if not any(
            _metadata_chain_supported_at_identity(
                quote,
                identity=role.raw_name,
                identity_span=span,
                items=metadata,
                window=occurrence.window,
                allow_identity_equivalence=True,
            )
            for span in spans
        ):
            return False
    return True


def _unit_supported_at_occurrence(
    candidate: CandidateObservation,
    quote: str,
    occurrence: _ResultOccurrence,
    derived: dict[CandidateField, str],
) -> bool:
    if CandidateField.UNIT in derived:
        return True
    units = {
        unit
        for unit in (
            candidate.metric.unit if candidate.metric is not None else None,
            candidate.value.unit if candidate.value is not None else None,
        )
        if unit
    }
    if len(units) != 1:
        return False
    unit = next(iter(units))
    raw_end = occurrence.raw_span[1]
    suffix = quote[raw_end : occurrence.window[1]]
    if unit == "percent":
        return bool(re.match(r"^\s*%", suffix))
    return any(
        span[0] >= raw_end and _locally_joined_spans(quote, (occurrence.raw_span, span))
        for span in _claim_spans(quote, unit)
        if occurrence.window[0] <= span[0] and span[1] <= occurrence.window[1]
    )


def _prose_supports_load_bearing_tuple(
    candidate: CandidateObservation,
    anchor: EvidenceAnchor,
) -> bool:
    """Require every populated output field in one atomic result occurrence."""

    association_fields = _populated_association_fields(candidate)
    for occurrence in _prose_load_bearing_tuple_windows(candidate, anchor):
        start, end = occurrence.window
        local = _candidate_for_atomic_quote(candidate, anchor, start=start, end=end)
        local_derived = _atomic_quote_derivations(local)
        local_bindings = _direct_quote_bindings(local, derived=local_derived)
        direct_fields_bound = all(
            _field_binding(local, field, local_bindings, local_derived)[0]
            is FieldBindingStatus.BOUND
            for field in association_fields
        )
        if not direct_fields_bound:
            continue
        if not _identity_fields_supported_at_occurrence(
            candidate,
            anchor.quote,
            occurrence,
            local_derived,
        ):
            continue
        if _field_present(candidate, CandidateField.SETTING) and not (
            _setting_supported_at_occurrence(candidate, anchor.quote, occurrence)
        ):
            continue
        if _field_present(candidate, CandidateField.UNIT) and not _unit_supported_at_occurrence(
            candidate,
            anchor.quote,
            occurrence,
            local_derived,
        ):
            continue
        return True
    return False


def _load_bearing_tuple_association_supported(
    candidate: CandidateObservation,
    bindings: dict[
        CandidateField,
        tuple[FieldBindingStatus, list[FieldSourceRef], str | None],
    ],
    *,
    derived: dict[CandidateField, str],
) -> bool:
    """Require one atomic prose occurrence to join every populated core field."""

    association_fields = _populated_association_fields(candidate)
    if any(
        _field_binding(candidate, field, bindings, derived)[0] is not FieldBindingStatus.BOUND
        for field in association_fields
    ):
        return True
    return any(
        anchor.kind is EvidenceKind.PROSE and _prose_supports_load_bearing_tuple(candidate, anchor)
        for anchor in candidate.evidence
    )


def _enforce_load_bearing_tuple_association(
    candidate: CandidateObservation,
    bindings: dict[
        CandidateField,
        tuple[FieldBindingStatus, list[FieldSourceRef], str | None],
    ],
    *,
    derived: dict[CandidateField, str],
) -> None:
    if _load_bearing_tuple_association_supported(candidate, bindings, derived=derived):
        return
    reason = "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    for field in _populated_association_fields(candidate):
        status, sources, _ = _field_binding(candidate, field, bindings, derived)
        if status is FieldBindingStatus.BOUND:
            bindings[field] = FieldBindingStatus.AMBIGUOUS, sources, reason


_COMPARATOR_PATTERNS = {
    ValueComparator.LESS_THAN: (r"(?:<(?!=)|\bless than\b(?!\s+or\s+equal)|\bbelow\b)"),
    ValueComparator.LESS_THAN_OR_EQUAL: (
        r"(?:<=|≤|\bless than or equal(?: to)?\b|\bat most\b|\bno more than\b)"
    ),
    ValueComparator.GREATER_THAN: (r"(?:>(?!=)|\bgreater than\b(?!\s+or\s+equal)|\babove\b)"),
    ValueComparator.GREATER_THAN_OR_EQUAL: (
        r"(?:>=|≥|\bgreater than or equal(?: to)?\b|\bat least\b|\bno less than\b)"
    ),
    ValueComparator.APPROXIMATELY: (r"(?:~|≈|\bapproximately\b|\babout\b|\broughly\b|\bcirca\b)"),
}


def _comparator_cues(text: str, *, anchored: bool = False) -> set[ValueComparator]:
    suffix = r"\s*$" if anchored else ""
    return {
        comparator
        for comparator, pattern in _COMPARATOR_PATTERNS.items()
        if re.search(f"(?:{pattern}){suffix}", text)
    }


def _normalized_literal_occurrences(text: str, literal: str) -> list[re.Match[str]]:
    normalized_text = unicodedata.normalize("NFKC", text).replace("\N{MINUS SIGN}", "-").casefold()
    normalized_literal = (
        unicodedata.normalize("NFKC", literal).replace("\N{MINUS SIGN}", "-").casefold()
    )
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    normalized_literal = re.sub(r"\s+", " ", normalized_literal).strip()
    if not normalized_literal:
        return []
    pattern = re.compile(rf"(?<![\w.,]){re.escape(normalized_literal)}(?![\w.,])")
    return list(pattern.finditer(normalized_text))


def _comparator_supported(quote: str, raw: str, comparator: ValueComparator) -> bool:
    """Validate the comparator at the exact raw-value occurrence in one quote."""

    normalized_quote = (
        unicodedata.normalize("NFKC", quote).replace("\N{MINUS SIGN}", "-").casefold()
    )
    normalized_quote = re.sub(r"\s+", " ", normalized_quote).strip()
    normalized_raw = unicodedata.normalize("NFKC", raw).replace("\N{MINUS SIGN}", "-").casefold()
    normalized_raw = re.sub(r"\s+", " ", normalized_raw).strip()
    raw_cues = _comparator_cues(normalized_raw)
    occurrence_cues: list[set[ValueComparator]] = []
    for occurrence in _normalized_literal_occurrences(quote, raw):
        # A source cue must immediately govern this value.  This avoids borrowing an
        # inequality from another cell or clause in the same exact excerpt.
        prefix = normalized_quote[: occurrence.start()]
        occurrence_cues.append(raw_cues | _comparator_cues(prefix, anchored=True))
    if not occurrence_cues:
        return False
    if comparator is ValueComparator.EXACT:
        return all(not cues for cues in occurrence_cues)
    return all(cues == {comparator} for cues in occurrence_cues)


def _numeric_projection_matches_raw(value: ReportedValue) -> bool:
    """Require ``raw`` to be one point estimate and its exact numeric projection."""

    matches = list(_NUMBER.finditer(value.raw))
    if len(matches) != 1:
        return False
    match = matches[0]
    token = match.group().replace("\N{MINUS SIGN}", "-").replace(",", "")
    try:
        if Decimal(token) != Decimal(str(value.numeric)):
            return False
    except InvalidOperation:
        return False

    residue = value.raw[: match.start()] + " " + value.raw[match.end() :]
    residue = unicodedata.normalize("NFKC", residue).casefold()
    for pattern in _COMPARATOR_PATTERNS.values():
        residue = re.sub(pattern, " ", residue, flags=re.IGNORECASE)
    if value.unit:
        residue = re.sub(re.escape(value.unit.casefold()), " ", residue)
    # Percent notation may itself resolve the unit.  Bracketing and conventional
    # non-alphanumeric footnote marks do not change the point estimate.
    residue = re.sub(r"[%*†‡§=\s\[\](){}.,:]+", "", residue)
    return not residue


def _matching_number_occurrences(quote: str, value: int | float | Decimal) -> list[re.Match[str]]:
    try:
        expected = Decimal(str(value))
    except InvalidOperation:
        return []
    matches: list[re.Match[str]] = []
    for match in _NUMBER.finditer(quote):
        token = match.group().replace("\N{MINUS SIGN}", "-").replace(",", "")
        try:
            if Decimal(token) == expected:
                matches.append(match)
        except InvalidOperation:
            continue
    return matches


def _number_has_local_statistic_label(
    quote: str,
    value: int | float,
    label: str,
    *,
    raw_end: int,
    allowed_words: Iterable[str] = (),
) -> bool:
    """Require a statistic label to govern the same adjacent numeric token."""

    for match in _matching_number_occurrences(quote, value):
        if match.start() < raw_end:
            continue
        prefix = quote[raw_end : match.start()]
        suffix = quote[match.end() : min(len(quote), match.end() + 48)]
        before = re.search(rf"(?:{label})\s*(?:=|:)?\s*$", prefix, re.IGNORECASE)
        if before and _local_expression_follows_raw(
            quote,
            raw_end=raw_end,
            expression_start=raw_end + before.start(),
            allowed_words=allowed_words,
        ):
            return True
        if re.match(rf"^\s*(?:{label})\b", suffix, re.IGNORECASE) and (
            _local_expression_follows_raw(
                quote,
                raw_end=raw_end,
                expression_start=match.start(),
                allowed_words=allowed_words,
            )
        ):
            return True
    return False


def _local_expression_follows_raw(
    quote: str,
    *,
    raw_end: int,
    expression_start: int,
    allowed_words: Iterable[str] = (),
) -> bool:
    """Reject another value or actor phrase between a point estimate and qualifier."""

    if expression_start < raw_end:
        return False
    intervening = quote[raw_end:expression_start]
    if _NUMBER.search(intervening):
        return False
    tokens = re.findall(r"[a-z]+", intervening.casefold())
    allowed = {"and", "as", "at", "is", "of", "reported", "was", "with"}
    allowed.update(word.casefold() for word in allowed_words)
    return set(tokens) <= allowed


def _ci_expression_start(quote: str, *, raw_end: int, cue_start: int) -> int:
    """Include an immediately preceding confidence percentage in the CI expression."""

    preceding = [match for match in _NUMBER.finditer(quote[raw_end:cue_start])]
    if not preceding:
        return cue_start
    last = preceding[-1]
    absolute_start = raw_end + last.start()
    absolute_end = raw_end + last.end()
    if re.fullmatch(r"\s*%\s*", quote[absolute_end:cue_start]):
        return absolute_start
    return cue_start


def _confidence_interval_supported(
    quote: str,
    lower: float | None,
    upper: float | None,
    *,
    raw_end: int,
) -> bool:
    if lower is None or upper is None:
        return False
    cue_matches = list(re.finditer(r"(?:\bci\b|confidence\s+interval)", quote, re.IGNORECASE))
    separator = re.compile(r"^[\s\[\](){}:;,\-–—]+(?:to\s+)?[\s\[\](){}:;,\-–—]*$", re.I)
    for left in _matching_number_occurrences(quote, lower):
        for right in _matching_number_occurrences(quote, upper):
            if (
                left.start() < raw_end
                or right.start() <= left.end()
                or not separator.fullmatch(quote[left.end() : right.start()])
            ):
                continue
            for cue in cue_matches:
                if (
                    raw_end <= cue.start()
                    and cue.end() <= left.start()
                    and not _NUMBER.search(quote[cue.end() : left.start()])
                    and _local_expression_follows_raw(
                        quote,
                        raw_end=raw_end,
                        expression_start=_ci_expression_start(
                            quote,
                            raw_end=raw_end,
                            cue_start=cue.start(),
                        ),
                    )
                ):
                    return True
                if (
                    cue.start() >= right.end()
                    and not _NUMBER.search(quote[right.end() : cue.start()])
                    and _local_expression_follows_raw(
                        quote,
                        raw_end=raw_end,
                        expression_start=left.start(),
                    )
                ):
                    return True
    return False


def _confidence_level_supported(quote: str, level: float, *, raw_end: int) -> bool:
    expected = Decimal(str(level))
    percent = expected * 100 if Decimal(0) <= expected <= Decimal(1) else None
    for match in _NUMBER.finditer(quote):
        if match.start() < raw_end:
            continue
        token = match.group().replace("\N{MINUS SIGN}", "-").replace(",", "")
        try:
            actual = Decimal(token)
        except InvalidOperation:
            continue
        suffix = quote[match.end() : min(len(quote), match.end() + 48)]
        prefix = quote[max(0, match.start() - 48) : match.start()]
        percent_suffix = re.match(r"^\s*%", suffix)
        equivalent = actual == expected or (
            percent is not None and actual == percent and bool(percent_suffix)
        )
        if not equivalent:
            continue
        before = re.search(
            r"(?:\bci\b|confidence\s+(?:interval|level))\s*(?:=|:)?\s*$",
            prefix,
            re.IGNORECASE,
        )
        after = re.match(
            r"^\s*%?\s*(?:\bci\b|confidence\s+(?:interval|level))",
            suffix,
            re.IGNORECASE,
        )
        expression_start = match.start()
        if before is not None:
            expression_start = match.start() - (len(prefix) - before.start())
        if (before is not None or after is not None) and _local_expression_follows_raw(
            quote,
            raw_end=raw_end,
            expression_start=expression_start,
        ):
            return True
    return False


def _sample_count_supported(quote: str, count: int, *, raw_end: int) -> bool:
    for match in _matching_number_occurrences(quote, count):
        if match.start() < raw_end:
            continue
        prefix = quote[raw_end : match.start()]
        suffix = quote[match.end() : min(len(quote), match.end() + 32)]
        before = re.search(
            r"(?:\bn\s*=|\bsamples?\s*(?:=|:)?)\s*$",
            prefix,
            re.IGNORECASE,
        )
        if before and _local_expression_follows_raw(
            quote,
            raw_end=raw_end,
            expression_start=raw_end + before.start(),
        ):
            return True
        if re.match(r"^\s+samples?\b", suffix, re.IGNORECASE) and (
            _local_expression_follows_raw(
                quote,
                raw_end=raw_end,
                expression_start=match.start(),
            )
        ):
            return True
    return False


def _uncertainty_method_supported(quote: str, method: str, *, raw_end: int) -> bool:
    """Require the method cue to begin the same local uncertainty expression."""

    return any(
        start >= raw_end
        and _local_expression_follows_raw(
            quote,
            raw_end=raw_end,
            expression_start=start,
        )
        for start, _ in _claim_spans(quote, method)
    )


def _uncertainty_supported(quote: str, raw: str, uncertainty: Uncertainty) -> bool:
    values = uncertainty.model_dump(mode="python")
    if not any(value is not None and value != "" for value in values.values()):
        return False
    raw_match = _NUMBER.search(raw)
    if raw_match is None:
        return False
    raw_token = raw_match.group().replace("\N{MINUS SIGN}", "-").replace(",", "")
    try:
        raw_value = Decimal(raw_token)
    except InvalidOperation:
        return False
    raw_occurrences = _matching_number_occurrences(quote, raw_value)
    if len(raw_occurrences) != 1:
        return False
    raw_end = raw_occurrences[0].end()
    method_words = re.findall(r"[a-z]+", (uncertainty.method or "").casefold())
    if uncertainty.standard_error is not None and not _number_has_local_statistic_label(
        quote,
        uncertainty.standard_error,
        r"(?:\bse\b|standard\s+error)",
        raw_end=raw_end,
        allowed_words=method_words,
    ):
        return False
    if uncertainty.standard_deviation is not None and not _number_has_local_statistic_label(
        quote,
        uncertainty.standard_deviation,
        r"(?:\bsd\b|standard\s+deviation)",
        raw_end=raw_end,
        allowed_words=method_words,
    ):
        return False
    interval = (
        uncertainty.confidence_interval_lower,
        uncertainty.confidence_interval_upper,
    )
    if any(value is not None for value in interval) and not _confidence_interval_supported(
        quote,
        *interval,
        raw_end=raw_end,
    ):
        return False
    if uncertainty.confidence_level is not None and not _confidence_level_supported(
        quote,
        uncertainty.confidence_level,
        raw_end=raw_end,
    ):
        return False
    if uncertainty.method and not _uncertainty_method_supported(
        quote,
        uncertainty.method,
        raw_end=raw_end,
    ):
        return False
    return uncertainty.num_samples is None or _sample_count_supported(
        quote,
        uncertainty.num_samples,
        raw_end=raw_end,
    )


def value_quote_support_issue(value: ReportedValue, quote: str) -> str | None:
    """Return the deterministic blocker for one proposed value and exact quote."""

    if not _numeric_projection_matches_raw(value):
        return "value_numeric_projection_not_supported_by_raw"
    if not _claim_present(quote, value.raw):
        return "field_identity_not_supported_by_evidence_quotes"
    if not _comparator_supported(quote, value.raw, value.comparator) or (
        value.uncertainty is not None
        and not _uncertainty_supported(quote, value.raw, value.uncertainty)
    ):
        return "value_comparator_or_uncertainty_not_supported_by_evidence_quotes"
    return None


def _value_binding(
    candidate: CandidateObservation,
) -> tuple[FieldBindingStatus, list[FieldSourceRef], str | None]:
    value = candidate.value
    if value is None:
        return FieldBindingStatus.UNSUPPORTED, [], "field_not_present"
    if not _numeric_projection_matches_raw(value):
        return (
            FieldBindingStatus.UNSUPPORTED,
            [],
            "value_numeric_projection_not_supported_by_raw",
        )
    raw_anchors = [
        anchor for anchor in candidate.evidence if _claim_present(anchor.quote, value.raw)
    ]
    raw_sources = _canonical_sources(_source_for_anchor(anchor) for anchor in raw_anchors)
    if not raw_sources:
        return FieldBindingStatus.UNSUPPORTED, [], "field_identity_not_supported_by_evidence_quotes"
    supported = [
        anchor for anchor in raw_anchors if value_quote_support_issue(value, anchor.quote) is None
    ]
    if supported:
        return (
            FieldBindingStatus.BOUND,
            _canonical_sources(_source_for_anchor(anchor) for anchor in supported),
            None,
        )
    return (
        FieldBindingStatus.AMBIGUOUS,
        raw_sources,
        "value_comparator_or_uncertainty_not_supported_by_evidence_quotes",
    )


def _field_present(candidate: CandidateObservation, field: CandidateField) -> bool:
    if field is CandidateField.SYSTEM:
        return any(role.role is ActorRole.EVALUATED_SYSTEM for role in candidate.roles)
    if field is CandidateField.DATASET_SCOPE:
        return candidate.scope is not None
    if field is CandidateField.METRIC:
        return candidate.metric is not None
    if field is CandidateField.VALUE:
        return candidate.value is not None
    if field is CandidateField.UNIT:
        return bool(
            (candidate.metric is not None and candidate.metric.unit)
            or (candidate.value is not None and candidate.value.unit)
        )
    scope = candidate.scope
    metric = candidate.metric

    def present(value: object) -> bool:
        return value is not None and value != "" and value != () and value != [] and value != {}

    return any(
        present(value)
        for value in (
            scope.split if scope else None,
            scope.subset if scope else None,
            scope.group if scope else None,
            scope.language if scope else None,
            scope.sample_count if scope else None,
            scope.aggregation if scope else None,
            scope.raw_scope if scope else None,
            metric.parameters if metric else None,
            candidate.evaluation_construct,
            candidate.operationalization,
            candidate.decision_rule,
            candidate.evaluation_date,
            [role for role in candidate.roles if role.role is not ActorRole.EVALUATED_SYSTEM],
        )
    )


_FieldBinding = tuple[FieldBindingStatus, list[FieldSourceRef], str | None]


def _direct_quote_bindings(
    candidate: CandidateObservation,
    *,
    derived: dict[CandidateField, str],
) -> dict[CandidateField, _FieldBinding]:
    evaluated = [role for role in candidate.roles if role.role is ActorRole.EVALUATED_SYSTEM]
    scope = candidate.scope
    bindings: dict[CandidateField, _FieldBinding] = {}
    system_names = [role.raw_name for role in evaluated]
    bindings[CandidateField.SYSTEM] = _identity_claims_binding(
        candidate,
        system_names,
        (
            (label, value)
            for role in evaluated
            for label, value in (
                ("canonical_id", role.canonical_id),
                ("version", role.version),
                ("provider", role.provider),
            )
        ),
    )
    dataset_identity = [scope.dataset_raw] if scope is not None else []
    bindings[CandidateField.DATASET_SCOPE] = _identity_claims_binding(
        candidate,
        dataset_identity,
        (
            (label, value)
            for label, value in (
                ("dataset_id", scope.dataset_id if scope is not None else None),
                ("dataset_url", scope.dataset_url if scope is not None else None),
                ("dataset_version", scope.dataset_version if scope is not None else None),
            )
        ),
    )
    metric_identity = [candidate.metric.raw_name] if candidate.metric is not None else []
    metric_metadata: list[tuple[str, object]] = []
    if candidate.metric is not None and CandidateField.METRIC not in derived:
        metric_metadata = [
            ("canonical_id", candidate.metric.canonical_id),
            ("kind", candidate.metric.kind),
            ("lower_is_better", candidate.metric.lower_is_better),
            ("min_score", candidate.metric.min_score),
            ("max_score", candidate.metric.max_score),
        ]
    bindings[CandidateField.METRIC] = _identity_claims_binding(
        candidate,
        metric_identity,
        metric_metadata,
    )
    bindings[CandidateField.SETTING] = _setting_binding(candidate)
    bindings[CandidateField.VALUE] = _value_binding(candidate)
    bindings[CandidateField.UNIT] = _unit_binding(candidate, bindings, derived)
    return bindings


def _unit_binding(
    candidate: CandidateObservation,
    bindings: dict[CandidateField, _FieldBinding],
    derived: dict[CandidateField, str],
) -> _FieldBinding:
    sources = _unit_sources(candidate)
    reason = None
    status = FieldBindingStatus.BOUND if sources else FieldBindingStatus.UNSUPPORTED
    if not sources:
        reason = "field_not_supported_by_evidence_quotes"
    derivation = derived.get(CandidateField.UNIT)
    metric_status = bindings[CandidateField.METRIC][0]
    value_sources = bindings[CandidateField.VALUE][1]
    metric_sources = bindings[CandidateField.METRIC][1]
    if not sources and derivation and metric_status is FieldBindingStatus.BOUND:
        sources = _canonical_sources([*metric_sources, *value_sources])
        if sources:
            status = FieldBindingStatus.BOUND
            reason = derivation
    return status, sources, reason


def _field_binding(
    candidate: CandidateObservation,
    field: CandidateField,
    bindings: dict[CandidateField, _FieldBinding],
    derived: dict[CandidateField, str],
) -> _FieldBinding:
    if not _field_present(candidate, field):
        return FieldBindingStatus.UNSUPPORTED, [], "field_not_present"
    status, sources, reason = bindings[field]
    if status is FieldBindingStatus.BOUND and field in derived:
        reason = derived[field]
    return status, sources, reason


def direct_quote_field_binding(
    candidate: CandidateObservation,
    field: CandidateField,
    *,
    derived_reasons: dict[CandidateField, str] | None = None,
) -> _FieldBinding:
    """Return typed direct support for one field without cross-field association policy.

    Review code can use this on a candidate copy whose evidence contains only the exact
    attested spans.  Every primitive component is checked by the same field-specific
    predicates as ordinary validation; callers remain responsible for proving that the
    separately supported fields form one atomic tuple.
    """

    derived = derived_reasons or {}
    bindings = _direct_quote_bindings(candidate, derived=derived)
    return _field_binding(candidate, field, bindings, derived)


def direct_quote_tuple_group_association_supported(
    candidate: CandidateObservation,
    texts: Iterable[str],
) -> bool:
    """Prove one tuple relation inside a caller-validated physical span group.

    ``texts`` may contain exact spans or exact page slices formed by joining adjacent
    spans.  The caller remains responsible for validating those physical offsets.  This
    helper only accepts a group in which all populated core identities, the reported
    value, and every populated setting component form one atomic result occurrence.
    """

    for ordinal, text in enumerate(texts, start=1):
        if not text.strip():
            continue
        anchor = EvidenceAnchor(
            source_id=f"association_group_{ordinal}",
            page=1,
            kind=EvidenceKind.PROSE,
            quote=text,
        )
        grouped = candidate.model_copy(deep=True, update={"evidence": [anchor]})
        if _prose_supports_load_bearing_tuple(grouped, anchor):
            return True
    return False


def quote_field_provenance(
    candidate: CandidateObservation,
    *,
    derived_reasons: dict[CandidateField, str] | None = None,
) -> list[CandidateFieldProvenance]:
    """Bind legacy tuple fields only to quotes that directly support each field.

    Legacy proposals can see a whole result block while citing only one row.  A quote's
    occurrence therefore proves neither the surrounding header nor any other tuple field.
    Deterministic resolution may explicitly derive metric metadata or a unit, but only from
    directly supported metric/value inputs.
    """

    hashes = candidate.field_value_sha256s()
    derived = derived_reasons or {}
    provenance: list[CandidateFieldProvenance] = []
    bindings = _direct_quote_bindings(candidate, derived=derived)
    _enforce_load_bearing_tuple_association(candidate, bindings, derived=derived)

    for field in CandidateField:
        status, sources, reason = _field_binding(candidate, field, bindings, derived)
        provenance.append(
            CandidateFieldProvenance(
                field=field,
                value_sha256=hashes[field],
                status=status,
                sources=sources,
                reason=reason,
            )
        )
    return provenance


def _row_source(
    row: EnumerationRow,
    *,
    kind: FieldSourceKind,
    value: EnumerationValue | None = None,
    header_ids: list[str] | None = None,
) -> FieldSourceRef:
    return FieldSourceRef(
        kind=kind,
        source_id=row.source_id,
        page=row.page,
        region_id=row.region_id,
        planned_row_id=row.row_id,
        row_label_cell_id=(
            row.row_label_binding.cell_id
            if kind is FieldSourceKind.ROW_LABEL and row.row_label_binding is not None
            else None
        ),
        physical_cell_id=value.cell_id if value is not None else None,
        numeric_token_id=value.numeric_token_id if value is not None else None,
        header_ids=header_ids or [],
    )


def _row_context_texts(row: EnumerationRow) -> list[str]:
    return [
        text
        for text in (
            row.raw_text,
            row.caption,
            *(header.raw for level in row.headers for header in level.columns),
        )
        if text
    ]


def _row_metadata_sources(
    candidate: CandidateObservation,
    *,
    identity: object,
    label: str,
    value: object,
    row: EnumerationRow,
    header_source: FieldSourceRef,
    caption_source: FieldSourceRef,
) -> list[FieldSourceRef]:
    sources = [
        _source_for_anchor(anchor)
        for anchor in candidate.evidence
        if _entity_metadata_present(
            anchor.quote,
            identity=identity,
            label=label,
            claim=value,
        )
    ]
    if row.caption and _entity_metadata_present(
        row.caption,
        identity=identity,
        label=label,
        claim=value,
    ):
        sources.append(caption_source)
    if any(
        _entity_metadata_present(
            header.raw,
            identity=identity,
            label=label,
            claim=value,
        )
        for level in row.headers
        for header in level.columns
    ):
        sources.append(header_source)
    return _canonical_sources(sources)


def _registry_metric_metadata(
    candidate: CandidateObservation,
    row: EnumerationRow,
) -> dict[str, object]:
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
    resolved, _, _ = resolve_metric_value(
        primitive_metric,
        primitive_value,
        _row_context_texts(row),
    )
    if resolved.canonical_id is None:
        return {}
    return {
        name: getattr(resolved, name)
        for name in (
            "canonical_id",
            "kind",
            "lower_is_better",
            "min_score",
            "max_score",
        )
    }


def _row_direction_supported(row: EnumerationRow, lower_is_better: bool) -> bool:
    text = " ".join(_row_context_texts(row)).casefold()
    if lower_is_better:
        return "↓" in text or "lower is better" in text
    return "↑" in text or "higher is better" in text


def _row_unit_is_direct_or_derived(
    candidate: CandidateObservation,
    value: EnumerationValue,
    header_cells: list[EnumerationHeaderCell],
    unit: str,
) -> bool:
    direct_texts = [value.raw, value.cell_raw, *(header.raw for header in header_cells)]
    if unit == "percent" and (
        (candidate.value is not None and "%" in candidate.value.raw)
        or any("%" in text for text in direct_texts)
    ):
        return True
    if any(_phrase_present(text, unit) for text in direct_texts):
        return True
    if candidate.metric is None or candidate.value is None:
        return False
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
        direct_texts,
    )
    return bool(note and inferred_metric.unit == unit and inferred_value.unit == unit)


def row_field_provenance(
    candidate: CandidateObservation,
    row: EnumerationRow,
    value: EnumerationValue,
) -> list[CandidateFieldProvenance]:
    """Bind one row candidate and reject contradictory header/field sources."""

    evaluated = [role for role in candidate.roles if role.role is ActorRole.EVALUATED_SYSTEM]
    if len(evaluated) != 1 or not _compatible(evaluated[0].raw_name, row.row_label):
        raise ValueError("row candidate system conflicts with the physical row label")
    if any(
        binding.state in {HeaderBindingState.AMBIGUOUS, HeaderBindingState.MISSING}
        for binding in value.header_path
    ):
        raise ValueError("row candidate value has an ambiguous or missing header path")
    header_cells = [header for binding in value.header_path for header in binding.headers]
    if not header_cells:
        raise ValueError("row candidate value has no header path")
    leaf = header_cells[-1]
    if candidate.metric is None or not _compatible(candidate.metric.raw_name, leaf.raw):
        raise ValueError("row candidate metric conflicts with the value header path")
    if any(
        anchor.column is not None and not _compatible(anchor.column, leaf.raw)
        for anchor in candidate.evidence
    ):
        raise ValueError("row candidate evidence column conflicts with the header path")

    printed_percent = "%" in value.raw or any("%" in header.raw for header in header_cells)
    units = {
        unit
        for unit in (
            candidate.metric.unit if candidate.metric is not None else None,
            candidate.value.unit if candidate.value is not None else None,
        )
        if unit is not None
    }
    if len(units) > 1 or (printed_percent and units and units != {"percent"}):
        raise ValueError("row candidate unit conflicts with its printed cell/header source")

    hashes = candidate.field_value_sha256s()
    header_ids = [header.header_id for header in header_cells]
    header_source = _row_source(
        row,
        kind=FieldSourceKind.HEADER_PATH,
        header_ids=header_ids,
    )
    caption_source = _row_source(row, kind=FieldSourceKind.TABLE_CAPTION)
    value_cell_source = _row_source(row, kind=FieldSourceKind.PHYSICAL_CELL, value=value)
    numeric_source = _row_source(row, kind=FieldSourceKind.NUMERIC_TOKEN, value=value)
    row_label_source = _row_source(row, kind=FieldSourceKind.ROW_LABEL)
    value_status, _, value_reason = _value_binding(candidate)

    system_status = FieldBindingStatus.BOUND
    system_reason = None
    system_sources = [row_label_source]
    for label, metadata in (
        ("canonical_id", evaluated[0].canonical_id),
        ("version", evaluated[0].version),
        ("provider", evaluated[0].provider),
    ):
        if metadata is None or metadata == "":
            continue
        matches = _row_metadata_sources(
            candidate,
            identity=evaluated[0].raw_name,
            label=label,
            value=metadata,
            row=row,
            header_source=header_source,
            caption_source=caption_source,
        )
        if matches:
            system_sources.extend(matches)
        else:
            system_status = FieldBindingStatus.AMBIGUOUS
            system_reason = "field_partially_supported_by_physical_row"

    dataset_status = FieldBindingStatus.AMBIGUOUS
    dataset_sources = [header_source, caption_source]
    dataset_reason = "dataset_scope_uses_table_context"
    if candidate.scope is not None and any(
        _compatible(candidate.scope.dataset_raw, text)
        for text in [row.caption, *(header.raw for header in header_cells)]
        if text
    ):
        dataset_status = FieldBindingStatus.BOUND
        dataset_reason = None
        for label, metadata in (
            ("dataset_id", candidate.scope.dataset_id),
            ("dataset_url", candidate.scope.dataset_url),
            ("dataset_version", candidate.scope.dataset_version),
        ):
            if metadata is None or metadata == "":
                continue
            matches = _row_metadata_sources(
                candidate,
                identity=candidate.scope.dataset_raw,
                label=label,
                value=metadata,
                row=row,
                header_source=header_source,
                caption_source=caption_source,
            )
            if matches:
                dataset_sources.extend(matches)
            else:
                dataset_status = FieldBindingStatus.AMBIGUOUS
                dataset_reason = "field_partially_supported_by_physical_row"

    metric_status = FieldBindingStatus.BOUND
    metric_reason = None
    metric_sources = [header_source]
    registry_metadata = _registry_metric_metadata(candidate, row)
    assert candidate.metric is not None
    for label, metadata in (
        ("canonical_id", candidate.metric.canonical_id),
        ("kind", candidate.metric.kind),
        ("lower_is_better", candidate.metric.lower_is_better),
        ("min_score", candidate.metric.min_score),
        ("max_score", candidate.metric.max_score),
    ):
        if metadata is None or metadata == "":
            continue
        if registry_metadata.get(label) == metadata or (
            label == "lower_is_better"
            and isinstance(metadata, bool)
            and _row_direction_supported(row, metadata)
        ):
            continue
        matches = _row_metadata_sources(
            candidate,
            identity=candidate.metric.raw_name,
            label=label,
            value=metadata,
            row=row,
            header_source=header_source,
            caption_source=caption_source,
        )
        if matches:
            metric_sources.extend(matches)
        else:
            metric_status = FieldBindingStatus.AMBIGUOUS
            metric_reason = "field_partially_supported_by_physical_row"

    unit_status = FieldBindingStatus.UNSUPPORTED
    unit_reason = "field_not_present"
    unit_sources: list[FieldSourceRef] = []
    if units:
        unit = next(iter(units))
        if _row_unit_is_direct_or_derived(candidate, value, header_cells, unit):
            unit_status = FieldBindingStatus.BOUND
            unit_reason = None
            if printed_percent:
                unit_sources.append(numeric_source)
            if any(_phrase_present(header.raw, unit) for header in header_cells):
                unit_sources.append(header_source)
            if candidate.value is not None and _phrase_present(candidate.value.raw, unit):
                unit_sources.append(value_cell_source)
            if not unit_sources:
                unit_sources.extend([header_source, numeric_source])
        else:
            unit_reason = "field_not_supported_by_physical_row"

    return [
        CandidateFieldProvenance(
            field=CandidateField.SYSTEM,
            value_sha256=hashes[CandidateField.SYSTEM],
            status=system_status,
            sources=_canonical_sources(system_sources),
            reason=system_reason,
        ),
        CandidateFieldProvenance(
            field=CandidateField.DATASET_SCOPE,
            value_sha256=hashes[CandidateField.DATASET_SCOPE],
            status=(
                dataset_status if candidate.scope is not None else FieldBindingStatus.UNSUPPORTED
            ),
            sources=dataset_sources if candidate.scope is not None else [],
            reason=dataset_reason if candidate.scope is not None else "field_not_present",
        ),
        CandidateFieldProvenance(
            field=CandidateField.METRIC,
            value_sha256=hashes[CandidateField.METRIC],
            status=metric_status,
            sources=_canonical_sources(metric_sources),
            reason=metric_reason,
        ),
        CandidateFieldProvenance(
            field=CandidateField.SETTING,
            value_sha256=hashes[CandidateField.SETTING],
            status=(
                FieldBindingStatus.AMBIGUOUS
                if _field_present(candidate, CandidateField.SETTING)
                else FieldBindingStatus.UNSUPPORTED
            ),
            sources=(
                [header_source, caption_source]
                if _field_present(candidate, CandidateField.SETTING)
                else []
            ),
            reason=(
                "setting_uses_table_context"
                if _field_present(candidate, CandidateField.SETTING)
                else "field_not_present"
            ),
        ),
        CandidateFieldProvenance(
            field=CandidateField.VALUE,
            value_sha256=hashes[CandidateField.VALUE],
            status=value_status,
            sources=(
                [numeric_source] if value_status is not FieldBindingStatus.UNSUPPORTED else []
            ),
            reason=value_reason,
        ),
        CandidateFieldProvenance(
            field=CandidateField.UNIT,
            value_sha256=hashes[CandidateField.UNIT],
            status=unit_status,
            sources=_canonical_sources(unit_sources),
            reason=unit_reason,
        ),
    ]
