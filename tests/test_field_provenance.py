from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from proceedings_to_eee.domain.observation import (
    CandidateObservation,
    EvidenceAnchor,
    MetricSpec,
    ObservationScope,
    ReportedValue,
    RoleAssignment,
    Uncertainty,
)
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
    ValueComparator,
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.result_blocks import segment_page_result_blocks
from proceedings_to_eee.extraction.row_enumeration import (
    EnumerationRow,
    EnumerationValue,
    RowEnumerationConfig,
    build_row_enumeration_plan,
)
from proceedings_to_eee.validation.candidates import (
    resolve_references,
    validate_non_origin_candidates,
)
from proceedings_to_eee.validation.field_provenance import (
    direct_quote_tuple_group_association_supported,
    quote_field_provenance,
    row_field_provenance,
)


def _candidate(
    *,
    quote: str,
    system: str,
    provider: str | None,
    scope: ObservationScope,
    metric: MetricSpec,
    value: ReportedValue,
    construct: str | None = None,
) -> CandidateObservation:
    return CandidateObservation(
        schema_version="candidate-observation/0.3",
        paper_id="fixture-paper",
        claim_type=ClaimType.PRIMARY_RESULT,
        roles=[
            RoleAssignment(
                role=ActorRole.EVALUATED_SYSTEM,
                raw_name=system,
                provider=provider,
                confidence=0.95,
            )
        ],
        scope=scope,
        metric=metric,
        value=value,
        evidence=[
            EvidenceAnchor(
                source_id="src_fixture",
                page=1,
                kind=EvidenceKind.TABLE,
                quote=quote,
            )
        ],
        construct=construct,
        extraction_method="fixture",
        extraction_confidence=0.95,
    )


def _by_field(
    candidate: CandidateObservation,
) -> dict[CandidateField, CandidateFieldProvenance]:
    return {item.field: item for item in quote_field_provenance(candidate)}


def _validate_against_exact_quote(candidate: CandidateObservation) -> CandidateObservation:
    candidate.field_provenance = quote_field_provenance(candidate)
    [anchor] = candidate.evidence
    page_text = anchor.quote + "\n"
    layout = PdfLayout(
        source_id=anchor.source_id,
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[
            PageFragment(
                fragment_id=f"frag_{anchor.source_id}_0001",
                source_id=anchor.source_id,
                page=1,
                text=page_text,
                text_sha256=hashlib.sha256(page_text.encode()).hexdigest(),
                character_count=len(page_text),
                numeric_token_count=sum(character.isdigit() for character in page_text),
                result_signal_score=1.0,
            )
        ],
    )
    return validate_non_origin_candidates([candidate], {anchor.source_id: layout})[0]


def _verified_table_row() -> tuple[EnumerationRow, EnumerationValue]:
    text = (
        "Table 1. Dataset A results.\n"
        "System          Accuracy\n"
        "System A        0.76\n"
        "System B        0.50\n"
    )
    page = PageFragment(
        fragment_id="frag_src_fixture_0001",
        source_id="src_fixture",
        page=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_count=len(text),
        numeric_token_count=2,
        result_signal_score=5.0,
    )
    layout = PdfLayout(
        source_id="src_fixture",
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[page],
    )
    plan = build_row_enumeration_plan(
        layout,
        segment_page_result_blocks(page),
        RowEnumerationConfig(min_dense_table_rows=2),
    )
    row = next(item for item in plan.rows if item.row_label == "System A")
    return row, row.values[0]


@pytest.mark.parametrize("numeric", [float("nan"), float("inf"), float("-inf")])
def test_reported_value_rejects_nonfinite_numeric(numeric: float) -> None:
    with pytest.raises(ValidationError):
        ReportedValue(raw="nonfinite", numeric=numeric)


@pytest.mark.parametrize(
    "field",
    [
        "standard_error",
        "standard_deviation",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "confidence_level",
    ],
)
@pytest.mark.parametrize("numeric", [float("nan"), float("inf"), float("-inf")])
def test_uncertainty_rejects_nonfinite_numeric(field: str, numeric: float) -> None:
    with pytest.raises(ValidationError):
        Uncertainty.model_validate({field: numeric})


@pytest.mark.parametrize("field", ["standard_error", "standard_deviation"])
def test_uncertainty_rejects_negative_dispersion(field: str) -> None:
    with pytest.raises(ValidationError):
        Uncertainty.model_validate({field: -0.01})


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (
            _candidate(
                quote="Microsoft  75.8%  81.7%  75.7%  20.4%  28.1%",
                system="Microsoft",
                provider="Microsoft",
                scope=ObservationScope(
                    dataset_raw="Civil Comments [11]",
                    subset="Explicit / Long",
                    sample_count=50_000,
                    raw_scope="Civil Comments [11] Explicit / Long: 50,000 samples",
                ),
                metric=MetricSpec(
                    raw_name="ACC",
                    canonical_id="accuracy",
                    kind="accuracy",
                    unit="percent",
                    lower_is_better=False,
                    min_score=0,
                    max_score=100,
                ),
                value=ReportedValue(raw="75.8%", numeric=75.8, unit="percent"),
                construct="Content moderation accuracy",
            ),
            {
                CandidateField.SYSTEM: FieldBindingStatus.BOUND,
                CandidateField.DATASET_SCOPE: FieldBindingStatus.UNSUPPORTED,
                CandidateField.METRIC: FieldBindingStatus.UNSUPPORTED,
                CandidateField.SETTING: FieldBindingStatus.UNSUPPORTED,
                CandidateField.VALUE: FieldBindingStatus.BOUND,
                CandidateField.UNIT: FieldBindingStatus.BOUND,
            },
        ),
        (
            _candidate(
                quote="Perspective  87.8%  97.2%  87.7%  3.3%  20.9%",
                system="Perspective",
                provider="Google/Jigsaw",
                scope=ObservationScope(
                    dataset_raw="Civil Comments [11]",
                    subset="Explicit / Long",
                    sample_count=50_000,
                    raw_scope="Civil Comments [11] Explicit / Long: 50,000 samples",
                ),
                metric=MetricSpec(
                    raw_name="AUC",
                    canonical_id="auroc",
                    kind="auroc",
                    unit="percent",
                    lower_is_better=False,
                    min_score=0,
                    max_score=100,
                ),
                value=ReportedValue(raw="97.2%", numeric=97.2, unit="percent"),
                construct="Area under the ROC curve",
            ),
            {
                CandidateField.SYSTEM: FieldBindingStatus.AMBIGUOUS,
                CandidateField.DATASET_SCOPE: FieldBindingStatus.UNSUPPORTED,
                CandidateField.METRIC: FieldBindingStatus.UNSUPPORTED,
                CandidateField.SETTING: FieldBindingStatus.UNSUPPORTED,
                CandidateField.VALUE: FieldBindingStatus.BOUND,
                CandidateField.UNIT: FieldBindingStatus.BOUND,
            },
        ),
        (
            _candidate(
                quote="Baseline  0.960  0.952",
                system="Baseline",
                provider=None,
                scope=ObservationScope(
                    dataset_raw="general",
                    split="test",
                    aggregation="mean",
                    raw_scope="general test sets",
                ),
                metric=MetricSpec(
                    raw_name="Mean AUC",
                    canonical_id="auroc",
                    kind="performance",
                    lower_is_better=False,
                    min_score=0,
                    max_score=1,
                ),
                value=ReportedValue(raw="0.960", numeric=0.96),
            ),
            {
                CandidateField.SYSTEM: FieldBindingStatus.BOUND,
                CandidateField.DATASET_SCOPE: FieldBindingStatus.UNSUPPORTED,
                CandidateField.METRIC: FieldBindingStatus.UNSUPPORTED,
                CandidateField.SETTING: FieldBindingStatus.UNSUPPORTED,
                CandidateField.VALUE: FieldBindingStatus.BOUND,
                CandidateField.UNIT: FieldBindingStatus.UNSUPPORTED,
            },
        ),
        (
            _candidate(
                quote="Hateful  Yes  397  0.30  0.74  0.43  0.76",
                system="ChatGPT",
                provider="OpenAI",
                scope=ObservationScope(
                    dataset_raw="HOT based on Prompt 1",
                    subset="Hateful",
                    raw_scope="Category Hateful",
                ),
                metric=MetricSpec(
                    raw_name="Accuracy",
                    canonical_id="accuracy",
                    kind="accuracy",
                    unit="proportion",
                    lower_is_better=False,
                    min_score=0,
                    max_score=1,
                ),
                value=ReportedValue(raw="0.76", numeric=0.76, unit="proportion"),
                construct="Hateful content classification",
            ),
            {
                CandidateField.SYSTEM: FieldBindingStatus.UNSUPPORTED,
                CandidateField.DATASET_SCOPE: FieldBindingStatus.UNSUPPORTED,
                CandidateField.METRIC: FieldBindingStatus.UNSUPPORTED,
                CandidateField.SETTING: FieldBindingStatus.UNSUPPORTED,
                CandidateField.VALUE: FieldBindingStatus.BOUND,
                CandidateField.UNIT: FieldBindingStatus.UNSUPPORTED,
            },
        ),
        (
            _candidate(
                quote="0.80  0.753  0.551  0.636  0.510  0.583  0.544  0.740  0.481  0.583",
                system="Perspective Severe Toxicity",
                provider=None,
                scope=ObservationScope(
                    dataset_raw="Telegram data",
                    language="English",
                    raw_scope="English",
                ),
                metric=MetricSpec(
                    raw_name="F1",
                    canonical_id="f1",
                    kind="f1",
                    unit="proportion",
                    lower_is_better=False,
                    min_score=0,
                    max_score=1,
                    parameters={"threshold": 0.8},
                ),
                value=ReportedValue(raw="0.636", numeric=0.636, unit="proportion"),
                construct="F1 score of Perspective Severe Toxicity at threshold 0.80",
            ),
            {
                CandidateField.SYSTEM: FieldBindingStatus.UNSUPPORTED,
                CandidateField.DATASET_SCOPE: FieldBindingStatus.UNSUPPORTED,
                CandidateField.METRIC: FieldBindingStatus.UNSUPPORTED,
                CandidateField.SETTING: FieldBindingStatus.UNSUPPORTED,
                CandidateField.VALUE: FieldBindingStatus.BOUND,
                CandidateField.UNIT: FieldBindingStatus.UNSUPPORTED,
            },
        ),
    ],
    ids=["item-1", "item-2", "item-5", "item-8", "item-9"],
)
def test_legacy_row_quote_does_not_bind_omitted_context(
    candidate: CandidateObservation,
    expected: dict[CandidateField, FieldBindingStatus],
) -> None:
    provenance = _by_field(candidate)

    assert {field: item.status for field, item in provenance.items()} == expected
    assert provenance[CandidateField.VALUE].sources
    for field, status in expected.items():
        if status is FieldBindingStatus.UNSUPPORTED:
            assert provenance[field].sources == []


def test_claimed_row_id_does_not_replace_atomic_tuple_support() -> None:
    candidate = _candidate(
        quote="Atlas Moderation API  74.6%",
        system="Atlas Moderation API",
        provider=None,
        scope=ObservationScope(dataset_raw="Civil Comments", split="test"),
        metric=MetricSpec(raw_name="AUC", unit="percent"),
        value=ReportedValue(raw="74.6%", numeric=74.6, unit="percent"),
    )
    candidate.evidence[0].planned_row_id = "row_atlas"
    candidate.evidence.extend(
        [
            EvidenceAnchor(
                source_id="src_fixture",
                page=1,
                kind=EvidenceKind.TABLE,
                quote="Moderation Service  AUC",
                planned_row_id="row_atlas",
            ),
            EvidenceAnchor(
                source_id="src_fixture",
                page=1,
                kind=EvidenceKind.TABLE,
                quote="Civil Comments split test",
                planned_row_id="row_atlas",
            ),
        ]
    )

    provenance = _by_field(candidate)
    hashes = {anchor.quote: anchor.quote_sha256 for anchor in candidate.evidence}

    assert all(
        provenance[field].status is FieldBindingStatus.AMBIGUOUS
        for field in (
            CandidateField.SYSTEM,
            CandidateField.DATASET_SCOPE,
            CandidateField.METRIC,
            CandidateField.VALUE,
        )
    )
    assert provenance[CandidateField.SETTING].status is FieldBindingStatus.AMBIGUOUS
    assert provenance[CandidateField.SETTING].reason == (
        "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    )
    assert provenance[CandidateField.UNIT].status is FieldBindingStatus.AMBIGUOUS
    assert provenance[CandidateField.UNIT].reason == (
        "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    )
    assert [source.quote_sha256 for source in provenance[CandidateField.SYSTEM].sources] == [
        hashes["Atlas Moderation API  74.6%"]
    ]
    assert [source.quote_sha256 for source in provenance[CandidateField.METRIC].sources] == [
        hashes["Moderation Service  AUC"]
    ]
    assert [source.quote_sha256 for source in provenance[CandidateField.DATASET_SCOPE].sources] == [
        hashes["Civil Comments split test"]
    ]
    assert [source.quote_sha256 for source in provenance[CandidateField.SETTING].sources] == [
        hashes["Civil Comments split test"]
    ]
    assert [source.quote_sha256 for source in provenance[CandidateField.VALUE].sources] == [
        hashes["Atlas Moderation API  74.6%"]
    ]


def test_system_canonical_id_cannot_borrow_metric_label() -> None:
    candidate = _candidate(
        quote="Atlas 0.76",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.roles[0] = candidate.roles[0].model_copy(update={"canonical_id": "accuracy"})
    candidate.evidence.append(
        EvidenceAnchor(
            source_id="src_fixture",
            page=1,
            kind=EvidenceKind.TABLE,
            quote="Dataset A Accuracy",
        )
    )

    binding = _by_field(candidate)[CandidateField.SYSTEM]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert [source.quote_sha256 for source in binding.sources] == [
        candidate.evidence[0].quote_sha256
    ]


def test_system_metadata_cannot_cross_another_result_value() -> None:
    candidate = _candidate(
        quote=("Atlas Dataset A Accuracy 0.76 Baseline canonical id baseline/model"),
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.roles[0] = candidate.roles[0].model_copy(update={"canonical_id": "baseline/model"})

    binding = _by_field(candidate)[CandidateField.SYSTEM]

    assert binding.status is FieldBindingStatus.AMBIGUOUS


def test_system_metadata_cannot_cross_an_unclaimed_actor_without_a_number() -> None:
    candidate = _candidate(
        quote="Atlas Dataset A Accuracy 0.76 score",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.roles[0] = candidate.roles[0].model_copy(update={"canonical_id": "baseline/model"})
    candidate.evidence[0].kind = EvidenceKind.PROSE
    candidate.evidence.append(
        EvidenceAnchor(
            source_id="src_fixture",
            page=1,
            kind=EvidenceKind.PROSE,
            quote="Atlas and Baseline canonical id baseline/model",
        )
    )

    binding = _by_field(candidate)[CandidateField.SYSTEM]

    assert binding.status is FieldBindingStatus.AMBIGUOUS


def test_exact_system_provider_self_identification_remains_supported() -> None:
    candidate = _candidate(
        quote="Microsoft Dataset A Accuracy 0.76 score",
        system="Microsoft",
        provider="Microsoft",
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    binding = _by_field(candidate)[CandidateField.SYSTEM]

    assert binding.status is FieldBindingStatus.BOUND


def test_dataset_id_cannot_borrow_metric_label() -> None:
    candidate = _candidate(
        quote="Atlas 0.76",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A", dataset_id="accuracy"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.evidence.append(
        EvidenceAnchor(
            source_id="src_fixture",
            page=1,
            kind=EvidenceKind.TABLE,
            quote="Dataset A",
        )
    )
    candidate.evidence.append(
        EvidenceAnchor(
            source_id="src_fixture",
            page=1,
            kind=EvidenceKind.TABLE,
            quote="Accuracy",
        )
    )

    binding = _by_field(candidate)[CandidateField.DATASET_SCOPE]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert [source.quote_sha256 for source in binding.sources] == [
        candidate.evidence[1].quote_sha256
    ]


def test_reference_resolution_recomputes_stale_legacy_provenance() -> None:
    candidate = _candidate(
        quote="Hateful  Yes  397  0.30  0.74  0.43  0.76",
        system="ChatGPT",
        provider="OpenAI",
        scope=ObservationScope(
            dataset_raw="HOT based on Prompt 1",
            subset="Hateful",
            raw_scope="Category Hateful",
        ),
        metric=MetricSpec(
            raw_name="Accuracy",
            canonical_id="accuracy",
            lower_is_better=False,
            min_score=0,
            max_score=1,
        ),
        value=ReportedValue(raw="0.76", numeric=0.76),
        construct="Hateful content classification",
    )
    hashes = candidate.field_value_sha256s()
    source = FieldSourceRef(
        kind=FieldSourceKind.EVIDENCE_QUOTE,
        source_id="src_fixture",
        page=1,
        quote_sha256=candidate.evidence[0].quote_sha256,
    )
    stale = [
        CandidateFieldProvenance(
            field=field,
            value_sha256=hashes[field],
            status=(
                FieldBindingStatus.UNSUPPORTED
                if field is CandidateField.UNIT
                else FieldBindingStatus.BOUND
            ),
            sources=[] if field is CandidateField.UNIT else [source],
            reason="field_not_present" if field is CandidateField.UNIT else None,
        )
        for field in CandidateField
    ]
    payload = candidate.model_dump(mode="json")
    payload["field_provenance"] = [item.model_dump(mode="json") for item in stale]
    candidate = CandidateObservation.model_validate(payload)

    resolve_references(candidate)

    provenance = {item.field: item for item in candidate.field_provenance}
    assert candidate.metric is not None and candidate.metric.kind == "accuracy"
    assert candidate.metric.unit == "proportion"
    assert candidate.value is not None and candidate.value.unit == "proportion"
    assert provenance[CandidateField.SYSTEM].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.DATASET_SCOPE].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.METRIC].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.SETTING].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.VALUE].status is FieldBindingStatus.BOUND
    assert provenance[CandidateField.UNIT].status is FieldBindingStatus.UNSUPPORTED
    assert all(
        item.value_sha256 == candidate.field_value_sha256s()[item.field]
        for item in candidate.field_provenance
    )


def test_reference_resolution_records_a_supported_derived_unit_explicitly() -> None:
    candidate = _candidate(
        quote="ChatGPT HOT Prompt 1 Accuracy 0.76",
        system="ChatGPT",
        provider=None,
        scope=ObservationScope(dataset_raw="HOT Prompt 1"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE
    candidate.field_provenance = quote_field_provenance(candidate)

    resolve_references(candidate)

    provenance = {item.field: item for item in candidate.field_provenance}
    assert provenance[CandidateField.METRIC].status is FieldBindingStatus.BOUND
    assert provenance[CandidateField.METRIC].reason == (
        "deterministic_reference_resolution=metric_registry"
    )
    assert provenance[CandidateField.UNIT].status is FieldBindingStatus.BOUND
    assert provenance[CandidateField.UNIT].reason == (
        "deterministic_reference_resolution="
        "unit_resolved_as_proportion_from_registry_bounded_metric_and_printed_decimal"
    )
    assert {source.quote_sha256 for source in provenance[CandidateField.UNIT].sources} == {
        candidate.evidence[0].quote_sha256,
    }

    resolve_references(candidate)
    replayed = {item.field: item for item in candidate.field_provenance}
    assert replayed[CandidateField.METRIC].status is FieldBindingStatus.BOUND
    assert replayed[CandidateField.METRIC].reason == (
        "deterministic_reference_resolution=metric_registry"
    )
    assert replayed[CandidateField.UNIT].status is FieldBindingStatus.BOUND
    assert replayed[CandidateField.UNIT].reason == provenance[CandidateField.UNIT].reason


def test_reference_resolution_refreshes_quotes_per_field_in_mixed_provenance() -> None:
    candidate = _candidate(
        quote="0.76",
        system="ChatGPT",
        provider=None,
        scope=ObservationScope(dataset_raw="HOT Prompt 1"),
        metric=MetricSpec(
            raw_name="Accuracy",
            canonical_id="accuracy",
            kind="accuracy",
            unit="proportion",
            lower_is_better=False,
            min_score=0,
            max_score=1,
        ),
        value=ReportedValue(raw="0.76", numeric=0.76, unit="proportion"),
    )
    hashes = candidate.field_value_sha256s()
    stale_quote = FieldSourceRef(
        kind=FieldSourceKind.EVIDENCE_QUOTE,
        source_id="src_fixture",
        page=1,
        quote_sha256=candidate.evidence[0].quote_sha256,
    )
    dataset_header = FieldSourceRef(
        kind=FieldSourceKind.HEADER_PATH,
        source_id="src_fixture",
        page=1,
        header_ids=["header_dataset"],
    )
    candidate.field_provenance = [
        CandidateFieldProvenance(
            field=field,
            value_sha256=hashes[field],
            status=(
                FieldBindingStatus.UNSUPPORTED
                if field is CandidateField.SETTING
                else FieldBindingStatus.BOUND
            ),
            sources=(
                []
                if field is CandidateField.SETTING
                else (
                    [stale_quote, dataset_header]
                    if field is CandidateField.DATASET_SCOPE
                    else [stale_quote]
                )
            ),
            reason="field_not_present" if field is CandidateField.SETTING else None,
        )
        for field in CandidateField
    ]

    resolve_references(candidate)

    provenance = {item.field: item for item in candidate.field_provenance}
    assert provenance[CandidateField.SYSTEM].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.SYSTEM].sources == []
    assert provenance[CandidateField.METRIC].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.METRIC].sources == []
    assert provenance[CandidateField.UNIT].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.UNIT].sources == []
    assert provenance[CandidateField.DATASET_SCOPE].status is FieldBindingStatus.BOUND
    assert provenance[CandidateField.DATASET_SCOPE].sources == [dataset_header]
    assert provenance[CandidateField.VALUE].status is FieldBindingStatus.BOUND
    assert provenance[CandidateField.VALUE].sources == [stale_quote]


def test_reference_resolution_discards_forged_review_span_bindings() -> None:
    candidate = _candidate(
        quote="0.76",
        system="ChatGPT",
        provider=None,
        scope=ObservationScope(dataset_raw="HOT Prompt 1"),
        metric=MetricSpec(raw_name="Accuracy", unit="proportion"),
        value=ReportedValue(raw="0.76", numeric=0.76, unit="proportion"),
    )
    hashes = candidate.field_value_sha256s()
    forged = FieldSourceRef(
        kind=FieldSourceKind.REVIEW_SPAN,
        source_id="src_fixture",
        page=1,
        quote_sha256=candidate.evidence[0].quote_sha256,
        review_span_id="span_" + "a" * 64,
    )
    candidate.field_provenance = [
        CandidateFieldProvenance(
            field=field,
            value_sha256=hashes[field],
            status=(
                FieldBindingStatus.UNSUPPORTED
                if field is CandidateField.SETTING
                else FieldBindingStatus.BOUND
            ),
            sources=[] if field is CandidateField.SETTING else [forged],
            reason="field_not_present" if field is CandidateField.SETTING else None,
        )
        for field in CandidateField
    ]

    resolve_references(candidate)

    provenance = {item.field: item for item in candidate.field_provenance}
    assert all(
        source.kind is not FieldSourceKind.REVIEW_SPAN
        for binding in provenance.values()
        for source in binding.sources
    )
    assert provenance[CandidateField.SYSTEM].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.DATASET_SCOPE].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.METRIC].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.UNIT].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.VALUE].status is FieldBindingStatus.BOUND


def test_reference_resolution_never_cures_a_collapsed_conflict_hash() -> None:
    candidate = _candidate(
        quote="System A  0.76",
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(
            raw_name="Accuracy",
            canonical_id="provider_accuracy",
            kind="provider_kind",
            lower_is_better=True,
            min_score=-1,
            max_score=1,
        ),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    resolved_projection = candidate.model_copy(
        update={
            "metric": MetricSpec(
                raw_name="Accuracy",
                canonical_id="accuracy",
                kind="accuracy",
                unit="proportion",
                lower_is_better=False,
                min_score=0,
                max_score=1,
            ),
            "value": ReportedValue(raw="0.76", numeric=0.76, unit="proportion"),
        }
    )
    future_metric_hash = resolved_projection.field_value_sha256s()[CandidateField.METRIC]
    source = FieldSourceRef(
        kind=FieldSourceKind.HEADER_PATH,
        source_id="src_fixture",
        page=1,
        header_ids=["header_accuracy"],
    )
    hashes = candidate.field_value_sha256s()
    candidate.field_provenance = [
        CandidateFieldProvenance(
            field=field,
            value_sha256=hashes[field],
            status=(
                FieldBindingStatus.CONFLICT
                if field is CandidateField.METRIC
                else FieldBindingStatus.BOUND
            ),
            sources=[source],
            alternate_value_sha256s=(
                [future_metric_hash] if field is CandidateField.METRIC else []
            ),
            reason="fixture metric conflict" if field is CandidateField.METRIC else None,
        )
        for field in CandidateField
    ]

    resolve_references(candidate)

    provenance = {item.field: item for item in candidate.field_provenance}
    metric = provenance[CandidateField.METRIC]
    assert metric.status is FieldBindingStatus.CONFLICT
    assert metric.alternate_value_sha256s == [hashes[CandidateField.METRIC]]
    assert metric.reason == "fixture metric conflict"
    assert metric.value_sha256 == candidate.field_value_sha256s()[CandidateField.METRIC]


def test_reference_resolution_never_cures_quote_only_conflict() -> None:
    candidate = _candidate(
        quote="System A Accuracy 0.76",
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    provenance = quote_field_provenance(candidate)
    candidate.field_provenance = [
        (
            item.model_copy(
                update={
                    "status": FieldBindingStatus.CONFLICT,
                    "alternate_value_sha256s": ["f" * 64],
                    "reason": "fixture value conflict",
                }
            )
            if item.field is CandidateField.VALUE
            else item
        )
        for item in provenance
    ]

    resolve_references(candidate)

    value = {item.field: item for item in candidate.field_provenance}[CandidateField.VALUE]
    assert value.status is FieldBindingStatus.CONFLICT
    assert value.alternate_value_sha256s == ["f" * 64]
    assert value.reason == "fixture value conflict"


@pytest.mark.parametrize(
    "value",
    [
        ReportedValue(
            raw="0.76",
            numeric=0.76,
            comparator=ValueComparator.LESS_THAN,
        ),
        ReportedValue(
            raw="0.76",
            numeric=0.76,
            uncertainty=Uncertainty(
                confidence_interval_lower=0.70,
                confidence_interval_upper=0.80,
                confidence_level=0.95,
            ),
        ),
    ],
    ids=["fabricated-comparator", "fabricated-confidence-interval"],
)
def test_value_quote_does_not_bind_fabricated_qualifiers(value: ReportedValue) -> None:
    candidate = _candidate(
        quote="System A AUC 0.76",
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="AUC"),
        value=value,
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert binding.reason == "value_comparator_or_uncertainty_not_supported_by_evidence_quotes"


@pytest.mark.parametrize(
    "uncertainty",
    [
        Uncertainty(standard_error=2),
        Uncertainty(standard_deviation=2),
    ],
    ids=["unlabeled-standard-error", "unlabeled-standard-deviation"],
)
def test_bare_plus_minus_does_not_bind_typed_uncertainty(
    uncertainty: Uncertainty,
) -> None:
    candidate = _candidate(
        quote="System A Accuracy 0.76 ± 2",
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76, uncertainty=uncertainty),
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert binding.reason == "value_comparator_or_uncertainty_not_supported_by_evidence_quotes"


def test_uncertainty_does_not_borrow_number_from_another_statistic() -> None:
    candidate = _candidate(
        quote="System A Accuracy 0.76; SD 2; SE 3",
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(
            raw="0.76",
            numeric=0.76,
            uncertainty=Uncertainty(standard_error=2),
        ),
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert binding.reason == "value_comparator_or_uncertainty_not_supported_by_evidence_quotes"


def test_uncertainty_does_not_cross_an_intervening_actor_boundary() -> None:
    candidate = _candidate(
        quote="Atlas Accuracy 0.76; baseline SE: 2",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(
            raw="0.76",
            numeric=0.76,
            uncertainty=Uncertainty(standard_error=2),
        ),
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert binding.reason == "value_comparator_or_uncertainty_not_supported_by_evidence_quotes"


def test_uncertainty_method_does_not_borrow_from_another_actor() -> None:
    candidate = _candidate(
        quote="Atlas Accuracy 0.76 with SE 2; Baseline uses bootstrap",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(
            raw="0.76",
            numeric=0.76,
            uncertainty=Uncertainty(standard_error=2, method="bootstrap"),
        ),
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert binding.reason == "value_comparator_or_uncertainty_not_supported_by_evidence_quotes"


def test_local_uncertainty_method_and_statistic_bind() -> None:
    candidate = _candidate(
        quote="Atlas Accuracy 0.76 with bootstrap SE 2",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(
            raw="0.76",
            numeric=0.76,
            uncertainty=Uncertainty(standard_error=2, method="bootstrap"),
        ),
    )

    assert _by_field(candidate)[CandidateField.VALUE].status is FieldBindingStatus.BOUND


def test_confidence_interval_does_not_cross_an_intervening_value_boundary() -> None:
    candidate = _candidate(
        quote="Atlas Accuracy 0.76; baseline 0.50 (95% CI 0.40-0.60)",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(
            raw="0.76",
            numeric=0.76,
            uncertainty=Uncertainty(
                confidence_interval_lower=0.40,
                confidence_interval_upper=0.60,
                confidence_level=0.95,
            ),
        ),
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert binding.reason == "value_comparator_or_uncertainty_not_supported_by_evidence_quotes"


def test_explicit_labeled_standard_error_binds() -> None:
    candidate = _candidate(
        quote="System A Accuracy 0.76; SE=2",
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(
            raw="0.76",
            numeric=0.76,
            uncertainty=Uncertainty(standard_error=2),
        ),
    )

    assert _by_field(candidate)[CandidateField.VALUE].status is FieldBindingStatus.BOUND


@pytest.mark.parametrize(
    "quote",
    [
        "System A Accuracy < 0.76",
        "System A Accuracy <= 0.76",
        "System A Accuracy > 0.76",
        "System A Accuracy >= 0.76",
        "System A Accuracy approximately 0.76",
        "System A Accuracy less than or equal to 0.76",
        "System A Accuracy roughly 0.76",
        "System A Accuracy 0.76; alarm below 0.76",
    ],
)
def test_exact_value_does_not_bind_nonexact_source_occurrence(quote: str) -> None:
    candidate = _candidate(
        quote=quote,
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert binding.reason == "value_comparator_or_uncertainty_not_supported_by_evidence_quotes"


def test_value_quote_does_not_bind_fabricated_numeric_projection() -> None:
    candidate = _candidate(
        quote="System A AUC 0.76",
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="AUC"),
        value=ReportedValue(raw="0.76", numeric=0.99),
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.UNSUPPORTED
    assert binding.reason == "value_numeric_projection_not_supported_by_raw"


def test_value_projection_does_not_silently_rescale_printed_percent() -> None:
    candidate = _candidate(
        quote="System A Accuracy 73.4%",
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy", unit="percent"),
        value=ReportedValue(raw="73.4%", numeric=0.734, unit="percent"),
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.UNSUPPORTED
    assert binding.reason == "value_numeric_projection_not_supported_by_raw"


def test_row_value_does_not_bind_fabricated_numeric_projection() -> None:
    text = (
        "Table 1. Dataset A results.\n"
        "System          Accuracy\n"
        "System A        0.76\n"
        "System B        0.50\n"
    )
    page = PageFragment(
        fragment_id="frag_src_fixture_0001",
        source_id="src_fixture",
        page=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_count=len(text),
        numeric_token_count=2,
        result_signal_score=5.0,
    )
    layout = PdfLayout(
        source_id="src_fixture",
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[page],
    )
    plan = build_row_enumeration_plan(
        layout,
        segment_page_result_blocks(page),
        RowEnumerationConfig(min_dense_table_rows=2),
    )
    row = next(item for item in plan.rows if item.row_label == "System A")
    candidate = _candidate(
        quote=row.raw_text,
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.99),
    )

    binding = {item.field: item for item in row_field_provenance(candidate, row, row.values[0])}[
        CandidateField.VALUE
    ]

    assert binding.status is FieldBindingStatus.UNSUPPORTED
    assert binding.reason == "value_numeric_projection_not_supported_by_raw"


def test_row_system_identity_does_not_bind_unsupported_metadata() -> None:
    row, value = _verified_table_row()
    candidate = _candidate(
        quote=row.raw_text,
        system="System A",
        provider="Fabricated Provider",
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.roles[0].canonical_id = "fabricated/system"
    candidate.roles[0].version = "999"

    binding = {item.field: item for item in row_field_provenance(candidate, row, value)}[
        CandidateField.SYSTEM
    ]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert binding.reason == "field_partially_supported_by_physical_row"


def test_row_dataset_identity_does_not_bind_unsupported_metadata() -> None:
    row, value = _verified_table_row()
    candidate = _candidate(
        quote=row.raw_text,
        system="System A",
        provider=None,
        scope=ObservationScope(
            dataset_raw="Dataset A",
            dataset_id="fabricated-dataset",
            dataset_url="https://example.invalid/fabricated",
            dataset_version="999",
        ),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )

    binding = {item.field: item for item in row_field_provenance(candidate, row, value)}[
        CandidateField.DATASET_SCOPE
    ]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert binding.reason == "field_partially_supported_by_physical_row"


def test_row_metric_identity_does_not_bind_unsupported_metadata() -> None:
    row, value = _verified_table_row()
    candidate = _candidate(
        quote=row.raw_text,
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(
            raw_name="Accuracy",
            canonical_id="f1",
            kind="f1",
            lower_is_better=True,
            min_score=-1,
            max_score=999,
        ),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )

    binding = {item.field: item for item in row_field_provenance(candidate, row, value)}[
        CandidateField.METRIC
    ]

    assert binding.status is FieldBindingStatus.AMBIGUOUS
    assert binding.reason == "field_partially_supported_by_physical_row"


def test_row_unit_does_not_bind_absent_incompatible_unit() -> None:
    row, value = _verified_table_row()
    candidate = _candidate(
        quote=row.raw_text,
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy", unit="seconds"),
        value=ReportedValue(raw="0.76", numeric=0.76, unit="seconds"),
    )

    binding = {item.field: item for item in row_field_provenance(candidate, row, value)}[
        CandidateField.UNIT
    ]

    assert binding.status is FieldBindingStatus.UNSUPPORTED
    assert binding.reason == "field_not_supported_by_physical_row"


def test_verified_row_provenance_remains_bound_after_reference_resolution() -> None:
    text = (
        "Table 1. Dataset A results.\n"
        "System          Accuracy\n"
        "System A        0.76\n"
        "System B        0.50\n"
    )
    page = PageFragment(
        fragment_id="frag_src_fixture_0001",
        source_id="src_fixture",
        page=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_count=len(text),
        numeric_token_count=2,
        result_signal_score=5.0,
    )
    layout = PdfLayout(
        source_id="src_fixture",
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[page],
    )
    plan = build_row_enumeration_plan(
        layout,
        segment_page_result_blocks(page),
        RowEnumerationConfig(min_dense_table_rows=2),
    )
    row = next(item for item in plan.rows if item.row_label == "System A")
    candidate = _candidate(
        quote=row.raw_text,
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.field_provenance = row_field_provenance(candidate, row, row.values[0])

    resolve_references(candidate)
    provenance = {item.field: item for item in candidate.field_provenance}

    assert all(
        provenance[field].status is FieldBindingStatus.BOUND
        for field in (
            CandidateField.SYSTEM,
            CandidateField.DATASET_SCOPE,
            CandidateField.METRIC,
            CandidateField.VALUE,
            CandidateField.UNIT,
        )
    )
    assert all(
        any(
            source.kind is not FieldSourceKind.EVIDENCE_QUOTE
            for source in provenance[field].sources
        )
        for field in (
            CandidateField.SYSTEM,
            CandidateField.DATASET_SCOPE,
            CandidateField.METRIC,
            CandidateField.VALUE,
        )
    )


def test_nonnumeric_raw_is_a_noncurable_projection_failure_before_identity() -> None:
    candidate = _candidate(
        quote="Atlas Dataset A Custom foobar",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Custom"),
        value=ReportedValue(raw="foo", numeric=123),
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.UNSUPPORTED
    assert binding.sources == []
    assert binding.reason == "value_numeric_projection_not_supported_by_raw"


@pytest.mark.parametrize(
    ("raw", "numeric"),
    [
        ("95% CI [0.7, 0.8]", 95),
        ("range 0.7-0.8", 0.7),
        ("n=95; score 0.7", 0.7),
        ("n=95", 95),
    ],
    ids=["confidence-interval", "range", "sample-and-score", "sample-count"],
)
def test_multi_number_or_typed_statistic_raw_is_not_a_point_estimate(
    raw: str,
    numeric: float,
) -> None:
    candidate = _candidate(
        quote=f"Atlas Dataset A Accuracy {raw}",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy", unit="percent"),
        value=ReportedValue(raw=raw, numeric=numeric, unit="percent"),
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.UNSUPPORTED
    assert binding.sources == []
    assert binding.reason == "value_numeric_projection_not_supported_by_raw"


def test_value_quote_binds_explicit_comparator_and_confidence_interval() -> None:
    candidate = _candidate(
        quote="System A AUC <0.76 (95% CI 0.70-0.80)",
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="AUC"),
        value=ReportedValue(
            raw="0.76",
            numeric=0.76,
            comparator=ValueComparator.LESS_THAN,
            uncertainty=Uncertainty(
                confidence_interval_lower=0.70,
                confidence_interval_upper=0.80,
                confidence_level=0.95,
            ),
        ),
    )

    binding = _by_field(candidate)[CandidateField.VALUE]

    assert binding.status is FieldBindingStatus.BOUND
    assert binding.reason is None
    assert [source.quote_sha256 for source in binding.sources] == [
        candidate.evidence[0].quote_sha256
    ]


def test_validation_withholds_bounded_metric_with_dimensional_unit() -> None:
    quote = "System A Dataset A test Accuracy 74.6 seconds"
    candidate = _candidate(
        quote=quote,
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A", split="test"),
        metric=MetricSpec(raw_name="Accuracy", unit="seconds"),
        value=ReportedValue(raw="74.6", numeric=74.6, unit="seconds"),
    )
    candidate.field_provenance = quote_field_provenance(candidate)
    page_text = quote + "\n"
    layout = PdfLayout(
        source_id="src_fixture",
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[
            PageFragment(
                fragment_id="frag_src_fixture_0001",
                source_id="src_fixture",
                page=1,
                text=page_text,
                text_sha256=hashlib.sha256(page_text.encode()).hexdigest(),
                character_count=len(page_text),
                numeric_token_count=1,
                result_signal_score=1.0,
            )
        ],
    )

    validated = validate_non_origin_candidates(
        [candidate],
        {"src_fixture": layout},
    )[0]
    provenance = {item.field: item for item in validated.field_provenance}

    assert validated.referential_status is ReferentialStatus.UNRESOLVED
    assert validated.export_status is ExportStatus.NEEDS_REVIEW
    assert validated.export_reason == "referential_status=unresolved"
    assert provenance[CandidateField.UNIT].status is FieldBindingStatus.UNSUPPORTED
    assert provenance[CandidateField.UNIT].reason == (
        "incompatible_unit_for_registry_bounded_metric"
    )


def test_validation_withholds_unsupported_evaluation_instrument() -> None:
    quote = "System A Dataset A Accuracy proportion 0.76"
    candidate = _candidate(
        quote=quote,
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy", unit="proportion"),
        value=ReportedValue(raw="0.76", numeric=0.76, unit="proportion"),
    )
    candidate.roles.append(
        RoleAssignment(
            role=ActorRole.EVALUATION_INSTRUMENT,
            raw_name="GPT-4",
            confidence=0.99,
        )
    )
    candidate.field_provenance = quote_field_provenance(candidate)
    page_text = quote + "\n"
    layout = PdfLayout(
        source_id="src_fixture",
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[
            PageFragment(
                fragment_id="frag_src_fixture_0001",
                source_id="src_fixture",
                page=1,
                text=page_text,
                text_sha256=hashlib.sha256(page_text.encode()).hexdigest(),
                character_count=len(page_text),
                numeric_token_count=1,
                result_signal_score=1.0,
            )
        ],
    )

    validated = validate_non_origin_candidates(
        [candidate],
        {"src_fixture": layout},
    )[0]
    setting = {item.field: item for item in validated.field_provenance}[CandidateField.SETTING]

    assert CandidateField.SETTING in validated.populated_fields()
    assert setting.status is FieldBindingStatus.UNSUPPORTED
    assert validated.export_status is ExportStatus.NEEDS_REVIEW
    assert validated.export_reason == "field_provenance=ambiguous_or_unsupported"


@pytest.mark.parametrize(
    ("raw", "numeric", "unit"),
    [("146%", 146.0, "percent"), ("1.46", 1.46, "proportion")],
)
def test_validation_withholds_out_of_range_bounded_metric_value(
    raw: str,
    numeric: float,
    unit: str,
) -> None:
    quote = f"System A Dataset A test Accuracy {raw}"
    candidate = _candidate(
        quote=quote,
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A", split="test"),
        metric=MetricSpec(raw_name="Accuracy", unit=unit),
        value=ReportedValue(raw=raw, numeric=numeric, unit=unit),
    )
    candidate.field_provenance = quote_field_provenance(candidate)
    page_text = quote + "\n"
    layout = PdfLayout(
        source_id="src_fixture",
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[
            PageFragment(
                fragment_id="frag_src_fixture_0001",
                source_id="src_fixture",
                page=1,
                text=page_text,
                text_sha256=hashlib.sha256(page_text.encode()).hexdigest(),
                character_count=len(page_text),
                numeric_token_count=1,
                result_signal_score=1.0,
            )
        ],
    )

    validated = validate_non_origin_candidates(
        [candidate],
        {"src_fixture": layout},
    )[0]
    value_binding = {item.field: item for item in validated.field_provenance}[CandidateField.VALUE]

    assert validated.referential_status is ReferentialStatus.UNRESOLVED
    assert validated.export_status is ExportStatus.NEEDS_REVIEW
    assert validated.export_reason == "referential_status=unresolved"
    assert value_binding.status is FieldBindingStatus.UNSUPPORTED
    assert value_binding.reason == "value_outside_registry_metric_bounds"


def test_validation_withholds_out_of_range_confidence_interval() -> None:
    quote = "System A Dataset A test Accuracy 0.76 (95% CI 0.70-1.20)"
    candidate = _candidate(
        quote=quote,
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A", split="test"),
        metric=MetricSpec(raw_name="Accuracy", unit="proportion"),
        value=ReportedValue(
            raw="0.76",
            numeric=0.76,
            unit="proportion",
            uncertainty=Uncertainty(
                confidence_interval_lower=0.70,
                confidence_interval_upper=1.20,
                confidence_level=0.95,
            ),
        ),
    )
    candidate.field_provenance = quote_field_provenance(candidate)

    resolve_references(candidate)

    value_binding = {item.field: item for item in candidate.field_provenance}[CandidateField.VALUE]
    assert candidate.referential_status is ReferentialStatus.UNRESOLVED
    assert value_binding.status is FieldBindingStatus.UNSUPPORTED
    assert value_binding.reason == "uncertainty_interval_outside_registry_metric_bounds"


def test_reference_resolution_marks_custom_metric_explicit_range_violation() -> None:
    candidate = _candidate(
        quote="System A Dataset A Custom bounded score 2",
        system="System A",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(
            raw_name="Custom bounded score",
            min_score=0,
            max_score=1,
        ),
        value=ReportedValue(raw="2", numeric=2),
    )
    candidate.field_provenance = quote_field_provenance(candidate)

    resolve_references(candidate)

    value_binding = {item.field: item for item in candidate.field_provenance}[CandidateField.VALUE]
    assert candidate.referential_status is ReferentialStatus.UNRESOLVED
    assert value_binding.status is FieldBindingStatus.UNSUPPORTED
    assert value_binding.reason == "value_outside_registry_metric_bounds"


def test_sample_count_does_not_borrow_a_metric_range_endpoint() -> None:
    candidate = _candidate(
        quote="Atlas Dataset A Accuracy range 0-100 0.76",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A", sample_count=100),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )

    setting = _by_field(candidate)[CandidateField.SETTING]

    assert setting.status is FieldBindingStatus.UNSUPPORTED
    assert setting.sources == []


def test_scope_setting_does_not_cross_an_unclaimed_actor() -> None:
    candidate = _candidate(
        quote="Atlas Dataset A Accuracy 0.76 score",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A", split="dev"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE
    candidate.evidence.append(
        EvidenceAnchor(
            source_id="src_fixture",
            page=1,
            kind=EvidenceKind.PROSE,
            quote="Dataset A and Baseline split dev",
        )
    )

    setting = _by_field(candidate)[CandidateField.SETTING]

    assert setting.status is FieldBindingStatus.UNSUPPORTED
    assert setting.sources == []


def test_scope_setting_binds_only_in_same_atomic_result_occurrence() -> None:
    candidate = _candidate(
        quote="Atlas Dataset A Accuracy 0.76 split dev",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A", split="dev"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    provenance = _by_field(candidate)

    assert provenance[CandidateField.SETTING].status is FieldBindingStatus.BOUND
    assert all(
        provenance[field].status is FieldBindingStatus.BOUND
        for field in (
            CandidateField.SYSTEM,
            CandidateField.DATASET_SCOPE,
            CandidateField.METRIC,
            CandidateField.SETTING,
            CandidateField.VALUE,
        )
    )


@pytest.mark.parametrize(
    "quote",
    [
        "Atlas D1 Accuracy 0.76 and Baseline Accuracy split dev",
        "Atlas D1 Accuracy 0.76 and Baseline D1 split dev",
        "Atlas D1 Accuracy 0.76 and Baseline 0.76 split dev",
    ],
    ids=["repeated-metric", "repeated-dataset", "repeated-value"],
)
def test_setting_cannot_borrow_from_later_actor_in_same_prose_anchor(quote: str) -> None:
    candidate = _candidate(
        quote=quote,
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D1", split="dev"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    provenance = _by_field(candidate)

    assert provenance[CandidateField.SETTING].status is FieldBindingStatus.AMBIGUOUS
    assert provenance[CandidateField.SETTING].reason == (
        "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    )


def test_unit_cannot_borrow_from_later_actor_in_same_prose_anchor() -> None:
    candidate = _candidate(
        quote=(
            "Atlas D1 Score canonical id score lower is better false 0.76 "
            "and Baseline latency 50 seconds"
        ),
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D1"),
        metric=MetricSpec(
            raw_name="Score",
            canonical_id="score",
            unit="seconds",
            lower_is_better=False,
        ),
        value=ReportedValue(raw="0.76", numeric=0.76, unit="seconds"),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    provenance = _by_field(candidate)

    assert provenance[CandidateField.UNIT].status is FieldBindingStatus.AMBIGUOUS
    assert provenance[CandidateField.UNIT].reason == (
        "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    )


def test_metric_metadata_cannot_borrow_from_later_actor_in_same_prose_anchor() -> None:
    candidate = _candidate(
        quote=(
            "Atlas D1 Score 0.76% and Baseline Score canonical id baseline-score "
            "lower is better false"
        ),
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D1"),
        metric=MetricSpec(
            raw_name="Score",
            canonical_id="baseline-score",
            unit="percent",
            lower_is_better=False,
        ),
        value=ReportedValue(raw="0.76%", numeric=0.76, unit="percent"),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    provenance = _by_field(candidate)

    assert provenance[CandidateField.METRIC].status is FieldBindingStatus.AMBIGUOUS
    assert provenance[CandidateField.METRIC].reason == (
        "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    )


def test_dataset_metadata_cannot_borrow_from_later_actor_in_same_prose_anchor() -> None:
    candidate = _candidate(
        quote=(
            "Atlas D1 Score canonical id score lower is better false 0.76% "
            "and Baseline D1 dataset id baseline-data"
        ),
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D1", dataset_id="baseline-data"),
        metric=MetricSpec(
            raw_name="Score",
            canonical_id="score",
            unit="percent",
            lower_is_better=False,
        ),
        value=ReportedValue(raw="0.76%", numeric=0.76, unit="percent"),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    provenance = _by_field(candidate)

    assert provenance[CandidateField.DATASET_SCOPE].status is FieldBindingStatus.AMBIGUOUS
    assert provenance[CandidateField.DATASET_SCOPE].reason == (
        "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    )


def test_system_metadata_cannot_borrow_from_later_actor_in_same_prose_anchor() -> None:
    candidate = _candidate(
        quote=(
            "Atlas D1 Score canonical id score lower is better false 0.76% "
            "and Baseline Atlas canonical id baseline-model"
        ),
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D1"),
        metric=MetricSpec(
            raw_name="Score",
            canonical_id="score",
            unit="percent",
            lower_is_better=False,
        ),
        value=ReportedValue(raw="0.76%", numeric=0.76, unit="percent"),
    )
    candidate.roles[0] = candidate.roles[0].model_copy(update={"canonical_id": "baseline-model"})
    candidate.evidence[0].kind = EvidenceKind.PROSE

    provenance = _by_field(candidate)

    assert provenance[CandidateField.SYSTEM].status is FieldBindingStatus.AMBIGUOUS
    assert provenance[CandidateField.SYSTEM].reason == (
        "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    )


@pytest.mark.parametrize("connector", ["but", "however", "against"])
def test_lowercase_actor_setting_borrow_is_not_eligible(connector: str) -> None:
    candidate = _candidate(
        quote=f"atlas d1 accuracy 0.76 {connector} challenger accuracy split dev",
        system="atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="d1", split="dev"),
        metric=MetricSpec(raw_name="accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    validated = _validate_against_exact_quote(candidate)
    provenance = {item.field: item for item in validated.field_provenance}

    assert validated.referential_status is ReferentialStatus.RESOLVED
    assert validated.export_status is ExportStatus.NEEDS_REVIEW
    assert validated.export_reason == "field_provenance=ambiguous_or_unsupported"
    assert provenance[CandidateField.SETTING].status is FieldBindingStatus.AMBIGUOUS
    assert provenance[CandidateField.SETTING].reason == (
        "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    )


@pytest.mark.parametrize("connector", ["but", "however", "against"])
def test_lowercase_actor_metric_metadata_borrow_is_not_eligible(connector: str) -> None:
    candidate = _candidate(
        quote=(
            f"atlas d1 score 0.76% {connector} challenger score canonical id "
            "challenger-score lower is better false"
        ),
        system="atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="d1"),
        metric=MetricSpec(
            raw_name="score",
            canonical_id="challenger-score",
            unit="percent",
            lower_is_better=False,
        ),
        value=ReportedValue(raw="0.76%", numeric=0.76, unit="percent"),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    validated = _validate_against_exact_quote(candidate)
    provenance = {item.field: item for item in validated.field_provenance}

    assert validated.referential_status is ReferentialStatus.RESOLVED
    assert validated.export_status is ExportStatus.NEEDS_REVIEW
    assert validated.export_reason == "field_provenance=ambiguous_or_unsupported"
    assert provenance[CandidateField.METRIC].status is FieldBindingStatus.AMBIGUOUS
    assert provenance[CandidateField.METRIC].reason == (
        "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    )


@pytest.mark.parametrize("connector", ["but", "however", "against"])
def test_lowercase_actor_unit_borrow_is_not_eligible(connector: str) -> None:
    candidate = _candidate(
        quote=(
            "atlas d1 score canonical id score lower is better false 0.76 "
            f"{connector} challenger latency seconds"
        ),
        system="atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="d1"),
        metric=MetricSpec(
            raw_name="score",
            canonical_id="score",
            unit="seconds",
            lower_is_better=False,
        ),
        value=ReportedValue(raw="0.76", numeric=0.76, unit="seconds"),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    validated = _validate_against_exact_quote(candidate)
    provenance = {item.field: item for item in validated.field_provenance}

    assert validated.referential_status is ReferentialStatus.RESOLVED
    assert validated.export_status is ExportStatus.NEEDS_REVIEW
    assert validated.export_reason == "field_provenance=ambiguous_or_unsupported"
    assert provenance[CandidateField.UNIT].status is FieldBindingStatus.AMBIGUOUS
    assert provenance[CandidateField.UNIT].reason == (
        "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    )


def test_attested_span_group_rejects_a_two_row_frankenstein_tuple() -> None:
    candidate = _candidate(
        quote="Atlas D1 Accuracy 80.0%",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D2"),
        metric=MetricSpec(raw_name="F1", unit="percent"),
        value=ReportedValue(raw="80.0%", numeric=80.0, unit="percent"),
    )

    assert not direct_quote_tuple_group_association_supported(
        candidate,
        ["Atlas D1 Accuracy 80.0%\nBaseline D2 F1 70.0%"],
    )


def test_attested_span_group_allows_adjacent_header_and_result_row() -> None:
    candidate = _candidate(
        quote="Atlas 80.0%",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D2"),
        metric=MetricSpec(raw_name="Accuracy", unit="percent"),
        value=ReportedValue(raw="80.0%", numeric=80.0, unit="percent"),
    )

    assert direct_quote_tuple_group_association_supported(
        candidate,
        ["Dataset D2 Accuracy\nAtlas 80.0%"],
    )


def test_attested_span_group_allows_exact_hyphenated_system_line_wrap() -> None:
    candidate = _candidate(
        quote="ToxicBert 0.76",
        system="ToxicBert",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )

    assert direct_quote_tuple_group_association_supported(
        candidate,
        ["Dataset A Accuracy agreement is 0.76 with Tox-\nicBert"],
    )
    assert not direct_quote_tuple_group_association_supported(
        candidate,
        ["Dataset A Accuracy agreement is 0.76 with Tox-\nOtherBert"],
    )


def test_metric_parameter_key_and_value_cannot_bind_across_anchors() -> None:
    candidate = _candidate(
        quote="Atlas Dataset A Accuracy 0.76",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy", parameters={"threshold": 0.8}),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.evidence.extend(
        [
            EvidenceAnchor(
                source_id="src_fixture",
                page=1,
                kind=EvidenceKind.TABLE,
                quote="threshold",
            ),
            EvidenceAnchor(
                source_id="src_fixture",
                page=1,
                kind=EvidenceKind.TABLE,
                quote="0.8",
            ),
        ]
    )

    setting = _by_field(candidate)[CandidateField.SETTING]

    assert setting.status is FieldBindingStatus.UNSUPPORTED
    assert setting.sources == []


def test_metric_parameter_binds_when_key_and_value_are_adjacent() -> None:
    candidate = _candidate(
        quote="Atlas Dataset A Accuracy 0.76 threshold=0.8",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy", parameters={"threshold": 0.8}),
        value=ReportedValue(raw="0.76", numeric=0.76),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    assert _by_field(candidate)[CandidateField.SETTING].status is FieldBindingStatus.BOUND


def test_percent_unit_does_not_borrow_an_identical_baseline_occurrence() -> None:
    candidate = _candidate(
        quote="Atlas Dataset A Accuracy 0.76; baseline 0.76%",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy", unit="percent"),
        value=ReportedValue(raw="0.76", numeric=0.76, unit="percent"),
    )

    unit = _by_field(candidate)[CandidateField.UNIT]

    assert unit.status is FieldBindingStatus.UNSUPPORTED
    assert unit.sources == []


@pytest.mark.parametrize(
    "quote",
    [
        "Atlas D1 Accuracy 0.8; Baseline D2 F1 0.7",
        "Atlas D1 Accuracy 0.8 Baseline D2 F1 0.7",
        "Atlas D1 Accuracy 0.8, Baseline D2 F1 0.7",
        "Atlas D1 Accuracy 0.8 / Baseline D2 F1 0.7",
    ],
    ids=["semicolon", "whitespace", "comma", "slash"],
)
def test_load_bearing_tuple_cannot_splice_across_result_clauses(quote: str) -> None:
    candidate = _candidate(
        quote=quote,
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D2"),
        metric=MetricSpec(raw_name="F1"),
        value=ReportedValue(raw="0.7", numeric=0.7),
    )
    candidate.field_provenance = quote_field_provenance(candidate)
    initial = {item.field: item for item in candidate.field_provenance}

    assert all(
        initial[field].status is FieldBindingStatus.AMBIGUOUS
        and initial[field].reason
        == "load_bearing_tuple_association_not_supported_by_evidence_quotes"
        for field in (
            CandidateField.SYSTEM,
            CandidateField.DATASET_SCOPE,
            CandidateField.METRIC,
            CandidateField.VALUE,
        )
    )

    page_text = quote + "\n"
    layout = PdfLayout(
        source_id="src_fixture",
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[
            PageFragment(
                fragment_id="frag_src_fixture_0001",
                source_id="src_fixture",
                page=1,
                text=page_text,
                text_sha256=hashlib.sha256(page_text.encode()).hexdigest(),
                character_count=len(page_text),
                numeric_token_count=2,
                result_signal_score=1.0,
            )
        ],
    )

    validated = validate_non_origin_candidates([candidate], {"src_fixture": layout})[0]

    assert validated.referential_status is ReferentialStatus.RESOLVED
    assert validated.export_status is ExportStatus.NEEDS_REVIEW
    assert validated.export_reason == "field_provenance=ambiguous_or_unsupported"


def test_metric_and_value_cannot_splice_across_same_row_metric_cells() -> None:
    candidate = _candidate(
        quote="Atlas D1 Accuracy 0.8 F1 0.7",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D1"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.7", numeric=0.7),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    provenance = _by_field(candidate)

    assert all(
        provenance[field].status is FieldBindingStatus.AMBIGUOUS
        and provenance[field].reason
        == "load_bearing_tuple_association_not_supported_by_evidence_quotes"
        for field in (
            CandidateField.SYSTEM,
            CandidateField.DATASET_SCOPE,
            CandidateField.METRIC,
            CandidateField.VALUE,
        )
    )


def test_forged_planned_row_id_cannot_prove_a_table_tuple() -> None:
    candidate = _candidate(
        quote="Atlas D1 Accuracy 0.8 Baseline D2 F1 0.7",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D2"),
        metric=MetricSpec(raw_name="F1"),
        value=ReportedValue(raw="0.7", numeric=0.7),
    )
    candidate.evidence[0].planned_row_id = "row_forged"

    provenance = _by_field(candidate)

    assert all(
        provenance[field].status is FieldBindingStatus.AMBIGUOUS
        for field in (
            CandidateField.SYSTEM,
            CandidateField.DATASET_SCOPE,
            CandidateField.METRIC,
            CandidateField.VALUE,
        )
    )


def test_prose_tuple_cannot_cross_an_unscored_actor_clause() -> None:
    candidate = _candidate(
        quote=(
            "We tested Atlas on D1 using Accuracy and Baseline on D2 using F1 obtained 0.7 score"
        ),
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D2"),
        metric=MetricSpec(raw_name="F1"),
        value=ReportedValue(raw="0.7", numeric=0.7),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    provenance = _by_field(candidate)

    assert all(
        provenance[field].status is FieldBindingStatus.AMBIGUOUS
        for field in (
            CandidateField.SYSTEM,
            CandidateField.DATASET_SCOPE,
            CandidateField.METRIC,
            CandidateField.VALUE,
        )
    )


def test_one_clause_load_bearing_tuple_remains_bound() -> None:
    candidate = _candidate(
        quote="Atlas D2 F1 0.7",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D2"),
        metric=MetricSpec(raw_name="F1"),
        value=ReportedValue(raw="0.7", numeric=0.7),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE

    provenance = _by_field(candidate)

    assert all(
        provenance[field].status is FieldBindingStatus.BOUND
        for field in (
            CandidateField.SYSTEM,
            CandidateField.DATASET_SCOPE,
            CandidateField.METRIC,
            CandidateField.VALUE,
        )
    )


def test_no_scope_tuple_cannot_splice_system_metric_and_value() -> None:
    candidate = CandidateObservation(
        schema_version="candidate-observation/0.3",
        paper_id="fixture-paper",
        claim_type=ClaimType.SECONDARY_CLAIM,
        roles=[
            RoleAssignment(
                role=ActorRole.EVALUATED_SYSTEM,
                raw_name="Atlas",
                confidence=0.95,
            )
        ],
        metric=MetricSpec(raw_name="F1"),
        value=ReportedValue(raw="0.7", numeric=0.7),
        evidence=[
            EvidenceAnchor(
                source_id="src_fixture",
                page=1,
                kind=EvidenceKind.PROSE,
                quote="Atlas Accuracy 0.8 Baseline F1 0.7",
            )
        ],
        extraction_confidence=0.95,
    )

    provenance = _by_field(candidate)

    assert all(
        provenance[field].status is FieldBindingStatus.AMBIGUOUS
        and provenance[field].reason
        == "load_bearing_tuple_association_not_supported_by_evidence_quotes"
        for field in (
            CandidateField.SYSTEM,
            CandidateField.METRIC,
            CandidateField.VALUE,
        )
    )


def test_non_evaluated_role_must_share_the_atomic_result_occurrence() -> None:
    candidate = _candidate(
        quote="Atlas D1 Accuracy 0.8; Baseline uses GPT-4",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="D1"),
        metric=MetricSpec(raw_name="Accuracy"),
        value=ReportedValue(raw="0.8", numeric=0.8),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE
    candidate.roles.append(
        RoleAssignment(
            role=ActorRole.EVALUATION_INSTRUMENT,
            raw_name="GPT-4",
            confidence=0.95,
        )
    )

    provenance = _by_field(candidate)

    assert provenance[CandidateField.SETTING].status is FieldBindingStatus.AMBIGUOUS
    assert provenance[CandidateField.SETTING].reason == (
        "load_bearing_tuple_association_not_supported_by_evidence_quotes"
    )


def test_legacy_setting_hash_remains_parseable_and_is_refreshed() -> None:
    candidate = _candidate(
        quote="Atlas Dataset A Accuracy using GPT-4 proportion 0.76",
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy", unit="proportion"),
        value=ReportedValue(raw="0.76", numeric=0.76, unit="proportion"),
    )
    candidate.roles.append(
        RoleAssignment(
            role=ActorRole.EVALUATION_INSTRUMENT,
            raw_name="GPT-4",
            confidence=0.95,
        )
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE
    candidate.field_provenance = quote_field_provenance(candidate)
    payload = candidate.model_dump(mode="json")
    legacy_setting_hash = candidate._legacy_setting_value_sha256()
    for item in payload["field_provenance"]:
        if item["field"] == CandidateField.SETTING:
            item["value_sha256"] = legacy_setting_hash

    restored = CandidateObservation.model_validate(payload)
    resolve_references(restored)
    setting = {item.field: item for item in restored.field_provenance}[CandidateField.SETTING]

    assert setting.value_sha256 == restored.field_value_sha256s()[CandidateField.SETTING]
    assert setting.value_sha256 != legacy_setting_hash
    assert setting.status is FieldBindingStatus.BOUND


def test_legacy_setting_hash_cannot_launder_an_unevidenced_new_role() -> None:
    quote = "Atlas Dataset A Accuracy 76%"
    candidate = _candidate(
        quote=quote,
        system="Atlas",
        provider=None,
        scope=ObservationScope(dataset_raw="Dataset A"),
        metric=MetricSpec(raw_name="Accuracy", unit="percent"),
        value=ReportedValue(raw="76%", numeric=76, unit="percent"),
    )
    candidate.evidence[0].kind = EvidenceKind.PROSE
    candidate.roles.append(
        RoleAssignment(
            role=ActorRole.EVALUATION_INSTRUMENT,
            raw_name="GPT-4",
            confidence=0.95,
        )
    )
    provenance = quote_field_provenance(candidate)
    legacy_source = FieldSourceRef(
        kind=FieldSourceKind.TABLE_CAPTION,
        source_id="src_fixture",
        page=1,
    )
    candidate.field_provenance = [
        (
            item.model_copy(
                update={
                    "value_sha256": candidate._legacy_setting_value_sha256(),
                    "status": FieldBindingStatus.BOUND,
                    "sources": [legacy_source],
                    "reason": None,
                }
            )
            if item.field is CandidateField.SETTING
            else item
        )
        for item in provenance
    ]
    payload = candidate.model_dump(mode="json")
    restored = CandidateObservation.model_validate(payload)

    page_text = quote + "\n"
    layout = PdfLayout(
        source_id="src_fixture",
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[
            PageFragment(
                fragment_id="frag_src_fixture_0001",
                source_id="src_fixture",
                page=1,
                text=page_text,
                text_sha256=hashlib.sha256(page_text.encode()).hexdigest(),
                character_count=len(page_text),
                numeric_token_count=1,
                result_signal_score=1.0,
            )
        ],
    )

    validated = validate_non_origin_candidates([restored], {"src_fixture": layout})[0]
    setting = {item.field: item for item in validated.field_provenance}[CandidateField.SETTING]

    assert setting.value_sha256 == validated.field_value_sha256s()[CandidateField.SETTING]
    assert setting.status is FieldBindingStatus.UNSUPPORTED
    assert setting.sources == []
    assert validated.export_status is ExportStatus.NEEDS_REVIEW
    assert validated.export_reason == "field_provenance=ambiguous_or_unsupported"
