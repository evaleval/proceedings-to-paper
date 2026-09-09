from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from proceedings_to_eee.domain.attribution import AttributionState, AttributionVerdict
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
    ValueComparator,
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from proceedings_to_eee.resources import EEE_SCHEMA_SHA256
from proceedings_to_eee.reviewed_export import workflow as reviewed_workflow
from proceedings_to_eee.reviewed_export.models import (
    DecisionAuthority,
    DecisionStatus,
    DerivedRunManifest,
    OriginDecision,
    ReviewAuthorityMode,
    ReviewedExportDecision,
    ReviewedExportProvenance,
    ReviewEvidenceAnchor,
    ReviewFieldAttestation,
    ReviewItem,
    ReviewManifest,
    TupleDecision,
    model_payload,
)
from proceedings_to_eee.reviewed_export.workflow import (
    DERIVED_MANIFEST_NAME,
    DERIVED_OUTCOMES_NAME,
    DERIVED_PROVENANCE_NAME,
    DERIVED_SHA256SUMS_NAME,
    DERIVED_VERIFICATION_NAME,
    REVIEW_DECISIONS_NAME,
    REVIEW_ITEMS_NAME,
    REVIEW_LOCK_NAME,
    REVIEW_MANIFEST_NAME,
    ReviewedExportError,
    ReviewedExportErrorCode,
    compose_reviewed_eee,
    prepare_export_review,
    validate_export_review,
    verify_contextual_derived_run,
    verify_derived_run,
)
from proceedings_to_eee.run_seal import RUN_SEAL_NAME, seal_run_tree, verify_run_seal
from proceedings_to_eee.sources.manifest import (
    FrozenSource,
    LicenseDisposition,
    SourceManifest,
    SourceRole,
)

PAPER_ID = "synthetic-two-page"
SOURCE_ID = "src_synthetic"
RESULT_TEXT = (
    "Table 1. Synthetic Benchmark split test evaluation (AUC)\n"
    "Atlas Moderation API  74.6%\n"
    "Atlas Moderation API alternate F1 74.6%\n"
    "Atlas Moderation API Synthetic Benchmark split test AUC 72.3% SE 1.2% bootstrap\n"
    "Atlas Moderation API Synthetic Benchmark split test AUC threshold < 73.2%\n"
    "Atlas Moderation API foobar percent result was observed.\n"
    "Atlas Moderation API foo percent result was observed.\n"
    "Atlas Moderation API D1 Accuracy 80.0%\n"
    "Baseline D2 F1 70.0%\n"
    "Atlas Moderation API AUC 70.0%\n"
    "Atlas Moderation API Synthetic Benchmark 70.0%\n"
    "Evaluation instrument ToxicBert.\n"
    "Synthetic Benchmark split test Agreement is 96.1% with Tox-\n"
    "icBert on the benchmark.\n"
    "GPT-4 result on Synthetic Benchmark split test AUC 74.6%\n"
    "GPT-4.1 result on Synthetic Benchmark split test AUC 74.6%\n"
    "Table 2. Synthetic Benchmark split test AUC range 0-100\n"
    "Atlas Moderation API .76; baseline 10%\n"
    "  Atlas Moderation API whitespace Synthetic Benchmark split test AUC 88.8%\n"
    "Atlas Moderation API 95% CI [0.7,0.8]\n"
    "Atlas Moderation API 0.7762 (0.12 %)\n"
    "Atlas Moderation API Synthetic Benchmark split test AUC 72.3% with bootstrap\n"
    "Atlas Moderation API Synthetic Benchmark split test AUC 72.3%; Baseline bootstrap\n"
    "Atlas Moderation API Synthetic Benchmark split test CustomScore 74.6%\n"
    "Atlas Moderation API D3 Accuracy 81.0% score\n"
    "Atlas Moderation API Synthetic Benchmark split test AUC 74.6%\n"
    "                 Dataset                Characteristics          "
    "Moderation Service       ACC        AUC         F1        FPR       FNR\n"
    "                                        Explicit / Long:\n"
    "                 Civil Comments [11]\n"
    "                                        50,000 samples\n"
    "                                                                       "
    "Microsoft          75.8%      81.7%     75.7%      20.4%     28.1%\n"
    "              Dataset       AUC      Pinned AUC                           "
    "Model              General    Phrase Templates\n"
    "              Combined      0.79         N/A                              "
    "Baseline            0.960          0.952\n"
    "                                                                  "
    "Table 4: Mean AUC on the general and phrase templates test\n"
    "                    Table 3: AUC results.                         sets.\n"
    "5.2.3 Results. Table 2 reports results on the JMTCC dataset. Our"
    "                    unrelated right-column prose\n"
    "results show that our best UTC† achieves 0.9367 AUC-ROC, outper-"
    "                    unrelated continuation\n"
)
ORIGIN_TEXT = (
    "Methods\n"
    "We evaluated Atlas Moderation API ourselves on the synthetic benchmark.\n"
    "We evaluated ToxicBert ourselves on the synthetic benchmark.\n"
    "We evaluated GPT-4.1 ourselves on the synthetic benchmark.\n"
    "We evaluated Microsoft ourselves on Civil Comments.\n"
    "We trained and evaluated the Baseline ourselves.\n"
    "We present and evaluate UTC† ourselves.\n"
)

MICROSOFT_HEADER = "Moderation Service       ACC        AUC         F1        FPR       FNR"
MICROSOFT_SUBSET = "Explicit / Long:"
MICROSOFT_DATASET = "Civil Comments [11]"
MICROSOFT_SAMPLES = "50,000 samples"
MICROSOFT_ROW = "Microsoft          75.8%      81.7%     75.7%      20.4%     28.1%"
BASELINE_HEADER = "Model              General    Phrase Templates"
BASELINE_ROW = "Baseline            0.960          0.952"
BASELINE_CAPTION = "Table 4: Mean AUC on the general and phrase templates test"
BASELINE_CAPTION_CONTINUATION = "                         sets."
UTC_SCOPE = "5.2.3 Results. Table 2 reports results on the JMTCC dataset. Our"
UTC_RESULT = "results show that our best UTC† achieves 0.9367 AUC-ROC, outper-"


def _page(number: int, text: str) -> PageFragment:
    return PageFragment(
        fragment_id=f"page-{number}",
        source_id=SOURCE_ID,
        page=number,
        text=text,
        text_sha256=sha256_bytes(text.encode("utf-8")),
        character_count=len(text),
        numeric_token_count=1 if number == 2 else 0,
        result_signal_score=5.0 if number == 2 else 0.0,
    )


def _candidate(
    *, confidence: float = 0.99, attribution: AttributionState = AttributionState.NO_SIGNAL
) -> CandidateObservation:
    return CandidateObservation(
        paper_id=PAPER_ID,
        claim_type=ClaimType.PRIMARY_RESULT,
        roles=[
            RoleAssignment(
                role=ActorRole.EVALUATED_SYSTEM,
                raw_name="Atlas Moderation API",
                confidence=1.0,
            )
        ],
        scope=ObservationScope(dataset_raw="Synthetic Benchmark", split="test"),
        metric=MetricSpec(
            raw_name="AUC",
            canonical_id="auroc",
            kind="auroc",
            unit="percent",
            lower_is_better=False,
            min_score=0,
            max_score=100,
        ),
        value=ReportedValue(raw="74.6%", numeric=74.6, unit="percent"),
        evidence=[
            EvidenceAnchor(
                source_id=SOURCE_ID,
                page=2,
                kind=EvidenceKind.TABLE,
                label="Table 1",
                row="Atlas Moderation API",
                column="AUC",
                quote="Atlas Moderation API  74.6%",
            ),
            EvidenceAnchor(
                source_id=SOURCE_ID,
                page=2,
                kind=EvidenceKind.TABLE,
                label="Table 1",
                quote="Table 1. Synthetic Benchmark split test evaluation (AUC)",
            ),
        ],
        export_status=ExportStatus.NEEDS_REVIEW,
        attribution=AttributionVerdict(state=attribution, rule_id="synthetic_existing_cue"),
        extraction_method="fixture",
        extraction_confidence=confidence,
    )


def _with_quote_field_provenance(
    candidate: CandidateObservation,
    sources_by_field: dict[CandidateField, list[EvidenceAnchor]],
) -> CandidateObservation:
    """Attach explicit quote bindings to a synthetic candidate fixture."""

    hashes = candidate.field_value_sha256s()
    provenance: list[CandidateFieldProvenance] = []
    for field in CandidateField:
        anchors = sources_by_field.get(field, [])
        provenance.append(
            CandidateFieldProvenance(
                field=field,
                value_sha256=hashes[field],
                status=(FieldBindingStatus.BOUND if anchors else FieldBindingStatus.UNSUPPORTED),
                sources=[
                    FieldSourceRef(
                        kind=FieldSourceKind.EVIDENCE_QUOTE,
                        source_id=anchor.source_id,
                        page=anchor.page,
                        quote_sha256=anchor.quote_sha256,
                    )
                    for anchor in anchors
                ],
                reason=None if anchors else "not_bound_in_fixture",
            )
        )
    payload = model_payload(candidate)
    payload["schema_version"] = "candidate-observation/0.3"
    payload["field_provenance"] = [model_payload(item) for item in provenance]
    return CandidateObservation.model_validate(payload)


def _metric_context_candidate() -> CandidateObservation:
    context = EvidenceAnchor(
        source_id=SOURCE_ID,
        page=2,
        kind=EvidenceKind.TABLE,
        label="Table 1",
        quote="Table 1. Synthetic Benchmark split test evaluation (AUC)",
    )
    row = EvidenceAnchor(
        source_id=SOURCE_ID,
        page=2,
        kind=EvidenceKind.TABLE,
        label="Table 1",
        row="Atlas Moderation API",
        quote="Atlas Moderation API  74.6%",
    )
    payload = model_payload(_candidate())
    payload["observation_id"] = None
    payload["evidence"] = [model_payload(context), model_payload(row)]
    candidate = CandidateObservation.model_validate(payload)
    return _with_quote_field_provenance(
        candidate,
        {
            CandidateField.SYSTEM: [row],
            CandidateField.DATASET_SCOPE: [row],
            CandidateField.METRIC: [context],
            CandidateField.SETTING: [row],
            CandidateField.VALUE: [row],
            CandidateField.UNIT: [row],
        },
    )


def _production_microsoft_table_candidate() -> CandidateObservation:
    anchors = [
        EvidenceAnchor(
            source_id=SOURCE_ID,
            page=2,
            kind=EvidenceKind.TABLE,
            quote=quote,
        )
        for quote in (
            MICROSOFT_HEADER,
            MICROSOFT_SUBSET,
            MICROSOFT_DATASET,
            MICROSOFT_SAMPLES,
            MICROSOFT_ROW,
        )
    ]
    payload = model_payload(_candidate())
    payload.update(
        {
            "observation_id": None,
            "roles": [
                model_payload(
                    RoleAssignment(
                        role=ActorRole.EVALUATED_SYSTEM,
                        raw_name="Microsoft",
                        provider="Microsoft",
                        confidence=1.0,
                    )
                )
            ],
            "scope": model_payload(
                ObservationScope(
                    dataset_raw="Civil Comments [11]",
                    subset="Explicit / Long",
                    sample_count=50_000,
                    raw_scope="Civil Comments [11] Explicit / Long: 50,000 samples",
                )
            ),
            "metric": model_payload(
                MetricSpec(
                    raw_name="ACC",
                    canonical_id="accuracy",
                    kind="accuracy",
                    unit="percent",
                    lower_is_better=False,
                    min_score=0,
                    max_score=100,
                )
            ),
            "value": model_payload(ReportedValue(raw="75.8%", numeric=75.8, unit="percent")),
            "evidence": [model_payload(anchor) for anchor in anchors],
            "construct": "Content moderation accuracy",
            "operationalization": None,
        }
    )
    candidate = CandidateObservation.model_validate(payload)
    return _with_quote_field_provenance(
        candidate,
        {
            CandidateField.SYSTEM: [anchors[-1]],
            CandidateField.VALUE: [anchors[-1]],
            CandidateField.UNIT: [anchors[-1]],
        },
    )


def _production_baseline_table_candidate() -> CandidateObservation:
    anchors = [
        EvidenceAnchor(
            source_id=SOURCE_ID,
            page=2,
            kind=EvidenceKind.TABLE,
            quote=quote,
        )
        for quote in (
            BASELINE_HEADER,
            BASELINE_ROW,
            BASELINE_CAPTION,
            BASELINE_CAPTION_CONTINUATION,
        )
    ]
    payload = model_payload(_candidate())
    payload.update(
        {
            "observation_id": None,
            "roles": [
                model_payload(
                    RoleAssignment(
                        role=ActorRole.EVALUATED_SYSTEM,
                        raw_name="Baseline",
                        confidence=0.95,
                    )
                )
            ],
            "scope": model_payload(
                ObservationScope(
                    dataset_raw="general",
                    split="test",
                    aggregation="mean",
                    raw_scope="general test sets",
                )
            ),
            "metric": model_payload(
                MetricSpec(
                    raw_name="Mean AUC",
                    canonical_id="auroc",
                    kind="performance",
                    lower_is_better=False,
                    min_score=0,
                    max_score=1,
                )
            ),
            "value": model_payload(ReportedValue(raw="0.960", numeric=0.96)),
            "evidence": [model_payload(anchor) for anchor in anchors],
            "construct": None,
            "operationalization": None,
        }
    )
    candidate = CandidateObservation.model_validate(payload)
    return _with_quote_field_provenance(
        candidate,
        {
            CandidateField.SYSTEM: [anchors[1]],
            CandidateField.DATASET_SCOPE: [anchors[2], anchors[3]],
            CandidateField.METRIC: [anchors[2], anchors[3]],
            CandidateField.VALUE: [anchors[0], anchors[1]],
        },
    )


def _production_utc_prose_candidate() -> CandidateObservation:
    scope_anchor = EvidenceAnchor(
        source_id=SOURCE_ID,
        page=2,
        kind=EvidenceKind.PROSE,
        quote=UTC_SCOPE,
    )
    result_anchor = EvidenceAnchor(
        source_id=SOURCE_ID,
        page=2,
        kind=EvidenceKind.PROSE,
        quote=UTC_RESULT,
    )
    payload = model_payload(_candidate())
    payload.update(
        {
            "observation_id": None,
            "roles": [
                model_payload(
                    RoleAssignment(
                        role=ActorRole.EVALUATED_SYSTEM,
                        raw_name="UTC†",
                        confidence=0.95,
                    )
                )
            ],
            "scope": model_payload(ObservationScope(dataset_raw="JMTCC dataset")),
            "metric": model_payload(
                MetricSpec(
                    raw_name="AUC-ROC",
                    canonical_id="auroc",
                    kind="auroc",
                    unit="proportion",
                    lower_is_better=False,
                    min_score=0,
                    max_score=1,
                )
            ),
            "value": model_payload(ReportedValue(raw="0.9367", numeric=0.9367, unit="proportion")),
            "evidence": [model_payload(scope_anchor), model_payload(result_anchor)],
            "construct": None,
            "operationalization": None,
        }
    )
    candidate = CandidateObservation.model_validate(payload)
    return _with_quote_field_provenance(
        candidate,
        {
            CandidateField.SYSTEM: [result_anchor],
            CandidateField.DATASET_SCOPE: [scope_anchor],
            CandidateField.METRIC: [result_anchor],
            CandidateField.VALUE: [result_anchor],
        },
    )


def _field_conflict_candidate() -> CandidateObservation:
    candidate = _metric_context_candidate()
    payload = model_payload(candidate)
    metric = next(
        binding for binding in payload["field_provenance"] if binding["field"] == "metric"
    )
    metric.update(
        {
            "status": "conflict",
            "alternate_value_sha256s": ["f" * 64],
            "reason": "fixture metric identity conflict",
        }
    )
    return CandidateObservation.model_validate(payload)


def _canonicalization_drift_candidate() -> CandidateObservation:
    """Return a sealed source tuple whose registry replay changes its stable ID."""

    payload = model_payload(_metric_context_candidate())
    payload.update(
        {
            "schema_version": "candidate-observation/0.2",
            "observation_id": None,
            "field_provenance": [],
        }
    )
    payload["metric"].update(
        {
            "canonical_id": None,
            "kind": "provider_supplied_auc_label",
            "lower_is_better": None,
            "min_score": None,
            "max_score": None,
        }
    )
    return CandidateObservation.model_validate(payload)


def _value_contradiction_candidate(kind: str) -> CandidateObservation:
    payload = model_payload(_candidate())
    payload["observation_id"] = None
    if kind == "numeric":
        payload["value"]["numeric"] = 75.0
    elif kind == "comparator":
        payload["value"]["comparator"] = ValueComparator.LESS_THAN.value
    elif kind == "uncertainty":
        payload["value"]["uncertainty"] = model_payload(Uncertainty(standard_error=1.2))
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unsupported contradiction fixture: {kind}")
    candidate = CandidateObservation.model_validate(payload)
    row = candidate.evidence[0]
    return _with_quote_field_provenance(
        candidate,
        {field: [row] for field in CandidateField},
    )


def _legacy_value_contradiction_candidate(kind: str) -> CandidateObservation:
    payload = model_payload(_value_contradiction_candidate(kind))
    payload.update(
        {
            "schema_version": "candidate-observation/0.2",
            "observation_id": None,
            "field_provenance": [],
        }
    )
    return CandidateObservation.model_validate(payload)


def _nonnumeric_value_candidate(quote: str) -> CandidateObservation:
    payload = model_payload(_candidate())
    payload.update(
        {
            "observation_id": None,
            "metric": model_payload(MetricSpec(raw_name="AUC")),
            "value": model_payload(ReportedValue(raw="foo", numeric=123.0)),
            "evidence": [
                model_payload(
                    EvidenceAnchor(
                        source_id=SOURCE_ID,
                        page=2,
                        kind=EvidenceKind.PROSE,
                        quote=quote,
                    )
                ),
                model_payload(
                    EvidenceAnchor(
                        source_id=SOURCE_ID,
                        page=2,
                        kind=EvidenceKind.TABLE,
                        label="Table 1",
                        quote="Table 1. Synthetic Benchmark split test evaluation (AUC)",
                    )
                ),
            ],
        }
    )
    candidate = CandidateObservation.model_validate(payload)
    return _with_quote_field_provenance(
        candidate,
        {field: list(candidate.evidence) for field in CandidateField},
    )


def _multi_number_raw_candidate(
    raw: str,
    numeric: float,
    *,
    unit: str | None,
    schema_version: str = "candidate-observation/0.3",
) -> CandidateObservation:
    payload = model_payload(_candidate())
    payload.update(
        {
            "schema_version": "candidate-observation/0.2",
            "observation_id": None,
            "metric": model_payload(MetricSpec(raw_name="AUC", unit=unit)),
            "value": model_payload(ReportedValue(raw=raw, numeric=numeric, unit=unit)),
            "evidence": [
                model_payload(
                    EvidenceAnchor(
                        source_id=SOURCE_ID,
                        page=2,
                        kind=EvidenceKind.PROSE,
                        quote=f"Atlas Moderation API {raw}",
                    )
                ),
                model_payload(
                    EvidenceAnchor(
                        source_id=SOURCE_ID,
                        page=2,
                        kind=EvidenceKind.TABLE,
                        label="Table 1",
                        quote="Table 1. Synthetic Benchmark split test evaluation (AUC)",
                    )
                ),
            ],
            "field_provenance": [],
            "operationalization": None,
        }
    )
    candidate = CandidateObservation.model_validate(payload)
    if schema_version == "candidate-observation/0.2":
        return candidate
    row = candidate.evidence[0]
    return _with_quote_field_provenance(
        candidate,
        {field: [row] for field in CandidateField},
    )


def _uncertainty_method_candidate(quote: str) -> CandidateObservation:
    payload = model_payload(_candidate())
    payload.update(
        {
            "observation_id": None,
            "value": model_payload(
                ReportedValue(
                    raw="72.3%",
                    numeric=72.3,
                    unit="percent",
                    uncertainty=Uncertainty(method="bootstrap"),
                )
            ),
            "evidence": [
                model_payload(
                    EvidenceAnchor(
                        source_id=SOURCE_ID,
                        page=2,
                        kind=EvidenceKind.PROSE,
                        quote=quote,
                    )
                ),
                payload["evidence"][1],
            ],
            "operationalization": None,
        }
    )
    return CandidateObservation.model_validate(payload)


def _identifier_variant_candidate(result_name: str) -> CandidateObservation:
    quote = f"{result_name} result on Synthetic Benchmark split test AUC 74.6%"
    payload = model_payload(_candidate())
    payload.update(
        {
            "observation_id": None,
            "roles": [
                model_payload(
                    RoleAssignment(
                        role=ActorRole.EVALUATED_SYSTEM,
                        raw_name="GPT-4",
                        confidence=1.0,
                    )
                )
            ],
            "evidence": [
                model_payload(
                    EvidenceAnchor(
                        source_id=SOURCE_ID,
                        page=2,
                        kind=EvidenceKind.PROSE,
                        quote=quote,
                    )
                )
            ],
            "operationalization": None,
        }
    )
    return CandidateObservation.model_validate(payload)


def _fabricated_metadata_candidate(field: CandidateField) -> CandidateObservation:
    payload = model_payload(_candidate())
    payload["observation_id"] = None
    if field is CandidateField.SYSTEM:
        payload["roles"][0].update(
            {
                "canonical_id": "fabricated-atlas",
                "version": "v999",
                "provider": "Fabricated Provider",
            }
        )
    elif field is CandidateField.DATASET_SCOPE:
        payload["scope"].update(
            {
                "dataset_id": "fabricated-dataset",
                "dataset_url": "https://fabricated.invalid/dataset",
                "dataset_version": "v999",
            }
        )
    elif field is CandidateField.METRIC:
        payload["metric"].update(
            {
                "raw_name": "CustomScore",
                "canonical_id": "fabricated-metric",
                "kind": "fabricated-kind",
                "lower_is_better": True,
            }
        )
        payload["evidence"] = [
            model_payload(
                EvidenceAnchor(
                    source_id=SOURCE_ID,
                    page=2,
                    kind=EvidenceKind.PROSE,
                    quote=("Atlas Moderation API Synthetic Benchmark split test CustomScore 74.6%"),
                )
            )
        ]
        payload["operationalization"] = None
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unsupported metadata fixture field: {field}")
    return CandidateObservation.model_validate(payload)


def _sample_count_borrowing_metric_range_candidate() -> CandidateObservation:
    payload = model_payload(_candidate())
    payload["observation_id"] = None
    payload["scope"]["sample_count"] = 100
    payload["evidence"].append(
        model_payload(
            EvidenceAnchor(
                source_id=SOURCE_ID,
                page=2,
                kind=EvidenceKind.TABLE,
                label="Table 2",
                quote="Table 2. Synthetic Benchmark split test AUC range 0-100",
            )
        )
    )
    return CandidateObservation.model_validate(payload)


def _unit_borrowing_unrelated_percent_candidate() -> CandidateObservation:
    payload = model_payload(_candidate())
    payload.update(
        {
            "observation_id": None,
            "metric": model_payload(MetricSpec(raw_name="AUC", unit="percent")),
            "value": model_payload(ReportedValue(raw=".76", numeric=0.76, unit="percent")),
            "evidence": [
                model_payload(
                    EvidenceAnchor(
                        source_id=SOURCE_ID,
                        page=2,
                        kind=EvidenceKind.PROSE,
                        quote="Atlas Moderation API .76; baseline 10%",
                    )
                ),
                payload["evidence"][1],
            ],
            "operationalization": None,
        }
    )
    return CandidateObservation.model_validate(payload)


def _frankenstein_candidate() -> CandidateObservation:
    payload = model_payload(_candidate())
    payload.update(
        {
            "observation_id": None,
            "scope": model_payload(ObservationScope(dataset_raw="D2")),
            "metric": model_payload(
                MetricSpec(
                    raw_name="F1",
                    canonical_id="f1",
                    kind="f1",
                    unit="percent",
                    lower_is_better=False,
                    min_score=0,
                    max_score=100,
                )
            ),
            "value": model_payload(ReportedValue(raw="70.0%", numeric=70.0, unit="percent")),
            "evidence": [
                model_payload(
                    EvidenceAnchor(
                        source_id=SOURCE_ID,
                        page=2,
                        kind=EvidenceKind.TABLE,
                        label="Table 1",
                        row="Atlas Moderation API",
                        column="Accuracy",
                        quote="Atlas Moderation API D1 Accuracy 80.0%",
                    )
                ),
                model_payload(
                    EvidenceAnchor(
                        source_id=SOURCE_ID,
                        page=2,
                        kind=EvidenceKind.TABLE,
                        label="Table 1",
                        row="Baseline",
                        column="F1",
                        quote="Baseline D2 F1 70.0%",
                    )
                ),
            ],
            "schema_version": "candidate-observation/0.2",
            "field_provenance": [],
        }
    )
    return CandidateObservation.model_validate(payload)


def _absent_field_candidate(field: CandidateField) -> CandidateObservation:
    payload = model_payload(_candidate())
    payload["observation_id"] = None
    payload["scope"]["split"] = None
    payload["value"] = model_payload(ReportedValue(raw="70.0%", numeric=70.0, unit="percent"))
    if field is CandidateField.DATASET_SCOPE:
        payload["scope"]["dataset_raw"] = "FABRICATED-DATASET-Z9"
        quote = "Atlas Moderation API AUC 70.0%"
    elif field is CandidateField.METRIC:
        payload["metric"].update(
            {
                "raw_name": "FABRICATED-METRIC-Q7",
                "canonical_id": "f1",
                "kind": "f1",
            }
        )
        quote = "Atlas Moderation API Synthetic Benchmark 70.0%"
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unsupported absent field fixture: {field}")
    payload["evidence"] = [
        model_payload(
            EvidenceAnchor(
                source_id=SOURCE_ID,
                page=2,
                kind=EvidenceKind.PROSE,
                quote=quote,
            )
        )
    ]
    candidate = CandidateObservation.model_validate(payload)
    return _with_quote_field_provenance(
        candidate,
        {bound_field: [candidate.evidence[0]] for bound_field in CandidateField},
    )


def _same_physical_cell_candidates(kind: str) -> tuple[CandidateObservation, CandidateObservation]:
    context = EvidenceAnchor(
        source_id=SOURCE_ID,
        page=2,
        kind=EvidenceKind.TABLE,
        label="Table 1",
        quote="Table 1. Synthetic Benchmark split test evaluation (AUC)",
    )
    common_anchor = {
        "source_id": SOURCE_ID,
        "page": 2,
        "kind": EvidenceKind.TABLE,
        "label": "Table 1",
        "row": "Atlas Moderation API",
        "column": "AUC",
        "region_id": "region_table_1",
        "planned_row_id": "row_atlas",
        "cell_id": "cell_atlas_auc",
        "numeric_token_id": "token_atlas_auc",
    }
    if kind == "uncertainty":
        quote = "Atlas Moderation API Synthetic Benchmark split test AUC 72.3% SE 1.2% bootstrap"
        first_value = ReportedValue(
            raw="72.3%",
            numeric=72.3,
            unit="percent",
        )
        second_value = ReportedValue(
            raw="72.3%",
            numeric=72.3,
            unit="percent",
            uncertainty=Uncertainty(standard_error=1.2),
        )
        anchors = (
            EvidenceAnchor(quote=quote, **common_anchor),
            EvidenceAnchor(quote=quote, **common_anchor),
        )
    elif kind == "comparator":
        first_value = ReportedValue(
            raw="73.2%",
            numeric=73.2,
            unit="percent",
            comparator=ValueComparator.LESS_THAN,
        )
        second_value = ReportedValue(raw="73.2%", numeric=73.2, unit="percent")
        anchors = (
            EvidenceAnchor(
                quote=("Atlas Moderation API Synthetic Benchmark split test AUC threshold < 73.2%"),
                **common_anchor,
            ),
            EvidenceAnchor(
                quote=("Atlas Moderation API Synthetic Benchmark split test AUC threshold < 73.2%"),
                **common_anchor,
            ),
        )
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unsupported physical-cell fixture: {kind}")

    candidates: list[CandidateObservation] = []
    for value, anchor in zip((first_value, second_value), anchors, strict=True):
        payload = model_payload(_candidate())
        payload.update(
            {
                "observation_id": None,
                "value": model_payload(value),
                "evidence": [model_payload(context), model_payload(anchor)],
            }
        )
        candidates.append(CandidateObservation.model_validate(payload))
    return candidates[0], candidates[1]


def _mixed_anchor_granularity_candidates() -> tuple[CandidateObservation, CandidateObservation]:
    payloads = [model_payload(_candidate()), model_payload(_candidate())]
    for payload in payloads:
        payload["observation_id"] = None
        payload["evidence"][0]["region_id"] = "region_table_1"
    payloads[0]["evidence"][0].update(
        {
            "planned_row_id": "row_atlas",
            "cell_id": "cell_atlas_auc",
            "numeric_token_id": "token_atlas_auc",
        }
    )
    first, second = (CandidateObservation.model_validate(payload) for payload in payloads)
    return first, second


def _same_value_occurrence_with_extra_context() -> tuple[
    CandidateObservation, CandidateObservation
]:
    payloads = [model_payload(_candidate()), model_payload(_candidate())]
    for payload in payloads:
        payload["observation_id"] = None
        payload["evidence"][0].update(
            {
                "kind": EvidenceKind.PROSE.value,
                "label": None,
                "row": None,
                "column": None,
            }
        )
    payloads[1]["evidence"].append(
        model_payload(
            EvidenceAnchor(
                source_id=SOURCE_ID,
                page=2,
                kind=EvidenceKind.PROSE,
                quote="Evaluation instrument ToxicBert.",
            )
        )
    )
    first, second = (CandidateObservation.model_validate(payload) for payload in payloads)
    return first, second


def _same_value_occurrence_with_whitespace_variant() -> tuple[
    CandidateObservation, CandidateObservation
]:
    quotes = (
        "  Atlas Moderation API whitespace Synthetic Benchmark split test AUC 88.8%",
        "Atlas Moderation API whitespace Synthetic Benchmark split test AUC 88.8%",
    )
    candidates: list[CandidateObservation] = []
    for quote in quotes:
        payload = model_payload(_candidate())
        payload.update(
            {
                "observation_id": None,
                "value": model_payload(ReportedValue(raw="88.8%", numeric=88.8, unit="percent")),
                "evidence": [
                    model_payload(
                        EvidenceAnchor(
                            source_id=SOURCE_ID,
                            page=2,
                            kind=EvidenceKind.PROSE,
                            quote=quote,
                        )
                    )
                ],
                "operationalization": None,
            }
        )
        candidates.append(CandidateObservation.model_validate(payload))
    candidates[1].observation_id = "obs_whitespace_variant"
    return candidates[0], candidates[1]


def _same_value_occurrence_with_narrow_and_wide_quotes() -> tuple[
    CandidateObservation, CandidateObservation
]:
    quotes = (
        ("Atlas Moderation API D3 Accuracy 81.0%", "Occurrence A"),
        ("Atlas Moderation API D3 Accuracy 81.0% score", "Occurrence B"),
    )
    candidates: list[CandidateObservation] = []
    for quote, label in quotes:
        payload = model_payload(_candidate())
        payload.update(
            {
                "observation_id": None,
                "scope": model_payload(ObservationScope(dataset_raw="D3")),
                "metric": model_payload(MetricSpec(raw_name="Accuracy", unit="percent")),
                "value": model_payload(ReportedValue(raw="81.0%", numeric=81.0, unit="percent")),
                "evidence": [
                    model_payload(
                        EvidenceAnchor(
                            source_id=SOURCE_ID,
                            page=2,
                            kind=EvidenceKind.PROSE,
                            label=label,
                            quote=quote,
                        )
                    )
                ],
                "operationalization": None,
            }
        )
        candidates.append(CandidateObservation.model_validate(payload))
    return candidates[0], candidates[1]


def _same_claimed_cell_with_descriptive_drift() -> tuple[
    CandidateObservation, CandidateObservation
]:
    payloads = [model_payload(_candidate()), model_payload(_candidate())]
    for payload in payloads:
        payload["observation_id"] = None
        payload["evidence"][0].update(
            {
                "region_id": "region_table_1",
                "planned_row_id": "row_atlas",
                "cell_id": "cell_atlas_auc",
                "numeric_token_id": "token_atlas_auc",
            }
        )
    payloads[1]["evidence"][0].update(
        {
            "label": "Table X",
            "quote": "Atlas Moderation API Synthetic Benchmark split test AUC 74.6%",
            "quote_sha256": None,
        }
    )
    first, second = (CandidateObservation.model_validate(payload) for payload in payloads)
    return first, second


def _split_system_name_candidate() -> CandidateObservation:
    value_anchor = EvidenceAnchor(
        source_id=SOURCE_ID,
        page=2,
        kind=EvidenceKind.PROSE,
        quote="Synthetic Benchmark split test Agreement is 96.1% with Tox-",
    )
    continuation_anchor = EvidenceAnchor(
        source_id=SOURCE_ID,
        page=2,
        kind=EvidenceKind.PROSE,
        quote="icBert on the benchmark.",
    )
    payload = model_payload(_candidate())
    payload.update(
        {
            "observation_id": None,
            "roles": [
                model_payload(
                    RoleAssignment(
                        role=ActorRole.EVALUATED_SYSTEM,
                        raw_name="ToxicBert",
                        confidence=1.0,
                    )
                ),
            ],
            "metric": model_payload(
                MetricSpec(
                    raw_name="Agreement",
                    unit="percent",
                )
            ),
            "value": model_payload(ReportedValue(raw="96.1%", numeric=96.1, unit="percent")),
            "evidence": [
                model_payload(value_anchor),
                model_payload(continuation_anchor),
            ],
            "operationalization": None,
        }
    )
    candidate = CandidateObservation.model_validate(payload)
    return _with_quote_field_provenance(
        candidate,
        {
            CandidateField.SYSTEM: [value_anchor, continuation_anchor],
            CandidateField.DATASET_SCOPE: [continuation_anchor],
            CandidateField.METRIC: [value_anchor],
            CandidateField.SETTING: [value_anchor, continuation_anchor],
            CandidateField.VALUE: [value_anchor],
            CandidateField.UNIT: [value_anchor],
        },
    )


def _single_span_frankenstein_candidate() -> CandidateObservation:
    combined = EvidenceAnchor(
        source_id=SOURCE_ID,
        page=2,
        kind=EvidenceKind.PROSE,
        quote="Atlas Moderation API D1 Accuracy 80.0%\nBaseline D2 F1 70.0%",
    )
    payload = model_payload(_candidate())
    payload.update(
        {
            "observation_id": None,
            "scope": model_payload(ObservationScope(dataset_raw="D2")),
            "metric": model_payload(MetricSpec(raw_name="F1", unit="percent")),
            "value": model_payload(ReportedValue(raw="70.0%", numeric=70.0, unit="percent")),
            "evidence": [model_payload(combined)],
            "operationalization": None,
        }
    )
    candidate = CandidateObservation.model_validate(payload)
    return _with_quote_field_provenance(
        candidate,
        {field: [combined] for field in CandidateField},
    )


def _cross_row_context_frankenstein_candidate() -> CandidateObservation:
    system_value = EvidenceAnchor(
        source_id=SOURCE_ID,
        page=2,
        kind=EvidenceKind.PROSE,
        quote="Atlas Moderation API D1 Accuracy 80.0%",
    )
    dataset_metric = EvidenceAnchor(
        source_id=SOURCE_ID,
        page=2,
        kind=EvidenceKind.PROSE,
        quote="Baseline D2 F1 70.0%",
    )
    payload = model_payload(_candidate())
    payload.update(
        {
            "observation_id": None,
            "scope": model_payload(ObservationScope(dataset_raw="D2")),
            "metric": model_payload(MetricSpec(raw_name="F1", unit="percent")),
            "value": model_payload(ReportedValue(raw="80.0%", numeric=80.0, unit="percent")),
            "evidence": [model_payload(system_value), model_payload(dataset_metric)],
            "operationalization": None,
        }
    )
    candidate = CandidateObservation.model_validate(payload)
    return _with_quote_field_provenance(
        candidate,
        {
            CandidateField.SYSTEM: [system_value],
            CandidateField.DATASET_SCOPE: [dataset_metric],
            CandidateField.METRIC: [dataset_metric],
            CandidateField.VALUE: [system_value],
            CandidateField.UNIT: [system_value],
        },
    )


def _instrument_only_setting_candidate() -> CandidateObservation:
    """Return a tuple whose only populated SETTING component is an actor role."""

    payload = model_payload(_candidate())
    payload["observation_id"] = None
    payload["scope"]["split"] = None
    payload["roles"].append(
        model_payload(
            RoleAssignment(
                role=ActorRole.EVALUATION_INSTRUMENT,
                raw_name="ToxicBert",
                confidence=1.0,
            )
        )
    )
    payload["evidence"].append(
        model_payload(
            EvidenceAnchor(
                source_id=SOURCE_ID,
                page=2,
                kind=EvidenceKind.PROSE,
                quote="Evaluation instrument ToxicBert.",
            )
        )
    )
    return CandidateObservation.model_validate(payload)


def _physical_conflict_candidate() -> CandidateObservation:
    payload = model_payload(_metric_context_candidate())
    payload["observation_id"] = None
    payload["notes"] = ["semantic safety: incompatible proposals share one physical value cell"]
    return CandidateObservation.model_validate(payload)


def _build_sealed_run(
    tmp_path: Path,
    *,
    confidence: float = 0.99,
    attribution: AttributionState = AttributionState.NO_SIGNAL,
    candidates: list[CandidateObservation] | None = None,
    corpus_manifest_override: dict[str, Any] | bytes | None = None,
) -> Path:
    raw = tmp_path / "raw-run"
    paper = raw / PAPER_ID
    private = paper / "private"
    private.mkdir(parents=True)
    source = FrozenSource(
        source_id=SOURCE_ID,
        paper_id=PAPER_ID,
        role=SourceRole.PAPER,
        original_uri="https://example.org/synthetic.pdf",
        resolved_uri="https://example.org/synthetic.pdf",
        retrieved_at=datetime(2026, 9, 2, tzinfo=UTC),
        sha256="a" * 64,
        byte_size=123,
        media_type="application/pdf",
        cache_relpath="data/sources/aa/synthetic.pdf",
        license_disposition=LicenseDisposition.REDISTRIBUTABLE,
    )
    manifest = SourceManifest(
        paper_id=PAPER_ID,
        title="Synthetic Two Page Study",
        sources=[source],
    )
    layout = PdfLayout(
        source_id=SOURCE_ID,
        parser="synthetic-layout",
        parser_version="1.0",
        page_count=2,
        pages=[_page(1, ORIGIN_TEXT), _page(2, RESULT_TEXT)],
    )
    if candidates is None:
        candidates = [_candidate(confidence=confidence, attribution=attribution)]
    source_manifest_sha = write_json(paper / "source-manifest.json", manifest)
    write_json(private / "layout.json", layout)
    write_json(private / "result-blocks.json", [{"block_id": "block-1", "page": 2}])
    write_jsonl(paper / "observations.jsonl", candidates)
    paper_run = {
        "schema_version": "pipeline-run/0.2",
        "paper_id": PAPER_ID,
        "source_manifest_sha256": source_manifest_sha,
        "layout_parser": layout.parser,
        "layout_parser_version": layout.parser_version,
        "extractor": {
            "prompt_sha256": "b" * 64,
            "request_contract_sha256": "c" * 64,
        },
        "row_enumeration": {
            "prompt_sha256": "d" * 64,
            "request_contract_sha256": "e" * 64,
        },
        "eee_schema": {"version": "0.2.2", "sha256": EEE_SCHEMA_SHA256},
        "code": {"git_commit": "f" * 40, "source_tree_sha256": "1" * 64},
    }
    write_json(paper / "run.json", paper_run)
    corpus_manifest: dict[str, Any] | bytes = (
        {
            "schema_version": "corpus-run/0.2",
            "status": "completed_with_review",
            "papers": 1,
            "runs": [paper_run],
        }
        if corpus_manifest_override is None
        else corpus_manifest_override
    )
    if isinstance(corpus_manifest, bytes):
        (raw / "corpus-run.json").write_bytes(corpus_manifest)
    else:
        write_json(raw / "corpus-run.json", corpus_manifest)
    sealed = tmp_path / "sealed-run"
    seal_run_tree(raw, sealed)
    return sealed


def _records(path: Path, model: type[Any]) -> list[Any]:
    return [model.model_validate(json.loads(line)) for line in path.read_text().splitlines()]


def _anchor(
    review_root: Path,
    *,
    page_number: int,
    excerpt: str,
    kind: EvidenceKind,
    label: str | None = None,
    row: str | None = None,
    column: str | None = None,
) -> ReviewEvidenceAnchor:
    manifest = ReviewManifest.model_validate(read_json(review_root / REVIEW_MANIFEST_NAME))
    paper = manifest.papers[0]
    source_manifest = SourceManifest.model_validate(
        read_json(review_root / paper.source_manifest.review_copy.path)
    )
    layout = PdfLayout.model_validate(read_json(review_root / paper.layout.review_copy.path))
    source = next(item for item in source_manifest.sources if item.source_id == SOURCE_ID)
    page = layout.pages[page_number - 1]
    start = page.text.index(excerpt)
    end = start + len(excerpt)
    return ReviewEvidenceAnchor(
        source_id=SOURCE_ID,
        source_sha256=str(source.sha256),
        source_manifest_sha256=paper.source_manifest.sha256,
        layout_sha256=paper.layout.sha256,
        parser=layout.parser,
        parser_version=layout.parser_version,
        page=page_number,
        page_text_sha256=page.text_sha256,
        exact_excerpt=excerpt,
        excerpt_sha256=sha256_bytes(excerpt.encode("utf-8")),
        char_start=start,
        char_end=end,
        start_line=page.text.count("\n", 0, start) + 1,
        end_line=page.text.count("\n", 0, end - 1) + 1,
        kind=kind,
        label=label,
        row=row,
        column=column,
    )


def _complete_decisions(
    review_root: Path,
    *,
    origin: OriginDecision = OriginDecision.PAPER_PRODUCED,
    mode: ReviewAuthorityMode = ReviewAuthorityMode.SINGLE_EXPERT,
) -> list[ReviewedExportDecision]:
    decisions = _records(review_root / REVIEW_DECISIONS_NAME, ReviewedExportDecision)
    manifest = ReviewManifest.model_validate(read_json(review_root / REVIEW_MANIFEST_NAME))
    completed: list[ReviewedExportDecision] = []
    for decision in decisions:
        payload = model_payload(decision)
        value_hashes = decision.reviewed_tuple.field_value_sha256s()
        span_ids = sorted(
            anchor.span_id for anchor in decision.result_evidence if anchor.span_id is not None
        )
        payload.update(
            {
                "status": DecisionStatus.COMPLETED.value,
                "tuple_decision": TupleDecision.CONFIRMED.value,
                "origin_decision": origin.value,
                "authority": model_payload(
                    DecisionAuthority(
                        mode=mode,
                        reviewer_ids=(
                            ["reviewer-a", "reviewer-b"]
                            if mode is not ReviewAuthorityMode.SINGLE_EXPERT
                            else ["reviewer-a"]
                        ),
                        adjudicator_id=(
                            "adjudicator-c" if mode is ReviewAuthorityMode.ADJUDICATED else None
                        ),
                        protocol_sha256=manifest.protocol.sha256,
                    )
                ),
                "decided_at": "2026-09-02T12:00:00Z",
                "field_attestations": [
                    model_payload(
                        ReviewFieldAttestation(
                            field=field,
                            value_sha256=value_hashes[field],
                            span_ids=span_ids,
                        )
                    )
                    for field in sorted(
                        decision.reviewed_tuple.populated_fields(), key=lambda item: item.value
                    )
                ],
                "origin_evidence": [
                    model_payload(
                        _anchor(
                            review_root,
                            page_number=1,
                            excerpt=(
                                "We evaluated Atlas Moderation API ourselves on the synthetic "
                                "benchmark."
                            ),
                            kind=EvidenceKind.PROSE,
                        )
                    )
                ],
            }
        )
        completed.append(ReviewedExportDecision.model_validate(payload))
    write_jsonl(
        review_root / REVIEW_DECISIONS_NAME,
        [model_payload(decision) for decision in completed],
    )
    return completed


def _complete_decision(
    review_root: Path,
    *,
    origin: OriginDecision = OriginDecision.PAPER_PRODUCED,
    mode: ReviewAuthorityMode = ReviewAuthorityMode.SINGLE_EXPERT,
) -> ReviewedExportDecision:
    [completed] = _complete_decisions(review_root, origin=origin, mode=mode)
    return completed


def _replace_completed_field_map(
    review_root: Path,
    *,
    field_excerpts: dict[CandidateField, tuple[str, ...]],
    origin_excerpt: str,
) -> ReviewedExportDecision:
    [decision] = _records(review_root / REVIEW_DECISIONS_NAME, ReviewedExportDecision)
    spans_by_excerpt = {anchor.exact_excerpt: anchor for anchor in decision.result_evidence}
    hashes = decision.reviewed_tuple.field_value_sha256s()
    payload = model_payload(decision)
    payload["field_attestations"] = [
        model_payload(
            ReviewFieldAttestation(
                field=field,
                value_sha256=hashes[field],
                span_ids=sorted(spans_by_excerpt[value].span_id for value in excerpts),
            )
        )
        for field, excerpts in sorted(field_excerpts.items(), key=lambda item: item[0].value)
    ]
    payload["origin_evidence"] = [
        model_payload(
            _anchor(
                review_root,
                page_number=1,
                excerpt=origin_excerpt,
                kind=EvidenceKind.PROSE,
            )
        )
    ]
    completed = ReviewedExportDecision.model_validate(payload)
    write_jsonl(review_root / REVIEW_DECISIONS_NAME, [model_payload(completed)])
    return completed


def _prepare(tmp_path: Path, **run_options: Any) -> tuple[Path, Path]:
    sealed = _build_sealed_run(tmp_path, **run_options)
    review = tmp_path / "review"
    prepare_export_review(run_root=sealed, output_root=review)
    return sealed, review


def _compose_reviewed_fixture(sealed: Path, review: Path, derived: Path) -> DerivedRunManifest:
    return compose_reviewed_eee(
        run_root=sealed,
        decisions_path=review / REVIEW_DECISIONS_NAME,
        output_root=derived,
    )


def _outcomes(derived: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (derived / DERIVED_OUTCOMES_NAME).read_text().splitlines()]


def _fully_rehash_derived(derived: Path) -> None:
    manifest_payload = read_json(derived / DERIVED_MANIFEST_NAME)
    inventory = reviewed_workflow._payload_inventory(derived)
    manifest_payload["payload_files"] = [model_payload(item) for item in inventory]
    manifest_payload["payload_tree_sha256"] = sha256_bytes(
        canonical_json_bytes(manifest_payload["payload_files"])
    )
    sums = reviewed_workflow._sha256s_bytes(inventory)
    (derived / DERIVED_SHA256SUMS_NAME).write_bytes(sums)
    manifest_payload["sha256s_sha256"] = sha256_bytes(sums)
    manifest = DerivedRunManifest.model_validate(manifest_payload)
    write_json(derived / DERIVED_MANIFEST_NAME, manifest)
    verification = read_json(derived / DERIVED_VERIFICATION_NAME)
    verification.update(
        {
            "derived_run_sha256": sha256_file(derived / DERIVED_MANIFEST_NAME),
            "sha256s_sha256": manifest.sha256s_sha256,
            "payload_tree_sha256": manifest.payload_tree_sha256,
            "payload_file_count": len(manifest.payload_files),
        }
    )
    write_json(derived / DERIVED_VERIFICATION_NAME, verification)


def _derived_payloads(
    derived: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    manifest = read_json(derived / DERIVED_MANIFEST_NAME)
    [outcome] = _outcomes(derived)
    [provenance] = [
        json.loads(line) for line in (derived / DERIVED_PROVENANCE_NAME).read_text().splitlines()
    ]
    eee_path = derived / outcome["eee_path"]
    record = read_json(eee_path)
    return manifest, outcome, provenance, record, eee_path


def test_two_page_reviewed_origin_composes_one_verified_eee_without_mutation(
    tmp_path: Path,
) -> None:
    sealed, review = _prepare(tmp_path)
    before = verify_run_seal(sealed)
    _complete_decision(review)
    lock, lock_sha = validate_export_review(review)

    derived = tmp_path / "derived"
    manifest = _compose_reviewed_fixture(sealed, review, derived)
    verification = verify_derived_run(derived, run_root=sealed, review_root=review)
    contextual = verify_contextual_derived_run(
        derived,
        run_root=sealed,
        review_root=review,
    )
    after = verify_run_seal(sealed)

    eee_files = list(derived.glob(f"{PAPER_ID}/eee/*.json"))
    assert len(eee_files) == 1
    record = read_json(eee_files[0])
    assert len(record["evaluation_results"]) == 1
    details = record["evaluation_results"][0]["score_details"]["details"]
    assert details["review_lock_sha256"] == lock_sha
    assert details["reviewed_result_evidence_1_page"] == "2"
    assert details["reviewed_origin_evidence_1_page"] == "1"
    assert "exact_excerpt" not in json.dumps(record)
    assert ORIGIN_TEXT.strip() not in json.dumps(record)
    assert manifest.counts["eee_observations"] == 1
    assert verification.exported_observation_count == 1
    assert contextual.verification == verification
    assert contextual.authority_mode_counts == {
        "single_expert": 1,
        "dual_consensus": 0,
        "adjudicated": 0,
    }
    assert contextual.provenance_mode_counts == {
        "tuple_gated_production": 0,
        "tuple_gated_unverified": 0,
        "tuple_audited_human_reviewed": 0,
        "legacy_manual": 0,
        "legacy_human_reviewed": 1,
    }
    assert lock.completed_count == 1
    assert before == after


def test_identical_run_and_decisions_recompose_byte_identically(tmp_path: Path) -> None:
    sealed = _build_sealed_run(tmp_path)
    roots: list[Path] = []
    for name in ("a", "b"):
        review = tmp_path / f"review-{name}"
        prepare_export_review(run_root=sealed, output_root=review)
        _complete_decision(review)
        validate_export_review(review)
        derived = tmp_path / f"derived-{name}"
        _compose_reviewed_fixture(sealed, review, derived)
        roots.append(derived)
    left = {
        path.relative_to(roots[0]).as_posix(): path.read_bytes()
        for path in roots[0].rglob("*")
        if path.is_file()
    }
    right = {
        path.relative_to(roots[1]).as_posix(): path.read_bytes()
        for path in roots[1].rglob("*")
        if path.is_file()
    }
    assert left == right


@pytest.mark.parametrize("corruption", ["page", "page_hash", "offset", "excerpt"])
def test_origin_anchor_corruption_cannot_be_locked(tmp_path: Path, corruption: str) -> None:
    _, review = _prepare(tmp_path)
    decision = _complete_decision(review)
    payload = model_payload(decision)
    anchor = payload["origin_evidence"][0]
    if corruption == "page":
        anchor["page"] = 2
    elif corruption == "page_hash":
        anchor["page_text_sha256"] = "9" * 64
    elif corruption == "offset":
        anchor["char_start"] += 1
        anchor["char_end"] += 1
    else:
        anchor["exact_excerpt"] = "Atlas Moderation API was evaluated."
        anchor["excerpt_sha256"] = sha256_bytes(anchor["exact_excerpt"].encode())
        anchor["char_end"] = anchor["char_start"] + len(anchor["exact_excerpt"])
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.ORIGIN_EVIDENCE_INVALID
    assert not (review / "review-lock.json").exists()


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        (OriginDecision.EXTERNALLY_SOURCED, "ORIGIN_EXTERNAL"),
        (OriginDecision.UNRESOLVED, "ORIGIN_UNRESOLVED"),
    ],
)
def test_nonpositive_completed_origin_is_locked_but_never_exported(
    tmp_path: Path, origin: OriginDecision, expected: str
) -> None:
    sealed, review = _prepare(tmp_path)
    decision = _complete_decision(review, origin=origin)
    if origin is OriginDecision.UNRESOLVED:
        payload = model_payload(decision)
        payload["origin_evidence"] = []
        write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    assert not list(derived.glob("*/eee/*.json"))
    assert _outcomes(derived)[0]["failure_codes"] == [expected]


def test_pending_decision_is_a_typed_withheld_outcome(tmp_path: Path) -> None:
    sealed, review = _prepare(tmp_path)
    lock, _ = validate_export_review(review)
    assert lock.pending_count == 1
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    assert _outcomes(derived)[0]["failure_codes"] == ["DECISION_PENDING"]
    assert not list(derived.glob("*/eee/*.json"))


def test_origin_approval_cannot_override_low_confidence(tmp_path: Path) -> None:
    sealed, review = _prepare(tmp_path, confidence=0.5)
    [item] = _records(review / REVIEW_ITEMS_NAME, ReviewItem)
    assert item.base_gate_status == "needs_review"
    _complete_decision(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    assert "LOW_CONFIDENCE" in _outcomes(derived)[0]["failure_codes"]
    assert not list(derived.glob("*/eee/*.json"))


def test_origin_only_approval_cannot_bypass_human_tuple_confirmation(tmp_path: Path) -> None:
    sealed, review = _prepare(tmp_path)
    decision = _complete_decision(review)
    payload = model_payload(decision)
    payload["tuple_decision"] = TupleDecision.UNRESOLVED.value
    payload["field_attestations"] = []
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])
    validate_export_review(review)

    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)

    assert _outcomes(derived)[0]["failure_codes"] == ["TUPLE_UNRESOLVED"]
    assert not list(derived.glob("*/eee/*.json"))


def test_single_expert_cannot_silently_override_deterministic_external_cue(
    tmp_path: Path,
) -> None:
    sealed, review = _prepare(tmp_path, attribution=AttributionState.EXTERNALLY_SOURCED)
    _complete_decision(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    assert _outcomes(derived)[0]["failure_codes"] == ["DETERMINISTIC_EXTERNAL_CONFLICT"]


def test_changed_sealed_candidate_is_rejected_before_recomposition(tmp_path: Path) -> None:
    sealed, review = _prepare(tmp_path)
    _complete_decision(review)
    validate_export_review(review)
    observations = sealed / PAPER_ID / "observations.jsonl"
    observations.chmod(0o600)
    observations.write_text(observations.read_text().replace("74.6", "75.6"), encoding="utf-8")

    with pytest.raises(ReviewedExportError) as captured:
        compose_reviewed_eee(
            run_root=sealed,
            decisions_path=review / REVIEW_DECISIONS_NAME,
            output_root=tmp_path / "derived",
        )
    assert captured.value.code is ReviewedExportErrorCode.RUN_SEAL_INVALID
    assert not (tmp_path / "derived").exists()


def test_prepare_rejects_sealed_corpus_with_unparseable_manifest(tmp_path: Path) -> None:
    sealed = _build_sealed_run(
        tmp_path,
        corpus_manifest_override=b'{"schema_version":"corpus-run/0.2","runs":',
    )

    with pytest.raises(ReviewedExportError) as captured:
        prepare_export_review(run_root=sealed, output_root=tmp_path / "review")

    assert captured.value.code is ReviewedExportErrorCode.MANIFEST_INVALID
    assert not (tmp_path / "review").exists()


def test_prepare_rejects_paper_run_absent_from_corpus_membership(tmp_path: Path) -> None:
    sealed = _build_sealed_run(
        tmp_path,
        corpus_manifest_override={
            "schema_version": "corpus-run/0.2",
            "status": "completed_with_review",
            "papers": 0,
            "runs": [],
        },
    )

    with pytest.raises(ReviewedExportError) as captured:
        prepare_export_review(run_root=sealed, output_root=tmp_path / "review")

    assert captured.value.code is ReviewedExportErrorCode.MANIFEST_INVALID
    assert not (tmp_path / "review").exists()


def test_tampered_derived_payload_fails_standalone_verification(tmp_path: Path) -> None:
    sealed, review = _prepare(tmp_path)
    _complete_decision(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    eee_path = next(derived.glob("*/eee/*.json"))
    eee_path.write_text(eee_path.read_text().replace("74.6", "75.6"), encoding="utf-8")
    with pytest.raises(ReviewedExportError) as captured:
        verify_derived_run(derived)
    assert captured.value.code in {
        ReviewedExportErrorCode.ARTIFACT_HASH_MISMATCH,
        ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
    }


@pytest.mark.parametrize(
    ("relative_path", "contents"),
    [
        ("PRIVATE_QUOTES.txt", "Atlas Moderation API 74.6%"),
        ("private/annotations.json", '{"reviewer_note":"source excerpt"}'),
    ],
)
def test_fully_rehashed_unrecognized_payload_is_rejected(
    tmp_path: Path, relative_path: str, contents: str
) -> None:
    sealed, review = _prepare(tmp_path)
    _complete_decision(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    extra = derived / relative_path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(contents, encoding="utf-8")
    _fully_rehash_derived(derived)

    for contextual in (False, True):
        with pytest.raises(ReviewedExportError) as captured:
            verify_derived_run(
                derived,
                run_root=sealed if contextual else None,
                review_root=review if contextual else None,
            )
        assert captured.value.code is ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("candidate_fingerprint_sha256", "9" * 64),
        ("reviewed_result_evidence_count", "3"),
        ("reviewed_origin_evidence_1_excerpt_sha256", "8" * 64),
    ],
)
def test_fully_rehashed_eee_to_provenance_mismatch_fails_portable_verification(
    tmp_path: Path, field: str, replacement: str
) -> None:
    sealed, review = _prepare(tmp_path)
    _complete_decision(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    _, _, _, record, eee_path = _derived_payloads(derived)
    record["evaluation_results"][0]["score_details"]["details"][field] = replacement
    write_json(eee_path, record)
    _fully_rehash_derived(derived)

    with pytest.raises(ReviewedExportError) as captured:
        verify_derived_run(derived)
    assert captured.value.code is ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE


@pytest.mark.parametrize(
    "corruption",
    ["duplicate_result_span", "dangling_attestation", "tampered_review_span"],
)
def test_fully_rehashed_inconsistent_field_attestation_fails_portable_verification(
    tmp_path: Path, corruption: str
) -> None:
    sealed, review = _prepare(tmp_path, candidates=[_metric_context_candidate()])
    _complete_decision(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    provenance_path = derived / DERIVED_PROVENANCE_NAME
    [provenance] = [json.loads(line) for line in provenance_path.read_text().splitlines()]
    if corruption == "duplicate_result_span":
        provenance["result_evidence"].append(dict(provenance["result_evidence"][0]))
    elif corruption == "dangling_attestation":
        provenance["field_attestations"][0]["span_ids"] = ["span_" + "f" * 64]
    else:
        value_binding = next(
            binding for binding in provenance["field_provenance"] if binding["field"] == "value"
        )
        review_source = next(
            source for source in value_binding["sources"] if source["kind"] == "review_span"
        )
        review_source["quote_sha256"] = "0" * 64
    write_jsonl(provenance_path, [provenance])
    _fully_rehash_derived(derived)

    with pytest.raises(ReviewedExportError) as captured:
        verify_derived_run(derived)
    assert captured.value.code is ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE


@pytest.mark.parametrize("substantive_field", ["score", "metric", "dataset"])
def test_fully_rehashed_substantive_eee_tamper_fails_contextual_recomposition(
    tmp_path: Path, substantive_field: str
) -> None:
    sealed, review = _prepare(tmp_path)
    _complete_decision(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    _, _, _, record, eee_path = _derived_payloads(derived)
    [result] = record["evaluation_results"]
    if substantive_field == "score":
        result["score_details"]["score"] = 99.9
    elif substantive_field == "metric":
        result["metric_config"]["metric_id"] = "accuracy"
        result["metric_config"]["metric_name"] = "Accuracy"
        result["metric_config"]["metric_kind"] = "accuracy"
    else:
        result["source_data"]["dataset_name"] = "Tampered Benchmark"
    write_json(eee_path, record)
    _fully_rehash_derived(derived)

    verify_derived_run(derived)
    with pytest.raises(ReviewedExportError) as captured:
        verify_derived_run(derived, run_root=sealed, review_root=review)
    assert captured.value.code is ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE


@pytest.mark.parametrize(
    "binding",
    [
        "candidate",
        "review_manifest",
        "review_lock",
        "review_decision",
        "authority",
        "protocol",
        "result_evidence",
        "origin_evidence",
    ],
)
def test_fully_rehashed_public_binding_cannot_impersonate_sealed_review_context(
    tmp_path: Path, binding: str
) -> None:
    sealed, review = _prepare(tmp_path)
    _complete_decision(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    manifest, outcome, provenance, record, eee_path = _derived_payloads(derived)
    details = record["evaluation_results"][0]["score_details"]["details"]
    if binding == "candidate":
        replacement = "9" * 64
        outcome["candidate_payload_sha256"] = replacement
        provenance["candidate_payload_sha256"] = replacement
        details["candidate_fingerprint_sha256"] = replacement
    elif binding == "review_manifest":
        replacement = "8" * 64
        manifest["review_manifest_sha256"] = replacement
        provenance["review_manifest_sha256"] = replacement
        details["review_manifest_sha256"] = replacement
    elif binding == "review_lock":
        replacement = "7" * 64
        manifest["review_lock_sha256"] = replacement
        provenance["review_lock_sha256"] = replacement
        details["review_lock_sha256"] = replacement
    elif binding == "review_decision":
        replacement = "6" * 64
        provenance["review_decision_sha256"] = replacement
        details["review_decision_sha256"] = replacement
    elif binding == "authority":
        provenance["decision_authority"] = "dual_consensus"
        details["decision_authority"] = "dual_consensus"
    elif binding == "protocol":
        provenance["decision_protocol_id"] = "tampered-review-protocol/0.1"
        details["decision_protocol_id"] = "tampered-review-protocol/0.1"
    elif binding == "result_evidence":
        replacement = "5" * 64
        evidence = provenance["result_evidence"][0]
        old_span_id = evidence["span_id"]
        evidence["excerpt_sha256"] = replacement
        evidence["span_id"] = reviewed_workflow.review_evidence_span_id(
            source_id=evidence["source_id"],
            page=evidence["page"],
            page_text_sha256=evidence["page_text_sha256"],
            char_start=evidence["char_start"],
            char_end=evidence["char_end"],
            excerpt_sha256=replacement,
        )
        details["reviewed_result_evidence_1_excerpt_sha256"] = replacement
        details["reviewed_result_evidence_1_span_id"] = evidence["span_id"]
        for attestation in provenance["field_attestations"]:
            attestation["span_ids"] = sorted(
                evidence["span_id"] if span_id == old_span_id else span_id
                for span_id in attestation["span_ids"]
            )
        for field_binding in provenance["field_provenance"]:
            for source in field_binding["sources"]:
                if source.get("review_span_id") == old_span_id:
                    source["review_span_id"] = evidence["span_id"]
                    source["quote_sha256"] = replacement
            field_binding["sources"].sort(
                key=lambda source: json.dumps(
                    source,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    else:
        replacement = "4" * 64
        evidence = provenance["origin_evidence"][0]
        evidence["excerpt_sha256"] = replacement
        evidence["span_id"] = reviewed_workflow.review_evidence_span_id(
            source_id=evidence["source_id"],
            page=evidence["page"],
            page_text_sha256=evidence["page_text_sha256"],
            char_start=evidence["char_start"],
            char_end=evidence["char_end"],
            excerpt_sha256=replacement,
        )
        details["reviewed_origin_evidence_1_excerpt_sha256"] = replacement
        details["reviewed_origin_evidence_1_span_id"] = evidence["span_id"]
    write_json(derived / DERIVED_MANIFEST_NAME, manifest)
    write_jsonl(derived / DERIVED_OUTCOMES_NAME, [outcome])
    write_jsonl(derived / DERIVED_PROVENANCE_NAME, [provenance])
    write_json(eee_path, record)
    _fully_rehash_derived(derived)

    verify_derived_run(derived)
    with pytest.raises(ReviewedExportError) as captured:
        verify_derived_run(derived, run_root=sealed, review_root=review)
    assert captured.value.code is ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("tuple_decision", TupleDecision.REJECTED.value),
        ("origin_decision", OriginDecision.EXTERNALLY_SOURCED.value),
    ],
)
def test_fully_rehashed_export_cannot_outlive_changed_locked_decision(
    tmp_path: Path, field: str, replacement: str
) -> None:
    sealed, review = _prepare(tmp_path)
    _complete_decision(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)

    lock_path = review / REVIEW_LOCK_NAME
    decisions_path = review / REVIEW_DECISIONS_NAME
    lock_path.chmod(0o600)
    lock_path.unlink()
    decisions_path.chmod(0o600)
    [decision_payload] = [json.loads(line) for line in decisions_path.read_text().splitlines()]
    decision_payload[field] = replacement
    if field == "tuple_decision":
        decision_payload["field_attestations"] = []
    write_jsonl(decisions_path, [decision_payload])
    decision = ReviewedExportDecision.model_validate(decision_payload)
    lock, lock_sha = validate_export_review(review)

    manifest, outcome, provenance, record, eee_path = _derived_payloads(derived)
    decision_sha = reviewed_workflow._decision_sha256(decision)
    manifest["review_lock_sha256"] = lock_sha
    manifest["decisions_sha256"] = lock.decisions_sha256
    manifest["locked_at"] = lock.locked_at.isoformat() if lock.locked_at is not None else None
    provenance["review_lock_sha256"] = lock_sha
    provenance["review_decision_sha256"] = decision_sha
    details = record["evaluation_results"][0]["score_details"]["details"]
    details["review_lock_sha256"] = lock_sha
    details["review_decision_sha256"] = decision_sha
    write_json(derived / DERIVED_MANIFEST_NAME, manifest)
    write_jsonl(derived / DERIVED_OUTCOMES_NAME, [outcome])
    write_jsonl(derived / DERIVED_PROVENANCE_NAME, [provenance])
    write_json(eee_path, record)
    _fully_rehash_derived(derived)

    verify_derived_run(derived)
    with pytest.raises(ReviewedExportError) as captured:
        verify_derived_run(derived, run_root=sealed, review_root=review)
    assert captured.value.code is ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE


@pytest.mark.parametrize(
    "count_key",
    [
        "review_items",
        "decisions_completed",
        "decisions_pending",
        "outcomes_exported",
        "outcomes_withheld",
        "outcomes_failed",
        "eee_records",
        "eee_observations",
    ],
)
def test_tampered_derived_counts_fail_with_recomputed_receipt(
    tmp_path: Path, count_key: str
) -> None:
    sealed, review = _prepare(tmp_path)
    _complete_decision(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)

    manifest_path = derived / DERIVED_MANIFEST_NAME
    manifest = read_json(manifest_path)
    manifest["counts"][count_key] += 999_999
    write_json(manifest_path, manifest)
    verification_path = derived / DERIVED_VERIFICATION_NAME
    verification = read_json(verification_path)
    verification["derived_run_sha256"] = sha256_file(manifest_path)
    write_json(verification_path, verification)

    with pytest.raises(ReviewedExportError) as captured:
        verify_derived_run(derived)
    assert captured.value.code is ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE


def test_review_and_derived_outputs_never_enter_the_source_seal(tmp_path: Path) -> None:
    sealed, review = _prepare(tmp_path)
    original_files = {item["path"] for item in verify_run_seal(sealed).files}
    assert REVIEW_MANIFEST_NAME not in original_files
    assert REVIEW_DECISIONS_NAME not in original_files
    assert RUN_SEAL_NAME not in original_files
    assert review.is_dir()


def test_changed_reviewed_tuple_cannot_be_locked(tmp_path: Path) -> None:
    _, review = _prepare(tmp_path)
    decision = _complete_decision(review)
    payload = model_payload(decision)
    payload["reviewed_tuple"]["value"]["numeric"] = 75.6
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.TUPLE_MISMATCH
    assert not (review / "review-lock.json").exists()


def test_result_evidence_without_the_raw_value_cannot_be_locked(tmp_path: Path) -> None:
    _, review = _prepare(tmp_path)
    decision = _complete_decision(review)
    payload = model_payload(decision)
    payload["result_evidence"] = [
        model_payload(
            _anchor(
                review,
                page_number=2,
                excerpt="Table 1. Synthetic Benchmark split test evaluation",
                kind=EvidenceKind.TABLE,
            )
        )
    ]
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert not (review / "review-lock.json").exists()


@pytest.mark.parametrize("excerpt", ["174.6%", "74.60%", "1,074.6%"])
def test_raw_value_support_rejects_numeric_substrings(excerpt: str) -> None:
    assert not reviewed_workflow._raw_value_supported(_candidate(), excerpt)


def test_raw_value_support_rejects_lexical_substrings() -> None:
    candidate = _nonnumeric_value_candidate(
        "Atlas Moderation API foobar percent result was observed."
    )
    assert not reviewed_workflow._raw_value_supported(
        candidate, "Atlas Moderation API foobar percent result was observed."
    )


@pytest.mark.parametrize(
    ("excerpt", "supported"),
    [
        ("We evaluated GPT-4 ourselves.", True),
        ("We evaluated GPT-4o ourselves.", False),
        ("We evaluated GPT-40 ourselves.", False),
        ("We evaluated GPT-4.1 ourselves.", False),
        ("We evaluated GPT-4.5 ourselves.", False),
        ("We evaluated GPT-4-mini ourselves.", False),
        ("We evaluated GPT-4/vision ourselves.", False),
        ("GPT methods used four tests.", False),
    ],
)
def test_system_identity_requires_exact_alphanumeric_entity(excerpt: str, supported: bool) -> None:
    payload = model_payload(_candidate())
    payload["observation_id"] = None
    payload["roles"][0]["raw_name"] = "GPT-4"
    candidate = CandidateObservation.model_validate(payload)

    assert reviewed_workflow._system_is_named(candidate, excerpt) is supported


def test_result_attestation_cannot_bind_base_model_to_versioned_variant(
    tmp_path: Path,
) -> None:
    _, review = _prepare(tmp_path, candidates=[_identifier_variant_candidate("GPT-4.1")])
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "system attestation" in str(captured.value)


def test_origin_attestation_cannot_bind_base_model_to_versioned_variant(
    tmp_path: Path,
) -> None:
    _, review = _prepare(tmp_path, candidates=[_identifier_variant_candidate("GPT-4")])
    decision = _complete_decision(review)
    payload = model_payload(decision)
    payload["origin_evidence"] = [
        model_payload(
            _anchor(
                review,
                page_number=1,
                excerpt="We evaluated GPT-4.1 ourselves on the synthetic benchmark.",
                kind=EvidenceKind.PROSE,
            )
        )
    ]
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.ORIGIN_EVIDENCE_INVALID


def test_nonnumeric_raw_projection_cannot_use_human_identity_cure(tmp_path: Path) -> None:
    candidate = _nonnumeric_value_candidate("Atlas Moderation API foo percent result was observed.")
    _, review = _prepare(tmp_path, candidates=[candidate])
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "value attestation" in str(captured.value)
    assert not (review / REVIEW_LOCK_NAME).exists()


def test_substring_only_nonnumeric_raw_value_cannot_be_locked(tmp_path: Path) -> None:
    _, review = _prepare(
        tmp_path,
        candidates=[
            _nonnumeric_value_candidate("Atlas Moderation API foobar percent result was observed.")
        ],
    )
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID


@pytest.mark.parametrize(
    ("raw", "numeric", "unit"),
    [
        ("95% CI [0.7,0.8]", 95.0, "percent"),
        ("0.7762 (0.12 %)", 0.7762, None),
    ],
)
def test_multi_number_raw_cannot_be_human_attested_as_one_point_estimate(
    tmp_path: Path,
    raw: str,
    numeric: float,
    unit: str | None,
) -> None:
    candidate = _multi_number_raw_candidate(raw, numeric, unit=unit)
    _, review = _prepare(tmp_path, candidates=[candidate])
    [item] = _records(review / REVIEW_ITEMS_NAME, ReviewItem)
    value_binding = next(
        binding
        for binding in item.reviewed_tuple.field_provenance
        if binding.field is CandidateField.VALUE
    )
    assert value_binding.reason == "value_numeric_projection_not_supported_by_raw"
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "value attestation" in str(captured.value)


@pytest.mark.parametrize(
    "schema_version",
    ["candidate-observation/0.2", "candidate-observation/0.3"],
)
@pytest.mark.parametrize(
    ("raw", "numeric", "unit"),
    [
        ("95% CI [0.7,0.8]", 95.0, "percent"),
        ("0.7762 (0.12 %)", 0.7762, None),
    ],
)
def test_multi_number_raw_fails_schema_independent_composer_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_version: str,
    raw: str,
    numeric: float,
    unit: str | None,
) -> None:
    candidate = _multi_number_raw_candidate(
        raw,
        numeric,
        unit=unit,
        schema_version=schema_version,
    )
    sealed, review = _prepare(tmp_path, candidates=[candidate])
    _complete_decision(review)
    real_support = reviewed_workflow._field_attestation_has_direct_support

    def legacy_lock_support(**kwargs: Any) -> bool:
        if kwargs["field"] is CandidateField.VALUE:
            return True
        return real_support(**kwargs)

    derived = tmp_path / "derived"
    with monkeypatch.context() as patch:
        patch.setattr(
            reviewed_workflow,
            "_field_attestation_has_direct_support",
            legacy_lock_support,
        )
        patch.setattr(
            reviewed_workflow,
            "_attestations_share_coherent_result_occurrence",
            lambda **_: True,
        )
        patch.setattr(
            reviewed_workflow,
            "_validate_attested_tuple_association",
            lambda **_: None,
        )
        validate_export_review(review)
        _compose_reviewed_fixture(sealed, review, derived)

    [outcome] = _outcomes(derived)
    assert outcome["state"] == "failed"
    assert outcome["failure_codes"] == ["VALUE_EVIDENCE_CONTRADICTION"]
    assert not list(derived.glob("*/eee/*.json"))
    verify_derived_run(derived)


def test_value_attestation_accepts_local_uncertainty_method(tmp_path: Path) -> None:
    _, review = _prepare(
        tmp_path,
        candidates=[
            _uncertainty_method_candidate(
                "Atlas Moderation API Synthetic Benchmark split test AUC 72.3% with bootstrap"
            )
        ],
    )
    _complete_decision(review)

    lock, _ = validate_export_review(review)

    assert lock.completed_count == 1


def test_value_attestation_cannot_borrow_uncertainty_method_from_another_clause(
    tmp_path: Path,
) -> None:
    _, review = _prepare(
        tmp_path,
        candidates=[
            _uncertainty_method_candidate(
                "Atlas Moderation API Synthetic Benchmark split test AUC 72.3%; Baseline bootstrap"
            )
        ],
    )
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "value attestation" in str(captured.value)


def test_metric_context_anchor_can_accompany_raw_value_anchor(tmp_path: Path) -> None:
    _, review = _prepare(tmp_path, candidates=[_metric_context_candidate()])
    _complete_decision(review)

    lock, _ = validate_export_review(review)

    assert lock.completed_count == 1


def test_value_attestation_can_select_only_the_exact_raw_value_span(tmp_path: Path) -> None:
    _, review = _prepare(tmp_path, candidates=[_metric_context_candidate()])
    decision = _complete_decision(review)
    payload = model_payload(decision)
    row_span_id = next(
        anchor["span_id"]
        for anchor in payload["result_evidence"]
        if anchor["exact_excerpt"] == "Atlas Moderation API  74.6%"
    )
    value_attestation = next(
        item for item in payload["field_attestations"] if item["field"] == "value"
    )
    value_attestation["span_ids"] = [row_span_id]
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    lock, _ = validate_export_review(review)

    assert lock.completed_count == 1


def test_value_attestation_cannot_cite_only_a_caption_without_the_raw_value(
    tmp_path: Path,
) -> None:
    _, review = _prepare(tmp_path, candidates=[_metric_context_candidate()])
    decision = _complete_decision(review)
    payload = model_payload(decision)
    caption_span_id = next(
        anchor["span_id"]
        for anchor in payload["result_evidence"]
        if anchor["exact_excerpt"] == "Table 1. Synthetic Benchmark split test evaluation (AUC)"
    )
    value_attestation = next(
        item for item in payload["field_attestations"] if item["field"] == "value"
    )
    value_attestation["span_ids"] = [caption_span_id]
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "value attestation" in str(captured.value)


def test_field_attestations_cannot_join_system_and_value_from_different_rows(
    tmp_path: Path,
) -> None:
    _, review = _prepare(tmp_path, candidates=[_frankenstein_candidate()])
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "coherent result occurrence" in str(captured.value)


def test_one_exact_span_cannot_launder_a_cross_row_frankenstein_tuple(
    tmp_path: Path,
) -> None:
    _, review = _prepare(tmp_path, candidates=[_single_span_frankenstein_candidate()])
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "coherent result occurrence" in str(captured.value)


def test_context_attestations_cannot_splice_dataset_and_metric_from_another_row(
    tmp_path: Path,
) -> None:
    _, review = _prepare(
        tmp_path,
        candidates=[_cross_row_context_frankenstein_candidate()],
    )
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "coherent atomic tuple occurrence" in str(captured.value)


@pytest.mark.parametrize("case", ["microsoft_table", "baseline_table", "utc_prose"])
def test_production_shaped_multispan_field_maps_lock_compose_and_verify(
    tmp_path: Path,
    case: str,
) -> None:
    if case == "microsoft_table":
        candidate = _production_microsoft_table_candidate()
        mapping = {
            CandidateField.SYSTEM: (MICROSOFT_ROW,),
            CandidateField.DATASET_SCOPE: (MICROSOFT_DATASET, MICROSOFT_ROW),
            CandidateField.METRIC: (MICROSOFT_HEADER, MICROSOFT_ROW),
            CandidateField.SETTING: (
                MICROSOFT_HEADER,
                MICROSOFT_SUBSET,
                MICROSOFT_DATASET,
                MICROSOFT_SAMPLES,
                MICROSOFT_ROW,
            ),
            CandidateField.VALUE: (MICROSOFT_ROW,),
            CandidateField.UNIT: (MICROSOFT_ROW,),
        }
        origin_excerpt = "We evaluated Microsoft ourselves on Civil Comments."
    elif case == "baseline_table":
        candidate = _production_baseline_table_candidate()
        mapping = {
            CandidateField.SYSTEM: (BASELINE_ROW,),
            CandidateField.DATASET_SCOPE: (
                BASELINE_ROW,
                BASELINE_CAPTION,
                BASELINE_CAPTION_CONTINUATION,
            ),
            CandidateField.METRIC: (
                BASELINE_ROW,
                BASELINE_CAPTION,
                BASELINE_CAPTION_CONTINUATION,
            ),
            CandidateField.SETTING: (
                BASELINE_HEADER,
                BASELINE_ROW,
                BASELINE_CAPTION,
                BASELINE_CAPTION_CONTINUATION,
            ),
            CandidateField.VALUE: (BASELINE_HEADER, BASELINE_ROW),
            CandidateField.UNIT: (
                BASELINE_ROW,
                BASELINE_CAPTION,
                BASELINE_CAPTION_CONTINUATION,
            ),
        }
        origin_excerpt = "We trained and evaluated the Baseline ourselves."
    else:
        candidate = _production_utc_prose_candidate()
        mapping = {
            CandidateField.SYSTEM: (UTC_RESULT,),
            CandidateField.DATASET_SCOPE: (UTC_SCOPE,),
            CandidateField.METRIC: (UTC_RESULT,),
            CandidateField.VALUE: (UTC_RESULT,),
            CandidateField.UNIT: (UTC_RESULT,),
        }
        origin_excerpt = "We present and evaluate UTC† ourselves."

    sealed, review = _prepare(tmp_path, candidates=[candidate])
    if case == "baseline_table":
        [item] = _records(review / REVIEW_ITEMS_NAME, ReviewItem)
        unit_binding = next(
            binding
            for binding in item.reviewed_tuple.field_provenance
            if binding.field is CandidateField.UNIT
        )
        assert item.reviewed_tuple.metric is not None
        assert item.reviewed_tuple.metric.kind == "auroc"
        assert item.reviewed_tuple.metric.unit == "proportion"
        assert unit_binding.status is FieldBindingStatus.BOUND
        assert unit_binding.reason == (
            "deterministic_reference_resolution="
            "unit_resolved_as_proportion_from_registry_bounded_metric_and_printed_decimal"
        )
    _complete_decision(review)
    _replace_completed_field_map(
        review,
        field_excerpts=mapping,
        origin_excerpt=origin_excerpt,
    )

    lock, _ = validate_export_review(review)
    assert lock.completed_count == 1
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    assert _outcomes(derived)[0]["state"] == "exported"
    assert verify_derived_run(derived).status == "verified"
    assert (
        verify_contextual_derived_run(
            derived,
            run_root=sealed,
            review_root=review,
        ).verification.status
        == "verified"
    )


@pytest.mark.parametrize(
    "field",
    [CandidateField.DATASET_SCOPE, CandidateField.METRIC],
)
def test_field_attestation_cannot_bless_primitive_absent_from_source_page(
    tmp_path: Path, field: CandidateField
) -> None:
    _, review = _prepare(tmp_path, candidates=[_absent_field_candidate(field)])
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert f"{field.value} attestation" in str(captured.value)


@pytest.mark.parametrize(
    "field",
    [CandidateField.DATASET_SCOPE, CandidateField.METRIC],
)
def test_field_attestation_must_cite_the_span_supporting_its_primitive(
    tmp_path: Path, field: CandidateField
) -> None:
    _, review = _prepare(tmp_path, candidates=[_metric_context_candidate()])
    decision = _complete_decision(review)
    payload = model_payload(decision)
    row_span_id = next(
        anchor["span_id"]
        for anchor in payload["result_evidence"]
        if anchor["exact_excerpt"] == "Atlas Moderation API  74.6%"
    )
    attestation = next(
        item for item in payload["field_attestations"] if item["field"] == field.value
    )
    attestation["span_ids"] = [row_span_id]
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert f"{field.value} attestation" in str(captured.value)


@pytest.mark.parametrize(
    "field",
    [CandidateField.SYSTEM, CandidateField.DATASET_SCOPE, CandidateField.METRIC],
)
def test_field_attestation_cannot_bless_fabricated_entity_metadata(
    tmp_path: Path, field: CandidateField
) -> None:
    _, review = _prepare(tmp_path, candidates=[_fabricated_metadata_candidate(field)])
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert f"{field.value} attestation" in str(captured.value)


def test_setting_attestation_cannot_borrow_sample_count_from_metric_range(
    tmp_path: Path,
) -> None:
    _, review = _prepare(
        tmp_path,
        candidates=[_sample_count_borrowing_metric_range_candidate()],
    )
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "setting attestation" in str(captured.value)


def test_unit_attestation_cannot_borrow_percent_from_another_value(
    tmp_path: Path,
) -> None:
    _, review = _prepare(
        tmp_path,
        candidates=[_unit_borrowing_unrelated_percent_candidate()],
    )
    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "unit attestation" in str(captured.value)


def test_confirmed_tuple_cannot_drop_metric_field_quote_anchor(tmp_path: Path) -> None:
    _, review = _prepare(tmp_path, candidates=[_metric_context_candidate()])
    decision = _complete_decision(review)
    payload = model_payload(decision)
    payload["result_evidence"] = [
        anchor
        for anchor in payload["result_evidence"]
        if anchor["exact_excerpt"] == "Atlas Moderation API  74.6%"
    ]
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "field attestation" in str(captured.value)


def test_split_system_name_context_can_accompany_raw_value_anchor(tmp_path: Path) -> None:
    _, review = _prepare(tmp_path, candidates=[_split_system_name_candidate()])
    decision = _complete_decision(review)

    assert [anchor.exact_excerpt for anchor in decision.result_evidence] == [
        "Synthetic Benchmark split test Agreement is 96.1% with Tox-",
        "icBert on the benchmark.",
    ]
    payload = model_payload(decision)
    payload["origin_evidence"] = [
        model_payload(
            _anchor(
                review,
                page_number=1,
                excerpt="We evaluated ToxicBert ourselves on the synthetic benchmark.",
                kind=EvidenceKind.PROSE,
            )
        )
    ]
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])
    lock, _ = validate_export_review(review)
    assert lock.completed_count == 1


def test_human_field_attestation_cures_field_provenance_only_blocker(tmp_path: Path) -> None:
    sealed, review = _prepare(tmp_path, candidates=[_metric_context_candidate()])
    [item] = _records(review / REVIEW_ITEMS_NAME, ReviewItem)
    assert item.base_gate_reason == "field_provenance=ambiguous_or_unsupported"
    nonbound_reasons = {
        binding.reason
        for binding in item.reviewed_tuple.field_provenance
        if binding.status is not FieldBindingStatus.BOUND
        and binding.field in item.reviewed_tuple.populated_fields()
    }
    assert nonbound_reasons
    assert nonbound_reasons <= {
        "dataset_scope_uses_table_context",
        "field_identity_not_supported_by_evidence_quotes",
        "field_not_supported_by_evidence_quotes",
        "field_partially_supported_by_evidence_quotes",
        "load_bearing_tuple_association_not_supported_by_evidence_quotes",
        "setting_uses_table_context",
    }
    _complete_decision(review)
    validate_export_review(review)

    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)

    assert _outcomes(derived)[0]["state"] == "exported"
    [provenance] = _records(derived / DERIVED_PROVENANCE_NAME, ReviewedExportProvenance)
    populated = item.reviewed_tuple.populated_fields()
    reviewed = {binding.field: binding for binding in provenance.field_provenance}
    assert all(reviewed[field].status is FieldBindingStatus.BOUND for field in populated)
    assert all(
        any(source.kind is FieldSourceKind.REVIEW_SPAN for source in reviewed[field].sources)
        for field in populated
    )


@pytest.mark.parametrize(
    ("kind", "expected_reason"),
    [
        ("numeric", "value_numeric_projection_not_supported_by_raw"),
        (
            "comparator",
            "value_comparator_or_uncertainty_not_supported_by_evidence_quotes",
        ),
        (
            "uncertainty",
            "value_comparator_or_uncertainty_not_supported_by_evidence_quotes",
        ),
    ],
)
def test_human_field_attestation_cannot_cure_deterministic_value_contradiction(
    tmp_path: Path, kind: str, expected_reason: str
) -> None:
    candidate = _value_contradiction_candidate(kind)
    _, review = _prepare(tmp_path, candidates=[candidate])
    [item] = _records(review / REVIEW_ITEMS_NAME, ReviewItem)
    value_binding = next(
        binding
        for binding in item.reviewed_tuple.field_provenance
        if binding.field is CandidateField.VALUE
    )
    assert value_binding.status is not FieldBindingStatus.BOUND
    assert value_binding.reason == expected_reason

    _complete_decision(review)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
    assert "value attestation" in str(captured.value)
    assert not (review / REVIEW_LOCK_NAME).exists()


@pytest.mark.parametrize("kind", ["numeric", "comparator", "uncertainty"])
def test_legacy_value_contradiction_fails_full_reviewed_composition(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _legacy_value_contradiction_candidate(kind)
    sealed, review = _prepare(tmp_path, candidates=[candidate])
    _complete_decision(review)
    derived = tmp_path / "derived"
    real_support = reviewed_workflow._field_attestation_has_direct_support

    def legacy_lock_support(**kwargs: Any) -> bool:
        if kwargs["field"] is CandidateField.VALUE:
            return True
        return real_support(**kwargs)

    # Simulate a previously locked review created before strict VALUE
    # attestations.  The schema-independent compose gate must still fail closed.
    with monkeypatch.context() as patch:
        patch.setattr(
            reviewed_workflow,
            "_field_attestation_has_direct_support",
            legacy_lock_support,
        )
        patch.setattr(
            reviewed_workflow,
            "_attestations_share_coherent_result_occurrence",
            lambda **_: True,
        )
        patch.setattr(
            reviewed_workflow,
            "_validate_attested_tuple_association",
            lambda **_: None,
        )
        validate_export_review(review)
        _compose_reviewed_fixture(sealed, review, derived)

    [outcome] = _outcomes(derived)
    assert outcome["state"] == "failed"
    assert outcome["failure_codes"] == ["VALUE_EVIDENCE_CONTRADICTION"]
    assert not list(derived.glob("*/eee/*.json"))
    verify_derived_run(derived)


@pytest.mark.parametrize(
    "missing_field",
    [CandidateField.METRIC, CandidateField.DATASET_SCOPE, CandidateField.SETTING],
)
def test_confirmed_tuple_requires_every_populated_field_attestation(
    tmp_path: Path, missing_field: CandidateField
) -> None:
    _, review = _prepare(tmp_path, candidates=[_metric_context_candidate()])
    decision = _complete_decision(review)
    payload = model_payload(decision)
    payload["field_attestations"] = [
        item for item in payload["field_attestations"] if item["field"] != missing_field.value
    ]
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID


def test_non_evaluated_role_requires_matching_setting_attestation(tmp_path: Path) -> None:
    candidate = _instrument_only_setting_candidate()
    _, review = _prepare(tmp_path, candidates=[candidate])
    [item] = _records(review / REVIEW_ITEMS_NAME, ReviewItem)

    assert CandidateField.SETTING in candidate.populated_fields()
    assert CandidateField.SETTING in item.reviewed_tuple.populated_fields()
    assert (
        item.reviewed_tuple.field_value_sha256s()[CandidateField.SETTING]
        == candidate.field_value_sha256s()[CandidateField.SETTING]
    )

    decision = _complete_decision(review)
    payload = model_payload(decision)
    payload["field_attestations"] = [
        attestation
        for attestation in payload["field_attestations"]
        if attestation["field"] != CandidateField.SETTING.value
    ]
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID


def test_confirmed_tuple_rejects_stale_field_value_hash(tmp_path: Path) -> None:
    _, review = _prepare(tmp_path, candidates=[_metric_context_candidate()])
    decision = _complete_decision(review)
    payload = model_payload(decision)
    metric = next(item for item in payload["field_attestations"] if item["field"] == "metric")
    metric["value_sha256"] = "0" * 64
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID


def test_confirmed_tuple_requires_canonical_field_attestation_order(tmp_path: Path) -> None:
    _, review = _prepare(tmp_path, candidates=[_metric_context_candidate()])
    decision = _complete_decision(review)
    payload = model_payload(decision)
    payload["field_attestations"].reverse()
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.OBSERVATION_LEDGER_INVALID


@pytest.mark.parametrize("corruption", ["tampered_span", "unknown_span"])
def test_confirmed_tuple_rejects_invalid_field_span_binding(
    tmp_path: Path, corruption: str
) -> None:
    _, review = _prepare(tmp_path, candidates=[_metric_context_candidate()])
    decision = _complete_decision(review)
    payload = model_payload(decision)
    if corruption == "tampered_span":
        payload["result_evidence"][0]["span_id"] = "span_" + "0" * 64
    else:
        payload["field_attestations"][0]["span_ids"] = ["span_" + "f" * 64]
    write_jsonl(review / REVIEW_DECISIONS_NAME, [payload])

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID


def test_human_field_attestation_cannot_cure_physical_conflict(tmp_path: Path) -> None:
    sealed, review = _prepare(tmp_path, candidates=[_physical_conflict_candidate()])
    [item] = _records(review / REVIEW_ITEMS_NAME, ReviewItem)
    assert "physical-cell conflict" in (item.base_gate_reason or "")
    _complete_decision(review)
    validate_export_review(review)

    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)

    assert "PHYSICAL_CELL_CONFLICT" in _outcomes(derived)[0]["failure_codes"]
    assert not list(derived.glob("*/eee/*.json"))


def test_human_field_attestation_cannot_cure_field_conflict(tmp_path: Path) -> None:
    sealed, review = _prepare(tmp_path, candidates=[_field_conflict_candidate()])
    [item] = _records(review / REVIEW_ITEMS_NAME, ReviewItem)
    assert item.base_gate_reason == "field_provenance=conflict"
    _complete_decision(review)
    validate_export_review(review)

    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)

    assert "FIELD_PROVENANCE_CONFLICT" in _outcomes(derived)[0]["failure_codes"]
    assert not list(derived.glob("*/eee/*.json"))


def test_tampered_copied_observation_ledger_fails_immutable_hash_check(
    tmp_path: Path,
) -> None:
    _, review = _prepare(tmp_path)
    manifest = ReviewManifest.model_validate(read_json(review / REVIEW_MANIFEST_NAME))
    observations = review / manifest.papers[0].observations.review_copy.path
    observations.chmod(0o600)
    observations.write_text(
        observations.read_text(encoding="utf-8").replace("74.6", "75.6"),
        encoding="utf-8",
    )

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.ARTIFACT_HASH_MISMATCH


def test_rebound_tampered_review_candidate_still_fails_source_fingerprint(
    tmp_path: Path,
) -> None:
    _, review = _prepare(tmp_path)
    items_path = review / REVIEW_ITEMS_NAME
    [item] = [json.loads(line) for line in items_path.read_text().splitlines()]
    item["candidate"]["notes"].append("tampered review candidate")
    items_path.chmod(0o600)
    items_sha = write_jsonl(items_path, [item])

    manifest_payload = read_json(review / REVIEW_MANIFEST_NAME)
    manifest_payload["items"]["sha256"] = items_sha
    manifest_payload["items"]["size_bytes"] = items_path.stat().st_size
    write_json(review / REVIEW_MANIFEST_NAME, manifest_payload)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.CANDIDATE_FINGERPRINT_MISMATCH


def test_schema_failure_produces_zero_eee_and_a_typed_failed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed, review = _prepare(tmp_path)
    _complete_decision(review)
    validate_export_review(review)
    real_compose = reviewed_workflow.compose_eee_records

    def compose_invalid_record(**kwargs: Any) -> list[dict[str, Any]]:
        records = real_compose(**kwargs)
        for record in records:
            record.pop("model_info")
        return records

    monkeypatch.setattr(reviewed_workflow, "compose_eee_records", compose_invalid_record)
    derived = tmp_path / "derived"
    manifest = _compose_reviewed_fixture(sealed, review, derived)

    assert not list(derived.glob("*/eee/*.json"))
    assert _outcomes(derived)[0]["state"] == "failed"
    assert _outcomes(derived)[0]["failure_codes"] == ["EEE_SCHEMA_INVALID"]
    assert manifest.counts["eee_records"] == 0
    assert manifest.counts["eee_observations"] == 0
    assert verify_derived_run(derived).eee_record_count == 0


def test_duplicate_physical_cells_are_never_composed(tmp_path: Path) -> None:
    first = _candidate()
    second = _candidate()
    second.observation_id = "obs_distinct_duplicate_cell"
    sealed = _build_sealed_run(tmp_path, candidates=[first, second])
    review = tmp_path / "review"
    prepare_export_review(run_root=sealed, output_root=review)
    items = _records(review / REVIEW_ITEMS_NAME, ReviewItem)
    assert len(items) == 2
    assert (
        items[0].fingerprint.structural_identity_sha256
        == items[1].fingerprint.structural_identity_sha256
    )
    _complete_decisions(review)
    validate_export_review(review)

    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)
    outcomes = _outcomes(derived)
    assert len(outcomes) == 2
    assert {tuple(outcome["failure_codes"]) for outcome in outcomes} == {
        ("PHYSICAL_CELL_DUPLICATE",)
    }
    assert not list(derived.glob("*/eee/*.json"))


@pytest.mark.parametrize("difference", ["uncertainty", "comparator"])
def test_same_physical_cell_with_different_value_interpretation_is_never_both_composed(
    tmp_path: Path, difference: str
) -> None:
    candidates = list(_same_physical_cell_candidates(difference))
    sealed = _build_sealed_run(tmp_path, candidates=candidates)
    review = tmp_path / "review"
    prepare_export_review(run_root=sealed, output_root=review)
    items = _records(review / REVIEW_ITEMS_NAME, ReviewItem)
    assert len(items) == 2
    assert len({item.fingerprint.observation_id for item in items}) == 2
    assert len({item.fingerprint.structural_identity_sha256 for item in items}) == 1
    _complete_decisions(review)
    if difference == "comparator":
        with pytest.raises(ReviewedExportError) as captured:
            validate_export_review(review)
        assert captured.value.code is ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID
        assert "value attestation" in str(captured.value)
        return
    validate_export_review(review)

    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)

    outcomes = _outcomes(derived)
    assert len(outcomes) == 2
    assert all(outcome["state"] == "failed" for outcome in outcomes)
    assert all(outcome["failure_codes"] for outcome in outcomes)
    assert len(list(derived.glob("*/eee/*.json"))) == sum(
        outcome["state"] == "exported" for outcome in outcomes
    )
    verify_derived_run(derived, run_root=sealed, review_root=review)


@pytest.mark.parametrize(
    ("candidate_pair", "structural_id_count"),
    [
        pytest.param(_mixed_anchor_granularity_candidates, 1, id="structured-and-legacy"),
        pytest.param(_same_value_occurrence_with_extra_context, 1, id="extra-context"),
        pytest.param(_same_value_occurrence_with_whitespace_variant, 1, id="whitespace"),
        pytest.param(_same_value_occurrence_with_narrow_and_wide_quotes, 1, id="quote-width"),
        pytest.param(_same_claimed_cell_with_descriptive_drift, 2, id="descriptive-drift"),
    ],
)
def test_same_physical_result_is_not_duplicated_by_anchor_variants(
    tmp_path: Path,
    candidate_pair: Callable[[], tuple[CandidateObservation, CandidateObservation]],
    structural_id_count: int,
) -> None:
    sealed, review = _prepare(tmp_path, candidates=list(candidate_pair()))
    items = _records(review / REVIEW_ITEMS_NAME, ReviewItem)
    assert len({item.fingerprint.observation_id for item in items}) == 2
    assert (
        len({item.fingerprint.structural_identity_sha256 for item in items}) == structural_id_count
    )
    _complete_decisions(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)

    outcomes = _outcomes(derived)
    assert len(outcomes) == 2
    assert {tuple(outcome["failure_codes"]) for outcome in outcomes} == {
        ("PHYSICAL_CELL_DUPLICATE",)
    }
    assert not list(derived.glob("*/eee/*.json"))


def test_duplicate_eee_output_path_fails_before_any_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed, review = _prepare(tmp_path)
    _complete_decisions(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    real_compose = reviewed_workflow.compose_eee_records

    def duplicate_path_records(**kwargs: Any) -> list[dict[str, Any]]:
        records = real_compose(**kwargs)
        assert len(records) == 1
        return [records[0], json.loads(json.dumps(records[0]))]

    with monkeypatch.context() as patch:
        patch.setattr(reviewed_workflow, "compose_eee_records", duplicate_path_records)
        with pytest.raises(ReviewedExportError) as captured:
            _compose_reviewed_fixture(sealed, review, derived)

    assert captured.value.code is ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE
    assert "duplicate EEE output path" in str(captured.value)
    assert not derived.exists()


def test_fully_rehashed_duplicate_exports_fail_contextual_duplicate_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _candidate()
    second = _candidate()
    second.observation_id = "obs_contextual_duplicate"
    candidates = [first, second]
    sealed = _build_sealed_run(tmp_path, candidates=candidates)
    review = tmp_path / "review"
    prepare_export_review(run_root=sealed, output_root=review)
    _complete_decisions(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    with monkeypatch.context() as patch:
        patch.setattr(
            reviewed_workflow,
            "_approved_duplicate_failure_codes",
            lambda _: {},
        )
        patch.setattr(
            reviewed_workflow,
            "_assert_unique_eee_output_path",
            lambda *_: None,
        )
        _compose_reviewed_fixture(sealed, review, derived)
    _fully_rehash_derived(derived)

    assert [outcome["state"] for outcome in _outcomes(derived)] == ["exported", "exported"]
    verify_derived_run(derived)
    with pytest.raises(ReviewedExportError) as captured:
        verify_derived_run(derived, run_root=sealed, review_root=review)
    assert captured.value.code is ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE
    assert "duplicate classification" in str(captured.value)


def test_fully_rebound_structural_fingerprint_cannot_bypass_duplicate_gate(
    tmp_path: Path,
) -> None:
    first = _candidate()
    second = _candidate()
    second.observation_id = "obs_distinct_duplicate_cell"
    sealed = _build_sealed_run(tmp_path, candidates=[first, second])
    review = tmp_path / "review"
    prepare_export_review(run_root=sealed, output_root=review)

    items_path = review / REVIEW_ITEMS_NAME
    items = [json.loads(line) for line in items_path.read_text().splitlines()]
    items[1]["fingerprint"]["structural_identity_sha256"] = "9" * 64
    items_path.chmod(0o600)
    items_sha = write_jsonl(items_path, items)
    manifest_path = review / REVIEW_MANIFEST_NAME
    manifest = read_json(manifest_path)
    manifest["items"].update(
        {
            "sha256": items_sha,
            "size_bytes": items_path.stat().st_size,
            "records": len(items),
        }
    )
    write_json(manifest_path, manifest)

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.CANDIDATE_FINGERPRINT_MISMATCH


def test_registry_replay_keeps_source_observation_id_for_reviewed_composition(
    tmp_path: Path,
) -> None:
    source_candidate = _canonicalization_drift_candidate()
    source_id = source_candidate.observation_id
    assert source_id is not None
    sealed, review = _prepare(tmp_path, candidates=[source_candidate])
    [item] = _records(review / REVIEW_ITEMS_NAME, ReviewItem)
    assert item.fingerprint.observation_id == source_id
    assert item.reviewed_tuple.metric is not None
    assert item.reviewed_tuple.metric.canonical_id == "auroc"
    assert item.reviewed_tuple.metric.kind == "auroc"
    assert item.reviewed_tuple.metric.model_dump(mode="json") != source_candidate.metric.model_dump(
        mode="json"
    )

    _complete_decision(review)
    validate_export_review(review)
    derived = tmp_path / "derived"
    _compose_reviewed_fixture(sealed, review, derived)

    [outcome] = _outcomes(derived)
    assert outcome["state"] == "exported"
    assert outcome["evaluation_result_id"] == source_id
    record = read_json(derived / outcome["eee_path"])
    [result] = record["evaluation_results"]
    assert result["evaluation_result_id"] == source_id
    assert result["metric_config"]["metric_id"] == "auroc"
    verify_derived_run(derived, run_root=sealed, review_root=review)


def test_prepare_and_compose_reject_output_overlap(tmp_path: Path) -> None:
    sealed = _build_sealed_run(tmp_path)
    with pytest.raises(ReviewedExportError) as captured:
        prepare_export_review(run_root=sealed, output_root=sealed / "review")
    assert captured.value.code is ReviewedExportErrorCode.OUTPUT_OVERLAP

    review = tmp_path / "review"
    prepare_export_review(run_root=sealed, output_root=review)
    _complete_decision(review)
    validate_export_review(review)
    with pytest.raises(ReviewedExportError) as captured:
        compose_reviewed_eee(
            run_root=sealed,
            decisions_path=review / REVIEW_DECISIONS_NAME,
            output_root=review / "derived",
        )
    assert captured.value.code is ReviewedExportErrorCode.OUTPUT_OVERLAP


def test_dangling_output_symlink_is_treated_as_an_existing_output(tmp_path: Path) -> None:
    sealed = _build_sealed_run(tmp_path)
    output = tmp_path / "dangling-output"
    output.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)

    with pytest.raises(ReviewedExportError) as captured:
        prepare_export_review(run_root=sealed, output_root=output)
    assert captured.value.code is ReviewedExportErrorCode.OUTPUT_EXISTS


def test_symlink_in_review_inputs_cannot_be_locked(tmp_path: Path) -> None:
    sealed, review = _prepare(tmp_path)
    manifest = ReviewManifest.model_validate(read_json(review / REVIEW_MANIFEST_NAME))
    layout_copy = review / manifest.papers[0].layout.review_copy.path
    layout_copy.unlink()
    layout_copy.symlink_to(sealed / PAPER_ID / "private" / "layout.json")

    with pytest.raises(ReviewedExportError) as captured:
        validate_export_review(review)
    assert captured.value.code is ReviewedExportErrorCode.ARTIFACT_TREE_INVALID


def test_derived_tree_contains_no_private_evidence_or_reviewer_identity(
    tmp_path: Path,
) -> None:
    sealed, review = _prepare(tmp_path)
    _complete_decision(review, mode=ReviewAuthorityMode.ADJUDICATED)
    validate_export_review(review)
    derived = tmp_path / "derived"
    manifest = _compose_reviewed_fixture(sealed, review, derived)

    public_bytes = b"\n".join(
        path.read_bytes() for path in sorted(derived.rglob("*")) if path.is_file()
    )
    for private_value in (
        b"reviewer-a",
        b"reviewer-b",
        b"adjudicator-c",
        b"exact_excerpt",
        b"We evaluated Atlas Moderation API ourselves",
        str(tmp_path).encode(),
        b"/Users/",
    ):
        assert private_value not in public_bytes
    assert manifest.contains_evidence_quotations is False
    assert manifest.contains_reviewer_identities is False
    assert manifest.contains_absolute_paths is False
    verify_derived_run(derived)
