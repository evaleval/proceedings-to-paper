"""Tiered producer-origin export, descriptive field provenance, and offline replay.

Canonical export means positively established origin, and the deterministic resolver
never establishes it. These tests pin the two things that make a tier honest: the default
policy behaves exactly as before, and a record granted export under the tiered policy
carries the basis it was granted on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from proceedings_to_eee.census_recompose import recompose_paper
from proceedings_to_eee.domain.attribution import (
    AttributionState,
    AttributionVerdict,
    OriginBasis,
    OriginExportPolicy,
)
from proceedings_to_eee.domain.observation import (
    CandidateObservation,
    EvidenceAnchor,
    MetricSpec,
    ObservationScope,
    ReportedValue,
    RoleAssignment,
)
from proceedings_to_eee.domain.status import (
    ActorRole,
    ClaimType,
    EvidenceKind,
    ExportStatus,
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.validation.candidates import (
    candidate_origin_basis,
    route_candidate_attribution,
    validate_candidates,
)
from proceedings_to_eee.validation.eee_schema import load_schema
from proceedings_to_eee.validation.field_provenance import quote_field_provenance

SOURCE_ID = "src_tiered_fixture"
PAPER_ID = "tiered-fixture"
QUOTE = "PerspectiveAPI reaches 0.81 macro F1 on the HateCheck test split."
PAGE_TEXT = f"Results\n\n{QUOTE}\n\nTable 1: Scores.\n"
_MANIFEST = json.dumps(
    {
        "schema_version": "source-manifest/0.1",
        "paper_id": PAPER_ID,
        "title": "Tiered fixture",
        "sources": [
            {
                "source_id": SOURCE_ID,
                "paper_id": PAPER_ID,
                "role": "paper",
                "original_uri": "https://example.invalid/paper.pdf",
                "resolved_uri": "https://example.invalid/paper.pdf",
                "retrieved_at": "2026-09-04T00:00:00+00:00",
                "sha256": "0" * 64,
                "byte_size": 10,
                "media_type": "application/pdf",
                "cache_relpath": "data/sources/fixture.pdf",
                "access_status": "available",
                "license_disposition": "unknown",
            }
        ],
    }
)


def _layout() -> PdfLayout:
    return PdfLayout(
        source_id=SOURCE_ID,
        parser="fixture",
        parser_version="fixture-1",
        page_count=1,
        pages=[
            PageFragment(
                fragment_id="fragment-1",
                source_id=SOURCE_ID,
                page=1,
                text=PAGE_TEXT,
                text_sha256=hashlib.sha256(PAGE_TEXT.encode()).hexdigest(),
                character_count=len(PAGE_TEXT),
                numeric_token_count=1,
                result_signal_score=1,
            )
        ],
    )


def _candidate(**overrides: object) -> CandidateObservation:
    payload: dict[str, object] = {
        # 0.3 is the version the field-provenance gate applies to.
        "schema_version": "candidate-observation/0.3",
        "paper_id": PAPER_ID,
        "claim_type": ClaimType.PRIMARY_RESULT,
        "extraction_confidence": 0.95,
        "roles": [RoleAssignment(role=ActorRole.EVALUATED_SYSTEM, raw_name="PerspectiveAPI")],
        "scope": ObservationScope(dataset_raw="HateCheck", split="test"),
        "metric": MetricSpec(raw_name="macro F1", canonical_id="macro_f1", unit="proportion"),
        "value": ReportedValue(raw="0.81", numeric=0.81, unit="proportion"),
        "evidence": [
            EvidenceAnchor(
                source_id=SOURCE_ID,
                page=1,
                kind=EvidenceKind.PROSE,
                quote=QUOTE,
            )
        ],
    }
    payload.update(overrides)
    candidate = CandidateObservation(**payload)
    # Provenance is attached at extraction time, not by validation, so a fixture that
    # wants the field-provenance gate to run has to bind it the same way.
    return candidate.model_copy(update={"field_provenance": quote_field_provenance(candidate)})


def test_the_default_policy_still_demotes_everything_the_resolver_cannot_establish() -> None:
    candidate = _candidate()
    candidate.export_status = ExportStatus.ELIGIBLE

    route_candidate_attribution([candidate], {})

    assert candidate.export_status is ExportStatus.NEEDS_REVIEW
    assert candidate.export_reason is not None
    assert candidate.export_reason.startswith("attribution=")
    assert candidate.attribution is not None
    assert candidate.attribution.state is AttributionState.UNRESOLVED


def test_the_tiered_policy_keeps_it_eligible_and_names_the_basis() -> None:
    candidate = _candidate()
    candidate.export_status = ExportStatus.ELIGIBLE

    route_candidate_attribution([candidate], {}, OriginExportPolicy.TIERED)

    assert candidate.export_status is ExportStatus.ELIGIBLE
    assert candidate_origin_basis(candidate) is OriginBasis.MODEL_ASSERTED_PRIMARY_UNCHECKED
    # The verdict itself is untouched: nothing here calls the number paper-produced.
    assert candidate.attribution is not None
    assert candidate.attribution.state is AttributionState.UNRESOLVED


def test_an_externally_sourced_candidate_never_exports_under_any_policy() -> None:
    candidate = _candidate()
    candidate.export_status = ExportStatus.ELIGIBLE
    candidate.attribution = AttributionVerdict(
        state=AttributionState.EXTERNALLY_SOURCED, rule_id="row_scoped_foreign_cue"
    )

    assert candidate_origin_basis(candidate) is OriginBasis.NONE
    assert not OriginExportPolicy.TIERED.permits(OriginBasis.NONE)
    assert not OriginExportPolicy.POSITIVE_ONLY.permits(OriginBasis.NONE)


def test_descriptive_setting_prose_no_longer_gates_but_a_literal_still_does() -> None:
    """`operationalization` is descriptive prose; `language` requires literal binding."""

    described = _candidate(operationalization="threshold 0.7 on the toxicity score")
    literal = _candidate(scope=ObservationScope(dataset_raw="HateCheck", language="Portuguese"))
    layouts = {SOURCE_ID: _layout()}

    validate_candidates([described], layouts, min_confidence=0.8)
    validate_candidates([literal], layouts, min_confidence=0.8)

    # The prose one clears field provenance and stops at attribution instead.
    assert described.export_reason is not None
    assert described.export_reason.startswith("attribution=")
    # The literal one never gets that far.
    assert literal.export_reason == "field_provenance=ambiguous_or_unsupported"


def test_recompose_writes_tiered_records_and_leaves_observations_untouched(
    tmp_path: Path,
) -> None:
    paper_dir = tmp_path / PAPER_ID
    (paper_dir / "private").mkdir(parents=True)
    candidate = _candidate()
    validate_candidates([candidate], {SOURCE_ID: _layout()}, min_confidence=0.8)
    observations = paper_dir / "observations.jsonl"
    observations.write_text(json.dumps(candidate.model_dump(mode="json")) + "\n", encoding="utf-8")
    before = observations.read_bytes()
    (paper_dir / "private" / "layout.json").write_text(
        json.dumps(_layout().model_dump(mode="json")), encoding="utf-8"
    )
    (paper_dir / "source-manifest.json").write_text(_MANIFEST, encoding="utf-8")
    schema, authority = load_schema()

    positive = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.POSITIVE_ONLY,
        schema=schema,
        schema_version=authority.version,
    )
    tiered = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )

    assert positive.records == []
    assert len(tiered.records) == 1
    details = tiered.records[0]["source_metadata"]["additional_details"]
    assert details["review_tier"] == "deterministic"
    assert details["producer_origin_basis"] == "model_asserted_primary_unchecked"
    assert details["origin_export_policy"] == "tiered"
    result_details = tiered.records[0]["evaluation_results"][0]["score_details"]["details"]
    assert result_details["producer_origin_basis"] == "model_asserted_primary_unchecked"
    assert result_details["attribution_state"] == "unresolved"
    assert result_details["review_tier"] == "deterministic"
    assert result_details["evidence_page"] == "1"
    # The replay reads the run; it never rewrites it.
    assert observations.read_bytes() == before
    # And it is idempotent.
    again = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )
    assert again.records == tiered.records


def test_a_reviewed_candidate_composes_at_the_model_reviewed_tier(tmp_path: Path) -> None:
    """This tier records a failed deterministic gate without treating it as a blocker."""

    from proceedings_to_eee.census_review import candidate_payload_sha256
    from proceedings_to_eee.validation.eee_schema import validate_eee_record

    paper_dir = tmp_path / PAPER_ID
    (paper_dir / "private").mkdir(parents=True)
    # A literal setting component that is not in the quote, so field provenance fails.
    candidate = _candidate(
        scope=ObservationScope(dataset_raw="HateCheck", split="test", language="Portuguese")
    )
    validate_candidates([candidate], {SOURCE_ID: _layout()}, min_confidence=0.8)
    assert candidate.export_status is not ExportStatus.ELIGIBLE
    record = candidate.model_dump(mode="json")
    (paper_dir / "observations.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (paper_dir / "private" / "layout.json").write_text(
        json.dumps(_layout().model_dump(mode="json")), encoding="utf-8"
    )
    (paper_dir / "source-manifest.json").write_text(_MANIFEST, encoding="utf-8")
    (paper_dir / "model-review-gates.json").write_text(
        json.dumps(
            {
                "schema_version": "census-model-review-gates/0.1",
                "model": "fixture/model",
                "gates": {
                    str(record["observation_id"]): {
                        "decision": "accept",
                        "candidate_payload_sha256": candidate_payload_sha256(record),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    schema, authority = load_schema()

    tiered = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )

    assert len(tiered.records) == 1
    details = tiered.records[0]["source_metadata"]["additional_details"]
    assert details["review_tier"] == "model_reviewed"
    assert details["producer_origin_basis"] == "model_reviewed_origin_quote"
    result_details = tiered.records[0]["evaluation_results"][0]["score_details"]["details"]
    # The deterministic failure is carried in the record rather than hidden by the tier.
    assert result_details["field_provenance_gate"] == "unbound"
    assert validate_eee_record(tiered.records[0], schema) == []

    # A gate written for a different tuple is ignored, not trusted.
    (paper_dir / "model-review-gates.json").write_text(
        json.dumps(
            {
                "schema_version": "census-model-review-gates/0.1",
                "model": "fixture/model",
                "gates": {
                    str(record["observation_id"]): {
                        "decision": "accept",
                        "candidate_payload_sha256": "0" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    stale = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )
    assert stale.records == []
    assert stale.stale_review_gates == 1


def test_a_metric_with_no_known_scale_is_left_out_and_counted(tmp_path: Path) -> None:
    """EEE 0.2.2 cannot express a continuous score without min and max, and its
    metric_config `if` matches vacuously when score_type is absent, so such a record is
    rejected either way. Unknown bounds must not be invented, so the candidate is left
    out and counted instead of producing an invalid record."""

    from proceedings_to_eee.census_review import candidate_payload_sha256
    from proceedings_to_eee.composition.eee import _metric_config
    from proceedings_to_eee.resolution.metrics import resolve_metric
    from proceedings_to_eee.validation.eee_schema import load_schema

    # chrf is identity-only: the registry gives it no unit and no bounds.
    chrf = resolve_metric(MetricSpec(raw_name="ChrF"))
    assert chrf.min_score is None and chrf.max_score is None
    config = _metric_config(_candidate(metric=chrf, value=ReportedValue(raw="0.62", numeric=0.62)))
    assert config["score_type"] == "continuous"
    assert "min_score" not in config

    paper_dir = tmp_path / PAPER_ID
    (paper_dir / "private").mkdir(parents=True)
    candidate = _candidate(metric=chrf, value=ReportedValue(raw="0.62", numeric=0.62))
    validate_candidates([candidate], {SOURCE_ID: _layout()}, min_confidence=0.8)
    (paper_dir / "observations.jsonl").write_text(
        json.dumps(candidate.model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    (paper_dir / "private" / "layout.json").write_text(
        json.dumps(_layout().model_dump(mode="json")), encoding="utf-8"
    )
    (paper_dir / "source-manifest.json").write_text(_MANIFEST, encoding="utf-8")
    # A reviewer accept is what carried these past the reference gate in production.
    record = json.loads((paper_dir / "observations.jsonl").read_text())
    (paper_dir / "model-review-gates.json").write_text(
        json.dumps(
            {
                "schema_version": "census-model-review-gates/0.1",
                "model": "fixture/model",
                "gates": {
                    str(record["observation_id"]): {
                        "decision": "accept_tuple_only",
                        "candidate_payload_sha256": candidate_payload_sha256(record),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    schema, authority = load_schema()

    tiered = recompose_paper(
        paper_dir,
        origin_policy=OriginExportPolicy.TIERED,
        schema=schema,
        schema_version=authority.version,
    )

    assert tiered.records == []
    assert tiered.skipped_unbounded_metric == 1
    # Nothing invalid was written; the exclusion is counted, not a failed validation.
    assert tiered.invalid_records == []
