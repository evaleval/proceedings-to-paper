from __future__ import annotations

from pathlib import Path

import pytest

from proceedings_to_eee.composition.eee import compose_eee_records
from proceedings_to_eee.domain.attribution import AttributionState, AttributionVerdict
from proceedings_to_eee.domain.export_provenance import (
    CandidateExportAuthorization,
    ExportCompositionProvenance,
    ExportProvenanceMode,
    candidate_export_sha256,
    legacy_export_provenance,
    tuple_audited_review_export_provenance,
    tuple_gated_export_provenance,
    tuple_unverified_review_provenance,
)
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ExportStatus
from proceedings_to_eee.sources.manifest import SourceManifest
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "eee-0.2.2" / "eval.schema.json"
SCHEMA_SHA = "088fed8029d42fb3a607aa67e1a05c39e425241b5cd90803705b37562f402f2a"


def _legacy(candidates: list[CandidateObservation]):
    eligible = [
        candidate
        for candidate in candidates
        if candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}
        and candidate.attribution is not None
        and candidate.attribution.allows_canonical_export
    ]
    return legacy_export_provenance(eligible, reason="unit_test_manual_composition")


def test_eligible_candidate_composes_to_valid_eee(
    manifest: SourceManifest, eligible_candidate: CandidateObservation
) -> None:
    schema, authority = load_schema(SCHEMA, SCHEMA_SHA)
    before = eligible_candidate.model_dump(mode="json")
    candidates = [eligible_candidate]
    records = compose_eee_records(
        manifest=manifest,
        candidates=candidates,
        schema_version=authority.version,
        provenance=_legacy(candidates),
    )
    assert len(records) == 1
    result = records[0]["evaluation_results"][0]
    assert result["score_details"]["score"] == 74.6
    provenance = result["score_details"]["details"]
    assert provenance["paper_id"] == "synthetic-audit-study"
    assert provenance["evidence_anchor_count"] == "1"
    assert provenance["evidence_1_source_id"] == "src_paper"
    assert provenance["evidence_1_source_role"] == "paper"
    assert provenance["evidence_1_source_sha256"] == "a" * 64
    assert provenance["evidence_1_page"] == "7"
    assert provenance["evidence_1_kind"] == "table"
    assert provenance["evidence_1_label"] == "Table 2"
    assert provenance["evidence_1_row"] == "Atlas Moderation API · Synthetic Speech Set"
    assert provenance["evidence_1_column"] == "AUC"
    assert provenance["evidence_1_quote_sha256"] == eligible_candidate.evidence[0].quote_sha256
    assert provenance["export_provenance_mode"] == "legacy_manual"
    assert provenance["legacy_provenance_reason"] == "unit_test_manual_composition"
    assert eligible_candidate.evidence[0].quote not in provenance.values()
    assert result["metric_config"]["metric_unit"] == "percent"
    assert result["metric_config"]["max_score"] == 100
    assert validate_eee_record(records[0], schema) == []
    assert eligible_candidate.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "state",
    [
        None,
        AttributionState.EXTERNALLY_SOURCED,
        AttributionState.UNRESOLVED,
        AttributionState.NO_SIGNAL,
    ],
)
@pytest.mark.parametrize("status", [ExportStatus.ELIGIBLE, ExportStatus.EXPORTED])
def test_non_paper_produced_candidate_never_composes(
    manifest: SourceManifest,
    eligible_candidate: CandidateObservation,
    state: AttributionState | None,
    status: ExportStatus,
) -> None:
    eligible_candidate.export_status = status
    eligible_candidate.attribution = (
        AttributionVerdict(state=state, rule_id="test_origin") if state is not None else None
    )

    records = compose_eee_records(
        manifest=manifest,
        candidates=[eligible_candidate],
        schema_version="0.2.2",
        provenance=_legacy([eligible_candidate]),
    )

    assert records == []
    assert eligible_candidate.export_status is status


def test_paper_produced_origin_does_not_override_other_export_gates(
    manifest: SourceManifest, eligible_candidate: CandidateObservation
) -> None:
    eligible_candidate.export_status = ExportStatus.NEEDS_REVIEW

    assert eligible_candidate.attribution is not None
    assert eligible_candidate.attribution.state is AttributionState.PAPER_PRODUCED
    assert (
        compose_eee_records(
            manifest=manifest,
            candidates=[eligible_candidate],
            schema_version="0.2.2",
            provenance=_legacy([eligible_candidate]),
        )
        == []
    )


def test_schema_version_equality_is_checked(
    manifest: SourceManifest, eligible_candidate: CandidateObservation
) -> None:
    schema, _ = load_schema(SCHEMA, SCHEMA_SHA)
    before = eligible_candidate.model_dump(mode="json")
    record = compose_eee_records(
        manifest=manifest,
        candidates=[eligible_candidate],
        schema_version="wrong",
        provenance=_legacy([eligible_candidate]),
    )[0]
    issues = validate_eee_record(record, schema)
    assert any(issue.path == "schema_version" for issue in issues)
    assert eligible_candidate.model_dump(mode="json") == before
    assert eligible_candidate.export_status is ExportStatus.ELIGIBLE


def test_same_system_with_optional_versions_composes_deterministically(
    manifest: SourceManifest, eligible_candidate: CandidateObservation
) -> None:
    schema, authority = load_schema(SCHEMA, SCHEMA_SHA)
    unversioned = eligible_candidate.model_copy(deep=True)
    versioned = eligible_candidate.model_copy(deep=True)
    versioned.roles[0].version = "v2"
    versioned.observation_id = versioned.stable_id()

    records = compose_eee_records(
        manifest=manifest,
        candidates=[versioned, unversioned],
        schema_version=authority.version,
        provenance=_legacy([versioned, unversioned]),
    )
    reversed_candidates = [unversioned.model_copy(deep=True), versioned.model_copy(deep=True)]
    reversed_records = compose_eee_records(
        manifest=manifest,
        candidates=reversed_candidates,
        schema_version=authority.version,
        provenance=_legacy(reversed_candidates),
    )

    assert [record["evaluation_id"] for record in records] == [
        record["evaluation_id"] for record in reversed_records
    ]
    assert len({record["evaluation_id"] for record in records}) == 2
    assert "reported_version" not in records[0]["model_info"]["additional_details"]
    assert records[1]["model_info"]["additional_details"]["reported_version"] == "v2"
    assert all(validate_eee_record(record, schema) == [] for record in records)


def test_composer_requires_explicit_typed_provenance(
    manifest: SourceManifest, eligible_candidate: CandidateObservation
) -> None:
    with pytest.raises(TypeError):
        compose_eee_records(  # type: ignore[call-arg]
            manifest=manifest,
            candidates=[eligible_candidate],
            schema_version="0.2.2",
        )


def test_stale_or_spliced_candidate_authorization_fails_closed(
    manifest: SourceManifest, eligible_candidate: CandidateObservation
) -> None:
    provenance = legacy_export_provenance([eligible_candidate], reason="explicit_manual_fixture")
    eligible_candidate.metric.raw_name = "spliced metric"

    with pytest.raises(ValueError, match="stale or spliced"):
        compose_eee_records(
            manifest=manifest,
            candidates=[eligible_candidate],
            schema_version="0.2.2",
            provenance=provenance,
        )


def test_tuple_gated_composition_requires_and_exposes_exact_passing_gate(
    manifest: SourceManifest, eligible_candidate: CandidateObservation
) -> None:
    observation_id = str(eligible_candidate.observation_id)
    provenance = tuple_gated_export_provenance(
        [eligible_candidate],
        tuple_sidecar_sha256="a" * 64,
        tuple_gates={observation_id: "b" * 64},
        verifier_sidecar_sha256="c" * 64,
        verifier_gates={observation_id: "d" * 64},
    )

    records = compose_eee_records(
        manifest=manifest,
        candidates=[eligible_candidate],
        schema_version="0.2.2",
        provenance=provenance,
    )
    details = records[0]["evaluation_results"][0]["score_details"]["details"]
    assert details["export_provenance_mode"] == "tuple_gated_production"
    assert details["tuple_gate_sha256"] == "b" * 64
    assert details["tuple_sidecar_sha256"] == "a" * 64
    assert details["verifier_gate_required"] == "true"
    assert details["verifier_gate_sha256"] == "d" * 64
    assert details["verifier_sidecar_sha256"] == "c" * 64

    with pytest.raises(ValueError, match="explicit passing gate"):
        ExportCompositionProvenance(
            mode=ExportProvenanceMode.TUPLE_GATED_PRODUCTION,
            tuple_sidecar_sha256="a" * 64,
            verifier_gate_required=False,
            authorizations=[
                CandidateExportAuthorization(
                    observation_id=observation_id,
                    candidate_sha256=candidate_export_sha256(eligible_candidate),
                    tuple_gate_sha256="b" * 64,
                    tuple_gate_passed=False,
                )
            ],
        )


def test_human_reviewed_mode_audits_failed_model_gates_without_treating_them_as_passes(
    manifest: SourceManifest, eligible_candidate: CandidateObservation
) -> None:
    observation_id = str(eligible_candidate.observation_id)
    provenance = tuple_audited_review_export_provenance(
        [eligible_candidate],
        tuple_sidecar_sha256="a" * 64,
        tuple_gates={observation_id: ("b" * 64, False)},
        verifier_sidecar_sha256="c" * 64,
        verifier_gates={observation_id: ("d" * 64, False)},
        review_manifest_sha256="e" * 64,
    )

    records = compose_eee_records(
        manifest=manifest,
        candidates=[eligible_candidate],
        schema_version="0.2.2",
        provenance=provenance,
    )
    details = records[0]["evaluation_results"][0]["score_details"]["details"]
    assert details["export_provenance_mode"] == "tuple_audited_human_reviewed"
    assert details["source_verifier_gate_passed"] == "false"
    assert details["verifier_gate_required"] == "false"
    assert details["review_manifest_sha256"] == "e" * 64


def test_tuple_only_automatic_mode_is_explicitly_unverified_and_cannot_authorize_eee(
    manifest: SourceManifest, eligible_candidate: CandidateObservation
) -> None:
    observation_id = str(eligible_candidate.observation_id)
    with pytest.raises(ValueError, match="requires verifier sidecar"):
        tuple_gated_export_provenance(
            [eligible_candidate],
            tuple_sidecar_sha256="a" * 64,
            tuple_gates={observation_id: "b" * 64},
        )

    provenance = tuple_unverified_review_provenance(tuple_sidecar_sha256="a" * 64)
    with pytest.raises(ValueError, match="does not exactly cover"):
        compose_eee_records(
            manifest=manifest,
            candidates=[eligible_candidate],
            schema_version="0.2.2",
            provenance=provenance,
        )
