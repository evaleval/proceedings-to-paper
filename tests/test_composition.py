from __future__ import annotations

from pathlib import Path

import pytest

from proceedings_to_eee.composition.eee import compose_eee_records
from proceedings_to_eee.domain.attribution import AttributionState, AttributionVerdict
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ExportStatus
from proceedings_to_eee.sources.manifest import SourceManifest
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "eee-0.2.2" / "eval.schema.json"
SCHEMA_SHA = "088fed8029d42fb3a607aa67e1a05c39e425241b5cd90803705b37562f402f2a"


def test_pinned_schema_hash_and_version() -> None:
    _, authority = load_schema(SCHEMA, SCHEMA_SHA)
    assert authority.version == "0.2.2"


def test_eligible_candidate_composes_to_valid_eee(
    manifest: SourceManifest, eligible_candidate: CandidateObservation
) -> None:
    schema, authority = load_schema(SCHEMA, SCHEMA_SHA)
    records = compose_eee_records(
        manifest=manifest, candidates=[eligible_candidate], schema_version=authority.version
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
    assert eligible_candidate.evidence[0].quote not in provenance.values()
    assert result["metric_config"]["metric_unit"] == "percent"
    assert result["metric_config"]["max_score"] == 100
    assert validate_eee_record(records[0], schema) == []


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
        )
        == []
    )


def test_schema_version_equality_is_checked(
    manifest: SourceManifest, eligible_candidate: CandidateObservation
) -> None:
    schema, _ = load_schema(SCHEMA, SCHEMA_SHA)
    record = compose_eee_records(
        manifest=manifest, candidates=[eligible_candidate], schema_version="wrong"
    )[0]
    issues = validate_eee_record(record, schema)
    assert any(issue.path == "schema_version" for issue in issues)


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
    )
    reversed_records = compose_eee_records(
        manifest=manifest,
        candidates=[unversioned.model_copy(deep=True), versioned.model_copy(deep=True)],
        schema_version=authority.version,
    )

    assert [record["evaluation_id"] for record in records] == [
        record["evaluation_id"] for record in reversed_records
    ]
    assert len({record["evaluation_id"] for record in records}) == 2
    assert "reported_version" not in records[0]["model_info"]["additional_details"]
    assert records[1]["model_info"]["additional_details"]["reported_version"] == "v2"
    assert all(validate_eee_record(record, schema) == [] for record in records)
