"""Deterministic dataset binding: rules, split derivation, hash invariance, composition.

Binding attaches a sidecar and nothing else. The raw dataset string stays inside the
observation id and every gate hash, and a paper without an inventory composes exactly as
before.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from proceedings_to_eee.census_recompose import recompose_paper
from proceedings_to_eee.census_review import candidate_payload_sha256
from proceedings_to_eee.composition.eee import compose_eee_records
from proceedings_to_eee.domain.attribution import OriginExportPolicy
from proceedings_to_eee.domain.dataset_binding import DatasetBinding, DatasetBindingClass
from proceedings_to_eee.domain.export_provenance import legacy_export_provenance
from proceedings_to_eee.domain.observation import (
    CandidateObservation,
    EvidenceAnchor,
    MetricSpec,
    ObservationScope,
    ReportedValue,
    RoleAssignment,
)
from proceedings_to_eee.domain.status import ActorRole, ClaimType, EvidenceKind, ExportStatus
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.io import write_json
from proceedings_to_eee.public_snapshot import PublicSnapshotError, _project_composer_eee_record
from proceedings_to_eee.resolution.dataset_binding import (
    RULE_CAPTION_SINGLE_ENTRY,
    RULE_EXACT_ALIAS,
    RULE_EXACT_NAME,
    RULE_NONE_CAPTION_AMBIGUOUS,
    RULE_NONE_CAPTION_OVERRIDES_NAMED_RAW,
    RULE_NONE_EXACT_AMBIGUOUS,
    RULE_NONE_NO_RULE_MATCHED,
    RULE_UNIQUE_GENERIC_WORD,
    RULE_UNIQUE_LANGUAGE_NAME,
    RULE_UNIQUE_SPLIT_WORD,
    RULE_UNIQUE_TABLE_REFERENCE,
    bind_paper_candidates,
    normalise_name,
)
from proceedings_to_eee.resolution.dataset_inventory import entry_id, inventory_content_sha256
from proceedings_to_eee.sources.manifest import SourceManifest
from proceedings_to_eee.validation.candidates import (
    route_candidate_attribution,
    validate_candidates,
)
from proceedings_to_eee.validation.eee_schema import load_schema
from proceedings_to_eee.validation.field_provenance import quote_field_provenance

SOURCE_ID = "src_binding_fixture"
PAPER_ID = "binding-fixture"
INVENTORY_SHA256 = "f" * 64

PROSE = "PerspectiveAPI reaches 0.81 macro F1 on the HateCheck test split."

PAGE_TEXT = f"""Results

{PROSE}

Table 2: Macro F1 on the test set of HateCheck.

     System              Test      Dev
     PerspectiveAPI      0.81      0.79
     Baseline            0.62      0.60


Table 3: Scores on OLID and HASOC.

     System              OLID      HASOC
     PerspectiveAPI      0.70      0.65
"""

HATECHECK_QUOTE = "PerspectiveAPI      0.81      0.79"
OLID_QUOTE = "PerspectiveAPI      0.70      0.65"

# A two-column page whose table runs across both columns. The gutter severs it, so
# a quote from a row exists in no column panel and is located on the physical lines.
FULL_WIDTH_PAGE = """       Asset                   Measure-A  ScaleB     Index-C        Measure-D  ScaleE   Index-F
       System Cedar              0.42       0.81      0.55            0.37      0.69      0.48
       System Juniper            0.57       0.63      0.60            0.49      0.62      0.55

                     Table 2. Invented scoring grid across two fixture categories.

Two reviewers covered three categories and we               The right column discusses tradeoffs across
report aggregate scores for every system.                   invented model variants in this fixture.
All systems used the same synthetic split                   System Juniper prioritizes recall over
and the same deterministic scoring script.                  precision in the illustrative values.
Results average three invented trials and                   Combining the variants can support
require no additional calibration.                          decisions in a synthetic deployment.
"""
FULL_WIDTH_QUOTE = "System Juniper 0.57 0.63 0.60 0.49 0.62 0.55"


def _layout(text: str = PAGE_TEXT) -> PdfLayout:
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
                text=text,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                character_count=len(text),
                numeric_token_count=1,
                result_signal_score=1,
            )
        ],
    )


def _entry(name: str, kind: str = "dataset", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "entry_id": entry_id(name, kind),
        "name_raw": name,
        "aliases_in_paper": [],
        "kind": kind,
        "languages": [],
        "splits": [],
        "introduced_here": False,
        "anchors": [{"page": 1, "quote_sha256": "0" * 64, "verified": True}],
    }
    payload.update(overrides)
    return payload


HATECHECK = _entry(
    "HateCheck",
    aliases_in_paper=["HC"],
    languages=["English"],
    splits=[{"name": "test", "size_reported": None}],
)
OLID = _entry("OLID")
HASOC = _entry("HASOC")
INVENTORY = {"schema_version": "dataset-inventory/0.1", "entries": [HATECHECK, OLID, HASOC]}


def _candidate(**overrides: Any) -> CandidateObservation:
    payload: dict[str, Any] = {
        "schema_version": "candidate-observation/0.3",
        "paper_id": PAPER_ID,
        "claim_type": ClaimType.PRIMARY_RESULT,
        "extraction_confidence": 0.95,
        "roles": [RoleAssignment(role=ActorRole.EVALUATED_SYSTEM, raw_name="PerspectiveAPI")],
        "scope": ObservationScope(dataset_raw="HateCheck"),
        "metric": MetricSpec(
            raw_name="macro F1", canonical_id="macro_f1", unit="proportion", lower_is_better=False
        ),
        "value": ReportedValue(raw="0.81", numeric=0.81, unit="proportion"),
        "evidence": [
            EvidenceAnchor(
                source_id=SOURCE_ID,
                page=1,
                kind=EvidenceKind.TABLE,
                label="Table 2",
                row="PerspectiveAPI",
                column="Test",
                quote=HATECHECK_QUOTE,
            )
        ],
    }
    payload.update(overrides)
    candidate = CandidateObservation(**payload)
    return candidate.model_copy(update={"field_provenance": quote_field_provenance(candidate)})


def _bind(
    candidate: CandidateObservation,
    inventory: dict[str, Any] = INVENTORY,
    layout: PdfLayout | None = None,
) -> DatasetBinding:
    bind_paper_candidates(
        [candidate],
        inventory=inventory,
        inventory_sha256=INVENTORY_SHA256,
        layout=layout or _layout(),
    )
    assert candidate.dataset_binding is not None
    return candidate.dataset_binding


def _prose_anchor() -> EvidenceAnchor:
    """A second anchor that carries every tuple field, as run candidates usually do."""

    return EvidenceAnchor(source_id=SOURCE_ID, page=1, kind=EvidenceKind.PROSE, quote=PROSE)


def _olid_anchor(column: str = "OLID") -> EvidenceAnchor:
    return EvidenceAnchor(
        source_id=SOURCE_ID,
        page=1,
        kind=EvidenceKind.TABLE,
        label="Table 3",
        row="PerspectiveAPI",
        column=column,
        quote=OLID_QUOTE,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HateCheck", "hatecheck"),
        ("  Hate-Check,  (v2) ", "hate check v2"),
        ("OLID.", "olid"),
        (None, ""),
    ],
)
def test_normalisation_lowercases_strips_punctuation_and_collapses_spaces(
    raw: str | None, expected: str
) -> None:
    assert normalise_name(raw) == expected


@pytest.mark.parametrize(
    ("raw", "rule_id"),
    [("HateCheck", RULE_EXACT_NAME), ("hatecheck", RULE_EXACT_NAME), ("HC", RULE_EXACT_ALIAS)],
)
def test_the_exact_rule_binds_a_name_or_alias_regardless_of_case(raw: str, rule_id: str) -> None:
    binding = _bind(_candidate(scope=ObservationScope(dataset_raw=raw)))

    assert binding.binding_class is DatasetBindingClass.EXACT
    assert binding.rule_id == rule_id
    assert binding.entry_id == HATECHECK["entry_id"]
    assert binding.entry_name == "HateCheck"
    assert binding.entry_kind == "dataset"
    assert binding.languages == ["English"]
    assert binding.splits == ["test"]
    assert binding.inventory_sha256 == INVENTORY_SHA256


def test_an_alias_shared_by_two_entries_is_ambiguous_and_binds_nothing() -> None:
    inventory = {
        "entries": [
            _entry("HateCheck", aliases_in_paper=["the benchmark"]),
            _entry("OLID", aliases_in_paper=["the benchmark"]),
        ]
    }

    binding = _bind(_candidate(scope=ObservationScope(dataset_raw="The Benchmark")), inventory)

    assert binding.binding_class is DatasetBindingClass.NONE
    assert binding.rule_id == RULE_NONE_EXACT_AMBIGUOUS
    assert binding.entry_id is None


def test_the_caption_rule_binds_when_the_anchored_table_names_one_entry() -> None:
    binding = _bind(_candidate(scope=ObservationScope(dataset_raw="Table 2")))

    assert binding.binding_class is DatasetBindingClass.CAPTION
    assert binding.rule_id == RULE_CAPTION_SINGLE_ENTRY
    assert binding.entry_name == "HateCheck"


def test_the_caption_rule_reads_the_header_row_too() -> None:
    inventory = {"entries": [HATECHECK, OLID]}
    candidate = _candidate(
        scope=ObservationScope(dataset_raw="overall"), evidence=[_olid_anchor(column="F1")]
    )

    binding = _bind(candidate, inventory)

    # Table 3's caption names OLID and HASOC, but only OLID is an inventory entry here,
    # and the header row names it as well: exactly one entry.
    assert binding.binding_class is DatasetBindingClass.CAPTION
    assert binding.entry_name == "OLID"


@pytest.mark.parametrize(
    ("raw", "aliases", "binding_class", "rule_id"),
    [
        ("our test set", [], DatasetBindingClass.CAPTION, RULE_CAPTION_SINGLE_ENTRY),
        ("the test data", [], DatasetBindingClass.CAPTION, RULE_CAPTION_SINGLE_ENTRY),
        ("English", [], DatasetBindingClass.CAPTION, RULE_CAPTION_SINGLE_ENTRY),
        (
            "functional suite",
            ["HateCheck functional suite"],
            DatasetBindingClass.CAPTION,
            RULE_CAPTION_SINGLE_ENTRY,
        ),
        (
            "HateCheck and HateX",
            [],
            DatasetBindingClass.NONE,
            RULE_NONE_CAPTION_OVERRIDES_NAMED_RAW,
        ),
        (
            "Fox suite",
            ["functional suite"],
            DatasetBindingClass.NONE,
            RULE_NONE_CAPTION_OVERRIDES_NAMED_RAW,
        ),
        ("Subtask A", [], DatasetBindingClass.NONE, RULE_NONE_CAPTION_OVERRIDES_NAMED_RAW),
    ],
    ids=[
        "split_phrase_after_determiner",
        "generic_phrase",
        "language",
        "words_covered_by_alias",
        "second_source_not_in_inventory",
        "word_outside_entry",
        "task_label",
    ],
)
def test_the_caption_rule_never_narrows_a_raw_string_that_names_more(
    raw: str, aliases: list[str], binding_class: DatasetBindingClass, rule_id: str
) -> None:
    # Table 2's caption names HateCheck only. The rule may replace a raw string that
    # names nothing beyond that entry; one that names a second source (an entry the
    # inventory lost, or a combined evaluation) is kept as the paper printed it.
    inventory = {"entries": [_entry("HateCheck", aliases_in_paper=aliases)]}
    candidate = _candidate(scope=ObservationScope(dataset_raw=raw))

    binding = _bind(candidate, inventory)

    assert binding.binding_class is binding_class
    assert binding.rule_id == rule_id
    assert binding.entry_name == (
        "HateCheck" if binding_class is DatasetBindingClass.CAPTION else None
    )


def test_the_caption_rule_reads_a_full_width_table_on_a_two_column_page() -> None:
    # The row quote exists in no column panel, so the locator falls back to the
    # physical page lines and a single-panel index whose ids collide with the panelled
    # index's ids. The caption must come from the full-width table, not from the
    # panelled first-column fragment that happens to carry the same id.
    inventory = {"entries": [_entry("Invented scoring grid"), _entry("OLID")]}
    candidate = _candidate(
        scope=ObservationScope(dataset_raw="Table 2"),
        evidence=[
            EvidenceAnchor(
                source_id=SOURCE_ID,
                page=1,
                kind=EvidenceKind.TABLE,
                label="Table 2",
                row="System Juniper",
                column="Measure-A",
                quote=FULL_WIDTH_QUOTE,
            )
        ],
    )

    binding = _bind(candidate, inventory, layout=_layout(FULL_WIDTH_PAGE))

    assert binding.binding_class is DatasetBindingClass.CAPTION
    assert binding.rule_id == RULE_CAPTION_SINGLE_ENTRY
    assert binding.entry_name == "Invented scoring grid"


def test_a_caption_naming_two_entries_is_ambiguous_and_binds_nothing() -> None:
    binding = _bind(
        _candidate(scope=ObservationScope(dataset_raw="overall"), evidence=[_olid_anchor()])
    )

    assert binding.binding_class is DatasetBindingClass.NONE
    assert binding.rule_id == RULE_NONE_CAPTION_AMBIGUOUS


def test_a_prose_anchor_has_no_table_and_falls_through() -> None:
    prose = EvidenceAnchor(
        source_id=SOURCE_ID, page=1, kind=EvidenceKind.PROSE, quote="Macro F1 on the test set"
    )
    binding = _bind(_candidate(scope=ObservationScope(dataset_raw="Table 2"), evidence=[prose]))

    assert binding.binding_class is DatasetBindingClass.NONE
    assert binding.rule_id == RULE_NONE_NO_RULE_MATCHED


@pytest.mark.parametrize(
    ("raw", "rule_id"),
    [
        ("test set", RULE_UNIQUE_SPLIT_WORD),
        ("Dev", RULE_UNIQUE_SPLIT_WORD),
        ("English", RULE_UNIQUE_LANGUAGE_NAME),
        ("overall", RULE_UNIQUE_GENERIC_WORD),
        ("Table 5", RULE_UNIQUE_TABLE_REFERENCE),
    ],
)
def test_the_unique_rule_binds_a_non_identifying_string_to_the_only_entry(
    raw: str, rule_id: str
) -> None:
    inventory = {"entries": [OLID, _entry("GLUE", kind="benchmark")]}
    # A prose anchor so that the caption rule cannot fire first.
    prose = EvidenceAnchor(source_id=SOURCE_ID, page=1, kind=EvidenceKind.PROSE, quote="Results")

    candidate = _candidate(scope=ObservationScope(dataset_raw=raw), evidence=[prose])

    binding = _bind(candidate, inventory)

    assert binding.binding_class is DatasetBindingClass.UNIQUE
    assert binding.rule_id == rule_id
    assert binding.entry_name == "OLID"


@pytest.mark.parametrize(
    ("raw", "entries"),
    [
        ("SomeOtherSet", [OLID]),
        ("test set", [OLID, HASOC]),
        ("test set", [_entry("GLUE", kind="benchmark")]),
    ],
    ids=["identifying_string", "two_dataset_entries", "benchmark_only"],
)
def test_the_unique_rule_needs_one_dataset_entry_and_a_non_identifying_string(
    raw: str, entries: list[dict[str, Any]]
) -> None:
    prose = EvidenceAnchor(source_id=SOURCE_ID, page=1, kind=EvidenceKind.PROSE, quote="Results")

    binding = _bind(
        _candidate(scope=ObservationScope(dataset_raw=raw), evidence=[prose]),
        {"entries": entries},
    )

    assert binding.binding_class is DatasetBindingClass.NONE
    assert binding.rule_id == RULE_NONE_NO_RULE_MATCHED


def test_a_split_is_read_from_the_column_header_first() -> None:
    binding = _bind(_candidate())

    assert binding.split == "test"
    assert binding.split_source is not None
    assert binding.split_source.value == "column_header"
    assert binding.split_page == 1


def test_a_split_is_read_from_the_caption_when_no_header_names_one() -> None:
    candidate = _candidate()
    candidate.evidence[0].column = "F1"

    binding = _bind(candidate)

    assert binding.split == "test"
    assert binding.split_source is not None
    assert binding.split_source.value == "caption"


@pytest.mark.parametrize("column", ["Testing time", "F1", None])
def test_no_split_is_invented_from_a_header_that_names_none(column: str | None) -> None:
    candidate = _candidate(
        scope=ObservationScope(dataset_raw="OLID"), evidence=[_olid_anchor(column=column or "")]
    )
    if column is None:
        candidate.evidence[0].column = None

    binding = _bind(candidate)

    assert binding.split is None
    assert binding.split_source is None


def test_a_stated_split_is_never_overwritten_or_duplicated() -> None:
    binding = _bind(_candidate(scope=ObservationScope(dataset_raw="HateCheck", split="dev")))

    assert binding.split is None
    assert binding.split_source is None


def test_binding_leaves_the_observation_id_the_gate_payload_and_the_scope_untouched() -> None:
    candidate = _candidate()
    validate_candidates([candidate], {SOURCE_ID: _layout()}, min_confidence=0.8)
    before_id = candidate.observation_id
    before_payload = candidate_payload_sha256(candidate.model_dump(mode="json"))
    before_fields = candidate.field_value_sha256s()
    before_scope = candidate.scope.model_dump(mode="json") if candidate.scope else None

    binding = _bind(candidate)

    assert binding.binding_class is DatasetBindingClass.EXACT
    assert binding.split == "test"
    assert candidate.observation_id == before_id == candidate.stable_id()
    assert candidate_payload_sha256(candidate.model_dump(mode="json")) == before_payload
    assert candidate.field_value_sha256s() == before_fields
    assert candidate.scope is not None
    assert candidate.scope.model_dump(mode="json") == before_scope
    assert candidate.scope.split is None
    # A round trip through JSON keeps the sidecar and the id.
    again = CandidateObservation.model_validate(candidate.model_dump(mode="json"))
    assert again.observation_id == before_id
    assert again.dataset_binding == binding


def _compose(candidate: CandidateObservation, manifest: SourceManifest) -> dict[str, Any]:
    """Compose one candidate under the tiered policy, past the gates composition itself
    does not re-run, so these tests see only what binding changes in a record."""

    candidate.export_status = ExportStatus.ELIGIBLE
    route_candidate_attribution([candidate], {}, OriginExportPolicy.TIERED)
    provenance = legacy_export_provenance([candidate], reason="binding fixture").model_copy(
        update={"origin_policy": OriginExportPolicy.TIERED}
    )
    [record] = compose_eee_records(
        manifest=manifest,
        candidates=[candidate],
        schema_version="0.3.0",
        provenance=provenance,
    )
    return record


@pytest.fixture
def binding_manifest(manifest: SourceManifest) -> SourceManifest:
    source = manifest.sources[0].model_copy(update={"source_id": SOURCE_ID, "paper_id": PAPER_ID})
    return manifest.model_copy(update={"paper_id": PAPER_ID, "sources": [source]})


def test_a_bound_record_names_the_entry_and_keeps_the_raw_string(
    binding_manifest: SourceManifest,
) -> None:
    candidate = _candidate(scope=ObservationScope(dataset_raw="hatecheck"))
    _bind(candidate)

    record = _compose(candidate, binding_manifest)

    result = record["evaluation_results"][0]
    assert result["source_data"]["dataset_name"] == "HateCheck"
    assert result["evaluation_name"] == "HateCheck / macro F1"
    details = result["source_data"]["additional_details"]
    assert details["dataset_raw"] == "hatecheck"
    assert details["dataset_binding_class"] == "exact"
    assert details["dataset_binding_rule"] == RULE_EXACT_NAME
    assert details["split"] == "test"
    assert details["split_source"] == "column_header"
    assert details["language"] == "English"
    assert record["source_metadata"]["additional_details"]["dataset_inventory_sha256"] == (
        INVENTORY_SHA256
    )
    # The public projection accepts every new key as-is.
    assert _project_composer_eee_record(record, paper_id=PAPER_ID, context="eee") == record


def test_an_unbound_record_keeps_the_raw_name_but_states_the_class(
    binding_manifest: SourceManifest,
) -> None:
    candidate = _candidate(
        scope=ObservationScope(dataset_raw="overall", language="Portuguese"),
        evidence=[_olid_anchor()],
    )
    _bind(candidate)

    record = _compose(candidate, binding_manifest)

    result = record["evaluation_results"][0]
    assert result["source_data"]["dataset_name"] == "overall"
    assert result["evaluation_name"] == "overall / macro F1"
    details = result["source_data"]["additional_details"]
    assert details["dataset_binding_class"] == "none"
    assert details["language"] == "Portuguese"
    assert "split" not in details


def test_a_candidate_with_its_own_language_and_split_keeps_them(
    binding_manifest: SourceManifest,
) -> None:
    candidate = _candidate(
        scope=ObservationScope(dataset_raw="HateCheck", split="dev", language="Portuguese")
    )
    _bind(candidate)

    details = _compose(candidate, binding_manifest)["evaluation_results"][0]["source_data"][
        "additional_details"
    ]

    assert details["split"] == "dev"
    assert "split_source" not in details
    assert details["language"] == "Portuguese"


def test_a_record_without_a_binding_carries_none_of_the_new_keys(
    binding_manifest: SourceManifest,
) -> None:
    record = _compose(_candidate(), binding_manifest)

    details = record["evaluation_results"][0]["source_data"]["additional_details"]
    assert "dataset_raw" not in details
    assert "dataset_binding_class" not in details
    assert "dataset_inventory_sha256" not in record["source_metadata"]["additional_details"]


@pytest.mark.parametrize(
    ("key", "value"),
    [("dataset_binding_class", "guessed"), ("split_source", "model")],
)
def test_the_public_projection_rejects_labels_outside_the_vocabulary(
    binding_manifest: SourceManifest, key: str, value: str
) -> None:
    candidate = _candidate()
    _bind(candidate)
    record = _compose(candidate, binding_manifest)
    record["evaluation_results"][0]["source_data"]["additional_details"][key] = value

    with pytest.raises(PublicSnapshotError, match="unsupported"):
        _project_composer_eee_record(record, paper_id=PAPER_ID, context="eee")


def _write_paper(tmp_path: Path, candidate: CandidateObservation) -> Path:
    paper_dir = tmp_path / PAPER_ID
    (paper_dir / "private").mkdir(parents=True, exist_ok=True)
    layout = _layout()
    validate_candidates([candidate], {SOURCE_ID: layout}, min_confidence=0.8)
    (paper_dir / "observations.jsonl").write_text(
        json.dumps(candidate.model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    (paper_dir / "private" / "layout.json").write_text(
        json.dumps(layout.model_dump(mode="json")), encoding="utf-8"
    )
    (paper_dir / "source-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "source-manifest/0.1",
                "paper_id": PAPER_ID,
                "title": "Binding fixture",
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
        ),
        encoding="utf-8",
    )
    return paper_dir


def test_recompose_binds_only_when_an_inventory_exists_and_names_it(tmp_path: Path) -> None:
    candidate = _candidate(evidence=[_candidate().evidence[0], _prose_anchor()])
    paper_dir = _write_paper(tmp_path, candidate)
    run_bytes = (paper_dir / "observations.jsonl").read_bytes()
    schema, authority = load_schema()

    def recompose():
        return recompose_paper(
            paper_dir,
            origin_policy=OriginExportPolicy.TIERED,
            schema=schema,
            schema_version=authority.version,
        )

    before = recompose()
    assert len(before.records) == 1
    assert before.dataset_inventory_present is False
    assert before.dataset_binding_counts == {}
    unbound = before.records[0]["evaluation_results"][0]
    assert unbound["source_data"]["dataset_name"] == "HateCheck"
    assert "dataset_raw" not in unbound["source_data"]["additional_details"]
    assert "split" not in unbound["source_data"]["additional_details"]

    write_json(paper_dir / "dataset-inventory.json", INVENTORY)
    after = recompose()

    assert after.dataset_inventory_present is True
    assert after.dataset_inventory_entries == 3
    assert after.dataset_binding_counts == {"exact": 1}
    assert after.dataset_binding_rules == {RULE_EXACT_NAME: 1}
    assert after.split_derived_counts == {"column_header": 1}
    bound = after.records[0]["evaluation_results"][0]
    assert bound["evaluation_result_id"] == unbound["evaluation_result_id"]
    assert bound["source_data"]["dataset_name"] == "HateCheck"
    assert bound["source_data"]["additional_details"]["dataset_raw"] == "HateCheck"
    assert bound["source_data"]["additional_details"]["split"] == "test"
    assert bound["source_data"]["additional_details"]["split_source"] == "column_header"
    # Everything but the binding details is identical to the unbound record, except the
    # two provenance hashes, which bind the full candidate dump and so see the sidecar.
    rebound = {"export_candidate_sha256", "export_composition_sha256"}

    def comparable(result: dict[str, Any]) -> dict[str, Any]:
        details = {
            key: value
            for key, value in result["score_details"]["details"].items()
            if key not in rebound
        }
        return {
            **{key: value for key, value in result.items() if key != "source_data"},
            "score_details": {**result["score_details"], "details": details},
        }

    assert comparable(bound) == comparable(unbound)
    # The record names the inventory by content, so a gate rewritten with other
    # telemetry (a rerun's latency and cost) would still name the same inventory.
    assert after.records[0]["source_metadata"]["additional_details"][
        "dataset_inventory_sha256"
    ] == inventory_content_sha256(INVENTORY)
    write_json(
        paper_dir / "dataset-inventory.json",
        {**INVENTORY, "telemetry": {"latency_seconds": 3.0, "cost_usd": 0.01}},
    )
    assert recompose().records[0]["source_metadata"]["additional_details"][
        "dataset_inventory_sha256"
    ] == inventory_content_sha256(INVENTORY)
    # The run's candidates are never rewritten; the sidecar lives only in the replay.
    assert (paper_dir / "observations.jsonl").read_bytes() == run_bytes


def test_an_unbound_candidate_keeps_the_export_hash_it_had_before_the_sidecar_existed() -> None:
    from proceedings_to_eee.domain.export_provenance import candidate_export_sha256
    from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes

    candidate = _candidate()
    payload = candidate.model_dump(mode="json")
    assert payload["dataset_binding"] is None
    del payload["dataset_binding"]

    unbound = candidate_export_sha256(candidate)

    assert unbound == sha256_bytes(canonical_json_bytes(payload))
    _bind(candidate)
    assert candidate_export_sha256(candidate) != unbound
