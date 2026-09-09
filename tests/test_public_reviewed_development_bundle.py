from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from test_public_development_preview import (
    _build as _build_preview,
)
from test_public_development_preview import (
    _fixture as _preview_fixture,
)
from test_public_development_preview import (
    _ZeroCandidateFiveStageRowClient,
)
from test_public_development_summary import _complete_current_review, _FiveStageRowClient
from typer.testing import CliRunner

import proceedings_to_eee.reporting.public_reviewed_development_bundle as bundle_module
from proceedings_to_eee.cli import app
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from proceedings_to_eee.providers.openrouter import StructuredResponse
from proceedings_to_eee.reporting.public_development_preview import (
    PublicDevelopmentPreviewError,
    assert_no_private_evidence_text,
    collect_private_evidence_texts,
)
from proceedings_to_eee.reporting.public_reviewed_development_bundle import (
    PUBLIC_REVIEWED_CANONICAL_RESULTS_SCHEMA_VERSION,
    PublicReviewedDevelopmentBundleError,
    build_public_reviewed_development_bundle,
    verify_public_reviewed_development_bundle,
)
from proceedings_to_eee.reviewed_export.models import DerivedRunManifest
from proceedings_to_eee.reviewed_export.workflow import (
    REVIEW_DECISIONS_NAME,
    compose_reviewed_eee,
    prepare_export_review,
    validate_export_review,
)
from proceedings_to_eee.run_seal import RUN_SEAL_NAME, seal_run_tree, verify_run_seal

RUNNER = CliRunner()
PRIVATE_EVIDENCE_QUOTE = "Private source evidence phrase describing the fixture operationalization."
PRIVATE_REVIEW_EXCERPT = "We evaluate System A on Dataset A using AUC."


class _EvidenceFieldFiveStageClient(_FiveStageRowClient):
    def __init__(self, evidence_text: str = PRIVATE_EVIDENCE_QUOTE) -> None:
        super().__init__()
        self.evidence_text = evidence_text

    def structured_chat(self, **kwargs: object) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] != "paper_table_row_dispositions":
            return response
        payload = json.loads(json.dumps(response.payload))
        for disposition in payload["dispositions"]:
            for observation in disposition["observations"]:
                observation["operationalization"] = self.evidence_text
        return replace(
            response,
            payload=payload,
            call=response.call.model_copy(
                update={
                    "response_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
                    ).hexdigest()
                }
            ),
        )


def _augment_sealed_private_evidence(preview_inputs: object, tmp_path: Path) -> object:
    source_root = preview_inputs.sealed_root
    raw_root = tmp_path / "augmented-unsealed-run"
    shutil.copytree(source_root, raw_root)
    (raw_root / RUN_SEAL_NAME).unlink()
    write_json(
        raw_root / "private" / "regression-evidence.json",
        {"quote": PRIVATE_EVIDENCE_QUOTE},
    )
    sealed_root = tmp_path / "augmented-sealed-run"
    seal_run_tree(raw_root, sealed_root)
    return replace(preview_inputs, sealed_root=sealed_root)


def _reviewed_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    empty: bool = False,
    private_evidence_reproduction: bool = False,
    review_evidence_reproduction: bool = False,
    review_mode: str = "complete",
) -> tuple[Path, Path, Path, Path]:
    assert not (private_evidence_reproduction and review_evidence_reproduction)
    preview_inputs = _preview_fixture(
        monkeypatch,
        tmp_path,
        client=(
            _ZeroCandidateFiveStageRowClient()
            if empty
            else _EvidenceFieldFiveStageClient(
                PRIVATE_EVIDENCE_QUOTE if private_evidence_reproduction else PRIVATE_REVIEW_EXCERPT
            )
            if private_evidence_reproduction or review_evidence_reproduction
            else None
        ),
    )
    if private_evidence_reproduction:
        preview_inputs = _augment_sealed_private_evidence(preview_inputs, tmp_path)
    preview_root = _build_preview(preview_inputs)
    review_root = tmp_path / "private-review"
    prepare_export_review(run_root=preview_inputs.sealed_root, output_root=review_root)
    if empty or review_mode == "pending":
        validate_export_review(review_root)
    elif review_mode == "partial":
        pending = [
            json.loads(line)
            for line in (review_root / REVIEW_DECISIONS_NAME).read_text().splitlines()
        ]
        completed_root = tmp_path / "completed-review-shadow"
        shutil.copytree(review_root, completed_root)
        _complete_current_review(completed_root)
        completed = [
            json.loads(line)
            for line in (completed_root / REVIEW_DECISIONS_NAME).read_text().splitlines()
        ]
        assert len(completed) >= 2
        write_jsonl(review_root / REVIEW_DECISIONS_NAME, [completed[0], *pending[1:]])
        validate_export_review(review_root)
    else:
        assert review_mode == "complete"
        _complete_current_review(review_root)
    derived_root = tmp_path / "reviewed-derived"
    compose_reviewed_eee(
        run_root=preview_inputs.sealed_root,
        decisions_path=review_root / REVIEW_DECISIONS_NAME,
        output_root=derived_root,
    )
    return preview_inputs.sealed_root, preview_root, derived_root, review_root


def _build_reviewed(
    inputs: tuple[Path, Path, Path, Path],
    output_root: Path,
    *,
    bundle_id: str = "fixture-reviewed",
) -> Path:
    run_root, preview_root, derived_root, review_root = inputs
    return build_public_reviewed_development_bundle(
        bundle_id=bundle_id,
        run_root=run_root,
        preview_root=preview_root,
        reviewed_derived_root=derived_root,
        review_root=review_root,
        output_root=output_root,
    )


@pytest.fixture
def reviewed_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    return _build_reviewed(_reviewed_inputs(monkeypatch, tmp_path), tmp_path / "public")


def _refresh_outer_controls(root: Path) -> None:
    manifest = read_json(root / "publication-manifest.json")
    payloads = manifest["files"]
    from proceedings_to_eee.reporting.public_reviewed_development_bundle import (
        _payload_tree,
        _verification,
    )

    attestation = manifest["private_evidence_nonreproduction"]
    attestation["attested_payload_tree_sha256"] = _payload_tree(payloads)
    attestation["attested_payload_file_count"] = len(payloads)
    (root / "publication-manifest.json").write_bytes(canonical_json_bytes(manifest))
    (root / "verification.json").write_bytes(
        canonical_json_bytes(_verification(manifest=manifest, payload_files=payloads))
    )
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _rehash_payload(root: Path, path: str) -> None:
    manifest = read_json(root / "publication-manifest.json")
    _update_outer_payload_reference(manifest, root, path)
    if path == "reviewed/canonical-results.json":
        manifest["outputs"]["canonical_results"]["sha256"] = sha256_file(root / path)
    elif path == "reviewed/canonical-results.html":
        manifest["outputs"]["static_html"]["sha256"] = sha256_file(root / path)
    elif path == "reviewed/public-summary.json":
        manifest["outputs"]["public_summary"]["sha256"] = sha256_file(root / path)
    manifest["files"] = sorted(manifest["files"], key=lambda item: item["path"])
    (root / "publication-manifest.json").write_bytes(canonical_json_bytes(manifest))
    _refresh_outer_controls(root)


def _update_outer_payload_reference(manifest: dict[str, object], root: Path, path: str) -> None:
    for artifact in manifest["files"]:
        if artifact["path"] == path:
            artifact["sha256"] = sha256_file(root / path)
            artifact["size_bytes"] = (root / path).stat().st_size
            return
    raise AssertionError(path)


def _fully_rehash_published_eee(root: Path, content: bytes) -> Path:
    manifest = read_json(root / "publication-manifest.json")
    [eee_entry] = manifest["reviewed_derived"]["eee_files"]
    bundle_path = eee_entry["bundle_path"]
    eee_path = root / bundle_path
    eee_path.write_bytes(content)
    eee_entry["sha256"] = sha256_file(eee_path)
    eee_entry["size_bytes"] = eee_path.stat().st_size
    _update_outer_payload_reference(manifest, root, bundle_path)

    derived_path = root / "reviewed/derived-manifest.json"
    derived_payload = read_json(derived_path)
    for artifact in derived_payload["payload_files"]:
        if artifact["path"] == eee_entry["source_path"]:
            artifact["sha256"] = eee_entry["sha256"]
            artifact["size_bytes"] = eee_entry["size_bytes"]
            break
    else:
        raise AssertionError(eee_entry["source_path"])
    provisional = DerivedRunManifest.model_validate(derived_payload)
    canonical_inventory = [
        artifact.model_dump(mode="json", by_alias=True, exclude_none=False)
        for artifact in provisional.payload_files
    ]
    derived_payload["payload_tree_sha256"] = sha256_bytes(canonical_json_bytes(canonical_inventory))
    derived_payload["sha256s_sha256"] = sha256_bytes(
        "".join(
            f"{artifact.sha256}  {artifact.path}\n" for artifact in provisional.payload_files
        ).encode()
    )
    derived = DerivedRunManifest.model_validate(derived_payload)
    derived_path.write_bytes(canonical_json_bytes(derived))
    manifest["reviewed_derived"]["derived_run_sha256"] = sha256_file(derived_path)
    _update_outer_payload_reference(manifest, root, "reviewed/derived-manifest.json")

    summary_path = root / "reviewed/public-summary.json"
    summary = read_json(summary_path)
    summary["export_provenance_modes"]["reviewed_derived"]["derived_run_sha256"] = sha256_file(
        derived_path
    )
    summary_path.write_bytes(canonical_json_bytes(summary))
    manifest["outputs"]["public_summary"]["sha256"] = sha256_file(summary_path)
    _update_outer_payload_reference(manifest, root, "reviewed/public-summary.json")

    manifest["files"] = sorted(manifest["files"], key=lambda item: item["path"])
    (root / "publication-manifest.json").write_bytes(canonical_json_bytes(manifest))
    _refresh_outer_controls(root)
    return eee_path


def test_builds_and_verifies_deterministic_reviewed_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _reviewed_inputs(monkeypatch, tmp_path)
    first = _build_reviewed(inputs, tmp_path / "public-one")
    second = _build_reviewed(inputs, tmp_path / "public-two")

    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    receipt = verify_public_reviewed_development_bundle(first)
    assert receipt["status"] == "verified"
    assert receipt["eee_records"] == 1
    assert receipt["evaluation_results"] > 0

    preview_root = inputs[1]
    assert {path.name: path.read_bytes() for path in (first / "candidate-preview").iterdir()} == {
        path.name: path.read_bytes() for path in preview_root.iterdir()
    }
    canonical = read_json(first / "reviewed/canonical-results.json")
    assert canonical["schema_version"] == PUBLIC_REVIEWED_CANONICAL_RESULTS_SCHEMA_VERSION
    assert canonical["development_only"] is True
    assert canonical["independent_validation"] is False
    assert canonical["result_count"] == receipt["evaluation_results"]
    identities = [
        (row["paper_id"], row["evaluation_id"], row["evaluation_result_id"])
        for row in canonical["results"]
    ]
    assert identities == sorted(identities)
    assert len(identities) == len(set(identities))
    row = canonical["results"][0]
    assert list(row) == [
        "eee_path",
        "eval_library",
        "evaluation_id",
        "evaluation_result",
        "evaluation_result_id",
        "model_info",
        "paper_id",
        "retrieved_timestamp",
        "source_metadata",
    ]
    assert "<script" not in (first / "reviewed/canonical-results.html").read_text().lower()


def test_empty_reviewed_eee_is_valid_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _reviewed_inputs(monkeypatch, tmp_path, empty=True)
    output = _build_reviewed(inputs, tmp_path / "public-empty")

    receipt = verify_public_reviewed_development_bundle(output)
    canonical = read_json(output / "reviewed/canonical-results.json")
    manifest = read_json(output / "publication-manifest.json")
    assert receipt["eee_records"] == 0
    assert receipt["evaluation_results"] == 0
    assert canonical["results"] == []
    assert manifest["reviewed_derived"]["eee_files"] == []
    assert manifest["status"] == "verified_reviewed_empty_development"
    assert manifest["artifact_status"]["publication_ready"] is False
    assert not (output / "reviewed/eee").exists()


def test_all_pending_candidates_are_withheld_and_not_publication_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = _build_reviewed(
        _reviewed_inputs(monkeypatch, tmp_path, review_mode="pending"),
        tmp_path / "public",
    )

    verify_public_reviewed_development_bundle(output)
    manifest = read_json(output / "publication-manifest.json")
    canonical = read_json(output / "reviewed/canonical-results.json")
    assert manifest["status"] == "verified_review_incomplete_development"
    assert manifest["artifact_status"]["human_review_complete"] is False
    assert manifest["artifact_status"]["publication_ready"] is False
    assert manifest["outputs"]["review_items"] > 0
    assert manifest["outputs"]["decisions_completed"] == 0
    assert manifest["outputs"]["decisions_pending"] == manifest["outputs"]["review_items"]
    assert manifest["outputs"]["outcomes_withheld"] == manifest["outputs"]["review_items"]
    assert canonical["results"] == []


def test_reviewed_exports_remain_ready_when_pending_candidates_are_withheld(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = _build_reviewed(
        _reviewed_inputs(monkeypatch, tmp_path, review_mode="partial"),
        tmp_path / "public",
    )

    verify_public_reviewed_development_bundle(output)
    manifest = read_json(output / "publication-manifest.json")
    canonical = read_json(output / "reviewed/canonical-results.json")
    assert manifest["status"] == ("verified_human_reviewed_development_with_pending_withheld")
    assert manifest["artifact_status"]["human_review_complete"] is False
    assert manifest["artifact_status"]["publication_ready"] is True
    assert manifest["outputs"]["decisions_completed"] == 1
    assert manifest["outputs"]["decisions_pending"] == 1
    assert manifest["outputs"]["outcomes_exported"] == 1
    assert manifest["outputs"]["outcomes_withheld"] == 1
    assert canonical["result_count"] == 1
    # Readers of the HTML need the pending-review limitation as well as the metadata.
    html = (output / "reviewed/canonical-results.html").read_text()
    scope_note = html.split('id="scope-note">', 1)[1].split("</p>", 1)[0]
    assert "pending" in scope_note and "withheld" in scope_note


@pytest.mark.parametrize("evidence_source", ["run", "review_only"])
def test_build_rejects_reviewed_fields_that_reproduce_private_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evidence_source: str,
) -> None:
    review_only = evidence_source == "review_only"
    quote = PRIVATE_REVIEW_EXCERPT if review_only else PRIVATE_EVIDENCE_QUOTE
    inputs = _reviewed_inputs(
        monkeypatch,
        tmp_path,
        private_evidence_reproduction=not review_only,
        review_evidence_reproduction=review_only,
    )
    assert quote in collect_private_evidence_texts(inputs[3 if review_only else 0])
    if review_only:
        assert quote not in collect_private_evidence_texts(inputs[0])
    assert any(
        result["metric_config"].get("additional_details", {}).get("operationalization") == quote
        for path in inputs[2].glob("*/eee/*.json")
        for result in read_json(path)["evaluation_results"]
    )
    output_root = tmp_path / "public-leak"

    with pytest.raises(
        PublicReviewedDevelopmentBundleError,
        match="private-evidence nonreproduction gate",
    ):
        _build_reviewed(inputs, output_root, bundle_id="must-not-publish")
    assert not (output_root / "must-not-publish").exists()


def test_private_evidence_guard_checks_free_form_mapping_keys() -> None:
    with pytest.raises(PublicDevelopmentPreviewError, match="evidence quotation"):
        assert_no_private_evidence_text(
            {PRIVATE_EVIDENCE_QUOTE: "otherwise safe"},
            {PRIVATE_EVIDENCE_QUOTE},
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("tamper", "verification failed"),
        ("extra", "inventory"),
        ("symlink", "symbolic link"),
    ],
)
def test_standalone_verifier_rejects_tree_tamper_extra_and_symlink(
    reviewed_bundle: Path,
    mutation: str,
    match: str,
) -> None:
    output = reviewed_bundle
    if mutation == "tamper":
        (output / "reviewed/canonical-results.html").write_text("changed", encoding="utf-8")
    elif mutation == "extra":
        (output / "unexpected.txt").write_text("extra", encoding="utf-8")
    else:
        (output / "unexpected-link").symlink_to(output / "README.md")

    with pytest.raises(PublicReviewedDevelopmentBundleError, match=match):
        verify_public_reviewed_development_bundle(output)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("canonical_score", "projection disagrees"),
        ("summary_scope", "development-only"),
        ("private_reviewers", "private field"),
    ],
)
def test_fully_rehashed_json_projection_tamper_is_rejected(
    reviewed_bundle: Path, mutation: str, message: str
) -> None:
    relative = (
        "reviewed/public-summary.json"
        if mutation == "summary_scope"
        else "reviewed/canonical-results.json"
    )
    path = reviewed_bundle / relative
    value = read_json(path)
    if mutation == "canonical_score":
        value["results"][0]["evaluation_result"]["score_details"]["score"] = 0.123456
    elif mutation == "summary_scope":
        value["scope"]["independent_human_validation"] = True
    else:
        value["results"][0]["reviewer_ids"] = ["private-reviewer"]
    path.write_bytes(canonical_json_bytes(value))
    _rehash_payload(reviewed_bundle, relative)

    with pytest.raises(PublicReviewedDevelopmentBundleError, match=message):
        verify_public_reviewed_development_bundle(reviewed_bundle)


def test_fully_rehashed_html_tamper_is_rejected(
    reviewed_bundle: Path,
) -> None:
    output = reviewed_bundle
    html = output / "reviewed/canonical-results.html"
    html.write_text(html.read_text().replace("Reviewed development", "Altered development"))
    _rehash_payload(output, "reviewed/canonical-results.html")

    with pytest.raises(PublicReviewedDevelopmentBundleError, match="HTML is not deterministic"):
        verify_public_reviewed_development_bundle(output)


def test_fully_rehashed_noncanonical_eee_is_rejected(
    reviewed_bundle: Path,
) -> None:
    output = reviewed_bundle
    manifest = read_json(output / "publication-manifest.json")
    [eee_entry] = manifest["reviewed_derived"]["eee_files"]
    record = read_json(output / eee_entry["bundle_path"])
    noncanonical = (json.dumps(record, ensure_ascii=False, indent=4) + "\n").encode()
    _fully_rehash_published_eee(output, noncanonical)

    with pytest.raises(PublicReviewedDevelopmentBundleError, match="not canonical JSON"):
        verify_public_reviewed_development_bundle(output)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("request_id", "private-request-123", "private field"),
        ("quote", "private exact source excerpt", "private field"),
        ("raw_payload", "opaque private payload", "private field"),
        ("artifact_path", "/custom/root/paper.json", "local path"),
        ("/secret", "otherwise-safe", "local path"),
        ("transport_note", "Bearer abcdefghijklmnop", "credential material"),
    ],
)
def test_fully_rehashed_eee_privacy_material_is_rejected(
    reviewed_bundle: Path,
    key: str,
    value: str,
    match: str,
) -> None:
    output = reviewed_bundle
    manifest = read_json(output / "publication-manifest.json")
    [eee_entry] = manifest["reviewed_derived"]["eee_files"]
    record = read_json(output / eee_entry["bundle_path"])
    details = record["evaluation_results"][0]["score_details"]["details"]
    details[key] = value
    _fully_rehash_published_eee(output, canonical_json_bytes(record))

    with pytest.raises(PublicReviewedDevelopmentBundleError, match=match):
        verify_public_reviewed_development_bundle(output)


def test_fully_rehashed_eee_review_binding_mismatch_is_rejected(
    reviewed_bundle: Path,
) -> None:
    output = reviewed_bundle
    manifest = read_json(output / "publication-manifest.json")
    [eee_entry] = manifest["reviewed_derived"]["eee_files"]
    record = read_json(output / eee_entry["bundle_path"])
    record["evaluation_results"][0]["score_details"]["details"]["review_lock_sha256"] = "0" * 64
    _fully_rehash_published_eee(output, canonical_json_bytes(record))

    with pytest.raises(PublicReviewedDevelopmentBundleError, match="review binding disagrees"):
        verify_public_reviewed_development_bundle(output)


def test_outer_manifest_cannot_authorize_eee_changed_from_copied_derived_manifest(
    reviewed_bundle: Path,
) -> None:
    output = reviewed_bundle
    manifest = read_json(output / "publication-manifest.json")
    [eee_entry] = manifest["reviewed_derived"]["eee_files"]
    eee_path = output / eee_entry["bundle_path"]
    eee = read_json(eee_path)
    eee["evaluation_results"][0]["score_details"]["score"] = 0.123456

    eee_path.write_bytes(canonical_json_bytes(eee))
    changed_hash = sha256_file(eee_path)
    changed_size = eee_path.stat().st_size
    eee_entry["sha256"] = changed_hash
    eee_entry["size_bytes"] = changed_size
    for artifact in manifest["files"]:
        if artifact["path"] == eee_entry["bundle_path"]:
            artifact["sha256"] = changed_hash
            artifact["size_bytes"] = changed_size
            break
    (output / "publication-manifest.json").write_bytes(canonical_json_bytes(manifest))
    _refresh_outer_controls(output)

    with pytest.raises(
        PublicReviewedDevelopmentBundleError,
        match="verified derived manifest",
    ):
        verify_public_reviewed_development_bundle(output)


def test_build_rejects_preview_from_another_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _reviewed_inputs(monkeypatch, tmp_path / "first")
    second = _reviewed_inputs(monkeypatch, tmp_path / "second")

    with pytest.raises(PublicReviewedDevelopmentBundleError, match="bind different source"):
        build_public_reviewed_development_bundle(
            bundle_id="mismatch",
            run_root=first[0],
            preview_root=second[1],
            reviewed_derived_root=first[2],
            review_root=first[3],
            output_root=tmp_path / "public",
        )


def test_build_is_fresh_and_rejects_overlapping_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _reviewed_inputs(monkeypatch, tmp_path)
    _build_reviewed(inputs, tmp_path / "public")
    with pytest.raises(PublicReviewedDevelopmentBundleError, match="destination exists"):
        _build_reviewed(inputs, tmp_path / "public")
    with pytest.raises(PublicReviewedDevelopmentBundleError, match="disjoint"):
        _build_reviewed(inputs, inputs[0], bundle_id="inside-run")


@pytest.mark.parametrize("dotdot_variant", [False, True])
def test_nonexistent_output_nested_under_source_is_rejected_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dotdot_variant: bool,
) -> None:
    inputs = _reviewed_inputs(monkeypatch, tmp_path)
    source_root = inputs[0]
    before_seal = verify_run_seal(source_root)
    before_files = {
        path.relative_to(source_root).as_posix(): (sha256_file(path), path.stat().st_size)
        for path in source_root.rglob("*")
        if path.is_file()
    }
    output_root = source_root / "new-public"
    if dotdot_variant:
        output_root = (
            source_root.parent / "nonexistent-parent" / ".." / source_root.name / "new-public"
        )

    with pytest.raises(PublicReviewedDevelopmentBundleError, match="disjoint"):
        _build_reviewed(inputs, output_root, bundle_id="must-not-exist")

    assert not (source_root / "new-public").exists()
    assert not (source_root.parent / "nonexistent-parent").exists()
    assert verify_run_seal(source_root) == before_seal
    assert {
        path.relative_to(source_root).as_posix(): (sha256_file(path), path.stat().st_size)
        for path in source_root.rglob("*")
        if path.is_file()
    } == before_files


def test_exclusive_publish_preserves_race_created_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _reviewed_inputs(monkeypatch, tmp_path)
    output_root = tmp_path / "public"
    destination = output_root / "race-reviewed"
    original_mkdir = bundle_module.os.mkdir
    raced = False

    def racing_mkdir(
        path: object,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal raced
        if path == destination.name and dir_fd is not None and not raced:
            raced = True
            original_mkdir(path, mode, dir_fd=dir_fd)
            (destination / "sentinel.txt").write_text("owned by another publisher")
            raise FileExistsError(destination.name)
        original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(bundle_module.os, "mkdir", racing_mkdir)
    with pytest.raises(PublicReviewedDevelopmentBundleError, match="appeared during publication"):
        _build_reviewed(inputs, output_root, bundle_id=destination.name)

    assert raced is True
    assert (destination / "sentinel.txt").read_text() == "owned by another publisher"
    assert {path.name for path in destination.iterdir()} == {"sentinel.txt"}


def test_standalone_verifier_rejects_mutation_during_verification(
    reviewed_bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = reviewed_bundle
    original_render = bundle_module._render_results_html
    mutated = False

    def mutating_render(canonical: object) -> str:
        nonlocal mutated
        if not mutated:
            mutated = True
            (output / "README.md").write_text("changed concurrently")
        return original_render(canonical)

    monkeypatch.setattr(bundle_module, "_render_results_html", mutating_render)
    with pytest.raises(PublicReviewedDevelopmentBundleError, match="changed during verification"):
        verify_public_reviewed_development_bundle(output)
    assert mutated is True


def test_cli_build_verify_and_safe_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, preview_root, derived_root, review_root = _reviewed_inputs(monkeypatch, tmp_path)
    output_root = tmp_path / "cli-public"
    result = RUNNER.invoke(
        app,
        [
            "build-public-reviewed-development-bundle",
            "cli-reviewed",
            str(run_root),
            str(preview_root),
            "--reviewed-derived-root",
            str(derived_root),
            "--review-root",
            str(review_root),
            "--output-root",
            str(output_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "public-reviewed-development-bundle-written"

    verify_result = RUNNER.invoke(
        app,
        ["verify-public-reviewed-development-bundle", str(output_root / "cli-reviewed")],
    )
    assert verify_result.exit_code == 0, verify_result.output
    assert json.loads(verify_result.output)["status"] == "verified"

    broken = tmp_path / "broken-copy"
    shutil.copytree(output_root / "cli-reviewed", broken)
    (broken / "reviewed/canonical-results.json").write_text("private marker")
    failed = RUNNER.invoke(app, ["verify-public-reviewed-development-bundle", str(broken)])
    assert failed.exit_code == 1
    payload = json.loads(failed.output)
    assert payload["status"] == "public-reviewed-development-bundle-verification-failed"
    assert "private marker" not in failed.output


def test_build_cli_generic_failure_does_not_echo_private_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _reviewed_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "proceedings_to_eee.cli.build_public_reviewed_development_bundle",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("private-reviewer-secret")),
    )
    result = RUNNER.invoke(
        app,
        [
            "build-public-reviewed-development-bundle",
            "safe-error",
            str(inputs[0]),
            str(inputs[1]),
            "--reviewed-derived-root",
            str(inputs[2]),
            "--review-root",
            str(inputs[3]),
            "--output-root",
            str(tmp_path / "public"),
        ],
    )
    assert result.exit_code == 1
    assert "private-reviewer-secret" not in result.output
