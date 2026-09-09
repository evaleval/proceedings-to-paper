from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_origin_pipeline import _origin_fixture
from test_public_development_summary import _FiveStageRowClient
from typer.testing import CliRunner

import proceedings_to_eee.reporting.public_development_preview as preview_module
from proceedings_to_eee.cli import app
from proceedings_to_eee.corpus import (
    CorpusSpec,
    ExpectedSpotCheck,
    build_corpus_binding,
    load_corpus,
)
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from proceedings_to_eee.pipeline import run_corpus
from proceedings_to_eee.reporting.public_development_preview import (
    PublicDevelopmentPreviewError,
    build_public_development_preview,
    verify_public_development_preview,
)
from proceedings_to_eee.run_seal import seal_run_tree, verify_run_seal

RUNNER = CliRunner()
COMMIT = "1" * 40
SOURCE_TREE = "2" * 64
ROUTE_STAGES = (
    "block_candidate_extraction",
    "row_disposition",
    "tuple_resolution",
    "independent_verification",
    "origin_retrieval",
)
EXPECTED_FILES = {
    "README.md",
    "SHA256SUMS",
    "corpus.json",
    "evidence-map.html",
    "evidence-map.json",
    "publication-manifest.json",
    "route-selection.json",
    "run-summary.json",
    "sources.json",
    "usage.json",
    "verification.json",
}


@dataclass(frozen=True)
class _PreviewFixture:
    sealed_root: Path
    corpus_path: Path
    route_path: Path
    output_root: Path


class _ZeroCandidateFiveStageRowClient(_FiveStageRowClient):
    def structured_chat(self, **kwargs: Any):
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] == "paper_table_row_dispositions":
            payload = {
                "dispositions": [
                    {
                        "row_id": item["row_id"],
                        "disposition": "not_result",
                        "observations": [],
                        "note": "no extractable result",
                    }
                    for item in response.payload["dispositions"]
                ],
                "warnings": [],
            }
        elif kwargs["schema_name"] == "paper_evaluation_candidates":
            payload = {
                "observations": [],
                "page_summary": "no extractable observations",
                "warnings": [],
            }
        else:
            return response
        return replace(
            response,
            payload=payload,
            call=response.call.model_copy(
                update={
                    "response_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True).encode()
                    ).hexdigest()
                }
            ),
        )


class _ExcessiveAttemptFiveStageRowClient(_FiveStageRowClient):
    def structured_chat(self, **kwargs: Any):
        response = super().structured_chat(**kwargs)
        return replace(response, call=response.call.model_copy(update={"attempts": 10_000}))


def _route(models: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "repair05-diagnostic-route-selection/0.1",
        "status": "diagnostic_only",
        "experiment_id": "fixture-experiment",
        "amendment_id": "fixture-amendment",
        "code_commit": COMMIT,
        "code_tree": "3" * 40,
        "runner_sha256": "4" * 64,
        "campaign_seal_sha256": "5" * 64,
        "plan_bundle_sha256": "6" * 64,
        "stages": {
            stage: {
                "execution_selection_sha256": f"{index + 1:x}" * 64,
                "quality_source_artifact_sha256": None,
                "selected_model": models[stage],
                "selection_basis": "technical_only_unmeasured_quality",
            }
            for index, stage in enumerate(ROUTE_STAGES)
        },
    }


def _precall(
    *,
    route_path: Path,
    corpus_path: Path,
    corpus: CorpusSpec,
    settings: Any,
    models: dict[str, str],
) -> dict[str, Any]:
    provider_budget = read_json(settings.output_root / "corpus-run.json")["provider_budget"]
    contract_event = json.loads(
        (settings.output_root / provider_budget["ledger"])
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    provider_budget_contract_sha256 = hashlib.sha256(
        json.dumps(
            contract_event["contract"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert provider_budget_contract_sha256 != provider_budget["contract_sha256"]
    result = {
        "schema_version": "repair05-ten-paper-page-scoped-precall-seal/0.1",
        "status": "sealed_before_page_scoped_provider_execution",
        "sealed_at": "2026-09-03T00:00:00Z",
        "bindings": {
            "campaign_plan_sha256": "a" * 64,
            "route_selection_sha256": sha256_file(route_path),
            "route_ranking_audit_sha256": "b" * 64,
            "campaign_seal_sha256": "5" * 64,
            "campaign_seal_file_sha256": "c" * 64,
            "corpus_config_sha256": sha256_file(corpus_path),
            "corpus_spec_sha256": build_corpus_binding(corpus)["corpus_spec_sha256"],
            "paper_ids_sha256": build_corpus_binding(corpus)["paper_ids_sha256"],
            "corpus_preflight_sha256": "d" * 64,
            "corpus_freeze_sha256": "e" * 64,
            "preflight_inventory_sha256": "f" * 64,
            "schema_sha256": settings.schema_sha256,
            "provider_budget_contract_sha256": provider_budget_contract_sha256,
            "morning_packet_manifest_sha256": "8" * 64,
        },
        "declared_models": sorted(set(models.values())),
        "selected_models": models,
        "command_argv": ["python", "-m", "proceedings_to_eee", "run-corpus"],
        "budget": {
            "max_structured_calls": provider_budget["max_structured_calls"],
            "max_provider_cost_usd": provider_budget["max_cost_usd"],
            "provider_call_cost_reservation_usd": provider_budget["cost_reservation_per_call_usd"],
            "max_transport_attempts": provider_budget["max_structured_calls"] * 4,
        },
        "privacy": {
            "zdr": True,
            "data_collection": "deny",
            "label_blind_source_derived_scientific_payloads_only": True,
            (
                "human_labels_answers_scores_holdout_references_reviewer_ids_"
                "credentials_or_private_annotations_sent"
            ): False,
        },
        "outputs": {
            "raw_root": str(settings.output_root),
            "sealed_root": str(settings.output_root.parent / "sealed"),
            "raw_root_was_fresh_before_seal": True,
            "sealed_root_was_fresh_before_seal": True,
        },
        "execution_policy": {
            "working_directory": str(settings.project_root),
            "command_argv": ["python", "-m", "proceedings_to_eee", "run-corpus"],
            "seal_command_argv": ["python", "-m", "proceedings_to_eee", "seal-run"],
            "accepted_terminal_exit_codes": [0, 1, 2, 3],
            "completed_with_review_exit_code": 2,
            "bounded_incomplete_exit_code": 3,
            "generic_budget_doubling_forbidden": True,
            "copy_precall_seal_into_raw_root_before_execution": True,
            "seal_regardless_of_terminal_exit": True,
            "verify_seal_before_semantic_inspection": True,
        },
    }
    result["seal_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def _fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    client_mode: str = "positive",
    local_source_uris: bool = False,
    route_extra_key: bool = False,
    route_model_mismatch: bool = False,
    precall_zdr: bool = True,
    precall_copy_before_execution: bool = True,
    precall_budget_contract_mismatch: bool = False,
    reference_material: bool = False,
    client: Any | None = None,
) -> _PreviewFixture:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    settings = replace(settings, row_enumeration_enabled=True)

    if reference_material:
        spec = spec.model_copy(
            update={
                "expected_spot_checks": [
                    ExpectedSpotCheck(
                        system="System A",
                        dataset="Dataset A",
                        metric="AUC",
                        raw_value="0.80",
                        page=2,
                        label="private-reference-label",
                    )
                ]
            }
        )

    if local_source_uris:
        manifest_path = settings.output_root / spec.paper_id / "source-manifest.json"
        manifest = read_json(manifest_path)
        local_path = (tmp_path / manifest["sources"][0]["cache_relpath"]).resolve()
        spec = spec.model_copy(update={"pdf_url": None, "pdf_path": str(local_path)})
        manifest["sources"][0]["original_uri"] = str(local_path)
        manifest["sources"][0]["resolved_uri"] = local_path.as_uri()
        write_json(manifest_path, manifest)

    corpus = CorpusSpec(
        corpus_id="five-stage-development",
        evaluation_split="development",
        description="Provider-free five-stage preview fixture.",
        papers=[spec],
    )
    corpus_path = tmp_path / "inputs" / "corpus.yaml"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(
        yaml.safe_dump(corpus.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    corpus = load_corpus(corpus_path)

    monkeypatch.setattr(
        "proceedings_to_eee.pipeline._code_state",
        lambda _root: {
            "git_commit": COMMIT,
            "git_dirty": False,
            "git_available": True,
            "source_tree_sha256": SOURCE_TREE,
        },
    )
    run_corpus(
        corpus=corpus,
        settings=settings,
        client=client or _FiveStageRowClient(client_mode),
    )

    models = {
        "block_candidate_extraction": "fixture/model",
        "row_disposition": "fixture/model",
        "tuple_resolution": "fixture/tuple",
        "independent_verification": "fixture/verifier",
        "origin_retrieval": "fixture/origin",
    }
    route = _route(models)
    if route_extra_key:
        route["stages"]["row_disposition"]["unexpected"] = True
    if route_model_mismatch:
        route["stages"]["origin_retrieval"]["selected_model"] = "fixture/other-origin"
        models = {**models, "origin_retrieval": "fixture/other-origin"}
    route_path = tmp_path / "inputs" / "caller-selected-route.json"
    write_json(route_path, route)

    precall = _precall(
        route_path=route_path,
        corpus_path=corpus_path,
        corpus=corpus,
        settings=settings,
        models=models,
    )
    precall["privacy"]["zdr"] = precall_zdr
    precall["execution_policy"]["copy_precall_seal_into_raw_root_before_execution"] = (
        precall_copy_before_execution
    )
    if precall_budget_contract_mismatch:
        precall["bindings"]["provider_budget_contract_sha256"] = "7" * 64
    precall["seal_sha256"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in precall.items() if key != "seal_sha256"})
    )
    write_json(settings.output_root / "prospective-contract.json", precall)
    sealed_root = tmp_path / "sealed-source-run"
    seal_run_tree(settings.output_root, sealed_root)
    return _PreviewFixture(
        sealed_root=sealed_root,
        corpus_path=corpus_path,
        route_path=route_path,
        output_root=tmp_path / "public",
    )


def _build(inputs: _PreviewFixture, *, output_root: Path | None = None) -> Path:
    return build_public_development_preview(
        bundle_id="fixture-preview",
        run_root=inputs.sealed_root,
        corpus_path=inputs.corpus_path,
        route_manifest_path=inputs.route_path,
        output_root=output_root or inputs.output_root,
        expected_paper_count=1,
    )


def _refresh_checksums(output: Path) -> None:
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(output / name)}  {name}\n"
            for name in sorted(EXPECTED_FILES - {"SHA256SUMS"})
        ),
        encoding="utf-8",
    )


def test_stage_usage_keeps_exact_fields_null_when_global_coverage_is_lower_bound() -> None:
    usage = {
        "calls_attempted": 1,
        "calls_field_count": 1,
        "resumed_calls_field_count": 0,
        "call_accounting_basis": "calls_plus_resumed_calls",
        "model_returned_matches_requested_calls": 1,
        "model_returned_unverified_calls": 0,
        "models_returned": ["fixture/model"],
        "providers_returned": ["Fixture"],
        "attempts_lower_bound": 1,
        "retries_lower_bound": 0,
    }
    for prefix, lower in (
        ("cost", 0.01),
        ("input_tokens", 10),
        ("output_tokens", 2),
        ("reasoning_tokens", 0),
        ("total_tokens", 12),
    ):
        usage[f"{prefix}_reported_calls"] = 1
        usage[f"{prefix}_missing_calls"] = 0
        usage[f"{prefix}_usd_lower_bound" if prefix == "cost" else f"{prefix}_lower_bound"] = lower
        usage["cost_usd" if prefix == "cost" else prefix] = None

    validated = preview_module._validate_stage_usage(usage, "fixture usage", require_exact=False)
    assert validated["cost_usd"] is None
    with pytest.raises(PublicDevelopmentPreviewError, match="exact cost disagrees"):
        preview_module._validate_stage_usage(usage, "fixture usage", require_exact=True)


def test_private_evidence_guard_rejects_long_embedded_normalized_quote() -> None:
    quote = "Scores for System A were taken from the public leaderboard for Dataset A AUC."
    assert len(quote) >= preview_module._MIN_EMBEDDED_PRIVATE_EVIDENCE_CHARS
    public_value = {
        "statement": (
            "Aggregate context only: Scores for System A were taken\n"
            "from the public leaderboard for Dataset A AUC. Human review remains pending."
        )
    }

    with pytest.raises(PublicDevelopmentPreviewError, match="evidence quotation"):
        preview_module._assert_no_evidence_text(public_value, {quote})

    short_common_text = "human review pending"
    assert len(short_common_text) < preview_module._MIN_EMBEDDED_PRIVATE_EVIDENCE_CHARS
    preview_module._assert_no_evidence_text(
        {"statement": "Human review pending; aggregate counts only."}, {short_common_text}
    )


def test_builds_deterministic_pre_human_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    first = _build(inputs, output_root=tmp_path / "public-one")
    second = _build(inputs, output_root=tmp_path / "public-two")

    assert {path.name for path in first.iterdir()} == EXPECTED_FILES
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    assert verify_public_development_preview(first)["status"] == "verified"

    manifest = read_json(first / "publication-manifest.json")
    assert manifest["artifact_status"] == {
        "canonical_real_paper_eee_available": False,
        "classification": "pre_human_development_preview",
        "human_review_status": "pending",
        "release_ready": False,
    }
    assert manifest["outputs"]["reviewed_derived"] == {
        "expected_relative_path": "reviewed/eee",
        "present": False,
        "records": None,
        "status": "pending_human_review",
    }
    assert manifest["outputs"]["automatic_source_run"]["canonical_eee_records"] == 0

    evidence = read_json(first / "evidence-map.json")
    paper = evidence["papers"][0]
    assert evidence["status"] == "model_proposed_human_review_pending"
    assert paper["page_scope"] == {
        "selected_pages": [2],
        "selection_mode": "configured_page_scope",
    }
    assert paper["review_boundary"]["human_review_decisions_completed"] == 0
    assert paper["review_boundary"]["canonical_eee_records"] == 0
    assert paper["review_boundary"]["final_reviewed_records"] is None
    assert "System A                 0.80" not in json.dumps(evidence)

    usage = read_json(first / "usage.json")
    assert usage["qualification_campaign_usage"] == {
        "combined_with_source_run": False,
        "status": "separate_not_included",
    }
    assert usage["provider_budget"]["structured_calls_pending"] == 0
    assert usage["replayed_provider_usage"]["recorded_structured_invocations"] == sum(
        stage["completed_calls"] for stage in usage["stage_receipts"].values()
    )
    assert usage["operational_accounting"]["wall_clock_seconds"] >= 0

    serialized = b"\n".join(path.read_bytes() for path in sorted(first.iterdir()))
    for forbidden in (
        b'"exact_excerpt":',
        b'"observation_id":',
        b'"proposal_id":',
        b'"request_id":',
        b'"reviewer_id":',
        b"tuple_ev_",
        b"obs_",
        b"/Users/",
        b"file://",
    ):
        assert forbidden not in serialized


def test_successful_zero_candidate_run_is_truthfully_pending_and_verifiable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(
        monkeypatch,
        tmp_path,
        client=_ZeroCandidateFiveStageRowClient(),
    )

    output = _build(inputs)

    summary = read_json(output / "run-summary.json")
    evidence = read_json(output / "evidence-map.json")
    manifest = read_json(output / "publication-manifest.json")
    assert summary["outputs"]["candidates"] == 0
    assert summary["outputs"]["candidate_proposal_removal_rate"] is None
    assert summary["canonical_eee"]["records"] == 0
    assert evidence["papers"][0]["review_boundary"] == {
        "accepted_by_model_gates": 0,
        "canonical_eee_records": 0,
        "final_reviewed_records": None,
        "human_review_decisions_completed": 0,
        "not_accepted_by_complete_model_chain": 0,
        "status": "model_proposed_human_review_pending",
        "withheld_from_canonical_eee": 0,
    }
    assert manifest["outputs"]["reviewed_derived"]["status"] == "pending_human_review"
    assert verify_public_development_preview(output)["status"] == "verified"


def test_local_source_uris_are_omitted_not_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path, local_source_uris=True)

    output = _build(inputs)

    source = read_json(output / "sources.json")["papers"][0]["sources"][0]
    assert source["public_original_uri"] is None
    assert source["public_resolved_uri"] is None
    assert "/Users/" not in (output / "sources.json").read_text(encoding="utf-8")


def test_typed_stage_failure_remains_pre_human_and_not_release_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path, client_mode="invalid")

    output = _build(inputs)

    summary = read_json(output / "run-summary.json")
    manifest = read_json(output / "publication-manifest.json")
    assert summary["technical_health"]["status"] == "error"
    assert summary["run_binding"]["stage_chain"]["five_stage_gate_status"] == "partial_failure"
    assert manifest["artifact_status"]["release_ready"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"route_extra_key": True}, "exact public schema"),
        ({"route_model_mismatch": True}, "model disagrees with route"),
        ({"precall_zdr": False}, "privacy assertions"),
        ({"precall_copy_before_execution": False}, "execution policy"),
        (
            {"precall_budget_contract_mismatch": True},
            "provider budget contract disagrees",
        ),
    ],
)
def test_route_and_precall_tampering_fails_closed_and_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    inputs = _fixture(monkeypatch, tmp_path, **kwargs)

    with pytest.raises(PublicDevelopmentPreviewError, match=message):
        _build(inputs)

    assert not (inputs.output_root / "fixture-preview").exists()


def test_precall_transport_attempt_ceiling_is_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(
        monkeypatch,
        tmp_path,
        client=_ExcessiveAttemptFiveStageRowClient(),
    )

    with pytest.raises(PublicDevelopmentPreviewError, match="attempts exceed"):
        _build(inputs)

    assert not (inputs.output_root / "fixture-preview").exists()


def test_corpus_binding_tamper_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    text = inputs.corpus_path.read_text(encoding="utf-8")
    inputs.corpus_path.write_text(
        text.replace("five-stage-development", "another-development"), encoding="utf-8"
    )

    with pytest.raises(PublicDevelopmentPreviewError, match="corpus specification"):
        _build(inputs)

    assert not (inputs.output_root / "fixture-preview").exists()


def test_output_must_be_fresh_and_outside_sealed_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    existing = inputs.output_root / "fixture-preview"
    existing.mkdir(parents=True)
    with pytest.raises(PublicDevelopmentPreviewError, match="already exists"):
        _build(inputs)

    with pytest.raises(PublicDevelopmentPreviewError, match="outside the sealed run"):
        _build(inputs, output_root=inputs.sealed_root / "public")


@pytest.mark.parametrize("dotdot_variant", [False, True])
def test_nonexistent_nested_output_is_rejected_before_mutating_sealed_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dotdot_variant: bool,
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    before_seal = verify_run_seal(inputs.sealed_root)
    before_files = {
        path.relative_to(inputs.sealed_root).as_posix(): (sha256_file(path), path.stat().st_size)
        for path in inputs.sealed_root.rglob("*")
        if path.is_file()
    }
    output_root = inputs.sealed_root / "new-public"
    if dotdot_variant:
        output_root = (
            inputs.sealed_root.parent
            / "nonexistent-parent"
            / ".."
            / inputs.sealed_root.name
            / "new-public"
        )

    with pytest.raises(PublicDevelopmentPreviewError, match="outside the sealed run"):
        _build(inputs, output_root=output_root)

    assert not (inputs.sealed_root / "new-public").exists()
    assert not (inputs.sealed_root.parent / "nonexistent-parent").exists()
    assert verify_run_seal(inputs.sealed_root) == before_seal
    assert {
        path.relative_to(inputs.sealed_root).as_posix(): (sha256_file(path), path.stat().st_size)
        for path in inputs.sealed_root.rglob("*")
        if path.is_file()
    } == before_files


def test_standalone_verifier_rejects_checksum_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    readme = output / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nbenign change\n", encoding="utf-8")

    with pytest.raises(PublicDevelopmentPreviewError, match="checksum verification"):
        verify_public_development_preview(output)


def test_standalone_verifier_rejects_extra_file_and_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    extra = output / "private.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PublicDevelopmentPreviewError, match="allowlist"):
        verify_public_development_preview(output)

    extra.unlink()
    readme = output / "README.md"
    readme.write_text("OPENROUTER_API_KEY\n", encoding="utf-8")
    with pytest.raises(PublicDevelopmentPreviewError, match="credential-like"):
        verify_public_development_preview(output)


def test_standalone_verifier_rejects_nested_schema_extension_with_fresh_checksum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    evidence_path = output / "evidence-map.json"
    evidence = read_json(evidence_path)
    evidence["papers"][0]["review_boundary"]["human_label"] = "not-reviewed"
    write_json(evidence_path, evidence)
    _refresh_checksums(output)

    with pytest.raises(PublicDevelopmentPreviewError, match="exact public schema"):
        verify_public_development_preview(output)


@pytest.mark.parametrize(
    ("file_name", "mutation"),
    [
        ("corpus.json", "corpus-root"),
        ("run-summary.json", "summary-nested"),
        ("usage.json", "usage-deep"),
    ],
)
def test_standalone_verifier_rejects_deep_schema_extensions_with_fresh_checksums(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    file_name: str,
    mutation: str,
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    path = output / file_name
    payload = read_json(path)
    if mutation == "corpus-root":
        payload["papers_detail"][0]["extractor"]["extra_public_claim"] = "not schema-bound"
    elif mutation == "summary-nested":
        payload["outputs"]["extra_count"] = 0
    else:
        payload["paper_accounting"][0]["stages"]["origin_retrieval"]["usage"]["extra_count"] = 0
    write_json(path, payload)
    _refresh_checksums(output)

    with pytest.raises(PublicDevelopmentPreviewError, match="exact public schema"):
        verify_public_development_preview(output)


def test_reference_and_label_material_is_rejected_before_public_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path, reference_material=True)

    with pytest.raises(PublicDevelopmentPreviewError, match="reference labels"):
        _build(inputs)

    assert not (inputs.output_root / "fixture-preview").exists()


def test_standalone_verifier_rejects_reference_evaluation_with_fresh_checksums(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    corpus_path = output / "corpus.json"
    corpus = read_json(corpus_path)
    corpus["reference_evaluation"] = {"precision": 1.0, "human_label": "accepted"}
    write_json(corpus_path, corpus)
    _refresh_checksums(output)

    with pytest.raises(PublicDevelopmentPreviewError, match="reference evaluation"):
        verify_public_development_preview(output)


def test_standalone_verifier_rejects_usage_algebra_tamper_with_fresh_checksums(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    usage_path = output / "usage.json"
    usage = read_json(usage_path)
    usage["replayed_provider_usage"]["cost_usd_lower_bound"] += 1
    write_json(usage_path, usage)
    _refresh_checksums(output)

    with pytest.raises(PublicDevelopmentPreviewError, match="disagrees|understates"):
        verify_public_development_preview(output)


def test_standalone_verifier_rejects_calls_above_precall_ceiling_with_fresh_checksums(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    manifest_path = output / "publication-manifest.json"
    manifest = read_json(manifest_path)
    usage_path = output / "usage.json"
    usage = read_json(usage_path)
    completed_calls = usage["provider_budget"]["structured_calls_completed"]
    assert completed_calls > 1
    tampered_ceiling = completed_calls - 1
    manifest["prospective_execution"]["budget"]["max_structured_calls"] = tampered_ceiling
    manifest["prospective_execution"]["budget"]["max_transport_attempts"] = tampered_ceiling * 4
    write_json(manifest_path, manifest)
    _refresh_checksums(output)

    with pytest.raises(PublicDevelopmentPreviewError, match="prospective execution ceiling"):
        verify_public_development_preview(output)


def test_standalone_verifier_rejects_attempts_above_precall_ceiling_with_fresh_checksums(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    manifest = read_json(output / "publication-manifest.json")
    attempt_ceiling = manifest["prospective_execution"]["budget"]["max_transport_attempts"]

    corpus_path = output / "corpus.json"
    corpus = read_json(corpus_path)
    summary_path = output / "run-summary.json"
    summary = read_json(summary_path)
    usage_path = output / "usage.json"
    usage = read_json(usage_path)
    extra_attempts = attempt_ceiling + 1 - usage["replayed_provider_usage"]["attempts_lower_bound"]
    corpus_stage = corpus["papers_detail"][0]["extractor"]["usage"]
    usage_stage = usage["paper_accounting"][0]["stages"]["block_candidate_extraction"]["usage"]
    operational_stage = usage["operational_accounting"]["stages"]["block_candidate_extraction"]
    for stage in (corpus_stage, usage_stage, operational_stage):
        stage["attempts_lower_bound"] += extra_attempts
        stage["retries_lower_bound"] += extra_attempts
    usage["replayed_provider_usage"]["attempts_lower_bound"] += extra_attempts
    usage["replayed_provider_usage"]["retries_lower_bound"] += extra_attempts
    summary["provider_usage_recorded"]["attempts_lower_bound"] += extra_attempts
    summary["provider_usage_recorded"]["retries_lower_bound"] += extra_attempts
    write_json(corpus_path, corpus)
    write_json(summary_path, summary)
    write_json(usage_path, usage)
    _refresh_checksums(output)

    with pytest.raises(PublicDevelopmentPreviewError, match="prospective execution ceiling"):
        verify_public_development_preview(output)


@pytest.mark.parametrize(
    "mutation",
    [
        "corpus-execution-partition",
        "corpus-request-api",
        "summary-row-semantics",
        "evidence-lineage-total",
    ],
)
def test_standalone_verifier_rejects_semantic_tamper_with_fresh_checksums(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    if mutation.startswith("corpus-"):
        path = output / "corpus.json"
        payload = read_json(path)
        if mutation == "corpus-execution-partition":
            payload["papers_detail"][0]["extractor"]["execution"]["blocks_total"] += 1
        else:
            payload["papers_detail"][0]["verifier"]["request_contract"]["schema"]["schema_name"] = (
                "unbound_verifier_api"
            )
    elif mutation == "summary-row-semantics":
        path = output / "run-summary.json"
        payload = read_json(path)
        payload["row_enumeration"]["all_rows_accounted_for"] = False
    else:
        path = output / "evidence-map.json"
        payload = read_json(path)
        payload["papers"][0]["candidate_layer"]["model_proposals"] += 1
    write_json(path, payload)
    _refresh_checksums(output)

    with pytest.raises(PublicDevelopmentPreviewError):
        verify_public_development_preview(output)


def test_standalone_verifier_requires_canonical_checksum_rendering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    checksums = output / "SHA256SUMS"
    checksums.write_bytes(checksums.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(PublicDevelopmentPreviewError, match="rendering is not canonical"):
        verify_public_development_preview(output)


def test_standalone_verifier_rejects_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    copied = tmp_path / "copied-preview"
    shutil.copytree(output, copied)
    (copied / "README.md").unlink()
    (copied / "README.md").symlink_to(output / "README.md")

    with pytest.raises(PublicDevelopmentPreviewError, match="non-regular"):
        verify_public_development_preview(copied)


def test_build_rejects_dangling_and_ancestor_output_symlinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    dangling = tmp_path / "dangling-output"
    dangling.symlink_to(tmp_path / "missing-output", target_is_directory=True)
    with pytest.raises(PublicDevelopmentPreviewError, match="symbolic link"):
        _build(inputs, output_root=dangling)

    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(PublicDevelopmentPreviewError, match="symbolic link"):
        _build(inputs, output_root=linked_parent / "nested")


def test_build_preserves_dangling_destination_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    inputs.output_root.mkdir()
    destination = inputs.output_root / "fixture-preview"
    target = tmp_path / "missing-destination-target"
    destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(PublicDevelopmentPreviewError, match="already exists"):
        _build(inputs)

    assert destination.is_symlink()
    assert destination.readlink() == target
    assert not target.exists()


def test_standalone_verifier_rejects_ancestor_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    linked_parent = tmp_path / "linked-public-parent"
    linked_parent.symlink_to(inputs.output_root, target_is_directory=True)

    with pytest.raises(PublicDevelopmentPreviewError, match="symbolic link"):
        verify_public_development_preview(linked_parent / output.name)


def test_exclusive_publish_preserves_destination_created_during_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    original_publish = preview_module._publish_contents_exclusive

    def race_publish(*, parent: Path, bundle_id: str, contents: dict[str, bytes]) -> Path:
        raced = parent / bundle_id
        raced.mkdir()
        (raced / "sentinel.txt").write_text("preserve me\n", encoding="utf-8")
        return original_publish(parent=parent, bundle_id=bundle_id, contents=contents)

    monkeypatch.setattr(preview_module, "_publish_contents_exclusive", race_publish)

    with pytest.raises(PublicDevelopmentPreviewError, match="appeared during build"):
        _build(inputs)

    destination = inputs.output_root / "fixture-preview"
    assert (destination / "sentinel.txt").read_text(encoding="utf-8") == "preserve me\n"
    assert {path.name for path in destination.iterdir()} == {"sentinel.txt"}


def test_standalone_verifier_detects_mutation_after_captured_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output = _build(inputs)
    original_validate = preview_module._validate_standalone_bundle

    def mutate_after_validation(contents: dict[str, bytes]) -> dict[str, object]:
        result = original_validate(contents)
        readme = output / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8") + "\nraced\n", encoding="utf-8")
        return result

    monkeypatch.setattr(preview_module, "_validate_standalone_bundle", mutate_after_validation)

    with pytest.raises(PublicDevelopmentPreviewError, match="changed during verification"):
        verify_public_development_preview(output)


def test_cli_build_and_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    inputs = _fixture(monkeypatch, tmp_path)
    output_root = tmp_path / "cli-public"

    result = RUNNER.invoke(
        app,
        [
            "build-public-development-preview",
            "cli-preview",
            str(inputs.sealed_root),
            str(inputs.corpus_path),
            str(inputs.route_path),
            "--expected-paper-count",
            "1",
            "--output-root",
            str(output_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "public-development-preview-written"

    verify = RUNNER.invoke(
        app,
        ["verify-public-development-preview", str(output_root / "cli-preview")],
    )
    assert verify.exit_code == 0, verify.output
    assert json.loads(verify.output)["status"] == "verified"


def test_cli_failure_is_stable_and_does_not_echo_private_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fixture(monkeypatch, tmp_path, precall_zdr=False)

    result = RUNNER.invoke(
        app,
        [
            "build-public-development-preview",
            "cli-preview",
            str(inputs.sealed_root),
            str(inputs.corpus_path),
            str(inputs.route_path),
            "--expected-paper-count",
            "1",
            "--output-root",
            str(inputs.output_root),
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.output)["status"] == "public-development-preview-not-written"
    assert str(tmp_path) not in result.output
    assert not (inputs.output_root / "cli-preview").exists()
