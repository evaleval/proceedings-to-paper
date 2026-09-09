from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_origin_pipeline import _origin_fixture, _OriginPipelineClient
from test_pipeline import _end_to_end_fixture, _EndToEndClient, _NeverCallClient

from proceedings_to_eee.cli import _run_command
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    write_json,
    write_jsonl,
)
from proceedings_to_eee.pipeline import run_paper
from proceedings_to_eee.providers.budget import ProviderBudgetExhausted
from proceedings_to_eee.providers.openrouter import (
    ProviderResponseValidationError,
    StructuredResponse,
)
from proceedings_to_eee.resolution.tuple_resolution import (
    TupleResolutionInput,
    TupleWireProposal,
    verify_tuple_resolution_proposal,
)
from proceedings_to_eee.reviewed_export.models import ReviewItem, model_payload
from proceedings_to_eee.reviewed_export.workflow import (
    REVIEW_ITEMS_NAME,
    REVIEW_MANIFEST_NAME,
    ReviewedExportError,
    ReviewedExportErrorCode,
    prepare_export_review,
    validate_export_review,
)
from proceedings_to_eee.run_seal import seal_run_tree


class _MismatchTupleClient(_OriginPipelineClient):
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] != "candidate_tuple_resolution":
            return response
        payload = json.loads(json.dumps(response.payload))
        payload["evaluated_system"] = {"raw_name": "System"}
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


class _TupleBudgetStopClient(_OriginPipelineClient):
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        if kwargs["schema_name"] == "candidate_tuple_resolution":
            self.requests.append(kwargs)
            raise ProviderBudgetExhausted(
                reason="structured_call_limit",
                summary={"status": "exhausted"},
            )
        return super().structured_chat(**kwargs)


class _VerifierFailureClient(_OriginPipelineClient):
    def __init__(self, failure: str) -> None:
        super().__init__("positive")
        self.failure = failure

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        if kwargs["schema_name"] != "candidate_evidence_verification_v2":
            return super().structured_chat(**kwargs)
        if self.failure == "transport":
            self.requests.append(kwargs)
            raise RuntimeError("fixture transport failure")
        response = super().structured_chat(**kwargs)
        if self.failure == "returned_model":
            return replace(
                response,
                call=response.call.model_copy(update={"model_returned": "fixture/wrong-model"}),
            )
        raise ProviderResponseValidationError(call=response.call, code="wire_validation")


class _TupleReturnedModelMismatchClient(_OriginPipelineClient):
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] != "candidate_tuple_resolution":
            return response
        return replace(
            response,
            call=response.call.model_copy(update={"model_returned": "fixture/wrong-model"}),
        )


def test_tuple_stage_is_parameter_strict_private_and_demotion_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    client = _OriginPipelineClient("positive")

    result = run_paper(spec=spec, settings=settings, client=client)

    tuple_request = next(
        item for item in client.requests if item["schema_name"] == "candidate_tuple_resolution"
    )
    assert tuple_request["model"] == settings.tuple_model
    assert tuple_request["max_tokens"] == settings.tuple_max_tokens
    assert tuple_request["require_parameters"] is True
    assert result["tuple_resolution"]["request_contract"]["routing"]["require_parameters"] is True
    assert result["tuple_resolution"]["execution"]["candidates_passed"] == 1
    checkpoint = read_json(
        settings.output_root / spec.paper_id / "private" / "tuple-resolution-checkpoint.json"
    )
    entry = next(iter(checkpoint["candidates"].values()))
    assert entry["provider_call"]["require_parameters"] is True
    assert entry["assessment"]["allows_origin_or_export"] is False

    private_text = (
        settings.output_root / spec.paper_id / "private" / "tuple-resolution.json"
    ).read_text()
    public_text = (settings.output_root / spec.paper_id / "run.json").read_text()
    assert "exact_excerpt" in private_text
    assert "accepted_tuple" in private_text
    for forbidden in ("exact_excerpt", "accepted_tuple", "tuple_ev_"):
        assert forbidden not in public_text
    assert '"request_id":' not in public_text
    lineage = read_json(settings.output_root / spec.paper_id / "candidate-lineage.json")
    assert lineage["export_provenance_mode"] == "tuple_gated_production"
    assert lineage["tuple_sidecar_sha256"] == result["tuple_resolution"]["sidecar"]["sha256"]
    assert lineage["verifier_gate_required"] is True
    assert lineage["verifier_sidecar_sha256"] == result["verifier"]["sidecar"]["sha256"]
    assert lineage["candidates"][0]["tuple_gate_sha256"] == next(
        iter(
            read_json(settings.output_root / spec.paper_id / "private" / "tuple-resolution.json")[
                "candidate_gates"
            ].values()
        )
    )


def test_tuple_only_pipeline_is_review_only_and_never_claims_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    settings = replace(settings, verifier_model=None, origin_model=None)
    client = _OriginPipelineClient("positive")

    result = run_paper(spec=spec, settings=settings, client=client)

    schema_names = [item["schema_name"] for item in client.requests]
    assert len(schema_names) == 2
    assert schema_names[-1] == "candidate_tuple_resolution"
    assert "candidate_evidence_verification_v2" not in schema_names
    assert "producer_origin_proposal" not in schema_names
    assert result["counts"]["eee_records"] == 0
    lineage = read_json(settings.output_root / spec.paper_id / "candidate-lineage.json")
    assert lineage["export_provenance_mode"] == "tuple_gated_unverified"
    assert lineage["tuple_sidecar_sha256"] == result["tuple_resolution"]["sidecar"]["sha256"]
    assert lineage["verifier_gate_required"] is False
    assert lineage.get("verifier_sidecar_sha256") is None
    assert all(item["export_status"] != "exported" for item in lineage["candidates"])


@pytest.mark.parametrize("gate_field", ["source_tuple_gate_sha256", "source_verifier_gate_sha256"])
def test_review_packet_binds_stage_sidecars_and_rejects_rehashed_item_gate_splice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gate_field: str,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))
    sealed = tmp_path / "sealed-paper"
    seal_run_tree(settings.output_root / spec.paper_id, sealed)
    review = tmp_path / "review"

    manifest = prepare_export_review(run_root=sealed, output_root=review)

    assert manifest.papers[0].tuple_resolution is not None
    assert manifest.papers[0].verifier_gates is not None
    items_path = review / REVIEW_ITEMS_NAME
    items = [
        ReviewItem.model_validate(json.loads(line)) for line in items_path.read_text().splitlines()
    ]
    assert items[0].source_export_mode.value == "tuple_gated_production"
    assert items[0].source_tuple_sidecar_sha256 == manifest.papers[0].tuple_resolution.sha256
    assert items[0].source_verifier_sidecar_sha256 == manifest.papers[0].verifier_gates.sha256
    assert items[0].source_verifier_gate_passed is True

    payload = model_payload(items[0])
    payload[gate_field] = "f" * 64
    new_items_sha = write_jsonl(items_path, [payload])
    manifest_payload = model_payload(manifest)
    manifest_payload["items"]["sha256"] = new_items_sha
    manifest_payload["items"]["size_bytes"] = items_path.stat().st_size
    write_json(review / REVIEW_MANIFEST_NAME, manifest_payload)

    with pytest.raises(ReviewedExportError) as error:
        validate_export_review(review)
    assert error.value.code is ReviewedExportErrorCode.CANDIDATE_BINDING_INVALID


def test_tuple_checkpoint_resumes_without_any_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))

    resumed = run_paper(spec=spec, settings=settings, client=_NeverCallClient())

    assert resumed["counts"]["tuple_resumed"] == 1
    assert resumed["tuple_resolution"]["execution"]["candidates_resumed"] == 1
    assert resumed["counts"]["verifier_resumed"] == 1
    assert resumed["counts"]["origin_resumed"] == 1


def test_tuple_returned_model_mismatch_is_paid_failure_telemetry_not_route_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    first = run_paper(
        spec=spec,
        settings=settings,
        client=_TupleReturnedModelMismatchClient("positive"),
    )

    checkpoint = read_json(
        settings.output_root / spec.paper_id / "private" / "tuple-resolution-checkpoint.json"
    )
    entry = next(iter(checkpoint["candidates"].values()))
    assert entry["status"] == "response_failure"
    assert entry["error_code"] == "provider_response_returned_model_mismatch"
    assert entry["provider_call"]["model_returned"] == "fixture/wrong-model"
    assert first["counts"]["tuple_passed"] == 0
    assert first["tuple_resolution"]["completed_call_telemetry"]["calls"] == 1

    resumed = run_paper(spec=spec, settings=settings, client=_NeverCallClient())
    assert resumed["counts"]["tuple_resumed"] == 1
    assert resumed["counts"]["tuple_passed"] == 0


def test_rehashed_tuple_checkpoint_returned_model_mutation_is_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))
    path = settings.output_root / spec.paper_id / "private" / "tuple-resolution-checkpoint.json"
    checkpoint = read_json(path)
    entry = next(iter(checkpoint["candidates"].values()))
    entry["provider_call"]["model_returned"] = "fixture/wrong-model"
    unsigned = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    write_json(path, checkpoint)

    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=settings, client=client)

    assert "candidate_tuple_resolution" in {request["schema_name"] for request in client.requests}
    assert result["counts"]["tuple_resumed"] == 0


def test_fully_rehashed_tuple_proposal_splice_fails_response_cross_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))
    path = settings.output_root / spec.paper_id / "private" / "tuple-resolution-checkpoint.json"
    checkpoint = read_json(path)
    entry = next(iter(checkpoint["candidates"].values()))
    entry["proposal"]["summary"] = "A different but schema-valid spliced proposal."
    request = TupleResolutionInput.model_validate(entry["input"])
    proposal = TupleWireProposal.model_validate(entry["proposal"])
    assessment = verify_tuple_resolution_proposal(request=request, proposal=proposal)
    entry["assessment"] = assessment.model_dump(mode="json")
    unsigned = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    write_json(path, checkpoint)

    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=settings, client=client)

    assert [item["schema_name"] for item in client.requests] == ["candidate_tuple_resolution"]
    assert result["counts"]["tuple_resumed"] == 0
    assert result["counts"]["verifier_resumed"] == 1


def test_changed_tuple_settings_invalidate_all_downstream_stage_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))
    changed = replace(settings, tuple_max_tokens=settings.tuple_max_tokens + 1)

    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=changed, client=client)

    assert [item["schema_name"] for item in client.requests] == [
        "candidate_tuple_resolution",
        "candidate_evidence_verification_v2",
        "candidate_producer_origin_selection",
    ]
    assert result["counts"]["tuple_resumed"] == 0
    assert result["counts"]["verifier_resumed"] == 0
    assert result["counts"]["origin_resumed"] == 0


def test_locally_valid_but_candidate_mismatched_tuple_blocks_verifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    client = _MismatchTupleClient("positive")

    result = run_paper(spec=spec, settings=settings, client=client)

    assert [item["schema_name"] for item in client.requests] == [
        "paper_evaluation_candidates",
        "candidate_tuple_resolution",
    ]
    assert result["counts"]["tuple_passed"] == 0
    assert result["counts"]["verifications"] == 0
    assert result["tuple_resolution"]["execution"]["candidates_mismatched"] == 1
    observation = json.loads(
        (settings.output_root / spec.paper_id / "observations.jsonl").read_text()
    )
    assert observation["export_status"] == "needs_review"
    assert observation["export_reason"] == "tuple_resolution=candidate_mismatch"


def test_missing_physical_cell_is_unsupported_without_tuple_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    settings = replace(settings, tuple_model="fixture/tuple")
    client = _EndToEndClient()

    result = run_paper(spec=spec, settings=settings, client=client)

    assert [item["schema_name"] for item in client.requests] == ["paper_evaluation_candidates"]
    assert result["counts"]["tuple_unsupported"] == 1
    assert result["counts"]["tuple_passed"] == 0
    checkpoint = read_json(
        settings.output_root / spec.paper_id / "private" / "tuple-resolution-checkpoint.json"
    )
    entry = next(iter(checkpoint["candidates"].values()))
    assert entry["status"] == "unsupported"
    assert entry["error_code"] == "missing_physical_cell_evidence"
    assert entry["provider_call"] is None


def test_budget_stop_before_tuple_call_keeps_incomplete_checkpoint_for_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)

    with pytest.raises(ProviderBudgetExhausted):
        run_paper(spec=spec, settings=settings, client=_TupleBudgetStopClient("positive"))

    checkpoint = read_json(
        settings.output_root / spec.paper_id / "private" / "tuple-resolution-checkpoint.json"
    )
    assert checkpoint["candidates"] == {}
    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=settings, client=client)
    assert [item["schema_name"] for item in client.requests] == [
        "candidate_tuple_resolution",
        "candidate_evidence_verification_v2",
        "candidate_producer_origin_selection",
    ]
    assert result["extractor"]["execution"]["blocks_resumed"] == 1


def test_changed_verifier_contract_invalidates_origin_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))
    changed = replace(settings, verifier_model="fixture/verifier-v2")

    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=changed, client=client)

    assert [item["schema_name"] for item in client.requests] == [
        "candidate_evidence_verification_v2",
        "candidate_producer_origin_selection",
    ]
    assert result["counts"]["tuple_resumed"] == 1
    assert result["counts"]["verifier_resumed"] == 0
    assert result["counts"]["origin_resumed"] == 0


@pytest.mark.parametrize(
    ("failure", "status", "error_code", "completed_calls"),
    [
        ("response", "response_failure", "provider_response_wire_validation", 1),
        (
            "returned_model",
            "response_failure",
            "provider_response_returned_model_mismatch",
            1,
        ),
    ],
)
def test_verifier_response_failure_is_checkpointed_and_resumed_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    status: str,
    error_code: str,
    completed_calls: int,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)

    first = run_paper(
        spec=spec,
        settings=settings,
        client=_VerifierFailureClient(failure),
    )

    assert first["status"] == "partial_failure"
    assert first["review_state"]["status"] == "needs_review"
    assert first["counts"]["verifications"] == 0
    assert first["counts"]["verifier_failed"] == 1
    assert first["counts"]["origin_candidates"] == 0
    assert first["verifier"]["successful_call_telemetry"]["calls"] == 0
    assert first["verifier"]["completed_call_telemetry"]["calls"] == completed_calls
    checkpoint = read_json(
        settings.output_root / spec.paper_id / "private" / "verifier-checkpoint.json"
    )
    entry = next(iter(checkpoint["candidates"].values()))
    assert entry["status"] == status
    assert entry["error_code"] == error_code

    resumed = run_paper(spec=spec, settings=settings, client=_NeverCallClient())

    assert resumed["status"] == "partial_failure"
    assert resumed["counts"]["verifier_failed"] == 1
    assert resumed["counts"]["verifier_resumed"] == 1
    assert resumed["verifier"]["completed_call_telemetry"]["calls"] == completed_calls


def test_reproduction_command_preserves_tuple_model_and_tokens(tmp_path: Path) -> None:
    command = _run_command(
        corpus_path=tmp_path / "corpus.yaml",
        model="fixture/extractor",
        schema_path=tmp_path / "schema.json",
        schema_sha256="a" * 64,
        tuple_model="fixture/tuple",
        tuple_max_tokens=3456,
        verifier_model="fixture/verifier",
        verifier_max_tokens=2000,
        origin_model=None,
        origin_max_tokens=2500,
        output=tmp_path / "runs",
        min_confidence=0.8,
        row_enumeration=False,
        row_model=None,
        row_estimated_call_cost_usd=None,
        max_structured_calls=10,
        max_provider_cost_usd=5.0,
        provider_call_cost_reservation_usd=0.1,
        paper_id=None,
    )

    assert "--tuple-model fixture/tuple" in command
    assert "--tuple-max-tokens 3456" in command
