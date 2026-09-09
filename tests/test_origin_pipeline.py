from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest
from test_pipeline import (
    _end_to_end_fixture,
    _NeverCallClient,
    _VerifierAcceptClient,
)

from proceedings_to_eee.domain.attribution import AttributionState
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.io import canonical_json_bytes, read_json, sha256_bytes, write_json
from proceedings_to_eee.pipeline import run_paper
from proceedings_to_eee.providers.budget import ProviderBudgetExhausted
from proceedings_to_eee.providers.openrouter import (
    ProviderRequestRejectedError,
    ProviderResponseValidationError,
    StructuredResponse,
)
from proceedings_to_eee.resolution.origin_retrieval import layout_binding_sha256
from proceedings_to_eee.resolution.tuple_resolution import (
    TupleResolutionInput,
    TupleResultEvidence,
    tuple_candidate_binding_sha256,
)

METHODS_POSITIVE = """2 Experimental Setup
We evaluate System A on Dataset A using AUC.
Every run uses the same fixed seed.
"""

METHODS_GENERIC = """2 Experimental Setup
We evaluate all candidate systems on Dataset A using AUC.
Every run uses the same fixed seed.
"""

METHODS_EXTERNAL = """2 Experimental Setup
Scores for System A were taken from the public leaderboard for Dataset A AUC.
We preserve those published values without rerunning the systems.
"""

RESULTS = (
    "3 Results\n"
    "Table 1: AUC proportion on Dataset A test split\n"
    "System                    AUC\n"
    "System A                 0.80\n"
    "System B                 0.70\n"
    "System A split test Dataset A AUC proportion 0.80\n"
    "System A Dataset A AUC proportion 0.80 operationalization Private source evidence "
    "phrase describing the fixture operationalization.\n"
    "System A Dataset A AUC proportion 0.80 operationalization We evaluate System A on "
    "Dataset A using AUC.\n"
)


def _page(text: str, page: int) -> PageFragment:
    return PageFragment(
        fragment_id=f"page-{page}",
        source_id="src_end_to_end",
        page=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_count=len(text),
        numeric_token_count=2,
        result_signal_score=10.0,
    )


def _origin_fixture(monkeypatch, tmp_path, *, methods: str = METHODS_POSITIVE):
    spec, settings = _end_to_end_fixture(monkeypatch, tmp_path)
    layout = PdfLayout(
        source_id="src_end_to_end",
        parser="fixture",
        parser_version="fixture/1",
        page_count=2,
        pages=[_page(methods, 1), _page(RESULTS, 2)],
    )
    monkeypatch.setattr(
        "proceedings_to_eee.pipeline.extract_pdf_layout",
        lambda path, source: layout,
    )

    def bound_tuple_input(candidate, frozen_layout):
        anchor = candidate.evidence[0]
        page = next(page for page in frozen_layout.pages if page.page == anchor.page)
        start = page.text.index(anchor.quote)
        evidence_payload = {
            "schema_version": "tuple-result-evidence/0.1",
            "source_id": anchor.source_id,
            "page": anchor.page,
            "page_text_sha256": page.text_sha256,
            "kind": anchor.kind,
            "label": anchor.label,
            "row": anchor.row,
            "column": anchor.column,
            "region_id": "fixture-region",
            "planned_row_id": "fixture-row",
            "cell_id": "fixture-cell",
            "numeric_token_id": "fixture-token",
            "header_ids": [],
            "exact_excerpt": anchor.quote,
            "excerpt_sha256": anchor.quote_sha256,
            "char_start": start,
            "char_end": start + len(anchor.quote),
        }
        evidence_payload["evidence_id"] = (
            "tuple_ev_" + sha256_bytes(canonical_json_bytes(evidence_payload))[:24]
        )
        evidence = [TupleResultEvidence.model_validate(evidence_payload)]
        input_payload = {
            "schema_version": "tuple-resolution-input/0.1",
            "paper_id": candidate.paper_id,
            "candidate_binding_sha256": tuple_candidate_binding_sha256(candidate),
            "layout_binding_sha256": layout_binding_sha256(frozen_layout),
            "result_evidence_binding_sha256": sha256_bytes(canonical_json_bytes(evidence)),
            "candidate_tuple_fields": "withheld_untrusted",
            "result_evidence": evidence,
        }
        input_payload["input_sha256"] = sha256_bytes(canonical_json_bytes(input_payload))
        return TupleResolutionInput.model_validate(input_payload)

    monkeypatch.setattr(
        "proceedings_to_eee.pipeline.build_tuple_resolution_input",
        bound_tuple_input,
    )
    spec = spec.model_copy(update={"include_pages": [2], "reference_path": None})
    settings = replace(
        settings,
        tuple_model="fixture/tuple",
        verifier_model="fixture/verifier",
        origin_model="fixture/origin",
    )
    return spec, settings


def _origin_input(user: str) -> dict[str, Any]:
    raw = user.split("<ORIGIN_INPUT>\n", 1)[1].split("\n</ORIGIN_INPUT>", 1)[0]
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return parsed


def _tuple_input(user: str) -> dict[str, Any]:
    raw = user.split("<TUPLE_INPUT>\n", 1)[1].split("\n</TUPLE_INPUT>", 1)[0]
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return parsed


class _OriginPipelineClient(_VerifierAcceptClient):
    def __init__(self, mode: str = "positive") -> None:
        super().__init__()
        self.mode = mode

    @staticmethod
    def _wire_prompt_sha256(kwargs: dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": kwargs["system"]},
            {"role": "user", "content": kwargs["user"]},
        ]
        raw = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        if kwargs["schema_name"] == "candidate_producer_origin_selection" and (
            self.mode == "budget"
        ):
            self.requests.append(kwargs)
            raise ProviderBudgetExhausted(
                reason="structured_call_limit",
                summary={"status": "exhausted"},
            )
        response = super().structured_chat(**kwargs)
        response = replace(
            response,
            call=response.call.model_copy(
                update={"prompt_sha256": self._wire_prompt_sha256(kwargs)}
            ),
        )
        if kwargs["schema_name"] == "paper_evaluation_candidates":
            payload = json.loads(json.dumps(response.payload))
            observation = payload["observations"][0]
            observation["scope"]["split"] = "test"
            observation["evidence"][0]["row"] = "System A Dataset A test"
            observation["evidence"][0]["column"] = "AUC ↑ proportion"
            observation["evidence"][0]["quote"] = "System A                 0.80"
            observation["evidence"][1]["quote"] = "Table 1: AUC proportion on Dataset A test split"
            observation["evidence"][1]["column"] = "AUC ↑ proportion"
            observation["evidence"][2]["quote"] = (
                "System A split test Dataset A AUC proportion 0.80"
            )
            response_sha256 = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            return replace(
                response,
                payload=payload,
                call=response.call.model_copy(update={"response_sha256": response_sha256}),
            )
        if kwargs["schema_name"] == "candidate_tuple_resolution":
            tuple_input = _tuple_input(kwargs["user"])
            evidence_id = tuple_input["result_evidence"][0]["evidence_id"]
            payload = {
                "candidate_binding_sha256": tuple_input["candidate_binding_sha256"],
                "result_evidence_binding_sha256": tuple_input["result_evidence_binding_sha256"],
                "result_evidence_id": evidence_id,
                "evaluated_system": {"raw_name": "System A"},
                "system_version": None,
                "dataset": {"dataset_raw": "Dataset A"},
                "dataset_version": None,
                "metric": {"raw_name": "AUC"},
                "direction": {"raw_direction": "↑", "lower_is_better": False},
                "scale": {"raw_scale": "proportion", "min_score": 0, "max_score": 1},
                "value": {"raw": "0.80", "numeric": 0.8, "comparator": "exact"},
                "uncertainty": None,
                "unit": "proportion",
                "setting": None,
                "scope": {
                    "split": "test",
                    "subset": None,
                    "group": None,
                    "language": None,
                    "sample_count": None,
                    "aggregation": None,
                    "raw_scope": None,
                },
                "field_evidence": {
                    "system_evidence_id": evidence_id,
                    "system_version_evidence_id": None,
                    "dataset_evidence_id": evidence_id,
                    "dataset_version_evidence_id": None,
                    "metric_evidence_id": evidence_id,
                    "direction_evidence_id": evidence_id,
                    "scale_evidence_id": evidence_id,
                    "value_evidence_id": evidence_id,
                    "uncertainty_evidence_id": evidence_id,
                    "unit_evidence_id": evidence_id,
                    "setting_evidence_id": None,
                    "scope_evidence_id": evidence_id,
                },
                "unresolved_fields": [
                    "dataset_version",
                    "setting",
                    "system_version",
                ],
                "not_applicable_fields": ["uncertainty"],
                "summary": "Required fixture tuple fields are printed in the bound row.",
            }
            response_sha256 = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            return replace(
                response,
                payload=payload,
                call=response.call.model_copy(update={"response_sha256": response_sha256}),
            )
        if kwargs["schema_name"] != "candidate_producer_origin_selection":
            return response
        if self.mode == "invalid":
            raise ProviderResponseValidationError(call=response.call, code="wire_validation")

        origin_input = _origin_input(kwargs["user"])
        hits = origin_input["context_hits"]
        result_hit = next(hit for hit in hits if "result_anchor" in hit["matched_dimensions"])
        origin_hit = next(hit for hit in hits if hit["page"] == 1)
        if self.mode == "no_signal":
            state = AttributionState.NO_SIGNAL.value
            relation = "none"
            origin_context_hit_id = None
        elif self.mode == "unresolved":
            state = AttributionState.UNRESOLVED.value
            relation = "weak"
            origin_context_hit_id = origin_hit["context_hit_id"]
        elif self.mode == "external":
            state = AttributionState.EXTERNALLY_SOURCED.value
            relation = "direct"
            origin_context_hit_id = origin_hit["context_hit_id"]
        else:
            state = AttributionState.PAPER_PRODUCED.value
            relation = "direct"
            origin_context_hit_id = origin_hit["context_hit_id"]
        payload = {
            "candidate_binding_sha256": origin_input["candidate_binding_sha256"],
            "evaluated_system": "System A",
            "proposed_state": state,
            "evidence_relation": relation,
            "result_context_hit_id": result_hit["context_hit_id"],
            "origin_context_hit_id": origin_context_hit_id,
            "summary": "Bound fixture origin proposal.",
        }
        response_sha256 = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        return replace(
            response,
            payload=payload,
            call=response.call.model_copy(update={"response_sha256": response_sha256}),
        )


class _NoCallStageFailureClient(_OriginPipelineClient):
    """Fail one selected downstream stage without a completed provider response."""

    _SCHEMA_BY_STAGE = {
        "tuple": "candidate_tuple_resolution",
        "verifier": "candidate_evidence_verification_v2",
        "origin": "candidate_producer_origin_selection",
    }

    def __init__(self, *, stage: str, failure: str) -> None:
        super().__init__("positive")
        self.stage = stage
        self.failure = failure

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        if kwargs["schema_name"] != self._SCHEMA_BY_STAGE[self.stage]:
            return super().structured_chat(**kwargs)
        self.requests.append(kwargs)
        if self.failure == "request_rejected":
            raise ProviderRequestRejectedError(status_code=400)
        raise RuntimeError("fixture transport failure")


class _VerifierUngroundedAcceptClient(_OriginPipelineClient):
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        if kwargs["schema_name"] != "candidate_evidence_verification_v2":
            return response
        payload = json.loads(json.dumps(response.payload))
        payload["scope_evidence_line_ids"] = payload["role_evidence_line_ids"]
        response_sha256 = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        return replace(
            response,
            payload=payload,
            call=response.call.model_copy(update={"response_sha256": response_sha256}),
        )


def _observation(settings, spec) -> dict[str, Any]:
    path = settings.output_root / spec.paper_id / "observations.jsonl"
    return json.loads(path.read_text(encoding="utf-8"))


def test_off_page_positive_runs_after_verifier_and_stays_review_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    client = _OriginPipelineClient("positive")

    result = run_paper(spec=spec, settings=settings, client=client)

    assert [request["schema_name"] for request in client.requests] == [
        "paper_evaluation_candidates",
        "candidate_tuple_resolution",
        "candidate_evidence_verification_v2",
        "candidate_producer_origin_selection",
    ]
    assert result["counts"]["verifier_accepts"] == 1
    assert result["counts"]["origin_positive_review_only"] == 1
    assert result["counts"]["exported"] == 0
    assert result["counts"]["eee_records"] == 0
    assert _observation(settings, spec)["attribution"]["state"] == "unresolved"
    assert _observation(settings, spec)["export_status"] == "needs_review"
    assert result["origin_retrieval"]["automatic_paper_produced_forbidden"] is True

    private = settings.output_root / spec.paper_id / "private"
    checkpoint = read_json(private / "origin-retrieval-checkpoint.json")
    entry = next(iter(checkpoint["candidates"].values()))
    assert entry["proposal"]["result_anchor"]["page"] == 2
    assert entry["proposal"]["origin_anchor"]["page"] == 1
    assert "exact_excerpt" in json.dumps(checkpoint)
    run_text = (settings.output_root / spec.paper_id / "run.json").read_text()
    assert "exact_excerpt" not in run_text


def test_verified_external_origin_demotes_without_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path, methods=METHODS_EXTERNAL)

    result = run_paper(
        spec=spec,
        settings=settings,
        client=_OriginPipelineClient("external"),
    )

    assert result["counts"]["origin_external"] == 1
    assert result["counts"]["exported"] == 0
    observation = _observation(settings, spec)
    assert observation["attribution"]["state"] == "externally_sourced"
    assert observation["export_reason"] == "origin_retrieval=verified_external"


@pytest.mark.parametrize(
    ("mode", "methods", "state"),
    [
        ("no_signal", METHODS_POSITIVE, "no_signal"),
        ("unresolved", METHODS_GENERIC, "unresolved"),
    ],
)
def test_no_signal_and_unresolved_remain_distinct_review_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    mode: str,
    methods: str,
    state: str,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path, methods=methods)

    result = run_paper(
        spec=spec,
        settings=settings,
        client=_OriginPipelineClient(mode),
    )

    assert _observation(settings, spec)["attribution"]["state"] == state
    assert result["counts"][f"origin_{state}"] == 1
    assert result["counts"]["eee_records"] == 0


def test_origin_provider_failure_becomes_typed_candidate_review_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)

    result = run_paper(
        spec=spec,
        settings=settings,
        client=_OriginPipelineClient("invalid"),
    )

    assert result["status"] == "partial_failure"
    assert result["counts"]["origin_failed"] == 1
    assert result["origin_retrieval"]["execution"]["candidates_failed"] == 1
    assert _observation(settings, spec)["export_reason"] == (
        "origin_retrieval=provider_response_wire_validation"
    )
    assert "origin_retrieval_failed=provider_response_wire_validation" in result["warnings"]
    summary = read_json(settings.output_root / spec.paper_id / "private" / "origin-retrieval.json")
    assert summary["outcomes"][0]["status"] == "response_failure"
    assert summary["outcomes"][0]["completed_provider_call"] is True
    checkpoint = read_json(
        settings.output_root / spec.paper_id / "private" / "origin-retrieval-checkpoint.json"
    )
    entry = next(iter(checkpoint["candidates"].values()))
    assert entry["status"] == "response_failure"
    assert entry["error_code"] == "provider_response_wire_validation"

    resumed = run_paper(spec=spec, settings=settings, client=_NeverCallClient())

    assert resumed["status"] == "partial_failure"
    assert resumed["counts"]["origin_failed"] == 1
    assert resumed["counts"]["origin_resumed"] == 1
    assert resumed["origin_retrieval"]["completed_call_telemetry"]["calls"] == 1


@pytest.mark.parametrize(
    ("stage", "checkpoint_name", "count_name", "expected_retry_schemas"),
    [
        (
            "tuple",
            "tuple-resolution-checkpoint.json",
            "tuple_resumed",
            [
                "candidate_tuple_resolution",
                "candidate_evidence_verification_v2",
                "candidate_producer_origin_selection",
            ],
        ),
        (
            "verifier",
            "verifier-checkpoint.json",
            "verifier_resumed",
            [
                "candidate_evidence_verification_v2",
                "candidate_producer_origin_selection",
            ],
        ),
        (
            "origin",
            "origin-retrieval-checkpoint.json",
            "origin_resumed",
            ["candidate_producer_origin_selection"],
        ),
    ],
)
@pytest.mark.parametrize(
    ("failure", "error_code"),
    [
        ("transport", "provider_transport_failed"),
        ("request_rejected", "provider_request_rejected"),
    ],
)
def test_downstream_no_call_failures_are_typed_then_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    stage: str,
    checkpoint_name: str,
    count_name: str,
    expected_retry_schemas: list[str],
    failure: str,
    error_code: str,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    first = run_paper(
        spec=spec,
        settings=settings,
        client=_NoCallStageFailureClient(stage=stage, failure=failure),
    )

    assert first["status"] == "partial_failure"
    checkpoint = read_json(settings.output_root / spec.paper_id / "private" / checkpoint_name)
    entry = next(iter(checkpoint["candidates"].values()))
    assert entry["status"] == "provider_failure"
    assert entry["provider_call"] is None
    assert entry["error_code"] == error_code

    resumed_client = _OriginPipelineClient("positive")
    resumed = run_paper(spec=spec, settings=settings, client=resumed_client)

    assert [request["schema_name"] for request in resumed_client.requests] == (
        expected_retry_schemas
    )
    assert resumed["counts"][count_name] == 0
    assert resumed["status"] == "success"


def test_compatible_resume_reuses_extractor_verifier_and_origin_without_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    first = run_paper(
        spec=spec,
        settings=settings,
        client=_OriginPipelineClient("positive"),
    )
    assert first["counts"]["origin_resumed"] == 0
    checkpoint = read_json(
        settings.output_root / spec.paper_id / "private" / "origin-retrieval-checkpoint.json"
    )
    provider_call = next(iter(checkpoint["candidates"].values()))["provider_call"]
    assert provider_call["temperature"] is None
    assert provider_call["reasoning_effort"] == "minimal"
    assert provider_call["seed"] is None
    assert provider_call["reasoning_tokens"] is None
    assert provider_call["require_parameters"] is True

    second = run_paper(
        spec=spec,
        settings=settings,
        client=_NeverCallClient(),
    )

    assert second["extractor"]["execution"]["blocks_resumed"] == 1
    assert second["counts"]["verifier_resumed"] == 1
    assert second["counts"]["origin_resumed"] == 1
    assert second["verifier"]["execution"]["candidates_executed"] == 0
    assert second["origin_retrieval"]["execution"]["candidates_resumed"] == 1


def test_verifier_unbound_candidate_is_counted_in_selection_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "proceedings_to_eee.pipeline.bind_candidate_block",
        lambda candidate, blocks: None,
    )

    result = run_paper(
        spec=spec,
        settings=settings,
        client=_OriginPipelineClient("positive"),
    )

    execution = result["verifier"]["execution"]
    assert execution == {
        "candidates_selected": 1,
        "candidates_unbound": 1,
        "candidates_verified": 0,
        "candidates_failed": 0,
        "candidates_resumed": 0,
        "candidates_resumed_succeeded": 0,
        "candidates_resumed_failed": 0,
        "candidates_executed": 0,
        "candidates_executed_succeeded": 0,
        "candidates_executed_failed": 0,
    }
    assert execution["candidates_selected"] == (
        execution["candidates_unbound"]
        + execution["candidates_resumed"]
        + execution["candidates_executed"]
    )
    assert result["verifier"]["calls"] == []
    assert _observation(settings, spec)["export_reason"] == (
        "no frozen result block contains the evidence quote"
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("model_returned", "fixture/wrong-model"),
        ("temperature", 0.0),
        ("reasoning_effort", None),
        ("seed", 7),
        ("require_parameters", False),
    ],
)
def test_rehashed_origin_call_with_mismatched_request_controls_is_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    field: str,
    invalid_value: object,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))
    path = settings.output_root / spec.paper_id / "private" / "origin-retrieval-checkpoint.json"
    checkpoint = read_json(path)
    entry = next(iter(checkpoint["candidates"].values()))
    entry["provider_call"][field] = invalid_value
    unsigned = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    write_json(path, checkpoint)

    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=settings, client=client)

    assert [request["schema_name"] for request in client.requests] == [
        "candidate_producer_origin_selection"
    ]
    assert result["counts"]["verifier_resumed"] == 1
    assert result["counts"]["origin_resumed"] == 0


def test_tampered_origin_checkpoint_is_rejected_and_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))
    path = settings.output_root / spec.paper_id / "private" / "origin-retrieval-checkpoint.json"
    checkpoint = read_json(path)
    entry = next(iter(checkpoint["candidates"].values()))
    entry["proposal"]["summary"] = "tampered"
    write_json(path, checkpoint)

    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=settings, client=client)

    assert [request["schema_name"] for request in client.requests] == [
        "candidate_producer_origin_selection"
    ]
    assert result["counts"]["verifier_resumed"] == 1
    assert result["counts"]["origin_resumed"] == 0


def test_tampered_verifier_checkpoint_is_rejected_and_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))
    path = settings.output_root / spec.paper_id / "private" / "verifier-checkpoint.json"
    checkpoint = read_json(path)
    entry = next(iter(checkpoint["candidates"].values()))
    entry["verification"]["provider_assessment"]["justification"] = "tampered"
    write_json(path, checkpoint)

    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=settings, client=client)

    assert [request["schema_name"] for request in client.requests] == [
        "candidate_evidence_verification_v2"
    ]
    assert result["counts"]["verifier_resumed"] == 0
    assert result["counts"]["origin_resumed"] == 1


def test_rehashed_verifier_checkpoint_returned_model_mutation_is_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))
    path = settings.output_root / spec.paper_id / "private" / "verifier-checkpoint.json"
    checkpoint = read_json(path)
    entry = next(iter(checkpoint["candidates"].values()))
    entry["provider_call"]["model_returned"] = "fixture/wrong-model"
    unsigned = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    write_json(path, checkpoint)

    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=settings, client=client)

    assert "candidate_evidence_verification_v2" in {
        request["schema_name"] for request in client.requests
    }
    assert result["counts"]["verifier_resumed"] == 0


def test_fully_rehashed_local_grounding_splice_is_rejected_contextually(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    first = run_paper(
        spec=spec,
        settings=settings,
        client=_VerifierUngroundedAcceptClient("positive"),
    )
    assert first["counts"]["verifier_reviews"] == 1
    assert first["counts"]["origin_candidates"] == 0

    path = settings.output_root / spec.paper_id / "private" / "verifier-checkpoint.json"
    checkpoint = read_json(path)
    entry = next(iter(checkpoint["candidates"].values()))
    grounding = entry["verification"]["grounding"]
    for name in ("support", "role", "scope", "value", "metric"):
        grounding[name]["status"] = "grounded"
        grounding[name]["failure_codes"] = []
    grounding["passed"] = True
    grounding["failure_codes"] = []
    entry["verification"]["effective_decision"] = "accept"
    unsigned = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    write_json(path, checkpoint)

    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=settings, client=client)

    assert "candidate_evidence_verification_v2" in {
        request["schema_name"] for request in client.requests
    }
    assert result["counts"]["verifier_accepts"] == 1
    assert result["counts"]["origin_candidates"] == 1


def test_stale_origin_contract_is_rebuilt_for_changed_call_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))
    changed = replace(settings, origin_max_tokens=settings.origin_max_tokens + 1)

    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=changed, client=client)

    assert [request["schema_name"] for request in client.requests] == [
        "candidate_producer_origin_selection"
    ]
    assert client.requests[0]["max_tokens"] == changed.origin_max_tokens
    assert result["counts"]["verifier_resumed"] == 1
    assert result["counts"]["origin_resumed"] == 0


def test_origin_stage_requires_an_independent_verifier_accept(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    settings = replace(settings, verifier_model=None)
    client = _OriginPipelineClient("positive")

    with pytest.raises(ValueError, match="origin_model requires verifier_model"):
        run_paper(spec=spec, settings=settings, client=client)

    assert client.requests == []


def test_cleanup_removes_current_origin_summary_but_preserves_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    run_paper(spec=spec, settings=settings, client=_OriginPipelineClient("positive"))
    private = settings.output_root / spec.paper_id / "private"
    summary = private / "origin-retrieval.json"
    checkpoint = private / "origin-retrieval-checkpoint.json"
    assert summary.is_file()
    before = checkpoint.read_bytes()

    disabled = replace(settings, origin_model=None)
    result = run_paper(spec=spec, settings=disabled, client=_NeverCallClient())

    assert result["origin_retrieval"]["enabled"] is False
    assert not summary.exists()
    assert checkpoint.read_bytes() == before


def test_budget_stop_before_origin_preserves_verifier_checkpoint_for_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec, settings = _origin_fixture(monkeypatch, tmp_path)
    with pytest.raises(ProviderBudgetExhausted):
        run_paper(
            spec=spec,
            settings=settings,
            client=_OriginPipelineClient("budget"),
        )

    verifier_path = settings.output_root / spec.paper_id / "private" / "verifier-checkpoint.json"
    assert len(read_json(verifier_path)["candidates"]) == 1
    origin_path = (
        settings.output_root / spec.paper_id / "private" / "origin-retrieval-checkpoint.json"
    )
    assert read_json(origin_path)["candidates"] == {}

    client = _OriginPipelineClient("positive")
    result = run_paper(spec=spec, settings=settings, client=client)

    assert [request["schema_name"] for request in client.requests] == [
        "candidate_producer_origin_selection"
    ]
    assert result["counts"]["verifier_resumed"] == 1
    assert result["counts"]["origin_resumed"] == 0
