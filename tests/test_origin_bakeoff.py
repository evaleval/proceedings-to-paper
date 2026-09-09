from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from proceedings_to_eee.domain.attribution import AttributionState
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.evaluation import origin_bakeoff
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.io import read_json, write_json, write_jsonl
from proceedings_to_eee.providers.budget import ProviderBudgetExhausted
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderResponseValidationError,
    StructuredResponse,
    completion_token_parameter_for_model,
    structured_request_contract,
)
from proceedings_to_eee.resolution.origin_retrieval import (
    candidate_origin_binding_sha256,
    layout_binding_sha256,
)
from proceedings_to_eee.run_seal import seal_run_tree, verify_run_seal

_EXACT_RESULT = "Atlas Moderation API  61.3  74.6%  58.2"
_EXACT_METHOD = "We evaluated Atlas Moderation API on Synthetic Speech Set using AUC in this study."
_LABEL_SECRET = "ANNOTATOR-ONLY-NEVER-IN-PROMPT"


@dataclass(frozen=True)
class FrozenFixture:
    project_root: Path
    candidate: CandidateObservation
    layout: PdfLayout
    seal_sha256: str
    tree_sha256: str


@pytest.fixture
def frozen_fixture(tmp_path: Path, eligible_candidate: CandidateObservation) -> FrozenFixture:
    source = tmp_path / "development-run"
    private = source / "private"
    private.mkdir(parents=True)
    text = (
        "3 Methods\n"
        f"{_EXACT_METHOD}\n"
        "\n"
        "Table 2: Moderation results\n"
        "System                         F1    AUC    Recall\n"
        f"{_EXACT_RESULT}\n"
    )
    page = PageFragment(
        fragment_id="frag-src-paper-0007",
        source_id="src_paper",
        page=7,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=8,
        result_signal_score=8.0,
    )
    layout = PdfLayout(
        source_id="src_paper",
        parser="test-layout",
        parser_version="1",
        page_count=1,
        pages=[page],
    )
    write_json(private / "layout.json", layout)
    write_jsonl(source / "observations.jsonl", [eligible_candidate])
    sealed = tmp_path / "sealed-development-run"
    seal_run_tree(source, sealed)
    verified = verify_run_seal(sealed)
    return FrozenFixture(
        project_root=tmp_path,
        candidate=eligible_candidate,
        layout=layout,
        seal_sha256=verified.seal_sha256,
        tree_sha256=verified.tree_sha256,
    )


def _config(
    fixture: FrozenFixture,
    *,
    repetitions: int = 2,
    max_tokens: int = 900,
) -> origin_bakeoff.OriginBakeoffConfig:
    return origin_bakeoff.OriginBakeoffConfig(
        bakeoff_id="origin-test",
        sealed_run_path="sealed-development-run",
        sealed_run_seal_sha256=fixture.seal_sha256,
        sealed_run_tree_sha256=fixture.tree_sha256,
        models=[
            origin_bakeoff.OriginBakeoffModelSpec(
                model="vendor/paper-model", label="Paper proposer"
            ),
            origin_bakeoff.OriginBakeoffModelSpec(
                model="vendor/external-model", label="External proposer"
            ),
        ],
        candidates=[
            origin_bakeoff.FrozenOriginCandidateLocator(
                case_id="case-1",
                paper_id=fixture.candidate.paper_id,
                observation_id=fixture.candidate.observation_id or "",
                observations_path="observations.jsonl",
                layout_path="private/layout.json",
                candidate_binding_sha256=candidate_origin_binding_sha256(fixture.candidate),
                layout_binding_sha256=layout_binding_sha256(fixture.layout),
            )
        ],
        max_tokens=max_tokens,
        temperature=None,
        reasoning_effort=None,
        seed=None,
        fresh_repetitions=repetitions,
    )


def _selection_expectation(
    selection: origin_bakeoff.StageExecutionSelection,
) -> origin_bakeoff.StageExecutionSelectionExpectation:
    return origin_bakeoff.StageExecutionSelectionExpectation(
        experiment_id=selection.experiment_id,
        public_manifest_sha256=selection.public_manifest_sha256,
        wire_schema_gate=selection.wire_schema_gate,
    )


def test_provider_output_cannot_overwrite_sealed_origin_input(
    frozen_fixture: FrozenFixture,
    tmp_path: Path,
) -> None:
    target = frozen_fixture.project_root / "sealed-development-run" / "observations.jsonl"
    before = target.read_bytes()

    with pytest.raises(ValueError, match="outside the sealed run"):
        origin_bakeoff.run_origin_bakeoff_provider_phase(
            _config(frozen_fixture),
            project_root=frozen_fixture.project_root,
            checkpoint_path=tmp_path / "origin-checkpoint.json",
            output_path=target,
            client=object(),
        )

    assert target.read_bytes() == before


class FakeOriginClient:
    def __init__(
        self,
        *,
        contract_mismatch_models: set[str] | None = None,
        invalid_models: set[str] | None = None,
        interrupt_after: int | None = None,
        budget_after: int | None = None,
    ) -> None:
        self.contract_mismatch_models = contract_mismatch_models or set()
        self.invalid_models = invalid_models or set()
        self.interrupt_after = interrupt_after
        self.budget_after = budget_after
        self.requests: list[dict[str, Any]] = []

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        if self.budget_after is not None and len(self.requests) >= self.budget_after:
            raise ProviderBudgetExhausted(
                reason="structured_call_limit",
                summary={"structured_calls": len(self.requests)},
            )
        self.requests.append(dict(kwargs))
        if self.interrupt_after is not None and len(self.requests) > self.interrupt_after:
            raise KeyboardInterrupt
        assert _LABEL_SECRET not in kwargs["system"]
        assert _LABEL_SECRET not in kwargs["user"]
        marker = "<ORIGIN_INPUT>\n"
        payload_text = (
            kwargs["user"].split(marker, maxsplit=1)[1].split("\n</ORIGIN_INPUT>", maxsplit=1)[0]
        )
        origin_input = json.loads(payload_text)
        result_hit = next(
            hit
            for hit in origin_input["context_hits"]
            if "result_anchor" in hit["matched_dimensions"]
            and _EXACT_RESULT in hit["exact_excerpt"]
        )
        origin_hit = next(
            hit for hit in origin_input["context_hits"] if _EXACT_METHOD in hit["exact_excerpt"]
        )

        proposed_state = (
            "paper_produced" if kwargs["model"] == "vendor/paper-model" else "externally_sourced"
        )
        response_payload = {
            "candidate_binding_sha256": origin_input["candidate_binding_sha256"],
            "evaluated_system": origin_input["candidate"]["evaluated_system"],
            "proposed_state": proposed_state,
            "evidence_relation": "direct",
            "result_context_hit_id": result_hit["context_hit_id"],
            "origin_context_hit_id": origin_hit["context_hit_id"],
            "summary": "The supplied exact anchors support this proposal.",
        }
        actual_require = (
            False
            if kwargs["model"] in self.contract_mismatch_models
            else kwargs["require_parameters"]
        )
        contract = structured_request_contract(
            schema_name=kwargs["schema_name"],
            schema=kwargs["schema"],
            seed=kwargs["seed"],
            require_parameters=actual_require,
        )
        messages = [
            {"role": "system", "content": kwargs["system"]},
            {"role": "user", "content": kwargs["user"]},
        ]
        call = ProviderCall(
            model_requested=kwargs["model"],
            model_returned=kwargs["model"],
            provider_returned="test-provider",
            prompt_sha256=hashlib.sha256(
                json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            response_sha256=hashlib.sha256(
                json.dumps(response_payload, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            temperature=kwargs["temperature"],
            reasoning_effort=kwargs["reasoning_effort"],
            max_tokens=kwargs["max_tokens"],
            completion_token_parameter=completion_token_parameter_for_model(kwargs["model"]),
            seed=kwargs["seed"],
            schema_name=kwargs["schema_name"],
            schema_sha256=contract["schema"]["schema_sha256"],
            require_parameters=actual_require,
            latency_seconds=0.5,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            cost_usd=0.002,
            request_id=f"request-{len(self.requests)}",
            finish_reason="stop",
            attempts=1,
        )
        if kwargs["model"] in self.invalid_models:
            raise ProviderResponseValidationError(
                call=call,
                code="schema_validation",
                validation_path=("schema_version",),
                validation_keyword="const",
            )
        return StructuredResponse(payload=response_payload, call=call)


def _fixed_code_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        origin_bakeoff,
        "_code_state",
        lambda root: {"commit": "d" * 40, "dirty": False, "root": Path(root).name},
    )


def _write_labels(path: Path, config: origin_bakeoff.OriginBakeoffConfig) -> None:
    label = origin_bakeoff.OriginResultLabel(
        case_id="case-1",
        candidate_binding_sha256=config.candidates[0].candidate_binding_sha256,
        annotator=_LABEL_SECRET,
        result_label="result",
        origin_label=AttributionState.EXTERNALLY_SOURCED,
    )
    write_jsonl(path, [label])


def test_config_requires_common_contract_and_predeclared_repetitions(
    frozen_fixture: FrozenFixture,
) -> None:
    raw = _config(frozen_fixture).model_dump(mode="json")
    with pytest.raises(ValidationError):
        origin_bakeoff.OriginBakeoffConfig.model_validate(raw | {"models": raw["models"][:1]})
    with pytest.raises(ValidationError):
        origin_bakeoff.OriginBakeoffConfig.model_validate(raw | {"require_parameters": False})
    with pytest.raises(ValidationError):
        origin_bakeoff.OriginBakeoffConfig.model_validate(raw | {"fresh_repetitions": 0})
    with pytest.raises(ValidationError, match="must be an integer"):
        origin_bakeoff.OriginBakeoffConfig.model_validate(raw | {"seed": True})
    with pytest.raises(ValidationError, match="canonical relative path"):
        origin_bakeoff.OriginBakeoffConfig.model_validate(
            raw | {"sealed_run_path": "/tmp/not-frozen"}
        )

    repair04 = dict(raw)
    repair04["schema_version"] = "origin-bakeoff/0.2"
    for field in ("max_tokens", "temperature", "reasoning_effort", "seed"):
        repair04.pop(field)
    current = origin_bakeoff.OriginBakeoffConfig.model_validate(repair04)
    assert (current.max_tokens, current.temperature, current.reasoning_effort, current.seed) == (
        16_000,
        None,
        "minimal",
        None,
    )
    with pytest.raises(ValidationError, match="production request contract"):
        origin_bakeoff.OriginBakeoffConfig.model_validate(repair04 | {"reasoning_effort": None})


def test_provider_phase_resumes_exactly_and_keeps_exact_evidence_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_fixture: FrozenFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_fixture)
    checkpoint = tmp_path / "private" / "origin-checkpoint.json"
    first_client = FakeOriginClient()
    first = origin_bakeoff.run_origin_bakeoff_provider_phase(
        config,
        project_root=frozen_fixture.project_root,
        code_root=Path("/public/code-root"),
        checkpoint_path=checkpoint,
        client=first_client,
    )
    assert len(first_client.requests) == 4
    assert first["status"] == "sealed"
    assert first["aggregate"]["execution"]["logical_calls_predeclared"] == 4
    assert first["aggregate"]["schema"]["first_pass"]["calls"] == 2
    assert first["aggregate"]["usage"]["cost_usd"]["total"] == 0.008
    assert all(
        entry["assessment"]["allows_automatic_export"] is False for entry in first["entries"]
    )
    public_text = json.dumps(first, sort_keys=True)
    assert _EXACT_RESULT not in public_text
    assert _EXACT_METHOD not in public_text
    assert _EXACT_RESULT in checkpoint.read_text(encoding="utf-8")

    resumed_client = FakeOriginClient()
    resumed = origin_bakeoff.run_origin_bakeoff_provider_phase(
        config,
        project_root=frozen_fixture.project_root,
        code_root=Path("/public/code-root"),
        checkpoint_path=checkpoint,
        client=resumed_client,
    )
    assert resumed_client.requests == []
    assert resumed["resume"] == {
        "checkpoint_rejections": [],
        "logical_calls_reused": 4,
        "logical_calls_executed": 0,
    }


def test_checkpoint_is_immediate_and_resume_does_not_repeat_completed_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_fixture: FrozenFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_fixture)
    checkpoint = tmp_path / "private" / "interrupted.json"
    with pytest.raises(KeyboardInterrupt):
        origin_bakeoff.run_origin_bakeoff_provider_phase(
            config,
            project_root=frozen_fixture.project_root,
            checkpoint_path=checkpoint,
            client=FakeOriginClient(interrupt_after=1),
        )
    interrupted = read_json(checkpoint)
    assert interrupted["status"] == "in_progress"
    assert len(interrupted["entries"]) == 1

    client = FakeOriginClient()
    resumed = origin_bakeoff.run_origin_bakeoff_provider_phase(
        config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=checkpoint,
        client=client,
    )
    assert len(client.requests) == 3
    assert resumed["resume"]["logical_calls_reused"] == 1
    assert resumed["resume"]["logical_calls_executed"] == 3


def test_budget_stop_remains_in_progress_without_synthesizing_later_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_fixture: FrozenFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_fixture)
    checkpoint = tmp_path / "private" / "budget-stop.json"
    with pytest.raises(ProviderBudgetExhausted):
        origin_bakeoff.run_origin_bakeoff_provider_phase(
            config,
            project_root=frozen_fixture.project_root,
            checkpoint_path=checkpoint,
            client=FakeOriginClient(budget_after=1),
        )
    stopped = read_json(checkpoint)
    assert stopped["status"] == "in_progress"
    assert len(stopped["entries"]) == 1

    resumed_client = FakeOriginClient()
    resumed = origin_bakeoff.run_origin_bakeoff_provider_phase(
        config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=checkpoint,
        client=resumed_client,
    )
    assert len(resumed_client.requests) == 3
    assert resumed["resume"]["logical_calls_reused"] == 1
    assert resumed["resume"]["logical_calls_executed"] == 3


def test_tampered_and_stale_checkpoints_are_not_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_fixture: FrozenFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_fixture, repetitions=1)
    checkpoint = tmp_path / "private" / "tamper.json"
    origin_bakeoff.run_origin_bakeoff_provider_phase(
        config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=checkpoint,
        client=FakeOriginClient(),
    )
    raw = read_json(checkpoint)
    first_entry = next(iter(raw["entries"].values()))
    first_entry["request"]["settings"]["max_tokens"] = 123
    write_json(checkpoint, raw)
    tamper_client = FakeOriginClient()
    rebuilt = origin_bakeoff.run_origin_bakeoff_provider_phase(
        config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=checkpoint,
        client=tamper_client,
    )
    assert len(tamper_client.requests) == 2
    assert rebuilt["resume"]["checkpoint_rejections"] == ["checkpoint_invalid"]

    stale_config = _config(frozen_fixture, repetitions=1, max_tokens=901)
    stale_client = FakeOriginClient()
    stale = origin_bakeoff.run_origin_bakeoff_provider_phase(
        stale_config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=checkpoint,
        client=stale_client,
    )
    assert len(stale_client.requests) == 2
    assert stale["resume"]["checkpoint_rejections"] == ["checkpoint_configuration_stale"]


def test_fully_rehashed_proposal_splice_is_rejected_before_offline_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_fixture: FrozenFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_fixture, repetitions=1)
    checkpoint = tmp_path / "private" / "proposal-splice.json"
    origin_bakeoff.run_origin_bakeoff_provider_phase(
        config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=checkpoint,
        client=FakeOriginClient(),
    )

    raw = read_json(checkpoint)
    by_model = {entry["model"]: entry for entry in raw["entries"].values()}
    source = by_model["vendor/paper-model"]
    target = by_model["vendor/external-model"]
    # Splice a locally coherent proposal/assessment pair under another model's
    # provider call, then recompute every unkeyed integrity hash and phase seal.
    target["proposal"] = source["proposal"]
    target["assessment"] = source["assessment"]
    target["entry_sha256"] = origin_bakeoff._hash(
        {key: value for key, value in target.items() if key != "entry_sha256"}
    )
    raw["provider_phase_seal_sha256"] = origin_bakeoff._hash(
        {
            "schema_version": "origin-bakeoff-provider-seal/0.1",
            "configuration_sha256": raw["configuration_sha256"],
            "run_contract_sha256": raw["run_contract_sha256"],
            "entry_sha256s": {
                key: entry["entry_sha256"] for key, entry in sorted(raw["entries"].items())
            },
        }
    )
    raw["checkpoint_sha256"] = origin_bakeoff._hash(
        {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    )
    write_json(checkpoint, raw)

    labels = tmp_path / "private" / "labels-must-not-be-read.jsonl"
    labels_loaded = False

    def forbidden_labels(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal labels_loaded
        labels_loaded = True
        raise AssertionError("labels must remain unopened for an invalid provider checkpoint")

    monkeypatch.setattr(origin_bakeoff, "_load_labels", forbidden_labels)
    with pytest.raises(
        ValidationError,
        match="materialized origin proposal does not match provider response",
    ):
        origin_bakeoff.score_origin_bakeoff_offline(
            config,
            project_root=frozen_fixture.project_root,
            checkpoint_path=checkpoint,
            labels_path=labels,
        )
    assert labels_loaded is False


def test_offline_labels_are_prompt_isolated_and_safety_metrics_come_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_fixture: FrozenFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_fixture)
    labels = tmp_path / "private" / "origin-labels.jsonl"
    labels.parent.mkdir(parents=True)
    _write_labels(labels, config)
    checkpoint = tmp_path / "private" / "scored.json"
    client = FakeOriginClient()
    provider_result = origin_bakeoff.run_origin_bakeoff_provider_phase(
        config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=checkpoint,
        client=client,
    )
    assert provider_result["privacy"]["labels_loaded_during_provider_phase"] is False
    assert all(_LABEL_SECRET not in request["user"] for request in client.requests)

    score = origin_bakeoff.score_origin_bakeoff_offline(
        config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=checkpoint,
        labels_path=labels,
    )
    paper_quality = score["models"][0]["quality"]
    external_quality = score["models"][1]["quality"]
    assert next(iter(paper_quality)) == "unsafe_positive_proposals"
    assert paper_quality["unsafe_positive_proposals"]["count"] == 2
    assert external_quality["external_recall"] == {
        "numerator": 2,
        "denominator": 2,
        "value": 1.0,
    }
    assert paper_quality["review_burden"]["value"] == 1.0
    assert paper_quality["automatic_positive_promotion_violations"] == 0
    assert score["policy"]["automatic_positive_promotion"] is False
    assert _LABEL_SECRET not in json.dumps(score, sort_keys=True)


def test_contract_failures_and_wire_failures_keep_exact_repetition_denominators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_fixture: FrozenFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_fixture)
    checkpoint = tmp_path / "private" / "failures.json"
    result = origin_bakeoff.run_origin_bakeoff_provider_phase(
        config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=checkpoint,
        client=FakeOriginClient(
            contract_mismatch_models={"vendor/paper-model"},
            invalid_models={"vendor/external-model"},
        ),
    )
    aggregate = result["aggregate"]
    assert aggregate["execution"]["logical_calls_predeclared"] == 4
    assert aggregate["execution"]["contract_failure"] == 2
    assert aggregate["execution"]["response_validation_failure"] == 2
    assert aggregate["schema"]["all_repetitions"] == {
        "calls": 4,
        "valid": 2,
        "invalid": 2,
        "not_observed": 0,
        "valid_of_observed": {"numerator": 2, "denominator": 4, "value": 0.5},
        "valid_end_to_end": {"numerator": 2, "denominator": 4, "value": 0.5},
    }
    assert aggregate["schema"]["first_pass"]["calls"] == 2
    invalid_entries = [
        entry for entry in result["entries"] if entry["model"] == "vendor/external-model"
    ]
    assert invalid_entries
    assert all(
        entry["error"]
        == {
            "stage": "provider_response_validation",
            "type": "ProviderResponseValidationError",
            "code": "schema_validation",
            "validation_path": ["schema_version"],
            "validation_keyword": "const",
        }
        for entry in invalid_entries
    )
    private_invalid_entries = [
        entry
        for entry in read_json(checkpoint)["entries"].values()
        if entry["model"] == "vendor/external-model"
    ]
    assert [entry["error"] for entry in private_invalid_entries] == [
        entry["error"] for entry in invalid_entries
    ]
    assert aggregate["contract"] == {
        "satisfied": 2,
        "failed": 2,
        "not_observed": 0,
        "satisfaction_rate": {"numerator": 2, "denominator": 4, "value": 0.5},
    }
    assert all(
        entry.get("assessment", {}).get("allows_automatic_export") is not True
        for entry in result["entries"]
    )

    labels = tmp_path / "private" / "failure-labels.jsonl"
    _write_labels(labels, config)
    score = origin_bakeoff.score_origin_bakeoff_offline(
        config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=checkpoint,
        labels_path=labels,
    )
    by_model = {item["model"]: item for item in score["models"]}
    for item in by_model.values():
        assert item["quality_status"] == "unmeasured_incomplete_provider_contract"
        assert item["quality"] is None
        assert item["quality_measurement"]["status"] == "unmeasured"
        assert item["quality_measurement"]["valid_contract_complete_assessments"] == 0
        assert item["quality_measurement"]["incomplete_or_failed_calls"] == 2
    assert (
        by_model["vendor/paper-model"]["quality_measurement"]["terminal_status_counts"][
            "contract_failure"
        ]
        == 2
    )
    assert (
        by_model["vendor/external-model"]["quality_measurement"]["terminal_status_counts"][
            "response_validation_failure"
        ]
        == 2
    )


def test_sealed_smoke_selection_skips_ineligible_origin_model_and_keeps_null_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_fixture: FrozenFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    smoke_config = _config(frozen_fixture, repetitions=1)
    smoke_checkpoint = tmp_path / "private" / "origin-smoke.json"
    origin_bakeoff.run_origin_bakeoff_provider_phase(
        smoke_config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=smoke_checkpoint,
        client=FakeOriginClient(invalid_models={"vendor/external-model"}),
        code_root=Path("/public/code-root"),
    )
    selection = origin_bakeoff.derive_origin_execution_selection(
        smoke_config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=smoke_checkpoint,
        experiment_id="frozen-staged-comparison",
        public_manifest_sha256="a" * 64,
        code_root=Path("/public/code-root"),
    )
    assert selection.eligible_models == ("vendor/paper-model",)
    expectation = _selection_expectation(selection)

    full_config = _config(frozen_fixture, repetitions=2)
    full_checkpoint = tmp_path / "private" / "origin-full.json"
    full_client = FakeOriginClient()
    provider = origin_bakeoff.run_origin_bakeoff_provider_phase(
        full_config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=full_checkpoint,
        client=full_client,
        code_root=Path("/public/code-root"),
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert len(full_client.requests) == 2
    assert {request["model"] for request in full_client.requests} == {"vendor/paper-model"}
    assert provider["declared_models"] == [
        "vendor/paper-model",
        "vendor/external-model",
    ]
    assert provider["executed_models"] == ["vendor/paper-model"]
    skipped = provider["models"][1]
    assert skipped["matched_quality_status"] == "not_run"
    assert skipped["quality_status"] == "unmeasured_contract_ineligible"
    assert skipped["quality"] is None
    assert skipped["aggregate"] is None

    resumed_client = FakeOriginClient()
    resumed = origin_bakeoff.run_origin_bakeoff_provider_phase(
        full_config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=full_checkpoint,
        client=resumed_client,
        code_root=Path("/public/code-root"),
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert resumed_client.requests == []
    assert resumed["resume"]["logical_calls_reused"] == 2

    labels = tmp_path / "private" / "origin-labels.jsonl"
    _write_labels(labels, full_config)
    score = origin_bakeoff.score_origin_bakeoff_offline(
        full_config,
        project_root=frozen_fixture.project_root,
        checkpoint_path=full_checkpoint,
        labels_path=labels,
        code_root=Path("/public/code-root"),
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert score["models"][0]["quality_status"] == "measured"
    assert score["models"][1]["quality"] is None
    assert score["models"][1]["quality_status"] == ("unmeasured_contract_ineligible")

    stale_client = FakeOriginClient()
    with pytest.raises(ValueError, match="stale stage contract"):
        origin_bakeoff.run_origin_bakeoff_provider_phase(
            _config(frozen_fixture, repetitions=2, max_tokens=901),
            project_root=frozen_fixture.project_root,
            checkpoint_path=tmp_path / "private" / "stale-origin-full.json",
            client=stale_client,
            code_root=Path("/public/code-root"),
            execution_selection=selection,
            execution_selection_expectation=expectation,
        )
    assert stale_client.requests == []

    wrong_gate = expectation.model_copy(update={"wire_schema_gate": "0"})
    wrong_gate_client = FakeOriginClient()
    with pytest.raises(ValueError, match="another wire-schema gate"):
        origin_bakeoff.run_origin_bakeoff_provider_phase(
            full_config,
            project_root=frozen_fixture.project_root,
            checkpoint_path=tmp_path / "private" / "wrong-gate.json",
            client=wrong_gate_client,
            code_root=Path("/public/code-root"),
            execution_selection=selection,
            execution_selection_expectation=wrong_gate,
        )
    assert wrong_gate_client.requests == []

    tampered = read_json(smoke_checkpoint)
    tampered["provider_phase_seal_sha256"] = "0" * 64
    unsigned = {key: value for key, value in tampered.items() if key != "checkpoint_sha256"}
    tampered["checkpoint_sha256"] = origin_bakeoff._hash(unsigned)
    write_json(smoke_checkpoint, tampered)
    with pytest.raises(ValidationError, match="provider phase seal is invalid"):
        origin_bakeoff.derive_origin_execution_selection(
            smoke_config,
            project_root=frozen_fixture.project_root,
            checkpoint_path=smoke_checkpoint,
            experiment_id="frozen-staged-comparison",
            public_manifest_sha256="a" * 64,
            code_root=Path("/public/code-root"),
        )
