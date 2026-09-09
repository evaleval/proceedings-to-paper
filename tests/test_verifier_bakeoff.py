from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.evaluation import verifier_bakeoff
from proceedings_to_eee.extraction.pdf_layout import PageFragment
from proceedings_to_eee.extraction.result_blocks import (
    ResultBlock,
    segment_page_result_blocks,
)
from proceedings_to_eee.io import read_json, write_json, write_jsonl
from proceedings_to_eee.providers.budget import ProviderBudgetExhausted
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderResponseValidationError,
    StructuredResponse,
    completion_token_parameter_for_model,
    structured_request_contract,
)
from proceedings_to_eee.run_seal import seal_run_tree, verify_run_seal
from proceedings_to_eee.verification.binding import (
    bind_candidate_block,
    frozen_evidence_block,
)

_LABEL_SECRET = "ANNOTATOR-ONLY-VERIFIER-SECRET"
_QUOTE = "Atlas Moderation API  61.3  74.6%  58.2"
_PAGE_TEXT = (
    "Table 2: Synthetic Speech Set test split n=6400; AUC percent; "
    "reference Synthetic Speech Set labels\n"
    f"System                         F1    AUC    Recall\n{_QUOTE}\n"
)


@dataclass(frozen=True)
class FrozenVerifierFixture:
    project_root: Path
    candidates: tuple[CandidateObservation, ...]
    blocks: tuple[ResultBlock, ...]
    seal_sha256: str
    tree_sha256: str


@pytest.fixture
def frozen_verifier_fixture(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> FrozenVerifierFixture:
    candidates_list: list[CandidateObservation] = []
    for label in ("good", "bad", "uncertain"):
        payload = eligible_candidate.model_dump(mode="python")
        payload["observation_id"] = f"candidate-{label}"
        if label == "bad":
            payload["scope"]["dataset_raw"] = "Dataset Absent From Source"
        elif label == "uncertain":
            payload["scope"]["group"] = "Group Absent From Source"
        candidates_list.append(CandidateObservation.model_validate(payload))
    candidates = tuple(candidates_list)
    page = PageFragment(
        fragment_id="frag-src-paper-0007",
        source_id="src_paper",
        page=7,
        text=_PAGE_TEXT,
        text_sha256=hashlib.sha256(_PAGE_TEXT.encode("utf-8")).hexdigest(),
        character_count=len(_PAGE_TEXT),
        numeric_token_count=4,
        result_signal_score=8.0,
    )
    blocks = tuple(segment_page_result_blocks(page))
    assert len(blocks) == 1
    assert all(bind_candidate_block(candidate, list(blocks)) for candidate in candidates)
    source = tmp_path / "source-run"
    private = source / "private"
    private.mkdir(parents=True)
    write_jsonl(source / "observations.jsonl", candidates)
    write_json(private / "result-blocks.json", list(blocks))
    sealed = tmp_path / "sealed-run"
    seal_run_tree(source, sealed)
    verified = verify_run_seal(sealed)
    return FrozenVerifierFixture(
        project_root=tmp_path,
        candidates=candidates,
        blocks=blocks,
        seal_sha256=verified.seal_sha256,
        tree_sha256=verified.tree_sha256,
    )


def _locator(
    fixture: FrozenVerifierFixture,
    candidate: CandidateObservation,
) -> verifier_bakeoff.FrozenVerifierCaseLocator:
    support = bind_candidate_block(candidate, list(fixture.blocks))
    assert support is not None
    block, anchor = support
    evidence = frozen_evidence_block(
        paper_id=candidate.paper_id,
        block=block,
        anchor=anchor,
    )
    suffix = (candidate.observation_id or "").removeprefix("candidate-")
    return verifier_bakeoff.FrozenVerifierCaseLocator(
        case_id=f"case-{suffix}",
        paper_id=candidate.paper_id,
        observation_id=candidate.observation_id or "",
        observations_path="observations.jsonl",
        result_blocks_path="private/result-blocks.json",
        result_block_id=block.block_id,
        candidate_binding_sha256=(verifier_bakeoff.verifier_candidate_binding_sha256(candidate)),
        result_block_sha256=(verifier_bakeoff.verifier_result_block_binding_sha256(block)),
        evidence_anchor_sha256=(verifier_bakeoff.verifier_evidence_anchor_binding_sha256(anchor)),
        evidence_block_sha256=(verifier_bakeoff.verifier_evidence_block_binding_sha256(evidence)),
        candidate_block_binding_sha256=(
            verifier_bakeoff.verifier_candidate_block_binding_sha256(
                candidate=candidate,
                result_block=block,
                evidence_anchor=anchor,
                evidence_block=evidence,
            )
        ),
    )


def _config(
    fixture: FrozenVerifierFixture,
    *,
    repetitions: int = 2,
    max_tokens: int = 700,
) -> verifier_bakeoff.VerifierBakeoffConfig:
    return verifier_bakeoff.VerifierBakeoffConfig(
        bakeoff_id="verifier-test",
        sealed_run_path="sealed-run",
        sealed_run_seal_sha256=fixture.seal_sha256,
        sealed_run_tree_sha256=fixture.tree_sha256,
        models=[
            verifier_bakeoff.VerifierBakeoffModelSpec(model="vendor/safe", label="Safe"),
            verifier_bakeoff.VerifierBakeoffModelSpec(model="vendor/unsafe", label="Unsafe"),
        ],
        cases=[_locator(fixture, candidate) for candidate in fixture.candidates],
        max_tokens=max_tokens,
        temperature=None,
        reasoning_effort="minimal",
        seed=None,
        fresh_repetitions=repetitions,
    )


def _selection_expectation(
    selection: verifier_bakeoff.StageExecutionSelection,
) -> verifier_bakeoff.StageExecutionSelectionExpectation:
    return verifier_bakeoff.StageExecutionSelectionExpectation(
        experiment_id=selection.experiment_id,
        public_manifest_sha256=selection.public_manifest_sha256,
        wire_schema_gate=selection.wire_schema_gate,
    )


def test_provider_output_cannot_overwrite_sealed_verifier_input(
    frozen_verifier_fixture: FrozenVerifierFixture,
    tmp_path: Path,
) -> None:
    target = frozen_verifier_fixture.project_root / "sealed-run" / "observations.jsonl"
    before = target.read_bytes()

    with pytest.raises(ValueError, match="outside the sealed run"):
        verifier_bakeoff.run_verifier_bakeoff_provider_phase(
            _config(frozen_verifier_fixture),
            project_root=frozen_verifier_fixture.project_root,
            checkpoint_path=tmp_path / "verifier-checkpoint.json",
            output_path=target,
            client=object(),
        )

    assert target.read_bytes() == before


def _assessment(*, decision: str, request: dict[str, Any]) -> dict[str, Any]:
    if decision == "accept":
        findings = ["supported"] * 5
    elif decision == "reject":
        findings = ["supported", "supported", "contradicted", "supported", "supported"]
    else:
        findings = [
            "supported",
            "supported",
            "insufficient_evidence",
            "supported",
            "supported",
        ]
    candidate = request["candidate_claim_untrusted"]
    anchor = request["candidate_claimed_anchor_untrusted"]
    lines = request["trusted_frozen_source_block"]["lines"]

    def evidence_ids(*claims: object) -> list[str]:
        selected: list[str] = []
        for claim in claims:
            if claim is None or claim == "":
                continue
            line_id = next(
                (
                    line["line_id"]
                    for line in lines
                    if str(claim).casefold() in line["text"].casefold()
                ),
                None,
            )
            if line_id is not None and line_id not in selected:
                selected.append(line_id)
        return selected

    scope = candidate["scope"] or {}
    metric = candidate["metric"] or {}
    value = candidate["value"] or {}
    mapped = dict(zip(("support", "role", "scope", "value", "metric"), findings, strict=True))
    return {
        **mapped,
        "support_evidence_line_ids": evidence_ids(anchor["quote"]),
        "role_evidence_line_ids": evidence_ids(*(role["raw_name"] for role in candidate["roles"])),
        "scope_evidence_line_ids": (
            []
            if mapped["scope"] == "insufficient_evidence"
            else evidence_ids(
                scope.get("dataset_raw"),
                scope.get("split"),
                scope.get("sample_count"),
                scope.get("group"),
            )
        ),
        "value_evidence_line_ids": evidence_ids(value.get("raw"), value.get("unit")),
        "metric_evidence_line_ids": evidence_ids(metric.get("raw_name"), metric.get("unit")),
        "decision": decision,
        "justification": "The supplied frozen block supports this bounded assessment.",
    }


class FakeVerifierClient:
    def __init__(
        self,
        *,
        contract_mismatch_models: set[str] | None = None,
        invalid_models: set[str] | None = None,
        local_invalid_models: set[str] | None = None,
        budget_after: int | None = None,
    ) -> None:
        self.contract_mismatch_models = contract_mismatch_models or set()
        self.invalid_models = invalid_models or set()
        self.local_invalid_models = local_invalid_models or set()
        self.budget_after = budget_after
        self.requests: list[dict[str, Any]] = []

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        if self.budget_after is not None and len(self.requests) >= self.budget_after:
            raise ProviderBudgetExhausted(
                reason="structured_call_limit",
                summary={"structured_calls": len(self.requests)},
            )
        self.requests.append(dict(kwargs))
        assert kwargs["temperature"] is None
        assert kwargs["reasoning_effort"] == "minimal"
        assert kwargs["seed"] is None
        assert kwargs["require_parameters"] is True
        assert _LABEL_SECRET not in kwargs["system"]
        assert _LABEL_SECRET not in kwargs["user"]
        marker = "<VERIFICATION_INPUT>\n"
        serialized = kwargs["user"].split(marker, 1)[1].split("\n</VERIFICATION_INPUT>", 1)[0]
        request = json.loads(serialized)
        semantic_scope = request["candidate_claim_untrusted"]["scope"]
        if (
            kwargs["model"] == "vendor/unsafe"
            or semantic_scope["dataset_raw"] == ("Synthetic Speech Set")
            and semantic_scope["group"] is None
        ):
            decision = "accept"
        elif semantic_scope["dataset_raw"] == "Dataset Absent From Source":
            decision = "reject"
        else:
            decision = "review"
        payload = _assessment(decision=decision, request=request)
        if kwargs["model"] in self.local_invalid_models:
            payload["decision"] = "accept"
            payload["support"] = "contradicted"
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
        call = ProviderCall(
            model_requested=kwargs["model"],
            model_returned=kwargs["model"],
            provider_returned="test-provider",
            prompt_sha256=hashlib.sha256(
                json.dumps(
                    [
                        {"role": "system", "content": kwargs["system"]},
                        {"role": "user", "content": kwargs["user"]},
                    ],
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            response_sha256=hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            temperature=kwargs["temperature"],
            reasoning_effort=kwargs["reasoning_effort"],
            max_tokens=kwargs["max_tokens"],
            completion_token_parameter=completion_token_parameter_for_model(kwargs["model"]),
            seed=kwargs["seed"],
            schema_name=kwargs["schema_name"],
            schema_sha256=contract["schema"]["schema_sha256"],
            require_parameters=actual_require,
            latency_seconds=0.4,
            input_tokens=80,
            output_tokens=20,
            total_tokens=100,
            cost_usd=0.003,
            request_id=f"request-{len(self.requests)}",
            finish_reason="stop",
            attempts=1,
        )
        if kwargs["model"] in self.invalid_models:
            raise ProviderResponseValidationError(call=call, code="schema_validation")
        return StructuredResponse(payload=payload, call=call)


class AlwaysFailVerifierClient:
    def structured_chat(self, **_kwargs: Any) -> StructuredResponse:
        raise RuntimeError("provider transport failed without call telemetry")


def _write_labels(
    path: Path,
    config: verifier_bakeoff.VerifierBakeoffConfig,
) -> None:
    labels = ("good_candidate", "bad_candidate", "uncertain")
    write_jsonl(
        path,
        [
            verifier_bakeoff.VerifierCandidateLabel(
                case_id=case.case_id,
                candidate_binding_sha256=case.candidate_binding_sha256,
                candidate_block_binding_sha256=case.candidate_block_binding_sha256,
                annotator=_LABEL_SECRET,
                label=label,
            )
            for case, label in zip(config.cases, labels, strict=True)
        ],
    )


def _fixed_code_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        verifier_bakeoff,
        "_code_state",
        lambda root: {"commit": "d" * 40, "dirty": False, "root": Path(root).name},
    )


def test_strict_contract_and_stale_result_block_binding_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_verifier_fixture: FrozenVerifierFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    raw = _config(frozen_verifier_fixture).model_dump(mode="json")
    raw["require_parameters"] = False
    with pytest.raises(ValidationError):
        verifier_bakeoff.VerifierBakeoffConfig.model_validate(raw)
    raw = _config(frozen_verifier_fixture).model_dump(mode="json")
    raw["cases"][0]["result_block_sha256"] = "0" * 64
    stale = verifier_bakeoff.VerifierBakeoffConfig.model_validate(raw)
    with pytest.raises(ValueError, match="binding is stale"):
        verifier_bakeoff.run_verifier_bakeoff_provider_phase(
            stale,
            project_root=frozen_verifier_fixture.project_root,
            checkpoint_path=tmp_path / "stale.json",
            client=FakeVerifierClient(),
        )


def test_label_blind_repeated_provider_phase_and_offline_safety_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_verifier_fixture: FrozenVerifierFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_verifier_fixture)
    checkpoint = tmp_path / "private" / "checkpoint.json"
    labels = tmp_path / "private" / "labels.jsonl"
    labels.parent.mkdir(parents=True)
    _write_labels(labels, config)
    client = FakeVerifierClient()
    provider = verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=checkpoint,
        client=client,
    )
    assert len(client.requests) == 12
    assert provider["aggregate"]["execution"]["logical_calls_predeclared"] == 12
    assert provider["privacy"]["labels_loaded_during_provider_phase"] is False
    assert provider["privacy"]["raw_quotes_or_evidence_in_public_result"] is False
    public = json.dumps(provider, sort_keys=True)
    assert _LABEL_SECRET not in public
    assert _QUOTE not in public
    assert "candidate-good" not in public
    assert frozen_verifier_fixture.blocks[0].block_id not in public
    assert all(request["require_parameters"] is True for request in client.requests)
    assert all(request["temperature"] is None for request in client.requests)
    assert provider["models"][0]["stability"]["repetition_pair_denominator"] == 3

    score = verifier_bakeoff.score_verifier_bakeoff_offline(
        config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=checkpoint,
        labels_path=labels,
    )
    safe = score["models"][0]["quality"]
    unsafe = score["models"][1]["quality"]
    assert next(iter(safe)) == "unsafe_accepts"
    assert safe["unsafe_accepts"]["count"] == 0
    assert safe["bad_candidate_reject_or_review_recall"] == {
        "numerator": 2,
        "denominator": 2,
        "value": 1.0,
    }
    assert safe["good_candidate_accept_recall"]["value"] == 1.0
    assert safe["exact_denominators"]["logical_calls_predeclared"] == 6
    assert unsafe["unsafe_accepts"] == {
        "count": 0,
        "accept_predictions": 2,
        "rate_of_all_predeclared_calls": {
            "numerator": 0,
            "denominator": 6,
            "value": 0.0,
        },
        "rate_of_accept_predictions": {
            "numerator": 0,
            "denominator": 2,
            "value": 0.0,
        },
    }
    unsafe_entries = [
        entry
        for entry in provider["entries"]
        if entry["model"] == "vendor/unsafe"
        and entry["verification"]["provider_decision"] == "accept"
        and entry["verification"]["effective_decision"] == "review"
    ]
    assert len(unsafe_entries) == 4
    assert _LABEL_SECRET not in json.dumps(score, sort_keys=True)
    assert score["candidate_mutation_or_promotion"] is False


def test_schema_and_contract_failures_keep_exact_denominators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_verifier_fixture: FrozenVerifierFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_verifier_fixture, repetitions=1)
    checkpoint = tmp_path / "failures.json"
    result = verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=checkpoint,
        client=FakeVerifierClient(
            contract_mismatch_models={"vendor/safe"},
            invalid_models={"vendor/unsafe"},
        ),
    )
    aggregate = result["aggregate"]
    assert aggregate["execution"] == {
        "logical_calls_predeclared": 6,
        "terminal_entries": 6,
        "success": 0,
        "contract_failure": 3,
        "response_validation_failure": 3,
        "local_validation_failure": 0,
        "provider_failure": 0,
    }
    assert aggregate["wire_schema"]["all_repetitions"] == {
        "logical_calls": 6,
        "valid": 3,
        "invalid": 3,
        "not_observed": 0,
        "valid_of_observed": {"numerator": 3, "denominator": 6, "value": 0.5},
        "valid_end_to_end": {"numerator": 3, "denominator": 6, "value": 0.5},
    }
    assert aggregate["contract"]["failed"] == 3
    assert aggregate["contract"]["satisfied"] == 3

    labels = tmp_path / "failure-labels.jsonl"
    _write_labels(labels, config)
    score = verifier_bakeoff.score_verifier_bakeoff_offline(
        config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=checkpoint,
        labels_path=labels,
    )
    by_model = {item["model"]: item for item in score["models"]}
    for item in by_model.values():
        assert item["quality_status"] == "unmeasured_incomplete_provider_contract"
        assert item["quality"] is None
        assert item["quality_measurement"]["status"] == "unmeasured"
        assert item["quality_measurement"]["valid_contract_complete_verifications"] == 0
        assert item["quality_measurement"]["incomplete_or_failed_calls"] == 3
    assert (
        by_model["vendor/safe"]["quality_measurement"]["terminal_status_counts"]["contract_failure"]
        == 3
    )
    assert (
        by_model["vendor/unsafe"]["quality_measurement"]["terminal_status_counts"][
            "response_validation_failure"
        ]
        == 3
    )


def test_budget_stop_checkpoints_only_completed_slot_then_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_verifier_fixture: FrozenVerifierFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_verifier_fixture, repetitions=1)
    checkpoint = tmp_path / "budget.json"
    with pytest.raises(ProviderBudgetExhausted):
        verifier_bakeoff.run_verifier_bakeoff_provider_phase(
            config,
            project_root=frozen_verifier_fixture.project_root,
            checkpoint_path=checkpoint,
            client=FakeVerifierClient(budget_after=1),
        )
    stopped = read_json(checkpoint)
    assert stopped["status"] == "in_progress"
    assert len(stopped["entries"]) == 1

    resumed_client = FakeVerifierClient()
    resumed = verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=checkpoint,
        client=resumed_client,
    )
    assert len(resumed_client.requests) == 5
    assert resumed["resume"] == {
        "checkpoint_rejections": [],
        "logical_calls_reused": 1,
        "logical_calls_executed": 5,
    }
    no_calls = FakeVerifierClient()
    fully_resumed = verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=checkpoint,
        client=no_calls,
    )
    assert no_calls.requests == []
    assert fully_resumed["resume"]["logical_calls_reused"] == 6


def test_provider_failures_are_missing_from_usage_lower_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_verifier_fixture: FrozenVerifierFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    result = verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        _config(frozen_verifier_fixture, repetitions=1),
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=tmp_path / "provider-failures.json",
        client=AlwaysFailVerifierClient(),
    )
    aggregate = result["aggregate"]
    logical_calls = aggregate["execution"]["logical_calls_predeclared"]
    assert aggregate["execution"]["provider_failure"] == logical_calls
    for field in ("total_tokens", "cost_usd"):
        assert aggregate["usage"][field] == {
            "total": 0,
            "reported_calls": 0,
            "missing_calls": logical_calls,
        }


def test_tampered_checkpoint_rebuilds_and_unsealed_scoring_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_verifier_fixture: FrozenVerifierFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_verifier_fixture, repetitions=1)
    checkpoint = tmp_path / "tamper.json"
    verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=checkpoint,
        client=FakeVerifierClient(),
    )
    raw = read_json(checkpoint)
    first = next(iter(raw["entries"].values()))
    first["request"]["settings"]["max_tokens"] = 1
    write_json(checkpoint, raw)
    rebuilt_client = FakeVerifierClient()
    rebuilt = verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=checkpoint,
        client=rebuilt_client,
    )
    assert rebuilt["resume"]["checkpoint_rejections"] == ["checkpoint_invalid"]
    assert len(rebuilt_client.requests) == 6

    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, config)
    raw = read_json(checkpoint)
    raw["status"] = "in_progress"
    raw["provider_phase_seal_sha256"] = None
    unsigned = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    raw["checkpoint_sha256"] = verifier_bakeoff._hash(unsigned)
    write_json(checkpoint, raw)
    with pytest.raises(ValueError, match="not sealed"):
        verifier_bakeoff.score_verifier_bakeoff_offline(
            config,
            project_root=frozen_verifier_fixture.project_root,
            checkpoint_path=checkpoint,
            labels_path=labels,
        )


def test_rehashed_local_grounding_splice_is_contextually_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_verifier_fixture: FrozenVerifierFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_verifier_fixture, repetitions=1)
    checkpoint = tmp_path / "grounding-splice.json"
    verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=checkpoint,
        client=FakeVerifierClient(),
    )
    raw = read_json(checkpoint)
    entry = next(
        item
        for item in raw["entries"].values()
        if item["model"] == "vendor/unsafe" and item["case_id"] == "case-bad"
    )
    grounding = entry["verification"]["grounding"]
    for name in ("support", "role", "scope", "value", "metric"):
        grounding[name]["status"] = "grounded"
        grounding[name]["failure_codes"] = []
    grounding["passed"] = True
    grounding["failure_codes"] = []
    entry["verification"]["effective_decision"] = "accept"
    entry["entry_sha256"] = verifier_bakeoff._hash(
        {key: value for key, value in entry.items() if key != "entry_sha256"}
    )
    raw["provider_phase_seal_sha256"] = verifier_bakeoff._hash(
        {
            "schema_version": "verifier-bakeoff-provider-seal/0.2",
            "configuration_sha256": raw["configuration_sha256"],
            "run_contract_sha256": raw["run_contract_sha256"],
            "entry_sha256s": {
                key: value["entry_sha256"] for key, value in sorted(raw["entries"].items())
            },
        }
    )
    raw["checkpoint_sha256"] = verifier_bakeoff._hash(
        {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    )
    write_json(checkpoint, raw)

    client = FakeVerifierClient()
    rebuilt = verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=checkpoint,
        client=client,
    )

    assert len(client.requests) == 1
    assert rebuilt["resume"]["checkpoint_rejections"] == ["checkpoint_entry_stale_or_tampered"]
    repaired = next(
        item
        for item in rebuilt["entries"]
        if item["model"] == "vendor/unsafe" and item["case_id"] == "case-bad"
    )
    assert repaired["verification"]["provider_decision"] == "accept"
    assert repaired["verification"]["effective_decision"] == "review"


def test_sealed_smoke_selection_skips_ineligible_verifier_and_keeps_null_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_verifier_fixture: FrozenVerifierFixture,
) -> None:
    _fixed_code_state(monkeypatch)
    smoke_config = _config(frozen_verifier_fixture, repetitions=1)
    smoke_checkpoint = tmp_path / "private" / "verifier-smoke.json"
    verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        smoke_config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=smoke_checkpoint,
        client=FakeVerifierClient(invalid_models={"vendor/unsafe"}),
        code_root=Path("/public/code-root"),
    )
    selection = verifier_bakeoff.derive_verifier_execution_selection(
        smoke_config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=smoke_checkpoint,
        experiment_id="frozen-staged-comparison",
        public_manifest_sha256="a" * 64,
        code_root=Path("/public/code-root"),
    )
    assert selection.eligible_models == ("vendor/safe",)
    expectation = _selection_expectation(selection)

    full_config = _config(frozen_verifier_fixture, repetitions=2)
    full_checkpoint = tmp_path / "private" / "verifier-full.json"
    full_client = FakeVerifierClient()
    provider = verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        full_config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=full_checkpoint,
        client=full_client,
        code_root=Path("/public/code-root"),
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert len(full_client.requests) == 6
    assert {request["model"] for request in full_client.requests} == {"vendor/safe"}
    assert provider["declared_models"] == ["vendor/safe", "vendor/unsafe"]
    assert provider["executed_models"] == ["vendor/safe"]
    skipped = provider["models"][1]
    assert skipped["matched_quality_status"] == "not_run"
    assert skipped["quality_status"] == "unmeasured_contract_ineligible"
    assert skipped["quality"] is None
    assert skipped["aggregate"] is None

    resumed_client = FakeVerifierClient()
    resumed = verifier_bakeoff.run_verifier_bakeoff_provider_phase(
        full_config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=full_checkpoint,
        client=resumed_client,
        code_root=Path("/public/code-root"),
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert resumed_client.requests == []
    assert resumed["resume"]["logical_calls_reused"] == 6

    labels = tmp_path / "private" / "verifier-labels.jsonl"
    _write_labels(labels, full_config)
    score = verifier_bakeoff.score_verifier_bakeoff_offline(
        full_config,
        project_root=frozen_verifier_fixture.project_root,
        checkpoint_path=full_checkpoint,
        labels_path=labels,
        code_root=Path("/public/code-root"),
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert score["models"][0]["quality_status"] == "measured"
    assert score["models"][1]["quality"] is None
    assert score["models"][1]["quality_status"] == ("unmeasured_contract_ineligible")

    stale_client = FakeVerifierClient()
    with pytest.raises(ValueError, match="stale stage contract"):
        verifier_bakeoff.run_verifier_bakeoff_provider_phase(
            _config(frozen_verifier_fixture, repetitions=2, max_tokens=701),
            project_root=frozen_verifier_fixture.project_root,
            checkpoint_path=tmp_path / "private" / "stale-verifier-full.json",
            client=stale_client,
            code_root=Path("/public/code-root"),
            execution_selection=selection,
            execution_selection_expectation=expectation,
        )
    assert stale_client.requests == []

    wrong_manifest = expectation.model_copy(update={"public_manifest_sha256": "0" * 64})
    wrong_manifest_client = FakeVerifierClient()
    with pytest.raises(ValueError, match="another public manifest"):
        verifier_bakeoff.run_verifier_bakeoff_provider_phase(
            full_config,
            project_root=frozen_verifier_fixture.project_root,
            checkpoint_path=tmp_path / "private" / "wrong-manifest.json",
            client=wrong_manifest_client,
            code_root=Path("/public/code-root"),
            execution_selection=selection,
            execution_selection_expectation=wrong_manifest,
        )
    assert wrong_manifest_client.requests == []

    tampered = read_json(smoke_checkpoint)
    tampered["provider_phase_seal_sha256"] = "0" * 64
    unsigned = {key: value for key, value in tampered.items() if key != "checkpoint_sha256"}
    tampered["checkpoint_sha256"] = verifier_bakeoff._hash(unsigned)
    write_json(smoke_checkpoint, tampered)
    with pytest.raises(ValidationError, match="provider phase seal is invalid"):
        verifier_bakeoff.derive_verifier_execution_selection(
            smoke_config,
            project_root=frozen_verifier_fixture.project_root,
            checkpoint_path=smoke_checkpoint,
            experiment_id="frozen-staged-comparison",
            public_manifest_sha256="a" * 64,
            code_root=Path("/public/code-root"),
        )
