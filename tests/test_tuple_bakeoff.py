from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.evaluation import tuple_bakeoff
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
from proceedings_to_eee.resolution.origin_retrieval import layout_binding_sha256
from proceedings_to_eee.resolution.tuple_resolution import (
    TupleFields,
    build_tuple_resolution_input,
    materialize_tuple_wire_proposal,
    tuple_candidate_binding_sha256,
)
from proceedings_to_eee.run_seal import seal_run_tree, verify_run_seal

_LABEL_SECRET = "ANNOTATOR-ONLY-TUPLE-SECRET"
_PRIOR_TUPLE_SECRET = "UNTRUSTED-PRIOR-TUPLE-MUST-NOT-LEAK"
_ROW = (
    "Atlas Moderation API v2  Synthetic Speech Set v1  test  n=6400  "
    "zero-shot  AUC  higher is better  range 0 100  74.6%"
)


@dataclass(frozen=True)
class FrozenTupleFixture:
    project_root: Path
    candidate: CandidateObservation
    layout: PdfLayout
    seal_sha256: str
    tree_sha256: str


@pytest.fixture
def frozen_tuple_fixture(
    tmp_path: Path,
    eligible_candidate: CandidateObservation,
) -> FrozenTupleFixture:
    raw = eligible_candidate.model_dump(mode="json")
    raw["notes"] = [_PRIOR_TUPLE_SECRET]
    raw["evidence"][0].update(
        {
            "row": (
                "Atlas Moderation API v2 · Synthetic Speech Set v1 · test · n=6400 · zero-shot"
            ),
            "column": "AUC percent higher is better range 0 100",
            "quote": _ROW,
            "quote_sha256": hashlib.sha256(_ROW.encode("utf-8")).hexdigest(),
            "region_id": "tregion_fixture",
            "planned_row_id": "trow_fixture",
            "cell_id": "tcell_" + "a" * 20,
            "numeric_token_id": "ttoken_" + "b" * 20,
            "header_ids": ["theader_" + "c" * 20],
        }
    )
    raw["observation_id"] = None
    candidate = CandidateObservation.model_validate(raw)
    page_text = "Table 2: Moderation results\n" + _ROW + "\n"
    page = PageFragment(
        fragment_id="frag-src-paper-0007",
        source_id="src_paper",
        page=7,
        text=page_text,
        text_sha256=hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
        character_count=len(page_text),
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
    source = tmp_path / "source-run"
    private = source / "private"
    private.mkdir(parents=True)
    write_jsonl(source / "observations.jsonl", [candidate])
    write_json(private / "layout.json", layout)
    sealed = tmp_path / "sealed-run"
    seal_run_tree(source, sealed)
    verified = verify_run_seal(sealed)
    return FrozenTupleFixture(
        project_root=tmp_path,
        candidate=candidate,
        layout=layout,
        seal_sha256=verified.seal_sha256,
        tree_sha256=verified.tree_sha256,
    )


def _locator(fixture: FrozenTupleFixture) -> tuple_bakeoff.FrozenTupleCaseLocator:
    tuple_input = build_tuple_resolution_input(fixture.candidate, fixture.layout)
    return tuple_bakeoff.FrozenTupleCaseLocator(
        case_id="case-primary",
        paper_id=fixture.candidate.paper_id,
        observation_id=fixture.candidate.observation_id or "",
        observations_path="observations.jsonl",
        layout_path="private/layout.json",
        candidate_binding_sha256=tuple_candidate_binding_sha256(fixture.candidate),
        layout_binding_sha256=layout_binding_sha256(fixture.layout),
        tuple_input_sha256=tuple_input.input_sha256,
        result_evidence_binding_sha256=tuple_input.result_evidence_binding_sha256,
    )


def _config(
    fixture: FrozenTupleFixture,
    *,
    repetitions: int = 2,
    models: tuple[str, str] = ("vendor/safe", "vendor/unsafe"),
    max_tokens: int = 900,
) -> tuple_bakeoff.TupleBakeoffConfig:
    return tuple_bakeoff.TupleBakeoffConfig(
        bakeoff_id="tuple-test",
        sealed_run_path="sealed-run",
        sealed_run_seal_sha256=fixture.seal_sha256,
        sealed_run_tree_sha256=fixture.tree_sha256,
        models=[
            tuple_bakeoff.TupleBakeoffModelSpec(model=model, label=model.rsplit("/", 1)[-1])
            for model in models
        ],
        cases=[_locator(fixture)],
        max_tokens=max_tokens,
        temperature=None,
        reasoning_effort=None,
        seed=None,
        fresh_repetitions=repetitions,
    )


def _selection_expectation(
    selection: tuple_bakeoff.StageExecutionSelection,
) -> tuple_bakeoff.StageExecutionSelectionExpectation:
    return tuple_bakeoff.StageExecutionSelectionExpectation(
        experiment_id=selection.experiment_id,
        public_manifest_sha256=selection.public_manifest_sha256,
        wire_schema_gate=selection.wire_schema_gate,
    )


def _proposal_payload(tuple_input: dict[str, Any], *, model: str) -> dict[str, Any]:
    evidence_id = tuple_input["result_evidence"][0]["evidence_id"]
    unsafe = model == "vendor/unsafe"
    return {
        "candidate_binding_sha256": tuple_input["candidate_binding_sha256"],
        "result_evidence_binding_sha256": tuple_input["result_evidence_binding_sha256"],
        "result_evidence_id": evidence_id,
        "evaluated_system": {"raw_name": "Atlas Moderation API"},
        "system_version": "v2",
        "dataset": {"dataset_raw": "Synthetic Speech Set"},
        "dataset_version": "v1",
        "metric": {"raw_name": "AUC"},
        "direction": {"raw_direction": "higher is better", "lower_is_better": False},
        "scale": {"raw_scale": "range 0 100", "min_score": 0, "max_score": 100},
        "value": {"raw": "74.6%", "numeric": 74.6, "comparator": "exact"},
        "uncertainty": None,
        "unit": "percent",
        "setting": {"raw_setting": "zero-shot", "parameters": {}},
        "scope": {
            "split": None if unsafe else "test",
            "subset": "invented holdout" if unsafe else None,
            "group": None,
            "language": None,
            "sample_count": None if unsafe else 6400,
            "aggregation": None,
            "raw_scope": None,
        },
        "field_evidence": {
            "system_evidence_id": evidence_id,
            "system_version_evidence_id": evidence_id,
            "dataset_evidence_id": evidence_id,
            "dataset_version_evidence_id": evidence_id,
            "metric_evidence_id": evidence_id,
            "direction_evidence_id": evidence_id,
            "scale_evidence_id": evidence_id,
            "value_evidence_id": evidence_id,
            "uncertainty_evidence_id": evidence_id,
            "unit_evidence_id": evidence_id,
            "setting_evidence_id": evidence_id,
            "scope_evidence_id": evidence_id,
        },
        "unresolved_fields": [],
        "not_applicable_fields": ["uncertainty"],
        "summary": f"Evidence-bound tuple proposal from {model}.",
    }


def test_provider_output_cannot_overwrite_sealed_tuple_input(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    target = frozen_tuple_fixture.project_root / "sealed-run" / "observations.jsonl"
    before = target.read_bytes()

    with pytest.raises(ValueError, match="outside the sealed run"):
        tuple_bakeoff.run_tuple_bakeoff_provider_phase(
            _config(frozen_tuple_fixture),
            project_root=frozen_tuple_fixture.project_root,
            checkpoint_path=tmp_path / "tuple-checkpoint.json",
            output_path=target,
            client=object(),
        )

    assert target.read_bytes() == before


class FakeTupleClient:
    def __init__(
        self,
        *,
        invalid_models: set[str] | None = None,
        validation_failures: dict[str, list[dict[str, Any] | None]] | None = None,
        contract_mismatch_models: set[str] | None = None,
        wrong_hash_models: set[str] | None = None,
        budget_after: int | None = None,
    ) -> None:
        self.invalid_models = invalid_models or set()
        self.validation_failures = validation_failures or {}
        self.contract_mismatch_models = contract_mismatch_models or set()
        self.wrong_hash_models = wrong_hash_models or set()
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
        assert kwargs["reasoning_effort"] is None
        assert kwargs["seed"] is None
        assert kwargs["require_parameters"] is True
        assert _LABEL_SECRET not in kwargs["system"]
        assert _LABEL_SECRET not in kwargs["user"]
        assert _PRIOR_TUPLE_SECRET not in kwargs["user"]
        serialized = kwargs["user"].split("<TUPLE_INPUT>\n", 1)[1].split("\n</TUPLE_INPUT>", 1)[0]
        tuple_input = json.loads(serialized)
        assert tuple_input["candidate_tuple_fields"] == "withheld_untrusted"
        payload = _proposal_payload(tuple_input, model=kwargs["model"])
        model_call_index = sum(request["model"] == kwargs["model"] for request in self.requests) - 1
        failure_sequence = self.validation_failures.get(kwargs["model"], [])
        validation_failure = (
            failure_sequence[model_call_index] if model_call_index < len(failure_sequence) else None
        )
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
        response_sha256 = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if kwargs["model"] in self.wrong_hash_models:
            response_sha256 = "0" * 64
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
            response_sha256=response_sha256,
            temperature=kwargs["temperature"],
            reasoning_effort=kwargs["reasoning_effort"],
            max_tokens=kwargs["max_tokens"],
            completion_token_parameter=completion_token_parameter_for_model(kwargs["model"]),
            seed=kwargs["seed"],
            schema_name=kwargs["schema_name"],
            schema_sha256=contract["schema"]["schema_sha256"],
            require_parameters=actual_require,
            latency_seconds=0.25,
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            cost_usd=0.004,
            request_id=f"tuple-request-{len(self.requests)}",
            finish_reason=(
                validation_failure.get("finish_reason")
                if validation_failure is not None
                else "stop"
            ),
            attempts=1,
        )
        if validation_failure is not None:
            raise ProviderResponseValidationError(
                call=call,
                code=validation_failure["code"],
                validation_path=tuple(validation_failure.get("validation_path", ())),
                validation_keyword=validation_failure.get("validation_keyword"),
            )
        if kwargs["model"] in self.invalid_models:
            raise ProviderResponseValidationError(
                call=call,
                code="schema_validation",
                validation_path=("schema_version",),
                validation_keyword="const",
            )
        return StructuredResponse(payload=payload, call=call)


class AlwaysFailTupleClient:
    def structured_chat(self, **_kwargs: Any) -> StructuredResponse:
        raise RuntimeError("provider transport failed without call telemetry")


def _reference_tuple(fixture: FrozenTupleFixture) -> TupleFields:
    tuple_input = build_tuple_resolution_input(fixture.candidate, fixture.layout)
    proposal = materialize_tuple_wire_proposal(
        _proposal_payload(tuple_input.model_dump(mode="json"), model="vendor/safe")
    )
    return TupleFields(
        evaluated_system=proposal.evaluated_system,
        system_version=proposal.system_version,
        dataset=proposal.dataset,
        dataset_version=proposal.dataset_version,
        metric=proposal.metric,
        direction=proposal.direction,
        scale=proposal.scale,
        value=proposal.value,
        uncertainty=proposal.uncertainty,
        unit=proposal.unit,
        setting=proposal.setting,
        scope=proposal.scope,
    )


def _reference_authority(
    *,
    authority_type: str = "development_reference_derivation",
    human_annotation: bool = False,
) -> tuple_bakeoff.TupleReferenceAuthority:
    return tuple_bakeoff.TupleReferenceAuthority(
        authority_id=_LABEL_SECRET,
        authority_type=authority_type,
        human_annotation=human_annotation,
    )


def _write_labels(
    path: Path,
    fixture: FrozenTupleFixture,
    config: tuple_bakeoff.TupleBakeoffConfig,
) -> None:
    labels = [
        tuple_bakeoff.TupleReferenceLabel(
            case_id=case.case_id,
            tuple_input_sha256=case.tuple_input_sha256,
            reference_authority=_reference_authority(),
            reference_tuple=_reference_tuple(fixture),
            unresolved_fields=[],
            not_applicable_fields=["uncertainty"],
        )
        for case in config.cases
    ]
    path.write_text(
        "".join(
            json.dumps(
                label.model_dump(mode="json", exclude_none=False),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for label in labels
        ),
        encoding="utf-8",
    )


def test_reference_authority_contract_rejects_inconsistent_human_metadata(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="inconsistent with human_annotation"):
        tuple_bakeoff.TupleReferenceAuthority(
            authority_id="derived-reference",
            authority_type="development_reference_derivation",
            human_annotation=True,
        )
    with pytest.raises(ValidationError, match="inconsistent with human_annotation"):
        tuple_bakeoff.TupleReferenceAuthority(
            authority_id="human-reviewer",
            authority_type="human_annotation",
            human_annotation=False,
        )

    legacy = tuple_bakeoff.TupleReferenceLabel(
        case_id="legacy-case",
        tuple_input_sha256="c" * 64,
        reference_authority=_reference_authority(),
        reference_tuple=_reference_tuple(frozen_tuple_fixture),
        unresolved_fields=[],
        not_applicable_fields=["uncertainty"],
    ).model_dump(mode="json", exclude_none=False)
    legacy["schema_version"] = "tuple-bakeoff-label/0.1"
    legacy["annotator"] = legacy.pop("reference_authority")["authority_id"]
    with pytest.raises(ValidationError):
        tuple_bakeoff.TupleReferenceLabel.model_validate(legacy)

    reference = _reference_tuple(frozen_tuple_fixture)
    labels = [
        tuple_bakeoff.TupleReferenceLabel(
            case_id="case-a",
            tuple_input_sha256="a" * 64,
            reference_authority=_reference_authority(),
            reference_tuple=reference,
            unresolved_fields=[],
            not_applicable_fields=["uncertainty"],
        ),
        tuple_bakeoff.TupleReferenceLabel(
            case_id="case-b",
            tuple_input_sha256="b" * 64,
            reference_authority=_reference_authority(
                authority_type="human_annotation",
                human_annotation=True,
            ),
            reference_tuple=reference,
            unresolved_fields=[],
            not_applicable_fields=["uncertainty"],
        ),
    ]
    labels_path = tmp_path / "inconsistent-reference-authority.jsonl"
    labels_path.write_text(
        "".join(
            json.dumps(
                label.model_dump(mode="json", exclude_none=False),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for label in labels
        ),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        cases=[
            SimpleNamespace(
                locator=SimpleNamespace(
                    case_id="case-a",
                    tuple_input_sha256="a" * 64,
                )
            ),
            SimpleNamespace(
                locator=SimpleNamespace(
                    case_id="case-b",
                    tuple_input_sha256="b" * 64,
                )
            ),
        ]
    )
    with pytest.raises(ValueError, match="consistent reference-authority contract"):
        tuple_bakeoff._load_labels(labels_path, context)  # noqa: SLF001


def test_config_requires_common_contract_and_two_models(
    frozen_tuple_fixture: FrozenTupleFixture,
) -> None:
    raw = _config(frozen_tuple_fixture).model_dump(mode="json")
    raw["require_parameters"] = False
    with pytest.raises(ValidationError):
        tuple_bakeoff.TupleBakeoffConfig.model_validate(raw)
    raw = _config(frozen_tuple_fixture).model_dump(mode="json")
    raw["models"] = raw["models"][:1]
    with pytest.raises(ValidationError):
        tuple_bakeoff.TupleBakeoffConfig.model_validate(raw)

    repair04 = _config(frozen_tuple_fixture).model_dump(mode="json")
    repair04["schema_version"] = "tuple-bakeoff/0.2"
    for field in ("max_tokens", "temperature", "reasoning_effort", "seed"):
        repair04.pop(field)
    current = tuple_bakeoff.TupleBakeoffConfig.model_validate(repair04)
    assert (current.max_tokens, current.temperature, current.reasoning_effort, current.seed) == (
        16_000,
        None,
        "minimal",
        None,
    )
    with pytest.raises(ValidationError, match="production request contract"):
        tuple_bakeoff.TupleBakeoffConfig.model_validate(repair04 | {"max_tokens": 3_000})


def test_provider_phase_is_label_blind_private_resumable_and_exactly_accounted(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    config = _config(frozen_tuple_fixture)
    checkpoint = tmp_path / "tuple-checkpoint.json"
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, frozen_tuple_fixture, config)
    client = FakeTupleClient()

    result = tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=checkpoint,
        client=client,
    )

    assert result["schema_version"] == "tuple-bakeoff-provider-result/0.3"
    assert len(client.requests) == 4
    assert result["aggregate"]["execution"] == {
        "logical_calls_predeclared": 4,
        "terminal_entries": 4,
        "success": 4,
        "contract_failure": 0,
        "response_validation_failure": 0,
        "local_validation_failure": 0,
        "provider_failure": 0,
    }
    assert result["aggregate"]["wire_schema"]["first_pass"]["logical_calls"] == 2
    assert result["models"][0]["stability"]["repetition_pair_denominator"] == 1
    serialized = json.dumps(result, sort_keys=True)
    assert _ROW not in serialized
    assert _LABEL_SECRET not in serialized
    assert _PRIOR_TUPLE_SECRET not in serialized
    assert "tuple_ev_" not in serialized
    assert frozen_tuple_fixture.candidate.observation_id not in serialized
    assert result["origin_or_export_decisions"] is False

    resumed = FakeTupleClient()
    resumed_result = tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=checkpoint,
        client=resumed,
    )
    assert resumed.requests == []
    assert resumed_result["resume"]["logical_calls_reused"] == 4


def test_budget_stop_leaves_in_progress_checkpoint_and_resumes_remaining_slots(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    config = _config(frozen_tuple_fixture)
    checkpoint = tmp_path / "budget-checkpoint.json"
    stopping = FakeTupleClient(budget_after=1)

    with pytest.raises(ProviderBudgetExhausted):
        tuple_bakeoff.run_tuple_bakeoff_provider_phase(
            config,
            project_root=frozen_tuple_fixture.project_root,
            checkpoint_path=checkpoint,
            client=stopping,
        )
    partial = read_json(checkpoint)
    assert partial["status"] == "in_progress"
    assert partial["provider_phase_seal_sha256"] is None
    assert len(partial["entries"]) == 1

    resumed = FakeTupleClient()
    result = tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=checkpoint,
        client=resumed,
    )
    assert len(resumed.requests) == 3
    assert result["resume"]["logical_calls_reused"] == 1
    assert result["aggregate"]["execution"]["terminal_entries"] == 4


def test_provider_failures_are_missing_from_usage_lower_bounds(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    result = tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        _config(frozen_tuple_fixture, repetitions=1),
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=tmp_path / "provider-failures.json",
        client=AlwaysFailTupleClient(),
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


def test_fully_rehashed_proposal_splice_fails_response_cross_binding(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    config = _config(frozen_tuple_fixture, repetitions=1)
    checkpoint = tmp_path / "tampered-checkpoint.json"
    tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=checkpoint,
        client=FakeTupleClient(),
    )
    raw = read_json(checkpoint)
    entries = list(raw["entries"].values())
    entries[0]["proposal"] = entries[1]["proposal"]
    entries[0]["assessment"] = entries[1]["assessment"]
    entries[0]["entry_sha256"] = tuple_bakeoff._hash(  # noqa: SLF001
        {key: value for key, value in entries[0].items() if key != "entry_sha256"}
    )
    raw["provider_phase_seal_sha256"] = tuple_bakeoff._hash(  # noqa: SLF001
        {
            "schema_version": "tuple-bakeoff-provider-seal/0.1",
            "configuration_sha256": raw["configuration_sha256"],
            "run_contract_sha256": raw["run_contract_sha256"],
            "entry_sha256s": {
                key: value["entry_sha256"] for key, value in sorted(raw["entries"].items())
            },
        }
    )
    raw["checkpoint_sha256"] = tuple_bakeoff._hash(  # noqa: SLF001
        {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    )
    write_json(checkpoint, raw)
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, frozen_tuple_fixture, config)

    with pytest.raises(ValidationError, match="provider response hash"):
        tuple_bakeoff.score_tuple_bakeoff_offline(
            config,
            project_root=frozen_tuple_fixture.project_root,
            checkpoint_path=checkpoint,
            labels_path=labels,
        )


def test_offline_score_loads_labels_only_after_seal_and_measures_safe_rejects(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    config = _config(frozen_tuple_fixture)
    checkpoint = tmp_path / "score-checkpoint.json"
    with pytest.raises(ProviderBudgetExhausted):
        tuple_bakeoff.run_tuple_bakeoff_provider_phase(
            config,
            project_root=frozen_tuple_fixture.project_root,
            checkpoint_path=checkpoint,
            client=FakeTupleClient(budget_after=0),
        )
    with pytest.raises(ValueError, match="not sealed"):
        tuple_bakeoff.score_tuple_bakeoff_offline(
            config,
            project_root=frozen_tuple_fixture.project_root,
            checkpoint_path=checkpoint,
            labels_path=tmp_path / "missing-private-labels.jsonl",
        )

    tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=checkpoint,
        client=FakeTupleClient(),
    )
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, frozen_tuple_fixture, config)
    score = tuple_bakeoff.score_tuple_bakeoff_offline(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=checkpoint,
        labels_path=labels,
    )
    assert score["schema_version"] == "tuple-bakeoff-score/0.4"
    assert score["reference_authority"] == {
        "authority_type": "development_reference_derivation",
        "human_annotation": False,
    }
    assert "authority_id" not in score["reference_authority"]
    assert "authority_id_sha256" not in score["reference_authority"]
    assert "annotator_sha256" not in score
    assert _LABEL_SECRET not in json.dumps(score, sort_keys=True)

    safe, unsafe = score["models"]
    assert safe["quality_status"] == "measured"
    assert safe["quality"]["exact_joint_tuple_accuracy"]["value"] == 1.0
    field_quality = safe["quality"]["evidence_bound_field_accuracy"]
    assert set(field_quality) == {field.value for field in tuple_bakeoff.TupleField}
    assert field_quality["direction"]["exact_semantic_accuracy"]["value"] == 1.0
    assert field_quality["scale"]["exact_semantic_accuracy"]["value"] == 1.0
    assert field_quality["uncertainty"]["reference_states"]["not_applicable"] == 2
    assert field_quality["uncertainty"]["exact_state_accuracy"]["value"] == 1.0
    assert field_quality["setting"]["exact_semantic_accuracy"]["value"] == 1.0
    assert unsafe["quality_status"] == "measured"
    assert unsafe["quality"]["unsafe_value_or_scope_proposals"]["count"] == 2
    assert unsafe["quality"]["exact_denominators"]["logical_calls_predeclared"] == 2


def test_schema_or_request_contract_failure_is_unmeasured_not_wrong_tuple(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    config = _config(
        frozen_tuple_fixture,
        repetitions=1,
        models=("vendor/safe", "vendor/invalid"),
    )
    checkpoint = tmp_path / "failure-checkpoint.json"
    result = tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=checkpoint,
        client=FakeTupleClient(invalid_models={"vendor/invalid"}),
    )
    assert result["models"][1]["aggregate"]["wire_schema"]["first_pass"]["invalid"] == 1
    invalid_entry = next(entry for entry in result["entries"] if entry["model"] == "vendor/invalid")
    assert invalid_entry["error"] == {
        "stage": "provider_response_validation",
        "type": "ProviderResponseValidationError",
        "code": "schema_validation",
        "validation_path": ["schema_version"],
        "validation_keyword": "const",
    }
    private_entry = next(
        entry
        for entry in read_json(checkpoint)["entries"].values()
        if entry["model"] == "vendor/invalid"
    )
    assert private_entry["error"] == invalid_entry["error"]
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, frozen_tuple_fixture, config)
    score = tuple_bakeoff.score_tuple_bakeoff_offline(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=checkpoint,
        labels_path=labels,
    )
    invalid = score["models"][1]
    assert invalid["quality"] is None
    assert invalid["quality_status"] == "unmeasured_incomplete_matched_quality_partition"
    assert invalid["quality_accounting"] == {
        "logical_calls_predeclared": 1,
        "terminal_entries": 1,
        "wire_schema_valid": 0,
        "request_contract_satisfied": 1,
        "deterministic_semantic_assessments": 0,
        "reason_codes": [
            "wire_schema_invalid_or_unobserved",
            "semantic_assessment_missing",
        ],
        "complete": False,
    }

    contract_config = _config(
        frozen_tuple_fixture,
        repetitions=1,
        models=("vendor/safe", "vendor/contract-bad"),
    )
    contract_checkpoint = tmp_path / "contract-checkpoint.json"
    contract_result = tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        contract_config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=contract_checkpoint,
        client=FakeTupleClient(contract_mismatch_models={"vendor/contract-bad"}),
    )
    failure_codes = contract_result["models"][1]["aggregate"]["contract"]
    assert failure_codes["failed"] == 1
    bad_entry = next(
        entry for entry in contract_result["entries"] if entry["model"] == "vendor/contract-bad"
    )
    assert bad_entry["contract_failure_codes"] == [
        "require_parameters_mismatch",
        "request_contract_mismatch",
    ]
    contract_labels = tmp_path / "contract-labels.jsonl"
    _write_labels(contract_labels, frozen_tuple_fixture, contract_config)
    contract_score = tuple_bakeoff.score_tuple_bakeoff_offline(
        contract_config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=contract_checkpoint,
        labels_path=contract_labels,
    )
    assert contract_score["models"][1]["quality"] is None
    assert (
        "request_contract_unsatisfied_or_unobserved"
        in contract_score["models"][1]["quality_accounting"]["reason_codes"]
    )


def test_stability_separates_technical_outcomes_from_semantic_pairs(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    config = _config(
        frozen_tuple_fixture,
        repetitions=3,
        models=("vendor/safe", "vendor/mixed"),
    )
    result = tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=tmp_path / "mixed-stability-checkpoint.json",
        client=FakeTupleClient(
            validation_failures={
                "vendor/mixed": [
                    {
                        "code": "invalid_json",
                        "finish_reason": "length",
                    },
                    None,
                    None,
                ]
            }
        ),
    )

    stability = result["models"][1]["stability"]
    assert stability["repetition_pair_denominator"] == 3
    assert stability["technical_outcome_pair_denominator"] == 3
    assert stability["matching_technical_outcomes"] == 1
    assert stability["technical_outcome_match_rate"] == {
        "numerator": 1,
        "denominator": 3,
        "value": 0.333333,
    }
    assert stability["semantic_decision_pair_denominator"] == 1
    assert stability["matching_semantic_decisions"] == 1
    assert stability["semantic_decision_match_rate"] == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
    }
    assert stability["assessment_pair_denominator"] == 1
    assert stability["exact_semantic_assessment_matches"] == 1


def test_null_quality_retains_secret_free_tuple_failure_diagnostics(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    config = _config(
        frozen_tuple_fixture,
        repetitions=3,
        models=("vendor/invalid-json", "vendor/wire-invalid"),
    )
    checkpoint = tmp_path / "diagnostic-checkpoint.json"
    result = tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=checkpoint,
        client=FakeTupleClient(
            validation_failures={
                "vendor/invalid-json": [
                    {"code": "invalid_json", "finish_reason": "length"},
                    {"code": "invalid_json", "finish_reason": "error"},
                    {"code": "invalid_json", "finish_reason": None},
                ],
                "vendor/wire-invalid": [
                    {
                        "code": "wire_validation",
                        "validation_path": [],
                        "validation_keyword": "value_error",
                    },
                    {
                        "code": "wire_validation",
                        "validation_path": ["scale"],
                        "validation_keyword": "value_error",
                    },
                    {
                        "code": "wire_validation",
                        "validation_path": ["scale"],
                        "validation_keyword": "less_than",
                    },
                ],
            }
        ),
    )

    invalid_json = result["models"][0]
    assert invalid_json["aggregate"]["failure_diagnostics"] == {
        "technical_failure_entries": 3,
        "entries_with_error_metadata": 3,
        "error_code_counts": [{"code": "invalid_json", "count": 3}],
        "invalid_json": {
            "count": 3,
            "finish_reason_counts": {
                "length": 1,
                "error": 1,
                "missing": 1,
                "other": 0,
            },
        },
        "wire_validation": {
            "count": 0,
            "validation_path_counts": [],
            "validation_keyword_counts": [],
        },
    }
    assert invalid_json["stability"]["semantic_decision_pair_denominator"] == 0
    assert invalid_json["stability"]["decision_match_rate"]["value"] is None

    wire = result["models"][1]
    wire_diagnostics = wire["aggregate"]["failure_diagnostics"]
    assert wire_diagnostics["wire_validation"] == {
        "count": 3,
        "validation_path_counts": [
            {"validation_path": [], "count": 1},
            {"validation_path": ["scale"], "count": 2},
        ],
        "validation_keyword_counts": [
            {"validation_keyword": "less_than", "count": 1},
            {"validation_keyword": "value_error", "count": 2},
        ],
    }

    labels = tmp_path / "diagnostic-labels.jsonl"
    _write_labels(labels, frozen_tuple_fixture, config)
    score = tuple_bakeoff.score_tuple_bakeoff_offline(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=checkpoint,
        labels_path=labels,
    )
    for provider_model, scored_model in zip(result["models"], score["models"], strict=True):
        assert scored_model["quality"] is None
        assert scored_model["quality_status"] == "unmeasured_incomplete_matched_quality_partition"
        assert (
            scored_model["failure_diagnostics"]
            == provider_model["aggregate"]["failure_diagnostics"]
        )
        assert scored_model["stability"] == provider_model["stability"]
    serialized = json.dumps(score, sort_keys=True)
    assert _LABEL_SECRET not in serialized
    assert _ROW not in serialized


def test_smoke_selection_skips_ineligible_model_with_explicit_null_quality(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    models = ("vendor/safe", "vendor/invalid")
    smoke_config = _config(frozen_tuple_fixture, repetitions=2, models=models)
    smoke_checkpoint = tmp_path / "smoke-checkpoint.json"
    tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        smoke_config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=smoke_checkpoint,
        client=FakeTupleClient(invalid_models={"vendor/invalid"}),
    )
    selection = tuple_bakeoff.derive_tuple_execution_selection(
        smoke_config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=smoke_checkpoint,
        experiment_id="tuple-experiment",
        public_manifest_sha256="a" * 64,
        expected_fresh_repetitions=2,
    )
    assert selection.stage == "tuple_resolution"
    assert selection.eligible_models == ("vendor/safe",)
    assert selection.decision_for("vendor/safe").logical_calls_predeclared == 2
    assert selection.decision_for("vendor/invalid").first_pass_schema_invalid == 2
    expectation = _selection_expectation(selection)

    with pytest.raises(ValueError, match="repetition count differs"):
        tuple_bakeoff.derive_tuple_execution_selection(
            smoke_config,
            project_root=frozen_tuple_fixture.project_root,
            checkpoint_path=smoke_checkpoint,
            experiment_id="tuple-experiment",
            public_manifest_sha256="a" * 64,
            expected_fresh_repetitions=1,
        )

    stale_full = _config(
        frozen_tuple_fixture,
        repetitions=2,
        models=models,
        max_tokens=smoke_config.max_tokens + 1,
    )
    stale_client = FakeTupleClient()
    with pytest.raises(ValueError, match="stale stage contract"):
        tuple_bakeoff.run_tuple_bakeoff_provider_phase(
            stale_full,
            project_root=frozen_tuple_fixture.project_root,
            checkpoint_path=tmp_path / "stale-selection-checkpoint.json",
            client=stale_client,
            execution_selection=selection,
            execution_selection_expectation=expectation,
        )
    assert stale_client.requests == []

    full_config = _config(frozen_tuple_fixture, repetitions=2, models=models)
    full_client = FakeTupleClient()
    result = tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        full_config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=tmp_path / "full-checkpoint.json",
        client=full_client,
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert len(full_client.requests) == 2
    assert {request["model"] for request in full_client.requests} == {"vendor/safe"}
    assert result["models"][1]["matched_quality_status"] == "not_run"
    assert result["models"][1]["quality"] is None
    assert result["models"][1]["aggregate"] is None

    wrong_experiment = expectation.model_copy(update={"experiment_id": "another-experiment"})
    wrong_experiment_client = FakeTupleClient()
    with pytest.raises(ValueError, match="another experiment"):
        tuple_bakeoff.run_tuple_bakeoff_provider_phase(
            full_config,
            project_root=frozen_tuple_fixture.project_root,
            checkpoint_path=tmp_path / "wrong-experiment.json",
            client=wrong_experiment_client,
            execution_selection=selection,
            execution_selection_expectation=wrong_experiment,
        )
    assert wrong_experiment_client.requests == []


def test_stale_locator_and_plan_drift_fail_before_provider_calls(
    frozen_tuple_fixture: FrozenTupleFixture,
    tmp_path: Path,
) -> None:
    raw = _config(frozen_tuple_fixture).model_dump(mode="json")
    raw["cases"][0]["tuple_input_sha256"] = "0" * 64
    stale = tuple_bakeoff.TupleBakeoffConfig.model_validate(raw)
    client = FakeTupleClient()
    with pytest.raises(ValueError, match="binding is stale"):
        tuple_bakeoff.run_tuple_bakeoff_provider_phase(
            stale,
            project_root=frozen_tuple_fixture.project_root,
            checkpoint_path=tmp_path / "stale.json",
            client=client,
        )
    assert client.requests == []

    config = _config(frozen_tuple_fixture, repetitions=1)
    checkpoint = tmp_path / "drift-checkpoint.json"
    tuple_bakeoff.run_tuple_bakeoff_provider_phase(
        config,
        project_root=frozen_tuple_fixture.project_root,
        checkpoint_path=checkpoint,
        client=FakeTupleClient(),
    )
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, frozen_tuple_fixture, config)
    drifted = config.model_copy(update={"max_tokens": config.max_tokens + 1})
    with pytest.raises(ValueError, match="stale"):
        tuple_bakeoff.score_tuple_bakeoff_offline(
            drifted,
            project_root=frozen_tuple_fixture.project_root,
            checkpoint_path=checkpoint,
            labels_path=labels,
        )
