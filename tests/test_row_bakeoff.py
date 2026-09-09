from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from proceedings_to_eee.evaluation import row_bakeoff
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.result_blocks import segment_page_result_blocks
from proceedings_to_eee.extraction.row_enumeration import (
    EnumerationRow,
    RowEnumerationConfig,
    RowEnumerationPlan,
    build_row_enumeration_plan,
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

_LABEL_SECRET = "ANNOTATOR-ONLY-ROW-SECRET"
_LONG_LABEL = (
    "AspenWithAnExtremelyLongDescriptiveConfigurationNameThatCannotFitInsideThe"
    "ConfiguredProviderBatchAndIsRetainedForManualReviewOnlyBecauseItExceedsTheHardLimit"
)
_PAGE = f"""Table 7. Synthetic evaluation scores for invented systems.

       Engine          X      Y     Z                         U     V    W
       Cedar      .12 .64 .20                              .71 .44 .54
       Juniper    .27 .58 .37                              .76 .31 .44
       Maple      .21 .74 .33                              .82 .61 .70
       Willow     .16 .69 .26                              .79 .52 .63
       {_LONG_LABEL}    .25 .62 .36     .77 .35 .48
"""


@dataclass(frozen=True)
class FrozenRows:
    project_root: Path
    layout: PdfLayout
    plan: RowEnumerationPlan
    selected: tuple[EnumerationRow, ...]
    unsupported_row_id: str
    seal_sha256: str
    tree_sha256: str


@pytest.fixture
def frozen_rows(tmp_path: Path) -> FrozenRows:
    page = PageFragment(
        fragment_id="frag-src-paper-0001",
        source_id="src_paper",
        page=1,
        text=_PAGE,
        text_sha256=hashlib.sha256(_PAGE.encode("utf-8")).hexdigest(),
        character_count=len(_PAGE),
        numeric_token_count=0,
        result_signal_score=10.0,
    )
    layout = PdfLayout(
        source_id=page.source_id,
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[page],
    )
    plan = build_row_enumeration_plan(
        layout,
        segment_page_result_blocks(page),
        RowEnumerationConfig(
            max_rows_per_batch=4,
            max_value_tokens_per_batch=24,
            max_characters_per_batch=300,
            max_recovery_depth=1,
        ),
    )
    assert len(plan.batches) == 2
    assert len(plan.unbatchable_rows) == 1
    selected = tuple(plan.rows[:3])
    source = tmp_path / "source-run"
    private = source / "private"
    private.mkdir(parents=True)
    write_json(private / "layout.json", layout)
    write_json(private / "row-plan.json", plan)
    sealed = tmp_path / "sealed-run"
    seal_run_tree(source, sealed)
    verified = verify_run_seal(sealed)
    return FrozenRows(
        project_root=tmp_path,
        layout=layout,
        plan=plan,
        selected=selected,
        unsupported_row_id=plan.unbatchable_rows[0].row_id,
        seal_sha256=verified.seal_sha256,
        tree_sha256=verified.tree_sha256,
    )


def _config(
    fixture: FrozenRows,
    *,
    repetitions: int = 2,
    max_tokens: int = 800,
) -> row_bakeoff.RowBakeoffConfig:
    return row_bakeoff.RowBakeoffConfig(
        bakeoff_id="row-test",
        sealed_run_path="sealed-run",
        sealed_run_seal_sha256=fixture.seal_sha256,
        sealed_run_tree_sha256=fixture.tree_sha256,
        models=[
            row_bakeoff.RowBakeoffModelSpec(model="vendor/good", label="Good"),
            row_bakeoff.RowBakeoffModelSpec(model="vendor/risky", label="Risky"),
        ],
        cases=[
            row_bakeoff.FrozenRowCaseLocator(
                case_id="case-1",
                paper_id="paper-1",
                paper_title="Synthetic paper",
                source_id=fixture.layout.source_id,
                layout_path="private/layout.json",
                row_plan_path="private/row-plan.json",
                source_binding_sha256=row_bakeoff.row_source_binding_sha256(fixture.layout),
                layout_binding_sha256=row_bakeoff.row_layout_binding_sha256(fixture.layout),
                row_plan_sha256=row_bakeoff.row_plan_binding_sha256(fixture.plan),
                selected_row_ids=[
                    *(row.row_id for row in fixture.selected),
                    fixture.unsupported_row_id,
                ],
                balanced_rows_per_class=1,
            )
        ],
        max_tokens=max_tokens,
        temperature=None,
        reasoning_effort=None,
        seed=None,
        fresh_repetitions=repetitions,
    )


def _selection_expectation(
    selection: row_bakeoff.StageExecutionSelection,
) -> row_bakeoff.StageExecutionSelectionExpectation:
    return row_bakeoff.StageExecutionSelectionExpectation(
        experiment_id=selection.experiment_id,
        public_manifest_sha256=selection.public_manifest_sha256,
        wire_schema_gate=selection.wire_schema_gate,
    )


def _smoke_config(fixture: FrozenRows) -> row_bakeoff.RowBakeoffConfig:
    raw = _config(fixture, repetitions=1).model_dump(mode="json")
    raw.update(
        {
            "schema_version": "row-bakeoff/0.2",
            "frame_mode": "contract_smoke",
            "bakeoff_id": "row-smoke",
        }
    )
    raw["cases"][0]["selected_row_ids"] = [row.row_id for row in fixture.selected]
    raw["cases"][0]["balanced_rows_per_class"] = None
    return row_bakeoff.RowBakeoffConfig.model_validate(raw)


def _scope() -> dict[str, Any]:
    return {
        "dataset_raw": "FIXTURE SET",
        "dataset_id": None,
        "dataset_url": None,
        "dataset_version": None,
        "split": None,
        "subset": None,
        "group": None,
        "language": None,
        "sample_count": None,
        "aggregation": None,
        "raw_scope": None,
    }


def test_provider_output_cannot_overwrite_sealed_row_input(
    frozen_rows: FrozenRows,
    tmp_path: Path,
) -> None:
    target = frozen_rows.project_root / "sealed-run" / "private" / "layout.json"
    before = target.read_bytes()

    with pytest.raises(ValueError, match="outside the sealed run"):
        row_bakeoff.run_row_bakeoff_provider_phase(
            _config(frozen_rows),
            project_root=frozen_rows.project_root,
            checkpoint_path=tmp_path / "row-checkpoint.json",
            output_path=target,
            client=object(),
        )

    assert target.read_bytes() == before


def _observation(row: EnumerationRow) -> dict[str, Any]:
    value = row.values[0].raw
    return {
        "claim_type": "primary_result",
        "roles": [
            {
                "role": "evaluated_system",
                "raw_name": row.row_label or "System",
                "version": None,
                "provider": None,
                "confidence": 0.99,
            }
        ],
        "scope": _scope(),
        "metric": {
            "raw_name": "X",
            "canonical_id": None,
            "kind": None,
            "unit": "proportion",
            "lower_is_better": False,
            "min_score": 0.0,
            "max_score": 1.0,
            "parameters": {},
        },
        "value": {
            "raw": value,
            "numeric": float(value),
            "unit": "proportion",
            "comparator": "exact",
            "uncertainty": None,
        },
        "evidence": [
            {
                "kind": "table",
                "label": "provider label replaced locally",
                "row": "provider row replaced locally",
                "column": "X",
                "quote": row.raw_text,
            }
        ],
        "extraction_confidence": 0.99,
        "construct": None,
        "operationalization": None,
        "decision_rule": None,
        "evaluation_date": None,
        "notes": [],
    }


def _row_ids(user: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r'"row_id": "(trow_[0-9a-f]+)"', user)))


class FakeRowClient:
    def __init__(
        self,
        rows: tuple[EnumerationRow, ...],
        *,
        contract_mismatch_models: set[str] | None = None,
        invalid_models: set[str] | None = None,
        scripts: list[Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
        budget_after: int | None = None,
    ) -> None:
        self.rows = {row.row_id: row for row in rows}
        self.contract_mismatch_models = contract_mismatch_models or set()
        self.invalid_models = invalid_models or set()
        self.scripts = list(scripts or [])
        self.budget_after = budget_after
        self.requests: list[dict[str, Any]] = []

    def _default_payload(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        dispositions = []
        for row_id in _row_ids(kwargs["user"]):
            row = self.rows[row_id]
            if kwargs["model"] == "vendor/risky" or row.row_label == "Cedar":
                disposition = "result"
            elif row.row_label == "Juniper":
                disposition = "not_result"
            else:
                disposition = "uncertain"
            dispositions.append(
                {
                    "row_id": row_id,
                    "disposition": disposition,
                    "observations": [_observation(row)] if disposition == "result" else [],
                    "note": None,
                }
            )
        return {"dispositions": dispositions, "warnings": []}

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        if self.budget_after is not None and len(self.requests) >= self.budget_after:
            raise ProviderBudgetExhausted(
                reason="structured_call_limit",
                summary={"structured_calls": len(self.requests)},
            )
        self.requests.append(dict(kwargs))
        assert kwargs["require_parameters"] is True
        assert _LABEL_SECRET not in kwargs["system"]
        assert _LABEL_SECRET not in kwargs["user"]
        payload = self.scripts.pop(0)(kwargs) if self.scripts else self._default_payload(kwargs)
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
            latency_seconds=0.25,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            cost_usd=0.002,
            request_id=f"request-{len(self.requests)}",
            finish_reason="stop",
            attempts=1,
        )
        if kwargs["model"] in self.invalid_models:
            raise ProviderResponseValidationError(call=call, code="schema_validation")
        return StructuredResponse(payload=payload, call=call)


class AlwaysFailRowClient:
    def structured_chat(self, **_kwargs: Any) -> StructuredResponse:
        raise RuntimeError("provider transport failed without call telemetry")


def _labels(path: Path, config: row_bakeoff.RowBakeoffConfig) -> None:
    case = config.cases[0]
    values = ("result", "not_result", "uncertain", "unsupported")
    write_jsonl(
        path,
        [
            row_bakeoff.RowDispositionLabel(
                case_id=case.case_id,
                row_id=row_id,
                row_plan_sha256=case.row_plan_sha256,
                annotator=_LABEL_SECRET,
                label=label,
            )
            for row_id, label in zip(case.selected_row_ids, values, strict=True)
        ],
    )


def _fixed_code_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        row_bakeoff,
        "_code_state",
        lambda root: {"commit": "d" * 40, "dirty": False, "root": Path(root).name},
    )


def test_strict_config_and_plan_drift_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: FrozenRows,
) -> None:
    _fixed_code_state(monkeypatch)
    raw = _config(frozen_rows).model_dump(mode="json")
    raw["require_parameters"] = False
    with pytest.raises(ValidationError):
        row_bakeoff.RowBakeoffConfig.model_validate(raw)
    raw = _config(frozen_rows).model_dump(mode="json")
    raw["cases"][0]["row_plan_sha256"] = "0" * 64
    drifted = row_bakeoff.RowBakeoffConfig.model_validate(raw)
    with pytest.raises(ValueError, match="row plan binding"):
        row_bakeoff.run_row_bakeoff_provider_phase(
            drifted,
            project_root=frozen_rows.project_root,
            checkpoint_path=tmp_path / "drift.json",
            client=FakeRowClient(frozen_rows.selected),
        )


def test_provider_phase_repeats_contract_failures_and_offline_exact_denominators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: FrozenRows,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_rows)
    labels = tmp_path / "private" / "labels.jsonl"
    labels.parent.mkdir(parents=True)
    _labels(labels, config)
    checkpoint = tmp_path / "private" / "checkpoint.json"
    client = FakeRowClient(
        frozen_rows.selected,
        contract_mismatch_models={"vendor/risky"},
    )
    provider = row_bakeoff.run_row_bakeoff_provider_phase(
        config,
        project_root=frozen_rows.project_root,
        checkpoint_path=checkpoint,
        client=client,
    )
    assert provider["privacy"]["labels_loaded_during_provider_phase"] is False
    assert provider["privacy"]["row_ids_in_public_result"] is False
    assert _LABEL_SECRET not in json.dumps(provider)
    assert provider["accounting"]["repetitions_predeclared"] == 4
    assert provider["accounting"]["repetitions_terminal"] == 4
    assert provider["accounting"]["selected_row_decisions"] == 16
    risky = provider["models"][1]
    assert risky["aggregate"]["contract"]["failed"] > 0
    assert risky["stability"]["row_state_comparison_denominator"] == 4

    score = row_bakeoff.score_row_bakeoff_offline(
        config,
        project_root=frozen_rows.project_root,
        checkpoint_path=checkpoint,
        labels_path=labels,
    )
    good = score["models"][0]["quality"]
    risky_score = score["models"][1]
    assert next(iter(good)) == "unsafe_result_false_positives"
    assert good["unsafe_result_false_positives"]["count"] == 0
    assert good["exact_accounting"]["selected_row_decisions"] == 8
    assert good["per_class"]["result"]["recall"] == {
        "numerator": 2,
        "denominator": 2,
        "value": 1.0,
    }
    assert risky_score["quality_status"] == "unmeasured_incomplete_provider_contract"
    assert risky_score["quality"] is None
    assert risky_score["quality_measurement"] == {
        "status": "unmeasured",
        "reason": "one_or_more_repetitions_lacked_valid_contract_observation",
        "repetitions_predeclared": 2,
        "repetitions_contract_complete": 0,
        "repetitions_contract_incomplete": 2,
    }
    assert risky_score["terminal_telemetry"]["terminal_state_counts"] == {
        "result": 0,
        "not_result": 0,
        "uncertain": 0,
        "unresolved": 6,
        "unsupported": 2,
    }


def test_unknown_duplicate_and_omitted_ids_recover_without_losing_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: FrozenRows,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_rows, repetitions=1)

    def malformed(kwargs: dict[str, Any]) -> dict[str, Any]:
        ids = _row_ids(kwargs["user"])
        assert len(ids) == 2
        row = frozen_rows.selected[0]
        return {
            "dispositions": [
                {
                    "row_id": ids[0],
                    "disposition": "result",
                    "observations": [_observation(row)],
                    "note": None,
                },
                {
                    "row_id": ids[0],
                    "disposition": "result",
                    "observations": [_observation(row)],
                    "note": None,
                },
                {
                    "row_id": "unknown-row-id",
                    "disposition": "not_result",
                    "observations": [],
                    "note": None,
                },
            ],
            "warnings": [],
        }

    client = FakeRowClient(frozen_rows.selected, scripts=[malformed])
    checkpoint_path = tmp_path / "recover.json"
    provider = row_bakeoff.run_row_bakeoff_provider_phase(
        config,
        project_root=frozen_rows.project_root,
        checkpoint_path=checkpoint_path,
        client=client,
    )
    first = provider["attempts"][0]
    assert first["status"] == "partial_invalid"
    assert first["unknown_row_id_count"] == 1
    assert first["unresolved_row_count"] == 2
    assert provider["accounting"]["selected_row_decisions"] == 8
    checkpoint = read_json(checkpoint_path)
    assert all(len(item["terminal_by_row"]) == 4 for item in checkpoint["repetitions"].values())
    labels_path = tmp_path / "recovery-labels.jsonl"
    _labels(labels_path, config)
    score = row_bakeoff.score_row_bakeoff_offline(
        config,
        project_root=frozen_rows.project_root,
        checkpoint_path=checkpoint_path,
        labels_path=labels_path,
    )
    assert all(model["quality_status"] == "measured" for model in score["models"])
    assert all(model["quality"] is not None for model in score["models"])


def test_schema_failure_is_unmeasured_not_an_unresolved_row_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: FrozenRows,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_rows, repetitions=1)
    checkpoint_path = tmp_path / "schema-failure.json"
    row_bakeoff.run_row_bakeoff_provider_phase(
        config,
        project_root=frozen_rows.project_root,
        checkpoint_path=checkpoint_path,
        client=FakeRowClient(
            frozen_rows.selected,
            invalid_models={"vendor/risky"},
        ),
    )
    labels_path = tmp_path / "labels.jsonl"
    _labels(labels_path, config)
    score = row_bakeoff.score_row_bakeoff_offline(
        config,
        project_root=frozen_rows.project_root,
        checkpoint_path=checkpoint_path,
        labels_path=labels_path,
    )
    risky = score["models"][1]
    assert risky["quality_status"] == "unmeasured_incomplete_provider_contract"
    assert risky["quality"] is None
    assert risky["terminal_telemetry"]["attempts"]["schema"]["all_repetitions"]["invalid"] > 0
    assert "confusion" not in risky


def test_provider_failures_are_missing_from_usage_lower_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: FrozenRows,
) -> None:
    _fixed_code_state(monkeypatch)
    result = row_bakeoff.run_row_bakeoff_provider_phase(
        _config(frozen_rows, repetitions=1),
        project_root=frozen_rows.project_root,
        checkpoint_path=tmp_path / "provider-failures.json",
        client=AlwaysFailRowClient(),
    )
    aggregate = result["aggregate"]
    logical_attempts = aggregate["execution"]["logical_attempts_terminal"]
    assert logical_attempts > 0
    assert aggregate["execution"]["provider_failure"] == logical_attempts
    for field in ("total_tokens", "cost_usd"):
        assert aggregate["usage"][field] == {
            "total": 0,
            "reported_calls": 0,
            "missing_calls": logical_attempts,
        }


def test_budget_stop_checkpoints_only_completed_attempts_then_resumes_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: FrozenRows,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_rows, repetitions=1)
    checkpoint = tmp_path / "budget-stop.json"
    with pytest.raises(ProviderBudgetExhausted):
        row_bakeoff.run_row_bakeoff_provider_phase(
            config,
            project_root=frozen_rows.project_root,
            checkpoint_path=checkpoint,
            client=FakeRowClient(frozen_rows.selected, budget_after=1),
        )
    stopped = read_json(checkpoint)
    assert stopped["status"] == "in_progress"
    assert len(stopped["attempts"]) == 1
    assert stopped["repetitions"] == {}

    resumed_client = FakeRowClient(frozen_rows.selected)
    resumed = row_bakeoff.run_row_bakeoff_provider_phase(
        config,
        project_root=frozen_rows.project_root,
        checkpoint_path=checkpoint,
        client=resumed_client,
    )
    assert resumed["resume"]["logical_attempts_reused"] == 1
    assert resumed["resume"]["logical_attempts_executed"] == len(resumed_client.requests)
    sealed_checkpoint_bytes = checkpoint.read_bytes()
    no_calls = FakeRowClient(frozen_rows.selected)
    fully_resumed = row_bakeoff.run_row_bakeoff_provider_phase(
        config,
        project_root=frozen_rows.project_root,
        checkpoint_path=checkpoint,
        client=no_calls,
    )
    assert no_calls.requests == []
    assert fully_resumed["resume"]["logical_attempts_executed"] == 0
    assert checkpoint.read_bytes() == sealed_checkpoint_bytes


def test_tampered_checkpoint_is_rejected_and_labels_require_a_sealed_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: FrozenRows,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_rows, repetitions=1)
    checkpoint = tmp_path / "tamper.json"
    row_bakeoff.run_row_bakeoff_provider_phase(
        config,
        project_root=frozen_rows.project_root,
        checkpoint_path=checkpoint,
        client=FakeRowClient(frozen_rows.selected),
    )
    raw = read_json(checkpoint)
    first = next(iter(raw["attempts"].values()))
    first["request"]["settings"]["max_tokens"] = 12
    write_json(checkpoint, raw)
    rebuilt_client = FakeRowClient(frozen_rows.selected)
    rebuilt = row_bakeoff.run_row_bakeoff_provider_phase(
        config,
        project_root=frozen_rows.project_root,
        checkpoint_path=checkpoint,
        client=rebuilt_client,
    )
    assert rebuilt["resume"]["checkpoint_rejections"] == ["checkpoint_invalid"]
    assert rebuilt_client.requests

    labels = tmp_path / "labels.jsonl"
    _labels(labels, config)
    raw = read_json(checkpoint)
    raw["status"] = "in_progress"
    raw["provider_phase_seal_sha256"] = None
    raw_without_hash = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    raw["checkpoint_sha256"] = row_bakeoff._hash(raw_without_hash)
    write_json(checkpoint, raw)
    with pytest.raises(ValueError, match="not sealed"):
        row_bakeoff.score_row_bakeoff_offline(
            config,
            project_root=frozen_rows.project_root,
            checkpoint_path=checkpoint,
            labels_path=labels,
        )


def test_contract_smoke_frame_cannot_claim_balance_or_be_offline_scored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: FrozenRows,
) -> None:
    _fixed_code_state(monkeypatch)
    raw = _config(frozen_rows, repetitions=1).model_dump(mode="json")
    raw.update({"schema_version": "row-bakeoff/0.2", "frame_mode": "contract_smoke"})
    with pytest.raises(ValidationError, match="cannot claim class balance"):
        row_bakeoff.RowBakeoffConfig.model_validate(raw)

    raw = _config(frozen_rows).model_dump(mode="json")
    raw["schema_version"] = "row-bakeoff/0.2"
    raw["cases"][0]["balanced_rows_per_class"] = None
    with pytest.raises(ValidationError, match="require a class-balance declaration"):
        row_bakeoff.RowBakeoffConfig.model_validate(raw)

    smoke = _smoke_config(frozen_rows)
    assert smoke.cases[0].balanced_rows_per_class is None
    label_calls: list[str] = []
    monkeypatch.setattr(
        row_bakeoff,
        "_load_labels",
        lambda *_args, **_kwargs: label_calls.append("loaded"),
    )
    with pytest.raises(ValueError, match="balanced-quality frame"):
        row_bakeoff.score_row_bakeoff_offline(
            smoke,
            project_root=frozen_rows.project_root,
            checkpoint_path=tmp_path / "missing-checkpoint.json",
            labels_path=tmp_path / "missing-labels.jsonl",
        )
    assert label_calls == []


def test_repair04_row_config_defaults_and_locks_the_production_contract(
    frozen_rows: FrozenRows,
) -> None:
    raw = _config(frozen_rows).model_dump(mode="json")
    raw["schema_version"] = "row-bakeoff/0.3"
    for field in ("max_tokens", "temperature", "reasoning_effort", "seed"):
        raw.pop(field)

    config = row_bakeoff.RowBakeoffConfig.model_validate(raw)

    assert config.max_tokens == 16_000
    assert config.temperature is None
    assert config.reasoning_effort == "minimal"
    assert config.seed is None
    assert config.require_parameters is True

    raw["max_tokens"] = 4_000
    with pytest.raises(ValidationError, match="production request contract"):
        row_bakeoff.RowBakeoffConfig.model_validate(raw)


def test_sealed_row_smoke_uses_only_first_pass_base_attempts_and_selects_full_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: FrozenRows,
) -> None:
    _fixed_code_state(monkeypatch)
    smoke = _smoke_config(frozen_rows)

    def malformed_first_base(kwargs: dict[str, Any]) -> dict[str, Any]:
        row_ids = _row_ids(kwargs["user"])
        row = frozen_rows.selected[0]
        return {
            "dispositions": [
                {
                    "row_id": row_ids[0],
                    "disposition": "result",
                    "observations": [_observation(row)],
                    "note": None,
                },
                {
                    "row_id": row_ids[0],
                    "disposition": "result",
                    "observations": [_observation(row)],
                    "note": None,
                },
                {
                    "row_id": "unknown-row-id",
                    "disposition": "not_result",
                    "observations": [],
                    "note": None,
                },
            ],
            "warnings": [],
        }

    smoke_checkpoint = tmp_path / "smoke-checkpoint.json"
    smoke_provider = row_bakeoff.run_row_bakeoff_provider_phase(
        smoke,
        project_root=frozen_rows.project_root,
        checkpoint_path=smoke_checkpoint,
        client=FakeRowClient(
            frozen_rows.selected,
            invalid_models={"vendor/risky"},
            scripts=[malformed_first_base],
        ),
    )
    base_attempts_per_model = smoke_provider["source"]["cases"][0]["base_batches_per_repetition"]
    assert base_attempts_per_model == 2
    assert (
        smoke_provider["source"]["cases"][0]["maximum_attempts_per_repetition"]
        == base_attempts_per_model
    )
    assert smoke_provider["accounting"]["logical_attempts_terminal"] == (
        len(smoke.models) * base_attempts_per_model
    )
    assert len(read_json(smoke_checkpoint)["attempts"]) == (
        len(smoke.models) * base_attempts_per_model
    )
    assert {attempt["depth"] for attempt in smoke_provider["attempts"]} == {0}
    assert "normalized_wire_proposal" not in json.dumps(smoke_provider)
    private_smoke = read_json(smoke_checkpoint)
    successful = next(
        entry for entry in private_smoke["attempts"].values() if entry["schema_status"] == "valid"
    )
    assert set(successful["normalized_wire_proposal"]["structured_content"]) == {
        "dispositions",
        "warnings",
    }

    selection = row_bakeoff.derive_row_execution_selection(
        smoke,
        project_root=frozen_rows.project_root,
        checkpoint_path=smoke_checkpoint,
        experiment_id="staged-model-comparison-2026-09-02",
        public_manifest_sha256="a" * 64,
    )
    assert selection.stage == "row_disposition"
    assert selection.eligible_models == ("vendor/good",)
    good_decision = selection.decision_for("vendor/good")
    assert good_decision.logical_calls_predeclared == base_attempts_per_model
    assert good_decision.terminal_entries == base_attempts_per_model
    assert good_decision.first_pass_schema_valid == base_attempts_per_model
    risky_decision = selection.decision_for("vendor/risky")
    assert risky_decision.status == "contract_ineligible"
    assert risky_decision.reason_codes == ["wire_schema_below_gate"]
    expectation = _selection_expectation(selection)

    full = _config(frozen_rows)
    full_checkpoint = tmp_path / "full-checkpoint.json"
    full_client = FakeRowClient(frozen_rows.selected)
    provider = row_bakeoff.run_row_bakeoff_provider_phase(
        full,
        project_root=frozen_rows.project_root,
        checkpoint_path=full_checkpoint,
        client=full_client,
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert {request["model"] for request in full_client.requests} == {"vendor/good"}
    assert provider["declared_models"] == ["vendor/good", "vendor/risky"]
    assert provider["executed_models"] == ["vendor/good"]
    assert provider["execution_selection"]["selection_sha256"] == (selection.selection_sha256)
    executed, skipped = provider["models"]
    assert executed["matched_quality_status"] == "executed"
    assert executed["quality_status"] == "pending_offline_scoring"
    assert executed["quality"] is None
    assert skipped["matched_quality_status"] == "not_run"
    assert skipped["not_run_reason"] == "contract_ineligible_on_sealed_smoke"
    assert skipped["quality_status"] == "unmeasured_contract_ineligible"
    assert skipped["quality"] is None
    assert {entry["model"] for entry in read_json(full_checkpoint)["attempts"].values()} == {
        "vendor/good"
    }

    labels_path = tmp_path / "labels.jsonl"
    _labels(labels_path, full)
    score = row_bakeoff.score_row_bakeoff_offline(
        full,
        project_root=frozen_rows.project_root,
        checkpoint_path=full_checkpoint,
        labels_path=labels_path,
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert score["execution_selection_sha256"] == selection.selection_sha256
    assert score["models"][0]["quality_status"] == "measured"
    assert score["models"][0]["quality"] is not None
    assert score["models"][1]["matched_quality_status"] == "not_run"
    assert score["models"][1]["quality_status"] == "unmeasured_contract_ineligible"
    assert score["models"][1]["quality"] is None

    real_load_labels = row_bakeoff._load_labels
    label_calls: list[str] = []

    def tracked_labels(*args: Any, **kwargs: Any) -> Any:
        label_calls.append("loaded")
        return real_load_labels(*args, **kwargs)

    monkeypatch.setattr(row_bakeoff, "_load_labels", tracked_labels)
    with pytest.raises(ValueError, match="checkpoint is stale"):
        row_bakeoff.score_row_bakeoff_offline(
            full,
            project_root=frozen_rows.project_root,
            checkpoint_path=full_checkpoint,
            labels_path=labels_path,
        )
    assert label_calls == []


def test_row_selection_tamper_and_staleness_fail_before_calls_or_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: FrozenRows,
) -> None:
    _fixed_code_state(monkeypatch)
    smoke = _smoke_config(frozen_rows)
    smoke_checkpoint = tmp_path / "smoke.json"
    row_bakeoff.run_row_bakeoff_provider_phase(
        smoke,
        project_root=frozen_rows.project_root,
        checkpoint_path=smoke_checkpoint,
        client=FakeRowClient(
            frozen_rows.selected,
            invalid_models={"vendor/risky"},
        ),
    )
    selection = row_bakeoff.derive_row_execution_selection(
        smoke,
        project_root=frozen_rows.project_root,
        checkpoint_path=smoke_checkpoint,
        experiment_id="row-selection-integrity",
        public_manifest_sha256="a" * 64,
    )
    expectation = _selection_expectation(selection)
    full = _config(frozen_rows)

    tampered = selection.model_copy(update={"stage": "origin_retrieval"})
    tampered_client = FakeRowClient(frozen_rows.selected)
    with pytest.raises(ValidationError, match="selection hash is invalid"):
        row_bakeoff.run_row_bakeoff_provider_phase(
            full,
            project_root=frozen_rows.project_root,
            checkpoint_path=tmp_path / "tampered.json",
            client=tampered_client,
            execution_selection=tampered,
            execution_selection_expectation=expectation,
        )
    assert tampered_client.requests == []

    stale_payload = selection.model_dump(mode="json", exclude={"selection_sha256"})
    stale_payload["stage_contract_sha256"] = "0" * 64
    stale = row_bakeoff.StageExecutionSelection.model_validate(
        stale_payload | {"selection_sha256": row_bakeoff._hash(stale_payload)}
    )
    stale_client = FakeRowClient(frozen_rows.selected)
    with pytest.raises(ValueError, match="stale stage contract"):
        row_bakeoff.run_row_bakeoff_provider_phase(
            full,
            project_root=frozen_rows.project_root,
            checkpoint_path=tmp_path / "stale.json",
            client=stale_client,
            execution_selection=stale,
            execution_selection_expectation=expectation,
        )
    assert stale_client.requests == []

    label_calls: list[str] = []
    monkeypatch.setattr(
        row_bakeoff,
        "_load_labels",
        lambda *_args, **_kwargs: label_calls.append("loaded"),
    )
    with pytest.raises(ValueError, match="stale stage contract"):
        row_bakeoff.score_row_bakeoff_offline(
            full,
            project_root=frozen_rows.project_root,
            checkpoint_path=tmp_path / "missing.json",
            labels_path=tmp_path / "missing-labels.jsonl",
            execution_selection=stale,
            execution_selection_expectation=expectation,
        )
    assert label_calls == []

    wrong_gate = expectation.model_copy(update={"wire_schema_gate": "0"})
    wrong_gate_client = FakeRowClient(frozen_rows.selected)
    with pytest.raises(ValueError, match="another wire-schema gate"):
        row_bakeoff.run_row_bakeoff_provider_phase(
            full,
            project_root=frozen_rows.project_root,
            checkpoint_path=tmp_path / "wrong-gate.json",
            client=wrong_gate_client,
            execution_selection=selection,
            execution_selection_expectation=wrong_gate,
        )
    assert wrong_gate_client.requests == []


def test_fully_rehashed_cross_call_row_record_splice_fails_before_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: FrozenRows,
) -> None:
    _fixed_code_state(monkeypatch)
    config = _config(frozen_rows, repetitions=1)
    checkpoint_path = tmp_path / "checkpoint.json"
    row_bakeoff.run_row_bakeoff_provider_phase(
        config,
        project_root=frozen_rows.project_root,
        checkpoint_path=checkpoint_path,
        client=FakeRowClient(frozen_rows.selected),
    )
    raw = read_json(checkpoint_path)
    attempts = list(raw["attempts"].values())
    target = next(
        entry for entry in attempts if entry["model"] == "vendor/good" and entry["depth"] == 0
    )
    donor = next(
        entry
        for entry in attempts
        if entry["model"] == "vendor/risky"
        and entry["depth"] == 0
        and entry["base_batch_id"] == target["base_batch_id"]
    )
    assert target["call"]["response_sha256"] != donor["call"]["response_sha256"]
    for field in (
        "status",
        "records",
        "unresolved_row_ids",
        "unknown_row_ids",
        "invalid_row_reasons",
        "warnings",
    ):
        target[field] = donor[field]
    target_unsigned = {key: value for key, value in target.items() if key != "entry_sha256"}
    target["entry_sha256"] = row_bakeoff._hash(target_unsigned)
    raw["provider_phase_seal_sha256"] = row_bakeoff._hash(
        {
            "schema_version": "row-bakeoff-provider-seal/0.1",
            "configuration_sha256": raw["configuration_sha256"],
            "run_contract_sha256": raw["run_contract_sha256"],
            "attempt_sha256s": {
                key: value["entry_sha256"] for key, value in sorted(raw["attempts"].items())
            },
            "repetition_sha256s": {
                key: value["repetition_sha256"] for key, value in sorted(raw["repetitions"].items())
            },
        }
    )
    checkpoint_unsigned = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    raw["checkpoint_sha256"] = row_bakeoff._hash(checkpoint_unsigned)
    write_json(checkpoint_path, raw)

    label_calls: list[str] = []
    monkeypatch.setattr(
        row_bakeoff,
        "_load_labels",
        lambda *_args, **_kwargs: label_calls.append("loaded"),
    )
    with pytest.raises(ValueError, match="invalid base attempt"):
        row_bakeoff.score_row_bakeoff_offline(
            config,
            project_root=frozen_rows.project_root,
            checkpoint_path=checkpoint_path,
            labels_path=tmp_path / "labels.jsonl",
        )
    assert label_calls == []
