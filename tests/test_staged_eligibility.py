from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from proceedings_to_eee.evaluation import staged_eligibility
from proceedings_to_eee.evaluation.staged_eligibility import (
    StageExecutionSelection,
    StageExecutionSelectionExpectation,
    derive_stage_execution_selection,
    selected_models_for_execution,
    selection_expectation_from_public_manifest,
)


def _entry(
    model: str,
    *,
    schema_status: str = "valid",
    contract_status: str = "satisfied",
    call_observed: bool = True,
) -> dict[str, Any]:
    call = None
    if call_observed:
        call = {
            "model_requested": model,
            "model_returned": model,
            "provider_returned": "test-provider",
            "schema_strict": True,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
        }
    return {
        "model": model,
        "repetition_index": 1,
        "schema_status": schema_status,
        "contract_status": contract_status,
        "call": call,
    }


def _selection(entries: list[dict[str, Any]]) -> StageExecutionSelection:
    return derive_stage_execution_selection(
        experiment_id="frozen-experiment",
        stage="origin_retrieval",
        public_manifest_sha256="a" * 64,
        stage_contract_sha256="b" * 64,
        smoke_bakeoff_id="origin-smoke",
        smoke_configuration_sha256="c" * 64,
        smoke_run_contract_sha256="d" * 64,
        smoke_code_sha256="e" * 64,
        smoke_provider_phase_seal_sha256="f" * 64,
        smoke_checkpoint_sha256="1" * 64,
        wire_schema_gate="0.99",
        declared_models=["vendor/eligible", "vendor/ineligible"],
        expected_slots_per_model=1,
        entries=entries,
    )


def _expectation(
    *,
    experiment_id: str = "frozen-experiment",
    public_manifest_sha256: str = "a" * 64,
    wire_schema_gate: str = "0.99",
) -> StageExecutionSelectionExpectation:
    return StageExecutionSelectionExpectation(
        experiment_id=experiment_id,
        public_manifest_sha256=public_manifest_sha256,
        wire_schema_gate=wire_schema_gate,
    )


def test_selection_uses_exact_sealed_entry_denominators_and_is_self_hashed() -> None:
    selection = _selection(
        [
            _entry("vendor/eligible"),
            _entry("vendor/ineligible", schema_status="invalid"),
        ]
    )
    assert selection.eligible_models == ("vendor/eligible",)
    failed = selection.decision_for("vendor/ineligible")
    assert failed.status == "contract_ineligible"
    assert failed.reason_codes == ["wire_schema_below_gate"]
    assert failed.first_pass_schema_invalid == 1

    tampered = selection.model_dump(mode="json")
    tampered["models"][0]["status"] = "contract_ineligible"
    tampered["models"][0]["reason_codes"] = ["wire_schema_below_gate"]
    with pytest.raises(ValidationError, match="exact counts and gate"):
        StageExecutionSelection.model_validate(tampered)


def test_not_observed_contract_and_telemetry_cannot_be_misread_as_quality_zero() -> None:
    selection = _selection(
        [
            _entry("vendor/eligible"),
            _entry(
                "vendor/ineligible",
                schema_status="not_observed",
                contract_status="not_observed",
                call_observed=False,
            ),
        ]
    )
    failed = selection.decision_for("vendor/ineligible")
    assert failed.status == "contract_ineligible"
    assert failed.reason_codes == [
        "wire_schema_below_gate",
        "request_contract_unsatisfied",
        "telemetry_incomplete",
    ]
    assert not hasattr(failed, "quality")


def test_returned_model_mismatch_cannot_be_technically_eligible() -> None:
    mismatched = _entry("vendor/eligible")
    mismatched["call"]["model_returned"] = "vendor/different-model"
    selection = _selection([mismatched, _entry("vendor/ineligible")])

    decision = selection.decision_for("vendor/eligible")
    assert decision.status == "contract_ineligible"
    assert decision.reason_codes == ["telemetry_incomplete"]


def test_selection_is_rejected_when_stage_contract_code_or_models_are_stale() -> None:
    selection = _selection([_entry("vendor/eligible"), _entry("vendor/ineligible")])
    assert selected_models_for_execution(
        selection,
        expected_stage="origin_retrieval",
        declared_models=["vendor/eligible", "vendor/ineligible"],
        stage_contract_sha256="b" * 64,
        code_sha256="e" * 64,
        expectation=_expectation(),
    ) == ("vendor/eligible", "vendor/ineligible")

    with pytest.raises(ValueError, match="stale stage contract"):
        selected_models_for_execution(
            selection,
            expected_stage="origin_retrieval",
            declared_models=["vendor/eligible", "vendor/ineligible"],
            stage_contract_sha256="0" * 64,
            code_sha256="e" * 64,
            expectation=_expectation(),
        )
    with pytest.raises(ValueError, match="different code"):
        selected_models_for_execution(
            selection,
            expected_stage="origin_retrieval",
            declared_models=["vendor/eligible", "vendor/ineligible"],
            stage_contract_sha256="b" * 64,
            code_sha256="0" * 64,
            expectation=_expectation(),
        )
    with pytest.raises(ValueError, match="stale model declaration"):
        selected_models_for_execution(
            selection,
            expected_stage="origin_retrieval",
            declared_models=["vendor/eligible", "vendor/replacement"],
            stage_contract_sha256="b" * 64,
            code_sha256="e" * 64,
            expectation=_expectation(),
        )


def test_selection_rejects_self_hashed_eligibility_that_contradicts_counts() -> None:
    selection = _selection([_entry("vendor/eligible"), _entry("vendor/ineligible")])
    tampered = selection.model_dump(mode="json")
    decision = tampered["models"][0]
    decision["contract_satisfied"] = 0
    decision["contract_failed"] = 1
    decision["status"] = "contract_eligible"
    decision["reason_codes"] = []
    payload = {key: value for key, value in tampered.items() if key != "selection_sha256"}
    tampered["selection_sha256"] = staged_eligibility._hash(payload)  # noqa: SLF001

    with pytest.raises(ValidationError, match="differs from its exact counts and gate"):
        StageExecutionSelection.model_validate(tampered)


@pytest.mark.parametrize(
    ("expectation", "message"),
    [
        (_expectation(experiment_id="another-experiment"), "another experiment"),
        (_expectation(public_manifest_sha256="0" * 64), "another public manifest"),
        (_expectation(wire_schema_gate="0"), "another wire-schema gate"),
    ],
)
def test_selection_requires_exact_experiment_manifest_and_gate(
    expectation: StageExecutionSelectionExpectation,
    message: str,
) -> None:
    selection = _selection([_entry("vendor/eligible"), _entry("vendor/ineligible")])
    with pytest.raises(ValueError, match=message):
        selected_models_for_execution(
            selection,
            expected_stage="origin_retrieval",
            declared_models=["vendor/eligible", "vendor/ineligible"],
            stage_contract_sha256="b" * 64,
            code_sha256="e" * 64,
            expectation=expectation,
        )


def test_selection_expectation_hashes_exact_amendment_bytes(tmp_path) -> None:
    amendment = tmp_path / "amendment.json"
    amendment.write_text(
        '{"amendment_id":"repair-04","technical_qualification":{"wire_schema_gate":"0.99"}}\n',
        encoding="utf-8",
    )

    expectation = selection_expectation_from_public_manifest(
        experiment_id="frozen-experiment",
        expected_amendment_id="repair-04",
        public_manifest_path=amendment,
    )

    assert expectation.wire_schema_gate == "0.99"
    assert expectation.public_manifest_sha256 == staged_eligibility.sha256_bytes(
        amendment.read_bytes()
    )

    with pytest.raises(ValueError, match="amendment ID is unexpected"):
        selection_expectation_from_public_manifest(
            experiment_id="frozen-experiment",
            expected_amendment_id="another-repair",
            public_manifest_path=amendment,
        )
