from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from proceedings_to_eee.evaluation import bakeoff
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.result_blocks import segment_page_result_blocks
from proceedings_to_eee.providers.budget import ProviderBudgetExhausted
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderResponseValidationError,
    StructuredResponse,
    completion_token_parameter_for_model,
    structured_request_contract,
)
from proceedings_to_eee.reference import AnnotationCoverage, PaperReference
from proceedings_to_eee.sources.manifest import SourceManifest


def _config(**overrides: Any) -> bakeoff.ExtractorBakeoffConfig:
    raw: dict[str, Any] = {
        "bakeoff_id": "contract-test",
        "models": [
            {"model": "vendor/model-a", "label": "A"},
            {"model": "vendor/model-b", "label": "B"},
        ],
        "cases": [
            {
                "case_id": "case-1",
                "paper_id": "synthetic-audit-study",
                "page": 1,
                "manifest_path": "private/manifest.json",
                "reference_path": "private/reference.yaml",
            }
        ],
    }
    raw.update(overrides)
    return bakeoff.ExtractorBakeoffConfig.model_validate(raw)


def _selection_expectation(
    selection: bakeoff.StageExecutionSelection,
) -> bakeoff.StageExecutionSelectionExpectation:
    return bakeoff.StageExecutionSelectionExpectation(
        experiment_id=selection.experiment_id,
        public_manifest_sha256=selection.public_manifest_sha256,
        wire_schema_gate=selection.wire_schema_gate,
    )


def _preparation(manifest: SourceManifest) -> bakeoff._Preparation:
    text = "Table 1: Results\nModel     Accuracy\nAtlas     90.0%\nBoreal    80.0%\n"
    page = PageFragment(
        fragment_id="frag-1",
        source_id="src_paper",
        page=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=2,
        result_signal_score=5.0,
    )
    blocks = tuple(segment_page_result_blocks(page))
    assert len(blocks) == 1
    spec = bakeoff.BakeoffCaseSpec(
        case_id="case-1",
        paper_id=manifest.paper_id,
        page=1,
        manifest_path="private/manifest.json",
        reference_path="private/reference.yaml",
    )
    prepared = bakeoff._PreparedCase(
        spec=spec,
        manifest=manifest,
        reference_path=Path("unused-reference.yaml"),
        layout=PdfLayout(
            source_id="src_paper",
            parser="test-parser",
            parser_version="1",
            page_count=1,
            pages=[page],
        ),
        blocks=blocks,
        source_id="src_paper",
        source_sha256="a" * 64,
        manifest_sha256="b" * 64,
        reference_sha256="c" * 64,
    )
    return bakeoff._Preparation(
        spec=spec,
        prepared=prepared,
        public_result={
            "case_id": spec.case_id,
            "paper_id": spec.paper_id,
            "page": spec.page,
            "status": "success",
        },
    )


def _reference() -> PaperReference:
    return PaperReference(
        paper_id="synthetic-audit-study",
        source_sha256="a" * 64,
        annotation_protocol="test-only",
        annotation_status="complete",
        coverage=AnnotationCoverage(
            inclusion_rule="none",
            exclusion_rule="none",
        ),
        evidence=[],
        observations=[],
    )


def _reference_with_positive_and_control() -> PaperReference:
    return PaperReference.model_validate(
        {
            "paper_id": "synthetic-audit-study",
            "source_sha256": "a" * 64,
            "annotation_protocol": "test-only",
            "annotation_status": "complete",
            "coverage": {
                "fully_annotated_labels": ["Table 1"],
                "inclusion_rule": "fixture",
                "exclusion_rule": "fixture",
            },
            "evidence": [
                {
                    "evidence_id": "ev-atlas",
                    "purpose": "result",
                    "page": 1,
                    "kind": "table",
                    "label": "Table 1",
                    "row": "Atlas",
                    "column": "Accuracy",
                    "exact_quote": "Atlas     90.0%",
                },
                {
                    "evidence_id": "ev-boreal",
                    "purpose": "negative_control",
                    "page": 1,
                    "kind": "table",
                    "label": "Table 1",
                    "row": "Boreal",
                    "column": "Accuracy",
                    "exact_quote": "Boreal    80.0%",
                },
            ],
            "observations": [
                {
                    "reference_id": "ref-atlas",
                    "claim_type": "primary_result",
                    "actors": [{"role": "evaluated_system", "raw_name": "Atlas"}],
                    "scope": {"dataset_raw": "Synthetic benchmark"},
                    "metric": {"raw_name": "Accuracy", "unit": "%"},
                    "value": {"raw": "90.0%", "numeric": 90.0, "unit": "%"},
                    "result_evidence_ids": ["ev-atlas"],
                }
            ],
            "negative_controls": [
                {
                    "control_id": "nc-boreal",
                    "expected_claim_type": "secondary_claim",
                    "evidence_ids": ["ev-boreal"],
                    "reason_not_primary": "Fixture comparator.",
                }
            ],
        }
    )


class FakeClient:
    def __init__(self, behavior: dict[str, str] | None = None) -> None:
        self.behavior = behavior or {}
        self.requests: list[dict[str, Any]] = []

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        self.requests.append(dict(kwargs))
        requested_require = kwargs["require_parameters"]
        actual_require = (
            False
            if self.behavior.get(kwargs["model"]) == "contract_mismatch"
            else requested_require
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
            response_sha256=hashlib.sha256(b"response").hexdigest(),
            temperature=kwargs["temperature"],
            reasoning_effort=kwargs["reasoning_effort"],
            max_tokens=kwargs["max_tokens"],
            completion_token_parameter=completion_token_parameter_for_model(kwargs["model"]),
            seed=kwargs["seed"],
            schema_name=kwargs["schema_name"],
            schema_sha256=contract["schema"]["schema_sha256"],
            require_parameters=actual_require,
            latency_seconds=0.25,
            input_tokens=10,
            output_tokens=5,
            reasoning_tokens=2,
            total_tokens=15,
            cost_usd=0.001,
            request_id=f"request-{len(self.requests)}",
            finish_reason="stop",
            attempts=1,
        )
        behavior = self.behavior.get(kwargs["model"])
        if behavior == "invalid_schema":
            raise ProviderResponseValidationError(call=call, code="schema_validation")
        if behavior == "not_observed":
            raise RuntimeError("transport failed")
        return StructuredResponse(
            payload={"observations": [], "page_summary": "No result.", "warnings": []},
            call=call,
        )


class BoundCandidateClient(FakeClient):
    """Return one candidate whose provenance is bound to a model-specific response."""

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        base = super().structured_chat(**kwargs)
        payload = {
            "observations": [
                {
                    "claim_type": "primary_result",
                    "roles": [
                        {
                            "role": "evaluated_system",
                            "raw_name": "Atlas",
                            "version": None,
                            "provider": None,
                            "confidence": 0.99,
                        }
                    ],
                    "scope": {
                        "dataset_raw": "Synthetic benchmark",
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
                    },
                    "metric": {
                        "raw_name": "Accuracy",
                        "canonical_id": None,
                        "kind": None,
                        "unit": "%",
                        "lower_is_better": False,
                        "min_score": 0.0,
                        "max_score": 100.0,
                        "parameters": {},
                    },
                    "value": {
                        "raw": "90.0%",
                        "numeric": 90.0,
                        "unit": "%",
                        "comparator": "exact",
                        "uncertainty": None,
                    },
                    "evidence": [
                        {
                            "kind": "table",
                            "label": "Table 1",
                            "row": "Atlas",
                            "column": "Accuracy",
                            "quote": "Atlas     90.0%",
                        }
                    ],
                    "extraction_confidence": 0.99,
                    "construct": None,
                    "operationalization": None,
                    "decision_rule": None,
                    "evaluation_date": None,
                    "notes": [],
                }
            ],
            "page_summary": f"One result returned by {kwargs['model']}.",
            "warnings": [],
        }
        response_sha256 = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        return StructuredResponse(
            payload=payload,
            call=base.call.model_copy(update={"response_sha256": response_sha256}),
        )


def test_reference_examination_uses_only_successful_provider_blocks(
    manifest: SourceManifest,
) -> None:
    preparation = _preparation(manifest)
    assert preparation.prepared is not None
    block_id = preparation.prepared.blocks[0].block_id
    successful = bakeoff._ExtractionRun(
        public_result={"calls": [{"block_id": block_id, "status": "success"}]},
        candidates=(),
        candidate_fingerprints=(),
        scoring_eligible=True,
    )
    failed = bakeoff._ExtractionRun(
        public_result={"calls": [{"block_id": block_id, "status": "error"}]},
        candidates=(),
        candidate_fingerprints=(),
        scoring_eligible=False,
    )
    reference = _reference_with_positive_and_control()

    assert bakeoff._run_reference_examination(reference, preparation.prepared, successful) == (
        {"nc-boreal": True},
        {"ref-atlas": True},
    )
    assert bakeoff._run_reference_examination(reference, preparation.prepared, failed) == (
        {"nc-boreal": False},
        {"ref-atlas": False},
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
    config: bakeoff.ExtractorBakeoffConfig,
    client: FakeClient,
) -> dict[str, Any]:
    preparation = _preparation(manifest)
    expected_calls = len(config.models) * config.fresh_repetitions
    monkeypatch.setattr(bakeoff, "_prepare_cases", lambda *_args, **_kwargs: [preparation])

    def load_reference_after_calls(
        _prepared: bakeoff._PreparedCase,
    ) -> tuple[PaperReference, dict[str, int | float | None]]:
        assert len(client.requests) == expected_calls
        return _reference(), {"target_page": 1, "scoped_reference_observations": 0}

    monkeypatch.setattr(bakeoff, "_load_scoring_reference", load_reference_after_calls)
    monkeypatch.setattr(
        bakeoff,
        "_code_state",
        lambda root: {"root_name": Path(root).name, "commit": "d" * 40},
    )
    return bakeoff.run_extractor_bakeoff(
        config,
        project_root=Path("/private/input-tree"),
        code_root=Path("/public/code-tree"),
        client=client,
    )


def test_config_preserves_v01_defaults_and_enforces_v02_contract() -> None:
    historical = _config()
    assert historical.schema_version == "extractor-bakeoff/0.1"
    assert historical.temperature == 0.0
    assert historical.reasoning_effort == "minimal"
    assert historical.seed == 7
    assert historical.require_parameters is False
    assert historical.fresh_repetitions == 1

    common = _config(
        schema_version="extractor-bakeoff/0.2",
        temperature=None,
        reasoning_effort=None,
        seed=None,
        fresh_repetitions=2,
    )
    assert common.require_parameters is True
    assert common.temperature is None
    assert common.reasoning_effort is None
    assert common.seed is None

    repair04 = _config(schema_version="extractor-bakeoff/0.3")
    assert repair04.max_tokens == 16_000
    assert repair04.temperature is None
    assert repair04.reasoning_effort == "minimal"
    assert repair04.seed is None
    assert repair04.require_parameters is True

    with pytest.raises(ValidationError, match="requires require_parameters=true"):
        _config(schema_version="extractor-bakeoff/0.2", require_parameters=False)
    with pytest.raises(ValidationError, match="production request contract"):
        _config(schema_version="extractor-bakeoff/0.3", reasoning_effort=None)
    with pytest.raises(ValidationError, match="fresh_repetitions"):
        _config(fresh_repetitions=0)
    with pytest.raises(ValidationError, match="must be an integer"):
        _config(fresh_repetitions=True)
    with pytest.raises(ValidationError, match="integer or null"):
        _config(seed=True)


def test_fresh_repetitions_are_separate_bound_calls_with_stability_and_denominators(
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    config = _config(
        schema_version="extractor-bakeoff/0.2",
        temperature=None,
        reasoning_effort=None,
        seed=None,
        fresh_repetitions=2,
    )
    client = FakeClient()
    result = _run(monkeypatch, manifest, config, client)

    assert len(client.requests) == 4
    assert all(request["temperature"] is None for request in client.requests)
    assert all(request["reasoning_effort"] is None for request in client.requests)
    assert all(request["seed"] is None for request in client.requests)
    assert all(request["require_parameters"] is True for request in client.requests)
    assert result["code"]["root_name"] == "code-tree"
    assert result["determinism"]["reference_join_phase"] == "after_all_provider_calls"
    assert result["request_contract"]["routing"]["require_parameters"] is True

    model = result["models"][0]
    case = model["cases"][0]
    assert len(case["repetitions"]) == 2
    assert len(case["calls"]) == 2
    assert len({call["logical_call_id"] for call in case["calls"]}) == 2
    assert all(len(call["requested"]["request_sha256"]) == 64 for call in case["calls"])
    assert all(call["contract"]["status"] == "satisfied" for call in case["calls"])
    assert all(call["telemetry"]["model_returned"] == model["model"] for call in case["calls"])
    assert all(
        call["telemetry"]["model_returned_disposition"] == "matches_requested"
        for call in case["calls"]
    )
    assert all(call["telemetry"]["provider_returned"] is None for call in case["calls"])
    assert all(
        call["telemetry"]["provider_returned_disposition"] == "unrecognized_omitted"
        for call in case["calls"]
    )
    assert all(call["telemetry"]["reasoning_tokens"] == 2 for call in case["calls"])
    assert case["stability"]["planned_pair_denominator"] == 1
    assert case["stability"]["comparable_pair_denominator"] == 1
    assert case["stability"]["exact_candidate_set_match_rate"] == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
    }
    assert model["aggregate"]["schema"]["calls_in_end_to_end_denominator"] == 2
    assert model["aggregate"]["first_pass_wire_schema"]["calls_in_end_to_end_denominator"] == 1
    assert model["aggregate"]["execution"]["repetitions_scored"] == 2
    assert model["aggregate"]["usage"]["reasoning_tokens"] == {
        "total": 4,
        "reported_calls": 2,
        "missing_calls": 0,
    }


def test_contract_failure_is_preserved_and_excluded_from_quality(
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    config = _config(
        schema_version="extractor-bakeoff/0.2",
        temperature=None,
        reasoning_effort=None,
        seed=None,
    )
    client = FakeClient({"vendor/model-a": "contract_mismatch"})
    result = _run(monkeypatch, manifest, config, client)

    failed = result["models"][0]
    repetition = failed["cases"][0]["repetitions"][0]
    call = repetition["calls"][0]
    assert call["schema_status"] == "valid"
    assert call["status"] == "contract_failure"
    assert call["contract"]["status"] == "failed"
    assert "require_parameters_mismatch" in call["contract"]["failure_codes"]
    assert repetition["scoring"] == {
        "status": "not_scored",
        "reason": "request_contract_ineligible",
    }
    assert "score" not in repetition
    assert failed["aggregate"]["contract"]["failed"] == 1
    assert failed["aggregate"]["schema"]["structured_responses_valid"] == 1
    assert failed["aggregate"]["execution"]["repetitions_scored"] == 0

    successful = result["models"][1]
    assert successful["aggregate"]["contract"]["failed"] == 0
    assert successful["aggregate"]["execution"]["repetitions_scored"] == 1


def test_wire_schema_denominators_distinguish_invalid_from_not_observed(
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    config = _config(schema_version="extractor-bakeoff/0.2")
    client = FakeClient(
        {
            "vendor/model-a": "invalid_schema",
            "vendor/model-b": "not_observed",
        }
    )
    result = _run(monkeypatch, manifest, config, client)

    invalid = result["models"][0]["aggregate"]["first_pass_wire_schema"]
    assert invalid["calls_in_end_to_end_denominator"] == 1
    assert invalid["structured_responses_observed"] == 1
    assert invalid["structured_responses_invalid"] == 1
    assert invalid["structured_response_not_observed"] == 0
    assert invalid["valid_of_observed"] == {
        "numerator": 0,
        "denominator": 1,
        "value": 0.0,
    }

    missing = result["models"][1]["aggregate"]["first_pass_wire_schema"]
    assert missing["calls_in_end_to_end_denominator"] == 1
    assert missing["structured_responses_observed"] == 0
    assert missing["structured_responses_invalid"] == 0
    assert missing["structured_response_not_observed"] == 1
    assert missing["valid_of_observed"] == {
        "numerator": 0,
        "denominator": 0,
        "value": None,
    }
    for model in result["models"]:
        assert model["aggregate"]["quality_measurement"]["status"] == "unmeasured"
        assert model["aggregate"]["quality"] is None
        repetition = model["cases"][0]["repetitions"][0]
        assert repetition["scoring"]["status"] == "not_scored"
        assert "score" not in repetition


def test_configuration_hash_binds_null_settings_and_repetitions(
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    first = _run(
        monkeypatch,
        manifest,
        _config(
            schema_version="extractor-bakeoff/0.2",
            temperature=None,
            reasoning_effort=None,
            seed=None,
            fresh_repetitions=1,
        ),
        FakeClient(),
    )
    second = _run(
        monkeypatch,
        manifest,
        _config(
            schema_version="extractor-bakeoff/0.2",
            temperature=None,
            reasoning_effort=None,
            seed=None,
            fresh_repetitions=2,
        ),
        FakeClient(),
    )
    assert first["configuration_sha256"] != second["configuration_sha256"]
    assert (
        first["configuration_binding"]["requested_settings_sha256"]
        != second["configuration_binding"]["requested_settings_sha256"]
    )
    assert len(first["configuration_sha256"]) == 64


def test_budget_stop_aborts_before_later_calls_or_reference_join(
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    config = _config(
        schema_version="extractor-bakeoff/0.2",
        temperature=None,
        reasoning_effort=None,
        seed=None,
        fresh_repetitions=2,
    )
    preparation = _preparation(manifest)
    monkeypatch.setattr(bakeoff, "_prepare_cases", lambda *_args, **_kwargs: [preparation])
    reference_loaded = False

    def forbidden_reference_join(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal reference_loaded
        reference_loaded = True
        raise AssertionError("reference join must not occur after a bounded stop")

    monkeypatch.setattr(bakeoff, "_load_scoring_reference", forbidden_reference_join)

    class BudgetStoppedClient(FakeClient):
        def structured_chat(self, **kwargs: Any) -> StructuredResponse:
            self.requests.append(dict(kwargs))
            raise ProviderBudgetExhausted(
                reason="structured_call_limit",
                summary={"structured_calls": 0},
            )

    client = BudgetStoppedClient()
    with pytest.raises(ProviderBudgetExhausted):
        bakeoff.run_extractor_bakeoff(
            config,
            project_root=Path("/private/input-tree"),
            code_root=Path("/public/code-tree"),
            client=client,
        )
    assert len(client.requests) == 1
    assert reference_loaded is False


def _split_setup(
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> bakeoff._Preparation:
    preparation = _preparation(manifest)
    monkeypatch.setattr(
        bakeoff,
        "_prepare_cases",
        lambda *_args, **_kwargs: [preparation],
    )
    monkeypatch.setattr(
        bakeoff,
        "_code_state",
        lambda root: {"root_name": Path(root).name, "commit": "d" * 40},
    )
    return preparation


def _install_split_references(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calls: list[str] | None = None,
) -> None:
    def load_references(
        context: bakeoff._ExtractorProviderContext,
        *,
        project_root: Path,
    ) -> dict[str, tuple[PaperReference, dict[str, int | float | None], str]]:
        del project_root
        if calls is not None:
            calls.append("references_loaded")
        return {
            preparation.spec.case_id: (
                _reference(),
                {"target_page": 1, "scoped_reference_observations": 0},
                "e" * 64,
            )
            for preparation in context.preparations
        }

    monkeypatch.setattr(bakeoff, "_load_extractor_offline_references", load_references)


def test_provider_output_cannot_overwrite_extractor_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    _split_setup(monkeypatch, manifest)
    project_root = Path("/private/input-tree")
    target = project_root / "private" / "manifest.json"

    with pytest.raises(ValueError, match="aliases an immutable input"):
        bakeoff.run_extractor_bakeoff_provider_phase(
            _config(schema_version="extractor-bakeoff/0.3", require_parameters=True),
            project_root=project_root,
            code_root=Path("/public/code-tree"),
            checkpoint_path=tmp_path / "extractor-checkpoint.json",
            output_path=target,
            client=object(),
        )


class BudgetAfterClient(FakeClient):
    def __init__(self, *, allowed_calls: int) -> None:
        super().__init__()
        self.allowed_calls = allowed_calls

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        if len(self.requests) >= self.allowed_calls:
            raise ProviderBudgetExhausted(
                reason="structured_call_limit",
                summary={"structured_calls": len(self.requests)},
            )
        return super().structured_chat(**kwargs)


def test_provider_preparation_never_resolves_or_hashes_reference_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    config = _config(schema_version="extractor-bakeoff/0.2")
    preparation = _preparation(manifest)
    assert preparation.prepared is not None
    hashed: list[Path] = []
    monkeypatch.setattr(bakeoff, "read_json", lambda _path: manifest.model_dump(mode="json"))
    monkeypatch.setattr(bakeoff, "resolve_cached_path", lambda *_args: tmp_path / "paper.pdf")
    monkeypatch.setattr(
        bakeoff,
        "extract_pdf_layout",
        lambda *_args: preparation.prepared.layout,
    )

    def record_hash(path: Path) -> str:
        hashed.append(path)
        return "f" * 64

    monkeypatch.setattr(bakeoff, "sha256_file", record_hash)
    prepared = bakeoff._prepare_case(
        config.cases[0],
        project_root=tmp_path,
        segmentation=config.segmentation.production_config(),
        bind_reference=False,
    )
    assert prepared.reference_path is None
    assert prepared.reference_sha256 is None
    assert hashed == [(tmp_path / config.cases[0].manifest_path).resolve()]


def test_split_provider_phase_resumes_exactly_and_is_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    _split_setup(monkeypatch, manifest)
    config = _config(
        schema_version="extractor-bakeoff/0.2",
        temperature=None,
        reasoning_effort=None,
        seed=None,
        fresh_repetitions=2,
    )
    checkpoint = tmp_path / "private" / "extractor-checkpoint.json"
    provider_output = tmp_path / "provider-result.json"

    with pytest.raises(ProviderBudgetExhausted):
        bakeoff.run_extractor_bakeoff_provider_phase(
            config,
            project_root=Path("/private/input-tree"),
            code_root=Path("/public/code-tree"),
            checkpoint_path=checkpoint,
            client=BudgetAfterClient(allowed_calls=1),
            output_path=provider_output,
        )
    stopped = bakeoff.read_json(checkpoint)
    assert stopped["status"] == "in_progress"
    assert len(stopped["entries"]) == 1
    assert not provider_output.exists()

    resumed_client = FakeClient()
    first = bakeoff.run_extractor_bakeoff_provider_phase(
        config,
        project_root=Path("/private/input-tree"),
        code_root=Path("/public/code-tree"),
        checkpoint_path=checkpoint,
        client=resumed_client,
        output_path=provider_output,
    )
    assert len(resumed_client.requests) == 3
    assert first["status"] == "sealed"
    assert first["privacy"]["references_loaded_during_provider_phase"] is False
    checkpoint_bytes = checkpoint.read_bytes()
    provider_bytes = provider_output.read_bytes()

    no_calls = FakeClient()
    second = bakeoff.run_extractor_bakeoff_provider_phase(
        config,
        project_root=Path("/private/input-tree"),
        code_root=Path("/public/code-tree"),
        checkpoint_path=checkpoint,
        client=no_calls,
        output_path=provider_output,
    )
    assert no_calls.requests == []
    assert second == first
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert provider_output.read_bytes() == provider_bytes


def test_offline_scoring_validates_seal_before_loading_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    _split_setup(monkeypatch, manifest)
    config = _config(schema_version="extractor-bakeoff/0.2")
    checkpoint = tmp_path / "private" / "checkpoint.json"
    bakeoff.run_extractor_bakeoff_provider_phase(
        config,
        project_root=Path("/private/input-tree"),
        checkpoint_path=checkpoint,
        client=FakeClient(),
    )
    sealed_bytes = checkpoint.read_bytes()
    reference_calls: list[str] = []
    _install_split_references(monkeypatch, calls=reference_calls)

    raw = bakeoff.read_json(checkpoint)
    raw["status"] = "in_progress"
    raw["provider_phase_seal_sha256"] = None
    unsigned = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    raw["checkpoint_sha256"] = bakeoff._extractor_hash(unsigned)
    bakeoff.write_json(checkpoint, raw)
    with pytest.raises(ValueError, match="not sealed"):
        bakeoff.score_extractor_bakeoff_offline(
            config,
            project_root=Path("/private/input-tree"),
            checkpoint_path=checkpoint,
        )
    assert reference_calls == []

    checkpoint.write_bytes(sealed_bytes)
    raw = bakeoff.read_json(checkpoint)
    raw["provider_phase_seal_sha256"] = "0" * 64
    unsigned = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    raw["checkpoint_sha256"] = bakeoff._extractor_hash(unsigned)
    bakeoff.write_json(checkpoint, raw)
    with pytest.raises(ValueError, match="checkpoint is invalid"):
        bakeoff.score_extractor_bakeoff_offline(
            config,
            project_root=Path("/private/input-tree"),
            checkpoint_path=checkpoint,
        )
    assert reference_calls == []

    checkpoint.write_bytes(sealed_bytes)
    score = bakeoff.score_extractor_bakeoff_offline(
        config,
        project_root=Path("/private/input-tree"),
        checkpoint_path=checkpoint,
    )
    assert reference_calls == ["references_loaded"]
    assert score["privacy"]["references_loaded_after_provider_phase_sealed"] is True


def test_split_checkpoint_tamper_and_stale_configuration_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    _split_setup(monkeypatch, manifest)
    config = _config(schema_version="extractor-bakeoff/0.2")
    checkpoint = tmp_path / "checkpoint.json"
    bakeoff.run_extractor_bakeoff_provider_phase(
        config,
        project_root=Path("/private/input-tree"),
        checkpoint_path=checkpoint,
        client=FakeClient(),
    )

    stale_client = FakeClient()
    with pytest.raises(ValueError, match="configuration is stale"):
        bakeoff.run_extractor_bakeoff_provider_phase(
            config.model_copy(update={"max_tokens": config.max_tokens + 1}),
            project_root=Path("/private/input-tree"),
            checkpoint_path=checkpoint,
            client=stale_client,
        )
    assert stale_client.requests == []

    raw = bakeoff.read_json(checkpoint)
    first = next(iter(raw["entries"].values()))
    first["request"]["settings"]["max_tokens"] = 1
    bakeoff.write_json(checkpoint, raw)
    tamper_client = FakeClient()
    with pytest.raises(ValueError, match="checkpoint is invalid"):
        bakeoff.run_extractor_bakeoff_provider_phase(
            config,
            project_root=Path("/private/input-tree"),
            checkpoint_path=checkpoint,
            client=tamper_client,
        )
    assert tamper_client.requests == []


def test_split_failed_calls_have_explicitly_unmeasured_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    _split_setup(monkeypatch, manifest)
    _install_split_references(monkeypatch)
    config = _config(schema_version="extractor-bakeoff/0.2")
    checkpoint = tmp_path / "checkpoint.json"
    provider = bakeoff.run_extractor_bakeoff_provider_phase(
        config,
        project_root=Path("/private/input-tree"),
        checkpoint_path=checkpoint,
        client=FakeClient(
            {
                "vendor/model-a": "invalid_schema",
                "vendor/model-b": "not_observed",
            }
        ),
    )
    assert all("quality" not in model["aggregate"] for model in provider["models"])

    score = bakeoff.score_extractor_bakeoff_offline(
        config,
        project_root=Path("/private/input-tree"),
        checkpoint_path=checkpoint,
    )
    for model in score["models"]:
        assert model["aggregate"]["quality_measurement"] == {
            "status": "unmeasured",
            "scored_repetitions": 0,
            "unmeasured_repetitions": 1,
            "reason": "no_fully_successful_contract_eligible_repetition",
        }
        assert model["aggregate"]["quality"] is None
        repetition = model["cases"][0]["repetitions"][0]
        assert repetition["scoring"]["status"] == "not_scored"
        assert "score" not in repetition


def test_sealed_extractor_smoke_selects_and_runs_only_contract_eligible_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    _split_setup(monkeypatch, manifest)
    smoke_config = _config(
        schema_version="extractor-bakeoff/0.2",
        bakeoff_id="extractor-smoke",
        fresh_repetitions=1,
    )
    smoke_checkpoint = tmp_path / "smoke-checkpoint.json"
    smoke_client = FakeClient({"vendor/model-b": "invalid_schema"})
    bakeoff.run_extractor_bakeoff_provider_phase(
        smoke_config,
        project_root=Path("/private/input-tree"),
        code_root=Path("/public/code-tree"),
        checkpoint_path=smoke_checkpoint,
        client=smoke_client,
    )
    selection = bakeoff.derive_extractor_execution_selection(
        smoke_config,
        project_root=Path("/private/input-tree"),
        code_root=Path("/public/code-tree"),
        checkpoint_path=smoke_checkpoint,
        experiment_id="staged-model-comparison-2026-09-02",
        public_manifest_sha256="a" * 64,
    )
    assert selection.stage == "block_candidate_extraction"
    assert selection.eligible_models == ("vendor/model-a",)
    assert selection.decision_for("vendor/model-b").status == "contract_ineligible"
    assert selection.decision_for("vendor/model-b").reason_codes == ["wire_schema_below_gate"]
    expectation = _selection_expectation(selection)

    full_config = _config(
        schema_version="extractor-bakeoff/0.2",
        bakeoff_id="extractor-full",
        fresh_repetitions=2,
    )
    full_checkpoint = tmp_path / "full-checkpoint.json"
    full_output = tmp_path / "full-provider.json"
    full_client = FakeClient()
    provider = bakeoff.run_extractor_bakeoff_provider_phase(
        full_config,
        project_root=Path("/private/input-tree"),
        code_root=Path("/public/code-tree"),
        checkpoint_path=full_checkpoint,
        client=full_client,
        output_path=full_output,
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert [request["model"] for request in full_client.requests] == [
        "vendor/model-a",
        "vendor/model-a",
    ]
    assert provider["declared_models"] == ["vendor/model-a", "vendor/model-b"]
    assert provider["executed_models"] == ["vendor/model-a"]
    assert provider["execution_selection"] == {
        "selection_sha256": selection.selection_sha256,
        "smoke_provider_phase_seal_sha256": selection.smoke_provider_phase_seal_sha256,
        "smoke_checkpoint_sha256": selection.smoke_checkpoint_sha256,
        "eligible_models": ["vendor/model-a"],
    }
    executed, skipped = provider["models"]
    assert executed["matched_quality_status"] == "executed"
    assert executed["quality_status"] == "pending_offline_scoring"
    assert executed["quality"] is None
    assert skipped == {
        "model": "vendor/model-b",
        "label": "B",
        "contract_eligibility": "contract_ineligible",
        "eligibility_reason_codes": ["wire_schema_below_gate"],
        "matched_quality_status": "not_run",
        "not_run_reason": "contract_ineligible_on_sealed_smoke",
        "quality_status": "unmeasured_contract_ineligible",
        "quality": None,
        "aggregate": None,
        "cases": [],
    }
    assert {entry["model"] for entry in bakeoff.read_json(full_checkpoint)["entries"].values()} == {
        "vendor/model-a"
    }

    provider_bytes = full_output.read_bytes()
    checkpoint_bytes = full_checkpoint.read_bytes()
    no_calls = FakeClient()
    rerun = bakeoff.run_extractor_bakeoff_provider_phase(
        full_config,
        project_root=Path("/private/input-tree"),
        code_root=Path("/public/code-tree"),
        checkpoint_path=full_checkpoint,
        client=no_calls,
        output_path=full_output,
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert rerun == provider
    assert no_calls.requests == []
    assert full_output.read_bytes() == provider_bytes
    assert full_checkpoint.read_bytes() == checkpoint_bytes

    reference_calls: list[str] = []
    _install_split_references(monkeypatch, calls=reference_calls)
    score = bakeoff.score_extractor_bakeoff_offline(
        full_config,
        project_root=Path("/private/input-tree"),
        code_root=Path("/public/code-tree"),
        checkpoint_path=full_checkpoint,
        execution_selection=selection,
        execution_selection_expectation=expectation,
    )
    assert reference_calls == ["references_loaded"]
    assert score["execution_selection_sha256"] == selection.selection_sha256
    assert score["models"][0]["matched_quality_status"] == "executed"
    assert score["models"][0]["quality_status"] == "measured"
    assert score["models"][1]["matched_quality_status"] == "not_run"
    assert score["models"][1]["quality_status"] == "unmeasured_contract_ineligible"
    assert score["models"][1]["quality"] is None

    reference_calls.clear()
    with pytest.raises(ValueError, match="run contract is stale"):
        bakeoff.score_extractor_bakeoff_offline(
            full_config,
            project_root=Path("/private/input-tree"),
            code_root=Path("/public/code-tree"),
            checkpoint_path=full_checkpoint,
        )
    assert reference_calls == []


def test_extractor_selection_tamper_and_staleness_fail_before_calls_or_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    _split_setup(monkeypatch, manifest)
    smoke_config = _config(
        schema_version="extractor-bakeoff/0.2",
        bakeoff_id="extractor-smoke",
    )
    smoke_checkpoint = tmp_path / "smoke-checkpoint.json"
    bakeoff.run_extractor_bakeoff_provider_phase(
        smoke_config,
        project_root=Path("/private/input-tree"),
        code_root=Path("/public/code-tree"),
        checkpoint_path=smoke_checkpoint,
        client=FakeClient({"vendor/model-b": "invalid_schema"}),
    )
    selection = bakeoff.derive_extractor_execution_selection(
        smoke_config,
        project_root=Path("/private/input-tree"),
        code_root=Path("/public/code-tree"),
        checkpoint_path=smoke_checkpoint,
        experiment_id="selection-integrity-test",
        public_manifest_sha256="a" * 64,
    )
    expectation = _selection_expectation(selection)
    full_config = _config(
        schema_version="extractor-bakeoff/0.2",
        bakeoff_id="extractor-full",
        fresh_repetitions=2,
    )

    tampered = selection.model_copy(update={"stage": "origin_retrieval"})
    tampered_client = FakeClient()
    with pytest.raises(ValidationError, match="selection hash is invalid"):
        bakeoff.run_extractor_bakeoff_provider_phase(
            full_config,
            project_root=Path("/private/input-tree"),
            code_root=Path("/public/code-tree"),
            checkpoint_path=tmp_path / "tampered-checkpoint.json",
            client=tampered_client,
            execution_selection=tampered,
            execution_selection_expectation=expectation,
        )
    assert tampered_client.requests == []

    stale_payload = selection.model_dump(mode="json", exclude={"selection_sha256"})
    stale_payload["stage_contract_sha256"] = "0" * 64
    stale = bakeoff.StageExecutionSelection.model_validate(
        stale_payload | {"selection_sha256": bakeoff._json_sha256(stale_payload)}
    )
    stale_client = FakeClient()
    with pytest.raises(ValueError, match="stale stage contract"):
        bakeoff.run_extractor_bakeoff_provider_phase(
            full_config,
            project_root=Path("/private/input-tree"),
            code_root=Path("/public/code-tree"),
            checkpoint_path=tmp_path / "stale-checkpoint.json",
            client=stale_client,
            execution_selection=stale,
            execution_selection_expectation=expectation,
        )
    assert stale_client.requests == []

    reference_calls: list[str] = []
    _install_split_references(monkeypatch, calls=reference_calls)
    with pytest.raises(ValueError, match="stale stage contract"):
        bakeoff.score_extractor_bakeoff_offline(
            full_config,
            project_root=Path("/private/input-tree"),
            code_root=Path("/public/code-tree"),
            checkpoint_path=tmp_path / "missing-checkpoint.json",
            execution_selection=stale,
            execution_selection_expectation=expectation,
        )
    assert reference_calls == []

    wrong_manifest = expectation.model_copy(update={"public_manifest_sha256": "0" * 64})
    wrong_manifest_client = FakeClient()
    with pytest.raises(ValueError, match="another public manifest"):
        bakeoff.run_extractor_bakeoff_provider_phase(
            full_config,
            project_root=Path("/private/input-tree"),
            code_root=Path("/public/code-tree"),
            checkpoint_path=tmp_path / "wrong-manifest-checkpoint.json",
            client=wrong_manifest_client,
            execution_selection=selection,
            execution_selection_expectation=wrong_manifest,
        )
    assert wrong_manifest_client.requests == []


def test_fully_rehashed_candidate_splice_is_rejected_before_offline_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: SourceManifest,
) -> None:
    _split_setup(monkeypatch, manifest)
    config = _config(schema_version="extractor-bakeoff/0.2")
    checkpoint_path = tmp_path / "checkpoint.json"
    bakeoff.run_extractor_bakeoff_provider_phase(
        config,
        project_root=Path("/private/input-tree"),
        checkpoint_path=checkpoint_path,
        client=BoundCandidateClient(),
    )
    raw = bakeoff.read_json(checkpoint_path)
    by_model = {entry["model"]: entry for entry in raw["entries"].values()}
    target = by_model["vendor/model-a"]
    donor = by_model["vendor/model-b"]
    assert target["call"]["response_sha256"] != donor["call"]["response_sha256"]
    target["candidates"] = donor["candidates"]
    target["candidate_fingerprints"] = donor["candidate_fingerprints"]
    target_unsigned = {key: value for key, value in target.items() if key != "entry_sha256"}
    target["entry_sha256"] = bakeoff._extractor_hash(target_unsigned)
    raw["provider_phase_seal_sha256"] = bakeoff._json_sha256(
        {
            "schema_version": "extractor-bakeoff-provider-seal/0.1",
            "configuration_sha256": raw["configuration_sha256"],
            "run_contract_sha256": raw["run_contract_sha256"],
            "entry_sha256s": {
                slot_id: entry["entry_sha256"] for slot_id, entry in sorted(raw["entries"].items())
            },
        }
    )
    checkpoint_unsigned = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    raw["checkpoint_sha256"] = bakeoff._extractor_hash(checkpoint_unsigned)
    bakeoff.write_json(checkpoint_path, raw)

    reference_calls: list[str] = []
    _install_split_references(monkeypatch, calls=reference_calls)
    with pytest.raises(ValueError, match="checkpoint is invalid"):
        bakeoff.score_extractor_bakeoff_offline(
            config,
            project_root=Path("/private/input-tree"),
            checkpoint_path=checkpoint_path,
        )
    assert reference_calls == []


def _quality_score(
    *,
    reference_observations: int,
    true_positives: int,
    observable_references: int | None = None,
) -> dict[str, Any]:
    false_negatives = reference_observations - true_positives
    recall = true_positives / reference_observations if reference_observations else 0.0
    return {
        "reference_observations": reference_observations,
        "field_matching_basis": reference_observations,
        "detection": {
            "true_positives": true_positives,
            "precision_true_positives": true_positives,
            "precision_basis": true_positives,
            "recall_basis": reference_observations,
            "false_positives": 0,
            "false_negatives": false_negatives,
            "precision": 1.0 if true_positives else None,
            "recall": recall,
            "f1": 1.0 if true_positives == reference_observations and true_positives else None,
        },
        "field_accuracy": {
            field: 1.0 if reference_observations else 0.0 for field in bakeoff.QUALITY_FIELDS
        },
        "input_observability": (
            {
                "status": "measured",
                "observable_reference_observations": observable_references,
                "unobservable_reference_observations": (
                    reference_observations - observable_references
                ),
                "model_conditional_detection": {
                    "true_positives": min(true_positives, observable_references),
                    "false_negatives": max(0, observable_references - true_positives),
                },
            }
            if observable_references is not None
            else {"status": "not_assessed"}
        ),
    }


def test_quality_macro_recall_excludes_zero_reference_cases() -> None:
    zero_reference = _quality_score(reference_observations=0, true_positives=0)
    zero_reference["detection"]["f1"] = 0.0
    quality = bakeoff._aggregate_quality(
        [_quality_score(reference_observations=1, true_positives=1), zero_reference]
    )

    detection = quality["macro"]["detection"]
    assert detection["recall"] == 1.0
    assert detection["f1"] == 1.0
    assert detection["defined_cases"]["recall"] == 1
    assert detection["undefined_cases"]["recall"] == 1
    assert detection["defined_cases"]["f1"] == 1
    assert detection["undefined_cases"]["f1"] == 1


def test_quality_reports_model_conditional_recall_beside_pipeline_recall() -> None:
    quality = bakeoff._aggregate_quality(
        [
            _quality_score(
                reference_observations=2,
                true_positives=1,
                observable_references=1,
            )
        ]
    )

    assert quality["micro"]["detection"]["recall"] == 0.5
    assert quality["micro"]["input_observability"] == {
        "status": "measured",
        "cases_measured": 1,
        "cases_not_assessed": 0,
        "measured_reference_observations": 2,
        "reference_observations": 2,
        "observable_reference_observations": 1,
        "unobservable_reference_observations": 1,
        "observation_coverage": 0.5,
        "model_conditional_detection": {
            "true_positives": 1,
            "false_negatives": 0,
            "recall_basis": 1,
            "recall": 1.0,
        },
    }


def test_quality_never_publishes_observability_from_a_measured_subset() -> None:
    quality = bakeoff._aggregate_quality(
        [
            _quality_score(
                reference_observations=1,
                true_positives=1,
                observable_references=1,
            ),
            _quality_score(reference_observations=1, true_positives=1),
        ]
    )

    assert quality["micro"]["input_observability"] == {
        "status": "partially_assessed",
        "cases_measured": 1,
        "cases_not_assessed": 1,
        "measured_reference_observations": 1,
        "reference_observations": None,
        "observable_reference_observations": None,
        "unobservable_reference_observations": None,
        "observation_coverage": None,
        "model_conditional_detection": {
            "true_positives": None,
            "false_negatives": None,
            "recall_basis": None,
            "recall": None,
        },
    }
