from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import proceedings_to_eee.verification.replay as replay_module
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ClaimType, ExportStatus
from proceedings_to_eee.extraction.pdf_layout import PageFragment
from proceedings_to_eee.extraction.result_blocks import segment_page_result_blocks
from proceedings_to_eee.io import write_json, write_jsonl
from proceedings_to_eee.providers.openrouter import (
    ProviderCall,
    ProviderResponseValidationError,
    StructuredResponse,
    completion_token_parameter_for_model,
    structured_request_contract,
)
from proceedings_to_eee.verification.binding import bind_candidate_block, frozen_evidence_block
from proceedings_to_eee.verification.independent import (
    IndependentDecision,
    VerifierRequestSettings,
)
from proceedings_to_eee.verification.replay import (
    ReplayCheckpointError,
    ReplayScope,
    ReplaySettings,
    in_replay_scope,
    measure_replay,
    replay_paper,
    replay_run,
)

PAGE_TEXT = """
Table 2. ROC-AUC proportion on Synthetic Benchmark, reported alongside the invented
system developed for this synthetic fixture.

    Model                                ROC-AUC
    Paper System                            0.742
    Example Ensemble (Leaderboard Entry)    0.781
    Linear Baseline                         0.611

The evaluation reports the ROC-AUC of every model on the same held-out split.
"""


def _fragment() -> PageFragment:
    text = PAGE_TEXT
    return PageFragment(
        fragment_id="frag_src_paper_0013",
        source_id="src_paper",
        page=13,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=4,
        result_signal_score=9.0,
    )


def _blocks() -> list[Any]:
    blocks = segment_page_result_blocks(_fragment())
    assert blocks, "fixture page must segment into at least one result block"
    return blocks


def _candidate(quote: str, *, export_status: ExportStatus, row: str) -> CandidateObservation:
    return CandidateObservation.model_validate(
        {
            "paper_id": "synthetic-replay-fixture",
            "claim_type": ClaimType.PRIMARY_RESULT,
            "roles": [{"role": "evaluated_system", "raw_name": row, "confidence": 0.9}],
            "scope": {"dataset_raw": "Synthetic Benchmark"},
            "metric": {"raw_name": "ROC-AUC", "canonical_id": "auroc", "unit": "proportion"},
            "value": {"raw": quote.split()[-1], "numeric": float(quote.split()[-1])},
            "evidence": [
                {
                    "source_id": "src_paper",
                    "page": 13,
                    "kind": "table",
                    "label": "Table 2",
                    "row": row,
                    "column": "ROC-AUC",
                    "quote": quote,
                }
            ],
            "export_status": export_status,
            "extraction_method": "fixture",
            "extraction_confidence": 0.95,
        }
    )


# The frozen block text carries every row of the table, so a verdict key taken from the
# block would fire for every candidate. Key on the candidate payload instead.
LEADERBOARD_ROW_KEY = '"raw":"0.781"'


class RecordingClient:
    """Return a scripted verdict per candidate and record what was actually sent."""

    def __init__(self, verdicts: dict[str, str]) -> None:
        self.verdicts = verdicts
        self.sent: list[dict[str, Any]] = []

    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        self.sent.append(kwargs)
        user = kwargs["user"]
        raw = user.split("<VERIFICATION_INPUT>\n", 1)[1].split("\n</VERIFICATION_INPUT>", 1)[0]
        verifier_input = json.loads(raw)
        candidate = verifier_input["candidate_claim_untrusted"]
        anchor = verifier_input["candidate_claimed_anchor_untrusted"]
        lines = verifier_input["trusted_frozen_source_block"]["lines"]

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

        decision = next(
            (value for key, value in self.verdicts.items() if key in user),
            IndependentDecision.ACCEPT.value,
        )
        if decision == IndependentDecision.ACCEPT.value:
            findings = dict.fromkeys(("support", "role", "scope", "value", "metric"), "supported")
        elif decision == IndependentDecision.REJECT.value:
            findings = dict.fromkeys(("support", "role", "scope", "value", "metric"), "supported")
            findings["role"] = "contradicted"
        else:
            findings = dict.fromkeys(("support", "role", "scope", "value", "metric"), "supported")
            findings["scope"] = "insufficient_evidence"
        scope = candidate["scope"] or {}
        metric = candidate["metric"] or {}
        value = candidate["value"] or {}
        payload = {
            "support": findings["support"],
            "support_evidence_line_ids": evidence_ids(anchor["quote"]),
            "role": findings["role"],
            "role_evidence_line_ids": evidence_ids(
                *(role["raw_name"] for role in candidate["roles"])
            ),
            "scope": findings["scope"],
            "scope_evidence_line_ids": (
                []
                if findings["scope"] == "insufficient_evidence"
                else evidence_ids(scope.get("dataset_raw"))
            ),
            "value": findings["value"],
            "value_evidence_line_ids": evidence_ids(value.get("raw"), value.get("unit")),
            "metric": findings["metric"],
            "metric_evidence_line_ids": evidence_ids(metric.get("raw_name"), metric.get("unit")),
            "decision": decision,
            "justification": "fixture verdict",
        }
        messages = [
            {"role": "system", "content": kwargs["system"]},
            {"role": "user", "content": kwargs["user"]},
        ]
        request_contract = structured_request_contract(
            schema_name=kwargs["schema_name"],
            schema=kwargs["schema"],
            seed=kwargs["seed"],
            require_parameters=kwargs["require_parameters"],
        )
        return StructuredResponse(
            payload=payload,
            call=ProviderCall(
                model_requested=kwargs["model"],
                model_returned=kwargs["model"],
                prompt_sha256=hashlib.sha256(
                    json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
                response_sha256=hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()
                ).hexdigest(),
                temperature=kwargs["temperature"],
                reasoning_effort=kwargs["reasoning_effort"],
                max_tokens=kwargs["max_tokens"],
                completion_token_parameter=completion_token_parameter_for_model(kwargs["model"]),
                seed=kwargs["seed"],
                schema_name=kwargs["schema_name"],
                schema_sha256=request_contract["schema"]["schema_sha256"],
                require_parameters=kwargs["require_parameters"],
                latency_seconds=0.02,
                input_tokens=1200,
                output_tokens=60,
                total_tokens=1260,
                cost_usd=0.0004,
                request_id="private-fixture-request-id",
                attempts=1,
            ),
        )


def _write_paper_run(root: Path, candidates: list[CandidateObservation]) -> Path:
    paper_dir = root / "synthetic-replay-fixture"
    (paper_dir / "private").mkdir(parents=True, exist_ok=True)
    write_json(
        paper_dir / "run.json",
        {"paper_id": "synthetic-replay-fixture", "status": "success"},
    )
    write_jsonl(paper_dir / "observations.jsonl", candidates)
    write_json(paper_dir / "private" / "result-blocks.json", _blocks())
    return paper_dir


def _write_named_paper_run(
    root: Path, paper_id: str, candidates: list[CandidateObservation]
) -> Path:
    paper_dir = root / paper_id
    (paper_dir / "private").mkdir(parents=True, exist_ok=True)
    write_json(paper_dir / "run.json", {"paper_id": paper_id, "status": "success"})
    write_jsonl(paper_dir / "observations.jsonl", candidates)
    write_json(paper_dir / "private" / "result-blocks.json", _blocks())
    return paper_dir


def _single_candidate_replay(tmp_path: Path) -> tuple[Path, ReplaySettings]:
    run_root = tmp_path / "sealed"
    paper_dir = _write_paper_run(
        run_root,
        [
            _candidate(
                "Paper System                            0.742",
                export_status=ExportStatus.EXPORTED,
                row="Paper System",
            )
        ],
    )
    return paper_dir, ReplaySettings(
        run_root=run_root,
        output_root=tmp_path / "replay",
        verifier_model="fixture/verifier",
        concurrency=1,
    )


def test_binding_is_deterministic_and_quote_exact() -> None:
    blocks = _blocks()
    candidate = _candidate(
        "Paper System                            0.742",
        export_status=ExportStatus.EXPORTED,
        row="Paper System",
    )
    first = bind_candidate_block(candidate, blocks)
    second = bind_candidate_block(candidate, blocks)
    assert first is not None
    assert second is not None
    assert first[0].block_id == second[0].block_id

    absent = _candidate(
        "Nonexistent Model                      0.111",
        export_status=ExportStatus.EXPORTED,
        row="Nonexistent Model",
    )
    assert bind_candidate_block(absent, blocks) is None


def test_binding_prefers_value_anchor_in_result_body_over_earlier_context_anchor() -> None:
    blocks = _blocks()
    value_block = blocks[0]
    candidate = _candidate(
        "Paper System                            0.742",
        export_status=ExportStatus.EXPORTED,
        row="Paper System",
    )
    value_anchor = candidate.evidence[0]
    context_quote = "Table 2. ROC-AUC proportion on Synthetic Benchmark"
    context_anchor = value_anchor.model_copy(
        update={"quote": context_quote, "row": None, "column": None, "quote_sha256": None}
    )
    candidate.evidence = [context_anchor, value_anchor]
    context_only = value_block.model_copy(
        update={
            "block_id": "rblk_context_only",
            "context_text": context_quote,
            "body_text": "Unrelated methodological context.",
            "trailing_context_text": "",
        }
    )

    bound = bind_candidate_block(candidate, [context_only, value_block])

    assert bound is not None
    block, anchor = bound
    assert block.block_id == value_block.block_id
    assert anchor.quote == value_anchor.quote


def test_binding_rejects_a_wrong_block_with_only_normalized_cross_line_quote() -> None:
    value_block = _blocks()[0]
    candidate = _candidate(
        "Paper System                            0.742",
        export_status=ExportStatus.EXPORTED,
        row="Paper System",
    )
    wrong_block = value_block.model_copy(
        update={
            "block_id": "rblk_wrong_cross_line",
            "body_text": "Paper System\n0.742",
            "body_start_line": 40,
            "body_end_line": 41,
        }
    )

    assert bind_candidate_block(candidate, [wrong_block]) is None
    bound = bind_candidate_block(candidate, [wrong_block, value_block])
    assert bound is not None
    assert bound[0].block_id == value_block.block_id


def test_frozen_evidence_block_hash_binds_the_prompt_text() -> None:
    blocks = _blocks()
    candidate = _candidate(
        "Example Ensemble (Leaderboard Entry)    0.781",
        export_status=ExportStatus.EXPORTED,
        row="Example Ensemble (Leaderboard Entry)",
    )
    bound = bind_candidate_block(candidate, blocks)
    assert bound is not None
    block, anchor = bound
    evidence = frozen_evidence_block(
        paper_id="synthetic-replay-fixture", block=block, anchor=anchor
    )
    assert evidence.text_sha256 == hashlib.sha256(evidence.text.encode("utf-8")).hexdigest()
    assert evidence.page == 13


@pytest.mark.parametrize(
    ("claim_type", "export_status", "scope", "expected"),
    [
        (ClaimType.PRIMARY_RESULT, ExportStatus.EXPORTED, ReplayScope.EXPORT_GATE, True),
        (ClaimType.PRIMARY_RESULT, ExportStatus.NEEDS_REVIEW, ReplayScope.EXPORT_GATE, False),
        (ClaimType.PRIMARY_RESULT, ExportStatus.NEEDS_REVIEW, ReplayScope.PRIMARY, True),
        (ClaimType.SECONDARY_CLAIM, ExportStatus.NEEDS_REVIEW, ReplayScope.PRIMARY, False),
        (ClaimType.SECONDARY_CLAIM, ExportStatus.NEEDS_REVIEW, ReplayScope.ALL, True),
    ],
)
def test_replay_scope_matches_the_pipeline_gate(
    claim_type: ClaimType,
    export_status: ExportStatus,
    scope: ReplayScope,
    expected: bool,
) -> None:
    candidate = _candidate(
        "Paper System                            0.742",
        export_status=export_status,
        row="Paper System",
    )
    candidate.claim_type = claim_type
    assert in_replay_scope(candidate, scope) is expected


def test_replay_writes_verdicts_and_never_touches_the_source_run(tmp_path: Path) -> None:
    run_root = tmp_path / "sealed"
    candidates = [
        _candidate(
            "Paper System                            0.742",
            export_status=ExportStatus.EXPORTED,
            row="Paper System",
        ),
        _candidate(
            "Example Ensemble (Leaderboard Entry)    0.781",
            export_status=ExportStatus.EXPORTED,
            row="Example Ensemble (Leaderboard Entry)",
        ),
    ]
    paper_dir = _write_paper_run(run_root, candidates)
    before = {path: path.read_bytes() for path in sorted(run_root.rglob("*")) if path.is_file()}

    client = RecordingClient({LEADERBOARD_ROW_KEY: IndependentDecision.REJECT.value})
    settings = ReplaySettings(
        run_root=run_root,
        output_root=tmp_path / "replay",
        verifier_model="fixture/verifier",
        concurrency=1,
    )
    summary = replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert summary["candidates_in_scope"] == 2
    assert summary["bound"] == 2
    assert summary["verifications"] == 2
    assert summary["decisions"][IndependentDecision.REJECT.value] == 1
    assert summary["decisions"][IndependentDecision.ACCEPT.value] == 1
    assert summary["cost"]["cost_usd_lower_bound"] == pytest.approx(0.0008)

    after = {path: path.read_bytes() for path in sorted(run_root.rglob("*")) if path.is_file()}
    assert before == after, "the replay must not write into the source run tree"


def test_settings_reject_output_root_equal_to_or_inside_source_run(tmp_path: Path) -> None:
    run_root = tmp_path / "sealed"
    run_root.mkdir()

    with pytest.raises(ValueError, match="outside the read-only run_root"):
        ReplaySettings(
            run_root=run_root,
            output_root=run_root / "replay",
            verifier_model="fixture/verifier",
        )

    alias = tmp_path / "run-alias"
    alias.symlink_to(run_root, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the read-only run_root"):
        ReplaySettings(
            run_root=run_root,
            output_root=alias / "replay",
            verifier_model="fixture/verifier",
        )


def test_replay_rejects_ancestor_output_that_resolves_to_source_run(tmp_path: Path) -> None:
    run_root = tmp_path / "synthetic-replay-fixture"
    paper_dir = _write_paper_run(
        run_root,
        [
            _candidate(
                "Paper System                            0.742",
                export_status=ExportStatus.EXPORTED,
                row="Paper System",
            )
        ],
    )
    settings = ReplaySettings(
        run_root=run_root,
        output_root=tmp_path,
        verifier_model="fixture/verifier",
    )
    before = {path: path.read_bytes() for path in run_root.rglob("*") if path.is_file()}

    with pytest.raises(ReplayCheckpointError, match="aliases the read-only source run"):
        replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)

    after = {path: path.read_bytes() for path in run_root.rglob("*") if path.is_file()}
    assert before == after


def test_replay_rejects_output_artifact_symlink_to_source(tmp_path: Path) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    output_dir = settings.output_root / paper_dir.name
    output_dir.mkdir(parents=True)
    (output_dir / "verifications.jsonl").symlink_to(paper_dir / "observations.jsonl")
    before = (paper_dir / "observations.jsonl").read_bytes()

    with pytest.raises(ReplayCheckpointError, match="aliases the read-only source paper"):
        replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)

    assert (paper_dir / "observations.jsonl").read_bytes() == before


def test_replay_rejects_symlinked_paper_output_outside_output_root(tmp_path: Path) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    settings.output_root.mkdir(parents=True)
    outside = tmp_path / "outside-output-root"
    outside.mkdir()
    (settings.output_root / paper_dir.name).symlink_to(outside, target_is_directory=True)
    client = RecordingClient({})

    with pytest.raises(ReplayCheckpointError, match="direct non-symlink child"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []
    assert list(outside.iterdir()) == []


def test_replay_discovery_rejects_symlinked_source_paper(tmp_path: Path) -> None:
    external_run = tmp_path / "external"
    external_paper = _write_paper_run(
        external_run,
        [
            _candidate(
                "Paper System                            0.742",
                export_status=ExportStatus.EXPORTED,
                row="Paper System",
            )
        ],
    )
    run_root = tmp_path / "sealed"
    run_root.mkdir()
    (run_root / external_paper.name).symlink_to(external_paper, target_is_directory=True)
    settings = ReplaySettings(
        run_root=run_root,
        output_root=tmp_path / "replay",
        verifier_model="fixture/verifier",
        concurrency=1,
    )
    client = RecordingClient({})

    with pytest.raises(ReplayCheckpointError, match="symlinked paper directory"):
        replay_run(client=client, settings=settings)

    assert client.sent == []


@pytest.mark.parametrize(
    "relative_path",
    [Path("run.json"), Path("observations.jsonl"), Path("private/result-blocks.json")],
)
def test_replay_rejects_symlinked_source_artifacts_before_provider_call(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    source = paper_dir / relative_path
    outside = tmp_path / f"outside-{relative_path.name}"
    source.replace(outside)
    source.symlink_to(outside)
    client = RecordingClient({})

    with pytest.raises(ReplayCheckpointError, match="symlinked artifact"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []


def test_replay_rejects_source_paper_outside_exact_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "sealed"
    run_root.mkdir()
    external_paper = _write_paper_run(
        tmp_path / "external",
        [
            _candidate(
                "Paper System                            0.742",
                export_status=ExportStatus.EXPORTED,
                row="Paper System",
            )
        ],
    )
    settings = ReplaySettings(
        run_root=run_root,
        output_root=tmp_path / "replay",
        verifier_model="fixture/verifier",
        concurrency=1,
    )
    client = RecordingClient({})

    with pytest.raises(ReplayCheckpointError, match="not a direct child"):
        replay_paper(client=client, settings=settings, paper_dir=external_paper)

    assert client.sent == []


def test_replay_is_resumable_and_does_not_resend(tmp_path: Path) -> None:
    run_root = tmp_path / "sealed"
    candidates = [
        _candidate(
            "Paper System                            0.742",
            export_status=ExportStatus.EXPORTED,
            row="Paper System",
        ),
    ]
    paper_dir = _write_paper_run(run_root, candidates)
    settings = ReplaySettings(
        run_root=run_root,
        output_root=tmp_path / "replay",
        verifier_model="fixture/verifier",
        concurrency=1,
    )
    first = RecordingClient({})
    replay_paper(client=first, settings=settings, paper_dir=paper_dir)
    assert len(first.sent) == 1

    second = RecordingClient({})
    summary = replay_paper(client=second, settings=settings, paper_dir=paper_dir)
    assert second.sent == []
    assert summary["verifications"] == 1


def test_replay_rejects_duplicate_observation_ids_outside_selected_scope(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "sealed"
    candidate = _candidate(
        "Paper System                            0.742",
        export_status=ExportStatus.NEEDS_REVIEW,
        row="Paper System",
    )
    paper_dir = _write_paper_run(run_root, [candidate, candidate])
    settings = ReplaySettings(
        run_root=run_root,
        output_root=tmp_path / "replay",
        verifier_model="fixture/verifier",
        concurrency=1,
    )
    client = RecordingClient({})

    with pytest.raises(ReplayCheckpointError, match="duplicate observation IDs"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []


@pytest.mark.parametrize(
    ("change", "value"),
    [("verifier_model", "fixture/other-verifier"), ("max_tokens", 2_001)],
)
def test_resume_rejects_changed_model_or_max_tokens_before_provider_call(
    tmp_path: Path, change: str, value: str | int
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)

    changed = replace(settings, **{change: value})
    client = RecordingClient({})
    with pytest.raises(ReplayCheckpointError, match="checkpoint contract mismatch"):
        replay_paper(client=client, settings=changed, paper_dir=paper_dir)

    assert client.sent == []


def test_resume_rejects_changed_request_settings_before_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)
    monkeypatch.setattr(
        replay_module,
        "VERIFIER_REQUEST_SETTINGS",
        VerifierRequestSettings(
            temperature=0.0,
            reasoning_effort="minimal",
            seed=None,
            require_parameters=True,
        ),
    )

    client = RecordingClient({})
    with pytest.raises(ReplayCheckpointError, match="checkpoint contract mismatch"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []


def test_resume_rejects_changed_request_schema_before_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)
    original = replay_module.verifier_request_contract

    def changed_contract(**kwargs: Any) -> dict[str, Any]:
        contract = json.loads(json.dumps(original(**kwargs)))
        contract["schema"]["schema_sha256"] = "f" * 64
        return contract

    monkeypatch.setattr(replay_module, "verifier_request_contract", changed_contract)
    client = RecordingClient({})
    with pytest.raises(ReplayCheckpointError, match="checkpoint contract mismatch"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []


def test_resume_rejects_changed_prompt_contract_before_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)
    monkeypatch.setattr(
        replay_module,
        "VERIFIER_SYSTEM_PROMPT",
        f"{replay_module.VERIFIER_SYSTEM_PROMPT}\nChanged verifier instruction.",
    )

    client = RecordingClient({})
    with pytest.raises(ReplayCheckpointError, match="checkpoint contract mismatch"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []


def test_resume_rejects_changed_code_binding_before_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)
    original = replay_module._replay_code_binding()
    monkeypatch.setattr(
        replay_module,
        "_replay_code_binding",
        lambda: {**original, "source_tree_sha256": "f" * 64},
    )

    client = RecordingClient({})
    with pytest.raises(ReplayCheckpointError, match="checkpoint contract mismatch"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []


@pytest.mark.parametrize("changed_input", ["candidate", "evidence"])
def test_resume_rejects_changed_candidate_or_evidence_binding_before_provider_call(
    tmp_path: Path, changed_input: str
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)
    if changed_input == "candidate":
        candidate = replay_module.read_candidates(paper_dir)[0]
        candidate.extraction_confidence = 0.94
        write_jsonl(paper_dir / "observations.jsonl", [candidate])
    else:
        block = _blocks()[0]
        changed = block.model_copy(update={"result_signal_score": block.result_signal_score + 0.01})
        write_json(paper_dir / "private" / "result-blocks.json", [changed])

    client = RecordingClient({})
    with pytest.raises(ReplayCheckpointError, match="checkpoint contract mismatch"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []


def test_resume_explicitly_rejects_historical_v01_verification_entry(
    tmp_path: Path,
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)
    checkpoint_path = settings.output_root / paper_dir.name / "replay-checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    entry = next(iter(checkpoint["entries"].values()))
    entry["verification"] = {
        "schema_version": "candidate-verification/0.1",
        "observation_id": next(iter(checkpoint["entries"])),
        "decision": "accept",
    }
    entry["entry_sha256"] = replay_module._json_sha256(
        {key: value for key, value in entry.items() if key != "entry_sha256"}
    )
    write_json(checkpoint_path, checkpoint)

    client = RecordingClient({})
    with pytest.raises(ReplayCheckpointError, match="unsupported historical verifier result"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []


def test_resume_rejects_call_paired_with_a_different_response_assessment(
    tmp_path: Path,
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)
    checkpoint_path = settings.output_root / paper_dir.name / "replay-checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    entry = next(iter(checkpoint["entries"].values()))
    entry["provider_call"]["response_sha256"] = "f" * 64
    entry["entry_sha256"] = replay_module._json_sha256(
        {key: value for key, value in entry.items() if key != "entry_sha256"}
    )
    write_json(checkpoint_path, checkpoint)

    client = RecordingClient({})
    with pytest.raises(ReplayCheckpointError, match="response hash does not match"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []


def test_resume_rejects_rehashed_returned_model_mutation(
    tmp_path: Path,
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)
    checkpoint_path = settings.output_root / paper_dir.name / "replay-checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    entry = next(iter(checkpoint["entries"].values()))
    entry["provider_call"]["model_returned"] = "fixture/wrong-model"
    entry["entry_sha256"] = replay_module._json_sha256(
        {key: value for key, value in entry.items() if key != "entry_sha256"}
    )
    write_json(checkpoint_path, checkpoint)

    client = RecordingClient({})
    with pytest.raises(ReplayCheckpointError, match="immutable binding"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []


def test_historical_v01_jsonl_without_checkpoint_is_rejected_not_resumed(
    tmp_path: Path,
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    output_dir = settings.output_root / paper_dir.name
    output_dir.mkdir(parents=True)
    write_jsonl(
        output_dir / "verifications.jsonl",
        [
            {
                "schema_version": "candidate-verification/0.1",
                "observation_id": replay_module.read_candidates(paper_dir)[0].observation_id,
                "decision": "accept",
            }
        ],
    )

    client = RecordingClient({})
    with pytest.raises(ReplayCheckpointError, match="unsupported historical or torn artifacts"):
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []


def test_resume_reprojects_torn_success_call_ledger_without_resending(tmp_path: Path) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    replay_paper(client=RecordingClient({}), settings=settings, paper_dir=paper_dir)
    calls_path = settings.output_root / paper_dir.name / "verifier-calls.jsonl"
    calls_path.write_text("", encoding="utf-8")

    client = RecordingClient({})
    summary = replay_paper(client=client, settings=settings, paper_dir=paper_dir)

    assert client.sent == []
    assert summary["cost"]["calls"] == 1
    call = json.loads(calls_path.read_text(encoding="utf-8"))
    assert "request_id" not in call
    assert call["request_id_observed"] is True
    assert call["temperature"] is None
    assert call["seed"] is None


class ResponseValidationFailureClient(RecordingClient):
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        raise ProviderResponseValidationError(
            call=response.call,
            code="schema_validation",
            validation_keyword="required",
        )


class MismatchedResponseHashClient(RecordingClient):
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        return StructuredResponse(
            payload=response.payload,
            call=response.call.model_copy(update={"response_sha256": "f" * 64}),
        )


class MismatchedReturnedModelClient(RecordingClient):
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        return StructuredResponse(
            payload=response.payload,
            call=response.call.model_copy(update={"model_returned": "fixture/wrong-model"}),
        )


class InconsistentAssessmentClient(RecordingClient):
    def structured_chat(self, **kwargs: Any) -> StructuredResponse:
        response = super().structured_chat(**kwargs)
        payload = {
            **response.payload,
            "support": "contradicted",
            "decision": IndependentDecision.ACCEPT.value,
        }
        response_sha256 = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return StructuredResponse(
            payload=payload,
            call=response.call.model_copy(update={"response_sha256": response_sha256}),
        )


def test_response_hash_mismatch_is_terminal_with_atomic_call_telemetry(
    tmp_path: Path,
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    first = MismatchedResponseHashClient({})
    summary = replay_paper(client=first, settings=settings, paper_dir=paper_dir)
    assert summary["verifications"] == 0
    assert summary["errors"] == 1
    assert summary["cost"]["calls"] == 1

    output_dir = settings.output_root / paper_dir.name
    checkpoint = json.loads((output_dir / "replay-checkpoint.json").read_text(encoding="utf-8"))
    entry = next(iter(checkpoint["entries"].values()))
    assert entry["status"] == "contract_failure"
    assert entry["provider_call"]["response_sha256"] == "f" * 64

    second = MismatchedResponseHashClient({})
    resumed = replay_paper(client=second, settings=settings, paper_dir=paper_dir)
    assert second.sent == []
    assert resumed["errors"] == 1
    assert resumed["cost"]["calls"] == 1


def test_returned_model_mismatch_is_terminal_with_atomic_call_telemetry(
    tmp_path: Path,
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    first = MismatchedReturnedModelClient({})
    summary = replay_paper(client=first, settings=settings, paper_dir=paper_dir)
    assert summary["verifications"] == 0
    assert summary["errors"] == 1
    assert summary["cost"]["calls"] == 1

    output_dir = settings.output_root / paper_dir.name
    checkpoint = json.loads((output_dir / "replay-checkpoint.json").read_text(encoding="utf-8"))
    entry = next(iter(checkpoint["entries"].values()))
    assert entry["status"] == "contract_failure"
    assert entry["error"]["code"] == "returned_model_mismatch"
    assert entry["provider_call"]["model_returned"] == "fixture/wrong-model"

    second = MismatchedReturnedModelClient({})
    resumed = replay_paper(client=second, settings=settings, paper_dir=paper_dir)
    assert second.sent == []
    assert resumed["errors"] == 1
    assert resumed["cost"]["calls"] == 1


def test_response_validation_failure_is_atomic_and_torn_ledger_is_reprojected(
    tmp_path: Path,
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    first = ResponseValidationFailureClient({})
    summary = replay_paper(client=first, settings=settings, paper_dir=paper_dir)
    assert summary["errors"] == 1
    assert summary["cost"]["calls"] == 1

    output_dir = settings.output_root / paper_dir.name
    checkpoint = json.loads((output_dir / "replay-checkpoint.json").read_text(encoding="utf-8"))
    entry = next(iter(checkpoint["entries"].values()))
    assert entry["status"] == "response_failure"
    assert entry["provider_call"] is not None
    assert "request_id" not in entry["provider_call"]
    assert entry["request_id_observed"] is True
    assert entry["error"]["error"] == "provider_response_validation"
    (output_dir / "verifier-calls.jsonl").write_text("", encoding="utf-8")

    second = ResponseValidationFailureClient({})
    resumed = replay_paper(client=second, settings=settings, paper_dir=paper_dir)
    assert second.sent == []
    assert resumed["errors"] == 1
    assert resumed["cost"]["calls"] == 1
    assert (output_dir / "verifier-calls.jsonl").read_text(encoding="utf-8").strip()


def test_post_response_verifier_validation_keeps_paid_call_and_resumes(
    tmp_path: Path,
) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    first = InconsistentAssessmentClient({})
    summary = replay_paper(client=first, settings=settings, paper_dir=paper_dir)

    assert len(first.sent) == 1
    assert summary["verifications"] == 0
    assert summary["errors"] == 1
    assert summary["cost"]["calls"] == 1
    output_dir = settings.output_root / paper_dir.name
    checkpoint = json.loads((output_dir / "replay-checkpoint.json").read_text(encoding="utf-8"))
    entry = next(iter(checkpoint["entries"].values()))
    assert entry["status"] == "response_failure"
    assert entry["provider_call"] is not None
    assert entry["request_id_observed"] is True
    assert entry["error"]["code"] == "wire_validation"
    assert entry["error"]["validation_keyword"] == "candidate_verification_assessment"

    second = InconsistentAssessmentClient({})
    resumed = replay_paper(client=second, settings=settings, paper_dir=paper_dir)
    assert second.sent == []
    assert resumed["errors"] == 1
    assert resumed["cost"]["calls"] == 1


def test_measurement_joins_verdicts_to_the_frozen_reference_score(tmp_path: Path) -> None:
    run_root = tmp_path / "sealed"
    good = _candidate(
        "Paper System                            0.742",
        export_status=ExportStatus.EXPORTED,
        row="Paper System",
    )
    bad = _candidate(
        "Example Ensemble (Leaderboard Entry)    0.781",
        export_status=ExportStatus.EXPORTED,
        row="Example Ensemble (Leaderboard Entry)",
    )
    paper_dir = _write_paper_run(run_root, [good, bad])
    score_sha256 = write_json(
        paper_dir / "reference-score.json",
        {
            "schema_version": "reference-score/0.7",
            "paper_id": paper_dir.name,
            "matches": [{"observation_id": good.observation_id, "joint_semantics": True}],
            "unmatched_primary_candidate_ids_in_coverage": [bad.observation_id],
            "negative_control_safety": {
                "matched_candidate_ids": [bad.observation_id],
                "false_primary_candidate_ids": [bad.observation_id],
                "false_primary_export_candidate_ids": [bad.observation_id],
            },
        },
    )
    write_json(
        paper_dir / "run.json",
        {
            "paper_id": "synthetic-replay-fixture",
            "status": "success",
            "reference_evaluation": {
                "schema_version": "reference-score/0.7",
                "score_path": "reference-score.json",
                "score_sha256": score_sha256,
            },
        },
    )
    settings = ReplaySettings(
        run_root=run_root,
        output_root=tmp_path / "replay",
        verifier_model="fixture/verifier",
        concurrency=1,
    )
    client = RecordingClient({LEADERBOARD_ROW_KEY: IndependentDecision.REJECT.value})
    replay_run(client=client, settings=settings)

    report = measure_replay(run_root=run_root, replay_root=tmp_path / "replay")
    assert report["classes"]["reference_matched"]["accept"] == 1
    assert report["classes"]["false_primary_export"]["reject"] == 1
    assert report["headline"]["true_positive_retention"] == pytest.approx(1.0)
    assert report["headline"]["false_primary_caught"] == pytest.approx(1.0)
    assert (tmp_path / "replay" / "verifier-replay-measurement.json").is_file()


@pytest.mark.parametrize(
    ("score_update", "expected_error"),
    [
        ({"paper_id": "different-paper"}, "unsupported schema or paper binding"),
        ({"schema_version": "reference-score/0.6"}, "unsupported schema or paper binding"),
        (
            {"matches": [{"observation_id": "obs", "joint_semantics": "yes"}]},
            "malformed match record",
        ),
        (
            {"unmatched_primary_candidate_ids_in_coverage": "obs"},
            "must contain unique observation IDs",
        ),
    ],
)
def test_measurement_rejects_misbound_or_malformed_reference_score(
    tmp_path: Path,
    score_update: dict[str, Any],
    expected_error: str,
) -> None:
    run_root = tmp_path / "sealed"
    candidate = _candidate(
        "Paper System                            0.742",
        export_status=ExportStatus.EXPORTED,
        row="Paper System",
    )
    paper_dir = _write_paper_run(run_root, [candidate])
    score: dict[str, Any] = {
        "schema_version": "reference-score/0.7",
        "paper_id": paper_dir.name,
        "matches": [{"observation_id": candidate.observation_id, "joint_semantics": True}],
        "unmatched_primary_candidate_ids_in_coverage": [],
        "negative_control_safety": {
            "matched_candidate_ids": [],
            "false_primary_candidate_ids": [],
            "false_primary_export_candidate_ids": [],
        },
    }
    score.update(score_update)
    score_sha256 = write_json(paper_dir / "reference-score.json", score)
    write_json(
        paper_dir / "run.json",
        {
            "paper_id": paper_dir.name,
            "status": "success",
            "reference_evaluation": {
                "schema_version": score["schema_version"],
                "score_path": "reference-score.json",
                "score_sha256": score_sha256,
            },
        },
    )
    settings = ReplaySettings(
        run_root=run_root,
        output_root=tmp_path / "replay",
        verifier_model="fixture/verifier",
        concurrency=1,
    )
    replay_run(client=RecordingClient({}), settings=settings)

    with pytest.raises(ReplayCheckpointError, match=expected_error):
        measure_replay(run_root=run_root, replay_root=settings.output_root)


def test_measurement_rejects_same_named_but_different_source_run(tmp_path: Path) -> None:
    source_run = tmp_path / "source-a" / "sealed"
    candidate = _candidate(
        "Paper System                            0.742",
        export_status=ExportStatus.EXPORTED,
        row="Paper System",
    )
    _write_paper_run(source_run, [candidate])
    replay_root = tmp_path / "replay"
    settings = ReplaySettings(
        run_root=source_run,
        output_root=replay_root,
        verifier_model="fixture/verifier",
        concurrency=1,
    )
    replay_run(client=RecordingClient({}), settings=settings)

    different_run = tmp_path / "source-b" / "sealed"
    changed = candidate.model_copy(deep=True)
    changed.extraction_confidence = 0.94
    _write_paper_run(different_run, [changed])

    with pytest.raises(ReplayCheckpointError, match="does not bind the supplied source run"):
        measure_replay(run_root=different_run, replay_root=replay_root)


def test_measurement_rejects_tampered_summary_projection(tmp_path: Path) -> None:
    paper_dir, settings = _single_candidate_replay(tmp_path)
    replay_run(client=RecordingClient({}), settings=settings)
    summary_path = settings.output_root / "verifier-replay.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["totals"]["bound"] += 1
    write_json(summary_path, summary)

    with pytest.raises(ReplayCheckpointError, match="totals/cost"):
        measure_replay(run_root=paper_dir.parent, replay_root=settings.output_root)


def test_measurement_rejects_cross_paper_duplicate_observation_ids(tmp_path: Path) -> None:
    run_root = tmp_path / "sealed"
    duplicate_id = "obs_duplicate_across_papers"
    for paper_id in ("paper-one", "paper-two"):
        candidate = _candidate(
            "Paper System                            0.742",
            export_status=ExportStatus.EXPORTED,
            row="Paper System",
        )
        candidate.paper_id = paper_id
        candidate.observation_id = duplicate_id
        _write_named_paper_run(run_root, paper_id, [candidate])
    settings = ReplaySettings(
        run_root=run_root,
        output_root=tmp_path / "replay",
        verifier_model="fixture/verifier",
        concurrency=1,
    )
    replay_run(client=RecordingClient({}), settings=settings)

    with pytest.raises(ReplayCheckpointError, match="cross-paper duplicate observation IDs"):
        measure_replay(run_root=run_root, replay_root=settings.output_root)
