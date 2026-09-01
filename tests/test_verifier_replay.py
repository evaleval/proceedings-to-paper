from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ClaimType, ExportStatus
from proceedings_to_eee.extraction.pdf_layout import PageFragment
from proceedings_to_eee.extraction.result_blocks import segment_page_result_blocks
from proceedings_to_eee.io import write_json, write_jsonl
from proceedings_to_eee.providers.openrouter import ProviderCall, StructuredResponse
from proceedings_to_eee.verification.binding import bind_candidate_block, frozen_evidence_block
from proceedings_to_eee.verification.independent import IndependentDecision
from proceedings_to_eee.verification.replay import (
    ReplayScope,
    ReplaySettings,
    in_replay_scope,
    measure_replay,
    replay_paper,
    replay_run,
)

PAGE_TEXT = """
Table 2. Reported benchmark performance alongside the invented system developed
for this synthetic fixture.

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
        payload = {**findings, "decision": decision, "justification": "fixture verdict"}
        return StructuredResponse(
            payload=payload,
            call=ProviderCall(
                model_requested=kwargs["model"],
                model_returned=kwargs["model"],
                prompt_sha256="b" * 64,
                response_sha256=hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()
                ).hexdigest(),
                temperature=kwargs["temperature"],
                reasoning_effort=kwargs["reasoning_effort"],
                max_tokens=kwargs["max_tokens"],
                seed=kwargs["seed"],
                schema_name=kwargs["schema_name"],
                schema_sha256="c" * 64,
                latency_seconds=0.02,
                input_tokens=1200,
                output_tokens=60,
                total_tokens=1260,
                cost_usd=0.0004,
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
    write_json(
        paper_dir / "reference-score.json",
        {
            "matches": [{"observation_id": good.observation_id, "joint_semantics": True}],
            "unmatched_primary_candidate_ids_in_coverage": [bad.observation_id],
            "negative_control_safety": {
                "matched_candidate_ids": [bad.observation_id],
                "false_primary_candidate_ids": [bad.observation_id],
                "false_primary_export_candidate_ids": [bad.observation_id],
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
