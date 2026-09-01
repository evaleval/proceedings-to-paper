"""Offline replay of the independent verifier over an already completed run tree.

The prototype recorded ``verifier.enabled=false`` for every holdout paper, so no
measurement of the verifier exists. Re-running the whole pipeline to obtain one would
re-run extraction and produce a different candidate set, which would confound the
verifier's effect with extraction variance.

This module instead replays the verifier against a frozen run: it reads the recorded
candidates and the recorded result blocks, binds them with the same deterministic
function the pipeline uses, and issues exactly the verification calls the pipeline
would have issued. The source run tree is opened read-only; every artifact is written
under a separate output root.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ClaimType, ExportStatus
from proceedings_to_eee.extraction.result_blocks import ResultBlock
from proceedings_to_eee.io import read_json, write_json, write_jsonl
from proceedings_to_eee.providers.openrouter import (
    OpenRouterClient,
    ProviderRequestRejectedError,
    ProviderResponseValidationError,
)
from proceedings_to_eee.verification.binding import bind_candidate_block, frozen_evidence_block
from proceedings_to_eee.verification.independent import (
    VERIFIER_SEED,
    IndependentDecision,
    verifier_request_contract,
    verify_candidate,
)

REPLAY_SCHEMA_VERSION = "verifier-replay/0.1"

_OBSERVATIONS = "observations.jsonl"
_RESULT_BLOCKS = Path("private") / "result-blocks.json"
_REFERENCE_SCORE = "reference-score.json"
_RUN = "run.json"


class ReplayScope(StrEnum):
    """Which recorded candidates the replay sends to the verifier."""

    EXPORT_GATE = "export_gate"
    """Exactly the pipeline gate: primary_result candidates that reached export."""

    PRIMARY = "primary"
    """Every primary_result candidate, including those already held for review."""

    ALL = "all"
    """Every recorded candidate."""


_EXPORT_GATE_STATUSES = frozenset({ExportStatus.ELIGIBLE, ExportStatus.EXPORTED})


@dataclass(frozen=True)
class ReplaySettings:
    """Inputs for one replay. ``run_root`` is never written to."""

    run_root: Path
    output_root: Path
    verifier_model: str
    scope: ReplayScope = ReplayScope.EXPORT_GATE
    max_tokens: int = 2_000
    concurrency: int = 4
    paper_ids: tuple[str, ...] = ()
    max_candidates_per_paper: int | None = None

    def __post_init__(self) -> None:
        if not self.verifier_model.strip():
            raise ValueError("verifier model is required")
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.max_candidates_per_paper is not None and self.max_candidates_per_paper < 1:
            raise ValueError("max_candidates_per_paper must be positive when provided")


def in_replay_scope(candidate: CandidateObservation, scope: ReplayScope) -> bool:
    """Decide membership without inspecting any reference annotation."""

    if scope is ReplayScope.ALL:
        return True
    if candidate.claim_type != ClaimType.PRIMARY_RESULT:
        return False
    if scope is ReplayScope.PRIMARY:
        return True
    return candidate.export_status in _EXPORT_GATE_STATUSES


def read_candidates(paper_dir: Path) -> list[CandidateObservation]:
    path = paper_dir / _OBSERVATIONS
    if not path.is_file():
        return []
    return [
        CandidateObservation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_result_blocks(paper_dir: Path) -> list[ResultBlock]:
    path = paper_dir / _RESULT_BLOCKS
    if not path.is_file():
        return []
    return [ResultBlock.model_validate(item) for item in read_json(path)]


def discover_paper_dirs(run_root: Path, paper_ids: tuple[str, ...] = ()) -> list[Path]:
    """Return paper directories of a corpus run in deterministic order."""

    candidates = sorted(
        path for path in run_root.iterdir() if path.is_dir() and (path / _RUN).is_file()
    )
    if not paper_ids:
        return candidates
    wanted = set(paper_ids)
    selected = [path for path in candidates if path.name in wanted]
    missing = wanted - {path.name for path in selected}
    if missing:
        raise ValueError(f"run root has no paper directories for {sorted(missing)}")
    return selected


@dataclass(frozen=True)
class BindingRecord:
    """Why one in-scope candidate did or did not reach the verifier."""

    observation_id: str
    bound: bool
    block_id: str | None
    page: int | None
    export_status: str
    claim_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "bound": self.bound,
            "block_id": self.block_id,
            "page": self.page,
            "export_status": self.export_status,
            "claim_type": self.claim_type,
        }


def _completed_observation_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        observation_id = json.loads(line).get("observation_id")
        if observation_id is not None:
            ids.add(observation_id)
    return ids


def _append_line(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def _rewrite_sorted(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows.sort(key=lambda row: str(row.get("observation_id", "")))
    write_jsonl(path, rows)
    return rows


def replay_paper(
    *,
    client: OpenRouterClient,
    settings: ReplaySettings,
    paper_dir: Path,
) -> dict[str, Any]:
    """Replay one paper. Resumable: candidates already recorded are not re-sent."""

    paper_id = paper_dir.name
    output_dir = settings.output_root / paper_id
    output_dir.mkdir(parents=True, exist_ok=True)
    verifications_path = output_dir / "verifications.jsonl"
    calls_path = output_dir / "verifier-calls.jsonl"
    errors_path = output_dir / "verifier-errors.jsonl"

    candidates = read_candidates(paper_dir)
    blocks = read_result_blocks(paper_dir)
    in_scope = [item for item in candidates if in_replay_scope(item, settings.scope)]
    if settings.max_candidates_per_paper is not None:
        in_scope = in_scope[: settings.max_candidates_per_paper]

    bindings: list[BindingRecord] = []
    work: list[tuple[CandidateObservation, ResultBlock, Any]] = []
    for candidate in in_scope:
        support = bind_candidate_block(candidate, blocks)
        observation_id = candidate.observation_id or candidate.stable_id()
        if support is None:
            bindings.append(
                BindingRecord(
                    observation_id=observation_id,
                    bound=False,
                    block_id=None,
                    page=candidate.evidence[0].page,
                    export_status=str(candidate.export_status),
                    claim_type=str(candidate.claim_type),
                )
            )
            continue
        block, anchor = support
        bindings.append(
            BindingRecord(
                observation_id=observation_id,
                bound=True,
                block_id=block.block_id,
                page=block.page,
                export_status=str(candidate.export_status),
                claim_type=str(candidate.claim_type),
            )
        )
        work.append((candidate, block, anchor))

    already_done = _completed_observation_ids(verifications_path) | _completed_observation_ids(
        errors_path
    )
    pending = [
        item for item in work if (item[0].observation_id or item[0].stable_id()) not in already_done
    ]

    lock = threading.Lock()

    def run_one(item: tuple[CandidateObservation, ResultBlock, Any]) -> None:
        candidate, block, anchor = item
        observation_id = candidate.observation_id or candidate.stable_id()
        evidence_block = frozen_evidence_block(paper_id=paper_id, block=block, anchor=anchor)
        try:
            verification, call = verify_candidate(
                client=client,
                model=settings.verifier_model,
                candidate=candidate,
                evidence_block=evidence_block,
                max_tokens=settings.max_tokens,
            )
        except ProviderResponseValidationError as error:
            _append_line(
                errors_path,
                {
                    "observation_id": observation_id,
                    "block_id": block.block_id,
                    "error": "provider_response_validation",
                    "code": error.code,
                },
                lock,
            )
            _append_line(
                calls_path,
                {
                    "observation_id": observation_id,
                    **error.call.model_dump(mode="json", exclude_none=True),
                },
                lock,
            )
            return
        except ProviderRequestRejectedError as error:
            _append_line(
                errors_path,
                {
                    "observation_id": observation_id,
                    "block_id": block.block_id,
                    "error": "provider_request_rejected",
                    "code": str(error.status_code),
                },
                lock,
            )
            return
        except (RuntimeError, ValueError) as error:
            _append_line(
                errors_path,
                {
                    "observation_id": observation_id,
                    "block_id": block.block_id,
                    "error": type(error).__name__,
                    "code": "call_failed",
                },
                lock,
            )
            return
        _append_line(
            verifications_path, verification.model_dump(mode="json", exclude_none=True), lock
        )
        _append_line(
            calls_path,
            {
                "observation_id": observation_id,
                **call.model_dump(mode="json", exclude_none=True),
            },
            lock,
        )

    if pending:
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            list(pool.map(run_one, pending))

    verification_rows = _rewrite_sorted(verifications_path)
    error_rows = _rewrite_sorted(errors_path)
    call_rows = _rewrite_sorted(calls_path)
    write_json(output_dir / "bindings.json", [record.as_dict() for record in bindings])

    decisions = {value.value: 0 for value in IndependentDecision}
    for row in verification_rows:
        decisions[row["decision"]] += 1
    return {
        "paper_id": paper_id,
        "candidates_recorded": len(candidates),
        "candidates_in_scope": len(in_scope),
        "bound": sum(1 for record in bindings if record.bound),
        "unbound": sum(1 for record in bindings if not record.bound),
        "verifications": len(verification_rows),
        "errors": len(error_rows),
        "decisions": decisions,
        "cost": _call_totals(call_rows),
    }


def _call_totals(call_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate secret-free call telemetry. Missing fields make totals lower bounds."""

    reported_cost = [row for row in call_rows if row.get("cost_usd") is not None]
    reported_tokens = [row for row in call_rows if row.get("total_tokens") is not None]
    return {
        "calls": len(call_rows),
        "cost_reported_calls": len(reported_cost),
        "cost_usd_lower_bound": round(sum(row["cost_usd"] for row in reported_cost), 7),
        "token_reported_calls": len(reported_tokens),
        "input_tokens_lower_bound": sum(row.get("input_tokens") or 0 for row in call_rows),
        "output_tokens_lower_bound": sum(row.get("output_tokens") or 0 for row in call_rows),
        "total_tokens_lower_bound": sum(row.get("total_tokens") or 0 for row in call_rows),
        "latency_seconds_total": round(
            sum(row.get("latency_seconds") or 0.0 for row in call_rows), 6
        ),
        "attempts_lower_bound": sum(row.get("attempts") or 0 for row in call_rows),
        "basis": (
            "Sums cover calls that returned the field. Absent provider metadata is not "
            "reconstructed, so monetary and token totals are lower bounds."
        ),
    }


def replay_run(*, client: OpenRouterClient, settings: ReplaySettings) -> dict[str, Any]:
    """Replay every selected paper and write a single reproducible summary."""

    paper_dirs = discover_paper_dirs(settings.run_root, settings.paper_ids)
    settings.output_root.mkdir(parents=True, exist_ok=True)
    papers = [
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)
        for paper_dir in paper_dirs
    ]
    decisions = {value.value: 0 for value in IndependentDecision}
    for paper in papers:
        for key, count in paper["decisions"].items():
            decisions[key] += count
    all_calls: list[dict[str, Any]] = []
    for paper in papers:
        calls_path = settings.output_root / paper["paper_id"] / "verifier-calls.jsonl"
        if calls_path.is_file():
            all_calls.extend(
                json.loads(line)
                for line in calls_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    summary = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "run_root": settings.run_root.name,
        "scope": settings.scope.value,
        "verifier": {
            "enabled": True,
            "model": settings.verifier_model,
            "max_tokens": settings.max_tokens,
            "seed": VERIFIER_SEED,
            "request_contract": verifier_request_contract(),
        },
        "totals": {
            "papers": len(papers),
            "candidates_recorded": sum(item["candidates_recorded"] for item in papers),
            "candidates_in_scope": sum(item["candidates_in_scope"] for item in papers),
            "bound": sum(item["bound"] for item in papers),
            "unbound": sum(item["unbound"] for item in papers),
            "verifications": sum(item["verifications"] for item in papers),
            "errors": sum(item["errors"] for item in papers),
            "decisions": decisions,
        },
        "cost": _call_totals(all_calls),
        "papers": papers,
    }
    write_json(settings.output_root / "verifier-replay.json", summary)
    return summary


# --------------------------------------------------------------------------------------
# Measurement: what the verifier actually catches, joined to the frozen reference score.
# --------------------------------------------------------------------------------------


REFERENCE_CLASSES = (
    "reference_matched",
    "reference_matched_joint_semantics",
    "unmatched_primary_in_coverage",
    "control_matched",
    "false_primary",
    "false_primary_export",
)
"""Candidate partitions taken verbatim from the run's own frozen reference score.

``reference_matched`` is the recall numerator: every candidate the scorer paired with an
annotated reference observation. ``reference_matched_joint_semantics`` is the stricter
subset whose system, dataset, metric, value, unit, and slice all agreed.
"""


def _reference_classes(paper_dir: Path) -> dict[str, set[str]]:
    """Partition recorded observation IDs using only the already-frozen scoring output."""

    path = paper_dir / _REFERENCE_SCORE
    empty: dict[str, set[str]] = {name: set() for name in REFERENCE_CLASSES}
    if not path.is_file():
        return empty
    score = read_json(path)
    safety = score.get("negative_control_safety") or {}
    return {
        "reference_matched": {match["observation_id"] for match in score.get("matches") or []},
        "reference_matched_joint_semantics": {
            match["observation_id"]
            for match in score.get("matches") or []
            if match.get("joint_semantics")
        },
        "unmatched_primary_in_coverage": set(
            score.get("unmatched_primary_candidate_ids_in_coverage") or []
        ),
        "control_matched": set(safety.get("matched_candidate_ids") or []),
        "false_primary": set(safety.get("false_primary_candidate_ids") or []),
        "false_primary_export": set(safety.get("false_primary_export_candidate_ids") or []),
    }


def _class_decisions(ids: set[str], decisions: dict[str, str], unbound: set[str]) -> dict[str, Any]:
    counts = {value.value: 0 for value in IndependentDecision}
    unverified = 0
    for observation_id in ids:
        decision = decisions.get(observation_id)
        if decision is None:
            unverified += 1
            continue
        counts[decision] += 1
    return {
        "total": len(ids),
        "verified": len(ids) - unverified,
        "unverified": unverified,
        "unbound": len(ids & unbound),
        **counts,
    }


def measure_replay(*, run_root: Path, replay_root: Path) -> dict[str, Any]:
    """Join replay verdicts to the frozen reference score without re-scoring anything.

    Every class below is defined by the sealed run's own ``reference-score.json``. The
    replay never sees a reference annotation, so this join happens strictly after the
    verifier has committed to its verdicts.
    """

    summary_path = replay_root / "verifier-replay.json"
    if not summary_path.is_file():
        raise ValueError(f"no replay summary at {summary_path}")
    replay = read_json(summary_path)

    classes = REFERENCE_CLASSES
    aggregate: dict[str, set[str]] = {name: set() for name in classes}
    all_decisions: dict[str, str] = {}
    all_unbound: set[str] = set()
    per_paper: list[dict[str, Any]] = []

    for entry in replay["papers"]:
        paper_id = entry["paper_id"]
        paper_dir = run_root / paper_id
        output_dir = replay_root / paper_id
        decisions: dict[str, str] = {}
        verifications_path = output_dir / "verifications.jsonl"
        if verifications_path.is_file():
            for line in verifications_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                decisions[row["observation_id"]] = row["decision"]
        bindings_path = output_dir / "bindings.json"
        unbound = set()
        if bindings_path.is_file():
            unbound = {
                record["observation_id"]
                for record in read_json(bindings_path)
                if not record["bound"]
            }
        reference = _reference_classes(paper_dir)
        per_paper.append(
            {
                "paper_id": paper_id,
                "classes": {
                    name: _class_decisions(reference[name], decisions, unbound) for name in classes
                },
            }
        )
        for name in classes:
            aggregate[name] |= reference[name]
        all_decisions.update(decisions)
        all_unbound |= unbound

    corpus = {
        name: _class_decisions(aggregate[name], all_decisions, all_unbound) for name in classes
    }
    matched = corpus["reference_matched"]
    controls = corpus["false_primary"]
    retained = matched[IndependentDecision.ACCEPT.value]
    caught = controls[IndependentDecision.REJECT.value] + controls[IndependentDecision.REVIEW.value]
    report = {
        "schema_version": "verifier-replay-measurement/0.1",
        "run_root": run_root.name,
        "replay_root": replay_root.name,
        "scope": replay["scope"],
        "verifier_model": replay["verifier"]["model"],
        "totals": replay["totals"],
        "cost": replay["cost"],
        "classes": corpus,
        "headline": {
            "true_positive_retention": (
                retained / matched["verified"] if matched["verified"] else None
            ),
            "true_positive_retention_basis": matched["verified"],
            "false_primary_caught": (
                caught / controls["verified"] if controls["verified"] else None
            ),
            "false_primary_caught_basis": controls["verified"],
            "basis_note": (
                "Denominators are the frozen reference annotation only: one audited primary "
                "target and one negative control per paper. These are not whole-paper rates."
            ),
        },
        "papers": per_paper,
    }
    write_json(replay_root / "verifier-replay-measurement.json", report)
    return report
