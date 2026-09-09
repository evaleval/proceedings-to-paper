"""Resumable, budget-bounded driver that runs one corpus paper per worker.

`run_corpus` processes a corpus serially against one shared budget ledger. This becomes
expensive for a large census: the shared ledger re-reads, re-parses and
re-hashes itself on every call, so per-call overhead grows with the run, and a serial
pass over papers that each take minutes would need days.

This driver keeps the per-paper pipeline exactly as it is and changes only what surrounds
it. Each paper gets its own budget ledger, so ledger cost stays flat. Papers already
finished under the same extraction contract are skipped without a provider call. A global
guard stops dispatching before the run can exceed its cost ceiling, and a signal drains
the pool instead of killing workers mid-call, because a killed call leaves a reservation
that is charged forever with no provider telemetry behind it.
"""

from __future__ import annotations

import contextlib
import json
import signal
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from proceedings_to_eee.corpus import CorpusSpec, PaperSpec, build_corpus_binding
from proceedings_to_eee.extraction.llm_schema import (
    provider_json_schema,
    row_provider_json_schema,
)
from proceedings_to_eee.extraction.prompt import prompt_hash, row_prompt_hash
from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes, sha256_file, write_json
from proceedings_to_eee.pipeline import PipelineSettings, _safe_error_message, run_paper
from proceedings_to_eee.providers.budget import (
    BudgetedProviderClient,
    ProviderBudgetContractError,
    ProviderBudgetExhausted,
    ProviderBudgetLimits,
    provider_budget_contract,
)

CENSUS_LEDGER_SCHEMA_VERSION = "census-ledger/0.1"
CENSUS_PROGRESS_SCHEMA_VERSION = "census-progress/0.1"

#: Terminal states a paper can reach. Every ordered paper ends in exactly one of them.
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"
SOURCE_UNAVAILABLE = "source_unavailable"
NOT_STARTED = "not_started"
BUDGET_STOPPED = "budget_stopped"
RETRY_CAPPED = "retry_capped"


@dataclass(frozen=True)
class CensusLimits:
    """Bounds for one census leg. Per-paper limits bound a paper; the rest bound the run."""

    workers: int = 16
    max_cost_usd: float = 25.0
    per_paper_max_cost_usd: float = 1.5
    per_paper_max_calls: int = 400
    reservation_usd: float = 0.05
    max_technical_retries: int = 2

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.per_paper_max_cost_usd <= self.reservation_usd:
            raise ValueError("per-paper cost ceiling must exceed the per-call reservation")
        if self.max_cost_usd < self.per_paper_max_cost_usd:
            raise ValueError("global cost ceiling must not be below the per-paper ceiling")


def extraction_contract_key(settings: PipelineSettings, manifest_path: Path) -> str:
    """Identify the work a completed paper represents.

    Deliberately excludes the checkpoint contract, which embeds the git commit, the dirty
    flag and a hash of the whole source tree. Keying on that would mark every finished
    paper stale after any commit, so a census could never be resumed across a code change.
    What actually determines the extraction is the prompt, the wire schema, the model, the
    row configuration and the exact frozen source.
    """

    payload = {
        "schema_version": "census-skip-key/0.1",
        "model": settings.model,
        "min_confidence": settings.min_confidence,
        "max_result_pages_default": None,
        "max_blocks_per_page": settings.max_blocks_per_page,
        "row_enumeration_enabled": settings.row_enumeration_enabled,
        "row_config": _row_config_payload(settings),
        "prompt_sha256": prompt_hash(),
        "row_prompt_sha256": row_prompt_hash(),
        "provider_schema_sha256": sha256_bytes(canonical_json_bytes(provider_json_schema())),
        "row_provider_schema_sha256": sha256_bytes(
            canonical_json_bytes(row_provider_json_schema())
        ),
        "source_manifest_sha256": sha256_file(manifest_path),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _row_config_payload(settings: PipelineSettings) -> dict[str, Any]:
    config = settings.row_enumeration_config
    return {key: getattr(config, key) for key in sorted(vars(config)) if not key.startswith("_")}


def _paper_dir(settings: PipelineSettings, paper_id: str) -> Path:
    return settings.output_root / paper_id


def paper_state(
    settings: PipelineSettings, spec: PaperSpec, *, expected_key: str | None = None
) -> tuple[str, dict[str, Any] | None]:
    """Classify one paper before any provider call is considered."""

    directory = _paper_dir(settings, spec.paper_id)
    manifest = directory / "source-manifest.json"
    if not manifest.is_file():
        return SOURCE_UNAVAILABLE, None
    run_path = directory / "run.json"
    if not run_path.is_file():
        return NOT_STARTED, None
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return NOT_STARTED, None
    if run.get("status") not in {"success", "partial_failure"}:
        return NOT_STARTED, run
    if expected_key is not None and run.get("census_contract_key") != expected_key:
        return NOT_STARTED, run
    return SKIPPED, run


ATTEMPTS_FILENAME = "technical-attempts.json"


def _attempts_path(directory: Path) -> Path:
    return directory / "private" / ATTEMPTS_FILENAME


def technical_attempts(directory: Path) -> int:
    """Count consecutive launches that did not leave a usable run.json.

    A failed or ceiling-stopped paper writes no run.json, so counting superseded run
    files alone misses exactly the papers a retry cap exists for: resume would re-launch
    them forever, re-charging their prior spend to the guard each time. The driver
    therefore records each launch and clears the record once a paper produces a run. A
    directory written before this existed is read from its superseded run files, so
    history still counts.
    """

    recorded = 0
    path = _attempts_path(directory)
    if path.is_file():
        try:
            recorded = int(json.loads(path.read_text(encoding="utf-8")).get("attempts") or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            recorded = 0
    superseded = len(list(directory.glob("run.superseded-*.json")))
    return max(recorded, superseded)


def _record_attempt(directory: Path, attempts: int) -> None:
    path = _attempts_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"schema_version": "census-technical-attempts/0.1", "attempts": attempts})


def _clear_attempts(directory: Path) -> None:
    _attempts_path(directory).unlink(missing_ok=True)


def supersede_stale_ledger(directory: Path) -> str | None:
    """Retire a budget ledger written under a different run contract.

    A ledger binds its run contract, so a paper that never finished cannot be resumed
    after the contract legitimately changes: a new schema version, a new reasoning
    setting, a new export policy. Refusing to run the paper would strand it forever;
    overwriting the ledger would destroy the record of money already spent. So the file
    is retired beside its replacement, exactly as run.json is superseded, and cost
    accounting reads superseded ledgers too.
    """

    ledger = directory / "private" / "provider-budget-ledger.jsonl"
    if not ledger.is_file():
        return None
    index = 1
    while (directory / "private" / f"provider-budget-ledger.superseded-{index}.jsonl").exists():
        index += 1
    retired = directory / "private" / f"provider-budget-ledger.superseded-{index}.jsonl"
    ledger.rename(retired)
    head = Path(f"{ledger}.head.json")
    if head.is_file():
        head.rename(Path(f"{retired}.head.json"))
    return retired.name


def _supersede(directory: Path) -> None:
    """Preserve a previous run instead of silently overwriting it."""

    run_path = directory / "run.json"
    if not run_path.is_file():
        return
    index = 1
    while (directory / f"run.superseded-{index}.json").exists():
        index += 1
    run_path.rename(directory / f"run.superseded-{index}.json")


class _LedgerWriter:
    """Append-only census ledger. One line per finished paper, flushed immediately."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


class _BudgetGuard:
    """Track completed spend and estimated costs of in-flight papers.

    Reserving every in-flight paper at its full ceiling can stop work prematurely when
    typical paper costs are much lower than that ceiling.

    Until enough papers have reported, an in-flight paper is reserved at the ceiling.
    After that, the reservation uses a high quantile of observed paper costs, never
    above the ceiling. The per-paper ledger still enforces each paper's ceiling, so the
    absolute worst case remains bounded and is reported as `worst_case_usd`.
    """

    #: Reports needed before observed cost is trusted over the configured ceiling.
    _MIN_OBSERVATIONS = 5

    def __init__(self, limits: CensusLimits) -> None:
        self._limits = limits
        self._lock = threading.Lock()
        self._reported = 0.0
        self._in_flight = 0
        self._observed: list[float] = []
        self.stopped_for_budget = False

    @property
    def reported(self) -> float:
        with self._lock:
            return self._reported

    @property
    def worst_case(self) -> float:
        """Most this run could still cost if every in-flight paper hit its ceiling."""

        with self._lock:
            return self._reported + self._in_flight * self._limits.per_paper_max_cost_usd

    def _estimate_locked(self) -> float:
        if len(self._observed) < self._MIN_OBSERVATIONS:
            return self._limits.per_paper_max_cost_usd
        ordered = sorted(self._observed)
        index = min(len(ordered) - 1, int(0.95 * len(ordered)))
        p95 = ordered[index]
        mean = sum(ordered) / len(ordered)
        # Two independent headroom rules, whichever is larger, and never above the
        # ceiling the per-paper ledger already enforces.
        return min(self._limits.per_paper_max_cost_usd, max(p95, 2.0 * mean))

    def try_reserve(self) -> bool:
        with self._lock:
            estimate = self._estimate_locked()
            committed = self._reported + self._in_flight * estimate
            if committed + estimate > self._limits.max_cost_usd:
                self.stopped_for_budget = True
                return False
            self._in_flight += 1
            return True

    def settle(self, reported_cost: float) -> None:
        with self._lock:
            self._in_flight -= 1
            self._reported += reported_cost
            if reported_cost > 0:
                self._observed.append(reported_cost)


def _budget_contract(settings: PipelineSettings, corpus: CorpusSpec, limits: ProviderBudgetLimits):
    from proceedings_to_eee.pipeline import (
        _candidate_validation_run_configuration,
        _extractor_run_configuration,
        _origin_run_configuration,
        _row_enumeration_run_configuration,
        _tuple_run_configuration,
        _verifier_run_configuration,
    )

    return provider_budget_contract(
        corpus_binding=build_corpus_binding(corpus),
        provider_run_contract={
            "candidate_validation": _candidate_validation_run_configuration(settings),
            "extractor": _extractor_run_configuration(settings),
            "row_enumeration": _row_enumeration_run_configuration(settings),
            "tuple_resolution": _tuple_run_configuration(settings),
            "verifier": _verifier_run_configuration(settings),
            "origin_retrieval": _origin_run_configuration(settings),
        },
        limits=limits,
    )


def run_census(
    *,
    corpus: CorpusSpec,
    settings: PipelineSettings,
    order: Sequence[str],
    limits: CensusLimits,
    client_factory: Callable[[], Any] | None = None,
    dry_run: bool = False,
    progress_seconds: float = 60.0,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run the ordered papers, resuming what is done and stopping inside the budget."""

    by_id = {paper.paper_id: paper for paper in corpus.papers}
    ordered = [by_id[slug] for slug in order if slug in by_id]
    ledger = _LedgerWriter(settings.output_root / "census-ledger.jsonl")
    progress_path = settings.output_root / "census-progress.json"
    guard = _BudgetGuard(limits)
    counters: dict[str, int] = {
        DONE: 0,
        FAILED: 0,
        SKIPPED: 0,
        SOURCE_UNAVAILABLE: 0,
        NOT_STARTED: 0,
        RETRY_CAPPED: 0,
    }
    counters_lock = threading.Lock()
    draining = threading.Event()
    started_at = now()

    def _record(slug: str, state: str, **fields: Any) -> None:
        with counters_lock:
            counters[state] = counters.get(state, 0) + 1
        ledger.append(
            {
                "schema_version": CENSUS_LEDGER_SCHEMA_VERSION,
                "paper_id": slug,
                "state": state,
                **fields,
            }
        )

    def _write_progress(final: bool = False) -> None:
        with counters_lock:
            snapshot = dict(counters)
        finished = sum(snapshot.values())
        elapsed = max(now() - started_at, 1e-9)
        rate = snapshot[DONE] / elapsed if elapsed else 0.0
        write_json(
            progress_path,
            {
                "schema_version": CENSUS_PROGRESS_SCHEMA_VERSION,
                "ordered": len(ordered),
                "finished": finished,
                "remaining": len(ordered) - finished,
                "counts": snapshot,
                "reported_cost_usd": round(guard.reported, 6),
                "worst_case_usd": round(guard.worst_case, 6),
                "cost_ceiling_usd": limits.max_cost_usd,
                "stopped_for_budget": guard.stopped_for_budget,
                "draining": draining.is_set(),
                "papers_per_second": round(rate, 6),
                "elapsed_seconds": round(elapsed, 3),
                "final": final,
            },
        )

    def _handle_signal(signum: int, _frame: Any) -> None:
        # Do not kill workers: an interrupted call leaves a reservation charged at the
        # full per-call rate with no provider telemetry to reconcile it against.
        draining.set()

    previous_handlers: list[tuple[int, Any]] = []
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                previous_handlers.append((signum, signal.getsignal(signum)))
                signal.signal(signum, _handle_signal)
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                pass

    def _process(spec: PaperSpec) -> None:
        if draining.is_set():
            _record(spec.paper_id, NOT_STARTED, reason="draining")
            return
        directory = _paper_dir(settings, spec.paper_id)
        manifest_path = directory / "source-manifest.json"
        if not manifest_path.is_file():
            _record(spec.paper_id, SOURCE_UNAVAILABLE, reason="no frozen source manifest")
            return
        key = extraction_contract_key(settings, manifest_path)
        state, previous = paper_state(settings, spec, expected_key=key)
        if state == SKIPPED:
            _record(
                spec.paper_id,
                SKIPPED,
                status=(previous or {}).get("status"),
                candidates=((previous or {}).get("counts") or {}).get("candidates"),
            )
            return
        attempts = technical_attempts(directory)
        if attempts >= limits.max_technical_retries:
            _record(
                spec.paper_id,
                RETRY_CAPPED,
                reason="max_technical_retries",
                attempts=attempts,
                max_technical_retries=limits.max_technical_retries,
            )
            return
        if dry_run:
            _record(spec.paper_id, NOT_STARTED, reason="dry_run")
            return
        if not guard.try_reserve():
            _record(spec.paper_id, NOT_STARTED, reason="budget_ceiling")
            return

        # Failed and ceiling-stopped papers may already have spent provider budget.
        # Read each paper's ledger on every outcome so those costs also reach the guard.
        budget_client: Any = None
        superseded_ledger: str | None = None
        state_to_record: tuple[str, dict[str, Any]] | None = None
        try:
            _record_attempt(directory, attempts + 1)
            if previous is not None:
                _supersede(directory)
            paper_limits = ProviderBudgetLimits(
                max_structured_calls=limits.per_paper_max_calls,
                max_cost_usd=limits.per_paper_max_cost_usd,
                cost_reservation_per_call_usd=limits.reservation_usd,
            )
            single = corpus.model_copy(update={"papers": [spec]})
            paper_contract = _budget_contract(settings, single, paper_limits)
            ledger_path = directory / "private" / "provider-budget-ledger.jsonl"
            try:
                budget_client = BudgetedProviderClient(
                    client=(client_factory or _default_client_factory)(),
                    ledger_path=ledger_path,
                    contract=paper_contract,
                    limits=paper_limits,
                )
            except ProviderBudgetContractError:
                # The paper never finished and the contract has since moved. Retire the
                # old ledger rather than stranding the paper or overwriting its record.
                retired = supersede_stale_ledger(directory)
                superseded_ledger = retired
                budget_client = BudgetedProviderClient(
                    client=(client_factory or _default_client_factory)(),
                    ledger_path=ledger_path,
                    contract=paper_contract,
                    limits=paper_limits,
                )
            summary = run_paper(spec=spec, settings=settings, client=budget_client)
            run_path = directory / "run.json"
            if run_path.is_file():
                payload = json.loads(run_path.read_text(encoding="utf-8"))
                payload["census_contract_key"] = key
                write_json(run_path, payload)
            _clear_attempts(directory)
            state_to_record = (
                DONE,
                {
                    "status": summary.get("status"),
                    "candidates": (summary.get("counts") or {}).get("candidates"),
                    "wall_clock_seconds": summary.get("wall_clock_seconds"),
                },
            )
        except ProviderBudgetExhausted:
            state_to_record = (BUDGET_STOPPED, {"reason": "per_paper_ceiling"})
        except Exception as error:  # noqa: BLE001 - a paper must not end the census
            state_to_record = (
                FAILED,
                {"error_type": type(error).__name__, "error": _safe_error_message(error)},
            )
        finally:
            reported = _reported_cost(budget_client)
            guard.settle(reported)
            if state_to_record is not None:
                state, fields = state_to_record
                if superseded_ledger is not None:
                    fields = {**fields, "superseded_ledger": superseded_ledger}
                _record(spec.paper_id, state, reported_cost_usd=round(reported, 6), **fields)

    stop_progress = threading.Event()

    def _progress_loop() -> None:
        while not stop_progress.wait(progress_seconds):
            _write_progress()

    reporter = threading.Thread(target=_progress_loop, daemon=True)
    reporter.start()
    try:
        if limits.workers == 1 or dry_run:
            for spec in ordered:
                _process(spec)
        else:
            with ThreadPoolExecutor(max_workers=limits.workers) as pool:
                for _ in pool.map(_process, ordered):
                    pass
    finally:
        stop_progress.set()
        reporter.join(timeout=5)
        for signum, handler in previous_handlers:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signum, handler)
        _write_progress(final=True)

    with counters_lock:
        snapshot = dict(counters)
    return {
        "schema_version": "census-run/0.1",
        "ordered": len(ordered),
        "counts": snapshot,
        "reported_cost_usd": round(guard.reported, 6),
        "stopped_for_budget": guard.stopped_for_budget,
        "drained": draining.is_set(),
        "ledger": str(settings.output_root / "census-ledger.jsonl"),
        "progress": str(progress_path),
    }


def _reported_cost(budget_client: Any) -> float:
    """Read a paper's provider-reported cost, whatever outcome it reached.

    Returns 0.0 only when there is genuinely nothing to read: no client was built, or
    its ledger cannot be summarised. Never used to mean "this paper was free".
    """

    if budget_client is None:
        return 0.0
    try:
        summary = budget_client.summary or {}
        return float(summary.get("provider_reported_cost_usd") or 0.0)
    except Exception:  # noqa: BLE001 - accounting must not mask the paper's own outcome
        return 0.0


def _default_client_factory() -> Any:
    from proceedings_to_eee.pipeline import runtime_key
    from proceedings_to_eee.providers.openrouter import OpenRouterClient

    return OpenRouterClient(api_key=runtime_key())
