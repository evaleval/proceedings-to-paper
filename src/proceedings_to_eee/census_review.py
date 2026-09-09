"""Page-scoped model review of census candidates, with a deterministic local defence.

The extractor proposes; deterministic code decides. This adds a second model stage with
the same discipline: a reviewer reads one page and one candidate and may accept, accept
the tuple only, reject, or abstain, and every positive claim it makes must be backed by a
quotation this module then checks against the page itself. An accept whose quotes do not
survive that check becomes an abstention, never a rejection, because a reviewer that
cannot show its evidence has told us nothing, not something negative.

What the provider never sees: census columns, gate outcomes, attribution state,
`claim_type`, candidate notes. The reviewer is asked what the page says, not what the
pipeline already concluded.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proceedings_to_eee.census_audit import _normalize, quotation_is_on_page
from proceedings_to_eee.io import write_json

REVIEW_SCHEMA_VERSION = "candidate-page-review/0.1"
REVIEW_SCHEMA_NAME = "candidate_page_review"
REVIEW_GATES_SCHEMA_VERSION = "census-model-review-gates/0.1"

#: Fields the reviewer judges. Same axes the audit uses, minus producer origin, which
#: the reviewer answers with a quotation instead of a verdict.
REVIEWED_FIELDS = (
    "value",
    "metric",
    "unit",
    "system_identity",
    "system_role",
    "dataset",
    "scope",
)
FIELD_VERDICTS = ("correct", "incorrect", "unsupported", "cannot_tell")
DECISIONS = ("accept", "accept_tuple_only", "reject", "abstain")

MAX_RATIONALE_CHARS = 300

#: Determinism settings for every review call. They are passed explicitly rather than
#: left to the client's defaults because the budget ledger fingerprints the request
#: before dispatch and rejects a completion whose telemetry disagrees with it.
REVIEW_TEMPERATURE = 0.0
REVIEW_REASONING_EFFORT = "minimal"
REVIEW_SEED = 7

#: Consecutive failures tolerated before a review run gives up. A systematic error, a
#: rejected contract or a bad schema fails every call identically, and a run that keeps
#: going through one burns its budget to learn nothing.
MAX_CONSECUTIVE_FAILURES = 25

SYSTEM_PROMPT = """You review one extracted evaluation result against the full page it came from.

You are given the complete text of one page of a scientific paper and one extracted
record claiming that this page reports a particular value for a particular system,
dataset, metric and scope.

Decide one of:
- accept: the page supports the whole record, and the page also shows that this paper
  produced the number rather than quoting it from another work.
- accept_tuple_only: the page supports the record, but the page does not show who
  produced the number.
- reject: the page contradicts the record, for example the value belongs to a different
  row, system, dataset or metric than the one claimed.
- abstain: the page does not settle it either way.

Quotation rules, which are checked mechanically after you answer:
1. result_quote must be copied verbatim from the supplied page, must contain the reported
   value, and must contain the name of the evaluated system. Never invent or repair a
   quotation, and never quote text that is not on this page.
2. origin_quote is required for accept and must be a different verbatim span of this page
   that names the evaluated system and shows this paper producing the number, for example
   by describing the system as the paper's own model, method or annotation.
3. If you cannot produce a quotation that satisfies these rules, do not accept. Choose
   accept_tuple_only or abstain instead.

Judge only from the supplied page. Do not use outside knowledge about the systems,
datasets or papers involved. Prefer abstain over guessing: an honest abstention is more
useful than a confident error. Keep the rationale under 300 characters.
"""


def review_response_schema() -> dict[str, Any]:
    """Strict wire schema for one page review. Every judged field is present and typed."""

    field_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdict", "page_quote"],
        "properties": {
            "verdict": {"type": "string", "enum": list(FIELD_VERDICTS)},
            "page_quote": {"type": ["string", "null"]},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "fields", "result_quote", "origin_quote", "rationale"],
        "properties": {
            "decision": {"type": "string", "enum": list(DECISIONS)},
            "fields": {
                "type": "object",
                "additionalProperties": False,
                "required": list(REVIEWED_FIELDS),
                "properties": {name: field_schema for name in REVIEWED_FIELDS},
            },
            "result_quote": {"type": ["string", "null"]},
            "origin_quote": {"type": ["string", "null"]},
            "rationale": {"type": ["string", "null"]},
        },
    }


def review_prompt_sha256() -> str:
    """Hash of the reviewer's system prompt, recorded on every gate it writes."""

    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def review_schema_sha256() -> str:
    payload = json.dumps(review_response_schema(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReviewItem:
    """One selected candidate with the page it will be judged against."""

    paper_id: str
    observation_id: str
    page: int
    page_text: str
    raw_value: str
    evidence_quote: str
    evaluated_system: str | None
    system_role: str | None
    dataset: str | None
    metric: str | None
    unit: str | None
    scope: str | None
    candidate_payload_sha256: str


def candidate_payload_sha256(record: dict[str, Any]) -> str:
    """Bind a gate to the exact candidate tuple it was written for.

    Only the fields the reviewer was shown go in. Gate outcomes and attribution move
    between runs without changing what the page says, so including them would expire
    every gate for no reason.
    """

    scope = record.get("scope") or {}
    metric = record.get("metric") or {}
    value = record.get("value") or {}
    evidence = (record.get("evidence") or [{}])[0]
    payload = {
        "paper_id": record.get("paper_id"),
        "roles": [
            {"role": role.get("role"), "raw_name": role.get("raw_name")}
            for role in record.get("roles") or []
        ],
        "dataset_raw": scope.get("dataset_raw"),
        "raw_scope": scope.get("raw_scope"),
        "split": scope.get("split"),
        "metric_raw": metric.get("raw_name"),
        "unit": metric.get("unit") or value.get("unit"),
        "raw_value": value.get("raw"),
        "page": evidence.get("page"),
        "quote_sha256": evidence.get("quote_sha256"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _evaluated_system(record: dict[str, Any]) -> str | None:
    for role in record.get("roles") or []:
        if role.get("role") == "evaluated_system":
            return role.get("raw_name")
    return None


def _scope_summary(scope: dict[str, Any]) -> str | None:
    parts = [
        f"{label}={scope[key]}"
        for label, key in (
            ("split", "split"),
            ("subset", "subset"),
            ("group", "group"),
            ("language", "language"),
            ("aggregation", "aggregation"),
            ("raw scope", "raw_scope"),
        )
        if scope.get(key)
    ]
    return "; ".join(parts) or None


def collect_review_items(
    run_root: Path,
    *,
    paper_ids: Iterable[str] | None = None,
) -> list[ReviewItem]:
    """Select the candidates worth reviewing: supported, primary, not already exporting.

    An externally sourced candidate is excluded because no review may move it: another
    party produced that number and no page quote changes it.
    """

    wanted = set(paper_ids) if paper_ids is not None else None
    items: list[ReviewItem] = []
    for observations in sorted(run_root.glob("*/observations.jsonl")):
        paper_id = observations.parent.name
        if wanted is not None and paper_id not in wanted:
            continue
        layout_path = observations.parent / "private" / "layout.json"
        if not layout_path.is_file():
            continue
        try:
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pages = {int(page["page"]): page.get("text", "") for page in layout.get("pages", [])}
        for line in observations.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("text_support") != "supported":
                continue
            if record.get("claim_type") != "primary_result":
                continue
            if (record.get("attribution") or {}).get("state") == "externally_sourced":
                continue
            if record.get("export_status") == "eligible":
                continue
            evidence = (record.get("evidence") or [{}])[0]
            page = evidence.get("page")
            if page is None or int(page) not in pages:
                continue
            value = record.get("value") or {}
            if not value.get("raw"):
                continue
            scope = record.get("scope") or {}
            metric = record.get("metric") or {}
            items.append(
                ReviewItem(
                    paper_id=paper_id,
                    observation_id=str(record.get("observation_id")),
                    page=int(page),
                    page_text=pages[int(page)],
                    raw_value=str(value.get("raw")),
                    evidence_quote=str(evidence.get("quote") or ""),
                    evaluated_system=_evaluated_system(record),
                    system_role="evaluated_system",
                    dataset=scope.get("dataset_raw"),
                    metric=metric.get("raw_name"),
                    unit=metric.get("unit") or value.get("unit"),
                    scope=_scope_summary(scope),
                    candidate_payload_sha256=candidate_payload_sha256(record),
                )
            )
    return items


def review_prompt(item: ReviewItem) -> str:
    """Render the user message. Only the page and the tuple the reviewer must judge."""

    lines = [
        "PAGE TEXT",
        "---",
        item.page_text,
        "---",
        "",
        "EXTRACTED RECORD",
        f"evaluated system: {item.evaluated_system or '(none stated)'}",
        f"system role: {item.system_role or '(none stated)'}",
        f"dataset: {item.dataset or '(none stated)'}",
        f"metric: {item.metric or '(none stated)'}",
        f"unit: {item.unit or '(none stated)'}",
        f"reported value: {item.raw_value}",
        f"scope: {item.scope or '(none stated)'}",
        f"evidence quote recorded by the extractor: {item.evidence_quote}",
    ]
    return "\n".join(lines)


def _numeric_token_present(quote: str, raw_value: str) -> bool:
    """The value must appear as a whole number, not inside a longer one."""

    token = re.sub(r"^[<>=~≈≤≥\s]+", "", raw_value.strip())
    token = re.sub(r"[%*†‡§\s]+$", "", token)
    if not token:
        return False
    pattern = re.compile(rf"(?<![\d.,]){re.escape(token)}(?![\d,]|\.\d)")
    return pattern.search(_normalize(quote)) is not None


def _literal_present(quote: str, literal: str | None) -> bool:
    if not literal:
        return False
    return _normalize(literal).casefold() in _normalize(quote).casefold()


@dataclass
class DefenceOutcome:
    """What the local defence made of one reviewer response."""

    decision: str
    result_quote_on_page: bool
    result_quote_has_value: bool
    result_quote_has_system: bool
    origin_quote_on_page: bool
    origin_quote_has_system: bool
    origin_quote_distinct: bool
    demotions: list[str] = field(default_factory=list)


def apply_local_defence(item: ReviewItem, payload: dict[str, Any]) -> DefenceOutcome:
    """Check every quotation against the page before the decision is allowed to stand.

    Demote-only. An accept can fall to accept_tuple_only or to abstain, and anything
    positive can fall to abstain, but nothing is ever promoted and a reject is never
    softened.
    """

    decision = str(payload.get("decision") or "abstain")
    if decision not in DECISIONS:
        decision = "abstain"
    result_quote = payload.get("result_quote")
    origin_quote = payload.get("origin_quote")

    result_on_page = quotation_is_on_page(result_quote, item.page_text)
    result_has_value = bool(result_quote) and _numeric_token_present(result_quote, item.raw_value)
    result_has_system = bool(result_quote) and _literal_present(result_quote, item.evaluated_system)
    origin_on_page = quotation_is_on_page(origin_quote, item.page_text)
    origin_has_system = bool(origin_quote) and _literal_present(origin_quote, item.evaluated_system)
    origin_distinct = (
        bool(origin_quote)
        and bool(result_quote)
        and (_normalize(str(origin_quote)) != _normalize(str(result_quote)))
    )

    outcome = DefenceOutcome(
        decision=decision,
        result_quote_on_page=result_on_page,
        result_quote_has_value=result_has_value,
        result_quote_has_system=result_has_system,
        origin_quote_on_page=origin_on_page,
        origin_quote_has_system=origin_has_system,
        origin_quote_distinct=origin_distinct,
    )
    if decision in {"reject", "abstain"}:
        return outcome

    if decision == "accept" and not (origin_on_page and origin_has_system and origin_distinct):
        outcome.decision = "accept_tuple_only"
        outcome.demotions.append("origin_quote_failed_local_defence")
    if not (result_on_page and result_has_value and result_has_system):
        outcome.decision = "abstain"
        outcome.demotions.append("result_quote_failed_local_defence")
    return outcome


def _quote_sha256(quote: Any) -> str | None:
    if not isinstance(quote, str) or not quote.strip():
        return None
    return hashlib.sha256(quote.encode("utf-8")).hexdigest()


def load_gates(paper_dir: Path) -> dict[str, Any]:
    path = paper_dir / "model-review-gates.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.get("gates")
    return gates if isinstance(gates, dict) else {}


def _write_gates(paper_dir: Path, gates: dict[str, Any], *, model: str) -> None:
    write_json(
        paper_dir / "model-review-gates.json",
        {
            "schema_version": REVIEW_GATES_SCHEMA_VERSION,
            "model": model,
            "review_prompt_sha256": review_prompt_sha256(),
            "review_schema_sha256": review_schema_sha256(),
            "gates": gates,
        },
    )


def _append_private_record(paper_dir: Path, record: dict[str, Any]) -> None:
    path = paper_dir / "private" / "model-review.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_review(
    *,
    run_root: Path,
    model: str,
    call: Callable[[str, str], tuple[dict[str, Any], dict[str, Any]]],
    paper_ids: Iterable[str] | None = None,
    workers: int = 1,
    items: Sequence[ReviewItem] | None = None,
    stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Review every selected candidate that has no gate yet, and write both artifacts.

    `call` returns the parsed payload and a secret-free telemetry mapping. `stop` is
    consulted before each call so a budget ceiling can end the run without a partial
    write; every completed review is already on disk when it does.
    """

    selected = (
        list(items) if items is not None else collect_review_items(run_root, paper_ids=paper_ids)
    )
    by_paper: dict[str, list[ReviewItem]] = {}
    for item in selected:
        by_paper.setdefault(item.paper_id, []).append(item)

    counts: dict[str, int] = {decision: 0 for decision in DECISIONS}
    demotions: dict[str, int] = {}
    failure_classes: dict[str, int] = {}
    reviewed = skipped = failed = 0
    stopped_for_budget = False
    consecutive_failures = 0
    stopped_for_failures = False
    lock = threading.Lock()

    def record_failure(error: BaseException) -> None:
        nonlocal failed, consecutive_failures
        with lock:
            failed += 1
            consecutive_failures += 1
            name = type(error).__name__
            failure_classes[name] = failure_classes.get(name, 0) + 1

    for paper_id, paper_items in sorted(by_paper.items()):
        paper_dir = run_root / paper_id
        gates = load_gates(paper_dir)
        pending = [item for item in paper_items if item.observation_id not in gates]
        skipped += len(paper_items) - len(pending)
        if not pending:
            continue

        def review_one(
            item: ReviewItem,
        ) -> tuple[ReviewItem, DefenceOutcome, dict[str, Any]] | None:
            if stop is not None and stop():
                return None
            payload, telemetry = call(SYSTEM_PROMPT, review_prompt(item))
            return item, apply_local_defence(item, payload), {**telemetry, "payload": payload}

        results: list[tuple[ReviewItem, DefenceOutcome, dict[str, Any]]] = []
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(review_one, item) for item in pending]
                for future in futures:
                    try:
                        outcome = future.result()
                    except Exception as error:  # noqa: BLE001 - one review must not end the run
                        record_failure(error)
                        continue
                    if outcome is None:
                        stopped_for_budget = True
                        continue
                    consecutive_failures = 0
                    results.append(outcome)
        else:
            for item in pending:
                try:
                    outcome = review_one(item)
                except Exception as error:  # noqa: BLE001 - one review must not end the run
                    record_failure(error)
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        stopped_for_failures = True
                        break
                    continue
                if outcome is None:
                    stopped_for_budget = True
                    break
                consecutive_failures = 0
                results.append(outcome)

        for item, outcome, telemetry in results:
            payload = telemetry.pop("payload")
            rationale = payload.get("rationale")
            if isinstance(rationale, str):
                rationale = rationale[:MAX_RATIONALE_CHARS]
            _append_private_record(
                paper_dir,
                {
                    "schema_version": REVIEW_SCHEMA_VERSION,
                    "observation_id": item.observation_id,
                    "model": model,
                    "proposed_decision": payload.get("decision"),
                    "decision": outcome.decision,
                    "demotions": outcome.demotions,
                    "result_quote": payload.get("result_quote"),
                    "origin_quote": payload.get("origin_quote"),
                    "fields": payload.get("fields"),
                    "rationale": rationale,
                    "candidate_payload_sha256": item.candidate_payload_sha256,
                    **telemetry,
                },
            )
            gates[item.observation_id] = {
                "decision": outcome.decision,
                "proposed_decision": payload.get("decision"),
                "demotions": outcome.demotions,
                "candidate_payload_sha256": item.candidate_payload_sha256,
                "review_prompt_sha256": review_prompt_sha256(),
                "review_schema_sha256": review_schema_sha256(),
                "result_quote_sha256": _quote_sha256(payload.get("result_quote")),
                "origin_quote_sha256": _quote_sha256(payload.get("origin_quote")),
                "model": model,
                "page": item.page,
            }
            counts[outcome.decision] += 1
            for demotion in outcome.demotions:
                demotions[demotion] = demotions.get(demotion, 0) + 1
            reviewed += 1
        _write_gates(paper_dir, gates, model=model)
        if stopped_for_budget or stopped_for_failures:
            break
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            stopped_for_failures = True
            break

    return {
        "schema_version": REVIEW_GATES_SCHEMA_VERSION,
        "model": model,
        "selected": len(selected),
        "reviewed": reviewed,
        "already_gated": skipped,
        "failed": failed,
        "failure_classes": failure_classes,
        "stopped_for_budget": stopped_for_budget,
        "stopped_for_repeated_failures": stopped_for_failures,
        "decisions": counts,
        "quotation_defence_demotions": demotions,
        "claim_boundary": [
            "These are model judgements checked against the page, not human labels.",
            "Human labels scored with ere score-review-labels measure candidate tuple "
            "correctness conditional on the sampled model decision, not exported EEE accuracy.",
        ],
    }


LABEL_PACKET_SCHEMA_VERSION = "census-review-label-packet/0.1"
LABEL_SCORE_SCHEMA_VERSION = "census-review-label-score/0.1"
BLINDED_LABEL_PACKET_SCHEMA_VERSION = "census-review-blinded-label-packet/0.1"

#: The only values a human may write into a packet's blank label field.
HUMAN_LABELS = ("correct", "incorrect", "cannot_tell")


def _packet_key(seed: str, observation_id: str) -> str:
    return hashlib.sha256(f"{seed}|packet|{observation_id}".encode()).hexdigest()


def build_label_packet(
    *,
    run_root: Path,
    size: int,
    negatives: int,
    seed: str,
    paper_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Assemble what a person needs to judge the reviewer, with the label left blank.

    Positives and negatives are drawn separately and deliberately: a packet of accepts
    only would measure precision and call it accuracy. The agent never fills `label`.
    """

    items: dict[str, ReviewItem] = {}
    for item in collect_review_items(run_root, paper_ids=paper_ids):
        if item.observation_id in items:
            raise ValueError("ambiguous observation_id in review items; cannot build a packet")
        items[item.observation_id] = item
    accepted: list[tuple[str, ReviewItem, dict[str, Any]]] = []
    rejected: list[tuple[str, ReviewItem, dict[str, Any]]] = []
    for observations in sorted(run_root.glob("*/observations.jsonl")):
        paper_dir = observations.parent
        for observation_id, gate in sorted(load_gates(paper_dir).items()):
            item = items.get(observation_id)
            if item is None or item.paper_id != paper_dir.name:
                continue
            accepted_decision = gate.get("decision") in {"accept", "accept_tuple_only"}
            bucket = accepted if accepted_decision else rejected
            bucket.append((observation_id, item, gate))

    def take(pool: list[tuple[str, ReviewItem, dict[str, Any]]], count: int) -> list[Any]:
        ordered = sorted(pool, key=lambda entry: _packet_key(seed, entry[0]))
        return ordered[:count]

    chosen = take(accepted, size) + take(rejected, negatives)
    entries = [
        {
            "observation_id": observation_id,
            "paper_id": item.paper_id,
            "page": item.page,
            "page_text": item.page_text,
            "candidate": {
                "evaluated_system": item.evaluated_system,
                "dataset": item.dataset,
                "metric": item.metric,
                "unit": item.unit,
                "reported_value": item.raw_value,
                "scope": item.scope,
                "extractor_evidence_quote": item.evidence_quote,
            },
            "model_decision": gate.get("decision"),
            "model_proposed_decision": gate.get("proposed_decision"),
            "candidate_payload_sha256": gate.get("candidate_payload_sha256"),
            "label": None,
            "label_note": None,
        }
        for observation_id, item, gate in chosen
    ]
    packet = {
        "schema_version": LABEL_PACKET_SCHEMA_VERSION,
        "seed": seed,
        "requested": {"size": size, "negatives": negatives},
        "available": {"accepted": len(accepted), "not_accepted": len(rejected)},
        "entries": entries,
        "instructions": [
            "For each entry, read the page text and judge whether the extracted record is "
            "correct as stated. Write 'correct', 'incorrect' or 'cannot_tell' into label.",
            "Judge the record against the page, not the model's decision.",
            "Scoring human labels measures candidate tuple correctness conditional on "
            "the sampled model decision, not exported EEE accuracy or producer origin.",
        ],
    }
    _validated_packet_entries(packet)
    return packet


def _validated_packet_entries(packet: dict[str, Any]) -> list[dict[str, Any]]:
    entries = packet.get("entries") if isinstance(packet, dict) else None
    if not isinstance(entries, list):
        raise ValueError("packet has no entries list")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("packet entries must be objects")
        observation_id = entry.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ValueError("packet observation_id must be a nonempty string")
        if observation_id in seen:
            raise ValueError("duplicate or ambiguous observation_id in packet")
        seen.add(observation_id)
        if entry.get("model_decision") not in DECISIONS:
            raise ValueError("packet has an invalid or missing model_decision")
    return entries


def _label_packet_sha256(packet: dict[str, Any]) -> str:
    payload = json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_blinded_label_packet(packet: dict[str, Any], *, seed: str) -> dict[str, Any]:
    """Copy blank items in shuffled order without disclosing the model's decisions.

    Keep the original packet unchanged for scoring. This view blinds decision strata,
    not the extracted tuple or source identity, and does not supply any human labels.
    """

    entries = _validated_packet_entries(packet)
    if any(entry.get("label") is not None for entry in entries):
        raise ValueError("prepare the blinded view from a blank original packet")
    fields = ("observation_id", "paper_id", "page", "page_text", "candidate")
    ordered = sorted(entries, key=lambda entry: _packet_key(seed, entry["observation_id"]))
    # A JSON copy prevents later edits to the view from mutating the original packet.
    blinded = [
        {
            **json.loads(json.dumps({key: entry[key] for key in fields if key in entry})),
            "label": None,
            "label_note": None,
        }
        for entry in ordered
    ]
    return {
        "schema_version": BLINDED_LABEL_PACKET_SCHEMA_VERSION,
        "source_packet_sha256": _label_packet_sha256(packet),
        "shuffle_seed": seed,
        "entries": blinded,
        "instructions": [
            "Read the supplied page and judge the candidate tuple as stated. The label "
            "is about the candidate, not whether a model decision was reasonable.",
            "Write correct, incorrect, or cannot_tell into label, and explain the "
            "relevant field or unresolved evidence in label_note. Do not edit the candidate.",
            "The model's decisions are withheld. Retain the original packet for scoring.",
        ],
    }


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""

    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3.0e-15:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), computed directly so the interval needs no third-party dependency."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return (
        1.0
        - math.exp(b * math.log(1.0 - x) + a * math.log(x) - _log_beta(b, a))
        * _betacf(b, a, 1.0 - x)
        / b
    )


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval, by inverting the beta CDF with bisection.

    Exact rather than normal-approximate because the counts here are small and near the
    ends, where a Wald interval can leave the unit interval altogether.
    """

    if trials <= 0:
        raise ValueError("an interval needs at least one trial")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie within the trials")
    alpha = 1.0 - confidence

    def invert(target: float, a: float, b: float) -> float:
        low, high = 0.0, 1.0
        for _ in range(200):
            mid = (low + high) / 2.0
            if regularized_incomplete_beta(a, b, mid) < target:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    lower = 0.0 if successes == 0 else invert(alpha / 2.0, successes, trials - successes + 1)
    upper = (
        1.0 if successes == trials else invert(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return lower, upper


def score_review_labels(
    *, packet: dict[str, Any], labels: dict[str, Any], confidence: float = 0.95
) -> dict[str, Any]:
    """Validate labels before scoring; unfinished work is explicitly reported as partial."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    packet_entries = _validated_packet_entries(packet)
    packet_by_id = {entry["observation_id"]: entry for entry in packet_entries}
    filled = labels.get("entries") if isinstance(labels, dict) else None
    if not isinstance(filled, list):
        raise ValueError("label file has no entries list")
    source_packet_sha256 = labels.get("source_packet_sha256")
    if (
        labels.get("schema_version") == BLINDED_LABEL_PACKET_SCHEMA_VERSION
        and source_packet_sha256 is None
    ):
        raise ValueError("blinded labels require a source_packet_sha256")
    if source_packet_sha256 is not None and source_packet_sha256 != _label_packet_sha256(packet):
        raise ValueError("label file is bound to a different source packet")
    by_id: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for entry in filled:
        if not isinstance(entry, dict):
            raise ValueError("label entries must be objects")
        observation_id = entry.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ValueError("label observation_id must be a nonempty string")
        if observation_id in seen:
            raise ValueError("duplicate observation_id in labels")
        seen.add(observation_id)
        if observation_id not in packet_by_id:
            raise ValueError("unknown observation_id in labels")
        original = packet_by_id[observation_id]
        for field_name in ("paper_id", "page", "page_text", "candidate"):
            if field_name in entry and entry[field_name] != original.get(field_name):
                raise ValueError("label item evidence or candidate differs from the source packet")
        label = entry.get("label")
        if label is not None and label not in HUMAN_LABELS:
            raise ValueError("invalid label; use correct, incorrect, cannot_tell, or null")
        if label is not None:
            by_id[observation_id] = entry
    accepted_correct = accepted_total = 0
    not_accepted_correct = not_accepted_total = 0
    undecided = 0
    for entry in packet_entries:
        labelled = by_id.get(entry["observation_id"])
        if labelled is None:
            continue
        label = labelled["label"]
        if label == "cannot_tell":
            undecided += 1
            continue
        accepted = entry.get("model_decision") in {"accept", "accept_tuple_only"}
        if accepted:
            accepted_total += 1
            accepted_correct += label == "correct"
        else:
            not_accepted_total += 1
            not_accepted_correct += label == "incorrect"

    def interval(successes: int, trials: int) -> dict[str, Any]:
        if trials == 0:
            return {"k": successes, "n": trials, "rate": None, "lower": None, "upper": None}
        lower, upper = clopper_pearson(successes, trials, confidence)
        return {
            "k": successes,
            "n": trials,
            "rate": successes / trials,
            "lower": lower,
            "upper": upper,
        }

    return {
        "schema_version": LABEL_SCORE_SCHEMA_VERSION,
        "confidence": confidence,
        "packet_seed": packet.get("seed"),
        "completion_status": "complete" if len(by_id) == len(packet_entries) else "partial",
        "packet_entries": len(packet_entries),
        "labelled": len(by_id),
        "unlabelled": len(packet_entries) - len(by_id),
        "source_packet_binding_verified": source_packet_sha256 is not None,
        "undecidable_by_the_labeller": undecided,
        "model_accepted_and_human_says_correct": interval(accepted_correct, accepted_total),
        "model_did_not_accept_and_human_says_incorrect": interval(
            not_accepted_correct, not_accepted_total
        ),
        "claim_boundary": [
            "This is a single-labeller measurement unless the label file says otherwise. "
            "It is not an agreement estimate and not independent validation.",
            "Intervals are exact Clopper-Pearson bounds at the stated confidence.",
            "Rates condition on the sampled model-decision stratum and decidable labels. "
            "They are not whole-corpus accuracy, recall, producer-origin accuracy, or "
            "the precision of the final exported EEE records.",
            "The nonaccepted stratum includes both reject and abstain. Its reported "
            "rate is the fraction of those candidates humans judged incorrect, not specificity.",
            "A partial result describes completed labels only; report unlabelled and "
            "cannot_tell counts alongside every rate. Paper clustering is not modeled.",
        ],
    }
