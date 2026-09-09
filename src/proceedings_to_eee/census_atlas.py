"""Build the Toxicity Evaluation Evidence Atlas from a completed census run.

The atlas is the product; the pipeline is the machinery behind it. It answers questions
about how a literature measures things: which systems are evaluated on which datasets,
with which metrics, in which languages, and what is systematically missing.

Three deliberate choices shape it.

Every candidate field is flattened rather than a chosen subset, because the fields that
carry the study's axes are exactly the ones a summary view tends to drop: construct,
operationalization, decision rule, metric parameters, notes, and the per-field binding
statuses.

Missingness is representable. `rows.csv` records every planned dense table row and how it
ended, so a reader can tell a result that was never reported from one the pipeline failed
to extract. A summary of present rows alone cannot express that difference.

Census classification metadata is joined under `census_` names and never treated as
ground truth. It is unvalidated screening metadata of unknown producer, and it describes
a paper rather than an individual result, so a per-observation match is computed
deterministically and reported as a retrieval hint rather than a label.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from proceedings_to_eee.domain.observation import MetricSpec
from proceedings_to_eee.io import write_json
from proceedings_to_eee.resolution.metrics import metric_canonical_source

ATLAS_SCHEMA_VERSION = "census-atlas/0.1"

CLAIM_BOUNDARY = [
    "This atlas is an unreviewed candidate layer. Almost every row is a proposal that "
    "passed deterministic checks and awaits review; it is not a verified result.",
    "Canonical EEE is empty by design. The origin resolver cannot positively establish "
    "that a paper produced a number, so nothing passes the canonical export gate. "
    "Records composed under the tiered policy are not canonical: each states the review "
    "tier and producer-origin basis it was granted on, and none claims paper_produced.",
    "Coverage is limited to the result-bearing pages the selector chose, at most 8 per "
    "paper. An absent value is not evidence that a paper reported nothing.",
    "The model and route were chosen on ten inspected ACM papers. That does not "
    "validate performance on this ACL-layout population.",
    "Papers were processed under more than one code state; per-paper code hashes are "
    "listed in summary.json.",
    "census_ columns are unvalidated screening metadata of unknown producer. They "
    "describe a paper, not an individual result.",
    "Export status and reason are what each run recorded, not what today's code would "
    "decide. Metric identity is recomputed at read time, so the two can disagree: a name "
    "the registry has since learned still shows the gate outcome its run wrote. "
    "census-eee-summary.json from `ere recompose-census` is the authority for what the "
    "current gates decide.",
]

_OBSERVATION_COLUMNS = [
    "paper_id",
    "census_paper_id",
    "observation_id",
    "claim_type",
    "reporting_status",
    "extraction_method",
    "extraction_stage",
    "extraction_confidence",
    "evaluated_system",
    "evaluated_system_version",
    "evaluated_system_confidence",
    "other_roles",
    "role_count",
    "dataset_raw",
    "dataset_id",
    "dataset_key",
    "split",
    "subset",
    "group",
    "language_raw",
    "language_iso639",
    "sample_count",
    "aggregation",
    "raw_scope",
    "metric_raw",
    "metric_canonical",
    "metric_canonical_source",
    "review_decision",
    "review_tier",
    "producer_origin_basis",
    "eee_record_id",
    "metric_registry_resolved",
    "metric_kind",
    "metric_unit",
    "metric_lower_is_better",
    "metric_parameters",
    "value_raw",
    "value_numeric",
    "value_unit",
    "value_comparator",
    "construct",
    "operationalization",
    "decision_rule",
    "evaluation_date",
    "evidence_page",
    "evidence_kind",
    "evidence_label",
    "evidence_row",
    "evidence_column",
    "evidence_quote_sha256",
    "evidence_anchor_count",
    "text_support",
    "referential_status",
    "attribution_state",
    "attribution_rule",
    "export_status",
    "export_reason",
    "census_instrument_match",
    "census_match_field",
    "census_match_kind",
    "notes",
]

#: A small, checked-in table. Deliberately not a language-detection model: this maps
#: what extraction actually writes into a stable code, and abstains otherwise.
_LANGUAGE_CODES = {
    "english": "en",
    "en": "en",
    "german": "de",
    "de": "de",
    "deutsch": "de",
    "french": "fr",
    "fr": "fr",
    "spanish": "es",
    "es": "es",
    "italian": "it",
    "it": "it",
    "portuguese": "pt",
    "pt": "pt",
    "dutch": "nl",
    "nl": "nl",
    "polish": "pl",
    "pl": "pl",
    "russian": "ru",
    "ru": "ru",
    "arabic": "ar",
    "ar": "ar",
    "hindi": "hi",
    "hi": "hi",
    "chinese": "zh",
    "zh": "zh",
    "mandarin": "zh",
    "japanese": "ja",
    "ja": "ja",
    "korean": "ko",
    "ko": "ko",
    "turkish": "tr",
    "tr": "tr",
    "greek": "el",
    "el": "el",
    "hebrew": "he",
    "he": "he",
    "danish": "da",
    "da": "da",
    "swedish": "sv",
    "sv": "sv",
    "norwegian": "no",
    "no": "no",
    "finnish": "fi",
    "fi": "fi",
    "czech": "cs",
    "cs": "cs",
    "romanian": "ro",
    "ro": "ro",
    "hungarian": "hu",
    "hu": "hu",
    "bengali": "bn",
    "bn": "bn",
    "tamil": "ta",
    "ta": "ta",
    "telugu": "te",
    "te": "te",
    "urdu": "ur",
    "ur": "ur",
    "indonesian": "id",
    "id": "id",
    "vietnamese": "vi",
    "vi": "vi",
    "thai": "th",
    "th": "th",
    "persian": "fa",
    "fa": "fa",
    "malayalam": "ml",
    "ml": "ml",
    "kannada": "kn",
    "kn": "kn",
    "marathi": "mr",
    "mr": "mr",
    "nepali": "ne",
    "ne": "ne",
    "amharic": "am",
    "am": "am",
    "swahili": "sw",
    "sw": "sw",
    "ukrainian": "uk",
    "uk": "uk",
    "bulgarian": "bg",
    "bg": "bg",
    "croatian": "hr",
    "hr": "hr",
    "slovenian": "sl",
    "sl": "sl",
    "catalan": "ca",
    "ca": "ca",
    "basque": "eu",
    "eu": "eu",
    "galician": "gl",
    "gl": "gl",
    "estonian": "et",
    "et": "et",
    "latvian": "lv",
    "lv": "lv",
    "lithuanian": "lt",
    "lt": "lt",
    "slovak": "sk",
    "sk": "sk",
    "serbian": "sr",
    "sr": "sr",
    "albanian": "sq",
    "sq": "sq",
    "multilingual": "mul",
    "mul": "mul",
}


def language_code(raw: str | None) -> str | None:
    """Map a written language name onto a code, or abstain."""

    if not raw:
        return None
    return _LANGUAGE_CODES.get(re.sub(r"[^a-z]+", "", raw.casefold()))


def dataset_key(raw: str | None) -> str | None:
    """Collapse a dataset name so the same benchmark groups across papers.

    Only whitespace, case, punctuation and a trailing parenthetical are normalised. No
    alias guessing: `KLEJ benchmark (CBD)` and `KLEJ benchmark` become one key, while two
    genuinely different names stay apart.
    """

    if not raw or not raw.strip():
        return None
    text = re.sub(r"\s*\([^)]*\)\s*$", "", raw.strip())
    text = re.sub(r"\b(?:benchmark|dataset|corpus|data ?set)\b", " ", text, flags=re.I)
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    return text or None


def _tokens(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def census_instrument_match(
    instruments: str | None, haystacks: dict[str, str | None]
) -> tuple[str | None, str | None, str]:
    """Match a census instrument mention against what extraction actually found.

    A hit means the paper's screening note names something the extracted record also
    names. A miss is not evidence of irrelevance: the screening note may name an
    instrument used on a page extraction never saw.
    """

    candidates = _tokens(instruments)
    if not candidates:
        return None, None, "none"
    normalised = {
        field: re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).strip()
        for field, text in haystacks.items()
    }
    for token in candidates:
        needle = re.sub(r"[^a-z0-9]+", " ", token.casefold()).strip()
        if not needle:
            continue
        for field, text in normalised.items():
            if text and needle == text:
                return token, field, "exact"
    for token in candidates:
        needle = re.sub(r"[^a-z0-9]+", " ", token.casefold()).strip()
        if len(needle) < 3:
            continue
        for field, text in normalised.items():
            if text and needle in text:
                return token, field, "substring"
    return None, None, "none"


def paper_ledger_cost(paper_dir: Path) -> tuple[float, int]:
    """Read one paper's real spend from its own budget ledger.

    Failed and ceiling-stopped papers may lack a successful run summary but still incur
    provider costs. Read the per-paper ledgers directly so the atlas counts those costs
    regardless of which driver version wrote the census summary.
    """

    private = paper_dir / "private"
    # Superseded ledgers count too. A paper whose run contract moved gets a fresh ledger
    # beside the old one; the money the old one records was still spent.
    paths = [
        private / "provider-budget-ledger.jsonl",
        *sorted(private.glob("provider-budget-ledger.superseded-*.jsonl")),
    ]
    cost = 0.0
    calls = 0
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_type") != "completion":
                continue
            calls += 1
            cost += (event.get("provider_call") or {}).get("cost_usd") or 0.0
    return cost, calls


def _last_ledger_records(run_root: Path) -> dict[str, dict[str, Any]]:
    """Take the last record per paper.

    `census-ledger.jsonl` is append-only across runs, so repeated attempts can write
    multiple records for the same paper.
    """

    path = run_root / "census-ledger.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        latest[record["paper_id"]] = record
    return latest


def _review_index(paper_dir: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Read a paper's review decisions and composed-record index, both by stored id.

    The recompose writes the index keyed by the observation id the run wrote, because
    validation re-stamps ids from the resolved tuple and the atlas reads what the run
    wrote. Missing files simply mean the paper was never reviewed or composed.
    """

    decisions: dict[str, str] = {}
    gates_path = paper_dir / "model-review-gates.json"
    if gates_path.is_file():
        payload = json.loads(gates_path.read_text(encoding="utf-8"))
        decisions = {
            observation_id: str(gate.get("decision"))
            for observation_id, gate in (payload.get("gates") or {}).items()
        }
    records: dict[str, dict[str, str]] = {}
    index_path = paper_dir / "eee-tiered" / "record-index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        records = payload.get("by_observation_id") or {}
    return decisions, records


def _observation_row(
    record: dict[str, Any],
    *,
    paper_id: str,
    census: dict[str, Any],
    review_decisions: dict[str, str] | None = None,
    composed_records: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    composed = (composed_records or {}).get(str(record.get("observation_id")), {})
    roles = record.get("roles") or []
    evaluated = next((role for role in roles if role.get("role") == "evaluated_system"), None)
    others = [
        f"{role.get('role')}:{role.get('raw_name')}" for role in roles if role is not evaluated
    ]
    scope = record.get("scope") or {}
    metric = record.get("metric") or {}
    value = record.get("value") or {}
    evidence = record.get("evidence") or []
    anchor = evidence[0] if evidence else {}
    attribution = record.get("attribution") or {}
    # A candidate can carry no metric at all, and MetricSpec requires a non-empty name.
    raw_metric_name = metric.get("raw_name")
    if raw_metric_name:
        spec = MetricSpec(raw_name=raw_metric_name, canonical_id=metric.get("canonical_id"))
        canonical_source = metric_canonical_source(spec)
        # A concordant hit is registry-backed too: a short alias plus the extractor's
        # own canonical id naming the same metric. The source column keeps the two
        # apart for anyone who wants plain hits only.
        registry_resolved = canonical_source in {"registry", "registry_concordant"}
    else:
        canonical_source = "none"
        registry_resolved = False
    method = record.get("extraction_method") or ""
    instrument, match_field, match_kind = census_instrument_match(
        census.get("census_instrument_mentions"),
        {
            "system": (evaluated or {}).get("raw_name"),
            "dataset": scope.get("dataset_raw"),
            "metric": metric.get("raw_name"),
            "notes": " ".join(record.get("notes") or []),
        },
    )
    return {
        "paper_id": paper_id,
        "census_paper_id": census.get("census_paper_id"),
        "observation_id": record.get("observation_id"),
        "claim_type": record.get("claim_type"),
        "reporting_status": record.get("reporting_status"),
        "extraction_method": method,
        "extraction_stage": "row" if ":row-enumeration" in method else "block",
        "extraction_confidence": record.get("extraction_confidence"),
        "evaluated_system": (evaluated or {}).get("raw_name"),
        "evaluated_system_version": (evaluated or {}).get("version"),
        "evaluated_system_confidence": (evaluated or {}).get("confidence"),
        "other_roles": " | ".join(others),
        "role_count": len(roles),
        "dataset_raw": scope.get("dataset_raw"),
        "dataset_id": scope.get("dataset_id"),
        "dataset_key": dataset_key(scope.get("dataset_raw")),
        "split": scope.get("split"),
        "subset": scope.get("subset"),
        "group": scope.get("group"),
        "language_raw": scope.get("language"),
        "language_iso639": language_code(scope.get("language")),
        "sample_count": scope.get("sample_count"),
        "aggregation": scope.get("aggregation"),
        "raw_scope": scope.get("raw_scope"),
        "metric_raw": metric.get("raw_name"),
        "metric_canonical": metric.get("canonical_id"),
        "metric_canonical_source": canonical_source,
        "review_decision": (review_decisions or {}).get(str(record.get("observation_id"))),
        "review_tier": composed.get("review_tier"),
        "producer_origin_basis": composed.get("producer_origin_basis"),
        "eee_record_id": composed.get("eee_record_id"),
        "metric_registry_resolved": registry_resolved,
        "metric_kind": metric.get("kind"),
        "metric_unit": metric.get("unit"),
        "metric_lower_is_better": metric.get("lower_is_better"),
        "metric_parameters": json.dumps(metric.get("parameters") or {}, sort_keys=True),
        "value_raw": value.get("raw"),
        "value_numeric": value.get("numeric"),
        "value_unit": value.get("unit"),
        "value_comparator": value.get("comparator"),
        "construct": record.get("evaluation_construct") or record.get("construct"),
        "operationalization": record.get("operationalization"),
        "decision_rule": record.get("decision_rule"),
        "evaluation_date": record.get("evaluation_date"),
        "evidence_page": anchor.get("page"),
        "evidence_kind": anchor.get("kind"),
        "evidence_label": anchor.get("label"),
        "evidence_row": anchor.get("row"),
        "evidence_column": anchor.get("column"),
        "evidence_quote_sha256": anchor.get("quote_sha256"),
        "evidence_anchor_count": len(evidence),
        "text_support": record.get("text_support"),
        "referential_status": record.get("referential_status"),
        "attribution_state": attribution.get("state"),
        "attribution_rule": attribution.get("rule_id") or attribution.get("rule"),
        "export_status": record.get("export_status"),
        "export_reason": record.get("export_reason"),
        "census_instrument_match": instrument,
        "census_match_field": match_field,
        "census_match_kind": match_kind,
        "notes": " | ".join(record.get("notes") or []),
    }


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


_RESERVED_DIRECTORY_NAMES = frozenset({"atlas", "audit", "eee", "private"})
"""Run-root children that hold run-level artifacts rather than one paper."""

_LEDGER_STATE_TO_STATUS = {
    "failed": "failed",
    "budget_stopped": "budget_stopped",
    "source_unavailable": "unavailable",
    "not_started": "not_started",
    "retry_capped": "retry_capped",
}


def _excluded_paper_ids(run_root: Path) -> set[str]:
    path = run_root / "census-exclusions.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(entry.get("paper_id")) for entry in payload.get("excluded", []) if entry.get("paper_id")
    }


def _paper_universe(
    run_root: Path,
    metadata: dict[str, Any],
    ledger: dict[str, dict[str, Any]],
) -> list[str]:
    """Every paper the atlas must account for, whether or not it produced anything.

    Keying the atlas only on `run.json` would hide papers that failed, hit a ceiling,
    were never started, had no frozen source, or were held back. The universe is
    the census metadata, the driver ledger, and any directory that was frozen or ran. The
    sealed reserve is never opened: its papers appear here only as census rows with no
    directory and no ledger entry, and are labelled `reserve` on that basis alone.
    """

    universe = set(metadata) | set(ledger)
    for child in run_root.iterdir():
        if not child.is_dir() or child.name in _RESERVED_DIRECTORY_NAMES:
            continue
        if (child / "source-manifest.json").is_file() or (child / "run.json").is_file():
            universe.add(child.name)
    return sorted(universe)


def _paper_status(
    *,
    paper_id: str,
    run: dict[str, Any] | None,
    ledger_record: dict[str, Any] | None,
    paper_dir: Path,
    excluded_ids: set[str],
) -> tuple[str, str]:
    """Return one paper's status and where that status was read from."""

    if run is not None:
        return str(run.get("status") or "success"), "run.json"
    if paper_id in excluded_ids:
        return "excluded", "census-exclusions.json"
    state = (ledger_record or {}).get("state")
    if state in _LEDGER_STATE_TO_STATUS:
        return _LEDGER_STATE_TO_STATUS[state], "census-ledger.jsonl"
    if state in {"done", "skipped"}:
        # The driver reported the paper finished but no run.json survives; say so rather
        # than guessing which failure it was.
        return "missing_run_artifact", "census-ledger.jsonl"
    if (paper_dir / "source-manifest.json").is_file():
        return "not_started", "frozen source, no run"
    if paper_dir.is_dir():
        return "unavailable", "directory without a frozen source"
    return "reserve", "census row absent from the run"


def run_root_ledger_cost(run_root: Path) -> tuple[float, int, int]:
    """Sum every per-paper budget ledger under the run root.

    Returns cost, structured calls, and the number of ledgers read.
    """

    cost = 0.0
    calls = 0
    ledgers = 0
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "private" / "provider-budget-ledger.jsonl").is_file():
            continue
        ledgers += 1
        paper_cost, paper_calls = paper_ledger_cost(child)
        cost += paper_cost
        calls += paper_calls
    return cost, calls, ledgers


def build_atlas(
    *,
    run_root: Path,
    output: Path,
    census_metadata: dict[str, Any] | None = None,
    census_provenance: dict[str, Any] | None = None,
    route_validated: str = "partial",
) -> dict[str, Any]:
    """Write the atlas over whatever the census run completed."""

    metadata = (census_metadata or {}).get("by_slug", {})
    ledger = _last_ledger_records(run_root)
    excluded_ids = _excluded_paper_ids(run_root)
    output.mkdir(parents=True, exist_ok=True)

    observation_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    code_hashes: set[str] = set()
    unresolved_metrics: Counter[str] = Counter()

    for paper_id in _paper_universe(run_root, metadata, ledger):
        paper_dir = run_root / paper_id
        run_path = paper_dir / "run.json"
        observations = paper_dir / "observations.jsonl"
        run: dict[str, Any] | None = None
        if run_path.is_file():
            run = json.loads(run_path.read_text(encoding="utf-8"))
            code_hashes.add(str((run.get("code") or {}).get("source_tree_sha256")))
        status, status_basis = _paper_status(
            paper_id=paper_id,
            run=run,
            ledger_record=ledger.get(paper_id),
            paper_dir=paper_dir,
            excluded_ids=excluded_ids,
        )
        census = metadata.get(paper_id, {})

        candidates: list[dict[str, Any]] = []
        if observations.is_file():
            for line in observations.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    candidates.append(json.loads(line))
        review_decisions, composed_records = _review_index(paper_dir)
        for record in candidates:
            row = _observation_row(
                record,
                paper_id=paper_id,
                census=census,
                review_decisions=review_decisions,
                composed_records=composed_records,
            )
            observation_rows.append(row)
            # Count a name as unresolved only when the reference gate is what actually
            # stopped the candidate. Counting every non-registry name conflated metrics
            # the registry could not identify with candidates that failed a later gate
            # for unrelated reasons, and made the list read as a work queue it was not.
            if row["metric_raw"] and row["export_reason"] == "referential_status=unresolved":
                unresolved_metrics[row["metric_raw"]] += 1

        terminal_path = paper_dir / "private" / "row-terminal-states.json"
        planned = resolved_rows = 0
        if terminal_path.is_file():
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            counts = terminal.get("counts") or {}
            planned = int(counts.get("planned") or 0)
            resolved_rows = planned - int(counts.get("unresolved") or 0)
            for row_id, entry in sorted((terminal.get("terminal_by_row") or {}).items()):
                disposition = entry.get("disposition") or {}
                row_rows.append(
                    {
                        "paper_id": paper_id,
                        "row_id": row_id,
                        "state": entry.get("state"),
                        "disposition": disposition.get("disposition"),
                        "candidate_count": len(disposition.get("candidates") or []),
                        "invalid_reason": (terminal.get("invalid_row_reasons") or {}).get(row_id),
                    }
                )

        ledger_cost, ledger_calls = paper_ledger_cost(paper_dir)
        executed = (run or {}).get("extractor", {}).get("execution", {})
        supported = sum(1 for r in candidates if r.get("text_support") == "supported")
        paper_rows.append(
            {
                "paper_id": paper_id,
                "census_paper_id": census.get("census_paper_id"),
                "census_year": census.get("census_year"),
                "census_venue": census.get("census_venue"),
                "census_title": census.get("census_title"),
                "census_construct_claimed": census.get("census_construct_claimed"),
                "census_measurement_role": census.get("census_measurement_role"),
                "census_confidence": census.get("census_confidence"),
                "status": status,
                "status_basis": status_basis,
                "has_source_manifest": (paper_dir / "source-manifest.json").is_file(),
                "run_json_status": (run or {}).get("status"),
                "ledger_state": (ledger.get(paper_id) or {}).get("state"),
                "pages_selected": len((run or {}).get("selected_pages") or []),
                "blocks_total": executed.get("blocks_total"),
                "blocks_succeeded": executed.get("blocks_succeeded"),
                "rows_planned": planned,
                "rows_resolved": resolved_rows,
                "candidates": len(candidates),
                "supported": supported,
                "needs_review": sum(
                    1 for r in candidates if r.get("export_status") == "needs_review"
                ),
                "not_eligible": sum(
                    1 for r in candidates if r.get("export_status") == "not_eligible"
                ),
                "canonical_eee": (run or {}).get("counts", {}).get("eee_records"),
                "reported_cost_usd": round(ledger_cost, 6),
                "structured_calls": ledger_calls,
                "wall_clock_seconds": (run or {}).get("wall_clock_seconds"),
                "code_source_tree_sha256": ((run or {}).get("code") or {}).get(
                    "source_tree_sha256"
                ),
            }
        )

    observations_written = _write_csv(
        output / "observations.csv", _OBSERVATION_COLUMNS, observation_rows
    )
    with (output / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for row in observation_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_csv(
        output / "papers.csv",
        list(paper_rows[0].keys()) if paper_rows else ["paper_id"],
        paper_rows,
    )
    _write_csv(
        output / "rows.csv",
        ["paper_id", "row_id", "state", "disposition", "candidate_count", "invalid_reason"],
        row_rows,
    )

    summary = _summarize(
        observation_rows=observation_rows,
        paper_rows=paper_rows,
        row_rows=row_rows,
        ledger=ledger,
        run_ledger_totals=run_root_ledger_cost(run_root),
        code_hashes={h for h in code_hashes if h and h != "None"},
        unresolved_metrics=unresolved_metrics,
        census_provenance=census_provenance,
        route_validated=route_validated,
    )
    write_json(output / "summary.json", summary)
    (output / "index.html").write_text(_render_index(summary, paper_rows), encoding="utf-8")
    summary["observations_written"] = observations_written
    return summary


def _null_rate(rows: Sequence[dict[str, Any]], column: str) -> float | None:
    if not rows:
        return None
    missing = sum(1 for row in rows if row.get(column) in (None, "", "{}"))
    return round(missing / len(rows), 6)


def _summarize(
    *,
    observation_rows: Sequence[dict[str, Any]],
    paper_rows: Sequence[dict[str, Any]],
    row_rows: Sequence[dict[str, Any]],
    ledger: dict[str, dict[str, Any]],
    run_ledger_totals: tuple[float, int, int],
    code_hashes: set[str],
    unresolved_metrics: Counter[str],
    census_provenance: dict[str, Any] | None,
    route_validated: str,
) -> dict[str, Any]:
    by_stage = Counter(row["extraction_stage"] for row in observation_rows)
    supported = [row for row in observation_rows if row["text_support"] == "supported"]
    ledger_states = Counter(record["state"] for record in ledger.values())
    # Summed from the per-paper ledgers, not from the census ledger, which undercounted
    # papers that failed or hit their per-paper ceiling.
    reported, structured_calls, ledgers_read = run_ledger_totals
    from_paper_rows = sum(float(row.get("reported_cost_usd") or 0.0) for row in paper_rows)
    ran = [row for row in paper_rows if row.get("run_json_status")]
    with_results = [row for row in paper_rows if (row.get("candidates") or 0) > 0]
    role_counts: dict[str, Any] = defaultdict(lambda: {"papers": 0, "candidates": 0})
    for paper in ran:
        role = paper.get("census_measurement_role") or "unknown"
        role_counts[role]["papers"] += 1
        role_counts[role]["candidates"] += paper.get("candidates") or 0

    return {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "route_validated": route_validated,
        "papers_total": len(paper_rows),
        "papers_by_status": dict(Counter(str(row.get("status")) for row in paper_rows)),
        "papers_ran": len(ran),
        "papers_with_results": len(with_results),
        "papers_with_results_basis": (
            "papers whose observations.jsonl holds at least one candidate; `papers_ran` "
            "counts papers with a run.json, `papers_total` every census row the run "
            "accounts for, including not started, unavailable and reserve"
        ),
        "observations": len(observation_rows),
        "supported_observations": len(supported),
        "by_extraction_stage": dict(by_stage),
        "export_status": dict(Counter(row["export_status"] for row in observation_rows)),
        "text_support": dict(Counter(row["text_support"] for row in observation_rows)),
        "referential_status": dict(Counter(row["referential_status"] for row in observation_rows)),
        "attribution_state": dict(Counter(row["attribution_state"] for row in observation_rows)),
        "metric_canonical_source": dict(
            Counter(row["metric_canonical_source"] for row in observation_rows)
        ),
        "review_decision": dict(
            Counter(row["review_decision"] for row in observation_rows if row["review_decision"])
        ),
        "review_tier": dict(
            Counter(row["review_tier"] for row in observation_rows if row["review_tier"])
        ),
        "producer_origin_basis": dict(
            Counter(
                row["producer_origin_basis"]
                for row in observation_rows
                if row["producer_origin_basis"]
            )
        ),
        "observations_in_an_eee_record": sum(1 for row in observation_rows if row["eee_record_id"]),
        "eee_records": len(
            {row["eee_record_id"] for row in observation_rows if row["eee_record_id"]}
        ),
        "census_match_kind": dict(Counter(row["census_match_kind"] for row in observation_rows)),
        "planned_rows": len(row_rows),
        "row_states": dict(Counter(row["state"] for row in row_rows)),
        "field_completeness_null_rate": {
            column: _null_rate(observation_rows, column)
            for column in (
                "evaluated_system",
                "dataset_raw",
                "dataset_key",
                "metric_raw",
                "metric_canonical",
                "value_raw",
                "value_unit",
                "language_raw",
                "language_iso639",
                "split",
                "subset",
                "construct",
                "operationalization",
                "decision_rule",
                "metric_parameters",
                "evidence_label",
                "evidence_row",
            )
        },
        "by_measurement_role": {role: dict(counts) for role, counts in role_counts.items()},
        "top_unresolved_metric_names": [
            {"metric_raw": name, "count": count}
            for name, count in unresolved_metrics.most_common(50)
        ],
        "ledger_states": dict(ledger_states),
        "provider_reported_cost_usd": round(reported, 6),
        "provider_reported_cost_basis": (
            f"summed from every private/provider-budget-ledger.jsonl under the run root "
            f"({ledgers_read} ledgers), which counts papers that failed, hit their "
            f"per-paper ceiling, or wrote no run.json"
        ),
        "provider_reported_cost_usd_from_paper_rows": round(from_paper_rows, 6),
        "by_measurement_role_basis": "papers with a run.json",
        "structured_calls": structured_calls,
        "code_source_tree_sha256_set": sorted(code_hashes),
        "census_provenance": census_provenance
        or {"producer": "unknown", "validation_status": "none"},
    }


def _render_index(summary: dict[str, Any], paper_rows: Sequence[dict[str, Any]]) -> str:
    boundary = "".join(f"<li>{item}</li>" for item in summary["claim_boundary"])
    header = (
        "<tr><th>paper</th><th>census id</th><th>year</th><th>venue</th>"
        "<th>candidates</th><th>supported</th><th>status</th><th>review</th></tr>"
    )
    # Only papers that actually ran get a row: the others have no review page to link to,
    # and their counts are in the status table below.
    ran = [row for row in paper_rows if row.get("run_json_status")]
    body = "".join(
        "<tr>"
        f"<td>{row['paper_id']}</td><td>{row.get('census_paper_id') or ''}</td>"
        f"<td>{row.get('census_year') or ''}</td><td>{row.get('census_venue') or ''}</td>"
        f"<td>{row.get('candidates')}</td><td>{row.get('supported')}</td>"
        f"<td>{row.get('status')}</td>"
        f"<td><a href='../{row['paper_id']}/review.html'>review</a></td>"
        "</tr>"
        for row in sorted(ran, key=lambda item: -(item.get("candidates") or 0))
    )
    status_rows = "".join(
        f"<tr><td>{status}</td><td>{count}</td></tr>"
        for status, count in sorted(summary.get("papers_by_status", {}).items())
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Toxicity Evaluation Evidence Atlas</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 70rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 0.35rem 0.6rem; text-align: left; }}
th {{ background: #f4f4f4; }}
.boundary {{ background: #fff8e1; border-left: 4px solid #e6a700; padding: 1rem; }}
</style></head><body>
<h1>Toxicity Evaluation Evidence Atlas</h1>
<div class="boundary"><strong>What this is, and is not</strong><ul>{boundary}</ul></div>
<p>{summary["papers_with_results"]} papers with results of
{summary["papers_ran"]} that ran, out of {summary["papers_total"]} census papers
accounted for. {summary["observations"]} candidate observations,
{summary["supported_observations"]} evidence-supported,
{summary["planned_rows"]} planned dense-table rows accounted for.</p>
<h2>Every census paper by status</h2>
<table><tr><th>status</th><th>papers</th></tr>{status_rows}</table>
<h2>Papers that ran</h2>
<table>{header}{body}</table>
</body></html>
"""
