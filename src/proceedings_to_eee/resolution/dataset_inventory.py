"""Post-hoc dataset inventory of a finished run: one bounded model call per paper.

The sixth model boundary. A paper names the data it evaluated on in its captions and in
its data, task and experiment sections, and the extractor's per-block view rarely sees
that name next to the number. This stage asks a model to list those names, with verbatim
quotes, from exactly that material; deterministic code then re-finds every quote on the
frozen layout and drops any entry whose evidence it cannot find. The inventory is an
annotation beside the run. It never writes `dataset_raw` or any field that enters an
observation id, a field-provenance hash or a review gate hash; binding candidates to its
entries is a separate deterministic step (`dataset_binding`).

What the provider never sees: candidates, gate outcomes, attribution state, review
decisions. It is asked what the paper says about its data, not what the pipeline made
of it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proceedings_to_eee.census_audit import quotation_is_on_page
from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.extraction.region_index import RegionKind, build_region_index
from proceedings_to_eee.io import write_json
from proceedings_to_eee.sources.manifest import SourceManifest

INVENTORY_SCHEMA_VERSION = "dataset-inventory/0.2"
INVENTORY_PROPOSAL_SCHEMA_VERSION = "dataset-inventory-proposal/0.1"
INVENTORY_SUMMARY_SCHEMA_VERSION = "dataset-inventory-summary/0.2"

#: How a quote is matched against the frozen layout: whitespace collapsed, nothing
#: else. A quote the layout breaks with a line-end hyphen therefore fails. The gate
#: names this so a reader of its drop counts knows what was and was not relaxed.
QUOTE_NORMALISATION = "whitespace_only"
INVENTORY_SCHEMA_NAME = "dataset_inventory"
INVENTORY_FILENAME = "dataset-inventory.json"
INVENTORY_PRIVATE_FILENAME = "dataset-inventory.jsonl"

#: The census extractor model. The inventory reads the same papers under the same
#: provider, so the historical identifier is the default, not a recommendation.
DEFAULT_INVENTORY_MODEL = "google/gemini-3.5-flash-lite"

#: Determinism settings, passed explicitly because the budget ledger fingerprints the
#: request before dispatch, exactly as the review stage does.
INVENTORY_TEMPERATURE = 0.0
INVENTORY_REASONING_EFFORT = "minimal"
INVENTORY_SEED = 7

#: Upper bound on the section text sent per paper. Generous: a long paper's data and
#: experiment sections fit, and the gate records how much was sent and whether it was cut.
DEFAULT_MAX_SECTION_CHARACTERS = 40_000

#: Consecutive failures tolerated before the run gives up; the review stage's rule.
MAX_CONSECUTIVE_FAILURES = 25

ENTRY_KINDS = ("dataset", "shared_task_subtask", "benchmark", "unknown")

#: Section headings whose text is sent. Matched against the enclosing section title the
#: region index carries across page breaks, case-insensitively.
SECTION_HEADING_PATTERN = re.compile(
    r"data|dataset|corpus|task|benchmark|experimental setup|experiments|evaluation",
    re.IGNORECASE,
)

_SPACES = re.compile(r"\s+")

_SECTION_REGION_KINDS = frozenset(
    {RegionKind.HEADING, RegionKind.PROSE, RegionKind.TABLE, RegionKind.CAPTION}
)

SYSTEM_PROMPT = """You inventory the datasets a scientific paper evaluates systems on.

You are given the paper's title, every table and figure caption, and the text of its
data, task, benchmark, experimental-setup, experiments and evaluation sections, each
piece labelled with its page number.

List every dataset, corpus, shared-task subtask or benchmark on which this paper reports
evaluation results. For each entry give:
- name_raw: the name exactly as the paper prints it, copied verbatim.
- aliases_in_paper: other spellings, abbreviations or labels the paper uses for the same
  data, each copied verbatim. Empty when there are none.
- kind: dataset, shared_task_subtask, benchmark, or unknown when the paper does not say.
- languages: the languages the paper states for this data. Empty when it states none.
- splits: the splits the paper names for this data (for example train, dev, test), with
  the size the paper prints for each split or null when it prints none.
- introduced_here: true only when the paper presents this data as its own new release.
- evidence: one to three quotations, each copied verbatim from the supplied text, each
  with the page number it was labelled with, that show the paper evaluating on this data
  or describing it.

Rules, checked mechanically after you answer:
1. Never invent a name. Every name_raw and alias must occur in the supplied text.
2. Every quotation must be an exact, contiguous copy of the supplied text from the page
   you cite. Do not repair, shorten with ellipses, or merge text from different places.
3. Do not list data the paper only mentions in related work or cites as prior results
   without evaluating on it. When in doubt, leave it out.
4. Judge only from the supplied text. Do not use outside knowledge about the datasets.
Return an empty entries list when the paper evaluates on no identifiable data.
"""


def inventory_response_schema() -> dict[str, Any]:
    """Strict wire schema for one paper's inventory. Every field present and typed."""

    split_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "size_reported"],
        "properties": {
            "name": {"type": "string"},
            "size_reported": {"type": ["string", "null"]},
        },
    }
    evidence_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["page", "exact_quote"],
        "properties": {
            "page": {"type": "integer"},
            "exact_quote": {"type": "string"},
        },
    }
    entry_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "name_raw",
            "aliases_in_paper",
            "kind",
            "languages",
            "splits",
            "introduced_here",
            "evidence",
        ],
        "properties": {
            "name_raw": {"type": "string"},
            "aliases_in_paper": {"type": "array", "items": {"type": "string"}},
            "kind": {"type": "string", "enum": list(ENTRY_KINDS)},
            "languages": {"type": "array", "items": {"type": "string"}},
            "splits": {"type": "array", "items": split_schema},
            "introduced_here": {"type": "boolean"},
            "evidence": {"type": "array", "items": evidence_schema},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["entries"],
        "properties": {"entries": {"type": "array", "items": entry_schema}},
    }


def inventory_prompt_sha256() -> str:
    """Hash of the system prompt, recorded on every gate it writes."""

    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def inventory_schema_sha256() -> str:
    payload = json.dumps(inventory_response_schema(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def entry_id(name_raw: str, kind: str) -> str:
    """Stable id of one inventory entry: a hash of its printed name and kind."""

    return "ds_" + _sha256([name_raw, kind])[:20]


@dataclass(frozen=True)
class InventoryInput:
    """What one paper's inventory call is built from, plus the frozen text it is checked on."""

    paper_id: str
    title: str
    #: (page, caption text) for every caption region on every page.
    captions: list[tuple[int, str]]
    #: (page, section title, region text) for every region inside a matching section.
    sections: list[tuple[int, str, str]]
    sections_truncated: bool
    section_characters: int
    page_count: int
    #: Page number -> texts a quote may be re-found in: the page text and its regions.
    searchable: dict[int, list[str]] = field(repr=False)

    def hashes(self) -> dict[str, Any]:
        """Quote-free record of what was sent. Hashes only; the text stays private."""

        return {
            "title_sha256": hashlib.sha256(self.title.encode("utf-8")).hexdigest(),
            "captions_sha256": _sha256(self.captions),
            "sections_sha256": _sha256(self.sections),
            "caption_count": len(self.captions),
            "section_count": len(self.sections),
            "section_characters": self.section_characters,
            "sections_truncated": self.sections_truncated,
            "page_count": self.page_count,
        }


def _load_layout(paper_dir: Path) -> PdfLayout | None:
    path = paper_dir / "private" / "layout.json"
    if not path.is_file():
        return None
    return PdfLayout.model_validate(json.loads(path.read_text(encoding="utf-8")))


def collect_inventory_input(
    paper_dir: Path, *, max_section_characters: int = DEFAULT_MAX_SECTION_CHARACTERS
) -> InventoryInput | None:
    """Gather the title, captions and matching sections from the frozen layout.

    Returns None when the paper has no manifest or layout. Captions and the searchable
    text come from every page. Section text is taken in page and reading order until
    the character bound is reached; the cut is recorded and stops only the sections.
    """

    manifest_path = paper_dir / "source-manifest.json"
    if not manifest_path.is_file():
        return None
    layout = _load_layout(paper_dir)
    if layout is None:
        return None
    manifest = SourceManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    indexes = build_region_index(layout)
    captions: list[tuple[int, str]] = []
    sections: list[tuple[int, str, str]] = []
    searchable: dict[int, list[str]] = {}
    characters = 0
    truncated = False
    for page in sorted(layout.pages, key=lambda item: item.page):
        index = indexes[page.page]
        searchable[page.page] = [page.text, *(region.text for region in index.regions)]
        for region in index.regions:
            if region.kind is RegionKind.CAPTION and region.caption is not None:
                captions.append((page.page, region.caption.text))
            if truncated:
                continue
            if region.kind not in _SECTION_REGION_KINDS or region.in_references:
                continue
            title = " ".join(region.section_path).strip()
            if not title or SECTION_HEADING_PATTERN.search(title) is None:
                continue
            text = region.text.strip()
            if not text:
                continue
            if characters + len(text) > max_section_characters:
                truncated = True
                remaining = max_section_characters - characters
                if remaining > 0:
                    sections.append((page.page, title, text[:remaining]))
                    characters += remaining
                continue
            sections.append((page.page, title, text))
            characters += len(text)
    return InventoryInput(
        paper_id=manifest.paper_id,
        title=manifest.title,
        captions=captions,
        sections=sections,
        sections_truncated=truncated,
        section_characters=characters,
        page_count=layout.page_count,
        searchable=searchable,
    )


def inventory_prompt(source: InventoryInput) -> str:
    """Render the user message: title, captions and section text, each page-labelled."""

    lines = ["PAPER TITLE", source.title, "", "CAPTIONS"]
    if source.captions:
        lines.extend(f"[page {page}] {text}" for page, text in source.captions)
    else:
        lines.append("(none found)")
    lines.extend(["", "SECTION TEXT"])
    if source.sections:
        for page, title, text in source.sections:
            lines.extend([f"--- page {page}, section: {title} ---", text])
    else:
        lines.append("(no matching section found)")
    if source.sections_truncated:
        lines.extend(["", "(section text was cut at the configured character bound)"])
    return "\n".join(lines)


@dataclass
class VerifiedInventory:
    """What survived the deterministic check of one proposal."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    proposed: int = 0
    dropped_unverified: int = 0
    #: Entries whose printed name is nowhere in the frozen layout.
    dropped_unnamed: int = 0
    dropped_malformed: int = 0
    dropped_duplicate: int = 0
    #: Aliases removed from kept entries because the layout does not print them.
    dropped_aliases: int = 0
    #: Per proposed entry, the verified flag of each of its evidence items.
    verification: list[list[bool]] = field(default_factory=list)


def _quote_sha256(quote: str) -> str:
    return hashlib.sha256(quote.encode("utf-8")).hexdigest()


def _printed_text(searchable: dict[int, list[str]]) -> str:
    """Every page's text and regions, whitespace-collapsed and case-folded, for names."""

    joined = " \n ".join(text for texts in searchable.values() for text in texts)
    return _SPACES.sub(" ", joined).casefold()


def _is_printed(name: str, printed: str) -> bool:
    """A name counts as printed when, whitespace-collapsed and ignoring case, it occurs
    anywhere in the layout. Wording is never relaxed; a necessary condition against an
    invented name, not an identity match (a footnote digit glued to a name still passes)."""

    needle = _SPACES.sub(" ", name).strip().casefold()
    return bool(needle) and needle in printed


def _clean_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in cleaned:
            cleaned.append(value.strip())
    return cleaned


def verify_inventory(
    payload: dict[str, Any], searchable: dict[int, list[str]]
) -> VerifiedInventory:
    """Keep an entry only when its quotes and its name are re-found on the frozen layout.

    A quote is re-found when, after whitespace normalisation only, it occurs in the page
    text or in one of the page's layout regions of the cited page; a column-panel region
    keeps a quote whole that the interleaved page text splits. A wrong page, a repaired
    quote, a quote broken by a line-end hyphen or an invented one all fail, and an entry
    without one verified quote is dropped and counted. The entry's printed name must
    then occur, whitespace-collapsed and ignoring case, somewhere in the layout, or the
    entry is dropped and counted as unnamed; an alias that does not is removed and
    counted. A kind outside the vocabulary is malformed, not repaired. Quotes never
    enter the returned entries; only their hashes do.
    """

    outcome = VerifiedInventory()
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return outcome
    printed = _printed_text(searchable)
    seen: set[str] = set()
    for proposed in entries:
        outcome.proposed += 1
        if not isinstance(proposed, dict):
            outcome.dropped_malformed += 1
            outcome.verification.append([])
            continue
        name_raw = proposed.get("name_raw")
        kind = proposed.get("kind")
        if not isinstance(name_raw, str) or not name_raw.strip() or kind not in ENTRY_KINDS:
            outcome.dropped_malformed += 1
            outcome.verification.append([])
            continue
        name_raw = name_raw.strip()
        anchors: list[dict[str, Any]] = []
        flags: list[bool] = []
        for item in proposed.get("evidence") or []:
            if not isinstance(item, dict):
                continue
            page = item.get("page")
            quote = item.get("exact_quote")
            if not isinstance(page, int) or isinstance(page, bool) or not isinstance(quote, str):
                continue
            verified = any(quotation_is_on_page(quote, text) for text in searchable.get(page, []))
            flags.append(verified)
            anchors.append(
                {"page": page, "quote_sha256": _quote_sha256(quote), "verified": verified}
            )
        outcome.verification.append(flags)
        if not any(flags):
            outcome.dropped_unverified += 1
            continue
        if not _is_printed(name_raw, printed):
            outcome.dropped_unnamed += 1
            continue
        aliases = _clean_strings(proposed.get("aliases_in_paper"))
        printed_aliases = [alias for alias in aliases if _is_printed(alias, printed)]
        outcome.dropped_aliases += len(aliases) - len(printed_aliases)
        identity = entry_id(name_raw, kind)
        if identity in seen:
            outcome.dropped_duplicate += 1
            continue
        seen.add(identity)
        splits: list[dict[str, Any]] = []
        for split in proposed.get("splits") or []:
            if not isinstance(split, dict):
                continue
            name = split.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            size = split.get("size_reported")
            splits.append(
                {"name": name.strip(), "size_reported": size if isinstance(size, str) else None}
            )
        outcome.entries.append(
            {
                "entry_id": identity,
                "name_raw": name_raw,
                "aliases_in_paper": printed_aliases,
                "kind": kind,
                "languages": _clean_strings(proposed.get("languages")),
                "splits": splits,
                "introduced_here": proposed.get("introduced_here") is True,
                "anchors": anchors,
            }
        )
    return outcome


def load_inventory(paper_dir: Path) -> dict[str, Any] | None:
    """Read a paper's quote-free inventory gate, or None when it has none."""

    path = paper_dir / INVENTORY_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def inventory_content_sha256(gate: Mapping[str, Any]) -> str:
    """Hash of what the gate says, so a record can name the inventory it was bound to.

    Telemetry (latency, cost, request and response hashes) is left out: two runs that
    verify the same entries under the same prompt, schema, model and input hash the
    same, and a record names the inventory content rather than one call's timing.
    """

    return _sha256({key: value for key, value in gate.items() if key != "telemetry"})


def _gate_is_current(gate: dict[str, Any] | None) -> bool:
    return (
        gate is not None
        and gate.get("inventory_prompt_sha256") == inventory_prompt_sha256()
        and gate.get("inventory_schema_sha256") == inventory_schema_sha256()
    )


def _append_private_record(paper_dir: Path, record: dict[str, Any]) -> None:
    path = paper_dir / "private" / INVENTORY_PRIVATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def select_papers(run_root: Path, *, paper_ids: Iterable[str] | None = None) -> list[Path]:
    """Paper directories with candidates, in id order, restricted to `paper_ids` if given."""

    wanted = set(paper_ids) if paper_ids is not None else None
    papers: list[Path] = []
    for observations in sorted(run_root.glob("*/observations.jsonl")):
        if wanted is None or observations.parent.name in wanted:
            papers.append(observations.parent)
    return papers


def run_inventory(
    *,
    run_root: Path,
    model: str,
    call: Callable[[str, str], tuple[dict[str, Any], dict[str, Any]]],
    paper_ids: Iterable[str] | None = None,
    max_section_characters: int = DEFAULT_MAX_SECTION_CHARACTERS,
    papers: Sequence[Path] | None = None,
    stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Inventory every selected paper without a current gate, one call each.

    `call` returns the parsed payload and a secret-free telemetry mapping. `stop` is
    consulted before each call so a budget ceiling ends the run between papers; every
    completed inventory is already on disk when it does. A paper whose gate carries the
    current prompt and schema hashes is skipped, so a rerun resumes.
    """

    selected = list(papers) if papers is not None else select_papers(run_root, paper_ids=paper_ids)
    inventoried = already = failed = no_input = 0
    entries_total = 0
    kinds: Counter[str] = Counter()
    dropped: Counter[str] = Counter({"unverified": 0, "unnamed": 0, "malformed": 0, "duplicate": 0})
    failure_classes: Counter[str] = Counter()
    papers_without_entries: list[str] = []
    truncated_papers: list[str] = []
    gate_file_sha256s: dict[str, str] = {}
    dropped_aliases = 0
    stopped_for_budget = stopped_for_failures = False
    consecutive_failures = 0

    for paper_dir in selected:
        if _gate_is_current(load_inventory(paper_dir)):
            already += 1
            continue
        source = collect_inventory_input(paper_dir, max_section_characters=max_section_characters)
        if source is None:
            no_input += 1
            continue
        if stop is not None and stop():
            stopped_for_budget = True
            break
        try:
            payload, telemetry = call(SYSTEM_PROMPT, inventory_prompt(source))
        except Exception as error:  # noqa: BLE001 - one paper must not end the run
            failed += 1
            consecutive_failures += 1
            failure_classes[type(error).__name__] += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                stopped_for_failures = True
                break
            continue
        consecutive_failures = 0
        verified = verify_inventory(payload, source.searchable)
        _append_private_record(
            paper_dir,
            {
                "schema_version": INVENTORY_PROPOSAL_SCHEMA_VERSION,
                "paper_id": paper_dir.name,
                "model": model,
                "inventory_prompt_sha256": inventory_prompt_sha256(),
                "inventory_schema_sha256": inventory_schema_sha256(),
                "proposal": payload,
                "evidence_verified": verified.verification,
                **telemetry,
            },
        )
        gate = {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "paper_id": paper_dir.name,
            "model": model,
            "inventory_prompt_sha256": inventory_prompt_sha256(),
            "inventory_schema_sha256": inventory_schema_sha256(),
            "max_section_characters": max_section_characters,
            "quote_normalisation": QUOTE_NORMALISATION,
            "input": source.hashes(),
            "entries": verified.entries,
            "proposed_entries": verified.proposed,
            "dropped_unverified": verified.dropped_unverified,
            "dropped_unnamed": verified.dropped_unnamed,
            "dropped_malformed": verified.dropped_malformed,
            "dropped_duplicate": verified.dropped_duplicate,
            "dropped_aliases": verified.dropped_aliases,
            "telemetry": telemetry,
        }
        gate_file_sha256s[paper_dir.name] = write_json(paper_dir / INVENTORY_FILENAME, gate)
        inventoried += 1
        entries_total += len(verified.entries)
        kinds.update(entry["kind"] for entry in verified.entries)
        dropped["unverified"] += verified.dropped_unverified
        dropped["unnamed"] += verified.dropped_unnamed
        dropped["malformed"] += verified.dropped_malformed
        dropped["duplicate"] += verified.dropped_duplicate
        dropped_aliases += verified.dropped_aliases
        if not verified.entries:
            papers_without_entries.append(paper_dir.name)
        if source.sections_truncated:
            truncated_papers.append(paper_dir.name)

    return {
        "schema_version": INVENTORY_SUMMARY_SCHEMA_VERSION,
        "model": model,
        "inventory_prompt_sha256": inventory_prompt_sha256(),
        "inventory_schema_sha256": inventory_schema_sha256(),
        "max_section_characters": max_section_characters,
        "quote_normalisation": QUOTE_NORMALISATION,
        "selected": len(selected),
        "inventoried": inventoried,
        "already_inventoried": already,
        "skipped_no_input": no_input,
        "failed": failed,
        "failure_classes": dict(failure_classes),
        "stopped_for_budget": stopped_for_budget,
        "stopped_for_repeated_failures": stopped_for_failures,
        "entries": entries_total,
        "entries_by_kind": dict(sorted(kinds.items())),
        "dropped": dict(sorted(dropped.items())),
        "dropped_aliases": dropped_aliases,
        "papers_without_entries": papers_without_entries,
        "papers_with_truncated_sections": truncated_papers,
        "gate_file_sha256s": gate_file_sha256s,
        "claim_boundary": [
            "Entries are model proposals whose quotes deterministic code re-found on the "
            "frozen layout. They are not human labels, and their error rate is unmeasured.",
            "The inventory never writes dataset_raw or any id-bearing candidate field; "
            "binding candidates to entries is a separate deterministic step.",
        ],
    }
