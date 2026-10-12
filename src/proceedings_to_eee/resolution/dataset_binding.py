"""Deterministic binding of candidates to a paper's verified dataset inventory.

Three rules, applied in order and recorded by id. `exact`: the normalised raw dataset
string equals an entry's name or alias. `caption`: the caption or header text of the
table the candidate is anchored in names exactly one entry, and the raw string names
nothing that entry does not (it is structurally not a name, or each of its content words
occurs in the entry's name or aliases). `unique`: the paper has exactly one dataset or
subtask entry and the raw string is structurally not a name (a split word, a language, a
generic word, a table reference). Ambiguity at any rule is `none`, and so is a caption
match that would replace a raw string naming more than the entry. A split is derived only
when the candidate states none, from the anchor's column header or the attached caption,
and lives in the sidecar beside the binding.

Nothing here touches `scope`: the raw string stays where the extractor put it, inside
the observation id and every hash that binds a gate to the candidate.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from proceedings_to_eee.domain.dataset_binding import (
    DatasetBinding,
    DatasetBindingClass,
    SplitSource,
)
from proceedings_to_eee.domain.observation import CandidateObservation, EvidenceAnchor
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.region_index import (
    PageRegionIndex,
    Region,
    RegionKind,
    build_page_region_index,
    build_region_index,
    locate_quote_in_index,
)

#: Raw strings the `unique` rule may bind: a split word, optionally followed by a
#: set/split/data word; a table reference; one of the generic words.
SPLIT_WORD = re.compile(
    r"^(train|training|dev|development|validation|val|test|testing)"
    r"(?: (?:set|split|data|partition|fold))?$"
)
TABLE_REFERENCE = re.compile(r"^(?:table|tab|tbl) \d+[a-z]?$")
GENERIC_WORDS = frozenset({"dataset", "data", "corpus", "overall", "all", "total"})

#: A leading determiner the caption rule looks past: "our test set" is a split phrase.
_DETERMINER = re.compile(r"^(?:our|the) ")
#: Words that carry no dataset identity on their own. A raw string made of these and of
#: the words of the matched entry names nothing the entry does not.
_SPLIT_WORDS = frozenset(
    {"train", "training", "dev", "development", "validation", "val", "test", "testing"}
)
_FILLER_WORDS = frozenset(
    {"the", "our", "a", "an", "of", "on", "in", "for", "and", "with", "set", "split",
     "subset", "partition", "fold"}
)  # fmt: skip

#: Language names and two-letter codes a raw dataset string may consist of. Names only;
#: a string that is wholly made of these is a language, not a dataset.
LANGUAGE_NAMES = frozenset(
    {
        "arabic", "bengali", "bulgarian", "chinese", "croatian", "czech", "danish",
        "dutch", "english", "estonian", "finnish", "french", "german", "greek",
        "gujarati", "hebrew", "hindi", "hungarian", "indonesian", "italian", "japanese",
        "kannada", "korean", "latvian", "lithuanian", "malayalam", "marathi", "nepali",
        "norwegian", "persian", "polish", "portuguese", "punjabi", "romanian", "russian",
        "serbian", "sinhala", "slovak", "slovenian", "spanish", "swahili", "swedish",
        "tamil", "telugu", "thai", "turkish", "ukrainian", "urdu", "vietnamese",
        "ar", "bn", "bg", "zh", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
        "el", "gu", "he", "hi", "hu", "id", "it", "ja", "kn", "ko", "lv", "lt", "ml",
        "mr", "ne", "no", "fa", "pl", "pt", "pa", "ro", "ru", "sr", "si", "sk", "sl",
        "es", "sw", "sv", "ta", "te", "th", "tr", "uk", "ur", "vi",
    }
)  # fmt: skip

COLUMN_SPLIT = re.compile(r"^(test|dev|development|validation|val|train(?:ing)?)\b", re.IGNORECASE)
CAPTION_SPLIT = re.compile(
    r"\bon the (test|dev|development|validation) (?:set|split|data)\b", re.IGNORECASE
)

_PUNCTUATION = re.compile(r"[^\w\s]+")
_SPACES = re.compile(r"\s+")

RULE_EXACT_NAME = "exact_name"
RULE_EXACT_ALIAS = "exact_alias"
RULE_CAPTION_SINGLE_ENTRY = "caption_single_entry"
RULE_UNIQUE_SPLIT_WORD = "unique_split_word"
RULE_UNIQUE_LANGUAGE_NAME = "unique_language_name"
RULE_UNIQUE_GENERIC_WORD = "unique_generic_word"
RULE_UNIQUE_TABLE_REFERENCE = "unique_table_reference"
RULE_NONE_EXACT_AMBIGUOUS = "none_exact_ambiguous"
RULE_NONE_CAPTION_AMBIGUOUS = "none_caption_ambiguous"
RULE_NONE_CAPTION_OVERRIDES_NAMED_RAW = "none_caption_overrides_named_raw"
RULE_NONE_NO_RULE_MATCHED = "none_no_rule_matched"

#: Entry kinds the `unique` rule counts. A benchmark or an unknown entry does not make a
#: paper single-dataset.
_UNIQUE_KINDS = frozenset({"dataset", "shared_task_subtask"})


def normalise_name(value: str | None) -> str:
    """Lower-case, strip punctuation, collapse spaces. Never wording or digits."""

    if not value:
        return ""
    return _SPACES.sub(" ", _PUNCTUATION.sub(" ", value.casefold())).strip()


@dataclass(frozen=True)
class InventoryEntry:
    """One verified inventory entry with its names pre-normalised for matching."""

    entry_id: str
    name_raw: str
    kind: str
    languages: tuple[str, ...]
    splits: tuple[str, ...]
    #: Normalised name and aliases, name first, blanks and very short terms removed.
    terms: tuple[str, ...]
    name_term: str


def inventory_entries(inventory: Mapping[str, Any]) -> list[InventoryEntry]:
    """Typed view of the gate's entries. Malformed entries are left out, never repaired."""

    entries: list[InventoryEntry] = []
    for raw in inventory.get("entries") or []:
        if not isinstance(raw, dict):
            continue
        name_raw = raw.get("name_raw")
        identity = raw.get("entry_id")
        if not isinstance(name_raw, str) or not isinstance(identity, str):
            continue
        name_term = normalise_name(name_raw)
        aliases = [
            normalise_name(alias)
            for alias in raw.get("aliases_in_paper") or []
            if isinstance(alias, str)
        ]
        terms = tuple(dict.fromkeys(term for term in [name_term, *aliases] if len(term) >= 2))
        if not terms:
            continue
        entries.append(
            InventoryEntry(
                entry_id=identity,
                name_raw=name_raw,
                kind=str(raw.get("kind") or "unknown"),
                languages=tuple(
                    item for item in raw.get("languages") or [] if isinstance(item, str)
                ),
                splits=tuple(
                    split["name"]
                    for split in raw.get("splits") or []
                    if isinstance(split, dict) and isinstance(split.get("name"), str)
                ),
                terms=terms,
                name_term=name_term,
            )
        )
    return entries


def _contains_term(haystack: str, term: str) -> bool:
    return f" {term} " in f" {haystack} "


@dataclass
class _PaperIndex:
    """Region indexes for one layout, with the single-panel fallback cached per page."""

    pages: dict[int, PageFragment]
    indexes: dict[int, PageRegionIndex]
    single_panel: dict[int, PageRegionIndex]

    @classmethod
    def build(cls, layout: PdfLayout) -> _PaperIndex:
        return cls(
            pages={page.page: page for page in layout.pages},
            indexes=build_region_index(layout),
            single_panel={},
        )

    def table_region(self, anchor: EvidenceAnchor) -> Region | None:
        """The table region an anchor's quote sits in, or None when it sits elsewhere."""

        page = self.pages.get(anchor.page)
        index = self.indexes.get(anchor.page)
        if page is None or index is None:
            return None
        location = locate_quote_in_index(index, page, anchor.quote)
        if location is None or location.kind is not RegionKind.TABLE:
            return None
        if location.single_panel:
            # The quote was found only on the physical page lines (a full-width table
            # on a two-column page) and located against a single-panel index. Its
            # region ids collide with the panelled index's ids, so the id is looked up
            # in the index that produced it, not in the one whose first-column
            # fragment happens to carry the same id.
            if anchor.page not in self.single_panel:
                self.single_panel[anchor.page] = build_page_region_index(page, single_panel=True)
            index = self.single_panel[anchor.page]
        return _region_by_id(index, location.region_id)


def _region_by_id(index: PageRegionIndex, region_id: str) -> Region | None:
    return next((item for item in index.regions if item.region_id == region_id), None)


def _header_text(region: Region) -> str:
    return " ".join(cell for row in region.rows if row.is_header for cell in row.cells)


def _anchored_table(
    candidate: CandidateObservation, index: _PaperIndex
) -> tuple[EvidenceAnchor, Region] | None:
    for anchor in candidate.evidence:
        region = index.table_region(anchor)
        if region is not None:
            return anchor, region
    return None


def _exact_matches(raw: str, entries: Sequence[InventoryEntry]) -> list[tuple[InventoryEntry, str]]:
    matches: list[tuple[InventoryEntry, str]] = []
    for entry in entries:
        if raw == entry.name_term:
            matches.append((entry, RULE_EXACT_NAME))
        elif raw in entry.terms:
            matches.append((entry, RULE_EXACT_ALIAS))
    return matches


def _caption_matches(text: str, entries: Sequence[InventoryEntry]) -> list[InventoryEntry]:
    haystack = normalise_name(text)
    if not haystack:
        return []
    return [
        entry for entry in entries if any(_contains_term(haystack, term) for term in entry.terms)
    ]


def _unique_rule(raw: str) -> str | None:
    if SPLIT_WORD.match(raw):
        return RULE_UNIQUE_SPLIT_WORD
    if TABLE_REFERENCE.match(raw):
        return RULE_UNIQUE_TABLE_REFERENCE
    if raw in GENERIC_WORDS:
        return RULE_UNIQUE_GENERIC_WORD
    words = raw.split()
    if words and all(word in LANGUAGE_NAMES for word in words):
        return RULE_UNIQUE_LANGUAGE_NAME
    return None


def _raw_names_nothing_beyond(raw: str, entry: InventoryEntry) -> bool:
    """True when binding `raw` to `entry` discards no name the paper printed.

    Either the raw string is structurally not a name (a split phrase, optionally after
    "our" or "the", a generic word, a table reference, a language), or every content
    word of it occurs in the entry's name or aliases. "Twitter and Fox News" names more
    than an entry called "Twitter"; "six movies" names nothing beyond an entry whose
    alias is "six movies dataset".
    """

    if _unique_rule(_DETERMINER.sub("", raw)) is not None:
        return True
    entry_words = {word for term in entry.terms for word in term.split()}
    content = [
        word
        for word in raw.split()
        if word not in _FILLER_WORDS
        and word not in _SPLIT_WORDS
        and word not in GENERIC_WORDS
        and word not in LANGUAGE_NAMES
    ]
    return all(word in entry_words for word in content)


def _derive_split(
    candidate: CandidateObservation, anchored: tuple[EvidenceAnchor, Region] | None
) -> tuple[str, SplitSource, int] | None:
    """Read a split from the anchor column header, else from the attached caption."""

    assert candidate.scope is not None
    if candidate.scope.split and candidate.scope.split.strip():
        return None
    for anchor in candidate.evidence:
        if anchor.column:
            match = COLUMN_SPLIT.match(anchor.column.strip())
            if match:
                return match.group(1).casefold(), SplitSource.COLUMN_HEADER, anchor.page
    if anchored is not None:
        anchor, region = anchored
        if region.caption is not None:
            match = CAPTION_SPLIT.search(region.caption.text)
            if match:
                return match.group(1).casefold(), SplitSource.CAPTION, anchor.page
    return None


def bind_candidate(
    candidate: CandidateObservation,
    *,
    entries: Sequence[InventoryEntry],
    index: _PaperIndex,
    inventory_sha256: str,
) -> DatasetBinding:
    """Apply the rules in order to one candidate and return its sidecar."""

    assert candidate.scope is not None
    raw = normalise_name(candidate.scope.dataset_raw)
    anchored = _anchored_table(candidate, index)
    binding_class = DatasetBindingClass.NONE
    rule_id = RULE_NONE_NO_RULE_MATCHED
    entry: InventoryEntry | None = None

    exact = _exact_matches(raw, entries)
    distinct = {match.entry_id for match, _ in exact}
    if len(distinct) == 1:
        entry, rule_id = exact[0]
        binding_class = DatasetBindingClass.EXACT
    elif len(distinct) > 1:
        rule_id = RULE_NONE_EXACT_AMBIGUOUS
    elif anchored is not None and (
        matches := _caption_matches(
            f"{anchored[1].caption.text if anchored[1].caption else ''} "
            f"{_header_text(anchored[1])}",
            entries,
        )
    ):
        if len({match.entry_id for match in matches}) > 1:
            rule_id = RULE_NONE_CAPTION_AMBIGUOUS
        elif _raw_names_nothing_beyond(raw, matches[0]):
            entry, rule_id = matches[0], RULE_CAPTION_SINGLE_ENTRY
            binding_class = DatasetBindingClass.CAPTION
        else:
            # The caption names one entry, but the raw string names something else
            # too (an inventory that lost an entry, or a combined evaluation). The
            # raw string is kept rather than narrowed.
            rule_id = RULE_NONE_CAPTION_OVERRIDES_NAMED_RAW
    else:
        candidates = [item for item in entries if item.kind in _UNIQUE_KINDS]
        unique_rule = _unique_rule(raw)
        if len(candidates) == 1 and unique_rule is not None:
            entry, rule_id = candidates[0], unique_rule
            binding_class = DatasetBindingClass.UNIQUE

    split = _derive_split(candidate, anchored)
    return DatasetBinding(
        inventory_sha256=inventory_sha256,
        binding_class=binding_class,
        rule_id=rule_id,
        entry_id=entry.entry_id if entry else None,
        entry_name=entry.name_raw if entry else None,
        entry_kind=entry.kind if entry else None,
        languages=list(entry.languages) if entry else [],
        splits=list(entry.splits) if entry else [],
        split=split[0] if split else None,
        split_source=split[1] if split else None,
        split_page=split[2] if split else None,
    )


@dataclass
class BindingOutcome:
    """Counts of what the binding step did to one paper's candidates."""

    classes: Counter[str]
    rules: Counter[str]
    split_sources: Counter[str]


def bind_paper_candidates(
    candidates: Sequence[CandidateObservation],
    *,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
    layout: PdfLayout,
) -> BindingOutcome:
    """Attach a sidecar to every candidate with a scope. In place, sidecar only."""

    entries = inventory_entries(inventory)
    index = _PaperIndex.build(layout)
    outcome = BindingOutcome(Counter(), Counter(), Counter())
    for candidate in candidates:
        if candidate.scope is None:
            continue
        binding = bind_candidate(
            candidate, entries=entries, index=index, inventory_sha256=inventory_sha256
        )
        candidate.dataset_binding = binding
        outcome.classes[binding.binding_class.value] += 1
        outcome.rules[binding.rule_id] += 1
        if binding.split_source is not None:
            outcome.split_sources[binding.split_source.value] += 1
    return outcome
