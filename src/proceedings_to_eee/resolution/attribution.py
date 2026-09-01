"""Deterministic provenance attribution over the region index.

The mechanism rests on one committed invariant:

    A decisive attribution verdict requires a ROW-SCOPED signal. Caption-, section- and
    paper-scoped signals may only abstain, never decide.

A results table containing both an external leaderboard comparator and the paper's own
system is the proof that the invariant is necessary rather than tidy. Its caption carries
both foreign-source and first-party language. Any rule keyed on the caption demotes both
rows; only the row label separates them.

Nothing here knows which system a paper is "about". A paper may report primary results
for many systems at once, so no rule may treat one entity as the target and the rest as
distractors. Cues attach to attributing language, never to identity.

The mechanism is demote-only. It can move a candidate out of the export gate and into
review; it can never move one in. In particular, v0 never emits ``PAPER_PRODUCED``:
absence of an external cue is not positive evidence that the current paper produced a
number.

Only row-scoped decisive cues and structural abstentions change state. Caption cues and
bare citation brackets are recorded on the verdict and never gate, because they can
describe the table or dataset without identifying who produced a particular row.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from proceedings_to_eee.domain.attribution import (
    AttributionCue,
    AttributionState,
    AttributionVerdict,
)
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.extraction.region_index import (
    PageRegionIndex,
    QuoteLocation,
    Region,
    RegionKind,
    build_region_index,
    locate_quote_in_index,
)
from proceedings_to_eee.resources import DEFAULT_ATTRIBUTION_LEXICON_PATH

DEFAULT_LEXICON_PATH = DEFAULT_ATTRIBUTION_LEXICON_PATH


@dataclass(frozen=True)
class _Cue:
    cue_id: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class AttributionLexicon:
    """Cue set loaded from data, identified by the hash of the file it came from."""

    lexicon_id: str
    sha256: str
    decisive_row: tuple[_Cue, ...]
    abstaining_row: tuple[_Cue, ...]
    abstaining_caption: tuple[_Cue, ...]

    def decisive_matches(self, text: str | None) -> list[AttributionCue]:
        return _matches(self.decisive_row, text, "row_label")

    def abstaining_row_matches(self, text: str | None) -> list[AttributionCue]:
        return _matches(self.abstaining_row, text, "row_label")

    def caption_matches(self, text: str | None) -> list[AttributionCue]:
        return _matches(self.abstaining_caption, text, "caption")


def _matches(cues: tuple[_Cue, ...], text: str | None, scope: str) -> list[AttributionCue]:
    if not text:
        return []
    found: list[AttributionCue] = []
    for cue in cues:
        match = cue.pattern.search(text)
        if match:
            found.append(
                AttributionCue(cue_id=cue.cue_id, scope=scope, matched_text=match.group(0))
            )
    return found


def _compile(entries: list[dict[str, str]] | None) -> tuple[_Cue, ...]:
    return tuple(
        _Cue(cue_id=entry["id"], pattern=re.compile(entry["pattern"], re.IGNORECASE))
        for entry in entries or []
    )


@lru_cache(maxsize=8)
def load_lexicon(path: Path = DEFAULT_LEXICON_PATH) -> AttributionLexicon:
    """Load and hash the cue file. The hash is recorded on every verdict."""

    raw = path.read_bytes()
    document = yaml.safe_load(raw.decode("utf-8"))
    return AttributionLexicon(
        lexicon_id=document["lexicon_id"],
        sha256=hashlib.sha256(raw).hexdigest(),
        decisive_row=_compile(document.get("decisive_row_cues")),
        abstaining_row=_compile(document.get("abstaining_row_cues")),
        abstaining_caption=_compile(document.get("abstaining_caption_cues")),
    )


def _normalize(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").casefold()).strip()


def _agrees(left: str | None, right: str | None) -> bool:
    """Model-supplied anchor text may only cost throughput, never grant attribution."""

    a, b = _normalize(left), _normalize(right)
    if not a or not b:
        return True
    return a == b or a in b or b in a


def _labelled_data_rows(region: Region) -> list[str]:
    return [
        row.effective_row_label
        for row in region.rows
        if not row.is_header and row.effective_row_label
    ]


def _region_of(index: PageRegionIndex, region_id: str | None) -> Region | None:
    if region_id is None:
        return None
    return next((region for region in index.regions if region.region_id == region_id), None)


def _verdict(
    state: AttributionState,
    rule_id: str,
    lexicon: AttributionLexicon,
    *,
    cues: list[AttributionCue] | None = None,
    location: QuoteLocation | None = None,
    page: int | None = None,
    contrast: tuple[int, int] | None = None,
) -> AttributionVerdict:
    return AttributionVerdict(
        state=state,
        rule_id=rule_id,
        lexicon_id=lexicon.lexicon_id,
        lexicon_sha256=lexicon.sha256,
        cues=cues or [],
        region_id=location.region_id if location else None,
        page=page,
        row_label=location.row_label if location else None,
        table_label=location.table_label if location else None,
        contrast_rows_total=contrast[0] if contrast else None,
        contrast_rows_matched=contrast[1] if contrast else None,
    )


def attribute_candidate(
    candidate: CandidateObservation,
    page_index: PageRegionIndex | None,
    page: PageFragment | None,
    lexicon: AttributionLexicon,
) -> AttributionVerdict:
    """Decide one candidate's attribution without ever asserting paper production."""

    if page_index is None or page is None:
        return _verdict(AttributionState.UNRESOLVED, "no_page_index", lexicon)
    anchor = candidate.evidence[0]
    location = locate_quote_in_index(page_index, page, anchor.quote)
    if location is None:
        return _verdict(AttributionState.UNRESOLVED, "unlocatable", lexicon, page=page.page)
    if location.kind is not RegionKind.TABLE:
        # Prose attribution needs sentence-scoped reasoning and is out of scope for v0.
        return _verdict(
            AttributionState.UNRESOLVED,
            "not_a_table_row",
            lexicon,
            location=location,
            page=page.page,
        )

    region = _region_of(page_index, location.region_id)
    siblings = _labelled_data_rows(region) if region else []
    table_decisive = [label for label in siblings if lexicon.decisive_matches(label)]
    contrast = (len(siblings), len(table_decisive))

    row_label = location.row_label
    row_trusted = row_label is not None and _agrees(row_label, anchor.row)

    if row_trusted:
        decisive = lexicon.decisive_matches(row_label)
        if decisive:
            if table_decisive and len(table_decisive) == len(siblings) and len(siblings) > 1:
                # Every labelled row carries the marker. That is either a wholly external
                # table or shared vocabulary in the label column, and the two cannot be
                # told apart here. Abstain rather than demote a whole table on a guess.
                return _verdict(
                    AttributionState.UNRESOLVED,
                    "decisive_cue_without_contrast",
                    lexicon,
                    cues=decisive,
                    location=location,
                    page=page.page,
                    contrast=contrast,
                )
            return _verdict(
                AttributionState.EXTERNALLY_SOURCED,
                "row_scoped_foreign_cue",
                lexicon,
                cues=decisive,
                location=location,
                page=page.page,
                contrast=contrast,
            )

    if not row_trusted:
        # The candidate's physical row cannot be checked reliably. A foreign marker in a
        # sibling strengthens the concern, but lack of one cannot establish that no cue
        # exists on the candidate's row.
        return _verdict(
            AttributionState.UNRESOLVED,
            "table_marker_row_unresolved" if table_decisive else "row_unresolved",
            lexicon,
            location=location,
            page=page.page,
            contrast=contrast,
        )

    # Weak cues cannot decide external origin. They still make origin unresolved under
    # the canonical paper-produced-only policy. A leaderboard phrase in a caption may
    # govern both a paper-owned row and a comparator, while a citation marker can name a
    # dataset rather than a result source. Record such cues without treating them as proof.
    weak = lexicon.abstaining_row_matches(row_label) if row_trusted else []
    caption = lexicon.caption_matches(location.caption_text)
    if weak or caption:
        return _verdict(
            AttributionState.UNRESOLVED,
            "weak_cue_recorded",
            lexicon,
            cues=[*weak, *caption],
            location=location,
            page=page.page,
            contrast=contrast,
        )
    # This is the only NO_SIGNAL path: the quote was located in a table, its row was
    # structurally trusted, and the complete v0 cue check found nothing. It remains an
    # abstention, never evidence of PAPER_PRODUCED.
    return _verdict(
        AttributionState.NO_SIGNAL,
        "no_cue",
        lexicon,
        location=location,
        page=page.page,
        contrast=contrast,
    )


def attribute_candidates(
    candidates: list[CandidateObservation],
    layouts: dict[str, PdfLayout],
    lexicon: AttributionLexicon | None = None,
) -> dict[str, AttributionVerdict]:
    """Attribute a paper's candidates, keyed by observation identity."""

    lexicon = lexicon or load_lexicon()
    indexes: dict[str, dict[int, PageRegionIndex]] = {}
    pages: dict[str, dict[int, PageFragment]] = {}
    for source_id, layout in layouts.items():
        indexes[source_id] = build_region_index(layout)
        pages[source_id] = {fragment.page: fragment for fragment in layout.pages}
    verdicts: dict[str, AttributionVerdict] = {}
    for candidate in candidates:
        anchor = candidate.evidence[0]
        page_index = indexes.get(anchor.source_id, {}).get(anchor.page)
        page = pages.get(anchor.source_id, {}).get(anchor.page)
        key = candidate.observation_id or candidate.stable_id()
        verdicts[key] = attribute_candidate(candidate, page_index, page, lexicon)
    return verdicts
