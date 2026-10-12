"""Deterministic provenance attribution over the region index.

The mechanism rests on one committed invariant:

    A decisive attribution verdict requires a ROW-SCOPED signal. Header-, caption-,
    section- and paper-scoped signals may only abstain, never decide.

A results table containing both an external leaderboard comparator and the paper's own
system is the proof that the invariant is necessary rather than tidy. Its caption carries
both foreign-source and first-party language. Any rule keyed on the caption demotes both
rows; only the row label separates them.

Nothing here knows which system a paper is "about". A paper may report primary results
for many systems at once, so no rule may treat one entity as the target and the rest as
distractors. Cues attach to attributing language, never to identity. The paper-context
cues of lexicon v1 are the one deliberate exception, and they keep the invariant: a
title of the form ``<team> at <venue>`` lets the resolver record which rows of a ranked
table do not even name that team, and that record abstains like every other table-scoped
signal.

The mechanism is demote-only. It can move a candidate out of the export gate and into
review; it can never move one in. In particular, it never emits ``PAPER_PRODUCED``:
absence of an external cue is not positive evidence that the current paper produced a
number.

That is why canonical EEE is empty, and why a second, explicitly weaker policy exists.
``OriginExportPolicy.TIERED`` lets a composition run export a candidate on a named
weaker basis, which the record then carries. It changes nothing here: this resolver
still never emits ``PAPER_PRODUCED``, and a verdict is never promoted. See
``domain/attribution.py`` for the bases and ``census_recompose.py`` for the replay.

Only row-scoped decisive cues and structural abstentions change state. Header cells,
captions, bare citation brackets and the paper-context prior are recorded on the verdict
of every candidate anchored in the table, on every path, and never gate, because they
can describe the table or dataset without identifying who produced a particular row.
Each cue carries the scope it was read from (``row_label``, ``table_header``,
``caption`` or ``paper_context``), so a reader of the verdict can tell the one scope
that may decide from the three that only abstain.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from proceedings_to_eee.domain.attribution import (
    SELF_REFERENCE_RULE_ID,
    AttributionCue,
    AttributionState,
    AttributionVerdict,
    PaperContext,
    PaperKind,
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
from proceedings_to_eee.resources import (
    ATTRIBUTION_LEXICON_PATHS,
    DEFAULT_ATTRIBUTION_LEXICON_PATH,
)

DEFAULT_LEXICON_PATH = DEFAULT_ATTRIBUTION_LEXICON_PATH

#: Scope of a cue the paper-context prior produced. Such a cue is read from the frozen
#: title together with the table, never from the row alone, and it abstains like every
#: other table-scoped signal even when it names a row.
PAPER_CONTEXT_CUE_SCOPE = "paper_context"


@dataclass(frozen=True)
class _Cue:
    cue_id: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class _ContextCue:
    """A cue that fires from the paper kind, on rows of a table that carries a table cue.

    `exempt` makes the cue row-conditional: it then needs a trusted row label, and a
    label that matches the pattern or names the title's team is left alone. Without a
    pattern the cue applies to every row of the table.
    """

    cue_id: str
    paper_kind: PaperKind
    exempt: re.Pattern[str] | None


@dataclass(frozen=True)
class AttributionLexicon:
    """Cue set loaded from data, identified by the hash of the file it came from."""

    lexicon_id: str
    sha256: str
    decisive_row: tuple[_Cue, ...]
    abstaining_row: tuple[_Cue, ...]
    abstaining_caption: tuple[_Cue, ...]
    abstaining_table_header: tuple[_Cue, ...] = ()
    #: Row labels that name the paper itself. Row-scoped: such a label exempts its own
    #: row from table- and paper-scoped cues, which stay recorded, and decides nothing.
    self_reference_row: tuple[_Cue, ...] = ()
    abstaining_paper_context: tuple[_ContextCue, ...] = ()
    #: Header or caption cues that say the table ranks or lists other parties'
    #: submissions. Only these let a paper-context cue fire.
    table_level_cue_ids: frozenset[str] = frozenset()

    def decisive_matches(self, text: str | None) -> list[AttributionCue]:
        return _matches(self.decisive_row, text, "row_label")

    def abstaining_row_matches(self, text: str | None) -> list[AttributionCue]:
        return _matches(self.abstaining_row, text, "row_label")

    def self_reference_matches(self, text: str | None) -> list[AttributionCue]:
        return _matches(self.self_reference_row, text, "row_label")

    def caption_matches(self, text: str | None) -> list[AttributionCue]:
        return _matches(self.abstaining_caption, text, "caption")

    def header_matches(self, cells: Iterable[str]) -> list[AttributionCue]:
        found: list[AttributionCue] = []
        for cell in cells:
            found.extend(_matches(self.abstaining_table_header, cell.strip(), "table_header"))
        return found

    def context_cues(self, paper_kind: PaperKind) -> tuple[_ContextCue, ...]:
        return tuple(cue for cue in self.abstaining_paper_context if cue.paper_kind is paper_kind)

    def has_table_level_cue(self, cues: Iterable[AttributionCue]) -> bool:
        return any(cue.cue_id in self.table_level_cue_ids for cue in cues)


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


def _compile_context(entries: list[dict[str, str]] | None) -> tuple[_ContextCue, ...]:
    return tuple(
        _ContextCue(
            cue_id=entry["id"],
            paper_kind=PaperKind(entry["paper_kind"]),
            exempt=(re.compile(entry["pattern"], re.IGNORECASE) if entry.get("pattern") else None),
        )
        for entry in entries or []
    )


@lru_cache(maxsize=8)
def load_lexicon(path: Path = DEFAULT_LEXICON_PATH) -> AttributionLexicon:
    """Load and hash the cue file. The hash is recorded on every verdict."""

    raw = path.read_bytes()
    document = yaml.safe_load(raw.decode("utf-8"))
    lexicon = AttributionLexicon(
        lexicon_id=document["lexicon_id"],
        sha256=hashlib.sha256(raw).hexdigest(),
        decisive_row=_compile(document.get("decisive_row_cues")),
        abstaining_row=_compile(document.get("abstaining_row_cues")),
        abstaining_caption=_compile(document.get("abstaining_caption_cues")),
        abstaining_table_header=_compile(document.get("abstaining_table_header_cues")),
        abstaining_paper_context=_compile_context(document.get("abstaining_paper_context_cues")),
        self_reference_row=_compile(document.get("self_reference_row_cues")),
        table_level_cue_ids=frozenset(document.get("table_level_cues") or ()),
    )
    table_scoped = {
        cue.cue_id for cue in (*lexicon.abstaining_table_header, *lexicon.abstaining_caption)
    }
    unknown = sorted(lexicon.table_level_cue_ids - table_scoped)
    if unknown:
        raise ValueError(
            f"{lexicon.lexicon_id}: table_level_cues names no header or caption cue: {unknown}"
        )
    return lexicon


def load_recorded_lexicon(lexicon_id: str | None, sha256: str | None) -> AttributionLexicon:
    """Load the shipped lexicon a finished run recorded on its verdicts.

    A replay must run under the lexicon the run used, not under whatever is the default
    now, or the verdicts it reproduces differ from the ones on file. The id must name a
    shipped file and the hash must still match it; anything else is refused rather than
    silently replaced.
    """

    if lexicon_id is None or sha256 is None:
        raise ValueError("recorded attribution lexicon lacks its id or sha256")
    path = ATTRIBUTION_LEXICON_PATHS.get(lexicon_id)
    if path is None:
        raise ValueError(f"recorded attribution lexicon {lexicon_id} is not shipped")
    lexicon = load_lexicon(path)
    if lexicon.sha256 != sha256:
        raise ValueError(f"recorded attribution lexicon {lexicon_id} differs from the shipped copy")
    return lexicon


def _normalize(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").casefold()).strip()


def _agrees(left: str | None, right: str | None) -> bool:
    """Model-supplied anchor text may only cost throughput, never grant attribution."""

    a, b = _normalize(left), _normalize(right)
    if not a or not b:
        return True
    return a == b or a in b or b in a


# ----------------------------------------------------------------------------------
# Paper context from the frozen manifest title
# ----------------------------------------------------------------------------------

_OVERVIEW_TITLE = re.compile(
    r"^(?:(?:the|an?)\s+)?overview\s+of\b|^findings\s+of\s+the\b.*\b(?:shared[- ]task|task\s*\d)",
    re.IGNORECASE,
)
_TEAM_VENUE_SEPARATOR = re.compile(r"\s+at\s+|\s*@\s*")
_VENUE_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_VENUE_WORD = re.compile(r"\b(?:shared[- ]task|task\s*\d|challenge|competition)\b", re.IGNORECASE)
_ARTICLE = frozenset({"the", "a", "an"})
_MAX_TEAM_TOKENS = 5


def _dated_venue(text: str) -> bool:
    """A year or a task word: SemEval-2020 Task 12, GermEval 2024 Shared Task 1."""

    return bool(_VENUE_YEAR.search(text) or _VENUE_WORD.search(text))


def _looks_like_venue(text: str) -> bool:
    """A dated venue, or a proper-noun first token such as MultiPride or CASE."""

    if _dated_venue(text):
        return True
    tokens = [token for token in text.split() if token.casefold() not in _ARTICLE]
    if not tokens:
        return False
    first = tokens[0]
    return any(character.isdigit() for character in first) or any(
        character.isupper() for character in first[1:]
    )


def _looks_like_handle(text: str) -> bool:
    """A team handle rather than a plain word: KIT-TIP-NLP, Dll5143A, byteSizedLLM."""

    return any(re.search(r"[A-Z0-9_-]", token[1:]) for token in text.split())


def paper_context_from_title(title: str | None) -> PaperContext:
    """Derive the paper kind and team name from a title, deterministically.

    The team is the text before the last `at` or `@` of the title's first segment (the
    part before a colon) whose remainder looks like a venue. An undated venue such as
    `MultiPride` counts only when the team reads as a handle, so `Looking at BERT`
    names no team. Only `Overview of` and `Findings of the ... task` titles name an
    overview paper. Anything else is `other`.
    """

    text = (title or "").strip()
    if not text:
        return PaperContext(title=None)
    if _OVERVIEW_TITLE.search(text):
        return PaperContext(title=text, paper_kind=PaperKind.OVERVIEW_PAPER)
    segment = text.split(":", 1)[0].strip()
    for separator in reversed(list(_TEAM_VENUE_SEPARATOR.finditer(segment))):
        team = segment[: separator.start()].strip()
        venue = segment[separator.end() :].strip()
        if not team or not venue or len(team.split()) > _MAX_TEAM_TOKENS:
            continue
        if not _normalize(team) or not _looks_like_venue(venue):
            continue
        if not _dated_venue(venue) and not _looks_like_handle(team):
            continue
        return PaperContext(
            title=text, paper_kind=PaperKind.SHARED_TASK_SYSTEM_PAPER, team_name=team
        )
    return PaperContext(title=text)


def row_names_team(context: PaperContext | None, row_label: str | None) -> bool:
    """True when the normalised row label contains the team name as whole words."""

    if context is None or context.team_name is None:
        return False
    team = _normalize(context.team_name)
    if team.startswith("team "):
        team = team[len("team ") :]
    label = _normalize(row_label)
    if not team or not label:
        return False
    return f" {team} " in f" {label} "


def has_table_level_cue(verdict: AttributionVerdict, lexicon: AttributionLexicon) -> bool:
    """True when the table the verdict is anchored in carries a table-level cue."""

    return lexicon.has_table_level_cue(verdict.cues)


# ----------------------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------------------


def _labelled_data_rows(region: Region) -> list[str]:
    return [
        row.effective_row_label
        for row in region.rows
        if not row.is_header and row.effective_row_label
    ]


def _anchored_region(index: PageRegionIndex, location: QuoteLocation) -> Region | None:
    """The table the quote stands in, or None when the index cannot vouch for one.

    A quote in a full-width table on a two-column page is located against a rebuilt
    single-panel index whose region ids this index does not share, so the id alone is
    not trusted: the region must also be a table whose span covers the matched line.
    """

    region = next(
        (region for region in index.regions if region.region_id == location.region_id), None
    )
    if region is None or region.kind is not RegionKind.TABLE:
        return None
    if not region.span.start_line <= location.span.start_line <= region.span.end_line:
        return None
    return region


def _table_cues(
    lexicon: AttributionLexicon, region: Region | None, location: QuoteLocation
) -> list[AttributionCue]:
    """Header and caption cues: table-scoped, recorded for every row, never decisive."""

    header_cells = [
        cell for row in (region.rows if region else []) if row.is_header for cell in row.cells
    ]
    return [*lexicon.header_matches(header_cells), *lexicon.caption_matches(location.caption_text)]


def _context_cues(
    lexicon: AttributionLexicon,
    context: PaperContext | None,
    table_cues: list[AttributionCue],
    trusted_row_label: str | None,
) -> list[AttributionCue]:
    """Paper-context cues fire only inside a table that carries a table-level cue.

    A caption that merely says `reported` is not enough: the trigger is the lexicon's
    declared `table_level_cues`, the header and caption cues that say the table ranks
    or lists other parties' submissions.
    """

    if context is None or not lexicon.has_table_level_cue(table_cues):
        return []
    found: list[AttributionCue] = []
    for cue in lexicon.context_cues(context.paper_kind):
        if cue.exempt is None:
            found.append(
                AttributionCue(
                    cue_id=cue.cue_id,
                    scope=PAPER_CONTEXT_CUE_SCOPE,
                    matched_text=context.title or cue.cue_id,
                )
            )
            continue
        if trusted_row_label is None or row_names_team(context, trusted_row_label):
            continue
        if cue.exempt.search(_normalize(trusted_row_label)):
            continue
        found.append(
            AttributionCue(
                cue_id=cue.cue_id, scope=PAPER_CONTEXT_CUE_SCOPE, matched_text=trusted_row_label
            )
        )
    return found


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
    context: PaperContext | None = None,
) -> AttributionVerdict:
    """Decide one candidate's attribution without ever asserting paper production."""

    if page_index is None or page is None:
        return _verdict(AttributionState.UNRESOLVED, "no_page_index", lexicon)
    anchor = candidate.evidence[0]
    location = locate_quote_in_index(page_index, page, anchor.quote)
    if location is None:
        return _verdict(AttributionState.UNRESOLVED, "unlocatable", lexicon, page=page.page)
    if location.kind is not RegionKind.TABLE:
        # Prose attribution needs sentence-scoped reasoning and is out of scope.
        return _verdict(
            AttributionState.UNRESOLVED,
            "not_a_table_row",
            lexicon,
            location=location,
            page=page.page,
        )

    region = _anchored_region(page_index, location)
    siblings = _labelled_data_rows(region) if region else []
    table_decisive = [label for label in siblings if lexicon.decisive_matches(label)]
    contrast = (len(siblings), len(table_decisive))

    row_label = location.row_label
    row_trusted = row_label is not None and _agrees(row_label, anchor.row)

    # Table-scoped cues are recorded on every path below. They never change the state
    # logic: the state still follows the row, and only a row-scoped cue may decide.
    table_cues = _table_cues(lexicon, region, location)
    abstaining = [
        *table_cues,
        *_context_cues(lexicon, context, table_cues, row_label if row_trusted else None),
    ]

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
                    cues=[*decisive, *abstaining],
                    location=location,
                    page=page.page,
                    contrast=contrast,
                )
            return _verdict(
                AttributionState.EXTERNALLY_SOURCED,
                "row_scoped_foreign_cue",
                lexicon,
                cues=[*decisive, *abstaining],
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
            cues=abstaining,
            location=location,
            page=page.page,
            contrast=contrast,
        )

    # A label that names the paper itself is the one row-scoped signal in the paper's
    # favour. It decides nothing, but the table- and paper-scoped cues recorded beside
    # it describe the table, not this row, so they do not hold it back. A weak row cue
    # on the same label (a citation bracket) still does.
    self_reference = lexicon.self_reference_matches(row_label)
    if self_reference and not lexicon.abstaining_row_matches(row_label):
        return _verdict(
            AttributionState.NO_SIGNAL,
            SELF_REFERENCE_RULE_ID,
            lexicon,
            cues=[*self_reference, *abstaining],
            location=location,
            page=page.page,
            contrast=contrast,
        )
    # Weak cues cannot decide external origin. They still make origin unresolved under
    # the canonical paper-produced-only policy. A leaderboard phrase in a caption may
    # govern both a paper-owned row and a comparator, while a citation marker can name a
    # dataset rather than a result source. Record such cues without treating them as proof.
    weak = lexicon.abstaining_row_matches(row_label)
    if weak or abstaining:
        return _verdict(
            AttributionState.UNRESOLVED,
            "weak_cue_recorded",
            lexicon,
            cues=[*weak, *abstaining],
            location=location,
            page=page.page,
            contrast=contrast,
        )
    # This is the only NO_SIGNAL path: the quote was located in a table, its row was
    # structurally trusted, and the complete cue check found nothing. It remains an
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
    context: PaperContext | None = None,
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
        verdicts[key] = attribute_candidate(candidate, page_index, page, lexicon, context)
    return verdicts
