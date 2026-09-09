"""Import a topical paper census into a corpus, a sealed reserve, and a seeded order.

The census is an external CSV of candidate papers. Its classification columns are
unvalidated screening metadata of unknown producer: they never enter a prompt and they
never gate anything here. This module only decides which papers are processable, which
are held back unseen, and in which order the rest are run.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from proceedings_to_eee.corpus import CorpusSpec, PaperSpec
from proceedings_to_eee.io import sha256_file, write_json

CENSUS_SCHEMA_VERSION = "paper-census/0.1"
DEFAULT_PERSPECTIVE_ROLE = "Primary paper to inspect for reported evaluation results."
DEFAULT_MAX_RESULT_PAGES = 8

# PaperSpec.paper_id accepts only lowercase alphanumeric segments joined by hyphens, so
# a census identifier such as `2020.acl-main.111` or `D17-1117` cannot be used directly.
_PAPER_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_REQUIRED_COLUMNS = (
    "paper_id",
    "year",
    "venue",
    "title",
    "pdf_url",
    "construct_claimed",
    "measurement_role",
    "confidence",
    "instrument_mentions",
    "reason",
)
_METADATA_COLUMNS = (
    "construct_claimed",
    "measurement_role",
    "confidence",
    "instrument_mentions",
    "reason",
)


class CensusImportError(ValueError):
    """The census file cannot be read as a paper census."""


@dataclass(frozen=True)
class CensusRow:
    """One census record that survived structural validation."""

    paper_id: str
    slug: str
    year: int
    venue: str
    title: str
    pdf_url: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def stratum(self) -> tuple[int, str]:
        return (self.year, self.metadata.get("measurement_role", ""))


@dataclass(frozen=True)
class CensusExclusion:
    """A census record that cannot become a corpus paper, with a typed reason."""

    paper_id: str
    reason: str
    detail: str = ""


def slugify_paper_id(paper_id: str) -> str:
    """Derive a PaperSpec-legal identifier from an arbitrary census identifier."""

    lowered = paper_id.strip().lower()
    collapsed = re.sub(r"[^a-z0-9]+", "-", lowered)
    return collapsed.strip("-")


def _selection_key(seed: str, purpose: str, paper_id: str) -> str:
    payload = f"{seed}|{purpose}|{paper_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def load_census_rows(
    csv_path: Path,
) -> tuple[list[CensusRow], list[CensusExclusion]]:
    """Read the census, returning processable rows and typed exclusions.

    A row is excluded when it has no PDF URL, when its identifier cannot produce a legal
    paper id, or when that identifier collides with one already accepted.
    """

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in _REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise CensusImportError(f"census is missing required columns: {sorted(missing)}")
        records = list(reader)

    rows: list[CensusRow] = []
    exclusions: list[CensusExclusion] = []
    seen_slugs: dict[str, str] = {}
    seen_census_ids: set[str] = set()

    for record in records:
        paper_id = (record.get("paper_id") or "").strip()
        if not paper_id:
            exclusions.append(CensusExclusion(paper_id="", reason="missing_paper_id"))
            continue
        if paper_id in seen_census_ids:
            exclusions.append(
                CensusExclusion(paper_id=paper_id, reason="duplicate_census_paper_id")
            )
            continue
        seen_census_ids.add(paper_id)

        pdf_url = (record.get("pdf_url") or "").strip()
        if not pdf_url:
            exclusions.append(CensusExclusion(paper_id=paper_id, reason="missing_pdf_url"))
            continue
        if not pdf_url.lower().startswith(("http://", "https://")):
            exclusions.append(
                CensusExclusion(
                    paper_id=paper_id,
                    reason="pdf_url_not_absolute_http",
                    detail=pdf_url[:120],
                )
            )
            continue

        slug = slugify_paper_id(paper_id)
        if not _PAPER_ID_PATTERN.match(slug):
            exclusions.append(
                CensusExclusion(paper_id=paper_id, reason="unusable_paper_id", detail=slug[:120])
            )
            continue
        if slug in seen_slugs:
            exclusions.append(
                CensusExclusion(
                    paper_id=paper_id,
                    reason="duplicate_derived_paper_id",
                    detail=f"collides with {seen_slugs[slug]}",
                )
            )
            continue
        seen_slugs[slug] = paper_id

        year_text = (record.get("year") or "").strip()
        try:
            year = int(year_text)
        except ValueError:
            exclusions.append(
                CensusExclusion(paper_id=paper_id, reason="unreadable_year", detail=year_text[:40])
            )
            continue

        rows.append(
            CensusRow(
                paper_id=paper_id,
                slug=slug,
                year=year,
                venue=(record.get("venue") or "").strip(),
                title=(record.get("title") or "").strip(),
                pdf_url=pdf_url,
                metadata={
                    column: (record.get(column) or "").strip() for column in _METADATA_COLUMNS
                },
            )
        )

    return rows, exclusions


def stratified_order(rows: Sequence[CensusRow], *, seed: str, purpose: str) -> list[CensusRow]:
    """Order rows so that every prefix is representative of the strata.

    Papers are grouped by (year, measurement_role), shuffled deterministically inside
    each stratum, then interleaved by their within-stratum position so that each
    stratum's share stays close to its overall share at any cut point.
    """

    strata: dict[tuple[int, str], list[CensusRow]] = {}
    for row in rows:
        strata.setdefault(row.stratum, []).append(row)

    decorated: list[tuple[float, str, CensusRow]] = []
    for stratum, members in strata.items():
        ordered = sorted(members, key=lambda row: _selection_key(seed, purpose, row.paper_id))
        size = len(ordered)
        for index, row in enumerate(ordered):
            fraction = (index + 0.5) / size
            decorated.append((fraction, _selection_key(seed, purpose, row.paper_id), row))
        del stratum

    decorated.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in decorated]


def draw_reserve(
    rows: Sequence[CensusRow],
    *,
    seed: str,
    size: int,
    ineligible_slugs: Iterable[str] = (),
) -> list[CensusRow]:
    """Draw a stratified metadata-only reserve that no later phase may open."""

    if size < 0:
        raise CensusImportError("reserve size must not be negative")
    blocked = set(ineligible_slugs)
    eligible = [row for row in rows if row.slug not in blocked]
    if size > len(eligible):
        raise CensusImportError(
            f"reserve of {size} exceeds {len(eligible)} eligible papers",
        )
    return stratified_order(eligible, seed=seed, purpose="reserve")[:size]


def find_previously_inspected(
    rows: Sequence[CensusRow],
    *,
    corpus_paths: Iterable[Path] = (),
    extra_paper_ids: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Flag census papers already opened by earlier corpora, by URL or identifier."""

    known_urls: dict[str, str] = {}
    for corpus_path in corpus_paths:
        if not corpus_path.is_file():
            continue
        try:
            payload = yaml.safe_load(corpus_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        for paper in payload.get("papers", []) or []:
            if not isinstance(paper, dict):
                continue
            url = str(paper.get("pdf_url") or "").strip()
            if url:
                known_urls[url.rstrip("/")] = corpus_path.name

    extra = {identifier.strip() for identifier in extra_paper_ids if identifier.strip()}
    hits: list[dict[str, str]] = []
    for row in rows:
        source = known_urls.get(row.pdf_url.rstrip("/"))
        if source is not None:
            hits.append(
                {"paper_id": row.paper_id, "slug": row.slug, "reason": f"pdf_url in {source}"}
            )
        elif row.paper_id in extra:
            hits.append(
                {
                    "paper_id": row.paper_id,
                    "slug": row.slug,
                    "reason": "listed as previously inspected or reserved for replacement",
                }
            )
    return hits


def build_corpus_spec(
    rows: Sequence[CensusRow],
    *,
    corpus_id: str,
    description: str,
    max_result_pages: int = DEFAULT_MAX_RESULT_PAGES,
) -> CorpusSpec:
    """Project census rows onto the existing corpus contract, metadata excluded."""

    papers = [
        PaperSpec(
            paper_id=row.slug,
            title=row.title or row.paper_id,
            year=row.year,
            venue=row.venue or "unknown",
            pdf_url=row.pdf_url,
            perspective_role=DEFAULT_PERSPECTIVE_ROLE,
            max_result_pages=max_result_pages,
        )
        for row in rows
    ]
    return CorpusSpec(
        corpus_id=corpus_id,
        evaluation_split="development",
        description=description,
        papers=papers,
    )


def _corpus_yaml_bytes(corpus: CorpusSpec) -> bytes:
    payload = corpus.model_dump(mode="json", exclude_none=True)
    payload["papers"] = [
        {key: value for key, value in paper.items() if value not in ([], None)}
        for paper in payload["papers"]
    ]
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).encode("utf-8")


def _write_corpus(path: Path, corpus: CorpusSpec) -> str:
    from proceedings_to_eee.io import atomic_write_bytes, sha256_bytes

    content = _corpus_yaml_bytes(corpus)
    atomic_write_bytes(path, content)
    return sha256_bytes(content)


def import_census(
    *,
    csv_path: Path,
    output_dir: Path,
    corpus_dir: Path,
    corpus_id: str,
    seed: str,
    reserve_size: int,
    shard_size: int,
    as_of: str,
    known_corpus_paths: Iterable[Path] = (),
    known_paper_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Write every phase-1 artifact and return a summary with exact counts."""

    if shard_size < 1:
        raise CensusImportError("shard size must be a positive integer")

    rows, exclusions = load_census_rows(csv_path)
    if not rows:
        raise CensusImportError("census contains no processable papers")

    inspected = find_previously_inspected(
        rows, corpus_paths=known_corpus_paths, extra_paper_ids=known_paper_ids
    )
    inspected_slugs = {hit["slug"] for hit in inspected}

    reserve = draw_reserve(rows, seed=seed, size=reserve_size, ineligible_slugs=inspected_slugs)
    reserve_slugs = {row.slug for row in reserve}
    runnable = [row for row in rows if row.slug not in reserve_slugs]
    ordered = stratified_order(runnable, seed=seed, purpose="order")

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    census_sha256 = sha256_file(csv_path)

    reserve_payload = {
        "schema_version": CENSUS_SCHEMA_VERSION,
        "purpose": (
            "Metadata-only reserve held back from every freeze, run, audit and atlas so a "
            "future sealed unseen evaluation remains possible. Do not open these papers."
        ),
        "seed": seed,
        "size": len(reserve),
        "census_sha256": census_sha256,
        "papers": [
            {
                "paper_id": row.paper_id,
                "slug": row.slug,
                "title": row.title,
                "pdf_url": row.pdf_url,
                "year": row.year,
                "venue": row.venue,
            }
            for row in reserve
        ],
    }
    reserve_sha256 = write_json(output_dir / "census-reserve-sealed.json", reserve_payload)

    order_sha256 = write_json(
        output_dir / "census-order.json",
        {
            "schema_version": CENSUS_SCHEMA_VERSION,
            "seed": seed,
            "count": len(ordered),
            "basis": "stratified by (year, measurement_role); every prefix is representative",
            "slugs": [row.slug for row in ordered],
        },
    )

    write_json(
        output_dir / "census-exclusions.json",
        {
            "schema_version": CENSUS_SCHEMA_VERSION,
            "count": len(exclusions),
            "excluded": [
                {
                    "paper_id": item.paper_id,
                    "reason": item.reason,
                    **({"detail": item.detail} if item.detail else {}),
                }
                for item in exclusions
            ],
        },
    )

    write_json(
        output_dir / "census-id-map.json",
        {
            "schema_version": CENSUS_SCHEMA_VERSION,
            "note": (
                "Corpus paper ids are slugs derived from census identifiers, which contain "
                "characters the corpus contract forbids. Join the atlas back through this map."
            ),
            "slug_to_census_paper_id": {row.slug: row.paper_id for row in rows},
        },
    )

    write_json(
        output_dir / "census-metadata.json",
        {
            "schema_version": CENSUS_SCHEMA_VERSION,
            "warning": (
                "Unvalidated screening metadata. Never sent to a provider, never used as a "
                "gate, and not ground truth. See census-provenance.json."
            ),
            "by_slug": {
                row.slug: {
                    "census_paper_id": row.paper_id,
                    "census_year": row.year,
                    "census_venue": row.venue,
                    "census_title": row.title,
                    **{f"census_{column}": row.metadata[column] for column in _METADATA_COLUMNS},
                }
                for row in rows
            },
        },
    )

    write_json(
        output_dir / "census-provenance.json",
        {
            "schema_version": CENSUS_SCHEMA_VERSION,
            "source_path_basename": csv_path.name,
            "source_sha256": census_sha256,
            "rows_in_file": len(rows) + len(exclusions),
            "rows_usable": len(rows),
            "producer": "unknown",
            "producer_note": (
                "The classification columns read as screening labels but no record on disk "
                "names who or what produced them. Aris must fill this in before any writing "
                "that relies on them."
            ),
            "validation_status": "none",
            "imported_as_of": as_of,
        },
    )

    write_json(
        output_dir / "census-previously-inspected.json",
        {
            "schema_version": CENSUS_SCHEMA_VERSION,
            "count": len(inspected),
            "basis": "pdf_url match against earlier corpora, plus explicitly listed identifiers",
            "papers": inspected,
        },
    )

    by_slug = {row.slug: row for row in runnable}
    ordered_rows = [by_slug[slug] for slug in (row.slug for row in ordered)]
    description = (
        f"Topical paper census imported from {csv_path.name} "
        f"(sha256 {census_sha256[:12]}). Development split; classification metadata is held "
        "outside this file and never enters a prompt. "
        f"{len(reserve)} papers are held back in census-reserve-sealed.json."
    )
    corpus = build_corpus_spec(ordered_rows, corpus_id=corpus_id, description=description)
    corpus_path = corpus_dir / f"{corpus_id}.yaml"
    corpus_sha256 = _write_corpus(corpus_path, corpus)

    shard_paths: list[str] = []
    for index in range(0, len(ordered_rows), shard_size):
        shard_number = index // shard_size + 1
        shard_rows = ordered_rows[index : index + shard_size]
        span = f"{index + 1}-{index + len(shard_rows)}"
        shard = build_corpus_spec(
            shard_rows,
            corpus_id=f"{corpus_id}-shard-{shard_number:02d}",
            description=(f"{description} Shard {shard_number}, papers {span} of the seeded order."),
        )
        shard_path = corpus_dir / f"{corpus_id}.shard-{shard_number:02d}.yaml"
        _write_corpus(shard_path, shard)
        shard_paths.append(shard_path.name)

    return {
        "census_sha256": census_sha256,
        "rows_usable": len(rows),
        "excluded": len(exclusions),
        "exclusion_reasons": sorted({item.reason for item in exclusions}),
        "reserve_size": len(reserve),
        "reserve_sha256": reserve_sha256,
        "previously_inspected": len(inspected),
        "ordered": len(ordered),
        "order_sha256": order_sha256,
        "corpus_path": str(corpus_path),
        "corpus_sha256": corpus_sha256,
        "shards": shard_paths,
    }
