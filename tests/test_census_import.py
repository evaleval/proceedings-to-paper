"""Census import: identifier safety, sealed reserve, determinism, and metadata isolation."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from proceedings_to_eee.census import (
    CensusImportError,
    build_corpus_spec,
    draw_reserve,
    find_previously_inspected,
    import_census,
    load_census_rows,
    slugify_paper_id,
    stratified_order,
)
from proceedings_to_eee.corpus import load_corpus

_COLUMNS = [
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
]


def _row(
    paper_id: str,
    *,
    year: int = 2024,
    venue: str = "acl",
    pdf_url: str | None = None,
    role: str = "builds_own_instrument",
    title: str | None = None,
) -> dict[str, str]:
    return {
        "paper_id": paper_id,
        "year": str(year),
        "venue": venue,
        "title": title or f"Paper {paper_id}",
        "pdf_url": (pdf_url if pdf_url is not None else f"https://aclanthology.org/{paper_id}.pdf"),
        "construct_claimed": "hate speech",
        "measurement_role": role,
        "confidence": "high",
        "instrument_mentions": "Perspective API",
        "reason": "Screening note that must never reach a provider.",
    }


def _write_census(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _census_of(count: int, *, start: int = 1) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(start, start + count):
        year = 2017 + (index % 8)
        role = (
            "builds_own_instrument",
            "applies_existing_instrument",
            "analyzes_existing_annotations",
        )[index % 3]
        rows.append(_row(f"{year}.acl-main.{index}", year=year, role=role))
    return rows


def test_slug_makes_census_identifiers_legal_paper_ids() -> None:
    assert slugify_paper_id("2020.acl-main.111") == "2020-acl-main-111"
    assert slugify_paper_id("D17-1117") == "d17-1117"
    assert slugify_paper_id("W19-36.16") == "w19-36-16"
    assert slugify_paper_id("  2021.EMNLP-main.7  ") == "2021-emnlp-main-7"


def test_rows_without_a_usable_source_are_excluded_with_typed_reasons(tmp_path: Path) -> None:
    census = _write_census(
        tmp_path / "census.csv",
        [
            _row("2024.acl-main.1"),
            _row("2024.acl-main.2", pdf_url=""),
            _row("https://doi.org/10.26615/x", pdf_url=""),
            _row("2024.acl-main.3", pdf_url="ftp://example.org/paper.pdf"),
            _row("...", pdf_url="https://aclanthology.org/dots.pdf"),
            _row("2024.acl-main.4", year=0),
        ],
    )
    rows, exclusions = load_census_rows(census)

    assert [row.slug for row in rows] == ["2024-acl-main-1", "2024-acl-main-4"]
    reasons = {item.paper_id: item.reason for item in exclusions}
    assert reasons["2024.acl-main.2"] == "missing_pdf_url"
    assert reasons["https://doi.org/10.26615/x"] == "missing_pdf_url"
    assert reasons["2024.acl-main.3"] == "pdf_url_not_absolute_http"
    assert reasons["..."] == "unusable_paper_id"


def test_duplicate_derived_identifiers_are_excluded_not_merged(tmp_path: Path) -> None:
    census = _write_census(
        tmp_path / "census.csv",
        [
            _row("2024.acl-main.7"),
            _row("2024-acl-main-7", pdf_url="https://aclanthology.org/other.pdf"),
        ],
    )
    rows, exclusions = load_census_rows(census)

    assert len(rows) == 1
    assert exclusions[0].reason == "duplicate_derived_paper_id"
    assert "2024.acl-main.7" in exclusions[0].detail


def test_duplicate_titles_are_kept_because_they_are_distinct_papers(tmp_path: Path) -> None:
    census = _write_census(
        tmp_path / "census.csv",
        [
            _row("2024.acl-main.8", title="Shared Title"),
            _row("2024.acl-main.9", title="Shared Title"),
        ],
    )
    rows, exclusions = load_census_rows(census)

    assert len(rows) == 2
    assert exclusions == []


def test_order_is_deterministic_under_the_seed_and_changes_with_it(tmp_path: Path) -> None:
    census = _write_census(tmp_path / "census.csv", _census_of(90))
    rows, _ = load_census_rows(census)

    first = [row.slug for row in stratified_order(rows, seed="20260903", purpose="order")]
    again = [row.slug for row in stratified_order(rows, seed="20260903", purpose="order")]
    other = [row.slug for row in stratified_order(rows, seed="different", purpose="order")]

    assert first == again
    assert sorted(first) == sorted(other)
    assert first != other


def test_every_order_prefix_is_representative_of_the_strata(tmp_path: Path) -> None:
    census = _write_census(tmp_path / "census.csv", _census_of(300))
    rows, _ = load_census_rows(census)
    ordered = stratified_order(rows, seed="20260903", purpose="order")

    overall = Counter(row.metadata["measurement_role"] for row in rows)
    for cut in (30, 90, 150):
        prefix = Counter(row.metadata["measurement_role"] for row in ordered[:cut])
        for role, total in overall.items():
            expected = total / len(rows) * cut
            assert abs(prefix[role] - expected) <= 2


def test_reserve_is_disjoint_stratified_and_excludes_inspected_papers(tmp_path: Path) -> None:
    census = _write_census(tmp_path / "census.csv", _census_of(120))
    rows, _ = load_census_rows(census)

    blocked = {rows[0].slug, rows[5].slug}
    reserve = draw_reserve(rows, seed="20260903", size=12, ineligible_slugs=blocked)

    assert len(reserve) == 12
    assert len({row.slug for row in reserve}) == 12
    assert blocked.isdisjoint({row.slug for row in reserve})
    assert len({row.stratum for row in reserve}) > 1


def test_reserve_larger_than_the_census_is_refused(tmp_path: Path) -> None:
    census = _write_census(tmp_path / "census.csv", _census_of(5))
    rows, _ = load_census_rows(census)

    with pytest.raises(CensusImportError, match="exceeds"):
        draw_reserve(rows, seed="20260903", size=6)


def test_previously_inspected_papers_are_found_by_url_and_identifier(tmp_path: Path) -> None:
    census = _write_census(
        tmp_path / "census.csv",
        [_row("2023.emnlp-main.472"), _row("2023.findings-emnlp.663"), _row("2024.acl-main.5")],
    )
    rows, _ = load_census_rows(census)
    earlier = tmp_path / "holdout-10.yaml"
    earlier.write_text(
        yaml.safe_dump(
            {"papers": [{"pdf_url": "https://aclanthology.org/2023.emnlp-main.472.pdf"}]}
        ),
        encoding="utf-8",
    )

    hits = find_previously_inspected(
        rows, corpus_paths=[earlier], extra_paper_ids=["2023.findings-emnlp.663"]
    )

    assert {hit["paper_id"] for hit in hits} == {
        "2023.emnlp-main.472",
        "2023.findings-emnlp.663",
    }
    assert "holdout-10.yaml" in next(h["reason"] for h in hits if h["paper_id"].endswith("472"))


def test_generated_corpus_loads_and_carries_no_census_metadata(tmp_path: Path) -> None:
    census = _write_census(tmp_path / "census.csv", _census_of(20))
    rows, _ = load_census_rows(census)
    corpus = build_corpus_spec(rows, corpus_id="census-test", description="d")

    path = tmp_path / "census-test.yaml"
    path.write_text(
        yaml.safe_dump(corpus.model_dump(mode="json", exclude_none=True), sort_keys=False),
        encoding="utf-8",
    )
    reloaded = load_corpus(path)

    assert reloaded.evaluation_split == "development"
    assert len(reloaded.papers) == 20
    text = path.read_text(encoding="utf-8")
    for leaked in ("hate speech", "builds_own_instrument", "Perspective API", "Screening note"):
        assert leaked not in text


def test_import_writes_every_artifact_and_holds_the_reserve_back(tmp_path: Path) -> None:
    census = _write_census(tmp_path / "census.csv", _census_of(120))
    output = tmp_path / "run"
    corpus_dir = tmp_path / "corpora"

    summary = import_census(
        csv_path=census,
        output_dir=output,
        corpus_dir=corpus_dir,
        corpus_id="census-test",
        seed="20260903",
        reserve_size=10,
        shard_size=40,
        as_of="2026-09-03",
    )

    assert summary["rows_usable"] == 120
    assert summary["reserve_size"] == 10
    assert summary["ordered"] == 110

    reserve = json.loads((output / "census-reserve-sealed.json").read_text())
    order = json.loads((output / "census-order.json").read_text())
    reserve_slugs = {paper["slug"] for paper in reserve["papers"]}

    assert len(order["slugs"]) == 110
    assert reserve_slugs.isdisjoint(order["slugs"])

    corpus = load_corpus(corpus_dir / "census-test.yaml")
    assert len(corpus.papers) == 110
    assert reserve_slugs.isdisjoint({paper.paper_id for paper in corpus.papers})

    shards = sorted(corpus_dir.glob("census-test.shard-*.yaml"))
    assert len(shards) == 3
    sharded = [paper.paper_id for shard in shards for paper in load_corpus(shard).papers]
    assert sharded == order["slugs"]

    provenance = json.loads((output / "census-provenance.json").read_text())
    assert provenance["producer"] == "unknown"
    assert provenance["validation_status"] == "none"
    assert provenance["imported_as_of"] == "2026-09-03"

    identifiers = json.loads((output / "census-id-map.json").read_text())
    assert identifiers["slug_to_census_paper_id"][order["slugs"][0]].count(".") >= 1


def test_import_is_byte_identical_when_repeated(tmp_path: Path) -> None:
    census = _write_census(tmp_path / "census.csv", _census_of(60))
    digests: list[tuple[str, str]] = []
    for run in ("a", "b"):
        output = tmp_path / run
        summary = import_census(
            csv_path=census,
            output_dir=output,
            corpus_dir=tmp_path / f"corpora-{run}",
            corpus_id="census-test",
            seed="20260903",
            reserve_size=6,
            shard_size=25,
            as_of="2026-09-03",
        )
        digests.append((summary["reserve_sha256"], summary["order_sha256"]))
    assert digests[0] == digests[1]


def test_census_missing_required_columns_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("paper_id,year\nX,2024\n", encoding="utf-8")

    with pytest.raises(CensusImportError, match="missing required columns"):
        load_census_rows(path)
