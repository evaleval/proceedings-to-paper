"""The dataset inventory call and its deterministic defence.

An entry stands only on a quote re-found on the frozen layout. Nothing here is a human
label, and the inventory never touches a candidate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.resolution.dataset_inventory import (
    INVENTORY_FILENAME,
    INVENTORY_PRIVATE_FILENAME,
    MAX_CONSECUTIVE_FAILURES,
    QUOTE_NORMALISATION,
    collect_inventory_input,
    entry_id,
    inventory_content_sha256,
    inventory_prompt,
    inventory_prompt_sha256,
    inventory_response_schema,
    inventory_schema_sha256,
    load_inventory,
    run_inventory,
    select_papers,
    verify_inventory,
)

SOURCE_ID = "src_inventory_fixture"

PAGE_ONE = """A Synthetic Study of Toxicity Detection

1   Introduction
We study toxicity in short messages.

2   Datasets
We evaluate on HateCheck (Example et al.), a suite of functional tests in English,
and on the OLID corpus. HateCheck has a test split of 3,728 cases.

3   Method
We fine-tune a classifier on the training data and report HC scores per functionality.
"""

PAGE_TWO = """4   Experimental Setup
All systems are trained on the OLID training split and tested on the OLID test set.

Table 1: Macro F1 on the HateCheck test split.

     System              Test
     PerspectiveAPI      0.81

5   Conclusion
The fixture ends here.
"""

HATECHECK_QUOTE = (
    "We evaluate on HateCheck (Example et al.), a suite of functional tests in English,"
)
OLID_QUOTE = "All systems are trained on the OLID training split and tested on the OLID test set."


def _page(text: str, page: int) -> PageFragment:
    return PageFragment(
        fragment_id=f"frag_{page}",
        source_id=SOURCE_ID,
        page=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_count=len(text),
        numeric_token_count=1,
        result_signal_score=1.0,
    )


def _layout() -> PdfLayout:
    return PdfLayout(
        source_id=SOURCE_ID,
        parser="fixture",
        parser_version="fixture-1",
        page_count=2,
        pages=[_page(PAGE_ONE, 1), _page(PAGE_TWO, 2)],
    )


def _paper(root: Path, slug: str = "inventory-fixture") -> Path:
    paper_dir = root / slug
    (paper_dir / "private").mkdir(parents=True, exist_ok=True)
    (paper_dir / "observations.jsonl").write_text(
        json.dumps({"observation_id": "obs_fixture", "paper_id": slug}) + "\n", encoding="utf-8"
    )
    (paper_dir / "private" / "layout.json").write_text(
        json.dumps(_layout().model_dump(mode="json")), encoding="utf-8"
    )
    (paper_dir / "source-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "source-manifest/0.1",
                "paper_id": slug,
                "title": "A Synthetic Study of Toxicity Detection",
                "sources": [
                    {
                        "source_id": SOURCE_ID,
                        "paper_id": slug,
                        "role": "paper",
                        "original_uri": "https://example.invalid/paper.pdf",
                        "resolved_uri": "https://example.invalid/paper.pdf",
                        "retrieved_at": "2026-09-04T00:00:00+00:00",
                        "sha256": "0" * 64,
                        "byte_size": 10,
                        "media_type": "application/pdf",
                        "cache_relpath": "data/sources/fixture.pdf",
                        "access_status": "available",
                        "license_disposition": "unknown",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return paper_dir


def _entry(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name_raw": "HateCheck",
        "aliases_in_paper": ["HC"],
        "kind": "dataset",
        "languages": ["English"],
        "splits": [{"name": "test", "size_reported": "3,728"}],
        "introduced_here": False,
        "evidence": [{"page": 1, "exact_quote": HATECHECK_QUOTE}],
    }
    payload.update(overrides)
    return payload


def _searchable() -> dict[int, list[str]]:
    source = collect_inventory_input(_paper_root_cache["dir"])
    assert source is not None
    return source.searchable


_paper_root_cache: dict[str, Path] = {}


@pytest.fixture(autouse=True)
def _paper_root(tmp_path: Path) -> None:
    _paper_root_cache["dir"] = _paper(tmp_path)


def test_the_input_takes_captions_and_matching_sections_only(tmp_path: Path) -> None:
    source = collect_inventory_input(tmp_path / "inventory-fixture")

    assert source is not None
    assert source.title == "A Synthetic Study of Toxicity Detection"
    assert [page for page, _ in source.captions] == [2]
    assert source.captions[0][1].startswith("Table 1:")
    titles = {title for _, title, _ in source.sections}
    assert titles == {"Datasets", "Experimental Setup"}
    joined = "\n".join(text for _, _, text in source.sections)
    assert HATECHECK_QUOTE in joined
    assert OLID_QUOTE in joined
    assert "We fine-tune a classifier" not in joined
    assert "We study toxicity" not in joined
    assert source.sections_truncated is False
    # Every page is searchable through its page text and each of its regions.
    assert source.searchable[1][0] == PAGE_ONE
    assert len(source.searchable[1]) > 1


def test_the_section_bound_cuts_sections_only_and_records_the_cut(tmp_path: Path) -> None:
    # Page 1's Datasets section alone exceeds the bound, so the cut lands before page 2.
    source = collect_inventory_input(tmp_path / "inventory-fixture", max_section_characters=60)

    assert source is not None
    assert source.sections_truncated is True
    assert source.section_characters <= 60
    assert {page for page, _, _ in source.sections} == {1}
    prompt = inventory_prompt(source)
    assert "cut at the configured character bound" in prompt
    hashes = source.hashes()
    assert hashes["sections_truncated"] is True
    assert set(hashes) >= {"title_sha256", "captions_sha256", "sections_sha256"}
    # Captions after the cut are still sent, and a quote from a later page still
    # verifies: the bound limits the section text, not what the model may cite.
    assert [page for page, _ in source.captions] == [2]
    assert "[page 2] Table 1: Macro F1 on the HateCheck test split." in prompt
    assert set(source.searchable) == {1, 2}
    outcome = verify_inventory(
        {"entries": [_entry(evidence=[{"page": 2, "exact_quote": OLID_QUOTE}])]},
        source.searchable,
    )
    assert len(outcome.entries) == 1
    assert outcome.dropped_unverified == 0


def test_the_prompt_carries_title_captions_and_page_labelled_sections(tmp_path: Path) -> None:
    source = collect_inventory_input(tmp_path / "inventory-fixture")
    assert source is not None

    prompt = inventory_prompt(source)

    assert "A Synthetic Study of Toxicity Detection" in prompt
    assert "[page 2] Table 1: Macro F1 on the HateCheck test split." in prompt
    assert "--- page 1, section: Datasets ---" in prompt
    assert "--- page 2, section: Experimental Setup ---" in prompt


def test_a_verified_entry_is_kept_with_hashed_anchors() -> None:
    outcome = verify_inventory({"entries": [_entry()]}, _searchable())

    assert outcome.proposed == 1
    assert outcome.dropped_unverified == 0
    [entry] = outcome.entries
    assert entry["entry_id"] == entry_id("HateCheck", "dataset")
    assert entry["aliases_in_paper"] == ["HC"]
    assert entry["languages"] == ["English"]
    assert entry["splits"] == [{"name": "test", "size_reported": "3,728"}]
    assert entry["anchors"] == [
        {
            "page": 1,
            "quote_sha256": hashlib.sha256(HATECHECK_QUOTE.encode()).hexdigest(),
            "verified": True,
        }
    ]
    assert "exact_quote" not in json.dumps(entry)
    assert HATECHECK_QUOTE not in json.dumps(entry)


@pytest.mark.parametrize(
    "evidence",
    [
        [{"page": 1, "exact_quote": "We evaluate on HateCheck, a suite of unit tests."}],
        [{"page": 2, "exact_quote": HATECHECK_QUOTE}],
        [],
    ],
    ids=["repaired_quote", "wrong_page", "no_evidence"],
)
def test_an_entry_without_a_verified_anchor_is_dropped_and_counted(
    evidence: list[dict[str, Any]],
) -> None:
    outcome = verify_inventory({"entries": [_entry(evidence=evidence)]}, _searchable())

    assert outcome.entries == []
    assert outcome.dropped_unverified == 1
    assert outcome.proposed == 1


def test_one_verified_anchor_keeps_the_entry_and_flags_the_other() -> None:
    outcome = verify_inventory(
        {
            "entries": [
                _entry(
                    evidence=[
                        {"page": 1, "exact_quote": "an invented sentence"},
                        {"page": 1, "exact_quote": HATECHECK_QUOTE},
                    ]
                )
            ]
        },
        _searchable(),
    )

    [entry] = outcome.entries
    assert [anchor["verified"] for anchor in entry["anchors"]] == [False, True]


def test_a_quote_whole_only_inside_a_layout_region_still_verifies() -> None:
    page_text = "left one          right one\nleft two          right two\n"
    region_text = "left one\nleft two"
    outcome = verify_inventory(
        {
            "entries": [
                _entry(
                    name_raw="left two",
                    aliases_in_paper=[],
                    evidence=[{"page": 1, "exact_quote": "left one left two"}],
                )
            ]
        },
        {1: [page_text, region_text]},
    )

    assert len(outcome.entries) == 1


def test_a_hyphenated_line_break_is_not_relaxed() -> None:
    # The layout breaks "HateCheck" across a line; the whitespace-only match fails and
    # the gate says so by naming its normalisation, rather than repairing the quote.
    page_text = "We evaluate on Hate-\nCheck, a suite of functional tests.\n"
    outcome = verify_inventory(
        {
            "entries": [
                _entry(evidence=[{"page": 1, "exact_quote": "We evaluate on HateCheck, a suite"}])
            ]
        },
        {1: [page_text]},
    )

    assert outcome.dropped_unverified == 1
    assert QUOTE_NORMALISATION == "whitespace_only"


def test_an_entry_whose_name_the_paper_never_prints_is_dropped_and_counted() -> None:
    # The quote is verbatim, but the model named the data something the paper does not.
    outcome = verify_inventory(
        {"entries": [_entry(name_raw="Invented Functional Suite")]}, _searchable()
    )

    assert outcome.entries == []
    assert outcome.dropped_unverified == 0
    assert outcome.dropped_unnamed == 1


@pytest.mark.parametrize(
    ("aliases", "kept", "dropped"),
    [
        (["HC", "Hate Check Suite"], ["HC"], 1),
        (["hatecheck", "olid"], ["hatecheck", "olid"], 0),
        (["  HC  ", "HC"], ["HC"], 0),
    ],
    ids=["invented_alias_removed", "case_is_ignored", "duplicates_collapse"],
)
def test_aliases_survive_only_when_the_paper_prints_them(
    aliases: list[str], kept: list[str], dropped: int
) -> None:
    outcome = verify_inventory({"entries": [_entry(aliases_in_paper=aliases)]}, _searchable())

    [entry] = outcome.entries
    assert entry["aliases_in_paper"] == kept
    assert outcome.dropped_aliases == dropped


def test_malformed_duplicate_and_unknown_kind_entries_are_dropped_not_repaired() -> None:
    outcome = verify_inventory(
        {
            "entries": [
                _entry(name_raw="   "),
                _entry(),
                _entry(aliases_in_paper=["HateCheck suite"]),
                _entry(
                    name_raw="OLID",
                    kind="not_a_kind",
                    evidence=[{"page": 2, "exact_quote": OLID_QUOTE}],
                ),
            ]
        },
        _searchable(),
    )

    # A kind outside the vocabulary is malformed, like a blank name; it is not coerced.
    assert outcome.dropped_malformed == 2
    assert outcome.dropped_duplicate == 1
    assert [entry["name_raw"] for entry in outcome.entries] == ["HateCheck"]


def test_entry_ids_are_stable_and_distinguish_kind() -> None:
    assert entry_id("HateCheck", "dataset") == entry_id("HateCheck", "dataset")
    assert entry_id("HateCheck", "dataset") != entry_id("HateCheck", "benchmark")
    assert entry_id("HateCheck", "dataset").startswith("ds_")


def test_a_run_writes_a_quote_free_gate_and_a_private_log_and_never_repeats(
    tmp_path: Path,
) -> None:
    from proceedings_to_eee.census_review import review_response_schema
    from proceedings_to_eee.extraction.llm_schema import provider_json_schema

    calls: list[str] = []

    def call(system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(user)
        return (
            {"entries": [_entry(), _entry(name_raw="Invented Corpus", evidence=[])]},
            {"cost_usd": 0.002, "request_sha256": "a" * 64},
        )

    first = run_inventory(run_root=tmp_path, model="fixture/model", call=call)
    second = run_inventory(run_root=tmp_path, model="fixture/model", call=call)

    assert first["inventoried"] == 1
    assert first["entries"] == 1
    assert first["dropped"] == {"duplicate": 0, "malformed": 0, "unnamed": 0, "unverified": 1}
    assert first["dropped_aliases"] == 0
    assert first["quote_normalisation"] == QUOTE_NORMALISATION
    assert second["inventoried"] == 0
    assert second["already_inventoried"] == 1
    assert len(calls) == 1
    paper_dir = tmp_path / "inventory-fixture"
    gate = load_inventory(paper_dir)
    assert gate is not None
    assert gate["inventory_prompt_sha256"] == inventory_prompt_sha256()
    assert gate["inventory_schema_sha256"] == inventory_schema_sha256()
    assert gate["model"] == "fixture/model"
    assert gate["dropped_unverified"] == 1
    assert gate["dropped_unnamed"] == 0
    assert gate["dropped_aliases"] == 0
    assert gate["quote_normalisation"] == QUOTE_NORMALISATION
    assert gate["telemetry"]["cost_usd"] == 0.002
    assert gate["input"]["caption_count"] == 1
    gate_text = (paper_dir / INVENTORY_FILENAME).read_text(encoding="utf-8")
    assert HATECHECK_QUOTE not in gate_text
    private = (paper_dir / "private" / INVENTORY_PRIVATE_FILENAME).read_text(encoding="utf-8")
    assert HATECHECK_QUOTE in private
    assert (
        first["gate_file_sha256s"]["inventory-fixture"]
        == hashlib.sha256((paper_dir / INVENTORY_FILENAME).read_bytes()).hexdigest()
    )
    # The inventory's own contract is distinct from the extractor's and the reviewer's.
    assert inventory_response_schema() != provider_json_schema()
    assert inventory_response_schema() != review_response_schema()


def test_the_inventory_content_hash_names_entries_not_telemetry() -> None:
    gate = {
        "schema_version": "dataset-inventory/0.2",
        "entries": [{"entry_id": "ds_1", "name_raw": "HateCheck"}],
        "inventory_prompt_sha256": "a" * 64,
        "telemetry": {"latency_seconds": 1.2, "cost_usd": 0.001},
    }
    rerun = {**gate, "telemetry": {"latency_seconds": 4.8, "cost_usd": 0.002}}
    other_entries = {**gate, "entries": [{"entry_id": "ds_2", "name_raw": "OLID"}]}

    assert inventory_content_sha256(gate) == inventory_content_sha256(rerun)
    assert inventory_content_sha256(gate) != inventory_content_sha256(other_entries)
    assert inventory_content_sha256(gate) == inventory_content_sha256(
        {key: value for key, value in gate.items() if key != "telemetry"}
    )


def test_a_gate_from_another_prompt_or_schema_is_redone(tmp_path: Path) -> None:
    paper_dir = tmp_path / "inventory-fixture"
    calls = {"count": 0}

    def call(system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls["count"] += 1
        return {"entries": [_entry()]}, {}

    run_inventory(run_root=tmp_path, model="fixture/model", call=call)
    gate = json.loads((paper_dir / INVENTORY_FILENAME).read_text(encoding="utf-8"))
    gate["inventory_prompt_sha256"] = "0" * 64
    (paper_dir / INVENTORY_FILENAME).write_text(json.dumps(gate), encoding="utf-8")

    summary = run_inventory(run_root=tmp_path, model="fixture/model", call=call)

    assert calls["count"] == 2
    assert summary["inventoried"] == 1
    assert summary["already_inventoried"] == 0


def test_the_cost_ceiling_stops_between_papers_and_keeps_what_it_bought(
    tmp_path: Path,
) -> None:
    _paper(tmp_path, slug="paper-b")
    budget_hit = {"value": False}

    def call(system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        budget_hit["value"] = True
        return {"entries": [_entry()]}, {"cost_usd": 9.0}

    summary = run_inventory(
        run_root=tmp_path, model="fixture/model", call=call, stop=lambda: budget_hit["value"]
    )

    assert summary["stopped_for_budget"] is True
    assert summary["inventoried"] == 1
    assert load_inventory(tmp_path / "inventory-fixture") is not None
    assert load_inventory(tmp_path / "paper-b") is None


def test_repeated_failures_stop_the_run_and_name_the_failure(tmp_path: Path) -> None:
    for index in range(MAX_CONSECUTIVE_FAILURES + 5):
        _paper(tmp_path, slug=f"paper-{index:03d}")
    calls = {"count": 0}

    def call(system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls["count"] += 1
        raise ValueError("provider call telemetry is not request-bound")

    summary = run_inventory(run_root=tmp_path, model="fixture/model", call=call)

    assert summary["stopped_for_repeated_failures"] is True
    assert calls["count"] == MAX_CONSECUTIVE_FAILURES
    assert summary["failure_classes"] == {"ValueError": MAX_CONSECUTIVE_FAILURES}
    assert summary["inventoried"] == 0


def test_the_selection_honours_paper_ids_and_a_paper_without_layout_is_skipped(
    tmp_path: Path,
) -> None:
    other = _paper(tmp_path, slug="paper-b")
    (other / "private" / "layout.json").unlink()

    assert [paper.name for paper in select_papers(tmp_path)] == ["inventory-fixture", "paper-b"]
    assert [paper.name for paper in select_papers(tmp_path, paper_ids={"paper-b"})] == ["paper-b"]
    summary = run_inventory(
        run_root=tmp_path,
        model="fixture/model",
        call=lambda system, user: ({"entries": []}, {}),
        paper_ids={"paper-b"},
    )
    assert summary["skipped_no_input"] == 1
    assert summary["inventoried"] == 0


def test_the_cli_call_carries_the_determinism_fields_the_ledger_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from proceedings_to_eee import cli
    from proceedings_to_eee.resolution.dataset_inventory import (
        INVENTORY_REASONING_EFFORT,
        INVENTORY_SCHEMA_NAME,
        INVENTORY_SEED,
        INVENTORY_TEMPERATURE,
    )

    requests: list[dict[str, Any]] = []

    class RecordingProvider:
        def structured_chat(self, **kwargs: Any) -> None:
            requests.append(kwargs)
            raise RuntimeError("Offline test stops after the budget ledger accepts the request")

    monkeypatch.setattr(cli, "runtime_key", lambda: "fixture-key")
    monkeypatch.setattr(cli, "OpenRouterClient", lambda **kwargs: RecordingProvider())
    result = CliRunner().invoke(
        cli.app,
        ["inventory-datasets", str(tmp_path), "--model", "fixture/model", "--max-cost-usd", "1"],
    )

    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    assert requests[0]["model"] == "fixture/model"
    assert requests[0]["schema_name"] == INVENTORY_SCHEMA_NAME
    assert requests[0]["temperature"] == INVENTORY_TEMPERATURE
    assert requests[0]["reasoning_effort"] == INVENTORY_REASONING_EFFORT
    assert requests[0]["seed"] == INVENTORY_SEED
    assert (tmp_path / "private" / "dataset-inventory-budget-ledger.jsonl").is_file()
    summary = json.loads((tmp_path / "dataset-inventory-summary.json").read_text(encoding="utf-8"))
    assert summary["failed"] == 1
    assert summary["max_cost_usd"] == 1.0


@pytest.mark.parametrize(
    "ids_text",
    [
        json.dumps({"slugs": ["paper-b"]}),
        json.dumps(["paper-b"]),
        "# one id per line\npaper-b\n\n",
    ],
    ids=["slugs_object", "json_list", "one_per_line"],
)
def test_the_cli_dry_run_reports_the_selection_without_a_provider(
    tmp_path: Path, ids_text: str
) -> None:
    from proceedings_to_eee import cli

    _paper(tmp_path, slug="paper-b")
    ids = tmp_path / "ids.txt"
    ids.write_text(ids_text, encoding="utf-8")

    result = CliRunner().invoke(
        cli.app,
        ["inventory-datasets", str(tmp_path), "--dry-run", "--paper-ids", str(ids)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "selected": 1,
        "already_inventoried": 0,
        "provider_calls": 0,
    }
