from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proceedings_to_eee import cli, pipeline
from proceedings_to_eee import preflight as preflight_module
from proceedings_to_eee.cli import app
from proceedings_to_eee.corpus import CorpusSpec, PaperSpec
from proceedings_to_eee.evaluation import row_plan as row_plan_module
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.io import read_json
from proceedings_to_eee.pipeline import PipelineSettings
from proceedings_to_eee.preflight import PreflightCostAssumptions, preflight_corpus
from proceedings_to_eee.providers.openrouter import OpenRouterClient

PAGE = """Table 2. Detection performance per model.

     Model            Precision   Recall    F1
     System Cedar        0.42       0.81   0.55
     System Juniper      0.57       0.63   0.60
     System Maple        0.67       0.49   0.57
"""


def _fragment(*, source_id: str, page: int, text: str, score: float) -> PageFragment:
    return PageFragment(
        fragment_id=f"frag_{source_id}_{page:04d}",
        source_id=source_id,
        page=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        character_count=len(text),
        numeric_token_count=sum(character.isdigit() for character in text),
        result_signal_score=score,
    )


def _layout(source_id: str) -> PdfLayout:
    return PdfLayout(
        source_id=source_id,
        parser="fixture-layout",
        parser_version="fixture-layout/1",
        page_count=2,
        pages=[
            _fragment(source_id=source_id, page=1, text="Introduction\n", score=0.0),
            _fragment(source_id=source_id, page=2, text=PAGE, score=10.0),
        ],
    )


def _write_local_corpus(tmp_path: Path) -> Path:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nprovider-free fixture\n")
    corpus = tmp_path / "corpus.yaml"
    corpus.write_text(
        """schema_version: pilot-corpus/0.2
corpus_id: provider-free-fixture
evaluation_split: unspecified
description: Local provider-free preflight fixture.
papers:
  - paper_id: fixture-paper
    title: Fixture Paper
    year: 2026
    venue: Fixture Venue
    pdf_path: paper.pdf
    perspective_role: evaluated_system
    include_pages: [2]
""",
        encoding="utf-8",
    )
    return corpus


def _forbid_provider(*args, **kwargs):
    del args, kwargs
    raise AssertionError("provider construction or access is forbidden during preflight")


def test_preflight_cli_is_provider_free_and_writes_auditable_exact_bounds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus_path = _write_local_corpus(tmp_path)
    output = tmp_path / "preflight"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        preflight_module, "extract_pdf_layout", lambda path, source_id: _layout(source_id)
    )
    monkeypatch.setattr(OpenRouterClient, "__init__", _forbid_provider)
    monkeypatch.setattr(OpenRouterClient, "structured_chat", _forbid_provider)
    monkeypatch.setattr(cli, "runtime_key", _forbid_provider)

    result = CliRunner().invoke(
        app,
        [
            "preflight-corpus",
            str(corpus_path),
            "--output",
            str(output),
            "--row-enumeration",
            "--tuple-call-ceiling",
            "2",
            "--verifier-call-ceiling",
            "2",
            "--origin-call-ceiling",
            "2",
            "--legacy-call-cost-usd",
            "0.10",
            "--row-call-cost-usd",
            "0.20",
            "--tuple-call-cost-usd",
            "0.25",
            "--verifier-call-cost-usd",
            "0.30",
            "--origin-call-cost-usd",
            "0.40",
        ],
    )

    assert result.exit_code == 0, result.output
    compact = json.loads(result.stdout)
    assert compact["provider_calls_made"] == 0
    assert compact["provider_credentials_required"] is False
    assert compact["papers"] == [
        {
            "paper_id": "fixture-paper",
            "status": "success",
            "preflight": str(output / "fixture-paper" / "preflight.json"),
            "selected_pages": [2],
            "result_blocks": 1,
            "rows_planned": 3,
            "maximum_extraction_calls_before_verifier_or_origin": 8,
        }
    ]
    bound = compact["structured_call_bound"]
    assert bound["legacy_initial_calls"] == 1
    assert bound["legacy_maximum_calls"] == 5
    assert bound["row_base_calls"] == 1
    assert bound["row_maximum_calls"] == 3
    assert bound["tuple_call_ceiling"] == 2
    assert bound["verifier_call_ceiling"] == 2
    assert bound["origin_call_ceiling"] == 2
    assert bound["configured_base_structured_calls"] == 8
    assert bound["maximum_structured_calls"] == 14
    assert compact["cost_estimates"]["estimated_maximum_total_cost_usd"] == 3.0

    expected_artifacts = (
        output / "corpus-freeze.json",
        output / "corpus-preflight.json",
        output / "fixture-paper" / "source-manifest.json",
        output / "fixture-paper" / "preflight.json",
        output / "fixture-paper" / "private" / "layout.json",
        output / "fixture-paper" / "private" / "result-blocks.json",
        output / "fixture-paper" / "private" / "row-enumeration-plan.json",
    )
    assert all(path.is_file() for path in expected_artifacts)
    persisted = read_json(output / "corpus-preflight.json")
    assert persisted["provider_calls_made"] == 0
    assert persisted["counts"] == {
        "selected_pages": 1,
        "result_blocks": 1,
        "rows_planned": 3,
        "unbatchable_rows": 0,
    }
    paper = read_json(output / "fixture-paper" / "preflight.json")
    assert paper["selection"]["selected_pages"][0]["page"] == 2
    assert len(paper["selected_blocks"]) == 1
    assert [row["page"] for row in paper["row_plan"]["rows"]] == [2, 2, 2]
    assert all(item["sha256"] for item in paper["artifacts"].values())


def test_preflight_does_not_infer_cost_without_user_assumptions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nprovider-free fixture\n")
    corpus = CorpusSpec(
        corpus_id="fixture-corpus",
        description="fixture",
        papers=[
            PaperSpec(
                paper_id="fixture-paper",
                title="Fixture Paper",
                year=2026,
                venue="Fixture Venue",
                pdf_path=str(pdf),
                perspective_role="evaluated_system",
                include_pages=[2],
            )
        ],
    )
    settings = PipelineSettings(
        project_root=tmp_path,
        schema_path=tmp_path / "unused-schema.json",
        schema_sha256="0" * 64,
        output_root=tmp_path / "preflight",
        model="not-used",
        row_enumeration_enabled=True,
    )
    monkeypatch.setattr(
        preflight_module, "extract_pdf_layout", lambda path, source_id: _layout(source_id)
    )

    report = preflight_corpus(corpus=corpus, settings=settings)

    costs = report["cost_estimates"]
    assert costs["available"] is False
    assert costs["missing_user_assumptions"] == ["legacy", "row"]
    assert costs["estimated_maximum_total_cost_usd"] is None
    assert all(stage["estimated_maximum_cost_usd"] is None for stage in costs["stages"].values())


@pytest.mark.parametrize(
    ("tuple_ceiling", "verifier_ceiling", "origin_ceiling", "message"),
    [
        (0, 1, 0, "verifier call ceiling cannot exceed tuple call ceiling"),
        (2, 1, 2, "origin call ceiling cannot exceed verifier call ceiling"),
    ],
)
def test_preflight_rejects_impossible_nested_stage_ceiling_before_source_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tuple_ceiling: int,
    verifier_ceiling: int,
    origin_ceiling: int,
    message: str,
) -> None:
    corpus = CorpusSpec(
        corpus_id="fixture",
        description="fixture",
        papers=[
            PaperSpec(
                paper_id="fixture-paper",
                title="Fixture Paper",
                year=2026,
                venue="Fixture Venue",
                pdf_path=str(tmp_path / "unused.pdf"),
                perspective_role="evaluated_system",
            )
        ],
    )
    settings = PipelineSettings(
        project_root=tmp_path,
        schema_path=tmp_path / "unused.json",
        schema_sha256="0" * 64,
        output_root=tmp_path / "preflight",
        model="not-used",
    )
    source_work_started = False

    def forbidden_freeze(*args: Any, **kwargs: Any) -> Any:
        nonlocal source_work_started
        source_work_started = True
        raise AssertionError("source work must not start")

    monkeypatch.setattr(preflight_module, "freeze_corpus", forbidden_freeze)

    with pytest.raises(ValueError, match=message):
        preflight_corpus(
            corpus=corpus,
            settings=settings,
            tuple_call_ceiling=tuple_ceiling,
            verifier_call_ceiling=verifier_ceiling,
            origin_call_ceiling=origin_ceiling,
        )

    assert source_work_started is False


def test_preflight_cli_renders_invalid_corpus_without_traceback(tmp_path: Path) -> None:
    corpus = tmp_path / "invalid.yaml"
    corpus.write_text(
        """corpus_id: invalid
description: invalid
papers:
  - paper_id: ../unsafe
    title: Unsafe
    year: 2026
    venue: Fixture
    pdf_path: missing.pdf
    perspective_role: evaluated_system
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["preflight-corpus", str(corpus)])

    assert result.exit_code == 1
    assert "preflight-failed" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "runs").exists()


def test_next_run_reconstruction_accepts_frozen_local_pdf_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nlocal source\n")
    spec = PaperSpec(
        paper_id="fixture-paper",
        title="Fixture Paper",
        year=2026,
        venue="Fixture Venue",
        pdf_path=str(pdf),
        perspective_role="evaluated_system",
    )
    settings = PipelineSettings(
        project_root=tmp_path,
        schema_path=tmp_path / "unused-schema.json",
        schema_sha256="0" * 64,
        output_root=tmp_path / "run",
        model="not-used",
    )
    frozen = pipeline.freeze_paper(spec, settings)
    monkeypatch.setattr(
        row_plan_module,
        "extract_pdf_layout",
        lambda path, source_id: _layout(source_id),
    )

    reconstructed, layout = row_plan_module._reconstruct_frozen_layout(
        paper_dir=settings.output_root / spec.paper_id,
        spec=spec,
        project_root=tmp_path,
    )

    assert reconstructed == frozen
    assert layout.source_id == frozen.sources[0].source_id


def test_preflight_cost_assumptions_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        PreflightCostAssumptions(legacy_call_usd=float("nan"))
