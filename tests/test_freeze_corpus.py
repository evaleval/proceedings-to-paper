from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from proceedings_to_eee import cli, pipeline
from proceedings_to_eee.cli import app
from proceedings_to_eee.corpus import CorpusSpec, PaperSpec
from proceedings_to_eee.io import write_json
from proceedings_to_eee.pipeline import PipelineSettings
from proceedings_to_eee.sources.manifest import FrozenSource, SourceManifest, SourceRole


def _paper(paper_id: str) -> PaperSpec:
    return PaperSpec(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=2026,
        venue="ACM Test",
        pdf_url=f"https://example.org/{paper_id}.pdf",
        perspective_role="evaluated_system",
    )


def _manifest(paper: PaperSpec) -> SourceManifest:
    digest = paper.paper_id[0] * 64
    return SourceManifest(
        paper_id=paper.paper_id,
        title=paper.title,
        sources=[
            FrozenSource(
                source_id=f"src_{paper.paper_id}",
                paper_id=paper.paper_id,
                role=SourceRole.PAPER,
                original_uri=str(paper.pdf_url),
                resolved_uri=str(paper.pdf_url),
                retrieved_at=datetime(2026, 8, 4, tzinfo=UTC),
                sha256=digest,
                byte_size=100,
                media_type="application/pdf",
                cache_relpath=f"data/sources/{digest[:2]}/{digest}.pdf",
            )
        ],
    )


def _settings(tmp_path: Path) -> PipelineSettings:
    return PipelineSettings(
        project_root=tmp_path,
        schema_path=tmp_path / "schema.json",
        schema_sha256="0" * 64,
        output_root=tmp_path / "run",
        model="not-used",
    )


def test_freeze_corpus_continues_after_a_paper_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    papers = [_paper("alpha"), _paper("bravo"), _paper("charlie")]
    corpus = CorpusSpec(corpus_id="test-corpus", description="fixture", papers=papers)
    attempted: list[str] = []

    def fake_freeze_paper(paper: PaperSpec, settings: PipelineSettings) -> SourceManifest:
        del settings
        attempted.append(paper.paper_id)
        if paper.paper_id == "bravo":
            raise RuntimeError("download unavailable")
        return _manifest(paper)

    monkeypatch.setattr(pipeline, "freeze_paper", fake_freeze_paper)

    summary = pipeline.freeze_corpus(corpus, _settings(tmp_path))

    assert attempted == ["alpha", "bravo", "charlie"]
    assert summary["status"] == "partial_failure"
    assert summary["papers"] == 3
    assert summary["papers_succeeded"] == 2
    assert summary["papers_failed"] == 1
    assert summary["sources"] == 2
    assert [item["paper_id"] for item in summary["manifests"]] == ["alpha", "charlie"]
    assert [item["status"] for item in summary["results"]] == [
        "success",
        "error",
        "success",
    ]
    assert summary["results"][1] == {
        "paper_id": "bravo",
        "status": "error",
        "error": {"type": "RuntimeError", "message": "download unavailable"},
    }
    persisted = json.loads((tmp_path / "run" / "corpus-freeze.json").read_text(encoding="utf-8"))
    assert persisted == summary


def test_freeze_corpus_preserves_success_manifest_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    papers = [_paper("alpha"), _paper("bravo")]
    corpus = CorpusSpec(corpus_id="test-corpus", description="fixture", papers=papers)
    monkeypatch.setattr(pipeline, "freeze_paper", lambda paper, settings: _manifest(paper))

    summary = pipeline.freeze_corpus(corpus, _settings(tmp_path))

    assert summary["status"] == "success"
    assert summary["papers"] == summary["papers_succeeded"] == 2
    assert summary["papers_failed"] == 0
    assert summary["manifests"] == [
        {
            "paper_id": "alpha",
            "source_ids": ["src_alpha"],
            "sha256": ["a" * 64],
        },
        {
            "paper_id": "bravo",
            "source_ids": ["src_bravo"],
            "sha256": ["b" * 64],
        },
    ]


def test_freeze_corpus_redacts_bounded_error_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = _paper("alpha")
    corpus = CorpusSpec(corpus_id="test-corpus", description="fixture", papers=[paper])
    fake_bearer = "Bear" + "er bearer-secret"
    fake_provider_key = "sk" + "-or-provider-secret"
    fake_local_path = "/" + "Users/example/private-paper.pdf"
    sensitive = (
        "https://example.org/paper.pdf?X-Amz-Credential=signed-secret "
        f"{fake_bearer} api_key=api-secret {fake_provider_key} "
        f"{fake_local_path} " + "x" * 2_000
    )

    def fail_freeze(paper: PaperSpec, settings: PipelineSettings) -> SourceManifest:
        del paper, settings
        raise RuntimeError(sensitive)

    monkeypatch.setattr(pipeline, "freeze_paper", fail_freeze)

    summary = pipeline.freeze_corpus(corpus, _settings(tmp_path))

    message = summary["results"][0]["error"]["message"]
    assert len(message) <= 1_000
    assert "signed-secret" not in message
    assert "bearer-secret" not in message
    assert "api-secret" not in message
    assert "provider-secret" not in message
    assert "/" + "Users/example" not in message
    assert "https://example.org/paper.pdf?[REDACTED]" in message
    assert "[LOCAL_PATH]" in message


def test_freeze_paper_rejects_config_uri_drift(tmp_path: Path) -> None:
    paper = _paper("alpha")
    settings = _settings(tmp_path)
    write_json(settings.output_root / paper.paper_id / "source-manifest.json", _manifest(paper))
    changed = paper.model_copy(update={"pdf_url": "https://example.org/replacement-alpha.pdf"})

    with pytest.raises(ValueError, match="configured source bundle changed"):
        pipeline.freeze_paper(changed, settings)


def test_paper_spec_requires_commit_pinned_repository() -> None:
    with pytest.raises(ValueError, match="required together"):
        PaperSpec(
            paper_id="alpha",
            title="Paper alpha",
            year=2026,
            venue="ACM Test",
            pdf_url="https://example.org/alpha.pdf",
            repository_url="https://github.com/example/repository",
            perspective_role="evaluated_system",
        )


def test_freeze_paper_rejects_added_supplement_against_cached_manifest(
    tmp_path: Path,
) -> None:
    paper = _paper("alpha")
    settings = _settings(tmp_path)
    write_json(settings.output_root / paper.paper_id / "source-manifest.json", _manifest(paper))
    changed = paper.model_copy(update={"supplement_urls": ["https://example.org/supplement.zip"]})

    with pytest.raises(ValueError, match="configured source bundle changed"):
        pipeline.freeze_paper(changed, settings)


@pytest.mark.parametrize(("papers_failed", "exit_code"), [(0, 0), (1, 1)])
def test_freeze_corpus_cli_exit_code_reflects_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    papers_failed: int,
    exit_code: int,
) -> None:
    corpus_path = tmp_path / "corpus.yaml"
    corpus_path.write_text("fixture: true\n", encoding="utf-8")
    summary = {
        "status": "success" if papers_failed == 0 else "partial_failure",
        "papers_failed": papers_failed,
    }
    monkeypatch.setattr(cli, "load_corpus", lambda path: object())
    monkeypatch.setattr(cli, "freeze_corpus", lambda corpus, settings: summary)

    result = CliRunner().invoke(
        app,
        [
            "freeze-corpus",
            str(corpus_path),
            "--schema-path",
            str(Path("schemas/eee-0.2.2/eval.schema.json").resolve()),
            "--output",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == exit_code
    assert json.loads(result.stdout) == summary
