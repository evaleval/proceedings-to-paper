from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from proceedings_to_eee.corpus import CorpusSpec, PaperSpec, load_corpus


def _paper(**updates: object) -> PaperSpec:
    payload: dict[str, object] = {
        "paper_id": "safe-paper-2026",
        "title": "Safe paper",
        "year": 2026,
        "venue": "Test venue",
        "pdf_url": "https://example.org/paper.pdf",
        "perspective_role": "evaluated system",
    }
    payload.update(updates)
    return PaperSpec.model_validate(payload)


@pytest.mark.parametrize(
    "paper_id",
    ["../escape", "nested/paper", "UPPER", "has space", "under_score", ".hidden", "-bad"],
)
def test_paper_id_must_be_a_safe_lowercase_slug(paper_id: str) -> None:
    with pytest.raises(ValidationError, match="paper_id"):
        _paper(paper_id=paper_id)


def test_corpus_rejects_duplicate_paper_ids() -> None:
    paper = _paper()
    with pytest.raises(ValidationError, match="paper_id values must be unique"):
        CorpusSpec(corpus_id="fixture", description="fixture", papers=[paper, paper])


def test_paper_requires_exactly_one_pdf_location() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        _paper(pdf_url=None)
    with pytest.raises(ValidationError, match="exactly one"):
        _paper(pdf_path="paper.pdf")


def test_load_corpus_resolves_local_pdf_relative_to_corpus_file(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "nested" / "config"
    source_dir = tmp_path / "nested" / "sources"
    corpus_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    pdf_path = source_dir / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nfixture\n")
    corpus_path = corpus_dir / "corpus.yaml"
    corpus_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "pilot-corpus/0.2",
                "corpus_id": "local-fixture",
                "description": "local path fixture",
                "papers": [
                    {
                        "paper_id": "local-paper",
                        "title": "Local paper",
                        "year": 2026,
                        "venue": "Test venue",
                        "pdf_path": "../sources/paper.pdf",
                        "perspective_role": "evaluated system",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    corpus = load_corpus(corpus_path)

    assert corpus.papers[0].pdf_url is None
    assert corpus.papers[0].pdf_path == str(pdf_path.resolve())


def test_load_corpus_rejects_missing_local_pdf_before_pipeline_writes(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.yaml"
    corpus_path.write_text(
        yaml.safe_dump(
            {
                "corpus_id": "missing-fixture",
                "description": "missing local source",
                "papers": [
                    {
                        "paper_id": "missing-paper",
                        "title": "Missing paper",
                        "year": 2026,
                        "venue": "Test venue",
                        "pdf_path": "missing.pdf",
                        "perspective_role": "evaluated system",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        load_corpus(corpus_path)
