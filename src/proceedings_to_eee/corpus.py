"""Versioned YAML corpus specifications for reproducible pilots."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, HttpUrl, field_validator, model_validator

from proceedings_to_eee.domain.observation import StrictModel
from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes


class ExpectedSpotCheck(StrictModel):
    """Development-visible reference used only for scoring, never extraction prompts."""

    system: str
    dataset: str
    metric: str
    raw_value: str
    page: int | None = Field(default=None, ge=1)
    label: str | None = None
    claim_type: str = "primary_result"


class PaperSpec(StrictModel):
    paper_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str
    year: int
    venue: str
    doi: str | None = None
    arxiv_id: str | None = None
    acm_url: HttpUrl | None = None
    pdf_url: HttpUrl | None = None
    pdf_path: str | None = None
    supplement_urls: list[HttpUrl] = Field(default_factory=list)
    repository_url: HttpUrl | None = None
    repository_commit: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{40}$")
    perspective_role: str
    include_pages: list[int] = Field(default_factory=list)
    max_result_pages: int = Field(default=8, ge=1, le=30)
    expected_spot_checks: list[ExpectedSpotCheck] = Field(default_factory=list)
    reference_path: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("pdf_path", "reference_path")
    @classmethod
    def paths_are_well_formed(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        path = Path(value)
        if info.field_name == "reference_path" and (path.is_absolute() or ".." in path.parts):
            raise ValueError("reference_path must stay project-relative")
        return str(path) if path.is_absolute() else path.as_posix()

    @model_validator(mode="after")
    def immutable_repository_and_unique_sources(self) -> PaperSpec:
        if (self.pdf_url is None) == (self.pdf_path is None):
            raise ValueError("exactly one of pdf_url or pdf_path is required")
        if (self.repository_url is None) != (self.repository_commit is None):
            raise ValueError("repository_url and immutable repository_commit are required together")
        supplement_urls = [str(url) for url in self.supplement_urls]
        if len(supplement_urls) != len(set(supplement_urls)):
            raise ValueError("supplement_urls must be unique")
        return self


class CorpusSpec(StrictModel):
    schema_version: str = "pilot-corpus/0.2"
    corpus_id: str
    evaluation_split: Literal["development", "holdout", "unspecified"] = "unspecified"
    description: str
    papers: list[PaperSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def paper_ids_are_unique(self) -> CorpusSpec:
        paper_ids = [paper.paper_id for paper in self.papers]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("paper_id values must be unique within a corpus")
        return self


def build_corpus_binding(corpus: CorpusSpec) -> dict[str, str]:
    """Bind a run/preflight to one ordered, explicitly classified corpus spec."""

    return {
        "schema_version": corpus.schema_version,
        "corpus_id": corpus.corpus_id,
        "evaluation_split": corpus.evaluation_split,
        "corpus_spec_sha256": sha256_bytes(canonical_json_bytes(corpus)),
        "paper_ids_sha256": sha256_bytes(
            canonical_json_bytes([paper.paper_id for paper in corpus.papers])
        ),
    }


def load_corpus(path: Path) -> CorpusSpec:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    corpus = CorpusSpec.model_validate(payload)
    base_dir = path.resolve().parent
    resolved_papers: list[PaperSpec] = []
    for paper in corpus.papers:
        if paper.pdf_path is None:
            resolved_papers.append(paper)
            continue
        configured_path = Path(paper.pdf_path)
        resolved_path = (
            configured_path if configured_path.is_absolute() else base_dir / configured_path
        ).resolve(strict=True)
        if not resolved_path.is_file():
            raise ValueError(f"local PDF path is not a file for paper_id={paper.paper_id}")
        resolved_papers.append(paper.model_copy(update={"pdf_path": str(resolved_path)}))
    return corpus.model_copy(update={"papers": resolved_papers})
