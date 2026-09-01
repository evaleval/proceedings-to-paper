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
    paper_id: str
    title: str
    year: int
    venue: str
    doi: str | None = None
    arxiv_id: str | None = None
    acm_url: HttpUrl | None = None
    pdf_url: HttpUrl
    supplement_urls: list[HttpUrl] = Field(default_factory=list)
    repository_url: HttpUrl | None = None
    repository_commit: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{40}$")
    perspective_role: str
    include_pages: list[int] = Field(default_factory=list)
    max_result_pages: int = Field(default=8, ge=1, le=30)
    expected_spot_checks: list[ExpectedSpotCheck] = Field(default_factory=list)
    reference_path: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("reference_path")
    @classmethod
    def reference_stays_project_relative(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("reference_path must stay project-relative")
        return path.as_posix()

    @model_validator(mode="after")
    def immutable_repository_and_unique_sources(self) -> PaperSpec:
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
    return CorpusSpec.model_validate(payload)
