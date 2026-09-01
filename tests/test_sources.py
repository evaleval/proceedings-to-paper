from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from proceedings_to_eee.io import sha256_file
from proceedings_to_eee.sources.manifest import (
    SourceRole,
    freeze_local_source,
    freeze_repository_source,
)


def test_local_freeze_is_content_addressed(tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-fixture")
    source = freeze_local_source(
        paper_id="paper",
        role=SourceRole.PAPER,
        path=paper,
        cache_root=tmp_path / "data" / "sources",
    )
    cached = tmp_path / source.cache_relpath
    assert cached.exists()
    assert source.sha256 == sha256_file(paper) == sha256_file(cached)
    assert source.cache_relpath.startswith("data/sources/")


def test_repository_freeze_verifies_full_advertised_commit(monkeypatch) -> None:
    commit = "a" * 40

    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(stdout=f"{commit}\tHEAD\n")

    monkeypatch.setattr("proceedings_to_eee.sources.manifest.subprocess.run", fake_run)

    source = freeze_repository_source(
        paper_id="paper",
        url="https://github.com/example/repository.git",
        git_commit=commit,
    )

    assert source.role == SourceRole.REPOSITORY
    assert source.git_commit == commit
    assert source.sha256 is None
    assert source.cache_relpath is None
