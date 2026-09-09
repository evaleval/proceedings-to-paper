from __future__ import annotations

import os
from pathlib import Path

import pytest

from proceedings_to_eee.evaluation.artifact_safety import assert_artifact_paths_safe


def test_rejects_hardlink_alias_of_sealed_input(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    source = sealed / "run.json"
    source.write_text("sealed", encoding="utf-8")
    alias = tmp_path / "result.json"
    os.link(source, alias)

    with pytest.raises(ValueError, match="aliases an immutable input"):
        assert_artifact_paths_safe([alias], sealed_root=sealed)


def test_rejects_symbolic_link_artifact(tmp_path: Path) -> None:
    protected = tmp_path / "input.json"
    protected.write_text("input", encoding="utf-8")
    alias = tmp_path / "output.json"
    alias.symlink_to(protected)

    with pytest.raises(ValueError, match="symbolic link"):
        assert_artifact_paths_safe([alias], protected_paths=[protected])


def test_accepts_distinct_output_outside_sealed_tree(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "run.json").write_text("sealed", encoding="utf-8")

    assert_artifact_paths_safe([tmp_path / "result.json"], sealed_root=sealed)
