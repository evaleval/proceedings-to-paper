from __future__ import annotations

import json
from pathlib import Path

import pytest

from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes
from proceedings_to_eee.run_seal import RUN_SEAL_NAME, seal_run_tree


def _source_tree(tmp_path: Path) -> Path:
    source = tmp_path / "holdout-run"
    (source / "paper-a" / "eee").mkdir(parents=True)
    (source / "corpus-run.json").write_text('{"status":"success"}\n', encoding="utf-8")
    (source / "paper-a" / "run.json").write_text('{"paper_id":"paper-a"}\n', encoding="utf-8")
    (source / "paper-a" / "eee" / "record.json").write_bytes(b'{"score":0.7}\n')
    return source


def test_seal_run_tree_preserves_and_hashes_exact_files(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    target = tmp_path / "sealed" / "holdout-first-run"

    manifest = seal_run_tree(source, target)

    persisted = json.loads((target / RUN_SEAL_NAME).read_text(encoding="utf-8"))
    assert persisted == manifest
    assert manifest["schema_version"] == "run-tree-seal/0.1"
    assert manifest["source_run_name"] == "holdout-run"
    assert manifest["file_count"] == 3
    assert manifest["tree_sha256"] == sha256_bytes(canonical_json_bytes(manifest["files"]))
    assert [entry["path"] for entry in manifest["files"]] == [
        "corpus-run.json",
        "paper-a/eee/record.json",
        "paper-a/run.json",
    ]
    assert all(not entry["path"].startswith("/") for entry in manifest["files"])
    assert (target / "paper-a" / "eee" / "record.json").read_bytes() == b'{"score":0.7}\n'

    (source / "paper-a" / "run.json").write_text("changed", encoding="utf-8")
    assert (target / "paper-a" / "run.json").read_text(encoding="utf-8") != "changed"


def test_seal_run_tree_refuses_overwrite_and_nested_destination(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        seal_run_tree(source, existing)
    with pytest.raises(ValueError, match="outside the source tree"):
        seal_run_tree(source, source / "sealed")


def test_seal_run_tree_rejects_symlinks_and_prior_seal(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    (source / "linked-run.json").symlink_to(source / "paper-a" / "run.json")
    with pytest.raises(ValueError, match="symbolic link"):
        seal_run_tree(source, tmp_path / "sealed-link")

    (source / "linked-run.json").unlink()
    (source / RUN_SEAL_NAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already contains"):
        seal_run_tree(source, tmp_path / "sealed-twice")
