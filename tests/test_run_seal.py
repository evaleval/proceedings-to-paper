from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes, sha256_file, write_json
from proceedings_to_eee.run_seal import (
    RUN_SEAL_NAME,
    RunSealErrorCode,
    RunSealVerificationError,
    seal_run_tree,
    verify_run_seal,
)


def _source_tree(tmp_path: Path) -> Path:
    source = tmp_path / "holdout-run"
    (source / "paper-a" / "eee").mkdir(parents=True)
    (source / "corpus-run.json").write_text('{"status":"success"}\n', encoding="utf-8")
    (source / "paper-a" / "run.json").write_text('{"paper_id":"paper-a"}\n', encoding="utf-8")
    (source / "paper-a" / "eee" / "record.json").write_bytes(b'{"score":0.7}\n')
    return source


def _sealed_tree(tmp_path: Path) -> Path:
    target = tmp_path / "sealed" / "holdout-first-run"
    seal_run_tree(_source_tree(tmp_path), target)
    return target


def _read_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / RUN_SEAL_NAME).read_text(encoding="utf-8"))


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    write_json(root / RUN_SEAL_NAME, manifest)


def _refresh_manifest_aggregates(manifest: dict[str, Any]) -> None:
    manifest["file_count"] = len(manifest["files"])
    manifest["total_bytes"] = sum(entry["size_bytes"] for entry in manifest["files"])
    manifest["tree_sha256"] = sha256_bytes(canonical_json_bytes(manifest["files"]))


def _assert_code(root: Path, code: RunSealErrorCode) -> None:
    with pytest.raises(RunSealVerificationError) as captured:
        verify_run_seal(root)
    assert captured.value.code == code.value
    assert str(root) not in str(captured.value)


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


def test_verify_run_seal_returns_bound_metadata_and_excludes_itself(tmp_path: Path) -> None:
    root = _sealed_tree(tmp_path)
    manifest = _read_manifest(root)

    verified = verify_run_seal(root)

    assert verified.manifest == manifest
    assert verified.seal_sha256 == sha256_file(root / RUN_SEAL_NAME)
    assert verified.tree_sha256 == manifest["tree_sha256"]
    assert verified.file_count == 3
    assert verified.total_bytes == manifest["total_bytes"]
    assert verified.schema_version == "run-tree-seal/0.1"
    assert verified.source_run_name == "holdout-run"
    assert tuple(entry["path"] for entry in verified.files) == (
        "corpus-run.json",
        "paper-a/eee/record.json",
        "paper-a/run.json",
    )
    assert RUN_SEAL_NAME not in {entry["path"] for entry in verified.files}


def test_sealing_is_byte_deterministic_and_verification_is_read_only(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    first = tmp_path / "sealed-a"
    second = tmp_path / "sealed-b"

    first_manifest = seal_run_tree(source, first)
    second_manifest = seal_run_tree(source, second)
    before = {
        path.relative_to(first).as_posix(): sha256_file(path)
        for path in first.rglob("*")
        if path.is_file()
    }

    verify_run_seal(first)

    after = {
        path.relative_to(first).as_posix(): sha256_file(path)
        for path in first.rglob("*")
        if path.is_file()
    }
    assert first_manifest == second_manifest
    assert (first / RUN_SEAL_NAME).read_bytes() == (second / RUN_SEAL_NAME).read_bytes()
    assert before == after


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


@pytest.mark.parametrize(
    ("setup", "code"),
    [
        ("missing_root", RunSealErrorCode.ROOT_MISSING),
        ("file_root", RunSealErrorCode.ROOT_NOT_DIRECTORY),
        ("root_symlink", RunSealErrorCode.ROOT_SYMLINK),
        ("missing_seal", RunSealErrorCode.SEAL_MISSING),
        ("seal_symlink", RunSealErrorCode.SEAL_SYMLINK),
        ("seal_directory", RunSealErrorCode.SEAL_NOT_REGULAR),
    ],
)
def test_verify_run_seal_rejects_missing_or_nonregular_inputs(
    tmp_path: Path,
    setup: str,
    code: RunSealErrorCode,
) -> None:
    if setup == "missing_root":
        root = tmp_path / "secret-missing-root"
    elif setup == "file_root":
        root = tmp_path / "secret-file-root"
        root.write_text("not a directory", encoding="utf-8")
    elif setup == "root_symlink":
        target = _sealed_tree(tmp_path)
        root = tmp_path / "secret-root-link"
        root.symlink_to(target, target_is_directory=True)
    else:
        root = _sealed_tree(tmp_path)
        seal = root / RUN_SEAL_NAME
        if setup == "missing_seal":
            seal.unlink()
        elif setup == "seal_symlink":
            seal.unlink()
            seal.symlink_to(root / "corpus-run.json")
        else:
            seal.unlink()
            seal.mkdir()
    _assert_code(root, code)


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (b"not json\n", RunSealErrorCode.INVALID_JSON),
        (b"\xff", RunSealErrorCode.INVALID_JSON),
        (b"[]\n", RunSealErrorCode.INVALID_STRUCTURE),
        (b'{"schema_version":"one","schema_version":"two"}\n', RunSealErrorCode.INVALID_JSON),
    ],
)
def test_verify_run_seal_rejects_invalid_json(
    tmp_path: Path,
    content: bytes,
    code: RunSealErrorCode,
) -> None:
    root = _sealed_tree(tmp_path)
    (root / RUN_SEAL_NAME).write_bytes(content)
    _assert_code(root, code)


def test_verify_run_seal_rejects_unknown_schema_and_fields(tmp_path: Path) -> None:
    root = _sealed_tree(tmp_path)
    manifest = _read_manifest(root)
    manifest["schema_version"] = "run-tree-seal/9.9"
    _write_manifest(root, manifest)
    _assert_code(root, RunSealErrorCode.SCHEMA_VERSION_MISMATCH)

    manifest["schema_version"] = "run-tree-seal/0.1"
    manifest["unexpected"] = True
    _write_manifest(root, manifest)
    _assert_code(root, RunSealErrorCode.INVALID_STRUCTURE)


@pytest.mark.parametrize(
    "invalid_path",
    [
        "../outside.json",
        "/absolute.json",
        "paper-a//run.json",
        "./corpus-run.json",
        "paper-a/../run.json",
        "paper-a\\run.json",
        "paper-a/",
    ],
)
def test_verify_run_seal_rejects_traversal_and_noncanonical_paths(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    root = _sealed_tree(tmp_path)
    manifest = _read_manifest(root)
    manifest["files"][0]["path"] = invalid_path
    _refresh_manifest_aggregates(manifest)
    _write_manifest(root, manifest)
    _assert_code(root, RunSealErrorCode.PATH_NONCANONICAL)


def test_verify_run_seal_rejects_self_duplicate_and_unsorted_inventory(tmp_path: Path) -> None:
    root = _sealed_tree(tmp_path)
    original = _read_manifest(root)

    manifest = deepcopy(original)
    manifest["files"][0]["path"] = RUN_SEAL_NAME
    _refresh_manifest_aggregates(manifest)
    _write_manifest(root, manifest)
    _assert_code(root, RunSealErrorCode.SELF_INVENTORY)

    manifest = deepcopy(original)
    manifest["files"][1]["path"] = manifest["files"][0]["path"]
    _refresh_manifest_aggregates(manifest)
    _write_manifest(root, manifest)
    _assert_code(root, RunSealErrorCode.DUPLICATE_PATH)

    manifest = deepcopy(original)
    manifest["files"].reverse()
    _refresh_manifest_aggregates(manifest)
    _write_manifest(root, manifest)
    _assert_code(root, RunSealErrorCode.INVENTORY_ORDER)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("file_count", 99, RunSealErrorCode.FILE_COUNT_MISMATCH),
        ("total_bytes", 99, RunSealErrorCode.TOTAL_BYTES_MISMATCH),
        ("tree_sha256", "0" * 64, RunSealErrorCode.TREE_HASH_MISMATCH),
    ],
)
def test_verify_run_seal_rejects_aggregate_mismatches(
    tmp_path: Path,
    field: str,
    value: int | str,
    code: RunSealErrorCode,
) -> None:
    root = _sealed_tree(tmp_path)
    manifest = _read_manifest(root)
    manifest[field] = value
    _write_manifest(root, manifest)
    _assert_code(root, code)


def test_verify_run_seal_rejects_extra_and_missing_files(tmp_path: Path) -> None:
    root = _sealed_tree(tmp_path)
    (root / "secret-extra.json").write_text("{}\n", encoding="utf-8")
    _assert_code(root, RunSealErrorCode.INVENTORY_EXTRA)

    (root / "secret-extra.json").unlink()
    (root / "corpus-run.json").unlink()
    _assert_code(root, RunSealErrorCode.INVENTORY_MISSING)


def test_verify_run_seal_rejects_size_and_hash_mismatches(tmp_path: Path) -> None:
    root = _sealed_tree(tmp_path)
    path = root / "corpus-run.json"
    original = path.read_bytes()
    path.write_bytes(original + b"x")
    _assert_code(root, RunSealErrorCode.FILE_SIZE_MISMATCH)

    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    _assert_code(root, RunSealErrorCode.FILE_HASH_MISMATCH)


def test_verify_run_seal_rejects_tree_symlinks(tmp_path: Path) -> None:
    root = _sealed_tree(tmp_path)
    target = root / "corpus-run.json"
    target.unlink()
    target.symlink_to(root / "paper-a" / "run.json")
    _assert_code(root, RunSealErrorCode.TREE_SYMLINK)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform has no FIFO support")
def test_verify_run_seal_rejects_nonregular_entries(tmp_path: Path) -> None:
    root = _sealed_tree(tmp_path)
    os.mkfifo(root / "secret-pipe")
    _assert_code(root, RunSealErrorCode.TREE_NON_REGULAR)
