"""Non-overwriting, content-addressed preservation of a private run tree."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes, sha256_file, write_json

RUN_SEAL_NAME = "RUN-SEAL.json"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _inventory(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"run tree contains a symbolic link: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"run tree contains a non-regular entry: {relative}")
        if relative == RUN_SEAL_NAME:
            raise ValueError(f"source run already contains {RUN_SEAL_NAME}")
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise ValueError("run tree contains no files")
    return entries


def _manifest(source_run_name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "run-tree-seal/0.1",
        "source_run_name": source_run_name,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "tree_sha256": sha256_bytes(canonical_json_bytes(entries)),
        "files": entries,
    }


def seal_run_tree(run_root: Path, destination: Path) -> dict[str, Any]:
    """Copy and checksum a completed run without exposing or rewriting its content.

    The destination must not exist and must be outside the source tree. The
    source is inventoried before and after copying, so a concurrently changing
    provider run cannot be mistaken for a stable first-run copy.
    """

    source = run_root.resolve()
    target = destination.resolve()
    if not source.is_dir():
        raise ValueError("run root must be an existing directory")
    if target.exists():
        raise FileExistsError("sealed run destination already exists")
    if _is_within(target, source):
        raise ValueError("sealed run destination must be outside the source tree")

    before = _inventory(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=f".{target.name}.sealing-", dir=target.parent))
    temporary_tree = temporary_parent / "tree"
    try:
        shutil.copytree(source, temporary_tree, copy_function=shutil.copy2)
        copied = _inventory(temporary_tree)
        after = _inventory(source)
        if before != after:
            raise RuntimeError("run tree changed while it was being sealed")
        if before != copied:
            raise RuntimeError("sealed run copy does not match the source inventory")
        manifest = _manifest(source.name, copied)
        write_json(temporary_tree / RUN_SEAL_NAME, manifest)
        temporary_tree.rename(target)
    except BaseException:
        shutil.rmtree(temporary_parent, ignore_errors=True)
        raise
    temporary_parent.rmdir()
    return manifest
