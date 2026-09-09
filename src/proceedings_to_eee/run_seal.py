"""Non-overwriting preservation and verification of sealed run trees."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes, sha256_file, write_json

RUN_SEAL_NAME = "RUN-SEAL.json"
RUN_SEAL_SCHEMA_VERSION = "run-tree-seal/0.1"
_SHA256_LENGTH = 64
_MANIFEST_FIELDS = {
    "schema_version",
    "source_run_name",
    "file_count",
    "total_bytes",
    "tree_sha256",
    "files",
}
_FILE_FIELDS = {"path", "size_bytes", "sha256"}


class RunSealErrorCode(StrEnum):
    """Stable, secret-safe reasons that a sealed run failed verification."""

    ROOT_MISSING = "run_seal_root_missing"
    ROOT_NOT_DIRECTORY = "run_seal_root_not_directory"
    ROOT_SYMLINK = "run_seal_root_symlink"
    SEAL_MISSING = "run_seal_missing"
    SEAL_SYMLINK = "run_seal_symlink"
    SEAL_NOT_REGULAR = "run_seal_not_regular"
    SEAL_UNREADABLE = "run_seal_unreadable"
    INVALID_JSON = "run_seal_invalid_json"
    INVALID_STRUCTURE = "run_seal_invalid_structure"
    SCHEMA_VERSION_MISMATCH = "run_seal_schema_version_mismatch"
    PATH_NONCANONICAL = "run_seal_path_noncanonical"
    DUPLICATE_PATH = "run_seal_duplicate_path"
    INVENTORY_ORDER = "run_seal_inventory_order"
    SELF_INVENTORY = "run_seal_self_inventory"
    FILE_COUNT_MISMATCH = "run_seal_file_count_mismatch"
    TOTAL_BYTES_MISMATCH = "run_seal_total_bytes_mismatch"
    TREE_HASH_MISMATCH = "run_seal_tree_hash_mismatch"
    TREE_SYMLINK = "run_seal_tree_symlink"
    TREE_NON_REGULAR = "run_seal_tree_non_regular"
    TREE_UNREADABLE = "run_seal_tree_unreadable"
    INVENTORY_EXTRA = "run_seal_inventory_extra"
    INVENTORY_MISSING = "run_seal_inventory_missing"
    FILE_SIZE_MISMATCH = "run_seal_file_size_mismatch"
    FILE_HASH_MISMATCH = "run_seal_file_hash_mismatch"


_ERROR_MESSAGES = {
    RunSealErrorCode.ROOT_MISSING: "sealed run root does not exist",
    RunSealErrorCode.ROOT_NOT_DIRECTORY: "sealed run root is not a directory",
    RunSealErrorCode.ROOT_SYMLINK: "sealed run root must not be a symbolic link",
    RunSealErrorCode.SEAL_MISSING: "run seal is missing",
    RunSealErrorCode.SEAL_SYMLINK: "run seal must not be a symbolic link",
    RunSealErrorCode.SEAL_NOT_REGULAR: "run seal is not a regular file",
    RunSealErrorCode.SEAL_UNREADABLE: "run seal could not be read",
    RunSealErrorCode.INVALID_JSON: "run seal is not valid UTF-8 JSON",
    RunSealErrorCode.INVALID_STRUCTURE: "run seal has an invalid structure",
    RunSealErrorCode.SCHEMA_VERSION_MISMATCH: "run seal schema version is unsupported",
    RunSealErrorCode.PATH_NONCANONICAL: "run seal contains a non-canonical relative path",
    RunSealErrorCode.DUPLICATE_PATH: "run seal contains a duplicate inventory path",
    RunSealErrorCode.INVENTORY_ORDER: "run seal inventory is not in canonical order",
    RunSealErrorCode.SELF_INVENTORY: "run seal incorrectly inventories itself",
    RunSealErrorCode.FILE_COUNT_MISMATCH: "run seal file count does not match its inventory",
    RunSealErrorCode.TOTAL_BYTES_MISMATCH: "run seal byte count does not match its inventory",
    RunSealErrorCode.TREE_HASH_MISMATCH: "run seal tree hash does not match its inventory",
    RunSealErrorCode.TREE_SYMLINK: "sealed run contains a symbolic link",
    RunSealErrorCode.TREE_NON_REGULAR: "sealed run contains a non-regular entry",
    RunSealErrorCode.TREE_UNREADABLE: "sealed run inventory could not be read",
    RunSealErrorCode.INVENTORY_EXTRA: "sealed run contains an unsealed file",
    RunSealErrorCode.INVENTORY_MISSING: "sealed run is missing an inventoried file",
    RunSealErrorCode.FILE_SIZE_MISMATCH: "sealed run file size does not match its seal",
    RunSealErrorCode.FILE_HASH_MISMATCH: "sealed run file hash does not match its seal",
}


class RunSealVerificationError(ValueError):
    """A sealed run failed verification without exposing a local path or filename."""

    def __init__(self, code: RunSealErrorCode) -> None:
        self.code = code.value
        super().__init__(_ERROR_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class VerifiedRunSeal:
    """Metadata bound to a successfully verified sealed run tree."""

    manifest: dict[str, Any]
    seal_sha256: str
    tree_sha256: str
    file_count: int
    total_bytes: int
    schema_version: str
    source_run_name: str
    files: tuple[dict[str, Any], ...]


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError
        result[key] = value
    return result


def _fail(code: RunSealErrorCode) -> NoReturn:
    raise RunSealVerificationError(code)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_canonical_relative_path(value: str) -> bool:
    if not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and value == path.as_posix()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _inventory(root: Path) -> list[dict[str, Any]]:
    """Inventory a source tree for sealing.

    This retains the original sealing contract: a seal cannot be nested in its
    source inventory, symbolic links and non-files are rejected, and output is
    deterministically ordered.
    """

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
        if not _is_canonical_relative_path(relative):
            raise ValueError(f"run tree contains a non-canonical path: {relative}")
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
        "schema_version": RUN_SEAL_SCHEMA_VERSION,
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


def _read_manifest(seal_path: Path) -> tuple[dict[str, Any], str]:
    try:
        content = seal_path.read_bytes()
    except OSError:
        _fail(RunSealErrorCode.SEAL_UNREADABLE)
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKeyError,
        RecursionError,
        OverflowError,
        ValueError,
    ):
        _fail(RunSealErrorCode.INVALID_JSON)
    if not isinstance(value, dict):
        _fail(RunSealErrorCode.INVALID_STRUCTURE)
    return value, sha256_bytes(content)


def _validated_manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if set(manifest) != _MANIFEST_FIELDS:
        _fail(RunSealErrorCode.INVALID_STRUCTURE)
    if manifest.get("schema_version") != RUN_SEAL_SCHEMA_VERSION:
        _fail(RunSealErrorCode.SCHEMA_VERSION_MISMATCH)

    source_run_name = manifest.get("source_run_name")
    if (
        not isinstance(source_run_name, str)
        or not source_run_name
        or source_run_name in {".", ".."}
        or "/" in source_run_name
        or "\\" in source_run_name
        or "\x00" in source_run_name
    ):
        _fail(RunSealErrorCode.INVALID_STRUCTURE)

    file_count = manifest.get("file_count")
    total_bytes = manifest.get("total_bytes")
    tree_sha256 = manifest.get("tree_sha256")
    raw_entries = manifest.get("files")
    if (
        not _is_nonnegative_int(file_count)
        or not _is_nonnegative_int(total_bytes)
        or not _is_sha256(tree_sha256)
        or not isinstance(raw_entries, list)
        or not raw_entries
    ):
        _fail(RunSealErrorCode.INVALID_STRUCTURE)

    entries: list[dict[str, Any]] = []
    paths: list[str] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != _FILE_FIELDS:
            _fail(RunSealErrorCode.INVALID_STRUCTURE)
        path = raw_entry.get("path")
        size_bytes = raw_entry.get("size_bytes")
        digest = raw_entry.get("sha256")
        if not isinstance(path, str) or not _is_canonical_relative_path(path):
            _fail(RunSealErrorCode.PATH_NONCANONICAL)
        if path == RUN_SEAL_NAME:
            _fail(RunSealErrorCode.SELF_INVENTORY)
        if not _is_nonnegative_int(size_bytes) or not _is_sha256(digest):
            _fail(RunSealErrorCode.INVALID_STRUCTURE)
        paths.append(path)
        entries.append({"path": path, "size_bytes": size_bytes, "sha256": digest})

    if len(set(paths)) != len(paths):
        _fail(RunSealErrorCode.DUPLICATE_PATH)
    if paths != sorted(paths):
        _fail(RunSealErrorCode.INVENTORY_ORDER)
    if file_count != len(entries):
        _fail(RunSealErrorCode.FILE_COUNT_MISMATCH)
    if total_bytes != sum(entry["size_bytes"] for entry in entries):
        _fail(RunSealErrorCode.TOTAL_BYTES_MISMATCH)
    if tree_sha256 != sha256_bytes(canonical_json_bytes(entries)):
        _fail(RunSealErrorCode.TREE_HASH_MISMATCH)
    return entries


def _verification_inventory(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        for path in paths:
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                _fail(RunSealErrorCode.TREE_SYMLINK)
            if path.is_dir():
                continue
            if not path.is_file():
                _fail(RunSealErrorCode.TREE_NON_REGULAR)
            if relative == RUN_SEAL_NAME:
                continue
            if not _is_canonical_relative_path(relative):
                _fail(RunSealErrorCode.PATH_NONCANONICAL)
            entries.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    except RunSealVerificationError:
        raise
    except OSError:
        _fail(RunSealErrorCode.TREE_UNREADABLE)
    return entries


def verify_run_seal(root: Path) -> VerifiedRunSeal:
    """Verify a sealed run tree and bind its exact seal-file bytes.

    The verifier never follows paths from the manifest. It first validates all
    paths as canonical POSIX-relative names, independently inventories the tree,
    and then compares the two inventories. ``RUN-SEAL.json`` is intentionally
    outside the tree inventory because a file cannot include its own digest; its
    exact raw-byte SHA-256 is returned separately as ``seal_sha256``.
    """

    if root.is_symlink():
        _fail(RunSealErrorCode.ROOT_SYMLINK)
    if not root.exists():
        _fail(RunSealErrorCode.ROOT_MISSING)
    if not root.is_dir():
        _fail(RunSealErrorCode.ROOT_NOT_DIRECTORY)

    seal_path = root / RUN_SEAL_NAME
    if seal_path.is_symlink():
        _fail(RunSealErrorCode.SEAL_SYMLINK)
    if not seal_path.exists():
        _fail(RunSealErrorCode.SEAL_MISSING)
    if not seal_path.is_file():
        _fail(RunSealErrorCode.SEAL_NOT_REGULAR)

    manifest, seal_sha256 = _read_manifest(seal_path)
    expected = _validated_manifest_entries(manifest)
    actual = _verification_inventory(root)
    expected_by_path = {entry["path"]: entry for entry in expected}
    actual_by_path = {entry["path"]: entry for entry in actual}

    if actual_by_path.keys() - expected_by_path.keys():
        _fail(RunSealErrorCode.INVENTORY_EXTRA)
    if expected_by_path.keys() - actual_by_path.keys():
        _fail(RunSealErrorCode.INVENTORY_MISSING)
    for path in sorted(expected_by_path):
        expected_entry = expected_by_path[path]
        actual_entry = actual_by_path[path]
        if actual_entry["size_bytes"] != expected_entry["size_bytes"]:
            _fail(RunSealErrorCode.FILE_SIZE_MISMATCH)
        if actual_entry["sha256"] != expected_entry["sha256"]:
            _fail(RunSealErrorCode.FILE_HASH_MISMATCH)

    return VerifiedRunSeal(
        manifest=manifest,
        seal_sha256=seal_sha256,
        tree_sha256=manifest["tree_sha256"],
        file_count=manifest["file_count"],
        total_bytes=manifest["total_bytes"],
        schema_version=manifest["schema_version"],
        source_run_name=manifest["source_run_name"],
        files=tuple(expected),
    )
