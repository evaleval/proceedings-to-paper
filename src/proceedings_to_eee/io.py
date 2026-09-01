"""Deterministic and atomic artifact I/O."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a public artifact with stable ordering and UTF-8."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Replace one artifact atomically without exposing partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> str:
    content = canonical_json_bytes(value)
    atomic_write_bytes(path, content)
    return sha256_bytes(content)


def write_jsonl(path: Path, values: Iterable[Any]) -> str:
    lines: list[bytes] = []
    for value in values:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json", exclude_none=True)
        lines.append(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
        )
    content = b"".join(lines)
    atomic_write_bytes(path, content)
    return sha256_bytes(content)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
