#!/usr/bin/env python3
"""Check the proposed public Git index without reading private, untracked files.

This is a release guard, not a replacement for content or Git-history review. Stage
the complete candidate tree before running it; unstaged changes are not inspected.
Diagnostics identify locations and categories without echoing sensitive content.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT_FILES = frozenset(
    {
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "AGENTS.md",
        "LICENSE",
        "PROJECT_CONTRACT.md",
        "README.md",
        "pyproject.toml",
        "uv.lock",
    }
)
ROOT_DIRECTORIES = frozenset(
    {
        ".github",
        "assets",
        "configs",
        "docs",
        "examples",
        "results",
        "schemas",
        "scripts",
        "src",
        "tests",
    }
)
PRIVATE_COMPONENTS = frozenset(
    {
        ".agents",
        ".codex",
        ".firecrawl",
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "annotations",
        "artifacts",
        "build",
        "cache",
        "caches",
        "data",
        "derived",
        "dist",
        "experiments",
        "output",
        "private",
        "references",
        "reviews",
        "runs",
        "tmp",
        "traces",
        "vault",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".7z",
        ".bz2",
        ".csv",
        ".db",
        ".gz",
        ".jsonl",
        ".log",
        ".parquet",
        ".pdf",
        ".pickle",
        ".pkl",
        ".pyc",
        ".pyo",
        ".rar",
        ".sqlite",
        ".sqlite3",
        ".tar",
        ".tgz",
        ".tsv",
        ".whl",
        ".xz",
        ".zip",
    }
)
PUBLIC_RESULTS = frozenset({"results/census-summary-2026-09-05.json"})
PUBLIC_IMAGES = frozenset({"assets/proceedings-to-eee-pipeline.png"})
SYNTHETIC_HOME_USERS = frozenset({"example", "private", "reviewer", "synthetic", "test"})
SECRET_PATTERNS = (
    re.compile(r"\bsk-(?:or-v1-|proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
HOME_PATH = re.compile(r"(?:/(?:Users|home)/|[A-Za-z]:[\\/]+Users[\\/]+)([\w.-]+)[\\/]")
ROOT_HOME_PATH = re.compile(r"(?<![\w.])/root/[\w.-]+")
PRIVATE_DOCUMENT = re.compile(r"(?:handoff|meeting|morning-review|overnight)", re.IGNORECASE)
MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")
SHA256 = re.compile(r"[0-9a-f]{64}")
PUBLIC_LIMITATIONS = [
    "Counts describe a stored development run, not a fresh run of this release.",
    "Textual support and schema validity do not establish semantic correctness.",
    "Tiered export is weaker than canonical export. The model-reviewed path may retain "
    "failed referential or field-provenance checks.",
    "Reviewer error rate, current precision, whole-paper recall and generalization "
    "remain unmeasured.",
    "The annotation sample evaluates candidate tuple correctness, not producer origin or "
    "final exported-record accuracy.",
    "Selection covers at most eight result-bearing pages per paper.",
    "Source artifacts and the annotation packet are private; only aggregate counts and "
    "binding hashes are included.",
]
SUMMARY_SHAPE = {
    "schema_version": "research-status-summary/0.1",
    "snapshot_date": "2026-09-05",
    "classification": "historical_open_development_census_snapshot",
    "rerun_under_this_release": False,
    "sources": {
        "atlas_summary_sha256": SHA256,
        "eee_summary_sha256": SHA256,
        "label_packet_sha256": SHA256,
    },
    "population": {
        "papers_total": int,
        "papers_processed": int,
        "papers_with_candidates": int,
        "papers_unstarted": int,
        "papers_reserved": int,
        "papers_unavailable": int,
        "papers_budget_stopped": int,
    },
    "candidates": {"total": int, "text_supported": int},
    "eee": {
        "schema_version": "0.2.2",
        "schema_sha256": SHA256,
        "origin_policy": "tiered",
        "records": int,
        "schema_invalid_records": int,
        "papers_with_records": int,
        "atomic_observations": int,
        "canonical_records_reported": int,
        "records_by_tier": {"deterministic": int, "model_reviewed": int, "human_confirmed": int},
        "records_by_producer_origin_basis": {
            "model_asserted_primary_no_external_cue": int,
            "model_asserted_primary_unchecked": int,
            "model_reviewed_origin_quote": int,
        },
    },
    "human_annotation": {
        "packet_items": int,
        "labels_filled": int,
        "independent_validation_available": False,
    },
    "limitations": PUBLIC_LIMITATIONS,
}


@dataclass(frozen=True)
class IndexedFile:
    path: str
    mode: str
    content: bytes


@dataclass(frozen=True)
class Finding:
    path: str
    category: str
    line: int | None = None

    def __str__(self) -> str:
        location = f"{self.path}:{self.line}" if self.line is not None else self.path
        return f"{location}: {self.category}"


def read_index(root: Path) -> list[IndexedFile]:
    """Read staged blobs, including a newly initialized repository with no HEAD."""
    command = ["git", "-C", str(root)]
    listing = subprocess.run(
        [*command, "ls-files", "--stage", "--full-name", "-z", "--", ":/"],
        check=True,
        capture_output=True,
    ).stdout
    entries = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if stage != "0":
            entries.append(IndexedFile(path, "unmerged", b""))
            continue
        if mode not in {"100644", "100755"}:
            entries.append(IndexedFile(path, mode, b""))
            continue
        content = subprocess.run(
            [*command, "cat-file", "blob", object_id],
            check=True,
            capture_output=True,
        ).stdout
        entries.append(IndexedFile(path, mode, content))
    return entries


def safe_png(content: bytes) -> bool:
    """Accept only a bounded PNG with valid chunks and no text/EXIF metadata."""
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    allowed = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"sRGB", b"gAMA", b"cHRM", b"pHYs"}
    offset = 8
    seen_header = False
    seen_data = False
    while offset + 12 <= len(content):
        length = struct.unpack_from(">I", content, offset)[0]
        kind = content[offset + 4 : offset + 8]
        end = offset + 12 + length
        if kind not in allowed or end > len(content):
            return False
        payload = content[offset + 8 : end - 4]
        checksum = struct.unpack_from(">I", content, end - 4)[0]
        if zlib.crc32(kind + payload) != checksum:
            return False
        if kind == b"IHDR":
            if seen_header or offset != 8 or length != 13:
                return False
            width, height = struct.unpack_from(">II", payload)
            if not (0 < width <= 10000 and 0 < height <= 10000):
                return False
            seen_header = True
        elif not seen_header:
            return False
        if kind == b"IDAT":
            seen_data = True
        if kind == b"IEND":
            return length == 0 and seen_header and seen_data and end == len(content)
        offset = end
    return False


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _matches_summary_shape(value: object, shape: object) -> bool:
    if isinstance(shape, dict):
        return (
            isinstance(value, dict)
            and value.keys() == shape.keys()
            and all(_matches_summary_shape(value[key], expected) for key, expected in shape.items())
        )
    if shape is int:
        return type(value) is int and value >= 0
    if isinstance(shape, re.Pattern):
        return isinstance(value, str) and shape.fullmatch(value) is not None
    return type(value) is type(shape) and value == shape


def safe_public_summary(text: str) -> bool:
    """Only approved aggregate fields, fixed public text, and hashes may ship."""
    try:
        data = json.loads(text, object_pairs_hook=_unique_json_object)
    except (ValueError, RecursionError):
        return False
    if not _matches_summary_shape(data, SUMMARY_SHAPE):
        return False
    population = data["population"]
    candidates = data["candidates"]
    records = data["eee"]
    annotation = data["human_annotation"]
    return (
        population["papers_total"]
        == sum(
            population[key]
            for key in (
                "papers_processed",
                "papers_unstarted",
                "papers_reserved",
                "papers_unavailable",
                "papers_budget_stopped",
            )
        )
        and population["papers_with_candidates"] <= population["papers_processed"]
        and candidates["text_supported"] <= candidates["total"]
        and sum(records["records_by_tier"].values()) == records["records"]
        and sum(records["records_by_producer_origin_basis"].values()) == records["records"]
        and records["schema_invalid_records"] <= records["records"]
        and records["papers_with_records"] <= population["papers_with_candidates"]
        and records["atomic_observations"] <= candidates["total"]
        and annotation["labels_filled"] <= annotation["packet_items"]
    )


def check_public_links(path: str, text: str, indexed_paths: set[str]) -> list[Finding]:
    """Check inline local Markdown links outside code against the staged tree."""
    findings = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = re.sub(r"`+[^`]*`+", "", line)
        for match in MARKDOWN_LINK.finditer(line):
            target = match.group(1).removeprefix("<").removesuffix(">")
            try:
                parsed = urlsplit(target)
            except ValueError:
                findings.append(Finding(path, "malformed documentation link", number))
                continue
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            relative = unquote(parsed.path)
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), relative))
            if relative.startswith("/") or resolved == ".." or resolved.startswith("../"):
                findings.append(
                    Finding(path, "documentation link escapes the release tree", number)
                )
            elif resolved not in indexed_paths and not any(
                item.startswith(resolved.rstrip("/") + "/") for item in indexed_paths
            ):
                findings.append(
                    Finding(path, "documentation link is absent from the index", number)
                )
    return findings


def check_files(
    entries: list[IndexedFile], *, max_file_bytes: int = 2_000_000, check_links: bool = True
) -> list[Finding]:
    findings: list[Finding] = []
    indexed_paths = {entry.path for entry in entries}
    if not entries:
        return [Finding("<index>", "empty candidate index; stage the release files first")]
    for entry in entries:
        path = entry.path
        parts = PurePosixPath(path).parts
        lowered = [part.lower() for part in parts]
        if (
            not parts
            or path.startswith("/")
            or ".." in parts
            or "\\" in path
            or any(ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF for character in path)
        ):
            findings.append(Finding("<invalid path>", "non-portable or unsafe index path"))
            continue
        if path not in ROOT_FILES and (len(parts) < 2 or parts[0] not in ROOT_DIRECTORIES):
            findings.append(Finding(path, "not in the public top-level allowlist"))
        if any(part in PRIVATE_COMPONENTS for part in lowered):
            findings.append(Finding(path, "private or generated artifact directory"))
        if any(part.startswith(".env") for part in lowered) and path != ".env.example":
            findings.append(Finding(path, "environment file is private"))
        if any(suffix.lower() in FORBIDDEN_SUFFIXES for suffix in PurePosixPath(path).suffixes):
            findings.append(Finding(path, "disallowed artifact extension"))
        if parts[0] == "results" and path not in PUBLIC_RESULTS:
            findings.append(Finding(path, "result artifact is not explicitly approved"))
        if parts[0] == "docs" and PRIVATE_DOCUMENT.search(parts[-1]):
            findings.append(Finding(path, "private working-note filename"))
        if entry.mode not in {"100644", "100755"}:
            findings.append(Finding(path, "symlink, submodule, or unresolved index entry"))
            continue
        if len(entry.content) > max_file_bytes:
            findings.append(Finding(path, "file exceeds the release size limit"))
            continue
        if entry.content.startswith(b"%PDF-"):
            findings.append(Finding(path, "PDF content is not a public repository artifact"))
            continue
        if path in PUBLIC_IMAGES:
            if not safe_png(entry.content):
                findings.append(Finding(path, "unsafe or metadata-bearing PNG"))
            continue
        try:
            if b"\0" in entry.content:
                raise UnicodeError
            text = entry.content.decode("utf-8")
        except UnicodeError:
            findings.append(Finding(path, "binary content is not explicitly approved"))
            continue
        if path in PUBLIC_RESULTS and not safe_public_summary(text):
            findings.append(
                Finding(path, "public result does not match the approved aggregate schema")
            )
        if any(ord(character) < 32 and character not in "\n\r\t" for character in text):
            findings.append(Finding(path, "non-text control bytes"))
        for number, line in enumerate(text.splitlines(), 1):
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                findings.append(Finding(path, "secret-shaped content", number))
            for match in HOME_PATH.finditer(line):
                if not (parts[0] == "tests" and match.group(1) in SYNTHETIC_HOME_USERS):
                    findings.append(Finding(path, "local home path", number))
            if ROOT_HOME_PATH.search(line):
                findings.append(Finding(path, "local root home path", number))
            if (
                path == ".env.example"
                and not line.lstrip().startswith("#")
                and "=" in line
                and line.partition("=")[2].strip()
            ):
                findings.append(
                    Finding(path, "environment example must leave values empty", number)
                )
        is_public_document = path.endswith(".md") and (
            len(parts) == 1 or parts[0] in {"docs", "examples"}
        )
        if check_links and is_public_document:
            findings.extend(check_public_links(path, text, indexed_paths))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository directory")
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    parser.add_argument("--no-check-links", action="store_true")
    args = parser.parse_args(argv)
    if args.max_file_bytes <= 0:
        parser.error("--max-file-bytes must be positive")
    try:
        entries = read_index(args.root)
    except (OSError, subprocess.CalledProcessError, ValueError):
        print("Unable to read the candidate Git index.", file=sys.stderr)
        return 2
    findings = check_files(
        entries, max_file_bytes=args.max_file_bytes, check_links=not args.no_check_links
    )
    for finding in findings:
        print(finding, file=sys.stderr)
    if findings:
        print(f"Public release check failed: {len(findings)} findings.", file=sys.stderr)
        return 1
    print(f"Public release check passed: {len(entries)} indexed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
