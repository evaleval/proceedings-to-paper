"""Content-addressed source freeze manifests."""

from __future__ import annotations

import hashlib
import mimetypes
import re
import shutil
import subprocess
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import Field, HttpUrl, model_validator

from proceedings_to_eee.domain.observation import StrictModel
from proceedings_to_eee.io import sha256_file


class SourceRole(StrEnum):
    PAPER = "paper"
    SUPPLEMENT = "supplement"
    REPOSITORY = "repository"
    PROCEEDINGS_RECORD = "proceedings_record"


class AccessStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    EXPIRED = "expired"
    RESTRICTED = "restricted"


class LicenseDisposition(StrEnum):
    PRIVATE_USE_ONLY = "private_use_only"
    DERIVED_METADATA_ONLY = "derived_metadata_only"
    REDISTRIBUTABLE = "redistributable"
    UNKNOWN = "unknown"


class FrozenSource(StrictModel):
    """One exact byte source used by a reproducible run."""

    source_id: str
    paper_id: str
    role: SourceRole
    original_uri: str
    resolved_uri: str | None = None
    retrieved_at: datetime
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    byte_size: int | None = Field(default=None, ge=0)
    media_type: str | None = None
    cache_relpath: str | None = None
    git_commit: str | None = None
    access_status: AccessStatus = AccessStatus.AVAILABLE
    license_disposition: LicenseDisposition = LicenseDisposition.UNKNOWN
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def available_sources_are_content_addressed(self) -> FrozenSource:
        if (
            self.access_status == AccessStatus.AVAILABLE
            and self.role != SourceRole.REPOSITORY
            and (not self.sha256 or self.byte_size is None)
        ):
            raise ValueError("available byte source requires sha256 and byte_size")
        if (
            self.role == SourceRole.REPOSITORY
            and self.access_status == AccessStatus.AVAILABLE
            and not self.git_commit
        ):
            raise ValueError("available repository source requires immutable git_commit")
        return self


class SourceManifest(StrictModel):
    """Exact inputs for one paper bundle."""

    schema_version: str = "source-manifest/0.2"
    paper_id: str
    title: str
    doi: str | None = None
    arxiv_id: str | None = None
    proceedings_url: HttpUrl | None = None
    sources: list[FrozenSource] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique_and_bound(self) -> SourceManifest:
        ids = [source.source_id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source_id values must be unique")
        if any(source.paper_id != self.paper_id for source in self.sources):
            raise ValueError("all sources must share manifest paper_id")
        return self


def _source_id(paper_id: str, role: SourceRole, digest: str) -> str:
    identity = f"{paper_id}\0{role}\0{digest}".encode()
    return "src_" + hashlib.sha256(identity).hexdigest()[:20]


def freeze_local_source(
    *,
    paper_id: str,
    role: SourceRole,
    path: Path,
    cache_root: Path,
    original_uri: str | None = None,
    license_disposition: LicenseDisposition = LicenseDisposition.DERIVED_METADATA_ONLY,
) -> FrozenSource:
    """Hash and copy one local file into a content-addressed, gitignored cache."""

    resolved = path.resolve(strict=True)
    digest = sha256_file(resolved)
    suffix = resolved.suffix.lower()
    destination = cache_root / digest[:2] / f"{digest}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(resolved, destination)
    if sha256_file(destination) != digest:
        raise OSError("cached source hash mismatch")
    relpath = destination.relative_to(cache_root.parents[1]).as_posix()
    return FrozenSource(
        source_id=_source_id(paper_id, role, digest),
        paper_id=paper_id,
        role=role,
        original_uri=original_uri or resolved.as_uri(),
        resolved_uri=resolved.as_uri(),
        retrieved_at=datetime.now(UTC),
        sha256=digest,
        byte_size=resolved.stat().st_size,
        media_type=mimetypes.guess_type(resolved.name)[0] or "application/octet-stream",
        cache_relpath=relpath,
        license_disposition=license_disposition,
    )


def download_and_freeze_source(
    *,
    paper_id: str,
    role: SourceRole,
    url: str,
    cache_root: Path,
    timeout_seconds: float = 90.0,
    license_disposition: LicenseDisposition = LicenseDisposition.DERIVED_METADATA_ONLY,
) -> FrozenSource:
    """Download a public source, then pin final URL, bytes, time, and digest."""

    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("source URL must be absolute HTTP(S)")
    with httpx.Client(follow_redirects=True, timeout=timeout_seconds) as client:
        response = client.get(url, headers={"User-Agent": "Proceedings-to-EEE/0.2"})
        response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
    digest = hashlib.sha256(response.content).hexdigest()
    suffix = ".pdf" if content_type == "application/pdf" or url.endswith(".pdf") else ".bin"
    destination = cache_root / digest[:2] / f"{digest}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(response.content)
    if sha256_file(destination) != digest:
        raise OSError("cached source hash mismatch")
    return FrozenSource(
        source_id=_source_id(paper_id, role, digest),
        paper_id=paper_id,
        role=role,
        original_uri=url,
        resolved_uri=str(response.url),
        retrieved_at=datetime.now(UTC),
        sha256=digest,
        byte_size=len(response.content),
        media_type=content_type or mimetypes.guess_type(url)[0] or "application/octet-stream",
        cache_relpath=destination.relative_to(cache_root.parents[1]).as_posix(),
        license_disposition=license_disposition,
    )


def freeze_repository_source(
    *,
    paper_id: str,
    url: str,
    git_commit: str,
    timeout_seconds: float = 90.0,
    license_disposition: LicenseDisposition = LicenseDisposition.UNKNOWN,
) -> FrozenSource:
    """Verify and pin one immutable commit from a public Git repository."""

    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("repository URL must be absolute HTTP(S)")
    commit = git_commit.casefold()
    if re.fullmatch(r"[a-f0-9]{40}", commit) is None:
        raise ValueError("repository commit must be a full 40-character SHA-1")
    result = subprocess.run(
        ["git", "ls-remote", url],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    advertised_commits = {
        line.split(maxsplit=1)[0].casefold() for line in result.stdout.splitlines() if line.strip()
    }
    if commit not in advertised_commits:
        raise ValueError("configured repository commit is not advertised by the remote")
    return FrozenSource(
        source_id=_source_id(paper_id, SourceRole.REPOSITORY, commit),
        paper_id=paper_id,
        role=SourceRole.REPOSITORY,
        original_uri=url,
        resolved_uri=url,
        retrieved_at=datetime.now(UTC),
        git_commit=commit,
        media_type="application/x-git",
        access_status=AccessStatus.AVAILABLE,
        license_disposition=license_disposition,
    )


def resolve_cached_path(source: FrozenSource, project_root: Path) -> Path:
    if not source.cache_relpath:
        raise ValueError(f"source {source.source_id} has no cached bytes")
    path = (project_root / source.cache_relpath).resolve()
    expected_root = (project_root / "data").resolve()
    if not path.is_relative_to(expected_root):
        raise ValueError("source cache path escaped project data directory")
    if sha256_file(path) != source.sha256:
        raise OSError(f"source hash drift for {source.source_id}")
    return path
