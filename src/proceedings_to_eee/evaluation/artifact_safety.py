"""Fail-closed path guards for staged provider and offline evaluation artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def _existing_regular_identity(path: Path) -> tuple[int, int] | None:
    """Return a stable device/inode identity without accepting links or directories."""

    if path.is_symlink():
        raise ValueError("evaluation artifact path must not be a symbolic link")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("evaluation artifact path must be a regular file")
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def assert_artifact_paths_safe(
    artifact_paths: Iterable[Path],
    *,
    sealed_root: Path | None = None,
    protected_paths: Iterable[Path] = (),
) -> None:
    """Reject outputs that could overwrite, alias, or live inside immutable inputs.

    Resolution catches aliases through symlinked parent directories. Existing leaf
    symlinks and hard links are rejected explicitly. The check runs before any paid
    provider call or artifact write.
    """

    artifacts = [Path(path) for path in artifact_paths]
    protected = [Path(path) for path in protected_paths]
    root = sealed_root.resolve(strict=True) if sealed_root is not None else None

    resolved_artifacts: list[Path] = []
    artifact_identities: list[tuple[int, int] | None] = []
    for path in artifacts:
        identity = _existing_regular_identity(path)
        resolved = path.resolve(strict=False)
        if root is not None and (resolved == root or resolved.is_relative_to(root)):
            raise ValueError("evaluation artifacts must stay outside the sealed run")
        resolved_artifacts.append(resolved)
        artifact_identities.append(identity)

    if len(resolved_artifacts) != len(set(resolved_artifacts)):
        raise ValueError("evaluation artifact paths must be distinct")
    nonnull_identities = [item for item in artifact_identities if item is not None]
    if len(nonnull_identities) != len(set(nonnull_identities)):
        raise ValueError("evaluation artifact paths must not be hard-link aliases")

    protected_resolved: set[Path] = set()
    protected_identities: set[tuple[int, int]] = set()
    for path in protected:
        protected_resolved.add(path.resolve(strict=False))
        if path.exists():
            stat = path.stat()
            protected_identities.add((stat.st_dev, stat.st_ino))

    if root is not None:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("sealed run contains a symbolic link")
            if path.is_file():
                stat = path.stat()
                protected_identities.add((stat.st_dev, stat.st_ino))

    for resolved, identity in zip(resolved_artifacts, artifact_identities, strict=True):
        if resolved in protected_resolved or (
            identity is not None and identity in protected_identities
        ):
            raise ValueError("evaluation artifact path aliases an immutable input")
