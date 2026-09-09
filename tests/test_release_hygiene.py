from __future__ import annotations

import tomllib
from pathlib import Path


def test_sdist_excludes_private_and_user_owned_runtime_artifacts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    exclusions = set(configuration["tool"]["hatch"]["build"]["exclude"])

    required = {
        "/.env",
        "/.env.*",
        "/.codex",
        "/.agents",
        "/data",
        "/runs",
        "/examples/erster_lauf",
        "/docs/.Rhistory",
        "**/*.pdf",
        "**/*.zip",
    }
    assert required <= exclusions


def test_private_review_and_annotation_roots_are_gitignored() -> None:
    project_root = Path(__file__).resolve().parents[1]
    patterns = {
        line.strip()
        for line in (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {"reviews/", "derived/", "annotations/", "references/", "vault/"} <= patterns
