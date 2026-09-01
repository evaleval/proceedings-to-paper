"""Failure-tolerant audit of every versioned corpus reference against frozen PDFs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from proceedings_to_eee.corpus import CorpusSpec
from proceedings_to_eee.evaluation.reference_audit import audit_reference_pdf
from proceedings_to_eee.io import read_json, write_json
from proceedings_to_eee.sources.manifest import SourceManifest, SourceRole, resolve_cached_path


def _project_path(project_root: Path, configured: str) -> Path:
    root = project_root.resolve()
    path = (root / configured).resolve()
    if not path.is_relative_to(root):
        raise ValueError("configured reference path escaped project root")
    return path


def audit_corpus_references(
    corpus: CorpusSpec,
    *,
    project_root: Path,
    frozen_run_root: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Audit all configured references and preserve per-paper failures."""

    results: list[dict[str, Any]] = []
    for paper in corpus.papers:
        if paper.reference_path is None:
            results.append({"paper_id": paper.paper_id, "status": "skipped_no_reference"})
            continue
        try:
            manifest_path = frozen_run_root / paper.paper_id / "source-manifest.json"
            manifest = SourceManifest.model_validate(read_json(manifest_path))
            if manifest.paper_id != paper.paper_id:
                raise ValueError("manifest paper_id mismatch")
            paper_sources = [
                source for source in manifest.sources if source.role == SourceRole.PAPER
            ]
            if len(paper_sources) != 1:
                raise ValueError("manifest must contain exactly one paper source")
            pdf_path = resolve_cached_path(paper_sources[0], project_root)
            audit = audit_reference_pdf(
                _project_path(project_root, paper.reference_path),
                pdf_path,
            )
        except Exception as error:
            results.append(
                {
                    "paper_id": paper.paper_id,
                    "status": "error",
                    "error_type": type(error).__name__,
                }
            )
            continue
        results.append(
            {
                "paper_id": paper.paper_id,
                "status": "passed" if audit.passed else "failed",
                "source_hash_matches": audit.source_hash_matches,
                "page_count": audit.page_count,
                "text_verified": audit.text_verified,
                "visual_verified": audit.visual_verified,
                "failed_evidence": audit.failed,
            }
        )
    passed = sum(item["status"] == "passed" for item in results)
    failed = sum(item["status"] in {"failed", "error"} for item in results)
    skipped = sum(item["status"] == "skipped_no_reference" for item in results)
    summary = {
        "schema_version": "corpus-reference-audit/0.1",
        "corpus_id": corpus.corpus_id,
        "status": "passed" if failed == 0 else "failed",
        "papers": len(corpus.papers),
        "papers_passed": passed,
        "papers_failed": failed,
        "papers_skipped": skipped,
        "text_verified": sum(int(item.get("text_verified", 0)) for item in results),
        "visual_verified": sum(int(item.get("visual_verified", 0)) for item in results),
        "failed_evidence": sum(int(item.get("failed_evidence", 0)) for item in results),
        "results": results,
    }
    if output_path is not None:
        write_json(output_path, summary)
    return summary
