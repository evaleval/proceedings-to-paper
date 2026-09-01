from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from proceedings_to_eee.corpus import CorpusSpec, PaperSpec
from proceedings_to_eee.evaluation.corpus_audit import audit_corpus_references
from proceedings_to_eee.evaluation.reference_audit import ReferenceAudit
from proceedings_to_eee.io import write_json
from proceedings_to_eee.sources.manifest import FrozenSource, SourceManifest, SourceRole


def _paper(paper_id: str, *, reference: bool = True) -> PaperSpec:
    return PaperSpec(
        paper_id=paper_id,
        title=paper_id,
        year=2025,
        venue="ACM Test",
        pdf_url=f"https://example.org/{paper_id}.pdf",
        perspective_role="evaluated_system",
        reference_path=f"references/{paper_id}.yaml" if reference else None,
    )


def _freeze(project_root: Path, frozen_root: Path, paper: PaperSpec) -> None:
    content = b"%PDF-fixture"
    digest = hashlib.sha256(content).hexdigest()
    cache_path = project_root / "data" / "sources" / digest[:2] / f"{digest}.pdf"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)
    manifest = SourceManifest(
        paper_id=paper.paper_id,
        title=paper.title,
        sources=[
            FrozenSource(
                source_id=f"src_{paper.paper_id}",
                paper_id=paper.paper_id,
                role=SourceRole.PAPER,
                original_uri=str(paper.pdf_url),
                retrieved_at=datetime(2026, 8, 4, tzinfo=UTC),
                sha256=digest,
                byte_size=len(content),
                media_type="application/pdf",
                cache_relpath=cache_path.relative_to(project_root).as_posix(),
            )
        ],
    )
    write_json(frozen_root / paper.paper_id / "source-manifest.json", manifest)


def test_corpus_reference_audit_continues_and_aggregates(monkeypatch, tmp_path: Path) -> None:
    first, second, skipped = _paper("first"), _paper("second"), _paper("skip", reference=False)
    corpus = CorpusSpec(
        corpus_id="audit-fixture",
        description="fixture",
        papers=[first, second, skipped],
    )
    frozen_root = tmp_path / "runs"
    _freeze(tmp_path, frozen_root, first)
    _freeze(tmp_path, frozen_root, second)
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "first.yaml").write_text("fixture", encoding="utf-8")
    (tmp_path / "references" / "second.yaml").write_text("fixture", encoding="utf-8")

    def fake_audit(reference_path: Path, pdf_path: Path) -> ReferenceAudit:
        del pdf_path
        paper_id = reference_path.stem
        return ReferenceAudit(
            paper_id=paper_id,
            source_hash_matches=True,
            page_count=2,
            text_verified=3,
            visual_verified=1,
            failed=0 if paper_id == "first" else 1,
            passed=paper_id == "first",
            items=[],
        )

    monkeypatch.setattr(
        "proceedings_to_eee.evaluation.corpus_audit.audit_reference_pdf", fake_audit
    )
    output = tmp_path / "audit.json"

    result = audit_corpus_references(
        corpus,
        project_root=tmp_path,
        frozen_run_root=frozen_root,
        output_path=output,
    )

    assert result["status"] == "failed"
    assert result["papers_passed"] == 1
    assert result["papers_failed"] == 1
    assert result["papers_skipped"] == 1
    assert result["text_verified"] == 6
    assert result["visual_verified"] == 2
    assert output.exists()
