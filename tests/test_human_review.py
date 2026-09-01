from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from proceedings_to_eee.cli import app
from proceedings_to_eee.domain.observation import (
    CandidateObservation,
    EvidenceAnchor,
    MetricSpec,
    ObservationScope,
    ReportedValue,
    RoleAssignment,
)
from proceedings_to_eee.domain.status import (
    ActorRole,
    ClaimType,
    EvidenceKind,
    ExportStatus,
    ReferentialStatus,
    TextSupportStatus,
)
from proceedings_to_eee.evaluation.human_review import (
    RISK_WEIGHTS,
    HumanReviewTemplate,
    PaperWithoutCandidatesReviewItem,
    PaperWithoutEEEReviewItem,
    RiskReason,
    build_human_review_template,
    project_paper_review_outcomes,
    summarize_human_review,
    write_human_review_artifacts,
)
from proceedings_to_eee.io import read_json, write_json, write_jsonl


def _candidate(
    paper_id: str,
    *,
    raw_name: str,
    quote: str,
    exported: bool = True,
    missing_units: bool = False,
    missing_version: bool = False,
    incomplete_table_anchor: bool = False,
    confidence: float = 0.95,
    complex_role: bool = False,
) -> CandidateObservation:
    roles = [
        RoleAssignment(
            role=ActorRole.EVALUATED_SYSTEM,
            raw_name=raw_name,
            version=None if missing_version else "v1",
            confidence=confidence,
        )
    ]
    if complex_role:
        roles.append(
            RoleAssignment(
                role=ActorRole.HUMAN_REFERENCE,
                raw_name="Human reference",
                version=None,
                confidence=0.85,
            )
        )
    unit = None if missing_units else "proportion"
    return CandidateObservation(
        paper_id=paper_id,
        claim_type=ClaimType.PRIMARY_RESULT,
        roles=roles,
        scope=ObservationScope(dataset_raw="Dataset", dataset_version=None),
        metric=MetricSpec(
            raw_name="AUC",
            canonical_id="auroc",
            unit=unit,
            lower_is_better=False,
        ),
        value=ReportedValue(raw="0.80", numeric=0.8, unit=unit),
        evidence=[
            EvidenceAnchor(
                source_id=f"source-{paper_id}",
                page=3,
                kind=EvidenceKind.TABLE,
                label=None if incomplete_table_anchor else "Table 2",
                row=None if incomplete_table_anchor else raw_name,
                column=None if incomplete_table_anchor else "AUC",
                quote=quote,
            )
        ],
        text_support=TextSupportStatus.SUPPORTED,
        referential_status=ReferentialStatus.RESOLVED,
        export_status=ExportStatus.EXPORTED if exported else ExportStatus.NEEDS_REVIEW,
        extraction_confidence=confidence,
    )


def _write_paper_run(
    root: Path,
    paper_id: str,
    candidates: list[CandidateObservation],
    *,
    eee_records: int | None = None,
) -> None:
    paper_root = root / paper_id
    write_json(
        paper_root / "run.json",
        {
            "schema_version": "pipeline-run/0.2",
            "status": "success",
            "paper_id": paper_id,
            "title": "Fixture paper",
            "counts": {
                "candidates": len(candidates),
                "eee_records": (int(bool(candidates)) if eee_records is None else eee_records),
            },
            "private_diagnostic": "Bearer raw-provider-secret /" + "Users/private/run",
        },
    )
    write_jsonl(paper_root / "observations.jsonl", candidates)


def _run_fixture(tmp_path: Path) -> tuple[Path, str]:
    run_root = tmp_path / "paper-runs"
    private_quote = "PRIVATE_EVIDENCE_QUOTE  Model A  0.80"
    _write_paper_run(
        run_root,
        "paper-a",
        [
            _candidate(
                "paper-a",
                raw_name="Model A",
                quote=private_quote,
                missing_units=True,
                missing_version=True,
                incomplete_table_anchor=True,
                confidence=0.75,
            ),
            _candidate(
                "paper-a",
                raw_name="Model A2",
                quote="Model A2  0.81",
                exported=False,
            ),
        ],
    )
    _write_paper_run(
        run_root,
        "paper-b",
        [
            _candidate(
                "paper-b",
                raw_name="Model B",
                quote="Model B  0.77",
                complex_role=True,
            )
        ],
    )
    return run_root, private_quote


def test_risk_sample_is_deterministic_covers_papers_and_renders_evidence(
    tmp_path: Path,
) -> None:
    run_root, private_quote = _run_fixture(tmp_path)

    first = build_human_review_template(run_root, sample_size=2)
    second = build_human_review_template(run_root, sample_size=2)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.population_candidates == 3
    assert {item.candidate.paper_id for item in first.items} == {"paper-a", "paper-b"}
    paper_a = next(item for item in first.items if item.candidate.paper_id == "paper-a")
    assert set(paper_a.risk_reasons) >= {
        RiskReason.EXPORTED,
        RiskReason.ROLE_VERSION_MISSING,
        RiskReason.METRIC_UNIT_MISSING,
        RiskReason.VALUE_UNIT_MISSING,
        RiskReason.EXTRACTION_CONFIDENCE_LOW,
        RiskReason.TABLE_ANCHOR,
        RiskReason.TABLE_STRUCTURE_INCOMPLETE,
    }

    template_path = tmp_path / "review" / "template.json"
    report_path = tmp_path / "review" / "review.html"
    write_human_review_artifacts(
        run_root,
        template_path=template_path,
        report_path=report_path,
        sample_size=2,
    )
    template_bytes = template_path.read_bytes()
    report_bytes = report_path.read_bytes()
    write_human_review_artifacts(
        run_root,
        template_path=template_path,
        report_path=report_path,
        sample_size=2,
    )
    assert template_path.read_bytes() == template_bytes
    assert report_path.read_bytes() == report_bytes
    assert "Local analyst review" in report_path.read_text(encoding="utf-8")
    assert "not independent human validation" in report_path.read_text(encoding="utf-8")
    assert private_quote in report_path.read_text(encoding="utf-8")
    assert str(run_root.resolve()) not in template_path.read_text(encoding="utf-8")
    assert str(run_root.resolve()) not in report_path.read_text(encoding="utf-8")


def test_completed_cli_review_emits_only_aggregate_public_summary(tmp_path: Path) -> None:
    run_root, private_quote = _run_fixture(tmp_path)
    template_path = tmp_path / "review" / "template.json"
    report_path = tmp_path / "review" / "review.html"
    summary_path = tmp_path / "public" / "summary.json"
    runner = CliRunner()

    prepared = runner.invoke(
        app,
        [
            "prepare-human-review",
            str(run_root),
            "--template",
            str(template_path),
            "--report",
            str(report_path),
            "--sample-size",
            "2",
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    assert str(run_root.resolve()) not in prepared.output
    with pytest.raises(ValueError, match="decisions are still pending"):
        summarize_human_review(template_path)

    template = read_json(template_path)
    template["items"][0]["decision"] = {
        "outcome": "confirmed",
        "issue_codes": [],
        "notes": f"private note with {private_quote} and {tmp_path.resolve()}",
    }
    template["items"][1]["decision"] = {
        "outcome": "incorrect",
        "issue_codes": ["unit", "evidence"],
        "notes": "Authorization: Bearer raw-provider-secret",
    }
    write_json(template_path, template)

    summarized = runner.invoke(
        app,
        [
            "summarize-human-review",
            str(template_path),
            "--output",
            str(summary_path),
        ],
    )
    assert summarized.exit_code == 0, summarized.output
    summary = read_json(summary_path)
    assert summary["decisions"]["outcome_counts"] == {
        "confirmed": 1,
        "incorrect": 1,
        "needs_followup": 0,
    }
    assert summary["decisions"]["issue_counts"]["unit"] == 1
    assert summary["decisions"]["issue_counts"]["evidence"] == 1
    assert summary["privacy"] == {
        "contains_candidate_payloads": False,
        "contains_evidence_quotes": False,
        "contains_local_paths": False,
        "contains_provider_raw_data": False,
        "contains_reviewer_notes": False,
    }
    public_text = summary_path.read_text(encoding="utf-8")
    for forbidden in (
        private_quote,
        "Model A",
        "raw-provider-secret",
        "Authorization",
        str(tmp_path.resolve()),
        '"candidate": {',
        '"quote"',
        '"notes"',
    ):
        assert forbidden not in public_text


def test_review_sample_covers_candidate_absence_for_every_paper(tmp_path: Path) -> None:
    run_root = tmp_path / "ten-paper-run"
    private_quote = "PRIVATE_ZERO_CANDIDATE_FIXTURE Model A 0.80"
    for index in range(10):
        paper_id = f"paper-{index:02d}"
        candidates = (
            [
                _candidate(
                    paper_id,
                    raw_name=f"Model {index}",
                    quote=private_quote if index == 0 else f"Model {index} 0.80",
                )
            ]
            if index < 2
            else []
        )
        _write_paper_run(run_root, paper_id, candidates)

    template = build_human_review_template(run_root, sample_size=10)

    assert template.population_candidates == 2
    assert template.population_papers == 10
    assert len(template.items) == 10
    paper_ids = {
        item.paper_id
        if isinstance(item, PaperWithoutCandidatesReviewItem)
        else item.candidate.paper_id
        for item in template.items
    }
    assert paper_ids == {f"paper-{index:02d}" for index in range(10)}
    absence_items = [
        item for item in template.items if isinstance(item, PaperWithoutCandidatesReviewItem)
    ]
    assert len(absence_items) == 8
    assert all(item.item_type == "paper_without_candidates" for item in absence_items)
    assert all(item.risk_reasons == [RiskReason.NO_CANDIDATES] for item in absence_items)

    template_path = tmp_path / "review" / "template.json"
    report_path = tmp_path / "review" / "review.html"
    write_human_review_artifacts(
        run_root,
        template_path=template_path,
        report_path=report_path,
        sample_size=10,
    )
    report = report_path.read_text(encoding="utf-8")
    assert report.count("No candidate observations were produced for this paper.") == 8
    assert "paper-09" in report

    payload = read_json(template_path)
    for item in payload["items"]:
        item["decision"] = {
            "outcome": "confirmed",
            "issue_codes": [],
            "notes": f"local-only {private_quote} {tmp_path.resolve()}",
        }
    write_json(template_path, payload)
    summary = summarize_human_review(template_path)

    assert summary["population"] == {
        "candidates": 2,
        "papers": 10,
        "papers_without_candidates": 8,
    }
    assert summary["sample"]["papers_reviewed"] == 10
    assert summary["sample"]["paper_coverage"] == 1.0
    assert summary["sample"]["item_type_counts"] == {
        "candidate": 2,
        "paper_without_candidates": 8,
    }
    assert summary["sample"]["papers_without_candidates_reviewed"] == 8
    assert summary["sample"]["risk_reason_counts"]["no_candidates"] == 8
    summary_text = str(summary)
    assert private_quote not in summary_text
    assert str(tmp_path.resolve()) not in summary_text
    assert "paper-09" not in summary_text


def test_review_sample_explicitly_covers_candidate_bearing_zero_eee_paper(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "paper-runs"
    private_quote = "PRIVATE_ZERO_EEE_QUOTE Model A 0.80"
    _write_paper_run(
        run_root,
        "paper-zero-eee",
        [_candidate("paper-zero-eee", raw_name="Model A", quote=private_quote)],
        eee_records=0,
    )
    _write_paper_run(
        run_root,
        "paper-with-eee",
        [_candidate("paper-with-eee", raw_name="Model B", quote="Model B 0.77")],
        eee_records=1,
    )

    template = build_human_review_template(run_root, sample_size=2)

    zero_eee = next(item for item in template.items if isinstance(item, PaperWithoutEEEReviewItem))
    assert zero_eee.item_type == "paper_without_eee"
    assert zero_eee.candidate.paper_id == "paper-zero-eee"
    assert RiskReason.NEEDS_REVIEW in zero_eee.risk_reasons
    assert zero_eee.risk_score == sum(
        {
            RiskReason.EXPORTED: 10,
            RiskReason.NEEDS_REVIEW: 9,
            RiskReason.DATASET_VERSION_MISSING: 2,
            RiskReason.TABLE_ANCHOR: 2,
        }.values()
    )

    template_path = tmp_path / "review" / "template.json"
    report_path = tmp_path / "review" / "review.html"
    write_human_review_artifacts(
        run_root,
        template_path=template_path,
        report_path=report_path,
        sample_size=2,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "paper without EEE output" in report
    assert "produced candidate observations but no EEE record" in report
    assert private_quote in report

    payload = read_json(template_path)
    for item in payload["items"]:
        item["decision"] = {
            "outcome": "confirmed",
            "issue_codes": [],
            "notes": f"local-only {private_quote} {tmp_path.resolve()}",
        }
    write_json(template_path, payload)
    summary = summarize_human_review(template_path)

    # The stable public summary treats a candidate-bearing absence as a candidate
    # review while the private template preserves the explicit output state.
    assert summary["sample"]["item_type_counts"] == {
        "candidate": 2,
        "paper_without_candidates": 0,
    }
    assert "paper-zero-eee" not in str(summary)
    assert private_quote not in str(summary)


def test_paper_review_projection_requires_complete_coverage_and_is_sanitized(
    tmp_path: Path,
) -> None:
    run_root, private_quote = _run_fixture(tmp_path)
    template_path = tmp_path / "review" / "template.json"
    template = build_human_review_template(run_root, sample_size=3).model_dump(mode="json")
    paper_a_items = [
        item for item in template["items"] if item["candidate"]["paper_id"] == "paper-a"
    ]
    paper_b_item = next(
        item for item in template["items"] if item["candidate"]["paper_id"] == "paper-b"
    )
    paper_a_items[0]["decision"] = {
        "outcome": "confirmed",
        "issue_codes": ["unit"],
        "notes": f"private {private_quote} {tmp_path.resolve()}",
    }
    paper_a_items[1]["decision"] = {
        "outcome": "needs_followup",
        "issue_codes": ["evidence"],
        "notes": "Authorization: Bearer raw-provider-secret",
    }
    paper_b_item["decision"] = {
        "outcome": "incorrect",
        "issue_codes": ["role"],
        "notes": "private reviewer note",
    }
    write_json(template_path, template)

    projection = project_paper_review_outcomes(template_path, run_root=run_root)

    assert projection == {
        "paper-a": {
            "audit_id": template["audit_id"],
            "outcome": "included_in_analyst_review",
            "decision": "withheld_in_public_artifacts",
        },
        "paper-b": {
            "audit_id": template["audit_id"],
            "outcome": "included_in_analyst_review",
            "decision": "withheld_in_public_artifacts",
        },
    }
    serialized = str(projection)
    assert private_quote not in serialized
    assert str(tmp_path.resolve()) not in serialized
    assert "raw-provider-secret" not in serialized
    assert "notes" not in serialized
    assert "confirmed" not in serialized
    assert "incorrect" not in serialized
    assert "needs_followup" not in serialized
    assert "issue_codes" not in serialized
    assert "evidence" not in serialized
    assert "unit" not in serialized
    assert "role" not in serialized

    changed_run_path = run_root / "paper-a" / "run.json"
    changed_run = read_json(changed_run_path)
    changed_run["title"] = "Changed after analyst QA"
    write_json(changed_run_path, changed_run)
    with pytest.raises(ValueError, match="does not match the current paper runs"):
        project_paper_review_outcomes(template_path, run_root=run_root)

    partial = build_human_review_template(run_root, sample_size=1).model_dump(mode="json")
    partial["items"][0]["decision"] = {
        "outcome": "confirmed",
        "issue_codes": [],
        "notes": None,
    }
    write_json(template_path, partial)
    with pytest.raises(ValueError, match="does not cover every paper"):
        project_paper_review_outcomes(template_path)


def test_zero_eee_review_item_cannot_be_assigned_to_paper_with_eee(tmp_path: Path) -> None:
    run_root, _ = _run_fixture(tmp_path)
    payload = build_human_review_template(run_root, sample_size=2).model_dump(mode="json")
    payload["items"][0]["item_type"] = "paper_without_eee"
    payload["items"][0]["risk_reasons"] = sorted(
        {*payload["items"][0]["risk_reasons"], "needs_review"}
    )
    payload["items"][0]["risk_score"] = sum(
        RISK_WEIGHTS[RiskReason(reason)] for reason in payload["items"][0]["risk_reasons"]
    )

    with pytest.raises(ValueError, match="paper_without_eee requires zero EEE records"):
        HumanReviewTemplate.model_validate(payload)


def test_legacy_candidate_review_item_without_item_type_still_loads(tmp_path: Path) -> None:
    run_root, _ = _run_fixture(tmp_path)
    payload = build_human_review_template(run_root, sample_size=2).model_dump(mode="json")
    for item in payload["items"]:
        item.pop("item_type")
    for artifact in payload["source_artifacts"]:
        artifact.pop("eee_record_count")

    restored = HumanReviewTemplate.model_validate(payload)

    assert all(item.item_type == "candidate" for item in restored.items)


def test_candidate_review_item_rejects_duplicate_risk_reasons(tmp_path: Path) -> None:
    run_root, _ = _run_fixture(tmp_path)
    payload = build_human_review_template(run_root, sample_size=1).model_dump(mode="json")
    payload["items"][0]["risk_reasons"] = ["exported", "exported"]

    with pytest.raises(ValueError, match="risk_reasons must be unique"):
        HumanReviewTemplate.model_validate(payload)
