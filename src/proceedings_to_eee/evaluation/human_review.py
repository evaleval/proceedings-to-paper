"""Deterministic, offline human-review sampling over completed paper runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import (
    ActorRole,
    EvidenceKind,
    ExportStatus,
    ReferentialStatus,
    TextSupportStatus,
    ValueComparator,
)
from proceedings_to_eee.io import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_file,
    write_json,
)

SAMPLING_POLICY = "risk-stratified-paper-coverage/0.1"


class ReviewModel(BaseModel):
    """Strict base for editable review artifacts."""

    model_config = ConfigDict(extra="forbid")


class RiskReason(StrEnum):
    """Public-safe reason why an observation merits manual inspection."""

    EXPORTED = "exported"
    ELIGIBLE = "eligible"
    NEEDS_REVIEW = "needs_review"
    TEXT_SUPPORT_RISK = "text_support_risk"
    REFERENTIAL_RISK = "referential_risk"
    ROLES_MISSING = "roles_missing"
    ROLE_VERSION_MISSING = "role_version_missing"
    ROLE_CONFIDENCE_LOW = "role_confidence_low"
    COMPLEX_ROLE_ASSIGNMENT = "complex_role_assignment"
    METRIC_UNIT_MISSING = "metric_unit_missing"
    VALUE_UNIT_MISSING = "value_unit_missing"
    UNIT_MISMATCH = "unit_mismatch"
    DATASET_VERSION_MISSING = "dataset_version_missing"
    EXTRACTION_CONFIDENCE_LOW = "extraction_confidence_low"
    TABLE_ANCHOR = "table_anchor"
    TABLE_STRUCTURE_INCOMPLETE = "table_structure_incomplete"
    NON_EXACT_VALUE = "non_exact_value"
    NO_CANDIDATES = "no_candidates"


RISK_WEIGHTS: dict[RiskReason, int] = {
    RiskReason.EXPORTED: 10,
    RiskReason.ELIGIBLE: 8,
    RiskReason.NEEDS_REVIEW: 9,
    RiskReason.TEXT_SUPPORT_RISK: 10,
    RiskReason.REFERENTIAL_RISK: 8,
    RiskReason.ROLES_MISSING: 10,
    RiskReason.ROLE_VERSION_MISSING: 3,
    RiskReason.ROLE_CONFIDENCE_LOW: 5,
    RiskReason.COMPLEX_ROLE_ASSIGNMENT: 5,
    RiskReason.METRIC_UNIT_MISSING: 7,
    RiskReason.VALUE_UNIT_MISSING: 7,
    RiskReason.UNIT_MISMATCH: 10,
    RiskReason.DATASET_VERSION_MISSING: 2,
    RiskReason.EXTRACTION_CONFIDENCE_LOW: 6,
    RiskReason.TABLE_ANCHOR: 2,
    RiskReason.TABLE_STRUCTURE_INCOMPLETE: 6,
    RiskReason.NON_EXACT_VALUE: 4,
    RiskReason.NO_CANDIDATES: 10,
}


class ReviewOutcome(StrEnum):
    """Editable disposition for one sampled observation."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    INCORRECT = "incorrect"
    NEEDS_FOLLOWUP = "needs_followup"


class ReviewIssue(StrEnum):
    """Aggregate-safe issue taxonomy; free-text notes remain local only."""

    CLAIM_TYPE = "claim_type"
    ROLE = "role"
    VERSION = "version"
    SCOPE = "scope"
    METRIC = "metric"
    UNIT = "unit"
    VALUE = "value"
    EVIDENCE = "evidence"
    EXPORT_DECISION = "export_decision"
    DUPLICATE = "duplicate"
    OTHER = "other"


class ReviewDecision(ReviewModel):
    outcome: ReviewOutcome = ReviewOutcome.PENDING
    issue_codes: list[ReviewIssue] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def unique_issue_codes(self) -> ReviewDecision:
        if len(self.issue_codes) != len(set(self.issue_codes)):
            raise ValueError("issue_codes must be unique")
        return self


class ReviewSourceArtifact(ReviewModel):
    """Path-free fingerprint of one paper run used to construct the audit."""

    paper_id: str
    run_status: str
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_count: int = Field(ge=0)
    eee_record_count: int | None = Field(default=None, ge=0)


class HumanReviewItem(ReviewModel):
    item_type: Literal["candidate"] = "candidate"
    review_id: str = Field(pattern=r"^review_[0-9a-f]{16}$")
    risk_score: int = Field(ge=0)
    risk_reasons: list[RiskReason]
    candidate: CandidateObservation
    decision: ReviewDecision = Field(default_factory=ReviewDecision)

    @model_validator(mode="after")
    def unique_risk_reasons(self) -> HumanReviewItem:
        if len(self.risk_reasons) != len(set(self.risk_reasons)):
            raise ValueError("risk_reasons must be unique")
        return self


class PaperWithoutCandidatesReviewItem(ReviewModel):
    """Reviewable absence for a completed paper that yielded no candidates."""

    item_type: Literal["paper_without_candidates"] = "paper_without_candidates"
    review_id: str = Field(pattern=r"^review_[0-9a-f]{16}$")
    paper_id: str = Field(min_length=1)
    risk_score: int = Field(default=RISK_WEIGHTS[RiskReason.NO_CANDIDATES], ge=0)
    risk_reasons: list[RiskReason] = Field(default_factory=lambda: [RiskReason.NO_CANDIDATES])
    decision: ReviewDecision = Field(default_factory=ReviewDecision)

    @model_validator(mode="after")
    def absence_risk_is_explicit(self) -> PaperWithoutCandidatesReviewItem:
        if self.risk_reasons != [RiskReason.NO_CANDIDATES]:
            raise ValueError("paper_without_candidates requires only the no_candidates risk")
        if self.risk_score != RISK_WEIGHTS[RiskReason.NO_CANDIDATES]:
            raise ValueError("paper_without_candidates risk_score does not match policy")
        return self


class PaperWithoutEEEReviewItem(ReviewModel):
    """Candidate-bearing review item for a paper that produced no EEE record."""

    item_type: Literal["paper_without_eee"] = "paper_without_eee"
    review_id: str = Field(pattern=r"^review_[0-9a-f]{16}$")
    risk_score: int = Field(ge=0)
    risk_reasons: list[RiskReason]
    candidate: CandidateObservation
    decision: ReviewDecision = Field(default_factory=ReviewDecision)

    @model_validator(mode="after")
    def explicit_output_absence(self) -> PaperWithoutEEEReviewItem:
        if len(self.risk_reasons) != len(set(self.risk_reasons)):
            raise ValueError("risk_reasons must be unique")
        if RiskReason.NEEDS_REVIEW not in self.risk_reasons:
            raise ValueError("paper_without_eee requires the needs_review risk")
        expected_score = sum(RISK_WEIGHTS[reason] for reason in self.risk_reasons)
        if self.risk_score != expected_score:
            raise ValueError("paper_without_eee risk_score does not match policy")
        return self


HumanReviewEntry = HumanReviewItem | PaperWithoutCandidatesReviewItem | PaperWithoutEEEReviewItem


class HumanReviewTemplate(ReviewModel):
    """Local-only editable template; it deliberately contains source evidence quotes."""

    schema_version: Literal["human-review-template/0.1"] = "human-review-template/0.1"
    visibility: Literal["local-review-only"] = "local-review-only"
    audit_id: str = Field(pattern=r"^audit_[0-9a-f]{20}$")
    sampling_policy: Literal["risk-stratified-paper-coverage/0.1"] = SAMPLING_POLICY
    sample_requested: int = Field(ge=1)
    population_candidates: int = Field(ge=0)
    population_papers: int = Field(ge=1)
    source_artifacts: list[ReviewSourceArtifact] = Field(min_length=1)
    items: list[HumanReviewEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts_and_ids(self) -> HumanReviewTemplate:
        if self.population_candidates != sum(
            artifact.candidate_count for artifact in self.source_artifacts
        ):
            raise ValueError("population_candidates does not match source artifacts")
        if self.population_papers != len(self.source_artifacts):
            raise ValueError("population_papers does not match source artifacts")
        review_ids = [item.review_id for item in self.items]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("review IDs must be unique")
        artifacts = {artifact.paper_id: artifact for artifact in self.source_artifacts}
        if len(artifacts) != len(self.source_artifacts):
            raise ValueError("source artifact paper IDs must be unique")
        for item in self.items:
            paper_id = _review_item_paper_id(item)
            if paper_id not in artifacts:
                raise ValueError("review item paper_id has no source artifact")
            if (
                isinstance(item, PaperWithoutCandidatesReviewItem)
                and artifacts[paper_id].candidate_count != 0
            ):
                raise ValueError("absence review item requires a zero-candidate source artifact")
            if isinstance(item, PaperWithoutEEEReviewItem):
                artifact = artifacts[paper_id]
                if artifact.candidate_count == 0:
                    raise ValueError("paper_without_eee requires candidate observations")
                if artifact.eee_record_count != 0:
                    raise ValueError("paper_without_eee requires zero EEE records")
                if item.candidate.paper_id != paper_id:
                    raise ValueError("paper_without_eee candidate paper_id does not match")
        reviewed_papers = {_review_item_paper_id(item) for item in self.items}
        for paper_id in reviewed_papers:
            artifact = artifacts[paper_id]
            if artifact.candidate_count == 0 and not any(
                isinstance(item, PaperWithoutCandidatesReviewItem) and item.paper_id == paper_id
                for item in self.items
            ):
                raise ValueError("reviewed zero-candidate paper requires an explicit absence item")
            if (
                artifact.candidate_count > 0
                and artifact.eee_record_count == 0
                and not any(
                    isinstance(item, PaperWithoutEEEReviewItem)
                    and item.candidate.paper_id == paper_id
                    for item in self.items
                )
            ):
                raise ValueError("reviewed zero-EEE paper requires an explicit absence item")
        return self


def _observation_paths(run_root: Path) -> list[Path]:
    if not run_root.is_dir():
        raise ValueError("run root must be a directory")
    paths: list[Path] = []
    direct = run_root / "observations.jsonl"
    if direct.is_file():
        paths.append(direct)
    paths.extend(
        child / "observations.jsonl"
        for child in sorted(run_root.iterdir(), key=lambda item: item.name)
        if child.is_dir() and (child / "observations.jsonl").is_file()
    )
    if not paths:
        raise ValueError("run root contains no paper observations.jsonl artifacts")
    return paths


def _load_observations(path: Path) -> list[CandidateObservation]:
    observations: list[CandidateObservation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            observations.append(CandidateObservation.model_validate(payload))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid observations.jsonl record at line {line_number}") from error
    return observations


def _load_run_artifacts(
    run_root: Path,
) -> tuple[list[ReviewSourceArtifact], list[CandidateObservation]]:
    artifacts: list[ReviewSourceArtifact] = []
    candidates: list[CandidateObservation] = []
    seen_observation_ids: set[str] = set()
    for observations_path in _observation_paths(run_root):
        run_path = observations_path.parent / "run.json"
        if not run_path.is_file():
            raise ValueError("every observations.jsonl artifact requires an adjacent run.json")
        run_payload = json.loads(run_path.read_text(encoding="utf-8"))
        if not isinstance(run_payload, dict):
            raise ValueError("run.json must contain one JSON object")
        paper_id = run_payload.get("paper_id")
        run_status = run_payload.get("status")
        if not isinstance(paper_id, str) or not paper_id:
            raise ValueError("run.json requires a non-empty paper_id")
        if not isinstance(run_status, str) or not run_status:
            raise ValueError("run.json requires a non-empty status")
        counts = run_payload.get("counts", {})
        if not isinstance(counts, dict):
            raise ValueError("run.json counts must be a JSON object")
        eee_record_count = counts.get("eee_records")
        if eee_record_count is None:
            eee_root = observations_path.parent / "eee"
            eee_record_count = (
                sum(path.is_file() and not path.is_symlink() for path in eee_root.glob("*.json"))
                if eee_root.is_dir()
                else 0
            )
        if (
            not isinstance(eee_record_count, int)
            or isinstance(eee_record_count, bool)
            or eee_record_count < 0
        ):
            raise ValueError("run.json counts.eee_records must be a non-negative integer")
        paper_candidates = _load_observations(observations_path)
        for candidate in paper_candidates:
            if candidate.paper_id != paper_id:
                raise ValueError("candidate paper_id does not match adjacent run.json")
            assert candidate.observation_id is not None
            if candidate.observation_id in seen_observation_ids:
                raise ValueError("duplicate observation_id across paper-run artifacts")
            seen_observation_ids.add(candidate.observation_id)
        artifacts.append(
            ReviewSourceArtifact(
                paper_id=paper_id,
                run_status=run_status,
                run_sha256=sha256_file(run_path),
                observations_sha256=sha256_file(observations_path),
                candidate_count=len(paper_candidates),
                eee_record_count=eee_record_count,
            )
        )
        candidates.extend(paper_candidates)
    artifacts.sort(key=lambda artifact: artifact.paper_id)
    return artifacts, candidates


def candidate_risk(candidate: CandidateObservation) -> tuple[int, list[RiskReason]]:
    """Return a stable review-priority score without inferring missing facts."""

    reasons: set[RiskReason] = set()
    if candidate.export_status == ExportStatus.EXPORTED:
        reasons.add(RiskReason.EXPORTED)
    elif candidate.export_status == ExportStatus.ELIGIBLE:
        reasons.add(RiskReason.ELIGIBLE)
    elif candidate.export_status == ExportStatus.NEEDS_REVIEW:
        reasons.add(RiskReason.NEEDS_REVIEW)
    if candidate.text_support != TextSupportStatus.SUPPORTED:
        reasons.add(RiskReason.TEXT_SUPPORT_RISK)
    if candidate.referential_status != ReferentialStatus.RESOLVED:
        reasons.add(RiskReason.REFERENTIAL_RISK)
    if not candidate.roles:
        reasons.add(RiskReason.ROLES_MISSING)
    if any(role.version is None for role in candidate.roles):
        reasons.add(RiskReason.ROLE_VERSION_MISSING)
    if any(role.confidence < 0.9 for role in candidate.roles):
        reasons.add(RiskReason.ROLE_CONFIDENCE_LOW)
    if any(
        role.role
        in {ActorRole.EVALUATION_INSTRUMENT, ActorRole.LABEL_GENERATOR, ActorRole.HUMAN_REFERENCE}
        for role in candidate.roles
    ):
        reasons.add(RiskReason.COMPLEX_ROLE_ASSIGNMENT)
    if candidate.metric is not None and candidate.metric.unit is None:
        reasons.add(RiskReason.METRIC_UNIT_MISSING)
    if candidate.value is not None and candidate.value.unit is None:
        reasons.add(RiskReason.VALUE_UNIT_MISSING)
    if (
        candidate.metric is not None
        and candidate.value is not None
        and candidate.metric.unit is not None
        and candidate.value.unit is not None
        and candidate.metric.unit != candidate.value.unit
    ):
        reasons.add(RiskReason.UNIT_MISMATCH)
    if candidate.scope is not None and candidate.scope.dataset_version is None:
        reasons.add(RiskReason.DATASET_VERSION_MISSING)
    if candidate.extraction_confidence < 0.9:
        reasons.add(RiskReason.EXTRACTION_CONFIDENCE_LOW)
    table_anchors = [anchor for anchor in candidate.evidence if anchor.kind == EvidenceKind.TABLE]
    if table_anchors:
        reasons.add(RiskReason.TABLE_ANCHOR)
    if any(
        anchor.label is None or anchor.row is None or anchor.column is None
        for anchor in table_anchors
    ):
        reasons.add(RiskReason.TABLE_STRUCTURE_INCOMPLETE)
    if candidate.value is not None and candidate.value.comparator != ValueComparator.EXACT:
        reasons.add(RiskReason.NON_EXACT_VALUE)
    ordered = sorted(reasons, key=str)
    return sum(RISK_WEIGHTS[reason] for reason in ordered), ordered


def _rank_key(candidate: CandidateObservation) -> tuple[int, str, str]:
    score, _ = candidate_risk(candidate)
    assert candidate.observation_id is not None
    return (-score, candidate.paper_id, candidate.observation_id)


ReviewSubject = CandidateObservation | ReviewSourceArtifact


def _subject_rank_key(subject: ReviewSubject) -> tuple[int, str, str]:
    if isinstance(subject, CandidateObservation):
        return _rank_key(subject)
    return (-RISK_WEIGHTS[RiskReason.NO_CANDIDATES], subject.paper_id, "")


def _sample_review_subjects(
    artifacts: list[ReviewSourceArtifact],
    candidates: list[CandidateObservation],
    sample_size: int,
) -> list[ReviewSubject]:
    """Cover every paper when capacity permits, including explicit absences."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    ranked = sorted(candidates, key=_rank_key)
    by_paper: dict[str, list[CandidateObservation]] = defaultdict(list)
    for candidate in ranked:
        by_paper[candidate.paper_id].append(candidate)
    representatives: list[ReviewSubject] = [
        by_paper[artifact.paper_id][0] if by_paper[artifact.paper_id] else artifact
        for artifact in artifacts
    ]
    representatives.sort(key=_subject_rank_key)
    selected = representatives[: min(sample_size, len(representatives))]
    selected_ids = {
        subject.observation_id for subject in selected if isinstance(subject, CandidateObservation)
    }
    reviewable_population = len(candidates) + sum(
        artifact.candidate_count == 0 for artifact in artifacts
    )
    for candidate in ranked:
        if len(selected) >= min(sample_size, reviewable_population):
            break
        if candidate.observation_id not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate.observation_id)
    return sorted(selected, key=_subject_rank_key)


def _review_item_paper_id(item: HumanReviewEntry) -> str:
    if isinstance(item, PaperWithoutCandidatesReviewItem):
        return item.paper_id
    return item.candidate.paper_id


def build_human_review_template(
    run_root: Path,
    *,
    sample_size: int = 20,
) -> HumanReviewTemplate:
    """Build a deterministic, evidence-bearing local review template."""

    artifacts, candidates = _load_run_artifacts(run_root)
    sample = _sample_review_subjects(artifacts, candidates, sample_size)
    audit_basis = {
        "sampling_policy": SAMPLING_POLICY,
        "sample_requested": sample_size,
        "source_artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
    }
    audit_digest = hashlib.sha256(canonical_json_bytes(audit_basis)).hexdigest()
    artifacts_by_paper = {artifact.paper_id: artifact for artifact in artifacts}
    zero_eee_items_created: set[str] = set()
    items: list[HumanReviewEntry] = []
    for subject in sample:
        if isinstance(subject, ReviewSourceArtifact):
            review_digest = hashlib.sha256(
                f"{subject.paper_id}:paper_without_candidates".encode()
            ).hexdigest()
            items.append(
                PaperWithoutCandidatesReviewItem(
                    review_id=f"review_{review_digest[:16]}",
                    paper_id=subject.paper_id,
                )
            )
            continue
        candidate = subject
        assert candidate.observation_id is not None
        review_digest = hashlib.sha256(
            f"{candidate.paper_id}:{candidate.observation_id}".encode()
        ).hexdigest()
        score, reasons = candidate_risk(candidate)
        artifact = artifacts_by_paper[candidate.paper_id]
        if artifact.eee_record_count == 0 and candidate.paper_id not in zero_eee_items_created:
            zero_eee_reasons = sorted({*reasons, RiskReason.NEEDS_REVIEW}, key=str)
            review_digest = hashlib.sha256(
                f"{candidate.paper_id}:paper_without_eee:{candidate.observation_id}".encode()
            ).hexdigest()
            items.append(
                PaperWithoutEEEReviewItem(
                    review_id=f"review_{review_digest[:16]}",
                    risk_score=sum(RISK_WEIGHTS[reason] for reason in zero_eee_reasons),
                    risk_reasons=zero_eee_reasons,
                    candidate=candidate,
                )
            )
            zero_eee_items_created.add(candidate.paper_id)
            continue
        items.append(
            HumanReviewItem(
                review_id=f"review_{review_digest[:16]}",
                risk_score=score,
                risk_reasons=reasons,
                candidate=candidate,
            )
        )
    return HumanReviewTemplate(
        audit_id=f"audit_{audit_digest[:20]}",
        sample_requested=sample_size,
        population_candidates=len(candidates),
        population_papers=len(artifacts),
        source_artifacts=artifacts,
        items=items,
    )


def write_human_review_artifacts(
    run_root: Path,
    *,
    template_path: Path,
    report_path: Path,
    sample_size: int = 20,
) -> HumanReviewTemplate:
    """Write an editable local template and a quote-bearing HTML review report."""

    from proceedings_to_eee.reporting.human_review_html import render_human_review_html

    template = build_human_review_template(run_root, sample_size=sample_size)
    write_json(template_path, template)
    html = render_human_review_html(template, decision_file_name=template_path.name)
    atomic_write_bytes(report_path, html.encode("utf-8"))
    return template


def project_paper_review_outcomes(
    template_path: Path,
    *,
    run_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Project privacy-safe per-paper review coverage for public cards.

    The editable template and every individual decision remain private. Public cards
    receive only a statement that the paper was included in analyst review and that
    its decision is withheld. Decision totals and issue counts belong exclusively in
    the aggregate public summary. Partial coverage is rejected so callers cannot
    silently publish a paper-level QA claim without a completed review item.
    """

    template = HumanReviewTemplate.model_validate(
        json.loads(template_path.read_text(encoding="utf-8"))
    )
    if run_root is not None:
        current_artifacts, _ = _load_run_artifacts(run_root)
        if current_artifacts != template.source_artifacts:
            raise ValueError("human-review template does not match the current paper runs")
    pending = sum(item.decision.outcome == ReviewOutcome.PENDING for item in template.items)
    if pending:
        raise ValueError(f"{pending} human-review decisions are still pending")
    expected_papers = {artifact.paper_id for artifact in template.source_artifacts}
    items_by_paper: dict[str, list[HumanReviewEntry]] = defaultdict(list)
    for item in template.items:
        items_by_paper[_review_item_paper_id(item)].append(item)
    missing = sorted(expected_papers - set(items_by_paper))
    if missing:
        raise ValueError("human-review template does not cover every paper: " + ", ".join(missing))

    projected: dict[str, dict[str, Any]] = {}
    for paper_id in sorted(expected_papers):
        projected[paper_id] = {
            "audit_id": template.audit_id,
            "outcome": "included_in_analyst_review",
            "decision": "withheld_in_public_artifacts",
        }
    return projected


def summarize_human_review(
    template_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Create an aggregate-only public summary after every decision is filled."""

    template = HumanReviewTemplate.model_validate(
        json.loads(template_path.read_text(encoding="utf-8"))
    )
    pending = sum(item.decision.outcome == ReviewOutcome.PENDING for item in template.items)
    if pending:
        raise ValueError(f"{pending} human-review decisions are still pending")
    outcome_counts = Counter(item.decision.outcome.value for item in template.items)
    issue_counts = Counter(
        issue.value for item in template.items for issue in item.decision.issue_codes
    )
    risk_counts = Counter(reason.value for item in template.items for reason in item.risk_reasons)
    sampled_papers = len({_review_item_paper_id(item) for item in template.items})
    risk_scores = [item.risk_score for item in template.items]
    candidate_items = sum(
        not isinstance(item, PaperWithoutCandidatesReviewItem) for item in template.items
    )
    papers_without_candidates = sum(
        artifact.candidate_count == 0 for artifact in template.source_artifacts
    )
    papers_without_candidates_reviewed = sum(
        isinstance(item, PaperWithoutCandidatesReviewItem) for item in template.items
    )
    summary: dict[str, Any] = {
        "schema_version": "human-review-summary/0.1",
        "audit_id": template.audit_id,
        "sampling_policy": template.sampling_policy,
        "population": {
            "candidates": template.population_candidates,
            "papers": template.population_papers,
            "papers_without_candidates": papers_without_candidates,
        },
        "sample": {
            "requested": template.sample_requested,
            "reviewed": len(template.items),
            "papers_reviewed": sampled_papers,
            "paper_coverage": round(sampled_papers / template.population_papers, 6),
            "item_type_counts": {
                # A paper_without_eee item contains one sampled candidate. Keep the
                # stable public 0.1 summary taxonomy while the private item makes the
                # paper-level output absence explicit to the analyst.
                "candidate": candidate_items,
                "paper_without_candidates": papers_without_candidates_reviewed,
            },
            "papers_without_candidates_reviewed": papers_without_candidates_reviewed,
            "risk_score_min": min(risk_scores),
            "risk_score_max": max(risk_scores),
            "risk_score_mean": round(sum(risk_scores) / len(risk_scores), 6),
            "risk_reason_counts": dict(sorted(risk_counts.items())),
        },
        "decisions": {
            "completed": len(template.items),
            "outcome_counts": {
                outcome.value: outcome_counts.get(outcome.value, 0)
                for outcome in ReviewOutcome
                if outcome != ReviewOutcome.PENDING
            },
            "issue_counts": {
                issue.value: issue_counts.get(issue.value, 0) for issue in ReviewIssue
            },
        },
        "privacy": {
            "contains_evidence_quotes": False,
            "contains_candidate_payloads": False,
            "contains_provider_raw_data": False,
            "contains_local_paths": False,
            "contains_reviewer_notes": False,
        },
    }
    if output_path is not None:
        write_json(output_path, summary)
    return summary
