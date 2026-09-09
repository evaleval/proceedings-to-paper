"""Immutable review locking and deterministic reviewed EEE recomposition."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ValidationError

from proceedings_to_eee.composition.eee import compose_eee_records
from proceedings_to_eee.domain.attribution import AttributionState, AttributionVerdict
from proceedings_to_eee.domain.export_provenance import (
    ExportCompositionProvenance,
    ExportProvenanceMode,
    legacy_export_provenance,
    tuple_audited_review_export_provenance,
)
from proceedings_to_eee.domain.lineage import CandidateLineageArtifact
from proceedings_to_eee.domain.observation import CandidateObservation, EvidenceAnchor
from proceedings_to_eee.domain.provenance import (
    CandidateField,
    CandidateFieldProvenance,
    FieldBindingStatus,
    FieldSourceKind,
    FieldSourceRef,
)
from proceedings_to_eee.domain.status import (
    ActorRole,
    ClaimType,
    EvidenceKind,
    ExportStatus,
    ReferentialStatus,
    TextSupportStatus,
)
from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.io import (
    atomic_write_bytes,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from proceedings_to_eee.resolution.metrics import resolve_metric_value
from proceedings_to_eee.resources import DEFAULT_EEE_SCHEMA_PATH, EEE_SCHEMA_SHA256
from proceedings_to_eee.reviewed_export.models import (
    DERIVED_COUNT_KEYS,
    ArtifactRef,
    BoundArtifact,
    CandidateFingerprint,
    DecisionStatus,
    DerivedRunManifest,
    DerivedVerification,
    ExportOutcome,
    ExportOutcomeState,
    LockedDecision,
    OriginDecision,
    PaperRunBinding,
    QuoteFreeEvidence,
    ReviewAuthorityMode,
    ReviewedExportDecision,
    ReviewedExportProvenance,
    ReviewedTuple,
    ReviewEvidenceAnchor,
    ReviewItem,
    ReviewLock,
    ReviewManifest,
    RunSealBinding,
    TupleDecision,
    model_payload,
    review_evidence_span_id,
)
from proceedings_to_eee.run_seal import RunSealVerificationError, verify_run_seal
from proceedings_to_eee.sources.manifest import SourceManifest
from proceedings_to_eee.validation.candidates import (
    bounded_claim_present,
    normalize_evidence_text,
    validate_non_origin_candidates,
)
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record
from proceedings_to_eee.validation.field_provenance import (
    direct_quote_field_binding,
    direct_quote_tuple_group_association_supported,
    value_quote_support_issue,
)

REVIEW_MANIFEST_NAME = "review-manifest.json"
REVIEW_ITEMS_NAME = "items.jsonl"
REVIEW_DECISIONS_NAME = "decisions.jsonl"
REVIEW_PROTOCOL_NAME = "protocol.md"
REVIEW_LOCK_NAME = "review-lock.json"

DERIVED_MANIFEST_NAME = "derived-run.json"
DERIVED_SHA256SUMS_NAME = "SHA256SUMS"
DERIVED_VERIFICATION_NAME = "verification.json"
DERIVED_PROVENANCE_NAME = "reviewed-export-provenance.jsonl"
DERIVED_OUTCOMES_NAME = "export-outcomes.jsonl"

_VERIFIER_GATE_SCHEMA_BY_PIPELINE_RUN = {
    "pipeline-run/0.3": "independent-verifier-gates/0.1",
    "pipeline-run/0.4": "independent-verifier-gates/0.2",
}


def _expected_verifier_gate_schema(run_schema: object) -> str | None:
    """Return the only verifier-sidecar schema paired with a pipeline receipt."""

    if not isinstance(run_schema, str):
        return None
    return _VERIFIER_GATE_SCHEMA_BY_PIPELINE_RUN.get(run_schema)


class ReviewedExportErrorCode(StrEnum):
    RUN_NOT_SEALED = "RUN_NOT_SEALED"
    RUN_SEAL_INVALID = "RUN_SEAL_INVALID"
    RUN_BINDING_MISMATCH = "RUN_BINDING_MISMATCH"
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_HASH_MISMATCH = "ARTIFACT_HASH_MISMATCH"
    ARTIFACT_PATH_INVALID = "ARTIFACT_PATH_INVALID"
    ARTIFACT_TREE_INVALID = "ARTIFACT_TREE_INVALID"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    OBSERVATION_LEDGER_INVALID = "OBSERVATION_LEDGER_INVALID"
    CANDIDATE_FINGERPRINT_MISMATCH = "CANDIDATE_FINGERPRINT_MISMATCH"
    CANDIDATE_BINDING_INVALID = "CANDIDATE_BINDING_INVALID"
    DECISION_SET_INVALID = "DECISION_SET_INVALID"
    DECISION_BINDING_MISMATCH = "DECISION_BINDING_MISMATCH"
    TUPLE_MISMATCH = "TUPLE_MISMATCH"
    AUTHORITY_INVALID = "AUTHORITY_INVALID"
    RESULT_EVIDENCE_INVALID = "RESULT_EVIDENCE_INVALID"
    ORIGIN_EVIDENCE_INVALID = "ORIGIN_EVIDENCE_INVALID"
    REVIEW_NOT_LOCKED = "REVIEW_NOT_LOCKED"
    REVIEW_LOCK_MISMATCH = "REVIEW_LOCK_MISMATCH"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    OUTPUT_OVERLAP = "OUTPUT_OVERLAP"
    EEE_SCHEMA_INVALID = "EEE_SCHEMA_INVALID"
    DERIVED_INTEGRITY_FAILURE = "DERIVED_INTEGRITY_FAILURE"


class ReviewedExportError(RuntimeError):
    """Stable, path-free failure suitable for CLI output and regression tests."""

    def __init__(self, code: ReviewedExportErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


@dataclass(frozen=True)
class _RunPaper:
    paper_id: str
    paper_root: str
    run_path: Path
    observations_path: Path
    candidate_lineage_path: Path | None
    tuple_resolution_path: Path | None
    verifier_gates_path: Path | None
    source_manifest_path: Path
    layout_path: Path
    result_blocks_path: Path
    run_manifest: dict[str, Any]
    source_manifest: SourceManifest
    layout: PdfLayout
    candidates: tuple[CandidateObservation, ...]
    ledger_lines: tuple[bytes, ...]
    candidate_lineage: CandidateLineageArtifact | None
    tuple_resolution: dict[str, Any] | None
    verifier_gates: dict[str, Any] | None


@dataclass(frozen=True)
class _ReviewPaper:
    binding: PaperRunBinding
    run_manifest: dict[str, Any]
    source_manifest: SourceManifest
    layout: PdfLayout
    candidates: tuple[CandidateObservation, ...]
    ledger_lines: tuple[bytes, ...]
    candidate_lineage: CandidateLineageArtifact | None
    tuple_resolution: dict[str, Any] | None
    verifier_gates: dict[str, Any] | None


@dataclass(frozen=True)
class ValidatedReview:
    root: Path
    manifest: ReviewManifest
    manifest_sha256: str
    items: tuple[ReviewItem, ...]
    decisions: tuple[ReviewedExportDecision, ...]
    papers: dict[str, _ReviewPaper]
    lock: ReviewLock | None
    lock_sha256: str | None


@dataclass(frozen=True)
class ContextualDerivedVerification:
    """Portable receipt plus quote-free counts from context-bound exports."""

    verification: DerivedVerification
    authority_mode_counts: dict[str, int]
    provenance_mode_counts: dict[str, int]


def _error(code: ReviewedExportErrorCode, detail: str) -> ReviewedExportError:
    return ReviewedExportError(code, detail)


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise _error(
            ReviewedExportErrorCode.ARTIFACT_PATH_INVALID,
            "artifact path is not a normalized relative path",
        )
    return path


def _confined_file(root: Path, relative: str) -> Path:
    rel = _relative(relative)
    if root.is_symlink() or not root.is_dir():
        raise _error(
            ReviewedExportErrorCode.ARTIFACT_TREE_INVALID,
            "artifact root is not a regular directory",
        )
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise _error(
                ReviewedExportErrorCode.ARTIFACT_TREE_INVALID,
                "artifact tree contains a symbolic link",
            )
    if not current.is_file():
        raise _error(ReviewedExportErrorCode.ARTIFACT_MISSING, "required artifact is missing")
    try:
        current.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise _error(
            ReviewedExportErrorCode.ARTIFACT_PATH_INVALID,
            "artifact path escapes its root",
        ) from exc
    return current


def _require_disjoint_output(output: Path, *inputs: Path) -> None:
    target = output.resolve()
    if output.exists() or output.is_symlink():
        raise _error(ReviewedExportErrorCode.OUTPUT_EXISTS, "output already exists")
    for input_root in inputs:
        source = input_root.resolve()
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise _error(
                ReviewedExportErrorCode.OUTPUT_OVERLAP,
                "output must be disjoint from every input tree",
            )


def _atomic_tree(output: Path, builder: Any) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    tree = temporary_parent / "tree"
    tree.mkdir(mode=0o700)
    try:
        builder(tree)
        tree.rename(output)
    except BaseException:
        shutil.rmtree(temporary_parent, ignore_errors=True)
        raise
    temporary_parent.rmdir()


def _jsonl_records[ModelT: BaseModel](
    path: Path, model: type[ModelT], label: str
) -> tuple[ModelT, ...]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _error(
            ReviewedExportErrorCode.OBSERVATION_LEDGER_INVALID,
            f"{label} is unavailable or not UTF-8",
        ) from exc
    records: list[ModelT] = []
    for line in text.splitlines():
        if not line.strip():
            raise _error(
                ReviewedExportErrorCode.OBSERVATION_LEDGER_INVALID,
                f"{label} contains a blank line",
            )
        try:
            payload = json.loads(line)
            records.append(model.model_validate(payload))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise _error(
                ReviewedExportErrorCode.OBSERVATION_LEDGER_INVALID,
                f"{label} contains an invalid record",
            ) from exc
    return tuple(records)


def _candidate_records(path: Path) -> tuple[tuple[CandidateObservation, ...], tuple[bytes, ...]]:
    raw = path.read_bytes()
    raw_lines = tuple(raw.splitlines(keepends=True))
    if any(not line.strip() for line in raw_lines):
        raise _error(
            ReviewedExportErrorCode.OBSERVATION_LEDGER_INVALID,
            "observation ledger contains a blank line",
        )
    candidates: list[CandidateObservation] = []
    for line in raw_lines:
        try:
            candidates.append(CandidateObservation.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise _error(
                ReviewedExportErrorCode.OBSERVATION_LEDGER_INVALID,
                "observation ledger contains an invalid candidate",
            ) from exc
    return tuple(candidates), raw_lines


def _candidate_payload_sha256(candidate: CandidateObservation) -> str:
    return sha256_bytes(canonical_json_bytes(model_payload(candidate)))


def _tuple_sha256(value: ReviewedTuple) -> str:
    return sha256_bytes(canonical_json_bytes(model_payload(value)))


def _literal_spans(text: str, literal: str) -> list[tuple[int, int]]:
    """Return bounded exact literal spans without losing source-text offsets."""

    if not literal:
        return []
    boundary = r"[\w.,]" if any(character.isdigit() for character in literal) else r"\w"
    return [
        match.span()
        for match in re.finditer(
            rf"(?<!{boundary}){re.escape(literal)}(?!{boundary})",
            text,
            re.IGNORECASE,
        )
    ]


def _absolute_value_occurrences(
    candidate: CandidateObservation, layout: PdfLayout
) -> list[dict[str, object]]:
    """Locate the proposed raw value in immutable layout coordinates."""

    if candidate.value is None:
        return []
    pages = {(page.source_id, page.page): page.text for page in layout.pages}
    occurrences: set[tuple[str, int, int, int]] = set()
    for anchor in candidate.evidence:
        page_text = pages.get((anchor.source_id, anchor.page))
        if page_text is None:
            continue
        relative_spans = _literal_spans(anchor.quote, candidate.value.raw)
        if not relative_spans:
            continue
        start = 0
        while True:
            anchor_start = page_text.find(anchor.quote, start)
            if anchor_start < 0:
                break
            for relative_start, relative_end in relative_spans:
                occurrences.add(
                    (
                        anchor.source_id,
                        anchor.page,
                        anchor_start + relative_start,
                        anchor_start + relative_end,
                    )
                )
            start = anchor_start + 1
    return [
        {
            "source_id": source_id,
            "page": page,
            "char_start": char_start,
            "char_end": char_end,
        }
        for source_id, page, char_start, char_end in sorted(occurrences)
    ]


def _structural_identity_sha256(candidate: CandidateObservation, layout: PdfLayout) -> str:
    """Fingerprint the physical value occurrence, not auxiliary context or semantics."""

    absolute_occurrences = _absolute_value_occurrences(candidate, layout)
    if absolute_occurrences:
        # Containing excerpts, labels, and whitespace variants may differ while the
        # immutable printed numeric token is the same.  Page offsets are the common
        # occurrence identity available to every exact anchor representation.
        locations = absolute_occurrences
    else:
        # Unsupported proposals still need a deterministic review identity.  They can
        # never use this fallback as proof of eligibility, and the composer replays the
        # broader conservative duplicate groups before writing any EEE.
        locations = sorted(
            (
                {
                    "source_id": anchor.source_id,
                    "page": anchor.page,
                    "kind": anchor.kind.value,
                    "normalized_quote_sha256": sha256_bytes(
                        normalize_evidence_text(anchor.quote).casefold().encode("utf-8")
                    ),
                }
                for anchor in candidate.evidence
            ),
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    payload = {
        "paper_id": candidate.paper_id,
        "value_occurrence": locations,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _load_corpus_membership(
    run_root: Path, corpus_run_path: Path
) -> tuple[list[tuple[str, Path]], dict[str, dict[str, Any]]]:
    try:
        corpus_run = read_json(corpus_run_path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise _error(
            ReviewedExportErrorCode.MANIFEST_INVALID,
            "corpus run manifest is not valid JSON",
        ) from exc
    if not isinstance(corpus_run, dict):
        raise _error(
            ReviewedExportErrorCode.MANIFEST_INVALID,
            "corpus run manifest is not an object",
        )
    if corpus_run.get("schema_version") not in {
        "corpus-run/0.1",
        "corpus-run/0.2",
        "corpus-run/0.3",
    }:
        raise _error(
            ReviewedExportErrorCode.MANIFEST_INVALID,
            "corpus run manifest schema is unsupported",
        )
    runs = corpus_run.get("runs")
    declared_count = corpus_run.get("papers")
    if (
        not isinstance(runs, list)
        or type(declared_count) is not int
        or declared_count < 0
        or declared_count != len(runs)
    ):
        raise _error(
            ReviewedExportErrorCode.MANIFEST_INVALID,
            "corpus run paper count and membership are invalid",
        )

    declared: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for run in runs:
        paper_id = run.get("paper_id") if isinstance(run, dict) else None
        if (
            not isinstance(paper_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", paper_id) is None
            or paper_id in declared
        ):
            raise _error(
                ReviewedExportErrorCode.MANIFEST_INVALID,
                "corpus run paper membership is unsafe or duplicated",
            )
        declared[paper_id] = run
        ordered_ids.append(paper_id)

    discovered_ids = {
        entry.name
        for entry in run_root.iterdir()
        if entry.is_dir()
        and (entry / "run.json").is_file()
        and not (entry / "run.json").is_symlink()
    }
    if discovered_ids != set(ordered_ids):
        raise _error(
            ReviewedExportErrorCode.MANIFEST_INVALID,
            "corpus run membership differs from paper run directories",
        )
    return [(paper_id, run_root / paper_id) for paper_id in ordered_ids], declared


def _load_run_papers(run_root: Path) -> tuple[str, list[_RunPaper], Path | None]:
    root_run_path = run_root / "run.json"
    corpus_candidate_path = run_root / "corpus-run.json"
    declared_runs: dict[str, dict[str, Any]] | None = None
    if root_run_path.is_file():
        if corpus_candidate_path.exists() or corpus_candidate_path.is_symlink():
            raise _error(
                ReviewedExportErrorCode.MANIFEST_INVALID,
                "sealed run cannot be both a paper run and a corpus run",
            )
        roots = [(".", run_root)]
        run_kind = "paper"
        corpus_run_path = None
    else:
        corpus_run_path = corpus_candidate_path
        if not corpus_run_path.is_file() or corpus_run_path.is_symlink():
            raise _error(
                ReviewedExportErrorCode.ARTIFACT_MISSING,
                "sealed run is neither a paper run nor a corpus run",
            )
        roots, declared_runs = _load_corpus_membership(run_root, corpus_run_path)
        run_kind = "corpus"
    if not roots:
        raise _error(ReviewedExportErrorCode.ARTIFACT_MISSING, "sealed run contains no papers")

    papers: list[_RunPaper] = []
    seen: set[str] = set()
    for paper_root, root in roots:
        paths = {
            "run": root / "run.json",
            "observations": root / "observations.jsonl",
            "source": root / "source-manifest.json",
            "layout": root / "private" / "layout.json",
            "blocks": root / "private" / "result-blocks.json",
        }
        if any(path.is_symlink() or not path.is_file() for path in paths.values()):
            raise _error(
                ReviewedExportErrorCode.ARTIFACT_MISSING,
                "paper run is missing a required review binding",
            )
        try:
            run_manifest = read_json(paths["run"])
            source_manifest = SourceManifest.model_validate(read_json(paths["source"]))
            layout = PdfLayout.model_validate(read_json(paths["layout"]))
        except (OSError, ValueError, ValidationError) as exc:
            raise _error(
                ReviewedExportErrorCode.MANIFEST_INVALID,
                "paper run contains an invalid manifest or layout",
            ) from exc
        if not isinstance(run_manifest, dict):
            raise _error(ReviewedExportErrorCode.MANIFEST_INVALID, "run manifest is not an object")
        run_schema = run_manifest.get("schema_version")
        if run_schema not in {"pipeline-run/0.2", "pipeline-run/0.3", "pipeline-run/0.4"}:
            raise _error(
                ReviewedExportErrorCode.MANIFEST_INVALID,
                "paper run manifest schema is unsupported",
            )
        paper_id = run_manifest.get("paper_id")
        if (
            not isinstance(paper_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", paper_id) is None
            or source_manifest.paper_id != paper_id
            or (paper_root != "." and paper_root != paper_id)
        ):
            raise _error(
                ReviewedExportErrorCode.MANIFEST_INVALID,
                "paper identity is unsafe or inconsistent",
            )
        if declared_runs is not None and run_manifest != declared_runs.get(paper_id):
            raise _error(
                ReviewedExportErrorCode.MANIFEST_INVALID,
                "paper run differs from its corpus membership record",
            )
        if paper_id in seen:
            raise _error(ReviewedExportErrorCode.MANIFEST_INVALID, "paper identity is duplicated")
        seen.add(paper_id)
        source_ids = {source.source_id for source in source_manifest.sources}
        if layout.source_id not in source_ids:
            raise _error(
                ReviewedExportErrorCode.MANIFEST_INVALID,
                "layout source is absent from the source manifest",
            )
        for page in layout.pages:
            if page.text_sha256 != sha256_bytes(page.text.encode("utf-8")):
                raise _error(
                    ReviewedExportErrorCode.MANIFEST_INVALID,
                    "layout page text hash is invalid",
                )
        candidates, ledger_lines = _candidate_records(paths["observations"])
        if any(candidate.paper_id != paper_id for candidate in candidates):
            raise _error(
                ReviewedExportErrorCode.CANDIDATE_BINDING_INVALID,
                "observation ledger contains a candidate for another paper",
            )
        lineage_path = root / "candidate-lineage.json"
        candidate_lineage: CandidateLineageArtifact | None = None
        if run_schema in {"pipeline-run/0.3", "pipeline-run/0.4"}:
            lineage_binding = run_manifest.get("candidate_lineage")
            if (
                not isinstance(lineage_binding, dict)
                or lineage_binding.get("path") != "candidate-lineage.json"
                or lineage_path.is_symlink()
                or not lineage_path.is_file()
                or lineage_binding.get("artifact_sha256") != sha256_file(lineage_path)
            ):
                raise _error(
                    ReviewedExportErrorCode.MANIFEST_INVALID,
                    "new-schema run lacks an exact candidate-lineage binding",
                )
            try:
                candidate_lineage = CandidateLineageArtifact.model_validate(read_json(lineage_path))
            except (OSError, ValueError, ValidationError) as exc:
                raise _error(
                    ReviewedExportErrorCode.MANIFEST_INVALID,
                    "candidate lineage is invalid",
                ) from exc
            if (
                candidate_lineage.paper_id != paper_id
                or candidate_lineage.observations_sha256 != sha256_file(paths["observations"])
                or lineage_binding.get("counts") != candidate_lineage.counts.model_dump(mode="json")
            ):
                raise _error(
                    ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                    "candidate lineage does not bind the observation ledger",
                )
        tuple_resolution_path: Path | None = None
        tuple_resolution: dict[str, Any] | None = None
        tuple_config = run_manifest.get("tuple_resolution")
        tuple_enabled = bool(isinstance(tuple_config, dict) and tuple_config.get("enabled") is True)
        declared_verifier = run_manifest.get("verifier")
        expected_tuple_mode = (
            ExportProvenanceMode.TUPLE_GATED_PRODUCTION
            if isinstance(declared_verifier, dict) and declared_verifier.get("enabled") is True
            else ExportProvenanceMode.TUPLE_GATED_UNVERIFIED
        )
        if tuple_enabled:
            sidecar_binding = tuple_config.get("sidecar")
            candidate_lineage_binding = run_manifest.get("candidate_lineage")
            tuple_resolution_path = root / "private" / "tuple-resolution.json"
            if (
                not isinstance(sidecar_binding, dict)
                or sidecar_binding.get("schema_version") != "tuple-resolution-run/0.1"
                or sidecar_binding.get("path") != "private/tuple-resolution.json"
                or not isinstance(sidecar_binding.get("sha256"), str)
                or tuple_resolution_path.is_symlink()
                or not tuple_resolution_path.is_file()
                or sidecar_binding["sha256"] != sha256_file(tuple_resolution_path)
                or candidate_lineage is None
                or candidate_lineage.export_provenance_mode is not expected_tuple_mode
                or candidate_lineage.tuple_sidecar_sha256 != sidecar_binding["sha256"]
                or not isinstance(candidate_lineage_binding, dict)
                or candidate_lineage_binding.get("tuple_gate_sidecar_sha256")
                != sidecar_binding["sha256"]
                or candidate_lineage_binding.get("export_provenance_mode")
                != expected_tuple_mode.value
                or candidate_lineage_binding.get("export_composition_sha256")
                != candidate_lineage.export_composition_sha256
            ):
                raise _error(
                    ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                    "tuple-enabled run lacks exact sidecar and lineage bindings",
                )
            try:
                raw_tuple = read_json(tuple_resolution_path)
            except (OSError, ValueError) as exc:
                raise _error(
                    ReviewedExportErrorCode.MANIFEST_INVALID,
                    "tuple-resolution sidecar is invalid",
                ) from exc
            if not isinstance(raw_tuple, dict):
                raise _error(
                    ReviewedExportErrorCode.MANIFEST_INVALID,
                    "tuple-resolution sidecar is not an object",
                )
            gates = raw_tuple.get("candidate_gates")
            outcomes = raw_tuple.get("outcomes")
            candidate_ids = {str(candidate.observation_id) for candidate in candidates}
            if (
                raw_tuple.get("schema_version") != "tuple-resolution-run/0.1"
                or not isinstance(gates, dict)
                or set(gates) != candidate_ids
                or any(
                    not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                    for value in gates.values()
                )
                or not isinstance(outcomes, list)
            ):
                raise _error(
                    ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                    "tuple-resolution sidecar does not exactly cover candidates",
                )
            pass_by_id = {candidate_id: False for candidate_id in candidate_ids}
            seen_outcomes: set[str] = set()
            for outcome in outcomes:
                observation_id = (
                    outcome.get("observation_id") if isinstance(outcome, dict) else None
                )
                if (
                    not isinstance(observation_id, str)
                    or observation_id not in gates
                    or observation_id in seen_outcomes
                    or outcome.get("gate_sha256") != gates[observation_id]
                    or type(outcome.get("passed")) is not bool
                ):
                    raise _error(
                        ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                        "tuple-resolution outcome binding is invalid",
                    )
                seen_outcomes.add(observation_id)
                pass_by_id[observation_id] = outcome["passed"]
            lineage_by_id = {
                item.final_observation_id: item for item in candidate_lineage.candidates
            }
            if set(lineage_by_id) != candidate_ids or any(
                lineage_by_id[observation_id].tuple_gate_sha256 != gate_sha256
                for observation_id, gate_sha256 in gates.items()
            ):
                raise _error(
                    ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                    "candidate lineage does not bind every tuple gate",
                )
            tuple_resolution = {
                "sidecar_sha256": sidecar_binding["sha256"],
                "candidate_gates": gates,
                "passed": pass_by_id,
            }
        elif candidate_lineage is not None and (
            candidate_lineage.export_provenance_mode is not None
            and candidate_lineage.export_provenance_mode.has_tuple_audit
        ):
            raise _error(
                ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                "tuple-disabled run claims tuple-gated lineage",
            )
        verifier_gates_path: Path | None = None
        verifier_gates: dict[str, Any] | None = None
        verifier_config = run_manifest.get("verifier")
        verifier_enabled = bool(
            isinstance(verifier_config, dict) and verifier_config.get("enabled") is True
        )
        if verifier_enabled:
            expected_verifier_schema = _expected_verifier_gate_schema(run_schema)
            sidecar_binding = verifier_config.get("sidecar")
            candidate_lineage_binding = run_manifest.get("candidate_lineage")
            verifier_gates_path = root / "private" / "verifier-gates.json"
            if (
                tuple_resolution is None
                or expected_verifier_schema is None
                or not isinstance(sidecar_binding, dict)
                or sidecar_binding.get("schema_version") != expected_verifier_schema
                or sidecar_binding.get("path") != "private/verifier-gates.json"
                or not isinstance(sidecar_binding.get("sha256"), str)
                or verifier_gates_path.is_symlink()
                or not verifier_gates_path.is_file()
                or sidecar_binding["sha256"] != sha256_file(verifier_gates_path)
                or candidate_lineage is None
                or candidate_lineage.verifier_gate_required is not True
                or candidate_lineage.verifier_sidecar_sha256 != sidecar_binding["sha256"]
                or not isinstance(candidate_lineage_binding, dict)
                or candidate_lineage_binding.get("verifier_gate_required") is not True
                or candidate_lineage_binding.get("verifier_gate_sidecar_sha256")
                != sidecar_binding["sha256"]
            ):
                raise _error(
                    ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                    "verifier-enabled run lacks exact sidecar and lineage bindings",
                )
            try:
                raw_verifier = read_json(verifier_gates_path)
            except (OSError, ValueError) as exc:
                raise _error(
                    ReviewedExportErrorCode.MANIFEST_INVALID,
                    "verifier-gate sidecar is invalid",
                ) from exc
            gates = raw_verifier.get("candidate_gates") if isinstance(raw_verifier, dict) else None
            passed_ids = (
                raw_verifier.get("passed_candidate_ids") if isinstance(raw_verifier, dict) else None
            )
            candidate_ids = {str(candidate.observation_id) for candidate in candidates}
            if (
                not isinstance(raw_verifier, dict)
                or raw_verifier.get("schema_version") != expected_verifier_schema
                or not isinstance(gates, dict)
                or not set(gates).issubset(candidate_ids)
                or any(
                    not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                    for value in gates.values()
                )
                or not isinstance(passed_ids, list)
                or any(not isinstance(value, str) for value in passed_ids)
                or passed_ids != sorted(set(passed_ids))
                or not set(passed_ids).issubset(gates)
            ):
                raise _error(
                    ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                    "verifier-gate sidecar has an invalid candidate partition",
                )
            lineage_by_id = {
                item.final_observation_id: item for item in candidate_lineage.candidates
            }
            if any(
                lineage_by_id[observation_id].verifier_gate_sha256 != gate_sha256
                for observation_id, gate_sha256 in gates.items()
            ) or any(
                item.verifier_gate_sha256 is not None and item.final_observation_id not in gates
                for item in candidate_lineage.candidates
            ):
                raise _error(
                    ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                    "candidate lineage does not exactly bind verifier gates",
                )
            verifier_gates = {
                "sidecar_sha256": sidecar_binding["sha256"],
                "candidate_gates": gates,
                "passed": {
                    candidate_id: candidate_id in set(passed_ids) for candidate_id in candidate_ids
                },
            }
        elif candidate_lineage is not None and candidate_lineage.verifier_gate_required is True:
            raise _error(
                ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                "verifier-disabled run claims verifier-gated lineage",
            )
        papers.append(
            _RunPaper(
                paper_id=paper_id,
                paper_root=paper_root,
                run_path=paths["run"],
                observations_path=paths["observations"],
                candidate_lineage_path=(lineage_path if candidate_lineage is not None else None),
                tuple_resolution_path=tuple_resolution_path,
                verifier_gates_path=verifier_gates_path,
                source_manifest_path=paths["source"],
                layout_path=paths["layout"],
                result_blocks_path=paths["blocks"],
                run_manifest=run_manifest,
                source_manifest=source_manifest,
                layout=layout,
                candidates=candidates,
                ledger_lines=ledger_lines,
                candidate_lineage=candidate_lineage,
                tuple_resolution=tuple_resolution,
                verifier_gates=verifier_gates,
            )
        )
    return run_kind, papers, corpus_run_path


def _nested_string(payload: dict[str, Any], *keys: str) -> str | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, str) else None


def _run_relative(run_root: Path, path: Path) -> str:
    relative = path.relative_to(run_root).as_posix()
    _relative(relative)
    return relative


def _copy_bound_artifact(
    *, run_root: Path, source: Path, tree: Path, review_path: str, records: int | None = None
) -> BoundArtifact:
    destination = tree / _relative(review_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != sha256_file(source):
        raise _error(
            ReviewedExportErrorCode.ARTIFACT_HASH_MISMATCH,
            "copied review input differs from sealed run",
        )
    destination.chmod(0o444)
    return BoundArtifact(
        run_path=_run_relative(run_root, source),
        review_copy=ArtifactRef(
            path=review_path,
            sha256=sha256_file(destination),
            size_bytes=destination.stat().st_size,
            records=records,
        ),
    )


def _line_span(text: str, start: int, end: int) -> tuple[int, int]:
    return text.count("\n", 0, start) + 1, text.count("\n", 0, max(end - 1, 0)) + 1


def _suggested_result_anchor(
    *,
    candidate: CandidateObservation,
    evidence_index: int,
    source_manifest: SourceManifest,
    source_manifest_sha256: str,
    layout: PdfLayout,
    layout_sha256: str,
) -> ReviewEvidenceAnchor | None:
    anchor = candidate.evidence[evidence_index]
    if anchor.source_id != layout.source_id or anchor.page > layout.page_count:
        return None
    page = layout.pages[anchor.page - 1]
    offsets = [match.start() for match in re.finditer(re.escape(anchor.quote), page.text)]
    if len(offsets) != 1:
        return None
    source = next(
        (item for item in source_manifest.sources if item.source_id == anchor.source_id), None
    )
    if source is None or source.sha256 is None:
        return None
    start = offsets[0]
    end = start + len(anchor.quote)
    start_line, end_line = _line_span(page.text, start, end)
    return ReviewEvidenceAnchor(
        source_id=anchor.source_id,
        source_sha256=source.sha256,
        source_manifest_sha256=source_manifest_sha256,
        layout_sha256=layout_sha256,
        parser=layout.parser,
        parser_version=layout.parser_version,
        page=anchor.page,
        page_text_sha256=page.text_sha256,
        exact_excerpt=anchor.quote,
        excerpt_sha256=sha256_bytes(anchor.quote.encode("utf-8")),
        char_start=start,
        char_end=end,
        start_line=start_line,
        end_line=end_line,
        kind=anchor.kind,
        region_id=anchor.region_id,
        label=anchor.label,
        row=anchor.row,
        column=anchor.column,
        bounding_box=anchor.bounding_box,
    )


def _review_protocol() -> str:
    return """# Reviewed EEE export protocol

This is a private, candidate-bound export decision packet. Do not commit it or send it
to a model provider. The immutable item records contain source excerpts; only the
quote-free derived EEE and provenance sidecar may be considered for public release.

For each candidate you decide, set `status` to `completed`. Confirm, reject, or leave
the full tuple unresolved; decide origin as `paper_produced`, `externally_sourced`, or
`unresolved`; and record the declared authority and an offset-aware exact evidence
anchor. A result-evidence bundle may contain multiple exact anchors: at least one must
support the reported raw value. For a confirmed tuple, add exactly one
`field_attestations` entry for every populated tuple field (including a populated
setting), copy that field's unchanged value hash, and bind it to one or more retained
result `span_id` values. Keep pending templates' field attestations empty: suggestions
are not review claims. This allows load-bearing captions, headers, and wrapped names to
remain beside the value anchor. Result evidence and producer-origin evidence are
separate and may come from different pages. Each excerpt must be an exact UTF-8
substring of one declared frozen layout page. Do not normalize whitespace or join
columns. A paper-produced decision must cite origin evidence that identifies the
evaluated system, not generic language such as “we evaluate”.

Pending records are valid locked abstentions and never export. Review may resolve
producer origin only after separately reattesting the complete unchanged candidate tuple
and strict result evidence. A full authorized tuple-and-evidence confirmation may
resolve an untrusted tuple/verifier review, rejection, failure, or abstention; a model
outcome is neither human authority nor a permanent veto. Review cannot rewrite the
tuple, override unsupported evidence, low confidence, wrong scope, conflicts,
duplicate cells, or schema failure. The composer reopens the sealed run, verifies the
source gate outcomes, and recomputes deterministic non-origin safety checks before
using a decision. Origin-only approval never authorizes export.
"""


def prepare_export_review(
    *, run_root: Path, output_root: Path, min_confidence: float = 0.8
) -> ReviewManifest:
    """Create a private, immutable-input review packet from one sealed run tree."""

    _require_disjoint_output(output_root, run_root)
    try:
        verified_seal = verify_run_seal(run_root)
    except RunSealVerificationError as exc:
        code = (
            ReviewedExportErrorCode.RUN_NOT_SEALED
            if getattr(exc, "code", "") == "run_seal_missing"
            else ReviewedExportErrorCode.RUN_SEAL_INVALID
        )
        raise _error(code, "source run seal verification failed") from exc
    run_kind, papers, corpus_run_path = _load_run_papers(run_root)

    prepared: list[tuple[ReviewItem, ReviewedExportDecision]] = []
    for paper in papers:
        validated = validate_non_origin_candidates(
            paper.candidates,
            {paper.layout.source_id: paper.layout},
            min_confidence=min_confidence,
        )
        ledger_sha = sha256_file(paper.observations_path)
        ledger_path = _run_relative(run_root, paper.observations_path)
        source_manifest_sha = sha256_file(paper.source_manifest_path)
        layout_sha = sha256_file(paper.layout_path)
        for line_number, (candidate, base_candidate, raw_line) in enumerate(
            zip(paper.candidates, validated, paper.ledger_lines, strict=True), start=1
        ):
            assert candidate.observation_id is not None
            payload_sha = _candidate_payload_sha256(candidate)
            fingerprint = CandidateFingerprint(
                paper_id=paper.paper_id,
                observation_id=candidate.observation_id,
                observation_ledger_path=ledger_path,
                observation_ledger_sha256=ledger_sha,
                observation_ledger_line=line_number,
                observation_ledger_line_sha256=sha256_bytes(raw_line),
                candidate_schema_version=candidate.schema_version,
                candidate_payload_sha256=payload_sha,
                structural_identity_sha256=_structural_identity_sha256(candidate, paper.layout),
            )
            item_seed = f"{paper.paper_id}\0{line_number}\0{payload_sha}".encode()
            item_id = "review_" + sha256_bytes(item_seed)[:24]
            suggested = [
                result
                for index in range(len(base_candidate.evidence))
                if (
                    result := _suggested_result_anchor(
                        candidate=base_candidate,
                        evidence_index=index,
                        source_manifest=paper.source_manifest,
                        source_manifest_sha256=source_manifest_sha,
                        layout=paper.layout,
                        layout_sha256=layout_sha,
                    )
                )
                is not None
            ]
            reviewed_tuple = ReviewedTuple.from_candidate(base_candidate)
            source_export_mode = (
                paper.candidate_lineage.export_provenance_mode
                if paper.tuple_resolution is not None and paper.candidate_lineage is not None
                else ExportProvenanceMode.LEGACY_MANUAL
            )
            item = ReviewItem(
                item_id=item_id,
                fingerprint=fingerprint,
                candidate=candidate.model_copy(deep=True),
                reviewed_tuple=reviewed_tuple,
                base_gate_status=base_candidate.export_status.value,
                base_gate_reason=base_candidate.export_reason,
                source_export_mode=source_export_mode,
                source_tuple_sidecar_sha256=(
                    paper.tuple_resolution["sidecar_sha256"]
                    if paper.tuple_resolution is not None
                    else None
                ),
                source_tuple_gate_sha256=(
                    paper.tuple_resolution["candidate_gates"][candidate.observation_id]
                    if paper.tuple_resolution is not None
                    else None
                ),
                source_verifier_sidecar_sha256=(
                    paper.verifier_gates["sidecar_sha256"]
                    if paper.verifier_gates is not None
                    else None
                ),
                source_verifier_gate_sha256=(
                    paper.verifier_gates["candidate_gates"].get(candidate.observation_id)
                    if paper.verifier_gates is not None
                    else None
                ),
                source_verifier_gate_passed=(
                    paper.verifier_gates["passed"][candidate.observation_id]
                    if paper.verifier_gates is not None
                    else None
                ),
                suggested_result_evidence=suggested,
            )
            decision_id = "decision_" + sha256_bytes(item_id.encode())[:24]
            decision = ReviewedExportDecision(
                decision_id=decision_id,
                item_id=item_id,
                candidate_payload_sha256=payload_sha,
                reviewed_tuple=reviewed_tuple.model_copy(deep=True),
                result_evidence=[anchor.model_copy(deep=True) for anchor in suggested],
            )
            prepared.append((item, decision))
    if len({item.item_id for item, _ in prepared}) != len(prepared):
        raise _error(
            ReviewedExportErrorCode.CANDIDATE_BINDING_INVALID,
            "review item identity collision",
        )

    result: dict[str, ReviewManifest] = {}

    def build(tree: Path) -> None:
        paper_bindings: list[PaperRunBinding] = []
        for paper in papers:
            prefix = f"inputs/{paper.paper_id}"
            run_binding = _copy_bound_artifact(
                run_root=run_root,
                source=paper.run_path,
                tree=tree,
                review_path=f"{prefix}/run.json",
            )
            observation_binding = _copy_bound_artifact(
                run_root=run_root,
                source=paper.observations_path,
                tree=tree,
                review_path=f"{prefix}/observations.jsonl",
                records=len(paper.candidates),
            )
            lineage_binding = (
                _copy_bound_artifact(
                    run_root=run_root,
                    source=paper.candidate_lineage_path,
                    tree=tree,
                    review_path=f"{prefix}/candidate-lineage.json",
                )
                if paper.candidate_lineage_path is not None
                else None
            )
            tuple_resolution_binding = (
                _copy_bound_artifact(
                    run_root=run_root,
                    source=paper.tuple_resolution_path,
                    tree=tree,
                    review_path=f"{prefix}/tuple-resolution.json",
                )
                if paper.tuple_resolution_path is not None
                else None
            )
            verifier_gates_binding = (
                _copy_bound_artifact(
                    run_root=run_root,
                    source=paper.verifier_gates_path,
                    tree=tree,
                    review_path=f"{prefix}/verifier-gates.json",
                )
                if paper.verifier_gates_path is not None
                else None
            )
            source_binding = _copy_bound_artifact(
                run_root=run_root,
                source=paper.source_manifest_path,
                tree=tree,
                review_path=f"{prefix}/source-manifest.json",
            )
            layout_binding = _copy_bound_artifact(
                run_root=run_root,
                source=paper.layout_path,
                tree=tree,
                review_path=f"{prefix}/layout.json",
            )
            blocks_binding = _copy_bound_artifact(
                run_root=run_root,
                source=paper.result_blocks_path,
                tree=tree,
                review_path=f"{prefix}/result-blocks.json",
            )
            row_config = paper.run_manifest.get("row_enumeration")
            row_config = row_config if isinstance(row_config, dict) else {}
            code = paper.run_manifest.get("code")
            code = code if isinstance(code, dict) else {}
            schema = paper.run_manifest.get("eee_schema")
            if not isinstance(schema, dict) or not isinstance(schema.get("version"), str):
                raise _error(
                    ReviewedExportErrorCode.MANIFEST_INVALID,
                    "run manifest lacks an EEE schema binding",
                )
            schema_sha = schema.get("sha256")
            if not isinstance(schema_sha, str) or re.fullmatch(r"[0-9a-f]{64}", schema_sha) is None:
                raise _error(
                    ReviewedExportErrorCode.MANIFEST_INVALID,
                    "run manifest EEE schema hash is invalid",
                )
            paper_bindings.append(
                PaperRunBinding(
                    paper_id=paper.paper_id,
                    paper_root=paper.paper_root,
                    run_manifest=run_binding,
                    observations=observation_binding,
                    candidate_lineage=lineage_binding,
                    tuple_resolution=tuple_resolution_binding,
                    verifier_gates=verifier_gates_binding,
                    source_manifest=source_binding,
                    layout=layout_binding,
                    result_blocks=blocks_binding,
                    layout_parser=paper.layout.parser,
                    layout_parser_version=paper.layout.parser_version,
                    extractor_prompt_sha256=_nested_string(
                        paper.run_manifest, "extractor", "prompt_sha256"
                    ),
                    extractor_request_contract_sha256=_nested_string(
                        paper.run_manifest, "extractor", "request_contract_sha256"
                    ),
                    row_prompt_sha256=(
                        row_config.get("prompt_sha256")
                        if isinstance(row_config.get("prompt_sha256"), str)
                        else None
                    ),
                    row_request_contract_sha256=(
                        row_config.get("request_contract_sha256")
                        if isinstance(row_config.get("request_contract_sha256"), str)
                        else None
                    ),
                    schema_version=schema["version"],
                    schema_sha256=schema_sha,
                    code_git_commit=(
                        code.get("git_commit") if isinstance(code.get("git_commit"), str) else None
                    ),
                    code_source_tree_sha256=(
                        code.get("source_tree_sha256")
                        if isinstance(code.get("source_tree_sha256"), str)
                        else None
                    ),
                )
            )
        corpus_binding = None
        if corpus_run_path is not None:
            corpus_binding = _copy_bound_artifact(
                run_root=run_root,
                source=corpus_run_path,
                tree=tree,
                review_path="inputs/corpus-run.json",
            )
        protocol_bytes = _review_protocol().encode("utf-8")
        atomic_write_bytes(tree / REVIEW_PROTOCOL_NAME, protocol_bytes)
        (tree / REVIEW_PROTOCOL_NAME).chmod(0o444)
        item_dicts = [model_payload(item) for item, _ in prepared]
        decision_dicts = [model_payload(decision) for _, decision in prepared]
        items_sha = write_jsonl(tree / REVIEW_ITEMS_NAME, item_dicts)
        decisions_sha = write_jsonl(tree / REVIEW_DECISIONS_NAME, decision_dicts)
        (tree / REVIEW_ITEMS_NAME).chmod(0o444)
        (tree / REVIEW_DECISIONS_NAME).chmod(0o600)
        manifest = ReviewManifest(
            run_kind=run_kind,
            run_seal=RunSealBinding(
                seal_sha256=verified_seal.seal_sha256,
                tree_sha256=verified_seal.tree_sha256,
                file_count=verified_seal.file_count,
                total_bytes=verified_seal.total_bytes,
                source_run_name=verified_seal.source_run_name,
            ),
            corpus_run=corpus_binding,
            papers=paper_bindings,
            min_confidence=min_confidence,
            protocol=ArtifactRef(
                path=REVIEW_PROTOCOL_NAME,
                sha256=sha256_bytes(protocol_bytes),
                size_bytes=len(protocol_bytes),
            ),
            items=ArtifactRef(
                path=REVIEW_ITEMS_NAME,
                sha256=items_sha,
                size_bytes=(tree / REVIEW_ITEMS_NAME).stat().st_size,
                records=len(prepared),
            ),
            decision_template=ArtifactRef(
                path=REVIEW_DECISIONS_NAME,
                sha256=decisions_sha,
                size_bytes=(tree / REVIEW_DECISIONS_NAME).stat().st_size,
                records=len(prepared),
            ),
            item_count=len(prepared),
        )
        write_json(tree / REVIEW_MANIFEST_NAME, manifest)
        (tree / REVIEW_MANIFEST_NAME).chmod(0o444)
        result["manifest"] = manifest

    _atomic_tree(output_root, build)
    _load_review(output_root, require_lock=False)
    return result["manifest"]


def _verify_artifact(root: Path, artifact: ArtifactRef) -> Path:
    path = _confined_file(root, artifact.path)
    if path.stat().st_size != artifact.size_bytes or sha256_file(path) != artifact.sha256:
        raise _error(
            ReviewedExportErrorCode.ARTIFACT_HASH_MISMATCH,
            "immutable review artifact hash or size changed",
        )
    if artifact.records is not None:
        try:
            count = len(path.read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeDecodeError) as exc:
            raise _error(
                ReviewedExportErrorCode.ARTIFACT_HASH_MISMATCH,
                "record artifact is not valid UTF-8",
            ) from exc
        if count != artifact.records:
            raise _error(
                ReviewedExportErrorCode.ARTIFACT_HASH_MISMATCH,
                "record artifact count changed",
            )
    return path


def _expected_review_files(manifest: ReviewManifest, *, locked: bool) -> set[str]:
    files = {
        REVIEW_MANIFEST_NAME,
        manifest.protocol.path,
        manifest.items.path,
        manifest.decision_template.path,
    }
    if locked:
        files.add(REVIEW_LOCK_NAME)
    if manifest.corpus_run is not None:
        files.add(manifest.corpus_run.review_copy.path)
    for paper in manifest.papers:
        files.update(
            artifact.review_copy.path
            for artifact in (
                paper.run_manifest,
                paper.observations,
                paper.source_manifest,
                paper.layout,
                paper.result_blocks,
            )
        )
        if paper.candidate_lineage is not None:
            files.add(paper.candidate_lineage.review_copy.path)
        if paper.tuple_resolution is not None:
            files.add(paper.tuple_resolution.review_copy.path)
        if paper.verifier_gates is not None:
            files.add(paper.verifier_gates.review_copy.path)
    return files


def _validate_exact_tree(root: Path, expected: set[str]) -> None:
    actual: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise _error(
                ReviewedExportErrorCode.ARTIFACT_TREE_INVALID,
                "artifact tree contains a symbolic link",
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise _error(
                ReviewedExportErrorCode.ARTIFACT_TREE_INVALID,
                "artifact tree contains a non-regular entry",
            )
        actual.add(relative)
    if actual != expected:
        raise _error(
            ReviewedExportErrorCode.ARTIFACT_TREE_INVALID,
            "artifact tree contains missing or unexpected files",
        )


def _load_review_paper(review_root: Path, binding: PaperRunBinding) -> _ReviewPaper:
    for artifact in (
        binding.run_manifest,
        binding.observations,
        binding.source_manifest,
        binding.layout,
        binding.result_blocks,
    ):
        _verify_artifact(review_root, artifact.review_copy)
    if binding.candidate_lineage is not None:
        _verify_artifact(review_root, binding.candidate_lineage.review_copy)
    if binding.tuple_resolution is not None:
        _verify_artifact(review_root, binding.tuple_resolution.review_copy)
    if binding.verifier_gates is not None:
        _verify_artifact(review_root, binding.verifier_gates.review_copy)
    try:
        run_manifest = read_json(review_root / binding.run_manifest.review_copy.path)
        source_manifest = SourceManifest.model_validate(
            read_json(review_root / binding.source_manifest.review_copy.path)
        )
        layout = PdfLayout.model_validate(read_json(review_root / binding.layout.review_copy.path))
    except (OSError, ValueError, ValidationError) as exc:
        raise _error(
            ReviewedExportErrorCode.MANIFEST_INVALID,
            "copied paper binding is invalid",
        ) from exc
    if not isinstance(run_manifest, dict):
        raise _error(ReviewedExportErrorCode.MANIFEST_INVALID, "copied run manifest is invalid")
    if (
        run_manifest.get("schema_version") in {"pipeline-run/0.3", "pipeline-run/0.4"}
        and binding.candidate_lineage is None
    ):
        raise _error(
            ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
            "new-schema review binding omits candidate lineage",
        )
    if (
        source_manifest.paper_id != binding.paper_id
        or run_manifest.get("paper_id") != binding.paper_id
    ):
        raise _error(
            ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
            "copied paper identities do not match the review manifest",
        )
    if (
        layout.parser != binding.layout_parser
        or layout.parser_version != binding.layout_parser_version
    ):
        raise _error(
            ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
            "copied layout parser binding changed",
        )
    if layout.source_id not in {source.source_id for source in source_manifest.sources}:
        raise _error(
            ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
            "copied layout source is absent from source manifest",
        )
    for page in layout.pages:
        if page.text_sha256 != sha256_bytes(page.text.encode("utf-8")):
            raise _error(
                ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                "copied layout page text hash is invalid",
            )
    observations_path = review_root / binding.observations.review_copy.path
    candidates, lines = _candidate_records(observations_path)
    if len(candidates) != binding.observations.review_copy.records:
        raise _error(
            ReviewedExportErrorCode.OBSERVATION_LEDGER_INVALID,
            "copied observation count changed",
        )
    candidate_lineage = None
    if binding.candidate_lineage is not None:
        try:
            candidate_lineage = CandidateLineageArtifact.model_validate(
                read_json(review_root / binding.candidate_lineage.review_copy.path)
            )
        except (OSError, ValueError, ValidationError) as exc:
            raise _error(
                ReviewedExportErrorCode.MANIFEST_INVALID,
                "copied candidate lineage is invalid",
            ) from exc
        if (
            candidate_lineage.paper_id != binding.paper_id
            or candidate_lineage.observations_sha256 != binding.observations.sha256
        ):
            raise _error(
                ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                "copied candidate lineage does not bind observations",
            )
    tuple_resolution = None
    tuple_config = run_manifest.get("tuple_resolution")
    tuple_enabled = bool(isinstance(tuple_config, dict) and tuple_config.get("enabled") is True)
    if tuple_enabled != (binding.tuple_resolution is not None):
        raise _error(
            ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
            "review tuple-resolution binding does not match the run",
        )
    if binding.tuple_resolution is not None:
        try:
            raw_tuple = read_json(review_root / binding.tuple_resolution.review_copy.path)
        except (OSError, ValueError) as exc:
            raise _error(
                ReviewedExportErrorCode.MANIFEST_INVALID,
                "copied tuple-resolution sidecar is invalid",
            ) from exc
        sidecar = tuple_config.get("sidecar") if isinstance(tuple_config, dict) else None
        if (
            not isinstance(raw_tuple, dict)
            or not isinstance(sidecar, dict)
            or sidecar.get("sha256") != binding.tuple_resolution.sha256
            or raw_tuple.get("schema_version") != "tuple-resolution-run/0.1"
            or not isinstance(raw_tuple.get("candidate_gates"), dict)
            or not isinstance(raw_tuple.get("outcomes"), list)
            or candidate_lineage is None
            or candidate_lineage.tuple_sidecar_sha256 != binding.tuple_resolution.sha256
        ):
            raise _error(
                ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                "copied tuple-resolution sidecar is not exactly bound",
            )
        gates = raw_tuple["candidate_gates"]
        candidate_ids = {str(candidate.observation_id) for candidate in candidates}
        pass_by_id = {candidate_id: False for candidate_id in candidate_ids}
        seen_outcomes: set[str] = set()
        if set(gates) != candidate_ids:
            raise _error(
                ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                "copied tuple-resolution sidecar does not cover candidates",
            )
        for outcome in raw_tuple["outcomes"]:
            observation_id = outcome.get("observation_id") if isinstance(outcome, dict) else None
            if (
                not isinstance(observation_id, str)
                or observation_id not in gates
                or observation_id in seen_outcomes
                or outcome.get("gate_sha256") != gates[observation_id]
                or type(outcome.get("passed")) is not bool
            ):
                raise _error(
                    ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                    "copied tuple-resolution outcome binding is invalid",
                )
            seen_outcomes.add(observation_id)
            pass_by_id[observation_id] = outcome["passed"]
        lineage_by_id = {item.final_observation_id: item for item in candidate_lineage.candidates}
        if any(
            lineage_by_id.get(observation_id) is None
            or lineage_by_id[observation_id].tuple_gate_sha256 != gate_sha256
            for observation_id, gate_sha256 in gates.items()
        ):
            raise _error(
                ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                "copied candidate lineage differs from tuple gates",
            )
        tuple_resolution = {
            "sidecar_sha256": binding.tuple_resolution.sha256,
            "candidate_gates": gates,
            "passed": pass_by_id,
        }
    verifier_gates = None
    verifier_config = run_manifest.get("verifier")
    verifier_enabled = bool(
        isinstance(verifier_config, dict) and verifier_config.get("enabled") is True
    )
    if verifier_enabled != (binding.verifier_gates is not None):
        raise _error(
            ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
            "review verifier-gate binding does not match the run",
        )
    if binding.verifier_gates is not None:
        expected_verifier_schema = _expected_verifier_gate_schema(
            run_manifest.get("schema_version")
        )
        try:
            raw_verifier = read_json(review_root / binding.verifier_gates.review_copy.path)
        except (OSError, ValueError) as exc:
            raise _error(
                ReviewedExportErrorCode.MANIFEST_INVALID,
                "copied verifier-gate sidecar is invalid",
            ) from exc
        sidecar = verifier_config.get("sidecar") if isinstance(verifier_config, dict) else None
        gates = raw_verifier.get("candidate_gates") if isinstance(raw_verifier, dict) else None
        passed_ids = (
            raw_verifier.get("passed_candidate_ids") if isinstance(raw_verifier, dict) else None
        )
        candidate_ids = {str(candidate.observation_id) for candidate in candidates}
        if (
            tuple_resolution is None
            or expected_verifier_schema is None
            or not isinstance(sidecar, dict)
            or sidecar.get("sha256") != binding.verifier_gates.sha256
            or not isinstance(raw_verifier, dict)
            or raw_verifier.get("schema_version") != expected_verifier_schema
            or not isinstance(gates, dict)
            or not set(gates).issubset(candidate_ids)
            or any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in gates.values()
            )
            or not isinstance(passed_ids, list)
            or any(not isinstance(value, str) for value in passed_ids)
            or passed_ids != sorted(set(passed_ids))
            or not set(passed_ids).issubset(gates)
            or candidate_lineage is None
            or candidate_lineage.verifier_sidecar_sha256 != binding.verifier_gates.sha256
        ):
            raise _error(
                ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                "copied verifier-gate sidecar is not exactly bound",
            )
        lineage_by_id = {item.final_observation_id: item for item in candidate_lineage.candidates}
        if any(
            lineage_by_id[observation_id].verifier_gate_sha256 != gate_sha256
            for observation_id, gate_sha256 in gates.items()
        ) or any(
            item.verifier_gate_sha256 is not None and item.final_observation_id not in gates
            for item in candidate_lineage.candidates
        ):
            raise _error(
                ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                "copied candidate lineage differs from verifier gates",
            )
        passed_set = set(passed_ids)
        verifier_gates = {
            "sidecar_sha256": binding.verifier_gates.sha256,
            "candidate_gates": gates,
            "passed": {candidate_id: candidate_id in passed_set for candidate_id in candidate_ids},
        }
    return _ReviewPaper(
        binding=binding,
        run_manifest=run_manifest,
        source_manifest=source_manifest,
        layout=layout,
        candidates=candidates,
        ledger_lines=lines,
        candidate_lineage=candidate_lineage,
        tuple_resolution=tuple_resolution,
        verifier_gates=verifier_gates,
    )


def _decision_sha256(decision: ReviewedExportDecision) -> str:
    return sha256_bytes(canonical_json_bytes(model_payload(decision)))


def _expected_decision_id(item_id: str) -> str:
    return "decision_" + sha256_bytes(item_id.encode())[:24]


def _validate_item_binding(item: ReviewItem, paper: _ReviewPaper) -> None:
    fingerprint = item.fingerprint
    line_index = fingerprint.observation_ledger_line - 1
    if line_index < 0 or line_index >= len(paper.candidates):
        raise _error(
            ReviewedExportErrorCode.CANDIDATE_BINDING_INVALID,
            "candidate ledger line is outside the bound ledger",
        )
    candidate = paper.candidates[line_index]
    raw_line = paper.ledger_lines[line_index]
    if (
        fingerprint.paper_id != paper.binding.paper_id
        or fingerprint.observation_ledger_path != paper.binding.observations.run_path
        or fingerprint.observation_ledger_sha256 != paper.binding.observations.sha256
        or fingerprint.observation_ledger_line_sha256 != sha256_bytes(raw_line)
        or fingerprint.candidate_schema_version != candidate.schema_version
        or fingerprint.observation_id != candidate.observation_id
        or fingerprint.candidate_payload_sha256 != _candidate_payload_sha256(candidate)
        or fingerprint.structural_identity_sha256
        != _structural_identity_sha256(candidate, paper.layout)
        or item.candidate.model_dump(mode="json", by_alias=True, exclude_none=False)
        != candidate.model_dump(mode="json", by_alias=True, exclude_none=False)
    ):
        raise _error(
            ReviewedExportErrorCode.CANDIDATE_FINGERPRINT_MISMATCH,
            "candidate fingerprint does not match the immutable observation ledger",
        )
    expected_mode = (
        paper.candidate_lineage.export_provenance_mode
        if paper.tuple_resolution is not None and paper.candidate_lineage is not None
        else ExportProvenanceMode.LEGACY_MANUAL
    )
    if (
        item.source_export_mode is not expected_mode
        or item.source_tuple_sidecar_sha256
        != (
            paper.tuple_resolution["sidecar_sha256"] if paper.tuple_resolution is not None else None
        )
        or item.source_tuple_gate_sha256
        != (
            paper.tuple_resolution["candidate_gates"][fingerprint.observation_id]
            if paper.tuple_resolution is not None
            else None
        )
        or item.source_verifier_sidecar_sha256
        != (paper.verifier_gates["sidecar_sha256"] if paper.verifier_gates is not None else None)
        or item.source_verifier_gate_sha256
        != (
            paper.verifier_gates["candidate_gates"].get(fingerprint.observation_id)
            if paper.verifier_gates is not None
            else None
        )
        or item.source_verifier_gate_passed
        != (
            paper.verifier_gates["passed"][fingerprint.observation_id]
            if paper.verifier_gates is not None
            else None
        )
    ):
        raise _error(
            ReviewedExportErrorCode.CANDIDATE_BINDING_INVALID,
            "review item export provenance differs from its sealed source run",
        )


def _page_for_anchor(
    anchor: ReviewEvidenceAnchor, paper: _ReviewPaper, code: ReviewedExportErrorCode
) -> str:
    binding = paper.binding
    if (
        anchor.source_manifest_sha256 != binding.source_manifest.sha256
        or anchor.layout_sha256 != binding.layout.sha256
        or anchor.parser != binding.layout_parser
        or anchor.parser_version != binding.layout_parser_version
    ):
        raise _error(code, "evidence source or layout binding does not match")
    sources = {source.source_id: source for source in paper.source_manifest.sources}
    source = sources.get(anchor.source_id)
    if source is None or source.sha256 != anchor.source_sha256:
        raise _error(code, "evidence source hash does not match")
    if anchor.source_id != paper.layout.source_id or anchor.page > paper.layout.page_count:
        raise _error(code, "evidence page is unavailable in the bound layout")
    page = paper.layout.pages[anchor.page - 1]
    if (
        page.page != anchor.page
        or page.text_sha256 != anchor.page_text_sha256
        or page.text_sha256 != sha256_bytes(page.text.encode("utf-8"))
    ):
        raise _error(code, "evidence page hash does not match")
    if (
        anchor.char_end > len(page.text)
        or page.text[anchor.char_start : anchor.char_end] != anchor.exact_excerpt
    ):
        raise _error(code, "evidence excerpt is not exact at the declared offsets")
    start_line, end_line = _line_span(page.text, anchor.char_start, anchor.char_end)
    if (start_line, end_line) != (anchor.start_line, anchor.end_line):
        raise _error(code, "evidence line span does not match its exact offsets")
    expected_span_id = review_evidence_span_id(
        source_id=anchor.source_id,
        page=anchor.page,
        page_text_sha256=anchor.page_text_sha256,
        char_start=anchor.char_start,
        char_end=anchor.char_end,
        excerpt_sha256=anchor.excerpt_sha256,
    )
    if anchor.span_id != expected_span_id:
        raise _error(code, "evidence span id does not match its exact occurrence")
    return page.text


def _raw_value_supported(candidate: CandidateObservation, excerpt: str) -> bool:
    if candidate.value is None:
        return False
    quote = normalize_evidence_text(excerpt).casefold()
    raw = normalize_evidence_text(candidate.value.raw).casefold()
    if any(character.isdigit() for character in raw):
        return bounded_claim_present(quote, raw)
    return bool(raw and re.search(rf"(?<!\w){re.escape(raw)}(?!\w)", quote))


def _normalized_entity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[\u2010-\u2015\u2212]", "-", normalized)
    normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "", normalized)
    return normalize_evidence_text(normalized)


def _entity_literal_present(text: str, literal: str) -> bool:
    normalized_text = _normalized_entity_text(text)
    normalized_literal = _normalized_entity_text(literal)
    return bool(
        normalized_literal
        and re.search(
            # Reject identifier continuations such as GPT-4o, GPT-4.1,
            # GPT-4-mini, GPT-4/vision, and Model 1-XL while retaining ordinary
            # sentence punctuation after an exact entity name.
            rf"(?<!\w)(?<!\w[./-]){re.escape(normalized_literal)}(?!\w)(?![./-]\w)",
            normalized_text,
        )
    )


_NON_CURABLE_VALUE_SUPPORT_ISSUES = frozenset(
    {
        "value_numeric_projection_not_supported_by_raw",
        "value_comparator_or_uncertainty_not_supported_by_evidence_quotes",
    }
)


def _deterministic_value_support_issue(candidate: CandidateObservation) -> str | None:
    """Return a schema-independent contradiction in the candidate's value evidence."""

    if candidate.value is None:
        return None
    issues = [
        value_quote_support_issue(candidate.value, anchor.quote) for anchor in candidate.evidence
    ]
    if "value_numeric_projection_not_supported_by_raw" in issues:
        return "value_numeric_projection_not_supported_by_raw"
    if None in issues:
        return None
    if "value_comparator_or_uncertainty_not_supported_by_evidence_quotes" in issues:
        return "value_comparator_or_uncertainty_not_supported_by_evidence_quotes"
    return None


def _system_is_named(candidate: CandidateObservation, excerpt: str) -> bool:
    names = [
        role.raw_name
        for role in candidate.roles
        if role.role is ActorRole.EVALUATED_SYSTEM and role.raw_name.strip()
    ]
    return any(_entity_literal_present(excerpt, name) for name in names)


def _attested_texts(
    span_ids: list[str],
    retained_by_id: dict[str, ReviewEvidenceAnchor],
    page_text_by_span: dict[str, str],
) -> list[str]:
    """Return exact spans plus defensible whitespace-adjacent span groups."""

    anchors = [retained_by_id[span_id] for span_id in span_ids]
    texts = [anchor.exact_excerpt for anchor in anchors]
    for index, left in enumerate(anchors):
        for right in anchors[index + 1 :]:
            if left.source_id != right.source_id or left.page != right.page:
                continue
            first, second = sorted((left, right), key=lambda item: item.char_start)
            page_text = page_text_by_span[first.span_id]
            gap = page_text[first.char_end : second.char_start]
            if first.char_end <= second.char_start and not gap.strip():
                texts.append(page_text[first.char_start : second.char_end])
    return texts


@dataclass(frozen=True)
class _VisualCell:
    """One whitespace-delimited cell in the frozen monospace page layout."""

    source_id: str
    page: int
    line: int
    start_column: int
    end_column: int
    text: str
    span_id: str


def _literal_match_spans(text: str, literal: object) -> list[tuple[int, int]]:
    """Return boundary-aware literal spans without discarding version digits."""

    tokens = re.findall(
        r"[a-z0-9]+",
        unicodedata.normalize("NFKC", str(literal)).casefold(),
    )
    if not tokens:
        return []
    pattern = re.compile(
        r"(?<!\w)" + r"\W+".join(re.escape(token) for token in tokens) + r"(?!\w)",
        re.IGNORECASE,
    )
    return [match.span() for match in pattern.finditer(text)]


def _literal_is_attested(texts: list[str], literal: object) -> bool:
    return any(_literal_match_spans(text, literal) for text in texts)


def _anchor_visual_cells(
    anchor: ReviewEvidenceAnchor,
    page_text: str,
) -> list[_VisualCell]:
    """Project an exact span into its visually separated monospace cells."""

    assert anchor.span_id is not None
    cells: list[_VisualCell] = []
    line_start = page_text.rfind("\n", 0, anchor.char_start) + 1
    line_number = page_text.count("\n", 0, line_start) + 1
    while line_start < anchor.char_end:
        newline = page_text.find("\n", line_start)
        line_end = len(page_text) if newline < 0 else newline
        segment_start = max(anchor.char_start, line_start)
        segment_end = min(anchor.char_end, line_end)
        segment = page_text[segment_start:segment_end]
        cursor = 0
        for part in re.split(r"[ \t]{2,}", segment):
            part_start = segment.find(part, cursor)
            cursor = part_start + len(part)
            stripped = part.strip()
            if not stripped:
                continue
            leading = len(part) - len(part.lstrip())
            start_column = segment_start - line_start + part_start + leading
            cells.append(
                _VisualCell(
                    source_id=anchor.source_id,
                    page=anchor.page,
                    line=line_number,
                    start_column=start_column,
                    end_column=start_column + len(stripped),
                    text=stripped,
                    span_id=anchor.span_id,
                )
            )
        if newline < 0 or newline >= anchor.char_end:
            break
        line_start = newline + 1
        line_number += 1
    return cells


def _visual_cells(
    anchors: list[ReviewEvidenceAnchor],
    page_text_by_span: dict[str, str],
) -> list[_VisualCell]:
    return [
        cell
        for anchor in anchors
        for cell in _anchor_visual_cells(anchor, page_text_by_span[anchor.span_id])
    ]


def _visual_cell_texts(cells: list[_VisualCell]) -> list[str]:
    """Include exact cells and same-column continuations on consecutive lines."""

    texts = [cell.text for cell in cells]
    for first in cells:
        group = [first]
        current = first
        for candidate in sorted(cells, key=lambda item: (item.line, item.start_column)):
            if (
                candidate.source_id == current.source_id
                and candidate.page == current.page
                and candidate.line == current.line + 1
                and abs(candidate.start_column - current.start_column) <= 4
                and candidate.span_id != current.span_id
            ):
                group.append(candidate)
                current = candidate
        if len(group) > 1:
            texts.append("\n".join(cell.text for cell in group))
    return list(dict.fromkeys(texts))


def _sample_count_is_attested(texts: list[str], sample_count: int) -> bool:
    forms = {str(sample_count), f"{sample_count:,}"}
    return any(
        re.search(
            rf"(?<![\w.])(?:{'|'.join(re.escape(value) for value in forms)})"
            r"(?![\w.])\s+(?:labeled\s+)?(?:samples?|examples?|instances?|records?|comments?)\b",
            text,
            re.IGNORECASE,
        )
        for text in texts
    )


def _raw_scope_is_composed_from_attested_parts(
    candidate: CandidateObservation,
    texts: list[str],
) -> bool:
    scope = candidate.scope
    if scope is None or not scope.raw_scope:
        return True
    if _literal_is_attested(texts, scope.raw_scope):
        return True
    if not _literal_is_attested(texts, scope.dataset_raw):
        return False
    values: list[object] = [scope.dataset_raw]
    values.extend(
        value
        for value in (
            scope.split,
            scope.subset,
            scope.group,
            scope.language,
            scope.aggregation,
            scope.sample_count,
        )
        if value is not None and value != ""
    )

    def tokens(value: object) -> set[str]:
        normalized = unicodedata.normalize("NFKC", str(value)).casefold().replace(",", "")
        return set(re.findall(r"[a-z0-9]+", normalized))

    supported = set().union(*(tokens(value) for value in values))
    supported.update(
        {
            "aggregation",
            "data",
            "dataset",
            "evaluation",
            "group",
            "language",
            "raw",
            "sample",
            "samples",
            "scope",
            "set",
            "sets",
            "split",
            "test",
            "train",
            "training",
            "validation",
        }
    )
    return tokens(scope.raw_scope) <= supported


def _table_setting_projection_supported(
    candidate: CandidateObservation,
    texts: list[str],
) -> bool:
    """Validate every table setting primitive from exact visual cells."""

    scope = candidate.scope
    owners: list[object] = [
        role.raw_name for role in candidate.roles if role.role is ActorRole.EVALUATED_SYSTEM
    ]
    if scope is not None:
        owners.append(scope.dataset_raw)
    if candidate.metric is not None:
        owners.append(candidate.metric.raw_name)
    if not any(_literal_is_attested(texts, owner) for owner in owners):
        return False
    if scope is not None:
        for value in (
            scope.split,
            scope.subset,
            scope.group,
            scope.language,
            scope.aggregation,
        ):
            if value is not None and value != "" and not _literal_is_attested(texts, value):
                return False
        if scope.sample_count is not None and not _sample_count_is_attested(
            texts, scope.sample_count
        ):
            return False
        if not _raw_scope_is_composed_from_attested_parts(candidate, texts):
            return False
    if candidate.metric is not None and candidate.metric.parameters:
        for key, value in candidate.metric.parameters.items():
            if not any(
                _literal_match_spans(text, key) and _literal_match_spans(text, value)
                for text in texts
            ):
                return False
    if any(role.role is not ActorRole.EVALUATED_SYSTEM for role in candidate.roles):
        return False
    for value in (
        candidate.operationalization,
        candidate.decision_rule,
        candidate.evaluation_date,
    ):
        if value is not None and value != "" and not _literal_is_attested(texts, value):
            return False
    construct = candidate.evaluation_construct
    if construct is None or construct == "" or _literal_is_attested(texts, construct):
        return True
    return bool(
        normalize_evidence_text(construct).casefold() == "content moderation accuracy"
        and candidate.metric is not None
        and candidate.metric.canonical_id == "accuracy"
        and _literal_is_attested(texts, candidate.metric.raw_name)
        and any("moderation service" in normalize_evidence_text(text).casefold() for text in texts)
    )


def _table_field_attestation_has_direct_support(
    *,
    field: CandidateField,
    candidate: CandidateObservation,
    anchors: list[ReviewEvidenceAnchor],
    page_text_by_span: dict[str, str],
) -> bool:
    if not anchors or any(anchor.kind is not EvidenceKind.TABLE for anchor in anchors):
        return False
    cells = _visual_cells(anchors, page_text_by_span)
    texts = _visual_cell_texts(cells)
    if field is CandidateField.SETTING:
        return _table_setting_projection_supported(candidate, texts)
    return False


def _attested_candidate(
    candidate: CandidateObservation,
    anchors: list[ReviewEvidenceAnchor],
) -> CandidateObservation:
    evidence = [
        EvidenceAnchor(
            source_id=anchor.source_id,
            page=anchor.page,
            kind=anchor.kind,
            label=anchor.label,
            row=anchor.row,
            column=anchor.column,
            region_id=anchor.region_id,
            quote=anchor.exact_excerpt,
            bounding_box=anchor.bounding_box,
        )
        for anchor in anchors
    ]
    return candidate.model_copy(deep=True, update={"evidence": evidence})


def _candidate_with_reviewed_tuple(
    candidate: CandidateObservation, reviewed: ReviewedTuple
) -> CandidateObservation:
    """Project the immutable reviewed tuple onto its sealed candidate evidence."""

    return candidate.model_copy(
        deep=True,
        update={
            "roles": [role.model_copy(deep=True) for role in reviewed.roles],
            "scope": reviewed.scope.model_copy(deep=True) if reviewed.scope else None,
            "metric": reviewed.metric.model_copy(deep=True) if reviewed.metric else None,
            "value": reviewed.value.model_copy(deep=True) if reviewed.value else None,
            "evaluation_construct": reviewed.evaluation_construct,
            "operationalization": reviewed.operationalization,
            "decision_rule": reviewed.decision_rule,
            "evaluation_date": reviewed.evaluation_date,
            "field_provenance": [
                binding.model_copy(deep=True) for binding in reviewed.field_provenance
            ],
        },
    )


def _attested_derivations(candidate: CandidateObservation) -> dict[CandidateField, str]:
    """Recompute the only metadata exemptions allowed for exact review spans."""

    if candidate.metric is None or candidate.value is None:
        return {}
    metric_seed = candidate.metric.model_copy(
        update={
            "canonical_id": None,
            "kind": None,
            "lower_is_better": None,
            "min_score": None,
            "max_score": None,
        }
    )
    inferred_metric, _, _ = resolve_metric_value(
        metric_seed,
        candidate.value,
        (anchor.quote for anchor in candidate.evidence),
    )
    derived: dict[CandidateField, str] = {}
    metric_fields = ("canonical_id", "kind", "lower_is_better", "min_score", "max_score")
    if any(getattr(inferred_metric, name) is not None for name in metric_fields) and all(
        getattr(candidate.metric, name) == getattr(inferred_metric, name) for name in metric_fields
    ):
        derived[CandidateField.METRIC] = "deterministic_reference_resolution=metric_registry"
    unit_seed = metric_seed.model_copy(update={"unit": None})
    value_seed = candidate.value.model_copy(update={"unit": None})
    inferred_metric, inferred_value, note = resolve_metric_value(
        unit_seed,
        value_seed,
        (anchor.quote for anchor in candidate.evidence),
    )
    if (
        note
        and candidate.metric.unit == inferred_metric.unit
        and candidate.value.unit == inferred_value.unit
    ):
        suffix = re.sub(r"[^a-z0-9]+", "_", note.casefold()).strip("_")
        derived[CandidateField.UNIT] = f"deterministic_reference_resolution={suffix}"
    return derived


def _field_attestation_has_direct_support(
    *,
    field: CandidateField,
    candidate: CandidateObservation,
    anchors: list[ReviewEvidenceAnchor],
    texts: list[str],
    page_text_by_span: dict[str, str],
) -> bool:
    """Re-prove the complete field projection from only its attested exact spans."""

    attested = _attested_candidate(candidate, anchors)
    status, _, _ = direct_quote_field_binding(
        attested,
        field,
        derived_reasons=_attested_derivations(attested),
    )
    if status is FieldBindingStatus.BOUND:
        return True
    if _table_field_attestation_has_direct_support(
        field=field,
        candidate=candidate,
        anchors=anchors,
        page_text_by_span=page_text_by_span,
    ):
        return True
    if field is CandidateField.SYSTEM:
        evaluated = [role for role in candidate.roles if role.role is ActorRole.EVALUATED_SYSTEM]
        return bool(evaluated) and all(
            role.canonical_id is None
            and role.version is None
            and role.provider is None
            and any(_entity_literal_present(text, role.raw_name) for text in texts)
            for role in evaluated
        )
    return False


def _validate_field_attestations(
    decision: ReviewedExportDecision,
    candidate: CandidateObservation,
    page_text_by_span: dict[str, str],
) -> None:
    """Validate a confirmed tuple's complete field-to-exact-span mapping."""

    if decision.tuple_decision is not TupleDecision.CONFIRMED:
        return
    candidate = _candidate_with_reviewed_tuple(candidate, decision.reviewed_tuple)
    populated = decision.reviewed_tuple.populated_fields()
    by_field = {attestation.field: attestation for attestation in decision.field_attestations}
    if set(by_field) != populated:
        raise _error(
            ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID,
            "field attestations do not exactly cover every populated tuple field",
        )
    expected_hashes = decision.reviewed_tuple.field_value_sha256s()
    if any(
        attestation.value_sha256 != expected_hashes[field]
        for field, attestation in by_field.items()
    ):
        raise _error(
            ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID,
            "field attestation value hash differs from the unchanged reviewed tuple",
        )
    retained_span_ids = [anchor.span_id for anchor in decision.result_evidence]
    if None in retained_span_ids or len(retained_span_ids) != len(set(retained_span_ids)):
        raise _error(
            ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID,
            "result evidence contains a missing or duplicated exact span identity",
        )
    retained = set(retained_span_ids)
    if any(
        span_id not in retained
        for attestation in decision.field_attestations
        for span_id in attestation.span_ids
    ):
        raise _error(
            ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID,
            "field attestation refers to an exact span not retained as result evidence",
        )
    retained_by_id = {
        anchor.span_id: anchor for anchor in decision.result_evidence if anchor.span_id is not None
    }
    for field, attestation in by_field.items():
        attested_anchors = [retained_by_id[span_id] for span_id in attestation.span_ids]
        if not _field_attestation_has_direct_support(
            field=field,
            candidate=candidate,
            anchors=attested_anchors,
            texts=_attested_texts(
                attestation.span_ids,
                retained_by_id,
                page_text_by_span,
            ),
            page_text_by_span=page_text_by_span,
        ):
            raise _error(
                ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID,
                f"{field.value} attestation lacks direct support for its field primitive",
            )
    value_attestation = by_field.get(CandidateField.VALUE)
    if value_attestation is None or not any(
        _raw_value_supported(candidate, retained_by_id[span_id].exact_excerpt)
        for span_id in value_attestation.span_ids
    ):
        raise _error(
            ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID,
            "value attestation does not itself cite an exact span supporting the raw value",
        )
    system_attestation = by_field.get(CandidateField.SYSTEM)
    if system_attestation is None or not _attestations_share_coherent_result_occurrence(
        candidate=candidate,
        system_span_ids=system_attestation.span_ids,
        value_span_ids=value_attestation.span_ids,
        retained_by_id=retained_by_id,
        page_text_by_span=page_text_by_span,
    ):
        raise _error(
            ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID,
            "system and value attestations do not share a coherent result occurrence",
        )
    _validate_attested_tuple_association(
        candidate=candidate,
        populated=populated,
        by_field=by_field,
        retained_by_id=retained_by_id,
        page_text_by_span=page_text_by_span,
    )


def _attestations_share_coherent_result_occurrence(
    *,
    candidate: CandidateObservation,
    system_span_ids: list[str],
    value_span_ids: list[str],
    retained_by_id: dict[str, ReviewEvidenceAnchor],
    page_text_by_span: dict[str, str],
) -> bool:
    """Bind system and value to one exact span or whitespace-adjacent wrap."""

    def atomic_clause_support(text: str) -> bool:
        if candidate.value is None:
            return False
        clauses = [
            clause.strip()
            for clause in re.split(r"(?:[\r\n;；]+|\s+\|\s+)", text)
            if clause.strip()
        ]
        return any(
            _system_is_named(candidate, clause)
            and value_quote_support_issue(candidate.value, clause) is None
            for clause in clauses
        )

    for system_span_id in system_span_ids:
        system_span = retained_by_id[system_span_id]
        for value_span_id in value_span_ids:
            value_span = retained_by_id[value_span_id]
            if system_span.source_id != value_span.source_id or system_span.page != value_span.page:
                continue
            page_text = page_text_by_span[system_span_id]
            start = min(system_span.char_start, value_span.char_start)
            end = max(system_span.char_end, value_span.char_end)
            if system_span.char_end <= value_span.char_start:
                gap = page_text[system_span.char_end : value_span.char_start]
            elif value_span.char_end <= system_span.char_start:
                gap = page_text[value_span.char_end : system_span.char_start]
            else:
                gap = ""
            if gap.strip():
                continue
            combined = page_text[start:end]
            if candidate.value is None or value_quote_support_issue(candidate.value, combined):
                continue
            same_span = system_span_id == value_span_id
            atomic_support = atomic_clause_support(combined)
            combined_system_support = _system_is_named(candidate, combined)
            first, second = sorted(
                (system_span, value_span), key=lambda item: (item.char_start, item.char_end)
            )
            hyphenated_wrap = bool(
                first.exact_excerpt.rstrip().endswith("-")
                and re.match(r"^\s*[A-Za-z0-9]", second.exact_excerpt)
            )
            split_system_support = combined_system_support and not (
                _system_is_named(candidate, system_span.exact_excerpt)
                or _system_is_named(candidate, value_span.exact_excerpt)
            )
            if (same_span and atomic_support) or (hyphenated_wrap and split_system_support):
                return True
    return False


def _attested_physical_group_texts(
    span_ids: set[str],
    retained_by_id: dict[str, ReviewEvidenceAnchor],
    page_text_by_span: dict[str, str],
) -> list[str]:
    """Join only overlapping or whitespace-adjacent validated exact spans."""

    by_page: dict[tuple[str, int], list[ReviewEvidenceAnchor]] = defaultdict(list)
    for span_id in span_ids:
        anchor = retained_by_id[span_id]
        by_page[(anchor.source_id, anchor.page)].append(anchor)
    groups: list[str] = []
    for anchors in by_page.values():
        ordered = sorted(anchors, key=lambda item: (item.char_start, item.char_end))
        page_text = page_text_by_span[ordered[0].span_id]
        group_start = ordered[0].char_start
        group_end = ordered[0].char_end
        for anchor in ordered[1:]:
            gap = page_text[group_end : anchor.char_start] if group_end <= anchor.char_start else ""
            if anchor.char_start <= group_end or not gap.strip():
                group_end = max(group_end, anchor.char_end)
                continue
            groups.append(page_text[group_start:group_end])
            group_start, group_end = anchor.char_start, anchor.char_end
        groups.append(page_text[group_start:group_end])
    return groups


_RESULT_LIKE_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+)%?(?![\w.])")


def _cell_has_only_structural_numbers(
    cell: _VisualCell,
    claim_span: tuple[int, int],
    candidate: CandidateObservation,
) -> bool:
    """Reject a supposed header cell that is actually another result row."""

    for match in _RESULT_LIKE_NUMBER.finditer(cell.text):
        if claim_span[0] <= match.start() and match.end() <= claim_span[1]:
            continue
        prefix = cell.text[max(0, match.start() - 16) : match.start()]
        suffix = cell.text[match.end() : match.end() + 16]
        if re.search(r"(?:table|figure|fig\.?|section|sec\.?)\s*$", prefix, re.IGNORECASE):
            continue
        if (
            cell.text[max(0, match.start() - 1) : match.start()] == "["
            and cell.text[match.end() : match.end() + 1] == "]"
        ):
            continue
        if re.match(r"\s+(?:samples?|examples?|instances?|records?|comments?)\b", suffix, re.I):
            scope = candidate.scope
            if scope is not None and scope.sample_count is not None:
                printed = match.group().replace(",", "").rstrip("%")
                if printed == str(scope.sample_count):
                    continue
        return False
    return True


def _cells_are_column_aligned(
    value_cell: _VisualCell,
    value_span: tuple[int, int],
    header_cell: _VisualCell,
    header_span: tuple[int, int],
) -> bool:
    value_center = value_cell.start_column + sum(value_span) / 2
    header_center = header_cell.start_column + sum(header_span) / 2
    return abs(value_center - header_center) <= 5


def _table_tuple_association_supported(
    *,
    candidate: CandidateObservation,
    by_field: dict[CandidateField, Any],
    retained_by_id: dict[str, ReviewEvidenceAnchor],
    page_text_by_span: dict[str, str],
) -> bool:
    """Prove a human-selected table cell through frozen fixed-width alignment."""

    if candidate.value is None:
        return False
    system_ids = set(by_field[CandidateField.SYSTEM].span_ids)
    value_ids = set(by_field[CandidateField.VALUE].span_ids)
    result_rows = [
        retained_by_id[span_id]
        for span_id in sorted(system_ids & value_ids)
        if retained_by_id[span_id].kind is EvidenceKind.TABLE
        and _system_is_named(candidate, retained_by_id[span_id].exact_excerpt)
        and value_quote_support_issue(candidate.value, retained_by_id[span_id].exact_excerpt)
        is None
    ]
    if not result_rows:
        return False
    attested_ids = {
        span_id for attestation in by_field.values() for span_id in attestation.span_ids
    }
    anchors = [retained_by_id[span_id] for span_id in sorted(attested_ids)]
    if any(anchor.kind is not EvidenceKind.TABLE for anchor in anchors):
        return False
    discriminators: list[object] = []
    if candidate.metric is not None:
        discriminators.append(candidate.metric.raw_name)
    if candidate.scope is not None:
        discriminators.extend(
            value
            for value in (
                candidate.scope.dataset_raw,
                candidate.scope.split,
                candidate.scope.subset,
                candidate.scope.group,
                candidate.scope.language,
                candidate.scope.aggregation,
            )
            if value is not None and value != ""
        )
    for result_row in result_rows:
        if any(
            anchor.source_id != result_row.source_id or anchor.page != result_row.page
            for anchor in anchors
        ):
            continue
        page_text = page_text_by_span[result_row.span_id]
        row_cells = _anchor_visual_cells(result_row, page_text)
        value_occurrences = [
            (cell, span)
            for cell in row_cells
            for span in _literal_match_spans(cell.text, candidate.value.raw)
        ]
        if not value_occurrences:
            continue
        context = [
            anchor
            for anchor in anchors
            if anchor.span_id != result_row.span_id
            and anchor.source_id == result_row.source_id
            and anchor.page == result_row.page
        ]
        context_cells = _visual_cells(context, page_text_by_span)
        for header_cell in context_cells:
            for discriminator in discriminators:
                for header_span in _literal_match_spans(header_cell.text, discriminator):
                    if not _cell_has_only_structural_numbers(header_cell, header_span, candidate):
                        continue
                    if any(
                        _cells_are_column_aligned(
                            value_cell,
                            value_span,
                            header_cell,
                            header_span,
                        )
                        for value_cell, value_span in value_occurrences
                    ):
                        return True
    return False


def _prose_context_has_only_structural_numbers(text: str) -> bool:
    section_prefix = re.match(r"^\s*\d+(?:\.\d+){1,3}\b", text)
    for match in _RESULT_LIKE_NUMBER.finditer(text):
        if section_prefix is not None and match.start() < section_prefix.end():
            continue
        prefix = text[max(0, match.start() - 12) : match.start()]
        if re.search(r"(?:table|figure|fig\.?|section|sec\.?)\s*$", prefix, re.IGNORECASE):
            continue
        return False
    return True


def _prose_visual_continuation_supported(
    *,
    candidate: CandidateObservation,
    by_field: dict[CandidateField, Any],
    retained_by_id: dict[str, ReviewEvidenceAnchor],
    page_text_by_span: dict[str, str],
) -> bool:
    """Join only consecutive same-column prose fragments in the frozen layout."""

    if candidate.value is None:
        return False
    system_ids = set(by_field[CandidateField.SYSTEM].span_ids)
    value_ids = set(by_field[CandidateField.VALUE].span_ids)
    result_ids = {
        span_id
        for span_id in system_ids & value_ids
        if _system_is_named(candidate, retained_by_id[span_id].exact_excerpt)
        and value_quote_support_issue(candidate.value, retained_by_id[span_id].exact_excerpt)
        is None
    }
    if not result_ids:
        return False
    attested_ids = {
        span_id for attestation in by_field.values() for span_id in attestation.span_ids
    }
    anchors = sorted(
        (retained_by_id[span_id] for span_id in attested_ids),
        key=lambda item: (item.source_id, item.page, item.char_start, item.char_end),
    )
    if len(anchors) < 2 or any(anchor.kind is not EvidenceKind.PROSE for anchor in anchors):
        return False
    if len({(anchor.source_id, anchor.page) for anchor in anchors}) != 1:
        return False
    for left, right in zip(anchors, anchors[1:], strict=False):
        if right.start_line != left.end_line + 1:
            return False
        page_text = page_text_by_span[left.span_id]
        left_cells = _anchor_visual_cells(left, page_text)
        right_cells = _anchor_visual_cells(right, page_text)
        if not left_cells or not right_cells:
            return False
        left_edge = min(cell.start_column for cell in left_cells if cell.line == left.end_line)
        right_edge = min(cell.start_column for cell in right_cells if cell.line == right.start_line)
        if abs(left_edge - right_edge) > 4:
            return False
        left_text = left.exact_excerpt.rstrip()
        right_text = right.exact_excerpt.lstrip()
        if (
            not left_text
            or not right_text
            or not (
                left_text.endswith("-") or (left_text[-1].isalnum() and right_text[0].islower())
            )
        ):
            return False
    return all(
        span_id in result_ids
        or _prose_context_has_only_structural_numbers(retained_by_id[span_id].exact_excerpt)
        for span_id in attested_ids
    )


def _validate_attested_tuple_association(
    *,
    candidate: CandidateObservation,
    populated: set[CandidateField],
    by_field: dict[CandidateField, Any],
    retained_by_id: dict[str, ReviewEvidenceAnchor],
    page_text_by_span: dict[str, str],
) -> None:
    """Require one physical span group to support the complete atomic tuple."""

    fields = {
        CandidateField.SYSTEM,
        CandidateField.DATASET_SCOPE,
        CandidateField.METRIC,
        CandidateField.VALUE,
    }
    if CandidateField.SETTING in populated:
        fields.add(CandidateField.SETTING)
    span_ids = {
        span_id for field in fields if field in by_field for span_id in by_field[field].span_ids
    }
    texts = _attested_physical_group_texts(span_ids, retained_by_id, page_text_by_span)
    if not (
        direct_quote_tuple_group_association_supported(candidate, texts)
        or _prose_visual_continuation_supported(
            candidate=candidate,
            by_field=by_field,
            retained_by_id=retained_by_id,
            page_text_by_span=page_text_by_span,
        )
        or _table_tuple_association_supported(
            candidate=candidate,
            by_field=by_field,
            retained_by_id=retained_by_id,
            page_text_by_span=page_text_by_span,
        )
    ):
        raise _error(
            ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID,
            "field attestations do not support one coherent atomic tuple occurrence",
        )


def _validate_decision_evidence(
    decision: ReviewedExportDecision, item: ReviewItem, paper: _ReviewPaper
) -> None:
    candidate = item.candidate
    raw_value_supported = False
    page_text_by_span: dict[str, str] = {}
    for anchor in decision.result_evidence:
        page_text = _page_for_anchor(anchor, paper, ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID)
        assert anchor.span_id is not None
        page_text_by_span[anchor.span_id] = page_text
        if not any(
            original.source_id == anchor.source_id
            and original.page == anchor.page
            and original.kind is anchor.kind
            for original in candidate.evidence
        ):
            raise _error(
                ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID,
                "result evidence is not bound to a candidate result location",
            )
        raw_value_supported = raw_value_supported or _raw_value_supported(
            candidate, anchor.exact_excerpt
        )
    if not raw_value_supported:
        raise _error(
            ReviewedExportErrorCode.RESULT_EVIDENCE_INVALID,
            "result evidence bundle does not support the candidate raw value",
        )
    _validate_field_attestations(decision, candidate, page_text_by_span)
    for anchor in decision.origin_evidence:
        _page_for_anchor(anchor, paper, ReviewedExportErrorCode.ORIGIN_EVIDENCE_INVALID)
        if not _system_is_named(candidate, anchor.exact_excerpt):
            raise _error(
                ReviewedExportErrorCode.ORIGIN_EVIDENCE_INVALID,
                "origin evidence is not candidate-specific to the evaluated system",
            )


def _load_review(review_root: Path, *, require_lock: bool) -> ValidatedReview:
    manifest_path = _confined_file(review_root, REVIEW_MANIFEST_NAME)
    try:
        manifest = ReviewManifest.model_validate(read_json(manifest_path))
    except (OSError, ValueError, ValidationError) as exc:
        raise _error(
            ReviewedExportErrorCode.MANIFEST_INVALID, "review manifest is invalid"
        ) from exc
    manifest_sha = sha256_file(manifest_path)
    lock_path = review_root / REVIEW_LOCK_NAME
    locked = lock_path.is_file() and not lock_path.is_symlink()
    if require_lock and not locked:
        raise _error(ReviewedExportErrorCode.REVIEW_NOT_LOCKED, "review has no valid lock")
    _validate_exact_tree(review_root, _expected_review_files(manifest, locked=locked))
    _verify_artifact(review_root, manifest.protocol)
    _verify_artifact(review_root, manifest.items)
    if manifest.corpus_run is not None:
        _verify_artifact(review_root, manifest.corpus_run.review_copy)
    papers = {
        binding.paper_id: _load_review_paper(review_root, binding) for binding in manifest.papers
    }
    items = _jsonl_records(
        _confined_file(review_root, manifest.items.path), ReviewItem, "review items"
    )
    decisions = _jsonl_records(
        _confined_file(review_root, manifest.decision_template.path),
        ReviewedExportDecision,
        "review decisions",
    )
    if len(items) != manifest.item_count or len(decisions) != manifest.item_count:
        raise _error(
            ReviewedExportErrorCode.DECISION_SET_INVALID,
            "item and decision counts do not match the review manifest",
        )
    if len({item.item_id for item in items}) != len(items):
        raise _error(ReviewedExportErrorCode.DECISION_SET_INVALID, "review item IDs are duplicated")
    if [decision.item_id for decision in decisions] != [item.item_id for item in items]:
        raise _error(
            ReviewedExportErrorCode.DECISION_SET_INVALID,
            "decision membership or order differs from immutable items",
        )
    if len({decision.decision_id for decision in decisions}) != len(decisions):
        raise _error(
            ReviewedExportErrorCode.DECISION_SET_INVALID,
            "review decision IDs are duplicated",
        )
    for item, decision in zip(items, decisions, strict=True):
        paper = papers.get(item.fingerprint.paper_id)
        if paper is None:
            raise _error(
                ReviewedExportErrorCode.CANDIDATE_BINDING_INVALID,
                "review item refers to an unbound paper",
            )
        _validate_item_binding(item, paper)
        if (
            decision.decision_id != _expected_decision_id(item.item_id)
            or decision.candidate_payload_sha256 != item.fingerprint.candidate_payload_sha256
        ):
            raise _error(
                ReviewedExportErrorCode.DECISION_BINDING_MISMATCH,
                "decision is stale or bound to another candidate",
            )
        if _tuple_sha256(decision.reviewed_tuple) != _tuple_sha256(item.reviewed_tuple):
            raise _error(
                ReviewedExportErrorCode.TUPLE_MISMATCH,
                "reviewed tuple differs from the deterministic candidate tuple",
            )
        if decision.status is DecisionStatus.PENDING:
            if [model_payload(anchor) for anchor in decision.result_evidence] != [
                model_payload(anchor) for anchor in item.suggested_result_evidence
            ]:
                raise _error(
                    ReviewedExportErrorCode.DECISION_BINDING_MISMATCH,
                    "pending decision changed its immutable evidence suggestions",
                )
            continue
        assert decision.authority is not None
        if (
            decision.authority.protocol_id != manifest.protocol_id
            or decision.authority.protocol_sha256 != manifest.protocol.sha256
        ):
            raise _error(
                ReviewedExportErrorCode.AUTHORITY_INVALID,
                "decision authority does not bind the review protocol",
            )
        _validate_decision_evidence(decision, item, paper)

    lock: ReviewLock | None = None
    lock_sha: str | None = None
    if locked:
        try:
            lock = ReviewLock.model_validate(read_json(lock_path))
        except (OSError, ValueError, ValidationError) as exc:
            raise _error(
                ReviewedExportErrorCode.REVIEW_LOCK_MISMATCH, "review lock is invalid"
            ) from exc
        lock_sha = sha256_file(lock_path)
        expected_decisions = [
            LockedDecision(
                decision_id=decision.decision_id,
                item_id=decision.item_id,
                decision_sha256=_decision_sha256(decision),
                status=decision.status,
            )
            for decision in decisions
        ]
        completed = sum(decision.status is DecisionStatus.COMPLETED for decision in decisions)
        decided_times = [
            decision.decided_at
            for decision in decisions
            if decision.status is DecisionStatus.COMPLETED and decision.decided_at is not None
        ]
        expected_lock = ReviewLock(
            review_manifest_sha256=manifest_sha,
            items_sha256=manifest.items.sha256,
            decisions_sha256=sha256_file(
                _confined_file(review_root, manifest.decision_template.path)
            ),
            protocol_sha256=manifest.protocol.sha256,
            source_run_seal_sha256=manifest.run_seal.seal_sha256,
            source_run_tree_sha256=manifest.run_seal.tree_sha256,
            decision_count=len(decisions),
            completed_count=completed,
            pending_count=len(decisions) - completed,
            locked_at=max(decided_times) if decided_times else None,
            decisions=expected_decisions,
        )
        if model_payload(lock) != model_payload(expected_lock):
            raise _error(
                ReviewedExportErrorCode.REVIEW_LOCK_MISMATCH,
                "review lock does not match exact decisions and inputs",
            )
    return ValidatedReview(
        root=review_root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        items=items,
        decisions=decisions,
        papers=papers,
        lock=lock,
        lock_sha256=lock_sha,
    )


def validate_export_review(review_root: Path) -> tuple[ReviewLock, str]:
    """Validate exact private decisions and lock their byte representation."""

    validated = _load_review(review_root, require_lock=False)
    if validated.lock is not None:
        assert validated.lock_sha256 is not None
        return validated.lock, validated.lock_sha256
    completed = sum(decision.status is DecisionStatus.COMPLETED for decision in validated.decisions)
    decided_times = [
        decision.decided_at
        for decision in validated.decisions
        if decision.status is DecisionStatus.COMPLETED and decision.decided_at is not None
    ]
    lock = ReviewLock(
        review_manifest_sha256=validated.manifest_sha256,
        items_sha256=validated.manifest.items.sha256,
        decisions_sha256=sha256_file(review_root / validated.manifest.decision_template.path),
        protocol_sha256=validated.manifest.protocol.sha256,
        source_run_seal_sha256=validated.manifest.run_seal.seal_sha256,
        source_run_tree_sha256=validated.manifest.run_seal.tree_sha256,
        decision_count=len(validated.decisions),
        completed_count=completed,
        pending_count=len(validated.decisions) - completed,
        locked_at=max(decided_times) if decided_times else None,
        decisions=[
            LockedDecision(
                decision_id=decision.decision_id,
                item_id=decision.item_id,
                decision_sha256=_decision_sha256(decision),
                status=decision.status,
            )
            for decision in validated.decisions
        ],
    )
    lock_sha = write_json(review_root / REVIEW_LOCK_NAME, lock)
    (review_root / REVIEW_LOCK_NAME).chmod(0o444)
    (review_root / validated.manifest.decision_template.path).chmod(0o444)
    reloaded = _load_review(review_root, require_lock=True)
    if reloaded.lock_sha256 != lock_sha:
        raise _error(
            ReviewedExportErrorCode.REVIEW_LOCK_MISMATCH,
            "persisted review lock differs from validated lock",
        )
    assert reloaded.lock is not None
    return reloaded.lock, lock_sha


def _bind_original_run(run_root: Path, review: ValidatedReview) -> dict[str, _RunPaper]:
    try:
        verified = verify_run_seal(run_root)
    except RunSealVerificationError as exc:
        raise _error(
            ReviewedExportErrorCode.RUN_SEAL_INVALID,
            "source run seal verification failed during composition",
        ) from exc
    if (
        verified.seal_sha256 != review.manifest.run_seal.seal_sha256
        or verified.tree_sha256 != review.manifest.run_seal.tree_sha256
        or verified.file_count != review.manifest.run_seal.file_count
        or verified.total_bytes != review.manifest.run_seal.total_bytes
    ):
        raise _error(
            ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
            "source run differs from the review manifest",
        )
    run_kind, papers, corpus_run_path = _load_run_papers(run_root)
    if run_kind != review.manifest.run_kind:
        raise _error(ReviewedExportErrorCode.RUN_BINDING_MISMATCH, "source run kind changed")
    if review.manifest.corpus_run is not None and (
        corpus_run_path is None or sha256_file(corpus_run_path) != review.manifest.corpus_run.sha256
    ):
        raise _error(
            ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
            "source corpus manifest changed",
        )
    originals = {paper.paper_id: paper for paper in papers}
    if set(originals) != {paper.paper_id for paper in review.manifest.papers}:
        raise _error(ReviewedExportErrorCode.RUN_BINDING_MISMATCH, "source paper set changed")
    for binding in review.manifest.papers:
        paper = originals[binding.paper_id]
        paths = {
            binding.run_manifest.run_path: paper.run_path,
            binding.observations.run_path: paper.observations_path,
            binding.source_manifest.run_path: paper.source_manifest_path,
            binding.layout.run_path: paper.layout_path,
            binding.result_blocks.run_path: paper.result_blocks_path,
        }
        expected = {
            binding.run_manifest.run_path: binding.run_manifest.sha256,
            binding.observations.run_path: binding.observations.sha256,
            binding.source_manifest.run_path: binding.source_manifest.sha256,
            binding.layout.run_path: binding.layout.sha256,
            binding.result_blocks.run_path: binding.result_blocks.sha256,
        }
        if binding.candidate_lineage is not None:
            if paper.candidate_lineage_path is None:
                raise _error(
                    ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                    "source candidate lineage disappeared",
                )
            paths[binding.candidate_lineage.run_path] = paper.candidate_lineage_path
            expected[binding.candidate_lineage.run_path] = binding.candidate_lineage.sha256
        if binding.tuple_resolution is not None:
            if paper.tuple_resolution_path is None:
                raise _error(
                    ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                    "source tuple-resolution sidecar disappeared",
                )
            paths[binding.tuple_resolution.run_path] = paper.tuple_resolution_path
            expected[binding.tuple_resolution.run_path] = binding.tuple_resolution.sha256
        if binding.verifier_gates is not None:
            if paper.verifier_gates_path is None:
                raise _error(
                    ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                    "source verifier-gate sidecar disappeared",
                )
            paths[binding.verifier_gates.run_path] = paper.verifier_gates_path
            expected[binding.verifier_gates.run_path] = binding.verifier_gates.sha256
        if set(paths) != set(expected) or any(
            _run_relative(run_root, path) != relative or sha256_file(path) != expected[relative]
            for relative, path in paths.items()
        ):
            raise _error(
                ReviewedExportErrorCode.RUN_BINDING_MISMATCH,
                "source paper artifact changed",
            )
    return originals


def _non_origin_failure_codes(
    candidate: CandidateObservation, *, min_confidence: float | None = None
) -> list[str]:
    codes: list[str] = []
    if candidate.claim_type is not ClaimType.PRIMARY_RESULT:
        codes.append("NOT_PRIMARY_RESULT")
    if candidate.text_support is not TextSupportStatus.SUPPORTED:
        codes.append("TEXT_UNSUPPORTED")
    if candidate.referential_status is ReferentialStatus.WRONG_SCOPE:
        codes.append("WRONG_SCOPE")
    elif candidate.referential_status is not ReferentialStatus.RESOLVED:
        codes.append("REFERENTIAL_UNRESOLVED")
    if candidate.export_status is not ExportStatus.ELIGIBLE:
        reason = candidate.export_reason or ""
        if reason == "field_provenance=conflict":
            codes.append("FIELD_PROVENANCE_CONFLICT")
        elif "field_provenance" in reason:
            codes.append("FIELD_PROVENANCE_UNSUPPORTED")
        if "extraction_confidence" in reason or (
            min_confidence is not None and candidate.extraction_confidence < min_confidence
        ):
            codes.append("LOW_CONFIDENCE")
        if "physical-cell conflict" in reason:
            codes.append("PHYSICAL_CELL_CONFLICT")
        if "ambiguous evidence" in reason or "semantic safety" in reason:
            codes.append("SEMANTIC_CONFLICT")
    return list(dict.fromkeys(codes or ["NON_ORIGIN_GATE_FAILED"]))


def _unit_resolution_change_is_attested(
    candidate: CandidateObservation,
    decision: ReviewedExportDecision,
) -> bool:
    """Require exact metric/value review spans to re-derive a replay-added unit."""

    attestations = {item.field: item for item in decision.field_attestations}
    required = {CandidateField.METRIC, CandidateField.VALUE, CandidateField.UNIT}
    if not required <= set(attestations):
        return False
    spans_by_id = {
        anchor.span_id: anchor for anchor in decision.result_evidence if anchor.span_id is not None
    }
    span_ids = {span_id for field in required for span_id in attestations[field].span_ids}
    if not span_ids or not span_ids <= set(spans_by_id):
        return False
    attested = _attested_candidate(
        candidate,
        [spans_by_id[span_id] for span_id in sorted(span_ids)],
    )
    derived = _attested_derivations(attested)
    status, _, _ = direct_quote_field_binding(
        attested,
        CandidateField.UNIT,
        derived_reasons=derived,
    )
    return CandidateField.UNIT in derived and status is FieldBindingStatus.BOUND


def _field_provenance_is_only_blocker(
    candidate: CandidateObservation,
    decision: ReviewedExportDecision,
    *,
    min_confidence: float,
) -> bool:
    """Return whether exact human field attestation may cure this candidate."""

    human_curable_reasons = {
        "dataset_scope_uses_table_context",
        "field_identity_not_supported_by_evidence_quotes",
        "field_not_supported_by_evidence_quotes",
        "field_partially_supported_by_evidence_quotes",
        "load_bearing_tuple_association_not_supported_by_evidence_quotes",
        "setting_uses_table_context",
    }
    populated = candidate.populated_fields()
    field_support_is_curable = True
    for binding in candidate.field_provenance:
        if binding.field not in populated or binding.status is FieldBindingStatus.BOUND:
            continue
        ordinary_cure = (
            binding.status in {FieldBindingStatus.AMBIGUOUS, FieldBindingStatus.UNSUPPORTED}
            and binding.reason in human_curable_reasons
        )
        replay_unit_cure = (
            binding.field is CandidateField.UNIT
            and binding.status is FieldBindingStatus.UNSUPPORTED
            and binding.reason == "reference_resolution_changed_unbound_unit"
            and _unit_resolution_change_is_attested(candidate, decision)
        )
        if not (ordinary_cure or replay_unit_cure):
            field_support_is_curable = False
            break
    return (
        candidate.claim_type is ClaimType.PRIMARY_RESULT
        and candidate.text_support is TextSupportStatus.SUPPORTED
        and candidate.referential_status is ReferentialStatus.RESOLVED
        and candidate.schema_version == "candidate-observation/0.3"
        and bool(candidate.field_provenance)
        and all(
            binding.status is not FieldBindingStatus.CONFLICT
            for binding in candidate.field_provenance
        )
        and field_support_is_curable
        and candidate.extraction_confidence >= min_confidence
        and candidate.export_status is ExportStatus.NEEDS_REVIEW
        and candidate.export_reason == "field_provenance=ambiguous_or_unsupported"
        and _deterministic_value_support_issue(candidate) is None
    )


def _source_matches_attested_spans(
    source: FieldSourceRef, spans: list[ReviewEvidenceAnchor]
) -> bool:
    """Keep a bound structural source only when an attested span shares its location."""

    return any(
        source.source_id == span.source_id
        and source.page == span.page
        and (
            source.region_id is None
            or (span.region_id is not None and source.region_id == span.region_id)
        )
        for span in spans
    )


def _reviewed_field_provenance(
    candidate: CandidateObservation, decision: ReviewedExportDecision
) -> list[CandidateFieldProvenance]:
    """Materialize validated human span bindings as quote-free field provenance."""

    expected_hashes = candidate.field_value_sha256s()
    original = {binding.field: binding for binding in candidate.field_provenance}
    attestations = {item.field: item for item in decision.field_attestations}
    if any(
        attestation.value_sha256 != expected_hashes[field]
        for field, attestation in attestations.items()
    ):
        raise _error(
            ReviewedExportErrorCode.TUPLE_MISMATCH,
            "reviewed field hash differs from the recomputed candidate tuple",
        )
    spans_by_id = {
        anchor.span_id: anchor for anchor in decision.result_evidence if anchor.span_id is not None
    }
    populated = decision.reviewed_tuple.populated_fields()
    reviewed: list[CandidateFieldProvenance] = []
    for field in CandidateField:
        if field not in populated:
            binding = original.get(field)
            reviewed.append(
                binding.model_copy(deep=True)
                if binding is not None and binding.value_sha256 == expected_hashes[field]
                else CandidateFieldProvenance(
                    field=field,
                    value_sha256=expected_hashes[field],
                    status=FieldBindingStatus.UNSUPPORTED,
                    reason="field_not_present",
                )
            )
            continue

        attestation = attestations[field]
        spans = [spans_by_id[span_id] for span_id in attestation.span_ids]
        preserved = []
        binding = original.get(field)
        if binding is not None and binding.status is FieldBindingStatus.BOUND:
            preserved = [
                source.model_copy(deep=True)
                for source in binding.sources
                if source.kind not in {FieldSourceKind.EVIDENCE_QUOTE, FieldSourceKind.REVIEW_SPAN}
                and _source_matches_attested_spans(source, spans)
            ]
        human_sources = [
            FieldSourceRef(
                kind=FieldSourceKind.REVIEW_SPAN,
                source_id=span.source_id,
                page=span.page,
                region_id=span.region_id,
                quote_sha256=span.excerpt_sha256,
                review_span_id=span.span_id,
            )
            for span in spans
        ]
        sources_by_payload = {
            json.dumps(
                source.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ): source
            for source in [*preserved, *human_sources]
        }
        reviewed.append(
            CandidateFieldProvenance(
                field=field,
                value_sha256=expected_hashes[field],
                status=FieldBindingStatus.BOUND,
                sources=[sources_by_payload[key] for key in sorted(sources_by_payload)],
                reason="human_reviewed_field_attestation",
            )
        )
    return reviewed


def _apply_reviewed_field_attestation(
    candidate: CandidateObservation, decision: ReviewedExportDecision
) -> None:
    candidate.field_provenance = _reviewed_field_provenance(candidate, decision)
    candidate.export_status = ExportStatus.ELIGIBLE
    candidate.export_reason = "human_reviewed_field_attestation"


def _initial_outcome(
    item: ReviewItem,
    decision: ReviewedExportDecision,
    state: ExportOutcomeState,
    *failure_codes: str,
) -> ExportOutcome:
    return ExportOutcome(
        item_id=item.item_id,
        decision_id=decision.decision_id,
        paper_id=item.fingerprint.paper_id,
        observation_id=item.fingerprint.observation_id,
        candidate_payload_sha256=item.fingerprint.candidate_payload_sha256,
        state=state,
        failure_codes=list(failure_codes),
    )


def _flatten_review_evidence(
    details: dict[str, str], prefix: str, anchors: list[ReviewEvidenceAnchor]
) -> None:
    details[f"{prefix}_count"] = str(len(anchors))
    for index, anchor in enumerate(anchors, start=1):
        key = f"{prefix}_{index}"
        values = {
            "source_id": anchor.source_id,
            "source_sha256": anchor.source_sha256,
            "source_manifest_sha256": anchor.source_manifest_sha256,
            "layout_sha256": anchor.layout_sha256,
            "parser": anchor.parser,
            "parser_version": anchor.parser_version,
            "page": str(anchor.page),
            "page_text_sha256": anchor.page_text_sha256,
            "span_id": str(anchor.span_id),
            "excerpt_sha256": anchor.excerpt_sha256,
            "char_start": str(anchor.char_start),
            "char_end": str(anchor.char_end),
            "start_line": str(anchor.start_line),
            "end_line": str(anchor.end_line),
            "kind": anchor.kind.value,
        }
        for name in ("region_id", "label", "row", "column"):
            value = getattr(anchor, name)
            if value is not None:
                values[name] = value
        if anchor.bounding_box is not None:
            values["bounding_box"] = ",".join(str(value) for value in anchor.bounding_box)
        details.update({f"{key}_{name}": value for name, value in values.items()})


def _inject_review_provenance(
    *,
    result: dict[str, Any],
    item: ReviewItem,
    decision: ReviewedExportDecision,
    review: ValidatedReview,
) -> None:
    assert review.lock_sha256 is not None
    assert decision.authority is not None
    score_details = result.setdefault("score_details", {})
    details = score_details.setdefault("details", {})
    if not isinstance(details, dict):
        raise _error(
            ReviewedExportErrorCode.EEE_SCHEMA_INVALID,
            "EEE score details are not an object",
        )
    details.update(
        {
            "candidate_fingerprint_sha256": item.fingerprint.candidate_payload_sha256,
            "review_manifest_sha256": review.manifest_sha256,
            "review_lock_sha256": review.lock_sha256,
            "review_decision_sha256": _decision_sha256(decision),
            "origin_decision": "paper_produced",
            "decision_authority": decision.authority.mode.value,
            "decision_protocol_id": decision.authority.protocol_id,
            "decision_protocol_sha256": decision.authority.protocol_sha256,
        }
    )
    _flatten_review_evidence(details, "reviewed_result_evidence", decision.result_evidence)
    _flatten_review_evidence(details, "reviewed_origin_evidence", decision.origin_evidence)


def _quote_free_provenance(
    *,
    item: ReviewItem,
    decision: ReviewedExportDecision,
    review: ValidatedReview,
    export_provenance: ExportCompositionProvenance,
    field_provenance: list[CandidateFieldProvenance],
    evaluation_id: str,
    eee_path: str,
) -> ReviewedExportProvenance:
    assert review.lock_sha256 is not None
    assert decision.authority is not None
    return ReviewedExportProvenance(
        paper_id=item.fingerprint.paper_id,
        observation_id=item.fingerprint.observation_id,
        evaluation_id=evaluation_id,
        evaluation_result_id=item.fingerprint.observation_id,
        eee_path=eee_path,
        candidate_payload_sha256=item.fingerprint.candidate_payload_sha256,
        review_manifest_sha256=review.manifest_sha256,
        review_lock_sha256=review.lock_sha256,
        review_decision_sha256=_decision_sha256(decision),
        export_provenance_mode=export_provenance.mode,
        export_composition_sha256=export_provenance.sha256,
        source_tuple_sidecar_sha256=item.source_tuple_sidecar_sha256,
        source_tuple_gate_sha256=item.source_tuple_gate_sha256,
        source_verifier_sidecar_sha256=item.source_verifier_sidecar_sha256,
        source_verifier_gate_sha256=item.source_verifier_gate_sha256,
        source_verifier_gate_passed=item.source_verifier_gate_passed,
        decision_authority=decision.authority.mode,
        decision_protocol_id=decision.authority.protocol_id,
        decision_protocol_sha256=decision.authority.protocol_sha256,
        result_evidence=[
            QuoteFreeEvidence.from_private(anchor) for anchor in decision.result_evidence
        ],
        origin_evidence=[
            QuoteFreeEvidence.from_private(anchor) for anchor in decision.origin_evidence
        ],
        proposal_ids=decision.reviewed_tuple.proposal_ids,
        candidate_occurrence_ids=decision.reviewed_tuple.candidate_occurrence_ids,
        field_provenance=field_provenance,
        field_attestations=decision.field_attestations,
    )


def _restore_source_observation_ids(
    validated: list[CandidateObservation], source: tuple[CandidateObservation, ...]
) -> None:
    """Keep sealed ledger identities while retaining recomputed tuple semantics.

    Non-origin replay intentionally resolves registries again, and those canonical
    projections can change ``stable_id()``. Review items, source tuple/verifier gates,
    and immutable ledger lines are all keyed by the sealed source observation ID, so a
    reviewed projection must retain that identity rather than silently minting a new
    result during replay.
    """

    if len(validated) != len(source):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "revalidated candidates no longer align with the sealed observation ledger",
        )
    for candidate, original in zip(validated, source, strict=True):
        if original.observation_id is None:
            raise _error(
                ReviewedExportErrorCode.CANDIDATE_BINDING_INVALID,
                "sealed source candidate lacks an immutable observation identity",
            )
        candidate.observation_id = original.observation_id


def _approved_duplicate_failure_codes(
    approved: list[tuple[ReviewItem, ReviewedExportDecision, CandidateObservation]],
) -> dict[str, str]:
    """Classify duplicate source cells and immutable candidate-ID collisions."""

    structural_groups: dict[
        str, list[tuple[ReviewItem, ReviewedExportDecision, CandidateObservation]]
    ] = defaultdict(list)
    observation_groups: dict[
        str, list[tuple[ReviewItem, ReviewedExportDecision, CandidateObservation]]
    ] = defaultdict(list)
    coarse_location_groups: dict[
        str, dict[str, tuple[ReviewItem, ReviewedExportDecision, CandidateObservation]]
    ] = defaultdict(dict)
    claimed_cell_groups: dict[
        str, dict[str, tuple[ReviewItem, ReviewedExportDecision, CandidateObservation]]
    ] = defaultdict(dict)
    for triplet in approved:
        structural_groups[triplet[0].fingerprint.structural_identity_sha256].append(triplet)
        observation_groups[triplet[0].fingerprint.observation_id].append(triplet)
        item, _, candidate = triplet
        for anchor in candidate.evidence:
            if (
                anchor.row is not None
                or anchor.column is not None
                or anchor.bounding_box is not None
            ):
                coarse_key = json.dumps(
                    {
                        "paper_id": candidate.paper_id,
                        "source_id": anchor.source_id,
                        "page": anchor.page,
                        "kind": anchor.kind.value,
                        "region_id": anchor.region_id,
                        "row": anchor.row,
                        "column": anchor.column,
                        "bounding_box": anchor.bounding_box,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                coarse_location_groups[coarse_key][item.item_id] = triplet
            if all(
                (
                    anchor.planned_row_id,
                    anchor.cell_id,
                    anchor.numeric_token_id,
                )
            ):
                claimed_key = json.dumps(
                    {
                        "paper_id": candidate.paper_id,
                        "source_id": anchor.source_id,
                        "page": anchor.page,
                        "region_id": anchor.region_id,
                        "planned_row_id": anchor.planned_row_id,
                        "cell_id": anchor.cell_id,
                        "numeric_token_id": anchor.numeric_token_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                claimed_cell_groups[claimed_key][item.item_id] = triplet
    failures: dict[str, str] = {}
    for groups, code in (
        (structural_groups, "PHYSICAL_CELL_DUPLICATE"),
        (observation_groups, "CANDIDATE_ID_COLLISION"),
    ):
        for triplets in groups.values():
            if len(triplets) < 2:
                continue
            for item, _, _ in triplets:
                failures[item.item_id] = code
    for groups in (coarse_location_groups, claimed_cell_groups):
        for by_item in groups.values():
            if len(by_item) < 2:
                continue
            for item, _, _ in by_item.values():
                failures[item.item_id] = "PHYSICAL_CELL_DUPLICATE"
    return failures


def _reviewed_composition_provenance(
    *,
    paper: _ReviewPaper,
    candidates: list[CandidateObservation],
    review: ValidatedReview,
) -> ExportCompositionProvenance:
    """Rebuild the exact composition authorization used by compose and verify."""

    source_tuple = paper.tuple_resolution
    source_verifier = paper.verifier_gates
    if source_tuple is None:
        return legacy_export_provenance(
            candidates,
            reason="human_review_of_legacy_source_run",
            review_manifest_sha256=review.manifest_sha256,
        )
    return tuple_audited_review_export_provenance(
        candidates,
        tuple_sidecar_sha256=source_tuple["sidecar_sha256"],
        tuple_gates={
            str(candidate.observation_id): (
                source_tuple["candidate_gates"][str(candidate.observation_id)],
                source_tuple["passed"][str(candidate.observation_id)],
            )
            for candidate in candidates
        },
        review_manifest_sha256=review.manifest_sha256,
        verifier_sidecar_sha256=(
            source_verifier["sidecar_sha256"] if source_verifier is not None else None
        ),
        verifier_gates=(
            {
                str(candidate.observation_id): (
                    source_verifier["candidate_gates"].get(str(candidate.observation_id)),
                    source_verifier["passed"][str(candidate.observation_id)],
                )
                for candidate in candidates
            }
            if source_verifier is not None
            else None
        ),
    )


def _reviewed_candidate_projection(
    *,
    item: ReviewItem,
    decision: ReviewedExportDecision,
    candidate: CandidateObservation,
    original: CandidateObservation,
    min_confidence: float,
) -> CandidateObservation | None:
    """Return the exact candidate projection eligible for reviewed composition."""

    if candidate.observation_id != item.fingerprint.observation_id:
        return None
    if (
        decision.status is not DecisionStatus.COMPLETED
        or decision.tuple_decision is not TupleDecision.CONFIRMED
        or decision.origin_decision is not OriginDecision.PAPER_PRODUCED
        or decision.authority is None
    ):
        return None
    candidate = candidate.model_copy(deep=True)
    if _deterministic_value_support_issue(candidate) in _NON_CURABLE_VALUE_SUPPORT_ISSUES:
        return None
    needs_field_cure = candidate.export_status is not ExportStatus.ELIGIBLE
    if (
        needs_field_cure
        and not _field_provenance_is_only_blocker(
            candidate,
            decision,
            min_confidence=min_confidence,
        )
    ) or _tuple_sha256(ReviewedTuple.from_candidate(candidate)) != _tuple_sha256(
        decision.reviewed_tuple
    ):
        return None
    if (
        original.attribution is not None
        and original.attribution.state is AttributionState.EXTERNALLY_SOURCED
        and decision.authority.mode.value != "adjudicated"
    ):
        return None
    _apply_reviewed_field_attestation(candidate, decision)
    candidate.attribution = AttributionVerdict(
        state=AttributionState.PAPER_PRODUCED,
        rule_id=f"reviewed_origin:{decision.decision_id}",
    )
    candidate.export_status = ExportStatus.ELIGIBLE
    return candidate


def _payload_inventory(root: Path) -> list[ArtifactRef]:
    excluded = {DERIVED_MANIFEST_NAME, DERIVED_SHA256SUMS_NAME, DERIVED_VERIFICATION_NAME}
    artifacts: list[ArtifactRef] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "derived tree contains a symbolic link",
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "derived tree contains a non-regular entry",
            )
        if relative in excluded:
            continue
        artifacts.append(
            ArtifactRef(
                path=relative,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return artifacts


def _sha256s_bytes(files: list[ArtifactRef]) -> bytes:
    return "".join(f"{artifact.sha256}  {artifact.path}\n" for artifact in files).encode("utf-8")


def _derived_counts(
    *,
    outcomes: tuple[ExportOutcome, ...] | list[ExportOutcome],
    eee_record_count: int,
    eee_observation_count: int,
) -> dict[str, int]:
    state_counts = Counter(outcome.state.value for outcome in outcomes)
    pending = sum("DECISION_PENDING" in outcome.failure_codes for outcome in outcomes)
    counts = {
        "review_items": len(outcomes),
        "decisions_completed": len(outcomes) - pending,
        "decisions_pending": pending,
        "outcomes_exported": state_counts[ExportOutcomeState.EXPORTED.value],
        "outcomes_withheld": state_counts[ExportOutcomeState.WITHHELD.value],
        "outcomes_failed": state_counts[ExportOutcomeState.FAILED.value],
        "eee_records": eee_record_count,
        "eee_observations": eee_observation_count,
    }
    assert tuple(counts) == DERIVED_COUNT_KEYS
    return counts


def _expected_verification(
    *, manifest_path: Path, manifest: DerivedRunManifest, root: Path
) -> DerivedVerification:
    eee_paths = [artifact.path for artifact in manifest.payload_files if "/eee/" in artifact.path]
    outcomes = _jsonl_records(
        _confined_file(root, DERIVED_OUTCOMES_NAME), ExportOutcome, "derived outcomes"
    )
    try:
        provenance = _jsonl_records(
            _confined_file(root, DERIVED_PROVENANCE_NAME),
            ReviewedExportProvenance,
            "derived provenance",
        )
    except ReviewedExportError as exc:
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived reviewed provenance is internally inconsistent",
        ) from exc
    exported = [item for item in outcomes if item.state is ExportOutcomeState.EXPORTED]
    return DerivedVerification(
        derived_run_sha256=sha256_file(manifest_path),
        sha256s_sha256=manifest.sha256s_sha256,
        payload_tree_sha256=manifest.payload_tree_sha256,
        payload_file_count=len(manifest.payload_files),
        eee_record_count=len(eee_paths),
        exported_observation_count=len(exported),
        quote_free_dual_provenance_count=len(provenance),
    )


def _assert_unique_eee_output_path(
    records: list[tuple[str, dict[str, Any]]], candidate_path: str
) -> None:
    """Fail closed before two substantive records can overwrite one output path."""

    if any(path == candidate_path for path, _ in records):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            f"reviewed composition produced a duplicate EEE output path: {candidate_path}",
        )


def compose_reviewed_eee(
    *,
    run_root: Path,
    decisions_path: Path,
    output_root: Path,
    schema_path: Path = DEFAULT_EEE_SCHEMA_PATH,
    schema_sha256: str = EEE_SCHEMA_SHA256,
) -> DerivedRunManifest:
    """Recompute gates and atomically compose a separate quote-free EEE tree."""

    review_root = decisions_path.parent
    if decisions_path.name != REVIEW_DECISIONS_NAME:
        raise _error(
            ReviewedExportErrorCode.DECISION_SET_INVALID,
            "decisions must be the review root decisions.jsonl",
        )
    _require_disjoint_output(output_root, run_root, review_root)
    review = _load_review(review_root, require_lock=True)
    assert review.lock is not None and review.lock_sha256 is not None
    originals = _bind_original_run(run_root, review)
    try:
        schema, authority = load_schema(schema_path, schema_sha256)
    except (OSError, ValueError) as exc:
        raise _error(
            ReviewedExportErrorCode.EEE_SCHEMA_INVALID, "EEE schema binding failed"
        ) from exc
    if any(
        paper.schema_version != authority.version or paper.schema_sha256 != authority.sha256
        for paper in review.manifest.papers
    ):
        raise _error(
            ReviewedExportErrorCode.EEE_SCHEMA_INVALID,
            "reviewed run and requested EEE schema bindings differ",
        )

    items = {item.item_id: item for item in review.items}
    validated_by_paper: dict[str, list[CandidateObservation]] = {}
    for paper_id, paper in originals.items():
        validated = validate_non_origin_candidates(
            paper.candidates,
            {paper.layout.source_id: paper.layout},
            min_confidence=review.manifest.min_confidence,
        )
        _restore_source_observation_ids(validated, paper.candidates)
        validated_by_paper[paper_id] = validated

    outcomes: dict[str, ExportOutcome] = {}
    approved: list[tuple[ReviewItem, ReviewedExportDecision, CandidateObservation]] = []
    for decision in review.decisions:
        item = items[decision.item_id]
        if decision.status is DecisionStatus.PENDING:
            outcomes[item.item_id] = _initial_outcome(
                item, decision, ExportOutcomeState.WITHHELD, "DECISION_PENDING"
            )
            continue
        if decision.tuple_decision is TupleDecision.REJECTED:
            outcomes[item.item_id] = _initial_outcome(
                item, decision, ExportOutcomeState.WITHHELD, "TUPLE_REJECTED"
            )
            continue
        if decision.tuple_decision is TupleDecision.UNRESOLVED:
            outcomes[item.item_id] = _initial_outcome(
                item, decision, ExportOutcomeState.WITHHELD, "TUPLE_UNRESOLVED"
            )
            continue
        if decision.origin_decision is OriginDecision.EXTERNALLY_SOURCED:
            outcomes[item.item_id] = _initial_outcome(
                item, decision, ExportOutcomeState.WITHHELD, "ORIGIN_EXTERNAL"
            )
            continue
        if decision.origin_decision is OriginDecision.UNRESOLVED:
            outcomes[item.item_id] = _initial_outcome(
                item, decision, ExportOutcomeState.WITHHELD, "ORIGIN_UNRESOLVED"
            )
            continue
        line_index = item.fingerprint.observation_ledger_line - 1
        candidate = validated_by_paper[item.fingerprint.paper_id][line_index]
        if _deterministic_value_support_issue(candidate) in _NON_CURABLE_VALUE_SUPPORT_ISSUES:
            outcomes[item.item_id] = _initial_outcome(
                item,
                decision,
                ExportOutcomeState.FAILED,
                "VALUE_EVIDENCE_CONTRADICTION",
            )
            continue
        needs_field_cure = candidate.export_status is not ExportStatus.ELIGIBLE
        if needs_field_cure and not (
            _field_provenance_is_only_blocker(
                candidate,
                decision,
                min_confidence=review.manifest.min_confidence,
            )
        ):
            outcomes[item.item_id] = _initial_outcome(
                item,
                decision,
                ExportOutcomeState.FAILED,
                *_non_origin_failure_codes(
                    candidate, min_confidence=review.manifest.min_confidence
                ),
            )
            continue
        if _tuple_sha256(ReviewedTuple.from_candidate(candidate)) != _tuple_sha256(
            decision.reviewed_tuple
        ):
            outcomes[item.item_id] = _initial_outcome(
                item, decision, ExportOutcomeState.FAILED, "TUPLE_RECOMPUTE_MISMATCH"
            )
            continue
        _apply_reviewed_field_attestation(candidate, decision)
        original = originals[item.fingerprint.paper_id].candidates[line_index]
        assert decision.authority is not None
        if (
            original.attribution is not None
            and original.attribution.state is AttributionState.EXTERNALLY_SOURCED
            and decision.authority.mode.value != "adjudicated"
        ):
            outcomes[item.item_id] = _initial_outcome(
                item,
                decision,
                ExportOutcomeState.FAILED,
                "DETERMINISTIC_EXTERNAL_CONFLICT",
            )
            continue
        candidate.attribution = AttributionVerdict(
            state=AttributionState.PAPER_PRODUCED,
            rule_id=f"reviewed_origin:{decision.decision_id}",
        )
        candidate.export_status = ExportStatus.ELIGIBLE
        approved.append((item, decision, candidate))

    duplicate_failures = _approved_duplicate_failure_codes(approved)
    for item, decision, _ in approved:
        if item.item_id in duplicate_failures:
            outcomes[item.item_id] = _initial_outcome(
                item,
                decision,
                ExportOutcomeState.FAILED,
                duplicate_failures[item.item_id],
            )
    approved = [triplet for triplet in approved if triplet[0].item_id not in duplicate_failures]

    records_to_write: list[tuple[str, dict[str, Any]]] = []
    provenance: list[ReviewedExportProvenance] = []
    composition_provenances: list[ExportCompositionProvenance] = []
    for paper_id in sorted(originals):
        triplets = [triplet for triplet in approved if triplet[0].fingerprint.paper_id == paper_id]
        if not triplets:
            continue
        decision_by_result = {
            item.fingerprint.observation_id: (item, decision) for item, decision, _ in triplets
        }
        field_provenance_by_item = {
            item.item_id: candidate.field_provenance for item, _, candidate in triplets
        }
        paper_candidates = [candidate for _, _, candidate in triplets]
        export_provenance = _reviewed_composition_provenance(
            paper=originals[paper_id],
            candidates=paper_candidates,
            review=review,
        )
        records = compose_eee_records(
            manifest=originals[paper_id].source_manifest,
            candidates=paper_candidates,
            schema_version=authority.version,
            provenance=export_provenance,
        )
        observed_results: set[str] = set()
        composition_used = False
        for record in records:
            evaluation_id = record.get("evaluation_id")
            if not isinstance(evaluation_id, str):
                continue
            suffix = evaluation_id.rsplit("/", 1)[-1]
            eee_path = f"{paper_id}/eee/{suffix}.json"
            record_items: list[tuple[ReviewItem, ReviewedExportDecision]] = []
            for result in record.get("evaluation_results", []):
                result_id = result.get("evaluation_result_id")
                if not isinstance(result_id, str) or result_id not in decision_by_result:
                    continue
                item, decision = decision_by_result[result_id]
                observed_results.add(result_id)
                _inject_review_provenance(
                    result=result,
                    item=item,
                    decision=decision,
                    review=review,
                )
                record_items.append((item, decision))
            issues = validate_eee_record(record, schema)
            if issues or not record_items:
                for item, decision in record_items:
                    outcomes[item.item_id] = _initial_outcome(
                        item, decision, ExportOutcomeState.FAILED, "EEE_SCHEMA_INVALID"
                    )
                continue
            _assert_unique_eee_output_path(records_to_write, eee_path)
            records_to_write.append((eee_path, record))
            composition_used = True
            for item, decision in record_items:
                outcomes[item.item_id] = ExportOutcome(
                    item_id=item.item_id,
                    decision_id=decision.decision_id,
                    paper_id=item.fingerprint.paper_id,
                    observation_id=item.fingerprint.observation_id,
                    candidate_payload_sha256=item.fingerprint.candidate_payload_sha256,
                    state=ExportOutcomeState.EXPORTED,
                    evaluation_id=evaluation_id,
                    evaluation_result_id=item.fingerprint.observation_id,
                    eee_path=eee_path,
                )
                provenance.append(
                    _quote_free_provenance(
                        item=item,
                        decision=decision,
                        review=review,
                        export_provenance=export_provenance,
                        field_provenance=field_provenance_by_item[item.item_id],
                        evaluation_id=evaluation_id,
                        eee_path=eee_path,
                    )
                )
        if composition_used:
            composition_provenances.append(export_provenance)
        for item, decision, _ in triplets:
            if (
                item.fingerprint.observation_id not in observed_results
                and item.item_id not in outcomes
            ):
                outcomes[item.item_id] = _initial_outcome(
                    item, decision, ExportOutcomeState.FAILED, "COMPOSITION_OMISSION"
                )

    if set(outcomes) != set(items):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "not every review item received exactly one export outcome",
        )

    result: dict[str, DerivedRunManifest] = {}

    def build(tree: Path) -> None:
        for relative, record in sorted(records_to_write):
            write_json(tree / relative, record)
        ordered_outcomes = [outcomes[item.item_id] for item in review.items]
        ordered_provenance = sorted(
            provenance,
            key=lambda item: (item.paper_id, item.evaluation_id, item.evaluation_result_id),
        )
        write_jsonl(tree / DERIVED_OUTCOMES_NAME, ordered_outcomes)
        write_jsonl(tree / DERIVED_PROVENANCE_NAME, ordered_provenance)
        payload_files = _payload_inventory(tree)
        payload_tree_sha = sha256_bytes(
            canonical_json_bytes([model_payload(artifact) for artifact in payload_files])
        )
        sha256s = _sha256s_bytes(payload_files)
        atomic_write_bytes(tree / DERIVED_SHA256SUMS_NAME, sha256s)
        counts = _derived_counts(
            outcomes=ordered_outcomes,
            eee_record_count=sum("/eee/" in artifact.path for artifact in payload_files),
            eee_observation_count=len(ordered_provenance),
        )
        manifest = DerivedRunManifest(
            source_run_name=review.manifest.run_seal.source_run_name,
            source_run_seal_sha256=review.manifest.run_seal.seal_sha256,
            source_run_tree_sha256=review.manifest.run_seal.tree_sha256,
            review_manifest_sha256=review.manifest_sha256,
            review_lock_sha256=review.lock_sha256,
            decisions_sha256=review.lock.decisions_sha256,
            eee_schema_version=authority.version,
            eee_schema_sha256=authority.sha256,
            export_provenance_modes=sorted(
                {item.mode for item in composition_provenances}, key=lambda item: item.value
            ),
            tuple_sidecar_sha256s=sorted(
                {
                    item.tuple_sidecar_sha256
                    for item in composition_provenances
                    if item.tuple_sidecar_sha256 is not None
                }
            ),
            verifier_sidecar_sha256s=sorted(
                {
                    item.verifier_sidecar_sha256
                    for item in composition_provenances
                    if item.verifier_sidecar_sha256 is not None
                }
            ),
            locked_at=review.lock.locked_at,
            counts=counts,
            payload_files=payload_files,
            payload_tree_sha256=payload_tree_sha,
            sha256s_sha256=sha256_bytes(sha256s),
        )
        write_json(tree / DERIVED_MANIFEST_NAME, manifest)
        verification = _expected_verification(
            manifest_path=tree / DERIVED_MANIFEST_NAME,
            manifest=manifest,
            root=tree,
        )
        write_json(tree / DERIVED_VERIFICATION_NAME, verification)
        result["manifest"] = manifest
        verify_derived_run(
            tree,
            schema_path=schema_path,
            schema_sha256=schema_sha256,
            run_root=run_root,
            review_root=review_root,
        )

    _atomic_tree(output_root, build)
    _bind_original_run(run_root, review)
    return result["manifest"]


def _privacy_walk(value: Any) -> None:
    if isinstance(value, dict):
        forbidden_keys = {"exact_excerpt", "reviewer_ids", "adjudicator_id", "matched_text"}
        if forbidden_keys.intersection(value):
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "derived JSON contains private text or reviewer identity fields",
            )
        for item in value.values():
            _privacy_walk(item)
    elif isinstance(value, list):
        for item in value:
            _privacy_walk(item)
    elif isinstance(value, str):
        if value.startswith("/") or value.startswith("file://") or "/Users/" in value:
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "derived JSON contains an absolute local path",
            )


_REVIEW_DETAIL_KEYS = frozenset(
    {
        "candidate_fingerprint_sha256",
        "review_manifest_sha256",
        "review_lock_sha256",
        "review_decision_sha256",
        "origin_decision",
        "decision_authority",
        "decision_protocol_id",
        "decision_protocol_sha256",
    }
)
_PRIVATE_EVIDENCE_BINDING_NAMES = frozenset(
    {"source_manifest_sha256", "layout_sha256", "parser", "parser_version"}
)


def _quote_free_evidence_details(prefix: str, anchors: list[QuoteFreeEvidence]) -> dict[str, str]:
    details = {f"{prefix}_count": str(len(anchors))}
    for index, anchor in enumerate(anchors, start=1):
        expected_span_id = review_evidence_span_id(
            source_id=anchor.source_id,
            page=anchor.page,
            page_text_sha256=anchor.page_text_sha256,
            char_start=anchor.char_start,
            char_end=anchor.char_end,
            excerpt_sha256=anchor.excerpt_sha256,
        )
        if anchor.span_id != expected_span_id:
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "quote-free evidence span id does not match its exact occurrence",
            )
        key = f"{prefix}_{index}"
        values = {
            "source_id": anchor.source_id,
            "source_sha256": anchor.source_sha256,
            "page": str(anchor.page),
            "page_text_sha256": anchor.page_text_sha256,
            "span_id": str(anchor.span_id),
            "excerpt_sha256": anchor.excerpt_sha256,
            "char_start": str(anchor.char_start),
            "char_end": str(anchor.char_end),
            "start_line": str(anchor.start_line),
            "end_line": str(anchor.end_line),
            "kind": anchor.kind.value,
        }
        for name in ("region_id", "label", "row", "column"):
            value = getattr(anchor, name)
            if value is not None:
                values[name] = value
        if anchor.bounding_box is not None:
            values["bounding_box"] = ",".join(str(value) for value in anchor.bounding_box)
        details.update({f"{key}_{name}": value for name, value in values.items()})
    return details


def _review_detail_view(details: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in details.items()
        if key in _REVIEW_DETAIL_KEYS
        or key.startswith("reviewed_result_evidence_")
        or key.startswith("reviewed_origin_evidence_")
    }


def _validate_public_result_binding(
    *,
    details: dict[str, Any],
    outcome: ExportOutcome,
    provenance: ReviewedExportProvenance,
) -> None:
    expected = {
        "candidate_fingerprint_sha256": provenance.candidate_payload_sha256,
        "review_manifest_sha256": provenance.review_manifest_sha256,
        "review_lock_sha256": provenance.review_lock_sha256,
        "review_decision_sha256": provenance.review_decision_sha256,
        "origin_decision": provenance.origin_decision,
        "decision_authority": provenance.decision_authority.value,
        "decision_protocol_id": provenance.decision_protocol_id,
        "decision_protocol_sha256": provenance.decision_protocol_sha256,
    }
    expected.update(
        _quote_free_evidence_details("reviewed_result_evidence", provenance.result_evidence)
    )
    expected.update(
        _quote_free_evidence_details("reviewed_origin_evidence", provenance.origin_evidence)
    )
    allowed_keys = set(expected)
    for prefix, anchors in (
        ("reviewed_result_evidence", provenance.result_evidence),
        ("reviewed_origin_evidence", provenance.origin_evidence),
    ):
        for index in range(1, len(anchors) + 1):
            allowed_keys.update(
                f"{prefix}_{index}_{name}" for name in _PRIVATE_EVIDENCE_BINDING_NAMES
            )
    observed = _review_detail_view(details)
    if (
        set(observed) != allowed_keys
        or any(observed.get(key) != value for key, value in expected.items())
        or any(
            not isinstance(observed.get(key), str) or not observed[key]
            for key in allowed_keys - set(expected)
        )
        or outcome.paper_id != provenance.paper_id
        or outcome.observation_id != provenance.observation_id
        or outcome.candidate_payload_sha256 != provenance.candidate_payload_sha256
        or outcome.evaluation_result_id != provenance.evaluation_result_id
        or provenance.observation_id != provenance.evaluation_result_id
    ):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "EEE result, outcome, and reviewed provenance bindings differ",
        )


def _recompose_contextual_eee_records(
    *,
    approved: list[tuple[ReviewItem, ReviewedExportDecision, CandidateObservation]],
    originals: dict[str, _ReviewPaper],
    review: ValidatedReview,
    schema: dict[str, Any],
    schema_version: str,
) -> dict[str, dict[str, Any]]:
    """Recompose complete EEE payloads from sealed candidates and locked decisions."""

    expected: dict[str, dict[str, Any]] = {}
    for paper_id in sorted(originals):
        triplets = [triplet for triplet in approved if triplet[0].fingerprint.paper_id == paper_id]
        if not triplets:
            continue
        candidates = [candidate for _, _, candidate in triplets]
        composition = _reviewed_composition_provenance(
            paper=originals[paper_id],
            candidates=candidates,
            review=review,
        )
        decisions_by_result = {
            item.fingerprint.observation_id: (item, decision) for item, decision, _ in triplets
        }
        for record in compose_eee_records(
            manifest=originals[paper_id].source_manifest,
            candidates=candidates,
            schema_version=schema_version,
            provenance=composition,
        ):
            if validate_eee_record(record, schema):
                continue
            evaluation_id = record.get("evaluation_id")
            if not isinstance(evaluation_id, str):
                raise ValueError("recomposed EEE record lacks an evaluation identity")
            for result in record.get("evaluation_results", []):
                result_id = result.get("evaluation_result_id")
                if not isinstance(result_id, str) or result_id not in decisions_by_result:
                    raise ValueError("recomposed EEE result lacks a reviewed source identity")
                item, decision = decisions_by_result[result_id]
                _inject_review_provenance(
                    result=result,
                    item=item,
                    decision=decision,
                    review=review,
                )
            suffix = evaluation_id.rsplit("/", 1)[-1]
            path = f"{paper_id}/eee/{suffix}.json"
            if path in expected:
                raise ValueError("recomposition produced a duplicate EEE path")
            expected[path] = record
    return expected


def _validate_contextual_derived_bindings(
    *,
    derived_root: Path,
    manifest: DerivedRunManifest,
    schema: dict[str, Any],
    outcomes: tuple[ExportOutcome, ...],
    provenance_by_identity: dict[tuple[str, str, str], ReviewedExportProvenance],
    details_by_identity: dict[tuple[str, str, str], dict[str, Any]],
    run_root: Path,
    review_root: Path,
) -> None:
    review = _load_review(review_root, require_lock=True)
    assert review.lock is not None and review.lock_sha256 is not None
    originals = _bind_original_run(run_root, review)
    if (
        manifest.source_run_name != review.manifest.run_seal.source_run_name
        or manifest.source_run_seal_sha256 != review.manifest.run_seal.seal_sha256
        or manifest.source_run_tree_sha256 != review.manifest.run_seal.tree_sha256
        or manifest.review_manifest_sha256 != review.manifest_sha256
        or manifest.review_lock_sha256 != review.lock_sha256
        or manifest.decisions_sha256 != review.lock.decisions_sha256
        or manifest.locked_at != review.lock.locked_at
        or any(
            paper.schema_version != manifest.eee_schema_version
            or paper.schema_sha256 != manifest.eee_schema_sha256
            for paper in review.manifest.papers
        )
        or manifest.counts["review_items"] != len(review.items)
        or manifest.counts["decisions_completed"] != review.lock.completed_count
        or manifest.counts["decisions_pending"] != review.lock.pending_count
    ):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived manifest differs from the sealed run or locked review",
        )

    items = {item.item_id: item for item in review.items}
    decisions = {decision.item_id: decision for decision in review.decisions}
    if {outcome.item_id for outcome in outcomes} != set(items):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived outcomes do not exactly cover locked review items",
        )
    validated_by_paper: dict[str, list[CandidateObservation]] = {}
    for paper_id, paper in originals.items():
        validated = validate_non_origin_candidates(
            paper.candidates,
            {paper.layout.source_id: paper.layout},
            min_confidence=review.manifest.min_confidence,
        )
        _restore_source_observation_ids(validated, paper.candidates)
        validated_by_paper[paper_id] = validated

    potentially_approved: list[tuple[ReviewItem, ReviewedExportDecision, CandidateObservation]] = []
    for decision in review.decisions:
        item = items[decision.item_id]
        line_index = item.fingerprint.observation_ledger_line - 1
        original = originals[item.fingerprint.paper_id].candidates[line_index]
        candidate = _reviewed_candidate_projection(
            item=item,
            decision=decision,
            candidate=validated_by_paper[item.fingerprint.paper_id][line_index],
            original=original,
            min_confidence=review.manifest.min_confidence,
        )
        if candidate is not None:
            potentially_approved.append((item, decision, candidate))
    duplicate_failures = _approved_duplicate_failure_codes(potentially_approved)
    approved = [
        triplet for triplet in potentially_approved if triplet[0].item_id not in duplicate_failures
    ]
    approved_by_item = {item.item_id: candidate for item, _, candidate in approved}
    outcomes_by_item = {outcome.item_id: outcome for outcome in outcomes}
    if any(
        outcomes_by_item[item_id].state is not ExportOutcomeState.FAILED
        or outcomes_by_item[item_id].failure_codes != [code]
        for item_id, code in duplicate_failures.items()
    ):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived outcomes differ from replayed duplicate classification",
        )

    for outcome in outcomes:
        item = items[outcome.item_id]
        decision = decisions[outcome.item_id]
        if (
            outcome.decision_id != decision.decision_id
            or outcome.paper_id != item.fingerprint.paper_id
            or outcome.observation_id != item.fingerprint.observation_id
            or outcome.candidate_payload_sha256 != item.fingerprint.candidate_payload_sha256
        ):
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "derived outcome differs from its sealed candidate or locked decision",
            )
        if outcome.state is not ExportOutcomeState.EXPORTED:
            continue
        candidate = approved_by_item.get(item.item_id)
        if candidate is None:
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "exported candidate fails deterministic reviewed composition gates",
            )
        identity = (outcome.evaluation_id, outcome.evaluation_result_id, outcome.eee_path)
        provenance = provenance_by_identity[identity]
        details = details_by_identity[identity]
        if (
            decision.status is not DecisionStatus.COMPLETED
            or decision.tuple_decision is not TupleDecision.CONFIRMED
            or decision.origin_decision is not OriginDecision.PAPER_PRODUCED
            or decision.authority is None
        ):
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "export is not authorized by a completed tuple and origin decision",
            )
        expected_mode = (
            ExportProvenanceMode.TUPLE_AUDITED_HUMAN_REVIEWED
            if item.source_tuple_sidecar_sha256 is not None
            else ExportProvenanceMode.LEGACY_HUMAN_REVIEWED
        )
        if (
            provenance.candidate_payload_sha256 != item.fingerprint.candidate_payload_sha256
            or provenance.review_manifest_sha256 != review.manifest_sha256
            or provenance.review_lock_sha256 != review.lock_sha256
            or provenance.review_decision_sha256 != _decision_sha256(decision)
            or provenance.export_provenance_mode is not expected_mode
            or provenance.source_tuple_sidecar_sha256 != item.source_tuple_sidecar_sha256
            or provenance.source_tuple_gate_sha256 != item.source_tuple_gate_sha256
            or provenance.source_verifier_sidecar_sha256 != item.source_verifier_sidecar_sha256
            or provenance.source_verifier_gate_sha256 != item.source_verifier_gate_sha256
            or provenance.source_verifier_gate_passed != item.source_verifier_gate_passed
            or provenance.decision_authority is not decision.authority.mode
            or provenance.decision_protocol_id != decision.authority.protocol_id
            or provenance.decision_protocol_sha256 != decision.authority.protocol_sha256
            or provenance.proposal_ids != decision.reviewed_tuple.proposal_ids
            or provenance.candidate_occurrence_ids
            != decision.reviewed_tuple.candidate_occurrence_ids
            or provenance.field_provenance != candidate.field_provenance
            or provenance.field_attestations != decision.field_attestations
            or [model_payload(anchor) for anchor in provenance.result_evidence]
            != [
                model_payload(QuoteFreeEvidence.from_private(anchor))
                for anchor in decision.result_evidence
            ]
            or [model_payload(anchor) for anchor in provenance.origin_evidence]
            != [
                model_payload(QuoteFreeEvidence.from_private(anchor))
                for anchor in decision.origin_evidence
            ]
        ):
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "reviewed provenance differs from its sealed candidate or locked decision",
            )
        expected_result: dict[str, Any] = {"score_details": {"details": {}}}
        _inject_review_provenance(
            result=expected_result,
            item=item,
            decision=decision,
            review=review,
        )
        if _review_detail_view(details) != _review_detail_view(
            expected_result["score_details"]["details"]
        ):
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "EEE result differs from its exact locked review evidence",
            )
    try:
        expected_records = _recompose_contextual_eee_records(
            approved=approved,
            originals=originals,
            review=review,
            schema=schema,
            schema_version=manifest.eee_schema_version,
        )
        actual_paths = {
            outcome.eee_path
            for outcome in outcomes
            if outcome.state is ExportOutcomeState.EXPORTED and outcome.eee_path is not None
        }
        if set(expected_records) != actual_paths or any(
            canonical_json_bytes(expected)
            != canonical_json_bytes(read_json(_confined_file(derived_root, path)))
            for path, expected in expected_records.items()
        ):
            raise ValueError("derived EEE payload differs from deterministic recomposition")
    except (OSError, ValueError, ReviewedExportError) as exc:
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "substantive EEE payload differs from the sealed reviewed tuple",
        ) from exc


def verify_derived_run(
    root: Path,
    *,
    schema_path: Path = DEFAULT_EEE_SCHEMA_PATH,
    schema_sha256: str = EEE_SCHEMA_SHA256,
    run_root: Path | None = None,
    review_root: Path | None = None,
) -> DerivedVerification:
    """Verify a reviewed tree, optionally against its sealed run and locked review."""

    if (run_root is None) != (review_root is None):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "contextual verification requires both source run and private review roots",
        )

    try:
        manifest_path = _confined_file(root, DERIVED_MANIFEST_NAME)
        manifest = DerivedRunManifest.model_validate(read_json(manifest_path))
        verification_path = _confined_file(root, DERIVED_VERIFICATION_NAME)
        persisted_verification = DerivedVerification.model_validate(read_json(verification_path))
    except (OSError, ValueError, ValidationError, ReviewedExportError) as exc:
        if isinstance(exc, ReviewedExportError):
            raise
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived control manifest is invalid",
        ) from exc
    try:
        schema, authority = load_schema(schema_path, schema_sha256)
    except (OSError, ValueError) as exc:
        raise _error(
            ReviewedExportErrorCode.EEE_SCHEMA_INVALID, "EEE schema binding failed"
        ) from exc
    if (
        manifest.eee_schema_version != authority.version
        or manifest.eee_schema_sha256 != authority.sha256
    ):
        raise _error(
            ReviewedExportErrorCode.EEE_SCHEMA_INVALID,
            "derived run uses another EEE schema",
        )
    expected_files = {artifact.path for artifact in manifest.payload_files} | {
        DERIVED_MANIFEST_NAME,
        DERIVED_SHA256SUMS_NAME,
        DERIVED_VERIFICATION_NAME,
    }
    _validate_exact_tree(root, expected_files)
    if len({artifact.path for artifact in manifest.payload_files}) != len(manifest.payload_files):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived payload inventory contains duplicate paths",
        )
    for artifact in manifest.payload_files:
        _verify_artifact(root, artifact)
    actual_inventory = _payload_inventory(root)
    if [model_payload(item) for item in actual_inventory] != [
        model_payload(item) for item in manifest.payload_files
    ]:
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived payload inventory differs from manifest",
        )
    expected_tree_sha = sha256_bytes(
        canonical_json_bytes([model_payload(artifact) for artifact in actual_inventory])
    )
    if expected_tree_sha != manifest.payload_tree_sha256:
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived payload tree hash differs",
        )
    sums_path = _confined_file(root, DERIVED_SHA256SUMS_NAME)
    expected_sums = _sha256s_bytes(actual_inventory)
    if (
        sums_path.read_bytes() != expected_sums
        or sha256_bytes(expected_sums) != manifest.sha256s_sha256
    ):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "SHA256SUMS differs from payload inventory",
        )

    outcomes = _jsonl_records(
        _confined_file(root, DERIVED_OUTCOMES_NAME), ExportOutcome, "derived outcomes"
    )
    try:
        provenance = _jsonl_records(
            _confined_file(root, DERIVED_PROVENANCE_NAME),
            ReviewedExportProvenance,
            "derived provenance",
        )
    except ReviewedExportError as exc:
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived reviewed provenance is internally inconsistent",
        ) from exc
    exported = [outcome for outcome in outcomes if outcome.state is ExportOutcomeState.EXPORTED]
    if len({outcome.item_id for outcome in outcomes}) != len(outcomes):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived outcomes contain duplicate items",
        )
    if len({item.evaluation_result_id for item in provenance}) != len(provenance):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived provenance contains duplicate result IDs",
        )
    expected_eee_paths: set[str] = set()
    for outcome in exported:
        assert outcome.evaluation_id is not None and outcome.eee_path is not None
        expected_path = f"{outcome.paper_id}/eee/{outcome.evaluation_id.rsplit('/', 1)[-1]}.json"
        if outcome.eee_path != expected_path:
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "derived outcome EEE path is not its deterministic composition path",
            )
        expected_eee_paths.add(expected_path)
    expected_payload_paths = {
        DERIVED_OUTCOMES_NAME,
        DERIVED_PROVENANCE_NAME,
        *expected_eee_paths,
    }
    if {artifact.path for artifact in manifest.payload_files} != expected_payload_paths:
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived payload inventory contains an unrecognized artifact",
        )
    exported_identities = [
        (outcome.evaluation_id, outcome.evaluation_result_id, outcome.eee_path)
        for outcome in exported
    ]
    provenance_identities = [
        (item.evaluation_id, item.evaluation_result_id, item.eee_path) for item in provenance
    ]
    if len(set(exported_identities)) != len(exported_identities) or set(exported_identities) != set(
        provenance_identities
    ):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "export outcomes and dual provenance are not one-to-one",
        )
    provenance_by_identity = {
        (item.evaluation_id, item.evaluation_result_id, item.eee_path): item for item in provenance
    }
    if (
        manifest.export_provenance_modes
        != sorted({item.export_provenance_mode for item in provenance}, key=lambda item: item.value)
        or manifest.tuple_sidecar_sha256s
        != sorted(
            {
                item.source_tuple_sidecar_sha256
                for item in provenance
                if item.source_tuple_sidecar_sha256 is not None
            }
        )
        or manifest.verifier_sidecar_sha256s
        != sorted(
            {
                item.source_verifier_sidecar_sha256
                for item in provenance
                if item.source_verifier_sidecar_sha256 is not None
            }
        )
    ):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived manifest export provenance differs from result provenance",
        )

    outcomes_by_identity = dict(zip(exported_identities, exported, strict=True))
    observed_results: set[tuple[str, str, str]] = set()
    details_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    eee_artifacts = [
        artifact for artifact in manifest.payload_files if artifact.path in expected_eee_paths
    ]
    for artifact in eee_artifacts:
        try:
            record = read_json(_confined_file(root, artifact.path))
        except (OSError, ValueError) as exc:
            raise _error(
                ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                "EEE payload is invalid JSON",
            ) from exc
        if not isinstance(record, dict) or validate_eee_record(record, schema):
            raise _error(
                ReviewedExportErrorCode.EEE_SCHEMA_INVALID,
                "derived EEE payload fails its pinned schema",
            )
        evaluation_id = record.get("evaluation_id")
        for result in record.get("evaluation_results", []):
            result_id = result.get("evaluation_result_id")
            if not isinstance(evaluation_id, str) or not isinstance(result_id, str):
                raise _error(
                    ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                    "EEE payload lacks stable result identities",
                )
            identity = (evaluation_id, result_id, artifact.path)
            if identity in observed_results:
                raise _error(
                    ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                    "EEE payload contains a duplicate result identity",
                )
            observed_results.add(identity)
            details = result.get("score_details", {}).get("details", {})
            required = {
                "candidate_fingerprint_sha256",
                "review_manifest_sha256",
                "review_lock_sha256",
                "review_decision_sha256",
                "origin_decision",
                "reviewed_result_evidence_count",
                "reviewed_origin_evidence_count",
                "export_provenance_mode",
                "export_composition_sha256",
            }
            if not isinstance(details, dict) or not required.issubset(details):
                raise _error(
                    ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                    "EEE result lacks reviewed dual provenance",
                )
            provenance_item = provenance_by_identity.get(identity)
            outcome = outcomes_by_identity.get(identity)
            if (
                provenance_item is None
                or outcome is None
                or (
                    details.get("export_provenance_mode")
                    != provenance_item.export_provenance_mode.value
                    or details.get("export_composition_sha256")
                    != provenance_item.export_composition_sha256
                    or details.get("tuple_sidecar_sha256")
                    != provenance_item.source_tuple_sidecar_sha256
                    or details.get("tuple_gate_sha256") != provenance_item.source_tuple_gate_sha256
                    or details.get("verifier_sidecar_sha256")
                    != provenance_item.source_verifier_sidecar_sha256
                    or details.get("verifier_gate_sha256")
                    != provenance_item.source_verifier_gate_sha256
                    or details.get("source_verifier_gate_passed")
                    != (
                        str(provenance_item.source_verifier_gate_passed).lower()
                        if provenance_item.source_verifier_gate_passed is not None
                        else None
                    )
                )
            ):
                raise _error(
                    ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
                    "EEE result export provenance differs from its sidecar",
                )
            _validate_public_result_binding(
                details=details,
                outcome=outcome,
                provenance=provenance_item,
            )
            details_by_identity[identity] = details
        _privacy_walk(record)
    if observed_results != {
        (item.evaluation_id, item.evaluation_result_id, item.eee_path) for item in provenance
    }:
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "EEE results and provenance sidecar are not one-to-one",
        )
    expected_counts = _derived_counts(
        outcomes=outcomes,
        eee_record_count=len(eee_artifacts),
        eee_observation_count=len(observed_results),
    )
    if manifest.counts != expected_counts:
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "derived manifest counts differ from recomputed records",
        )
    if run_root is not None and review_root is not None:
        _validate_contextual_derived_bindings(
            derived_root=root,
            manifest=manifest,
            schema=schema,
            outcomes=outcomes,
            provenance_by_identity=provenance_by_identity,
            details_by_identity=details_by_identity,
            run_root=run_root,
            review_root=review_root,
        )
    _privacy_walk([model_payload(item) for item in outcomes])
    _privacy_walk([model_payload(item) for item in provenance])
    expected_verification = _expected_verification(
        manifest_path=manifest_path,
        manifest=manifest,
        root=root,
    )
    if model_payload(persisted_verification) != model_payload(expected_verification):
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "verification receipt differs from recomputed state",
        )
    return expected_verification


def verify_contextual_derived_run(
    root: Path,
    *,
    run_root: Path,
    review_root: Path,
    schema_path: Path = DEFAULT_EEE_SCHEMA_PATH,
    schema_sha256: str = EEE_SCHEMA_SHA256,
) -> ContextualDerivedVerification:
    """Contextually verify a derived run and expose only safe authority aggregates."""

    verification = verify_derived_run(
        root,
        schema_path=schema_path,
        schema_sha256=schema_sha256,
        run_root=run_root,
        review_root=review_root,
    )
    provenance = _jsonl_records(
        _confined_file(root, DERIVED_PROVENANCE_NAME),
        ReviewedExportProvenance,
        "derived provenance",
    )
    observed = Counter(item.decision_authority.value for item in provenance)
    counts = {mode.value: observed[mode.value] for mode in ReviewAuthorityMode}
    provenance_observed = Counter(item.export_provenance_mode.value for item in provenance)
    provenance_counts = {
        mode.value: provenance_observed[mode.value] for mode in ExportProvenanceMode
    }
    if sum(counts.values()) != verification.exported_observation_count:
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "authority aggregates differ from context-bound exports",
        )
    if sum(provenance_counts.values()) != verification.exported_observation_count:
        raise _error(
            ReviewedExportErrorCode.DERIVED_INTEGRITY_FAILURE,
            "provenance aggregates differ from context-bound exports",
        )
    return ContextualDerivedVerification(
        verification=verification,
        authority_mode_counts=counts,
        provenance_mode_counts=provenance_counts,
    )
