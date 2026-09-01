"""Deterministic, quote-free Paper Extraction Review Cards.

Extraction Review Cards summarize one paper-run for demonstration and audit.
They are deliberately not Evaluation Cards and contain no source quotations,
provider traces, request identifiers, credentials, private notes, or local paths.
"""

# ruff: noqa: E501 -- compact self-contained HTML and CSS are intentionally inline.

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from html import escape
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import parse_qsl, urlparse

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ExportStatus
from proceedings_to_eee.io import (
    atomic_write_bytes,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from proceedings_to_eee.resources import DEFAULT_EEE_SCHEMA_PATH
from proceedings_to_eee.sources.manifest import SourceManifest
from proceedings_to_eee.validation.eee_schema import (
    SchemaAuthority,
    load_schema,
    validate_eee_record,
)

CARD_SCHEMA_VERSION = "paper-extraction-review-card/0.1"
INDEX_SCHEMA_VERSION = "paper-extraction-review-index/0.1"
CARD_STATEMENT = "Extraction Review Card; not an Evaluation Card."
PINNED_EEE_SCHEMA_VERSION = "0.2.2"
PINNED_EEE_SCHEMA_SHA256 = "088fed8029d42fb3a607aa67e1a05c39e425241b5cd90803705b37562f402f2a"
PINNED_EEE_SCHEMA_PATH = DEFAULT_EEE_SCHEMA_PATH
POST_HOC_CORRECTED_HOLDOUT_LIMITATION = (
    "The immutable first holdout run is preserved separately; this card reflects the "
    "post-hoc corrected run, not the original held-out result."
)

Split = Literal["development", "holdout"]
EvaluationStatus = Literal["development", "holdout", "post_hoc_corrected"]

__all__ = [
    "CARD_SCHEMA_VERSION",
    "CARD_STATEMENT",
    "INDEX_SCHEMA_VERSION",
    "PINNED_EEE_SCHEMA_SHA256",
    "PINNED_EEE_SCHEMA_VERSION",
    "POST_HOC_CORRECTED_HOLDOUT_LIMITATION",
    "CorpusCardInput",
    "ExtractionReviewCardError",
    "build_extraction_review_index",
    "build_paper_extraction_review_card",
    "render_extraction_review_index",
    "render_paper_extraction_review_card",
    "write_extraction_review_bundle",
]

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_PATH = re.compile(
    r"(?:^|[\s\"'=:(])(?:/Users/|/home/|/private/|/tmp/|/var/folders/|"
    r"[A-Za-z]:\\Users\\|file://)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~-]{12,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|authorization)\s*[:=]\s*[\"'][^\"']{8,}[\"']", re.I),
)
_FORBIDDEN_KEYS = {
    "api_key",
    "authorization",
    "cache_relpath",
    "calls",
    "command",
    "completion",
    "cookie",
    "exact_quote",
    "messages",
    "notes",
    "prompt",
    "provider_request",
    "provider_response",
    "quote",
    "raw_payload",
    "raw_response",
    "request_id",
    "reviewer_notes",
    "secret",
    "warnings",
}
_REASON_NAMES = (
    "paper_run_error",
    "no_candidates",
    "unsupported_evidence",
    "unverified_evidence",
    "unresolved_role",
    "unresolved_scope",
    "unresolved_metric",
    "unresolved_unit",
    "non_primary_claim_type",
    "low_confidence",
    "not_export_eligible",
    "schema_rejection",
    "correct_abstention",
)
_RUN_REVIEW_REASONS = {
    "candidate_review_required",
    "selected_result_blocks_produced_zero_candidates",
    "zero_selected_result_blocks",
    "zero_valid_eee_records",
    "eee_schema_validation_failure",
    "paper_run_error",
}
_METRIC_FIELDS = (
    "claim_type",
    "dataset",
    "evidence_column",
    "evidence_kind",
    "evidence_label",
    "evidence_row",
    "evidence_structure",
    "evidence_supported",
    "joint_semantics",
    "metric",
    "missingness",
    "page",
    "slice",
    "system",
    "unit",
    "value",
)


class ExtractionReviewCardError(ValueError):
    """One or more private artifacts cannot be projected safely or consistently."""


@lru_cache(maxsize=1)
def _pinned_eee_schema() -> tuple[dict[str, Any], SchemaAuthority]:
    """Load the repository-pinned EEE authority used for every public copy."""

    try:
        schema, authority = load_schema(
            PINNED_EEE_SCHEMA_PATH,
            expected_sha256=PINNED_EEE_SCHEMA_SHA256,
        )
    except (OSError, ValueError) as error:
        raise ExtractionReviewCardError(
            "the pinned EEE schema is unavailable or changed"
        ) from error
    if authority.version != PINNED_EEE_SCHEMA_VERSION:
        raise ExtractionReviewCardError("the pinned EEE schema version is unexpected")
    return schema, authority


@dataclass(frozen=True)
class CorpusCardInput:
    """Inputs for one split in a multi-corpus review-card bundle."""

    split: Split
    run_root: Path
    evaluation_status: EvaluationStatus | None = None
    human_review_summary_path: Path | None = None
    eee_link_prefix: str | None = None
    known_limitations: tuple[str, ...] = ()
    paper_limitations: Mapping[str, Sequence[str]] = field(default_factory=dict)
    paper_review_outcomes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    correct_abstentions: frozenset[str] = frozenset()


def _resolve_evaluation_status(
    split: Split,
    evaluation_status: EvaluationStatus | None,
) -> EvaluationStatus:
    resolved: EvaluationStatus = split if evaluation_status is None else evaluation_status
    allowed: set[EvaluationStatus] = (
        {"development"} if split == "development" else {"holdout", "post_hoc_corrected"}
    )
    if resolved not in allowed:
        raise ExtractionReviewCardError(
            f"evaluation status {resolved!r} is incompatible with the {split} split"
        )
    return resolved


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExtractionReviewCardError(f"{context} must be a JSON object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ExtractionReviewCardError(f"{context} must be a JSON array")
    return value


def _nonnegative_int(value: Any, context: str, *, default: int = 0) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ExtractionReviewCardError(f"{context} must be a non-negative integer")
    return value


def _optional_nonnegative_float(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExtractionReviewCardError(f"{context} must be a non-negative number or null")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ExtractionReviewCardError(f"{context} must be a non-negative finite number")
    return result


def _finite_rate(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExtractionReviewCardError(f"{context} must be a number or null")
    result = float(value)
    if not math.isfinite(result) or result < 0 or result > 1:
        raise ExtractionReviewCardError(f"{context} must be between zero and one")
    return result


def _safe_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ExtractionReviewCardError(f"{context} is not a safe stable ID")
    return value


def _safe_public_url(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ExtractionReviewCardError(f"{context} must be a public HTTP(S) URL")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
        raise ExtractionReviewCardError(f"{context} must be a public HTTP(S) URL")
    sensitive_query_names = {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "signature",
        "token",
    }
    if any(name.casefold() in sensitive_query_names for name, _ in parse_qsl(parsed.query)):
        raise ExtractionReviewCardError(f"{context} contains a credential-like query parameter")
    return value


def _safe_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExtractionReviewCardError(f"{context} must be non-empty text")
    if _LOCAL_PATH.search(value) or any(pattern.search(value) for pattern in _SECRET_PATTERNS):
        raise ExtractionReviewCardError(f"{context} contains private or secret material")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ExtractionReviewCardError(f"{context} must be a lowercase SHA-256")
    return value


def _relative_href(value: str, context: str) -> str:
    if not value or "\\" in value or _LOCAL_PATH.search(value):
        raise ExtractionReviewCardError(f"{context} must be a relative public path")
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc or value.startswith("/"):
        raise ExtractionReviewCardError(f"{context} must be a relative public path")
    parts = PurePosixPath(value).parts
    if not parts or any(part in {"", "."} for part in parts):
        raise ExtractionReviewCardError(f"{context} must be a normalized relative path")
    return value


def _assert_public_payload(value: Any, context: str = "card") -> None:
    """Reject privacy leaks after allowlist projection and before publication."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExtractionReviewCardError(f"{context} has a non-string key")
            folded = key.casefold()
            permitted_quote_key = folded in {
                "quote_hashes_present",
                "quote_sha256",
            } or folded.endswith("_quote_sha256")
            if folded in _FORBIDDEN_KEYS or ("quote" in folded and not permitted_quote_key):
                raise ExtractionReviewCardError(f"{context} contains forbidden key {key}")
            _assert_public_payload(item, f"{context}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _assert_public_payload(item, f"{context}[{index}]")
        return
    if isinstance(value, str):
        if _LOCAL_PATH.search(value) or any(pattern.search(value) for pattern in _SECRET_PATTERNS):
            raise ExtractionReviewCardError(f"{context} contains private or secret material")
        return
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ExtractionReviewCardError(f"{context} contains an unsupported value")


def _assert_no_evidence_text(
    payload: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]
) -> None:
    """Ensure public free text did not accidentally reproduce a meaningful source quote."""

    quote_norms: set[str] = set()
    for candidate in observations:
        evidence = candidate.get("evidence")
        if not isinstance(evidence, Sequence) or isinstance(evidence, str | bytes | bytearray):
            continue
        for raw_anchor in evidence:
            if not isinstance(raw_anchor, Mapping):
                continue
            quote = raw_anchor.get("quote")
            if isinstance(quote, str) and len(quote.strip()) >= 16:
                quote_norms.add(re.sub(r"\s+", " ", quote).strip())

    def audit(item: Any) -> None:
        if isinstance(item, Mapping):
            for child in item.values():
                audit(child)
        elif isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            for child in item:
                audit(child)
        elif isinstance(item, str) and re.sub(r"\s+", " ", item).strip() in quote_norms:
            raise ExtractionReviewCardError("public artifact reproduces an evidence quotation")

    audit(payload)


def _public_eee_projection(path: Path, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create one quote-free projection and revalidate it against pinned EEE."""

    quote_norms = {
        re.sub(r"\s+", " ", str(anchor.get("quote"))).strip()
        for candidate in observations
        for anchor in candidate.get("evidence", [])
        if isinstance(anchor, Mapping)
        and isinstance(anchor.get("quote"), str)
        and len(str(anchor.get("quote")).strip()) >= 16
    }

    def project(item: Any) -> Any:
        if isinstance(item, Mapping):
            projected: dict[str, Any] = {}
            for key, child in item.items():
                if (
                    re.fullmatch(r"evidence_[1-9][0-9]*_(?:label|row|column)", str(key))
                    and isinstance(child, str)
                    and re.sub(r"\s+", " ", child).strip() in quote_norms
                ):
                    continue
                projected[str(key)] = project(child)
            return projected
        if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            return [project(child) for child in item]
        return item

    payload = _mapping(read_json(path), f"EEE artifact {path.name}")
    result = _mapping(project(payload), f"EEE artifact {path.name}")
    projected = dict(result)
    _assert_public_payload(projected, f"EEE artifact {path.name}")
    _assert_no_evidence_text(projected, observations)
    schema, _ = _pinned_eee_schema()
    issues = validate_eee_record(projected, schema)
    if issues:
        summary = "; ".join(f"{issue.path}: {issue.message}" for issue in issues[:3])
        raise ExtractionReviewCardError(
            f"public EEE artifact {path.name} failed pinned-schema validation: {summary}"
        )
    return projected


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ExtractionReviewCardError(f"missing regular artifact: {path.name}")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = _mapping(json.loads(line), f"{path.name} line {line_number}")
        except json.JSONDecodeError as error:
            raise ExtractionReviewCardError(
                f"invalid JSON in {path.name} line {line_number}"
            ) from error
        try:
            CandidateObservation.model_validate(record)
        except ValueError as error:
            raise ExtractionReviewCardError(
                f"invalid candidate observation in {path.name} line {line_number}"
            ) from error
        records.append(record)
    return records


def _configuration_projection(run: Mapping[str, Any]) -> dict[str, Any]:
    extractor = _mapping(run.get("extractor", {}), "run.extractor")
    verifier = _mapping(run.get("verifier", {}), "run.verifier")
    eee_schema = _mapping(run.get("eee_schema", {}), "run.eee_schema")
    segmentation = _mapping(run.get("result_block_segmentation", {}), "segmentation")
    extractor_contract = _mapping(extractor.get("request_contract", {}), "extractor contract")
    verifier_contract = _mapping(verifier.get("request_contract", {}), "verifier contract")
    return {
        "layout_parser": run.get("layout_parser"),
        "layout_parser_version": run.get("layout_parser_version"),
        "result_block_segmentation": dict(sorted(segmentation.items())),
        "extractor": {
            "model": extractor.get("model"),
            "provider": extractor.get("provider"),
            "prompt_sha256": extractor.get("prompt_sha256"),
            "temperature": extractor.get("temperature"),
            "seed": extractor.get("seed"),
            "max_tokens": extractor.get("max_tokens"),
            "reasoning_effort": extractor.get("reasoning_effort"),
            "request_contract": extractor_contract,
        },
        "verifier": {
            "enabled": verifier.get("enabled"),
            "model": verifier.get("model"),
            "request_contract": verifier_contract,
        },
        "eee_schema": {
            "version": eee_schema.get("version"),
            "sha256": eee_schema.get("sha256"),
        },
    }


def _source_projection(manifest: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    sources = _sequence(manifest.get("sources"), "source manifest.sources")
    public_sources: list[dict[str, Any]] = []
    for index, raw in enumerate(sources):
        source = _mapping(raw, f"source manifest.sources[{index}]")
        source_id = _safe_id(source.get("source_id"), "source_id")
        role = _safe_text(source.get("role"), "source role")
        url_value = source.get("original_uri") or source.get("resolved_uri")
        digest = source.get("sha256")
        git_commit = source.get("git_commit")
        if digest is not None:
            digest = _sha256(digest, "source SHA-256")
        if git_commit is not None and (
            not isinstance(git_commit, str) or re.fullmatch(r"[0-9a-f]{40}", git_commit) is None
        ):
            raise ExtractionReviewCardError("source git_commit must be a lowercase SHA-1")
        if digest is None and git_commit is None:
            raise ExtractionReviewCardError(
                "every public source requires a content hash or git commit"
            )
        media_type = source.get("media_type")
        public_sources.append(
            {
                "source_id": source_id,
                "role": role,
                "public_url": _safe_public_url(url_value, "source URL"),
                "retrieved_at": _safe_text(source.get("retrieved_at"), "retrieval time"),
                "sha256": digest,
                "git_commit": git_commit,
                "media_type": (
                    _safe_text(media_type, "source media type") if media_type is not None else None
                ),
            }
        )
    if not public_sources:
        raise ExtractionReviewCardError("source manifest has no frozen sources")
    public_sources.sort(key=lambda item: (item["role"], item["source_id"]))
    proceedings_url = manifest.get("proceedings_url")
    public_url = (
        _safe_public_url(proceedings_url, "paper public URL")
        if proceedings_url is not None
        else public_sources[0]["public_url"]
    )
    return public_url, public_sources


def _validate_observation_bindings(
    observations: Sequence[Mapping[str, Any]],
    *,
    paper_id: str,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, CandidateObservation], dict[str, CandidateObservation]]:
    """Bind typed observations to this paper and its frozen source manifest."""

    source_ids = {_safe_id(source.get("source_id"), "source_id") for source in sources}
    candidates: dict[str, CandidateObservation] = {}
    exported: dict[str, CandidateObservation] = {}
    for index, raw in enumerate(observations):
        try:
            candidate = CandidateObservation.model_validate(raw)
        except ValueError as error:
            raise ExtractionReviewCardError(
                f"observations.jsonl candidate {index} failed typed validation"
            ) from error
        if candidate.paper_id != paper_id:
            raise ExtractionReviewCardError("candidate paper_id does not match its paper run")
        observation_id = _safe_id(candidate.observation_id, "observation_id")
        if observation_id in candidates:
            raise ExtractionReviewCardError("observation IDs must be unique within a paper run")
        for anchor in candidate.evidence:
            if anchor.source_id not in source_ids:
                raise ExtractionReviewCardError(
                    "candidate evidence source_id is absent from the frozen source manifest"
                )
        candidates[observation_id] = candidate
        if candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}:
            exported[observation_id] = candidate
    return candidates, exported


def _validate_public_eee_provenance(
    record: Mapping[str, Any],
    *,
    paper_id: str,
    exported: Mapping[str, CandidateObservation],
    sources: Sequence[Mapping[str, Any]],
    context: str,
) -> set[str]:
    """Bind each public EEE result to one exported candidate and frozen source."""

    source_by_id = {_safe_id(source.get("source_id"), "source_id"): source for source in sources}
    source_metadata = _mapping(record.get("source_metadata"), f"{context}.source_metadata")
    source_details = _mapping(
        source_metadata.get("additional_details"),
        f"{context}.source_metadata.additional_details",
    )
    if source_details.get("paper_id") != paper_id:
        raise ExtractionReviewCardError(f"{context} has mismatched paper provenance")

    result_ids: set[str] = set()
    for index, raw_result in enumerate(
        _sequence(record.get("evaluation_results"), f"{context}.evaluation_results")
    ):
        result_context = f"{context}.evaluation_results[{index}]"
        result = _mapping(raw_result, result_context)
        result_id = _safe_id(result.get("evaluation_result_id"), "EEE result ID")
        score_details = _mapping(result.get("score_details"), f"{result_context}.score_details")
        details = _mapping(score_details.get("details"), f"{result_context}.details")
        candidate_id = _safe_id(details.get("candidate_observation_id"), "candidate ID")
        if result_id != candidate_id:
            raise ExtractionReviewCardError(
                f"{result_context} result ID does not match candidate provenance"
            )
        candidate = exported.get(result_id)
        if candidate is None:
            raise ExtractionReviewCardError(
                f"{result_context} is not backed by one exported candidate"
            )
        if result_id in result_ids:
            raise ExtractionReviewCardError(f"{context} contains a duplicate EEE result ID")
        result_ids.add(result_id)
        if details.get("paper_id") != paper_id:
            raise ExtractionReviewCardError(f"{result_context} has mismatched paper provenance")
        if candidate.value is None or score_details.get("score") != candidate.value.numeric:
            raise ExtractionReviewCardError(
                f"{result_context} score does not match the exported candidate"
            )

        raw_anchor_count = details.get("evidence_anchor_count")
        if not isinstance(raw_anchor_count, str) or not re.fullmatch(
            r"[1-9][0-9]*", raw_anchor_count
        ):
            raise ExtractionReviewCardError(f"{result_context} has invalid anchor count")
        if int(raw_anchor_count) != len(candidate.evidence):
            raise ExtractionReviewCardError(
                f"{result_context} anchor count does not match the exported candidate"
            )
        for anchor_index, anchor in enumerate(candidate.evidence, start=1):
            prefix = f"evidence_{anchor_index}"
            source = source_by_id.get(anchor.source_id)
            if source is None:
                raise ExtractionReviewCardError(
                    f"{result_context} references an unknown frozen source"
                )
            expected = {
                f"{prefix}_source_id": anchor.source_id,
                f"{prefix}_source_role": source.get("role"),
                f"{prefix}_page": str(anchor.page),
                f"{prefix}_kind": anchor.kind.value,
                f"{prefix}_quote_sha256": anchor.quote_sha256,
            }
            source_sha256 = source.get("sha256")
            source_git_commit = source.get("git_commit")
            if source_sha256 is not None:
                expected[f"{prefix}_source_sha256"] = source_sha256
            elif source_git_commit is not None:
                expected[f"{prefix}_source_git_commit"] = source_git_commit
            else:  # guarded by source projection, retained as a defensive invariant
                raise ExtractionReviewCardError(f"{result_context} source is not immutable")
            for name, expected_value in expected.items():
                if details.get(name) != expected_value:
                    raise ExtractionReviewCardError(
                        f"{result_context} has source/evidence provenance mismatch at {name}"
                    )

            normalized_quote = re.sub(r"\s+", " ", anchor.quote).strip()
            for name in ("label", "row", "column"):
                key = f"{prefix}_{name}"
                value = getattr(anchor, name)
                quote_equivalent = (
                    value is not None and re.sub(r"\s+", " ", value).strip() == normalized_quote
                )
                if value is None or quote_equivalent:
                    if key in details:
                        raise ExtractionReviewCardError(
                            f"{result_context} retains a non-public structural field at {key}"
                        )
                elif details.get(key) != value:
                    raise ExtractionReviewCardError(
                        f"{result_context} has structural provenance mismatch at {key}"
                    )
    return result_ids


def _structure_field(
    anchor: Mapping[str, Any], name: str, normalized_quote: str | None
) -> str | None:
    value = anchor.get(name)
    if not isinstance(value, str):
        return None
    safe_value = _safe_text(value, f"anchor {name}")
    if normalized_quote == re.sub(r"\s+", " ", safe_value).strip():
        return None
    return safe_value


def _provenance_projection(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    numeric_exported = 0
    complete_observations = 0
    for candidate in observations:
        if candidate.get("export_status") not in {"exported", "eligible"}:
            continue
        value = candidate.get("value")
        if not isinstance(value, Mapping) or not isinstance(value.get("numeric"), int | float):
            continue
        numeric_exported += 1
        observation_id = _safe_id(candidate.get("observation_id"), "observation_id")
        evidence = _sequence(candidate.get("evidence", []), "candidate.evidence")
        observation_complete = bool(evidence)
        for raw_anchor in evidence:
            anchor = _mapping(raw_anchor, "candidate evidence anchor")
            quote = anchor.get("quote")
            normalized_quote = (
                re.sub(r"\s+", " ", quote).strip() if isinstance(quote, str) else None
            )

            projected = {
                "observation_id": observation_id,
                "source_id": _safe_id(anchor.get("source_id"), "anchor source_id"),
                "page": _nonnegative_int(anchor.get("page"), "anchor page"),
                "kind": _safe_text(anchor.get("kind"), "anchor kind"),
                "label": _structure_field(anchor, "label", normalized_quote),
                "row": _structure_field(anchor, "row", normalized_quote),
                "column": _structure_field(anchor, "column", normalized_quote),
                "quote_sha256": _sha256(anchor.get("quote_sha256"), "anchor quote SHA-256"),
            }
            if projected["page"] < 1:
                raise ExtractionReviewCardError("anchor page must be positive")
            anchors.append(projected)
            observation_complete = observation_complete and all(
                projected[name] is not None
                for name in ("source_id", "page", "kind", "quote_sha256")
            )
        complete_observations += int(observation_complete)
    anchors.sort(
        key=lambda item: (
            item["observation_id"],
            item["source_id"],
            item["page"],
            item["kind"],
            item["quote_sha256"],
        )
    )
    source_ids = sorted({anchor["source_id"] for anchor in anchors})
    pages = sorted({anchor["page"] for anchor in anchors})
    kind_counts = Counter(anchor["kind"] for anchor in anchors)
    return {
        "policy": "quote-free source/page/structure/hash provenance",
        "numeric_exported_observations": numeric_exported,
        "complete_numeric_observations": complete_observations,
        "complete": numeric_exported > 0 and numeric_exported == complete_observations,
        "quote_hashes_present": bool(anchors) and all(anchor["quote_sha256"] for anchor in anchors),
        "anchor_count": len(anchors),
        "source_ids": source_ids,
        "pages": pages,
        "structure_kind_counts": dict(sorted(kind_counts.items())),
        "anchors": anchors,
    }


def _reason_counts(
    observations: Sequence[Mapping[str, Any]],
    *,
    eee_count: int,
    schema_issues: int,
    correct_abstention: bool,
    paper_run_error: bool,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if paper_run_error:
        counts["paper_run_error"] = 1
    if not observations:
        counts["no_candidates"] = 1
    for candidate in observations:
        if candidate.get("export_status") in {"exported", "eligible"}:
            continue
        text_support = candidate.get("text_support")
        if text_support in {"unsupported", "partially_supported"}:
            counts["unsupported_evidence"] += 1
        elif text_support != "supported":
            counts["unverified_evidence"] += 1
        roles = candidate.get("roles")
        if not isinstance(roles, Sequence) or isinstance(roles, str | bytes | bytearray):
            roles = []
        evaluated = [
            role
            for role in roles
            if isinstance(role, Mapping) and role.get("role") == "evaluated_system"
        ]
        if len(evaluated) != 1 or candidate.get("referential_status") != "resolved":
            counts["unresolved_role"] += 1
        if candidate.get("scope") is None or candidate.get("referential_status") != "resolved":
            counts["unresolved_scope"] += 1
        metric = candidate.get("metric")
        if not isinstance(metric, Mapping) or metric.get("canonical_id") is None:
            counts["unresolved_metric"] += 1
        value = candidate.get("value")
        metric_unit = metric.get("unit") if isinstance(metric, Mapping) else None
        value_unit = value.get("unit") if isinstance(value, Mapping) else None
        if metric_unit is None or value_unit is None or metric_unit != value_unit:
            counts["unresolved_unit"] += 1
        if candidate.get("claim_type") != "primary_result":
            counts["non_primary_claim_type"] += 1
        reason = candidate.get("export_reason")
        if isinstance(reason, str) and "confidence" in reason.casefold():
            counts["low_confidence"] += 1
        counts["not_export_eligible"] += 1
    counts["schema_rejection"] = schema_issues
    if correct_abstention:
        if eee_count:
            raise ExtractionReviewCardError("correct abstention cannot be set for a paper with EEE")
        counts["correct_abstention"] = 1
    return {reason: counts.get(reason, 0) for reason in _REASON_NAMES}


def _primary_abstention_reason(counts: Mapping[str, int]) -> str | None:
    priority = (
        "paper_run_error",
        "no_candidates",
        "schema_rejection",
        "unsupported_evidence",
        "unresolved_role",
        "unresolved_scope",
        "unresolved_metric",
        "unresolved_unit",
        "non_primary_claim_type",
        "low_confidence",
        "not_export_eligible",
        "correct_abstention",
    )
    return next((reason for reason in priority if counts.get(reason, 0)), None)


def _reference_projection(run: Mapping[str, Any]) -> dict[str, Any]:
    raw = run.get("reference_evaluation")
    if not isinstance(raw, Mapping):
        return {"status": "not_measured", "coverage_statement": "No reference targets scored."}
    coverage = _mapping(raw.get("coverage", {}), "reference coverage")
    detection = _mapping(raw.get("detection", {}), "reference detection")
    fields = _mapping(raw.get("field_accuracy", {}), "reference field accuracy")
    safety = _mapping(raw.get("negative_control_safety", {}), "negative-control safety")
    fully_annotated = coverage.get("fully_annotated_labels", [])
    sampled = coverage.get("sampled_labels", [])
    return {
        "status": "measured",
        "coverage": {
            "statement": "Selected reference targets only; not whole-paper recall.",
            "recall_basis": _nonnegative_int(detection.get("recall_basis"), "recall basis"),
            "precision_basis": _nonnegative_int(
                detection.get("precision_basis"), "precision basis"
            ),
            "fully_annotated_regions": len(_sequence(fully_annotated, "fully annotated labels")),
            "sampled_regions": len(_sequence(sampled, "sampled labels")),
        },
        "detection": {
            name: _finite_rate(detection.get(name), f"reference detection.{name}")
            for name in ("precision", "recall", "f1")
        }
        | {
            "true_positives": _nonnegative_int(detection.get("true_positives"), "TP"),
            "false_positives": _nonnegative_int(detection.get("false_positives"), "FP"),
            "false_negatives": _nonnegative_int(detection.get("false_negatives"), "FN"),
        },
        "field_accuracy": {
            name: _finite_rate(fields.get(name), f"reference field_accuracy.{name}")
            for name in _METRIC_FIELDS
        },
        "negative_control_safety": {
            "measurement_status": str(safety.get("measurement_status", "not_measured")),
            "controls_total": _nonnegative_int(safety.get("controls_total"), "controls total"),
            "controls_matched": _nonnegative_int(
                safety.get("matched_control_count"), "controls matched"
            ),
            "false_primary_candidates": _nonnegative_int(
                safety.get("false_primary_count"), "false-primary candidates"
            ),
            "false_primary_exports": _nonnegative_int(
                safety.get("false_primary_export_count"), "false-primary exports"
            ),
        },
    }


def _quality_gate_projection(aggregate: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(aggregate, Mapping):
        return []
    raw_gates = aggregate.get("quality_gates")
    if not isinstance(raw_gates, Mapping):
        return []
    gates: list[dict[str, Any]] = []
    for name, raw in sorted(raw_gates.items()):
        gate = _mapping(raw, f"quality gate {name}")
        status = gate.get("status")
        if status not in {"passed", "failed", "not_measured"}:
            raise ExtractionReviewCardError(f"quality gate {name} has an invalid status")
        value = gate.get("value")
        threshold = gate.get("threshold")
        gates.append(
            {
                "name": _safe_text(name, "quality gate name"),
                "scope": "split",
                "status": status,
                "direction": _safe_text(gate.get("direction"), "quality gate direction"),
                "value": float(value) if isinstance(value, int | float) else None,
                "threshold": float(threshold) if isinstance(threshold, int | float) else None,
            }
        )
    return gates


def _aggregate_review_projection(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    if summary.get("schema_version") != "human-review-summary/0.1":
        raise ExtractionReviewCardError("unsupported human-review summary schema")
    decisions = _mapping(summary.get("decisions"), "human-review decisions")
    sample = _mapping(summary.get("sample"), "human-review sample")
    outcomes = _mapping(decisions.get("outcome_counts"), "human-review outcomes")
    issues = _mapping(decisions.get("issue_counts"), "human-review issues")
    return {
        "scope": "split_aggregate",
        "audit_id": _safe_id(summary.get("audit_id"), "human-review audit_id"),
        "completed": _nonnegative_int(decisions.get("completed"), "completed reviews"),
        "papers_reviewed": _nonnegative_int(sample.get("papers_reviewed"), "papers reviewed"),
        "paper_coverage": _finite_rate(sample.get("paper_coverage"), "review paper coverage"),
        "outcome_counts": {
            name: _nonnegative_int(outcomes.get(name), f"review outcome {name}")
            for name in ("confirmed", "incorrect", "needs_followup")
        },
        "issue_counts": {
            str(name): _nonnegative_int(value, f"review issue {name}")
            for name, value in sorted(issues.items())
        },
    }


def _validate_review_population(summary: Mapping[str, Any] | None, raw_runs: Sequence[Any]) -> None:
    expected_papers = len(raw_runs)
    expected_candidates = 0
    paper_ids: set[str] = set()
    for item in raw_runs:
        run = _mapping(item, "paper run")
        paper_ids.add(_safe_id(run.get("paper_id"), "paper run ID"))
        counts = _mapping(run.get("counts", {}), "paper run counts")
        expected_candidates += _nonnegative_int(counts.get("candidates"), "candidate count")
    if len(paper_ids) != expected_papers:
        raise ExtractionReviewCardError("corpus-run paper IDs must be unique")
    if summary is None:
        return
    population = _mapping(summary.get("population"), "human-review population")
    if _nonnegative_int(population.get("papers"), "review paper population") != expected_papers:
        raise ExtractionReviewCardError("human-review paper population does not match split")
    if (
        _nonnegative_int(population.get("candidates"), "review candidate population")
        != expected_candidates
    ):
        raise ExtractionReviewCardError("human-review candidate population does not match split")


def _paper_review_projection(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    outcome = value.get("outcome")
    if outcome != "included_in_analyst_review":
        raise ExtractionReviewCardError("paper review outcome is invalid")
    decision = value.get("decision")
    if decision != "withheld_in_public_artifacts":
        raise ExtractionReviewCardError("paper review decision must be withheld")
    return {
        "scope": "paper",
        "audit_id": _safe_id(value.get("audit_id"), "paper review audit_id"),
        "outcome": outcome,
        "decision": decision,
    }


def _automatic_limitations(
    run: Mapping[str, Any],
    reference: Mapping[str, Any],
    reason_counts: Mapping[str, int],
    eee_count: int,
) -> list[str]:
    limitations = (
        ["Reference metrics cover selected annotations only and are not whole-paper recall."]
        if reference.get("status") == "measured"
        else ["Reference quality was not measured for this paper."]
    )
    counts = _mapping(run.get("counts", {}), "run counts")
    if run.get("status") != "success":
        limitations.append("The technical paper run did not complete successfully.")
    if _nonnegative_int(counts.get("eee_schema_issues"), "EEE schema issues"):
        limitations.append("One or more composed EEE records failed pinned-schema validation.")
    if eee_count == 0:
        reason = _primary_abstention_reason(reason_counts)
        limitations.append(
            "No EEE record was produced; the structured abstention reasons identify the cause."
            if reason
            else "No EEE record was produced and no reason could be established from public artifacts."
        )
    false_primary = _nonnegative_int(counts.get("negative_control_false_primary"), "false primary")
    if false_primary:
        limitations.append("Known false-primary candidates remain in the scored run.")
    return limitations


def _run_review_state_projection(
    run: Mapping[str, Any],
    *,
    candidate_count: int,
    eee_count: int,
    schema_issues: int,
) -> dict[str, Any]:
    """Project the optional run-level review signal, with a legacy fallback."""

    raw = run.get("review_state")
    if isinstance(raw, Mapping):
        status = raw.get("status")
        if status not in {"ready", "needs_review", "blocked"}:
            raise ExtractionReviewCardError("run review_state.status is invalid")
        raw_reasons = _sequence(raw.get("reasons", []), "run review_state.reasons")
        reasons = sorted({_safe_text(reason, "run review reason") for reason in raw_reasons})
        unknown = set(reasons) - _RUN_REVIEW_REASONS
        if unknown:
            raise ExtractionReviewCardError(
                f"run review_state contains unsupported reasons: {sorted(unknown)}"
            )
        return {"status": status, "reasons": reasons, "source": "run_manifest"}

    reasons: list[str] = []
    if run.get("status") != "success":
        reasons.append("paper_run_error")
    if not _sequence(run.get("selected_blocks", []), "selected blocks"):
        reasons.append("zero_selected_result_blocks")
    elif candidate_count == 0:
        reasons.append("selected_result_blocks_produced_zero_candidates")
    counts = _mapping(run.get("counts", {}), "run counts")
    if _nonnegative_int(counts.get("candidates_needing_review"), "candidate reviews"):
        reasons.append("candidate_review_required")
    if eee_count == 0:
        reasons.append("zero_valid_eee_records")
    if schema_issues:
        reasons.append("eee_schema_validation_failure")
    status = "blocked" if run.get("status") != "success" else "needs_review" if reasons else "ready"
    return {"status": status, "reasons": sorted(set(reasons)), "source": "derived_legacy_fallback"}


def _eee_links(
    paper_root: Path,
    *,
    href_prefix: str,
    paper_id: str,
    exported: Mapping[str, CandidateObservation],
    sources: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], set[str]]:
    eee_root = paper_root / "eee"
    if not eee_root.exists():
        return [], set()
    if not eee_root.is_dir() or eee_root.is_symlink():
        raise ExtractionReviewCardError("EEE artifact root must be a regular directory")
    prefix = _relative_href(href_prefix, "EEE link prefix").rstrip("/")
    observations = _read_jsonl(paper_root / "observations.jsonl")
    links: list[dict[str, str]] = []
    result_ids: set[str] = set()
    for path in sorted(eee_root.glob("*.json"), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink():
            raise ExtractionReviewCardError("EEE artifacts must be regular JSON files")
        public_payload = _public_eee_projection(path, observations)
        file_result_ids = _validate_public_eee_provenance(
            public_payload,
            paper_id=paper_id,
            exported=exported,
            sources=sources,
            context=f"EEE artifact {path.name}",
        )
        duplicate_ids = result_ids & file_result_ids
        if duplicate_ids:
            raise ExtractionReviewCardError("EEE result IDs must be unique across paper files")
        result_ids.update(file_result_ids)
        links.append(
            {
                "name": path.name,
                "href": _relative_href(f"{prefix}/{paper_root.name}/{path.name}", "EEE href"),
                "sha256": sha256_bytes(canonical_json_bytes(public_payload)),
            }
        )
    return links, result_ids


def _copy_card_eee_artifacts(
    card: Mapping[str, Any],
    *,
    paper_root: Path,
    card_dir: Path,
    output_root: Path,
) -> None:
    """Copy only card-listed, privacy-audited EEE JSON files into the bundle."""

    eee = _mapping(card.get("eee"), "card EEE")
    links = _sequence(eee.get("links", []), "card EEE links")
    observations = _read_jsonl(paper_root / "observations.jsonl")
    output_boundary = output_root.resolve()
    for index, raw_link in enumerate(links):
        link = _mapping(raw_link, f"card EEE link {index}")
        name = link.get("name")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".json"):
            raise ExtractionReviewCardError("EEE artifact name must be one safe JSON filename")
        source = paper_root / "eee" / name
        if not source.is_file() or source.is_symlink():
            raise ExtractionReviewCardError("card-listed EEE source is not a regular file")
        expected_sha256 = _sha256(link.get("sha256"), "card-listed EEE SHA-256")
        payload = _public_eee_projection(source, observations)
        public_bytes = canonical_json_bytes(payload)
        if sha256_bytes(public_bytes) != expected_sha256:
            raise ExtractionReviewCardError("card-listed public EEE projection changed")
        href = _relative_href(str(link.get("href")), "EEE href")
        destination = (card_dir / Path(*PurePosixPath(href).parts)).resolve()
        try:
            destination.relative_to(output_boundary)
        except ValueError as error:
            raise ExtractionReviewCardError("EEE href escapes the public bundle") from error
        if destination.exists():
            raise ExtractionReviewCardError("two card links resolve to the same EEE artifact")
        atomic_write_bytes(destination, public_bytes)
        if sha256_file(destination) != expected_sha256:
            raise ExtractionReviewCardError("copied EEE artifact failed hash verification")


def build_paper_extraction_review_card(
    paper_root: Path,
    *,
    split: Split,
    evaluation_status: EvaluationStatus | None = None,
    aggregate_evaluation: Mapping[str, Any] | None = None,
    aggregate_review: Mapping[str, Any] | None = None,
    paper_review: Mapping[str, Any] | None = None,
    eee_link_prefix: str = "../../eee",
    known_limitations: Sequence[str] = (),
    correct_abstention: bool = False,
) -> dict[str, Any]:
    """Project private paper-run artifacts into one strict public card."""

    if split not in {"development", "holdout"}:
        raise ExtractionReviewCardError("split must be development or holdout")
    resolved_evaluation_status = _resolve_evaluation_status(split, evaluation_status)
    run_path = paper_root / "run.json"
    manifest_path = paper_root / "source-manifest.json"
    observations_path = paper_root / "observations.jsonl"
    for path in (run_path, manifest_path):
        if not path.is_file() or path.is_symlink():
            raise ExtractionReviewCardError(f"missing regular artifact: {path.name}")
    run = _mapping(read_json(run_path), "run.json")
    manifest = _mapping(read_json(manifest_path), "source-manifest.json")
    observations = _read_jsonl(observations_path)
    paper_id = _safe_id(run.get("paper_id"), "run paper_id")
    if manifest.get("paper_id") != paper_id or paper_root.name != paper_id:
        raise ExtractionReviewCardError("paper IDs differ across run root and source manifest")
    title = _safe_text(run.get("title"), "paper title")
    try:
        manifest_model = SourceManifest.model_validate(manifest)
    except ValueError as error:
        raise ExtractionReviewCardError("source-manifest.json failed typed validation") from error
    if manifest_model.schema_version != "source-manifest/0.2":
        raise ExtractionReviewCardError("unsupported source manifest schema version")
    if manifest_model.title != title:
        raise ExtractionReviewCardError("paper titles differ across run and source manifest")
    public_url, sources = _source_projection(manifest)
    _, exported_candidates = _validate_observation_bindings(
        observations,
        paper_id=paper_id,
        sources=sources,
    )
    counts = _mapping(run.get("counts", {}), "run counts")
    candidate_count = _nonnegative_int(counts.get("candidates"), "candidate count")
    if candidate_count != len(observations):
        raise ExtractionReviewCardError("run candidate count does not match observations.jsonl")
    exported_count = _nonnegative_int(counts.get("exported"), "exported count")
    if exported_count != len(exported_candidates):
        raise ExtractionReviewCardError(
            "run exported count does not match export-eligible observations"
        )
    _, schema_authority = _pinned_eee_schema()
    eee_schema = _mapping(run.get("eee_schema", {}), "EEE schema")
    if eee_schema.get("version") != schema_authority.version:
        raise ExtractionReviewCardError("paper run does not declare the pinned EEE schema version")
    if eee_schema.get("sha256") != schema_authority.sha256:
        raise ExtractionReviewCardError("paper run does not declare the pinned EEE schema hash")
    configuration = _configuration_projection(run)
    configuration_sha256 = sha256_bytes(canonical_json_bytes(configuration))
    extractor = _mapping(run.get("extractor", {}), "run.extractor")
    verifier = _mapping(run.get("verifier", {}), "run.verifier")
    code = _mapping(run.get("code", {}), "run.code")
    selected_pages = sorted(
        {
            _nonnegative_int(page, "selected page")
            for page in _sequence(run.get("selected_pages", []), "selected pages")
        }
    )
    if any(page < 1 for page in selected_pages):
        raise ExtractionReviewCardError("selected pages must be positive")
    selected_blocks = _sequence(run.get("selected_blocks", []), "selected blocks")
    execution = _mapping(extractor.get("execution", {}), "extractor execution")
    eee_count = _nonnegative_int(counts.get("eee_records"), "EEE record count")
    schema_issues = _nonnegative_int(counts.get("eee_schema_issues"), "EEE schema issues")
    eee_links, eee_result_ids = _eee_links(
        paper_root,
        href_prefix=eee_link_prefix,
        paper_id=paper_id,
        exported=exported_candidates,
        sources=sources,
    )
    if eee_count != len(eee_links):
        raise ExtractionReviewCardError(
            "reported EEE record count does not match discovered EEE artifact files"
        )
    if eee_result_ids != set(exported_candidates):
        raise ExtractionReviewCardError(
            "public EEE results do not match the exported observation population"
        )
    run_review_state = _run_review_state_projection(
        run,
        candidate_count=candidate_count,
        eee_count=eee_count,
        schema_issues=schema_issues,
    )
    reason_counts = _reason_counts(
        observations,
        eee_count=eee_count,
        schema_issues=schema_issues,
        correct_abstention=correct_abstention,
        paper_run_error=run.get("status") in {"error", "partial_failure"},
    )
    reference = _reference_projection(run)
    limitations = _automatic_limitations(run, reference, reason_counts, eee_count)
    limitations.extend(_safe_text(item, "known limitation") for item in known_limitations)
    if resolved_evaluation_status == "post_hoc_corrected":
        limitations.append(POST_HOC_CORRECTED_HOLDOUT_LIMITATION)
    limitations = sorted(set(limitations))
    actual_source_manifest_sha256 = sha256_file(manifest_path)
    source_manifest_sha256 = run.get("source_manifest_sha256")
    if source_manifest_sha256 is None:
        source_manifest_sha256 = actual_source_manifest_sha256
    source_manifest_sha256 = _sha256(source_manifest_sha256, "source manifest SHA-256")
    if source_manifest_sha256 != actual_source_manifest_sha256:
        raise ExtractionReviewCardError(
            "run source manifest SHA-256 does not match source-manifest.json"
        )
    prompt_sha256 = extractor.get("prompt_sha256")
    if prompt_sha256 is not None:
        prompt_sha256 = _sha256(prompt_sha256, "extractor prompt SHA-256")
    request_contract = _mapping(extractor.get("request_contract", {}), "extractor contract")
    request_contract_sha256 = sha256_bytes(canonical_json_bytes(request_contract))
    eee_schema_sha256 = eee_schema.get("sha256")
    if eee_schema_sha256 is not None:
        eee_schema_sha256 = _sha256(eee_schema_sha256, "EEE schema SHA-256")
    aggregate_review_projection = _aggregate_review_projection(aggregate_review)
    paper_review_projection = _paper_review_projection(paper_review)
    if (
        aggregate_review_projection is not None
        and paper_review_projection is not None
        and aggregate_review_projection["audit_id"] != paper_review_projection["audit_id"]
    ):
        raise ExtractionReviewCardError(
            "paper review coverage and aggregate decisions come from different audits"
        )
    public_review = {
        "label": "Analyst review",
        "independent_human_validation": False,
        "aggregate": aggregate_review_projection,
        "paper": paper_review_projection,
    }
    if public_review["aggregate"] is None and public_review["paper"] is None:
        public_review["status"] = "not_available"
    else:
        public_review["status"] = "available"
    card: dict[str, Any] = {
        "schema_version": CARD_SCHEMA_VERSION,
        "statement": CARD_STATEMENT,
        "split": split,
        "evaluation_status": resolved_evaluation_status,
        "paper": {
            "paper_id": paper_id,
            "title": title,
            "public_url": public_url,
        },
        "sources": sources,
        "source_manifest_sha256": source_manifest_sha256,
        "pipeline": {
            "git_commit": code.get("git_commit")
            if isinstance(code.get("git_commit"), str)
            else None,
            "source_tree_sha256": (
                _sha256(code.get("source_tree_sha256"), "source tree SHA-256")
                if code.get("source_tree_sha256") is not None
                else None
            ),
            "configuration_sha256": configuration_sha256,
            "extractor_prompt_sha256": prompt_sha256,
            "extractor_request_contract_sha256": request_contract_sha256,
            "eee_schema_version": eee_schema.get("version"),
            "eee_schema_sha256": eee_schema_sha256,
        },
        "models": {
            "extractor": _safe_text(extractor.get("model"), "extractor model"),
            "provider": _safe_text(extractor.get("provider"), "extractor provider"),
            "verifier_enabled": verifier.get("enabled") is True,
            "verifier_model": verifier.get("model")
            if isinstance(verifier.get("model"), str)
            else None,
        },
        "processing": {
            "status": _safe_text(run.get("status"), "run status"),
            "review_state": run_review_state,
            "selected_pages": selected_pages,
            "selected_page_count": len(selected_pages),
            "selected_block_count": len(selected_blocks),
            "blocks_total": _nonnegative_int(
                execution.get("blocks_total"), "blocks total", default=len(selected_blocks)
            ),
            "blocks_succeeded": _nonnegative_int(
                execution.get("blocks_succeeded"), "blocks succeeded"
            ),
            "blocks_failed": _nonnegative_int(execution.get("blocks_failed"), "blocks failed"),
            "blocks_resumed": _nonnegative_int(execution.get("blocks_resumed"), "blocks resumed"),
        },
        "counts": {
            "candidates": candidate_count,
            "exported_observations": _nonnegative_int(counts.get("exported"), "exports"),
            "eee_records": eee_count,
            "eee_files": len(eee_links),
            "eee_schema_issues": schema_issues,
        },
        "eee": {
            "links": eee_links,
            "reported_count_matches_files": eee_count == len(eee_links),
        },
        "reference_evaluation": reference,
        "quality_gates": _quality_gate_projection(aggregate_evaluation),
        "review": public_review,
        "abstention": {
            "applies": eee_count == 0,
            "classification": (
                "confirmed_correct"
                if correct_abstention
                else "unresolved"
                if eee_count == 0
                else "not_applicable"
            ),
            "primary_reason": _primary_abstention_reason(reason_counts) if eee_count == 0 else None,
            "reason_counts": reason_counts,
        },
        "provenance": _provenance_projection(observations),
        "known_limitations": limitations,
        "privacy": {
            "source_pdfs_included": False,
            "evidence_quotations_included": False,
            "provider_traces_included": False,
            "credentials_included": False,
            "request_ids_included": False,
            "local_paths_included": False,
            "private_annotations_included": False,
        },
    }
    _assert_public_payload(card)
    _assert_no_evidence_text(card, observations)
    return card


def _format_rate(value: Any) -> str:
    if value is None:
        return "Not measured"
    return f"{float(value) * 100:.1f}%"


def _format_cost_lower_bound(value: Any) -> str:
    if value is None:
        return "Not recorded"
    return f"USD {float(value):.6f}+"


def _render_links(links: Sequence[Mapping[str, Any]]) -> str:
    if not links:
        return '<span class="muted">No EEE files</span>'
    return " ".join(
        f'<a href="{escape(str(link["href"]), quote=True)}">{escape(str(link["name"]))}</a>'
        for link in links
    )


def render_paper_extraction_review_card(card: Mapping[str, Any]) -> str:
    """Render a compact standalone HTML card from a sanitized card mapping."""

    _assert_public_payload(card)
    if card.get("schema_version") != CARD_SCHEMA_VERSION or card.get("statement") != CARD_STATEMENT:
        raise ExtractionReviewCardError("unsupported Extraction Review Card payload")
    paper = _mapping(card.get("paper"), "card.paper")
    processing = _mapping(card.get("processing"), "card.processing")
    counts = _mapping(card.get("counts"), "card.counts")
    reference = _mapping(card.get("reference_evaluation"), "card.reference_evaluation")
    detection = (
        reference.get("detection") if isinstance(reference.get("detection"), Mapping) else {}
    )
    abstention = _mapping(card.get("abstention"), "card.abstention")
    provenance = _mapping(card.get("provenance"), "card.provenance")
    eee = _mapping(card.get("eee"), "card.eee")
    links = _sequence(eee.get("links", []), "card EEE links")
    sources = _sequence(card.get("sources"), "card sources")
    gates = _sequence(card.get("quality_gates", []), "card gates")
    review = _mapping(card.get("review"), "card review")
    paper_review = review.get("paper") if isinstance(review.get("paper"), Mapping) else None
    aggregate_review = (
        review.get("aggregate") if isinstance(review.get("aggregate"), Mapping) else None
    )
    review_text = "Analyst review: not available (not independent human validation)"
    if paper_review is not None:
        review_text = (
            "Included in analyst review; the individual decision is withheld in "
            "public artifacts "
            "(not independent human validation)"
        )
    elif aggregate_review is not None:
        review_text = (
            f"Analyst review split aggregate: {aggregate_review['completed']} decisions; "
            f"{_format_rate(aggregate_review['paper_coverage'])} paper coverage "
            "(not independent human validation)"
        )
    source_rows = "".join(
        "<tr>"
        f"<td>{escape(str(source['role']))}</td>"
        f"<td><code>{escape(str(source['source_id']))}</code></td>"
        f'<td><a href="{escape(str(source["public_url"]), quote=True)}">public source</a></td>'
        f"<td><code>{escape(str(source['sha256'] or source['git_commit']))}</code></td>"
        "</tr>"
        for source in sources
    )
    gate_rows = (
        "".join(
            "<tr>"
            f"<td>{escape(str(gate['name']))}</td>"
            f'<td><span class="badge {escape(str(gate["status"]))}">{escape(str(gate["status"]))}</span></td>'
            f"<td>{escape(_format_rate(gate.get('value')))}</td>"
            f"<td>{escape(_format_rate(gate.get('threshold')))}</td>"
            "</tr>"
            for gate in gates
        )
        or '<tr><td colspan="4" class="muted">No split-level gates supplied.</td></tr>'
    )
    reason_rows = (
        "".join(
            f"<li><span>{escape(name.replace('_', ' '))}</span><strong>{count}</strong></li>"
            for name, count in _mapping(abstention.get("reason_counts"), "reason counts").items()
            if count
        )
        or '<li class="muted">No abstention reasons.</li>'
    )
    limitations = "".join(
        f"<li>{escape(str(item))}</li>"
        for item in _sequence(card.get("known_limitations"), "limitations")
    )
    pages = ", ".join(str(page) for page in processing.get("selected_pages", [])) or "None"
    provenance_pages = ", ".join(str(page) for page in provenance.get("pages", [])) or "None"
    numeric_exported = int(provenance["numeric_exported_observations"])
    provenance_summary = (
        f"{provenance['complete_numeric_observations']}/{numeric_exported} numeric "
        "exported observations have complete source/page/structure/hash provenance. "
        "Evidence quotations are not included."
        if numeric_exported
        else "No numeric exported observations; provenance completeness is not applicable."
    )
    provenance_kinds = (
        ", ".join(
            f"{kind}: {count}"
            for kind, count in _mapping(
                provenance.get("structure_kind_counts", {}), "provenance kinds"
            ).items()
        )
        or "None"
    )
    title = escape(str(paper["title"]))
    paper_id = escape(str(paper["paper_id"]))
    split = escape(str(card["split"]))
    evaluation_status = escape(str(card.get("evaluation_status", card["split"])))
    evaluation_label = split if evaluation_status == split else f"{split} · {evaluation_status}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Extraction Review Card</title><style>
:root{{--ink:#172033;--muted:#667085;--line:#e4e7ec;--paper:#fff;--canvas:#f5f7fb;--accent:#4058d6;--good:#067647;--warn:#b54708;--bad:#b42318}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--canvas);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,sans-serif}}main{{width:min(980px,calc(100% - 32px));margin:28px auto 40px}}header,section{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:12px}}.kicker{{color:var(--accent);font-weight:750;text-transform:uppercase;font-size:11px;letter-spacing:.08em}}h1{{font-size:26px;line-height:1.15;margin:.35rem 0}}h2{{font-size:15px;margin:0 0 10px}}.muted,.meta{{color:var(--muted)}}.statement{{font-weight:750;color:var(--bad)}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.stat{{border:1px solid var(--line);border-radius:9px;padding:11px}}.stat span,.stat strong{{display:block}}.stat span{{color:var(--muted);font-size:11px}}.stat strong{{font-size:20px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{font-size:11px;text-transform:uppercase;color:var(--muted)}}code{{font-size:11px;word-break:break-all}}a{{color:var(--accent)}}ul{{margin:0;padding-left:20px}}.reasons{{list-style:none;padding:0;display:grid;grid-template-columns:repeat(2,1fr);gap:5px 18px}}.reasons li{{display:flex;justify-content:space-between}}.badge{{border-radius:999px;padding:2px 7px;font-size:11px;font-weight:700}}.passed{{background:#ecfdf3;color:var(--good)}}.failed{{background:#fef3f2;color:var(--bad)}}.not_measured{{background:#f2f4f7;color:var(--muted)}}@media(max-width:720px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}@media print{{body{{background:#fff}}main{{width:100%;margin:0}}}}
</style></head><body><main>
<header><div class="kicker">{evaluation_label} · {paper_id}</div><h1>{title}</h1>
<div class="statement">{CARD_STATEMENT}</div><div class="meta">Status: {escape(str(processing["status"]))} · Selected pages: {escape(pages)}</div></header>
<section><div class="grid">
<div class="stat"><span>Candidates</span><strong>{counts["candidates"]}</strong></div>
<div class="stat"><span>Exported observations</span><strong>{counts["exported_observations"]}</strong></div>
<div class="stat"><span>EEE records</span><strong>{counts["eee_records"]}</strong></div>
<div class="stat"><span>Selected blocks</span><strong>{processing["selected_block_count"]}</strong></div>
</div></section>
<section><h2>EEE output</h2>{_render_links(links)}</section>
<section><h2>Reference coverage and QA</h2><p>Recall {_format_rate(detection.get("recall"))} · Covered precision {_format_rate(detection.get("precision"))} · F1 {_format_rate(detection.get("f1"))}</p><p>{escape(review_text)}</p>
<table><thead><tr><th>Split gate</th><th>Status</th><th>Value</th><th>Threshold</th></tr></thead><tbody>{gate_rows}</tbody></table></section>
<section><h2>Abstention and unresolved reasons</h2><p>Applies: {str(abstention["applies"]).lower()} · Classification: {escape(str(abstention["classification"]))} · Primary reason: {escape(str(abstention.get("primary_reason") or "none"))}</p><ul class="reasons">{reason_rows}</ul></section>
<section><h2>Quote-free provenance</h2><p>{provenance_summary}</p>
<p>Anchors: {provenance["anchor_count"]} · Pages: {escape(provenance_pages)} · Structures: {escape(provenance_kinds)} · Evidence hashes present: {str(provenance["quote_hashes_present"]).lower()}</p>
<table><thead><tr><th>Role</th><th>Source ID</th><th>URL</th><th>SHA-256 / Git commit</th></tr></thead><tbody>{source_rows}</tbody></table></section>
<section><h2>Known limitations</h2><ul>{limitations}</ul></section>
<footer class="muted">Card schema {CARD_SCHEMA_VERSION}. Self-contained HTML; no source PDF, evidence quotation, provider trace, request ID, credential, local path, or private annotation is embedded.</footer>
</main></body></html>
"""


def _split_summary(
    corpus_run: Mapping[str, Any],
    cards: Sequence[Mapping[str, Any]],
    *,
    evaluation_status: EvaluationStatus,
    aggregate_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation = aggregate_evaluation
    if not isinstance(evaluation, Mapping):
        raw_evaluation = corpus_run.get("reference_evaluation")
        evaluation = raw_evaluation if isinstance(raw_evaluation, Mapping) else {}
    detection = (
        evaluation.get("detection") if isinstance(evaluation.get("detection"), Mapping) else {}
    )
    fields = (
        evaluation.get("field_accuracy")
        if isinstance(evaluation.get("field_accuracy"), Mapping)
        else {}
    )
    derived = (
        evaluation.get("derived_accuracy")
        if isinstance(evaluation.get("derived_accuracy"), Mapping)
        else {}
    )
    bases = evaluation.get("bases") if isinstance(evaluation.get("bases"), Mapping) else {}
    claim_type = (
        evaluation.get("claim_type_classification")
        if isinstance(evaluation.get("claim_type_classification"), Mapping)
        else {}
    )
    safety = (
        evaluation.get("negative_control_safety")
        if isinstance(evaluation.get("negative_control_safety"), Mapping)
        else {}
    )
    operations = (
        corpus_run.get("operations") if isinstance(corpus_run.get("operations"), Mapping) else {}
    )
    extractor_operations = (
        operations.get("extractor") if isinstance(operations.get("extractor"), Mapping) else {}
    )
    card_counts = [_mapping(card.get("counts"), "card counts") for card in cards]
    card_processing = [_mapping(card.get("processing"), "card processing") for card in cards]
    card_provenance = [_mapping(card.get("provenance"), "card provenance") for card in cards]
    reason_counts: Counter[str] = Counter()
    for card in cards:
        abstention = _mapping(card.get("abstention"), "card abstention")
        for name, value in _mapping(
            abstention.get("reason_counts"), "card abstention reasons"
        ).items():
            reason_counts[str(name)] += _nonnegative_int(value, f"abstention reason {name}")
    quality_gates = _quality_gate_projection(evaluation)
    gate_status_counts = Counter(str(gate["status"]) for gate in quality_gates)
    eee_records = sum(
        _nonnegative_int(item.get("eee_records"), "EEE records") for item in card_counts
    )
    eee_files = sum(_nonnegative_int(item.get("eee_files"), "EEE files") for item in card_counts)
    schema_issues = sum(
        _nonnegative_int(item.get("eee_schema_issues"), "EEE schema issues") for item in card_counts
    )
    numeric_exported = sum(
        _nonnegative_int(item.get("numeric_exported_observations"), "numeric exported")
        for item in card_provenance
    )
    numeric_complete = sum(
        _nonnegative_int(item.get("complete_numeric_observations"), "numeric provenance")
        for item in card_provenance
    )
    return {
        "evaluation_status": evaluation_status,
        "papers": len(cards),
        "papers_successful": sum(item.get("status") == "success" for item in card_processing),
        "papers_technically_completed": sum(
            item.get("status") in {"success", "quality_failure"} for item in card_processing
        ),
        "papers_with_eee": sum(
            _nonnegative_int(item.get("eee_records"), "EEE records") > 0 for item in card_counts
        ),
        "candidates": sum(
            _nonnegative_int(item.get("candidates"), "candidates") for item in card_counts
        ),
        "exported_observations": sum(
            _nonnegative_int(item.get("exported_observations"), "exported observations")
            for item in card_counts
        ),
        "eee_records": eee_records,
        "eee_files": eee_files,
        "eee_schema_issues": schema_issues,
        "eee_schema_validity": {
            "valid_files": eee_files,
            "files": eee_files,
            "all_valid": eee_records == eee_files,
            "all_compositions_valid": schema_issues == 0,
        },
        "numeric_provenance": {
            "complete_observations": numeric_complete,
            "numeric_exported_observations": numeric_exported,
            "all_complete": numeric_complete == numeric_exported,
        },
        "reference_metrics": {
            "micro_recall": _finite_rate(detection.get("recall"), "split micro recall"),
            "macro_recall": _finite_rate(detection.get("macro_recall"), "split macro recall"),
            "covered_precision": _finite_rate(detection.get("precision"), "split precision"),
            "f1": _finite_rate(detection.get("f1"), "split F1"),
            "exact_numeric_value_and_unit": _finite_rate(
                derived.get("exact_numeric_value_and_unit"), "split value plus unit"
            ),
            "joint_semantics": _finite_rate(fields.get("joint_semantics"), "split joint semantics"),
            "page_and_text_support": _finite_rate(
                derived.get("evidence_page_and_text_support"), "split page and text support"
            ),
            "evidence_structure": _finite_rate(
                fields.get("evidence_structure"), "split evidence structure"
            ),
            "honest_missingness": _finite_rate(fields.get("missingness"), "split missingness"),
            "claim_type_macro_f1": _finite_rate(
                claim_type.get("macro_f1"), "split claim-type Macro-F1"
            ),
        },
        "reference_bases": {
            str(name): _nonnegative_int(value, f"reference basis {name}")
            for name, value in sorted(bases.items())
        },
        "negative_control_safety": {
            "measurement_status": str(safety.get("measurement_status", "not_measured")),
            "controls_total": _nonnegative_int(safety.get("controls_total"), "controls total"),
            "controls_matched": _nonnegative_int(
                safety.get("controls_matched"), "controls matched"
            ),
            "control_match_coverage": _finite_rate(
                safety.get("control_match_coverage"), "control match coverage"
            ),
            "false_primary_candidates": _nonnegative_int(
                safety.get("false_primary_count"), "false-primary candidates"
            ),
            "false_primary_exports": _nonnegative_int(
                safety.get("false_primary_export_count"), "false-primary exports"
            ),
        },
        "abstention_reason_counts": {name: reason_counts.get(name, 0) for name in _REASON_NAMES},
        "operations": {
            "status": "measured" if extractor_operations else "not_recorded",
            "wall_clock_seconds": _optional_nonnegative_float(
                operations.get("wall_clock_seconds"), "split wall clock"
            ),
            "successful_calls": _nonnegative_int(
                extractor_operations.get("calls"), "successful calls"
            ),
            "cost_usd_lower_bound": _optional_nonnegative_float(
                extractor_operations.get("cost_usd_lower_bound"), "split cost"
            ),
            "input_tokens_lower_bound": _nonnegative_int(
                extractor_operations.get("input_tokens_lower_bound"), "input tokens"
            ),
            "output_tokens_lower_bound": _nonnegative_int(
                extractor_operations.get("output_tokens_lower_bound"), "output tokens"
            ),
            "total_tokens_lower_bound": _nonnegative_int(
                extractor_operations.get("total_tokens_lower_bound"), "total tokens"
            ),
            "latency_seconds_total": _optional_nonnegative_float(
                extractor_operations.get("latency_seconds_total"), "split latency"
            ),
            "retries_lower_bound": _nonnegative_int(
                extractor_operations.get("retries_lower_bound"), "split retries"
            ),
            "blocks_total": _nonnegative_int(
                extractor_operations.get("blocks_total"), "split blocks"
            ),
            "blocks_resumed": _nonnegative_int(
                extractor_operations.get("blocks_resumed"), "split resumed blocks"
            ),
            "blocks_failed": _nonnegative_int(
                extractor_operations.get("blocks_failed"), "split failed blocks"
            ),
        },
        "quality_gates": quality_gates,
        "quality_gate_status_counts": {
            status: gate_status_counts.get(status, 0)
            for status in ("passed", "failed", "not_measured")
        },
    }


def _comparison(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    development = summaries.get("development")
    holdout = summaries.get("holdout")
    if development is None or holdout is None:
        return {"status": "not_available", "reason": "Both splits are required for comparison."}
    dev_metrics = _mapping(development.get("reference_metrics"), "development metrics")
    holdout_metrics = _mapping(holdout.get("reference_metrics"), "holdout metrics")
    deltas: dict[str, float | None] = {}
    for name in dev_metrics:
        left, right = dev_metrics[name], holdout_metrics.get(name)
        deltas[name] = (
            round(float(right) - float(left), 6) if left is not None and right is not None else None
        )
    return {
        "status": "available",
        "direction": "holdout_minus_development",
        "development": development,
        "holdout": holdout,
        "reference_metric_deltas": deltas,
    }


def build_extraction_review_index(
    card_entries: Sequence[Mapping[str, Any]],
    *,
    split_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic corpus index for any number of paper cards."""

    ordered = sorted(card_entries, key=lambda item: (str(item["split"]), str(item["paper_id"])))
    split_counts = Counter(str(item["split"]) for item in ordered)
    index = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "statement": CARD_STATEMENT,
        "card_count": len(ordered),
        "split_counts": {
            "development": split_counts.get("development", 0),
            "holdout": split_counts.get("holdout", 0),
        },
        "cards": ordered,
        "split_summaries": {name: split_summaries[name] for name in sorted(split_summaries)},
        "development_holdout_comparison": _comparison(split_summaries),
        "privacy": {
            "evidence_quotations_included": False,
            "provider_traces_included": False,
            "credentials_included": False,
            "local_paths_included": False,
            "private_annotations_included": False,
        },
    }
    _assert_public_payload(index, "index")
    return index


def render_extraction_review_index(index: Mapping[str, Any]) -> str:
    """Render a compact standalone HTML index with links to every card."""

    _assert_public_payload(index, "index")
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ExtractionReviewCardError("unsupported Extraction Review Card index")
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(item['split']))}"
        + (
            ""
            if item.get("evaluation_status", item["split"]) == item["split"]
            else f"<br><strong>{escape(str(item['evaluation_status']))}</strong>"
        )
        + "</td>"
        f"<td><strong>{escape(str(item['title']))}</strong><br><code>{escape(str(item['paper_id']))}</code></td>"
        f"<td>{escape(str(item['status']))}</td>"
        f"<td>{item['candidates']}</td><td>{item['exported_observations']}</td><td>{item['eee_records']}</td>"
        f"<td>{escape(str(item['review_outcome'] or 'not available'))}</td>"
        f"<td>{escape(str(item['abstention_reason'] or 'none'))}</td>"
        f'<td><a href="{escape(str(item["html_href"]), quote=True)}">HTML</a> · <a href="{escape(str(item["json_href"]), quote=True)}">JSON</a></td>'
        "</tr>"
        for item in _sequence(index.get("cards"), "index cards")
    )
    summaries = _mapping(index.get("split_summaries"), "split summaries")
    summary_cards = "".join(
        f'<div class="summary"><h2>{escape(name)}</h2><strong>Evaluation status: {escape(str(value.get("evaluation_status", name)))}</strong><br><b>{value["papers"]}</b> papers · <b>{value["papers_with_eee"]}</b> with EEE · <b>{value["eee_records"]}</b> EEE records<br>Micro recall {_format_rate(value["reference_metrics"]["micro_recall"])} · Macro recall {_format_rate(value["reference_metrics"]["macro_recall"])} · Covered precision {_format_rate(value["reference_metrics"]["covered_precision"])}<br>Schema-valid EEE {value["eee_schema_validity"]["valid_files"]}/{value["eee_schema_validity"]["files"]} · Complete numeric provenance {value["numeric_provenance"]["complete_observations"]}/{value["numeric_provenance"]["numeric_exported_observations"]} · Cost {_format_cost_lower_bound(value["operations"]["cost_usd_lower_bound"])}<br>Gates: {value["quality_gate_status_counts"]["passed"]} passed · {value["quality_gate_status_counts"]["failed"]} failed · {value["quality_gate_status_counts"]["not_measured"]} unmeasured</div>'
        for name, value in sorted(summaries.items())
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Paper Extraction Review Cards</title><style>
body{{margin:0;background:#f5f7fb;color:#172033;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}}main{{width:min(1400px,calc(100% - 32px));margin:30px auto}}header,.summary,.table{{background:#fff;border:1px solid #e4e7ec;border-radius:12px;padding:17px;margin-bottom:12px}}h1{{margin:4px 0}}.statement{{color:#b42318;font-weight:750}}.summaries{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.summary h2{{text-transform:capitalize;margin:0 0 8px}}.table{{overflow:auto;padding:0}}table{{width:100%;border-collapse:collapse;min-width:1050px}}th,td{{padding:10px;border-bottom:1px solid #e4e7ec;text-align:left}}th{{font-size:11px;text-transform:uppercase;color:#667085}}code{{font-size:11px}}a{{color:#4058d6}}@media(max-width:720px){{.summaries{{grid-template-columns:1fr}}}}@media print{{body{{background:#fff}}main{{width:100%;margin:0}}}}
</style></head><body><main><header><div>Proceedings → EEE audit</div><h1>{index["card_count"]} Paper Extraction Review Cards</h1><div class="statement">{CARD_STATEMENT}</div></header><div class="summaries">{summary_cards}</div><div class="table"><table><thead><tr><th>Split</th><th>Paper</th><th>Status</th><th>Candidates</th><th>Exported</th><th>EEE</th><th>Review</th><th>Abstention</th><th>Card</th></tr></thead><tbody>{rows}</tbody></table></div></main></body></html>"""


def _write_extraction_review_bundle(
    corpora: Sequence[CorpusCardInput],
    output_root: Path,
) -> Path:
    """Write a bundle into an isolated, empty staging directory."""

    if not corpora:
        raise ExtractionReviewCardError("at least one corpus input is required")
    if len({corpus.split for corpus in corpora}) != len(corpora):
        raise ExtractionReviewCardError("each split may be supplied only once")
    output_root.mkdir(parents=True, exist_ok=True)
    card_entries: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    for corpus in sorted(corpora, key=lambda item: item.split):
        evaluation_status = _resolve_evaluation_status(
            corpus.split,
            corpus.evaluation_status,
        )
        corpus_run_path = corpus.run_root / "corpus-run.json"
        if not corpus_run_path.is_file() or corpus_run_path.is_symlink():
            raise ExtractionReviewCardError("each run root requires corpus-run.json")
        corpus_run = _mapping(read_json(corpus_run_path), "corpus-run.json")
        aggregate_evaluation = corpus_run.get("reference_evaluation")
        if not isinstance(aggregate_evaluation, Mapping):
            evaluation_path = corpus.run_root / "corpus-evaluation.json"
            aggregate_evaluation = (
                _mapping(read_json(evaluation_path), "corpus-evaluation.json")
                if evaluation_path.is_file()
                else None
            )
        aggregate_review = (
            _mapping(read_json(corpus.human_review_summary_path), "human-review summary")
            if corpus.human_review_summary_path is not None
            else None
        )
        raw_runs = _sequence(corpus_run.get("runs"), "corpus runs")
        _validate_review_population(aggregate_review, raw_runs)
        cards: list[dict[str, Any]] = []
        for raw_run in sorted(
            raw_runs, key=lambda item: str(_mapping(item, "paper run").get("paper_id"))
        ):
            run = _mapping(raw_run, "paper run")
            paper_id = _safe_id(run.get("paper_id"), "paper run ID")
            paper_root = corpus.run_root / paper_id
            limitations = tuple(corpus.known_limitations) + tuple(
                corpus.paper_limitations.get(paper_id, ())
            )
            card = build_paper_extraction_review_card(
                paper_root,
                split=corpus.split,
                evaluation_status=evaluation_status,
                aggregate_evaluation=aggregate_evaluation,
                aggregate_review=aggregate_review,
                paper_review=corpus.paper_review_outcomes.get(paper_id),
                eee_link_prefix=corpus.eee_link_prefix or f"../../eee/{corpus.split}",
                known_limitations=limitations,
                correct_abstention=paper_id in corpus.correct_abstentions,
            )
            cards.append(card)
            card_dir = output_root / "cards" / corpus.split
            json_path = card_dir / f"{paper_id}.json"
            html_path = card_dir / f"{paper_id}.html"
            write_json(json_path, card)
            atomic_write_bytes(html_path, render_paper_extraction_review_card(card).encode("utf-8"))
            _copy_card_eee_artifacts(
                card,
                paper_root=paper_root,
                card_dir=card_dir,
                output_root=output_root,
            )
            paper_review = _mapping(card.get("review"), "card review").get("paper")
            card_entries.append(
                {
                    "split": corpus.split,
                    "evaluation_status": evaluation_status,
                    "paper_id": paper_id,
                    "title": _mapping(card["paper"], "card paper")["title"],
                    "status": _mapping(card["processing"], "card processing")["status"],
                    "candidates": _mapping(card["counts"], "card counts")["candidates"],
                    "exported_observations": _mapping(card["counts"], "card counts")[
                        "exported_observations"
                    ],
                    "eee_records": _mapping(card["counts"], "card counts")["eee_records"],
                    "review_outcome": paper_review.get("outcome")
                    if isinstance(paper_review, Mapping)
                    else None,
                    "abstention_reason": _mapping(card["abstention"], "card abstention").get(
                        "primary_reason"
                    ),
                    "json_href": f"cards/{corpus.split}/{paper_id}.json",
                    "html_href": f"cards/{corpus.split}/{paper_id}.html",
                    "json_sha256": sha256_file(json_path),
                    "html_sha256": sha256_file(html_path),
                }
            )
        split_summaries[corpus.split] = _split_summary(
            corpus_run,
            cards,
            evaluation_status=evaluation_status,
            aggregate_evaluation=aggregate_evaluation,
        )
    index = build_extraction_review_index(card_entries, split_summaries=split_summaries)
    write_json(output_root / "extraction-review-index.json", index)
    atomic_write_bytes(
        output_root / "extraction-review-index.html",
        render_extraction_review_index(index).encode("utf-8"),
    )
    files = sorted(
        path for path in output_root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}" for path in files
    ]
    atomic_write_bytes(output_root / "SHA256SUMS", ("\n".join(checksum_lines) + "\n").encode())
    for path in files:
        if path.suffix in {".json", ".html"}:
            text = path.read_text(encoding="utf-8")
            if _LOCAL_PATH.search(text) or any(
                pattern.search(text) for pattern in _SECRET_PATTERNS
            ):
                raise ExtractionReviewCardError(
                    f"public artifact failed privacy audit: {path.name}"
                )
    return output_root


def write_extraction_review_bundle(
    corpora: Sequence[CorpusCardInput],
    output_root: Path,
) -> Path:
    """Atomically write cards, HTML views, an index, and deterministic checksums."""

    if output_root.exists():
        raise FileExistsError(f"Extraction Review Card bundle already exists: {output_root.name}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.review-cards.",
            dir=output_root.parent,
        )
    )
    try:
        _write_extraction_review_bundle(corpora, temporary)
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_root
