"""Deterministic public closure for a human-reviewed development run.

This module is deliberately provider-free.  Construction first verifies the
sealed pre-human preview and the private review/derived join in context, then
publishes only the preview bytes, schema-valid reviewed EEE records, and
quote-free deterministic projections.  The standalone verifier needs no
private inputs.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from html import escape
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes
from proceedings_to_eee.reporting.public_development_preview import (
    PRIVATE_EVIDENCE_SET_SCHEMA_VERSION,
    PRIVATE_EVIDENCE_TEXT_CANONICALIZATION,
    PublicDevelopmentPreviewError,
    assert_no_private_evidence_text,
    collect_private_evidence_texts,
    private_evidence_set_binding,
    verify_public_development_preview,
)
from proceedings_to_eee.reporting.public_development_summary import (
    PUBLIC_DEVELOPMENT_SUMMARY_SCHEMA_VERSION,
    PublicDevelopmentSummaryError,
    build_public_development_summary,
)
from proceedings_to_eee.resources import (
    DEFAULT_EEE_SCHEMA_PATH,
    EEE_SCHEMA_SHA256,
    EEE_SCHEMA_VERSION,
)
from proceedings_to_eee.reviewed_export.models import DerivedRunManifest
from proceedings_to_eee.reviewed_export.workflow import (
    DERIVED_MANIFEST_NAME,
    ReviewedExportError,
    verify_contextual_derived_run,
)
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record

PUBLIC_REVIEWED_DEVELOPMENT_BUNDLE_SCHEMA_VERSION = "public-reviewed-development-bundle/0.1"
PUBLIC_REVIEWED_DEVELOPMENT_VERIFICATION_SCHEMA_VERSION = (
    "public-reviewed-development-bundle-verification/0.1"
)
PUBLIC_REVIEWED_CANONICAL_RESULTS_SCHEMA_VERSION = "public-reviewed-canonical-results/0.1"

_PREVIEW_FILES = {
    "README.md",
    "SHA256SUMS",
    "corpus.json",
    "evidence-map.html",
    "evidence-map.json",
    "publication-manifest.json",
    "route-selection.json",
    "run-summary.json",
    "sources.json",
    "usage.json",
    "verification.json",
}
_CONTROL_FILES = {
    "README.md",
    "publication-manifest.json",
    "verification.json",
    "SHA256SUMS",
}
_REVIEWED_PROJECTION_FILES = {
    "reviewed/derived-manifest.json",
    "reviewed/public-summary.json",
    "reviewed/canonical-results.json",
    "reviewed/canonical-results.html",
}
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SAFE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_PATH = re.compile(
    r"(?:^|[\s\"'=(`])(?:"
    r"/(?!/)[A-Za-z0-9._~+%-]+(?:/[A-Za-z0-9._~+%-]+)*"
    r"|//[A-Za-z0-9._-]+(?:[/\\][A-Za-z0-9._~+%-]+)+"
    r"|~[/\\][^\s\"']+"
    r"|[A-Za-z]:[\\/](?:[^\s\"']+)"
    r"|file://)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~-]{12,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|authorization)\s*[:=]\s*[\"'][^\"']{8,}[\"']", re.I),
    re.compile(r"\bOPENROUTER_API_KEY\b"),
)
_FORBIDDEN_KEYS = {
    "annotation",
    "annotations",
    "answer_key",
    "answer_keys",
    "adjudicator_id",
    "adjudicator_identity",
    "api_key",
    "authorization",
    "cookie",
    "credentials",
    "evaluation_score",
    "evaluation_scores",
    "excerpt",
    "excerpts",
    "exact_excerpt",
    "exact_quote",
    "holdout_reference",
    "holdout_references",
    "human_label",
    "human_labels",
    "matched_text",
    "messages",
    "notes",
    "prompt",
    "prompt_template",
    "private_annotations",
    "provider_request",
    "provider_response",
    "quotation",
    "quotations",
    "quote",
    "quotes",
    "raw_completion",
    "raw_payload",
    "raw_request",
    "raw_response",
    "request_id",
    "reviewer_id",
    "reviewer_ids",
    "reviewer_identity",
    "reviewer_notes",
    "secret",
    "source_layout",
    "source_layout_text",
    "system_prompt",
    "user_prompt",
}
_MAX_FILE_BYTES = 10_000_000
_MAX_TOTAL_BYTES = 100_000_000
_DERIVED_COUNT_KEYS = {
    "review_items",
    "decisions_completed",
    "decisions_pending",
    "outcomes_exported",
    "outcomes_withheld",
    "outcomes_failed",
    "eee_records",
    "eee_observations",
}


class PublicReviewedDevelopmentBundleError(ValueError):
    """A reviewed public bundle cannot be built or verified safely."""


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicReviewedDevelopmentBundleError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise PublicReviewedDevelopmentBundleError(f"{context} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise PublicReviewedDevelopmentBundleError(f"{context} fields disagree")


def _safe_id(value: object, context: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PublicReviewedDevelopmentBundleError(f"{context} is not a safe identifier")
    return value


def _hash(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PublicReviewedDevelopmentBundleError(f"{context} is not a lowercase SHA-256")
    return value


def _count(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicReviewedDevelopmentBundleError(f"{context} must be a non-negative integer")
    return value


def _strict_json(content: bytes, context: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise PublicReviewedDevelopmentBundleError(
                    f"{context} contains a duplicate JSON key"
                )
            output[key] = value
        return output

    def reject_constant(_value: str) -> Any:
        raise PublicReviewedDevelopmentBundleError(f"{context} contains a non-finite JSON number")

    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicReviewedDevelopmentBundleError(f"{context} is not strict UTF-8 JSON") from error


def _relative_path(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicReviewedDevelopmentBundleError(f"{context} is not a relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or value in {".", ""}
        or ".." in path.parts
        or len(path.parts) > 8
        or any(_SAFE_PART.fullmatch(part) is None for part in path.parts)
    ):
        raise PublicReviewedDevelopmentBundleError(f"{context} is not a normalized safe path")
    return value


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path, context: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise PublicReviewedDevelopmentBundleError(f"{context} contains a symbolic link")
    return absolute


def _resolve_existing_directory(path: Path, context: str) -> Path:
    absolute = _reject_symlink_components(path, context)
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise PublicReviewedDevelopmentBundleError(f"{context} does not exist") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise PublicReviewedDevelopmentBundleError(f"{context} is not a regular directory")
    return resolved


def _read_regular(path: Path, context: str) -> bytes:
    _reject_symlink_components(path, context)
    try:
        before = path.lstat()
    except OSError as error:
        raise PublicReviewedDevelopmentBundleError(f"{context} could not be read") from error
    if not stat.S_ISREG(before.st_mode):
        raise PublicReviewedDevelopmentBundleError(f"{context} is not a regular file")
    try:
        content = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise PublicReviewedDevelopmentBundleError(f"{context} could not be read") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(content) != before.st_size:
        raise PublicReviewedDevelopmentBundleError(f"{context} changed while it was read")
    if len(content) > _MAX_FILE_BYTES:
        raise PublicReviewedDevelopmentBundleError(f"{context} exceeds the public file limit")
    return content


def _scan_text(text: str, context: str) -> None:
    if "\x00" in text:
        raise PublicReviewedDevelopmentBundleError(f"{context} contains binary data")
    if _LOCAL_PATH.search(text):
        raise PublicReviewedDevelopmentBundleError(f"{context} contains a local path")
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        raise PublicReviewedDevelopmentBundleError(f"{context} contains credential material")


def _audit_public_value(value: object, context: str = "$") -> None:
    if isinstance(value, Mapping):
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, str):
                raise PublicReviewedDevelopmentBundleError(f"{context} has a non-string key")
            _scan_text(key, f"{context} mapping key")
            normalized_key = key.casefold().replace("-", "_")
            private_quote_variant = (
                any(
                    token in {"quote", "quotes", "quotation", "quotations", "excerpt", "excerpts"}
                    for token in normalized_key.split("_")
                )
                and not normalized_key.endswith(("_sha256", "_included"))
                and not normalized_key.startswith("contains_")
            )
            if (
                normalized_key in _FORBIDDEN_KEYS
                or normalized_key.endswith("_request_id")
                or private_quote_variant
            ):
                raise PublicReviewedDevelopmentBundleError(f"{context} contains a private field")
            _audit_public_value(item, f"{context} mapping value {index}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _audit_public_value(item, f"{context}[{index}]")
    elif isinstance(value, str):
        _scan_text(value, context)
    elif isinstance(value, float) and not math.isfinite(value):
        raise PublicReviewedDevelopmentBundleError(f"{context} contains a non-finite number")


def _artifact(path: str, content: bytes, *, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path,
        "sha256": sha256_bytes(content),
        "size_bytes": len(content),
    }
    if records is not None:
        result["records"] = records
    return result


def _validate_artifact(value: object, context: str, *, records: bool = False) -> dict[str, Any]:
    artifact = dict(_mapping(value, context))
    expected = {"path", "sha256", "size_bytes"} | ({"records"} if records else set())
    _exact_keys(artifact, expected, context)
    artifact["path"] = _relative_path(artifact["path"], f"{context}.path")
    artifact["sha256"] = _hash(artifact["sha256"], f"{context}.sha256")
    artifact["size_bytes"] = _count(artifact["size_bytes"], f"{context}.size_bytes")
    if records:
        artifact["records"] = _count(artifact["records"], f"{context}.records")
    return artifact


def _preview_contents(root: Path) -> dict[str, bytes]:
    names = {path.name for path in root.iterdir()}
    if names != _PREVIEW_FILES:
        raise PublicReviewedDevelopmentBundleError("candidate preview inventory changed")
    return {
        f"candidate-preview/{name}": _read_regular(root / name, "candidate preview file")
        for name in sorted(_PREVIEW_FILES)
    }


def _manifest_eee_entries(
    derived_root: Path,
    manifest: DerivedRunManifest,
) -> list[tuple[str, str, bytes, int]]:
    selected: list[tuple[str, str, bytes, int]] = []
    for artifact in manifest.payload_files:
        source_path = PurePosixPath(artifact.path)
        if len(source_path.parts) != 3 or source_path.parts[1] != "eee":
            continue
        paper_id, _, filename = source_path.parts
        _safe_id(paper_id, "derived EEE paper ID")
        if _SAFE_PART.fullmatch(filename) is None or not filename.endswith(".json"):
            raise PublicReviewedDevelopmentBundleError("derived EEE filename is unsafe")
        bundle_path = f"reviewed/eee/{paper_id}/{filename}"
        content = _read_regular(derived_root / artifact.path, "derived EEE record")
        if len(content) != artifact.size_bytes or sha256_bytes(content) != artifact.sha256:
            raise PublicReviewedDevelopmentBundleError("derived EEE differs from its manifest")
        record = _strict_json(content, "derived EEE record")
        result_count = len(
            _sequence(
                _mapping(record, "derived EEE record").get("evaluation_results"),
                "evaluation_results",
            )
        )
        selected.append((artifact.path, bundle_path, content, result_count))
    selected.sort(key=lambda item: item[1])
    if len({item[1] for item in selected}) != len(selected):
        raise PublicReviewedDevelopmentBundleError("derived EEE paths collide in the public bundle")
    return selected


def _validated_eee_records(
    eee_contents: Mapping[str, bytes],
    eee_files: Sequence[Mapping[str, Any]],
    *,
    review_manifest_sha256: str,
    review_lock_sha256: str,
) -> list[tuple[str, str, Mapping[str, Any]]]:
    try:
        schema, authority = load_schema(DEFAULT_EEE_SCHEMA_PATH, EEE_SCHEMA_SHA256)
    except (OSError, ValueError) as error:
        raise PublicReviewedDevelopmentBundleError("pinned EEE schema is unavailable") from error
    if authority.version != EEE_SCHEMA_VERSION:
        raise PublicReviewedDevelopmentBundleError("pinned EEE schema version changed")
    records: list[tuple[str, str, Mapping[str, Any]]] = []
    for entry in eee_files:
        source_path = _relative_path(entry.get("source_path"), "EEE source path")
        bundle_path = _relative_path(entry.get("bundle_path"), "EEE bundle path")
        source_parts = PurePosixPath(source_path).parts
        bundle_parts = PurePosixPath(bundle_path).parts
        if (
            len(source_parts) != 3
            or source_parts[1] != "eee"
            or len(bundle_parts) != 4
            or bundle_parts[:2] != ("reviewed", "eee")
            or source_parts[0] != bundle_parts[2]
            or source_parts[2] != bundle_parts[3]
        ):
            raise PublicReviewedDevelopmentBundleError("EEE publication path mapping is invalid")
        try:
            content = eee_contents[bundle_path]
        except KeyError as error:
            raise PublicReviewedDevelopmentBundleError(
                "manifest-listed EEE file is missing"
            ) from error
        if sha256_bytes(content) != _hash(entry.get("sha256"), "EEE hash") or len(
            content
        ) != _count(entry.get("size_bytes"), "EEE size"):
            raise PublicReviewedDevelopmentBundleError("published EEE checksum disagrees")
        raw = _strict_json(content, "published EEE record")
        record = _mapping(raw, "published EEE record")
        if content != canonical_json_bytes(record):
            raise PublicReviewedDevelopmentBundleError("published EEE record is not canonical JSON")
        if validate_eee_record(dict(record), schema):
            raise PublicReviewedDevelopmentBundleError("published EEE fails the pinned schema")
        _audit_public_value(record, "published EEE record")
        results = _sequence(record.get("evaluation_results"), "EEE evaluation_results")
        source_details = _mapping(
            _mapping(record.get("source_metadata"), "EEE source metadata").get(
                "additional_details"
            ),
            "EEE source metadata details",
        )
        if (
            source_details.get("paper_id") != source_parts[0]
            or source_details.get("review_manifest_sha256") != review_manifest_sha256
        ):
            raise PublicReviewedDevelopmentBundleError("published EEE review binding disagrees")
        for result in results:
            score_details = _mapping(
                _mapping(result, "EEE evaluation result").get("score_details"),
                "EEE score details",
            )
            details = _mapping(score_details.get("details"), "EEE score provenance details")
            if (
                details.get("paper_id") != source_parts[0]
                or details.get("review_manifest_sha256") != review_manifest_sha256
                or details.get("review_lock_sha256") != review_lock_sha256
            ):
                raise PublicReviewedDevelopmentBundleError("published EEE review binding disagrees")
        if len(results) != _count(entry.get("evaluation_results"), "EEE result count"):
            raise PublicReviewedDevelopmentBundleError("EEE result count disagrees")
        records.append((source_parts[0], bundle_path, record))
    return records


def _canonical_results(
    *,
    source_binding: Mapping[str, Any],
    eee_schema: Mapping[str, Any],
    records: Sequence[tuple[str, str, Mapping[str, Any]]],
    publication_state: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for paper_id, bundle_path, record in records:
        evaluation_id = record.get("evaluation_id")
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise PublicReviewedDevelopmentBundleError("EEE evaluation ID is missing")
        source_metadata = _mapping(record.get("source_metadata"), "EEE source metadata")
        source_details = source_metadata.get("additional_details", {})
        if isinstance(source_details, Mapping) and source_details.get("paper_id") not in {
            None,
            paper_id,
        }:
            raise PublicReviewedDevelopmentBundleError("EEE paper identity disagrees with its path")
        for result in _sequence(record.get("evaluation_results"), "EEE evaluation results"):
            item = _mapping(result, "EEE evaluation result")
            result_id = item.get("evaluation_result_id")
            if not isinstance(result_id, str) or not result_id:
                raise PublicReviewedDevelopmentBundleError(
                    "EEE evaluation result lacks a stable identifier"
                )
            rows.append(
                {
                    "paper_id": paper_id,
                    "evaluation_id": evaluation_id,
                    "evaluation_result_id": result_id,
                    "eee_path": bundle_path,
                    "retrieved_timestamp": record.get("retrieved_timestamp"),
                    "source_metadata": source_metadata,
                    "model_info": _mapping(record.get("model_info"), "EEE model info"),
                    "eval_library": _mapping(record.get("eval_library"), "EEE library"),
                    "evaluation_result": item,
                }
            )
    rows.sort(
        key=lambda item: (
            item["paper_id"],
            item["evaluation_id"],
            item["evaluation_result_id"],
        )
    )
    identities = [
        (item["paper_id"], item["evaluation_id"], item["evaluation_result_id"]) for item in rows
    ]
    if len(identities) != len(set(identities)):
        raise PublicReviewedDevelopmentBundleError("reviewed result identities are not unique")
    result = {
        "schema_version": PUBLIC_REVIEWED_CANONICAL_RESULTS_SCHEMA_VERSION,
        "status": publication_state["status"],
        "development_only": True,
        "independent_validation": False,
        "human_review_complete": publication_state["human_review_complete"],
        "publication_ready": publication_state["publication_ready"],
        "source_run": dict(source_binding),
        "eee_schema": dict(eee_schema),
        "result_count": len(rows),
        "results": rows,
    }
    _audit_public_value(result, "canonical results")
    return result


def _review_publication_state(
    derived: DerivedRunManifest,
    *,
    result_count: int,
) -> dict[str, Any]:
    pending = derived.counts["decisions_pending"]
    if pending and result_count:
        return {
            "status": "verified_human_reviewed_development_with_pending_withheld",
            "classification": "human_reviewed_development_with_pending_withheld",
            "human_review_status": "locked_reviewed_exports_with_pending_withheld",
            "human_review_complete": False,
            "publication_ready": True,
        }
    if pending:
        return {
            "status": "verified_review_incomplete_development",
            "classification": "review_incomplete_development",
            "human_review_status": "locked_with_pending_decisions",
            "human_review_complete": False,
            "publication_ready": False,
        }
    if result_count:
        return {
            "status": "verified_human_reviewed_development",
            "classification": "human_reviewed_development",
            "human_review_status": "locked_and_contextually_verified",
            "human_review_complete": True,
            "publication_ready": True,
        }
    return {
        "status": "verified_reviewed_empty_development",
        "classification": "reviewed_empty_development",
        "human_review_status": "locked_complete_no_exported_results",
        "human_review_complete": True,
        "publication_ready": False,
    }


def _validate_review_count_algebra(
    derived: DerivedRunManifest,
    *,
    result_count: int,
) -> None:
    counts = derived.counts
    if set(counts) != _DERIVED_COUNT_KEYS:
        raise PublicReviewedDevelopmentBundleError("derived count fields disagree")
    if (
        counts["decisions_completed"] + counts["decisions_pending"] != counts["review_items"]
        or counts["outcomes_exported"] + counts["outcomes_withheld"] + counts["outcomes_failed"]
        != counts["review_items"]
        or counts["outcomes_withheld"] < counts["decisions_pending"]
        or counts["outcomes_exported"] != result_count
        or counts["eee_observations"] != result_count
    ):
        raise PublicReviewedDevelopmentBundleError("review and outcome count algebra disagrees")


def _display(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _render_results_html(canonical: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for raw in _sequence(canonical.get("results"), "canonical results"):
        row = _mapping(raw, "canonical result")
        result = _mapping(row.get("evaluation_result"), "canonical evaluation result")
        model = _mapping(row.get("model_info"), "canonical model")
        source_data = _mapping(result.get("source_data"), "canonical source data")
        metric = _mapping(result.get("metric_config"), "canonical metric")
        score = _mapping(result.get("score_details"), "canonical score").get("score")
        cells = (
            row.get("paper_id"),
            model.get("name") or model.get("id"),
            source_data.get("dataset_name") or source_data.get("source_type"),
            result.get("evaluation_name"),
            metric.get("metric_name") or metric.get("metric_id"),
            score,
            metric.get("metric_unit"),
            row.get("evaluation_result_id"),
        )
        first, *remaining = cells
        rows.append(
            '      <tr><th scope="row">'
            + escape(_display(first))
            + "</th>"
            + "".join(f"<td>{escape(_display(cell))}</td>" for cell in remaining)
            + "</tr>"
        )
    empty = (
        '    <p id="empty-state">No reviewed EEE evaluation results were exported.</p>\n'
        if not rows
        else ""
    )
    body = "\n".join(rows)
    if canonical.get("human_review_complete") is True:
        scope_note = (
            "Every exported row was human-reviewed, and the locked review has no pending "
            "decisions. These development results are not holdout evidence or independent "
            "validation."
        )
    else:
        scope_note = (
            "Every exported row was human-reviewed. Other candidates remain pending and are "
            "withheld; see the publication manifest for counts. These development results are "
            "not holdout evidence or independent validation."
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reviewed development EEE results</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 2rem auto; max-width: 96rem; padding: 0 1rem; line-height: 1.45; }}
    .notice {{ max-width: 72rem; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    caption {{ font-weight: 700; padding: .75rem; text-align: left; }}
    th, td {{ border: 1px solid currentColor; padding: .5rem; text-align: left;
      vertical-align: top; }}
    th {{ background: Canvas; position: sticky; top: 0; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <main>
    <h1>Reviewed development EEE results</h1>
    <p class="notice" id="scope-note">
      {escape(scope_note)}</p>
{empty}    <div class="table-wrap">
      <table aria-describedby="scope-note">
        <caption>{canonical["result_count"]} reviewed evaluation results</caption>
        <thead>
          <tr><th scope="col">Paper</th><th scope="col">Evaluated system</th>
            <th scope="col">Dataset</th><th scope="col">Evaluation</th>
            <th scope="col">Metric</th><th scope="col">Score</th>
            <th scope="col">Unit</th><th scope="col">Result ID</th></tr>
        </thead>
        <tbody>
{body}
        </tbody>
      </table>
    </div>
  </main>
</body>
</html>
"""


def _render_readme(manifest: Mapping[str, Any]) -> str:
    outputs = _mapping(manifest.get("outputs"), "manifest outputs")
    status = _mapping(manifest.get("artifact_status"), "manifest artifact status")
    if status.get("publication_ready") is True and status.get("human_review_complete") is not True:
        opening = (
            "This bundle contains individually human-reviewed exported results. "
            "Unreviewed decisions remain pending and are safely withheld."
        )
    elif status.get("human_review_complete") is not True:
        opening = (
            "This bundle records a locked review state with pending decisions. "
            "It is not publication-ready."
        )
    elif status.get("publication_ready") is not True:
        opening = (
            "This bundle records a completed review with no exported EEE results. "
            "It is not labeled publication-ready."
        )
    else:
        opening = (
            "This bundle closes one sealed pre-human candidate preview with a locked and "
            "contextually verified human review."
        )
    return f"""# Reviewed-development closure bundle

{opening} It contains {outputs["eee_records"]} EEE records and
{outputs["evaluation_results"]} evaluation results from an inspected development set.

Review accounting: {outputs["review_items"]} items; {outputs["decisions_completed"]} completed
and {outputs["decisions_pending"]} pending decisions; {outputs["outcomes_exported"]} exported,
{outputs["outcomes_withheld"]} withheld, and {outputs["outcomes_failed"]} failed outcomes.

The results are not holdout evidence, independent validation, or evidence of generalization.
No aggregate score is computed across papers or evaluation results.

## Contents

- `candidate-preview/` is build-time-attested as a byte-for-byte copy and verifies standalone.
- `reviewed/eee/` contains only manifest-listed, schema-valid reviewed EEE records.
- `reviewed/derived-manifest.json` is build-time-attested as the contextual manifest copy;
  standalone verification rechecks its public structure and bindings.
- `reviewed/canonical-results.json` is a one-row-per-result deterministic projection.
- `reviewed/canonical-results.html` is the accessible static view of that projection.
- `reviewed/public-summary.json` binds the reviewed output to the sealed source run.
- `publication-manifest.json`, `verification.json`, and `SHA256SUMS` make the bundle
  independently checkable without private review files.

The manifest binds the contextual build-time evidence-nonreproduction attestation to the sealed
source run and outgoing payload tree. Standalone verification rederives the public denylist,
privacy, schema, inventory, and rendering checks. It cannot rederive the private evidence set.
"""


def _payload_tree(files: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(files)))


def _assert_no_private_evidence_in_contents(
    contents: Mapping[str, bytes],
    evidence_texts: set[str],
) -> None:
    for path, content in sorted(contents.items()):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PublicReviewedDevelopmentBundleError(
                "outgoing public payload is not UTF-8"
            ) from error
        try:
            value: object = (
                _strict_json(content, "outgoing public JSON") if path.endswith(".json") else text
            )
            assert_no_private_evidence_text(value, evidence_texts)
        except PublicDevelopmentPreviewError as error:
            raise PublicReviewedDevelopmentBundleError(
                "outgoing public payload failed the private-evidence nonreproduction gate"
            ) from error


def _private_evidence_attestation(
    *,
    evidence_binding: Mapping[str, object],
    source_binding: Mapping[str, Any],
    review_lock_sha256: str,
    decisions_sha256: str,
    payload_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "public-private-evidence-nonreproduction-attestation/0.1",
        "status": "attested_at_contextual_build",
        "source_run_seal_sha256": source_binding["seal_sha256"],
        "review_lock_sha256": review_lock_sha256,
        "decisions_sha256": decisions_sha256,
        "evidence_set": dict(evidence_binding),
        "attested_payload_tree_sha256": _payload_tree(payload_files),
        "attested_payload_file_count": len(payload_files),
        "standalone_private_evidence_rederivation_available": False,
    }


def _source_binding(
    preview_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    derived: DerivedRunManifest,
) -> dict[str, Any]:
    preview_source = _mapping(preview_manifest.get("source_run"), "preview source run")
    preview_seal = _mapping(preview_source.get("run_seal"), "preview source seal")
    summary_run = _mapping(summary.get("run_binding"), "summary run binding")
    summary_seal = _mapping(summary_run.get("run_seal"), "summary run seal")
    seal_sha256 = _hash(preview_seal.get("seal_sha256"), "preview source seal")
    tree_sha256 = _hash(preview_seal.get("tree_sha256"), "preview source tree")
    run_name = _safe_id(summary_run.get("run_id"), "summary source run name")
    if (
        summary_seal.get("seal_sha256") != seal_sha256
        or summary_seal.get("tree_sha256") != tree_sha256
        or derived.source_run_name != run_name
        or derived.source_run_seal_sha256 != seal_sha256
        or derived.source_run_tree_sha256 != tree_sha256
    ):
        raise PublicReviewedDevelopmentBundleError(
            "preview, summary, and reviewed result bind different source runs"
        )
    return {
        "source_run_name": run_name,
        "seal_sha256": seal_sha256,
        "tree_sha256": tree_sha256,
    }


def _validate_contextual_summary(
    summary: Mapping[str, Any],
    *,
    source_binding: Mapping[str, Any],
    derived: DerivedRunManifest,
    eee_records: int,
    evaluation_results: int,
) -> None:
    if summary.get("schema_version") != PUBLIC_DEVELOPMENT_SUMMARY_SCHEMA_VERSION:
        raise PublicReviewedDevelopmentBundleError("public summary schema is unsupported")
    scope = _mapping(summary.get("scope"), "public summary scope")
    if (
        scope.get("split") != "development"
        or scope.get("holdout_included") is not False
        or scope.get("private_human_annotations_included") is not False
        or scope.get("independent_human_validation") is not False
    ):
        raise PublicReviewedDevelopmentBundleError("public summary scope is not development-only")
    reviewed = _mapping(
        _mapping(summary.get("export_provenance_modes"), "summary provenance").get(
            "reviewed_derived"
        ),
        "summary reviewed result",
    )
    _exact_keys(
        reviewed,
        {
            "status",
            "locked_review_context_verified",
            "independence_status",
            "exported_observation_authority_mode_counts",
            "source_run_binding",
            "derived_run_sha256",
            "eee_records",
            "eee_observations",
            "outcomes_exported",
            "outcomes_withheld",
            "outcomes_failed",
            "provenance_modes_present",
            "provenance_mode_counts",
            "provenance_mode_count_status",
            "source_and_reviewed_counts_combined",
        },
        "summary reviewed result",
    )
    authority_counts = _mapping(
        reviewed.get("exported_observation_authority_mode_counts"),
        "review authority counts",
    )
    _exact_keys(
        authority_counts,
        {"single_expert", "dual_consensus", "adjudicated"},
        "review authority counts",
    )
    provenance_counts = _mapping(
        reviewed.get("provenance_mode_counts"),
        "reviewed provenance counts",
    )
    _exact_keys(
        provenance_counts,
        {
            "tuple_gated_production",
            "tuple_gated_unverified",
            "tuple_audited_human_reviewed",
            "legacy_manual",
            "legacy_human_reviewed",
            "unclassified_legacy_source",
        },
        "reviewed provenance counts",
    )
    expected_provenance_modes = [mode.value for mode in derived.export_provenance_modes]
    if (
        reviewed.get("status") != "contextually_verified"
        or reviewed.get("locked_review_context_verified") is not True
        or reviewed.get("independence_status") != "not_measured"
        or reviewed.get("source_run_binding")
        != {
            "seal_sha256": source_binding["seal_sha256"],
            "tree_sha256": source_binding["tree_sha256"],
        }
        or reviewed.get("derived_run_sha256") != sha256_bytes(canonical_json_bytes(derived))
        or reviewed.get("eee_records") != eee_records
        or reviewed.get("eee_observations") != evaluation_results
        or reviewed.get("outcomes_exported") != derived.counts["outcomes_exported"]
        or reviewed.get("outcomes_withheld") != derived.counts["outcomes_withheld"]
        or reviewed.get("outcomes_failed") != derived.counts["outcomes_failed"]
        or reviewed.get("provenance_modes_present") != expected_provenance_modes
        or reviewed.get("provenance_mode_count_status") != "measured"
        or sum(_count(value, "reviewed provenance count") for value in provenance_counts.values())
        != evaluation_results
        or sum(_count(value, "review authority count") for value in authority_counts.values())
        != evaluation_results
        or reviewed.get("source_and_reviewed_counts_combined") is not False
    ):
        raise PublicReviewedDevelopmentBundleError("public summary reviewed binding disagrees")
    provenance = _mapping(summary.get("export_provenance_modes"), "summary provenance")
    if provenance.get("counts_combined") is not False:
        raise PublicReviewedDevelopmentBundleError("public summary combines incomparable counts")
    annotation = _mapping(summary.get("annotation_status"), "summary annotation status")
    _exact_keys(
        annotation,
        {
            "source_run_human_annotations_included",
            "reviewed_derived_supplied",
            "reviewed_derived_authority_mode_counts",
            "inter_annotator_agreement_available",
        },
        "summary annotation status",
    )
    if (
        annotation.get("source_run_human_annotations_included") is not False
        or annotation.get("reviewed_derived_supplied") is not True
        or annotation.get("reviewed_derived_authority_mode_counts") != authority_counts
        or annotation.get("inter_annotator_agreement_available") is not False
    ):
        raise PublicReviewedDevelopmentBundleError("public summary omits reviewed authority")
    privacy = _mapping(summary.get("privacy"), "summary privacy")
    _exact_keys(
        privacy,
        {
            "evidence_quotations_included",
            "paper_level_rows_or_labels_included",
            "provider_traces_included",
            "request_identifiers_included",
            "credentials_included",
            "local_paths_included",
            "private_annotations_included",
        },
        "summary privacy",
    )
    if any(value is not False for value in privacy.values()):
        raise PublicReviewedDevelopmentBundleError("public summary privacy boundary is invalid")
    _audit_public_value(summary, "public summary")


def _manifest(
    *,
    bundle_id: str,
    source_binding: Mapping[str, Any],
    preview_manifest: Mapping[str, Any],
    preview_verification: Mapping[str, Any],
    derived: DerivedRunManifest,
    derived_manifest_sha256: str,
    summary_sha256: str,
    canonical_sha256: str,
    html_sha256: str,
    eee_files: Sequence[Mapping[str, Any]],
    payload_files: Sequence[Mapping[str, Any]],
    private_evidence_attestation: Mapping[str, Any],
    publication_state: Mapping[str, Any],
    result_count: int,
) -> dict[str, Any]:
    result = {
        "schema_version": PUBLIC_REVIEWED_DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
        "status": publication_state["status"],
        "bundle_id": bundle_id,
        "artifact_status": {
            "classification": publication_state["classification"],
            "development_only": True,
            "independent_validation": False,
            "human_review_status": publication_state["human_review_status"],
            "human_review_complete": publication_state["human_review_complete"],
            "publication_ready": publication_state["publication_ready"],
        },
        "source_run": dict(source_binding),
        "candidate_preview": {
            "bundle_id": _safe_id(preview_manifest.get("bundle_id"), "preview bundle ID"),
            "copied_root": "candidate-preview",
            "publication_manifest_sha256": sha256_bytes(canonical_json_bytes(preview_manifest)),
            "checksums_sha256": _hash(
                preview_verification.get("checksums_sha256"), "preview checksums"
            ),
            "copied_byte_for_byte_at_build": True,
        },
        "reviewed_derived": {
            "derived_run_sha256": derived_manifest_sha256,
            "manifest_path": "reviewed/derived-manifest.json",
            "manifest_copied_byte_for_byte_at_build": True,
            "review_manifest_sha256": derived.review_manifest_sha256,
            "review_lock_sha256": derived.review_lock_sha256,
            "decisions_sha256": derived.decisions_sha256,
            "contextually_verified_at_build": True,
            "eee_schema_version": derived.eee_schema_version,
            "eee_schema_sha256": derived.eee_schema_sha256,
            "eee_files": list(eee_files),
        },
        "private_evidence_nonreproduction": dict(private_evidence_attestation),
        "outputs": {
            "review_items": derived.counts["review_items"],
            "decisions_completed": derived.counts["decisions_completed"],
            "decisions_pending": derived.counts["decisions_pending"],
            "outcomes_exported": derived.counts["outcomes_exported"],
            "outcomes_withheld": derived.counts["outcomes_withheld"],
            "outcomes_failed": derived.counts["outcomes_failed"],
            "eee_records": len(eee_files),
            "evaluation_results": result_count,
            "aggregate_results": 0,
            "public_summary": {
                "path": "reviewed/public-summary.json",
                "sha256": summary_sha256,
            },
            "canonical_results": {
                "path": "reviewed/canonical-results.json",
                "sha256": canonical_sha256,
                "rows": result_count,
            },
            "static_html": {
                "path": "reviewed/canonical-results.html",
                "sha256": html_sha256,
                "rows": result_count,
            },
        },
        "files": list(payload_files),
        "limitations": [
            "This bundle reports an inspected development set, not a holdout.",
            "Independent validation was not measured.",
            "Results are listed one by one and are not aggregated.",
        ],
    }
    _audit_public_value(result, "publication manifest")
    return result


def _verification(
    *, manifest: Mapping[str, Any], payload_files: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_REVIEWED_DEVELOPMENT_VERIFICATION_SCHEMA_VERSION,
        "status": "verified",
        "bundle_id": manifest["bundle_id"],
        "publication_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "payload_tree_sha256": _payload_tree(payload_files),
        "payload_file_count": len(payload_files),
        "source_run": manifest["source_run"],
        "private_evidence_nonreproduction": manifest["private_evidence_nonreproduction"],
        "checks": {
            "candidate_preview_standalone_verified": True,
            "candidate_preview_copy_verified_at_build": True,
            "reviewed_derived_contextually_verified_at_build": True,
            "copied_derived_manifest_structure_and_bindings_recomputed_standalone": True,
            "source_run_binding_consistent": True,
            "manifest_listed_eee_only": True,
            "all_eee_schema_valid": True,
            "canonical_rows_recomputed": True,
            "html_recomputed": True,
            "no_aggregate_result_computed": True,
            "no_unexpected_files": True,
            "no_symbolic_links": True,
            "private_evidence_nonreproduction_attested_at_contextual_build": True,
            "private_evidence_nonreproduction_rederived_standalone": False,
            "standalone_denylist_and_privacy_scan_rederived": True,
        },
    }


def _checksums(contents: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{sha256_bytes(content)}  {path}\n"
        for path, content in sorted(contents.items())
        if path != "SHA256SUMS"
    ).encode("utf-8")


def _write_tree(root: Path, contents: Mapping[str, bytes]) -> None:
    for relative, content in sorted(contents.items()):
        target = root.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)


def _write_tree_at(root_descriptor: int, contents: Mapping[str, bytes]) -> None:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    for relative, content in sorted(contents.items()):
        parts = PurePosixPath(relative).parts
        current = os.dup(root_descriptor)
        try:
            for part in parts[:-1]:
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o755, dir_fd=current)
                child = os.open(part, directory_flags, dir_fd=current)
                os.close(current)
                current = child
            descriptor = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=current,
            )
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
            os.fsync(current)
        finally:
            os.close(current)


def _open_relative_directory_at(root_descriptor: int, parts: Sequence[str]) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.dup(root_descriptor)
    try:
        for part in parts:
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _remove_known_tree_at(root_descriptor: int, contents: Mapping[str, bytes]) -> None:
    """Best-effort cleanup confined to names created below an already-open root."""

    for relative in sorted(contents, reverse=True):
        parts = PurePosixPath(relative).parts
        parent_descriptor: int | None = None
        try:
            parent_descriptor = _open_relative_directory_at(root_descriptor, parts[:-1])
            os.unlink(parts[-1], dir_fd=parent_descriptor)
        except OSError:
            pass
        finally:
            if parent_descriptor is not None:
                os.close(parent_descriptor)
    directories = {
        parent.parts
        for relative in contents
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    for parts in sorted(directories, key=lambda item: (-len(item), item)):
        parent_descriptor = None
        try:
            parent_descriptor = _open_relative_directory_at(root_descriptor, parts[:-1])
            os.rmdir(parts[-1], dir_fd=parent_descriptor)
        except OSError:
            pass
        finally:
            if parent_descriptor is not None:
                os.close(parent_descriptor)


def _publish_tree(output: Path, contents: Mapping[str, bytes]) -> Path:
    """Exclusively publish a fresh verified tree without replacing an existing entry."""

    if os.path.lexists(output):
        raise PublicReviewedDevelopmentBundleError("public reviewed bundle destination exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    tree = temporary_parent / "tree"
    tree.mkdir(mode=0o700)
    try:
        _write_tree(tree, contents)
        verify_public_reviewed_development_bundle(tree)
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(output.parent, parent_flags)
        destination_descriptor: int | None = None
        destination_identity: tuple[int, int] | None = None
        created = False
        try:
            try:
                os.mkdir(output.name, mode=0o700, dir_fd=parent_descriptor)
                created = True
            except FileExistsError as error:
                raise PublicReviewedDevelopmentBundleError(
                    "public reviewed bundle destination appeared during publication"
                ) from error
            destination_descriptor = os.open(
                output.name,
                parent_flags,
                dir_fd=parent_descriptor,
            )
            destination_stat = os.fstat(destination_descriptor)
            destination_identity = (destination_stat.st_dev, destination_stat.st_ino)
            _write_tree_at(destination_descriptor, contents)
            verification = verify_public_reviewed_development_bundle(output)
            if verification.get("status") != "verified":
                raise PublicReviewedDevelopmentBundleError(
                    "published reviewed bundle did not verify"
                )
            current = os.stat(output.name, dir_fd=parent_descriptor, follow_symlinks=False)
            if destination_identity != (current.st_dev, current.st_ino):
                raise PublicReviewedDevelopmentBundleError(
                    "public reviewed bundle destination changed during publication"
                )
            os.fchmod(destination_descriptor, 0o755)
            os.fsync(destination_descriptor)
            os.fsync(parent_descriptor)
        except BaseException:
            if destination_descriptor is not None:
                _remove_known_tree_at(destination_descriptor, contents)
            if created and destination_identity is not None:
                try:
                    current = os.stat(
                        output.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if destination_identity == (current.st_dev, current.st_ino):
                        os.rmdir(output.name, dir_fd=parent_descriptor)
                except OSError:
                    pass
            raise
        finally:
            if destination_descriptor is not None:
                os.close(destination_descriptor)
            os.close(parent_descriptor)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)
    return output


def _validate_private_evidence_attestation(
    value: object,
    *,
    source_binding: Mapping[str, Any],
    reviewed_binding: Mapping[str, Any],
    payload_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attestation = dict(_mapping(value, "private evidence attestation"))
    _exact_keys(
        attestation,
        {
            "schema_version",
            "status",
            "source_run_seal_sha256",
            "review_lock_sha256",
            "decisions_sha256",
            "evidence_set",
            "attested_payload_tree_sha256",
            "attested_payload_file_count",
            "standalone_private_evidence_rederivation_available",
        },
        "private evidence attestation",
    )
    evidence_set = _mapping(attestation.get("evidence_set"), "private evidence set")
    _exact_keys(
        evidence_set,
        {
            "schema_version",
            "canonicalization",
            "minimum_exact_match_characters",
            "minimum_embedded_match_characters",
            "evidence_text_count",
            "evidence_set_sha256",
        },
        "private evidence set",
    )
    if (
        attestation.get("schema_version")
        != "public-private-evidence-nonreproduction-attestation/0.1"
        or attestation.get("status") != "attested_at_contextual_build"
        or attestation.get("source_run_seal_sha256") != source_binding.get("seal_sha256")
        or attestation.get("review_lock_sha256") != reviewed_binding.get("review_lock_sha256")
        or attestation.get("decisions_sha256") != reviewed_binding.get("decisions_sha256")
        or evidence_set.get("schema_version") != PRIVATE_EVIDENCE_SET_SCHEMA_VERSION
        or evidence_set.get("canonicalization") != PRIVATE_EVIDENCE_TEXT_CANONICALIZATION
        or evidence_set.get("minimum_exact_match_characters") != 16
        or evidence_set.get("minimum_embedded_match_characters") != 40
        or attestation.get("attested_payload_tree_sha256") != _payload_tree(payload_files)
        or attestation.get("attested_payload_file_count") != len(payload_files)
        or attestation.get("standalone_private_evidence_rederivation_available") is not False
    ):
        raise PublicReviewedDevelopmentBundleError("private evidence attestation binding disagrees")
    _count(evidence_set.get("evidence_text_count"), "private evidence text count")
    _hash(evidence_set.get("evidence_set_sha256"), "private evidence set hash")
    _hash(
        attestation.get("attested_payload_tree_sha256"),
        "attested payload tree hash",
    )
    _hash(attestation.get("review_lock_sha256"), "attested review lock hash")
    _hash(attestation.get("decisions_sha256"), "attested decisions hash")
    return attestation


def _parse_manifest(content: bytes) -> dict[str, Any]:
    manifest = dict(_mapping(_strict_json(content, "publication manifest"), "manifest"))
    _exact_keys(
        manifest,
        {
            "schema_version",
            "status",
            "bundle_id",
            "artifact_status",
            "source_run",
            "candidate_preview",
            "reviewed_derived",
            "private_evidence_nonreproduction",
            "outputs",
            "files",
            "limitations",
        },
        "publication manifest",
    )
    supported_statuses = {
        "verified_human_reviewed_development",
        "verified_human_reviewed_development_with_pending_withheld",
        "verified_review_incomplete_development",
        "verified_reviewed_empty_development",
    }
    if (
        manifest.get("schema_version") != PUBLIC_REVIEWED_DEVELOPMENT_BUNDLE_SCHEMA_VERSION
        or manifest.get("status") not in supported_statuses
    ):
        raise PublicReviewedDevelopmentBundleError("publication manifest schema/status is invalid")
    _safe_id(manifest.get("bundle_id"), "bundle ID")
    status = _mapping(manifest.get("artifact_status"), "artifact status")
    _exact_keys(
        status,
        {
            "classification",
            "development_only",
            "independent_validation",
            "human_review_status",
            "human_review_complete",
            "publication_ready",
        },
        "artifact status",
    )
    supported_artifact_states = {
        "verified_human_reviewed_development": {
            "classification": "human_reviewed_development",
            "development_only": True,
            "independent_validation": False,
            "human_review_status": "locked_and_contextually_verified",
            "human_review_complete": True,
            "publication_ready": True,
        },
        "verified_human_reviewed_development_with_pending_withheld": {
            "classification": "human_reviewed_development_with_pending_withheld",
            "development_only": True,
            "independent_validation": False,
            "human_review_status": "locked_reviewed_exports_with_pending_withheld",
            "human_review_complete": False,
            "publication_ready": True,
        },
        "verified_review_incomplete_development": {
            "classification": "review_incomplete_development",
            "development_only": True,
            "independent_validation": False,
            "human_review_status": "locked_with_pending_decisions",
            "human_review_complete": False,
            "publication_ready": False,
        },
        "verified_reviewed_empty_development": {
            "classification": "reviewed_empty_development",
            "development_only": True,
            "independent_validation": False,
            "human_review_status": "locked_complete_no_exported_results",
            "human_review_complete": True,
            "publication_ready": False,
        },
    }
    if status != supported_artifact_states[manifest["status"]]:
        raise PublicReviewedDevelopmentBundleError("artifact status overstates the evidence")
    source = _mapping(manifest.get("source_run"), "source run")
    _exact_keys(source, {"source_run_name", "seal_sha256", "tree_sha256"}, "source run")
    _safe_id(source.get("source_run_name"), "source run name")
    _hash(source.get("seal_sha256"), "source seal")
    _hash(source.get("tree_sha256"), "source tree")
    preview = _mapping(manifest.get("candidate_preview"), "candidate preview")
    _exact_keys(
        preview,
        {
            "bundle_id",
            "copied_root",
            "publication_manifest_sha256",
            "checksums_sha256",
            "copied_byte_for_byte_at_build",
        },
        "candidate preview",
    )
    _safe_id(preview.get("bundle_id"), "preview bundle ID")
    if (
        preview.get("copied_root") != "candidate-preview"
        or preview.get("copied_byte_for_byte_at_build") is not True
    ):
        raise PublicReviewedDevelopmentBundleError("candidate preview copy claim is invalid")
    _hash(preview.get("publication_manifest_sha256"), "preview manifest hash")
    _hash(preview.get("checksums_sha256"), "preview checksums hash")
    reviewed = _mapping(manifest.get("reviewed_derived"), "reviewed derived")
    _exact_keys(
        reviewed,
        {
            "derived_run_sha256",
            "manifest_path",
            "manifest_copied_byte_for_byte_at_build",
            "review_manifest_sha256",
            "review_lock_sha256",
            "decisions_sha256",
            "contextually_verified_at_build",
            "eee_schema_version",
            "eee_schema_sha256",
            "eee_files",
        },
        "reviewed derived",
    )
    for field in (
        "derived_run_sha256",
        "review_manifest_sha256",
        "review_lock_sha256",
        "decisions_sha256",
        "eee_schema_sha256",
    ):
        _hash(reviewed.get(field), f"reviewed derived {field}")
    if (
        reviewed.get("contextually_verified_at_build") is not True
        or reviewed.get("manifest_path") != "reviewed/derived-manifest.json"
        or reviewed.get("manifest_copied_byte_for_byte_at_build") is not True
        or reviewed.get("eee_schema_version") != EEE_SCHEMA_VERSION
        or reviewed.get("eee_schema_sha256") != EEE_SCHEMA_SHA256
    ):
        raise PublicReviewedDevelopmentBundleError("reviewed derived authority is invalid")
    eee_files: list[dict[str, Any]] = []
    for index, raw in enumerate(_sequence(reviewed.get("eee_files"), "EEE files")):
        entry = dict(_mapping(raw, f"EEE file {index}"))
        _exact_keys(
            entry,
            {"source_path", "bundle_path", "sha256", "size_bytes", "evaluation_results"},
            f"EEE file {index}",
        )
        _relative_path(entry.get("source_path"), "EEE source path")
        _relative_path(entry.get("bundle_path"), "EEE bundle path")
        _hash(entry.get("sha256"), "EEE hash")
        _count(entry.get("size_bytes"), "EEE size")
        _count(entry.get("evaluation_results"), "EEE results")
        eee_files.append(entry)
    if eee_files != sorted(eee_files, key=lambda item: item["bundle_path"]):
        raise PublicReviewedDevelopmentBundleError("EEE file inventory is not sorted")
    if len({item["bundle_path"] for item in eee_files}) != len(eee_files):
        raise PublicReviewedDevelopmentBundleError("EEE file inventory has duplicate paths")
    outputs = _mapping(manifest.get("outputs"), "outputs")
    _exact_keys(
        outputs,
        {
            "eee_records",
            "evaluation_results",
            "aggregate_results",
            "review_items",
            "decisions_completed",
            "decisions_pending",
            "outcomes_exported",
            "outcomes_withheld",
            "outcomes_failed",
            "public_summary",
            "canonical_results",
            "static_html",
        },
        "outputs",
    )
    if _count(outputs.get("eee_records"), "EEE records") != len(eee_files):
        raise PublicReviewedDevelopmentBundleError("EEE record count disagrees")
    for field in (
        "review_items",
        "decisions_completed",
        "decisions_pending",
        "outcomes_exported",
        "outcomes_withheld",
        "outcomes_failed",
    ):
        _count(outputs.get(field), f"outputs {field}")
    result_count = _count(outputs.get("evaluation_results"), "evaluation results")
    if outputs.get("aggregate_results") != 0:
        raise PublicReviewedDevelopmentBundleError("aggregate results must remain zero")
    for key, expected_path, has_rows in (
        ("public_summary", "reviewed/public-summary.json", False),
        ("canonical_results", "reviewed/canonical-results.json", True),
        ("static_html", "reviewed/canonical-results.html", True),
    ):
        item = _mapping(outputs.get(key), f"outputs {key}")
        _exact_keys(item, {"path", "sha256"} | ({"rows"} if has_rows else set()), key)
        if item.get("path") != expected_path:
            raise PublicReviewedDevelopmentBundleError(f"{key} path is invalid")
        _hash(item.get("sha256"), f"{key} hash")
        if has_rows and item.get("rows") != result_count:
            raise PublicReviewedDevelopmentBundleError(f"{key} row count disagrees")
    files = [
        _validate_artifact(item, f"payload file {index}")
        for index, item in enumerate(_sequence(manifest.get("files"), "payload files"))
    ]
    if files != sorted(files, key=lambda item: item["path"]):
        raise PublicReviewedDevelopmentBundleError("payload inventory is not sorted")
    if len({item["path"] for item in files}) != len(files):
        raise PublicReviewedDevelopmentBundleError("payload inventory has duplicate paths")
    expected_payloads = (
        {f"candidate-preview/{name}" for name in _PREVIEW_FILES}
        | _REVIEWED_PROJECTION_FILES
        | {item["bundle_path"] for item in eee_files}
    )
    if {item["path"] for item in files} != expected_payloads:
        raise PublicReviewedDevelopmentBundleError("payload allowlist disagrees")
    _validate_private_evidence_attestation(
        manifest.get("private_evidence_nonreproduction"),
        source_binding=source,
        reviewed_binding=reviewed,
        payload_files=files,
    )
    limitations = _sequence(manifest.get("limitations"), "limitations")
    if not limitations or any(not isinstance(item, str) or not item for item in limitations):
        raise PublicReviewedDevelopmentBundleError("limitations are invalid")
    _audit_public_value(manifest, "publication manifest")
    return manifest


def _tree_entry_identity(path: Path, *, kind: str) -> tuple[str, int, int, int, int, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PublicReviewedDevelopmentBundleError(
            "public bundle changed during verification"
        ) from error
    return (
        kind,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _capture_tree(
    root: Path,
    expected: set[str],
) -> tuple[dict[str, bytes], tuple[tuple[str, tuple[str, int, int, int, int, int]], ...]]:
    contents: dict[str, bytes] = {}
    directories: set[str] = set()
    identities: dict[str, tuple[str, int, int, int, int, int]] = {
        ".": _tree_entry_identity(root, kind="directory")
    }
    total = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise PublicReviewedDevelopmentBundleError("public bundle contains a symbolic link")
        if path.is_dir():
            directories.add(relative)
            identities[relative] = _tree_entry_identity(path, kind="directory")
            continue
        if not path.is_file():
            raise PublicReviewedDevelopmentBundleError("public bundle contains a non-file entry")
        content = _read_regular(path, "public bundle file")
        total += len(content)
        if total > _MAX_TOTAL_BYTES:
            raise PublicReviewedDevelopmentBundleError("public bundle exceeds its size limit")
        contents[relative] = content
        identities[relative] = _tree_entry_identity(path, kind="file")
    if set(contents) != expected:
        raise PublicReviewedDevelopmentBundleError("public bundle file inventory disagrees")
    expected_directories = {
        PurePosixPath(path).parent.as_posix()
        for path in expected
        if PurePosixPath(path).parent.as_posix() != "."
    }
    expected_directories |= {
        parent.as_posix()
        for path in expected_directories.copy()
        for parent in PurePosixPath(path).parents
        if parent.as_posix() != "."
    }
    if directories != expected_directories:
        raise PublicReviewedDevelopmentBundleError("public bundle directory inventory disagrees")
    if identities["."] != _tree_entry_identity(root, kind="directory"):
        raise PublicReviewedDevelopmentBundleError("public bundle changed during verification")
    return contents, tuple(sorted(identities.items()))


def _verify_checksums(contents: Mapping[str, bytes], expected: set[str]) -> None:
    checksum_bytes = contents.get("SHA256SUMS")
    if checksum_bytes is None:
        raise PublicReviewedDevelopmentBundleError("SHA256SUMS is missing")
    try:
        lines = checksum_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise PublicReviewedDevelopmentBundleError("SHA256SUMS is not UTF-8") from error
    names = sorted(expected - {"SHA256SUMS"})
    if len(lines) != len(names):
        raise PublicReviewedDevelopmentBundleError("SHA256SUMS inventory is incomplete")
    parsed: list[tuple[str, str]] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._/-]*)", line)
        if match is None:
            raise PublicReviewedDevelopmentBundleError("SHA256SUMS line is invalid")
        path = _relative_path(match.group(2), "SHA256SUMS path")
        parsed.append((match.group(1), path))
    if [path for _, path in parsed] != names:
        raise PublicReviewedDevelopmentBundleError("SHA256SUMS order/allowlist disagrees")
    if any(sha256_bytes(contents[path]) != digest for digest, path in parsed):
        raise PublicReviewedDevelopmentBundleError("SHA256SUMS verification failed")
    if checksum_bytes != "".join(f"{digest}  {path}\n" for digest, path in parsed).encode():
        raise PublicReviewedDevelopmentBundleError("SHA256SUMS rendering is not canonical")


def _verify_verification(
    value: object,
    *,
    manifest: Mapping[str, Any],
    payload_files: Sequence[Mapping[str, Any]],
) -> None:
    verification = _mapping(value, "verification receipt")
    expected = _verification(manifest=manifest, payload_files=payload_files)
    if verification != expected:
        raise PublicReviewedDevelopmentBundleError("verification receipt disagrees")


def _verify_copied_derived_manifest(
    content: bytes,
    *,
    reviewed: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    eee_files: Sequence[Mapping[str, Any]],
) -> DerivedRunManifest:
    try:
        derived = DerivedRunManifest.model_validate(
            _strict_json(content, "copied derived manifest")
        )
    except (ValidationError, ValueError) as error:
        raise PublicReviewedDevelopmentBundleError("copied derived manifest is invalid") from error
    if content != canonical_json_bytes(derived):
        raise PublicReviewedDevelopmentBundleError("copied derived manifest is not canonical JSON")
    if (
        sha256_bytes(content) != reviewed.get("derived_run_sha256")
        or derived.source_run_name != source_binding.get("source_run_name")
        or derived.source_run_seal_sha256 != source_binding.get("seal_sha256")
        or derived.source_run_tree_sha256 != source_binding.get("tree_sha256")
        or derived.review_manifest_sha256 != reviewed.get("review_manifest_sha256")
        or derived.review_lock_sha256 != reviewed.get("review_lock_sha256")
        or derived.decisions_sha256 != reviewed.get("decisions_sha256")
        or derived.eee_schema_version != reviewed.get("eee_schema_version")
        or derived.eee_schema_sha256 != reviewed.get("eee_schema_sha256")
        or derived.contains_evidence_quotations is not False
        or derived.contains_reviewer_identities is not False
        or derived.contains_absolute_paths is not False
    ):
        raise PublicReviewedDevelopmentBundleError("copied derived manifest binding disagrees")
    canonical_payload_files = [
        artifact.model_dump(mode="json", by_alias=True, exclude_none=False)
        for artifact in derived.payload_files
    ]
    expected_payload_tree_sha256 = sha256_bytes(canonical_json_bytes(canonical_payload_files))
    expected_sha256s = "".join(
        f"{artifact.sha256}  {artifact.path}\n" for artifact in derived.payload_files
    ).encode("utf-8")
    if (
        derived.payload_tree_sha256 != expected_payload_tree_sha256
        or derived.sha256s_sha256 != sha256_bytes(expected_sha256s)
    ):
        raise PublicReviewedDevelopmentBundleError(
            "copied derived manifest inventory hashes disagree"
        )
    payload_paths = [artifact.path for artifact in derived.payload_files]
    if payload_paths != sorted(payload_paths) or len(payload_paths) != len(set(payload_paths)):
        raise PublicReviewedDevelopmentBundleError(
            "copied derived manifest payload inventory is not canonical"
        )
    non_eee_paths: set[str] = set()
    for payload_path in payload_paths:
        parts = PurePosixPath(payload_path).parts
        if payload_path in {"export-outcomes.jsonl", "reviewed-export-provenance.jsonl"}:
            non_eee_paths.add(payload_path)
            continue
        if (
            len(parts) != 3
            or parts[1] != "eee"
            or _SAFE_ID.fullmatch(parts[0]) is None
            or _SAFE_PART.fullmatch(parts[2]) is None
            or not parts[2].endswith(".json")
        ):
            raise PublicReviewedDevelopmentBundleError(
                "copied derived manifest payload inventory has an unsupported path"
            )
    if non_eee_paths != {"export-outcomes.jsonl", "reviewed-export-provenance.jsonl"}:
        raise PublicReviewedDevelopmentBundleError(
            "copied derived manifest non-EEE inventory disagrees"
        )
    listed_eee = {
        artifact.path: artifact
        for artifact in derived.payload_files
        if len(PurePosixPath(artifact.path).parts) == 3
        and PurePosixPath(artifact.path).parts[1] == "eee"
        and PurePosixPath(artifact.path).suffix == ".json"
    }
    observed = {str(item["source_path"]): item for item in eee_files}
    if set(listed_eee) != set(observed):
        raise PublicReviewedDevelopmentBundleError(
            "copied EEE inventory differs from the verified derived manifest"
        )
    for source_path, entry in observed.items():
        artifact = listed_eee[source_path]
        if artifact.sha256 != entry.get("sha256") or artifact.size_bytes != entry.get("size_bytes"):
            raise PublicReviewedDevelopmentBundleError(
                "copied EEE differs from the verified derived manifest"
            )
    if derived.counts["eee_records"] != len(eee_files) or derived.counts["eee_observations"] != sum(
        _count(item.get("evaluation_results"), "EEE results") for item in eee_files
    ):
        raise PublicReviewedDevelopmentBundleError(
            "copied EEE counts differ from the verified derived manifest"
        )
    _audit_public_value(
        derived.model_dump(mode="json", exclude_none=True),
        "copied derived manifest",
    )
    return derived


def verify_public_reviewed_development_bundle(root: Path) -> dict[str, Any]:
    """Verify a reviewed development bundle without private source or review inputs."""

    resolved = _resolve_existing_directory(root, "public reviewed bundle root")
    manifest_content = _read_regular(resolved / "publication-manifest.json", "manifest")
    manifest = _parse_manifest(manifest_content)
    payload_files = list(_sequence(manifest["files"], "payload files"))
    expected = {item["path"] for item in payload_files} | _CONTROL_FILES
    contents, captured_identity = _capture_tree(resolved, expected)
    _verify_checksums(contents, expected)
    for artifact in payload_files:
        content = contents[artifact["path"]]
        if len(content) != artifact["size_bytes"] or sha256_bytes(content) != artifact["sha256"]:
            raise PublicReviewedDevelopmentBundleError("payload differs from publication manifest")
    if contents["publication-manifest.json"] != canonical_json_bytes(manifest):
        raise PublicReviewedDevelopmentBundleError("publication manifest is not canonical JSON")
    verification = _strict_json(contents["verification.json"], "verification receipt")
    _verify_verification(verification, manifest=manifest, payload_files=payload_files)
    if contents["verification.json"] != canonical_json_bytes(verification):
        raise PublicReviewedDevelopmentBundleError("verification receipt is not canonical JSON")

    preview_root = resolved / "candidate-preview"
    try:
        preview_verification = verify_public_development_preview(preview_root)
    except PublicDevelopmentPreviewError as error:
        raise PublicReviewedDevelopmentBundleError(
            "copied candidate preview failed standalone verification"
        ) from error
    preview_manifest = _mapping(
        _strict_json(contents["candidate-preview/publication-manifest.json"], "preview manifest"),
        "preview manifest",
    )
    preview_binding = _mapping(manifest["candidate_preview"], "candidate preview binding")
    if (
        preview_manifest.get("bundle_id") != preview_binding.get("bundle_id")
        or sha256_bytes(contents["candidate-preview/publication-manifest.json"])
        != preview_binding.get("publication_manifest_sha256")
        or preview_verification.get("checksums_sha256") != preview_binding.get("checksums_sha256")
    ):
        raise PublicReviewedDevelopmentBundleError("copied candidate preview binding disagrees")
    preview_source = _mapping(preview_manifest.get("source_run"), "preview source run")
    preview_seal = _mapping(preview_source.get("run_seal"), "preview source seal")
    source_binding = _mapping(manifest.get("source_run"), "source binding")
    if preview_seal.get("seal_sha256") != source_binding.get("seal_sha256") or preview_seal.get(
        "tree_sha256"
    ) != source_binding.get("tree_sha256"):
        raise PublicReviewedDevelopmentBundleError("candidate preview source binding disagrees")

    reviewed = _mapping(manifest.get("reviewed_derived"), "reviewed derived")
    eee_files = [
        _mapping(item, "EEE file") for item in _sequence(reviewed.get("eee_files"), "EEE files")
    ]
    eee_contents = {item["bundle_path"]: contents[item["bundle_path"]] for item in eee_files}
    copied_derived = _verify_copied_derived_manifest(
        contents["reviewed/derived-manifest.json"],
        reviewed=reviewed,
        source_binding=source_binding,
        eee_files=eee_files,
    )
    records = _validated_eee_records(
        eee_contents,
        eee_files,
        review_manifest_sha256=copied_derived.review_manifest_sha256,
        review_lock_sha256=copied_derived.review_lock_sha256,
    )
    result_count = sum(len(record[2]["evaluation_results"]) for record in records)
    _validate_review_count_algebra(copied_derived, result_count=result_count)
    publication_state = _review_publication_state(
        copied_derived,
        result_count=result_count,
    )
    canonical_expected = _canonical_results(
        source_binding=source_binding,
        eee_schema={
            "version": reviewed["eee_schema_version"],
            "sha256": reviewed["eee_schema_sha256"],
        },
        records=records,
        publication_state=publication_state,
    )
    canonical_observed = _strict_json(
        contents["reviewed/canonical-results.json"], "canonical results"
    )
    _audit_public_value(canonical_observed, "canonical results")
    if canonical_observed != canonical_expected or contents[
        "reviewed/canonical-results.json"
    ] != canonical_json_bytes(canonical_expected):
        raise PublicReviewedDevelopmentBundleError("canonical result projection disagrees")
    expected_html = _render_results_html(canonical_expected).encode("utf-8")
    if contents["reviewed/canonical-results.html"] != expected_html:
        raise PublicReviewedDevelopmentBundleError("reviewed result HTML is not deterministic")
    html_text = expected_html.decode("utf-8")
    if re.search(r"<(?:script|img|iframe|link)\b", html_text, re.I):
        raise PublicReviewedDevelopmentBundleError("reviewed result HTML is not self-contained")

    summary = _mapping(
        _strict_json(contents["reviewed/public-summary.json"], "public summary"),
        "public summary",
    )
    _audit_public_value(summary, "public summary")
    if contents["reviewed/public-summary.json"] != canonical_json_bytes(summary):
        raise PublicReviewedDevelopmentBundleError("public summary is not canonical JSON")
    _validate_contextual_summary(
        summary,
        source_binding=source_binding,
        derived=copied_derived,
        eee_records=len(records),
        evaluation_results=canonical_expected["result_count"],
    )
    outputs = _mapping(manifest.get("outputs"), "outputs")
    if (
        manifest.get("status") != publication_state["status"]
        or _mapping(manifest.get("artifact_status"), "artifact status")
        != {
            "classification": publication_state["classification"],
            "development_only": True,
            "independent_validation": False,
            "human_review_status": publication_state["human_review_status"],
            "human_review_complete": publication_state["human_review_complete"],
            "publication_ready": publication_state["publication_ready"],
        }
        or any(
            outputs.get(output_key) != copied_derived.counts[derived_key]
            for output_key, derived_key in (
                ("review_items", "review_items"),
                ("decisions_completed", "decisions_completed"),
                ("decisions_pending", "decisions_pending"),
                ("outcomes_exported", "outcomes_exported"),
                ("outcomes_withheld", "outcomes_withheld"),
                ("outcomes_failed", "outcomes_failed"),
                ("eee_records", "eee_records"),
                ("evaluation_results", "eee_observations"),
            )
        )
        or outputs.get("eee_records") != len(records)
        or outputs.get("evaluation_results") != canonical_expected["result_count"]
        or outputs.get("public_summary", {}).get("sha256")
        != sha256_bytes(contents["reviewed/public-summary.json"])
        or outputs.get("canonical_results", {}).get("sha256")
        != sha256_bytes(contents["reviewed/canonical-results.json"])
        or outputs.get("static_html", {}).get("sha256")
        != sha256_bytes(contents["reviewed/canonical-results.html"])
    ):
        raise PublicReviewedDevelopmentBundleError("publication output binding disagrees")
    expected_readme = _render_readme(manifest).encode("utf-8")
    if contents["README.md"] != expected_readme:
        raise PublicReviewedDevelopmentBundleError("bundle README is not deterministic")
    for path, content in contents.items():
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PublicReviewedDevelopmentBundleError(
                "public bundle contains non-UTF-8 data"
            ) from error
        _scan_text(text, path)
    final_contents, final_identity = _capture_tree(resolved, expected)
    if final_contents != contents or final_identity != captured_identity:
        raise PublicReviewedDevelopmentBundleError("public bundle changed during verification")
    return {
        "schema_version": PUBLIC_REVIEWED_DEVELOPMENT_VERIFICATION_SCHEMA_VERSION,
        "status": "verified",
        "bundle_id": manifest["bundle_id"],
        "file_count": len(contents),
        "eee_records": len(records),
        "evaluation_results": canonical_expected["result_count"],
        "private_evidence_nonreproduction": (
            "build_attestation_integrity_verified_not_rederived_standalone"
        ),
        "checksums_sha256": sha256_bytes(contents["SHA256SUMS"]),
    }


def build_public_reviewed_development_bundle(
    *,
    bundle_id: str,
    run_root: Path,
    preview_root: Path,
    reviewed_derived_root: Path,
    review_root: Path,
    output_root: Path,
) -> Path:
    """Build a fresh public bundle from one exact reviewed development context."""

    bundle_id = _safe_id(bundle_id, "bundle ID")
    source_root = _resolve_existing_directory(run_root, "sealed source run")
    preview_root = _resolve_existing_directory(preview_root, "candidate preview")
    derived_root = _resolve_existing_directory(reviewed_derived_root, "reviewed derived root")
    review_root = _resolve_existing_directory(review_root, "private review root")
    raw_output = _reject_symlink_components(output_root, "public output root")
    raw_destination = raw_output / bundle_id
    for input_root in (source_root, preview_root, derived_root, review_root):
        if _is_within(raw_destination, input_root) or _is_within(input_root, raw_destination):
            raise PublicReviewedDevelopmentBundleError(
                "public reviewed bundle output must be disjoint from every input"
            )
    try:
        raw_output.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(raw_output, "public output root")
        output_parent = raw_output.resolve(strict=True)
    except OSError as error:
        raise PublicReviewedDevelopmentBundleError("public output root is unavailable") from error
    destination = output_parent / bundle_id
    if os.path.lexists(destination):
        raise PublicReviewedDevelopmentBundleError("public reviewed bundle destination exists")
    for input_root in (source_root, preview_root, derived_root, review_root):
        if _is_within(destination, input_root) or _is_within(input_root, destination):
            raise PublicReviewedDevelopmentBundleError(
                "public reviewed bundle output must be disjoint from every input"
            )

    try:
        preview_verification_before = verify_public_development_preview(preview_root)
    except PublicDevelopmentPreviewError as error:
        raise PublicReviewedDevelopmentBundleError(
            "candidate preview failed standalone verification"
        ) from error
    preview_bytes = _preview_contents(preview_root)
    preview_manifest = _mapping(
        _strict_json(
            preview_bytes["candidate-preview/publication-manifest.json"], "preview manifest"
        ),
        "preview manifest",
    )
    try:
        contextual_before = verify_contextual_derived_run(
            derived_root,
            run_root=source_root,
            review_root=review_root,
            schema_path=DEFAULT_EEE_SCHEMA_PATH,
            schema_sha256=EEE_SCHEMA_SHA256,
        )
        derived_manifest_content = _read_regular(
            derived_root / DERIVED_MANIFEST_NAME, "derived manifest"
        )
        derived = DerivedRunManifest.model_validate(
            _strict_json(derived_manifest_content, "derived manifest")
        )
    except (ReviewedExportError, ValidationError, OSError, ValueError) as error:
        raise PublicReviewedDevelopmentBundleError(
            "reviewed result failed contextual verification"
        ) from error
    if contextual_before.verification.derived_run_sha256 != sha256_bytes(derived_manifest_content):
        raise PublicReviewedDevelopmentBundleError("derived verification hash disagrees")
    try:
        summary = build_public_development_summary(
            source_root,
            reviewed_derived_root=derived_root,
            review_root=review_root,
        )
    except PublicDevelopmentSummaryError as error:
        raise PublicReviewedDevelopmentBundleError(
            "contextual public summary construction failed"
        ) from error
    source_binding = _source_binding(preview_manifest, summary, derived)
    if (
        derived.eee_schema_version != EEE_SCHEMA_VERSION
        or derived.eee_schema_sha256 != EEE_SCHEMA_SHA256
    ):
        raise PublicReviewedDevelopmentBundleError("reviewed result uses another EEE schema")
    selected = _manifest_eee_entries(derived_root, derived)
    eee_file_manifest = [
        {
            "source_path": source_path,
            "bundle_path": bundle_path,
            "sha256": sha256_bytes(content),
            "size_bytes": len(content),
            "evaluation_results": result_count,
        }
        for source_path, bundle_path, content, result_count in selected
    ]
    eee_contents = {bundle_path: content for _, bundle_path, content, _ in selected}
    records = _validated_eee_records(
        eee_contents,
        eee_file_manifest,
        review_manifest_sha256=derived.review_manifest_sha256,
        review_lock_sha256=derived.review_lock_sha256,
    )
    result_count = sum(len(record[2]["evaluation_results"]) for record in records)
    _validate_review_count_algebra(derived, result_count=result_count)
    publication_state = _review_publication_state(
        derived,
        result_count=result_count,
    )
    canonical = _canonical_results(
        source_binding=source_binding,
        eee_schema={"version": EEE_SCHEMA_VERSION, "sha256": EEE_SCHEMA_SHA256},
        records=records,
        publication_state=publication_state,
    )
    _validate_contextual_summary(
        summary,
        source_binding=source_binding,
        derived=derived,
        eee_records=len(selected),
        evaluation_results=canonical["result_count"],
    )

    contents: dict[str, bytes] = {**preview_bytes, **eee_contents}
    contents["reviewed/derived-manifest.json"] = derived_manifest_content
    contents["reviewed/public-summary.json"] = canonical_json_bytes(summary)
    contents["reviewed/canonical-results.json"] = canonical_json_bytes(canonical)
    contents["reviewed/canonical-results.html"] = _render_results_html(canonical).encode("utf-8")
    try:
        private_evidence_texts = collect_private_evidence_texts(
            source_root
        ) | collect_private_evidence_texts(review_root)
        evidence_binding = private_evidence_set_binding(private_evidence_texts)
        _assert_no_private_evidence_in_contents(contents, private_evidence_texts)
    except PublicDevelopmentPreviewError as error:
        raise PublicReviewedDevelopmentBundleError(
            "public payload failed the sealed evidence nonreproduction gate"
        ) from error
    payload_files = [_artifact(path, content) for path, content in sorted(contents.items())]
    evidence_attestation = _private_evidence_attestation(
        evidence_binding=evidence_binding,
        source_binding=source_binding,
        review_lock_sha256=derived.review_lock_sha256,
        decisions_sha256=derived.decisions_sha256,
        payload_files=payload_files,
    )
    manifest = _manifest(
        bundle_id=bundle_id,
        source_binding=source_binding,
        preview_manifest=preview_manifest,
        preview_verification=preview_verification_before,
        derived=derived,
        derived_manifest_sha256=sha256_bytes(derived_manifest_content),
        summary_sha256=sha256_bytes(contents["reviewed/public-summary.json"]),
        canonical_sha256=sha256_bytes(contents["reviewed/canonical-results.json"]),
        html_sha256=sha256_bytes(contents["reviewed/canonical-results.html"]),
        eee_files=eee_file_manifest,
        payload_files=payload_files,
        private_evidence_attestation=evidence_attestation,
        publication_state=publication_state,
        result_count=canonical["result_count"],
    )
    contents["publication-manifest.json"] = canonical_json_bytes(manifest)
    contents["README.md"] = _render_readme(manifest).encode("utf-8")
    verification = _verification(manifest=manifest, payload_files=payload_files)
    contents["verification.json"] = canonical_json_bytes(verification)
    contents["SHA256SUMS"] = _checksums(contents)
    _assert_no_private_evidence_in_contents(contents, private_evidence_texts)

    try:
        preview_verification_after = verify_public_development_preview(preview_root)
        contextual_after = verify_contextual_derived_run(
            derived_root,
            run_root=source_root,
            review_root=review_root,
            schema_path=DEFAULT_EEE_SCHEMA_PATH,
            schema_sha256=EEE_SCHEMA_SHA256,
        )
    except (PublicDevelopmentPreviewError, ReviewedExportError) as error:
        raise PublicReviewedDevelopmentBundleError(
            "bundle inputs changed during construction"
        ) from error
    if (
        preview_verification_after != preview_verification_before
        or _preview_contents(preview_root) != preview_bytes
        or contextual_after != contextual_before
        or _read_regular(derived_root / DERIVED_MANIFEST_NAME, "derived manifest")
        != derived_manifest_content
        or any(
            _read_regular(derived_root / source_path, "derived EEE record") != content
            for source_path, _, content, _ in selected
        )
    ):
        raise PublicReviewedDevelopmentBundleError("bundle inputs changed during construction")
    return _publish_tree(destination, contents)


__all__ = [
    "PUBLIC_REVIEWED_CANONICAL_RESULTS_SCHEMA_VERSION",
    "PUBLIC_REVIEWED_DEVELOPMENT_BUNDLE_SCHEMA_VERSION",
    "PUBLIC_REVIEWED_DEVELOPMENT_VERIFICATION_SCHEMA_VERSION",
    "PublicReviewedDevelopmentBundleError",
    "build_public_reviewed_development_bundle",
    "verify_public_reviewed_development_bundle",
]
