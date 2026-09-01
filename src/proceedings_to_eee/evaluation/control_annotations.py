"""Private, label-free annotation packets for proposed control-table rows.

The proposal worklist is useful for selecting bounded annotation units, but it also
contains rule names and reasons that can bias a human decision.  This module projects
only source-bound evidence into a private packet, creates independent blank response
files, and keeps adjudication separate from both original responses.

No function in this module writes reference annotations or assigns a scientific label.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.io import (
    atomic_write_bytes,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)

PACKET_SCHEMA_VERSION = "control-row-annotation-packet/0.1"
ITEM_SCHEMA_VERSION = "control-row-annotation-item/0.1"
RESPONSE_SCHEMA_VERSION = "control-row-annotation-response/0.1"
ADJUDICATION_SCHEMA_VERSION = "control-row-adjudication/0.1"
AGREEMENT_SCHEMA_VERSION = "control-row-agreement/0.2"
PROTOCOL_VERSION = "control-row-origin-annotation/0.1"

DEFAULT_ANNOTATORS = ("annotator-a", "annotator-b")
_PROPOSAL_HINT_FIELDS = frozenset({"confirmed", "expected_claim_type", "reason", "rule_id"})
_HEX_64 = r"^[0-9a-f]{64}$"
JOINT_CATEGORY_ORDER = (
    "not_result",
    "uncertain_result_bearing",
    "result/paper_produced",
    "result/externally_sourced",
    "result/uncertain",
)


PROTOCOL_TEXT = f"""# Control-row result and origin annotation protocol

Protocol version: `{PROTOCOL_VERSION}`

## Boundary and blinding

The annotation unit is one table row selected before annotation. Two annotators label
every item independently. They may inspect the frozen local paper around the cited page,
table, headers, caption, methods, and citations, but must not inspect reference YAML,
extraction candidates, proposal rules or reasons, scores, or the other annotator's work.
The item file contains source text and remains private. Never copy it into a public report.

## Fields

`result_bearing` is `yes`, `no`, or `uncertain`.

- `yes`: at least one cell reports a quantitative evaluation outcome or measured behavior.
- `no`: the row is only setup, method metadata, a sample count, a heading, or other
  non-result material.
- `uncertain`: the available source context does not support either decision safely, or the
  row mixes incompatible units that require adjudication.

`origin` is `paper_produced`, `externally_sourced`, or `uncertain`. It must be null when
`result_bearing` is `no`, and non-null otherwise. A value computed or collected by the
current paper is `paper_produced`, including a baseline rerun by its authors. A value copied
or cited from another paper, vendor, or leaderboard without a current-paper rerun is
`externally_sourced`. System ownership is not result origin. Do not infer origin merely from
the model or dataset name.

`exact_evidence` must be a short verbatim source excerpt that supports the decision, not a
paraphrase. `confidence` is `high`, `medium`, or `low`. Low confidence and either uncertain
state must be adjudicated. Initial response and adjudication files deliberately contain
nulls; null is not a label and must not be replaced until a human performs the review.

## Independent annotation and adjudication

Complete and lock both response files before comparing them. Preserve both originals.
A third person adjudicates every disagreement, low-confidence decision, and uncertain state
against the frozen source. Adjudication is written separately and never overwrites either
annotator's response. Genuine source insufficiency may remain uncertain; do not force a
binary decision.

## Agreement

Measure agreement before adjudication. The primary measure is unweighted Cohen's kappa on
the joint nominal disposition over all items: `not_result`, `uncertain_result_bearing`,
`result/paper_produced`, `result/externally_sourced`, and `result/uncertain`. Also report raw
agreement, the full confusion matrix, category counts, and the denominator. If expected
agreement is one and kappa is undefined, report `undefined`, never zero. Any field-specific
agreement must state its denominator explicitly.

## Privacy and handoff

Individual labels, exact evidence, notes, row names, and local paths remain private and
uncommitted. A public artifact may contain aggregate counts and agreement only after an
explicit privacy projection. Nothing in this packet is copied into `references/` without a
separate, authorized, completed adjudication step.
"""


class ResultBearing(StrEnum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class ResultOrigin(StrEnum):
    PAPER_PRODUCED = "paper_produced"
    EXTERNALLY_SOURCED = "externally_sourced"
    UNCERTAIN = "uncertain"


class AnnotationConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnnotationItem(StrictModel):
    """One source-bound row shown to both annotators, with no proposed answer."""

    schema_version: Literal["control-row-annotation-item/0.1"] = ITEM_SCHEMA_VERSION
    item_id: str = Field(pattern=r"^ann_[0-9a-f]{20}$")
    paper_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    page: int = Field(ge=1)
    table: str = Field(min_length=1)
    row: str = Field(min_length=1)
    block_id: str = Field(min_length=1)
    exact_evidence: str = Field(min_length=1, max_length=2_000)
    exact_evidence_sha256: str = Field(pattern=_HEX_64)
    source_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_HEX_64)
    source_manifest_sha256: str = Field(pattern=_HEX_64)
    layout_sha256: str = Field(pattern=_HEX_64)
    page_text_sha256: str = Field(pattern=_HEX_64)

    @model_validator(mode="after")
    def evidence_digest_matches(self) -> AnnotationItem:
        digest = hashlib.sha256(self.exact_evidence.encode("utf-8")).hexdigest()
        if self.exact_evidence_sha256 != digest:
            raise ValueError("exact_evidence_sha256 does not match exact_evidence")
        identity = {
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "paper_id": self.paper_id,
            "page": self.page,
            "table": self.table,
            "row": self.row,
            "block_id": self.block_id,
            "exact_evidence_sha256": self.exact_evidence_sha256,
        }
        if self.item_id != _item_id(identity):
            raise ValueError("item_id does not match the source-bound annotation identity")
        return self


class AnnotationResponse(StrictModel):
    """One annotator's response. A fully null decision is the only blank state."""

    schema_version: Literal["control-row-annotation-response/0.1"] = RESPONSE_SCHEMA_VERSION
    item_id: str = Field(pattern=r"^ann_[0-9a-f]{20}$")
    annotator: str = Field(pattern=r"^annotator-[a-z0-9][a-z0-9-]*$")
    result_bearing: ResultBearing | None = None
    origin: ResultOrigin | None = None
    exact_evidence: str | None = Field(default=None, min_length=1, max_length=2_000)
    confidence: AnnotationConfidence | None = None

    @model_validator(mode="after")
    def enforce_null_and_completed_states(self) -> AnnotationResponse:
        if self.result_bearing is None:
            blank_fields = (self.origin, self.exact_evidence, self.confidence)
            if any(value is not None for value in blank_fields):
                raise ValueError("a blank response must leave every decision field null")
            return self
        if self.exact_evidence is None or self.confidence is None:
            raise ValueError("a completed response requires exact_evidence and confidence")
        if self.result_bearing is ResultBearing.NO:
            if self.origin is not None:
                raise ValueError("origin must be null when result_bearing is no")
        elif self.origin is None:
            raise ValueError("origin is required unless result_bearing is no")
        return self

    @property
    def is_blank(self) -> bool:
        return self.result_bearing is None


class AdjudicationRecord(StrictModel):
    """A separate final decision that cannot overwrite either original response."""

    schema_version: Literal["control-row-adjudication/0.1"] = ADJUDICATION_SCHEMA_VERSION
    item_id: str = Field(pattern=r"^ann_[0-9a-f]{20}$")
    adjudicator: str | None = Field(default=None, pattern=r"^adjudicator-[a-z0-9][a-z0-9-]*$")
    result_bearing: ResultBearing | None = None
    origin: ResultOrigin | None = None
    exact_evidence: str | None = Field(default=None, min_length=1, max_length=2_000)
    rationale: str | None = Field(default=None, min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def enforce_blank_or_completed_state(self) -> AdjudicationRecord:
        decision = (
            self.adjudicator,
            self.result_bearing,
            self.origin,
            self.exact_evidence,
            self.rationale,
        )
        if all(value is None for value in decision):
            return self
        if self.adjudicator is None or self.result_bearing is None:
            raise ValueError("partial adjudication is not allowed")
        if self.exact_evidence is None or self.rationale is None:
            raise ValueError("completed adjudication requires exact evidence and rationale")
        if self.result_bearing is ResultBearing.NO:
            if self.origin is not None:
                raise ValueError("origin must be null when adjudicated result_bearing is no")
        elif self.origin is None:
            raise ValueError("adjudicated origin is required unless result_bearing is no")
        return self

    @property
    def is_blank(self) -> bool:
        return all(
            value is None
            for value in (
                self.adjudicator,
                self.result_bearing,
                self.origin,
                self.exact_evidence,
                self.rationale,
            )
        )


class PacketFile(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_HEX_64)
    records: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def path_is_relative_and_confined(self) -> PacketFile:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != self.path:
            raise ValueError("packet file path must be a normalized relative path")
        return self


class PacketPrivacy(StrictModel):
    contains_source_text: Literal[True] = True
    contains_item_labels: Literal[False] = False
    contains_proposal_hints: Literal[False] = False
    contains_local_paths: Literal[False] = False
    private_uncommitted_required: Literal[True] = True


class AnnotationPacketManifest(StrictModel):
    schema_version: Literal["control-row-annotation-packet/0.1"] = PACKET_SCHEMA_VERSION
    status: Literal["prepared-unlabeled"] = "prepared-unlabeled"
    protocol_version: Literal["control-row-origin-annotation/0.1"] = PROTOCOL_VERSION
    source_run_id: str = Field(min_length=1)
    source_proposals_sha256: str = Field(pattern=_HEX_64)
    item_count: int = Field(ge=1)
    paper_count: int = Field(ge=1)
    annotators: list[str] = Field(min_length=2, max_length=2)
    files: list[PacketFile]
    privacy: PacketPrivacy = Field(default_factory=PacketPrivacy)

    @model_validator(mode="after")
    def manifest_invariants(self) -> AnnotationPacketManifest:
        if len(set(self.annotators)) != 2:
            raise ValueError("the two annotator identifiers must be distinct")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest file paths must be unique")
        return self


class AgreementSummary(StrictModel):
    schema_version: Literal["control-row-agreement/0.2"] = AGREEMENT_SCHEMA_VERSION
    pre_adjudication: Literal[True] = True
    denominator: int = Field(ge=1)
    row_annotator: str = Field(pattern=r"^annotator-[a-z0-9][a-z0-9-]*$")
    column_annotator: str = Field(pattern=r"^annotator-[a-z0-9][a-z0-9-]*$")
    raw_agreement_count: int = Field(ge=0)
    raw_agreement: float = Field(ge=0.0, le=1.0)
    expected_agreement: float = Field(ge=0.0, le=1.0)
    cohen_kappa: float | Literal["undefined"]
    kappa_status: Literal["defined", "undefined_no_expected_variance"]
    categories: list[str]
    annotator_category_counts: dict[str, dict[str, int]]
    confusion_matrix: dict[str, dict[str, int]]
    confusion_matrix_counts: list[list[int]]


def _normalized(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\u2212", "-")
    return re.sub(r"\s+", " ", normalized).strip()


def _item_id(payload: dict[str, Any]) -> str:
    digest = sha256_bytes(canonical_json_bytes(payload))
    return f"ann_{digest[:20]}"


def _page(layout: dict[str, Any], page_number: int, paper_id: str) -> dict[str, Any]:
    pages = layout.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"{paper_id}: layout has no pages list")
    matches = [item for item in pages if isinstance(item, dict) and item.get("page") == page_number]
    if len(matches) != 1:
        raise ValueError(f"{paper_id}: page {page_number} is not unique in the layout")
    return matches[0]


def _source_fingerprint(
    manifest: dict[str, Any], layout: dict[str, Any], paper_id: str
) -> tuple[str, str]:
    if manifest.get("paper_id") != paper_id:
        raise ValueError(f"{paper_id}: source manifest paper_id does not match its run folder")
    source_id = layout.get("source_id")
    sources = manifest.get("sources")
    if not isinstance(source_id, str) or not isinstance(sources, list):
        raise ValueError(f"{paper_id}: source manifest/layout is incomplete")
    matches = [
        source
        for source in sources
        if isinstance(source, dict)
        and source.get("source_id") == source_id
        and source.get("role") == "paper"
    ]
    if len(matches) != 1:
        raise ValueError(f"{paper_id}: layout source is not unique in source manifest")
    source_sha256 = matches[0].get("sha256")
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError(f"{paper_id}: source SHA-256 is invalid")
    return source_id, source_sha256


def _load_paper_fingerprints(run_root: Path, paper_id: str) -> dict[str, Any]:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", paper_id) is None:
        raise ValueError("annotation paper_id must be a canonical path-safe identifier")
    paper_root = run_root / paper_id
    manifest_path = paper_root / "source-manifest.json"
    layout_path = paper_root / "private" / "layout.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not layout_path.is_file()
        or layout_path.is_symlink()
    ):
        raise ValueError(f"{paper_id}: source manifest or private layout is missing")
    manifest = read_json(manifest_path)
    layout = read_json(layout_path)
    if not isinstance(manifest, dict) or not isinstance(layout, dict):
        raise ValueError(f"{paper_id}: source manifest/layout must be JSON objects")
    source_id, source_sha256 = _source_fingerprint(manifest, layout, paper_id)
    return {
        "manifest": manifest,
        "layout": layout,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "source_manifest_sha256": sha256_file(manifest_path),
        "layout_sha256": sha256_file(layout_path),
    }


def build_annotation_items(
    proposal_document: dict[str, Any],
    *,
    run_root: Path,
    expected_items: int = 53,
) -> list[AnnotationItem]:
    """Project proposal rows into source-bound items without proposal hints or labels."""

    rows = proposal_document.get("rows_needing_a_human_label")
    if not isinstance(rows, list):
        raise ValueError("proposal document has no rows_needing_a_human_label list")
    if len(rows) != expected_items:
        raise ValueError(f"expected {expected_items} proposal rows, found {len(rows)}")

    fingerprints: dict[str, dict[str, Any]] = {}
    items: list[AnnotationItem] = []
    for proposal in rows:
        if not isinstance(proposal, dict):
            raise ValueError("every proposal row must be an object")
        paper_id = proposal.get("paper_id")
        if not isinstance(paper_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]*", paper_id) is None:
            raise ValueError("proposal row has no canonical path-safe paper_id")
        fingerprint = fingerprints.setdefault(
            paper_id, _load_paper_fingerprints(run_root, paper_id)
        )
        page_number = proposal.get("page")
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
            raise ValueError(f"{paper_id}: proposal page is invalid")
        page = _page(fingerprint["layout"], page_number, paper_id)
        page_text = page.get("text")
        if not isinstance(page_text, str):
            raise ValueError(f"{paper_id}: page {page_number} has no layout text")
        page_digest = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
        recorded_page_digest = page.get("text_sha256")
        if recorded_page_digest != page_digest:
            raise ValueError(f"{paper_id}: page {page_number} text SHA-256 is invalid")

        exact_evidence = proposal.get("exact_quote")
        table = proposal.get("label")
        row = proposal.get("row")
        block_id = proposal.get("block_id")
        required_text = (exact_evidence, table, row, block_id)
        if not all(isinstance(value, str) and value for value in required_text):
            raise ValueError(f"{paper_id}: proposal lacks table, row, block, or exact evidence")
        if _normalized(exact_evidence) not in _normalized(page_text):
            raise ValueError(
                f"{paper_id}: proposal evidence is absent from normalized page {page_number}"
            )

        identity = {
            "source_id": fingerprint["source_id"],
            "source_sha256": fingerprint["source_sha256"],
            "paper_id": paper_id,
            "page": page_number,
            "table": table,
            "row": row,
            "block_id": block_id,
            "exact_evidence_sha256": hashlib.sha256(exact_evidence.encode("utf-8")).hexdigest(),
        }
        items.append(
            AnnotationItem(
                item_id=_item_id(identity),
                paper_id=paper_id,
                page=page_number,
                table=table,
                row=row,
                block_id=block_id,
                exact_evidence=exact_evidence,
                exact_evidence_sha256=identity["exact_evidence_sha256"],
                source_id=fingerprint["source_id"],
                source_sha256=fingerprint["source_sha256"],
                source_manifest_sha256=fingerprint["source_manifest_sha256"],
                layout_sha256=fingerprint["layout_sha256"],
                page_text_sha256=page_digest,
            )
        )
    item_ids = [item.item_id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("annotation item IDs are not unique")
    return items


def _blank_responses(items: list[AnnotationItem], annotator: str) -> list[AnnotationResponse]:
    return [AnnotationResponse(item_id=item.item_id, annotator=annotator) for item in items]


def _blank_adjudications(items: list[AnnotationItem]) -> list[AdjudicationRecord]:
    return [AdjudicationRecord(item_id=item.item_id) for item in items]


def prepare_annotation_packet(
    *,
    proposals_path: Path,
    run_root: Path,
    output_dir: Path,
    expected_items: int = 53,
    annotators: tuple[str, str] = DEFAULT_ANNOTATORS,
) -> tuple[AnnotationPacketManifest, str]:
    """Write one deterministic, private, entirely unlabeled annotation bundle.

    Existing target files are never overwritten because they may contain human work.
    The returned digest is the SHA-256 of ``manifest.json``.
    """

    if len(set(annotators)) != 2:
        raise ValueError("two distinct annotator identifiers are required")
    targets = (
        output_dir / "manifest.json",
        output_dir / "protocol.md",
        output_dir / "items.jsonl",
        output_dir / "responses" / f"{annotators[0]}.jsonl",
        output_dir / "responses" / f"{annotators[1]}.jsonl",
        output_dir / "adjudication.jsonl",
    )
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite an existing annotation packet")

    proposal_document = read_json(proposals_path)
    if not isinstance(proposal_document, dict):
        raise ValueError("proposal worklist must be a JSON object")
    items = build_annotation_items(
        proposal_document, run_root=run_root, expected_items=expected_items
    )

    protocol_bytes = PROTOCOL_TEXT.encode("utf-8")
    atomic_write_bytes(output_dir / "protocol.md", protocol_bytes)
    item_sha = write_jsonl(
        output_dir / "items.jsonl", [item.model_dump(mode="json") for item in items]
    )
    response_files: list[PacketFile] = []
    for annotator in annotators:
        responses = _blank_responses(items, annotator)
        relative = f"responses/{annotator}.jsonl"
        digest = write_jsonl(
            output_dir / relative,
            [response.model_dump(mode="json") for response in responses],
        )
        response_files.append(PacketFile(path=relative, sha256=digest, records=len(responses)))
    adjudications = _blank_adjudications(items)
    adjudication_sha = write_jsonl(
        output_dir / "adjudication.jsonl",
        [record.model_dump(mode="json") for record in adjudications],
    )
    manifest = AnnotationPacketManifest(
        source_run_id=run_root.name,
        source_proposals_sha256=sha256_file(proposals_path),
        item_count=len(items),
        paper_count=len({item.paper_id for item in items}),
        annotators=list(annotators),
        files=[
            PacketFile(path="protocol.md", sha256=sha256_bytes(protocol_bytes), records=None),
            PacketFile(path="items.jsonl", sha256=item_sha, records=len(items)),
            *response_files,
            PacketFile(
                path="adjudication.jsonl",
                sha256=adjudication_sha,
                records=len(adjudications),
            ),
        ],
    )
    manifest_sha = write_json(output_dir / "manifest.json", manifest)
    validate_initial_packet(output_dir)
    return manifest, manifest_sha


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path.name}:{line_number}: blank JSONL line")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number}: JSONL row must be an object")
        rows.append(value)
    return rows


def validate_initial_packet(output_dir: Path) -> AnnotationPacketManifest:
    """Validate hashes, membership, privacy, and the all-null human boundary."""

    manifest_path = output_dir / "manifest.json"
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("annotation packet root must be a regular directory")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("annotation packet manifest must be a regular file")
    manifest = AnnotationPacketManifest.model_validate(read_json(manifest_path))
    expected_records = {
        "protocol.md": None,
        "items.jsonl": manifest.item_count,
        **{
            f"responses/{annotator}.jsonl": manifest.item_count for annotator in manifest.annotators
        },
        "adjudication.jsonl": manifest.item_count,
    }
    declared_records = {declared.path: declared.records for declared in manifest.files}
    if declared_records != expected_records:
        raise ValueError("annotation packet manifest has an unexpected file contract")
    for declared in manifest.files:
        path = output_dir
        for part in PurePosixPath(declared.path).parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"packet file must not use symlinks: {declared.path}")
        if not path.is_file() or sha256_file(path) != declared.sha256:
            raise ValueError(f"packet file hash mismatch: {declared.path}")
    if (output_dir / "protocol.md").read_bytes() != PROTOCOL_TEXT.encode("utf-8"):
        raise ValueError("annotation protocol differs from its declared version")
    item_rows = _read_jsonl(output_dir / "items.jsonl")
    if any(_PROPOSAL_HINT_FIELDS & set(row) for row in item_rows):
        raise ValueError("annotator items contain proposal hints")
    items = [AnnotationItem.model_validate(row) for row in item_rows]
    item_ids = [item.item_id for item in items]
    if (
        len(items) != manifest.item_count
        or len(item_ids) != len(set(item_ids))
        or len({item.paper_id for item in items}) != manifest.paper_count
    ):
        raise ValueError("item count or uniqueness does not match the manifest")
    for annotator in manifest.annotators:
        rows = _read_jsonl(output_dir / "responses" / f"{annotator}.jsonl")
        responses = [AnnotationResponse.model_validate(row) for row in rows]
        if [response.item_id for response in responses] != item_ids:
            raise ValueError(f"{annotator}: response membership does not match items")
        wrong_annotator = any(response.annotator != annotator for response in responses)
        if len(responses) != len(items) or wrong_annotator:
            raise ValueError(f"{annotator}: response count or annotator binding is invalid")
        if any(not response.is_blank for response in responses):
            raise ValueError(f"{annotator}: initial response file contains a label")
    adjudications = [
        AdjudicationRecord.model_validate(row)
        for row in _read_jsonl(output_dir / "adjudication.jsonl")
    ]
    if len(adjudications) != len(items) or [item.item_id for item in adjudications] != item_ids:
        raise ValueError("adjudication membership does not match items")
    if any(not item.is_blank for item in adjudications):
        raise ValueError("initial adjudication file contains a decision")
    if manifest.privacy != PacketPrivacy():
        raise ValueError("packet privacy declaration is not the required private state")
    return manifest


def _joint_disposition(response: AnnotationResponse) -> str:
    if response.is_blank:
        raise ValueError("agreement cannot be computed with blank responses")
    if response.result_bearing is ResultBearing.NO:
        return "not_result"
    if response.result_bearing is ResultBearing.UNCERTAIN:
        return "uncertain_result_bearing"
    assert response.origin is not None
    return f"result/{response.origin.value}"


def measure_agreement(
    left: list[AnnotationResponse], right: list[AnnotationResponse]
) -> AgreementSummary:
    """Compute pre-adjudication unweighted Cohen kappa on joint dispositions."""

    if not left or not right:
        raise ValueError("agreement requires two non-empty response sets")
    left_by_id = {item.item_id: item for item in left}
    right_by_id = {item.item_id: item for item in right}
    if len(left_by_id) != len(left) or len(right_by_id) != len(right):
        raise ValueError("response sets contain duplicate item IDs")
    if set(left_by_id) != set(right_by_id):
        raise ValueError("response sets cover different items")
    left_annotators = {item.annotator for item in left}
    right_annotators = {item.annotator for item in right}
    if len(left_annotators) != 1 or len(right_annotators) != 1:
        raise ValueError("each response set must contain exactly one annotator")
    if left_annotators & right_annotators:
        raise ValueError("agreement requires two distinct annotators")

    row_annotator = next(iter(left_annotators))
    column_annotator = next(iter(right_annotators))

    pairs = [
        (_joint_disposition(left_by_id[item_id]), _joint_disposition(right_by_id[item_id]))
        for item_id in sorted(left_by_id)
    ]
    categories = list(JOINT_CATEGORY_ORDER)
    denominator = len(pairs)
    confusion = {
        left_category: {
            right_category: sum(pair == (left_category, right_category) for pair in pairs)
            for right_category in categories
        }
        for left_category in categories
    }
    left_counts = Counter(pair[0] for pair in pairs)
    right_counts = Counter(pair[1] for pair in pairs)
    agreed = sum(left_category == right_category for left_category, right_category in pairs)
    observed = agreed / denominator
    expected_numerator = sum(
        left_counts[category] * right_counts[category] for category in categories
    )
    expected = expected_numerator / (denominator * denominator)
    if expected_numerator == denominator * denominator:
        kappa: float | Literal["undefined"] = "undefined"
        status = "undefined_no_expected_variance"
    else:
        kappa = round(
            (agreed * denominator - expected_numerator)
            / (denominator * denominator - expected_numerator),
            6,
        )
        status = "defined"
    return AgreementSummary(
        denominator=denominator,
        row_annotator=row_annotator,
        column_annotator=column_annotator,
        raw_agreement_count=agreed,
        raw_agreement=round(observed, 6),
        expected_agreement=round(expected, 6),
        cohen_kappa=kappa,
        kappa_status=status,
        categories=categories,
        annotator_category_counts={
            row_annotator: {category: left_counts[category] for category in categories},
            column_annotator: {category: right_counts[category] for category in categories},
        },
        confusion_matrix=confusion,
        confusion_matrix_counts=[
            [confusion[row_category][column_category] for column_category in categories]
            for row_category in categories
        ],
    )
