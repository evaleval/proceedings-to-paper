"""Private human workflow for the immutable 53-row control annotation packet.

The prepared packet is an all-null, manifest-pinned template.  This module never edits
that packet.  It creates two isolated working bundles, validates mutable completed
responses without comparing them to blank-template hashes, locks exact-byte originals
before comparison, computes pre-adjudication agreement, and prepares a separate
adjudication subset.

All outputs contain private source text or human decisions and must stay under ignored,
access-controlled storage.  Nothing here writes references or assigns a scientific label.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.evaluation.control_annotations import (
    AdjudicationRecord,
    AgreementSummary,
    AnnotationConfidence,
    AnnotationItem,
    AnnotationPacketManifest,
    AnnotationResponse,
    ResultBearing,
    ResultOrigin,
    measure_agreement,
    validate_initial_packet,
)
from proceedings_to_eee.io import (
    atomic_write_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from proceedings_to_eee.sources.manifest import (
    SourceManifest,
    SourceRole,
    resolve_cached_path,
)

WORKSPACE_SCHEMA_VERSION = "control-row-annotation-workspace/0.1"
COMPLETION_SCHEMA_VERSION = "control-row-annotation-completion/0.1"
SELECTION_SCHEMA_VERSION = "control-row-adjudication-selection/0.1"
TASK_SCHEMA_VERSION = "control-row-adjudication-task/0.1"
ADJUDICATION_WORKSPACE_SCHEMA_VERSION = "control-row-adjudication-workspace/0.1"

OPEN_DEVELOPMENT_SCOPE = (
    "These 53 selected rows are open development evidence for row-level "
    "result-bearing and origin. They do not establish extraction precision, "
    "joint-tuple accuracy, or generalization."
)

PRACTICE_EXAMPLE_ID = "practice_open_development_outside_53"
PRACTICE_EXAMPLE_APPENDIX = f"""

## One synthetic practice example (no answer key)

This invented example is not one of the 53 selected rows. It is excluded from item IDs,
response validation, agreement, adjudication, and the evaluation denominator.

- Practice ID: `{PRACTICE_EXAMPLE_ID}`
- Paper: *Synthetic Practice Paper* (invented fixture)
- Page/table: page 1, Table 1
- Source excerpt: `System Cedar      0.61          0.72`
- Nearby context: `Table 1 reports two illustrative scores for the synthetic system.`

Rehearse the process only: inspect the row and context, decide whether it is
result-bearing, decide who produced the result, copy a short exact supporting excerpt,
and record confidence. No decision or correct answer is supplied here. Do not add this
practice ID or any practice response to `items.jsonl` or `response.jsonl`.
"""

# Immutable workspace protocols prepared before the synthetic practice fixture was
# substituted remain valid by exact digest. Keeping only the digest preserves existing
# annotation receipts without retaining the superseded source-derived example in code.
_SUPPORTED_LEGACY_PROTOCOL_SHA256 = frozenset(
    {"95d7954d25fb59a7a2c70b9ac6b1b1ce75c675278865cef202e98d1724377689"}
)

_HEX_64 = r"^[0-9a-f]{64}$"
_ANNOTATOR_PATTERN = r"^annotator-[a-z0-9][a-z0-9-]*$"
_ADJUDICATOR_PATTERN = r"^adjudicator-[a-z0-9][a-z0-9-]*$"


class RelativeFile(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_HEX_64)
    records: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def confined_path(self) -> RelativeFile:
        _relative_path(self.path)
        return self


class FrozenPdfCopy(RelativeFile):
    paper_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    byte_size: int = Field(ge=1)


class AnnotatorBundle(StrictModel):
    annotator_id: str = Field(pattern=_ANNOTATOR_PATTERN)
    directory: str = Field(min_length=1)
    protocol: RelativeFile
    items: RelativeFile
    mutable_response_path: str
    initial_blank_response_sha256: str = Field(pattern=_HEX_64)
    pdfs: list[FrozenPdfCopy] = Field(min_length=1)

    @model_validator(mode="after")
    def relative_paths(self) -> AnnotatorBundle:
        _relative_path(self.directory)
        _relative_path(self.mutable_response_path)
        return self


class PracticeExample(StrictModel):
    practice_id: Literal["practice_open_development_outside_53"] = PRACTICE_EXAMPLE_ID
    included_in_items: Literal[False] = False
    included_in_denominator: Literal[False] = False
    answer_key_supplied: Literal[False] = False
    location: Literal["annotator protocol appendix"] = "annotator protocol appendix"


class WorkspacePrivacy(StrictModel):
    contains_source_text: Literal[True] = True
    contains_human_labels: Literal[False] = False
    contains_pseudonymous_ids: Literal[True] = True
    contains_local_paths: Literal[False] = False
    private_ignored_uncommitted_required: Literal[True] = True


class AnnotationWorkspaceManifest(StrictModel):
    schema_version: Literal["control-row-annotation-workspace/0.1"] = WORKSPACE_SCHEMA_VERSION
    status: Literal["prepared-unlabeled-working-copies"] = "prepared-unlabeled-working-copies"
    source_packet_manifest_sha256: str = Field(pattern=_HEX_64)
    source_run_id: str = Field(min_length=1)
    item_count: int = Field(ge=1)
    evaluation_denominator: int = Field(ge=1)
    annotators: list[str] = Field(min_length=2, max_length=2)
    coordinator_readme: RelativeFile
    bundles: list[AnnotatorBundle] = Field(min_length=2, max_length=2)
    practice_example: PracticeExample = Field(default_factory=PracticeExample)
    scientific_scope: Literal[OPEN_DEVELOPMENT_SCOPE] = OPEN_DEVELOPMENT_SCOPE
    privacy: WorkspacePrivacy = Field(default_factory=WorkspacePrivacy)

    @model_validator(mode="after")
    def workspace_invariants(self) -> AnnotationWorkspaceManifest:
        if len(set(self.annotators)) != 2:
            raise ValueError("workspace requires two distinct pseudonymous annotators")
        if [bundle.annotator_id for bundle in self.bundles] != self.annotators:
            raise ValueError("bundle order must match annotator order")
        if self.item_count != self.evaluation_denominator:
            raise ValueError("practice examples cannot enter the evaluation denominator")
        return self


class CompletedResponseFile(StrictModel):
    annotator_id: str = Field(pattern=_ANNOTATOR_PATTERN)
    status: Literal["complete"] = "complete"
    file: RelativeFile


class AdjudicationReason(StrEnum):
    JOINT_DISPOSITION_DISAGREEMENT = "joint_disposition_disagreement"
    ANNOTATOR_A_LOW_CONFIDENCE = "annotator_a_low_confidence"
    ANNOTATOR_B_LOW_CONFIDENCE = "annotator_b_low_confidence"
    ANNOTATOR_A_UNCERTAIN_RESULT_BEARING = "annotator_a_uncertain_result_bearing"
    ANNOTATOR_B_UNCERTAIN_RESULT_BEARING = "annotator_b_uncertain_result_bearing"
    ANNOTATOR_A_UNCERTAIN_ORIGIN = "annotator_a_uncertain_origin"
    ANNOTATOR_B_UNCERTAIN_ORIGIN = "annotator_b_uncertain_origin"


class AdjudicationSelection(StrictModel):
    schema_version: Literal["control-row-adjudication-selection/0.1"] = SELECTION_SCHEMA_VERSION
    item_id: str = Field(pattern=r"^ann_[0-9a-f]{20}$")
    reasons: list[AdjudicationReason] = Field(min_length=1)


class CompletionPrivacy(StrictModel):
    contains_source_text: Literal[True] = True
    contains_human_labels: Literal[True] = True
    contains_pseudonymous_ids: Literal[True] = True
    contains_local_paths: Literal[False] = False
    private_ignored_uncommitted_required: Literal[True] = True


class AnnotationCompletionManifest(StrictModel):
    schema_version: Literal["control-row-annotation-completion/0.1"] = COMPLETION_SCHEMA_VERSION
    status: Literal["independent-responses-complete"] = "independent-responses-complete"
    source_packet_manifest_sha256: str = Field(pattern=_HEX_64)
    source_workspace_manifest_sha256: str = Field(pattern=_HEX_64)
    item_count: int = Field(ge=1)
    evaluation_denominator: int = Field(ge=1)
    annotators: list[str] = Field(min_length=2, max_length=2)
    responses: list[CompletedResponseFile] = Field(min_length=2, max_length=2)
    agreement: RelativeFile
    adjudication_selection: RelativeFile
    adjudication_required_count: int = Field(ge=0)
    adjudication_reason_counts: dict[str, int]
    originals_preserved_before_comparison: Literal[True] = True
    comparison_basis: Literal["locked exact-byte response copies"] = (
        "locked exact-byte response copies"
    )
    practice_example_in_denominator: Literal[False] = False
    scientific_scope: Literal[OPEN_DEVELOPMENT_SCOPE] = OPEN_DEVELOPMENT_SCOPE
    privacy: CompletionPrivacy = Field(default_factory=CompletionPrivacy)

    @model_validator(mode="after")
    def completion_invariants(self) -> AnnotationCompletionManifest:
        if len(set(self.annotators)) != 2:
            raise ValueError("completion requires two distinct annotators")
        if [item.annotator_id for item in self.responses] != self.annotators:
            raise ValueError("response order must match completion annotator order")
        if self.item_count != self.evaluation_denominator:
            raise ValueError("completion denominator must exclude the practice example")
        if self.adjudication_selection.records != self.adjudication_required_count:
            raise ValueError("adjudication selection count does not match manifest")
        return self


class AdjudicationTask(StrictModel):
    schema_version: Literal["control-row-adjudication-task/0.1"] = TASK_SCHEMA_VERSION
    item: AnnotationItem
    annotator_a_response: AnnotationResponse
    annotator_b_response: AnnotationResponse
    reasons: list[AdjudicationReason] = Field(min_length=1)


class AdjudicationWorkspaceManifest(StrictModel):
    schema_version: Literal["control-row-adjudication-workspace/0.1"] = (
        ADJUDICATION_WORKSPACE_SCHEMA_VERSION
    )
    status: Literal["prepared-unlabeled-adjudication"] = "prepared-unlabeled-adjudication"
    source_packet_manifest_sha256: str = Field(pattern=_HEX_64)
    source_completion_manifest_sha256: str = Field(pattern=_HEX_64)
    annotators: list[str] = Field(min_length=2, max_length=2)
    adjudication_required_count: int = Field(ge=0)
    protocol: RelativeFile
    tasks: RelativeFile
    mutable_response_path: str
    initial_blank_response_sha256: str = Field(pattern=_HEX_64)
    pdfs: list[FrozenPdfCopy]
    originals_remain_in_completion_bundle: Literal[True] = True
    scientific_scope: Literal[OPEN_DEVELOPMENT_SCOPE] = OPEN_DEVELOPMENT_SCOPE
    privacy: CompletionPrivacy = Field(default_factory=CompletionPrivacy)

    @model_validator(mode="after")
    def adjudication_paths(self) -> AdjudicationWorkspaceManifest:
        _relative_path(self.mutable_response_path)
        if self.tasks.records != self.adjudication_required_count:
            raise ValueError("adjudication task count does not match manifest")
        return self


@dataclass(frozen=True)
class SourceDocument:
    paper_id: str
    source_id: str
    source_sha256: str
    path: Path
    byte_size: int
    page_text_by_number: dict[int, str]


@dataclass(frozen=True)
class ValidatedResponse:
    annotator_id: str
    sha256: str
    raw_bytes: bytes
    responses: tuple[AnnotationResponse, ...]


@dataclass(frozen=True)
class ValidatedCompletion:
    manifest: AnnotationCompletionManifest
    agreement: AgreementSummary
    responses: tuple[ValidatedResponse, ValidatedResponse]
    selections: tuple[AdjudicationSelection, ...]


@dataclass(frozen=True)
class ValidatedAdjudication:
    adjudicator_id: str | None
    sha256: str
    record_count: int
    complete: bool


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("workflow path must be a normalized relative path")
    return path


def _confined_regular_file(root: Path, value: str, label: str) -> Path:
    """Resolve one declared file without accepting symlinks or tree escapes."""

    relative = _relative_path(value)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label}: artifact root must be a regular directory")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"{label}: symlinked artifact path is forbidden")
    if not path.is_file():
        raise ValueError(f"{label}: artifact file is unavailable")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label}: artifact path escapes its private root") from error
    return path


def _require_private_output(output_dir: Path, project_root: Path) -> None:
    """Keep generated labels and source text under the repository's ignored private root."""

    project = project_root.resolve()
    private_root = (project / "runs" / "private").resolve()
    target = output_dir.resolve()
    if not private_root.is_relative_to(project):
        raise ValueError("private output root escapes the project")
    if target == private_root or not target.is_relative_to(private_root):
        raise ValueError("annotation workflow output must be below runs/private")


def _require_disjoint_output(output_dir: Path, *input_roots: Path) -> None:
    target = output_dir.resolve()
    for input_root in input_roots:
        source = input_root.resolve()
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError("annotation output must be disjoint from every input tree")


def _normalized(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\u2212", "-")
    return re.sub(r"\s+", " ", normalized).strip()


def _jsonl_bytes(values: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for value in values
    )


def _parse_jsonl_bytes(content: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label}: response is not valid UTF-8") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{label}:{line_number}: blank JSONL line")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label}:{line_number}: invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number}: JSONL row must be an object")
        rows.append(value)
    return rows


def _load_items(packet_dir: Path) -> tuple[AnnotationPacketManifest, list[AnnotationItem]]:
    manifest = validate_initial_packet(packet_dir)
    rows = _parse_jsonl_bytes((packet_dir / "items.jsonl").read_bytes(), "items.jsonl")
    items = [AnnotationItem.model_validate(row) for row in rows]
    if len(items) != manifest.item_count:
        raise ValueError("packet item count changed after initial validation")
    if len(items) != 53:
        raise ValueError("control annotation workflow requires the pinned 53-item denominator")
    return manifest, items


def _source_documents(
    items: list[AnnotationItem], *, run_root: Path, project_root: Path
) -> dict[str, SourceDocument]:
    documents: dict[str, SourceDocument] = {}
    papers = list(dict.fromkeys(item.paper_id for item in items))
    for paper_id in papers:
        paper_items = [item for item in items if item.paper_id == paper_id]
        manifest_path = run_root / paper_id / "source-manifest.json"
        layout_path = run_root / paper_id / "private" / "layout.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"{paper_id}: frozen source manifest is unavailable")
        if not layout_path.is_file() or layout_path.is_symlink():
            raise ValueError(f"{paper_id}: frozen layout is unavailable")
        expected_manifest_hashes = {item.source_manifest_sha256 for item in paper_items}
        expected_layout_hashes = {item.layout_sha256 for item in paper_items}
        if expected_manifest_hashes != {sha256_file(manifest_path)}:
            raise ValueError(f"{paper_id}: source manifest hash differs from annotation items")
        if expected_layout_hashes != {sha256_file(layout_path)}:
            raise ValueError(f"{paper_id}: layout hash differs from annotation items")

        manifest = SourceManifest.model_validate(read_json(manifest_path))
        if manifest.paper_id != paper_id:
            raise ValueError(f"{paper_id}: frozen source manifest belongs to another paper")
        source_ids = {item.source_id for item in paper_items}
        if len(source_ids) != 1:
            raise ValueError(f"{paper_id}: annotation items do not share one paper source")
        source_id = next(iter(source_ids))
        matches = [
            source
            for source in manifest.sources
            if source.role is SourceRole.PAPER and source.source_id == source_id
        ]
        if len(matches) != 1:
            raise ValueError(f"{paper_id}: paper source is not unique in frozen manifest")
        source = matches[0]
        expected_source_hashes = {item.source_sha256 for item in paper_items}
        if source.sha256 is None or expected_source_hashes != {source.sha256}:
            raise ValueError(f"{paper_id}: source hash differs from annotation items")
        source_path = resolve_cached_path(source, project_root)
        if (
            source_path.is_symlink()
            or not source_path.is_file()
            or sha256_file(source_path) != source.sha256
            or source.byte_size != source_path.stat().st_size
        ):
            raise ValueError(f"{paper_id}: cached frozen PDF differs from source manifest")

        layout = read_json(layout_path)
        if not isinstance(layout, dict) or layout.get("source_id") != source_id:
            raise ValueError(f"{paper_id}: layout source does not match annotation items")
        pages = layout.get("pages")
        if not isinstance(pages, list):
            raise ValueError(f"{paper_id}: layout pages are unavailable")
        page_by_number: dict[int, dict[str, Any]] = {}
        page_text_by_number: dict[int, str] = {}
        for page in pages:
            if not isinstance(page, dict):
                raise ValueError(f"{paper_id}: layout page is not an object")
            page_number = page.get("page")
            page_text = page.get("text")
            if not isinstance(page_number, int) or not isinstance(page_text, str):
                raise ValueError(f"{paper_id}: layout page is incomplete")
            if page.get("text_sha256") != sha256_bytes(page_text.encode("utf-8")):
                raise ValueError(f"{paper_id}: layout page text digest is invalid")
            if page_number in page_text_by_number:
                raise ValueError(f"{paper_id}: layout page number is duplicated")
            page_by_number[page_number] = page
            page_text_by_number[page_number] = page_text
        for item in paper_items:
            page = page_by_number.get(item.page)
            if page is None or page.get("text_sha256") != item.page_text_sha256:
                raise ValueError(f"{paper_id}: cited page hash differs from annotation item")
            page_text = page.get("text")
            if not isinstance(page_text, str) or _normalized(
                item.exact_evidence
            ) not in _normalized(page_text):
                raise ValueError(f"{paper_id}: annotation evidence is absent from cited page")

        documents[paper_id] = SourceDocument(
            paper_id=paper_id,
            source_id=source_id,
            source_sha256=source.sha256,
            path=source_path,
            byte_size=source_path.stat().st_size,
            page_text_by_number=page_text_by_number,
        )
    return documents


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("human handoff paths must remain inside the project root") from error


def _coordinator_readme(
    *, packet_dir: Path, run_root: Path, output_dir: Path, project_root: Path
) -> str:
    packet = _project_relative(packet_dir, project_root)
    run = _project_relative(run_root, project_root)
    workspace = _project_relative(output_dir, project_root)
    completion = f"{workspace}-completion"
    adjudication = f"{workspace}-adjudication"
    return f"""# Human annotation handoff

This is a private coordinator workspace. Do not commit, force-add, email, or place it in a
shared/cloud-synced directory. {OPEN_DEVELOPMENT_SCOPE}

Run every coordinator command below from the repository root. Preserve the
`completion_manifest_sha256` printed by the lock command as an external receipt; pass it
unchanged to every later step.

## Give each annotator only their own bundle

- Annotator A receives `{workspace}/annotator-a/` only.
- Annotator B receives `{workspace}/annotator-b/` only.
- Each bundle contains only read-only `protocol.md`, read-only `items.jsonl`, writable
  `response.jsonl`, and read-only copies of the seven frozen PDFs.
- Never provide the packet manifest, peer response, proposals, candidates, scores,
  references, layouts, completion files, or pseudonym lookup.

The generated file modes are a guard against accidents, not an access-control proof.
Use separate accounts/ACLs or separate read-only mounts for the inputs. The coordinator
must attest that `annotator-a` and `annotator-b` are two different people and that the
pseudonyms contain no names, emails, or other identity information.

The protocol contains one no-answer-key practice example outside the 53. It is not in
`items.jsonl` and validators keep the denominator exactly 53.

## Exact commands

Validate each returned response independently; this validates schema, all 53 ordered item
IDs, source fingerprints, and verbatim evidence, but deliberately does not expect the
blank-template response hash:

```bash
.venv/bin/ere validate-control-annotation-response {workspace} \\
  --packet {packet} --run-root {run} --annotator annotator-a
.venv/bin/ere validate-control-annotation-response {workspace} \\
  --packet {packet} --run-root {run} --annotator annotator-b
```

After both files validate, preserve exact-byte locked copies and create the separate
completion manifest before any comparison:

```bash
.venv/bin/ere lock-control-annotation-responses {workspace} \\
  --packet {packet} --run-root {run} --output {completion}
COMPLETION_MANIFEST_SHA256=PASTE_SHA256_FROM_LOCK_OUTPUT
```

Revalidate the locked originals and print only aggregate pre-adjudication agreement:

```bash
.venv/bin/ere measure-control-annotation-agreement {completion} \\
  --workspace {workspace} --packet {packet} --run-root {run} \\
  --completion-manifest-sha256 "$COMPLETION_MANIFEST_SHA256"
```

Prepare a separate adjudication workspace containing only disagreements, low-confidence
decisions, and uncertain cases. This never overwrites either primary response:

```bash
.venv/bin/ere prepare-control-annotation-adjudication {completion} \\
  --workspace {workspace} --packet {packet} --run-root {run} \\
  --completion-manifest-sha256 "$COMPLETION_MANIFEST_SHA256" \\
  --output {adjudication}
```

After a third person completes `{adjudication}/response.jsonl`, validate it without
changing the two locked originals:

```bash
.venv/bin/ere validate-control-annotation-adjudication {adjudication} \\
  --completion {completion} --workspace {workspace} --packet {packet} --run-root {run} \\
  --completion-manifest-sha256 "$COMPLETION_MANIFEST_SHA256"
```

Undefined Cohen's kappa is emitted as the string `undefined`, never as zero. Agreement is
pre-adjudication and its exact denominator is the 53 selected rows. Labels, evidence,
pseudonyms, adjudication, and local paths remain private and uncommitted.
"""


def _build_tree(output_dir: Path, builder: Any) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing private output: {output_dir.name}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent)
    )
    tree = temporary_parent / "tree"
    tree.mkdir(mode=0o700)
    try:
        builder(tree)
        tree.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary_parent, ignore_errors=True)
        raise
    temporary_parent.rmdir()


def _copy_read_only(source: Path, destination: Path) -> str:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"refusing non-regular source file: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    digest = sha256_file(destination)
    if digest != sha256_file(source):
        raise OSError(f"copied file hash mismatch: {destination.name}")
    destination.chmod(0o444)
    return digest


def _bundle_expected_files(bundle: AnnotatorBundle) -> set[str]:
    return {
        bundle.protocol.path,
        bundle.items.path,
        bundle.mutable_response_path,
        *(pdf.path for pdf in bundle.pdfs),
    }


def prepare_annotation_workspace(
    *, packet_dir: Path, run_root: Path, project_root: Path, output_dir: Path
) -> tuple[AnnotationWorkspaceManifest, str]:
    """Create two label-free, non-overlapping annotator bundles from the pinned packet."""

    _require_private_output(output_dir, project_root)
    _require_disjoint_output(output_dir, packet_dir, run_root)
    packet, items = _load_items(packet_dir)
    documents = _source_documents(items, run_root=run_root, project_root=project_root)
    if len(documents) != 7:
        raise ValueError(f"expected seven frozen paper PDFs, found {len(documents)}")
    packet_sha = sha256_file(packet_dir / "manifest.json")
    protocol_bytes = (packet_dir / "protocol.md").read_bytes()
    derived_protocol = protocol_bytes + PRACTICE_EXAMPLE_APPENDIX.encode("utf-8")
    items_bytes = (packet_dir / "items.jsonl").read_bytes()
    readme = _coordinator_readme(
        packet_dir=packet_dir,
        run_root=run_root,
        output_dir=output_dir,
        project_root=project_root,
    )

    result: dict[str, Any] = {}

    def build(tree: Path) -> None:
        readme_sha = sha256_bytes(readme.encode("utf-8"))
        atomic_write_bytes(tree / "README.md", readme.encode("utf-8"))
        (tree / "README.md").chmod(0o600)
        bundles: list[AnnotatorBundle] = []
        for annotator in packet.annotators:
            bundle_root = tree / annotator
            input_root = bundle_root / "input"
            pdf_root = input_root / "pdfs"
            work_root = bundle_root / "work"
            pdf_root.mkdir(parents=True, mode=0o755)
            work_root.mkdir(parents=True, mode=0o700)
            protocol_path = input_root / "protocol.md"
            items_path = input_root / "items.jsonl"
            response_path = work_root / "response.jsonl"
            atomic_write_bytes(protocol_path, derived_protocol)
            atomic_write_bytes(items_path, items_bytes)
            protocol_path.chmod(0o444)
            items_path.chmod(0o444)
            blank_source = packet_dir / "responses" / f"{annotator}.jsonl"
            atomic_write_bytes(response_path, blank_source.read_bytes())
            response_path.chmod(0o600)
            pdfs: list[FrozenPdfCopy] = []
            for paper_id, document in documents.items():
                relative = f"{annotator}/input/pdfs/{paper_id}.pdf"
                destination = tree / relative
                digest = _copy_read_only(document.path, destination)
                pdfs.append(
                    FrozenPdfCopy(
                        path=relative,
                        sha256=digest,
                        paper_id=paper_id,
                        source_id=document.source_id,
                        byte_size=document.byte_size,
                    )
                )
            pdf_root.chmod(0o555)
            input_root.chmod(0o555)
            bundle_root.chmod(0o700)
            bundles.append(
                AnnotatorBundle(
                    annotator_id=annotator,
                    directory=annotator,
                    protocol=RelativeFile(
                        path=f"{annotator}/input/protocol.md",
                        sha256=sha256_bytes(derived_protocol),
                    ),
                    items=RelativeFile(
                        path=f"{annotator}/input/items.jsonl",
                        sha256=sha256_bytes(items_bytes),
                        records=len(items),
                    ),
                    mutable_response_path=f"{annotator}/work/response.jsonl",
                    initial_blank_response_sha256=sha256_file(response_path),
                    pdfs=pdfs,
                )
            )
        manifest = AnnotationWorkspaceManifest(
            source_packet_manifest_sha256=packet_sha,
            source_run_id=run_root.name,
            item_count=len(items),
            evaluation_denominator=len(items),
            annotators=packet.annotators,
            coordinator_readme=RelativeFile(path="README.md", sha256=readme_sha),
            bundles=bundles,
        )
        manifest_sha = write_json(tree / "workspace-manifest.json", manifest)
        (tree / "workspace-manifest.json").chmod(0o600)
        result.update(manifest=manifest, manifest_sha=manifest_sha)

    _build_tree(output_dir, build)
    manifest = result["manifest"]
    validate_annotation_workspace(
        packet_dir=packet_dir,
        workspace_dir=output_dir,
        run_root=run_root,
        project_root=project_root,
    )
    return manifest, result["manifest_sha"]


def validate_annotation_workspace(
    *, packet_dir: Path, workspace_dir: Path, run_root: Path, project_root: Path
) -> AnnotationWorkspaceManifest:
    """Validate immutable bundle inputs while deliberately ignoring mutable response hashes."""

    packet, items = _load_items(packet_dir)
    documents = _source_documents(items, run_root=run_root, project_root=project_root)
    expected_protocol_sha = sha256_bytes(
        (packet_dir / "protocol.md").read_bytes() + PRACTICE_EXAMPLE_APPENDIX.encode("utf-8")
    )
    accepted_protocol_hashes = {
        expected_protocol_sha,
        *_SUPPORTED_LEGACY_PROTOCOL_SHA256,
    }
    expected_items_sha = sha256_file(packet_dir / "items.jsonl")
    expected_readme_sha = sha256_bytes(
        _coordinator_readme(
            packet_dir=packet_dir,
            run_root=run_root,
            output_dir=workspace_dir,
            project_root=project_root,
        ).encode("utf-8")
    )
    manifest_path = _confined_regular_file(
        workspace_dir, "workspace-manifest.json", "workspace manifest"
    )
    manifest = AnnotationWorkspaceManifest.model_validate(read_json(manifest_path))
    if manifest.source_packet_manifest_sha256 != sha256_file(packet_dir / "manifest.json"):
        raise ValueError("workspace source packet hash mismatch")
    if manifest.annotators != packet.annotators or manifest.item_count != len(items):
        raise ValueError("workspace annotator or item contract differs from packet")
    if (
        manifest.coordinator_readme.path != "README.md"
        or manifest.coordinator_readme.sha256 != expected_readme_sha
    ):
        raise ValueError("workspace coordinator README differs from deterministic handoff")
    readme_path = _confined_regular_file(
        workspace_dir, manifest.coordinator_readme.path, "workspace README"
    )
    if sha256_file(readme_path) != manifest.coordinator_readme.sha256:
        raise ValueError("workspace coordinator README hash mismatch")
    for bundle in manifest.bundles:
        expected_directory = bundle.annotator_id
        if (
            bundle.directory != expected_directory
            or bundle.protocol.path != f"{expected_directory}/input/protocol.md"
            or bundle.protocol.sha256 not in accepted_protocol_hashes
            or bundle.items.path != f"{expected_directory}/input/items.jsonl"
            or bundle.items.sha256 != expected_items_sha
            or bundle.items.records != len(items)
            or bundle.mutable_response_path != f"{expected_directory}/work/response.jsonl"
            or bundle.initial_blank_response_sha256
            != sha256_file(packet_dir / "responses" / f"{bundle.annotator_id}.jsonl")
        ):
            raise ValueError(f"{bundle.annotator_id}: bundle differs from deterministic handoff")
        expected = _bundle_expected_files(bundle)
        bundle_root = workspace_dir / _relative_path(bundle.directory)
        actual = {
            path.relative_to(workspace_dir).as_posix()
            for path in bundle_root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual != expected:
            raise ValueError(f"{bundle.annotator_id}: bundle contains unauthorized files")
        for declared in (bundle.protocol, bundle.items, *bundle.pdfs):
            path = _confined_regular_file(
                workspace_dir, declared.path, f"{bundle.annotator_id} immutable input"
            )
            if sha256_file(path) != declared.sha256:
                raise ValueError(f"{bundle.annotator_id}: immutable bundle input changed")
        _confined_regular_file(
            workspace_dir,
            bundle.mutable_response_path,
            f"{bundle.annotator_id} mutable response",
        )
        expected_pdfs = {
            (paper_id, document.source_id, document.source_sha256, document.byte_size)
            for paper_id, document in documents.items()
        }
        declared_pdfs = {
            (pdf.paper_id, pdf.source_id, pdf.sha256, pdf.byte_size) for pdf in bundle.pdfs
        }
        if declared_pdfs != expected_pdfs:
            raise ValueError(f"{bundle.annotator_id}: frozen PDF set differs from packet sources")
    return manifest


def _evidence_occurs_on_cited_page(
    evidence: str, item: AnnotationItem, document: SourceDocument
) -> bool:
    needle = _normalized(evidence)
    page_text = document.page_text_by_number.get(item.page)
    return bool(needle) and page_text is not None and needle in _normalized(page_text)


def validate_completed_response(
    *,
    packet_dir: Path,
    response_path: Path,
    expected_annotator: str,
    run_root: Path,
    project_root: Path,
) -> ValidatedResponse:
    """Validate one completed mutable response without consulting blank response hashes."""

    packet, items = _load_items(packet_dir)
    if expected_annotator not in packet.annotators:
        raise ValueError("expected annotator is not declared by the packet")
    template_paths = {
        (packet_dir / "responses" / f"{annotator}.jsonl").resolve()
        for annotator in packet.annotators
    }
    if response_path.resolve() in template_paths:
        raise ValueError("completed response must not be the manifest-pinned blank template")
    if response_path.is_symlink() or not response_path.is_file():
        raise ValueError("completed response must be a regular file")
    raw = response_path.read_bytes()
    rows = _parse_jsonl_bytes(raw, response_path.name)
    responses = [AnnotationResponse.model_validate(row) for row in rows]
    expected_ids = [item.item_id for item in items]
    actual_ids = [response.item_id for response in responses]
    if actual_ids != expected_ids:
        raise ValueError("completed response IDs must exactly match packet order and membership")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("completed response contains duplicate item IDs")
    if any(response.annotator != expected_annotator for response in responses):
        raise ValueError("completed response is bound to the wrong or mixed annotator")
    if any(response.is_blank for response in responses):
        raise ValueError("completed response still contains blank decisions")
    documents = _source_documents(items, run_root=run_root, project_root=project_root)
    item_by_id = {item.item_id: item for item in items}
    for response in responses:
        assert response.exact_evidence is not None
        item = item_by_id[response.item_id]
        if not _evidence_occurs_on_cited_page(
            response.exact_evidence, item, documents[item.paper_id]
        ):
            raise ValueError(f"{response.item_id}: exact evidence is absent from cited frozen page")
    return ValidatedResponse(
        annotator_id=expected_annotator,
        sha256=sha256_bytes(raw),
        raw_bytes=raw,
        responses=tuple(responses),
    )


def validate_workspace_response(
    *,
    packet_dir: Path,
    workspace_dir: Path,
    annotator: str,
    run_root: Path,
    project_root: Path,
) -> ValidatedResponse:
    manifest = validate_annotation_workspace(
        packet_dir=packet_dir,
        workspace_dir=workspace_dir,
        run_root=run_root,
        project_root=project_root,
    )
    bundles = {bundle.annotator_id: bundle for bundle in manifest.bundles}
    if annotator not in bundles:
        raise ValueError("annotator has no isolated working bundle")
    return validate_completed_response(
        packet_dir=packet_dir,
        response_path=workspace_dir / _relative_path(bundles[annotator].mutable_response_path),
        expected_annotator=annotator,
        run_root=run_root,
        project_root=project_root,
    )


def _joint_disposition(response: AnnotationResponse) -> str:
    if response.is_blank:
        raise ValueError("joint disposition requires a completed response")
    if response.result_bearing is ResultBearing.NO:
        return "not_result"
    if response.result_bearing is ResultBearing.UNCERTAIN:
        return "uncertain_result_bearing"
    assert response.origin is not None
    return f"result/{response.origin.value}"


def build_adjudication_selection(
    items: list[AnnotationItem],
    left: tuple[AnnotationResponse, ...],
    right: tuple[AnnotationResponse, ...],
) -> list[AdjudicationSelection]:
    if [item.item_id for item in items] != [response.item_id for response in left]:
        raise ValueError("annotator A response order differs from items")
    if [item.item_id for item in items] != [response.item_id for response in right]:
        raise ValueError("annotator B response order differs from items")
    selected: list[AdjudicationSelection] = []
    for item, left_response, right_response in zip(items, left, right, strict=True):
        reasons: list[AdjudicationReason] = []
        if _joint_disposition(left_response) != _joint_disposition(right_response):
            reasons.append(AdjudicationReason.JOINT_DISPOSITION_DISAGREEMENT)
        if left_response.confidence is AnnotationConfidence.LOW:
            reasons.append(AdjudicationReason.ANNOTATOR_A_LOW_CONFIDENCE)
        if right_response.confidence is AnnotationConfidence.LOW:
            reasons.append(AdjudicationReason.ANNOTATOR_B_LOW_CONFIDENCE)
        if left_response.result_bearing is ResultBearing.UNCERTAIN:
            reasons.append(AdjudicationReason.ANNOTATOR_A_UNCERTAIN_RESULT_BEARING)
        if right_response.result_bearing is ResultBearing.UNCERTAIN:
            reasons.append(AdjudicationReason.ANNOTATOR_B_UNCERTAIN_RESULT_BEARING)
        if left_response.origin is ResultOrigin.UNCERTAIN:
            reasons.append(AdjudicationReason.ANNOTATOR_A_UNCERTAIN_ORIGIN)
        if right_response.origin is ResultOrigin.UNCERTAIN:
            reasons.append(AdjudicationReason.ANNOTATOR_B_UNCERTAIN_ORIGIN)
        if reasons:
            selected.append(AdjudicationSelection(item_id=item.item_id, reasons=reasons))
    return selected


def lock_completed_responses(
    *,
    packet_dir: Path,
    workspace_dir: Path,
    run_root: Path,
    project_root: Path,
    output_dir: Path,
) -> tuple[AnnotationCompletionManifest, AgreementSummary, str]:
    """Lock exact completed originals before comparing them and write completion metadata."""

    _require_private_output(output_dir, project_root)
    _require_disjoint_output(output_dir, packet_dir, workspace_dir, run_root)
    workspace = validate_annotation_workspace(
        packet_dir=packet_dir,
        workspace_dir=workspace_dir,
        run_root=run_root,
        project_root=project_root,
    )
    validated = [
        validate_workspace_response(
            packet_dir=packet_dir,
            workspace_dir=workspace_dir,
            annotator=annotator,
            run_root=run_root,
            project_root=project_root,
        )
        for annotator in workspace.annotators
    ]
    packet, items = _load_items(packet_dir)
    result: dict[str, Any] = {}

    def build(tree: Path) -> None:
        response_descriptors: list[CompletedResponseFile] = []
        locked: list[ValidatedResponse] = []
        for response in validated:
            relative = f"responses/{response.annotator_id}.jsonl"
            destination = tree / relative
            atomic_write_bytes(destination, response.raw_bytes)
            destination.chmod(0o444)
            if destination.read_bytes() != response.raw_bytes:
                raise OSError("locked response differs from completed original")
            locked_response = validate_completed_response(
                packet_dir=packet_dir,
                response_path=destination,
                expected_annotator=response.annotator_id,
                run_root=run_root,
                project_root=project_root,
            )
            if locked_response.sha256 != response.sha256:
                raise OSError("locked response hash differs from completed original")
            locked.append(locked_response)
            response_descriptors.append(
                CompletedResponseFile(
                    annotator_id=response.annotator_id,
                    file=RelativeFile(
                        path=relative,
                        sha256=response.sha256,
                        records=len(response.responses),
                    ),
                )
            )

        agreement = measure_agreement(list(locked[0].responses), list(locked[1].responses))
        agreement_sha = write_json(tree / "agreement.json", agreement)
        selections = build_adjudication_selection(items, locked[0].responses, locked[1].responses)
        selection_sha = write_jsonl(
            tree / "adjudication-selection.jsonl",
            [selection.model_dump(mode="json") for selection in selections],
        )
        reason_counts = Counter(
            reason.value for selection in selections for reason in selection.reasons
        )
        manifest = AnnotationCompletionManifest(
            source_packet_manifest_sha256=sha256_file(packet_dir / "manifest.json"),
            source_workspace_manifest_sha256=sha256_file(workspace_dir / "workspace-manifest.json"),
            item_count=len(items),
            evaluation_denominator=len(items),
            annotators=packet.annotators,
            responses=response_descriptors,
            agreement=RelativeFile(path="agreement.json", sha256=agreement_sha),
            adjudication_selection=RelativeFile(
                path="adjudication-selection.jsonl",
                sha256=selection_sha,
                records=len(selections),
            ),
            adjudication_required_count=len(selections),
            adjudication_reason_counts={
                reason.value: reason_counts[reason.value] for reason in AdjudicationReason
            },
        )
        manifest_sha = write_json(tree / "completion-manifest.json", manifest)
        result.update(
            manifest=manifest,
            agreement=agreement,
            manifest_sha=manifest_sha,
        )

    _build_tree(output_dir, build)
    validate_completion_bundle(
        packet_dir=packet_dir,
        workspace_dir=workspace_dir,
        completion_dir=output_dir,
        run_root=run_root,
        project_root=project_root,
        expected_manifest_sha256=result["manifest_sha"],
    )
    return result["manifest"], result["agreement"], result["manifest_sha"]


def validate_completion_bundle(
    *,
    packet_dir: Path,
    workspace_dir: Path,
    completion_dir: Path,
    run_root: Path,
    project_root: Path,
    expected_manifest_sha256: str,
) -> ValidatedCompletion:
    validate_annotation_workspace(
        packet_dir=packet_dir,
        workspace_dir=workspace_dir,
        run_root=run_root,
        project_root=project_root,
    )
    packet, items = _load_items(packet_dir)
    manifest_path = _confined_regular_file(
        completion_dir, "completion-manifest.json", "completion manifest"
    )
    if not re.fullmatch(_HEX_64, expected_manifest_sha256):
        raise ValueError("expected completion manifest SHA-256 is invalid")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("completion manifest differs from the external lock receipt")
    manifest = AnnotationCompletionManifest.model_validate(read_json(manifest_path))
    expected_files = {
        "completion-manifest.json",
        manifest.agreement.path,
        manifest.adjudication_selection.path,
        *(descriptor.file.path for descriptor in manifest.responses),
    }
    actual_files = {
        path.relative_to(completion_dir).as_posix()
        for path in completion_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError("completion bundle contains unauthorized files")
    if manifest.source_packet_manifest_sha256 != sha256_file(packet_dir / "manifest.json"):
        raise ValueError("completion source packet hash mismatch")
    if manifest.source_workspace_manifest_sha256 != sha256_file(
        workspace_dir / "workspace-manifest.json"
    ):
        raise ValueError("completion source workspace hash mismatch")
    if manifest.annotators != packet.annotators or manifest.item_count != len(items):
        raise ValueError("completion item or annotator contract differs from packet")

    responses: list[ValidatedResponse] = []
    for descriptor in manifest.responses:
        path = _confined_regular_file(
            completion_dir, descriptor.file.path, "locked completed response"
        )
        if sha256_file(path) != descriptor.file.sha256:
            raise ValueError("locked completed response hash mismatch")
        response = validate_completed_response(
            packet_dir=packet_dir,
            response_path=path,
            expected_annotator=descriptor.annotator_id,
            run_root=run_root,
            project_root=project_root,
        )
        if descriptor.file.records != len(response.responses):
            raise ValueError("locked response count differs from completion manifest")
        responses.append(response)

    agreement_path = _confined_regular_file(
        completion_dir, manifest.agreement.path, "agreement artifact"
    )
    if sha256_file(agreement_path) != manifest.agreement.sha256:
        raise ValueError("agreement artifact hash mismatch")
    stored_agreement = AgreementSummary.model_validate(read_json(agreement_path))
    agreement = measure_agreement(list(responses[0].responses), list(responses[1].responses))
    if stored_agreement != agreement:
        raise ValueError("stored agreement differs from locked response originals")

    selection_path = _confined_regular_file(
        completion_dir,
        manifest.adjudication_selection.path,
        "adjudication selection artifact",
    )
    if sha256_file(selection_path) != manifest.adjudication_selection.sha256:
        raise ValueError("adjudication selection hash mismatch")
    selection_rows = _parse_jsonl_bytes(selection_path.read_bytes(), selection_path.name)
    selections = [AdjudicationSelection.model_validate(row) for row in selection_rows]
    expected_selections = build_adjudication_selection(
        items, responses[0].responses, responses[1].responses
    )
    if selections != expected_selections:
        raise ValueError("adjudication selection differs from locked response originals")
    if manifest.adjudication_required_count != len(selections):
        raise ValueError("adjudication required count differs from selection")
    reason_counts = Counter(
        reason.value for selection in selections for reason in selection.reasons
    )
    expected_reason_counts = {
        reason.value: reason_counts[reason.value] for reason in AdjudicationReason
    }
    if manifest.adjudication_reason_counts != expected_reason_counts:
        raise ValueError("adjudication reason counts differ from selection")
    return ValidatedCompletion(
        manifest=manifest,
        agreement=agreement,
        responses=(responses[0], responses[1]),
        selections=tuple(selections),
    )


def _adjudication_protocol(required_count: int) -> str:
    return f"""# Adjudication protocol

This private workspace contains {required_count} of the 53 open-development rows. A row
appears only because the locked independent responses disagree on the joint disposition,
one response is low-confidence, or one response is uncertain. The two locked originals
remain unchanged in the completion bundle.

A third person must inspect the frozen paper, both responses, and the listed reason codes.
Complete every row in `response.jsonl`; preserve genuine uncertainty rather than forcing a
binary answer. Use a pseudonym matching `adjudicator-...`, quote exact source evidence, and
write a concise rationale. Do not edit `tasks.jsonl`, either primary response, or the 53-row
packet. {OPEN_DEVELOPMENT_SCOPE}
"""


def prepare_adjudication_workspace(
    *,
    packet_dir: Path,
    workspace_dir: Path,
    completion_dir: Path,
    run_root: Path,
    project_root: Path,
    output_dir: Path,
    expected_completion_manifest_sha256: str,
) -> tuple[AdjudicationWorkspaceManifest, str]:
    _require_private_output(output_dir, project_root)
    _require_disjoint_output(output_dir, packet_dir, workspace_dir, completion_dir, run_root)
    completion = validate_completion_bundle(
        packet_dir=packet_dir,
        workspace_dir=workspace_dir,
        completion_dir=completion_dir,
        run_root=run_root,
        project_root=project_root,
        expected_manifest_sha256=expected_completion_manifest_sha256,
    )
    packet, items = _load_items(packet_dir)
    item_by_id = {item.item_id: item for item in items}
    left_by_id = {item.item_id: item for item in completion.responses[0].responses}
    right_by_id = {item.item_id: item for item in completion.responses[1].responses}
    documents = _source_documents(items, run_root=run_root, project_root=project_root)
    tasks = [
        AdjudicationTask(
            item=item_by_id[selection.item_id],
            annotator_a_response=left_by_id[selection.item_id],
            annotator_b_response=right_by_id[selection.item_id],
            reasons=selection.reasons,
        )
        for selection in completion.selections
    ]
    selected_paper_ids = list(dict.fromkeys(task.item.paper_id for task in tasks))
    selected_documents = {paper_id: documents[paper_id] for paper_id in selected_paper_ids}
    blanks = [AdjudicationRecord(item_id=task.item.item_id) for task in tasks]
    protocol = _adjudication_protocol(len(tasks))
    result: dict[str, Any] = {}

    def build(tree: Path) -> None:
        protocol_sha = sha256_bytes(protocol.encode("utf-8"))
        atomic_write_bytes(tree / "protocol.md", protocol.encode("utf-8"))
        (tree / "protocol.md").chmod(0o444)
        tasks_sha = write_jsonl(
            tree / "tasks.jsonl", [task.model_dump(mode="json") for task in tasks]
        )
        (tree / "tasks.jsonl").chmod(0o444)
        response_sha = write_jsonl(
            tree / "response.jsonl",
            [record.model_dump(mode="json") for record in blanks],
        )
        (tree / "response.jsonl").chmod(0o600)
        pdfs: list[FrozenPdfCopy] = []
        pdf_root = tree / "pdfs"
        pdf_root.mkdir(mode=0o755)
        for paper_id, document in selected_documents.items():
            relative = f"pdfs/{paper_id}.pdf"
            digest = _copy_read_only(document.path, tree / relative)
            pdfs.append(
                FrozenPdfCopy(
                    path=relative,
                    sha256=digest,
                    paper_id=paper_id,
                    source_id=document.source_id,
                    byte_size=document.byte_size,
                )
            )
        pdf_root.chmod(0o555)
        manifest = AdjudicationWorkspaceManifest(
            source_packet_manifest_sha256=sha256_file(packet_dir / "manifest.json"),
            source_completion_manifest_sha256=sha256_file(
                completion_dir / "completion-manifest.json"
            ),
            annotators=packet.annotators,
            adjudication_required_count=len(tasks),
            protocol=RelativeFile(path="protocol.md", sha256=protocol_sha),
            tasks=RelativeFile(path="tasks.jsonl", sha256=tasks_sha, records=len(tasks)),
            mutable_response_path="response.jsonl",
            initial_blank_response_sha256=response_sha,
            pdfs=pdfs,
        )
        manifest_sha = write_json(tree / "adjudication-manifest.json", manifest)
        result.update(manifest=manifest, manifest_sha=manifest_sha)

    _build_tree(output_dir, build)
    validate_adjudication_workspace(
        packet_dir=packet_dir,
        workspace_dir=workspace_dir,
        completion_dir=completion_dir,
        adjudication_dir=output_dir,
        run_root=run_root,
        project_root=project_root,
        expected_completion_manifest_sha256=expected_completion_manifest_sha256,
        require_complete=False,
    )
    return result["manifest"], result["manifest_sha"]


def validate_adjudication_workspace(
    *,
    packet_dir: Path,
    workspace_dir: Path,
    completion_dir: Path,
    adjudication_dir: Path,
    run_root: Path,
    project_root: Path,
    expected_completion_manifest_sha256: str,
    require_complete: bool = True,
) -> ValidatedAdjudication:
    completion = validate_completion_bundle(
        packet_dir=packet_dir,
        workspace_dir=workspace_dir,
        completion_dir=completion_dir,
        run_root=run_root,
        project_root=project_root,
        expected_manifest_sha256=expected_completion_manifest_sha256,
    )
    _, items = _load_items(packet_dir)
    documents = _source_documents(items, run_root=run_root, project_root=project_root)
    item_by_id = {item.item_id: item for item in items}
    expected_ids = [selection.item_id for selection in completion.selections]
    expected_protocol_sha = sha256_bytes(_adjudication_protocol(len(expected_ids)).encode("utf-8"))
    expected_blank_sha = sha256_bytes(
        _jsonl_bytes(
            [
                AdjudicationRecord(item_id=item_id).model_dump(mode="json")
                for item_id in expected_ids
            ]
        )
    )
    manifest_path = _confined_regular_file(
        adjudication_dir, "adjudication-manifest.json", "adjudication manifest"
    )
    manifest = AdjudicationWorkspaceManifest.model_validate(read_json(manifest_path))
    if manifest.source_packet_manifest_sha256 != sha256_file(packet_dir / "manifest.json"):
        raise ValueError("adjudication source packet hash mismatch")
    if manifest.source_completion_manifest_sha256 != sha256_file(
        completion_dir / "completion-manifest.json"
    ):
        raise ValueError("adjudication source completion hash mismatch")
    if manifest.annotators != completion.manifest.annotators:
        raise ValueError("adjudication annotators differ from completion")
    if (
        manifest.protocol.path != "protocol.md"
        or manifest.protocol.sha256 != expected_protocol_sha
        or manifest.tasks.path != "tasks.jsonl"
        or manifest.tasks.records != len(expected_ids)
        or manifest.mutable_response_path != "response.jsonl"
        or manifest.initial_blank_response_sha256 != expected_blank_sha
    ):
        raise ValueError("adjudication manifest differs from deterministic handoff")
    for declared in (manifest.protocol, manifest.tasks, *manifest.pdfs):
        path = _confined_regular_file(
            adjudication_dir, declared.path, "immutable adjudication input"
        )
        if sha256_file(path) != declared.sha256:
            raise ValueError("immutable adjudication input hash mismatch")
    expected_files = {
        "adjudication-manifest.json",
        manifest.protocol.path,
        manifest.tasks.path,
        manifest.mutable_response_path,
        *(pdf.path for pdf in manifest.pdfs),
    }
    actual_files = {
        path.relative_to(adjudication_dir).as_posix()
        for path in adjudication_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError("adjudication workspace contains unauthorized files")
    task_rows = _parse_jsonl_bytes(
        _confined_regular_file(
            adjudication_dir, manifest.tasks.path, "adjudication tasks"
        ).read_bytes(),
        "tasks.jsonl",
    )
    tasks = [AdjudicationTask.model_validate(row) for row in task_rows]
    left_by_id = {response.item_id: response for response in completion.responses[0].responses}
    right_by_id = {response.item_id: response for response in completion.responses[1].responses}
    expected_tasks = [
        AdjudicationTask(
            item=item_by_id[selection.item_id],
            annotator_a_response=left_by_id[selection.item_id],
            annotator_b_response=right_by_id[selection.item_id],
            reasons=selection.reasons,
        )
        for selection in completion.selections
    ]
    if tasks != expected_tasks:
        raise ValueError("adjudication tasks differ from completion selection")
    if manifest.adjudication_required_count != len(tasks):
        raise ValueError("adjudication task count differs from completion selection")
    selected_paper_ids = list(dict.fromkeys(task.item.paper_id for task in expected_tasks))
    expected_pdfs = [
        (
            paper_id,
            documents[paper_id].source_id,
            documents[paper_id].source_sha256,
            documents[paper_id].byte_size,
        )
        for paper_id in selected_paper_ids
    ]
    declared_pdfs = [
        (pdf.paper_id, pdf.source_id, pdf.sha256, pdf.byte_size) for pdf in manifest.pdfs
    ]
    if declared_pdfs != expected_pdfs:
        raise ValueError("adjudication PDF set differs from selected frozen sources")
    response_path = _confined_regular_file(
        adjudication_dir, manifest.mutable_response_path, "adjudication response"
    )
    raw = response_path.read_bytes()
    rows = _parse_jsonl_bytes(raw, "response.jsonl")
    records = [AdjudicationRecord.model_validate(row) for row in rows]
    if [record.item_id for record in records] != expected_ids:
        raise ValueError("adjudication response IDs must exactly match selected task order")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("adjudication selection contains duplicate item IDs")
    blank = [record for record in records if record.is_blank]
    if require_complete and blank:
        raise ValueError("completed adjudication still contains blank decisions")
    completed = [record for record in records if not record.is_blank]
    adjudicators = {record.adjudicator for record in completed}
    if completed and (len(adjudicators) != 1 or None in adjudicators):
        raise ValueError("adjudication must use one pseudonymous adjudicator")
    adjudicator_id = next(iter(adjudicators)) if adjudicators else None
    if adjudicator_id in completion.manifest.annotators:
        raise ValueError("adjudicator pseudonym must differ from both annotators")
    for record in completed:
        assert record.exact_evidence is not None
        item = item_by_id[record.item_id]
        if not _evidence_occurs_on_cited_page(
            record.exact_evidence, item, documents[item.paper_id]
        ):
            raise ValueError(f"{record.item_id}: adjudication evidence is absent from cited page")
    return ValidatedAdjudication(
        adjudicator_id=adjudicator_id,
        sha256=sha256_bytes(raw),
        record_count=len(records),
        complete=not blank,
    )
