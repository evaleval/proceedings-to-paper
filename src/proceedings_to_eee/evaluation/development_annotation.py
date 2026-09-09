"""Private, mixed, single-reviewer development annotation handoffs.

This workflow is intentionally separate from the immutable two-reviewer control-row
study.  It accepts a coordinator-only mixed selection plan, projects away per-item
selection intent, and creates one label-free reviewer bundle.  Source excerpts, PDFs,
and human decisions are private artifacts and are never written inside the public
repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)

SELECTION_SCHEMA_VERSION = "mixed-development-annotation-selection/0.1"
ITEM_SCHEMA_VERSION = "mixed-development-annotation-item/0.1"
RESPONSE_SCHEMA_VERSION = "mixed-development-annotation-response/0.1"
PACKAGE_SCHEMA_VERSION = "mixed-development-annotation-package/0.1"
PROTOCOL_VERSION = "mixed-development-result-origin/0.1"

_HEX_64 = r"^[0-9a-f]{64}$"
_PAPER_ID = r"^[a-z0-9][a-z0-9-]*$"
_REVIEWER_ID = r"^reviewer-[a-z0-9][a-z0-9-]*$"


class SelectionIntent(StrEnum):
    """Coordinator-only sampling intent; never projected into reviewer items."""

    RESULT_PAPER_PRODUCED_CANDIDATE = "result_paper_produced_candidate"
    RESULT_EXTERNAL_OR_COPIED_CANDIDATE = "result_external_or_copied_candidate"
    RESULT_MIXED_OR_UNCERTAIN_ORIGIN_CANDIDATE = "result_mixed_or_uncertain_origin_candidate"
    NON_RESULT_HEADER = "non_result_header"
    NON_RESULT_SAMPLE_COUNT = "non_result_sample_count"
    NON_RESULT_SETUP_VALUE = "non_result_setup_value"
    NON_RESULT_PARAMETER = "non_result_parameter"
    NON_RESULT_THRESHOLD = "non_result_threshold"
    NON_RESULT_CAPTION = "non_result_caption"


REQUIRED_MIXED_INTENTS = frozenset(SelectionIntent)
RESULT_INTENTS = frozenset(
    {
        SelectionIntent.RESULT_PAPER_PRODUCED_CANDIDATE,
        SelectionIntent.RESULT_EXTERNAL_OR_COPIED_CANDIDATE,
        SelectionIntent.RESULT_MIXED_OR_UNCERTAIN_ORIGIN_CANDIDATE,
    }
)


class ResultBearing(StrEnum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class ResultOrigin(StrEnum):
    PAPER_PRODUCED = "paper_produced"
    EXTERNALLY_SOURCED = "externally_sourced"
    UNCERTAIN = "uncertain"


class ReviewConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DevelopmentSelectionRecord(StrictModel):
    paper_id: str = Field(pattern=_PAPER_ID)
    page: int = Field(ge=1)
    locator: str = Field(min_length=1, max_length=500)
    selection_excerpt: str = Field(min_length=1, max_length=2_000)
    selection_intent: SelectionIntent

    @model_validator(mode="after")
    def excerpt_is_one_physical_line(self) -> DevelopmentSelectionRecord:
        if "\n" in self.selection_excerpt or "\r" in self.selection_excerpt:
            raise ValueError("selection_excerpt must be a contiguous single-line excerpt")
        return self


class DevelopmentSelectionPlan(StrictModel):
    schema_version: Literal["mixed-development-annotation-selection/0.1"] = SELECTION_SCHEMA_VERSION
    source_scope: Literal["development-and-error-analysis-only"] = (
        "development-and-error-analysis-only"
    )
    future_unseen_data_used: Literal[False] = False
    source_run_id: str = Field(min_length=1)
    minimum_result_candidates: int = Field(ge=1)
    minimum_result_papers: int = Field(ge=1)
    required_intents: list[SelectionIntent] = Field(min_length=len(REQUIRED_MIXED_INTENTS))
    items: list[DevelopmentSelectionRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def mixed_contract_is_complete(self) -> DevelopmentSelectionPlan:
        required = set(self.required_intents)
        if len(required) != len(self.required_intents):
            raise ValueError("required_intents must not contain duplicates")
        if required != REQUIRED_MIXED_INTENTS:
            missing = sorted(intent.value for intent in REQUIRED_MIXED_INTENTS - required)
            extra = sorted(intent.value for intent in required - REQUIRED_MIXED_INTENTS)
            raise ValueError(f"mixed selection contract mismatch; missing={missing}, extra={extra}")
        counts = Counter(item.selection_intent for item in self.items)
        missing_items = sorted(intent.value for intent in required if counts[intent] == 0)
        if missing_items:
            raise ValueError(f"selection has no item for required intents: {missing_items}")
        identities = [
            (item.paper_id, item.page, item.locator, item.selection_excerpt) for item in self.items
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("selection records must be unique")
        result_items = [item for item in self.items if item.selection_intent in RESULT_INTENTS]
        if len(result_items) < self.minimum_result_candidates:
            raise ValueError("selection does not meet minimum_result_candidates")
        result_papers = {item.paper_id for item in result_items}
        if len(result_papers) < self.minimum_result_papers:
            raise ValueError("selection does not meet minimum_result_papers")
        return self


class DevelopmentAnnotationItem(StrictModel):
    schema_version: Literal["mixed-development-annotation-item/0.1"] = ITEM_SCHEMA_VERSION
    item_id: str = Field(pattern=r"^devann_[0-9a-f]{20}$")
    paper_id: str = Field(pattern=_PAPER_ID)
    page: int = Field(ge=1)
    locator: str = Field(min_length=1, max_length=500)
    source_excerpt: str = Field(min_length=1, max_length=2_000)
    source_excerpt_sha256: str = Field(pattern=_HEX_64)
    source_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_HEX_64)
    source_manifest_sha256: str = Field(pattern=_HEX_64)
    layout_sha256: str = Field(pattern=_HEX_64)
    page_text_sha256: str = Field(pattern=_HEX_64)

    @model_validator(mode="after")
    def immutable_identity_matches(self) -> DevelopmentAnnotationItem:
        if sha256_bytes(self.source_excerpt.encode("utf-8")) != self.source_excerpt_sha256:
            raise ValueError("source_excerpt_sha256 does not match source_excerpt")
        if self.item_id != _item_id(self.model_dump(mode="json", exclude={"item_id"})):
            raise ValueError("item_id does not match the complete source-bound identity")
        return self


class EvidenceAnchor(StrictModel):
    source_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    exact_excerpt: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def excerpt_is_one_physical_line(self) -> EvidenceAnchor:
        if "\n" in self.exact_excerpt or "\r" in self.exact_excerpt:
            raise ValueError("exact evidence must be a contiguous single-line excerpt")
        return self


class DevelopmentAnnotationResponse(StrictModel):
    schema_version: Literal["mixed-development-annotation-response/0.1"] = RESPONSE_SCHEMA_VERSION
    item_id: str = Field(pattern=r"^devann_[0-9a-f]{20}$")
    reviewer: str = Field(pattern=_REVIEWER_ID)
    result_bearing: ResultBearing | None = None
    result_evidence: EvidenceAnchor | None = None
    origin: ResultOrigin | None = None
    origin_evidence: EvidenceAnchor | None = None
    confidence: ReviewConfidence | None = None
    notes: str | None = Field(default=None, min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def blank_or_complete(self) -> DevelopmentAnnotationResponse:
        if self.result_bearing is None:
            if any(
                value is not None
                for value in (
                    self.result_evidence,
                    self.origin,
                    self.origin_evidence,
                    self.confidence,
                    self.notes,
                )
            ):
                raise ValueError("a blank response must leave every decision field null")
            return self
        if self.result_evidence is None or self.confidence is None:
            raise ValueError("a completed response requires result_evidence and confidence")
        if self.result_bearing is ResultBearing.YES:
            if self.origin is None:
                raise ValueError("a result-bearing row requires an origin decision")
            if self.origin is not ResultOrigin.UNCERTAIN and self.origin_evidence is None:
                raise ValueError("a resolved origin decision requires origin_evidence")
        elif self.origin is not None or self.origin_evidence is not None:
            raise ValueError("origin fields must be null unless result_bearing is yes")
        return self

    @property
    def is_blank(self) -> bool:
        return self.result_bearing is None


class PackageFile(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_HEX_64)
    records: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def path_is_confined(self) -> PackageFile:
        _relative_path(self.path)
        return self


class PackagePrivacy(StrictModel):
    contains_source_text: Literal[True] = True
    contains_source_pdfs: Literal[True] = True
    contains_human_labels: Literal[False] = False
    contains_per_item_selection_intent: Literal[False] = False
    contains_local_paths: Literal[False] = False
    private_uncommitted_required: Literal[True] = True


class DevelopmentAnnotationPackageManifest(StrictModel):
    schema_version: Literal["mixed-development-annotation-package/0.1"] = PACKAGE_SCHEMA_VERSION
    status: Literal["prepared-unlabeled"] = "prepared-unlabeled"
    protocol_version: Literal["mixed-development-result-origin/0.1"] = PROTOCOL_VERSION
    source_scope: Literal["development-and-error-analysis-only"] = (
        "development-and-error-analysis-only"
    )
    future_unseen_data_used: Literal[False] = False
    source_run_id: str = Field(min_length=1)
    selection_plan_sha256: str = Field(pattern=_HEX_64)
    mixed_selection_contract_satisfied: Literal[True] = True
    reviewer: str = Field(pattern=_REVIEWER_ID)
    item_count: int = Field(ge=1)
    result_candidate_count: int = Field(ge=1)
    result_candidate_paper_count: int = Field(ge=1)
    paper_count: int = Field(ge=1)
    files: list[PackageFile] = Field(min_length=4)
    privacy: PackagePrivacy = Field(default_factory=PackagePrivacy)

    @model_validator(mode="after")
    def file_paths_are_unique(self) -> DevelopmentAnnotationPackageManifest:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("package file paths must be unique")
        return self


class ValidatedEvidenceAnchor(StrictModel):
    source_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    exact_excerpt_sha256: str = Field(pattern=_HEX_64)
    character_start: int = Field(ge=0)
    character_end: int = Field(gt=0)
    line_number: int = Field(ge=1)


class ValidatedDevelopmentDecision(StrictModel):
    item_id: str = Field(pattern=r"^devann_[0-9a-f]{20}$")
    result_evidence: ValidatedEvidenceAnchor
    origin_evidence: ValidatedEvidenceAnchor | None = None


@dataclass(frozen=True)
class ValidatedDevelopmentResponse:
    reviewer: str
    response_sha256: str
    responses: tuple[DevelopmentAnnotationResponse, ...]
    evidence_bindings: tuple[ValidatedDevelopmentDecision, ...]


@dataclass(frozen=True)
class _FrozenPaper:
    paper_id: str
    source_id: str
    source_sha256: str
    byte_size: int
    pdf_path: Path
    source_manifest_sha256: str
    layout_sha256: str
    page_text: dict[int, str]
    page_text_sha256: dict[int, str]


PROTOCOL_TEXT = f"""# Mixed single-reviewer development annotation

Protocol version: `{PROTOCOL_VERSION}`

This is a development and error-analysis review, not independent validation. One reviewer
labels every item. Do not calculate inter-annotator agreement from this package. Do not use
the decisions as an unseen holdout result.

## Decision

Set `result_bearing` to `yes` only when the selected unit reports a quantitative evaluation
outcome or measured behavior. Use `no` for headers, captions, sample counts, setup values,
parameters, thresholds, or other metadata that do not themselves report an outcome. Use
`uncertain` when the frozen paper does not support either decision safely.

For `yes`, set `origin` to `paper_produced`, `externally_sourced`, or `uncertain`.
`paper_produced` includes a baseline rerun by the current paper's authors. A number copied
from a cited paper, vendor, or leaderboard is `externally_sourced`. System ownership is not
result origin. Do not infer paper production from silence or from a model/dataset name.

## Two independent evidence anchors

`result_evidence` supports the result-bearing decision and must cite the item's selected
page. `origin_evidence` separately supports who produced the result and may cite another
page in the same frozen PDF. For a `no` or `uncertain` result-bearing decision, leave both
origin fields null. For a `yes` decision, always complete `origin`. A `paper_produced` or
`externally_sourced` origin requires exact origin evidence. When origin is genuinely
`uncertain`, `origin_evidence` may be null if the paper supplies no page-local cue; otherwise
cite the strongest exact excerpt that explains the ambiguity.

Every evidence excerpt must be copied verbatim from one physical line on its declared page.
Do not normalize whitespace, join columns, reconstruct a wrapped sentence, combine pages, or
paraphrase. Shorter exact fragments are preferable to reconstructed text. The validator
checks exact UTF-8 substring membership on the declared page and rejects all pooled or
whitespace-normalized matches.

Inspect the supplied PDF as needed. Do not consult extraction candidates, model rationales,
reference labels, prior responses, or the coordinator's selection plan. Preserve nulls until
you personally review the item. Use `notes` only for a short private explanation.

## Response shape

Edit only the decision fields in each existing JSONL object. Copy `source_id` from the item.
These are structural examples with placeholders, not answer keys:

```json
{{
  "result_bearing": "yes",
  "result_evidence": {{
    "source_id": "COPY_FROM_ITEM",
    "page": 1,
    "exact_excerpt": "COPY_EXACT_RESULT_TEXT"
  }},
  "origin": "uncertain",
  "origin_evidence": null,
  "confidence": "low",
  "notes": "optional private note"
}}
```

```json
{{
  "result_bearing": "no",
  "result_evidence": {{
    "source_id": "COPY_FROM_ITEM",
    "page": 1,
    "exact_excerpt": "COPY_EXACT_TEXT"
  }},
  "origin": null,
  "origin_evidence": null,
  "confidence": "high",
  "notes": null
}}
```

Keep the existing `schema_version`, `item_id`, and `reviewer` fields on every line. A result
marked `uncertain` uses the second shape: its origin fields remain null.

This directory contains copyrighted source text and PDFs. Keep it private and uncommitted.
"""


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("package file path must be normalized, relative, and confined")
    return path


def _require_outside_public_repo(path: Path, public_repo_root: Path, label: str) -> None:
    resolved = path.resolve()
    root = public_repo_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return
    raise ValueError(f"{label} must remain outside the public repository")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path.name}:{line_number}: blank JSONL line")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number}: row must be a JSON object")
        rows.append(value)
    return rows


def _item_id(payload: dict[str, Any]) -> str:
    return f"devann_{sha256_bytes(canonical_json_bytes(payload))[:20]}"


def _single_line_occurrence(excerpt: str, page_text: str) -> bool:
    return (
        bool(excerpt)
        and "\n" not in excerpt
        and "\r" not in excerpt
        and any(excerpt in line for line in page_text.splitlines())
    )


def _source_pdf_path(cache_relpath: str, source_project_root: Path) -> Path:
    relative = _relative_path(cache_relpath)
    path = source_project_root.joinpath(*relative.parts)
    try:
        path.resolve().relative_to(source_project_root.resolve())
    except ValueError as error:
        raise ValueError("source PDF path escapes source project root") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError("frozen source PDF must be a regular non-symlink file")
    return path


def _load_frozen_paper(paper_id: str, *, run_root: Path, source_project_root: Path) -> _FrozenPaper:
    paper_root = run_root / paper_id
    manifest_path = paper_root / "source-manifest.json"
    layout_path = paper_root / "private" / "layout.json"
    for path, label in ((manifest_path, "source manifest"), (layout_path, "layout")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{paper_id}: frozen {label} is unavailable")
    manifest = read_json(manifest_path)
    layout = read_json(layout_path)
    if not isinstance(manifest, dict) or not isinstance(layout, dict):
        raise ValueError(f"{paper_id}: frozen manifest/layout must be JSON objects")
    if manifest.get("paper_id") != paper_id:
        raise ValueError(f"{paper_id}: source manifest paper_id mismatch")
    source_id = layout.get("source_id")
    sources = manifest.get("sources")
    if not isinstance(source_id, str) or not isinstance(sources, list):
        raise ValueError(f"{paper_id}: frozen source metadata is incomplete")
    source_matches = [
        source
        for source in sources
        if isinstance(source, dict)
        and source.get("source_id") == source_id
        and source.get("role") == "paper"
    ]
    if len(source_matches) != 1:
        raise ValueError(f"{paper_id}: paper source is not unique")
    source = source_matches[0]
    source_sha256 = source.get("sha256")
    byte_size = source.get("byte_size")
    cache_relpath = source.get("cache_relpath")
    if (
        not isinstance(source_sha256, str)
        or re.fullmatch(_HEX_64, source_sha256) is None
        or not isinstance(byte_size, int)
        or isinstance(byte_size, bool)
        or byte_size < 1
        or not isinstance(cache_relpath, str)
    ):
        raise ValueError(f"{paper_id}: paper source fingerprint is invalid")
    pdf_path = _source_pdf_path(cache_relpath, source_project_root)
    if pdf_path.stat().st_size != byte_size or sha256_file(pdf_path) != source_sha256:
        raise ValueError(f"{paper_id}: cached PDF differs from the frozen source manifest")

    pages = layout.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"{paper_id}: layout pages are unavailable")
    page_text: dict[int, str] = {}
    page_hashes: dict[int, str] = {}
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError(f"{paper_id}: layout page must be an object")
        page_number = page.get("page")
        text = page.get("text")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
            or not isinstance(text, str)
            or page_number in page_text
        ):
            raise ValueError(f"{paper_id}: layout page is invalid or duplicated")
        digest = sha256_bytes(text.encode("utf-8"))
        if page.get("text_sha256") != digest:
            raise ValueError(f"{paper_id}: layout page text digest is invalid")
        page_text[page_number] = text
        page_hashes[page_number] = digest
    return _FrozenPaper(
        paper_id=paper_id,
        source_id=source_id,
        source_sha256=source_sha256,
        byte_size=byte_size,
        pdf_path=pdf_path,
        source_manifest_sha256=sha256_file(manifest_path),
        layout_sha256=sha256_file(layout_path),
        page_text=page_text,
        page_text_sha256=page_hashes,
    )


def load_development_selection(path: Path) -> DevelopmentSelectionPlan:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("development selection plan must be a JSON object")
    return DevelopmentSelectionPlan.model_validate(value)


def build_development_annotation_items(
    plan: DevelopmentSelectionPlan, *, run_root: Path, source_project_root: Path
) -> tuple[list[DevelopmentAnnotationItem], dict[str, _FrozenPaper]]:
    if run_root.name != plan.source_run_id:
        raise ValueError("selection source_run_id does not match run root")
    papers: dict[str, _FrozenPaper] = {}
    items: list[DevelopmentAnnotationItem] = []
    for selected in plan.items:
        paper = papers.setdefault(
            selected.paper_id,
            _load_frozen_paper(
                selected.paper_id,
                run_root=run_root,
                source_project_root=source_project_root,
            ),
        )
        page_text = paper.page_text.get(selected.page)
        if page_text is None:
            raise ValueError(f"{selected.paper_id}: selected page is absent from frozen layout")
        if not _single_line_occurrence(selected.selection_excerpt, page_text):
            raise ValueError(
                f"{selected.paper_id}: selection excerpt is not an exact single-line "
                f"substring of page {selected.page}"
            )
        payload = {
            "schema_version": ITEM_SCHEMA_VERSION,
            "paper_id": selected.paper_id,
            "page": selected.page,
            "locator": selected.locator,
            "source_excerpt": selected.selection_excerpt,
            "source_excerpt_sha256": sha256_bytes(selected.selection_excerpt.encode("utf-8")),
            "source_id": paper.source_id,
            "source_sha256": paper.source_sha256,
            "source_manifest_sha256": paper.source_manifest_sha256,
            "layout_sha256": paper.layout_sha256,
            "page_text_sha256": paper.page_text_sha256[selected.page],
        }
        items.append(DevelopmentAnnotationItem(item_id=_item_id(payload), **payload))
    ids = [item.item_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("source-bound development annotation item IDs are not unique")
    return items, papers


def _package_file(path: Path, root: Path, *, records: int | None = None) -> PackageFile:
    return PackageFile(
        path=path.relative_to(root).as_posix(), sha256=sha256_file(path), records=records
    )


def _make_read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~0o222)


def _prepare_tree(
    *,
    tree: Path,
    plan: DevelopmentSelectionPlan,
    plan_path: Path,
    items: list[DevelopmentAnnotationItem],
    papers: dict[str, _FrozenPaper],
    reviewer: str,
) -> DevelopmentAnnotationPackageManifest:
    protocol_path = tree / "protocol.md"
    protocol_path.write_text(PROTOCOL_TEXT, encoding="utf-8")
    items_path = tree / "items.jsonl"
    write_jsonl(items_path, [item.model_dump(mode="json") for item in items])
    responses = [
        DevelopmentAnnotationResponse(item_id=item.item_id, reviewer=reviewer) for item in items
    ]
    response_path = tree / "response.jsonl"
    write_jsonl(response_path, [item.model_dump(mode="json") for item in responses])

    pdf_paths: list[Path] = []
    for paper_id, paper in papers.items():
        destination = tree / "pdfs" / f"{paper_id}.pdf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paper.pdf_path, destination)
        if destination.stat().st_size != paper.byte_size or sha256_file(destination) != (
            paper.source_sha256
        ):
            raise ValueError(f"{paper_id}: copied PDF differs from frozen source")
        pdf_paths.append(destination)

    for immutable in (protocol_path, items_path, *pdf_paths):
        _make_read_only(immutable)

    result_items = [item for item in plan.items if item.selection_intent in RESULT_INTENTS]
    manifest = DevelopmentAnnotationPackageManifest(
        source_run_id=plan.source_run_id,
        selection_plan_sha256=sha256_file(plan_path),
        reviewer=reviewer,
        item_count=len(items),
        result_candidate_count=len(result_items),
        result_candidate_paper_count=len({item.paper_id for item in result_items}),
        paper_count=len(papers),
        files=[
            _package_file(protocol_path, tree),
            _package_file(items_path, tree, records=len(items)),
            _package_file(response_path, tree, records=len(items)),
            *(_package_file(path, tree) for path in sorted(pdf_paths)),
        ],
    )
    manifest_path = tree / "manifest.json"
    write_json(manifest_path, manifest)
    _make_read_only(manifest_path)
    return manifest


def prepare_development_annotation_package(
    *,
    selection_path: Path,
    run_root: Path,
    source_project_root: Path,
    output_dir: Path,
    public_repo_root: Path,
    reviewer: str = "reviewer-primary",
) -> tuple[DevelopmentAnnotationPackageManifest, str]:
    """Create one deterministic, private, all-null mixed development handoff."""

    _require_outside_public_repo(selection_path, public_repo_root, "selection plan")
    _require_outside_public_repo(output_dir, public_repo_root, "annotation package")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite an annotation package or human work")
    DevelopmentAnnotationResponse(
        item_id="devann_" + "0" * 20,
        reviewer=reviewer,
    )
    plan = load_development_selection(selection_path)
    items, papers = build_development_annotation_items(
        plan, run_root=run_root, source_project_root=source_project_root
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        manifest = _prepare_tree(
            tree=staging,
            plan=plan,
            plan_path=selection_path,
            items=items,
            papers=papers,
            reviewer=reviewer,
        )
        _validate_package(
            package_dir=staging,
            selection_path=selection_path,
            run_root=run_root,
            source_project_root=source_project_root,
            public_repo_root=public_repo_root,
            require_blank=True,
        )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest, sha256_file(output_dir / "manifest.json")


def _declared_files(
    package_dir: Path,
    manifest: DevelopmentAnnotationPackageManifest,
    *,
    require_blank: bool,
) -> dict[str, Path]:
    actual = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected = {"manifest.json", *(item.path for item in manifest.files)}
    if actual != expected:
        raise ValueError("annotation package has an unexpected file contract")
    paths: dict[str, Path] = {}
    for declared in manifest.files:
        path = package_dir.joinpath(*_relative_path(declared.path).parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError("annotation package files must be regular and non-symlinked")
        if (declared.path != "response.jsonl" or require_blank) and (
            sha256_file(path) != declared.sha256
        ):
            raise ValueError(f"annotation package file hash mismatch: {declared.path}")
        paths[declared.path] = path
    return paths


def _validate_package(
    *,
    package_dir: Path,
    selection_path: Path,
    run_root: Path,
    source_project_root: Path,
    public_repo_root: Path,
    require_blank: bool,
) -> tuple[
    DevelopmentAnnotationPackageManifest,
    list[DevelopmentAnnotationItem],
    list[DevelopmentAnnotationResponse],
    dict[str, _FrozenPaper],
]:
    _require_outside_public_repo(selection_path, public_repo_root, "selection plan")
    _require_outside_public_repo(package_dir, public_repo_root, "annotation package")
    manifest_path = package_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("annotation package manifest must be a regular file")
    manifest = DevelopmentAnnotationPackageManifest.model_validate(read_json(manifest_path))
    if manifest.selection_plan_sha256 != sha256_file(selection_path):
        raise ValueError("annotation package is bound to another selection plan")
    plan = load_development_selection(selection_path)
    expected_items, papers = build_development_annotation_items(
        plan, run_root=run_root, source_project_root=source_project_root
    )
    paths = _declared_files(package_dir, manifest, require_blank=require_blank)
    if paths["protocol.md"].read_text(encoding="utf-8") != PROTOCOL_TEXT:
        raise ValueError("annotation protocol differs from its declared version")
    item_rows = _read_jsonl(paths["items.jsonl"])
    items = [DevelopmentAnnotationItem.model_validate(row) for row in item_rows]
    if items != expected_items:
        raise ValueError("annotation items differ from the coordinator selection projection")
    response_rows = _read_jsonl(paths["response.jsonl"])
    responses = [DevelopmentAnnotationResponse.model_validate(row) for row in response_rows]
    if [row.item_id for row in responses] != [item.item_id for item in items]:
        raise ValueError("annotation responses must exactly match item order and membership")
    if any(row.reviewer != manifest.reviewer for row in responses):
        raise ValueError("annotation responses contain a wrong or mixed reviewer")
    if require_blank and any(not row.is_blank for row in responses):
        raise ValueError("initial annotation response must be entirely blank")
    if manifest.item_count != len(items) or manifest.paper_count != len(papers):
        raise ValueError("annotation manifest counts differ from selected items")
    result_items = [item for item in plan.items if item.selection_intent in RESULT_INTENTS]
    if manifest.result_candidate_count != len(
        result_items
    ) or manifest.result_candidate_paper_count != len({item.paper_id for item in result_items}):
        raise ValueError("annotation manifest result-candidate counts are invalid")
    for paper_id, paper in papers.items():
        copy = paths.get(f"pdfs/{paper_id}.pdf")
        if copy is None or copy.stat().st_size != paper.byte_size:
            raise ValueError(f"{paper_id}: annotation package PDF is absent or truncated")
        if sha256_file(copy) != paper.source_sha256:
            raise ValueError(f"{paper_id}: annotation package PDF hash mismatch")
    return manifest, items, responses, papers


def validate_initial_development_annotation_package(
    *,
    package_dir: Path,
    selection_path: Path,
    run_root: Path,
    source_project_root: Path,
    public_repo_root: Path,
) -> DevelopmentAnnotationPackageManifest:
    manifest, _, _, _ = _validate_package(
        package_dir=package_dir,
        selection_path=selection_path,
        run_root=run_root,
        source_project_root=source_project_root,
        public_repo_root=public_repo_root,
        require_blank=True,
    )
    return manifest


def _bind_evidence(anchor: EvidenceAnchor, item: DevelopmentAnnotationItem, paper: _FrozenPaper):
    if anchor.source_id != item.source_id or anchor.source_id != paper.source_id:
        raise ValueError(f"{item.item_id}: evidence source_id differs from selected source")
    text = paper.page_text.get(anchor.page)
    if text is None or not _single_line_occurrence(anchor.exact_excerpt, text):
        raise ValueError(
            f"{item.item_id}: exact evidence is absent as a contiguous single-line substring "
            f"of declared page {anchor.page}"
        )
    start = text.index(anchor.exact_excerpt)
    return ValidatedEvidenceAnchor(
        source_id=anchor.source_id,
        page=anchor.page,
        exact_excerpt_sha256=hashlib.sha256(anchor.exact_excerpt.encode("utf-8")).hexdigest(),
        character_start=start,
        character_end=start + len(anchor.exact_excerpt),
        line_number=text.count("\n", 0, start) + 1,
    )


def validate_completed_development_annotation_response(
    *,
    package_dir: Path,
    selection_path: Path,
    run_root: Path,
    source_project_root: Path,
    public_repo_root: Path,
    expected_manifest_sha256: str,
) -> ValidatedDevelopmentResponse:
    """Validate one complete response with strict, separate page-local evidence."""

    if re.fullmatch(_HEX_64, expected_manifest_sha256) is None:
        raise ValueError("expected package manifest SHA-256 is invalid")
    manifest_path = package_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("annotation package manifest must be a regular file")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("annotation package manifest differs from the preserved receipt")
    manifest, items, responses, papers = _validate_package(
        package_dir=package_dir,
        selection_path=selection_path,
        run_root=run_root,
        source_project_root=source_project_root,
        public_repo_root=public_repo_root,
        require_blank=False,
    )
    if any(response.is_blank for response in responses):
        raise ValueError("development annotation response still contains blank decisions")
    bindings: list[ValidatedDevelopmentDecision] = []
    for item, response in zip(items, responses, strict=True):
        assert response.result_evidence is not None
        if response.result_evidence.page != item.page:
            raise ValueError(f"{item.item_id}: result evidence must cite the selected item page")
        paper = papers[item.paper_id]
        result_binding = _bind_evidence(response.result_evidence, item, paper)
        origin_binding = (
            _bind_evidence(response.origin_evidence, item, paper)
            if response.origin_evidence is not None
            else None
        )
        bindings.append(
            ValidatedDevelopmentDecision(
                item_id=item.item_id,
                result_evidence=result_binding,
                origin_evidence=origin_binding,
            )
        )
    response_path = package_dir / "response.jsonl"
    return ValidatedDevelopmentResponse(
        reviewer=manifest.reviewer,
        response_sha256=sha256_file(response_path),
        responses=tuple(responses),
        evidence_bindings=tuple(bindings),
    )
