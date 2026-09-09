"""Exact per-run accounting for how each frozen source was processed."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from proceedings_to_eee.domain.observation import StrictModel
from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.io import canonical_json_bytes, sha256_bytes
from proceedings_to_eee.sources.manifest import SourceManifest, SourceRole

SOURCE_PROCESSING_SCHEMA_VERSION = "source-processing/0.1"


class SourceProcessingState(StrEnum):
    PROCESSED = "processed"
    UNSUPPORTED = "unsupported"


class SourceProcessingReason(StrEnum):
    PRIMARY_LAYOUT_EXTRACTED = "primary_layout_extracted"
    NO_ADAPTER = "no_adapter"
    COMMIT_ADVERTISED_ONLY_NO_TREE_BYTES = "commit_advertised_only_no_tree_bytes"


class SourceLayoutArtifact(StrictModel):
    path: Literal["private/layout.json"] = "private/layout.json"
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser: str
    parser_version: str
    page_count: int = Field(ge=1)


class SourceProcessingRecord(StrictModel):
    source_id: str
    role: SourceRole
    state: SourceProcessingState
    reason: SourceProcessingReason
    layout: SourceLayoutArtifact | None = None

    @model_validator(mode="after")
    def state_matches_evidence(self) -> SourceProcessingRecord:
        if self.role is SourceRole.PAPER:
            if (
                self.state is not SourceProcessingState.PROCESSED
                or self.reason is not SourceProcessingReason.PRIMARY_LAYOUT_EXTRACTED
                or self.layout is None
            ):
                raise ValueError("processed source requires the primary layout artifact")
        elif self.state is not SourceProcessingState.UNSUPPORTED or self.layout is not None:
            raise ValueError("unsupported sources cannot claim a processed layout artifact")
        if (
            self.role is SourceRole.SUPPLEMENT
            and self.reason is not SourceProcessingReason.NO_ADAPTER
        ):
            raise ValueError("supplement processing must state no_adapter")
        if (
            self.role is SourceRole.REPOSITORY
            and self.reason is not SourceProcessingReason.COMMIT_ADVERTISED_ONLY_NO_TREE_BYTES
        ):
            raise ValueError("repository processing must state its commit-only evidence basis")
        return self


class SourceProcessingCounts(StrictModel):
    sources: int = Field(ge=1)
    processed: int = Field(ge=0)
    unsupported: int = Field(ge=0)

    @model_validator(mode="after")
    def states_sum_to_sources(self) -> SourceProcessingCounts:
        if self.processed + self.unsupported != self.sources:
            raise ValueError("source-processing counts do not sum to all manifest sources")
        return self


class SourceProcessingArtifact(StrictModel):
    schema_version: Literal["source-processing/0.1"] = SOURCE_PROCESSING_SCHEMA_VERSION
    paper_id: str
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_by_id: dict[str, SourceProcessingRecord]
    counts: SourceProcessingCounts

    @model_validator(mode="after")
    def keys_and_counts_match(self) -> SourceProcessingArtifact:
        if any(key != value.source_id for key, value in self.source_by_id.items()):
            raise ValueError("source-processing key does not match source_id")
        processed = sum(
            record.state is SourceProcessingState.PROCESSED for record in self.source_by_id.values()
        )
        expected = SourceProcessingCounts(
            sources=len(self.source_by_id),
            processed=processed,
            unsupported=len(self.source_by_id) - processed,
        )
        if self.counts != expected:
            raise ValueError("source-processing counts do not match source records")
        return self

    @property
    def processed_source_ids(self) -> set[str]:
        return {
            source_id
            for source_id, record in self.source_by_id.items()
            if record.state is SourceProcessingState.PROCESSED
        }


def build_source_processing_artifact(
    *,
    manifest: SourceManifest,
    layout: PdfLayout,
    layout_sha256: str,
) -> SourceProcessingArtifact:
    """Partition every manifest source into processed or explicitly unsupported."""

    manifest_by_id = {source.source_id: source for source in manifest.sources}
    paper_sources = [source for source in manifest.sources if source.role is SourceRole.PAPER]
    if len(paper_sources) != 1:
        raise ValueError("source processing requires exactly one primary paper source")
    if layout.source_id != paper_sources[0].source_id:
        raise ValueError("primary layout is not bound to the manifest paper source")

    records: dict[str, SourceProcessingRecord] = {}
    for source in manifest.sources:
        if source.role is SourceRole.PAPER:
            record = SourceProcessingRecord(
                source_id=source.source_id,
                role=source.role,
                state=SourceProcessingState.PROCESSED,
                reason=SourceProcessingReason.PRIMARY_LAYOUT_EXTRACTED,
                layout=SourceLayoutArtifact(
                    sha256=layout_sha256,
                    parser=layout.parser,
                    parser_version=layout.parser_version,
                    page_count=layout.page_count,
                ),
            )
        elif source.role is SourceRole.REPOSITORY:
            record = SourceProcessingRecord(
                source_id=source.source_id,
                role=source.role,
                state=SourceProcessingState.UNSUPPORTED,
                reason=SourceProcessingReason.COMMIT_ADVERTISED_ONLY_NO_TREE_BYTES,
            )
        else:
            record = SourceProcessingRecord(
                source_id=source.source_id,
                role=source.role,
                state=SourceProcessingState.UNSUPPORTED,
                reason=SourceProcessingReason.NO_ADAPTER,
            )
        records[source.source_id] = record
    if set(records) != set(manifest_by_id):
        raise AssertionError("source-processing construction lost a manifest source")
    processed = sum(record.state is SourceProcessingState.PROCESSED for record in records.values())
    artifact = SourceProcessingArtifact(
        paper_id=manifest.paper_id,
        source_manifest_sha256=sha256_bytes(canonical_json_bytes(manifest)),
        source_by_id=records,
        counts=SourceProcessingCounts(
            sources=len(records),
            processed=processed,
            unsupported=len(records) - processed,
        ),
    )
    validate_source_processing_artifact(manifest=manifest, artifact=artifact, layout=layout)
    return artifact


def validate_source_processing_artifact(
    *,
    manifest: SourceManifest,
    artifact: SourceProcessingArtifact,
    layout: PdfLayout | None = None,
) -> None:
    """Validate exact source ownership without conflating access and processing state."""

    if artifact.paper_id != manifest.paper_id:
        raise ValueError("source-processing artifact belongs to another paper")
    if artifact.source_manifest_sha256 != sha256_bytes(canonical_json_bytes(manifest)):
        raise ValueError("source-processing artifact manifest hash mismatch")
    manifest_by_id = {source.source_id: source for source in manifest.sources}
    if set(artifact.source_by_id) != set(manifest_by_id):
        raise ValueError("source-processing artifact does not partition every manifest source")
    for source_id, record in artifact.source_by_id.items():
        source = manifest_by_id[source_id]
        if record.role is not source.role:
            raise ValueError("source-processing role does not match the frozen manifest")
        if source.role is SourceRole.PAPER:
            if record.state is not SourceProcessingState.PROCESSED or record.layout is None:
                raise ValueError("primary paper source was not processed")
            if layout is not None and (
                layout.source_id != source_id
                or record.layout.parser != layout.parser
                or record.layout.parser_version != layout.parser_version
                or record.layout.page_count != layout.page_count
            ):
                raise ValueError("source-processing layout metadata does not match the layout")
        elif record.state is not SourceProcessingState.UNSUPPORTED:
            raise ValueError("source without an adapter cannot be marked processed")
