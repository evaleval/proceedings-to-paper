"""Freeze-gated, blinded annotation packages for a future unseen evaluation.

This module defines the framework only.  It never discovers papers, downloads sources,
runs the pipeline, chooses semantic cases, or writes human labels.  Package creation is
possible only after an engineering/model freeze, a metadata-only selection bound to that
freeze, a pre-run sampling/retry contract, one sealed pipeline run, and a label-free
annotation pool all exist and validate together.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlparse

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
from proceedings_to_eee.run_seal import VerifiedRunSeal, verify_run_seal

METADATA_SCHEMA_VERSION = "unseen-metadata-selection/0.1"
REGISTRY_SCHEMA_VERSION = "inspected-paper-registry/0.1"
SAMPLING_SCHEMA_VERSION = "unseen-sampling-frame/0.1"
RETRY_SCHEMA_VERSION = "unseen-transport-retry-policy/0.1"
FREEZE_SCHEMA_VERSION = "unseen-engineering-freeze/0.1"
CORPUS_SCHEMA_VERSION = "unseen-corpus-definition/0.1"
RUN_RECEIPT_SCHEMA_VERSION = "unseen-run-receipt/0.1"
PAPER_BINDING_SCHEMA_VERSION = "unseen-paper-freeze-binding/0.1"
POOL_SCHEMA_VERSION = "unseen-annotation-pool-item/0.1"
ITEM_SCHEMA_VERSION = "unseen-annotation-item/0.1"
RESPONSE_SCHEMA_VERSION = "unseen-annotation-response/0.1"
PACKAGE_SCHEMA_VERSION = "unseen-annotation-package/0.1"
WORKSPACE_SCHEMA_VERSION = "unseen-annotation-workspace/0.1"
PROTOCOL_VERSION = "unseen-independent-result-origin/0.1"

RUN_RECEIPT_PATH = "future-evaluation-run-receipt.json"
CORPUS_DEFINITION_PATH = "future-corpus.json"
POOL_PATH = "private/future-annotation-pool.jsonl"
ATTEMPTS_PATH = "private/transport-attempts.jsonl"
PAPER_BINDING_NAME = "future-freeze-binding.json"

_HEX_64 = r"^[0-9a-f]{64}$"
_HEX_40 = r"^[0-9a-f]{40}$"
_SAFE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_PAPER_ID = r"^[a-z0-9][a-z0-9-]*$"
_ACTOR_ID = r"^(?:reviewer|adjudicator)-[a-z0-9][a-z0-9-]*$"
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")


class PoolStratum(StrEnum):
    ELIGIBLE_ATOMIC_RESULT_CANDIDATE = "eligible_atomic_result_candidate"
    NON_RESULT_CANDIDATE = "non_result_candidate"
    EXTERNAL_RESULT_CANDIDATE = "external_result_candidate"
    SECONDARY_RESULT_CANDIDATE = "secondary_result_candidate"


CONTROL_STRATA = frozenset(
    {
        PoolStratum.NON_RESULT_CANDIDATE,
        PoolStratum.EXTERNAL_RESULT_CANDIDATE,
        PoolStratum.SECONDARY_RESULT_CANDIDATE,
    }
)
REQUIRED_STAGES = frozenset(
    {
        "block_extraction",
        "row_disposition",
        "tuple_resolution",
        "origin_retrieval",
        "verification",
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


class ClaimScope(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    UNCERTAIN = "uncertain"


class Atomicity(StrEnum):
    ATOMIC = "atomic"
    NOT_ATOMIC = "not_atomic"
    UNCERTAIN = "uncertain"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PackageRole(StrEnum):
    REVIEWER = "reviewer"
    ADJUDICATION = "adjudication"


class TransportFailureReason(StrEnum):
    TIMEOUT = "timeout"
    CONNECTION_RESET = "connection_reset"
    HTTP_429 = "http_429"
    HTTP_5XX = "http_5xx"


class AttemptOutcome(StrEnum):
    SUCCESS = "success"
    TRANSPORT_FAILURE = "transport_failure"
    PROVIDER_REJECTION = "provider_rejection"
    SCHEMA_FAILURE = "schema_failure"
    SEMANTIC_REJECTION = "semantic_rejection"


class MetadataPaper(StrictModel):
    paper_id: str = Field(pattern=_PAPER_ID)
    title: str = Field(min_length=1, max_length=1_000)
    year: int = Field(ge=1900, le=2200)
    venue: str = Field(min_length=1, max_length=300)
    publisher: str = Field(min_length=1, max_length=300)
    landing_url: str = Field(min_length=1, max_length=2_000)
    layout_stratum: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def metadata_url_only(self) -> MetadataPaper:
        parsed = urlparse(self.landing_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("landing_url must be an HTTP(S) metadata URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("landing_url must not contain credentials")
        return self


class PriorPaperRegistry(StrictModel):
    schema_version: Literal["inspected-paper-registry/0.1"] = REGISTRY_SCHEMA_VERSION
    status: Literal["complete-before-metadata-selection"] = "complete-before-metadata-selection"
    snapshot_id: str = Field(min_length=1, max_length=200)
    inspected_or_development_paper_ids: list[str]

    @model_validator(mode="after")
    def unique_safe_ids(self) -> PriorPaperRegistry:
        ids = self.inspected_or_development_paper_ids
        if len(ids) != len(set(ids)):
            raise ValueError("inspected/development paper registry contains duplicates")
        if any(re.fullmatch(_PAPER_ID, paper_id) is None for paper_id in ids):
            raise ValueError("inspected/development registry contains an unsafe paper_id")
        return self


class MetadataSelectionManifest(StrictModel):
    schema_version: Literal["unseen-metadata-selection/0.1"] = METADATA_SCHEMA_VERSION
    status: Literal["locked-after-engineering-freeze-before-access"] = (
        "locked-after-engineering-freeze-before-access"
    )
    selected_after_engineering_freeze: Literal[True] = True
    engineering_freeze_sha256: str = Field(pattern=_HEX_64)
    metadata_only_selection: Literal[True] = True
    semantic_content_accessed: Literal[False] = False
    full_text_downloaded: Literal[False] = False
    pipeline_run_started: Literal[False] = False
    future_unseen: Literal[True] = True
    prior_registry_sha256: str = Field(pattern=_HEX_64)
    paper_count: int = Field(ge=10)
    papers: list[MetadataPaper] = Field(min_length=10)

    @model_validator(mode="after")
    def exact_unique_paper_count(self) -> MetadataSelectionManifest:
        ids = [paper.paper_id for paper in self.papers]
        if self.paper_count != len(self.papers):
            raise ValueError("metadata selection paper_count differs from papers")
        if len(ids) != len(set(ids)):
            raise ValueError("metadata selection paper IDs must be unique")
        return self


class SamplingFrame(StrictModel):
    schema_version: Literal["unseen-sampling-frame/0.1"] = SAMPLING_SCHEMA_VERSION
    status: Literal["locked-before-pipeline-run"] = "locked-before-pipeline-run"
    selection_algorithm: Literal["sha256-ranked-within-pipeline-stratum/0.1"] = (
        "sha256-ranked-within-pipeline-stratum/0.1"
    )
    engineering_freeze_sha256: str = Field(pattern=_HEX_64)
    metadata_selection_sha256: str = Field(pattern=_HEX_64)
    randomization_seed: str = Field(pattern=_HEX_64)
    eligible_atomic_result_target: int = Field(ge=100)
    control_targets: dict[PoolStratum, int]
    minimum_sampled_papers: int = Field(ge=10)
    allow_post_run_denominator_change: Literal[False] = False
    allow_post_run_stratum_change: Literal[False] = False

    @model_validator(mode="after")
    def complete_control_contract(self) -> SamplingFrame:
        if set(self.control_targets) != CONTROL_STRATA:
            raise ValueError("control_targets must contain non-result, external, and secondary")
        if any(value < 1 for value in self.control_targets.values()):
            raise ValueError("every control stratum requires a positive target")
        if sum(self.control_targets.values()) < 50:
            raise ValueError("control target must total at least 50")
        return self

    @property
    def denominator(self) -> int:
        return self.eligible_atomic_result_target + sum(self.control_targets.values())


class TransportRetryPolicy(StrictModel):
    schema_version: Literal["unseen-transport-retry-policy/0.1"] = RETRY_SCHEMA_VERSION
    status: Literal["locked-before-pipeline-run"] = "locked-before-pipeline-run"
    engineering_freeze_sha256: str = Field(pattern=_HEX_64)
    metadata_selection_sha256: str = Field(pattern=_HEX_64)
    maximum_attempts_per_request: int = Field(ge=1, le=5)
    allowed_retry_reasons: list[TransportFailureReason] = Field(min_length=1)
    retry_after_success: Literal[False] = False
    semantic_retry_allowed: Literal[False] = False
    schema_retry_allowed: Literal[False] = False
    provider_rejection_retry_allowed: Literal[False] = False

    @model_validator(mode="after")
    def unique_reasons(self) -> TransportRetryPolicy:
        if len(self.allowed_retry_reasons) != len(set(self.allowed_retry_reasons)):
            raise ValueError("allowed retry reasons must be unique")
        return self


class FrozenStage(StrictModel):
    stage: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=300)
    prompt_sha256: str = Field(pattern=_HEX_64)
    provider_schema_sha256: str = Field(pattern=_HEX_64)
    request_contract_sha256: str = Field(pattern=_HEX_64)
    settings_sha256: str = Field(pattern=_HEX_64)


class EngineeringFreezeManifest(StrictModel):
    schema_version: Literal["unseen-engineering-freeze/0.1"] = FREEZE_SCHEMA_VERSION
    status: Literal["locked-before-metadata-selection"] = "locked-before-metadata-selection"
    freeze_id: str = Field(pattern=_SAFE_ID)
    code_git_commit: str = Field(pattern=_HEX_40)
    code_source_tree_sha256: str = Field(pattern=_HEX_64)
    dependency_lock_sha256: str = Field(pattern=_HEX_64)
    thresholds_sha256: str = Field(pattern=_HEX_64)
    attribution_policy_sha256: str = Field(pattern=_HEX_64)
    stages: list[FrozenStage] = Field(min_length=len(REQUIRED_STAGES))
    execution_contract_sha256: str = Field(pattern=_HEX_64)

    @model_validator(mode="after")
    def exact_execution_contract(self) -> EngineeringFreezeManifest:
        stages = [stage.stage for stage in self.stages]
        if len(stages) != len(set(stages)) or set(stages) != REQUIRED_STAGES:
            raise ValueError("engineering freeze must bind each required stage exactly once")
        if self.execution_contract_sha256 != execution_contract_sha256(self):
            raise ValueError("execution_contract_sha256 does not match frozen execution fields")
        return self


def execution_contract_sha256(value: EngineeringFreezeManifest | dict[str, Any]) -> str:
    """Hash only execution-affecting freeze fields, excluding the self-digest."""

    payload = (
        value.model_dump(mode="json") if isinstance(value, EngineeringFreezeManifest) else value
    )
    keys = (
        "code_git_commit",
        "code_source_tree_sha256",
        "dependency_lock_sha256",
        "thresholds_sha256",
        "attribution_policy_sha256",
        "stages",
    )
    return sha256_bytes(canonical_json_bytes({key: payload[key] for key in keys}))


class FrozenCorpusDefinition(StrictModel):
    schema_version: Literal["unseen-corpus-definition/0.1"] = CORPUS_SCHEMA_VERSION
    engineering_freeze_sha256: str = Field(pattern=_HEX_64)
    metadata_selection_sha256: str = Field(pattern=_HEX_64)
    paper_count: int = Field(ge=10)
    paper_ids: list[str] = Field(min_length=10)

    @model_validator(mode="after")
    def exact_unique_papers(self) -> FrozenCorpusDefinition:
        if self.paper_count != len(self.paper_ids):
            raise ValueError("frozen corpus paper_count differs from paper_ids")
        if len(self.paper_ids) != len(set(self.paper_ids)):
            raise ValueError("frozen corpus paper_ids must be unique")
        if any(re.fullmatch(_PAPER_ID, paper_id) is None for paper_id in self.paper_ids):
            raise ValueError("frozen corpus contains an unsafe paper_id")
        return self


class SealedRunReceipt(StrictModel):
    schema_version: Literal["unseen-run-receipt/0.1"] = RUN_RECEIPT_SCHEMA_VERSION
    status: Literal["completed-once-before-semantic-inspection"] = (
        "completed-once-before-semantic-inspection"
    )
    pipeline_execution_count: Literal[1] = 1
    semantic_inspection_before_seal: Literal[False] = False
    freeze_manifest_sha256: str = Field(pattern=_HEX_64)
    execution_contract_sha256: str = Field(pattern=_HEX_64)
    metadata_selection_sha256: str = Field(pattern=_HEX_64)
    prior_registry_sha256: str = Field(pattern=_HEX_64)
    sampling_frame_sha256: str = Field(pattern=_HEX_64)
    retry_policy_sha256: str = Field(pattern=_HEX_64)
    corpus_definition_sha256: str = Field(pattern=_HEX_64)
    annotation_pool_path: Literal["private/future-annotation-pool.jsonl"] = POOL_PATH
    annotation_pool_sha256: str = Field(pattern=_HEX_64)
    annotation_pool_records: int = Field(ge=150)
    transport_attempts_path: Literal["private/transport-attempts.jsonl"] = ATTEMPTS_PATH
    transport_attempts_sha256: str = Field(pattern=_HEX_64)
    transport_attempt_records: int = Field(ge=1)


class PaperFreezeBinding(StrictModel):
    schema_version: Literal["unseen-paper-freeze-binding/0.1"] = PAPER_BINDING_SCHEMA_VERSION
    paper_id: str = Field(pattern=_PAPER_ID)
    freeze_manifest_sha256: str = Field(pattern=_HEX_64)
    execution_contract_sha256: str = Field(pattern=_HEX_64)
    run_manifest_sha256: str = Field(pattern=_HEX_64)
    source_manifest_sha256: str = Field(pattern=_HEX_64)
    layout_sha256: str = Field(pattern=_HEX_64)


class TransportAttempt(StrictModel):
    request_slot: str = Field(pattern=_SAFE_ID)
    request_fingerprint: str = Field(pattern=_HEX_64)
    call_id: str = Field(pattern=_SAFE_ID)
    paper_id: str = Field(pattern=_PAPER_ID)
    stage: str = Field(min_length=1, max_length=100)
    attempt_number: int = Field(ge=1)
    outcome: AttemptOutcome
    failure_reason: TransportFailureReason | None = None
    retry_reason: TransportFailureReason | None = None

    @model_validator(mode="after")
    def outcome_fields_match(self) -> TransportAttempt:
        if self.outcome is AttemptOutcome.TRANSPORT_FAILURE:
            if self.failure_reason is None:
                raise ValueError("transport failure requires failure_reason")
        elif self.failure_reason is not None:
            raise ValueError("failure_reason is only valid for a transport failure")
        if self.attempt_number == 1 and self.retry_reason is not None:
            raise ValueError("the first attempt cannot have retry_reason")
        if self.attempt_number > 1 and self.retry_reason is None:
            raise ValueError("a repeated attempt requires retry_reason")
        return self


class AnnotationPoolRecord(StrictModel):
    schema_version: Literal["unseen-annotation-pool-item/0.1"] = POOL_SCHEMA_VERSION
    unit_id: str = Field(pattern=r"^pool_[0-9a-f]{24}$")
    pipeline_stratum: PoolStratum
    paper_id: str = Field(pattern=_PAPER_ID)
    source_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_HEX_64)
    source_manifest_sha256: str = Field(pattern=_HEX_64)
    layout_sha256: str = Field(pattern=_HEX_64)
    page: int = Field(ge=1)
    page_text_sha256: str = Field(pattern=_HEX_64)
    locator: str = Field(min_length=1, max_length=500)
    source_excerpt: str = Field(min_length=1, max_length=2_000)
    source_excerpt_sha256: str = Field(pattern=_HEX_64)
    source_record_path: str = Field(min_length=1)
    source_record_sha256: str = Field(pattern=_HEX_64)

    @model_validator(mode="after")
    def source_bound_identity(self) -> AnnotationPoolRecord:
        _relative_path(self.source_record_path)
        if not self.source_record_path.startswith(f"{self.paper_id}/"):
            raise ValueError("source_record_path must remain inside its paper run")
        if any(character in _LINE_BREAKS for character in self.source_excerpt):
            raise ValueError("pool excerpt must be a contiguous single-line excerpt")
        if sha256_bytes(self.source_excerpt.encode("utf-8")) != self.source_excerpt_sha256:
            raise ValueError("pool source_excerpt_sha256 mismatch")
        if self.unit_id != annotation_pool_unit_id(
            self.model_dump(mode="json", exclude={"unit_id"})
        ):
            raise ValueError("pool unit_id does not match its complete immutable identity")
        return self


def annotation_pool_unit_id(payload: dict[str, Any]) -> str:
    return f"pool_{sha256_bytes(canonical_json_bytes(payload))[:24]}"


class UnseenAnnotationItem(StrictModel):
    schema_version: Literal["unseen-annotation-item/0.1"] = ITEM_SCHEMA_VERSION
    item_id: str = Field(pattern=r"^unseen_[0-9a-f]{24}$")
    paper_id: str = Field(pattern=_PAPER_ID)
    source_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_HEX_64)
    source_manifest_sha256: str = Field(pattern=_HEX_64)
    layout_sha256: str = Field(pattern=_HEX_64)
    page: int = Field(ge=1)
    page_text_sha256: str = Field(pattern=_HEX_64)
    locator: str = Field(min_length=1, max_length=500)
    source_excerpt: str = Field(min_length=1, max_length=2_000)
    source_excerpt_sha256: str = Field(pattern=_HEX_64)

    @model_validator(mode="after")
    def immutable_identity(self) -> UnseenAnnotationItem:
        if any(character in _LINE_BREAKS for character in self.source_excerpt):
            raise ValueError("annotation excerpt must be a contiguous single-line excerpt")
        if sha256_bytes(self.source_excerpt.encode("utf-8")) != self.source_excerpt_sha256:
            raise ValueError("annotation source excerpt digest mismatch")
        if self.item_id != _annotation_item_id(self.model_dump(mode="json", exclude={"item_id"})):
            raise ValueError("annotation item ID does not match its immutable identity")
        return self


class EvidenceAnchor(StrictModel):
    source_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    exact_excerpt: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def one_physical_line(self) -> EvidenceAnchor:
        if any(character in _LINE_BREAKS for character in self.exact_excerpt):
            raise ValueError("exact evidence must be a contiguous single-line excerpt")
        return self


class UnseenAnnotationResponse(StrictModel):
    schema_version: Literal["unseen-annotation-response/0.1"] = RESPONSE_SCHEMA_VERSION
    item_id: str = Field(pattern=r"^unseen_[0-9a-f]{24}$")
    actor_id: str = Field(pattern=_ACTOR_ID)
    result_bearing: ResultBearing | None = None
    result_evidence: EvidenceAnchor | None = None
    origin: ResultOrigin | None = None
    origin_evidence: EvidenceAnchor | None = None
    claim_scope: ClaimScope | None = None
    atomicity: Atomicity | None = None
    confidence: Confidence | None = None
    notes: str | None = Field(default=None, min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def blank_or_complete(self) -> UnseenAnnotationResponse:
        decision_fields = (
            self.result_evidence,
            self.origin,
            self.origin_evidence,
            self.claim_scope,
            self.atomicity,
            self.confidence,
            self.notes,
        )
        if self.result_bearing is None:
            if any(value is not None for value in decision_fields):
                raise ValueError("a blank unseen response must leave every decision field null")
            return self
        if self.result_evidence is None or self.confidence is None:
            raise ValueError("a completed unseen response requires result_evidence and confidence")
        if self.result_bearing is ResultBearing.YES:
            if self.origin is None or self.claim_scope is None or self.atomicity is None:
                raise ValueError(
                    "a result-bearing response requires origin, claim_scope, and atomicity"
                )
            if self.origin is not ResultOrigin.UNCERTAIN and self.origin_evidence is None:
                raise ValueError("a resolved origin requires exact origin_evidence")
        elif any(
            value is not None
            for value in (self.origin, self.origin_evidence, self.claim_scope, self.atomicity)
        ):
            raise ValueError("origin, scope, and atomicity must be null for a non-result/uncertain")
        return self

    @property
    def is_blank(self) -> bool:
        return self.result_bearing is None


class PackageFile(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_HEX_64)
    records: int | None = Field(default=None, ge=0)
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def confined_path(self) -> PackageFile:
        _relative_path(self.path)
        return self


class RunSealBinding(StrictModel):
    seal_sha256: str = Field(pattern=_HEX_64)
    tree_sha256: str = Field(pattern=_HEX_64)
    file_count: int = Field(ge=1)
    total_bytes: int = Field(ge=1)


class PackagePrivacy(StrictModel):
    contains_source_text: Literal[True] = True
    contains_source_pdfs: Literal[True] = True
    contains_human_labels: Literal[False] = False
    contains_peer_outputs: Literal[False] = False
    contains_pipeline_strata: Literal[False] = False
    contains_local_paths: Literal[False] = False
    private_uncommitted_required: Literal[True] = True


class UnseenPackageManifest(StrictModel):
    schema_version: Literal["unseen-annotation-package/0.1"] = PACKAGE_SCHEMA_VERSION
    status: Literal["prepared-entirely-blank"] = "prepared-entirely-blank"
    role: PackageRole
    actor_id: str = Field(pattern=_ACTOR_ID)
    protocol_version: Literal["unseen-independent-result-origin/0.1"] = PROTOCOL_VERSION
    run_seal: RunSealBinding
    freeze_manifest_sha256: str = Field(pattern=_HEX_64)
    execution_contract_sha256: str = Field(pattern=_HEX_64)
    corpus_definition_sha256: str = Field(pattern=_HEX_64)
    code_git_commit: str = Field(pattern=_HEX_40)
    code_source_tree_sha256: str = Field(pattern=_HEX_64)
    dependency_lock_sha256: str = Field(pattern=_HEX_64)
    thresholds_sha256: str = Field(pattern=_HEX_64)
    attribution_policy_sha256: str = Field(pattern=_HEX_64)
    frozen_stages_sha256: str = Field(pattern=_HEX_64)
    metadata_selection_sha256: str = Field(pattern=_HEX_64)
    prior_registry_sha256: str = Field(pattern=_HEX_64)
    sampling_frame_sha256: str = Field(pattern=_HEX_64)
    retry_policy_sha256: str = Field(pattern=_HEX_64)
    sample_membership_sha256: str = Field(pattern=_HEX_64)
    denominator: int = Field(ge=150)
    paper_count: int = Field(ge=10)
    files: list[PackageFile] = Field(min_length=4)
    privacy: PackagePrivacy = Field(default_factory=PackagePrivacy)

    @model_validator(mode="after")
    def exact_file_contract(self) -> UnseenPackageManifest:
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("unseen package file paths must be unique")
        if self.role is PackageRole.REVIEWER and not self.actor_id.startswith("reviewer-"):
            raise ValueError("reviewer package requires a reviewer pseudonym")
        if self.role is PackageRole.ADJUDICATION and not self.actor_id.startswith("adjudicator-"):
            raise ValueError("adjudication package requires an adjudicator pseudonym")
        return self


class WorkspacePackage(StrictModel):
    role: PackageRole
    directory: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=_HEX_64)

    @model_validator(mode="after")
    def confined_directory(self) -> WorkspacePackage:
        _relative_path(self.directory)
        return self


class WorkspacePrivacy(StrictModel):
    contains_human_labels: Literal[False] = False
    reviewer_packages_are_isolated: Literal[True] = True
    adjudication_contains_reviewer_outputs: Literal[False] = False
    public_commit_forbidden: Literal[True] = True


class UnseenWorkspaceManifest(StrictModel):
    schema_version: Literal["unseen-annotation-workspace/0.1"] = WORKSPACE_SCHEMA_VERSION
    status: Literal["prepared-three-isolated-blank-packages"] = (
        "prepared-three-isolated-blank-packages"
    )
    run_seal: RunSealBinding
    freeze_manifest_sha256: str = Field(pattern=_HEX_64)
    execution_contract_sha256: str = Field(pattern=_HEX_64)
    corpus_definition_sha256: str = Field(pattern=_HEX_64)
    metadata_selection_sha256: str = Field(pattern=_HEX_64)
    prior_registry_sha256: str = Field(pattern=_HEX_64)
    sampling_frame_sha256: str = Field(pattern=_HEX_64)
    retry_policy_sha256: str = Field(pattern=_HEX_64)
    annotation_pool_sha256: str = Field(pattern=_HEX_64)
    sample_membership_sha256: str = Field(pattern=_HEX_64)
    denominator: int = Field(ge=150)
    eligible_atomic_result_target: int = Field(ge=100)
    control_target: int = Field(ge=50)
    paper_count: int = Field(ge=10)
    packages: list[WorkspacePackage] = Field(min_length=3, max_length=3)
    privacy: WorkspacePrivacy = Field(default_factory=WorkspacePrivacy)

    @model_validator(mode="after")
    def two_reviewers_one_adjudication(self) -> UnseenWorkspaceManifest:
        roles = Counter(package.role for package in self.packages)
        if roles != Counter({PackageRole.REVIEWER: 2, PackageRole.ADJUDICATION: 1}):
            raise ValueError("workspace requires two reviewer packages and one adjudication")
        directories = [package.directory for package in self.packages]
        if len(directories) != len(set(directories)):
            raise ValueError("workspace package directories must be distinct")
        if self.denominator != self.eligible_atomic_result_target + self.control_target:
            raise ValueError("workspace denominator differs from predeclared targets")
        return self


class ValidatedEvidence(StrictModel):
    source_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    exact_excerpt_sha256: str = Field(pattern=_HEX_64)
    character_start: int = Field(ge=0)
    character_end: int = Field(gt=0)
    line_number: int = Field(ge=1)


@dataclass(frozen=True)
class ValidatedUnseenResponse:
    actor_id: str
    response_sha256: str
    response_count: int
    evidence: tuple[tuple[ValidatedEvidence, ValidatedEvidence | None], ...]


@dataclass(frozen=True)
class _FrozenPaper:
    paper_id: str
    source_id: str
    source_sha256: str
    source_manifest_sha256: str
    layout_sha256: str
    pdf_path: Path | None
    byte_size: int
    page_text: dict[int, str]
    page_text_sha256: dict[int, str]


@dataclass(frozen=True)
class _PreparationContext:
    registry: PriorPaperRegistry
    selection: MetadataSelectionManifest
    sampling: SamplingFrame
    retry_policy: TransportRetryPolicy
    freeze: EngineeringFreezeManifest
    corpus: FrozenCorpusDefinition
    receipt: SealedRunReceipt
    verified_run: VerifiedRunSeal
    papers: dict[str, _FrozenPaper]
    pool: tuple[AnnotationPoolRecord, ...]
    items: tuple[UnseenAnnotationItem, ...]
    hashes: dict[str, str]


REVIEWER_PROTOCOL = f"""# Independent unseen-evaluation annotation

Protocol version: `{PROTOCOL_VERSION}`

This is one of two isolated reviewer packages for a frozen unseen evaluation. Work
independently. Do not inspect the peer package, adjudication package, pipeline outputs,
sampling strata, prior annotations, or scores. Do not communicate decisions before both
original responses are locked.

For every item, decide `result_bearing` as `yes`, `no`, or `uncertain`. A result is a
quantitative evaluation outcome or measured behavior. Headers, captions, sample counts,
setup values, parameters, and thresholds are not results merely because they contain a
number. For a result, also decide `origin`, `claim_scope`, and `atomicity`.

`paper_produced` includes a baseline actually rerun by the current paper. A copied or cited
number is `externally_sourced`. Do not infer origin from system ownership or from silence.
`primary` versus `secondary` concerns whether the current paper reports the result as its
own evaluation target rather than merely discussing prior work. `atomic` means one physical
result cell with one system/scope/metric/value setting.

`result_evidence` must cite the selected item page. `origin_evidence` is separate and may
cite another page in the same frozen PDF. Resolved paper-produced or external origin requires
origin evidence. Genuinely uncertain origin may leave it null when the paper supplies no
page-local cue. Every excerpt must be copied verbatim from one physical line on its declared
page. Never normalize whitespace, reconstruct wrapped text, join columns, or combine pages.

Edit only `response.jsonl`; preserve order, IDs, actor ID, and schema version. Keep all
decisions null until personally reviewed. This package is private and must not be committed.
"""


ADJUDICATION_PROTOCOL = f"""# Blank unseen-evaluation adjudication package

Protocol version: `{PROTOCOL_VERSION}`

This package is intentionally blank and isolated from both reviewer outputs. Do not begin
adjudication until the coordinator has independently locked both original responses and
computed a disagreement/uncertainty selection without changing the original denominator.
Never overwrite an original reviewer response or copy a peer response into this directory.

When adjudication is later authorized, apply the same result, origin, scope, atomicity, and
strict page-local evidence rules as the reviewer protocol. Resolved origin needs a separate
exact origin anchor; uncertain origin may have no anchor when the paper has no page-local cue.
This initial package contains no decisions, labels, reviewer identities, or reviewer outputs.
"""


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("artifact path must be normalized, relative, and confined")
    return path


def _annotation_item_id(payload: dict[str, Any]) -> str:
    return f"unseen_{sha256_bytes(canonical_json_bytes(payload))[:24]}"


def _regular_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink JSON file")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink JSONL file")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{label}:{line_number}: blank line")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _outside_public_repo(path: Path, public_repo_root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(public_repo_root.resolve())
    except ValueError:
        return
    raise ValueError(f"{label} must remain outside the public repository")


def _single_line_occurrence(excerpt: str, text: str) -> bool:
    return (
        bool(excerpt)
        and not any(character in _LINE_BREAKS for character in excerpt)
        and any(excerpt in line for line in text.splitlines())
    )


def _seal_binding(verified: VerifiedRunSeal) -> RunSealBinding:
    return RunSealBinding(
        seal_sha256=verified.seal_sha256,
        tree_sha256=verified.tree_sha256,
        file_count=verified.file_count,
        total_bytes=verified.total_bytes,
    )


def _seal_files(verified: VerifiedRunSeal) -> dict[str, dict[str, Any]]:
    return {entry["path"]: entry for entry in verified.files}


def _source_pdf(cache_relpath: str, source_project_root: Path) -> Path:
    relative = _relative_path(cache_relpath)
    path = source_project_root.joinpath(*relative.parts)
    try:
        path.resolve().relative_to(source_project_root.resolve())
    except ValueError as error:
        raise ValueError("cached source PDF escapes source project root") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError("cached source PDF must be regular and non-symlinked")
    return path


def _load_paper(
    *,
    paper_id: str,
    sealed_run_root: Path,
    source_project_root: Path | None,
    seal_files: dict[str, dict[str, Any]],
    freeze_sha256: str,
    execution_sha256: str,
) -> _FrozenPaper:
    prefix = f"{paper_id}/"
    manifest_rel = prefix + "source-manifest.json"
    layout_rel = prefix + "private/layout.json"
    run_rel = prefix + "run.json"
    binding_rel = prefix + f"private/{PAPER_BINDING_NAME}"
    for relative in (manifest_rel, layout_rel, run_rel, binding_rel):
        if relative not in seal_files:
            raise ValueError(f"{paper_id}: sealed run lacks a required paper artifact")
    manifest_path = sealed_run_root / manifest_rel
    layout_path = sealed_run_root / layout_rel
    run_path = sealed_run_root / run_rel
    binding_path = sealed_run_root / binding_rel
    manifest = _regular_json(manifest_path, f"{paper_id} source manifest")
    layout = _regular_json(layout_path, f"{paper_id} layout")
    binding = PaperFreezeBinding.model_validate(
        _regular_json(binding_path, f"{paper_id} freeze binding")
    )
    manifest_sha = sha256_file(manifest_path)
    layout_sha = sha256_file(layout_path)
    if (
        binding.paper_id != paper_id
        or binding.freeze_manifest_sha256 != freeze_sha256
        or binding.execution_contract_sha256 != execution_sha256
        or binding.run_manifest_sha256 != sha256_file(run_path)
        or binding.source_manifest_sha256 != manifest_sha
        or binding.layout_sha256 != layout_sha
    ):
        raise ValueError(f"{paper_id}: per-paper freeze binding differs from sealed artifacts")
    if manifest.get("paper_id") != paper_id:
        raise ValueError(f"{paper_id}: source manifest paper_id mismatch")
    source_id = layout.get("source_id")
    sources = manifest.get("sources")
    if not isinstance(source_id, str) or not isinstance(sources, list):
        raise ValueError(f"{paper_id}: source manifest/layout is incomplete")
    source_matches = [
        source
        for source in sources
        if isinstance(source, dict)
        and source.get("source_id") == source_id
        and source.get("role") == "paper"
    ]
    if len(source_matches) != 1:
        raise ValueError(f"{paper_id}: frozen paper source is not unique")
    source = source_matches[0]
    source_sha = source.get("sha256")
    byte_size = source.get("byte_size")
    cache_relpath = source.get("cache_relpath")
    if (
        not isinstance(source_sha, str)
        or re.fullmatch(_HEX_64, source_sha) is None
        or not isinstance(byte_size, int)
        or isinstance(byte_size, bool)
        or byte_size < 1
        or not isinstance(cache_relpath, str)
    ):
        raise ValueError(f"{paper_id}: frozen source fingerprint is invalid")
    pdf_path: Path | None = None
    if source_project_root is not None:
        pdf_path = _source_pdf(cache_relpath, source_project_root)
        if pdf_path.stat().st_size != byte_size or sha256_file(pdf_path) != source_sha:
            raise ValueError(f"{paper_id}: cached PDF differs from frozen source manifest")
    pages = layout.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"{paper_id}: frozen layout pages are unavailable")
    page_text: dict[int, str] = {}
    page_hashes: dict[int, str] = {}
    for raw_page in pages:
        if not isinstance(raw_page, dict):
            raise ValueError(f"{paper_id}: frozen layout page must be an object")
        page = raw_page.get("page")
        text = raw_page.get("text")
        if (
            not isinstance(page, int)
            or isinstance(page, bool)
            or page < 1
            or not isinstance(text, str)
            or page in page_text
        ):
            raise ValueError(f"{paper_id}: frozen layout page is invalid or duplicated")
        digest = sha256_bytes(text.encode("utf-8"))
        if raw_page.get("text_sha256") != digest:
            raise ValueError(f"{paper_id}: frozen page text digest mismatch")
        page_text[page] = text
        page_hashes[page] = digest
    return _FrozenPaper(
        paper_id=paper_id,
        source_id=source_id,
        source_sha256=source_sha,
        source_manifest_sha256=manifest_sha,
        layout_sha256=layout_sha,
        pdf_path=pdf_path,
        byte_size=byte_size,
        page_text=page_text,
        page_text_sha256=page_hashes,
    )


def _validate_attempts(
    attempts: list[TransportAttempt],
    *,
    policy: TransportRetryPolicy,
    paper_ids: set[str],
) -> None:
    if not attempts:
        raise ValueError("sealed unseen run must retain transport attempt records")
    call_ids = [attempt.call_id for attempt in attempts]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("transport attempt call IDs must be unique")
    if any(attempt.paper_id not in paper_ids for attempt in attempts):
        raise ValueError("transport attempts include a paper outside the frozen corpus")
    if any(attempt.stage not in REQUIRED_STAGES for attempt in attempts):
        raise ValueError("transport attempt uses a stage outside the frozen execution contract")
    grouped: dict[tuple[str, str, str], list[TransportAttempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[(attempt.paper_id, attempt.stage, attempt.request_slot)].append(attempt)
    allowed = set(policy.allowed_retry_reasons)
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: item.attempt_number)
        if len({item.request_fingerprint for item in ordered}) != 1:
            raise ValueError("a retry changed the frozen logical request fingerprint")
        expected_numbers = list(range(1, len(ordered) + 1))
        if [item.attempt_number for item in ordered] != expected_numbers:
            raise ValueError("transport attempt numbers must be contiguous from one")
        if len(ordered) > policy.maximum_attempts_per_request:
            raise ValueError("transport retry exceeded the predeclared attempt ceiling")
        for index, attempt in enumerate(ordered[1:], start=1):
            previous = ordered[index - 1]
            if previous.outcome is not AttemptOutcome.TRANSPORT_FAILURE:
                raise ValueError("only a technical transport failure may be retried")
            if previous.failure_reason not in allowed:
                raise ValueError("transport failure reason was not predeclared as retryable")
            if attempt.retry_reason is not previous.failure_reason:
                raise ValueError("retry reason does not match the preceding transport failure")


def _sample_pool(
    pool: tuple[AnnotationPoolRecord, ...], sampling: SamplingFrame
) -> tuple[AnnotationPoolRecord, ...]:
    quotas = {
        PoolStratum.ELIGIBLE_ATOMIC_RESULT_CANDIDATE: sampling.eligible_atomic_result_target,
        **sampling.control_targets,
    }
    selected: list[AnnotationPoolRecord] = []
    for stratum, target in quotas.items():
        candidates = [record for record in pool if record.pipeline_stratum is stratum]
        if len(candidates) < target:
            raise ValueError(f"sealed annotation pool cannot satisfy {stratum.value} target")
        ranked = sorted(
            candidates,
            key=lambda record: (
                sha256_bytes(f"{sampling.randomization_seed}\0{record.unit_id}".encode()),
                record.unit_id,
            ),
        )
        selected.extend(ranked[:target])
    selected.sort(key=lambda item: (item.paper_id, item.page, item.locator, item.unit_id))
    if len(selected) != sampling.denominator:
        raise ValueError("deterministic sample differs from predeclared denominator")
    if len({item.unit_id for item in selected}) != len(selected):
        raise ValueError("deterministic sample contains duplicate units")
    if len({item.paper_id for item in selected}) < sampling.minimum_sampled_papers:
        raise ValueError("deterministic sample misses the predeclared paper coverage")
    return tuple(selected)


def _project_items(
    selected: tuple[AnnotationPoolRecord, ...],
) -> tuple[UnseenAnnotationItem, ...]:
    items: list[UnseenAnnotationItem] = []
    for record in selected:
        payload = {
            key: value
            for key, value in record.model_dump(mode="json").items()
            if key
            not in {
                "schema_version",
                "unit_id",
                "pipeline_stratum",
                "source_record_path",
                "source_record_sha256",
            }
        }
        payload["schema_version"] = ITEM_SCHEMA_VERSION
        items.append(UnseenAnnotationItem(item_id=_annotation_item_id(payload), **payload))
    ids = [item.item_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("projected unseen annotation item IDs must be unique")
    return tuple(items)


def _validate_pool_records(
    pool: tuple[AnnotationPoolRecord, ...],
    *,
    papers: dict[str, _FrozenPaper],
    seal_files: dict[str, dict[str, Any]],
) -> None:
    unit_ids = [record.unit_id for record in pool]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("sealed annotation pool unit IDs must be unique")
    physical_ids: list[tuple[str, int, str, str]] = []
    for record in pool:
        paper = papers.get(record.paper_id)
        if paper is None:
            raise ValueError("annotation pool includes a paper outside metadata selection")
        if (
            record.source_id != paper.source_id
            or record.source_sha256 != paper.source_sha256
            or record.source_manifest_sha256 != paper.source_manifest_sha256
            or record.layout_sha256 != paper.layout_sha256
            or paper.page_text_sha256.get(record.page) != record.page_text_sha256
        ):
            raise ValueError(f"{record.paper_id}: annotation pool source binding mismatch")
        page_text = paper.page_text.get(record.page)
        if page_text is None or not _single_line_occurrence(record.source_excerpt, page_text):
            raise ValueError(
                f"{record.paper_id}: pool excerpt is absent from its exact frozen page line"
            )
        sealed_record = seal_files.get(record.source_record_path)
        if sealed_record is None or sealed_record["sha256"] != record.source_record_sha256:
            raise ValueError("annotation pool source record is absent or hash-mismatched")
        physical_ids.append(
            (
                record.source_id,
                record.page,
                record.locator,
                record.source_excerpt_sha256,
            )
        )
    if len(physical_ids) != len(set(physical_ids)):
        raise ValueError("annotation pool duplicates a physical source unit")


def _load_context(
    *,
    metadata_selection_path: Path,
    prior_registry_path: Path,
    freeze_manifest_path: Path,
    sampling_frame_path: Path,
    retry_policy_path: Path,
    sealed_run_root: Path,
    source_project_root: Path | None,
) -> _PreparationContext:
    hashes = {
        "metadata": sha256_file(metadata_selection_path),
        "registry": sha256_file(prior_registry_path),
        "freeze": sha256_file(freeze_manifest_path),
        "sampling": sha256_file(sampling_frame_path),
        "retry": sha256_file(retry_policy_path),
    }
    registry = PriorPaperRegistry.model_validate(
        _regular_json(prior_registry_path, "prior paper registry")
    )
    freeze = EngineeringFreezeManifest.model_validate(
        _regular_json(freeze_manifest_path, "engineering freeze manifest")
    )
    selection = MetadataSelectionManifest.model_validate(
        _regular_json(metadata_selection_path, "metadata-only paper selection")
    )
    if (
        selection.engineering_freeze_sha256 != hashes["freeze"]
        or selection.prior_registry_sha256 != hashes["registry"]
    ):
        raise ValueError("metadata selection is bound to another freeze or prior registry")
    selected_ids = {paper.paper_id for paper in selection.papers}
    prior_ids = set(registry.inspected_or_development_paper_ids)
    overlap = sorted(selected_ids & prior_ids)
    if overlap:
        raise ValueError("metadata selection contains an inspected/development paper ID")
    sampling = SamplingFrame.model_validate(
        _regular_json(sampling_frame_path, "predeclared sampling frame")
    )
    retry_policy = TransportRetryPolicy.model_validate(
        _regular_json(retry_policy_path, "predeclared transport retry policy")
    )
    if (
        sampling.engineering_freeze_sha256 != hashes["freeze"]
        or sampling.metadata_selection_sha256 != hashes["metadata"]
        or retry_policy.engineering_freeze_sha256 != hashes["freeze"]
        or retry_policy.metadata_selection_sha256 != hashes["metadata"]
    ):
        raise ValueError("pre-run sampling/retry contract is bound to another freeze or selection")

    verified = verify_run_seal(sealed_run_root)
    seal_files = _seal_files(verified)
    for required in (CORPUS_DEFINITION_PATH, RUN_RECEIPT_PATH, POOL_PATH, ATTEMPTS_PATH):
        if required not in seal_files:
            raise ValueError("sealed unseen run lacks a required freeze/evaluation artifact")
    corpus_path = sealed_run_root / CORPUS_DEFINITION_PATH
    corpus = FrozenCorpusDefinition.model_validate(
        _regular_json(corpus_path, "sealed corpus definition")
    )
    hashes["corpus"] = sha256_file(corpus_path)
    expected_ids = [paper.paper_id for paper in selection.papers]
    if (
        corpus.engineering_freeze_sha256 != hashes["freeze"]
        or corpus.metadata_selection_sha256 != hashes["metadata"]
        or corpus.paper_ids != expected_ids
    ):
        raise ValueError("sealed corpus membership differs from metadata-only selection")
    receipt = SealedRunReceipt.model_validate(
        _regular_json(sealed_run_root / RUN_RECEIPT_PATH, "sealed unseen run receipt")
    )
    if (
        receipt.freeze_manifest_sha256 != hashes["freeze"]
        or receipt.execution_contract_sha256 != freeze.execution_contract_sha256
        or receipt.metadata_selection_sha256 != hashes["metadata"]
        or receipt.prior_registry_sha256 != hashes["registry"]
        or receipt.sampling_frame_sha256 != hashes["sampling"]
        or receipt.retry_policy_sha256 != hashes["retry"]
        or receipt.corpus_definition_sha256 != hashes["corpus"]
    ):
        raise ValueError("sealed run receipt differs from the pre-run frozen contract")
    pool_path = sealed_run_root / POOL_PATH
    attempts_path = sealed_run_root / ATTEMPTS_PATH
    if receipt.annotation_pool_sha256 != sha256_file(
        pool_path
    ) or receipt.transport_attempts_sha256 != sha256_file(attempts_path):
        raise ValueError("sealed run receipt artifact hashes are stale")

    papers = {
        paper_id: _load_paper(
            paper_id=paper_id,
            sealed_run_root=sealed_run_root,
            source_project_root=source_project_root,
            seal_files=seal_files,
            freeze_sha256=hashes["freeze"],
            execution_sha256=freeze.execution_contract_sha256,
        )
        for paper_id in expected_ids
    }
    pool = tuple(
        AnnotationPoolRecord.model_validate(row)
        for row in _read_jsonl(pool_path, "sealed annotation pool")
    )
    if len(pool) != receipt.annotation_pool_records:
        raise ValueError("sealed annotation pool count differs from run receipt")
    _validate_pool_records(pool, papers=papers, seal_files=seal_files)
    attempts = [
        TransportAttempt.model_validate(row)
        for row in _read_jsonl(attempts_path, "sealed transport attempts")
    ]
    if len(attempts) != receipt.transport_attempt_records:
        raise ValueError("transport attempt count differs from run receipt")
    _validate_attempts(attempts, policy=retry_policy, paper_ids=selected_ids)
    selected = _sample_pool(pool, sampling)
    items = _project_items(selected)
    return _PreparationContext(
        registry=registry,
        selection=selection,
        sampling=sampling,
        retry_policy=retry_policy,
        freeze=freeze,
        corpus=corpus,
        receipt=receipt,
        verified_run=verified,
        papers=papers,
        pool=pool,
        items=items,
        hashes=hashes,
    )


def _package_file(path: Path, root: Path, *, records: int | None = None) -> PackageFile:
    return PackageFile(
        path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        records=records,
        size_bytes=path.stat().st_size,
    )


def _make_read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~0o222)


def _frozen_stages_sha256(freeze: EngineeringFreezeManifest) -> str:
    return sha256_bytes(
        canonical_json_bytes([stage.model_dump(mode="json") for stage in freeze.stages])
    )


def _sample_membership_sha256(context: _PreparationContext) -> str:
    selected = _sample_pool(context.pool, context.sampling)
    payload = [
        {
            "pipeline_stratum": record.pipeline_stratum.value,
            "unit_id": record.unit_id,
        }
        for record in selected
    ]
    return sha256_bytes(canonical_json_bytes(payload))


def _package_manifest_fields(context: _PreparationContext) -> dict[str, Any]:
    sampled_paper_ids = {item.paper_id for item in context.items}
    return {
        "run_seal": _seal_binding(context.verified_run),
        "freeze_manifest_sha256": context.hashes["freeze"],
        "execution_contract_sha256": context.freeze.execution_contract_sha256,
        "corpus_definition_sha256": context.hashes["corpus"],
        "code_git_commit": context.freeze.code_git_commit,
        "code_source_tree_sha256": context.freeze.code_source_tree_sha256,
        "dependency_lock_sha256": context.freeze.dependency_lock_sha256,
        "thresholds_sha256": context.freeze.thresholds_sha256,
        "attribution_policy_sha256": context.freeze.attribution_policy_sha256,
        "frozen_stages_sha256": _frozen_stages_sha256(context.freeze),
        "metadata_selection_sha256": context.hashes["metadata"],
        "prior_registry_sha256": context.hashes["registry"],
        "sampling_frame_sha256": context.hashes["sampling"],
        "retry_policy_sha256": context.hashes["retry"],
        "sample_membership_sha256": _sample_membership_sha256(context),
        "denominator": context.sampling.denominator,
        "paper_count": len(sampled_paper_ids),
    }


def _write_package(
    *,
    tree: Path,
    role: PackageRole,
    actor_id: str,
    context: _PreparationContext,
) -> UnseenPackageManifest:
    tree.mkdir(parents=True)
    protocol_path = tree / "protocol.md"
    protocol_path.write_text(
        REVIEWER_PROTOCOL if role is PackageRole.REVIEWER else ADJUDICATION_PROTOCOL,
        encoding="utf-8",
    )
    items_path = tree / "items.jsonl"
    write_jsonl(items_path, [item.model_dump(mode="json") for item in context.items])
    response_path = tree / "response.jsonl"
    blank_responses = [
        UnseenAnnotationResponse(item_id=item.item_id, actor_id=actor_id) for item in context.items
    ]
    write_jsonl(
        response_path,
        [response.model_dump(mode="json") for response in blank_responses],
    )

    pdf_paths: list[Path] = []
    sampled_paper_ids = sorted({item.paper_id for item in context.items})
    for paper_id in sampled_paper_ids:
        paper = context.papers[paper_id]
        if paper.pdf_path is None:
            raise ValueError("package creation requires hash-verified private source PDFs")
        destination = tree / "pdfs" / f"{paper_id}.pdf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paper.pdf_path, destination)
        if (
            destination.stat().st_size != paper.byte_size
            or sha256_file(destination) != paper.source_sha256
        ):
            raise ValueError(f"{paper_id}: copied PDF differs from frozen source")
        pdf_paths.append(destination)

    for immutable in (protocol_path, items_path, *pdf_paths):
        _make_read_only(immutable)
    files = [
        _package_file(protocol_path, tree),
        _package_file(items_path, tree, records=len(context.items)),
        _package_file(response_path, tree, records=len(context.items)),
        *(_package_file(path, tree) for path in pdf_paths),
    ]
    manifest = UnseenPackageManifest(
        role=role,
        actor_id=actor_id,
        files=files,
        **_package_manifest_fields(context),
    )
    manifest_path = tree / "manifest.json"
    write_json(manifest_path, manifest)
    _make_read_only(manifest_path)
    return manifest


def _package_specs(
    reviewer_ids: tuple[str, str], adjudicator_id: str
) -> tuple[tuple[str, PackageRole, str], ...]:
    if len(set((*reviewer_ids, adjudicator_id))) != 3:
        raise ValueError("reviewer and adjudicator pseudonyms must be distinct")
    for reviewer in reviewer_ids:
        if re.fullmatch(_ACTOR_ID, reviewer) is None or not reviewer.startswith("reviewer-"):
            raise ValueError("each reviewer needs a safe reviewer pseudonym")
    if re.fullmatch(_ACTOR_ID, adjudicator_id) is None or not adjudicator_id.startswith(
        "adjudicator-"
    ):
        raise ValueError("the adjudicator needs a safe adjudicator pseudonym")
    return (
        ("package-a", PackageRole.REVIEWER, reviewer_ids[0]),
        ("package-b", PackageRole.REVIEWER, reviewer_ids[1]),
        ("package-c", PackageRole.ADJUDICATION, adjudicator_id),
    )


def _private_orchestration_paths(
    *,
    metadata_selection_path: Path,
    prior_registry_path: Path,
    freeze_manifest_path: Path,
    sampling_frame_path: Path,
    retry_policy_path: Path,
    sealed_run_root: Path,
    source_project_root: Path,
    output_dir: Path,
    public_repo_root: Path,
) -> None:
    paths = {
        "metadata-only selection": metadata_selection_path,
        "prior-paper registry": prior_registry_path,
        "engineering freeze": freeze_manifest_path,
        "sampling frame": sampling_frame_path,
        "retry policy": retry_policy_path,
        "sealed unseen run": sealed_run_root,
        "private source project": source_project_root,
        "unseen annotation workspace": output_dir,
    }
    for label, path in paths.items():
        _outside_public_repo(path, public_repo_root, label)


def _workspace_manifest(
    *,
    context: _PreparationContext,
    packages: list[WorkspacePackage],
) -> UnseenWorkspaceManifest:
    return UnseenWorkspaceManifest(
        run_seal=_seal_binding(context.verified_run),
        freeze_manifest_sha256=context.hashes["freeze"],
        execution_contract_sha256=context.freeze.execution_contract_sha256,
        corpus_definition_sha256=context.hashes["corpus"],
        metadata_selection_sha256=context.hashes["metadata"],
        prior_registry_sha256=context.hashes["registry"],
        sampling_frame_sha256=context.hashes["sampling"],
        retry_policy_sha256=context.hashes["retry"],
        annotation_pool_sha256=context.receipt.annotation_pool_sha256,
        sample_membership_sha256=_sample_membership_sha256(context),
        denominator=context.sampling.denominator,
        eligible_atomic_result_target=context.sampling.eligible_atomic_result_target,
        control_target=sum(context.sampling.control_targets.values()),
        paper_count=len({item.paper_id for item in context.items}),
        packages=packages,
    )


def prepare_unseen_annotation_workspace(
    *,
    metadata_selection_path: Path,
    prior_registry_path: Path,
    freeze_manifest_path: Path,
    sampling_frame_path: Path,
    retry_policy_path: Path,
    sealed_run_root: Path,
    source_project_root: Path,
    output_dir: Path,
    public_repo_root: Path,
    reviewer_ids: tuple[str, str] = ("reviewer-a", "reviewer-b"),
    adjudicator_id: str = "adjudicator-primary",
) -> tuple[UnseenWorkspaceManifest, str]:
    """Create three isolated, deterministic, all-null packages after the unseen run seal."""

    specs = _package_specs(reviewer_ids, adjudicator_id)
    _private_orchestration_paths(
        metadata_selection_path=metadata_selection_path,
        prior_registry_path=prior_registry_path,
        freeze_manifest_path=freeze_manifest_path,
        sampling_frame_path=sampling_frame_path,
        retry_policy_path=retry_policy_path,
        sealed_run_root=sealed_run_root,
        source_project_root=source_project_root,
        output_dir=output_dir,
        public_repo_root=public_repo_root,
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite an unseen annotation workspace")
    if source_project_root.is_symlink() or not source_project_root.is_dir():
        raise ValueError("private source project must be a regular directory")
    context = _load_context(
        metadata_selection_path=metadata_selection_path,
        prior_registry_path=prior_registry_path,
        freeze_manifest_path=freeze_manifest_path,
        sampling_frame_path=sampling_frame_path,
        retry_policy_path=retry_policy_path,
        sealed_run_root=sealed_run_root,
        source_project_root=source_project_root,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        packages: list[WorkspacePackage] = []
        for directory, role, actor_id in specs:
            _write_package(
                tree=staging / directory,
                role=role,
                actor_id=actor_id,
                context=context,
            )
            packages.append(
                WorkspacePackage(
                    role=role,
                    directory=directory,
                    manifest_sha256=sha256_file(staging / directory / "manifest.json"),
                )
            )
        manifest = _workspace_manifest(context=context, packages=packages)
        workspace_manifest_path = staging / "manifest.json"
        write_json(workspace_manifest_path, manifest)
        _make_read_only(workspace_manifest_path)
        _validate_initial_workspace(
            workspace_dir=staging,
            context=context,
            specs=specs,
        )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest, sha256_file(output_dir / "manifest.json")


def _package_file_contract(
    *,
    package_dir: Path,
    manifest: UnseenPackageManifest,
    expected_pdf_ids: set[str],
    response_is_mutable: bool,
) -> dict[str, Path]:
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise ValueError("unseen annotation package must be a regular directory")
    for path in package_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError("unseen annotation package must not contain symbolic links")
    expected_paths = [
        "protocol.md",
        "items.jsonl",
        "response.jsonl",
        *(f"pdfs/{paper_id}.pdf" for paper_id in sorted(expected_pdf_ids)),
    ]
    declared_paths = [file.path for file in manifest.files]
    if declared_paths != expected_paths:
        raise ValueError("unseen annotation package has a changed file declaration")
    actual_files = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != {"manifest.json", *expected_paths}:
        raise ValueError("unseen annotation package has an unexpected file contract")
    actual_dirs = {
        path.relative_to(package_dir).as_posix() for path in package_dir.rglob("*") if path.is_dir()
    }
    if actual_dirs != {"pdfs"}:
        raise ValueError("unseen annotation package has an unexpected directory contract")

    expected_records = {
        "items.jsonl": manifest.denominator,
        "response.jsonl": manifest.denominator,
    }
    paths: dict[str, Path] = {}
    for declared in manifest.files:
        path = package_dir.joinpath(*_relative_path(declared.path).parts)
        records = expected_records.get(declared.path)
        if declared.records != records:
            raise ValueError(f"unexpected record count declaration: {declared.path}")
        if not (response_is_mutable and declared.path == "response.jsonl"):
            if path.stat().st_size != declared.size_bytes:
                raise ValueError(f"unseen package file size mismatch: {declared.path}")
            if sha256_file(path) != declared.sha256:
                raise ValueError(f"unseen package file hash mismatch: {declared.path}")
        paths[declared.path] = path
    return paths


def _validate_package_against_context(
    *,
    package_dir: Path,
    context: _PreparationContext,
    role: PackageRole,
    actor_id: str,
    expected_manifest_sha256: str,
) -> UnseenPackageManifest:
    manifest_path = package_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("unseen package manifest must be a regular non-symlink file")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("unseen package manifest differs from workspace binding")
    manifest = UnseenPackageManifest.model_validate(
        _regular_json(manifest_path, "unseen package manifest")
    )
    if manifest.role is not role or manifest.actor_id != actor_id:
        raise ValueError("unseen package role or actor differs from its isolated assignment")
    for field, expected in _package_manifest_fields(context).items():
        if getattr(manifest, field) != expected:
            raise ValueError(f"unseen package binding differs from freeze: {field}")
    sampled_paper_ids = {item.paper_id for item in context.items}
    paths = _package_file_contract(
        package_dir=package_dir,
        manifest=manifest,
        expected_pdf_ids=sampled_paper_ids,
        response_is_mutable=False,
    )
    expected_protocol = REVIEWER_PROTOCOL if role is PackageRole.REVIEWER else ADJUDICATION_PROTOCOL
    if paths["protocol.md"].read_text(encoding="utf-8") != expected_protocol:
        raise ValueError("unseen annotation protocol differs from its frozen version")
    items = [
        UnseenAnnotationItem.model_validate(row)
        for row in _read_jsonl(paths["items.jsonl"], "unseen annotation items")
    ]
    if items != list(context.items):
        raise ValueError("unseen package items differ from deterministic frozen sampling")
    responses = [
        UnseenAnnotationResponse.model_validate(row)
        for row in _read_jsonl(paths["response.jsonl"], "unseen blank responses")
    ]
    if [response.item_id for response in responses] != [item.item_id for item in items]:
        raise ValueError("unseen blank responses changed item membership or order")
    if any(response.actor_id != actor_id for response in responses):
        raise ValueError("unseen package contains a wrong or mixed actor")
    if any(not response.is_blank for response in responses):
        raise ValueError("initial unseen annotation responses must be entirely blank")
    for paper_id in sampled_paper_ids:
        paper = context.papers[paper_id]
        pdf = paths[f"pdfs/{paper_id}.pdf"]
        if pdf.stat().st_size != paper.byte_size or sha256_file(pdf) != paper.source_sha256:
            raise ValueError(f"{paper_id}: package PDF differs from frozen source")
    return manifest


def _validate_initial_workspace(
    *,
    workspace_dir: Path,
    context: _PreparationContext,
    specs: tuple[tuple[str, PackageRole, str], ...],
) -> UnseenWorkspaceManifest:
    if workspace_dir.is_symlink() or not workspace_dir.is_dir():
        raise ValueError("unseen annotation workspace must be a regular directory")
    for path in workspace_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError("unseen annotation workspace must not contain symbolic links")
    expected_top = {"manifest.json", *(directory for directory, _, _ in specs)}
    if {path.name for path in workspace_dir.iterdir()} != expected_top:
        raise ValueError("unseen annotation workspace has an unexpected top-level contract")
    workspace_manifest_path = workspace_dir / "manifest.json"
    manifest = UnseenWorkspaceManifest.model_validate(
        _regular_json(workspace_manifest_path, "unseen workspace manifest")
    )
    actual_packages: list[WorkspacePackage] = []
    item_digests: set[str] = set()
    for directory, role, actor_id in specs:
        package_manifest_path = workspace_dir / directory / "manifest.json"
        package_digest = sha256_file(package_manifest_path)
        package_manifest = _validate_package_against_context(
            package_dir=workspace_dir / directory,
            context=context,
            role=role,
            actor_id=actor_id,
            expected_manifest_sha256=package_digest,
        )
        item_file = next(file for file in package_manifest.files if file.path == "items.jsonl")
        item_digests.add(item_file.sha256)
        actual_packages.append(
            WorkspacePackage(
                role=role,
                directory=directory,
                manifest_sha256=package_digest,
            )
        )
    if len(item_digests) != 1:
        raise ValueError("isolated unseen packages do not contain identical blinded items")
    expected_manifest = _workspace_manifest(context=context, packages=actual_packages)
    if manifest != expected_manifest:
        raise ValueError("unseen workspace manifest differs from the frozen package set")
    return manifest


def validate_initial_unseen_annotation_workspace(
    *,
    workspace_dir: Path,
    metadata_selection_path: Path,
    prior_registry_path: Path,
    freeze_manifest_path: Path,
    sampling_frame_path: Path,
    retry_policy_path: Path,
    sealed_run_root: Path,
    source_project_root: Path,
    public_repo_root: Path,
    reviewer_ids: tuple[str, str] = ("reviewer-a", "reviewer-b"),
    adjudicator_id: str = "adjudicator-primary",
) -> UnseenWorkspaceManifest:
    """Re-derive and verify all three initial packages before distribution."""

    specs = _package_specs(reviewer_ids, adjudicator_id)
    _private_orchestration_paths(
        metadata_selection_path=metadata_selection_path,
        prior_registry_path=prior_registry_path,
        freeze_manifest_path=freeze_manifest_path,
        sampling_frame_path=sampling_frame_path,
        retry_policy_path=retry_policy_path,
        sealed_run_root=sealed_run_root,
        source_project_root=source_project_root,
        output_dir=workspace_dir,
        public_repo_root=public_repo_root,
    )
    context = _load_context(
        metadata_selection_path=metadata_selection_path,
        prior_registry_path=prior_registry_path,
        freeze_manifest_path=freeze_manifest_path,
        sampling_frame_path=sampling_frame_path,
        retry_policy_path=retry_policy_path,
        sealed_run_root=sealed_run_root,
        source_project_root=source_project_root,
    )
    return _validate_initial_workspace(
        workspace_dir=workspace_dir,
        context=context,
        specs=specs,
    )


def _bind_evidence(
    anchor: EvidenceAnchor,
    *,
    item: UnseenAnnotationItem,
    paper: _FrozenPaper,
) -> ValidatedEvidence:
    if anchor.source_id != item.source_id or anchor.source_id != paper.source_id:
        raise ValueError(f"{item.item_id}: evidence source differs from frozen item source")
    text = paper.page_text.get(anchor.page)
    if text is None or not _single_line_occurrence(anchor.exact_excerpt, text):
        raise ValueError(
            f"{item.item_id}: evidence is absent from one physical line of declared page "
            f"{anchor.page}"
        )
    start = text.index(anchor.exact_excerpt)
    return ValidatedEvidence(
        source_id=anchor.source_id,
        page=anchor.page,
        exact_excerpt_sha256=hashlib.sha256(anchor.exact_excerpt.encode("utf-8")).hexdigest(),
        character_start=start,
        character_end=start + len(anchor.exact_excerpt),
        line_number=text.count("\n", 0, start) + 1,
    )


def _load_completed_reviewer_package(
    *,
    package_dir: Path,
    sealed_run_root: Path,
    expected_manifest_sha256: str,
) -> tuple[
    UnseenPackageManifest,
    list[UnseenAnnotationItem],
    list[UnseenAnnotationResponse],
    dict[str, _FrozenPaper],
]:
    if re.fullmatch(_HEX_64, expected_manifest_sha256) is None:
        raise ValueError("expected unseen package manifest SHA-256 is invalid")
    manifest_path = package_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("unseen reviewer manifest must be a regular non-symlink file")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("unseen reviewer manifest differs from preserved handoff receipt")
    manifest = UnseenPackageManifest.model_validate(
        _regular_json(manifest_path, "completed unseen reviewer manifest")
    )
    if manifest.role is not PackageRole.REVIEWER:
        raise ValueError("only an original reviewer package can be validated here")
    verified = verify_run_seal(sealed_run_root)
    if manifest.run_seal != _seal_binding(verified):
        raise ValueError("unseen reviewer package is bound to another sealed run")
    receipt = SealedRunReceipt.model_validate(
        _regular_json(sealed_run_root / RUN_RECEIPT_PATH, "sealed unseen run receipt")
    )
    receipt_bindings = {
        "freeze_manifest_sha256": receipt.freeze_manifest_sha256,
        "execution_contract_sha256": receipt.execution_contract_sha256,
        "corpus_definition_sha256": receipt.corpus_definition_sha256,
        "metadata_selection_sha256": receipt.metadata_selection_sha256,
        "prior_registry_sha256": receipt.prior_registry_sha256,
        "sampling_frame_sha256": receipt.sampling_frame_sha256,
        "retry_policy_sha256": receipt.retry_policy_sha256,
    }
    for field, expected in receipt_bindings.items():
        if getattr(manifest, field) != expected:
            raise ValueError(f"unseen reviewer package differs from run receipt: {field}")

    declared = {file.path: file for file in manifest.files}
    item_ref = declared.get("items.jsonl")
    if item_ref is None:
        raise ValueError("unseen reviewer package lacks its item file declaration")
    item_path = package_dir / "items.jsonl"
    if item_path.is_symlink() or not item_path.is_file():
        raise ValueError("unseen reviewer items must be a regular non-symlink file")
    if item_path.stat().st_size != item_ref.size_bytes or sha256_file(item_path) != item_ref.sha256:
        raise ValueError("unseen reviewer items differ from the blank handoff")
    items = [
        UnseenAnnotationItem.model_validate(row)
        for row in _read_jsonl(item_path, "completed unseen reviewer items")
    ]
    paper_ids = {item.paper_id for item in items}
    paths = _package_file_contract(
        package_dir=package_dir,
        manifest=manifest,
        expected_pdf_ids=paper_ids,
        response_is_mutable=True,
    )
    if paths["protocol.md"].read_text(encoding="utf-8") != REVIEWER_PROTOCOL:
        raise ValueError("completed reviewer protocol differs from frozen reviewer protocol")
    if len(items) != manifest.denominator:
        raise ValueError("completed reviewer item denominator differs from its manifest")
    if len(paper_ids) != manifest.paper_count:
        raise ValueError("completed reviewer paper count differs from its manifest")
    responses = [
        UnseenAnnotationResponse.model_validate(row)
        for row in _read_jsonl(paths["response.jsonl"], "completed unseen responses")
    ]
    if [response.item_id for response in responses] != [item.item_id for item in items]:
        raise ValueError("completed unseen responses changed item order or denominator")
    if any(response.actor_id != manifest.actor_id for response in responses):
        raise ValueError("completed unseen responses contain a wrong or mixed reviewer")
    if any(response.is_blank for response in responses):
        raise ValueError("completed unseen reviewer response still contains blank decisions")

    seal_files = _seal_files(verified)
    papers = {
        paper_id: _load_paper(
            paper_id=paper_id,
            sealed_run_root=sealed_run_root,
            source_project_root=None,
            seal_files=seal_files,
            freeze_sha256=manifest.freeze_manifest_sha256,
            execution_sha256=manifest.execution_contract_sha256,
        )
        for paper_id in sorted(paper_ids)
    }
    for item in items:
        paper = papers[item.paper_id]
        if (
            item.source_id != paper.source_id
            or item.source_sha256 != paper.source_sha256
            or item.source_manifest_sha256 != paper.source_manifest_sha256
            or item.layout_sha256 != paper.layout_sha256
            or item.page_text_sha256 != paper.page_text_sha256.get(item.page)
        ):
            raise ValueError(f"{item.item_id}: item binding differs from sealed source/layout")
        text = paper.page_text.get(item.page)
        if text is None or not _single_line_occurrence(item.source_excerpt, text):
            raise ValueError(f"{item.item_id}: item excerpt differs from exact sealed page")
    for paper_id, paper in papers.items():
        pdf = paths[f"pdfs/{paper_id}.pdf"]
        if pdf.stat().st_size != paper.byte_size or sha256_file(pdf) != paper.source_sha256:
            raise ValueError(f"{paper_id}: reviewer PDF differs from frozen source")
    return manifest, items, responses, papers


def validate_completed_unseen_reviewer_response(
    *,
    package_dir: Path,
    sealed_run_root: Path,
    public_repo_root: Path,
    expected_manifest_sha256: str,
) -> ValidatedUnseenResponse:
    """Validate one locked reviewer response and both kinds of exact evidence."""

    _outside_public_repo(package_dir, public_repo_root, "unseen reviewer package")
    _outside_public_repo(sealed_run_root, public_repo_root, "sealed unseen run")
    manifest, items, responses, papers = _load_completed_reviewer_package(
        package_dir=package_dir,
        sealed_run_root=sealed_run_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    evidence: list[tuple[ValidatedEvidence, ValidatedEvidence | None]] = []
    for item, response in zip(items, responses, strict=True):
        assert response.result_evidence is not None
        if response.result_evidence.page != item.page:
            raise ValueError(f"{item.item_id}: result evidence must cite the selected item page")
        paper = papers[item.paper_id]
        result_binding = _bind_evidence(
            response.result_evidence,
            item=item,
            paper=paper,
        )
        origin_binding = (
            _bind_evidence(response.origin_evidence, item=item, paper=paper)
            if response.origin_evidence is not None
            else None
        )
        evidence.append((result_binding, origin_binding))
    return ValidatedUnseenResponse(
        actor_id=manifest.actor_id,
        response_sha256=sha256_file(package_dir / "response.jsonl"),
        response_count=len(responses),
        evidence=tuple(evidence),
    )
