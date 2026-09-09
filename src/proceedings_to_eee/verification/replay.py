"""Offline replay of the independent verifier over an already completed run tree.

Re-running extraction to evaluate a verifier can produce a different candidate set,
confounding the verifier's effect with extraction variance.

This module instead replays the verifier against a frozen run: it reads the recorded
candidates and the recorded result blocks, binds them with the same deterministic
function the pipeline uses, and issues exactly the verification calls the pipeline
would have issued. The source run tree is opened read-only; every artifact is written
under a separate output root.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import ClaimType, ExportStatus
from proceedings_to_eee.extraction.result_blocks import ResultBlock
from proceedings_to_eee.io import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from proceedings_to_eee.providers.openrouter import (
    OpenRouterClient,
    ProviderCall,
    ProviderRequestRejectedError,
    ProviderResponseValidationError,
    completion_token_parameter_for_model,
    public_provider_call,
    structured_request_contract_from_call,
)
from proceedings_to_eee.verification.binding import bind_candidate_block, frozen_evidence_block
from proceedings_to_eee.verification.independent import (
    VERIFIER_REQUEST_SETTINGS,
    VERIFIER_SCHEMA_NAME,
    VERIFIER_SYSTEM_PROMPT,
    CandidateVerification,
    FrozenEvidenceBlock,
    IndependentDecision,
    VerificationRequest,
    contextualize_verification,
    verification_prompt,
    verifier_assessment_sha256,
    verifier_evidence_block_sha256,
    verifier_request_contract,
    verify_candidate,
)

REPLAY_SCHEMA_VERSION = "verifier-replay/0.3"
REPLAY_CHECKPOINT_SCHEMA_VERSION = "verifier-replay-checkpoint/0.1"
REPLAY_CHECKPOINT_CONTRACT_VERSION = "verifier-replay-checkpoint-contract/0.1"
REPLAY_ENTRY_SCHEMA_VERSION = "verifier-replay-entry/0.1"
REPLAY_ENTRY_BINDING_VERSION = "verifier-replay-entry-binding/0.1"
REPLAY_CODE_BINDING_VERSION = "verifier-replay-code-binding/0.1"

_OBSERVATIONS = "observations.jsonl"
_RESULT_BLOCKS = Path("private") / "result-blocks.json"
_REFERENCE_SCORE = "reference-score.json"
_RUN = "run.json"
_CHECKPOINT = "replay-checkpoint.json"
_OUTPUT_ARTIFACTS = (
    _CHECKPOINT,
    "bindings.json",
    "verifications.jsonl",
    "verifier-calls.jsonl",
    "verifier-errors.jsonl",
)


class ReplayCheckpointError(ValueError):
    """The output root cannot be resumed under the current immutable binding."""


class ReplayScope(StrEnum):
    """Which recorded candidates the replay sends to the verifier."""

    EXPORT_GATE = "export_gate"
    """Exactly the pipeline gate: primary_result candidates that reached export."""

    PRIMARY = "primary"
    """Every primary_result candidate, including those already held for review."""

    ALL = "all"
    """Every recorded candidate."""


_EXPORT_GATE_STATUSES = frozenset({ExportStatus.ELIGIBLE, ExportStatus.EXPORTED})


@dataclass(frozen=True)
class ReplaySettings:
    """Inputs for one replay. ``run_root`` is never written to."""

    run_root: Path
    output_root: Path
    verifier_model: str
    scope: ReplayScope = ReplayScope.EXPORT_GATE
    max_tokens: int = 2_000
    concurrency: int = 4
    paper_ids: tuple[str, ...] = ()
    max_candidates_per_paper: int | None = None

    def __post_init__(self) -> None:
        if not self.verifier_model.strip():
            raise ValueError("verifier model is required")
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.max_candidates_per_paper is not None and self.max_candidates_per_paper < 1:
            raise ValueError("max_candidates_per_paper must be positive when provided")
        resolved_run = self.run_root.resolve(strict=False)
        resolved_output = self.output_root.resolve(strict=False)
        if resolved_output == resolved_run or resolved_output.is_relative_to(resolved_run):
            raise ValueError("replay output_root must be outside the read-only run_root")


def _validated_source_artifact(
    *,
    paper_dir: Path,
    relative_path: Path,
    required: bool,
) -> Path | None:
    """Resolve one source file without following a paper-local symlink boundary."""

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("source artifact path must be paper-relative")
    if paper_dir.is_symlink():
        raise ReplayCheckpointError(
            f"source paper directory {paper_dir.name!r} must not be a symlink"
        )
    resolved_paper = paper_dir.resolve(strict=True)
    if not resolved_paper.is_dir():
        raise ReplayCheckpointError(f"source paper {resolved_paper} is not a directory")
    path = resolved_paper
    for component in relative_path.parts:
        path /= component
        if path.is_symlink():
            raise ReplayCheckpointError(
                f"source paper {resolved_paper.name!r} contains symlinked artifact "
                f"{relative_path.as_posix()!r}"
            )
    if not path.exists():
        if required:
            raise ReplayCheckpointError(
                f"source paper {resolved_paper.name!r} is missing {relative_path.as_posix()!r}"
            )
        return None
    if not path.is_file():
        raise ReplayCheckpointError(
            f"source artifact {relative_path.as_posix()!r} is not a regular file"
        )
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_relative_to(resolved_paper):
        raise ReplayCheckpointError(
            f"source artifact {relative_path.as_posix()!r} escapes its exact paper directory"
        )
    return resolved


def _validated_source_paper(*, run_root: Path, paper_dir: Path) -> Path:
    """Require one real paper directory and all replay inputs inside the exact run."""

    if paper_dir.is_symlink():
        raise ReplayCheckpointError(
            f"source paper directory {paper_dir.name!r} must not be a symlink"
        )
    resolved_run_root = run_root.resolve(strict=True)
    resolved_paper = paper_dir.resolve(strict=True)
    if not resolved_run_root.is_dir() or not resolved_paper.is_dir():
        raise ReplayCheckpointError("source run and paper must be directories")
    if resolved_paper.parent != resolved_run_root:
        raise ReplayCheckpointError(
            f"source paper {resolved_paper.name!r} is not a direct child of the exact run root"
        )
    _validated_source_artifact(
        paper_dir=resolved_paper,
        relative_path=Path(_RUN),
        required=True,
    )
    _validated_source_artifact(
        paper_dir=resolved_paper,
        relative_path=Path(_OBSERVATIONS),
        required=False,
    )
    _validated_source_artifact(
        paper_dir=resolved_paper,
        relative_path=_RESULT_BLOCKS,
        required=False,
    )
    return resolved_paper


def in_replay_scope(candidate: CandidateObservation, scope: ReplayScope) -> bool:
    """Decide membership without inspecting any reference annotation."""

    if scope is ReplayScope.ALL:
        return True
    if candidate.claim_type != ClaimType.PRIMARY_RESULT:
        return False
    if scope is ReplayScope.PRIMARY:
        return True
    return candidate.export_status in _EXPORT_GATE_STATUSES


def read_candidates(paper_dir: Path) -> list[CandidateObservation]:
    path = _validated_source_artifact(
        paper_dir=paper_dir,
        relative_path=Path(_OBSERVATIONS),
        required=False,
    )
    if path is None:
        return []
    return [
        CandidateObservation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_result_blocks(paper_dir: Path) -> list[ResultBlock]:
    path = _validated_source_artifact(
        paper_dir=paper_dir,
        relative_path=_RESULT_BLOCKS,
        required=False,
    )
    if path is None:
        return []
    return [ResultBlock.model_validate(item) for item in read_json(path)]


def discover_paper_dirs(run_root: Path, paper_ids: tuple[str, ...] = ()) -> list[Path]:
    """Return paper directories of a corpus run in deterministic order."""

    resolved_run_root = run_root.resolve(strict=True)
    wanted = set(paper_ids)
    candidates: list[Path] = []
    for path in sorted(resolved_run_root.iterdir()):
        if path.is_symlink():
            if path.is_dir() or path.name in wanted:
                raise ReplayCheckpointError(
                    f"source run contains symlinked paper directory {path.name!r}"
                )
            continue
        if not path.is_dir():
            continue
        run_manifest = path / _RUN
        if run_manifest.is_symlink():
            raise ReplayCheckpointError(f"source paper {path.name!r} contains symlinked {_RUN}")
        if run_manifest.is_file():
            candidates.append(path)
    if not paper_ids:
        return candidates
    selected = [path for path in candidates if path.name in wanted]
    missing = wanted - {path.name for path in selected}
    if missing:
        raise ValueError(f"run root has no paper directories for {sorted(missing)}")
    return selected


@dataclass(frozen=True)
class BindingRecord:
    """Why one in-scope candidate did or did not reach the verifier."""

    observation_id: str
    bound: bool
    block_id: str | None
    page: int | None
    export_status: str
    claim_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "bound": self.bound,
            "block_id": self.block_id,
            "page": self.page,
            "export_status": self.export_status,
            "claim_type": self.claim_type,
        }


@dataclass(frozen=True)
class ReplayWorkItem:
    """One candidate plus the exact evidence block selected before any provider call."""

    observation_id: str
    candidate: CandidateObservation
    result_block: ResultBlock
    evidence_block: FrozenEvidenceBlock
    binding: dict[str, Any]


@dataclass(frozen=True)
class PreparedReplayPaper:
    """All deterministic replay inputs and bindings prepared without provider access."""

    candidates: list[CandidateObservation]
    in_scope: list[CandidateObservation]
    bindings: list[BindingRecord]
    work: list[ReplayWorkItem]
    request_contract: dict[str, Any]
    code_binding: dict[str, Any]
    checkpoint_contract: dict[str, Any]


def _json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _replay_code_binding() -> dict[str, Any]:
    """Hash the installed package tree that determines replay semantics.

    Absolute paths are deliberately excluded. The combined hash covers the verifier,
    binding, transport, schemas, and every transitive local helper in the installed
    package. It is conservative: any package-code change requires a fresh output root.
    """

    package_root = Path(__file__).resolve().parents[1]
    paths = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix in {".json", ".py", ".toml", ".yaml", ".yml"}
    )
    digest = hashlib.sha256()
    for path in paths:
        label = path.relative_to(package_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "schema_version": REPLAY_CODE_BINDING_VERSION,
        "package": "proceedings_to_eee",
        "files_hashed": len(paths),
        "source_tree_sha256": digest.hexdigest(),
    }


def _wire_prompt_sha256(
    *, candidate: CandidateObservation, evidence_block: FrozenEvidenceBlock
) -> str:
    request = VerificationRequest(candidate=candidate, evidence_block=evidence_block)
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": verification_prompt(request)},
    ]
    encoded = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _public_provider_call(call: ProviderCall) -> dict[str, Any]:
    return public_provider_call(call)


def _checkpoint_provider_call(call: ProviderCall) -> dict[str, Any]:
    """Retain rehydratable secret-free call state only inside the private checkpoint."""

    return call.model_dump(mode="json", exclude={"request_id"}, exclude_none=False)


def _provider_assessment_sha256(verification: CandidateVerification) -> str:
    """Reproduce the provider response fingerprint used by the production verifier."""

    return verifier_assessment_sha256(verification.provider_assessment)


def _entry_binding(
    *,
    settings: ReplaySettings,
    observation_id: str,
    candidate: CandidateObservation,
    result_block: ResultBlock,
    evidence_block: FrozenEvidenceBlock,
    request_contract: dict[str, Any],
    code_binding_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": REPLAY_ENTRY_BINDING_VERSION,
        "observation_id": observation_id,
        "candidate_sha256": _json_sha256(candidate.model_dump(mode="json", exclude_none=False)),
        "evidence": {
            "block_id": evidence_block.block_id,
            "source_id": evidence_block.source_id,
            "page": evidence_block.page,
            "result_block_sha256": _json_sha256(
                result_block.model_dump(mode="json", exclude_none=False)
            ),
            "frozen_evidence_schema_version": evidence_block.schema_version,
            "frozen_evidence_block_sha256": verifier_evidence_block_sha256(evidence_block),
            "source_text_sha256": evidence_block.text_sha256,
            "claimed_anchor_sha256": _json_sha256(
                evidence_block.claimed_anchor_untrusted.model_dump(mode="json", exclude_none=False)
            ),
        },
        "verifier": {
            "model": settings.verifier_model,
            "max_tokens": settings.max_tokens,
            "completion_token_parameter": completion_token_parameter_for_model(
                settings.verifier_model
            ),
            "request_settings": VERIFIER_REQUEST_SETTINGS.as_dict(),
            "request_contract": request_contract,
            "request_contract_sha256": _json_sha256(request_contract),
            "schema_name": VERIFIER_SCHEMA_NAME,
            "schema_sha256": request_contract["schema"]["schema_sha256"],
            "system_prompt_sha256": hashlib.sha256(
                VERIFIER_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "wire_prompt_sha256": _wire_prompt_sha256(
                candidate=candidate, evidence_block=evidence_block
            ),
        },
        "code_binding_sha256": code_binding_sha256,
    }


def _checkpoint_contract(
    *,
    settings: ReplaySettings,
    paper_dir: Path,
    request_contract: dict[str, Any],
    code_binding: dict[str, Any],
    work: list[ReplayWorkItem],
    bindings: list[BindingRecord],
) -> dict[str, Any]:
    workset = {
        item.observation_id: _json_sha256(item.binding)
        for item in sorted(work, key=lambda item: item.observation_id)
    }
    return {
        "schema_version": REPLAY_CHECKPOINT_CONTRACT_VERSION,
        "paper_id": paper_dir.name,
        "source_run": {
            "run_sha256": sha256_file(paper_dir / _RUN),
            "observations_sha256": (
                sha256_file(paper_dir / _OBSERVATIONS)
                if (paper_dir / _OBSERVATIONS).is_file()
                else None
            ),
            "result_blocks_sha256": (
                sha256_file(paper_dir / _RESULT_BLOCKS)
                if (paper_dir / _RESULT_BLOCKS).is_file()
                else None
            ),
        },
        "selection": {
            "scope": settings.scope.value,
            "max_candidates_per_paper": settings.max_candidates_per_paper,
            "binding_ledger_sha256": _json_sha256([record.as_dict() for record in bindings]),
        },
        "verifier": {
            "model": settings.verifier_model,
            "max_tokens": settings.max_tokens,
            "completion_token_parameter": completion_token_parameter_for_model(
                settings.verifier_model
            ),
            "request_settings": VERIFIER_REQUEST_SETTINGS.as_dict(),
            "request_contract": request_contract,
            "request_contract_sha256": _json_sha256(request_contract),
            "system_prompt_sha256": hashlib.sha256(
                VERIFIER_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
        },
        "code": code_binding,
        "code_binding_sha256": _json_sha256(code_binding),
        "bound_workset_sha256": _json_sha256(workset),
    }


def _prepare_replay_paper(*, settings: ReplaySettings, paper_dir: Path) -> PreparedReplayPaper:
    """Build the exact candidate/evidence/workset contract without reading labels."""

    paper_dir = _validated_source_paper(run_root=settings.run_root, paper_dir=paper_dir)
    paper_id = paper_dir.name
    candidates = read_candidates(paper_dir)
    candidate_ids = [candidate.observation_id or candidate.stable_id() for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ReplayCheckpointError(
            f"source run contains duplicate observation IDs for paper {paper_id!r}"
        )
    blocks = read_result_blocks(paper_dir)
    in_scope = [item for item in candidates if in_replay_scope(item, settings.scope)]
    if settings.max_candidates_per_paper is not None:
        in_scope = in_scope[: settings.max_candidates_per_paper]

    bindings: list[BindingRecord] = []
    prepared: list[tuple[str, CandidateObservation, ResultBlock, FrozenEvidenceBlock]] = []
    seen_observation_ids: set[str] = set()
    for candidate in in_scope:
        support = bind_candidate_block(candidate, blocks)
        observation_id = candidate.observation_id or candidate.stable_id()
        if observation_id in seen_observation_ids:
            raise ReplayCheckpointError(
                f"source run contains duplicate observation_id {observation_id!r}"
            )
        seen_observation_ids.add(observation_id)
        if support is None:
            bindings.append(
                BindingRecord(
                    observation_id=observation_id,
                    bound=False,
                    block_id=None,
                    page=candidate.evidence[0].page,
                    export_status=str(candidate.export_status),
                    claim_type=str(candidate.claim_type),
                )
            )
            continue
        block, anchor = support
        evidence_block = frozen_evidence_block(paper_id=paper_id, block=block, anchor=anchor)
        bindings.append(
            BindingRecord(
                observation_id=observation_id,
                bound=True,
                block_id=block.block_id,
                page=block.page,
                export_status=str(candidate.export_status),
                claim_type=str(candidate.claim_type),
            )
        )
        prepared.append((observation_id, candidate, block, evidence_block))

    request_contract = verifier_request_contract(
        model=settings.verifier_model,
        max_tokens=settings.max_tokens,
    )
    code_binding = _replay_code_binding()
    code_binding_sha256 = _json_sha256(code_binding)
    work = [
        ReplayWorkItem(
            observation_id=observation_id,
            candidate=candidate,
            result_block=block,
            evidence_block=evidence_block,
            binding=_entry_binding(
                settings=settings,
                observation_id=observation_id,
                candidate=candidate,
                result_block=block,
                evidence_block=evidence_block,
                request_contract=request_contract,
                code_binding_sha256=code_binding_sha256,
            ),
        )
        for observation_id, candidate, block, evidence_block in prepared
    ]
    contract = _checkpoint_contract(
        settings=settings,
        paper_dir=paper_dir,
        request_contract=request_contract,
        code_binding=code_binding,
        work=work,
        bindings=bindings,
    )
    return PreparedReplayPaper(
        candidates=candidates,
        in_scope=in_scope,
        bindings=bindings,
        work=work,
        request_contract=request_contract,
        code_binding=code_binding,
        checkpoint_contract=contract,
    )


def _new_checkpoint(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REPLAY_CHECKPOINT_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": _json_sha256(contract),
        "entries": {},
    }


def _load_checkpoint(*, output_dir: Path, contract: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = output_dir / _CHECKPOINT
    if not checkpoint_path.is_file():
        stale = [name for name in _OUTPUT_ARTIFACTS[1:] if (output_dir / name).exists()]
        if stale:
            raise ReplayCheckpointError(
                "replay output contains unsupported historical or torn artifacts without "
                f"{_CHECKPOINT}: {stale}; use an empty output root"
            )
        checkpoint = _new_checkpoint(contract)
        write_json(checkpoint_path, checkpoint)
        return checkpoint
    try:
        checkpoint = read_json(checkpoint_path)
    except (OSError, ValueError) as error:
        raise ReplayCheckpointError(
            "replay checkpoint is unreadable; use an empty output root"
        ) from error
    if not isinstance(checkpoint, dict):
        raise ReplayCheckpointError("replay checkpoint must be a JSON object")
    schema_version = checkpoint.get("schema_version")
    if schema_version != REPLAY_CHECKPOINT_SCHEMA_VERSION:
        raise ReplayCheckpointError(
            "unsupported replay checkpoint schema "
            f"{schema_version!r}; expected {REPLAY_CHECKPOINT_SCHEMA_VERSION!r}; "
            "use an empty output root"
        )
    if checkpoint.get("contract") != contract or checkpoint.get("contract_sha256") != _json_sha256(
        contract
    ):
        raise ReplayCheckpointError(
            "replay checkpoint contract mismatch (source, selection, model, token limit, "
            "request settings/schema/prompt, or code changed); use an empty output root"
        )
    if not isinstance(checkpoint.get("entries"), dict):
        raise ReplayCheckpointError("replay checkpoint entries must be an object")
    return checkpoint


def _validate_provider_call(
    payload: Any,
    *,
    binding: dict[str, Any],
    require_returned_model_match: bool = True,
) -> ProviderCall:
    if not isinstance(payload, dict) or "request_id" in payload:
        raise ReplayCheckpointError("replay provider call is malformed or contains a private ID")
    try:
        call = ProviderCall.model_validate(payload)
    except (TypeError, ValueError) as error:
        raise ReplayCheckpointError(
            "replay provider call does not match the current schema"
        ) from error
    verifier = binding["verifier"]
    request_settings = verifier["request_settings"]
    if any(
        (
            call.provider != "openrouter",
            call.model_requested != verifier["model"],
            require_returned_model_match and call.model_returned != verifier["model"],
            call.prompt_sha256 != verifier["wire_prompt_sha256"],
            call.temperature != request_settings["temperature"],
            call.reasoning_effort != request_settings["reasoning_effort"],
            call.max_tokens != verifier["max_tokens"],
            call.completion_token_parameter != verifier["completion_token_parameter"],
            call.seed != request_settings["seed"],
            call.schema_name != verifier["schema_name"],
            call.schema_sha256 != verifier["schema_sha256"],
            call.require_parameters != request_settings["require_parameters"],
            call.data_collection != "deny",
            not call.zdr,
            structured_request_contract_from_call(call) != verifier["request_contract"],
        )
    ):
        raise ReplayCheckpointError("replay provider call does not match its immutable binding")
    return call


def _entry_payload(
    *,
    status: str,
    binding: dict[str, Any],
    verification: CandidateVerification | None,
    call: ProviderCall | None,
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    entry = {
        "schema_version": REPLAY_ENTRY_SCHEMA_VERSION,
        "status": status,
        "binding": binding,
        "binding_sha256": _json_sha256(binding),
        "verification": (
            verification.model_dump(mode="json", exclude_none=False)
            if verification is not None
            else None
        ),
        "provider_call": _checkpoint_provider_call(call) if call is not None else None,
        "request_id_observed": call is not None and call.request_id is not None,
        "error": error,
    }
    entry["entry_sha256"] = _json_sha256(entry)
    return entry


def _validated_entry(
    raw: Any,
    *,
    item: ReplayWorkItem,
) -> tuple[
    CandidateVerification | None,
    ProviderCall | None,
    dict[str, Any] | None,
    bool,
]:
    if not isinstance(raw, dict) or raw.get("schema_version") != REPLAY_ENTRY_SCHEMA_VERSION:
        raise ReplayCheckpointError(
            "unsupported verifier replay entry schema; historical rows cannot be resumed"
        )
    recorded_sha256 = raw.get("entry_sha256")
    unsigned = {key: value for key, value in raw.items() if key != "entry_sha256"}
    if not isinstance(recorded_sha256, str) or recorded_sha256 != _json_sha256(unsigned):
        raise ReplayCheckpointError("verifier replay entry integrity hash mismatch")
    if raw.get("binding") != item.binding or raw.get("binding_sha256") != _json_sha256(
        item.binding
    ):
        raise ReplayCheckpointError(
            f"verifier replay entry binding mismatch for {item.observation_id}"
        )

    status = raw.get("status")
    verification_payload = raw.get("verification")
    call_payload = raw.get("provider_call")
    request_id_observed = raw.get("request_id_observed")
    error_payload = raw.get("error")
    if not isinstance(request_id_observed, bool):
        raise ReplayCheckpointError("replay entry request-ID disposition is malformed")
    if status == "success":
        if call_payload is None or error_payload is not None:
            raise ReplayCheckpointError("successful replay entry lacks an atomic outcome/call pair")
        try:
            verification = CandidateVerification.model_validate(verification_payload)
        except (TypeError, ValueError) as error:
            raise ReplayCheckpointError(
                "unsupported historical verifier result; candidate-verification/0.2 is required"
            ) from error
        expected = contextualize_verification(
            candidate=item.candidate,
            evidence_block=item.evidence_block,
            provider_assessment=verification.provider_assessment,
        )
        if verification != expected:
            raise ReplayCheckpointError(
                f"verifier replay result is not bound to {item.observation_id}"
            )
        call = _validate_provider_call(call_payload, binding=item.binding)
        if call.response_sha256 != _provider_assessment_sha256(verification):
            raise ReplayCheckpointError(
                "verifier replay provider call response hash does not match its assessment"
            )
        return verification, call, None, request_id_observed

    if status not in {
        "contract_failure",
        "local_failure",
        "provider_failure",
        "response_failure",
    }:
        raise ReplayCheckpointError(f"unsupported verifier replay entry status {status!r}")
    if verification_payload is not None or not isinstance(error_payload, dict):
        raise ReplayCheckpointError("failed replay entry has an invalid terminal outcome")
    if (
        error_payload.get("observation_id") != item.observation_id
        or error_payload.get("block_id") != item.evidence_block.block_id
        or not isinstance(error_payload.get("error"), str)
        or not isinstance(error_payload.get("code"), str)
    ):
        raise ReplayCheckpointError("failed replay entry error is not bound to its observation")
    if status == "response_failure":
        if call_payload is None:
            raise ReplayCheckpointError(
                "response-validation failure lacks its atomic provider call telemetry"
            )
        if (
            error_payload.get("error") != "provider_response_validation"
            or error_payload.get("code")
            not in {"invalid_json", "schema_validation", "wire_validation"}
            or not isinstance(error_payload.get("validation_path"), list)
            or any(
                isinstance(component, bool) or not isinstance(component, int | str)
                for component in error_payload["validation_path"]
            )
            or (
                error_payload.get("validation_keyword") is not None
                and not isinstance(error_payload.get("validation_keyword"), str)
            )
        ):
            raise ReplayCheckpointError(
                "response-validation failure diagnostics do not match the typed error contract"
            )
        call = _validate_provider_call(call_payload, binding=item.binding)
    elif status == "contract_failure":
        contract_failure_code = error_payload.get("code")
        if (
            call_payload is None
            or error_payload.get("error") != "provider_response_binding"
            or contract_failure_code not in {"response_hash_mismatch", "returned_model_mismatch"}
        ):
            raise ReplayCheckpointError(
                "provider-response contract failure lacks its atomic call binding"
            )
        returned_model_mismatch = contract_failure_code == "returned_model_mismatch"
        call = _validate_provider_call(
            call_payload,
            binding=item.binding,
            require_returned_model_match=not returned_model_mismatch,
        )
        if returned_model_mismatch and call.model_returned == item.binding["verifier"]["model"]:
            raise ReplayCheckpointError(
                "returned-model mismatch failure contains matching call telemetry"
            )
    else:
        if call_payload is not None:
            raise ReplayCheckpointError(
                "non-response failure unexpectedly contains a provider call"
            )
        call = None
    if (call is None and request_id_observed) or (call is not None and call.request_id is not None):
        raise ReplayCheckpointError("replay entry retained an invalid request-ID state")
    return None, call, error_payload, request_id_observed


def _persist_entry(
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    observation_id: str,
    entry: dict[str, Any],
    lock: threading.Lock,
) -> None:
    """Atomically mark one observation terminal with its outcome and call together."""

    with lock:
        if observation_id in checkpoint["entries"]:
            raise ReplayCheckpointError(f"duplicate replay entry for {observation_id}")
        updated_entries = {**checkpoint["entries"], observation_id: entry}
        updated = {**checkpoint, "entries": updated_entries}
        write_json(checkpoint_path, updated)
        checkpoint.clear()
        checkpoint.update(updated)


def _checkpoint_rows(
    *, checkpoint: dict[str, Any], work_by_id: dict[str, ReplayWorkItem]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate atomic entries and return their deterministic public rows."""

    verification_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    for observation_id, raw in sorted(checkpoint["entries"].items()):
        item = work_by_id.get(observation_id)
        if item is None:
            raise ReplayCheckpointError(
                f"replay checkpoint contains an out-of-workset entry: {observation_id}"
            )
        verification, call, error, request_id_observed = _validated_entry(raw, item=item)
        if verification is not None:
            verification_rows.append(verification.model_dump(mode="json", exclude_none=False))
        if error is not None:
            error_rows.append(error)
        if call is not None:
            projected_call = _public_provider_call(call)
            projected_call["request_id_observed"] = request_id_observed
            call_rows.append({"observation_id": observation_id, **projected_call})
    return verification_rows, error_rows, call_rows


def _project_checkpoint(
    *,
    output_dir: Path,
    checkpoint: dict[str, Any],
    work_by_id: dict[str, ReplayWorkItem],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize deterministic JSONL views from the atomic checkpoint authority."""

    verification_rows, error_rows, call_rows = _checkpoint_rows(
        checkpoint=checkpoint, work_by_id=work_by_id
    )
    write_jsonl(output_dir / "verifications.jsonl", verification_rows)
    write_jsonl(output_dir / "verifier-errors.jsonl", error_rows)
    write_jsonl(output_dir / "verifier-calls.jsonl", call_rows)
    return verification_rows, error_rows, call_rows


def _reject_source_aliases(*, output_dir: Path, paper_dir: Path) -> None:
    """Reject symlink or hard-link aliases before any replay artifact is written."""

    if not output_dir.exists():
        return
    source_files = [path for path in paper_dir.rglob("*") if path.is_file()]
    for name in _OUTPUT_ARTIFACTS:
        output = output_dir / name
        if not output.exists() and not output.is_symlink():
            continue
        resolved = output.resolve(strict=False)
        if resolved == paper_dir or resolved.is_relative_to(paper_dir):
            raise ReplayCheckpointError(
                f"replay output artifact {name!r} aliases the read-only source paper"
            )
        if output.exists() and any(output.samefile(source) for source in source_files):
            raise ReplayCheckpointError(
                f"replay output artifact {name!r} hard-links a read-only source artifact"
            )


def _validated_output_paper_dir(*, output_root: Path, paper_id: str) -> Path:
    """Return an exact direct child without following a pre-existing paper symlink."""

    expected_root = output_root.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_root = output_root.resolve(strict=True)
    if resolved_root != expected_root or not resolved_root.is_dir():
        raise ReplayCheckpointError("replay output root changed while preparing the replay")
    output_dir = resolved_root / paper_id
    if output_dir.is_symlink():
        raise ReplayCheckpointError(
            "replay paper output must be a direct non-symlink child of the exact output root"
        )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ReplayCheckpointError("replay paper output path is not a directory")
        resolved_output = output_dir.resolve(strict=True)
        if resolved_output != output_dir or resolved_output.parent != resolved_root:
            raise ReplayCheckpointError(
                "replay paper output must be a direct non-symlink child of the exact output root"
            )
    return output_dir


def _paper_summary(
    *,
    paper_id: str,
    prepared: PreparedReplayPaper,
    verification_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    call_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    decisions = {value.value: 0 for value in IndependentDecision}
    provider_decisions = {value.value: 0 for value in IndependentDecision}
    for row in verification_rows:
        decisions[row["effective_decision"]] += 1
        provider_decisions[row["provider_assessment"]["decision"]] += 1
    return {
        "paper_id": paper_id,
        "candidates_recorded": len(prepared.candidates),
        "candidates_in_scope": len(prepared.in_scope),
        "bound": sum(1 for record in prepared.bindings if record.bound),
        "unbound": sum(1 for record in prepared.bindings if not record.bound),
        "verifications": len(verification_rows),
        "errors": len(error_rows),
        "decisions": decisions,
        "provider_decisions": provider_decisions,
        "cost": _call_totals(call_rows),
    }


def replay_paper(
    *,
    client: OpenRouterClient,
    settings: ReplaySettings,
    paper_dir: Path,
) -> dict[str, Any]:
    """Replay one paper under a strict, atomic, versioned resume checkpoint."""

    paper_dir = _validated_source_paper(run_root=settings.run_root, paper_dir=paper_dir)
    prepared = _prepare_replay_paper(settings=settings, paper_dir=paper_dir)
    paper_id = paper_dir.name
    resolved_run_root = settings.run_root.resolve(strict=True)
    output_dir = _validated_output_paper_dir(
        output_root=settings.output_root,
        paper_id=paper_id,
    )
    if output_dir == resolved_run_root or output_dir.is_relative_to(resolved_run_root):
        raise ReplayCheckpointError(
            "replay paper output aliases the read-only source run; choose a separate output root"
        )
    if output_dir == paper_dir or output_dir.is_relative_to(paper_dir):
        raise ReplayCheckpointError(
            "replay paper output aliases the read-only source paper; choose a separate output root"
        )
    _reject_source_aliases(output_dir=output_dir, paper_dir=paper_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bindings = prepared.bindings
    work = prepared.work
    work_by_id = {item.observation_id: item for item in work}
    contract = prepared.checkpoint_contract
    checkpoint_path = output_dir / _CHECKPOINT
    checkpoint = _load_checkpoint(output_dir=output_dir, contract=contract)
    extra_entries = set(checkpoint["entries"]) - set(work_by_id)
    if extra_entries:
        raise ReplayCheckpointError(
            f"replay checkpoint contains entries outside the bound workset: {sorted(extra_entries)}"
        )
    # Repair any missing/torn JSONL projection from the authoritative atomic entries
    # before making another paid call.
    _project_checkpoint(output_dir=output_dir, checkpoint=checkpoint, work_by_id=work_by_id)
    pending = [item for item in work if item.observation_id not in checkpoint["entries"]]

    lock = threading.Lock()

    def run_one(item: ReplayWorkItem) -> None:
        observation_id = item.observation_id
        candidate = item.candidate
        block = item.result_block
        evidence_block = item.evidence_block
        try:
            verification, call = verify_candidate(
                client=client,
                model=settings.verifier_model,
                candidate=candidate,
                evidence_block=evidence_block,
                max_tokens=settings.max_tokens,
            )
        except ProviderResponseValidationError as error:
            contract_failure_code = (
                error.validation_keyword
                if error.validation_keyword in {"response_hash_mismatch", "returned_model_mismatch"}
                else None
            )
            entry = _entry_payload(
                status="contract_failure" if contract_failure_code else "response_failure",
                binding=item.binding,
                verification=None,
                call=error.call,
                error=(
                    {
                        "observation_id": observation_id,
                        "block_id": block.block_id,
                        "error": "provider_response_binding",
                        "code": contract_failure_code,
                    }
                    if contract_failure_code
                    else {
                        "observation_id": observation_id,
                        "block_id": block.block_id,
                        "error": "provider_response_validation",
                        "code": error.code,
                        "validation_path": list(error.validation_path),
                        "validation_keyword": (
                            str(error.validation_keyword)
                            if error.validation_keyword is not None
                            else None
                        ),
                    }
                ),
            )
        except ProviderRequestRejectedError as error:
            entry = _entry_payload(
                status="provider_failure",
                binding=item.binding,
                verification=None,
                call=None,
                error={
                    "observation_id": observation_id,
                    "block_id": block.block_id,
                    "error": "provider_request_rejected",
                    "code": str(error.status_code),
                },
            )
        except (RuntimeError, ValueError) as error:
            entry = _entry_payload(
                status="local_failure",
                binding=item.binding,
                verification=None,
                call=None,
                error={
                    "observation_id": observation_id,
                    "block_id": block.block_id,
                    "error": type(error).__name__,
                    "code": "call_failed",
                },
            )
        else:
            if call.response_sha256 != _provider_assessment_sha256(verification):
                entry = _entry_payload(
                    status="contract_failure",
                    binding=item.binding,
                    verification=None,
                    call=call,
                    error={
                        "observation_id": observation_id,
                        "block_id": block.block_id,
                        "error": "provider_response_binding",
                        "code": "response_hash_mismatch",
                    },
                )
            else:
                entry = _entry_payload(
                    status="success",
                    binding=item.binding,
                    verification=verification,
                    call=call,
                    error=None,
                )
        # Validate the complete pair before its one atomic terminal write.
        _validated_entry(entry, item=item)
        _persist_entry(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            observation_id=observation_id,
            entry=entry,
            lock=lock,
        )

    if pending:
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            list(pool.map(run_one, pending))

    final_prepared = _prepare_replay_paper(settings=settings, paper_dir=paper_dir)
    if final_prepared.checkpoint_contract != prepared.checkpoint_contract:
        raise ReplayCheckpointError(
            "source run, verifier contract, or code changed during replay; output remains "
            "bound to the original checkpoint and cannot be summarized"
        )
    verification_rows, error_rows, call_rows = _project_checkpoint(
        output_dir=output_dir, checkpoint=checkpoint, work_by_id=work_by_id
    )
    write_json(output_dir / "bindings.json", [record.as_dict() for record in bindings])

    return _paper_summary(
        paper_id=paper_id,
        prepared=prepared,
        verification_rows=verification_rows,
        error_rows=error_rows,
        call_rows=call_rows,
    )


def _call_totals(call_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate secret-free call telemetry. Missing fields make totals lower bounds."""

    reported_cost = [row for row in call_rows if row.get("cost_usd") is not None]
    reported_tokens = [row for row in call_rows if row.get("total_tokens") is not None]
    reported_reasoning_tokens = [
        row for row in call_rows if row.get("reasoning_tokens") is not None
    ]
    return {
        "calls": len(call_rows),
        "cost_reported_calls": len(reported_cost),
        "cost_usd_lower_bound": round(sum(row["cost_usd"] for row in reported_cost), 7),
        "token_reported_calls": len(reported_tokens),
        "input_tokens_lower_bound": sum(row.get("input_tokens") or 0 for row in call_rows),
        "output_tokens_lower_bound": sum(row.get("output_tokens") or 0 for row in call_rows),
        "reasoning_tokens_reported_calls": len(reported_reasoning_tokens),
        "reasoning_tokens_lower_bound": sum(
            row.get("reasoning_tokens") or 0 for row in reported_reasoning_tokens
        ),
        "total_tokens_lower_bound": sum(row.get("total_tokens") or 0 for row in call_rows),
        "latency_seconds_total": round(
            sum(row.get("latency_seconds") or 0.0 for row in call_rows), 6
        ),
        "attempts_lower_bound": sum(row.get("attempts") or 0 for row in call_rows),
        "basis": (
            "Sums cover calls that returned the field. Absent provider metadata is not "
            "reconstructed, so monetary and token totals are lower bounds."
        ),
    }


def replay_run(*, client: OpenRouterClient, settings: ReplaySettings) -> dict[str, Any]:
    """Replay every selected paper and write a single reproducible summary."""

    paper_dirs = discover_paper_dirs(settings.run_root, settings.paper_ids)
    settings.output_root.mkdir(parents=True, exist_ok=True)
    papers = [
        replay_paper(client=client, settings=settings, paper_dir=paper_dir)
        for paper_dir in paper_dirs
    ]
    decisions = {value.value: 0 for value in IndependentDecision}
    provider_decisions = {value.value: 0 for value in IndependentDecision}
    for paper in papers:
        for key, count in paper["decisions"].items():
            decisions[key] += count
        for key, count in paper["provider_decisions"].items():
            provider_decisions[key] += count
    all_calls: list[dict[str, Any]] = []
    for paper in papers:
        calls_path = settings.output_root / paper["paper_id"] / "verifier-calls.jsonl"
        if calls_path.is_file():
            all_calls.extend(
                json.loads(line)
                for line in calls_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    summary = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "checkpoint_schema_version": REPLAY_CHECKPOINT_SCHEMA_VERSION,
        "run_root": settings.run_root.name,
        "scope": settings.scope.value,
        "paper_ids": list(settings.paper_ids),
        "max_candidates_per_paper": settings.max_candidates_per_paper,
        "verifier": {
            "enabled": True,
            "model": settings.verifier_model,
            "max_tokens": settings.max_tokens,
            "completion_token_parameter": completion_token_parameter_for_model(
                settings.verifier_model
            ),
            **VERIFIER_REQUEST_SETTINGS.as_dict(),
            "request_contract": verifier_request_contract(
                model=settings.verifier_model,
                max_tokens=settings.max_tokens,
            ),
        },
        "totals": {
            "papers": len(papers),
            "candidates_recorded": sum(item["candidates_recorded"] for item in papers),
            "candidates_in_scope": sum(item["candidates_in_scope"] for item in papers),
            "bound": sum(item["bound"] for item in papers),
            "unbound": sum(item["unbound"] for item in papers),
            "verifications": sum(item["verifications"] for item in papers),
            "errors": sum(item["errors"] for item in papers),
            "decisions": decisions,
            "provider_decisions": provider_decisions,
        },
        "cost": _call_totals(all_calls),
        "papers": papers,
    }
    write_json(settings.output_root / "verifier-replay.json", summary)
    return summary


# --------------------------------------------------------------------------------------
# Measurement: what the verifier actually catches, joined to the frozen reference score.
# --------------------------------------------------------------------------------------


REFERENCE_CLASSES = (
    "reference_matched",
    "reference_matched_joint_semantics",
    "unmatched_primary_in_coverage",
    "control_matched",
    "false_primary",
    "false_primary_export",
)
"""Candidate partitions taken verbatim from the run's own frozen reference score.

``reference_matched`` is the recall numerator: every candidate the scorer paired with an
annotated reference observation. ``reference_matched_joint_semantics`` is the stricter
subset whose system, dataset, metric, value, unit, and slice all agreed.
"""


def _reference_classes(paper_dir: Path) -> dict[str, set[str]]:
    """Partition recorded observation IDs using only the already-frozen scoring output."""

    path = paper_dir / _REFERENCE_SCORE
    empty: dict[str, set[str]] = {name: set() for name in REFERENCE_CLASSES}
    if not path.is_file():
        return empty
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(paper_dir):
        raise ReplayCheckpointError(
            f"reference score for {paper_dir.name} must be a real file inside its source paper"
        )
    run_manifest = read_json(paper_dir / _RUN)
    reference_binding = (
        run_manifest.get("reference_evaluation") if isinstance(run_manifest, dict) else None
    )
    if (
        not isinstance(reference_binding, dict)
        or reference_binding.get("score_path") != _REFERENCE_SCORE
        or reference_binding.get("score_sha256") != sha256_file(path)
    ):
        raise ReplayCheckpointError(
            f"reference score for {paper_dir.name} is not hash-bound by its source run manifest"
        )
    score = read_json(path)
    if (
        not isinstance(run_manifest, dict)
        or run_manifest.get("paper_id") != paper_dir.name
        or not isinstance(score, dict)
        or score.get("schema_version") != "reference-score/0.7"
        or score.get("paper_id") != paper_dir.name
        or reference_binding.get("schema_version") != score.get("schema_version")
    ):
        raise ReplayCheckpointError(
            f"reference score for {paper_dir.name} has an unsupported schema or paper binding"
        )

    def typed_unique_ids(value: Any, *, field: str) -> set[str]:
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))
        ):
            raise ReplayCheckpointError(
                f"reference score field {field!r} must contain unique observation IDs"
            )
        return set(value)

    raw_matches = score.get("matches")
    if not isinstance(raw_matches, list):
        raise ReplayCheckpointError("reference score matches must be a typed list")
    matched_ids: list[str] = []
    joint_ids: list[str] = []
    for match in raw_matches:
        if (
            not isinstance(match, dict)
            or (
                match.get("observation_id") is not None
                and (
                    not isinstance(match.get("observation_id"), str) or not match["observation_id"]
                )
            )
            or not isinstance(match.get("joint_semantics"), bool)
        ):
            raise ReplayCheckpointError("reference score contains a malformed match record")
        observation_id = match.get("observation_id")
        if observation_id is not None:
            matched_ids.append(observation_id)
            if match["joint_semantics"]:
                joint_ids.append(observation_id)
    if len(matched_ids) != len(set(matched_ids)):
        raise ReplayCheckpointError("reference score reuses a matched observation ID")

    unmatched_primary = typed_unique_ids(
        score.get("unmatched_primary_candidate_ids_in_coverage"),
        field="unmatched_primary_candidate_ids_in_coverage",
    )
    safety = score.get("negative_control_safety")
    if not isinstance(safety, dict):
        raise ReplayCheckpointError("reference score negative-control safety must be an object")
    control_matched = typed_unique_ids(
        safety.get("matched_candidate_ids"), field="negative_control_safety.matched_candidate_ids"
    )
    false_primary = typed_unique_ids(
        safety.get("false_primary_candidate_ids"),
        field="negative_control_safety.false_primary_candidate_ids",
    )
    false_primary_export = typed_unique_ids(
        safety.get("false_primary_export_candidate_ids"),
        field="negative_control_safety.false_primary_export_candidate_ids",
    )
    matched = set(matched_ids)
    joint = set(joint_ids)
    if (
        not joint <= matched
        or matched & unmatched_primary
        or not false_primary <= control_matched
        or not false_primary_export <= false_primary
    ):
        raise ReplayCheckpointError("reference score class partitions violate subset invariants")
    return {
        "reference_matched": matched,
        "reference_matched_joint_semantics": joint,
        "unmatched_primary_in_coverage": unmatched_primary,
        "control_matched": control_matched,
        "false_primary": false_primary,
        "false_primary_export": false_primary_export,
    }


def _class_decisions(ids: set[str], decisions: dict[str, str], unbound: set[str]) -> dict[str, Any]:
    counts = {value.value: 0 for value in IndependentDecision}
    unverified = 0
    for observation_id in ids:
        decision = decisions.get(observation_id)
        if decision is None:
            unverified += 1
            continue
        counts[decision] += 1
    return {
        "total": len(ids),
        "verified": len(ids) - unverified,
        "unverified": unverified,
        "unbound": len(ids & unbound),
        **counts,
    }


def measure_replay(*, run_root: Path, replay_root: Path) -> dict[str, Any]:
    """Join replay verdicts to the frozen reference score without re-scoring anything.

    Every class below is defined by the sealed run's own ``reference-score.json``. The
    replay never sees a reference annotation, so this join happens strictly after the
    verifier has committed to its verdicts.
    """

    run_root = run_root.resolve(strict=True)
    replay_root = replay_root.resolve(strict=True)
    if replay_root == run_root or replay_root.is_relative_to(run_root):
        raise ReplayCheckpointError(
            "replay measurement output must remain outside the read-only source run"
        )
    summary_path = replay_root / "verifier-replay.json"
    if not summary_path.is_file():
        raise ValueError(f"no replay summary at {summary_path}")
    replay = read_json(summary_path)
    if not isinstance(replay, dict) or replay.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ReplayCheckpointError(
            "unsupported verifier replay summary schema; historical replay rows cannot be "
            "measured under the current candidate-verification shape"
        )
    if (
        replay.get("checkpoint_schema_version") != REPLAY_CHECKPOINT_SCHEMA_VERSION
        or replay.get("run_root") != run_root.name
        or not isinstance(replay.get("papers"), list)
        or not isinstance(replay.get("paper_ids"), list)
        or not isinstance(replay.get("verifier"), dict)
    ):
        raise ReplayCheckpointError("verifier replay summary binding is incomplete or mismatched")
    selected_paper_ids = replay["paper_ids"]
    if any(not isinstance(paper_id, str) for paper_id in selected_paper_ids) or len(
        selected_paper_ids
    ) != len(set(selected_paper_ids)):
        raise ReplayCheckpointError("verifier replay paper selection is malformed")
    expected_source_papers = discover_paper_dirs(run_root, tuple(selected_paper_ids))
    expected_paper_ids = [paper_dir.name for paper_dir in expected_source_papers]
    summary_paper_ids = [
        entry.get("paper_id") if isinstance(entry, dict) else None for entry in replay["papers"]
    ]
    checkpoint_paper_ids = sorted(
        path.name
        for path in replay_root.iterdir()
        if path.is_dir() and (path / _CHECKPOINT).is_file()
    )
    if summary_paper_ids != expected_paper_ids or checkpoint_paper_ids != expected_paper_ids:
        raise ReplayCheckpointError(
            "verifier replay paper list is not the exact selected source/checkpoint projection"
        )
    try:
        scope = ReplayScope(replay.get("scope"))
    except ValueError as error:
        raise ReplayCheckpointError("verifier replay summary has an unsupported scope") from error
    verifier = replay["verifier"]
    model = verifier.get("model")
    max_tokens = verifier.get("max_tokens")
    max_candidates = replay.get("max_candidates_per_paper")
    if (
        not isinstance(model, str)
        or not model.strip()
        or isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
        or (
            max_candidates is not None
            and (
                isinstance(max_candidates, bool)
                or not isinstance(max_candidates, int)
                or max_candidates < 1
            )
        )
    ):
        raise ReplayCheckpointError("verifier replay summary has invalid execution settings")
    expected_verifier = {
        "enabled": True,
        "model": model,
        "max_tokens": max_tokens,
        "completion_token_parameter": completion_token_parameter_for_model(model),
        **VERIFIER_REQUEST_SETTINGS.as_dict(),
        "request_contract": verifier_request_contract(
            model=model,
            max_tokens=max_tokens,
        ),
    }
    if verifier != expected_verifier:
        raise ReplayCheckpointError(
            "verifier replay summary request settings/schema no longer match current execution"
        )

    classes = REFERENCE_CLASSES
    aggregate: dict[str, set[str]] = {name: set() for name in classes}
    all_decisions: dict[str, str] = {}
    all_unbound: set[str] = set()
    per_paper: list[dict[str, Any]] = []
    computed_papers: list[dict[str, Any]] = []
    all_call_rows: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    seen_reference_ids: set[str] = set()
    seen_paper_ids: set[str] = set()

    for entry in replay["papers"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("paper_id"), str):
            raise ReplayCheckpointError("verifier replay paper summary is malformed")
        paper_id = entry["paper_id"]
        if paper_id in seen_paper_ids:
            raise ReplayCheckpointError(f"duplicate replay paper_id {paper_id!r}")
        seen_paper_ids.add(paper_id)
        paper_dir = (run_root / paper_id).resolve(strict=True)
        if paper_dir.parent != run_root:
            raise ReplayCheckpointError(f"replay paper {paper_id!r} is outside the source run")
        output_dir = (replay_root / paper_id).resolve(strict=True)
        if output_dir == run_root or output_dir.is_relative_to(run_root):
            raise ReplayCheckpointError(
                f"replay output for {paper_id!r} aliases the read-only source run"
            )
        decisions: dict[str, str] = {}
        checkpoint_path = output_dir / _CHECKPOINT
        if not checkpoint_path.is_file():
            raise ReplayCheckpointError(
                f"replay measurement requires the atomic checkpoint for {paper_id}"
            )
        try:
            checkpoint = read_json(checkpoint_path)
        except (OSError, ValueError) as error:
            raise ReplayCheckpointError(
                f"replay checkpoint for {paper_id} is unreadable"
            ) from error
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema_version") != REPLAY_CHECKPOINT_SCHEMA_VERSION
            or not isinstance(checkpoint.get("contract"), dict)
            or checkpoint.get("contract_sha256") != _json_sha256(checkpoint["contract"])
            or not isinstance(checkpoint.get("entries"), dict)
        ):
            raise ReplayCheckpointError(
                f"replay checkpoint for {paper_id} is unsupported or fails integrity checks"
            )
        measurement_settings = ReplaySettings(
            run_root=run_root,
            output_root=replay_root,
            verifier_model=model,
            scope=scope,
            max_tokens=max_tokens,
            concurrency=1,
            paper_ids=(paper_id,),
            max_candidates_per_paper=max_candidates,
        )
        prepared = _prepare_replay_paper(settings=measurement_settings, paper_dir=paper_dir)
        if checkpoint.get("contract") != prepared.checkpoint_contract or checkpoint.get(
            "contract_sha256"
        ) != _json_sha256(prepared.checkpoint_contract):
            raise ReplayCheckpointError(
                f"replay checkpoint for {paper_id} does not bind the supplied source run"
            )
        work_by_id = {item.observation_id: item for item in prepared.work}
        if set(checkpoint["entries"]) != set(work_by_id):
            raise ReplayCheckpointError(
                f"replay checkpoint for {paper_id} is not terminal for its exact workset"
            )
        verification_rows, error_rows, call_rows = _checkpoint_rows(
            checkpoint=checkpoint, work_by_id=work_by_id
        )
        computed_paper = _paper_summary(
            paper_id=paper_id,
            prepared=prepared,
            verification_rows=verification_rows,
            error_rows=error_rows,
            call_rows=call_rows,
        )
        if entry != computed_paper:
            raise ReplayCheckpointError(
                f"verifier replay paper summary for {paper_id} is not a checkpoint projection"
            )
        computed_papers.append(computed_paper)
        all_call_rows.extend(call_rows)
        decisions = {row["observation_id"]: row["effective_decision"] for row in verification_rows}
        bindings_path = output_dir / "bindings.json"
        if not bindings_path.is_file():
            raise ReplayCheckpointError(f"replay binding ledger is missing for {paper_id}")
        binding_records = read_json(bindings_path)
        expected_bindings = [record.as_dict() for record in prepared.bindings]
        if binding_records != expected_bindings:
            raise ReplayCheckpointError(
                f"replay binding ledger for {paper_id} does not match its checkpoint"
            )
        unbound = {
            record["observation_id"]
            for record in binding_records
            if isinstance(record, dict) and not record.get("bound")
        }
        candidate_ids = {
            candidate.observation_id or candidate.stable_id() for candidate in prepared.candidates
        }
        duplicates = seen_candidate_ids & candidate_ids
        if duplicates:
            raise ReplayCheckpointError(
                f"cross-paper duplicate observation IDs: {sorted(duplicates)}"
            )
        seen_candidate_ids |= candidate_ids
        reference = _reference_classes(paper_dir)
        reference_ids = set().union(*reference.values())
        unknown_reference_ids = reference_ids - candidate_ids
        if unknown_reference_ids:
            raise ReplayCheckpointError(
                f"reference score for {paper_id} names unknown observations: "
                f"{sorted(unknown_reference_ids)}"
            )
        cross_paper_reference_ids = seen_reference_ids & reference_ids
        if cross_paper_reference_ids:
            raise ReplayCheckpointError(
                "reference scores reuse observation IDs across papers: "
                f"{sorted(cross_paper_reference_ids)}"
            )
        seen_reference_ids |= reference_ids
        per_paper.append(
            {
                "paper_id": paper_id,
                "classes": {
                    name: _class_decisions(reference[name], decisions, unbound) for name in classes
                },
            }
        )
        for name in classes:
            aggregate[name] |= reference[name]
        all_decisions.update(decisions)
        all_unbound |= unbound

    expected_decisions = {value.value: 0 for value in IndependentDecision}
    expected_provider_decisions = {value.value: 0 for value in IndependentDecision}
    for paper in computed_papers:
        for key, count in paper["decisions"].items():
            expected_decisions[key] += count
        for key, count in paper["provider_decisions"].items():
            expected_provider_decisions[key] += count
    expected_totals = {
        "papers": len(computed_papers),
        "candidates_recorded": sum(item["candidates_recorded"] for item in computed_papers),
        "candidates_in_scope": sum(item["candidates_in_scope"] for item in computed_papers),
        "bound": sum(item["bound"] for item in computed_papers),
        "unbound": sum(item["unbound"] for item in computed_papers),
        "verifications": sum(item["verifications"] for item in computed_papers),
        "errors": sum(item["errors"] for item in computed_papers),
        "decisions": expected_decisions,
        "provider_decisions": expected_provider_decisions,
    }
    expected_cost = _call_totals(all_call_rows)
    if replay.get("totals") != expected_totals or replay.get("cost") != expected_cost:
        raise ReplayCheckpointError(
            "verifier replay corpus totals/cost are not exact checkpoint projections"
        )

    corpus = {
        name: _class_decisions(aggregate[name], all_decisions, all_unbound) for name in classes
    }
    matched = corpus["reference_matched"]
    controls = corpus["false_primary"]
    retained = matched[IndependentDecision.ACCEPT.value]
    caught = controls[IndependentDecision.REJECT.value] + controls[IndependentDecision.REVIEW.value]
    report = {
        "schema_version": "verifier-replay-measurement/0.3",
        "run_root": run_root.name,
        "replay_root": replay_root.name,
        "scope": replay["scope"],
        "verifier_model": replay["verifier"]["model"],
        "totals": expected_totals,
        "cost": expected_cost,
        "classes": corpus,
        "headline": {
            "true_positive_retention": (
                retained / matched["verified"] if matched["verified"] else None
            ),
            "true_positive_retention_basis": matched["verified"],
            "false_primary_caught": (
                caught / controls["verified"] if controls["verified"] else None
            ),
            "false_primary_caught_basis": controls["verified"],
            "basis_note": (
                "Denominators are the frozen per-paper reference annotations and negative-control "
                "matches recorded by reference-score/0.7. These sampled class rates are not "
                "whole-paper rates."
            ),
        },
        "papers": per_paper,
    }
    write_json(replay_root / "verifier-replay-measurement.json", report)
    return report
