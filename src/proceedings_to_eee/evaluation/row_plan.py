"""Offline aggregate planning for the bounded dense-table row stage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from proceedings_to_eee.corpus import CorpusSpec, PaperSpec, build_corpus_binding
from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.extraction.llm import (
    EXTRACTOR_SEED,
    extractor_request_contract,
    row_extractor_request_contract,
)
from proceedings_to_eee.extraction.pdf_layout import (
    PdfLayout,
    extract_pdf_layout,
    select_result_pages,
)
from proceedings_to_eee.extraction.prompt import prompt_hash, row_prompt_hash
from proceedings_to_eee.extraction.result_blocks import (
    ResultBlock,
    ResultBlockConfig,
    maximum_legacy_block_invocations,
    segment_page_result_blocks,
)
from proceedings_to_eee.extraction.row_enumeration import (
    RowEnumerationConfig,
    RowEnumerationPlan,
    build_row_enumeration_plan,
)
from proceedings_to_eee.io import canonical_json_bytes, read_json, sha256_bytes, write_json
from proceedings_to_eee.providers.openrouter import ProviderCall
from proceedings_to_eee.sources.manifest import (
    SourceManifest,
    SourceRole,
    resolve_cached_path,
)

ROW_PLAN_REPORT_SCHEMA_VERSION = "row-plan-report/0.1"
NEXT_RUN_ROW_PLAN_REPORT_SCHEMA_VERSION = "next-run-row-plan-report/0.1"

_EXTRACTOR_CHECKPOINT_SCHEMA_VERSION = "extractor-block-checkpoint/0.1"
_EXTRACTOR_CHECKPOINT_CONTRACT_VERSION = "extractor-block-checkpoint-contract/0.1"
_ROW_CHECKPOINT_SCHEMA_VERSION = "row-enumeration-checkpoint/0.1"
_ROW_CHECKPOINT_CONTRACT_VERSION = "row-enumeration-checkpoint-contract/0.1"
_MAX_TRANSPORT_ATTEMPTS_PER_INVOCATION = 4


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nonnegative_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{label} must be a non-negative number")
    return float(value)


def _paper_baseline(manifest: Mapping[str, Any], paper_id: str) -> dict[str, Any]:
    extractor = _mapping(manifest.get("extractor"), f"{paper_id}: extractor")
    telemetry = _mapping(
        extractor.get("successful_call_telemetry"),
        f"{paper_id}: extractor.successful_call_telemetry",
    )
    calls = _nonnegative_int(
        telemetry.get("calls"),
        f"{paper_id}: successful_call_telemetry.calls",
    )

    cost_reported_raw = telemetry.get("cost_reported_calls")
    cost_raw = telemetry.get("cost_usd_lower_bound")
    cost_basis_complete = False
    cost_reported_calls: int | None = None
    cost_usd_lower_bound: float | None = None
    if cost_reported_raw is not None:
        cost_reported_calls = _nonnegative_int(
            cost_reported_raw,
            f"{paper_id}: successful_call_telemetry.cost_reported_calls",
        )
        if cost_reported_calls > calls:
            raise ValueError(f"{paper_id}: cost_reported_calls exceeds successful calls")
    if cost_raw is not None:
        cost_usd_lower_bound = _nonnegative_number(
            cost_raw,
            f"{paper_id}: successful_call_telemetry.cost_usd_lower_bound",
        )
    cost_basis_complete = cost_reported_calls == calls and cost_usd_lower_bound is not None

    return {
        "legacy_block_calls": calls,
        "legacy_call_basis": "run.json extractor.successful_call_telemetry.calls",
        "cost_basis_complete": cost_basis_complete,
        "cost_reported_calls": cost_reported_calls,
        "cost_usd_lower_bound": cost_usd_lower_bound,
    }


def build_run_row_plan_report(
    run_root: Path,
    *,
    config: RowEnumerationConfig | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Replay stored layout/block artifacts without calling a provider.

    This compatibility API is a historical diagnostic, not a next-run preflight: current
    page selection or segmentation may differ from the stored artifacts. Cost projections
    require explicit cost for every historical successful legacy call.
    """

    if not run_root.is_dir():
        raise ValueError(f"run root does not exist or is not a directory: {run_root}")
    paper_dirs = [
        path
        for path in sorted(run_root.iterdir())
        if path.is_dir() and (path / "run.json").is_file()
    ]
    if not paper_dirs:
        raise ValueError(f"run root contains no paper run manifests: {run_root}")

    config = config or RowEnumerationConfig()
    papers: list[dict[str, Any]] = []
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name
        manifest = _mapping(read_json(paper_dir / "run.json"), f"{paper_id}: run.json")
        layout_path = paper_dir / "private" / "layout.json"
        blocks_path = paper_dir / "private" / "result-blocks.json"
        if not layout_path.is_file() or not blocks_path.is_file():
            raise ValueError(f"{paper_id}: row planning requires private layout and result blocks")
        layout = PdfLayout.model_validate(read_json(layout_path))
        blocks_raw = read_json(blocks_path)
        if not isinstance(blocks_raw, list):
            raise ValueError(f"{paper_id}: result-blocks.json must be a list")
        blocks = [ResultBlock.model_validate(item) for item in blocks_raw]
        plan = build_row_enumeration_plan(layout, blocks, config=config)
        baseline = _paper_baseline(manifest, paper_id)
        papers.append(
            {
                "paper_id": paper_id,
                **baseline,
                "tables_considered": plan.telemetry.tables_considered,
                "dense_tables": plan.telemetry.dense_tables,
                "rows_planned": plan.telemetry.rows_planned,
                "unbatchable_rows": plan.telemetry.unbatchable_rows,
                "base_batches": plan.telemetry.base_batches,
                "expected_row_calls": plan.telemetry.expected_calls,
                "hard_max_row_calls": plan.telemetry.maximum_calls,
            }
        )

    legacy_calls = sum(item["legacy_block_calls"] for item in papers)
    expected_row_calls = sum(item["expected_row_calls"] for item in papers)
    hard_max_row_calls = sum(item["hard_max_row_calls"] for item in papers)
    expected_total_calls = legacy_calls + expected_row_calls
    hard_max_total_calls = legacy_calls + hard_max_row_calls
    expected_multiplier = expected_total_calls / legacy_calls if legacy_calls else None
    hard_max_multiplier = hard_max_total_calls / legacy_calls if legacy_calls else None

    cost_complete = legacy_calls > 0 and all(item["cost_basis_complete"] for item in papers)
    cost_estimates: dict[str, Any] | None = None
    if cost_complete:
        historical_cost = sum(item["cost_usd_lower_bound"] for item in papers)
        mean_cost = historical_cost / legacy_calls
        expected_additional = mean_cost * expected_row_calls
        hard_max_additional = mean_cost * hard_max_row_calls
        cost_estimates = {
            "available": True,
            "estimate_label": (
                "Planning estimate from the historical mean cost of successful final legacy "
                "block calls; not a quote, budget guarantee, or measured row-stage cost."
            ),
            "historical_basis": (
                "All successful legacy calls in each included run manifest report cost; the "
                "historical total remains a lower bound when failed/superseded attempts are absent."
            ),
            "historical_cost_usd_lower_bound": historical_cost,
            "historical_successful_calls": legacy_calls,
            "historical_mean_cost_per_successful_call_usd": mean_cost,
            "estimated_expected_additional_row_cost_usd": expected_additional,
            "estimated_expected_combined_cost_usd": historical_cost + expected_additional,
            "estimated_hard_max_additional_row_cost_usd": hard_max_additional,
            "estimated_hard_max_combined_cost_usd": historical_cost + hard_max_additional,
        }

    report = {
        "schema_version": ROW_PLAN_REPORT_SCHEMA_VERSION,
        "run_root": run_root.name,
        "mode": "offline_stored_artifact_diagnostic",
        "config": config.model_dump(mode="json"),
        "papers_planned": len(papers),
        "legacy_block_baseline": {
            "successful_calls": legacy_calls,
            "basis": (
                "Sum of run.json extractor.successful_call_telemetry.calls; these are "
                "successful final legacy block calls, not all provider attempts."
            ),
        },
        "row_plan": {
            "tables_considered": sum(item["tables_considered"] for item in papers),
            "dense_tables": sum(item["dense_tables"] for item in papers),
            "rows_planned": sum(item["rows_planned"] for item in papers),
            "unbatchable_rows": sum(item["unbatchable_rows"] for item in papers),
            "base_batches": sum(item["base_batches"] for item in papers),
            "expected_row_calls": expected_row_calls,
            "hard_max_row_calls": hard_max_row_calls,
            "basis": (
                "Expected row calls assume one call per base batch. Hard maximum applies the "
                "configured deterministic recovery bound to every base batch."
            ),
        },
        "call_estimates": {
            "estimate_label": "Planning estimates; no row-stage provider calls were made.",
            "expected_total_calls": expected_total_calls,
            "hard_max_total_calls": hard_max_total_calls,
            "expected_total_call_multiplier": expected_multiplier,
            "hard_max_total_call_multiplier": hard_max_multiplier,
            "multiplier_denominator": "historical successful final legacy block calls",
        },
        "cost_estimate_status": {
            "available": cost_complete,
            "reason": (
                "complete explicit per-call historical cost basis"
                if cost_complete
                else "requires explicit cost for every historical successful legacy call"
            ),
        },
        "cost_estimates": cost_estimates,
        "papers": papers,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def _current_extractor_configuration(model: str) -> dict[str, Any]:
    return {
        "provider": "openrouter",
        "model": model,
        "temperature": 0.0,
        "reasoning_effort": "minimal",
        "max_tokens": 16_000,
        "seed": EXTRACTOR_SEED,
        "prompt_sha256": prompt_hash(),
        "request_contract": extractor_request_contract(seed=EXTRACTOR_SEED),
    }


def _current_row_extractor_configuration(model: str) -> dict[str, Any]:
    return {
        "provider": "openrouter",
        "model": model,
        "temperature": 0.0,
        "reasoning_effort": "minimal",
        "max_tokens": 16_000,
        "seed": EXTRACTOR_SEED,
        "prompt_sha256": row_prompt_hash(),
        "request_contract": row_extractor_request_contract(seed=EXTRACTOR_SEED),
    }


def _reconstruct_frozen_layout(
    *,
    paper_dir: Path,
    spec: PaperSpec,
    project_root: Path,
) -> tuple[SourceManifest, PdfLayout]:
    """Re-run the current local layout parser over the already frozen paper bytes."""

    manifest_path = paper_dir / "source-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{spec.paper_id}: missing frozen source manifest")
    manifest = SourceManifest.model_validate(read_json(manifest_path))
    if manifest.paper_id != spec.paper_id:
        raise ValueError(f"{spec.paper_id}: source manifest paper id mismatch")
    configured_sources: list[tuple[SourceRole, str, str | None]] = [
        (SourceRole.PAPER, str(spec.pdf_url), None),
        *[(SourceRole.SUPPLEMENT, str(url), None) for url in spec.supplement_urls],
    ]
    if spec.repository_url and spec.repository_commit:
        configured_sources.append(
            (
                SourceRole.REPOSITORY,
                str(spec.repository_url),
                spec.repository_commit.casefold(),
            )
        )
    frozen_sources = [
        (source.role, source.original_uri, source.git_commit) for source in manifest.sources
    ]
    if frozen_sources != configured_sources:
        raise ValueError(f"{spec.paper_id}: corpus source bundle differs from frozen manifest")

    paper_sources = [source for source in manifest.sources if source.role is SourceRole.PAPER]
    if len(paper_sources) != 1:
        raise ValueError(f"{spec.paper_id}: expected exactly one frozen paper source")
    for source in manifest.sources:
        if source.role is not SourceRole.REPOSITORY:
            resolve_cached_path(source, project_root)
    paper_source = paper_sources[0]
    paper_path = resolve_cached_path(paper_source, project_root)
    return manifest, extract_pdf_layout(paper_path, paper_source.source_id)


def _current_blocks(
    *,
    layout: PdfLayout,
    spec: PaperSpec,
    max_blocks_per_page: int,
) -> tuple[list[int], list[ResultBlock]]:
    if spec.include_pages:
        missing = [page for page in spec.include_pages if page < 1 or page > layout.page_count]
        if missing:
            raise ValueError(f"configured pages outside PDF for {spec.paper_id}: {missing}")
        selected = [layout.pages[page - 1] for page in spec.include_pages]
    else:
        selected = select_result_pages(layout, limit=spec.max_result_pages)
    block_config = ResultBlockConfig(max_blocks_per_page=max_blocks_per_page)
    blocks = [
        block
        for page in selected
        for block in segment_page_result_blocks(page, config=block_config)
    ]
    block_ids = [block.block_id for block in blocks]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError(f"{spec.paper_id}: current segmentation produced duplicate block ids")
    return [page.page for page in selected], blocks


def _checkpoint_entry_is_reusable(entry: object, block: ResultBlock) -> bool:
    if not isinstance(entry, Mapping) or entry.get("block_text_sha256") != block.text_sha256:
        return False
    candidates = entry.get("candidates")
    warnings = entry.get("warnings")
    if not isinstance(candidates, list) or not isinstance(warnings, list):
        return False
    if not all(isinstance(warning, str) for warning in warnings):
        return False
    try:
        for candidate in candidates:
            CandidateObservation.model_validate(candidate)
        ProviderCall.model_validate(entry.get("call"))
    except (TypeError, ValueError):
        return False
    return True


def _checkpoint_reuse(
    *,
    paper_dir: Path,
    spec: PaperSpec,
    manifest: SourceManifest,
    layout: PdfLayout,
    blocks: list[ResultBlock],
    extractor_model: str,
) -> dict[str, Any]:
    checkpoint_path = paper_dir / "private" / "extractor-checkpoint.json"
    expected_envelope = {
        "schema_version": _EXTRACTOR_CHECKPOINT_CONTRACT_VERSION,
        "paper_id": spec.paper_id,
        "paper_title": spec.title,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_parser": layout.parser,
        "layout_parser_version": layout.parser_version,
        "extractor": _current_extractor_configuration(extractor_model),
    }
    status = "missing"
    mismatched_fields: list[str] = []
    entries: Mapping[str, Any] = {}
    stored_entry_count = 0
    if checkpoint_path.is_file():
        try:
            payload = read_json(checkpoint_path)
        except (OSError, ValueError):
            payload = None
        if (
            isinstance(payload, Mapping)
            and payload.get("schema_version") == _EXTRACTOR_CHECKPOINT_SCHEMA_VERSION
        ):
            contract = payload.get("contract")
            raw_entries = payload.get("blocks")
            if isinstance(contract, Mapping) and isinstance(raw_entries, Mapping):
                stored_entry_count = len(raw_entries)
                mismatched_fields = [
                    key
                    for key, expected in expected_envelope.items()
                    if contract.get(key) != expected
                ]
                if not mismatched_fields:
                    status = "compatible"
                    entries = raw_entries
                else:
                    status = "incompatible_request_envelope"
            else:
                status = "malformed"
        else:
            status = "malformed"

    reusable_block_ids = {
        block.block_id
        for block in blocks
        if _checkpoint_entry_is_reusable(entries.get(block.block_id), block)
    }
    new_blocks = [block for block in blocks if block.block_id not in reusable_block_ids]
    reusable = len(reusable_block_ids)
    return {
        "status": status,
        "current_blocks": len(blocks),
        "checkpoint_entries": stored_entry_count,
        "reusable_blocks": reusable,
        "new_block_calls": len(blocks) - reusable,
        "hard_max_new_structured_chat_invocations": sum(
            maximum_legacy_block_invocations(block) for block in new_blocks
        ),
        "reuse_fraction": reusable / len(blocks) if blocks else None,
        "mismatched_request_envelope_fields": mismatched_fields,
        "basis": (
            "Mirrors pipeline checkpoint migration and rehydration: the request envelope "
            "must match, then block id, exact block text hash, candidates, warnings, and "
            "provider-call telemetry must validate."
        ),
    }


def _row_checkpoint_reuse(
    *,
    paper_dir: Path,
    spec: PaperSpec,
    manifest: SourceManifest,
    layout: PdfLayout,
    plan: RowEnumerationPlan,
    extractor_model: str,
) -> dict[str, Any]:
    """Validate exact row batches under the pipeline's request-reuse envelope."""

    # Keep compatibility and typed rehydration identical to the execution path without
    # making the pipeline depend on its evaluation-only preflight module.
    from proceedings_to_eee.pipeline import (
        _row_checkpoint_reuse_envelope,
        _validated_row_checkpoint_entry,
    )

    checkpoint_path = paper_dir / "private" / "row-enumeration-checkpoint.json"
    expected_contract = {
        "schema_version": _ROW_CHECKPOINT_CONTRACT_VERSION,
        "paper_id": spec.paper_id,
        "paper_title": spec.title,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "layout_parser": layout.parser,
        "layout_parser_version": layout.parser_version,
        "plan_sha256": sha256_bytes(canonical_json_bytes(plan)),
        "row_enumeration": {
            "enabled": True,
            **_current_row_extractor_configuration(extractor_model),
            "limits": plan.config.model_dump(mode="json"),
        },
    }
    expected_envelope = _row_checkpoint_reuse_envelope(expected_contract)
    status = "missing"
    mismatched_fields: list[str] = []
    entries: Mapping[str, Any] = {}
    stored_entry_count = 0
    if checkpoint_path.is_file():
        try:
            payload = read_json(checkpoint_path)
        except (OSError, ValueError):
            payload = None
        if (
            isinstance(payload, Mapping)
            and payload.get("schema_version") == _ROW_CHECKPOINT_SCHEMA_VERSION
        ):
            contract = payload.get("contract")
            raw_entries = payload.get("batches")
            if isinstance(contract, Mapping) and isinstance(raw_entries, Mapping):
                stored_entry_count = len(raw_entries)
                mismatched_fields = [
                    key
                    for key, expected in expected_envelope.items()
                    if contract.get(key) != expected
                ]
                if not mismatched_fields:
                    status = "compatible"
                    entries = raw_entries
                else:
                    status = "incompatible_request_envelope"
            else:
                status = "malformed"
        else:
            status = "malformed"

    reusable = sum(
        _validated_row_checkpoint_entry(entries.get(batch.batch_id), batch=batch) is not None
        for batch in plan.batches
    )
    return {
        "status": status,
        "current_base_batches": len(plan.batches),
        "checkpoint_entries": stored_entry_count,
        "reusable_checkpoint_batches": reusable,
        "new_base_batches": len(plan.batches) - reusable,
        "reuse_fraction": reusable / len(plan.batches) if plan.batches else None,
        "mismatched_request_envelope_fields": mismatched_fields,
        "basis": (
            "Mirrors pipeline row-checkpoint migration and rehydration: the request envelope "
            "and exact plan must match, then batch id, exact batch hash, typed records, call "
            "telemetry, and the resolved/unresolved row partition must validate. Code remains "
            "recorded for provenance but is not itself a provider-request input."
        ),
    }


def _stored_artifact_comparison(
    *,
    paper_dir: Path,
    config: RowEnumerationConfig,
) -> dict[str, Any] | None:
    layout_path = paper_dir / "private" / "layout.json"
    blocks_path = paper_dir / "private" / "result-blocks.json"
    if not layout_path.is_file() or not blocks_path.is_file():
        return None
    layout = PdfLayout.model_validate(read_json(layout_path))
    blocks_raw = read_json(blocks_path)
    if not isinstance(blocks_raw, list):
        raise ValueError(f"{paper_dir.name}: stored result-blocks.json must be a list")
    blocks = [ResultBlock.model_validate(item) for item in blocks_raw]
    plan = build_row_enumeration_plan(layout, blocks, config=config)
    return {
        "result_blocks": len(blocks),
        **plan.telemetry.model_dump(mode="json"),
    }


def build_next_run_row_plan_report(
    run_root: Path,
    *,
    corpus: CorpusSpec,
    project_root: Path,
    extractor_model: str,
    config: RowEnumerationConfig | None = None,
    max_blocks_per_page: int = 6,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Reconstruct the exact current-code next-run preflight from frozen source bytes.

    This function never contacts a provider or the network. It reruns only the local PDF
    layout parser, page selection, block segmentation, row planning, and exact typed
    checkpoint validation.
    """

    if not run_root.is_dir():
        raise ValueError(f"run root does not exist or is not a directory: {run_root}")
    if not extractor_model.strip():
        raise ValueError("extractor model is required for exact checkpoint reuse")
    if max_blocks_per_page < 1:
        raise ValueError("max_blocks_per_page must be positive")
    paper_dirs = {
        path.name: path
        for path in sorted(run_root.iterdir())
        if path.is_dir() and (path / "run.json").is_file()
    }
    specs = {spec.paper_id: spec for spec in corpus.papers}
    if len(specs) != len(corpus.papers):
        raise ValueError("corpus contains duplicate paper ids")
    if set(paper_dirs) != set(specs):
        raise ValueError(
            "corpus paper ids must exactly match existing run directories; "
            f"only_in_run={sorted(set(paper_dirs) - set(specs))!r}, "
            f"only_in_corpus={sorted(set(specs) - set(paper_dirs))!r}"
        )

    config = config or RowEnumerationConfig()
    papers: list[dict[str, Any]] = []
    for spec in corpus.papers:
        paper_dir = paper_dirs[spec.paper_id]
        run_manifest = _mapping(read_json(paper_dir / "run.json"), f"{spec.paper_id}: run.json")
        historical = _paper_baseline(run_manifest, spec.paper_id)
        source_manifest, layout = _reconstruct_frozen_layout(
            paper_dir=paper_dir,
            spec=spec,
            project_root=project_root,
        )
        selected_pages, blocks = _current_blocks(
            layout=layout,
            spec=spec,
            max_blocks_per_page=max_blocks_per_page,
        )
        plan = build_row_enumeration_plan(layout, blocks, config=config)
        checkpoint = _checkpoint_reuse(
            paper_dir=paper_dir,
            spec=spec,
            manifest=source_manifest,
            layout=layout,
            blocks=blocks,
            extractor_model=extractor_model,
        )
        row_checkpoint = _row_checkpoint_reuse(
            paper_dir=paper_dir,
            spec=spec,
            manifest=source_manifest,
            layout=layout,
            plan=plan,
            extractor_model=extractor_model,
        )
        papers.append(
            {
                "paper_id": spec.paper_id,
                **historical,
                "selected_pages": selected_pages,
                "current_result_blocks": len(blocks),
                "stored_artifacts": _stored_artifact_comparison(
                    paper_dir=paper_dir,
                    config=config,
                ),
                "checkpoint_reuse": checkpoint,
                "row_checkpoint_reuse": row_checkpoint,
                "tables_considered": plan.telemetry.tables_considered,
                "dense_tables": plan.telemetry.dense_tables,
                "rows_planned": plan.telemetry.rows_planned,
                "unbatchable_rows": plan.telemetry.unbatchable_rows,
                "base_batches": plan.telemetry.base_batches,
                "expected_row_calls": plan.telemetry.expected_calls,
                "hard_max_row_calls": plan.telemetry.maximum_calls,
            }
        )

    historical_calls = sum(item["legacy_block_calls"] for item in papers)
    current_blocks = sum(item["current_result_blocks"] for item in papers)
    reusable_blocks = sum(item["checkpoint_reuse"]["reusable_blocks"] for item in papers)
    new_block_calls = current_blocks - reusable_blocks
    hard_max_new_legacy_invocations = sum(
        item["checkpoint_reuse"]["hard_max_new_structured_chat_invocations"] for item in papers
    )
    expected_row_calls = sum(item["expected_row_calls"] for item in papers)
    hard_max_row_calls = sum(item["hard_max_row_calls"] for item in papers)
    reusable_row_batches = sum(
        item["row_checkpoint_reuse"]["reusable_checkpoint_batches"] for item in papers
    )
    new_row_base_batches = expected_row_calls - reusable_row_batches
    hard_max_new_row_invocations = new_row_base_batches * (1 + 2 * config.max_recovery_depth)
    expected_logical_calls = current_blocks + expected_row_calls
    hard_max_logical_calls = (
        reusable_blocks
        + hard_max_new_legacy_invocations
        + expected_row_calls
        + 2 * config.max_recovery_depth * new_row_base_batches
    )
    expected_new_invocations = new_block_calls + new_row_base_batches
    hard_max_new_invocations = hard_max_new_legacy_invocations + hard_max_new_row_invocations

    stored = [item["stored_artifacts"] for item in papers]
    stored_complete = all(item is not None for item in stored)
    stored_summary = None
    if stored_complete:
        stored_summary = {
            "result_blocks": sum(item["result_blocks"] for item in stored),
            "tables_considered": sum(item["tables_considered"] for item in stored),
            "dense_tables": sum(item["dense_tables"] for item in stored),
            "rows_planned": sum(item["rows_planned"] for item in stored),
            "unbatchable_rows": sum(item["unbatchable_rows"] for item in stored),
            "base_batches": sum(item["base_batches"] for item in stored),
            "expected_calls": sum(item["expected_calls"] for item in stored),
            "maximum_calls": sum(item["maximum_calls"] for item in stored),
            "basis": "Stored layout/result-block artifacts; diagnostic only, not next-run input.",
        }

    cost_complete = historical_calls > 0 and all(item["cost_basis_complete"] for item in papers)
    cost_estimates = None
    if cost_complete:
        historical_cost = sum(item["cost_usd_lower_bound"] for item in papers)
        mean_cost = historical_cost / historical_calls
        cost_estimates = {
            "available": True,
            "estimate_label": (
                "Next-run planning estimate from the historical mean cost of successful final "
                "legacy structured-chat invocations; not a provider quote or budget guarantee. "
                "Transport attempts are not priced independently here."
            ),
            "historical_cost_usd_lower_bound": historical_cost,
            "historical_successful_structured_chat_invocations": historical_calls,
            "historical_mean_cost_per_successful_structured_chat_invocation_usd": mean_cost,
            "estimated_expected_new_run_cost_usd": mean_cost * expected_new_invocations,
            "estimated_hard_max_new_run_cost_usd": mean_cost * hard_max_new_invocations,
        }

    report = {
        "schema_version": NEXT_RUN_ROW_PLAN_REPORT_SCHEMA_VERSION,
        "run_root": run_root.name,
        "corpus_id": corpus.corpus_id,
        "corpus_binding": build_corpus_binding(corpus),
        "mode": "offline_current_code_frozen_source_preflight",
        "provider_or_network_calls": 0,
        "extractor_model": extractor_model,
        "extractor_contract": _current_extractor_configuration(extractor_model),
        "row_extractor_contract": _current_row_extractor_configuration(extractor_model),
        "row_config": config.model_dump(mode="json"),
        "papers_planned": len(papers),
        "stored_artifact_comparison": stored_summary,
        "next_run_blocks": {
            "current_result_blocks": current_blocks,
            "reusable_checkpoint_blocks": reusable_blocks,
            "new_block_calls": new_block_calls,
            "reuse_fraction": reusable_blocks / current_blocks if current_blocks else None,
            "basis": (
                "Current local layout extraction, current corpus page selection, current block "
                "segmentation, and fully validated exact historical checkpoint entries."
            ),
        },
        "row_plan": {
            "tables_considered": sum(item["tables_considered"] for item in papers),
            "dense_tables": sum(item["dense_tables"] for item in papers),
            "rows_planned": sum(item["rows_planned"] for item in papers),
            "unbatchable_rows": sum(item["unbatchable_rows"] for item in papers),
            "base_batches": sum(item["base_batches"] for item in papers),
            "expected_row_calls": expected_row_calls,
            "hard_max_row_calls": hard_max_row_calls,
        },
        "next_run_rows": {
            "current_base_batches": expected_row_calls,
            "reusable_checkpoint_batches": reusable_row_batches,
            "new_base_batches": new_row_base_batches,
            "reuse_fraction": (
                reusable_row_batches / expected_row_calls if expected_row_calls else None
            ),
            "expected_new_structured_chat_invocations": new_row_base_batches,
            "hard_max_new_structured_chat_invocations": hard_max_new_row_invocations,
            "basis": (
                "Current exact row plan and fully validated checkpoint entries. Each new base "
                "batch requires one expected invocation and permits one bounded split level."
            ),
        },
        "next_run_preflight": {
            "expected_logical_calls": expected_logical_calls,
            "hard_max_logical_calls": hard_max_logical_calls,
            "expected_logical_call_multiplier": (
                expected_logical_calls / current_blocks if current_blocks else None
            ),
            "hard_max_logical_call_multiplier": (
                hard_max_logical_calls / current_blocks if current_blocks else None
            ),
            "reused_legacy_block_calls": reusable_blocks,
            "reused_row_base_batches": reusable_row_batches,
            "expected_new_row_structured_chat_invocations": new_row_base_batches,
            "hard_max_new_row_structured_chat_invocations": hard_max_new_row_invocations,
            "expected_new_structured_chat_invocations": expected_new_invocations,
            "hard_max_new_structured_chat_invocations": hard_max_new_invocations,
            "max_transport_attempts_per_structured_chat_invocation": (
                _MAX_TRANSPORT_ATTEMPTS_PER_INVOCATION
            ),
            "max_transport_attempts_at_expected_invocation_plan": (
                expected_new_invocations * _MAX_TRANSPORT_ATTEMPTS_PER_INVOCATION
            ),
            "hard_max_new_transport_attempts": (
                hard_max_new_invocations * _MAX_TRANSPORT_ATTEMPTS_PER_INVOCATION
            ),
            "expected_new_invocation_multiplier_vs_historical": (
                expected_new_invocations / historical_calls if historical_calls else None
            ),
            "hard_max_new_invocation_multiplier_vs_historical": (
                hard_max_new_invocations / historical_calls if historical_calls else None
            ),
            "basis": (
                "Logical calls include reused legacy blocks and row base batches. Structured-chat "
                "invocation bounds subtract fully validated legacy and row checkpoints, derive "
                "each new legacy block's bounded recursive split tree from its body lines, and "
                "apply the row stage's configured recovery bound only to new row batches. "
                "OpenRouter transport attempts are separately bounded at four per "
                "structured-chat invocation."
            ),
        },
        "cost_estimate_status": {
            "available": cost_complete,
            "reason": (
                "complete explicit historical cost basis"
                if cost_complete
                else "requires explicit cost for every historical successful legacy call"
            ),
        },
        "cost_estimates": cost_estimates,
        "papers": papers,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report
