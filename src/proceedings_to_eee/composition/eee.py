"""Compose schema-ready EEE records from eligible observations only."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from proceedings_to_eee.domain.observation import CandidateObservation, RoleAssignment
from proceedings_to_eee.domain.status import ActorRole, ExportStatus
from proceedings_to_eee.sources.manifest import FrozenSource, SourceManifest


def _evaluated_system(candidate: CandidateObservation) -> RoleAssignment:
    return next(role for role in candidate.roles if role.role == ActorRole.EVALUATED_SYSTEM)


def _stable_key(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{slug[:48]}-{digest}" if slug else digest


def _evaluation_system_key(system_id: str, version: str | None) -> str:
    """Keep system-version groups deterministic and collision-resistant."""

    system_key = _stable_key(system_id)
    if version is None:
        return system_key
    return f"{system_key}-version-{_stable_key(version)}"


def _scope_details(candidate: CandidateObservation) -> dict[str, str]:
    if candidate.scope is None:
        return {}
    details: dict[str, str] = {}
    for name in (
        "dataset_version",
        "split",
        "subset",
        "group",
        "language",
        "aggregation",
        "raw_scope",
    ):
        value = getattr(candidate.scope, name)
        if value is not None:
            details[name] = str(value)
    if candidate.scope.sample_count is not None:
        details["samples_number_reported"] = str(candidate.scope.sample_count)
    return details


def _source_data(candidate: CandidateObservation) -> dict[str, Any]:
    assert candidate.scope is not None
    if candidate.scope.dataset_url:
        return {
            "dataset_name": candidate.scope.dataset_raw,
            "source_type": "url",
            "url": [candidate.scope.dataset_url],
            "additional_details": _scope_details(candidate),
        }
    return {
        "dataset_name": candidate.scope.dataset_raw,
        "source_type": "other",
        "additional_details": _scope_details(candidate),
    }


def _metric_config(candidate: CandidateObservation) -> dict[str, Any]:
    assert candidate.metric is not None
    metric = candidate.metric
    config: dict[str, Any] = {
        "metric_id": metric.canonical_id,
        "metric_name": metric.raw_name,
        "metric_kind": metric.kind or metric.canonical_id,
        "metric_unit": metric.unit or (candidate.value.unit if candidate.value else None),
        "metric_parameters": metric.parameters,
        "lower_is_better": metric.lower_is_better,
    }
    if metric.min_score is not None and metric.max_score is not None:
        config.update(
            {
                "score_type": "continuous",
                "min_score": metric.min_score,
                "max_score": metric.max_score,
            }
        )
    details: dict[str, str] = {}
    if candidate.evaluation_construct:
        details["construct"] = candidate.evaluation_construct
    if candidate.operationalization:
        details["operationalization"] = candidate.operationalization
    if candidate.decision_rule:
        details["decision_rule"] = candidate.decision_rule
    instrument_names = [
        role.raw_name for role in candidate.roles if role.role == ActorRole.EVALUATION_INSTRUMENT
    ]
    if instrument_names:
        details["evaluation_instrument"] = " | ".join(instrument_names)
    if details:
        config["additional_details"] = details
    return {key: value for key, value in config.items() if value is not None}


def _evidence_provenance(
    candidate: CandidateObservation,
    sources: Mapping[str, FrozenSource],
) -> dict[str, str]:
    """Flatten quote-free source anchors into EEE's string-only details map."""

    provenance = {
        "paper_id": candidate.paper_id,
        "evidence_anchor_count": str(len(candidate.evidence)),
    }
    for index, anchor in enumerate(candidate.evidence, start=1):
        source = sources.get(anchor.source_id)
        if source is None:
            raise ValueError(
                f"candidate evidence source {anchor.source_id!r} is absent from the manifest"
            )
        prefix = f"evidence_{index}"
        provenance.update(
            {
                f"{prefix}_source_id": anchor.source_id,
                f"{prefix}_source_role": source.role.value,
                f"{prefix}_page": str(anchor.page),
                f"{prefix}_kind": anchor.kind.value,
                f"{prefix}_quote_sha256": str(anchor.quote_sha256),
            }
        )
        if source.sha256:
            provenance[f"{prefix}_source_sha256"] = source.sha256
        if source.git_commit:
            provenance[f"{prefix}_source_git_commit"] = source.git_commit
        for name in ("label", "row", "column"):
            value = getattr(anchor, name)
            if value is not None:
                provenance[f"{prefix}_{name}"] = value
    return provenance


def _score_details(
    candidate: CandidateObservation,
    sources: Mapping[str, FrozenSource],
) -> dict[str, Any]:
    assert candidate.value is not None
    details: dict[str, Any] = {
        "score": candidate.value.numeric,
        "details": {
            "raw_reported_value": candidate.value.raw,
            "value_comparator": candidate.value.comparator.value,
            "candidate_observation_id": str(candidate.observation_id),
            **_evidence_provenance(candidate, sources),
        },
    }
    uncertainty = candidate.value.uncertainty
    if uncertainty:
        mapped: dict[str, Any] = {}
        if uncertainty.standard_error is not None:
            mapped["standard_error"] = {
                "value": uncertainty.standard_error,
                **({"method": uncertainty.method} if uncertainty.method else {}),
            }
        if (
            uncertainty.confidence_interval_lower is not None
            and uncertainty.confidence_interval_upper is not None
        ):
            interval: dict[str, Any] = {
                "lower": uncertainty.confidence_interval_lower,
                "upper": uncertainty.confidence_interval_upper,
            }
            if uncertainty.confidence_level is not None:
                interval["confidence_level"] = uncertainty.confidence_level
            if uncertainty.method:
                interval["method"] = uncertainty.method
            mapped["confidence_interval"] = interval
        if uncertainty.standard_deviation is not None:
            mapped["standard_deviation"] = uncertainty.standard_deviation
        if uncertainty.num_samples is not None:
            mapped["num_samples"] = uncertainty.num_samples
        if mapped:
            details["uncertainty"] = mapped
    return details


def compose_eee_records(
    *,
    manifest: SourceManifest,
    candidates: Iterable[CandidateObservation],
    schema_version: str,
) -> list[dict[str, Any]]:
    """Group positively paper-produced eligible observations by evaluated system."""

    eligible = [
        candidate
        for candidate in candidates
        if candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}
        and candidate.attribution is not None
        and candidate.attribution.allows_canonical_export
    ]
    groups: dict[tuple[str, str | None], list[CandidateObservation]] = defaultdict(list)
    for candidate in eligible:
        system = _evaluated_system(candidate)
        groups[(system.canonical_id or system.raw_name, system.version)].append(candidate)
    retrieved_at = max(source.retrieved_at for source in manifest.sources)
    sources = {source.source_id: source for source in manifest.sources}
    records: list[dict[str, Any]] = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1] is not None,
            item[0][1] or "",
        ),
    )
    for (system_id, version), observations in ordered_groups:
        observations = sorted(
            observations,
            key=lambda item: (
                item.scope.dataset_raw if item.scope else "",
                item.scope.raw_scope or "" if item.scope else "",
                item.metric.canonical_id or item.metric.raw_name if item.metric else "",
                item.value.numeric if item.value else 0.0,
                item.observation_id or "",
            ),
        )
        role = _evaluated_system(observations[0])
        model_details = {"identity_status": "canonical" if role.canonical_id else "raw_name"}
        if role.version:
            model_details["reported_version"] = role.version
        model_info: dict[str, Any] = {
            "name": role.raw_name,
            "id": role.canonical_id or role.raw_name,
            "additional_details": model_details,
        }
        if role.provider:
            model_info["developer"] = role.provider
        evaluation_results: list[dict[str, Any]] = []
        for candidate in observations:
            assert candidate.scope and candidate.metric and candidate.value
            result: dict[str, Any] = {
                "evaluation_result_id": candidate.observation_id,
                "evaluation_name": f"{candidate.scope.dataset_raw} / {candidate.metric.raw_name}",
                "source_data": _source_data(candidate),
                "metric_config": _metric_config(candidate),
                "score_details": _score_details(candidate, sources),
            }
            if candidate.evaluation_date:
                result["evaluation_timestamp"] = candidate.evaluation_date
            evaluation_results.append(result)
            candidate.export_status = ExportStatus.EXPORTED
        record = {
            "schema_version": schema_version,
            "evaluation_id": (
                f"paper/{manifest.paper_id}/{_evaluation_system_key(system_id, version)}"
            ),
            "retrieved_timestamp": str(int(retrieved_at.timestamp())),
            "source_metadata": {
                "source_name": manifest.title,
                "source_type": "documentation",
                "source_organization_name": "paper authors",
                "evaluator_relationship": "other",
                "additional_details": {
                    "paper_id": manifest.paper_id,
                    **({"doi": manifest.doi} if manifest.doi else {}),
                    **({"arxiv_id": manifest.arxiv_id} if manifest.arxiv_id else {}),
                },
            },
            "model_info": model_info,
            "eval_library": {
                "name": "paper-reported",
                "version": "unknown",
                "additional_details": {"ingestion_method": "proceedings-to-eee"},
            },
            "evaluation_results": evaluation_results,
        }
        records.append(record)
    return records
