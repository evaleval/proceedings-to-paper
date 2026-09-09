"""Compose schema-ready EEE records from eligible observations only."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from proceedings_to_eee.domain.attribution import OriginBasis, ReviewTier
from proceedings_to_eee.domain.export_provenance import (
    CandidateExportAuthorization,
    ExportCompositionProvenance,
    candidate_export_sha256,
)
from proceedings_to_eee.domain.observation import CandidateObservation, RoleAssignment
from proceedings_to_eee.domain.provenance import CandidateField, FieldBindingStatus
from proceedings_to_eee.domain.status import ActorRole, ExportStatus
from proceedings_to_eee.sources.manifest import FrozenSource, SourceManifest
from proceedings_to_eee.validation.field_provenance import (
    literal_setting_components_are_bound,
)


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
    # `score_type` is always stated. EEE 0.2.2 requires `level_names` and
    # `has_unknown_level` when score_type is "levels", and its `if` clause carries no
    # `required`, so an absent score_type satisfies it vacuously and the record is
    # rejected for fields that describe a kind of score it does not have. Every value
    # composed here is a numeric measurement, so "continuous" is the honest answer;
    # the bounds are added only when the registry knows them.
    config["score_type"] = "continuous"
    if metric.min_score is not None and metric.max_score is not None:
        config.update({"min_score": metric.min_score, "max_score": metric.max_score})
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
        for name in (
            "region_id",
            "planned_row_id",
            "cell_id",
            "numeric_token_id",
        ):
            value = getattr(anchor, name)
            if value is not None:
                provenance[f"{prefix}_{name}"] = value
        if anchor.header_ids:
            provenance[f"{prefix}_header_ids"] = "|".join(anchor.header_ids)
    for field in sorted(candidate.field_provenance, key=lambda item: item.field.value):
        prefix = f"field_{field.field.value}"
        provenance[f"{prefix}_value_sha256"] = field.value_sha256
        provenance[f"{prefix}_status"] = field.status.value
        provenance[f"{prefix}_source_count"] = str(len(field.sources))
        if field.reason is not None:
            provenance[f"{prefix}_reason"] = field.reason
        for index, source in enumerate(field.sources, start=1):
            source_prefix = f"{prefix}_source_{index}"
            provenance[f"{source_prefix}_kind"] = source.kind.value
            provenance[f"{source_prefix}_source_id"] = source.source_id
            provenance[f"{source_prefix}_page"] = str(source.page)
            for name in (
                "region_id",
                "planned_row_id",
                "row_label_cell_id",
                "physical_cell_id",
                "numeric_token_id",
                "quote_sha256",
            ):
                value = getattr(source, name)
                if value is not None:
                    provenance[f"{source_prefix}_{name}"] = value
            if source.header_ids:
                provenance[f"{source_prefix}_header_ids"] = "|".join(source.header_ids)
    if candidate.proposal_traces:
        provenance["proposal_ids"] = "|".join(
            sorted(trace.proposal_id for trace in candidate.proposal_traces)
        )
        provenance["candidate_occurrence_ids"] = "|".join(
            sorted(trace.candidate_occurrence_id for trace in candidate.proposal_traces)
        )
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


#: Weakest first. A record is only as strong as its weakest result, so the record-level
#: labels report the minimum over the results it carries.
_ORIGIN_BASIS_STRENGTH = {
    OriginBasis.NONE: 0,
    OriginBasis.MODEL_ASSERTED_PRIMARY_UNCHECKED: 1,
    OriginBasis.MODEL_ASSERTED_PRIMARY_NO_EXTERNAL_CUE: 2,
    OriginBasis.MODEL_REVIEWED_ORIGIN_QUOTE: 3,
    OriginBasis.POSITIVE_STRUCTURAL: 4,
    OriginBasis.HUMAN_CONFIRMED: 5,
}

_REVIEW_TIER_STRENGTH = {
    ReviewTier.DETERMINISTIC: 0,
    ReviewTier.MODEL_REVIEWED: 1,
    ReviewTier.HUMAN_CONFIRMED: 2,
}


def _origin_basis(
    candidate: CandidateObservation, authorization: CandidateExportAuthorization
) -> OriginBasis:
    """The basis this record may claim: what authorized it, else what the verdict allows."""

    if authorization.origin_basis is not None:
        return authorization.origin_basis
    if candidate.attribution is None:
        return OriginBasis.NONE
    return candidate.attribution.origin_basis(candidate.claim_type)


def _review_tier(
    candidate: CandidateObservation, authorization: CandidateExportAuthorization
) -> ReviewTier:
    if authorization.review_tier is not None:
        return authorization.review_tier
    if _origin_basis(candidate, authorization) is OriginBasis.HUMAN_CONFIRMED:
        return ReviewTier.HUMAN_CONFIRMED
    return ReviewTier.DETERMINISTIC


def _weakest_origin_basis(
    observations: Sequence[CandidateObservation],
    authorizations: dict[str, CandidateExportAuthorization],
) -> OriginBasis:
    bases = [
        _origin_basis(candidate, authorizations[candidate.observation_id or candidate.stable_id()])
        for candidate in observations
    ]
    return min(bases, key=lambda basis: _ORIGIN_BASIS_STRENGTH[basis])


def _weakest_review_tier(
    observations: Sequence[CandidateObservation],
    authorizations: dict[str, CandidateExportAuthorization],
) -> ReviewTier:
    tiers = [
        _review_tier(candidate, authorizations[candidate.observation_id or candidate.stable_id()])
        for candidate in observations
    ]
    return min(tiers, key=lambda tier: _REVIEW_TIER_STRENGTH[tier])


def _field_provenance_gate(candidate: CandidateObservation) -> str:
    """Say plainly how this candidate stood with the field-provenance gate."""

    if not candidate.field_provenance:
        return "absent"
    populated = candidate.populated_fields()
    unbound = [
        item
        for item in candidate.field_provenance
        if item.field in populated and item.status is not FieldBindingStatus.BOUND
    ]
    if not unbound:
        return "all_bound"
    if all(item.field is CandidateField.SETTING for item in unbound) and (
        literal_setting_components_are_bound(candidate)
    ):
        return "descriptive_setting_unbound"
    return "unbound"


def compose_eee_records(
    *,
    manifest: SourceManifest,
    candidates: Iterable[CandidateObservation],
    schema_version: str,
    provenance: ExportCompositionProvenance,
) -> list[dict[str, Any]]:
    """Group authorized paper-produced observations by evaluated system."""

    candidates = list(candidates)
    policy = provenance.effective_origin_policy
    eligible = [
        candidate
        for candidate in candidates
        if candidate.export_status in {ExportStatus.ELIGIBLE, ExportStatus.EXPORTED}
        and candidate.attribution is not None
        and policy.permits(candidate.attribution.origin_basis(candidate.claim_type))
    ]
    authorizations = {item.observation_id: item for item in provenance.authorizations}
    eligible_ids = {candidate.observation_id or candidate.stable_id() for candidate in eligible}
    if set(authorizations) != eligible_ids:
        raise ValueError("export provenance does not exactly cover eligible candidates")
    for candidate in eligible:
        observation_id = candidate.observation_id or candidate.stable_id()
        authorization = authorizations[observation_id]
        if authorization.candidate_sha256 != candidate_export_sha256(candidate):
            raise ValueError("export provenance candidate binding is stale or spliced")
        if provenance.mode.is_tuple_gated and (
            authorization.tuple_gate_sha256 is None or authorization.tuple_gate_passed is not True
        ):
            raise ValueError("tuple-gated composition lacks a verified candidate gate")
        if provenance.verifier_gate_required and (
            authorization.verifier_gate_sha256 is None
            or authorization.verifier_gate_passed is not True
        ):
            raise ValueError("verifier-gated composition lacks an accepted candidate gate")
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
            observation_id = candidate.observation_id or candidate.stable_id()
            authorization = authorizations[observation_id]
            result: dict[str, Any] = {
                "evaluation_result_id": observation_id,
                "evaluation_name": f"{candidate.scope.dataset_raw} / {candidate.metric.raw_name}",
                "source_data": _source_data(candidate),
                "metric_config": _metric_config(candidate),
                "score_details": _score_details(candidate, sources),
            }
            result["score_details"]["details"].update(
                {
                    "producer_origin_basis": _origin_basis(candidate, authorization).value,
                    "attribution_state": candidate.attribution.state.value,
                    "attribution_rule_id": candidate.attribution.rule_id,
                    "review_tier": _review_tier(candidate, authorization).value,
                    "referential_status": candidate.referential_status.value,
                    "field_provenance_gate": _field_provenance_gate(candidate),
                    "text_support": candidate.text_support.value,
                    "candidate_observation_id": observation_id,
                    "evidence_page": str(candidate.evidence[0].page),
                    "origin_export_policy": policy.value,
                    "export_provenance_mode": provenance.mode.value,
                    "export_composition_sha256": provenance.sha256,
                    "export_candidate_sha256": authorization.candidate_sha256,
                    **(
                        {
                            "tuple_gate_sha256": authorization.tuple_gate_sha256,
                            "tuple_sidecar_sha256": provenance.tuple_sidecar_sha256,
                        }
                        if provenance.mode.has_tuple_audit
                        else {"legacy_provenance_reason": provenance.legacy_reason}
                    ),
                    **(
                        {"review_manifest_sha256": provenance.review_manifest_sha256}
                        if provenance.review_manifest_sha256 is not None
                        else {}
                    ),
                    "verifier_gate_required": str(provenance.verifier_gate_required).lower(),
                    **(
                        {
                            "source_verifier_gate_passed": str(
                                authorization.verifier_gate_passed
                            ).lower(),
                            "verifier_sidecar_sha256": provenance.verifier_sidecar_sha256,
                            **(
                                {"verifier_gate_sha256": authorization.verifier_gate_sha256}
                                if authorization.verifier_gate_sha256 is not None
                                else {}
                            ),
                        }
                        if provenance.verifier_sidecar_sha256 is not None
                        else {}
                    ),
                }
            )
            if candidate.evaluation_date:
                result["evaluation_timestamp"] = candidate.evaluation_date
            evaluation_results.append(result)
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
                    "review_tier": _weakest_review_tier(observations, authorizations).value,
                    "producer_origin_basis": _weakest_origin_basis(
                        observations, authorizations
                    ).value,
                    "origin_export_policy": policy.value,
                    **(
                        {"pipeline_git_commit": provenance.pipeline_git_commit}
                        if provenance.pipeline_git_commit
                        else {}
                    ),
                    "export_provenance_mode": provenance.mode.value,
                    "export_composition_sha256": provenance.sha256,
                    **({"doi": manifest.doi} if manifest.doi else {}),
                    **({"arxiv_id": manifest.arxiv_id} if manifest.arxiv_id else {}),
                    **(
                        {"tuple_sidecar_sha256": provenance.tuple_sidecar_sha256}
                        if provenance.mode.has_tuple_audit
                        else {"legacy_provenance_reason": provenance.legacy_reason}
                    ),
                    **(
                        {"review_manifest_sha256": provenance.review_manifest_sha256}
                        if provenance.review_manifest_sha256 is not None
                        else {}
                    ),
                    "verifier_gate_required": str(provenance.verifier_gate_required).lower(),
                    **(
                        {"verifier_sidecar_sha256": provenance.verifier_sidecar_sha256}
                        if provenance.verifier_sidecar_sha256 is not None
                        else {}
                    ),
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
