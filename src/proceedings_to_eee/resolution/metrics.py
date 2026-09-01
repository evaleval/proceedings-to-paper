"""Small explicit metric registry for deterministic scale and direction resolution."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from proceedings_to_eee.domain.observation import MetricSpec, ReportedValue
from proceedings_to_eee.domain.units import canonicalize_unit


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    kind: str
    aliases: tuple[str, ...]
    lower_is_better: bool
    default_unit: str | None = None
    min_score: float | None = None
    max_score: float | None = None
    standard_bounded_scale: bool = False


METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "accuracy", "accuracy", ("acc", "accuracy"), False, standard_bounded_scale=True
    ),
    MetricDefinition(
        "auroc",
        "auroc",
        ("auc", "auroc", "auc roc", "auc-roc", "roc auc", "roc-auc"),
        False,
        standard_bounded_scale=True,
    ),
    MetricDefinition(
        "f1", "f1", ("f1", "f1 score", "f1-score"), False, standard_bounded_scale=True
    ),
    MetricDefinition("precision", "precision", ("precision",), False, standard_bounded_scale=True),
    MetricDefinition(
        "recall",
        "recall",
        ("recall", "tpr", "true positive rate"),
        False,
        standard_bounded_scale=True,
    ),
    MetricDefinition(
        "fpr",
        "false_positive_rate",
        ("fpr", "false positive rate"),
        True,
        standard_bounded_scale=True,
    ),
    MetricDefinition(
        "fnr",
        "false_negative_rate",
        ("fnr", "false negative rate"),
        True,
        standard_bounded_scale=True,
    ),
    MetricDefinition(
        "error_rate",
        "error_rate",
        ("error rate",),
        True,
        standard_bounded_scale=True,
    ),
    MetricDefinition("mae", "mae", ("mae", "mean absolute error"), True),
    MetricDefinition("rmse", "rmse", ("rmse", "root mean squared error"), True),
    MetricDefinition("toxicity_score", "toxicity_score", ("toxicity", "toxicity score"), False),
)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _definition_for(metric: MetricSpec) -> MetricDefinition | None:
    normalized = _normalize(metric.raw_name)
    matches = [
        definition
        for definition in METRICS
        if normalized in {_normalize(alias) for alias in definition.aliases}
    ]
    return matches[0] if len(matches) == 1 else None


def resolve_metric(metric: MetricSpec, value_unit: str | None = None) -> MetricSpec:
    """Fill only registry-backed metric facts; never fuzzy-guess a family."""

    definition = _definition_for(metric)
    if definition is None:
        return metric
    unit = canonicalize_unit(metric.unit or value_unit or definition.default_unit)
    min_score = metric.min_score
    max_score = metric.max_score
    if min_score is None and max_score is None:
        if unit == "percent":
            min_score, max_score = 0.0, 100.0
        elif unit in {"proportion", "probability"}:
            min_score, max_score = 0.0, 1.0
        else:
            min_score, max_score = definition.min_score, definition.max_score
    return metric.model_copy(
        update={
            "canonical_id": metric.canonical_id or definition.metric_id,
            "kind": metric.kind or definition.kind,
            "unit": unit,
            "lower_is_better": (
                definition.lower_is_better
                if metric.lower_is_better is None
                else metric.lower_is_better
            ),
            "min_score": min_score,
            "max_score": max_score,
        }
    )


def resolve_metric_value(
    metric: MetricSpec,
    value: ReportedValue,
    evidence_quotes: Iterable[str] = (),
) -> tuple[MetricSpec, ReportedValue, str | None]:
    """Resolve a shared unit only when source notation and the registry make it unambiguous.

    Decimal values in ``[0, 1]`` are interpreted as proportions only for the conventional
    bounded classification metrics in this registry. A printed percent sign is likewise
    preserved as percent. Values above one without a percent sign remain unresolved.
    """

    definition = _definition_for(metric)
    if definition is None:
        return metric, value, None

    inferred_unit: str | None = None
    resolution_note: str | None = None
    if metric.unit is None and value.unit is None and definition.standard_bounded_scale:
        if "%" in value.raw:
            inferred_unit = "percent"
            resolution_note = "unit resolved as percent from printed percent sign"
        elif _evidence_prints_percent(value.raw, evidence_quotes):
            inferred_unit = "percent"
            resolution_note = "unit resolved as percent from evidence-bound printed percent sign"
        elif 0.0 <= value.numeric <= 1.0:
            inferred_unit = "proportion"
            resolution_note = (
                "unit resolved as proportion from registry-bounded metric and printed decimal"
            )

    shared_unit = canonicalize_unit(metric.unit or value.unit or inferred_unit)
    resolved_metric = resolve_metric(metric, shared_unit)
    resolved_value = value
    if value.unit is None and shared_unit is not None:
        resolved_value = value.model_copy(update={"unit": shared_unit})
    return resolved_metric, resolved_value, resolution_note


def _evidence_prints_percent(raw_value: str, evidence_quotes: Iterable[str]) -> bool:
    """Return true only when the candidate's exact raw token is printed with `%`."""

    token = re.sub(r"^[<>=~≈≤≥\s]+", "", raw_value.strip())
    token = re.sub(r"[%*†‡§\s]+$", "", token)
    if not token:
        return False
    printed_percent = re.compile(rf"(?<![\d.,]){re.escape(token)}\s*%(?![\w%])")
    return any(printed_percent.search(quote) for quote in evidence_quotes)
