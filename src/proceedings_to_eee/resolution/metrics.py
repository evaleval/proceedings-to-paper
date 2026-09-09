"""Small explicit metric registry for deterministic scale and direction resolution."""

from __future__ import annotations

import math
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
    #: Names too short or too common to identify a metric on their own. A table header
    #: "P" can mean precision or a p-value, and "A" can mean agreement or accuracy.
    #: These resolve only when the alias and extractor-proposed canonical id agree;
    #: ambiguous wording alone cannot grant the registry's scale.
    short_aliases: tuple[str, ...] = ()


METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "accuracy",
        "accuracy",
        (
            "acc",
            "accuracy",
            "classification accuracy",
            "accuracy in percentage",
            "accuracy score",
        ),
        False,
        standard_bounded_scale=True,
        short_aliases=("a",),
    ),
    MetricDefinition(
        "auroc",
        "auroc",
        (
            "auc",
            "auroc",
            "auc roc",
            "roc auc",
            "mean auc",
            "subgroup auc",
            "pinned auc",
        ),
        False,
        standard_bounded_scale=True,
    ),
    MetricDefinition(
        "f1",
        "f1",
        ("f1", "f1 score"),
        False,
        standard_bounded_scale=True,
        short_aliases=("f", "f score", "f measure", "f1 measure"),
    ),
    MetricDefinition(
        "precision",
        "precision",
        ("precision", "prec", "precision score"),
        False,
        standard_bounded_scale=True,
        short_aliases=("p",),
    ),
    MetricDefinition(
        "recall",
        "recall",
        ("recall", "rec", "recall score", "tpr", "true positive rate"),
        False,
        standard_bounded_scale=True,
        short_aliases=("r",),
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
    # Averaged F1 variants are genuinely different numbers and must not pool with each
    # other or with plain F1, so each takes its own canonical id rather than an alias.
    MetricDefinition(
        "macro_f1",
        "f1",
        (
            "macro f1",
            "macro f1 score",
            "f1 macro",
            "macro averaged f1",
            "macro averaged f1 score",
            "macro average f1",
            "macro avg f1",
        ),
        False,
        standard_bounded_scale=True,
    ),
    MetricDefinition(
        "micro_f1",
        "f1",
        ("micro f1", "micro f1 score", "f1 micro", "micro averaged f1", "micro avg f1"),
        False,
        standard_bounded_scale=True,
    ),
    MetricDefinition(
        "weighted_f1",
        "f1",
        ("weighted f1", "weighted f1 score", "f1 weighted", "weighted averaged f1"),
        False,
        standard_bounded_scale=True,
    ),
    # Correlation and agreement statistics run from -1 to 1, so they must not use the
    # bounded-rate scale that reads a bare decimal as a 0-to-1 proportion.
    MetricDefinition(
        "spearman",
        "correlation",
        ("spearman", "spearman correlation", "spearman rho", "spearman s rho"),
        False,
        min_score=-1.0,
        max_score=1.0,
    ),
    MetricDefinition(
        "pearson",
        "correlation",
        ("pearson", "pearson correlation", "pearson r", "pearson s r"),
        False,
        min_score=-1.0,
        max_score=1.0,
    ),
    MetricDefinition(
        "cohen_kappa",
        "agreement",
        ("cohen kappa", "cohen s kappa", "cohens kappa"),
        False,
        min_score=-1.0,
        max_score=1.0,
        # Bare "kappa" names a family, not a coefficient: Cohen, Fleiss and Light all
        # print as kappa, and only the extractor's own canonical id can pick one.
        short_aliases=("kappa",),
    ),
    MetricDefinition(
        "krippendorff_alpha",
        "agreement",
        ("krippendorff alpha", "krippendorff s alpha", "krippendorffs alpha", "krippendorff"),
        False,
        min_score=-1.0,
        max_score=1.0,
    ),
    MetricDefinition(
        "mcc",
        "correlation",
        ("mcc", "matthews correlation coefficient", "matthews correlation"),
        False,
        min_score=-1.0,
        max_score=1.0,
    ),
    MetricDefinition(
        "concordance_index",
        "auroc",
        ("concordance index", "c index", "harrell s c index"),
        False,
        standard_bounded_scale=True,
    ),
    # Bounded 0 to 1, so it takes the standard bounded scale rather than an explicit
    # range: a printed decimal is a proportion and a printed percent is a percent, the
    # same reading the rate metrics get. The identity-only entries below are the
    # contrast, where no unit or scale may be inferred at all.
    MetricDefinition(
        "mace_competence",
        "agreement",
        ("mace competence", "mace", "annotator competence"),
        False,
        standard_bounded_scale=True,
    ),
    # Identity only. Both have conventional ranges that vary by implementation, so the
    # registry names them and stops there rather than inventing a unit or a scale.
    MetricDefinition("chrf", "generation", ("chrf", "chrf score"), False),
    MetricDefinition(
        "jensen_shannon_divergence",
        "divergence",
        ("jensen shannon divergence", "jensen shannon", "js divergence", "jsd"),
        True,
    ),
    MetricDefinition("bleu", "generation", ("bleu", "bleu score"), False),
    MetricDefinition("rouge", "generation", ("rouge", "rouge score"), False),
    MetricDefinition("perplexity", "perplexity", ("perplexity", "ppl"), True),
)

_STANDARD_BOUNDED_UNITS = frozenset({"percent", "proportion", "probability"})


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


#: How a metric's identity was established. `registry` is a plain alias hit;
#: `registry_concordant` is a short alias that the extractor's own canonical id agreed
#: with; `model` is a canonical id the registry never confirmed; `none` is neither.
REGISTRY_SOURCE = "registry"
REGISTRY_CONCORDANT_SOURCE = "registry_concordant"


def _validate_alias_uniqueness() -> None:
    """Refuse an ambiguous registry at import rather than resolving one silently.

    Two definitions sharing a normalised alias would make `_definition_for` return None
    for that name, which reads as "unknown metric" and hides a registry bug.
    """

    seen: dict[str, str] = {}
    for definition in METRICS:
        for alias in (*definition.aliases, *definition.short_aliases):
            normalized = _normalize(alias)
            if not normalized:
                raise ValueError(f"metric {definition.metric_id} has an empty alias {alias!r}")
            if normalized in seen:
                raise ValueError(
                    f"alias {normalized!r} is claimed by both {seen[normalized]} "
                    f"and {definition.metric_id}"
                )
            seen[normalized] = definition.metric_id


_validate_alias_uniqueness()


def _plain_definition_for(metric: MetricSpec) -> MetricDefinition | None:
    normalized = _normalize(metric.raw_name)
    matches = [
        definition
        for definition in METRICS
        if normalized in {_normalize(alias) for alias in definition.aliases}
    ]
    return matches[0] if len(matches) == 1 else None


def _concordant_definition_for(metric: MetricSpec) -> MetricDefinition | None:
    """Resolve a short alias only when the extractor independently named the same metric."""

    if not metric.canonical_id:
        return None
    normalized = _normalize(metric.raw_name)
    matches = [
        definition
        for definition in METRICS
        if normalized in {_normalize(alias) for alias in definition.short_aliases}
        and metric.canonical_id == definition.metric_id
    ]
    return matches[0] if len(matches) == 1 else None


def _definition_for(metric: MetricSpec) -> MetricDefinition | None:
    return _plain_definition_for(metric) or _concordant_definition_for(metric)


def metric_registry_resolved(metric: MetricSpec) -> bool:
    """Report whether the registry, rather than a model, backs this metric's identity.

    `resolve_metric` returns an unrecognised metric unchanged, so a model-proposed
    `canonical_id` survives with nothing to distinguish it from a registry-backed one.
    Consumers that aggregate across papers need that distinction to avoid treating an
    unconfirmed or paper-specific metric as a known registry entry.
    """

    return _plain_definition_for(metric) is not None


def metric_canonical_source(metric: MetricSpec) -> str:
    """Classify where a metric's canonical identity came from.

    `registry_concordant` is deliberately its own value rather than folded into
    `registry`: it took the extractor's agreement to resolve, so a consumer counting
    registry-backed identities can still exclude it.
    """

    if _plain_definition_for(metric) is not None:
        return REGISTRY_SOURCE
    if _concordant_definition_for(metric) is not None:
        return REGISTRY_CONCORDANT_SOURCE
    return "model" if metric.canonical_id else "none"


def metric_unit_is_compatible(metric: MetricSpec, unit: str | None = None) -> bool:
    """Reject dimensional units for registry-defined bounded rate metrics."""

    definition = _definition_for(metric)
    resolved_unit = canonicalize_unit(unit if unit is not None else metric.unit)
    return (
        definition is None
        or not definition.standard_bounded_scale
        or resolved_unit is None
        or resolved_unit in _STANDARD_BOUNDED_UNITS
    )


def registry_value_range_issue(metric: MetricSpec, value: ReportedValue) -> str | None:
    """Return a deterministic reason when an explicit/resolved range is violated."""

    if not math.isfinite(value.numeric):
        return "value_outside_registry_metric_bounds"
    if metric.min_score is not None and value.numeric < metric.min_score:
        return "value_outside_registry_metric_bounds"
    if metric.max_score is not None and value.numeric > metric.max_score:
        return "value_outside_registry_metric_bounds"
    uncertainty = value.uncertainty
    if uncertainty is None:
        return None
    lower = uncertainty.confidence_interval_lower
    upper = uncertainty.confidence_interval_upper
    if lower is None and upper is None:
        return None
    if (
        lower is None
        or upper is None
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower > upper
        or (metric.min_score is not None and lower < metric.min_score)
        or (metric.max_score is not None and upper > metric.max_score)
        or not lower <= value.numeric <= upper
    ):
        return "uncertainty_interval_outside_registry_metric_bounds"
    return None


def resolve_metric(metric: MetricSpec, value_unit: str | None = None) -> MetricSpec:
    """Fill only registry-backed metric facts; never fuzzy-guess a family."""

    definition = _definition_for(metric)
    if definition is None:
        return metric
    unit = canonicalize_unit(metric.unit or value_unit or definition.default_unit)
    if definition.standard_bounded_scale and unit == "percent":
        min_score, max_score = 0.0, 100.0
    elif definition.standard_bounded_scale and unit in {"proportion", "probability"}:
        min_score, max_score = 0.0, 1.0
    else:
        min_score, max_score = definition.min_score, definition.max_score
    return metric.model_copy(
        update={
            "canonical_id": definition.metric_id,
            # ``kind`` is a registry-backed semantic fact, not a free-form
            # provider label.  Once a metric name resolves unambiguously, keep
            # its kind aligned with the canonical metric definition.
            "kind": definition.kind,
            "unit": unit,
            "lower_is_better": definition.lower_is_better,
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
        evidence_notation = _evidence_percent_notation(value.raw, evidence_quotes)
        if "%" in value.raw:
            inferred_unit = "percent"
            resolution_note = "unit resolved as percent from printed percent sign"
        elif evidence_notation == "percent":
            inferred_unit = "percent"
            resolution_note = "unit resolved as percent from evidence-bound printed percent sign"
        elif evidence_notation != "mixed" and 0.0 <= value.numeric <= 1.0:
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


def _evidence_percent_notation(raw_value: str, evidence_quotes: Iterable[str]) -> str:
    """Classify percent notation across every exact occurrence of one raw token."""

    token = re.sub(r"^[<>=~≈≤≥\s]+", "", raw_value.strip())
    token = re.sub(r"[%*†‡§\s]+$", "", token)
    if not token:
        return "absent"
    occurrence = re.compile(rf"(?<![\d.,]){re.escape(token)}(?P<percent>\s*%)?(?![\w%])")
    notations = [
        match.group("percent") is not None
        for quote in evidence_quotes
        for match in occurrence.finditer(quote)
    ]
    if not notations:
        return "absent"
    if all(notations):
        return "percent"
    if any(notations):
        return "mixed"
    return "unmarked"
