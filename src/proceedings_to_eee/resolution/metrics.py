"""Small explicit metric registry for deterministic scale and direction resolution."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from proceedings_to_eee.domain.observation import MetricSpec, ReportedValue
from proceedings_to_eee.domain.units import canonicalize_unit
from proceedings_to_eee.io import sha256_file
from proceedings_to_eee.resources import (
    DEFAULT_EEE_METRIC_SNAPSHOT_PATH,
    EEE_METRIC_SNAPSHOT_SHA256,
)


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
        "auprc",
        "auprc",
        (
            "auprc",
            "aupr",
            "auc pr",
            "pr auc",
            "precision recall auc",
            "area under the precision recall curve",
        ),
        False,
        standard_bounded_scale=True,
    ),
    MetricDefinition(
        "f1",
        "f1",
        ("f1", "f1 score", "f1 scores"),
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
            "f1 score macro",
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
    # Macro-averaged precision and recall are distinct numbers for the same reason as
    # the F1 variants. "macro P" is unambiguous where a bare "P" is not: nobody
    # macro-averages a p-value.
    MetricDefinition(
        "macro_precision",
        "precision",
        (
            "macro precision",
            "macro p",
            "macro prec",
            "precision macro",
            "macro averaged precision",
            "macro average precision",
            "macro avg precision",
        ),
        False,
        standard_bounded_scale=True,
    ),
    MetricDefinition(
        "macro_recall",
        "recall",
        (
            "macro recall",
            "macro r",
            "macro rec",
            "recall macro",
            "macro averaged recall",
            "macro average recall",
            "macro avg recall",
        ),
        False,
        standard_bounded_scale=True,
    ),
    MetricDefinition(
        "mrr",
        "ranking",
        ("mrr", "mean reciprocal rank"),
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
        "cosine_similarity",
        "similarity",
        ("cosine similarity", "cosine sim"),
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
    # ROUGE-L is its own number, not a spelling of ROUGE, so it gets its own identity.
    MetricDefinition("rouge_l", "generation", ("rouge l", "rouge l score"), False),
    MetricDefinition("perplexity", "perplexity", ("perplexity", "ppl"), True),
)

_STANDARD_BOUNDED_UNITS = frozenset({"percent", "proportion", "probability"})

#: Notes `resolve_metric_value` returns, one per rule that can supply a unit. The
#: validation layer turns each into a field-provenance reason id, so the wording is an
#: identifier: add a note for a new rule, never reword an existing one.
PERCENT_SIGN_NOTE = "unit resolved as percent from printed percent sign"
EVIDENCE_PERCENT_NOTE = "unit resolved as percent from evidence-bound printed percent sign"
PROPORTION_NOTE = "unit resolved as proportion from registry-bounded metric and printed decimal"
UNMARKED_PERCENT_NOTE = (
    "unit resolved as percent from registry-bounded metric and unmarked value above one"
)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


#: How a metric's identity was established. `registry` is a plain alias hit;
#: `registry_concordant` is a short alias that the extractor's own canonical id agreed
#: with; `eee_registry` is a hit in the vendored EEE registry snapshot after the local
#: registry failed; `model` is a canonical id the registry never confirmed; `none` is
#: neither.
REGISTRY_SOURCE = "registry"
REGISTRY_CONCORDANT_SOURCE = "registry_concordant"
EEE_REGISTRY_SOURCE = "eee_registry"
MODEL_SOURCE = "model"
NO_SOURCE = "none"
REGISTRY_BACKED_SOURCES = frozenset(
    {REGISTRY_SOURCE, REGISTRY_CONCORDANT_SOURCE, EEE_REGISTRY_SOURCE}
)


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


@dataclass(frozen=True)
class EeeMetricSnapshot:
    """Vendored, curated slice of the EEE entity registry's metric list."""

    path: Path
    sha256: str
    retrieved_date: str
    definitions: dict[str, MetricDefinition]
    #: Normalised name -> metric id, or None when two metrics claim the same name.
    names: dict[str, str | None]

    def lookup(self, raw_name: str) -> MetricDefinition | None:
        metric_id = self.names.get(_eee_normalize(raw_name))
        return self.definitions[metric_id] if metric_id else None


def _eee_normalize(name: str) -> str:
    """Lower-case, underscores to hyphens, collapsed whitespace: the registry's own key."""

    return re.sub(r"\s+", " ", name.casefold().replace("_", "-")).strip()


_RATE_BOUNDS = frozenset({(0.0, 1.0), (0.0, 100.0)})


def _snapshot_definition(row: dict[str, object]) -> MetricDefinition | None:
    """Build a definition from one vendored metric, or None when it cannot resolve.

    A direction is required: without one the metric could never satisfy the reference
    gate, and the extractor's own direction must not be laundered through a registry hit.
    Bounds of exactly [0, 1] or [0, 100] mark a rate, which takes the same unit-dependent
    scale as the local bounded-rate metrics: the registry's choice between the two is a
    convention about how the rate is usually printed, not a fact about the page, so a win
    rate printed as 0.65 must not export against a ceiling of 100. Any other pair of
    numbers is an explicit range; a half-open or absent range resolves identity and
    direction only.
    """

    metric_id = row["id"]
    lower_is_better = row["lower_is_better"]
    if not isinstance(metric_id, str) or not isinstance(lower_is_better, bool):
        return None
    minimum, maximum = row.get("min_score"), row.get("max_score")
    if isinstance(minimum, int | float) and isinstance(maximum, int | float):
        if (float(minimum), float(maximum)) in _RATE_BOUNDS:
            return MetricDefinition(
                metric_id, metric_id, (), lower_is_better, standard_bounded_scale=True
            )
        return MetricDefinition(
            metric_id,
            metric_id,
            (),
            lower_is_better,
            min_score=float(minimum),
            max_score=float(maximum),
        )
    return MetricDefinition(metric_id, metric_id, (), lower_is_better)


@cache
def load_eee_metric_snapshot(
    path: Path = DEFAULT_EEE_METRIC_SNAPSHOT_PATH,
    expected_sha256: str | None = EEE_METRIC_SNAPSHOT_SHA256,
) -> EeeMetricSnapshot:
    """Load the vendored snapshot once, refusing a copy whose bytes drifted from the pin."""

    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(
            f"EEE metric snapshot hash mismatch: expected {expected_sha256}, got {digest}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload["_meta"]
    definitions: dict[str, MetricDefinition] = {}
    names: dict[str, set[str]] = {}
    for row in payload["metrics"]:
        definition = _snapshot_definition(row)
        if definition is None:
            continue
        definitions[definition.metric_id] = definition
        for name in (row["id"], row["display_name"]):
            names.setdefault(_eee_normalize(name), set()).add(definition.metric_id)
    for row in payload["aliases"]:
        if row["canonical_id"] in definitions:
            names.setdefault(_eee_normalize(row["raw_value"]), set()).add(row["canonical_id"])
    return EeeMetricSnapshot(
        path=path,
        sha256=digest,
        retrieved_date=str(meta["retrieved_date"]),
        definitions=definitions,
        names={key: next(iter(ids)) if len(ids) == 1 else None for key, ids in names.items()},
    )


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


def _eee_definition_for(metric: MetricSpec) -> MetricDefinition | None:
    """Fall back to the vendored EEE registry snapshot by exact normalised name."""

    return load_eee_metric_snapshot().lookup(metric.raw_name)


def _definition_for(metric: MetricSpec) -> MetricDefinition | None:
    return (
        _plain_definition_for(metric)
        or _concordant_definition_for(metric)
        or _eee_definition_for(metric)
    )


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
    registry-backed identities can still exclude it. `eee_registry` is likewise kept
    apart: the snapshot is consulted only after the local registry fails.
    """

    if _plain_definition_for(metric) is not None:
        return REGISTRY_SOURCE
    if _concordant_definition_for(metric) is not None:
        return REGISTRY_CONCORDANT_SOURCE
    if _eee_definition_for(metric) is not None:
        return EEE_REGISTRY_SOURCE
    return MODEL_SOURCE if metric.canonical_id else NO_SOURCE


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
    preserved as percent. A value in ``(1, 100]`` printed without a percent sign is read
    as a percent only for a plain local-registry rate metric, and only when every
    occurrence of the token on the page is unmarked. A token printed both with and
    without a percent sign stays unresolved: the page disagrees with itself about the
    unit, and neither reading may win.
    """

    definition = _definition_for(metric)
    if definition is None:
        return metric, value, None

    inferred_unit: str | None = None
    resolution_note: str | None = None
    if metric.unit is None and value.unit is None and definition.standard_bounded_scale:
        evidence_notation = _evidence_percent_notation(value.raw, evidence_quotes)
        if evidence_notation != "mixed":
            if "%" in value.raw:
                inferred_unit = "percent"
                resolution_note = PERCENT_SIGN_NOTE
            elif evidence_notation == "percent":
                inferred_unit = "percent"
                resolution_note = EVIDENCE_PERCENT_NOTE
            elif 0.0 <= value.numeric <= 1.0:
                inferred_unit = "proportion"
                resolution_note = PROPORTION_NOTE
            elif (
                evidence_notation == "unmarked"
                and 1.0 < value.numeric <= 100.0
                and _plain_definition_for(metric) is not None
            ):
                inferred_unit = "percent"
                resolution_note = UNMARKED_PERCENT_NOTE

    shared_unit = canonicalize_unit(metric.unit or value.unit or inferred_unit)
    resolved_metric = resolve_metric(metric, shared_unit)
    resolved_value = value
    if value.unit is None and shared_unit is not None:
        resolved_value = value.model_copy(update={"unit": shared_unit})
    return resolved_metric, resolved_value, resolution_note


def unit_inference_note(
    metric: MetricSpec,
    value: ReportedValue,
    evidence_quotes: Iterable[str] = (),
) -> str | None:
    """Re-derive which rule supplied a stored unit, from primitive inputs only.

    Returns the rule's note when the derivation, started without any provider-supplied
    metric metadata or unit, reproduces exactly the units the candidate carries; None when
    the stored unit was explicit or no rule reproduces it.
    """

    primitive_metric = metric.model_copy(
        update={
            "canonical_id": None,
            "kind": None,
            "unit": None,
            "lower_is_better": None,
            "min_score": None,
            "max_score": None,
        }
    )
    primitive_value = value.model_copy(update={"unit": None})
    inferred_metric, inferred_value, note = resolve_metric_value(
        primitive_metric, primitive_value, evidence_quotes
    )
    if note and inferred_metric.unit == metric.unit and inferred_value.unit == value.unit:
        return note
    return None


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
