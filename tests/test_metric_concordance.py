"""Short metric aliases resolve only when the extractor independently agrees.

A table header "P" is precision in one paper and a p-value in the next, and on a real
census paper "A" was an agreement percentage, not accuracy. These names are only worth
resolving when a second, independent signal names the same metric.
"""

from __future__ import annotations

import pytest

from proceedings_to_eee.domain.observation import MetricSpec, ReportedValue
from proceedings_to_eee.resolution.metrics import (
    METRICS,
    _normalize,
    metric_canonical_source,
    metric_registry_resolved,
    resolve_metric,
    resolve_metric_value,
)


def test_agree_percent_stays_unresolved_unless_the_extractor_says_accuracy() -> None:
    """The regression this rule exists for: "A" was "Agree", an agreement percentage."""

    bare = MetricSpec(raw_name="A")
    mislabelled = MetricSpec(raw_name="A", canonical_id="percentage")
    concordant = MetricSpec(raw_name="A", canonical_id="accuracy")

    assert metric_canonical_source(bare) == "none"
    assert resolve_metric(bare).canonical_id is None
    assert metric_canonical_source(mislabelled) == "model"
    assert resolve_metric(mislabelled).canonical_id == "percentage"
    assert metric_canonical_source(concordant) == "registry_concordant"
    assert resolve_metric(concordant).canonical_id == "accuracy"
    # Concordance is not a plain registry hit, and the two stay distinguishable.
    assert not metric_registry_resolved(concordant)


def test_bare_kappa_names_a_family_not_a_coefficient() -> None:
    bare = MetricSpec(raw_name="kappa")
    concordant = MetricSpec(raw_name="kappa", canonical_id="cohen_kappa")

    assert metric_canonical_source(bare) == "none"
    assert resolve_metric(bare).min_score is None
    assert metric_canonical_source(concordant) == "registry_concordant"
    resolved = resolve_metric(concordant)
    assert (resolved.min_score, resolved.max_score) == (-1.0, 1.0)


def test_an_abbreviated_precision_header_resolves_with_its_unit() -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="Prec."), ReportedValue(raw="0.785", numeric=0.785)
    )

    assert metric.canonical_id == "precision"
    assert metric.unit == "proportion"
    assert value.unit == "proportion"
    assert note is not None
    assert metric_canonical_source(MetricSpec(raw_name="Prec.")) == "registry"


def test_chrf_is_identified_without_inventing_a_unit_or_a_scale() -> None:
    resolved = resolve_metric(MetricSpec(raw_name="ChrF"))

    assert resolved.canonical_id == "chrf"
    assert resolved.unit is None
    assert resolved.min_score is None
    assert resolved.max_score is None
    # And it stays unit-free even next to a decimal, unlike a bounded-scale metric.
    metric, value, _ = resolve_metric_value(
        MetricSpec(raw_name="ChrF"), ReportedValue(raw="0.62", numeric=0.62)
    )
    assert metric.unit is None
    assert value.unit is None


def test_every_alias_is_globally_unique_after_normalisation() -> None:
    seen: dict[str, str] = {}
    for definition in METRICS:
        for alias in (*definition.aliases, *definition.short_aliases):
            normalized = _normalize(alias)
            assert normalized, f"{definition.metric_id} has an empty alias {alias!r}"
            assert normalized not in seen, (
                f"{normalized!r} is claimed by {seen[normalized]} and {definition.metric_id}"
            )
            seen[normalized] = definition.metric_id


def test_a_short_alias_never_resolves_a_different_metric() -> None:
    """Concordance means agreement, not any canonical id at all."""

    assert metric_canonical_source(MetricSpec(raw_name="P", canonical_id="precision")) == (
        "registry_concordant"
    )
    assert metric_canonical_source(MetricSpec(raw_name="P", canonical_id="recall")) == "model"
    assert metric_canonical_source(MetricSpec(raw_name="F", canonical_id="f1")) == (
        "registry_concordant"
    )


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("classification accuracy", "accuracy"),
        ("Accuracy in percentage", "accuracy"),
        ("macro averaged F1 score", "macro_f1"),
        ("Rec.", "recall"),
        ("MACE competence", "mace_competence"),
        ("JSD", "jensen_shannon_divergence"),
    ],
)
def test_plain_aliases_resolve_without_any_help_from_the_extractor(
    raw_name: str, expected: str
) -> None:
    metric = MetricSpec(raw_name=raw_name)

    assert metric_canonical_source(metric) == "registry"
    assert resolve_metric(metric).canonical_id == expected
