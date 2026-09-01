from __future__ import annotations

import pytest

from proceedings_to_eee.domain.observation import MetricSpec, ReportedValue
from proceedings_to_eee.resolution.metrics import resolve_metric_value


@pytest.mark.parametrize("raw,numeric", [(".7314*", 0.7314), ("0.46", 0.46)])
def test_bounded_decimal_metric_resolves_to_proportion(raw: str, numeric: float) -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="AUC-ROC"), ReportedValue(raw=raw, numeric=numeric)
    )

    assert metric.canonical_id == "auroc"
    assert metric.unit == "proportion"
    assert metric.min_score == 0.0
    assert metric.max_score == 1.0
    assert value.unit == "proportion"
    assert note is not None


def test_printed_percent_is_preserved_without_rescaling() -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="Accuracy"), ReportedValue(raw="73.4%", numeric=73.4)
    )

    assert metric.unit == "percent"
    assert metric.min_score == 0.0
    assert metric.max_score == 100.0
    assert value.numeric == 73.4
    assert value.unit == "percent"
    assert note is not None


def test_unmarked_value_above_one_remains_unit_unknown() -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="Accuracy"), ReportedValue(raw="73.4", numeric=73.4)
    )

    assert metric.canonical_id == "accuracy"
    assert metric.unit is None
    assert value.unit is None
    assert note is None


def test_percent_unit_can_be_resolved_from_the_exact_evidence_token() -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="F1"),
        ReportedValue(raw="73.4", numeric=73.4),
        ["System Citrine 48.2% 73.4% 66.1%"],
    )

    assert metric.unit == "percent"
    assert metric.max_score == 100.0
    assert value.unit == "percent"
    assert note == "unit resolved as percent from evidence-bound printed percent sign"


def test_unrelated_percent_in_evidence_does_not_supply_a_unit() -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="F1"),
        ReportedValue(raw="73.4", numeric=73.4),
        ["Threshold 73.4 with comparison score 66.1%"],
    )

    assert metric.unit is None
    assert value.unit is None
    assert note is None


def test_unknown_metric_does_not_gain_a_unit_from_numeric_range() -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="Custom score"), ReportedValue(raw="0.41", numeric=0.41)
    )

    assert metric.canonical_id is None
    assert metric.unit is None
    assert value.unit is None
    assert note is None


def test_explicit_metric_unit_is_propagated_to_value() -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="F1", unit="proportion"),
        ReportedValue(raw="0.63", numeric=0.63),
    )

    assert metric.unit == "proportion"
    assert value.unit == "proportion"
    assert note is None


def test_explicit_percent_symbol_is_canonicalized_and_propagated_without_rescaling() -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="F1", unit="%"),
        ReportedValue(raw="73.4", numeric=73.4),
    )

    assert metric.unit == "percent"
    assert metric.max_score == 100.0
    assert value.unit == "percent"
    assert value.numeric == 73.4
    assert note is None
