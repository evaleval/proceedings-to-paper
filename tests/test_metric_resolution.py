from __future__ import annotations

import pytest

from proceedings_to_eee.domain.observation import MetricSpec, ReportedValue
from proceedings_to_eee.resolution.metrics import (
    EVIDENCE_PERCENT_NOTE,
    UNMARKED_PERCENT_NOTE,
    metric_unit_is_compatible,
    registry_value_range_issue,
    resolve_metric_value,
    unit_inference_note,
)


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


@pytest.mark.parametrize("raw_name", ["Mean AUC", "Subgroup AUC", "Pinned AUC"])
def test_exact_corpus_auc_variants_resolve_without_trusting_provider_metadata(
    raw_name: str,
) -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(
            raw_name=raw_name,
            canonical_id="auroc",
            kind="performance",
            lower_is_better=True,
        ),
        ReportedValue(raw="0.960", numeric=0.96),
    )

    assert metric.canonical_id == "auroc"
    assert metric.kind == "auroc"
    assert metric.lower_is_better is False
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
    assert note


def test_unrelated_percent_in_evidence_is_not_what_supplies_the_unit() -> None:
    """The percent on another token does not transfer; the unmarked-value rule decides."""

    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="F1"),
        ReportedValue(raw="73.4", numeric=73.4),
        ["Threshold 73.4 with comparison score 66.1%"],
    )

    assert note == UNMARKED_PERCENT_NOTE
    assert note != EVIDENCE_PERCENT_NOTE
    assert metric.unit == "percent"
    assert value.unit == "percent"


def test_unrelated_percent_in_evidence_does_not_supply_a_unit_above_one_hundred() -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="F1"),
        ReportedValue(raw="734", numeric=734.0),
        ["Threshold 734 with comparison score 66.1%"],
    )

    assert metric.unit is None
    assert value.unit is None
    assert note is None


@pytest.mark.parametrize(
    ("raw_name", "raw", "numeric"),
    [("Accuracy", "73.4", 73.4), ("macro F1", "1.2", 1.2), ("FPR", "100", 100.0)],
)
def test_unmarked_value_above_one_resolves_to_percent_for_a_plain_registry_rate_metric(
    raw_name: str, raw: str, numeric: float
) -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name=raw_name),
        ReportedValue(raw=raw, numeric=numeric),
        [f"Atlas {raw} 0.91 Challenger 61.2"],
    )

    assert note == UNMARKED_PERCENT_NOTE
    assert metric.unit == "percent"
    assert value.unit == "percent"
    assert (metric.min_score, metric.max_score) == (0.0, 100.0)


@pytest.mark.parametrize(
    ("metric", "raw", "numeric", "quotes"),
    [
        # The token must be found on the page; an absent token proves nothing.
        (MetricSpec(raw_name="Accuracy"), "73.4", 73.4, ["Atlas 61.2 Challenger 58.0"]),
        # Above one hundred no percent reading is possible.
        (MetricSpec(raw_name="Accuracy"), "173.4", 173.4, ["Atlas 173.4"]),
        # A short alias resolved through extractor concordance is not a plain hit.
        (MetricSpec(raw_name="P", canonical_id="precision"), "73.4", 73.4, ["Atlas 73.4"]),
        # The vendored EEE registry supplies identity, never this inference.
        (MetricSpec(raw_name="Exact Match"), "73.4", 73.4, ["Atlas 73.4"]),
        # Identity-only metrics have no rate scale to read a percent against.
        (MetricSpec(raw_name="BLEU"), "27.3", 27.3, ["Atlas 27.3"]),
    ],
)
def test_unmarked_percent_rule_stays_narrow(
    metric: MetricSpec, raw: str, numeric: float, quotes: list[str]
) -> None:
    resolved, value, note = resolve_metric_value(
        metric, ReportedValue(raw=raw, numeric=numeric), quotes
    )

    assert resolved.unit is None
    assert value.unit is None
    assert note is None


def test_mixed_notation_leaves_a_printed_percent_sign_unresolved() -> None:
    """The same token appears with and without a sign, so the page contradicts itself."""

    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="Accuracy"),
        ReportedValue(raw="73.4%", numeric=73.4),
        ["Atlas accuracy 73.4%", "baseline threshold 73.4"],
    )

    assert metric.unit is None
    assert value.unit is None
    assert (metric.min_score, metric.max_score) == (None, None)
    assert note is None


def test_unit_inference_note_names_the_rule_that_reproduces_a_stored_unit() -> None:
    quotes = ["Atlas 73.4 Challenger 61.2"]
    metric, value, _ = resolve_metric_value(
        MetricSpec(raw_name="Accuracy"), ReportedValue(raw="73.4", numeric=73.4), quotes
    )

    assert unit_inference_note(metric, value, quotes) == UNMARKED_PERCENT_NOTE
    # An explicit unit that no rule reproduces has no inference note.
    explicit_metric = metric.model_copy(update={"unit": "proportion"})
    explicit_value = value.model_copy(update={"unit": "proportion"})
    assert unit_inference_note(explicit_metric, explicit_value, quotes) is None


def test_mixed_identical_raw_notation_does_not_supply_percent_unit() -> None:
    metric, value, note = resolve_metric_value(
        MetricSpec(raw_name="Accuracy"),
        ReportedValue(raw="0.76", numeric=0.76),
        ["Atlas Accuracy 0.76; baseline 0.76%"],
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


@pytest.mark.parametrize(
    "proposed_kind",
    ["area_under_curve", "performance", "synthetic_bias_metric"],
)
def test_registry_metric_kind_overrides_free_form_provider_kind(proposed_kind: str) -> None:
    metric, _, _ = resolve_metric_value(
        MetricSpec(
            raw_name="AUC-ROC",
            canonical_id="provider_invented_auc",
            kind=proposed_kind,
        ),
        ReportedValue(raw="0.81", numeric=0.81),
    )

    assert metric.canonical_id == "auroc"
    assert metric.kind == "auroc"


def test_registry_metric_overrides_conflicting_direction_and_bounded_range() -> None:
    metric, _, _ = resolve_metric_value(
        MetricSpec(
            raw_name="F1",
            lower_is_better=True,
            unit="proportion",
            min_score=-1,
            max_score=7,
        ),
        ReportedValue(raw="0.43", numeric=0.43),
    )

    assert metric.lower_is_better is False
    assert metric.min_score == 0.0
    assert metric.max_score == 1.0


@pytest.mark.parametrize("unit", ["percent", "proportion", "probability"])
def test_registry_bounded_metric_accepts_only_supported_rate_units(unit: str) -> None:
    metric, value, _ = resolve_metric_value(
        MetricSpec(raw_name="Accuracy", unit=unit),
        ReportedValue(raw="0.74", numeric=0.74, unit=unit),
    )

    assert metric_unit_is_compatible(metric, metric.unit)
    assert value.unit == unit


def test_registry_bounded_metric_rejects_dimensional_unit() -> None:
    metric, value, _ = resolve_metric_value(
        MetricSpec(raw_name="Accuracy", unit="seconds"),
        ReportedValue(raw="74.6", numeric=74.6, unit="seconds"),
    )

    assert metric.canonical_id == "accuracy"
    assert metric.unit == "seconds"
    assert value.unit == "seconds"
    assert not metric_unit_is_compatible(metric, metric.unit)


@pytest.mark.parametrize(
    ("raw", "numeric", "unit"),
    [("146%", 146.0, "percent"), ("1.46", 1.46, "proportion")],
)
def test_registry_bounded_metric_rejects_out_of_range_value(
    raw: str,
    numeric: float,
    unit: str,
) -> None:
    metric, value, _ = resolve_metric_value(
        MetricSpec(raw_name="Accuracy", unit=unit),
        ReportedValue(raw=raw, numeric=numeric, unit=unit),
    )

    assert registry_value_range_issue(metric, value) == ("value_outside_registry_metric_bounds")


def test_unknown_metric_honors_explicit_range() -> None:
    metric, value, _ = resolve_metric_value(
        MetricSpec(raw_name="Custom bounded score", min_score=0, max_score=1),
        ReportedValue(raw="2", numeric=2),
    )

    assert metric.canonical_id is None
    assert registry_value_range_issue(metric, value) == ("value_outside_registry_metric_bounds")
