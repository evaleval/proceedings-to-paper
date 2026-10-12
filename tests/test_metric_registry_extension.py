"""Registry extension: averaged F1 and P/R, ranking, correlation and agreement metrics.

Every alias added here was chosen from names that actually occur in the development and
probe runs, or is an unambiguous spelling of one of them. Ambiguous names stay out on
purpose and are asserted to stay out.
"""

from __future__ import annotations

import pytest

from proceedings_to_eee.domain.observation import MetricSpec, ReportedValue
from proceedings_to_eee.resolution.metrics import (
    metric_canonical_source,
    metric_registry_resolved,
    resolve_metric,
    resolve_metric_value,
)


def _spec(raw_name: str, **fields: object) -> MetricSpec:
    return MetricSpec(raw_name=raw_name, **fields)


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("F1", "f1"),
        ("Macro F1", "macro_f1"),
        ("macro-F1", "macro_f1"),
        ("F1 (macro)", "macro_f1"),
        ("F1 scores", "f1"),
        ("F1-score (macro)", "macro_f1"),
        ("Micro F1", "micro_f1"),
        ("Weighted F1", "weighted_f1"),
        ("Macro-P", "macro_precision"),
        ("Macro Precision", "macro_precision"),
        ("macro-averaged precision", "macro_precision"),
        ("Macro-R", "macro_recall"),
        ("Macro Recall", "macro_recall"),
        ("macro-averaged recall", "macro_recall"),
        ("AUC-PR", "auprc"),
        ("PR-AUC", "auprc"),
        ("AUPRC", "auprc"),
        ("MRR", "mrr"),
        ("Mean Reciprocal Rank", "mrr"),
        ("Cosine Similarity", "cosine_similarity"),
        ("ROUGE-L", "rouge_l"),
        ("Spearman correlation", "spearman"),
        ("Spearman's rho", "spearman"),
        ("Pearson r", "pearson"),
        ("Cohen's kappa", "cohen_kappa"),
        ("Krippendorff's alpha", "krippendorff_alpha"),
        ("MCC", "mcc"),
        ("Matthews correlation coefficient", "mcc"),
        ("Concordance index", "concordance_index"),
        ("BLEU", "bleu"),
        ("ROUGE", "rouge"),
        ("Perplexity", "perplexity"),
    ],
)
def test_added_names_resolve_to_their_own_canonical_id(raw_name: str, expected: str) -> None:
    resolved = resolve_metric(_spec(raw_name))
    assert resolved.canonical_id == expected
    assert metric_registry_resolved(_spec(raw_name)) is True


@pytest.mark.parametrize(
    "raw_name",
    ["Spearman correlation", "Pearson r", "Cohen's kappa", "MCC", "Cosine Similarity"],
)
def test_correlation_metrics_keep_their_negative_range(raw_name: str) -> None:
    resolved = resolve_metric(_spec(raw_name))
    assert (resolved.min_score, resolved.max_score) == (-1.0, 1.0)
    assert resolved.lower_is_better is False


@pytest.mark.parametrize("raw_name", ["Macro-P", "Macro-R", "AUPRC", "MRR"])
def test_added_rate_metrics_take_the_bounded_rate_scale(raw_name: str) -> None:
    metric, value, _ = resolve_metric_value(
        _spec(raw_name), ReportedValue(raw="0.46", numeric=0.46)
    )
    assert metric.unit == value.unit == "proportion"
    assert (metric.min_score, metric.max_score) == (0.0, 1.0)
    assert metric.lower_is_better is False


def test_rouge_l_is_its_own_identity_without_a_scale() -> None:
    resolved = resolve_metric(_spec("ROUGE-L"))
    assert resolved.canonical_id != resolve_metric(_spec("ROUGE")).canonical_id
    assert (resolved.min_score, resolved.max_score, resolved.unit) == (None, None, None)


def test_perplexity_is_lower_is_better_and_unbounded() -> None:
    resolved = resolve_metric(_spec("Perplexity"))
    assert resolved.lower_is_better is True
    assert resolved.min_score is None and resolved.max_score is None


@pytest.mark.parametrize(
    "ambiguous",
    ["alpha", "CI", "AVG", "percentage", "Performance", "R-score", "p-value", "Avg Acc"],
)
def test_ambiguous_names_stay_unresolved(ambiguous: str) -> None:
    """These occur in real runs and each has more than one plausible meaning."""

    assert metric_registry_resolved(_spec(ambiguous)) is False


def test_canonical_source_separates_registry_from_model_and_absent() -> None:
    assert metric_canonical_source(_spec("Macro F1")) == "registry"
    assert (
        metric_canonical_source(_spec("Over-Moderation (False Positives)", canonical_id="fpr"))
        == "model"
    )
    assert metric_canonical_source(_spec("R-score")) == "none"


def test_an_unrecognised_metric_is_returned_unchanged() -> None:
    original = _spec("Over-Moderation (False Positives)", canonical_id="fpr")
    assert resolve_metric(original) == original
