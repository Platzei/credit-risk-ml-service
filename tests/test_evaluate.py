import pytest

from credit_risk.evaluate import (
    DEFAULT_THRESHOLDS,
    FN_COST,
    FP_COST,
    calculate_probability_metrics,
    calculate_reliability_table,
    calculate_threshold_metrics,
    select_threshold_by_cost,
)


def test_calculate_probability_metrics_returns_roc_and_pr_auc() -> None:
    y_true = [0, 0, 1, 1]
    probabilities = [0.10, 0.40, 0.35, 0.80]

    metrics = calculate_probability_metrics(y_true, probabilities)

    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    assert metrics["brier_score"] == pytest.approx(0.158125)
    assert metrics["log_loss"] == pytest.approx(0.4722879538)


def test_calculate_probability_metrics_returns_threshold_counts() -> None:
    y_true = [0, 0, 1, 1]
    probabilities = [0.10, 0.40, 0.35, 0.80]

    threshold_metrics = calculate_threshold_metrics(
        y_true,
        probabilities,
        thresholds=(0.30, 0.50),
    )

    assert list(DEFAULT_THRESHOLDS) == [0.20, 0.30, 0.40, 0.50, 0.60]
    assert threshold_metrics.to_dict(orient="records") == [
        {
            "threshold": 0.30,
            "precision": pytest.approx(2 / 3),
            "recall": 1.0,
            "f1": pytest.approx(0.8),
            "true_positives": 2,
            "false_positives": 1,
            "true_negatives": 1,
            "false_negatives": 0,
            "total_cost": 1,
        },
        {
            "threshold": 0.50,
            "precision": 1.0,
            "recall": 0.5,
            "f1": pytest.approx(2 / 3),
            "true_positives": 1,
            "false_positives": 0,
            "true_negatives": 2,
            "false_negatives": 1,
            "total_cost": 10,
        },
    ]


def test_threshold_cost_uses_configured_fp_and_fn_costs() -> None:
    threshold_metrics = calculate_threshold_metrics(
        y_true=[0, 0, 1, 1],
        probabilities=[0.10, 0.40, 0.35, 0.80],
        thresholds=(0.30, 0.50),
        fp_cost=2,
        fn_cost=7,
    )

    assert FP_COST == 1
    assert FN_COST == 10
    assert list(threshold_metrics["total_cost"]) == [2, 7]


def test_select_threshold_by_cost_returns_minimum_cost_threshold() -> None:
    selected, threshold_metrics = select_threshold_by_cost(
        y_true=[0, 0, 1, 1],
        probabilities=[0.10, 0.40, 0.35, 0.80],
        thresholds=(0.30, 0.50),
    )

    assert selected["threshold"] == pytest.approx(0.30)
    assert selected["total_cost"] == 1
    assert selected["total_cost"] == threshold_metrics["total_cost"].min()


def test_calculate_reliability_table_returns_bin_statistics() -> None:
    y_true = [0, 1, 0, 1]
    probabilities = [0.05, 0.15, 0.25, 0.95]

    reliability_table = calculate_reliability_table(
        y_true,
        probabilities,
        n_bins=2,
    )

    assert reliability_table.to_dict(orient="records") == [
        {
            "bin_lower": 0.0,
            "bin_upper": 0.5,
            "count": 3,
            "mean_predicted_probability": pytest.approx(0.15),
            "observed_positive_rate": pytest.approx(1 / 3),
        },
        {
            "bin_lower": 0.5,
            "bin_upper": 1.0,
            "count": 1,
            "mean_predicted_probability": pytest.approx(0.95),
            "observed_positive_rate": 1.0,
        },
    ]
