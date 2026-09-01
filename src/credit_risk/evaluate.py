import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

DEFAULT_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60)
FP_COST = 1
FN_COST = 10
COST_THRESHOLDS = tuple(value / 100 for value in range(1, 100))


def calculate_probability_metrics(
    y_true,
    probabilities,
) -> dict[str, float]:
    """Calculate probability metrics, treating class 1 as high risk."""
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
    }


def calculate_threshold_metrics(
    y_true,
    probabilities,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> pd.DataFrame:
    """Calculate classification metrics and illustrative cost by threshold."""
    threshold_rows = []

    for threshold in thresholds:
        predictions = [int(probability >= threshold) for probability in probabilities]
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": int(true_positive),
                "false_positives": int(false_positive),
                "true_negatives": int(true_negative),
                "false_negatives": int(false_negative),
                "total_cost": false_positive * fp_cost + false_negative * fn_cost,
            }
        )

    return pd.DataFrame(threshold_rows)


def select_threshold_by_cost(
    y_true,
    probabilities,
    thresholds: tuple[float, ...] = COST_THRESHOLDS,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> tuple[pd.Series, pd.DataFrame]:
    """Select the first validation threshold with minimum illustrative cost."""
    threshold_metrics = calculate_threshold_metrics(
        y_true,
        probabilities,
        thresholds=thresholds,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
    selected = threshold_metrics.loc[threshold_metrics["total_cost"].idxmin()]
    return selected, threshold_metrics


def calculate_reliability_table(
    y_true,
    probabilities,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize predicted and observed risk within equal-width probability bins."""
    reliability_data = pd.DataFrame(
        {
            "observed": list(y_true),
            "probability": list(probabilities),
        }
    )
    reliability_data["bin"] = (
        (reliability_data["probability"] * n_bins)
        .astype(int)
        .clip(lower=0, upper=n_bins - 1)
    )

    reliability_table = reliability_data.groupby("bin", as_index=False).agg(
        count=("observed", "size"),
        mean_predicted_probability=("probability", "mean"),
        observed_positive_rate=("observed", "mean"),
    )
    reliability_table["bin_lower"] = reliability_table["bin"] / n_bins
    reliability_table["bin_upper"] = (reliability_table["bin"] + 1) / n_bins

    return reliability_table[
        [
            "bin_lower",
            "bin_upper",
            "count",
            "mean_predicted_probability",
            "observed_positive_rate",
        ]
    ]
