import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data import TARGET_COLUMN, load_data
from credit_risk.evaluate import (
    FN_COST,
    FP_COST,
    calculate_probability_metrics,
    calculate_reliability_table,
    calculate_threshold_metrics,
    select_threshold_by_cost,
)

RANDOM_STATE = 42
SELECTED_BUSINESS_THRESHOLD = 0.09


def split_data(data: pd.DataFrame) -> tuple:
    """Split data into stratified 70% train, 15% validation, and 15% test sets."""
    features = data.drop(columns=TARGET_COLUMN)
    target = data[TARGET_COLUMN]

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_pipeline() -> Pipeline:
    """Build the logistic regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def build_catboost_model() -> CatBoostClassifier:
    """Build the CatBoost baseline model."""
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=50,
        allow_writing_files=False,
    )


def split_training_for_calibration(X_train, y_train) -> tuple:
    """Create disjoint model-fit, early-stopping, and calibration partitions."""
    X_model_train, X_remaining, y_model_train, y_remaining = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    X_early_stopping, X_calibration, y_early_stopping, y_calibration = (
        train_test_split(
            X_remaining,
            y_remaining,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y_remaining,
        )
    )

    return (
        X_model_train,
        X_early_stopping,
        X_calibration,
        y_model_train,
        y_early_stopping,
        y_calibration,
    )


def main() -> None:
    data = load_data()
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(data)

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_validation.shape}")
    print(f"Train positive class rate: {y_train.mean():.4f}")
    print(f"Validation positive class rate: {y_validation.mean():.4f}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    logistic_validation_scores = pipeline.predict_proba(X_validation)[:, 1]
    logistic_validation_roc_auc = roc_auc_score(
        y_validation, logistic_validation_scores
    )
    logistic_validation_pr_auc = average_precision_score(
        y_validation, logistic_validation_scores
    )

    (
        X_model_train,
        X_early_stopping,
        X_calibration,
        y_model_train,
        y_early_stopping,
        y_calibration,
    ) = split_training_for_calibration(X_train, y_train)

    catboost_model = build_catboost_model()
    catboost_model.fit(
        X_model_train,
        y_model_train,
        eval_set=(X_early_stopping, y_early_stopping),
    )
    catboost_validation_scores = catboost_model.predict_proba(X_validation)[:, 1]
    catboost_metrics = calculate_probability_metrics(
        y_validation, catboost_validation_scores
    )
    threshold_metrics = calculate_threshold_metrics(
        y_validation, catboost_validation_scores
    )
    selected_threshold, cost_threshold_metrics = select_threshold_by_cost(
        y_validation, catboost_validation_scores
    )
    threshold_05 = cost_threshold_metrics.loc[
        cost_threshold_metrics["threshold"].eq(0.50)
    ].iloc[0]

    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(catboost_model),
        method="sigmoid",
    )
    calibrated_model.fit(X_calibration, y_calibration)
    calibrated_validation_scores = calibrated_model.predict_proba(X_validation)[:, 1]
    calibrated_metrics = calculate_probability_metrics(
        y_validation, calibrated_validation_scores
    )

    print(f"Logistic Regression validation ROC-AUC: {logistic_validation_roc_auc:.4f}")
    print(f"Logistic Regression validation PR-AUC: {logistic_validation_pr_auc:.4f}")
    probability_comparison = pd.DataFrame(
        [
            {"probabilities": "Uncalibrated", **catboost_metrics},
            {"probabilities": "Sigmoid calibrated", **calibrated_metrics},
        ]
    )
    print("\nCatBoost validation probability metrics:")
    print(
        probability_comparison.to_string(
            index=False,
            formatters={
                "roc_auc": "{:.4f}".format,
                "pr_auc": "{:.4f}".format,
                "brier_score": "{:.4f}".format,
                "log_loss": "{:.4f}".format,
            },
        )
    )

    best_iteration = getattr(catboost_model, "best_iteration_", None)
    if best_iteration is not None:
        print(f"CatBoost best_iteration_: {best_iteration}")

    print("\nCatBoost validation threshold analysis:")
    print(
        threshold_metrics.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
            },
        )
    )

    print(
        "\nIllustrative validation cost analysis "
        f"(FP={FP_COST}, FN={FN_COST} cost units; not real bank economics):"
    )
    print(f"Selected threshold: {selected_threshold['threshold']:.2f}")
    print(f"Total illustrative cost: {selected_threshold['total_cost']:.0f}")
    print(f"Precision: {selected_threshold['precision']:.4f}")
    print(f"Recall: {selected_threshold['recall']:.4f}")
    print(f"F1: {selected_threshold['f1']:.4f}")
    print(
        "TP / FP / TN / FN: "
        f"{int(selected_threshold['true_positives'])} / "
        f"{int(selected_threshold['false_positives'])} / "
        f"{int(selected_threshold['true_negatives'])} / "
        f"{int(selected_threshold['false_negatives'])}"
    )
    print(f"Threshold 0.50 total illustrative cost: {threshold_05['total_cost']:.0f}")
    cost_difference = threshold_05["total_cost"] - selected_threshold["total_cost"]
    print(f"Illustrative cost reduction versus 0.50: {cost_difference:.0f}")

    for label, probabilities in (
        ("Uncalibrated", catboost_validation_scores),
        ("Sigmoid calibrated", calibrated_validation_scores),
    ):
        reliability_table = calculate_reliability_table(
            y_validation,
            probabilities,
        )
        print(f"\n{label} validation reliability table:")
        print(
            reliability_table.to_string(
                index=False,
                formatters={
                    "bin_lower": "{:.1f}".format,
                    "bin_upper": "{:.1f}".format,
                    "mean_predicted_probability": "{:.4f}".format,
                    "observed_positive_rate": "{:.4f}".format,
                },
            )
        )

    test_probabilities = catboost_model.predict_proba(X_test)[:, 1]
    test_probability_metrics = calculate_probability_metrics(
        y_test,
        test_probabilities,
    )
    test_threshold_metrics = calculate_threshold_metrics(
        y_test,
        test_probabilities,
        thresholds=(SELECTED_BUSINESS_THRESHOLD, 0.50),
        fp_cost=FP_COST,
        fn_cost=FN_COST,
    )

    print("\nFinal held-out test evaluation:")
    print(
        f"Threshold {SELECTED_BUSINESS_THRESHOLD:.2f} was selected on validation, "
        "not on test."
    )
    print("Test results must not be used for tuning, recalibration, or reselection.")
    print(f"ROC-AUC: {test_probability_metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {test_probability_metrics['pr_auc']:.4f}")
    print(f"Brier score: {test_probability_metrics['brier_score']:.4f}")
    print(f"Log loss: {test_probability_metrics['log_loss']:.4f}")
    print(
        test_threshold_metrics.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
