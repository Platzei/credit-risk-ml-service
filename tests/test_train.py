import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from credit_risk.data import TARGET_COLUMN
from credit_risk.train import (
    SELECTED_BUSINESS_THRESHOLD,
    build_catboost_model,
    build_pipeline,
    split_data,
    split_training_for_calibration,
)


def test_split_data_is_stratified_and_reproducible() -> None:
    data = pd.DataFrame(
        {
            "row_id": range(200),
            "feature": range(1000, 1200),
            TARGET_COLUMN: [0] * 160 + [1] * 40,
        }
    )

    first_split = split_data(data)
    second_split = split_data(data)
    X_train, X_validation, X_test, y_train, y_validation, y_test = first_split

    assert (len(X_train), len(X_validation), len(X_test)) == (140, 30, 30)
    assert TARGET_COLUMN not in X_train.columns
    assert y_train.mean() == y_validation.mean() == y_test.mean() == 0.2
    assert set(X_train.index).isdisjoint(X_validation.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_validation.index).isdisjoint(X_test.index)

    for first, second in zip(first_split, second_split, strict=True):
        assert first.index.equals(second.index)


def test_build_pipeline_uses_expected_steps() -> None:
    pipeline = build_pipeline()

    assert list(pipeline.named_steps) == ["imputer", "scaler", "model"]
    assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
    assert pipeline.named_steps["imputer"].strategy == "median"
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["model"], LogisticRegression)
    assert pipeline.named_steps["model"].random_state == 42


def test_build_catboost_model_uses_expected_parameters() -> None:
    model = build_catboost_model()
    parameters = model.get_params()

    assert isinstance(model, CatBoostClassifier)
    assert parameters["loss_function"] == "Logloss"
    assert parameters["eval_metric"] == "AUC"
    assert parameters["random_seed"] == 42
    assert parameters["verbose"] is False
    assert parameters["early_stopping_rounds"] == 50
    assert parameters["allow_writing_files"] is False


def test_calibration_partitions_are_disjoint_and_reproducible() -> None:
    X_train = pd.DataFrame({"row_id": range(200)})
    y_train = pd.Series([0] * 160 + [1] * 40)

    first_split = split_training_for_calibration(X_train, y_train)
    second_split = split_training_for_calibration(X_train, y_train)
    X_model, X_early_stopping, X_calibration, _, _, _ = first_split

    assert (len(X_model), len(X_early_stopping), len(X_calibration)) == (160, 20, 20)
    assert set(X_model.index).isdisjoint(X_early_stopping.index)
    assert set(X_model.index).isdisjoint(X_calibration.index)
    assert set(X_early_stopping.index).isdisjoint(X_calibration.index)

    for first, second in zip(first_split, second_split, strict=True):
        assert first.index.equals(second.index)


def test_selected_business_threshold_is_frozen_from_validation() -> None:
    assert SELECTED_BUSINESS_THRESHOLD == 0.09
