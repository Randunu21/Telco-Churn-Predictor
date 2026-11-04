import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from telco_churn.api.inference import ModelOutputError, customers_to_frame, predict
from telco_churn.api.model_loader import LoadedModel, check_features
from telco_churn.api.schemas import FEATURE_COLUMNS, CustomerFeatures
from telco_churn.features.preprocessing import build_preprocessor, recode_transformer


class _RecordingModel:
    """Stub model: returns fixed scores and records the DataFrame it was given,
    so tests can inspect exactly what the real model would have received."""

    def __init__(self, positive_scores):
        self.positive_scores = np.asarray(positive_scores, dtype=float)
        self.received = None

    def predict_proba(self, X):
        self.received = X
        return np.column_stack([1 - self.positive_scores, self.positive_scores])


def _loaded(model, version="7", feature_columns=tuple(FEATURE_COLUMNS)):
    return LoadedModel(
        model=model,
        model_name="test-model",
        model_version=version,
        model_alias="champion",
        run_id="test-run",
        model_uri="models:/test-model@champion",
        feature_columns=feature_columns,
    )


def _customers(payload, ids):
    return [CustomerFeatures(**{**payload, "customerID": cid}) for cid in ids]


# ------------------------------------------------------------ DataFrame
def test_frame_has_exactly_feature_columns(valid_customer_payload):
    frame = customers_to_frame(_customers(valid_customer_payload, ["A"]))
    assert list(frame.columns) == FEATURE_COLUMNS
    assert "customerID" not in frame.columns


def test_frame_integer_columns_stay_integers(valid_customer_payload):
    # Training data had int64 for these; a float here would be train/serve skew.
    frame = customers_to_frame(_customers(valid_customer_payload, ["A", "B"]))
    assert frame["tenure"].dtype == "int64"
    assert frame["SeniorCitizen"].dtype == "int64"


def test_model_receives_the_frame_without_customer_id(valid_customer_payload):
    model = _RecordingModel([0.3])
    predict(_loaded(model), _customers(valid_customer_payload, ["A"]), threshold=0.5)
    assert list(model.received.columns) == FEATURE_COLUMNS


# ---------------------------------------------------------- predictions
def test_order_and_ids_preserved(valid_customer_payload):
    model = _RecordingModel([0.1, 0.9, 0.4])
    results = predict(
        _loaded(model),
        _customers(valid_customer_payload, ["A", "B", "C"]),
        threshold=0.5,
    )
    assert [r.customerID for r in results] == ["A", "B", "C"]
    assert [r.churn_score for r in results] == pytest.approx([0.1, 0.9, 0.4])
    assert [r.churn_prediction for r in results] == ["No", "Yes", "No"]


def test_score_equal_to_threshold_is_yes(valid_customer_payload):
    # Locks in >=, matching evaluate() in train.py.
    model = _RecordingModel([0.5])
    result = predict(_loaded(model), _customers(valid_customer_payload, ["A"]), 0.5)[0]
    assert result.churn_prediction == "Yes"


def test_score_just_below_threshold_is_no(valid_customer_payload):
    model = _RecordingModel([0.4999])
    result = predict(_loaded(model), _customers(valid_customer_payload, ["A"]), 0.5)[0]
    assert result.churn_prediction == "No"


def test_threshold_and_version_reported(valid_customer_payload):
    model = _RecordingModel([0.2, 0.8])
    results = predict(
        _loaded(model, version="7"),
        _customers(valid_customer_payload, ["A", "B"]),
        threshold=0.3,
    )
    assert all(r.threshold == 0.3 for r in results)
    assert all(r.model_version == "7" for r in results)


def test_n_customers_in_n_predictions_out(valid_customer_payload):
    n = 25
    model = _RecordingModel(np.linspace(0, 1, n))
    results = predict(
        _loaded(model),
        _customers(valid_customer_payload, [str(i) for i in range(n)]),
        threshold=0.5,
    )
    assert len(results) == n
    assert len(model.received) == n


@pytest.mark.parametrize("bad_score", [np.nan, np.inf])
def test_non_finite_scores_raise_model_output_error(valid_customer_payload, bad_score):
    model = _RecordingModel([0.2, bad_score, 0.7])
    with pytest.raises(ModelOutputError, match="1 non-finite"):
        predict(
            _loaded(model), _customers(valid_customer_payload, ["A", "B", "C"]), 0.5
        )


# --------------------------------------------------------- column order
def test_model_receives_columns_in_its_training_order(valid_customer_payload):
    training_order = tuple(reversed(FEATURE_COLUMNS))
    model = _RecordingModel([0.3])
    predict(
        _loaded(model, feature_columns=training_order),
        _customers(valid_customer_payload, ["A"]),
        threshold=0.5,
    )
    assert tuple(model.received.columns) == training_order
    # Matched by name: every value is still under its own column.
    for column in FEATURE_COLUMNS:
        assert model.received[column].iloc[0] == valid_customer_payload[column]


def test_real_pipeline_trained_in_another_column_order(valid_customer_payload):
    """Regression: the recode step relabels columns by position, so a model
    trained on a different column order used to get swapped values (found by
    the CI smoke test: 'unknown categories [Month-to-month]')."""
    variants = [
        {"Contract": "Month-to-month", "InternetService": "Fiber optic"},
        {"Contract": "One year", "InternetService": "DSL", "TechSupport": "Yes"},
        {"Contract": "Two year", "InternetService": "No"},
        {"Contract": "Month-to-month", "InternetService": "DSL", "Partner": "No"},
    ]
    rows = []
    for variant in variants:
        row = {**valid_customer_payload, **variant}
        if row["InternetService"] == "No":
            for col in ("OnlineSecurity", "OnlineBackup", "DeviceProtection"):
                row[col] = "No internet service"
            for col in ("TechSupport", "StreamingTV", "StreamingMovies"):
                row[col] = "No internet service"
        rows.append(row)
    training_order = list(reversed(FEATURE_COLUMNS))
    X = pd.DataFrame(rows * 3)[training_order]
    y = np.array([1, 0, 0, 1] * 3)
    # The real recode + preprocessing steps (SMOTENC is skipped at inference
    # anyway, and needs more rows than this test has).
    model = Pipeline(
        [
            ("recode", recode_transformer),
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression()),
        ]
    ).fit(X, y)

    loaded = _loaded(model, feature_columns=check_features(model))
    customer = _customers(valid_customer_payload, ["A"])
    result = predict(loaded, customer, threshold=0.5)[0]

    expected = model.predict_proba(customers_to_frame(customer)[training_order])[0, 1]
    assert result.churn_score == pytest.approx(expected)