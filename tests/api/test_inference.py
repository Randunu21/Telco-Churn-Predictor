import numpy as np
import pytest

from telco_churn.api.inference import ModelOutputError, customers_to_frame, predict
from telco_churn.api.model_loader import LoadedModel
from telco_churn.api.schemas import FEATURE_COLUMNS, CustomerFeatures


class _RecordingModel:
    """Stub model: returns fixed scores and records the DataFrame it was given,
    so tests can inspect exactly what the real model would have received."""

    def __init__(self, positive_scores):
        self.positive_scores = np.asarray(positive_scores, dtype=float)
        self.received = None

    def predict_proba(self, X):
        self.received = X
        return np.column_stack([1 - self.positive_scores, self.positive_scores])


def _loaded(model, version="7"):
    return LoadedModel(
        model=model,
        model_name="test-model",
        model_version=version,
        model_alias="champion",
        run_id="test-run",
        model_uri="models:/test-model@champion",
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
        predict(_loaded(model), _customers(valid_customer_payload, ["A", "B", "C"]), 0.5)