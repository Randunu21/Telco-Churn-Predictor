import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from telco_churn.models.pipeline import build_pipeline
from telco_churn.models.train import evaluate, train_and_log

EXPECTED_METRICS = {"test_pr_auc", "test_roc_auc", "test_f1", "test_precision", "test_recall"}


class _FixedProbaModel:
    """Stand-in model with known outputs, so metric values can be asserted exactly."""

    def __init__(self, positive_proba):
        self.positive_proba = np.asarray(positive_proba, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.positive_proba, self.positive_proba])


def test_evaluate_perfect_model_scores_one():
    y = np.array([0, 0, 1, 1])
    model = _FixedProbaModel([0.1, 0.2, 0.8, 0.9])

    metrics = evaluate(model, X_test=None, y_test=y, threshold=0.5)

    assert set(metrics) == EXPECTED_METRICS
    assert all(value == pytest.approx(1.0) for value in metrics.values())


def test_evaluate_threshold_changes_only_threshold_dependent_metrics():
    y = np.array([0, 0, 1, 1])
    model = _FixedProbaModel([0.1, 0.4, 0.45, 0.9])

    at_half = evaluate(model, None, y, threshold=0.5)
    at_low = evaluate(model, None, y, threshold=0.3)

    # Ranking metrics don't depend on the threshold...
    assert at_half["test_pr_auc"] == at_low["test_pr_auc"]
    assert at_half["test_roc_auc"] == at_low["test_roc_auc"]
    # ...but recall does: at 0.5 one churner (0.45) is missed; at 0.3 neither is.
    assert at_half["test_recall"] == pytest.approx(0.5)
    assert at_low["test_recall"] == pytest.approx(1.0)


def test_train_and_log_logs_metrics_params_and_a_loadable_model(pipeline_ready_data, tmp_path):
    X_train, X_test, y_train, y_test = pipeline_ready_data

    # Disposable, test-local tracking store (SQLite, matching the real backend).
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_test.db")
    mlflow.set_experiment("test-experiment")

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    run_id = train_and_log(pipeline, "test-run", X_train, y_train, X_test, y_test)

    run = mlflow.get_run(run_id)
    assert run.info.run_name == "test-run"

    # Test-set metrics, all present and in [0, 1]
    assert EXPECTED_METRICS <= set(run.data.metrics)
    assert all(0.0 <= run.data.metrics[m] <= 1.0 for m in EXPECTED_METRICS)

    assert run.data.params["classifier"] == "LogisticRegression"

    # Round-trip: the logged model loads back (proving the pipeline, including
    # recode_no_service, unpickles from its importable path) and predicts the
    # same probabilities as the in-memory pipeline. This is exactly what the
    # API will do in Phase 4.
    loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    np.testing.assert_allclose(
        loaded.predict_proba(X_test), pipeline.predict_proba(X_test)
    )


def test_xgb_pipeline_also_logs_and_reloads(pipeline_ready_data, tmp_path):
    # XGBoost adds its own types to the skops trusted list; this guards
    # against that list drifting out of sync with the candidates in train.py.
    from xgboost import XGBClassifier

    X_train, X_test, y_train, y_test = pipeline_ready_data
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_test.db")
    mlflow.set_experiment("test-experiment")

    pipeline = build_pipeline(XGBClassifier(eval_metric="logloss"))
    run_id = train_and_log(pipeline, "xgb-run", X_train, y_train, X_test, y_test)

    loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    np.testing.assert_allclose(
        loaded.predict_proba(X_test), pipeline.predict_proba(X_test)
    )