import mlflow
from sklearn.linear_model import LogisticRegression

from telco_churn.models.pipeline import build_pipeline
from telco_churn.models.train import train_and_log


def test_train_and_log_creates_a_run_and_fits_the_pipeline(pipeline_ready_data, tmp_path):
    X_train, X_test, y_train, y_test = pipeline_ready_data

    # Disposable, test-local tracking store (SQLite, matching the project's
    # real backend choice) — avoids writing into the real mlflow.db. This is
    # exactly the "MLflow side effects in tests" pitfall flagged back when
    # pytest was first set up.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_test.db")
    mlflow.set_experiment("test-experiment")

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    train_and_log(pipeline, "test-run", X_train, y_train, X_test, y_test)

    experiment = mlflow.get_experiment_by_name("test-experiment")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.mlflow.runName"] == "test-run"

    # train_and_log doesn't call autolog() itself (that's __main__'s job),
    # so this checks only what the function actually does: open a run,
    # fit the pipeline inside it. hasattr(..., "coef_") confirms it was
    # genuinely FIT, not just constructed — LogisticRegression only gets
    # this attribute after .fit() runs.
    assert hasattr(pipeline.named_steps["classifier"], "coef_")