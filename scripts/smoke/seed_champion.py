"""Train a model on the synthetic data, register it and make it the champion.

CI smoke test only: gives a fresh MLflow (empty registry) the
`telco-churn@champion` model the API needs to start. Runs the real training
code (validation, cleaning, pipeline, MLflow logging) with Logistic Regression
only, then registers the run's model and points the alias at it. That is the
register -> alias -> serve path where the psycopg 3 bug showed up in Phase 5.

Configuration comes from the usual settings/env vars, set by the workflow:
  DATA_RAW_PATH        the synthetic CSV (scripts/smoke/make_data.py)
  MLFLOW_TRACKING_URI  the Compose MLflow, e.g. http://127.0.0.1:5000
"""

import mlflow
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression

from telco_churn.config import settings
from telco_churn.data.data import build_target_encoder, clean, load_raw, split
from telco_churn.data.schema import validate_raw
from telco_churn.models.pipeline import build_pipeline
from telco_churn.models.train import train_and_log

MODEL_NAME = "telco-churn"
ALIAS = "champion"


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    df = clean(validate_raw(load_raw()))
    X_train, X_test, y_train, y_test = split(df)
    encoder = build_target_encoder(y_train)
    y_train = encoder.transform(y_train.to_numpy().reshape(-1, 1)).ravel()
    y_test = encoder.transform(y_test.to_numpy().reshape(-1, 1)).ravel()

    classifier = LogisticRegression(random_state=settings.random_state, max_iter=1000)
    run_id = train_and_log(
        build_pipeline(classifier), "smoke-lr", X_train, y_train, X_test, y_test
    )

    # MLflow 3 stores the model as a "logged model" (models:/m-...) linked to
    # the run, so register that URI rather than a runs:/ artifact path.
    model_uri = mlflow.last_logged_model().model_uri
    version = mlflow.register_model(model_uri, MODEL_NAME).version
    MlflowClient().set_registered_model_alias(MODEL_NAME, ALIAS, version)
    print(f"Registered {MODEL_NAME} v{version} (run {run_id}) as @{ALIAS}")


if __name__ == "__main__":
    main()
