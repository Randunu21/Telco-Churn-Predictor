import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from telco_churn.config import settings
from telco_churn.data.data import build_target_encoder, clean, load_raw, split
from telco_churn.data.schema import validate_raw
from telco_churn.models.pipeline import build_pipeline

# MLflow 3.x saves sklearn models with skops by default (safer than pickle:
# loading can't run arbitrary code). skops only loads types it trusts, so every
# non-sklearn type inside our pipelines must be listed. If you add a new
# classifier or custom transformer, log_model fails loudly (and
# test_train.py with it) until the new type is added here.
SKOPS_TRUSTED_TYPES = [
    "imblearn.pipeline.Pipeline",
    "imblearn.over_sampling._smote.base.SMOTENC",
    "telco_churn.features.preprocessing.recode_no_service",
    "xgboost.sklearn.XGBClassifier",
    "xgboost.core.Booster",
    "numpy.dtype",
    "scipy.sparse._csr.csr_matrix",
]


def evaluate(model, X_test, y_test, threshold=None):
    """Compute test-set metrics. Primary selection metric: test_pr_auc."""
    if threshold is None:
        threshold = settings.decision_threshold

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    return {
        # Threshold-independent (ranking quality)
        "test_pr_auc": average_precision_score(y_test, proba),
        "test_roc_auc": roc_auc_score(y_test, proba),
        # Threshold-dependent (at settings.decision_threshold)
        "test_f1": f1_score(y_test, preds, zero_division=0),
        "test_precision": precision_score(y_test, preds, zero_division=0),
        "test_recall": recall_score(y_test, preds, zero_division=0),
    }


def build_params(pipeline):
    """Params worth recording for a run: the classifier's own hyperparameters
    plus the project-level settings that shape training."""
    classifier = pipeline.named_steps["classifier"]
    params = {
        f"classifier__{name}": value
        for name, value in classifier.get_params().items()
    }
    params.update({
        "classifier": type(classifier).__name__,
        "test_size": settings.test_size,
        "random_state": settings.random_state,
        "decision_threshold": settings.decision_threshold,
        "smote_k_neighbors": pipeline.named_steps["smote"].k_neighbors,
    })
    return params


def train_and_log(pipeline, run_name, X_train, y_train, X_test, y_test):
    """Fit the pipeline, log params + test metrics + the model to MLflow.
    Returns the run ID."""
    with mlflow.start_run(run_name=run_name) as run:
        pipeline.fit(X_train, y_train)

        mlflow.log_params(build_params(pipeline))
        mlflow.log_metrics(evaluate(pipeline, X_test, y_test))

        # The signature documents the input contract (the 14 post-clean()
        # columns and their dtypes) and the output (predict_proba's 2 columns).
        # pyfunc_predict_fn keeps the generic pyfunc flavor consistent with
        # that signature; the API will load the native sklearn flavor anyway.
        input_example = X_train.head(5)
        signature = infer_signature(input_example, pipeline.predict_proba(input_example))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    return run.info.run_id


if __name__ == "__main__":
#Purpose of writing this if __name__ == "__main__" is to ensure that the code inside 
# this block is only executed when the script is run directly, and not when it is
#  imported as a module in another script. This is a common practice in Python to allow
#  for modularity and reusability of code. If not included, the code would execute even 
# when the script is imported(If we only wanted train_and_log from this file...
# that import will run this whole file if not for this), which may not be desired behavior.
#This is called a dunder (double underscore) method in Python. 


    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    df = load_raw()
    df = validate_raw(df)
    df = clean(df)
    X_train, X_test, y_train, y_test = split(df)
    target_encoder = build_target_encoder(y_train)
    y_train = target_encoder.transform(y_train.to_numpy().reshape(-1, 1)).ravel()
    y_test = target_encoder.transform(y_test.to_numpy().reshape(-1, 1)).ravel()

    candidates = {
        "xgb": XGBClassifier(random_state=settings.random_state, eval_metric="logloss"),
        "lr": LogisticRegression(random_state=settings.random_state, max_iter=1000),
    }
    for run_name, classifier in candidates.items():
        run_id = train_and_log(
            build_pipeline(classifier), run_name, X_train, y_train, X_test, y_test
        )
        print(f"{run_name}: logged run {run_id}")