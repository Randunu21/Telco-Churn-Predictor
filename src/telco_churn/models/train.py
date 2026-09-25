from telco_churn.config import settings
from telco_churn.data.data import load_raw, clean, split, build_target_encoder
from telco_churn.models.pipeline import build_pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
from telco_churn.data.schema import validate_raw


def train_and_log(pipeline, run_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)
        # score = pipeline.score(X_test, y_test)



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
    mlflow.sklearn.autolog()

    df = load_raw()
    df = validate_raw(df)
    df = clean(df)
    X_train, X_test, y_train, y_test = split(df)
    target_encoder = build_target_encoder(y_train)
    y_train = target_encoder.transform(y_train.values.reshape(-1, 1)).ravel()
    y_test = target_encoder.transform(y_test.values.reshape(-1, 1)).ravel()

    pipeline_xgb = build_pipeline(XGBClassifier(random_state=settings.random_state, eval_metric="logloss"))
    train_and_log(pipeline_xgb, "xgb", X_train, y_train, X_test, y_test)

    pipeline_lr = build_pipeline(LogisticRegression(random_state=settings.random_state, max_iter=1000))
    train_and_log(pipeline_lr, "lr", X_train, y_train, X_test, y_test)