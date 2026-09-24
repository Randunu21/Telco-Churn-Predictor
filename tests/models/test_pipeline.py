import numpy as np
from sklearn.linear_model import LogisticRegression

from telco_churn.models.pipeline import build_pipeline


def test_build_pipeline_returns_fresh_instances_each_call():
    # Regression guard for the shared-mutable-state issue from the original
    # notebook (pipeline_lr / pipeline_xgb sharing one preprocessor/smote
    # object). Two calls must produce genuinely independent instances.
    pipeline_a = build_pipeline(LogisticRegression())
    pipeline_b = build_pipeline(LogisticRegression())

    assert pipeline_a.named_steps["preprocessor"] is not pipeline_b.named_steps["preprocessor"]
    assert pipeline_a.named_steps["smote"] is not pipeline_b.named_steps["smote"]


def test_pipeline_fits_and_predicts_without_error(pipeline_ready_data):
    X_train, X_test, y_train, y_test = pipeline_ready_data
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    assert len(preds) == len(X_test)


def test_pipeline_predict_returns_one_row_per_input_row_not_resampled_count(pipeline_ready_data):
    # SMOTENC changes row count DURING fit; imblearn's Pipeline auto-skips
    # resampling at predict time. This pins that behavior down explicitly
    # rather than just trusting it silently.
    X_train, X_test, y_train, y_test = pipeline_ready_data
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    assert len(preds) == len(X_test)  # NOT the SMOTE-resampled training size


def test_pipeline_predict_proba_is_valid_probability(pipeline_ready_data):
    X_train, X_test, y_train, y_test = pipeline_ready_data
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)

    assert proba.shape == (len(X_test), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_pipeline_predictions_contain_no_nan(pipeline_ready_data):
    X_train, X_test, y_train, y_test = pipeline_ready_data
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)

    assert not np.isnan(proba).any()


def test_pipeline_does_not_error_on_category_unseen_in_training(pipeline_ready_data):
    # Regression guard for the ORIGINAL leakage bug: encoders used to be fit
    # on train AND test combined, so "unseen category" problems were
    # invisible in development. A correctly train-only-fit pipeline should
    # handle a never-before-seen category gracefully (via
    # handle_unknown="ignore"), not raise.
    X_train, X_test, y_train, y_test = pipeline_ready_data
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    X_test_new = X_test.copy()
    X_test_new.iloc[0, X_test_new.columns.get_loc("PaymentMethod")] = "Some Brand New Payment Method"

    pipeline.predict(X_test_new)  # should not raise