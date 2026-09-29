import numpy as np
import pandas as pd

from telco_churn.config import settings
from telco_churn.features.preprocessing import build_smote, recode_no_service


def _smote_ready_frame(n=1000, seed=0):
    """Post-recode shaped data: every categorical is a plain string category,
    SeniorCitizen is 0/1, and seniors are deliberately 50% of churners so any
    distortion in synthetic samples is easy to detect."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "SeniorCitizen": rng.choice([0, 1], size=n),
        "tenure": rng.integers(1, 72, size=n),
    })
    for col in settings.ohe_cols:
        X[col] = rng.choice(["a", "b", "c"], size=n)
    for col in settings.binary_cols:
        X[col] = rng.choice(["Yes", "No"], size=n)
    y = rng.choice([0, 1], size=n, p=[0.75, 0.25])
    return X, y


def test_smote_keeps_senior_citizen_binary_and_undistorted():
    # Regression guard: SeniorCitizen used to be missing from SMOTENC's
    # categorical_features, so it was interpolated (e.g. 0.6) and the int64
    # cast truncated it to 0, under-representing seniors among synthetic
    # churners (~50% real -> ~30% synthetic).
    X, y = _smote_ready_frame()
    X_res, _ = build_smote().fit_resample(X, y)

    synthetic = X_res.iloc[len(X):]
    assert set(synthetic["SeniorCitizen"].unique()) <= {0, 1}

    real_rate = X.loc[y == 1, "SeniorCitizen"].mean()
    synthetic_rate = synthetic["SeniorCitizen"].mean()
    assert abs(synthetic_rate - real_rate) < 0.1


def test_recode_no_service_replaces_only_the_service_value():
    X = pd.DataFrame({col: ["No internet service", "Yes", "No"]
                      for col in settings.service_cols_to_recode})
    X["InternetService"] = ["No", "DSL", "Fiber optic"]

    out = recode_no_service(X)

    for col in settings.service_cols_to_recode:
        assert out[col].tolist() == ["No", "Yes", "No"]
    # Columns outside service_cols_to_recode are untouched...
    assert out["InternetService"].tolist() == ["No", "DSL", "Fiber optic"]
    # ...and the input isn't mutated in place.
    assert X[settings.service_cols_to_recode[0]].iloc[0] == "No internet service"