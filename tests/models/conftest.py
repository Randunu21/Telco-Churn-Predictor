import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


@pytest.fixture
def pipeline_ready_data():
    rng = np.random.default_rng(42)
    n = 60

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.4, 0.4, 0.2])

    def service_col():
        # Mirrors real Telco data: if there's no internet service, every
        # internet-dependent column MUST read "No internet service" — this
        # is exactly the pattern recode_no_service() exists to normalize.
        return [
            "No internet service" if internet_service[i] == "No" else rng.choice(["Yes", "No"])
            for i in range(n)
        ]

    X = pd.DataFrame({
        "SeniorCitizen": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
        "Partner": rng.choice(["Yes", "No"], size=n),
        "Dependents": rng.choice(["Yes", "No"], size=n),
        "tenure": rng.integers(1, 72, size=n),
        "InternetService": internet_service,
        "OnlineSecurity": service_col(),
        "OnlineBackup": service_col(),
        "DeviceProtection": service_col(),
        "TechSupport": service_col(),
        "StreamingTV": service_col(),
        "StreamingMovies": service_col(),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            size=n,
        ),
    })
    # Already 0/1 here, matching what train.py actually hands the pipeline
    # after target encoding — build_pipeline() itself does no target encoding.
    y = pd.Series(rng.choice([0, 1], size=n, p=[0.7, 0.3]), name="Churn")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test