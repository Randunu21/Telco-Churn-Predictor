"""Turn validated customers into predictions.

No FastAPI in here on purpose: plain Python, unit-testable with a stub model,
and reusable outside the API (e.g. batch scoring in Phase 8).
"""

import numpy as np
import pandas as pd

from telco_churn.api.model_loader import LoadedModel
from telco_churn.api.schemas import (
    FEATURE_COLUMNS,
    CustomerFeatures,
    PredictionResponse,
)


class ModelOutputError(Exception):
    """The model produced output the API refuses to serve (e.g. NaN scores).

    A domain error, not an HTTP one: defined here (no FastAPI imports) so
    inference stays reusable outside the API. api/errors.py translates it
    into an HTTP 500.
    """


def customers_to_frame(customers: list[CustomerFeatures]) -> pd.DataFrame:
    """One row per customer, exactly FEATURE_COLUMNS, in input order.

    Selecting FEATURE_COLUMNS explicitly is what keeps customerID (and any
    future non-feature field) out of the model's input.
    """
    rows = [customer.model_dump() for customer in customers]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def predict(
    loaded_model: LoadedModel,
    customers: list[CustomerFeatures],
    threshold: float,
) -> list[PredictionResponse]:
    """Score customers and apply the decision threshold.

    Single and batch endpoints both call this: a single prediction is just
    a batch of one.
    """
    frame = customers_to_frame(customers)
    # Column 1 = class 1 = "Yes" (churn); guaranteed by check_classes at load.
    scores = loaded_model.model.predict_proba(frame)[:, 1]

    # Fail with a precise error rather than letting Pydantic's 0-1 check
    # fail later as a confusing response-validation error.
    not_finite = ~np.isfinite(scores)
    if not_finite.any():
        raise ModelOutputError(
            f"Model returned {int(not_finite.sum())} non-finite score(s) "
            f"out of {len(scores)}."
        )

    return [
        PredictionResponse(
            churn_score=float(score),
            # >= matches evaluate() in train.py, so offline metrics and live
            # behaviour use exactly the same rule.
            churn_prediction="Yes" if score >= threshold else "No",
            threshold=threshold,
            model_version=loaded_model.model_version,
            customerID=customer.customerID,
        )
        for customer, score in zip(customers, scores, strict=True)
    ]