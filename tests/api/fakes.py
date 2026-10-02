"""Test doubles shared by the API tests (imported, not fixtures)."""

import numpy as np

from telco_churn.api.model_loader import LoadedModel


class StubModel:
    """Stands in for the real pipeline: returns configurable scores, can be
    told to fail, and records the DataFrame it received."""

    def __init__(self, scores=(0.7,)):
        self.classes_ = [0, 1]  # passes check_classes, like the real model
        self.scores = list(scores)
        self.error: Exception | None = None
        self.received = None

    def predict_proba(self, X):
        self.received = X
        if self.error is not None:
            raise self.error
        # One score given -> use it for every row; otherwise one per row.
        scores = self.scores * len(X) if len(self.scores) == 1 else self.scores
        positive = np.asarray(scores, dtype=float)
        return np.column_stack([1 - positive, positive])


def make_loaded_model(model, version="42") -> LoadedModel:
    # Distinctive fake metadata: a response showing version "42" can only
    # have come from this fake, never from a real registry.
    return LoadedModel(
        model=model,
        model_name="test-model",
        model_version=version,
        model_alias="champion",
        run_id="fake-run-id",
        model_uri="models:/test-model@champion",
    )