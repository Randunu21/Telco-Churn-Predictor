"""Functions endpoints use (via Depends) to get what they need.

Endpoints never read app.state directly. In tests (4.6) these functions are
swapped out with app.dependency_overrides, so no real MLflow is needed.
"""

from fastapi import HTTPException, Request, status

from telco_churn.api.model_loader import LoadedModel
from telco_churn.config import Settings, settings


def get_settings() -> Settings:
    return settings


def get_loaded_model(request: Request) -> LoadedModel:
    # Reach the app through the request, never by importing main.py
    # (main imports the routes; routes importing main would be circular).
    loaded_model = getattr(request.app.state, "loaded_model", None)
    if loaded_model is None:
        # 503: temporarily unable to serve (not 500: nothing "broke").
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return loaded_model