"""Prediction endpoints, mounted under /v1.

Both are plain `def`, NOT `async def`: predict_proba is CPU-bound, and inside
`async def` it would block the event loop, stalling every other request
(including /health/live). Plain `def` runs in FastAPI's thread pool instead.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends

from telco_churn.api.dependencies import get_loaded_model, get_settings
from telco_churn.api.errors import PREDICTION_ERROR_RESPONSES
from telco_churn.api.inference import predict
from telco_churn.api.model_loader import LoadedModel
from telco_churn.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    PredictionResponse,
)
from telco_churn.config import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses=PREDICTION_ERROR_RESPONSES,
)
def predict_one(
    customer: CustomerFeatures,
    loaded_model: Annotated[LoadedModel, Depends(get_loaded_model)],
    app_settings: Annotated[Settings, Depends(get_settings)],
) -> PredictionResponse:
    return predict(loaded_model, [customer], app_settings.decision_threshold)[0]


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses=PREDICTION_ERROR_RESPONSES,
)
def predict_batch(
    batch: BatchPredictionRequest,  # not "request": that name means the raw Request
    loaded_model: Annotated[LoadedModel, Depends(get_loaded_model)],
    app_settings: Annotated[Settings, Depends(get_settings)],
) -> BatchPredictionResponse:
    start = time.perf_counter()
    predictions = predict(
        loaded_model, batch.customers, app_settings.decision_threshold
    )
    # Log counts and timings only, never payloads (customer data is personal data).
    logger.info(
        "Scored %d customers in %.1f ms",
        len(predictions),
        (time.perf_counter() - start) * 1000,
    )
    return BatchPredictionResponse(
        model_version=loaded_model.model_version,
        predictions=predictions,
    )