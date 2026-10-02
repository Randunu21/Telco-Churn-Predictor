"""FastAPI application: assembles the lifespan and the routers.

Run locally with:
    uvicorn telco_churn.api.main:app --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from importlib.metadata import version


from fastapi import FastAPI

from telco_churn.api.model_loader import load_model
from telco_churn.api.routes import health, model_info
from telco_churn.config import settings
from telco_churn.api.routes import health, model_info, predict
from telco_churn.api.errors import register_exception_handlers
from telco_churn.api.middleware import add_request_id_middleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: runs once, before any request is served.
    # Fail fast: no try/except. If the model can't load, startup aborts with
    # a clear traceback (and Kubernetes will retry with backoff).
    app.state.loaded_model = load_model(
        model_uri=settings.model_uri,
        tracking_uri=settings.mlflow_tracking_uri,
    )
    yield
    # Shutdown
    app.state.loaded_model = None
    logger.info("Shutting down: model released.")


app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "Serves the champion churn model from the MLflow Model Registry. "
        "churn_score is a model score (SMOTE-trained), not a calibrated probability."
    ),
    version=version("telco-churn-mlops"),
    lifespan=lifespan,
)

register_exception_handlers(app)
add_request_id_middleware(app)

app.include_router(health.router)  # /health/live, /health/ready
app.include_router(model_info.router, prefix="/v1")  # /v1/model-info
app.include_router(predict.router, prefix="/v1")  # /v1/predict, /v1/predict/batch