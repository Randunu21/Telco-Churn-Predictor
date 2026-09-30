from typing import Annotated

from fastapi import APIRouter, Depends

from telco_churn.api.dependencies import get_loaded_model, get_settings
from telco_churn.api.model_loader import LoadedModel
from telco_churn.api.schemas import ModelInfoResponse
from telco_churn.config import Settings

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info(
    loaded_model: Annotated[LoadedModel, Depends(get_loaded_model)],
    app_settings: Annotated[Settings, Depends(get_settings)],
) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name=loaded_model.model_name,
        model_version=loaded_model.model_version,
        model_alias=loaded_model.model_alias,
        run_id=loaded_model.run_id,
        model_uri=loaded_model.model_uri,
        decision_threshold=app_settings.decision_threshold,
    )