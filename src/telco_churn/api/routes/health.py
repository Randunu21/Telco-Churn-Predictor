"""Infrastructure endpoints, mapped to Kubernetes probes in Phase 7.

Mounted at the root (not under /v1): they are not part of the API contract.
"""

from fastapi import APIRouter, Request, Response, status

from telco_churn.api.schemas import LivenessResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=LivenessResponse)
async def live() -> LivenessResponse:
    # Must never touch the model or MLflow: if it did, an MLflow outage would
    # make Kubernetes restart healthy pods in a loop.
    return LivenessResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "Model not loaded"}},
)
async def ready(request: Request, response: Response) -> ReadinessResponse:
    # Checks app.state directly rather than using get_loaded_model: here
    # "no model" is a normal answer to report, not an error to raise.
    if getattr(request.app.state, "loaded_model", None) is None:
        # Kubernetes reads the status code, not the body.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="not_ready", model_loaded=False)
    return ReadinessResponse(status="ready", model_loaded=True)