"""Exception handlers that turn every error into the same ErrorResponse shape.

Rule: clients get a code, a safe message and a request_id. Tracebacks and
exception text go to the server log only, tagged with the same request_id.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from telco_churn.api.inference import ModelOutputError
from telco_churn.api.middleware import REQUEST_ID_HEADER, get_request_id
from telco_churn.api.schemas import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)

GENERIC_SERVER_ERROR = (
    "An internal error occurred. Quote the request_id when reporting this issue."
)

# Machine-readable codes for HTTP errors raised by FastAPI/Starlette or by us
_HTTP_ERROR_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    503: "model_unavailable",
}


# Documents the error shape in /docs for the prediction routes.
PREDICTION_ERROR_RESPONSES = {
    422: {"model": ErrorResponse, "description": "Invalid request body."},
    500: {"model": ErrorResponse, "description": "Internal error."},
    503: {"model": ErrorResponse, "description": "Model not loaded."},
}


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[dict] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = get_request_id(request)
    body = ErrorResponse(
        error=ErrorDetail(
            code=code, message=message, request_id=request_id, details=details
        )
    )
    # Set the request-ID header here too: the catch-all handler runs outside
    # the middleware, so the middleware can't add it to that response.
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(exclude_none=True),
        headers={**(headers or {}), REQUEST_ID_HEADER: request_id},
    )


def _validation_details(exc: RequestValidationError) -> list[dict]:
    details = []
    for error in exc.errors():
        detail = {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
        # Echo the rejected value back only if it's a single value: echoing a
        # whole 501-item list back to the client would be absurd.
        value = error.get("input")
        if value is None or isinstance(value, str | int | float | bool):
            detail["input"] = value
        details.append(jsonable_encoder(detail))
    return details


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Client's mistake: INFO, no traceback, and never the payload itself.
    logger.info(
        "Validation error (request_id=%s): %d problem(s)",
        get_request_id(request),
        len(exc.errors()),
    )
    return _error_response(
        request,
        status_code=422,
        code="validation_error",
        message="The request body is invalid. See details.",
        details=_validation_details(exc),
    )


async def http_error_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    # Registered for Starlette's HTTPException (not FastAPI's subclass), so it
    # also catches the router's own 404 and 405 errors.
    log = logger.warning if exc.status_code >= 500 else logger.info
    log(
        "HTTP %d (request_id=%s): %s",
        exc.status_code,
        get_request_id(request),
        exc.detail,
    )
    return _error_response(
        request,
        status_code=exc.status_code,
        code=_HTTP_ERROR_CODES.get(exc.status_code, "http_error"),
        message=str(exc.detail),
        headers=exc.headers,  # keeps e.g. the Allow header on 405 responses
    )


async def model_output_error_handler(
    request: Request, exc: ModelOutputError
) -> JSONResponse:
    logger.error(
        "Model output error (request_id=%s): %s",
        get_request_id(request),
        exc,
        exc_info=exc,
    )
    return _error_response(
        request,
        status_code=500,
        code="model_output_error",
        message=GENERIC_SERVER_ERROR,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Full details go to the log only. Starlette re-raises the exception after
    # this response is sent, so the server may log the traceback a second time.
    logger.error(
        "Unhandled error (request_id=%s): %s",
        get_request_id(request),
        exc,
        exc_info=exc,
    )
    return _error_response(
        request,
        status_code=500,
        code="internal_error",
        message=GENERIC_SERVER_ERROR,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(ModelOutputError, model_output_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)