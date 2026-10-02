"""Request-ID middleware: every request gets an ID, returned in the
X-Request-ID response header and used in error bodies and log lines, so a
client's error report can be matched to the server log without exposing
any internals to the client.
"""

import re
import uuid

from fastapi import FastAPI, Request

REQUEST_ID_HEADER = "X-Request-ID"

# Accept a caller's ID only if it is short and made of safe characters.
# Anything else (very long values, spaces, newlines that could forge fake
# log lines) is replaced with a fresh UUID.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9-]{1,64}")


def is_valid_request_id(value: str) -> bool:
    return bool(_VALID_REQUEST_ID.fullmatch(value))


def get_request_id(request: Request) -> str:
    # Fallback for the rare case the middleware never ran for this request.
    return getattr(request.state, "request_id", "unknown")


def add_request_id_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if is_valid_request_id(incoming) else str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response