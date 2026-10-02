import logging

import numpy as np

from telco_churn.api.middleware import REQUEST_ID_HEADER

# ---------------------------------------------------------- 4xx envelopes


def test_validation_error_envelope(client, valid_customer_payload):
    valid_customer_payload["Contract"] = "Weekly"
    response = client.post("/v1/predict", json=valid_customer_payload)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert error["details"][0]["loc"][-1] == "Contract"
    assert error["details"][0]["input"] == "Weekly"


def test_404_uses_the_same_envelope(client):
    response = client.get("/v1/nothing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_405_keeps_allow_header(client):
    response = client.get("/v1/predict")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"
    assert "POST" in response.headers["allow"]


# ------------------------------------------------------------- request IDs


def test_every_response_has_a_request_id(client, valid_customer_payload):
    ok = client.post("/v1/predict", json=valid_customer_payload)
    not_found = client.get("/v1/nothing")
    assert ok.headers[REQUEST_ID_HEADER]
    assert not_found.headers[REQUEST_ID_HEADER]
    assert ok.headers[REQUEST_ID_HEADER] != not_found.headers[REQUEST_ID_HEADER]


def test_valid_incoming_request_id_is_echoed(client):
    response = client.get("/v1/nothing", headers={REQUEST_ID_HEADER: "test-123"})
    assert response.headers[REQUEST_ID_HEADER] == "test-123"
    assert response.json()["error"]["request_id"] == "test-123"


def test_invalid_incoming_request_id_is_replaced(client):
    response = client.get("/health/live", headers={REQUEST_ID_HEADER: "has space"})
    assert response.headers[REQUEST_ID_HEADER] != "has space"


# ------------------------------------------------------------------- 5xx


def test_model_output_error_is_500(client_no_raise, stub_model, valid_customer_payload):
    stub_model.scores = [np.nan]
    response = client_no_raise.post("/v1/predict", json=valid_customer_payload)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "model_output_error"


def test_unhandled_error_never_leaks_details(
    client_no_raise, stub_model, valid_customer_payload, caplog
):
    caplog.set_level(logging.INFO)
    stub_model.error = RuntimeError("secret internal detail")

    response = client_no_raise.post("/v1/predict", json=valid_customer_payload)

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "internal_error"
    # The core rule of 4.5: the client sees nothing internal...
    assert "secret" not in response.text
    # ...while the log has the detail, tagged with the same request ID.
    assert "secret internal detail" in caplog.text
    assert error["request_id"] in caplog.text


def test_validation_errors_do_not_log_the_payload(
    client, valid_customer_payload, caplog
):
    caplog.set_level(logging.INFO)
    valid_customer_payload["customerID"] = "PRIVATE-123"
    valid_customer_payload["Contract"] = "Weekly"

    client.post("/v1/predict", json=valid_customer_payload)

    assert "Validation error" in caplog.text
    assert "PRIVATE-123" not in caplog.text
    assert "Weekly" not in caplog.text