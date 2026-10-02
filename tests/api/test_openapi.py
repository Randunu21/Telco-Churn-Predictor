"""Contract tests: check the API's description, not its behaviour.

app.openapi() is exactly what /docs renders. These would have caught
response_model/responses accidentally written as function parameters.
"""

import pytest

from telco_churn.api.main import app

PREDICTION_PATHS = ["/v1/predict", "/v1/predict/batch"]
ERROR_REF = "#/components/schemas/ErrorResponse"


@pytest.fixture(scope="module")
def schema():
    return app.openapi()


def test_expected_paths_exist(schema):
    assert set(schema["paths"]) >= {
        "/health/live",
        "/health/ready",
        "/v1/model-info",
        *PREDICTION_PATHS,
    }


@pytest.mark.parametrize("path", PREDICTION_PATHS)
def test_prediction_routes_take_only_a_body(path, schema):
    operation = schema["paths"][path]["post"]
    assert operation.get("parameters", []) == []
    assert "requestBody" in operation


@pytest.mark.parametrize("path", PREDICTION_PATHS)
@pytest.mark.parametrize("status", ["422", "500", "503"])
def test_prediction_routes_document_errors(path, status, schema):
    response = schema["paths"][path]["post"]["responses"][status]
    assert response["content"]["application/json"]["schema"]["$ref"] == ERROR_REF


def test_customer_example_is_documented(schema):
    customer_schema = schema["components"]["schemas"]["CustomerFeatures"]
    assert customer_schema["examples"][0]["Contract"] == "Month-to-month"