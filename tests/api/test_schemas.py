from typing import get_args

import pytest
from pydantic import ValidationError

from telco_churn.api.schemas import (
    FEATURE_COLUMNS,
    BatchPredictionRequest,
    CustomerFeatures,
    PredictionResponse,
)
from telco_churn.config import settings
from telco_churn.data.schema import raw_schema


# ------------------------------------------------------------ valid input
def test_valid_payload_passes(valid_customer_payload):
    customer = CustomerFeatures(**valid_customer_payload)
    assert customer.Contract == "Month-to-month"


def test_documented_example_is_valid():
    # Guarantees /docs never shows an example that the API would reject.
    example = CustomerFeatures.model_config["json_schema_extra"]["examples"][0]
    CustomerFeatures(**example)


def test_customer_id_is_optional(valid_customer_payload):
    del valid_customer_payload["customerID"]
    assert CustomerFeatures(**valid_customer_payload).customerID is None


# ---------------------------------------------------------- invalid input
def test_invalid_category_rejected(valid_customer_payload):
    valid_customer_payload["Contract"] = "Weekly"
    with pytest.raises(ValidationError):
        CustomerFeatures(**valid_customer_payload)


def test_negative_tenure_rejected(valid_customer_payload):
    valid_customer_payload["tenure"] = -1
    with pytest.raises(ValidationError):
        CustomerFeatures(**valid_customer_payload)


def test_missing_required_field_rejected(valid_customer_payload):
    del valid_customer_payload["PaymentMethod"]
    with pytest.raises(ValidationError):
        CustomerFeatures(**valid_customer_payload)


def test_senior_citizen_outside_zero_one_rejected(valid_customer_payload):
    valid_customer_payload["SeniorCitizen"] = 2
    with pytest.raises(ValidationError):
        CustomerFeatures(**valid_customer_payload)


def test_senior_citizen_string_is_rejected(valid_customer_payload):
    # Literal fields match exactly: the string "1" is not the integer 1,
    # even in Pydantic's default lax mode.
    valid_customer_payload["SeniorCitizen"] = "1"
    with pytest.raises(ValidationError):
        CustomerFeatures(**valid_customer_payload)


def test_tenure_numeric_string_is_coerced(valid_customer_payload):
    # Documents a deliberate choice: plain int fields use lax mode, which
    # converts "5" -> 5 (but rejects "5.5" or 5.5). Accepted because the value
    # is unambiguous. Use strict mode (and flip this test) to forbid it.
    valid_customer_payload["tenure"] = "5"
    assert CustomerFeatures(**valid_customer_payload).tenure == 5


# ------------------------------------------------------------ extra fields
def test_extra_fields_are_ignored(valid_customer_payload):
    valid_customer_payload["gender"] = "Female"
    valid_customer_payload["MonthlyCharges"] = 29.85
    dumped = CustomerFeatures(**valid_customer_payload).model_dump()
    assert "gender" not in dumped
    assert "MonthlyCharges" not in dumped


def test_feature_columns_exclude_customer_id():
    assert "customerID" not in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 14


def test_feature_columns_match_training_features():
    # The API's features must be exactly the columns the model was trained on:
    # every raw column, minus what clean() drops, minus the target.
    training_features = set(raw_schema.columns) - set(settings.cols_to_drop) - {"Churn"}
    assert set(FEATURE_COLUMNS) == training_features


# ------------------------------------------------------------------ batch
def test_batch_rejects_empty_list():
    with pytest.raises(ValidationError):
        BatchPredictionRequest(customers=[])


def test_batch_rejects_more_than_max(valid_customer_payload):
    too_many = [valid_customer_payload] * (settings.max_batch_size + 1)
    with pytest.raises(ValidationError):
        BatchPredictionRequest(customers=too_many)


def test_batch_accepts_max(valid_customer_payload):
    at_limit = [valid_customer_payload] * settings.max_batch_size
    assert (
        len(BatchPredictionRequest(customers=at_limit).customers)
        == settings.max_batch_size
    )


# --------------------------------------------------------------- response
@pytest.mark.parametrize("bad_score", [-0.1, 1.5])
def test_prediction_response_rejects_out_of_range_score(bad_score):
    with pytest.raises(ValidationError):
        PredictionResponse(
            churn_score=bad_score,
            churn_prediction="Yes",
            threshold=0.5,
            model_version="1",
        )


# ----------------------------------------------------- Pydantic <-> Pandera
def _pandera_allowed_values(column_name):
    for check in raw_schema.columns[column_name].checks:
        if "allowed_values" in check.statistics:
            return set(check.statistics["allowed_values"])
    return None  # column has no isin check (e.g. tenure uses ge=0)


@pytest.mark.parametrize("column_name", FEATURE_COLUMNS)
def test_allowed_values_match_pandera_schema(column_name):
    pandera_values = _pandera_allowed_values(column_name)
    if pandera_values is None:
        pytest.skip(f"{column_name} has no categorical check in the Pandera schema")

    pydantic_values = set(
        get_args(CustomerFeatures.model_fields[column_name].annotation)
    )
    assert pydantic_values == pandera_values, (
        f"{column_name}: API allows {sorted(map(str, pydantic_values))}, "
        f"training schema allows {sorted(map(str, pandera_values))}"
    )