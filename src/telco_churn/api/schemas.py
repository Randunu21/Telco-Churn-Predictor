"""Request/response contract for the churn prediction API.

Allowed categorical values deliberately mirror the Pandera raw schema in
telco_churn.data.schema. They are written twice on purpose (no pandera
import here: the serving image stays lean); tests/api/test_schemas.py has a
parity test that fails if the two ever drift apart.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from telco_churn.config import settings

# Reusable value sets (defined once, used by several fields)
YesNo = Literal["Yes", "No"]
InternetAddOn = Literal["Yes", "No", "No internet service"]

# One real row from the Telco dataset. Shown pre-filled in /docs.
EXAMPLE_CUSTOMER = {
    "customerID": "7590-VHVEG",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


# ---------------------------------------------------------------- requests
class CustomerFeatures(BaseModel):
    """One customer's features: the 14 columns that survive clean()."""

    # extra="ignore": clients may send a full raw Telco row (gender,
    # MonthlyCharges, ...); unknown keys are silently dropped.
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={"examples": [EXAMPLE_CUSTOMER]},
    )

    # Not a model feature: echoed back so clients can match predictions
    # to their own rows. Must never reach the model (see FEATURE_COLUMNS).
    customerID: str | None = Field(
        default=None, description="Optional client-side ID, echoed back unchanged."
    )

    SeniorCitizen: Literal[0, 1] = Field(
        description="1 if the customer is 65 or older."
    )
    Partner: YesNo = Field(description="Whether the customer has a partner.")
    Dependents: YesNo = Field(description="Whether the customer has dependents.")
    tenure: int = Field(
        ge=0, description="Months the customer has stayed with the company."
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        description="Internet service provider type."
    )
    OnlineSecurity: InternetAddOn = Field(description="Online security add-on.")
    OnlineBackup: InternetAddOn = Field(description="Online backup add-on.")
    DeviceProtection: InternetAddOn = Field(description="Device protection add-on.")
    TechSupport: InternetAddOn = Field(description="Tech support add-on.")
    StreamingTV: InternetAddOn = Field(description="Streaming TV add-on.")
    StreamingMovies: InternetAddOn = Field(description="Streaming movies add-on.")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        description="Contract term."
    )
    PaperlessBilling: YesNo = Field(description="Whether billing is paperless.")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(description="Payment method.")


# The exact columns the model expects, in schema order. Use this (not
# model_dump() as-is) when building the DataFrame in 4.4, so customerID
# can never leak into the pipeline.
FEATURE_COLUMNS: list[str] = [
    name for name in CustomerFeatures.model_fields if name != "customerID"
]


class BatchPredictionRequest(BaseModel):
    """Wrapped in an object (not a bare list) so options can be added later
    without breaking existing clients."""

    customers: list[CustomerFeatures] = Field(
        min_length=1,
        max_length=settings.max_batch_size,
        description=f"1 to {settings.max_batch_size} customers.",
    )


# --------------------------------------------------------------- responses
class PredictionResponse(BaseModel):
    # churn_score, not churn_probability: the model was trained on
    # SMOTE-balanced data, so scores rank customers well but are not
    # calibrated real-world probabilities.
    churn_score: float = Field(
        ge=0.0, le=1.0, description="Model score (not calibrated)."
    )
    churn_prediction: Literal["Yes", "No"] = Field(
        description="'Yes' if churn_score >= threshold."
    )
    threshold: float = Field(description="Decision threshold that was applied.")
    model_version: str = Field(
        description="Registry version that produced this prediction."
    )
    customerID: str | None = Field(default=None, description="Echoed from the request.")


class BatchPredictionResponse(BaseModel):
    model_version: str
    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_alias: str | None = Field(
        default=None, description="None when model_uri pins a version or a path."
    )
    run_id: str
    model_uri: str
    decision_threshold: float


class LivenessResponse(BaseModel):
    status: Literal["alive"] = "alive"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool