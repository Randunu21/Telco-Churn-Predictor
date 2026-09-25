import pandera.pandas as pa
from pandera.pandas import Column, Check

raw_schema = pa.DataFrameSchema(
    {
        "customerID": Column(str, unique=True),
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(["Yes", "No"])),
        "Dependents": Column(str, Check.isin(["Yes", "No"])),
        "tenure": Column(int, Check.ge(0)),
        "PhoneService": Column(str, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(str, Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(str, Check.isin(["Yes", "No", "No internet service"])),
        "OnlineBackup": Column(str, Check.isin(["Yes", "No", "No internet service"])),
        "DeviceProtection": Column(str, Check.isin(["Yes", "No", "No internet service"])),
        "TechSupport": Column(str, Check.isin(["Yes", "No", "No internet service"])),
        "StreamingTV": Column(str, Check.isin(["Yes", "No", "No internet service"])),
        "StreamingMovies": Column(str, Check.isin(["Yes", "No", "No internet service"])),
        "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(str, Check.isin([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])),
        "MonthlyCharges": Column(float, Check.ge(0)),
        "TotalCharges": Column(str, nullable=True),
        "Churn": Column(str, Check.isin(["Yes", "No"])),
    },
    strict=False,
    coerce=False,
)


def validate_raw(df):
    return raw_schema.validate(df, lazy=True)