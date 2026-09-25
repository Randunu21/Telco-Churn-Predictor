import pandas as pd
import pytest


@pytest.fixture
def raw_telco_df():
    return pd.DataFrame({
        "customerID": [f"CUST{i:04d}" for i in range(1, 12)],
        "gender": ["Female", "Male", "Male", "Male", "Female", "Female",
                    "Male", "Female", "Female", "Male", "Male"],
        "SeniorCitizen": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "Partner": ["Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No"],
        "Dependents": ["No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No"],
        "tenure": [1, 34, 2, 45, 2, 8, 22, 10, 28, 62, 5],
        "PhoneService": ["No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "No", "No phone service", "No", "Yes",
                            "Yes", "No phone service", "Yes", "No", "No"],
        "InternetService": ["DSL", "DSL", "DSL", "DSL", "Fiber optic", "Fiber optic",
                              "Fiber optic", "DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "Yes", "Yes", "Yes", "No", "No", "No", "Yes", "No", "Yes",
                            "No internet service"],
        "OnlineBackup": ["Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes",
                          "No internet service"],
        "DeviceProtection": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No",
                              "No internet service"],
        "TechSupport": ["No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No",
                         "No internet service"],
        "StreamingTV": ["No", "No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No",
                         "No internet service"],
        "StreamingMovies": ["No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No",
                             "No internet service"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "One year",
                      "Month-to-month", "Month-to-month", "Month-to-month", "Month-to-month",
                      "Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Mailed check",
                           "Bank transfer (automatic)", "Electronic check", "Electronic check",
                           "Credit card (automatic)", "Mailed check", "Electronic check",
                           "Bank transfer (automatic)", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 29.75, 104.80, 56.15, 20.05],
        "TotalCharges": ["29.85", "1889.5", "108.15", "1840.75", "151.65", "820.5", "1949.4", "301.9", "3046.05", "3487.95", "100.25"],
        "Churn": ["No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No"],
    })