"""Generate a small synthetic Telco churn dataset for the CI smoke test.

The real dataset is DVC-tracked and CI has no DVC credentials, so the smoke
test trains on this instead. It matches the raw Pandera schema (same columns,
values and dtypes) and has enough churners for SMOTENC. Seeded, so every run
gets the same data. The model it produces is only good enough to prove the
stack works end to end; its scores mean nothing.

Usage: python scripts/smoke/make_data.py <output.csv> [--rows N]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
YES_NO = ["Yes", "No"]
# Column order of the real dataset (data/raw/Telco_Cusomer_Churn.csv).
REAL_COLUMN_ORDER = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def make_dataset(rows: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def pick(values, p=None):
        return rng.choice(values, size=rows, p=p)

    tenure = rng.integers(0, 73, size=rows)
    tenure[:3] = 0  # brand-new customers, as in the real data
    contract = pick(["Month-to-month", "One year", "Two year"], p=[0.55, 0.25, 0.2])
    internet = pick(["DSL", "Fiber optic", "No"], p=[0.35, 0.45, 0.2])
    phone = pick(YES_NO, p=[0.9, 0.1])
    monthly = np.round(rng.uniform(18.25, 118.75, size=rows), 2)

    df = pd.DataFrame(
        {
            "customerID": [f"SMOKE-{i:05d}" for i in range(rows)],
            "gender": pick(["Male", "Female"]),
            "SeniorCitizen": rng.integers(0, 2, size=rows),
            "Partner": pick(YES_NO),
            "Dependents": pick(YES_NO, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(phone == "No", "No phone service", pick(YES_NO)),
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": pick(YES_NO, p=[0.6, 0.4]),
            "PaymentMethod": pick(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
            "MonthlyCharges": monthly,
        }
    )
    for col in SERVICE_COLS:
        df[col] = np.where(internet == "No", "No internet service", pick(YES_NO))

    # The real column is text, with a blank " " for tenure-0 customers. Keeping
    # those blanks makes pandas read it back as text, which the schema expects.
    total = np.round(monthly * tenure, 2).astype(str)
    df["TotalCharges"] = np.where(tenure == 0, " ", total)

    # Churn loosely follows the real pattern (short month-to-month contracts
    # churn most), so the model learns something; about 30% churners.
    risk = (
        0.08
        + 0.30 * (contract == "Month-to-month")
        + 0.20 * (tenure < 12)
        + 0.10 * (internet == "Fiber optic")
    )
    df["Churn"] = np.where(rng.random(rows) < risk, "Yes", "No")

    # Same column order as the real Kaggle CSV. The trained pipeline remembers
    # the order it was fitted on, so the synthetic model must see the real one.
    return df[REAL_COLUMN_ORDER]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="CSV file to write")
    parser.add_argument("--rows", type=int, default=300)
    args = parser.parse_args()

    df = make_dataset(args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    churners = int((df["Churn"] == "Yes").sum())
    print(f"Wrote {len(df)} rows ({churners} churners) to {args.output}")


if __name__ == "__main__":
    main()
