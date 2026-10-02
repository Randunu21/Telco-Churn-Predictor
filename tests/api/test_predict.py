import numpy as np
import pytest

from telco_churn.config import settings

# ------------------------------------------------------------------ single


def test_predict_returns_stub_score(client, stub_model, valid_customer_payload):
    stub_model.scores = [0.7]
    response = client.post("/v1/predict", json=valid_customer_payload)

    assert response.status_code == 200
    assert response.json() == {
        "churn_score": pytest.approx(0.7),
        "churn_prediction": "Yes",
        "threshold": settings.decision_threshold,
        "model_version": "42",
        "customerID": valid_customer_payload["customerID"],
    }


def test_score_equal_to_threshold_is_yes(client, stub_model, valid_customer_payload):
    stub_model.scores = [settings.decision_threshold]
    response = client.post("/v1/predict", json=valid_customer_payload)
    assert response.json()["churn_prediction"] == "Yes"


def test_threshold_override_changes_label(
    client, stub_model, override_settings, valid_customer_payload
):
    stub_model.scores = [0.7]
    override_settings(decision_threshold=0.9)

    body = client.post("/v1/predict", json=valid_customer_payload).json()

    assert body["churn_prediction"] == "No"
    assert body["threshold"] == 0.9


def test_full_raw_telco_row_is_accepted(client, stub_model, valid_customer_payload):
    raw_row = {
        **valid_customer_payload,
        "gender": "Female",
        "PhoneService": "No",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
        "Churn": "No",
    }
    response = client.post("/v1/predict", json=raw_row)

    assert response.status_code == 200
    # ...and none of the extras reached the model
    assert "gender" not in stub_model.received.columns


# ------------------------------------------------------------------- batch


def test_batch_preserves_order_and_ids(client, stub_model, valid_customer_payload):
    stub_model.scores = [0.1, 0.9, 0.4]
    customers = [{**valid_customer_payload, "customerID": cid} for cid in "ABC"]

    body = client.post("/v1/predict/batch", json={"customers": customers}).json()

    assert body["model_version"] == "42"
    assert [p["customerID"] for p in body["predictions"]] == ["A", "B", "C"]
    assert [p["churn_prediction"] for p in body["predictions"]] == ["No", "Yes", "No"]


def test_batch_rejects_empty_list(client):
    response = client.post("/v1/predict/batch", json={"customers": []})
    assert response.status_code == 422


def test_batch_rejects_more_than_max(client, valid_customer_payload):
    too_many = [valid_customer_payload] * (settings.max_batch_size + 1)
    response = client.post("/v1/predict/batch", json={"customers": too_many})
    assert response.status_code == 422


# ---------------------------------------------- real pipeline, no MLflow


def _to_json_rows(frame):
    # NumPy integers aren't JSON-serialisable: convert to plain Python values.
    return [
        {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in row.items()
        }
        for row in frame.to_dict("records")
    ]


def test_api_matches_real_pipeline_exactly(real_pipeline_client):
    # 4.4's manual consistency check, automated: the API's DataFrame must
    # produce exactly the scores the real pipeline gives on the same rows.
    client, pipeline, X_test = real_pipeline_client
    rows = X_test.head(5)

    response = client.post(
        "/v1/predict/batch", json={"customers": _to_json_rows(rows)}
    )

    assert response.status_code == 200
    api_scores = [p["churn_score"] for p in response.json()["predictions"]]
    direct_scores = pipeline.predict_proba(rows)[:, 1]
    np.testing.assert_array_equal(api_scores, direct_scores)