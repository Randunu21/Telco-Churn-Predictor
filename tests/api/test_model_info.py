from telco_churn.config import settings


def test_model_info_reports_loaded_model(client):
    response = client.get("/v1/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "test-model",
        "model_version": "42",
        "model_alias": "champion",
        "run_id": "fake-run-id",
        "model_uri": "models:/test-model@champion",
        "decision_threshold": settings.decision_threshold,
    }


def test_threshold_comes_from_settings(client, override_settings):
    override_settings(decision_threshold=0.83)
    response = client.get("/v1/model-info")
    assert response.json()["decision_threshold"] == 0.83


def test_model_info_without_model_is_503(client_without_model):
    response = client_without_model.get("/v1/model-info")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"