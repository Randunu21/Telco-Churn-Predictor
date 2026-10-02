def test_live_returns_alive(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_live_does_not_depend_on_the_model(client_without_model):
    # Liveness must never depend on the model, or an MLflow outage would make
    # Kubernetes restart healthy pods in a loop.
    response = client_without_model.get("/health/live")
    assert response.status_code == 200


def test_ready_when_model_loaded(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_loaded": True}


def test_not_ready_without_model(client_without_model):
    response = client_without_model.get("/health/ready")
    # Kubernetes reads the status code, not the body.
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "model_loaded": False}