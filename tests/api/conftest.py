import pytest
from fastapi.testclient import TestClient

from telco_churn.api.dependencies import get_settings
from telco_churn.api.main import app
from telco_churn.api.schemas import EXAMPLE_CUSTOMER
from telco_churn.config import settings
from tests.api.fakes import StubModel, make_loaded_model


@pytest.fixture
def valid_customer_payload():
    """A fresh dict per test, so a test that mutates it can't affect others."""
    return dict(EXAMPLE_CUSTOMER)


@pytest.fixture
def stub_model():
    """Tests can change stub_model.scores / stub_model.error before a request."""
    return StubModel()


@pytest.fixture
def fake_loaded_model(stub_model):
    return make_loaded_model(stub_model)


def _client_with(monkeypatch, loaded_model, **client_kwargs):
    # Patch the name where it's USED: main.py did `from ... import load_model`,
    # so main holds its own reference. Patching model_loader.load_model would
    # leave main calling the real one (and trying to reach MLflow).
    monkeypatch.setattr(
        "telco_churn.api.main.load_model", lambda **kwargs: loaded_model
    )
    # `with` runs the real lifespan: startup stores the fake in app.state
    # exactly as in production; leaving the block runs shutdown.
    with TestClient(app, **client_kwargs) as client:
        yield client
    # app is one shared object: never let an override leak into later tests.
    app.dependency_overrides.clear()


@pytest.fixture
def client(monkeypatch, fake_loaded_model):
    yield from _client_with(monkeypatch, fake_loaded_model)


@pytest.fixture
def client_no_raise(monkeypatch, fake_loaded_model):
    """For 5xx tests: return the error response instead of re-raising the
    server-side exception into the test."""
    yield from _client_with(
        monkeypatch, fake_loaded_model, raise_server_exceptions=False
    )


@pytest.fixture
def client_without_model():
    """No `with`: the lifespan never runs, so no model is ever loaded."""
    app.state.loaded_model = None
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def override_settings():
    """Usage: override_settings(decision_threshold=0.9). Uses the dependency
    override mechanism that get_settings() was designed for in 4.3."""

    def _override(**changes):
        overridden = settings.model_copy(update=changes)
        app.dependency_overrides[get_settings] = lambda: overridden
        return overridden

    yield _override
    app.dependency_overrides.clear()


@pytest.fixture
def real_pipeline_client(monkeypatch, pipeline_ready_data):
    """The API serving a REAL fitted imblearn pipeline (no MLflow), so the
    API's DataFrame is checked against the real preprocessing."""
    from sklearn.linear_model import LogisticRegression

    from telco_churn.models.pipeline import build_pipeline

    X_train, X_test, y_train, _y_test = pipeline_ready_data
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    for client in _client_with(monkeypatch, make_loaded_model(pipeline)):
        yield client, pipeline, X_test