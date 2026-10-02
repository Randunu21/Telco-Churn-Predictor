"""Load the serving model from MLflow and resolve the facts about it.

No FastAPI in here on purpose: this is plain Python, testable on its own and
reusable outside the API (e.g. by an Airflow task in Phase 8).
"""

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

# Placeholder metadata when model_uri is not a registry URI (runs:/ or a path)
UNREGISTERED = "unregistered"
UNKNOWN = "unknown"

# The API maps predict_proba column 1 to "Yes" (churn). That is only true if
# the model's classes are [0, 1], i.e. training encoded No=0, Yes=1.
EXPECTED_CLASSES = [0, 1]
REGISTRY_PREFIX = "models:/"
RUNS_PREFIX = "runs:/"


@dataclass(frozen=True)
class ParsedModelUri:
    """What a model URI says about the registry, before asking MLflow anything."""

    name: str | None = None
    alias: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class LoadedModel:
    """A loaded model plus the facts the API reports about it."""

    model: Any
    model_name: str
    model_version: str
    model_alias: str | None
    run_id: str
    model_uri: str


def parse_model_uri(model_uri: str) -> ParsedModelUri:
    """Pure parsing, no network calls.

    models:/<name>@<alias>   -> name + alias
    models:/<name>/<version> -> name + version
    anything else            -> no registry information
    """
    if not model_uri.startswith(REGISTRY_PREFIX):
        return ParsedModelUri()

    reference = model_uri[len(REGISTRY_PREFIX) :]

    if "@" in reference:
        name, alias = reference.split("@", 1)
        if not name or not alias:
            raise ValueError(f"Malformed alias URI: {model_uri!r}")
        return ParsedModelUri(name=name, alias=alias)

    if "/" in reference:
        name, version = reference.rsplit("/", 1)
        if not name or not version.isdigit():
            raise ValueError(
                f"Malformed version URI: {model_uri!r}. Expected "
                "models:/<name>/<number>. Stage URIs (e.g. /Production) are "
                "deprecated; use an alias: models:/<name>@<alias>."
            )
        return ParsedModelUri(name=name, version=version)

    raise ValueError(
        f"Registry URI {model_uri!r} names no alias or version. "
        "Use models:/<name>@<alias> or models:/<name>/<version>."
    )


def _run_id_from_runs_uri(model_uri: str) -> str:
    """runs:/<run_id>/<path> -> <run_id>; anything else -> 'unknown'."""
    if model_uri.startswith(RUNS_PREFIX):
        return model_uri[len(RUNS_PREFIX) :].split("/", 1)[0] or UNKNOWN
    return UNKNOWN


def check_classes(model: Any) -> None:
    """Fail fast if the model's class order isn't what the API assumes.

    Protects against a future retrain with a different label encoding
    silently flipping every prediction.
    """
    classes = list(getattr(model, "classes_", []))
    if classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Model classes are {classes}, expected {EXPECTED_CLASSES} "
            "(No=0, Yes=1). Refusing to serve: predict_proba columns would "
            "be misinterpreted."
        )

def load_model(model_uri: str, tracking_uri: str) -> LoadedModel:
    """Resolve the model's metadata, load it, and return both.

    Raises on any failure (fail fast): the API should not start without a model.
    """
    # Without this, MLflow silently falls back to a local SQLite file and
    # reports the registered model as missing (the 4.1 pitfall).
    mlflow.set_tracking_uri(tracking_uri)
    parsed = parse_model_uri(model_uri)

    if parsed.name is None:
        # runs:/ URI or local path: no registry to ask.
        name, version, alias = UNREGISTERED, UNREGISTERED, None
        run_id = _run_id_from_runs_uri(model_uri)
        load_uri = model_uri
    else:
        client = MlflowClient()
        if parsed.alias is not None:
            model_version = client.get_model_version_by_alias(parsed.name, parsed.alias)
        else:
            model_version = client.get_model_version(parsed.name, parsed.version)

        name, version, alias = parsed.name, str(model_version.version), parsed.alias
        run_id = model_version.run_id or UNKNOWN
        # Load the version the alias resolved to, not the alias itself, so the
        # version we report is guaranteed to be the one we loaded even if the
        # alias moves between these two calls.
        load_uri = f"{REGISTRY_PREFIX}{name}/{version}"

    model = mlflow.sklearn.load_model(load_uri)
    check_classes(model)

    logger.info(
        "Loaded model %s version %s (alias=%s, run_id=%s) from %s",
        name,
        version,
        alias,
        run_id,
        model_uri,
    )
    return LoadedModel(
        model=model,
        model_name=name,
        model_version=version,
        model_alias=alias,
        run_id=run_id,
        model_uri=model_uri,
    )