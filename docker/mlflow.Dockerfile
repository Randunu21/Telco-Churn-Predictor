# syntax=docker/dockerfile:1
#
# MLflow tracking server image (experiments, model registry, artifact proxy).
# Build from the REPO ROOT:
#
#   docker build -f docker/mlflow.Dockerfile -t telco-churn-mlflow:dev .
#
# Single stage on purpose: every dependency ships as a prebuilt wheel
# (psycopg2-binary bundles libpq), so there is nothing to compile and no build
# tools to leave behind. No project code goes in this image.
#
# All settings are environment variables read by `mlflow server`, so Compose /
# Kubernetes can override any of them. The defaults below let the image run on
# its own with SQLite; Compose (5.6) switches the backend to Postgres:
#   MLFLOW_BACKEND_STORE_URI=postgresql://<user>:<password>@<host>:5432/<db>
# and must add the service name to the allowed Host headers, e.g.
#   MLFLOW_SERVER_ALLOWED_HOSTS=mlflow:5000,localhost:*,...
# (Setting it REPLACES MLflow's default list of localhost + private IPs.)

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only the requirement files this image needs (mlflow-server.txt includes mlflow.txt).
COPY requirements/mlflow.txt requirements/mlflow-server.txt /tmp/requirements/
RUN pip install -r /tmp/requirements/mlflow-server.txt \
 && rm -rf /tmp/requirements

# Non-root user, same UID as the API image.
# /mlflow is created here and owned by that user so that a fresh Docker volume
# mounted on it (5.5) inherits the ownership. Otherwise the first artifact
# upload fails with "Permission denied".
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --home-dir /home/app \
            --shell /usr/sbin/nologin app \
 && mkdir -p /mlflow/artifacts \
 && chown -R app:app /mlflow

# Server settings (each is the env var behind an `mlflow server` CLI option).
# MLFLOW_DISABLE_TELEMETRY: no usage telemetry from the server container.
ENV MLFLOW_HOST=0.0.0.0 \
    MLFLOW_PORT=5000 \
    MLFLOW_WORKERS=2 \
    MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db \
    MLFLOW_ARTIFACTS_DESTINATION=/mlflow/artifacts \
    MLFLOW_DISABLE_TELEMETRY=true

WORKDIR /mlflow
USER app

EXPOSE 5000

# Exec form, no --reload. Options come from the ENV block above.
CMD ["mlflow", "server"]
