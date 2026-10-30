# syntax=docker/dockerfile:1
#
# Telco Churn API image: multi-stage build.
# Build from the REPO ROOT (the build context must contain pyproject.toml,
# requirements/ and src/):
#
#   docker build -f docker/api.Dockerfile -t telco-churn-api:dev .
#
# Configuration comes from environment variables at runtime (MODEL_URI,
# MLFLOW_TRACKING_URI, DECISION_THRESHOLD, ...). No .env is baked into the image.

# Pinned to a Debian release so a new Debian version can't silently change
# system libraries. Must match the Python used for training and tests (3.12).
ARG PYTHON_IMAGE=python:3.12-slim-bookworm


# ---------------------------------------------------------------------------
# Stage 1: builder: installs everything into a self-contained venv
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Same path as in the runtime stage: venv scripts (e.g. bin/uvicorn) hard-code it.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# 1) Dependencies first: this slow layer stays cached until requirements change.
COPY requirements/ requirements/
RUN pip install -r requirements/api.txt

# 2) Our package last: editing a .py file only re-runs this cheap step.
#    Non-editable (the runtime stage has no source tree), and --no-deps because
#    every dependency is already pinned by api.txt above.
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-deps .


# ---------------------------------------------------------------------------
# Stage 2: runtime: only the finished venv, no build tools, no source
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# MLFLOW_DISABLE_TELEMETRY: no usage telemetry from the serving container.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_TELEMETRY=true

# Non-root user with a fixed UID (Kubernetes runAsNonRoot can check it in Phase 7).
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --home-dir /home/app \
            --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER app

EXPOSE 8000

# Exec form: uvicorn is PID 1 and receives SIGTERM directly (clean shutdown).
# 0.0.0.0 so the port is reachable from outside the container.
# No --reload / --workers: one process per container; scale with replicas.
CMD ["uvicorn", "telco_churn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
