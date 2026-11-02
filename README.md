# Telco Churn Predictor

An end-to-end MLOps portfolio project: a customer-churn model for the Kaggle
Telco Customer Churn dataset, built as a production-style system rather than a
notebook. The focus is the engineering around the model: versioned data,
experiment tracking and a model registry, tests, a validated REST API, and
containers, with CI/CD, Kubernetes, scheduled retraining and monitoring to come.

Everything is self-hosted. No hyperscaler cloud services (AWS / GCP / Azure /
Databricks) are used.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Project setup (src-layout package, pinned requirements) | Done |
| 1 | Data: DVC versioning (Backblaze B2 remote), Pandera validation | Done |
| 2 | Training: scikit-learn / XGBoost pipelines, MLflow tracking + registry | Done |
| 3 | Testing: pytest suite (data, model and API tests) | Done |
| 4 | Serving: FastAPI + Pydantic, model loaded from the MLflow registry | Done |
| 5 | Containers: multi-stage Docker images, Docker Compose stack | Done |
| 6 | CI/CD: GitHub Actions, images pushed to ghcr.io | Next |
| 7 | Kubernetes: k3s, ingress-nginx, cert-manager | Planned |
| 8 | Airflow retraining pipeline | Planned |
| 9 | Monitoring: Prometheus + Grafana | Planned |
| 10 | Deployment: local k3s, optional VPS | Planned |

## Architecture (Docker Compose)

```
            your machine / Codespace
  ┌────────────────────────────────────────────────────────┐
  │  :8000 ─────────► api       FastAPI, loads the model   │
  │                    │        models:/telco-churn@champion│
  │                    ▼  HTTP (registry + artifact proxy) │
  │  127.0.0.1:5000 ─► mlflow   tracking server + UI       │
  │                    │   └──► volume: mlartifacts         │
  │                    ▼  SQL (psycopg2)                    │
  │  (not published)   postgres ──► volume: pgdata          │
  └────────────────────────────────────────────────────────┘
```

- **api**: serves predictions. On startup it resolves the `champion` alias to a
  concrete model version and loads it once. It fails fast if it can't.
- **mlflow**: experiments, model registry, and an **artifact proxy**, so models
  are uploaded and downloaded over HTTP through MLflow. The API never touches
  the artifact storage or the database, and holds no database credentials.
- **postgres**: MLflow's backend store. Only reachable from inside the Compose
  network.
- **Ports**: MLflow has no authentication, so it is published on `127.0.0.1`
  only (Docker-published ports bypass host firewalls such as `ufw`). Postgres
  is not published at all.
- **Start order** is enforced with healthchecks: Postgres (`pg_isready`), then
  MLflow (`/health`), then the API (`/health/ready`).

Compose is for local development, reviewers and CI integration tests.
Kubernetes (Phase 7) is the deployment target.

## Run with Docker Compose

### Prerequisites

- Docker with the Compose v2 plugin (`docker compose version`)
- Only for training: Python 3.12 and the dataset (`dvc pull`, needs the DVC
  remote credentials)

### 1. Configure credentials

```bash
cp compose.env.example compose.env      # compose.env is git-ignored
openssl rand -hex 24                     # generate a password
```

Put the generated password into **both** places in `compose.env`:
`POSTGRES_PASSWORD` **and** inside `MLFLOW_BACKEND_STORE_URI`. They must be
identical. If they differ, MLflow can't log in to Postgres and keeps
restarting (`docker compose ps` shows it as unhealthy).

Notes:
- Use letters and digits only. Characters such as `@ / :` would need
  URL-encoding inside the URI.
- Postgres applies `POSTGRES_*` only on the **first** start (empty volume).
  Changing the password later in `compose.env` has no effect on an existing
  database.
- Keep the explicit `postgresql+psycopg2://` driver in the URI (see
  [Design notes](#design-notes)).

### 2. Start the stack

```bash
docker compose up -d --build
docker compose ps        # wait until postgres and mlflow are (healthy)
```

### 3. First start only: register a champion model

A fresh stack has an empty registry, so the API exits ("Registered Model with
name=telco-churn not found") and Docker keeps restarting it. Train and register
a model once:

```bash
pip install -e . -r requirements/dev.txt     # if not done yet
dvc pull                                       # dataset
python -m telco_churn.models.train             # logs an XGBoost and an LR run to localhost:5000
```

Then in the MLflow UI at <http://localhost:5000>:
1. Experiment **telco-churn**: compare the runs on `test_pr_auc`.
2. Open the best run's model, then **Register model** with the name **`telco-churn`**.
3. **Models → telco-churn → Version 1**, then add the alias **`champion`**.

```bash
docker compose restart api
docker compose ps        # api becomes (healthy)
```

The registry and models live in Docker volumes, so this is only needed once.

### 4. Try it

```bash
curl -s localhost:8000/v1/model-info
curl -s -X POST localhost:8000/v1/predict \
  -H 'Content-Type: application/json' -d @examples/customer.json
```

Interactive API docs: <http://localhost:8000/docs>

| Endpoint | Purpose |
|---|---|
| `POST /v1/predict` | Score one customer |
| `POST /v1/predict/batch` | Score up to 500 customers (`{"customers": [...]}`) |
| `GET /v1/model-info` | Model name, version, alias, run ID, threshold |
| `GET /health/live` | Liveness (never touches the model) |
| `GET /health/ready` | Readiness (503 until the model is loaded) |

`churn_score` is a model score, not a calibrated probability: the model is
trained on SMOTE-balanced data.

### 5. Everyday commands

```bash
docker compose ps                    # status and health
docker compose logs -f api           # follow one service's logs
docker compose up -d --build api     # rebuild after code changes
docker compose restart api           # reload after promoting a new champion
docker compose down                  # stop everything, KEEP the data
```

> **Warning:** `docker compose down -v` also deletes the volumes, which means
> the database and every stored model are gone for good. Only use it to start
> from scratch on purpose.

## Local development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt
pip install -e .
python -m pytest            # use python -m, so the editable install is found
```

Start an MLflow server and the API locally:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
uvicorn telco_churn.api.main:app --port 8000 --reload
```

Only one MLflow can listen on port 5000: stop the Compose stack first
(`docker compose stop mlflow`), or use the Compose MLflow directly.

## Configuration

The API is configured through environment variables (pydantic-settings,
`src/telco_churn/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server (`http://mlflow:5000` in Compose) |
| `MODEL_URI` | `models:/telco-churn@champion` | Registry alias, pinned version (`models:/telco-churn/1`), `runs:/` URI or local path |
| `DECISION_THRESHOLD` | `0.5` | Score at or above which the prediction is "Yes" |
| `MAX_BATCH_SIZE` | `500` | Maximum customers per batch request |

Aliases are resolved at startup only: after moving `champion` to a new version,
restart the API.

## Project layout

```
├── compose.yaml              # Postgres + MLflow + API stack
├── compose.env.example       # credentials template (copy to compose.env)
├── docker/
│   ├── api.Dockerfile        # multi-stage, non-root, mlflow-skinny
│   └── mlflow.Dockerfile     # MLflow server + Postgres driver
├── examples/
│   └── customer.json         # sample request body
├── requirements/
│   ├── base.txt              # shared: pandas, numpy, pydantic
│   ├── model.txt             # everything needed to load the pipeline (train AND serve)
│   ├── mlflow.txt            # the single full-MLflow pin
│   ├── train.txt             # base + model + mlflow + pandera + dvc
│   ├── api.txt               # base + model + fastapi + uvicorn + mlflow-skinny
│   ├── mlflow-server.txt     # mlflow + psycopg2 (server image only)
│   └── dev.txt               # train + api + pytest, linters, jupyter
├── data/raw/                 # DVC-tracked (only the .dvc file is in git)
├── src/telco_churn/
│   ├── config.py             # central settings (pydantic-settings)
│   ├── data/                 # loading, cleaning, splitting, Pandera schema
│   ├── features/             # preprocessing (ColumnTransformer, SMOTENC)
│   ├── models/               # pipeline + training with MLflow logging
│   └── api/                  # FastAPI app: schemas, loader, routes, errors
├── tests/                    # mirrors src/: data, features, models, api
└── notebooks/                # exploration only, never imported by src/
```

## Design notes

- **One model pin for training and serving.** `model.txt` is included by both
  `train.txt` and `api.txt`, so the libraries that load the model can't drift
  apart. A containerized API, the local API and a direct pipeline call produce
  identical scores.
- **Reproducible training.** Fixed seeds (split, SMOTENC, classifiers), a
  DVC-versioned dataset and pinned libraries: retraining from scratch against a
  brand-new MLflow (Postgres) reproduced the original champion's scores exactly.
- **MLflow client and server versions match.** `mlflow.txt` pins full MLflow
  (training and server); the API uses `mlflow-skinny` at the same version.
  `pip install -r requirements/dev.txt` fails if the two ever diverge, because
  full MLflow requires that exact skinny version.
- **Lean, non-root API image.** Multi-stage build; `mlflow-skinny` instead of
  full MLflow cut the image from 670 MB to 554 MB (compressed). Runs as UID
  10001, exec-form `CMD` for clean `SIGTERM` shutdown, MLflow telemetry
  disabled.
- **Postgres driver: psycopg2, named explicitly.** SQLAlchemy 2.1 made psycopg 3
  the default for a bare `postgresql://`, but MLflow 3.15.2's model registry
  fails on psycopg 3 when setting an alias (`operator does not exist: integer =
  character varying`). The URI therefore always says `postgresql+psycopg2://`.
- **Fail fast at startup.** The API refuses to start without a valid model and
  lets the orchestrator (Compose restart policy, later Kubernetes) retry.
  MLflow's client retries connection errors for up to ~4 minutes before the
  API gives up; set `MLFLOW_HTTP_REQUEST_MAX_RETRIES=0` when debugging for
  instant feedback.

## Tech stack

DVC · Pandera · scikit-learn · imbalanced-learn · XGBoost · MLflow · pytest ·
FastAPI · Pydantic · Docker · Docker Compose · PostgreSQL · *(next)* GitHub
Actions · ghcr.io · k3s · Airflow · Prometheus · Grafana
