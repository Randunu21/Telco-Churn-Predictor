# Telco-Churn-Predictor


telco-churn-mlops/
├── README.md
├── .gitignore
├── .python-version
├── requirements/
│   ├── base.txt          # shared deps (pandas, pydantic, etc.)
│   ├── train.txt         # training-specific (lightgbm/xgboost, mlflow, pandera)
│   ├── api.txt           # serving-specific (fastapi, uvicorn)
│   └── dev.txt           # pytest, linters, notebook tools
│
├── data/
│   ├── raw/               # gitignored, DVC-tracked
│   ├── processed/         # gitignored, DVC-tracked
│   └── .gitkeep
│
├── src/
│   └── telco_churn/
│       ├── __init__.py
│       ├── data/          # loading, validation (Pandera schemas)
│       ├── features/      # feature engineering
│       ├── models/        # training, evaluation logic
│       ├── api/            # FastAPI app (added Phase 4)
│       └── config.py       # central config/paths
│
├── notebooks/              # exploration only, never imported by src/
│
├── tests/
│   ├── data/
│   ├── models/
│   └── api/
│
├── configs/                 # yaml/json configs (model params, paths)
│
├── docker/                  # Dockerfiles, added Phase 5
├── .github/workflows/       # CI/CD, added Phase 6
├── k8s/                     # manifests, added Phase 7
├── airflow/                 # dags/, added Phase 8
│
├── mlruns/                  # gitignored — MLflow local tracking
├── .dvc/                    # created by `dvc init`, Phase 1
└── .dvcignore