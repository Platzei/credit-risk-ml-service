# Credit Risk ML Service

## Project goal

Build a reproducible end-to-end machine learning system for predicting serious credit delinquency, covering model development, evaluation, and eventual deployment as a small inference service.

## Dataset

The project uses Kaggle's **Give Me Some Credit** dataset. The target column is `SeriousDlqin2yrs`, and the expected training dataset path is `data/raw/cs-training.csv`. Raw data is not tracked by Git.

## Planned pipeline

The planned workflow covers reproducible data loading, leakage-safe preprocessing, CatBoost training, evaluation with ROC-AUC as the primary comparison metric and PR-AUC as a secondary metric, model serving through FastAPI, containerization with Docker, automated tests, and continuous integration.

## Current status

Data loading, baseline training and evaluation, CatBoost model persistence, and a minimal FastAPI inference service are implemented. Containerization and CI have not been implemented.

## Local setup

Python 3.11 or newer is required. Create and activate a virtual environment, then install the project and development dependencies:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Place the Kaggle training CSV at `data/raw/cs-training.csv`, then train the model and create the ignored artifacts:

```powershell
python -m credit_risk.train
```

Start the inference service after the artifacts have been created:

```powershell
python -m uvicorn credit_risk.api:app --reload
```

## Docker

Build the inference image from the existing model artifacts:

```powershell
docker build -t credit-risk-ml-service .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 credit-risk-ml-service
```

Open the interactive API documentation at [http://localhost:8000/docs](http://localhost:8000/docs). Check service health with:

```powershell
curl.exe http://localhost:8000/health
```
