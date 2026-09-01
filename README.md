[Русский](#credit-risk-ml-service) | [English](#en)

# Credit Risk ML Service

[![CI](https://github.com/Platzei/credit-risk-ml-service/actions/workflows/ci.yml/badge.svg)](https://github.com/Platzei/credit-risk-ml-service/actions/workflows/ci.yml)
![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)

Сервис оценки риска серьезной кредитной просрочки на датасете Kaggle **Give Me Some Credit**. Репозиторий включает загрузку данных, обучение и оценку моделей, сохранение CatBoost, FastAPI, Docker, тесты и CI.

Стек: Python 3.14, pandas, scikit-learn, CatBoost, FastAPI, Pydantic, Docker, pytest, Ruff и GitHub Actions.

## Результаты и основные решения

Однократная оценка выбранной некалиброванной модели CatBoost на отложенной тестовой выборке:

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.8624 |
| PR-AUC | 0.3917 |
| Brier score | 0.0494 |
| Log loss | 0.1793 |

Результаты на test при пороге `0.09`, выбранном заранее на validation:

| precision | recall | F1 | TP | FP | TN | FN | Условная стоимость |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2594 | 0.7041 | 0.3791 | 1059 | 3024 | 17972 | 445 | 7474 |

- Logistic Regression использовалась как baseline.
- CatBoost превзошел baseline на validation и был выбран как финальное семейство моделей.
- Sigmoid calibration проверялась на отдельной выборке, но ухудшила Brier score и Log loss. Поэтому используются некалиброванные вероятности CatBoost.
- Порог `0.09` выбран только на validation по минимуму `FP × 1 + FN × 10`. При пороге `0.50` условная стоимость на test составила `12532` против `7474` при `0.09`.
- `FP cost = 1` и `FN cost = 10` — иллюстративные относительные единицы, а не реальные денежные потери банка.
- Test использовался только для финальной оценки, не для выбора модели, калибровки, порога или гиперпараметров.
- После фиксации решений финальный `CatBoostClassifier` обучен на полной исходной обучающей выборке с 316 итерациями бустинга. Порядок признаков и threshold `0.09` сохранены в JSON metadata.

## Структура проекта

```text
credit-risk-ml-service/
├── .github/workflows/ci.yml
├── artifacts/                  # Локальные model artifacts; исключены из Git
├── data/raw/                   # CSV с Kaggle; исключен из Git
├── notebooks/                  # Исследовательский анализ
├── src/credit_risk/
│   ├── api.py                  # FastAPI
│   ├── artifacts.py            # Сохранение и загрузка модели
│   ├── data.py                 # Загрузка и проверка данных
│   ├── evaluate.py             # Метрики и пороговый анализ
│   └── train.py                # Обучение и создание artifacts
├── tests/
├── Dockerfile
└── pyproject.toml
```

## Локальная установка

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

На macOS или Linux: `source .venv/bin/activate`.

## Датасет и обучение

Скачайте `cs-training.csv` из Kaggle **Give Me Some Credit** и сохраните его как `data/raw/cs-training.csv`. Target — `SeriousDlqin2yrs`; значение `1` означает серьезную просрочку в течение двух лет. Исходные данные не коммитятся и не нужны для тестов.

```powershell
python -m credit_risk.train
```

Команда сохраняет `artifacts/catboost_model.cbm` и `artifacts/metadata.json`. Сгенерированные model artifacts не коммитятся.

## API

Модель и метадата загружаются один раз при старте приложения. Поля `MonthlyIncome` и `NumberOfDependents` могут быть `null`.

```powershell
python -m uvicorn credit_risk.api:app --reload
```

Endpoints: `GET /health`, `POST /predict`; Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs).

Пример запроса к `POST /predict` из PowerShell:

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{"RevolvingUtilizationOfUnsecuredLines":0.42,"age":45,"NumberOfTime30-59DaysPastDueNotWorse":1,"DebtRatio":0.35,"MonthlyIncome":null,"NumberOfOpenCreditLinesAndLoans":8,"NumberOfTimes90DaysLate":0,"NumberRealEstateLoansOrLines":1,"NumberOfTime60-89DaysPastDueNotWorse":0,"NumberOfDependents":null}'
```

Пример ответа:

```json
{
  "serious_delinquency_probability": 0.052092103518956386,
  "high_risk": false,
  "threshold": 0.09
}
```

## Docker

Образ использует существующие artifacts и не запускает обучение. Перед сборкой выполните training command выше.

```powershell
docker build -t credit-risk-ml-service .
docker run --rm -p 8000:8000 credit-risk-ml-service
curl.exe http://localhost:8000/health
```

## Тесты и CI

```powershell
python -m pytest
python -m ruff check .
```

GitHub Actions запускает pytest и Ruff с Python 3.14 при push в `main` и для pull request. CI не требует датасет или model artifacts, не обучает модель и не собирает Docker-образ.

## Ограничения

- Датасет не содержит реальных финансовых потерь и полного контекста кредитования.
- Условные стоимости не отражают реальную экономику банка, а порог `0.09` не является готовой кредитной политикой.
- Модель не подходит для реальных кредитных решений без дополнительной валидации, анализа справедливости, управления моделью, проверки безопасности и экспертизы в предметной области.
- В API нет аутентификации, базы данных и мониторинга.

---

<a id="en"></a>

# Credit Risk ML Service — English version

Credit delinquency risk service built on Kaggle's **Give Me Some Credit** dataset. The repository contains model training and evaluation, persisted CatBoost artifacts, a FastAPI, Docker support, pytest and Ruff checks, and GitHub Actions CI. It uses Python 3.14, pandas, scikit-learn, CatBoost, FastAPI, and Pydantic.

## Results

| Metric | Held-out test |
|---|---:|
| ROC-AUC | 0.8624 |
| PR-AUC | 0.3917 |
| Brier score | 0.0494 |
| Log loss | 0.1793 |

Logistic Regression was the baseline. CatBoost performed better on validation and was selected as the final model family. Sigmoid calibration worsened Brier score and Log loss, so inference uses uncalibrated probabilities.

Threshold `0.09` was selected on validation using illustrative assumptions `FP cost = 1` and `FN cost = 10`. Test cost was `7474` at `0.09`, versus `12532` at `0.50`. These units do not represent real bank economics. The held-out test set was used only for final evaluation. The final artifact was trained on the full original training partition with 316 boosting iterations.

## Setup and commands

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Place the uncommitted Kaggle file at `data/raw/cs-training.csv`, then train the model. Generated model artifacts are also not committed.

```powershell
python -m credit_risk.train
python -m uvicorn credit_risk.api:app --reload
```

The API exposes `GET /health`, `POST /predict`, and [http://localhost:8000/docs](http://localhost:8000/docs).

The copy-paste `curl.exe` example in the Russian API section calls the same `POST /predict` endpoint with all ten model features.

Build only after `artifacts/` exists locally:

```powershell
docker build -t credit-risk-ml-service .
docker run --rm -p 8000:8000 credit-risk-ml-service
```

## Limitations

- The cost assumptions are illustrative; threshold `0.09` is not a real lending policy.
- The model is not suitable for real lending decisions without further validation, fairness analysis, model governance, security review, and domain review.
- The API has no authentication, database, or monitoring.
