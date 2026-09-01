FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CREDIT_RISK_ARTIFACT_DIR=/app/artifacts

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir .

COPY artifacts ./artifacts

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "credit_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
