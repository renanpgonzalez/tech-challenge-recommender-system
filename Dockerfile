FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
RUN pip install --no-cache-dir --upgrade pip poetry

WORKDIR /build

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root \
    && pip uninstall -y torch triton 2>/dev/null; \
       pip freeze | grep -i "^nvidia" | cut -d= -f1 | xargs -r pip uninstall -y 2>/dev/null; \
       pip install --no-cache-dir "torch==2.12.0+cpu" \
         --index-url https://download.pytorch.org/whl/cpu


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/workspace/src \
    MLFLOW_SERVER_ALLOWED_HOSTS="*"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /opt/venv /opt/venv
COPY . .

CMD mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT:-mlruns}" \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 1 \
    --cors-allowed-origins "${MLFLOW_CORS_ORIGINS:-http://localhost:5000}"
