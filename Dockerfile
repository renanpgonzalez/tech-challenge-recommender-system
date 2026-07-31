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
    && pip install --no-cache-dir --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/workspace/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /opt/venv /opt/venv
COPY . .

CMD ["python", "scripts/validate_env.py"]
