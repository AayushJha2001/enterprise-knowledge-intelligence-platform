# ---- base: runtime dependencies only ----
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10
WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- dev: adds test/lint tools; source is bind-mounted by docker compose ----
FROM base AS dev
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ---- prod: minimal image, non-root user ----
FROM base AS prod
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/uploads /data/models \
    && chown -R appuser:appuser /data
ENV UPLOAD_DIR=/data/uploads \
    MODEL_CACHE_DIR=/data/models
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers"]
