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

# ---- prod: minimal image, non-root user, models baked in ----
FROM base AS prod
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/uploads /opt/models \
    && chown -R appuser:appuser /data /opt/models
ENV ENVIRONMENT=production \
    UPLOAD_DIR=/data/uploads \
    MODEL_CACHE_DIR=/opt/models \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    RERANKER_MODEL=${RERANKER_MODEL} \
    HF_HUB_OFFLINE=1
USER appuser
# Download the models at build time (before copying the code, so code changes don't
# re-download them): containers start without network access or a cold-start download.
RUN HF_HUB_OFFLINE=0 python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('${EMBEDDING_MODEL}', cache_dir='/opt/models'); \
TextCrossEncoder('${RERANKER_MODEL}', cache_dir='/opt/models')"
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser migrations ./migrations
COPY --chown=appuser:appuser app ./app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4)"
# One process per container (each loads the models, ~0.5 GB): scale with replicas.
# Migrations run as a separate one-off job (`alembic upgrade head`), not on every start.
# Proxy headers are trusted because the app runs behind a load balancer / ingress; don't
# expose this port directly to the internet without one.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips=*"]
