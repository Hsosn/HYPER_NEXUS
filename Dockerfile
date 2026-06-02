# ──────────────────────────────────────────────────────────────
# Hyper-Nexus — single-image runtime
# Runs the FastAPI web server + the Celery worker + the Celery beat
# scheduler in one container. Suitable for self-hosting and
# "I just want to try it" docker-compose setups.
# ──────────────────────────────────────────────────────────────

# ── Stage 1: build a wheel cache so the runtime stage stays small ───────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System deps for some wheels (cryptography, lxml, pillow) and a sane
# shell so the entrypoint script can supervise processes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        libjpeg-dev \
        libpng-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt


# ── Stage 2: minimal runtime ────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NEXUS_HOST=0.0.0.0 \
    NEXUS_PORT=8000 \
    REDIS_URL=redis://redis:6379/0

# tini gives us proper signal handling for the supervised processes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash nexus

WORKDIR /app

# Pull in the prebuilt deps from the builder stage.
COPY --from=builder /install /usr/local

# Application code. We copy the whole repo — `nexus/`, `webui/`, skill
# folders, the entrypoint, and the env example.
COPY --chown=nexus:nexus nexus/         ./nexus/
COPY --chown=nexus:nexus webui/         ./webui/
COPY --chown=nexus:nexus nexus_ml_skills/ ./nexus_ml_skills/
COPY --chown=nexus:nexus nexus3d_skills/  ./nexus3d_skills/
COPY --chown=nexus:nexus requirements.txt ./
COPY --chown=nexus:nexus .env.example     ./.env.example
COPY --chown=nexus:nexus LICENSE          ./LICENSE
COPY --chown=nexus:nexus README.md         ./README.md
COPY --chown=nexus:nexus docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Pre-create the sandboxed workspace + a place for SQLite, logs, and
# the .env the user is expected to mount or `cp` in.
RUN mkdir -p /app/data/workspace /app/data/db /app/data/logs \
    && chown -R nexus:nexus /app/data

USER nexus

EXPOSE 8000

# Lightweight healthcheck — hits the OpenAPI schema endpoint, which
# exists in every build of the app.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${NEXUS_PORT}/openapi.json" >/dev/null \
        || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
CMD ["web", "worker", "beat"]
