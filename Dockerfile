# ==============================================================================
# Multi-Stage Production Dockerfile — Mandya Agro-Weather (SIH-26074)
# Stage 1: Build & wheels caching
# Stage 2: Minimal slim runtime with non-root security & healthcheck
# ==============================================================================

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install --no-warn-script-location -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Production Runner
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install runtime libraries and curl for docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root user for cloud production hardening
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code and required serving artifacts
COPY src/ /app/src/
COPY frontend/ /app/frontend/
COPY data/serving/ /app/data/serving/
COPY models/checkpoints/ /app/models/checkpoints/

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Healthcheck testing API readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/forecast/215504 || exit 1

# Production ASGI server with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
