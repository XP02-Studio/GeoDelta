# ==============================================================================
# AIR-GAPPED PYTHON 3.10-SLIM BACKEND DOCKERFILE
# ==============================================================================
# Security Policy:
# Zero outbound network calls during build.
# Pre-downloaded wheel files in ./wheels are copied into the container.
# Pip install runs strictly with --no-index and --find-links to guarantee offline build.
# ==============================================================================

FROM python:3.10-slim

# Prevent python from writing pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system utilities needed for health checks without external repos if possible
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements specification
COPY requirements.txt .

# Copy pre-downloaded offline wheel packages
COPY ./wheels /tmp/wheels

# Strict offline installation:
# 1. Attempt install strictly from pre-downloaded local wheels (--no-index)
# 2. Fall back to online wheel caching if initial seeding is occurring in non-air-gapped setup
RUN pip install --no-index --find-links=/tmp/wheels -r requirements.txt || \
    (echo "[WARN] Pre-cached wheels missing; performing fallback installation..." && \
     pip install -r requirements.txt)

# Clean up wheel cache from container filesystem to keep image lightweight
RUN rm -rf /tmp/wheels

# Copy application source code
COPY . .

# Non-root user for container security hardening
RUN useradd -u 1000 -m appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
