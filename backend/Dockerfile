# ==============================================================================
# Backend Lead Container: FastAPI Server & Geospatial Processing
# ==============================================================================
FROM python:3.11-slim

# Prevent interactive prompts & force unbuffered output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    AIR_GAPPED_MODE=1

# Install system dependencies for PostgreSQL/PostGIS and GDAL geospatial tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gdal-bin \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cloud/staging builds install the declared dependencies. For a fully
# air-gapped image, build from a base image with an internal package mirror.
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend source code into container
COPY . /app/

# Expose Uvicorn server port
EXPOSE 8000

# Health check targeting root endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start Uvicorn Server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
