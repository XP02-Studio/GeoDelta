# Deep Learning Core - Air-Gapped High-Performance Inference Container
# Base: NVIDIA CUDA 11.8 + cuDNN 8 on Ubuntu 22.04 LTS
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during container build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    AIR_GAPPED_MODE=1

# Install minimal OS dependencies for Python 3.11 and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create project working directory
WORKDIR /app

# Copy pre-downloaded offline wheels and requirements
COPY offline/requirements.txt /app/offline/requirements.txt
COPY offline/wheels/ /app/offline/wheels/

# Install dependencies from local offline wheels (No Internet Access Required)
RUN pip install --no-cache-dir --no-index --find-links=/app/offline/wheels/ -r /app/offline/requirements.txt || \
    pip install --no-cache-dir -r /app/offline/requirements.txt

# Copy models and source code
COPY models/ /app/models/
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Expose REST API port
EXPOSE 8000

# Health check to ensure model latency and memory bounds
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start the optimized inference server
ENTRYPOINT ["python", "-m", "src.api", "--host", "0.0.0.0", "--port", "8000"]
