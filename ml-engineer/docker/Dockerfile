# Multi-stage air-gapped Dockerfile for Deep Learning Core
# Base image: Official NVIDIA CUDA 11.8 Runtime with cuDNN 8 on Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Deep Learning Core ML Engineer <ml-core@antigravity.ai>"
LABEL description="Air-gapped ChangeFormer and RS-CLIP inference service with TensorRT & FP16 optimizations"

# Set environment variables for non-interactive installs and offline runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ALLOW_INTERNET=0 \
    CUDA_VISIBLE_DEVICES=0 \
    PORT=8000

# Install system dependencies (OpenCV, GL, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python aliases
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /workspace

# Copy pre-downloaded offline wheels and requirements
COPY offline/ /workspace/offline/
COPY offline/requirements.txt /workspace/requirements.txt

# Install dependencies strictly from offline wheels if available, or requirements
RUN if [ -d "/workspace/offline/wheels" ] && [ "$(ls -A /workspace/offline/wheels)" ]; then \
        pip install --no-cache-dir --no-index --find-links=/workspace/offline/wheels -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy pre-downloaded model checkpoints and source code
COPY models/ /workspace/models/
COPY src/ /workspace/src/
COPY cli.py /workspace/cli.py
COPY benchmark.py /workspace/benchmark.py

# Expose API port
EXPOSE 8000

# Healthcheck to verify offline model warm-up status
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch the optimized production inference server
CMD ["python", "-m", "src.server", "--host", "0.0.0.0", "--port", "8000", "--use-fp16"]
