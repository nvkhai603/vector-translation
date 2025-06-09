# ========= Stage 1: Base Layer =========
# Common base to be extended by both GPU and CPU variants
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    && ln -s -f /usr/bin/python3.11 /usr/bin/python \
    && ln -s -f /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ========= Stage 2: GPU Runtime =========
FROM base AS gpu

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ========= Stage 3: CPU Runtime =========
FROM base AS cpu

# Install PyTorch CPU-only
RUN pip install --no-cache-dir \
    torch torchvision

# ========= Stage 4: Final Runtime (Selectable) =========
FROM gpu AS final

WORKDIR /app

# Copy requirements and install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Healthcheck for container
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose port
EXPOSE 5000

# Set environment variables
ENV PORT=5000
ENV CUDA_VISIBLE_DEVICES=0

# Start application with Gunicorn
CMD ["sh", "-c", "gunicorn --workers 2 --worker-class sync --timeout 300 --bind 0.0.0.0:$PORT clip_api:app"]
