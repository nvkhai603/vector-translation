# Use an official Python runtime with CUDA support for GPU acceleration
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as gpu-base

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Fallback CPU-only image for environments without GPU
FROM python:3.11-slim as cpu-base

# Multi-stage build - use GPU base if available, fallback to CPU
FROM gpu-base as runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install PyTorch with CUDA support and other requirements
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV PORT=5000
ENV CUDA_VISIBLE_DEVICES=0

# Run the application using Gunicorn with GPU-optimized settings
CMD ["sh", "-c", "gunicorn --workers 2 --worker-class sync --timeout 300 --bind 0.0.0.0:$PORT clip_api:app"]
