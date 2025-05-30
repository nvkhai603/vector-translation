# Vector Translation API

API Flask để trích xuất vector từ hình ảnh sử dụng CLIP và ResNet-50 models với hỗ trợ GPU.

## 🚀 Tính năng

### ✅ **GPU Support**
- Tự động phát hiện và sử dụng GPU (CUDA) nếu có sẵn
- Fallback sang CPU nếu GPU không khả dụng
- Optimized memory management và GPU utilization

### ✅ **Models hỗ trợ**
- **CLIP Model**: `patrickjohncyh/fashion-clip` - Vector embeddings cho image + text
- **ResNet-50**: Feature extraction từ hình ảnh

### ✅ **API Endpoints**
- `POST /get_clip_vector` - Trích xuất CLIP vector
- `POST /get_resnet_vector` - Trích xuất ResNet-50 vector  
- `GET /health` - Health check và resource monitoring

### ✅ **Fault Tolerance**
- Comprehensive error handling
- Memory monitoring và automatic cleanup
- Health check endpoints
- Graceful degradation

## 📋 Requirements

### GPU Environment
```bash
# NVIDIA Docker runtime
nvidia-docker2
# hoặc Docker với nvidia-container-toolkit

# CUDA 12.1+ compatible GPU
# Minimum 4GB GPU memory recommended
```

### CPU Fallback
```bash
# Standard Docker installation
docker >= 20.10
docker-compose >= 1.29
```

## 🛠 Installation & Usage

### 1. Kiểm tra GPU Support
```bash
# Kiểm tra GPU
nvidia-smi

# Kiểm tra NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi
```

### 2. Build và Run với GPU
```bash
# Clone repository
git clone <repo-url>
cd vector-translation

# Build và run với GPU
docker-compose up --build

# Hoặc chỉ GPU service
docker-compose up vector-translation-gpu
```

### 3. CPU Fallback
```bash
# Nếu không có GPU, sử dụng CPU version
docker-compose --profile cpu-fallback up vector-translation-cpu
```

### 4. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Download models (nếu cần)
python download_model.py

# Run local
python clip_api_local.py
```

## 📊 API Usage

### CLIP Vector Extraction
```bash
# Với file upload
curl -X POST http://localhost:5000/get_clip_vector \
  -F "image_file=@sample.jpg" \
  -F "text=fashion clothing"

# Với URL
curl -X POST http://localhost:5000/get_clip_vector \
  -F "image_url=https://example.com/image.jpg" \
  -F "text=fashion clothing"
```

### ResNet-50 Vector Extraction  
```bash
# Với file upload
curl -X POST http://localhost:5000/get_resnet_vector \
  -F "image_file=@sample.jpg"

# Với URL
curl -X POST http://localhost:5000/get_resnet_vector \
  -F "image_url=https://example.com/image.jpg"
```

### Health Check & Monitoring
```bash
# Health check
curl http://localhost:5000/health

# Response example:
{
  "status": "healthy",
  "models": {
    "clip_model_loaded": true,
    "resnet_model_loaded": true,
    "device": "cuda:0"
  },
  "resources": {
    "cpu_percent": 15.2,
    "memory_percent": 45.8,
    "memory_available_gb": 8.2,
    "memory_used_gb": 7.8,
    "gpu_memory_used_gb": 2.1,
    "gpu_memory_total_gb": 8.0,
    "gpu_memory_percent": 26.25,
    "gpu_utilization_percent": 12
  },
  "warnings": []
}
```

## 🔧 Configuration

### Environment Variables
```bash
PORT=5000                    # API port
CUDA_VISIBLE_DEVICES=0       # GPU device selection
```

### Docker Compose Override
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  vector-translation-gpu:
    environment:
      - CUDA_VISIBLE_DEVICES=0,1  # Multiple GPUs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2  # Use 2 GPUs
              capabilities: [gpu]
```

## 📈 Performance Monitoring

### GPU Utilization
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Container stats
docker stats vector-translation-gpu
```

### Memory Management
- Automatic memory cleanup after each request
- GPU memory cache clearing
- Memory usage warnings in health check

## 🚨 Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Check Docker GPU support
   docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi
   ```

2. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size hoặc image resolution
   # Monitor memory usage via /health endpoint
   curl http://localhost:5000/health
   ```

3. **Model loading errors**
   ```bash
   # Check logs
   docker logs vector-translation-gpu
   
   # Ensure internet connection for model download
   # Check disk space for model cache
   ```

## 📝 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   Flask API      │───▶│   GPU/CPU       │
│                 │    │   (clip_api.py)  │    │   Processing    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Health Check   │
                       │   & Monitoring   │
                       └──────────────────┘
```

## 🔒 Security Considerations

- Input validation cho image uploads
- File size limits
- Memory leak prevention
- Resource monitoring và alerts

## 📚 Development

### Adding new models
1. Update model loading section
2. Add new endpoint
3. Update health check
4. Test GPU compatibility

### Testing
```bash
# Unit tests
python -m pytest tests/

# Load testing
ab -n 100 -c 10 http://localhost:5000/health
``` 