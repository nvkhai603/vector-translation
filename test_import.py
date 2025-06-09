#!/usr/bin/env python3
"""
Test script để kiểm tra import các thư viện
"""

print("Testing imports...")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch import error: {e}")

try:
    from transformers import CLIPProcessor, CLIPModel
    print("✅ Transformers: OK")
except ImportError as e:
    print(f"❌ Transformers import error: {e}")

try:
    from flask import Flask
    print("✅ Flask: OK")
except ImportError as e:
    print(f"❌ Flask import error: {e}")

try:
    from PIL import Image
    print("✅ Pillow: OK")
except ImportError as e:
    print(f"❌ Pillow import error: {e}")

try:
    import psutil
    print("✅ psutil: OK")
except ImportError as e:
    print(f"❌ psutil import error: {e}")

try:
    import pynvml
    print("✅ pynvml: OK")
except ImportError as e:
    print(f"❌ pynvml import error: {e}")

print("\nAll imports tested!") 