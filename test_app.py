#!/usr/bin/env python3
"""
Test app đơn giản để kiểm tra Flask và GPU
"""

import torch
from flask import Flask, jsonify

app = Flask(__name__)

# Device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'torch_version': torch.__version__
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Vector Translation API is running!',
        'device': str(device),
        'endpoints': ['/health', '/']
    })

if __name__ == '__main__':
    print("Starting test Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True) 