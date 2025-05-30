import os
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO
from flask import Flask, request, jsonify
import torchvision.models as models
import torchvision.transforms as transforms
import gc
import psutil
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

app = Flask(__name__)

# --- GPU/CPU Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# --- CLIP Model Loading ---
clip_model_name = "patrickjohncyh/fashion-clip"
try:
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.to(device)  # Move to GPU if available
    clip_model.eval()
    print(f"CLIP model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    clip_model = None
    clip_processor = None

# --- ResNet-50 Model Loading ---
try:
    resnet_model = models.resnet50(pretrained=True)
    # Remove the final classification layer to get features
    resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
    resnet_model.to(device)  # Move to GPU if available
    resnet_model.eval()
    print(f"ResNet-50 model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading ResNet-50 model: {e}")
    resnet_model = None

# --- ResNet-50 Image Transformations ---
resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Helper Functions ---
def get_image_from_url(url):
    """Downloads an image from a URL.

    Args:
        url (str): The URL of the image.

    Returns:
        PIL.Image.Image: The downloaded image.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

def get_clip_vector(image, text):
    """Gets the CLIP vector for a given image and text.

    Args:
        image (PIL.Image.Image): The image to process.
        text (str): The text to associate with the image.

    Returns:
        torch.Tensor: The CLIP vector.
    """
    if clip_model is None or clip_processor is None:
        raise RuntimeError("CLIP model not loaded")
    
    try:
        inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
        # Move inputs to the same device as model
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            # Use the image embeddings directly
            image_embeddings = outputs.image_embeds
            # Move back to CPU for JSON serialization
            return image_embeddings.cpu()
    except Exception as e:
        print(f"Error in get_clip_vector: {e}")
        raise

def get_resnet_vector(image):
    """Gets the ResNet-50 feature vector for a given image.

    Args:
        image (PIL.Image.Image): The image to process.

    Returns:
        torch.Tensor: The ResNet-50 feature vector.
    """
    if resnet_model is None:
        raise RuntimeError("ResNet-50 model not loaded")
    
    try:
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_t = resnet_transform(image)
        batch_t = torch.unsqueeze(img_t, 0).to(device)  # Move to GPU if available

        with torch.no_grad():
            features = resnet_model(batch_t)
            # Flatten the features and move back to CPU
            vector = features.view(batch_t.size(0), -1).cpu()
        return vector
    except Exception as e:
        print(f"Error in get_resnet_vector: {e}")
        raise

# --- System Resource Monitoring ---
def get_system_resources():
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    resources = {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3),
        'memory_used_gb': memory.used / (1024**3)
    }
    
    # GPU monitoring if available
    if torch.cuda.is_available() and PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            resources.update({
                'gpu_memory_used_gb': gpu_memory.used / (1024**3),
                'gpu_memory_total_gb': gpu_memory.total / (1024**3),
                'gpu_memory_percent': (gpu_memory.used / gpu_memory.total) * 100,
                'gpu_utilization_percent': gpu_util.gpu
            })
        except Exception as e:
            print(f"GPU monitoring error: {e}")
    
    return resources

def cleanup_memory():
    """Clean up memory to prevent OOM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        resources = get_system_resources()
        
        # Check if models are loaded
        models_status = {
            'clip_model_loaded': clip_model is not None,
            'resnet_model_loaded': resnet_model is not None,
            'device': str(device)
        }
        
        # Warning thresholds
        warnings = []
        if resources['memory_percent'] > 85:
            warnings.append("High memory usage")
        if torch.cuda.is_available() and 'gpu_memory_percent' in resources:
            if resources['gpu_memory_percent'] > 85:
                warnings.append("High GPU memory usage")
        
        return jsonify({
            'status': 'healthy',
            'models': models_status,
            'resources': resources,
            'warnings': warnings
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/get_clip_vector', methods=['POST']) # Renamed for clarity
def get_clip_vector_endpoint():
    try:
        # Check memory before processing
        resources = get_system_resources()
        if resources['memory_percent'] > 90:
            cleanup_memory()
        
        image = None
        text = request.form.get('text', "")  # Optional text

        # Check if an image file is uploaded
        if 'image_file' in request.files:
            file = request.files['image_file']
            if file.filename != '':
                try:
                    # Ensure image is RGB for CLIP as well
                    image = Image.open(file.stream).convert('RGB')
                except Exception as e:
                    print(f"Error opening uploaded image: {e}")
                    return jsonify({'error': 'Failed to process uploaded image'}), 500
            else:
                return jsonify({'error': 'No selected file'}), 400

        # If no file is uploaded, check for image URL
        elif 'image_url' in request.form:
            image_url = request.form.get('image_url')
            if image_url:
                image = get_image_from_url(image_url)
                if not image:
                    return jsonify({'error': 'Failed to download image from URL'}), 500
                # Ensure downloaded image is RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            else:
                return jsonify({'error': 'image_url is empty'}), 400
        else:
            return jsonify({'error': 'Either image_file or image_url is required'}), 400

        # Process the image if available
        # Process the image with CLIP if available
        if image:
            try:
                vector = get_clip_vector(image, text)
                vector_list = vector.tolist()
                
                # Cleanup after processing
                cleanup_memory()
                
                return jsonify({'vector': vector_list}) # Key updated
            except Exception as e:
                print(f"Error getting CLIP vector: {e}")
                cleanup_memory()
                return jsonify({'error': 'Failed to get CLIP vector'}), 500
        else:
            # This case should ideally not be reached if logic above is correct
            return jsonify({'error': 'Could not obtain image'}), 500
            
    except Exception as e:
        print(f"Unexpected error in CLIP endpoint: {e}")
        cleanup_memory()
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/get_resnet_vector', methods=['POST'])
def get_resnet_vector_endpoint():
    try:
        # Check memory before processing
        resources = get_system_resources()
        if resources['memory_percent'] > 90:
            cleanup_memory()
            
        image = None

        # Check if an image file is uploaded
        if 'image_file' in request.files:
            file = request.files['image_file']
            if file.filename != '':
                try:
                    image = Image.open(file.stream)
                except Exception as e:
                    print(f"Error opening uploaded image: {e}")
                    return jsonify({'error': 'Failed to process uploaded image'}), 500
            else:
                return jsonify({'error': 'No selected file'}), 400

        # If no file is uploaded, check for image URL
        elif 'image_url' in request.form:
            image_url = request.form.get('image_url')
            if image_url:
                image = get_image_from_url(image_url)
                if not image:
                    return jsonify({'error': 'Failed to download image from URL'}), 500
            else:
                return jsonify({'error': 'image_url is empty'}), 400
        else:
            return jsonify({'error': 'Either image_file or image_url is required'}), 400

        # Process the image with ResNet if available
        if image:
            try:
                vector = get_resnet_vector(image)
                vector_list = vector.tolist()
                
                # Cleanup after processing
                cleanup_memory()
                
                return jsonify({'vector': vector_list})
            except Exception as e:
                print(f"Error getting ResNet vector: {e}")
                cleanup_memory()
                return jsonify({'error': 'Failed to get ResNet vector'}), 500
        else:
            # This case should ideally not be reached if logic above is correct
            return jsonify({'error': 'Could not obtain image'}), 500
            
    except Exception as e:
        print(f"Unexpected error in ResNet endpoint: {e}")
        cleanup_memory()
        return jsonify({'error': 'Internal server error'}), 500

# Gunicorn will run the 'app' object directly
# The following block is removed as Gunicorn handles server startup in production
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
