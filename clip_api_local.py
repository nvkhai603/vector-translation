import os
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO
from flask import Flask, request, jsonify
import torchvision.models as models
import torchvision.transforms as transforms

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
    clip_model.to(device)
    clip_model.eval()
    print(f"CLIP model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    clip_model = None
    clip_processor = None

# --- ResNet-50 Model Loading ---
try:
    resnet_model = models.resnet50(pretrained=True)
    if os.path.exists('resnet50.pth'):
        resnet_model.load_state_dict(torch.load('resnet50.pth', map_location=device))
        print("Loaded ResNet-50 from local file")
    # Remove the final classification layer to get features
    resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
    resnet_model.to(device)
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
        batch_t = torch.unsqueeze(img_t, 0).to(device)

        with torch.no_grad():
            features = resnet_model(batch_t)
            # Flatten the features and move back to CPU
            vector = features.view(batch_t.size(0), -1).cpu()
        return vector
    except Exception as e:
        print(f"Error in get_resnet_vector: {e}")
        raise

# --- API Endpoints ---
@app.route('/get_clip_vector', methods=['POST']) # Renamed for clarity
def get_clip_vector_endpoint():
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
            return jsonify({'vector': vector_list}) # Key updated
        except Exception as e:
            print(f"Error getting CLIP vector: {e}")
            return jsonify({'error': 'Failed to get CLIP vector'}), 500
    else:
        # This case should ideally not be reached if logic above is correct
        return jsonify({'error': 'Could not obtain image'}), 500


@app.route('/get_resnet_vector', methods=['POST'])
def get_resnet_vector_endpoint():
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
            return jsonify({'vector': vector_list})
        except Exception as e:
            print(f"Error getting ResNet vector: {e}")
            return jsonify({'error': 'Failed to get ResNet vector'}), 500
    else:
        # This case should ideally not be reached if logic above is correct
        return jsonify({'error': 'Could not obtain image'}), 500

# Gunicorn will run the 'app' object directly
# The following block is removed as Gunicorn handles server startup in production
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
