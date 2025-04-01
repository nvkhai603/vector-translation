import os
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
model_name = "patrickjohncyh/fashion-clip"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

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
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the image embeddings directly
        image_embeddings = outputs.image_embeds
    return image_embeddings

@app.route('/get_vector', methods=['POST'])
def get_vector():
    image = None
    text = request.form.get('text', "")  # Optional text

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

    # Process the image if available
    if image:
        try:
            vector = get_clip_vector(image, text)
            vector_list = vector.tolist()
            return jsonify({'vector': vector_list})
        except Exception as e:
            print(f"Error getting CLIP vector: {e}")
            return jsonify({'error': 'Failed to get CLIP vector'}), 500
    else:
        # This case should ideally not be reached if logic above is correct
        return jsonify({'error': 'Could not obtain image'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
