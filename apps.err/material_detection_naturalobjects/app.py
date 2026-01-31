#!/usr/bin/env python3
"""
Material Detection Natural Objects Flask App

This app provides a web interface for material detection using pretrained CNN models.
It processes .sto files to extract spatiotemporal data and performs classification.
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
import base64
import io
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'material_detection_secret_key'

# Configuration
config = {
    "ch": 3,
    "numclasses": None  # will be set from weights
}

# Class labels (matching the training script)
CLASSES_12 = [
    "bowl__purpleplastic",
    "bowl__whiteceramic", 
    "carrot__natural",
    "eggplant__natural",
    "greenpepper__natural",
    "potato__natural",
    "3dmodel",
    "redpepper__natural",
    "teacup__ceramic",
    "tomato__natural",
    "LEDscreen",
    "yellowpepper__natural"
]

CLASSES_18 = [
    "bowl__purpleplastic",
    "bowl__whiteceramic",
    "carrot__3dmodel",
    "carrot__natural",
    "eggplant__3dmodel",
    "eggplant__natural",
    "greenpepper__3dmodel",
    "greenpepper__natural",
    "potato__3dmodel",
    "potato__natural",
    "redpepper__3dmodel",
    "redpepper__natural",
    "teacup__ceramic",
    "tomato__3dmodel",
    "tomato__natural",
    "yellowpepper__3dmodel",
    "LEDscreen",
    "yellowpepper__natural"
]

classes = CLASSES_12

# Model architecture (matching the training script)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(config['ch'], 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(2048, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        out_features = config['numclasses'] if config['numclasses'] else 18
        self.fc4 = nn.Linear(512, out_features)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.flat(x)
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x

# Global model variable
model = None

def load_model():
    """Load the pretrained model"""
    global model
    
    preferred_12 = "weights/epoch_399  Accuracy_98.25%__cons3DLEDeq.pth"
    preferred_18 = "weights/epoch_491  Accuracy_98.819%__cons3Deq.pth"

    if os.path.exists(preferred_12):
        model_path = preferred_12
    elif os.path.exists(preferred_18):
        model_path = preferred_18
    else:
        model_path = None
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("No pretrained model weights found!")
    
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('msd', checkpoint)
    out_shape = state_dict['fc4.weight'].shape[0]

    global classes
    if out_shape == 12:
        classes = CLASSES_12
    elif out_shape == 18:
        classes = CLASSES_18
    else:
        classes = (CLASSES_18 if out_shape > 12 else CLASSES_12)[:out_shape]

    config['numclasses'] = out_shape
    model = ConvNet()
    model.load_state_dict(state_dict)
    
    model.eval()
    print("Model loaded successfully!")

def process_sto_file(sto_file):
    """
    Process .sto file to extract:
    - Index 1: Transient image (for model input)
    - Index 3: RGB image (for display)
    """
    try:
        # Load .sto file
        with open(sto_file, 'rb') as f:
            data = pickle.load(f)
        
        if len(data) < 4:
            raise ValueError("Invalid .sto file format: insufficient data")
        
        # Extract index 1 (transient image) for model input
        transient_image = data[1]  # Index 1
        
        # Extract index 3 (RGB image) for display
        rgb_image = data[3]  # Index 3
        
        return transient_image, rgb_image
        
    except Exception as e:
        raise ValueError(f"Error processing .sto file: {str(e)}")

def preprocess_image(image_array):
    """Preprocess image for model input"""
    # Convert to PIL Image if it's a numpy array
    if isinstance(image_array, np.ndarray):
        # Ensure it's RGB
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image = Image.fromarray(image_array.astype(np.uint8), mode='RGB')
        else:
            raise ValueError("Image must be RGB format")
    else:
        image = image_array
    
    # Resize to 32x32 (matching training data)
    image = image.resize((32, 32))
    
    # Apply transforms (matching training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Add batch dimension
    tensor = transform(image).unsqueeze(0)
    
    return tensor

def predict_material(image_tensor):
    """Get predictions from the model"""
    global model
    
    if model is None:
        raise RuntimeError("Model not loaded!")
    
    with torch.no_grad():
        # Get predictions
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3, dim=1)
        
        # Convert to lists
        top3_prob = top3_prob.squeeze().tolist()
        top3_indices = top3_indices.squeeze().tolist()
        
        # Create results
        results = []
        for i, (prob, idx) in enumerate(zip(top3_prob, top3_indices)):
            results.append({
                'rank': i + 1,
                'class': classes[idx],
                'confidence': float(prob * 100)  # Convert to percentage
            })
        
        return results

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Main page"""
    return render_template('material_detection_demo.html')

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for material detection"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if it's a .sto file
        if not file.filename.lower().endswith('.sto'):
            return jsonify({
                'error': 'Selected image only contains spatially resolved signal. Select correct spatiotemporal image.',
                'error_type': 'wrong_format'
            }), 400
        
        # Save uploaded file temporarily
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sto"
        file.save(temp_path)
        
        try:
            # Process .sto file
            transient_image, rgb_image = process_sto_file(temp_path)
            
            # Preprocess for model
            image_tensor = preprocess_image(transient_image)
            
            # Get predictions
            predictions = predict_material(image_tensor)
            
            # Convert RGB image to base64 for display
            if isinstance(rgb_image, np.ndarray):
                display_image = Image.fromarray(rgb_image.astype(np.uint8), mode='RGB')
            else:
                display_image = rgb_image
            
            img_base64 = image_to_base64(display_image)
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'image': img_base64
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    try:
        load_model()
        print("Material Detection App started successfully!")
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"Failed to start app: {e}")
