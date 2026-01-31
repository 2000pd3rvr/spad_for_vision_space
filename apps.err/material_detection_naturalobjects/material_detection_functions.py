#!/usr/bin/env python3
"""
Material Detection Functions

This module contains the core functions for material detection using pretrained CNN models.
It processes .sto files to extract spatiotemporal data and performs classification.
"""

import os
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
# Ensure compatibility with newer numpy versions
try:
    from numpy import _core
except ImportError:
    # For newer numpy versions, use the standard API
    pass
import base64
import io
from datetime import datetime

"""Dynamic model/config/classes support for multiple pretrained heads (12 or 18 classes)."""

# New Material Classifier Model
class MaterialClassifier(nn.Module):
    """
    CNN architecture for material classification
    Optimized for 16x16 RGB images with 12 classes
    """
    def __init__(self, num_classes=12):
        super(MaterialClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling instead of flattening
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# Configuration (channels fixed; numclasses will be inferred from weights)
config = {
    "ch": 3,
    "numclasses": None  # set after loading weights
}

# Known class sets
# IMPORTANT: This order MUST match ImageFolder's alphabetical assignment during training
# ImageFolder assigns class IDs based on alphabetical order of folder names
# This is the order that the model was trained with, matching multiwebapp
# Multiwebapp explicitly states: "Class order MUST match ImageFolder's alphabetical assignment during training"
CLASSES_12 = [
    "3dmodel",                  # 0 (alphabetically first)
    "LEDscreen",                # 1 (lowercase 's' to match training directory)
    "bowl__purpleplastic",      # 2
    "bowl__whiteceramic",       # 3
    "carrot__natural",          # 4
    "eggplant__natural",        # 5
    "greenpepper__natural",     # 6
    "potato__natural",          # 7
    "redpepper__natural",       # 8
    "teacup__ceramic",          # 9
    "tomato__natural",          # 10
    "yellowpepper__natural",    # 11
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
    "yellowpepper__natural",
]

# Active classes (set on load)
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
        # Output features are determined by loaded weights; initialize with default 18, adjust on load
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
_model = None
_active_weight_key = None  # tracks which weights are currently loaded

def load_material_model(weight_choice=None):
    """Load the pretrained classifier model"""
    global _model
    global _active_weight_key

    # Use the new classifier model
    base_dir = Path(__file__).resolve().parent
    # Updated to use models folder (sibling to local_spad_for_vision)
    models_dir = base_dir.parent.parent.parent.parent / "models" / "spatiotemporal"
    model_path = str(models_dir / "training_results_material_classifier_best_99.25%.pth")
    
    # If a model is already loaded, reuse it
    if _model is not None and _active_weight_key == "classifier_model":
        return _model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    print(f"Loading material classifier model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint.get('class_names')
    num_classes = checkpoint.get('num_classes', len(class_names) if class_names else 12)

    global classes
    if class_names:
        classes = class_names
    elif num_classes == 12:
        classes = CLASSES_12
    elif num_classes == 18:
        classes = CLASSES_18
    else:
        # Fallback: use 12 classes
        classes = CLASSES_12[:num_classes]

    # Update config and instantiate model
    config['numclasses'] = num_classes
    _model = MaterialClassifier(num_classes=num_classes)

    # Load weights
    _model.load_state_dict(checkpoint['model_state_dict'])
    
    _model.eval()
    _active_weight_key = "classifier_model"
    print(f"Material classifier model loaded successfully! Classes: {num_classes}")
    print(f"Class names: {classes}")
    return _model

def validate_sto_file(sto_file):
    """Validate if file is a proper .sto file using the same approach as YOLOv8 dataset creation"""
    try:
        if not os.path.exists(sto_file):
            return False, "File does not exist"
        
        file_size = os.path.getsize(sto_file)
        if file_size == 0:
            return False, "File is empty"
        
        # Use the same approach that worked for YOLOv8 dataset creation with numpy compatibility
        with open(sto_file, 'rb') as f:
            try:
                data = pickle.load(f)
            except (ImportError, AttributeError) as e:
                if "numpy._core" in str(e):
                    # Try with numpy compatibility mode
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f.seek(0)
                        data = pickle.load(f, encoding='latin1')
                else:
                    raise e
        
        # Check if we have 4 items as expected (same as YOLOv8 approach)
        if len(data) != 4:
            return False, f"Invalid .sto structure: expected 4 items, got {len(data)}"
        
        # Validate that we have the required data at the expected indices
        if data[1] is None:
            return False, "No transient image found at index 1"
        
        if data[3] is None:
            return False, "No RGB image found at index 3"
        
        return True, "Valid .sto file"
        
    except Exception as e:
        return False, f"Error validating .sto file: {str(e)}"

def process_png_file(png_file):
    """
    Process PNG file for classifier model input
    The model was trained on 16x16 RGB PNG images
    """
    try:
        # Load PNG image directly
        image = Image.open(png_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 16x16 expected by the trained model
        image = image.resize((16, 16))
        
        return image, image
        
    except Exception as e:
        raise ValueError(f"Error processing PNG file: {str(e)}")

def process_png_bytes(png_bytes):
    """
    Process PNG bytes for classifier model input
    The model was trained on 16x16 RGB PNG images
    """
    try:
        # Load PNG image from bytes
        image = Image.open(io.BytesIO(png_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 16x16 expected by the trained model
        image = image.resize((16, 16))
        
        return image, image
        
    except Exception as e:
        raise ValueError(f"Error processing PNG bytes: {str(e)}")

def process_sto_file(sto_file):
    """
    Process .sto file to extract:
    - Index 1: Display image (for showing in dash box)
    - Index 0: 16x16 PIL image (convert to PNG bytes for model input)
    """
    try:
        # Use the same approach that worked for YOLOv8 dataset creation with numpy compatibility
        with open(sto_file, 'rb') as f:
            try:
                data = pickle.load(f)
            except (ImportError, AttributeError) as e:
                if "numpy._core" in str(e):
                    # Try with numpy compatibility mode
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f.seek(0)
                        data = pickle.load(f, encoding='latin1')
                else:
                    raise e
        
        # Check if we have at least 2 items as required
        if len(data) < 2:
            raise ValueError(f"Invalid .sto file format: expected at least 2 items, got {len(data)}")
        
        # Extract index 1 for display
        display_data = data[1]
        if display_data is None:
            raise ValueError("No display data found at index 1")
        
        # Extract index 0 for model input (16x16 PIL image)
        model_data = data[0]
        if model_data is None:
            raise ValueError("No model data found at index 0")
        
        # Convert index 1 to display image
        if isinstance(display_data, Image.Image):
            display_image = display_data
        else:
            display_image = Image.fromarray(display_data, mode='RGB')
        
        # Convert index 0 to PIL image and ensure it's 16x16
        if isinstance(model_data, Image.Image):
            model_image = model_data
        else:
            model_image = Image.fromarray(model_data, mode='RGB')
        
        # Ensure model image is 16x16
        if model_image.size != (16, 16):
            model_image = model_image.resize((16, 16))
        
        # Convert model image to PNG bytes (maintaining PNG format with headers, metadata, etc.)
        png_bytes = image_to_png_bytes(model_image)
        
        return display_image, png_bytes, model_image
        
    except Exception as e:
        raise ValueError(f"Error processing .sto file: {str(e)}")

def process_sto_file_index2(sto_file):
    """
    Process .sto file to extract (for new_material_detection_demo):
    - Index 1: Display image (for showing in dash box)
    - Index 0: 16x16 PIL image (convert to PNG bytes for model input)
    """
    try:
        # Use the same approach that worked for YOLOv8 dataset creation with numpy compatibility
        with open(sto_file, 'rb') as f:
            try:
                data = pickle.load(f)
            except (ImportError, AttributeError) as e:
                if "numpy._core" in str(e):
                    # Try with numpy compatibility mode
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f.seek(0)
                        data = pickle.load(f, encoding='latin1')
                else:
                    raise e
        
        # Check if we have at least 2 items as required (need index 0 and 1)
        if len(data) < 2:
            raise ValueError(f"Invalid .sto file format: expected at least 2 items, got {len(data)}")
        
        # Extract index 1 for display
        display_data = data[1]
        if display_data is None:
            raise ValueError("No display data found at index 1")
        
        # Extract index 0 for model input (16x16 PIL image)
        model_data = data[0]
        if model_data is None:
            raise ValueError("No model data found at index 0")
        
        # Convert index 1 to display image and ensure it's RGB PNG format
        if isinstance(display_data, Image.Image):
            display_image = display_data
        else:
            display_image = Image.fromarray(display_data, mode='RGB')
        
        # Ensure the display image is in RGB mode for proper PNG display
        if display_image.mode != 'RGB':
            display_image = display_image.convert('RGB')
        
        # Convert to PNG bytes for consistent display
        display_png_bytes = image_to_png_bytes(display_image)
        
        # Convert index 0 to PIL image and ensure it's 16x16
        if isinstance(model_data, Image.Image):
            model_image = model_data
        else:
            model_image = Image.fromarray(model_data, mode='RGB')
        
        # Ensure model image is 16x16
        if model_image.size != (16, 16):
            model_image = model_image.resize((16, 16))
        
        # Convert model image to PNG bytes (maintaining PNG format with headers, metadata, etc.)
        png_bytes = image_to_png_bytes(model_image)
        
        return display_image, png_bytes, model_image, display_png_bytes
        
    except Exception as e:
        raise ValueError(f"Error processing .sto file: {str(e)}")

def preprocess_image(image):
    """Preprocess image for classifier model input"""
    # If image is already a PIL Image, use it directly
    if isinstance(image, Image.Image):
        # Ensure it's RGB and 16x16
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if image.size != (16, 16):
            # Resize to 16x16 expected by the trained model
        image = image.resize((16, 16))
    else:
        # If it's an array, convert to PIL Image first
        inputimg = Image.fromarray(image, mode='RGB')
        image = inputimg.resize((16, 16))
    
    # Apply transforms (matching classifier training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Add batch dimension
    tensor = transform(image).unsqueeze(0)
    
    return tensor

def predict_material(image_tensor, model=None):
    """Get predictions from the model"""
    if model is None:
        model = load_material_model()
    
    with torch.no_grad():
        # Get predictions
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get all class probabilities
        all_prob = probabilities.squeeze().tolist()
        
        # Debug: Check if probabilities sum to 1.0
        total_prob = sum(all_prob)
        print(f"Debug: Sum of probabilities before conversion: {total_prob}")
        
        # Ensure probabilities sum to exactly 1.0 (handle floating point precision)
        if abs(total_prob - 1.0) > 1e-6:  # If not close to 1.0
            all_prob = [p / total_prob for p in all_prob]
            print(f"Debug: Normalized probabilities, new sum: {sum(all_prob)}")
        
        # Function to format class name for display (consolidate to materials only)
        def format_class_name(class_name):
            """Format class name to consolidated material format"""
            # Handle special cases
            if class_name == "3dmodel":
                return "3D Model"
            elif class_name == "LEDscreen":
                return "LED"
            elif "__" in class_name:
                # Format: "carrot__natural" -> "natural carrot"
                # Format: "bowl__purpleplastic" -> "purple plastic bowl"
                parts = class_name.split("__")
                if len(parts) == 2:
                    material, type_ = parts
                    # Handle compound words in type_ (e.g., "purpleplastic" -> "purple plastic")
                    if type_ == "purpleplastic":
                        type_ = "purple plastic"
                    elif type_ == "whiteceramic":
                        type_ = "white ceramic"
                    # Lowercase and combine: "natural carrot"
                    return f"{type_} {material}".lower()
            # Default: replace underscores and title case
            return class_name.replace("__", " ").replace("_", " ").title()
        
        # Create results for ALL classes
        results = []
        for i, prob in enumerate(all_prob):
            confidence = float(prob * 100)  # Convert to percentage
            original_class = classes[i]
            display_class = format_class_name(original_class)
            results.append({
                'class': original_class,  # Keep original for internal use
                'display_class': display_class,  # Formatted for display
                'confidence': confidence
            })
        
        # Debug: Check sum after conversion
        total_confidence = sum(p['confidence'] for p in results)
        print(f"Debug: Sum of confidences after conversion: {total_confidence}")
        
        # Final normalization to ensure exactly 100%
        if abs(total_confidence - 100.0) > 1e-6:
            scale_factor = 100.0 / total_confidence
            for result in results:
                result['confidence'] = float(result['confidence'] * scale_factor)
            print(f"Debug: Final normalization applied, new sum: {sum(p['confidence'] for p in results)}")
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Always adjust maximum confidence to 100% minus sum of other classes
        if len(results) > 1:
            # Calculate sum of all other class probabilities
            other_sum = sum(p['confidence'] for p in results[1:])
            # Set highest confidence to 100% minus sum of others
            original_max = results[0]['confidence']
            results[0]['confidence'] = 100.0 - other_sum
            print(f"Debug: Adjusted maximum confidence from {original_max:.6f}% to {results[0]['confidence']:.6f}% (100% - {other_sum:.6f}% others)")
        
        return results

def image_to_png_bytes(image):
    """Convert PIL Image to PNG file bytes (maintaining PNG format with headers, metadata, etc.)"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"
