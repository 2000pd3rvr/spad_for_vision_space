import pickle
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
import base64
from io import BytesIO

# Custom unpickler to handle NumPy compatibility
class NumpyCompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        elif module == 'numpy._core.umath':
            module = 'numpy.core.umath'
        elif module == 'numpy.core':
            module = 'numpy.core'
        return super().find_class(module, name)

def validate_sto_structure(sto_file_path):
    """
    Validate that the .sto file has exactly 2 items:
    - Index 0: 16x16 PIL Image
    - Index 1: PIL Image (any size)
    """
    try:
        # Check if file exists and has content
        if not os.path.exists(sto_file_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(sto_file_path)
        if file_size == 0:
            return False, "File is empty"
        
        with open(sto_file_path, 'rb') as f:
            try:
                unpickler = NumpyCompatibleUnpickler(f)
                data = unpickler.load()
            except (ImportError, AttributeError) as e:
                if "numpy._core" in str(e):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f.seek(0)
                        data = pickle.load(f, encoding='latin1')
                else:
                    raise e
        
        # Check if data is a list
        if not isinstance(data, list):
            return False, f"Expected list, got {type(data)}"
        
        # Check if we have exactly 2 items
        if len(data) != 2:
            return False, f"Expected exactly 2 items, got {len(data)}"
        
        # Check index 0: should be PIL Image
        try:
            item_0 = data[0]
            if not isinstance(item_0, Image.Image):
                return False, f"Index 0 should be PIL Image, got {type(item_0)}"
            
            # Check if it's 16x16 (optional check, can be flexible)
            if hasattr(item_0, 'size') and item_0.size != (16, 16):
                # Resize to 16x16 if needed
                item_0 = item_0.resize((16, 16), Image.Resampling.LANCZOS)
        except IndexError:
            return False, "Index 0 does not exist"
        
        # Check index 1: should be PIL Image (any size)
        try:
            item_1 = data[1]
            if not isinstance(item_1, Image.Image):
                return False, f"Index 1 should be PIL Image, got {type(item_1)}"
        except IndexError:
            return False, "Index 1 does not exist"
        
        return True, "Valid STO structure"
        
    except Exception as e:
        return False, f"Error validating STO file: {str(e)}"

def process_sto_file_spatiotemporal(sto_file_path):
    """
    Process .sto file to extract:
    - Index 0: 16x16 image for MD.pth (Material Detection)
    - Index 1: Image for OD.pt (Object Detection)
    """
    try:
        with open(sto_file_path, 'rb') as f:
            try:
                unpickler = NumpyCompatibleUnpickler(f)
                data = unpickler.load()
            except (ImportError, AttributeError) as e:
                if "numpy._core" in str(e):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f.seek(0)
                        data = pickle.load(f, encoding='latin1')
                else:
                    raise e
        
        # Basic validation - check if we have at least 2 items
        if len(data) < 2:
            raise ValueError(f"Expected at least 2 items, got {len(data)}")
        
        # Extract images
        md_image = data[0]  # 16x16 image for material detection
        od_image = data[1]  # Any size image for object detection
        
        # Ensure RGB mode
        if md_image.mode != 'RGB':
            md_image = md_image.convert('RGB')
        if od_image.mode != 'RGB':
            od_image = od_image.convert('RGB')
        
        return md_image, od_image
        
    except Exception as e:
        raise ValueError(f"Error processing .sto file: {str(e)}")

def load_md_model():
    """Load the Material Detection model (MD.pth)"""
    try:
        model_path = "/Users/pd3rvr/Documents/object_detection/multiwebapp/apps/material_detection_naturalobjects/spatiotemporal_model/weights/MD.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MD model not found at {model_path}")
        
        # Load the model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check if it's a dictionary (checkpoint) or model object
        if isinstance(checkpoint, dict):
            # If it's a checkpoint, we need to reconstruct the model
            # For now, let's use the material detection model from the existing functions
            import sys
            sys.path.append('apps/material_detection_naturalobjects')
            from material_detection_functions import load_material_model
            model = load_material_model()
        else:
            # If it's a model object
            model = checkpoint
            model.eval()
        
        print(f"MD model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        raise Exception(f"Error loading MD model: {str(e)}")

def load_od_model():
    """Load the Object Detection model (OD.pt)"""
    try:
        model_path = "/Users/pd3rvr/Documents/object_detection/multiwebapp/apps/material_detection_naturalobjects/spatiotemporal_model/weights/OD.pt"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"OD model not found at {model_path}")
        
        # Load YOLO model
        model = YOLO(model_path)
        
        print(f"OD model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        raise Exception(f"Error loading OD model: {str(e)}")

def preprocess_md_image(image):
    """Preprocess image for Material Detection model"""
    try:
        # Resize to 16x16 if not already
        if image.size != (16, 16):
            image = image.resize((16, 16), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor
        
    except Exception as e:
        raise Exception(f"Error preprocessing MD image: {str(e)}")

def predict_material_detection(image_tensor, model):
    """Predict using Material Detection model"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            
            # Get predictions (assuming model outputs logits)
            if isinstance(outputs, torch.Tensor):
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Convert to percentage
                confidence_percent = confidence.item() * 100
                predicted_class_id = predicted.item()
                
                # Map class ID to class name (assuming 8 classes: bell_pepper, bowl, carrot, eggplant, poor_vis, potato, teacup, tomato)
                class_names = ['bell_pepper', 'bowl', 'carrot', 'eggplant', 'poor_vis', 'potato', 'teacup', 'tomato']
                predicted_class_name = class_names[predicted_class_id] if predicted_class_id < len(class_names) else f"Class_{predicted_class_id}"
                
                return {
                    'predicted_class': predicted_class_name,
                    'confidence': confidence_percent,
                    'probabilities': probabilities[0].tolist()
                }
            else:
                raise ValueError("Unexpected model output format")
                
    except Exception as e:
        raise Exception(f"Error in material detection prediction: {str(e)}")

def predict_object_detection(image, model):
    """Predict using Object Detection model"""
    try:
        # Run YOLO inference
        results = model(image)
        
        # Process results
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = model.names[class_id] if hasattr(model, 'names') else f"Class_{class_id}"
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': float(confidence * 100),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        return detections
        
    except Exception as e:
        raise Exception(f"Error in object detection prediction: {str(e)}")

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    try:
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        raise Exception(f"Error converting image to base64: {str(e)}")

def process_spatiotemporal_sto(sto_file_path):
    """
    Complete processing pipeline for spatiotemporal .sto files
    """
    try:
        # Load models
        md_model = load_md_model()
        od_model = load_od_model()
        
        # Process STO file
        md_image, od_image = process_sto_file_spatiotemporal(sto_file_path)
        
        # Preprocess for MD model
        md_tensor = preprocess_md_image(md_image)
        
        # Get predictions
        md_prediction = predict_material_detection(md_tensor, md_model)
        od_predictions = predict_object_detection(od_image, od_model)
        
        # Convert images to base64 for display
        md_image_b64 = image_to_base64(md_image)
        od_image_b64 = image_to_base64(od_image)
        
        return {
            'success': True,
            'md_prediction': md_prediction,
            'od_predictions': od_predictions,
            'md_image': md_image_b64,
            'od_image': od_image_b64,
            'md_image_size': md_image.size,
            'od_image_size': od_image.size
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
