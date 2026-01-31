#!/usr/bin/env python3
"""
Evaluation script for material classifier in spatiotemporal detection.
Tests the material classifier to ensure it makes correct inference from selected weights.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
from io import BytesIO
import json
import requests
from pathlib import Path

# Add paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'apps.err', 'material_detection_naturalobjects'))

def load_sto_index0(sto_path):
    """Extract index 1 (16x16 material detection image) from STO file
    STO structure: Index 0=metadata, Index 1=16x16 material image, Index 2=OD metadata, Index 3=640x640 OD image"""
    with open(sto_path, 'rb') as f:
        sto_data = pickle.load(f)
        if len(sto_data) < 2:
            raise ValueError('STO file does not have index 1 (material detection image)')
        # Extract index 1 for material detection (16x16 image)
        sto_item = sto_data[1]
        if isinstance(sto_item, bytes):
            image = Image.open(BytesIO(sto_item)).convert('RGB')
        elif hasattr(sto_item, 'mode'):
            image = sto_item.convert('RGB')
        else:
            raise ValueError(f'Invalid STO file structure at index 1: expected image, got {type(sto_item).__name__}')
        return image

def test_material_classifier_local(weight_path, test_image_path, expected_class=None):
    """
    Test material classifier locally (without Flask server)
    """
    print(f"\n{'='*80}")
    print(f"Testing Material Classifier")
    print(f"{'='*80}")
    print(f"Weight path: {weight_path}")
    print(f"Test image: {test_image_path}")
    print(f"Expected class: {expected_class}")
    print(f"{'='*80}\n")
    
    # Check if weight file exists
    if not os.path.exists(weight_path):
        print(f"ERROR: Weight file not found: {weight_path}")
        return False
    
    # Load image
    if test_image_path.endswith('.sto'):
        print("Loading STO file and extracting index 1 (16x16 material detection image)...")
        image = load_sto_index0(test_image_path)
    else:
        print("Loading image file...")
        image = Image.open(test_image_path).convert('RGB')
    
    print(f"Image size: {image.size}, mode: {image.mode}")
    
    # Preprocess image (same as in app.py)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if image.size != (16, 16):
        image = image.resize((16, 16), Image.Resampling.LANCZOS)
    
    image_tensor = transform(image).unsqueeze(0)
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Image tensor range: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
    
    # Define model architecture (same as in app.py)
    class ConvNetMaterialDetectionHead(nn.Module):
        """ConvNet for material detection head - 12 classes"""
        def __init__(self):
            super(ConvNetMaterialDetectionHead, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
            self.act1 = nn.ReLU()
            self.drop1 = nn.Dropout(0.3)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
            self.act2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
            self.flat = nn.Flatten()
            self.fc3 = nn.Linear(2048, 512)
            self.act3 = nn.ReLU()
            self.drop3 = nn.Dropout(0.5)
            self.fc4 = nn.Linear(512, 12)
        
        def forward(self, x):
            x = self.act1(self.conv1(x))
            x = self.act2(self.conv2(x))
            x = self.pool2(x)
            x = self.flat(x)
            x = self.act3(self.fc3(x))
            x = self.fc4(x)
            return x
    
    # Class names (same order as in app.py)
    class_names = [
        '3dmodel',
        'LEDscreen',
        'bowl__purpleplastic',
        'bowl__whiteceramic',
        'carrot__natural',
        'eggplant__natural',
        'greenpepper__natural',
        'potato__natural',
        'redpepper__natural',
        'teacup__ceramic',
        'tomato__natural',
        'yellowpepper__natural'
    ]
    
    # Load model
    print(f"\nLoading model from checkpoint...")
    model = ConvNetMaterialDetectionHead()
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # Try different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'msd' in checkpoint:
            model.load_state_dict(checkpoint['msd'], strict=True)
            print("Loaded from 'msd' key")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Loaded from 'state_dict' key")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("Loaded from 'model_state_dict' key")
        else:
            try:
                model.load_state_dict(checkpoint, strict=True)
                print("Loaded from checkpoint dict directly")
            except Exception as e:
                print(f"ERROR loading checkpoint: {e}")
                return False
    else:
        print("ERROR: Unknown checkpoint format")
        return False
    
    model.eval()
    model = model.cpu()
    
    # Run inference
    print(f"\nRunning inference...")
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Apply softmax
    import torch.nn.functional as F
    probabilities = F.softmax(predictions, dim=1)[0]
    
    # Get predicted class
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item()
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities, min(3, len(class_names)))
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Predicted class: {predicted_class} (index {predicted_class_idx})")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"\nTop 3 predictions:")
    for i in range(len(top3_indices)):
        idx = top3_indices[i].item()
        prob = top3_probs[i].item()
        print(f"  {i+1}. {class_names[idx]}: {prob:.4f} ({prob*100:.2f}%)")
    
    print(f"\nAll class probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, probabilities.tolist())):
        marker = " <-- PREDICTED" if i == predicted_class_idx else ""
        print(f"  {i:2d}. {name:25s}: {prob:.4f} ({prob*100:.2f}%){marker}")
    
    # Check if prediction matches expected class
    if expected_class:
        if predicted_class == expected_class:
            print(f"\n✓ SUCCESS: Predicted class matches expected class!")
            return True
        else:
            print(f"\n✗ FAILURE: Predicted '{predicted_class}' but expected '{expected_class}'")
            return False
    else:
        print(f"\n✓ Inference completed successfully!")
        return True

def test_material_classifier_api(weight_path, test_image_path, base_url="http://localhost:7888"):
    """
    Test material classifier via API endpoint
    """
    print(f"\n{'='*80}")
    print(f"Testing Material Classifier via API")
    print(f"{'='*80}")
    print(f"API URL: {base_url}")
    print(f"Weight path: {weight_path}")
    print(f"Test image: {test_image_path}")
    print(f"{'='*80}\n")
    
    # Load image
    if test_image_path.endswith('.sto'):
        print("Loading STO file and extracting index 1 (16x16 material detection image)...")
        image = load_sto_index0(test_image_path)
        # Convert to PNG bytes for API
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        filename = 'test_index1_material.png'
    else:
        with open(test_image_path, 'rb') as f:
            img_bytes = f.read()
        filename = os.path.basename(test_image_path)
    
    # Prepare request
    files = {'file': (filename, img_bytes, 'image/png')}
    data = {'weight_path': weight_path}
    
    print(f"Sending request to {base_url}/api/detect_material_head...")
    try:
        response = requests.post(f"{base_url}/api/detect_material_head", files=files, data=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if result.get('success'):
            print(f"\n{'='*80}")
            print(f"API RESULTS")
            print(f"{'='*80}")
            print(f"Predicted class: {result.get('predicted_class')}")
            print(f"Confidence: {result.get('confidence'):.4f} ({result.get('confidence')*100:.2f}%)")
            print(f"Inference time: {result.get('inference_time', 0):.2f} ms")
            print(f"\nTop 3 predictions:")
            for i, pred in enumerate(result.get('top3_predictions', [])[:3]):
                print(f"  {i+1}. {pred.get('display_class', pred.get('class'))}: {pred.get('probability'):.4f} ({pred.get('probability')*100:.2f}%)")
            print(f"\n✓ API test completed successfully!")
            return True
        else:
            print(f"\n✗ API test failed: {result.get('error', 'Unknown error')}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"\n✗ API request failed: {e}")
        return False

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate material classifier')
    parser.add_argument('--weight-path', type=str, required=True,
                        help='Path to model weight file')
    parser.add_argument('--test-image', type=str, required=True,
                        help='Path to test image (PNG or STO file)')
    parser.add_argument('--expected-class', type=str, default=None,
                        help='Expected class name (for validation)')
    parser.add_argument('--use-api', action='store_true',
                        help='Test via API instead of local inference')
    parser.add_argument('--api-url', type=str, default='http://localhost:7888',
                        help='API base URL (default: http://localhost:7888)')
    
    args = parser.parse_args()
    
    # Resolve paths
    weight_path = os.path.abspath(args.weight_path) if not os.path.isabs(args.weight_path) else args.weight_path
    test_image_path = os.path.abspath(args.test_image) if not os.path.isabs(args.test_image) else args.test_image
    
    # Check files exist
    if not os.path.exists(weight_path):
        print(f"ERROR: Weight file not found: {weight_path}")
        sys.exit(1)
    
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found: {test_image_path}")
        sys.exit(1)
    
    # Run test
    if args.use_api:
        success = test_material_classifier_api(weight_path, test_image_path, args.api_url)
    else:
        success = test_material_classifier_local(weight_path, test_image_path, args.expected_class)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

