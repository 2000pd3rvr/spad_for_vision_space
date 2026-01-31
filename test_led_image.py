#!/usr/bin/env python3
"""
Test LED image classification to debug why LED images are predicted as 3dmodel
"""

import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
import os

# Add paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'apps.err', 'material_detection_naturalobjects'))

def test_led_image(sto_path, weight_path):
    """Test LED image classification"""
    print(f"\n{'='*80}")
    print(f"Testing LED Image Classification")
    print(f"{'='*80}")
    print(f"STO file: {sto_path}")
    print(f"Weight: {weight_path}")
    print(f"{'='*80}\n")
    
    # Load STO file
    with open(sto_path, 'rb') as f:
        sto_data = pickle.load(f)
    
    print(f"STO file structure:")
    print(f"  Total items: {len(sto_data)}")
    print(f"  Index 0: {type(sto_data[0])} = {sto_data[0] if not isinstance(sto_data[0], Image.Image) else f'PIL Image {sto_data[0].size}'}")
    print(f"  Index 1: {type(sto_data[1])} = {f'PIL Image {sto_data[1].size}' if isinstance(sto_data[1], Image.Image) else sto_data[1]}")
    if len(sto_data) > 2:
        print(f"  Index 2: {type(sto_data[2])}")
    if len(sto_data) > 3:
        print(f"  Index 3: {type(sto_data[3])} = {f'PIL Image {sto_data[3].size}' if isinstance(sto_data[3], Image.Image) else sto_data[3]}")
    
    # Extract index 1 (material detection image)
    if len(sto_data) < 2:
        print("ERROR: STO file doesn't have index 1")
        return
    
    material_image = sto_data[1]
    if not isinstance(material_image, Image.Image):
        print(f"ERROR: Index 1 is not a PIL Image, it's {type(material_image)}")
        return
    
    print(f"\nMaterial detection image (Index 1):")
    print(f"  Size: {material_image.size}")
    print(f"  Mode: {material_image.mode}")
    
    # Convert to numpy for analysis
    img_array = np.array(material_image)
    print(f"  Pixel range: [{img_array.min()}, {img_array.max()}]")
    print(f"  Mean: {img_array.mean():.2f}, Std: {img_array.std():.2f}")
    print(f"  Sample pixels (first 3x3, RGB):")
    print(f"    {img_array[:3, :3]}")
    
    # Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if material_image.size != (16, 16):
        material_image = material_image.resize((16, 16), Image.Resampling.LANCZOS)
    
    image_tensor = transform(material_image).unsqueeze(0)
    print(f"\nPreprocessed tensor:")
    print(f"  Shape: {image_tensor.shape}")
    print(f"  Range: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
    print(f"  Mean: {image_tensor.mean():.4f}, Std: {image_tensor.std():.4f}")
    
    # Load model
    class ConvNetMaterialDetectionHead(nn.Module):
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
    
    class_names = [
        '3dmodel', 'LEDscreen', 'bowl__purpleplastic', 'bowl__whiteceramic',
        'carrot__natural', 'eggplant__natural', 'greenpepper__natural',
        'potato__natural', 'redpepper__natural', 'teacup__ceramic',
        'tomato__natural', 'yellowpepper__natural'
    ]
    
    print(f"\nLoading model...")
    model = ConvNetMaterialDetectionHead()
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'msd' in checkpoint:
            model.load_state_dict(checkpoint['msd'], strict=True)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
    
    model.eval()
    
    # Run inference
    print(f"\nRunning inference...")
    with torch.no_grad():
        predictions = model(image_tensor)
    
    import torch.nn.functional as F
    probabilities = F.softmax(predictions, dim=1)[0]
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item()
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Predicted: {predicted_class} (index {predicted_class_idx})")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"\nAll probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, probabilities.tolist())):
        marker = " <-- PREDICTED" if i == predicted_class_idx else ""
        print(f"  {i:2d}. {name:25s}: {prob:.6f} ({prob*100:.2f}%){marker}")
    
    # Check if it's wrong
    if 'LED' in sto_path.upper() or 'LED' in str(sto_data[0]).upper():
        expected = 'LEDscreen'
        if predicted_class != expected:
            print(f"\n⚠️  WARNING: Expected {expected} but got {predicted_class}!")
            print(f"   This is a misclassification.")
        else:
            print(f"\n✓ Correctly predicted {expected}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sto-file', required=True)
    parser.add_argument('--weight', default='../models/material_detection_head/epoch_399_Accuracy_98.25.pth')
    args = parser.parse_args()
    test_led_image(args.sto_file, args.weight)

