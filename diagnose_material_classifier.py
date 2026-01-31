#!/usr/bin/env python3
"""
Diagnostic script to investigate material classifier predictions.
Checks image extraction, preprocessing, and compares different normalization methods.
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
import matplotlib.pyplot as plt

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

def inspect_image(image_path):
    """Inspect the extracted image"""
    print(f"\n{'='*80}")
    print("IMAGE INSPECTION")
    print(f"{'='*80}")
    
    if image_path.endswith('.sto'):
        image = load_sto_index0(image_path)
        print(f"Extracted index 1 (16x16 material detection image) from STO file: {image_path}")
    else:
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded from: {image_path}")
    
    print(f"Original size: {image.size}")
    print(f"Mode: {image.mode}")
    
    # Convert to numpy array
    img_array = np.array(image)
    print(f"Array shape: {img_array.shape}")
    print(f"Array dtype: {img_array.dtype}")
    print(f"Pixel value range: [{img_array.min()}, {img_array.max()}]")
    print(f"Mean: {img_array.mean():.2f}, Std: {img_array.std():.2f}")
    
    # Resize to 16x16
    if image.size != (16, 16):
        image_16x16 = image.resize((16, 16), Image.Resampling.LANCZOS)
        print(f"\nResized to 16x16")
    else:
        image_16x16 = image
    
    img_array_16x16 = np.array(image_16x16)
    print(f"16x16 array shape: {img_array_16x16.shape}")
    print(f"16x16 pixel value range: [{img_array_16x16.min()}, {img_array_16x16.max()}]")
    print(f"16x16 Mean: {img_array_16x16.mean():.2f}, Std: {img_array_16x16.std():.2f}")
    
    # Show sample pixels
    print(f"\nSample pixels (first 3x3):")
    print(img_array_16x16[:3, :3])
    
    return image, image_16x16

def compare_preprocessing(image_16x16):
    """Compare different preprocessing methods"""
    print(f"\n{'='*80}")
    print("PREPROCESSING COMPARISON")
    print(f"{'='*80}")
    
    # Method 1: Normalize(0.5, 0.5, 0.5) - used in app.py for material_detection_head
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensor1 = transform1(image_16x16)
    print(f"\nMethod 1: Normalize(0.5, 0.5, 0.5) - Maps [0,1] to [-1,1]")
    print(f"  Shape: {tensor1.shape}")
    print(f"  Range: [{tensor1.min():.4f}, {tensor1.max():.4f}]")
    print(f"  Mean: {tensor1.mean():.4f}, Std: {tensor1.std():.4f}")
    print(f"  Sample (first channel, 3x3):")
    print(f"    {tensor1[0, :3, :3]}")
    
    # Method 2: ImageNet normalization - used in material_detection_functions.py
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor2 = transform2(image_16x16)
    print(f"\nMethod 2: ImageNet Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
    print(f"  Shape: {tensor2.shape}")
    print(f"  Range: [{tensor2.min():.4f}, {tensor2.max():.4f}]")
    print(f"  Mean: {tensor2.mean():.4f}, Std: {tensor2.std():.4f}")
    print(f"  Sample (first channel, 3x3):")
    print(f"    {tensor2[0, :3, :3]}")
    
    # Method 3: No normalization (just ToTensor)
    transform3 = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor3 = transform3(image_16x16)
    print(f"\nMethod 3: No normalization (just ToTensor)")
    print(f"  Shape: {tensor3.shape}")
    print(f"  Range: [{tensor3.min():.4f}, {tensor3.max():.4f}]")
    print(f"  Mean: {tensor3.mean():.4f}, Std: {tensor3.std():.4f}")
    print(f"  Sample (first channel, 3x3):")
    print(f"    {tensor3[0, :3, :3]}")
    
    return tensor1, tensor2, tensor3

def test_with_different_preprocessing(weight_path, image_16x16):
    """Test model with different preprocessing methods"""
    print(f"\n{'='*80}")
    print("MODEL INFERENCE WITH DIFFERENT PREPROCESSING")
    print(f"{'='*80}")
    
    class_names = [
        '3dmodel', 'LEDscreen', 'bowl__purpleplastic', 'bowl__whiteceramic',
        'carrot__natural', 'eggplant__natural', 'greenpepper__natural',
        'potato__natural', 'redpepper__natural', 'teacup__ceramic',
        'tomato__natural', 'yellowpepper__natural'
    ]
    
    # Define model
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
    
    # Load model
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
    
    # Test with Method 1: Normalize(0.5, 0.5, 0.5)
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensor1 = transform1(image_16x16).unsqueeze(0)
    
    with torch.no_grad():
        pred1 = model(tensor1)
        prob1 = torch.nn.functional.softmax(pred1, dim=1)[0]
        idx1 = torch.argmax(prob1).item()
    
    print(f"\nMethod 1 (Normalize 0.5):")
    print(f"  Predicted: {class_names[idx1]} (index {idx1})")
    print(f"  Confidence: {prob1[idx1]:.4f} ({prob1[idx1]*100:.2f}%)")
    print(f"  Top 3:")
    top3_1 = torch.topk(prob1, 3)
    for i, (prob, idx) in enumerate(zip(top3_1.values, top3_1.indices)):
        print(f"    {i+1}. {class_names[idx.item()]}: {prob.item():.4f}")
    
    # Test with Method 2: ImageNet normalization
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor2 = transform2(image_16x16).unsqueeze(0)
    
    with torch.no_grad():
        pred2 = model(tensor2)
        prob2 = torch.nn.functional.softmax(pred2, dim=1)[0]
        idx2 = torch.argmax(prob2).item()
    
    print(f"\nMethod 2 (ImageNet Normalize):")
    print(f"  Predicted: {class_names[idx2]} (index {idx2})")
    print(f"  Confidence: {prob2[idx2]:.4f} ({prob2[idx2]*100:.2f}%)")
    print(f"  Top 3:")
    top3_2 = torch.topk(prob2, 3)
    for i, (prob, idx) in enumerate(zip(top3_2.values, top3_2.indices)):
        print(f"    {i+1}. {class_names[idx.item()]}: {prob.item():.4f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose material classifier')
    parser.add_argument('--weight-path', type=str, required=True,
                        help='Path to model weight file')
    parser.add_argument('--test-image', type=str, required=True,
                        help='Path to test image (PNG or STO file)')
    
    args = parser.parse_args()
    
    weight_path = os.path.abspath(args.weight_path) if not os.path.isabs(args.weight_path) else args.weight_path
    test_image_path = os.path.abspath(args.test_image) if not os.path.isabs(args.test_image) else args.test_image
    
    if not os.path.exists(weight_path):
        print(f"ERROR: Weight file not found: {weight_path}")
        sys.exit(1)
    
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found: {test_image_path}")
        sys.exit(1)
    
    # Inspect image
    image, image_16x16 = inspect_image(test_image_path)
    
    # Compare preprocessing
    tensor1, tensor2, tensor3 = compare_preprocessing(image_16x16)
    
    # Test with different preprocessing
    test_with_different_preprocessing(weight_path, image_16x16)
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

