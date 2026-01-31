#!/usr/bin/env python3
"""
Interactive inference script that processes files one by one,
prints results to terminal, and waits for user input to continue.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pickle
from pathlib import Path

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

def load_png_as_rgb(png_path: str) -> Image.Image:
    """Load PNG file and convert to RGB, ensuring 16x16 size"""
    rgb = Image.open(png_path).convert('RGB')
    if rgb.size != (16, 16):
        rgb = rgb.resize((16, 16), Image.NEAREST)
    return rgb

def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def interactive_inference():
    """Interactive inference on PNG files in test_sto"""
    
    print("=" * 80)
    print("INTERACTIVE MATERIAL CLASSIFICATION")
    print("=" * 80)
    
    # Configuration
    weights_path = "training_results/material_classifier_best_99.25%.pth"
    input_dir = "test_sto"
    
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        return
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return
    
    # Load model
    print(f"\n1. Loading model from {weights_path}...")
    device = torch.device('cpu')
    
    try:
        ckpt = torch.load(weights_path, map_location=device)
        class_names = ckpt.get('class_names')
        num_classes = ckpt.get('num_classes', len(class_names) if class_names else 12)
        
        model = MaterialClassifier(num_classes=num_classes)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"   ✓ Model loaded successfully!")
        print(f"   ✓ Classes: {num_classes}")
        print(f"   ✓ Class names: {class_names}")
        
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Get PNG files
    print(f"\n2. Scanning for PNG files in {input_dir}...")
    png_files = sorted([f for f in Path(input_dir).glob('*.png')])
    
    if not png_files:
        print(f"   ✗ No PNG files found in {input_dir}")
        return
    
    print(f"   ✓ Found {len(png_files)} PNG files")
    
    # Setup transforms
    transform = get_val_transform()
    
    print(f"\n3. Starting interactive inference...")
    print("   Press ENTER to process next file, 'q' + ENTER to quit")
    print("-" * 80)
    
    processed = 0
    for i, file_path in enumerate(png_files):
        try:
            # Load and preprocess image
            img = load_png_as_rgb(str(file_path))
            x = transform(img).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get predictions
            names = class_names if class_names else [f'class_{i}' for i in range(len(probs))]
            predictions = [(name, float(prob)) for name, prob in zip(names, probs)]
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Display results
            print(f"\n[{i+1:4d}/{len(png_files)}] File: {file_path.name}")
            print(f"     Path: {file_path}")
            print(f"     Image size: {img.size}")
            print(f"\n     Top 5 Predictions:")
            for j, (name, prob) in enumerate(predictions[:5]):
                print(f"       {j+1}. {name:25s}: {prob*100:6.2f}%")
            
            print(f"\n     All Class Probabilities:")
            for name, prob in predictions:
                print(f"       {name:25s}: {prob*100:6.2f}%")
            
            processed += 1
            
            # Wait for user input
            print(f"\n     Processed: {processed}/{len(png_files)} files")
            user_input = input("     Press ENTER to continue, 'q' + ENTER to quit: ").strip().lower()
            
            if user_input == 'q':
                print(f"\n   ✓ Quitting early. Processed {processed}/{len(png_files)} files")
                break
                
        except Exception as e:
            print(f"\n[{i+1:4d}/{len(png_files)}] ERROR processing {file_path.name}")
            print(f"     Error: {e}")
            user_input = input("     Press ENTER to continue, 'q' + ENTER to quit: ").strip().lower()
            
            if user_input == 'q':
                print(f"\n   ✓ Quitting early. Processed {processed}/{len(png_files)} files")
                break
    
    print(f"\n" + "=" * 80)
    print(f"INTERACTIVE INFERENCE COMPLETE")
    print(f"Processed: {processed}/{len(png_files)} files")
    print("=" * 80)

if __name__ == '__main__':
    interactive_inference()
