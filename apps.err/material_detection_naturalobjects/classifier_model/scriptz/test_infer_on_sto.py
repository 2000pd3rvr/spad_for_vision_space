#!/usr/bin/env python3
import os
import sys
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class MaterialClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(MaterialClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def load_sto_index1_as_rgb(sto_path: str) -> Image.Image:
    with open(sto_path, 'rb') as f:
        data = pickle.load(f)
    # Expect index 1 to be HxW (or HxW or 16x16) array; convert to RGB
    arr = np.array(data[1])
    if arr.ndim == 2:
        # If values are not uint8, scale to 0-255
        if arr.dtype != np.uint8:
            arr_min = float(arr.min())
            arr_max = float(arr.max())
            denom = (arr_max - arr_min) if (arr_max - arr_min) != 0 else 1.0
            arr = ((arr - arr_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        rgb = Image.fromarray(arr, mode='L').convert('RGB')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # Already RGB-like
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        rgb = Image.fromarray(arr, mode='RGB')
    else:
        raise ValueError(f"Unexpected array shape at index 1: {arr.shape}")
    # Ensure 16x16 as in training; if different, resize with nearest to avoid blur
    if rgb.size != (16, 16):
        rgb = rgb.resize((16, 16), Image.NEAREST)
    return rgb


def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def infer(weights_path: str, sto_path: str):
    device = torch.device('cpu')

    # Load checkpoint
    ckpt = torch.load(weights_path, map_location=device)
    class_names = ckpt.get('class_names')
    num_classes = ckpt.get('num_classes', len(class_names) if class_names else 12)

    model = MaterialClassifier(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Prepare input
    image_rgb = load_sto_index1_as_rgb(sto_path)
    transform = get_val_transform()
    x = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Build result mapping
    if not class_names:
        class_names = [f'class_{i}' for i in range(len(probs))]
    result = {name: float(prob) for name, prob in zip(class_names, probs)}

    # Print sorted
    sorted_items = sorted(result.items(), key=lambda kv: kv[1], reverse=True)
    print("Top predictions:")
    for name, p in sorted_items[:5]:
        print(f"  {name}: {p*100:.2f}%")
    print("\nAll class probabilities (json):")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Infer on .sto (index1->RGB) using saved classifier weights')
    parser.add_argument('--weights', type=str, default='training_results/material_classifier_best_99.25%.pth', help='Path to .pth checkpoint')
    parser.add_argument('--sto', type=str, required=True, help='Path to .sto file')
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        print(f"Weights not found: {args.weights}")
        sys.exit(1)
    if not os.path.isfile(args.sto):
        print(f".sto not found: {args.sto}")
        sys.exit(1)

    infer(args.weights, args.sto)


