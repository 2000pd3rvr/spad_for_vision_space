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
from PIL import Image, ImageDraw


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch infer on .png files using saved classifier weights')
    parser.add_argument('--weights', type=str, default='training_results/material_classifier_best_99.25%.pth', help='Path to .pth checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .png files')
    args = parser.parse_args()

    device = torch.device('cpu')

    ckpt = torch.load(args.weights, map_location=device)
    class_names = ckpt.get('class_names')
    num_classes = ckpt.get('num_classes', len(class_names) if class_names else 12)

    model = MaterialClassifier(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    transform = get_val_transform()

    input_dir = Path(args.input_dir)
    png_files = sorted([p for p in input_dir.glob('*.png')])
    if not png_files:
        print(f"No .png files found in {input_dir}")
        sys.exit(0)

    for png_path in png_files:
        try:
            img = load_png_as_rgb(str(png_path))
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            names = class_names if class_names else [f'class_{i}' for i in range(len(probs))]
            result = {name: float(prob) for name, prob in zip(names, probs)}

            # Save JSON next to .png
            out_json = png_path.with_suffix('.probs.json')
            with open(out_json, 'w') as f:
                json.dump(result, f, indent=2)

            # Save a small preview PNG with top-3 overlay
            top3 = sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:3]
            preview = img.resize((128, 128), Image.NEAREST).convert('RGB')
            draw = ImageDraw.Draw(preview)
            y = 4
            for name, p in top3:
                draw.text((4, y), f"{name}: {p*100:.2f}%", fill=(255, 0, 0))
                y += 14
            out_png = png_path.with_suffix('.preview.png')
            preview.save(out_png)

            print(f"Saved: {out_json.name}, {out_png.name}")
        except Exception as e:
            print(f"Failed {png_path.name}: {e}")


if __name__ == '__main__':
    main()


