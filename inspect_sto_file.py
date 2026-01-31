#!/usr/bin/env python3
"""Inspect STO file structure and contents"""

import pickle
import os
from PIL import Image
import numpy as np
from io import BytesIO

file_path = 'datasets/testmages_spatiotemporal/bowl__LEDimagepurpleplastic_14_dup_838.sto'

if not os.path.exists(file_path):
    print(f'File not found: {file_path}')
    exit(1)

print('='*80)
print('STO FILE ANALYSIS')
print('='*80)
print(f'File: {file_path}')
print(f'File size: {os.path.getsize(file_path)} bytes')
print()

# Load the STO file
with open(file_path, 'rb') as f:
    try:
        data = pickle.load(f)
    except Exception as e:
        print(f'Error loading pickle: {e}')
        # Try with numpy compatibility
        try:
            f.seek(0)
            data = pickle.load(f, encoding='latin1')
            print('Loaded with latin1 encoding')
        except Exception as e2:
            print(f'Error with latin1 encoding: {e2}')
            exit(1)

print(f'Data type: {type(data)}')
if hasattr(data, '__len__'):
    print(f'Data length: {len(data)}')
else:
    print('Data length: N/A')
print()

# Inspect each index
if isinstance(data, (list, tuple)):
    for i, item in enumerate(data):
        print(f'Index {i}:')
        print(f'  Type: {type(item).__name__}')
        
        if isinstance(item, bytes):
            print(f'  Size: {len(item)} bytes')
            try:
                img = Image.open(BytesIO(item))
                print(f'  Image format: {img.format}')
                print(f'  Image size: {img.size}')
                print(f'  Image mode: {img.mode}')
            except Exception as e:
                print(f'  Error opening as image: {e}')
        elif isinstance(item, Image.Image):
            print(f'  Image size: {item.size}')
            print(f'  Image mode: {item.mode}')
            print(f'  Image format: {item.format if hasattr(item, "format") else "N/A"}')
            # Get pixel statistics
            arr = np.array(item)
            print(f'  Pixel range: [{arr.min()}, {arr.max()}]')
            print(f'  Pixel mean: {arr.mean():.2f}, std: {arr.std():.2f}')
        elif isinstance(item, np.ndarray):
            print(f'  Array shape: {item.shape}')
            print(f'  Array dtype: {item.dtype}')
            print(f'  Array min/max: {item.min()}/{item.max()}')
            print(f'  Array mean: {item.mean():.2f}, std: {item.std():.2f}')
        elif item is None:
            print(f'  Value: None')
        else:
            val_str = str(item)
            if len(val_str) > 100:
                val_str = val_str[:100] + '...'
            print(f'  Value: {val_str}')
        print()
else:
    print(f'Data structure: {data}')

print('='*80)
print('STO FILE STRUCTURE SUMMARY')
print('='*80)
if isinstance(data, (list, tuple)) and len(data) >= 2:
    print('Standard STO structure:')
    print('  Index 0: Model input image (typically 16x16)')
    print('  Index 1: Display image (any size)')
    if len(data) > 2:
        print(f'  Additional indices: {len(data) - 2} more items')

