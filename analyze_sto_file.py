#!/usr/bin/env python3
"""
Analyze and describe an STO file structure and contents.
"""

import pickle
import os
import sys
from PIL import Image
import numpy as np
from io import BytesIO

def analyze_sto_file(sto_path):
    """Analyze an STO file and describe its contents"""
    print(f"\n{'='*80}")
    print(f"STO File Analysis")
    print(f"{'='*80}")
    print(f"File path: {sto_path}")
    print(f"File exists: {os.path.exists(sto_path)}")
    
    if not os.path.exists(sto_path):
        print(f"ERROR: File not found!")
        return
    
    # Get file size
    file_size = os.path.getsize(sto_path)
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    
    # Load STO file
    print(f"\n{'='*80}")
    print("Loading STO file...")
    print(f"{'='*80}")
    
    try:
        with open(sto_path, 'rb') as f:
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
        
        print(f"✓ Successfully loaded STO file")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, (list, tuple)):
            print(f"Data structure: {type(data).__name__} with {len(data)} items")
            
            for i, item in enumerate(data):
                print(f"\n{'='*80}")
                print(f"Index {i}:")
                print(f"{'='*80}")
                print(f"  Type: {type(item)}")
                
                if isinstance(item, Image.Image):
                    print(f"  PIL Image")
                    print(f"  Size: {item.size}")
                    print(f"  Mode: {item.mode}")
                    print(f"  Format: {item.format}")
                    
                    # Convert to numpy for analysis
                    img_array = np.array(item)
                    print(f"  Array shape: {img_array.shape}")
                    print(f"  Array dtype: {img_array.dtype}")
                    print(f"  Pixel range: [{img_array.min()}, {img_array.max()}]")
                    print(f"  Mean: {img_array.mean():.2f}, Std: {img_array.std():.2f}")
                    
                elif isinstance(item, bytes):
                    print(f"  Bytes object")
                    print(f"  Length: {len(item):,} bytes")
                    
                    # Try to open as image
                    try:
                        img = Image.open(BytesIO(item))
                        print(f"  ✓ Can be opened as image")
                        print(f"  Image size: {img.size}")
                        print(f"  Image mode: {img.mode}")
                        print(f"  Image format: {img.format}")
                        
                        # Convert to numpy
                        img_array = np.array(img)
                        print(f"  Array shape: {img_array.shape}")
                        print(f"  Array dtype: {img_array.dtype}")
                        print(f"  Pixel range: [{img_array.min()}, {img_array.max()}]")
                        print(f"  Mean: {img_array.mean():.2f}, Std: {img_array.std():.2f}")
                    except Exception as e:
                        print(f"  ✗ Cannot be opened as image: {e}")
                
                elif isinstance(item, np.ndarray):
                    print(f"  NumPy array")
                    print(f"  Shape: {item.shape}")
                    print(f"  Dtype: {item.dtype}")
                    print(f"  Size: {item.nbytes:,} bytes")
                    print(f"  Pixel range: [{item.min()}, {item.max()}]")
                    print(f"  Mean: {item.mean():.2f}, Std: {item.std():.2f}")
                    
                    # Try to convert to PIL Image
                    try:
                        if len(item.shape) == 2:
                            img = Image.fromarray(item, mode='L')
                        elif len(item.shape) == 3:
                            if item.shape[2] == 3:
                                img = Image.fromarray(item, mode='RGB')
                            elif item.shape[2] == 4:
                                img = Image.fromarray(item, mode='RGBA')
                            else:
                                img = None
                        else:
                            img = None
                        
                        if img:
                            print(f"  ✓ Can be converted to PIL Image")
                            print(f"  Image size: {img.size}")
                            print(f"  Image mode: {img.mode}")
                    except Exception as e:
                        print(f"  ✗ Cannot convert to PIL Image: {e}")
                
                elif isinstance(item, dict):
                    print(f"  Dictionary with {len(item)} keys:")
                    for key in item.keys():
                        print(f"    - {key}: {type(item[key])}")
                
                else:
                    print(f"  Value: {str(item)[:100]}...")
                    if hasattr(item, '__len__'):
                        try:
                            print(f"  Length: {len(item)}")
                        except:
                            pass
        
        elif isinstance(data, dict):
            print(f"Data structure: Dictionary with {len(data)} keys")
            for key in data.keys():
                print(f"  Key: {key}, Type: {type(data[key])}")
        
        else:
            print(f"Data structure: {type(data)}")
            print(f"Value: {str(data)[:200]}...")
        
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        print(f"File: {os.path.basename(sto_path)}")
        print(f"Size: {file_size:,} bytes")
        if isinstance(data, (list, tuple)):
            print(f"Contains {len(data)} items")
            if len(data) > 0:
                print(f"Index 0 type: {type(data[0])}")
                if len(data) > 1:
                    print(f"Index 1 type: {type(data[1])}")
        
    except Exception as e:
        import traceback
        print(f"\n{'='*80}")
        print("ERROR")
        print(f"{'='*80}")
        print(f"Error loading STO file: {e}")
        print(f"\nTraceback:")
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_sto_file.py <path_to_sto_file>")
        sys.exit(1)
    
    sto_path = sys.argv[1]
    analyze_sto_file(sto_path)

