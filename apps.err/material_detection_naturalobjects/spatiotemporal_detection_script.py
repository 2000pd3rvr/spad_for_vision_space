#!/usr/bin/env python3
"""
Spatiotemporal Detection Script for STO Files
Handles .sto files with at least 2 indices and displays PIL image at index 0
"""

import pickle
import sys
import os
from PIL import Image
import numpy as np
import base64
import io

class NumpyCompatibleUnpickler(pickle.Unpickler):
    """Custom unpickler to handle numpy version compatibility issues"""
    def find_class(self, module, name):
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        elif module == 'numpy._core.umath':
            module = 'numpy.core.umath'
        return super().find_class(module, name)

def validate_sto_structure(sto_file_path):
    """
    Step 1: Validate .sto file has at least 2 indices
    
    Args:
        sto_file_path (str): Path to the .sto file
        
    Returns:
        tuple: (is_valid, data, error_message)
    """
    try:
        print(f"Validating STO file: {sto_file_path}")
        
        # Load STO file with numpy compatibility
        with open(sto_file_path, 'rb') as f:
            try:
                unpickler = NumpyCompatibleUnpickler(f)
                data = unpickler.load()
            except Exception:
                f.seek(0)
                data = pickle.load(f, encoding='latin1')
        
        print(f"STO file loaded successfully. Length: {len(data)}")
        print(f"Data types: {[type(item).__name__ for item in data]}")
        
        # Check if we have at least 2 items
        if len(data) < 2:
            error_msg = f"Invalid .sto file format: expected at least 2 items, got {len(data)}"
            print(f"❌ {error_msg}")
            return False, None, error_msg
        
        # Check if index 0 is a PIL Image
        if not isinstance(data[0], Image.Image):
            error_msg = f"Index 0 must be a PIL Image, got {type(data[0])}"
            print(f"❌ {error_msg}")
            return False, None, error_msg
        
        print("✅ STO file structure validation passed")
        return True, data, None
        
    except Exception as e:
        error_msg = f"Error validating STO file: {str(e)}"
        print(f"❌ {error_msg}")
        return False, None, error_msg

def extract_index0_image(data):
    """
    Step 2: Extract PIL image at index 0
    
    Args:
        data (list): Loaded STO file data
        
    Returns:
        tuple: (image, error_message)
    """
    try:
        print("Extracting PIL image at index 0...")
        
        # Get the image at index 0
        image = data[0]
        
        # Ensure it's a PIL Image
        if not isinstance(image, Image.Image):
            error_msg = f"Index 0 is not a PIL Image: {type(image)}"
            print(f"❌ {error_msg}")
            return None, error_msg
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            print(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        print(f"✅ Successfully extracted PIL image at index 0")
        print(f"   Image size: {image.size}")
        print(f"   Image mode: {image.mode}")
        
        return image, None
        
    except Exception as e:
        error_msg = f"Error extracting image at index 0: {str(e)}"
        print(f"❌ {error_msg}")
        return None, error_msg

def image_to_base64(image):
    """
    Convert PIL Image to base64 string for display
    
    Args:
        image (PIL.Image): Image to convert
        
    Returns:
        str: Base64 encoded image string
    """
    try:
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return img_base64
        
    except Exception as e:
        print(f"❌ Error converting image to base64: {str(e)}")
        return None

def process_sto_file(sto_file_path):
    """
    Main function to process STO file and extract image at index 0
    
    Args:
        sto_file_path (str): Path to the .sto file
        
    Returns:
        dict: Results containing success status, image data, and error messages
    """
    print("=" * 60)
    print("SPATIOTEMPORAL DETECTION SCRIPT")
    print("=" * 60)
    
    # Step 1: Validate STO file structure
    print("\nStep 1: Validating STO file structure...")
    is_valid, data, error_msg = validate_sto_structure(sto_file_path)
    
    if not is_valid:
        return {
            'success': False,
            'error': error_msg,
            'image': None
        }
    
    # Step 2: Extract PIL image at index 0
    print("\nStep 2: Extracting PIL image at index 0...")
    image, error_msg = extract_index0_image(data)
    
    if image is None:
        return {
            'success': False,
            'error': error_msg,
            'image': None
        }
    
    # Step 3: Convert to base64 for display
    print("\nStep 3: Converting image to base64 for display...")
    img_base64 = image_to_base64(image)
    
    if img_base64 is None:
        return {
            'success': False,
            'error': 'Failed to convert image to base64',
            'image': None
        }
    
    print("\n✅ SUCCESS: STO file processed successfully!")
    print(f"   Image size: {image.size}")
    print(f"   Image mode: {image.mode}")
    print(f"   Base64 length: {len(img_base64)} characters")
    
    return {
        'success': True,
        'error': None,
        'image': img_base64,
        'image_size': image.size,
        'image_mode': image.mode,
        'sto_length': len(data)
    }

def main():
    """Test the script with AAAA.sto file"""
    # Test with AAAA.sto file
    sto_file_path = "/Users/pd3rvr/Documents/object_detection/multiwebapp/AAAA.sto"
    
    if not os.path.exists(sto_file_path):
        print(f"❌ STO file not found: {sto_file_path}")
        return
    
    # Process the STO file
    result = process_sto_file(sto_file_path)
    
    if result['success']:
        print(f"\n🎉 SUCCESS!")
        print(f"   Image size: {result['image_size']}")
        print(f"   Image mode: {result['image_mode']}")
        print(f"   STO length: {result['sto_length']}")
        print(f"   Base64 length: {len(result['image'])} characters")
    else:
        print(f"\n❌ FAILED: {result['error']}")

if __name__ == "__main__":
    main()
