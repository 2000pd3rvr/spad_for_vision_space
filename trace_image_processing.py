#!/usr/bin/env python3
"""
Trace image processing pipeline to see what happens to the image after extraction
"""

import pickle
import base64
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

def trace_image_processing(sto_path):
    """Trace what happens to an image from STO extraction to model input"""
    print(f"\n{'='*80}")
    print(f"TRACING IMAGE PROCESSING PIPELINE")
    print(f"{'='*80}")
    print(f"STO file: {sto_path}")
    print(f"{'='*80}\n")
    
    # Step 1: Extract from STO file
    print("STEP 1: Extract from STO file (index 1)")
    print("-" * 80)
    with open(sto_path, 'rb') as f:
        sto_data = pickle.load(f)
    
    original_image = sto_data[1]
    if not isinstance(original_image, Image.Image):
        print(f"ERROR: Index 1 is not a PIL Image")
        return
    
    print(f"Original image from STO:")
    print(f"  Size: {original_image.size}")
    print(f"  Mode: {original_image.mode}")
    print(f"  Format: {original_image.format}")
    
    img_array_orig = np.array(original_image)
    print(f"  Pixel range: [{img_array_orig.min()}, {img_array_orig.max()}]")
    print(f"  Mean: {img_array_orig.mean():.2f}, Std: {img_array_orig.std():.2f}")
    print(f"  Sample pixels (first 3x3, RGB):")
    print(f"    {img_array_orig[:3, :3]}")
    
    # Step 2: Convert to RGB (if needed)
    print(f"\nSTEP 2: Convert to RGB")
    print("-" * 80)
    if original_image.mode != 'RGB':
        image_rgb = original_image.convert('RGB')
        print(f"  Converted from {original_image.mode} to RGB")
    else:
        image_rgb = original_image.copy()
        print(f"  Already RGB, no conversion needed")
    
    img_array_rgb = np.array(image_rgb)
    print(f"  After RGB conversion:")
    print(f"    Pixel range: [{img_array_rgb.min()}, {img_array_rgb.max()}]")
    print(f"    Mean: {img_array_rgb.mean():.2f}, Std: {img_array_rgb.std():.2f}")
    print(f"    Arrays equal: {np.array_equal(img_array_orig, img_array_rgb)}")
    
    # Step 3: Resize to 16x16 (if needed)
    print(f"\nSTEP 3: Resize to 16x16")
    print("-" * 80)
    if image_rgb.size != (16, 16):
        image_16x16 = image_rgb.resize((16, 16), Image.Resampling.LANCZOS)
        print(f"  Resized from {image_rgb.size} to (16, 16) using LANCZOS")
    else:
        image_16x16 = image_rgb.copy()
        print(f"  Already 16x16, no resize needed")
    
    img_array_16x16 = np.array(image_16x16)
    print(f"  After resize:")
    print(f"    Pixel range: [{img_array_16x16.min()}, {img_array_16x16.max()}]")
    print(f"    Mean: {img_array_16x16.mean():.2f}, Std: {img_array_16x16.std():.2f}")
    
    # Step 4: Simulate frontend conversion (base64 -> blob -> PIL)
    print(f"\nSTEP 4: Simulate frontend conversion (base64 -> blob -> PIL)")
    print("-" * 80)
    # Convert to JPEG (as frontend does)
    img_buffer_jpeg = io.BytesIO()
    image_16x16.save(img_buffer_jpeg, format='JPEG', quality=95)
    img_buffer_jpeg.seek(0)
    jpeg_bytes = img_buffer_jpeg.getvalue()
    print(f"  Converted to JPEG: {len(jpeg_bytes)} bytes")
    
    # Convert back to PIL (as API does)
    image_from_jpeg = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
    img_array_jpeg = np.array(image_from_jpeg)
    print(f"  After JPEG round-trip:")
    print(f"    Size: {image_from_jpeg.size}")
    print(f"    Mode: {image_from_jpeg.mode}")
    print(f"    Pixel range: [{img_array_jpeg.min()}, {img_array_jpeg.max()}]")
    print(f"    Mean: {img_array_jpeg.mean():.2f}, Std: {img_array_jpeg.std():.2f}")
    print(f"    Arrays equal to original: {np.array_equal(img_array_16x16, img_array_jpeg)}")
    print(f"    Max difference: {np.abs(img_array_16x16.astype(int) - img_array_jpeg.astype(int)).max()}")
    
    # Step 5: Apply transforms
    print(f"\nSTEP 5: Apply transforms (ToTensor + Normalize)")
    print("-" * 80)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Test with original 16x16 image
    tensor_original = transform(image_16x16)
    print(f"  Tensor from original 16x16:")
    print(f"    Shape: {tensor_original.shape}")
    print(f"    Range: [{tensor_original.min():.4f}, {tensor_original.max():.4f}]")
    print(f"    Mean: {tensor_original.mean():.4f}, Std: {tensor_original.std():.4f}")
    
    # Test with JPEG round-trip image
    tensor_jpeg = transform(image_from_jpeg)
    print(f"  Tensor from JPEG round-trip:")
    print(f"    Shape: {tensor_jpeg.shape}")
    print(f"    Range: [{tensor_jpeg.min():.4f}, {tensor_jpeg.max():.4f}]")
    print(f"    Mean: {tensor_jpeg.mean():.4f}, Std: {tensor_jpeg.std():.4f}")
    print(f"    Tensors equal: {torch.allclose(tensor_original, tensor_jpeg, atol=1e-5)}")
    print(f"    Max difference: {(tensor_original - tensor_jpeg).abs().max():.6f}")
    
    # Step 6: Check if resize happens again
    print(f"\nSTEP 6: Check if resize happens again in API")
    print("-" * 80)
    # The API checks if size != (16, 16) and resizes again
    if image_from_jpeg.size != (16, 16):
        image_resized_again = image_from_jpeg.resize((16, 16), Image.Resampling.LANCZOS)
        print(f"  API would resize again from {image_from_jpeg.size} to (16, 16)")
        tensor_resized_again = transform(image_resized_again)
        print(f"  Tensor after double resize:")
        print(f"    Range: [{tensor_resized_again.min():.4f}, {tensor_resized_again.max():.4f}]")
        print(f"    Mean: {tensor_resized_again.mean():.4f}, Std: {tensor_resized_again.std():.4f}")
        print(f"    Difference from original: {(tensor_original - tensor_resized_again).abs().max():.6f}")
    else:
        print(f"  No double resize (already 16x16)")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("Potential issues:")
    print("1. JPEG compression (lossy) when converting to base64 in frontend")
    print("2. Double resize if image is not exactly 16x16 after JPEG round-trip")
    print("3. Color space conversion issues")
    print("4. Image format changes")

if __name__ == '__main__':
    import sys
    sto_path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/testmages_spatiotemporal/bowl__LEDimagepurpleplastic_7.sto'
    trace_image_processing(sto_path)

