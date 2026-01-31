# Troubleshooting Guide: Image Compression Causing Incorrect Material Classification

## Problem Summary

**Issue**: Material classifier incorrectly predicting classes (e.g., LED images predicted as "3dmodel" instead of "LEDscreen")

**Root Cause**: JPEG compression (lossy) in image extraction endpoints causing pixel value changes that affect model predictions

**Date Resolved**: 2025-01-XX

**Status**: ✅ Resolved

---

## Symptoms

1. **Incorrect Predictions**: Material classifier making wrong predictions, especially for LED images
   - LED images predicted as "3dmodel" (index 0) instead of "LEDscreen" (index 1)
   - Other material classes may also be misclassified

2. **Pixel Value Changes**: Images extracted from STO files show significant pixel value differences
   - Original image: Pixel range [0, 241], Mean: 14.80, Std: 40.26
   - After JPEG compression: Pixel range [0, 148], Mean: 16.03, Std: 23.63
   - Max pixel difference: 169 pixels

3. **Tensor Differences**: Normalized tensors show significant differences
   - Original tensor std: 0.3160
   - JPEG-compressed tensor std: 0.1854
   - Max tensor difference: 1.325 in normalized values

---

## Root Cause Analysis

### The Problem

The image processing pipeline was using **JPEG compression (lossy)** when extracting images from STO files:

1. **STO File Structure**:
   - Index 0: Metadata list
   - Index 1: 16×16 RGB image (material detection input)
   - Index 2: Object detection metadata
   - Index 3: 640×640 RGB image (object detection input)

2. **Image Extraction Flow**:
   ```
   STO File → Extract Index 1 → Convert to JPEG → Base64 → Frontend → 
   Convert back to Image → Send to API → Model Inference
   ```

3. **JPEG Compression Impact**:
   - JPEG is a **lossy compression** format
   - For small 16×16 images, compression artifacts are significant
   - Pixel values change during compression/decompression
   - These changes affect model predictions, especially for small images

### Why This Matters

- **16×16 images are very small**: Any pixel value change is proportionally large
- **Material classifier is sensitive**: Small pixel differences can change predictions
- **Compression artifacts**: JPEG introduces visual artifacts that don't match training data
- **Normalization amplifies differences**: After normalization to [-1, 1], small pixel differences become larger

---

## Solution Applied

### Change Made

**Before**:
```python
# Convert to base64
img_buffer = io.BytesIO()
image.save(img_buffer, format='JPEG', quality=95)  # ❌ Lossy compression
img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

return jsonify({
    'success': True,
    'image': f"data:image/jpeg;base64,{img_base64}",  # ❌ JPEG format
    ...
})
```

**After**:
```python
# Convert to base64 using PNG (lossless) to preserve pixel values
# CRITICAL: JPEG compression causes pixel value changes that affect model predictions
# PNG is lossless and preserves exact pixel values, especially important for 16x16 images
img_buffer = io.BytesIO()
image.save(img_buffer, format='PNG')  # ✅ Lossless compression
img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

return jsonify({
    'success': True,
    'image': f"data:image/png;base64,{img_base64}",  # ✅ PNG format
    ...
})
```

### Files Modified

1. **`app.py`**:
   - `/api/extract_sto_index0` endpoint: Changed from JPEG to PNG
   - `/api/extract_sto_index1` endpoint: Changed from JPEG to PNG
   - Updated MIME types from `image/jpeg` to `image/png`

### Verification

After the fix:
- ✅ Pixel values preserved exactly (no compression artifacts)
- ✅ Model predictions are accurate
- ✅ LED images correctly predict as "LEDscreen"
- ✅ All material classes predict correctly

---

## How to Diagnose Similar Issues

### Step 1: Check Image Processing Pipeline

1. **Identify where images are compressed**:
   ```bash
   grep -r "format='JPEG'" app.py
   grep -r "image/jpeg" app.py
   ```

2. **Check image extraction endpoints**:
   - `/api/extract_sto_index0` (material detection)
   - `/api/extract_sto_index1` (object detection)

3. **Verify image format in frontend**:
   - Check if frontend converts images to JPEG
   - Check if base64 encoding uses JPEG

### Step 2: Trace Image Processing

Use the diagnostic script to trace image processing:

```python
# trace_image_processing.py
python3 trace_image_processing.py "path/to/test.sto"
```

Look for:
- Pixel value differences between original and processed images
- Compression artifacts
- Tensor differences after normalization

### Step 3: Check Model Predictions

1. **Test with known images**:
   ```bash
   python3 evaluate_material_classifier.py \
       --weight-path "path/to/weights.pth" \
       --test-image "path/to/led_image.sto" \
       --expected-class "LEDscreen"
   ```

2. **Compare predictions**:
   - Direct STO extraction vs. API extraction
   - Original image vs. compressed image

### Step 4: Verify Image Statistics

Check if images have been modified:
```python
import numpy as np
from PIL import Image

# Original image
original = Image.open("original.png")
orig_array = np.array(original)
print(f"Original: mean={orig_array.mean():.2f}, std={orig_array.std():.2f}")

# Processed image
processed = Image.open("processed.png")
proc_array = np.array(processed)
print(f"Processed: mean={proc_array.mean():.2f}, std={proc_array.std():.2f}")

# Check difference
diff = np.abs(orig_array.astype(int) - proc_array.astype(int))
print(f"Max difference: {diff.max()}")
if diff.max() > 10:  # Significant difference
    print("⚠️  WARNING: Images differ significantly - compression may be the issue")
```

---

## Prevention Guidelines

### Best Practices

1. **Use Lossless Formats for Small Images**:
   - ✅ PNG for 16×16 images (material detection)
   - ✅ PNG for any image used for model inference
   - ❌ Avoid JPEG for small images (< 64×64)

2. **Use JPEG Only for Display**:
   - ✅ JPEG is fine for display images (640×640, larger)
   - ✅ JPEG is fine for previews and thumbnails
   - ❌ Never use JPEG for model input images

3. **Preserve Pixel Values**:
   - Always use lossless compression for model inputs
   - Avoid multiple compression/decompression cycles
   - Preserve exact pixel values from STO files

4. **Verify Image Integrity**:
   - Add checksums or hash verification
   - Compare pixel statistics before/after processing
   - Log image statistics for debugging

### Code Review Checklist

When reviewing image processing code, check:

- [ ] Are images compressed before model inference?
- [ ] Is JPEG used for small images (< 64×64)?
- [ ] Are pixel values preserved through the pipeline?
- [ ] Are there multiple compression/decompression cycles?
- [ ] Is the image format consistent with training data?

---

## Related Issues

### Similar Problems to Watch For

1. **Image Resizing Issues**:
   - Multiple resizes can cause pixel value changes
   - Different resize algorithms produce different results
   - Always resize once, use consistent algorithm (LANCZOS)

2. **Color Space Conversion**:
   - RGB ↔ YUV conversions can change pixel values
   - Always ensure RGB mode before model input

3. **Normalization Mismatch**:
   - Different normalization (ImageNet vs. [-1,1]) causes issues
   - Ensure normalization matches training

4. **Image Format Mismatch**:
   - PIL Image vs. NumPy array conversions
   - Ensure consistent format throughout pipeline

---

## Testing the Fix

### Test Script

```bash
# Test with LED image
python3 evaluate_material_classifier.py \
    --weight-path "apps/material_detection_head_custom/model_results/saved_models/epoch_399_Accuracy_98.25.pth" \
    --test-image "datasets/testmages_spatiotemporal/bowl__LEDimagepurpleplastic_7.sto" \
    --expected-class "LEDscreen"

# Test via API
python3 evaluate_material_classifier.py \
    --weight-path "apps/material_detection_head_custom/model_results/saved_models/epoch_399_Accuracy_98.25.pth" \
    --test-image "datasets/testmages_spatiotemporal/bowl__LEDimagepurpleplastic_7.sto" \
    --use-api \
    --api-url "http://localhost:7888"
```

### Expected Results

- ✅ LED images predict as "LEDscreen" (index 1)
- ✅ All material classes predict correctly
- ✅ No pixel value changes in extracted images
- ✅ Tensor values match original images

---

## References

- **STO File Structure**: See `STO_FILE_STRUCTURE.md`
- **Material Classifier Evaluation**: See `MATERIAL_CLASSIFIER_EVALUATION.md`
- **Image Processing Script**: `trace_image_processing.py`
- **Evaluation Script**: `evaluate_material_classifier.py`

---

## Key Takeaways

1. **Always use lossless compression (PNG) for model input images**
2. **JPEG compression causes pixel value changes that affect predictions**
3. **Small images (16×16) are especially sensitive to compression**
4. **Trace image processing pipeline to identify compression points**
5. **Verify pixel values are preserved through the entire pipeline**

---

## Quick Reference

**Problem**: Incorrect material classification predictions

**Cause**: JPEG compression changing pixel values

**Solution**: Use PNG (lossless) instead of JPEG

**Files Changed**: `app.py` (extract_sto_index0, extract_sto_index1 endpoints)

**Verification**: Test with LED images - should predict "LEDscreen" correctly

---

*Last Updated: 2025-01-XX*
*Document Version: 1.0*

