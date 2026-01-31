# STO File Structure for Spatiotemporal Detection

## Overview

STO (Spatiotemporal Object) files are pickle-serialized Python lists containing images and metadata for both material detection and object detection models.

## File Structure

STO files contain **4 items** in the following order:

### Index 0: Metadata List
- **Type**: `list`
- **Content**: `[frame_number, class_label]`
- **Example**: `[70, 'bowl__LEDimagepurpleplastic']`
- **Purpose**: Contains frame number and class label metadata

### Index 1: Material Detection Image
- **Type**: `PIL.Image.Image` (RGB)
- **Size**: **16×16 pixels**
- **Mode**: RGB
- **Purpose**: **Input for material detection model**
- **Usage**: This is the image used by the material classifier to predict material type

### Index 2: Object Detection Metadata
- **Type**: `list` with one element
- **Content**: `[[x, y, width, height, class_label]]`
- **Example**: `[[279.85, 463.38, 266.15, 233.85, 'bowl']]`
- **Purpose**: Contains bounding box coordinates and class label for object detection

### Index 3: Object Detection Image
- **Type**: `PIL.Image.Image` (RGB)
- **Size**: **640×640 pixels**
- **Mode**: RGB
- **Purpose**: **Input for object detection model**
- **Usage**: This is the image used by object detection models (YOLOv3, YOLOv8, DINOv3) to detect objects

## Code Usage

### Material Detection

```python
import pickle
from PIL import Image
from io import BytesIO

# Load STO file
with open('file.sto', 'rb') as f:
    sto_data = pickle.load(f)

# Extract material detection image (Index 1)
material_image = sto_data[1]  # 16x16 PIL Image
if material_image.mode != 'RGB':
    material_image = material_image.convert('RGB')
```

### Object Detection

```python
# Extract object detection image (Index 3)
object_image = sto_data[3]  # 640x640 PIL Image
if object_image.mode != 'RGB':
    object_image = object_image.convert('RGB')
```

## API Endpoints

### Material Detection
- **Endpoint**: `/api/detect_material_head`
- **STO Index Used**: **Index 1** (16×16 image)
- **Note**: The endpoint automatically extracts index 1 from STO files

### Object Detection
- **Endpoints**: `/api/detect_yolov3`, `/api/detect_yolov8_custom`, `/api/detect_dinov3`
- **STO Index Used**: **Index 3** (640×640 image)

### Image Extraction Endpoints
- **`/api/extract_sto_index0`**: Extracts **Index 1** (16×16 material detection image)
  - *Note: Despite the name "index0", this extracts index 1*
- **`/api/extract_sto_index1`**: Extracts **Index 3** (640×640 object detection image)
  - *Note: Despite the name "index1", this extracts index 3*

## Important Notes

1. **Index Confusion**: The API endpoint names (`extract_sto_index0`, `extract_sto_index1`) are misleading:
   - `extract_sto_index0` actually extracts **index 1** (material detection)
   - `extract_sto_index1` actually extracts **index 3** (object detection)
   - This naming is kept for backward compatibility with the frontend

2. **File Validation**: Always check that the STO file has at least 4 items:
   ```python
   if len(sto_data) < 4:
       raise ValueError('Invalid STO file: need at least 4 items')
   ```

3. **Image Sizes**: 
   - Material detection images are always **16×16 pixels**
   - Object detection images are always **640×640 pixels**
   - If images don't match these sizes, they should be resized accordingly

4. **Image Format**: All images are in RGB mode. Convert if necessary:
   ```python
   if image.mode != 'RGB':
       image = image.convert('RGB')
   ```

## Example STO File Analysis

```
File: bowl__LEDimagepurpleplastic_14_dup_838.sto
Size: 1,238,257 bytes (1.21 MB)

Index 0: [70, 'bowl__LEDimagepurpleplastic']  (metadata)
Index 1: PIL Image, 16×16, RGB                (material detection)
Index 2: [[279.85, 463.38, 266.15, 233.85, 'bowl']]  (OD metadata)
Index 3: PIL Image, 640×640, RGB              (object detection)
```

## Updated Code

All code has been updated to use the correct indices:
- Material detection: **Index 1** (not index 0)
- Object detection: **Index 3** (not index 1)

This ensures correct inference from both models.

