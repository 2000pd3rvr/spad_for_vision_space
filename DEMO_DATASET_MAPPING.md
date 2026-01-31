# Demo App to Dataset Path Mapping

This document maps each demo application to its corresponding test dataset directory.

## Base Directory
All paths are relative to: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision/`

**Note:** All test datasets have been moved to a centralized `datasets/` folder for better organization.

## Demo Mappings

### 1. DINOv3 Custom Demo
- **Route**: `/dinov3_demo`
- **App Directory**: `apps/dinov3_custom/`
- **Test Dataset Path**: `datasets/testmages_dino/`
- **Full Path**: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision/datasets/testmages_dino/`
- **Description**: Contains test images for DINOv3 custom model testing

### 2. YOLOv3 Custom Demo
- **Route**: `/detect_yolov3` (referenced in demos.html)
- **App Directory**: `apps/yolov3_custom/`
- **Test Dataset Path**: `datasets/testmages__yolov3/`
- **Full Path**: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision/datasets/testmages__yolov3/`
- **Description**: Test images for YOLOv3 custom object detection

### 3. YOLOv8 Custom Demo
- **Route**: `/custom_yolov8_demo`
- **App Directory**: `apps/yolov8_custom/`
- **Test Dataset Path**: `datasets/testmages__yolov8/`
- **Full Path**: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision/datasets/testmages__yolov8/`
- **Description**: Test images for YOLOv8 custom model

### 4. Flat Surface Detection
- **Route**: `/flat_surface_detection`
- **App Directory**: `apps/flat_surface_detection/`
- **Test Dataset Path**: `datasets/testmages__flatsurface/`
- **Full Path**: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision/datasets/testmages__flatsurface/`
- **Description**: Test images organized by class (BCB, BNT, WGF, WNT)
- **Classes**: 
  - BCB (Blue Ceramic Board)
  - BNT (Brown Natural Tile)
  - WGF (White Glass Fiber)
  - WNT (White Natural Tile)

### 5. Material Purity / Fluid Purity Demo
- **Route**: `/fluid_purity_demo`
- **App Directory**: `apps/material_purity/`
- **Test Dataset Path**: `datasets/testmages__milkpurity/`
- **Full Path**: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision/datasets/testmages__milkpurity/`
- **Description**: Test images for fluid purity detection (pure vs impure milk)
- **Classes**:
  - `pure/` - Pure milk samples
  - `impure/` - Impure milk samples

### 6. Material Detection Head
- **Route**: `/material_detection_head`
- **App Directory**: `apps/material_detection_head_custom/`
- **Test Dataset Path**: `datasets/val_natural_material_detection/`
- **Full Path**: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision/datasets/val_natural_material_detection/`
- **Description**: Validation dataset for natural material detection (12 or 18 classes)

### 7. Spatiotemporal Detection
- **Route**: `/spatiotemporal_detection`
- **App Directory**: `apps/spatiotemporal_detection/`
- **Test Dataset Path**: `datasets/testmages_spatiotemporal/`
- **Full Path**: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision/datasets/testmages_spatiotemporal/`
- **Description**: Test STO (spatiotemporal) files for temporal sequence analysis

## Summary Table

| Demo Route | App Directory | Test Dataset Path |
|------------|---------------|-------------------|
| `/dinov3_demo` | `apps/dinov3_custom/` | `datasets/testmages_dino/` |
| `/detect_yolov3` | `apps/yolov3_custom/` | `datasets/testmages__yolov3/` |
| `/custom_yolov8_demo` | `apps/yolov8_custom/` | `datasets/testmages__yolov8/` |
| `/flat_surface_detection` | `apps/flat_surface_detection/` | `datasets/testmages__flatsurface/` |
| `/fluid_purity_demo` | `apps/material_purity/` | `datasets/testmages__milkpurity/` |
| `/material_detection_head` | `apps/material_detection_head_custom/` | `datasets/val_natural_material_detection/` |
| `/spatiotemporal_detection` | `apps/spatiotemporal_detection/` | `datasets/testmages_spatiotemporal/` |

## Notes

- All test dataset directories use the naming convention: `testmages_*` or `val_*` (note: "testmages" appears to be a typo for "testimages" but is consistently used)
- Some apps also have training/validation datasets in `data/` subdirectories
- The spatiotemporal detection uses `.sto` files (pickle format) rather than standard image files
- Material detection head uses validation data from training, not a separate test set

