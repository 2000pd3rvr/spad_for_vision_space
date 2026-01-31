# Models Consolidation

## Overview

All model files have been consolidated into a centralized `models/` folder at the same level as `local_spad_for_vision/`, similar to how datasets were consolidated. This provides a single location for all model weights and checkpoints, making it easier to manage and reference models across the application.

## Models Folder Structure

```
models/  (sibling to local_spad_for_vision/)
├── material_detection_head/     # Material detection head models
│   ├── epoch_399_Accuracy_98.25.pth
│   ├── epoch_398_Accuracy_98.1875__2025-09-18 02:12:23.905420.pth
│   └── backup/                 # Backup weights
│       └── epoch_186_Accuracy_80.pth
├── material_purity/             # Material purity classification models
│   ├── epoch_194_Accuracy_100.pth
│   ├── epoch_198_Accuracy_100.0__2025-11-06 09:11:41.505146.pth
│   └── ... (many training checkpoints)
├── flat_surface/                # Flat surface detection models
│   └── ... (training checkpoints)
├── yolov3/                      # YOLOv3 object detection models
│   ├── yolov3.pt
│   └── best_epoch_*.pt
├── yolov8/                      # YOLOv8 object detection models
│   ├── yolov8s.pt
│   ├── best.pt
│   └── last.pt
├── dinov3/                      # DINOv3 models
│   ├── pretrained/
│   │   └── dinov3_vitb16_pretrain.pth
│   └── lastweight_epoch_*.pth
├── spatiotemporal/              # Spatiotemporal detection models
│   ├── weights_MD.pth
│   ├── weights_OD.pt
│   └── training_results_material_classifier_best_99.25%.pth
└── pretrained/                  # Pretrained models
    └── yolov8n.pt
```

**Note:** Models are now located in `models/` (sibling to `local_spad_for_vision/`) to prepare for Hugging Face Spaces deployment where they will be in separate repositories.

## Migration Summary

### Files Moved
- **1,181 model files** were copied to the `models/` folder (sibling to `local_spad_for_vision/`)
- Models organized by type into subdirectories
- Original files remain in their original locations (copied, not moved)

### Code Updates

#### `app.py`
Updated all model path references:
- `apps/material_detection_head_custom/model_results/saved_models/` → `../models/material_detection_head/`
- `apps/material_purity/app_weights/` → `../models/material_purity/`
- `apps/flat_surface_detection/app_weights/` → `../models/flat_surface/`
- `apps/yolov3_custom/runs/train/fast_yolov3/weights/` → `../models/yolov3/`
- `apps/yolov8_custom/runs/detect/train/weights/` → `../models/yolov8/`
- `apps/dinov3_custom/production_results/.../weights/` → `../models/dinov3/`

#### `material_detection_functions.py`
- Updated default model path to use `../models/spatiotemporal/`

#### Documentation
- `MATERIAL_CLASSIFIER_EVALUATION.md` - Updated all example paths
- `test_material_classifier.sh` - Updated default weight path
- `test_led_image.py` - Updated default weight path

## Usage

### Accessing Models in Code

All model paths should now reference the `../models/` folder (relative to `local_spad_for_vision/`, prepared for Hugging Face Spaces deployment):

```python
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Material detection head
weight_path = os.path.join(BASE_DIR, "..", "models", "material_detection_head", "epoch_399_Accuracy_98.25.pth")

# Material purity
weight_path = os.path.join(BASE_DIR, "..", "models", "material_purity", "epoch_194_Accuracy_100.pth")

# YOLOv3
weight_path = os.path.join(BASE_DIR, "..", "models", "yolov3", "best_epoch_146_train_0_0002_val_0_0004_fitness_0_9996.pt")

# YOLOv8
weight_path = os.path.join(BASE_DIR, "..", "models", "yolov8", "best.pt")

# DINOv3
weight_path = os.path.join(BASE_DIR, "..", "models", "dinov3", "lastweight_epoch_82_train_0_0411_val_0_1622_acc_96_88%.pth")
```

### API Endpoints

All API endpoints that list model weights now automatically look in the `models/` folder:

- `/api/material_detection_head_weights` → `../models/material_detection_head/`
- `/api/material_purity_weights` → `../models/material_purity/`
- `/api/flat_surface_weights` → `../models/flat_surface/`
- `/api/yolov3_weights` → `../models/yolov3/`
- `/api/yolov8_weights` → `../models/yolov8/`
- `/api/dinov3_weights` → `../models/dinov3/`

## Benefits

1. **Centralized Management**: All models in one location
2. **Consistent Paths**: Uniform path structure across the application
3. **Easier Deployment**: Single folder to backup or deploy
4. **Better Organization**: Models grouped by type
5. **Simplified Updates**: Easy to update model references

## Notes

- Original model files remain in their original locations (files were copied, not moved)
- The consolidation script (`consolidate_models.py`) can be run again to update if new models are added
- Some duplicate filenames were renamed with parent directory prefix to avoid conflicts
- Model loading code automatically handles both old and new paths for backward compatibility

## Verification

To verify models are accessible:

```bash
# Check material detection head
ls ../models/material_detection_head/epoch_399_Accuracy_98.25.pth

# Check material purity
ls ../models/material_purity/epoch_194_Accuracy_100.pth

# List all model types
ls -d ../models/*/
```

## Future Additions

When adding new models:

1. Place them in the appropriate `../models/<type>/` subdirectory (relative to `local_spad_for_vision/`)
2. Update any hardcoded paths if necessary
3. Models will automatically appear in API endpoints that scan the models folder

---

*Last Updated: 2025-01-31*
*Total Models Consolidated: 1,181 files*
*Location: `models/` (sibling to `local_spad_for_vision/`, prepared for Hugging Face Spaces deployment)*

