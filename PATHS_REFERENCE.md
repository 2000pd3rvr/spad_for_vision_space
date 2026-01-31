# Path Reference for spad_for_vision_space

## Base Directory
- **BASE_DIR**: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision_space`
  - Defined in `app.py` as: `os.path.dirname(os.path.abspath(__file__))`

## Models Search Paths

### From app.py (main application):
All model weights are searched in: `BASE_DIR/models/<model_type>/`

1. **Flat Surface Detection**: `BASE_DIR/models/flat_surface/*.pth`
2. **Material Purity**: `BASE_DIR/models/material_purity/*.pth`
3. **Material Detection Head**: `BASE_DIR/models/material_detection_head/*.pth`
4. **YOLOv3**: `BASE_DIR/models/yolov3/*.pt`
5. **YOLOv8**: `BASE_DIR/models/yolov8/*.pt`
6. **DINOv3**: `BASE_DIR/models/dinov3/*.pth`

### From material_detection_functions.py:
**Spatiotemporal Material Detection Model**: `BASE_DIR/models/spatiotemporal/training_results_material_classifier_best_99.25%.pth`

**Note**: The path calculation in `material_detection_functions.py` goes up 2 levels from the file location:
- File location: `apps.err/material_detection_naturalobjects/material_detection_functions.py`
- Goes up to: `spad_for_vision_space/`
- Then: `models/spatiotemporal/`

## Test Images/Datasets Search Paths

All test images are searched in: `BASE_DIR/datasets/<dataset_name>/`

1. **Flat Surface Detection**: `BASE_DIR/datasets/testmages__flatsurface/`
2. **Fluid Purity Demo**: `BASE_DIR/datasets/testmages__milkpurity/`
3. **DINOv3 Demo**: `BASE_DIR/datasets/testmages_dino/`
4. **Custom YOLOv8 Demo**: `BASE_DIR/datasets/testmages__yolov8/`
5. **Spatiotemporal Detection**: `BASE_DIR/datasets/testmages_spatiotemporal/`
6. **Detect YOLOv3**: `BASE_DIR/datasets/testmages__yolov3/`
7. **Material Detection Head**: `BASE_DIR/datasets/val_natural_material_detection/`

## Model Architecture Files

1. **YOLOv3 Custom**: `BASE_DIR/apps/yolov3_custom/train_fast_yolov3.py`
2. **DINOv3 Custom**: `BASE_DIR/apps/dinov3_custom/train_dinov3.py`

## Hugging Face Hub Download

On startup, if models/datasets don't exist locally, the app will attempt to download from:
- **Models**: `mvplus/spatiotemporal_models` (repo-type: model)
- **Datasets**: `mvplus/spatiotemporal_dataset` (repo-type: dataset)

**Note**: Downloads are skipped for local testing (only happen on Hugging Face Spaces).

