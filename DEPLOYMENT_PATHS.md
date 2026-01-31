# Deployment Paths Configuration

## Overview
This document explains how paths work for both local testing and Hugging Face Spaces deployment.

## Directory Structure

### Local Structure
```
huggingface/
  ├── models/              # Sibling to spad_for_vision_space
  │   ├── spatiotemporal/
  │   ├── material_detection_head/
  │   └── ...
  ├── datasets/             # Sibling to spad_for_vision_space
  │   ├── testmages_spatiotemporal/
  │   └── ...
  └── spad_for_vision_space/
      └── app.py           # BASE_DIR
```

### Hugging Face Spaces Structure
```
/app/                      # BASE_DIR (Space root)
  ├── models/              # Downloaded from Hub
  │   ├── spatiotemporal/
  │   └── ...
  ├── datasets/            # Downloaded from Hub
  │   ├── testmages_spatiotemporal/
  │   └── ...
  └── app.py
```

## Path Resolution

### Base Directory
- **BASE_DIR**: Always resolves to the directory containing `app.py`
- **Local**: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision_space`
- **Hugging Face Spaces**: `/app` (or wherever the Space runs from)

### Models Paths
The app uses `get_models_dir()` helper function to determine the correct path:

**Local Testing:**
- Models are siblings to `spad_for_vision_space`: `../models/<model_type>/`
- Example: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/models/spatiotemporal/training_results_material_classifier_best_99.25%.pth`

**Hugging Face Spaces:**
- Models are downloaded from `mvplus/spatiotemporal_models` to: `BASE_DIR/models/`
- The download preserves folder structure, so files end up in: `BASE_DIR/models/<model_type>/<file>`
- Example: `/app/models/spatiotemporal/training_results_material_classifier_best_99.25%.pth`

### Dataset Paths
The app uses `get_datasets_dir()` helper function to determine the correct path:

**Local Testing:**
- Datasets are siblings to `spad_for_vision_space`: `../datasets/<dataset_name>/`
- Example: `/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/datasets/testmages_spatiotemporal/`

**Hugging Face Spaces:**
- Datasets are downloaded from `mvplus/spatiotemporal_dataset` to: `BASE_DIR/datasets/`
- The download preserves folder structure, so files end up in: `BASE_DIR/datasets/<dataset_name>/`
- Example: `/app/datasets/testmages_spatiotemporal/`

## Upload Strategy

### Models Upload
When uploading models to `mvplus/spatiotemporal_models`, preserve the folder structure:
- Upload `spatiotemporal/training_results_material_classifier_best_99.25%.pth` (not just the file)
- This ensures it downloads to `BASE_DIR/models/spatiotemporal/training_results_material_classifier_best_99.25%.pth`

### Datasets Upload
When uploading datasets to `mvplus/spatiotemporal_dataset`, preserve the folder structure:
- Upload `testmages_spatiotemporal/` directory (with all files inside)
- This ensures it downloads to `BASE_DIR/datasets/testmages_spatiotemporal/`

## Verification

The `setup_huggingface_resources()` function:
1. Downloads to `BASE_DIR/models/` and `BASE_DIR/datasets/`
2. Verifies expected files exist after download
3. Prints warnings if expected structure is not found

## Key Points

✅ **Consistent Paths**: All paths use `BASE_DIR`, which works the same locally and on Hugging Face Spaces

✅ **Folder Structure Preserved**: Uploads maintain folder structure, so downloads end up in correct locations

✅ **Automatic Download**: On Hugging Face Spaces, models/datasets download automatically on first startup

✅ **Local Skip**: Local testing skips downloads (models/datasets must be present locally)

