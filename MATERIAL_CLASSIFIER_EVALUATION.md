# Material Classifier Evaluation

This document describes how to evaluate the material classifier to ensure it makes correct inference from selected weights in the spatiotemporal detection endpoint.

## Overview

The evaluation script `evaluate_material_classifier.py` tests the material classifier by:
1. Loading a model with specified weights
2. Processing a test image (PNG or STO file)
3. Running inference and displaying detailed results
4. Optionally validating against expected class labels

## Usage

### Basic Local Evaluation

Test the material classifier locally (without Flask server):

```bash
python3 evaluate_material_classifier.py \
    --weight-path "../models/material_detection_head/epoch_399_Accuracy_98.25.pth" \
    --test-image "datasets/testmages_spatiotemporal/carrot__natural_1.sto"
```

### With Expected Class Validation

Validate that the prediction matches an expected class:

```bash
python3 evaluate_material_classifier.py \
    --weight-path "../models/material_detection_head/epoch_399_Accuracy_98.25.pth" \
    --test-image "datasets/testmages_spatiotemporal/carrot__natural_1.sto" \
    --expected-class "carrot__natural"
```

### API Evaluation

Test via the Flask API endpoint (server must be running):

```bash
python3 evaluate_material_classifier.py \
    --weight-path "../models/material_detection_head/epoch_399_Accuracy_98.25.pth" \
    --test-image "datasets/testmages_spatiotemporal/carrot__natural_1.sto" \
    --use-api \
    --api-url "http://localhost:7888"
```

### Quick Test Script

Use the provided shell script for quick testing:

```bash
./test_material_classifier.sh [weight_path] [test_image] [api_url]
```

Example:
```bash
./test_material_classifier.sh \
    "models/material_detection_head/epoch_399_Accuracy_98.25.pth" \
    "datasets/testmages_spatiotemporal/carrot__natural_1.sto" \
    "http://localhost:7888"
```

## Available Weights

Material detection head weights are located in:
```
../models/material_detection_head/
```

Available weights can be listed via the API:
```bash
curl http://localhost:7888/api/material_detection_head_weights
```

## Test Images

Test images are available in:
- `datasets/testmages_spatiotemporal/` - STO files for spatiotemporal detection
- `datasets/val_natural_material_detection/` - Validation images organized by class

## Class Names

The material classifier uses 12 classes in alphabetical order (as ImageFolder sorts them):

1. `3dmodel`
2. `LEDscreen`
3. `bowl__purpleplastic`
4. `bowl__whiteceramic`
5. `carrot__natural`
6. `eggplant__natural`
7. `greenpepper__natural`
8. `potato__natural`
9. `redpepper__natural`
10. `teacup__ceramic`
11. `tomato__natural`
12. `yellowpepper__natural`

## Output

The evaluation script provides:
- Predicted class and confidence
- Top 3 predictions with probabilities
- All class probabilities
- Validation result (if expected class provided)
- Inference time (for API tests)

## Example Output

```
================================================================================
RESULTS
================================================================================
Predicted class: greenpepper__natural (index 6)
Confidence: 0.9669 (96.69%)

Top 3 predictions:
  1. greenpepper__natural: 0.9669 (96.69%)
  2. carrot__natural: 0.0331 (3.31%)
  3. potato__natural: 0.0000 (0.00%)

All class probabilities:
   0. 3dmodel                  : 0.0000 (0.00%)
   1. LEDscreen                : 0.0000 (0.00%)
   ...
   6. greenpepper__natural     : 0.9669 (96.69%) <-- PREDICTED
   ...
```

## Troubleshooting

### Weight file not found
- Ensure the weight path is correct and absolute, or relative to the project root
- Check that the weight file exists in the specified location

### Test image not found
- Verify the image path is correct
- For STO files, ensure they contain valid pickle data with at least one image

### API connection failed
- Ensure the Flask server is running: `python app.py`
- Check that the server is accessible at the specified URL (default: http://localhost:7888)

### Incorrect predictions
- Verify the weight file matches the expected model architecture
- Check that image preprocessing matches training (16x16, normalized to [-1, 1])
- Ensure class order matches training (alphabetical ImageFolder order)
- **Note**: Some test images may be mislabeled or ambiguous. If a prediction doesn't match the expected class:
  - Test with multiple images of the same class to verify consistency
  - Check if the image filename accurately reflects its content
  - Review the top-3 predictions - if the expected class appears in top-3, the model may be seeing similar features
  - Use the diagnostic script (`diagnose_material_classifier.py`) to inspect the image and preprocessing

## Diagnostic Tools

### Image and Preprocessing Inspection

Use `diagnose_material_classifier.py` to inspect extracted images and compare preprocessing methods:

```bash
python3 diagnose_material_classifier.py \
    --weight-path "../models/material_detection_head/epoch_399_Accuracy_98.25.pth" \
    --test-image "datasets/testmages_spatiotemporal/carrot__natural_1.sto"
```

This script will:
- Show image statistics (size, pixel ranges, mean, std)
- Compare different preprocessing methods (Normalize 0.5 vs ImageNet normalization)
- Test model predictions with each preprocessing method
- Help identify if preprocessing mismatch is causing issues

## Integration with Spatiotemporal Detection

The material classifier is used in the `/spatiotemporal_detection` endpoint:
1. User uploads an STO file
2. Frontend extracts index 0 image
3. Sends to `/api/detect_material_head` with selected weight path
4. Backend loads model, runs inference, returns predictions

The evaluation script tests the same pipeline to ensure correctness.

