#!/bin/bash
# Quick test script for material classifier evaluation

# Default values
WEIGHT_PATH="${1:-../models/material_detection_head/epoch_399_Accuracy_98.25.pth}"
TEST_IMAGE="${2:-datasets/testmages_spatiotemporal/carrot__natural_1.sto}"
API_URL="${3:-http://localhost:7888}"

echo "=========================================="
echo "Material Classifier Evaluation Test"
echo "=========================================="
echo "Weight path: $WEIGHT_PATH"
echo "Test image: $TEST_IMAGE"
echo "API URL: $API_URL"
echo "=========================================="
echo ""

# Check if files exist
if [ ! -f "$WEIGHT_PATH" ]; then
    echo "ERROR: Weight file not found: $WEIGHT_PATH"
    exit 1
fi

if [ ! -f "$TEST_IMAGE" ]; then
    echo "ERROR: Test image not found: $TEST_IMAGE"
    exit 1
fi

# Test 1: Local inference
echo "Test 1: Local Inference"
echo "----------------------"
python3 evaluate_material_classifier.py \
    --weight-path "$WEIGHT_PATH" \
    --test-image "$TEST_IMAGE" \
    --expected-class "carrot__natural"

echo ""
echo "=========================================="
echo ""

# Test 2: API inference (if server is running)
echo "Test 2: API Inference"
echo "----------------------"
if curl -s "$API_URL" > /dev/null 2>&1; then
    python3 evaluate_material_classifier.py \
        --weight-path "$WEIGHT_PATH" \
        --test-image "$TEST_IMAGE" \
        --use-api \
        --api-url "$API_URL"
else
    echo "WARNING: Server not running at $API_URL"
    echo "Skipping API test. Start the server with: python app.py"
fi

echo ""
echo "=========================================="
echo "Evaluation complete!"

