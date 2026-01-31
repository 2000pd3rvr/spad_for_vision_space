#!/usr/bin/env python3
"""
Script to consolidate all model files into the models folder
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"

# Define model file mappings: (source_pattern, target_subdirectory)
MODEL_MAPPINGS = [
    # Material detection head
    ("apps/material_detection_head_custom/model_results/saved_models/*.pth", "material_detection_head"),
    ("apps/material_detection_head_custom/backup_weights/*.pth", "material_detection_head/backup"),
    
    # Material purity
    ("apps/material_purity/app_weights/*.pth", "material_purity"),
    ("apps/material_purity/model__BC_pureVsimpure/**/*.pth", "material_purity"),
    
    # Flat surface
    ("apps/flat_surface_detection/data/**/saved_models/*.pth", "flat_surface"),
    ("apps/flat_surface_detection/xive/saved_models/*.pth", "flat_surface"),
    
    # YOLOv3
    ("apps/yolov3_custom/yolov3.pt", "yolov3"),
    ("apps/yolov3_custom/runs/train/**/weights/*.pt", "yolov3"),
    
    # YOLOv8
    ("apps/yolov8_custom/yolov8s.pt", "yolov8"),
    ("apps/yolov8_custom/runs/detect/train/weights/*.pt", "yolov8"),
    
    # DINOv3
    ("apps/dinov3_custom/dinov3_vitb16_pretrain.pth", "dinov3/pretrained"),
    ("apps/dinov3_custom/production_results/**/weights/*.pth", "dinov3"),
    
    # Spatiotemporal models
    ("apps.err/material_detection_naturalobjects/weights/*.pth", "spatiotemporal"),
    ("apps.err/material_detection_naturalobjects/spatiotemporal_model/weights/*.pth", "spatiotemporal"),
    ("apps.err/material_detection_naturalobjects/classifier_model/training_results/*.pth", "spatiotemporal"),
    ("apps.err/material_detection_naturalobjects/classifier_model/*.pth", "spatiotemporal"),
    
    # Root level pretrained
    ("yolov8n.pt", "pretrained"),
]

def consolidate_models():
    """Move all model files to models folder"""
    import glob
    
    moved_count = 0
    skipped_count = 0
    
    print(f"Consolidating models to: {MODELS_DIR}")
    print("=" * 80)
    
    for source_pattern, target_subdir in MODEL_MAPPINGS:
        source_path = BASE_DIR / source_pattern
        target_dir = MODELS_DIR / target_subdir
        
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all matching files
        matches = list(BASE_DIR.glob(source_pattern))
        
        if not matches:
            # Try with ** for recursive
            if "**" not in source_pattern:
                recursive_pattern = source_pattern.replace("/*", "/**/*")
                matches = list(BASE_DIR.glob(recursive_pattern))
        
        for source_file in matches:
            if not source_file.is_file():
                continue
            
            # Get relative path to preserve structure if needed
            filename = source_file.name
            
            # Handle duplicates by including parent directory name
            target_file = target_dir / filename
            
            # If file already exists, add parent dir to name
            if target_file.exists() and target_file != source_file:
                parent_name = source_file.parent.name
                target_file = target_dir / f"{parent_name}_{filename}"
            
            try:
                # Copy file (we'll move after verifying)
                shutil.copy2(source_file, target_file)
                print(f"✓ Copied: {source_file.relative_to(BASE_DIR)} -> {target_file.relative_to(BASE_DIR)}")
                moved_count += 1
            except Exception as e:
                print(f"✗ Error copying {source_file}: {e}")
                skipped_count += 1
    
    print("=" * 80)
    print(f"Total files copied: {moved_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"\nModels consolidated to: {MODELS_DIR}")
    print("\nNext step: Update code references to point to models/ directory")

if __name__ == "__main__":
    consolidate_models()

