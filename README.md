---
title: SPAD for Vision
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# SPAD for Vision - Demonstrating Combined Spatial and Structural Features

## Overview

This web application is a **demonstration platform** that showcases the benefits of combining **spatial** (geometric/positional) and **structural** (material/compositional) features for vision tasks. The application emphasizes that relying solely on spatially resolved images has limitations, and demonstrates how incorporating structural information improves detection and classification capabilities.

**Important Note**: This is **NOT** a traditional object detection model trained on massive datasets. Instead, it is a research demonstration that highlights the relevance of multi-modal feature fusion, particularly in scenarios where spatial information alone is insufficient.

## Core Concept

Traditional computer vision approaches often rely heavily on spatially resolved images (high-resolution geometric information). However, many real-world scenarios require understanding both:
- **Spatial features**: Where objects are located, their shapes, and geometric relationships
- **Structural features**: What materials objects are made of, their composition, and material properties

This web application demonstrates that combining both feature types leads to more robust and informative detection systems, especially in constrained or specialized domains where large-scale training data may not be available.

## Available Demos

### 1. Spatiotemporal Detection (`/spatiotemporal_detection`)
**Purpose**: The flagship demonstration that combines spatial object detection with structural material classification in a unified pipeline.

- **Spatial Component**: Detects and localizes objects in the scene (using YOLOv3, YOLOv8, or DINOv3)
- **Structural Component**: Classifies materials present in the scene (using material detection head)
- **Combined Output**: Provides both object locations and material properties, demonstrating how spatial and structural information complement each other
- **Input Format**: STO (Spatiotemporal Object) files containing both spatial and structural data

### 2. Material Detection - Flat Homogeneous Surfaces (`/flat_surface_detection`)
**Purpose**: Demonstrates material classification for flat, homogeneous surfaces where spatial features are minimal but structural features are critical.

- Focuses on material properties rather than object geometry
- Shows how structural features can be effective even when spatial information is limited
- Useful for quality control and material verification tasks

### 3. Material Detection - Fluid Purity (`/fluid_purity_demo`)
**Purpose**: Demonstrates material classification for fluid purity assessment, specifically for homogenized milk.

- Emphasizes structural features (composition, purity) over spatial features
- Shows application in food safety and quality control
- Demonstrates that structural analysis can be more relevant than spatial analysis for certain tasks

### 4. Material Detection Head (`/material_detection_head`)
**Purpose**: Standalone demonstration of the structural feature extraction component used in spatiotemporal detection.

- Shows material classification capabilities independently
- Processes 16×16 pixel patches to extract structural/material information
- Demonstrates that even low-resolution structural data can be highly informative

### 5. Custom Vision Models (YOLOv3, YOLOv8, DINOv3)
**Purpose**: Demonstrates spatial feature extraction using custom-trained vision models.

- **YOLOv3 Custom** (`/detect_yolov3`): Custom-trained YOLOv3 for object localization
- **YOLOv8 Custom** (`/custom_yolov8_demo`): Custom-trained YOLOv8 for object detection
- **DINOv3 Custom** (`/dinov3_demo`): Custom-trained DINOv3 for object classification

These models are **not** trained on massive datasets like ImageNet or COCO. Instead, they are specialized models trained on domain-specific data, demonstrating that effective spatial feature extraction can be achieved with focused, smaller-scale training.

## Key Features

- **Multi-Modal Fusion**: Combines spatial (object detection) and structural (material classification) features
- **Specialized Models**: Custom-trained models for specific domains rather than general-purpose object detection
- **STO File Support**: Handles Spatiotemporal Object files that contain both spatial and structural information
- **Real-time Inference**: Fast inference with optimized models
- **Interactive Demos**: Multiple demonstration interfaces for different aspects of the system

## Technical Details

- **Framework**: Flask
- **Python Version**: 3.10
- **Deployment**: Docker
- **Port**: 7860 (Hugging Face Spaces)

## Model and Dataset Sources

Models and datasets are hosted on Hugging Face Hub:
- **Models**: Individual model repositories under `mvplus/` organization
  - `mvplus/dinov3`
  - `mvplus/flat_surface`
  - `mvplus/material_detection_head`
  - `mvplus/material_purity`
  - `mvplus/spatiotemporal`
  - `mvplus/yolov3`
  - `mvplus/yolov8`
- **Datasets**: Individual dataset repositories under `mvplus/` organization
  - `mvplus/testmages__flatsurface`
  - `mvplus/testmages__milkpurity`
  - `mvplus/testmages__yolov3`
  - `mvplus/testmages__yolov8`
  - `mvplus/testmages_dino`
  - `mvplus/testmages_spatiotemporal`
  - `mvplus/val_natural_material_detection`

## Usage

1. **Spatiotemporal Detection**: Upload a `.sto` file, select spatial and material detection models, and run detection to see combined results
2. **Material Detection**: Upload images or STO files to classify materials based on structural features
3. **Spatial Detection**: Upload images to detect and localize objects using custom-trained vision models

## Research Context

This application serves as a demonstration of research findings that:
- Spatial features alone are insufficient for many real-world vision tasks
- Structural/material features provide complementary information
- Combining both feature types improves detection and classification performance
- Specialized, domain-focused models can be effective without massive training datasets

The models used in this demonstration are **not** general-purpose object detection systems trained on large-scale datasets. Instead, they are specialized models designed to showcase the benefits of multi-modal feature fusion in constrained domains.

## Repository Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates for each demo interface
- `static/`: CSS, JavaScript, and static assets
- `apps.err/material_detection_naturalobjects/`: Material detection functions and spatiotemporal processing
- `Dockerfile`: Container configuration for deployment

## License

See repository for license information.
