---
title: SPAD for Vision - Spatiotemporal Detection & Material Classification
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
tags:
  - computer-vision
  - object-detection
  - material-classification
  - spatiotemporal-detection
  - yolov3
  - yolov8
  - dinov3
  - pytorch
  - deep-learning
  - vision-systems
  - multi-modal-fusion
  - material-analysis
  - quality-control
  - computer-vision-demo
---

# SPAD for Vision - Demonstrating Combined Spatial and Structural Features

**Keywords**: spatiotemporal detection, material classification, object detection, computer vision, YOLOv3, YOLOv8, DINOv3, SPAD vision, spatial features, structural features, multi-modal fusion, material detection, vision systems, deep learning, PyTorch, Flask web app, STO files, spatiotemporal object detection, material property classification, custom vision models, domain-specific models, quality control vision, food safety detection, material verification, structural analysis, spatial analysis, combined features, feature fusion, vision research, specialized models, material purity detection, flat surface detection, natural object detection, material composition analysis, computer vision demo, vision demonstration platform

## Overview

This web application is a **demonstration platform** for **spatiotemporal detection** and **material classification** that showcases the benefits of combining **spatial** (geometric/positional) and **structural** (material/compositional) features for computer vision tasks. The application emphasizes that relying solely on spatially resolved images has limitations, and demonstrates how incorporating structural information improves detection and classification capabilities.

**Search Terms**: spatiotemporal object detection, material classification system, combined spatial and structural features, multi-modal computer vision, YOLOv3 custom model, YOLOv8 custom model, DINOv3 custom model, material detection head, material property classification, vision system with material analysis, SPAD-based vision, spatiotemporal vision processing

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
- **Computer Vision**: Advanced deep learning models for object detection and material classification
- **Material Analysis**: Structural feature extraction for material property identification
- **Spatiotemporal Processing**: Combined spatial-temporal analysis for comprehensive scene understanding

## Technical Details

- **Framework**: Flask (Python web framework)
- **Python Version**: 3.10
- **Deep Learning**: PyTorch-based models (YOLOv3, YOLOv8, DINOv3, custom CNNs)
- **Deployment**: Docker containerized application
- **Port**: 7860 (Hugging Face Spaces)
- **Computer Vision Libraries**: OpenCV, PIL/Pillow, torchvision
- **Model Formats**: PyTorch (.pth), YOLO (.pt) checkpoints

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

## Search & Discovery

This repository and application are optimized for discovery through search terms including:
- **Primary**: spatiotemporal detection, material classification, object detection, computer vision
- **Models**: YOLOv3, YOLOv8, DINOv3, custom vision models, PyTorch models
- **Features**: spatial features, structural features, multi-modal fusion, material properties
- **Applications**: quality control, food safety, material verification, material analysis
- **Technology**: SPAD vision, deep learning, computer vision systems, vision research
- **File Formats**: STO files, spatiotemporal object files, material detection data

## License

See repository for license information.
