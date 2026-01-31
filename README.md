---
title: SPAD for Vision
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# SPAD for Vision - Spatiotemporal Detection

A comprehensive vision detection system for spatiotemporal object and material detection using multiple deep learning models.

## Features

- **Spatiotemporal Detection**: Combined object and material detection from STO files
- **Multiple Model Support**: YOLOv8, YOLOv3, and DINOv3 for object detection
- **Material Classification**: Advanced material detection using custom trained models
- **Real-time Inference**: Fast inference with optimized models

## Models

The application uses models downloaded from:
- Models: `mvplus/spatiotemporal_models`
- Datasets: `mvplus/spatiotemporal_dataset`

## Usage

1. Upload a `.sto` file for spatiotemporal detection
2. Select spatial and material detection heads
3. Choose model weights
4. Run detection to get object and material predictions

## Technical Details

- **Framework**: Flask
- **Python Version**: 3.10
- **Deployment**: Docker
- **Port**: 7860 (Hugging Face Spaces)

