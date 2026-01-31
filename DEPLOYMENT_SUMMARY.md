# Deployment Summary - Hugging Face Spaces

## ✅ Completed Steps

### 1. Folder Reorganization
- ✅ Created sibling-level structure:
  ```
  huggingface/
  ├── models/              # All model weights (separate HF Model repo)
  ├── datasets/            # All test datasets (separate HF Dataset repo)
  └── local_spad_for_vision/  # Application code (HF Space repo)
  ```
- ✅ Models and datasets are now siblings to `local_spad_for_vision/`

### 2. Path Updates
- ✅ Updated all model paths in `app.py`:
  - `models/` → `../models/` (relative to `local_spad_for_vision/`)
- ✅ Updated all dataset paths in `app.py`:
  - `datasets/` → `../datasets/` (relative to `local_spad_for_vision/`)
- ✅ Updated `material_detection_functions.py`:
  - Model path now uses `../models/spatiotemporal/`
- ✅ Updated documentation files
- ✅ Updated test scripts

## 📋 Next Steps for Hugging Face Spaces Deployment

### Step 1: Upload Models to Hugging Face Hub

Create a **Model repository** and upload all models:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Create model repository (if not exists)
# Go to https://huggingface.co/new and create a Model repository
# Example: your-username/spad-models

# Upload models (from huggingface/ directory)
cd /Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface
huggingface-cli upload your-username/spad-models models/ --repo-type model
```

### Step 2: Upload Datasets to Hugging Face Hub

Create a **Dataset repository** and upload all datasets:

```bash
# Create dataset repository
# Go to https://huggingface.co/new and create a Dataset repository
# Example: your-username/spad-datasets

# Upload datasets (from huggingface/ directory)
cd /Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface
huggingface-cli upload your-username/spad-datasets datasets/ --repo-type dataset
```

### Step 3: Update app.py for Auto-Download

Add this function to `app.py` (before `if __name__ == "__main__"`):

```python
def setup_huggingface_resources():
    """Download models and datasets from Hugging Face Hub if not present"""
    from huggingface_hub import snapshot_download
    import os
    
    # Models and datasets are siblings to local_spad_for_vision
    models_dir = os.path.join(BASE_DIR, "..", "models")
    datasets_dir = os.path.join(BASE_DIR, "..", "datasets")
    
    # Convert to absolute paths
    models_dir = os.path.abspath(models_dir)
    datasets_dir = os.path.abspath(datasets_dir)
    
    # Download models if not present
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        print("Downloading models from Hugging Face Hub...")
        try:
            snapshot_download(
                repo_id="your-username/spad-models",
                local_dir=models_dir,
                repo_type="model",
                token=os.environ.get("HF_TOKEN")
            )
            print("✓ Models downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading models: {e}")
    
    # Download datasets if not present
    if not os.path.exists(datasets_dir) or not os.listdir(datasets_dir):
        print("Downloading datasets from Hugging Face Hub...")
        try:
            snapshot_download(
                repo_id="your-username/spad-datasets",
                local_dir=datasets_dir,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN")
            )
            print("✓ Datasets downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading datasets: {e}")

# Call before app runs
if __name__ == "__main__":
    setup_huggingface_resources()
    # ... rest of app startup code
```

### Step 4: Create Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - **Space name**: `spad-for-vision` (or your preferred name)
   - **SDK**: `Docker` (for Flask app)
   - **Hardware**: Choose based on model sizes (CPU, GPU, etc.)
   - **Visibility**: Public or Private

### Step 5: Create Space Files

Create these files in your Space repository:

#### `Dockerfile`
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface_hub

# Copy application
COPY . .

# Create directories for models and datasets (will be downloaded)
RUN mkdir -p /app/../models /app/../datasets || true

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Run app (models/datasets will download on first run)
CMD ["python", "app.py"]
```

#### `README.md` (for Space)
```markdown
---
title: SPAD for Vision
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# SPAD for Vision

Spatiotemporal detection and material classification system.

Models and datasets are downloaded automatically from:
- Models: https://huggingface.co/your-username/spad-models
- Datasets: https://huggingface.co/your-username/spad-datasets
```

#### `.gitignore` (for Space)
```
../models/
../datasets/
__pycache__/
*.pyc
*.db
temp/
uploads/
*.log
```

### Step 6: Configure Space Secrets

In Space Settings → Secrets, add:
- `HF_TOKEN`: Your Hugging Face access token (if using private repos)

### Step 7: Push to Space

```bash
# From local_spad_for_vision directory
cd /Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/local_spad_for_vision

# Add Space as remote
git remote add space https://huggingface.co/spaces/your-username/spad-for-vision

# Push code (without models/datasets - they'll download automatically)
git push space main
```

## 📁 Final Structure

### Local Structure:
```
huggingface/
├── models/                    # Model repository (upload to HF Model repo)
│   ├── material_detection_head/
│   ├── yolov8/
│   └── ...
├── datasets/                  # Dataset repository (upload to HF Dataset repo)
│   ├── testmages_spatiotemporal/
│   └── ...
└── local_spad_for_vision/    # Space repository (upload to HF Space)
    ├── app.py
    ├── templates/
    ├── static/
    └── ...
```

### Hugging Face Structure:
```
your-username/
├── spad-models (Model repo)      # Contains all models/
├── spad-datasets (Dataset repo)  # Contains all datasets/
└── spad-for-vision (Space repo)  # Contains local_spad_for_vision/
```

## 🔍 Verification

After deployment, check:
1. ✅ Space builds successfully
2. ✅ Models download on first run to `../models/`
3. ✅ Datasets download on first run to `../datasets/`
4. ✅ All API endpoints work
5. ✅ File uploads work
6. ✅ Detection results display correctly

## 📝 Notes

- Models and datasets are downloaded to sibling directories on Space
- Use `os.path.abspath()` to resolve relative paths correctly
- Space has 50GB storage limit
- Consider using model cards for documentation
- Models and datasets are cached, so subsequent builds are faster

## Path Resolution

In the Space environment:
- `BASE_DIR` = `/app` (local_spad_for_vision directory)
- `../models` = `/models` (sibling directory)
- `../datasets` = `/datasets` (sibling directory)

The download function will create these directories and download from Hugging Face Hub.

---

*See `HUGGINGFACE_DEPLOYMENT.md` for detailed deployment guide.*
