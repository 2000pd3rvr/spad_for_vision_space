# Hugging Face Spaces Deployment Guide

## Overview

This guide explains how to deploy the SPAD for Vision application to Hugging Face Spaces, including how to handle models and datasets.

## Current Structure

After reorganization, the project structure is:
```
huggingface/
├── models/              # All model weights (sibling to local_spad_for_vision)
├── datasets/            # All test datasets (sibling to local_spad_for_vision)
└── local_spad_for_vision/  # Application code
    ├── app.py           # Main Flask application
    ├── requirements.txt  # Python dependencies
    ├── templates/        # HTML templates
    ├── static/           # CSS, JS, images
    └── ...              # Other project files
```

**Note:** Models and datasets are siblings to `local_spad_for_vision/` because they will be in separate Hugging Face repositories.

## Deployment Strategy

### Option 1: Recommended - Use Hugging Face Hub for Models and Datasets

**Best for:** Large models and datasets that shouldn't be in the git repository

#### Steps:

1. **Upload Models to Hugging Face Hub:**
   ```bash
   # Install huggingface_hub
   pip install huggingface_hub
   
   # Login to Hugging Face
   huggingface-cli login
   
   # Upload models (from huggingface/ directory)
   cd /path/to/huggingface
   huggingface-cli upload your-username/spad-models models/ --repo-type model
   # This uploads the entire models/ directory
   ```

2. **Upload Datasets to Hugging Face Hub:**
   ```bash
   # Upload datasets (from huggingface/ directory)
   cd /path/to/huggingface
   huggingface-cli upload your-username/spad-datasets datasets/ --repo-type dataset
   # This uploads the entire datasets/ directory
   ```

3. **Update Code to Download from Hub:**
   ```python
   from huggingface_hub import hf_hub_download, snapshot_download
   import os
   
   # Models and datasets are siblings to local_spad_for_vision
   models_dir = os.path.join(BASE_DIR, "..", "models")
   datasets_dir = os.path.join(BASE_DIR, "..", "datasets")
   
   # Convert to absolute paths
   models_dir = os.path.abspath(models_dir)
   datasets_dir = os.path.abspath(datasets_dir)
   
   # Download models on first run
   if not os.path.exists(models_dir) or not os.listdir(models_dir):
       snapshot_download(
           repo_id="your-username/spad-models",
           local_dir=models_dir,
           repo_type="model"
       )
   
   # Download datasets on first run
   if not os.path.exists(datasets_dir) or not os.listdir(datasets_dir):
       snapshot_download(
           repo_id="your-username/spad-datasets",
           local_dir=datasets_dir,
           repo_type="dataset"
       )
   ```

4. **Create Hugging Face Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Gradio" or "Streamlit" (or use Docker for Flask)
   - Name: `your-username/spad-for-vision`
   - SDK: Docker (for Flask app)

5. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       libgl1-mesa-glx \
       libglib2.0-0 \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Install Hugging Face Hub
   RUN pip install huggingface_hub
   
   # Copy application code
   COPY . .
   
   # Create directories for models and datasets (will be downloaded as siblings)
   RUN mkdir -p /models /datasets
   
   # Expose port
   EXPOSE 7860
   
   # Run Flask app
   CMD ["python", "app.py"]
   ```

6. **Create README.md for Space:**
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
   ```

### Option 2: Git LFS for Models (Alternative)

**Best for:** Smaller models that can be versioned with code

1. **Install Git LFS:**
   ```bash
   git lfs install
   ```

2. **Track model files:**
   ```bash
   git lfs track "huggingface/models/**/*.pth"
   git lfs track "huggingface/models/**/*.pt"
   git lfs track "huggingface/datasets/**/*.sto"
   ```

3. **Add .gitattributes:**
   ```
   huggingface/models/**/*.pth filter=lfs diff=lfs merge=lfs -text
   huggingface/models/**/*.pt filter=lfs diff=lfs merge=lfs -text
   huggingface/datasets/**/*.sto filter=lfs diff=lfs merge=lfs -text
   ```

4. **Commit and push:**
   ```bash
   git add .gitattributes
   git add huggingface/
   git commit -m "Add models and datasets with Git LFS"
   git push
   ```

### Option 3: Separate Model Repository (Recommended for Large Models)

**Best for:** Very large models (>1GB each)

1. **Create separate model repository:**
   - Repository: `your-username/spad-models`
   - Type: Model repository
   - Structure:
     ```
     spad-models/
     ├── material_detection_head/
     ├── yolov8/
     ├── dinov3/
     └── ...
     ```

2. **Create separate dataset repository:**
   - Repository: `your-username/spad-datasets`
   - Type: Dataset repository
   - Structure:
     ```
     spad-datasets/
     ├── testmages_spatiotemporal/
     ├── testmages__yolov8/
     └── ...
     ```

3. **Update app.py to download on startup:**
   ```python
   import os
   from huggingface_hub import snapshot_download
   
   def download_models_and_datasets():
       """Download models and datasets from Hugging Face Hub on first run"""
       models_dir = os.path.join(BASE_DIR, "huggingface", "models")
       datasets_dir = os.path.join(BASE_DIR, "huggingface", "datasets")
       
       if not os.path.exists(models_dir) or not os.listdir(models_dir):
           print("Downloading models from Hugging Face Hub...")
           snapshot_download(
               repo_id="your-username/spad-models",
               local_dir=models_dir,
               repo_type="model"
           )
       
       if not os.path.exists(datasets_dir) or not os.listdir(datasets_dir):
           print("Downloading datasets from Hugging Face Hub...")
           snapshot_download(
               repo_id="your-username/spad-datasets",
               local_dir=datasets_dir,
               repo_type="dataset"
           )
   
   # Call on app startup
   if __name__ == "__main__":
       download_models_and_datasets()
       app.run(host="0.0.0.0", port=7860)
   ```

## Recommended Approach

**For Hugging Face Spaces deployment, I recommend Option 1 or Option 3:**

1. **Upload models to a separate Hugging Face Model repository**
2. **Upload datasets to a separate Hugging Face Dataset repository**
3. **Use `snapshot_download` in your Space to download them on startup**
4. **Keep the Space repository lightweight (code only)**

## Space Configuration

### app.py modifications for Spaces:

```python
import os
from huggingface_hub import snapshot_download

# At the top of app.py, after BASE_DIR definition
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
                token=os.environ.get("HF_TOKEN")  # Use token from Space secrets
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

# Call setup before app runs
if __name__ == "__main__":
    setup_huggingface_resources()
    app.run(host="0.0.0.0", port=7860)
```

## Space Secrets

In your Hugging Face Space settings, add:
- `HF_TOKEN`: Your Hugging Face access token (for private repos)

## File Structure for Space

```
your-space/ (local_spad_for_vision/)
├── app.py                    # Main application
├── requirements.txt          # Dependencies
├── Dockerfile               # Docker configuration (if using Docker)
├── README.md                # Space description
├── .gitignore               # Ignore downloaded models/datasets
├── templates/               # HTML templates
└── static/                  # CSS, JS, images

# At runtime (sibling directories):
../models/                   # Downloaded from Hub (sibling to Space)
../datasets/                 # Downloaded from Hub (sibling to Space)
```

## .gitignore for Space

```
../models/
../datasets/
__pycache__/
*.pyc
*.db
temp/
uploads/
```

## Testing Locally

Before deploying, test the download mechanism:

```python
# test_download.py
from huggingface_hub import snapshot_download
import os

# Test model download
snapshot_download(
    repo_id="your-username/spad-models",
    local_dir="huggingface/models",
    repo_type="model"
)

# Test dataset download
snapshot_download(
    repo_id="your-username/spad-datasets",
    local_dir="huggingface/datasets",
    repo_type="dataset"
)

print("Download complete!")
```

## Deployment Checklist

- [ ] Upload all models to Hugging Face Model repository
- [ ] Upload all datasets to Hugging Face Dataset repository
- [ ] Update app.py with download logic
- [ ] Create Dockerfile (if using Docker SDK)
- [ ] Create README.md for Space
- [ ] Update .gitignore to exclude downloaded files
- [ ] Test download mechanism locally
- [ ] Create Space on Hugging Face
- [ ] Configure Space secrets (HF_TOKEN if needed)
- [ ] Push code to Space repository
- [ ] Verify models and datasets download on Space startup
- [ ] Test all endpoints work correctly

## Notes

- Hugging Face Spaces have a 50GB storage limit
- Models are cached, so subsequent builds are faster
- Use `repo_type="model"` for model repositories
- Use `repo_type="dataset"` for dataset repositories
- Consider using model cards and dataset cards for documentation

---

*Last Updated: 2025-01-31*

