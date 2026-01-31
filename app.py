from flask import Flask, render_template, request, jsonify, send_file
import os
import sqlite3
from datetime import datetime
import requests
import secrets
import sys
import subprocess

# Add material detection path (import functions lazily to avoid blocking startup)
sys.path.append('apps.err/material_detection_naturalobjects')
# Import functions only when needed, not at module level

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# Base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models and datasets are siblings to spad_for_vision_space (same level)
# Locally: ../models and ../datasets relative to BASE_DIR
# On Hugging Face Spaces: models/ and datasets/ will be downloaded to BASE_DIR/models and BASE_DIR/datasets
def get_models_dir():
    """Get models directory path - sibling to BASE_DIR locally, or BASE_DIR/models on Spaces"""
    if os.environ.get("SPACE_ID"):
        # On Hugging Face Spaces, download to BASE_DIR/models
        return os.path.join(BASE_DIR, "models")
    else:
        # Local: models is sibling to spad_for_vision_space
        return os.path.join(os.path.dirname(BASE_DIR), "models")

def get_datasets_dir():
    """Get datasets directory path - sibling to BASE_DIR locally, or BASE_DIR/datasets on Spaces"""
    if os.environ.get("SPACE_ID"):
        # On Hugging Face Spaces, download to BASE_DIR/datasets
        return os.path.join(BASE_DIR, "datasets")
    else:
        # Local: datasets is sibling to spad_for_vision_space
        return os.path.join(os.path.dirname(BASE_DIR), "datasets")

DB_PATH = "visitors.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT,
            country TEXT,
            city TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Initialize database on startup (with error handling)
try:
    init_db()
except Exception as e:
    print(f"Warning: Database initialization failed (non-critical): {e}")
    # Continue anyway - database will be created on first use

def get_visitor_location(ip_address: str):
    try:
        resp = requests.get(f"http://ipapi.co/{ip_address}/json/", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "country": data.get("country_name", "Unknown"),
                "city": data.get("city", "Unknown"),
                "region": data.get("region", "Unknown"),
            }
    except Exception:
        pass
    return {"country": "Unknown", "city": "Unknown", "region": "Unknown"}

def log_visitor(ip_address: str):
    location = get_visitor_location(ip_address)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO visitors (ip_address, country, city)
        VALUES (?, ?, ?)
        """,
        (ip_address, location["country"], location["city"]),
    )
    conn.commit()
    conn.close()

def get_visitor_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM visitors")
    total_visitors = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT ip_address) FROM visitors")
    unique_visitors = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT country, COUNT(*) AS count
        FROM visitors
        GROUP BY country
        ORDER BY count DESC
        LIMIT 10
        """
    )
    countries = cursor.fetchall()
    conn.close()

    return {
        "total_visitors": total_visitors,
        "unique_visitors": unique_visitors,
        "countries": countries,
    }


@app.route("/favicon.ico")
def favicon():
    favicon_path = os.path.join(app.static_folder, "favicon.ico")
    if os.path.exists(favicon_path):
        return send_file(favicon_path, mimetype="image/vnd.microsoft.icon")
    return "", 404

@app.route("/")
def home():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("index.html", stats=stats)

@app.route("/custom_yolov8_demo")
def custom_yolov8_demo():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("custom_yolov8_demo.html", stats=stats)

@app.route("/spatiotemporal_detection")
def spatiotemporal_detection():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("spatiotemporal_detection.html", stats=stats)

@app.route("/flat_surface_detection")
def flat_surface_detection():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("flat_surface_detection.html", stats=stats)

@app.route("/material_purity")
def material_purity():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("material_purity.html", stats=stats)

@app.route("/material_detection_head")
def material_detection_head():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("material_detection_head.html", stats=stats)

@app.route("/fluid_purity_demo")
def fluid_purity_demo():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("fluid_purity_demo.html", stats=stats)

@app.route("/demos")
def demos():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("demos.html", stats=stats)

@app.route("/dinov3_demo")
def dinov3_demo():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("dinov3_demo.html", stats=stats)

@app.route("/detect_yolov3")
def detect_yolov3():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("yolov3_demo.html", stats=stats)

@app.route("/deliverables")
def deliverables():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("deliverables.html", stats=stats, deliverables=[])

@app.route("/deliverables/create", methods=["GET", "POST"])
def create_deliverable():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    if request.method == "POST":
        return jsonify({"success": True, "message": "Deliverable creation not yet implemented"})
    return render_template("create_deliverable.html", stats=stats, projects=[])

@app.route("/deliverables/<int:deliverable_id>")
def deliverable_detail(deliverable_id):
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("deliverable_detail.html", stats=stats, deliverable={"id": deliverable_id})

@app.route("/deliverables/<int:deliverable_id>/edit", methods=["GET", "POST"])
def edit_deliverable(deliverable_id):
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    if request.method == "POST":
        return jsonify({"success": True, "message": "Deliverable editing not yet implemented"})
    return render_template("edit_deliverable.html", stats=stats, deliverable={"id": deliverable_id}, projects=[])

@app.route("/deliverables/<int:deliverable_id>/delete", methods=["POST"])
def delete_deliverable(deliverable_id):
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return jsonify({"success": True, "message": "Deliverable deletion not yet implemented"})

@app.route("/deliverables/<int:deliverable_id>/toggle-status", methods=["POST"])
def toggle_deliverable_status(deliverable_id):
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return jsonify({"success": True, "message": "Status toggle not yet implemented"})

@app.route("/deliverables/in-progress")
def deliverables_in_progress():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    # Return a simple page - can be enhanced later
    return render_template("in_progress_deliverables.html", stats=stats, deliverables=[])

@app.route("/deliverables/completed")
def deliverables_completed():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    # Return a simple page - can be enhanced later
    return render_template("completed_deliverables.html", stats=stats, deliverables=[])

@app.route("/version_control")
def version_control():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    # Create a simple version control page
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Version Control - MV+</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 40px; text-align: center; }
            h1 { color: #B91D30; }
        </style>
    </head>
    <body>
        <h1>Version Control</h1>
        <p>Version control features coming soon.</p>
        <p><a href="/">Return to Home</a></p>
    </body>
    </html>
    """

# Projects route removed - only Demos is accessible via dropdown menu

@app.route("/projects/create", methods=["GET", "POST"])
def create_project():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    if request.method == "POST":
        # Handle project creation (placeholder)
        return jsonify({"success": True, "message": "Project creation not yet implemented"})
    # For GET, return a simple message page
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Create Project - MV+</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 40px; text-align: center; }}
            h1 {{ color: #B91D30; }}
            a {{ color: #00CED1; text-decoration: none; }}
        </style>
    </head>
    <body>
        <h1>Create Project</h1>
        <p>Project creation feature coming soon.</p>
        <p><a href="/projects">← Back to Projects</a></p>
    </body>
    </html>
    """

@app.route("/projects/<int:project_id>")
def project_detail(project_id):
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Project Details - MV+</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 40px; text-align: center; }}
            h1 {{ color: #B91D30; }}
            a {{ color: #00CED1; text-decoration: none; }}
        </style>
    </head>
    <body>
        <h1>Project Details</h1>
        <p>Project ID: {project_id}</p>
        <p>Project details feature coming soon.</p>
        <p><a href="/projects">← Back to Projects</a></p>
    </body>
    </html>
    """

@app.route("/projects/<int:project_id>/edit", methods=["GET", "POST"])
def edit_project(project_id):
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    if request.method == "POST":
        # Handle project edit (placeholder)
        return jsonify({"success": True, "message": "Project editing not yet implemented"})
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Edit Project - MV+</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 40px; text-align: center; }}
            h1 {{ color: #B91D30; }}
            a {{ color: #00CED1; text-decoration: none; }}
        </style>
    </head>
    <body>
        <h1>Edit Project</h1>
        <p>Project ID: {project_id}</p>
        <p>Project editing feature coming soon.</p>
        <p><a href="/projects">← Back to Projects</a></p>
    </body>
    </html>
    """

@app.route("/about")
def about():
    visitor_ip = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    log_visitor(visitor_ip)
    stats = get_visitor_stats()
    return render_template("about.html", stats=stats)

@app.route("/api/detect_spatiotemporal", methods=["POST"])
def api_detect_spatiotemporal():
    """
    API endpoint for spatiotemporal detection using STO files.
    Wraps process_sto_file from material_detection_naturalobjects/spatiotemporal_detection_script.py
    """
    try:
        # Import the new spatiotemporal detection script
        import sys
        sys.path.append('apps.err/material_detection_naturalobjects')
        from spatiotemporal_detection_script import process_sto_file as process_sto_file_spatiotemporal
        
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        filename_lower = file.filename.lower()
        if not filename_lower.endswith(".sto"):
            return jsonify({
                "error": "Please upload a STO spatiotemporal file.",
                "error_type": "wrong_format",
            }), 400

        temp_path = f"temp_spatiotemporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sto"
        file.save(temp_path)

        try:
            result = process_sto_file_spatiotemporal(temp_path)
            if result.get("success"):
                return jsonify({
                    "success": True,
                    "image": result.get("image"),
                    "image_size": result.get("image_size"),
                    "image_mode": result.get("image_mode"),
                    "sto_length": result.get("sto_length"),
                    "filename": file.filename,
                })
            else:
                return jsonify({
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                }), 400
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Spatiotemporal API Error: {e}")
        print(error_details)
        return jsonify({
            "error": str(e),
            "error_type": "server_error",
            "details": error_details,
        }), 500

@app.route("/api/stats")
def api_stats():
    stats = get_visitor_stats()
    return jsonify(stats)

@app.route("/api/flat_surface_detection_weights", methods=["GET"])
def api_flat_surface_detection_weights():
    """API endpoint to get flat surface detection model weights from local directory or Hugging Face Hub"""
    import os
    import glob
    import re
    
    try:
        weights = []
        weights_dir = os.path.join(get_models_dir(), "flat_surface")
        repo_id = "mvplus/spatiotemporal_models"
        subfolder = "flat_surface"
        
        # First, try to get weights from Hugging Face Hub
        hub_files = fetch_weights_from_hub(repo_id, subfolder)
        for hub_file in hub_files:
            filename = os.path.basename(hub_file)
            # Extract accuracy and epoch from filename
            pattern_match = re.search(r'epoch_(\d+)_Accuracy_([\d\.]+)', filename)
            if pattern_match:
                epoch = int(pattern_match.group(1))
                try:
                    acc_str = pattern_match.group(2).rstrip('.')
                    accuracy = float(acc_str)
                    display_name = f"Epoch {epoch} ({accuracy:.2f}% accuracy)"
                    weight_type = "Checkpoint"
                except ValueError:
                    epoch = 0
                    accuracy = 0.0
                    display_name = filename.replace('.pth', '')
                    weight_type = "Checkpoint"
            else:
                epoch = 0
                accuracy = 0.0
                display_name = filename.replace('.pth', '')
                weight_type = "Checkpoint"
            
            # Use Hub path format for on-demand download
            weights.append({
                "filename": filename,
                "path": f"hub://{repo_id}/{hub_file}",  # Hub path for on-demand download
                "display_name": display_name,
                "accuracy": accuracy,
                "epoch": epoch,
                "weight_type": weight_type,
                "source": "hub"
            })
        
        # Also check local files
        if os.path.exists(weights_dir):
            weight_files = glob.glob(os.path.join(weights_dir, "*.pth"))
            
            for weight_file in weight_files:
                filename = os.path.basename(weight_file)
                
                # Extract accuracy and epoch from filename
                # Pattern: epoch_133_Accuracy_98.81517028808594__2025-11-04 23:05:50.452709.pth
                # Or: epoch_150_Accuracy_99.605.pth
                pattern_match = re.search(r'epoch_(\d+)_Accuracy_([\d\.]+)', filename)
                if pattern_match:
                    epoch = int(pattern_match.group(1))
                    try:
                        # Remove trailing dots and convert to float
                        acc_str = pattern_match.group(2).rstrip('.')
                        accuracy = float(acc_str)
                        display_name = f"Epoch {epoch} ({accuracy:.2f}% accuracy)"
                        weight_type = "Checkpoint"
                    except ValueError:
                        # If conversion fails, use defaults
                        epoch = 0
                        accuracy = 0.0
                        display_name = filename.replace('.pth', '')
                        weight_type = "Checkpoint"
                elif filename == "1.pth":
                    # Special case for 1.pth
                    epoch = 0
                    accuracy = 0.0
                    display_name = "Model Weight (1.pth)"
                    weight_type = "Default"
                else:
                    epoch = 0
                    accuracy = 0.0
                    display_name = filename.replace('.pth', '')
                    weight_type = "Checkpoint"
                
                weights.append({
                    "filename": filename,
                    "path": weight_file,  # Use local path
                    "display_name": display_name,
                    "accuracy": accuracy,
                    "epoch": epoch,
                    "weight_type": weight_type
                })
            
            # Sort by accuracy (highest first), then by epoch
            weights.sort(key=lambda x: (x['accuracy'], x['epoch']), reverse=True)
        else:
            print(f"DEBUG: Weights directory not found: {weights_dir}")
        
        return jsonify({
            'success': True,
            'weights': weights
        })
    except Exception as e:
        import traceback
        print(f"Error loading flat surface weights: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'weights': []
        }), 500

@app.route("/api/fluid_purity_weights", methods=["GET"])
def api_fluid_purity_weights():
    """API endpoint to get fluid purity model weights from local directory or Hugging Face Hub"""
    import os
    import glob
    import re
    
    try:
        weights = []
        weights_dir = os.path.join(get_models_dir(), "material_purity")
        repo_id = "mvplus/spatiotemporal_models"
        subfolder = "material_purity"
        
        # First, try to get weights from Hugging Face Hub
        hub_files = fetch_weights_from_hub(repo_id, subfolder)
        for hub_file in hub_files:
            filename = os.path.basename(hub_file)
            # Extract accuracy and epoch from filename
            pattern_match = re.search(r'epoch_(\d+)_Accuracy_([\d\.]+)', filename)
            if pattern_match:
                epoch = int(pattern_match.group(1))
                try:
                    acc_str = pattern_match.group(2).rstrip('.')
                    accuracy = float(acc_str)
                    display_name = f"Epoch {epoch} ({accuracy:.2f}% accuracy)"
                    weight_type = "Checkpoint"
                except ValueError:
                    epoch = 0
                    accuracy = 0.0
                    display_name = filename.replace('.pth', '')
                    weight_type = "Checkpoint"
            else:
                epoch = 0
                accuracy = 0.0
                display_name = filename.replace('.pth', '')
                weight_type = "Checkpoint"
            
            weights.append({
                "filename": filename,
                "path": f"hub://{repo_id}/{hub_file}",
                "display_name": display_name,
                "accuracy": accuracy,
                "epoch": epoch,
                "weight_type": weight_type,
                "source": "hub"
            })
        
        # Also check local files
        if os.path.exists(weights_dir):
            weight_files = glob.glob(os.path.join(weights_dir, "*.pth"))
            
            for weight_file in weight_files:
                filename = os.path.basename(weight_file)
                
                # Extract accuracy and epoch from filename
                # Pattern: epoch_105_Accuracy_100.0__2025-11-06 09:02:10.335104.pth
                # Or: epoch_194_Accuracy_100.pth
                pattern_match = re.search(r'epoch_(\d+)_Accuracy_([\d\.]+)', filename)
                if pattern_match:
                    epoch = int(pattern_match.group(1))
                    try:
                        # Remove trailing dots and convert to float
                        acc_str = pattern_match.group(2).rstrip('.')
                        accuracy = float(acc_str)
                        display_name = f"Epoch {epoch} ({accuracy:.2f}% accuracy)"
                        weight_type = "Checkpoint"
                    except ValueError:
                        # If conversion fails, use defaults
                        epoch = 0
                        accuracy = 0.0
                        display_name = filename.replace('.pth', '')
                        weight_type = "Checkpoint"
                else:
                    epoch = 0
                    accuracy = 0.0
                    display_name = filename.replace('.pth', '')
                    weight_type = "Checkpoint"
                
                # Check if this weight is already in the list from Hub
                existing = next((w for w in weights if w['filename'] == filename), None)
                if not existing:
                    weights.append({
                        "filename": filename,
                        "path": weight_file,  # Use local path
                        "display_name": display_name,
                        "accuracy": accuracy,
                        "epoch": epoch,
                        "weight_type": weight_type,
                        "source": "local"
                    })
            
            # Sort by accuracy (highest first), then by epoch
            weights.sort(key=lambda x: (x['accuracy'], x['epoch']), reverse=True)
        else:
            print(f"DEBUG: Weights directory not found: {weights_dir}")
        
        return jsonify({
            'success': True,
            'weights': weights
        })
    except Exception as e:
        import traceback
        print(f"Error loading fluid purity weights: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'weights': []
        }), 500

@app.route("/api/material_detection_head_weights", methods=["GET"])
def api_material_detection_head_weights():
    """API endpoint to get material detection head model weights from local directory or Hugging Face Hub"""
    import os
    import glob
    import re
    
    try:
        weights = []
        weights_dir = os.path.join(get_models_dir(), "material_detection_head")
        repo_id = "mvplus/spatiotemporal_models"
        subfolder = "material_detection_head"
        
        # First, try to get weights from Hugging Face Hub
        hub_files = fetch_weights_from_hub(repo_id, subfolder)
        for hub_file in hub_files:
            filename = os.path.basename(hub_file)
            # Extract accuracy and epoch from filename
            pattern_match = re.search(r'epoch_(\d+)_Accuracy_([\d\.]+)', filename)
            if pattern_match:
                epoch = int(pattern_match.group(1))
                try:
                    acc_str = pattern_match.group(2).rstrip('.')
                    accuracy = float(acc_str)
                    display_name = f"Epoch {epoch} ({accuracy:.2f}% accuracy)"
                    weight_type = "Checkpoint"
                except ValueError:
                    epoch = 0
                    accuracy = 0.0
                    display_name = filename.replace('.pth', '')
                    weight_type = "Checkpoint"
            else:
                epoch = 0
                accuracy = 0.0
                display_name = filename.replace('.pth', '')
                weight_type = "Checkpoint"
            
            weights.append({
                "filename": filename,
                "path": f"hub://{repo_id}/{hub_file}",
                "display_name": display_name,
                "accuracy": accuracy,
                "epoch": epoch,
                "weight_type": weight_type,
                "source": "hub"
            })
        
        # Also check local files
        if os.path.exists(weights_dir):
            weight_files = glob.glob(os.path.join(weights_dir, "*.pth"))
            
            for weight_file in weight_files:
                filename = os.path.basename(weight_file)
                
                # Extract accuracy and epoch from filename
                # Pattern: epoch_186_Accuracy_80.pth
                # Or: epoch_399_Accuracy_98.25.pth
                pattern_match = re.search(r'epoch_(\d+)_Accuracy_([\d\.]+)', filename)
                if pattern_match:
                    epoch = int(pattern_match.group(1))
                    try:
                        # Remove trailing dots and convert to float
                        acc_str = pattern_match.group(2).rstrip('.')
                        accuracy = float(acc_str)
                        display_name = f"Epoch {epoch} ({accuracy:.2f}% accuracy)"
                        weight_type = "Checkpoint"
                    except ValueError:
                        # If conversion fails, use defaults
                        epoch = 0
                        accuracy = 0.0
                        display_name = filename.replace('.pth', '')
                        weight_type = "Checkpoint"
                else:
                    epoch = 0
                    accuracy = 0.0
                    display_name = filename.replace('.pth', '')
                    weight_type = "Checkpoint"
                
                # Check if this weight is already in the list from Hub
                existing = next((w for w in weights if w['filename'] == filename), None)
                if not existing:
                    weights.append({
                        "filename": filename,
                        "path": weight_file,  # Use local path
                        "display_name": display_name,
                        "accuracy": accuracy,
                        "epoch": epoch,
                        "weight_type": weight_type,
                        "source": "local"
                    })
            
            # Sort by accuracy (highest first), then by epoch
            weights.sort(key=lambda x: (x['accuracy'], x['epoch']), reverse=True)
        else:
            print(f"DEBUG: Weights directory not found: {weights_dir}")
        
        return jsonify({
            'success': True,
            'weights': weights
        })
    except Exception as e:
        import traceback
        print(f"Error loading material detection head weights: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'weights': []
        }), 500

@app.route("/api/get_model_info", methods=["POST"])
def api_get_model_info():
    """API endpoint to get model architecture and classes based on weight path"""
    weight_path = request.form.get('weight_path', '')
    
    if not weight_path:
        return jsonify({'error': 'No weight path provided'}), 400
    
    # Determine model type and class names based on weight path
    if 'material_purity' in weight_path or 'fluid_purity' in weight_path or 'purity' in weight_path.lower():
        class_names = ['Impure', 'Pure']
        num_classes = 2
        model_architecture = 'Material Purity Classifier (Binary)'
    elif 'flat_surface' in weight_path.lower() or 'flatsurface' in weight_path.lower():
        class_names = ['BCB', 'BNT', 'WGF', 'WNT']
        num_classes = 4
        model_architecture = 'Flat Surface Detection ConvNet'
    else:
        # Default fallback
        class_names = []
        num_classes = 0
        model_architecture = 'Unknown'
    
    classes_str = ', '.join(str(c) for c in class_names) if class_names else '-'
    
    return jsonify({
        'success': True,
        'architecture': model_architecture,
        'classes': class_names if isinstance(class_names, list) else list(class_names),
        'classes_display': classes_str,
        'num_classes': num_classes
    })

@app.route("/api/detect_material_head", methods=["POST"])
def api_detect_material_head():
    """API endpoint for material detection head - handles both flat_surface and material_purity"""
    import time
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    import base64
    import io
    import pickle
    from io import BytesIO
    from huggingface_hub import hf_hub_download
    
    # Set deterministic behavior
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    
    start_time = time.time()
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get selected weight path (can be local path or Hugging Face Hub path)
        weight_path = request.form.get('weight_path', '')
        if not weight_path:
            return jsonify({'error': 'No model weight selected'}), 400
        
        print(f"DEBUG: Received weight_path: '{weight_path}'")
        print(f"DEBUG: weight_path is absolute: {os.path.isabs(weight_path) if weight_path else False}")
        print(f"DEBUG: weight_path exists: {os.path.exists(weight_path) if weight_path else False}")
        
        # Determine model type from weight path
        is_material_purity = ('material_purity' in weight_path.lower() or 'fluid_purity' in weight_path.lower() or 'purity' in weight_path.lower())
        is_flat_surface = ('flat_surface' in weight_path.lower() or 'flatsurface' in weight_path.lower())
        is_material_detection_head = ('material_detection_head' in weight_path.lower() or 'naturalobjects' in weight_path.lower())
        
        print(f"DEBUG: Model type detection:")
        print(f"DEBUG:   is_material_purity: {is_material_purity}")
        print(f"DEBUG:   is_flat_surface: {is_flat_surface}")
        print(f"DEBUG:   is_material_detection_head: {is_material_detection_head}")
        print(f"DEBUG:   weight_path: {weight_path}")
        
        # Check if weight_path is a local file path or Hugging Face Hub path
        local_model_path = None
        
        # First, check if it's an absolute local path
        if os.path.isabs(weight_path) and os.path.exists(weight_path):
            # Absolute local file path
            print(f"DEBUG: Using absolute local weight file: {weight_path}")
            local_model_path = weight_path
        # Check if it exists as-is (relative path)
        elif os.path.exists(weight_path):
            print(f"DEBUG: Using local weight file (found as-is): {weight_path}")
            local_model_path = weight_path
        # Try relative to BASE_DIR
        else:
            relative_path = os.path.join(BASE_DIR, weight_path)
            if os.path.exists(relative_path):
                print(f"DEBUG: Using relative local weight file: {relative_path}")
                local_model_path = relative_path
        
        # If not found as local path, check if it's a Hugging Face Hub path
        if local_model_path is None:
            # Check if it's a Hub path (hub://repo_id/path/to/file)
            if weight_path.startswith('hub://'):
                downloaded_path = download_model_from_hub(weight_path)
                if downloaded_path:
                    local_model_path = downloaded_path
                else:
                    return jsonify({'error': f'Failed to download model from Hub: {weight_path}'}), 400
            else:
                return jsonify({'error': f'Weight path not found: {weight_path}. Please check the path is correct.'}), 400
        
        # Verify the local model file exists
        if not local_model_path or not os.path.exists(local_model_path):
            return jsonify({'error': f'Model file not found: {local_model_path or weight_path}'}), 400
        
        # Check if file is STO or regular image
        filename_lower = file.filename.lower()
        is_sto_file = filename_lower.endswith('.sto')
        
        # Load image
        print(f"DEBUG: File type check - is_sto_file: {is_sto_file}, filename: {file.filename}")
        if is_sto_file:
            # Handle STO file - extract index 1 (16x16 material detection image)
            # STO structure: Index 0=metadata, Index 1=16x16 material image, Index 2=OD metadata, Index 3=640x640 OD image
            temp_sto_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sto"
            with open(temp_sto_path, 'wb') as temp_file:
                file.stream.seek(0)
                temp_file.write(file.stream.read())
            
            try:
                with open(temp_sto_path, 'rb') as f:
                    sto_data = pickle.load(f)
                    if len(sto_data) < 2:
                        return jsonify({'error': 'Invalid STO file - need at least 2 items'}), 400
                    # Extract index 1 for material detection (16x16 image)
                    sto_item = sto_data[1]
                    if isinstance(sto_item, bytes):
                        image = Image.open(BytesIO(sto_item)).convert('RGB')
                        print(f"DEBUG: Extracted image from STO index 1 (bytes), size: {image.size}, mode: {image.mode}")
                    elif hasattr(sto_item, 'mode'):
                        image = sto_item.convert('RGB')
                        print(f"DEBUG: Extracted image from STO index 1 (PIL), size: {image.size}, mode: {image.mode}")
                        # Additional debug: Check if image is mostly black/empty (might indicate wrong index)
                        import numpy as np
                        img_array = np.array(image)
                        if img_array.mean() < 5 and img_array.std() < 5:
                            print(f"⚠️  WARNING: Image at index 1 appears to be mostly black (mean={img_array.mean():.2f}, std={img_array.std():.2f})")
                            print(f"⚠️  This might indicate the wrong image is being extracted from the STO file")
                    else:
                        return jsonify({'error': f'Invalid STO file structure at index 1: expected image, got {type(sto_item).__name__}'}), 400
            finally:
                if os.path.exists(temp_sto_path):
                    os.remove(temp_sto_path)
        else:
            # Regular image file - use material_detection_functions for processing
            file.stream.seek(0)
            # Use the working predict_material function from material_detection_functions
            try:
                # Lazy import to avoid blocking startup
                from material_detection_functions import process_png_bytes
                # Process the image using the working function
                processed_image, _ = process_png_bytes(file.stream.read())
                image = processed_image
                print(f"DEBUG: Processed image using material_detection_functions, size: {image.size}, mode: {image.mode}")
                
                # Debug: Check image pixel values
                import numpy as np
                img_array = np.array(image)
                print(f"DEBUG: Image array shape: {img_array.shape}")
                print(f"DEBUG: Image array min/max: {img_array.min()}/{img_array.max()}")
                print(f"DEBUG: Image array mean: {img_array.mean():.2f}, std: {img_array.std():.2f}")
                print(f"DEBUG: Image array sample (first 3x3 pixels): {img_array[:3, :3] if len(img_array.shape) == 2 else img_array[:3, :3, :]}")
            except Exception as e:
                print(f"DEBUG: Error using process_png_bytes, falling back to direct loading: {e}")
                file.stream.seek(0)
                image = Image.open(file.stream).convert('RGB')
                print(f"DEBUG: Loaded regular image directly, size: {image.size}, mode: {image.mode}")
        
        # Preprocess image based on model type
        # CRITICAL: Material detection head uses ToTensor + Normalize(0.5, 0.5, 0.5) as in eval script
        # For material_detection_head, we can optionally use preprocess_image from material_detection_functions
        # Lazy import to avoid blocking startup
        try:
            from material_detection_functions import preprocess_image, predict_material
        except ImportError as e:
            print(f"Warning: Could not import material_detection_functions: {e}")
            preprocess_image = None
            predict_material = None
        # but it uses ImageNet normalization, so we'll stick with the eval script normalization
        if is_material_purity or is_flat_surface or is_material_detection_head:
            # All use: ToTensor + Normalize(0.5, 0.5, 0.5) - EXACTLY as in eval script
            transform = transforms.Compose([
                transforms.ToTensor(),  # Converts PIL to tensor, normalizes to [0,1], converts to CHW
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Maps [0,1] to [-1,1]
            ])
            
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to 16x16 if not already (EXACTLY as in eval script)
            if image.size != (16, 16):
                image = image.resize((16, 16), Image.Resampling.LANCZOS)
            
            # Apply transform - returns [C, H, W] tensor
            print(f"DEBUG: Before transform - image size: {image.size}, mode: {image.mode}")
            
            # CRITICAL: Check if image has actual pixel variation
            import numpy as np
            img_array_before = np.array(image)
            print(f"DEBUG: Image array before transform - shape: {img_array_before.shape}, min: {img_array_before.min()}, max: {img_array_before.max()}, mean: {img_array_before.mean():.2f}, std: {img_array_before.std():.2f}")
            if img_array_before.std() < 1.0:
                print(f"⚠️  WARNING: Image has very low std ({img_array_before.std():.2f}), might be nearly uniform!")
            
            image_tensor = transform(image)
            print(f"DEBUG: After transform - tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            print(f"DEBUG: After transform - tensor range: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
            print(f"DEBUG: After transform - tensor mean: {image_tensor.mean():.4f}, std: {image_tensor.std():.4f}")
            
            # CRITICAL: Check if tensor is nearly uniform (would cause all predictions to be similar)
            if image_tensor.std() < 0.01:
                print(f"⚠️  WARNING: Tensor has very low std ({image_tensor.std():.4f}) after normalization, might cause uniform predictions!")
            
            # CRITICAL: Ensure tensor is exactly [C, H, W] before adding batch dimension
            if len(image_tensor.shape) == 3:
                # Add batch dimension: [C, H, W] -> [1, C, H, W]
                image_tensor = image_tensor.unsqueeze(0)
                print(f"DEBUG: Added batch dimension, shape: {image_tensor.shape}")
            elif len(image_tensor.shape) == 4:
                # Already has batch dimension, but verify it's [1, C, H, W]
                if image_tensor.shape[0] != 1:
                    # Remove extra batch dimensions and re-add
                    while len(image_tensor.shape) > 3:
                        image_tensor = image_tensor.squeeze(0)
                    image_tensor = image_tensor.unsqueeze(0)
                print(f"DEBUG: Already had batch dimension, shape: {image_tensor.shape}")
            else:
                raise ValueError(f"Unexpected image tensor shape after transform: {image_tensor.shape}")
            
            # Final verification: tensor must be [1, 3, 16, 16]
            if image_tensor.shape != torch.Size([1, 3, 16, 16]):
                print(f"WARNING: Image tensor shape {image_tensor.shape} != expected [1, 3, 16, 16]")
                # Try to fix it
                while len(image_tensor.shape) > 4:
                    image_tensor = image_tensor.squeeze(0)
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                if image_tensor.shape != torch.Size([1, 3, 16, 16]):
                    raise ValueError(f"Cannot fix image tensor shape: {image_tensor.shape}, expected [1, 3, 16, 16]")
            print(f"DEBUG: Final tensor shape verified: {image_tensor.shape}")
        else:
            # Default preprocessing
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((16, 16), Image.Resampling.LANCZOS)
            image_tensor = transform(image).unsqueeze(0)
        
        # Define model architectures
        class MaterialPurityClassifier(nn.Module):
            """Binary CNN classifier for material purity"""
            def __init__(self):
                super(MaterialPurityClassifier, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu1 = nn.ReLU(inplace=True)
                self.pool1 = nn.MaxPool2d(kernel_size=2)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu2 = nn.ReLU(inplace=True)
                self.pool2 = nn.MaxPool2d(kernel_size=2)
                self.fc1 = nn.Linear(32 * 4 * 4, 64)
                self.relu3 = nn.ReLU(inplace=True)
                self.fc2 = nn.Linear(64, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.pool1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu2(x)
                x = self.pool2(x)
                x = x.view(-1, 32 * 4 * 4)
                x = self.fc1(x)
                x = self.relu3(x)
                x = self.fc2(x)
                x = self.sigmoid(x)
                return x
        
        class ConvNetFlatSurface(nn.Module):
            """ConvNet for flat surface detection - 4 classes (matches eval script)"""
            def __init__(self):
                super(ConvNetFlatSurface, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
                self.act1 = nn.ReLU()
                self.drop1 = nn.Dropout(0.3)
                self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
                self.act2 = nn.ReLU()
                self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
                self.flat = nn.Flatten()
                self.fc3 = nn.Linear(2048, 512)  # 32 * 8 * 8 = 2048 after pooling
                self.act3 = nn.ReLU()
                self.drop3 = nn.Dropout(0.5)
                self.fc4 = nn.Linear(512, 4)  # 4 classes: BCB, BNT, WGF, WNT
            
            def forward(self, x):
                x = self.act1(self.conv1(x))
                x = self.drop1(x)
                x = self.act2(self.conv2(x))
                x = self.pool2(x)
                x = self.flat(x)
                x = self.act3(self.fc3(x))
                x = self.drop3(x)
                x = self.fc4(x)
                return x
        
        class ConvNetMaterialDetectionHead(nn.Module):
            """ConvNet for material detection head - 12 classes (matches eval script)"""
            def __init__(self):
                super(ConvNetMaterialDetectionHead, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
                self.act1 = nn.ReLU()
                self.drop1 = nn.Dropout(0.3)
                self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
                self.act2 = nn.ReLU()
                self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
                self.flat = nn.Flatten()
                self.fc3 = nn.Linear(2048, 512)  # 32 * 8 * 8 = 2048 after pooling from 16x16
                self.act3 = nn.ReLU()
                self.drop3 = nn.Dropout(0.5)
                self.fc4 = nn.Linear(512, 12)  # 12 classes
            
            def forward(self, x):
                # EXACTLY as in eval script - dropout is defined but NOT called in forward
                # The eval script shows: #==x=self.drop1(x) (commented out)
                # Dropout layers exist but are not used during inference
                x = self.act1(self.conv1(x))
                x = self.act2(self.conv2(x))
                x = self.pool2(x)
                x = self.flat(x)
                x = self.act3(self.fc3(x))
                x = self.fc4(x)
                return x
        
        # Load checkpoint
        checkpoint = torch.load(local_model_path, map_location='cpu')
        
        # Initialize model based on type
        if is_material_purity:
            model = MaterialPurityClassifier()
            class_names = ['Impure', 'Pure']
            model_architecture = 'Material Purity Classifier (Binary)'
        elif is_flat_surface:
            model = ConvNetFlatSurface()
            class_names = ['BCB', 'BNT', 'WGF', 'WNT']
            model_architecture = 'Flat Surface Detection ConvNet'
        elif is_material_detection_head:
            # Use the working approach: use predict_material from material_detection_functions
            # Lazy import to avoid blocking startup
            from material_detection_functions import predict_material, preprocess_image
            # But we need to load the ConvNetMaterialDetectionHead model first
            model = ConvNetMaterialDetectionHead()
            # Class names in alphabetical order (as ImageFolder sorts them during training)
            # IMPORTANT: ImageFolder assigns class IDs based on alphabetical order of folder names
            # This is the order that the model was trained with, matching multiwebapp
            # Multiwebapp explicitly states: "Class order MUST match ImageFolder's alphabetical assignment during training"
            class_names = [
                '3dmodel',                  # 0 (alphabetically first)
                'LEDscreen',                # 1 (lowercase 's' to match training directory)
                'bowl__purpleplastic',       # 2
                'bowl__whiteceramic',       # 3
                'carrot__natural',          # 4
                'eggplant__natural',        # 5
                'greenpepper__natural',     # 6
                'potato__natural',          # 7
                'redpepper__natural',       # 8
                'teacup__ceramic',          # 9
                'tomato__natural',          # 10
                'yellowpepper__natural'     # 11
            ]
            model_architecture = 'Material Detection Head ConvNet (12 classes)'
            
            # CRITICAL: Use preprocess_image from material_detection_functions for correct preprocessing
            # Lazy import to avoid blocking startup
            from material_detection_functions import preprocess_image
            # But we need to override it to use the correct normalization for ConvNetMaterialDetectionHead
            print(f"DEBUG: Using ConvNetMaterialDetectionHead with material_detection_functions preprocessing")
        else:
            return jsonify({'error': f'Unknown model type. Weight path: {weight_path}'}), 400
        
        # Load weights - handle different checkpoint formats
        print(f"DEBUG: Loading model from checkpoint: {local_model_path}")
        print(f"DEBUG: Model type: {model_architecture}")
        print(f"DEBUG: Expected num_classes: {len(class_names)}")
        
        if isinstance(checkpoint, dict):
            if 'msd' in checkpoint:
                # Our trained models use 'msd' key
                print(f"DEBUG: Loading from 'msd' key")
                model.load_state_dict(checkpoint['msd'], strict=True)
                print(f"DEBUG: Model loaded successfully from 'msd'")
            elif 'state_dict' in checkpoint:
                print(f"DEBUG: Loading from 'state_dict' key")
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                print(f"DEBUG: Model loaded successfully from 'state_dict'")
            elif 'model_state_dict' in checkpoint:
                print(f"DEBUG: Loading from 'model_state_dict' key")
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                print(f"DEBUG: Model loaded successfully from 'model_state_dict'")
            else:
                # Try loading as state_dict directly
                try:
                    print(f"DEBUG: Trying to load checkpoint dict directly as state_dict")
                    model.load_state_dict(checkpoint, strict=True)
                    print(f"DEBUG: Model loaded successfully from checkpoint dict")
                except Exception as e:
                    print(f"Warning: Could not load as state_dict: {e}")
                    # If checkpoint is the model itself (unlikely but possible)
                    if hasattr(checkpoint, 'forward'):
                        model = checkpoint
                        print(f"DEBUG: Using checkpoint as model directly")
                    else:
                        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
        else:
            # Checkpoint might be the model itself (unlikely for our models)
            if hasattr(checkpoint, 'forward'):
                model = checkpoint
                print(f"DEBUG: Using checkpoint as model directly")
            else:
                # Try to load as state_dict
                try:
                    print(f"DEBUG: Trying to load checkpoint as state_dict")
                    model.load_state_dict(checkpoint, strict=True)
                    print(f"DEBUG: Model loaded successfully from checkpoint")
                except Exception as e:
                    print(f"DEBUG: Failed to load checkpoint: {e}")
                    raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
        
        # Verify model output shape
        if hasattr(model, 'fc4'):
            print(f"DEBUG: Model fc4 output features: {model.fc4.out_features}")
            if model.fc4.out_features != len(class_names):
                print(f"WARNING: Model output features ({model.fc4.out_features}) != num classes ({len(class_names)})")
        
        # CRITICAL: Set model to eval mode BEFORE moving to CPU
        # This ensures dropout layers are disabled
        model.eval()
        torch.set_grad_enabled(False)
        
        # CRITICAL: Ensure model and tensor are on CPU (matching multiwebapp)
        model = model.cpu()
        image_tensor = image_tensor.cpu()
        
        # CRITICAL: Double-check model is in eval mode after moving to CPU
        model.eval()
        
        # Verify dropout is disabled
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                if module.training:
                    print(f"WARNING: Dropout layer {name} is still in training mode!")
                    module.eval()
        
        # Debug: Print input tensor info
        print(f"DEBUG: ========== FINAL TENSOR SHAPE CHECK ==========")
        print(f"DEBUG: Input tensor shape: {image_tensor.shape}")
        print(f"DEBUG: Input tensor ndim: {len(image_tensor.shape)}")
        print(f"DEBUG: Expected shape: [1, 3, 16, 16]")
        if image_tensor.shape != torch.Size([1, 3, 16, 16]):
            print(f"DEBUG: WARNING - Tensor shape {image_tensor.shape} != expected [1, 3, 16, 16]")
        print(f"DEBUG: Input tensor range: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
        print(f"DEBUG: Input tensor mean: {image_tensor.mean():.4f}, std: {image_tensor.std():.4f}")
        print(f"DEBUG: Input tensor device: {image_tensor.device}")
        print(f"DEBUG: Model device: {next(model.parameters()).device if list(model.parameters()) else 'N/A'}")
        print(f"DEBUG: ==============================================")
        
        # Run inference
        with torch.no_grad():
            print(f"DEBUG: About to run inference with model type: {type(model)}")
            
            # CRITICAL: Compute hash of input tensor to verify it's different for different images
            import hashlib
            tensor_bytes = image_tensor.cpu().numpy().tobytes()
            tensor_hash = hashlib.md5(tensor_bytes).hexdigest()
            print(f"DEBUG: Input tensor hash (MD5): {tensor_hash}")
            print(f"DEBUG: Input tensor sample (first 10 values): {image_tensor[0, 0, :3, :3].cpu().numpy().flatten()[:10]}")
            
            # Verify model is actually in eval mode
            print(f"DEBUG: Model training mode: {model.training}")
            if model.training:
                print("ERROR: Model is in training mode! Setting to eval mode...")
                model.eval()
            
            predictions = model(image_tensor)
            
            # Compute hash of predictions to verify they're different
            pred_hash = hashlib.md5(predictions.cpu().numpy().tobytes()).hexdigest()
            print(f"DEBUG: Predictions hash (MD5): {pred_hash}")
            
            # CRITICAL: Check if predictions are all zeros or all the same
            import numpy as np
            pred_np = predictions.cpu().numpy()[0]
            if np.allclose(pred_np, 0):
                print(f"⚠️  ERROR: All predictions are zeros!")
            elif np.allclose(pred_np, pred_np[0]):
                print(f"⚠️  ERROR: All predictions are the same value: {pred_np[0]}")
            else:
                print(f"DEBUG: Predictions vary (min: {pred_np.min():.4f}, max: {pred_np.max():.4f}, std: {pred_np.std():.4f})")
        
        # Debug: Print raw predictions
        print(f"DEBUG: ========== MODEL INFERENCE OUTPUT ==========")
        print(f"DEBUG: Raw predictions shape: {predictions.shape}")
        print(f"DEBUG: Raw predictions (logits): {predictions.cpu().numpy()}")
        print(f"DEBUG: Raw predictions min: {predictions.min():.4f}, max: {predictions.max():.4f}")
        print(f"DEBUG: Model type: {type(model)}")
        print(f"DEBUG: Using class_names: {class_names}")
        print(f"DEBUG: Number of classes: {len(class_names)}")
        print(f"DEBUG: ============================================")
        
        # Process predictions
        if is_material_purity:
            # Binary classification - sigmoid output
            prob_pure = predictions[0, 0].item() if predictions.shape == (1, 1) else predictions.flatten()[0].item()
            prob_impure = 1.0 - prob_pure
            predicted_class = 1 if prob_pure >= 0.5 else 0
            confidence = prob_pure if predicted_class == 1 else prob_impure
            
            top3_predictions = [
                {'class': 'Pure', 'probability': prob_pure},
                {'class': 'Impure', 'probability': prob_impure}
            ]
            top3_predictions.sort(key=lambda x: x['probability'], reverse=True)
        else:
            # Multi-class classification - softmax (EXACTLY as in eval script)
            # The eval script uses: prob = F.softmax(prediction, dim=1)
            import torch.nn.functional as F
            
            # CRITICAL: Verify predictions tensor is valid
            if predictions.shape[0] == 0 or predictions.shape[1] != len(class_names):
                print(f"ERROR: Invalid predictions shape: {predictions.shape}, expected [1, {len(class_names)}]")
                return jsonify({'error': f'Invalid model output shape: {predictions.shape}'}), 500
            
            probabilities = F.softmax(predictions, dim=1)[0]  # Get probabilities for first (and only) image
            
            # CRITICAL: Verify probabilities are valid
            if torch.any(torch.isnan(probabilities)) or torch.any(torch.isinf(probabilities)):
                print(f"ERROR: Invalid probabilities (NaN or Inf): {probabilities}")
                return jsonify({'error': 'Model produced invalid probabilities (NaN or Inf)'}), 500
            
            # Get predicted class - ensure it's within valid range
            predicted_class = torch.argmax(probabilities, dim=0).item()
            if predicted_class < 0 or predicted_class >= len(class_names):
                print(f"ERROR: Predicted class index {predicted_class} out of range [0, {len(class_names)-1}]")
                return jsonify({'error': f'Invalid predicted class index: {predicted_class}'}), 500
            
            confidence = probabilities[predicted_class].item()
            
            # CRITICAL: Double-check that predicted_class is actually the argmax
            actual_argmax = torch.argmax(probabilities).item()
            if predicted_class != actual_argmax:
                print(f"ERROR: predicted_class ({predicted_class}) != actual_argmax ({actual_argmax})")
                predicted_class = actual_argmax
                confidence = probabilities[predicted_class].item()
            
            # Debug: Print probabilities
            print(f"DEBUG: ========== MATERIAL DETECTION HEAD INFERENCE ==========")
            print(f"DEBUG: Raw predictions (logits): {predictions.cpu().numpy()}")
            print(f"DEBUG: Probabilities after softmax: {probabilities.tolist()}")
            print(f"DEBUG: Predicted class index: {predicted_class}")
            print(f"DEBUG: Class names: {class_names}")
            print(f"DEBUG: Class names length: {len(class_names)}")
            print(f"DEBUG: Predicted class name: {class_names[predicted_class] if predicted_class < len(class_names) else 'OUT_OF_RANGE'}")
            print(f"DEBUG: Confidence: {confidence:.6f}")
            print(f"DEBUG: All class probabilities: {probabilities.tolist()}")
            print(f"DEBUG: Expected class order (ImageFolder alphabetical): {class_names}")
            print(f"DEBUG: CRITICAL - Class index {predicted_class} maps to class name: {class_names[predicted_class] if predicted_class < len(class_names) else 'OUT_OF_RANGE'}")
            
            # Check if all probabilities are the same (would indicate a problem)
            prob_values = probabilities.tolist()
            if len(set([round(p, 6) for p in prob_values])) == 1:
                print(f"⚠️  WARNING: All probabilities are identical! This suggests a model or input issue.")
            elif max(prob_values) - min(prob_values) < 0.001:
                print(f"⚠️  WARNING: Probabilities are nearly identical (range < 0.001). This suggests a model or input issue.")
            
            # Check if predicted class is always 0 (3dmodel) - this might indicate an issue
            if predicted_class == 0:
                print(f"⚠️  WARNING: Predicted class is index 0 (3dmodel). This might be incorrect!")
                print(f"DEBUG: Probability for index 0 (3dmodel): {probabilities[0]:.6f}")
                print(f"DEBUG: Probability for index 1 (LEDscreen): {probabilities[1]:.6f}")
                print(f"DEBUG: Max probability index: {torch.argmax(probabilities).item()}")
                print(f"DEBUG: All probabilities: {[f'{i}:{p:.6f}' for i, p in enumerate(prob_values)]}")
                print(f"DEBUG: If this is an LED image, it should be predicted as 'LEDscreen' (index 1), not '3dmodel' (index 0)")
                print(f"DEBUG: Check if the STO file has the correct image at index 1 (16x16 material detection image)")
            
            print(f"DEBUG: ======================================================")
            
            # Get top 3 predictions (as in eval script: top_p, top_class = prob.topk(1, dim=1))
            top3_probs, top3_indices = torch.topk(probabilities, min(3, len(class_names)))
            print(f"DEBUG: Top 3 indices: {top3_indices.tolist()}")
            print(f"DEBUG: Top 3 probabilities: {top3_probs.tolist()}")
            
            # Function to format class name for display (consolidate to materials only)
            def format_class_name(class_name):
                """Format class name to consolidated material format"""
                # Handle special cases
                if class_name == "3dmodel":
                    return "3D Model"
                elif class_name == "LEDscreen":
                    return "LED"
                elif "__" in class_name:
                    # Format: "carrot__natural" -> "natural carrot"
                    # Format: "bowl__purpleplastic" -> "purple plastic bowl"
                    parts = class_name.split("__")
                    if len(parts) == 2:
                        material, type_ = parts
                        # Handle compound words in type_ (e.g., "purpleplastic" -> "purple plastic")
                        if type_ == "purpleplastic":
                            type_ = "purple plastic"
                        elif type_ == "whiteceramic":
                            type_ = "white ceramic"
                        # Lowercase and combine: "natural carrot"
                        return f"{type_} {material}".lower()
                # Default: replace underscores and title case
                return class_name.replace("__", " ").replace("_", " ").title()
            
            # Get top 3 predictions
            top3_predictions = []
            for i in range(min(3, len(class_names))):
                class_idx = top3_indices[i].item()
                prob = top3_probs[i].item()
                class_name = class_names[class_idx] if class_idx < len(class_names) else f'class_{class_idx}'
                display_class = format_class_name(class_name)
                top3_predictions.append({
                    'class': class_name,  # Keep original for internal use
                    'display_class': display_class,  # Formatted for display
                    'probability': prob
                })
                print(f"DEBUG: Top {i+1}: class_idx={class_idx}, class_name={class_name}, prob={prob:.4f}")
        
        inference_time = (time.time() - start_time) * 1000
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'predicted_class': class_names[predicted_class] if predicted_class < len(class_names) else 'unknown',
            'confidence': confidence,
            'top3_predictions': top3_predictions,
            'inference_time': inference_time,
            'image': f"data:image/jpeg;base64,{img_base64}",
            'architecture': model_architecture,
            'model_size': 'Variable',
            'input_size': '16x16',
            'batch_size': 1,
            'classes': class_names,
            'classes_display': ', '.join(class_names)
        })
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"ERROR in api_detect_material_head: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({
            'success': False,
            'error': f'Detection failed: {str(e)}',
            'error_type': 'detection_failed',
            'traceback': error_traceback
        }), 500

@app.route("/api/extract_sto_index0", methods=["POST"])
def api_extract_sto_index0():
    """API endpoint to extract index 1 (16x16 material detection image) from STO file
    Note: Despite the name 'index0', this extracts index 1 which is the material detection image.
    STO structure: Index 0=metadata, Index 1=16x16 material image, Index 2=OD metadata, Index 3=640x640 OD image"""
    import pickle
    import io
    import base64
    from io import BytesIO
    from PIL import Image
    
    temp_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        filename_lower = file.filename.lower()
        if not filename_lower.endswith('.sto'):
            return jsonify({'success': False, 'error': 'Please upload a STO file'}), 400
        
        # Save and load STO file
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_sto_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.sto")
        file.save(temp_path)
        
        if not os.path.exists(temp_path):
            return jsonify({'success': False, 'error': 'Failed to save uploaded file'}), 500
        
        # Load STO file
        with open(temp_path, 'rb') as f:
            sto_data = pickle.load(f)
        
        if not isinstance(sto_data, (list, tuple)):
            return jsonify({'success': False, 'error': f'Invalid STO file format: expected list/tuple, got {type(sto_data).__name__}'}), 400
        
        if len(sto_data) == 0:
            return jsonify({'success': False, 'error': 'STO file is empty'}), 400
        
        # Extract index 1 (16x16 material detection image)
        # STO structure: Index 0=metadata, Index 1=16x16 material image, Index 2=OD metadata, Index 3=640x640 OD image
        if len(sto_data) < 2:
            return jsonify({'success': False, 'error': 'STO file does not have index 1 (material detection image)'}), 400
        
        sto_item = sto_data[1]
        image = None
        
        if isinstance(sto_item, bytes):
            image = Image.open(BytesIO(sto_item)).convert('RGB')
        elif isinstance(sto_item, Image.Image):
            image = sto_item.convert('RGB')
        elif hasattr(sto_item, 'mode'):
            # PIL Image object
            image = sto_item.convert('RGB')
        elif isinstance(sto_item, (list, tuple)) and len(sto_item) > 0:
            # Nested structure, try first element
            nested_item = sto_item[0]
            if isinstance(nested_item, bytes):
                image = Image.open(BytesIO(nested_item)).convert('RGB')
            elif isinstance(nested_item, Image.Image):
                image = nested_item.convert('RGB')
            else:
                return jsonify({'success': False, 'error': f'Invalid STO structure at index 1: unexpected type {type(sto_item).__name__}'}), 400
        else:
            return jsonify({'success': False, 'error': f'Invalid STO structure at index 1: expected image, got {type(sto_item).__name__}'}), 400
        
        if image is None:
            return jsonify({'success': False, 'error': 'Failed to extract image from STO file'}), 500
        
        # Convert to base64 using PNG (lossless) to preserve pixel values for material detection
        # CRITICAL: JPEG compression causes pixel value changes that affect model predictions
        # PNG is lossless and preserves exact pixel values, especially important for 16x16 images
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{img_base64}",
            'image_size': image.size,
            'image_mode': image.mode
        })
    except pickle.UnpicklingError as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to unpickle STO file: {str(e)}',
            'error_type': 'unpickling_failed'
        }), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.route("/api/extract_sto_index1", methods=["POST"])
def api_extract_sto_index1():
    """API endpoint to extract index 3 (640x640 object detection image) from STO file
    Note: Despite the name 'index1', this extracts index 3 which is the object detection image.
    STO structure: Index 0=metadata, Index 1=16x16 material image, Index 2=OD metadata, Index 3=640x640 OD image"""
    import pickle
    import io
    import base64
    from io import BytesIO
    from PIL import Image
    
    temp_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        filename_lower = file.filename.lower()
        if not filename_lower.endswith('.sto'):
            return jsonify({'success': False, 'error': 'Please upload a STO file'}), 400
        
        # Save and load STO file
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_sto_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.sto")
        file.save(temp_path)
        
        if not os.path.exists(temp_path):
            return jsonify({'success': False, 'error': 'Failed to save uploaded file'}), 500
        
        # Load STO file
        with open(temp_path, 'rb') as f:
            sto_data = pickle.load(f)
        
        if not isinstance(sto_data, (list, tuple)):
            return jsonify({'success': False, 'error': f'Invalid STO file format: expected list/tuple, got {type(sto_data).__name__}'}), 400
        
        if len(sto_data) < 4:
            return jsonify({'success': False, 'error': 'STO file does not have index 3 (object detection image)'}), 400
        
        # Extract index 3 (640x640 object detection image)
        # STO structure: Index 0=metadata, Index 1=16x16 material image, Index 2=OD metadata, Index 3=640x640 OD image
        sto_item = sto_data[3]
        image = None
        
        if isinstance(sto_item, bytes):
            image = Image.open(BytesIO(sto_item)).convert('RGB')
        elif isinstance(sto_item, Image.Image):
            image = sto_item.convert('RGB')
        elif hasattr(sto_item, 'mode'):
            # PIL Image object
            image = sto_item.convert('RGB')
        elif isinstance(sto_item, (list, tuple)) and len(sto_item) > 0:
            # Nested structure, try first element
            nested_item = sto_item[0]
            if isinstance(nested_item, bytes):
                image = Image.open(BytesIO(nested_item)).convert('RGB')
            elif isinstance(nested_item, Image.Image):
                image = nested_item.convert('RGB')
            else:
                return jsonify({'success': False, 'error': f'Invalid STO structure at index 1: unexpected type {type(sto_item).__name__}'}), 400
        else:
            return jsonify({'success': False, 'error': f'Invalid STO structure at index 1: unexpected type {type(sto_item).__name__}'}), 400
        
        if image is None:
            return jsonify({'success': False, 'error': 'Failed to extract image from STO file'}), 500
        
        # Convert to base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=95)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_base64}",
            'image_size': image.size,
            'image_mode': image.mode
        })
    except pickle.UnpicklingError as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to unpickle STO file: {str(e)}',
            'error_type': 'unpickling_failed'
        }), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.route('/api/yolov3_weights', methods=['GET'])
def api_yolov3_weights():
    """API endpoint to get YOLOv3 model weights from local directory or Hugging Face Hub"""
    try:
        import glob
        import re
        
        yolov3_weights = []
        weights_dir = os.path.join(get_models_dir(), "yolov3")
        repo_id = "mvplus/spatiotemporal_models"
        subfolder = "yolov3"
        
        # First, try to get weights from Hugging Face Hub
        hub_files = fetch_weights_from_hub(repo_id, subfolder)
        for hub_file in hub_files:
            filename = os.path.basename(hub_file)
            # Parse YOLOv3 model filenames
            if "best" in filename.lower():
                weight_type = "Best"
                acc_match = re.search(r'acc_(\d+\.?\d*)%', filename)
                if acc_match:
                    acc_score = float(acc_match.group(1))
                    display_name = f"YOLOv3 Best ({acc_score}% accuracy)"
                else:
                    display_name = f"YOLOv3 Best"
            elif "last" in filename.lower():
                weight_type = "Last"
                acc_match = re.search(r'acc_(\d+\.?\d*)%', filename)
                if acc_match:
                    acc_score = float(acc_match.group(1))
                    display_name = f"YOLOv3 Last ({acc_score}% accuracy)"
                else:
                    display_name = f"YOLOv3 Last"
            else:
                weight_type = "Checkpoint"
                display_name = f"YOLOv3 Checkpoint ({filename.replace('.pt', '')})"
            
            yolov3_weights.append({
                "filename": filename,
                "path": f"hub://{repo_id}/{hub_file}",
                "display_name": display_name,
                "weight_type": weight_type,
                "source": "hub"
            })
        
        # Also check local files
        
        # Look for .pt files in the weights directory
        if os.path.exists(weights_dir):
            weight_files = glob.glob(os.path.join(weights_dir, "*.pt"))
            
            for weight_file in weight_files:
                filename = os.path.basename(weight_file)
                
                # Parse YOLOv3 model filenames
                if "best" in filename.lower():
                    weight_type = "Best"
                    # Extract accuracy from filename if available
                    acc_match = re.search(r'acc_(\d+\.?\d*)%', filename)
                    if acc_match:
                        acc_score = float(acc_match.group(1))
                        display_name = f"YOLOv3 Best ({acc_score}% accuracy)"
                    else:
                        display_name = f"YOLOv3 Best"
                elif "last" in filename.lower():
                    weight_type = "Last"
                    # Extract accuracy from filename if available
                    acc_match = re.search(r'acc_(\d+\.?\d*)%', filename)
                    if acc_match:
                        acc_score = float(acc_match.group(1))
                        display_name = f"YOLOv3 Last ({acc_score}% accuracy)"
                    else:
                        display_name = f"YOLOv3 Last"
                else:
                    weight_type = "Checkpoint"
                    display_name = f"YOLOv3 Checkpoint ({filename.replace('.pt', '')})"
                
                # Check if this weight is already in the list from Hub
                existing = next((w for w in yolov3_weights if w['filename'] == filename), None)
                if not existing:
                    yolov3_weights.append({
                        "filename": filename,
                        "path": weight_file,
                        "display_name": display_name,
                        "weight_type": weight_type,
                        "source": "local"
                    })
        
        # Sort by weight type (Best first), then by filename
        yolov3_weights.sort(key=lambda x: (x['weight_type'] == 'Best', x['filename']), reverse=True)
        
        return jsonify({
            'success': True,
            'weights': yolov3_weights
        })
    except Exception as e:
        import traceback
        print(f"Error loading YOLOv3 weights: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'weights': []
        }), 500

@app.route('/api/yolov8_custom_weights', methods=['GET'])
def api_yolov8_custom_weights():
    """API endpoint to get YOLOv8 custom model weights from local directory or Hugging Face Hub"""
    import os
    import glob
    import re
    
    try:
        yolov8_weights = []
        weights_dir = os.path.join(get_models_dir(), "yolov8")
        repo_id = "mvplus/spatiotemporal_models"
        subfolder = "yolov8"
        
        # First, try to get weights from Hugging Face Hub
        hub_files = fetch_weights_from_hub(repo_id, subfolder)
        for hub_file in hub_files:
            filename = os.path.basename(hub_file)
            # Parse YOLOv8 model filenames
            if "best" in filename.lower():
                weight_type = "Best"
                map_match = re.search(r'(\d+\.?\d*)mAp', filename, re.IGNORECASE)
                if map_match:
                    map_score = float(map_match.group(1))
                    display_name = f"YOLOv8 Best ({map_score}% mAP)"
                else:
                    display_name = f"YOLOv8 Best"
            elif "last" in filename.lower():
                weight_type = "Last"
                map_match = re.search(r'(\d+\.?\d*)mAp', filename, re.IGNORECASE)
                if map_match:
                    map_score = float(map_match.group(1))
                    display_name = f"YOLOv8 Last ({map_score}% mAP)"
                else:
                    display_name = f"YOLOv8 Last"
            else:
                weight_type = "Checkpoint"
                display_name = f"YOLOv8 Checkpoint ({filename.replace('.pt', '')})"
            
            yolov8_weights.append({
                "filename": filename,
                "path": f"hub://{repo_id}/{hub_file}",
                "display_name": display_name,
                "weight_type": weight_type,
                "source": "hub"
            })
        
        # Also check local files
        
        # Look for .pt files in the weights directory
        if os.path.exists(weights_dir):
            weight_files = glob.glob(os.path.join(weights_dir, "*.pt"))
            
            for weight_file in weight_files:
                filename = os.path.basename(weight_file)
                
                # Parse YOLOv8 model filenames
                if "best" in filename.lower():
                    weight_type = "Best"
                    # Extract mAP from filename if available
                    map_match = re.search(r'(\d+\.?\d*)mAp', filename, re.IGNORECASE)
                    if map_match:
                        map_score = float(map_match.group(1))
                        display_name = f"YOLOv8 {weight_type} ({map_score}% mAP)"
                    else:
                        display_name = f"YOLOv8 {weight_type}"
                elif "last" in filename.lower():
                    weight_type = "Last"
                    # Extract mAP from filename if available
                    map_match = re.search(r'(\d+\.?\d*)mAp', filename, re.IGNORECASE)
                    if map_match:
                        map_score = float(map_match.group(1))
                        display_name = f"YOLOv8 {weight_type} ({map_score}% mAP)"
                    else:
                        display_name = f"YOLOv8 {weight_type}"
                else:
                    weight_type = "Checkpoint"
                    display_name = f"YOLOv8 {weight_type} ({filename.replace('.pt', '')})"
                
                # Check if this weight is already in the list from Hub
                existing = next((w for w in yolov8_weights if w['filename'] == filename), None)
                if not existing:
                    yolov8_weights.append({
                        "filename": filename,
                        "path": weight_file,
                        "display_name": display_name,
                        "weight_type": weight_type,
                        "source": "local"
                    })
        
        # Sort by weight type (Best first, then Last, then others)
        weight_priority = {"Best": 0, "Last": 1, "Checkpoint": 2}
        yolov8_weights.sort(key=lambda x: (weight_priority.get(x['weight_type'], 3), x['filename']))
        
        return jsonify({
            'success': True,
            'weights': yolov8_weights
        })
    except Exception as e:
        import traceback
        print(f"Error loading YOLOv8 weights: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'weights': []
        }), 500

@app.route('/api/dinov3_weights', methods=['GET'])
def api_dinov3_weights():
    """API endpoint to get DINOv3 model weights from local directory or Hugging Face Hub"""
    import os
    import glob
    import re
    
    try:
        dinov3_weights = []
        weights_dir = os.path.join(get_models_dir(), "dinov3")
        repo_id = "mvplus/spatiotemporal_models"
        subfolder = "dinov3"
        
        # First, try to get weights from Hugging Face Hub
        hub_files = fetch_weights_from_hub(repo_id, subfolder)
        for hub_file in hub_files:
            filename = os.path.basename(hub_file)
            # Parse DINOv3 model filenames
            epoch_match = re.search(r'epoch_(\d+)', filename)
            acc_match = re.search(r'acc_(\d+)_(\d+)%', filename)
            epoch = int(epoch_match.group(1)) if epoch_match else 0
            acc_whole = int(acc_match.group(1)) if acc_match else 0
            acc_decimal = int(acc_match.group(2)) if acc_match and len(acc_match.groups()) > 1 else 0
            accuracy = float(f"{acc_whole}.{acc_decimal}") if acc_match else 0.0
            
            # Determine weight type
            if "best" in filename.lower() or accuracy >= 97.0:
                weight_type = "Best"
            elif "last" in filename.lower() or epoch >= 80:
                weight_type = "Last"
            else:
                weight_type = "Checkpoint"
            
            display_name = f"DINOv3 Epoch {epoch} ({accuracy}% accuracy)" if accuracy > 0 else f"DINOv3 Epoch {epoch}"
            
            dinov3_weights.append({
                "filename": filename,
                "path": f"hub://{repo_id}/{hub_file}",
                "display_name": display_name,
                "weight_type": weight_type,
                "epoch": epoch,
                "accuracy": accuracy,
                "source": "hub"
            })
        
        # Also check local files
        
        # Look for .pth files in the weights directory
        if os.path.exists(weights_dir):
            weight_files = glob.glob(os.path.join(weights_dir, "*.pth"))
            
            for weight_file in weight_files:
                filename = os.path.basename(weight_file)
                
                # Parse DINOv3 model filenames (e.g., "lastweight_epoch_82_train_0_0411_val_0_1622_acc_96_88%.pth")
                # Extract epoch, accuracy, and loss values
                epoch_match = re.search(r'epoch_(\d+)', filename)
                acc_match = re.search(r'acc_(\d+)_(\d+)%', filename)
                train_loss_match = re.search(r'train_([\d_]+)', filename)
                val_loss_match = re.search(r'val_([\d_]+)', filename)
                
                epoch = int(epoch_match.group(1)) if epoch_match else 0
                acc_whole = int(acc_match.group(1)) if acc_match else 0
                acc_decimal = int(acc_match.group(2)) if acc_match and len(acc_match.groups()) > 1 else 0
                accuracy = float(f"{acc_whole}.{acc_decimal}") if acc_match else 0.0
                
                # Determine weight type based on accuracy and epoch
                if "best" in filename.lower() or accuracy >= 97.0:
                    weight_type = "Best"
                elif "last" in filename.lower() or epoch >= 80:
                    weight_type = "Last"
                else:
                    weight_type = "Checkpoint"
                
                # Create display name with accuracy
                if accuracy > 0:
                    display_name = f"DINOv3 Epoch {epoch} ({accuracy}% accuracy)"
                else:
                    display_name = f"DINOv3 Epoch {epoch}"
                
                # Check if this weight is already in the list from Hub
                existing = next((w for w in dinov3_weights if w['filename'] == filename), None)
                if not existing:
                    dinov3_weights.append({
                        "filename": filename,
                        "path": weight_file,
                        "display_name": display_name,
                        "weight_type": weight_type,
                        "epoch": epoch,
                        "accuracy": accuracy,
                        "source": "local"
                    })
        
        # Sort by weight type (Best first, then Last, then by accuracy/epoch)
        weight_priority = {"Best": 0, "Last": 1, "Checkpoint": 2}
        dinov3_weights.sort(key=lambda x: (
            weight_priority.get(x.get('weight_type', ''), 4),
            -x.get('accuracy', 0),  # Higher accuracy first
            -x.get('epoch', 0)  # Higher epoch first
        ))
        
        return jsonify({
            'success': True,
            'weights': dinov3_weights
        })
    except Exception as e:
        import traceback
        print(f"Error loading DINOv3 weights: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'weights': []
        }), 500

@app.route('/api/detect_dinov3', methods=['POST'])
def api_detect_dinov3():
    """API endpoint for DINOv3 model detection - CLASSIFICATION"""
    import time
    from PIL import Image
    import numpy as np
    import base64
    import io
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import vit_b_16
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get selected weight
        weight_path = request.form.get('weight_path', '')
        print(f"DEBUG: Received weight_path: '{weight_path}'")
        
        if not weight_path:
            return jsonify({'error': 'No model weight selected'}), 400
        
        # Handle Hub paths - download on demand
        if weight_path.startswith('hub://'):
            downloaded_path = download_model_from_hub(weight_path)
            if downloaded_path:
                weight_path = downloaded_path
            else:
                return jsonify({'error': f'Failed to download model from Hub: {weight_path}'}), 400
        
        # Load image
        file.stream.seek(0)
        image = Image.open(file.stream).convert('RGB')
        
        print(f"DEBUG: Original image size: {image.size}")
        
        # Check if weight file exists
        if not os.path.exists(weight_path):
            print(f"DEBUG: Weight file does not exist: {weight_path}")
            return jsonify({'error': f'Model weight file not found: {weight_path}'}), 400
        
        print(f"DEBUG: Weight file exists, size: {os.path.getsize(weight_path)} bytes")
        
        # Start timing
        start_time = time.time()
        
        try:
            # Import the DINOv3 model class
            import sys
            dinov3_path = os.path.join(BASE_DIR, "apps", "dinov3_custom")
            sys.path.append(dinov3_path)
            try:
                from train_dinov3 import DINOv3Model
            except ImportError as import_error:
                print(f"DEBUG: Failed to import DINOv3Model: {import_error}")
                # Define DINOv3Model inline if import fails
                class DINOv3Model(nn.Module):
                    """DINOv3 model with Vision Transformer backbone"""
                    def __init__(self, num_classes, pretrained_path=None):
                        super().__init__()
                        self.num_classes = num_classes
                        self.backbone = vit_b_16(pretrained=True)
                        original_head = self.backbone.heads[0]
                        self.backbone.heads = nn.Sequential(
                            nn.Linear(original_head.in_features, num_classes)
                        )
                        if pretrained_path and os.path.exists(pretrained_path):
                            try:
                                checkpoint = torch.load(pretrained_path, map_location='cpu')
                                backbone_state = {}
                                for key, value in checkpoint.items():
                                    if not key.startswith('head'):
                                        backbone_state[key] = value
                                self.backbone.load_state_dict(backbone_state, strict=False)
                            except Exception as e:
                                print(f"Warning: Could not load DINOv3 weights: {e}")
                    def forward(self, x):
                        return self.backbone(x)
            
            # Load checkpoint
            ckpt = torch.load(weight_path, map_location='cpu')
            print(f"DEBUG: Checkpoint keys: {list(ckpt.keys())}")
            
            # Infer num_classes from model state dict
            if 'model_state_dict' in ckpt:
                msd = ckpt['model_state_dict']
                # Find the head layer to get num_classes
                for key in msd.keys():
                    if 'heads' in key and 'weight' in key:
                        num_classes = msd[key].shape[0]
                        print(f"DEBUG: Inferred num_classes from {key}: {num_classes}")
                        break
                else:
                    raise ValueError("Could not infer num_classes from model state dict")
            else:
                raise ValueError("Checkpoint does not contain model_state_dict")
            
            # Get class names from checkpoint or data directory
            if 'class_names' in ckpt:
                class_names = ckpt['class_names']
                if isinstance(class_names, str):
                    class_names = [class_names]
            else:
                # Try to load class names from data directory
                data_dir = os.path.join(BASE_DIR, "apps", "dinov3_custom", "data", "train")
                class_names = []
                if os.path.exists(data_dir):
                    from pathlib import Path
                    class_set = set()
                    for img_path in Path(data_dir).glob('*.jpg'):
                        filename = img_path.stem
                        class_name = filename.split('__')[0]
                        class_set.add(class_name)
                    class_names = sorted(list(class_set))
                    print(f"DEBUG: Loaded class names from data directory: {class_names}")
                
                # If still no class names, use mapping based on sorted alphabetical order
                # DINOv3 training uses sorted(class_set) which gives alphabetical order
                # This is different from YOLOv8's order!
                if not class_names or len(class_names) != num_classes:
                    # Default class names in ALPHABETICAL ORDER (as DINOv3 training uses sorted())
                    # This matches how train_dinov3.py creates class_names: sorted(list(class_set))
                    # Actual classes from data directory (9 classes, sorted alphabetically):
                    DEFAULT_CLASS_NAMES_ALPHABETICAL = [
                        'bowl',         # 0
                        'carrot',       # 1
                        'eggplant',     # 2
                        'greenpepper',  # 3
                        'potato',       # 4
                        'redpepper',    # 5
                        'teacup',       # 6
                        'tomato',       # 7
                        'yellowpepper'  # 8
                    ]
                    
                    # Use default names if we have the right number, otherwise use generic
                    if num_classes <= len(DEFAULT_CLASS_NAMES_ALPHABETICAL):
                        class_names = DEFAULT_CLASS_NAMES_ALPHABETICAL[:num_classes]
                        print(f"DEBUG: Using alphabetical default class names for DINOv3: {class_names}")
                    else:
                        class_names = [f'class_{i}' for i in range(num_classes)]
                        print(f"DEBUG: Warning - using generic class names: {class_names}")
            
            # Ensure class_names length matches num_classes
            if len(class_names) != num_classes:
                print(f"DEBUG: Warning - class_names length ({len(class_names)}) != num_classes ({num_classes}), adjusting...")
                if len(class_names) < num_classes:
                    # Add generic names for missing classes
                    for i in range(len(class_names), num_classes):
                        class_names.append(f'class_{i}')
                else:
                    # Truncate if too many
                    class_names = class_names[:num_classes]
            
            print(f"DEBUG: DINOv3 - num_classes: {num_classes}, class_names: {class_names}")
            
            # Create model instance
            model = DINOv3Model(num_classes=num_classes)
            
            # Load weights
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            elif 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            elif 'msd' in ckpt:
                model.load_state_dict(ckpt['msd'])
            else:
                # Try loading checkpoint as state_dict directly
                try:
                    model.load_state_dict(ckpt)
                except:
                    raise ValueError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")
            
            model.eval()
            print(f"DEBUG: DINOv3 model loaded successfully")
            
            # Preprocess image - DINOv3 uses standard ImageNet normalization
            img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = img_transform(image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                logits = model(img_tensor)
            
            print(f"DEBUG: Inference complete, logits shape: {logits.shape}")
            
            # Apply softmax to get probabilities
            import torch.nn.functional as F
            probabilities = F.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities, dim=0).item()
            confidence = probabilities[predicted_class].item()
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, min(3, num_classes))
            top3_predictions = []
            for i in range(min(3, num_classes)):
                class_idx = top3_indices[i].item()
                prob = top3_probs[i].item()
                top3_predictions.append({
                    'class': class_names[class_idx] if class_idx < len(class_names) else f'class_{class_idx}',
                    'probability': float(prob)
                })
            
            # Get all predictions
            all_predictions = []
            for i in range(num_classes):
                all_predictions.append({
                    'class': class_names[i] if i < len(class_names) else f'class_{i}',
                    'probability': float(probabilities[i].item())
                })
            all_predictions.sort(key=lambda x: x['probability'], reverse=True)
            
            inference_time = (time.time() - start_time) * 1000
            
            # Convert image to base64 for display
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Extract fitness/accuracy from checkpoint if available
            fitness_score = None
            if 'fitness' in ckpt:
                fitness_score = float(ckpt.get('fitness', 0.0))
            elif 'best_fitness' in ckpt:
                fitness_score = float(ckpt.get('best_fitness', 0.0))
            elif 'accuracy' in ckpt:
                fitness_score = float(ckpt.get('accuracy', 0.0))
            
            return jsonify({
                'success': True,
                'predicted_class': class_names[predicted_class] if predicted_class < len(class_names) else 'unknown',
                'confidence': float(confidence),
                'top3_predictions': top3_predictions,
                'all_predictions': all_predictions,
                'inference_time': float(inference_time),
                'image': f"data:image/jpeg;base64,{img_base64}",
                'architecture': 'DINOv3 (Vision Transformer)',
                'model_size': '86M parameters (ViT-B/16)',
                'input_size': '224x224',
                'batch_size': 1,
                'classes': class_names,
                'classes_display': ', '.join(class_names),
                'backbone': 'Vision Transformer (ViT-B/16)',
                'detection_heads': '1',
                'anchors': 'N/A (Classification)',
                'fitness_score': fitness_score if fitness_score is not None else 'N/A',
                'used_weight_path': weight_path,
                'inferred_num_classes': int(num_classes)
            })
            
        except Exception as model_error:
            print(f"DEBUG: Model loading/inference failed: {model_error}")
            import traceback
            traceback_str = traceback.format_exc()
            print(f"FULL TRACEBACK:\n{traceback_str}")
            return jsonify({
                'success': False,
                'error': str(model_error),
                'traceback': traceback_str
            }), 500
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"ERROR in api_detect_dinov3: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({
            'error': f'Detection failed: {str(e)}',
            'error_type': 'detection_failed'
        }), 500

@app.route('/api/detect_yolov8_custom', methods=['POST'])
def api_detect_yolov8_custom():
    """API endpoint for YOLOv8 custom model detection - PROPER OBJECT DETECTION"""
    import time
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np
    import base64
    import io
    from PIL import ImageDraw, ImageFont
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get selected weight
        weight_path = request.form.get('weight_path', '')
        print(f"DEBUG: Received weight_path: '{weight_path}'")
        
        if not weight_path:
            return jsonify({'error': 'No model weight selected'}), 400
        
        # Handle Hub paths (hub://repo_id/path/to/file)
        if weight_path.startswith('hub://'):
            # Extract repo_id and file path
            parts = weight_path.replace('hub://', '').split('/', 1)
            if len(parts) == 2:
                repo_id = parts[0]
                file_path = parts[1]
                # Download from Hub
                downloaded_path = download_model_from_hub(repo_id, file_path)
                if downloaded_path:
                    weight_path = downloaded_path
                else:
                    return jsonify({'error': f'Failed to download model from Hub: {weight_path}'}), 400
            else:
                return jsonify({'error': f'Invalid Hub path format: {weight_path}'}), 400
        
        # Load image - ensure stream is at beginning
        file.stream.seek(0)
        image = Image.open(file.stream).convert('RGB')
        
        print(f"DEBUG: Original image size: {image.size}")
        
        # Check if weight file exists
        if not os.path.exists(weight_path):
            print(f"DEBUG: Weight file does not exist: {weight_path}")
            return jsonify({'error': f'Model weight file not found: {weight_path}'}), 400
        
        print(f"DEBUG: Weight file exists, size: {os.path.getsize(weight_path)} bytes")
        
        # Start timing
        start_time = time.time()
        
        try:
            # Load the YOLOv8 model using Ultralytics
            model = YOLO(weight_path)
            print(f"DEBUG: YOLOv8 model loaded successfully")
            
            # Get model metadata
            num_classes = len(model.names)
            class_list = ', '.join(list(model.names.values())[:5]) + ('...' if num_classes > 5 else '')
            
            # Run inference - PROPER OBJECT DETECTION
            results = model(image, conf=0.01)  # Lower threshold to detect more objects
            
            # Process detection results
            detections = []
            all_class_confidences = {}
            
            # Class names from the model
            class_names = model.names
            print(f"DEBUG: Model class names: {class_names}")
            
            # Initialize all class confidences to 0
            for class_id, class_name in class_names.items():
                all_class_confidences[class_name] = 0.0
            
            # Process each detection
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    print(f"DEBUG: YOLOv8 found {len(boxes)} detections from model")
                    for i in range(len(boxes)):
                        # Get bounding box coordinates
                        box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                        confidence = boxes.conf[i].cpu().numpy()
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = class_names[class_id]  # Use class name directly from model
                        
                        # Add detection
                        detections.append({
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': {
                                'x1': float(box[0]),
                                'y1': float(box[1]),
                                'x2': float(box[2]),
                                'y2': float(box[3])
                            }
                        })
            
            # Sort detections by confidence (highest first)
            detections.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
            
            # Aggregate detection confidences by class
            # Use max confidence per class (or sum, depending on what makes sense)
            # For YOLOv8, we'll use max confidence per class to represent the model's confidence in that class
            class_aggregated = {}
            for det in detections:
                class_name = det['class']  # Already using display name
                conf = det['confidence']
                # Use max confidence for each class (represents best detection of that class)
                if class_name not in class_aggregated:
                    class_aggregated[class_name] = 0.0
                class_aggregated[class_name] = max(class_aggregated[class_name], conf)
            
            print(f"DEBUG: Raw aggregated class confidences (before normalization): {class_aggregated}")
            
            # Now normalize the aggregated class confidences to sum to 100%
            # This distributes the probability across detected classes
            total_class_confidence = sum(class_aggregated.values())
            if total_class_confidence > 0:
                # Normalize so they sum to 1.0 (100%)
                for class_name in class_aggregated:
                    class_aggregated[class_name] = float(class_aggregated[class_name] / total_class_confidence)
            else:
                # If all confidences are 0, distribute equally
                num_classes = len(class_aggregated)
                if num_classes > 0:
                    equal_prob = 1.0 / num_classes
                    for class_name in class_aggregated:
                        class_aggregated[class_name] = float(equal_prob)
            
            # Verify sum is exactly 1.0 (100%)
            total_norm_conf = sum(class_aggregated.values())
            if abs(total_norm_conf - 1.0) > 0.01 and len(class_aggregated) > 0:
                print(f"DEBUG: WARNING - Normalized class confidences sum to {total_norm_conf:.4f}, not 1.0. Re-normalizing...")
                if total_norm_conf > 0:
                    for class_name in class_aggregated:
                        class_aggregated[class_name] = float(class_aggregated[class_name] / total_norm_conf)
            
            print(f"DEBUG: Normalized aggregated class confidences (sum={sum(class_aggregated.values()):.4f}): {class_aggregated}")
            
            # Get ALL predictions from normalized class confidences
            all_predictions = []
            
            if len(class_aggregated) > 0:
                # Sort classes by normalized confidence (highest first)
                sorted_class_aggregated = sorted(class_aggregated.items(), key=lambda x: x[1], reverse=True)
                
                # Create predictions for ALL detected classes
                # These are already normalized to sum to 100%
                for class_name, normalized_conf in sorted_class_aggregated:
                    all_predictions.append({
                        'class': class_name,
                        'probability': float(normalized_conf)
                    })
            else:
                # No detections, use all_class_confidences (all zeros)
                sorted_classes = sorted(all_class_confidences.items(), key=lambda x: x[1], reverse=True)
                for class_name, prob in sorted_classes:
                    all_predictions.append({
                        'class': class_name,
                        'probability': float(prob)
                    })
            
            # Get top 3 predictions for display
            top3_predictions = all_predictions[:3]
            
            all_predictions_debug = [(p['class'], f"{p['probability']*100:.2f}%") for p in all_predictions[:5]]
            top3_debug = [(p['class'], f"{p['probability']*100:.2f}%") for p in top3_predictions]
            print(f"DEBUG: All predictions (sum={sum(p['probability'] for p in all_predictions):.4f}): {all_predictions_debug}")
            print(f"DEBUG: Top 3 predictions: {top3_debug}")
            
            # Ensure first detection matches first prediction
            if len(detections) > 0 and len(all_predictions) > 0:
                first_pred_class = all_predictions[0]['class']
                first_pred_conf = all_predictions[0]['probability']
                
                # Find matching detection
                matching_detection = None
                for det in detections:
                    if det.get('class') == first_pred_class:
                        matching_detection = det
                        break
                
                if matching_detection:
                    if detections[0].get('class') != first_pred_class:
                        detections.remove(matching_detection)
                        detections.insert(0, matching_detection)
                    detections[0]['confidence'] = float(first_pred_conf)
                else:
                    if len(detections) > 0:
                        detections[0]['class'] = first_pred_class
                        detections[0]['confidence'] = float(first_pred_conf)
            
            # Draw bounding boxes
            try:
                result_image = image.copy()
                draw = ImageDraw.Draw(result_image)
                
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
                except:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
                    except:
                        font = ImageFont.load_default()
            except Exception as draw_error:
                print(f"DEBUG: Error creating draw context: {draw_error}")
                result_image = image.copy()  # Use original image if drawing fails
                draw = None
            
            # Draw only the highest confidence detection
            if len(detections) > 0:
                detection = detections[0]
                bbox = detection.get('bbox')
                class_name = detection.get('class', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                if isinstance(bbox, dict) and all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
                    x1 = float(bbox['x1'])
                    y1 = float(bbox['y1'])
                    x2 = float(bbox['x2'])
                    y2 = float(bbox['y2'])
                    
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                    
                    x1 = max(0, min(image.width, x1))
                    y1 = max(0, min(image.height, y1))
                    x2 = max(0, min(image.width, x2))
                    y2 = max(0, min(image.height, y2))
                    
                    if x1 < x2 and y1 < y2:
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                        label = f"{class_name}: {confidence * 100:.2f}%"
                        # Use textsize for compatibility (textbbox is newer)
                        try:
                            # Try textbbox first (Pillow >= 9.2.0)
                            label_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                        except AttributeError:
                            # Fallback to textsize for older Pillow versions
                            text_width, text_height = draw.textsize(label, font=font)
                            label_bbox = (x1, y1 - 25, x1 + text_width, y1 - 25 + text_height)
                        draw.rectangle(label_bbox, fill='red')
                        draw.text((x1, y1 - 25), label, fill='white', font=font)
            
            # Convert to base64
            try:
                img_buffer = io.BytesIO()
                result_image.save(img_buffer, format='JPEG', quality=95)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            except Exception as img_error:
                print(f"DEBUG: Error encoding image: {img_error}")
                import traceback
                traceback.print_exc()
                # Fallback: use original image
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=95)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            inference_time = (time.time() - start_time) * 1000
            
            return jsonify({
                'success': True,
                'detections': detections,
                'top3_predictions': top3_predictions,
                'all_predictions': all_predictions,
                'inference_time': float(inference_time),
                'image': f"data:image/jpeg;base64,{img_base64}",
                'filename': str(file.filename) if file.filename else 'unknown',
                'architecture': 'YOLOv8 Object Detection',
                'model_size': '25.9M parameters',
                'input_size': '640x640',
                'batch_size': 1,
                'used_weight_path': weight_path,
                'inferred_num_classes': int(num_classes),
                'model_type': 'YOLOv8',
                'backbone': 'CSPDarknet',
                'detection_heads': '1',
                'anchors': '3 scales',
                'classes': f"{num_classes} ({class_list})",
                'fitness_score': 'N/A'
            })
            
        except Exception as model_error:
            print(f"DEBUG: Model loading/inference failed: {model_error}")
            import traceback
            traceback_str = traceback.format_exc()
            print(f"FULL TRACEBACK:\n{traceback_str}")
            return jsonify({
                'success': False,
                'error': str(model_error),
                'traceback': traceback_str
            }), 500
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"ERROR in api_detect_yolov8_custom: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({
            'error': f'Detection failed: {str(e)}',
            'error_type': 'detection_failed'
        }), 500

@app.route('/api/detect_yolov3', methods=['POST'])
def api_detect_yolov3():
    """API endpoint for YOLOv3 model detection - PROPER OBJECT DETECTION"""
    import time
    from PIL import Image
    import numpy as np
    import base64
    import io
    import torch
    from torchvision import transforms
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get selected weight
        weight_path = request.form.get('weight_path', '')
        print(f"DEBUG: Received weight_path: '{weight_path}'")
        
        if not weight_path:
            return jsonify({'error': 'No model weight selected'}), 400
        
        # Handle Hub paths - download on demand
        if weight_path.startswith('hub://'):
            downloaded_path = download_model_from_hub(weight_path)
            if downloaded_path:
                weight_path = downloaded_path
            else:
                return jsonify({'error': f'Failed to download model from Hub: {weight_path}'}), 400
        
        # Load image
        image = Image.open(file.stream).convert('RGB')
        
        print(f"DEBUG: Original image size: {image.size}")
        
        # Check if weight file exists
        if not os.path.exists(weight_path):
            print(f"DEBUG: Weight file does not exist: {weight_path}")
            return jsonify({'error': f'Model weight file not found: {weight_path}'}), 400
        
        print(f"DEBUG: Weight file exists, size: {os.path.getsize(weight_path)} bytes")
        
        # Start timing
        start_time = time.time()
        
        # Initialize fitness_score variable (will be set if found)
        fitness_score = None
        
        try:
            # Import the custom model architecture
            import sys
            yolov3_path = os.path.join(BASE_DIR, "apps", "yolov3_custom")
            sys.path.append(yolov3_path)
            try:
                from train_fast_yolov3 import UltraFastYOLOv3Model
            except ImportError as import_error:
                print(f"DEBUG: Failed to import UltraFastYOLOv3Model: {import_error}")
                return jsonify({
                    'success': False,
                    'error': f'Failed to import model: {import_error}'
                }), 500
            
            # Load checkpoint
            ckpt = torch.load(weight_path, map_location='cpu')
            print(f"DEBUG: Custom checkpoint keys: {list(ckpt.keys())}")
            
            # Try to extract fitness score from checkpoint or filename
            if 'fitness' in ckpt:
                fitness_score = float(ckpt.get('fitness', 0.0))
                print(f"DEBUG: Found fitness in checkpoint: {fitness_score}")
            elif 'best_fitness' in ckpt:
                fitness_score = float(ckpt.get('best_fitness', 0.0))
                print(f"DEBUG: Found best_fitness in checkpoint: {fitness_score}")
            else:
                # Try to extract from filename
                import re
                filename = os.path.basename(weight_path)
                fitness_match = re.search(r'fitness_([\d\.]+)', filename)
                if fitness_match:
                    fitness_score = float(fitness_match.group(1))
                    print(f"DEBUG: Extracted fitness from filename: {fitness_score}")
            
            # Get number of classes and class names from checkpoint
            class_names = ckpt.get('class_names', ['object'] * 80)
            if isinstance(class_names, str):
                class_names = [class_names]
            num_classes = len(class_names)
            print(f"DEBUG: Custom YOLOv3 - num_classes: {num_classes}, class_names: {class_names}")
            
            # Create model instance
            model = UltraFastYOLOv3Model(num_classes=num_classes)
            
            # Load weights
            if 'model_state_dict' not in ckpt:
                raise KeyError(f"Checkpoint does not contain 'model_state_dict' key. Available keys: {list(ckpt.keys())}")
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            print(f"DEBUG: Custom YOLOv3 model loaded successfully")
            
            # Preprocess image
            img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = img_transform(image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output = model(img_tensor)  # [1, 3, 5+num_classes, 7, 7]
            
            results = {'output': output, 'model': model, 'class_names': class_names}
            print(f"DEBUG: Inference complete, output shape: {output.shape}")
            
            # Process detection results from custom YOLOv3 output
            detections = []
            all_class_confidences = {}
            
            # Initialize all class confidences to 0
            for class_name in class_names:
                all_class_confidences[class_name] = 0.0
            
            # Process the model output tensor
            output = results['output']
            
            # Convert to numpy if it's a tensor
            if hasattr(output, 'cpu'):
                output = output.cpu().numpy()
            elif hasattr(output, 'numpy'):
                output = output.numpy()
            
            print(f"DEBUG: Output numpy shape: {output.shape}")
            
            # Validate output shape
            if len(output.shape) != 5:
                raise ValueError(f"Expected output shape [1, 3, 5+num_classes, 7, 7], got {output.shape}")
            
            # Extract objectness and class scores
            objectness = output[0, :, 4, :, :]  # [3, 7, 7]
            class_scores = output[0, :, 5:, :, :]  # [3, num_classes, 7, 7]
            
            # Process detections
            confidence_threshold = 0.01
            scale_x = image.width / 224.0
            scale_y = image.height / 224.0
            cell_width = 224.0 / 7.0
            cell_height = 224.0 / 7.0
            
            # Aggregate class scores
            num_anchors = class_scores.shape[0]
            reshaped_class_scores = class_scores.transpose(1, 0, 2, 3).reshape(len(class_names), -1).T
            
            # Apply softmax
            max_scores = np.max(reshaped_class_scores, axis=1, keepdims=True)
            exp_scores = np.exp(reshaped_class_scores - max_scores)
            class_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            # Aggregate probabilities using MAX instead of MEAN
            # This helps capture detections at any position, not just average them out
            # For position-sensitive objects like eggplant, max aggregation is better
            aggregated_class_probs = np.max(class_probs, axis=0)
            
            # Also weight by objectness to give more weight to cells with high objectness
            flat_objectness_for_weighting = objectness.flatten()
            objectness_weights = 1.0 / (1.0 + np.exp(-flat_objectness_for_weighting))  # Sigmoid
            objectness_weights = objectness_weights / (np.sum(objectness_weights) + 1e-8)  # Normalize
            
            # Weighted aggregation: combine max with objectness-weighted mean
            weighted_mean = np.average(class_probs, axis=0, weights=objectness_weights)
            # Use max for primary signal, but blend with weighted mean for robustness
            aggregated_class_probs = 0.7 * aggregated_class_probs + 0.3 * weighted_mean
            
            total_prob = np.sum(aggregated_class_probs)
            if total_prob > 0:
                normalized_class_probs = aggregated_class_probs / total_prob
            else:
                normalized_class_probs = aggregated_class_probs
            
            # Store normalized probabilities
            for i, class_name in enumerate(class_names):
                all_class_confidences[class_name] = float(normalized_class_probs[i])
            
            # Process ALL detections (not just top 10) to catch objects at any position
            # This helps with position-sensitive detections like eggplant
            flat_objectness = objectness.flatten()
            # Process all cells, not just top 10
            all_indices = np.argsort(flat_objectness)[::-1]  # Sort all by objectness
            
            # Lower confidence threshold to catch more detections
            confidence_threshold = 0.001  # Lowered from 0.01 to catch more detections
            
            for idx in all_indices:
                flat_idx = idx
                anchor_idx = flat_idx // (7 * 7)
                pos_in_grid = flat_idx % (7 * 7)
                y_idx = pos_in_grid // 7
                x_idx = pos_in_grid % 7
                
                obj_value = objectness[anchor_idx, y_idx, x_idx]
                confidence = 1.0 / (1.0 + np.exp(-obj_value))
                confidence = float(confidence)
                
                anchor_class_probs = class_probs[flat_idx, :]
                best_class_idx = int(np.argmax(anchor_class_probs))
                best_class_prob = float(anchor_class_probs[best_class_idx])
                
                combined_confidence = confidence * best_class_prob
                
                # Also check if objectness itself is significant (even if class prob is low)
                # This helps detect objects that might be in unusual positions
                objectness_threshold = 0.1  # Lower threshold for objectness
                
                if (combined_confidence > confidence_threshold or confidence > objectness_threshold) and best_class_idx < len(class_names):
                    class_name = class_names[best_class_idx]
                    
                    # Calculate bbox
                    x_center_cell = (x_idx + 0.5) * cell_width
                    y_center_cell = (y_idx + 0.5) * cell_height
                    bbox_width = cell_width * 1.5
                    bbox_height = cell_height * 1.5
                    
                    x1_model = x_center_cell - bbox_width / 2
                    y1_model = y_center_cell - bbox_height / 2
                    x2_model = x_center_cell + bbox_width / 2
                    y2_model = y_center_cell + bbox_height / 2
                    
                    # Scale to actual image dimensions
                    x1 = x1_model * scale_x
                    y1 = y1_model * scale_y
                    x2 = x2_model * scale_x
                    y2 = y2_model * scale_y
                    
                    # Clamp to image bounds
                    x1 = max(0, min(image.width, x1))
                    y1 = max(0, min(image.height, y1))
                    x2 = max(0, min(image.width, x2))
                    y2 = max(0, min(image.height, y2))
                    
                    if x1 < x2 and y1 < y2:
                        detections.append({
                            'class': str(class_name),
                            'confidence': float(combined_confidence),
                            'bbox': {
                                'x1': float(x1), 'y1': float(y1),
                                'x2': float(x2), 'y2': float(y2)
                            }
                        })
            
            # Apply Non-Maximum Suppression (NMS) to filter overlapping detections
            # This helps remove duplicate detections of the same object
            def calculate_iou(bbox1, bbox2):
                """Calculate Intersection over Union (IoU) between two bounding boxes"""
                x1_1, y1_1, x2_1, y2_1 = bbox1['x1'], bbox1['y1'], bbox1['x2'], bbox1['y2']
                x1_2, y1_2, x2_2, y2_2 = bbox2['x1'], bbox2['y1'], bbox2['x2'], bbox2['y2']
                
                # Calculate intersection
                x1_i = max(x1_1, x1_2)
                y1_i = max(y1_1, y1_2)
                x2_i = min(x2_1, x2_2)
                y2_i = min(y2_1, y2_2)
                
                if x2_i <= x1_i or y2_i <= y1_i:
                    return 0.0
                
                intersection = (x2_i - x1_i) * (y2_i - y1_i)
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                union = area1 + area2 - intersection
                
                if union == 0:
                    return 0.0
                return intersection / union
            
            # Sort detections by confidence (highest first)
            detections.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
            
            # Apply NMS: keep detections with highest confidence, remove overlapping ones
            nms_threshold = 0.5  # IoU threshold for NMS
            filtered_detections = []
            for det in detections:
                should_keep = True
                for kept_det in filtered_detections:
                    # Only apply NMS to detections of the same class
                    if det.get('class') == kept_det.get('class'):
                        iou = calculate_iou(det['bbox'], kept_det['bbox'])
                        if iou > nms_threshold:
                            should_keep = False
                            break
                if should_keep:
                    filtered_detections.append(det)
            
            detections = filtered_detections
            
            # Get all predictions
            all_predictions = []
            sorted_classes = sorted(all_class_confidences.items(), key=lambda x: x[1], reverse=True)
            
            for class_name, prob in sorted_classes:
                all_predictions.append({
                    'class': class_name,
                    'probability': float(prob)
                })
            
            # Get top 3 predictions
            top3_predictions = all_predictions[:3]
            
            # Ensure the first detection matches the first prediction (highest confidence class)
            # This ensures consistency between the detection class and top prediction
            if len(detections) > 0 and len(all_predictions) > 0:
                first_pred_class = all_predictions[0]['class']
                first_pred_conf = all_predictions[0]['probability']
                
                print(f"DEBUG: First prediction (highest confidence from all_class_confidences): {first_pred_class} = {first_pred_conf*100:.2f}%")
                
                # Find the detection with the first prediction class
                matching_detection = None
                for det in detections:
                    if det.get('class') == first_pred_class:
                        matching_detection = det
                        break
                
                if matching_detection:
                    # Move matching detection to first position
                    if detections[0].get('class') != first_pred_class:
                        detections.remove(matching_detection)
                        detections.insert(0, matching_detection)
                        print(f"DEBUG: Moved detection with class '{first_pred_class}' to first position")
                    
                    # Update detection confidence to match first prediction's confidence
                    detections[0]['confidence'] = float(first_pred_conf)
                    print(f"DEBUG: Updated first detection confidence to match first prediction: {first_pred_class} = {first_pred_conf*100:.2f}%")
                else:
                    # No detection found for first prediction class, update first detection
                    print(f"DEBUG: No detection found for first prediction class '{first_pred_class}', updating first detection")
                    if len(detections) > 0:
                        detections[0]['class'] = first_pred_class
                        detections[0]['confidence'] = float(first_pred_conf)
                        print(f"DEBUG: Updated first detection to class '{first_pred_class}' with confidence {first_pred_conf*100:.2f}%")
            
            print(f"DEBUG: Final detections count: {len(detections)}")
            print(f"DEBUG: Top 3 predictions: {top3_predictions}")
            if len(detections) > 0:
                print(f"DEBUG: First detection: {detections[0].get('class')} = {detections[0].get('confidence')*100:.2f}%")
            if len(all_predictions) > 0:
                print(f"DEBUG: First prediction: {all_predictions[0]['class']} = {all_predictions[0]['probability']*100:.2f}%")
            
            # Draw bounding boxes
            from PIL import ImageDraw, ImageFont
            result_image = image.copy()
            draw = ImageDraw.Draw(result_image)
            
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Draw only the highest confidence detection
            if len(detections) > 0:
                detection = detections[0]
                bbox = detection.get('bbox')
                class_name = detection.get('class', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                if isinstance(bbox, dict) and all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
                    x1 = float(bbox['x1'])
                    y1 = float(bbox['y1'])
                    x2 = float(bbox['x2'])
                    y2 = float(bbox['y2'])
                    
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                    
                    x1 = max(0, min(image.width, x1))
                    y1 = max(0, min(image.height, y1))
                    x2 = max(0, min(image.width, x2))
                    y2 = max(0, min(image.height, y2))
                    
                    if x1 < x2 and y1 < y2:
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                        label = f"{class_name}: {confidence * 100:.2f}%"
                        # Use textsize for compatibility (textbbox is newer)
                        try:
                            # Try textbbox first (Pillow >= 9.2.0)
                            label_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                        except AttributeError:
                            # Fallback to textsize for older Pillow versions
                            text_width, text_height = draw.textsize(label, font=font)
                            label_bbox = (x1, y1 - 25, x1 + text_width, y1 - 25 + text_height)
                        draw.rectangle(label_bbox, fill='red')
                        draw.text((x1, y1 - 25), label, fill='white', font=font)

            # Convert to base64
            img_buffer = io.BytesIO()
            result_image.save(img_buffer, format='JPEG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            inference_time = (time.time() - start_time) * 1000
            
            # Format fitness score
            fitness_display = 'N/A'
            if fitness_score is not None:
                fitness_display = f"{fitness_score:.4f}"
            
            return jsonify({
                'success': True,
                'detections': detections,
                'top3_predictions': top3_predictions,
                'all_predictions': all_predictions,
                'inference_time': float(inference_time),
                'image': f"data:image/jpeg;base64,{img_base64}",
                'filename': str(file.filename) if file.filename else 'unknown',
                'architecture': 'YOLOv3 Object Detection',
                'model_size': '246M parameters',
                'input_size': '224x224',
                'batch_size': 1,
                'used_weight_path': weight_path,
                'inferred_num_classes': int(num_classes),
                'backbone': 'Darknet-53',
                'detection_heads': '3 scales',
                'anchors': '9 anchors (3 per scale)',
                'fitness_score': fitness_display
            })
            
        except Exception as model_error:
            print(f"DEBUG: Model loading/inference failed: {model_error}")
            import traceback
            traceback_str = traceback.format_exc()
            print(f"FULL TRACEBACK:\n{traceback_str}")
            return jsonify({
                'success': False,
                'error': str(model_error),
                'traceback': traceback_str
            }), 500
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"ERROR in api_detect_yolov3: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({
            'error': f'Detection failed: {str(e)}',
            'error_type': 'detection_failed'
        }), 500

# Map app names to their testimages directories (datasets is sibling to BASE_DIR)
# Lazy evaluation of TESTIMAGES_DIRS to avoid issues if directories don't exist at startup
def get_testimages_dirs():
    """Get testimages directories dictionary - evaluated lazily to avoid startup issues"""
    return {
        'flat_surface_detection': os.path.join(get_datasets_dir(), 'testmages__flatsurface'),
        'fluid_purity_demo': os.path.join(get_datasets_dir(), 'testmages__milkpurity'),
        'dinov3_demo': os.path.join(get_datasets_dir(), 'testmages_dino'),
        'custom_yolov8_demo': os.path.join(get_datasets_dir(), 'testmages__yolov8'),
        'spatiotemporal_detection': os.path.join(get_datasets_dir(), 'testmages_spatiotemporal'),
        'detect_yolov3': os.path.join(get_datasets_dir(), 'testmages__yolov3'),
        'material_detection_head': os.path.join(get_datasets_dir(), 'val_natural_material_detection')
    }

# Keep for backward compatibility but use function instead
TESTIMAGES_DIRS = get_testimages_dirs()

@app.route('/api/list_testimages/<app_name>')
@app.route('/api/list_testimages/<app_name>/<path:subpath>')
def list_testimages(app_name, subpath=''):
    """List files and directories in the testimages directory for a specific app, with optional subdirectory navigation"""
    try:
        testimages_dirs = get_testimages_dirs()
        if app_name not in testimages_dirs:
            return jsonify({'error': f'Unknown app: {app_name}'}), 404
        
        testimages_dir = testimages_dirs[app_name]
        
        # Build the full path to the directory to list
        if subpath:
            # Join subpath to base directory
            current_dir = os.path.join(testimages_dir, subpath)
            # Normalize path to prevent directory traversal attacks
            current_dir = os.path.normpath(current_dir)
            testimages_dir = os.path.normpath(testimages_dir)
            # Security check: ensure we're still within the testimages directory
            if not current_dir.startswith(testimages_dir):
                return jsonify({'error': 'Invalid directory path'}), 403
        else:
            current_dir = testimages_dir
        
        if not os.path.exists(current_dir):
            return jsonify({'error': f'Directory not found: {current_dir}'}), 404
        
        if not os.path.isdir(current_dir):
            return jsonify({'error': f'Path is not a directory: {current_dir}'}), 400
        
        # Get all files and directories
        files = []
        directories = []
        
        # For yolov3_custom, yolov8_custom, dinov3_custom, and spatiotemporal_detection, accept all image formats
        # For other apps, use the original allowed extensions
        if app_name in ['detect_yolov3', 'custom_yolov8_demo', 'dinov3_demo', 'spatiotemporal_detection']:
            # Accept all image formats and common file formats
            allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif',
                                '.heic', '.HEIC', '.PNG', '.JPG', '.JPEG', '.GIF', '.WEBP', '.BMP', '.TIFF', '.TIF',
                                '.sto', '.STO'}
        else:
            # Original allowed extensions for other apps
            allowed_extensions = {'.png', '.jpg', '.jpeg', '.sto', '.heic', '.HEIC', '.PNG', '.JPG', '.JPEG'}
        
        for itemname in os.listdir(current_dir):
            itempath = os.path.join(current_dir, itemname)
            
            if os.path.isdir(itempath):
                # It's a directory
                dir_stat = os.stat(itempath)
                directories.append({
                    'name': itemname,
                    'type': 'directory',
                    'modified': datetime.fromtimestamp(dir_stat.st_mtime).isoformat()
                })
            elif os.path.isfile(itempath):
                # It's a file
                _, ext = os.path.splitext(itemname)
                # For yolov3_custom, yolov8_custom, dinov3_custom, and spatiotemporal_detection, accept any file format
                if app_name in ['detect_yolov3', 'custom_yolov8_demo', 'dinov3_demo', 'spatiotemporal_detection']:
                    # Accept any file format
                    file_stat = os.stat(itempath)
                    files.append({
                        'name': itemname,
                        'type': 'file',
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })
                else:
                    # For other apps, check extension
                    if ext.lower() in [e.lower() for e in allowed_extensions]:
                        file_stat = os.stat(itempath)
                        files.append({
                            'name': itemname,
                            'type': 'file',
                            'size': file_stat.st_size,
                            'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                        })
        
        # Sort: directories first, then files, both alphabetically
        directories.sort(key=lambda x: x['name'].lower())
        files.sort(key=lambda x: x['name'].lower())
        
        # Combine directories and files (directories first)
        items = directories + files
        
        # Calculate relative path for display
        if subpath:
            relative_path = subpath
        else:
            relative_path = ''
        
        return jsonify({
            'success': True,
            'app_name': app_name,
            'directory': current_dir,
            'relative_path': relative_path,
            'items': items,
            'directories': directories,
            'files': files,
            'count': len(items)
        })
    except Exception as e:
        print(f"ERROR listing testimages for {app_name}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_testimage/<app_name>/<path:filename>')
def get_testimage(app_name, filename):
    """Serve a test image file from the testimages directory, supporting subdirectory paths"""
    try:
        testimages_dirs = get_testimages_dirs()
        if app_name not in testimages_dirs:
            return jsonify({'error': f'Unknown app: {app_name}'}), 404
        
        testimages_dir = testimages_dirs[app_name]
        # filename can now include subdirectory paths (e.g., "subdir/image.jpg")
        filepath = os.path.join(testimages_dir, filename)
        
        # Security check: ensure file is within the testimages directory
        filepath = os.path.normpath(filepath)
        testimages_dir = os.path.normpath(testimages_dir)
        if not filepath.startswith(testimages_dir):
            return jsonify({'error': 'Invalid file path'}), 403
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        if not os.path.isfile(filepath):
            return jsonify({'error': 'Path is not a file'}), 400
        
        # Determine MIME type based on file extension
        _, ext = os.path.splitext(filename)
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.heic': 'image/heic',
            '.HEIC': 'image/heic',
            '.sto': 'application/octet-stream',
            '.STO': 'application/octet-stream'
        }
        mime_type = mime_types.get(ext.lower(), 'application/octet-stream')
        
        return send_file(filepath, mimetype=mime_type)
    except Exception as e:
        print(f"ERROR getting testimage for {app_name}/{filename}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def kill_existing_processes_on_port(port):
    """Kill any existing processes running on the specified port (excluding current process)."""
    try:
        current_pid = str(os.getpid())
        # Find processes using the port (works on macOS/Linux)
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            killed_any = False
            for pid in pids:
                if pid and pid != current_pid:
                    try:
                        subprocess.run(['kill', '-9', pid], check=False)
                        print(f"Killed process {pid} on port {port}")
                        killed_any = True
                    except Exception as e:
                        print(f"Error killing process {pid}: {e}")
            if not killed_any:
                print(f"No other processes found on port {port} (current PID: {current_pid})")
        else:
            print(f"No existing processes found on port {port}")
    except FileNotFoundError:
        # lsof might not be available, try alternative method
        try:
            # Alternative: use netstat (if available)
            result = subprocess.run(
                ['netstat', '-anv', '|', 'grep', f':{port}'],
                shell=True,
                capture_output=True,
                text=True
            )
        except Exception as e:
            print(f"Could not check for existing processes: {e}")
    except Exception as e:
        print(f"Error checking for existing processes on port {port}: {e}")

def download_model_from_hub(hub_path):
    """Download a specific model file from Hugging Face Hub on demand
    Args:
        hub_path: Path in format 'hub://repo_id/path/to/file' or just local path
    Returns:
        Local file path if successful, None otherwise
    """
    try:
        from huggingface_hub import hf_hub_download
        import os
        
        # If not a Hub path, return as-is
        if not hub_path.startswith('hub://'):
            return hub_path if os.path.exists(hub_path) else None
        
        # Parse Hub path: hub://repo_id/path/to/file
        parts = hub_path.replace('hub://', '').split('/', 1)
        if len(parts) != 2:
            print(f"Invalid Hub path format: {hub_path}")
            return None
        
        repo_id = parts[0]
        file_path = parts[1]
        
        models_dir = get_models_dir()
        local_file_path = os.path.join(models_dir, file_path)
        
        # If file already exists locally, use it
        if os.path.exists(local_file_path):
            print(f"Using cached model: {local_file_path}")
            return local_file_path
        
        # Download from Hub
        print(f"Downloading {file_path} from {repo_id}...")
        local_dir = os.path.dirname(local_file_path)
        os.makedirs(local_dir, exist_ok=True)
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="model",
            local_dir=models_dir,
            token=os.environ.get("HF_TOKEN"),
            resume_download=True
        )
        print(f"✓ Downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Error downloading model {hub_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import time
    startup_start = time.time()
    
    print("=" * 60)
    print("Starting SPAD for Vision application...")
    print("=" * 60)
    print("Note: Models will be downloaded on-demand when selected from dropdowns")
    print("=" * 60)
    
    # Use port 7860 for Hugging Face Spaces (default), 7889 for local testing
    # Hugging Face Spaces sets PORT=7860, but we check SPACE_ID to be sure
    if os.environ.get("SPACE_ID"):
        # On Hugging Face Spaces, always use 7860
        port = 7860
    else:
        # Local testing, use PORT env var or default to 7889
        port = int(os.environ.get("PORT", 7889))
    print(f"Starting Flask server on port {port}...")
    print(f"SPACE_ID: {os.environ.get('SPACE_ID', 'Not set')}")
    print(f"PORT env var: {os.environ.get('PORT', 'Not set')}")
    
    # Kill any existing processes on the port before starting (for local testing)
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        kill_existing_processes_on_port(port)
    
    startup_time = time.time() - startup_start
    print(f"Startup completed in {startup_time:.2f} seconds")
    print("=" * 60)
    
    # Enable debug mode on Hugging Face Spaces for better visibility
    # Debug mode shows detailed error messages and auto-reloads on code changes
    debug_mode = os.environ.get("SPACE_ID") is not None
    if debug_mode:
        print("Debug mode: ENABLED (Hugging Face Spaces)")
    else:
        print("Debug mode: DISABLED (Local testing)")
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
