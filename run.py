#!/usr/bin/env python3
"""
Machine Vision Plus Web Application
Simple startup script for the Flask application
"""

import os
import sys
import subprocess
import time
import psutil

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import ultralytics
        import cv2
        import numpy
        import requests
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    """Main function to start the application"""
    print("🚀 Starting Machine Vision Plus Web Application...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Kill any existing instances of the app (run.py or app.py) before starting
    try:
        current_pid = os.getpid()
        print("🧹 Checking for existing app instances to terminate...")
        killed = []
        target_ports = set(range(5000, 5011))
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['pid'] == current_pid:
                    continue
                name = (proc.info.get('name') or '').lower()
                cmd  = ' '.join(proc.info.get('cmdline') or []).lower()
                # Match by command line
                matches_cmd = (
                    'flask' in cmd or 'werkzeug' in cmd or 'gunicorn' in cmd or 'uvicorn' in cmd or
                    'app.py' in cmd or 'run.py' in cmd or 'python -m flask' in cmd
                )
                # Match by listening ports
                matches_port = False
                try:
                    for c in proc.connections(kind='inet'):
                        if c.laddr and c.status == psutil.CONN_LISTEN and c.laddr.port in target_ports:
                            matches_port = True
                            break
                except Exception:
                    pass
                if matches_cmd or matches_port:
                    print(f" - Killing PID {proc.info['pid']} -> {cmd}")
                    proc.kill()
                    killed.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        if killed:
            print(f"✅ Killed {len(killed)} process(es): {killed}")
            time.sleep(1.5)
        else:
            print("✅ No existing app instances found")
    except Exception as e:
        print(f"⚠️  Could not pre-kill processes: {e}")

    # Start the Flask application
    print("🌐 Starting Flask server...")
    print("📍 Application will be available at: http://localhost:5002")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Try to set high priority for network/CPU
    try:
        # Set nice value to -10 (higher priority, lower number = higher priority)
        # Note: Going lower than -10 typically requires root privileges
        os.nice(-10)
        print("✅ Process priority set to high (-10)")
    except (OSError, PermissionError):
        # If we can't set high priority (permission denied), try lower priority
        try:
            os.nice(-5)
            print("✅ Process priority set to elevated (-5)")
        except (OSError, PermissionError):
            print("⚠️  Could not set process priority (requires elevated privileges)")
            print("   Run with 'nice -n -10 python3 run.py' or use set_network_priority.sh for full priority")
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5002, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
