#!/usr/bin/env python3
"""Script to check for potential runtime errors before deployment"""

import sys
import os

print("=" * 60)
print("Runtime Error Checker for Hugging Face Space")
print("=" * 60)

errors = []
warnings = []

# Check 1: Python syntax
print("\n1. Checking Python syntax...")
try:
    compile(open('app.py').read(), 'app.py', 'exec')
    print("   ✓ app.py syntax is valid")
except SyntaxError as e:
    errors.append(f"Syntax error in app.py: {e}")
    print(f"   ✗ Syntax error: {e}")

# Check 2: Required files
print("\n2. Checking required files...")
required_files = [
    'app.py',
    'requirements.txt',
    'Dockerfile',
    'templates/base.html',
    'static/css/style.css'
]
for file in required_files:
    if os.path.exists(file):
        print(f"   ✓ {file} exists")
    else:
        errors.append(f"Missing required file: {file}")
        print(f"   ✗ {file} missing")

# Check 3: Required directories
print("\n3. Checking required directories...")
required_dirs = [
    'templates',
    'static',
    'static/css',
    'apps.err/material_detection_naturalobjects'
]
for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"   ✓ {dir_path} exists")
    else:
        errors.append(f"Missing required directory: {dir_path}")
        print(f"   ✗ {dir_path} missing")

# Check 4: Import checks
print("\n4. Checking critical imports...")
try:
    import flask
    print("   ✓ flask imported")
except ImportError as e:
    errors.append(f"Cannot import flask: {e}")
    print(f"   ✗ flask import failed: {e}")

try:
    import torch
    print("   ✓ torch imported")
except ImportError as e:
    warnings.append(f"Cannot import torch (may not be critical at startup): {e}")
    print(f"   ⚠ torch import failed (may be OK): {e}")

# Check 5: Port configuration
print("\n5. Checking port configuration...")
with open('app.py', 'r') as f:
    content = f.read()
    if 'port = 7860' in content or 'port=7860' in content:
        print("   ✓ Port 7860 configured for Hugging Face")
    else:
        warnings.append("Port 7860 may not be configured correctly")
        print("   ⚠ Port configuration may need checking")

# Check 6: SPACE_ID handling
print("\n6. Checking SPACE_ID handling...")
if 'SPACE_ID' in content:
    print("   ✓ SPACE_ID environment variable handling found")
else:
    warnings.append("SPACE_ID handling not found")
    print("   ⚠ SPACE_ID handling not found")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if errors:
    print(f"\n❌ ERRORS FOUND ({len(errors)}):")
    for error in errors:
        print(f"   - {error}")
else:
    print("\n✓ No critical errors found")

if warnings:
    print(f"\n⚠ WARNINGS ({len(warnings)}):")
    for warning in warnings:
        print(f"   - {warning}")
else:
    print("\n✓ No warnings")

print("\n" + "=" * 60)
print("To check runtime errors on Hugging Face:")
print("1. Go to https://huggingface.co/spaces/mvplus/spad_for_vision")
print("2. Click on the 'Logs' tab")
print("3. Look for error messages in the build or runtime logs")
print("=" * 60)

sys.exit(1 if errors else 0)

