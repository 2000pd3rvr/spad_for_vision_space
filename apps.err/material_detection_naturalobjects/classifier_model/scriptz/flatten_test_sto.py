#!/usr/bin/env python3
"""
Script to move all PNG files from subfolders of test_sto directly into test_sto folder.
This flattens the directory structure and logs all actions.
"""

import os
import shutil
from pathlib import Path

def find_png_files(directory):
    """Find all PNG files in directory and subdirectories"""
    png_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    return png_files

def flatten_test_sto():
    """Move all PNG files from subfolders to test_sto root"""
    test_sto_dir = "test_sto"
    
    print("=" * 80)
    print("FLATTENING TEST_STO DIRECTORY - VERBOSE MODE")
    print("=" * 80)
    
    if not os.path.exists(test_sto_dir):
        print(f"Error: {test_sto_dir} directory not found!")
        return
    
    # Find all PNG files in subdirectories
    print(f"\n1. Scanning for PNG files in {test_sto_dir} subdirectories...")
    png_files = find_png_files(test_sto_dir)
    
    # Filter out files already in the root directory
    root_png_files = []
    subdir_png_files = []
    
    for file_path in png_files:
        relative_path = os.path.relpath(file_path, test_sto_dir)
        if os.path.dirname(relative_path) == '':
            # File is already in root directory
            root_png_files.append(file_path)
        else:
            # File is in a subdirectory
            subdir_png_files.append(file_path)
    
    print(f"   Found {len(png_files)} total PNG files")
    print(f"   Already in root: {len(root_png_files)} files")
    print(f"   In subdirectories: {len(subdir_png_files)} files")
    
    if not subdir_png_files:
        print("\n2. No PNG files found in subdirectories - nothing to move")
        return
    
    print(f"\n2. Moving {len(subdir_png_files)} PNG files to root directory...")
    print("-" * 60)
    
    moved_count = 0
    failed_count = 0
    renamed_count = 0
    
    for i, file_path in enumerate(subdir_png_files, 1):
        filename = os.path.basename(file_path)
        destination = os.path.join(test_sto_dir, filename)
        
        print(f"[{i:4d}/{len(subdir_png_files)}] Moving: {filename}")
        print(f"         From: {file_path}")
        print(f"         To:   {destination}")
        
        try:
            # Check if destination already exists
            if os.path.exists(destination):
                # Generate unique filename
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(destination):
                    new_filename = f"{base}_{counter}{ext}"
                    destination = os.path.join(test_sto_dir, new_filename)
                    counter += 1
                print(f"         Renamed to: {os.path.basename(destination)} (conflict resolved)")
                renamed_count += 1
            
            # Move the file
            shutil.move(file_path, destination)
            moved_count += 1
            print(f"         Status: ✓ MOVED")
            
        except Exception as e:
            failed_count += 1
            print(f"         Status: ✗ FAILED - {e}")
        
        print()
    
    print("-" * 60)
    print(f"MOVEMENT COMPLETE:")
    print(f"  Successfully moved: {moved_count} files")
    print(f"  Renamed due to conflicts: {renamed_count} files")
    print(f"  Failed to move: {failed_count} files")
    print(f"  Total processed: {len(subdir_png_files)} files")
    
    # Clean up empty subdirectories
    print(f"\n3. Cleaning up empty subdirectories...")
    cleaned_dirs = 0
    
    for root, dirs, files in os.walk(test_sto_dir, topdown=False):
        # Skip the root directory itself
        if root == test_sto_dir:
            continue
            
        # Check if directory is empty
        if not os.listdir(root):
            try:
                os.rmdir(root)
                print(f"   Removed empty directory: {root}")
                cleaned_dirs += 1
            except Exception as e:
                print(f"   Failed to remove {root}: {e}")
    
    print(f"   Cleaned up {cleaned_dirs} empty directories")
    
    print("\n" + "=" * 80)
    print("FLATTENING COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    flatten_test_sto()
