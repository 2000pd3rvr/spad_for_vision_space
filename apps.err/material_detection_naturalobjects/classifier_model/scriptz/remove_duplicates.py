#!/usr/bin/env python3
"""
Remove duplicate PNG files from test_sto that exist in training data
"""
import os
from pathlib import Path

def find_png_files(directory):
    """Find all PNG files in directory and subdirectories"""
    png_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    return png_files

def get_basename(filepath):
    """Get just the filename without path"""
    return os.path.basename(filepath)

def main():
    # Define directories
    training_tr = "data_consolidated3DLED_unequal_samples/tr"
    training_te = "data_consolidated3DLED_unequal_samples/te"
    test_sto_tr = "test_sto/tr"
    test_sto_te = "test_sto/te"
    
    print("=" * 80)
    print("DUPLICATE FILE REMOVAL - VERBOSE MODE")
    print("=" * 80)
    
    print("\n1. Finding PNG files in training data...")
    training_files = find_png_files(training_tr) + find_png_files(training_te)
    training_basenames = {get_basename(f) for f in training_files}
    print(f"   Found {len(training_files)} PNG files in training data")
    print(f"   Training TR: {len(find_png_files(training_tr))} files")
    print(f"   Training TE: {len(find_png_files(training_te))} files")
    
    print("\n2. Finding PNG files in test_sto...")
    test_sto_files = find_png_files(test_sto_tr) + find_png_files(test_sto_te)
    print(f"   Found {len(test_sto_files)} PNG files in test_sto")
    print(f"   Test STO TR: {len(find_png_files(test_sto_tr))} files")
    print(f"   Test STO TE: {len(find_png_files(test_sto_te))} files")
    
    print("\n3. Finding matching filenames...")
    matches = []
    for test_file in test_sto_files:
        basename = get_basename(test_file)
        if basename in training_basenames:
            matches.append(test_file)
            print(f"   MATCH FOUND: {basename}")
            print(f"     Test file: {test_file}")
            # Find the corresponding training file
            for train_file in training_files:
                if get_basename(train_file) == basename:
                    print(f"     Train file: {train_file}")
                    break
            print()
    
    print(f"\n4. SUMMARY:")
    print(f"   Total matches found: {len(matches)}")
    
    if matches:
        print(f"\n5. DELETION PROCESS:")
        print(f"   Deleting {len(matches)} files from test_sto...")
        print("-" * 60)
        
        deleted_count = 0
        failed_count = 0
        
        for i, match in enumerate(matches, 1):
            basename = get_basename(match)
            print(f"[{i:3d}/{len(matches)}] Deleting: {basename}")
            print(f"         Path: {match}")
            
            try:
                os.remove(match)
                deleted_count += 1
                print(f"         Status: ✓ DELETED")
            except Exception as e:
                failed_count += 1
                print(f"         Status: ✗ FAILED - {e}")
            print()
        
        print("-" * 60)
        print(f"DELETION COMPLETE:")
        print(f"  Successfully deleted: {deleted_count} files")
        print(f"  Failed to delete: {failed_count} files")
        print(f"  Total processed: {len(matches)} files")
    else:
        print("\n5. No matching files found - nothing to delete")
    
    print("\n" + "=" * 80)
    print("PROCESS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
