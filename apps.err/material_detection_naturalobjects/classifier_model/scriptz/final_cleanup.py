#!/usr/bin/env python3
"""
Final cleanup script to ensure no common files exist between test_sto and data_consolidated3DLED_unequal_samples.
This script checks for any remaining matches and deletes them from test_sto.
"""

import os
from pathlib import Path

def find_all_files(directory):
    """Find all files in directory and subdirectories"""
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def get_basename(filepath):
    """Get just the filename without path"""
    return os.path.basename(filepath)

def final_cleanup():
    """Remove any remaining duplicate files between test_sto and training data"""
    
    print("=" * 80)
    print("FINAL CLEANUP - ENSURING NO COMMON FILES")
    print("=" * 80)
    
    # Define directories
    training_dir = "data_consolidated3DLED_unequal_samples"
    test_sto_dir = "test_sto"
    
    if not os.path.exists(training_dir):
        print(f"Error: {training_dir} directory not found!")
        return
    
    if not os.path.exists(test_sto_dir):
        print(f"Error: {test_sto_dir} directory not found!")
        return
    
    print(f"\n1. Scanning all files in {training_dir}...")
    training_files = find_all_files(training_dir)
    training_basenames = {get_basename(f) for f in training_files}
    print(f"   Found {len(training_files)} total files in training data")
    print(f"   Unique filenames: {len(training_basenames)}")
    
    print(f"\n2. Scanning all files in {test_sto_dir}...")
    test_sto_files = find_all_files(test_sto_dir)
    print(f"   Found {len(test_sto_files)} total files in test_sto")
    
    print(f"\n3. Finding remaining matches...")
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
        print(f"   Deleting {len(matches)} remaining duplicate files from test_sto...")
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
        
        # Final verification
        print(f"\n6. FINAL VERIFICATION:")
        remaining_test_files = find_all_files(test_sto_dir)
        remaining_matches = 0
        for test_file in remaining_test_files:
            basename = get_basename(test_file)
            if basename in training_basenames:
                remaining_matches += 1
        
        if remaining_matches == 0:
            print(f"   ✓ SUCCESS: No common files found between test_sto and training data")
            print(f"   ✓ test_sto now contains {len(remaining_test_files)} unique files")
        else:
            print(f"   ⚠ WARNING: {remaining_matches} common files still exist")
            
    else:
        print("\n5. No remaining matches found - test_sto is already clean!")
        print("   ✓ SUCCESS: No common files exist between test_sto and training data")
    
    print("\n" + "=" * 80)
    print("FINAL CLEANUP COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    final_cleanup()
