#!/usr/bin/env python3
"""
Script to generate customized CV from files in /Users/pd3rvr/Downloads/chance
"""
import os
import json
from pathlib import Path

def read_files_from_directory(directory):
    """Read all files from the specified directory"""
    files_content = {}
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Directory {directory} does not exist")
        return files_content
    
    for file_path in directory_path.iterdir():
        if file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    files_content[file_path.name] = f.read()
                print(f"Read file: {file_path.name}")
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")
    
    return files_content

def main():
    chance_dir = "/Users/pd3rvr/Downloads/chance"
    files_content = read_files_from_directory(chance_dir)
    
    print(f"\nFound {len(files_content)} files:")
    for filename in files_content.keys():
        print(f"  - {filename}")
    
    # Save the content to a JSON file in the workspace for processing
    output_file = "/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision/cv_source_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(files_content, f, indent=2, ensure_ascii=False)
    
    print(f"\nContent saved to {output_file}")
    return files_content

if __name__ == "__main__":
    main()

