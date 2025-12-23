# analyze_datasets.py

import os
import sys

# Define the root of the datasets folder relative to this script
DATASETS_ROOT = 'datasets'

def analyze_directory(path, indent=''):
    """Recursively analyzes and prints the structure and contents of a directory."""
    
    # Get a list of all items in the current directory
    try:
        items = os.listdir(path)
    except FileNotFoundError:
        print(f"\nFATAL: Datasets root not found at '{path}'. Please check your directory structure.")
        return

    # Check the contents and types for this directory
    file_count = len([i for i in items if os.path.isfile(os.path.join(path, i))])
    dir_count = len([i for i in items if os.path.isdir(os.path.join(path, i))])
    
    # Limit to showing the first 5 files and checking file types
    file_list = [i for i in items if os.path.isfile(os.path.join(path, i))][:5]
    
    # Determine the most common file extension for context
    extensions = [os.path.splitext(f)[1].lower() for f in items if os.path.isfile(os.path.join(path, f))]
    ext_counts = {}
    for ext in extensions:
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    
    common_ext = max(ext_counts, key=ext_counts.get) if ext_counts else 'None'
    
    # Print the summary for the current directory
    summary = f"({dir_count} Dirs, {file_count} Files; Content: {file_count}x '{common_ext}' files)"
    print(f"{indent}│\n{indent}├── {os.path.basename(path)}/ {summary}")

    # Recursively call for subdirectories
    for item in sorted(items):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            # Limit depth for readability (e.g., up to 4 levels deep)
            if len(indent) < 12: 
                analyze_directory(item_path, indent + '│   ')
            else:
                print(f"{indent}│   ├── ... (Deep directory skipped)")
                
    # If there are important files at this level, list them
    if file_list:
        file_summary = ", ".join([f"'{f}'" for f in file_list])
        print(f"{indent}│   ├── [Files include: {file_summary}, ...]")

def main():
    """Main function to start the analysis."""
    print("----------------------------------------------------------------------")
    print("ANALYZING DATASET STRUCTURE AND CONTENT")
    print("----------------------------------------------------------------------")
    
    analyze_directory(DATASETS_ROOT)
    
    print("\n----------------------------------------------------------------------")
    print("CONTEXT: This report confirms the existence, number, and primary file types")
    print("for all subdirectories under 'datasets/'. This is key for validating paths.")
    print("----------------------------------------------------------------------")

if __name__ == '__main__':
    # Ensure script is run from the project root
    if not os.path.exists(DATASETS_ROOT):
        print(f"Error: Could not find '{DATASETS_ROOT}' folder. Run this script from the project root directory.")
        sys.exit(1)
    
    main()