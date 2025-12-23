import os
import subprocess
import shutil
from pathlib import Path

# --- Configuration ---
DATASET_ID = "deathtrooper/multichannel-glaucoma-benchmark-dataset"
DATASETS_DIR_NAME = "datasets"
# The path where the script expects the data to land
TARGET_DIR = Path(os.getcwd()) / DATASETS_DIR_NAME

# The Kaggle download command template
DOWNLOAD_CMD = [
    "kaggle", "datasets", "download", 
    "-d", DATASET_ID, 
    "--path", str(TARGET_DIR), # Download to the datasets folder
    "--unzip" # Automatically unzip the files
]

def check_kaggle_setup():
    """Verifies that the Kaggle CLI is installed and configured."""
    print("Checking Kaggle CLI setup...")
    try:
        # Check if 'kaggle' command is available
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.DEVNULL)
        
        # Check for credentials (not foolproof, but a good indicator)
        kaggle_config_dir = Path.home() / '.kaggle'
        if not (kaggle_config_dir / 'kaggle.json').exists():
            print("⚠️ WARNING: 'kaggle.json' not found in your home directory.")
            print("         Please ensure your Kaggle API credentials are set up.")
        
        print("Kaggle CLI is ready.")
        return True
    except subprocess.CalledProcessError:
        print("❌ ERROR: Kaggle CLI not found or failed to run.")
        print("         Please install it: 'pip install kaggle'")
        return False
    except FileNotFoundError:
        print("❌ ERROR: The 'kaggle' command was not found.")
        print("         Please ensure 'pip install kaggle' was successful and it's in your PATH.")
        return False

def download_and_structure_dataset():
    """
    Downloads the dataset and handles the resulting directory structure.
    """
    if not check_kaggle_setup():
        return

    print(f"\nAttempting to download '{DATASET_ID}' to '{TARGET_DIR}'...")
    
    # 1. Create the target directory if it doesn't exist
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Execute the download command
    try:
        subprocess.run(DOWNLOAD_CMD, check=True)
        print("\n✅ Download and Unzipping successful.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR during download: {e}")
        print("    Check your internet connection, API token, and ensure the dataset ID is correct.")
        return
    
    # 3. Clean up and structure (Kaggle sometimes creates extra nested folders)
    # The expected contents are 'full-fundus', 'optic-cup', 'optic-disc', etc.
    # The `--unzip` flag places the contents directly in TARGET_DIR.
    
    expected_subdirs = ["full-fundus", "optic-cup", "optic-disc"]
    
    print("\nVerifying dataset structure...")
    
    all_found = all((TARGET_DIR / subdir).exists() for subdir in expected_subdirs)

    if all_found:
        print("✅ Dataset subfolders verified. Structure is correct.")
    else:
        # This block addresses cases where the downloaded ZIP has a single parent folder
        print("⚠️ WARNING: Expected subfolders not directly found in the root of the download.")
        
        # Try to find the single root folder created by the unzip process
        download_contents = list(TARGET_DIR.iterdir())
        
        # Find a single directory that contains the expected subdirs
        potential_root_folder = None
        for item in download_contents:
            if item.is_dir() and any((item / subdir).exists() for subdir in expected_subdirs):
                potential_root_folder = item
                break
        
        if potential_root_folder:
            print(f"Moving contents from nested folder: '{potential_root_folder.name}'...")
            
            # Move contents up one level
            for item in potential_root_folder.iterdir():
                shutil.move(str(item), str(TARGET_DIR / item.name))
            
            # Remove the empty nested folder
            shutil.rmtree(potential_root_folder)
            print("✅ Directory structure fixed.")
        else:
            print("❌ ERROR: Failed to find and correct nested dataset structure.")
            print("         Please manually inspect the contents of the 'datasets' folder.")


if __name__ == "__main__":
    download_and_structure_dataset()
    print("\n--- Next Step ---")
    print(f"1. Run the U-Net training script: python train_unet_segmenter.py")