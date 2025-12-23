import os
import argparse
import sys
from pathlib import Path

# --- 1. Version Check ---
print("====================================")
print("1. Library Version Check 🧐")
print("====================================")
try:
    import tensorflow as tf
    print(f"TensorFlow Version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed.")

try:
    import numpy as np
    print(f"NumPy Version: {np.__version__}")
except ImportError:
    print("NumPy not installed.")

try:
    import h5py
    print(f"h5py Version: {h5py.__version__}")
except ImportError:
    print("h5py not installed (required to inspect .h5 files).")

print(f"\nPython Executable: {sys.executable}")
print(f"Script Execution Path: {Path(os.getcwd())}\n")

# --- 2. H5 File Inspection ---

def analyze_h5_file(weights_path: str):
    """Inspects the structure of the H5 weight file."""
    try:
        import h5py
    except ImportError:
        print("h5py is not installed. Cannot inspect the .h5 file structure.")
        return

    print("====================================")
    print(f"2. Inspecting Weights File: {weights_path} 🔬")
    print("====================================")

    if not Path(weights_path).exists():
        print(f"Error: Weights file not found at {weights_path}")
        return

    try:
        with h5py.File(weights_path, 'r') as f:
            print(f"Total Groups/Layers found: {len(f.keys())}")
            print("\nTop-level keys (Potential Layer Names):")
            for i, key in enumerate(f.keys()):
                # List first 10 keys and their object types
                if i < 10:
                    obj = f[key]
                    print(f"  - {key} (Type: {type(obj).__name__})")
                    # For a Keras group, try to look inside
                    if isinstance(obj, h5py.Group) and 'kernel:0' in obj:
                        print(f"    -> Weights found: Kernel shape {obj['kernel:0'].shape}")
                else:
                    print("  ... and more (showing first 10 keys only)")
                    break

            # Check for the Keras V2 attribute which indicates Keras-saved format
            if 'keras_version' in f.attrs:
                print(f"\nKeras Version (from file attrs): {f.attrs['keras_version']}")
            else:
                print("\nWarning: 'keras_version' attribute not found in file.")

    except Exception as e:
        print(f"\nError opening/reading H5 file: {type(e).__name__}: {e}")
        print("This could indicate file corruption or a very old/incompatible H5 format.")

# --- Main execution block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic script for Keras weight loading errors.")
    parser.add_argument(
        "--weights_path",
        default=os.path.join("models", "glaucoma_classifier_weights.h5"),
        help="Path to trained classifier weights (.h5) to analyze.",
    )
    args = parser.parse_args()

    # Run the H5 analysis with the specified weights path
    analyze_h5_file(args.weights_path)
    
    print("\n====================================")
    print("3. Analysis Summary")
    print("====================================")
    print("Actionable Insight: Compare the TensorFlow version reported above with the version used to train the model.")
    print("If they differ, try to **align the versions** to resolve the `ValueError: axes don't match array`.")
    print("Also, check if the names and shapes of the weights listed in Section 2 match the layers of your currently built model.")