import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Configuration (Must match training settings) ---
IMAGE_SIZE = 128  # matches train_unet_segmenter.py
IMG_CHANNELS = 3
NUM_CLASSES = 3  # 0=Background, 1=Optic Disc, 2=Optic Cup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'segmentation', 'ocod_unet_segmenter.h5')
DATASETS_ROOT = os.path.join(BASE_DIR, 'datasets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'predictions')


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def resolve_image_path(image_name, datasets_root):
    """Resolve image path handling nested folder structures."""
    rel_path = f'full-fundus/{image_name}.png'

    direct_path = os.path.normpath(os.path.join(datasets_root, rel_path))
    if os.path.exists(direct_path):
        return direct_path

    nested_path = os.path.normpath(os.path.join(datasets_root, 'full-fundus', 'full-fundus', f'{image_name}.png'))
    if os.path.exists(nested_path):
        return nested_path

    return direct_path


def load_model_and_image(image_name):
    """Builds the model, loads weights, and loads one fundus image."""
    # Ensure the directory containing build_unet is in path for import
    sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
    try:
        from train_unet_segmenter import build_unet
    except ImportError:
        print("Error: Could not import build_unet. Ensure train_unet_segmenter.py is in the scripts folder.")
        return None, None, None, None
    finally:
        sys.path.pop(0)

    try:
        model = build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNELS), num_classes=NUM_CLASSES)
        if os.path.exists(MODEL_PATH):
            model.load_weights(MODEL_PATH)
            print(f"Model weights loaded successfully from: {MODEL_PATH}")
        else:
            print(f"Error: Model file not found at {MODEL_PATH}")
            return None, None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    image_path = resolve_image_path(image_name, DATASETS_ROOT)
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Tried:")
        print(f"  - {os.path.normpath(os.path.join(DATASETS_ROOT, f'full-fundus/{image_name}.png'))}")
        print(f"  - {os.path.normpath(os.path.join(DATASETS_ROOT, 'full-fundus', 'full-fundus', f'{image_name}.png'))}")
        return None, None, None, None

    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error reading image at {image_path}")
        return None, None, None, None
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    resized_img = cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE))
    input_tensor = resized_img / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

    return model, input_tensor, original_img, resized_img


def calculate_cdr(mask_od, mask_oc):
    """Calculates vertical Cup-to-Disc Ratio (VCDR)."""
    if mask_od.ndim == 3:
        mask_od = mask_od.squeeze()
        mask_oc = mask_oc.squeeze()

    # NOTE: The threshold here is 0.5 because the mask passed to this function 
    # has ALREADY been thresholded and dilated in perform_glaucoma_prediction.
    
    coords_od = np.argwhere(mask_od > 0.5)
    if coords_od.size == 0:
        return 0.0, "Error: Optic Disc not detected."
    od_min_row, od_max_row = coords_od[:, 0].min(), coords_od[:, 0].max()
    od_height = od_max_row - od_min_row

    coords_oc = np.argwhere(mask_oc > 0.5)
    if coords_oc.size == 0:
        return 0.0, "Error: Optic Cup not detected. CDR calculation aborted."
    oc_min_row, oc_max_row = coords_oc[:, 0].min(), coords_oc[:, 0].max()
    oc_height = oc_max_row - oc_min_row

    if od_height == 0:
        return 0.0, "Error: Disc height is zero."

    vcdr = oc_height / od_height
    return round(vcdr, 3), "Success"


def classify_vcdr(vcdr):
    """
    Classify VCDR using standard clinical thresholds:
      <= 0.5   : Normal
      0.5-0.6  : Borderline/Suspicious
      > 0.6    : Glaucomatous
    """
    if vcdr <= 0.5:
        return "Normal", "Low probability of glaucoma."
    if vcdr <= 0.6:
        return "Suspicious", "Requires further clinical examination."
    return "Glaucomatous", "High probability of glaucoma damage."


def visualize_masks(original_img, mask_od, mask_oc, image_name, output_dir):
    """
    Overlay OD (green) and OC (red) masks on the original image, draw bounding boxes,
    and save to output_dir/{image_name}_overlay.png.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure binary then resize to original image size
    mask_od_bin = (mask_od > 0).astype(np.uint8)
    mask_oc_bin = (mask_oc > 0).astype(np.uint8)
    h, w = original_img.shape[:2]
    mask_od_bin = cv2.resize(mask_od_bin, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_oc_bin = cv2.resize(mask_oc_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    def bbox(mask):
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return (x_min, y_min, x_max, y_max)

    bbox_od = bbox(mask_od_bin)
    bbox_oc = bbox(mask_oc_bin)

    overlay = original_img.copy()
    overlay[mask_oc_bin == 1] = [255, 0, 0]   # Red for cup
    overlay[mask_od_bin == 1] = [0, 255, 0]   # Green for disc

    blended = cv2.addWeighted(original_img, 0.6, overlay, 0.4, 0)

    if bbox_od:
        cv2.rectangle(blended, (bbox_od[0], bbox_od[1]), (bbox_od[2], bbox_od[3]), (0, 255, 0), 2)
    if bbox_oc:
        cv2.rectangle(blended, (bbox_oc[0], bbox_oc[1]), (bbox_oc[2], bbox_oc[3]), (255, 0, 0), 2)

    out_path = os.path.join(output_dir, f"{image_name}_overlay.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(blended)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Overlay saved to: {out_path}")


def perform_glaucoma_prediction(image_name):
    """Run the prediction pipeline and save visualization."""
    model, input_tensor, original_img, resized_img = load_model_and_image(image_name)
    if model is None or input_tensor is None:
        print("Failed to load model or image. Exiting.")
        return

    print("Running prediction...")
    prediction = model.predict(input_tensor, verbose=0)
    predicted_masks = prediction[0]

    # Indices 1 and 2 correspond to OD and OC in the 3-class output (0=BG, 1=OD, 2=OC)
    mask_od_prob = predicted_masks[..., 1]
    mask_oc_prob = predicted_masks[..., 2]

    # --- FIXES IMPLEMENTED HERE ---
    
    # 1. Lower threshold from 0.5 to 0.3 to capture more edge pixels (Fix A)
    threshold = 0.3
    mask_od_binary = (mask_od_prob > threshold).astype(np.uint8)
    mask_oc_binary = (mask_oc_prob > threshold).astype(np.uint8)
    
    # 2. Apply dilation to connect fragmented pixels (Fix B)
    kernel = np.ones((3,3), np.uint8)
    mask_od_binary = cv2.dilate(mask_od_binary, kernel, iterations=1)
    mask_oc_binary = cv2.dilate(mask_oc_binary, kernel, iterations=1)
    
    # --- END FIXES ---

    vcdr, status = calculate_cdr(mask_od_binary, mask_oc_binary)
    vcdr_class, vcdr_note = classify_vcdr(vcdr)

    visualize_masks(original_img, mask_od_binary, mask_oc_binary, image_name, OUTPUT_DIR)

    print("\n--- Prediction Results ---")
    print(f"Image: {image_name}")
    print(f"Status: {status}")
    print(f"Calculated Vertical Cup-to-Disc Ratio (VCDR): {vcdr}")
    print(f"VCDR Classification: {vcdr_class}")
    print(f"Clinical Trail: {vcdr_note}")
    print(f"OD mask pixels: {np.sum(mask_od_binary)}")
    print(f"OC mask pixels: {np.sum(mask_oc_binary)}")


# ----------------------------------------------------------------------
# Execution Example
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import random
    import os
    
    if len(sys.argv) > 1:
        # If image name provided as argument
        image_name = sys.argv[1]
        perform_glaucoma_prediction(image_name)
    else:
        # Default: test random images to find normal cases
        img_dir = os.path.join(BASE_DIR, 'datasets', 'full-fundus', 'full-fundus')
        if os.path.exists(img_dir):
            all_images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            random.shuffle(all_images)
            selected = all_images[:5]  # Test 5 random images
            
            print(f"\n{'='*60}")
            print("Testing 5 Random Images (Looking for Normal Cases)")
            print(f"{'='*60}\n")
            
            normal_cases = []
            for img_file in selected:
                image_name = os.path.splitext(img_file)[0]
                print(f"\n{'='*60}")
                print(f"Testing: {image_name}")
                print(f"{'='*60}")
                
                try:
                    # Temporarily modify to capture VCDR
                    model, input_tensor, original_img, resized_img = load_model_and_image(image_name)
                    if model is None or input_tensor is None:
                        continue
                    
                    prediction = model.predict(input_tensor, verbose=0)
                    predicted_masks = prediction[0]
                    mask_od_prob = predicted_masks[..., 1]
                    mask_oc_prob = predicted_masks[..., 2]
                    
                    threshold = 0.3
                    mask_od_binary = (mask_od_prob > threshold).astype(np.uint8)
                    mask_oc_binary = (mask_oc_prob > threshold).astype(np.uint8)
                    kernel = np.ones((3,3), np.uint8)
                    mask_od_binary = cv2.dilate(mask_od_binary, kernel, iterations=1)
                    mask_oc_binary = cv2.dilate(mask_oc_binary, kernel, iterations=1)
                    
                    vcdr, status = calculate_cdr(mask_od_binary, mask_oc_binary)
                    vcdr_class, vcdr_note = classify_vcdr(vcdr)
                    
                    visualize_masks(original_img, mask_od_binary, mask_oc_binary, image_name, OUTPUT_DIR)
                    
                    print(f"\n--- Results ---")
                    print(f"VCDR: {vcdr}")
                    print(f"Classification: {vcdr_class}")
                    print(f"Clinical Trail: {vcdr_note}")
                    
                    if vcdr <= 0.5:
                        normal_cases.append((image_name, vcdr, vcdr_class))
                        print(f"✓ NORMAL CASE FOUND! (VCDR <= 0.5)")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            print(f"\n{'='*60}")
            print("Summary:")
            print(f"{'='*60}")
            if normal_cases:
                print(f"\nFound {len(normal_cases)} Normal case(s) (VCDR <= 0.5):")
                for name, vcdr, cls in normal_cases:
                    print(f"  - {name}: VCDR={vcdr}, Class={cls}")
            else:
                print("\nNo normal cases found in this batch. Try running again for different random images.")
        else:
            # Fallback to default example
            example_image_name = 'OIA-ODIR-TEST-OFFLINE-1'
            perform_glaucoma_prediction(example_image_name)