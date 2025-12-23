# utils_cdr.py - Only for CDR computation (no TensorFlow dependencies)

import numpy as np
import cv2

def compute_cdr_from_masks(cup_mask_path: str, disc_mask_path: str) -> float:
    """
    Computes the vertical Cup-to-Disc Ratio (vCDR) from mask file paths.
    """
    try:
        # Load masks as grayscale (0: background, >0: object)
        cup_mask = cv2.imread(cup_mask_path, cv2.IMREAD_GRAYSCALE)
        disc_mask = cv2.imread(disc_mask_path, cv2.IMREAD_GRAYSCALE)
        
        if cup_mask is None or disc_mask is None: return np.nan 

        cup_binary = (cup_mask > 0).astype(np.uint8)
        disc_binary = (disc_mask > 0).astype(np.uint8)

        # Disc Diameter (D)
        disc_coords = np.where(disc_binary)
        if len(disc_coords[0]) == 0: return np.nan
        min_y_disc, max_y_disc = np.min(disc_coords[0]), np.max(disc_coords[0])
        D_vertical = max_y_disc - min_y_disc
        
        # Cup Diameter (C)
        cup_coords = np.where(cup_binary)
        if len(cup_coords[0]) == 0:
            C_vertical = 0
        else:
            min_y_cup, max_y_cup = np.min(cup_coords[0]), np.max(cup_coords[0])
            C_vertical = max_y_cup - min_y_cup

        # vCDR = C / D
        if D_vertical > 0:
            v_cdr = C_vertical / D_vertical
            return np.clip(v_cdr, 0.0, 1.0)
        else:
            return np.nan

    except Exception:
        return np.nan

# NOTE: The other functions (focal_loss, apply_clahe) remain in the original utils.py