import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K # Import Keras backend for compatibility

# --- 1. Cross-Domain Normalization: CLAHE (Task 2.1.1) ---

def apply_clahe(image_array: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) 
    to an RGB image to standardize local contrast across datasets (Domain Shift Fix).
    """
    temp_array = image_array.copy()
    if temp_array.dtype != np.uint8:
        # Scale 0-1 float back to 0-255 for cv2 processing if necessary
        if np.max(temp_array) <= 1.0 + np.finfo(float).eps: 
            temp_array = (temp_array * 255).astype(np.uint8) 
        else:
            temp_array = temp_array.astype(np.uint8)

    # Convert to LAB color space
    lab = cv2.cvtColor(temp_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel with the a and b channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to RGB
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Return as float32 (compatible with TensorFlow models)
    return final_img.astype(np.float32)


# --- 2. Focal Loss (Imbalance Fix) ---

def focal_loss(gamma=2.0, alpha=0.75):
    """
    Focal Loss for binary classification, using TensorFlow/Keras Backend ops.
    Addresses severe class imbalance.
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Clip y_pred to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Calculate cross entropy (CE)
        # Use Keras Backend for stable binary crossentropy
        ce = K.binary_crossentropy(y_true, y_pred)
        
        # Calculate P_t (P_t = p if y=1, 1-p if y=0)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        
        # Calculate Alpha_t
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Total loss: alpha_t * (1 - p_t)^gamma * CE
        loss = alpha_t * tf.math.pow((1 - p_t), gamma) * ce
        
        # Return mean loss
        return tf.reduce_mean(loss)

    return focal_loss_fixed