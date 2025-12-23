import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. ⚙️ GLOBAL CONFIGURATION & OPTIMIZATION
# ==============================================================================

# **CRITICAL FIX 1: Enable Mixed Precision for VRAM reduction**
# This uses float16 (half-precision) for most operations.
mixed_precision.set_global_policy('mixed_float16')
print("GPU Optimization: Enabled mixed_float16 precision policy.")

# **CRITICAL FIX 2: Aggressive memory reduction for 4GB VRAM**
# Reduced batch size, image size, and model complexity
IMAGE_SIZE = 128  # Reduced from 256 to save memory
BATCH_SIZE = 1   # Reduced from 2 to 1 for maximum memory savings
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
NUM_CLASSES = 3   # 3 classes: Background (0), Optic Disc (1), Optic Cup (2)
IMG_CHANNELS = 3  # RGB images

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(BASE_DIR, "datasets", "metadata.csv")
DATASETS_ROOT = os.path.join(BASE_DIR, "datasets")

# ==============================================================================
# 2. 🧱 U-NET BUILDING BLOCKS
# ==============================================================================

def double_conv_block(inputs, n_filters, kernel_size=3):
    """A standard convolution block used in U-Net"""
    x = Conv2D(n_filters, kernel_size, padding="same", 
               kernel_initializer='he_normal', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = Conv2D(n_filters, kernel_size, padding="same", 
               kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    return x

def downsample_block(inputs, n_filters):
    """Encoder path: Conv block followed by Max Pooling and Dropout"""
    conv = double_conv_block(inputs, n_filters)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    pool = Dropout(0.3)(pool)
    return conv, pool # Return both convolution output (for skip connection) and pool output

def upsample_block(inputs, skip_connection, n_filters):
    """Decoder path: UpSampling, Concatenation, and Conv block"""
    # UpSampling is generally less memory-intensive than Conv2DTranspose
    x = UpSampling2D(size=(2, 2))(inputs)
    
    # The concatenation (skip connection)
    x = concatenate([x, skip_connection])
    x = Dropout(0.3)(x)
    
    x = double_conv_block(x, n_filters)
    return x

# ==============================================================================
# 3. 🧠 THE U-NET MODEL FUNCTION (Using Keras Functional API)
# ==============================================================================

def build_unet(input_shape, num_classes):
    # CRITICAL FIX: Reduced model complexity for 4GB VRAM
    # Reduced filter counts: 64->32, 128->64, 256->128, 512->256, 1024->512
    
    # Input layer uses float32 for stability at the entry point
    inputs = Input(input_shape, dtype=tf.float32) 
    
    # --- ENCODER PATH (Downsampling) - REDUCED FILTERS ---
    c1, p1 = downsample_block(inputs, 32)   # Reduced from 64
    c2, p2 = downsample_block(p1, 64)      # Reduced from 128
    c3, p3 = downsample_block(p2, 128)     # Reduced from 256
    c4, p4 = downsample_block(p3, 256)     # Reduced from 512
    
    # --- BOTTLENECK - REDUCED ---
    bottleneck = double_conv_block(p4, 512)  # Reduced from 1024
    
    # --- DECODER PATH (Upsampling) - REDUCED FILTERS ---
    u4 = upsample_block(bottleneck, c4, 256)  # Reduced from 512
    u3 = upsample_block(u4, c3, 128)           # Reduced from 256
    u2 = upsample_block(u3, c2, 64)           # Reduced from 128
    u1 = upsample_block(u2, c1, 32)           # Reduced from 64
    
    # --- FINAL OUTPUT LAYER ---
    # For multi-class segmentation, use softmax instead of sigmoid
    if num_classes > 1:
        outputs = Conv2D(num_classes, (1, 1), padding="same", activation="softmax",
                         dtype='float32')(u1)
    else:
        outputs = Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid",
                         dtype='float32')(u1)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# ==============================================================================
# 4. 🖼️ DATA GENERATOR (Essential for handling large datasets with low VRAM)
# ==============================================================================

def load_and_preprocess_data(metadata_path, datasets_root, img_height, img_width, num_classes):
    """Load fundus images and create multi-class masks for cup/disc segmentation."""
    from PIL import Image
    from tqdm import tqdm
    
    images = []
    masks = []
    skipped = 0
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return None, None
    
    df = pd.read_csv(metadata_path)
    
    # Filter for instances that have both OC and OD segmentation files
    df_filtered = df[(df['fundus_oc_seg'].notna()) & (df['fundus_od_seg'].notna())].copy()
    
    print(f"Found {len(df_filtered)} instances with both OD and OC masks in metadata.")
    
    def resolve_path(rel_path, datasets_root):
        """Try multiple path resolution strategies for nested folder structures."""
        # Normalize the relative path - handle both / and \ separators
        rel_path = str(rel_path).lstrip('/').lstrip('\\')
        # Replace all separators with OS-specific separator
        rel_path = rel_path.replace('/', os.sep).replace('\\', os.sep)
        
        # Strategy 1: Direct path
        direct_path = os.path.normpath(os.path.join(datasets_root, rel_path))
        if os.path.exists(direct_path):
            return direct_path
        
        # Strategy 2: Nested folder (e.g., full-fundus/full-fundus/file.png)
        # Split on OS separator (should work now after normalization)
        path_parts = [p for p in rel_path.split(os.sep) if p]  # Remove empty parts
        if len(path_parts) >= 1:
            folder = path_parts[0]
            filename = path_parts[-1] if len(path_parts) > 1 else path_parts[0]
            # Try: datasets/folder/folder/filename
            nested_path = os.path.normpath(os.path.join(datasets_root, folder, folder, filename))
            if os.path.exists(nested_path):
                return nested_path
        
        # Return the direct path even if it doesn't exist (will be caught by error handling)
        return direct_path
    
    for index, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Loading Data"):
        # Resolve all paths using the helper function
        fundus_path = resolve_path(row['fundus'], datasets_root)
        oc_mask_path = resolve_path(row['fundus_oc_seg'], datasets_root)
        od_mask_path = resolve_path(row['fundus_od_seg'], datasets_root)
        
        try:
            if not all(os.path.exists(p) for p in [fundus_path, oc_mask_path, od_mask_path]):
                raise FileNotFoundError(f"One or more files missing")
            
            # Load and resize images
            fundus_img = Image.open(fundus_path).convert('RGB').resize((img_width, img_height))
            fundus_img = np.array(fundus_img, dtype=np.float32) / 255.0
            
            oc_mask = Image.open(oc_mask_path).convert('L').resize((img_width, img_height))
            od_mask = Image.open(od_mask_path).convert('L').resize((img_width, img_height))
            
            oc_mask_arr = (np.array(oc_mask) > 0).astype(np.uint8)
            od_mask_arr = (np.array(od_mask) > 0).astype(np.uint8)
            
            # Create 3-class mask: 0=Background, 1=Disc, 2=Cup
            combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            combined_mask[oc_mask_arr == 1] = 2  # Cup
            combined_mask[(od_mask_arr == 1) & (combined_mask == 0)] = 1  # Disc
            
            # Convert to one-hot encoding
            one_hot_mask = tf.keras.utils.to_categorical(combined_mask, num_classes=num_classes)
            
            images.append(fundus_img)
            masks.append(one_hot_mask)
            
        except Exception as e:
            print(f"\n[SKIP] Row {index}: {e}")
            skipped += 1
            continue
    
    print(f"\nSuccessfully loaded {len(images)} images. Skipped {skipped}.")
    
    if len(images) == 0:
        return None, None
    
    return np.array(images), np.array(masks)

# ==============================================================================
# 5. 🏃 TRAINING EXECUTION
# ==============================================================================

def train_unet_segmenter():
    # GPU Memory Setup
    print("--- GPU Memory Configuration ---")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s): {gpus}")
        try:
            MEMORY_LIMIT_MB = 3500  # Reduced from 4096 to leave headroom
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=MEMORY_LIMIT_MB)]
            )
            print(f"Set memory limit: {MEMORY_LIMIT_MB}MB")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Running on CPU.")
    print("--------------------------------\n")
    
    # Load and Prepare Data
    print("Loading data...")
    X, Y = load_and_preprocess_data(
        METADATA_PATH, DATASETS_ROOT,
        IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES
    )
    
    if X is None or Y is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded {len(X)} images. Shape: {X.shape}")
    
    # Split data
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.15, random_state=42
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build and Compile Model
    print("\nBuilding model...")
    model = build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNELS), num_classes=NUM_CLASSES)
    
    # Metrics for multi-class segmentation
    def dice_coeff(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',  # Changed from binary_crossentropy for multi-class
        metrics=['accuracy', dice_coeff]
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    model_save_path = os.path.join(BASE_DIR, 'models', 'segmentation', 'ocod_unet_segmenter.h5')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
                      verbose=1, restore_best_weights=True),
        ModelCheckpoint(model_save_path, monitor='val_loss',
                       save_best_only=True, verbose=1)
    ]
    
    print("\nStarting model training...")
    print(f"Batch size: {BATCH_SIZE}, Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    # Train Model
    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\nTraining completed. Model saved to: {model_save_path}")
    
if __name__ == '__main__':
    train_unet_segmenter()