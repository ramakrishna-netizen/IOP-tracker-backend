import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

# --- CONFIGURATION (aligned with predict_glaucoma_overlay.py) ---
IMAGE_SIZE = 128
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'segmentation', 'ocod_unet_segmenter.h5')
THRESHOLD = 0.3  # probability threshold for masks
DILATION_KERNEL = np.ones((3, 3), np.uint8)
# ---------------------------------------------------------------

# --- 1. Build model and load weights once ---
def load_unet_model():
    try:
        sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))
        from train_unet_segmenter import build_unet  # type: ignore
        model = build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=3)
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model weights not found at {MODEL_PATH}")
            return None
        model.load_weights(MODEL_PATH)
        print(f"Model weights loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"ERROR: Model loading failed. {e}")
        return None
    finally:
        # Clean path insertion
        script_path = os.path.join(ROOT_DIR, 'scripts')
        if script_path in sys.path:
            sys.path.remove(script_path)


MODEL = load_unet_model()

app = Flask(__name__)


# --- Helper functions copied from predict_glaucoma_overlay.py ---
def calculate_cdr(mask_od, mask_oc):
    """Calculates vertical Cup-to-Disc Ratio (VCDR)."""
    if mask_od.ndim == 3:
        mask_od = mask_od.squeeze()
        mask_oc = mask_oc.squeeze()

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
    """Classify VCDR using standard clinical thresholds."""
    if vcdr <= 0.5:
        return "Normal", "Low probability of glaucoma."
    if vcdr <= 0.6:
        return "Suspicious", "Suspicious/Borderline.Requires further clinical examination."
    return "Glaucomatous", "High probability of glaucoma damage."


# --- 2. Image Preprocessing and Prediction Function ---
def preprocess_and_predict(image_file_stream):
    """
    Handles image loading, preprocessing, model prediction, and post-processing.
    Logic aligned with predict_glaucoma_overlay.py (threshold 0.3 + dilation).
    """
    if MODEL is None:
        return {"error": "Prediction model is not available."}, 500

    try:
        # Load image from the file stream using PIL (RGB)
        pil_img = Image.open(image_file_stream.stream).convert("RGB")
        original_img_np = np.array(pil_img)

        # Resize to model input
        img_resized = cv2.resize(original_img_np, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # Normalize and add batch dimension
        img_normalized = img_resized.astype('float32') / 255.0
        input_tensor = np.expand_dims(img_normalized, axis=0)  # (1, 128, 128, 3)

        # Predict
        predicted_masks = MODEL.predict(input_tensor)[0]  # (128,128,3)
        mask_od_prob = predicted_masks[..., 1]
        mask_oc_prob = predicted_masks[..., 2]

        # Post-processing: threshold + dilate
        mask_od_binary = (mask_od_prob > THRESHOLD).astype(np.uint8)
        mask_od_binary = cv2.dilate(mask_od_binary, DILATION_KERNEL, iterations=1)

        mask_oc_binary = (mask_oc_prob > THRESHOLD).astype(np.uint8)
        mask_oc_binary = cv2.dilate(mask_oc_binary, DILATION_KERNEL, iterations=1)

        # VCDR + classification
        vcdr, status = calculate_cdr(mask_od_binary, mask_oc_binary)
        vcdr_class, vcdr_note = classify_vcdr(vcdr)

        result = {
            "status": status,
            "vcdr": float(vcdr),
            "classification": vcdr_class,
            "clinical_trail": vcdr_note,  # Clinical interpretation note
            #"od_pixels": int(np.sum(mask_od_binary)),
            #"oc_pixels": int(np.sum(mask_oc_binary)),
        }
        return result, 200

    except Exception as e:
        return {"error": f"An unexpected error occurred during prediction: {e}"}, 500


# --- 3. Health Check Endpoint ---
@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint to verify server and model status."""
    if MODEL is None:
        return jsonify({
            "status": "unhealthy",
            "message": "Model not loaded",
            "model_available": False
        }), 503  # Service Unavailable
    
    return jsonify({
        "status": "healthy",
        "message": "Server is running and model is loaded",
        "model_available": True,
        "model_path": MODEL_PATH
    }), 200


# --- 4. Prediction Endpoint ---
@app.route('/predict_glaucoma', methods=['POST',"GET"])
def api_predict_glaucoma():
    """Endpoint to receive an image and return the prediction results."""
    
    # Handle GET request for health check (optional, but /health is preferred)
    if request.method == 'GET':
        return health_check()

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request. Ensure the field name is 'file'."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400


    print(file)
    prediction_result, status_code = preprocess_and_predict(file)
    print(jsonify(prediction_result))
    return jsonify(prediction_result), status_code


# --- 4. Run the Application ---
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)