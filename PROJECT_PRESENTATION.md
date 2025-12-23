# Glaucoma Detection System - Technical Presentation

## 1. Project Overview

**Project Title:** Automated Glaucoma Detection Using Deep Learning and Cup-to-Disc Ratio Analysis

**Objective:** Develop an AI-powered system to detect glaucoma from fundus images using:
- Deep learning-based segmentation (U-Net) for optic cup and disc detection
- Vertical Cup-to-Disc Ratio (VCDR) calculation
- Clinical classification based on VCDR thresholds

**Application Domain:** Medical Imaging, Ophthalmology, Computer-Aided Diagnosis

---

## 2. Problem Statement

- **Manual glaucoma screening** is time-consuming and requires expert ophthalmologists
- **Early detection** is crucial to prevent vision loss
- **Cup-to-Disc Ratio (CDR)** is a key clinical indicator for glaucoma
- Need for **automated, accurate, and fast** glaucoma detection system

---

## 3. Technical Architecture

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│         Fundus Images (RGB, Various Sizes)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              PREPROCESSING LAYER                         │
│  • Image Resizing (128×128)                             │
│  • Normalization (0-1 range)                            │
│  • CLAHE (Contrast Limited Adaptive Histogram)          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           SEGMENTATION MODEL (U-Net)                     │
│  • Multi-class Segmentation (3 classes)                 │
│  • Output: Background, Optic Disc, Optic Cup             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            POST-PROCESSING LAYER                         │
│  • Thresholding (0.3 probability)                       │
│  • Morphological Dilation (3×3 kernel)                  │
│  • Binary Mask Generation                                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              VCDR CALCULATION                            │
│  • Vertical Cup Height / Vertical Disc Height           │
│  • Bounding Box Extraction                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           CLINICAL CLASSIFICATION                        │
│  • Normal (VCDR ≤ 0.5)                                   │
│  • Suspicious (0.5 < VCDR ≤ 0.6)                         │
│  • Glaucomatous (VCDR > 0.6)                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              OUTPUT LAYER                                │
│  • VCDR Value                                            │
│  • Classification                                        │
│  • Clinical Trail/Notes                                  │
│  • Visualization Overlay                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Technologies & Tools

### 4.1 Core Technologies

| Category | Technology | Version/Purpose |
|----------|-----------|-----------------|
| **Programming Language** | Python | 3.x |
| **Deep Learning Framework** | TensorFlow / Keras | Model training & inference |
| **Computer Vision** | OpenCV (cv2) | Image processing, dilation |
| **Image Processing** | PIL/Pillow | Image loading & conversion |
| **Numerical Computing** | NumPy | Array operations |
| **Data Handling** | Pandas | CSV metadata management |
| **Machine Learning** | scikit-learn | Data splitting, metrics |
| **Visualization** | Matplotlib | Overlay generation |
| **Web Framework** | Flask | REST API server |
| **GPU Acceleration** | CUDA/cuDNN | NVIDIA RTX 3050 optimization |

### 4.2 Model Architectures

- **U-Net**: Segmentation model for optic cup and disc detection
- **EfficientNetB0**: Classification backbone (for future classification model)

### 4.3 Optimization Techniques

- **Mixed Precision Training**: float16 for reduced VRAM usage
- **Memory Growth**: Dynamic GPU memory allocation
- **Data Augmentation**: Rotation, shifting, zooming, flipping
- **Class Imbalance Handling**: Class weights, oversampling strategies

---

## 5. Model Architecture Details

### 5.1 U-Net Segmentation Model

**Input Shape:** (128, 128, 3) - RGB images resized to 128×128

**Architecture:**
```
Encoder Path (Downsampling):
  - Conv Block 1: 32 filters → MaxPool
  - Conv Block 2: 64 filters → MaxPool
  - Conv Block 3: 128 filters → MaxPool
  - Conv Block 4: 256 filters → MaxPool
  - Bottleneck: 512 filters

Decoder Path (Upsampling):
  - Up Block 1: 256 filters (with skip connection)
  - Up Block 2: 128 filters (with skip connection)
  - Up Block 3: 64 filters (with skip connection)
  - Up Block 4: 32 filters (with skip connection)

Output Layer:
  - Conv2D: 3 classes (Background, Optic Disc, Optic Cup)
  - Activation: Softmax
```

**Key Features:**
- Reduced filter counts (32→64→128→256→512) for 4GB VRAM compatibility
- Batch Normalization after each convolution
- Skip connections for feature preservation
- Dropout (0.3) for regularization

**Training Configuration:**
- **Batch Size:** 1 (optimized for 4GB VRAM)
- **Image Size:** 128×128 (reduced from 256×256)
- **Epochs:** 50 (with early stopping)
- **Optimizer:** Adam (learning rate: 1e-4)
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy, IoU, Dice Coefficient

---

## 6. Data Pipeline

### 6.1 Dataset Sources

- **REFUGE**: Retinal Fundus Glaucoma Challenge dataset
- **ORIGA**: Online Retinal Fundus Image dataset
- **Drishti-GS**: Glaucoma Screening dataset
- **Combined Dataset:** 12,449+ fundus images with segmentation masks

### 6.2 Data Preprocessing

1. **Path Resolution:**
   - Handles nested folder structures
   - Normalizes path separators (Windows/Linux compatibility)

2. **Image Preprocessing:**
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Resize to 128×128
   - Normalize to [0, 1] range

3. **Mask Processing:**
   - Binary thresholding (> 0)
   - Multi-class encoding (Background=0, Disc=1, Cup=2)
   - One-hot encoding for training

### 6.3 Data Augmentation

- Rotation (±15°)
- Horizontal/Vertical shifts
- Zoom (0.9-1.1)
- Horizontal flip
- Brightness adjustment

---

## 7. Implementation Details

### 7.1 Training Pipeline

**Script:** `scripts/train_unet_segmenter.py`

**Process:**
1. Load metadata from CSV
2. Filter images with both OD and OC masks
3. Apply CLAHE preprocessing
4. Create multi-class masks
5. Train/validation split (85/15)
6. Model training with callbacks:
   - Early Stopping (patience: 10)
   - Model Checkpoint (save best weights)
   - Reduce LR on Plateau

**GPU Optimization:**
- Mixed precision (float16)
- Memory limit: 3500MB
- Batch size: 1

### 7.2 Inference Pipeline

**Script:** `scripts/predict_glaucoma_overlay.py`

**Process:**
1. Load model architecture and weights
2. Preprocess input image (resize, normalize)
3. Run segmentation prediction
4. Post-process masks:
   - Threshold at 0.3 (lower than 0.5 for edge detection)
   - Dilation (3×3 kernel, 1 iteration)
5. Calculate VCDR from bounding boxes
6. Classify based on VCDR thresholds
7. Generate visualization overlay

### 7.3 VCDR Calculation

```python
# Vertical Cup-to-Disc Ratio
vcdr = (cup_vertical_height) / (disc_vertical_height)

# Classification:
if vcdr <= 0.5:
    return "Normal"
elif vcdr <= 0.6:
    return "Suspicious"
else:
    return "Glaucomatous"
```

---

## 8. API/Server Implementation

### 8.1 Flask REST API

**File:** `server/app.py`

**Endpoints:**

1. **GET /health** or **GET /**
   - Health check endpoint
   - Returns 200 if server and model are ready
   - Returns 503 if model not loaded

2. **POST /predict_glaucoma**
   - Accepts: Image file (multipart/form-data)
   - Returns: JSON with VCDR, classification, clinical trail

**Response Format:**
```json
{
  "status": "Success",
  "vcdr": 0.643,
  "classification": "Glaucomatous",
  "clinical_trail": "High probability of glaucoma damage."
}
```

### 8.2 API Features

- Model loaded once at startup
- Handles image preprocessing
- Returns structured JSON response
- Error handling for missing files/model

---

## 9. Results & Performance

### 9.1 Model Performance

- **Segmentation Model:** U-Net trained on 3-class segmentation
- **Input Resolution:** 128×128 (optimized for memory)
- **Post-processing:** Threshold 0.3 + Dilation for better mask connectivity

### 9.2 Test Results (Sample)

From 10 random test images:
- **7 Glaucomatous** cases (VCDR > 0.6)
- **2 Suspicious** cases (0.5 < VCDR ≤ 0.6)
- **1 Detection error** (Optic Disc not detected)

**VCDR Range Observed:** 0.55 - 0.864

### 9.3 Clinical Classification Accuracy

Based on VCDR thresholds:
- **Normal:** VCDR ≤ 0.5 → "Low probability of glaucoma"
- **Suspicious:** 0.5 < VCDR ≤ 0.6 → "Requires further clinical examination"
- **Glaucomatous:** VCDR > 0.6 → "High probability of glaucoma damage"

---

## 10. Key Technical Challenges & Solutions

### 10.1 Memory Constraints

**Challenge:** 4GB VRAM on RTX 3050 insufficient for standard U-Net

**Solutions:**
- Reduced image size: 256×256 → 128×128
- Reduced model complexity: Filter counts halved
- Mixed precision training (float16)
- Batch size: 1
- Fixed memory limit: 3500MB

### 10.2 Class Imbalance

**Challenge:** Dataset imbalance (Normal: 1220, Glaucoma: 280)

**Solutions:**
- Class weights in loss function
- Data augmentation
- Oversampling strategies (explored)

### 10.3 Path Resolution

**Challenge:** Inconsistent folder structures across datasets

**Solutions:**
- Robust path normalization
- Multiple path resolution strategies
- Handles nested folders (e.g., `folder/folder/file.png`)

### 10.4 Mask Fragmentation

**Challenge:** Small, fragmented masks at 128×128 resolution

**Solutions:**
- Lower threshold (0.3 instead of 0.5)
- Morphological dilation (3×3 kernel)
- Better edge pixel capture

---

## 11. Project Structure

```
capstone/
├── scripts/
│   ├── train_unet_segmenter.py      # Model training
│   ├── predict_glaucoma_overlay.py  # Inference with visualization
│   ├── predict_glaucoma.py          # Simple inference
│   └── train_classifier.py          # Classification model (future)
├── server/
│   ├── app.py                       # Flask REST API
│   └── sample_client.py             # API test client
├── models/
│   └── segmentation/
│       └── ocod_unet_segmenter.h5   # Trained model weights
├── datasets/
│   ├── full-fundus/                 # Fundus images
│   ├── optic-disc/                  # Disc masks
│   ├── optic-cup/                   # Cup masks
│   └── metadata.csv                 # Dataset metadata
├── outputs/
│   └── predictions/                 # Generated overlays
└── test_random_predictions.py        # Batch testing script
```

---

## 12. Future Enhancements

1. **Classification Model Integration:**
   - Combine segmentation + classification
   - Use EfficientNetB0 for direct glaucoma classification

2. **Model Improvements:**
   - Train on higher resolution (256×256) with more VRAM
   - Ensemble methods
   - Transfer learning from medical imaging models

3. **API Enhancements:**
   - Batch prediction endpoint
   - Image quality validation
   - Result caching

4. **Deployment:**
   - Docker containerization
   - Cloud deployment (AWS, GCP, Azure)
   - Mobile app integration

5. **Evaluation Metrics:**
   - Dice coefficient on test set
   - IoU (Intersection over Union)
   - Correlation with ground truth VCDR

---

## 13. Clinical Significance

- **Automated Screening:** Reduces workload on ophthalmologists
- **Early Detection:** Identifies glaucoma at early stages
- **Standardized Assessment:** Consistent VCDR calculation
- **Accessibility:** Can be deployed in remote areas
- **Cost-Effective:** Reduces need for expert consultation

---

## 14. Conclusion

This project demonstrates:
- **Deep learning** for medical image segmentation
- **Clinical integration** using VCDR thresholds
- **Production-ready API** for real-world deployment
- **Optimization** for resource-constrained environments
- **End-to-end pipeline** from training to inference

**Key Achievement:** Successfully developed a working glaucoma detection system that combines state-of-the-art segmentation with clinical decision-making criteria.

---

## 15. References & Resources

- **Datasets:** REFUGE, ORIGA, Drishti-GS
- **Model Architecture:** U-Net (Ronneberger et al., 2015)
- **Clinical Guidelines:** VCDR thresholds based on ophthalmology standards
- **Frameworks:** TensorFlow, Keras, Flask

---

**Presentation Prepared By:** [Your Name]  
**Date:** December 2025  
**Project:** Glaucoma Detection System

