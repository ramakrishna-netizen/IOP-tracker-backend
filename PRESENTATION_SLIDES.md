# Glaucoma Detection System - Presentation Slides

## Slide 1: Title Slide
**Automated Glaucoma Detection System**
Using Deep Learning and Cup-to-Disc Ratio Analysis

[Your Name/Team]
[Date]

---

## Slide 2: Problem Statement
- Manual glaucoma screening is time-consuming
- Early detection crucial to prevent vision loss
- Need for automated, accurate detection
- Cup-to-Disc Ratio (CDR) as key clinical indicator

---

## Slide 3: Solution Overview
**AI-Powered Glaucoma Detection System**
- Deep Learning Segmentation (U-Net)
- Automatic VCDR Calculation
- Clinical Classification
- REST API for Integration

---

## Slide 4: System Architecture
```
Input Image → Preprocessing → U-Net Segmentation → 
Post-processing → VCDR Calculation → Clinical Classification → Output
```

**Key Components:**
- Image Preprocessing (CLAHE, Resize)
- U-Net Model (3-class segmentation)
- VCDR Calculation
- Clinical Decision Making

---

## Slide 5: Technologies Used
**Core Stack:**
- Python 3.x
- TensorFlow/Keras
- OpenCV
- Flask (REST API)

**Hardware:**
- NVIDIA RTX 3050 (6GB VRAM)
- CUDA/cuDNN acceleration

**Optimization:**
- Mixed Precision (float16)
- Memory-efficient architecture

---

## Slide 6: Model Architecture - U-Net
**U-Net Segmentation Model:**
- Input: 128×128 RGB images
- Encoder-Decoder architecture
- 3-class output: Background, Optic Disc, Optic Cup
- Skip connections for feature preservation

**Optimized for 4GB VRAM:**
- Reduced filter counts
- Batch size: 1
- Image size: 128×128

---

## Slide 7: Training Pipeline
**Training Configuration:**
- Dataset: REFUGE + ORIGA + Drishti-GS (12,449+ images)
- Batch Size: 1
- Epochs: 50 (with early stopping)
- Optimizer: Adam (lr: 1e-4)
- Loss: Categorical Crossentropy

**Preprocessing:**
- CLAHE enhancement
- Data augmentation
- Class imbalance handling

---

## Slide 8: VCDR Calculation & Classification
**Vertical Cup-to-Disc Ratio:**
```
VCDR = Cup Vertical Height / Disc Vertical Height
```

**Clinical Classification:**
- **Normal** (VCDR ≤ 0.5): Low probability of glaucoma
- **Suspicious** (0.5 < VCDR ≤ 0.6): Requires further examination
- **Glaucomatous** (VCDR > 0.6): High probability of glaucoma damage

---

## Slide 9: Post-Processing Pipeline
**Mask Refinement:**
1. Probability threshold: 0.3 (captures edge pixels)
2. Morphological dilation (3×3 kernel)
3. Binary mask generation
4. Bounding box extraction
5. VCDR calculation

**Why 0.3 threshold?**
- Better edge detection at 128×128 resolution
- Captures fragmented pixels
- Dilation connects small regions

---

## Slide 10: API Implementation
**Flask REST API:**
- **GET /health**: Server health check
- **POST /predict_glaucoma**: Image prediction

**Response Format:**
```json
{
  "status": "Success",
  "vcdr": 0.643,
  "classification": "Glaucomatous",
  "clinical_trail": "High probability..."
}
```

---

## Slide 11: Results & Performance
**Test Results (10 random images):**
- 7 Glaucomatous cases
- 2 Suspicious cases
- 1 Detection error

**VCDR Range:** 0.55 - 0.864

**Features:**
- Automatic overlay visualization
- Clinical interpretation notes
- Batch prediction capability

---

## Slide 12: Technical Challenges & Solutions
| Challenge | Solution |
|-----------|----------|
| 4GB VRAM limitation | Reduced model size, mixed precision |
| Class imbalance | Class weights, augmentation |
| Path inconsistencies | Robust path resolution |
| Mask fragmentation | Lower threshold + dilation |

---

## Slide 13: Project Structure
```
scripts/          - Training & inference scripts
server/           - Flask API
models/           - Trained weights
datasets/         - Fundus images & masks
outputs/          - Generated overlays
```

**Key Files:**
- `train_unet_segmenter.py`: Model training
- `predict_glaucoma_overlay.py`: Inference with visualization
- `server/app.py`: REST API

---

## Slide 14: Clinical Significance
**Benefits:**
- ✅ Automated screening reduces workload
- ✅ Early detection capability
- ✅ Standardized VCDR assessment
- ✅ Remote area accessibility
- ✅ Cost-effective solution

**Impact:**
- Faster diagnosis
- Consistent evaluation
- Scalable deployment

---

## Slide 15: Future Enhancements
1. **Higher Resolution:** Train on 256×256 with more VRAM
2. **Classification Model:** Direct glaucoma classification
3. **Ensemble Methods:** Combine multiple models
4. **Cloud Deployment:** AWS/GCP/Azure integration
5. **Mobile App:** On-device inference

---

## Slide 16: Demo
**Live Demonstration:**
- Upload fundus image
- Real-time prediction
- VCDR calculation
- Clinical classification
- Visualization overlay

---

## Slide 17: Conclusion
**Key Achievements:**
- ✅ Working glaucoma detection system
- ✅ Clinical decision support
- ✅ Production-ready API
- ✅ Optimized for resource constraints
- ✅ End-to-end pipeline

**Impact:** Automated, accurate, and accessible glaucoma screening

---

## Slide 18: Q&A
**Thank You!**

Questions?

**Contact:**
[Your Contact Information]

**Repository:** [GitHub Link if applicable]

