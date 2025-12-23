# Glaucoma Detection Pipeline — Backend Server

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/) [![Install](https://img.shields.io/badge/Install-See%20Installation-blue)](#installation) [![Run Server](https://img.shields.io/badge/Run-See%20Run%20Instructions-brightgreen)](#start-the-inference-server)

A GPU-accelerated deep learning pipeline for **automated glaucoma detection** from retinal fundus images. Includes model training, validation, and a FastAPI/Flask inference server for real-time predictions, designed for seamless integration with Flutter mobile applications.


## 📊 Model Architecture

```
Input Image (224×224×3)
    ↓
EfficientNetB0 Backbone (ImageNet pretrained, 5.3M params)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) → Dropout(0.5)
    ↓
Dense(128, relu) → Dropout(0.3)
    ↓
Dense(1, sigmoid) → Probability [0, 1]
```

**Framework**: TensorFlow/Keras  
**Loss**: Binary crossentropy with class weights  
**Metrics**: Accuracy, AUC, Precision, Recall  
**Optimization**: Adam with learning rate scheduling


## 🛠️ Installation

### Prerequisites

- **Windows/Linux/Mac** with Python 3.9+
- **Git** for version control

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/glaucoma-detection.git
   cd glaucoma-detection
   ```

2. **Create virtual environment:**
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   & .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```


## 🏋️ Training

### Quick Start

Run training with default settings (10 epochs phase 1, 15 epochs phase 2):

```powershell
& .\.venv\Scripts\Activate.ps1

python .\scripts\train_classifier.py `
  --combined_csv .\combined_training_labels.csv `
  --output_dir .\models `
  --batch_size 32 `
  --epochs_phase1 10 `
  --epochs_phase2 15 `
  --val_split 0.2 `
  --learning_rate 1e-4 `
  --use_optimal_threshold
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--combined_csv` | `./combined_training_labels.csv` | Path to training labels CSV |
| `--output_dir` | `./models` | Output directory for checkpoints & plots |
| `--batch_size` | 32 | Training batch size |
| `--epochs_phase1` | 10 | Epochs for phase 1 (frozen backbone) |
| `--epochs_phase2` | 15 | Epochs for phase 2 (fine-tuning) |
| `--val_split` | 0.2 | Validation split ratio |
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--use_optimal_threshold` | False | Use ROC-optimal threshold instead of 0.5 |




### Start the Inference Server

```powershell
& .\.venv\Scripts\Activate.ps1

cd .\server
python appy.py
```

Server starts on `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "decision_threshold": 0.3482
}
```

#### 2. Single Image Prediction
```http
POST /predict
Content-Type: multipart/form-data

file: <image.jpg>
```

**Response:**
```json
{
  "probability": 0.72,
  "decision": "glaucoma",
  "risk_score": 72,
  "average_cdr": null,
  "image_quality": "acceptable",
  "timestamp": "2025-12-23T10:30:45Z",
  "confidence": 0.22
}
```
**Response:**
```json
{
  "predictions": [
    {
      "filename": "image1.jpg",
      "probability": 0.72,
      "decision": "glaucoma",
      "risk_score": 72
    },
    
  ],
  "count": 2
}
```


### Example Client (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@fundus_image.jpg"
```

## 📊 Prediction Output Format

Each prediction includes:

| Field | Type | Description |
|-------|------|-------------|
| `probability` | float | Raw model output [0, 1] |
| `decision` | str | "glaucoma" or "normal" (based on threshold) |
| `risk_score` | int | Probability × 100 (0-100 scale) |
| `average_cdr` | float \| null | Cup-to-disc ratio (if segmentation available) |
| `image_quality` | str | "acceptable", "borderline", "poor" |
| `timestamp` | str | ISO8601 timestamp |
| `confidence` | float | Distance from 0.5 (0 = uncertain, 0.5 = confident) |
