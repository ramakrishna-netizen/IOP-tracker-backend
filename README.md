# Glaucoma Detection\IoP(Intraocular Pressure) Tracking Pipeline — Backend Server

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/) [![Install](https://img.shields.io/badge/Install-See%20Installation-blue)](#installation) [![Run Server](https://img.shields.io/badge/Run-See%20Run%20Instructions-brightgreen)](#start-the-inference-server)

A deep learning pipeline for **automated glaucoma detection** from retinal fundus images. Includes model training, validation, and a FlaskAPI inference server for real-time predictions, designed for seamless integration with Flutter mobile applications.


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

## 🛠️ Installation


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
    "status": "healthy",
    "message": "Server is running and model is loaded",
    "model_available": True,
    "model_path": MODEL_PATH
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
    "classification": "Glaucomatous",
    "Note": "High probability of glaucoma damage.",
    "status": "Success",
    "vcdr": 0.778
}
```



### Example Client (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@fundus_image.jpg"
``` |
