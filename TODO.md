### Project TODOs: Glaucoma CDR + Classification

#### 1. Unified labels (DONE)
- **Goal**: One CSV (`combined_training_labels.csv`) listing all images and labels.
- **Status**: Completed via `scripts/build_unified_labels.py`.
- **Details**:
  - Datasets: `REFUGE`, `ORIGA`, `Drishti-GS`.
  - Columns: `image_id, image_path, label, label_source, dataset, split, vCDR, mask_path, cup_mask_path, disc_mask_path`.
  - Paths validated so that non-empty `image_path` and mask paths exist.

#### 2. Segmentation / CDR pipeline
- **2.1 Classification model (DONE)**
  - Train classifier with `scripts/train_classifier.py` using `combined_training_labels.csv`.
  - Handle imbalance using **class weights only** (no oversampling).
  - Use EfficientNetB0 backbone, two-phase training, lower LR (~`1e-4`), and higher early-stopping patience (10).
  - Save weights to `models/glaucoma_classifier_weights.h5`.
  - Provide single-image prediction via `scripts/predict_single.py`.

- **2.2 CDR computation (PENDING)**
  - For REFUGE, ORIGA, Drishti:
    - Map images to provided masks.
    - Convert masks to cup/disc binary masks.
    - Use `utils.compute_cdr_from_masks` to compute vertical CDR per image.
    - Store/refresh `vCDR` column in `combined_training_labels.csv`.
  - (Optional, time-permitting):
    - Train a segmentation model on all masks.
    - Export to `models/segmentation/saved_model` for use at inference.

#### 3. Classifier training & evaluation (DONE, can be iterated)
- Train with:
  - Input: `combined_training_labels.csv`.
  - Imbalance handling: class weights from train distribution.
  - Monitoring: `val_auc`, accuracy, precision, recall.
- Export artifacts:
  - Weights: `glaucoma_classifier_weights.h5`.
  - Plots: `training_history.png`, `roc_curve.png`, `confusion_matrix.png`.
  - Predictions: `val_predictions.csv`.
- Choose and document a decision threshold:
  - Default `0.5` (higher accuracy, lower glaucoma recall).
  - ROC-optimal threshold (better glaucoma recall, lower accuracy).

#### 4. Inference pipeline: CDR + classification (PARTIAL)
- **Classifier path**:
  - Build model via `build_model` in `scripts/train_classifier.py`.
  - Load `glaucoma_classifier_weights.h5`.
  - For a new image, output probability + label.
- **CDR path**:
  - Prefer: use precomputed `vCDR` from CSV if available.
  - Else: use masks + `compute_cdr_from_masks`, or segmentation model if trained.
- **API integration**:
  - Extend `server/app.py` (FastAPI) to return:
    - `model.probability`, `model.label`, `risk_score`.
    - `measurements.average_cdr` (real or clearly-marked pseudo).

#### 5. Final evaluation & reporting (PARTIAL)
- **Classification**:
  - Summarize:
    - Validation AUC.
    - Accuracy, precision, recall, F1 for glaucoma at chosen threshold.
    - Confusion matrix and ROC curve.
- **CDR**:
  - Plot CDR distributions for each dataset.
  - On datasets with provided CDR (ORIGA/Drishti), compute error/correlation between computed and provided CDR.
- **Documentation**:
  - Describe:
    - Data sources and how labels/CDR were derived.
    - Model architectures (classifier, optional segmenter).
    - Key metrics and trade-offs (accuracy vs glaucoma recall).


