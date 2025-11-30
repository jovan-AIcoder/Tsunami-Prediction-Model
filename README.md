# Tsunami Prediction Using Neural Networks

A deep learning tensorflow model for binary tsunami classification based on earthquake catalog parameters.
This repository contains a complete end-to-end pipeline including preprocessing, scaling, model training, evaluation, and ROC visualization.

---

## 1. Overview

This project explores whether tsunami occurrence can be predicted directly from earthquake parameters using a deep neural network.
The workflow includes:

* Dataset ingestion (`tsunami.csv`)
* Feature preprocessing and scaling
* Model training with Swish activation
* Saving the trained model and scaler
* Evaluation using classification metrics and ROC–AUC
* Visual output (ROC curve)

The model achieves strong performance (AUC ≈ 0.88), indicating meaningful discriminative power even with minimal catalog-based features.

---

## 2. Folder Structure

```
TSUNAMI PREDICTION/
│
├── .venv/                # Python virtual environment
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── tsunami.csv           # Dataset (input)
├── tsunami_model.h5      # Saved Keras model
├── scaler.pkl            # Saved Scaler
└── roc_curve.png         # ROC curve plot
```

---

## 3. Dataset 

The dataset `tsunami.csv` is downloaded from Kaggle

---

## 4. Model Architecture

The neural network uses:

* **Swish activation** for all hidden layers
* **HeNormal initialization**
* **Adamax optimizer** (stable for wide-range gradients)
* **Binary cross-entropy loss**




---

## 5. Training the Model

Run:

```bash
python train.py
```

This script performs:

1. Loading CSV
2. Feature/label separation
3. Train–test split (20% test)
4. Data scaling using StandardScaler
5. Model training (150 epochs)
6. Saving:

   * `tsunami_model.h5`
   * `scaler.pkl`

---

## 6. Evaluating the Model

Run:

```bash
python evaluate.py
```

This script loads the trained model and scaler, then produces:

* Predictions on the test set
* Accuracy, precision, recall, F1-score
* Confusion matrix
* ROC–AUC score
* ROC curve (`roc_curve.png`)

---

## 7. Example Evaluation Results

(Sample results based on current dataset)

* **Accuracy**: 0.81
* **Precision (class 1)**: 0.72
* **Recall (class 1)**: 0.85
* **F1-score (class 1)**: 0.78
* **ROC–AUC**: 0.88

These values indicate strong signal detection capability despite noisy geophysical data.

---







## 8. Future Work

Potential extensions:

* Incorporate waveform-based features (seismograms)
* Time–series modeling (RNN / Transformer)
* Physics-informed constraints (rupture depth thresholds, energy release bounds)
* Regionalization of predictions
* Real-time inference using streaming seismic feeds

---

## 9. License

This project is released under the MIT License.

---

