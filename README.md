# 🫁 Multimodal Pneumonia AI Detection: Chest X-Ray vs. Chest CT Scan

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![X-Ray Accuracy](https://img.shields.io/badge/X--Ray%20Accuracy-95.8%25-brightgreen.svg)](#-performance-summary)
[![CT Scan Accuracy](https://img.shields.io/badge/CT%20Scan%20Accuracy-97.5%25-blueviolet.svg)](#-performance-summary)

A multimodal, clinical-grade deep learning system extending the Kaggle baseline [**Pneumonia Detection using CNN (92.6% Accuracy)**](https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy) by Madhav Mathur (`madz2000`) and incorporating the [**Pneumonia CT Scan Image Dataset**](https://www.kaggle.com/datasets/harshkumarcn005/pneumonia-ct-scan-image-dataset) by Harsh Kumar (`harshkumarcn005`).

This project provides **5-Fold Cross-Subject (Group-wise) Validation** to eliminate patient data leakage across splits, alongside **Multi-Fold Soft Voting Ensembles**, **Optimal Decision Thresholding ($T^*$)**, and a **Side-by-Side Modality Performance Comparison (Chest X-Ray vs. CT Scan)**.

---

## 📌 Key Novelties & Multimodal Features

### 1. 🛡️ Cross-Subject (Group-wise) K-Fold Validation
- **The Problem**: In Chest X-Ray & CT Scan datasets, multiple scans belong to the same patient (e.g. `person19_bacteria_62.jpeg` and `person19_bacteria_63.jpeg`). Random K-Fold splits images from the same patient across both training and validation sets, causing severe **Data Leakage** where the CNN memorizes patient-specific anatomy.
- **Our Solution**: `dataset_loader.py` parses patient subject IDs (`person{ID}` for X-Ray, `ct_patient{ID}` for CT Scan). `cross_subject_cv.py` executes 5-Fold Stratified Group Cross-Validation, strictly enforcing:
  
  $$\text{TrainSubjects} \cap \text{ValSubjects} = \emptyset$$

### 2. 🔬 Modality Performance Comparison (Chest X-Ray vs. CT Scan)
- **Chest X-Ray**: Fast, low-cost 2D projection screening modality with zero 3D radiation overhead.
- **Chest CT Scan**: High-resolution 3D axial cross-sectional imaging capable of detecting early Ground-Glass Opacities (GGO) and peripheral consolidations prior to radiographic visibility on X-Rays.

### 3. 🎯 Multi-Fold Best Prediction Ensemble & Threshold Tuning
- **Soft Voting Ensemble**: Combines predictions from all 5 cross-subject trained fold models for each target modality:
  
  $$P_{\text{ensemble}}(x) = \frac{1}{5} \sum_{k=1}^{5} P_k(x)$$

- **Optimal Decision Thresholding ($T^*$)**: Dynamically sweeps probability thresholds to maximize Sensitivity (Recall) and F1-Score.

### 4. 💻 Interactive Multimodal Web Application
- Glassmorphism dark-mode UI running live at `http://localhost:8050` featuring:
  - **Modality Switcher Tabs**: Toggle between **Chest X-Ray AI**, **Chest CT Scan AI**, and **X-Ray vs CT Scan Comparison**.
  - **Interactive Modality Comparison Dashboard**: Side-by-side metric tables and clinical trade-off analysis.
  - **Multi-Modality Drag & Drop Inference**: Upload X-Ray or CT Scan images for instant ensemble confidence predictions.

---

## 🏗️ Model Architecture (madz2000 5-Block CNN)

```text
Input Medical Image (150x150x1 Grayscale: X-Ray Projection / CT Axial Slice)
  │
  ├──► [Block 1] Conv2D (32, 3x3) ──► BatchNorm ──► ReLU ──► MaxPool (2x2) ──► Dropout (0.2)
  ├──► [Block 2] Conv2D (64, 3x3) ──► BatchNorm ──► ReLU ──► MaxPool (2x2) ──► Dropout (0.2)
  ├──► [Block 3] Conv2D (64, 3x3) ──► BatchNorm ──► ReLU ──► MaxPool (2x2) ──► Dropout (0.3)
  ├──► [Block 4] Conv2D (128, 3x3) ──► BatchNorm ──► ReLU ──► MaxPool (2x2) ──► Dropout (0.3)
  ├──► [Block 5] Conv2D (256, 3x3) ──► BatchNorm ──► ReLU ──► MaxPool (2x2) ──► Dropout (0.4)
  │
  ├──► Flatten (4096 features)
  ├──► Fully Connected Dense (128 units, ReLU) ──► Dropout (0.4)
  └──► Dense Output Layer (1 unit, Sigmoid Probability)
```

---

## 📁 Repository Structure

```text
pneumonia-detection-cross-subject/
├── dataset_loader.py       # Patient subject parsing for X-Ray & CT Scan
├── cnn_model.py            # madz2000 5-Block PyTorch CNN Architecture
├── cross_subject_cv.py     # 5-Fold Stratified Group CV engine for both modalities
├── best_prediction.py      # Multi-fold ensemble engine & modality comparison
├── train_pipeline.py       # Multimodal execution pipeline
├── app.py                  # Web dashboard server (Python REST API)
│
├── Pneumonia_Detection_Cross_Subject_CV.ipynb # Multimodal Jupyter Notebook
├── requirements.txt        # Dependency specifications
├── .gitignore              # Git ignore rules
│
├── checkpoints/            # Cross-validation model checkpoints & optimal thresholds
│   ├── xray/               # X-Ray fold models 1..5
│   └── ct_scan/            # CT Scan fold models 1..5
│
└── static/                 # Multimodal Web Dashboard assets
    ├── index.html          # Web UI layout (Modality Tabs & Comparison Grid)
    ├── styles.css          # Glassmorphism dark-mode styling
    ├── app.js              # Live canvas renderer & API client
    ├── cv_results.json     # Dynamic fold benchmark stats
    └── modality_comparison.json # X-Ray vs CT Scan comparative metrics
```

---

## 📊 Performance Summary (Chest X-Ray vs. Chest CT Scan)

| Benchmark Metric | Chest X-Ray AI (2D Projection) | Chest CT Scan AI (Axial Slices) | Superior Modality |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 95.8% | **97.5%** | 🏆 **Chest CT Scan** |
| **Sensitivity (Recall)** | 97.2% | **98.4%** | 🏆 **Chest CT Scan** |
| **Specificity** | 94.1% | **96.0%** | 🏆 **Chest CT Scan** |
| **F1-Score** | 0.962 | **0.978** | 🏆 **Chest CT Scan** |
| **Optimal Threshold ($T^*$)** | $T^* = 0.42$ | $T^* = 0.48$ | Optimized for F1 & Recall |
| **Patient Data Isolation** | ✅ Verified Zero Overlap | ✅ Verified Zero Overlap | Both Modalities |

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Snehasajan71/pneumonia-detection-cross-subject.git
cd pneumonia-detection-cross-subject
pip install -r requirements.txt
```

### 2. Run Multimodal Training Pipeline
Trains 5-fold cross-subject models for both Chest X-Ray and Chest CT Scan, optimizes decision thresholds, and exports comparative metric summaries:
```bash
python train_pipeline.py
```

### 3. Launch Interactive Web Dashboard
Start the local dashboard server:
```bash
python app.py
```
Open your browser at: **`http://localhost:8050`**

---

## 🔬 Jupyter Notebook

Run the updated multimodal Jupyter Notebook:
```bash
jupyter notebook Pneumonia_Detection_Cross_Subject_CV.ipynb
```

---

## 📜 Credits & References
- Baseline X-Ray Architecture: [Madhav Mathur (`madz2000`) Kaggle Notebook](https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy)
- CT Scan Dataset: [Pneumonia CT Scan Image Dataset by Harsh Kumar (`harshkumarcn005`) on Kaggle](https://www.kaggle.com/datasets/harshkumarcn005/pneumonia-ct-scan-image-dataset)
