# Hand Activity Recognition using Wearable Sensors

A comprehensive machine learning pipeline for recognizing hand-oriented activities from accelerometer and gyroscope data collected by smartwatches and smartphones.

## Overview

This project implements and evaluates multiple machine learning models for Human Activity Recognition (HAR), specifically focusing on fine-grained hand movements. Using the WISDM-51 dataset with 51 subjects performing 18 different activities, we achieve state-of-the-art classification accuracy through feature engineering and ensemble methods.

## Key Features

- **High Accuracy**: 94.98% accuracy with Random Forest classifier
- **Multiple Models**: Comparison of 7 different ML/DL approaches (Random Forest, XGBoost, Decision Trees, ANN, SVM, AdaBoost, Naive Bayes)
- **Comprehensive Feature Engineering**: 84-dimensional feature vectors combining time-domain and frequency-domain features
- **Signal Processing Pipeline**: Butterworth filtering, windowing strategies, and noise reduction
- **Cross-Subject Validation**: LOSO (Leave-One-Subject-Out) evaluation for real-world generalization

## Activities Recognized

12 hand-oriented activities:
- **Eating**: Soup, chips, pasta, sandwich
- **Sports**: Dribbling basketball, playing catch
- **Daily tasks**: Typing, writing, brushing teeth, folding clothes, clapping, drinking

Plus 6 non-hand activities (walking, jogging, stairs, sitting, standing, kicking)

## Dataset

**WISDM-51 Dataset**
- **Download**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
- **Dataset Description**: [WISDM Dataset Paper](./documents/dataset_description.pdf)
- **Our Project Report**: [Final Report PDF](./documents/final_report.pdf)

**Specifications:**
- 51 participants
- 18 activity classes
- Dual sensors: smartwatch (wrist) + smartphone (pocket)
- 6-axis IMU data: 3-axis accelerometer + 3-axis gyroscope
- Sampling rate: 20 Hz
- Total samples: 15,630,426

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Random Forest | 94.98% | 94.94% | 94.97% |
| XGBoost | 94.27% | 94.22% | 94.25% |
| Decision Tree | 89.38% | 89.33% | 89.37% |
| ANN | 81.34% | 81.28% | 81.38% |
| SVM (RBF) | 74.45% | 74.48% | 74.60% |

## Methodology

1. **Signal Preprocessing**: 4th-order Butterworth low-pass filter (5 Hz cutoff)
2. **Windowing**: 9-second sliding windows (180 samples)
3. **Feature Extraction**:
   - Time-domain: Mean, Standard Deviation, RMS, Zero-Crossing Rate
   - Frequency-domain: Dominant Frequency, Spectral Energy, Spectral Entropy
4. **Model Training**: 80/20 train-test split with subject-independent evaluation

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/hand-activity-recognition.git
cd hand-activity-recognition

# Install dependencies
pip install -r requirements.txt
```

## Usage
```python
# Load and preprocess data
from preprocessing import load_data, apply_filter, create_windows

data = load_data('wisdm-dataset/raw/watch/accel/')
filtered_data = apply_filter(data, cutoff=5, order=4)
windows = create_windows(filtered_data, window_size=180)

# Extract features
from features import extract_features

features = extract_features(windows)

# Train model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```
