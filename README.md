# 🎵 Music Genre Classification using Deep Learning

## 📘 Overview

This project focuses on classifying music tracks into one of **10 genres** using **deep learning models** and audio features extracted from the **GTZAN Music Genre Dataset**.

Genres:
`Blues`, `Classical`, `Country`, `Disco`, `Hip-Hop`, `Jazz`, `Metal`, `Pop`, `Reggae`, `Rock`

## 🎯 Objective

To identify the most effective neural network architecture and regularization strategy for accurate music genre classification using audio-based features.

## 📊 Dataset

* **Source**: GTZAN Music Genre Dataset
* **Size**: 1,000 audio tracks (30 seconds each)
* **Format**: WAV files
* **Features Extracted**:

  * MFCC (Mel-frequency cepstral coefficients)
  * Chroma features
  * Spectral centroid, bandwidth, rolloff
  * Tempo, Zero Crossing Rate

## 🧠 Models Trained

| Model   | Description                                |
| ------- | ------------------------------------------ |
| Model 1 | Basic Feedforward NN (no regularization)   |
| Model 2 | NN with L2 Regularization + Dropout        |
| Model 3 | NN with L2 + Dropout + Batch Normalization |

> ✅ Best performance: Model 3 (regularized + batch normalization)
> ⚠️ Observed overfitting in the base model

## 🔍 Project Workflow

1. **Preprocessing**

   * Feature extraction using `librosa`
   * Standardization of features

2. **Exploratory Data Analysis (EDA)**

   * Distribution plots, correlation heatmaps, feature importance

3. **Model Development & Tuning**

   * Neural network architectures
   * Regularization (L2, Dropout, BatchNorm)
   * Hyperparameter tuning via KerasTuner / manual tuning

4. **Evaluation**

   * Accuracy, Loss, Confusion Matrix
   * Generalization performance across models

5. **Findings**

   * Regularization improved generalization
   * Batch normalization helped stabilize training

## 🎯 Impact

* Useful for music streaming services, ML practitioners in audio, and researchers.
* Demonstrates the effectiveness of deep learning with handcrafted audio features.

## 🛠 Tech Stack

* Python (NumPy, Pandas)
* Librosa (Audio Feature Extraction)
* Keras / TensorFlow (Model Development)
* Matplotlib / Seaborn (Visualization)
