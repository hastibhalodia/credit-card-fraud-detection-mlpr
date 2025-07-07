# Credit Card Fraud Detection

## Overview

This project implements an end-to-end machine learning pipeline to detect fraudulent credit card transactions using an imbalanced dataset. It covers data loading, exploration, preprocessing, splitting, model training, and systematic evaluation using multiple classification algorithms to identify the best-performing model for fraud detection.

Developed as part of **Machine Learning and Pattern Recognition (MLPR) lab** to gain practical experience in handling real-world imbalanced datasets using Python.

---

## Dataset

The dataset used is the **Credit Card Fraud Detection dataset**, containing transactions made by European cardholders in September 2013.

- **Features:** Numerical values obtained using PCA for confidentiality (V1, V2, ..., V28), along with `Time` and `Amount`.
- **Target:** `Class` (0 = Not Fraud, 1 = Fraud).
- **Imbalance:** The dataset is highly imbalanced, with fraudulent transactions being a small fraction of total transactions.

### Download Dataset

Due to the large size, download the dataset from your Google Drive:

[Download creditcard.csv from Google Drive](https://drive.google.com/file/d/1sw2oxGHDbZt2zmPYlsXndN4ljtQ_Mfmv/view?usp=sharing)

Place the downloaded `creditcard.csv` in your working directory before running the scripts.

---

## Project Workflow

### 1️⃣ Data Loading and Exploration
- Load CSV using pandas.
- Check shape, data types, head, and null values.
- Understand class distribution.

### 2️⃣ Data Preprocessing
- Replace `Class` labels for readability (`0` to `Not Fraud`, `1` to `Fraud`).
- Visualize class distribution using a pie chart.
- Scale numerical features using `StandardScaler` to normalize feature values.

### 3️⃣ Train-Test Split
- Split data into training and testing sets (80%-20%) using stratified sampling to maintain class distribution across splits.

### 4️⃣ Model Training
Train the following models:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Decision Tree

### 5️⃣ Model Evaluation
Evaluate each model using:
- Classification Report (Precision, Recall, F1-Score, Support)
- Confusion Matrix
- Accuracy Score

### 6️⃣ Model Comparison
Compare the performance of all models to identify the best approach for fraud detection on this imbalanced dataset.

---

## Results

- All models were successfully trained and evaluated.
- Insights were gained on how different models handle imbalanced datasets.
- Accuracy, confusion matrices, and classification reports provided a clear comparative analysis to select the best-performing model.

---

## Installation

Ensure you have **Python 3.8+** and install the required libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib imbalanced-learn
