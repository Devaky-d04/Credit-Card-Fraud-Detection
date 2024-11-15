# Credit-Card-Fraud-Detection

This project aims to detect fraudulent transactions in credit card data using various machine learning models. The dataset contains anonymized features and is highly imbalanced, making fraud detection challenging.

Requirements
Python 3.9
scikit-learn 0.24.2
pandas 1.3.3
numpy 1.21.2
imbalanced-learn 0.8.0
xgboost 1.5.0
lightgbm 3.2.1
matplotlib 3.4.3
seaborn 0.11.2
How to Run
Data Preprocessing:

Use preprocess_data.py to load and preprocess the dataset, including scaling features and handling class imbalance with SMOTE.
Train Models:

Run train_models.py to train and evaluate different models, including Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM.
Evaluate Results:

Check evaluate_results.py for model performance comparison, precision-recall analysis, and final metrics.
Project Overview
Dataset: 85,443 transactions with 30 features (V1-V28, transaction amount)
Goal: Identify fraudulent transactions in an imbalanced dataset
Best Models: Random Forest and XGBoost demonstrated high accuracy and AUC scores but require further tuning for fraud detection precision.
