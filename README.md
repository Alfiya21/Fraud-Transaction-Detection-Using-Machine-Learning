# Fraud Transaction Detection Using Machine Learning

## Overview
This project implements an end-to-end fraud detection system using transactional data. 
Machine learning models are trained to classify transactions as fraudulent or legitimate 
based on behavioral and risk-based features.

## Dataset
The dataset consists of daily transaction records stored as `.pkl` files and includes:
- Transaction timestamp
- Customer ID
- Terminal ID
- Transaction amount
- Fraud label

Fraud scenarios include high-value fraud, terminal compromise, and customer credential leakage.

## Features Engineered
- Time-based features (hour, day, weekend)
- Customer behavior features
- Terminal risk features

## Models Used
- Random Forest Classifier
- XGBoost Classifier

## Evaluation Metrics
- Precision
- Recall
- F1-score
- ROC-AUC

## Results
XGBoost outperformed Random Forest in terms of recall and ROC-AUC, making it more effective for fraud detection on imbalanced data.

## How to Run
1. Place dataset `.pkl` files in the `data/` folder
2. Open `Fraud_Detection.ipynb`
3. Run all cells sequentially

## Author
Alfiya Mulla
