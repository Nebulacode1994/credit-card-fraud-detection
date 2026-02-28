# credit-card-fraud-detection# Credit Card Fraud Detection using Machine Learning

## Overview

This project builds a machine learning model to detect fraudulent credit card transactions using Logistic Regression with class imbalance handling and threshold tuning.

## Dataset

* Source: Kaggle Credit Card Fraud Detection Dataset
* Transactions: 284,807
* Fraud cases: 492 (0.17%)

Dataset is not included due to size. Download from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it in:
data/creditcard.csv

## Features

* Pipeline with StandardScaler and Logistic Regression
* Class imbalance handled using class_weight="balanced"
* Threshold tuning to improve fraud precision
* Model saved using joblib

## Results

Threshold: 0.95

Precision (Fraud): 0.44
Recall (Fraud): 0.87
F1-Score: 0.58

## Project Structure

credit-card-fraud-detection/

src/
model.py

data/
creditcard.csv

fraud_model.pkl

notebooks/

README.md

## Tech Stack

Python
Scikit-learn
Pandas
NumPy
Joblib

## Author

Mohamed Ahmed
AI / Machine Learning Engineer
Berlin, Germany
