# Fraud_Detection_System
This is a system which uses synthetic Creditcard Fraud Dataset in order to create a Fraud detection system. It comprises over 10,000 records and the data has been anonymized to protect the cardholders' identities. The primary objective of this dataset is to facilitate the development of fraud detection algorithms and models to identify potentially fraudulent transactions.
# Requirements:
- ﻿imbalanced-learn==0.13.0
- joblib==1.4.2
- numpy==2.2.1
- pandas==2.2.3
- scikit-learn==1.6.1
- scipy==1.15.1
- sklearn-compat==0.1.3
# Dataset:

The dataset is created synthetically by coding and is  located in the `datasets/` directory as `creditcard_fraud_detection.csv`. After preprocessing, the resampled data is saved as `creditcard_fraud_detection_resampled.csv`.
# Steps for Project:

## 1. Data Preprocessing:
This script handles loading the dataset, balancing the classes using SMOTE, and splitting the data into training and testing sets.
## 2. Model Training:
This script actually trains a Random Forest model to effectively detect fraudulent transactions.The training process includes splitting the data into training and testing sets, fitting the model to the training data, and saving the trained model to the designated directory for further use.
## 3. Model Evaluation:
Evaluate the trained model's performance by calculating precision, recall, and F1-score using this script. This script loads the Random Forest model and test dataset to measure its effectiveness in identifying fraudulent transactions, ensuring accuracy and reliability.
## 4. Testing Interface:
It is used to interact with the fraud detection system by inputting transaction features. This script allows users to provide feature values for a transaction and predicts whether the transaction is fraudulent or not.
# Features:

- **Imbalanced Data Handling**: Utilizes SMOTE for oversampling the minority class.
- **Model Training**: Implements a Random Forest classifier for fraud detection.
- **Evaluation Metrics**: Computes precision, recall, and F1-score for model evaluation.
- **User Interface**: A simple CLI for testing transactions.





