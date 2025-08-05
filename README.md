# 📊 Fraud Detection in Mobile Transactions using PaySim Dataset

This project focuses on detecting fraudulent transactions using the **PaySim synthetic dataset**, which simulates mobile money transactions. The goal is to analyze the data, explore patterns, and build machine learning models to accurately identify fraudulent activities.

---
![image alt](https://github.com/Lalit-sharma123/PaySim_fraud_detectiion/blob/main/Screenshot%202025-08-05%20100333.png)
## 📁 Project Structure

- **`final.csv`**  
  A cleaned version of the PaySim dataset used for analysis and modeling.  
  Contains ~500,000 rows (a 1/4 scale of the original dataset of 6.3 million records).

- **`fraud detection mobile transaction.ipynb`**  
  This notebook includes:
  - Exploratory Data Analysis (EDA)
  - Visualizations of transaction types and fraud distribution
  - Insights into fraudulent vs. legitimate transactions

- **`Model.ipynb`**  
  This notebook includes:
  - Data preprocessing and feature engineering
  - Building multiple machine learning models (e.g., Logistic Regression, Random Forest, XGBoost)
  - Evaluation using metrics like accuracy, precision, recall, F1-score, and ROC-AUC

---

## 📌 Dataset Overview

The PaySim dataset simulates mobile money transactions, and includes:
- **Transaction Types**: `CASH_OUT`, `TRANSFER`, `DEBIT`, `PAYMENT`, etc.
- **Important Fields**: `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`

Each row represents a single transaction. The key target variable is:
- **`isFraud`**: Binary class indicating whether the transaction is fraudulent (1) or not (0).

---

## 🚀 Objective

To build a fraud detection system that can:
- Understand and visualize transaction behavior
- Detect fraudulent transactions with high precision
- Compare the performance of different classification algorithms

---

## 📈 ML Techniques Used

- Data cleaning & feature selection
- Handling class imbalance (e.g., under/oversampling)
- Supervised classification models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
- Evaluation metrics:
  - Confusion Matrix
  - ROC-AUC Curve
  - Precision-Recall Curve

---

## 🛠 Tools & Libraries

- Python (Pandas, NumPy)
- Scikit-learn
- Matplotlib, Seaborn
- XGBoost
- Imbalanced-learn (SMOTE, RandomUnderSampler)
- Jupyter Notebook

---

## 📚 Reference

- PaySim Paper: *"PaySim: A financial mobile money simulator for fraud detection"*  
  DOI: [10.1016/j.eswa.2018.03.050](https://doi.org/10.1016/j.eswa.2018.03.050)


