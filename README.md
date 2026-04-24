# Credit Risk Modelling Platform

End-to-end credit risk analytics project using the Home Credit dataset to predict probability of default (PD) and support borrower risk segmentation.

## Project Overview

This project simulates a real-world credit risk workflow used by banks and fintech lenders. It combines multi-table feature engineering, imbalanced classification, threshold optimisation, and model persistence for reproducible scoring.

## Dataset

The project uses the Home Credit Default Risk dataset.

- Base application records: 307,511 customers
- Default rate: ~8.07%
- Raw behavioural records processed: 17M+
- Final modelling table: 307,511 customers with 140+ engineered features

Raw data is excluded from the repository using `.gitignore`.

## Feature Engineering

Engineered customer-level risk indicators from multiple data sources:

- Bureau credit history
- Previous loan applications
- Instalment payment behaviour
- Late payment ratios
- Overdue exposure
- Refusal and approval rates
- Credit utilisation and repayment behaviour

## Modelling

Models used:

- Logistic Regression baseline
- XGBoost classifier

Evaluation focused on imbalanced classification metrics:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion matrix

Baseline Logistic Regression achieved:

- ROC-AUC: 0.750
- PR-AUC: 0.231

Threshold tuning was applied to improve default detection. At threshold 0.35, the model captured approximately 87% of defaulters.

## Business Relevance

The project demonstrates how credit risk teams can use data analytics and machine learning to:

- Rank borrowers by probability of default
- Identify high-risk applicants
- Support credit review decisions
- Balance false positives and false negatives
- Build reproducible risk-scoring pipelines

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Jupyter Notebook
- Git/GitHub

## Project Structure

```text
credit-risk-platform/
├── data/
│   ├── raw/          # ignored
│   └── processed/    # ignored
├── notebooks/
│   └── 01_eda.ipynb
├── src/
├── sql/
├── outputs/
├── .gitignore
└── README.md