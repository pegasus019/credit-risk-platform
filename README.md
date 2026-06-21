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

## Explainability

Model explainability was implemented using XGBoost gain-based feature importance to identify the strongest credit-risk drivers contributing to probability-of-default predictions.

The top-risk indicators were used to support interpretability for credit decisioning, including behavioural repayment signals, historical application outcomes, overdue exposure, and credit history patterns.

SHAP explainability was investigated, but excluded from the current version due to compatibility issues between the installed SHAP and XGBoost versions when parsing booster metadata. It is listed as a future improvement.

An Optuna-tuned LightGBM challenger model was also tested. Although tuning marginally improved ROC-AUC, it slightly reduced PR-AUC compared with the baseline LightGBM model. Since PR-AUC is more important for this imbalanced default-risk problem, the baseline LightGBM model was retained as the champion model.

## Probability Calibration

The initial LightGBM model produced strong ranking performance but inflated raw risk scores due to class imbalance weighting. To convert model scores into more reliable Probability of Default (PD) estimates, sigmoid calibration was applied using a separate calibration split.

Calibration preserved ranking performance while significantly improving probability quality.

| Metric              | Raw LightGBM | Calibrated LightGBM |
| ------------------- | -----------: | ------------------: |
| ROC-AUC             |       0.7715 |              0.7715 |
| PR-AUC              |       0.2696 |              0.2696 |
| Brier Score         |       0.1794 |              0.0670 |
| Mean Prediction     |       38.54% |               7.97% |
| Actual Default Rate |        8.07% |               8.07% |

The calibrated model is used to generate borrower-level Probability of Default values for risk segmentation and dashboard reporting.

## Calibration Curve & Reliability Analysis

A calibration curve was added to compare raw LightGBM risk scores against calibrated probability-of-default estimates.

The raw LightGBM model showed inflated probability outputs due to class imbalance weighting. After sigmoid calibration, the model's mean predicted PD aligned closely with the observed default rate while maintaining the same ranking performance.

Calibration improved probability reliability significantly:

- Raw Brier Score: 0.1794
- Calibrated Brier Score: 0.0670
- Raw Mean Prediction: 38.54%
- Calibrated Mean PD: 7.97%
- Actual Default Rate: 8.07%

## Fairness & Bias Analysis

This project evaluates model fairness across multiple demographic and socioeconomic groups:

- Gender (`CODE_GENDER`)
- Education Level (`NAME_EDUCATION_TYPE`)
- Family Status (`NAME_FAMILY_STATUS`)
- Occupation (`OCCUPATION_TYPE`)
- Housing Type (`NAME_HOUSING_TYPE`)

For each group, the following metrics are compared:

- Actual Default Rate
- Average Predicted Probability of Default (PD)
- Predicted Default Rate
- False Positive Rate (FPR)
- False Negative Rate (FNR)

This analysis helps identify potential disparities in credit decisions and demonstrates responsible AI practices in financial risk modeling.

The calibrated model is therefore more suitable for borrower-level PD reporting, risk segmentation, and expected loss estimation.

## Expected Loss Estimation

Expected Loss (EL) was calculated as:

EL = PD × LGD × EAD

where:

PD = Probability of Default
LGD = Loss Given Default
EAD = Exposure at Default

This enables portfolio-level risk estimation.
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