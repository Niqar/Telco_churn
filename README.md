### Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the root directory as `telco_churn.csv`

---

# 📡 Telco Customer Churn Predictor

A machine learning project that predicts whether a telecom customer will churn, with an interactive Streamlit web application for real-time predictions.

---

## 📌 Project Overview

Customer churn is one of the most critical challenges in the telecom industry — acquiring a new customer costs approximately **5× more** than retaining an existing one. This project builds a classification model to identify at-risk customers before they leave, enabling targeted retention efforts.

**Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
7,032 customers · 21 features · Binary target: `Churn` (Yes / No)

---

## 🗂️ Repository Structure

```
├── telco_churn.ipynb        # Full ML pipeline (EDA → training → evaluation)
├── app.py                   # Streamlit web application
├── best_pipeline_small.pkl  # Saved model (8-feature pipeline)
├── telco_churn.csv          # Dataset (download from Kaggle)
└── README.md
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Inspected all 21 columns for nulls, data types, and distributions
- Converted `TotalCharges` from object to numeric (11 blank rows filled with median)
- Visualized churn rate, numerical distributions, and categorical breakdowns
- Confirmed no outliers in `Tenure`, `MonthlyCharges`, or `TotalCharges` via IQR analysis

### 2. Feature Engineering
- Renamed `tenure` → `Tenure` for consistency
- Simplified redundant categories: `"No internet service"` and `"No phone service"` → `"No"` across 7 columns
- Encoded binary Yes/No columns as 0/1

### 3. Train / Test Split
- 80/20 stratified split → **5,625 train** / **1,407 test** samples

### 4. Model Comparison

Three models were evaluated using 5-fold stratified cross-validation:

| Model | CV AUC (mean) | CV AUC (std) | Test AUC | F1 | Recall |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.8453** | 0.0056 | **0.8358** | **0.610** | **0.791** |
| Gradient Boosting | 0.8441 | 0.0073 | 0.8336 | 0.431 | 0.316 |
| Random Forest | 0.8360 | 0.0058 | 0.8282 | 0.600 | 0.789 |

✅ **Logistic Regression** selected as best model — highest Test AUC and best recall for the churn class.

### 5. Hyperparameter Tuning

GridSearchCV (5-fold, scoring = ROC-AUC) on Logistic Regression:

| Parameter | Best Value |
|---|---|
| C | 10 |
| penalty | L1 |
| solver | liblinear |

| Metric | Before Tuning | After Tuning |
|---|---|---|
| Test AUC | 0.8358 | 0.8346 |
| F1 (Churn) | 0.610 | 0.610 |
| Recall (Churn) | 0.791 | 0.800 |

> Tuning produced marginal gains — the model had largely converged.

### 6. Threshold Analysis

| | Threshold 0.5 | Threshold 0.72 |
|---|---|---|
| False Positives | 300 | 121 |
| False Negatives | 78 | 155 |

The default **threshold of 0.5** is used in production. Missing a churner (false negative) is more costly than a false alarm in the telecom context.

### 7. Feature Selection for App

Top 10 features by L1 coefficient magnitude were mapped back to 8 original columns:

| Feature | Coefficient |
|---|---|
| InternetService (No) | −2.15 |
| InternetService (Fiber optic) | +2.09 |
| MonthlyCharges | −1.64 |
| Contract (Two year) | −1.49 |
| Tenure | −1.29 |
| Contract (One year) | −0.80 |
| StreamingTV | +0.77 |
| StreamingMovies | +0.75 |

A lightweight pipeline (`best_pipeline_small.pkl`) was retrained on these 8 features for the Streamlit app.

---

## 🖥️ Streamlit App

The app allows users to input customer details and receive an instant churn risk prediction.

### Input Features
- **Tenure** (months) — slider
- **Monthly Charges** ($) — slider
- **Total Charges** ($) — slider
- **Internet Service** — DSL / Fiber optic / No
- **Contract Type** — Month-to-month / One year / Two year
- **Multiple Lines** — Yes / No
- **Streaming TV** — Yes / No
- **Streaming Movies** — Yes / No

### Output
- 🔴 **High churn risk** (≥ 50%) — shown in red
- 🟢 **Low churn risk** (< 50%) — shown in green

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit pandas scikit-learn joblib
```

### Run the App

```bash
streamlit run app.py
```

> ⚠️ Make sure `best_pipeline_small.pkl` is in the same directory as `app.py`.

### Run the Notebook

Open `telco_churn.ipynb` in Jupyter or VS Code. The notebook will generate `best_pipeline_small.pkl` when run end-to-end.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | ML pipeline, preprocessing, modeling |
| matplotlib / seaborn | Visualization |
| joblib | Model serialization |
| Streamlit | Web application |

---

## 📊 Final Model Performance

```
Test AUC : 0.8346
Accuracy : 73%

              precision  recall  f1-score
No (Stay)       0.90     0.70      0.79
Yes (Churn)     0.49     0.80      0.61
```

The model prioritizes **recall for the churn class** (0.80) — catching as many churners as possible — which aligns with the business goal of proactive retention.
