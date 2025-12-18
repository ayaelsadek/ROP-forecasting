📘 Rate of Penetration (ROP) Prediction — Machine Learning Case Study
Author: Aya Elsadek

📌 Overview
This project presents a complete machine learning workflow for predicting the Rate of Penetration (ROP) in drilling operations using real drilling log data. The notebook includes data preprocessing, feature engineering, model training, hyperparameter optimization, explainability using SHAP, and diagnostic evaluation.

The goal is to build an accurate and interpretable model that captures the nonlinear behavior of drilling parameters and their interactions.

📂 Project Structure
كتابة تعليمات برمجية
├── models/

│   ├── final_rop_model.pkl

│   ├── best_rf.joblib

│   ├── scaler+poly.pkl

│   ├── features.pkl

│   └── comparison_table_after_fixes.csv

├── notebook.ipynb

├── README.md

└── data/ (not included)

🛠️ Technologies Used
Python (Pandas, NumPy)

Scikit‑learn

XGBoost

SHAP

Matplotlib / Seaborn

Joblib

📊 Dataset
The dataset contains 151 samples with the following drilling parameters:

Feature	Description
Depth	Measured depth
WOB	Weight on bit
SURF_RPM	Surface RPM
PHIF	Porosity
VSH	Shale volume
SW	Water saturation
KLOGH	Permeability log
ROP_AVG	Target variable

🔧 Data Preprocessing
The notebook performs several preprocessing steps:

✅ Outlier Handling
IQR-based winsorization applied to all numeric features.

✅ Feature Engineering

Includes both physical and statistical features:

SE (Specific Energy)

MSE (Mechanical Specific Energy)

EFF (Drilling Efficiency)

HHP_est (Hydraulic Horsepower estimate)

Log transform of permeability

Interaction terms (e.g., WOB × RPM)

Rate-of-change features (first differences)

Rolling window features (MA3)

✅ Scaling
StandardScaler

RobustScaler

PolynomialFeatures (degree = 2)

🤖 Models Trained

Several baseline models were trained:

Linear Regression

Ridge Regression

Lasso Regression

Random Forest

XGBoost

Both baseline and tuned versions were evaluated.

🏆 Model Performance
The best-performing model was:

✅ XGBoost (baseline)
RMSE: 0.000871

R²: 0.5825

A comparison table is saved in: models/comparison_table_after_fixes.csv

🔍 Model Explainability (SHAP)
SHAP was used to interpret the XGBoost model:

✅ Global Insights
EFF and EFF² strongly increase ROP

WOB and WOB×RPM interactions reduce ROP

Only a small subset of polynomial features significantly influence predictions

✅ Local Explanation
Waterfall and force plots show how individual features push predictions up or down.

✅ Error Analysis
Two diagnostic plots were generated:

1. Predicted vs Actual ROP
Shows strong positive correlation and good model fit.

2. Residuals vs Depth
Residuals are randomly scattered around zero → no depth‑related bias.

💾 Saved Artifacts
The following files are saved for deployment or reuse:

final_rop_model.pkl — final XGBoost model

scaler+poly.pkl — preprocessing pipeline

features.pkl — feature list

best_rf.joblib — tuned Random Forest

comparison_table_after_fixes.csv — model comparison

🚀 How to Run
Install dependencies:

bash
pip install -r requirements.txt
Load the model:

python
import joblib
model = joblib.load("models/final_rop_model.pkl")
Prepare input features and predict:

python
y_pred = model.predict(X_processed)

📌 Conclusion
This case study demonstrates a full ML pipeline for drilling ROP prediction, including:

Advanced feature engineering

Polynomial expansion

Model tuning

Explainability with SHAP

Diagnostic evaluation

The workflow is reproducible, interpretable, and ready for deployment.
