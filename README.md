# ChurnSignal

Credit card churn classifier with behavioral insight analysis
and automated retention briefing.

## Four Things the Data Shows That You Wouldn't Expect

1. Every customer who contacted the bank 6 times in a year
   churned. Contact frequency is a distress signal, not loyalty.

2. Zero revolving balance customers churn at 3.8x the rate of
   customers carrying any balance. Full-payers are highest risk.

3. Not one customer doing 100+ transactions per year churned.
   Habit is a complete retention shield.

4. Churn risk at 4 months inactive is 7x higher than at 1 month.
   The intervention window closes fast.

## Stack
Python · XGBoost · SHAP · SMOTE · Streamlit · Plotly

## Dataset
BankChurners — 10,127 real customers · Kaggle

## Verified Model Performance
AUC: 0.9914 · Threshold: 0.687 (F1-optimal) ·
Precision: 0.9529 · Recall: 0.8708 · F1: 0.910

## Run
pip install -r requirements.txt
streamlit run app.py
