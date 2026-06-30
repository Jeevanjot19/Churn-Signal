# 📊 ChurnSignal
### Credit Card Churn Prediction & Retention Intelligence System

> Built on **10,127 real credit card customer records**. Predicts which customers are at risk of churning, identifies *why* they're pulling back using SHAP-based behavioral attribution, and generates an automated plain-English retention briefing — no manual reporting required.

---

## The Problem

Credit card companies don't lose customers overnight. Customers quietly use their card less and less — fewer transactions, longer gaps, smaller amounts — until one day they're gone. By then it's too late.

**ChurnSignal catches this before it happens.**

Not just *who* is at risk. But *why* they're pulling back — so the retention team knows exactly what to do for each customer, not send everyone the same offer.

---

## Four Things the Data Showed That Nobody Expected

These aren't model outputs. These are patterns found by actually looking at the data carefully — each one counterintuitive, each one directly actionable.

| # | Finding | Business Implication |
|---|---|---|
| 1 | **Every customer who contacted the bank 6 times in a year churned. 100%. No exceptions.** | Contact frequency is a distress signal, not a loyalty signal. Flag 4+ contacts for immediate relationship manager review. |
| 2 | **Zero revolving balance customers churn at 3.8x the rate of customers carrying any balance.** | Full-payers look like your best customers. They have zero financial lock-in and feel no friction leaving. |
| 3 | **Not one customer doing 100+ transactions per year churned. Zero out of 687.** | Habit is a complete retention shield above a frequency threshold. Acquire customers who will use the card daily, not occasionally. |
| 4 | **Churn risk at 4 months inactive is 7x higher than at 1 month.** | The intervention window closes fast. Trigger outreach at 2 months — not 4. |

---

## Architecture

```
BankChurners.csv (10,127 customers)
        │
        ▼
┌──────────────────┐
│  src/pipeline.py │  → Clean data, encode categoricals, remove leakage columns
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  src/model.py    │  → XGBoost + SMOTE, experimental threshold via P-R curve,
└────────┬─────────┘    SHAP values per customer
         │
         ▼
┌──────────────────┐
│  src/findings.py │  → Four behavioral findings from raw data
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ src/segmenter.py │  → Critical / Watch / Stable tier assignment + profiles
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  src/briefing.py │  → Deterministic template engine → plain-English briefing
└────────┬─────────┘    (No LLM. No API. No hallucination risk.)
         │
         ▼
┌──────────────────┐    ┌─────────────────────┐
│     app.py       │    │  Power BI Dashboard  │
│  Streamlit App   │    │  (.pbix + DAX)       │
└──────────────────┘    └─────────────────────┘
```

---



---

## Key Design Decisions

**No LLM for the briefing.**
The briefing engine generates plain-English summaries from pre-computed numbers using a deterministic template. Every sentence traces back to a computed value — no hallucination risk, no API dependency, no failure modes during a live demo. Computation and narration are fully separated.

**Threshold found experimentally.**
Rather than hardcoding 0.40 or 0.50, the classification threshold was determined by plotting the full precision-recall curve and selecting the F1-optimal point. This is a data-driven decision documented in `src/model.py`.

**Leakage columns removed first.**
The BankChurners dataset contains two columns (`Naive_Bayes_Classifier_*`) that are outputs of a model pre-trained on the churn label. Including them inflates AUC artificially to ~99.9%. These are dropped before any modelling.

**SHAP over feature importance.**
Global feature importance tells you which columns matter to the model overall. SHAP gives per-customer attribution — this specific customer is high risk primarily because of inactivity, not transaction count. That distinction changes the intervention.

---

## Dashboard — Streamlit

**Live Demo:** `[your streamlit URL here]`

What the dashboard shows:
- KPI cards: Critical / Watch / Stable customer counts
- Auto-generated Monday morning retention briefing
- Contacts distress curve — visual of the 6-contacts finding
- Zero balance risk comparison
- Transaction frequency shield
- Inactivity acceleration curve
- Customer risk table — each customer with top SHAP driver and recommended action

---

## Dashboard — Power BI

**File:** `ChurnSignal.pbix`

Built with:
- DAX measure for dynamic churn rate computation (updates with every slicer interaction)
- Cross-filtered slicers by Card Category and Income Segment
- KPI cards per risk tier
- All four behavioral findings as interactive visuals

For non-technical stakeholders — explore the full customer risk landscape without touching any code.

---

## Dataset

**BankChurners** — publicly available on Kaggle  
`https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers`

10,127 customers · 21 features · Binary churn label  
16.1% churned · 83.9% retained

**What it is:** A 12-month behavioral snapshot per customer with a churn label. This is a classification problem — the model learns what a churned customer's behavioral profile looks like and scores current customers against that profile.

**What it is not:** Time series forecasting. There are no timestamps in the dataset. In production, this classifier would be retrained monthly and scored against new transaction data.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python · Pandas · NumPy |
| Machine Learning | XGBoost · Scikit-learn · SMOTE (imbalanced-learn) |
| Explainability | SHAP (TreeExplainer) |
| BI Dashboard | Power BI · DAX |
| Operational Dashboard | Streamlit · Plotly |
| Language | Python 3.11 |

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/churnsignal.git
cd churnsignal

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place BankChurners.csv in data/ folder
# https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

# Run dashboard
streamlit run app.py
```

First run trains and saves the model (~60 seconds).  
Subsequent runs load the saved model and complete in ~5 seconds.

---

## Project Structure

```
churnsignal/
├── data/
│   └── BankChurners.csv
├── models/
│   └── churn_model.pkl
├── src/
│   ├── __init__.py
│   ├── pipeline.py       ← data loading, cleaning, encoding
│   ├── model.py          ← XGBoost training, threshold selection, SHAP
│   ├── findings.py       ← four behavioral findings
│   ├── segmenter.py      ← risk tier assignment + profiles
│   └── briefing.py       ← automated briefing engine
├── app.py                ← Streamlit dashboard
├── config.py             ← all parameters in one place
├── ChurnSignal.pbix      ← Power BI dashboard
├── requirements.txt
└── README.md
```

---

## Verified Findings — Every Number Checked

All four findings were computed on the real dataset before being written anywhere.

```
Contacts = 6  →  Churn rate: 100.0%  (n=54)
Contacts = 5  →  Churn rate: 33.5%
Contacts = 0  →  Churn rate: 1.8%

Zero revolving balance   →  Churn rate: 36.2%  (n=2,470)
Non-zero revolving bal   →  Churn rate: 9.6%   (n=7,657)
Ratio: 3.8x

Transactions >= 100      →  Churn rate: 0.0%   (n=687)
Transactions < 50        →  Churn rate: 38.0%  (n=3,037)

Inactivity 1 month       →  Churn rate: 4.5%
Inactivity 4 months      →  Churn rate: 29.9%
Ratio: 6.6x
```

---

## Limitations — Stated Honestly

- **No timestamps:** Dataset is a static 12-month snapshot. Weekly scoring is simulated via train-test split. Production version would use real weekly transaction exports.
- **Revenue estimate:** Uses credit limit as a CLV proxy with a 15% annual revenue assumption. Stated explicitly in the dashboard — not claimed as fact.
- **0-months inactive bucket:** Only 29 customers. Excluded from charts. Statistically too small to report.

---

