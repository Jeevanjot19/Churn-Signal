# config.py

DATA_PATH       = "data/BankChurners.csv"
MODEL_PATH      = "models/churn_model.pkl"
RANDOM_STATE    = 42

# Threshold found experimentally via precision-recall curve
# F1-optimal on this dataset = 0.678
# See src/model.py find_optimal_threshold() for how this was found
CHURN_THRESHOLD = 0.678

WEEK_LABEL      = "2024-W49"
PREV_WEEK_LABEL = "2024-W48"

CATEGORICAL_COLUMNS = [
    "Gender", "Education_Level", "Marital_Status",
    "Income_Category", "Card_Category"
]

# Tier thresholds set from probability distribution
# >=0.70: top ~14.6% of scored customers → Critical
# >=0.40: top ~17.3% → Watch (includes Critical)
# <0.40:  Stable
TIER_CRITICAL_THRESHOLD = 0.90
TIER_WATCH_THRESHOLD    = 0.40