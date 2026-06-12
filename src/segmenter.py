# src/segmenter.py

import numpy as np
import pandas as pd
from config import TIER_CRITICAL_THRESHOLD, TIER_WATCH_THRESHOLD

CRITICAL = "Critical"
WATCH    = "Watch"
STABLE   = "Stable"


def assign_tiers(df_scored: pd.DataFrame) -> pd.DataFrame:
    df = df_scored.copy()

    def _tier(p):
        if p >= TIER_CRITICAL_THRESHOLD: return CRITICAL
        elif p >= TIER_WATCH_THRESHOLD:  return WATCH
        else:                            return STABLE

    df["risk_tier"] = df["churn_probability"].apply(_tier)
    return df


def build_tier_profiles(df_tiered: pd.DataFrame,
                         shap_values,
                         feature_cols: list) -> dict:
    profiles = {}

    for tier in [CRITICAL, WATCH, STABLE]:
        mask    = (df_tiered["risk_tier"] == tier).values
        tier_df = df_tiered[df_tiered["risk_tier"] == tier]
        t_shap  = shap_values[mask]

        if len(tier_df) == 0:
            profiles[tier] = {"count": 0}
            continue

        mean_abs = np.abs(t_shap).mean(axis=0)
        top3_idx = np.argsort(mean_abs)[::-1][:3]
        top3     = [feature_cols[i] for i in top3_idx]

        profiles[tier] = {
            "count":              len(tier_df),
            "mean_churn_prob":    round(tier_df["churn_probability"].mean(), 3),
            "top_drivers":        top3,
            "mean_trans_count":   round(tier_df["Total_Trans_Ct"].mean(), 1),
            "mean_inactive_mo":   round(tier_df["Months_Inactive_12_mon"].mean(), 2),
            "mean_contacts":      round(tier_df["Contacts_Count_12_mon"].mean(), 2),
            "zero_balance_pct":   round(
                (tier_df["Total_Revolving_Bal"] == 0).mean(), 3),
            "mean_credit_limit":  round(tier_df["Credit_Limit"].mean(), 0),
        }

    return profiles