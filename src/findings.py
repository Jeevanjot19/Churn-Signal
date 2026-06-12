# src/findings.py

import pandas as pd


def finding_contacts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finding 1: Contact frequency vs churn rate.
    Verified: Contacts=6 has churn_rate=1.000 (n=54).
    """
    return (
        df.groupby("Contacts_Count_12_mon")["Churned"]
        .agg(churn_rate="mean", customer_count="count")
        .round(3)
        .reset_index()
    )


def finding_zero_balance(df: pd.DataFrame) -> dict:
    """
    Finding 2: Zero revolving balance vs churn rate.
    Verified: zero=0.362, non-zero=0.096, ratio=3.8x.
    """
    zero    = df[df["Total_Revolving_Bal"] == 0]
    nonzero = df[df["Total_Revolving_Bal"] > 0]
    return {
        "zero_churn_rate":    round(zero["Churned"].mean(), 3),
        "zero_count":         len(zero),
        "nonzero_churn_rate": round(nonzero["Churned"].mean(), 3),
        "nonzero_count":      len(nonzero),
        "ratio":              round(
            zero["Churned"].mean() / nonzero["Churned"].mean(), 1
        ),
    }


def finding_transaction_shield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finding 3: Transaction count thresholds vs churn rate.
    Verified: >=100 transactions → churn_rate=0.000 (n=687).
    """
    thresholds = [20, 40, 60, 80, 100, 120]
    rows = []
    for t in thresholds:
        subset = df[df["Total_Trans_Ct"] >= t]
        rows.append({
            "min_transactions": t,
            "churn_rate":       round(subset["Churned"].mean(), 3),
            "customer_count":   len(subset)
        })
    return pd.DataFrame(rows)


def finding_inactivity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finding 4: Months inactive vs churn rate.
    Verified: 1 month=4.5%, 4 months=29.9%.
    0-month bucket n=29 — excluded from charts (too small).
    """
    result = (
        df.groupby("Months_Inactive_12_mon")["Churned"]
        .agg(churn_rate="mean", customer_count="count")
        .round(3)
        .reset_index()
    )
    result["small_sample"] = result["customer_count"] < 100
    return result


def get_all_findings(df: pd.DataFrame) -> dict:
    return {
        "contacts":   finding_contacts(df),
        "balance":    finding_zero_balance(df),
        "tx_shield":  finding_transaction_shield(df),
        "inactivity": finding_inactivity_curve(df),
    }