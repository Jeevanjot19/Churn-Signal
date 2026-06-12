# src/pipeline.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import DATA_PATH, RANDOM_STATE, CATEGORICAL_COLUMNS


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Remove target leakage columns
    leaked = [c for c in df.columns if "Naive_Bayes" in c]
    df.drop(columns=leaked, inplace=True)

    # Binary target: 1=churned, 0=retained
    df["Churned"] = (
        df["Attrition_Flag"] == "Attrited Customer"
    ).astype(int)
    df.drop(columns=["Attrition_Flag"], inplace=True)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in ["CLIENTNUM", "Churned"]]

