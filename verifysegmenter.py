from pathlib import Path

from config import MODEL_PATH
from src.model import (
    get_shap_values,
    get_top_driver_per_customer,
    load_model,
    score_customers,
    train_and_evaluate,
)
from src.pipeline import load_and_clean, encode_categoricals, get_feature_columns
from src.segmenter import CRITICAL, WATCH, STABLE, assign_tiers, build_tier_profiles


def get_model(df_enc, feat_cols):
    if Path(MODEL_PATH).exists():
        return load_model()

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    model, threshold, _ = train_and_evaluate(df_enc, feat_cols)
    return model, threshold


def main():
    df_raw = load_and_clean()
    df_enc = encode_categoricals(df_raw)
    feat_cols = get_feature_columns(df_enc)
    model, threshold = get_model(df_enc, feat_cols)

    curr_scored = score_customers(model, threshold, df_enc, feat_cols)
    shap_curr = get_shap_values(model, df_enc, feat_cols)
    curr_tiered = assign_tiers(curr_scored)
    curr_tiered["top_driver"] = get_top_driver_per_customer(shap_curr, feat_cols)
    curr_profiles = build_tier_profiles(curr_tiered, shap_curr, feat_cols)

    total = sum(curr_profiles[t]["count"] for t in [CRITICAL, WATCH, STABLE])
    assert total == len(df_enc), "Missing customers"
    print(f"Critical: {curr_profiles[CRITICAL]['count']}")
    print(f"Top drivers: {curr_profiles[CRITICAL]['top_drivers']}")
    print("Step 4 OK")


if __name__ == "__main__":
    main()
