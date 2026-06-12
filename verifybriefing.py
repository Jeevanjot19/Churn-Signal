from pathlib import Path

from config import MODEL_PATH, PREV_WEEK_LABEL, WEEK_LABEL
from src.briefing import generate_briefing
from src.findings import get_all_findings
from src.model import get_shap_values, load_model, score_customers, train_and_evaluate
from src.pipeline import load_and_clean, encode_categoricals, get_feature_columns
from src.segmenter import assign_tiers, build_tier_profiles


def get_model(df_enc, feat_cols):
    if Path(MODEL_PATH).exists():
        return load_model()

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    model, threshold, _ = train_and_evaluate(df_enc, feat_cols)
    return model, threshold


def build_profiles(model, threshold, df_week, feat_cols):
    scored = score_customers(model, threshold, df_week, feat_cols)
    shap_values = get_shap_values(model, df_week, feat_cols)
    tiered = assign_tiers(scored)
    return build_tier_profiles(tiered, shap_values, feat_cols)


def main():
    df_raw = load_and_clean()
    df_enc = encode_categoricals(df_raw)
    feat_cols = get_feature_columns(df_enc)
    model, threshold = get_model(df_enc, feat_cols)

    curr_profiles = build_profiles(model, threshold, df_enc, feat_cols)
    prev_profiles = {}
    findings = get_all_findings(df_raw)

    briefing = generate_briefing(
        curr_profiles,
        prev_profiles,
        findings,
        WEEK_LABEL,
        PREV_WEEK_LABEL,
    )

    paragraphs = briefing.split("\n\n")
    assert len(paragraphs) == 3
    assert WEEK_LABEL in briefing
    assert "Critical" in briefing
    assert "Watch" in briefing
    print(briefing)
    print("Briefing OK")


if __name__ == "__main__":
    main()
