from pathlib import Path

from config import MODEL_PATH
from src.model import train_and_evaluate
from src.pipeline import load_and_clean, encode_categoricals, get_feature_columns


def main():
    df_raw = load_and_clean()
    df_enc = encode_categoricals(df_raw)
    feat_cols = get_feature_columns(df_enc)

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    model, threshold, metrics, _ = train_and_evaluate(df_enc, feat_cols)
    print(metrics)
    # Expected output:
    # auc       : ~0.9914
    # threshold : ~0.687
    # precision : ~0.953
    # recall    : ~0.871
    # f1        : ~0.910
    assert metrics["auc"] > 0.97
    assert 0.65 <= metrics["threshold"] <= 0.72
    print("Step 2 OK")


if __name__ == "__main__":
    main()
