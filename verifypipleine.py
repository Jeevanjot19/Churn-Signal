from src.pipeline import load_and_clean, encode_categoricals, get_feature_columns

df_raw = load_and_clean()
assert df_raw.shape == (10127, 21), f"Got {df_raw.shape}"
assert round(df_raw["Churned"].mean(), 3) == 0.161
assert df_raw.isnull().sum().sum() == 0

df_enc = encode_categoricals(df_raw)
feat_cols = get_feature_columns(df_enc)
assert len(feat_cols) == 19, f"Got {len(feat_cols)}"
print("Step 1 OK")