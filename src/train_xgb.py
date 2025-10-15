import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from feature_engineering import build_features, TARGET_COLUMNS


def train_and_predict(train_csv: Path, test_csv: Path, out_csv: Path) -> None:
    train_df = pd.read_csv(train_csv, parse_dates=["id"])  # id is timestamp
    test_df = pd.read_csv(test_csv, parse_dates=["id"])  # only id

    # Build features on concatenated frame
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    for col in TARGET_COLUMNS:
        test_df[col] = np.nan

    full = pd.concat([train_df, test_df], ignore_index=True)
    full_feat = build_features(full)

    feat_cols = [
        c
        for c in full_feat.columns
        if c not in ("id", "is_train") and c not in TARGET_COLUMNS
    ]

    train_feat = full_feat[full_feat["is_train"] == 1].reset_index(drop=True)
    test_feat = full_feat[full_feat["is_train"] == 0].reset_index(drop=True)

    preds = {"id": test_feat["id"].astype(str)}

    for target in TARGET_COLUMNS:
        y = train_feat[target].values
        X = train_feat[feat_cols]

        # Drop rows with NaNs from lags/rolling
        valid_mask = ~np.isnan(y)
        for c in feat_cols:
            valid_mask &= ~X[c].isna().values
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        n = len(X_valid)
        val_size = min(24 * 7, max(1, n // 10))
        train_end = n - val_size

        X_tr, y_tr = X_valid.iloc[:train_end], y_valid[:train_end]
        X_va, y_va = X_valid.iloc[train_end:], y_valid[train_end:]

        model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            tree_method="hist",
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        preds[target] = model.predict(test_feat[feat_cols])

    sub = pd.DataFrame(preds)
    sub = sub[["id"] + TARGET_COLUMNS]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost and predict test set")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "submissions"
            / "submission_xgb.csv"
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    out_csv = Path(args.out)

    train_and_predict(train_csv, test_csv, out_csv)
    print(f"Saved submission to: {out_csv}")


if __name__ == "__main__":
    main()


