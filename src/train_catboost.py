import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error

from feature_engineering import build_features, TARGET_COLUMNS
from winsorize_targets import winsorize


def time_series_split_indices(n: int, n_splits: int = 2, val_len: int = 24 * 7) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start_val = n - n_splits * val_len
    if start_val <= 0:
        start_val = val_len
    for i in range(n_splits):
        train_end = start_val + i * val_len
        val_start = train_end
        val_end = min(val_start + val_len, n)
        if val_end - val_start < max(1, val_len // 2):
            break
        splits.append((idx[:train_end], idx[val_start:val_end]))
    return splits


def train_and_predict(train_csv: Path, test_csv: Path, out_csv: Path, cv_folds: int = 0, cv_val_len: int = 24*7, cv_verbose: bool = True) -> None:
    print(f"Loading train from {train_csv}")
    train_df = pd.read_csv(train_csv, parse_dates=["id"])  # id is timestamp
    train_df = winsorize(train_df)
    print(f"Loading test from {test_csv}")
    test_df = pd.read_csv(test_csv, parse_dates=["id"])  # only id

    # Build features on concatenated frame to ensure identical transforms
    train_df["is_train"], test_df["is_train"] = 1, 0
    test_df = test_df.assign(**{c: np.nan for c in TARGET_COLUMNS})
    full = pd.concat([train_df, test_df], ignore_index=True)
    print("Building features...")
    full_feat = build_features(full)
    print("Features built.")

    feat_cols = [c for c in full_feat.columns if c not in ("id", "is_train") and c not in TARGET_COLUMNS]
    train_feat = full_feat[full_feat["is_train"] == 1].reset_index(drop=True)
    test_feat = full_feat[full_feat["is_train"] == 0].reset_index(drop=True)

    preds = {"id": test_feat["id"].dt.strftime("%Y-%m-%d %H")}

    print(f"Training targets: {TARGET_COLUMNS}")
    for target in TARGET_COLUMNS:
        print(f"\n=== Target: {target} ===")
        y = train_feat[target].values
        X = train_feat[feat_cols]

        # Drop rows with NaNs from lags/rolling at the head
        valid_mask = ~np.isnan(y)
        for c in feat_cols:
            valid_mask &= ~X[c].isna().values
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        params = dict(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3.0,
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        model = CatBoostRegressor(**params)

        if cv_folds and cv_folds > 0:
            splits = time_series_split_indices(len(X_valid), n_splits=cv_folds, val_len=cv_val_len)
            fold_mae: List[float] = []
            for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
                X_tr, y_tr = X_valid.iloc[tr_idx], y_valid[tr_idx]
                X_va, y_va = X_valid.iloc[va_idx], y_valid[va_idx]
                model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_va, y_va), use_best_model=True)
                y_hat = model.predict(X_va)
                mae = mean_absolute_error(y_va, y_hat)
                fold_mae.append(mae)
                if cv_verbose:
                    print(f"Fold {fold_idx}: MAE={mae:.5f} (n_train={len(X_tr)}, n_val={len(X_va)})")
            if cv_verbose and fold_mae:
                print(f"CV MAE {target}: mean={np.mean(fold_mae):.5f}, std={np.std(fold_mae):.5f}")
            # Refit on all valid data
            model.fit(Pool(X_valid, y_valid))
        else:
            n = len(X_valid)
            val_size = min(cv_val_len, max(cv_val_len, n // 10))
            train_end = n - val_size
            X_tr, y_tr = X_valid.iloc[:train_end], y_valid[:train_end]
            X_va, y_va = X_valid.iloc[train_end:], y_valid[train_end:]
            print(f"Train size: {len(X_tr)}, Val size: {len(X_va)}, Test rows: {len(test_feat)}")
            model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_va, y_va), use_best_model=True)
            print("Fitted. Predicting test...")

        preds[target] = model.predict(test_feat[feat_cols])

    sub = pd.DataFrame(preds)
    sub = sub[["id"] + TARGET_COLUMNS]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"Wrote submission to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CatBoost and predict test set")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "submissions" / "submission_catboost.csv"))
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--cv-horizon-hours", type=int, default=168)
    parser.add_argument("--cv-quiet", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    out_csv = Path(args.out)

    train_and_predict(
        train_csv,
        test_csv,
        out_csv,
        cv_folds=max(0, int(args.cv_folds)),
        cv_val_len=max(1, int(args.cv_horizon_hours)),
        cv_verbose=not args.cv_quiet,
    )


if __name__ == "__main__":
    main()


