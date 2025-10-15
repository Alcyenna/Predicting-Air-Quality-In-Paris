import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = [
    "valeur_NO2",
    "valeur_CO",
    "valeur_O3",
    "valeur_PM10",
    "valeur_PM25",
]


def seasonal_naive(train_csv: Path, test_csv: Path, out_csv: Path) -> None:
    train = pd.read_csv(train_csv, parse_dates=["id"])  # id is timestamp
    test = pd.read_csv(test_csv, parse_dates=["id"])   # only id

    train["hour"] = train["id"].dt.hour
    train["dow"] = train["id"].dt.dayofweek
    test["hour"] = test["id"].dt.hour
    test["dow"] = test["id"].dt.dayofweek

    # Key: (hour, dow)
    key_cols = ["hour", "dow"]
    agg = train.groupby(key_cols)[TARGET_COLUMNS].median().reset_index()

    sub = test[["id", "hour", "dow"]].merge(agg, on=key_cols, how="left")
    # Fallback to global mean if any missing
    for col in TARGET_COLUMNS:
        if sub[col].isna().any():
            sub[col].fillna(train[col].mean(), inplace=True)

    # Match required id format
    sub["id"] = sub["id"].dt.strftime("%Y-%m-%d %H")
    sub = sub[["id"] + TARGET_COLUMNS]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seasonal naive baseline (hour x DOW)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "submissions" / "submission_seasonal.csv"),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    seasonal_naive(data_dir / "train.csv", data_dir / "test.csv", Path(args.out))
    print(f"Saved submission to: {args.out}")


if __name__ == "__main__":
    main()


