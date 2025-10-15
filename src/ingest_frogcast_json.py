import argparse
import json
from pathlib import Path

import pandas as pd


def normalize_frogcast_json(input_json: Path, out_parquet: Path) -> pd.DataFrame:
    data = json.loads(input_json.read_text())
    df = pd.DataFrame(data)
    # Expect keys: timestamp (UTC), all_sky_global_horizontal_irradiance
    if "timestamp" not in df.columns:
        raise ValueError("Missing 'timestamp' column in frogcast JSON")
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Paris")
    df["id"] = ts.dt.strftime("%Y-%m-%d %H")
    if "all_sky_global_horizontal_irradiance" in df.columns:
        df = df.rename(
            columns={
                "all_sky_global_horizontal_irradiance": "ghi",
            }
        )
    df = df[[c for c in df.columns if c in ("id", "ghi")]]
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return df


def build_hour_dow_climatology(df: pd.DataFrame, train_csv: Path, out_parquet: Path) -> None:
    train = pd.read_csv(train_csv, parse_dates=["id"]).sort_values("id")
    # restrict to train bounds
    df_times = pd.to_datetime(df["id"])
    mask = (df_times >= train["id"].min()) & (df_times <= train["id"].max())
    base = df.loc[mask].copy()
    if base.empty:
        base = df.copy()  # fallback if no overlap
    base["hour"] = pd.to_datetime(base["id"]).dt.hour
    base["dow"] = pd.to_datetime(base["id"]).dt.dayofweek
    clim = base.groupby(["hour", "dow"], as_index=False)[[c for c in base.columns if c != "id"]].mean()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    clim.to_parquet(out_parquet, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Frogcast JSON and build climatology")
    parser.add_argument("--in-json", type=str, required=True, help="Path to raw Frogcast JSON file")
    parser.add_argument(
        "--out-parquet",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "external" / "weather" / "frogcast_ghi.parquet"),
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "train.csv"),
    )
    parser.add_argument(
        "--out-clim",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "weather"
            / "frogcast_ghi_climatology.parquet"
        ),
    )
    args = parser.parse_args()

    in_json = Path(args.in_json)
    out_parquet = Path(args.out_parquet)
    df = normalize_frogcast_json(in_json, out_parquet)
    build_hour_dow_climatology(df, Path(args.train_csv), Path(args.out_clim))
    print(f"Wrote: {out_parquet} and {args.out_clim}")


if __name__ == "__main__":
    main()


