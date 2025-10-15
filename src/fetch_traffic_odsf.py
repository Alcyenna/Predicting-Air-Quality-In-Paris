import argparse
import math
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests


BASE_URL = "https://parisdata.opendatasoft.com/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/records"


def fetch_chunk(start_iso: str, end_iso: str, limit: int, offset: int) -> List[dict]:
    params = {
        "where": f"t_1h >= '{start_iso}' AND t_1h < '{end_iso}'",
        "limit": str(limit),
        "offset": str(offset),
        "order_by": "t_1h",
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def to_hourly_agg(records: List[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["id", "flow_sum", "flow_mean", "occ_mean"])  # empty
    df = pd.json_normalize(records)
    # Required fields: t_1h (UTC), q (flow), k (occupancy)
    # Some records may miss q/k; coerce to numeric
    df["t_1h"] = pd.to_datetime(df["t_1h"], utc=True)
    df["id"] = df["t_1h"].dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d %H")
    for col in ("q", "k"):
        if col not in df.columns:
            df[col] = pd.NA
    df["q"] = pd.to_numeric(df["q"], errors="coerce")
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    agg = df.groupby("id").agg(
        flow_sum=("q", "sum"),
        flow_mean=("q", "mean"),
        occ_mean=("k", "mean"),
    )
    agg = agg.reset_index()
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch traffic data from Opendatasoft and aggregate hourly for Paris")
    parser.add_argument("--start", type=str, default="2020-01-01T00:00:00Z", help="ISO start (UTC)")
    parser.add_argument("--end", type=str, default="2024-09-05T00:00:00Z", help="ISO end (UTC, exclusive)")
    parser.add_argument("--page-size", type=int, default=10000)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1]/"data"/"external"/"traffic"/"traffic_hourly_agg.parquet"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_aggs: List[pd.DataFrame] = []
    for page in range(args.max_pages):
        offset = page * args.page_size
        recs = fetch_chunk(args.start, args.end, args.page_size, offset)
        if not recs:
            break
        agg = to_hourly_agg(recs)
        all_aggs.append(agg)
        # Be nice to API
        time.sleep(0.2)
        # Early stop if we are past end (but API already filters)
        if len(recs) < args.page_size:
            break

    if not all_aggs:
        print("No records fetched.")
        return

    merged = pd.concat(all_aggs, ignore_index=True)
    merged = merged.groupby("id", as_index=False).agg({
        "flow_sum": "sum",
        "flow_mean": "mean",
        "occ_mean": "mean",
    })
    merged.to_parquet(out_path, index=False)
    print(f"Wrote {len(merged)} hourly rows -> {out_path}")


if __name__ == "__main__":
    main()


