from __future__ import annotations

from dataclasses import dataclass
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS: List[str] = [
    "valeur_NO2",
    "valeur_CO",
    "valeur_O3",
    "valeur_PM10",
    "valeur_PM25",
]


def add_calendar_features(df: pd.DataFrame, ts_col: str = "id") -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    out["hour"] = out[ts_col].dt.hour
    out["dow"] = out[ts_col].dt.dayofweek
    out["dom"] = out[ts_col].dt.day
    out["month"] = out[ts_col].dt.month
    out["doy"] = out[ts_col].dt.dayofyear
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    # Rush hours (approx.)
    out["is_rush_morning"] = out["hour"].isin([7, 8, 9]).astype(int)
    out["is_rush_evening"] = out["hour"].isin([17, 18, 19]).astype(int)
    # August low-activity flag
    out["is_august"] = (out["month"] == 8).astype(int)
    # Holidays (France) if workalendar is available
    try:
        from workalendar.europe import France

        cal = France()
        out["is_holiday"] = out[ts_col].dt.date.apply(lambda d: int(cal.is_holiday(d)))
        # Bridge day: day before or after a holiday
        dates = out[ts_col].dt.floor("D").dt.date
        is_prev_holiday = dates.map(lambda d: cal.is_holiday(pd.to_datetime(d) - pd.Timedelta(days=1)))
        is_next_holiday = dates.map(lambda d: cal.is_holiday(pd.to_datetime(d) + pd.Timedelta(days=1)))
        out["is_bridge_day"] = (is_prev_holiday | is_next_holiday).astype(int)
    except Exception:
        out["is_holiday"] = 0
        out["is_bridge_day"] = 0
    # cyclic encodings
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["sin_dow"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["cos_dow"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["sin_doy"] = np.sin(2 * np.pi * out["doy"] / 366)
    out["cos_doy"] = np.cos(2 * np.pi * out["doy"] / 366)
    # DST flag (approx via pandas tz conversion to Europe/Paris)
    try:
        paris = out[ts_col].dt.tz_localize("Europe/Paris", ambiguous="NaT", nonexistent="shift_forward")
        out["is_dst"] = paris.dt.dst().ne(pd.Timedelta(0)).astype(int)
    except Exception:
        out["is_dst"] = 0
    return out


def add_lockdown_features(df: pd.DataFrame, ts_col: str = "id") -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    # Expect csv at data/external/france_lockdowns.csv with columns Start, End, Type
    lock_csv = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "external"
        / "france_lockdowns.csv"
    )
    try:
        locks = pd.read_csv(lock_csv)
        locks["Start"] = pd.to_datetime(locks["Start"])
        locks["End"] = pd.to_datetime(locks["End"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out["is_lockdown"] = 0
        out["is_lockdown_partial"] = 0
        out["is_lockdown_full"] = 0
        for _, row in locks.iterrows():
            mask = (out[ts_col] >= row["Start"]) & (out[ts_col] <= row["End"])
            out.loc[mask, "is_lockdown"] = 1
            if str(row.get("Type", "")).upper().startswith("FULL"):
                out.loc[mask, "is_lockdown_full"] = 1
            else:
                out.loc[mask, "is_lockdown_partial"] = 1
    except Exception:
        # If file missing or parsing fails, default to zeros
        out["is_lockdown"] = 0
        out["is_lockdown_partial"] = 0
        out["is_lockdown_full"] = 0
    return out


def add_lag_rolling_features(
    df: pd.DataFrame, lags: List[int], roll_windows: List[int]
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("id").reset_index(drop=True)
    for col in TARGET_COLUMNS:
        for l in lags:
            out[f"{col}_lag{l}"] = out[col].shift(l)
        for w in roll_windows:
            out[f"{col}_rollmean{w}"] = out[col].rolling(window=w, min_periods=max(1, w // 2)).mean()
            out[f"{col}_rollstd{w}"] = out[col].rolling(window=w, min_periods=max(1, w // 2)).std()
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    base = add_calendar_features(df)
    base = add_lockdown_features(base)
    # School breaks Zone C (optional CSV: data/external/calendar/zone_c_holidays.csv with a 'date' column)
    try:
        cal_csv = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "calendar"
            / "zone_c_holidays.csv"
        )
        cal = pd.read_csv(cal_csv, parse_dates=["date"])  # expects daily marks
        cal["is_school_break"] = 1
        base_date = base[["id"]].copy()
        base_date["date"] = base_date["id"].dt.floor("D")
        base = base.merge(cal[["date", "is_school_break"]], on="date", how="left").fillna({"is_school_break": 0})
    except Exception:
        base["is_school_break"] = 0

    # Traffic hourly aggregates (optional Parquet): data/external/traffic/traffic_hourly_agg.parquet
    try:
        traffic_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "traffic"
            / "traffic_hourly_agg.parquet"
        )
        traffic = pd.read_parquet(traffic_path)
        # Expect columns: id (YYYY-MM-DD HH), flow_sum, flow_mean, occ_mean
        base = base.merge(traffic, on="id", how="left")
        # Fill small gaps via forward fill per day
        for col in ("flow_sum", "flow_mean", "occ_mean"):
            if col in base.columns:
                base[col] = base[col].astype(float)
                base[col] = base[col].fillna(method="ffill")
    except Exception:
        pass
    # Typical lags and windows for hourly data
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]  # up to 7 days
    roll_windows = [3, 6, 12, 24, 48]
    feat = add_lag_rolling_features(base, lags=lags, roll_windows=roll_windows)
    # Add lags/rolling for traffic if present
    for tcol in ("flow_sum", "flow_mean", "occ_mean"):
        if tcol in feat.columns:
            feat = feat.sort_values("id").reset_index(drop=True)
            for l in lags:
                feat[f"{tcol}_lag{l}"] = feat[tcol].shift(l)
            for w in roll_windows:
                feat[f"{tcol}_rollmean{w}"] = feat[tcol].rolling(window=w, min_periods=max(1, w // 2)).mean()
                feat[f"{tcol}_rollstd{w}"] = feat[tcol].rolling(window=w, min_periods=max(1, w // 2)).std()
    return feat


