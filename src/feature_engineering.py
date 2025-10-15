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
        out["is_after_holiday"] = is_prev_holiday.astype(int)
        out["is_before_holiday"] = is_next_holiday.astype(int)
    except Exception:
        out["is_holiday"] = 0
        out["is_bridge_day"] = 0
        out["is_after_holiday"] = 0
        out["is_before_holiday"] = 0
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
    # Month boundaries & paydays
    out["is_month_start"] = out[ts_col].dt.is_month_start.astype(int)
    out["is_month_end"] = out[ts_col].dt.is_month_end.astype(int)
    out["is_payday"] = out["dom"].between(25, 30).astype(int)
    # Hour bins (3h)
    out["hour_bin_3h"] = (out["hour"] // 3).astype(int)
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
        # Fallback by climatology hour×dow if still missing
        if {"flow_mean", "occ_mean"}.issubset(base.columns):
            clim = (
                base[["hour", "dow", "flow_mean", "occ_mean"]]
                .groupby(["hour", "dow"], as_index=False)
                .mean()
                .rename(
                    columns={
                        "flow_mean": "flow_mean_clim",
                        "occ_mean": "occ_mean_clim",
                    }
                )
            )
            base = base.merge(clim, on=["hour", "dow"], how="left")
            if "flow_mean" in base.columns and "flow_mean_clim" in base.columns:
                base["flow_mean"] = base["flow_mean"].fillna(base["flow_mean_clim"])  
            if "occ_mean" in base.columns and "occ_mean_clim" in base.columns:
                base["occ_mean"] = base["occ_mean"].fillna(base["occ_mean_clim"])   
            base.drop(columns=[c for c in ["flow_mean_clim", "occ_mean_clim"] if c in base.columns], inplace=True)
        # Interaction features and stabilized transforms
        if "flow_mean" in base.columns:
            base["log1p_flow_mean"] = np.log1p(base["flow_mean"])
            base["flow_mean_rush_morning"] = base["flow_mean"] * base.get("is_rush_morning", 0)
            base["flow_mean_weekend"] = base["flow_mean"] * base.get("is_weekend", 0)
        if "flow_sum" in base.columns:
            base["log1p_flow_sum"] = np.log1p(base["flow_sum"])
        if "occ_mean" in base.columns:
            base["occ_mean_holiday"] = base["occ_mean"] * base.get("is_holiday", 0)
    except Exception:
        pass

    # Weather hourly from daily upsampling: data/external/weather/meteostat_hourly_from_daily.parquet
    try:
        weather_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "weather"
            / "meteostat_hourly_from_daily.parquet"
        )
        weather = pd.read_parquet(weather_path)
        # Expect columns: id, tavg, tmin, tmax, prcp, wind_u, wind_v, pres, tsun
        base = base.merge(weather, on="id", how="left")
        # Weather derived
        if {"wind_u", "wind_v"}.issubset(base.columns):
            base["wind_speed"] = np.sqrt(np.square(base["wind_u"]) + np.square(base["wind_v"]))
        if "prcp" in base.columns:
            base["prcp_gt_0"] = (base["prcp"] > 0).astype(int)
            base["prcp_heavy"] = (base["prcp"] >= 5.0).astype(int)
    except Exception:
        pass

    # Events (CSV FR headers tolerant): data/external/events/events.csv
    try:
        events_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "events"
            / "events.csv"
        )
        ev = pd.read_csv(events_path)
        cols = {c.lower().strip(): c for c in ev.columns}
        # Map common french headers
        def get_col(names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None
        date_col = get_col(["date", "jour"])  # simple date
        start_col = get_col(["debut", "début", "date_debut", "date début", "start"])  # period start
        end_col = get_col(["fin", "date_fin", "date fin", "end"])  # period end
        label_col = get_col(["libelle", "libellé", "titre", "nom", "label"])  # label

        dates: pd.Series
        if date_col is not None:
            dates = pd.to_datetime(ev[date_col]).dt.tz_localize("Europe/Paris", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert(None).dt.floor("D")
        elif start_col is not None and end_col is not None:
            starts = pd.to_datetime(ev[start_col]).dt.tz_localize("Europe/Paris", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert(None).dt.floor("D")
            ends = pd.to_datetime(ev[end_col]).dt.tz_localize("Europe/Paris", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert(None).dt.floor("D")
            all_days = []
            for s, e in zip(starts, ends):
                if pd.isna(s) or pd.isna(e):
                    continue
                rng = pd.date_range(s, e, freq="D")
                all_days.append(pd.Series(rng))
            dates = pd.concat(all_days, ignore_index=True) if all_days else pd.Series([], dtype="datetime64[ns]")
        else:
            dates = pd.Series([], dtype="datetime64[ns]")

        ev_days = pd.DataFrame({"date": pd.to_datetime(dates).dt.floor("D").astype("datetime64[ns]")})
        ev_days.drop_duplicates(inplace=True)
        base["date"] = base["id"].dt.floor("D").astype("datetime64[ns]")
        base = base.merge(ev_days.assign(is_event=1), on="date", how="left")
        base["is_event"] = base["is_event"].fillna(0).astype(int)
        # Specific label flags for frequent labels (slugify basic)
        if label_col is not None and date_col is not None:
            # Map date->label for per-day labels
            tmp = ev[[date_col, label_col]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col]).dt.floor("D").astype("datetime64[ns]")
            tmp[label_col] = (
                tmp[label_col]
                .astype(str)
                .str.normalize("NFKD")
                .str.encode("ascii", errors="ignore")
                .str.decode("ascii")
                .str.lower()
                .str.replace("[^a-z0-9]+", "_", regex=True)
                .str.strip("_")
            )
            # Keep top few labels
            top = tmp[label_col].value_counts().head(5).index.tolist()
            for lab in top:
                days = tmp.loc[tmp[label_col] == lab, date_col].dropna().unique()
                if len(days) == 0:
                    continue
                flag = f"is_{lab[:20]}"
                base[flag] = base["date"].isin(pd.to_datetime(days)).astype(int)
        # Interactions with traffic/weather if available
        if "flow_mean" in base.columns:
            base["event_flow_mean"] = base["is_event"] * base["flow_mean"]
        if "prcp" in base.columns:
            base["event_prcp"] = base["is_event"] * base["prcp"]
    except Exception:
        pass

    # Enforce no-leak: for test rows, replace external observations by climatologies (hour×dow) computed on train rows
    try:
        mask_train = base.get("is_train", 0).astype(int) == 1
        mask_test = base.get("is_train", 0).astype(int) == 0
        # Traffic climatology from train
        for cols in [("flow_sum",), ("flow_mean",), ("occ_mean",)]:
            col = cols[0]
            if col in base.columns and mask_train.any():
                clim = (
                    base.loc[mask_train, ["hour", "dow", col]]
                    .groupby(["hour", "dow"], as_index=False)
                    .mean()
                    .rename(columns={col: f"{col}_clim"})
                )
                base = base.merge(clim, on=["hour", "dow"], how="left")
                base.loc[mask_test, col] = base.loc[mask_test, f"{col}_clim"]
                base.drop(columns=[f"{col}_clim"], inplace=True)
        # Weather climatology from train
        weather_cols = [c for c in ["tavg","tmin","tmax","prcp","wind_u","wind_v","pres","tsun","wind_speed"] if c in base.columns]
        if weather_cols and mask_train.any():
            climw = (
                base.loc[mask_train, ["hour", "dow"] + weather_cols]
                .groupby(["hour", "dow"], as_index=False)
                .mean()
            )
            base = base.merge(climw, on=["hour", "dow"], how="left", suffixes=("", "_clim"))
            for c in weather_cols:
                base.loc[mask_test, c] = base.loc[mask_test, f"{c}_clim"]
            base.drop(columns=[f"{c}_clim" for c in weather_cols], inplace=True)
            # Recompute precipitation flags after replacement
            if "prcp" in base.columns:
                base.loc[mask_test, "prcp_gt_0"] = (base.loc[mask_test, "prcp"] > 0).astype(int)
                base.loc[mask_test, "prcp_heavy"] = (base.loc[mask_test, "prcp"] >= 5.0).astype(int)
        # Events: keep only a priori-known (we neutralize generic events on test)
        if "is_event" in base.columns:
            base.loc[mask_test, "is_event"] = 0
            # also zero-out derived event flags if present
            for c in list(base.columns):
                if c.startswith("is_") and c not in {"is_weekend","is_holiday","is_bridge_day","is_after_holiday","is_before_holiday","is_dst","is_school_break"}:
                    base.loc[mask_test, c] = 0
    except Exception:
        pass
    # Typical lags and windows for hourly data
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]  # up to 7 days
    roll_windows = [3, 6, 12, 24, 48]
    # Ensure time order for diffs/ratios
    base = base.sort_values("id").reset_index(drop=True)
    # Traffic extra features
    if "flow_mean" in base.columns:
        base["diff1h_flow_mean"] = base["flow_mean"].diff(1)
        base["diff24h_flow_mean"] = base["flow_mean"] - base["flow_mean"].shift(24)
    if {"flow_mean", "occ_mean"}.issubset(base.columns):
        base["occ_per_flow"] = base["occ_mean"] / (base["flow_mean"].replace(0, np.nan))
        base["occ_per_flow"].replace([np.inf, -np.inf], np.nan, inplace=True)
    # Interactions météo × trafic
    if {"flow_mean", "wind_u"}.issubset(base.columns):
        base["flow_wind_u"] = base["flow_mean"] * base["wind_u"]
    if {"flow_mean", "wind_v"}.issubset(base.columns):
        base["flow_wind_v"] = base["flow_mean"] * base["wind_v"]
    if {"occ_mean", "prcp"}.issubset(base.columns):
        base["occ_prcp"] = base["occ_mean"] * base["prcp"]

    feat = add_lag_rolling_features(base, lags=lags, roll_windows=roll_windows)
    # Add lags/rolling for traffic if present
    for tcol in ("flow_sum", "flow_mean", "occ_mean", "log1p_flow_sum", "log1p_flow_mean", "occ_per_flow", "diff1h_flow_mean", "diff24h_flow_mean"):
        if tcol in feat.columns:
            feat = feat.sort_values("id").reset_index(drop=True)
            for l in lags:
                feat[f"{tcol}_lag{l}"] = feat[tcol].shift(l)
            for w in roll_windows:
                feat[f"{tcol}_rollmean{w}"] = feat[tcol].rolling(window=w, min_periods=max(1, w // 2)).mean()
                feat[f"{tcol}_rollstd{w}"] = feat[tcol].rolling(window=w, min_periods=max(1, w // 2)).std()
    # Add lags/rolling for selected weather features if present
    for wcol in ("tavg", "tmin", "tmax", "prcp", "wind_u", "wind_v", "wind_speed", "pres", "tsun"):
        if wcol in feat.columns:
            feat = feat.sort_values("id").reset_index(drop=True)
            for l in lags:
                feat[f"{wcol}_lag{l}"] = feat[wcol].shift(l)
            for rw in roll_windows:
                feat[f"{wcol}_rollmean{rw}"] = feat[wcol].rolling(window=rw, min_periods=max(1, rw // 2)).mean()
                feat[f"{wcol}_rollstd{rw}"] = feat[wcol].rolling(window=rw, min_periods=max(1, rw // 2)).std()
    return feat


