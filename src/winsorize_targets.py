import numpy as np
import pandas as pd


TARGETS = ["valeur_NO2", "valeur_CO", "valeur_O3", "valeur_PM10", "valeur_PM25"]


def winsorize(df: pd.DataFrame, lower: float = 0.5, upper: float = 99.5) -> pd.DataFrame:
    out = df.copy()
    for col in TARGETS:
        if col in out.columns:
            lo = np.nanpercentile(out[col], lower)
            hi = np.nanpercentile(out[col], upper)
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out


