"""Data cleaning: remove rows that cannot be trusted and report what was removed.

Design choice: this step does NOT fill missing feature values. Imputation happens inside the
scikit-learn pipeline (src/severity.py) so medians/modes are learned on the training split only.
That avoids data leakage from the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def clean_accidents(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report). Raises ValueError if required columns are missing."""
    missing_cols = [c for c in cfg.REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    report: dict = {"rows_in": int(len(df))}
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    # 1. duplicate accident ids
    n = len(out)
    out = out.drop_duplicates(subset="accident_id", keep="first")
    report["duplicates_removed"] = int(n - len(out))

    # 2. missing / invalid target label
    n = len(out)
    out[cfg.TARGET] = pd.to_numeric(out[cfg.TARGET], errors="coerce")
    out = out[out[cfg.TARGET].isin(list(cfg.SEVERITY_LABELS))]
    report["bad_severity_removed"] = int(n - len(out))

    # 3. unreadable timestamps
    n = len(out)
    out = out[out["timestamp"].notna()]
    report["bad_timestamp_removed"] = int(n - len(out))

    # 4. impossible coordinates (out of range, or exactly 0,0 which usually means "unknown")
    n = len(out)
    lat = pd.to_numeric(out["latitude"], errors="coerce")
    lon = pd.to_numeric(out["longitude"], errors="coerce")
    valid = lat.between(-90, 90) & lon.between(-180, 180) & ~((lat == 0) & (lon == 0))
    out = out.assign(latitude=lat, longitude=lon)[valid]
    report["bad_coordinates_removed"] = int(n - len(out))

    # 5. implausible driver ages become "missing" (the pipeline imputes them later)
    age = pd.to_numeric(out["driver_age"], errors="coerce")
    out["driver_age"] = age.where(age.between(16, 90), np.nan)

    out[cfg.TARGET] = out[cfg.TARGET].astype(int)
    out = out.reset_index(drop=True)

    report["rows_out"] = int(len(out))
    na = out[cfg.REQUIRED_COLUMNS].isna().sum()
    report["missing_values_left"] = {k: int(v) for k, v in na.items() if v > 0}
    return out, report
