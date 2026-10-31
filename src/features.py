"""Feature engineering: turn the raw timestamp / weather into model-friendly columns."""
from __future__ import annotations

import pandas as pd

from . import config as cfg


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, month, is_weekend, is_rush_hour, is_night, weather_severity_index."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek          # Monday=0 ... Sunday=6
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_rush_hour"] = out["hour"].isin(cfg.RUSH_HOURS).astype(int)
    out["is_night"] = ((out["hour"] >= cfg.NIGHT_START_HOUR) | (out["hour"] < cfg.NIGHT_END_HOUR)).astype(int)
    # Unknown weather stays NaN here; the pipeline's imputer fills it using training data only.
    out["weather_severity_index"] = out["weather"].map(cfg.WEATHER_SEVERITY_INDEX)
    return out
