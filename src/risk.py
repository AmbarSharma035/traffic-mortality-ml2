"""Pillar 2 - Accident Risk Prediction: PROTOTYPE v0 (about 20% of the planned pillar).

What this is: a historical, severity-weighted crash score for every
(grid cell ~1 km) x (time band) x (weather group) combination, scaled to a 0-100 risk index.

What this is NOT (yet): a trained model, or an exposure-based rate. We have no traffic-volume
data, so "risk" here means "where/when severity-weighted crashes have concentrated before".
Pending work (see src/project_status.py): exposure data, a proper predictive model,
calibrated thresholds, validation on real data.

One honest check is included: build the score from the first 80% of the time range, then see
how much of the last 20% of (severity-weighted) crashes fall in the top-ranked cells (hit rate / PAI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .hotspots import severity_weight


def add_risk_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add grid_id, cell centre, time_band and weather_group columns."""
    out = df.copy()
    d = cfg.RISK_GRID_DECIMALS
    out["cell_lat"] = out["latitude"].round(d)
    out["cell_lon"] = out["longitude"].round(d)
    out["grid_id"] = out["cell_lat"].astype(str) + "_" + out["cell_lon"].astype(str)
    out["time_band"] = pd.cut(out["hour"], bins=cfg.TIME_BAND_BINS, labels=cfg.TIME_BAND_LABELS).astype(str)
    out["weather_group"] = np.where(out["weather"].isin(cfg.ADVERSE_WEATHER), "Adverse", "Normal")
    return out


def risk_level(index: pd.Series) -> pd.Series:
    """Placeholder tiers from percentiles (real thresholds are an open item, SRS TBD-05)."""
    return pd.cut(index, bins=[-0.01, 50, 80, 95, 100.01],
                  labels=["Low", "Medium", "High", "Critical"]).astype(str)


def build_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (grid cell, time band, weather group) that has at least one crash."""
    d = add_risk_keys(df)
    d["weight"] = severity_weight(d)
    d["is_fatal"] = (d[cfg.TARGET] == cfg.FATAL_CLASS).astype(int)
    table = d.groupby(["grid_id", "cell_lat", "cell_lon", "time_band", "weather_group"]).agg(
        n_crashes=("weight", "size"), n_fatal=("is_fatal", "sum"), weighted_score=("weight", "sum"),
    ).reset_index()
    table["risk_index"] = (table["weighted_score"].rank(pct=True) * 100).round(1)
    table["risk_level"] = risk_level(table["risk_index"])
    return table.sort_values("weighted_score", ascending=False).reset_index(drop=True)


def temporal_validation(df: pd.DataFrame, top_k: int = 20) -> dict:
    """Build the score from the past (first 80% of time); check it against the future (last 20%).

    Hit rate: share of FUTURE severity-weighted crashes that fall inside the top_k grid cells
    ranked by the PAST score. Compared with picking top_k cells at random (the baseline).
    PAI (Predictive Accuracy Index) = hit rate / baseline = how many times better than random.

    NOTE: on synthetic data the hotspots never move, so a high PAI is guaranteed by construction.
    This is a validation *harness*; it only becomes evidence once real data is plugged in.
    """
    d = add_risk_keys(df).sort_values("timestamp")
    split_at = d["timestamp"].quantile(cfg.RISK_TEMPORAL_SPLIT)
    past, future = d[d["timestamp"] <= split_at], d[d["timestamp"] > split_at]
    past_score = past.assign(w=severity_weight(past)).groupby("grid_id")["w"].sum()
    future_score = future.assign(w=severity_weight(future)).groupby("grid_id")["w"].sum()

    top_cells = past_score.nlargest(top_k).index
    hit_rate = future_score.reindex(top_cells).fillna(0).sum() / future_score.sum()

    step = 10 ** -cfg.RISK_GRID_DECIMALS
    n_cells = (round((d["cell_lat"].max() - d["cell_lat"].min()) / step) + 1) * \
              (round((d["cell_lon"].max() - d["cell_lon"].min()) / step) + 1)
    baseline = top_k / n_cells
    return {"top_k_cells": top_k, "hit_rate_pct": round(float(100 * hit_rate), 1),
            "random_baseline_pct": round(float(100 * baseline), 2),
            "pai": round(float(hit_rate / baseline), 1), "n_cells_in_area": int(n_cells),
            "split_date": str(split_at.date())}
