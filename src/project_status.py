"""Project status for the progress review.

Overall completion = sum(weight x fraction_done). The weights and fractions are the team's own
ESTIMATES - edit them here and the dashboard + tests follow. Be ready to defend each line:
"done" means it exists in this repo and is covered by the pipeline/tests; "pending" means it
does not exist yet.
"""
from __future__ import annotations

import pandas as pd

COMPONENTS = [
    {
        "component": "Data layer", "weight": 15, "fraction_done": 0.80,
        "done": "Synthetic data generator, CSV loader, cleaning + validation report, feature engineering",
        "pending": "Integrate a real dataset (MoRTH / STATS19 / Kaggle) and map its columns to our schema",
    },
    {
        "component": "Severity prediction (Pillar 1)", "weight": 25, "fraction_done": 0.80,
        "done": "LR / RF / XGBoost / LightGBM x {none, class-weight, SMOTE}; stratified split + k-fold CV; "
                "fatal-class recall / F2 / PR-AUC; best-model selection and saved model",
        "pending": "Hyper-parameter tuning (grid/Optuna), decision-threshold tuning, probability calibration, "
                   "validation on real data",
    },
    {
        "component": "Hotspot detection (Pillar 3)", "weight": 15, "fraction_done": 0.80,
        "done": "K-Means zones, DBSCAN black spots, KDE density surface, severity-weighted ranking, "
                "Folium map, silhouette scores",
        "pending": "Tune on real coordinates, temporal hotspots (time-of-day), cross-reference with severity model",
    },
    {
        "component": "Risk prediction (Pillar 2)", "weight": 15, "fraction_done": 0.20,
        "done": "Prototype v0: severity-weighted score per grid cell x time band x weather, 0-100 index, "
                "temporal sanity check",
        "pending": "Exposure data, trained predictive risk model, calibrated risk levels, real-data validation",
    },
    {
        "component": "Explainability", "weight": 10, "fraction_done": 0.50,
        "done": "Global SHAP importance for the fatal class (one-hot columns merged per feature)",
        "pending": "LIME, per-prediction explanations in the dashboard",
    },
    {
        "component": "Dashboard", "weight": 10, "fraction_done": 0.50,
        "done": "Streamlit app: status, severity results, hotspot map, risk prototype, single-crash what-if",
        "pending": "Unified filters by location / time window / road condition, side-by-side three-pillar view",
    },
    {
        "component": "Engineering and delivery", "weight": 10, "fraction_done": 0.30,
        "done": "One-command pipeline, unit tests, documentation set",
        "pending": "Docker / cloud deployment, CI, final benchmark against literature, streaming-data hook",
    },
]


def overall_completion() -> float:
    """Weighted completion in percent (0-100)."""
    total = sum(c["weight"] for c in COMPONENTS)
    return round(100 * sum(c["weight"] * c["fraction_done"] for c in COMPONENTS) / total, 1)


def status_table() -> pd.DataFrame:
    df = pd.DataFrame(COMPONENTS)
    df["done_pct"] = (df["fraction_done"] * 100).round(0).astype(int)
    df["contribution"] = (df["weight"] * df["fraction_done"]).round(1)
    return df[["component", "weight", "done_pct", "contribution", "done", "pending"]]
