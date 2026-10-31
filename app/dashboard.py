"""Streamlit dashboard.  Run from the project root:  streamlit run app/dashboard.py

It only READS the files created by `python run_pipeline.py` (reports/ and models/), except the
"Predict a crash" page which loads the saved model to score one what-if crash.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `from src import ...` works when launched by Streamlit

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from src import config as cfg  # noqa: E402
from src.features import engineer_features  # noqa: E402
from src.project_status import overall_completion, status_table  # noqa: E402
from src.severity import predict_severity  # noqa: E402

st.set_page_config(page_title="Traffic Mortality ML - progress demo", layout="wide")


# --------------------------------------------------------------------------- loaders
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def load_json(path_str: str, mtime: float) -> dict:
    return json.loads(Path(path_str).read_text())


@st.cache_data
def load_csv(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_resource
def load_model(path_str: str, mtime: float):
    return joblib.load(path_str)


def csv(path: Path) -> pd.DataFrame:
    return load_csv(str(path), _mtime(path))


def show_html_file(path: Path, height: int) -> None:
    """Embed a local HTML file. st.iframe is the new API; components.html is the fallback for older Streamlit."""
    if hasattr(st, "iframe"):
        st.iframe(path, height=height)
    else:
        components.html(path.read_text(), height=height)


def artifacts_ready() -> bool:
    return all(p.exists() for p in (cfg.METRICS_FILE, cfg.SEVERITY_RESULTS_FILE, cfg.DBSCAN_FILE))


# --------------------------------------------------------------------------- pages
def page_status(metrics: dict | None) -> None:
    st.header("Project status")
    pct = overall_completion()
    c1, c2 = st.columns([1, 3])
    c1.metric("Overall completion (estimate)", f"{pct:.0f}%")
    c2.progress(pct / 100)
    st.caption("Weighted estimate by the team - see src/project_status.py for the weights and what is done / pending.")
    table = status_table()
    fig = px.bar(table, x="done_pct", y="component", orientation="h", range_x=[0, 100],
                 text="done_pct", labels={"done_pct": "% done", "component": ""})
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)
    st.dataframe(table.rename(columns={"done_pct": "done %", "contribution": "points earned"}), hide_index=True)
    if metrics:
        d = metrics["data"]
        st.subheader("Last pipeline run")
        st.write(f"Data source: **{d['source']}** | rows after cleaning: **{d['cleaning_report']['rows_out']:,}** "
                 f"| runtime: **{metrics['runtime_seconds']} s** | generated: {metrics['generated_at']}")
        st.write("Class distribution:", {k: f"{v['pct']}%" for k, v in d["class_distribution"].items()})
        st.write("Cleaning report:", d["cleaning_report"])


def page_severity(metrics: dict) -> None:
    st.header("Pillar 1 - Severity prediction")
    sev = metrics["severity"]
    best = sev["best"]
    st.info(f"Best configuration: **{best['model']} + {best['strategy']}** "
            f"(rule: {sev['selection_rule']}). Evaluated on {sev['n_test']:,} held-out test crashes.")
    cols = st.columns(5)
    cols[0].metric("Fatal recall", f"{best['fatal_recall']:.3f}")
    cols[1].metric("Fatal precision", f"{best['fatal_precision']:.3f}")
    cols[2].metric("Fatal F2", f"{best['fatal_f2']:.3f}")
    cols[3].metric("Macro F1", f"{best['macro_f1']:.3f}")
    cols[4].metric("ROC-AUC (OvR)", f"{best['roc_auc_ovr']:.3f}")

    res = csv(cfg.SEVERITY_RESULTS_FILE)
    st.subheader("All 12 experiments (4 models x 3 imbalance strategies)")
    st.dataframe(res.sort_values("fatal_f2", ascending=False), hide_index=True)

    st.subheader("What imbalance handling buys us: fatal-class recall")
    fig = px.bar(res, x="model", y="fatal_recall", color="strategy", barmode="group",
                 labels={"fatal_recall": "fatal recall"})
    st.plotly_chart(fig)
    st.caption("Accuracy alone would hide this: with ~5% fatal crashes, a model that rarely predicts 'Fatal' "
               "still looks accurate.")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix (best model)")
        labels = [cfg.SEVERITY_LABELS[i] for i in sorted(cfg.SEVERITY_LABELS)]
        cm = pd.DataFrame(best["confusion_matrix"], index=labels, columns=labels)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Crashes"))
        st.plotly_chart(fig)
    with right:
        st.subheader("Why fatal? Global SHAP importance")
        if cfg.SHAP_FILE.exists():
            shap_df = csv(cfg.SHAP_FILE).head(12).iloc[::-1]
            fig = px.bar(shap_df, x="mean_abs_shap", y="feature", orientation="h",
                         labels={"mean_abs_shap": "mean |SHAP| (fatal class)", "feature": ""})
            st.plotly_chart(fig)
            st.caption(f"Explained model: {sev['shap'].get('model', 'n/a')}. LIME and per-crash explanations are pending.")
        else:
            st.warning(f"SHAP not available ({sev.get('shap', {}).get('status', 'not run')}).")

    if "cross_validation" in sev:
        st.subheader(f"Cross-validation on training data ({sev['cross_validation'][0]['folds']}-fold, "
                     f"{sev['cross_validation'][0]['strategy']})")
        st.dataframe(pd.DataFrame(sev["cross_validation"]).round(3), hide_index=True)


def page_hotspots(metrics: dict) -> None:
    st.header("Pillar 3 - Hotspot detection")
    hs = metrics["hotspots"]
    c = st.columns(4)
    c[0].metric("DBSCAN hotspots", hs["dbscan"]["n_clusters"])
    c[1].metric("Crashes labelled noise", f"{hs['dbscan']['noise_pct']}%")
    c[2].metric("DBSCAN silhouette", "n/a" if hs["dbscan"]["silhouette"] is None else f"{hs['dbscan']['silhouette']:.2f}")
    if "synthetic_recovery" in hs:
        r = hs["synthetic_recovery"]
        c[3].metric("Planted hotspots found", f"{r['recovered']} / {r['planted']}")
    if cfg.MAP_FILE.exists():
        show_html_file(cfg.MAP_FILE, height=560)
        st.caption("Map tiles load from OpenStreetMap, so the browser needs internet access.")
    st.subheader("Ranked black spots (DBSCAN) - severity-weighted: minor 1, serious 3, fatal 10")
    st.dataframe(csv(cfg.DBSCAN_FILE), hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("KDE crash-density surface")
        kde = csv(cfg.KDE_FILE)
        z = kde.pivot(index="latitude", columns="longitude", values="density")
        fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale="YlOrRd"))
        fig.update_layout(xaxis_title="longitude", yaxis_title="latitude", margin=dict(t=10))
        st.plotly_chart(fig)
    with right:
        st.subheader(f"K-Means broad zones (k={hs['kmeans']['k']})")
        st.dataframe(csv(cfg.KMEANS_FILE), hide_index=True)
        st.caption(f"K-Means silhouette: {hs['kmeans']['silhouette']}")


def page_risk(metrics: dict) -> None:
    st.header("Pillar 2 - Risk prediction (PROTOTYPE v0, ~20% done)")
    st.warning("This is a historical severity-weighted score, not yet a trained predictive model. "
               "Risk levels are percentile placeholders. Exposure data and a real model are pending.")
    if not cfg.RISK_FILE.exists():
        st.error("Run `python run_pipeline.py` first.")
        return
    risk = csv(cfg.RISK_FILE)
    c1, c2 = st.columns(2)
    band = c1.selectbox("Time band", cfg.TIME_BAND_LABELS, index=4)
    wx = c2.selectbox("Weather group", ["Normal", "Adverse"])
    sub = risk[(risk["time_band"] == band) & (risk["weather_group"] == wx)].head(15)
    st.write(f"Top 15 grid cells (~1 km) for **{band}**, **{wx}** weather:")
    st.dataframe(sub[["grid_id", "n_crashes", "n_fatal", "weighted_score", "risk_index", "risk_level"]],
                 hide_index=True)
    v = metrics["risk_prototype"]["temporal_validation"]
    st.subheader("Temporal sanity check (past -> future)")
    st.write(f"Score built from crashes up to {v['split_date']}. The top {v['top_k_cells']} cells then contain "
             f"**{v['hit_rate_pct']}%** of later severity-weighted crashes, versus {v['random_baseline_pct']}% "
             f"for random cells (PAI = **{v['pai']}x**).")
    if metrics["data"]["is_synthetic"]:
        st.caption("On synthetic data the hotspots never move, so this is guaranteed to look good. "
                   "It becomes evidence only with real data.")


def page_predict() -> None:
    st.header("Predict a crash (what-if)")
    if not cfg.BEST_MODEL_FILE.exists():
        st.error("No saved model. Run `python run_pipeline.py` first.")
        return
    try:
        model = load_model(str(cfg.BEST_MODEL_FILE), _mtime(cfg.BEST_MODEL_FILE))
    except Exception as exc:  # usually a scikit-learn version mismatch
        st.error(f"Could not load the saved model ({type(exc).__name__}). "
                 "Re-create it with `python run_pipeline.py` using the same Python environment.")
        return
    cv = cfg.CATEGORY_VALUES
    with st.form("predict"):
        a, b, c = st.columns(3)
        road = a.selectbox("Road type", cv["road_type"], index=3)
        speed = a.selectbox("Posted speed limit (km/h)", cfg.SPEED_LIMITS, index=3)
        junction = a.selectbox("Junction", cv["junction_type"])
        weather = b.selectbox("Weather", cv["weather"])
        light = b.selectbox("Light condition", cv["light_condition"])
        surface = b.selectbox("Road surface", cv["road_surface"])
        vehicle = c.selectbox("Vehicle / party", cv["vehicle_type"])
        age = c.slider("Driver age", 16, 90, 35)
        n_veh = c.selectbox("Vehicles involved", [1, 2, 3, 4], index=1)
        d1, d2, d3 = st.columns(3)
        day = d1.date_input("Date", value=date(2024, 1, 15))
        hour = d2.slider("Hour of day", 0, 23, 22)
        gear = d3.checkbox("Helmet / seatbelt used", value=False)
        go_btn = st.form_submit_button("Predict severity")
    if go_btn:
        row = pd.DataFrame([{
            "timestamp": datetime(day.year, day.month, day.day, hour), "road_type": road,
            "speed_limit": speed, "junction_type": junction, "weather": weather, "light_condition": light,
            "road_surface": surface, "vehicle_type": vehicle, "driver_age": age, "num_vehicles": n_veh,
            "safety_gear_used": int(gear),
        }])
        out = predict_severity(model, engineer_features(row))
        probs = out[["Minor", "Serious", "Fatal"]].iloc[0]
        st.metric("Probability of a FATAL outcome", f"{probs['Fatal']:.1%}")
        fig = px.bar(x=probs.index, y=probs.values, labels={"x": "", "y": "probability"})
        st.plotly_chart(fig)
        st.caption("Decision-support only. Trained on synthetic data. Probabilities from models trained with "
                   "class weights / SMOTE are not calibrated - read them as relative scores, not true risk.")


# --------------------------------------------------------------------------- main
def main() -> None:
    st.title("Reducing Traffic Mortality using Machine Learning")
    st.caption("Progress-review build (about 60% complete). Pillars: Severity | Risk | Hotspots")

    pages = ["Project status", "Severity model", "Hotspots", "Risk (prototype)", "Predict a crash"]
    page = st.sidebar.radio("Page", pages)
    st.sidebar.markdown("---")
    st.sidebar.caption("Run `python run_pipeline.py` to refresh all results.")

    if not artifacts_ready():
        page_status(None)
        st.error("No pipeline results found yet. Run `python run_pipeline.py` in the project folder, then reload.")
        return
    metrics = load_json(str(cfg.METRICS_FILE), _mtime(cfg.METRICS_FILE))
    if metrics["data"]["is_synthetic"]:
        st.warning("SYNTHETIC DEMO DATA - results check the pipeline; they are not findings about real roads.")

    if page == pages[0]:
        page_status(metrics)
    elif page == pages[1]:
        page_severity(metrics)
    elif page == pages[2]:
        page_hotspots(metrics)
    elif page == pages[3]:
        page_risk(metrics)
    else:
        page_predict()


main()
