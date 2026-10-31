"""Run the whole pipeline with one command:  python run_pipeline.py

Steps: load -> clean -> features -> severity models -> SHAP -> hotspots -> risk prototype -> save.
Everything the dashboard shows is written to models/ and reports/.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config as cfg
from src.data_generator import get_synthetic_hotspot_centers
from src.data_loader import load_accidents
from src.explain import shap_importance
from src.features import engineer_features
from src.hotspots import (build_folium_map, compute_kde_grid, run_dbscan, run_kmeans,
                          synthetic_recovery)
from src.preprocessing import clean_accidents
from src.project_status import overall_completion
from src.risk import build_risk_table, temporal_validation
from src.severity import cross_validate, run_experiments, save_model, split_data


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Cannot serialise {type(o)}")


def banner(text: str) -> None:
    print(f"\n=== {text} " + "=" * max(0, 60 - len(text)))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", default=None, help="CSV of real accidents (default: synthetic data)")
    ap.add_argument("--n-samples", type=int, default=cfg.DEFAULT_N_SAMPLES, help="synthetic rows")
    ap.add_argument("--cv-folds", type=int, default=cfg.CV_FOLDS, help="0 disables cross-validation")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--quick", action="store_true", help="demo mode: 6000 rows, 2 CV folds")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_samples, args.cv_folds = 6000, 2

    t_start = time.time()
    for d in (cfg.DATA_RAW, cfg.DATA_PROCESSED, cfg.MODELS_DIR, cfg.REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # ------------------------------------------------------------------ data
    banner("1/6 Load, clean, engineer features")
    raw, source = load_accidents(args.data_path, args.n_samples)
    clean, report = clean_accidents(raw)
    df = engineer_features(clean)
    df.to_csv(cfg.CLEAN_DATA_FILE, index=False)
    dist = df[cfg.TARGET].value_counts().sort_index()
    metrics["data"] = {
        "source": source, "is_synthetic": source == "synthetic", "cleaning_report": report,
        "class_distribution": {cfg.SEVERITY_LABELS[int(k)]: {"count": int(v), "pct": round(100 * v / len(df), 2)}
                               for k, v in dist.items()},
    }
    print(f"source={source}  rows in={report['rows_in']}  rows out={report['rows_out']}")
    print("cleaning:", {k: v for k, v in report.items() if k.endswith("removed")})
    print("classes :", {k: v["pct"] for k, v in metrics["data"]["class_distribution"].items()})

    # ------------------------------------------------------------------ severity
    banner("2/6 Severity models (4 models x 3 imbalance strategies)")
    X_train, X_test, y_train, y_test = split_data(df)
    exp = run_experiments(X_train, X_test, y_train, y_test)
    results = pd.DataFrame(exp["results"])
    results.drop(columns="confusion_matrix").round(4).to_csv(cfg.SEVERITY_RESULTS_FILE, index=False)
    best = exp["best"]
    save_model(best["pipe"])
    metrics["severity"] = {
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "selection_rule": "highest fatal-class F2 (recall weighted 4x precision); ties -> macro F1",
        "best": best["row"], "saved_model": cfg.BEST_MODEL_FILE.name,
    }
    print(f"best: {best['row']['model']} + {best['row']['strategy']}  "
          f"fatal recall={best['row']['fatal_recall']:.3f}")

    if args.cv_folds >= 2:
        banner(f"   cross-validation ({args.cv_folds}-fold, strategy={cfg.CV_STRATEGY})")
        metrics["severity"]["cross_validation"] = cross_validate(X_train, y_train, folds=args.cv_folds)

    # ------------------------------------------------------------------ SHAP
    banner("3/6 SHAP (fatal class)")
    if args.skip_shap:
        metrics["severity"]["shap"] = {"status": "skipped"}
        print("skipped")
    else:
        try:
            tree = exp["best_tree"]
            sample = X_test.sample(n=min(cfg.SHAP_SAMPLE_SIZE, len(X_test)), random_state=cfg.RANDOM_STATE)
            imp = shap_importance(tree["pipe"], sample)
            imp.round(5).to_csv(cfg.SHAP_FILE, index=False)
            metrics["severity"]["shap"] = {"status": "ok", "model": f"{tree['row']['model']} + {tree['row']['strategy']}",
                                           "top_features": imp.head(5)["feature"].tolist()}
            print("top features:", metrics["severity"]["shap"]["top_features"])
        except Exception as exc:  # SHAP/library mismatch must not kill the whole pipeline
            metrics["severity"]["shap"] = {"status": f"failed: {type(exc).__name__}: {exc}"}
            print("SHAP failed:", exc)

    # ------------------------------------------------------------------ hotspots
    banner("4/6 Hotspots (K-Means, DBSCAN, KDE)")
    _, zones, km_sil = run_kmeans(df)
    zones.to_csv(cfg.KMEANS_FILE, index=False)
    labels, ranked, db_info = run_dbscan(df)
    ranked.to_csv(cfg.DBSCAN_FILE, index=False)
    compute_kde_grid(df).round(5).to_csv(cfg.KDE_FILE, index=False)
    build_folium_map(df, ranked).save(str(cfg.MAP_FILE))
    metrics["hotspots"] = {"kmeans": {"k": cfg.KMEANS_K, "silhouette": round(km_sil, 3)},
                           "dbscan": db_info, "kde": {"bandwidth_km": cfg.KDE_BANDWIDTH_KM}}
    if source == "synthetic":
        metrics["hotspots"]["synthetic_recovery"] = synthetic_recovery(ranked, get_synthetic_hotspot_centers())
    print("dbscan:", db_info)
    print("recovery:", metrics["hotspots"].get("synthetic_recovery"))

    # ------------------------------------------------------------------ risk prototype
    banner("5/6 Risk prototype v0")
    risk = build_risk_table(df)
    risk.to_csv(cfg.RISK_FILE, index=False)
    metrics["risk_prototype"] = {"n_rows": int(len(risk)), "temporal_validation": temporal_validation(df)}
    print(metrics["risk_prototype"])

    # ------------------------------------------------------------------ save
    banner("6/6 Save")
    metrics["project_completion_pct"] = overall_completion()
    metrics["runtime_seconds"] = round(time.time() - t_start, 1)
    cfg.METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=_jsonable))
    print(f"wrote {cfg.METRICS_FILE.relative_to(cfg.ROOT)}  (runtime {metrics['runtime_seconds']}s)")
    print("\nNext:  streamlit run app/dashboard.py")
    return metrics


if __name__ == "__main__":
    main()
