"""Tests for the ML pillars: severity, hotspots, risk, project status."""
import numpy as np
import pytest

from src import config as cfg
from src.data_generator import generate_synthetic_accidents, get_synthetic_hotspot_centers
from src.features import engineer_features
from src.hotspots import compute_kde_grid, run_dbscan, run_kmeans, synthetic_recovery
from src.preprocessing import clean_accidents
from src.project_status import overall_completion
from src.risk import build_risk_table, temporal_validation
from src.severity import (cross_validate, evaluate, fit_pipeline, make_pipeline, predict_severity,
                          split_data)


@pytest.fixture(scope="module")
def df():
    clean, _ = clean_accidents(generate_synthetic_accidents(n=4000, seed=5))
    return engineer_features(clean)


# ----------------------------------------------------------------- severity
@pytest.mark.parametrize("model,strategy", [("logistic_regression", "smote"), ("lightgbm", "class_weight"),
                                            ("xgboost", "class_weight"), ("random_forest", "none")])
def test_severity_pipeline_trains_and_predicts(df, model, strategy):
    X_tr, X_te, y_tr, y_te = split_data(df)
    pipe = fit_pipeline(make_pipeline(model, strategy), X_tr, y_tr, model, strategy)
    m = evaluate(pipe, X_te, y_te)
    assert 0 <= m["fatal_recall"] <= 1 and 0 <= m["macro_f1"] <= 1
    assert m["roc_auc_ovr"] > 0.6  # the synthetic data has real signal
    out = predict_severity(pipe, X_te.head(5))
    assert np.allclose(out[["Minor", "Serious", "Fatal"]].sum(axis=1), 1.0)


def test_split_is_stratified(df):
    _, _, y_tr, y_te = split_data(df)
    assert abs((y_tr == 2).mean() - (y_te == 2).mean()) < 0.01


def test_class_weighting_raises_fatal_recall(df):
    X_tr, X_te, y_tr, y_te = split_data(df)
    rec = {}
    for s in ("none", "class_weight"):
        p = fit_pipeline(make_pipeline("logistic_regression", s), X_tr, y_tr, "logistic_regression", s)
        rec[s] = evaluate(p, X_te, y_te)["fatal_recall"]
    assert rec["class_weight"] > rec["none"]


def test_cross_validate_returns_one_row_per_model(df):
    X_tr, _, y_tr, _ = split_data(df)
    rows = cross_validate(X_tr, y_tr, models=["logistic_regression"], folds=2, verbose=False)
    assert len(rows) == 1 and 0 <= rows[0]["fatal_recall_mean"] <= 1


def test_unknown_model_or_strategy_raises():
    with pytest.raises(ValueError):
        make_pipeline("svm", "none")
    with pytest.raises(ValueError):
        make_pipeline("lightgbm", "magic")


# ----------------------------------------------------------------- hotspots
def test_dbscan_finds_and_ranks_planted_hotspots(df):
    labels, ranked, info = run_dbscan(df)
    assert info["n_clusters"] >= 5
    assert ranked["severity_score"].is_monotonic_decreasing
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
    rec = synthetic_recovery(ranked, get_synthetic_hotspot_centers())
    assert rec["recovered"] >= 6


def test_kmeans_and_kde_shapes(df):
    labels, zones, sil = run_kmeans(df, k=4)
    assert len(zones) == 4 and -1 <= sil <= 1
    kde = compute_kde_grid(df, grid_size=20)
    assert len(kde) == 400 and kde["density"].max() == pytest.approx(1.0)


# ----------------------------------------------------------------- risk + status
def test_risk_table_and_validation(df):
    table = build_risk_table(df)
    assert table["risk_index"].between(0, 100).all()
    assert set(table["risk_level"]) <= {"Low", "Medium", "High", "Critical"}
    v = temporal_validation(df)
    assert v["pai"] > 1  # better than random on (stationary) synthetic data


def test_overall_completion_is_about_sixty_percent():
    assert 58 <= overall_completion() <= 62
