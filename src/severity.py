"""Pillar 1 - Accident Severity Prediction (Minor / Serious / Fatal).

We train 4 models x 3 imbalance strategies and compare them with FATAL-class metrics, because a
missed fatal crash costs far more than a false alarm. Each experiment is one scikit-learn
Pipeline: imputing + encoding -> (optional SMOTE) -> classifier. SMOTE lives *inside* the
pipeline, so it only ever touches training data (never the test split or a validation fold).
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config as cfg

RS = cfg.RANDOM_STATE


# --------------------------------------------------------------------------- building blocks
def build_preprocessor() -> ColumnTransformer:
    """Numbers: median-impute + scale.  Categories: fill 'Unknown' + one-hot encode."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, cfg.NUMERIC_FEATURES),
        ("cat", categorical, cfg.CATEGORICAL_FEATURES),
    ])


def build_classifier(name: str, strategy: str):
    """Return an untrained classifier. 'class_weight' strategy makes rare classes count more."""
    cw = "balanced" if strategy == "class_weight" else None
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight=cw, random_state=RS)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5,
                                      class_weight=cw, n_jobs=-1, random_state=RS)
    if name == "xgboost":  # class weights are passed as sample weights in fit_pipeline()
        return XGBClassifier(objective="multi:softprob", n_estimators=150, max_depth=5,
                             learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", n_jobs=-1, random_state=RS, eval_metric="mlogloss")
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.1, class_weight=cw,
                              n_jobs=-1, random_state=RS, verbose=-1)
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(name: str, strategy: str) -> ImbPipeline:
    if strategy not in cfg.IMBALANCE_STRATEGIES:
        raise ValueError(f"Unknown imbalance strategy: {strategy}")
    steps = [("prep", build_preprocessor())]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=RS, k_neighbors=5)))
    steps.append(("clf", build_classifier(name, strategy)))
    return ImbPipeline(steps)


def fit_pipeline(pipe: ImbPipeline, X: pd.DataFrame, y: pd.Series, name: str, strategy: str) -> ImbPipeline:
    if name == "xgboost" and strategy == "class_weight":
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe


# --------------------------------------------------------------------------- data split
def split_data(df: pd.DataFrame):
    """Stratified 80/20 split, so the fatal ratio is the same in train and test."""
    X, y = df[cfg.FEATURE_COLUMNS], df[cfg.TARGET]
    return train_test_split(X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=RS)


# --------------------------------------------------------------------------- evaluation
def evaluate(pipe: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipe.predict_proba(X_test)
    pred = np.asarray(pipe.classes_)[proba.argmax(axis=1)]
    is_fatal, pred_fatal = (y_test == cfg.FATAL_CLASS), (pred == cfg.FATAL_CLASS)
    fatal_col = list(pipe.classes_).index(cfg.FATAL_CLASS)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "fatal_recall": recall_score(is_fatal, pred_fatal),
        "fatal_precision": precision_score(is_fatal, pred_fatal, zero_division=0),
        "fatal_f1": f1_score(is_fatal, pred_fatal, zero_division=0),
        "fatal_f2": fbeta_score(is_fatal, pred_fatal, beta=2, zero_division=0),
        "fatal_pr_auc": average_precision_score(is_fatal, proba[:, fatal_col]),
        "roc_auc_ovr": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(cfg.SEVERITY_LABELS)).tolist(),
    }


def run_experiments(X_train, X_test, y_train, y_test, models=None, strategies=None, verbose=True) -> dict:
    """Train every (model, strategy) pair. Returns results plus the best pipelines.

    Best = highest fatal-class F2 (recall counts 4x more than precision), ties broken by macro-F1.
    """
    models = models or cfg.MODEL_NAMES
    strategies = strategies or cfg.IMBALANCE_STRATEGIES
    results, best, best_tree = [], None, None

    for name in models:
        for strategy in strategies:
            t0 = time.time()
            pipe = fit_pipeline(make_pipeline(name, strategy), X_train, y_train, name, strategy)
            metrics = evaluate(pipe, X_test, y_test)
            row = {"model": name, "strategy": strategy, **metrics,
                   "fit_seconds": round(time.time() - t0, 1)}
            results.append(row)
            if verbose:
                print(f"  {name:<20} {strategy:<13} fatal recall={row['fatal_recall']:.3f} "
                      f"precision={row['fatal_precision']:.3f} macroF1={row['macro_f1']:.3f} "
                      f"({row['fit_seconds']}s)")
            key = (row["fatal_f2"], row["macro_f1"])
            if best is None or key > best["key"]:
                best = {"key": key, "pipe": pipe, "row": row}
            if name in cfg.TREE_MODELS and (best_tree is None or key > best_tree["key"]):
                best_tree = {"key": key, "pipe": pipe, "row": row}
    return {"results": results, "best": best, "best_tree": best_tree}


def cross_validate(X, y, models=None, strategy: str = cfg.CV_STRATEGY, folds: int = cfg.CV_FOLDS,
                   verbose=True) -> list[dict]:
    """Stratified k-fold on the TRAINING data. Reports mean +/- std of fatal recall / F2 / macro-F1."""
    models = models or cfg.MODEL_NAMES
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RS)
    rows = []
    for name in models:
        rec, f2, mf1 = [], [], []
        for tr, va in skf.split(X, y):
            pipe = fit_pipeline(make_pipeline(name, strategy), X.iloc[tr], y.iloc[tr], name, strategy)
            m = evaluate(pipe, X.iloc[va], y.iloc[va])
            rec.append(m["fatal_recall"]); f2.append(m["fatal_f2"]); mf1.append(m["macro_f1"])
        rows.append({"model": name, "strategy": strategy, "folds": folds,
                     "fatal_recall_mean": float(np.mean(rec)), "fatal_recall_std": float(np.std(rec)),
                     "fatal_f2_mean": float(np.mean(f2)), "macro_f1_mean": float(np.mean(mf1))})
        if verbose:
            print(f"  CV {name:<20} fatal recall {np.mean(rec):.3f} +/- {np.std(rec):.3f}")
    return rows


# --------------------------------------------------------------------------- inference
def save_model(pipe, path=cfg.BEST_MODEL_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path, compress=3)


def load_model(path=cfg.BEST_MODEL_FILE):
    return joblib.load(path)


def predict_severity(pipe, features: pd.DataFrame) -> pd.DataFrame:
    """Class probabilities (Minor / Serious / Fatal) plus the most likely class."""
    proba = pipe.predict_proba(features[cfg.FEATURE_COLUMNS])
    cols = [cfg.SEVERITY_LABELS[int(c)] for c in pipe.classes_]
    out = pd.DataFrame(proba, columns=cols, index=features.index)
    out["predicted"] = out[cols].idxmax(axis=1)
    return out
