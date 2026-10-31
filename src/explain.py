"""Explainability: global SHAP importance for the FATAL class (about half of the planned layer).

Done: SHAP (TreeExplainer) for tree models, with one-hot columns merged back into the original
feature (so "weather" shows as one bar, not five).
Pending: LIME and per-prediction explanations in the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def _select_class(shap_values, class_index: int) -> np.ndarray:
    """SHAP returns a list (old versions) or a 3-D array (new versions) for multi-class models."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])
    arr = np.asarray(shap_values)
    return arr[:, :, class_index] if arr.ndim == 3 else arr


def _original_name(transformed_name: str) -> str:
    kind, _, rest = transformed_name.partition("__")
    if kind == "cat":
        for col in cfg.CATEGORICAL_FEATURES:
            if rest.startswith(col + "_"):
                return col
    return rest


def shap_importance(pipe, X_sample: pd.DataFrame, class_index: int = cfg.FATAL_CLASS) -> pd.DataFrame:
    """Mean |SHAP| per original feature for one class. Needs a tree-based classifier in the pipeline."""
    import shap

    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Xt = prep.transform(X_sample[cfg.FEATURE_COLUMNS])
    names = prep.get_feature_names_out()
    values = _select_class(shap.TreeExplainer(clf).shap_values(Xt), class_index)

    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        groups.setdefault(_original_name(n), []).append(i)
    # sum SHAP over a feature's one-hot columns *per row*, then take the mean absolute value
    rows = [{"feature": f, "mean_abs_shap": float(np.abs(values[:, idx].sum(axis=1)).mean())}
            for f, idx in groups.items()]
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
