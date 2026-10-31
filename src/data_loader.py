"""Load accident data: a real CSV if you give a path, otherwise the synthetic generator."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as cfg
from .data_generator import generate_synthetic_accidents


def load_accidents(path: str | Path | None = None, n_samples: int = cfg.DEFAULT_N_SAMPLES,
                   save_synthetic: bool = True) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source_label).

    * path given   -> read that CSV (it must follow the schema in docs/DATA_FLOW.md)
    * path missing -> generate synthetic data and (optionally) save it to data/raw/
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        return pd.read_csv(p), f"real:{p.name}"

    df = generate_synthetic_accidents(n_samples)
    if save_synthetic:
        cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.RAW_DATA_FILE, index=False)
    return df, "synthetic"
