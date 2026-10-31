"""Tests for the data layer: generator, cleaning, feature engineering."""
import pandas as pd
import pytest

from src import config as cfg
from src.data_generator import generate_synthetic_accidents, get_synthetic_hotspot_centers
from src.features import engineer_features
from src.hotspots import haversine_km
from src.preprocessing import clean_accidents


@pytest.fixture(scope="module")
def raw():
    return generate_synthetic_accidents(n=4000, seed=1)


def test_generator_has_required_columns(raw):
    assert list(raw.columns) == cfg.REQUIRED_COLUMNS


def test_generator_is_deterministic():
    a = generate_synthetic_accidents(n=500, seed=3)
    b = generate_synthetic_accidents(n=500, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_generator_class_balance_is_imbalanced(raw):
    clean, _ = clean_accidents(raw)
    fatal_share = (clean[cfg.TARGET] == cfg.FATAL_CLASS).mean()
    assert 0.03 < fatal_share < 0.07


def test_planted_hotspots_are_separated():
    c = get_synthetic_hotspot_centers()
    assert len(c) == cfg.N_SYNTHETIC_HOTSPOTS
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            assert haversine_km(*c[i], *c[j]) >= cfg.HOTSPOT_MIN_SEPARATION_KM


def test_cleaning_removes_dirt_and_reports_it(raw):
    clean, report = clean_accidents(raw)
    assert report["rows_out"] == len(clean) < report["rows_in"]
    assert clean["accident_id"].is_unique
    assert clean[cfg.TARGET].isin([0, 1, 2]).all()
    assert clean["latitude"].between(-90, 90).all()
    assert not ((clean["latitude"] == 0) & (clean["longitude"] == 0)).any()


def test_cleaning_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        clean_accidents(pd.DataFrame({"accident_id": [1]}))


def test_feature_flags_are_correct():
    df = pd.DataFrame({"timestamp": ["2024-01-13 08:30", "2024-01-15 23:10", "2024-01-15 13:00"],
                       "weather": ["Fog", "Clear", None]})
    out = engineer_features(df)
    assert out["is_weekend"].tolist() == [1, 0, 0]       # 13 Jan 2024 is a Saturday
    assert out["is_rush_hour"].tolist() == [1, 0, 0]
    assert out["is_night"].tolist() == [0, 1, 0]
    assert out["weather_severity_index"].iloc[0] == 3
    assert pd.isna(out["weather_severity_index"].iloc[2])  # unknown stays NaN (imputed later, on train only)
