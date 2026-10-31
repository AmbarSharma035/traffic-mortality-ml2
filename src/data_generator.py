"""Synthetic accident data generator.

WHY THIS EXISTS: the real dataset (MoRTH / STATS19 / Kaggle) is not integrated yet (see
docs/PROJECT_OVERVIEW.md, project status). To build and test the whole pipeline end to end,
we generate data with a *known* structure:

* 8 planted hotspots (so we can check that clustering finds them),
* a hidden "danger score" that depends on speed, darkness, weather, vehicle, safety gear...
  (so severity models have real signal to learn),
* ~5% fatal / ~20% serious / ~75% minor crashes (class imbalance like real data),
* a little deliberate dirt: missing values, duplicates, invalid coordinates (so the cleaning
  step has something to do).

Results on this data are a *pipeline check*, NOT scientific findings about real roads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .hotspots import haversine_km

_SPEED_BY_ROAD = {
    "Expressway": [80, 100],
    "National Highway": [60, 80, 100],
    "State Highway": [50, 60, 80],
    "Urban Arterial": [40, 50, 60],
    "Local Road": [20, 30, 40],
}
_ROAD_EFFECT = {"Expressway": 0.50, "National Highway": 0.40, "State Highway": 0.25,
                "Urban Arterial": 0.0, "Local Road": -0.30}
_LIGHT_EFFECT = {"Daylight": 0.0, "Dusk/Dawn": 0.15, "Dark - lit": 0.25, "Dark - unlit": 0.70}
_WEATHER_EFFECT = {"Clear": 0.0, "Haze": 0.20, "Rain": 0.25, "Fog": 0.50, "Storm": 0.40}
_SURFACE_EFFECT = {"Dry": 0.0, "Wet": 0.15, "Damaged": 0.20}
_VEHICLE_EFFECT = {"Car": 0.0, "Auto-rickshaw": 0.20, "Bus": 0.30, "Two-wheeler": 0.50,
                   "Truck": 0.50, "Pedestrian-involved": 0.90}
_JUNCTION_EFFECT = {"Mid-block": 0.20, "T-Junction": 0.0, "Crossroads": 0.05, "Roundabout": -0.20}


def get_synthetic_hotspot_centers() -> np.ndarray:
    """The planted hotspot centres as a (k, 2) array of [lat, lon].

    Fixed seed, so tests can check recovery. Centres are at least HOTSPOT_MIN_SEPARATION_KM apart,
    otherwise two planted hotspots would overlap and look like one cluster.
    """
    rng = np.random.default_rng(cfg.HOTSPOT_SEED)
    b = cfg.BBOX
    centers: list[tuple[float, float]] = []
    for _ in range(10_000):
        cand = (rng.uniform(b["lat_min"] + 0.06, b["lat_max"] - 0.06),
                rng.uniform(b["lon_min"] + 0.06, b["lon_max"] - 0.06))
        if all(haversine_km(cand[0], cand[1], c[0], c[1]) >= cfg.HOTSPOT_MIN_SEPARATION_KM for c in centers):
            centers.append(cand)
        if len(centers) == cfg.N_SYNTHETIC_HOTSPOTS:
            break
    return np.array(centers)


def _map_values(values: np.ndarray, effect: dict) -> np.ndarray:
    return np.array([effect[v] for v in values], dtype=float)


def generate_synthetic_accidents(n: int = cfg.DEFAULT_N_SAMPLES, seed: int = cfg.RANDOM_STATE,
                                 inject_noise: bool = True) -> pd.DataFrame:
    """Return a DataFrame with the columns listed in config.REQUIRED_COLUMNS."""
    rng = np.random.default_rng(seed)
    b = cfg.BBOX

    # ---------------------------------------------------------------- location
    centers = get_synthetic_hotspot_centers()
    weights = np.random.default_rng(cfg.HOTSPOT_SEED + 1).dirichlet(np.ones(len(centers)) * 4)
    in_hotspot = rng.random(n) < cfg.HOTSPOT_SHARE
    which = rng.choice(len(centers), size=n, p=weights)
    sigma_lat = cfg.HOTSPOT_SIGMA_KM / 110.574
    sigma_lon = cfg.HOTSPOT_SIGMA_KM / (111.320 * np.cos(np.radians(centers[:, 0].mean())))
    lat = np.where(in_hotspot, centers[which, 0] + rng.normal(0, sigma_lat, n),
                   rng.uniform(b["lat_min"], b["lat_max"], n))
    lon = np.where(in_hotspot, centers[which, 1] + rng.normal(0, sigma_lon, n),
                   rng.uniform(b["lon_min"], b["lon_max"], n))
    lat = np.clip(lat, b["lat_min"], b["lat_max"])
    lon = np.clip(lon, b["lon_min"], b["lon_max"])

    # ---------------------------------------------------------------- time (2022-2023, rush-hour peaks)
    hours = np.arange(24)
    p_hour = (1.0 + 2.5 * np.exp(-((hours - 9) ** 2) / (2 * 1.5 ** 2))
              + 3.0 * np.exp(-((hours - 18) ** 2) / (2 * 1.8 ** 2)))
    p_hour[1:5] *= 0.6
    p_hour = p_hour / p_hour.sum()
    hour = rng.choice(hours, size=n, p=p_hour)
    day = rng.integers(0, 730, size=n)
    ts = (pd.Timestamp("2022-01-01") + pd.to_timedelta(day, unit="D")
          + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(rng.integers(0, 60, n), unit="m"))
    month = ts.month.to_numpy()

    # ---------------------------------------------------------------- road, weather, light
    road_type = rng.choice(cfg.CATEGORY_VALUES["road_type"], size=n, p=[0.08, 0.17, 0.15, 0.35, 0.25])
    speed = np.array([rng.choice(_SPEED_BY_ROAD[r]) for r in road_type])
    junction = rng.choice(cfg.CATEGORY_VALUES["junction_type"], size=n, p=[0.45, 0.25, 0.22, 0.08])

    weather = np.empty(n, dtype=object)
    season_p = {  # Clear, Haze, Rain, Fog, Storm
        "winter": [0.45, 0.20, 0.05, 0.27, 0.03],
        "monsoon": [0.45, 0.10, 0.38, 0.00, 0.07],
        "other": [0.65, 0.17, 0.12, 0.03, 0.03],
    }
    season = np.where(np.isin(month, [12, 1, 2]), "winter", np.where(np.isin(month, [7, 8, 9]), "monsoon", "other"))
    for s, p in season_p.items():
        mask = season == s
        weather[mask] = rng.choice(cfg.CATEGORY_VALUES["weather"], size=mask.sum(), p=p)

    light = np.where(
        (hour >= 6) & (hour <= 17), "Daylight",
        np.where((hour == 5) | (hour == 18), "Dusk/Dawn",
                 np.where(rng.random(n) < 0.65, "Dark - lit", "Dark - unlit")))
    wet = np.isin(weather, ["Rain", "Storm"]) & (rng.random(n) < 0.85)
    surface = np.where(wet, "Wet", np.where(rng.random(n) < 0.08, "Damaged", "Dry"))

    # ---------------------------------------------------------------- people and vehicles
    vehicle = rng.choice(cfg.CATEGORY_VALUES["vehicle_type"], size=n, p=[0.34, 0.27, 0.08, 0.05, 0.12, 0.14])
    age = np.clip(rng.normal(36, 12, n), 18, 80).round()
    num_vehicles = rng.choice([1, 2, 3, 4], size=n, p=[0.25, 0.55, 0.15, 0.05])
    gear = (rng.random(n) < 0.62).astype(int)
    gear[vehicle == "Pedestrian-involved"] = 0

    # ---------------------------------------------------------------- hidden danger score -> severity
    z = (-1.2
         + 0.025 * (speed - 50)
         + _map_values(road_type, _ROAD_EFFECT)
         + _map_values(light, _LIGHT_EFFECT)
         + _map_values(weather, _WEATHER_EFFECT)
         + _map_values(surface, _SURFACE_EFFECT)
         + _map_values(vehicle, _VEHICLE_EFFECT)
         + _map_values(junction, _JUNCTION_EFFECT)
         + 0.60 * (gear == 0)
         + 0.40 * (age > 60) + 0.20 * (age < 22)
         + 0.20 * (num_vehicles >= 3)
         + 0.15 * in_hotspot
         + rng.normal(0, 0.9, n))
    t_serious, t_fatal = np.quantile(z, [0.75, 0.95])  # -> ~75% minor, 20% serious, 5% fatal
    severity = np.where(z >= t_fatal, 2, np.where(z >= t_serious, 1, 0)).astype(float)

    df = pd.DataFrame({
        "accident_id": [f"ACC{i:07d}" for i in range(n)],
        "timestamp": ts, "latitude": lat.round(6), "longitude": lon.round(6),
        "road_type": road_type, "speed_limit": speed, "junction_type": junction,
        "weather": weather, "light_condition": light, "road_surface": surface,
        "vehicle_type": vehicle, "driver_age": age, "num_vehicles": num_vehicles,
        "safety_gear_used": gear, "severity": severity,
    })

    if inject_noise:
        df = _inject_noise(df, rng)
    return df[cfg.REQUIRED_COLUMNS]


def _inject_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add realistic dirt so that the cleaning step is actually exercised."""
    df = df.copy()
    n = len(df)
    for col, frac in [("weather", 0.015), ("light_condition", 0.010),
                      ("road_surface", 0.010), ("driver_age", 0.020)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    bad = rng.random(n)
    df.loc[bad < 0.0015, ["latitude", "longitude"]] = 0.0          # "null island"
    df.loc[(bad >= 0.0015) & (bad < 0.0020), "latitude"] = 999.0   # impossible latitude
    df.loc[rng.random(n) < 0.001, "severity"] = np.nan             # missing label
    dupes = df.sample(frac=0.003, random_state=int(rng.integers(0, 10_000)))
    return pd.concat([df, dupes], ignore_index=True)
