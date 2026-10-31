"""Pillar 3 - Accident Hotspot Detection (unsupervised, location only).

* K-Means  -> broad zones (you choose k)
* DBSCAN   -> irregular dense "black spots" (you choose a radius + minimum crashes; noise allowed)
* KDE      -> a smooth crash-density surface for a heat-map

Coordinates are converted from degrees to kilometres first (a simple local projection), so
"eps = 0.3" really means 300 metres. Clusters are ranked by a severity-weighted score
(minor=1, serious=3, fatal=10), not just by crash count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity

from . import config as cfg

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQ = 111.320


# --------------------------------------------------------------------------- geometry helpers
def to_km(lat, lon, lat0: float | None = None, lon0: float | None = None):
    """Project lat/lon (degrees) to local x/y in km. Returns (xy_array, (lat0, lon0))."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    lat0 = float(lat.mean()) if lat0 is None else lat0
    lon0 = float(lon.mean()) if lon0 is None else lon0
    x = (lon - lon0) * KM_PER_DEG_LON_EQ * np.cos(np.radians(lat0))
    y = (lat - lat0) * KM_PER_DEG_LAT
    return np.column_stack([x, y]), (lat0, lon0)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points (degrees)."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def severity_weight(df: pd.DataFrame) -> pd.Series:
    return df[cfg.TARGET].map(cfg.SEVERITY_WEIGHTS)


# --------------------------------------------------------------------------- ranking table
def summarise_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """One row per cluster (noise = -1 is ignored), ranked by severity-weighted score."""
    d = df.assign(cluster=labels, weight=severity_weight(df),
                  is_fatal=(df[cfg.TARGET] == cfg.FATAL_CLASS).astype(int),
                  is_serious=(df[cfg.TARGET] == 1).astype(int))
    d = d[d["cluster"] >= 0]
    out = d.groupby("cluster").agg(
        n_crashes=("weight", "size"), n_fatal=("is_fatal", "sum"), n_serious=("is_serious", "sum"),
        centroid_lat=("latitude", "mean"), centroid_lon=("longitude", "mean"),
        severity_score=("weight", "sum"),
    ).reset_index()
    out["fatal_share_pct"] = (100 * out["n_fatal"] / out["n_crashes"]).round(2)
    out = out.sort_values("severity_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# --------------------------------------------------------------------------- algorithms
def run_kmeans(df: pd.DataFrame, k: int = cfg.KMEANS_K, silhouette_sample: int = 5000):
    """Broad zones. Returns (labels, zones_table, silhouette)."""
    xy, _ = to_km(df["latitude"], df["longitude"])
    km = KMeans(n_clusters=k, n_init=10, random_state=cfg.RANDOM_STATE).fit(xy)
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    idx = rng.choice(len(xy), size=min(silhouette_sample, len(xy)), replace=False)
    sil = float(silhouette_score(xy[idx], km.labels_[idx]))
    return km.labels_, summarise_clusters(df, km.labels_), sil


def run_dbscan(df: pd.DataFrame, eps_km: float = cfg.DBSCAN_EPS_KM,
               min_samples: int = cfg.DBSCAN_MIN_SAMPLES, silhouette_sample: int = 5000):
    """Dense black spots. Returns (labels, ranked_table, info_dict)."""
    xy, _ = to_km(df["latitude"], df["longitude"])
    labels = DBSCAN(eps=eps_km, min_samples=min_samples).fit_predict(xy)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    info = {"eps_km": eps_km, "min_samples": min_samples, "n_clusters": n_clusters,
            "noise_pct": round(float((labels == -1).mean() * 100), 2), "silhouette": None}
    core = labels >= 0
    if n_clusters >= 2 and core.sum() > 10:
        idx = np.flatnonzero(core)
        rng = np.random.default_rng(cfg.RANDOM_STATE)
        idx = rng.choice(idx, size=min(silhouette_sample, len(idx)), replace=False)
        info["silhouette"] = float(silhouette_score(xy[idx], labels[idx]))
    return labels, summarise_clusters(df, labels), info


def compute_kde_grid(df: pd.DataFrame, bandwidth_km: float = cfg.KDE_BANDWIDTH_KM,
                     grid_size: int = cfg.KDE_GRID_SIZE, max_points: int = 20_000) -> pd.DataFrame:
    """Smooth crash density on a grid_size x grid_size lat/lon grid (density scaled 0..1)."""
    d = df.sample(n=min(len(df), max_points), random_state=cfg.RANDOM_STATE)
    xy, (lat0, lon0) = to_km(d["latitude"], d["longitude"])
    kde = KernelDensity(bandwidth=bandwidth_km, kernel="gaussian", rtol=1e-3).fit(xy)

    lat_g = np.linspace(df["latitude"].min(), df["latitude"].max(), grid_size)
    lon_g = np.linspace(df["longitude"].min(), df["longitude"].max(), grid_size)
    lon_m, lat_m = np.meshgrid(lon_g, lat_g)
    grid_xy, _ = to_km(lat_m.ravel(), lon_m.ravel(), lat0, lon0)
    dens = np.exp(kde.score_samples(grid_xy))
    return pd.DataFrame({"latitude": lat_m.ravel(), "longitude": lon_m.ravel(),
                         "density": dens / dens.max()})


# --------------------------------------------------------------------------- map
def build_folium_map(df: pd.DataFrame, ranked: pd.DataFrame, max_heat_points: int = 6000, top_n: int = 15):
    """Interactive Leaflet map: severity-weighted heat layer + ranked DBSCAN hotspot markers."""
    import folium
    from folium.plugins import HeatMap

    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=10,
                   tiles="OpenStreetMap")
    sample = df.sample(n=min(len(df), max_heat_points), random_state=cfg.RANDOM_STATE)
    w = severity_weight(sample).astype(float)
    heat = np.column_stack([sample["latitude"], sample["longitude"], w / w.max()]).tolist()
    HeatMap(heat, name="Crash density (severity-weighted sample)", radius=8, blur=12).add_to(m)

    layer = folium.FeatureGroup(name="DBSCAN hotspots (ranked)")
    for r in ranked.head(top_n).itertuples():
        color = "red" if r.rank <= 3 else "orange" if r.rank <= 6 else "blue"
        popup = (f"<b>Hotspot #{r.rank}</b><br>Crashes: {r.n_crashes}<br>Fatal: {r.n_fatal} "
                 f"({r.fatal_share_pct}%)<br>Severity score: {r.severity_score}")
        folium.CircleMarker([r.centroid_lat, r.centroid_lon], radius=9, color=color, fill=True,
                            fill_opacity=0.7, popup=folium.Popup(popup, max_width=250),
                            tooltip=f"#{r.rank}").add_to(layer)
    layer.add_to(m)
    folium.LayerControl().add_to(m)
    return m


# --------------------------------------------------------------------------- sanity check (synthetic only)
def synthetic_recovery(ranked: pd.DataFrame, centers: np.ndarray,
                       tol_km: float = cfg.SYNTHETIC_RECOVERY_TOL_KM) -> dict:
    """How many planted synthetic centres are matched by a detected hotspot within tol_km?

    One-to-one: each detected hotspot can be matched to at most one planted centre.
    """
    det_lat = ranked["centroid_lat"].to_numpy()
    det_lon = ranked["centroid_lon"].to_numpy()
    used = np.zeros(len(det_lat), dtype=bool)
    found = 0
    for lat, lon in centers:
        if len(det_lat) == 0:
            break
        d = haversine_km(lat, lon, det_lat, det_lon)
        d = np.where(used, np.inf, d)
        j = int(d.argmin())
        if d[j] <= tol_km:
            used[j] = True
            found += 1
    return {"planted": int(len(centers)), "recovered": found, "tolerance_km": tol_km}
