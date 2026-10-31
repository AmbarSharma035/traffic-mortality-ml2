"""Central configuration. Change values here instead of hunting through the code."""
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RAW_DATA_FILE = DATA_RAW / "accidents_synthetic.csv"
CLEAN_DATA_FILE = DATA_PROCESSED / "accidents_clean.csv"
BEST_MODEL_FILE = MODELS_DIR / "severity_best.joblib"
METRICS_FILE = REPORTS_DIR / "metrics.json"
SEVERITY_RESULTS_FILE = REPORTS_DIR / "severity_results.csv"
SHAP_FILE = REPORTS_DIR / "shap_importance_fatal.csv"
DBSCAN_FILE = REPORTS_DIR / "hotspots_dbscan.csv"
KMEANS_FILE = REPORTS_DIR / "kmeans_zones.csv"
KDE_FILE = REPORTS_DIR / "kde_grid.csv"
MAP_FILE = REPORTS_DIR / "hotspot_map.html"
RISK_FILE = REPORTS_DIR / "risk_prototype.csv"

# --------------------------------------------------------------------------- general
RANDOM_STATE = 42
DEFAULT_N_SAMPLES = 20_000  # size of the synthetic dataset

# --------------------------------------------------------------------------- target
TARGET = "severity"
SEVERITY_LABELS = {0: "Minor", 1: "Serious", 2: "Fatal"}
FATAL_CLASS = 2
# Severity weights used to rank hotspots / risk cells (a fatal crash counts 10x a minor one)
SEVERITY_WEIGHTS = {0: 1, 1: 3, 2: 10}

# --------------------------------------------------------------------------- schema
# Columns a (real or synthetic) accident CSV must contain.
REQUIRED_COLUMNS = [
    "accident_id", "timestamp", "latitude", "longitude",
    "road_type", "speed_limit", "junction_type", "weather", "light_condition",
    "road_surface", "vehicle_type", "driver_age", "num_vehicles",
    "safety_gear_used", "severity",
]

# Allowed values for categorical columns (used by the generator and by dashboard dropdowns)
CATEGORY_VALUES = {
    "road_type": ["Expressway", "National Highway", "State Highway", "Urban Arterial", "Local Road"],
    "junction_type": ["Mid-block", "T-Junction", "Crossroads", "Roundabout"],
    "weather": ["Clear", "Haze", "Rain", "Fog", "Storm"],
    "light_condition": ["Daylight", "Dusk/Dawn", "Dark - lit", "Dark - unlit"],
    "road_surface": ["Dry", "Wet", "Damaged"],
    "vehicle_type": ["Two-wheeler", "Car", "Auto-rickshaw", "Bus", "Truck", "Pedestrian-involved"],
}
SPEED_LIMITS = [20, 30, 40, 50, 60, 80, 100]

# --------------------------------------------------------------------------- model features
NUMERIC_FEATURES = [
    "speed_limit", "driver_age", "num_vehicles", "safety_gear_used",
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour", "is_night",
    "weather_severity_index",
]
CATEGORICAL_FEATURES = [
    "road_type", "junction_type", "weather", "light_condition", "road_surface", "vehicle_type",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RUSH_HOURS = [7, 8, 9, 17, 18, 19]
NIGHT_START_HOUR, NIGHT_END_HOUR = 21, 5  # night = 21:00 to 04:59
WEATHER_SEVERITY_INDEX = {"Clear": 0, "Haze": 1, "Rain": 2, "Fog": 3, "Storm": 4}

# --------------------------------------------------------------------------- severity experiments
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}
IMBALANCE_STRATEGIES = ["none", "class_weight", "smote"]
TEST_SIZE = 0.20
CV_FOLDS = 3
CV_STRATEGY = "smote"  # imbalance strategy used for the cross-validation table
SHAP_SAMPLE_SIZE = 500

# --------------------------------------------------------------------------- hotspots
KMEANS_K = 8
DBSCAN_EPS_KM = 0.30
DBSCAN_MIN_SAMPLES = 30
KDE_BANDWIDTH_KM = 0.80
KDE_GRID_SIZE = 60
SYNTHETIC_RECOVERY_TOL_KM = 1.5

# --------------------------------------------------------------------------- risk prototype
RISK_GRID_DECIMALS = 2  # 0.01 degree is roughly 1.1 km
TIME_BAND_BINS = [-1, 4, 6, 9, 15, 19, 23]
TIME_BAND_LABELS = [
    "Night (00-05)", "Early morning (05-07)", "Morning peak (07-10)",
    "Midday (10-16)", "Evening peak (16-20)", "Late evening (20-24)",
]
ADVERSE_WEATHER = {"Rain", "Fog", "Storm"}
RISK_TEMPORAL_SPLIT = 0.80  # first 80% of the time range = "history", last 20% = "future"

# --------------------------------------------------------------------------- synthetic data
# An arbitrary Delhi-NCR-sized bounding box. The data are SYNTHETIC; no real place is implied.
BBOX = {"lat_min": 28.40, "lat_max": 28.90, "lon_min": 76.95, "lon_max": 77.45}
N_SYNTHETIC_HOTSPOTS = 8
HOTSPOT_SEED = 7
HOTSPOT_SHARE = 0.65      # share of crashes generated inside hotspots
HOTSPOT_SIGMA_KM = 0.45   # spread of each synthetic hotspot
HOTSPOT_MIN_SEPARATION_KM = 5.0  # planted hotspots are at least this far apart
