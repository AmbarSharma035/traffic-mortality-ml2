# Data Flow

How one accident record travels through the project, from raw input to a number on the dashboard. Numbers quoted are from the documented run (synthetic data, seed 42, 20,000 requested rows).

## 1. The big picture

```
 (A) INPUT                 (B) CLEAN               (C) FEATURES            (D) THREE PILLARS               (E) OUTPUT
 ───────────               ─────────               ───────────             ──────────────────              ──────────
 CSV --data-path   ─┐
                    ├─► load_accidents ─► clean_accidents ─► engineer_features ─┬─► severity.py  ─► models/*.joblib, severity_results.csv
 synthetic          │    (20,060 rows)    (19,944 rows)      (+7 columns)       │      └► explain.py ► shap_importance_fatal.csv
 generator         ─┘                                                           ├─► hotspots.py  ─► hotspots_dbscan.csv, kmeans_zones.csv,
                                                                                │                   kde_grid.csv, hotspot_map.html
                                                                                └─► risk.py      ─► risk_prototype.csv
                                                                                                    │
                                                          run_pipeline.py ──────────────────────────┴─► metrics.json
                                                                                                    │
                                                                                          app/dashboard.py ─► browser
```

## 2. Step by step

### Step A - Input (`src/data_loader.py`)

| Source | Condition | Result |
|---|---|---|
| Your CSV | `--data-path file.csv` | `pd.read_csv`, source label `real:file.csv` |
| Synthetic | default | `generate_synthetic_accidents(n)`, saved to `data/raw/accidents_synthetic.csv`, label `synthetic` |

**Required schema** (`config.REQUIRED_COLUMNS`):

| Column | Type | Meaning |
|---|---|---|
| `accident_id` | text | unique id |
| `timestamp` | datetime text | when the crash happened |
| `latitude`, `longitude` | number | location in degrees |
| `road_type` | text | Expressway / National Highway / State Highway / Urban Arterial / Local Road |
| `speed_limit` | number | km/h |
| `junction_type` | text | Mid-block / T-Junction / Crossroads / Roundabout |
| `weather` | text | Clear / Haze / Rain / Fog / Storm |
| `light_condition` | text | Daylight / Dusk/Dawn / Dark - lit / Dark - unlit |
| `road_surface` | text | Dry / Wet / Damaged |
| `vehicle_type` | text | Two-wheeler / Car / Auto-rickshaw / Bus / Truck / Pedestrian-involved |
| `driver_age` | number | years |
| `num_vehicles` | number | vehicles involved |
| `safety_gear_used` | 0/1 | helmet or seatbelt used |
| `severity` | 0/1/2 | **target**: Minor / Serious / Fatal |

The synthetic generator deliberately adds dirt: ~1-2% missing values in four columns, ~0.3% duplicate rows, ~0.2% impossible coordinates, ~0.1% missing labels.

### Step B - Cleaning and validation (`src/preprocessing.py`, `clean_accidents`)

Checks run in this order; each removal is counted in the report (`reports/metrics.json` -> `data.cleaning_report`):

| # | Validation | Action | Count in the documented run |
|---|---|---|---:|
| 0 | All required columns present? | `ValueError` listing missing ones | - |
| 1 | Duplicate `accident_id` | drop later copies | 60 |
| 2 | `severity` not in {0,1,2} or missing | drop row | 16 |
| 3 | `timestamp` unreadable | drop row | 0 |
| 4 | Latitude outside [-90, 90], longitude outside [-180, 180], or exactly (0, 0) | drop row | 40 |
| 5 | `driver_age` outside 16-90 | set to missing (not dropped) | - |

20,060 rows in, **19,944 rows out**. Missing *feature* values are **not** filled here (see the leakage note below). Remaining missing: weather 295, light 214, road surface 184, driver age 426.

### Step C - Feature engineering (`src/features.py`)

| New column | Rule |
|---|---|
| `hour`, `day_of_week` (Mon=0), `month` | from `timestamp` |
| `is_weekend` | day_of_week >= 5 |
| `is_rush_hour` | hour in 07,08,09,17,18,19 |
| `is_night` | hour >= 21 or hour < 5 |
| `weather_severity_index` | Clear 0, Haze 1, Rain 2, Fog 3, Storm 4 (unknown stays missing) |

Result is saved as `data/processed/accidents_clean.csv`.

### Step D1 - Severity (`src/severity.py`)

```
cleaned table
   │  split_data: stratified 80 / 20  (15,955 train / 3,989 test; fatal share equal in both)
   ▼
for each of 4 models x 3 strategies (12 experiments):
   Pipeline = [ impute + scale numbers | fill "Unknown" + one-hot categories ]
              → (SMOTE, only for "smote")
              → classifier
   fit on TRAIN only → predict_proba on TEST → metrics
   ▼
pick best by fatal F2  → save model → (optional) 3-fold CV on TRAIN → results CSV + metrics.json
```

Validations / protections in this step:

| Protection | Where | Why |
|---|---|---|
| Stratified split | `split_data` | Keep ~5% fatal in both halves |
| Imputer, scaler, encoder, SMOTE inside the Pipeline | `make_pipeline` | They are fitted on training data only, so **no information leaks from the test set** |
| `OneHotEncoder(handle_unknown="ignore")` | `build_preprocessor` | A category never seen in training will not crash prediction |
| Unknown model/strategy name | `build_classifier`, `make_pipeline` | Clear `ValueError` |
| Only crash-time features used | `config.FEATURE_COLUMNS` | Post-crash facts (injury counts, damage) would leak the answer |

### Step D2 - SHAP (`src/explain.py`)

Take the best **tree-based** pipeline (this run: XGBoost + class weights), transform 500 test rows, run `TreeExplainer`, keep the **fatal-class** values, merge one-hot columns back into the original feature (sum per row, then mean absolute value), save `shap_importance_fatal.csv`.

### Step D3 - Hotspots (`src/hotspots.py`)

```
latitude/longitude ─► to_km (degrees → local km)
   ├─► KMeans(k=8)            ─► zones table + silhouette
   ├─► DBSCAN(eps 0.3 km, min 30) ─► clusters (-1 = noise) ─► summarise_clusters ─► ranked by severity score
   ├─► KernelDensity(bandwidth 0.8 km) on a 60 x 60 grid ─► density 0..1
   └─► Folium map (heat layer + top-15 ranked markers)
```
Severity score = 1 x minor + 3 x serious + 10 x fatal crashes in the cluster. If the data is synthetic, `synthetic_recovery` checks how many planted centres were found.

### Step D4 - Risk prototype (`src/risk.py`)

Round coordinates to 0.01 degree (~1.1 km) -> `grid_id`; hour -> `time_band` (6 bands); weather -> Normal/Adverse. Sum severity weights per (cell, band, weather group) -> percentile 0-100 `risk_index` -> placeholder `risk_level`. Then `temporal_validation` builds the score from the first 80% of time and measures how many later severity-weighted crashes fall in the top 20 cells.

### Step E - Output and display

`run_pipeline.py` writes `metrics.json` (a summary of everything) and the CSV/HTML/joblib files. `app/dashboard.py` reads them:

| Dashboard page | Reads |
|---|---|
| Project status | `src/project_status.py`, `metrics.json` |
| Severity model | `metrics.json`, `severity_results.csv`, `shap_importance_fatal.csv` |
| Hotspots | `metrics.json`, `hotspots_dbscan.csv`, `kmeans_zones.csv`, `kde_grid.csv`, `hotspot_map.html` |
| Risk (prototype) | `risk_prototype.csv`, `metrics.json` |
| Predict a crash | `models/severity_best.joblib` |

Caching: pages re-read a file when its modified time changes, so re-running the pipeline updates a running dashboard on reload.

## 3. The "what-if" prediction flow (user input -> model -> output)

```
user fills form (road, speed, weather, light, ..., date + hour, helmet?)
   │
   ▼  dashboard builds a 1-row DataFrame (with a timestamp)
engineer_features   → adds hour, is_night, is_rush_hour, weather_severity_index ...
   │
   ▼  predict_severity(model, row)
saved pipeline: impute/encode → logistic regression → 3 probabilities
   │
   ▼
"Probability of a FATAL outcome: x%"  + bar chart of Minor / Serious / Fatal
```
The same feature code and the same saved pipeline are used for training and prediction, so the form input is transformed exactly like the training data.

**Caveats shown on the page:** trained on synthetic data; probabilities from class-weighted/SMOTE models are **not calibrated** (read them as relative scores, not true chances); decision-support only.

## 4. Where data could go wrong (and where it is caught)

| Risk | Caught by |
|---|---|
| Wrong/missing columns | `clean_accidents` ValueError |
| Garbage coordinates | step B.4 |
| Duplicate crashes counted twice | step B.1 |
| Test data influencing training | everything fitted inside the Pipeline on the train split |
| Fatal class ignored | stratified split + class weights/SMOTE + fatal-class metrics |
| Hotspot units wrong (degrees vs km) | `to_km` projection |
| Dashboard shows stale/missing results | file-mtime caching + "run the pipeline" message |
