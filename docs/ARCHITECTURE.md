# Architecture

## 1. The short version

This is a **batch ML pipeline plus a read-only dashboard**. It is *not* a client-server web app with a database. Two programs exist:

| Program | Command | What it does |
|---|---|---|
| **Pipeline** (backend) | `python run_pipeline.py` | Loads/generates data, cleans it, trains models, finds hotspots, writes result files |
| **Dashboard** (frontend) | `streamlit run app/dashboard.py` | Reads those result files and shows them; also scores one "what-if" crash with the saved model |

They talk to each other **only through files** on disk (`models/`, `reports/`).

## 2. How the pieces map to the usual words

You may be asked about "frontend, backend, database, APIs, external services". Here is the honest mapping:

| Usual term | In this project |
|---|---|
| **Frontend** | Streamlit app `app/dashboard.py` (Python generates the web page; the browser shows it) |
| **Backend** | The `src/` Python modules, orchestrated by `run_pipeline.py`. Streamlit also runs Python on the server side, so the dashboard is frontend and light backend at once |
| **Database** | **None.** Plain CSV / JSON / joblib / HTML files (see `PROJECT_STRUCTURE.md`) |
| **REST API** | **None.** Modules call each other as normal Python functions. (The SRS plans Flask/FastAPI + PostgreSQL for a later multi-user version; not built) |
| **ML models** | scikit-learn pipelines (Logistic Regression, Random Forest, XGBoost, LightGBM), K-Means, DBSCAN, KDE |
| **External services at run time** | Only the user's **browser** fetching OpenStreetMap map tiles and the Leaflet library for the hotspot map. No API keys, no cloud calls from our code |

## 3. Diagram

```
                           +--------------------------------------------+
 INPUT                     |            BACKEND: python run_pipeline.py |
                           |                                            |
 real CSV  ---------+      |  +---------+   +--------+   +----------+   |
 (--data-path)      +----->|  | load    |-->| clean  |-->| features |   |
 synthetic          |      |  | data_   |   | prepro-|   | features |   |
 generator ---------+      |  | loader  |   | cessing|   |   .py    |   |
 (default)                 |  +---------+   +--------+   +----+-----+   |
                           |                                  |         |
                           |        +-------------------------+------+  |
                           |        |                         |      |  |
                           |   +----v-----+            +------v--+ +-v--------+
                           |   | SEVERITY |            | HOTSPOTS| | RISK     |
                           |   | severity |            | hotspots| | PROTOTYPE|
                           |   |  .py     |            |  .py    | | risk.py  |
                           |   | 4 models |            | KMeans  | | grid x   |
                           |   | x 3 imb. |            | DBSCAN  | | time x   |
                           |   | strategies|           | KDE     | | weather  |
                           |   +----+-----+            +----+----+ +----+-----+
                           |        |                       |           |
                           |   +----v-----+                 |           |
                           |   | SHAP     |                 |           |
                           |   | explain  |                 |           |
                           |   |  .py     |                 |           |
                           |   +----+-----+                 |           |
                           +--------|-----------------------|-----------|---+
                                    v                       v           v
 STORAGE (files, no DB)      models/severity_best.joblib   reports/*.csv, metrics.json,
                                    |                      hotspot_map.html
                                    |                       |
                           +--------v-----------------------v--------------+
 FRONTEND                  |      app/dashboard.py   (Streamlit, Plotly)   |
                           |  Status | Severity | Hotspots | Risk | Predict |
                           +------------------------+----------------------+
                                                    |
                                                    v
                                      Browser  (+ OpenStreetMap tiles)
                                      Traffic police / engineers / evaluators
```

## 4. Role of every major component

| Component | Responsibility | Talks to |
|---|---|---|
| `src/config.py` | All paths, column lists, constants, hotspot/risk settings. One place to change behaviour | Everyone imports it |
| `src/data_generator.py` | Builds synthetic accidents with a known structure (planted hotspots, hidden danger score, ~5% fatal, some dirt) | `data_loader` |
| `src/data_loader.py` | Reads a real CSV, or calls the generator and saves `data/raw/accidents_synthetic.csv` | `run_pipeline` |
| `src/preprocessing.py` | Validates columns and removes duplicates, bad labels, bad timestamps, bad coordinates; returns a report | `run_pipeline` |
| `src/features.py` | Adds time and weather features | `run_pipeline`, dashboard (for what-if input) |
| `src/severity.py` | Builds the preprocessing + model pipelines, trains 12 experiments, evaluates, cross-validates, saves/loads the model, predicts | `run_pipeline`, dashboard |
| `src/explain.py` | SHAP importance for the fatal class | `run_pipeline` |
| `src/hotspots.py` | K-Means, DBSCAN, KDE, ranking, Folium map, planted-hotspot recovery check | `run_pipeline`, `data_generator` (distance helper) |
| `src/risk.py` | Risk table and past-to-future check | `run_pipeline` |
| `src/project_status.py` | The weighted 60% table | dashboard, tests, `run_pipeline` |
| `run_pipeline.py` | Runs all steps in order, writes every artifact and `metrics.json` | all of `src/` |
| `app/dashboard.py` | Displays results, runs what-if predictions | reads `reports/`, `models/` |
| `tests/` | 24 automated checks | all of the above |

## 5. The key design ideas

### 5.1 One scikit-learn Pipeline per experiment

```
raw feature columns
   -> ColumnTransformer
        numbers   : median-impute -> StandardScaler
        categories: fill "Unknown" -> OneHotEncoder
   -> (SMOTE only if strategy == "smote")
   -> classifier
```

Because everything is inside one object, **cross-validation and the train/test split cannot leak** statistics from validation data into training, and the **same object scores new crashes** in the dashboard with no hand-written preprocessing.

### 5.2 Three imbalance strategies compared side by side

| Strategy | How |
|---|---|
| `none` | train as is (the "naive" baseline) |
| `class_weight` | rare classes count more in the loss (`class_weight="balanced"`, or sample weights for XGBoost) |
| `smote` | create synthetic fatal/serious examples in the training set only |

### 5.3 Model selection rule

Best model = highest **fatal-class F2**, ties broken by macro-F1. The best pipeline is saved. A second selection, the best *tree-based* pipeline, is kept for SHAP because `TreeExplainer` needs a tree model. **These two can differ**: in the current run the saved model is logistic regression + class weights, and SHAP explains XGBoost + class weights. This is a known limitation (a `LinearExplainer` path is pending).

### 5.4 Hotspots work in kilometres

Latitude/longitude degrees are not equal distances. `to_km` projects them to local x/y km so DBSCAN's `eps=0.3` really means 300 m and KDE's bandwidth really means 0.8 km.

### 5.5 Pillars are independent until the dashboard

Severity uses crash conditions; hotspots use only coordinates; risk uses coordinates + time + weather. That separation is deliberate (different questions, data grain and users); the dashboard is where they sit together.

## 6. What happens if something fails

| Failure | Behaviour |
|---|---|
| Input CSV missing columns | `clean_accidents` raises `ValueError` listing them |
| `--data-path` file not found | `FileNotFoundError` |
| SHAP fails (version problems) | Caught; pipeline continues; `metrics.json` records `"failed: ..."`; dashboard shows a warning |
| Dashboard opened before the pipeline ran | Shows the status page plus a clear "run the pipeline" message |
| Saved model does not load (different scikit-learn version) | "Predict" page shows an error telling you to re-run the pipeline; other pages still work |
| No internet in the browser | Hotspot map tiles are blank; everything else works |

## 7. Not built yet (so please do not claim it)

Database, REST API, authentication / roles, alerts and e-mail, PDF reports, streaming/telematics input, deep-learning model, Docker/CI, LIME.
