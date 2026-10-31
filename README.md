# Reducing Traffic Mortality using Machine Learning

> **Status: progress-review build, ~60% complete.** Everything listed under *Features* works and is tested.
> Items under *Pending* do not exist yet. **All results below come from SYNTHETIC data** (the real dataset
> is not integrated yet), so they check that the pipeline works; they are *not* findings about real roads.

One data pipeline, three coordinated outcomes, built to move road safety from hindsight to foresight:

| Pillar | Question it answers | Status |
|---|---|---|
| **Severity prediction** | How bad is this crash likely to be: Minor, Serious or Fatal? | Working (80%) |
| **Hotspot detection** | Where on the map does danger concentrate? | Working (80%) |
| **Risk prediction** | How dangerous is *this place, at this time, in this weather*? | **Prototype only** (20%) |

Team: Ambar Sharma, Jivesh Dhama, Aditya Tomar, Siddharth Soni, Dept. of CSE (Data Science), Inderprastha Engineering College.

---

## Project status (the 60%)

Overall completion = sum of `weight x fraction done`. These are the team's own estimates; the single source
of truth is [`src/project_status.py`](src/project_status.py) and a test keeps the total near 60%.

| Component | Weight | Done | Points | Done so far | Pending |
|---|---:|---:|---:|---|---|
| Data layer | 15 | 80% | 12.0 | Synthetic generator, CSV loader, cleaning + report, feature engineering | Integrate a real dataset (MoRTH / STATS19 / Kaggle) |
| Severity prediction | 25 | 80% | 20.0 | 4 models x 3 imbalance strategies, stratified split + CV, fatal-class metrics | Hyper-parameter tuning, threshold tuning, calibration, real-data validation |
| Hotspot detection | 15 | 80% | 12.0 | K-Means, DBSCAN, KDE, severity-weighted ranking, Folium map | Tune on real coordinates, time-of-day hotspots |
| Risk prediction | 15 | 20% | 3.0 | Prototype v0 (historical severity-weighted score) | Exposure data, a trained model, calibrated risk levels |
| Explainability | 10 | 50% | 5.0 | Global SHAP importance (fatal class) | LIME, per-prediction explanations |
| Dashboard | 10 | 50% | 5.0 | 5 pages incl. what-if prediction | Unified filters, three-pillar side-by-side view |
| Engineering and delivery | 10 | 30% | 3.0 | One-command pipeline, 24 tests, docs | Docker/cloud deployment, CI, final benchmark, streaming hook |
| **Total** | **100** | | **60.0** | | |

## Features

- **Data layer**: synthetic accident generator (with planted hotspots and deliberate dirt), CSV loader, cleaning step that reports every removed row, feature engineering (hour, weekend, rush hour, night, weather-severity index).
- **Severity models**: Logistic Regression, Random Forest, XGBoost, LightGBM, each trained with *no* imbalance handling, *class weights*, or *SMOTE* (12 experiments). Best model chosen by **fatal-class F2** (recall counts 4x more than precision), not accuracy.
- **Hotspots**: K-Means zones, DBSCAN black spots (300 m radius), KDE density surface, ranking by a severity-weighted score (minor 1, serious 3, fatal 10), interactive Folium map.
- **Risk prototype v0**: severity-weighted score per ~1 km grid cell x time band x weather group, plus a past-to-future sanity check.
- **Explainability**: global SHAP importance for the fatal class.
- **Dashboard**: Streamlit, 5 pages (status, severity, hotspots, risk prototype, what-if prediction).

## Results on synthetic data (one run, 36 s on 1 CPU)

20,060 raw rows -> 19,944 after cleaning (60 duplicates, 16 bad labels, 40 bad coordinates removed). Classes: Minor 75.0%, Serious 20.0%, Fatal 5.0%. Test set: 3,989 crashes (199 fatal).

| Model + strategy | Accuracy | Fatal recall | Fatal precision | Fatal F2 |
|---|---:|---:|---:|---:|
| Logistic Regression, no handling | 0.794 | 0.176 | 0.574 | 0.204 |
| **Logistic Regression, class weights (best)** | 0.685 | **0.683** | 0.252 | **0.509** |
| Random Forest, class weights | 0.721 | 0.523 | 0.314 | 0.461 |
| XGBoost, class weights | 0.701 | 0.563 | 0.274 | 0.465 |
| LightGBM, SMOTE | 0.783 | 0.322 | 0.395 | 0.334 |

Read this as: **without imbalance handling the model looks accurate (79%) but finds only 18% of fatal crashes; class weights find 68% of them at the price of more false alarms.** That trade-off is the core idea of the project.

- **Why does logistic regression win?** Our synthetic generator is additive (its hidden danger score is a sum of effects), which a linear model captures well. This is a property of the synthetic data; on real data, tree models may well win. Do not claim "LR is better than XGBoost" from this run.
- **Hotspots**: DBSCAN found 8 clusters (silhouette 0.91) and recovered all 8 *planted* hotspots (one-to-one, within 1.5 km); 36% of crashes are noise, which matches the 35% uniform background we generated.
- **Risk prototype**: the top 20 grid cells chosen from the first 80% of time contain 53.2% of later severity-weighted crashes (random: 0.77%, PAI 69x). This is guaranteed to look good because synthetic hotspots never move; it is a *test harness*, not evidence.
- **SHAP (fatal class)**: speed limit, vehicle type, road type, safety gear, driver age lead. This explains the **XGBoost + class-weight** model, not the saved logistic-regression model (see the docs).

## Tech stack

Python 3.11+ (tested 3.12) | pandas, NumPy | scikit-learn | imbalanced-learn (SMOTE) | XGBoost | LightGBM | SHAP | Streamlit | Plotly | Folium | joblib | pytest. Reasons for each: [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md). Planned but **not used yet**: LIME, GeoPandas, Optuna.

## Architecture summary

```
 CSV / synthetic generator --> clean --> features --> +--> Severity models --> SHAP --+
                                                      +--> Hotspots (KM/DBSCAN/KDE) --+--> reports/ + models/ --> Streamlit dashboard
                                                      +--> Risk prototype ------------+
```

No database and no REST API: stages communicate through files in `data/`, `models/` and `reports/`. Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation

```bash
git clone <your-repo-url> traffic-mortality-ml
cd traffic-mortality-ml
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Full fresh-computer guide (Windows / macOS / Linux): [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md).

## How to run

```bash
python run_pipeline.py            # ~40 s: generates data, trains, clusters, writes models/ and reports/
python run_pipeline.py --quick    # ~20 s demo run (6,000 rows, 2-fold CV)
streamlit run app/dashboard.py    # open http://localhost:8501
python -m pytest -q               # 24 tests
```

Pipeline options: `--data-path FILE.csv` (use real data, schema in [`docs/DATA_FLOW.md`](docs/DATA_FLOW.md)), `--n-samples N`, `--cv-folds K` (0 = off), `--skip-shap`, `--quick`.

## Environment variables

**None are required.** All settings live in [`src/config.py`](src/config.py). (A weather API key would only be needed if the planned weather-enrichment step is built.)

## Screenshots

Add your own after running the dashboard (placeholders):

| Page | File to add |
|---|---|
| Project status | `docs/images/status.png` |
| Severity model | `docs/images/severity.png` |
| Hotspot map | `docs/images/hotspots.png` |
| Risk prototype | `docs/images/risk.png` |

## API summary

There is **no REST API** in this version. The "API" is the Python function interface plus a command line:

| Where | Function | Purpose |
|---|---|---|
| `src/data_loader.py` | `load_accidents(path=None, n_samples)` | CSV or synthetic data |
| `src/preprocessing.py` | `clean_accidents(df)` | validate + remove bad rows, returns `(df, report)` |
| `src/features.py` | `engineer_features(df)` | add time and weather features |
| `src/severity.py` | `make_pipeline`, `run_experiments`, `cross_validate`, `predict_severity` | train / compare / score |
| `src/hotspots.py` | `run_kmeans`, `run_dbscan`, `compute_kde_grid`, `build_folium_map` | hotspot detection |
| `src/risk.py` | `build_risk_table`, `temporal_validation` | risk prototype |
| `src/explain.py` | `shap_importance` | global SHAP |
| `run_pipeline.py` | CLI | run everything |

## Database summary

**No database.** Data and results are plain files:

| Path | Content |
|---|---|
| `data/raw/accidents_synthetic.csv` | generated raw data (git-ignored) |
| `data/processed/accidents_clean.csv` | cleaned + engineered data (git-ignored) |
| `models/severity_best.joblib` | saved best severity pipeline |
| `reports/metrics.json` | all headline numbers for the dashboard |
| `reports/*.csv`, `reports/hotspot_map.html` | result tables and the map |

The SRS lists PostgreSQL/SQLite as an option for a later multi-user version; it is not built.

## Deployment instructions

Not deployed from this sandbox; these are the standard steps.

**Streamlit Community Cloud** (simplest): push the repo to GitHub (keep `models/` and `reports/`, they are small), then at share.streamlit.io choose *New app*, pick the repo, branch `main`, main file `app/dashboard.py`, and a Python version of 3.11 or newer under *Advanced settings*. If the model fails to load after deployment, scikit-learn versions differ: see `docs/TROUBLESHOOTING.md`.

**On your own machine / LAN**: `streamlit run app/dashboard.py --server.address 0.0.0.0 --server.port 8501`.

**Pending**: Docker image, CI, authentication, and any production hosting.

## Documentation

[`PROJECT_OVERVIEW`](docs/PROJECT_OVERVIEW.md) | [`ARCHITECTURE`](docs/ARCHITECTURE.md) | [`PROJECT_STRUCTURE`](docs/PROJECT_STRUCTURE.md) | [`SETUP_GUIDE`](docs/SETUP_GUIDE.md) | [`DATA_FLOW`](docs/DATA_FLOW.md) | [`CODE_EXPLANATION`](docs/CODE_EXPLANATION.md) | [`PROJECT_QA`](docs/PROJECT_QA.md) | [`TROUBLESHOOTING`](docs/TROUBLESHOOTING.md)

## Limitations (say these before you are asked)

1. Synthetic data only; no real-world claims.
2. Risk pillar is a historical-score prototype, not a predictive model.
3. No hyper-parameter or threshold tuning; probabilities are not calibrated.
4. SHAP explains XGBoost, while the saved/served model is logistic regression.
5. No LIME, no live/streaming data, no deployment yet.
