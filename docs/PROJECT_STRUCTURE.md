# Project Structure

## 1. Folder tree

```
traffic-mortality-ml/
├── README.md                  # front page: status, results, how to run
├── requirements.txt           # Python dependencies (lower bounds + tested versions)
├── pytest.ini                 # makes `import src` work in tests; hides noisy warnings
├── .gitignore                 # keeps generated CSV data and caches out of Git
├── run_pipeline.py            # ONE command that runs the whole ML pipeline
│
├── src/                       # the actual project code (a Python package)
│   ├── __init__.py            # package marker + version "0.6.0"
│   ├── config.py              # every path, constant, column list, setting
│   ├── data_generator.py      # synthetic accident data with planted hotspots
│   ├── data_loader.py         # real CSV or synthetic data
│   ├── preprocessing.py       # cleaning + validation report
│   ├── features.py            # time / weather feature engineering
│   ├── severity.py            # Pillar 1: models, imbalance strategies, evaluation
│   ├── hotspots.py            # Pillar 3: K-Means, DBSCAN, KDE, map
│   ├── risk.py                # Pillar 2: PROTOTYPE risk table + validation
│   ├── explain.py             # SHAP importance
│   └── project_status.py      # the weighted "60% complete" table
│
├── app/
│   └── dashboard.py           # Streamlit dashboard (5 pages)
│
├── tests/                     # 24 pytest tests
│   ├── test_data.py           # generator, cleaning, features
│   ├── test_models.py         # severity, hotspots, risk, status
│   └── test_dashboard.py      # every dashboard page renders
│
├── data/                      # created/filled by the pipeline
│   ├── raw/                   # accidents_synthetic.csv (git-ignored)
│   └── processed/             # accidents_clean.csv (git-ignored)
│
├── models/
│   └── severity_best.joblib   # saved best severity pipeline (small, committed)
│
├── reports/                   # everything the dashboard shows (small, committed)
│   ├── metrics.json           # headline numbers of the last run
│   ├── severity_results.csv   # 12 experiments
│   ├── shap_importance_fatal.csv
│   ├── hotspots_dbscan.csv    # ranked black spots
│   ├── kmeans_zones.csv       # broad zones
│   ├── kde_grid.csv           # density surface
│   ├── hotspot_map.html       # interactive Folium map
│   └── risk_prototype.csv     # risk table
│
└── docs/                      # you are here
    ├── PROJECT_OVERVIEW.md  ARCHITECTURE.md  PROJECT_STRUCTURE.md  SETUP_GUIDE.md
    ├── DATA_FLOW.md  CODE_EXPLANATION.md  PROJECT_QA.md  TROUBLESHOOTING.md
    └── images/                # put dashboard screenshots here
```

## 2. File by file: what each is responsible for

| File | Responsibility | Key contents |
|---|---|---|
| `run_pipeline.py` | Orchestrates the 6 steps (load/clean/features, severity, SHAP, hotspots, risk, save) and writes `metrics.json` | `main()`, command-line flags |
| `src/config.py` | Single source of settings | Paths, `FEATURE_COLUMNS`, `SEVERITY_LABELS`, `SEVERITY_WEIGHTS`, `MODEL_NAMES`, DBSCAN/KDE/risk settings, synthetic-data settings |
| `src/data_generator.py` | Creates realistic-looking test data with known structure | `generate_synthetic_accidents`, `get_synthetic_hotspot_centers` |
| `src/data_loader.py` | Chooses data source | `load_accidents` |
| `src/preprocessing.py` | Removes untrustworthy rows and reports counts | `clean_accidents` |
| `src/features.py` | Builds model features from raw columns | `engineer_features` |
| `src/severity.py` | Everything about severity prediction | `build_preprocessor`, `build_classifier`, `make_pipeline`, `fit_pipeline`, `split_data`, `evaluate`, `run_experiments`, `cross_validate`, `save_model`, `load_model`, `predict_severity` |
| `src/hotspots.py` | Everything about hotspots | `to_km`, `haversine_km`, `summarise_clusters`, `run_kmeans`, `run_dbscan`, `compute_kde_grid`, `build_folium_map`, `synthetic_recovery` |
| `src/risk.py` | Risk prototype | `add_risk_keys`, `build_risk_table`, `risk_level`, `temporal_validation` |
| `src/explain.py` | Global SHAP importance | `shap_importance` |
| `src/project_status.py` | Progress table | `COMPONENTS`, `overall_completion`, `status_table` |
| `app/dashboard.py` | UI | one `page_*` function per page, `main()` |
| `tests/*.py` | Safety net | see section 4 |
| `reports/metrics.json` | Contract between pipeline and dashboard | data summary, best model, CV, SHAP status, hotspot stats, risk validation |
| `models/severity_best.joblib` | The trained model | a full scikit-learn/imbalanced-learn pipeline |

## 3. Safe to modify vs. careful vs. do not touch

### Safe to modify (low risk)

| File / thing | Typical change | Then do |
|---|---|---|
| `src/config.py` values: `KMEANS_K`, `DBSCAN_EPS_KM`, `DBSCAN_MIN_SAMPLES`, `KDE_BANDWIDTH_KM`, `SEVERITY_WEIGHTS`, `DEFAULT_N_SAMPLES`, `CV_FOLDS`, `TIME_BAND_*` | Tune behaviour | Re-run `python run_pipeline.py` |
| `src/project_status.py` `COMPONENTS` | Update progress after more work | `pytest` (a test expects ~60%; update the test when your real number changes) |
| Model hyper-parameters in `build_classifier` (`src/severity.py`) | `n_estimators`, `max_depth`... | Re-run pipeline |
| Text, captions, titles in `app/dashboard.py` | Wording | Reload the page |
| `README.md`, `docs/` | Documentation | - |
| `requirements.txt` | Add a library you use | Reinstall |

### Change carefully (other files depend on them)

| File / thing | Why careful |
|---|---|
| `REQUIRED_COLUMNS`, `CATEGORY_VALUES`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES` in `config.py` | The generator, cleaning, pipeline, SHAP name-matching, dashboard dropdowns and saved model all assume them. If you change features, re-run the pipeline (old `.joblib` will not match) |
| `severity` label mapping (`SEVERITY_LABELS`, `FATAL_CLASS`) | Metrics, hotspot scores, confusion matrix and dashboard all assume 0/1/2 = Minor/Serious/Fatal |
| Keys inside `reports/metrics.json` (written in `run_pipeline.py`) | The dashboard reads them by name. Rename on both sides together |
| `data_generator.py` formulas | Changing them changes every "result". Fine for experiments, but update the numbers in README/docs |
| Order of steps in `Pipeline` (`make_pipeline`) | SMOTE must stay *after* preprocessing and *before* the classifier, and inside the pipeline (otherwise it leaks) |

### Do not edit by hand

| File | Why |
|---|---|
| `models/*.joblib` | Binary; regenerate with `python run_pipeline.py` |
| `reports/*` | Generated outputs; overwritten every run |
| `data/raw/*.csv`, `data/processed/*.csv` | Generated; overwritten every run |
| `__pycache__/`, `.pytest_cache/` | Python/pytest caches |

## 4. What each test file protects

| Test file | Guards against |
|---|---|
| `test_data.py` (7 tests) | Generator drift, un-clean data slipping through, missing-column input, wrong time flags, planted hotspots too close |
| `test_models.py` (12 tests) | Pipelines that fail to train/predict, leakage-prone splits, class weighting not helping fatal recall, DBSCAN not finding planted hotspots, bad risk index values, status drifting from ~60% |
| `test_dashboard.py` (5 tests) | A page raising an exception (needs the pipeline to have run; skipped otherwise) |

## 5. Rule of thumb for adding something

1. New setting -> `config.py`.
2. New data step -> a function in the right `src/` file, with a test.
3. New result the dashboard should show -> write it in `run_pipeline.py` (CSV/JSON in `reports/`), then read it in `dashboard.py`.
4. Update `project_status.py` and the README table.
