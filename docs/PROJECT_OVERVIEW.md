# Project Overview

## 1. What the project does (in one paragraph)

Road-safety data is usually used *after* the fact: someone counts last year's deaths and publishes the tally. This project turns the same accident records into **three coordinated tools** that point at what to do next:

1. **Severity prediction** - given the conditions of a crash (speed limit, road type, light, weather, vehicle, helmet/seatbelt, time...), predict whether it ends up **Minor, Serious or Fatal**.
2. **Hotspot detection** - find the places on the map where crashes keep concentrating, and rank them by *how severe* those crashes were, not just how many.
3. **Risk prediction** - score how dangerous a place + time + weather combination is *before* a crash happens. *(Prototype only in this build.)*

Everything runs from **one command** (`python run_pipeline.py`) and is shown in a **Streamlit dashboard**.

> **Honest status (progress review): about 60% complete, running on synthetic data.** See section 8.

---

## 2. Problem statement

From the project synopsis: road traffic crashes kill about **1.19 million** people a year worldwide (WHO, 2023), and in India **172,890** deaths in **480,583** accidents were recorded in 2023, the highest on record (MoRTH). Current practice is *descriptive*: investigate afterwards, publish annually. That does not tell a traffic officer which junction, at which hour, in which weather needs attention tonight.

Research gaps this project targets (from the synopsis):

| Gap in existing work | What we do about it |
|---|---|
| Severity and location are studied in separate papers | One pipeline feeding severity, hotspots and risk together |
| Models are judged on overall accuracy, which hides rare fatal cases | **Fatal-class recall / F2** is the primary metric; accuracy is reported but never used to pick a model |
| Little explainability for non-technical users | SHAP importance now; LIME and per-crash explanations are pending |
| Few papers deliver a usable tool | A working Streamlit dashboard |
| Everything is retrospective | Risk pillar (prototype) and a design that can later take streaming data |

---

## 3. Objectives (and where we stand)

| # | Objective (from the synopsis) | Status in this build |
|---|---|---|
| 1 | Collect, clean, preprocess accident data; handle missing values, encode categories, correct class imbalance | Done on synthetic data; real data pending |
| 2 | Benchmark ML classifiers (LR, RF, XGBoost, LightGBM) optimising fatal-class recall | Done (12 experiments + CV); no tuning yet |
| 3 | Find hotspots with K-Means / DBSCAN / KDE | Done |
| 4 | Explain predictions (SHAP / LIME) | SHAP global done; LIME pending |
| 5 | Interactive dashboard | 5 pages working; filters and unified view pending |
| 6 | Evaluate against published benchmarks | **Not started** (needs real data) |

---

## 4. Target users

| User | What they need from the tool |
|---|---|
| **Traffic police / patrol planners** | Which places and time bands deserve patrols (hotspots, risk prototype) |
| **Highway engineers / authorities** | A ranked black-spot list for signage, lighting, speed-calming budgets |
| **Policy makers** | Which factors drive fatal outcomes (speed, vehicle type, safety gear...) |
| **Data-science team / evaluators** | Reproducible experiments, metrics, tests, documentation |

The tool is **decision support only**. It does not decide fault, liability, or engineering action; a human analyst does.

---

## 5. Main features

| Feature | Where it lives | Notes |
|---|---|---|
| Synthetic data generator with planted hotspots | `src/data_generator.py` | Lets us test the whole pipeline before real data arrives |
| Cleaning with a removal report | `src/preprocessing.py` | Duplicates, bad labels, bad coordinates |
| Feature engineering | `src/features.py` | hour, day of week, month, weekend, rush hour, night, weather-severity index |
| 12 severity experiments | `src/severity.py` | 4 models x {none, class-weight, SMOTE} |
| Stratified split + k-fold CV | `src/severity.py` | Keeps the 5% fatal share in every split |
| Hotspots | `src/hotspots.py` | K-Means, DBSCAN, KDE, severity-weighted ranking, Folium map |
| Risk prototype v0 | `src/risk.py` | Historical score + past-to-future sanity check |
| SHAP importance | `src/explain.py` | Fatal class, one-hot columns merged per feature |
| Dashboard | `app/dashboard.py` | Status, Severity, Hotspots, Risk, What-if prediction |
| Tests | `tests/` | 24 tests |

---

## 6. Technologies used and why each was chosen

| Technology | Used for | Why this and not something else |
|---|---|---|
| **Python 3** | Everything | The standard language for data science; every library we need exists for it; the team already uses it |
| **pandas / NumPy** | Tables and numbers | The default tools for tabular data; fast enough for ~20,000 rows (and millions) |
| **scikit-learn** | Pipelines, Logistic Regression, Random Forest, K-Means, DBSCAN, KDE, metrics, splits | One consistent API for the whole ML workflow; `Pipeline` + `ColumnTransformer` stop data leakage |
| **imbalanced-learn** | SMOTE inside the pipeline | Its pipeline applies SMOTE only to training data, which scikit-learn's own Pipeline cannot do |
| **XGBoost** | Gradient-boosted trees | Strong on tabular data with non-linear interactions (named in the synopsis) |
| **LightGBM** | Gradient-boosted trees | Fast, handles large data; gives a second boosting family to compare (named in the synopsis) |
| **SHAP** | Explaining the fatal class | Gives per-feature contributions grounded in game theory, and works with tree models via `TreeExplainer` |
| **Streamlit** | Dashboard | Whole dashboard in plain Python (no HTML/JS team needed); easy free deployment |
| **Plotly** | Charts inside Streamlit | Interactive charts with very little code |
| **Folium** | Hotspot map | Produces a Leaflet web map from Python; saved as one HTML file |
| **joblib** | Saving the trained model | The standard way to persist scikit-learn pipelines |
| **pytest** | Tests | Simple, readable tests; Streamlit's `AppTest` lets us smoke-test the dashboard |
| **Git/GitHub** | Version control | Required for team work and for deployment |

Mentioned in the synopsis but **not used yet**: **LIME** (pending, local explanations), **GeoPandas** (we use a simple local km projection in `to_km`, which is enough at city scale), **Optuna** (tuning is pending).

---

## 7. Design decisions you should be able to explain

1. **Three pillars, modular.** Different question types, different data grain, different users. Fused only at the dashboard.
2. **Recall-first.** Missing a fatal crash costs far more than a false alarm, so we select models by fatal **F2** (recall weighted 4x precision).
3. **No leakage.** Imputation, scaling, encoding and SMOTE all sit *inside* the scikit-learn pipeline, so they are learned from training data only. We also exclude post-crash outcomes (like "number injured" or "damage level") from features, since you would not know them when predicting severity.
4. **Synthetic first.** Real data is slow to obtain and clean; a generator with known structure lets us prove the code works (for instance: do the hotspots come back where we planted them?).
5. **Files, not a database.** One user, batch data, a few MB of results: files are simpler and enough for now.

---

## 8. Status: what "60% complete" means

Weighted estimate, source of truth in `src/project_status.py`:

| Component | Weight | Done | Points |
|---|---:|---:|---:|
| Data layer | 15 | 80% | 12.0 |
| Severity prediction | 25 | 80% | 20.0 |
| Hotspot detection | 15 | 80% | 12.0 |
| Risk prediction | 15 | 20% | 3.0 |
| Explainability | 10 | 50% | 5.0 |
| Dashboard | 10 | 50% | 5.0 |
| Engineering and delivery | 10 | 30% | 3.0 |
| **Total** | 100 | | **60.0** |

**Be ready for these questions:** *"Is the data real?"* No, synthetic; the real dataset is the biggest pending item. *"Is the risk pillar a model?"* No, a historical-score prototype. *"Can you trust these accuracy numbers?"* They validate the pipeline, not real-world performance.

**Next steps, in order:** (1) integrate a real dataset and map its columns; (2) re-run and compare tree models vs logistic regression on real data; (3) tune hyper-parameters and the decision threshold; (4) build the real risk model; (5) add LIME and per-crash explanations; (6) deploy.
