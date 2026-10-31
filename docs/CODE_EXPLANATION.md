# Code Explanation

For presentations, viva and interviews. Files are in the order you should learn them. Snippets are shortened from the real code.

**Reading order:** `config.py` -> `preprocessing.py` -> `features.py` -> `severity.py` (most important) -> `hotspots.py` -> `risk.py` -> `explain.py` -> `run_pipeline.py` -> `dashboard.py`. `data_generator.py` can be skipped unless asked how the fake data works.

---

## 1. `src/config.py` - the control panel

No logic, just named constants: paths, the feature lists, the label map `{0: Minor, 1: Serious, 2: Fatal}`, severity weights `{0: 1, 1: 3, 2: 10}`, model names, DBSCAN/KDE settings.

**Why it matters:** change a setting once, everything follows. In an interview: *"I avoided magic numbers scattered through the code."*

---

## 2. `src/preprocessing.py` - `clean_accidents(df)`

**Job:** remove rows we cannot trust, and say how many were removed.

```python
out = out.drop_duplicates(subset="accident_id", keep="first")
out = out[out["severity"].isin([0, 1, 2])]
valid = lat.between(-90, 90) & lon.between(-180, 180) & ~((lat == 0) & (lon == 0))
```

**Important design point:** this function does **not** fill missing feature values. That happens inside the model pipeline, so the fill-in values (like the median age) are computed from the training split only. Filling them here, before splitting, would let test-set information leak into training.

Returns `(clean_df, report)`; the report goes into `metrics.json` and the dashboard.

---

## 3. `src/features.py` - `engineer_features(df)`

Turns the timestamp into model-friendly numbers: `hour`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour`, `is_night`, plus `weather_severity_index` (Clear 0 ... Storm 4).

Why: models cannot read a date. "Hour = 22" and "is_night = 1" let the model learn that night crashes are different. The **dashboard also calls this function** for what-if input, so training and prediction use identical features.

---

## 4. `src/severity.py` - Pillar 1 (the heart of the project)

### 4.1 `build_preprocessor()`

```python
numeric     = Pipeline([SimpleImputer(median), StandardScaler()])
categorical = Pipeline([SimpleImputer("Unknown"), OneHotEncoder(handle_unknown="ignore")])
ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])
```

| Piece | Plain-English meaning |
|---|---|
| `SimpleImputer(median)` | Fill missing numbers with the middle value of the training data |
| `StandardScaler` | Put numbers on a similar scale (matters for logistic regression) |
| `OneHotEncoder` | Turn "Rain/Fog/Clear" into separate 0/1 columns, since models need numbers |
| `handle_unknown="ignore"` | A new category at prediction time becomes all zeros instead of crashing |

### 4.2 `build_classifier(name, strategy)`

Returns one of four models. For the `class_weight` strategy it sets `class_weight="balanced"` (rare classes get bigger weight in the loss). XGBoost has no such argument for multi-class, so `fit_pipeline` passes **sample weights** instead (`compute_sample_weight("balanced", y)`).

| Model | One-line explanation |
|---|---|
| Logistic Regression | Adds up weighted evidence for each class; simple, fast, interpretable baseline |
| Random Forest | Many decision trees voting; handles non-linear patterns |
| XGBoost | Trees built one after another, each fixing the previous ones' mistakes |
| LightGBM | Same family as XGBoost, faster tree-building method |

### 4.3 `make_pipeline(name, strategy)`

```python
steps = [("prep", build_preprocessor())]
if strategy == "smote":
    steps.append(("smote", SMOTE(random_state=RS, k_neighbors=5)))
steps.append(("clf", build_classifier(name, strategy)))
return ImbPipeline(steps)
```

**This is the most important idea to understand.** Using `imblearn`'s pipeline, SMOTE runs only when *fitting* and only on the data given to `fit`. When we later call `predict` or evaluate on the test set, SMOTE is skipped. So synthetic examples never leak into evaluation.

**SMOTE in one sentence:** for a rare-class example, pick one of its nearest neighbours of the same class and create a new example somewhere on the line between them: `new = x + random(0..1) * (neighbour - x)`.

**Caveat to admit:** we apply SMOTE after one-hot encoding, so synthetic points can have fractional "category" values (like 0.4 Rain). It is common practice but not perfect; SMOTENC is an alternative.

### 4.4 `split_data(df)` and `cross_validate(...)`

`train_test_split(..., stratify=y)` keeps the 5% fatal share in both halves. `cross_validate` uses `StratifiedKFold` (3 folds): train on 2 parts, test on the 3rd, rotate, report mean +/- std of fatal recall. A small std means the result is stable.

### 4.5 `evaluate(pipe, X_test, y_test)`

Computes accuracy, macro-F1, ROC-AUC, and the **fatal-class** numbers:

| Metric | Meaning | Why we care |
|---|---|---|
| **Fatal recall** | Of all real fatal crashes, what fraction did we flag? | Missing one is the costliest error |
| **Fatal precision** | Of the crashes we flagged fatal, what fraction really were? | Controls false alarms |
| **Fatal F2** | `5*P*R / (4*P + R)`: like F1 but recall counts 4x more | Our selection metric |
| Fatal PR-AUC | Precision/recall quality over all thresholds | Honest under imbalance (a random model scores about 0.05) |
| Macro F1 | F1 averaged over the three classes | Checks we are not ignoring Serious |
| Accuracy | Overall % correct | Reported, but **never** used to choose the model |

### 4.6 `run_experiments(...)`

Loops over 4 models x 3 strategies, trains, evaluates, and tracks (a) the best overall by `(fatal_f2, macro_f1)` and (b) the best **tree-based** pipeline (needed for SHAP). Returns results + both winners.

### 4.7 `predict_severity(pipe, features)`

`pipe.predict_proba(...)` -> a DataFrame with `Minor / Serious / Fatal` probabilities and the most likely class. Used by the dashboard.

### Reading the results correctly

| Observation (documented run) | Explanation |
|---|---|
| "No handling" models have the highest accuracy (79%) but fatal recall of 10-22% | With 5% fatal, predicting "not fatal" almost always is accurate but useless |
| Class weights raise fatal recall to 51-68% and lower precision to 25-31% | We trade false alarms for catching fatal crashes |
| SMOTE helps less than class weights for tree models here | Trees can overfit synthetic points; not a general law |
| Logistic regression is best | The *synthetic* danger score is a sum of effects, which suits a linear model. Do not generalise this to real data |

---

## 5. `src/hotspots.py` - Pillar 3

### 5.1 `to_km(lat, lon)`

```python
x = (lon - lon0) * 111.320 * cos(radians(lat0))
y = (lat - lat0) * 110.574
```
Degrees of longitude shrink as you move away from the equator, so we convert to kilometres around the area's centre. Good enough at city scale. (GeoPandas could do exact projections; not needed yet.)

### 5.2 `run_kmeans(df, k)`

K-Means: pick `k` centres, assign each crash to the nearest, move centres to the cluster means, repeat. Gives **broad zones**; every point is assigned to some zone, even isolated ones. Evaluated by **silhouette score** (-1 bad ... +1 great; ours 0.59).

### 5.3 `run_dbscan(df, eps_km=0.3, min_samples=30)`

DBSCAN: a crash is a *core point* if >= 30 crashes lie within 300 m. Core points that touch are joined into one cluster; points near a cluster edge join it; everything else is **noise (-1)**. 

| Why DBSCAN for black spots | |
|---|---|
| No need to choose the number of clusters | Hotspots are whatever is dense |
| Handles irregular shapes (a road, not a circle) | K-Means assumes round blobs |
| Isolated crashes are labelled noise | They are not "hotspots" |

Documented run: 8 clusters, 36% noise (matches our 35% uniform background), silhouette 0.91.

### 5.4 `summarise_clusters(df, labels)`

Groups crashes by cluster; counts crashes/fatal/serious, centroid, and the **severity score** = `1*minor + 3*serious + 10*fatal`. Sorted high to low -> `rank`. **Key point:** a cluster with fewer but deadlier crashes can outrank a busier one.

### 5.5 `compute_kde_grid(df)`

Kernel Density Estimation puts a small smooth bump (Gaussian, width 0.8 km) on every crash and adds them up on a 60x60 grid. Result `density` is scaled 0-1 and drawn as a heat-map. It gives a *continuous surface* instead of hard clusters.

### 5.6 `build_folium_map(...)`

Builds a Leaflet map: a severity-weighted heat layer from a sample of crashes + markers for the top-15 DBSCAN hotspots (red = top 3, orange = 4-6, blue = rest). Saved as `reports/hotspot_map.html`.

### 5.7 `synthetic_recovery(...)`

Sanity check unique to synthetic data: for each *planted* hotspot centre, is there a detected hotspot within 1.5 km? **One-to-one matching** (a detected cluster can be used once) so merged clusters cannot inflate the score. Result: 8 of 8.

---

## 6. `src/risk.py` - Pillar 2 (prototype)

```python
table = d.groupby(["grid_id", "cell_lat", "cell_lon", "time_band", "weather_group"]).agg(
    n_crashes=..., n_fatal=..., weighted_score=("weight", "sum"))
table["risk_index"] = table["weighted_score"].rank(pct=True) * 100
```

**What it is:** a lookup of historical severity-weighted crash counts per (1 km cell, time band, weather group), turned into a 0-100 percentile. **What it is not:** a predictive model; there is no traffic-volume (exposure) data, and risk levels are percentile placeholders.

`temporal_validation` is the honest test: build scores from the first 80% of time; check how much of the last 20% of weighted crashes lands in the top-20 cells. **Hit rate** 53.2% vs **random baseline** 0.77% = **PAI 69x**. Because synthetic hotspots never move, this is guaranteed to look good; it becomes real evidence only on real data.

*(Bug we caught while building: a first version used rank correlation over cells that had at least one crash. Sparse cells cannot appear in both past and future, which produced a misleading negative correlation. The hit-rate/PAI method avoids that.)*

---

## 7. `src/explain.py` - SHAP

SHAP gives each feature a contribution to each prediction. We average the absolute contributions over 500 test rows for the **Fatal** class: bigger = the feature moves the fatal score more.

```python
values = _select_class(shap.TreeExplainer(clf).shap_values(Xt), class_index)
# sum one-hot columns of the same original feature per row, then mean(abs(...))
```
`_select_class` hides a library difference (older SHAP returns a list per class, newer returns a 3-D array). One-hot columns are merged so "weather" appears once. Documented run: speed limit (1.02), vehicle type (0.38), road type (0.36), safety gear (0.27), driver age (0.18).

**Limitation:** this explains XGBoost + class weights, not the saved logistic-regression model.

---

## 8. `run_pipeline.py`

`main()` runs six numbered steps and saves everything. Notes:
- Command-line flags via `argparse`.
- SHAP is wrapped in `try/except`, so a library problem does not stop the pipeline.
- `_jsonable` converts NumPy numbers so `json.dumps` works.
- Everything the dashboard needs ends up in `metrics.json` plus CSV/HTML/joblib files.

---

## 9. `app/dashboard.py`

| Function | Role |
|---|---|
| `load_json / load_csv / load_model` | Cached readers; the cache key includes the file's modified time, so new pipeline runs show up on reload |
| `show_html_file` | Embeds the Folium map (`st.iframe`, with a fallback for older Streamlit) |
| `page_status / severity / hotspots / risk / predict` | One function per page |
| `main` | Sidebar navigation; shows a "run the pipeline" error if outputs are missing; adds a SYNTHETIC DATA banner |

Dashboard code contains **no ML logic**; it only displays results and calls `predict_severity` for the what-if form.

---

## 10. `src/data_generator.py` (if asked "how did you get data?")

Creates accidents with: 8 planted hotspot centres (>= 5 km apart) holding 65% of crashes; rush-hour peaks; seasonal weather (more fog in winter, rain in monsoon); and a hidden danger score `z` = base + effects for speed, road, light, weather, surface, vehicle, junction, no safety gear, age, number of vehicles + random noise. Crashes are then labelled Fatal (top 5% of `z`), Serious (next 20%), Minor (rest). Finally it injects missing values, duplicates, and bad coordinates.

**Say this plainly:** the relationships are *our assumptions*, chosen to be plausible. The generator tests the code; it proves nothing about real roads.

---

## 11. Five things to be able to explain from memory

1. **Why not accuracy?** 95% of crashes are not fatal, so ignoring fatal crashes still scores high.
2. **Why is SMOTE inside the pipeline?** To avoid leakage into validation/test data.
3. **Why DBSCAN and K-Means?** Different jobs: broad zones vs dense irregular black spots + noise.
4. **Why severity-weighted ranking?** A fatal crash is not equal to a fender-bender.
5. **What is real and what is not?** Pipeline, tests and dashboard are real; data is synthetic; risk pillar is a prototype.
