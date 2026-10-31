# Project Q&A: Viva / Presentation / Interview

Answers are written to be spoken. Numbers come from the documented run on **synthetic data** (see README). **Rule one: if you do not know, say so and say how you would find out.** Examiners respect honesty far more than a confident wrong answer.

Contents: **Part A** core questions | **Part B** 20 beginner | **Part C** 20 intermediate | **Part D** 10 advanced | **Part E** tricky questions about our own status

---

## Part A: Core questions

### A1. What problem does the project solve?
Road-crash data is normally used after the fact (annual tallies). We turn it into three tools that help *act before* the next crash: predict how severe a crash is likely to be, find where crashes concentrate and rank those places by severity, and (prototype) score how dangerous a place and time are. The aim is to help police and highway engineers target limited resources. Context: ~1.19 million road deaths a year worldwide (WHO 2023); 172,890 deaths in India in 2023 (MoRTH).

### A2. How does the project work?
One command, `python run_pipeline.py`: (1) load data (real CSV or synthetic), (2) clean it and report what was removed, (3) add time/weather features, (4) train 12 severity experiments (4 models x 3 imbalance strategies) and pick the best by fatal-class F2, (5) compute SHAP importance, (6) find hotspots with K-Means, DBSCAN and KDE, (7) build the risk prototype, (8) save results. A Streamlit dashboard then displays them and scores what-if crashes.

### A3. Why did you choose these technologies?
Python + pandas/scikit-learn because they are the standard data-science stack and scikit-learn Pipelines prevent data leakage. imbalanced-learn for SMOTE inside the pipeline. XGBoost/LightGBM because boosted trees are strong on tabular data (named in our synopsis). SHAP for explanations. Streamlit + Plotly + Folium because a whole dashboard and map can be built in Python alone. Full table in `PROJECT_OVERVIEW.md`.

### A4. Explain the architecture.
A batch pipeline (backend) writes files; a Streamlit dashboard (frontend) reads them. No database, no REST API. Inside the pipeline there are three independent pillars (severity, hotspots, risk) fed by the same cleaned data, and an explainability step for severity. They are fused only at the dashboard. See the diagram in `ARCHITECTURE.md`.

### A5. Explain the database.
There is none, deliberately. The data is a few MB, there is one user, and everything is batch, so CSV/JSON/joblib files are simpler and enough. Results live in `reports/` and `models/`. The SRS mentions SQLite/PostgreSQL for a later multi-user version with audit logs and review decisions; that is not built, and I would not claim it is.

### A6. Explain the APIs.
There is no REST API in this version. Modules expose Python functions (`clean_accidents`, `make_pipeline`, `run_dbscan`, `predict_severity` ...), and `run_pipeline.py` is a command-line interface. At run time we call no external API; only the viewer's browser loads OpenStreetMap tiles for the map. A FastAPI/Flask layer around `predict_severity` is a natural next step.

### A7. Explain the important algorithms.
- **Logistic Regression / Random Forest / XGBoost / LightGBM**: the four classifiers, from simple/interpretable to powerful.
- **SMOTE and class weights**: two ways to stop the rare fatal class being ignored.
- **Stratified split and k-fold CV**: honest evaluation that keeps the 5% fatal share.
- **K-Means**: broad zones. **DBSCAN**: dense irregular black spots plus noise. **KDE**: smooth density surface.
- **SHAP**: how much each feature pushes the fatal score.
- **Severity-weighted scoring** (1/3/10) for ranking, **hit rate/PAI** for the risk sanity check.

### A8. What challenges were faced?
Real ones from building this:
1. **Imbalance**: the best-looking models by accuracy found only 10-22% of fatal crashes. Fixed by comparing three imbalance strategies and choosing by fatal F2.
2. **No real data yet**: so we built a generator with planted hotspots to be able to test the code (can the clustering find what we hid?).
3. **A planted-hotspot check that was too generous**: two planted centres were 1.3 km apart and merged into one cluster, yet "8 of 8" was reported. We enforced >= 5 km separation and made the matching one-to-one.
4. **A misleading validation**: our first risk check gave a negative correlation purely because sparse cells cannot appear in both past and future. We replaced it with hit-rate/PAI.
5. **Library churn**: Streamlit deprecated `components.html`; we moved to `st.iframe` with a fallback. SHAP's output shape differs across versions; `_select_class` handles both.

### A9. What could be improved?
In priority order: (1) real dataset, (2) hyper-parameter and threshold tuning, (3) probability calibration, (4) a real risk model with exposure data, (5) LIME and per-crash explanations, (6) SHAP for the saved model, (7) time/space-aware validation, (8) deployment and CI, (9) streaming input.

### A10. What happens if something fails?
Bad input columns -> clear `ValueError`. Bad rows -> removed and counted. SHAP failure -> caught, pipeline continues, dashboard shows a warning. Missing pipeline outputs -> dashboard says to run the pipeline. Model not loadable (version mismatch) -> the Predict page says to re-run the pipeline. No internet -> map tiles blank, rest fine. Table in `ARCHITECTURE.md` section 6.

---

## Part B: 20 beginner questions

**B1. Describe the project in one sentence.** It uses machine learning on accident records to predict crash severity, find dangerous locations and (in prototype) score risk, so road-safety teams can act before the next crash.

**B2. What is machine learning?** Letting a program learn patterns from examples instead of writing the rules by hand. Here: examples of past crashes with known outcomes.

**B3. What is classification?** Predicting a category. Ours: Minor, Serious or Fatal.

**B4. What are the three severity classes?** 0 = Minor, 1 = Serious, 2 = Fatal (`SEVERITY_LABELS` in `config.py`).

**B5. What is class imbalance?** One class is much rarer. Fatal is only ~5% of crashes (997 of 19,944), Serious ~20%, Minor ~75%.

**B6. Why is accuracy misleading here?** A model that almost never predicts "fatal" is still ~95% right on that class. In our run, logistic regression with no handling had 79% accuracy but found only 18% of fatal crashes.

**B7. What is recall?** Of all *real* fatal crashes, the fraction we caught. Best model: 136 of 199 = 0.683.

**B8. What is precision?** Of the crashes we *called* fatal, the fraction that really were. Best model: 0.252, so about 1 in 4 fatal alarms is right. That is the price of high recall.

**B9. What is a hotspot (black spot)?** A place where crashes concentrate far more than elsewhere.

**B10. What is clustering?** Grouping points without labels, by closeness. We cluster crash coordinates.

**B11. What is a training set vs a test set?** Training data is what the model learns from; test data is held back (20%) to check it works on crashes it has never seen.

**B12. What is a feature?** An input column the model uses: speed limit, weather, hour, vehicle type, helmet used...

**B13. What is one-hot encoding?** Turning a category into 0/1 columns: weather = Rain becomes `weather_Rain = 1`, others 0. Models need numbers.

**B14. What is imputation?** Filling missing values (median for numbers, "Unknown" for categories). We do it inside the pipeline so only training data decides the fill value.

**B15. What is synthetic data and why use it?** Data we generated with known rules. Real data is not integrated yet; synthetic data lets us test the whole pipeline and check, for example, that clustering finds hotspots we planted. It is **not** evidence about real roads.

**B16. What does `run_pipeline.py` do?** Runs all steps in order and writes every result file the dashboard needs.

**B17. What is a confusion matrix? Read ours.** A table of actual vs predicted. Ours (rows actual Minor/Serious/Fatal, columns predicted): `[[2254, 609, 129], [179, 344, 275], [7, 56, 136]]`. Bottom row: of 199 fatal crashes, 136 predicted fatal, 56 predicted serious, 7 predicted minor. So only 7 fatal crashes were called Minor.

**B18. What is SHAP?** A method that shows how much each feature pushes a prediction up or down. We use it to see what drives the *fatal* score: speed limit is the biggest.

**B19. What is overfitting?** Memorising the training data so it fails on new data. We guard with a separate test set, cross-validation and limited tree depth.

**B20. Why a dashboard?** A model in a notebook helps nobody on the road. The dashboard lets a non-technical user see results and try what-if crashes without code.

---

## Part C: 20 intermediate questions

**C1. Why choose the best model by fatal F2?** F2 weighs recall 4x more than precision (`5PR/(4P+R)`), reflecting that a missed fatal crash costs more than a false alarm. Pure recall would be gamed by predicting everything fatal; F2 keeps a little precision pressure.

**C2. How does SMOTE work and why is it inside the pipeline?** It creates synthetic minority examples between a rare-class point and one of its nearest same-class neighbours. Inside the `imblearn` pipeline it runs only on the data used for fitting (training split or CV training fold), never on test/validation data, so evaluation stays honest.

**C3. What does `class_weight="balanced"` do?** Makes mistakes on rare classes count more in the loss (weight inversely proportional to class frequency). No new data is created. In our run it beat SMOTE for the tree models.

**C4. What is stratified splitting / k-fold CV?** Splits that keep the class ratio. The 80/20 split and each CV fold have ~5% fatal. CV (3 folds) trains on 2/3, tests on 1/3 and rotates, giving mean +/- std (LR fatal recall 0.667 +/- 0.023).

**C5. Why did logistic regression win? Does that mean it is better than XGBoost?** No. Our generator's danger score is additive, which a linear model fits well. It says something about our synthetic data, not about real roads. On real, messier data tree ensembles may win; the pipeline lets us re-test in one command.

**C6. Explain DBSCAN's two parameters.** `eps` = neighbourhood radius (0.3 km); `min_samples` = crashes needed within that radius to count as dense (30). A point with >= 30 neighbours is a core point; touching core points join one cluster; sparse points are noise. We chose values by reasoning about the planted spread; on real data they must be tuned (e.g., k-distance plot).

**C7. K-Means vs DBSCAN?** K-Means needs `k`, forces every point into a cluster, assumes round clusters. DBSCAN needs a radius and density, finds any shape, labels sparse points noise. We use K-Means for broad zones, DBSCAN for black spots.

**C8. What is the silhouette score?** For each point, compares closeness to its own cluster vs the nearest other cluster: -1 bad to +1 great. K-Means 0.59, DBSCAN 0.91 (non-noise points, on a sample). Caveat: it favours compact blobs and is computed on a 5,000-point sample.

**C9. What is KDE and bandwidth?** Kernel Density Estimation places a smooth bump on each crash and sums them to a density surface. Bandwidth (0.8 km) = bump width: small = spiky, large = over-smoothed.

**C10. Why convert latitude/longitude to km?** A degree of longitude is shorter than a degree of latitude away from the equator, so raw degrees distort distance. `to_km` makes `eps=0.3` really 300 m.

**C11. Why weights 1 / 3 / 10 for the severity score?** They are our assumption, so a fatal crash counts much more than a minor one. Road-safety agencies use cost-based weights (like "equivalent property-damage-only" scores); we should replace ours with a sourced set. It is one config line (`SEVERITY_WEIGHTS`).

**C12. What is data leakage and how do you avoid it?** Test or future information influencing training. We avoid it by (a) fitting imputer/scaler/encoder/SMOTE inside the pipeline on training data only, (b) a stratified hold-out split, and (c) using only crash-time features: no injury counts or damage level, which are known only after the crash.

**C13. Why don't you impute in `clean_accidents`?** Computing medians on the full data before splitting leaks test-set statistics. Imputing inside the pipeline learns them from training data only.

**C14. ROC-AUC vs PR-AUC?** ROC-AUC (0.84) measures ranking quality overall but can look good under heavy imbalance. PR-AUC focuses on the positive (fatal) class: ours 0.36 against a random baseline of ~0.05, i.e., better than chance but modest.

**C15. How does the what-if prediction work?** The form builds a one-row table, `engineer_features` adds time/weather features, and the saved pipeline (imputer, encoder, logistic regression) returns three probabilities. Same code as training, so no mismatch.

**C16. Why joblib, and is there a risk?** It saves the whole fitted pipeline in one file. Risk: the file is tied to library versions; loading with a different scikit-learn can fail. That is why the dashboard shows "re-run the pipeline" instead of crashing.

**C17. How did you test it?** 24 pytest tests: generator/cleaning/feature checks, each model family trains and predicts, class weighting raises fatal recall, DBSCAN finds >= 6 of 8 planted hotspots, risk index in 0-100, status stays near 60%, and each dashboard page renders without errors (Streamlit `AppTest`).

**C18. Why these four models?** From the synopsis: Logistic Regression as an interpretable baseline; Random Forest, XGBoost and LightGBM to capture non-linear interactions. Comparing a linear model and three tree families shows whether complexity helps.

**C19. SHAP explains XGBoost but the saved model is logistic regression. Is that a problem?** Yes, a known limitation. SHAP's `TreeExplainer` needs a tree model, so we explain the best *tree* model. Fixes: use SHAP's linear explainer for the saved model, or save the best tree model for explanation. Listed as pending.

**C20. What exactly is the risk pillar now?** A table of historical severity-weighted crashes per ~1 km cell x time band x weather group, converted to a 0-100 percentile with placeholder levels. Not predictive, no exposure data. The past-to-future check (top-20 cells hold 53% of later weighted crashes vs 0.77% random) is guaranteed to look good because synthetic hotspots never move.

---

## Part D: 10 advanced questions

**D1. Are your predicted probabilities trustworthy?** No. Class weights and SMOTE shift the probabilities (they overstate the fatal chance), so they are relative scores. Fix: calibrate (Platt scaling or isotonic regression) on a validation set with the original class proportions, and check reliability curves / Brier score.

**D2. How would you choose the decision threshold?** `argmax` is an arbitrary cut. Better: sweep thresholds on a validation set and pick by expected cost, e.g., cost(missed fatal) = k x cost(false alarm), or require recall >= a target and maximise precision. Then confirm once on the untouched test set.

**D3. Is random splitting valid for crash data?** Not fully. Crashes are spatially and temporally correlated, so nearby/same-day crashes in train and test inflate scores. Use temporal splits (train on past, test on future) and spatial block CV. Our risk check uses a temporal split; the severity split is random (limitation).

**D4. What is wrong with SMOTE on categorical features?** SMOTE interpolates numbers; after one-hot encoding it can create fractional "categories". SMOTENC handles mixed types; or use class weights, which avoid synthetic points entirely. Our results (class weights >= SMOTE) are consistent with that concern, but not proof.

**D5. Why is "risk" not the same as "number of crashes"?** Risk is crashes per unit of exposure (traffic volume, distance). A busy road has many crashes even if safe per vehicle. Without exposure data our score only says where crashes concentrate. A proper model would use count models (Poisson / negative binomial) with exposure as an offset, and empirical-Bayes smoothing to reduce regression-to-the-mean (hotspots chosen after a spike often "improve" by chance).

**D6. What data problems exist with real crash records?** Under-reporting (esp. minor crashes), police coding differences, changing severity definitions (STATS19 has 3 levels; Indian reports use fatal / grievous / minor / non-injury), inaccurate GPS, and missing fields. Our 3-class target must be mapped carefully per dataset, and results are conditional on *reported* crashes.

**D7. Can you say model A is better than model B from these results?** Not from one test split. With only 199 fatal test cases, fatal recall 0.683 has a 95% interval of roughly +/- 6.5 points (binomial SE ~ 0.033). Differences like 0.52 vs 0.56 are inside the noise. We would need repeated CV, bootstrap intervals or a paired test (e.g., McNemar) before ranking models.

**D8. How would DBSCAN/KDE scale to millions of crashes?** DBSCAN with tree indexes is roughly O(n log n) in low dimensions but memory heavy; KDE evaluation is O(grid x n), so we sub-sample to 20,000 points and use a tolerance (`rtol`). Options: grid-bin counts first, HDBSCAN, spatial databases (PostGIS), or Spark for distributed processing.

**D9. How would you deploy it for real-time use?** Wrap `predict_severity` in a FastAPI service, ingest live feeds (message queue), keep a feature store for weather/traffic, version models, monitor data drift and recall on newly labelled crashes, add authentication and audit logs (SRS requirements), and retrain on a schedule. None of this exists yet.

**D10. What are the ethical risks?** Policing bias (more patrols create more recorded crashes, reinforcing the pattern), over-trusting a score, privacy of precise locations, and unfair burden on communities. Mitigation: decision-support only, human review, show uncertainty and limits, explain drivers, audit outcomes, and keep personal data out of outputs.

---

## Part E: Tricky questions about our own status

**E1. Is the data real?** No, it is synthetic. Everything is a pipeline check. Integrating a real dataset (MoRTH / STATS19 / Kaggle) is the largest pending item.

**E2. What does "60% complete" mean? How did you compute it?** A weighted estimate in `src/project_status.py`: data 15 x 80%, severity 25 x 80%, hotspots 15 x 80%, risk 15 x 20%, explainability 10 x 50%, dashboard 10 x 50%, engineering 10 x 30% = 60 points. It is our own estimate; the table lists exactly what is done and pending, so you can judge the number yourself.

**E3. Why is accuracy of the best model only 68.5%?** Because we optimise for finding fatal crashes, not overall correctness. The model with the best accuracy (79%) misses 82% of fatal crashes. We choose the trade-off on purpose, and the threshold is still untuned.

**E4. Your team's research paper reports about 96% accuracy. Why not here?** Different dataset (Illinois, USA municipal open data), a binary target (injury/tow vs none), and a different feature set; the numbers are not comparable. Be careful also with that paper's Table IX: features such as `INJURIES_TOTAL` and `MOST_SEVERE_INJURY` are only known *after* a crash, and if they were model inputs for an injury target they would leak the answer. Check this in the paper's code before defending those numbers. In this project such post-crash variables are excluded on purpose.

**E5. Which parts of the synopsis are not implemented?** LIME, GeoPandas, hyper-parameter search (Optuna), a real risk model, the unified dashboard queries, streaming/telematics input, deployment, and benchmarking against published results.

**E6. What would you do in the next month?** Get a real dataset and write the column-mapping step; re-run all experiments; add time-based validation, tuning, calibration and threshold selection; then build the real risk model.
