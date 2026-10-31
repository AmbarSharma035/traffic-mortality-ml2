# Troubleshooting

How to read this page:

- **[VERIFIED]** = I triggered this in the build sandbox (Ubuntu, Python 3.12.3) and the message/behaviour is exactly as written.
- **[COMMON]** = a standard environment problem for this kind of stack. The fix is the usual one, but I did **not** reproduce it here, so treat it as a strong hint, not a guarantee.

This project has **no database, no REST API, no environment variables and no separate frontend build**, so there are no errors of those kinds; the matching sections say so.

---

## 1. Python and dependency problems

### 1.1 `python` is not recognised / `command not found` **[COMMON]**
**Why:** Python is not installed, or not on PATH.
**Fix:** Install Python 3.11+ (tick *Add python.exe to PATH* on Windows). Try `py --version` (Windows) or `python3 --version` (macOS/Linux). Use whichever works consistently; in the commands below replace `python` with that name.

### 1.2 `ModuleNotFoundError: No module named 'sklearn'` (or `pandas`, `imblearn`, `xgboost`, `lightgbm`, `shap`, `streamlit`, `plotly`, `folium`) **[COMMON]**
**Why:** Libraries are not installed in the environment you are running, usually because the virtual environment is not activated.
**Fix:**
```bash
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Check: `which python` / `where python` should point inside `.venv`.

### 1.3 `ModuleNotFoundError: No module named 'src'` **[COMMON]**
**Why:** You are importing the project from outside its folder (for example `python -c "from src import ..."` from another directory, or pytest run from the wrong place).
**Fix:** `cd` into the project root. Running `python run_pipeline.py` and `streamlit run app/dashboard.py` work from any directory because they add the project folder to the path **[VERIFIED]**; `python -m pytest` must be run from the project root (it uses `pytest.ini`).

### 1.4 `pip install` fails while building a wheel / "No matching distribution" **[COMMON]**
**Why:** A very old or very new Python without prebuilt packages.
**Fix:** Use Python 3.11 or 3.12; `python -m pip install --upgrade pip`; retry.

### 1.5 `TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse_output'` **[COMMON]**
**Why:** scikit-learn older than 1.2.
**Fix:** `pip install --upgrade "scikit-learn>=1.4"`.

### 1.6 `OSError: libomp.dylib ... not found` (macOS, XGBoost/LightGBM) **[COMMON]**
**Why:** The OpenMP runtime is missing.
**Fix:** `brew install libomp`, restart the terminal.

### 1.7 PowerShell: "running scripts is disabled on this system" **[COMMON]**
**Why:** Windows blocks the venv activation script.
**Fix:** `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, reopen PowerShell, activate again. Or use `cmd` with `.venv\Scripts\activate.bat`.

### 1.8 Numbers differ slightly from the docs **[COMMON, expected]**
**Why:** Different library versions change tie-breaking and random streams. The documented numbers came from the versions listed at the top of `requirements.txt`.
**Fix:** Nothing to fix. Compare the *pattern* (class weights raise fatal recall; hotspots recovered), not the third decimal.

---

## 2. Pipeline and data errors

### 2.1 `ValueError: Input data is missing required columns: [...]` **[VERIFIED]**
**Why:** Your CSV lacks columns from the schema (`config.REQUIRED_COLUMNS`). Real datasets use different names/codes.
**Fix:** Rename/recode your columns to match the schema in `DATA_FLOW.md` (severity must be 0/1/2) in a small script or notebook, save a new CSV, then `python run_pipeline.py --data-path new.csv`.

### 2.2 `FileNotFoundError: Data file not found: missing.csv` **[VERIFIED]**
**Why:** The `--data-path` is wrong.
**Fix:** Give a correct path (quote paths with spaces on Windows).

### 2.3 `run_pipeline.py: error: unrecognized arguments: --bogus` **[VERIFIED]**
**Why:** A flag typo. Valid flags: `--data-path`, `--n-samples`, `--cv-folds`, `--skip-shap`, `--quick`, `-h`.

### 2.4 `ValueError: Unknown model: svm` / `Unknown imbalance strategy: magic` **[VERIFIED]**
**Why:** Names must be in `config.MODEL_NAMES` (`logistic_regression`, `random_forest`, `xgboost`, `lightgbm`) and `config.IMBALANCE_STRATEGIES` (`none`, `class_weight`, `smote`).

### 2.5 After cleaning, almost all rows disappear **[COMMON]**
**Why:** With real data: severity not coded 0/1/2 (everything dropped by step 2), coordinates stored as text, or timestamps in an unusual format.
**Fix:** Read `data.cleaning_report` in `reports/metrics.json` or the printed `cleaning:` line to see which check removed rows, then fix that column before loading.

### 2.6 SHAP step prints `SHAP failed: ...` and `metrics.json` shows `"status": "failed: ..."` **[COMMON]**
**Why:** SHAP/XGBoost/LightGBM version incompatibility. The pipeline deliberately continues.
**Fix:** `pip install --upgrade shap xgboost lightgbm`, or run with `--skip-shap`. The dashboard then shows a SHAP warning instead of the chart.

### 2.7 Pipeline is very slow or "Killed" (out of memory) **[COMMON]**
**Why:** Large `--n-samples` on a small machine; 12 experiments plus CV multiply the work.
**Fix:** `--quick`, smaller `--n-samples`, `--cv-folds 0`, or `--skip-shap`. In the sandbox (1 CPU, 3 GB) the default run took 36 s.

### 2.8 DBSCAN finds 0 clusters (or one giant cluster) on real data **[COMMON]**
**Why:** `DBSCAN_EPS_KM` / `DBSCAN_MIN_SAMPLES` suit the synthetic spread, not your data's density.
**Fix:** Increase `eps` or lower `min_samples` if 0 clusters; do the opposite if one giant cluster. Edit `src/config.py`, re-run.

### 2.9 SMOTE error about `n_neighbors` / "Expected n_neighbors <= n_samples" **[COMMON]**
**Why:** A class has too few examples in a training fold (tiny datasets).
**Fix:** Use more data, fewer CV folds, or the `class_weight` strategy.

---

## 3. Dashboard (frontend) errors

### 3.1 "No pipeline results found yet. Run `python run_pipeline.py` ..." **[VERIFIED]**
**Why:** `reports/metrics.json` (or the other result files) does not exist yet.
**Fix:** `python run_pipeline.py`, then reload the browser tab.

### 3.2 "Could not load the saved model (...). Re-create it with `python run_pipeline.py`" (Predict page) **[COMMON]**
**Why:** `models/severity_best.joblib` was saved with different library versions than the ones now installed (typical after deploying or switching Python).
**Fix:** Re-run the pipeline in the *same* environment as the dashboard.

### 3.3 Hotspot map is grey / has no tiles **[COMMON]**
**Why:** The browser must download OpenStreetMap tiles and the Leaflet library; no internet or a firewall blocks them.
**Fix:** Connect to the internet. The ranked table and KDE chart still work offline.

### 3.4 `Address already in use` / port 8501 busy **[COMMON]**
**Fix:** `streamlit run app/dashboard.py --server.port 8600`, or stop the other Streamlit (`Ctrl+C`).

### 3.5 Page shows old numbers after re-running the pipeline **[COMMON]**
**Why:** Browser tab not refreshed. Server-side caches are keyed on file modification time, so a reload is enough.
**Fix:** Press `R` in the app or reload the page.

### 3.6 `AttributeError: module 'streamlit' has no attribute 'iframe'` **[COMMON]**
The dashboard checks for `st.iframe` and falls back to the older `components.html`, so you should not see this. If you modify the code and call `st.iframe` directly on an older Streamlit, use `show_html_file` instead or `pip install --upgrade streamlit`.

### 3.7 Streamlit prints deprecation warnings **[VERIFIED for `components.html`]**
Streamlit deprecates APIs quickly. We moved the map embed to `st.iframe` for that reason. Warnings do not stop the app; fix them by upgrading the call named in the message.

### 3.8 Tests: dashboard tests are *skipped* **[VERIFIED by design]**
**Why:** `tests/test_dashboard.py` needs `reports/metrics.json`.
**Fix:** `python run_pipeline.py` first, then `python -m pytest -q` (expect `24 passed`).

---

## 4. Database / API / environment-variable issues

| Topic | Status |
|---|---|
| Database errors (connection refused, migrations...) | **Not applicable**: there is no database. Results are files in `data/`, `models/`, `reports/` |
| REST API errors (CORS, 404/500, tokens) | **Not applicable**: there is no API server. External calls at run time: only the browser loading map tiles |
| Missing environment variable / `.env` | **Not applicable**: none are required (settings are in `src/config.py`) |
| API keys (weather etc.) | Not used in this version. A weather-enrichment step is planned and would add a key later |

---

## 5. Deployment issues (Streamlit Community Cloud) **[COMMON; not deployed from the sandbox]**

| Symptom | Likely cause | Fix |
|---|---|---|
| Build fails installing requirements | Python version on the platform too old/new | In *Advanced settings* choose Python 3.11 or 3.12; keep `requirements.txt` at the repo root |
| App starts but shows "No pipeline results found" | `reports/` not committed (or ignored) | Commit `reports/` and `models/` (small), or run `python run_pipeline.py` and push the outputs |
| Predict page says model could not load | scikit-learn version differs from the one that saved the model | Re-run pipeline locally with versions matching the platform's, or pin versions in `requirements.txt`, re-save, push |
| `ModuleNotFoundError: src` | Wrong *main file path* | Set main file to `app/dashboard.py` (the file adds the project root to `sys.path` itself) |
| Slow first load | Cold start installing packages | Wait; later loads are quick |
| Large repository warnings | Committed `data/raw/*.csv` | They are git-ignored by default; do not force-add them |

---

## 6. Quick diagnostic checklist

```bash
python --version                                  # 3.11+ ?
pip list | grep -iE "scikit-learn|imbalanced|xgboost|lightgbm|shap|streamlit"
python run_pipeline.py --quick                    # does the pipeline run end to end?
python -m pytest -q                               # 24 passed ?
streamlit run app/dashboard.py                    # does the app open?
```
If `--quick` works but the default run fails, the cause is almost always memory or time; if tests fail, read the first failure only: later ones are usually knock-on effects.
