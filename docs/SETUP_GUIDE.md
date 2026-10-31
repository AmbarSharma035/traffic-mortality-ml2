# Setup Guide (from a fresh computer)

Tested in this build on **Ubuntu Linux, Python 3.12.3, 1 CPU core, 3 GB RAM** (full pipeline: 36 seconds). Windows/macOS steps below are the standard equivalents; they were not run in the build sandbox.

## 1. Required software

| Software | Version | Why | Check |
|---|---|---|---|
| **Python** | **3.11 or newer** (tested 3.12.3) | Runs everything. The tested scikit-learn 1.8 needs 3.11+ | `python --version` (Windows: `py --version`) |
| **pip** | comes with Python | Installs libraries | `pip --version` |
| **Git** | any recent | Clone / push the repository | `git --version` |
| **A code editor** | VS Code recommended | Reading and editing code | - |
| **A web browser** | Chrome / Edge / Firefox | Dashboard; needs internet for map tiles | - |

**Not needed:** Node.js (no JavaScript build), a database server, Docker, a GPU, any API key.

Hardware: any laptop with ~2 GB free RAM works for the default 20,000-row dataset.

## 2. Install the software

### Windows
1. Install Python from python.org. **Tick "Add python.exe to PATH"** on the first screen.
2. Install Git from git-scm.com (defaults are fine).
3. Open **PowerShell**.

### macOS
```bash
brew install python git
brew install libomp        # needed by LightGBM / XGBoost on macOS
```

### Linux (Debian/Ubuntu)
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git
```

## 3. Get the code

```bash
git clone <your-repo-url> traffic-mortality-ml
cd traffic-mortality-ml
```
(or unzip the project zip and `cd` into the folder.)

## 4. Create a virtual environment (strongly recommended)

A virtual environment keeps this project's libraries separate from everything else.

```bash
python -m venv .venv
```
Activate it:

| System | Command |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

You should now see `(.venv)` at the start of your prompt.

## 5. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

What gets installed (see `requirements.txt`): pandas, numpy, scipy, scikit-learn, imbalanced-learn, xgboost, lightgbm, shap, joblib, plotly, folium, streamlit, pytest. Takes a few minutes; the biggest downloads are lightgbm/xgboost/shap.

Verify:
```bash
python -c "import sklearn, imblearn, xgboost, lightgbm, shap, streamlit, plotly, folium; print('all imports OK')"
```

## 6. Environment variables

**None.** The project needs no `.env` file and no API keys. Every setting is a plain Python constant in `src/config.py`.

## 7. Database setup

**None.** There is no database. Data and results are files created automatically in `data/`, `models/`, `reports/`. (The SRS lists SQLite/PostgreSQL for a later multi-user version.)

## 8. Run the project

```bash
# Step 1: run the pipeline (generates synthetic data, trains models, finds hotspots, writes reports/)
python run_pipeline.py

# Step 2: open the dashboard
streamlit run app/dashboard.py
```
Streamlit prints a URL (normally http://localhost:8501) and opens your browser. Stop it with `Ctrl+C`.

Expected pipeline output (abridged):
```
=== 1/6 Load, clean, engineer features ===
source=synthetic  rows in=20060  rows out=19944
=== 2/6 Severity models (4 models x 3 imbalance strategies) ===
  logistic_regression  class_weight  fatal recall=0.683 ...
best: logistic_regression + class_weight  fatal recall=0.683
...
dbscan: {... 'n_clusters': 8 ...}
recovery: {'planted': 8, 'recovered': 8, ...}
wrote reports/metrics.json  (runtime 36.4s)
```
Your numbers should match when versions match; small differences with other library versions are normal.

### Useful variations

| Goal | Command |
|---|---|
| Fast demo (6,000 rows, 2-fold CV) | `python run_pipeline.py --quick` |
| Skip SHAP (if it errors) | `python run_pipeline.py --skip-shap` |
| No cross-validation | `python run_pipeline.py --cv-folds 0` |
| Bigger synthetic data | `python run_pipeline.py --n-samples 50000` |
| Your own CSV | `python run_pipeline.py --data-path path/to/accidents.csv` |
| Different port | `streamlit run app/dashboard.py --server.port 8600` |
| Run tests | `python -m pytest -q` (expect `24 passed`) |

### Using your own CSV
The file must contain these columns (names exact): `accident_id, timestamp, latitude, longitude, road_type, speed_limit, junction_type, weather, light_condition, road_surface, vehicle_type, driver_age, num_vehicles, safety_gear_used, severity`. `severity` must be 0 (Minor), 1 (Serious) or 2 (Fatal). Real datasets (MoRTH, STATS19, Kaggle) use different names and codes, so you must **rename/recode columns first**; that mapping step is the main pending data task. Details: `DATA_FLOW.md`.

## 9. Daily workflow

1. `cd` into the project, activate `.venv`.
2. Changed code or settings? `python run_pipeline.py`.
3. `python -m pytest -q`.
4. `streamlit run app/dashboard.py`.

## 10. Common setup problems (quick list; full details in `TROUBLESHOOTING.md`)

| Symptom | Likely cause | Quick fix |
|---|---|---|
| `python: command not found` / `'python' is not recognized` | Python not installed or not on PATH | Reinstall with "Add to PATH"; on Windows try `py` |
| `ModuleNotFoundError: No module named 'sklearn'` (or similar) | Dependencies not installed in the active environment | Activate `.venv`, then `pip install -r requirements.txt` |
| PowerShell: "running scripts is disabled" | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, reopen PowerShell |
| `libomp.dylib` not found (macOS) | OpenMP missing | `brew install libomp` |
| Dashboard says "No pipeline results found yet" | Pipeline not run | `python run_pipeline.py`, reload |
| Hotspot map is grey | No internet in browser | Connect to the internet |
| `Address already in use` | Port 8501 taken | Use `--server.port 8600` |
| pip fails building a package | Python too new/old for a wheel | Use Python 3.11 or 3.12 |
