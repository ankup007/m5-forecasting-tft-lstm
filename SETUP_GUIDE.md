# DeepAR M5 Experiment & Evaluation Guide

This guide describes the complete end-to-end workflow for running experiments, calculating metrics aligned with competition rules, and deploying the results to the interactive viewer.

## 1. Run an Experiment Sweep

Experiments are configured in `src/deepar_m5/experiments.py` via the `GRID_CONFIG` dictionary.

```powershell
python scripts\run_deepar_m5_experiments.py
```

**Outputs:**
- A unique run directory under `artifacts\deepar_m5_experiments\run_YYYYMMDD_...`
- `holdout_forecasts_*.csv`: Raw and rounded forecasts for multiple modes (mean, p50, etc.).
- `run_config.json`: Consolidated hyperparameters (subset size, hidden dimension, layers, etc.).
- `metrics.json`: Internal training/validation NLL history.

## 2. Calculate Accurate WRMSSE Scores

We use a local implementation that is mathematically aligned with the official `m5-wrmsse` package (trims leading zeros, uses correct holdout actuals).

```powershell
python scripts\calculate_run_wrmsse.py "artifacts\deepar_m5_experiments\run_NAME"
```

**Outputs:**
- `series_json\wrmsse.json`: Contains WRMSSE scores for all 12 hierarchical levels across all forecast variants.

## 3. Generate JSON Artifacts for the UI

Convert large CSV forecasts into per-series JSON objects to enable instant loading in the web viewer.

```powershell
python scripts\ui\build_series_json_artifacts.py "artifacts\deepar_m5_experiments\run_NAME"
```

**Outputs:**
- `series_json\series\*.json`: 30,000+ tiny JSON files.
- `series_json\run_summary.json`: Metadata for the UI dropdowns.

## 4. Deploy to the Documentation (Web UI)

This step moves the artifacts to the `docs/` folder and rebuilds the static HTML viewer.

```powershell
python scripts\ui\build_github_pages_bundle.py --run-dir "artifacts\deepar_m5_experiments\run_NAME"
```

**Outputs:**
- `docs\run_NAME\`: Copied artifacts.
- `docs\index.html`: Updated viewer including the new run.

---

## 5. View the UI Locally

Do not open `docs\index.html` by double-clicking it from the filesystem. The viewer loads JSON files and should be served over HTTP.

If you are already in the repository root, run:

```powershell
python -m http.server 8000 --directory docs
```

Then open:

```text
http://127.0.0.1:8000/index.html
```

If you prefer to `cd` into `docs/` first, the equivalent is:

```powershell
cd docs
python -m http.server 8000
```

Then open the same URL:

```text
http://127.0.0.1:8000/index.html
```

If port `8000` is already in use, replace it with another port such as `8080` in both the command and the browser URL.

---

## Directory Organization

- `scripts/`: Core training, evaluation, and experiment drivers.
- `scripts/ui/`: Artifact packaging, HTML report builders, and visualization tools.
- `src/deepar_m5/`: Source code for the DeepAR model, data loading, and evaluation logic.
- `docs/`: Deployment-ready JSON artifacts and the static HTML viewer.
