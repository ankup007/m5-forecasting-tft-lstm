# DeepAR M5 Setup Guide

This guide matches the current code in `scripts/` and `src/deepar_m5/`.
The workflow is local-first, with optional Weights & Biases tracking.

## 1. Create the environment

```powershell
conda env create -f environment.yml
conda activate m5-deepar-scratch
```

## 2. Smoke test

```powershell
python scripts\smoke_deepar_m5.py --subset-size 12 --context-length 28 --prediction-length 7 --device cpu
```

## 3. Single training run

```powershell
python scripts\train_deepar_m5.py --subset-size 1000 --context-length 56 --prediction-length 28 --batch-size 128 --epochs 10 --steps-per-epoch 200 --device auto --eval-holdout
```

Useful flags:

- `--sales-file sales_train_validation.csv` for validation-style training.
- `--eval-holdout` to write holdout forecasts and metrics at the end of training.
- `--artifact-dir artifacts\my_run` to control where checkpoints and metrics are written.

Default training writes:

- `best.pt`
- `latest.pt`
- `metrics.json`
- `holdout_forecasts_<mode>.csv` when `--eval-holdout` is set
- `holdout_forecasts_<mode>_rounded.csv` when `--eval-holdout` is set
- `holdout_metrics_all_modes.json` when `--eval-holdout --eval-wrmsse` is set

## 4. Experiment driver

The experiment script does not accept comma-separated sweep lists. It runs the hardcoded grid in `src/deepar_m5/experiments.py`.

```powershell
python scripts\run_deepar_m5_experiments.py --output-dir artifacts\deepar_m5_experiments --device cpu
```

Each run gets its own subdirectory under `output-dir`. The sweep also writes a timestamped summary file like `summary_YYYYMMDD_HHMMSS_ffffff.csv`.

Per run, the following are written:

- `best.pt`
- `latest.pt`
- `metrics.json`
- `holdout_forecasts_mean.csv`
- `holdout_forecasts_mean_rounded.csv`
- `holdout_forecasts_sample-mean.csv`
- `holdout_forecasts_sample-mean_rounded.csv`
- `holdout_forecasts_p25.csv`
- `holdout_forecasts_p25_rounded.csv`
- `holdout_forecasts_p75.csv`
- `holdout_forecasts_p75_rounded.csv`
- `holdout_metrics_all_modes.json` when `--eval-wrmsse` is set
- `holdout_metrics.json` as a compatibility copy when `--eval-wrmsse` is set

## 5. Evaluation mode

```powershell
python scripts\evaluate_deepar_m5.py --checkpoint artifacts\my_run\best.pt --sales-file sales_train_evaluation.csv --output-dir evaluations\test_run
```

This writes evaluation metrics and forecast CSV files for the checkpoint.

## 6. Prediction mode

```powershell
python scripts\predict_deepar_m5.py --checkpoint artifacts\my_run\best.pt --output predictions\submission_v1.csv
```

For quantile or sample-based forecasts:

```powershell
python scripts\predict_deepar_m5.py --checkpoint artifacts\my_run\best.pt --output predictions\submission_p90.csv --forecast-mode quantile --quantile 0.9 --num-samples 500 --sample-seed 42
```

## 7. Weights & Biases

Enable W&B with the CLI flag:

```powershell
python scripts\run_deepar_m5_experiments.py --wandb --output-dir artifacts\deepar_m5_experiments --device cpu
```

The script itself preconfigures the W&B project, group, tags, and mode defaults.

## 8. Full argument reference

The authoritative flag list is in the code:

- `src/deepar_m5/train.py`
- `src/deepar_m5/experiments.py`
- `src/deepar_m5/infer.py`

When in doubt, use `python scripts\train_deepar_m5.py --help` or the equivalent command for the script you are running.
