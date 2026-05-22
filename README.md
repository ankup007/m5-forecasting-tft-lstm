# M5 Forecasting: TFT vs DeepAR/LSTM Architecture Comparison

This repository examines how **Temporal Fusion Transformer (TFT)** and **DeepAR/LSTM** fit the [Kaggle M5 Forecasting Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) problem. The goal is not just to ask which model is more sophisticated, but which architecture is the more practical direction under M5's dataset complexity, hierarchy, covariates, and compute constraints.

## Project Overview

M5 is a large retail demand forecasting problem: 30,490 bottom-level item-store series, 42,840 evaluated hierarchical series, known future calendar and SNAP signals, weekly prices, sparse demand, and a fixed 28-day forecast horizon.

The analysis frames model choice around three practical questions:

- **Architecture fit:** TFT naturally matches M5's static metadata, known future covariates, observed history, and multi-horizon forecasting structure.
- **Operational feasibility:** DeepAR/LSTM is a more tractable first neural baseline because it is global, compact, probabilistic, and easier to scale.
- **Experimentation design:** Before training on all 30K+ bottom-level series, compare DeepAR/LSTM and a compact TFT on representative subsets that preserve complete time histories and hierarchy coverage.

## Contents

- [TFT vs DeepAR/LSTM for M5 Forecasting: Dataset Complexity, Architecture, and Compute Tradeoffs](./m5_tft_lstm_architectural_comparison.md)

  The core analysis document covers:
  - M5 dataset complexity and forecasting challenges.
  - DeepAR/LSTM architecture, strengths, limitations, and practical configuration.
  - TFT architecture, covariate handling, interpretability, and compute cost.
  - Mathematical framing of DeepAR's autoregressive likelihood versus TFT's direct multi-horizon modeling.
  - Compute and training requirements across memory, sequence length, feature count, sampling, and hardware.
  - A staged recommendation with subset-based experimentation before full-data training.

## Summary Findings

| Direction | Best For | Main Tradeoff |
|---|---|---|
| **DeepAR/LSTM** | First full-data neural baseline | Less explicit handling of known future covariates and feature relevance |
| **TFT** | Rich covariate-aware modeling and interpretability | Higher memory, tuning, and training-time cost |
| **Subset pilot** | Early architecture comparison | Must sample coherent series, not random rows |

## Recommended Workflow

Start with a representative pilot rather than immediately training both neural models on the full dataset. A useful pilot might use roughly 500 to 1,000 complete item-store series, stratified across category, department, store, state, sales volume, intermittency, and price behavior. Scale to 3,000 to 5,000 series only after the feature pipeline, time split, metrics, and runtime profile are stable.

The comparison should keep the validation horizon, feature set, context length, and training budget fixed across models. Track not only accuracy, but also memory use, training time, inference time, horizon-wise error, and error by demand bucket. This makes the final full-data direction an engineering decision rather than a guess based only on architectural appeal.

## DeepAR From-Scratch Pipeline

The repository includes a PyTorch-only DeepAR implementation under `src/deepar_m5`. It does not use forecasting libraries or built-in recurrent layers; the LSTM cell, negative-binomial likelihood, sampled training windows, training loop, and autoregressive inference are implemented directly.

For a code walkthrough, tuning guide, and extension notes, read [DeepAR Implementation Guide](./IMPLEMENTATION_README.md).

For the training/prediction mechanics, including teacher forcing, autoregressive rollout, Negative Binomial likelihood, and mean versus sampled forecasts, read [Understanding DeepAR Training And Prediction](./DEEPAR_TRAINING_AND_PREDICTION_README.md).

Create the isolated conda environment:

```powershell
conda env create -f environment.yml
conda activate m5-deepar-scratch
```

Run a smoke test:

```powershell
python scripts\smoke_deepar_m5.py --subset-size 12 --context-length 28 --prediction-length 7 --device cpu
```

Train a pilot model:

```powershell
python scripts\train_deepar_m5.py --subset-size 1000 --context-length 56 --prediction-length 28 --batch-size 128 --epochs 10 --steps-per-epoch 200 --device auto
```

By default, training uses `sales_train_evaluation.csv`, which contains the same 30,490 item-store series as `sales_train_validation.csv` plus the extra actual sales days `d_1914` through `d_1941`. With a 28-day horizon, the script trains on windows ending before those final 28 known days and uses `d_1914` through `d_1941` as its internal validation holdout. To simulate the original validation-stage forecast, train with `--sales-file sales_train_validation.csv`; that uses only `d_1` through `d_1913` as known sales and forecasts `d_1914` through `d_1941` at inference time.

Generate a submission-shaped file:

```powershell
python scripts\predict_deepar_m5.py --checkpoint artifacts\deepar_m5\best.pt --output artifacts\deepar_m5\submission.csv --device auto
```

Prediction defaults to deterministic mean decoding. For uncertainty-aware outputs, use sampled paths and summarize them, for example:

```powershell
python scripts\predict_deepar_m5.py --checkpoint artifacts\deepar_m5\best.pt --output artifacts\deepar_m5\submission_p90.csv --forecast-mode quantile --quantile 0.9 --num-samples 500 --sample-seed 42 --device auto
```

If the checkpoint was trained on a subset, prediction uses the saved
`selected_series_ids` from that checkpoint. Those series receive DeepAR model
forecasts; all other `sample_submission.csv` rows receive a recent-history
fallback so the output file is complete. The fallback uses the same sales file
recorded in the checkpoint, avoiding validation/evaluation leakage across the
two M5 sales files.

Run a hyperparameter sweep with evaluation-holdout metrics:

```powershell
python scripts\run_deepar_m5_experiments.py --subset-sizes 100 --context-lengths 28,56 --hidden-sizes 16,32 --embedding-dims 4,8 --epochs-list 2 --steps-per-epoch-list 20 --batch-sizes 16 --forecast-modes mean,quantile --quantiles 0.5,0.9 --num-samples 200 --output-dir artifacts\deepar_m5_experiments --device cpu
```

This trains on `sales_train_validation.csv`, forecasts `d_1914` through `d_1941`, compares against `sales_train_evaluation.csv`, and writes per-run checkpoints, forecasts, holdout metrics, and a `summary.csv`.

Add W&B tracking to compare runs in the dashboard:

```powershell
python scripts\run_deepar_m5_experiments.py --subset-sizes 100 --context-lengths 28,56 --hidden-sizes 16,32 --embedding-dims 4,8 --epochs-list 2 --steps-per-epoch-list 20 --batch-sizes 16 --forecast-modes mean,quantile --quantiles 0.5,0.9 --num-samples 200 --output-dir artifacts\deepar_m5_experiments --device cpu --wandb --wandb-project m5-competition --wandb-entity ankup25694 --wandb-group deepar-sweep-v1
```

## References

This analysis references foundational papers including:
- *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* (Lim et al., 2020)
- *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks* (Salinas et al., 2020)
- *The M5 competition: Background, organization, and implementation* (Makridakis et al., 2022)

---
*Created as part of an exploration into large-scale retail demand forecasting.*
