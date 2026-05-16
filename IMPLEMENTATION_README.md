# DeepAR Implementation Guide

This guide explains how the from-scratch DeepAR pipeline is organized, where to start reading, what to tune first, and where to extend the implementation.

## Where To Start

Start with the thin scripts in `scripts/`:

| File | Purpose |
|---|---|
| `scripts/smoke_deepar_m5.py` | Fast correctness check on a tiny subset. Run this before changing model or data code. |
| `scripts/train_deepar_m5.py` | Main training entrypoint. CLI flags control subset size, model size, context length, and training budget. |
| `scripts/predict_deepar_m5.py` | Loads a checkpoint and writes a Kaggle-shaped submission CSV. |

The real implementation lives in `src/deepar_m5/`:

| File | What it owns |
|---|---|
| `data.py` | M5 CSV loading, static encoders, covariate construction, scaling, sampled windows. |
| `model.py` | Custom LSTM cell, DeepAR network, autoregressive prediction, negative-binomial loss. |
| `train.py` | Training loop, validation loop, checkpoint/config/artifact writing. |
| `infer.py` | Checkpoint loading, future decoding, fallback forecasts, submission formatting. |
| `smoke.py` | End-to-end shape/loss/backprop/inference sanity test. |

The shortest path to understanding the code is:

1. Read `scripts/train_deepar_m5.py` to see how execution enters the package.
2. Read `train.py::main` for the full training flow.
3. Read `data.py::load_m5_bundle` and `data.py::WindowSampler` for how batches are built.
4. Read `model.py::DeepAR.forward` and `model.py::DeepAR.predict_mean` for training versus inference behavior.

## Execution Flow

Training follows this path:

1. `train.py` parses CLI flags and builds a `DataConfig`.
2. `load_m5_bundle` loads M5 sales, calendar, and price files.
3. Static category encoders are fit on the full M5 metadata so ids are stable.
4. Calendar, SNAP, and price covariates are built into a `(series, day, feature)` array.
5. `WindowSampler` samples rolling windows without materializing every possible window.
6. `DeepAR.forward` uses teacher forcing: each step receives the actual previous target.
7. `negative_binomial_nll` computes the masked loss only over the prediction part of each window.
8. Checkpoints and metadata are written to `artifacts/deepar_m5/`.

Inference follows this path:

1. `infer.py` loads `best.pt` or another checkpoint.
2. The same selected series and encoders are restored from the checkpoint.
3. `WindowSampler.make_inference_batch` builds context-plus-horizon inputs.
4. `DeepAR.predict_mean` decodes future steps autoregressively.
5. The output is written in `sample_submission.csv` format.
6. Series not present in a pilot checkpoint receive a simple recent-history fallback so the CSV is complete.

## What Is Implemented From Scratch

The implementation intentionally avoids forecasting libraries and built-in recurrent layers.

Implemented directly:

- Static categorical embeddings for M5 item/store metadata.
- A custom LSTM cell using raw PyTorch matrix operations.
- Global DeepAR-style recurrent model shared across all selected series.
- Negative-binomial distribution parameters and log likelihood.
- Rolling-window sampler for train and validation.
- Teacher-forced training and autoregressive inference.
- Checkpoint, config, encoder, metrics, and submission artifact handling.

PyTorch is still used for tensors, autograd, optimizers, modules, and device placement.

## Important Tuning Knobs

Start with data and training-budget knobs before increasing model size.

| Flag | Default | Tune when |
|---|---:|---|
| `--subset-size` | `1000` | Increase after smoke and pilot runs are stable. Use `0` or a value above dataset size for full data. |
| `--context-length` | `56` | Increase to `84` or `112` if the model needs more history. This raises memory and runtime. |
| `--prediction-length` | `28` | Keep at `28` for M5 submission training. Smaller values are useful for smoke tests. |
| `--batch-size` | `128` | Lower on memory pressure; raise for faster stable training if memory allows. |
| `--steps-per-epoch` | `200` | Controls how many sampled windows the model sees per epoch. Increase before over-tuning architecture. |
| `--epochs` | `10` | Increase once validation loss keeps improving. |
| `--hidden-size` | `64` | Main model capacity knob. Try `32`, `64`, `96`, `128`. |
| `--embedding-dim` | `16` | Capacity for static categories. Try `8` or `16` first. |
| `--num-layers` | `1` | Add a second layer only after the single-layer model is clearly underfitting. |
| `--learning-rate` | `1e-3` | Lower to `3e-4` if training is unstable. |

Recommended scaling path:

1. Smoke test with 12 series.
2. Pilot with 500-1,000 series.
3. Larger pilot with 3,000-5,000 series.
4. Full bottom-level M5 data only after runtime and validation behavior are understood.

## Data Pipeline Notes

`data.py` avoids building a giant long-format table. Sales remain in a wide NumPy array, while covariates are built as a compact series-day-feature cube.

Current covariates:

- Weekday sine/cosine.
- Month sine/cosine.
- Event presence flags.
- State-specific SNAP flag.
- Log sell price.
- Price missing flag.

Current static categories:

- `item_id`
- `dept_id`
- `cat_id`
- `store_id`
- `state_id`

Series scaling is computed from historical demand before the validation horizon. The model trains on scaled previous targets but predicts original-scale negative-binomial means.

## Extending The Data

To add a known future covariate:

1. Add it in `_build_covariate_cube` or `_common_calendar_covariates`.
2. Append a clear name to `covariate_columns`.
3. Run `scripts/smoke_deepar_m5.py` to verify shapes.
4. Train a tiny checkpoint and run prediction once.

Good next covariates:

- Explicit day index trend features.
- Lagged sales features such as 7, 28, and 364 days.
- Rolling mean or zero-rate features.
- Price change and relative price features.
- Event type categorical embeddings instead of only event-present flags.

Be careful with leakage. A feature used in the prediction horizon must be knowable at forecast time. Future sales-derived rolling features should not be computed from future target values.

## Extending The Model

Model code is centered in `model.py`.

Useful extensions:

- Add dropout between stacked custom LSTM layers.
- Add a learned projection for covariates before concatenation.
- Add lagged target inputs as extra per-step features.
- Add sampling-based forecast paths instead of mean-only decoding.
- Add quantile summaries from sampled negative-binomial trajectories.

If you change the model constructor or config, update `ModelConfig` and make sure checkpoints still save enough metadata for `infer.py` to reconstruct the model.

## Artifacts

Training writes to `artifacts/deepar_m5/` by default:

| Artifact | Purpose |
|---|---|
| `best.pt` | Best checkpoint by validation loss. |
| `latest.pt` | Most recent checkpoint. |
| `metrics.json` | Per-epoch train and validation losses. |
| `data_config.json` | Data/window settings used for training. |
| `model_config.json` | Model shape settings. |
| `encoders.json` | Static categorical id mappings. |
| `selected_series.csv` | Series used in the run. |
| `submission.csv` | Generated forecast output when running inference. |

`artifacts/` is ignored by git.

## Common Commands

Create the environment:

```powershell
conda env create -f environment.yml
conda activate m5-deepar-scratch
```

Run the smoke test:

```powershell
python scripts\smoke_deepar_m5.py --subset-size 12 --context-length 28 --prediction-length 7 --device cpu
```

Run a small pilot:

```powershell
python scripts\train_deepar_m5.py --subset-size 1000 --context-length 56 --prediction-length 28 --batch-size 128 --epochs 10 --steps-per-epoch 200 --device auto
```

Generate predictions:

```powershell
python scripts\predict_deepar_m5.py --checkpoint artifacts\deepar_m5\best.pt --output artifacts\deepar_m5\submission.csv --device auto
```

Run a faster debugging train:

```powershell
python scripts\train_deepar_m5.py --subset-size 50 --context-length 28 --prediction-length 7 --batch-size 8 --epochs 1 --steps-per-epoch 5 --hidden-size 16 --embedding-dim 4 --artifact-dir artifacts\deepar_m5_debug --device cpu
```

## Practical Reading Checklist

Before modifying the model:

- Run the smoke test.
- Confirm the expected tensor shapes in `WindowSampler`.
- Check that the new feature is available for both training and prediction horizons.
- Keep the first run tiny and CPU-based.
- Compare `metrics.json` across runs before scaling up.

