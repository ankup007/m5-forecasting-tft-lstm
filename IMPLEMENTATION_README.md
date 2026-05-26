# DeepAR Implementation Guide

This guide explains how the from-scratch DeepAR pipeline is organized, where to start reading, what to tune first, and where to extend the implementation.

For a deeper explanation of the training objective, teacher forcing, autoregressive prediction, Negative Binomial likelihood, and mean versus sampled forecasting, read [Understanding DeepAR Training And Prediction](./DEEPAR_TRAINING_AND_PREDICTION_README.md).

## Where To Start

Start with the thin scripts in `scripts/`:

| File | Purpose |
|---|---|
| `scripts/smoke_deepar_m5.py` | Fast correctness check on a tiny subset. Run this before changing model or data code. |
| `scripts/train_deepar_m5.py` | Main training entrypoint. CLI flags control subset size, model size, context length, and training budget. |
| `scripts/predict_deepar_m5.py` | Loads a checkpoint and writes a Kaggle-shaped submission CSV. |
| `scripts/run_deepar_m5_experiments.py` | Runs hyperparameter sweeps, trains on validation data, forecasts the evaluation holdout, and writes per-run metrics/artifacts. |

The real implementation lives in `src/deepar_m5/`:

| File | What it owns |
|---|---|
| `data.py` | M5 CSV loading, static encoders, covariate construction, scaling, sampled windows. |
| `model.py` | Custom LSTM cell, DeepAR network, autoregressive prediction, Negative Binomial and Tweedie losses. |
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
7. The configured forecast loss computes the masked loss only over the prediction part of each window.
8. Checkpoints and metadata are written to `artifacts/deepar_m5/`.

Inference follows this path:

1. `infer.py` loads `best.pt` or another checkpoint.
2. The same selected series and encoders are restored from the checkpoint.
3. `WindowSampler.make_inference_batch` builds context-plus-horizon inputs.
4. `DeepAR.predict_mean` decodes future steps autoregressively.
5. The output is written in `sample_submission.csv` format.
6. Series not present in a pilot checkpoint receive a simple recent-history fallback so the CSV is complete.

If training used only a subset, inference does not retrain or expand the neural
model to every M5 series. It reads `selected_series_ids` from the checkpoint and
uses the model only for those item-store ids. Every other row in
`sample_submission.csv` is filled with a fallback forecast based on the most
recent observed `prediction_length` days from the same sales file used during
training.

## What Is Implemented From Scratch

The implementation intentionally avoids forecasting libraries and built-in recurrent layers.

Implemented directly:

- Static categorical embeddings for M5 item/store metadata.
- A custom LSTM cell using raw PyTorch matrix operations.
- Global DeepAR-style recurrent model shared across all selected series.
- Distribution-specific forecast heads for Negative Binomial and Tweedie.
- Negative Binomial NLL, Tweedie deviance loss, and distribution-specific sampling.
- Rolling-window sampler for train and validation.
- Teacher-forced training and autoregressive inference.
- Checkpoint, config, encoder, metrics, and submission artifact handling.

PyTorch is still used for tensors, autograd, optimizers, modules, and device placement.

## Important Tuning Knobs

Start with data and training-budget knobs before increasing model size.

| Flag | Default | Tune when |
|---|---:|---|
| `--sales-file` | `sales_train_evaluation.csv` | Use `sales_train_validation.csv` when you want a blind forecast for `d_1914` through `d_1941`. Use the default when training up to `d_1941` and forecasting `d_1942` through `d_1969`. |
| `--subset-size` | `1000` | Increase after smoke and pilot runs are stable. Use `0` or a value above dataset size for full data. |
| `--context-length` | `56` | Increase to `84` or `112` if the model needs more history. This raises memory and runtime. |
| `--prediction-length` | `28` | Keep at `28` for M5 submission training. Smaller values are useful for smoke tests. |
| `--batch-size` | `128` | Lower on memory pressure; raise for faster stable training if memory allows. |
| `--steps-per-epoch` | `200` | Controls how many sampled windows the model sees per epoch. Increase before over-tuning architecture. |
| `--epochs` | `10` | Increase once validation loss keeps improving. |
| `--hidden-size` | `64` | Main model capacity knob. Try `32`, `64`, `96`, `128`. |
| `--num-layers` | `1` | Add a second layer only after the single-layer model is clearly underfitting. |
| `--learning-rate` | `1e-3` | Lower to `3e-4` if training is unstable. |
| `--loss` | `negative-binomial` | Use `tweedie` to test the intermittent-demand loss from the M5 NN approach while keeping the rest of the pipeline fixed. |
| `--scheduler` | `cosine` | Use `none` for exact no-scheduler comparisons. |
| `--tweedie-power` | `1.5` | Tune only for Tweedie runs; must stay between `1` and `2`. |
| `--tweedie-dispersion` | `1.0` | Fixed Tweedie dispersion used for loss scaling and compound Poisson-Gamma sampling variance. |

Recommended scaling path:

1. Smoke test with 12 series.
2. Pilot with 500-1,000 series.
3. Larger pilot with 3,000-5,000 series.
4. Full bottom-level M5 data only after runtime and validation behavior are understood.

## Data Pipeline Notes

`data.py` avoids building a giant long-format table. Sales remain in a wide NumPy array, while covariates are built as a compact series-day-feature cube.

### M5 Validation And Evaluation Files

M5 provides two sales files with the same 30,490 bottom-level item-store series:

| File | Known target days | Typical use |
|---|---:|---|
| `sales_train_validation.csv` | `d_1` through `d_1913` | Train as if you are in the original validation stage, then forecast `d_1914` through `d_1941`. |
| `sales_train_evaluation.csv` | `d_1` through `d_1941` | Train after validation actuals are known, then forecast the final evaluation horizon `d_1942` through `d_1969`. |

The item/store pairs are the same. The row ids differ only by suffix:

```text
HOBBIES_1_001_CA_1_validation
HOBBIES_1_001_CA_1_evaluation
```

The default CLI uses `sales_train_evaluation.csv`. With `--prediction-length 28`, the built-in validation split holds out the last 28 known days, so training windows stop at `d_1913` and validation scores `d_1914` through `d_1941`. That is useful for offline validation, but it means those 28 days are visible to the training run for validation/checkpoint selection.

If the goal is a blind forecast for `d_1914` through `d_1941`, use:

```powershell
python scripts\train_deepar_m5.py --sales-file sales_train_validation.csv --prediction-length 28 --context-length 56 --device auto
```

With that setting, the script's internal validation holdout becomes `d_1886` through `d_1913`, and inference from the checkpoint forecasts `d_1914` through `d_1941`.

Inference uses the checkpoint's saved `sales_file` setting:

| Checkpoint trained with | Inference context ends at | Model forecast horizon | Fallback repeats |
|---|---:|---:|---:|
| `sales_train_validation.csv` | `d_1913` | `d_1914` through `d_1941` | recent days before `d_1914`, normally `d_1886` through `d_1913` for a 28-day horizon |
| `sales_train_evaluation.csv` | `d_1941` | `d_1942` through `d_1969` | recent days before `d_1942`, normally `d_1914` through `d_1941` for a 28-day horizon |

For subset experiments, `predict_deepar_m5.py` still writes all rows from
`sample_submission.csv`. Rows belonging to checkpoint-selected series receive
DeepAR forecasts. Rows outside the subset receive fallback forecasts, and the
logs report how many rows were filled by each path.

Calendar and price data extend farther than the observed sales target:

```text
calendar.csv: d_1 through d_1969
sell_prices.csv: weekly prices keyed by wm_yr_wk
```

That is intentional. Future calendar, SNAP, and planned price information are known covariates, so they can be used for forecast horizons where future sales are unknown. The model must not use future sales-derived features, but it can use future known covariates.

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

## Running The Code

The setup and command examples live in [DeepAR M5 Setup Guide](./SETUP_GUIDE.md). Keep that guide as the single source of truth for environment setup, smoke tests, training, inference, experiment sweeps, and W&B toggles.

## Practical Reading Checklist

Before modifying the model:

- Run the smoke test.
- Confirm the expected tensor shapes in `WindowSampler`.
- Check that the new feature is available for both training and prediction horizons.
- Keep the first run tiny and CPU-based.
- Compare `metrics.json` across runs before scaling up.
