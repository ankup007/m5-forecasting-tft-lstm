# DeepAR M5: Execution & Experiment Guide

This guide provides instructions for training, evaluating, and generating predictions using the DeepAR model for the M5 competition. The workflow is **local-first**, with optional **Weights & Biases (W&B)** integration for experiment tracking and hyperparameter sweeps.

---

## 🚀 1. Training (Local)

To train a single model with specific hyperparameters locally without any cloud logging.

### Basic Training
```bash
python scripts/train_deepar_m5.py \
    --subset-size 1000 \
    --hidden-size 64 \
    --num-layers 2 \
    --epochs 10 \
    --artifact-dir artifacts/my_local_run
```

### Key Training Flags:
- `--subset-size`: Number of series to use (out of 30,490). Use a small number (e.g., 100) for smoke tests.
- `--context-length`: Lookback window (default: 56).
- `--prediction-length`: Forecast horizon (default: 28).
- `--eval-holdout`: If passed, runs a full WRMSSE evaluation on the last 28 days after training finishes.
- `--artifact-dir`: Where to save `best.pt`, `latest.pt`, and `metrics.json`.

---

## 📊 2. Local Hyperparameter Grid Search

To test multiple combinations of parameters locally. Each combination will be saved in its own sub-folder.

```bash
python scripts/run_deepar_m5_experiments.py \
    --subset-sizes 500,1000 \
    --hidden-sizes 32,64 \
    --learning-rates 0.001,0.0005 \
    --output-dir artifacts/my_grid_search
```

- **Output:** A `summary.csv` will be created in the `output-dir` containing metrics for all runs, allowing you to identify the best configuration locally.

---

## 📈 3. Evaluation Mode

Use this to evaluate a previously trained model against a specific sales file (e.g., `sales_train_evaluation.csv`). This calculates competition metrics (WRMSSE, RMSE, etc.) for the **last 28 days** of the file.

```bash
python scripts/evaluate_deepar_m5.py \
    --checkpoint artifacts/my_local_run/best.pt \
    --sales-file sales_train_evaluation.csv \
    --output-dir evaluations/test_run
```

- **Output:** Saves `eval_metrics.json` (scores) and `eval_forecasts.csv` (actuals vs. predictions).

---

## 🔮 4. Prediction Mode (Submission)

Generate a final `submission.csv` for the M5 competition (predicting the next 28 days for which actuals are not provided).

```bash
python scripts/predict_deepar_m5.py \
    --checkpoint artifacts/my_local_run/best.pt \
    --output predictions/submission_v1.csv
```

---

## 🌐 5. Weights & Biases (Optional)

W&B is used for live charting and cloud-based experiment management.

### Enabling W&B Logging
Add the `--wandb` flag to any training or experiment command:
```bash
python scripts/train_deepar_m5.py --wandb --wandb-project m5-deepar --wandb-run-name "high-capacity-test"
```

### Running a W&B Sweep (Hyperparameter Tuning)
1. **Initialize the Sweep:**
   ```bash
   wandb sweep sweep.yaml
   ```
   *This will return a `SWEEP_ID`.*

2. **Start an Agent:**
   ```bash
   wandb agent <ENTITY_NAME>/<PROJECT_NAME>/<SWEEP_ID>
   ```
   *The agent will pull configurations from `sweep.yaml` and execute `scripts/train_deepar_m5.py` for each iteration.*

---

## 🛠 6. Full Hyperparameter Reference

Each script has a set of configurable arguments. Below are the full lists for each major script.

### `scripts/train_deepar_m5.py` (Single Training Run)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--subset-size` | `1000` | Number of time series to randomly sample for training. |
| `--context-length` | `56` | Number of historical days the model "sees" before predicting. |
| `--prediction-length` | `28` | Number of days to forecast (horizon). |
| `--hidden-size` | `64` | Dimension of the LSTM hidden state and cell state. |
| `--num-layers` | `1` | Number of LSTM layers stacked on top of each other. |
| `--embedding-dim` | `16` | Dimension of the static categorical embeddings (item_id, store_id, etc). |
| `--batch-size` | `128` | Number of windows per gradient update step. |
| `--epochs` | `10` | Number of full training cycles. |
| `--steps-per-epoch` | `200` | Number of random batches to sample in each epoch. |
| `--learning-rate` | `0.001` | Optimizer learning rate (Adam). |
| `--dropout` | `0.0` | Dropout probability applied between LSTM layers. |
| `--grad-clip` | `10.0` | Maximum allowed norm for gradients to prevent explosion. |
| `--sales-file` | `sales_train_evaluation.csv` | Which M5 sales file to load for training data. |
| `--eval-holdout` | `False` | Flag: If set, run WRMSSE evaluation on the last 28 days after training. |
| `--forecast-mode` | `mean` | Prediction type: `mean`, `sample-mean`, or `quantile`. |
| `--num-samples` | `100` | Number of paths to draw for `sample-mean` or `quantile` modes. |
| `--quantile` | `0.5` | Target quantile if `--forecast-mode` is `quantile`. |
| `--device` | `auto` | Force device (`cpu`, `cuda`) or let script detect. |

### `scripts/run_deepar_m5_experiments.py` (Grid Search)

*Note: For grid search, parameters accept comma-separated lists (e.g., `--hidden-sizes 32,64`).*

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--subset-sizes` | `100` | List of subset sizes to iterate over. |
| `--context-lengths` | `56` | List of lookback windows to test. |
| `--hidden-sizes` | `32` | List of LSTM hidden dimensions to test. |
| `--embedding-dims` | `8` | List of embedding dimensions to test. |
| `--epochs-list` | `2` | List of epoch counts to test. |
| `--steps-per-epoch-list` | `20` | List of steps per epoch to test. |
| `--learning-rates` | `0.001` | List of learning rates to test. |
| `--forecast-modes` | `mean` | List of forecast modes (e.g., `mean,quantile`). |
| `--quantiles` | `0.5` | List of quantiles to test (only used for `quantile` mode). |
| `--train-sales-file` | `sales_train_validation.csv` | Training file used for the search. |

### `scripts/evaluate_deepar_m5.py` (Post-Training Evaluation)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--checkpoint` | `artifacts/deepar_m5/best.pt` | Path to the `.pt` file to evaluate. |
| `--sales-file` | `sales_train_evaluation.csv` | File containing the actuals for comparison. |
| `--forecast-mode` | `mean` | Forecast summary type for metrics. |
| `--num-samples` | `100` | Samples to draw for stochastic metrics. |
| `--batch-size` | `256` | Batch size for inference. |

---

## 🛠 Troubleshooting & Local Artifacts

Even when W&B is not used, the following files are always generated for every run:

| File | Description |
| :--- | :--- |
| `best.pt` | PyTorch checkpoint with the lowest validation loss. |
| `data_config.json` | The exact data parameters (subset, context, etc.) used. |
| `model_config.json` | The model architecture settings. |
| `train_metrics.json` | History of training and validation NLL per epoch. |
| `holdout_metrics.json` | WRMSSE and other error metrics (if evaluation was run). |

**Note:** All scripts automatically detect and use your GPU (`cuda`) if available, otherwise they fall back to `cpu`.
