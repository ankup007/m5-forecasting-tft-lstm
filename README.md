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

For setup, training, prediction, and experiment commands, use [DeepAR M5 Setup Guide](./SETUP_GUIDE.md). That guide is the canonical runbook and is kept in sync with the code.

For a code walkthrough, tuning guide, and extension notes, read [DeepAR Implementation Guide](./IMPLEMENTATION_README.md).

For the training/prediction mechanics, including teacher forcing, autoregressive rollout, Negative Binomial likelihood, and mean versus sampled forecasts, read [Understanding DeepAR Training And Prediction](./DEEPAR_TRAINING_AND_PREDICTION_README.md).

## Rolled DeepAR Experiments For M5

The current DeepAR path now includes an M5-focused training mode inspired by Jeon and Seong's third-place M5 Accuracy solution, which modified DeepAR for intermittent retail demand. The main problem being addressed is exposure bias: standard DeepAR trains with true previous targets as recurrent inputs, but inference must feed back the model's own forecasts. On M5, this mismatch is especially damaging because many item-store series contain long zero runs followed by sudden bursts.

The implementation supports:

- **Scheduled rolled feedback:** during forecast-horizon training steps, the model can feed back sampled predictions instead of always feeding ground truth. The probability is controlled by `rolled_feedback_max_prob`, `rolled_feedback_warmup_epochs`, and `rolled_feedback_ramp_epochs`.
- **Rolling-origin autoregressive validation:** checkpoints can be selected using forecasts generated the same way inference works, across several historical 28-day origins.
- **Spike diagnostics:** validation and holdout metrics include zero-day false positive rate, nonzero-day MAE, spike-day MAE, spike hit rate, and spike bias.
- **NB vs Tweedie comparison:** experiments can compare Negative Binomial and Tweedie objectives under the same autoregressive validation protocol.

The main experiment entry point is:

```powershell
python -m src.deepar_m5.experiments --data-dir m5-forecasting-accuracy --eval-wrmsse
```

Edit `GRID_CONFIG` in [experiments.py](./src/deepar_m5/experiments.py) before running. A practical first sweep is:

- `loss`: `["tweedie", "negative-binomial"]`
- `rolled_feedback_max_prob`: `[0.0, 0.5]`
- `autoreg_val_origins`: `[4]`
- `autoreg_val_stride`: `[28]`
- `checkpoint_metric`: `["autoreg_wrmsse"]`

For full-data runs, `autoreg_val_every=5` keeps rolling validation cost under control. For small subset debugging, set `autoreg_val_every=1`, reduce `epochs`, and reduce `steps_per_epoch`.

## References

This analysis references foundational papers including:
- *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* (Lim et al., 2020)
- *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks* (Salinas et al., 2020)
- *The M5 competition: Background, organization, and implementation* (Makridakis et al., 2022)
- *Robust recurrent network model for intermittent time-series forecasting* (Jeon and Seong, 2022)
- *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks* (Bengio et al., 2015)

---
*Created as part of an exploration into large-scale retail demand forecasting.*
