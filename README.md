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

## References

This analysis references foundational papers including:
- *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* (Lim et al., 2020)
- *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks* (Salinas et al., 2020)
- *The M5 competition: Background, organization, and implementation* (Makridakis et al., 2022)

---
*Created as part of an exploration into large-scale retail demand forecasting.*
