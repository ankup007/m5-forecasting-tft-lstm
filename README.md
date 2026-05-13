# M5 Forecasting: TFT vs LSTM/DeepAR Comparison

This repository contains a detailed architectural analysis of the **Temporal Fusion Transformer (TFT)** and **DeepAR/LSTM** models within the context of the [Kaggle M5 Forecasting Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) competition.

## Project Overview

The M5 competition presents a unique challenge: forecasting 42,840 hierarchical time series across Walmart's retail hierarchy over a 28-day horizon. This repository explores which neural architectures are best suited for this scale and complexity.

### Key Analysis Highlights:
- **TFT's Natural Alignment:** Why TFT's design for static, known future, and observed past covariates makes it conceptually superior for retail forecasting.
- **The Compute Reality:** Why DeepAR/LSTM remains the more practical choice for initial CPU-based experimentation at scale.
- **Practical Recommendations:** A staged approach to model development, starting with a robust baseline and moving toward more complex, interpretable architectures.

## Contents

- `m5_tft_lstm_architectural_comparison.md`: The core analysis document covering:
    - M5 Dataset properties and modeling challenges.
    - DeepAR/LSTM architecture, strengths, and limitations.
    - Temporal Fusion Transformer (TFT) architecture and suitability.
    - Detailed side-by-side architectural comparison.
    - Practical training and compute perspectives.

## Summary Findings

| Model | Best For | Main Tradeoff |
|---|---|---|
| **DeepAR/LSTM** | Practical, global neural baseline | Less structural handling of known future covariates |
| **TFT** | Interpretable, covariate-rich modeling | Significant compute and operational overhead |

## References

This analysis references foundational papers including:
- *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* (Lim et al., 2020)
- *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks* (Salinas et al., 2020)
- *The M5 competition: Background, organization, and implementation* (Makridakis et al., 2022)

---
*Created as part of an exploration into large-scale retail demand forecasting.*
