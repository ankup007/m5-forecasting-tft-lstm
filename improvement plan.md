# M5 DeepAR Improvement Plan

This document tracks implementation work inspired by the M5 3rd-place neural-network approach. The first batch is intentionally narrow: fix known correctness/config issues, then compare Tweedie loss against the existing Negative Binomial baseline while keeping the rest of the pipeline unchanged.

## Done

- [x] Studied the 3rd-place M5 NN approach and mapped it to this repo.
- [x] Identified the main gaps: Tweedie loss, rolled/sampled training, rolling WRMSSE validation, richer features, and ensembling.
- [x] Decided the first experiment should isolate loss/scheduler changes instead of changing features or validation at the same time.

## Current Batch

- [x] Fix history wiring so `prior_history` reaches training, validation, smoke tests, and standalone inference.
- [x] Remove unused `embedding_dim` from active model/training configuration.
- [x] Add `--loss negative-binomial|tweedie`, defaulting to `negative-binomial`.
- [x] Add masked Tweedie loss as a switch, not as a full replacement.
- [x] Separate Negative Binomial and Tweedie output heads so Tweedie no longer emits unused `alpha`.
- [x] Add Tweedie compound Poisson-Gamma sampling for sampled forecasts and quantiles.
- [x] Add `--scheduler none|cosine`, defaulting to cosine annealing.
- [ ] Run controlled baseline and Tweedie experiments with the same seed/config.
- [ ] Compare validation loss and holdout metrics/WRMSSE where available.

## Experiment Commands

Baseline with the existing loss:

```powershell
python scripts\train_deepar_m5.py --loss negative-binomial --scheduler cosine --subset-size 34590 --context-length 84 --batch-size 1024 --epochs 20 --steps-per-epoch 100 --hidden-size 64 --num-layers 2 --dropout 0.1 --learning-rate 0.001 --seed 42 --eval-holdout --eval-wrmsse
```

Tweedie test with the same setup:

```powershell
python scripts\train_deepar_m5.py --loss tweedie --tweedie-power 1.5 --tweedie-dispersion 1.0 --scheduler cosine --subset-size 34590 --context-length 84 --batch-size 1024 --epochs 20 --steps-per-epoch 100 --hidden-size 64 --num-layers 2 --dropout 0.1 --learning-rate 0.001 --seed 42 --eval-holdout --eval-wrmsse
```

For quick validation before full runs:

```powershell
python scripts\train_deepar_m5.py --loss tweedie --scheduler cosine --subset-size 64 --context-length 28 --prediction-length 7 --batch-size 8 --epochs 1 --steps-per-epoch 2 --hidden-size 16 --device cpu --log-level INFO
```

## Later

- [ ] Implement rolled/sampled training so horizon inputs can use sampled model outputs instead of pure teacher forcing.
- [ ] Add rolling validation over multiple 28-day windows and select by average WRMSSE instead of single-window likelihood.
- [ ] Expand feature engineering with zero-sales/staleness features, richer lag/rolling statistics, price-relative features, and event identity/distance features.
- [ ] Add seed/checkpoint ensembling and forecast averaging.
- [ ] Tune Tweedie power and dispersion if initial Tweedie results are competitive.
