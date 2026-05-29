# M5 DeepAR Improvement Plan

This document tracks implementation work inspired by the M5 3rd-place neural-network approach. The first batch is intentionally narrow: fix known correctness/config issues, then compare Tweedie loss against the existing Negative Binomial baseline while keeping the rest of the pipeline unchanged.

## Done

- [x] Studied the 3rd-place M5 NN approach and mapped it to this repo.
- [x] Identified the main gaps: Tweedie loss, rolled/sampled training, rolling WRMSSE validation, richer features, and ensembling.
- [x] Decided the first experiment should isolate loss/scheduler changes instead of changing features or validation at the same time.
- [x] Fix history wiring so `prior_history` reaches training, validation, smoke tests, and standalone inference.
- [x] Remove unused `embedding_dim` from active model/training configuration.
- [x] Add `--loss negative-binomial|tweedie`, defaulting to `negative-binomial`.
- [x] Add masked Tweedie loss as a switch, not as a full replacement.
- [x] Separate Negative Binomial and Tweedie output heads so Tweedie no longer emits unused `alpha`.
- [x] Add Tweedie compound Poisson-Gamma sampling for sampled forecasts and quantiles.
- [x] Add `--scheduler none|cosine`, defaulting to cosine annealing.
- [x] **Feature Engineering Expansion**:
    - [x] Implemented scale-aware dynamic zero-counter with 0.5 unit threshold.
    - [x] Implemented rolling 7 and 28-day means calculated on-the-fly.
    - [x] Added advanced price ratios (relative to historical max and daily department mean).
    - [x] Implemented normalized cyclic and linear time features (wday, month, year, week, day).
    - [x] Switched event handling to dedicated categorical embeddings for `event_name` and `event_type`.
- [x] Verified all features and model changes with a successful end-to-end smoke test.

## Current Batch (Experimentation)

- [ ] Run controlled baseline (Negative Binomial) and Tweedie experiments with the same seed/config.
- [ ] Compare validation loss and holdout metrics/WRMSSE where available.
- [ ] Document results in a new architectural comparison or performance log.

## Next Steps (Architecture & Tuning)

- [ ] Implement **rolled/sampled training**: Modify the training loop so that during the horizon, the model occasionally sees its own previous samples/means instead of 100% teacher forcing.
- [ ] Add **rolling validation** over multiple 28-day windows to better approximate competition leaderboard stability.
- [ ] Add seed/checkpoint ensembling and forecast averaging.
- [ ] Tune Tweedie power and dispersion based on initial experiment results.
