# TFT vs LSTM/DeepAR for M5 Forecasting: Architecture, Data Fit, and Compute Reality

## Introduction

The M5 Forecasting Accuracy competition is often described as a retail forecasting benchmark, but that description hides much of what makes the problem interesting. This is not just a collection of independent sales curves. It is a large, hierarchical, covariate-rich demand forecasting problem with thousands of sparse item-store time series, known future calendar information, prices, product metadata, store metadata, and a short 28-day forecast horizon.

That combination makes model choice less obvious than it first appears. A Temporal Fusion Transformer (TFT) looks attractive because the dataset has exactly the kind of static and time-varying covariates TFT was designed to use. An LSTM-based approach such as DeepAR looks attractive for a different reason: it is simpler, global, probabilistic, and much more practical when 30,000+ related time series have to be trained on CPU.

The useful question is therefore not simply "Is TFT better than LSTM?" A better question is:

**Given the structure and scale of the M5 data, which architecture is the more sensible direction to move toward?**

The short answer is:

- **TFT is more naturally aligned with the dataset architecture.**
- **DeepAR/LSTM is more practical as the first full-data neural training path, especially on CPU.**
- **If compute is constrained, DeepAR should usually come first.**
- **If interpretability, known future covariates, and horizon-specific behavior are the priority, TFT becomes the stronger second-stage direction.**

## The M5 Dataset Is Not a Simple Time Series Dataset

The M5 Accuracy competition asked participants to forecast Walmart daily unit sales for the next 28 days. The dataset contains 3,049 products sold across 10 stores in 3 US states. At the most granular level, this creates 30,490 item-store time series. Once the hierarchy is expanded, 42,840 series are evaluated across product, store, state, category, department, and aggregate levels. The M5 background paper describes the dataset as hierarchical, grouped, exogenous-variable-rich, and intermittent [Makridakis et al., 2022a](https://econpapers.repec.org/article/eeeintfor/v_3a38_3ay_3a2022_3ai_3a4_3ap_3a1325-1336.htm).

The core files reflect the modeling challenge:

| File | What it contributes |
|---|---|
| `sales_train_validation.csv` / `sales_train_evaluation.csv` | Historical daily unit sales at item-store level |
| `calendar.csv` | Date, weekday, event, month, year, and SNAP information |
| `sell_prices.csv` | Weekly item-store prices |
| `sample_submission.csv` | Required 28-day forecast format |

Several dataset properties matter directly for architecture choice.

| M5 property | Why it matters |
|---|---|
| 30,490 bottom-level series | A global model is needed. Local one-model-per-series training is not attractive. |
| 42,840 hierarchical series | Errors at item-store level can accumulate upward into aggregate bias. |
| 28-day horizon | The model must handle multi-step daily forecasting, not only one-step prediction. |
| Static metadata | Item, department, category, store, and state define persistent demand structure. |
| Known future covariates | Weekday, events, SNAP flags, and price-related features can be known or constructed for the forecast window. |
| Intermittent demand | Many item-store series contain long zero runs. Zero demand is not always the same as no interest. |
| Weekly prices, daily sales | Price changes occur at a different frequency from demand observations. |
| WRMSSE weighting | High-value series matter more than low-volume noise. |

The scale becomes clearer when the data is reshaped into model-ready long format. At the bottom level alone, 30,490 series across 1,941 days produce roughly 59 million item-store-day rows before lag features, rolling features, calendar joins, or price joins are added. If sliding windows are materialized naively, the training data can become much larger than the original dataset.

That scale is one of the reasons architecture cannot be discussed separately from compute.

## The Modeling Problem Hidden Inside M5

M5 combines four different forecasting problems.

First, there is **cross-series learning**. Many item-store histories are sparse, so a model must borrow information across related products and locations. A global neural model is much more natural than thousands of independent local models.

Second, there is **static-conditioned behavior**. A weekend, holiday, price drop, or SNAP flag does not affect every product in the same way. A Foods item in California and a Hobbies item in Wisconsin should not interpret the same calendar signal identically.

Third, there is **known future information**. The next 28 weekdays are known. Events are known. SNAP flags are known. Price information can also be used carefully. A model that only extrapolates target history leaves a major part of the signal unused.

Fourth, there is **intermittent and sparse demand**. Many item-store series have long stretches of zero sales. Those zeros may reflect genuinely low demand, product lifecycle, assortment, temporary unavailability, stockout, or local store behavior. Treating all zeros as the same kind of observation is usually too crude.

These four requirements create a natural tension:

- TFT is better designed for rich feature interaction and known future covariates.
- DeepAR/LSTM is easier to train globally at M5 scale, especially on CPU.

## LSTM and DeepAR: The Practical Neural Baseline

DeepAR is the most relevant LSTM-based approach for this discussion. A plain LSTM trained separately for each series would be a poor fit for M5 because many series are sparse and noisy. DeepAR instead uses one global recurrent model trained across many related time series. This lets low-volume item-store pairs borrow statistical strength from similar products, stores, departments, categories, and states.

DeepAR is usually built from the following pieces:

| Component | Role in M5 |
|---|---|
| Static categorical embeddings | Encode item, department, category, store, and state |
| Time-varying covariates | Add calendar, event, SNAP, price, and time-index features |
| Lagged target values | Expose recent demand and seasonal structure |
| LSTM or GRU backbone | Compress recent history into a recurrent hidden state |
| Distribution head | Predict demand uncertainty, often with count-friendly distributions |
| Autoregressive decoder | Generate the 28-day forecast path step by step |

The DeepAR paper frames the model as probabilistic forecasting with autoregressive recurrent networks trained across large collections of related series [Salinas et al., 2020](https://econpapers.repec.org/article/eeeintfor/v_3a36_3ay_3a2020_3ai_3a3_3ap_3a1181-1191.htm), [arXiv version](https://arxiv.org/abs/1704.04110). This global design is the main reason it remains relevant for M5.

### Why DeepAR Works Reasonably Well for M5

DeepAR matches the scale of M5 better than a local LSTM. A single global model can be trained over all 30,490 bottom-level series, using static embeddings to distinguish item-store behavior. This is exactly the kind of setting where global recurrent forecasting tends to be useful.

It also has a good probabilistic story for retail sales. M5 targets are non-negative, noisy, and often overdispersed. A negative binomial or similar distributional output is more natural than a plain squared-error head. Even if the final competition forecast is a point forecast, distributional training can make the model more robust to sparse and volatile demand.

DeepAR is also relatively compute-friendly. The dominant recurrent cost grows with batch size, sequence length, and hidden size. If the hidden size is kept moderate and windows are sampled rather than fully materialized, the model can be trained on CPU in a way that is at least operationally plausible.

That practicality should not be underestimated. For a dataset with tens of millions of long-format rows, a model that can be trained, debugged, and iterated is often more valuable than a richer model that cannot be run at meaningful scale.

### Where DeepAR Falls Short

DeepAR's weakness is that the model does not deeply structure the relationship between static metadata, observed history, and known future covariates. Calendar, SNAP, event, and price features can be passed into the model, but their relevance is learned implicitly through embeddings and recurrent states.

That can be limiting for M5. Price sensitivity is product-specific. SNAP effects are state- and category-specific. Event effects vary by department. A plain recurrent architecture has no special mechanism for saying, "This feature matters for this product context, but not for that one."

The second limitation is autoregressive decoding. DeepAR generates future values step by step. A day-28 forecast depends on the generated path through days 1 to 27. This is natural for probabilistic trajectories, but it can also propagate early mistakes. In sparse retail series, early near-zero predictions can make the rest of the horizon too conservative.

The third limitation is long-range memory. LSTMs are good at local temporal dynamics, but annual patterns or distant event analogs should usually be exposed through explicit lag features rather than relying on the recurrent state to remember hundreds of daily steps.

### Practical DeepAR Shape for M5

A sensible CPU-first DeepAR setup would be deliberately compact:

| Design choice | Practical direction |
|---|---|
| Model scope | One global model over all bottom-level item-store series |
| Forecast horizon | 28 days |
| Context length | 56 to 84 days initially |
| Recurrent layers | 1 or 2 |
| Hidden size | Roughly 40 to 80 on CPU |
| Static categoricals | item, department, category, store, state |
| Known future inputs | weekday, month, event flags, SNAP flags, price features |
| Lags | 1, 7, 14, 28, 56, and possibly 364 as an explicit feature |
| Output distribution | Negative binomial is a strong first choice |
| Windowing | Sample windows instead of enumerating all windows |

The key idea is to keep the recurrent model small, use strong feature construction, and rely on global learning. DeepAR is not the most expressive architecture for M5, but it is a practical first neural direction.

## Temporal Fusion Transformer: The Architecturally Natural Fit

Temporal Fusion Transformer was designed for interpretable multi-horizon forecasting when static covariates, known future inputs, and observed historical inputs are all present [Lim et al., 2020](https://arxiv.org/abs/1912.09363). That description maps unusually well to M5.

TFT separates inputs into groups:

| TFT input group | M5 examples |
|---|---|
| Static covariates | item, department, category, store, state |
| Known future inputs | weekday, month, event, SNAP flag, future price-related features |
| Observed historical inputs | sales, lagged sales, rolling demand statistics |

This distinction is important. In M5, not every feature has the same temporal meaning. Calendar variables are known in the future. Sales are only observed in the past. Product and store metadata are static. TFT is built around that taxonomy, while a standard LSTM usually receives these signals in a less structured way.

TFT also combines several mechanisms:

| Component | Why it matters for M5 |
|---|---|
| Static covariate encoders | Product/store identity can condition the entire forecast |
| Variable selection networks | The model can learn which covariates matter in each context |
| Gated residual networks | Unnecessary complexity can be suppressed through gating |
| LSTM sequence block | Recent local demand dynamics can still be modeled recurrently |
| Attention | Relevant historical positions can be emphasized |
| Direct multi-horizon output | The 28-day horizon can be predicted directly |

This is why TFT feels so relevant to M5. It is not simply a Transformer added to a time series. It is a hybrid architecture that keeps recurrent local processing while adding feature selection, static conditioning, known-future conditioning, gating, and attention.

### Why TFT Fits M5 So Well

The strongest argument for TFT is known future covariates. Each of the 28 future days has a known weekday, calendar position, event state, SNAP flag, and possibly price context. TFT can condition each forecast step directly on these future values. This is cleaner than forcing the model to rely mainly on a recurrent state and a recursive path.

The second argument is static-conditioned feature relevance. M5 demand drivers are heterogeneous. Price features may be very important for one department and less useful for another. SNAP flags may matter more for Foods than for Hobbies. Event effects may be localized by state or product category. TFT has machinery for learning these relationships more explicitly than DeepAR.

The third argument is interpretability. Variable selection weights and attention patterns can be inspected. These should not be treated as causal proof, but they are useful diagnostics. They can help answer whether the model is using price, SNAP, event, and lag features in plausible ways.

The fourth argument is direct multi-horizon modeling. TFT predicts the forecast horizon as a sequence of outputs rather than relying entirely on recursive generation. For a fixed 28-day horizon, this is attractive because each horizon position can be conditioned on its own known future inputs.

### Where TFT Becomes Expensive

TFT's problem is not conceptual fit. Its problem is operational cost.

Compared with DeepAR, TFT has more moving parts: embeddings, variable selection networks, gated residual networks, recurrent layers, attention, static enrichment, and multi-horizon output heads. Each part adds computation and activation memory.

The cost increases when:

| Driver | Effect |
|---|---|
| Encoder length is increased | More recurrent and attention work |
| More covariates are added | More feature projection and variable-selection work |
| Hidden size is increased | Dense, recurrent, and attention blocks become heavier |
| Attention heads are increased | Attention memory and computation increase |
| More quantiles are predicted | Output and loss computation grow |
| More windows are sampled | Training time grows directly |

This is the practical catch: the architecture that best matches the data can also be the architecture that is hardest to train over the full data on CPU.

For M5, a full TFT with a long encoder, many features, item-level embeddings, multiple attention heads, and all possible windows would be a poor first CPU experiment. It may be theoretically appealing, but it is unlikely to be the most efficient way to make progress.

### Practical TFT Shape for M5

If TFT is used, it should start small:

| Design choice | Practical direction |
|---|---|
| Model scope | Start with a representative subset, then expand |
| Forecast horizon | 28 days |
| Encoder length | 56 to 84 days on CPU |
| Hidden size | Roughly 8 to 24 on CPU |
| Attention heads | 1 or 2 |
| LSTM layers | 1 |
| Static categoricals | category, department, store, state first; item id can be added later |
| Known future inputs | weekday, month, event, SNAP, price features |
| Observed inputs | sales, selected lags, selected rolling statistics |
| Quantiles | Start with median only or a small set such as 0.1, 0.5, 0.9 |

The main principle is restraint. TFT should be introduced after the feature pipeline is stable, not before. Otherwise the model complexity can hide basic data issues.

## Architectural Comparison

The difference between DeepAR and TFT can be summarized as follows:

| Question | DeepAR/LSTM | TFT |
|---|---|---|
| Is it global across many related series? | Yes | Yes |
| Is it practical on CPU for all 30K+ series? | More practical | Much harder |
| Does it handle static metadata? | Yes, through embeddings | Yes, through static encoders and conditioning |
| Does it use known future covariates naturally? | Yes, but less structurally | Yes, by design |
| Does it handle sparse count-like demand naturally? | Stronger probabilistic framing | Possible, but more dependent on loss and sampling |
| Does it avoid recursive error propagation? | No, it is autoregressive | More so, because it is direct multi-horizon |
| Is it interpretable? | Limited | Better feature and attention diagnostics |
| Is it easy to tune? | Easier | Harder |
| Is it the richer architecture for M5? | No | Yes |
| Is it the better first CPU direction? | Yes | Usually no |

This is the central tradeoff. DeepAR is more practical. TFT is more expressive.

## Training and Compute Perspective

Both models require the same disciplined data foundation:

- a long item-store-day table;
- static categorical encodings;
- joined calendar variables;
- joined price variables;
- leakage-safe lag features;
- leakage-safe rolling features;
- target scaling or normalization;
- sampled training windows.

The difference is how much the architecture benefits from that structure.

DeepAR can work with a relatively lean feature set because the model is compact and recurrent. It benefits from explicit lags, known future calendar features, static embeddings, and count-friendly likelihoods.

TFT benefits from a cleaner feature taxonomy. Static covariates, known future inputs, and observed historical inputs should be explicitly separated. This separation is one of its main advantages, but it also makes data preparation more demanding.

From a CPU perspective:

| Dimension | DeepAR/LSTM | TFT |
|---|---|---|
| Training speed | Faster | Slower |
| Memory footprint | Lower | Higher |
| Sensitivity to feature count | Moderate | High |
| Sensitivity to encoder length | Manageable | Higher |
| First full-data attempt | Reasonable | Risky |
| Best use case | Practical global neural baseline | Rich covariate-aware architecture |

Several choices should be avoided regardless of architecture:

- training a separate LSTM per item-store series;
- starting with a large TFT on CPU;
- using a 365-day encoder before a smaller model works;
- materializing every possible sliding window;
- throwing all engineered features into TFT at once;
- ignoring known future calendar and price signals;
- treating all zero sales as equivalent;
- increasing neural depth before fixing the data pipeline.

## Which Direction Makes More Sense?

If the question is architectural elegance, TFT is the more natural model for M5. It matches the dataset's structure: static metadata, known future features, observed history, multi-horizon forecasting, and heterogeneous feature relevance.

If the question is practical progress under CPU constraints, DeepAR/LSTM is the more sensible first direction. It is global, compact, probabilistic, and feasible to train across all bottom-level series with sampled windows.

The most defensible path is therefore staged:

| Stage | Direction | Reason |
|---|---|---|
| First neural model | DeepAR/LSTM | Establish a tractable global neural baseline across all 30K+ series |
| Second-stage richer model | TFT | Exploit known future covariates, static conditioning, and interpretability |
| Competition-style benchmark | LightGBM | Historical M5 evidence strongly supports feature-engineered tree models |

So the final recommendation is:

**Move first toward DeepAR/LSTM if CPU is the main constraint and the goal is a full-data neural model. Move toward TFT when the objective shifts toward richer covariate handling, interpretability, or GPU-backed experimentation.**

This conclusion is not saying that DeepAR is more sophisticated. It is saying that DeepAR is the better first operating point. TFT remains the more architecturally relevant model for the full complexity of M5, but its advantages are only useful if the training setup can support them.

## References

- Kaggle. [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy).
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022a). [The M5 competition: Background, organization, and implementation](https://econpapers.repec.org/article/eeeintfor/v_3a38_3ay_3a2022_3ai_3a4_3ap_3a1325-1336.htm). International Journal of Forecasting, 38(4), 1325-1336.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022b). [M5 accuracy competition: Results, findings, and conclusions](https://econpapers.repec.org/article/eeeintfor/v_3a38_3ay_3a2022_3ai_3a4_3ap_3a1346-1364.htm). International Journal of Forecasting, 38(4), 1346-1364.
- Mcompetitions. [M5-methods: Data, Benchmarks, and methods submitted to the M5 forecasting competition](https://github.com/Mcompetitions/M5-methods).
- In, Y., and Jung, J. (2022). [Simple averaging of direct and recursive forecasts via partial pooling using machine learning](https://ideas.repec.org/a/eee/intfor/v38y2022i4p1386-1399.html). International Journal of Forecasting, 38(4), 1386-1399.
- Lim, B., Arik, S. O., Loeff, N., and Pfister, T. (2020). [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363). arXiv:1912.09363.
- Google Research. [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/).
- Salinas, D., Flunkert, V., Gasthaus, J., and Januschowski, T. (2020). [DeepAR: Probabilistic forecasting with autoregressive recurrent networks](https://econpapers.repec.org/article/eeeintfor/v_3a36_3ay_3a2020_3ai_3a3_3ap_3a1181-1191.htm). International Journal of Forecasting, 36(3), 1181-1191.
- Salinas, D., Flunkert, V., Gasthaus, J., and Januschowski, T. [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110). arXiv:1704.04110.
- PyTorch Forecasting. [Demand forecasting with the Temporal Fusion Transformer](https://pytorch-forecasting.readthedocs.io/en/v1.4.0/tutorials/stallion.html).
- PyTorch Forecasting. [Autoregressive modelling with DeepAR and DeepVAR](https://pytorch-forecasting.readthedocs.io/en/v1.4.0/tutorials/deepar.html).
- GluonTS. [gluonts.torch.model.deepar package](https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.deepar.html).
