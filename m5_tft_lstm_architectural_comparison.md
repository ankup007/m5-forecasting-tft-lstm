# TFT vs DeepAR/LSTM for M5 Forecasting: Dataset Complexity, Architecture, and Compute Tradeoffs

## Framing the Choice

The M5 Forecasting Accuracy competition is often described as a retail forecasting benchmark, but that description hides much of what makes the problem interesting. This is not just a collection of independent sales curves. It is a large, hierarchical, covariate-rich demand forecasting problem with thousands of sparse item-store time series, known future calendar information, prices, product metadata, store metadata, and a short 28-day forecast horizon.

That combination makes model choice less obvious than it first appears. A **Temporal Fusion Transformer (TFT)** looks attractive because the dataset has exactly the kind of **static and time-varying covariates** TFT was designed to use. An LSTM-based approach such as DeepAR looks attractive for a different reason: it is simpler, global, probabilistic, and easier to scale across 30,000+ related time series.

The useful question is therefore not simply "Is TFT better than LSTM?" A better question is:

**Given the structure and scale of the M5 data, which architecture is the more sensible direction to move toward?**

The short answer:

- **TFT is more naturally aligned with the dataset architecture.**
- **DeepAR/LSTM is the more practical first full-data neural baseline.**
- **The deciding constraint is not only CPU vs GPU; it is the full compute budget: memory, window sampling, sequence length, feature count, training time, and iteration speed.**
- **TFT becomes more attractive once that budget can support richer covariate handling, interpretability, and direct multi-horizon modeling.**

## M5 Dataset Complexity

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

## Forecasting Challenges in M5

M5 combines four different forecasting problems.

First, there is **cross-series learning**. Many item-store histories are sparse, so a model must borrow information across related products and locations. A global neural model is much more natural than thousands of independent local models.

Second, there is **static-conditioned behavior**. A weekend, holiday, price drop, or SNAP flag does not affect every product in the same way. A Foods item in California and a Hobbies item in Wisconsin should not interpret the same calendar signal identically.

Third, there is **known future information**. The next 28 weekdays are known. Events are known. SNAP flags are known. Price information can also be used carefully. A model that only extrapolates target history leaves a major part of the signal unused.

Fourth, there is **intermittent and sparse demand**. Many item-store series have long stretches of zero sales. Those zeros may reflect genuinely low demand, product lifecycle, assortment, temporary unavailability, stockout, or local store behavior. Treating all zeros as the same kind of observation is usually too crude.

These four requirements create the central tension: TFT is better designed for rich feature interaction and known future covariates, while DeepAR/LSTM is easier to train globally at M5 scale.

## DeepAR/LSTM as a Practical Neural Baseline

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

### Strengths for M5

DeepAR matches the scale of M5 better than a local LSTM. A single global model can be trained over all 30,490 bottom-level series, using static embeddings to distinguish item-store behavior. This is exactly the kind of setting where global recurrent forecasting tends to be useful.

It also has a good probabilistic story for retail sales. M5 targets are non-negative, noisy, and often overdispersed. A negative binomial or similar distributional output is more natural than a plain squared-error head. Even if the final competition forecast is a point forecast, distributional training can make the model more robust to sparse and volatile demand.

DeepAR is also relatively compute-friendly. The dominant recurrent cost grows with batch size, sequence length, number of recurrent layers, and hidden size. If hidden size is kept moderate and windows are sampled rather than fully materialized, the model remains plausible under constrained memory and training-time budgets.

For a dataset with tens of millions of long-format rows, a model that can be trained, debugged, and iterated is often more valuable than a richer model that cannot be run at meaningful scale.

### Limitations

DeepAR's weakness is that the model does not deeply structure the relationship between static metadata, observed history, and known future covariates. Calendar, SNAP, event, and price features can be passed into the model, but their relevance is learned implicitly through embeddings and recurrent states.

That can be limiting for M5. Price sensitivity is product-specific. SNAP effects are state- and category-specific. Event effects vary by department. A plain recurrent architecture has no special mechanism for saying, "This feature matters for this product context, but not for that one."

The second limitation is autoregressive decoding. DeepAR generates future values step by step. A day-28 forecast depends on the generated path through days 1 to 27. This is natural for probabilistic trajectories, but it can also propagate early mistakes. In sparse retail series, early near-zero predictions can make the rest of the horizon too conservative.

The third limitation is long-range memory. LSTMs are good at local temporal dynamics, but annual patterns or distant event analogs should usually be exposed through explicit lag features rather than relying on the recurrent state to remember hundreds of daily steps.

### Practical DeepAR Configuration

A sensible first DeepAR setup would be deliberately compact:

| Design choice | Practical direction |
|---|---|
| Model scope | One global model over all bottom-level item-store series |
| Forecast horizon | 28 days |
| Context length | 56 to 84 days initially |
| Recurrent layers | 1 or 2 |
| Hidden size | Roughly 40 to 80 for a constrained first run |
| Static categoricals | item, department, category, store, state |
| Known future inputs | weekday, month, event flags, SNAP flags, price features |
| Lags | 1, 7, 14, 28, 56, and possibly 364 as an explicit feature |
| Output distribution | Negative binomial is a strong first choice |
| Windowing | Sample windows instead of enumerating all windows |

The key idea is to keep the recurrent model small, use strong feature construction, and rely on global learning.

## TFT as the Natural Architectural Fit

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

TFT is not simply a Transformer added to a time series. It is a hybrid architecture that keeps recurrent local processing while adding feature selection, static conditioning, known-future conditioning, gating, and attention.

### TFT Strengths for M5

The strongest argument for TFT is known future covariates. Each of the 28 future days has a known weekday, calendar position, event state, SNAP flag, and possibly price context. TFT can condition each forecast step directly on these future values. This is cleaner than forcing the model to rely mainly on a recurrent state and a recursive path.

The second argument is static-conditioned feature relevance. M5 demand drivers are heterogeneous. Price features may be very important for one department and less useful for another. SNAP flags may matter more for Foods than for Hobbies. Event effects may be localized by state or product category. TFT has machinery for learning these relationships more explicitly than DeepAR.

The third argument is interpretability. Variable selection weights and attention patterns can be inspected. These should not be treated as causal proof, but they are useful diagnostics. They can help answer whether the model is using price, SNAP, event, and lag features in plausible ways.

The fourth argument is direct multi-horizon modeling. TFT predicts the forecast horizon as a sequence of outputs rather than relying entirely on recursive generation. For a fixed 28-day horizon, this is attractive because each horizon position can be conditioned on its own known future inputs.

### TFT Compute Cost

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

This is the practical catch: the architecture that best matches the data can also be the architecture that consumes more memory, trains more slowly, and is harder to iterate over the full dataset.

For M5, a full TFT with a long encoder, many features, item-level embeddings, multiple attention heads, and all possible windows would be a poor first experiment under limited compute. It may be theoretically appealing, but it is unlikely to be the most efficient way to make progress.

### Practical TFT Configuration

If TFT is used, it should start small:

| Design choice | Practical direction |
|---|---|
| Model scope | Start with a representative subset, then expand |
| Forecast horizon | 28 days |
| Encoder length | 56 to 84 days initially |
| Hidden size | Roughly 8 to 24 for a constrained first run |
| Attention heads | 1 or 2 |
| LSTM layers | 1 |
| Static categoricals | category, department, store, state first; item id can be added later |
| Known future inputs | weekday, month, event, SNAP, price features |
| Observed inputs | sales, selected lags, selected rolling statistics |
| Quantiles | Start with median only or a small set such as 0.1, 0.5, 0.9 |

The main principle is restraint. TFT should be introduced after the feature pipeline is stable, not before. Otherwise the model complexity can hide basic data issues.

## Mathematical Framing: DeepAR vs TFT

The qualitative difference between DeepAR and TFT becomes clearer if the M5 task is written as a conditional forecasting problem.

For item-store series `i`, let:

- $y_{i,t}$ be unit sales on day $t$;
- $s_i$ be static metadata such as item, department, category, store, and state;
- $x_{i,t}$ be time-varying covariates such as weekday, event flags, SNAP flags, price, lags, and rolling statistics;
- $H = 28$ be the forecast horizon;
- $C$ be the context length.

The model is trying to estimate:

```math
p\left(y_{i,T+1:T+H} \mid y_{i,T-C+1:T}, x_{i,T-C+1:T+H}, s_i\right)
```

The crucial detail is that `x` extends into the future for known covariates. Weekday, event, SNAP, and planned price features can be available for `T+1` through `T+28`, while future sales are not.

### DeepAR: Conditional Autoregressive Likelihood

DeepAR factorizes the joint forecast distribution into one-step conditional distributions:

```math
p\left(y_{i,T+1:T+H} \mid \text{history}, x_i, s_i\right)
=
\prod_{h=1}^{H}
p\left(y_{i,T+h} \mid y_{i,T-C+1:T+h-1}, x_{i,T-C+1:T+h}, s_i\right)
```

This is the mathematical meaning of "autoregressive." The day-`h` forecast conditions on previous target values. During training, those previous values are observed. During inference, they are partly generated by the model itself.

A compact DeepAR recurrence can be written as:

```math
\begin{aligned}
e_i &= \mathrm{Embed}(s_i) \\
z_{i,t} &= [y_{i,t-1}, x_{i,t}, e_i] \\
h_{i,t} &= \mathrm{LSTM}(h_{i,t-1}, z_{i,t}) \\
\theta_{i,t} &= W_o h_{i,t} + b_o \\
y_{i,t} &\sim \mathrm{Distribution}(\theta_{i,t})
\end{aligned}
```

For M5, the output distribution matters. Unit sales are non-negative counts, sparse, and overdispersed. A Gaussian head is usually a poor match because it allows negative demand and assumes symmetric noise. A negative binomial head is more natural:

```math
\begin{aligned}
y_{i,t} &\sim \mathrm{NegBin}(\mu_{i,t}, \alpha_{i,t}) \\
\mathrm{Var}(y_{i,t}) &= \mu_{i,t} + \alpha_{i,t}\mu_{i,t}^{2}
\end{aligned}
```

Here $\mu$ is the conditional mean and $\alpha$ controls overdispersion. This is useful for M5 because many item-store series have variance much larger than the mean. The model can learn that a high-volume food item and a low-volume hobby item should not have the same uncertainty structure.

The training objective is negative log likelihood:

```math
\mathcal{L}_{\mathrm{DeepAR}} = - \sum_i \sum_t \log p(y_{i,t} \mid \theta_{i,t})
```

If the likelihood is negative binomial, the loss rewards the model for assigning high probability to the observed count, not merely for getting close under squared error. This is one reason DeepAR can be a better neural baseline than a plain LSTM with an MSE head.

The mathematical weakness is also visible in the factorization. At inference time, each later prediction depends on earlier generated predictions:

```math
\hat{y}_{i,T+h}
\text{ depends on }
\hat{y}_{i,T+1:T+h-1}
\quad \text{for } h = 2,\ldots,28.
```

So errors can propagate through the forecast path. This is especially relevant for intermittent sales. If the model predicts near-zero demand early in the horizon, later hidden states may become too conservative unless strong covariates or lag features counteract that drift.

DeepAR's static conditioning is also mathematically simple. The static embedding `e_i` is usually concatenated into the recurrent input or used to initialize hidden states. That lets the LSTM learn product/store-specific behavior, but it does not explicitly ask which variables matter for which static context. The interaction is learned implicitly inside the recurrent state transition.

### TFT: Direct Multi-Horizon Modeling

TFT does not rely only on the autoregressive factorization above. It is designed to produce horizon-specific forecasts more directly:

```math
\hat{y}_{i,T+h}
=
f_h\left(y_{i,T-C+1:T}, x_{i,T-C+1:T+H}, s_i\right),
\quad h = 1, \ldots, H
```

In practice, TFT often predicts quantiles rather than a full count distribution:

```math
\hat{y}_{i,T+h}^{(q)}
=
\text{the } q\text{-th conditional quantile of } y_{i,T+h}
```

The common loss is quantile loss:

```math
\begin{aligned}
\mathcal{L}_q(y, \hat{y})
&=
\max\left(q(y-\hat{y}), (q-1)(y-\hat{y})\right) \\
\mathcal{L}_{\text{TFT}}
&=
\sum_i \sum_h \sum_q
\mathcal{L}_q\left(y_{i,T+h}, \hat{y}_{i,T+h}^{(q)}\right)
\end{aligned}
```

This has a different modeling flavor from DeepAR. DeepAR defines a parametric distribution such as the **negative binomial**. TFT, trained with **quantile loss**, directly learns selected points of the conditional predictive distribution. For a competition-style point forecast, the median or a mean-like postprocessing step can be used, but mathematically the training objective is different from a count-based likelihood.

Before selecting which covariates matter, each input variable is first transformed using a **Gated Residual Network (GRN)**. A GRN is **a small context-conditioned feed-forward block with residual connections and gating mechanisms** that can either amplify useful nonlinear transformations or suppress unnecessary ones. These GRNs are reused throughout the architecture, including variable selection, static enrichment, and temporal processing.

The main architectural advantage of TFT comes from how variables are processed before sequence modeling. For a set of input variables `j = 1, \ldots, m`, the variable selection network can be written in simplified form as:


```math
\begin{aligned}
v_{j,t}
&=
\mathrm{GRN}_j(x_{j,t}, c) \\

a_t
&=
\mathrm{softmax}
\Big(
\mathrm{GRN}_{gate}(x_{1,t}, \ldots, x_{m,t}, c)
\Big) \\

\tilde{x}_t
&=
\sum_{j=1}^{m} a_{j,t} v_{j,t}
\end{aligned}
```

The weights $a_{j,t}$ are feature-selection weights. The important part for M5 is the `context`: static product/store information can influence which time-varying variables are emphasized. This gives TFT a more explicit mechanism for statements like:

- price matters more for some departments than others;
- SNAP matters more for some states and categories;
- event effects differ by product type;
- lagged demand may dominate for stable products but not for intermittent ones.

DeepAR can learn these interactions too, but they are buried inside the LSTM hidden state. TFT makes them part of the computation graph through context-dependent gating.

TFT also uses gated residual networks. A simplified GRN can be written as:

```math
\begin{aligned}
\mathrm{GRN}(a,c)
&=
\mathrm{LayerNorm}
\Big(
a+\mathrm{Gate}(\eta)
\Big) \\

\eta
&=
W_2\,\mathrm{ELU}(W_1 a+W_c c+b_1)+b_2 \\

\mathrm{Gate}(\eta)
&=
\sigma(W_g\eta)\odot W_s\eta
\end{aligned}
```

The gate lets the model suppress unnecessary nonlinear transformations. In M5 terms, if a feature or interaction is not useful for a given context, TFT has an architectural path for reducing its influence rather than forcing every transformation to be fully active.

After variable selection, TFT still uses recurrent layers for local temporal processing:

```math
\begin{aligned}
u_t
&=
\mathrm{LSTM}_{enc}
(\tilde{x}_{T-C+1:T}) \\

v_h
&=
\mathrm{LSTM}_{dec}
(\tilde{x}_{T+1:T+H})
\end{aligned}
```

Then static enrichment injects static context into the temporal representations:

```math
r_t
=
\mathrm{GRN}(u_t, c_s)
```

where $c_s$ is a learned static context vector derived from $s_i$.

Finally, TFT applies interpretable multi-head attention over temporal positions:

```math
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
```

This allows each forecast horizon to weight relevant historical positions differently. For M5, day `T+7` might attend strongly to the same weekday last week, while day `T+28` might benefit more from four-week lag structure, nearby events, or holiday-adjacent behavior.

The cost is visible in the attention equation. For sequence length `L`, attention has roughly:

```math
O(L^2d)
```

time and memory behavior for the attention score matrix, while the recurrent part is closer to:

```math
O(Ld^2)
```

depending on hidden size `d`. TFT also adds variable-selection and GRN costs that DeepAR does not have. This is why TFT's richer math creates operational friction under constrained compute.

### Implications for M5

The two models are not merely "LSTM versus Transformer." They make different assumptions about the forecasting distribution and the forecast path.

| Mathematical issue | DeepAR | TFT | M5 implication |
|---|---|---|---|
| Forecast factorization | Autoregressive product of one-step distributions | Direct multi-horizon mapping | DeepAR is natural for trajectories, TFT is cleaner for a fixed 28-day horizon |
| Output uncertainty | Parametric likelihood, often negative binomial | Usually quantile loss | DeepAR better matches sparse count demand; TFT better gives horizon-specific quantiles |
| Static covariates | Embeddings inside recurrence | Static encoders condition selection, gating, enrichment | TFT more explicitly models product/store-dependent feature relevance |
| Known future inputs | Fed step by step into decoder | Central part of decoder inputs | TFT uses future calendar/SNAP/price structure more directly |
| Error propagation | Generated future values feed later steps | Less dependent on recursive generated targets | TFT can reduce path drift over 28 days |
| Feature interactions | Implicit in hidden state | Explicit through variable selection and GRNs | TFT is stronger for heterogeneous retail drivers |
| Compute | Smaller recurrent model | Recurrent + gating + attention + selection | DeepAR is easier to run first under limited compute |

DeepAR is not mathematically shallow; its strength is a clean probabilistic autoregressive likelihood across many related series. That is a good match for sparse item-store counts. TFT decomposes the problem into static context, observed history, known future covariates, variable relevance, gated nonlinear transformations, and horizon-specific outputs. That better matches the full structure of M5, but it is also a larger optimization and compute problem.

In short:

```math
\begin{aligned}
\text{DeepAR advantage}
&=
p(y_{\text{future}} \mid y_{\text{past}}, x, s)
\text{ with a count-friendly likelihood and global sharing.} \\
\text{TFT advantage}
&=
f_h(y_{\text{past}}, x_{\text{past}}, x_{\text{future}}, s)
\text{ with explicit context-dependent feature use.}
\end{aligned}
```

For M5, the more realistic staged path is to use DeepAR to establish the global probabilistic baseline, then use TFT when the feature pipeline, sampling strategy, and compute budget can support the extra mathematical machinery.

## Architecture Tradeoffs

The difference between DeepAR and TFT can be summarized as follows:

| Question | DeepAR/LSTM | TFT |
|---|---|---|
| Is it global across many related series? | Yes | Yes |
| Is it practical for all 30K+ series under limited compute? | More practical | Much harder |
| Does it handle static metadata? | Yes, through embeddings | Yes, through static encoders and conditioning |
| Does it use known future covariates naturally? | Yes, but less structurally | Yes, by design |
| Does it handle sparse count-like demand naturally? | Stronger probabilistic framing | Possible, but more dependent on loss and sampling |
| Does it avoid recursive error propagation? | No, it is autoregressive | More so, because it is direct multi-horizon |
| Is it interpretable? | Limited | Better feature and attention diagnostics |
| Is it easy to tune? | Easier | Harder |
| Is it the richer architecture for M5? | No | Yes |
| Is it the better first constrained-compute direction? | Yes | Usually no |

This table is the practical summary: DeepAR is the easier operating point; TFT is the richer model.

## Compute and Training Requirements

Both models require the same disciplined data foundation:

- a long item-store-day table;
- static categorical encodings;
- joined calendar variables;
- joined price variables;
- leakage-safe lag features;
- leakage-safe rolling features;
- target scaling or normalization;
- sampled training windows.

The difference is how much compute each architecture needs to use that structure effectively.

DeepAR can work with a relatively lean feature set because the model is compact and recurrent. It benefits from explicit lags, known future calendar features, static embeddings, and count-friendly likelihoods.

TFT benefits from a cleaner feature taxonomy. Static covariates, known future inputs, and observed historical inputs should be explicitly separated. This separation is one of its main advantages, but it also makes data preparation more demanding.

The practical compute comparison is broader than CPU availability:

| Compute dimension | DeepAR/LSTM | TFT |
|---|---|---|
| Main cost driver | Recurrent pass over sampled windows | Variable selection, GRNs, recurrence, attention, and multi-horizon heads |
| Memory footprint | Lower activation memory | Higher activation memory, especially with longer encoders and wider hidden layers |
| Training time | Faster per iteration and easier to repeat | Slower per iteration; tuning cycles are more expensive |
| Feature-count sensitivity | Moderate | High, because each variable participates in projection and selection |
| Sequence-length sensitivity | Mostly linear through recurrence | Recurrent cost plus attention cost that can grow roughly quadratically in sequence length |
| Window strategy | Sampled windows are usually enough | Sampling is still required, but representative coverage matters more |
| Full 30K+ series run | Reasonable as an early neural run | Better after feature pruning, sampling, and smaller pilot runs |
| Hardware preference | Can start on CPU or modest GPU | Benefits much more from GPU memory and throughput |
| Best use case | Practical global neural baseline | Rich covariate-aware architecture once the pipeline is stable |

Several choices should be avoided regardless of architecture:

- training a separate LSTM per item-store series;
- starting with a large TFT before memory, sampling, and training time are understood;
- using a 365-day encoder before a smaller model works;
- materializing every possible sliding window;
- throwing all engineered features into TFT at once;
- ignoring known future calendar and price signals;
- treating all zero sales as equivalent;
- increasing neural depth before fixing the data pipeline.

## Recommended Modeling Path

The most defensible path is staged:

| Stage | Direction | Reason |
|---|---|---|
| Pilot comparison | DeepAR/LSTM and compact TFT on a representative subset | Compare architectures before paying the full-data training cost |
| First full-data neural model | DeepAR/LSTM | Establish a tractable global neural baseline across all 30K+ series |
| Second-stage richer model | TFT | Exploit known future covariates, static conditioning, and interpretability after the pipeline is stable |
| Competition-style benchmark | LightGBM | Historical M5 evidence strongly supports feature-engineered tree models |

### Pilot Subset Design

The pilot should not be a random sample of rows. Time series models need complete histories, so the sample should select a manageable set of item-store series and keep their full time span, calendar joins, price joins, lags, and rolling features. A reasonable first comparison could use roughly 1,000 bottom-level item-store series, but the design matters more than the exact number.

A good subset should be stratified across the structure of M5:

| Sampling dimension | Design intent |
|---|---|
| Category and department | Include Foods, Household, and Hobbies, not only high-volume categories |
| Store and state | Cover all 10 stores and all 3 states so SNAP and local effects remain visible |
| Sales volume | Include high, medium, low, and near-zero demand series |
| Intermittency | Include both stable sellers and sparse item-store pairs |
| Price behavior | Include items with frequent price movement and items with stable prices |
| Hierarchy coverage | Prefer coherent product-store slices so aggregate diagnostics still mean something |

One practical design is to sample item-store series within each category-store or department-store cell, with weights that preserve a mix of volume and intermittency. Another is to select a smaller number of products across all stores, which keeps product-level behavior coherent and makes hierarchy-level checks cleaner. Either approach is better than taking arbitrary rows from the long table.

The scale-up path can then be explicit:

| Experiment size | Purpose |
|---|---|
| 500 to 1,000 item-store series | Validate feature pipeline, loss behavior, runtime, and rough model ranking |
| 3,000 to 5,000 item-store series | Test whether the ranking survives broader category, store, and demand coverage |
| All 30,490 bottom-level series | Train the full neural baseline only after the smaller runs are stable |

### Controlled Model Comparison

The pilot experiment should hold the comparison fixed:

| Experiment control | Recommendation |
|---|---|
| Time split | Use the same validation horizon for both models, ideally one or more 28-day backtests |
| Feature set | Use the same static, known future, observed, lag, and rolling features where applicable |
| Context length | Start with the same 56 to 84 day encoder/context window |
| Training budget | Compare under a fixed wall-clock or epoch budget, not only best possible accuracy |
| Metrics | Track validation loss, WRMSSE-style accuracy, horizon-wise error, and error by volume/intermittency bucket |
| Operational metrics | Record memory use, training time, inference time, and tuning sensitivity |

The subset is not meant to produce the final leaderboard model. Its purpose is to answer practical questions early: whether TFT's richer covariate handling is visible on M5-like data, whether DeepAR's simpler global likelihood is already competitive, which features are actually useful, and how quickly each model becomes difficult to train. If TFT only wins on a small clean subset but is unstable or too slow as the sample expands, that is useful information before committing to a full 30K+ series run.

**Move first toward DeepAR/LSTM when the goal is a full-data neural baseline under limited compute. Move toward TFT when the objective shifts toward richer covariate handling, interpretability, or a compute budget that can support heavier experimentation.**

TFT remains the more architecturally relevant model for the full complexity of M5, but its advantages are only useful if the training setup can support them.

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
