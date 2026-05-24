# Understanding DeepAR Training And Prediction

This note explains how the from-scratch DeepAR implementation trains and predicts, using the same mental model as the DeepAR paper: an autoregressive recurrent network that outputs parameters of a probability distribution at every time step.

Primary references:

- Salinas et al., [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)
- Published version: [International Journal of Forecasting, 2020](https://www.sciencedirect.com/science/article/pii/S0169207019301888)

## Core Idea

DeepAR models a target series autoregressively:

$$
\begin{aligned}
p(z_{t_0:T} \mid z_{1:t_0-1}, x_{1:T}) &= \prod_{t=t_0}^{T} p(z_t \mid z_{1:t-1}, x_{1:T})
\end{aligned}
$$

Here:

- $z_t$ is the target value, such as daily unit sales.
- $x_t$ contains known covariates, such as calendar, SNAP, and price features.
- The recurrent hidden state summarizes the previous target history.
- The neural network outputs distribution parameters, not a single fixed forecast value.

In this repo, the model outputs Negative Binomial parameters:

$$
\mu_t, \alpha_t = f_\theta(h_t)
$$

where $\mu_t$ is the mean and $\alpha_t$ is the shape parameter used by this implementation.

## What The LSTM Receives At Each Step

A common misunderstanding is that each LSTM step receives the full context window. It does not.

At time $t$, the LSTM receives only the previous target value directly:

$$
z_{t-1}
$$

plus current covariates and static features:

$$
x_t, s_i
$$

The longer history is carried through the hidden and cell states:

$$
h_{t-1}, c_{t-1}
$$

So each recurrent step is conceptually:

$$
h_t, c_t = \mathrm{LSTM}(z_{t-1}, x_t, s_i, h_{t-1}, c_{t-1})
$$

Then the output head converts $h_t$ into distribution parameters:

$$
\mu_t, \alpha_t = f_\theta(h_t)
$$

In code, the per-step input is built in [`model.py`](./src/deepar_m5/model.py):

```python
x = torch.cat([prev_scaled_target, covariates_t, static_emb, log_scale], dim=-1)
```

So the input contains:

- previous scaled target
- current time-step covariates
- static category embeddings
- log scale for the series

## Why The Sequence Length Is Context Plus Horizon

If:

```text
context_length = 14
prediction_length = 7
```

then:

```text
sequence_length = 21
```

Each training window is:

```text
days 1-14   context
days 15-21  prediction horizon
```

The LSTM unrolls over all 21 steps because the first 14 steps build up the recurrent state, and the final 7 steps are the forecast horizon being scored.

The loss mask is created in [`data.py`](./src/deepar_m5/data.py):

```python
loss_mask = np.zeros_like(targets, dtype=np.float32)
loss_mask[:, self.context_length :] = 1.0
```

So for a 14-context, 7-horizon window:

```text
context:  0 0 0 0 0 0 0 0 0 0 0 0 0 0
horizon:  1 1 1 1 1 1 1
```

The model may produce distribution parameters for all 21 steps, but only the final 7 steps contribute to the training loss.

## Training: Teacher Forcing

During training, the model knows the entire sampled window, including the future horizon inside that historical window. It therefore uses the actual previous target value at every step. This is teacher forcing.

In [`DeepAR.forward`](./src/deepar_m5/model.py), the key line is:

```python
prev_scaled = target[:, step : step + 1] / scale.clamp_min(1e-4)
```

That means after predicting parameters for step $t$, the next input uses the actual observed $z_t$.

Conceptually:

```text
step 1:  input previous value 0, predict day 1
step 2:  input actual day 1, predict day 2
step 3:  input actual day 2, predict day 3
...
step 15: input actual day 14, predict day 15
```

The loss is applied only on the horizon part of the window.

This follows the DeepAR training idea: maximize likelihood of observed target values under the distribution predicted by the recurrent network. In implementation terms, we minimize negative log likelihood:

$$\mathcal{L}(\theta) = - \sum_i \sum_{t \in \Omega_i} \log p_\theta(z_{i,t} \mid z_{i,1:t-1}, x_{i,1:T})$$

where $\Omega_i$ is the set of time steps being scored. In this repo, $\Omega_i$ is selected by `loss_mask`, so only forecast-horizon positions in each sampled window are included.

## Prediction: Autoregressive Rollout

During inference, future target values are not known.

The inference batch is built in [`WindowSampler.make_inference_batch`](./src/deepar_m5/data.py):

```python
targets = np.zeros((len(series_idx), self.sequence_length), dtype=np.float32)
targets[:, : self.context_length] = self.bundle.sales_values[series_idx, start:forecast_start]
```

So:

```text
context positions: actual known sales
future positions:  zero placeholders
```

The future zeros are not used as previous target values. In [`DeepAR.predict_mean`](./src/deepar_m5/model.py), the logic is:

```python
if step >= context_length:
    predictions.append(mu)
    prev_scaled = mu_scaled
else:
    prev_scaled = target[:, step : step + 1] / scale.clamp_min(1e-4)
```

So prediction works like this:

```text
context steps:
    use actual historical sales

future steps:
    predict the next distribution
    feed the previous prediction back into the next step
```

For a 7-day horizon:

$$p(z_{t+1:t+7} \mid z_{1:t}) = \prod_{k=1}^{7} p(z_{t+k} \mid z_{1:t}, \hat{z}_{t+1:t+k-1}, x_{1:t+7})$$

This is the autoregressive forecast rollout described by DeepAR.

## Mean Decoding Versus Sampling

The DeepAR paper is probabilistic. At prediction time, the full probabilistic approach samples future paths from the predicted distribution:

```text
sample path 1:  3, 5, 2, 0, 1, 4, 2
sample path 2:  4, 3, 2, 1, 0, 3, 6
sample path 3:  2, 2, 5, 1, 1, 2, 3
...
```

Those sample paths can then be summarized into:

- mean forecast
- median forecast
- quantiles such as P10, P50, P90
- prediction intervals

This repo supports both mean decoding and sampled trajectory decoding.

Mean decoding uses:

```python
prev_scaled = mu_scaled
```

That means it feeds the predicted mean back into the next future step instead of sampling a random value from the Negative Binomial distribution.

Sampled decoding uses:

```python
sample = sample_negative_binomial(mu, alpha)
prev_scaled = sample / repeated_scale.clamp_min(1e-4)
```

That means each simulated path feeds its own sampled value into the next future step.

So the implementation supports:

```text
probabilistic training loss
mean autoregressive prediction
sampled autoregressive prediction
sample means and quantiles
```

To match the paper more closely for uncertainty estimates, use the sampling path, which draws:

$$
z_t \sim \mathrm{NegativeBinomial}(\mu_t, \alpha_t)
$$

then rolls forward many paths and computes summary statistics.

Operational command examples are kept in [DeepAR M5 Setup Guide](./SETUP_GUIDE.md). Typical usage patterns are:

- Use `mean` when you want a fast point forecast.
- Use `sample-mean` when you want the expected value implied by sampled trajectories.
- Use `quantile` when you need uncertainty-aware forecasts such as P10, P50, or P90.
- Increase `--num-samples` for smoother quantiles, at the cost of slower inference.

## Why Negative Binomial For M5

M5 daily unit sales are count data:

```text
0, 1, 2, 3, ...
```

They are also often intermittent and overdispersed:

```text
0, 0, 0, 1, 0, 4, 0, 12, ...
```

For count data, common distribution choices are:

| Distribution | Good for | Limitation |
|---|---|---|
| Gaussian / Normal | continuous real-valued targets | can assign probability to negative sales |
| Student-t | continuous targets with outliers | still not naturally count-valued |
| Poisson | non-negative counts | assumes variance equals mean |
| Negative Binomial | non-negative counts with variance larger than mean | more parameters than Poisson |
| Zero-inflated Poisson / Negative Binomial | count data with many extra zeros | not implemented in this repo |

Poisson assumes:

$$
\mathrm{Var}(z_t) = \mathbb{E}[z_t]
$$

Retail demand often has variance larger than the mean. Negative Binomial allows overdispersion:

$$
\mathrm{Var}(z_t) = \mu_t + \frac{\mu_t^2}{\alpha_t}
$$

In this repo's parameterization, larger $\alpha_t$ means less extra dispersion; smaller $\alpha_t$ means more dispersion.

## Negative Binomial Loss In This Repo

The model outputs two positive parameters in [`DeepAR._step`](./src/deepar_m5/model.py):

```python
raw = self.output(x)
mu_scaled = F.softplus(raw[:, :1]) + 1e-4
alpha = F.softplus(raw[:, 1:2]) + 1e-4
```

The `softplus` function ensures both parameters are positive.

The model then rescales the mean:

```python
mus.append(mu_scaled * scale)
```

The Negative Binomial probability mass function used here is:

$$p(z \mid \mu, \alpha) = \frac{\Gamma(z + \alpha)} {\Gamma(\alpha)\Gamma(z + 1)} \left(\frac{\alpha}{\alpha + \mu}\right)^\alpha \left(\frac{\mu}{\alpha + \mu}\right)^z$$

Taking logs:

$$
 \begin{aligned}
 \log p(z \mid \mu, \alpha) &= \log\Gamma(z + \alpha) - \log\Gamma(\alpha) - \log\Gamma(z + 1) \\
 &\quad + \alpha \left [\log\alpha - \log(\alpha + \mu)\right ] + z \left [\log\mu - \log(\alpha + \mu)\right ]
 \end{aligned}
 $$

The loss is negative log likelihood:
$$\mathrm{NLL}(z, \mu, \alpha) = -\log p(z \mid \mu, \alpha)$$

In code:

```python
log_prob = (
    torch.lgamma(target + alpha)
    - torch.lgamma(alpha)
    - torch.lgamma(target + 1.0)
    + alpha * (torch.log(alpha) - torch.log(alpha + mu))
    + target * (torch.log(mu) - torch.log(alpha + mu))
)
loss = -log_prob
```

Then the mask keeps only the forecast-horizon positions:

```python
return (loss * mask).sum() / mask.sum().clamp_min(1.0)
```

## Paper Parameterization Note

The DeepAR paper discusses likelihood functions such as Gaussian and Negative Binomial, and trains by maximizing likelihood. Different implementations parameterize the Negative Binomial differently.

This repo uses:

$$
\mathrm{Var}(z) = \mu + \frac{\mu^2}{\alpha}
$$

Some texts use a dispersion parameter $\alpha_{\text{disp}}$ where:

$$
\mathrm{Var}(z) = \mu + \alpha_{\text{disp}}\mu^2
$$

The mapping is:

$$
\alpha_{\text{disp}} = \frac{1}{\alpha_{\text{repo}}}
$$

So when comparing formulas, check whether $\alpha$ means shape or dispersion.

## How This Maps To The Current Files

| Concept | Code location |
|---|---|
| Build context-plus-horizon windows | [`WindowSampler._make_batch`](./src/deepar_m5/data.py) |
| Build inference windows with future placeholders | [`WindowSampler.make_inference_batch`](./src/deepar_m5/data.py) |
| Teacher-forced training | [`DeepAR.forward`](./src/deepar_m5/model.py) |
| Mean autoregressive prediction | [`DeepAR.predict_mean`](./src/deepar_m5/model.py) |
| Sampled autoregressive prediction | [`DeepAR.predict_samples`](./src/deepar_m5/model.py) |
| Negative Binomial parameters | [`DeepAR._step`](./src/deepar_m5/model.py) |
| Negative Binomial NLL | [`negative_binomial_nll`](./src/deepar_m5/model.py) |
| Negative Binomial sampling | [`sample_negative_binomial`](./src/deepar_m5/model.py) |
| Prediction mode CLI | [`infer.py`](./src/deepar_m5/infer.py) |

## Practical Summary

During training:

```text
actual previous target -> LSTM -> distribution for current target
loss only on forecast horizon
```

During prediction:

```text
actual context targets -> LSTM state
predicted mean -> next future input
repeat until horizon is complete
```

The implementation is aligned with the DeepAR autoregressive likelihood idea. Use `--forecast-mode mean` for fast point forecasts, or `--forecast-mode quantile` / `--forecast-mode sample-mean` when you want predictions based on sampled future paths.
