from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)


def normalize_distribution(distribution: str) -> str:
    """Normalize user-facing distribution aliases."""

    normalized = distribution.lower().replace("_", "-")
    if normalized in {"negative-binomial", "nb"}:
        return "negative-binomial"
    if normalized == "tweedie":
        return "tweedie"
    raise ValueError(f"Unknown distribution: {distribution}")


@dataclass
class ModelConfig:
    """Shape and capacity settings needed to construct a DeepAR model."""

    cardinalities: list[int]
    event_cardinalities: list[int]
    covariate_dim: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    distribution: str = "negative-binomial"
    tweedie_power: float = 1.5
    tweedie_dispersion: float = 1.0
    prior_history_length: int = 28
    rolling_short_window: int = 7
    rolling_long_window: int = 28
    zero_counter_log_divisor: float = 4.0
    log_scale_divisor: float = 5.0
    nb_alpha_max: float = 1e4


class ScratchLSTMCell(nn.Module):
    """Single LSTM cell implemented from raw PyTorch tensor operations."""

    def __init__(self, input_size: int, hidden_size: int):
        """Initialize input/recurrent weights for one custom LSTM layer."""

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use stable initializers for input, recurrent, and bias parameters."""

        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias)
        # --- initializing bias with 1 ---
        # Since gates are ordered i, f, g, o: index 1 targets the forget gate
        with torch.no_grad():
            self.bias.chunk(4, dim=-1)[1].fill_(1.0)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one LSTM step and return the next hidden and cell states."""

        h_prev, c_prev = state
        gates = x @ self.weight_ih + h_prev @ self.weight_hh + self.bias
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class DeepAR(nn.Module):
    """Global DeepAR-style recurrent forecaster for related count time series."""

    def __init__(self, config: ModelConfig):
        """Create static embeddings, custom LSTM stack, and distribution head."""

        super().__init__()
        self.config = config

        def get_emb_dim(cardinality: int) -> int:
            """Use the fast.ai categorical embedding size heuristic."""

            return min(50, (cardinality + 1) // 2)
        
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, get_emb_dim(c)) for c in config.cardinalities]
        )
        static_dim = sum(get_emb_dim(c) for c in config.cardinalities)

        self.event_embeddings = nn.ModuleList(
            [nn.Embedding(c, get_emb_dim(c)) for c in config.event_cardinalities]
        )
        event_dim = sum(get_emb_dim(c) for c in config.event_cardinalities)
        if config.prior_history_length < config.rolling_long_window:
            raise ValueError("prior_history_length must be at least rolling_long_window.")

        # Input components:
        # 1. Previous scaled target (1)
        # 2. Dynamic rolling features: mean_7, mean_28 (2)
        # 3. Dynamic zero counter (1)
        # 4. Continuous covariates (config.covariate_dim)
        # 5. Event embeddings (event_dim)
        # 6. Static embeddings (static_dim)
        # 7. Log scale (1)
        input_size = 1 + 2 + 1 + config.covariate_dim + event_dim + static_dim + 1

        cells = [] 
        norms = []
        for layer_idx in range(config.num_layers):
            cells.append(
                ScratchLSTMCell(
                    input_size=input_size if layer_idx == 0 else config.hidden_size,
                    hidden_size=config.hidden_size,
                )
            )
            norms.append(nn.LayerNorm(config.hidden_size))

        self.cells = nn.ModuleList(cells)
        self.norms = nn.ModuleList(norms)
        # Match PyTorch LSTM dropout semantics: dropout is inter-layer only, so
        # a one-layer model intentionally sees no dropout from this module.
        self.dropout = nn.Dropout(config.dropout)
        self.distribution = normalize_distribution(config.distribution)
        output_dim = 2 if self.distribution == "negative-binomial" else 1
        self.output = nn.Linear(config.hidden_size, output_dim)

    def static_embedding(self, static_cats: torch.Tensor) -> torch.Tensor:
        """Embed and concatenate all static categorical identifiers."""

        pieces = [embedding(static_cats[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        return torch.cat(pieces, dim=-1)

    def event_embedding(self, event_ids: torch.Tensor) -> torch.Tensor:
        """Embed and concatenate all temporal categorical events."""

        pieces = [embedding(event_ids[:, idx].long()) for idx, embedding in enumerate(self.event_embeddings)]
        if not pieces:
            return event_ids.new_zeros((event_ids.shape[0], 0), dtype=torch.float32)
        return torch.cat(pieces, dim=-1)

    def initial_state(self, batch_size: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Create zero initial hidden/cell states for every recurrent layer."""

        return [
            (
                torch.zeros(batch_size, self.config.hidden_size, device=device),
                torch.zeros(batch_size, self.config.hidden_size, device=device),
            )
            for _ in range(self.config.num_layers)
        ]

    def _step(
        self,
        prev_scaled_target: torch.Tensor,
        target_history: torch.Tensor,
        zero_counter: torch.Tensor,
        covariates_t: torch.Tensor,
        static_emb: torch.Tensor,
        scale: torch.Tensor,
        log_scale: torch.Tensor,
        states: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run one autoregressive step and produce distribution parameters."""

        # Calculate rolling means from the recurrent history buffer. ``lag_28``
        # remains a known point-lag covariate, while this is an autoregressive
        # trailing average that follows teacher-forced targets during training
        # and model outputs during sampling.
        roll_7 = target_history[:, -self.config.rolling_short_window :].mean(dim=1, keepdim=True)
        roll_28 = target_history[:, -self.config.rolling_long_window :].mean(dim=1, keepdim=True)

        # Update zero counter: scale-aware threshold
        prev_units = prev_scaled_target * scale
        new_zero_counter = torch.where(prev_units < 0.5, zero_counter + 1.0, torch.zeros_like(zero_counter))
        
        # NORMALIZE counter for LSTM input: 
        # log1p maps [0, 50, 1000] -> [0, 3.9, 6.9]. 
        # Dividing by the configured divisor centers typical droughts in the [0, 1] range.
        zero_counter_norm = torch.log1p(new_zero_counter) / self.config.zero_counter_log_divisor
        
        # Split covariates_t into continuous and event IDs
        num_events = len(self.event_embeddings)
        continuous_covs = covariates_t[:, :self.config.covariate_dim]
        event_ids = covariates_t[:, self.config.covariate_dim : self.config.covariate_dim + num_events]
        
        event_emb = self.event_embedding(event_ids)

        # Also normalize the static log_scale input.
        log_scale_norm = log_scale / self.config.log_scale_divisor

        x = torch.cat([
            prev_scaled_target, 
            roll_7, 
            roll_28, 
            zero_counter_norm,
            continuous_covs, 
            event_emb,
            static_emb, 
            log_scale_norm
        ], dim=-1)
        
        next_states = []
        for layer_idx, cell in enumerate(self.cells):
            h, c = cell(x, states[layer_idx])
            # Store the normalized hidden state as recurrent state; this keeps
            # gate inputs stable but deliberately changes vanilla LSTM dynamics.
            h = self.norms[layer_idx](h)
            x = self.dropout(h) if layer_idx < len(self.cells) - 1 else h
            next_states.append((h, c))

        raw = self.output(x)
        mu_scaled = F.softplus(raw[:, :1]) + 1e-4
        alpha = F.softplus(raw[:, 1:2]) + 1e-4 if self.distribution == "negative-binomial" else None
        return mu_scaled, alpha, new_zero_counter, next_states

    def forward(
        self,
        target: torch.Tensor,
        covariates: torch.Tensor,
        static_cats: torch.Tensor,
        scale: torch.Tensor,
        prior_history: torch.Tensor | None = None,
        initial_zero_counter: torch.Tensor | None = None,
        context_length: int | None = None,
        rolled_feedback_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one training window with optional rolled feedback.

        By default this is standard DeepAR teacher forcing: every recurrent
        feedback value comes from the observed target. When
        ``rolled_feedback_prob`` is positive and ``context_length`` is supplied,
        context steps remain teacher-forced but forecast-horizon steps
        stochastically feed detached samples from the model's own predictive
        distribution. The loss is still computed against ground truth; only the
        recurrent input path is perturbed to reduce train/inference mismatch.
        """

        batch_size, seq_len = target.shape
        if context_length is None:
            context_length = seq_len
        if not 0 <= context_length <= seq_len:
            raise ValueError(f"context_length must be in [0, {seq_len}], found {context_length}.")
        rolled_feedback_prob = min(max(float(rolled_feedback_prob), 0.0), 1.0)

        static_emb = self.static_embedding(static_cats)
        log_scale = torch.log1p(scale)
        states = self.initial_state(batch_size, target.device)

        if prior_history is None:
            logger.warning("prior_history is not given - initialization will be affected!")
            prior_history = torch.zeros(batch_size, self.config.prior_history_length, device=target.device)
        elif prior_history.shape[1] != self.config.prior_history_length:
            raise ValueError(
                f"Expected prior_history length {self.config.prior_history_length}, "
                f"found {prior_history.shape[1]}."
            )
        
        if initial_zero_counter is None:
            initial_zero_counter = torch.zeros(batch_size, 1, device=target.device)
        elif initial_zero_counter.ndim == 1:
            initial_zero_counter = initial_zero_counter.unsqueeze(-1)
        
        # history buffer scaled
        history = prior_history / scale.clamp_min(1e-4)
        zero_counter = initial_zero_counter
        
        mus = []
        alphas = []
        for step in range(seq_len):
            prev_scaled = history[:, -1:]
            mu_scaled, alpha, zero_counter, states = self._step(
                prev_scaled,
                history,
                zero_counter,
                covariates[:, step, :],
                static_emb,
                scale,
                log_scale,
                states,
            )
            mus.append(mu_scaled * scale)
            if alpha is not None:
                alphas.append(alpha)
            
            truth_scaled = target[:, step : step + 1] / scale.clamp_min(1e-4)
            if rolled_feedback_prob > 0.0 and step >= context_length:
                with torch.no_grad():
                    sampled_units = self.sample(mu_scaled * scale, alpha)
                    sampled_scaled = sampled_units / scale.clamp_min(1e-4)
                use_sample = torch.rand_like(truth_scaled) < rolled_feedback_prob
                next_target_scaled = torch.where(use_sample, sampled_scaled, truth_scaled)
            else:
                next_target_scaled = truth_scaled
            history = torch.cat([history[:, 1:], next_target_scaled], dim=1)

        return torch.cat(mus, dim=1), torch.cat(alphas, dim=1) if alphas else None

    @torch.no_grad()
    def predict_mean(
        self,
        target: torch.Tensor,
        covariates: torch.Tensor,
        static_cats: torch.Tensor,
        scale: torch.Tensor,
        context_length: int,
        prior_history: torch.Tensor | None = None,
        initial_zero_counter: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode future means autoregressively after consuming known context."""

        batch_size, seq_len = target.shape
        static_emb = self.static_embedding(static_cats)
        log_scale = torch.log1p(scale)
        states = self.initial_state(batch_size, target.device)

        if prior_history is None:
            prior_history = torch.zeros(batch_size, self.config.prior_history_length, device=target.device)
        elif prior_history.shape[1] != self.config.prior_history_length:
            raise ValueError(
                f"Expected prior_history length {self.config.prior_history_length}, "
                f"found {prior_history.shape[1]}."
            )
        
        if initial_zero_counter is None:
            initial_zero_counter = torch.zeros(batch_size, 1, device=target.device)
        elif initial_zero_counter.ndim == 1:
            initial_zero_counter = initial_zero_counter.unsqueeze(-1)
        
        history = prior_history / scale.clamp_min(1e-4)
        zero_counter = initial_zero_counter

        predictions = []
        for step in range(seq_len):
            prev_scaled = history[:, -1:]
            mu_scaled, _, zero_counter, states = self._step(
                prev_scaled,
                history,
                zero_counter,
                covariates[:, step, :],
                static_emb,
                scale,
                log_scale,
                states,
            )
            mu = mu_scaled * scale
            
            if step >= context_length:
                predictions.append(mu)
                # Update history with prediction
                history = torch.cat([history[:, 1:], mu_scaled], dim=1)
            else:
                # Consume context (update history with ground truth)
                next_target_scaled = target[:, step : step + 1] / scale.clamp_min(1e-4)
                history = torch.cat([history[:, 1:], next_target_scaled], dim=1)

        return torch.cat(predictions, dim=1)

    @torch.no_grad()
    def predict_samples(
        self,
        target: torch.Tensor,
        covariates: torch.Tensor,
        static_cats: torch.Tensor,
        scale: torch.Tensor,
        context_length: int,
        num_samples: int,
        prior_history: torch.Tensor | None = None,
        initial_zero_counter: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Optimized decoding that returns samples as ``[num_samples, batch, horizon]``."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        batch_size, seq_len = target.shape
        horizon_length = seq_len - context_length

        static_emb = self.static_embedding(static_cats)
        log_scale = torch.log1p(scale)
        states = self.initial_state(batch_size, target.device)
        
        if prior_history is None:
            prior_history = torch.zeros(batch_size, self.config.prior_history_length, device=target.device)
        elif prior_history.shape[1] != self.config.prior_history_length:
            raise ValueError(
                f"Expected prior_history length {self.config.prior_history_length}, "
                f"found {prior_history.shape[1]}."
            )
        
        if initial_zero_counter is None:
            initial_zero_counter = torch.zeros(batch_size, 1, device=target.device)
        elif initial_zero_counter.ndim == 1:
            initial_zero_counter = initial_zero_counter.unsqueeze(-1)
        
        history = prior_history / scale.clamp_min(1e-4)
        zero_counter = initial_zero_counter

        # ------------------------------------------------------------------
        # PHASE 1: CONSUME CONTEXT
        # ------------------------------------------------------------------
        for step in range(context_length):
            prev_scaled = history[:, -1:]
            mu_scaled, alpha, zero_counter, states = self._step(
                prev_scaled,
                history,
                zero_counter,
                covariates[:, step, :],
                static_emb,
                scale,
                log_scale,
                states,
            )
            next_target_scaled = target[:, step : step + 1] / scale.clamp_min(1e-4)
            history = torch.cat([history[:, 1:], next_target_scaled], dim=1)

        # ------------------------------------------------------------------
        # THE SPLIT: CLONE THE WORLD
        # ------------------------------------------------------------------
        def duplicate(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.repeat_interleave(num_samples, dim=0)

        expanded_states = []
        for h, c in states:
            expanded_states.append((duplicate(h), duplicate(c)))
        states = expanded_states

        history = duplicate(history)
        zero_counter = duplicate(zero_counter)
        static_emb = duplicate(static_emb)
        log_scale = duplicate(log_scale)
        repeated_scale = duplicate(scale)
        repeated_covariates = duplicate(covariates[:, context_length:, :])

        # ------------------------------------------------------------------
        # PHASE 2: HORIZON GENERATION (WITH DUPLICATES)
        # ------------------------------------------------------------------
        samples = []
        for step in range(horizon_length):
            prev_scaled = history[:, -1:]
            mu_scaled, alpha, zero_counter, states = self._step(
                prev_scaled,
                history,
                zero_counter,
                repeated_covariates[:, step, :],
                static_emb,
                repeated_scale,
                log_scale,
                states,
            )
            mu = mu_scaled * repeated_scale
            sample = self.sample(mu, alpha)
            samples.append(sample)
            
            # Update history with prediction (sample)
            sample_scaled = sample / repeated_scale.clamp_min(1e-4)
            history = torch.cat([history[:, 1:], sample_scaled], dim=1)

        stacked = torch.cat(samples, dim=1)
        samples_by_batch = stacked.view(batch_size, num_samples, horizon_length)
        return samples_by_batch.transpose(0, 1).contiguous()

    def loss(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        aux: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the configured masked distribution loss."""

        if self.distribution == "negative-binomial":
            if aux is None:
                raise ValueError("Negative Binomial loss requires alpha.")
            return negative_binomial_nll(target, mu, aux, mask)
        return tweedie_deviance_loss(
            target,
            mu,
            power=self.config.tweedie_power,
            dispersion=self.config.tweedie_dispersion,
            mask=mask,
        )

    def sample(self, mu: torch.Tensor, aux: torch.Tensor | None) -> torch.Tensor:
        """Sample from the configured predictive distribution."""

        if self.distribution == "negative-binomial":
            if aux is None:
                raise ValueError("Negative Binomial sampling requires alpha.")
            return sample_negative_binomial(mu, aux, alpha_max=self.config.nb_alpha_max)
        return sample_tweedie(
            mu,
            power=self.config.tweedie_power,
            dispersion=self.config.tweedie_dispersion,
        )

    def to_config_dict(self) -> dict:
        """Return the serializable model configuration saved in checkpoints."""

        return asdict(self.config)


def negative_binomial_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked negative-binomial negative log likelihood.

    ``mu`` is the mean count and ``alpha`` is the positive shape parameter. The
    implementation uses lgamma identities directly instead of ``torch.distributions``.
    """

    target = target.clamp_min(0.0)
    mu = mu.clamp_min(1e-6)
    alpha = alpha.clamp_min(1e-6)

    log_prob = (
        torch.lgamma(target + alpha)
        - torch.lgamma(alpha)
        - torch.lgamma(target + 1.0)
        + alpha * (torch.log(alpha) - torch.log(alpha + mu))
        + target * (torch.log(mu) - torch.log(alpha + mu))
    )
    loss = -log_prob
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def tweedie_deviance_loss(
    target: torch.Tensor,
    mu: torch.Tensor,
    power: float = 1.5,
    dispersion: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked Tweedie deviance loss up to target-only constants.

    For ``1 < power < 2``, the Tweedie family models non-negative continuous
    values with a point mass at zero, which is a useful proxy for intermittent
    retail demand. The omitted terms do not depend on the prediction; fixed
    dispersion only scales the objective and controls sampling variance.
    """

    if not 1.0 < power < 2.0:
        raise ValueError("Tweedie power must be strictly between 1 and 2.")
    if dispersion <= 0.0:
        raise ValueError("Tweedie dispersion must be positive.")

    target = target.clamp_min(0.0)
    mu = mu.clamp_min(1e-6)
    loss = ((mu.pow(2.0 - power) / (2.0 - power)) - (
        target * mu.pow(1.0 - power) / (1.0 - power)
    )) / dispersion
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def tweedie_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    power: float = 1.5,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward-compatible alias for the fixed-dispersion Tweedie deviance."""

    return tweedie_deviance_loss(target, mu, power=power, mask=mask)


def masked_forecast_loss(
    loss_name: str,
    target: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor | None,
    mask: torch.Tensor | None = None,
    tweedie_power: float = 1.5,
    tweedie_dispersion: float = 1.0,
) -> torch.Tensor:
    """Dispatch a masked forecast loss without requiring a model instance."""

    normalized = normalize_distribution(loss_name)
    if normalized == "negative-binomial":
        if alpha is None:
            raise ValueError("Negative Binomial loss requires alpha.")
        return negative_binomial_nll(target, mu, alpha, mask)
    return tweedie_deviance_loss(target, mu, power=tweedie_power, dispersion=tweedie_dispersion, mask=mask)


def model_config_from_dict(payload: dict) -> ModelConfig:
    """Create ``ModelConfig`` and fail clearly on unsupported checkpoint keys."""

    valid_keys = set(ModelConfig.__dataclass_fields__)
    migration_keys = {"loss"}
    unknown = sorted(set(payload) - valid_keys - migration_keys)
    if unknown:
        raise ValueError(f"Unrecognized ModelConfig keys in checkpoint: {unknown}")
    filtered = {key: value for key, value in payload.items() if key in valid_keys}
    if "loss" in payload and "distribution" not in filtered:
        filtered["distribution"] = payload["loss"]
    return ModelConfig(**filtered)


def sample_negative_binomial(mu: torch.Tensor, alpha: torch.Tensor, alpha_max: float = 1e4) -> torch.Tensor:
    """Sample counts from the same Negative Binomial parameterization used by the loss."""

    mu = mu.clamp_min(1e-6)
    alpha = alpha.clamp(1e-6, max(float(alpha_max), 1.0))
    probs = (mu / (mu + alpha)).clamp(1e-6, 1.0 - 1e-6)
    distribution = torch.distributions.NegativeBinomial(total_count=alpha, probs=probs)
    return distribution.sample()


def sample_tweedie(mu: torch.Tensor, power: float = 1.5, dispersion: float = 1.0) -> torch.Tensor:
    """Sample Tweedie values using the compound Poisson-Gamma representation."""

    if not 1.0 < power < 2.0:
        raise ValueError("Tweedie power must be strictly between 1 and 2.")
    if dispersion <= 0.0:
        raise ValueError("Tweedie dispersion must be positive.")

    mu = mu.clamp_min(1e-6)
    poisson_rate = (mu.pow(2.0 - power) / (dispersion * (2.0 - power))).clamp_max(1e6)
    poisson_counts = torch.poisson(poisson_rate)
    gamma_shape = (2.0 - power) / (power - 1.0)
    gamma_scale = dispersion * (power - 1.0) * mu.pow(power - 1.0)
    gamma = torch.distributions.Gamma(
        concentration=(poisson_counts * gamma_shape).clamp_min(1e-6),
        rate=(1.0 / gamma_scale).clamp_min(1e-6),
    )
    sample = gamma.sample()
    return torch.where(poisson_counts > 0.0, sample, torch.zeros_like(sample))
