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
    covariate_dim: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    distribution: str = "negative-binomial"
    tweedie_power: float = 1.5
    tweedie_dispersion: float = 1.0


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
        # Input: prev_scaled_target(1), dynamic_lag_7(1), dynamic_roll_7(1), covariates(dim), static(dim), scale(1)
        # We add 2 to the input_size for lag_7 and rolling_mean_7
        input_size = 1 + 2 + config.covariate_dim + static_dim + 1

        cells = [] 
        for layer_idx in range(config.num_layers):
            cells.append(
                ScratchLSTMCell(
                    input_size=input_size if layer_idx == 0 else config.hidden_size,
                    hidden_size=config.hidden_size,
                )
            )
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(config.dropout)
        self.distribution = normalize_distribution(config.distribution)
        output_dim = 2 if self.distribution == "negative-binomial" else 1
        self.output = nn.Linear(config.hidden_size, output_dim)

    def static_embedding(self, static_cats: torch.Tensor) -> torch.Tensor:
        """Embed and concatenate all static categorical identifiers."""

        pieces = [embedding(static_cats[:, idx]) for idx, embedding in enumerate(self.embeddings)]
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
        covariates_t: torch.Tensor,
        static_emb: torch.Tensor,
        log_scale: torch.Tensor,
        states: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run one autoregressive step and produce distribution parameters."""

        # target_history shape: [batch, 28] (or whatever buffer size we chose)
        # We need lag_7 and rolling_mean_7 (last 7 days)
        # target_history contains the most recent predictions/targets up to t-1
        
        lag_7 = target_history[:, -7 : -6]
        roll_7 = target_history[:, -7:].mean(dim=1, keepdim=True)

        x = torch.cat([prev_scaled_target, lag_7, roll_7, covariates_t, static_emb, log_scale], dim=-1)
        
        next_states = []
        for layer_idx, cell in enumerate(self.cells):
            h, c = cell(x, states[layer_idx])
            x = self.dropout(h) if layer_idx < len(self.cells) - 1 else h
            next_states.append((h, c))

        raw = self.output(x)
        mu_scaled = F.softplus(raw[:, :1]) + 1e-4
        alpha = F.softplus(raw[:, 1:2]) + 1e-4 if self.distribution == "negative-binomial" else None
        return mu_scaled, alpha, next_states

    def forward(
        self,
        target: torch.Tensor,
        covariates: torch.Tensor,
        static_cats: torch.Tensor,
        scale: torch.Tensor,
        prior_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run teacher-forced training over a full context-plus-horizon window."""

        batch_size, seq_len = target.shape
        static_emb = self.static_embedding(static_cats)
        log_scale = torch.log1p(scale)
        states = self.initial_state(batch_size, target.device)

        if prior_history is None:
            logger.warning("prior_history is not given - initialization will be affected!")
            prior_history = torch.zeros(batch_size, 28, device=target.device)
        
        # history buffer scaled
        history = prior_history / scale.clamp_min(1e-4)
        
        mus = []
        alphas = []
        for step in range(seq_len):
            prev_scaled = history[:, -1:]
            mu_scaled, alpha, states = self._step(
                prev_scaled,
                history,
                covariates[:, step, :],
                static_emb,
                log_scale,
                states,
            )
            mus.append(mu_scaled * scale)
            if alpha is not None:
                alphas.append(alpha)
            
            # Update history with ground truth (teacher forcing)
            next_target_scaled = target[:, step : step + 1] / scale.clamp_min(1e-4)
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
    ) -> torch.Tensor:
        """Decode future means autoregressively after consuming known context."""

        batch_size, seq_len = target.shape
        static_emb = self.static_embedding(static_cats)
        log_scale = torch.log1p(scale)
        states = self.initial_state(batch_size, target.device)

        if prior_history is None:
            prior_history = torch.zeros(batch_size, 28, device=target.device)
        
        history = prior_history / scale.clamp_min(1e-4)

        predictions = []
        for step in range(seq_len):
            prev_scaled = history[:, -1:]
            mu_scaled, _, states = self._step(
                prev_scaled,
                history,
                covariates[:, step, :],
                static_emb,
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
    ) -> torch.Tensor:
        """Optimized decoding that avoids redundant context computation."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        batch_size, seq_len = target.shape
        horizon_length = seq_len - context_length

        static_emb = self.static_embedding(static_cats)
        log_scale = torch.log1p(scale)
        states = self.initial_state(batch_size, target.device)
        
        if prior_history is None:
            prior_history = torch.zeros(batch_size, 28, device=target.device)
        
        history = prior_history / scale.clamp_min(1e-4)

        # ------------------------------------------------------------------
        # PHASE 1: CONSUME CONTEXT
        # ------------------------------------------------------------------
        for step in range(context_length):
            prev_scaled = history[:, -1:]
            mu_scaled, alpha, states = self._step(
                prev_scaled,
                history,
                covariates[:, step, :],
                static_emb,
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
            mu_scaled, alpha, states = self._step(
                prev_scaled,
                history,
                repeated_covariates[:, step, :],
                static_emb,
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
        return stacked.view(batch_size, num_samples, -1).transpose(0, 1).contiguous()

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
            return sample_negative_binomial(mu, aux)
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
    """Create ``ModelConfig`` while ignoring stale checkpoint-only keys."""

    valid_keys = set(ModelConfig.__dataclass_fields__)
    filtered = {key: value for key, value in payload.items() if key in valid_keys}
    if "loss" in payload and "distribution" not in filtered:
        filtered["distribution"] = payload["loss"]
    return ModelConfig(**filtered)


def sample_negative_binomial(mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Sample counts from the same Negative Binomial parameterization used by the loss."""

    mu = mu.clamp_min(1e-6)
    alpha = alpha.clamp_min(1e-6)
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
