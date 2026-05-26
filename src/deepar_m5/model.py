from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Shape and capacity settings needed to construct a DeepAR model."""

    cardinalities: list[int]
    covariate_dim: int
    hidden_size: int = 64
    embedding_dim: int = 16
    num_layers: int = 1
    dropout: float = 0.0


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
        #  a helper function for the fast.ai heuristic
        def get_emb_dim(cardinality: int) -> int:
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
        self.output = nn.Linear(config.hidden_size, 2)

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
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run one autoregressive step and produce negative-binomial parameters."""

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
        alpha = F.softplus(raw[:, 1:2]) + 1e-4
        return mu_scaled, alpha, next_states

    def forward(
        self,
        target: torch.Tensor,
        covariates: torch.Tensor,
        static_cats: torch.Tensor,
        scale: torch.Tensor,
        prior_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            logger.info(f"forward: mu shape : {mu_scaled.shape}")
            logger.info(f"forward: alpha shape : {alpha.shape}")
            mus.append(mu_scaled * scale)
            alphas.append(alpha)
            
            # Update history with ground truth (teacher forcing)
            next_target_scaled = target[:, step : step + 1] / scale.clamp_min(1e-4)
            history = torch.cat([history[:, 1:], next_target_scaled], dim=1)

        return torch.cat(mus, dim=1), torch.cat(alphas, dim=1)

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
            logger.info(f"predict mean: mu shape : {mu_scaled.shape}")
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
            logger.info(f"predict samples step < context - mu shape : {mu_scaled.shape}")
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
            logger.info(f"predict samples step > context - mu shape : {mu_scaled.shape}")
            mu = mu_scaled * repeated_scale
            sample = sample_negative_binomial(mu, alpha)
            samples.append(sample)
            
            # Update history with prediction (sample)
            sample_scaled = sample / repeated_scale.clamp_min(1e-4)
            history = torch.cat([history[:, 1:], sample_scaled], dim=1)

        stacked = torch.cat(samples, dim=1)
        return stacked.view(batch_size, num_samples, -1).transpose(0, 1).contiguous()

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


def sample_negative_binomial(mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Sample counts from the same Negative Binomial parameterization used by the loss."""

    mu = mu.clamp_min(1e-6)
    alpha = alpha.clamp_min(1e-6)
    probs = (mu / (mu + alpha)).clamp(1e-6, 1.0 - 1e-6)
    distribution = torch.distributions.NegativeBinomial(total_count=alpha, probs=probs)
    return distribution.sample()
