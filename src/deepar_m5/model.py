from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    cardinalities: list[int]
    covariate_dim: int
    hidden_size: int = 64
    embedding_dim: int = 16
    num_layers: int = 1
    dropout: float = 0.0


class ScratchLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, config.embedding_dim) for cardinality in config.cardinalities]
        )
        static_dim = len(config.cardinalities) * config.embedding_dim
        input_size = 1 + config.covariate_dim + static_dim + 1

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
        pieces = [embedding(static_cats[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        return torch.cat(pieces, dim=-1)

    def initial_state(self, batch_size: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
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
        covariates_t: torch.Tensor,
        static_emb: torch.Tensor,
        log_scale: torch.Tensor,
        states: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = torch.cat([prev_scaled_target, covariates_t, static_emb, log_scale], dim=-1)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = target.shape
        static_emb = self.static_embedding(static_cats)
        log_scale = torch.log1p(scale)
        states = self.initial_state(batch_size, target.device)

        prev_scaled = torch.zeros(batch_size, 1, device=target.device)
        mus = []
        alphas = []
        for step in range(seq_len):
            mu_scaled, alpha, states = self._step(
                prev_scaled,
                covariates[:, step, :],
                static_emb,
                log_scale,
                states,
            )
            mus.append(mu_scaled * scale)
            alphas.append(alpha)
            prev_scaled = target[:, step : step + 1] / scale.clamp_min(1e-4)

        return torch.cat(mus, dim=1), torch.cat(alphas, dim=1)

    @torch.no_grad()
    def predict_mean(
        self,
        target: torch.Tensor,
        covariates: torch.Tensor,
        static_cats: torch.Tensor,
        scale: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        batch_size, seq_len = target.shape
        static_emb = self.static_embedding(static_cats)
        log_scale = torch.log1p(scale)
        states = self.initial_state(batch_size, target.device)

        prev_scaled = torch.zeros(batch_size, 1, device=target.device)
        predictions = []
        for step in range(seq_len):
            mu_scaled, _, states = self._step(
                prev_scaled,
                covariates[:, step, :],
                static_emb,
                log_scale,
                states,
            )
            mu = mu_scaled * scale
            if step >= context_length:
                predictions.append(mu)
                prev_scaled = mu_scaled
            else:
                prev_scaled = target[:, step : step + 1] / scale.clamp_min(1e-4)

        return torch.cat(predictions, dim=1)

    def to_config_dict(self) -> dict:
        return asdict(self.config)


def negative_binomial_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
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
