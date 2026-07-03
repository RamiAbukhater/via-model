"""Belief state network: perception as probabilistic inference.

A GRU-based recurrent filter reads pooled patch embeddings across time and
maintains a Gaussian belief N(mu, diag(sigma^2)) over a 256-d latent world
state. Trained self-supervised: a sample from the belief must predict the
*next* observation embedding. The variance head is calibrated by a Gaussian
NLL on that prediction — when the scene is occluded or ambiguous the network
can only lower its loss by admitting uncertainty, so sigma rises.

Interfaces:
    encode_obs(patches)          (B, P, 768) -> obs embed (B, 256)
    step(obs_embed, hidden)      one filtering update -> BeliefState
    rollout(patches_seq)         (B, T, P, 768) -> beliefs over time
    loss(patches_seq)            self-supervised training objective
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from via.contracts import C, assert_shape

_LOGVAR_MIN, _LOGVAR_MAX = -8.0, 4.0


@dataclass
class BeliefState:
    """Gaussian belief over the latent world state, plus the GRU carry."""

    mu: torch.Tensor        # (B, belief_dim)
    logvar: torch.Tensor    # (B, belief_dim)
    hidden: torch.Tensor    # (B, belief_hidden_dim)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvar)

    def sample(self, n: int = 1) -> torch.Tensor:
        """Reparameterized samples: (n, B, belief_dim)."""
        eps = torch.randn(n, *self.mu.shape, device=self.mu.device)
        return self.mu.unsqueeze(0) + self.sigma.unsqueeze(0) * eps

    def entropy(self) -> torch.Tensor:
        """Differential entropy of the diagonal Gaussian, per batch element (B,)."""
        d = self.mu.shape[-1]
        return 0.5 * (d * (1.0 + torch.log(torch.tensor(2.0 * torch.pi))) + self.logvar.sum(-1))

    def detach(self) -> "BeliefState":
        return BeliefState(self.mu.detach(), self.logvar.detach(), self.hidden.detach())


class BeliefStateNetwork(nn.Module):
    def __init__(
        self,
        patch_dim: int = C.patch_dim,
        obs_embed_dim: int = C.obs_embed_dim,
        belief_dim: int = C.belief_dim,
        hidden_dim: int = C.belief_hidden_dim,
        num_queries: int = 4,
        kl_weight: float = 1e-3,
    ):
        super().__init__()
        self.belief_dim = belief_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight

        # Attention pooling: learned queries summarize 196 patches spatially.
        self.queries = nn.Parameter(torch.randn(num_queries, obs_embed_dim) * 0.02)
        self.patch_proj = nn.Linear(patch_dim, obs_embed_dim)
        self.pool_attn = nn.MultiheadAttention(obs_embed_dim, num_heads=4, batch_first=True)
        self.pool_out = nn.Linear(num_queries * obs_embed_dim, obs_embed_dim)

        self.gru = nn.GRUCell(obs_embed_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, belief_dim)
        self.logvar_head = nn.Linear(hidden_dim, belief_dim)

        # Self-supervised head: sampled belief -> predicted next obs embedding.
        self.next_obs_mu = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, obs_embed_dim)
        )
        self.next_obs_logvar = nn.Parameter(torch.zeros(obs_embed_dim))

    # ---- filtering ----

    def init_hidden(self, batch: int, device: Optional[torch.device] = None) -> torch.Tensor:
        p = next(self.parameters())
        return torch.zeros(batch, self.hidden_dim, device=device or p.device, dtype=p.dtype)

    def encode_obs(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, P, patch_dim) -> (B, obs_embed_dim)."""
        assert_shape(patches, (-1, C.patch_count, C.patch_dim), "patches")
        B = patches.shape[0]
        kv = self.patch_proj(patches)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, kv, kv)               # (B, Q, D)
        return self.pool_out(pooled.flatten(1))             # (B, D)

    def step(self, obs_embed: torch.Tensor, hidden: torch.Tensor) -> BeliefState:
        """One filtering update from a pooled observation embedding."""
        h = self.gru(obs_embed, hidden)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(_LOGVAR_MIN, _LOGVAR_MAX)
        return BeliefState(mu=mu, logvar=logvar, hidden=h)

    def rollout(self, patches_seq: torch.Tensor) -> list[BeliefState]:
        """(B, T, P, patch_dim) -> list of T BeliefStates."""
        B, T = patches_seq.shape[:2]
        h = self.init_hidden(B, patches_seq.device)
        beliefs = []
        for t in range(T):
            b = self.step(self.encode_obs(patches_seq[:, t]), h)
            beliefs.append(b)
            h = b.hidden
        return beliefs

    # ---- training ----

    def loss(self, patches_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        """Self-supervised next-observation prediction with NLL-calibrated variance.

        A single reparameterized sample from b_t predicts obs_{t+1}; the
        Gaussian NLL makes the belief variance an honest uncertainty estimate.
        A small KL(b_t || N(0, I)) keeps the latent well-scaled.
        """
        B, T = patches_seq.shape[:2]
        if T < 2:
            raise ValueError("belief loss needs at least 2 timesteps")
        beliefs = self.rollout(patches_seq)
        with torch.no_grad():
            targets = torch.stack(
                [self.encode_obs(patches_seq[:, t]) for t in range(1, T)], dim=1
            )  # (B, T-1, D)

        nll_terms, kl_terms = [], []
        for t in range(T - 1):
            b = beliefs[t]
            z = b.sample(1).squeeze(0)                       # (B, belief_dim)
            pred_mu = self.next_obs_mu(z)
            pred_lv = self.next_obs_logvar.clamp(_LOGVAR_MIN, _LOGVAR_MAX)
            nll = 0.5 * ((targets[:, t] - pred_mu) ** 2 / pred_lv.exp() + pred_lv).sum(-1)
            kl = 0.5 * (b.mu**2 + b.logvar.exp() - 1.0 - b.logvar).sum(-1)
            nll_terms.append(nll)
            kl_terms.append(kl)

        nll = torch.stack(nll_terms).mean()
        kl = torch.stack(kl_terms).mean()
        total = nll + self.kl_weight * kl
        mean_sigma = torch.stack([b.sigma.mean() for b in beliefs]).mean()
        return {"loss": total, "nll": nll, "kl": kl, "mean_sigma": mean_sigma}
