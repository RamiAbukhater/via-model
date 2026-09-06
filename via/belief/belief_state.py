"""Belief state network: perception as probabilistic inference.

A GRU-based recurrent filter reads pooled patch embeddings across time and
maintains a Gaussian belief N(mu, diag(sigma^2)) over a 256-d latent world
state. Trained self-supervised: a sample from the belief must predict the
*next* observation embedding. The variance head is calibrated by a Gaussian
NLL on that prediction — when the scene is occluded or ambiguous the network
can only lower its loss by admitting uncertainty, so sigma rises.

Interfaces:
    encode_obs(patches, proprio) (B, P, 768), (B, proprio_dim) -> obs embed (B, 256)
    step(obs_embed, hidden)      one filtering update -> BeliefState
    rollout(patches_seq, proprio_seq)  (B, T, P, 768), (B, T, proprio_dim) -> beliefs over time
    loss(patches_seq, proprio_seq)     self-supervised training objective

`encode_obs` fuses the pooled visual embedding with a small proprioceptive
vector (end-effector position + gripper state -- see via/contracts.py) before
the final projection, so obs_embed_dim (and everything downstream: the RSSM,
the utility head) is unaffected -- only encode_obs's callers need to also
supply proprio.
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
        proprio_dim: int = C.proprio_dim,
        num_queries: int = 4,
        kl_weight: float = 1e-3,
        var_weight: float = 1.0,
    ):
        super().__init__()
        self.belief_dim = belief_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        self.var_weight = var_weight

        # Attention pooling: learned queries summarize 196 patches spatially.
        self.queries = nn.Parameter(torch.randn(num_queries, obs_embed_dim) * 0.02)
        self.patch_proj = nn.Linear(patch_dim, obs_embed_dim)
        self.pool_attn = nn.MultiheadAttention(obs_embed_dim, num_heads=4, batch_first=True)
        self.pool_out = nn.Linear(num_queries * obs_embed_dim, obs_embed_dim)

        # Proprioception fusion: end-effector position + gripper state fed
        # in alongside vision (see via/contracts.py's proprio_dim docstring).
        # Normalized first since raw ee_pos (meters, robot-frame) and
        # gripper finger joint positions live on different natural scales --
        # exactly the kind of unnormalized-scale trap that bit obs_embed
        # (see obs_norm below) and the eu/ig CEM objective terms.
        self.proprio_norm = nn.LayerNorm(proprio_dim)
        self.proprio_mlp = nn.Sequential(nn.Linear(proprio_dim, 64), nn.GELU())
        self.fuse = nn.Linear(obs_embed_dim + 64, obs_embed_dim)
        # obs_embed's absolute scale is otherwise unconstrained: var_loss only
        # pushes per-clip temporal std up (never down), so nothing stops it
        # from growing far past what next_obs_logvar's clamp (var <= e^4 ~=
        # 54.6) can represent as predictive variance. Real SigLIP patches
        # (std ~2.3) vs StubPerception (std ~0.24, scaled by *0.02 at init)
        # drive this to very different scales, so a fixed clamp tuned against
        # one silently breaks on the other. LayerNorm bounds obs_embed to a
        # consistent scale regardless of which perception encoder feeds it.
        self.obs_norm = nn.LayerNorm(obs_embed_dim)

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

    def encode_obs(self, patches: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """(B, P, patch_dim), (B, proprio_dim) -> (B, obs_embed_dim)."""
        assert_shape(patches, (-1, C.patch_count, C.patch_dim), "patches")
        B = patches.shape[0]
        kv = self.patch_proj(patches)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, kv, kv)               # (B, Q, D)
        vis = self.pool_out(pooled.flatten(1))               # (B, D)
        prop = self.proprio_mlp(self.proprio_norm(proprio))  # (B, 64)
        return self.obs_norm(self.fuse(torch.cat([vis, prop], dim=-1)))  # (B, D)

    def step(self, obs_embed: torch.Tensor, hidden: torch.Tensor) -> BeliefState:
        """One filtering update from a pooled observation embedding."""
        h = self.gru(obs_embed, hidden)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(_LOGVAR_MIN, _LOGVAR_MAX)
        return BeliefState(mu=mu, logvar=logvar, hidden=h)

    def rollout(self, patches_seq: torch.Tensor, proprio_seq: torch.Tensor) -> list[BeliefState]:
        """(B, T, P, patch_dim), (B, T, proprio_dim) -> list of T BeliefStates."""
        B, T = patches_seq.shape[:2]
        h = self.init_hidden(B, patches_seq.device)
        beliefs = []
        for t in range(T):
            b = self.step(self.encode_obs(patches_seq[:, t], proprio_seq[:, t]), h)
            beliefs.append(b)
            h = b.hidden
        return beliefs

    # ---- training ----

    def loss(
        self,
        patches_seq: torch.Tensor,
        proprio_seq: torch.Tensor,
        occluded: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Self-supervised next-observation prediction with NLL-calibrated variance.

        A single reparameterized sample from b_t predicts obs_{t+1}; the
        Gaussian NLL makes the belief variance an honest uncertainty estimate.
        A small KL(b_t || N(0, I)) keeps the latent well-scaled.

        obs_embed's own no-grad copy is the NLL's prediction target, so
        nothing stops the pooling head from collapsing to a near-constant
        vector and still minimizing the loss. var_loss is a VICReg-style
        variance floor (Bardes et al. 2022), computed per clip, across time,
        over visible frames only. A batch-wide floor is satisfied for free
        by between-clip differences (color, start position); an
        all-frames floor is satisfied for free by the occlusion blank/reveal
        jump. Restricting to visible frames within a clip forces real motion
        to be the only way to satisfy it. `occluded` is (B, T).
        """
        B, T = patches_seq.shape[:2]
        if T < 2:
            raise ValueError("belief loss needs at least 2 timesteps")
        beliefs = self.rollout(patches_seq, proprio_seq)
        with torch.no_grad():
            targets = torch.stack(
                [self.encode_obs(patches_seq[:, t], proprio_seq[:, t]) for t in range(1, T)], dim=1
            )  # (B, T-1, D)
        obs_embeds = torch.stack(
            [self.encode_obs(patches_seq[:, t], proprio_seq[:, t]) for t in range(T)], dim=1
        )  # (B, T, D), WITH grad — feeds the anti-collapse term only

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
        if occluded is None:
            visible = torch.ones(B, T, 1, device=obs_embeds.device)
        else:
            visible = (~occluded).float().unsqueeze(-1)  # (B, T, 1)
        count = visible.sum(dim=1).clamp(min=2.0)                                  # (B, 1)
        mean = (obs_embeds * visible).sum(dim=1) / count                           # (B, D)
        var = ((obs_embeds - mean.unsqueeze(1)) ** 2 * visible).sum(dim=1) / count  # (B, D)
        temporal_std = var.clamp(min=1e-12).sqrt()
        var_loss = F.relu(1.0 - temporal_std).mean()
        total = nll + self.kl_weight * kl + self.var_weight * var_loss
        mean_sigma = torch.stack([b.sigma.mean() for b in beliefs]).mean()
        return {
            "loss": total, "nll": nll, "kl": kl, "var_loss": var_loss, "mean_sigma": mean_sigma,
        }
