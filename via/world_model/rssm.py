"""World model: a Recurrent State-Space Model (RSSM) for mental simulation.

State s = (deter h, stoch z). Transition p(s' | s, a) lets the agent imagine
action consequences before committing (Hafner et al., Dreamer-style).

Deliberate design choice: the RSSM observes the *pooled observation embedding*
(obs_embed, 256-d) produced by the belief network's encoder — not raw pixels
and not the belief mu. Because its decoder reconstructs that same space,
imagined futures can be fed straight back into the belief GRU, which is how
the decision module computes expected information gain (how much would my
belief uncertainty shrink if I took these actions and saw what follows?).

Losses follow the standard ELBO: observation reconstruction + KL(posterior ||
prior) with free bits.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from via.contracts import C

_LOGVAR_MIN, _LOGVAR_MAX = -8.0, 4.0


@dataclass
class RSSMState:
    deter: torch.Tensor    # (B, deter_dim)
    mu: torch.Tensor       # (B, stoch_dim)
    logvar: torch.Tensor   # (B, stoch_dim)
    stoch: torch.Tensor    # (B, stoch_dim) — sampled

    @property
    def feature(self) -> torch.Tensor:
        """(B, deter_dim + stoch_dim) — input to decoder/utility heads."""
        return torch.cat([self.deter, self.stoch], dim=-1)

    def entropy(self) -> torch.Tensor:
        d = self.mu.shape[-1]
        return 0.5 * (d * (1.0 + torch.log(torch.tensor(2.0 * torch.pi))) + self.logvar.sum(-1))

    def detach(self) -> "RSSMState":
        return RSSMState(*(t.detach() for t in (self.deter, self.mu, self.logvar, self.stoch)))


def _gaussian(mu_logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, logvar = mu_logvar.chunk(2, dim=-1)
    logvar = logvar.clamp(_LOGVAR_MIN, _LOGVAR_MAX)
    stoch = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    return mu, logvar, stoch


class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int = C.obs_embed_dim,
        action_dim: int = C.action_dim,
        deter_dim: int = C.deter_dim,
        stoch_dim: int = C.stoch_dim,
        hidden_dim: int = 256,
        free_bits: float = 1.0,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.free_bits = free_bits

        self.input_proj = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim), nn.GELU()
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior_head = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * stoch_dim)
        )
        self.posterior_head = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, obs_dim)
        )

    def init_state(self, batch: int, device: torch.device | None = None) -> RSSMState:
        p = next(self.parameters())
        device = device or p.device
        z = torch.zeros(batch, self.stoch_dim, device=device)
        return RSSMState(
            deter=torch.zeros(batch, self.deter_dim, device=device),
            mu=z, logvar=torch.zeros_like(z), stoch=z.clone(),
        )

    # ---- one-step transitions ----

    def prior_step(self, state: RSSMState, action: torch.Tensor) -> RSSMState:
        """Imagination step p(s' | s, a) — no observation."""
        x = self.input_proj(torch.cat([state.stoch, action], dim=-1))
        deter = self.gru(x, state.deter)
        mu, logvar, stoch = _gaussian(self.prior_head(deter))
        return RSSMState(deter=deter, mu=mu, logvar=logvar, stoch=stoch)

    def posterior_step(
        self, state: RSSMState, action: torch.Tensor, obs_embed: torch.Tensor
    ) -> tuple[RSSMState, RSSMState]:
        """Filtering step: returns (posterior, prior) at t+1 given obs_{t+1}."""
        prior = self.prior_step(state, action)
        mu, logvar, stoch = _gaussian(
            self.posterior_head(torch.cat([prior.deter, obs_embed], dim=-1))
        )
        post = RSSMState(deter=prior.deter, mu=mu, logvar=logvar, stoch=stoch)
        return post, prior

    def decode(self, state: RSSMState) -> torch.Tensor:
        """Predicted observation embedding (B, obs_dim)."""
        return self.decoder(state.feature)

    # ---- sequence ops ----

    def observe(
        self, obs_embeds: torch.Tensor, actions: torch.Tensor
    ) -> tuple[list[RSSMState], list[RSSMState]]:
        """Filter a trajectory. obs_embeds (B, T, obs_dim), actions (B, T-1, A).

        Returns (posteriors[1..T-1], priors[1..T-1]); the initial state is
        seeded from obs_embeds[:, 0] via a posterior step with zero action.
        """
        B, T = obs_embeds.shape[:2]
        state = self.init_state(B, obs_embeds.device)
        zero_a = torch.zeros(B, actions.shape[-1], device=obs_embeds.device)
        state, _ = self.posterior_step(state, zero_a, obs_embeds[:, 0])
        posts, priors = [], []
        for t in range(T - 1):
            state, prior = self.posterior_step(state, actions[:, t], obs_embeds[:, t + 1])
            posts.append(state)
            priors.append(prior)
        return posts, priors

    def imagine(self, state: RSSMState, actions: torch.Tensor) -> list[RSSMState]:
        """Roll out p(s'|s,a) for actions (B, H, A) -> H imagined states."""
        states = []
        for t in range(actions.shape[1]):
            state = self.prior_step(state, actions[:, t])
            states.append(state)
        return states

    # ---- training ----

    def loss(self, obs_embeds: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        posts, priors = self.observe(obs_embeds, actions)
        recon = torch.stack(
            [F.mse_loss(self.decode(p), obs_embeds[:, t + 1]) for t, p in enumerate(posts)]
        ).mean()

        kl_terms = []
        for post, prior in zip(posts, priors):
            var_p, var_q = prior.logvar.exp(), post.logvar.exp()
            kl = 0.5 * (
                prior.logvar - post.logvar + (var_q + (post.mu - prior.mu) ** 2) / var_p - 1.0
            ).sum(-1)
            kl_terms.append(kl)
        kl = torch.stack(kl_terms).mean()
        kl_clipped = kl.clamp(min=self.free_bits)

        return {"loss": recon + kl_clipped, "recon": recon, "kl": kl}

    @torch.no_grad()
    def rollout_mse(
        self, obs_embeds: torch.Tensor, actions: torch.Tensor, k: int = 5
    ) -> dict[str, torch.Tensor]:
        """k-step open-loop prediction error vs the 'persistence' naive baseline
        (predict obs_{t+k} = obs_t). The Checkpoint-1 world-model metric."""
        B, T = obs_embeds.shape[:2]
        if T < k + 1:
            raise ValueError(f"need at least {k + 1} steps, got {T}")
        state = self.init_state(B, obs_embeds.device)
        zero_a = torch.zeros(B, actions.shape[-1], device=obs_embeds.device)
        state, _ = self.posterior_step(state, zero_a, obs_embeds[:, 0])
        imagined = self.imagine(state, actions[:, :k])
        pred = self.decode(imagined[-1])
        model_mse = F.mse_loss(pred, obs_embeds[:, k])
        naive_mse = F.mse_loss(obs_embeds[:, 0], obs_embeds[:, k])
        return {"model_mse": model_mse, "naive_mse": naive_mse}
