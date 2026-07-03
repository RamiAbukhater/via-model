"""Decision module: action selection as expected utility + epistemic value.

Candidate action sequences a_{1:H} are scored with the active-inference
objective (Friston 2017):

    J(a) = E[U(a)] + lambda(sigma_b) * IG(a)

E[U]   — expected goal achievement: the world model imagines the state
         trajectory under a, and a learned utility head scores each imagined
         state against the (probability-weighted) goal embedding.
IG(a)  — expected information gain: imagined future observations are decoded
         from the world model and pushed through the belief network's GRU;
         IG is the resulting drop in belief entropy. Actions that would
         reveal occluded or ambiguous parts of the scene score high.
lambda — an adaptive gate, a small learned function of the current belief
         sigma (and goal entropy): when the agent is confused lambda rises
         and it explores; when confident lambda falls and it exploits.

Optimization is cross-entropy-method (CEM) search over action sequences; the
first action of the best-scoring elite mean is executed (MPC-style).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from via.belief.belief_state import BeliefState, BeliefStateNetwork
from via.contracts import C
from via.world_model.rssm import RSSM, RSSMState


class UtilityHead(nn.Module):
    """U(state, goal): scores an imagined world-model state against a goal."""

    def __init__(
        self,
        feature_dim: int = C.deter_dim + C.stoch_dim,
        goal_dim: int = C.goal_embed_dim,
        hidden: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + goal_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feature: torch.Tensor, goal_embed: torch.Tensor) -> torch.Tensor:
        """(B, F), (B, G) -> (B,) utility."""
        return self.net(torch.cat([feature, goal_embed], dim=-1)).squeeze(-1)


class AdaptiveGate(nn.Module):
    """lambda(uncertainty) in [0, lambda_max], learned.

    Input: [mean belief sigma, max belief sigma, goal entropy] — cheap scalar
    summaries so the gate generalizes across scenes.
    """

    def __init__(self, lambda_max: float = 2.0, hidden: int = 32):
        super().__init__()
        self.lambda_max = lambda_max
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, belief: BeliefState, goal_entropy: torch.Tensor) -> torch.Tensor:
        """-> (B,) lambda values."""
        x = torch.stack(
            [belief.sigma.mean(-1), belief.sigma.amax(-1), goal_entropy], dim=-1
        )
        return self.lambda_max * torch.sigmoid(self.net(x)).squeeze(-1)


@dataclass
class CEMConfig:
    horizon: int = C.plan_horizon
    population: int = 64
    elites: int = 6
    iterations: int = 3
    init_std: float = 0.5


class CEMPlanner:
    """Cross-entropy method over action sequences (no gradients at plan time)."""

    def __init__(self, config: CEMConfig | None = None):
        self.cfg = config or CEMConfig()

    def plan(self, score_fn, batch: int, device: torch.device) -> torch.Tensor:
        """score_fn: (B, N, H, A) -> (B, N). Returns best first action (B, A)."""
        cfg = self.cfg
        mean = torch.zeros(batch, cfg.horizon, C.action_dim, device=device)
        std = torch.full_like(mean, cfg.init_std)
        for _ in range(cfg.iterations):
            noise = torch.randn(batch, cfg.population, cfg.horizon, C.action_dim, device=device)
            candidates = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(
                C.action_low, C.action_high
            )
            scores = score_fn(candidates)                                   # (B, N)
            elite_idx = scores.topk(cfg.elites, dim=1).indices              # (B, E)
            gather = elite_idx[..., None, None].expand(-1, -1, cfg.horizon, C.action_dim)
            elites = candidates.gather(1, gather)                           # (B, E, H, A)
            mean = elites.mean(1)
            std = elites.std(1).clamp(min=0.05)
        return mean[:, 0]


class DecisionModule(nn.Module):
    def __init__(
        self,
        world_model: RSSM,
        belief_net: BeliefStateNetwork,
        utility: UtilityHead | None = None,
        gate: AdaptiveGate | None = None,
        planner: CEMPlanner | None = None,
        ig_samples: int = 1,
    ):
        super().__init__()
        self.world_model = world_model
        self.belief_net = belief_net
        self.utility = utility or UtilityHead()
        self.gate = gate or AdaptiveGate()
        self.planner = planner or CEMPlanner()
        self.ig_samples = ig_samples

    # ---- objective terms ----

    def expected_utility(
        self, rssm_state: RSSMState, actions: torch.Tensor, goal_embed: torch.Tensor
    ) -> torch.Tensor:
        """actions (B*, H, A), goal (B*, G) -> (B*,) mean utility over the rollout."""
        states = self.world_model.imagine(rssm_state, actions)
        utils = torch.stack([self.utility(s.feature, goal_embed) for s in states])  # (H, B*)
        return utils.mean(0)

    def information_gain(
        self, belief: BeliefState, rssm_state: RSSMState, actions: torch.Tensor
    ) -> torch.Tensor:
        """IG(a) = H(b_t) - H(b_{t+H} | imagined observations)  -> (B*,).

        The world model imagines future observation embeddings under a; those
        are pushed through the belief GRU as if they had been perceived, and
        the entropy drop of the resulting belief is the epistemic value.
        """
        h0 = belief.entropy()
        states = self.world_model.imagine(rssm_state, actions)
        hidden = belief.hidden
        final = belief
        for s in states:
            obs_hat = self.world_model.decode(s)
            final = self.belief_net.step(obs_hat, hidden)
            hidden = final.hidden
        return h0 - final.entropy()

    # ---- action selection ----

    @torch.no_grad()
    def select_action(
        self,
        belief: BeliefState,
        rssm_state: RSSMState,
        goal_embed: torch.Tensor,
        goal_entropy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """MPC step: returns {action (B, A), lambda (B,), and diagnostics}."""
        B = belief.mu.shape[0]
        device = belief.mu.device
        lam = self.gate(belief, goal_entropy)                               # (B,)

        def score(candidates: torch.Tensor) -> torch.Tensor:
            Bc, N, H, A = candidates.shape
            flat = candidates.reshape(Bc * N, H, A)

            def tile(t: torch.Tensor) -> torch.Tensor:
                return t.repeat_interleave(N, dim=0)

            state_n = RSSMState(
                tile(rssm_state.deter), tile(rssm_state.mu),
                tile(rssm_state.logvar), tile(rssm_state.stoch),
            )
            belief_n = BeliefState(tile(belief.mu), tile(belief.logvar), tile(belief.hidden))
            eu = self.expected_utility(state_n, flat, tile(goal_embed))
            ig = self.information_gain(belief_n, state_n, flat)
            return (eu + tile(lam) * ig).view(Bc, N)

        action = self.planner.plan(score, B, device)
        return {"action": action, "lambda": lam}
