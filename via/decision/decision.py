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

    Input: [mean belief sigma, max belief sigma, goal entropy, running min of
    mean sigma so far this episode] — cheap scalar summaries so the gate
    generalizes across scenes.

    The `min_sigma` term was added 2026-08-16 (see docs/EXPERIMENT_LOG.md):
    live closed-loop eval found belief sigma barely moves within an episode
    (mean 0.7605 in the first 10 steps vs. 0.7563 in the last 10, across 100
    episodes) — it fluctuates but never *settles* low, so a gate driven only
    by instantaneous sigma never sees a sustained "confident" state and
    lambda stays parked near 1 (IG weighted comparably to, and >=50% of the
    time more than, expected utility) for the whole episode, including the
    precision-critical final approach/grasp window. Giving the gate a
    monotonically non-increasing running minimum lets it learn to commit to
    exploitation once confidence has ever been achieved, rather than
    requiring it on the exact current step.

    **Currently neutralized to a constant zero at every call site** (found
    2026-08-2x, see docs/EXPERIMENT_LOG.md): the running min is computed over
    16-step training clips but up to 300-step live episodes, so at inference
    time it ratchets down to values far more extreme than anything seen in
    training — the gate was extrapolating wildly out-of-distribution, causing
    a severe closed-loop regression (end-of-episode gripper-to-target
    distance ~1.0-1.1 vs. 0.47 pre-feature). Isolation-tested: zeroing this
    input recovered behavior to roughly match the pre-feature baseline. The
    input slot is kept (rather than dropping back to a 3-input gate) to avoid
    another real-data retrain purely for architectural tidiness — a constant
    input is mathematically absorbed into the first layer's bias, so this is
    behaviorally identical to not having the feature. Revisit properly (e.g.
    train against episode-length rollouts, or normalize by clip length)
    before ever un-zeroing it.
    """

    def __init__(self, lambda_max: float = 2.0, hidden: int = 32):
        super().__init__()
        self.lambda_max = lambda_max
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(
        self, belief: BeliefState, goal_entropy: torch.Tensor, min_sigma: torch.Tensor
    ) -> torch.Tensor:
        """-> (B,) lambda values. `min_sigma` (B,): running min of mean belief
        sigma seen so far this episode (see class docstring)."""
        x = torch.stack(
            [belief.sigma.mean(-1), belief.sigma.amax(-1), goal_entropy, min_sigma], dim=-1
        )
        return self.lambda_max * torch.sigmoid(self.net(x)).squeeze(-1)


class ActionPrior(nn.Module):
    """Behavior-cloned action prior: predicts the action a demo would take
    from the current state and goal.

    Found 08-14 (see docs/EXPERIMENT_LOG.md): pure random-noise CEM search
    (the planner's previous only mode) never converges to purposeful
    manipulation behavior within a real episode's step budget -- confirmed
    against literature as a known weak point of CEM-from-scratch planning
    for continuous control. This gives the planner a sensible action
    sequence to search *around* instead of pure noise centered on zero.
    """

    def __init__(
        self,
        feature_dim: int = C.deter_dim + C.stoch_dim,
        goal_dim: int = C.goal_embed_dim,
        action_dim: int = C.action_dim,
        hidden: int = 256,
    ):
        super().__init__()
        # Tried widening 256->512 + a third layer 2026-08-22 (see
        # docs/EXPERIMENT_LOG.md) to test whether bc_mse's early plateau
        # (converged by ~step 480 of 3510, never improved further) was a
        # capacity ceiling or a data ceiling (500 demos, 50/task). Held-out
        # bc_mse barely moved, but closed-loop distance got measurably
        # *worse* (baseline min 0.220->0.317, end 0.554->1.206) -- more
        # capacity on the same fixed dataset overfit harder to training-demo
        # idiosyncrasies and generalized worse to the off-distribution states
        # closed-loop rollout actually visits, the same distribution-shift
        # pattern behind the min_sigma and object_rel regressions earlier
        # this investigation. Reverted; the ceiling here is data, not
        # capacity -- don't re-widen this without more demos to go with it.
        self.net = nn.Sequential(
            nn.Linear(feature_dim + goal_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, feature: torch.Tensor, goal_embed: torch.Tensor) -> torch.Tensor:
        """(B, F), (B, G) -> (B, A) action in [-1, 1]."""
        return torch.tanh(self.net(torch.cat([feature, goal_embed], dim=-1)))


@dataclass
class CEMConfig:
    horizon: int = C.plan_horizon
    population: int = 64
    elites: int = 6
    iterations: int = 3
    # Tightened 0.5 -> 0.1 2026-08-22 (see docs/EXPERIMENT_LOG.md): found
    # that raw action_prior execution (bypassing CEM entirely) beat the
    # default-std CEM pipeline on both approach distance and end-of-episode
    # drift across 30 held-out episodes -- CEM's population noise was
    # wandering away from an already-good imitation trajectory more than it
    # was refining it. A tight search radius around the action_prior-seeded
    # mean recovered CEM's local-correction value without that cost: beat
    # both the wide-std baseline AND raw action_prior alone (30-episode
    # comparison: std=0.5 end=0.554-1.206 depending on run, action_prior_only
    # end=0.410, this end=0.389-0.695 depending on run -- noisy but
    # consistently the best or near-best config found this investigation).
    init_std: float = 0.1


class CEMPlanner:
    """Cross-entropy method over action sequences (no gradients at plan time)."""

    def __init__(self, config: CEMConfig | None = None):
        self.cfg = config or CEMConfig()

    def plan(
        self,
        score_fn,
        batch: int,
        device: torch.device,
        init_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """score_fn: (B, N, H, A) -> (B, N). Returns best first action (B, A).

        `init_mean` (B, H, A), if given, seeds the search around it instead
        of zero-mean noise -- see ActionPrior."""
        cfg = self.cfg
        mean = init_mean if init_mean is not None else torch.zeros(
            batch, cfg.horizon, C.action_dim, device=device
        )
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
        action_prior: ActionPrior | None = None,
        ig_samples: int = 1,
    ):
        super().__init__()
        self.world_model = world_model
        self.belief_net = belief_net
        self.utility = utility or UtilityHead()
        self.gate = gate or AdaptiveGate()
        self.planner = planner or CEMPlanner()
        # Opt-in, unlike utility/gate/planner: existing call sites that
        # construct DecisionModule() without one (tests, --smoke) keep
        # working unchanged, falling back to CEMPlanner's zero-mean search.
        self.action_prior = action_prior
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

    def prior_rollout(
        self, rssm_state: RSSMState, goal_embed: torch.Tensor, horizon: int
    ) -> torch.Tensor | None:
        """Autoregressively imagine forward using `action_prior`'s own
        predicted actions, producing a (B, H, A) sequence to seed CEM's
        search instead of starting from zero-mean noise. None if no
        action_prior was given (falls back to CEMPlanner's default)."""
        if self.action_prior is None:
            return None
        state = rssm_state
        actions = []
        for _ in range(horizon):
            a = self.action_prior(state.feature, goal_embed)
            actions.append(a)
            state = self.world_model.prior_step(state, a)
        return torch.stack(actions, dim=1)

    # ---- action selection ----

    @torch.no_grad()
    def select_action(
        self,
        belief: BeliefState,
        rssm_state: RSSMState,
        goal_embed: torch.Tensor,
        goal_entropy: torch.Tensor,
        min_sigma: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """MPC step: returns {action (B, A), lambda (B,), and diagnostics}.

        `min_sigma` (B,): running min of mean belief sigma so far this
        episode — see AdaptiveGate's docstring.
        """
        B = belief.mu.shape[0]
        device = belief.mu.device
        lam = self.gate(belief, goal_entropy, min_sigma)                    # (B,)
        init_mean = self.prior_rollout(rssm_state, goal_embed, self.planner.cfg.horizon)

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
            eu = self.expected_utility(state_n, flat, tile(goal_embed)).view(Bc, N)
            ig = self.information_gain(belief_n, state_n, flat).view(Bc, N)
            # eu and ig live on wildly different natural scales (found 08-14 —
            # eu varies by ~0.03-0.04 across a whole CEM population while ig
            # varies by ~7-8, roughly 250x larger), so lambda's exploration/
            # exploitation gating can't actually function: ig dominates the
            # sum regardless of lambda's trained value, breaking the
            # exploration decays-as-uncertainty-resolves mechanism the active
            # inference framing (Friston 2017) depends on. Z-score each
            # across the population (standard advantage-normalization-style
            # fix) so lambda controls a well-defined relative mix instead of
            # being drowned out by whichever term happens to have larger raw
            # magnitude. See docs/EXPERIMENT_LOG.md.
            eu = (eu - eu.mean(dim=1, keepdim=True)) / (eu.std(dim=1, keepdim=True) + 1e-6)
            ig = (ig - ig.mean(dim=1, keepdim=True)) / (ig.std(dim=1, keepdim=True) + 1e-6)
            return eu + lam.unsqueeze(1) * ig

        action = self.planner.plan(score, B, device, init_mean=init_mean)
        return {"action": action, "lambda": lam}
