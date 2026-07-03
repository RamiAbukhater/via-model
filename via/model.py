"""VIAModel: the full Vision-Inference-Action pipeline.

Per control step:
    1. Perception:      frame -> frozen patch embeddings (evidence)
    2. Belief update:   patches -> pooled obs embed -> GRU -> N(mu, sigma)
    3. Goal inference:  instruction + belief -> P(goal | u, b) via RSA
    4. World filtering: obs embed + last action -> RSSM posterior
    5. Decision:        CEM over J(a) = E[U] + lambda(sigma) * IG(a)

Every module hands a *distribution* downstream; `act()` also returns the
diagnostics (belief entropy, goal entropy, lambda) used for the uncertainty
analyses in the proposal.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from via.belief.belief_state import BeliefState, BeliefStateNetwork
from via.contracts import C
from via.decision.decision import DecisionModule
from via.goal.rsa import GoalDistribution, GoalInferenceRSA
from via.world_model.rssm import RSSM, RSSMState


@dataclass
class EpisodeState:
    """Everything the agent carries between control steps."""

    belief: BeliefState
    rssm: RSSMState
    goal: Optional[GoalDistribution]
    last_action: torch.Tensor
    lang_tokens: torch.Tensor
    lang_mask: torch.Tensor


class VIAModel(nn.Module):
    def __init__(
        self,
        perception: nn.Module,
        language: nn.Module,
        belief_net: Optional[BeliefStateNetwork] = None,
        goal_net: Optional[GoalInferenceRSA] = None,
        world_model: Optional[RSSM] = None,
        decision: Optional[DecisionModule] = None,
        reinfer_goal_every: int = 1,
    ):
        super().__init__()
        self.perception = perception
        self.language = language
        self.belief_net = belief_net or BeliefStateNetwork()
        self.goal_net = goal_net or GoalInferenceRSA()
        self.world_model = world_model or RSSM()
        self.decision = decision or DecisionModule(self.world_model, self.belief_net)
        self.reinfer_goal_every = reinfer_goal_every
        self._step_count = 0

    def reset(self, instructions: list[str], device: Optional[torch.device] = None) -> EpisodeState:
        """Start an episode: encode the instruction once, zero all carries."""
        B = len(instructions)
        lang_tokens, lang_mask = self.language(instructions)
        device = device or lang_tokens.device
        self._step_count = 0
        return EpisodeState(
            belief=BeliefState(
                mu=torch.zeros(B, C.belief_dim, device=device),
                logvar=torch.zeros(B, C.belief_dim, device=device),
                hidden=self.belief_net.init_hidden(B, device),
            ),
            rssm=self.world_model.init_state(B, device),
            goal=None,
            last_action=torch.zeros(B, C.action_dim, device=device),
            lang_tokens=lang_tokens.to(device),
            lang_mask=lang_mask.to(device),
        )

    @torch.no_grad()
    def act(self, frame: torch.Tensor, state: EpisodeState) -> tuple[torch.Tensor, EpisodeState, dict]:
        """One control step. frame (B, 3, 224, 224) in [0,1] -> action (B, 7).

        Returns (action, new_state, diagnostics).
        """
        patches = self.perception(frame.unsqueeze(1))[:, 0]        # (B, P, 768)
        obs_embed = self.belief_net.encode_obs(patches)            # (B, D)

        belief = self.belief_net.step(obs_embed, state.belief.hidden)
        rssm, _ = self.world_model.posterior_step(state.rssm, state.last_action, obs_embed)

        # Goal re-inference: the belief conditions the goal, so as the scene
        # disambiguates, the goal distribution narrows.
        goal = state.goal
        if goal is None or self._step_count % self.reinfer_goal_every == 0:
            goal = self.goal_net(state.lang_tokens, state.lang_mask, belief.mu)

        out = self.decision.select_action(belief, rssm, goal.embedding, goal.entropy())
        action = out["action"]
        self._step_count += 1

        new_state = EpisodeState(
            belief=belief, rssm=rssm, goal=goal, last_action=action,
            lang_tokens=state.lang_tokens, lang_mask=state.lang_mask,
        )
        diagnostics = {
            "belief_entropy": belief.entropy(),
            "belief_sigma_mean": belief.sigma.mean(-1),
            "goal_entropy": goal.entropy(),
            "goal_probs": goal.probs,
            "lambda": out["lambda"],
        }
        return action, new_state, diagnostics
