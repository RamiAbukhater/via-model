"""Decision module: objective terms, adaptive gate, CEM planning."""

import torch

from via.belief import BeliefStateNetwork
from via.contracts import C
from via.decision import AdaptiveGate, DecisionModule, UtilityHead
from via.decision.decision import CEMConfig, CEMPlanner
from via.world_model import RSSM

B = 2


def _module():
    rssm = RSSM()
    belief_net = BeliefStateNetwork()
    planner = CEMPlanner(CEMConfig(population=16, elites=4, iterations=2))
    return DecisionModule(rssm, belief_net, planner=planner), rssm, belief_net


def _belief(belief_net):
    return belief_net.step(torch.randn(B, C.obs_embed_dim), belief_net.init_hidden(B))


def test_utility_head_shape():
    head = UtilityHead()
    u = head(torch.randn(B, C.deter_dim + C.stoch_dim), torch.randn(B, C.goal_embed_dim))
    assert u.shape == (B,)


def test_gate_range():
    gate = AdaptiveGate(lambda_max=2.0)
    net = BeliefStateNetwork()
    lam = gate(_belief(net), torch.rand(B))
    assert lam.shape == (B,)
    assert torch.all(lam >= 0) and torch.all(lam <= 2.0)


def test_information_gain_shape():
    dm, rssm, belief_net = _module()
    ig = dm.information_gain(
        _belief(belief_net), rssm.init_state(B), torch.randn(B, 3, C.action_dim)
    )
    assert ig.shape == (B,)


def test_expected_utility_shape():
    dm, rssm, _ = _module()
    eu = dm.expected_utility(
        rssm.init_state(B), torch.randn(B, 3, C.action_dim), torch.randn(B, C.goal_embed_dim)
    )
    assert eu.shape == (B,)


def test_select_action_contract():
    dm, rssm, belief_net = _module()
    out = dm.select_action(
        _belief(belief_net), rssm.init_state(B), torch.randn(B, C.goal_embed_dim), torch.rand(B)
    )
    a = out["action"]
    assert a.shape == (B, C.action_dim)
    assert torch.all(a >= C.action_low) and torch.all(a <= C.action_high)
    assert out["lambda"].shape == (B,)


def test_cem_finds_known_optimum():
    """CEM should push the first action toward a planted optimum."""
    planner = CEMPlanner(CEMConfig(population=64, elites=6, iterations=4))
    target = torch.full((C.action_dim,), 0.8)

    def score(cands):  # (B, N, H, A) -> (B, N): reward closeness to target at t=0
        return -((cands[:, :, 0] - target) ** 2).sum(-1)

    a = planner.plan(score, batch=1, device=torch.device("cpu"))
    assert ((a[0] - target).abs() < 0.25).all()
