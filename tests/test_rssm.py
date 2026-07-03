"""RSSM world model: filtering, imagination, ELBO, rollout metric."""

import torch

from via.contracts import C
from via.world_model import RSSM

B, T, H = 2, 8, 5


def _traj():
    return torch.randn(B, T, C.obs_embed_dim), torch.randn(B, T - 1, C.action_dim)


def test_observe_contract():
    rssm = RSSM()
    obs, actions = _traj()
    posts, priors = rssm.observe(obs, actions)
    assert len(posts) == len(priors) == T - 1
    assert posts[0].deter.shape == (B, C.deter_dim)
    assert posts[0].stoch.shape == (B, C.stoch_dim)
    assert posts[0].feature.shape == (B, C.deter_dim + C.stoch_dim)


def test_imagine_contract():
    rssm = RSSM()
    state = rssm.init_state(B)
    states = rssm.imagine(state, torch.randn(B, H, C.action_dim))
    assert len(states) == H
    assert rssm.decode(states[-1]).shape == (B, C.obs_embed_dim)


def test_loss_backward():
    rssm = RSSM()
    obs, actions = _traj()
    losses = rssm.loss(obs, actions)
    losses["loss"].backward()
    grads = [p.grad for p in rssm.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_rollout_mse_returns_both_baselines():
    rssm = RSSM()
    obs, actions = _traj()
    out = rssm.rollout_mse(obs, actions, k=5)
    assert out["model_mse"] > 0 and out["naive_mse"] > 0


def test_training_reduces_loss():
    rssm = RSSM()
    opt = torch.optim.Adam(rssm.parameters(), lr=1e-3)
    obs, actions = _traj()
    first = None
    for _ in range(5):
        losses = rssm.loss(obs, actions)
        if first is None:
            first = losses["loss"].item()
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
    assert losses["loss"].item() < first
