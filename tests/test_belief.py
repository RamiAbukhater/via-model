"""Belief state network: contracts, uncertainty behavior, trainability."""

import torch

from via.belief import BeliefStateNetwork
from via.contracts import C

B, T = 2, 5


def _patches(B=B, T=T):
    return torch.randn(B, T, C.patch_count, C.patch_dim)


def _proprio(B=B, T=T):
    return torch.randn(B, T, C.proprio_dim)


def test_encode_obs_shape():
    net = BeliefStateNetwork()
    out = net.encode_obs(_patches()[:, 0], _proprio()[:, 0])
    assert out.shape == (B, C.obs_embed_dim)


def test_step_contract():
    net = BeliefStateNetwork()
    b = net.step(torch.randn(B, C.obs_embed_dim), net.init_hidden(B))
    assert b.mu.shape == (B, C.belief_dim)
    assert b.logvar.shape == (B, C.belief_dim)
    assert torch.all(b.sigma > 0)


def test_rollout_length_and_entropy():
    net = BeliefStateNetwork()
    beliefs = net.rollout(_patches(), _proprio())
    assert len(beliefs) == T
    assert beliefs[0].entropy().shape == (B,)


def test_sample_shape():
    net = BeliefStateNetwork()
    b = net.rollout(_patches(), _proprio())[-1]
    assert b.sample(7).shape == (7, B, C.belief_dim)


def test_loss_backward_updates_params():
    net = BeliefStateNetwork()
    losses = net.loss(_patches(), _proprio())
    losses["loss"].backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_training_reduces_loss():
    """Three optimizer steps on a fixed batch should reduce the objective."""
    net = BeliefStateNetwork()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    patches = _patches()
    proprio = _proprio()
    first = None
    for _ in range(5):
        losses = net.loss(patches, proprio)
        if first is None:
            first = losses["loss"].item()
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
    assert losses["loss"].item() < first
