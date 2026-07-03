"""RSA goal inference: contracts, calibration mechanics, RSA training loss."""

import torch

from via.contracts import C
from via.goal import GoalInferenceRSA

B, L = 4, 10


def _inputs(language=None):
    if language is not None:
        tokens, mask = language(["pick up the red block"] * B)
    else:
        tokens = torch.randn(B, L, C.lang_dim)
        mask = torch.ones(B, L, dtype=torch.bool)
    belief_mu = torch.randn(tokens.shape[0], C.belief_dim)
    return tokens, mask, belief_mu


def test_goal_distribution_contract():
    net = GoalInferenceRSA()
    dist = net(*_inputs())
    assert dist.probs.shape == (B, C.goal_slots)
    assert dist.embedding.shape == (B, C.goal_embed_dim)
    assert torch.allclose(dist.probs.sum(-1), torch.ones(B), atol=1e-5)
    assert dist.entropy().shape == (B,)


def test_entropy_bounds():
    net = GoalInferenceRSA()
    ent = net(*_inputs()).entropy()
    max_ent = torch.log(torch.tensor(float(C.goal_slots)))
    assert torch.all(ent >= 0) and torch.all(ent <= max_ent + 1e-4)


def test_works_with_stub_language(language):
    net = GoalInferenceRSA()
    dist = net(*_inputs(language))
    assert dist.probs.shape == (B, C.goal_slots)


def test_rsa_training_loss_backward():
    net = GoalInferenceRSA()
    tokens, mask, belief_mu = _inputs()
    targets = torch.randint(0, C.goal_slots, (B,))
    out = net.rsa_training_loss(tokens, mask, belief_mu, targets)
    out["loss"].backward()
    assert net.goal_prototypes.grad is not None
    assert net.raw_alpha.grad is not None


def test_training_learns_labels():
    """A few steps of the RSA loss should beat chance on a fixed batch."""
    net = GoalInferenceRSA()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    tokens, mask, belief_mu = _inputs()
    targets = torch.arange(B) % C.goal_slots
    for _ in range(60):
        out = net.rsa_training_loss(tokens, mask, belief_mu, targets)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    assert out["accuracy"] >= 0.75
