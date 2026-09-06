"""Action-chunking policy: shape contract and trainability."""

import torch

from via.contracts import C
from via.decision import ActionChunkingPolicy

B = 2
K = 8


def test_output_shape():
    policy = ActionChunkingPolicy(chunk_size=K)
    feature = torch.randn(B, C.deter_dim + C.stoch_dim)
    goal_embed = torch.randn(B, C.goal_embed_dim)
    actions = policy(feature, goal_embed)
    assert actions.shape == (B, K, C.action_dim)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


def test_training_reduces_loss():
    policy = ActionChunkingPolicy(chunk_size=K)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    feature = torch.randn(B, C.deter_dim + C.stoch_dim)
    goal_embed = torch.randn(B, C.goal_embed_dim)
    target = torch.rand(B, K, C.action_dim) * 2 - 1
    first = None
    for _ in range(10):
        pred = policy(feature, goal_embed)
        loss = torch.nn.functional.mse_loss(pred, target)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first
