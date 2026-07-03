"""End-to-end forward pass — the Foundation milestone test.

Full VIA pipeline on stub encoders: reset an episode, step it for several
frames, and check every contract plus the uncertainty diagnostics.
"""

import torch

from via.contracts import C
from via.decision.decision import CEMConfig, CEMPlanner, DecisionModule
from via.model import VIAModel

B, STEPS = 2, 3


def _model(perception, language):
    m = VIAModel(perception=perception, language=language)
    # Small planner so the test runs in seconds on CPU.
    m.decision = DecisionModule(
        m.world_model, m.belief_net, planner=CEMPlanner(CEMConfig(population=8, elites=2, iterations=1))
    )
    return m.eval()


def test_full_pipeline_forward(perception, language):
    model = _model(perception, language)
    state = model.reset(["pick up the red block", "push the bowl"])
    for _ in range(STEPS):
        frame = torch.rand(B, 3, C.image_size, C.image_size)
        action, state, diag = model.act(frame, state)
        assert action.shape == (B, C.action_dim)
        assert torch.isfinite(action).all()
        assert diag["belief_entropy"].shape == (B,)
        assert diag["goal_entropy"].shape == (B,)
        assert diag["goal_probs"].shape == (B, C.goal_slots)
        assert diag["lambda"].shape == (B,)
        assert torch.all(diag["lambda"] >= 0)


def test_belief_evolves_across_steps(perception, language):
    model = _model(perception, language)
    state = model.reset(["open the drawer"])
    frame = torch.rand(1, 3, C.image_size, C.image_size)
    _, s1, _ = model.act(frame, state)
    _, s2, _ = model.act(frame, s1)
    assert not torch.allclose(s1.belief.mu, s2.belief.mu)


def test_synthetic_datasets_contract():
    from via.data.synthetic import SyntheticInstructionDataset, SyntheticTrajectoryDataset

    traj = SyntheticTrajectoryDataset(size=2, clip_len=6)[0]
    assert traj["frames"].shape == (6, 3, C.image_size, C.image_size)
    assert traj["actions"].shape == (6, C.action_dim)
    assert traj["occluded"].any()

    instr = SyntheticInstructionDataset(size=8)[0]
    assert 0 <= instr["goal"] < C.goal_slots
    assert instr["ambiguity"] in (0, 1, 2)


def test_perception_contract(perception):
    imgs = torch.rand(B, 2, 3, C.image_size, C.image_size)
    patches = perception(imgs)
    assert patches.shape == (B, 2, C.patch_count, C.patch_dim)
    # Occlusion must change the embedding (belief tests depend on this).
    blank = torch.zeros_like(imgs)
    assert not torch.allclose(perception(blank), patches)
