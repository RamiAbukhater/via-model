"""Tensor contracts for the VIA model.

Every inter-module boundary is pinned here. Modules import these constants
instead of hard-coding dimensions, and tests assert against them, so a shape
change is a one-line edit that the test suite immediately re-validates.

Shape notation used in docstrings across the codebase:
    B = batch, T = time, P = patches, L = language tokens,
    K = goal slots, H = planning horizon, N = candidate action sequences
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Contracts:
    # ---- Perception (frozen SigLIP ViT-B/16, 224x224, 16x16 patches) ----
    image_size: int = 224
    patch_count: int = 196          # (224/16)^2
    patch_dim: int = 768            # ViT-B hidden size

    # ---- Belief state ----
    obs_embed_dim: int = 256        # pooled observation embedding fed to the GRU
    belief_dim: int = 256           # latent world state; belief is N(mu, diag(sigma^2))
    belief_hidden_dim: int = 512    # GRU hidden size

    # ---- Proprioception (fused into obs_embed alongside vision; see
    # docs/EXPERIMENT_LOG.md 2026-08-16 -- the pooled visual embedding alone
    # has no dedicated channel for precise end-effector position or gripper
    # open/close state, both of which live cheaply and precisely in
    # robosuite/LIBERO's own observations) ----
    # end-effector position (3) + gripper finger joint positions (2).
    # Orientation deliberately left out of v1: the offline demo HDF5s store
    # a 3-dim `ee_ori` of unconfirmed convention while the live sim exposes
    # a 4-dim quaternion (`robot0_eef_quat`) -- reconciling those risks a
    # silent train/inference mismatch, exactly the failure mode this project
    # has already been bitten by twice (obs_embed scale, eu/ig scale).
    #
    # Extended 2026-08-21 with target-object-relative position (3): robot-only
    # proprioception measurably recovered a regression but didn't move
    # closed-loop success on its own (see EXPERIMENT_LOG.md) -- it tells the
    # model where its own gripper is, not where the target object is, which
    # a coarse pooled visual embedding struggles to localize precisely.
    # Sourced from `<object>_to_robot0_eef_pos` (live sim, via
    # `obj_of_interest`) and, offline, from a one-time physics-state-replay
    # preprocessing pass over the demo files (scripts/extract_object_state.py)
    # since the raw demo HDF5s don't label object identity/position directly.
    proprio_dim: int = 8

    # ---- Language / goal inference ----
    lang_dim: int = 3072            # Phi-3-mini-4k-instruct hidden size
    goal_slots: int = 32            # K candidate goal prototypes
    goal_embed_dim: int = 256

    # ---- World model (RSSM) ----
    deter_dim: int = 256            # deterministic recurrent state
    stoch_dim: int = 64             # stochastic latent

    # ---- Action space (LIBERO / robosuite OSC: 6-DoF delta pose + gripper) ----
    action_dim: int = 7
    action_low: float = -1.0
    action_high: float = 1.0

    # ---- Planning ----
    plan_horizon: int = 5


C = Contracts()


def assert_shape(t: torch.Tensor, expected: tuple, name: str = "tensor") -> None:
    """Assert a tensor's shape, with -1 as a wildcard dimension."""
    if t.dim() != len(expected):
        raise AssertionError(
            f"{name}: expected {len(expected)} dims {expected}, got shape {tuple(t.shape)}"
        )
    for i, (got, want) in enumerate(zip(t.shape, expected)):
        if want != -1 and got != want:
            raise AssertionError(
                f"{name}: dim {i} expected {want}, got {got} (full shape {tuple(t.shape)})"
            )
