"""Shared training utilities: config, seeding, encoder factories, W&B, checkpoints."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    # Expand ~ and $VARS so the same config works across clusters/users.
    for section, key in (
        ("data", "libero_dir"), ("data", "object_state_dir"), ("data", "augment_dir"),
        ("checkpoints", "dir"),
    ):
        if key in cfg[section]:
            cfg[section][key] = os.path.expandvars(os.path.expanduser(cfg[section][key]))
    return cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny synthetic run on stub encoders (CPU-friendly CI check)")
    p.add_argument("--resume", action="store_true",
                   help="warm-start from an existing checkpoint if one exists "
                        "(for clusters with session time limits, e.g. DSMLP)")
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_perception(cfg: dict, smoke: bool = False) -> torch.nn.Module:
    from via.perception import SigLIPPerception, StubPerception

    if smoke or cfg["encoders"]["vision"] == "stub":
        return StubPerception()
    return SigLIPPerception(cfg["encoders"]["vision"])


def build_language(cfg: dict, smoke: bool = False) -> torch.nn.Module:
    from via.goal import Phi3LanguageEncoder, StubLanguageEncoder

    if smoke or cfg["encoders"]["language"] == "stub":
        return StubLanguageEncoder()
    return Phi3LanguageEncoder(cfg["encoders"]["language"])


def build_trajectory_dataset(cfg: dict, smoke: bool = False, split: str = "train"):
    """`split` only affects real LIBERO data (see LiberoTrajectoryDataset) —
    default "train" so every training script is held-out-safe by default
    without each call site having to remember to ask for it. Synthetic data
    is procedurally generated per-seed, not drawn from a finite pool, so a
    different seed (as eval/uncertainty_analysis.py's held-out seed=999
    already does vs. this seed=0 stream) is already genuinely unseen."""
    from via.data.synthetic import SyntheticTrajectoryDataset

    if smoke or cfg["data"]["source"] == "synthetic":
        n = 16 if smoke else cfg["data"]["synthetic_size"]
        return SyntheticTrajectoryDataset(size=n, clip_len=cfg["data"]["clip_len"])
    from via.data.libero import LiberoTrajectoryDataset

    return LiberoTrajectoryDataset(
        cfg["data"]["libero_dir"],
        clip_len=cfg["data"]["clip_len"],
        suite_filter=cfg["data"].get("suite"),
        object_state_dir=cfg["data"].get("object_state_dir"),
        augment_dir=cfg["data"].get("augment_dir"),
        split=split,
    )


def init_wandb(cfg: dict, run_name: str, disabled: bool):
    if disabled:
        return None
    try:
        import wandb

        return wandb.init(project=cfg["wandb"]["project"], name=run_name, config=cfg)
    except Exception as e:  # offline dev boxes: log locally and move on
        print(f"[warn] wandb unavailable ({e}); continuing without tracking")
        return None


def log_metrics(run, metrics: dict, step: int) -> None:
    scalars = {k: float(v) for k, v in metrics.items()}
    if run is not None:
        run.log(scalars, step=step)
    if step % 10 == 0:
        line = " ".join(f"{k}={v:.4f}" for k, v in scalars.items())
        print(f"step {step:6d} | {line}")


def checkpoint_path(cfg: dict, name: str) -> Path:
    d = REPO_ROOT / cfg["checkpoints"]["dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.pt"


def save_checkpoint(module: torch.nn.Module, cfg: dict, name: str) -> Path:
    path = checkpoint_path(cfg, name)
    torch.save(module.state_dict(), path)
    print(f"saved {name} -> {path}")
    return path


def load_checkpoint(module: torch.nn.Module, cfg: dict, name: str, device: str) -> torch.nn.Module:
    path = checkpoint_path(cfg, name)
    module.load_state_dict(torch.load(path, map_location=device))
    return module


def try_resume(module: torch.nn.Module, cfg: dict, name: str, device: str, enabled: bool) -> None:
    """Warm-start `module` from its checkpoint when --resume is set and one exists."""
    if not enabled:
        return
    path = checkpoint_path(cfg, name)
    if path.exists():
        module.load_state_dict(torch.load(path, map_location=device))
        print(f"resumed {name} from {path}")


@torch.no_grad()
def encode_frames(perception, frames: torch.Tensor, batch: int = 8) -> torch.Tensor:
    """(B, T, 3, H, W) -> (B, T, P, patch_dim), chunked to bound memory."""
    outs = [perception(frames[i : i + batch]) for i in range(0, frames.shape[0], batch)]
    return torch.cat(outs, dim=0)


@torch.no_grad()
def world_model_holdout_check(
    cfg: dict,
    perception,
    belief_net,
    rssm,
    device,
    n_clips: int = 64,
    k: int = 5,
    seed: int = 999,
    n_draws: int = 5,
) -> dict:
    """Held-out win-rate + mean MSE for the world model's Checkpoint-1 metric
    (rollout5_model_mse vs rollout5_naive_mse).

    Two failure modes an earlier version of this check missed, both found
    08-13 (see docs/EXPERIMENT_LOG.md):

    1. `rollout_mse`'s imagined rollout samples stochastically
       (`RSSM.prior_step`), so a single call's win-rate has real sampling
       noise independent of which clips were selected — the same checkpoint
       evaluated twice gives different numbers. `n_draws` averages per-clip
       model_mse over multiple stochastic rollouts before comparing (naive
       is deterministic, only drawn once).
    2. The naive "persistence" baseline (predict obs_{t+k} = obs_t) is
       near-unbeatable *by construction* on clips with little real motion in
       the window — no model can score much below its already-tiny error.
       An aggregate win-rate over a held-out set with a lot of near-static
       clips is dominated by this floor effect rather than reflecting
       whether the model learned real dynamics. Splitting by naive_mse
       (a cheap motion proxy: low naive_mse == little happened) into
       low/high-motion halves surfaces this — see 08-13 entry, where overall
       win-rate sat at chance (~51%) while low-motion was 9% (expected —
       naive genuinely is close to optimal there) and high-motion was 94%
       (the number that actually reflects whether the model learned
       anything). Report both; the aggregate alone is not trustworthy
       validation for this data.

    Real LIBERO uses the val split (LiberoTrajectoryDataset split="val",
    held out by demo — see via/data/libero.py) so this is genuinely unseen
    data, not just a different random draw from the training pool. Synthetic
    uses a fixed seed distinct from the training stream's seed=0.
    """
    from via.data.synthetic import SyntheticTrajectoryDataset

    if cfg["data"]["source"] == "libero":
        from via.data.libero import LiberoTrajectoryDataset

        ds = LiberoTrajectoryDataset(
            cfg["data"]["libero_dir"],
            clip_len=cfg["data"]["clip_len"],
            suite_filter=cfg["data"].get("suite"),
            object_state_dir=cfg["data"].get("object_state_dir"),
            augment_dir=cfg["data"].get("augment_dir"),
            split="val",
        )
    else:
        ds = SyntheticTrajectoryDataset(
            size=cfg["data"]["synthetic_size"], clip_len=cfg["data"]["clip_len"], seed=seed
        )
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[: min(n_clips, len(ds))]
    frames = torch.stack([ds[i]["frames"] for i in idx.tolist()]).to(device)
    actions = torch.stack([ds[i]["actions"] for i in idx.tolist()]).to(device)
    proprio = torch.stack([ds[i]["proprio"] for i in idx.tolist()]).to(device)
    patches = encode_frames(perception, frames)
    T = patches.shape[1]
    obs_embeds = torch.stack(
        [belief_net.encode_obs(patches[:, t], proprio[:, t]) for t in range(T)], dim=1
    )

    model_accum = torch.zeros(len(idx), device=device)
    naive_errs = None
    for _ in range(n_draws):
        roll = rssm.rollout_mse(obs_embeds, actions[:, :-1], k=k, reduce=False)
        model_accum += roll["model_mse"]
        naive_errs = roll["naive_mse"]  # deterministic, identical every draw
    model_errs = model_accum / n_draws

    win = model_errs < naive_errs
    low_motion = naive_errs <= naive_errs.median()
    return {
        "holdout_model_mse": model_errs.mean().item(),
        "holdout_naive_mse": naive_errs.mean().item(),
        "holdout_win_rate": win.float().mean().item(),
        "holdout_win_rate_low_motion": win[low_motion].float().mean().item(),
        "holdout_win_rate_high_motion": win[~low_motion].float().mean().item(),
        "holdout_n_clips": len(idx),
    }


@torch.no_grad()
def goal_holdout_check(
    cfg: dict, perception, belief_net, language, goal_net, device, n_clips: int = 64, seed: int = 999
) -> dict:
    """Held-out accuracy/entropy for real-LIBERO goal-inference fine-tuning
    (train/train_goal.py's libero path). split="val" — demos not used in
    fine-tuning, but note the *task labels* themselves (10 for
    libero_spatial) are still all seen during training; this tests
    generalization to new demo instances of known tasks, not to unseen
    tasks/instructions (libero_spatial doesn't have held-out tasks to test
    that with)."""
    from via.data.libero import LiberoTrajectoryDataset

    ds = LiberoTrajectoryDataset(
        cfg["data"]["libero_dir"],
        clip_len=cfg["data"]["clip_len"],
        suite_filter=cfg["data"].get("suite"),
        object_state_dir=cfg["data"].get("object_state_dir"),
        augment_dir=cfg["data"].get("augment_dir"),
        split="val",
    )
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[: min(n_clips, len(ds))]
    items = [ds[i] for i in idx.tolist()]
    frames = torch.stack([it["frames"] for it in items]).to(device)
    proprio = torch.stack([it["proprio"] for it in items]).to(device)
    task_id = torch.tensor([it["task_id"] for it in items], device=device)
    instructions = [it["instruction"] for it in items]

    patches = encode_frames(perception, frames)
    beliefs = belief_net.rollout(patches, proprio)
    tokens, mask = language(instructions)
    dist = goal_net(tokens.to(device), mask.to(device), beliefs[-1].mu)
    return {
        "goal_holdout_accuracy": (dist.probs.argmax(-1) == task_id).float().mean().item(),
        "goal_holdout_entropy": dist.entropy().mean().item(),
        "goal_holdout_n": len(idx),
    }


@torch.no_grad()
def decision_holdout_check(
    cfg: dict,
    perception,
    belief_net,
    rssm,
    goal_net,
    language,
    utility,
    gate,
    device,
    action_prior=None,
    n_clips: int = 64,
    n_draws: int = 5,
    seed: int = 999,
) -> dict:
    """Held-out check for both decision heads, split="val", real LIBERO data.

    Utility: `RSSM.observe`'s posterior_step samples stochastically (same
    issue as world_model_holdout_check found 08-13), so its MSE is averaged
    over `n_draws`. `belief_net.rollout` and the gate are deterministic
    (mu/logvar directly, no sampling) — no averaging needed there.

    Gate: not just "does lambda correlate with sigma in aggregate" — per
    the belief-calibration lesson (pooling hides real per-clip signal, see
    08-13), blank a window mid-clip and check *per clip* whether lambda
    rises during that window relative to that same clip's own visible
    baseline, then report the win-rate across clips.
    """
    from via.data.libero import LiberoTrajectoryDataset

    ds = LiberoTrajectoryDataset(
        cfg["data"]["libero_dir"],
        clip_len=cfg["data"]["clip_len"],
        suite_filter=cfg["data"].get("suite"),
        object_state_dir=cfg["data"].get("object_state_dir"),
        augment_dir=cfg["data"].get("augment_dir"),
        split="val",
    )
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[: min(n_clips, len(ds))]
    items = [ds[i] for i in idx.tolist()]
    frames = torch.stack([it["frames"] for it in items]).to(device)
    actions = torch.stack([it["actions"] for it in items]).to(device)
    reward = torch.stack([it["reward"] for it in items]).to(device)
    proprio = torch.stack([it["proprio"] for it in items]).to(device)
    instructions = [it["instruction"] for it in items]
    B = len(items)

    def pipeline(frames_in):
        # proprio is unaffected by the visual occlusion mask below (blanking
        # pixels doesn't blank the robot's own proprioception) — same
        # `proprio` used for both the clean and occluded calls.
        patches = encode_frames(perception, frames_in)
        obs_embeds = torch.stack(
            [belief_net.encode_obs(patches[:, t], proprio[:, t]) for t in range(patches.shape[1])], dim=1
        )
        beliefs = belief_net.rollout(patches, proprio)
        posts, _ = rssm.observe(obs_embeds, actions[:, :-1])
        tokens, mask = language(instructions)
        goal = goal_net(tokens.to(device), mask.to(device), beliefs[-1].mu)
        return beliefs, posts, goal

    # Utility: TD(0) bootstrap MSE, same formula as train/train_decision.py
    # (see its module docstring) -- averaged over stochastic draws (RSSM
    # posterior sampling).
    gamma = cfg["decision"].get("value_gamma", 0.95)
    r = reward[:, 1:]
    u_mse_accum = 0.0
    beliefs = goal = None
    for _ in range(n_draws):
        beliefs, posts, goal = pipeline(frames)
        feats = torch.stack([p.feature for p in posts], dim=1)
        goal_rep = goal.embedding.unsqueeze(1).expand(-1, feats.shape[1], -1)
        pred = utility(feats.flatten(0, 1), goal_rep.flatten(0, 1)).view(B, -1)
        v_next = pred[:, 1:]
        bootstrap_target = r[:, :-1] + gamma * v_next
        target = torch.cat([bootstrap_target, r[:, -1:]], dim=1)
        valid_mask = torch.ones_like(r)
        valid_mask[:, -1] = (r[:, -1] > 0.5).float()
        u_mse_accum += (
            (F.mse_loss(pred, target, reduction="none") * valid_mask).sum()
            / valid_mask.sum().clamp(min=1)
        ).item()
    u_mse = u_mse_accum / n_draws

    bc_mse = None
    if action_prior is not None:
        # Deterministic (no sampling in belief_net.rollout or posts[0], the
        # first posterior step, which is what action_prior conditions on at
        # inference time via prior_rollout's first call) — no draws needed.
        feats = torch.stack([p.feature for p in posts], dim=1)
        goal_rep = goal.embedding.unsqueeze(1).expand(-1, feats.shape[1], -1)
        action_pred = action_prior(feats.flatten(0, 1), goal_rep.flatten(0, 1)).view(
            B, feats.shape[1], -1
        )
        bc_mse = F.mse_loss(action_pred, actions[:, 1:]).item()

    # Gate: per-clip occlusion response (deterministic — no draws needed).
    T = frames.shape[1]
    start, length = T // 2 - 1, 3
    occ_mask = torch.zeros(T, dtype=torch.bool)
    occ_mask[start : start + length] = True
    occ_frames = frames.clone()
    occ_frames[:, occ_mask] = 0.0
    beliefs_occ, _, goal_occ = pipeline(occ_frames)
    sig_occ = torch.stack([b.sigma.mean(-1) for b in beliefs_occ], dim=1)      # (B, T)
    # Neutralized to a constant -- see AdaptiveGate's docstring
    # (via/decision/decision.py).
    min_sig_occ = torch.zeros_like(sig_occ)
    lam = torch.stack(
        [gate(beliefs_occ[t], goal_occ.entropy(), min_sig_occ[:, t]) for t in range(T)], dim=1
    )  # (B, T)
    lam_during = lam[:, occ_mask].mean(dim=1)
    lam_visible = lam[:, ~occ_mask].mean(dim=1)
    gate_win = (lam_during > lam_visible).float().mean().item()

    result = {
        "holdout_utility_mse": u_mse,
        "holdout_gate_win_rate": gate_win,
        "holdout_mean_lambda_occluded": lam_during.mean().item(),
        "holdout_mean_lambda_visible": lam_visible.mean().item(),
        "holdout_n_clips": B,
    }
    if bc_mse is not None:
        result["holdout_bc_mse"] = bc_mse
    return result
