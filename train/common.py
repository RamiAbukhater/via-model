"""Shared training utilities: config, seeding, encoder factories, W&B, checkpoints."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    # Expand ~ and $VARS so the same config works across clusters/users.
    for section, key in (("data", "libero_dir"), ("checkpoints", "dir")):
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


def build_trajectory_dataset(cfg: dict, smoke: bool = False):
    from via.data.synthetic import SyntheticTrajectoryDataset

    if smoke or cfg["data"]["source"] == "synthetic":
        n = 16 if smoke else cfg["data"]["synthetic_size"]
        return SyntheticTrajectoryDataset(size=n, clip_len=cfg["data"]["clip_len"])
    from via.data.libero import LiberoTrajectoryDataset

    return LiberoTrajectoryDataset(
        cfg["data"]["libero_dir"],
        clip_len=cfg["data"]["clip_len"],
        suite_filter=cfg["data"].get("suite"),
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
