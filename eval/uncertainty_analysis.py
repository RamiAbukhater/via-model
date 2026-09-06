"""Quantitative uncertainty analysis of the belief state network.

The Checkpoint-1 deliverable: does belief sigma track observability?
Runs the trained belief network over held-out clips with occlusion windows
and reports the occluded-vs-visible sigma ratio, plus a per-timestep trace
figure showing sigma spiking when the scene goes dark.

    python -m eval.uncertainty_analysis                 # trained ckpt + config data
    python -m eval.uncertainty_analysis --smoke         # untrained, synthetic, CPU

Outputs: results/uncertainty.csv, results/uncertainty.png (if matplotlib).
"""

import csv

import torch
from torch.utils.data import DataLoader

from train import common
from via.belief import BeliefStateNetwork
from via.data.synthetic import SyntheticTrajectoryDataset


def libero_occlusion_clips(cfg: dict, n_clips: int):
    """LIBERO clips with a 3-frame occlusion window blanked mid-clip.

    In-domain occlusion probe: the belief net trained on LIBERO frames, so
    the sigma comparison is between visible and blanked frames of the same
    distribution (a domain-matched version of the synthetic occlusion test).

    split="val" (via/data/libero.py, held out by demo): a fixed seed alone
    isn't enough here — unlike synthetic data, LIBERO is a finite pool, and
    training draws from the same pool by default, so a "held-out seed" was
    previously just a different random sample of clips the model had almost
    certainly already seen (see docs/EXPERIMENT_LOG.md 08-13). The split
    keeps eval demos entirely out of the training set.
    """
    from via.data.libero import LiberoTrajectoryDataset

    ds = LiberoTrajectoryDataset(
        cfg["data"]["libero_dir"],
        clip_len=cfg["data"]["clip_len"],
        suite_filter=cfg["data"].get("suite"),
        object_state_dir=cfg["data"].get("object_state_dir"),
        split="val",
    )
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(999))[:n_clips]
    for i in idx.tolist():
        item = ds[i]
        T = item["frames"].shape[0]
        occluded = torch.zeros(T, dtype=torch.bool)
        start = T // 2 - 1
        occluded[start : min(T, start + 3)] = True
        frames = item["frames"].clone()
        frames[occluded] = 0.0
        yield {
            "frames": frames.unsqueeze(0),
            "occluded": occluded.unsqueeze(0),
            "proprio": item["proprio"].unsqueeze(0),  # unaffected by visual occlusion
        }


@torch.no_grad()
def sigma_traces(perception, belief_net, loader, device) -> list[dict]:
    rows = []
    for clip_id, batch in enumerate(loader):
        frames = batch["frames"].to(device)
        proprio = batch["proprio"].to(device)
        patches = common.encode_frames(perception, frames)
        beliefs = belief_net.rollout(patches, proprio)
        occluded = batch.get("occluded")
        for t, b in enumerate(beliefs):
            rows.append({
                "clip": clip_id,
                "t": t,
                "sigma_mean": b.sigma.mean().item(),
                "entropy": b.entropy().mean().item(),
                "occluded": bool(occluded[0, t]) if occluded is not None else False,
            })
    return rows


def main() -> None:
    p = common.base_parser(__doc__)
    p.add_argument("--clips", type=int, default=64)
    args = p.parse_args()
    cfg = common.load_config(args.config)
    common.set_seed(cfg["seed"])
    device = torch.device(args.device)

    perception = common.build_perception(cfg, args.smoke).to(device)
    belief_net = BeliefStateNetwork().to(device)
    if not args.smoke:
        common.load_checkpoint(belief_net, cfg, "belief", args.device)
    belief_net.eval()

    # In-domain probe when the model trained on LIBERO; synthetic otherwise.
    if not args.smoke and cfg["data"]["source"] == "libero":
        loader = libero_occlusion_clips(cfg, args.clips)
    else:
        # Held-out seed: never used in training (train datasets use seed=0 stream).
        dataset = SyntheticTrajectoryDataset(
            size=4 if args.smoke else args.clips, clip_len=cfg["data"]["clip_len"], seed=999
        )
        loader = DataLoader(dataset, batch_size=1)
    rows = sigma_traces(perception, belief_net, loader, device)

    out_dir = common.REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "uncertainty.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    occ = [r["sigma_mean"] for r in rows if r["occluded"]]
    vis = [r["sigma_mean"] for r in rows if not r["occluded"]]
    if occ and vis:
        ratio = (sum(occ) / len(occ)) / (sum(vis) / len(vis))
        print(f"sigma occluded/visible ratio (pooled): {ratio:.3f} (want > 1.0 after training)")

    # Per-clip comparison, not just pooled: pooling occluded/visible sigma
    # across all clips lets between-clip differences in baseline sigma level
    # (e.g. some tasks/scenes are just busier) dilute a real within-clip
    # effect — found 08-13 (see docs/EXPERIMENT_LOG.md), where the pooled
    # ratio read as a thin 1.02 but the per-clip win-rate was a much
    # stronger, statistically robust 77%. Compare each clip's occluded-mean
    # sigma against its own visible-mean baseline instead.
    by_clip: dict[int, list[tuple[float, bool]]] = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append((r["sigma_mean"], r["occluded"]))
    diffs = []
    for pts in by_clip.values():
        c_occ = [s for s, o in pts if o]
        c_vis = [s for s, o in pts if not o]
        if c_occ and c_vis:
            diffs.append(sum(c_occ) / len(c_occ) - sum(c_vis) / len(c_vis))
    if diffs:
        wins = sum(1 for d in diffs if d > 0)
        print(
            f"sigma rises on occlusion, per-clip: {wins}/{len(diffs)} = "
            f"{wins / len(diffs):.3f} win rate (want > 0.5; this is the more "
            f"trustworthy number, see docs/EXPERIMENT_LOG.md 08-13)"
        )
    print(f"wrote {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clip0 = [r for r in rows if r["clip"] == 0]
        ts = [r["t"] for r in clip0]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ts, [r["sigma_mean"] for r in clip0], marker="o", label="mean sigma")
        for r in clip0:
            if r["occluded"]:
                ax.axvspan(r["t"] - 0.5, r["t"] + 0.5, alpha=0.2, color="red")
        ax.set_xlabel("timestep")
        ax.set_ylabel("belief sigma")
        ax.set_title("Belief uncertainty vs occlusion (red = occluded)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "uncertainty.png", dpi=150)
        print(f"wrote {out_dir / 'uncertainty.png'}")
    except ImportError:
        print("[info] matplotlib not installed; skipped figure")


if __name__ == "__main__":
    main()
