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


@torch.no_grad()
def sigma_traces(perception, belief_net, loader, device) -> list[dict]:
    rows = []
    for clip_id, batch in enumerate(loader):
        frames = batch["frames"].to(device)
        patches = common.encode_frames(perception, frames)
        beliefs = belief_net.rollout(patches)
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
        print(f"sigma occluded/visible ratio: {ratio:.3f} (want > 1.0 after training)")
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
