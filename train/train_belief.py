"""Train the belief state network (self-supervised next-observation prediction).

    python -m train.train_belief                 # real run (config-driven)
    python -m train.train_belief --smoke         # 1-epoch CPU sanity check

Milestone check (June, weeks 3-4): after training, run
eval/uncertainty_analysis.py and confirm sigma spikes on occluded frames.
"""

import torch
from torch.utils.data import DataLoader

from train import common
from via.belief import BeliefStateNetwork


def occlude_clips(
    frames: torch.Tensor, p: float, max_len: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sensor-dropout augmentation: blank a random window per clip with prob p.

    LIBERO demos contain no occlusions, so without this the next observation
    is always predictable and the NLL never pressures sigma to rise (occ/vis
    ratio 1.00 in runs 1-2). Blanked windows force the belief to admit
    uncertainty while blind.

    Returns (frames, occluded); the mask also feeds BeliefStateNetwork.loss
    so its anti-collapse variance term excludes blanked frames.
    """
    B, T = frames.shape[:2]
    frames = frames.clone()
    occluded = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        if T >= 6 and torch.rand(()) < p:
            length = int(torch.randint(2, max_len + 1, ()))
            start = int(torch.randint(1, T - length, ()))
            frames[b, start : start + length] = 0.0
            occluded[b, start : start + length] = True
    return frames, occluded


def main() -> None:
    args = common.base_parser(__doc__).parse_args()
    cfg = common.load_config(args.config)
    common.set_seed(cfg["seed"])
    device = torch.device(args.device)

    perception = common.build_perception(cfg, args.smoke).to(device)
    belief_net = BeliefStateNetwork(
        kl_weight=cfg["belief"]["kl_weight"],
        var_weight=cfg["belief"].get("var_weight", 1.0),
    ).to(device)
    common.try_resume(belief_net, cfg, "belief", args.device, args.resume)
    dataset = common.build_trajectory_dataset(cfg, args.smoke)
    loader = DataLoader(
        dataset,
        batch_size=2 if args.smoke else cfg["belief"]["batch_size"],
        shuffle=True,
        num_workers=0 if args.smoke else cfg["data"]["num_workers"],
    )
    opt = torch.optim.AdamW(belief_net.parameters(), lr=cfg["belief"]["lr"])
    run = common.init_wandb(cfg, "belief-state", args.no_wandb or args.smoke)

    epochs = 1 if args.smoke else cfg["belief"]["epochs"]
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            frames = batch["frames"].to(device)
            occluded = batch.get("occluded")
            occluded = occluded.to(device).bool() if occluded is not None else None
            occ_p = cfg["belief"].get("occlude_p", 0.0)
            if occ_p > 0:
                frames, occ_aug = occlude_clips(frames, occ_p)
                occ_aug = occ_aug.to(device)
                occluded = occ_aug if occluded is None else (occluded | occ_aug)
            patches = common.encode_frames(perception, frames)
            losses = belief_net.loss(patches, occluded=occluded)
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(belief_net.parameters(), 10.0)
            opt.step()
            common.log_metrics(run, losses, step)
            step += 1
            if args.smoke and step >= 3:
                break
        common.save_checkpoint(belief_net, cfg, "belief")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
