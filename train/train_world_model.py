"""Train the RSSM world model on trajectories, with the belief encoder frozen.

    python -m train.train_world_model            # requires checkpoints/belief.pt
    python -m train.train_world_model --smoke

Milestone check (July, weeks 5-6): logged `rollout5_model_mse` should beat
`rollout5_naive_mse` (5-step open-loop prediction vs persistence baseline).
"""

import torch
from torch.utils.data import DataLoader

from train import common
from via.belief import BeliefStateNetwork
from via.world_model import RSSM


def main() -> None:
    args = common.base_parser(__doc__).parse_args()
    cfg = common.load_config(args.config)
    common.set_seed(cfg["seed"])
    device = torch.device(args.device)

    perception = common.build_perception(cfg, args.smoke).to(device)
    belief_net = BeliefStateNetwork().to(device)
    if not args.smoke:
        common.load_checkpoint(belief_net, cfg, "belief", args.device)
    belief_net.eval()

    rssm = RSSM(free_bits=cfg["world_model"]["free_bits"]).to(device)
    common.try_resume(rssm, cfg, "world_model", args.device, args.resume)
    dataset = common.build_trajectory_dataset(cfg, args.smoke)
    loader = DataLoader(
        dataset,
        batch_size=2 if args.smoke else cfg["world_model"]["batch_size"],
        shuffle=True,
        num_workers=0 if args.smoke else cfg["data"]["num_workers"],
    )
    opt = torch.optim.AdamW(rssm.parameters(), lr=cfg["world_model"]["lr"])
    run = common.init_wandb(cfg, "world-model", args.no_wandb or args.smoke)

    epochs = 1 if args.smoke else cfg["world_model"]["epochs"]
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            proprio = batch["proprio"].to(device)
            with torch.no_grad():
                patches = common.encode_frames(perception, frames)
                B, T = patches.shape[:2]
                obs_embeds = torch.stack(
                    [belief_net.encode_obs(patches[:, t], proprio[:, t]) for t in range(T)], dim=1
                )
            losses = rssm.loss(obs_embeds, actions[:, :-1])
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), 10.0)
            opt.step()

            metrics = dict(losses)
            if step % 50 == 0 and T >= 6:
                roll = rssm.rollout_mse(obs_embeds, actions[:, :-1], k=5)
                metrics["rollout5_model_mse"] = roll["model_mse"]
                metrics["rollout5_naive_mse"] = roll["naive_mse"]
            # In-training-batch rollout5 (above) is not trustworthy alone —
            # 07-23 found it can look like a win on the batch mean while the
            # model loses on most individual clips. This periodic check runs
            # on genuinely held-out data (LiberoTrajectoryDataset split="val")
            # and reports win-rate, so a bad run can be caught well before
            # all `epochs` complete instead of only at the very end.
            if not args.smoke and step % 500 == 0:
                rssm.eval()
                holdout = common.world_model_holdout_check(cfg, perception, belief_net, rssm, device)
                rssm.train()
                common.log_metrics(run, holdout, step)
            common.log_metrics(run, metrics, step)
            step += 1
            if args.smoke and step >= 3:
                break
        common.save_checkpoint(rssm, cfg, "world_model")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
