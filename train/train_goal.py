"""Pretrain the RSA goal inference module on synthetic instruction/goal pairs.

    python -m train.train_goal
    python -m train.train_goal --smoke

Milestone check (August, weeks 7-8): the logged `entropy_by_ambiguity`
values should be monotone increasing — goal distribution entropy tracks
instruction ambiguity. After synthetic pretraining, fine-tune on LIBERO task
instructions by rerunning with data.source=libero in the config.
"""

import torch
from torch.utils.data import DataLoader

from train import common
from via.belief import BeliefStateNetwork
from via.contracts import C
from via.data.synthetic import SyntheticInstructionDataset
from via.goal import GoalInferenceRSA


def main() -> None:
    args = common.base_parser(__doc__).parse_args()
    cfg = common.load_config(args.config)
    common.set_seed(cfg["seed"])
    device = torch.device(args.device)

    language = common.build_language(cfg, args.smoke).to(device)
    goal_net = GoalInferenceRSA().to(device)
    common.try_resume(goal_net, cfg, "goal", args.device, args.resume)

    # Belief context: use the trained belief net over synthetic frames when
    # available; a zero belief (uninformative context) otherwise. Synthetic
    # pretraining is about the language->goal mapping and its calibration.
    belief_net = BeliefStateNetwork().to(device)
    try:
        common.load_checkpoint(belief_net, cfg, "belief", args.device)
    except FileNotFoundError:
        print("[info] no belief checkpoint; pretraining with zero belief context")

    dataset = SyntheticInstructionDataset(size=128 if args.smoke else cfg["goal"]["dataset_size"])
    loader = DataLoader(dataset, batch_size=8 if args.smoke else cfg["goal"]["batch_size"],
                        shuffle=True)
    opt = torch.optim.AdamW(goal_net.parameters(), lr=cfg["goal"]["lr"])
    run = common.init_wandb(cfg, "goal-rsa", args.no_wandb or args.smoke)

    epochs = 1 if args.smoke else cfg["goal"]["epochs"]
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            tokens, mask = language(list(batch["instruction"]))
            belief_mu = torch.zeros(len(batch["goal"]), C.belief_dim, device=device)
            losses = goal_net.rsa_training_loss(
                tokens.to(device), mask.to(device), belief_mu, batch["goal"].to(device)
            )
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()

            metrics = dict(losses)
            # Calibration probe: entropy should rise with ambiguity level.
            if step % 50 == 0:
                with torch.no_grad():
                    dist = goal_net(tokens.to(device), mask.to(device), belief_mu)
                    ent = dist.entropy()
                    for a in (0, 1, 2):
                        sel = batch["ambiguity"] == a
                        if sel.any():
                            metrics[f"entropy_amb{a}"] = ent[sel.to(device)].mean()
            common.log_metrics(run, metrics, step)
            step += 1
            if args.smoke and step >= 3:
                break
        common.save_checkpoint(goal_net, cfg, "goal")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
