"""Pretrain (synthetic) / fine-tune (real LIBERO) the RSA goal inference module.

    python -m train.train_goal                              # synthetic pretraining
    python -m train.train_goal --config configs/local.yaml   # real-data fine-tune (data.source: libero)
    python -m train.train_goal --smoke

Milestone check (August, weeks 7-8), synthetic path: the logged
`entropy_amb*` values should be monotone increasing — goal distribution
entropy tracks instruction ambiguity.

Real-data fine-tune (data.source: libero, added 08-13 — see
docs/EXPERIMENT_LOG.md): synthetic pretraining teaches the language->goal
mapping in isolation, always with zero belief context, on a tiny synthetic
vocabulary (4 verbs x 8 objects). It does not generalize to real LIBERO
instructions or real belief conditioning — checked directly before decision
training and confirmed broken (near-maximum entropy regardless of
instruction content). This path fixes that: LIBERO has no ready-made
(instruction, goal) labels the way the synthetic generator does, so each
task (one fixed instruction per demo file — 10 for libero_spatial) is
treated as its own goal-slot label, coarser than the synthetic setup's
compositional structure but the only labels the data provides. Warm-starts
from the existing `goal` checkpoint automatically (fine-tuning, not a fresh
run) and trains against real, non-zero belief context from the frozen
belief checkpoint.
"""

import torch
from torch.utils.data import DataLoader

from train import common
from via.belief import BeliefStateNetwork
from via.contracts import C
from via.data.synthetic import SyntheticInstructionDataset
from via.goal import GoalInferenceRSA


def _train_synthetic(args, cfg: dict, device: torch.device) -> None:
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


def _train_libero(args, cfg: dict, device: torch.device) -> None:
    perception = common.build_perception(cfg, args.smoke).to(device)
    language = common.build_language(cfg, args.smoke).to(device)
    goal_net = GoalInferenceRSA().to(device)
    try:
        common.load_checkpoint(goal_net, cfg, "goal", args.device)
        print("[info] warm-started goal_net from the existing checkpoint (synthetic pretraining)")
    except FileNotFoundError:
        print("[warn] no existing goal checkpoint; fine-tuning from random init")

    belief_net = BeliefStateNetwork().to(device)
    common.load_checkpoint(belief_net, cfg, "belief", args.device)
    belief_net.eval()

    dataset = common.build_trajectory_dataset(cfg, args.smoke)  # split="train" by default
    batch_size = cfg["goal"].get("libero_batch_size", cfg["goal"]["batch_size"])
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=cfg["data"]["num_workers"]
    )
    opt = torch.optim.AdamW(goal_net.parameters(), lr=cfg["goal"]["lr"])
    run = common.init_wandb(cfg, "goal-rsa-libero", args.no_wandb)

    epochs = cfg["goal"].get("libero_epochs", cfg["goal"]["epochs"])
    step = 0
    best_acc, best_state = -1.0, None
    for epoch in range(epochs):
        for batch in loader:
            frames = batch["frames"].to(device)
            task_id = batch["task_id"].to(device)
            proprio = batch["proprio"].to(device)
            with torch.no_grad():
                patches = common.encode_frames(perception, frames)
                beliefs = belief_net.rollout(patches, proprio)
                belief_mu = beliefs[-1].mu
            tokens, mask = language(list(batch["instruction"]))
            losses = goal_net.rsa_training_loss(tokens.to(device), mask.to(device), belief_mu, task_id)
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            common.log_metrics(run, losses, step)

            if step % 50 == 0:
                holdout = common.goal_holdout_check(cfg, perception, belief_net, language, goal_net, device)
                common.log_metrics(run, holdout, step)
                # Best-checkpoint selection, not just the final step: found
                # 2026-08-1x that held-out accuracy can peak mid-training and
                # then *decline* on this fine-tune (belief.mu is now a richer,
                # more per-demo-specific signal since it's proprio-fused, so
                # continued training can overfit trajectory idiosyncrasies of
                # the training demos rather than the shared task-level
                # signal) -- unlike this stage's previous, purely-visual-mu
                # runs, which climbed monotonically. See docs/EXPERIMENT_LOG.md.
                if holdout["goal_holdout_accuracy"] > best_acc:
                    best_acc = holdout["goal_holdout_accuracy"]
                    best_state = {k: v.detach().clone() for k, v in goal_net.state_dict().items()}
            step += 1
        common.save_checkpoint(goal_net, cfg, "goal")

    if best_state is not None:
        print(f"[info] restoring best held-out checkpoint (accuracy={best_acc:.4f}) over the final one")
        goal_net.load_state_dict(best_state)
        common.save_checkpoint(goal_net, cfg, "goal")

    if run is not None:
        run.finish()


def main() -> None:
    args = common.base_parser(__doc__).parse_args()
    cfg = common.load_config(args.config)
    common.set_seed(cfg["seed"])
    device = torch.device(args.device)

    if not args.smoke and cfg["data"]["source"] == "libero":
        _train_libero(args, cfg, device)
    else:
        _train_synthetic(args, cfg, device)


if __name__ == "__main__":
    main()
