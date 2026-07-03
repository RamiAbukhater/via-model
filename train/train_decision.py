"""Train the decision module's learned components on demonstration data.

Two parts, trained here offline from demos:

- UtilityHead: regresses task *progress* (fraction of the demo completed)
  from RSSM posterior features conditioned on the inferred goal embedding.
  Because LIBERO demos are successful, progress is a well-defined proxy for
  expected goal achievement, and imagined states inherit the scale.

- AdaptiveGate: initialized offline to a monotone prior — lambda should rise
  with belief sigma (explore when uncertain). This is an *initialization*;
  the gate's final calibration comes from closed-loop episodes (see
  eval/ablations.py), since only interaction reveals whether information
  gathering actually paid off. Both stages log to the same W&B run family.

    python -m train.train_decision       # requires belief, world_model, goal ckpts
    python -m train.train_decision --smoke
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import common
from via.belief import BeliefStateNetwork
from via.decision import AdaptiveGate, UtilityHead
from via.goal import GoalInferenceRSA
from via.world_model import RSSM


def main() -> None:
    args = common.base_parser(__doc__).parse_args()
    cfg = common.load_config(args.config)
    common.set_seed(cfg["seed"])
    device = torch.device(args.device)

    perception = common.build_perception(cfg, args.smoke).to(device)
    language = common.build_language(cfg, args.smoke).to(device)
    belief_net = BeliefStateNetwork().to(device)
    rssm = RSSM().to(device)
    goal_net = GoalInferenceRSA().to(device)
    if not args.smoke:
        common.load_checkpoint(belief_net, cfg, "belief", args.device)
        common.load_checkpoint(rssm, cfg, "world_model", args.device)
        common.load_checkpoint(goal_net, cfg, "goal", args.device)
    for m in (belief_net, rssm, goal_net):
        m.eval()

    utility = UtilityHead().to(device)
    gate = AdaptiveGate(lambda_max=cfg["decision"]["lambda_max"]).to(device)

    dataset = common.build_trajectory_dataset(cfg, args.smoke)
    loader = DataLoader(dataset, batch_size=2 if args.smoke else cfg["decision"]["batch_size"],
                        shuffle=True)
    opt = torch.optim.AdamW(
        list(utility.parameters()) + list(gate.parameters()), lr=cfg["decision"]["lr"]
    )
    run = common.init_wandb(cfg, "decision", args.no_wandb or args.smoke)

    epochs = 1 if args.smoke else cfg["decision"]["epochs"]
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            progress = batch["progress"].to(device)

            with torch.no_grad():
                patches = common.encode_frames(perception, frames)
                B, T = patches.shape[:2]
                obs_embeds = torch.stack(
                    [belief_net.encode_obs(patches[:, t]) for t in range(T)], dim=1
                )
                beliefs = belief_net.rollout(patches)
                posts, _ = rssm.observe(obs_embeds, actions[:, :-1])
                tokens, mask = language(list(batch["instruction"]))
                goal = goal_net(tokens.to(device), mask.to(device), beliefs[-1].mu)

            # Utility: progress regression on posterior features at t=1..T-1.
            feats = torch.stack([p.feature for p in posts], dim=1)          # (B, T-1, F)
            goal_rep = goal.embedding.unsqueeze(1).expand(-1, feats.shape[1], -1)
            pred = utility(feats.flatten(0, 1), goal_rep.flatten(0, 1)).view(B, -1)
            u_loss = F.mse_loss(pred, progress[:, 1:])

            # Gate init: monotone-in-sigma prior target.
            sig = torch.stack([b.sigma.mean(-1) for b in beliefs], dim=1)   # (B, T)
            target_lam = cfg["decision"]["lambda_max"] * torch.sigmoid(
                (sig - sig.mean()) / (sig.std() + 1e-6)
            )
            lam_pred = torch.stack(
                [gate(beliefs[t], goal.entropy()) for t in range(T)], dim=1
            )
            g_loss = F.mse_loss(lam_pred, target_lam)

            loss = u_loss + g_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            common.log_metrics(
                run, {"loss": loss, "utility_mse": u_loss, "gate_mse": g_loss}, step
            )
            step += 1
            if args.smoke and step >= 3:
                break
        common.save_checkpoint(utility, cfg, "utility")
        common.save_checkpoint(gate, cfg, "gate")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
