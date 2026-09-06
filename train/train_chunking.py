"""Train the action-chunking policy (via/decision/chunking.py), an
alternative to CEMPlanner + ActionPrior's single-step re-planning.

For each clip window, every valid start position k (features from
`posts[k]`, k=0..T-1-chunk_size) supervises a full K-action chunk
`actions[k+1 : k+1+chunk_size]` -- not just one start position per window,
so a clip_len=16 window with chunk_size=8 yields 8 training examples
instead of 1.

    python -m train.train_chunking       # requires belief, world_model, goal ckpts
    python -m train.train_chunking --smoke
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import common
from via.belief import BeliefStateNetwork
from via.decision import ActionChunkingPolicy
from via.goal import GoalInferenceRSA
from via.world_model import RSSM


def main() -> None:
    args = common.base_parser(__doc__).parse_args()
    cfg = common.load_config(args.config)
    common.set_seed(cfg["seed"])
    device = torch.device(args.device)
    chunking_cfg = cfg.get("chunking", {})

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

    chunk_size = chunking_cfg.get("chunk_size", 8)
    policy = ActionChunkingPolicy(chunk_size=chunk_size).to(device)
    common.try_resume(policy, cfg, "chunking_policy", args.device, args.resume)

    dataset = common.build_trajectory_dataset(cfg, args.smoke)
    loader = DataLoader(
        dataset, batch_size=2 if args.smoke else chunking_cfg.get("batch_size", 16), shuffle=True
    )
    opt = torch.optim.AdamW(policy.parameters(), lr=chunking_cfg.get("lr", 3.0e-4))
    run = common.init_wandb(cfg, "chunking", args.no_wandb or args.smoke)

    epochs = 1 if args.smoke else chunking_cfg.get("epochs", 10)
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
                beliefs = belief_net.rollout(patches, proprio)
                posts, _ = rssm.observe(obs_embeds, actions[:, :-1])
                tokens, mask = language(list(batch["instruction"]))
                goal = goal_net(tokens.to(device), mask.to(device), beliefs[-1].mu)

            feats = torch.stack([p.feature for p in posts], dim=1)  # (B, T-1, F)
            n_valid = feats.shape[1] - chunk_size  # valid start positions k=0..n_valid-1
            if n_valid <= 0:
                raise ValueError(
                    f"clip_len={cfg['data']['clip_len']} too short for chunk_size={chunk_size}"
                )
            goal_rep = goal.embedding.unsqueeze(1).expand(-1, n_valid, -1)
            pred = policy(
                feats[:, :n_valid].flatten(0, 1), goal_rep.flatten(0, 1)
            ).view(B, n_valid, chunk_size, -1)  # (B, n_valid, K, A)

            # Target chunk for start k (posts index k -> real frame k+1):
            # actions[k+1 : k+1+K].
            target = torch.stack(
                [actions[:, k + 1 : k + 1 + chunk_size] for k in range(n_valid)], dim=1
            )  # (B, n_valid, K, A)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            common.log_metrics(run, {"loss": loss}, step)
            step += 1
            if args.smoke and step >= 3:
                break
        common.save_checkpoint(policy, cfg, "chunking_policy")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
