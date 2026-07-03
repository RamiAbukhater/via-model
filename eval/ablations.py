"""Ablation study: isolate the contribution of each cognitive component.

Variants (the proposal's primary experiment):

    full          complete VIA model
    lambda0       epistemic value off (gate forced to 0) — pure exploitation
    point_goal    goal distribution collapsed to its argmax prototype —
                  language as command, not evidence
    no_belief     belief variance zeroed — point-estimate perception

    python -m eval.ablations --suite libero_spatial --episodes 5

Writes results/ablations.json and prints a markdown table for the paper.
"""

import argparse
import json

import torch

from eval.eval_libero import build_model
from train import common
from via.goal.rsa import GoalDistribution


def apply_variant(model, variant: str):
    if variant == "full":
        return model
    if variant == "lambda0":
        gate = model.decision.gate
        model.decision.gate = lambda belief, goal_ent: torch.zeros_like(
            gate(belief, goal_ent)
        )
    elif variant == "point_goal":
        inner = model.goal_net.forward

        def argmax_goal(tokens, mask, belief_mu):
            dist = inner(tokens, mask, belief_mu)
            hard = torch.zeros_like(dist.probs)
            hard.scatter_(-1, dist.probs.argmax(-1, keepdim=True), 1.0)
            emb = hard @ model.goal_net.goal_prototypes
            return GoalDistribution(probs=hard, log_probs=(hard + 1e-9).log(), embedding=emb)

        model.goal_net.forward = argmax_goal
    elif variant == "no_belief":
        step = model.belief_net.step

        def point_step(obs_embed, hidden):
            b = step(obs_embed, hidden)
            b.logvar.fill_(-8.0)
            return b

        model.belief_net.step = point_step
    else:
        raise ValueError(f"unknown variant {variant}")
    return model


def main() -> None:
    p = common.base_parser(__doc__)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--variants", nargs="+",
                   default=["full", "lambda0", "point_goal", "no_belief"])
    args = p.parse_args()
    cfg = common.load_config(args.config)

    from via.data.libero import LiberoEnvRunner

    runner = LiberoEnvRunner(suite=args.suite)
    table = {}
    for variant in args.variants:
        common.set_seed(cfg["seed"])  # identical episode conditions per variant
        model = apply_variant(build_model(cfg, args.device), variant)
        n, wins = 0, 0
        for task_id in range(runner.num_tasks()):
            for _ in range(args.episodes):
                out = runner.run_episode(
                    model, task_id, max_steps=args.max_steps, device=args.device
                )
                wins += out["success"]
                n += 1
        table[variant] = wins / n
        print(f"{variant}: {wins}/{n} = {wins / n:.1%}")

    out_dir = common.REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "ablations.json").write_text(json.dumps(table, indent=2))

    print("\n| Variant | Success rate |")
    print("|---|---|")
    for k, v in table.items():
        print(f"| {k} | {v:.1%} |")


if __name__ == "__main__":
    main()
