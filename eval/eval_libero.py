"""Closed-loop evaluation on LIBERO: success rate per task suite.

Requires the full LIBERO install and trained checkpoints (GPU/eval box):

    python -m eval.eval_libero --suite libero_spatial --episodes 10

Writes results/<suite>_results.json with per-task success and step-level
diagnostics (belief entropy, goal entropy, lambda) for the paper figures.
"""

import argparse
import json
from pathlib import Path

import torch

from train import common
from via.belief import BeliefStateNetwork
from via.decision import AdaptiveGate, DecisionModule, UtilityHead
from via.goal import GoalInferenceRSA
from via.model import VIAModel
from via.world_model import RSSM


def build_model(cfg: dict, device: str, smoke: bool = False) -> VIAModel:
    belief_net = BeliefStateNetwork()
    rssm = RSSM()
    goal_net = GoalInferenceRSA()
    utility = UtilityHead()
    gate = AdaptiveGate(lambda_max=cfg["decision"]["lambda_max"])
    if not smoke:
        common.load_checkpoint(belief_net, cfg, "belief", device)
        common.load_checkpoint(rssm, cfg, "world_model", device)
        common.load_checkpoint(goal_net, cfg, "goal", device)
        common.load_checkpoint(utility, cfg, "utility", device)
        common.load_checkpoint(gate, cfg, "gate", device)
    decision = DecisionModule(rssm, belief_net, utility, gate)
    model = VIAModel(
        perception=common.build_perception(cfg, smoke),
        language=common.build_language(cfg, smoke),
        belief_net=belief_net, goal_net=goal_net,
        world_model=rssm, decision=decision,
    )
    return model.to(device).eval()


def main() -> None:
    p = common.base_parser(__doc__)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--episodes", type=int, default=10, help="episodes per task")
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args()
    cfg = common.load_config(args.config)
    common.set_seed(cfg["seed"])

    from via.data.libero import LiberoEnvRunner

    model = build_model(cfg, args.device)
    runner = LiberoEnvRunner(suite=args.suite)

    results = []
    for task_id in range(runner.num_tasks()):
        successes = 0
        for ep in range(args.episodes):
            out = runner.run_episode(model, task_id, max_steps=args.max_steps, device=args.device)
            successes += out["success"]
            results.append({
                "task_id": task_id,
                "episode": ep,
                "instruction": out["instruction"],
                "success": out["success"],
                "diagnostics": [
                    {k: v.tolist() for k, v in d.items()} for d in out["diagnostics"]
                ],
            })
        print(f"task {task_id}: {successes}/{args.episodes} succeeded")

    out_dir = common.REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.suite}_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    rate = sum(r["success"] for r in results) / len(results)
    print(f"\n{args.suite}: overall success rate {rate:.1%} -> {out_path}")


if __name__ == "__main__":
    main()
