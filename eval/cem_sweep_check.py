"""Stage-0 diagnostic: sweep CEMPlanner search-time hyperparameters
(horizon, population, iterations) against the same gripper-to-target
distance metric as lambda_ablation_check.py.

These are pure planning-time settings (via/decision/decision.py's
CEMConfig), not learned weights, so this costs zero training -- only extra
sim/CEM compute per episode. Motivation: the world model's 5-step imagined
rollouts, searched with a fairly small population (64) / few iterations (3),
may simply not be enough search to find a precise grasp trajectory even if
the objective itself were perfect; this isolates that from the objective
issues lambda_ablation_check.py targets.

    python -m eval.cem_sweep_check --tasks 0 1 2 --episodes 2
"""

import argparse
import json

import numpy as np
import torch

from eval.eval_libero import build_model
from eval.lambda_ablation_check import run_episode_with_distance
from train import common
from via.decision.decision import CEMConfig, CEMPlanner


CONFIGS = {
    "baseline": CEMConfig(horizon=5, population=64, elites=6, iterations=3),
    "long_horizon": CEMConfig(horizon=10, population=64, elites=6, iterations=3),
    "wide_population": CEMConfig(horizon=5, population=128, elites=12, iterations=3),
    "more_iterations": CEMConfig(horizon=5, population=64, elites=6, iterations=6),
    # Tighter search radius around the action_prior-seeded mean, motivated
    # by 2026-08-22's finding (see docs/EXPERIMENT_LOG.md): raw action_prior
    # execution (no CEM at all) beat the default (init_std=0.5) CEM pipeline
    # on both approach distance and end-of-episode drift, suggesting CEM's
    # search noise was wandering away from an already-good imitation
    # trajectory rather than refining it. These narrow the search instead of
    # eliminating it, to see if CEM can still contribute local correction
    # once it's not allowed to drift as far.
    "tight_std_01": CEMConfig(horizon=5, population=64, elites=6, iterations=3, init_std=0.1),
    "tight_std_005": CEMConfig(horizon=5, population=64, elites=6, iterations=3, init_std=0.05),
}


def main() -> None:
    p = common.base_parser(__doc__)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=2, help="episodes per task")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    args = p.parse_args()
    cfg = common.load_config(args.config)

    from via.data.libero import LiberoEnvRunner

    runner = LiberoEnvRunner(suite=args.suite)
    results = {}
    for name in args.configs:
        common.set_seed(cfg["seed"])  # identical episode conditions per config
        model = build_model(cfg, args.device)
        model.decision.planner = CEMPlanner(CONFIGS[name])
        eps = []
        for task_id in args.tasks:
            for _ in range(args.episodes):
                out = run_episode_with_distance(model, runner, task_id, args.max_steps, args.device)
                eps.append(out)
                d = out["distances"]
                print(
                    f"[{name}] task {task_id} '{out['instruction'][:40]}...': "
                    f"start={d[0]:.3f} min={min(d):.3f} end={d[-1]:.3f} success={out['success']}"
                )
        results[name] = eps

    out_path = common.REPO_ROOT / "results" / "cem_sweep_check.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n--- summary (mean over episodes) ---")
    for name, eps in results.items():
        mins = [min(e["distances"]) for e in eps]
        ends = [e["distances"][-1] for e in eps]
        succ = [e["success"] for e in eps]
        print(
            f"{name}: min={sum(mins)/len(mins):.3f} end={sum(ends)/len(ends):.3f} "
            f"success_rate={sum(succ)}/{len(succ)}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
