"""Diagnostic: does bypassing CEM search entirely and just executing the
raw behavior-cloned action_prior get closer to the target than the full
CEM(utility, information_gain) pipeline?

Motivated by a single-episode probe (2026-08-22, see docs/EXPERIMENT_LOG.md)
where the raw action_prior reached min distance 0.105 vs. the full pipeline's
typical 0.24-0.27 -- suggesting CEM's population-based stochastic search may
be degrading precision relative to just trusting the imitation policy
directly, rather than refining it.

    python -m eval.action_prior_only_check --tasks 0 1 2 --episodes 2
"""

import json

import torch

from eval.eval_libero import build_model
from eval.lambda_ablation_check import run_episode_with_distance
from train import common


def action_prior_only(model):
    """Monkey-patch select_action to bypass CEM/utility/IG entirely --
    just the raw, single-step action_prior prediction."""
    action_prior = model.decision.action_prior

    def raw_select_action(belief, rssm_state, goal_embed, goal_entropy, min_sigma):
        a = action_prior(rssm_state.feature, goal_embed)
        return {"action": a, "lambda": torch.zeros(a.shape[0], device=a.device)}

    model.decision.select_action = raw_select_action
    return model


def main() -> None:
    p = common.base_parser(__doc__)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=2, help="episodes per task")
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args()
    cfg = common.load_config(args.config)

    from via.data.libero import LiberoEnvRunner

    runner = LiberoEnvRunner(suite=args.suite)
    results = {}
    for variant in ["full", "action_prior_only"]:
        common.set_seed(cfg["seed"])
        model = build_model(cfg, args.device)
        if variant == "action_prior_only":
            model = action_prior_only(model)
        eps = []
        for task_id in args.tasks:
            for _ in range(args.episodes):
                out = run_episode_with_distance(model, runner, task_id, args.max_steps, args.device)
                eps.append(out)
                d = out["distances"]
                print(
                    f"[{variant}] task {task_id} '{out['instruction'][:40]}...': "
                    f"start={d[0]:.3f} min={min(d):.3f} end={d[-1]:.3f} success={out['success']}"
                )
        results[variant] = eps

    out_path = common.REPO_ROOT / "results" / "action_prior_only_check.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n--- summary (mean over episodes) ---")
    for variant, eps in results.items():
        starts = [e["distances"][0] for e in eps]
        mins = [min(e["distances"]) for e in eps]
        ends = [e["distances"][-1] for e in eps]
        succ = [e["success"] for e in eps]
        print(
            f"{variant}: start={sum(starts)/len(starts):.3f} "
            f"min={sum(mins)/len(mins):.3f} end={sum(ends)/len(ends):.3f} "
            f"success_rate={sum(succ)}/{len(succ)}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
