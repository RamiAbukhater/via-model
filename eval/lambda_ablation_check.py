"""Stage-0 diagnostic: does forcing lambda=0 (pure exploitation) change
gripper-to-target-object distance over an episode, compared to the full
model?

Motivated by docs/EXPERIMENT_LOG.md's 2026-08-16 finding that belief sigma
(and therefore lambda, via AdaptiveGate) barely moves across a live episode
(early-mean 0.7605 -> late-mean 0.7563) -- the explore/exploit switch the
active-inference objective depends on never actually engages, so the
information-gain term stays weighted comparably to (and >= half the time,
more than) expected utility for the whole episode, including the final
approach/grasp window.

Reuses `apply_variant` from eval/ablations.py (the `lambda0` variant already
forces the gate to output 0) and LIBERO's own `obj_of_interest` (available
per-task on the live env) to identify the pick target object, so distance is
computed from `<target>_to_robot0_eef_pos`, an observation LIBERO already
exposes -- no new instrumentation of the simulator needed.

    python -m eval.lambda_ablation_check --tasks 0 1 2 --episodes 2
"""

import argparse
import json

import numpy as np
import torch

from eval.ablations import apply_variant
from eval.eval_libero import build_model
from train import common
from via.data.libero import clamp_object_rel


def run_episode_with_distance(model, runner, task_id: int, max_steps: int, device: str) -> dict:
    env, instruction = runner.make_env(task_id)
    try:
        env.reset()
        target = env.env.obj_of_interest[0]
        obs = env.env._get_observations()
        state = model.reset([instruction], device=torch.device(device))
        dists = []
        for _ in range(max_steps):
            dists.append(float(np.linalg.norm(obs[f"{target}_to_robot0_eef_pos"])))
            frame = (
                torch.from_numpy(obs["agentview_image"].copy())
                .permute(2, 0, 1).float().unsqueeze(0) / 255.0
            ).to(device)
            proprio = torch.from_numpy(
                np.concatenate([
                    obs["robot0_eef_pos"], obs["robot0_gripper_qpos"],
                    clamp_object_rel(obs[f"{target}_to_robot0_eef_pos"]),
                ])
            ).float().unsqueeze(0).to(device)
            action, state, _ = model.act(frame, proprio, state)
            obs, _, done, info = env.step(action[0].cpu().numpy())
            if done:
                break
        success = bool(env.env._check_success())
    finally:
        env.close()
    return {
        "task_id": task_id, "instruction": instruction, "target": target,
        "success": success, "distances": dists,
    }


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
    for variant in ["full", "lambda0"]:
        common.set_seed(cfg["seed"])  # identical episode conditions per variant
        model = apply_variant(build_model(cfg, args.device), variant)
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

    out_path = common.REPO_ROOT / "results" / "lambda_ablation_check.json"
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
