"""Generate synthetic demonstrations via trajectory retargeting
(scripts/retargeted_demo_augment.py) and save them as a companion HDF5 per
task -- same `data/demo_K/{obs/agentview_rgb, obs/ee_pos, obs/gripper_states,
actions, rewards}` layout as the real demo files, plus `object_rel` inline
(same signal scripts/extract_object_state.py produces for real demos, but
computed for free here since obj_pos is already read every step).

Written to a SEPARATE directory (`data.augment_dir`), never touching the
real demo files -- `LiberoTrajectoryDataset` merges both sources under the
same task_id (matched by filename, i.e. by instruction) rather than
treating augmented demos as new tasks, so goal_net's per-file classification
target stays correct (see docs/EXPERIMENT_LOG.md 2026-08-24 for why that
matters: a naive "just add more hdf5 files to the same folder" approach
would have given the same instruction text two different task_id labels).

Only successful rollouts are kept (matching "demos are successful by
construction," the same assumption `progress`/`reward` already rely on).

    python -m scripts.generate_augmented_demos --task 0 --target 100
    python -m scripts.generate_augmented_demos --all-tasks --target 100
"""

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np

from scripts.retargeted_demo_augment import (
    RetargetedController,
    find_demo_file_for_task,
    load_all_base_demos,
    nearest_demo,
)


def generate_for_task(runner, task_id: int, files, out_dir: Path, target: int,
                       max_steps: int = 300, max_attempts_factor: int = 3):
    env, instruction = runner.make_env(task_id)
    env.reset()
    target_obj, target_dest = env.env.obj_of_interest
    demo_path = find_demo_file_for_task(files, instruction)
    demos = load_all_base_demos(env, demo_path, target_obj, target_dest)

    settle_action = np.zeros(7, dtype=np.float32)
    settle_action[6] = -1.0

    out_path = out_dir / Path(demo_path).name
    saved = 0
    attempts = 0
    max_attempts = target * max_attempts_factor
    with h5py.File(out_path, "w") as dst:
        dst.attrs["source"] = "retargeted_demo_augment"
        dst.attrs["base_file"] = str(demo_path)
        while saved < target and attempts < max_attempts:
            attempts += 1
            env.reset()
            for _ in range(10):
                obs, _, _, _ = env.step(settle_action)
            base = nearest_demo(demos, obs[f"{target_obj}_pos"])
            obj_offset = obs[f"{target_obj}_pos"] - base["obj_pos"]
            dest_offset = obs[f"{target_dest}_pos"] - base["dest_pos"]
            controller = RetargetedController(base["ee_pos"], base["ori_delta"], base["gripper"])

            frames, ee_pos, gripper_states, actions_log, obj_rel_log = [], [], [], [], []
            for t in range(max_steps):
                frames.append(obs["agentview_image"].copy())
                ee_pos.append(obs["robot0_eef_pos"].copy())
                gripper_states.append(obs["robot0_gripper_qpos"].copy())
                obj_rel_log.append(obs[f"{target_obj}_to_robot0_eef_pos"].copy())
                action, done = controller.act(obs, obj_offset, dest_offset)
                actions_log.append(action.copy())
                obs, _, sim_done, info = env.step(action)
                if done or sim_done:
                    break
            success = bool(env.env._check_success())
            if not success:
                continue

            T = len(actions_log)
            rewards = np.zeros(T, dtype=np.uint8)
            rewards[-1] = 1
            g = dst.create_group(f"data/demo_{saved}")
            g.create_dataset("actions", data=np.stack(actions_log))
            g.create_dataset("rewards", data=rewards)
            g.create_dataset("obs/agentview_rgb", data=np.stack(frames), compression="gzip")
            g.create_dataset("obs/ee_pos", data=np.stack(ee_pos))
            g.create_dataset("obs/gripper_states", data=np.stack(gripper_states))
            g.create_dataset("object_rel", data=np.stack(obj_rel_log))
            saved += 1

        dst.attrs["num_demos"] = saved
        dst.attrs["num_attempts"] = attempts

    env.close()
    print(f"task {task_id} ({instruction[:50]}): {saved}/{attempts} attempts saved -> {out_path}")
    return saved, attempts


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--target", type=int, default=100, help="successful demos per task")
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args()
    if args.task is None and not args.all_tasks:
        raise SystemExit("pass --task N or --all-tasks")

    from train import common
    from via.data.libero import LiberoEnvRunner

    cfg = common.load_config("configs/local.yaml")
    files = sorted(glob.glob(f"{cfg['data']['libero_dir']}/libero_spatial/*.hdf5"))
    out_dir = Path(cfg["data"]["augment_dir"]) / "libero_spatial"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = LiberoEnvRunner(suite="libero_spatial")
    tasks = range(runner.num_tasks()) if args.all_tasks else [args.task]
    total_saved = total_attempts = 0
    for task_id in tasks:
        saved, attempts = generate_for_task(
            runner, task_id, files, out_dir, args.target, args.max_steps
        )
        total_saved += saved
        total_attempts += attempts
    print(f"\ntotal: {total_saved} demos saved from {total_attempts} attempts -> {out_dir}")


if __name__ == "__main__":
    main()
