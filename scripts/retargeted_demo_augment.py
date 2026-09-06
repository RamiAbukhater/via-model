"""Generate synthetic demonstrations by retargeting a REAL, proven-successful
demo's trajectory to a new episode's (different) object/target start
positions, rather than hand-coding grasp geometry from scratch.

Why: `scripts/scripted_pick_place.py`'s from-scratch P-controller found
`akita_black_bowl`'s rim geometry (radius ~5cm, close to or exceeding the
gripper's span) makes a naive top-down centroid grasp a genuinely hard
precision problem -- exactly the kind of thing the *real* demos already
solve correctly (whatever subtle approach/height nuance makes the grasp
work), but that's expensive to rediscover by trial and error offset-tuning.

Instead: take a real demo's recorded `obs/ee_pos` trajectory (known to
succeed), and for a new episode, retarget it by the position offset between
the new episode's object/destination and the base demo's own object/
destination at reset -- linearly interpolated between "object offset"
(pre-grasp) and "destination offset" (post-grasp, once carrying), switched
on the base demo's own recorded gripper open/close timing. Track the
retargeted waypoints with closed-loop P-control (not open-loop replay) so
small per-step dynamics differences self-correct rather than compound.
Orientation and gripper actions are replayed directly from the base demo
(rotation and the open/close *timing* don't need position retargeting).

Nearest-neighbor base selection (found 2026-08-24, see
docs/EXPERIMENT_LOG.md): a single fixed base demo works well when the new
episode's object position is close to that demo's own, but degrades badly
for larger offsets -- some tasks' `demo_0` happens to sit at an unusual
corner of the position range, causing consistent large-offset failures.
Precomputing all 50 real demos' own (settled) object positions per task and
picking whichever is closest to each new episode keeps the retargeting
distance small, which is what actually matters for the pure-translation
approach to hold up physically.

    python -m scripts.retargeted_demo_augment --task 0 --episodes 8
"""

import argparse
import glob

import h5py
import numpy as np


class RetargetedController:
    def __init__(self, base_ee_pos, base_ori_delta, base_gripper, pos_gain: float = 25.0):
        self.base_ee_pos = base_ee_pos          # (T, 3)
        self.base_ori_delta = base_ori_delta     # (T, 3) raw action[3:6], replayed as-is
        self.base_gripper = base_gripper         # (T,) raw action[6]
        self.pos_gain = pos_gain
        self.T = base_ee_pos.shape[0]

        closed = base_gripper > 0
        self.t_close = int(np.argmax(closed)) if closed.any() else self.T
        opened_again = (~closed) & (np.arange(self.T) > self.t_close)
        self.t_release = int(np.argmax(opened_again)) if opened_again.any() else self.T
        self.t = 0

    def _offset(self, obj_offset: np.ndarray, dest_offset: np.ndarray) -> np.ndarray:
        if self.t < self.t_close:
            return obj_offset
        if self.t >= self.t_release or self.t_release <= self.t_close:
            return dest_offset
        frac = (self.t - self.t_close) / (self.t_release - self.t_close)
        return obj_offset + (dest_offset - obj_offset) * frac

    def act(self, obs: dict, obj_offset: np.ndarray, dest_offset: np.ndarray) -> tuple[np.ndarray, bool]:
        eef = obs["robot0_eef_pos"]
        t = min(self.t, self.T - 1)
        goal = self.base_ee_pos[t] + self._offset(obj_offset, dest_offset)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip((goal - eef) * self.pos_gain, -1.0, 1.0)
        action[3:6] = self.base_ori_delta[t]
        action[6] = self.base_gripper[t]
        self.t += 1
        return action, self.t >= self.T


def load_all_base_demos(env, path, target_obj: str, target_dest: str) -> list[dict]:
    """Every demo in `path`, with its own (settled) obj/dest position
    computed once via the direct physics-state-set method (see
    scripts/extract_object_state.py's `force_update=True` finding)."""
    demos = []
    with h5py.File(path, "r") as h5:
        for demo_key in h5["data"].keys():
            g = h5[f"data/{demo_key}"]
            init_state = g.attrs["init_state"]
            env.env.sim.set_state_from_flattened(init_state)
            env.env.sim.forward()
            obs = env.env._get_observations(force_update=True)
            demos.append({
                "ee_pos": g["obs/ee_pos"][:].astype(np.float32),
                "ori_delta": g["actions"][:, 3:6].astype(np.float32),
                "gripper": g["actions"][:, 6].astype(np.float32),
                "obj_pos": obs[f"{target_obj}_pos"].copy(),
                "dest_pos": obs[f"{target_dest}_pos"].copy(),
            })
    return demos


def nearest_demo(demos: list[dict], obj_pos: np.ndarray) -> dict:
    dists = [np.linalg.norm(d["obj_pos"] - obj_pos) for d in demos]
    return demos[int(np.argmin(dists))]


def find_demo_file_for_task(files, instruction: str):
    from pathlib import Path

    from via.data.libero import instruction_from_filename

    for f in files:
        if instruction_from_filename(Path(f)) == instruction:
            return f
    raise ValueError(f"no demo file matches instruction {instruction!r}")


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args()

    from train import common
    from via.data.libero import LiberoEnvRunner

    cfg = common.load_config("configs/local.yaml")
    files = sorted(glob.glob(f"{cfg['data']['libero_dir']}/libero_spatial/*.hdf5"))

    runner = LiberoEnvRunner(suite="libero_spatial")
    env, instruction = runner.make_env(args.task)
    env.reset()
    target_obj, target_dest = env.env.obj_of_interest

    # Match by instruction text, NOT alphabetical file order -- found
    # 2026-08-24 (see docs/EXPERIMENT_LOG.md) that sorted(glob(...))'s file
    # order does not match LiberoEnvRunner's internal task_id ordering
    # (tasks 2 and 4 confirmed cleanly swapped). Every other script in this
    # investigation only ever used LiberoEnvRunner's own task_id end to end
    # and was never exposed to this; it's specific to mixing the two here.
    demo_path = find_demo_file_for_task(files, instruction)
    demos = load_all_base_demos(env, demo_path, target_obj, target_dest)
    print(f"loaded {len(demos)} base demos for task {args.task}")

    settle_action = np.zeros(7, dtype=np.float32)
    settle_action[6] = -1.0  # gripper open

    successes = 0
    for ep in range(args.episodes):
        env.reset()
        # Objects start at a "drop" height right after reset and need a few
        # physics steps to settle under gravity before their position is
        # meaningful -- found 2026-08-24 (see docs/EXPERIMENT_LOG.md).
        for _ in range(10):
            obs, _, _, _ = env.step(settle_action)
        base = nearest_demo(demos, obs[f"{target_obj}_pos"])
        obj_offset = obs[f"{target_obj}_pos"] - base["obj_pos"]
        dest_offset = obs[f"{target_dest}_pos"] - base["dest_pos"]
        controller = RetargetedController(base["ee_pos"], base["ori_delta"], base["gripper"])
        for t in range(args.max_steps):
            action, done = controller.act(obs, obj_offset, dest_offset)
            obs, _, sim_done, info = env.step(action)
            if done or sim_done:
                break
        success = bool(env.env._check_success())
        successes += success
        print(f"task {args.task} ep {ep}: {'success' if success else 'FAIL'} "
              f"(obj_offset={obj_offset.round(3)}) at step {t}")
    env.close()
    print(f"\n{successes}/{args.episodes} succeeded")


if __name__ == "__main__":
    main()
