"""One-time offline preprocessing: extract target-object-relative position
per timestep for every LIBERO demo, via direct physics-state replay through
the live simulator (no action-stepping, no drift risk -- each demo's
recorded `states` array is a full flattened MuJoCo state snapshot per
timestep; setting the sim to that exact state and reading back an
observable is exact, not approximate).

Motivation (see docs/EXPERIMENT_LOG.md, 2026-08-21): proprioception (end-
effector position + gripper state) alone didn't move closed-loop success --
it tells the model where its own gripper is, but not where the target
object is, which a coarse pooled visual embedding struggles to localize
precisely. LIBERO's live sim exposes `<object>_to_robot0_eef_pos` directly
per-task via `obj_of_interest`, but the offline demo HDF5s only store raw,
unlabeled full physics state -- this script recovers the same signal
offline by replaying that state through the same live env used for eval.

Writes one companion HDF5 per source demo file, same directory structure,
under a sibling `object_state_dir`, with `data/demo_K/object_rel`
(T, 3) datasets aligned 1:1 with the source file's `data/demo_K/actions`.

    python -m scripts.extract_object_state --config configs/local.yaml
"""

import h5py
import numpy as np

from train import common
from via.data.libero import LiberoEnvRunner


def main() -> None:
    p = common.base_parser(__doc__)
    args = p.parse_args()
    cfg = common.load_config(args.config)

    import pathlib

    libero_dir = cfg["data"]["libero_dir"]
    suite = cfg["data"].get("suite") or "libero_spatial"
    src_dir = pathlib.Path(libero_dir) / suite
    out_dir = pathlib.Path(cfg["data"]["object_state_dir"]) / suite
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no LIBERO hdf5 files under {src_dir}")

    runner = LiberoEnvRunner(suite=suite)

    for task_id, path in enumerate(files):
        env, instruction = runner.make_env(task_id)
        env.reset()
        raw = env.env
        target = raw.obj_of_interest[0]

        src = h5py.File(path, "r")
        demo_keys = list(src["data"].keys())
        out_path = out_dir / path.name
        with h5py.File(out_path, "w") as dst:
            dst.attrs["target_object"] = target
            dst.attrs["instruction"] = instruction
            for demo_key in demo_keys:
                states = src[f"data/{demo_key}/states"][:]
                T = states.shape[0]
                obj_rel = np.zeros((T, 3), dtype=np.float32)
                for t in range(T):
                    raw.sim.set_state_from_flattened(states[t])
                    raw.sim.forward()
                    obs = raw._get_observations(force_update=True)
                    obj_rel[t] = obs[f"{target}_to_robot0_eef_pos"]
                dst.create_dataset(f"data/{demo_key}/object_rel", data=obj_rel)
        src.close()
        env.close()
        print(f"task {task_id} ({target}, {len(demo_keys)} demos): wrote {out_path}")

    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
