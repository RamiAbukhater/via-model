"""LIBERO data access.

Two layers, so offline training never needs the simulator:

- `LiberoTrajectoryDataset` reads LIBERO's demonstration HDF5 files directly
  with h5py (structure: data/demo_K/{obs/agentview_rgb, actions, ...}).
  Needs only the dataset files + h5py. Yields fixed-length clips of
  (frames (T, 3, 224, 224) float [0,1], actions (T, 7), instruction str,
  progress (T,), proprio (T, 5) — end-effector position + gripper state,
  read from the same demo files' obs/ee_pos and obs/gripper_states) —
  everything the belief, world-model, and decision trainers consume.

- `LiberoEnvRunner` wraps the live simulator for closed-loop evaluation.
  Needs the full LIBERO install (robosuite/MuJoCo); import stays inside the
  class so the rest of the package works without it.

Download datasets with LIBERO's own script:
    python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
"""

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from via.contracts import C

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def instruction_from_filename(path: Path) -> str:
    """LIBERO encodes the task instruction in the demo filename, e.g.
    'pick_up_the_black_bowl..._demo.hdf5' -> 'pick up the black bowl ...'."""
    name = path.stem
    name = re.sub(r"_demo$", "", name)
    name = re.sub(r"^.*?SCENE\d+_", "", name)  # strip scene prefix if present
    return name.replace("_", " ").strip()


class LiberoTrajectoryDataset(Dataset):
    """Fixed-length clips sampled from LIBERO demonstration HDF5 files."""

    def __init__(
        self,
        data_dir: str | Path,
        clip_len: int = 16,
        camera: str = "agentview_rgb",
        suite_filter: str | None = None,
        split: str = "all",
        val_every: int = 5,
        object_state_dir: str | Path | None = None,
        augment_dir: str | Path | None = None,
    ):
        """`split`: "all" (no split, previous behavior), "train", or "val".

        Held out by *demo*, not by clip window: LIBERO clips within a demo
        overlap (stride clip_len // 2), so holding out clip windows instead
        of whole demos would leak — a "held-out" clip could share most of
        its frames with a training clip from the same demo. Split is
        per-task (every `val_every`-th demo index in *each* task's file) so
        val still covers every task's distribution rather than testing
        generalization to entirely unseen tasks, and is deterministic
        (demo index parity, not random) so train/val membership doesn't
        depend on load order or a seed staying fixed across code changes.

        `object_state_dir`, if given: directory of companion HDF5s (one per
        source demo file, matched by filename) holding a per-timestep
        target-object-relative position, produced once offline by
        scripts/extract_object_state.py. See via/contracts.py's proprio_dim.

        `augment_dir`, if given: directory of companion HDF5s holding
        synthetic demos from trajectory retargeting (one per real task file,
        matched by filename -- see scripts/generate_augmented_demos.py),
        merged in under the same task_id as their matching real file (never
        a new task_id) so goal_net's per-file classification target stays
        correct. Always assigned to split="train", never "val" -- held-out
        checks throughout this project measure generalization on real,
        human-collected data specifically; synthetic demos shouldn't dilute
        that. Self-contained (actions/rewards/obs/object_rel all inline in
        the augment file), unlike real demos which pull object_rel from
        `object_state_dir` separately.
        """
        import h5py  # local import: only needed when real data is used

        self.h5py = h5py
        self.clip_len = clip_len
        self.camera = camera
        self.object_state_dir = (
            Path(object_state_dir) / suite_filter
            if object_state_dir and suite_filter
            else (Path(object_state_dir) if object_state_dir else None)
        )
        augment_root = (
            Path(augment_dir) / suite_filter
            if augment_dir and suite_filter
            else (Path(augment_dir) if augment_dir else None)
        )

        data_dir = Path(data_dir)
        pattern = f"{suite_filter}/**/*.hdf5" if suite_filter else "**/*.hdf5"
        self.files = sorted(data_dir.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"no LIBERO hdf5 files under {data_dir}")
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be 'all', 'train', or 'val', got {split!r}")

        # Parallel to self.files: augmented companion file per task, if any.
        self.augment_files: list[Path | None] = [
            (augment_root / f.name) if augment_root and (augment_root / f.name).exists() else None
            for f in self.files
        ]

        # Index: (file_idx, demo_key, start_t, is_augment) for every valid
        # clip start.
        self.index: list[tuple[int, str, int, bool]] = []
        for fi, f in enumerate(self.files):
            with h5py.File(f, "r") as h5:
                for demo_key in h5["data"].keys():
                    demo_idx = int(demo_key.split("_")[1])
                    is_val = demo_idx % val_every == 0
                    if split == "train" and is_val:
                        continue
                    if split == "val" and not is_val:
                        continue
                    T = h5[f"data/{demo_key}/actions"].shape[0]
                    for start in range(0, max(1, T - clip_len + 1), clip_len // 2):
                        if start + clip_len <= T:
                            self.index.append((fi, demo_key, start, False))
            aug_f = self.augment_files[fi]
            if aug_f is not None and split in ("all", "train"):
                with h5py.File(aug_f, "r") as h5:
                    for demo_key in h5["data"].keys():
                        T = h5[f"data/{demo_key}/actions"].shape[0]
                        for start in range(0, max(1, T - clip_len + 1), clip_len // 2):
                            if start + clip_len <= T:
                                self.index.append((fi, demo_key, start, True))
        if not self.index:
            raise ValueError(f"split={split!r} produced an empty dataset under {data_dir}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        fi, demo_key, start, is_augment = self.index[i]
        path = self.augment_files[fi] if is_augment else self.files[fi]
        end = start + self.clip_len
        with self.h5py.File(path, "r") as h5:
            g = h5[f"data/{demo_key}"]
            rgb = torch.from_numpy(g[f"obs/{self.camera}"][start:end])   # (T, H, W, 3) uint8
            actions = torch.from_numpy(g["actions"][start:end]).float()  # (T, 7)
            ee_pos = torch.from_numpy(g["obs/ee_pos"][start:end]).float()          # (T, 3)
            gripper = torch.from_numpy(g["obs/gripper_states"][start:end]).float()  # (T, 2)
            reward = torch.from_numpy(g["rewards"][start:end]).float()  # (T,) sparse: 1 only at true demo end
            total = g["actions"].shape[0]
            if is_augment:
                # Self-contained: object_rel is stored inline, not in a
                # separate object_state_dir companion (see
                # scripts/generate_augmented_demos.py).
                obj_rel = torch.from_numpy(g["object_rel"][start:end]).float()

        proprio_parts = [ee_pos, gripper]
        if is_augment:
            proprio_parts.append(obj_rel)
        elif self.object_state_dir is not None:
            obj_path = self.object_state_dir / self.files[fi].name
            with self.h5py.File(obj_path, "r") as obj_h5:
                obj_rel = torch.from_numpy(
                    obj_h5[f"data/{demo_key}/object_rel"][start:end]
                ).float()  # (T, 3)
            proprio_parts.append(obj_rel)

        frames = rgb.permute(0, 3, 1, 2).float() / 255.0
        if frames.shape[-1] != C.image_size:
            frames = F.interpolate(
                frames, size=(C.image_size, C.image_size), mode="bilinear", align_corners=False
            )
        proprio = torch.cat(proprio_parts, dim=-1)   # (T, 5 or 8) — see C.proprio_dim
        # Monotone progress toward demo completion — the utility head's
        # regression target (demos are successful by construction).
        progress = torch.arange(start, end, dtype=torch.float32) / max(total - 1, 1)
        return {
            "frames": frames,                                # (T, 3, 224, 224)
            "actions": actions,                              # (T, 7)
            "instruction": instruction_from_filename(path),
            "progress": progress,                            # (T,)
            "reward": reward,                                # (T,) sparse terminal reward
            "proprio": proprio,                              # (T, C.proprio_dim)
            # File index as a task id: each LIBERO demo file is one task, so
            # this is a stable (if coarse — one label per task, not per
            # instruction phrasing) goal-slot target for real-data goal
            # inference training. See train/train_goal.py.
            "task_id": fi,
        }


def clamp_object_rel(vec: np.ndarray, max_norm: float = 0.5) -> np.ndarray:
    """Clamp a target-object-relative position vector's norm to roughly the
    range ever observed in training demos (measured max 0.487, see
    scripts/extract_object_state.py's output).

    Found 2026-08-2x (see docs/EXPERIMENT_LOG.md): live closed-loop episodes
    can drift the gripper-to-target distance well past 1.0 once the policy
    makes even a small early mistake, since all 500 training demos are
    *successful* and never show a large, growing distance -- the network was
    extrapolating into a region of object_rel-space it had literally never
    seen, plausibly compounding rather than correcting the drift. Preserves
    direction, only rescales magnitude, so the network still gets a "which
    way is the target" signal even far away, just bounded to where it has
    actually been trained.
    """
    norm = np.linalg.norm(vec)
    if norm <= max_norm or norm == 0:
        return vec
    return vec * (max_norm / norm)


class LiberoEnvRunner:
    """Closed-loop episode runner on the live LIBERO benchmark (GPU/eval box)."""

    def __init__(self, suite: str = "libero_spatial", camera_size: int = C.image_size):
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        self._env_cls = OffScreenRenderEnv
        self.camera_size = camera_size
        self.suite = benchmark.get_benchmark_dict()[suite]()

    def num_tasks(self) -> int:
        return self.suite.n_tasks

    def make_env(self, task_id: int):
        task = self.suite.get_task(task_id)
        env = self._env_cls(
            bddl_file_name=self.suite.get_task_bddl_file_path(task_id),
            camera_heights=self.camera_size,
            camera_widths=self.camera_size,
        )
        return env, task.language

    def run_episode(self, model, task_id: int, max_steps: int = 300, device: str = "cuda") -> dict:
        """Roll one episode of `model` (a VIAModel) on task `task_id`."""
        env, instruction = self.make_env(task_id)
        try:
            env.reset()
            target = env.env.obj_of_interest[0]
            obs = env.env._get_observations()
            state = model.reset([instruction], device=torch.device(device))
            diags = []
            for _ in range(max_steps):
                frame = (
                    torch.from_numpy(obs["agentview_image"].copy())
                    .permute(2, 0, 1).float().unsqueeze(0) / 255.0
                ).to(device)
                # robosuite's standard proprioception keys -- the same
                # underlying quantities the offline demo HDF5s save as
                # obs/ee_pos and obs/gripper_states (robomimic dataset
                # convention), so training and live inference see the same
                # proprio layout. `<target>_to_robot0_eef_pos` is the live
                # counterpart of scripts/extract_object_state.py's offline
                # replay -- see via/contracts.py's proprio_dim.
                proprio = torch.from_numpy(
                    np.concatenate([
                        obs["robot0_eef_pos"], obs["robot0_gripper_qpos"],
                        clamp_object_rel(obs[f"{target}_to_robot0_eef_pos"]),
                    ])
                ).float().unsqueeze(0).to(device)
                action, state, d = model.act(frame, proprio, state)
                obs, _, done, info = env.step(action[0].cpu().numpy())
                diags.append({k: v.detach().cpu() for k, v in d.items() if v.dim() <= 1})
                if done:
                    break
            success = bool(env.env._check_success())
        finally:
            env.close()
        return {"success": success, "instruction": instruction, "diagnostics": diags}
