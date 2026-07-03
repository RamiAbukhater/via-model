"""LIBERO data access.

Two layers, so offline training never needs the simulator:

- `LiberoTrajectoryDataset` reads LIBERO's demonstration HDF5 files directly
  with h5py (structure: data/demo_K/{obs/agentview_rgb, actions, ...}).
  Needs only the dataset files + h5py. Yields fixed-length clips of
  (frames (T, 3, 224, 224) float [0,1], actions (T, 7), instruction str,
  progress (T,)) — everything the belief, world-model, and decision trainers
  consume.

- `LiberoEnvRunner` wraps the live simulator for closed-loop evaluation.
  Needs the full LIBERO install (robosuite/MuJoCo); import stays inside the
  class so the rest of the package works without it.

Download datasets with LIBERO's own script:
    python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
"""

import re
from pathlib import Path

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
    ):
        import h5py  # local import: only needed when real data is used

        self.h5py = h5py
        self.clip_len = clip_len
        self.camera = camera

        data_dir = Path(data_dir)
        pattern = f"{suite_filter}/**/*.hdf5" if suite_filter else "**/*.hdf5"
        self.files = sorted(data_dir.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"no LIBERO hdf5 files under {data_dir}")

        # Index: (file_idx, demo_key, start_t) for every valid clip start.
        self.index: list[tuple[int, str, int]] = []
        for fi, f in enumerate(self.files):
            with h5py.File(f, "r") as h5:
                for demo_key in h5["data"].keys():
                    T = h5[f"data/{demo_key}/actions"].shape[0]
                    for start in range(0, max(1, T - clip_len + 1), clip_len // 2):
                        if start + clip_len <= T:
                            self.index.append((fi, demo_key, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        fi, demo_key, start = self.index[i]
        path = self.files[fi]
        end = start + self.clip_len
        with self.h5py.File(path, "r") as h5:
            g = h5[f"data/{demo_key}"]
            rgb = torch.from_numpy(g[f"obs/{self.camera}"][start:end])   # (T, H, W, 3) uint8
            actions = torch.from_numpy(g["actions"][start:end]).float()  # (T, 7)
            total = g["actions"].shape[0]

        frames = rgb.permute(0, 3, 1, 2).float() / 255.0
        if frames.shape[-1] != C.image_size:
            frames = F.interpolate(
                frames, size=(C.image_size, C.image_size), mode="bilinear", align_corners=False
            )
        # Monotone progress toward demo completion — the utility head's
        # regression target (demos are successful by construction).
        progress = torch.arange(start, end, dtype=torch.float32) / max(total - 1, 1)
        return {
            "frames": frames,                                # (T, 3, 224, 224)
            "actions": actions,                              # (T, 7)
            "instruction": instruction_from_filename(path),
            "progress": progress,                            # (T,)
        }


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
            obs = env.env._get_observations()
            state = model.reset([instruction], device=torch.device(device))
            diags = []
            for _ in range(max_steps):
                frame = (
                    torch.from_numpy(obs["agentview_image"].copy())
                    .permute(2, 0, 1).float().unsqueeze(0) / 255.0
                ).to(device)
                action, state, d = model.act(frame, state)
                obs, _, done, info = env.step(action[0].cpu().numpy())
                diags.append({k: v.detach().cpu() for k, v in d.items() if v.dim() <= 1})
                if done:
                    break
            success = bool(env.env._check_success())
        finally:
            env.close()
        return {"success": success, "instruction": instruction, "diagnostics": diags}
