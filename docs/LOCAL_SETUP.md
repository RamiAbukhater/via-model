# Local environment reference (Windows box, RTX 3060 Ti)

Quick-reference for continuing this project in a new session. For the full
narrative of *why* things are the way they are, see
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) — this file is just the "how do I get
back to a working state" checklist.

## Current status (as of 2026-08-23)

All 4 training stages (belief, world_model, goal, decision) trained and
validated on real `libero_spatial` data. State representation now includes
proprioception (end-effector position + gripper state) *and*
target-object-relative position (`proprio_dim=8` — via
`scripts/extract_object_state.py`'s offline physics-replay preprocessing,
companion HDF5s in `D:/via-data/libero_object_state/`). UtilityHead trains
via TD(0) against LIBERO's own sparse terminal reward, not the original
`progress = t/T` regression target. CEM's default search radius
(`init_std`) is tightened to 0.1 (from 0.5) — found to matter a lot, see
EXPERIMENT_LOG.md's 2026-08-21/23 entry for the full story and why each
piece is there. Checkpoints in `D:\via-data\checkpoints\` (`belief.pt`,
`world_model.pt`, `goal.pt`, `utility.pt`, `gate.pt`, `action_prior.pt`;
`goal_synthetic_only.pt` is a backup from a much earlier fine-tune,
effectively stale now). **Pre-2026-08-21 checkpoints no longer exist** —
overwritten in place across multiple retrains with no backup taken first;
there's no clean rollback to any earlier system state without retraining
from scratch.

Live LIBERO simulator (robosuite/MuJoCo) working locally. Closed-loop
`eval/eval_libero.py` / `eval/ablations.py` full-scale (30 episodes/variant)
runs at the current best-known configuration: still 0/30 across every
variant, but gripper-to-target end-of-episode distance is roughly halved
versus the original pre-Stage-2 baseline (~0.5-0.6 -> ~0.3-0.4, noisy
across runs) via three independently-verified fixes (object-relative
state, TD value target, tightened CEM search). Diminishing returns from
further configuration tuning were confirmed directly (bigger ActionPrior
capacity made things *worse* — data-scarcity overfitting, not undertrained)
and no more real demo data exists to download for this suite (50/task is
the complete official LIBERO-v1 release, confirmed via the dataset's own
metadata). Remaining paths are qualitatively bigger investments (more
data, or a different policy architecture class) — see EXPERIMENT_LOG.md's
closing summary.

**GPU crash pattern, separate issue:** seven "GPU is lost" crashes total
across the 2026-08-18/23 sessions, each needing a full reboot. Multiple
confirmed at a freshly-verified 130W cap; the only fully clean training
completions have been at 100W. See `[[project-via-model-hardware]]`
memory — 100W is the current working assumption, not a proven-safe
conclusion. User's explicit direction after the pattern repeated: keep
training at 100W and treat it as tolerable rather than pausing to
physically investigate the cable further.

**Before doing anything else in a new session: check `git status` in
`via-model/`.** As of this writing there are uncommitted changes covering
essentially this entire session's work (see bottom of this file) — confirm
whether they've since been committed before assuming the working tree
matches git history.

## Python environment

Venv lives on **D:**, not alongside the repo (C: has almost no free space —
was ~12GB free / 98% full at the start of this work, watch it):

```
D:\via-data\envs\via\Scripts\python.exe
```

Activate it directly by calling that interpreter, or `Scripts\activate` in
a Windows shell. `via-model` itself (the repo) is installed editable into
this venv, as is `libero` (see below) — no need to reinstall either unless
the venv is rebuilt from scratch.

**Always set `HF_HOME=D:/via-data/hf`** before running anything that touches
HuggingFace models (SigLIP), or it'll default to `C:` and eat into the
already-tight free space there:

```bash
HF_HOME=/d/via-data/hf "/d/via-data/envs/via/Scripts/python.exe" -m train.train_belief --config configs/local.yaml
```

Config to use for all real-data work: **`configs/local.yaml`** (not
`default.yaml` — that's the offline/synthetic/stub config, still used for
`--smoke` runs and the original CPU/synthetic milestone path). `local.yaml`
points at real SigLIP, real `libero_spatial`, and D:-drive paths throughout.

If pip's wheel cache fills up C: again (`pip cache purge` if `df -h /c`
looks tight — it's happened twice already this session, ~5-6GB each time),
that's just a download cache, safe to clear anytime.

## LIBERO / robosuite / MuJoCo (Windows-specific — only needed if rebuilding the venv)

Real LIBERO data (`D:\via-data\libero\libero_spatial\*.hdf5`, 500 demos)
and the LIBERO code repo (`D:\via-data\LIBERO`) are **the same directory**
on disk — Windows filesystems are case-insensitive, so `LIBERO` and `libero`
collided. Harmless in practice, but don't be surprised finding both dataset
files and repo code under one folder.

The `libero` Python package is **not** pip-installed editable (that broke —
empty package-discovery mapping, likely confused by the dataset files
sitting in the same directory). Instead there's a plain `.pth` file:

```
D:\via-data\envs\via\Lib\site-packages\libero_repo.pth
  -> contains the single line: D:\via-data\LIBERO
```

LIBERO's own config lives at `C:\Users\super\.libero\config.yaml`, hand-written
to point at the D: paths (avoids an interactive first-run prompt LIBERO
would otherwise ask). If this file goes missing, LIBERO's `__init__.py` will
try to prompt interactively on next import — recreate it:

```yaml
benchmark_root: D:\via-data\LIBERO\libero\libero
bddl_files: D:\via-data\LIBERO\libero\libero\bddl_files
init_states: D:\via-data\LIBERO\libero\libero\init_files
datasets: D:\via-data\libero
assets: D:\via-data\LIBERO\libero\libero\assets
```

**Three real Windows-incompatibilities in `robosuite==1.4.1`, all patched
directly in the installed package** (`D:\via-data\envs\via\Lib\site-packages\robosuite\`)
— these do **not** live in the `via-model` repo and will need reapplying if
the venv is ever rebuilt:

1. `robosuite/utils/binding_utils.py` ctypes-loads a `mujoco.dll` from its
   own package dir on Windows; the pip wheel doesn't ship one. Fix: copy
   the real one over —
   `cp .../site-packages/mujoco/mujoco.dll .../site-packages/robosuite/utils/mujoco.dll`
2. `robosuite/macros.py` hardcodes `MUJOCO_GPU_RENDERING = True`, which maps
   to `MUJOCO_GL=egl` (Linux-only) for any non-Darwin system. Patched to
   `MUJOCO_GPU_RENDERING = platform.system() != "Windows"` so it falls
   through to GLFW instead.
3. `mujoco` must be pinned to **`3.2.6`** specifically — it's the earliest
   version with a Python 3.13 wheel *and* still has the `qM` attribute
   robosuite 1.4.1's controller code calls (`mj_fullM(..., data.qM)`); newer
   mujoco renamed it to `.M`, which breaks robosuite's controller init.
   `pip install mujoco==3.2.6` (after installing robosuite, which will pull
   a newer mujoco by default — downgrade it back down afterward).

Full diagnostic trail (why each of these, how they were found) is in
EXPERIMENT_LOG.md's 08-14 entry if anything related breaks in a new way.

## Hardware — GPU power cap and sleep (safety-relevant, please read)

The PC crashed twice early in this project under sustained GPU load,
root-caused to a daisy-chained PCIe power cable (see EXPERIMENT_LOG.md's
08-11 entries for the full diagnosis with a hardware-troubleshooting AI).
Two mitigations are in place **but the power cap does not survive a reboot
or driver reset** — check and reapply if training crashes mysteriously
again or if the PC has been restarted:

```powershell
# Check current cap (should read 130.00 W if still applied):
nvidia-smi --query-gpu=power.limit --format=csv

# Reapply if it's back to 200W (needs an elevated/Administrator PowerShell):
nvidia-smi -pl 130
```

Sleep-on-AC-power is disabled (`powercfg /change standby-timeout-ac 0`) —
this setting *does* persist across reboots, shouldn't need reapplying.

**Status of the actual fix:** a replacement PCIe power cable was ordered to
stop daisy-chaining two connectors off one wire run. Confirm with the user
whether it's arrived/been installed — if so, the 130W cap is likely safe to
remove (`nvidia-smi -pl 200` from an elevated terminal) and training should
run at full speed again; if not, keep the cap and sleep-disable in place.

## Uncommitted work (as of 2026-08-15/16 — check `git status` to see if this is still true)

The following were modified/added this session and were **not yet
committed** when this doc was written:

```
modified: docs/EXPERIMENT_LOG.md, eval/eval_libero.py, eval/uncertainty_analysis.py,
          train/common.py, train/train_belief.py, train/train_decision.py,
          train/train_goal.py, train/train_world_model.py,
          via/belief/belief_state.py, via/data/libero.py,
          via/decision/__init__.py, via/decision/decision.py,
          via/world_model/rssm.py
untracked: configs/local.yaml
```

This covers: the `obs_embed` LayerNorm fix, the train/val split in
`LiberoTrajectoryDataset`, all the held-out-check utilities in
`train/common.py`, the real-data goal fine-tuning path, `ActionPrior` +
score normalization + counterfactual utility training in the decision
module, and the `eval_libero.py --smoke` bug fix. If starting a new session
and these still show as uncommitted, that's real, valuable work sitting
only in the working tree — worth committing before doing anything that
could disturb it.
