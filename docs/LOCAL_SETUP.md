# Local environment reference (Windows box, RTX 3060 Ti)

Quick-reference for continuing this project in a new session. For the full
narrative of *why* things are the way they are, see
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) — this file is just the "how do I get
back to a working state" checklist.

## Current status (as of 2026-09-21 — see docs/HANDOFF.md first)

**Start here, not in this file:** [`docs/HANDOFF.md`](HANDOFF.md) is the
up-to-date entry point for a new session — project state, the validated
paper result, and the critical checkpoint-mismatch caveat below are all
explained there in one place. This section is being kept as a secondary
summary but may lag; HANDOFF.md is the source of truth for "what's the
state right now."

All 4 original training stages (belief, world_model, goal, decision/CEM)
plus a newer 5th (`chunking` — `via/decision/chunking.py`, an action-chunking
policy) have been trained on real `libero_spatial` data at various points.
State representation includes proprioception (end-effector position +
gripper state) *and* target-object-relative position (`proprio_dim=8`).
UtilityHead trains via TD(0) against LIBERO's own sparse terminal reward.
Demo data was expanded via scripted trajectory retargeting/augmentation
(`scripts/generate_augmented_demos.py`) from 500 to several thousand demos.

**The project's one validated positive result:** the action-chunking
policy hit a reproducible (bit-for-bit, two independent runs) **10%
closed-loop success rate (3/30)** on `libero_spatial`, vs. ~1.7% pooled for
the tuned CEM+MLP pipeline — this is what's reported in the paper
(`documents/paper/via_sdutc_paper.tex`, outside this repo, not under git).

**Critical caveat — checkpoints on disk right now do NOT reproduce that
10%.** A later 4500-demo full-cascade retrain (2026-09-01/03) partially
overwrote `belief.pt` before being killed under deadline pressure, and the
recovery retrain built on top of it verified at 0/30, not 10%. See
`docs/EXPERIMENT_LOG.md`'s **2026-09-01/03** entry for the full incident,
and `docs/HANDOFF.md` for exactly which checkpoints are stale/mismatched
and what to do about it before trusting any eval run. Short version:
`utility.pt`/`gate.pt`/`action_prior.pt` (CEM pipeline) are stale
(2026-08-26, pre-crisis); `belief.pt`/`world_model.pt`/`goal.pt`/
`chunking_policy.pt` are mutually consistent but only verified at 0/30.

Live LIBERO simulator (robosuite/MuJoCo) working locally throughout.
Diminishing returns from CEM/ActionPrior tuning alone were confirmed
directly (bigger ActionPrior capacity made things *worse* — data-scarcity
overfitting) before the pivot to action-chunking, which is what actually
moved the needle. No more real demo data exists to download for this
suite (50/task is the complete official LIBERO-v1 release).

**GPU crash pattern, separate issue:** multiple "GPU is lost" crashes
across the project, root-caused to a daisy-chained PCIe power cable (see
`[[project-via-model-hardware]]` memory). **100W is the confirmed-reliable
power cap** (130W crashed repeatedly; 100W has run cleanly for the large
majority of this project's later training). A replacement PCIe cable was
obtained but, as of the last confirmed state, **not yet installed** — the
100W-cap routine below should be treated as still necessary until the
user explicitly confirms otherwise.

**Before doing anything else in a new session: check `git status` in
`via-model/`.** The repo was clean as of the last commit `4c91b13`
("Real-data closed-loop pipeline: 0/100 -> 10% via proprioception, TD(0)
value learning, CEM tuning, data augmentation, and action chunking"),
pushed to `origin/main`. Confirm nothing has drifted since.

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

The PC crashed multiple times across this project under sustained GPU load,
root-caused to a daisy-chained PCIe power cable (see EXPERIMENT_LOG.md's
08-11 entries for the full diagnosis with a hardware-troubleshooting AI).
130W was tried first and crashed repeatedly; **100W is the confirmed-reliable
cap**. This mitigation is in place **but does not survive a reboot or driver
reset** — check and reapply if training crashes mysteriously again or if the
PC has been restarted:

```powershell
# Check current cap (should read 100.00 W if still applied):
nvidia-smi --query-gpu=power.limit --format=csv

# Reapply if it's back to 200W (needs an elevated/Administrator PowerShell):
nvidia-smi -pl 100
```

Sleep-on-AC-power is disabled (`powercfg /change standby-timeout-ac 0`) —
this setting *does* persist across reboots, shouldn't need reapplying.

**Status of the actual fix:** a replacement PCIe power cable was obtained to
stop daisy-chaining two connectors off one wire run, but as of the last
confirmed state (user deferred the physical install: "I won't plug things in
right now, maybe later") it had **not** been installed. Confirm with the
user whether it's since gone in — if so, the 100W cap is likely safe to
remove (`nvidia-smi -pl 200` from an elevated terminal) and training should
run at full speed again; if not, keep the cap and sleep-disable in place.

## Uncommitted work

As of 2026-09-03 (commit `4c91b13`), everything was committed and pushed
to `origin/main` — including the action-chunking policy, the TD(0) utility
rewrite, data augmentation scripts, and all docs through that date. This
section (previously listing 2026-08-15/16 stragglers) is stale and kept
only as a reminder of the habit: **run `git status` in `via-model/` at the
start of a new session** rather than trusting this file's last-known state.
