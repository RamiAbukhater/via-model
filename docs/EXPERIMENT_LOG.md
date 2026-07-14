# VIA Experiment Log

Running lab notebook. One entry per training run or significant infrastructure
event; newest last. Numbers here feed the checkpoint reports and the paper.

---

## 2026-07-13 — Environment bring-up on UCSD DSMLP

**Platform:** DSMLP GPU pods (`launch-scipy-ml.sh -g 1 -c 8 -m 32 -v a5000`,
RTX A5000 24 GB), venv in home directory, training driven by
`hpc/dsmlp_train.sh` inside tmux on the login node.

Issues hit and resolved, in order:

1. **Home disk quota exceeded** during Phi-3 (7.6 GB) and LIBERO dataset
   downloads. Resolution: datasets moved to pod-local `/tmp/via-libero`
   (outside quota, ephemeral; auto re-downloaded per pod by the train
   wrapper, ~40 s on the campus network). Phi-3 deferred — not needed until
   the goal-inference stage. Quota increase requested from datahub support.
2. **`torch.cuda` driver mismatch**: default pip torch (cu126+) requires a
   newer driver than the nodes' CUDA 12.2. Resolution: pinned
   `torch==2.5.1 --index-url .../whl/cu121` (works on the 12.2 driver;
   satisfies transformers' >=2.4 requirement).
3. SigLIP load prints a wall of `UNEXPECTED` text-tower keys — expected and
   harmless; we load only the vision tower from the joint checkpoint.

**Data:** `libero_spatial` suite (2.88 GB, ~432 human demos, HDF5), cut into
16-frame clips, stride 8. Instruction text parsed from filenames.

---

## 2026-07-13 — Belief run 1: variance collapse (kl_weight = 1e-3)

**Config:** `configs/dsmlp.yaml` — lr 3e-4, batch 16, 20 epochs, clip_len 16,
kl_weight **1e-3**. Frozen SigLIP ViT-B/16 perception; belief GRU trained
with next-observation Gaussian NLL + KL(belief ‖ N(0, I)).

**Course of training (~9k steps):**
- Steps 0–2.3k: healthy. NLL fell after the usual early transient;
  mean sigma descended from 1.0 and hovered ~0.1; KL plateaued ~2,300.
- Step ~2.3k: **optimization spike** — loss/NLL briefly hit ~3.5M. The run
  recovered, but into a different regime: KL jumped to ~7,400 and stayed
  there; mean sigma ground down to **0.028**, essentially the logvar clamp
  floor (sigma_min ≈ 0.018).

**Calibration probe** (`eval/uncertainty_analysis.py`, in-domain variant:
3-frame blanked windows inside held-out-seed LIBERO clips):
- **sigma occluded/visible ratio = 1.00** — the variance is flat regardless
  of observability. Variance collapse confirmed: the network predicts fine
  but no longer reports uncertainty, which defeats the belief-state design
  (the sigma-on-occlusion result, information-gain term, and adaptive lambda
  gate all consume that variance).

**Diagnosis:** the KL term at weight 1e-3 contributed only ~7 of a ~1,900
loss — too weak a leash. When the step-2.3k instability pushed the posterior
into a high-magnitude latent regime (KL 2,300 → 7,400), nothing pulled it
back, and the NLL objective then favored grinding sigma to the floor.

**Intervention:** `kl_weight` 1e-3 → **1e-2** (commit `1ec77d8`). Collapsed
checkpoint preserved as `belief_collapsed_kl1e-3.pt` for a before/after
calibration comparison figure.

**Expected signatures of a healthy retrain:** KL stabilizing well below
7,400 (hundreds to ~2,000); mean sigma settling in ~0.1–0.4, clearly off the
floor; occluded/visible ratio meaningfully > 1.0. Escalation if still
collapsed: kl_weight 1e-1, at the risk of the opposite failure (sigma pinned
near 1, underfit).

**Status:** retrain launched 2026-07-13 evening. Ratio verdict pending.

---

## 2026-07-14 — Belief run 2: ratio still 1.00 → root cause is the data, not the objective

**Config:** as run 1 but kl_weight **1e-2**. Trained to completion overnight.

**Result:** occluded/visible sigma ratio = **1.000** again.

**Revised diagnosis.** Two identical ratio failures under different KL
weights rule out the optimization story as the root cause. The real issue:
**LIBERO demonstrations contain no occlusions** — every next observation in
training is predictable, so the NLL objective never once pressured sigma to
rise, and the variance head had no reason to become input-dependent. The
kl_weight bump addressed the sigma-floor symptom (run 1's collapse) but
cannot create sensitivity to observability that the data never demanded.
(The synthetic trajectory dataset was designed with occlusion windows for
exactly this reason; the LIBERO clips lack the equivalent pressure.)

**Intervention: occlusion augmentation (sensor dropout).** During belief
training, each clip has probability `occlude_p = 0.5` of a random 2-4 frame
window blanked to zeros (`occlude_clips` in `train/train_belief.py`). The
network now repeatedly experiences going blind and is penalized (NLL at the
reveal frames) for remaining confident through it — direct training pressure
for sigma to rise under occlusion, and it makes the evaluation probe
in-distribution.

**Expected outcome:** run 3 ratio clearly > 1.0. If mean sigma is healthy
(0.1-0.4) but the ratio is still ~1.0 even with augmentation, next suspects
are the variance head's capacity/gradient path rather than the data.

**Run 2 closing numbers** (recovered from the tmux scrollback): completed
all 20 epochs (~8.8k steps); final mean sigma 0.058, KL ~5,900, NLL ~2.4k.
The kl_weight bump did lift sigma off the floor (0.028 → 0.058) but not into
the healthy band — consistent with the revised diagnosis that the data,
not the objective weight, was the binding constraint.

---

## 2026-07-14 — Run 3 (augmented): healthy training, killed by infrastructure

**Config:** kl_weight 1e-2 + occlude_p 0.5 (first augmented run), from scratch.

**Observed:** NLL ~15k-53k and batch-noisy — *expected*, not a divergence:
clips with blanked windows are genuinely hard to predict, and batches vary
in how many they draw. mean sigma settled ~0.15, KL ~3,000. All healthy.

**Death at epoch 6/20 (11:23):** the run was launched outside tmux; the ssh
connection dropped and took the process with it (no traceback — clean stop
after the epoch-6 checkpoint save). The 0713 overnight pod's scary
`exit code 137 / CompletedDeadlineExceeded` message was unrelated: run 2 had
already finished; the idle pod simply hit its 6 h deadline at 2 AM.

**Action:** resumed from the epoch-6 checkpoint inside tmux (run 3b, log
`0714-1645`, W&B run `occw0oia`). Note: resume restores weights but not the
Adam state; early post-resume NLL is elevated (~200k) and should settle —
under monitoring.

**Process change:** training babysitting is now automated — a recurring
monitor reads the DSMLP logs over ssh (ControlMaster), restarts dead runs
inside tmux, applies diagnosed fixes (committed + logged), and runs the
calibration eval on completion. Escalation rule: same failure surviving two
distinct fixes → stop and consult.

---
