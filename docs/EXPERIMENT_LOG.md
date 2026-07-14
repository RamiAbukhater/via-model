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
