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

## 2026-07-22 — Belief milestone confirmed on the local CPU/synthetic path

**Status of DSMLP run 3b:** unconfirmed — the assistant session has no way to
authenticate to `dsmlp` (Duo push requires the phone), so it could not check
tmux/logs. Still open; needs a manual check.

**What ran instead:** `python -m train.train_belief --no-wandb` on
`configs/default.yaml` (stub encoders, `SyntheticTrajectoryDataset`, CPU) — 20
epochs, 640 steps, kl_weight 1e-2, occlude_p 0.5 (same fixes as the DSMLP
config, carried over). ~33 min wall clock. No optimization spikes; final mean
sigma ~0.6-0.65, KL ~185-195, NLL settling with the expected batch noise from
occluded windows.

**Calibration** (`eval/uncertainty_analysis.py`, synthetic held-out seed 999,
768 samples): **sigma occluded/visible ratio = 1.335** (occluded 0.64-0.78,
visible 0.53-0.56, clean separation, no floor collapse). This is the
Checkpoint-1 milestone from the README ("σ spikes on occlusion") — confirmed
on the offline path.

**Reading on this vs. the DSMLP runs:** this validates that the belief
network + occlusion-augmentation code (commit `28795ea`) produce
input-dependent variance when the training signal actually demands it — the
synthetic dataset's occlusion windows are a stronger, cleaner pressure than
LIBERO's augmented clips (denser occlusion frequency by construction), so a
ratio here doesn't guarantee the same on LIBERO, but it does confirm the
mechanism works and the runs 1-2 failure mode (flat 1.00 ratio) is fixed
where the data provides the pressure. The real read on LIBERO still depends
on run 3b's outcome.

---

## 2026-07-23 — World-model milestone fails on the belief checkpoint above:
## representation collapse in obs_embed, fixed, belief retrained twice more

**Symptom.** Trained `train.train_world_model` on the 07-22 belief checkpoint
(local CPU/synthetic path, 20 epochs). Training-time `rollout5_model_mse`
mostly beat `rollout5_naive_mse`, but a clean held-out check (64 synthetic
clips, seed 999, disjoint from training) told a different story: model beat
naive on only 26/64 clips (40.6%) despite winning on the batch mean. Digging
into individual clips: on clips with no occlusion in the 5-step window, naive
(persistence: predict obs_{t+5} = obs_t) scored MSE ~0, and the model
couldn't beat that trivial baseline at all.

**Root cause.** `belief_net.encode_obs` (the pooled observation embedding fed
to both the belief GRU and the world model) had collapsed to a near-constant
vector: across a 12-frame clip where a 32px square crosses the whole 224px
canvas, obs_embed moved <0.05 in norm while the underlying per-frame patch
embeddings moved by ~14 (confirmed directly — raw patches DO vary correctly
with position; only the pooling head's output doesn't). Mechanism: the
next-obs NLL target is `encode_obs`'s own output one step ahead, computed
under `no_grad` — nothing stops the trainable pooling head from satisfying
that objective by making its output ignore the input almost entirely. The
sigma-on-occlusion milestone still passed throughout because detecting
"blanked or not" needs far less precision than tracking continuous position,
so it never surfaced this.

**Fix, three iterations** (all in `via/belief/belief_state.py`,
`BeliefStateNetwork.loss`, `configs/{default,dsmlp}.yaml` `belief.var_weight`):
a VICReg-style (Bardes et al. 2022) variance-floor term, `var_weight *
relu(1 - std).mean()`, added to the belief loss to make collapse costly.

1. **Batch-wide floor** (std across all `B*T` samples per dim): satisfied
   for free via *between-clip* differences (color, start position) while
   staying flat *within* a clip — useless for the world model, which needs
   within-clip dynamics.
2. **Per-clip, across-time floor**, still including occluded frames:
   satisfied for free via the blank/reveal jump at an occlusion window's
   edges (a ~300-unit spike) while every genuinely-visible-but-moving frame
   stayed flat. Confirmed by inspecting step-to-step distances directly.
3. **Per-clip, across-time, visible-frames-only floor**: `occlude_clips`
   (train/train_belief.py) now returns its blanking mask; combined with the
   dataset's own occlusion flag (`SyntheticTrajectoryDataset`'s `occluded`
   key) and passed into `belief_net.loss(patches, occluded=mask)`, which
   excludes those timesteps from the variance statistic. This is the version
   that actually worked — occluded frames can no longer supply "free"
   variance, so only real motion can satisfy the floor.

**Weight tuning, also iterative.** `var_weight=1.0` and `var_weight=20.0`
both looked fine in *short, hand-rolled reimplementations* of the training
loop but **rebounded toward collapse partway through the real full run**
whenever an NLL optimization spike hit (same instability shape as the
kl_weight story above) — nothing held the line during the spike. Important
process note: the hand-rolled reimplementation is not a faithful stand-in
for `train.train_belief` — it diverges in RNG state (model-init order,
DataLoader worker seeding) and gave misleadingly stable numbers that did not
reproduce for real. Reliable trials have to go through the actual script
(a `--config` pointing at a copy with fewer epochs is a fine, cheap proxy).
`var_weight=100.0`, tested this way for 8 epochs, gave a steady,
non-rebounding descent and was confirmed on the full 20-epoch run.

**Final belief checkpoint** (var_weight=100, occlusion-excluded per-clip
temporal floor): `var_loss` settled ~0.3-0.55 (not fully to floor, but no
longer collapsed). Empirically: obs_embed now moves by tens of units across
a clip on non-occluded synthetic data (vs. <0.05 before), tracking real
position. Sigma-on-occlusion re-checked and still holds: ratio **1.219**
(was 1.335 pre-fix; some of that margin traded for real motion-sensitivity,
still comfortably > 1.0).

**World model retrained on the fixed belief checkpoint.** Held-out check (64
clips, seed 999): model beats naive on mean (18.28 vs 19.50), median (3.09
vs 3.30), **and** majority of clips (36/64, up from 26/64 pre-fix) — the
Checkpoint-1 world-model milestone from the README, now on a representation
that's actually earning it rather than free-riding on occlusion artifacts or
a weak baseline.

**Open thread:** this is all on the local CPU/synthetic path. DSMLP run 3b's
outcome is still unconfirmed (see 2026-07-22 entry), and the belief/RSSM
checkpoints trained on LIBERO will need the same var_weight fix — the
collapse mechanism is generic (any self-predictive NLL target with no
gradient-stopped anchor), not synthetic-data-specific, so it's likely
present on LIBERO too and worth checking directly rather than assuming.

---
