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
   downloads. Datasets moved to pod-local `/tmp/via-libero` (outside quota,
   ephemeral, auto re-downloaded per pod by the train wrapper, ~40s on
   campus network). Phi-3 deferred until the goal-inference stage. Quota
   increase requested from datahub support.
2. **`torch.cuda` driver mismatch**: default pip torch (cu126+) needs a
   newer driver than the nodes' CUDA 12.2. Pinned
   `torch==2.5.1 --index-url .../whl/cu121`.
3. SigLIP load prints a wall of `UNEXPECTED` text-tower keys — expected,
   we only load the vision tower from the joint checkpoint.

**Data:** `libero_spatial` suite (2.88 GB, ~432 human demos, HDF5), cut into
16-frame clips, stride 8. Instruction text parsed from filenames.

---

## 2026-07-13 — Belief run 1: variance collapse (kl_weight = 1e-3)

**Config:** `configs/dsmlp.yaml` — lr 3e-4, batch 16, 20 epochs, clip_len 16,
kl_weight 1e-3. Frozen SigLIP ViT-B/16 perception; belief GRU trained with
next-observation Gaussian NLL + KL(belief ‖ N(0, I)).

**Course of training (~9k steps):** steps 0-2.3k healthy (NLL fell after the
usual early transient, mean sigma descended from 1.0 to ~0.1, KL plateaued
~2,300). Around step 2.3k an optimization spike hit (loss/NLL briefly ~3.5M).
The run recovered but into a worse regime: KL jumped to ~7,400 and stayed
there, mean sigma ground down to 0.028 — essentially the logvar clamp floor
(sigma_min ≈ 0.018).

**Calibration probe** (`eval/uncertainty_analysis.py`, 3-frame blanked
windows in held-out LIBERO clips): sigma occluded/visible ratio = 1.00. Flat
regardless of observability — variance collapse. The network predicts fine
but no longer reports uncertainty, which breaks the belief-state design
(sigma-on-occlusion, the information-gain term, and the adaptive gate all
depend on that variance).

**Diagnosis:** kl_weight 1e-3 contributed only ~7 of a ~1,900 loss, too weak
a leash. When the step-2.3k spike pushed KL from 2,300 to 7,400, nothing
pulled it back, and the NLL objective then favored grinding sigma to the
floor.

**Fix:** kl_weight 1e-3 → 1e-2 (commit `1ec77d8`). Collapsed checkpoint kept
as `belief_collapsed_kl1e-3.pt` for a before/after comparison.

**Expected for a healthy retrain:** KL stabilizing well below 7,400
(hundreds to ~2,000), mean sigma in ~0.1-0.4, ratio meaningfully > 1.0.
Escalation if still collapsed: kl_weight 1e-1, risking the opposite failure
(sigma pinned near 1, underfit).

**Status:** retrain launched 2026-07-13 evening.

---

## 2026-07-14 — Belief run 2: ratio still 1.00, root cause is the data

**Config:** as run 1 but kl_weight 1e-2. Trained to completion overnight.

**Result:** occluded/visible sigma ratio = 1.000 again.

**Revised diagnosis:** two identical ratio failures under different KL
weights rule out optimization as the root cause. LIBERO demonstrations
contain no occlusions — every next observation in training is predictable,
so the NLL never pressures sigma to rise. The kl_weight bump fixed run 1's
floor collapse but can't create sensitivity to observability the data never
demanded.

**Fix: occlusion augmentation (sensor dropout).** Each training clip has
probability `occlude_p = 0.5` of a random 2-4 frame window blanked to zeros
(`occlude_clips` in `train/train_belief.py`). The network now repeatedly
goes blind and is penalized at the reveal frames for staying confident
through it.

**Expected:** run 3 ratio clearly > 1.0. If mean sigma is healthy (0.1-0.4)
but the ratio is still ~1.0 with augmentation on, next suspects are the
variance head's capacity/gradient path, not the data.

**Run 2 closing numbers** (from tmux scrollback): completed all 20 epochs
(~8.8k steps), final mean sigma 0.058, KL ~5,900, NLL ~2.4k. Sigma lifted
off the floor (0.028 → 0.058) but not into the healthy band, consistent with
the data being the binding constraint.

---

## 2026-07-14 — Run 3 (augmented): healthy training, killed by infrastructure

**Config:** kl_weight 1e-2 + occlude_p 0.5, from scratch.

**Observed:** NLL ~15k-53k and batch-noisy, expected rather than a
divergence — clips with blanked windows are genuinely hard to predict, and
batches vary in how many they draw. Mean sigma settled ~0.15, KL ~3,000.
Healthy.

**Death at epoch 6/20 (11:23):** launched outside tmux; the ssh connection
dropped and took the process with it (clean stop after the epoch-6
checkpoint save, no traceback). The 0713 overnight pod's
`exit code 137 / CompletedDeadlineExceeded` was unrelated — run 2 had
already finished, the idle pod just hit its 6h deadline at 2am.

**Action:** resumed from the epoch-6 checkpoint inside tmux (run 3b, log
`0714-1645`, W&B run `occw0oia`). Resume restores weights but not the Adam
state; early post-resume NLL is elevated (~200k), should settle.

**Process change:** training babysitting is meant to be automated going
forward — a recurring monitor over ssh (ControlMaster) restarts dead runs
inside tmux, applies diagnosed fixes, and runs the calibration eval on
completion. Escalation rule: same failure surviving two distinct fixes,
stop and consult.

---

## 2026-07-22 — Belief milestone confirmed on the local CPU/synthetic path

**DSMLP run 3b status:** unconfirmed — no way to authenticate to `dsmlp`
from this session (Duo push needs the phone). Still open.

**What ran instead:** `python -m train.train_belief --no-wandb` on
`configs/default.yaml` (stub encoders, `SyntheticTrajectoryDataset`, CPU),
20 epochs / 640 steps, kl_weight 1e-2, occlude_p 0.5 (same fixes as the
DSMLP config). ~33 min wall clock. No optimization spikes; final mean sigma
~0.6-0.65, KL ~185-195.

**Calibration** (`eval/uncertainty_analysis.py`, synthetic held-out seed
999, 768 samples): sigma occluded/visible ratio = 1.335 (occluded
0.64-0.78, visible 0.53-0.56, clean separation, no floor collapse). The
README's "σ spikes on occlusion" milestone, confirmed on the offline path.

**Reading vs. the DSMLP runs:** confirms the belief network + occlusion
augmentation (commit `28795ea`) produce input-dependent variance when the
data actually demands it. Synthetic's occlusion windows are a stronger,
cleaner pressure than LIBERO's augmented clips, so this doesn't guarantee
the same ratio on LIBERO, but it confirms the mechanism works. Real read on
LIBERO still depends on run 3b.

---

## 2026-07-23 — World-model milestone fails on the belief checkpoint above; representation collapse in obs_embed, fixed, belief retrained

**Symptom.** Trained `train.train_world_model` on the 07-22 belief
checkpoint. Training-time `rollout5_model_mse` mostly beat
`rollout5_naive_mse`, but a clean held-out check (64 synthetic clips, seed
999) told a different story: model beat naive on only 26/64 clips (40.6%)
despite winning on the batch mean. On clips with no occlusion in the 5-step
window, naive (persistence: predict obs_{t+5} = obs_t) scored MSE ~0 and the
model couldn't beat it.

**Root cause.** `belief_net.encode_obs`, the pooled observation embedding
fed to both the belief GRU and the world model, had collapsed to a
near-constant vector: across a 12-frame clip where a 32px square crosses the
whole 224px canvas, obs_embed moved <0.05 in norm while the underlying
patch embeddings moved by ~14 (raw patches vary correctly with position;
only the pooling head's output doesn't). The next-obs NLL target is
`encode_obs`'s own output one step ahead, computed under `no_grad`, so
nothing stops the trainable pooling head from satisfying that objective by
ignoring its input. The sigma-on-occlusion milestone kept passing throughout
because detecting "blanked or not" needs far less precision than tracking
continuous position.

**Fix, three iterations** (`via/belief/belief_state.py`
`BeliefStateNetwork.loss`, `configs/{default,dsmlp}.yaml` `belief.var_weight`):
a VICReg-style variance-floor term, `var_weight * relu(1 - std).mean()`,
added to the belief loss.

1. Batch-wide floor (std across all `B*T` samples per dim): satisfied for
   free via between-clip differences (color, start position) while staying
   flat within a clip. Useless for the world model, which needs within-clip
   dynamics.
2. Per-clip, across-time floor, still including occluded frames: satisfied
   for free via the blank/reveal jump at an occlusion window's edges
   (~300-unit spike) while every genuinely-moving visible frame stayed flat.
3. Per-clip, across-time, visible-frames-only floor: `occlude_clips` now
   returns its blanking mask, combined with the dataset's own occlusion
   flag and passed into `belief_net.loss(patches, occluded=mask)`, which
   excludes those timesteps from the variance statistic. This is the
   version that worked — occluded frames can no longer supply free
   variance, so only real motion satisfies the floor.

**Weight tuning, also iterative.** var_weight=1.0 and 20.0 both looked fine
in short, hand-rolled reimplementations of the training loop but rebounded
toward collapse partway through the real full run whenever an NLL spike
hit. The hand-rolled version isn't a faithful stand-in for
`train.train_belief` — it diverges in RNG state (model-init order,
DataLoader worker seeding) and gave misleadingly stable numbers that didn't
reproduce for real. Trials need to go through the actual script (a
`--config` with fewer epochs is a cheap enough proxy). var_weight=100,
tested this way for 8 epochs, gave a steady, non-rebounding descent and was
confirmed on the full 20-epoch run.

**Final belief checkpoint** (var_weight=100, occlusion-excluded per-clip
temporal floor): var_loss settled ~0.3-0.55, no longer collapsed. obs_embed
now moves by tens of units across a clip on non-occluded synthetic data
(vs. <0.05 before), tracking real position. Sigma-on-occlusion re-checked
and still holds: ratio 1.219 (was 1.335 pre-fix — some margin traded for
real motion-sensitivity, still comfortably > 1.0).

**World model retrained on the fixed belief checkpoint.** Held-out check
(64 clips, seed 999): model beats naive on mean (18.28 vs 19.50), median
(3.09 vs 3.30), and majority of clips (36/64, up from 26/64 pre-fix). The
Checkpoint-1 world-model milestone, now earned rather than free-riding on
occlusion artifacts or a weak baseline.

**Open thread:** all of this is on the local CPU/synthetic path. DSMLP run
3b's outcome is still unconfirmed, and the belief/RSSM checkpoints trained
on LIBERO will likely need the same var_weight fix — the collapse mechanism
is generic (any self-predictive NLL target with no gradient-stopped anchor),
not synthetic-data-specific, so it's worth checking directly rather than
assuming.

---

## 2026-07-24 — Goal inference (RSA) trained and confirmed, stage 3/4 done

**Config:** `train.train_goal` on `configs/default.yaml` defaults — lr
3e-4, batch 64, 10 epochs (640 steps), `SyntheticInstructionDataset` (4096
items, graded ambiguity 0/1/2), zero belief context throughout (by design —
this stage pretrains the language->goal mapping in isolation; belief
conditioning only matters once fine-tuned on real scenes). Independent of
the belief/world-model checkpoints and their collapse issue above. ~40s on
CPU, no vision encoder involved.

**Training-time signal:** entropy-by-ambiguity ordering wasn't monotone
until partway through training (ambiguity-0 briefly dipped below
ambiguity-1 between steps ~100-450, both still well under ambiguity-2).
Full instructions converge to low entropy fastest; ambiguity-1's genuine
residual uncertainty (which of 8 objects) takes longer to settle. Ordering
locked in as monotone by ~step 500 and held.

**Held-out check** (seed 999, 512 fresh instruction/goal draws): mean
entropy 1.499 (amb 0) < 1.969 (amb 1) < 3.358 (amb 2) — the README's "goal
entropy tracks ambiguity" milestone, monotone and clean. Accuracy tells the
same story: 79.5% / 13.0% / 2.4% by ambiguity level. Ambiguity-1's 13.0% is
close to chance among the 8 objects once the verb is known (12.5%);
ambiguity-2's 2.4% is close to chance over all 32 goal slots (3.1%), since
its instruction string is fixed and carries no goal information.

**Status:** 3 of 4 training stages done (belief, world model, goal). Stage
4 (`train.train_decision`) is next. Real LIBERO/DSMLP results remain
outstanding.

---

## 2026-07-24 — Decision module (utility head + adaptive gate) trained, stage 4/4 done

**Config:** `train.train_decision` on `configs/default.yaml` defaults — lr
3e-4, batch 16, 10 epochs (320 steps), using the belief/world-model/goal
checkpoints above. Converged smoothly, no instability: utility MSE
0.366 → 0.005, gate MSE 0.240 → 0.042. No self-referential-target collapse
risk here — `progress` (utility's regression target) is an external
ground-truth label from the dataset (`torch.linspace(0, 1, T)`), not
something the network generates for itself.

**Qualitative check on the gate:** does lambda actually rise with belief
uncertainty? On 32 held-out occluded synthetic clips (seed 999): mean
lambda occluded 1.075 vs. visible 0.818; corr(sigma, lambda) = 0.842 across
all timesteps. Note for next time: this check was first written with
`for item in ds:` over a raw `SyntheticTrajectoryDataset` instead of
`DataLoader`/`range(len(ds))` — the dataset's `__getitem__` has no bounds
check, so that idiom never raises `IndexError` and loops forever. Left a
runaway process burning CPU for several days before it was caught and
killed. Iterate raw datasets by `range(len(ds))`, not `for x in dataset`,
unless a `DataLoader` is involved.

**Status:** all 4 training stages complete on the local CPU/synthetic path.
The two remaining README milestones — end-to-end LIBERO success rate
(`eval/eval_libero.py`) and the ablation table (`eval/ablations.py`) — need
`LiberoEnvRunner`, which needs a real LIBERO + robosuite install; neither is
available here (`import robosuite` fails). Both, plus the real LIBERO/SigLIP
DSMLP training run (3b's outcome still unconfirmed), remain open and
require the GPU/LIBERO box.

---
