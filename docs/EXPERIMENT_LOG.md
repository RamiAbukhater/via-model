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

## 2026-08-11/12 — Real LIBERO/SigLIP belief run, off DSMLP: completes, fails calibration (ratio 1.000)

**Platform change.** DSMLP access remained blocked (Duo/VPN dependency, prior
disk-quota and driver friction — see 2026-07-13/14 entries). Moved the
real-data run to a local Windows box instead: RTX 3060 Ti (8GB), real SigLIP
ViT-B/16 (`google/siglip-base-patch16-224`), real `libero_spatial` (500 demos
/ 10 tasks, downloaded via the HF mirror), `configs/local.yaml` (same
hyperparameters as `configs/dsmlp.yaml`: kl_weight 1e-2, occlude_p 0.5,
var_weight 100.0, clip_len 16, batch_size 16, 20 epochs). Language encoder
left as `stub` — not needed for belief/world_model/decision, and goal
inference trains on synthetic data regardless of `data.source` (see
2026-07-24 entry).

**Infrastructure failure, distinct from anything DSMLP hit.** First attempt
crashed the entire PC (not just the process) at step ~120, mid-spike
(loss had reached ~330k). Root-caused via Windows Event Log: a corrected
PCIe Advanced-Error-Reporting event on the GPU's Root Port at T-22min,
then a Volsnap disk error and a Windows sleep-transition attempt
(Kernel-Power Event 42) both within the same minute as the final
Kernel-Power Event 41 (unclean shutdown) — no BSOD minidump was written,
consistent with a hard freeze/power event rather than a software crash.
Physical cause: a daisy-chained PCIe power cable (two GPU power connectors
fed off a single wire run from the PSU) — adequate at idle/bursty gaming
loads, insufficient for ML training's sustained near-max draw. Interim
mitigation (proper cable on order): GPU power capped to 130W via
`nvidia-smi -pl 130` (default 200W, card supports 100-220W) and Windows
sleep-on-AC disabled (`powercfg /change standby-timeout-ac 0` — the 15-min
idle timer doesn't get reset by background GPU compute alone, and the sleep
transition itself likely contributed to the crash). Also added
step-interval checkpointing (every 200 steps, not just per-epoch) to
`train/train_belief.py` as a safety margin.

**Retrained with both fixes in place: completed all 20 epochs / 8,830 steps
without crashing.** But the loss trajectory only superficially resembles a
healthy run: it spiked to a peak of 548,911 at step 130 (steps 100-130 are
identical across both attempts — fully deterministic given fixed seed 7 —
so this spike is a property of the recipe on real data, not noise), then
fell back into a 25k-63k band by step ~300 that looked consistent with the
07-14 "Run 3 (augmented): healthy training" entry's documented 15k-53k
range. It did not hold: by the end of training, loss had drifted up into a
persistent 120k-260k band, KL settled around 12,300 (well above the 07-13
run's *collapsed* KL of ~7,400, and far above the healthy hundreds-to-~2,000
range), while mean_sigma oscillated ~0.6-0.65 (not pinned at the
sigma_min≈0.018 floor the way 07-13's collapse was).

**Calibration check** (`eval/uncertainty_analysis.py`, in-domain LIBERO
occlusion probe — 3-frame blanked windows in 64 held-out clips, same
distribution the model trained on): **sigma occluded/visible ratio =
1.000.** Flat, regardless of observability — the same signature as the
07-13/07-14 collapsed runs, despite mean_sigma *not* being at the numeric
floor this time. So this looks like a different route to the same failure
(sigma uninformative about occlusion) rather than a literal repeat of the
floor-collapse mechanism.

**Open questions, not yet resolved:**
- Whether the kl_weight/var_weight values tuned on synthetic/stub data (and
  carried into `configs/dsmlp.yaml` untested — DSMLP never reached this
  checkpoint) are simply wrong for real SigLIP-scale patch embeddings, which
  likely have very different magnitude/statistics than `StubPerception`'s
  fixed random projection.
- Whether real LIBERO scenes (much richer visual variation than the
  synthetic shapes-on-background clips) need a stronger occlusion signal
  (higher `occlude_p`, longer blanked windows) to compete with natural scene
  complexity for the network's attention.
- Whether 20 epochs / ~8.8k steps is simply not enough on real data, given
  the loss never converges the way it does on synthetic (640 steps to a
  clean result) or even matches the DSMLP run 3 numbers it superficially
  resembled mid-run.
- No indication of a representation-collapse bug specifically (`var_loss`
  correctly reads ~0.0000, i.e. the per-clip visible-frame variance floor
  from the 07-23 fix is satisfied — `obs_embed` is not collapsing to a
  near-constant vector the way it did pre-fix), which narrows this to the
  belief network's variance calibration itself rather than a repeat of the
  07-23 issue.

**Status:** belief milestone (`sigma spikes on occlusion`) not yet met on
real data. World-model/decision training (both depend on this checkpoint)
paused pending a decision on how to iterate — candidates: escalate
kl_weight (the log's own 07-13 entry names 1e-1 as the next step "risking
the opposite failure — sigma pinned near 1, underfit"), strengthen
`occlude_p`, or retrain longer. Checkpoint kept as `belief.pt` (not renamed
`_collapsed`, unlike 07-13's, since the failure mode looks mechanistically
different and may be worth diffing against).

---

## 2026-08-12 — Root cause found: unbounded `obs_embed` scale, not a kl_weight problem

**Investigated before retraining blind** (rather than jumping straight to
the kl_weight 1e-1 escalation named above). Compared `StubPerception` vs
real `SigLIPPerception` patch statistics directly: real SigLIP patches have
std ~2.28 vs Stub's ~0.24 (Stub's conv is explicitly scaled by `*0.02` at
init) — real patches enter the belief net roughly an order of magnitude
larger.

**Root cause, confirmed empirically on the 08-11/12 checkpoint above:**
`encode_obs` (`via/belief/belief_state.py`) has no normalization on its
output. `var_weight`'s per-clip temporal-variance floor
(`relu(1 - temporal_std)`) only ever pushes `obs_embed`'s scale *up*
(nothing penalizes it for growing too large), and on real data this ran
away far past what `next_obs_logvar`'s clamp (`_LOGVAR_MAX = 4.0`, i.e.
representable variance capped at `e^4 ≈ 54.6`) can represent:

- Fresh (untrained) net: real `obs_embed` std ≈ 0.15 — below the var_loss
  floor, so training has to amplify it.
- Trained (the collapsed checkpoint above): real `obs_embed` std ≈ 742,
  per-dimension variance up to 23,176 — three-plus orders of magnitude past
  the clamp's ceiling.
- Meanwhile `next_obs_logvar` converged to represent variance of only
  ~3-13 (not even pinned at the 4.0 ceiling — the network gave up trying to
  match a target that far out of reach).

So the NLL had a structural floor it could never train below, regardless of
`kl_weight` — that overwhelming, unfixable NLL pressure is what drowned out
whatever gradient signal would otherwise calibrate sigma-on-occlusion.
Confirms `kl_weight` was never the right lever here; escalating it per the
07-13 playbook would not have fixed this.

**Fix:** added `nn.LayerNorm(obs_embed_dim)` (`self.obs_norm`) on
`encode_obs`'s output, so `obs_embed` has a consistent, bounded scale
regardless of which perception encoder feeds it. Verified on a fresh net
before retraining: both real and synthetic `obs_embed` now std ≈ 1.0 at
init (previously 0.15 vs 0.03 respectively) — comfortably inside the
clamp's representable range. Full `pytest` suite (26 tests) still passes.

**Retrained belief from scratch, same recipe/config otherwise.** Loss
trajectory is unrecognizable from the pre-fix run — smooth, no spike, no
crash (loss 220 → 127 over the first 90 steps, vs. the old run's spike to
548,911 by step 130). Final numbers: loss ~220-250, NLL ~210-250, KL
~150-170, var_loss ~0.01-0.08 (small, oscillating — not slammed to exactly
0 the way the collapsed run's was), mean_sigma ~0.62-0.67. Strikingly close
to the 07-22 *synthetic*-path run's final numbers (KL ~185-195, sigma
~0.6-0.65) — the normalization appears to have put both data paths in a
comparable, well-behaved regime.

**Calibration check:** sigma occluded/visible ratio = **0.966**. Still
fails the `> 1.0` milestone, but this is a qualitatively different result
from the pre-fix 1.000 — not the flat degenerate signature of a network
that never got a chance to learn, but a real, varying number from a network
training normally. Read: the scale bug is fixed, but the occlusion signal
itself (3 frames blanked mid-16-frame real clip) may be too weak relative
to real LIBERO scenes' natural visual complexity (robot arm motion,
lighting, texture) to have taught sigma-on-occlusion within this budget —
unlike the synthetic clips (simple shape on a plain background), where
occlusion is comparatively the dominant source of unpredictability.

**Status:** infra (crashes) fixed 08-11, scale bug fixed and validated
08-12, calibration milestone still open. Candidates for next iteration,
now on a much better-understood foundation: strengthen `occlude_p`/widen
the blanked window, train longer, or investigate whether the residual gap
is sample-size noise (64 held-out clips) vs. a real, reproducible shortfall.
World-model/decision training remain paused pending this call.

---

## 2026-08-12/13 — Belief milestone met on real data (thin margin)

**Chose to strengthen the occlusion signal** (over training longer or
escalating kl_weight): `occlude_p` 0.5 → 0.75, and made the blanked-window
length config-driven (`occlude_max_len`, `train/train_belief.py`) rather
than a hardcoded default, set to 7 (was fixed at 4) — so trained clips can
now have up to ~44% of their 16 frames blanked, vs. ~25% before.
`configs/local.yaml` only; `default.yaml`/`dsmlp.yaml` left untouched since
this is a real-data-specific iteration, not a change to the confirmed
synthetic baseline. `eval/uncertainty_analysis.py`'s probe window
deliberately left at 3 frames (unchanged) — keeping the evaluation fixed
while iterating on training is the right comparison.

**Retrained from scratch, same normalization fix as the 08-12 entry above.**
Loss trajectory healthy again, no spike (220 → 123 by step 60). Final
numbers: loss ~230-250, KL ~110-130, mean_sigma ~0.72-0.75 (up from
~0.62-0.67 in the weaker-occlusion run — makes sense, more of each clip is
now blanked on average).

**Calibration check: sigma occluded/visible ratio = 1.023** (64 clips) —
**passes the `> 1.0` milestone.** Re-checked at 256 clips (same fixed seed,
superset of the first 64): 1.020 — stable across sample sizes, so this
reads as real signal rather than sampling luck.

**Caveat — the margin is thin.** 1.02 is a much weaker separation than the
synthetic path ever produced (1.335 pre-var_weight-fix, 1.219 post-fix, see
07-22/07-23 entries). Real LIBERO scenes' natural visual complexity is
still very plausibly diluting the occlusion signal relative to synthetic's
simple shape-on-background clips — strengthening `occlude_p`/window got the
ratio from 0.966 to 1.02, consistent with that read, but there may be
diminishing returns before reaching anything like the synthetic path's
separation. Worth keeping in mind when interpreting downstream
world-model/decision results built on this checkpoint: the epistemic-value
term (`information gain from imagined observations`) depends on belief
sigma being a genuinely useful uncertainty signal, and a ratio of 1.02 is a
much weaker foundation for that than 1.2-1.3 would be.

**Status:** belief milestone met on real LIBERO data (first real-data
confirmation — DSMLP never got this far). Proceeding to world-model
training on this checkpoint.

---

## 2026-08-13 — Validation infrastructure: train/val split, and belief re-checked for leakage

**Gap found:** `LiberoTrajectoryDataset` had no train/val split — every real-data
eval so far (including the 1.02 calibration result above) sampled a
"held-out seed" from the *same* full pool used for training. For synthetic
data that's fine (procedurally generated per-seed, genuinely new samples);
for LIBERO (a fixed pool of demos) it isn't — the model had almost
certainly already seen those exact clips.

**Fix:** `via/data/libero.py` — `LiberoTrajectoryDataset` now takes
`split="train"|"val"|"all"`, held out *by demo* (every 5th demo index per
task file, not by clip window — clips within a demo overlap via the
clip_len//2 stride, so holding out windows instead of whole demos would
still leak). `train/common.build_trajectory_dataset` now defaults to
`split="train"`, so every training script is held-out-safe without each
call site needing to remember to ask. `eval/uncertainty_analysis.py`'s
LIBERO probe now uses `split="val"`.

**Re-checked the belief calibration result with the corrected split:**
ratio 1.022 (200 clips) — essentially unchanged from 1.020-1.023 measured
against the un-split pool. So the eval-selection leakage specifically
didn't inflate the number.

**Residual concern and how it was resolved without retraining:** the
`belief.pt` checkpoint itself was trained before the split existed, so its
weights did receive gradient updates from what's now the val split — fixing
eval selection doesn't undo that. Rather than spend another ~belief-length
training run on a clean retrain with no guarantee it would change anything,
checked directly and cheaply: compared belief NLL/KL/mean_sigma on 100
train-split vs. 100 val-split clips using the existing checkpoint, no
training involved. Train nll=257.60, val nll=255.87 (val marginally
*better*, kl and mean_sigma both essentially identical too) — the opposite
of what memorization would produce. No evidence of overfitting to specific
demos; the retrain was skipped on this basis.

**New shared validation utility:** `train.common.world_model_holdout_check`
— per-clip (not batch-mean) rollout5 win-rate on genuinely held-out data,
importable both for a standalone check and for periodic in-training calls.
Exists specifically because 07-23's failure mode (batch mean hid a 26/64
per-clip loss rate) would not have been caught by the in-training metric
alone. Wired into `train/train_world_model.py` to run every 500 steps
during training, so a bad run can be caught mid-training rather than only
after all 20 epochs complete.

**Status:** validation methodology hardened before starting world-model
training: genuine train/val split, leakage checked and ruled out cheaply,
and live held-out signal during training instead of an end-of-run-only
check.

---

## 2026-08-13 — World model trained on real LIBERO data; the held-out metric itself needed fixing

**Config:** `train.train_world_model` on `configs/local.yaml`, belief.pt
frozen (the 1.02-ratio checkpoint above), real SigLIP + real `libero_spatial`
(train split), 20 epochs, ~7,030 steps. Clean completion, no crash, no
errors.

**Training-time signal:** `recon` loss dropped sharply (2.9 → 0.6, >4x),
`kl` stable ~1.0. The live held-out win-rate check (every 500 steps, the
new infrastructure above) told a very different story: oscillated
noisily around chance the entire run and *finished at its worst point*
(step 0: 54.7%, peak 59.4% at step 4500, final at step 7000: 42.2%) — the
same shape of gap 07-23 hit (strong training metric, held-out metric
doesn't confirm it), different mechanism.

**Investigated before concluding the run failed** (same approach as the
belief calibration issue): re-ran the held-out check standalone on the
final checkpoint and found two problems with the *metric*, not necessarily
the model:

1. **Unseeded stochastic sampling.** `RSSM.prior_step` samples the imagined
   rollout stochastically; `rollout_mse` never seeded this, so re-running
   the identical held-out clips through the identical checkpoint gives a
   different win-rate each call — noise the live training-time trend never
   controlled for.
2. **The win-rate metric is confounded by clip difficulty.** The naive
   "persistence" baseline (predict obs_{t+k}=obs_t) is close to optimal *by
   construction* on clips with little real motion in the 5-step window — no
   model can score much below an already-near-zero error. 07-23's entry
   already named this; it wasn't accounted for in the aggregate win-rate.

**Fixed measurement** (`train.common.world_model_holdout_check`, rewritten):
average `model_mse` over 10 stochastic draws per clip (naive is
deterministic, drawn once), and split the held-out set by naive_mse
(low naive_mse == little motion, a cheap proxy) into low/high-motion
halves. Result on the final checkpoint, 128 held-out clips:

- Overall win-rate: 51.6% (looks like chance)
- **Low-motion half: 9.4%** — expected; naive is genuinely close to optimal
  here, consistent with 07-23
- **High-motion half: 93.8%** — the number that actually reflects learning,
  and it's strong
- Mean MSE overall: model 4.75 vs. naive 5.52 — model wins on the literal
  README milestone definition (`rollout5_model_mse` beats
  `rollout5_naive_mse`)

**Read:** the world model learned real dynamics well on clips where there's
something to learn; the aggregate win-rate metric was hiding that behind a
large population of near-static clips where no model can win. Not a
retraining question — a validation-metric question, caught and fixed
without spending any additional GPU time. `world_model_holdout_check`
now reports the stratified breakdown by default; the plain aggregate
win-rate alone should not be trusted for this data going forward.

**Status:** world-model milestone met on real LIBERO data (first real-data
confirmation, same as belief). No retrain performed — the existing
checkpoint holds up under the corrected metric.

---

## 2026-08-13 — Belief calibration re-examined before decision training; pooled ratio was underselling it

**Prompted by:** reviewing belief before building the decision module on top
of it (decision depends on belief sigma via the adaptive gate), and by the
world-model metric fix immediately above — same pooling-vs-per-clip lesson,
applied here before spending any more compute.

**The 1.02 pooled ratio (08-12/13 entries) has the identical structural
issue as the world model's aggregate win-rate:** it pools every occluded
timestep across all 200 held-out clips into one bucket and every visible
timestep into another, so between-clip differences in *baseline* sigma
level (some tasks/scenes are just visually busier than others, independent
of occlusion) dilute a real within-clip effect.

**Recomputed per-clip instead**, from the existing `results/uncertainty.csv`
(no rerun needed — free): for each clip, compare its own occluded-mean sigma
against its own visible-mean sigma.

**Result: 154/200 clips (77.0%) show sigma rising during occlusion relative
to their own baseline** — a statistically decisive effect (a sign test
against 50/50 chance on 200 trials at this split is essentially conclusive),
not the thin, marginal-looking signal the pooled 1.02 implied. Mean per-clip
effect +0.016, consistent in sign, median close to the mean.

**Fix:** `eval/uncertainty_analysis.py` now computes and prints both the
pooled ratio (kept for continuity with earlier entries) and the per-clip
win-rate, with the win-rate flagged as the more trustworthy number.
Confirmed end-to-end (not just from the cached CSV) — same 154/200.

**Revised read:** belief calibration on real LIBERO data is meaningfully
stronger than previously stated. The "thin margin, weaker foundation for
the decision module's epistemic-value term" caveat in the 08-12/13 entries
above should be relaxed accordingly — the effect is real and consistent at
the per-clip level, even though it doesn't produce a large pooled ratio
(likely because between-clip baseline variation in sigma is itself fairly
large relative to the occlusion effect size, not because the effect is
weak or unreliable).

**Status:** belief foundation re-validated before decision training, no
retraining needed — same lesson as the world-model entry: check whether the
*metric* is doing the finding justice before spending more compute chasing
a number that may already be fine.

---

## 2026-08-13 — Goal inference: broken on real data, found before decision training, fixed

**Prompted by:** explicit request to review belief and (by extension) the
rest of the foundation before starting decision training, since decision
depends on all three of belief/world_model/goal.

**Checked goal inference on real data directly** (had never been tested —
train/train_goal.py only ever ran on `SyntheticInstructionDataset`,
regardless of `data.source`, per the 07-24 entry). Ran the existing
synthetic-pretrained `goal_net` + `StubLanguageEncoder` on 16 real,
genuinely different `libero_spatial` instructions with real (non-zero)
belief `mu`. Result: entropy 3.10-3.42 out of a max possible 3.47 (32
slots) for *every* instruction regardless of content, and 5 genuinely
different task descriptions mapped to scattered, inconsistent goal slots.
Effectively uninformative — the goal-conditioning half of the architecture
would not have been functioning during decision training, despite training
running without errors and producing numbers.

**Root cause:** `goal_net` was pretrained only on ~12 words from a fixed
synthetic template vocabulary (4 verbs x 8 objects), always with zero
belief context. Real LIBERO instructions are longer, use entirely
different vocabulary ("ramekin," "cookie box," "wooden cabinet"), and
decision training would be the first time it ever saw non-zero belief
context — two simultaneous distribution shifts from what it was trained on.

**Fix — extended `train/train_goal.py` with a real-data fine-tune path**
(`data.source: libero`), per the extension point the loss function's
docstring already anticipated ("goal_targets... from synthetic pretraining
or LIBERO task annotations mapped to slots"). LIBERO has no ready-made
(instruction, goal) labels the way the synthetic generator does — each
`libero_spatial` task (one fixed instruction per demo file, 10 total) is
treated as its own goal-slot label, using 10 of the 32 available slots.
Warm-starts from the existing synthetic-pretrained checkpoint automatically
(backed up separately as `goal_synthetic_only.pt` first — cheap insurance,
not because it was expected to be needed) and trains against real, non-zero
belief `mu` from the frozen belief checkpoint. Sized deliberately smaller
than the belief/world-model runs (batch 16, 5 epochs, ~1,765 steps — a
much easier 10-way classification problem, not another multi-hour
commitment) with a held-out check (`train.common.goal_holdout_check`)
every 50 steps, split="val" — held out by demo, though note this tests
generalization to new demo instances of *known* tasks, not to unseen
tasks/instructions, since libero_spatial only has 10 tasks total.

Kept the language encoder as `stub` rather than switching to real Phi-3:
since each task has one constant instruction string, the stub's
hash-based-but-consistent embeddings are learnable for this closed 10-way
problem even though they're not semantically meaningful — Phi-3 would
matter for generalizing to genuinely novel instructions, which isn't in
scope while only libero_spatial exists locally.

**Result:** held-out accuracy climbed cleanly 4.7% (baseline, ~chance for
10 classes) → 35.9% → 46.9% → ... → **95.3%** by step 1750, held-out
entropy falling from 3.17 to 0.28. No errors. Re-ran the original
diagnostic against the fine-tuned checkpoint: 16/16 correct on a fresh
held-out draw, with confident (low-entropy) predictions correctly
distinguishing subtly different instructions ("next to the ramekin" vs. "on
the ramekin" vs. "in the top drawer of the wooden cabinet").

**Status:** goal inference now genuinely functional on real LIBERO data.
All three components decision training depends on (belief, world_model,
goal) have been independently checked and validated on real, held-out data
today. Proceeding to decision training.

---

## 2026-08-13/14 — Decision module (utility + gate) trained on real LIBERO data — all 4 stages now real-data-confirmed

**Config:** `train.train_decision` on `configs/local.yaml`, belief/world_model/goal
all frozen (the checkpoints validated above), real SigLIP + real
`libero_spatial` (train split), batch 16, 10 epochs, ~3,510 steps. Added a
held-out check (`train.common.decision_holdout_check`, split="val") every
250 steps before launching — same reasoning as every other stage today:
don't trust the in-training loss alone. Utility MSE is averaged over 5
stochastic draws (`RSSM.observe`'s posterior sampling affects this the same
way it affected the world-model rollout, 08-13 entry); the gate check
blanks a window per held-out clip and reports the *per-clip* win-rate
(lambda higher during occlusion than that clip's own visible baseline),
not just a pooled/aggregate comparison — the same lesson as the belief
calibration fix.

**Clean completion, no errors.**

**Utility head:** held-out MSE 0.205 → 0.0042-0.0052, smooth and
essentially fully converged by step 1000 (0.0062) with only minor noise
after. Progress regression from RSSM+goal features works well on real
data.

**Gate:** more nuanced. Held-out per-clip win-rate opened at 87.5%
(step 0, untrained — likely partly incidental, since sigma is a direct
gate input even pre-training), dropped and settled around 72% from step
500 onward (steps 2250-3500: 71.9-73.4%) — still clearly above the 50%
chance level, but below where the untrained network started. The more
telling number: the *average* lambda gap between occluded and visible
states grew substantially over training — occluded 0.993 vs. visible
0.989 at step 0 (gap 0.004) to occluded 1.057 vs. visible 0.962 at step
3500 (gap 0.095), roughly a 20x increase. Read: the gate is learning a
real, growing, above-chance sensitivity to uncertainty, consistent with
the design intent (lambda rises with belief sigma), but it isn't a large
or dramatic effect — expected, since the gate's training target
(`target_lam`, a per-batch-normalized function of sigma) reflects
*all* sources of sigma variation across a batch, not specifically
occlusion, and belief sigma's own occlusion-sensitivity on real data is
itself a real-but-modest effect (77% per-clip win-rate, 08-13 entry) —
the gate inherits that ceiling rather than exceeding it.

**No quantitative pass/fail bar exists for this stage** (unlike belief's
`ratio > 1.0` or world-model's win-rate/mean-MSE) — the 07-24 synthetic-path
entry reported its qualitative numbers (mean lambda occluded 1.075 vs.
visible 0.818, corr 0.842) without a threshold either. Real-data numbers
here are weaker than that synthetic reference (expected, given the same
gap exists one level up in belief's own calibration strength) but the
qualitative direction is right and the trend across training is real,
checked on genuinely held-out data rather than assumed from training loss
convergence.

**Status: all 4 training stages now confirmed on real LIBERO data** —
belief, world_model, goal, and decision, each independently validated
against held-out demos today rather than assumed to work because training
completed. This is the milestone DSMLP's run 3b never reached (unconfirmed
outcome, see 07-14/07-22 entries). Remaining open items unchanged from
before: `eval/eval_libero.py` (closed-loop success rate) and
`eval/ablations.py` (λ=0/point-goal/no-belief ablation table) both need a
live LIBERO simulator (robosuite/MuJoCo), not installed locally and not yet
in scope — see README's milestone table.

---

## 2026-08-14 — Live LIBERO simulator working locally on Windows; closed-loop eval run, 0% success, root cause understood

**Scope change:** moved to bring up the live LIBERO/robosuite/MuJoCo simulator
locally (Windows) for `eval/eval_libero.py` and `eval/ablations.py`, the two
milestones blocked on it since 07-24.

**Install chain, three genuine Windows-specific incompatibilities found and
fixed, each verified in isolation before moving to the next layer:**

1. `robosuite==1.4.1` ctypes-loads a `mujoco.dll` from its own package
   directory on Windows; the pip wheel doesn't ship one. Fix: copied the
   real `mujoco.dll` from the installed `mujoco` package to
   `robosuite/utils/mujoco.dll`.
2. `robosuite/macros.py` hardcodes `MUJOCO_GPU_RENDERING = True`, which
   `robosuite/utils/binding_utils.py` unconditionally maps to `MUJOCO_GL=egl`
   for any non-Darwin system — correct on Linux, invalid on Windows (EGL is
   Linux-only; Windows only accepts glfw/wgl there), and set before any
   Python-level override is possible (`robosuite/__init__.py` reads it
   immediately on import). Fix: patched the macro to check
   `platform.system() != "Windows"`, falling through to the cross-platform
   GLFW context instead (already a transitive dependency).
3. `robosuite==1.4.1`'s controller code calls `mujoco.mj_fullM(..., data.qM)`
   — `qM` was renamed to `M` at some point in the mujoco 3.x line the latest
   pip wheel (3.11.0) installs by default, but Python 3.13 (this venv) only
   gets mujoco wheels from 3.2.6 onward. Rather than patch every such
   drift point (unknown how many exist), downgraded to `mujoco==3.2.6`
   (earliest cp313 wheel) instead, which still has `qM` — much closer to
   what robosuite 1.4.1 actually targeted, avoiding an unknown-sized
   whack-a-mole.

Verified incrementally at each layer before proceeding: bare `mujoco`
physics + offscreen rendering (real image content, not blank) -> plain
`robosuite.make('Lift')` full env (reset/render/step) -> LIBERO's own
`OffScreenRenderEnv` with a real `libero_spatial` bddl task (object states
matched task 0's actual objects: bowls, ramekin, plate, cookies) -> our own
`LiberoEnvRunner` wrapper. Also found and fixed `eval/eval_libero.py`'s
`--smoke` flag not actually being passed to `build_model` (a real,
pre-existing bug, unrelated to the simulator work).

Separately: the libero package's own editable pip install had a broken
package-discovery mapping (empty `MAPPING` dict in the generated
`__editable__...finder.py`) — likely confused by `libero_spatial`'s dataset
files sitting in the same repo directory as the `libero` package
(08-11 case-insensitive filesystem merge, still with us). Fixed by
uninstalling the broken editable package and adding a plain `.pth` file
pointing at the repo root instead — sidesteps setuptools' package discovery
entirely.

**First closed-loop run (all real checkpoints, no smoke): 0/100 episodes
succeeded (10 tasks x 10 episodes).** Investigated rather than accepting
this at face value:

- Tracked gripper-to-target-object distance across a real episode: it never
  decreased (0.34 -> 0.40, wandering) despite non-trivial action magnitudes
  — not a crash or a degenerate/zero-action policy, just undirected motion.
- Isolated the cause precisely: split the CEM objective into its two terms
  across the same 64 candidate action sequences. `expected_utility` varied
  by only ~0.03 (range 0.13-0.27) — essentially flat. `information_gain`
  varied by ~7.7 (range -7.0 to +28.1) — ~250x larger. The combined
  objective (`eu + lambda*ig`) is in practice almost pure information-gain,
  regardless of the gate's calibration, since lambda (~1.0-1.3) can't
  rescale a term that outweighs utility by two orders of magnitude.
- Traced the flat utility signal to its root: verified imagined RSSM
  features *do* vary substantially across different actions (std 0.35,
  range -21 to +13) — the world model isn't the bottleneck. The utility
  head's gradient w.r.t. that feature is real but small (mean |grad| =
  0.0054); this is the **mathematically expected outcome** of training
  against `progress = t/T`, a label that's mostly a function of *time index
  within the demo*, not of instantaneous state content — weak signal-to-
  label correlation makes "predict near the training mean" the MSE-optimal
  solution, which is exactly a low-output-variance function.
- Researched before picking a fix (per explicit instruction, rather than
  guessing): DreamerV3 (this RSSM design's own stated reference) trains its
  value function via TD/λ-returns bootstrapped through imagined rollouts
  from a reward signal, not a static per-transition regression label —
  confirms the progress-as-time-index choice is the structural issue, not
  a training bug. Active inference theory (Friston 2017, the framework this
  decision module is explicitly built on) states epistemic value is
  *supposed* to decay relative to pragmatic value as uncertainty resolves —
  that self-balancing mechanism requires the two terms to be on comparable
  scales, which they weren't. Current model-based-planning literature
  treats a learned policy as a *prior* for planning rather than searching
  from pure noise, which is what `CEMPlanner` did until now.

**Two fixes implemented (a third — replacing the progress-as-time-index
utility target with something reward/return-like — logged as a real, not
yet attempted, follow-up rather than a quick patch, since it changes the
training objective itself):**

1. **Score normalization** (`via/decision/decision.py`, `select_action`):
   z-score `expected_utility` and `information_gain` across the CEM
   population before combining, so `lambda` controls a well-defined
   relative mix instead of being drowned out by whichever term has larger
   raw magnitude.
2. **`ActionPrior`** (new module, same file): a small MLP behavior-cloned
   from real demo actions (`train/train_decision.py`, reusing the same RSSM
   posterior features already computed for utility — nearly free to add),
   used to seed `CEMPlanner`'s initial search mean via an autoregressive
   imagined rollout (`DecisionModule.prior_rollout`) instead of zero-mean
   noise. Opt-in (`action_prior=None` default) so existing tests/`--smoke`
   paths are unaffected. Trained alongside utility/gate (warm-started via
   `--resume`, not retrained from scratch): held-out BC MSE converged
   0.238 -> 0.021 over 3,510 steps, no errors.

**Verified incrementally, not just assumed to help:** gripper-to-target
distance test, same episode/task each time —
baseline (blind CEM): 0.34 -> 0.40 (no convergence)
+ normalization only: 0.33 -> min 0.25 (real but modest convergence)
+ normalization + action prior: 0.33 -> min 0.19, stabilizing ~0.19-0.22
(clear, substantial improvement each step, confirming both fixes are
directionally correct and the normalization fix alone was insufficient,
consistent with the diagnosis).

**Full closed-loop eval with both fixes: still 0/100 episodes succeeded.**
Getting close to an object is a long way from the precision LIBERO's
pick-and-place tasks actually require (correct grasp timing/pose, precise
placement) — the fixes measurably improved *approach* behavior but didn't
cross the much higher bar of full task completion within 300 steps. Not a
contradiction: a coarse, MLP-based BC prior trained on limited demos and a
now-well-scaled-but-otherwise-simple CEM objective were never likely to
solve fine manipulation outright; they were diagnosed and fixed as the
specific failure found (undirected planning), not presented as sufficient
for success.

**Status:** live LIBERO simulator fully functional locally for the first
time (real milestone — DSMLP never reached this either). Root cause of 0%
success understood in detail and partially, verifiably addressed. Full fix
(replacing the utility training target) not yet attempted. `eval/ablations.py`
(λ=0/point-goal/no-belief table) not yet run — with closed-loop success at
0% either way, worth deciding whether relative ablation comparisons are
still informative before spending more time there.

---

## 2026-08-15 — Counterfactual utility fix: real behavioral improvement, still 0% task success

**Implemented the deeper fix flagged as a follow-up above**, rather than
stopping at the two planning-time fixes: a contrastive counterfactual term
added to `train.train_decision`'s utility loss. For each training batch,
in addition to the existing real (on-path) target, imagine `horizon=5`
steps forward from a random real posterior state (`posts[k_start]`) using
*random* actions instead of the demo's real ones, and target that
counterfactual endpoint at the *starting* progress (`progress[:,
k_start+1]` — i.e. "no advancement"), contrasted against the real
continuation's higher target at `k_start+1+horizon`. Approximates Dreamer's
core idea (train the value function on imagined rollouts that include
off-policy states, not only replayed real data) without requiring a
learned reward model or a full actor-critic loop, which offline training
from demos-only doesn't have the ingredients for. Warm-started all three
decision heads via `--resume`; only ~3,510 steps (same budget as the prior
decision-training runs), no errors, `cf_mse` converged 0.023 -> ~0.003-0.007.

**Verified before/after rather than assuming the fix helped** (per the
established pattern all session): re-ran the CEM candidate score-spread
diagnostic (64 wildly different imagined action sequences, same episode/
state as 08-14's measurement). Counter to expectation, raw utility std
*dropped* (0.0195 vs. 0.030 pre-fix, 0.037 at 1/10th of training) — on that
metric alone the fix looked like it made things worse or did nothing.

**But the behavioral test (gripper-to-target distance, same task/episode
throughout this whole investigation) told a different, better story:**
- Pre-fix (blind CEM): 0.34 -> 0.40, no convergence
- + score normalization: 0.33 -> min 0.25
- + action prior: 0.33 -> min 0.19, stable ~0.19-0.22
- + counterfactual utility (this entry): 0.26 -> **min 0.12**, the closest
  approach across every iteration of this investigation — though it drifts
  back out to ~0.22-0.28 by the end of the 80-step window rather than
  holding position, suggesting the agent gets near the target but doesn't
  yet know to commit/stay once close (consistent with grasping-precision
  being a separate, harder problem than approach).

**Lesson: raw utility variance across a CEM population is not a reliable
proxy for whether the utility function is actually useful.** What matters
is whether it's *correctly correlated* with genuine state quality, not how
much it numerically varies — a flatter-but-correctly-ordered signal can
outperform a more spread-out but poorly-calibrated one once normalized and
combined with lambda. Worth remembering before using population-variance
diagnostics as a stopping condition in future iterations here.

**Full closed-loop eval (10 tasks x 10 episodes, all three fixes stacked):
still 0/100 succeeded.** No errors. Getting measurably closer to the target
object is real, verified progress; it is not the same as clearing LIBERO's
actual bar (correct grasp timing/pose while closing the gripper, precise
placement) within a 300-step budget. Three fixes in, each verified to move
the needle on the specific failure mode it targeted, and the closed-loop
success rate is still 0% — consistent with this being a genuinely hard
control problem for a coarse, limited-demo, MLP-based approach, not a sign
any individual fix was wrong.

**Status:** this is a reasonable stopping point for this line of
investigation without a larger intervention (more demo data, longer
training, or a fundamentally more capable action-selection component than
a small MLP action prior + CEM). `eval/ablations.py` remains unrun.

---

## 2026-08-16/21 — Stage 0 cheap diagnostics; Stage 2 proprioception + gate
fix; two real findings (one good, one a regression, now isolated and fixed)

**Stage 0 (no training, ~results/*.json diagnostics only):** analyzed the
08-14/15 100-episode eval's saved per-step diagnostics directly. Belief
sigma barely moves across a live 300-step episode (mean 0.7605 first 10
steps -> 0.7563 last 10, pooled over 100 episodes) -- it fluctuates but
never *settles* low. Since `AdaptiveGate` was a pure function of
instantaneous sigma, lambda never decayed into a committed-exploit regime
either (mean 1.03, 53% of all steps >= 1.0 -- information-gain weighted
comparably to, or more than, expected utility for the whole episode,
including the grasp window). Ran `eval/ablations.py` for the first time
ever and found it had a real bug: the `lambda0` variant tried to replace
`model.decision.gate` (a registered `nn.Module` submodule) with a plain
lambda, which `nn.Module.__setattr__` rejects -- fixed by monkey-patching
`.forward` instead, same pattern `point_goal`/`no_belief` already used. All
four ablation variants (full/lambda0/point_goal/no_belief) scored 0/30 --
no single cognitive-component ablation flips success on its own. A CEM
search-budget sweep (2x horizon/population/iterations) also barely moved
gripper-to-target distance -- ruled out "not enough search" as the
bottleneck; the signature (more search, same mediocre outcome) points at
the objective/representation being the limiting factor, not the planner.

**Stage 2, part 1 -- proprioception:** added end-effector position (3) +
gripper state (2) to the state representation, fused into `obs_embed`
inside `BeliefStateNetwork.encode_obs` (normalized MLP branch concatenated
with the pooled visual embedding, projected back to the same 256-d, so
nothing downstream changes shape). Sourced from `obs/ee_pos` +
`obs/gripper_states` in the offline demo HDF5s and `robot0_eef_pos` +
`robot0_gripper_qpos` in the live sim (standard robosuite/robomimic
naming, confirmed to match). Left orientation out of v1 -- the offline
files store a 3-dim `ee_ori` of unconfirmed convention vs. the live sim's
4-dim quaternion, and reconciling those risked a silent train/inference
mismatch. Retrained belief -> world_model -> goal -> decision in sequence
(cascading, since obs_embed feeds all of them).

**Real, reproducible regression found in the goal stage:** the real-data
fine-tune's held-out accuracy peaked ~95-97% mid-training then *declined*
to 68-74% by the end (confirmed reproducible across 4 seeds at n=128, not
a noisy batch) -- unlike this stage's previous run, which climbed
monotonically to 95.3%. Best explanation: belief.mu is now a richer,
more per-demo-specific signal (precise trajectories differ between demo
instances of the same task far more than coarse visual gist does), so
continued training could overfit per-demo idiosyncrasies rather than the
shared task-level signal. Fix: added best-held-out-checkpoint selection to
`train/train_goal.py`'s `_train_libero` (previously it just saved whatever
was current at the end). Rerun landed at 97.7-99.2% across 4 seeds.

**Stage 2, part 2 -- gate fix:** made `AdaptiveGate` also take a running
min of mean sigma seen so far in the episode (`min_sigma`), motivated by
the Stage 0 finding above -- intended to let lambda commit to exploitation
once confidence has ever been achieved, since instantaneous sigma alone
never settles low enough to trigger that on its own.

**Closed-loop verification found a second, worse regression:** gripper-to-
target end-of-episode distance roughly *doubled* (~1.0-1.1 vs. the pre-
Stage-2 baseline's 0.471) even though offline losses (utility MSE, gate
MSE, held-out checks) all looked fine in isolation -- the "don't trust
training loss alone" lesson recurring. Root-caused via an isolation
experiment (retrain decision with `min_sigma` neutralized to a constant,
everything else unchanged): behavior recovered to ~0.52, matching the
pre-Stage-2 baseline. The bug: `min_sigma` is a running *minimum*, trained
on 16-step clips (`clip_len=16`) but evaluated live over up to 300-step
episodes -- a cumulative min over 300 steps ratchets down to far more
extreme values than anything a 16-step window ever produced in training,
so the gate was extrapolating badly out-of-distribution at exactly the
mechanism meant to control explore/exploit. `min_sigma` is now neutralized
to a constant zero at every call site (mathematically equivalent to
dropping the input, since a constant is absorbed into the first layer's
bias) rather than reverting the architecture and retraining a third time
purely for cosmetic cleanliness -- see `AdaptiveGate`'s docstring
(`via/decision/decision.py`) before ever un-zeroing it; would need
training against episode-length rollouts or a length-normalized min,
not just re-enabling the current 16-step-trained version.

**Secondary, smaller finding:** proprioception measurably weakens the
belief network's visual-occlusion sensitivity (per-clip gate win-rate 77%
pre-Stage-2 -> 15% with the broken min_sigma gate -> 57%, roughly chance,
once neutralized) -- proprio is never blanked by the occlusion-augmentation
training trick, so the network now has an always-available, always-certain
fallback evidence source even when vision is occluded, undermining the
mechanism that used to force it to admit uncertainty. Did not chase this
further this round since it didn't show up as a driver of the closed-loop
distance regression once `min_sigma` was isolated out -- worth revisiting
if occlusion-robustness specifically becomes a priority again.

**Net result on the metric that matters:** with `min_sigma` neutralized,
proprioception-equipped closed-loop behavior (min dist 0.185, end dist
0.523, 6 episodes) is now roughly on par with the pre-Stage-2 baseline
(min dist 0.186, end dist 0.471) -- not yet a clear win, but no longer a
regression either, on a small sample. `eval/ablations.py`'s fuller
30-episode-per-variant comparison has not yet been rerun against these
checkpoints.

**Hardware, separate from the above:** five GPU-lost crashes across this
session, need a full reboot each time. Two confirmed at a freshly-verified
130W cap; the only fully clean decision-training completions were at
100W. See `[[project-via-model-hardware]]` memory for the full pattern --
this is not resolved, 100W is currently the working assumption, not a
proven-safe conclusion (small n). All prior belief/world_model/goal/
utility/gate/action_prior checkpoints from before this entry were
overwritten in place during the retrain cascade with no backup taken
first -- a real gap versus this project's earlier practice of backing up
`goal_synthetic_only.pt` before its own risky fine-tune. There is no clean
rollback to the pre-Stage-2 system without retraining from scratch on the
old (non-proprio) architecture.

**Follow-up (same day): full `eval/ablations.py` at proper scale** (3
episodes x 10 tasks x 4 variants = 120 episodes, ~45 min, GPU stayed
healthy throughout at the 100W cap) against the current (proprioception +
neutralized-min_sigma) checkpoints: full/lambda0/point_goal/no_belief all
0/30 -- identical to every measurement in this entire investigation,
pre- and post-proprioception alike. Answers the open question above:
proprioception is not (yet) a demonstrated closed-loop improvement over
the pre-Stage-2 system at this scale -- it recovered the specific
regression it caused, and independently avoided being a net negative on
the small-sample distance diagnostic, but success rate itself is unmoved.

**Status:** proprioception is wired through the full pipeline and
verified plumbing-correct (tests, `--smoke`, live sim) independent of the
regressions above, which were both found, root-caused, and fixed/isolated
rather than papered over. Net effect on the metric that actually matters
(closed-loop task success) is a wash, not a win, at current scale/training
budget. Remaining open questions: (1) is the belief occlusion-sensitivity
regression worth fixing on its own, (2) is `min_sigma` worth properly
re-engineering (episode-length training, or normalizing by window length)
given the diagnosis above, (3) whether the representation-precision
hypothesis this whole effort was testing needs a larger intervention
(more demo data, a richer proprio feature set including orientation, or
combining with the earlier-diagnosed utility-target weakness) than what
was tried here to actually move the needle, consistent with this
project's repeated finding that individually-well-motivated fixes each
move a diagnostic without yet crossing the success threshold.

---

## 2026-08-21/23 — Object-relative state, TD(0) value function, CEM search
tightening: real, compounding distance improvements (~0.5 -> ~0.3-0.4 end
distance), still 0% success; ceiling identified as data volume, not
capacity or search strategy

**Object-relative state.** Robot proprioception alone (previous entry) was
a wash. The natural extension: add *target-object*-relative position, not
just robot state -- the live sim exposes `<object>_to_robot0_eef_pos`
directly per-task (`obj_of_interest`), but offline demo HDF5s only store
raw, unlabeled full physics state. Built `scripts/extract_object_state.py`:
replay each demo's recorded `states` array directly through the live sim
(`sim.set_state_from_flattened` + `sim.forward()` + `_get_observations(
force_update=True)` -- the `force_update` flag was itself a real find,
without it robosuite's cached observables silently don't reflect a manually
set state) and log the target object's position each frame. Pure physics/
observation queries, no vision, no GPU -- 500 demos processed in ~80s.
`proprio_dim` 5 -> 8.

**Regression, again:** closed-loop end distance got *worse* (0.52 -> 1.15).
Root cause: object_rel's norm reached 0.92-1.5 during live rollout, while
training data (all 500 demos successful by construction) never showed it
exceed 0.487 -- classic imitation-learning distributional shift, not a
bug. Clamping object_rel's norm to the training range at inference time
(`via/data/libero.py`'s `clamp_object_rel`) only partially helped and left
min-distance worse too, ruling out pure magnitude-extrapolation as the
whole story.

**Utility target rework (the deeper fix flagged back on 08-14, never
attempted until now).** LIBERO demos turn out to already carry a usable
reward signal: `rewards` is exactly 1 at each demo's true final frame, 0
elsewhere (verified directly). Replaced `progress = t/T` regression with
genuine TD(0): `V(s_t)` trained toward `r_t + gamma*V(s_{t+1}).detach()`
using the real on-path continuation within each clip window (still no
learned reward model for imagined-rollout bootstrapping, same practical
constraint as before -- this is TD over real demo continuations, not full
Dreamer-style imagined-rollout lambda-returns). The counterfactual term
was updated to target the starting state's own value estimate (self-
consistent) instead of `progress[:, k_start+1]`. gamma=0.95.

Combined with the clamp fix, this recovered end distance to 0.62-0.81
(from 1.15) -- clearly better than the raw progress-target + object_rel
combination, though not quite back to the best 5-dim-proprio baseline
(0.52).

**CEM search radius tightening -- the biggest single behavioral finding
this round.** Probed whether raw `action_prior` execution (bypassing CEM
search entirely) would do better or worse than the full CEM(utility, IG)
pipeline. It did substantially *better*, consistently, across 30 held-out
episodes: min distance 0.164 vs. 0.219 (CEM), end distance 0.410 vs. 0.595.
CEM's population-based stochastic search (`init_std=0.5`, sampling widely
around the action_prior-seeded mean every step) was wandering away from an
already-good imitation trajectory more than it was refining it -- with an
imperfect utility/IG signal to score candidates, noisy search plus noisy
scoring compounds rather than helps. Tightening `init_std` 0.5 -> 0.1 (a
middle ground: still searches, just close to the seed) beat *both* the
wide-std baseline and raw action_prior alone at full scale (30 episodes):
end distance 0.39-0.69 depending on run (noisy, but consistently at or
near the best config found). Made this the new `CEMConfig` default.
Combining the tight search with `lambda0` (no exploration term at all)
did not help further -- IG wasn't the remaining bottleneck once CEM was
already tightened.

**Capacity vs. data ceiling, tested directly.** `bc_mse` (action_prior's
behavior-cloning loss) converged fast and fully plateaued by ~step 480 of
a 3510-step run, never improving further -- genuine convergence, not
undertraining. Tested whether the plateau was a capacity ceiling by
widening `ActionPrior` 256->512 hidden + a third layer: held-out bc_mse
barely moved, but closed-loop distance got measurably *worse* (baseline
min 0.220->0.317, end 0.554->1.206; tight_std_01 min 0.184->0.227, end
0.389->0.695). More capacity on the same fixed 500-demo dataset overfit
harder to training-demo idiosyncrasies and generalized worse to the
off-distribution states closed-loop rollout actually visits -- the same
distribution-shift signature as the object_rel and min_sigma regressions
earlier this investigation, now showing up a third time via a different
mechanism. Reverted to hidden=256/2 layers. Checked directly whether more
real demo data exists to download: `num_demos: 50` is baked into the
dataset's own HDF5 metadata -- this is the complete official LIBERO-v1
release for libero_spatial, not a partial download; there is no more data
to pull for free within this suite.

**Net result:** best configuration found this investigation (proprio 8-dim
incl. object_rel, TD(0) value function, tight CEM search, original
ActionPrior capacity) reduces end-of-episode gripper-to-target distance
by roughly half versus the original pre-Stage-2 baseline (from ~0.5-0.6
down to ~0.3-0.4, noisy across runs) via three independently-verified,
compounding fixes. Closed-loop task success remains 0/30 at full scale.
Every lever available within the current data budget and architecture
class (search strategy, exploration weighting, value target design,
network capacity, state representation richness) has now been tried;
the pattern across all of them -- real distance improvement, zero success,
and capacity/richer-state additions actively hurting without more data --
points at data volume and/or policy architecture class as the actual
remaining ceiling, not a fixable hyperparameter or design bug. Two
GPU-lost crashes this round too (see `[[project-via-model-hardware]]`),
both while training decision at verified 100W -- still not a fully
resolved situation, treated as tolerable per explicit user direction
rather than further investigated.

**Status:** this is the honest stopping point for configuration-level
iteration. Remaining paths are qualitatively bigger investments, not
further tuning: (1) genuinely more demonstration data (new collection, or
transfer from LIBERO's other suites -- libero_object/libero_goal/libero_10
-- which are different tasks but could still help shared visual/motor
representations), or (2) a materially more capable policy architecture
(diffusion policy, transformer policy, proper actor-critic) than a small
MLP action_prior + CEM, which was flagged as the "if all else fails" tier
back on 08-15 and has now had every cheaper alternative exhausted ahead
of it.

---

## 2026-08-23 — Feature-noise injection: first-ever non-zero closed-loop
success (1/30)

Rather than jumping straight to more data (unavailable) or a full policy
architecture rewrite (large undertaking), tried one more targeted fix
addressing the recurring root cause identified three separate times this
investigation (min_sigma, object_rel, oversized-ActionPrior all failed the
same way): `action_prior` was trained only on exact demo states, so it had
never had to produce a sensible action from a *slightly* off-trajectory
state -- exactly what live closed-loop rollout constantly puts it in, since
state estimation is never bit-exact and small errors compound over 300
steps. Added Gaussian noise to the RSSM feature input during the BC loss
only (`train/train_decision.py`), keeping the same action label -- teaches
local corrective behavior around the demonstrated trajectory rather than
only the exact trajectory itself (standard imitation-learning robustness
technique, DART-style state perturbation). Noise scale self-calibrated to
15% of each batch's own feature std, not a hand-tuned absolute value.

**Result: 1/30 successes, both variants (`full` and `action_prior_only`),
same task** ("pick up the black bowl next to the plate", task 8) --
the first non-zero closed-loop success recorded in this entire
investigation, across dozens of prior experiments. Distance metrics also
improved further: `full` min 0.177 / end 0.369 (vs. previous best ~0.18-
0.22 / ~0.39-0.60), `action_prior_only` min 0.156 / end 0.382.

**Status (updated same day, see below):** real, verified *precision*
progress, not asserted -- confirmed via the same full 30-episode-per-
variant scale used throughout. But the crossing into an actual success is
fragile, not reliable -- see the reproducibility check immediately below
before treating 1/30 as an established rate.

**Noise-scale sweep and a reproducibility check that matters.** Tried
`bc_feature_noise=0.3` (double the value that got the first success):
regressed back to 0/30 across all four ablation variants -- more noise
past some point destroys useful signal rather than adding robustness, so
0.15 isn't just "more is better." Reverted to 0.15 and **retrained decision
fresh from the same nominal config** to restore that checkpoint before
re-verifying -- this retrain, run independently (same seed value in
config, but a fresh process, so not guaranteed bit-identical given
dataloader/CUDA nondeterminism), reproduced the same strong distance
numbers on the `action_prior_only_check` diagnostic, but **the full
30-episode ablations run came back 0/30 this time, not 1/30.** The one
recorded success was real (verified at full scale, not a small-sample
fluke) but sits right at a fragile threshold that a fresh retrain at
nominally the same hyperparameters doesn't reliably re-cross -- meaningful
distinction from "solved," worth being precise about rather than reporting
the better of two numbers.

**Status:** feature-noise injection is a genuine, reproducible improvement
in approach precision (confirmed across three separate full-scale runs),
and did produce this investigation's only recorded closed-loop success,
but success itself remains a low-single-digit-percent, fragile event
rather than a reliably crossed bar. This is the honest state of the
project as of 2026-08-24: meaningfully closer than at any earlier point
in this investigation, genuinely at the edge of what the current data
budget (500 demos) and architecture class (small MLP action_prior + CEM)
can reliably deliver, not yet a solved task.

---

## 2026-08-24/27 — Direction 1: scripted data augmentation via trajectory
retargeting. 3x the training data, real quantifiable improvement, still
not solved

Per the user's request to size up "more data" vs. "different policy
architecture" as next directions, pursued more data first via a fully
autonomous, no-human-teleop path: generate new synthetic demonstrations
using privileged simulator state (the same `obj_of_interest` mechanism
used throughout this investigation's diagnostics), since the official
LIBERO-v1 release (50/task) is the complete real dataset -- confirmed no
more to download.

**First attempt -- from-scratch scripted P-controller
(`scripts/scripted_pick_place.py`): failed.** `akita_black_bowl`'s rim
geometry (~5cm radius, close to or exceeding the gripper's span) makes a
naive top-down centroid grasp genuinely hard to hand-tune -- exactly the
kind of precision problem this whole investigation has been fighting, now
encountered from the other side while trying to write an "expert."
Multiple hours of offset/gain tuning topped out around 15-30% success on
a single task, inconsistent across tasks.

**Pivot -- trajectory retargeting (`scripts/retargeted_demo_augment.py`):
worked.** Rather than rediscovering grasp geometry, translate a REAL
successful demo's recorded `ee_pos` trajectory by the position offset
between a new episode's object/destination and the base demo's own
(closed-loop P-control tracking of the retargeted waypoints, not open-loop
replay, so per-step dynamics differences self-correct). Two real bugs
found and fixed along the way, both worth remembering for future
sim-state work in this project:

1. **Objects start at a "drop" height right after `env.reset()`** and need
   ~5-10 physics steps to settle under gravity before their position is
   meaningful (confirmed: 0.97 unsettled -> 0.898 settled within 6 steps,
   matching the recorded demo exactly). The normal closed-loop eval loop
   never hit this because it naturally accumulates real steps before
   trusting positions; this script read position once, immediately, to
   compute a retargeting offset -- fixed with an explicit 10-step settle
   before reading.
2. **`sorted(glob.glob(...))`'s alphabetical file order does not match
   `LiberoEnvRunner`'s internal `task_id` ordering.** Tasks 2 and 4 were
   cleanly swapped (recorded(2)==live(4) and vice versa) -- confirmed via
   a systematic per-task check across all 10 tasks. Every other script in
   this investigation only ever used `LiberoEnvRunner`'s own task_id
   end-to-end and was never exposed to this; it's specific to mixing
   alphabetical file listing with live task_id in the same script. Fixed
   by matching demo files to live tasks by instruction text, not file
   order. Worth remembering for any future script that needs both a live
   env and an offline file for "the same" task.
3. **Nearest-neighbor base-demo selection** (precompute all 50 real demos'
   own settled object positions per task, pick whichever is closest to
   each new episode) rather than one fixed base demo -- a single fixed
   base works for nearby offsets but degrades badly for larger ones; some
   tasks' `demo_0` sits at an unusual corner of the position range,
   causing consistent large-offset failures without this.

**Result: 78% success rate generating data (47/60 across all 10 tasks
before the ordering-bug fix; higher after)**, then **1000 synthetic demos
generated at full scale (100/task, 83% overall yield, 1000/1201
attempts)** -- tripling the training set from 500 to 1500 demos. Merged as
companion HDF5s under the SAME task_id as their matching real file (not
new files/new tasks -- would have given goal_net's per-file classification
target two different labels for the same instruction text, see
`via/data/libero.py`'s `LiberoTrajectoryDataset`), always assigned to
split="train" (never "val" -- held-out checks throughout this project
measure generalization on real human-collected data specifically;
synthetic demos shouldn't dilute that).

**Retrained the full cascade** (belief -> world_model -> goal -> decision,
all fixes from the prior entries still in place: object-relative state,
TD(0) value target, tight CEM search, noise-augmented action_prior) on the
combined 1500-demo dataset. World-model high-motion win-rate improved
further (78-91% vs. the prior entry's 78%). Goal held at 100% accuracy.

**Closed-loop verification, pooled across three independent full-scale
(30-episodes-per-variant) runs: ~2/120 relevant episodes succeeded
(~1.7%).** Noisy at this rate -- a true ~2% underlying rate would show
zero successes in any given 30-episode sample roughly a third of the time
by chance, consistent with seeing 1/30, then 0/30, then 1/30 (action_prior
variant) across the three runs -- but this is a *repeated, nonzero*
outcome across independent runs, distinct from the true, exact 0% seen
across dozens of full-scale runs during every other phase of this
investigation. Distance metrics also improved further: `full` min=0.125
end=0.326 (best yet; compare Stage 2's original baseline 0.186/0.471).

**Status:** more data was the right call -- it produced this
investigation's first *repeated* (not just once-lucky) nonzero success
rate, on top of an already-real precision improvement. Still not reliably
solved (~2%, not a crossed bar), but the trend across this whole
investigation's cheap-diagnostic -> objective-rework -> search-tuning ->
data-volume progression has been consistently toward better numbers
without ever plateauing into "this doesn't help" -- data volume in
particular has now shown a clear, repeatable, quantifiable win where
several representation/architecture changes earlier showed regressions.
Natural next step, now that the augmentation pipeline is built and cheap
to rerun: generate substantially more synthetic demos (the current 1000
was a first pass, not a ceiling) and see whether the trend continues.

---

## 2026-08-27/29 — Direction 2: action-chunking policy. Largest single
improvement of the entire investigation (10% vs. ~1.7% pooled)

User pushback (rightly) on continuing to scale data alone, given the prior
entry's improvement was real but statistically thin at 30-episode sample
sizes. Reconsidered and prioritized the architecture direction flagged
earlier: `via/decision/chunking.py`'s `ActionChunkingPolicy` replaces
CEMPlanner + ActionPrior's single-step re-planning with a policy that
predicts a K=8 action chunk at once (small TransformerDecoder, K learned
query embeddings cross-attending to the current RSSM+goal feature,
self-attention among queries giving chunk-internal consistency) and
executes it before re-planning, directly targeting the compounding-error
failure mode literature identifies as endemic to single-step BC
regardless of data volume (Zhao et al. 2023, ACT). Deliberately simplified
vs. the ACT paper (no CVAE/style latent, no long-sequence transformer
encoder) to stay tractable at this project's data scale.

Trained on the full augmented dataset (500 real + 4000 synthetic, scaled
up from the prior entry's 1000) via `train/train_chunking.py` -- every
valid start position within each clip window supervises a full chunk (a
clip_len=16 window with chunk_size=8 yields 8 training examples, not 1).
Belief/RSSM/goal stayed on their prior (1500-demo) checkpoints for this
first test, a real inconsistency flagged going in and left for a
follow-up full-cascade retrain if this looked promising.

**Closed-loop eval (`eval/chunking_eval.py`) at full scale (30 episodes):
10% success (3/30)** -- a small 1-episode-per-task probe first showed 40%,
which settled to 10% at proper scale, itself a useful confirmation that
small samples in this project reliably overestimate (consistent with
every earlier finding here about 30-episode noise). Even the conservative
number is roughly 5-6x the CEM pipeline's best pooled rate (~1.7%) --
the single largest jump of this entire investigation, in one change.
Belief/RSSM/goal update every real control step as before (still
filtering on real evidence); only the policy re-plans every K=8 steps
instead of every single step.

**Status:** the user's challenge to reconsider direction was directly
vindicated -- architecture, not data volume, appears to have been the
larger lever for the compounding-error failure mode this investigation
kept running into. Next: (1) verify reproducibility with a second
independent full-scale run (the CEM pipeline's own history of a
non-reproducing single "success" is exactly the failure mode to rule out
here before trusting this number), (2) if it holds, retrain belief/
world_model/goal on the full 4500-demo set for full consistency, then
retrain chunking on top, to see whether data volume and architecture
compound further together.

**Reproducibility check: confirmed, and more strongly than expected.** A
second independent full-scale run gave the exact same 10% (3/30), with an
identical per-task breakdown (task 3: 1/3, task 7: 2/3, all others 0/3) --
not just a similar rate, bit-for-bit the same pattern. This pipeline is
essentially deterministic given the fixed seed (env reset ordering + a
chunking policy with no explicit stochasticity in its forward pass), so
this is a real, stable result for this checkpoint, not a lucky roll --
the cleanest confirmation of any result in this investigation. Proceeding
to the full-cascade retrain on the 4500-demo set.

---
