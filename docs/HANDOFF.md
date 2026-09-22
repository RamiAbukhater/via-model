# Handoff — VIA (SDUTC Track 2)

Written 2026-09-21, for a new agent/session picking this project up with no
prior context. Read this file first; it links to everything else.

## What this project is

VIA is a cognitively-grounded VLA (vision-language-action) model for robot
manipulation: a belief state (GRU + Gaussian uncertainty) feeds a goal
inference module (RSA — Rational Speech Acts), a world model (RSSM,
Dreamer-style), and a decision module, evaluated closed-loop on the LIBERO
benchmark (`libero_spatial` suite, 10 pick-and-place tasks, robosuite/MuJoCo
sim). Two candidate decision/policy architectures exist side by side in this
repo:

1. **CEM+MLP pipeline** (`via/decision/decision.py`) — the original design:
   cross-entropy-method planning over an `ActionPrior` + learned `UtilityHead`
   + `AdaptiveGate`.
2. **Action-chunking policy** (`via/decision/chunking.py`, added later) — a
   small transformer that predicts an 8-step action chunk from one context
   vector and re-plans every 8 steps instead of every step (ACT-inspired,
   Zhao et al. 2023). This is the one that actually worked.

## The validated result (what's in the paper)

Action-chunking reached a **reproducible 10% closed-loop success rate
(3/30 episodes, `libero_spatial`)** — confirmed bit-for-bit identical across
two independent full runs (same 3/30 split: task 3 1/3, task 7 2/3, all
else 0/3). The CEM+MLP pipeline topped out around **1.7% pooled** after
three rounds of tuning (proprioception, TD(0) value target, CEM search-radius
tightening). This comparison is what `documents/paper/via_sdutc_paper.tex`
reports (that file lives outside this repo, at
`C:\Users\super\OneDrive\Desktop\VIA\documents\paper\`, not under git — see
"Paper" below). It is real, honestly-reproduced science — not in question.

Full derivation of both numbers, and everything that was tried and ruled
out to get there, is in `docs/EXPERIMENT_LOG.md` (see its 2026-08-27/29 and
earlier entries).

## The one thing you must not skip: checkpoint state right now

**The checkpoints currently on disk (`D:\via-data\checkpoints\`) do NOT
reproduce the paper's 10% number.** Here's why and what's actually there:

After the 10% result was confirmed, a further retrain was attempted
(2026-09-01/03, full detail in `EXPERIMENT_LOG.md`'s entry of that date) to
bring `belief`/`world_model`/`goal` up to the full 4500-demo dataset size.
That run took 48+ hours, ran into the Sept 4 paper deadline, was killed
mid-training, and in the process **partially overwrote `belief.pt` in
place** — the exact checkpoint the 10% result depended on no longer exists
and cannot be regenerated bit-for-bit (training isn't that deterministic
across a fresh run). A time-boxed recovery retrain (`world_model`/`goal`/
`chunking_policy`, all with epoch counts cut for time — see
`configs/local.yaml`'s inline comments) was built on top of the new
`belief.pt` to at least get a self-consistent set of checkpoints before the
deadline. **That recovered pipeline verified at 0/30, not 10%.**

Concretely, as of 2026-09-03 (unchanged since):

| Checkpoint | Last trained | Status |
|---|---|---|
| `belief.pt`, `world_model.pt`, `goal.pt`, `chunking_policy.pt` | 2026-09-01/03 | Mutually consistent (all trained on top of each other in sequence), but this *combination* is only verified at **0/30** |
| `utility.pt`, `gate.pt`, `action_prior.pt` (CEM pipeline) | 2026-08-26 | **Stale** — trained against the *original*, now-overwritten belief/world_model/goal. Mismatched with what's currently on disk. |

**Do not run `eval/eval_libero.py` or `eval/ablations.py` (the CEM
pipeline) against current checkpoints and treat the result as meaningful —
it's evaluating a version-mismatched combination**, the same failure mode
this project diagnosed multiple times elsewhere (see "richer without more
robust" pattern in EXPERIMENT_LOG.md) via a different route. Retrain
`train/train_decision.py` on the current belief first if you need a fair
CEM-pipeline number.

Also note `configs/local.yaml` still has the deadline-cut values
(`world_model.epochs: 2`, `goal.libero_epochs: 2`, both commented with the
cut rationale) — restore to `20` / `5` respectively before any from-scratch
retrain meant to match the project's original validated quality bar.

## Suggested next steps, roughly in order of value

1. **Reproduce a working chunking checkpoint.** Either retrain the original
   1500-demo cascade (belief -> world_model -> goal -> chunking) with the
   restored (non-deadline-cut) epoch counts, or accept the current 4500-demo
   belief and retrain world_model/goal/chunking on top of it at full epochs
   (not the cut 2/2) to see if a properly-trained version of this recovery
   path reaches 10% or better. Either result is informative and should be
   logged in `EXPERIMENT_LOG.md` the same way everything else has been.
2. **Retrain the CEM pipeline** (`train/train_decision.py`) against whichever
   belief you land on in (1), for a fair, current, apples-to-apples
   chunking-vs-CEM comparison (the paper's comparison used the earlier,
   now-gone checkpoint combination).
3. **Expand beyond `libero_spatial`** to LIBERO's Object/Goal/Long suites, if
   the project continues past this paper's deadline — everything here
   (data pipeline, augmentation scripts, both policy architectures) should
   generalize directly; only `LiberoEnvRunner`'s suite argument changes.
4. **A fuller ACT implementation** (CVAE latent, longer-context encoder) is
   the natural next architecture step if chunking keeps being the strongest
   lever — the current `ActionChunkingPolicy` docstring
   (`via/decision/chunking.py`) explicitly documents which ACT features
   were simplified away for time.

None of these are committed to — they're this session's best judgment of
where marginal effort pays off most, not instructions.

## Hardware (read before starting any training)

RTX 3060 Ti, history of "GPU is lost" crashes traced to a daisy-chained
PCIe power cable. **Power cap must be at 100W** (confirmed reliable; 130W
crashed repeatedly) and **does not survive a reboot** — check with
`nvidia-smi --query-gpu=power.limit --format=csv` and reapply with
`nvidia-smi -pl 100` (elevated PowerShell) if a session starts after any
reboot. A replacement PCIe cable (the real fix) was obtained but, last
confirmed, not yet physically installed — ask the user before assuming
otherwise. Full detail: `[[project-via-model-hardware]]` memory and
`docs/LOCAL_SETUP.md`'s hardware section.

## Where everything else is

- **`docs/LOCAL_SETUP.md`** — environment setup, venv location, LIBERO/
  robosuite Windows patches, config file to use (`configs/local.yaml`).
- **`docs/EXPERIMENT_LOG.md`** — the full narrative, dated entries, every
  hypothesis tried/confirmed/ruled out from project start through the
  2026-09-01/03 crisis. The primary source of truth for "why is it built
  this way."
- **Paper**: `C:\Users\super\OneDrive\Desktop\VIA\documents\paper\
  via_sdutc_paper.tex` — NOT in this git repo (that whole directory tree
  isn't under version control). Compiles with `pdflatex` (MiKTeX). Reports
  the validated 10%/1.7% comparison above; one deliberately-unfilled gap
  remains (a footnote noting a concurrent-work citation is "still being
  finalized" — search the .tex for that phrase if closing it out).
- **This repo's git**: branch `main`, remote `origin` ->
  `https://github.com/RamiAbukhater/via-model.git`. Last commit as of this
  writing: `4c91b13`. The parent `VIA/` folder and `documents/paper/` are
  outside this repo and not tracked by git at all — don't expect `git
  status` run from `via-model/` to see paper changes.
