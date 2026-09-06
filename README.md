# VIA: Vision-Inference-Action

**Toward Cognitively Grounded VLA Models** — perception as inference, language
as evidence, action as decision. (SDUTC Track 2, Research.)

Claude was used in the development of this project.

Standard VLAs map observations directly to actions with no uncertainty, no
memory, and no planning. VIA replaces that stimulus-response mapping with four
modules, each implementing a principle from Bayesian cognitive science, with
probability distributions flowing through the entire system:

| Module | Cognitive principle | Implementation |
|---|---|---|
| Perception ([via/perception](via/perception/siglip.py)) | perception provides evidence | frozen SigLIP ViT-B/16 → 196×768 patch embeddings |
| Belief state ([via/belief](via/belief/belief_state.py)) | perception as probabilistic inference (Helmholtz; Rao & Ballard 1999) | GRU filter → N(μ, σ²) over a 256-d latent world state; σ rises under occlusion |
| Goal inference ([via/goal](via/goal/rsa.py)) | language as Bayesian goal inference (RSA; Goodman & Frank 2016) | language cross-attends to belief; pragmatic listener L1 ∝ L0^α · P(g\|belief) over K goal slots |
| World model ([via/world_model](via/world_model/rssm.py)) | planning as mental simulation | RSSM p(s′\|s, a) in the belief's observation-embedding space |
| Decision ([via/decision](via/decision/decision.py)) | action as expected utility + epistemic value (Friston 2017) | CEM over J(a) = E[U] + λ(σ)·IG(a), with a learned adaptive gate λ |

The information-gain term is computed by imagining future observations with
the world model, pushing them back through the belief filter, and measuring
the entropy drop — the agent literally asks "how much less confused would I be
if I did this and saw what follows?"

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                          # full contract + forward-pass suite, CPU, ~1 min
```

Everything runs offline on CPU via contract-identical stub encoders
(`StubPerception`, `StubLanguageEncoder`). On the GPU box, additionally:

```bash
pip install "torch>=2.4" "transformers>=4.44"      # real SigLIP + Phi-3
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..
python LIBERO/benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
```

then set `encoders.vision/language` to the HF model names and
`data.source: libero` in [configs/default.yaml](configs/default.yaml).

**Currently running locally on a Windows/RTX 3060 Ti box** (the GPU/eval
box referenced above is a local machine, not a shared cluster — DSMLP
access was abandoned after repeated infra failures, see
[EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)). That setup has its own venv
location, path conventions, and three Windows-specific patches needed for
the LIBERO simulator to work at all — see
[docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md) before assuming the generic
setup above applies as-is, and use `configs/local.yaml` (not
`default.yaml`) for real-data work in that environment.

## Training (module by module, per the proposal timeline)

Each stage has a `--smoke` flag that runs a tiny CPU pass end-to-end — use it
to validate the pipeline before spending GPU hours.

```bash
python -m train.train_belief          # 1. belief state (self-supervised next-obs NLL)
python -m train.train_world_model     # 2. RSSM on trajectories (needs belief ckpt)
python -m train.train_goal            # 3. RSA goal inference (synthetic pretraining)
python -m train.train_decision        # 4. utility head + adaptive gate init
```

## Evaluation

```bash
python -m eval.uncertainty_analysis   # Checkpoint 1: σ occluded/visible ratio + trace figure
python -m eval.eval_libero --suite libero_spatial   # success rates + diagnostics JSON
python -m eval.ablations              # full vs λ=0 vs point-goal vs no-belief
```

`eval/ablations.py` produces the paper's primary table: it isolates the
contribution of the epistemic-value term (`lambda0`), the goal *distribution*
(`point_goal`), and the belief variance (`no_belief`).

## Milestone → code map

| Milestone (proposal) | Where |
|---|---|
| Tensor contracts + stubs pass forward test | [via/contracts.py](via/contracts.py), `pytest tests/test_pipeline.py` |
| σ spikes on occlusion | [eval/uncertainty_analysis.py](eval/uncertainty_analysis.py) |
| 5-step rollout beats naive baseline | `rollout5_*` metrics in [train/train_world_model.py](train/train_world_model.py) |
| Goal entropy tracks ambiguity | `entropy_amb{0,1,2}` metrics in [train/train_goal.py](train/train_goal.py) |
| End-to-end on a LIBERO suite | [eval/eval_libero.py](eval/eval_libero.py) |
| λ=0 vs full ablation table | [eval/ablations.py](eval/ablations.py) |

## Extra (outside the proposal)

Not part of the original grant timeline or milestone table above; exploratory
work kept clearly separate from the proposal deliverables:

```bash
python -m eval.integration_probe      # encoding-vs-integration probe ("memory without meaning")
```

See [docs/embodied_comprehension_bridge.md](docs/embodied_comprehension_bridge.md)
for the framing (grounded-cognition angle on VIA) this eval comes from.

## Repo layout

```
via/            the model package (perception, belief, goal, world_model, decision, model.py)
via/data/       LIBERO hdf5 loader + synthetic generators (offline-friendly)
train/          one script per module + shared utilities (W&B, checkpoints)
eval/           success rate, uncertainty analysis, ablations
tests/          contract tests per module + end-to-end forward pass
configs/        default.yaml (dims, hyperparams, encoder/data selection)
```
