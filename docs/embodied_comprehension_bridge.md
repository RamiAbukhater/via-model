# Handoff: Embodied Language Comprehension → VIA

**Status:** Q3 implemented ([eval/integration_probe.py](../eval/integration_probe.py),
smoke-tested, kept out of the proposal's milestone table — see README's "Extra"
section). Q1 and Q2 remain conceptual/exploratory pending a decision on whether
to pursue them. Purpose: decide whether the "is comprehension constituted by
simulation, or is simulation an epiphenomenon?" question from the grounded-cognition
literature can be turned into a research angle *inside* VIA, and what it would
touch if so.

Source material: COGS 153 HW1, Topic 1 ("Computational and Neural Models of Embodied
Language Comprehension"). Key references carried over: Barsalou (2008), Feldman &
Narayanan (2004), Boulenger et al. (2006), Tettamanti et al. (2005), Chatterjee (2010).

---

## 1. The idea in one paragraph

Grounded/embodied theories claim that understanding language *partly re-runs* the
brain's perceptual and motor systems — reading "kick" reactivates leg-motor circuitry.
The open, hard question is whether that reactivation is **constitutive** of meaning
(the simulation *is* the understanding) or merely an **epiphenomenon** (a downstream
co-activation that rides along but does no work). The whole difficulty in humans is
that you cannot cleanly lesion the motor system to check, and reaction-time or fMRI
co-activation shows *that* activity happens, never *that it is necessary* — Chatterjee's
"activation is not constitution." VIA is interesting here because it is a working system
in which the "perceptual-motor simulation" substrate is an explicit, ablatable module.
In other words, VIA can be treated as a **model organism** for the constitution-vs-
epiphenomenon question that the human paradigms can only circle.

## 2. Why VIA is the right substrate

VIA already commits to language-as-evidence and perception-as-inference, so the mapping
is unusually direct. The human "sensorimotor system that may or may not constitute
meaning" has a clean analog in VIA's simulation machinery:

| Embodied-comprehension construct | VIA analog |
|---|---|
| Sensorimotor simulation (Barsalou's re-enactment; Feldman & Narayanan's x-schemas) | The **RSSM world model** rolling out `p(s′\|s,a)` in belief-embedding space — "planning as mental simulation" |
| Comprehending an utterance | **RSA goal inference** — language cross-attends to belief to infer a goal distribution |
| Effector-specific motor recruitment (Tettamanti) | Action-conditioned rollouts specific to the commanded action/effector |
| "Activation is not constitution" (Chatterjee) | The difference between language *reaching* a module and language *changing its distributions* |

The critical observation: in VIA, language currently flows into **goal inference**, and
the world model is used for **planning/epistemic value**. The embodied claim, translated,
is a claim about whether comprehension *must route through* the simulation substrate
(belief + world model) or can sit in an amodal goal slot that never engages it. That is
exactly a wiring/ablation question VIA can pose to itself.

## 3. The three questions worth stealing from the literature

**Q1 — Constitution vs. epiphenomenon (necessity).**
Does language comprehension in VIA *require* the world-model simulation, or does it
survive when simulation is severed? Human work can't lesion motor cortex; VIA can
ablate the language→world-model path (or the world model itself) and watch whether goal
inference and task success degrade. If comprehension survives intact without simulation,
VIA's "understanding" is amodal and the grounding is epiphenomenal *in this system*. If
it collapses, simulation is constitutive here. Either result is a real finding and maps
onto VIA's existing ablation philosophy (`eval/ablations.py` already isolates `no_belief`,
`point_goal`, `lambda0`).

**Q2 — Timing (when does simulation happen relative to comprehension?).**
Boulenger et al. found the sign of the language-motor interaction *flips* with timing —
facilitation before movement onset, interference within ~200 ms of it. VIA's analog is
*when* linguistic evidence is injected relative to the belief update / rollout step:
language provided before a rollout vs. concurrent with it. A clean model experiment is to
vary the injection step and measure whether early language helps and concurrent language
disrupts — a computational echo of the timing-dependent cross-talk.

**Q3 — Memory vs. meaning (the integration signature).**
This is the sharpest borrow, and it comes from Topic 2's ERP work (Mangardich & Sabbagh's
N400 "memory without meaning"), but it's just as usable here: distinguish an instruction
that is *encoded* (present in the language embedding, reportable) from one that is
*integrated* (actually shifts the belief/goal posterior and the rollout). Define an
**integration index** — e.g., the KL divergence the instruction induces in the goal or
belief distribution, or the entropy drop it produces — and contrast it with mere
recoverability of the instruction tokens. A word VIA "remembers" but that moves no
downstream distribution is the model's version of memory without meaning. An **N400
analog** falls out naturally: the prediction-error/surprise signal when a world-model
rollout conditioned on the instruction meets an observation that contradicts it.

## 4. What this would touch (light, for scoping only)

Nothing here is a committed change — just where each question would land:

- **Q1 (necessity):** a new ablation sibling in `eval/ablations.py` that cuts the
  language→world-model influence (or zeros the rollout) while leaving goal inference wired,
  vs. the reverse. Compares against the existing `no_belief` / `point_goal` columns.
- **Q2 (timing):** a controlled eval that varies the step at which language evidence
  enters the belief filter relative to the action rollout; measure success and goal-entropy
  as a function of injection timing.
- **Q3 (integration) — done:** [eval/integration_probe.py](../eval/integration_probe.py)
  logs the instruction-induced KL/entropy shift in the goal posterior (integration index,
  entropy drop) next to the grounded utterance embedding's distance from a null
  instruction (encoding distance). The gap between them is the memory-vs-meaning measure.
  Scoped down from the original idea: the rollout-vs-observation N400 analog is deferred —
  it needs language to condition the world model, which isn't wired yet (Q1 territory).

All three reuse machinery VIA already has (belief posteriors, RSSM rollouts, the ablation
harness), so the lift is instrumentation and analysis, not new architecture.

## 5. Why bother — the payoff

The reason this is more than a cute analogy: **VIA can give a cleaner answer to the
constitution question than any human study, because it can actually sever the simulation
substrate.** That reframes VIA from "a cognitively grounded VLA" into a testbed for a
30-year-old debate in grounded cognition — a defensible research contribution, and a
paper angle that ablations already support. It also tightens VIA's own story: if grounding
turns out to be *necessary* for comprehension in the model, that is direct evidence for
the design thesis in the README; if it's epiphenomenal, that's a finding that tells you
where the architecture is really doing its work.

## 6. Note on the sibling (trust) thread

The COGS 153 course project is going with Topic 2 (perceived trust / epistemic vigilance),
not this one — but the two share machinery worth flagging. Epistemic vigilance is, in VIA
terms, **weighting language evidence by source reliability inside the RSA listener** — a
trust prior on the `P(g|belief)` update. So if the trust line ever wants a computational
home, it lands in the *same* goal-inference module, as a reliability-weighted evidence
term. Worth keeping in mind so the two threads don't get implemented twice.
