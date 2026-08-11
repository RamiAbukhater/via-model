# Handoff: Embodied Language Comprehension → VIA

Status: Q3 implemented ([eval/integration_probe.py](../eval/integration_probe.py),
smoke-tested, kept out of the proposal's milestone table — see README's "Extra"
section). Q1 and Q2 are still just ideas. Purpose: figure out whether the
"is comprehension constituted by simulation, or is simulation an
epiphenomenon?" question from grounded-cognition can become a real research
angle inside VIA, and what it would touch if so.

Source: COGS 153 HW1, Topic 1 ("Computational and Neural Models of Embodied
Language Comprehension"). References: Barsalou (2008), Feldman & Narayanan
(2004), Boulenger et al. (2006), Tettamanti et al. (2005), Chatterjee (2010).

---

## 1. The idea

Grounded/embodied theories claim that understanding language partly re-runs
the brain's perceptual and motor systems — reading "kick" reactivates
leg-motor circuitry. The hard question is whether that reactivation is
constitutive of meaning or just an epiphenomenon riding along without doing
any work. Humans can't cleanly lesion the motor system to check, and
co-activation (reaction time, fMRI) shows that activity happens, never that
it's necessary — Chatterjee's "activation is not constitution." VIA is
interesting here because the "perceptual-motor simulation" substrate is an
explicit, ablatable module, not something you have to infer from indirect
signals.

## 2. Why VIA fits

VIA already commits to language-as-evidence and perception-as-inference, so
the mapping is fairly direct:

| Embodied-comprehension construct | VIA analog |
|---|---|
| Sensorimotor simulation (Barsalou's re-enactment; Feldman & Narayanan's x-schemas) | The RSSM world model rolling out `p(s′\|s,a)` in belief-embedding space |
| Comprehending an utterance | RSA goal inference — language cross-attends to belief to infer a goal distribution |
| Effector-specific motor recruitment (Tettamanti) | Action-conditioned rollouts specific to the commanded action/effector |
| "Activation is not constitution" (Chatterjee) | The difference between language reaching a module and language changing its distributions |

In VIA, language currently flows into goal inference, and the world model is
used for planning/epistemic value. Translated, the embodied claim becomes:
does comprehension have to route through the simulation substrate (belief +
world model), or can it sit in an amodal goal slot that never touches it?
That's a wiring/ablation question VIA can pose to itself.

## 3. Three questions worth borrowing

**Q1 — Constitution vs. epiphenomenon (necessity).** Does language
comprehension in VIA require the world-model simulation, or does it survive
when simulation is severed? Ablate the language→world-model path (or the
world model itself) and watch whether goal inference and task success
degrade. If comprehension survives without simulation, VIA's "understanding"
is amodal here. If it collapses, simulation is constitutive. Either result
is real, and it maps onto VIA's existing ablation setup (`eval/ablations.py`
already isolates `no_belief`, `point_goal`, `lambda0`).

**Q2 — Timing.** Boulenger et al. found the language-motor interaction flips
sign with timing — facilitation before movement onset, interference within
~200ms of it. VIA's analog is when linguistic evidence enters the belief
update / rollout step, relative to it. Vary the injection step and measure
whether early language helps and concurrent language disrupts.

**Q3 — Memory vs. meaning.** Borrowed from Topic 2's ERP work (Mangardich &
Sabbagh's N400 "memory without meaning"): distinguish an instruction that's
encoded (present in the language embedding, reportable) from one that's
integrated (actually shifts the belief/goal posterior). Define an
integration index — the KL divergence or entropy drop the instruction
induces in the goal/belief distribution — and contrast it with mere
recoverability of the instruction tokens. A word VIA "remembers" but that
moves no downstream distribution is the model's version of memory without
meaning. An N400 analog falls out too: the surprise signal when a
world-model rollout conditioned on the instruction meets a contradicting
observation.

## 4. Scope (no committed changes beyond Q3)

- **Q1:** a new ablation sibling in `eval/ablations.py` that cuts the
  language→world-model influence while leaving goal inference wired, vs.
  the reverse. Compares against `no_belief` / `point_goal`.
- **Q2:** a controlled eval varying the step at which language enters the
  belief filter relative to the action rollout; measure success and
  goal-entropy as a function of injection timing.
- **Q3 — done:** [eval/integration_probe.py](../eval/integration_probe.py)
  logs the instruction-induced KL/entropy shift in the goal posterior next
  to the grounded utterance embedding's distance from a null instruction.
  The gap between them is the memory-vs-meaning measure. The N400
  rollout-mismatch half is deferred — needs language to condition the world
  model, which isn't wired yet (Q1 territory).

All three reuse machinery VIA already has (belief posteriors, RSSM
rollouts, the ablation harness) — instrumentation and analysis, not new
architecture.

## 5. Why bother

VIA can actually sever the simulation substrate, which gives a cleaner
answer to the constitution question than a human study can. That's a real
angle: a cognitively-grounded VLA doubling as a testbed for a grounded
cognition debate. It also sharpens VIA's own story either way — if grounding
turns out necessary for comprehension in the model, that supports the
README's design thesis; if it's epiphenomenal, that tells you where the
architecture is actually doing its work.

## 6. Note on the sibling (trust) thread

The COGS 153 course project is going with Topic 2 (perceived trust /
epistemic vigilance), not this one, but the two share machinery. Epistemic
vigilance is, in VIA terms, weighting language evidence by source
reliability inside the RSA listener — a trust prior on the `P(g|belief)`
update. If the trust line ever wants a computational home, it lands in the
same goal-inference module. Worth keeping in mind so the two don't get
implemented twice.
