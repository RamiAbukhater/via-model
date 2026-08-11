"""EXTRA — not part of the original grant proposal/timeline. Companion eval
for docs/embodied_comprehension_bridge.md, Q3: does an instruction that's
linguistically encoded also get integrated into the goal posterior, or can
the two come apart?

    encoding distance  = ||utterance_embedding(u) - utterance_embedding("")||
                         (is u even represented differently from no
                         instruction? via GoalInferenceRSA.utterance_embedding)
    integration index  = KL(L1(g|u,b) || L1(g|"",b))
                         (how far u actually moves the goal posterior from
                         the belief-only baseline)
    entropy drop       = H(L1(g|"",b)) - H(L1(g|u,b))
                         (how much u narrows the posterior)

Run over SyntheticInstructionDataset's three ambiguity levels (full
instruction / object dropped / verb+object dropped). Expectation: encoding
distance stays roughly flat across levels, while integration index and
entropy drop fall as ambiguity rises. A large encoding distance paired with
a small integration index is this system's analog of "memory without
meaning" (Mangardich & Sabbagh's N400 finding).

Not covered here: the N400 surprise analog (a world-model rollout that
contradicts the instruction) needs language to condition the world model,
which VIA doesn't wire yet — that's Q1 territory.

    python -m eval.integration_probe            # trained goal ckpt + config
    python -m eval.integration_probe --smoke     # untrained, CPU sanity check

Writes results/integration_probe.csv.
"""

import csv
from collections import defaultdict

import torch

from train import common
from via.belief import BeliefStateNetwork
from via.data.synthetic import SyntheticInstructionDataset, SyntheticTrajectoryDataset
from via.goal import GoalInferenceRSA


@torch.no_grad()
def grounding_belief_mu(perception, belief_net, cfg, device) -> torch.Tensor:
    """A single non-trivial belief context to condition goal inference on —
    the terminal mu of one rolled-out synthetic clip, mirroring the
    belief-conditions-goal flow in via/model.py's act()."""
    clip = SyntheticTrajectoryDataset(size=1, clip_len=cfg["data"]["clip_len"], seed=999)[0]
    patches = common.encode_frames(perception, clip["frames"].unsqueeze(0).to(device))
    beliefs = belief_net.rollout(patches)
    return beliefs[-1].mu


@torch.no_grad()
def probe(goal_net, language, belief_mu, items) -> list[dict]:
    device = belief_mu.device
    null_tokens, null_mask = language([""])
    null_tokens, null_mask = null_tokens.to(device), null_mask.to(device)
    null_dist = goal_net(null_tokens, null_mask, belief_mu)
    null_embed = goal_net.utterance_embedding(null_tokens, null_mask, belief_mu)

    rows = []
    for item in items:
        tokens, mask = language([item["instruction"]])
        tokens, mask = tokens.to(device), mask.to(device)
        dist = goal_net(tokens, mask, belief_mu)
        embed = goal_net.utterance_embedding(tokens, mask, belief_mu)

        encoding_distance = (embed - null_embed).norm(dim=-1)
        integration_index = (dist.probs * (dist.log_probs - null_dist.log_probs)).sum(-1)
        entropy_drop = null_dist.entropy() - dist.entropy()
        correct = (dist.probs.argmax(-1) == item["goal"]).float()

        rows.append({
            "instruction": item["instruction"],
            "ambiguity": item["ambiguity"],
            "encoding_distance": encoding_distance.mean().item(),
            "integration_index": integration_index.mean().item(),
            "entropy_drop": entropy_drop.mean().item(),
            "goal_correct": correct.mean().item(),
        })
    return rows


def main() -> None:
    p = common.base_parser(__doc__)
    p.add_argument("--n-per-level", type=int, default=32)
    args = p.parse_args()
    cfg = common.load_config(args.config)
    common.set_seed(cfg["seed"])
    device = torch.device(args.device)

    perception = common.build_perception(cfg, args.smoke).to(device)
    language = common.build_language(cfg, args.smoke).to(device)
    belief_net = BeliefStateNetwork().to(device)
    goal_net = GoalInferenceRSA().to(device)
    if not args.smoke:
        common.load_checkpoint(belief_net, cfg, "belief", args.device)
        common.load_checkpoint(goal_net, cfg, "goal", args.device)
    belief_net.eval()
    goal_net.eval()

    belief_mu = grounding_belief_mu(perception, belief_net, cfg, device)

    n_per_level = 2 if args.smoke else args.n_per_level
    dataset = SyntheticInstructionDataset(size=n_per_level * 20, seed=999)
    by_level = defaultdict(list)
    for item in dataset:
        if len(by_level[item["ambiguity"]]) < n_per_level:
            by_level[item["ambiguity"]].append(item)
    items = [it for lvl in sorted(by_level) for it in by_level[lvl]]

    rows = probe(goal_net, language, belief_mu, items)

    out_dir = common.REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "integration_probe.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'level':>5} {'n':>4} {'encoding_dist':>14} {'integration_idx':>16} {'entropy_drop':>13}")
    for lvl in sorted(by_level):
        lvl_rows = [r for r in rows if r["ambiguity"] == lvl]
        n = len(lvl_rows)
        enc = sum(r["encoding_distance"] for r in lvl_rows) / n
        integ = sum(r["integration_index"] for r in lvl_rows) / n
        ent = sum(r["entropy_drop"] for r in lvl_rows) / n
        print(f"{lvl:>5} {n:>4} {enc:>14.4f} {integ:>16.4f} {ent:>13.4f}")

    print(f"wrote {csv_path}")
    print("want: encoding_dist roughly flat across levels, integration_idx/entropy_drop "
          "falling 0 > 1 > 2 (encoded-but-not-integrated = memory-without-meaning analog)")


if __name__ == "__main__":
    main()
