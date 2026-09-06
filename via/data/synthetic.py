"""Synthetic data generators.

Two purposes, both from the proposal's timeline:

- `SyntheticTrajectoryDataset`: action-conditioned moving-square videos with
  scheduled *occlusion windows* (frames blanked). Used for belief/RSSM smoke
  training and for the qualitative test that belief sigma spikes under
  occlusion before any LIBERO data is available.

- `SyntheticInstructionDataset`: templated instruction/goal pairs at graded
  ambiguity levels for RSA pretraining ("goal distribution entropy correlates
  with instruction ambiguity" milestone). Goals are (verb, object) combos
  mapped onto the K goal slots; ambiguous instructions drop the object or
  the verb, so several slots are consistent with the utterance.
"""

import torch
from torch.utils.data import Dataset

from via.contracts import C

# ---- instruction templates ----

VERBS = ["pick up", "push", "open", "close"]
OBJECTS = ["the red block", "the blue block", "the bowl", "the mug",
           "the drawer", "the plate", "the bottle", "the box"]
# K = len(VERBS) * len(OBJECTS) = 32 = C.goal_slots
AMBIGUOUS_OBJECTS = {"something", "that thing", "it", "the object"}


def goal_slot(verb: str, obj: str) -> int:
    return VERBS.index(verb) * len(OBJECTS) + OBJECTS.index(obj)


class SyntheticInstructionDataset(Dataset):
    """(instruction, goal_slot, ambiguity in {0,1,2}) triples.

    ambiguity 0: full instruction        "pick up the red block"
    ambiguity 1: object omitted          "pick up something"
    ambiguity 2: verb + object omitted   "do something with that thing"
    The goal label is always the true underlying goal; the model can only
    lower its loss on ambiguous utterances by spreading probability, which is
    exactly the calibration we want to measure.
    """

    def __init__(self, size: int = 4096, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(size):
            v = VERBS[int(torch.randint(len(VERBS), (1,), generator=gen))]
            o = OBJECTS[int(torch.randint(len(OBJECTS), (1,), generator=gen))]
            amb = int(torch.randint(3, (1,), generator=gen))
            if amb == 0:
                text = f"{v} {o}"
            elif amb == 1:
                text = f"{v} something"
            else:
                text = "do something with that thing"
            self.items.append({"instruction": text, "goal": goal_slot(v, o), "ambiguity": amb})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class SyntheticTrajectoryDataset(Dataset):
    """Action-conditioned moving-square clips with occlusion windows.

    A 32px colored square moves on a gray canvas; actions[:, :2] are its
    velocity (so the dynamics are genuinely action-conditioned and the RSSM
    has something real to learn). Frames inside the occlusion window are
    blanked to zeros — the observability signal for uncertainty tests.
    """

    def __init__(self, size: int = 256, clip_len: int = 12, occlude: bool = True, seed: int = 0):
        self.size = size
        self.clip_len = clip_len
        self.occlude = occlude
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> dict:
        gen = torch.Generator().manual_seed(self.seed * 100_003 + i)
        T, S, sq = self.clip_len, C.image_size, 32
        pos = torch.rand(2, generator=gen) * (S - sq)
        color = torch.rand(3, generator=gen) * 0.7 + 0.3
        actions = (torch.rand(T, C.action_dim, generator=gen) * 2 - 1)

        occ_start, occ_end = -1, -1
        if self.occlude and T >= 5:
            # Keep at least 2 visible frames on each side of the window.
            hi = max(T - 3, 3)
            occ_start = int(torch.randint(2, hi, (1,), generator=gen))
            occ_end = min(T - 1, occ_start + 3)

        frames = torch.full((T, 3, S, S), 0.5)
        occluded = torch.zeros(T, dtype=torch.bool)
        for t in range(T):
            pos = (pos + actions[t, :2] * 10.0).clamp(0, S - sq)
            if occ_start <= t < occ_end:
                frames[t] = 0.0
                occluded[t] = True
            else:
                x, y = int(pos[0]), int(pos[1])
                frames[t, :, y : y + sq, x : x + sq] = color.view(3, 1, 1)

        reward = torch.zeros(T)
        reward[-1] = 1.0  # sparse terminal reward, matching LiberoTrajectoryDataset's convention

        return {
            "frames": frames,                                   # (T, 3, 224, 224)
            "actions": actions,                                 # (T, 7)
            "occluded": occluded,                               # (T,)
            "progress": torch.linspace(0.0, 1.0, T),            # (T,)
            "reward": reward,                                   # (T,)
            # No real gripper/end-effector in this synthetic task -- zeros,
            # not a fabricated signal, so smoke/CPU runs exercise the same
            # code path real data uses without pretending to be meaningful.
            "proprio": torch.zeros(T, C.proprio_dim),           # (T, 5)
            "instruction": "push the square to the corner",
        }
