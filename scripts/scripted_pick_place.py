"""Scripted pick-and-place controller for libero_spatial, used to generate
additional synthetic demonstrations to augment the 500 real (50/task) demos
-- the official LIBERO-v1 release, confirmed to be the complete dataset
(see docs/EXPERIMENT_LOG.md 2026-08-23). Motivated directly by this whole
investigation's repeated distributional-shift finding: the learned policy
has only ever seen exact, always-successful demo trajectories, never
recovery from off-trajectory states. More demonstration coverage -- even
scripted, not human -- is the most direct way to widen that distribution
without a different policy architecture.

Uses privileged simulator state (ground-truth object/target position, the
same `obj_of_interest` mechanism used throughout this investigation's
diagnostics) to drive a simple proportional-control state machine:

    ABOVE_OBJECT -> DESCEND -> CLOSE -> LIFT -> ABOVE_TARGET -> PLACE -> RELEASE

Orientation is left at zero delta throughout -- libero_spatial's bowl-pick
tasks don't require dynamic reorientation (checked directly against a real
demo's action trace: orientation deltas stay small throughout, position
deltas dominate).

    python -m scripts.scripted_pick_place --task 0 --episodes 5 --render-check
"""

import numpy as np


class ScriptedPickPlace:
    """State machine: (obs, target_obj, target_dest) -> action (7,)."""

    ABOVE_OBJECT, DESCEND, CLOSE, LIFT, ABOVE_TARGET, PLACE, RELEASE, DONE = range(8)

    def __init__(
        self,
        above_offset: float = 0.10,
        grasp_offset: float = -0.015,
        place_offset: float = 0.12,
        pos_gain: float = 25.0,
        pos_thresh: float = 0.02,
        close_steps: int = 10,
        release_steps: int = 5,
        stall_thresh: float = 0.001,
        stall_patience: int = 40,
    ):
        self.above_offset = above_offset
        self.grasp_offset = grasp_offset
        self.place_offset = place_offset
        self.pos_gain = pos_gain
        self.pos_thresh = pos_thresh
        self.close_steps = close_steps
        self.release_steps = release_steps
        # Stall detection: contact (e.g. the gripper resting on the object's
        # physical body, since obj_pos reports its center not its rim) can
        # stop the end-effector well short of a tight position threshold
        # indefinitely. If it hasn't moved more than stall_thresh for
        # stall_patience consecutive steps, treat "as close as physically
        # possible" as good enough and advance anyway, rather than looping
        # forever on an unreachable exact target.
        self.stall_thresh = stall_thresh
        self.stall_patience = stall_patience
        self.reset()

    def reset(self) -> None:
        self.phase = self.ABOVE_OBJECT
        self._phase_step = 0
        self._last_eef = None
        self._stall_count = 0

    def _move_toward(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = (target - current) * self.pos_gain
        return np.clip(delta, -1.0, 1.0)

    def _reached_or_stalled(self, eef: np.ndarray, goal: np.ndarray, axes=slice(None)) -> bool:
        if np.linalg.norm((eef - goal)[axes]) < self.pos_thresh:
            return True
        if self._last_eef is not None and np.linalg.norm(eef - self._last_eef) < self.stall_thresh:
            self._stall_count += 1
        else:
            self._stall_count = 0
        return self._stall_count >= self.stall_patience

    def _advance_if(self, reached: bool, next_phase: int) -> None:
        if reached:
            self.phase = next_phase
            self._phase_step = 0
            self._stall_count = 0

    def act(self, obs: dict, target_obj: str, target_dest: str) -> tuple[np.ndarray, bool]:
        """-> (action (7,), phase_done: True once RELEASE has completed)."""
        eef = obs["robot0_eef_pos"]
        obj_pos = obs[f"{target_obj}_pos"]
        dest_pos = obs[f"{target_dest}_pos"]
        action = np.zeros(7, dtype=np.float32)
        gripper_open, gripper_close = -1.0, 1.0

        if self.phase == self.ABOVE_OBJECT:
            goal = obj_pos + np.array([0, 0, self.above_offset])
            action[:3] = self._move_toward(eef, goal)
            action[6] = gripper_open
            self._advance_if(self._reached_or_stalled(eef, goal), self.DESCEND)

        elif self.phase == self.DESCEND:
            goal = obj_pos + np.array([0, 0, self.grasp_offset])
            action[:3] = self._move_toward(eef, goal)
            action[6] = gripper_open
            self._advance_if(self._reached_or_stalled(eef, goal), self.CLOSE)

        elif self.phase == self.CLOSE:
            action[6] = gripper_close
            self._phase_step += 1
            if self._phase_step >= self.close_steps:
                self.phase = self.LIFT
                self._phase_step = 0
                self._stall_count = 0

        elif self.phase == self.LIFT:
            goal = obj_pos + np.array([0, 0, self.above_offset])
            action[:3] = self._move_toward(eef, goal)
            action[6] = gripper_close
            self._advance_if(
                self._reached_or_stalled(eef, goal, axes=slice(2, 3)), self.ABOVE_TARGET
            )

        elif self.phase == self.ABOVE_TARGET:
            goal = dest_pos + np.array([0, 0, self.above_offset])
            action[:3] = self._move_toward(eef, goal)
            action[6] = gripper_close
            self._advance_if(
                self._reached_or_stalled(eef, goal, axes=slice(0, 2)), self.PLACE
            )

        elif self.phase == self.PLACE:
            goal = dest_pos + np.array([0, 0, self.place_offset])
            action[:3] = self._move_toward(eef, goal)
            action[6] = gripper_close
            self._advance_if(self._reached_or_stalled(eef, goal), self.RELEASE)

        elif self.phase == self.RELEASE:
            action[6] = gripper_open
            self._phase_step += 1
            if self._phase_step >= self.release_steps:
                self.phase = self.DONE

        elif self.phase == self.DONE:
            action[6] = gripper_open

        self._last_eef = eef.copy()
        return action, self.phase == self.DONE


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args()

    from via.data.libero import LiberoEnvRunner

    runner = LiberoEnvRunner(suite="libero_spatial")
    env, instruction = runner.make_env(args.task)
    successes = 0
    for ep in range(args.episodes):
        env.reset()
        target_obj, target_dest = env.env.obj_of_interest
        obs = env.env._get_observations()
        controller = ScriptedPickPlace()
        for t in range(args.max_steps):
            action, done = controller.act(obs, target_obj, target_dest)
            obs, _, sim_done, info = env.step(action)
            if done or sim_done:
                break
        success = bool(env.env._check_success())
        successes += success
        print(f"task {args.task} ep {ep}: {'success' if success else 'FAIL'} at step {t}")
    env.close()
    print(f"\n{successes}/{args.episodes} succeeded")


if __name__ == "__main__":
    main()
