"""Closed-loop evaluation of the action-chunking policy (an alternative to
CEMPlanner + ActionPrior, see via/decision/chunking.py), on the same
LIBERO tasks and success metric as eval/eval_libero.py.

Belief/RSSM/goal still update every real control step (they're filtering
processes that should always incorporate real incoming evidence) -- only
the *policy* re-plans every `chunk_size` steps instead of every single
step, executing its predicted chunk in between.

    python -m eval.chunking_eval --episodes 3
"""

import json

import numpy as np
import torch

from train import common
from via.belief import BeliefStateNetwork
from via.decision import ActionChunkingPolicy
from via.goal import GoalInferenceRSA
from via.world_model import RSSM


class ChunkingModel:
    """Mirrors VIAModel's belief/RSSM/goal filtering loop, but action
    selection comes from a re-planned-every-K-steps ActionChunkingPolicy
    instead of DecisionModule's CEM."""

    def __init__(self, perception, language, belief_net, rssm, goal_net, policy, device):
        self.perception = perception
        self.language = language
        self.belief_net = belief_net
        self.rssm = rssm
        self.goal_net = goal_net
        self.policy = policy
        self.device = device

    def reset(self, instruction: str):
        lang_tokens, lang_mask = self.language([instruction])
        B = 1
        from via.contracts import C

        return {
            "belief_mu": torch.zeros(B, C.belief_dim, device=self.device),
            "belief_logvar": torch.zeros(B, C.belief_dim, device=self.device),
            "belief_hidden": self.belief_net.init_hidden(B, self.device),
            "rssm": self.rssm.init_state(B, self.device),
            "last_action": torch.zeros(B, C.action_dim, device=self.device),
            "lang_tokens": lang_tokens.to(self.device),
            "lang_mask": lang_mask.to(self.device),
            "chunk": None,
            "chunk_idx": 0,
        }

    @torch.no_grad()
    def act(self, frame: torch.Tensor, proprio: torch.Tensor, state: dict):
        patches = self.perception(frame.unsqueeze(1))[:, 0]
        obs_embed = self.belief_net.encode_obs(patches, proprio)
        belief = self.belief_net.step(obs_embed, state["belief_hidden"])

        rssm_state, _ = self.rssm.posterior_step(state["rssm"], state["last_action"], obs_embed)
        goal = self.goal_net(state["lang_tokens"], state["lang_mask"], belief.mu)

        chunk = state["chunk"]
        idx = state["chunk_idx"]
        if chunk is None or idx >= self.policy.chunk_size:
            chunk = self.policy(rssm_state.feature, goal.embedding)  # (1, K, A)
            idx = 0
        action = chunk[:, idx]

        new_state = dict(state)
        new_state.update({
            "belief_mu": belief.mu, "belief_logvar": belief.logvar, "belief_hidden": belief.hidden,
            "rssm": rssm_state, "last_action": action,
            "chunk": chunk, "chunk_idx": idx + 1,
        })
        return action, new_state


def run_episode(model: ChunkingModel, runner, task_id: int, max_steps: int, device: str) -> dict:
    env, instruction = runner.make_env(task_id)
    try:
        env.reset()
        target = env.env.obj_of_interest[0]
        obs = env.env._get_observations()
        state = model.reset(instruction)
        dists = []
        for _ in range(max_steps):
            dists.append(float(np.linalg.norm(obs[f"{target}_to_robot0_eef_pos"])))
            frame = (
                torch.from_numpy(obs["agentview_image"].copy())
                .permute(2, 0, 1).float().unsqueeze(0) / 255.0
            ).to(device)
            from via.data.libero import clamp_object_rel

            proprio = torch.from_numpy(
                np.concatenate([
                    obs["robot0_eef_pos"], obs["robot0_gripper_qpos"],
                    clamp_object_rel(obs[f"{target}_to_robot0_eef_pos"]),
                ])
            ).float().unsqueeze(0).to(device)
            action, state = model.act(frame, proprio, state)
            obs, _, done, info = env.step(action[0].cpu().numpy())
            if done:
                break
        success = bool(env.env._check_success())
    finally:
        env.close()
    return {"task_id": task_id, "instruction": instruction, "success": success, "distances": dists}


def main() -> None:
    p = common.base_parser(__doc__)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--episodes", type=int, default=3, help="episodes per task")
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args()
    cfg = common.load_config(args.config)
    common.set_seed(cfg["seed"])

    device = args.device
    perception = common.build_perception(cfg).to(device).eval()
    language = common.build_language(cfg).to(device).eval()
    belief_net = BeliefStateNetwork().to(device).eval()
    rssm = RSSM().to(device).eval()
    goal_net = GoalInferenceRSA().to(device).eval()
    chunk_size = cfg.get("chunking", {}).get("chunk_size", 8)
    policy = ActionChunkingPolicy(chunk_size=chunk_size).to(device).eval()
    for name, m in [
        ("belief", belief_net), ("world_model", rssm), ("goal", goal_net),
        ("chunking_policy", policy),
    ]:
        common.load_checkpoint(m, cfg, name, device)

    model = ChunkingModel(perception, language, belief_net, rssm, goal_net, policy, device)

    from via.data.libero import LiberoEnvRunner

    runner = LiberoEnvRunner(suite=args.suite)
    results = []
    for task_id in range(runner.num_tasks()):
        successes = 0
        for ep in range(args.episodes):
            out = run_episode(model, runner, task_id, args.max_steps, device)
            successes += out["success"]
            results.append(out)
        print(f"task {task_id}: {successes}/{args.episodes} succeeded")

    out_path = common.REPO_ROOT / "results" / "chunking_eval.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    rate = sum(r["success"] for r in results) / len(results)
    mins = [min(r["distances"]) for r in results]
    ends = [r["distances"][-1] for r in results]
    print(f"\n{args.suite}: overall success rate {rate:.1%} "
          f"min={sum(mins)/len(mins):.3f} end={sum(ends)/len(ends):.3f} -> {out_path}")


if __name__ == "__main__":
    main()
