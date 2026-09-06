"""Train the decision module's learned components on demonstration data.

Three parts, trained here offline from demos:

- UtilityHead: a TD(0)-bootstrapped value function V(state, goal), trained
  against LIBERO's own sparse terminal reward (1 at the true last frame of
  each demo, 0 elsewhere -- see docs/EXPERIMENT_LOG.md 2026-08-22). Replaced
  the original `progress = t/T` regression target (see below for why).

- AdaptiveGate: initialized offline to a monotone prior — lambda should rise
  with belief sigma (explore when uncertain). This is an *initialization*;
  the gate's final calibration comes from closed-loop episodes (see
  eval/ablations.py), since only interaction reveals whether information
  gathering actually paid off.

- ActionPrior: behavior-cloned from real demo actions. Added 08-14 (see
  docs/EXPERIMENT_LOG.md) after closed-loop eval showed 0% task success —
  pure random-noise CEM search never converged to purposeful manipulation
  within a real episode. Seeds CEMPlanner's search instead of starting from
  zero-mean noise. Reuses the same RSSM posterior features already computed
  for utility training, so this is nearly free to add.

UtilityHead's original `progress = t/T` target was root-caused 08-14: trained
solely on real (always successful) demo trajectories, `progress = t/T` is
almost a function of elapsed time alone, giving the utility head very little
actual state-content signal to learn — confirmed by measuring near-zero
variance in utility scores across a whole CEM population of wildly different
imagined action sequences. A counterfactual contrastive term (added 08-14,
kept below in TD-consistent form) partially compensated but didn't fix the
underlying issue. Literature at the time (DreamerV3) pointed at TD/lambda-
return bootstrapping against a real reward signal as the actual fix, deferred
then for lack of a reward model. LIBERO demos turn out to already carry
exactly the reward this needs (`rewards`, sparse, 1 at the true final frame,
0 elsewhere -- checked directly, see docs/EXPERIMENT_LOG.md), so this
replaces the static regression target with genuine TD(0): V(s_t) is trained
toward `r_t + gamma * V(s_{t+1}).detach()` using the *real* demo continuation
within each clip window (not an imagined rollout — offline training from
demos-only doesn't have a learned reward model for imagined transitions
either, so this stays on-path, same practical constraint as before, just
with a self-consistent bootstrap instead of a fixed time-based label).

    python -m train.train_decision       # requires belief, world_model, goal ckpts
    python -m train.train_decision --smoke
"""

import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import common
from via.belief import BeliefStateNetwork
from via.decision import ActionPrior, AdaptiveGate, UtilityHead
from via.goal import GoalInferenceRSA
from via.world_model import RSSM


def main() -> None:
    args = common.base_parser(__doc__).parse_args()
    cfg = common.load_config(args.config)
    common.set_seed(cfg["seed"])
    device = torch.device(args.device)

    perception = common.build_perception(cfg, args.smoke).to(device)
    language = common.build_language(cfg, args.smoke).to(device)
    belief_net = BeliefStateNetwork().to(device)
    rssm = RSSM().to(device)
    goal_net = GoalInferenceRSA().to(device)
    if not args.smoke:
        common.load_checkpoint(belief_net, cfg, "belief", args.device)
        common.load_checkpoint(rssm, cfg, "world_model", args.device)
        common.load_checkpoint(goal_net, cfg, "goal", args.device)
    for m in (belief_net, rssm, goal_net):
        m.eval()

    utility = UtilityHead().to(device)
    gate = AdaptiveGate(lambda_max=cfg["decision"]["lambda_max"]).to(device)
    action_prior = ActionPrior().to(device)
    common.try_resume(utility, cfg, "utility", args.device, args.resume)
    common.try_resume(gate, cfg, "gate", args.device, args.resume)
    common.try_resume(action_prior, cfg, "action_prior", args.device, args.resume)

    dataset = common.build_trajectory_dataset(cfg, args.smoke)
    loader = DataLoader(dataset, batch_size=2 if args.smoke else cfg["decision"]["batch_size"],
                        shuffle=True)
    opt = torch.optim.AdamW(
        list(utility.parameters()) + list(gate.parameters()) + list(action_prior.parameters()),
        lr=cfg["decision"]["lr"],
    )
    run = common.init_wandb(cfg, "decision", args.no_wandb or args.smoke)

    epochs = 1 if args.smoke else cfg["decision"]["epochs"]
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            reward = batch["reward"].to(device)
            proprio = batch["proprio"].to(device)

            with torch.no_grad():
                patches = common.encode_frames(perception, frames)
                B, T = patches.shape[:2]
                obs_embeds = torch.stack(
                    [belief_net.encode_obs(patches[:, t], proprio[:, t]) for t in range(T)], dim=1
                )
                beliefs = belief_net.rollout(patches, proprio)
                posts, _ = rssm.observe(obs_embeds, actions[:, :-1])
                tokens, mask = language(list(batch["instruction"]))
                goal = goal_net(tokens.to(device), mask.to(device), beliefs[-1].mu)

            # Utility: TD(0) value regression on posterior features at
            # t=1..T-1. r (B, T-1) is LIBERO's own sparse reward, aligned
            # with posts[0..T-2] (posts[k] = state after observing frame
            # k+1, so r[k] = reward received at frame k+1).
            feats = torch.stack([p.feature for p in posts], dim=1)          # (B, T-1, F)
            goal_rep = goal.embedding.unsqueeze(1).expand(-1, feats.shape[1], -1)
            pred = utility(feats.flatten(0, 1), goal_rep.flatten(0, 1)).view(B, -1)  # (B, T-1)
            r = reward[:, 1:]                                                # (B, T-1)

            # Bootstrap target for k=0..T-3: r[k] + gamma * V(posts[k+1]),
            # using the real (on-path) next state within this clip window.
            # r[k] is 0 for all these positions by construction (LIBERO's
            # reward is 1 only at a demo's true final frame, which -- since
            # LiberoTrajectoryDataset windows never extend past a demo's own
            # length -- can only ever land at this clip's *last* position).
            gamma = cfg["decision"].get("value_gamma", 0.95)
            with torch.no_grad():
                v_next = pred[:, 1:]
                bootstrap_target = r[:, :-1] + gamma * v_next                 # (B, T-2)
                # Last position: no posts[T-1] exists within this window to
                # bootstrap from. Valid on its own only if it's a genuine
                # terminal frame (r==1, target=1 directly, no continuation);
                # otherwise there is no well-defined target, so it's excluded
                # from the loss via valid_mask rather than guessed at.
                target = torch.cat([bootstrap_target, r[:, -1:]], dim=1)      # (B, T-1)
                valid_mask = torch.ones_like(r)
                valid_mask[:, -1] = (r[:, -1] > 0.5).float()
            u_loss = (
                F.mse_loss(pred, target, reduction="none") * valid_mask
            ).sum() / valid_mask.sum().clamp(min=1)

            # Gate init: monotone-in-sigma prior target.
            sig = torch.stack([b.sigma.mean(-1) for b in beliefs], dim=1)   # (B, T)
            target_lam = cfg["decision"]["lambda_max"] * torch.sigmoid(
                (sig - sig.mean()) / (sig.std() + 1e-6)
            )
            # Neutralized to a constant, matching via/model.py's act() --
            # see AdaptiveGate's docstring (via/decision/decision.py).
            min_sig = torch.zeros_like(sig)                                # (B, T)
            lam_pred = torch.stack(
                [gate(beliefs[t], goal.entropy(), min_sig[:, t]) for t in range(T)], dim=1
            )
            g_loss = F.mse_loss(lam_pred, target_lam)

            # Action prior: behavior-clone real actions from the same
            # posterior features utility uses. posts[t] is the state after
            # observing frame t+1 (see RSSM.observe's docstring), so it
            # pairs with actions[:, 1:] -- the action actually taken next.
            #
            # Feature noise injection (added 2026-08-23, see
            # docs/EXPERIMENT_LOG.md): trained only on exact demo states, the
            # prior has never had to predict a sensible action from a
            # slightly-off-trajectory state -- exactly the situation live
            # closed-loop rollout puts it in constantly (state estimation is
            # never bit-exact, and small approach errors compound over 300
            # steps), and this project has now found that same distribution-
            # shift failure mode three separate times through three different
            # mechanisms. Perturbing the input feature and keeping the same
            # action label teaches local corrective behavior around the
            # demonstrated trajectory instead of only the exact trajectory
            # itself -- standard practice for imitation-learning robustness
            # (DART-style state perturbation), without needing new demos.
            # Noise scale is relative to this batch's own feature std so it
            # doesn't need hand-tuning to whatever scale RSSM features
            # happen to sit at.
            noise_scale = cfg["decision"].get("bc_feature_noise", 0.15)
            feats_noisy = feats + torch.randn_like(feats) * (noise_scale * feats.std())
            action_pred = action_prior(feats_noisy.flatten(0, 1), goal_rep.flatten(0, 1)).view(
                B, feats.shape[1], -1
            )
            bc_loss = F.mse_loss(action_pred, actions[:, 1:])

            # Counterfactual utility: imagine forward from a real posterior
            # state using RANDOM actions instead of the real continuation,
            # and target the endpoint at that *starting* state's own current
            # value estimate (no advancement expected from random actions) --
            # TD-consistent counterpart of the original 08-14 version, which
            # targeted `progress[:, k_start+1]` (removed along with the rest
            # of the progress-based target, see module docstring). Contrasts
            # against u_loss above (real actions -> real reward-bootstrapped
            # value) so the utility head learns actions matter, not just
            # elapsed time.
            horizon = cfg["decision"].get("counterfactual_horizon", 5)
            k_start = random.randint(0, len(posts) - 1)
            with torch.no_grad():
                random_actions = torch.rand(B, horizon, actions.shape[-1], device=device) * 2 - 1
                cf_states = rssm.imagine(posts[k_start], random_actions)
                cf_target = pred[:, k_start].detach()
            cf_pred = utility(cf_states[-1].feature, goal.embedding)
            cf_loss = F.mse_loss(cf_pred, cf_target)

            loss = u_loss + g_loss + bc_loss + cf_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            common.log_metrics(
                run,
                {
                    "loss": loss, "utility_mse": u_loss, "gate_mse": g_loss,
                    "bc_mse": bc_loss, "cf_mse": cf_loss,
                },
                step,
            )
            # Held-out check, real held-out demos (split="val") — per-clip
            # gate response to occlusion, not just the in-batch training
            # target's aggregate MSE. Same reasoning as belief/world_model/
            # goal: an in-training loss alone isn't trustworthy validation.
            if not args.smoke and step % 250 == 0:
                for m in (utility, gate, action_prior):
                    m.eval()
                holdout = common.decision_holdout_check(
                    cfg, perception, belief_net, rssm, goal_net, language, utility, gate, device,
                    action_prior=action_prior,
                )
                for m in (utility, gate, action_prior):
                    m.train()
                common.log_metrics(run, holdout, step)
            step += 1
            if args.smoke and step >= 3:
                break
        common.save_checkpoint(utility, cfg, "utility")
        common.save_checkpoint(gate, cfg, "gate")
        common.save_checkpoint(action_prior, cfg, "action_prior")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
