"""Goal inference: language understanding as Bayesian inference over intent (RSA).

The instruction is treated as evidence about the speaker's goal, not a command
to embed. We implement a neural amortization of the Rational Speech Acts model
(Goodman & Frank, 2016) over K learned goal prototypes:

    L0(g | u, b)  literal listener:   softmax over goals of a learned
                  utterance-goal compatibility score, computed after language
                  tokens cross-attend to the current belief state (grounding).
    S1(u | g)     pragmatic speaker:  softmax over the *in-batch alternative
                  utterances* of alpha * log L0 - cost. During training the
                  batch provides the alternative-utterance set, which is what
                  gives RSA its contrastive, implicature-like behavior.
    L1(g | u, b)  pragmatic listener: proportional to S1(u | g) * P(g | b),
                  where P(g | b) is a context prior decoded from the belief.

At inference time alternatives are not available, so we use the standard
amortized form  L1(g|u,b) ∝ L0(g|u,b)^alpha * P(g|b)  with learned rationality
alpha; training with in-batch S1 shapes L0 so this approximation is faithful.

Vague instructions -> wide (high-entropy) goal distributions; precise ones ->
narrow. Downstream, the decision module consumes the full distribution.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from via.contracts import C, assert_shape


@dataclass
class GoalDistribution:
    """Categorical over K goal slots plus the expected goal embedding."""

    probs: torch.Tensor       # (B, K)
    log_probs: torch.Tensor   # (B, K)
    embedding: torch.Tensor   # (B, goal_embed_dim) — probability-weighted prototype mix

    def entropy(self) -> torch.Tensor:
        """(B,) — high for ambiguous instructions, low for precise ones."""
        return -(self.probs * self.log_probs).sum(-1)


class GoalInferenceRSA(nn.Module):
    def __init__(
        self,
        lang_dim: int = C.lang_dim,
        belief_dim: int = C.belief_dim,
        goal_slots: int = C.goal_slots,
        goal_embed_dim: int = C.goal_embed_dim,
        d_model: int = 256,
        n_heads: int = 4,
    ):
        super().__init__()
        self.goal_slots = goal_slots

        self.lang_proj = nn.Linear(lang_dim, d_model)
        self.belief_proj = nn.Linear(belief_dim, d_model)
        # Language tokens (queries) attend to the belief state (keys/values):
        # the same utterance grounds differently in different world beliefs.
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.utt_norm = nn.LayerNorm(d_model)

        self.goal_prototypes = nn.Parameter(torch.randn(goal_slots, goal_embed_dim) * 0.02)
        self.goal_key = nn.Linear(goal_embed_dim, d_model)

        # Context prior P(g | belief): what goals are plausible in this scene.
        self.prior_head = nn.Sequential(
            nn.Linear(belief_dim, d_model), nn.GELU(), nn.Linear(d_model, goal_slots)
        )

        # RSA speaker-rationality alpha > 0, learned (softplus-parameterized).
        self.raw_alpha = nn.Parameter(torch.tensor(0.5))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha) + 1e-4

    # ---- listeners ----

    def _utterance_embedding(
        self, lang_tokens: torch.Tensor, lang_mask: torch.Tensor, belief_mu: torch.Tensor
    ) -> torch.Tensor:
        """Ground the utterance in the belief: (B, L, lang_dim) -> (B, d_model)."""
        q = self.lang_proj(lang_tokens)                       # (B, L, D)
        kv = self.belief_proj(belief_mu).unsqueeze(1)         # (B, 1, D)
        grounded, _ = self.cross_attn(q, kv, kv)              # (B, L, D)
        grounded = self.utt_norm(grounded + q)
        # Masked mean pool over real tokens.
        m = lang_mask.unsqueeze(-1).float()
        return (grounded * m).sum(1) / m.sum(1).clamp(min=1.0)

    def utterance_embedding(
        self, lang_tokens: torch.Tensor, lang_mask: torch.Tensor, belief_mu: torch.Tensor
    ) -> torch.Tensor:
        """Public accessor for the grounded utterance embedding (B, d_model).

        Used by eval/integration_probe.py to compare how an instruction is
        encoded here against how much it moves the goal posterior. See
        docs/embodied_comprehension_bridge.md, Q3.
        """
        return self._utterance_embedding(lang_tokens, lang_mask, belief_mu)

    def literal_logits(
        self, lang_tokens: torch.Tensor, lang_mask: torch.Tensor, belief_mu: torch.Tensor
    ) -> torch.Tensor:
        """L0 compatibility logits over goals: (B, K)."""
        assert_shape(lang_tokens, (-1, -1, C.lang_dim), "lang_tokens")
        u = self._utterance_embedding(lang_tokens, lang_mask, belief_mu)  # (B, D)
        keys = self.goal_key(self.goal_prototypes)                        # (K, D)
        return u @ keys.T / (keys.shape[-1] ** 0.5)

    def forward(
        self, lang_tokens: torch.Tensor, lang_mask: torch.Tensor, belief_mu: torch.Tensor
    ) -> GoalDistribution:
        """Amortized pragmatic listener L1(g | u, b)."""
        l0_logits = self.literal_logits(lang_tokens, lang_mask, belief_mu)
        prior_logits = self.prior_head(belief_mu)
        l1_logits = self.alpha * F.log_softmax(l0_logits, dim=-1) + F.log_softmax(
            prior_logits, dim=-1
        )
        log_probs = F.log_softmax(l1_logits, dim=-1)
        probs = log_probs.exp()
        embedding = probs @ self.goal_prototypes
        return GoalDistribution(probs=probs, log_probs=log_probs, embedding=embedding)

    # ---- training ----

    def rsa_training_loss(
        self,
        lang_tokens: torch.Tensor,
        lang_mask: torch.Tensor,
        belief_mu: torch.Tensor,
        goal_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Contrastive RSA objective over the in-batch alternative utterance set.

        Builds the full B×K literal-listener matrix, derives the pragmatic
        speaker S1 by normalizing over the batch's utterances, and trains the
        pragmatic listener L1 ∝ S1 * prior with cross-entropy on goal labels.
        The speaker normalization is what pushes L0 toward *discriminative*
        utterance-goal mappings (RSA's informativity pressure).

        goal_targets: (B,) int64 goal-slot labels (from synthetic pretraining
        or LIBERO task annotations mapped to slots).
        """
        l0_logits = self.literal_logits(lang_tokens, lang_mask, belief_mu)   # (B, K)
        log_l0 = F.log_softmax(l0_logits, dim=-1)

        # S1(u | g): normalize alpha * log L0 over the utterance axis (batch).
        log_s1 = F.log_softmax(self.alpha * log_l0, dim=0)                   # (B, K)

        log_prior = F.log_softmax(self.prior_head(belief_mu), dim=-1)        # (B, K)
        log_l1 = F.log_softmax(log_s1 + log_prior, dim=-1)                   # (B, K)

        loss = F.nll_loss(log_l1, goal_targets)
        with torch.no_grad():
            acc = (log_l1.argmax(-1) == goal_targets).float().mean()
            ent = -(log_l1.exp() * log_l1).sum(-1).mean()
        return {"loss": loss, "accuracy": acc, "mean_entropy": ent, "alpha": self.alpha.detach()}
