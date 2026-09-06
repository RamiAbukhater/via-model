"""Action-chunking policy: an alternative to CEMPlanner + ActionPrior,
predicting a short SEQUENCE of future actions at once from the current
state, rather than one action re-planned from scratch every control step.

Motivated 2026-08-27 (see docs/EXPERIMENT_LOG.md): scaling training data
3x (500->1500 demos) produced this investigation's first repeated nonzero
closed-loop success, but only ~1.7%, and the improvement is statistically
thin at that sample size. CEMPlanner+ActionPrior's single-step re-planning
is exactly the architecture class the imitation-learning literature
identifies as most prone to compounding closed-loop error, independent of
how much data it's trained on (see Zhao et al. 2023, "Action Chunking with
Transformers" / ACT) -- more data widens the state distribution it's seen,
but doesn't change that it re-decides from scratch every single step with
no committed short-term intent. Predicting a chunk and executing several
steps before re-planning directly targets that failure mode instead.

Deliberately simplified relative to the ACT paper (no CVAE / style latent,
no full transformer encoder over a long token history) to keep this
tractable to implement and train from the same ~500-1500 demo budget this
project has: K learned query embeddings cross-attend to a single context
vector (current RSSM feature + goal embedding, the same input ActionPrior
already uses) via a small TransformerDecoder, each query position
projected to one action. Self-attention among the K queries is what lets
the chunk be mutually consistent (e.g. a smooth descent-then-close motion)
rather than K independent single-step predictions -- the actual mechanism
expected to help, without the added machinery a full ACT reimplementation
would cost.
"""

import torch
import torch.nn as nn

from via.contracts import C


class ActionChunkingPolicy(nn.Module):
    def __init__(
        self,
        feature_dim: int = C.deter_dim + C.stoch_dim,
        goal_dim: int = C.goal_embed_dim,
        action_dim: int = C.action_dim,
        chunk_size: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.context_proj = nn.Sequential(
            nn.Linear(feature_dim + goal_dim, d_model), nn.GELU()
        )
        self.query_embed = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, feature: torch.Tensor, goal_embed: torch.Tensor) -> torch.Tensor:
        """(B, F), (B, G) -> (B, chunk_size, A) actions in [-1, 1]."""
        B = feature.shape[0]
        context = self.context_proj(torch.cat([feature, goal_embed], dim=-1)).unsqueeze(1)  # (B,1,d)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)
        out = self.decoder(tgt=queries, memory=context)  # (B, K, d)
        return torch.tanh(self.action_head(out))  # (B, K, A)
