"""Frozen language encoder: instruction text -> token hidden states.

Contract: list[str] of length B -> (tokens (B, L, 3072), mask (B, L)).

Phi-3-mini-4k-instruct is used purely as a frozen text encoder (last hidden
states); no generation. `StubLanguageEncoder` hashes words into a fixed
embedding table so goal-inference tests run offline with the same contract —
crucially, the same word always maps to the same vector, so 'pick up the cup'
vs 'pick up something' differ in a stable, learnable way.
"""

import torch
import torch.nn as nn

from via.contracts import C

PHI3_MODEL = "microsoft/Phi-3-mini-4k-instruct"


class Phi3LanguageEncoder(nn.Module):
    """Frozen Phi-3-mini. Requires `transformers` and torch>=2.4 (GPU box)."""

    def __init__(
        self,
        model_name: str = PHI3_MODEL,
        max_length: int = 64,
        dtype: torch.dtype = torch.float16,
    ):
        # fp16 by default: the encoder is frozen (inference only) and fp16
        # halves memory to ~7.6 GB, fitting the 11 GB GPUs on UCSD DSMLP.
        # Hidden states are cast back to fp32 before returning.
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.encoder.parameters()).device
        batch = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)
        hidden = self.encoder(**batch).last_hidden_state  # (B, L, 3072)
        return hidden.float(), batch["attention_mask"].bool()


class StubLanguageEncoder(nn.Module):
    """Offline stand-in: deterministic hashed word embeddings, same contract."""

    def __init__(self, vocab_buckets: int = 4096, max_length: int = 16, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        table = torch.randn(vocab_buckets, C.lang_dim, generator=gen) * 0.02
        self.register_buffer("table", table)
        self.vocab_buckets = vocab_buckets
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(instructions)
        L = self.max_length
        tokens = torch.zeros(B, L, C.lang_dim)
        mask = torch.zeros(B, L, dtype=torch.bool)
        for i, text in enumerate(instructions):
            words = text.lower().split()[:L]
            for j, w in enumerate(words):
                # Python's str hash is salted per-process; use a stable hash.
                idx = sum((k + 1) * ord(ch) for k, ch in enumerate(w)) % self.vocab_buckets
                tokens[i, j] = self.table[idx]
                mask[i, j] = True
            if not words:
                mask[i, 0] = True
        return tokens.to(self.table.device), mask.to(self.table.device)
