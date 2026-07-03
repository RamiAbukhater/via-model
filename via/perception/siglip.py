"""Perception module: frozen SigLIP ViT-B/16 patch embeddings.

Contract: images (B, T, 3, 224, 224) in [0, 1] -> patches (B, T, 196, 768).

The encoder is fully frozen — perception provides *evidence*; all inference
happens downstream in the belief state network. `StubPerception` implements
the identical contract with a fixed random projection so the rest of the
system can be developed and unit-tested without downloading weights.
"""

import torch
import torch.nn as nn

from via.contracts import C, assert_shape

SIGLIP_MODEL = "google/siglip-base-patch16-224"

# SigLIP preprocessing normalizes to [-1, 1]
_SIGLIP_MEAN = 0.5
_SIGLIP_STD = 0.5


class SigLIPPerception(nn.Module):
    """Frozen SigLIP ViT-B/16. Requires `transformers` and torch>=2.4 (GPU box)."""

    def __init__(self, model_name: str = SIGLIP_MODEL):
        super().__init__()
        from transformers import SiglipVisionModel

        self.encoder = SiglipVisionModel.from_pretrained(model_name)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images (B, T, 3, 224, 224) in [0,1] -> patches (B, T, 196, 768)."""
        assert_shape(images, (-1, -1, 3, C.image_size, C.image_size), "images")
        B, T = images.shape[:2]
        flat = images.flatten(0, 1)
        flat = (flat - _SIGLIP_MEAN) / _SIGLIP_STD
        out = self.encoder(pixel_values=flat).last_hidden_state  # (B*T, 196, 768)
        return out.view(B, T, C.patch_count, C.patch_dim)


class StubPerception(nn.Module):
    """Contract-identical stand-in: fixed random conv patchifier, no downloads.

    Deterministic (seeded) and frozen, so belief-state unit tests see a stable
    'encoder' whose output actually depends on pixel content (occlusion tests
    need blanked pixels to change the embedding).
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        proj = nn.Conv2d(3, C.patch_dim, kernel_size=16, stride=16)
        with torch.no_grad():
            proj.weight.copy_(torch.randn(proj.weight.shape, generator=gen) * 0.02)
            proj.bias.zero_()
        for p in proj.parameters():
            p.requires_grad_(False)
        self.proj = proj

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert_shape(images, (-1, -1, 3, C.image_size, C.image_size), "images")
        B, T = images.shape[:2]
        flat = images.flatten(0, 1)
        feat = self.proj(flat)                       # (B*T, 768, 14, 14)
        feat = feat.flatten(2).transpose(1, 2)       # (B*T, 196, 768)
        return feat.view(B, T, C.patch_count, C.patch_dim)
