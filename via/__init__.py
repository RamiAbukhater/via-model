"""VIA: Vision-Inference-Action.

A cognitively grounded alternative to standard VLA models. Four modules,
each implementing a principle from Bayesian cognitive science:

- perception: frozen SigLIP ViT-B/16 patch embeddings (perception as evidence)
- belief:     recurrent Gaussian belief state (perception as inference)
- goal:       RSA goal inference from language (language as evidence)
- world_model: RSSM for mental simulation (planning as imagination)
- decision:   expected utility + epistemic value (action as decision)
"""

from via.contracts import Contracts
from via.model import VIAModel

__version__ = "0.1.0"
__all__ = ["Contracts", "VIAModel"]
