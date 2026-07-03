import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def perception():
    from via.perception import StubPerception

    return StubPerception()


@pytest.fixture(scope="session")
def language():
    from via.goal import StubLanguageEncoder

    return StubLanguageEncoder()
