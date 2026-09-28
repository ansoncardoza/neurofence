from __future__ import annotations

import pytest

from tests.behavioral.model_fixtures import build_tiny_model


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    return build_tiny_model(seed=0)
