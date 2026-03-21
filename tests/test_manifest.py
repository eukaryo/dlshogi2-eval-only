from __future__ import annotations

import torch

from dlshogi2_eval.manifest import sha256_state_dict
from dlshogi2_eval.model import PolicyValueNetwork



def test_state_dict_hash_is_stable_for_same_weights() -> None:
    model = PolicyValueNetwork()
    h1 = sha256_state_dict(model.state_dict())
    h2 = sha256_state_dict(model.state_dict())
    assert h1 == h2
