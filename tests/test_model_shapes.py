from __future__ import annotations

import torch

from dlshogi2_eval.features import FEATURES_NUM, MOVE_LABELS_NUM
from dlshogi2_eval.model import PolicyValueNetwork



def test_model_output_shapes() -> None:
    model = PolicyValueNetwork()
    x = torch.randn(1, FEATURES_NUM, 9, 9)
    policy, value = model(x)
    assert tuple(policy.shape) == (1, MOVE_LABELS_NUM)
    assert tuple(value.shape) == (1, 1)
