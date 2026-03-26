from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from dlshogi2_eval.manifest import sha256_jsonable, sha256_named_arrays, sha256_state_dict
from dlshogi2_eval.model import PolicyValueNetwork


def test_state_dict_hash_is_stable_for_same_weights() -> None:
    model = PolicyValueNetwork()
    h1 = sha256_state_dict(model.state_dict())
    h2 = sha256_state_dict(model.state_dict())
    assert h1 == h2


def test_json_hash_ignores_dict_insertion_order() -> None:
    a = OrderedDict([("b", 2), ("a", 1)])
    b = OrderedDict([("a", 1), ("b", 2)])
    assert sha256_jsonable(a) == sha256_jsonable(b)


def test_named_array_hash_is_stable_for_same_arrays() -> None:
    arrays = {
        "features": np.zeros((1, 104, 9, 9), dtype=np.float32),
        "value_logit": np.zeros((1, 1), dtype=np.float32),
    }
    assert sha256_named_arrays(arrays) == sha256_named_arrays(arrays)
