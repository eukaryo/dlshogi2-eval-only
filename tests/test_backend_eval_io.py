from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dlshogi2_eval.infer import (
    BackendSpec,
    DensePV,
    dense_from_output_arrays,
    make_backend_payload,
    precision_name_to_torch_dtype,
    raw_output_arrays_from_dense,
)
from dlshogi2_eval.openheart_interpret import load_openheart_outputs


def test_precision_name_to_torch_dtype() -> None:
    assert precision_name_to_torch_dtype("fp32") == torch.float32
    assert precision_name_to_torch_dtype("fp16") == torch.float16
    assert precision_name_to_torch_dtype("bf16") == torch.bfloat16


def test_dense_raw_roundtrip_shapes() -> None:
    dense = DensePV(
        policy_logits=torch.arange(2187, dtype=torch.float32),
        value_logit=torch.tensor(0.25, dtype=torch.float32),
    )
    arrays = raw_output_arrays_from_dense(dense)
    assert arrays["policy_logits"].shape == (1, 2187)
    assert arrays["value_logit"].shape == (1, 1)

    reconstructed = dense_from_output_arrays(arrays)
    assert reconstructed.policy_logits.shape == (2187,)
    assert float(reconstructed.value_logit.item()) == 0.25


def test_make_backend_payload_contains_backend_block() -> None:
    dense = DensePV(
        policy_logits=torch.zeros(2187, dtype=torch.float32),
        value_logit=torch.tensor(0.0, dtype=torch.float32),
    )
    empty_legal = {
        "moves_usi": [],
        "move_labels": torch.empty((0,), dtype=torch.int64),
        "logits": torch.empty((0,), dtype=torch.float32),
        "probs": torch.empty((0,), dtype=torch.float32),
        "value_logit": torch.tensor(0.0, dtype=torch.float32),
    }
    from dlshogi2_eval.infer import LegalPV

    legal = LegalPV(**empty_legal)
    payload = make_backend_payload(
        backend=BackendSpec(kind="pytorch", device="cuda:0", precision="bf16", use_autocast=True),
        dense=dense,
        legal=legal,
        topk=5,
    )
    assert payload["backend"]["kind"] == "pytorch"
    assert payload["backend"]["precision"] == "bf16"
    assert payload["dense"]["policy_size"] == 2187
    assert payload["legal"]["num_legal_moves"] == 0


def test_load_openheart_outputs_from_npz_and_dir(tmp_path: Path) -> None:
    arrays = {
        "policy_logits": np.arange(2187, dtype=np.float32).reshape(1, 2187),
        "value_logit": np.array([[0.5]], dtype=np.float32),
    }
    npz_path = tmp_path / "outputs.npz"
    np.savez(npz_path, **arrays)

    loaded_a = load_openheart_outputs(outputs_npz=npz_path)
    assert loaded_a["policy_logits"].shape == (1, 2187)
    assert loaded_a["value_logit"].shape == (1, 1)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.save(run_dir / "policy_logits.npy", arrays["policy_logits"])
    np.save(run_dir / "value_logit.npy", arrays["value_logit"])
    loaded_b = load_openheart_outputs(run_dir=run_dir)
    assert np.allclose(loaded_b["policy_logits"], arrays["policy_logits"])
    assert np.allclose(loaded_b["value_logit"], arrays["value_logit"])
