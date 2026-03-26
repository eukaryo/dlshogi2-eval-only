from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .board_io import load_board
from .infer import DensePV, LegalPV, BackendSpec, dense_from_output_arrays, legal_from_dense


EXPECTED_OUTPUT_NAMES = ("policy_logits", "value_logit")


def _load_outputs_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in EXPECTED_OUTPUT_NAMES if name not in data.files]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        return {name: np.asarray(data[name]) for name in EXPECTED_OUTPUT_NAMES}


def _load_outputs_from_dir(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    candidates = [path / "outputs.npz", path / "reference_outputs.npz"]
    for candidate in candidates:
        if candidate.exists():
            return _load_outputs_npz(candidate)

    policy_path = path / "policy_logits.npy"
    value_path = path / "value_logit.npy"
    if policy_path.exists() and value_path.exists():
        return {
            "policy_logits": np.load(policy_path, allow_pickle=False),
            "value_logit": np.load(value_path, allow_pickle=False),
        }

    raise FileNotFoundError(
        "could not find openheart outputs under directory. Expected one of: "
        "outputs.npz, reference_outputs.npz, or the pair policy_logits.npy/value_logit.npy"
    )


def load_openheart_outputs(
    *,
    outputs_npz: str | Path | None = None,
    run_dir: str | Path | None = None,
    policy_logits_npy: str | Path | None = None,
    value_logit_npy: str | Path | None = None,
) -> dict[str, np.ndarray]:
    choices = [
        outputs_npz is not None,
        run_dir is not None,
        policy_logits_npy is not None or value_logit_npy is not None,
    ]
    if sum(1 for value in choices if value) != 1:
        raise ValueError(
            "specify exactly one of outputs_npz, run_dir, or the pair "
            "policy_logits_npy/value_logit_npy"
        )

    if outputs_npz is not None:
        return _load_outputs_npz(outputs_npz)
    if run_dir is not None:
        return _load_outputs_from_dir(run_dir)
    if policy_logits_npy is None or value_logit_npy is None:
        raise ValueError(
            "both policy_logits_npy and value_logit_npy are required when using separate .npy files"
        )
    return {
        "policy_logits": np.load(policy_logits_npy, allow_pickle=False),
        "value_logit": np.load(value_logit_npy, allow_pickle=False),
    }


def _extract_board_source_from_case_meta(meta_path: str | Path) -> dict[str, str | None]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    source = meta.get("source") or {}
    kind = source.get("kind")
    payload = source.get("payload")
    if kind == "usi_position":
        return {"position": payload, "sfen": None}
    if kind == "sfen":
        return {"position": None, "sfen": payload}
    raise ValueError(f"unsupported case meta source kind: {kind!r}")


def load_board_for_interpretation(
    *,
    position: str | None = None,
    sfen: str | None = None,
    case_meta: str | Path | None = None,
):
    if case_meta is not None:
        if position is not None or sfen is not None:
            raise ValueError("case_meta is mutually exclusive with position/sfen")
        source = _extract_board_source_from_case_meta(case_meta)
        return load_board(position=source["position"], sfen=source["sfen"])
    return load_board(position=position, sfen=sfen)


def interpret_openheart_outputs(
    *,
    outputs: dict[str, np.ndarray],
    position: str | None = None,
    sfen: str | None = None,
    case_meta: str | Path | None = None,
    temperature: float = 1.0,
) -> tuple[DensePV, LegalPV, BackendSpec]:
    dense = dense_from_output_arrays(outputs)
    board = load_board_for_interpretation(position=position, sfen=sfen, case_meta=case_meta)
    legal = legal_from_dense(board, dense, temperature=temperature)
    backend = BackendSpec(kind="openheart", device="external", precision="external", use_autocast=False)
    return dense, legal, backend
