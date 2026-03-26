from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .model import PolicyValueNetwork


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_jsonable(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(np.asarray(tensor).tobytes())
    return h.hexdigest()


def sha256_named_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays.keys()):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        h.update(name.encode("utf-8"))
        h.update(np.dtype(array.dtype).str.encode("utf-8"))
        h.update(json.dumps(list(array.shape)).encode("utf-8"))
        h.update(array.tobytes())
    return h.hexdigest()


def graph_sha256(exported_program: torch.export.ExportedProgram) -> str:
    graph_text = str(exported_program.graph)
    return sha256_bytes(graph_text.encode("utf-8"))


def example_input_sha256(example_input: torch.Tensor) -> str:
    arr = example_input.detach().cpu().contiguous().numpy()
    return sha256_bytes(arr.tobytes())


def dtype_to_manifest_string(dtype: np.dtype[Any] | torch.dtype | str | type[Any]) -> str:
    if isinstance(dtype, torch.dtype):
        text = str(dtype)
        if text.startswith("torch."):
            return text[len("torch.") :]
        return text
    return np.dtype(dtype).name


def build_reference_manifest(
    *,
    model: PolicyValueNetwork,
    checkpoint_path: str | Path,
    exported_program: torch.export.ExportedProgram,
    example_input: torch.Tensor,
    example_position: str | None,
    example_sfen: str | None,
    upstream_commit: str | None,
) -> dict[str, Any]:
    return {
        "model_class": model.__class__.__name__,
        "model_config": {
            "blocks": model.blocks_count,
            "channels": model.channels,
            "fcl": model.fcl,
        },
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "state_dict_sha256": sha256_state_dict(model.state_dict()),
        "graph_sha256": graph_sha256(exported_program),
        "example_input_sha256": example_input_sha256(example_input),
        "example_position": example_position,
        "example_sfen": example_sfen,
        "upstream_commit": upstream_commit,
    }


def dump_manifest_json(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
