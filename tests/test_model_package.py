from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dlshogi2_eval.model_package import (
    build_model_package_id,
    extract_torch_export_bindings,
    validate_model_package_dir,
)


class TinyNet(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x + 1.0
        z = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return y, z


def test_extract_torch_export_bindings_uses_user_io_only() -> None:
    model = TinyNet().eval()
    x = torch.randn(1, 1, 2, 2)
    ep = torch.export.export(model, (x,), strict=False)

    bindings = extract_torch_export_bindings(ep)
    assert bindings["inputs"] == [
        {
            "kind": "torch_export_user_input",
            "index": 0,
            "graph_name": "x",
        }
    ]
    assert bindings["outputs"][0]["kind"] == "torch_export_user_output"
    assert bindings["outputs"][0]["index"] == 0
    assert bindings["outputs"][1]["index"] == 1


def test_build_model_package_id_is_stable_for_same_payload() -> None:
    payload = {
        "package_type": "dlshogi_static_eval_model_package_v1",
        "contract_version": 1,
        "producer": {"name": "x", "version": None, "git_commit": None},
        "graph_artifact": {"kind": "pt2", "path": "graph/model.pt2"},
        "weights_artifact": {"manifest_path": "weights/manifest.json"},
        "inputs": [{"name": "features", "dtype": "float32", "shape": [1, 1, 2, 2], "layout": "logical_nchw"}],
        "outputs": [{"name": "policy_logits", "dtype": "float32", "shape": [1, 4], "layout": "logical_flat"}],
        "case_ids": ["case_000001"],
    }
    assert build_model_package_id(payload) == build_model_package_id(dict(payload))


def _write_minimal_valid_package(tmp_path: Path) -> Path:
    package_dir = tmp_path / "pkg"
    (package_dir / "graph").mkdir(parents=True)
    (package_dir / "weights").mkdir(parents=True)
    case_dir = package_dir / "cases" / "case_000001"
    case_dir.mkdir(parents=True)

    (package_dir / "graph" / "model.pt2").write_bytes(b"pt2")
    (package_dir / "weights" / "weights_000.safetensors").write_bytes(b"safe")

    weights_manifest = {
        "package_type": "dlshogi_static_eval_weights_manifest_v1",
        "weight_entries": [
            {
                "name": "linear.weight",
                "dtype": "float32",
                "shape": [1, 1],
                "file": "weights_000.safetensors",
                "key": "linear.weight",
            }
        ],
    }
    (package_dir / "weights" / "manifest.json").write_text(
        json.dumps(weights_manifest, indent=2),
        encoding="utf-8",
    )

    np.savez_compressed(
        case_dir / "inputs.npz",
        features=np.zeros((1, 1, 2, 2), dtype=np.float32),
    )
    np.savez_compressed(
        case_dir / "reference_outputs.npz",
        policy_logits=np.zeros((1, 4), dtype=np.float32),
        value_logit=np.zeros((1, 1), dtype=np.float32),
    )
    meta = {
        "case_id": "case_000001",
        "case_label": "synthetic",
        "source": {"kind": "synthetic", "payload": "zeros"},
        "reference_available": True,
    }
    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    manifest = {
        "package_type": "dlshogi_static_eval_model_package_v1",
        "contract_version": 1,
        "package_id": "dlshogi2-eval-test",
        "producer": {"name": "test", "version": None, "git_commit": None},
        "graph_artifact": {"kind": "pt2", "path": "graph/model.pt2"},
        "weights_artifact": {"manifest_path": "weights/manifest.json"},
        "inputs": [
            {
                "name": "features",
                "dtype": "float32",
                "shape": [1, 1, 2, 2],
                "layout": "logical_nchw",
                "binding": {"kind": "torch_export_user_input", "index": 0, "graph_name": "x"},
            }
        ],
        "outputs": [
            {
                "name": "policy_logits",
                "dtype": "float32",
                "shape": [1, 4],
                "layout": "logical_flat",
                "binding": {"kind": "torch_export_user_output", "index": 0, "graph_name": "add"},
            },
            {
                "name": "value_logit",
                "dtype": "float32",
                "shape": [1, 1],
                "layout": "logical_flat",
                "binding": {"kind": "torch_export_user_output", "index": 1, "graph_name": "unsqueeze"},
            },
        ],
        "case_ids": ["case_000001"],
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return package_dir


def test_validate_model_package_dir_accepts_minimal_valid_package(tmp_path: Path) -> None:
    package_dir = _write_minimal_valid_package(tmp_path)
    assert validate_model_package_dir(package_dir) == []


def test_validate_model_package_dir_reports_tensor_name_mismatch(tmp_path: Path) -> None:
    package_dir = _write_minimal_valid_package(tmp_path)
    case_dir = package_dir / "cases" / "case_000001"
    np.savez_compressed(
        case_dir / "reference_outputs.npz",
        wrong_name=np.zeros((1, 4), dtype=np.float32),
    )
    errors = validate_model_package_dir(package_dir)
    assert any("names mismatch" in error for error in errors)
