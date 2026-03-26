from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .board_io import load_board
from .export import export_reference_program, save_exported_program, write_export_text_dumps
from .features import FEATURES_NUM
from .infer import DensePV, LegalPV, PolicyValueEvaluator
from .manifest import (
    dtype_to_manifest_string,
    dump_manifest_json,
    graph_sha256,
    sha256_file,
    sha256_jsonable,
    sha256_named_arrays,
    sha256_state_dict,
)
from .model import PolicyValueNetwork

try:  # pragma: no cover - availability depends on runtime environment
    from safetensors.torch import save_file as save_safetensors_file
except ImportError:  # pragma: no cover - exercised only when dependency missing
    save_safetensors_file = None

PACKAGE_TYPE = "dlshogi_static_eval_model_package_v1"
WEIGHTS_MANIFEST_TYPE = "dlshogi_static_eval_weights_manifest_v1"
CONTRACT_VERSION = 1
DEFAULT_PRODUCER_NAME = "dlshogi2-eval-only"
DEFAULT_PACKAGE_ID_PREFIX = "dlshogi2-eval"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    raw_line: str
    line_number: int
    position: str | None
    sfen: str | None
    source_kind: str
    source_payload: str


@dataclass(frozen=True)
class CaseArtifact:
    case_id: str
    inputs_path: str
    meta_path: str
    reference_outputs_path: str | None
    readable_path: str | None
    inputs_sha256: str
    reference_outputs_sha256: str | None


@dataclass(frozen=True)
class ModelPackageArtifacts:
    package_dir: str
    manifest_path: str
    package_id: str
    graph_path: str
    weights_manifest_path: str
    case_artifacts: list[CaseArtifact]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_producer_git_commit(repo_root: str | Path | None = None) -> str | None:
    repo_root = Path(repo_root) if repo_root is not None else _default_repo_root()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def default_producer_version() -> str | None:
    try:
        return importlib_metadata.version(DEFAULT_PRODUCER_NAME)
    except importlib_metadata.PackageNotFoundError:
        return None


def parse_positions_file(positions_file: str | Path) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    next_index = 1
    with open(positions_file, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            position: str | None = None
            sfen: str | None = None
            source_kind: str
            if line.startswith("sfen "):
                sfen = line
                source_kind = "sfen"
            else:
                position = line
                source_kind = "usi_position"

            cases.append(
                CaseSpec(
                    case_id=f"case_{next_index:06d}",
                    raw_line=line,
                    line_number=line_number,
                    position=position,
                    sfen=sfen,
                    source_kind=source_kind,
                    source_payload=line,
                )
            )
            next_index += 1

    if not cases:
        raise ValueError(f"positions file contains no usable cases: {positions_file}")
    return cases


def make_zero_example_input(
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    return torch.zeros((1, FEATURES_NUM, 9, 9), dtype=torch.float32, device=device)


def _spec_kind_name(spec: Any) -> str | None:
    kind = getattr(spec, "kind", None)
    return getattr(kind, "name", None) or str(kind) if kind is not None else None


def extract_torch_export_bindings(exported_program: torch.export.ExportedProgram) -> dict[str, list[dict[str, Any]]]:
    """Return USER_INPUT / USER_OUTPUT binding metadata from ``torch.export``.

    The returned indices are zero-based among *user* inputs/outputs only. Parameters
    and buffers that appear in ``graph_signature.input_specs`` are intentionally
    excluded because the contract extension is meant to describe the external socket,
    not the lifted state.
    """
    graph_signature = getattr(exported_program, "graph_signature", None)
    if graph_signature is None:
        return {"inputs": [], "outputs": []}

    input_bindings: list[dict[str, Any]] = []
    output_bindings: list[dict[str, Any]] = []

    for spec in getattr(graph_signature, "input_specs", []):
        if _spec_kind_name(spec) != "USER_INPUT":
            continue
        arg = getattr(spec, "arg", None)
        graph_name = getattr(arg, "name", None)
        binding: dict[str, Any] = {
            "kind": "torch_export_user_input",
            "index": len(input_bindings),
        }
        if graph_name is not None:
            binding["graph_name"] = graph_name
        input_bindings.append(binding)

    for spec in getattr(graph_signature, "output_specs", []):
        if _spec_kind_name(spec) != "USER_OUTPUT":
            continue
        arg = getattr(spec, "arg", None)
        graph_name = getattr(arg, "name", None)
        binding = {
            "kind": "torch_export_user_output",
            "index": len(output_bindings),
        }
        if graph_name is not None:
            binding["graph_name"] = graph_name
        output_bindings.append(binding)

    return {"inputs": input_bindings, "outputs": output_bindings}


def _state_dict_cpu_contiguous(model: PolicyValueNetwork) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }


def write_safetensors_weights(
    state_dict: dict[str, torch.Tensor],
    out_path: str | Path,
) -> None:
    if save_safetensors_file is None:
        raise ImportError(
            "safetensors is required for model package export. "
            "Install it with `pip install safetensors`."
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors_file(state_dict, str(out_path))


def build_weights_manifest(
    state_dict: dict[str, torch.Tensor],
    *,
    weights_file_name: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        entries.append(
            {
                "name": name,
                "dtype": dtype_to_manifest_string(tensor.dtype),
                "shape": list(tensor.shape),
                "file": weights_file_name,
                "key": name,
            }
        )
    return {
        "package_type": WEIGHTS_MANIFEST_TYPE,
        "weight_entries": entries,
    }


def build_model_package_id(
    manifest_payload_without_id: dict[str, Any],
    *,
    prefix: str = DEFAULT_PACKAGE_ID_PREFIX,
) -> str:
    return f"{prefix}-{sha256_jsonable(manifest_payload_without_id)[:24]}"


def _prepare_output_dir(path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif any(path.iterdir()):
            raise FileExistsError(
                f"output directory already exists and is not empty: {path}. "
                "Pass overwrite=True to replace it."
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _forward_raw_outputs(
    evaluator: PolicyValueEvaluator,
    *,
    position: str | None,
    sfen: str | None,
) -> tuple[torch.Tensor, dict[str, np.ndarray], DensePV, LegalPV]:
    board = load_board(position=position, sfen=sfen)
    x = evaluator.encode_board(board)
    with torch.inference_mode():
        policy_logits, value_logit = evaluator.model(x)

    policy_logits_cpu = policy_logits.detach().cpu().contiguous()
    value_logit_cpu = value_logit.detach().cpu().contiguous()
    dense = DensePV(
        policy_logits=policy_logits_cpu.reshape(-1),
        value_logit=value_logit_cpu.reshape(-1)[0],
    )
    legal = evaluator.predict_legal(board, dense=dense)
    outputs = {
        "policy_logits": np.asarray(policy_logits_cpu.numpy(), dtype=np.float32),
        "value_logit": np.asarray(value_logit_cpu.numpy(), dtype=np.float32),
    }
    return x.detach().cpu().contiguous(), outputs, dense, legal


def _readable_case_text(
    case: CaseSpec,
    *,
    dense: DensePV,
    legal: LegalPV,
    raw_outputs: dict[str, np.ndarray],
    topk: int = 10,
) -> str:
    lines = [
        f"case_id: {case.case_id}",
        f"source_kind: {case.source_kind}",
        f"source_payload: {case.source_payload}",
        f"value_logit: {float(dense.value_logit.item()):.9g}",
        f"value_prob: {float(dense.value_prob.item()):.9g}",
        f"policy_logits_shape: {list(raw_outputs['policy_logits'].shape)}",
        f"value_logit_shape: {list(raw_outputs['value_logit'].shape)}",
        "top_legal_moves:",
    ]
    top = legal.topk(topk)
    for rank, (move, prob, logit) in enumerate(
        zip(top["moves"], top["probs"], top["logits"]),
        start=1,
    ):
        lines.append(
            f"  {rank:02d}. {move}  prob={float(prob):.9g}  logit={float(logit):.9g}"
        )
    if len(top["moves"]) == 0:
        lines.append("  (no legal moves)")
    return "\n".join(lines) + "\n"


def _build_case_meta(
    case: CaseSpec,
    *,
    input_array: np.ndarray,
    output_arrays: dict[str, np.ndarray] | None,
    inputs_sha256: str,
    reference_outputs_sha256: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "case_id": case.case_id,
        "case_label": case.raw_line,
        "source": {
            "kind": case.source_kind,
            "payload": case.source_payload,
        },
        "reference_available": output_arrays is not None,
        "reference_interpretation": {
            "policy_space": "dense_policy_logits",
            "value_space": "raw_value_logit",
        },
        "source_line_number": case.line_number,
        "hashes": {
            "inputs_sha256": inputs_sha256,
        },
        "input_tensors": {
            "features": {
                "dtype": dtype_to_manifest_string(input_array.dtype),
                "shape": list(input_array.shape),
            }
        },
    }
    if output_arrays is not None:
        payload["output_tensors"] = {
            name: {
                "dtype": dtype_to_manifest_string(array.dtype),
                "shape": list(array.shape),
            }
            for name, array in output_arrays.items()
        }
        payload["hashes"]["reference_outputs_sha256"] = reference_outputs_sha256
    return payload


def _maybe_binding(bindings: Sequence[dict[str, Any]], index: int) -> dict[str, Any] | None:
    if index < len(bindings):
        return dict(bindings[index])
    return None


def _build_input_entries(
    example_input: torch.Tensor,
    *,
    bindings: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    entry: dict[str, Any] = {
        "name": "features",
        "dtype": dtype_to_manifest_string(example_input.dtype),
        "shape": list(example_input.shape),
        "layout": "logical_nchw",
    }
    binding = _maybe_binding(bindings, 0)
    if binding is not None:
        entry["binding"] = binding
    return [entry]


def _build_output_entries(
    output_arrays: dict[str, np.ndarray],
    *,
    bindings: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, (name, array) in enumerate(output_arrays.items()):
        entry: dict[str, Any] = {
            "name": name,
            "dtype": dtype_to_manifest_string(array.dtype),
            "shape": list(array.shape),
            "layout": "logical_flat",
        }
        binding = _maybe_binding(bindings, index)
        if binding is not None:
            entry["binding"] = binding
        entries.append(entry)
    return entries


def validate_model_package_dir(package_dir: str | Path) -> list[str]:
    errors: list[str] = []
    package_dir = Path(package_dir)
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"missing manifest: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"failed to parse manifest.json: {exc}"]

    if manifest.get("package_type") != PACKAGE_TYPE:
        errors.append(
            f"manifest.json package_type must be {PACKAGE_TYPE!r}, got {manifest.get('package_type')!r}"
        )
    if manifest.get("contract_version") != CONTRACT_VERSION:
        errors.append(
            f"manifest.json contract_version must be {CONTRACT_VERSION}, got {manifest.get('contract_version')!r}"
        )

    graph_path_value = manifest.get("graph_artifact", {}).get("path")
    if not graph_path_value:
        errors.append("manifest.json graph_artifact.path is required")
    else:
        graph_path = package_dir / graph_path_value
        if not graph_path.exists():
            errors.append(f"missing graph artifact: {graph_path_value}")

    weights_manifest_rel = manifest.get("weights_artifact", {}).get("manifest_path")
    if not weights_manifest_rel:
        errors.append("manifest.json weights_artifact.manifest_path is required")
        weights_manifest = None
    else:
        weights_manifest_path = package_dir / weights_manifest_rel
        if not weights_manifest_path.exists():
            errors.append(f"missing weights manifest: {weights_manifest_rel}")
            weights_manifest = None
        else:
            try:
                weights_manifest = json.loads(weights_manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"failed to parse weights manifest {weights_manifest_rel}: {exc}")
                weights_manifest = None

    if weights_manifest is not None:
        if weights_manifest.get("package_type") != WEIGHTS_MANIFEST_TYPE:
            errors.append(
                "weights/manifest.json package_type must be "
                f"{WEIGHTS_MANIFEST_TYPE!r}, got {weights_manifest.get('package_type')!r}"
            )
        names_seen: set[str] = set()
        for entry in weights_manifest.get("weight_entries", []):
            name = entry.get("name")
            if name in names_seen:
                errors.append(f"duplicate weight entry name: {name}")
            if name is not None:
                names_seen.add(name)
            file_name = entry.get("file")
            if not file_name:
                errors.append(f"weight entry {name!r} missing file")
                continue
            weight_path = package_dir / "weights" / file_name
            if not weight_path.exists():
                errors.append(f"missing weights file: weights/{file_name}")

    inputs = manifest.get("inputs", [])
    outputs = manifest.get("outputs", [])
    input_names = [entry.get("name") for entry in inputs]
    output_names = [entry.get("name") for entry in outputs]
    if len(set(input_names)) != len(input_names):
        errors.append("manifest.json input names must be unique")
    if len(set(output_names)) != len(output_names):
        errors.append("manifest.json output names must be unique")

    for case_id in manifest.get("case_ids", []):
        case_dir = package_dir / "cases" / case_id
        if not case_dir.exists():
            errors.append(f"missing case directory: cases/{case_id}")
            continue

        inputs_npz = case_dir / "inputs.npz"
        meta_json = case_dir / "meta.json"
        if not inputs_npz.exists():
            errors.append(f"missing inputs file: cases/{case_id}/inputs.npz")
        if not meta_json.exists():
            errors.append(f"missing meta file: cases/{case_id}/meta.json")

        if inputs_npz.exists():
            with np.load(inputs_npz, allow_pickle=False) as data:
                files = set(data.files)
                expected = set(name for name in input_names if name is not None)
                if files != expected:
                    errors.append(
                        f"cases/{case_id}/inputs.npz names mismatch: expected {sorted(expected)!r}, got {sorted(files)!r}"
                    )
                for entry in inputs:
                    name = entry["name"]
                    if name not in data.files:
                        continue
                    array = data[name]
                    if list(array.shape) != list(entry["shape"]):
                        errors.append(
                            f"cases/{case_id}/inputs.npz {name!r} shape mismatch: "
                            f"expected {entry['shape']!r}, got {list(array.shape)!r}"
                        )
                    actual_dtype = dtype_to_manifest_string(array.dtype)
                    if actual_dtype != entry["dtype"]:
                        errors.append(
                            f"cases/{case_id}/inputs.npz {name!r} dtype mismatch: "
                            f"expected {entry['dtype']!r}, got {actual_dtype!r}"
                        )

        if meta_json.exists():
            try:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"failed to parse cases/{case_id}/meta.json: {exc}")
                meta = None
            if meta is not None and meta.get("case_id") != case_id:
                errors.append(
                    f"cases/{case_id}/meta.json case_id mismatch: got {meta.get('case_id')!r}"
                )
            reference_available = bool(meta.get("reference_available")) if meta is not None else False
        else:
            reference_available = False

        reference_npz = case_dir / "reference_outputs.npz"
        if reference_available and not reference_npz.exists():
            errors.append(f"missing reference outputs: cases/{case_id}/reference_outputs.npz")

        if reference_npz.exists():
            with np.load(reference_npz, allow_pickle=False) as data:
                files = set(data.files)
                expected = set(name for name in output_names if name is not None)
                if files != expected:
                    errors.append(
                        "cases/"
                        f"{case_id}/reference_outputs.npz names mismatch: expected {sorted(expected)!r}, got {sorted(files)!r}"
                    )
                for entry in outputs:
                    name = entry["name"]
                    if name not in data.files:
                        continue
                    array = data[name]
                    if list(array.shape) != list(entry["shape"]):
                        errors.append(
                            f"cases/{case_id}/reference_outputs.npz {name!r} shape mismatch: "
                            f"expected {entry['shape']!r}, got {list(array.shape)!r}"
                        )
                    actual_dtype = dtype_to_manifest_string(array.dtype)
                    if actual_dtype != entry["dtype"]:
                        errors.append(
                            f"cases/{case_id}/reference_outputs.npz {name!r} dtype mismatch: "
                            f"expected {entry['dtype']!r}, got {actual_dtype!r}"
                        )

    return errors


def _aggregate_case_hashes(case_artifacts: Iterable[CaseArtifact], case_specs: Sequence[CaseSpec]) -> dict[str, str]:
    source_items = [
        {
            "case_id": case.case_id,
            "source_kind": case.source_kind,
            "source_payload": case.source_payload,
            "line_number": case.line_number,
        }
        for case in case_specs
    ]
    input_items = [
        {
            "case_id": artifact.case_id,
            "inputs_sha256": artifact.inputs_sha256,
        }
        for artifact in case_artifacts
    ]
    output_items = [
        {
            "case_id": artifact.case_id,
            "reference_outputs_sha256": artifact.reference_outputs_sha256,
        }
        for artifact in case_artifacts
    ]
    return {
        "cases_source_sha256": sha256_jsonable(source_items),
        "cases_inputs_sha256": sha256_jsonable(input_items),
        "cases_reference_outputs_sha256": sha256_jsonable(output_items),
    }


def export_model_package(
    checkpoint_path: str | Path,
    *,
    positions_file: str | Path,
    out_dir: str | Path,
    device: str | torch.device = "cpu",
    strict_load: bool = True,
    strict_export: bool = False,
    upstream_commit: str | None = None,
    producer_name: str = DEFAULT_PRODUCER_NAME,
    producer_version: str | None = None,
    producer_git_commit: str | None = None,
    notes: str | None = None,
    write_reference_outputs: bool = True,
    write_readable: bool = True,
    overwrite: bool = False,
    package_id_prefix: str = DEFAULT_PACKAGE_ID_PREFIX,
) -> ModelPackageArtifacts:
    package_dir = _prepare_output_dir(out_dir, overwrite=overwrite)
    graph_dir = package_dir / "graph"
    weights_dir = package_dir / "weights"
    cases_root = package_dir / "cases"
    graph_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    cases_root.mkdir(parents=True, exist_ok=True)

    evaluator = PolicyValueEvaluator.from_checkpoint(
        str(checkpoint_path),
        device=device,
        strict=strict_load,
    )
    model = evaluator.model

    example_input = make_zero_example_input(device=device)
    exported_program = export_reference_program(
        model,
        example_input,
        strict=strict_export,
    )

    graph_path = graph_dir / "model.pt2"
    save_exported_program(exported_program, graph_path)
    write_export_text_dumps(exported_program, graph_path, output_dir=graph_dir, stem="model")

    state_dict = _state_dict_cpu_contiguous(model)
    weights_file_name = "weights_000.safetensors"
    weights_path = weights_dir / weights_file_name
    write_safetensors_weights(state_dict, weights_path)
    weights_manifest = build_weights_manifest(state_dict, weights_file_name=weights_file_name)
    weights_manifest_path = weights_dir / "manifest.json"
    dump_manifest_json(weights_manifest, weights_manifest_path)

    case_specs = parse_positions_file(positions_file)
    case_artifacts: list[CaseArtifact] = []
    sample_outputs: dict[str, np.ndarray] | None = None
    for case in case_specs:
        case_dir = cases_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        input_tensor, raw_outputs, dense, legal = _forward_raw_outputs(
            evaluator,
            position=case.position,
            sfen=case.sfen,
        )
        sample_outputs = sample_outputs or raw_outputs
        input_array = np.asarray(input_tensor.numpy(), dtype=np.float32)
        inputs_arrays = {"features": input_array}
        inputs_sha256 = sha256_named_arrays(inputs_arrays)
        np.savez_compressed(case_dir / "inputs.npz", **inputs_arrays)

        reference_outputs_sha256: str | None = None
        reference_outputs_path: str | None = None
        if write_reference_outputs:
            reference_outputs_sha256 = sha256_named_arrays(raw_outputs)
            np.savez_compressed(case_dir / "reference_outputs.npz", **raw_outputs)
            reference_outputs_path = str((Path("cases") / case.case_id / "reference_outputs.npz").as_posix())

        meta = _build_case_meta(
            case,
            input_array=input_array,
            output_arrays=raw_outputs if write_reference_outputs else None,
            inputs_sha256=inputs_sha256,
            reference_outputs_sha256=reference_outputs_sha256,
        )
        meta_path = case_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        readable_path: str | None = None
        if write_readable:
            readable_text = _readable_case_text(case, dense=dense, legal=legal, raw_outputs=raw_outputs)
            (case_dir / "readable.txt").write_text(readable_text, encoding="utf-8")
            readable_path = str((Path("cases") / case.case_id / "readable.txt").as_posix())

        case_artifacts.append(
            CaseArtifact(
                case_id=case.case_id,
                inputs_path=str((Path("cases") / case.case_id / "inputs.npz").as_posix()),
                meta_path=str((Path("cases") / case.case_id / "meta.json").as_posix()),
                reference_outputs_path=reference_outputs_path,
                readable_path=readable_path,
                inputs_sha256=inputs_sha256,
                reference_outputs_sha256=reference_outputs_sha256,
            )
        )

    assert sample_outputs is not None

    bindings = extract_torch_export_bindings(exported_program)
    input_entries = _build_input_entries(example_input.detach().cpu(), bindings=bindings["inputs"])
    output_entries = _build_output_entries(sample_outputs, bindings=bindings["outputs"])

    if producer_version is None:
        producer_version = default_producer_version()
    if producer_git_commit is None:
        producer_git_commit = default_producer_git_commit()

    producer_metadata: dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "model_config": {
            "blocks": model.blocks_count,
            "channels": model.channels,
            "fcl": model.fcl,
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "state_dict_sha256": sha256_state_dict(state_dict),
        "graph_sha256": graph_sha256(exported_program),
        "weights_manifest_sha256": sha256_jsonable(weights_manifest),
        "weights_file_sha256": sha256_file(weights_path),
        "torch_version": torch.__version__,
        "export_input_strategy": "zero_tensor",
        "binding_metadata_version": 1,
        "upstream_commit": upstream_commit,
    }
    producer_metadata.update(_aggregate_case_hashes(case_artifacts, case_specs))

    manifest_payload: dict[str, Any] = {
        "package_type": PACKAGE_TYPE,
        "contract_version": CONTRACT_VERSION,
        "producer": {
            "name": producer_name,
            "version": producer_version,
            "git_commit": producer_git_commit,
        },
        "graph_artifact": {
            "kind": "pt2",
            "path": "graph/model.pt2",
        },
        "weights_artifact": {
            "manifest_path": "weights/manifest.json",
        },
        "inputs": input_entries,
        "outputs": output_entries,
        "case_ids": [case.case_id for case in case_specs],
        "producer_metadata": producer_metadata,
    }
    if notes is not None:
        manifest_payload["notes"] = notes

    package_id = build_model_package_id(manifest_payload, prefix=package_id_prefix)
    manifest = {**manifest_payload, "package_id": package_id}
    manifest_path = package_dir / "manifest.json"
    dump_manifest_json(manifest, manifest_path)

    errors = validate_model_package_dir(package_dir)
    if errors:
        raise RuntimeError("exported model package failed validation:\n- " + "\n- ".join(errors))

    return ModelPackageArtifacts(
        package_dir=str(package_dir),
        manifest_path=str(manifest_path),
        package_id=package_id,
        graph_path=str(graph_path),
        weights_manifest_path=str(weights_manifest_path),
        case_artifacts=case_artifacts,
    )
