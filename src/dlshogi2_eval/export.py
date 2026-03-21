from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .board_io import load_board
from .infer import PolicyValueEvaluator
from .manifest import build_reference_manifest
from .model import PolicyValueNetwork



def get_example_input(*, position: str | None = None, sfen: str | None = None, device: str | torch.device = "cpu") -> torch.Tensor:
    board = load_board(position=position, sfen=sfen)
    # Cheap temporary evaluator wrapper is enough to reuse encoding logic.
    model = PolicyValueNetwork().to(device).eval()
    evaluator = PolicyValueEvaluator(model, device=device)
    return evaluator.encode_board(board)



def export_reference_program(
    model: PolicyValueNetwork,
    example_input: torch.Tensor,
    *,
    strict: bool = False,
) -> torch.export.ExportedProgram:
    model.eval()
    return torch.export.export(model, (example_input,), strict=strict)



def graph_module_from_exported_program(ep: torch.export.ExportedProgram) -> torch.fx.GraphModule:
    return ep.graph_module



def save_exported_program(
    ep: torch.export.ExportedProgram,
    out_path: str | Path,
    *,
    extra_files: dict[str, Any] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path, extra_files=extra_files)



def export_checkpoint_reference(
    checkpoint_path: str | Path,
    *,
    out_path: str | Path,
    position: str | None = None,
    sfen: str | None = None,
    device: str | torch.device = "cpu",
    strict_load: bool = True,
    strict_export: bool = False,
    upstream_commit: str | None = None,
) -> dict[str, Any]:
    evaluator = PolicyValueEvaluator.from_checkpoint(
        str(checkpoint_path),
        device=device,
        strict=strict_load,
    )
    example_input = get_example_input(position=position, sfen=sfen, device=device)
    ep = export_reference_program(evaluator.model, example_input, strict=strict_export)
    save_exported_program(ep, out_path)
    manifest = build_reference_manifest(
        model=evaluator.model,
        checkpoint_path=checkpoint_path,
        exported_program=ep,
        example_input=example_input,
        example_position=position,
        example_sfen=sfen,
        upstream_commit=upstream_commit,
    )
    return manifest
