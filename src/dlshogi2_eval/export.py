from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .manifest import build_reference_manifest
from .model import PolicyValueNetwork

if TYPE_CHECKING:  # pragma: no cover
    from .infer import PolicyValueEvaluator


@dataclass(frozen=True)
class ExportTextDumps:
    exported_program: str
    graph_ir: str
    graph_module_code: str


@dataclass(frozen=True)
class ExportArtifacts:
    manifest: dict[str, Any]
    text_dump_paths: dict[str, str]


def get_example_input(*, position: str | None = None, sfen: str | None = None, device: str | torch.device = "cpu") -> torch.Tensor:
    from .board_io import load_board
    from .infer import PolicyValueEvaluator

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



def build_export_text_dumps(ep: torch.export.ExportedProgram) -> ExportTextDumps:
    graph_module = graph_module_from_exported_program(ep)
    return ExportTextDumps(
        exported_program=str(ep),
        graph_ir=str(ep.graph),
        graph_module_code=graph_module.code,
    )



def default_text_dump_paths(
    out_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    stem: str | None = None,
) -> dict[str, Path]:
    out_path = Path(out_path)
    dump_dir = Path(output_dir) if output_dir is not None else out_path.parent
    dump_stem = stem or out_path.stem
    return {
        "exported_program": dump_dir / f"{dump_stem}.exported_program.txt",
        "graph_ir": dump_dir / f"{dump_stem}.graph_ir.txt",
        "graph_module_code": dump_dir / f"{dump_stem}.graph_module_code.py",
    }



def write_export_text_dumps(
    ep: torch.export.ExportedProgram,
    out_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    stem: str | None = None,
) -> dict[str, str]:
    dumps = build_export_text_dumps(ep)
    paths = default_text_dump_paths(out_path, output_dir=output_dir, stem=stem)
    contents = {
        "exported_program": dumps.exported_program,
        "graph_ir": dumps.graph_ir,
        "graph_module_code": dumps.graph_module_code,
    }
    for key, text in contents.items():
        path = paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}



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
    write_text_dumps: bool = True,
    text_dump_dir: str | Path | None = None,
    text_dump_stem: str | None = None,
) -> ExportArtifacts:
    from .infer import PolicyValueEvaluator

    evaluator = PolicyValueEvaluator.from_checkpoint(
        str(checkpoint_path),
        device=device,
        strict=strict_load,
    )
    example_input = get_example_input(position=position, sfen=sfen, device=device)
    ep = export_reference_program(evaluator.model, example_input, strict=strict_export)
    save_exported_program(ep, out_path)
    text_dump_paths: dict[str, str] = {}
    if write_text_dumps:
        text_dump_paths = write_export_text_dumps(
            ep,
            out_path,
            output_dir=text_dump_dir,
            stem=text_dump_stem,
        )
    manifest = build_reference_manifest(
        model=evaluator.model,
        checkpoint_path=checkpoint_path,
        exported_program=ep,
        example_input=example_input,
        example_position=position,
        example_sfen=sfen,
        upstream_commit=upstream_commit,
    )
    return ExportArtifacts(manifest=manifest, text_dump_paths=text_dump_paths)
