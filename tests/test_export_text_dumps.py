from __future__ import annotations

from pathlib import Path

import torch

from dlshogi2_eval.export import build_export_text_dumps, default_text_dump_paths, write_export_text_dumps


class TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).relu()


def test_write_export_text_dumps(tmp_path: Path) -> None:
    model = TinyNet().eval()
    x = torch.randn(1, 4)
    ep = torch.export.export(model, (x,), strict=False)

    paths = default_text_dump_paths(tmp_path / 'tiny.pt2')
    assert paths['exported_program'].name == 'tiny.exported_program.txt'
    assert paths['graph_ir'].name == 'tiny.graph_ir.txt'
    assert paths['graph_module_code'].name == 'tiny.graph_module_code.py'

    dumps = build_export_text_dumps(ep)
    assert 'ExportedProgram' in dumps.exported_program
    assert 'graph(' in dumps.graph_ir
    assert 'def forward' in dumps.graph_module_code

    written = write_export_text_dumps(ep, tmp_path / 'tiny.pt2')
    for key, path in written.items():
        text = Path(path).read_text(encoding='utf-8')
        assert text.strip(), key
