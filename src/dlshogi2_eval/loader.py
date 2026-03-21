from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import PolicyValueNetwork



def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
    blocks: int = 10,
    channels: int = 192,
    fcl: int = 256,
) -> PolicyValueNetwork:
    """Load the eval-only network from an upstream-style checkpoint.

    Upstream `python-dlshogi2` stores weights under `checkpoint['model']`.
    This loader also accepts a raw state_dict directly.
    """

    checkpoint_path = Path(checkpoint_path)
    model = PolicyValueNetwork(blocks=blocks, channels=channels, fcl=fcl)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if not strict and (missing or unexpected):
        print(
            f"[dlshogi2_eval] non-strict load: missing={missing}, unexpected={unexpected}",
            flush=True,
        )

    model.eval()
    return model
