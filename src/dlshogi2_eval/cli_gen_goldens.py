from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .infer import PolicyValueEvaluator



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate dense policy/value golden outputs for a list of positions.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--positions-file", required=True, help="Text file with one `position ...` or `sfen ...` per line")
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="cpu")
    return p



def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    evaluator = PolicyValueEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    index = []
    with open(args.positions_file, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            position = None
            sfen = None
            if line.startswith("position "):
                position = line
            elif line.startswith("sfen "):
                sfen = line
            else:
                position = line

            dense, legal = evaluator.predict_from_position(position=position, sfen=sfen)
            stem = f"pos{lineno:04d}"
            np.savez_compressed(
                outdir / f"{stem}.npz",
                policy_logits=dense.policy_logits.numpy(),
                value_logit=np.asarray([dense.value_logit.item()], dtype=np.float32),
                legal_labels=legal.move_labels.numpy(),
                legal_probs=legal.probs.numpy(),
            )
            payload = {
                "line": line,
                "value_logit": float(dense.value_logit.item()),
                "value_prob": float(dense.value_prob.item()),
                "moves_usi": legal.moves_usi,
                "legal_labels": legal.move_labels.tolist(),
            }
            (outdir / f"{stem}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            index.append({"stem": stem, "line": line})

    (outdir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote goldens to {outdir}")


if __name__ == "__main__":  # pragma: no cover
    main()
