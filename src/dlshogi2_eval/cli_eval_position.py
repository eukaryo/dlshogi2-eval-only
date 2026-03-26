from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .infer import PolicyValueEvaluator, make_backend_payload, raw_output_arrays_from_dense


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate one shogi position with the eval-only python-dlshogi2 core."
    )
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint.pth")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--position", help='USI position string, e.g. "position startpos moves 7g7f"')
    g.add_argument("--sfen", help="SFEN string")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Computation precision used for the PyTorch backend.",
    )
    p.add_argument(
        "--autocast",
        action="store_true",
        help="Wrap inference in torch.autocast using the selected precision.",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--include-all-legal", action="store_true")
    p.add_argument("--include-dense-policy", action="store_true")
    p.add_argument(
        "--raw-outputs-npz",
        help="Optional path to save canonical raw outputs as an .npz with keys policy_logits/value_logit.",
    )
    p.add_argument("--pretty", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    evaluator = PolicyValueEvaluator.from_checkpoint(
        args.checkpoint,
        device=args.device,
        precision=args.precision,
        use_autocast=args.autocast,
    )
    dense, legal = evaluator.predict_from_position(
        position=args.position,
        sfen=args.sfen,
        temperature=args.temperature,
    )
    if args.raw_outputs_npz:
        raw_outputs = raw_output_arrays_from_dense(dense)
        out_path = Path(args.raw_outputs_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **raw_outputs)

    payload = make_backend_payload(
        backend=evaluator.backend_spec,
        dense=dense,
        legal=legal,
        include_dense_policy=args.include_dense_policy,
        topk=args.topk,
        include_all_legal=args.include_all_legal,
    )
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
