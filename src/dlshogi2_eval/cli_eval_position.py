from __future__ import annotations

import argparse
import json

from .infer import PolicyValueEvaluator



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate one shogi position with the eval-only python-dlshogi2 core.")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint.pth")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--position", help='USI position string, e.g. "position startpos moves 7g7f"')
    g.add_argument("--sfen", help="SFEN string")
    p.add_argument("--device", default="cpu")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--include-all-legal", action="store_true")
    p.add_argument("--include-dense-policy", action="store_true")
    p.add_argument("--pretty", action="store_true")
    return p



def main() -> None:
    args = build_parser().parse_args()
    evaluator = PolicyValueEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    dense, legal = evaluator.predict_from_position(position=args.position, sfen=args.sfen, temperature=args.temperature)
    payload = {
        "dense": dense.to_jsonable(include_dense_policy=args.include_dense_policy),
        "legal": legal.to_jsonable(topk=args.topk, include_all_legal=args.include_all_legal),
    }
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
