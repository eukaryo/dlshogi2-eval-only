from __future__ import annotations

import argparse

from .export import export_checkpoint_reference
from .manifest import dump_manifest_json



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a torch.export reference program for the eval-only python-dlshogi2 core.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="Output .pt2 path")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--position")
    g.add_argument("--sfen")
    p.add_argument("--device", default="cpu")
    p.add_argument("--strict-load", action="store_true")
    p.add_argument("--strict-export", action="store_true")
    p.add_argument("--upstream-commit", default=None)
    p.add_argument("--manifest", default=None, help="Optional output path for manifest JSON")
    return p



def main() -> None:
    args = build_parser().parse_args()
    manifest = export_checkpoint_reference(
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        position=args.position,
        sfen=args.sfen,
        device=args.device,
        strict_load=args.strict_load,
        strict_export=args.strict_export,
        upstream_commit=args.upstream_commit,
    )
    manifest_path = args.manifest or f"{args.out}.manifest.json"
    dump_manifest_json(manifest, manifest_path)
    print(f"wrote export to {args.out}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
