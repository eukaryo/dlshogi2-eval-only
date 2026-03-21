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
    p.add_argument(
        "--no-text-dumps",
        action="store_true",
        help="Do not write human-readable exported_program / graph_ir / graph_module_code files.",
    )
    p.add_argument(
        "--text-dump-dir",
        default=None,
        help="Optional directory for text dumps. Defaults to the directory containing --out.",
    )
    p.add_argument(
        "--text-dump-stem",
        default=None,
        help="Optional filename stem for text dumps. Defaults to the stem of --out.",
    )
    return p



def main() -> None:
    args = build_parser().parse_args()
    artifacts = export_checkpoint_reference(
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        position=args.position,
        sfen=args.sfen,
        device=args.device,
        strict_load=args.strict_load,
        strict_export=args.strict_export,
        upstream_commit=args.upstream_commit,
        write_text_dumps=not args.no_text_dumps,
        text_dump_dir=args.text_dump_dir,
        text_dump_stem=args.text_dump_stem,
    )
    manifest_path = args.manifest or f"{args.out}.manifest.json"
    dump_manifest_json(artifacts.manifest, manifest_path)
    print(f"wrote export to {args.out}")
    print(f"wrote manifest to {manifest_path}")
    if artifacts.text_dump_paths:
        print("wrote text dumps:")
        for key, path in artifacts.text_dump_paths.items():
            print(f"  {key}: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
