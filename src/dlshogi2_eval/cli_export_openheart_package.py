from __future__ import annotations

import argparse

from .openheart_package import export_openheart_package


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export an openheart_model_package_v1 bundle from the eval-only "
            "python-dlshogi2 core."
        )
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--positions-file", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--strict-load", action="store_true")
    p.add_argument("--strict-export", action="store_true")
    p.add_argument("--upstream-commit", default=None)
    p.add_argument("--producer-name", default="dlshogi2-eval-only")
    p.add_argument("--producer-version", default=None)
    p.add_argument("--producer-git-commit", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--package-id-prefix", default="openheart-dlshogi2")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-reference-outputs", action="store_true")
    p.add_argument("--no-readable", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    artifacts = export_openheart_package(
        checkpoint_path=args.checkpoint,
        positions_file=args.positions_file,
        out_dir=args.outdir,
        device=args.device,
        strict_load=args.strict_load,
        strict_export=args.strict_export,
        upstream_commit=args.upstream_commit,
        producer_name=args.producer_name,
        producer_version=args.producer_version,
        producer_git_commit=args.producer_git_commit,
        notes=args.notes,
        write_reference_outputs=not args.no_reference_outputs,
        write_readable=not args.no_readable,
        overwrite=args.overwrite,
        package_id_prefix=args.package_id_prefix,
    )
    print(f"wrote openheart package to {artifacts.package_dir}")
    print(f"package_id: {artifacts.package_id}")
    print(f"manifest: {artifacts.manifest_path}")
    print(f"graph: {artifacts.graph_path}")
    print(f"weights_manifest: {artifacts.weights_manifest_path}")
    print(f"cases: {len(artifacts.case_artifacts)}")


if __name__ == "__main__":  # pragma: no cover
    main()
