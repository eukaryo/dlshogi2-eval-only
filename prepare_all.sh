#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end bootstrap for the frozen eval-only reference repo.
#
# What this script does by default:
#   1. Create / reuse a local venv
#   2. Install runtime + dev dependencies
#   3. Download a pinned upstream python-dlshogi2 snapshot from GitHub
#   4. Reuse the upstream checkpoint from that pinned snapshot
#   5. Run unit tests
#   6. Run smoke inference / export / golden generation
#   7. Write hashes and environment metadata under artifacts/prepare_all/
#
# The defaults are intentionally reproducibility-oriented. Override behavior with
# environment variables if needed. Examples:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu ./prepare_all.sh
#   RECREATE_VENV=1 ./prepare_all.sh
#   UPSTREAM_COMMIT=e053d8c CHECKPOINT_NAME=checkpoint-001.pth ./prepare_all.sh

LOG_PREFIX="[prepare_all]"

log() {
  printf '%s %s\n' "$LOG_PREFIX" "$*" >&2
}

fail() {
  printf '%s ERROR: %s\n' "$LOG_PREFIX" "$*" >&2
  exit 1
}

on_err() {
  local exit_code=$?
  printf '%s ERROR at line %s while running: %s\n' "$LOG_PREFIX" "$1" "$2" >&2
  exit "$exit_code"
}
trap 'on_err "$LINENO" "$BASH_COMMAND"' ERR

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

repo_root() {
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    "$PYTHON_BIN_REAL" - <<'PY' "$path"
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
h = hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
print(h.hexdigest())
PY
  fi
}

ROOT="$(repo_root)"
cd "$ROOT"

[[ -f pyproject.toml ]] || fail "run this script from the repo root (pyproject.toml not found)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_BIN_REAL="$(command -v "$PYTHON_BIN" || true)"
[[ -n "$PYTHON_BIN_REAL" ]] || fail "python executable not found: $PYTHON_BIN"

need_cmd "$PYTHON_BIN"
need_cmd curl
need_cmd tar
need_cmd mkdir
need_cmd rm
need_cmd printf

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT/artifacts/prepare_all}"
THIRD_PARTY_UPSTREAM_DIR="${THIRD_PARTY_UPSTREAM_DIR:-$ROOT/third_party/upstream}"
# By default, bootstrap from a maintained mirror fork under our control.
# Provenance is still recorded against the original upstream repository below.
UPSTREAM_OWNER="${UPSTREAM_OWNER:-eukaryo}"
UPSTREAM_REPO="${UPSTREAM_REPO:-python-dlshogi2}"
PROVENANCE_UPSTREAM_REPOSITORY="${PROVENANCE_UPSTREAM_REPOSITORY:-https://github.com/TadaoYamaoka/python-dlshogi2}"
BOOTSTRAP_MIRROR_REPOSITORY="${BOOTSTRAP_MIRROR_REPOSITORY:-https://github.com/${UPSTREAM_OWNER}/${UPSTREAM_REPO}}"
# Pin the exact snapshot used for this frozen eval-only reference.
UPSTREAM_COMMIT="${UPSTREAM_COMMIT:-358a704eb3ebc87871fff36a436eaad233d85a44}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint.pth}"
UPSTREAM_SNAPSHOT_BASENAME="${UPSTREAM_SNAPSHOT_BASENAME:-${UPSTREAM_OWNER}-${UPSTREAM_REPO}-${UPSTREAM_COMMIT}}"
UPSTREAM_ARCHIVE_URL="${UPSTREAM_ARCHIVE_URL:-https://codeload.github.com/${UPSTREAM_OWNER}/${UPSTREAM_REPO}/tar.gz/${UPSTREAM_COMMIT}}"
UPSTREAM_ARCHIVE_PATH="${UPSTREAM_ARCHIVE_PATH:-$THIRD_PARTY_UPSTREAM_DIR/${UPSTREAM_SNAPSHOT_BASENAME}.tar.gz}"
UPSTREAM_SNAPSHOT_DIR="${UPSTREAM_SNAPSHOT_DIR:-$THIRD_PARTY_UPSTREAM_DIR/${UPSTREAM_SNAPSHOT_BASENAME}}"
CHECKPOINT_REL_PATH="${CHECKPOINT_REL_PATH:-third_party/upstream/${UPSTREAM_SNAPSHOT_BASENAME}/checkpoints/${CHECKPOINT_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT/${CHECKPOINT_REL_PATH}}"
RECREATE_VENV="${RECREATE_VENV:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
RUN_COMPILEALL="${RUN_COMPILEALL:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_PIP_CHECK="${RUN_PIP_CHECK:-1}"
SMOKE_POSITION="${SMOKE_POSITION:-position startpos moves 7g7f 3c3d 2g2f 8c8d}"
EXPORT_POSITION="${EXPORT_POSITION:-position startpos}"
PIP_UPGRADE_SPECS=("pip>=24" "setuptools>=68" "wheel")
TORCH_SPEC="${TORCH_SPEC:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

if [[ -n "$TORCH_INDEX_URL" && -z "$TORCH_SPEC" ]]; then
  TORCH_SPEC="torch>=2.1"
fi

mkdir -p "$ARTIFACT_DIR" "$THIRD_PARTY_UPSTREAM_DIR"

log "Checking Python version."
"$PYTHON_BIN_REAL" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 is required, got {sys.version.split()[0]}")
print(sys.version.split()[0])
PY

if [[ "$RECREATE_VENV" == "1" && -d "$VENV_DIR" ]]; then
  log "Removing existing venv: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating venv at $VENV_DIR"
  "$PYTHON_BIN_REAL" -m venv "$VENV_DIR"
else
  log "Reusing existing venv at $VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
[[ -x "$VENV_PY" ]] || fail "venv python not found at $VENV_PY"
[[ -x "$VENV_PIP" ]] || fail "venv pip not found at $VENV_PIP"

log "Upgrading packaging tooling."
"$VENV_PIP" install -U "${PIP_UPGRADE_SPECS[@]}" ${PIP_EXTRA_ARGS}

if [[ -n "$TORCH_SPEC" ]]; then
  if [[ -n "$TORCH_INDEX_URL" ]]; then
    log "Installing torch via explicit index: $TORCH_INDEX_URL"
    "$VENV_PIP" install --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC" ${PIP_EXTRA_ARGS}
  else
    log "Installing explicit torch spec: $TORCH_SPEC"
    "$VENV_PIP" install "$TORCH_SPEC" ${PIP_EXTRA_ARGS}
  fi
fi

log "Installing this repo in editable mode with dev extras."
"$VENV_PIP" install -e ".[dev]" ${PIP_EXTRA_ARGS}

if [[ "$RUN_PIP_CHECK" == "1" ]]; then
  log "Running pip check."
  "$VENV_PY" -m pip check
fi

if [[ "$FORCE_DOWNLOAD" == "1" && -f "$UPSTREAM_ARCHIVE_PATH" ]]; then
  log "Removing cached upstream archive: $UPSTREAM_ARCHIVE_PATH"
  rm -f "$UPSTREAM_ARCHIVE_PATH"
fi
if [[ "$FORCE_DOWNLOAD" == "1" && -d "$UPSTREAM_SNAPSHOT_DIR" ]]; then
  log "Removing cached upstream snapshot: $UPSTREAM_SNAPSHOT_DIR"
  rm -rf "$UPSTREAM_SNAPSHOT_DIR"
fi

if [[ ! -f "$UPSTREAM_ARCHIVE_PATH" ]]; then
  log "Downloading pinned upstream snapshot: $UPSTREAM_ARCHIVE_URL"
  curl -L --fail --retry 3 --retry-delay 2 "$UPSTREAM_ARCHIVE_URL" -o "$UPSTREAM_ARCHIVE_PATH"
else
  log "Reusing cached upstream archive: $UPSTREAM_ARCHIVE_PATH"
fi

if [[ ! -d "$UPSTREAM_SNAPSHOT_DIR" ]]; then
  log "Extracting upstream snapshot to $UPSTREAM_SNAPSHOT_DIR"
  mkdir -p "$UPSTREAM_SNAPSHOT_DIR"
  tar -xzf "$UPSTREAM_ARCHIVE_PATH" -C "$UPSTREAM_SNAPSHOT_DIR" --strip-components=1
else
  log "Reusing extracted upstream snapshot: $UPSTREAM_SNAPSHOT_DIR"
fi

[[ -f "$CHECKPOINT_PATH" ]] || fail "checkpoint not found inside upstream snapshot: $CHECKPOINT_PATH"
[[ -f "$UPSTREAM_SNAPSHOT_DIR/pydlshogi2/features.py" ]] || fail "upstream features.py not found"
[[ -f "$UPSTREAM_SNAPSHOT_DIR/pydlshogi2/network/policy_value_resnet.py" ]] || fail "upstream policy_value_resnet.py not found"
[[ -f "$UPSTREAM_SNAPSHOT_DIR/LICENSE" ]] || fail "upstream LICENSE not found"

UPSTREAM_ARCHIVE_SHA256="$(sha256_file "$UPSTREAM_ARCHIVE_PATH")"
CHECKPOINT_SHA256="$(sha256_file "$CHECKPOINT_PATH")"
UPSTREAM_FEATURES_SHA256="$(sha256_file "$UPSTREAM_SNAPSHOT_DIR/pydlshogi2/features.py")"
UPSTREAM_MODEL_SHA256="$(sha256_file "$UPSTREAM_SNAPSHOT_DIR/pydlshogi2/network/policy_value_resnet.py")"
UPSTREAM_LICENSE_SHA256="$(sha256_file "$UPSTREAM_SNAPSHOT_DIR/LICENSE")"

log "Recording environment metadata."
"$VENV_PY" - <<'PY' "$ARTIFACT_DIR/environment.json"
import json, platform, sys
from pathlib import Path
import numpy as np
import torch
try:
    import cshogi
    cshogi_version = getattr(cshogi, '__version__', 'unknown')
except Exception as exc:  # pragma: no cover
    cshogi_version = f'unavailable: {exc}'
Path(sys.argv[1]).write_text(json.dumps({
    'python': sys.version,
    'platform': platform.platform(),
    'torch': torch.__version__,
    'numpy': np.__version__,
    'cshogi': cshogi_version,
}, indent=2, ensure_ascii=False), encoding='utf-8')
PY

"$VENV_PY" -m pip freeze > "$ARTIFACT_DIR/pip-freeze.txt"

cat > "$ARTIFACT_DIR/upstream_snapshot.json" <<EOF_JSON
{
  "bootstrap_mirror_repository": "${BOOTSTRAP_MIRROR_REPOSITORY}",
  "provenance_upstream_repository": "${PROVENANCE_UPSTREAM_REPOSITORY}",
  "upstream_owner": "${UPSTREAM_OWNER}",
  "upstream_repo": "${UPSTREAM_REPO}",
  "upstream_commit": "${UPSTREAM_COMMIT}",
  "upstream_archive_url": "${UPSTREAM_ARCHIVE_URL}",
  "upstream_archive_path": "${UPSTREAM_ARCHIVE_PATH}",
  "upstream_archive_sha256": "${UPSTREAM_ARCHIVE_SHA256}",
  "checkpoint_name": "${CHECKPOINT_NAME}",
  "checkpoint_path": "${CHECKPOINT_PATH}",
  "checkpoint_sha256": "${CHECKPOINT_SHA256}",
  "upstream_features_sha256": "${UPSTREAM_FEATURES_SHA256}",
  "upstream_model_sha256": "${UPSTREAM_MODEL_SHA256}",
  "upstream_license_sha256": "${UPSTREAM_LICENSE_SHA256}"
}
EOF_JSON

cat > "$ARTIFACT_DIR/UPSTREAM_SNAPSHOT.generated.md" <<EOF_MD
# Upstream snapshot record (generated by prepare_all.sh)

- Upstream repository: \`${PROVENANCE_UPSTREAM_REPOSITORY}\`
- Bootstrap mirror repository: \`${BOOTSTRAP_MIRROR_REPOSITORY}\`
- Upstream commit hash: \`${UPSTREAM_COMMIT}\`
- Snapshot archive URL: \`${UPSTREAM_ARCHIVE_URL}\`
- Checkpoint source path: \`${CHECKPOINT_PATH}\`
- Checkpoint SHA256: \`${CHECKPOINT_SHA256}\`
- Vendored / adapted files:
  - \`pydlshogi2/features.py\` (SHA256: \`${UPSTREAM_FEATURES_SHA256}\`)
  - \`pydlshogi2/network/policy_value_resnet.py\` (SHA256: \`${UPSTREAM_MODEL_SHA256}\`)
  - \`LICENSE\` (SHA256: \`${UPSTREAM_LICENSE_SHA256}\`)
EOF_MD

if [[ "$RUN_COMPILEALL" == "1" ]]; then
  log "Running compileall."
  "$VENV_PY" -m compileall src tests scripts
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  log "Running pytest."
  "$VENV_PY" -m pytest
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  log "Running smoke inference."
  "$VENV_PY" -m dlshogi2_eval.cli_eval_position \
    --checkpoint "$CHECKPOINT_REL_PATH" \
    --position "$SMOKE_POSITION" \
    --topk 10 \
    --pretty > "$ARTIFACT_DIR/smoke_eval.json"

  log "Exporting reference program and human-readable text dumps."
  "$VENV_PY" -m dlshogi2_eval.cli_export_reference \
    --checkpoint "$CHECKPOINT_REL_PATH" \
    --position "$EXPORT_POSITION" \
    --out "$ARTIFACT_DIR/reference_startpos.pt2" \
    --manifest "$ARTIFACT_DIR/reference_startpos.manifest.json" \
    --text-dump-dir "$ARTIFACT_DIR" \
    --text-dump-stem "reference_startpos" \
    --upstream-commit "$UPSTREAM_COMMIT"

  cat > "$ARTIFACT_DIR/positions.txt" <<'EOF_POS'
position startpos
position startpos moves 7g7f
position startpos moves 7g7f 3c3d
position startpos moves 7g7f 3c3d 2g2f
position startpos moves 7g7f 3c3d 2g2f 8c8d
EOF_POS

  log "Generating golden outputs."
  "$VENV_PY" -m dlshogi2_eval.cli_gen_goldens \
    --checkpoint "$CHECKPOINT_REL_PATH" \
    --positions-file "$ARTIFACT_DIR/positions.txt" \
    --outdir "$ARTIFACT_DIR/goldens"

  log "Running a checkpoint load / export roundtrip smoke check."
  "$VENV_PY" - <<'PY' "$CHECKPOINT_REL_PATH" "$ARTIFACT_DIR/reference_startpos.pt2" "$EXPORT_POSITION"
import sys
import torch
from dlshogi2_eval.export import get_example_input
from dlshogi2_eval.infer import PolicyValueEvaluator

ckpt, pt2_path, position = sys.argv[1:4]
evaluator = PolicyValueEvaluator.from_checkpoint(ckpt, device='cpu')
x = get_example_input(position=position, device='cpu')
policy_ref, value_ref = evaluator.model(x)
ep = torch.export.load(pt2_path)
module = ep.module()
policy_exp, value_exp = module(x)
if not torch.allclose(policy_ref, policy_exp):
    raise SystemExit('exported policy output differs from eager output')
if not torch.allclose(value_ref, value_exp):
    raise SystemExit('exported value output differs from eager output')
print('roundtrip ok')
PY
fi

cat > "$ARTIFACT_DIR/summary.txt" <<EOF_SUM
prepare_all completed successfully.

Repo root: $ROOT
Venv: $VENV_DIR
Artifacts: $ARTIFACT_DIR
Provenance upstream repository: $PROVENANCE_UPSTREAM_REPOSITORY
Bootstrap mirror repository: $BOOTSTRAP_MIRROR_REPOSITORY
Upstream commit: $UPSTREAM_COMMIT
Checkpoint (absolute): $CHECKPOINT_PATH
Checkpoint (repo-relative): $CHECKPOINT_REL_PATH
Checkpoint SHA256: $CHECKPOINT_SHA256
ExportedProgram text: $ARTIFACT_DIR/reference_startpos.exported_program.txt
Graph IR text: $ARTIFACT_DIR/reference_startpos.graph_ir.txt
GraphModule code: $ARTIFACT_DIR/reference_startpos.graph_module_code.py
EOF_SUM

log "Done. Summary:"
cat "$ARTIFACT_DIR/summary.txt"
