#!/usr/bin/env bash
# Setup pre-commit hooks for the byteff2 repository.
# Works with both venv (pip) and uv environments.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# -- ensure we are inside a git repository -----------------------------------
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: not inside a git repository. Aborting." >&2
    exit 1
fi

# -- detect package manager ---------------------------------------------------
# Prefer uv when available and the project uses it; otherwise fall back to pip.
USE_UV=false
if command -v uv &>/dev/null; then
    if [ -f "$REPO_ROOT/uv.lock" ] || [ -f "$REPO_ROOT/pyproject.toml" ]; then
        USE_UV=true
    fi
fi

# -- install pre-commit -------------------------------------------------------
echo "==> Installing pre-commit and ruff ..."
if $USE_UV; then
    uv pip install -U pre-commit ruff
else
    pip install -U pre-commit ruff
fi

# -- install the git hook scripts ---------------------------------------------
echo "==> Installing pre-commit hooks ..."
pre-commit install

echo "==> Done. Pre-commit hooks are active."
