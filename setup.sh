#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.11}"
REQ_LINUX="requirements-llm.txt"
REQ_OSX="requirements-llm-osx.txt"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python binary $PYTHON_BIN not found or not executable." >&2
  echo "Set PYTHON_BIN to a valid interpreter (e.g., /usr/bin/python3) and rerun." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

pip install --upgrade pip

case "$(uname -s)" in
  Darwin)
    pip install -r "$REQ_OSX"
    ;;
  Linux)
    pip install -r "$REQ_LINUX"
    ;;
  *)
    echo "Unsupported OS: $(uname -s). Install dependencies manually." >&2
    exit 1
    ;;
esac

echo "Setup complete. Activate the venv with: source $VENV_DIR/bin/activate"
