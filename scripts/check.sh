#!/usr/bin/env bash
# Verify everything: fast unit tests, then a real-model smoke run.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "== unit tests =="
"$PY" -m pytest -q

echo "== smoke test (real model) =="
"$PY" -m glassbox.smoke
