#!/usr/bin/env bash
# Verify everything: fast unit tests, then a real-model smoke run.
# --fast stops after the unit tests: the smoke run loads the model and takes minutes on a shared CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "== unit tests =="
"$PY" -m pytest -q

if [ "${1:-}" = "--fast" ]; then echo "== fast checks passed (smoke skipped; run without --fast before a PR) =="; exit 0; fi

echo "== smoke test (real model) =="
"$PY" -m glassbox.smoke
