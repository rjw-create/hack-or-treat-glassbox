# open-source-project

Hack or Treat hackathon entry (2026-10-31) for one category only: **Best Open-Source AI Project**.
Chosen idea: **Glassbox** (see the user's pitch in the build plan; the other candidates are in `2026-10-31-ideas.md`).
Several sessions work in parallel; who owns which paths is in `2026-10-31-ROLES.md`.

## Focus
- The deliverable is something other people can reuse: public repo, `LICENSE` (MIT or Apache 2.0), and a README a stranger can follow.
- Use only open-weight models in the core path. No closed, paid APIs that are required to run it.
- This machine has a GPU shared with another hackathon project. Ask before starting long GPU jobs (training, large models).

## Status
- Stack: Python 3.11 venv (`py -3.11 -m venv .venv`), torch (CPU build), transformers, sae-lens; pinned in `pyproject.toml`.
  UI (in `ui/`, owned by Surface): Gradio, as the optional `ui` extra (`pip install -e ".[ui]"`).
- Model: `google/gemma-3-1b-it` (Gemma terms, gated: say "open-weight") + Gemma Scope 2 SAE
  `gemma-scope-2-1b-it-res` / `layer_13_width_16k_l0_medium` (CC-BY-4.0, credit Google DeepMind).
- Hardware: Intel Arc 140V integrated GPU, no CUDA. Runs on CPU at ~10 tokens/s.
- Verify command (Git Bash): `./scripts/check.sh`, unit tests then a real-model smoke run (~1 min).
- API contract for UI and evals: `docs/2026-10-31-interface.md`.
