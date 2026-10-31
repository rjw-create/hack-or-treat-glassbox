"""Eval 4: performance on this machine. Load time, generation speed and the size of the first-run
download (measured from the Hugging Face cache, which holds exactly what was downloaded).

Run: python -m evals.run_perf [--fake]
Real model: one load plus 4 replies of up to 40 tokens, about 1.5 minutes.
"""
import os
import platform
import time
from pathlib import Path

from evals.common import generated_token_count, load_glassbox, parse_args, progress, results_path, utf8_stdout, write_jsonl

PROMPTS = ["Describe your favourite food.", "Explain what a bicycle is.", "Describe a city at night."]
MAX_NEW_TOKENS = 40
# Hugging Face cache folders for the model and the SAE repo (see glassbox/model.py and glassbox/sae.py).
CACHE_REPOS = ["models--google--gemma-3-1b-it", "models--google--gemma-scope-2-1b-it"]


def default_hf_cache() -> Path:
    # Same lookup order as huggingface_hub: HF_HUB_CACHE, then HF_HOME/hub, then ~/.cache/huggingface/hub.
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def folder_size_mb(folder: Path) -> float | None:
    if not folder.is_dir():
        return None
    total_bytes = 0
    for root, _, files in os.walk(folder):
        for name in files:
            path = Path(root) / name
            # Snapshot entries can be symlinks to blobs; count each file's bytes once.
            if not path.is_symlink():
                total_bytes += path.stat().st_size
    return round(total_bytes / 1e6, 1)


def run(gb, load_time_s: float, cache_dir: Path | None = None, prompts=PROMPTS) -> list[dict]:
    cache_dir = cache_dir or default_hf_cache()
    download_mb = {repo: folder_size_mb(cache_dir / repo) for repo in CACHE_REPOS}
    model = getattr(gb, "model", None)
    rows = [{
        "eval": "perf",
        "kind": "load",
        "load_time_s": round(load_time_s, 2),
        "download_mb": download_mb,
        "download_total_mb": round(sum(v for v in download_mb.values() if v), 1),
        "device": str(model.device) if model is not None else "fake",
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
    }]
    # The first reply also pays one-off warm-up costs, so it is recorded but marked.
    for i, prompt in enumerate([prompts[0], *prompts]):
        start = time.perf_counter()
        result = gb.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
        elapsed_s = time.perf_counter() - start
        n = generated_token_count(result)
        rows.append({
            "eval": "perf",
            "kind": "generate",
            "warmup": i == 0,
            "prompt": prompt,
            "generated_tokens": n,
            "elapsed_s": round(elapsed_s, 3),
            # Includes reading the prompt (prefill), so it is what a user waits for, not pure decode speed.
            "tokens_per_s": round(n / elapsed_s, 2) if elapsed_s > 0 else None,
        })
        progress(f"{'warm-up ' if i == 0 else ''}{prompt!r}: {n} tokens in {elapsed_s:.1f} s")
    return rows


def main(argv=None) -> int:
    utf8_stdout()
    args = parse_args(__doc__.splitlines()[0], argv)
    gb, load_time_s = load_glassbox(args.fake)
    progress(f"loaded in {load_time_s:.1f} s")
    rows = run(gb, load_time_s)
    path = results_path(args.out_dir, "perf", args.fake)
    write_jsonl(path, rows)
    print(f"wrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
