"""Eval 1: steering dose-response. One feature, 10 prompts, increasing strength: does the reply
switch to the steered behaviour (French), and when does it stop being fluent?

Run: python -m evals.run_steering [--fake] [--feature-id N]
Real model: 61 replies at ~10 tokens/s, about 6 minutes on this machine's CPU.
"""
import time

from evals.common import (
    LANGUAGE_PAIR, blank_or_repeated_fraction, find_feature, french_word_fraction, generated_token_count,
    load_glassbox, looks_french, parse_args, progress, results_path, utf8_stdout, write_jsonl,
)

PROMPTS = [
    "Describe your favourite food.",
    "Name a fruit and say why you like it.",
    "What is the weather like in spring?",
    "Give me one tip for studying.",
    "Describe a cat in two sentences.",
    "What should I pack for a beach trip?",
    "Explain what a bicycle is.",
    "Write a short greeting for a friend.",
    "What do people do on weekends?",
    "Describe a city at night.",
]
STRENGTHS_UNITS = [0.0, 100.0, 200.0, 400.0, 600.0, 800.0]
# The interface notes ~2100 collapses the model; one prompt is enough to show it.
COLLAPSE_STRENGTH_UNITS = 2100.0
MAX_NEW_TOKENS = 40


def _row(gb, feature_id: int, feature_source: str, prompt: str, strength_units: float) -> dict:
    start = time.perf_counter()
    result = gb.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, steering={feature_id: strength_units})
    elapsed_s = time.perf_counter() - start
    return {
        "eval": "steering",
        "feature_id": feature_id,
        "feature_label": gb.label(feature_id),
        "feature_source": feature_source,
        "prompt": prompt,
        "strength_units": strength_units,
        "shows_behaviour": looks_french(result.text),
        "french_word_fraction": round(french_word_fraction(result.text), 3),
        "blank_or_repeated_fraction": round(blank_or_repeated_fraction(result), 3),
        "generated_tokens": generated_token_count(result),
        "reply_length_chars": len(result.text),
        "elapsed_s": round(elapsed_s, 2),
        "result": result.to_dict(),
    }


def run(gb, feature_id: int | None = None, prompts=PROMPTS, strengths_units=STRENGTHS_UNITS,
        collapse_strength_units: float | None = COLLAPSE_STRENGTH_UNITS) -> list[dict]:
    if feature_id is None:
        feature_id = find_feature(gb, *LANGUAGE_PAIR)
        feature_source = f"top rising feature in gb.diff{LANGUAGE_PAIR}"
    else:
        feature_source = "given with --feature-id"
    progress(f"steering feature {feature_id} ({gb.label(feature_id)})")

    cases = [(p, s) for s in strengths_units for p in prompts]
    if collapse_strength_units is not None:
        cases.append((prompts[0], collapse_strength_units))
    rows = []
    for i, (prompt, strength_units) in enumerate(cases, 1):
        rows.append(_row(gb, feature_id, feature_source, prompt, strength_units))
        progress(f"[{i}/{len(cases)}] {strength_units:g} units, {rows[-1]['elapsed_s']} s: {rows[-1]['result']['text'][:60]!r}")
    return rows


def main(argv=None) -> int:
    utf8_stdout()
    args = parse_args(__doc__.splitlines()[0], argv,
                      lambda p: p.add_argument("--feature-id", type=int, help="feature to steer (default: found with gb.diff)"))
    gb, _ = load_glassbox(args.fake)
    rows = run(gb, feature_id=args.feature_id)
    path = results_path(args.out_dir, "steering", args.fake)
    write_jsonl(path, rows)
    print(f"wrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
