"""Eval 3: determinism. The same prompt and steering, generated twice, should give identical text
(Glassbox uses greedy decoding) and identical recorded features.

Run: python -m evals.run_determinism [--fake] [--feature-id N]
Real model: 12 replies of up to 30 tokens, about 1 minute after loading.
"""
from evals.common import LANGUAGE_PAIR, find_feature, load_glassbox, parse_args, progress, results_path, utf8_stdout, write_jsonl

PROMPTS = ["Describe your favourite food.", "Give me one tip for studying.", "Describe a city at night."]
STEERED_STRENGTH_UNITS = 400.0
MAX_NEW_TOKENS = 30


def _feature_ids(result) -> list[list[int]]:
    # Feature ids per token; activations are floats and are compared through the ids and the text.
    return [[f for f, _ in t["features"]] for t in result["tokens"]]


def run(gb, feature_id: int | None = None, prompts=PROMPTS) -> list[dict]:
    if feature_id is None:
        feature_id = find_feature(gb, *LANGUAGE_PAIR)
    rows = []
    for prompt in prompts:
        for steering in ({}, {feature_id: STEERED_STRENGTH_UNITS}):
            first = gb.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, steering=steering).to_dict()
            second = gb.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, steering=steering).to_dict()
            rows.append({
                "eval": "determinism",
                "prompt": prompt,
                "feature_id": feature_id if steering else None,
                "strength_units": steering.get(feature_id, 0.0),
                "identical_text": first["text"] == second["text"],
                "identical_feature_ids": _feature_ids(first) == _feature_ids(second),
                "first": first,
                "second": second,
            })
            progress(f"{prompt!r} at {rows[-1]['strength_units']:g} units: identical={rows[-1]['identical_text']}")
    return rows


def main(argv=None) -> int:
    utf8_stdout()
    args = parse_args(__doc__.splitlines()[0], argv,
                      lambda p: p.add_argument("--feature-id", type=int, help="feature to steer (default: found with gb.diff)"))
    gb, _ = load_glassbox(args.fake)
    rows = run(gb, feature_id=args.feature_id)
    path = results_path(args.out_dir, "determinism", args.fake)
    write_jsonl(path, rows)
    print(f"wrote {len(rows)} rows to {path}; identical text: {sum(r['identical_text'] for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
