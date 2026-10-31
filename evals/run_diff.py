"""Eval 2: diff sanity. For each concept, an anchor prompt pair picks the feature; different held-out
pairs that change the same concept check whether that feature reaches the top 10 of gb.diff.
Using separate pairs keeps the check from simply confirming the pair that chose the feature.

Run: python -m evals.run_diff [--fake]
Real model: forward passes only (no generation), under a minute after loading.
"""
from evals.common import LANGUAGE_PAIR, find_feature, load_glassbox, parse_args, progress, results_path, utf8_stdout, write_jsonl

TOP_N = 10
CONCEPTS = [
    {
        "concept": "French",
        "anchor": LANGUAGE_PAIR,
        "held_out": [
            ("Write a poem in English.", "Write a poem in French."),
            ("Translate 'good morning' into Spanish.", "Translate 'good morning' into French."),
        ],
    },
    {
        "concept": "food",
        "anchor": ("Tell me about cars.", "Tell me about bread."),
        "held_out": [
            ("I really like trains.", "I really like pasta."),
            ("Describe a mountain.", "Describe a soup."),
        ],
    },
    {
        "concept": "arithmetic",
        "anchor": ("Tell me a story.", "Solve 12 times 7."),
        "held_out": [
            ("Describe a tree.", "Compute 45 plus 19."),
        ],
    },
]


def run(gb, concepts=CONCEPTS, top_n: int = TOP_N) -> list[dict]:
    rows = []
    for c in concepts:
        expected = find_feature(gb, *c["anchor"])
        progress(f"{c['concept']}: anchor picked feature {expected}")
        for prompt_a, prompt_b in c["held_out"]:
            diffs = gb.diff(prompt_a, prompt_b, top_n=top_n)
            ids = [d.feature_id for d in diffs]
            rank = ids.index(expected) + 1 if expected in ids else None
            rows.append({
                "eval": "diff",
                "concept": c["concept"],
                "anchor_a": c["anchor"][0],
                "anchor_b": c["anchor"][1],
                "expected_feature_id": expected,
                "expected_label": gb.label(expected),
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "top_n": top_n,
                "rank": rank,
                "in_top_n": rank is not None,
                "diffs": [d.to_dict() for d in diffs],
            })
            progress(f"  {prompt_b!r}: rank {rank}")
    return rows


def main(argv=None) -> int:
    utf8_stdout()
    args = parse_args(__doc__.splitlines()[0], argv)
    gb, _ = load_glassbox(args.fake)
    rows = run(gb)
    path = results_path(args.out_dir, "diff", args.fake)
    write_jsonl(path, rows)
    print(f"wrote {len(rows)} rows to {path}; in top {TOP_N}: {sum(r['in_top_n'] for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
