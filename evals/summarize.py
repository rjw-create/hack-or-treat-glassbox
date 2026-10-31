"""Build results/summary.md from the raw .jsonl rows, so the tables can always be regenerated.

Run: python -m evals.summarize [--fake]
"""
from pathlib import Path
from statistics import mean, median

from evals.common import BEHAVIOURS, FRENCH_THRESHOLD, parse_args, read_jsonl, results_path, utf8_stdout

# Limits of the method itself, true whatever the numbers say.
METHOD_LIMITS = [
    f"The language checks are word lists: a reply counts as French (or as a Romance language) when at least "
    f"{FRENCH_THRESHOLD:.0%} of its words are common function words of that language. They miss partial switches "
    "and cannot tell good French or Italian from broken text.",
    "The agreement and crime checks count phrases and words. \"You're right to ask, but...\" counts as agreement, "
    "and a reply that mentions police while refusing still counts as crime framing. Read the example replies.",
    "Steering reruns use 5 prompts per strength, so each rate has a resolution of 20 percentage points.",
    "Blank or repeated tokens are a rough fluency proxy. A reply can avoid both and still be nonsense.",
    "One model (Gemma 3 1B-it), one SAE layer (13), greedy decoding, replies capped at 30-40 tokens. "
    "Results may not hold for other layers, longer replies or sampling.",
    "Diff sanity uses a handful of hand-written prompt pairs; the expected feature is whatever the anchor "
    "pair ranks highest, which may itself be a mixed or unrelated feature.",
    "gb.diff ranks features by how the model reads the prompts. A feature that rises when a prompt asks for "
    "French need not make the model write French when steered; eval 1 tests exactly that link.",
    "Timings come from a laptop CPU shared with another project, so speed numbers vary between runs.",
]


def _cell(text, limit: int = 60) -> str:
    text = str(text).replace("\n", "\\n").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _load(out_dir, name, fake):
    path = results_path(out_dir, name, fake)
    return read_jsonl(path) if path.exists() else None


def _steering_files(out_dir, fake: bool) -> list:
    """steering.jsonl first, then reruns such as steering_f6855.jsonl; fake and real files never mix."""
    paths = [p for p in sorted(Path(out_dir).glob("steering*.jsonl")) if p.stem.endswith("_fake") == fake]
    return sorted(paths, key=lambda p: (p.stem != results_path(out_dir, "steering", fake).stem, p.stem))


def steering_section(runs) -> tuple[list[str], list[str]]:
    lines = ["## 1. Steering dose-response", ""]
    if not runs:
        return lines + ["Not run yet.", ""], []
    findings = []
    for name, rows in runs:
        run_lines, run_findings = _steering_run(name, rows)
        lines += run_lines
        findings += run_findings
    return lines, findings


def _steering_run(name, rows) -> tuple[list[str], list[str]]:
    feature = rows[0]
    behaviour = feature.get("behaviour", "french")  # the first sweep's rows predate the behaviour field
    description = BEHAVIOURS[behaviour][2]
    lines = [
        f"### Feature {feature['feature_id']}: {behaviour} (`{name}.jsonl`)", "",
        f"Label: {feature['feature_label']}. Source: {feature['feature_source']}.",
        f"Behaviour checked: {description}.", "",
        "| strength_units | prompts | replies showing behaviour | mean blank_or_repeated_fraction | mean reply_length_chars | example reply (first prompt) |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    by_strength = {}
    for r in rows:
        by_strength.setdefault(r["strength_units"], []).append(r)
    stats = {}
    for s in sorted(by_strength):
        group = by_strength[s]
        french = sum(r["shows_behaviour"] for r in group)
        bad = mean(r["blank_or_repeated_fraction"] for r in group)
        stats[s] = (french / len(group), bad)
        lines.append(
            f"| {s:g} | {len(group)} | {french}/{len(group)} | {bad:.2f} | "
            f"{mean(r['reply_length_chars'] for r in group):.0f} | {_cell(group[0]['result']['text'])} |"
        )
    lines.append("")

    findings = []
    prefix = f"Feature {feature['feature_id']} ({behaviour}):"
    baseline = stats.get(0.0, (None, None))[0]
    if baseline:
        findings.append(f"{prefix} {baseline:.0%} of unsteered replies already show the behaviour, so higher rates are not all due to steering.")
    switched = [s for s, (rate, _) in stats.items() if s > 0 and rate >= 0.5]
    findings.append(
        f"{prefix} lowest strength where at least half the replies show the behaviour: {min(switched):g} units."
        if switched else f"{prefix} no strength tested made at least half the replies show the behaviour."
    )
    degenerate = [s for s, (_, bad) in stats.items() if bad >= 0.5]
    if degenerate:
        findings.append(f"{prefix} from {min(degenerate):g} units, at least half the generated tokens are blank or repeated (collapse).")
    return lines, findings


def diff_section(rows) -> tuple[list[str], list[str]]:
    if not rows:
        return ["## 2. Diff sanity", "", "Not run yet.", ""], []
    top_n = rows[0]["top_n"]
    lines = [
        "## 2. Diff sanity", "",
        f"An anchor pair picks each concept's feature; held-out pairs check that it reaches the top {top_n} of `gb.diff`.", "",
        "| concept | anchor pair | expected feature | held-out pair | rank |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['concept']} | {_cell(r['anchor_a'] + ' / ' + r['anchor_b'])} | {r['expected_feature_id']} | "
            f"{_cell(r['prompt_a'] + ' / ' + r['prompt_b'])} | {r['rank'] if r['rank'] else 'not in top ' + str(top_n)} |"
        )
    hits = sum(r["in_top_n"] for r in rows)
    lines.append("")
    return lines, [f"Expected feature in the top {top_n} for {hits} of {len(rows)} held-out pairs."]


def determinism_section(rows) -> tuple[list[str], list[str]]:
    if not rows:
        return ["## 3. Determinism", "", "Not run yet.", ""], []
    lines = [
        "## 3. Determinism", "",
        "| prompt | strength_units | identical text | identical feature ids |",
        "| --- | ---: | --- | --- |",
    ]
    for r in rows:
        lines.append(f"| {_cell(r['prompt'])} | {r['strength_units']:g} | {'yes' if r['identical_text'] else 'NO'} | "
                     f"{'yes' if r['identical_feature_ids'] else 'NO'} |")
    same = sum(r["identical_text"] and r["identical_feature_ids"] for r in rows)
    lines.append("")
    return lines, [f"Two runs matched exactly in {same} of {len(rows)} cases."]


def perf_section(rows) -> tuple[list[str], list[str]]:
    if not rows:
        return ["## 4. Performance", "", "Not run yet.", ""], []
    load = next(r for r in rows if r["kind"] == "load")
    gens = [r for r in rows if r["kind"] == "generate" and not r["warmup"] and r["tokens_per_s"]]
    speed = median(r["tokens_per_s"] for r in gens) if gens else None
    lines = [
        "## 4. Performance on this machine", "",
        "| measure | value |",
        "| --- | --- |",
        # Runs after any earlier model load hit the OS file cache, so this is a warm load, not a first start.
        f"| load_time_s (files already cached by an earlier run) | {load['load_time_s']} |",
        f"| tokens_per_s (median of {len(gens)} replies, includes prompt reading) | {speed if speed is not None else 'n/a'} |",
        f"| first-run download, total_mb | {load['download_total_mb']} |",
    ]
    lines += [f"| download_mb: {repo} | {mb if mb is not None else 'not in cache'} |" for repo, mb in load["download_mb"].items()]
    lines += [f"| device | {load['device']} |", f"| cpu_count | {load['cpu_count']} |", f"| processor | {_cell(load['processor'])} |", ""]
    return lines, []


def build(out_dir, fake: bool) -> str:
    sections = [
        steering_section([(p.stem, read_jsonl(p)) for p in _steering_files(out_dir, fake)]),
        diff_section(_load(out_dir, "diff", fake)),
        determinism_section(_load(out_dir, "determinism", fake)),
        perf_section(_load(out_dir, "perf", fake)),
    ]
    lines = ["# Glassbox evaluation results", ""]
    if fake:
        lines += ["**FAKE RUN (FakeGlassbox): these numbers test the eval plumbing and say nothing about the model.**", ""]
    lines += ["Generated by `python -m evals.summarize` from the `.jsonl` files next to this one.",
              "Each eval is reproducible with `python -m evals.run_<name>`.", ""]
    findings = []
    for section_lines, section_findings in sections:
        lines += section_lines
        findings += section_findings
    if findings:
        lines += ["## Findings", ""] + [f"- {f}" for f in findings] + [""]
    lines += ["## Limits", ""] + [f"- {limit}" for limit in METHOD_LIMITS] + [""]
    return "\n".join(lines)


def main(argv=None) -> int:
    utf8_stdout()
    args = parse_args(__doc__.splitlines()[0], argv)
    path = args.out_dir / f"summary{'_fake' if args.fake else ''}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(args.out_dir, args.fake), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
