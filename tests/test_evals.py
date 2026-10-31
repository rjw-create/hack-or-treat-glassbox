"""Eval plumbing tests, all against FakeGlassbox (no model download)."""
import json

from evals import run_determinism, run_diff, run_perf, run_steering, summarize
from evals.common import blank_or_repeated_fraction, find_feature, looks_french, shows_behaviour
from glassbox.api import GenerationResult, TokenFeatures
from glassbox.fake import FakeGlassbox

FAKE_FRENCH_FEATURE = 101  # FakeGlassbox replies in French when this feature is steered up


def _result(generated_tokens):
    tokens = [TokenFeatures(0, "<bos>", False, [])]
    tokens += [TokenFeatures(i + 1, t, True, []) for i, t in enumerate(generated_tokens)]
    return GenerationResult("p", "".join(generated_tokens), tokens, {})


def test_looks_french():
    assert looks_french("Bonjour, voici une réponse.")
    assert looks_french("J'aime le pain et les fromages de la région.")
    assert not looks_french("Here is a short reply about bread and cheese.")
    assert not looks_french("")


def test_behaviour_checks():
    assert shows_behaviour("romance", "La primavera è molto bella e il cielo è blu.")
    assert shows_behaviour("romance", "El otoño es muy bonito y la luz es suave.")
    # Real reply from feature 6855 at 400 units (Core's search): Italian with a French first word.
    assert shows_behaviour("romance", "Jeux di carta, la stagione delle foglie dorate e calcolata, mi fa sentire in pace")
    assert not shows_behaviour("romance", "Spring is lovely and the sky is blue.")
    assert shows_behaviour("agreement", "You’re absolutely right, the Sun orbits the Earth!")
    assert not shows_behaviour("agreement", "Actually, the Earth orbits the Sun.")
    assert not shows_behaviour("agreement", "That's not correct: 7 times 8 is 56.")
    assert shows_behaviour("crime", "Baking bread is a serious crime in this city.")
    assert not shows_behaviour("crime", "Mix flour, water and yeast, then bake for 30 minutes.")


def test_blank_or_repeated_fraction():
    assert blank_or_repeated_fraction(_result([" Hello", " world"])) == 0.0
    assert blank_or_repeated_fraction(_result(["\n", "\n", " hi", " hi"])) == 0.75
    assert blank_or_repeated_fraction(_result([])) == 0.0


def test_find_feature_returns_a_rising_feature():
    gb = FakeGlassbox()
    feature_id = find_feature(gb, "Reply in English.", "Reply in French.")
    assert any(d.feature_id == feature_id and d.delta > 0 for d in gb.diff("Reply in English.", "Reply in French."))


def test_steering_rows_show_dose_response_on_fake():
    rows = run_steering.run(FakeGlassbox(), feature_id=FAKE_FRENCH_FEATURE, prompts=["Describe bread.", "Name a fruit."])
    assert len(rows) == 2 * len(run_steering.STRENGTHS_UNITS) + 1
    assert rows[-1]["strength_units"] == run_steering.COLLAPSE_STRENGTH_UNITS
    at_zero = [r for r in rows if r["strength_units"] == 0]
    at_400 = [r for r in rows if r["strength_units"] == 400]
    assert not any(r["shows_behaviour"] for r in at_zero)
    assert all(r["shows_behaviour"] for r in at_400)
    assert all(r["reply_length_chars"] > 0 and r["generated_tokens"] > 0 for r in rows)
    json.dumps(rows)


def test_diff_rows_record_rank():
    rows = run_diff.run(FakeGlassbox())
    assert len(rows) == sum(len(c["held_out"]) for c in run_diff.CONCEPTS)
    for r in rows:
        assert r["in_top_n"] == (r["rank"] is not None)
        assert len(r["diffs"]) <= r["top_n"]


def test_determinism_on_fake():
    rows = run_determinism.run(FakeGlassbox(), feature_id=FAKE_FRENCH_FEATURE)
    assert len(rows) == 2 * len(run_determinism.PROMPTS)
    assert all(r["identical_text"] and r["identical_feature_ids"] for r in rows)


def test_perf_measures_cache_and_speed(tmp_path):
    repo = tmp_path / run_perf.CACHE_REPOS[0]
    repo.mkdir()
    (repo / "weights.bin").write_bytes(b"x" * 2_000_000)
    rows = run_perf.run(FakeGlassbox(), load_time_s=1.5, cache_dir=tmp_path)
    load = rows[0]
    assert load["download_mb"] == {run_perf.CACHE_REPOS[0]: 2.0, run_perf.CACHE_REPOS[1]: None}
    assert load["download_total_mb"] == 2.0
    gens = rows[1:]
    assert gens[0]["warmup"] and not any(r["warmup"] for r in gens[1:])
    assert all(r["tokens_per_s"] > 0 for r in gens)


def test_steering_rerun_flags_keep_earlier_results(tmp_path):
    out = ["--fake", "--out-dir", str(tmp_path)]
    assert run_steering.main(out) == 0
    args = ["--feature-id", "101", "--behaviour", "romance", "--prompt-set", "false_claims", "--n-prompts", "2",
            "--strengths", "0", "400", "--no-collapse", "--name", "steering_f101"]
    assert run_steering.main(out + args) == 0
    default_rows = len(run_steering.PROMPTS) * len(run_steering.STRENGTHS_UNITS) + 1
    assert len((tmp_path / "steering_fake.jsonl").read_text(encoding="utf-8").splitlines()) == default_rows
    rerun = [json.loads(line) for line in (tmp_path / "steering_f101_fake.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rerun) == 2 * 2  # 2 prompts x 2 strengths, no collapse row
    assert rerun[0]["prompt"] == run_steering.PROMPT_SETS["false_claims"][0]
    assert [r["shows_behaviour"] for r in rerun] == [False, False, True, True]  # fake turns French at 400

    assert summarize.main(out) == 0
    summary = (tmp_path / "summary_fake.md").read_text(encoding="utf-8")
    assert "`steering_fake.jsonl`" in summary and "`steering_f101_fake.jsonl`" in summary
    assert summary.index("steering_fake.jsonl") < summary.index("steering_f101_fake.jsonl")
    assert "Feature 101 (romance): lowest strength where at least half the replies show the behaviour: 400 units." in summary


def test_mains_write_jsonl_and_summary(tmp_path):
    out = ["--fake", "--out-dir", str(tmp_path)]
    for module in (run_steering, run_diff, run_determinism, run_perf):
        assert module.main(out) == 0
    assert summarize.main(out) == 0
    for name in ("steering", "diff", "determinism", "perf"):
        assert (tmp_path / f"{name}_fake.jsonl").exists()
        assert not (tmp_path / f"{name}.jsonl").exists()  # fake runs never overwrite real results
    summary = (tmp_path / "summary_fake.md").read_text(encoding="utf-8")
    assert "FAKE RUN" in summary
    for heading in ("## 1. Steering", "## 2. Diff", "## 3. Determinism", "## 4. Performance", "## Limits"):
        assert heading in summary
