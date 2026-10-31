"""End-to-end check with the real model through the public API: features line up with tokens,
steering changes the output, and diff returns ranked features.

Run with: python -m glassbox.smoke   (downloads ~2 GB on first run)
"""
import sys

from glassbox.api import Glassbox

# Strength in feature-activation units. The spike showed ~2100 collapses the model into blank lines.
SMOKE_STRENGTH = 400.0


def main() -> int:
    # Model output can contain emoji; the default Windows console encoding (cp1252) cannot print them.
    sys.stdout.reconfigure(encoding="utf-8")
    gb = Glassbox.load()
    failures = []

    base = gb.generate("Name a fruit.", max_new_tokens=10)
    with_features = [t for t in base.tokens if t.features]
    if len(with_features) != len(base.tokens) - 1 or base.tokens[-1].features:
        failures.append(f"features misaligned: {len(with_features)} of {len(base.tokens)} tokens have features")

    top_feature = base.tokens[-2].features[0][0]
    steered = gb.generate("Name a fruit.", max_new_tokens=10, steering={top_feature: SMOKE_STRENGTH})
    if steered.text == base.text:
        failures.append("steering did not change the output")

    diffs = gb.diff("Answer in English.", "Answer in French.", top_n=5)
    if len(diffs) != 5 or abs(diffs[0].delta) < abs(diffs[-1].delta):
        failures.append(f"diff not ranked: {[round(d.delta, 1) for d in diffs]}")

    print(f"tokens: {len(base.tokens)} ({sum(t.is_generated for t in base.tokens)} generated)")
    print(f"unsteered: {base.text!r}")
    print(f"steered (+feature {top_feature} x {SMOKE_STRENGTH}): {steered.text!r}")
    print("diff English -> French, top 5:", [(d.feature_id, round(d.delta, 1)) for d in diffs])
    for failure in failures:
        print("FAIL:", failure)
    if failures:
        return 1
    print("smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
