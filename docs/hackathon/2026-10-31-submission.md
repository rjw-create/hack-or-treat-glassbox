# Glassbox: Hack or Treat 2026 submission

**Category:** Best Open-Source AI Project  
**Repository:** https://github.com/rjw-create/hack-or-treat-glassbox (Apache 2.0)  
**Live demo:** https://rjw-create.github.io/hack-or-treat-glassbox/demo/ (precomputed real outputs; the full tool runs locally)

## One line
Glassbox lets you watch which sparse-autoencoder features fire, token by token, as an open-weight language model writes, and steer those features up or down to see how the reply changes.

## What it is
Glassbox is a Python package and a small web interface, built on Google's open-weight Gemma 3 1B model and Google DeepMind's Gemma Scope 2 sparse autoencoders.

- **Per-token features:** type a prompt and click any token in the reply. You see its 5 strongest features, with hand-written labels where we have checked them.
- **Steering:** set a feature's strength (-1,200 to +1,200) and see the steered reply next to the unsteered one.
- **Diff:** compare two prompts to find candidate features. Then confirm them by steering.
- **Guided tours:** three ready-made examples: a Romance-language feature, an agreement ("you're absolutely right") feature, and a crime feature.

## Why it fits open source
- **Reusable:** Apache 2.0 licence. `pip install -e ".[ui]"` from a clone, then `glassbox`. The same `Glassbox` class can be imported into your own scripts. Its interface is written down in `docs/2026-10-31-interface.md`.
- **Runs on ordinary hardware:** it runs on a laptop CPU, with no GPU or paid API. About 8 tokens per second on our laptop (median 7.83).
- **Only open-weight parts:** the model and the SAEs are free downloads (about 2.2 GB in total). Your prompts stay on your machine, and the interface turns off Gradio's telemetry.
- **Try it without the model:** `glassbox --fake` runs the interface on sample data, with no download or account.
- **Reproducible evidence:** every result below comes from a script in `evals/`, and the raw results are committed in `results/`. Decoding is greedy, and two repeat runs matched exactly in 6 of 6 cases.

## What we measured
5 prompts per strength (10 for feature 11207), greedy decoding, 30-40 tokens, checked with simple word lists. Details and limits are in `results/summary.md`.

| Feature | What it does when steered | Replies showing it |
| --- | --- | --- |
| 6855 | Switches the reply into a Romance language (mostly Italian-like) | 0/5 unsteered, **5/5 at 400**; garbled by 1,200 |
| 544 | "You're absolutely right" agreement | 1/5 unsteered, **3/5 at 800**, 4/5 at 1,200 |
| 1235 | Brings crime into harmless answers, e.g. baking bread | 0/5 unsteered, 2/5 at 800, **4/5 at 1,200** |
| 11207 | Nothing visible: top diff feature for "Reply in French", but steering never produced French | **0/10 at every strength up to 800**; collapses at 2,100 |

The last row is the honest lesson: a feature that ranks high in a diff does not always steer. Glassbox puts diff and steering side by side so you can check.

## Limits
- One model (Gemma 3 1B instruction-tuned), one layer (13), one SAE. Greedy decoding with short replies.
- 5 prompts per strength, so each rate moves in 20-point steps. The checks are word lists: they count a phrase, not whether the model really means it.
- Labels are written by hand from observed behaviour, not proven. We did not find a refusal feature.

## How it differs from Neuronpedia
Neuronpedia is the main hosted tool for exploring Gemma Scope features, and we recommend it for browsing. Glassbox is narrower:
- a local, scriptable package built around one prompt at a time;
- per-token features, steering and diff on one screen;
- with its own measured evaluation.

## Credits
- **Gemma 3** by Google: open-weight model, used under the Gemma Terms of Use. Weights are downloaded from Hugging Face by each user and are not redistributed.
- **Gemma Scope 2** by Google DeepMind: sparse autoencoders, CC BY 4.0.
- **SAELens:** used to load the SAEs.
- **Gradio:** used for the interface.
- **Neuronpedia:** inspiration and the reference tool in this space.
