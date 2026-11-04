---
title: Glassbox
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Watch which SAE features fire in Gemma 3, and steer them
---

# Glassbox

See which sparse-autoencoder features fire as Gemma 3 writes, and steer them.

**This Space runs in fake mode by default.** The interface works, but replies, features and labels are sample
data, not real model output. The free CPU tier is too slow to share the real model with visitors. For real
outputs, see the [precomputed demo](https://rjw-create.github.io/hack-or-treat-glassbox/demo/) or run Glassbox
on your own machine: [github.com/rjw-create/hack-or-treat-glassbox](https://github.com/rjw-create/hack-or-treat-glassbox).

To load the real model here, the Space owner sets the variable `GLASSBOX_REAL_MODEL` to `1` and adds a
Hugging Face read token as the secret `HF_TOKEN` (Gemma 3 is gated).

Code: Apache-2.0. Gemma 3 weights: [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
Gemma Scope 2 SAEs: Google DeepMind, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Neither is included in this Space; real mode downloads them at start-up.
