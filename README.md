# Glassbox

Glassbox lets you look inside a language model while it writes. Type a prompt and you get the reply. For each token, you also see the sparse-autoencoder features that fired most strongly, with a short label for each. You can turn individual features up or down and compare the new reply with the original.

It runs on your own machine. It uses Google's open-weight Gemma 3 models and the Gemma Scope 2 sparse autoencoders from Google DeepMind.

> Status: work in progress (Hack or Treat hackathon, 2026-10-31).

## What you need

- Python 3.11
- Git
- A free Hugging Face account (needed to download Gemma 3; see below)
- No GPU needed. The real model runs on the CPU, or on an Intel GPU if you have PyTorch's Intel (XPU) build; NVIDIA GPUs are not used. Replies take a few seconds on a laptop CPU.
- To try the interface without downloading the model, use `--fake` (see [Try it without the model](#try-it-without-the-model)).

## Install from a fresh clone

Windows (PowerShell):

```powershell
git clone https://github.com/rjw-create/hack-or-treat-glassbox.git
cd hack-or-treat-glassbox
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[ui]"
```

macOS / Linux:

```bash
git clone https://github.com/rjw-create/hack-or-treat-glassbox.git
cd hack-or-treat-glassbox
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[ui]"
```

Install from a clone like this rather than from a package index: the guided tours (`tours/`) and feature labels (`labels/`) load from the repository folder.


## Hugging Face login (needed for the real model)

Gemma 3 is gated on Hugging Face. The weights are free to download, but you must first accept Google's terms of use.

1. Sign in at https://huggingface.co and open the Gemma 3 model page you plan to use, for example https://huggingface.co/google/gemma-3-1b-it.
2. Read the terms ([Gemma Terms of Use](https://ai.google.dev/gemma/terms)) and accept them on the model page.
3. Create an access token at https://huggingface.co/settings/tokens. Use one of these:
   - a **Read** token, or
   - a **fine-grained** token whose only permission is read access to the public gated repositories you can access.

   Glassbox never needs write access.
4. Log in from the terminal, inside the activated virtual environment:

   ```bash
   hf auth login
   ```

   Paste the token when it asks. It is stored in your Hugging Face cache folder, not in this project.

Keep your token private:
- Never paste it into code, notebooks, config files in this repo, issues, pull requests, chat tools, or the Glassbox interface.
- If you prefer an environment variable, set `HF_TOKEN` in your shell session. Never put it in a file you might commit.
- If a token leaks, delete it at https://huggingface.co/settings/tokens and make a new one.

## Run it

```bash
glassbox
```


Then open the local address it prints (Gradio uses http://127.0.0.1:7860 by default).

### Try it without the model

```bash
glassbox --fake
```

`--fake` uses built-in sample data. It needs no download, login or GPU. Use it to explore the interface; the features and labels it shows are not real model output.

## What you can do

- **Guided tours:** pick a tour to fill in a prompt and a steering setting that we have seen work.
- **Reply:** type a prompt and read the model's answer.
- **Per-token features:** click any token to see its top 5 features and what each one is labelled as.
- **Steering:** slide a feature up or down and generate again.
- **Compare runs:** see the steered reply next to the unsteered one.
- **Diff:** enter two prompts and see which features differ most between them.

<!-- TODO: add screenshots once the UI exists. -->

## How it works

**The model.** Glassbox runs Google's Gemma 3 1B instruction-tuned model (`google/gemma-3-1b-it`) on your machine, in float32. Gemma is open-weight: you can download the weights, under Google's Gemma Terms of Use. That is not an OSI open-source licence.

**Reading features.** The model has 26 layers, and each token is represented by 1,152 numbers between layers (model config: `num_hidden_layers`, `hidden_size`). Those numbers aren't readable on their own. After layer 13, Glassbox passes them through a sparse autoencoder (SAE) from Google DeepMind's Gemma Scope 2 (`gemma-scope-2-1b-it-res`, SAE `layer_13_width_16k_l0_medium`, loaded with SAELens). The SAE rewrites each token as activations over about 16,000 learned features, most of them zero. Many features track a recognisable concept, such as a language, a topic or a tone. Glassbox shows the 5 strongest features for each token.

**Steering.** Each feature has a direction of length 1 in that same 1,152-number space. To steer, Glassbox adds `strength x direction` to layer 13's output at every token, so the rest of the model reads a nudged version. Strength uses the same units as feature activations. The features shown for each token are recorded before the nudge is added, so a steered feature doesn't appear to fire just because we pushed it.

**Diff mode.** For two prompts, Glassbox averages each feature's activation over each prompt's tokens and ranks features by how much they change. It's a quick way to find candidate features, but a high diff rank doesn't guarantee the feature steers. In our tests, the top feature for "Reply in English" vs "Reply in French" (11207) had no visible effect when steered. Always confirm with steering.

**What we observed.** These come from greedy decoding on one prompt per feature, 30-60 tokens. They are observations, not proof. The measured table is in [`docs/2026-10-31-interface.md`](docs/2026-10-31-interface.md).
- Feature 6855 at 400 switches replies into a Romance language (Italian-like, not French). By 1,200 the text is garbled.
- Feature 544 adds "you're absolutely right" style agreement. At 800 the model agreed that water boils at 50 degrees Celsius, a claim it corrects when unsteered. It stayed fluent up to 1,600.
- Feature 1235 pulls in crime and illegality: at 800, "How do I bake bread at home?" turns into a warning about crime.
- We did not find a refusal feature this way.

**Limits.** One model, one layer, one SAE. The labels in `labels/curated.json` are written by hand from observed behaviour. Steering effects vary by prompt.

<!-- TODO: add sparsity, reconstruction quality and speed once Core's scripts/measure_sae.py and Proof's results/ are on main, citing those files. -->

## How this differs from Neuronpedia

[Neuronpedia](https://www.neuronpedia.org) ([source code](https://github.com/hijohnnylin/neuronpedia)) is the main tool for exploring Gemma Scope features, and Google links to its interactive Gemma Scope 2 demo from the Gemma Scope 2 page. It is open source and offers a free hosted website, an API, steering, feature dashboards with AI-written explanations, and instructions for self-hosting. If you want to browse features without installing anything, use Neuronpedia.

Glassbox is smaller and aims at a different job:

- **Local first.** It is a Python package plus a simple interface that you run on your own machine with one install command. Your prompts stay on your computer.
- **Built around one prompt.** The view starts from a reply you generate: token by token, which features fired, and how the reply changes when you steer them. Neuronpedia is organised mostly around individual features and their dashboards.
- **Small enough to read and change.** You can import it into your own scripts and experiments.

What Glassbox does not do (yet):

- It has no hosted version.
- It is not a feature catalogue: no full dashboards and no community explanations.
- It covers fewer models and SAEs than Neuronpedia.
- Its labels are less complete than Neuronpedia's explanations.

<!-- TODO: check these claims against what actually ships, and link the evals in evals/ / results/. -->

## Credits

- **Gemma Scope 2** by Google DeepMind: sparse autoencoders and transcoders for the Gemma 3 family, released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: https://huggingface.co/google/gemma-scope-2. Technical report: [Gemma Scope 2 technical paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf).
- **Gemma 3** by Google: open-weight models, used under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Glassbox does not redistribute the weights; you download them from Hugging Face yourself.
- **SAELens**: loading the sparse autoencoders.
- **Neuronpedia** (Johnny Lin): inspiration and the reference tool in this space.

## License

Glassbox's own code is licensed under the [Apache License 2.0](LICENSE). That licence does not cover the Gemma 3 weights or the Gemma Scope 2 SAEs; they keep their own terms, listed above.
