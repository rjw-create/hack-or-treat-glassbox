# Glassbox

Glassbox lets you look inside a language model while it writes. Type a prompt and you get the reply. For each token, you also see the sparse-autoencoder features that fired most strongly, with a short label for each. You can turn individual features up or down and compare the new reply with the original.

It runs on your own machine. It uses Google's open-weight Gemma 3 models and the Gemma Scope 2 sparse autoencoders from Google DeepMind.

> Status: work in progress (Hack or Treat hackathon, 2026-10-31). Steps marked TODO depend on code that has not landed yet.

## What you need

- Python 3.11
- Git
- A free Hugging Face account (needed to download Gemma 3; see below)
- A GPU is recommended for the real model. You can try the interface without the model or a GPU by using `--fake` (see [Try it without the model](#try-it-without-the-model)).

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

- **Reply:** type a prompt and read the model's answer.
- **Per-token features:** click any token to see its top 5 features and what each one is labelled as.
- **Steering:** slide a feature up or down and generate again.
- **Compare runs:** see the steered reply next to the unsteered one.
- **Diff:** enter two prompts and see which features differ most between them.

<!-- TODO: add screenshots once the UI exists. -->

## How this differs from Neuronpedia

[Neuronpedia](https://www.neuronpedia.org) is the main tool for exploring Gemma Scope features, and Google links to its interactive Gemma Scope 2 demo from the Gemma Scope 2 page. It is open source and offers a free hosted website, an API, steering, feature dashboards with AI-written explanations, and instructions for self-hosting. If you want to browse features without installing anything, use Neuronpedia.

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
- **Neuronpedia** (Johnny Lin): inspiration and the reference tool in this space.

## License

Glassbox's own code is licensed under the [Apache License 2.0](LICENSE). That licence does not cover the Gemma 3 weights or the Gemma Scope 2 SAEs; they keep their own terms, listed above.
