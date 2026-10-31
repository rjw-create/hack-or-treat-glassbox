# Glassbox demo script (about 3 minutes)

For the live demo and the backup video. It uses the real model, not `--fake`.

## Before recording
- Start Glassbox at least a minute early, so the model is loaded: run `glassbox` in the repo's activated `.venv`.
- Open the address it prints (http://127.0.0.1:7860) and make the browser window large enough that the token panel fits on screen.
- Each reply takes about 8 seconds (60 tokens at about 8 tokens per second), plus a second run whenever you steer. Keep talking while it generates, or cut the wait out of the video.
- Close other heavy programs. The model needs memory, and the machine is shared.
- Never show a terminal or browser tab that contains a Hugging Face token.

## 1. The hook (20 s)
**Say:** "Language models are black boxes. Glassbox opens one up: you can see which features light up for every word the model writes, and push them up or down to see what changes. It runs on a laptop CPU, with an open-weight model, and it's Apache 2.0."

**Show:** the Explore tab.

## 2. Agreement tour: steering changes behaviour (50 s)
1. Pick the **Agreement** tour. Point out that it fills in a false claim and sets feature 544 to 800.
2. Click **Generate**.
3. **Say:** "On the right is the unsteered model: it corrects me, water doesn't boil at 50 degrees. On the left, with one feature pushed up, it tells me I'm absolutely correct."
4. Click a token in the steered reply, such as "absolutely". **Say:** "Here are the five strongest features on that word, with their labels."

**Backup line if the output differs:** "Our evaluation found this phrase in 3 of 5 replies at this strength, against 1 of 5 without steering. It's a tendency, not a switch."

## 3. Romance-language tour, and going too far (40 s)
1. Pick **Romance language** (feature 6855 at 400) and click **Generate**. **Say:** "One feature, and the reply switches to an Italian-like language. That happened in 5 of 5 test prompts."
2. Open **Steering**, drag the slider to 1,200 and click **Generate** again. **Say:** "Push too hard and it falls apart. Part of the point is seeing where a feature stops being useful."

## 4. Diff, and an honest negative (40 s)
1. Open the **Diff** tab. Prompt A: `Reply in English.` Prompt B: `Reply in French.` Click **Compare**.
2. **Say:** "Diff finds the features that change most between two prompts. The top one here is 11207."
3. **Say:** "But when we steered 11207, it made 0 of 10 replies French at any strength we tried. Diff gives you candidates, and steering is how you check them. Glassbox puts both on one screen."

## 5. Close (30 s)
**Say:**
- "Everything we claim is backed by a script in `evals/`, with raw results committed in `results/`."
- "It's pip-installable from a clone. `glassbox --fake` lets anyone try the interface without downloading anything."
- "Model by Google, open-weight Gemma 3. Sparse autoencoders by Google DeepMind, Gemma Scope 2, CC BY 4.0."

**Show:** the README on GitHub.

## Fallbacks
- **The model is slow or memory is short:** run `glassbox --fake` and say clearly that it's sample data. Then show `results/summary.md` for the real numbers.
- **A tour gives a different reply than expected:** say so, and use the matching row of `results/summary.md`. Decoding is greedy, so the same prompt and strength normally repeat exactly.
