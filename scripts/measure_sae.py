"""Measure the numbers the README quotes about the model and the SAE, so each one has a reproducible source.

Run from the repo root (loads the model, about 1-2 minutes on a laptop CPU):
    python scripts/measure_sae.py

Prints one JSON object and saves it to scripts/measure_sae_output.json, which the README cites:
- settings: device, dtype, library versions and the prompts measured
- model config: number of layers and hidden size
- SAE width (number of features)
- mean active features per token (L0)
- fraction of variance explained by the SAE's reconstruction at layer 13
- layer-13 residual-stream norm (median, p10, p90), the yardstick for steering strength
All token statistics skip position 0 (<bos>), whose activations are far larger than any other token's.
"""
import json
import sys
from importlib.metadata import version
from pathlib import Path

import torch

from glassbox.api import STEERING_LIMIT, Glassbox
from glassbox.model import decoder_layers
from glassbox.sae import DEFAULT_LAYER, DEFAULT_SAE_ID, DEFAULT_SAE_RELEASE

OUTPUT_PATH = Path(__file__).resolve().parent / "measure_sae_output.json"

PROMPTS = [
    "Write one sentence about autumn in Paris.",
    "Explain what a bicycle is.",
    "What is 12 times 7?",
    "Give me one tip for studying.",
    "Describe a city at night.",
    "Tell me about your morning.",
    "Parle-moi de ta matinée.",
    "Write a short greeting for a friend.",
    "How do I bake bread at home?",
    "What should I pack for a beach trip?",
]


def percentile(sorted_values: list[float], fraction: float) -> float:
    return sorted_values[int(fraction * (len(sorted_values) - 1))]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    gb = Glassbox.load()
    sae = gb.hook.sae
    captured = []
    handle = decoder_layers(gb.model)[DEFAULT_LAYER].register_forward_hook(
        lambda _m, _i, out: captured.append((out[0] if isinstance(out, tuple) else out)[0, 1:].detach())
    )
    try:
        with torch.inference_mode():
            for prompt in PROMPTS:
                gb.model(**gb._chat_inputs(prompt))
    finally:
        handle.remove()

    resid = torch.cat(captured).to(sae.W_dec.dtype)
    with torch.inference_mode():
        feats = sae.encode(resid)
        recon = sae.decode(feats)
    unexplained = (resid - recon).pow(2).sum() / (resid - resid.mean(dim=0)).pow(2).sum()
    norms = sorted(resid.norm(dim=-1).tolist())
    config = gb.model.config.get_text_config() if hasattr(gb.model.config, "get_text_config") else gb.model.config

    report = {
        "model": gb.model.config.name_or_path,
        "sae": f"{DEFAULT_SAE_RELEASE}/{DEFAULT_SAE_ID}",
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "sae_class": type(sae).__name__,
        "sae_features": sae.W_dec.shape[0],
        "layer": DEFAULT_LAYER,
        "device": str(gb.model.device),
        "dtype": str(resid.dtype),
        "versions": {lib: version(lib) for lib in ("torch", "transformers", "sae-lens")},
        "prompts": PROMPTS,
        "prompt_format": "chat template with generation prompt; forward pass only, no generated tokens",
        "tokens_measured": resid.shape[0],
        # Gemma Scope 2 SAEs use JumpReLU, so a feature is active exactly when its activation is above 0.
        "mean_active_features_per_token": round((feats > 0).sum(dim=-1).float().mean().item(), 1),
        "variance_explained": round(1 - unexplained.item(), 4),
        "resid_norm": {"median": round(percentile(norms, 0.5), 1), "p10": round(percentile(norms, 0.1), 1),
                       "p90": round(percentile(norms, 0.9), 1)},
        "steering_limit": STEERING_LIMIT,
        "steering_limit_as_fraction_of_median_norm": round(STEERING_LIMIT / percentile(norms, 0.5), 3),
    }
    text = json.dumps(report, ensure_ascii=False, indent=1)
    OUTPUT_PATH.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
