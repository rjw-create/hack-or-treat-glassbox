"""High-level API used by the UI and the evals. Contract: docs/2026-10-31-interface.md."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from glassbox.diff import rank_feature_diffs
from glassbox.hooks import FeatureHook
from glassbox.labels import label_for, load_labels
from glassbox.model import decoder_layers, load_model
from glassbox.sae import DEFAULT_LAYER, load_sae

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = REPO_ROOT / "labels"
TOP_K = 5
# In feature-activation units. Layer 13's residual norm has median ~6740, so 1200 is ~18% of it.
# Across 7 swept features, useful effects appeared at 400-1200 and most degenerated beyond that.
STEERING_LIMIT = 1200.0


@dataclass
class TokenFeatures:
    position: int
    token: str
    is_generated: bool
    features: list[tuple[int, float]]


@dataclass
class GenerationResult:
    prompt: str
    text: str
    tokens: list[TokenFeatures]
    steering: dict[int, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureDiff:
    feature_id: int
    label: str
    mean_a: float
    mean_b: float
    delta: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_tour(path) -> dict:
    tour = json.loads(Path(path).read_text(encoding="utf-8"))
    tour["steering"] = {int(s["feature_id"]): float(s["strength"]) for s in tour.get("steering", [])}
    return tour


class Glassbox:
    def __init__(self, tokenizer, model, sae, layer: int = DEFAULT_LAYER, labels_dir: Path = LABELS_DIR):
        self.tokenizer = tokenizer
        self.model = model
        self.labels_dir = Path(labels_dir)
        self.hook = FeatureHook(sae, top_k=TOP_K)
        self._handle = self.hook.attach(decoder_layers(model)[layer])
        self.reload_labels()

    @classmethod
    def load(cls, device: str | None = None) -> "Glassbox":
        tokenizer, model = load_model(device=device)
        return cls(tokenizer, model, load_sae(device=str(model.device)))

    def reload_labels(self) -> None:
        self.labels = load_labels(self.labels_dir / "curated.json")

    def label(self, feature_id: int) -> str:
        return label_for(self.labels, feature_id)

    def _chat_inputs(self, prompt: str):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

    def generate(self, prompt: str, max_new_tokens: int = 60, steering: dict[int, float] | None = None) -> GenerationResult:
        active = {f: s for f, s in (steering or {}).items() if s != 0}
        inputs = self._chat_inputs(prompt)
        n_prompt = inputs["input_ids"].shape[1]
        self.hook.clear_records()
        self.hook.set_steering(active)
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.hook.set_steering({})  # never leave a steering vector attached for the next call

        ids = out[0].tolist()
        records = self.hook.records
        # With the KV cache, the first pass covers every prompt position and each later pass one new
        # token, so records[i] belongs to position i. The final token is never fed back, so it has none.
        tokens = [
            TokenFeatures(pos, self.tokenizer.decode([tid]), pos >= n_prompt, records[pos] if pos < len(records) else [])
            for pos, tid in enumerate(ids)
        ]
        text = self.tokenizer.decode(ids[n_prompt:], skip_special_tokens=True)
        return GenerationResult(prompt, text, tokens, active)

    def _mean_features(self, prompt: str) -> torch.Tensor:
        self.hook.set_steering({})
        with torch.inference_mode():
            self.model(**self._chat_inputs(prompt))
        # Position 0 (<bos>) has activations far larger than any other token and would dominate the mean.
        return self.hook.last_features[1:].mean(dim=0)

    def diff(self, prompt_a: str, prompt_b: str, top_n: int = 20) -> list[FeatureDiff]:
        ranked = rank_feature_diffs(self._mean_features(prompt_a), self._mean_features(prompt_b), top_n)
        return [FeatureDiff(f, self.label(f), a, b, d) for f, a, b, d in ranked]
