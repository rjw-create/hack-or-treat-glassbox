"""Deterministic stand-in for Glassbox, for UI work and eval plumbing without loading the model.
Same methods and return types as glassbox.api.Glassbox; the numbers mean nothing."""
from glassbox.api import FeatureDiff, GenerationResult, TokenFeatures
from glassbox.labels import label_for

FAKE_LABELS = {101: "French language (fake)", 202: "food (fake)", 303: "formal tone (fake)"}
N_FEATURES = 16384  # same width as the real SAE, so out-of-range ids fail the same way in fake mode


def _fake_features(token: str) -> list[tuple[int, float]]:
    # Stable across runs (unlike hash()), so screenshots and tests repeat exactly.
    seed = sum(ord(c) for c in token)
    return [(101, 300.0 + seed % 200), (202, 150.0 + seed % 100), (303, 50.0 + seed % 50)]


class FakeGlassbox:
    def __init__(self):
        self.labels = dict(FAKE_LABELS)

    @classmethod
    def load(cls, device: str | None = None) -> "FakeGlassbox":
        return cls()

    def reload_labels(self) -> None:
        pass

    def label(self, feature_id: int) -> str:
        return label_for(self.labels, feature_id)

    def generate(self, prompt: str, max_new_tokens: int = 60, steering: dict[int, float] | None = None) -> GenerationResult:
        bad = sorted(f for f in (steering or {}) if not 0 <= f < N_FEATURES)
        if bad:
            raise ValueError(f"Feature id(s) {bad} out of range: this SAE has features 0 to {N_FEATURES - 1}.")
        steering = {f: s for f, s in (steering or {}).items() if s != 0}
        reply = "Bonjour, voici une réponse." if steering.get(101, 0) > 0 else "Here is a short reply."
        prompt_tokens = ["<bos>", "<start_of_turn>", "user", "\n", *(" " + w for w in prompt.split()), "<end_of_turn>", "\n", "<start_of_turn>", "model", "\n"]
        reply_tokens = [" " + w for w in reply.split()][: max(max_new_tokens, 1)]
        all_tokens = prompt_tokens + reply_tokens
        tokens = [
            TokenFeatures(i, t, i >= len(prompt_tokens), [] if i == len(all_tokens) - 1 else _fake_features(t))
            for i, t in enumerate(all_tokens)
        ]
        return GenerationResult(prompt, "".join(reply_tokens).strip(), tokens, steering)

    def diff(self, prompt_a: str, prompt_b: str, top_n: int = 20) -> list[FeatureDiff]:
        a, b = len(prompt_a) % 100, len(prompt_b) % 100
        rows = [FeatureDiff(f, self.label(f), float(a), float(b + i * 10), float(b + i * 10 - a)) for i, f in enumerate(FAKE_LABELS)]
        return sorted(rows, key=lambda r: abs(r.delta), reverse=True)[:top_n]
