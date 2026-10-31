"""Forward hook that reads SAE features from one layer and optionally steers the model."""
import torch


class FeatureHook:
    """Attach to one transformer block. Records the top-k SAE features per token position and,
    when steering is set, adds strength * decoder_direction to that block's output.

    Assumes batch size 1. Steering strength is in the same units as feature activations,
    because Gemma Scope decoder directions have unit length.
    """

    def __init__(self, sae, top_k: int = 5):
        self.sae = sae
        self.top_k = top_k
        self.steering: dict[int, float] = {}
        self.records: list[list[tuple[int, float]]] = []
        self.last_features: torch.Tensor | None = None  # full [positions, d_sae] from the latest forward pass

    def set_steering(self, steering: dict[int, float]) -> None:
        self.steering = {f: s for f, s in steering.items() if s != 0}

    def clear_records(self) -> None:
        self.records = []

    def attach(self, layer):
        """Register on a block; returns a handle whose .remove() detaches it."""
        return layer.register_forward_hook(self)

    def steering_vector(self) -> torch.Tensor | None:
        if not self.steering:
            return None
        return sum(strength * self.sae.W_dec[f] for f, strength in self.steering.items())

    def __call__(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            # Records show what the model computed before our nudge, so a steered feature
            # does not appear to "fire" just because we injected it.
            feats = self.sae.encode(hidden[0].to(self.sae.W_dec.dtype))
            self.last_features = feats
            values, ids = feats.topk(self.top_k, dim=-1)
            for pos_ids, pos_vals in zip(ids.tolist(), values.tolist()):
                self.records.append([(f, v) for f, v in zip(pos_ids, pos_vals) if v > 0])

        vec = self.steering_vector()
        if vec is None:
            return None
        steered = hidden + vec.to(hidden.dtype)
        return (steered, *output[1:]) if isinstance(output, tuple) else steered
