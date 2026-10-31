# Glassbox interface (owned by Core)

The contract between Core (`glassbox/`), Surface (UI, tours, labels) and Proof (evals).
Only Core edits this file; any change is called out in the PR description.
(`2026-10-31-ROLES.md` calls this `docs/interface.md`; the folder's file-naming hook requires the date prefix.)
Status: **v1, 2026-10-31.** `FakeGlassbox` is available now; the real `Glassbox` lands in Core's next PR.

## Import
```python
from glassbox.api import Glassbox, GenerationResult, TokenFeatures, FeatureDiff, STEERING_LIMIT, load_tour
from glassbox.fake import FakeGlassbox   # same methods, no model download, deterministic output
```

## `Glassbox`
| Call | Returns | Notes |
| --- | --- | --- |
| `Glassbox.load(device=None)` | `Glassbox` | Gemma 3 1B-it + Gemma Scope 2 SAE `layer_13_width_16k_l0_medium`. ~35 s on CPU after the first download. Load once and reuse. |
| `gb.generate(prompt, max_new_tokens=60, steering=None)` | `GenerationResult` | Greedy decoding (deterministic). `steering` is `{feature_id: strength}`; `None` or `{}` means unsteered. ~10 tokens/s on this machine's CPU. |
| `gb.diff(prompt_a, prompt_b, top_n=20)` | `list[FeatureDiff]` | Prompts only, no generation. Sorted by `abs(delta)`, largest first. |
| `gb.label(feature_id)` | `str` | Curated label, else `"unlabelled #<id>"`. |
| `gb.reload_labels()` | `None` | Re-reads `labels/*.json` without restarting. |

The prompt is wrapped in Gemma's chat template as a single user turn.

## Data shapes
```python
@dataclass
class TokenFeatures:
    position: int            # index in the full token sequence (prompt + reply)
    token: str               # decoded text of this one token, e.g. " Paris", "\n", "<start_of_turn>"
    is_generated: bool       # False for prompt tokens (including chat-template tokens)
    features: list[tuple[int, float]]   # up to 5 (feature_id, activation), highest first; activation > 0

@dataclass
class GenerationResult:
    prompt: str
    text: str                # the reply only, special tokens removed
    tokens: list[TokenFeatures]
    steering: dict[int, float]

@dataclass
class FeatureDiff:
    feature_id: int
    label: str
    mean_a: float            # mean activation over prompt_a's tokens (first token <bos> excluded)
    mean_b: float
    delta: float             # mean_b - mean_a
```
`GenerationResult.to_dict()` and `FeatureDiff.to_dict()` return plain JSON-ready dicts (for `results/`).

Features are recorded **before** steering is added, so a steered feature does not light up just because
we injected it. The last generated token has no features (the model never reads its own final token),
so its `features` list is empty.

## Steering strength
Units are feature-activation units (decoder directions have length 1). Typical activations on content
words are 200-500. The layer-13 residual stream has median norm ~6740 (p10 5300, p90 8200), so a
strength of 1200 adds a vector about 18% as long as the activations it nudges.
UI sliders: use `STEERING_LIMIT` (**1200**, changed from the provisional 800) as the +/- maximum, step 50.

Measured (greedy, 30-40 tokens; Core's tour search plus Proof's `results/steering.jsonl`):
| Feature | Effect | Fluent up to | Degenerate by |
| --- | --- | --- | --- |
| 6855 | Romance-language text (Italian-like, not French) | 400 | 1200 |
| 544 | "You're absolutely right" agreement | 1600 | not reached |
| 1235 | crime / illegality | 1200 | not reached at 1600 |
| 11207 | no visible effect (top diff for "Reply in French") | 800 | 2100 |

A feature that ranks high in `diff` does not always steer (11207). Check by steering before using it in a tour.

## Files Surface owns
`labels/curated.json`: hand-checked labels, `{"<feature_id>": "short label"}`.
`tours/<name>.json`:
```json
{
  "title": "Language switching",
  "description": "One sentence shown above the tour.",
  "prompt": "Describe your favourite food.",
  "max_new_tokens": 60,
  "steering": [{"feature_id": 1234, "strength": 400}]
}
```
`load_tour(path)` returns this as a dict, with `steering` converted to `{feature_id: strength}`.
Paths are relative to the repo root, so run the app from a clone (`pip install -e .`).

## Not in v1
Multi-turn chat, sampling (temperature), more than one layer, batching.
