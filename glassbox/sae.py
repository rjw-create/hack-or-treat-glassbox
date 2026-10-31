"""Load a Gemma Scope 2 sparse autoencoder through SAELens."""
from sae_lens import SAE

DEFAULT_SAE_RELEASE = "gemma-scope-2-1b-it-res"
DEFAULT_SAE_ID = "layer_13_width_16k_l0_medium"
DEFAULT_LAYER = 13  # must match the layer in DEFAULT_SAE_ID


def load_sae(release: str = DEFAULT_SAE_RELEASE, sae_id: str = DEFAULT_SAE_ID, device: str = "cpu"):
    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    # Older SAELens versions return (sae, cfg, sparsity); newer ones return the SAE alone.
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    return sae.eval()
