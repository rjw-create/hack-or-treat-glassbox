"""Load the language model and tokenizer."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "google/gemma-3-1b-it"


def pick_device() -> str:
    # Intel GPUs show up as "xpu" only with PyTorch's Intel build; the default pip build is CPU-only.
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def load_model(model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
    """Return (tokenizer, model). Float32 on purpose: the SAE was trained on unquantized activations."""
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
    return tokenizer, model


def decoder_layers(model):
    """The list of transformer blocks; layer i's output is SAELens' blocks.{i}.hook_resid_post."""
    return model.model.layers
