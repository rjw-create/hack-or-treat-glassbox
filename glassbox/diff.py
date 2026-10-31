"""Compare which features two prompts activate."""
import torch


def rank_feature_diffs(mean_a: torch.Tensor, mean_b: torch.Tensor, top_n: int = 20) -> list[tuple[int, float, float, float]]:
    """Return (feature_id, mean_a, mean_b, delta=mean_b-mean_a) for the top_n largest |delta|, largest first."""
    delta = mean_b - mean_a
    top = delta.abs().topk(min(top_n, delta.numel())).indices.tolist()
    return [(f, mean_a[f].item(), mean_b[f].item(), delta[f].item()) for f in top]
