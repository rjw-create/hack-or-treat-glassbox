import pytest
import torch

from glassbox.hooks import FeatureHook


class FakeSAE:
    """Identity-like SAE: feature i reads hidden dimension i, decoder rows are unit vectors."""

    def __init__(self, d_model: int = 4):
        self.W_dec = torch.eye(d_model)

    def encode(self, x):
        return torch.relu(x)


class TupleLayer(torch.nn.Module):
    def forward(self, x):
        return (x, "extra")


class TensorLayer(torch.nn.Module):
    def forward(self, x):
        return x


def test_records_top_k_features_per_position():
    hook = FeatureHook(FakeSAE(), top_k=2)
    layer = TensorLayer()
    hook.attach(layer)
    layer(torch.tensor([[[0.0, 3.0, 1.0, 2.0], [5.0, 0.0, 0.0, 0.0]]]))
    assert hook.records == [[(1, 3.0), (3, 2.0)], [(0, 5.0)]]  # zero activations are dropped


def test_steering_adds_exactly_strength_times_direction():
    hook = FeatureHook(FakeSAE(), top_k=1)
    hook.set_steering({2: 1.5, 0: -0.5})
    layer = TupleLayer()
    hook.attach(layer)
    x = torch.zeros(1, 3, 4)
    out, extra = layer(x)
    assert extra == "extra"
    assert torch.allclose(out, x + torch.tensor([-0.5, 0.0, 1.5, 0.0]))


def test_no_steering_leaves_output_unchanged():
    hook = FeatureHook(FakeSAE(), top_k=1)
    hook.set_steering({1: 0.0})  # a slider at zero means "off"
    layer = TensorLayer()
    hook.attach(layer)
    x = torch.randn(1, 2, 4)
    assert torch.equal(layer(x), x)


def test_out_of_range_feature_id_raises_clear_error():
    hook = FeatureHook(FakeSAE(), top_k=1)  # 4 features: ids 0-3
    with pytest.raises(ValueError, match=r"\[4, 99\] out of range"):
        hook.set_steering({99: 400.0, 4: 1.0, 2: 1.0})
    assert hook.steering == {}  # nothing half-applied


def test_records_are_taken_before_steering():
    hook = FeatureHook(FakeSAE(), top_k=1)
    hook.set_steering({3: 10.0})
    layer = TensorLayer()
    hook.attach(layer)
    layer(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))
    assert hook.records == [[(0, 1.0)]]
