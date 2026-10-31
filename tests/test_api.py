import json

import torch

from glassbox.api import load_tour
from glassbox.diff import rank_feature_diffs
from glassbox.fake import FakeGlassbox


def test_rank_feature_diffs_orders_by_absolute_change():
    mean_a = torch.tensor([0.0, 5.0, 1.0, 0.0])
    mean_b = torch.tensor([0.0, 0.0, 3.0, 9.0])
    ranked = rank_feature_diffs(mean_a, mean_b, top_n=3)
    assert [f for f, *_ in ranked] == [3, 1, 2]
    assert ranked[1] == (1, 5.0, 0.0, -5.0)


def test_load_tour_converts_steering_to_dict(tmp_path):
    path = tmp_path / "tour.json"
    path.write_text(json.dumps({"title": "t", "prompt": "p", "steering": [{"feature_id": "7", "strength": 400}]}), encoding="utf-8")
    assert load_tour(path)["steering"] == {7: 400.0}


def test_fake_matches_interface_shapes():
    gb = FakeGlassbox.load()
    result = gb.generate("Describe bread.", steering={101: 400, 202: 0})
    assert result.steering == {101: 400}  # zero-strength sliders are dropped
    assert result.tokens[-1].features == []
    assert all(t.features for t in result.tokens[:-1])
    assert any(t.is_generated for t in result.tokens) and not result.tokens[0].is_generated
    json.dumps(result.to_dict())  # must be JSON-ready for results/
    diffs = gb.diff("a", "bb", top_n=2)
    assert len(diffs) == 2 and abs(diffs[0].delta) >= abs(diffs[1].delta)
