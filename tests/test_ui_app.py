import json

import pytest

from glassbox.fake import FakeGlassbox
from ui.app import (
    STEERING_ROWS,
    TOURS_DIR,
    build_app,
    diff_rows,
    feature_rows,
    load_tours,
    reply_tokens,
    steering_from_inputs,
    token_spans,
    tour_inputs,
)


def test_steering_skips_empty_ids_and_zero_strengths():
    assert steering_from_inputs([101, None, 202], [400, 300, 0]) == {101: 400.0}


def test_steering_rejects_the_same_feature_twice():
    with pytest.raises(ValueError):
        steering_from_inputs([101, 101], [400, 400])


def test_token_spans_cover_reply_tokens_only():
    gb = FakeGlassbox.load()
    result = gb.generate("Describe bread.")
    spans = token_spans(gb, result)
    assert len(spans) == len(reply_tokens(result))
    assert spans[0][1] == "French language (fake)"
    assert spans[-1][1] is None  # the final token has no features


def test_feature_and_diff_rows_carry_labels():
    gb = FakeGlassbox.load()
    token = reply_tokens(gb.generate("Describe bread."))[0]
    assert [row[1] for row in feature_rows(gb, token)] == [gb.label(f) for f, _ in token.features]
    rows = diff_rows(gb.diff("a", "bb"))
    assert rows and all(len(row) == 5 for row in rows)


def test_app_builds_with_fake_backend_and_no_telemetry():
    app = build_app(FakeGlassbox.load(), fake=True)
    assert app.analytics_enabled is False


def test_tour_fills_first_row_and_clears_the_rest(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps({"title": "A", "prompt": "p", "max_new_tokens": 30, "steering": [{"feature_id": 7, "strength": 400}]}),
        encoding="utf-8",
    )
    tour = load_tours(tmp_path)["A"]
    values = tour_inputs(tour)
    assert values[:2] == ["p", 30]
    assert values[2 : 2 + STEERING_ROWS] == [7, None, None]
    assert values[2 + STEERING_ROWS :] == [400.0, 0, 0]


def test_missing_tours_folder_means_no_tours(tmp_path):
    assert load_tours(tmp_path / "nope") == {}


def test_repo_tours_are_valid_and_steer_curated_features():
    tours = load_tours(TOURS_DIR)
    curated = json.loads((TOURS_DIR.parent / "labels" / "curated.json").read_text(encoding="utf-8"))
    assert tours
    for tour in tours.values():
        assert tour["prompt"] and tour["description"]
        for feature_id in tour["steering"]:
            assert str(feature_id) in curated  # every tour feature has a hand-written label
