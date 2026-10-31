import pytest

from glassbox.fake import FakeGlassbox
from ui.app import build_app, diff_rows, feature_rows, reply_tokens, steering_from_inputs, token_spans


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


def test_app_builds_with_fake_backend():
    build_app(FakeGlassbox.load(), fake=True)
