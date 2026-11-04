"""Tour files only steer features with a hand-written label. Plain JSON on purpose: no torch, transformers or gradio."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _labelled_feature_ids() -> set[int]:
    ids: set[int] = set()
    for path in (REPO / "labels").glob("*.json"):
        ids |= {int(key) for key in json.loads(path.read_text(encoding="utf-8"))}
    return ids


def _tour_feature_ids(path: Path) -> list[int]:
    tour = json.loads(path.read_text(encoding="utf-8"))
    return [int(row["feature_id"]) for row in tour["steering"]]


def test_every_tour_feature_is_in_labels():
    labelled = _labelled_feature_ids()
    tour_paths = sorted((REPO / "tours").glob("*.json"))
    assert tour_paths
    for path in tour_paths:
        missing = set(_tour_feature_ids(path)) - labelled
        assert not missing, f"{path.name} steers features with no label: {sorted(missing)}"


def test_two_features_tour_steers_two_distinct_features():
    feature_ids = _tour_feature_ids(REPO / "tours" / "two-features.json")
    # The UI refuses a feature that appears in two steering rows, so a duplicate would break the tour.
    assert len(feature_ids) == 2 and len(set(feature_ids)) == 2
