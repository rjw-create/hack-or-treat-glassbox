import json

from glassbox.labels import label_for, load_labels


def test_later_files_override_earlier_and_missing_files_are_skipped(tmp_path):
    auto = tmp_path / "auto.json"
    curated = tmp_path / "curated.json"
    auto.write_text(json.dumps({"1": "auto one", "2": "auto two"}), encoding="utf-8")
    curated.write_text(json.dumps({"2": "French text"}), encoding="utf-8")

    labels = load_labels(auto, tmp_path / "missing.json", curated)

    assert labels == {1: "auto one", 2: "French text"}


def test_unknown_feature_gets_placeholder():
    assert label_for({}, 241) == "unlabelled #241"
