"""Human-readable names for SAE features."""
import json
from pathlib import Path


def load_labels(*paths: Path) -> dict[int, str]:
    """Merge JSON files of {"feature_id": "label"}; later files win. Missing files are skipped,
    so the app still runs before any labels have been written."""
    labels: dict[int, str] = {}
    for path in paths:
        path = Path(path)
        if path.exists():
            labels.update({int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()})
    return labels


def label_for(labels: dict[int, str], feature_id: int) -> str:
    return labels.get(feature_id, f"unlabelled #{feature_id}")
