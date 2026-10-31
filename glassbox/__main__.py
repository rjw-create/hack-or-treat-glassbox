"""Launch the explorer UI: `python -m glassbox [--fake]` or the `glassbox` command.

The UI lives in ui/app.py (owned by Surface) at the repo root, so this works from a clone
installed with `pip install -e ".[ui]"`.
"""
import importlib.util
import runpy
import sys
from pathlib import Path

# Not imported from glassbox.api: that pulls in torch, which is slow just to print a usage error.
APP_PATH = Path(__file__).resolve().parent.parent / "ui" / "app.py"


def main() -> None:
    if importlib.util.find_spec("gradio") is None:
        sys.exit('The Glassbox UI needs Gradio. Install it with: pip install -e ".[ui]"')
    if not APP_PATH.exists():
        sys.exit(f"UI not found at {APP_PATH}. Run Glassbox from a clone of the repo (pip install -e .).")
    sys.argv[0] = str(APP_PATH)  # so the UI's --help shows the right program name; other args pass through
    runpy.run_path(str(APP_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
