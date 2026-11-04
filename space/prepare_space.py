"""Assemble the files for the Hugging Face Space into one folder, ready to upload.

    python space/prepare_space.py <output-folder>

A Space needs its card as README.md and its app at the top of its own repository, so this copies
space/README.md, space/app.py and space/requirements.txt there, plus the code and data the app imports.
Existing files in the output folder are overwritten; nothing is deleted.
"""
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = REPO_ROOT / "space"
# Space path -> source path
FILES = {
    "README.md": SPACE_DIR / "README.md",
    "app.py": SPACE_DIR / "app.py",
    "requirements.txt": SPACE_DIR / "requirements.txt",
    "LICENSE": REPO_ROOT / "LICENSE",
    "ui/app.py": REPO_ROOT / "ui" / "app.py",
}
FOLDERS = ["glassbox", "labels", "tours"]


def prepare(out_dir: Path) -> None:
    out_dir = Path(out_dir).resolve()
    if out_dir == REPO_ROOT or REPO_ROOT in out_dir.parents:
        # Copying space/README.md over the repo's own README would destroy it.
        raise ValueError("Choose an output folder outside this repository.")
    for name, source in FILES.items():
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for folder in FOLDERS:
        shutil.copytree(
            REPO_ROOT / folder, out_dir / folder, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__")
        )


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python space/prepare_space.py <output-folder>")
    try:
        prepare(Path(sys.argv[1]))
    except ValueError as err:
        sys.exit(str(err))
    print(f"Space files written to {Path(sys.argv[1]).resolve()}")


if __name__ == "__main__":
    main()
