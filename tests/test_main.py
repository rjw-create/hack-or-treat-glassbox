import importlib.util
import sys

import pytest

import glassbox.__main__ as launcher


def test_runs_ui_script_with_arguments_passed_through(tmp_path, monkeypatch):
    app = tmp_path / "app.py"
    app.write_text("import sys, pathlib\npathlib.Path(sys.argv[0]).with_name('ran.txt').write_text(' '.join(sys.argv[1:]))\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "APP_PATH", app)
    monkeypatch.setattr(sys, "argv", ["glassbox", "--fake"])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    launcher.main()

    assert (tmp_path / "ran.txt").read_text() == "--fake"


def test_missing_gradio_gives_install_hint(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert '.[ui]' in str(exc.value)


def test_missing_ui_file_says_run_from_clone(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "APP_PATH", tmp_path / "nope.py")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert "clone" in str(exc.value)
