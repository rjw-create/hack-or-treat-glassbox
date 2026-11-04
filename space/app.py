"""Hugging Face Space entry point for Glassbox.

prepare_space.py copies this file to the root of the Space, next to glassbox/, ui/, labels/ and tours/,
so both imports below resolve from that folder.

Fake mode is the default: the free CPU tier is too slow for a shared real-model demo (see space/README.md).
Set the Space variable GLASSBOX_REAL_MODEL=1 to load Gemma 3 instead; that also needs the HF_TOKEN secret.
"""
import os
import sys

from glassbox.fake import FakeGlassbox
from ui.app import build_app


def use_real_model() -> bool:
    return os.environ.get("GLASSBOX_REAL_MODEL", "").strip() == "1"


def main() -> None:
    real = use_real_model()
    if real:
        # huggingface_hub reads HF_TOKEN from the environment by itself; only check it is there, never print it.
        if not os.environ.get("HF_TOKEN"):
            sys.exit("GLASSBOX_REAL_MODEL=1 needs a Space secret named HF_TOKEN (Gemma 3 is gated).")
        from glassbox.api import Glassbox

        gb = Glassbox.load()
    else:
        gb = FakeGlassbox.load()
    build_app(gb, fake=not real).launch()


if __name__ == "__main__":
    main()
