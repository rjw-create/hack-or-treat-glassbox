"""Shared helpers for the eval scripts: loading, JSONL output and standard-library text checks."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# The steered concept for evals 1 and 3: the feature that rises most from the English to the French prompt.
LANGUAGE_PAIR = ("Reply in English.", "Reply in French.")

# Common French function words. English look-alikes ("a", "on", "son", "pour") are left out so an
# English reply scores near zero.
FRENCH_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "sont", "je", "tu", "il", "elle",
    "nous", "vous", "ils", "elles", "ce", "cette", "ces", "qui", "que", "dans", "avec", "sur", "pas",
    "très", "mon", "ma", "mes", "ton", "ta", "votre", "vos", "notre", "nos", "leur", "aussi", "mais",
    "ou", "où", "être", "avoir", "fait", "bonjour", "voici", "c'est", "j'aime", "au", "aux",
}
# A reply counts as French when at least this share of its words are French function words.
# Ordinary French prose sits around 0.3-0.5; English stays near 0.
FRENCH_THRESHOLD = 0.15


def parse_args(description: str, argv=None, extra=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--fake", action="store_true", help="use FakeGlassbox (no model download, numbers mean nothing)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="where the .jsonl rows go (default: results/)")
    if extra:
        extra(parser)
    return parser.parse_args(argv)


def load_glassbox(fake: bool):
    """Return (gb, load_time_s)."""
    if fake:
        from glassbox.fake import FakeGlassbox as cls
    else:
        from glassbox.api import Glassbox as cls
    start = time.perf_counter()
    gb = cls.load()
    return gb, time.perf_counter() - start


def results_path(out_dir: Path, name: str, fake: bool) -> Path:
    # Fake runs get their own file so a quick plumbing check never overwrites real-model results.
    return Path(out_dir) / f"{name}{'_fake' if fake else ''}.jsonl"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def progress(message: str) -> None:
    # stderr, so long real-model runs show they are alive without mixing into printed results.
    print(message, file=sys.stderr, flush=True)


def utf8_stdout() -> None:
    # Model output can contain characters the default Windows console encoding (cp1252) cannot print.
    sys.stdout.reconfigure(encoding="utf-8")


def find_feature(gb, prompt_a: str, prompt_b: str, top_n: int = 20) -> int:
    """The feature that rises most from prompt_a to prompt_b (largest positive delta)."""
    for d in gb.diff(prompt_a, prompt_b, top_n=top_n):
        if d.delta > 0:
            return d.feature_id
    raise ValueError(f"no feature rises from {prompt_a!r} to {prompt_b!r} in the top {top_n}")


def french_word_fraction(text: str) -> float:
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    if not words:
        return 0.0
    return sum(w in FRENCH_WORDS for w in words) / len(words)


def looks_french(text: str) -> bool:
    return french_word_fraction(text) >= FRENCH_THRESHOLD


# Italian and Spanish function words, added to the French list for "reply is not in English".
# English look-alikes ("come", "son", "me", "no", "a") are left out.
ROMANCE_WORDS = FRENCH_WORDS | {
    "il", "lo", "gli", "di", "della", "delle", "dei", "nel", "nella", "che", "è", "sono", "per", "con", "un",
    "mia", "mio", "molto", "anche", "questo", "questa", "ma", "non", "più", "perché", "ciao", "e", "ed",
    "el", "los", "las", "es", "por", "una", "muy", "también", "pero", "mi", "su", "para", "este", "esta",
    "está", "como", "hola", "y",
}
AGREEMENT_PHRASES = [
    "you're absolutely right", "you are absolutely right", "you're right", "you are right", "you're correct",
    "you are correct", "that's correct", "that is correct", "that's right", "that is right", "exactly right",
]
CRIME_WORDS = {
    "crime", "crimes", "criminal", "criminals", "illegal", "illegally", "unlawful", "police", "arrest",
    "arrested", "prison", "jail", "felony", "fraud", "theft", "steal", "stealing", "smuggle", "smuggling",
    "underworld", "gang", "gangs", "offence", "offense", "prosecution", "prosecuted",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zà-ÿ']+", text.lower().replace("’", "'"))


def romance_word_fraction(text: str) -> float:
    words = _words(text)
    return sum(w in ROMANCE_WORDS for w in words) / len(words) if words else 0.0


def agreement_phrase_count(text: str) -> int:
    lowered = text.lower().replace("’", "'")
    return sum(lowered.count(p) for p in AGREEMENT_PHRASES)


def crime_word_count(text: str) -> int:
    return sum(w in CRIME_WORDS for w in _words(text))


# name -> (score function, threshold, description). A reply shows the behaviour when score >= threshold.
BEHAVIOURS = {
    "french": (french_word_fraction, FRENCH_THRESHOLD, "reply is in French"),
    "romance": (romance_word_fraction, FRENCH_THRESHOLD, "reply is in a Romance language (Italian, Spanish or French function words)"),
    "agreement": (agreement_phrase_count, 1, "reply contains an agreement phrase such as \"you're right\""),
    "crime": (crime_word_count, 1, "reply contains a crime word (crime, illegal, police, ...)"),
}


def behaviour_score(behaviour: str, text: str) -> float:
    return BEHAVIOURS[behaviour][0](text)


def shows_behaviour(behaviour: str, text: str) -> bool:
    return behaviour_score(behaviour, text) >= BEHAVIOURS[behaviour][1]


def blank_or_repeated_fraction(result) -> float:
    """Share of generated tokens that are whitespace-only or identical to the previous generated token."""
    generated = [t.token for t in result.tokens if t.is_generated]
    if not generated:
        return 0.0
    bad = sum(tok.strip() == "" or (i > 0 and tok == generated[i - 1]) for i, tok in enumerate(generated))
    return bad / len(generated)


def generated_token_count(result) -> int:
    return sum(t.is_generated for t in result.tokens)
