"""Data augmentation for the jas-rental intent dataset.

Re-implementation of the original YOGA-Chatbot ``augment_pattern`` (that
script wasn't included in the files you uploaded, so this isn't a byte
-for-byte port -- it's a from-scratch version that follows the same
idea: apply a few cheap, meaning-preserving transforms per pattern so
the classifier sees more surface variation without changing what the
sentence means).

Transforms applied (each independently, so one call can return several
variants of the same sentence):
  1. the original pattern, unchanged
  2. lowercase + stripped trailing punctuation ("?", "!", ".", "..")
  3. a polite-filler variant ("min"/"kak"/"dong" prepended or appended)
     -- skipped for the `tidak_dikenali` (out-of-scope) tag, since
     dressing up gibberish with "min" doesn't reflect how people
     actually type nonsense/off-topic messages
  4. a light single-character typo (drop or duplicate one letter in a
     longer word) to add robustness to typos

All randomness is driven off the ``random`` module seeded once in the
notebook (``random.seed(42)``), so results are reproducible.
"""
from __future__ import annotations

import random
import re

_TRAILING_PUNCT_RE = re.compile(r"[?!.]+$")
_FILLERS = ["min", "kak", "dong", "ya"]


def _strip_trailing_punct(text: str) -> str:
    return _TRAILING_PUNCT_RE.sub("", text).strip()


def _add_filler(text: str) -> str:
    filler = random.choice(_FILLERS)
    if random.random() < 0.5:
        return f"{filler} {text}"
    return f"{text} {filler}"


def _typo(text: str) -> str:
    words = text.split(" ")
    # pick a word long enough that a 1-char edit won't destroy it
    candidates = [i for i, w in enumerate(words) if len(w) >= 5]
    if not candidates:
        return text
    i = random.choice(candidates)
    w = words[i]
    pos = random.randint(1, len(w) - 2)
    if random.random() < 0.5:
        # drop a character
        w = w[:pos] + w[pos + 1:]
    else:
        # duplicate a character
        w = w[:pos] + w[pos] + w[pos:]
    words[i] = w
    return " ".join(words)


def augment_pattern(pattern: str, label: str) -> list[str]:
    variants = {pattern}

    stripped = _strip_trailing_punct(pattern.lower())
    if stripped and stripped != pattern:
        variants.add(stripped)

    if label != "tidak_dikenali":
        variants.add(_add_filler(pattern))

    typo_variant = _typo(pattern)
    if typo_variant != pattern:
        variants.add(typo_variant)

    return list(variants)
