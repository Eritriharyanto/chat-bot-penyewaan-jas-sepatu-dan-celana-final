"""Text normalisation used identically at train time and serve time.

Tries to use Sastrawi (the standard Indonesian stemmer) if it's
installed. If it isn't available in this environment, falls back to a
small rule-based suffix stripper so the notebook still runs end to end.

If you run this in an environment with internet access, install the
real stemmer for better accuracy:

    pip install Sastrawi
"""
from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s\[\]]")
_MULTISPACE_RE = re.compile(r"\s+")

# Common Indonesian derivational/inflectional suffixes & prefixes used by
# the fallback stemmer. Not linguistically complete -- it exists so the
# pipeline is runnable without network access. Swap in Sastrawi's
# StemmerFactory when available (see _try_load_sastrawi below).
_SUFFIXES = ("nya", "kah", "lah", "kan", "an", "i")
_PREFIXES = ("meng", "peng", "meny", "peny", "men", "pen", "mem", "pem",
             "ber", "ter", "per", "di", "ke", "se")

# Domain nouns the naive affix-stripper would otherwise mangle
# (e.g. "sepatu" -> "patu" by mistaking the "se" prefix). Sastrawi's real
# dictionary-based stemmer doesn't have this problem; this exception list
# only matters for the no-internet fallback path.
_STEM_EXCEPTIONS = {
    "jas", "celana", "sepatu", "ukuran", "warna", "sewa", "harga",
    "denda", "promo", "paket", "admin", "toko", "alamat", "lokasi",
    "bayar", "antar", "reschedule", "batal", "durasi",
    "selamat", "pagi", "siang", "sore", "malam", "terima", "kasih",
}


def _try_load_sastrawi():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        return StemmerFactory().create_stemmer()
    except ImportError:
        return None


def _light_stem(word: str) -> str:
    if len(word) <= 4 or word in _STEM_EXCEPTIONS:
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    for pre in _PREFIXES:
        if word.startswith(pre) and len(word) - len(pre) >= 3:
            word = word[len(pre):]
            break
    return word


class TextProcessor:
    def __init__(self):
        self._sastrawi = _try_load_sastrawi()

    def preprocess(self, text: str) -> str:
        t = text.lower().strip()
        t = _PUNCT_RE.sub(" ", t)
        t = _MULTISPACE_RE.sub(" ", t).strip()

        if self._sastrawi is not None:
            # Preserve placeholder tokens like [UKURAN]/[WARNA] verbatim;
            # stem everything else.
            tokens = []
            for tok in t.split(" "):
                if tok.startswith("[") and tok.endswith("]"):
                    tokens.append(tok)
                else:
                    tokens.append(self._sastrawi.stem(tok))
            return " ".join(tokens)

        tokens = [
            tok if (tok.startswith("[") and tok.endswith("]")) else _light_stem(tok)
            for tok in t.split(" ")
        ]
        return " ".join(tokens)
