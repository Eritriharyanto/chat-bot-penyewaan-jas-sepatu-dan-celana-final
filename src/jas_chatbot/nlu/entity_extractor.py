# entity_extractor.py

"""Entity extraction for the jas-rental chatbot.

This mirrors the role of the original YOGA-Chatbot ``EntityExtractor``
(which matched Yogyakarta ``kecamatan``/``kabupaten`` names), but is
adapted to this store's domain: the two entity types that actually show
up across the 20 intents are UKURAN (size) and WARNA (color).

Both entity lists are loaded from ``data/knowledge/entities_katalog.json``,
which is derived from ``knowledge_base.json`` (see notebook cell that
builds it) so it always matches what's actually in stock.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


class EntityExtractor:
    def __init__(self, entities_path: Path | str):
        with open(entities_path, encoding="utf-8") as f:
            data = json.load(f)

        # Longest-first so multi-word colors ("biru dongker") match before
        # a shorter substring color ("biru muda" vs "biru") would.
        self.warna = sorted(data["warna"], key=len, reverse=True)
        self.ukuran = sorted(data["ukuran"], key=len, reverse=True)

        warna_pat = "|".join(re.escape(w.lower()) for w in self.warna)
        # numeric sizes need word boundaries; alpha sizes (XS..5XL) too
        ukuran_pat = "|".join(re.escape(u.lower()) for u in self.ukuran)

        self._warna_re = re.compile(rf"\b({warna_pat})\b", re.IGNORECASE) if self.warna else None
        self._ukuran_re = re.compile(rf"\b({ukuran_pat})\b", re.IGNORECASE) if self.ukuran else None

    def find(self, text: str) -> dict:
        """Return SEMUA entity yang match sekaligus:
        {'warna': <value|None>, 'ukuran': <value|None>}"""
        t = text.lower()
        result = {"warna": None, "ukuran": None}
        if self._warna_re:
            m = self._warna_re.search(t)
            if m:
                result["warna"] = m.group(1)
        if self._ukuran_re:
            m = self._ukuran_re.search(t)
            if m:
                result["ukuran"] = m.group(1)
        return result
        
    def replace_with_placeholder(self, text: str) -> str:
        """Runtime-parity transform: neutralise the entity value so the
        classifier generalises across specific sizes/colors instead of
        memorising them (e.g. 'ada jas ukuran L?' and 'ada jas ukuran
        XXL?' should hit the same intent)."""
        out = text
        if self._warna_re:
            out = self._warna_re.sub("[WARNA]", out)
        if self._ukuran_re:
            out = self._ukuran_re.sub("[UKURAN]", out)
        return out
