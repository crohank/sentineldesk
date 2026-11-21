"""Language detection from raw text (D3-02).

Never trust the dataset's `language` column: ~26% of `de` rows have English bodies.
"""
from __future__ import annotations

import re

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # langdetect is random without a seed

SUPPORTED = ("en", "de")
UNDETERMINED = "und"
MIN_WORDS = 3
MIN_PROB = 0.70

_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
_TOKEN = re.compile(r"<[^<>]{1,30}>|\[[^\[\]]{1,30}\]")  # mask tokens like <tel_num>, [Name]


def detect_language(text: str | None) -> tuple[str, float]:
    """Return (language code or 'und', probability of that language)."""
    if not text:
        return UNDETERMINED, 0.0
    clean = _TOKEN.sub(" ", text)
    if len(_WORD.findall(clean)) < MIN_WORDS:
        return UNDETERMINED, 0.0
    try:
        best = detect_langs(clean)[0]
    except LangDetectException:
        return UNDETERMINED, 0.0
    if best.prob < MIN_PROB:
        return UNDETERMINED, round(best.prob, 3)
    return best.lang, round(best.prob, 3)


def is_supported(lang: str) -> bool:
    return lang in SUPPORTED
