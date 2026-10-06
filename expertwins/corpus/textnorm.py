"""Text normalisation.

Quote verification is character-level after normalisation and *never* fuzzy.
The reason: "effective in all subtypes" and "effective in all subtypes except
one" are highly similar under fuzzy metrics and mean opposite things. A fuzzy
matcher could accept a hallucination that drops the exception clause.

So the matcher must be exact — which means normalisation has to absorb every
typographic difference that a PDF can introduce without changing meaning,
including soft hyphens (``high\\u00adrisk`` vs ``high-risk``).
"""

from __future__ import annotations

import re
import unicodedata

# Ligatures. PDF extractors emit these constantly; no reader considers them
# different words.
_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st", "\u0132": "IJ",
    "\u0133": "ij", "\u0152": "OE", "\u0153": "oe", "\uFB00": "ff",
}

# Every dash-like and hyphen-like codepoint collapses to ASCII hyphen.
_DASHES = dict.fromkeys(
    "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00ad\u2043\uFE58\uFE63\uFF0D",
    "-",
)

# Every quote-like codepoint collapses to its ASCII form.
_QUOTES = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2032": "'", "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
}

# Space-like codepoints that are not ASCII space.
_SPACES = dict.fromkeys(
    "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
    "\u2009\u200a\u202f\u205f\u3000\u200b\u200c\u200d\ufeff",
    " ",
)

_TRANSLATION = str.maketrans({**_LIGATURES, **_DASHES, **_QUOTES, **_SPACES})

_WHITESPACE_RE = re.compile(r"\s+")

# A hyphen followed by a line break is PDF line-wrapping, not a real hyphen.
# "word-\nwrapped" is one word. Applied before whitespace collapsing.
_LINEWRAP_HYPHEN_RE = re.compile(r"(\w)-\s*\n\s*(\w)")


def canonical(text: str) -> str:
    """Normalise text for exact comparison.

    Absorbs typography, never meaning. Specifically it does NOT lowercase,
    strip punctuation, stem, or drop stopwords — all of which can invert a
    scientific claim. ``canonical`` is idempotent.
    """
    if not text:
        return ""
    # NFKC first: folds compatibility forms (superscripts, some ligatures,
    # full-width Latin) that publishers use inconsistently.
    text = unicodedata.normalize("NFKC", text)
    text = _LINEWRAP_HYPHEN_RE.sub(r"\1\2", text)
    text = text.translate(_TRANSLATION)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def canonical_fold(text: str) -> str:
    """``canonical`` plus case folding.

    For *indexing and search only*, never for quote verification. Search wants
    to match "TP53" against "Tp53"; a quote check must not.
    """
    return canonical(text).casefold()


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-'/.]*")


def words(text: str) -> list[str]:
    """Token list used for length floors on quotes and passages."""
    return _WORD_RE.findall(canonical(text))
