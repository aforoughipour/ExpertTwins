"""Document identifiers.

Machine ids like ``ref_35952419`` are unreadable inside a citation, and a
citation nobody can read is a citation nobody checks. So ``doc_id`` is a
human-readable slug — ``example2012nature`` — and the sha256 of the source file
is the integrity anchor kept alongside it.

Ids are sharded into two-character subdirectories. At the corpus sizes we are
targeting (tens of thousands of documents) a flat directory is painful on
Windows and slow to list anywhere.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from .textnorm import canonical

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-4]\d)\b")

# Particles that are part of a surname but read badly at the front of a slug.
_SURNAME_PARTICLES = {
    "van", "von", "de", "del", "della", "di", "da", "dos", "du", "la", "le",
    "el", "al", "bin", "ibn", "ter", "ten", "op", "den",
}


def _ascii_fold(text: str) -> str:
    """Strip accents so that ``Müller`` and ``Muller`` produce the same slug."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def slugify(text: str) -> str:
    text = _SLUG_STRIP_RE.sub("", _ascii_fold(canonical(text)).lower())
    return text


def surname_of(author: str) -> str:
    """Best-effort surname extraction from a free-form author string.

    Handles the three shapes publishers actually emit:
    ``"Example AA"``, ``"A. A. Example"``, ``"Example, Alex A."``
    """
    author = canonical(author).strip()
    if not author:
        return ""
    if "," in author:
        candidate = author.split(",", 1)[0]
    else:
        parts = author.split()
        if not parts:
            return ""
        # "Example AA" — trailing token is initials (short, no lowercase).
        last = parts[-1]
        if len(parts) > 1 and len(last) <= 3 and last.upper() == last:
            candidate = " ".join(parts[:-1])
        else:
            candidate = last
            # Pull in particles: "van Groningen" not "Groningen".
            i = len(parts) - 1
            while i > 0 and parts[i - 1].lower() in _SURNAME_PARTICLES:
                i -= 1
                candidate = parts[i] + candidate
    return slugify(candidate)


def venue_token(venue: str) -> str:
    """A short, stable token for a journal or venue.

    Deliberately crude: it only needs to disambiguate two papers by the same
    author in the same year, and it needs to be stable across metadata sources
    that disagree about whether the journal is "Nat Genet" or "Nature Genetics".
    """
    v = slugify(venue)
    if not v:
        return "na"
    return v[:12]


def make_doc_id(
    authors: list[str],
    year: int | None,
    venue: str = "",
    fallback: str = "",
) -> str:
    """Build ``<surname><year><venue>``, e.g. ``example2012naturegene``.

    Falls back to a hash-derived id only when there is genuinely nothing to
    build from — and marks it ``anon`` so it is greppable and can be repaired
    later rather than quietly persisting as an unreadable citation.
    """
    surname = surname_of(authors[0]) if authors else ""
    y = str(year) if year else ""
    if surname and y:
        return f"{surname}{y}{venue_token(venue)}"
    if surname:
        return f"{surname}{venue_token(venue)}"
    seed = fallback or venue or "unknown"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
    return f"anon{y}{digest}"


_ID_SAFE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,79}$")


def is_valid_doc_id(doc_id: str) -> bool:
    return bool(_ID_SAFE_RE.match(doc_id))


def disambiguate(doc_id: str, taken: set[str]) -> str:
    """Append ``a``, ``b``, ... until the id is free.

    Collisions are real and normal — the same group can publish twice in a year
    in the same journal. The caller must pass the taken set explicitly so the
    dedupe check cannot silently compare against the wrong collection.
    """
    if doc_id not in taken:
        return doc_id
    for i in range(26):
        candidate = f"{doc_id}{chr(ord('a') + i)}"
        if candidate not in taken:
            return candidate
    i = 2
    while f"{doc_id}v{i}" in taken:
        i += 1
    return f"{doc_id}v{i}"


def shard_of(doc_id: str) -> str:
    """Two-character shard directory for a doc id."""
    cleaned = doc_id if len(doc_id) >= 2 else (doc_id + "__")
    return cleaned[:2]


def doc_dir(library_root: Path, doc_id: str) -> Path:
    return library_root / "docs" / shard_of(doc_id) / doc_id


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_year(text: str) -> int | None:
    """Pull a plausible publication year out of a free-form string."""
    match = _YEAR_RE.search(text or "")
    return int(match.group(1)) if match else None
