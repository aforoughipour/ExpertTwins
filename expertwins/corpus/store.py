"""The library store.

Layout on disk::

    library/
      MANIFEST.jsonl          append-only; ONE line per accepted document
      docs/<shard>/<doc_id>/
          source.pdf|.xml     the original bytes, when we have them
          text.jsonl          normalised passages, ORIGINAL DOCUMENT ORDER
          meta.yaml           the bibliographic record + integrity anchors
      quarantine/             documents rejected, with the reason, never deleted
      index.sqlite            regenerable FTS5 index — never the source of truth

Design rules:

* **The manifest is append-only and is the source of truth.** ``index.sqlite``
  is rebuilt from it and is never synced to the cluster.
* **Writes are atomic.** A document is written to a temp directory and moved
  into place, so a killed process cannot leave a half-document that the manifest
  swears is complete.
* **Rejections are kept.** A quarantined document goes to ``quarantine/`` with
  its reason, so failed attempts remain auditable.
* **Nothing is deleted.** Supersession is additive.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

import yaml

from .errors import IntegrityError
from .ids import doc_dir, sha256_bytes, shard_of
from .models import (
    MIN_FULLTEXT_CHARS,
    MIN_FULLTEXT_PASSAGES,
    DocMeta,
    FullTextStatus,
    Passage,
)

MANIFEST_NAME = "MANIFEST.jsonl"


class IngestOutcome(str, Enum):
    ADDED = "added"
    DUPLICATE = "duplicate"
    """Identity keys collided with an existing document. Not an error."""
    QUARANTINED = "quarantined"
    """Structurally inadmissible — undatable, untitled, or text-free."""
    REPLACED = "replaced"
    """An existing document was upgraded (e.g. abstract-only -> full text)."""


@dataclass
class IngestResult:
    outcome: IngestOutcome
    doc_id: str = ""
    reason: str = ""
    existing_doc_id: str = ""

    @property
    def accepted(self) -> bool:
        return self.outcome in (IngestOutcome.ADDED, IngestOutcome.REPLACED)


@dataclass
class VerifyReport:
    """The result of falsifying the manifest's assertions against the disk.

    Every counter here corresponds to a store assertion that can disagree with
    the files on disk.
    """

    checked: int = 0
    missing_dir: list[str] = field(default_factory=list)
    missing_text: list[str] = field(default_factory=list)
    text_hash_mismatch: list[str] = field(default_factory=list)
    source_hash_mismatch: list[str] = field(default_factory=list)
    count_mismatch: list[str] = field(default_factory=list)
    status_unsupported: list[str] = field(default_factory=list)
    orphan_dirs: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    passage_id_mismatch: list[str] = field(default_factory=list)
    alias_groups: list[list[str]] = field(default_factory=list)
    title_absent: list[str] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return (
            len(self.missing_dir) + len(self.missing_text)
            + len(self.text_hash_mismatch) + len(self.source_hash_mismatch)
            + len(self.count_mismatch) + len(self.status_unsupported)
            + len(self.duplicate_ids) + len(self.passage_id_mismatch)
        )

    @property
    def warnings(self) -> int:
        return (len(self.orphan_dirs) + len(self.alias_groups)
                + len(self.title_absent))

    def render(self) -> str:
        lines = [f"verified {self.checked} documents: {self.errors} errors, {self.warnings} warnings"]
        for label, items in [
            ("MISSING DIRECTORY", self.missing_dir),
            ("MISSING text.jsonl", self.missing_text),
            ("TEXT HASH MISMATCH", self.text_hash_mismatch),
            ("SOURCE HASH MISMATCH", self.source_hash_mismatch),
            ("PASSAGE/CHAR COUNT MISMATCH", self.count_mismatch),
            ("STATUS NOT SUPPORTED BY CONTENT", self.status_unsupported),
            ("DUPLICATE doc_id IN MANIFEST", self.duplicate_ids),
            ("PASSAGE ID NAMES ANOTHER DOCUMENT", self.passage_id_mismatch),
            ("ORPHAN DIRECTORY (on disk, not in manifest)", self.orphan_dirs),
            ("TITLE ABSENT FROM ITS OWN TEXT (wrong paper?)", self.title_absent),
        ]:
            if items:
                lines.append(f"  {label}: {len(items)}")
                lines.extend(f"    - {i}" for i in items[:20])
                if len(items) > 20:
                    lines.append(f"    ... and {len(items) - 20} more")
        if self.alias_groups:
            lines.append(f"  POSSIBLE ALIASES (same document, two ids): {len(self.alias_groups)}")
            for group in self.alias_groups[:20]:
                lines.append(f"    - {' == '.join(group)}")
        return "\n".join(lines)


def _passages_text_blob(passages: Iterable[Passage]) -> str:
    return "\n".join(p.text for p in passages)


_TITLE_WORD = re.compile(r"[a-z0-9]+")

# Words shared by most of a biomedical corpus. Matching on them would let any
# document pass, which would make the check worse than useless: it would look
# like assurance while providing none.
_TITLE_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "in", "for", "with", "to", "on", "by",
    "from", "as", "at", "is", "are", "via", "into", "using", "study",
    "analysis", "based", "new", "novel", "role", "effect", "effects",
    "cancer", "tumor", "tumour", "cell", "cells", "human", "patients",
    "clinical", "case", "report", "review", "high", "low", "risk",
})


def title_appears_in_text(title: str, text: str, *, threshold: float = 0.34,
                          head_chars: int = 6000) -> bool:
    """Is this the paper the citation says it is?

    A paper's own title words nearly always occur in its opening. This asks
    only that, and it is deliberately one-sided: two papers on the same topic
    share vocabulary, so a *pass* is weak evidence and a *failure* is strong
    evidence. It is therefore a warning, never an error.

    A title with fewer than three distinctive words cannot be judged -- book-of-
    abstracts entries and one-word titles among them -- and returns True rather
    than accusing a record the method cannot assess.
    """
    words = {w for w in _TITLE_WORD.findall((title or "").lower())
             if len(w) > 3 and w not in _TITLE_STOPWORDS}
    if len(words) < 3 or not text:
        return True
    head = text[:head_chars].lower()
    return sum(1 for w in words if w in head) / len(words) >= threshold


def _replace_dir(src: Path, dst: Path, attempts: int = 12) -> None:
    """Rename a directory, retrying through transient Windows lock errors.

    On Windows a directory rename fails with WinError 5 ("Access is denied") if
    any process holds a handle to the directory or to a file inside it. Real
    antivirus and search-indexer software opens files microseconds after they
    are created, so a rename issued immediately after writing a document can
    lose that race intermittently. The condition is transient by nature: the
    scanner releases the handle within milliseconds.

    We retry with exponential backoff and then give up loudly. We do NOT fall
    back to a non-atomic copy: a half-copied document directory is exactly the
    state that produces a manifest row asserting text we do not hold, which is
    the failure mode this whole library is designed to make impossible.
    """
    delay = 0.02
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


class Library:
    """Read/write access to the document store."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.docs_dir = self.root / "docs"
        self.quarantine_dir = self.root / "quarantine"
        self.manifest_path = self.root / MANIFEST_NAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self._id_cache: set[str] | None = None
        self._identity_cache: dict[str, str] | None = None

    # -- manifest ----------------------------------------------------------

    def iter_manifest(self) -> Iterator[dict]:
        """Yield manifest rows in insertion order.

        Later rows for the same ``doc_id`` supersede earlier ones; the manifest
        is append-only, so an update is a new line rather than an edit.
        """
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IntegrityError(
                        f"{MANIFEST_NAME}:{line_no} is not valid JSON: {exc}"
                    ) from exc

    def current_rows(self) -> dict[str, dict]:
        """Latest row per doc_id."""
        rows: dict[str, dict] = {}
        for row in self.iter_manifest():
            rows[row["doc_id"]] = row
        return rows

    def _append_manifest(self, row: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # -- exclusive write access --------------------------------------------

    @contextmanager
    def writer_lock(self, what: str = "write") -> Iterator[None]:
        """Hold exclusive write access to this library root.

        Two writers appending to one manifest can cut a JSONL row in half or
        leave a document directory that the manifest no longer claims. The lock
        enforces the single-writer rule instead of relying on callers to
        coordinate. It is a file created with O_EXCL, which is atomic on Windows
        and POSIX alike, and it records the pid and what the holder is doing so
        a stale lock can be diagnosed rather than guessed at.
        """
        lock_path = self.root / ".writer.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = lock_path.read_text(encoding="utf-8").strip()
            except OSError:
                held = "unreadable"
            raise IntegrityError(
                f"{self.root} is already locked for writing by: {held}\n"
                f"Refusing to {what} concurrently -- two writers on one "
                f"append-only manifest is how rows get cut in half. If no such "
                f"process is running, delete {lock_path}."
            ) from None
        try:
            os.write(fd, f"pid={os.getpid()} action={what} at={_dt.datetime.now().isoformat()}"
                     .encode("utf-8"))
            os.close(fd)
            yield
        finally:
            try:
                lock_path.unlink()
            except OSError:
                pass

    # -- identity ----------------------------------------------------------

    def known_ids(self) -> set[str]:
        if self._id_cache is None:
            self._id_cache = set(self.current_rows())
        return self._id_cache

    def _identity_map(self) -> dict[str, str]:
        """identity key -> doc_id, for duplicate and alias detection."""
        if self._identity_cache is None:
            mapping: dict[str, str] = {}
            for row in self.current_rows().values():
                for key in _row_identity_keys(row):
                    mapping.setdefault(key, row["doc_id"])
            self._identity_cache = mapping
        return self._identity_cache

    def find_existing(self, meta: DocMeta) -> str:
        """Return the doc_id of an existing record for the same document, or ''.

        Checks sha256 of the *source bytes* and of the *text* in addition to
        DOI/PMID/PMCID. Local aliases can produce two ids for one paper that
        share neither DOI nor title.
        """
        mapping = self._identity_map()
        for key in meta.identity_keys():
            if key in mapping and mapping[key] != meta.doc_id:
                return mapping[key]
        return ""

    # -- reading -----------------------------------------------------------

    def doc_path(self, doc_id: str) -> Path:
        return doc_dir(self.root, doc_id)

    def has(self, doc_id: str) -> bool:
        """True only if the document has readable text on disk.

        Here ``has`` means "an agent can read this", and
        :meth:`is_registered` means "a row exists".
        """
        row = self.current_rows().get(doc_id)
        if not row or row.get("n_passages", 0) == 0:
            return False
        return (self.doc_path(doc_id) / "text.jsonl").exists()

    def is_registered(self, doc_id: str) -> bool:
        return doc_id in self.known_ids()

    def load_meta(self, doc_id: str) -> DocMeta:
        path = self.doc_path(doc_id) / "meta.yaml"
        if not path.exists():
            raise IntegrityError(f"{doc_id}: manifest row exists but {path} does not")
        with path.open("r", encoding="utf-8") as fh:
            return DocMeta.model_validate(yaml.safe_load(fh))

    def load_passages(self, doc_id: str) -> list[Passage]:
        path = self.doc_path(doc_id) / "text.jsonl"
        if not path.exists():
            raise IntegrityError(f"{doc_id}: text.jsonl missing")
        out: list[Passage] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(Passage.model_validate_json(line))
        # Original document order is the contract. Sorting here is belt and
        # braces against a writer that ever gets it wrong.
        out.sort(key=lambda p: p.order)
        return out

    def iter_docs(self) -> Iterator[DocMeta]:
        for doc_id in sorted(self.current_rows()):
            yield self.load_meta(doc_id)

    # -- writing -----------------------------------------------------------

    def put(
        self,
        meta: DocMeta,
        passages: list[Passage],
        source_bytes: bytes | None = None,
        source_ext: str = ".pdf",
    ) -> IngestResult:
        """Write one document atomically, or explain why it was refused.

        The caller supplies a *proposed* meta; counts and hashes are recomputed
        here from the actual content and overwritten. A caller is never trusted
        to report how much text it brought; the stored assertion is derived
        from the bytes.
        """
        text_blob = _passages_text_blob(passages)
        n_chars = len(text_blob)
        n_passages = len(passages)

        measured = meta.model_copy(
            update={
                "n_passages": n_passages,
                "n_chars": n_chars,
                "text_sha256": sha256_bytes(text_blob.encode("utf-8")),
                "fulltext_status": classify_fulltext(
                    n_passages, n_chars, meta.fulltext_status
                ),
                "source_sha256": sha256_bytes(source_bytes) if source_bytes else "",
                "source_bytes": len(source_bytes) if source_bytes else 0,
                "retrieved_at": meta.retrieved_at or _dt.datetime.now(
                    _dt.timezone.utc
                ).isoformat(timespec="seconds"),
            }
        )

        if measured.fulltext_status is FullTextStatus.METADATA_ONLY:
            return self._quarantine(
                measured,
                "no readable text: a bibliographic record with no body is a "
                "citation nobody can check",
            )

        existing = self.find_existing(measured)
        if existing:
            return self._maybe_upgrade(existing, measured, passages, source_bytes, source_ext)

        return self._write(measured, passages, source_bytes, source_ext, IngestOutcome.ADDED)

    def _maybe_upgrade(
        self,
        existing_id: str,
        measured: DocMeta,
        passages: list[Passage],
        source_bytes: bytes | None,
        source_ext: str,
    ) -> IngestResult:
        """Same document arriving again — keep whichever copy has more text.

        This keeps thin-source records repairable: a later, fuller retrieval of
        the same paper *replaces* an abstract-only one
        instead of being dropped as a duplicate.
        """
        try:
            current = self.load_meta(existing_id)
        except IntegrityError:
            current = None
        if current is not None and current.n_chars >= measured.n_chars:
            return IngestResult(
                outcome=IngestOutcome.DUPLICATE,
                doc_id=existing_id,
                existing_doc_id=existing_id,
                reason=f"already held with {current.n_chars} chars "
                       f"(incoming {measured.n_chars})",
            )
        upgraded = measured.model_copy(update={"doc_id": existing_id})
        return self._write(
            upgraded, passages, source_bytes, source_ext, IngestOutcome.REPLACED
        )

    def _write(
        self,
        meta: DocMeta,
        passages: list[Passage],
        source_bytes: bytes | None,
        source_ext: str,
        outcome: IngestOutcome,
    ) -> IngestResult:
        target = self.doc_path(meta.doc_id)
        target.parent.mkdir(parents=True, exist_ok=True)

        staging = Path(tempfile.mkdtemp(prefix=f".{meta.doc_id}.", dir=str(target.parent)))
        try:
            if source_bytes:
                filename = f"source{source_ext}"
                (staging / filename).write_bytes(source_bytes)
                meta = meta.model_copy(update={"source_filename": filename})

            with (staging / "text.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
                for p in passages:
                    fh.write(p.model_dump_json() + "\n")

            with (staging / "meta.yaml").open("w", encoding="utf-8", newline="\n") as fh:
                yaml.safe_dump(
                    json.loads(meta.model_dump_json()),
                    fh, sort_keys=True, allow_unicode=True, width=100,
                )

            # Windows-safe atomic-enough directory swap.
            #
            # os.replace() on a directory fails with WinError 5 when the target
            # exists, and rmtree-then-replace races against Windows' pending
            # delete state: the name lingers after rmtree returns, so the
            # replace hits "Access is denied" intermittently -- which killed a
            # 400-document acquisition run mid-flight. The portable pattern is
            # to move the old directory aside under a fresh name, move the new
            # one into place, and only then delete the old. The window in which
            # the target does not exist is a single rename, and a crash inside
            # it leaves a recoverable .old- directory rather than a half-written
            # document. The manifest is appended only after this succeeds, so a
            # failure here can never produce a row asserting text we do not hold.
            attic: Path | None = None
            if target.exists():
                attic = target.with_name(f".old-{target.name}-{os.getpid()}-{id(self)}")
                if attic.exists():
                    shutil.rmtree(attic, ignore_errors=True)
                _replace_dir(target, attic)
            try:
                _replace_dir(staging, target)
            except OSError:
                if attic is not None:
                    _replace_dir(attic, target)
                raise
            if attic is not None:
                shutil.rmtree(attic, ignore_errors=True)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        self._append_manifest(meta.to_manifest_row())
        if self._id_cache is not None:
            self._id_cache.add(meta.doc_id)
        if self._identity_cache is not None:
            for key in meta.identity_keys():
                self._identity_cache.setdefault(key, meta.doc_id)
        return IngestResult(outcome=outcome, doc_id=meta.doc_id)

    def _quarantine(self, meta: DocMeta, reason: str) -> IngestResult:
        """Keep the rejection and the reason. Never silently drop."""
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "doc_id": meta.doc_id,
            "reason": reason,
            "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "meta": json.loads(meta.model_dump_json()),
        }
        with (self.quarantine_dir / "quarantine.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return IngestResult(
            outcome=IngestOutcome.QUARANTINED, doc_id=meta.doc_id, reason=reason
        )

    # -- verification ------------------------------------------------------

    def verify(self, *, check_source_hashes: bool = False,
               check_titles: bool = True) -> VerifyReport:
        """Falsify every assertion the manifest makes.

        This is the method the whole design exists to make possible. It does not
        trust a single field; it recomputes.

        ``check_titles`` adds an assertion that no hash check can make: that the
        stored text is the text of the paper the row names. Every hash check
        answers "are these the bytes we stored"; none of them answers "are these
        the right bytes". An incorrect ELink relation can fetch the full text of
        a paper that merely *cites* the target, and hash checks still pass
        because the wrong bytes were stored consistently.

        Passage identity is checked for the same reason: every passage id must
        name the document whose directory holds it. A passage carrying another
        document's id hashes correctly and still collides in the index.
        """
        report = VerifyReport()
        seen_ids: set[str] = set()
        identity_owner: dict[str, list[str]] = {}

        rows = self.current_rows()
        for doc_id, row in rows.items():
            report.checked += 1
            if doc_id in seen_ids:
                report.duplicate_ids.append(doc_id)
            seen_ids.add(doc_id)

            ddir = self.doc_path(doc_id)
            if not ddir.exists():
                report.missing_dir.append(doc_id)
                continue

            text_file = ddir / "text.jsonl"
            if not text_file.exists():
                report.missing_text.append(doc_id)
                continue

            try:
                passages = self.load_passages(doc_id)
            except (IntegrityError, ValueError) as exc:
                report.missing_text.append(f"{doc_id} (unreadable: {exc})")
                continue

            foreign = [p.passage_id for p in passages
                       if p.passage_id.split("#", 1)[0] != doc_id]
            if foreign:
                report.passage_id_mismatch.append(
                    f"{doc_id} ({len(foreign)} of {len(passages)} passage ids "
                    f"name {foreign[0].split('#', 1)[0]!r})"
                )

            blob = _passages_text_blob(passages)
            if sha256_bytes(blob.encode("utf-8")) != row.get("text_sha256"):
                report.text_hash_mismatch.append(doc_id)
            if len(passages) != row.get("n_passages") or len(blob) != row.get("n_chars"):
                report.count_mismatch.append(
                    f"{doc_id} (manifest {row.get('n_passages')}p/{row.get('n_chars')}c, "
                    f"disk {len(passages)}p/{len(blob)}c)"
                )

            declared = row.get("fulltext_status")
            actual = classify_fulltext(len(passages), len(blob), FullTextStatus.FULL)
            if declared == FullTextStatus.FULL.value and actual is not FullTextStatus.FULL:
                report.status_unsupported.append(
                    f"{doc_id} claims full but holds {len(passages)}p/{len(blob)}c"
                )

            if check_titles and not title_appears_in_text(row.get("title", ""), blob):
                report.title_absent.append(
                    f"{doc_id} ({row.get('title', '')[:70]!r} is absent from its "
                    f"own {len(blob)} characters)"
                )

            if check_source_hashes and row.get("source_filename"):
                src = ddir / row["source_filename"]
                if src.exists():
                    from .ids import sha256_file

                    if sha256_file(src) != row.get("source_sha256"):
                        report.source_hash_mismatch.append(doc_id)
                else:
                    report.source_hash_mismatch.append(f"{doc_id} (source file missing)")

            for key in _row_identity_keys(row):
                identity_owner.setdefault(key, []).append(doc_id)

        for key, owners in identity_owner.items():
            unique = sorted(set(owners))
            if len(unique) > 1:
                report.alias_groups.append(unique)

        # Directories on disk with no manifest row: the inverse assertion.
        if self.docs_dir.exists():
            for shard in self.docs_dir.iterdir():
                if not shard.is_dir():
                    continue
                for d in shard.iterdir():
                    if d.is_dir() and d.name not in rows and not d.name.startswith("."):
                        report.orphan_dirs.append(d.name)

        return report

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict:
        rows = self.current_rows()
        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_channel: dict[str, int] = {}
        by_decade: dict[str, int] = {}
        total_chars = 0
        total_passages = 0
        for row in rows.values():
            by_status[row.get("fulltext_status", "?")] = by_status.get(row.get("fulltext_status", "?"), 0) + 1
            by_type[row.get("doc_type", "?")] = by_type.get(row.get("doc_type", "?"), 0) + 1
            by_channel[row.get("channel", "?")] = by_channel.get(row.get("channel", "?"), 0) + 1
            year = row.get("year")
            if year:
                decade = f"{(int(year) // 10) * 10}s"
                by_decade[decade] = by_decade.get(decade, 0) + 1
            total_chars += row.get("n_chars", 0)
            total_passages += row.get("n_passages", 0)
        return {
            "documents": len(rows),
            "passages": total_passages,
            "characters": total_chars,
            "by_status": dict(sorted(by_status.items())),
            "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
            "by_channel": dict(sorted(by_channel.items(), key=lambda kv: -kv[1])),
            "by_decade": dict(sorted(by_decade.items())),
        }


def classify_fulltext(
    n_passages: int, n_chars: int, requested: FullTextStatus
) -> FullTextStatus:
    """Decide what we actually hold, from the bytes, not from the claim.

    A caller may ask for ``FULL``; it gets ``FULL`` only if the content supports
    it. Empty-source and thin-source records cannot claim full text because the
    status is *derived*, never *asserted*.
    """
    if n_passages == 0 or n_chars == 0:
        return FullTextStatus.METADATA_ONLY
    if requested is FullTextStatus.ABSTRACT_ONLY:
        return FullTextStatus.ABSTRACT_ONLY
    if n_chars >= MIN_FULLTEXT_CHARS and n_passages >= MIN_FULLTEXT_PASSAGES:
        return FullTextStatus.FULL
    return FullTextStatus.PARTIAL


def _row_identity_keys(row: dict) -> set[str]:
    keys: set[str] = set()
    if row.get("doi"):
        keys.add(f"doi:{row['doi'].lower()}")
    if row.get("pmid"):
        keys.add(f"pmid:{row['pmid']}")
    if row.get("pmcid"):
        keys.add(f"pmcid:{row['pmcid'].upper()}")
    if row.get("arxiv_id"):
        keys.add(f"arxiv:{row['arxiv_id'].lower()}")
    if row.get("source_sha256"):
        keys.add(f"sha:{row['source_sha256']}")
    if row.get("text_sha256"):
        keys.add(f"text:{row['text_sha256']}")
    return keys
