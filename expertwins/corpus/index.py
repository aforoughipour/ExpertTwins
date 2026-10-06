"""BM25 retrieval over the library, via SQLite FTS5.

Why this and not a vector store:

* **BM25 is what the 2026 evidence recommends.** SAGE (2602.05975) found BM25
  beats LLM-based retrievers by ~30% for deep-research agents, because agents
  emit keyword-shaped sub-queries rather than natural-language questions. See
  also *BM25 Wins at Scale* (2607.26497). BM25 is used here on evidence.
* **It is one file with no services.** Moving retrieval to an air-gapped cluster
  is a file copy. No embedding model to license (NV-Embed-v2, the field's dense
  anchor, is cc-by-nc-4.0 — 2608.16096), no index server, no GPU.
* **It is regenerable, therefore never the source of truth.** If this file
  disagrees with ``MANIFEST.jsonl``, this file is wrong and gets rebuilt.

Two filters make this the backbone of the whole experiment:

* ``before_year`` — the temporal holdout. Retrieval restricted to documents
  published before a cut year is what makes pre-registered rediscovery testable.
* ``topics`` / ``doc_ids`` — the corpus partition. This, and nothing else, is
  what makes one agent's corpus different from another's.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .errors import Absence, IntegrityError
from .textnorm import canonical
from .store import Library

INDEX_NAME = "index.sqlite"

_SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS docs (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    first_author    TEXT,
    year            INTEGER,
    pub_date        TEXT,
    venue           TEXT,
    doc_type        TEXT,
    provenance      TEXT,
    fulltext_status TEXT,
    n_passages      INTEGER,
    n_chars         INTEGER,
    doi             TEXT,
    pmid            TEXT,
    pmcid           TEXT,
    channel         TEXT,
    topics_json     TEXT
);
CREATE INDEX IF NOT EXISTS docs_year ON docs(year);
CREATE INDEX IF NOT EXISTS docs_type ON docs(doc_type);

CREATE TABLE IF NOT EXISTS doc_topics (
    doc_id TEXT NOT NULL,
    topic  TEXT NOT NULL,
    PRIMARY KEY (doc_id, topic)
);
CREATE INDEX IF NOT EXISTS doc_topics_topic ON doc_topics(topic);

CREATE TABLE IF NOT EXISTS passages (
    rowid       INTEGER PRIMARY KEY,
    passage_id  TEXT UNIQUE NOT NULL,
    doc_id      TEXT NOT NULL,
    ord         INTEGER NOT NULL,
    section     TEXT,
    year        INTEGER,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS passages_doc ON passages(doc_id, ord);

-- 'content' links the FTS table to `passages` so text is stored once.
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    text,
    content='passages',
    content_rowid='rowid',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class Hit:
    passage_id: str
    doc_id: str
    order: int
    section: str
    year: int | None
    text: str
    score: float
    title: str = ""
    first_author: str = ""
    venue: str = ""
    doc_type: str = ""

    def citation(self) -> str:
        who = self.first_author or "?"
        return f"{who} {self.year or '?'}, {self.venue or '?'} [{self.passage_id}]"


# FTS5 treats these as syntax. A scientific query legitimately contains
# hyphens, slashes and parentheses ("COX-2", "HSP-90 inhibitor", "5-HT2A/5-HT2C"),
# so we quote every term rather than trying to escape the grammar.
_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-+']*")


def to_fts_query(text: str, mode: str = "OR") -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Each term is double-quoted, which makes it a literal phrase and disarms
    every FTS5 operator. This is the fix for a whole class of query-injection
    and syntax-error bugs that otherwise surface only on unusual gene names.
    """
    terms = _TERM_RE.findall(canonical(text))
    terms = [t for t in terms if len(t) > 1]
    if not terms:
        return ""
    quoted = [f'"{t}"' for t in terms]
    return f" {mode} ".join(quoted)


class Index:
    """Read/write access to the FTS5 index."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- building ----------------------------------------------------------

    @classmethod
    def rebuild(cls, library: Library, path: Path | None = None, verbose: bool = False) -> Index:
        """Drop and rebuild from the library. The manifest wins, always.

        The rebuild happens *in place* rather than by deleting the file: on
        Windows an open SQLite handle cannot be unlinked, so deleting the file
        crashed whenever any Index was already open on it.
        """
        path = Path(path or (library.root / INDEX_NAME))
        index = cls(path)
        conn = index.conn
        conn.executescript(
            "DROP TABLE IF EXISTS passages_fts;"
            "DROP TABLE IF EXISTS passages;"
            "DROP TABLE IF EXISTS doc_topics;"
            "DROP TABLE IF EXISTS docs;"
            "DROP TABLE IF EXISTS meta;"
        )
        conn.executescript(_SCHEMA)
        rows = library.current_rows()
        n_docs = 0
        n_passages = 0
        skipped: list[str] = []
        conn.execute("BEGIN")
        for doc_id, row in rows.items():
            if row.get("n_passages", 0) == 0:
                continue
            try:
                passages = library.load_passages(doc_id)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                # A document the manifest claims we hold but whose text will not
                # load is a corruption signal, never a routine skip.
                skipped.append(f"{doc_id}: {exc}")
                continue

            year = row.get("year")
            conn.execute(
                "INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    doc_id, row.get("title", ""), row.get("first_author", ""),
                    year, row.get("pub_date", ""), row.get("venue", ""),
                    row.get("doc_type", ""), row.get("provenance", ""),
                    row.get("fulltext_status", ""), row.get("n_passages", 0),
                    row.get("n_chars", 0), row.get("doi", ""), row.get("pmid", ""),
                    row.get("pmcid", ""), row.get("channel", ""),
                    json.dumps(row.get("topics", [])),
                ),
            )
            for topic in row.get("topics", []) or []:
                conn.execute(
                    "INSERT OR IGNORE INTO doc_topics VALUES (?,?)", (doc_id, topic)
                )
            for p in passages:
                cur = conn.execute(
                    "INSERT INTO passages (passage_id, doc_id, ord, section, year, text) "
                    "VALUES (?,?,?,?,?,?)",
                    (p.passage_id, doc_id, p.order, p.section, year, p.text),
                )
                conn.execute(
                    "INSERT INTO passages_fts(rowid, text) VALUES (?,?)",
                    (cur.lastrowid, p.text),
                )
                n_passages += 1
            n_docs += 1
            if verbose and n_docs % 500 == 0:
                print(f"  indexed {n_docs} docs / {n_passages} passages")

        conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('built_docs', ?)", (str(n_docs),)
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('built_passages', ?)", (str(n_passages),)
        )
        conn.commit()
        conn.execute("INSERT INTO passages_fts(passages_fts) VALUES('optimize')")
        conn.commit()
        if skipped:
            raise IntegrityError(
                f"{len(skipped)} document(s) are in the manifest but their text "
                f"would not load; the index would silently under-report them. "
                f"Run `python ops/doctor.py`. First: {skipped[0]}"
            )
        return index

    # -- searching ---------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        before_year: int | None = None,
        after_year: int | None = None,
        topics: list[str] | None = None,
        exclude_topics: list[str] | None = None,
        doc_types: list[str] | None = None,
        exclude_doc_ids: list[str] | None = None,
        mode: str = "OR",
    ) -> list[Hit] | Absence:
        """BM25 search with partition and temporal filters.

        Returns an :class:`Absence` on no results, carrying how many passages
        were in scope — so a caller can tell "the corpus does not contain this"
        from "the filters excluded everything", which is exactly the distinction
        a bare empty list cannot make.
        """
        match = to_fts_query(query, mode=mode)
        if not match:
            return Absence(
                what="usable query terms", method="fts5 tokenisation",
                query=query, examined=0,
            )

        where = ["passages_fts MATCH ?"]
        params: list[object] = [match]

        if before_year is not None:
            where.append("(p.year IS NOT NULL AND p.year < ?)")
            params.append(before_year)
        if after_year is not None:
            where.append("(p.year IS NOT NULL AND p.year >= ?)")
            params.append(after_year)
        if doc_types:
            where.append(f"d.doc_type IN ({','.join('?' * len(doc_types))})")
            params.extend(doc_types)
        if topics:
            where.append(
                "EXISTS (SELECT 1 FROM doc_topics t WHERE t.doc_id = p.doc_id "
                f"AND t.topic IN ({','.join('?' * len(topics))}))"
            )
            params.extend(topics)
        if exclude_topics:
            where.append(
                "NOT EXISTS (SELECT 1 FROM doc_topics t WHERE t.doc_id = p.doc_id "
                f"AND t.topic IN ({','.join('?' * len(exclude_topics))}))"
            )
            params.extend(exclude_topics)
        if exclude_doc_ids:
            where.append(f"p.doc_id NOT IN ({','.join('?' * len(exclude_doc_ids))})")
            params.extend(exclude_doc_ids)

        sql = f"""
            SELECT p.passage_id, p.doc_id, p.ord, p.section, p.year, p.text,
                   bm25(passages_fts) AS score,
                   d.title, d.first_author, d.venue, d.doc_type
            FROM passages_fts
            JOIN passages p ON p.rowid = passages_fts.rowid
            JOIN docs d ON d.doc_id = p.doc_id
            WHERE {' AND '.join(where)}
            ORDER BY score
            LIMIT ?
        """
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()

        if not rows:
            return Absence(
                what="matching passages", method="sqlite fts5 bm25",
                query=f"{query!r} -> {match}",
                examined=self.passages_in_scope(
                    before_year=before_year, after_year=after_year,
                    topics=topics, exclude_topics=exclude_topics,
                ),
                detail={
                    "before_year": before_year, "topics": topics,
                    "exclude_topics": exclude_topics,
                },
            )

        # bm25() returns a negative score where more negative is better; flip it
        # so that callers can sort descending like every other ranker.
        return [
            Hit(
                passage_id=r["passage_id"], doc_id=r["doc_id"], order=r["ord"],
                section=r["section"] or "", year=r["year"], text=r["text"],
                score=-float(r["score"]), title=r["title"] or "",
                first_author=r["first_author"] or "", venue=r["venue"] or "",
                doc_type=r["doc_type"] or "",
            )
            for r in rows
        ]

    def passages_in_scope(
        self,
        *,
        before_year: int | None = None,
        after_year: int | None = None,
        topics: list[str] | None = None,
        exclude_topics: list[str] | None = None,
    ) -> int:
        """How many passages the filters admit, ignoring the query.

        This number is what turns a zero-result search into evidence. A search
        that examined 0 passages says nothing about the world.
        """
        where = ["1=1"]
        params: list[object] = []
        if before_year is not None:
            where.append("(p.year IS NOT NULL AND p.year < ?)")
            params.append(before_year)
        if after_year is not None:
            where.append("(p.year IS NOT NULL AND p.year >= ?)")
            params.append(after_year)
        if topics:
            where.append(
                "EXISTS (SELECT 1 FROM doc_topics t WHERE t.doc_id = p.doc_id "
                f"AND t.topic IN ({','.join('?' * len(topics))}))"
            )
            params.extend(topics)
        if exclude_topics:
            where.append(
                "NOT EXISTS (SELECT 1 FROM doc_topics t WHERE t.doc_id = p.doc_id "
                f"AND t.topic IN ({','.join('?' * len(exclude_topics))}))"
            )
            params.extend(exclude_topics)
        sql = f"SELECT COUNT(*) FROM passages p WHERE {' AND '.join(where)}"
        return int(self.conn.execute(sql, params).fetchone()[0])

    def neighbours(self, doc_id: str, order: int, window: int = 1) -> list[Hit]:
        """Passages immediately around a hit, in document order.

        Retrieval returns fragments; a qualifier ("except the quiescent state")
        often sits in the *next* sentence. Restoring local document order before the
        agent reads is the OP-RAG/DOS RAG finding applied at read time.
        """
        rows = self.conn.execute(
            """SELECT p.passage_id, p.doc_id, p.ord, p.section, p.year, p.text,
                      d.title, d.first_author, d.venue, d.doc_type
               FROM passages p JOIN docs d ON d.doc_id = p.doc_id
               WHERE p.doc_id = ? AND p.ord BETWEEN ? AND ?
               ORDER BY p.ord""",
            (doc_id, order - window, order + window),
        ).fetchall()
        return [
            Hit(
                passage_id=r["passage_id"], doc_id=r["doc_id"], order=r["ord"],
                section=r["section"] or "", year=r["year"], text=r["text"],
                score=0.0, title=r["title"] or "", first_author=r["first_author"] or "",
                venue=r["venue"] or "", doc_type=r["doc_type"] or "",
            )
            for r in rows
        ]

    def stats(self) -> dict:
        cur = self.conn.execute("SELECT COUNT(*) FROM docs")
        docs = cur.fetchone()[0]
        passages = self.conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        topics = self.conn.execute(
            "SELECT topic, COUNT(*) c FROM doc_topics GROUP BY topic ORDER BY c DESC"
        ).fetchall()
        return {
            "documents": docs,
            "passages": passages,
            "topics": {r["topic"]: r["c"] for r in topics},
        }
