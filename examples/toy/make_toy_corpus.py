#!/usr/bin/env python3
r"""Build a deterministic toy ExpertTwins library and FTS index.

The generated corpus is intentionally fictional.  It contains short invented
"papers" about lanternmoss, echo pollen, and glasswing voles on the imaginary
island of Peloria.  It is only a structural test corpus for offline demos.

Usage:
    python make_toy_corpus.py C:\path\to\toy_library --force

Point the project at the result with:
    EXPERTWINS_LIBRARY=<toy_library>
    EXPERTWINS_INDEX=<toy_library>\index.sqlite
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Iterable

SEED = 1729
MIN_FULLTEXT_CHARS = 6000
MIN_FULLTEXT_PASSAGES = 5

SCHEMA = """
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

CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    text,
    content='passages',
    content_rowid='rowid',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

DROP_SCHEMA = """
DROP TABLE IF EXISTS passages_fts;
DROP TABLE IF EXISTS passages;
DROP TABLE IF EXISTS doc_topics;
DROP TABLE IF EXISTS docs;
DROP TABLE IF EXISTS meta;
"""

SPACE_RE = re.compile(r"\s+")
SLUG_RE = re.compile(r"[^a-z0-9]+")


def canonical(text: str) -> str:
    return SPACE_RE.sub(" ", text or "").strip()


def slugify(text: str) -> str:
    return SLUG_RE.sub("", canonical(text).lower())


def shard_of(doc_id: str) -> str:
    return (doc_id + "__")[:2]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_fulltext(n_passages: int, n_chars: int, requested: str) -> str:
    if n_passages == 0 or n_chars == 0:
        return "metadata_only"
    if requested == "abstract_only":
        return "abstract_only"
    if n_chars >= MIN_FULLTEXT_CHARS and n_passages >= MIN_FULLTEXT_PASSAGES:
        return "full"
    return "partial"


def text_blob(passages: Iterable[dict]) -> str:
    return "\n".join(p["text"] for p in passages)


AUTHORS = [
    ["Mira Brindle", "Oren Vale", "Safi Quill"],
    ["Toma Reed", "Ilya Fern", "Nessa Bright"],
    ["Lio Kestrel", "Mara Pin", "Jun Glass"],
    ["Rhea Moss", "Pavel Lume", "Ida Finch"],
    ["Soren Wick", "Amal Thistle", "Bryn Cove"],
    ["Elan Quire", "Nola Wren", "Fenn Alder"],
]

VENUES = [
    "Journal of Imaginary Fieldcraft",
    "Pelorian Annals of Make-Believe Biology",
    "Transactions of Fictional Ecology",
]

SUBJECTS = [
    ("lanternmoss", "prism spores", "moonlit terraces", "luminous bracts"),
    ("glasswing voles", "echo pollen", "hollow reeds", "resonant nests"),
    ("cloud kelp", "silver dew", "floating ponds", "mist anchors"),
    ("clockwork beetles", "amber gears", "basalt orchards", "tick songs"),
    ("velvet lichen", "blue lanterns", "north ravine", "quiet glow"),
    ("paperwing moths", "saffron dust", "folded leaves", "dawn spirals"),
]

METHODS = [
    "painted counting stones",
    "wind-up listening jars",
    "cardboard spectroscopes",
    "tin-rimmed dew trays",
    "thread maps tied to cedar pegs",
]

FINDINGS = [
    "co-occurred whenever the story moon was described as green",
    "changed direction after a bell was rung twice",
    "formed a ring around the narrator's boot prints",
    "faded when the field notebook was closed",
    "brightened beside nonsense syllables written in blue ink",
]


def make_docs() -> list[dict]:
    rng = random.Random(SEED)
    docs: list[dict] = []
    taken: set[str] = set()
    years = list(range(2011, 2029))
    for i in range(18):
        subject, term_a, place, term_b = SUBJECTS[i % len(SUBJECTS)]
        authors = AUTHORS[i % len(AUTHORS)]
        year = years[i]
        venue = VENUES[i % len(VENUES)]
        title = f"Fictional observations of {subject} and {term_a} on Peloria plot {i + 1:02d}"
        base_id = f"{slugify(authors[0].split()[-1])}{year}peloria"
        doc_id = base_id
        suffix = ord("a")
        while doc_id in taken:
            doc_id = f"{base_id}{chr(suffix)}"
            suffix += 1
        taken.add(doc_id)

        method = METHODS[i % len(METHODS)]
        finding = FINDINGS[(i * 2) % len(FINDINGS)]
        contrast_subject, contrast_a, contrast_place, contrast_b = SUBJECTS[(i + 2) % len(SUBJECTS)]
        rng_terms = [subject, term_a, place, term_b, contrast_subject, contrast_a]
        rng.shuffle(rng_terms)

        sections = ["Abstract", "Setting", "Methods", "Results", "Interpretation", "Toy note"]
        raw_passages = [
            f"{title}. This toy paper is a deliberately fictional record from the island of Peloria. The authors describe {subject}, {term_a}, and {term_b} in a make-believe landscape so the corpus can be shared without real scientific claims. The abstract says the work is for software testing only.",
            f"The setting was {place}, a drawn map square that exists only in the demonstration corpus. Field notes mention {subject} beside {contrast_place}, but every measurement is imaginary. The repeated vocabulary includes {rng_terms[0]}, {rng_terms[1]}, and {rng_terms[2]} to make retrieval predictable.",
            f"Methods used {method} and a fixed chant of invented terms. Observers counted {term_a} near {subject} and compared the count with {contrast_subject}. No animals, plants, patients, or real places were studied; the protocol is a structural stand-in for full text passages.",
            f"Results reported that {subject} and {term_a} {finding}. A second table, not actually present, would have listed {term_b}, {contrast_a}, and {place}. The passage repeats lanternmoss, echo pollen, glasswing voles, cloud kelp, and prism spores across different papers for search tests.",
            f"Interpretation stayed inside the fiction. The authors suggested that {term_b} might guide {subject} toward {place}, while {contrast_subject} provided a contrasting motif. The claim has no bearing on biology, medicine, ecology, or any real scientific literature.",
            f"Toy note: this document exists to exercise sharded directories, manifest rows, document topics, passage order, and SQLite FTS retrieval. A query for {subject} {term_a} should find this passage, and neighbouring passages should remain in original order.",
        ]
        passages = [
            {
                "passage_id": f"{doc_id}#p{j:04d}",
                "order": j,
                "section": sections[j],
                "text": canonical(text),
            }
            for j, text in enumerate(raw_passages)
        ]
        blob = text_blob(passages)
        # The `<surname>.authored` topic is what lets a seat's own corpus be
        # attributed without a name-matching pass. In a real library that topic
        # is written by `ops/people.py attribute`; here it is baked in so the
        # toy example can demonstrate the [YOURS] marker offline.
        topics = sorted({
            "toy",
            "peloria",
            subject.replace(" ", "_"),
            term_a.replace(" ", "_"),
            f"{slugify(authors[0].split()[-1])}.authored",
        })
        status = classify_fulltext(len(passages), len(blob), "full")
        pub_date = f"{year}-{(i % 12) + 1:02d}-15"
        retrieved_at = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc).isoformat(timespec="seconds")
        meta = {
            "doc_id": doc_id,
            "title": title,
            "authors": authors,
            "year": year,
            "pub_date": pub_date,
            "venue": venue,
            "doc_type": "research_article" if i % 5 else "review",
            "provenance": "published",
            "doi": "",
            "pmid": "",
            "pmcid": "",
            "arxiv_id": "",
            "url": "",
            "source_sha256": "",
            "text_sha256": sha256_text(blob),
            "source_filename": "",
            "source_bytes": 0,
            "fulltext_status": status,
            "n_passages": len(passages),
            "n_chars": len(blob),
            "topics": topics,
            "channel": "toy-offline-generator",
            "retrieved_at": retrieved_at,
            "notes": "Clearly fictional toy corpus entry; not a scientific source.",
        }
        manifest = {
            "doc_id": doc_id,
            "title": title,
            "first_author": authors[0],
            "n_authors": len(authors),
            "year": year,
            "pub_date": pub_date,
            "venue": venue,
            "doc_type": meta["doc_type"],
            "provenance": meta["provenance"],
            "doi": "",
            "pmid": "",
            "pmcid": "",
            "arxiv_id": "",
            "source_sha256": "",
            "text_sha256": meta["text_sha256"],
            "source_filename": "",
            "source_bytes": 0,
            "fulltext_status": status,
            "n_passages": len(passages),
            "n_chars": len(blob),
            "topics": topics,
            "channel": meta["channel"],
            "retrieved_at": retrieved_at,
        }
        docs.append({"meta": meta, "manifest": manifest, "passages": passages})
    return docs


def write_library(root: Path, docs: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(exist_ok=True)
    (root / "quarantine").mkdir(exist_ok=True)
    with (root / "MANIFEST.jsonl").open("w", encoding="utf-8", newline="\n") as manifest_fh:
        for doc in docs:
            meta = doc["meta"]
            doc_dir = root / "docs" / shard_of(meta["doc_id"]) / meta["doc_id"]
            doc_dir.mkdir(parents=True, exist_ok=True)
            with (doc_dir / "text.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
                for passage in doc["passages"]:
                    fh.write(json.dumps(passage, ensure_ascii=False, separators=(",", ":")) + "\n")
            # JSON is valid YAML, so expertwins.corpus.store.Library.load_meta()
            # can read this meta.yaml.
            with (doc_dir / "meta.yaml").open("w", encoding="utf-8", newline="\n") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.write("\n")
            manifest_fh.write(json.dumps(doc["manifest"], ensure_ascii=False, sort_keys=True) + "\n")


def build_index(root: Path, docs: list[dict]) -> tuple[int, int]:
    index_path = root / "index.sqlite"
    conn = sqlite3.connect(str(index_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(DROP_SCHEMA)
    conn.executescript(SCHEMA)
    n_docs = 0
    n_passages = 0
    conn.execute("BEGIN")
    for doc in docs:
        row = doc["manifest"]
        conn.execute(
            "INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["doc_id"], row["title"], row["first_author"], row["year"],
                row["pub_date"], row["venue"], row["doc_type"], row["provenance"],
                row["fulltext_status"], row["n_passages"], row["n_chars"],
                row["doi"], row["pmid"], row["pmcid"], row["channel"],
                json.dumps(row["topics"]),
            ),
        )
        for topic in row["topics"]:
            conn.execute("INSERT OR IGNORE INTO doc_topics VALUES (?,?)", (row["doc_id"], topic))
        for passage in doc["passages"]:
            cur = conn.execute(
                "INSERT INTO passages (passage_id, doc_id, ord, section, year, text) VALUES (?,?,?,?,?,?)",
                (passage["passage_id"], row["doc_id"], passage["order"], passage["section"], row["year"], passage["text"]),
            )
            conn.execute("INSERT INTO passages_fts(rowid, text) VALUES (?,?)", (cur.lastrowid, passage["text"]))
            n_passages += 1
        n_docs += 1
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('built_docs', ?)", (str(n_docs),))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('built_passages', ?)", (str(n_passages),))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('toy_seed', ?)", (str(SEED),))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('description', ?)", ("fictional offline toy corpus",))
    conn.commit()
    conn.execute("INSERT INTO passages_fts(passages_fts) VALUES('optimize')")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return n_docs, n_passages


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic fictional toy ExpertTwins corpus.")
    parser.add_argument("output", type=Path, help="Directory to create as the library root")
    parser.add_argument("--force", action="store_true", help="Delete the output directory first if it exists")
    args = parser.parse_args()

    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"Refusing to overwrite non-empty directory: {out} (use --force)")
        shutil.rmtree(out)
    docs = make_docs()
    write_library(out, docs)
    n_docs, n_passages = build_index(out, docs)
    n_chars = sum(d["manifest"]["n_chars"] for d in docs)
    statuses: dict[str, int] = {}
    for d in docs:
        s = d["manifest"]["fulltext_status"]
        statuses[s] = statuses.get(s, 0) + 1
    print(f"built toy ExpertTwins library at {out}")
    print(f"documents: {n_docs}; passages: {n_passages}; characters: {n_chars}; seed: {SEED}")
    print("fulltext_status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    print(f"manifest: {out / 'MANIFEST.jsonl'}")
    print(f"index: {out / 'index.sqlite'}")


if __name__ == "__main__":
    main()
