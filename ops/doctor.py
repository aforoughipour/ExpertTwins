"""Is this installation actually working? Run this first, and after every move.

The failure this exists to catch is the one that does not announce itself: a
copy of the project onto a cluster where the corpus did not come with it, or
came half-written, and every seat then reports "no evidence found" -- which
reads identically to a genuine scientific negative.

Every check is offline. Nothing here touches the network.

Besides the index probe, the library is audited against the files on disk:
every manifest row is checked against its document directory (text hash,
passage and character counts, full-text status, passage identity, title), and
the index is compared with the manifest. On a large corpus this reads every
document; `--quick` skips the file-level audit.

    python ops/doctor.py [--quick]
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expertwins import paths  # noqa: E402

OK, WARN, BAD = "ok  ", "WARN", "FAIL"

# Below this a corpus is a demo, not evidence. The toy example in examples/toy
# ships far fewer documents deliberately, so this check is expected to fail
# while you are working through the quickstart.
MIN_CORPUS_DOCS = 100


def main() -> int:
    quick = "--quick" in sys.argv[1:]
    print(paths.describe())
    print()
    problems = 0
    warnings = 0

    def say(level: str, what: str, detail: str = "") -> None:
        nonlocal problems, warnings
        print(f"[{level}] {what}" + (f"  -- {detail}" if detail else ""))
        if level == BAD:
            problems += 1
        elif level == WARN:
            warnings += 1

    # -- python and libraries ---------------------------------------------
    v = sys.version_info
    say(OK if v >= (3, 11) else BAD, f"python {v.major}.{v.minor}.{v.micro}",
        "" if v >= (3, 11) else "3.11 or newer is required")
    for mod in ("yaml", "numpy", "pydantic"):
        try:
            __import__(mod)
            say(OK, f"import {mod}")
        except ImportError as exc:
            say(BAD, f"import {mod}", str(exc))
    for mod, why in (("requests", "acquisition only"), ("lxml", "acquisition only"),
                     ("fitz", "PDF extraction, acquisition only")):
        try:
            __import__(mod)
            say(OK, f"import {mod}", why)
        except ImportError:
            say(WARN, f"import {mod} missing", f"{why}; the read path still works")
    try:
        import sentence_transformers  # noqa: F401
        say(OK, "sentence-transformers present", "semantic axis uses a real encoder")
    except ImportError:
        say(WARN, "sentence-transformers absent",
            "the semantic axis falls back to a tf-idf cosine kernel. That is a "
            "DIFFERENT backend and its numbers are not comparable with a run "
            "that had an encoder. The backend is recorded in every report.")

    # -- the corpus layer ---------------------------------------------------
    try:
        from expertwins.corpus.store import Library  # noqa: F401
        say(OK, "expertwins.corpus importable")
    except ImportError as exc:
        say(BAD, "expertwins.corpus not importable", str(exc))

    # -- the corpus --------------------------------------------------------
    if not paths.LIBRARY.exists():
        say(BAD, "library missing", f"{paths.LIBRARY} does not exist")
    elif not paths.INDEX.exists():
        say(BAD, "index missing",
            f"{paths.INDEX} does not exist. Rebuild it before running anything: "
            f"reading a half-written index returns zeros that look exactly like "
            f"a coverage gap.")
    else:
        try:
            db = sqlite3.connect(f"file:{paths.INDEX.as_posix()}?mode=ro", uri=True)
            n_docs = db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            n_pass = db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
            say(OK if n_docs > MIN_CORPUS_DOCS else BAD,
                f"index holds {n_docs:,} documents, {n_pass:,} passages",
                "" if n_docs > MIN_CORPUS_DOCS else
                f"below {MIN_CORPUS_DOCS} documents a seat's retrieval cannot "
                f"separate 'the corpus does not say this' from 'the corpus is "
                f"too small to say anything'. Expected on the toy example in "
                f"examples/toy, which ships 18 documents on purpose; not "
                f"acceptable for a real panel.")
            # Probe with a token drawn from a stored passage and require FTS to
            # return that same passage. This checks that the index answers and
            # that passages_fts.rowid is aligned with passages.rowid; a
            # misaligned index returns hits that point at the wrong passage.
            row = db.execute(
                "SELECT rowid, text FROM passages ORDER BY rowid LIMIT 1").fetchone()
            tok = None
            if row:
                m = re.search(r"[A-Za-z]{4,}", row[1] or "")
                tok = m.group(0) if m else None
            if tok is None:
                say(BAD, "FTS probe", "no passage with a searchable token")
            else:
                aligned = db.execute(
                    "SELECT COUNT(*) FROM passages_fts "
                    "WHERE passages_fts MATCH ? AND rowid = ?",
                    (f'"{tok}"', row[0])).fetchone()[0]
                say(OK if aligned else BAD,
                    f"FTS responds (probe '{tok}' returns its source passage)"
                    if aligned else f"FTS probe '{tok}' did not return its source passage",
                    "" if aligned else
                    "the index is stale or its rowids are misaligned with the "
                    "passages table; rebuild it with: python ops/people.py reindex")
            if not quick:
                _audit_library(db, say)
        except sqlite3.Error as exc:
            say(BAD, "index unreadable", str(exc))

    # -- seats -------------------------------------------------------------
    if not paths.SEATS.exists():
        say(BAD, "config/seats.yaml missing", "no seat is registered")
    else:
        import yaml
        seats = yaml.safe_load(paths.SEATS.read_text(encoding="utf-8"))["seats"]
        people = [s for s in seats if s.get("kind") == "individual"]
        say(OK, f"{len(seats)} seats registered", f"{len(people)} of them people")
        unattributed = []
        for s in people:
            p = paths.PEOPLE / f"{s['name']}.own.json"
            if not p.exists():
                unattributed.append(s["name"])
                continue
            j = json.loads(p.read_text(encoding="utf-8"))
            if j["n_accepted"] == 0:
                unattributed.append(s["name"])
            elif j["n_candidates"] and j["n_accepted"] / j["n_candidates"] < 0.5:
                say(WARN, f"{s['name']}: {j['n_accepted']} own papers accepted of "
                    f"{j['n_candidates']} candidates",
                    "more than half rejected -- read the rejection list before "
                    "deciding whether the query or the matcher is at fault")
            else:
                say(OK, f"{s['name']}: {j['n_accepted']} own papers "
                    f"({j['n_rejected']} rejected)")
        if unattributed:
            say(WARN, f"{len(unattributed)} person seat(s) with no own corpus",
                ", ".join(unattributed) + " -- these will behave as generalists "
                "until acquired and attributed, and their fidelity cannot be "
                "measured at all")

    print()
    if problems:
        print(f"{problems} FAILURE(S) and {warnings} warning(s). Do not run the "
              f"panel until the failures are fixed: a seat with no evidence "
              f"produces 'no evidence found', which is indistinguishable in the "
              f"transcript from a genuine scientific negative.")
        return 1
    print(f"No failures, {warnings} warning(s). The installation is usable.")
    return 0


def _audit_library(db: sqlite3.Connection, say) -> None:
    """Re-derive the library from the files, then compare the index with it."""
    try:
        from expertwins.corpus.store import Library
    except ImportError:
        return
    lib = Library(paths.LIBRARY)
    report = lib.verify()
    if report.errors:
        say(BAD, f"library audit: {report.errors} error(s) in "
            f"{report.checked:,} documents",
            "the files on disk disagree with the manifest. Details follow; "
            "ops/repair_passage_ids.py addresses passage-identity errors, and "
            "the index must be rebuilt after any repair")
    elif report.warnings:
        say(WARN, f"library audit: {report.warnings} warning(s) in "
            f"{report.checked:,} documents", "details follow")
    else:
        say(OK, f"library audit: {report.checked:,} documents agree with the manifest")
    if report.errors or report.warnings:
        print("\n".join("       " + line for line in report.render().splitlines()[1:]))

    rows = lib.current_rows()
    indexed = {d for (d,) in db.execute("SELECT doc_id FROM docs")}
    missing = set(rows) - indexed
    extra = indexed - set(rows)
    n_manifest = sum(int(r.get("n_passages") or 0) for r in rows.values())
    n_index = db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    if missing or extra or n_manifest != n_index:
        say(BAD, "index does not match the library",
            f"{len(missing)} document(s) not indexed, {len(extra)} indexed but "
            f"absent from the manifest, {n_index:,} indexed passages against "
            f"{n_manifest:,} in the manifest. Rebuild with: python ops/people.py reindex")
    else:
        say(OK, "index matches the library",
            f"{len(rows):,} documents, {n_index:,} passages")


if __name__ == "__main__":
    sys.exit(main())
