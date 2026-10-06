"""Repair passage ids that disagree with the document that holds them.

THE BUG. Rebuilding the index after merging multiple per-seat roots can fail
with

    sqlite3.IntegrityError: UNIQUE constraint failed: passages.passage_id

Diagnosis: a document stores passages whose ids name a DIFFERENT document. Two
documents, one set of passage ids.

The cause is slug disambiguation. When two papers reduce to the same
human-readable slug, the store appends a distinguishing suffix and writes the
document under the NEW id -- but the passages inside it kept the ids they were
built with, under the OLD one. The document directory, the manifest row and the
passage ids then disagree about which document this is.

WHY HASH CHECKS DO NOT CATCH IT. Every hash is correct: the bytes on disk are
exactly the bytes that were written. A document can hash correctly and still
contain text associated with a different identity -- *hash integrity and
identity integrity are different properties.* `Library.verify()` therefore
checks passage identity separately, and `ops/doctor.py` reports it.

It appears only when roots are MERGED, because the colliding pair has to coexist
in one index before the UNIQUE constraint can fire. ExpertTwins merges
acquisition roots, so the ids have to be checked at merge boundaries.

THE FIX, in the project's own idiom: DERIVE, NEVER ASSERT. A passage's id is not
something the passage gets to claim; it is a fact about which document holds it.
Every passage id is re-derived from its containing directory, and every
disagreement is reported so the scale of the defect is recorded rather than
silently smoothed away.

Usage:
    python ops/repair_passage_ids.py --scan     # report only
    python ops/repair_passage_ids.py --apply
    python ops/people.py reindex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expertwins import paths  # noqa: E402


def scan(root: Path, apply: bool) -> int:
    docs_dir = root / "docs"
    if not docs_dir.exists():
        print(f"no docs/ under {root}")
        return 1

    checked = mismatched = repaired = failed = 0
    examples: list[tuple[str, str]] = []

    for text_path in docs_dir.rglob("text.jsonl"):
        doc_id = text_path.parent.name
        checked += 1
        try:
            lines = text_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"  UNREADABLE {doc_id}: {exc}")
            continue

        rows: list[dict] = []
        bad = False
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A torn line is a DIFFERENT failure from a wrong id and is
                # reported as such rather than silently dropped.
                failed += 1
                bad = False
                rows = []
                break
            pid = row.get("passage_id", "")
            prefix = pid.split("#", 1)[0]
            if prefix != doc_id:
                bad = True
                suffix = pid.split("#", 1)[1] if "#" in pid else f"p{len(rows):04d}"
                row["passage_id"] = f"{doc_id}#{suffix}"
            rows.append(row)

        if bad and rows:
            mismatched += 1
            if len(examples) < 8:
                first = json.loads(lines[0]).get("passage_id", "") if lines else ""
                examples.append((doc_id, first))
            if apply:
                tmp = text_path.with_suffix(".jsonl.tmp")
                tmp.write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
                tmp.replace(text_path)
                repaired += 1

        if checked % 5000 == 0:
            print(f"  {checked} checked | mismatched {mismatched} "
                  f"repaired {repaired}", flush=True)

    print(f"\nchecked {checked} documents")
    print(f"passage ids disagreeing with their document: {mismatched}")
    print(f"repaired: {repaired}")
    print(f"unreadable/torn: {failed}")
    if examples:
        print("\nexamples (document -> the id its passages claimed):")
        for doc_id, first in examples:
            print(f"  {doc_id}  ->  {first}")
    if mismatched and not apply:
        print("\nThis was a scan. Re-run with --apply to repair, then reindex.")
    elif repaired:
        print("\nNow rebuild the index and WAIT FOR IT TO FINISH:")
        print("  python ops/people.py reindex")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="defaults to the merged library")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()
    root = paths.resolve(a.root) if a.root else paths.LIBRARY
    return scan(root, apply=a.apply and not a.scan)


if __name__ == "__main__":
    sys.exit(main())
