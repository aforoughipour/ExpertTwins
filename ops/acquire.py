"""Fetch literature into a library root using the acquisition pipeline in `expertwins.corpus`.

DESIGN NOTE. Acquisition delegates fetching, extraction and storage to the
corpus layer. Two corpus-store checks are central:

  * `classify_fulltext` -- the store measures the stored bytes and decides
    whether a document is full text. No caller may assert it.
  * `title_appears_in_text` -- the store checks whether the fetched text appears
    to belong to the requested title rather than to a citing or related paper.
    Hash integrity and bibliographic integrity are different properties.

WRITES TO A SEPARATE ROOT ON PURPOSE. Parallel acquisition runs must not share
an append-only manifest: two processes appending to one manifest interleave
lines and tear rows. Merge afterwards through the ordinary `put()` path so every
row is re-derived from stored bytes rather than copied from a source manifest.

Usage:
    python ops/acquire.py --config config/_acquire_x.yaml --root library_seat_x
    python ops/acquire.py --config config/acquire_example.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from expertwins import paths  # noqa: E402



def iter_nodes(cfg: dict, only_group: str | None):
    for group in cfg.get("groups", []):
        if only_group and group["id"] != only_group:
            continue
        for node in group.get("nodes", []):
            yield group["id"], node


def cmd_dry_run(cfg: dict, only_group: str | None) -> int:
    total_q = total_floor = 0
    print(f"{'node':<30s} {'min_full':>8s}  probe / queries")
    print("-" * 100)
    for _gid, node in iter_nodes(cfg, only_group):
        total_q += len(node.get("queries", []))
        total_floor += node.get("min_full", 0)
        print(f"{node['id']:<30s} {node.get('min_full', 0):>8d}  "
              f"probe: {node.get('probe','')!r}")
        for q in node.get("queries", []):
            print(f"{'':<40s}{q}")
    print("-" * 100)
    print(f"{total_q} queries, committed full-text floor {total_floor} documents")
    print("\nFloors are committed BEFORE acquisition so a thin node is a finding.")
    return 0


def cmd_acquire(cfg: dict, only_group: str | None, root: Path,
                per_pass: int) -> int:
    from expertwins.corpus.acquire.pipeline import Acquisition
    from expertwins.corpus.store import Library
    from expertwins.corpus.net import Fetcher

    root.mkdir(parents=True, exist_ok=True)
    library = Library(root)
    fetcher = Fetcher(cache_dir=paths.CACHE)
    run_dir = paths.RUNS / f"acquire-{root.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    acq = Acquisition(library, fetcher, run_dir=run_dir)

    summary: dict[str, dict] = {}
    column = cfg.get("column", "")
    for gid, node in iter_nodes(cfg, only_group):
        node_id = node["id"]
        print(f"\n=== {node_id}  (floor {node.get('min_full', 0)}) ===", flush=True)
        for query in node.get("queries", []):
            print(f"  query: {query}", flush=True)
            before = acq.stats.added + acq.stats.replaced
            topics = [node_id, f"grp:{gid}"] + ([column] if column else [])
            try:
                acq.harvest_query(query, topics=topics, per_pass=per_pass)
                gained = acq.stats.added + acq.stats.replaced - before
                print(f"    +{gained} docs | {acq.stats.render()}", flush=True)
                s = summary.setdefault(node_id, {"added": 0, "queries": 0})
                s["added"] += gained
                s["queries"] += 1
            except Exception as exc:                              # noqa: BLE001
                # Reported, never swallowed. A failed query that looks like an
                # empty query is exactly the class of bug this project exists to
                # stop: absence and failure must be different types.
                print(f"    QUERY FAILED: {type(exc).__name__}: {exc}", flush=True)
                summary.setdefault(node_id, {"added": 0, "queries": 0}) \
                    .setdefault("failed_queries", []).append(
                        {"query": query, "error": f"{type(exc).__name__}: {exc}"})

    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {paths.relative(out)}")

    print("\nFLOOR CHECK (committed before acquisition)")
    print(f"{'node':<30s} {'added':>7s} {'floor':>7s}  status")
    shortfall = 0
    for _gid, node in iter_nodes(cfg, only_group):
        got = summary.get(node["id"], {}).get("added", 0)
        floor = node.get("min_full", 0)
        if got < floor:
            shortfall += 1
        print(f"{node['id']:<30s} {got:>7d} {floor:>7d}  "
              f"{'ok' if got >= floor else 'THIN'}")
    if shortfall:
        print(f"\n{shortfall} node(s) below the committed floor. That is a "
              f"FINDING, not a failure: it says this literature is not reachable "
              f"through this channel, and the seat stays nominal until it is.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--group", default=None)
    ap.add_argument("--root", required=True, help="SEPARATE root; merge afterwards")
    ap.add_argument("--per-pass", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(paths.resolve(a.config).read_text(encoding="utf-8"))
    if a.dry_run:
        return cmd_dry_run(cfg, a.group)
    return cmd_acquire(cfg, a.group, paths.resolve(a.root), a.per_pass)


if __name__ == "__main__":
    sys.exit(main())
