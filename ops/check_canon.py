"""Check whether configured landmark titles are present in the corpus.

This script is intentionally data-free. Public releases should not hard-code a
private panel's canon; put project-specific landmarks in config/canon_check.yaml
instead.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from expertwins import paths  # noqa: E402

CONFIG = paths.CONFIG / "canon_check.yaml"


def _example() -> str:
    return (
        "No config/canon_check.yaml found; nothing to check.\n\n"
        "Create one as:\n"
        "  group name:\n"
        "    title fragment in lowercase: Human-readable label\n"
    )


def _load() -> dict[str, dict[str, str]] | None:
    if not CONFIG.exists():
        print(_example())
        return None
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{CONFIG}: expected mapping of group -> title_fragment -> label")
    out: dict[str, dict[str, str]] = {}
    for group, items in data.items():
        if not isinstance(items, dict):
            raise SystemExit(f"{CONFIG}: group {group!r} must be a mapping")
        out[str(group)] = {str(k).lower(): str(v) for k, v in items.items()}
    return out


def _connect() -> sqlite3.Connection | None:
    if not paths.INDEX.exists():
        print(f"No index at {paths.INDEX}; treating configured canon as missing.")
        return None
    return sqlite3.connect(f"file:{paths.INDEX.as_posix()}?mode=ro", uri=True)


def main() -> int:
    canon = _load()
    if canon is None:
        return 0
    db = _connect()
    missing: list[str] = []
    total = sum(len(v) for v in canon.values())
    for group, items in canon.items():
        print(f"\n=== {group} ===")
        for needle, label in items.items():
            rows = []
            if db is not None:
                rows = db.execute(
                    "SELECT title FROM docs WHERE lower(title) LIKE ? LIMIT 1",
                    (f"%{needle}%",),
                ).fetchall()
            if rows:
                print(f"  held    {label:<24} {rows[0][0][:62]}")
            else:
                missing.append(f"{label} ({needle})")
                print(f"  MISSING {label:<24} {needle}")
    if db is not None:
        db.close()
    print(f"\n{len(missing)} of {total} configured canonical papers absent")
    for item in missing:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
