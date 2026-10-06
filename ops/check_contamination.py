"""Check configured own-corpus contaminants and keepers.

The public script is config-driven: private seat names and paper titles belong in
config/contamination.yaml, not in code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from expertwins import paths  # noqa: E402

CONFIG = paths.CONFIG / "contamination.yaml"


def _example() -> str:
    return (
        "No config/contamination.yaml found; nothing to check.\n\n"
        "Create one as:\n"
        "  contaminants:\n"
        "    seat_name: [title fragment that must be absent]\n"
        "  keepers:\n"
        "    seat_name: [title fragment that must be present]\n"
    )


def _load() -> tuple[dict[str, list[str]], dict[str, list[str]]] | None:
    if not CONFIG.exists():
        print(_example())
        return None
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{CONFIG}: expected top-level mapping")
    def norm(key: str) -> dict[str, list[str]]:
        raw = data.get(key) or {}
        if not isinstance(raw, dict):
            raise SystemExit(f"{CONFIG}: {key!r} must map seat -> [title fragments]")
        out: dict[str, list[str]] = {}
        for seat, needles in raw.items():
            if isinstance(needles, str):
                needles = [needles]
            if not isinstance(needles, list):
                raise SystemExit(f"{CONFIG}: {key}.{seat} must be a list")
            out[str(seat)] = [str(n).lower() for n in needles]
        return out
    return norm("contaminants"), norm("keepers")


def _library():
    if not paths.LIBRARY.exists():
        print(f"No library at {paths.LIBRARY}; treating all own corpora as empty.")
        return None
    from expertwins.corpus.store import Library  # noqa: PLC0415
    return Library(paths.LIBRARY)


def _titles_for(seat: str, lib) -> list[str]:
    own_path = paths.PEOPLE / f"{seat}.own.json"
    if not own_path.exists():
        print(f"  no {paths.relative(own_path)}; own corpus is empty")
        return []
    try:
        accepted = json.loads(own_path.read_text(encoding="utf-8")).get("accepted", [])
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read {paths.relative(own_path)}: {exc}")
        return []
    if lib is None:
        return []
    titles: list[str] = []
    for doc_id in accepted:
        try:
            titles.append((lib.load_meta(doc_id).title or "").lower())
        except Exception:  # noqa: BLE001
            pass
    return titles


def main() -> int:
    loaded = _load()
    if loaded is None:
        return 0
    contaminants, keepers = loaded
    lib = _library()
    bad = 0
    missing = 0
    for seat in sorted(set(contaminants) | set(keepers)):
        print(f"\n=== {seat} ===")
        titles = _titles_for(seat, lib)
        print(f"  {len(titles)} own-paper titles available")
        for needle in contaminants.get(seat, []):
            hits = [t for t in titles if needle in t]
            if hits:
                bad += len(hits)
                for hit in hits:
                    print(f"  STILL CONTAMINATED [{needle}]: {hit[:76]}")
            else:
                print(f"  absent  [{needle}]")
        for needle in keepers.get(seat, []):
            hits = [t for t in titles if needle in t]
            if not hits:
                missing += 1
            print(f"  {'kept   ' if hits else 'MISSING'} [{needle}]"
                  + (f": {hits[0][:64]}" if hits else ""))
    print(f"\ncontaminants still present: {bad}")
    print(f"keepers missing: {missing}")
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
