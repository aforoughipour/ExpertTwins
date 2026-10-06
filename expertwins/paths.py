"""Where things are -- resolved, never hard-coded.

No absolute path appears anywhere in the package. The installation is meant to
be copied onto an HPC filesystem and run there, so every location is derived
from a single root that can be overridden from the environment.

Resolution order for the project root:

    1. $EXPERTWINS_ROOT if set                    -- the HPC escape hatch
    2. the directory containing this file's parent (the repo checkout)

Everything else is derived from the root. Nothing in this module reads the
network, and nothing on the read path anywhere in the package does either.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Project root. Override with the EXPERTWINS_ROOT environment variable.
ROOT: Path = Path(os.environ.get("EXPERTWINS_ROOT") or Path(__file__).resolve().parents[1])

#: The merged corpus and its FTS index.
LIBRARY: Path = Path(os.environ.get("EXPERTWINS_LIBRARY") or ROOT / "library")
INDEX: Path = Path(os.environ.get("EXPERTWINS_INDEX") or LIBRARY / "index.sqlite")

CONFIG: Path = ROOT / "config"
PEOPLE: Path = CONFIG / "people"
SEATS: Path = CONFIG / "seats.yaml"
RUNS: Path = Path(os.environ.get("EXPERTWINS_RUNS") or ROOT / "runs")
CACHE: Path = Path(os.environ.get("EXPERTWINS_CACHE") or ROOT / ".cache")


def seat_library(name: str) -> Path:
    """The per-seat acquisition root.

    Parallel acquisition runs must write to SEPARATE roots: two processes
    appending to one manifest interleave lines and tear rows.
    """
    return ROOT / f"library_seat_{name}"


def relative(p: Path) -> str:
    """Render a path relative to ROOT when possible.

    Absolute paths written into a run manifest are what stop a run directory
    from being copied to another machine and re-read. Manifests store relative.
    """
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def resolve(p: str | Path) -> Path:
    """Inverse of :func:`relative` -- interpret a manifest path under ROOT."""
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def describe() -> str:
    lines = [f"EXPERTWINS_ROOT   {ROOT}"]
    for label, path in (("library", LIBRARY), ("index", INDEX), ("config", CONFIG),
                        ("runs", RUNS), ("cache", CACHE)):
        mark = "" if path.exists() else "   MISSING"
        lines.append(f"{label:<10s} {path}{mark}")
    return "\n".join(lines)
