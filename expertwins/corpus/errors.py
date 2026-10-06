"""Typed errors and typed absences.

The dangerous state is not missing data. It is a record that *asserts* data is
present when it is not. A parser reporting absence and a thing being absent can
look identical unless the type system keeps them apart.

Nothing is allowed to return a bare empty list to mean "I looked and found
nothing". A search that finds nothing returns an :class:`Absence`, which
carries what was searched, how, and how much was examined. A search that could
not run raises. The two are different types and cannot be confused by a caller
that forgot to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class CorpusError(Exception):
    """Base for every error this package raises deliberately."""


class OfflineError(CorpusError):
    """A network path was attempted while EXPERTWINS_OFFLINE=1.

    Raised loudly rather than degrading silently, so that air-gapped operation
    is a tested configuration and not a surprise.
    """


class IntegrityError(CorpusError):
    """The store disagrees with itself: a manifest row that reality contradicts."""


class ExtractionError(CorpusError):
    """A document could not be turned into text. Distinct from 'has no text'."""


class QuarantineError(CorpusError):
    """A document is not admissible to the library (e.g. it cannot be dated)."""


@dataclass(frozen=True)
class Absence:
    """A verified negative result.

    This is what "nothing found" looks like when it is an *observation* rather
    than a shrug. Every field is required because it is part of the evidence
    needed to audit a parser's negative result.

    Attributes:
        what: plain description of what was sought.
        method: the concrete mechanism used (endpoint, regex name, parser).
        query: the literal query or pattern issued.
        examined: how many units (records, pages, characters) were inspected.
                  An absence reported after examining 0 units is not evidence
                  of absence, and :meth:`is_evidence` says so.
        detail: anything else worth keeping for the audit trail.
    """

    what: str
    method: str
    query: str
    examined: int
    detail: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """An Absence is always falsey, so `if result:` reads naturally."""
        return False

    def is_evidence(self) -> bool:
        """True only if something was actually inspected.

        An absence reported after a parser examines zero units is not evidence
        of anything; it only records that the attempted method did not inspect
        usable material.
        """
        return self.examined > 0

    def render(self) -> str:
        stance = "evidence of absence" if self.is_evidence() else "UNINFORMATIVE (examined 0)"
        return (
            f"no {self.what} via {self.method} "
            f"(query={self.query!r}, examined={self.examined}) — {stance}"
        )
