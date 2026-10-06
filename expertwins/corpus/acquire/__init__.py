"""Literature acquisition. Online only — the read path never comes here."""

from .europepmc import EuropePMC
from .ncbi import NCBI
from .pipeline import Acquisition, AcquireStats

__all__ = ["EuropePMC", "NCBI", "Acquisition", "AcquireStats"]
