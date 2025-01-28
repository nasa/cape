r"""
:mod:`cape.sequtils`: Tools to analyze 1D iterative histories
================================================================

This module contains functions that gather statistics and select an
averaging window of CFD iterative histories. These sequences are very
similar to time series but can have some subtle differences due to
variable time steps in CFD results.
"""

# Standard library
from typing import Optional

# Third-party
import numpy as np

# Local imports
from .argread._vendor.kwparse import KwargParser


# Type arrays
INT_TYPES = (int, np.integer)


# Options for processing
class SequenceKwargs(KwargParser):
    # No attributes
    __slots__ = ()

    # Permissible options
    _optlist = (
        "DeltaN",
        "NFirst",
        "NLast",
        "NMax",
        "NStats",
    )

    # Aliases
    _optmap = {
        "NAvg": "NStats",
        "NMin": "NFirst",
        "NMaxStats": "NMax",
        "NStatsMax": "NMax",
        "dN": "DeltaN",
        "deltaN": "DeltaN",
        "dn": "DeltaN",
        "nAvg": "NStats",
        "nFirst": "NFirst",
        "nLast": "NLast",
        "nMax": "NMax",
        "nMaxStats": "NMax",
        "nMin": "NFirst",
        "nStats": "NStats",
        "nStatsMax": "NMax",
        "navg": "NStats",
        "nfirst": "NFirst",
        "nlast": "NLast",
        "nmax": "NMax",
        "nmaxstats": "NMax",
        "nmin": "NFirst",
        "nstats": "NStats",
        "nstatsmax": "NMax",
    }

    # Types
    _opttypes = {
        "DeltaN": INT_TYPES,
        "NFirst": INT_TYPES,
        "NLast": INT_TYPES,
        "NMax": INT_TYPES,
        "NStats": INT_TYPES,
    }


# Iterative histories
class Iter1D(object):
    # Attributees
    __slots__ = (
        "i",
        "v",
        "description",
    )

    # Initialize
    def __init__(
            self,
            v: np.ndarray,
            i: Optional[np.ndarray] = None,
            description: Optional[str] = None):
        # Initialize iterations
        if i is None:
            i = np.arange(v.size)
        # Check sizes
        if i.size != v.size:
            raise ValueError(
                f"Iterative history for {description} has mismatching size; "
                f"i: {i.size}; v: {v.size}")
        # Save
        self.i = i
        self.v = v

