r"""
:mod:`cape.sequtils`: Tools to analyze 1D iterative histories
================================================================

This module contains functions that gather statistics and select an
averaging window of CFD iterative histories. These sequences are very
similar to time series but can have some subtle differences due to
variable time steps in CFD results.
"""

# Standard library

# Third-party
import numpy as np

# Local imports
from .argread._vendor.kwparse import KwargParser


# Options for processing
class SequenceKwargs(KwargParser):
    # No attributes
    __slots__ = ()

    # Permissible options
    _optlist = (
        "NStats",
        "NStatsMax",
        "NMax",
        "NMin",
    )

    # Aliases
    _optmap = {
        "nAvg": "NStats",
    }
