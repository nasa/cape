r"""
:mod:`cape.cfdx.triqfm`: Module for reading patch loads from one case
=======================================================================



"""


# Standard library
import os
from typing import Optional

# Third-party
import numpy as np

# Local imports
from ..dkit.rdb import DataKit


# Main class for TriqFM cases
class CaseTriqFM(DataKit):
    __slots__ = (
        "comp",
        "ftriq",
    )

    def __init__(self, comp: str, ftriq: str):
        # Save the component name
        self.comp = comp
        # Save the name of the TriQ file
        self.ftriq = ftriq

