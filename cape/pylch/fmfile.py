r"""
:mod:`cape.pylch.fmfile`: Read force & moment *coefficient* histories
=======================================================================

This module provides the class :mod:`ChemFMFile` which reads both the
dimensional forces and moments (from two separate files) and then
nondimensionalizes them using CAPE reference area settings, if possible.

Loci/CHEM only reports dimensional force and moment histories and has no
concept of a reference length or reference area, so CAPE is used to
acquire the same.
"""

# Standard library
import os

# Third-party

# Local imports
from .options.runctlopts import RunControlOpts
from ..dkit.textdata import TextDataFile


# Base class
class ChemFMFile(TextDataFile):

    # Read case settings
    def read_case_json(self) -> RunControlOpts:
        # Check if file is present
        if os.path.isfile("case.json"):

            return RunControlOpts("case.json")
        else:
            return RunControlOpts()
        
    # Get path to main CAPE file
    
