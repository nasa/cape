r"""
:mod:`cape.pylch.fmfile`: Read force & moment histories
=======================================================================

This module provides the class :mod:`ChemFMFile` which reads both the
dimensional forces and moments (from two separate files).

Loci/CHEM only reports dimensional force and moment histories and has no
concept of a reference length or reference area, so CAPE is used to
acquire the same.
"""

# Standard library

# Third-party

# Local imports
from ..dkit.textdata import TextDataFile


# Base class
class ChemFMFile(TextDataFile):

    pass

    # Get path to main CAPE file

