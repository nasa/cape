r"""
:mod:`cape.pyvul.inpfile`: Interface to VULCAN-CFD input files
=======================================================================

This module provides the class :class:`VulcanInpFile`, which is used to
parse and modify the main VULCAN-CFD input file
"""

# Standard library
import os
import re
from io import IOBase, StringIO
from typing import Any, Optional

# Third-party

# Local imports
from ..errors import assert_isinstance


# Other constants
SPECIAL_CHARS = "{}[]:=,;"

# Regular expressions
RE_FLOAT = re.compile(r"[+-]?[0-9]+\.?[0-9]*([EDed][+-]?[0-9]+)?")
RE_INT = re.compile(r"[+-]?[0-9]+")
RE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_ ]*")
RE_MULTIPLE = re.compile(r"(?P<grp>.+)\s*\^\s*(?P<exp>[0-9]+)")


# Base class
class VulcanInpFile(dict):
    r"""Interface to LAVA-Cartesian input files

    :Call:
        >>> inp = CartInputFile(fname=None)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            Name of input file
    :Outputs:
        *inp*: :class:`CartInputFile`
            Interface to one LAVA-Cartesian input file
    """
   # --- Class attributes ---
    __slots__ = (
        "fdir",
        "fname",
        "tab",
        "_section",
    )

   # --- __dunder__ ---
    def __init__(self, fname: Optional[str] = None):
        self.fdir = None
        self.fname = None
        self.tab = "    "
        # Initialize hidden attributes
        self._section = ""
        # Process up to one arg
        if isinstance(fname, str):
            # Read namelist
            self.read_inpfile(fname)
