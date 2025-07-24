r"""
:mod:`cape.cfd.dex`: Exchange data b/w CFD case and DataBook
=============================================================

"""

# Standard library
import os

# Local imports
from .cntlbase import CntlBase
from ..dkit.rdb import DataKit


# Base DataExchanger
class DataExchanger(DataKit):
  # *** CLASS ATTRIBUTES ***
   # --- Instance ---
    __slots__ = (
        "cntl",
        "comp",
        "comptype",
        "fname",
        "rootdir",
        "xcols",
    )

   # --- Component-type ---
    _mode = "1"
    _prefix = "aero"

  # *** DUNDER ***
    def __init__(self, cntl: CntlBase, comp: str):
        # Save basic attributes
        self.cntl = cntl
        self.comp = comp
        # Get type
        self.comptype = cntl.opts.get_DataBookType(comp)
        # Path to DataBook
        self.rootdir = cntl.opts.get_DataBookFolder()
        # File name ... TEMPORARY
        self.fname = f"{self._prefix}_{comp}.csv"
        # Absolute file
        absfile = os.path.join(self.rootdir, self.fname)
        # Read the file
        if os.path.isfile(absfile):
            DataKit.__init__(self, absfile)
        # Handleto run matrix
        x = cntl.x
        # Set list of value columns
        self.xcols = [
            k for k in x.cols
            if x.GetKeyDType(k) != "str"]

