r"""
:mod:`cape.cfd.dex`: Exchange data b/w CFD case and DataBook
=============================================================

"""

# Standard library
import os
from typing import Any

# Third-party
import numpy as np

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
        # Initialize
        DataKit.__init__(self)
        #: :class:`cape.cfdx.cntl.Cntl`
        #: Run matrix controller
        self.cntl = cntl
        #: :class:`str`
        #: Name of DataBook component
        self.comp = comp
        #: :class:`str`
        #: DataBook component type
        self.comptype = cntl.opts.get_DataBookType(comp)
        #: :class:`str
        #: Path to DataBook
        self.rootdir = cntl.opts.get_DataBookFolder()
        # Absolutize
        if not os.path.isabs(self.rootdir):
            self.rootdir = os.path.join(cntl.RootDir, self.rootdir)
        # Save for DataKit
        self.fdir = self.rootdir
        #: :class:`list`\ [:class:`str`]
        #: List of input columns to identify unique cases
        self.xcols = self.get_xcols()
        # File name ... TEMPORARY
        self.fname = f"{self._prefix}_{comp}.csv"
        # Absolute file
        absfile = os.path.join(self.rootdir, self.fname)
        # Read the file
        if os.path.isfile(absfile):
            DataKit.__init__(self, absfile)
        # Initialize any missing columns
        self.init_empty()

  # *** DATA ***
   # --- Initialize ---
    def init_empty(self):
        r"""Initialize required columns, if necessary

        :Call:
            >>> db.init_empty(col)
        :Inputs:
            *db*: :class:`DataExchanger`
                Data container customized for collecting CFD data
        :Versions:
            * 2025-08-05 ``@ddalle``: v1.0
        """
        # Get run matrix controller
        cntl = self.cntl
        # Initialize run matrix columns
        for col in cntl.x.cols:
            # Get reference value
            v = self.cntl.x[col][0]
            # Initialize
            self.init_col_like(col, v)
        # Initialzie output columns
        for col in self.get_datacols():
            # Initialize as float
            self.init_col_like(col, 0.0)
        # Add integer-like status columns
        for col in cntl.opts.get_DataBookIntCols(self.comp):
            # Initialize as int
            self.init_col_like(col, 0)
        # Add float-like status columns
        for col in cntl.opts.get_DataBookFloatCols(self.comp):
            # Initialzie as float
            self.init_col_like(col, 0.0)

    def init_col_like(self, col: str, v: Any):
        r"""Initialize a data column, if necessary

        :Call:
            >>> db.init_col_like(col, v)
        :Inputs:
            *db*: :class:`DataExchanger`
                Data container customized for collecting CFD data
            *col*: :class:`str`
                Name of column
            *v*: :class:`float` | :class:`int` | :class:`str`
                Reference value for data type of new column
        :Versions:
            * 2025-08-05 ``@ddalle``: v1.0
        """
        # Check if present
        if col in self:
            return
        # Check data type
        if isinstance(v, str):
            self.save_col(col, [])
            return
        else:
            self.save_col(col, np.zeros(0, np.asarray(v).dtype))

   # --- Merge ---
    def merge(self, db: DataKit):
        r"""Combine data w/o duplication

        This overwrites :func:`DataKit.merge` by using ``"nStats"`` as
        the status column.

        :Call:
            >>> db.merge(dbnew)
        :Inputs:
            *db*: :class:`DataExchanger`
                Data container customized for collecting CFD data
            *dbnew*: :class:`DataKit`
                Additional data container to merge data from
        :Versions:
            * 2025-07-24 ``@ddalle``: v1.0
        """
        DataKit.merge(self, db, statuscol="nIter")

   # --- Column lists ---
    def get_xcols(self) -> list:
        r"""Get list of cols to distinguish unique cases

        :Call:
            >>> xcols = db.get_xcols()
        :Inputs:
            *db*: :class:`DataExchanger`
                Data container customized for collecting CFD data
        :Outputs:
            *xcols*: :class:`list`\ [:class:`str`]
                List of identifying columns for each case
        :Versions:
            * 2025-08-05 ``@ddalle``: v1.0
        """
        # Handle to run matrix
        x = self.cntl.x
        # Set list of value columns
        xcols = []
        # Loop through list of columns
        for col in x.cols:
            # Get data type
            dtype = x.GetKeyDType(col)
            # Check if it shows up in name
            defn = x.defns.get(col, {})
            qname = defn.get("Label", False)
            # Check if we should include this in identifier cols
            if (dtype != "str") or qname:
                xcols.append(col)
        # Output
        return xcols

    def get_datacols(self) -> list:
        r"""Get list of columns to extract from CFD results

        :Call:
            >>> ycols = db.get_datacols()
        :Inputs:
            *db*: :class:`DataExchanger`
                Data container customized for collecting CFD data
        :Outputs:
            *ycols*: :class:`list`\ [:class:`str`]
                List of data columns
        :Versions:
            * 2025-08-05 ``@ddalle``: v1.0
        """
        # Initialize output
        cols = []
        # Run matrix controller options
        opts = self.cntl.opts
        # Get key data columns
        ycols = opts.get_DataBookCols(self.comp)
        cols.extend(ycols)
        # Add statistics columns if anny
        for ycol in ycols:
            # Get statistics columns
            scols = opts.get_DataBookColStats(self.comp, ycol)
            # Loop through those
            for suffix in scols:
                # Skip 'mu' (mean value)
                if suffix == 'mu':
                    continue
                # Full column name
                cols.append(f"{ycol}_{suffix}")
        # Output
        return cols

  # *** FILES ***
   # --- Folders ---
    def mkdirs(self):
        # Get components of path
        parts = self.rootdir.split(os.sep)
        # Cumulative path
        dirname = parts[0] + os.sep
        # Loop through parts
        for part in parts[1:]:
            # Join path
            dirname = os.path.join(dirname, part)
            # Create if necessary
            if not os.path.isdir(dirname):
                os.mkdir(dirname)

