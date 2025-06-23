# -*- coding: utf-8 -*-
r"""
:mod:`cape.pylch.databook`: Loci/CHEM data book module
=====================================================

This module provides interfaces to the various CFD outputs tracked by
the :mod:`cape` package. These versions are specific to Loci/CHEM.

"""

# Standard library
import os

# Third-party imports
import numpy as np

# Local imports
from ..cfdx import databook as cdbook
from ..cfdx.cntl import Cntl
from ..dkit import tsvfile
from ..fileutils import tail


# Constatns
FCOL_PAIRS = (
    ("Fx", "CA"),
    ("Fy", "CY"),
    ("Fz", "CN"),
)

MCOL_PAIRS = (
    ("Mx", "CLL"),
    ("My", "CLM"),
    ("Mz", "CLN"),
)


# Target databook class
class TargetDataBook(cdbook.TargetDataBook):
    pass


# Databook for one component
class FMDataBook(cdbook.FMDataBook):
    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *db*: :class:`DataBook`
                Databook for one run matrix
        :Outputs:
            *H*: :class:`CaseResid`
                Residual history
        :Versions:
            * 2024-09-30 ``@sneuhoff``: v1.0
        """
        # Read CaseResid object from PWD
        return CaseResid()

    # Read case FM history
    def ReadCase(self, comp: str):
        r"""Read a :class:`CaseFM` object

        :Call:
            >>> fm = db.ReadCase(comp)
        :Inputs:
            *db*: :class:`DataBook`
                Databook for one run matrix
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *fm*: :class:`CaseFM`
                Force and moment history
        :Versions:
            * 2024-09-30 ``@sneuhoff``: v1.0
        """
        # Read CaseResid object from PWD
        return CaseFM(comp)


class PropDataBook(cdbook.PropDataBook):
    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.pyfun.databook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid(self.proj)


class PyFuncDataBook(cdbook.PyFuncDataBook):
    pass


class TimeSeriesDataBook(cdbook.TimeSeriesDataBook):
    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.pyfun.databook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid(self.proj)


# Iterative F&M history
class CaseFM(cdbook.CaseFM):
    r"""Iterative force & moment history for one component, one case

    :Call:
        >>> fm = CaseFM(comp=None)
    :Inputs:
        *comp*: :class:`str`
            Name of component
    :Outputs:
        *fm*: :class:`CaseFM`
            One-case iterative history
    """

    # Minimal list of columns (the global ones like flowres + comps)
    # Most of these also have cp/cv, like "cd","cdp","cdv" for
    # pressure and viscous
    _base_cols = (
        "i",
        "t",
        "solver_iter",
        "mdot",
        "Fx",
        "Fy",
        "Fz",
        "Mx",
        "My",
        "Mz",
        "edot",
    )
    # Minimal list of "coeffs" (each comp gets one)
    _base_coeffs = (
        "mdot",
        "Fx",
        "Fy",
        "Fz",
        "Mx",
        "My",
        "Mz",
        "edot",
    )

    # List of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        :Call:
            >>> filelist = fm.get_filelist()
        :Inputs:
            *fm*: :class:`CaseFM`
                Component iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read to construct iterative history
        :Versions:
            * 2025-05-20 ``@ddalle``: v1.0
        """
        # Name of (single) file
        return [os.path.join("output", f"flux_{self.comp}.dat")]

    # Read a force file and its moment pairing
    def readfile(self, fname: str) -> tsvfile.TSVFile:
        r"""Read dimensional force & moment history

        :Call:
            >>> fm.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Component iterative history instance
            *fname*: :class:`str`
                Name of force history file to read
        :Versions:
            * 2025-05-20 ``@ddalle``: v1.0
        """
        # Read flux_{comp}.dat file
        db = tsvfile.TSVFile(
            fname, Translators={
                "col1": "i",
                "col2": "t",
                "col3": "mdot",
                "col4": "Fx",
                "col5": "Fy",
                "col6": "Fz",
                "col7": "edot",
                "col8": "area",
            })
        # Read moments if available
        self.read_moments(db)
        # Output
        return db

    def read_moments(self, db: tsvfile.TSVFile):
        r"""Read dimensional force & moment history

        :Call:
            >>> fm.read_moments(db)
        :Inputs:
            *fm*: :class:`CaseFM`
                Component iterative history instance
            *db*: :class:`tsvfile.TSVFile`
                Force-only iterative history
        :Versions:
            * 2025-05-20 ``@ddalle``: v1.0
        """
        # Name of moments file
        fname = os.path.join("output", f"moment_{self.comp}.dat")
        # Check for it
        if not os.path.isfile(fname):
            return
        # Read it
        mdb = tsvfile.TSVFile(
            fname, Translators={
                "col1": "i",
                "col2": "t",
                "col3": "Mx",
                "col4": "My",
                "col5": "Mz",
                "col6": "xMRP",
                "col7": "yMRP",
                "col8": "zMRP",
            })
        # Merge
        for col in ("Mx", "My", "Mz"):
            db.save_col(col, mdb[col])

    def normalize_by_cntl(self, cntl: Cntl, i: int):
        r"""Normalize a force & moment history using run matrix

        :Call:
            >>> fm.normalize_by_cntl(cntl, i)
        :Inputs:
            *fm*: :class:`CaseFM`
                Component iterative history instance
            *cntl*: :class:`Cntl`
                Run matrix control instance for this case
            *i*: :class:`int`
                Index of this case in run matrix
        :Versions:
            * 2025-05-23 ``@ddalle``: v1.0
        """
        # Check if normalized
        if "CA" in self.cols:
            return
        # Get dynamic pressure
        q = cntl.x.GetDynamicPressure(i)
        # Get reference area
        aref = cntl.opts.get_RefArea(self.comp)
        lref = cntl.opts.get_RefLength(self.comp)
        # Normalize
        self.normalize_by_value(q, aref, lref)

    def normalize_by_value(self, q: float, aref: float, lref: float):
        r"""Normalize a force & moment history using reference values

        :Call:
            >>> fm.normalize_by_value(a, aref, lref)
        :Inputs:
            *fm*: :class:`CaseFM`
                Component iterative history instance
            *q*: :class:`float`
                Freestream dynamic pressure [Pa]
            *aref*: :class:`float`
                Reference area [m^2]
            *lref*: :class:`float`
                Reference length [m]
        :Versions:
            * 2025-05-23 ``@ddalle``: v1.0
        """
        # Check if normalized
        if "CA" in self.cols:
            return
        # Loop through force comps
        for fcol, ccol in FCOL_PAIRS:
            if fcol in self:
                self.save_col(ccol, self[fcol] / (q*aref))
        # Loop through moment components
        for mcol, ccol in MCOL_PAIRS:
            if mcol in self:
                self.save_col(ccol, self[mcol]/(q*aref*lref))


# Class to keep track of residuals
class CaseResid(cdbook.CaseResid):
    r"""Iterative residual history for one case

    :Call:
        >>> h = CaseResid()
    :Inputs:
        *comp*: :class:`str`
            Name of component
    :Outputs:
        *h*: :class:`CaseResid`
            One-case iterative history
    """

    # Default residula
    _default_resid = "R_r"
    # Base columns
    _base_cols = (
        "i",
        "R_r",
        "R_m",
        "R_e",
        "L2Resid",
    )
    _base_coeffs = (
        "R_r",
        "R_m",
        "R_e",
        "L2Resid",
    )

    # Get list of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        :Call:
            >>> filelist = h.get_filelist()
        :Inputs:
            *h*: :class:`CaseResid`
                Iterative residual history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read for residual history
        :Versions:
            * 2025-05-20 ``@ddalle``: v1.0
        """
        return [os.path.join("output", "resid.dat")]

    # Read residual history
    def readfile(self, fname: str) -> tsvfile.TSVFile:
        r"""Read a Loci/CHEM iterative history file

        :Call:
            >>> db = h.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-component iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`tsvfile.TSVFile`
                Data read from *fname*
        :Versions:
            * 2025-05-20 ``@ddalle``: v1.0
        """
        # Read the simple dat file
        db = tsvfile.TSVFile(
            fname, Translators={
                "col1": "i",
                "col2": "R_r",
                "col3": "R_m",
                "col4": "R_e",
            })
        # Build residual
        self._build_l2(db)
        # Output
        return db

    # Calculate L2 norm(s)
    def _build_l2(self, db: tsvfile.TSVTecDatFile, suf: str = ''):
        r"""Calculate *L2* norm for initial, final, and subiter

        :Versions:
            * 2025-05-20 ``@ddalle``: v1.0
        """
        # Column name for iteration
        icol = f"i{suf}"
        # Column name for initial residual/value
        rcol = f"L2Resid{suf}"
        # Get iterations corresponding to this sufix
        iters = db.get(icol)
        # Skip if not present
        if iters is None:
            return
        # Initialize cumulative residual squared
        L2squared = np.zeros_like(iters, dtype="float")
        # Loop through potential residuals
        for c in ("R_r", "R_m", "R_e"):
            # Check for baseline
            col = c + suf
            # Get values
            v = db.get(col)
            # Assemble
            if v is not None:
                L2squared += v*v
        # Save residuals
        db.save_col(rcol, np.sqrt(L2squared))


# Aerodynamic history class
class DataBook(cdbook.DataBook):
    r"""Primary databook class for LAVA

    :Call:
        >>> db = DataBook(x, opts)
    :Inputs:
        *x*: :class:`RunMatrix`
            Current run matrix
        *opts*: :class:`Options`
            Global CAPE options instance
    :Outputs:
        *db*: :class:`DataBook`
            Databook instance
    :Versions:
        * 2024-09-30 ``@sneuhoff``: v1.0
    """
    _fm_cls = FMDataBook
    _ts_cls = TimeSeriesDataBook
    _prop_cls = PropDataBook
    _pyfunc_cls = PyFuncDataBook
  # ===========
  # Readers
  # ===========
  # <
  # >

  # ========
  # Case I/O
  # ========
  # <
    # Current iteration status
    def GetCurrentIter(self) -> int:
        r"""Determine iteration number of current folder

        :Call:
            >>> n = db.GetCurrentIter()
        :Inputs:
            *db*: :class:`DataBook`
                Databook for one run matrix
        :Outputs:
            *n*: :class:`int`
                Iteration number
        :Versions:
            * 2025-05-20 ``@ddalle``: v1.0
        """
        # Check for resid file
        try:
            # Read last line of residual file
            line = tail(os.path.join("output", "resid.dat"))
            # Get iteration number
            return int(line.split(maxsplit=1)[0])
        except Exception:
            return 0

  # >

