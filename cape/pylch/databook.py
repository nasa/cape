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
from ..dkit import tsvfile


# Target databook class
class DBTarget(cdbook.DBTarget):
    pass


# Databook for one component
class DBFM(cdbook.DBFM):
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


class DBProp(cdbook.DBProp):
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


class DBPyFunc(cdbook.DBPyFunc):
    pass


class DBTS(cdbook.DBTS):
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
    :Versions:
        * 2024-09-30 ``@sneuhoff``: v1.0;
    """

    # Minimal list of columns (the global ones like flowres + comps)
    # Most of these also have cp/cv, like "cd","cdp","cdv" for
    # pressure and viscous
    _base_cols = (
        "i",
        "solver_iter",
        "CL",
        "CD",
        "CA",
        "CY",
        "CN",
        "CLL",
        "CLM",
        "CLN",
    )
    # Minimal list of "coeffs" (each comp gets one)
    _base_coeffs = (
        "CL",
        "CD",
        "CA",
        "CY",
        "CN",
        "CLL",
        "CLM",
        "CLN",
    )

    # List of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        :Call:
            >>> filelist = fm.get_filelist()
        :Inputs:
            *prop*: :class:`CaseFM`
                Component iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read to construct iterative history
        :Versions:
            * 2024-09-18 ``@sneuhoff``: v1.0
        """
        # Name of (single) file
        return ["data.iter"]


# Class to keep track of residuals
class CaseResid(cdbook.CaseResid):
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
    _fm_cls = DBFM
    _ts_cls = DBTS
    _prop_cls = DBProp
    _pyfunc_cls = DBPyFunc
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
    def GetCurrentIter(self):
        r"""Determine iteration number of current folder

        :Call:
            >>> n = db.GetCurrentIter()
        :Inputs:
            *db*: :class:`DataBook`
                Databook for one run matrix
        :Outputs:
            *n*: :class:`int` | ``None``
                Iteration number
        :Versions:
            * 2024-09-18 ``@sneuhoff``: v1.0
            * 2024-10-11 ``@ddalle``: v1.1; use ``DataIterFile``
        """
        try:
            db = DataIterFile(meta=True)
            return db.n
        except Exception:
            return None

  # >

