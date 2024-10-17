# -*- coding: utf-8 -*-
r"""
:mod:`cape.pylch.databook`: Loci/CHEM data book module
=====================================================

This module provides interfaces to the various CFD outputs tracked by
the :mod:`cape` package. These versions are specific to Loci/CHEM.

"""

# Standard library

# Third-party imports

# Local imports
from ..cfdx import databook as cdbook
from ..dkit import basedata


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
  # ===========
  # Readers
  # ===========
  # <
    # Initialize a DBComp object
    def ReadDBComp(self, comp: str, check: bool = False, lock: bool = False):
        r"""Initialize data book for one component

        :Call:
            >>> db.ReadDBComp(comp, check=False, lock=False)
        :Inputs:
            *db*: :class:`DataBook`
                Databook for one run matrix
            *comp*: :class:`str`
                Name of component
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Versions:
            * 2024-10-17 ``@ddalle``: v1.0
        """
        # Read the data book
        self[comp] = DBComp(
            comp, self.cntl,
            targ=self.targ, check=check, lock=lock)
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
    def ReadCaseFM(self, comp: str):
        r"""Read a :class:`CaseFM` object

        :Call:
            >>> fm = db.ReadCaseFM(comp)
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
  # >


# Target databook class
class DBTarget(cdbook.DBTarget):
    pass


# Databook for one component
class DBComp(cdbook.DBComp):
    pass


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


