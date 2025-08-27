# -*- coding: utf-8 -*-
r"""
:mod:`cape.pylava.databook`: LAVA data book module
=====================================================

This module provides LAVA-specific interfaces to the various CFD
outputs tracked by the :mod:`cape` package.

"""

# Standard library
import os

# Third-party imports

# Local imports
from .dataiterfile import DataIterFile
from ..cfdx import databook as cdbook
from ..dkit import basedata


# Iterative F&M history
class CaseFM(cdbook.CaseFM):
    r"""Iterative LAVA component force & moment history

    :Call:
        >>> fm = CaseFMCartesian(comp=None)
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
            * 2025-07-17 ``@ddalle``: v1.1; merge Curv & Cart
        """
        # Name of (single) file
        cartfile = os.path.join("monitor", "Cart.data.iter")
        # Check for such a file
        if os.path.isfile(cartfile):
            return [cartfile]
        else:
            return ["data.iter"]

    # Read a raw data file
    def readfile(self, fname: str) -> dict:
        r"""Read the data.iter

        :Call:
            >>> db = fm.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-component iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`dict`
                Data read from data.iter
        :Versions:
            * 2024-09-18 ``@sneuhoff``: v1.0
            * 2024-10-11 ``@ddalle``: v1.1; use ``DataIterFile``
        """
        # Read the data.iter
        data = DataIterFile(fname)
        # Unpack component name
        comp = self.comp
        # Initialize data for output
        db = basedata.BaseData()
        # Identify iteration column to use
        icol = "nt" if "nt" in data else "iter"
        # Force coeff prefix
        fpre = "c" if f"cx_{comp}" in data else "cf"
        # Save data
        db.save_col("i", data[icol])
        db.save_col("solver_iter", data[icol])
        db.save_col("CL", data[f"cl_{comp}"])
        db.save_col("CD", data[f"cd_{comp}"])
        db.save_col("CA", data[f"{fpre}x_{comp}"])
        db.save_col("CY", data[f"{fpre}y_{comp}"])
        db.save_col("CN", data[f"{fpre}z_{comp}"])
        db.save_col("CLL", data[f"cmx_{comp}"])
        db.save_col("CLM", data[f"cmy_{comp}"])
        db.save_col("CLN", data[f"cmz_{comp}"])
        # Output
        return db


# Iterative residual history
class CaseResid(cdbook.CaseResid):
    r"""Iterative residual history for one component, one case

    :Call:
        >>> hist = CaseResid(comp=None)
    :Inputs:
        *comp*: {``None``} | :class:`str`
            Name of component
    :Outputs:
        *hist*: :class:`CaseResid`
            One-case iterative history
    :Versions:
        * 2024-09-30 ``@sneuhoff``: v1.0;
    """

    # Initialization method
    def __init__(self, comp: str = "body1"):
        r"""Initialization method

        :Versions:
            * 2024-09-30 ``@sneuhoff``: v1.0
        """
        # Initialize attributes
        self.comp = comp
        # Call parent method
        cdbook.CaseResid.__init__(self)

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
            * 2024-09-24 ``@sneuhoff``: v1.0
        """
        return ["data.iter"]

    # Read a raw data file
    def readfile(self, fname: str) -> dict:
        r"""Read the data.iter for residuals

        :Call:
            >>> db = fm.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-component iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`BaseData`
                Data read from *fname*
        :Versions:
            * 2024-09-30 ``@sneuhoff``: v1.0
            * 2024-10-11 ``@ddalle``: v1.1; use ``DataiterFile``
        """
        # Read the data.iter
        data = DataIterFile(fname)
        # Initialize data for output
        db = basedata.BaseData()
        db.save_col("i", data["iter"])
        db.save_col("L2Resid", data["flowres"])
        # Output
        return db

