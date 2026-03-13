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
from numpy import ndarray

# Local imports
from .dataiterfile import DataIterFile
from ..cfdx import casedata
from ..dkit import basedata
from ..dkit.rdb import DataKit


# Iterative F&M history
class CaseFM(casedata.CaseFM):
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
        comp = self.comp.lower()
        # Initialize data for output
        db = basedata.BaseData()
        # Identify iteration column to use
        icol = "nt" if "nt" in data else "iter"
        # Identify time column to use
        tcol = "time" if "time" in data else None
        tcol = "ctu" if ((tcol is None) and ("ctu" in data)) else tcol
        # Force coeff prefix
        infix = '' if (f'cx_{comp}' in data or f"fx_{comp}" in data) else 'f'
        # Save data
        db.save_col("i", data[icol])
        db.save_col("solver_iter", data[icol])
        # Save time
        if tcol is not None:
            db.save_col("t", data[tcol])
        # Save CTUs
        if 'ctu' in data:
            db.save_col("ctu", data["ctu"])
        # Save cell count
        if "numcells" in data:
            db.save_col("numcells", data["numcells"])
        # Save coefficients
        db.save_col("CL", self.get_datacol(data, '', 'l'))
        db.save_col("CD", self.get_datacol(data, '', 'd'))
        db.save_col("CA", self.get_datacol(data, infix, 'x'))
        db.save_col("CY", self.get_datacol(data, infix, 'y'))
        db.save_col("CN", self.get_datacol(data, infix, 'z'))
        db.save_col("CLL", self.get_datacol(data, '', 'mx', ''))
        db.save_col("CLM", self.get_datacol(data, '', 'my', ''))
        db.save_col("CLN", self.get_datacol(data, '', 'mz', ''))
        # Output
        return db

    def get_datacol(
            self,
            data: dict,
            infix: str,
            coeff: str,
            prefix: str = 'f') -> ndarray:
        # Possible col names
        col = f"{coeff}_{self.comp.lower()}"
        col1 = f"c{infix}{col}"
        col2 = f"{prefix}{col}"
        # Use best
        if col1 in data:
            # Coefficient defined directly
            return data[col1]
        else:
            # Use force
            return data.get(col2)


# Iterative point probe history
class CasePointProbe(casedata.CasePointProbe):
    # No extra attributes
    __slots__ = ()

    # List of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        :Call:
            >>> filelist = probe.get_filelist()
        :Inputs:
            *prop*: :class:`CaseFM`
                Component iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read to construct iterative history
        :Versions:
            * 2026-03-11 ``@ddalle``: v1.0
        """
        # Confirm active runner
        if self.runner is None:
            raise TypeError(
                f"Cannot use cape.pylava {self.__class__.__name__} " +
                "without a CaseRunner instance")
        # Get case index
        j = self.runner.get_dex_opt(self.pt, "Index")
        # Pattern for file names
        pat = os.path.join("point_probe", rf"Cart\.[0-9]+\.pnt{j:04d}.dat")
        # Find those files
        filelist_raw = self.runner.search_regex(pat)
        # Sort them
        return sorted(filelist_raw)

    # Read a file
    def readfile(self, fname: str) -> dict:
        r"""Read the point probe

        :Call:
            >>> db = probe.readfile(fname)
        :Inputs:
            *probe*: :class:`CasePointProbe`
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
        # Read the data file
        db = DataKit(
            tsv=fname,
            translators={
                "nt_finest": "i",
                "time": "t",
                "Pressure": "p",
                "Temperature": "T",
                "UVel": "u",
                "VVel": "v",
                "WVel": "w",
            })
        # Output
        return db


# Iterative residual history
class CaseResid(casedata.CaseResid):
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
    def __init__(self, comp: str = "body1", meta=False, runner=False, **kw):
        r"""Initialization method

        :Versions:
            * 2024-09-30 ``@sneuhoff``: v1.0
            * 2026-03-12 ``@ddalle``: v1.1; add args and kwargs
        """
        # Initialize attributes
        self.comp = comp
        # Call parent method
        casedata.CaseResid.__init__(self, meta=meta, runner=runner, **kw)

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

