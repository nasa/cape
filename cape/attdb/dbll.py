# -*- coding: utf-8 -*-
r"""
This module provides customizations of :mod:`cape.attdb.rdb` that are
especially useful for launch vehicle line load databases.  The
force & moment coefficient names follow common missile naming
conventions:

    =======  ===============  ============================
    Name     Symbol           Description
    =======  ===============  ============================
    *CA*     :math:`C_A`      Axial force
    *CY*     :math:`C_Y`      Side force
    *CN*     :math:`C_N`      Normal force
    *CLL*    :math:`C_\ell`   Rolling moment
    *CLM*    :math:`C_m`      Pitching moment
    *CLN*    :math:`C_n`      Yawing moment
    =======  ===============  ============================

Line loads are typically defined for the force coefficients, and the
standard for these is to have *dCN* as a column name, which actually
represents the value

    .. math::

        \frac{\Delta C_N}{\Delta x/L_\mathit{ref}

In the limit of finer slices, this becomes
:math:`\mathrm{d}C_N}/\mathrm{d}x/L_\mathit{ref}`.  In other words, the
primary data is the derivative of a force coefficient with respect to
the non-dimensional *x*-coordinate.

"""

# Standard library modules
import os
import sys

# Third-party modules
import numpy as np

# CAPE modules
import cape.attdb.convert as convert
import cape.tnakit.kwutils as kwutils
import cape.tnakit.typeutils as typeutils

# Local modules
from . import rdb
from . import dbfm


# Basic coefficient list
_coeffs = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]

# Options class for adjustments
class _LL3XOpts(dbfm._FMEvalOpts):
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Lists ---
    # All options
    _optlist = {
        "CompLLAdjustedCols",
        "CompFMFracCols",
        "CompLLCols",
        "LLAdjustedCols"
        "LLCols",
        "SliceColSave",
        "fm",
        "mask",
        "method",
        "nPOD",
        "xcol"
    }

    # Alternate names
    _optmap = {
        "CompBasisCols": "CompLLBasisCols",
        "Method": "method",
        "NPOD": "nPOD",
        "acols": "CompLLAdjustedCols",
        "bcols": "CompLLBasisCols",
        "icols": "FMCols",
        "llcols": "CompLLCols",
        "npod": "nPOD",
    }

    # Sections
    _optlists = {
        "aweights": {
            "CompFMCols",
            "CompLLCols",
            "CompY",
            "CompZ",
            "Lref",
            "SliceColSave",
            "TargetCols",
            "method",
            "xMRP",
            "yMRP",
            "zMRP"
        },
        "aweights_make": {
            "CompFMCols",
            "CompFMFracCols",
            "CompLLCols",
            "CompY",
            "CompZ",
            "Lref",
            "TargetCols",
            "method",
            "xMRP",
            "yMRP",
            "zMRP"
        },
        "basis": {
            "SliceColSave",
            "mask",
            "method",
            "nPOD",
            "xcol"
        },
        "basis_": {
            "method",
            "nPOD",
            "xcol"
        },
        "basis_make": {
            "CompLLBasisCols",
            "mask",
            "method",
            "nPOD",
            "xcol"
        },
        "fractions": {
            "SliceColSave",
            "CompFMCols",
            "CompLLCols",
            "CompY",
            "CompZ",
            "Lref",
            "TargetCols",
            "method",
            "xMRP",
            "yMRP",
            "zMRP"
        },
        "fractions_make": {
            "CompFMCols",
            "CompFMFracCols",
            "CompLLCols",
            "CompY",
            "CompZ",
            "Lref",
            "TargetCols",
            "method",
            "xMRP",
            "yMRP",
            "zMRP"
        },
        "integrals": {
            "CompLLCols",
            "CompY",
            "CompZ",
            "Lref",
            "SliceColSave",
            "method",
            "xMRP",
            "yMRP",
            "zMRP"
        },
        "integrals_make": {
            "CompFMCols",
            "CompLLCols",
            "CompY",
            "CompZ",
            "Lref",
            "method",
            "xMRP",
            "yMRP",
            "zMRP"
        },
        "integrals_comp": {
            "LLCols",
            "Lref",
            "method",
            "xMRP",
            "yMRP",
            "y",
            "zMRP",
            "z"
        },
        "integrals_comp_make": {
            "FMCols",
            "LLCols",
            "Lref",
            "method",
            "xMRP",
            "yMRP",
            "y",
            "zMRP",
            "z"
        },
    }

   # --- Type ---
    # Required types
    _optytpes = {
        "CompFMCols": dict,
        "CompFMFracCols"
        "CompLLCols": dict,
        "CompLLAdjustedCols": dict,
        "LLCols": dict,
        "SliceColSave": typeutils.strlike,
        "method": typeutils.strlike,
        "nPOD": int,
        "xcol": typeutils.strlike,
    }

   # --- Defaults ---
    # Default values
    _rc = {
        "method": "left",
        "nPOD": 10,
    }
  # >


# Combine options
_LL3XOpts.combine_optdefs()


# Improvised SVD with fallback for too-dense data
def svd(C):
    r"""Calculate an SVD with a fallback to skipping every other column
    
    :Call:
        >>> U, s, V = svd(C)
    :Inputs:
        *C*: :class:`np.ndarray`\ [:class:`float`
            * shape=(*m*, *n*)
            Input data array
    :Outputs:
        *U*: :class:`np.ndarray`
            * shape=(*m*, *m*)
            Singular column vectors
        *s*: :class:`np.ndarray`
            * shape=(*m*, *n*) or shape=(*m*, *n*\ /2)
            Diagonal matrix of singular values
        *V*: :class:`np.ndarray`
            * shape=(*n*, *n*) or shape=(*n*\ /2, *n*\ /2)
            Singular row vectors
    :Versions:
        * 2017-09-25 ``@ddalle``: First version
    """
    # Calculate row-wise mean
    c = np.mean(C, axis=1)
    # Number of columns
    nC = C.shape[1]
    # Create array
    B = C - np.repeat(np.transpose([c]), nC, axis=1)
    # First attempt
    try:
        # Use all columns
        U, s, V = np.linalg.svd(B)
    except Exception:
        # Skip every other column
        U, s, V = np.linalg.svd(B[:,::2])
    # Output
    return U, s, V


# DBFM options
class _DBLLOpts(dbfm._DBFMOpts):
    pass


# DBFM definition
class _DBLLDefn(dbfm._DBFMDefn):
    pass


# Create class
class DBLL(dbfm.DBFM):
    r"""Database class for launch vehicle line loads

    :Call:
        >>> db = dbll.DBLL(fname=None, **kw)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            File name; extension is used to guess data format
        *csv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVFile` read
        *textdata*: {``None``} | :class:`str`
            Explicit file name for :class:`TextDataFile`
        *simplecsv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVSimple`
        *xls*: {``None``} | :class:`str`
            File name for :class:`XLSFile`
        *mat*: {``None``} | :class:`str`
            File name for :class:`MATFile`
    :Outputs:
        *db*: :class:`cape.attdb.dbfm.DBFM`
            LV force & moment database
    :Versions:
        * 2020-05-19 ``@ddalle``: First version
    """
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Options ---
    # Class for options
    _optscls = _DBLLOpts
    # Class for definitions
    _defncls = _DBLLDefn

   # --- Tags ---
    # Additional tags
    _tagmap = {
        "CA":   "CA",
        "CLL":  "CLL",
        "CLM":  "CLM",
        "CLN":  "CLN",
        "CN":   "CN",
        "CY":   "CY",
        "Cl":   "CLL",
        "Cm":   "CLM",
        "Cn":   "CLN",
    }
  # >

  # ==================
  # Moments
  # ==================
  # <
   # --- Moment Line Loads ---
    # Calculate and store moments if needed
    def make_ll_moment(self, col, ax1, ax2, ocol=None, **kw):
        r"""Make a moment line load based on a force line load

        :Call:
            >>> v = db.make_ll_moment(col, ax1, ax2, ocol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *ax1*: ``0`` | ``1`` | ``2`` | ``"x"`` | ``"y"`` | ``"z"``
                Force axis direction
            *ax2*: ``0`` | ``1`` | ``2`` | ``"x"`` | ``"y"`` | ``"z"``
                Moment axis direction
            *ocol*: {``None``} | :class:`str`
                Name of col in which to store output; default is *col*
                plus ``"_M%s_d%s" % (ax2, ax1)``
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for moment arm
            *ycol*: {``None``} | :class:`str`
                Name of column to use as *y*-coords for moment arm
            *zcol*: {``None``} | :class:`str`
                Name of column to use as *z*-coords for moment arm
            *x*: {``None``} | :class:`np.ndarray`
                Explicit *x*-coords for moment arm
            *y*: {``None``} | :class:`np.ndarray`
                Explicit *y*-coords for moment arm
            *z*: {``None``} | :class:`np.ndarray`
                Explicit *z*-coords for moment arm
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate 
            *xMRP*: {``DB.xMRP``} | :class:`float`
                Moment reference point *x*-coordinate
            *yMRP*: {``DB.yMRP``} | :class:`float`
                Moment reference point *y*-coordinate
            *zMRP*: {``DB.yMRP``} | :class:`float`
                Moment reference point *z*-coordinate
            *Lref*: {``DB.Lref``} | :class:`float`
                Reference length for scaling moment (use ``1.0`` for
                dimensional moment loads)
        :Outputs:
            *v*: :class:`np.ndarray`
                2D array of moments derived from *db[col]*
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Default column to save
        if ocol is None:
            # Add axes to *col*
            ocol = col + ("_M%s_d%s" % (ax2, ax1))
        # Check if already present
        if ocol in self:
            return self.get_values(ocol, mask=kw.get("mask"))
        # Otherwise, create it
        return self.create_ll_moment(col, ax1, ax2, ocol=ocol, **kw)

    # Calculate pitching moment
    def make_dclm(self, col, xcol=None, ocol=None, **kw):
        r"""Create a *dCLM* line load from *dCN* if not present

        :Call:
            >>> v = db.make_dclm(col, xcol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for moment arm
            *x*: {``None``} | :class:`np.ndarray`
                Explicit *x*-coords for moment arm
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate
            *xMRP*: {``DB.xMRP``} | :class:`float`
                Moment reference point *x*-coordinate
            *Lref*: {``DB.Lref``} | :class:`float`
                Reference length for scaling moment (use ``1.0`` for
                dimensional moment loads)
        :Outputs:
            *v*: :class:`np.ndarray`
                2D array of moments derived from *db[col]*
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Default column to save
        if ocol is None:
            # Try to convert name
            ocol = self._getcol_CLM_from_CN(col)
        # Check if present
        if ocol in self:
            return self.get_values(ocol, mask=kw.get("mask"))
        # Call creation method
        return self.create_dclm(col, xcol=xcol, ocol=ocol, **kw)

    # Calculate yawing moment
    def make_dcln(self, col, xcol=None, ocol=None, **kw):
        r"""Create a *dCLN* line load from *dCY* if not present

        :Call:
            >>> v = db.make_dcln(col, xcol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for moment arm
            *x*: {``None``} | :class:`np.ndarray`
                Explicit *x*-coords for moment arm
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate
            *xMRP*: {``DB.xMRP``} | :class:`float`
                Moment reference point *x*-coordinate
            *Lref*: {``DB.Lref``} | :class:`float`
                Reference length for scaling moment (use ``1.0`` for
                dimensional moment loads)
        :Outputs:
            *v*: :class:`np.ndarray`
                2D array of moments derived from *db[col]*
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Default column to save
        if ocol is None:
            # Try to convert name
            ocol = self._getcol_CLN_from_CY(col)
        # Check if present
        if ocol in self:
            return self.get_values(ocol, mask=kw.get("mask"))
        # Call creation method
        return self.create_dcln(col, xcol=xcol, ocol=ocol, **kw)
        
    # Calculate and store moments
    def create_ll_moment(self, col, ax1, ax2, ocol=None, **kw):
        r"""Create a moment line load based on a force line load

        :Call:
            >>> v = db.create_ll_moment(col, ax1, ax2, ocol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *ax1*: ``0`` | ``1`` | ``2`` | ``"x"`` | ``"y"`` | ``"z"``
                Force axis direction
            *ax2*: ``0`` | ``1`` | ``2`` | ``"x"`` | ``"y"`` | ``"z"``
                Moment axis direction
            *ocol*: {``None``} | :class:`str`
                Name of col in which to store output; default is *col*
                plus ``"_M%s_d%s" % (ax2, ax1)``
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for moment arm
            *ycol*: {``None``} | :class:`str`
                Name of column to use as *y*-coords for moment arm
            *zcol*: {``None``} | :class:`str`
                Name of column to use as *z*-coords for moment arm
            *x*: {``None``} | :class:`np.ndarray`
                Explicit *x*-coords for moment arm
            *y*: {``None``} | :class:`np.ndarray`
                Explicit *y*-coords for moment arm
            *z*: {``None``} | :class:`np.ndarray`
                Explicit *z*-coords for moment arm
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate 
            *xMRP*: {``DB.xMRP``} | :class:`float`
                Moment reference point *x*-coordinate
            *yMRP*: {``DB.yMRP``} | :class:`float`
                Moment reference point *y*-coordinate
            *zMRP*: {``DB.yMRP``} | :class:`float`
                Moment reference point *z*-coordinate
            *Lref*: {``DB.Lref``} | :class:`float`
                Reference length for scaling moment (use ``1.0`` for
                dimensional moment loads)
        :Outputs:
            *v*: :class:`np.ndarray`
                2D array of moments derived from *db[col]*
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Calculate the moments
        v = self.genr8_ll_moment(col, ax1, ax2, **kw)
        # Default column to save
        if ocol is None:
            # Add axes to *col*
            ocol = col + ("_M%s_d%s" % (ax2, ax1))
        # Save values and create definition
        self.save_col(ocol, v)
        # Copy definitions
        self.set_output_xargs(ocol, self.get_output_xargs(col))
        # Output
        return v

    # Calculate pitching moment
    def create_dclm(self, col, xcol=None, ocol=None, **kw):
        r"""Create a *dCLM* line load based on *dCN* line load

        :Call:
            >>> v = db.create_dclm(col, xcol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for moment arm
            *x*: {``None``} | :class:`np.ndarray`
                Explicit *x*-coords for moment arm
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate
            *xMRP*: {``DB.xMRP``} | :class:`float`
                Moment reference point *x*-coordinate
            *Lref*: {``DB.Lref``} | :class:`float`
                Reference length for scaling moment (use ``1.0`` for
                dimensional moment loads)
        :Outputs:
            *v*: :class:`np.ndarray`
                2D array of moments derived from *db[col]*
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Calculate moment
        v = self.genr8_dclm(col, xcol, **kw)
        # Default column to save
        if ocol is None:
            # Try to convert name
            ocol = self._getcol_CLM_from_CN(col)
        # Save values and create definition
        self.save_col(ocol, v)
        # Copy definitions
        self.set_output_xargs(ocol, self.get_output_xargs(col))
        # Output
        return v

    # Calculate yawing moment
    def create_dcln(self, col, xcol=None, ocol=None, **kw):
        r"""Create a *dCLN* line load based on *dCY* line load

        :Call:
            >>> v = db.create_dcln(col, xcol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for moment arm
            *x*: {``None``} | :class:`np.ndarray`
                Explicit *x*-coords for moment arm
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate
            *xMRP*: {``DB.xMRP``} | :class:`float`
                Moment reference point *x*-coordinate
            *Lref*: {``DB.Lref``} | :class:`float`
                Reference length for scaling moment (use ``1.0`` for
                dimensional moment loads)
        :Outputs:
            *v*: :class:`np.ndarray`
                2D array of moments derived from *db[col]*
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Calculate moment
        v = self.genr8_dcln(col, xcol, **kw)
        # Default column to save
        if ocol is None:
            # Try to convert name
            ocol = self._getcol_CLN_from_CY(col)
        # Save values and create definition
        self.save_col(ocol, v)
        # Copy definitions
        self.set_output_xargs(ocol, self.get_output_xargs(col))
        # Output
        return v

    # Calculate moments
    def genr8_ll_moment(self, col, ax1, ax2, **kw):
        r"""Create a moment line load based on a force line load

        :Call:
            >>> v = db.genr8_ll_moment(col, ax1, ax2, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *ax1*: ``0`` | ``1`` | ``2`` | ``"x"`` | ``"y"`` | ``"z"``
                Force axis direction
            *ax2*: ``0`` | ``1`` | ``2`` | ``"x"`` | ``"y"`` | ``"z"``
                Moment axis direction
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for moment arm
            *ycol*: {``None``} | :class:`str`
                Name of column to use as *y*-coords for moment arm
            *zcol*: {``None``} | :class:`str`
                Name of column to use as *z*-coords for moment arm
            *x*: {``None``} | :class:`np.ndarray`
                Explicit *x*-coords for moment arm
            *y*: {``None``} | :class:`np.ndarray`
                Explicit *y*-coords for moment arm
            *z*: {``None``} | :class:`np.ndarray`
                Explicit *z*-coords for moment arm
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate 
            *x*: {``None``} | :class:`np.ndarray`
                Optional 1D or 2D *x*-coordinates directly specified
            *xMRP*: {``DB.xMRP``} | :class:`float`
                Moment reference point *x*-coordinate
            *Lref*: {``DB.Lref``} | :class:`float`
                Reference length for scaling moment (use ``1.0`` for
                dimensional moment loads)
        :Outputs:
            *v*: :class:`np.ndarray`
                2D array of moments derived from *db[col]*
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Get mask
        mask = kw.get("mask")
        # Get dimension of *col*
        ndim = self.get_col_prop(col, "Dimension")
        # Ensure 2D column
        if ndim != 2:
            raise ValueError("Col '%s' is not 2D" % col)
        # Get values
        v = self.get_values(col, mask)
        # Multiply
        return self._genr8_ll_moment(v, ax1, ax2, **kw)

    # Calculate moments
    def _genr8_ll_moment(self, v, ax1, ax2, **kw):
        r"""Create a moment line load based on a force line load

        :Call:
            >>> v = db._genr8_ll_moment(v, ax1, ax2, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *v*: :class:`np.ndarray`
                2D array of data to calculate moment of
            *ax1*: ``0`` | ``1`` | ``2`` | ``"x"`` | ``"y"`` | ``"z"``
                Force axis direction
            *ax2*: ``0`` | ``1`` | ``2`` | ``"x"`` | ``"y"`` | ``"z"``
                Moment axis direction
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for moment arm
            *ycol*: {``None``} | :class:`str`
                Name of column to use as *y*-coords for moment arm
            *zcol*: {``None``} | :class:`str`
                Name of column to use as *z*-coords for moment arm
            *x*: {``None``} | :class:`np.ndarray`
                Explicit *x*-coords for moment arm
            *y*: {``None``} | :class:`np.ndarray`
                Explicit *y*-coords for moment arm
            *z*: {``None``} | :class:`np.ndarray`
                Explicit *z*-coords for moment arm
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate 
            *x*: {``None``} | :class:`np.ndarray`
                Optional 1D or 2D *x*-coordinates directly specified
            *xMRP*: {``DB.xMRP``} | :class:`float`
                Moment reference point *x*-coordinate
            *Lref*: {``DB.Lref``} | :class:`float`
                Reference length for scaling moment (use ``1.0`` for
                dimensional moment loads)
        :Outputs:
            *v*: :class:`np.ndarray`
                2D array of moments derived from *db[col]*
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
       # --- Reference Quantities ---
        # Get reference coordinates and scales
        xMRP = kw.get("xMPR", self.__dict__.get("xMRP", 0.0))
        yMRP = kw.get("yMPR", self.__dict__.get("yMRP", 0.0))
        zMRP = kw.get("zMPR", self.__dict__.get("zMRP", 0.0))
        Lref = kw.get("Lref", self.__dict__.get("Lref", 1.0))
        # Nondimensionalize
        xmrp = xMRP / Lref
        ymrp = yMRP / Lref
        zmrp = zMRP / Lref
       # --- Axes ---
        # Check *ax1*
        if ax1 not in {0, 1, 2, "x", "y", "z"}:
            raise ValueError("Unrecognized value for moment axis (*ax1*)")
        # Check *ax2*
        if ax2 not in {0, 1, 2, "x", "y", "z"}:
            raise ValueError("Unrecognized value for moment axis (*ax2*)")
       # --- Values ---
        # Check dimension
        if not isinstance(v, np.ndarray):
            raise TypeError("Input loads must be array")
        elif v.ndim != 2:
            raise IndexError("Input loads are not 2D")
        # Number of conditions
        ncut, nv = v.shape
       # --- Cut Coordinates ---
        # Get mask
        mask = kw.get("mask")
        # Process *x*-coordinates
        xcol = kw.get("xcol")
        ycol = kw.get("ycol")
        zcol = kw.get("zcol")
        # Process *x*
        if xcol is None:
            # Use 0, 1, 2, ... as *x* coords
            x = kw.get("x", 0.0)
        else:
            # Get values
            x = self.get_all_values(xcol)
        # Check *x*
        if isinstance(x, float):
            # Constant is allowed
            ndimx = 0
        elif not isinstance(x, np.ndarray):
            # Bad type
            raise TypeError("x-coords for integration must be float or array")
        else:
            # Number of dimensions
            ndimx = x.ndim
            # Check it
            if ndimx > 2:
                raise IndexError("Cannot process %-D x-coords" % ndimx)
            # Apply *mask* if necessary
            if (xcol is not None) and (ndimx == 2):
                x = self.get_values(xcol, mask)
        # Process *y*
        if ycol is None:
            # Use 0, 1, 2, ... as *x* coords
            y = kw.get("y", 0.0)
        else:
            # Get values
            y = self.get_all_values(ycol)
        # Check *y*
        if isinstance(y, float):
            # Constant is allowed
            ndimy = 0
        elif not isinstance(y, np.ndarray):
            # Bad type
            raise TypeError("y-coords for integration must be float or array")
        else:
            # Number of dimensions
            ndimy = y.ndim
            # Check it
            if ndimy > 2:
                raise IndexError("Cannot process %-D y-coords" % ndimy)
            # Apply *mask* if necessary
            if (ycol is not None) and (ndimy == 2):
                y = self.get_values(ycol, mask)
        # Process *z*
        if zcol is None:
            # Use 0, 1, 2, ... as *x* coords
            z = kw.get("z", 0.0)
        else:
            # Get values
            z = self.get_all_values(zcol)
        # Check *z*
        if isinstance(z, float):
            # Constant is allowed
            ndimz = 0
        elif not isinstance(z, np.ndarray):
            # Bad type
            raise TypeError("z-coords for integration must be float or array")
        else:
            # Number of dimensions
            ndimz = z.ndim
            # Check it
            if ndimz > 2:
                raise IndexError("Cannot process %-D x-coords" % ndimz)
            # Apply *mask* if necessary
            if (zcol is not None) and (ndimz == 2):
                z = self.get_values(zcol, mask)
       # --- Calculations ---
        # Check which axes are involved
        if ax1 in ["x", 0]:
            # Integrate *dCA*
            if ax2 in ["x", 0]:
                # No possibility of *dCA* creating rolling moment
                r = 0.0
            elif ax2 in ["y", 1]:
                # Calculate *dCLM* (pitching moment) from *dCA*
                r = zmrp - z
                # Tile if 1D
                if ndimz == 1:
                    r = np.tile(r, (nv, 1)).T
            elif ax2 in ["z", 2]:
                # Calculate *dCLN* (yawing moment) from *dCA*
                r = y - ymrp
                # Tile if 1D
                if ndimy == 1:
                    r = np.tile(r, (nv, 1)).T
        elif ax1 in ["y", 1]:
            # Integrate *dCY*
            if ax2 in ["x", 0]:
                # Calculate *dCLL* (rolling moment) from *dCY*
                r = zmrp - z
                # Tile if 1D
                if ndimz == 1:
                    r = np.tile(r, (nv, 1)).T
            elif ax2 in ["y", 1]:
                # No possibility of *dCLM* (pitching moment) from *dCY*
                r = 0.0
            elif ax2 in ["z", 2]:
                # Calculate *dCLN* (yawing moment) from *dCY*
                r = xmrp - x
                # Tile if 1D
                if ndimx == 1:
                    r = np.tile(r, (nv, 1)).T
        elif ax1 in ["z", 2]:
            # Integrate *dCN*
            if ax2 in ["x", 0]:
                # Calculate *dCLL* (rolling moment) from *dCN*
                r = ymrp - y
                # Tile if 1D
                if ndimy == 1:
                    r = np.tile(r, (nv, 1)).T
            elif ax2 in ["y", 1]:
                # Calculate *dCLM* (pitching moment) from *dCN*
                r = xmrp - x
                # Tile if 1D
                if ndimx == 1:
                    r = np.tile(r, (nv, 1)).T
            elif ax2 in ["z", 2]:
                # No possibility of *dCLN*) (yawing moment) from *dCN*
                r = 0.0
       # --- Output ---
        # Multiply moment arm times load
        return r * v

    # Calculate pitching moment
    def genr8_dclm(self, col, xcol=None, **kw):
        r"""Create a *dCLM* line load based on *dCN* line load

        :Call:
            >>> v = db.genr8_dclm(col, xcol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for moment arm
            *x*: {``None``} | :class:`np.ndarray`
                Explicit *x*-coords for moment arm
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate
            *xMRP*: {``DB.xMRP``} | :class:`float`
                Moment reference point *x*-coordinate
            *Lref*: {``DB.Lref``} | :class:`float`
                Reference length for scaling moment (use ``1.0`` for
                dimensional moment loads)
        :Outputs:
            *v*: :class:`np.ndarray`
                2D array of moments derived from *db[col]*
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Default *xcol*
        if (xcol is None) and ("x" not in kw):
            # Try to get the output *xarg*
            xargs = self.get_output_xargs(col)
            # Check if list
            if isinstance(xargs, (list, tuple)) and len(xargs) > 0:
                # Default
                xcol = xargs[0]
        # Calculate the moment
        return self.genr8_ll_moment(col, 2, 1, xcol=xcol, **kw)

    # Calculate yawing moment
    def genr8_dcln(self, col, xcol=None, **kw):
        r"""Create a *dCLN* line load based on *dCY* line load

        :Call:
            >>> v = db.genr8_dcln(col, xcol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for moment arm
            *x*: {``None``} | :class:`np.ndarray`
                Explicit *x*-coords for moment arm
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate
            *xMRP*: {``DB.xMRP``} | :class:`float`
                Moment reference point *x*-coordinate
            *Lref*: {``DB.Lref``} | :class:`float`
                Reference length for scaling moment (use ``1.0`` for
                dimensional moment loads)
        :Outputs:
            *v*: :class:`np.ndarray`
                2D array of moments derived from *db[col]*
        :Versions:
            * 2020-06-04 ``@ddalle``: First version
        """
        # Default *xcol*
        if (xcol is None) and ("x" not in kw):
            # Try to get the output *xarg*
            xargs = self.get_output_xargs(col)
            # Check if list
            if isinstance(xargs, (list, tuple)) and len(xargs) > 0:
                # Default
                xcol = xargs[0]
        # Calculate the moment
        return self.genr8_ll_moment(col, 1, 2, xcol=xcol, **kw)
  # >

  # ==================
  # Combination
  # ==================
  # <
   # --- Components ---
    # Combine and save the combined loads
    def make_ll_combo(self, col, cols, x, **kw):
        r"""Combine line loads from several components

        This method can be used to combine several line loads with
        disparate *x*-coordinates, and point loads can also be
        injected into the combined line load.  All *cols* must have an
        *xcol* specified.  For existing line loads, a suitable default
        is often available via :func:`get_output_xargs`, but injected
        point loads must have an *xcol* specified.  The user may
        specify *xcol* as either a column name or a directly defined
        value.  For point loads, this *xcol* may be just a number.

        This method does not combine the loads if *col* is already
        present in the database.

        :Call:
            >>> v = db.make_ll_combo(col, cols, x, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name with which to save result
            *cols*: :class:`list`\ [:class:`str`]
                List of load columns to combine
            *x*: :class:`np.ndarray`\ [:class:`float`]
                * *ndim*: 1 | 2
                * *shape*: (*nx*,) | (*nx*, *ncase*)

                Array of *x*-coordinates for output
            *mask*: {``None``} | :class:`np.ndarray`
                Mask of which cases to include
            *xcols*: :class:`dict`\ [:class:`str` | :class:`np.ndarray`]
                Descriptor for what *x* coordinates to use for each
                *col* in *cols*; if :class:`str`, use it as a *col*
                name; if :class:`np.ndarray`, use as directly defined
                coords; defaults to ``db.get_output_xargs(col)``
        :Outputs:
            *v*: :class:`np.ndarray`\ [:class:`float`]
                * *ndim*: 2
                * *shape*: (*nx*, *ncase*)

                Combined loads
        :Versions:
            * 2020-06-10 ``@ddalle``: First version
        """
        # Check if column present
        if col in self.cols:
            # Output it
            return self.get_all_values(col)
        # Calculate combined loads
        v = self.genr8_ll_combo(cols, x, **kw)
        # Save values and create definition
        self.save_col(col, v)
        # Output
        return v

    # Combine and save the combined loads
    def create_ll_combo(self, col, cols, x, **kw):
        r"""Combine line loads from several components

        This method can be used to combine several line loads with
        disparate *x*-coordinates, and point loads can also be
        injected into the combined line load.  All *cols* must have an
        *xcol* specified.  For existing line loads, a suitable default
        is often available via :func:`get_output_xargs`, but injected
        point loads must have an *xcol* specified.  The user may
        specify *xcol* as either a column name or a directly defined
        value.  For point loads, this *xcol* may be just a number.

        :Call:
            >>> v = db.create_ll_combo(col, cols, x, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name with which to save result
            *cols*: :class:`list`\ [:class:`str`]
                List of load columns to combine
            *x*: :class:`np.ndarray`\ [:class:`float`]
                * *ndim*: 1 | 2
                * *shape*: (*nx*,) | (*nx*, *ncase*)

                Array of *x*-coordinates for output
            *mask*: {``None``} | :class:`np.ndarray`
                Mask of which cases to include
            *xcols*: :class:`dict`\ [:class:`str` | :class:`np.ndarray`]
                Descriptor for what *x* coordinates to use for each
                *col* in *cols*; if :class:`str`, use it as a *col*
                name; if :class:`np.ndarray`, use as directly defined
                coords; defaults to ``db.get_output_xargs(col)``
        :Outputs:
            *v*: :class:`np.ndarray`\ [:class:`float`]
                * *ndim*: 2
                * *shape*: (*nx*, *ncase*)

                Combined loads
        :Versions:
            * 2020-06-10 ``@ddalle``: First version
        """
        # Calculate combined loads
        v = self.genr8_ll_combo(cols, x, **kw)
        # Save values and create definition
        self.save_col(col, v)
        # Output
        return v

    # Combine line loads
    def genr8_ll_combo(self, cols, x, **kw):
        r"""Combine line loads from several components

        This method can be used to combine several line loads with
        disparate *x*-coordinates, and point loads can also be
        injected into the combined line load.  All *cols* must have an
        *xcol* specified.  For existing line loads, a suitable default
        is often available via :func:`get_output_xargs`, but injected
        point loads must have an *xcol* specified.  The user may
        specify *xcol* as either a column name or a directly defined
        value.  For point loads, this *xcol* may be just a number.

        :Call:
            >>> v = db.genr8_ll_combo(cols, x, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                List of load columns to combine
            *x*: :class:`np.ndarray`\ [:class:`float`]
                * *ndim*: 1 | 2
                * *shape*: (*nx*,) | (*nx*, *ncase*)

                Array of *x*-coordinates for output
            *mask*: {``None``} | :class:`np.ndarray`
                Mask of which cases to include
            *xcols*: :class:`dict`\ [:class:`str` | :class:`np.ndarray`]
                Descriptor for what *x* coordinates to use for each
                *col* in *cols*; if :class:`str`, use it as a *col*
                name; if :class:`np.ndarray`, use as directly defined
                coords; defaults to ``db.get_output_xargs(col)``
        :Outputs:
            *v*: :class:`np.ndarray`\ [:class:`float`]
                * *ndim*: 2
                * *shape*: (*nx*, *ncase*)

                Combined loads
        :Versions:
            * 2020-06-09 ``@ddalle``: First version
        """
        # Check *x*
        if not isinstance(x, np.ndarray):
            raise TypeError(
                "Combo load x-coords must be 'np.ndarray', got '%s'" % type(x))
        # Get *x* dimension
        ndimx = x.ndim
        # Check dimension
        if ndimx == 0:
            raise IndexError("Cannot calculate combo load on 0-D x-coord")
        elif ndimx > 2:
            raise IndexError(
                "Cannot calculate combo load on %i-D x-coords" % ndimx)
        # Get mask
        mask = kw.pop("mask", None)
        # Option for specific *x* values
        xcols = kw.get("xcols", {})
        # Initialize load data and x coords
        vdata = {}
        xdata = {}
        # Dimensions for both
        ndimvdata = {}
        ndimxdata = {}
        # Loop through *cols* to be combined
        for j, col in enumerate(cols):
            # Check type
            if not typeutils.isstr(col):
                raise TypeError("Combo col %i is not a 'str'" % j)
            # Get values
            vj = self.get_values(col, mask)
            # Check for manual *x*
            xoptj = xcols.get(col)
            # If ``None``, try to use named *x* var
            if xoptj is None:
                # Get *x* vars for line load
                xvarsj = self.get_output_xargs(col)
                # Check for singleton list
                if isinstance(xvarsj, list) and len(xvarsj) == 1:
                    # Use it
                    xoptj = xvarsj[0]
                else:
                    # Otherwise we have a problem
                    raise ValueError(
                        "Could not determine x-coords for combo col '%s'"
                        % col)
            # Check for valid manual coordinates
            if isinstance(xoptj, np.ndarray):
                # Use data directly
                xj = xoptj
                # Dimensions from array
                ndimxj = xj.ndim
            elif isinstance(xoptj, float):
                # Use data directly
                xj = xoptj
                # Scalar *x*-coordinate
                ndimxj = 0
            elif typeutils.isstr(xoptj):
                # Get dimension for named *col*
                ndimxj = self.get_ndim(xoptj)
                # Check for validity
                if ndimxj is None:
                    raise ValueError(
                        "Could not find x-coords '%s'" % xoptj)
                # Apply mask if 2D
                if ndimxj == 2:
                    # Get masked x-coords
                    xj = self.get_values(xoptj, mask)
                else:
                    # Get all values
                    xj = self.get_all_values(xoptj)
            else:
                # Bad type
                raise TypeError(
                    "Cannot use x-coord data of type '%s'" % type(xoptj))
            # Dimension of data for col *col*
            ndimvj = self.get_ndim(col)
            # Check consistency
            if ndimvj < 2:
                # Must have scalar *x* if scalar *v*
                if ndimxj > 1:
                    raise IndexError(
                        "Cannot combine 2D x-coords with scalar load '%s'"
                        % col)
            elif ndimvj == 2:
                # Ensure 1D or 2D
                if ndimxj == 0:
                    raise IndexError(
                        "Cannot use scalar *x* with 2D load '%s'" % col)
                elif ndimxj > 2:
                    raise IndexError("Cannot use %i-D x-coords" % ndimxj)
            else:
                # Cannot use 3D+ loads
                raise IndexError("Cannot combine %i-D loads" % ndimvj)
            # Check size
            if j == 0:
                # Save the number of cases
                ncase = vj.shape[-1]
            elif vj.shape[-1] != ncase:
                # Mismatch
                raise IndexError(
                    "Col %i (%s) has %i cases; expecting %i"
                    % (j, col, vj.shape[-1], ncase))
            # Save data
            xdata[col] = xj
            vdata[col] = vj
            ndimxdata[col] = ndimxj
            ndimvdata[col] = ndimvj
        # Initialize output load
        v = np.zeros((x.size, ncase), dtype=x.dtype)
        # Tolerance for *x* intervals
        xtol = kw.get("xtol", 1e-5 * (np.max(x) - np.min(x)))
        # Loop through columns
        for j, col in enumerate(cols):
            # Dimension for this col
            ndimxj = ndimxdata[col]
            ndimvj = ndimvdata[col]
            # Loop through cases
            for i in range(ncase):
                # Get x-coordinates for this case
                if ndimx == 1:
                    # Always the same for 1D
                    xi = x
                else:
                    # Select column
                    xi = x[:,i]
                # Check it
                if ndimvj == 0:
                    # Scalar, constant load for all case
                    xdataj = xdata[col]
                    vdataj = vdata[col]
                elif ndimvj == 1:
                    # Scalar, varies by case
                    xdataj = xdata[col]
                    vdataj = vdata[col][i]
                else:
                    # Line load
                    vdataj = vdata[col][:,i]
                    # Check *x* dimension
                    if ndimxj == 1:
                        # Common *x* for all line loads
                        xdataj = xdata[col]
                    else:
                        # Select *x* for this case
                        xdataj = xdata[col][:,i]
                # Combine *x* coordinates
                if ndimvj < 2:
                    # Find point *xj* position w.r.t. *xi*
                    if xdataj + xtol < xi[0] or xdataj - xtol > xi[-1]:
                        # Can't inject outside point load
                        raise ValueError(
                            ("Point-like load '%s' is outside " % col) +
                            ("x boundaries [%.2e, %.2e]" % (xi[0], xi[-1])))
                    elif xdataj < xi[0]:
                        # Nudge to the right
                        xdataj += xtol
                    elif xdataj > xi[-1]:
                        # Nudge to the left
                        xdataj -= xtol
                    # Find index of segment with *xdataj* inside
                    kcut = np.where(xdataj >= x)[0][0]
                    # Width of this interval
                    dxj = xdataj[kcut+1] - xdataj[kcut]
                    # Update the load on that segment
                    v[kcut, i] += vdataj / dxj
                    # Move on so we can dedent the ndimvj==2 code
                    continue
                # Check for fixed *x*
                if (i == 0) or (ndimxj == 2) or (ndimx == 2):
                    # Refilter *x* mapping
                    # Check limits
                    if xdataj[0] + xtol < xi[0]:
                        raise ValueError(
                            ("Line load '%s' start x-coord " % col) +
                            ("for case %i is outside bounds" % i))
                    elif xdataj[-1] - xtol > 2*xi[-1] - xi[-2]:
                        raise ValueError(
                            ("Line load '%s' end x-coord " % col) +
                            ("for case %i is outside bounds" % i))
                    elif xdataj[0] + xtol < xi[0]:
                        # Nudge start right
                        xdataj[0] += xtol
                    elif xdataj[-1] - xtol > 2*xi[-1] - xi[-2]:
                        # Nudge end left
                        xdataj[-1] -= xtol
                    # Number of cuts in input (*col*) and output
                    ncutj = xdataj.size
                    ncuti = xi.size
                    # Initialize matrix
                    A = np.zeros((ncuti, ncutj))
                    # Loop through cuts
                    for kcut, xa in enumerate(xdataj):
                        # End of segment
                        if kcut + 1 == ncutj:
                            # Extrapolate
                            xb = 2*xa - xdataj[-2]
                        else:
                            # Start of next segment
                            xb = xdataj[kcut+1]
                        # Find segment
                        ka = np.count_nonzero(xa >= xi) - 1
                        kb = np.count_nonzero(xb > xi) - 1
                        # Output x-coords
                        xia0 = xi[ka]
                        xib0 = xi[kb]
                        # Check for extrapolated segment
                        if ka + 1 == ncuti:
                            xia1 = 2*xi[-1] - xi[-2]
                        else:
                            xia1 = xi[ka+1]
                        # Check for extrapolated segment
                        if kb + 1 == ncuti:
                            xib1 = 2*xi[-1] - xi[-2]
                        else:
                            xib1 = xi[kb+1]
                        # Full segments
                        A[ka+1:kb, kcut] = 1.0
                        # Progress fractions
                        fa = (xa - xia0) / (xia1 - xia0)
                        fb = (xb - xib0) / (xib1 - xib0)
                        # Check for single segment
                        if ka == kb:
                            # Single fraction
                            A[ka, kcut] = fb - fa
                        else:
                            # Fractions for start and and chunks
                            A[ka, kcut] = 1.0 - fa
                            A[kb, kcut] = fb
                    # Special case: fixed mapping
                    if (ndimx == 1) and (ndimxj == 1):
                        # Multiply all cases at once
                        v += np.dot(A, vdata[col])
                        # Don't do the case loop
                        break
                # Multiply *A* to transform load for case *i* col *j*
                v[:, i] += np.dot(A, vdataj)
        # Output
        return v
  # >

  # ==================
  # Adjustment
  # ==================
  # <
   # --- Adjustment ---
    # Calculate adjusted loads
    def make_ll3x_adjustments(self, comps, db2, scol=None, **kw):
        r"""Retrieve [and calculate] adjusted line loads

        :Call:
            >>> lla = db.make_ll3x_adjustments(comps, db2, scol, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *db2*: :class:`cape.attdb.rdb.DataKit`
                Target database with analysis tools
            *comps*: :class:`list`\ [:class:`str`]
                List of components to divide integral F&M
            *scol*: {``None``} | :class:`str`
                Column used to slice database; output will be constant
                on each slice
            *SourceCols*: :class:`dict`\ [:class:`str`]
                Cols in *db* to use as comparison to *db2* for ``"CA"``,
                ``"CY"``, etc.
            *TargetCols*: :class:`dict`\ [:class:`str`]
                Cols in *db2* to use as targets for ``"CA"``, etc.
            *CompFMCols*: :class:`dict`\ [:class:`dict`]
                Columns to use as integral of force and moment on each
                *comp*. Defaults filled in by *comp*\ .*coeff* for
                *coeff* in ``["CA", "CY", "CN", "CLL", "CLM", "CLN"]``
            *CompLLCols*: :class:`dict`\ [:class:`dict`]
                Cols to use as line loads for ``"CA"``, ``"CY"``,
                ``"CN"`` for each *comp*
        :Outputs:
            *lla*: :class:`dict`\ [:class:`dict`\ [:class:`np.ndarray`]]
                Adjustment weight for each *comp* and each *coeff*
                caused by integral *coeff* change; for example
                ``"wCN.CLL"`` is shift to a comp's *CN* as a result of
                overall *CLL* shift
        :Versions:
            * 2020-06-23 ``@ddalle``: Version 1.0
        """
       # --- Options ---
        # Check and process component list
        comps = self._check_ll3x_comps(comps)
        # Form options
        opts = _LL3XOpts(**kw)
       # --- Retrieval ---
        # Initialize output if already present
        lla = {}
        # Loop through components
        for comp in comps:
            # Get names of adjusted line load columns
            llacols = self._getcols_lla_comp(None, comp)
            # Unpack
            colCA = llacols["CA"]
            colCY = llacols["CY"]
            colCN = llacols["CN"]
            # Check for primary loads
            if colCA not in self:
                # Not present
                break
            elif colCY not in self:
                # Not present
                break
            elif colCN not in self:
                # Not present
                break
            # Otherwise get those loads
            lla[comp] = {
                "CA": self[colCA],
                "CY": self[colCY],
                "CN": self[colCN],
            }
        else:
            # No ``break`` encountered
            return lla
       # --- Suboptions ---
        # Sections
        kw_int = opts.section_options("integrals_make")
        kw_frc = opts.section_options("fractions_make")
        kw_trg = opts.section_options("fmdelta_make")
        kw_bas = opts.section_options("basis_make")
       # --- Integrals FM ---
        # Calculate integral forces
        fm = self.make_ll3x_integrals(comps, **kw_int)
        # Calculate adjustment fractions
        w = self.make_ll3x_aweights(comps, scol, **kw_frc)
        # Evaluate target database
        dfm = self.genr8_target_deltafm(db2, **kw_trg)
        # Unpack differences
        deltaCA = dfm["CA"]
        deltaCY = dfm["CY"]
        deltaCN = dfm["CN"]
        deltaCLL = dfm["CLL"]
        deltaCLM = dfm["CLM"]
        deltaCLN = dfm["CLN"]
       # --- Basis ---
        # Get adjustment basis
        basis = self.make_ll3x_basis(comps, scol, **kw_bas)
       # --- Conditions/Mask ---
        # Get reference column
        col = self._getcols_ll3x_fmcomp(None, comp, **kw)["CA"]
        # Check if present
        if col not in self:
            raise KeyError("Missing 'CA' integral for comp '%s'" % comp)
        # Get conditions
        mask = opts.get_option("mask")
        # Get conditions
        I = self.prep_mask(mask, col=col)
        # Slice values
        s = w.get("s")
       # --- Adjustment ---
        # Initialize
        dfmcomp = {}
        # Loop through components
        for comp in comps:
            # Get names for LL cols
            llcols = self._getcols_ll_comp(None, comp)
            # Get weights
            wcomp = w[comp]
            # Bases
            bcomp = basis[comp]
            # Get existing load
            dCA = self.get_values(llcols["CA"], mask)
            dCY = self.get_values(llcols["CY"], mask)
            dCN = self.get_values(llcols["CN"], mask)
            # Copies for adjusted loads
            dCAa = dCA.copy()
            dCYa = dCY.copy()
            dCNa = dCN.copy()
            # Loop through cases
            for i in I:
                # Check for slicing
                if scol is None:
                    # Only one basis
                    j = 0
                else:
                    #Get slice col value
                    si = self.get_values(scol, i)
                    # Find index
                    j = np.argmin(np.abs(si - s))
                # Get weights
                wAA = wcomp["wCA.CA"][j]
                wYY = wcomp["wCY.CY"][j]
                wNN = wcomp["wCN.CN"][j]
                wYLL = wcomp["wCY.CLL"][j]
                wNLL = wcomp["wCN.CLL"][j]
                wmm = wcomp["wCLM.CLM"][j]
                wnn = wcomp["wCLN.CLN"][j]
                # Apply the weights
                dfCAi = wAA*deltaCA[i]
                dfCYi = wYY*deltaCY[i] + wYLL*deltaCLL[i]
                dfCNi = wNN*deltaCN[i] + wNLL*deltaCLL[i]
                dfCLMi = wmm*deltaCLM[i]
                dfCLNi = wnn*deltaCLN[i]
                # Apply corrections for forces
                dCAa[:,i] += dfCAi*bcomp["dCA.CA"][:,j]
                dCYa[:,i] += dfCYi*bcomp["dCY.CY"][:,j]
                dCNa[:,i] += dfCNi*bcomp["dCN.CN"][:,j]
                # Apply forces for moments
                dCYa[:,i] += dfCLNi*bcomp["dCY.CLN"][:,j]
                dCNa[:,i] += dfCLMi*bcomp["dCN.CLM"][:,j]
            # Save values for output
            lla[comp] = {
                "CA": dCAa,
                "CY": dCYa,
                "CN": dCNa,
            }
            # Column names to save
            llacols = self._getcols_lla_comp(None, comp)
            # Unpack columns
            lcolCA = llcols["CA"]
            lcolCY = llcols["CY"]
            lcolCN = llcols["CN"]
            acolCA = llacols["CA"]
            acolCY = llacols["CY"]
            acolCN = llacols["CN"]
            # Save the cols
            self.save_col(acolCA, dCAa)
            self.save_col(acolCY, dCYa)
            self.save_col(acolCN, dCNa)
            # Copy responses ...
            self.set_output_xargs(acolCA, self.get_output_xargs(lcolCA))
            self.set_output_xargs(acolCY, self.get_output_xargs(lcolCY))
            self.set_output_xargs(acolCN, self.get_output_xargs(lcolCN))
       # --- Cleanup ---
        # Output
        return lla
            
        

   # --- Adjustment Fraction ---
    # Calculate each component's contribution to adjusted loads
    def make_ll3x_aweights(self, comps, scol=None, **kw):
        r"""Retrieve [and calculate] each component's adjustment weight

        :Call:
            >>> w = db.make_ll3x_aweights(comps, scol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *comp*: :class:`str`
                Single component (trivial output)
            *comps*: :class:`list`\ [:class:`str`]
                List of components to divide integral F&M
            *scol*: {``None``} | :class:`str`
                Column used to slice database; output will be constant
                on each slice
            *CompFMCols*: :class:`dict`\ [:class:`dict`]
                Columns to use as integral of force and moment on each
                *comp*. Defaults filled in by *comp*\ .*coeff* for
                *coeff* in ``["CA", "CY", "CN", "CLL", "CLM", "CLN"]``
        :Outputs:
            *w*: :class:`dict`\ [:class:`dict`\ [:class:`np.ndarray`]]
                Adjustment weight for each *comp* and each *coeff*
                caused by integral *coeff* change; for example
                ``"wCN.CLL"`` is shift to a comp's *CN* as a result of
                overall *CLL* shift
        :Versions:
            * 2020-06-19 ``@ddalle``: First version
        """
        # Check the component list
        comps = self._check_ll3x_comps(comps)
        # Transfer options
        opts = _LL3XOpts(_section="aweights_make", **kw)
        # Name of slice field
        ocol = opts.get_option("SliceColSave", "adjust.%s" % scol)
        # Initialize output if all cols present
        w = {}
        # Check fir slice values
        if ocol in self:
            # Save it to *f*
            w["s"] = self[ocol]
            qmake = False
        else:
            qmake = True
        # Loop through the components
        for comp in comps:
            # Get column names
            wcols = self._getcols_fmweight_comp(None, comp=comp, **opts)
            # Unpack
            colCA = wcols["wCA.CA"]
            colCY = wcols["wCY.CY"]
            colCN = wcols["wCN.CN"]
            colCl = wcols["wCY.CLL"]
            colC2 = wcols["wCN.CLL"]
            colCm = wcols["wCLM.CLM"]
            colCn = wcols["wCLN.CLN"]
            # Check if all are present
            if colCA not in self:
                break
            elif colCY not in self:
                break
            elif colCN not in self:
                break
            elif colC1 not in self:
                break
            elif colC2 not in self:
                break
            elif colCm not in self:
                break
            elif colCn not in self:
                break
            else:
                # Everything already present
                w[comp] = {
                    "wCA.CA": self[colCA],
                    "wCY.CY": self[colCY],
                    "wCN.CN": self[colCN],
                    "wCY.CLL": self[colC1],
                    "wCN.CLL": self[colC2],
                    "wCLM.CLM": self[colCm],
                    "wCLN.CLN": self[colCn],
                }
        else:
            # If no ``break`` encountered, return output
            if not qmake:
                return w
        # Generate the weights and return them
        return self.create_ll3x_aweights(comps, scol, **opts)

    # Calculate each component's contribution to adjusted loads
    def create_ll3x_aweights(self, comps, scol=None, **kw):
        r"""Calculate and save each component's adjustment weight

        :Call:
            >>> w = db.create_ll3x_aweights(comps, scol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *comp*: :class:`str`
                Single component (trivial output)
            *comps*: :class:`list`\ [:class:`str`]
                List of components to divide integral F&M
            *scol*: {``None``} | :class:`str`
                Column used to slice database; output will be constant
                on each slice
            *CompFMCols*: :class:`dict`\ [:class:`dict`]
                Columns to use as integral of force and moment on each
                *comp*. Defaults filled in by *comp*\ .*coeff* for
                *coeff* in ``["CA", "CY", "CN", "CLL", "CLM", "CLN"]``
        :Outputs:
            *w*: :class:`dict`\ [:class:`dict`\ [:class:`np.ndarray`]]
                Adjustment weight for each *comp* and each *coeff*
                caused by integral *coeff* change; for example
                ``"wCN.CLL"`` is shift to a comp's *CN* as a result of
                overall *CLL* shift
        :Versions:
            * 2020-06-18 ``@ddalle``: First version
            * 2020-06-19 ``@ddalle``: Weights same size of bkpts
        """
        # Check and process component list
        comps = self._check_ll3x_comps(comps)
        # Form options
        opts = _LL3XOpts(_section="aweights_make", **kw)
        # Get options for genr8 function
        kwf = opts.section_options("aweights")
        # Calculate fractions
        w = self.genr8_ll3x_aweights(comps, scol, **kwf)
        # Get slice values
        s = w.get("s")
        # Save slice values
        if s is not None:
            # Name of slice field
            ocol = opts.get_option("SliceColSave", "adjust.%s" % scol)
            # Save values and definition
            self.save_col(ocol, s)
        # Loop through the components
        for comp in comps:
            # Get column names
            wcols = self._getcols_fmweight_comp(None, comp=comp, **opts)
            # Data for this component
            wcomp = w[comp]
            # Unpack
            colCA = wcols["wCA.CA"]
            colCY = wcols["wCY.CY"]
            colCN = wcols["wCN.CN"]
            colC1 = wcols["wCY.CLL"]
            colC2 = wcols["wCN.CLL"]
            colCm = wcols["wCLM.CLM"]
            colCn = wcols["wCLN.CLN"]
            # Save the integrated columns
            self.save_col(colCA, wcomp["wCA.CA"])
            self.save_col(colCY, wcomp["wCY.CY"])
            self.save_col(colCN, wcomp["wCN.CN"])
            self.save_col(colC1, wcomp["wCY.CLL"])
            self.save_col(colC2, wcomp["wCN.CLL"])
            self.save_col(colCm, wcomp["wCLM.CLM"])
            self.save_col(colCn, wcomp["wCLN.CLN"])
        # Output
        return w

    # Calculate each component's contribution to adjusted loads
    def genr8_ll3x_aweights(self, comps, scol=None, **kw):
        r"""Calculate each component's adjustment weight

        :Call:
            >>> w = db.genr8_ll3x_aweights(comps, scol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *comps*: :class:`list`\ [:class:`str`]
                List of components to divide integral F&M
            *scol*: {``None``} | :class:`str`
                Column used to slice database; output will be constant
                on each slice
            *CompFMCols*: :class:`dict`\ [:class:`dict`]
                Columns to use as integral of force and moment on each
                *comp*. Defaults filled in by *comp*\ .*coeff* for
                *coeff* in ``["CA", "CY", "CN", "CLL", "CLM", "CLN"]``
            *CompY*: :class:`dict`\ [:class:`float`]
                *y*-coordinate at which *comp* loads are applied
            *CompZ*: :class:`dict`\ [:class:`float`]
                *z*-coordinate at which *comp* loads are applied
        :Outputs:
            *w*: :class:`dict`\ [:class:`dict`\ [:class:`np.ndarray`]]
                Adjustment weight for each *comp* and each *coeff*
                caused by integral *coeff* change; for example
                ``"wCN.CLL"`` is shift to a comp's *CN* as a result of
                overall *CLL* shift
        :Versions:
            * 2020-06-18 ``@ddalle``: First version
            * 2020-06-19 ``@ddalle``: Weights same size of bkpts
        """
       # --- Checks ---
        # Check and process component list
        comps = self._check_ll3x_comps(comps)
        # Form options
        opts = _LL3XOpts(_section="aweights", **kw)
        # Number of components
        ncomp = len(comps)
       # --- FM Cols ---
        # Get columns that determine deltas for each component
        fmcols = opts.get_option("CompFMCols", {})
        # Coordinate options
        ycomp = opts.get_option("CompY", {})
        zcomp = opts.get_option("CompZ", {})
        # Reference length
        Lref = opts.get_option("Lref", self.__dict__.get("Lref", 1.0))
        # Coordinates
        Y = np.zeros(ncomp)
        Z = np.zeros(ncomp)
        # Loop through components
        for j, comp in enumerate(comps):
            # Get integral col names for this componet
            cols = self._getcols_ll3x_fmcomp(None, comp, CompFMCols=fmcols)
            # Check presence
            for coeff, col in cols.items():
                if col not in self:
                    raise KeyError(
                        "%s col '%s' for comp '%s' not present"
                        % (coeff, col, comp))
            # Save completed list
            fmcols[comp] = cols
            # Get reference coordinates
            Y[j] = ycomp.get(comp, 0.0) / Lref
            Z[j] = zcomp.get(comp, 0.0) / Lref
            # Reference column
            if j == 0:
                # Get reference column for *mask* application
                col = cols["CA"]
                # Get size of column
                nv = self.get_all_values(col).size
        # Save maximum *dy* and *dz* as flags for which constraints apply
        qdy = np.max(np.abs(Y - np.mean(Y))) > 1e-6
        qdz = np.max(np.abs(Z - np.mean(Z))) > 1e-6
       # --- Mask & Slicing ---
        # Get mask option
        mask = opts.get_option("mask")
        # Get indices
        mask_index = self.prep_mask(mask, col=col)
        # Slice column
        if scol is None:
            # Single slice with all cases in *mask*
            masks = [self.prep_mask(mask, col=col)]
            # No slice values
            s = None
            # Single mask
            nmask = 1
        else:
            # Get unique values
            s = self.genr8_bkpts(scol, nmin=5, tol=1e-8, mask=mask)
            # Divide those values into slices
            masks, _ = self.find([scol], s, mapped=True)
            # Number of slice values
            nmask = s.size
       # --- Weights ---
        # Initialize weights
        w = {"s": s}
        # Loop through components to initialize
        for comp in comps:
            # Init all needed weights
            w[comp] = {
                "wCA.CA": np.zeros(nmask),
                "wCY.CY": np.zeros(nmask),
                "wCN.CN": np.zeros(nmask),
                "wCY.CLL": np.zeros(nmask),
                "wCN.CLL": np.zeros(nmask),
                "wCLM.CLM": np.zeros(nmask),
                "wCLN.CLN": np.zeros(nmask),
            }
        # Loop through coefficients
        for coeff in _coeffs:
            # Initialize total
            FT = 0.0
            # Initialize dict per *comp*
            Fcomp = {}
            # Loop through components
            for comp in comps:
                # Component column
                col = fmcols[comp][coeff]
                # Get values
                F = np.abs(self[col])
                # Increment total
                FT += F
                # Save it
                Fcomp[comp] = F
            # Loop through masks
            for j, maskj in enumerate(masks):
                # Initialize weights with ideal fractions
                fj = np.zeros(ncomp)
                # Loop through components
                for k, comp in enumerate(comps):
                    # Get fraction
                    fj[k] = np.mean(Fcomp[comp][maskj] / FT[maskj])
                    # Save fraction as initial weight
                    if coeff == "CLL":
                        # Special case; no nominal *CLL* to shift
                        continue
                    # Weight name
                    wcol = "w%s.%s" % (coeff, coeff)
                    # Save it
                    w[comp][wcol][j] = fj[k]
                # Handle trivial cases first
                if not (qdy or qdz):
                    # Now moments shifted by forces
                    continue
                elif (coeff == "CY") and (not qdz):
                    # No *CLL* shifted by *CY*
                    continue
                elif (coeff == "CN") and (not qdy):
                    # No *CLL* shifted by *CN*
                    continue
                # Check which coefficient is under study
                if coeff == "CA":
                    # Calculate unintentional impact to *CLM*
                    d1 = np.dot(Z, fj)
                    # Calculate unintentional impact to *CLN*
                    d2 = np.dot(Y, fj)
                    # Create matrix
                    A = np.vstack((
                        np.ones((1, ncomp)),
                        [Z],
                        [Y]))
                    # Create constraints
                    b = np.array([0.0, -d1, -d2])
                    # Check trivial constraints
                    A = A[[True, qdz, qdy], :]
                    b = b[[True, qdz, qdy]]
                    # Solve least squares system
                    x, _, _, _ = np.linalg.lstsq(A, b)
                    # Apply *x* to fix any unintentional shifts
                    wj = fj + x
                    # Save the shifted weights
                    for k, comp in enumerate(comps):
                        # Weight name
                        w[comp]["wCA.CA"][j] = wj[k]
                elif coeff == "CY":
                    # Calculate unintentional impact to *CLL*
                    d1 = np.dot(Z, fj)
                    # Create matrix
                    A = np.vstack((
                        np.ones((1, ncomp)),
                        [Z]))
                    # Create constraints
                    b = np.array([0.0, -d1])
                    # Solve least squares system
                    x, _, _, _ = np.linalg.lstsq(A, b)
                    # Apply *x* to fix any unintentional shifts
                    wj = fj + x
                    # Save the shifted weights
                    for k, comp in enumerate(comps):
                        # Weight name       
                        w[comp]["wCY.CY"][j] = wj[k]
                elif coeff == "CN":
                    # Calculate unintentional impact to *CLL*
                    d1 = -np.dot(Y, fj)
                    # Create matrix
                    A = np.vstack((
                        np.ones((1, ncomp)),
                        [-Y]))
                    # Create constraints
                    b = np.array([0.0, -d1])
                    # Solve least squares system
                    x, _, _, _ = np.linalg.lstsq(A, b)
                    # Apply *x* to fix any unintentional shifts
                    wj = fj + x
                    # Save the shifted weights
                    for k, comp in enumerate(comps):
                        # Weight name
                        w[comp]["wCN.CN"][j] = wj[k]
                elif coeff == "CLL":
                    # Calculate total offset radii
                    R2 = Y*Y + Z*Z
                    # Which are nonzero?
                    qr = (R2 > 0)
                    # Initialize *CLL* divider weights
                    fY = np.zeros(ncomp)
                    fN = np.zeros(ncomp)
                    # Divide up *CLL* weight to
                    fY[qr] = Z[qr] / R2[qr] * fj[qr]
                    fN[qr] = -Y[qr] / R2[qr] * fj[qr]
                    # Calculate unintentional impact to *CY* and *CN*
                    d1 = np.sum(fY)
                    d2 = np.sum(fN)
                    # Create matrix
                    A = np.vstack((
                        [np.hstack((Z, -Y))],
                        np.hstack((np.ones(ncomp), np.zeros(ncomp))),
                        np.hstack((np.zeros(ncomp), np.ones(ncomp)))))
                    # Create constraints
                    b = np.array([0, -d1, -d2])
                    # Solve least squares system
                    x, _, _, _ = np.linalg.lstsq(A, b)
                    # Apply *x* to fix any unintentional shifts
                    wY = fY + x[:3]
                    wN = fN + x[3:]
                    # Save the shifted weights
                    for k, comp in enumerate(comps):
                        # Weight name
                        w[comp]["wCY.CLL"][j] = wY[k]
                        w[comp]["wCN.CLL"][j] = wN[k]
                elif coeff in ["CLM", "CLN"]:
                    # Already handled by desired fraction
                    pass
       # --- Output ---
        # Output
        return w

    # Calculate each component's contribution to adjusted loads
    def make_ll3x_fractions(self, comps, scol=None, **kw):
        r"""Calculate each component's contribution to integral forces

        :Call:
            >>> f = db.make_ll3x_fractions(comps, scol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *comp*: :class:`str`
                Single component (trivial output)
            *comps*: :class:`list`\ [:class:`str`]
                List of components to divide integral F&M
            *scol*: {``None``} | :class:`str`
                Column used to slice database; output will be constant
                on each slice
            *CompFMCols*: :class:`dict`\ [:class:`dict`]
                Columns to use as integral of force and moment on each
                *comp*. Defaults filled in by *comp*\ .*coeff* for
                *coeff* in ``["CA", "CY", "CN", "CLL", "CLM", "CLN"]``
        :Outputs:
            *f*: :class:`dict`\ [:class:`dict`\ [:class:`np.ndarray`]]
                Fraction for each *comp* and each *coeff*
        :Versions:
            * 2020-06-16 ``@ddalle``: First version
            * 2020-06-19 ``@ddalle``: Weights same size of bkpts
        """
        # Check the component list
        comps = self._check_ll3x_comps(comps)
        # Transfer options
        opts = _LL3XOpts(_section="fractions_make", **kw)
        # Options to integrator
        kw_frac = opts.section_options("fractions")
        # Name of slice field
        ocol = opts.get_option("SliceColSave", "adjust.%s" % scol)
        # Initialize output if all cols present
        f = {}
        # Check fir slice values
        if ocol in self:
            # Save it to *f*
            f["s"] = self[ocol]
            qmake = False
        else:
            qmake = True
        # Loop through the components
        for comp in comps:
            # Get column names
            fcols = self._getcols_fmfrac_comp(None, comp=comp, **opts)
            # Unpack
            colCA = fcols["CA"]
            colCY = fcols["CY"]
            colCN = fcols["CN"]
            colCl = fcols["CLL"]
            colCm = fcols["CLM"]
            colCn = fcols["CLN"]
            # Check if all are present
            if colCA not in self:
                break
            elif colCY not in self:
                break
            elif colCN not in self:
                break
            elif colCl not in self:
                break
            elif colCm not in self:
                break
            elif colCn not in self:
                break
            else:
                # Everything already present
                f[comp] = {
                    "CA": self[colCA],
                    "CY": self[colCY],
                    "CN": self[colCN],
                    "CLL": self[colCl],
                    "CLM": self[colCm],
                    "CLN": self[colCn],
                }
        else:
            # If no ``break`` encountered, return output
            if not qmake:
                return f
        # Generate the fractions
        return self.create_ll3x_fractions(comps, scol, **kw_frac)

    # Calculate each component's contribution to adjusted loads
    def create_ll3x_fractions(self, comps, scol=None, **kw):
        r"""Calculate each component's contribution to integral forces

        :Call:
            >>> f = db.create_ll3x_fractions(comps, scol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *comp*: :class:`str`
                Single component (trivial output)
            *comps*: :class:`list`\ [:class:`str`]
                List of components to divide integral F&M
            *scol*: {``None``} | :class:`str`
                Column used to slice database; output will be constant
                on each slice
            *CompFMCols*: :class:`dict`\ [:class:`dict`]
                Columns to use as integral of force and moment on each
                *comp*. Defaults filled in by *comp*\ .*coeff* for
                *coeff* in ``["CA", "CY", "CN", "CLL", "CLM", "CLN"]``
        :Outputs:
            *f*: :class:`dict`\ [:class:`dict`\ [:class:`np.ndarray`]]
                Fraction for each *comp* and each *coeff*
        :Versions:
            * 2020-06-16 ``@ddalle``: First version
            * 2020-06-19 ``@ddalle``: Weights same size of bkpts
        """
        # Check and process component list
        comps = self._check_ll3x_comps(comps)
        # Form options
        opts = _LL3XOpts(_section="fractions_make", **kw)
        # Get options for genr8 function
        kwf = opts.section_options("fractions")
        # Calculate fractions
        f = self.genr8_ll3x_fractions(comps, scol, **kwf)
        # Get slice values
        s = f.get("s")
        # Save slice values
        if s is not None:
            # Name of slice field
            col = opts.get_option("SliceColSave", "adjust.%s" % scol)
            # Save values and definition
            self.save_col(col, s)
        # Loop through the components
        for comp in comps:
            # Get column names
            fcols = self._getcols_fmfrac_comp(None, comp=comp, **opts)
            # Data for this component
            fcomp = f[comp]
            # Unpack
            colCA = fcols["CA"]
            colCY = fcols["CY"]
            colCN = fcols["CN"]
            colCl = fcols["CLL"]
            colCm = fcols["CLM"]
            colCn = fcols["CLN"]
            # Save the integrated columns
            self.save_col(colCA, fcomp["CA"])
            self.save_col(colCY, fcomp["CY"])
            self.save_col(colCN, fcomp["CN"])
            self.save_col(colCl, fcomp["CLL"])
            self.save_col(colCm, fcomp["CLM"])
            self.save_col(colCn, fcomp["CLN"])
        # Output
        return f

    # Calculate each component's contribution to adjusted loads
    def genr8_ll3x_fractions(self, comps, scol=None, **kw):
        r"""Calculate each component's contribution to integral forces

        :Call:
            >>> f = db.genr8_ll3x_fractions(comp, scol=None, **kw)
            >>> f = db.genr8_ll3x_fractions(comps, scol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *comp*: :class:`str`
                Single component (trivial output)
            *comps*: :class:`list`\ [:class:`str`]
                List of components to divide integral F&M
            *scol*: {``None``} | :class:`str`
                Column used to slice database; output will be constant
                on each slice
            *CompFMCols*: :class:`dict`\ [:class:`dict`]
                Columns to use as integral of force and moment on each
                *comp*. Defaults filled in by *comp*\ .*coeff* for
                *coeff* in ``["CA", "CY", "CN", "CLL", "CLM", "CLN"]``
        :Outputs:
            *f*: :class:`dict`\ [:class:`dict`\ [:class:`np.ndarray`]]
                Fraction for each *comp* and each *coeff*
            *f[comp][coeff]*: :class:`float` | :class:`np.ndarray`
                Average contribution of *comp* to *coeff* for each value
                of *scol*
            *f["s"]*: ``None`` | :class:`np.ndarray`
                Unique values of *scol*
        :Versions:
            * 2020-06-12 ``@ddalle``: First version
            * 2020-06-19 ``@ddalle``: Weights same size of bkpts
        """
       # --- Checks ---
        # Check and process component list
        comps = self._check_ll3x_comps(comps)
        # Form options
        opts = _LL3XOpts(_section="fractions", **kw)
       # --- FM Cols ---
        # Get columns that determine deltas for each component
        fmcols = opts.get_option("CompFMCols", {})
        # Loop through components
        for comp in comps:
            # Get integral col names for this componet
            cols = self._getcols_ll3x_fmcomp(None, comp, CompFMCols=fmcols)
            # Check presence
            for coeff, col in cols.items():
                if col not in self:
                    raise KeyError(
                        "%s col '%s' for comp '%s' not present"
                        % (coeff, col, comp))
            # Save completed list
            fmcols[comp] = cols
            # Reference column
            col = cols["CA"]
       # --- Mask & Slicing ---
        # Get mask option
        mask = opts.get_option("mask")
        # Get indices
        mask_index = self.prep_mask(mask, col=col)
        # Slice column
        if scol is None:
            # Single slice with all cases in *mask*
            masks = [self.prep_mask(mask, col=col)]
            # No slice values
            s = None
            # One mask
            nmask = 1
        else:
            # Get unique values
            s = self.genr8_bkpts(scol, nmin=5, tol=1e-8, mask=mask)
            # Divide those values into slices
            masks, _ = self.find([scol], s, mapped=True)
            # Number of masks
            nmask = s.size
       # --- Fractions ---
        # Initialize fractions
        f = {comp: {} for comp in comps}
        # Save the slice values
        f["s"] = s
        # Loop through coefficients
        for coeff in _coeffs:
            # Initialize total
            FT = 0.0
            # Initialize dict per *comp*
            Fcomp = {}
            # Loop through components
            for comp in comps:
                # Component column
                col = fmcols[comp][coeff]
                # Get values
                F = np.abs(self[col])
                # Increment total
                FT += F
                # Save it
                Fcomp[comp] = F
            # Component again
            for comp in comps:
                # Get fraction
                fcomp = Fcomp[comp] / FT
                # Column name
                col = fmcols[comp][coeff]
                # Initialize output
                fcc = np.zeros(nmask)
                # Slices
                for j, maskj in enumerate(masks):
                    # Get mean
                    fj = np.mean(fcomp[maskj])
                    # Save it
                    fcc[j] = fj
                # Save fraction
                f[comp][coeff] = fcc
       # --- Output ---
        # Output
        return f

   # --- Integration ---
    # # Generate integrals and save
    def make_ll3x_integrals(self, comps, **kw):
        r"""Integrate line loads for several columns

        For each *comp*, new loads are only integrated if any one of
        the six coefficients are missing for that component.

        :Call:
            >>> fm = db.make_ll3x_integrals(comps, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *nPOD*: {``10``} | ``None`` | :class:`int` > 0
                Number of POD/SVD modes to use during optimization
            *mask*: {``None``} | :class:`np.ndarray`
                Mask or indices of which cases to include in POD
                calculation
            *method*: {``"trapz"``} | ``"left"`` | **callable**
                Integration method used to integrate columns
        :Outputs:
            *fm*: :class:`dict`\ [:class:`np.ndarray`]
                Integrated force/moment for each coefficient
            *fm[comp][coeff]*: :class:`np.ndarray`
                Integrated *coeff* from line load for comp *comp*
        :Versions:
            * 2020-06-15 ``@ddalle``: First version
        """
        # Check the component list
        comps = self._check_ll3x_comps(comps)
        # Transfer options
        opts = _LL3XOpts(_section="integrals_make", **kw)
        # Options to integrator
        kw_int = opts.section_options("integrals")
        # Initialize output
        fm = {}
        # Loop through the components
        for comp in comps:
            # Get column names
            fmcols = self._getcols_ll3x_fmcomp(None, comp=comp, **opts)
            # Unpack
            colCA = fmcols["CA"]
            colCY = fmcols["CY"]
            colCN = fmcols["CN"]
            colCl = fmcols["CLL"]
            colCm = fmcols["CLM"]
            colCn = fmcols["CLN"]
            # Check if all are present
            if colCA not in self:
                pass
            elif colCY not in self:
                pass
            elif colCN not in self:
                pass
            elif colCl not in self:
                pass
            elif colCm not in self:
                pass
            elif colCn not in self:
                pass
            else:
                # Everything already present
                fm[comp] = {
                    "CA": self[colCA],
                    "CY": self[colCY],
                    "CN": self[colCN],
                    "CLL": self[colCl],
                    "CLM": self[colCm],
                    "CLN": self[colCn],
                }
                # Move to next component
                continue
            # Generate the integrals
            fmcomp = self.genr8_ll3x_integrals([comp], **kw_int)
            # Select the one column
            fmcomp = fmcomp[comp]
            # Save the integrated columns
            self.save_col(colCA, fmcomp["CA"])
            self.save_col(colCY, fmcomp["CY"])
            self.save_col(colCN, fmcomp["CN"])
            self.save_col(colCl, fmcomp["CLL"])
            self.save_col(colCm, fmcomp["CLM"])
            self.save_col(colCn, fmcomp["CLN"])
            # Save
            fm[comp] = fmcomp
        # Output
        return fm

    # Generate integrals and save
    def create_ll3x_integrals(self, comps, **kw):
        r"""Integrate line loads for several columns

        :Call:
            >>> fm = db.create_ll3x_integrals(comps, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *nPOD*: {``10``} | ``None`` | :class:`int` > 0
                Number of POD/SVD modes to use during optimization
            *mask*: {``None``} | :class:`np.ndarray`
                Mask or indices of which cases to include in POD
                calculation
            *method*: {``"trapz"``} | ``"left"`` | **callable**
                Integration method used to integrate columns
        :Outputs:
            *fm*: :class:`dict`\ [:class:`np.ndarray`]
                Integrated force/moment for each coefficient
            *fm[comp][coeff]*: :class:`np.ndarray`
                Integrated *coeff* from line load for comp *comp*
        :Versions:
            * 2020-06-15 ``@ddalle``: First version
        """
        # Check the component list
        comps = self._check_ll3x_comps(comps)
        # Transfer options
        opts = _LL3XOpts(_section="integrals_make", **kw)
        # Options to integrator
        kw_int = opts.section_options("integrals")
        # Generate the integrals
        fm = self.genr8_ll3x_integrals(comps, **kw_int)
        # Loop through the components
        for comp in comps:
            # Get column names
            fmcols = self._getcols_ll3x_fmcomp(None, comp=comp, **opts)
            # Unpack
            colCA = fmcols["CA"]
            colCY = fmcols["CY"]
            colCN = fmcols["CN"]
            colCl = fmcols["CLL"]
            colCm = fmcols["CLM"]
            colCn = fmcols["CLN"]
            # Component data
            fmcomp = fm[comp]
            # Save the integrated columns
            self.save_col(colCA, fmcomp["CA"])
            self.save_col(colCY, fmcomp["CY"])
            self.save_col(colCN, fmcomp["CN"])
            self.save_col(colCl, fmcomp["CLL"])
            self.save_col(colCm, fmcomp["CLM"])
            self.save_col(colCn, fmcomp["CLN"])
        # Output
        return fm

    # Generate integrals
    def genr8_ll3x_integrals(self, comps, **kw):
        r"""Integrate line load columns for several components

        :Call:
            >>> fm = db.genr8_ll3x_integrals(comps, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *comps*: :class:`list`\ [:class:`str`]
                List of components to integrate
            *CompLLCols*: :class:`dict`\ [:class:`list`]
                Optional lists of line load *cols* for each *comp*
                in *comps*; default is ``"<comp>.d<coeff>"``
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *mask*: {``None``} | :class:`np.ndarray`
                Mask or indices of which cases to include in POD
                calculation
            *method*: {``"trapz"``} | ``"left"`` | **callable**
                Integration method used to integrate columns
        :Outputs:
            *fm*: :class:`dict`\ [:class:`np.ndarray`]
                Integrated force/moment for each coefficient
            *fm[comp][coeff]*: :class:`np.ndarray`
                Integrated *coeff* from line load for comp *comp*
        :Versions:
            * 2020-06-15 ``@ddalle``: First version
        """
        # Check *comps*
        comps = self._check_ll3x_comps(comps)
        # Check options
        opts = _LL3XOpts(**kw)
        # Initialize output
        fm = {}
        # Get component cols
        compcols = opts.get_option("CompLLCols", {})
        # Coordinate shifts
        compy = opts.get_option("CompY", {})
        compz = opts.get_option("CompZ", {})
        # Loop through components
        for comp in comps:
            # Get columns
            cols = compcols.get(comp, [])
            # Ensure list
            if not isinstance(cols, list):
                raise TypeError("LL cols for comp '%s' is not a list" % comp)
            # Number of columns specified
            ncol = len(cols)
            # Default cols
            coldefs = ["%s.d%s" % (comp, col) for col in ["CA", "CY", "CN"]]
            # Add default columns
            if ncol < 3:
                cols += coldefs[ncol:]
            # Get coordinate positions
            kwcomp = opts.section_options("integrals_comp")
            # Set *y* and *z*
            kwcomp["y"] = compy.get(comp, 0.0)
            kwcomp["z"] = compz.get(comp, 0.0)
            # Generate line loads
            fm[comp] = self.genr8_ll3x_comp_integrals(cols, **kwcomp)
        # Output
        return fm

    # Generate integrals and save
    def make_ll3x_comp_integrals(self, cols, **kw):
        r"""Integrate 3 or 6 line load columns

        :Call:
            >>> fm = db.make_ll3x_comp_integrals(cols, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *nPOD*: {``10``} | ``None`` | :class:`int` > 0
                Number of POD/SVD modes to use during optimization
            *mask*: {``None``} | :class:`np.ndarray`
                Mask or indices of which cases to include in POD
                calculation
            *method*: {``"trapz"``} | ``"left"`` | **callable**
                Integration method used to integrate columns
        :Outputs:
            *fm*: :class:`dict`\ [:class:`np.ndarray`]
                Integrated force/moment for each coefficient
            *fm["CA"]*: :class:`np.ndarray`
                Integrated *cols[0]*
            *fm["CY"]*: :class:`np.ndarray`
                Integrated *cols[1]*
            *fm["CN"]*: :class:`np.ndarray`
                Integrated *cols[2]*
            *fm["CLL"]*: :class:`np.ndarray`
                Integrated *cols[3]* or rolling moment induced by
                *CY* and *CN*
            *fm["CLM"]*: :class:`np.ndarray`
                Integrated *cols[4]* or pitching moment integrated
                from *cols[2]* plus what's induced by *CA*
            *fm["CLN"]*: :class:`np.ndarray`
                Integrated *cols54]* or pitching moment integrated
                from *cols[1]* plus what's induced by *CA*
        :Versions:
            * 2020-06-12 ``@ddalle``: First version
        """
        # Transfer options
        opts = _LL3XOpts(_section="integrals_comp_make", **kw)
        # Check the columns
        self._check_ll3x_cols(cols)
        # Options to integrator
        kw_int = opts.section_options("integrals_comp")
        # Get column names
        fmcols = self._getcols_ll3x_fmcomp(cols, **opts)
        # Unpack
        colCA = fmcols["CA"]
        colCY = fmcols["CY"]
        colCN = fmcols["CN"]
        colCl = fmcols["CLL"]
        colCm = fmcols["CLM"]
        colCn = fmcols["CLN"]
        # Check if present
        if colCA not in self:
            pass
        elif colCY not in self:
            pass
        elif colCN not in self:
            pass
        elif colCl not in self:
            pass
        elif colCm not in self:
            pass
        elif colCn not in self:
            pass
        else:
            # All present!
            return {
                "CA": self[colCA],
                "CY": self[colCY],
                "CN": self[colCN],
                "CLL": self[colCl],
                "CLM": self[colCm],
                "CLN": self[colCn],
            }
        # Generate the integrals
        fm = self.genr8_ll3x_comp_integrals(cols, **kw_int)
        # Save the integrated columns
        self.save_col(colCA, fm["CA"])
        self.save_col(colCY, fm["CY"])
        self.save_col(colCN, fm["CN"])
        self.save_col(colCl, fm["CLL"])
        self.save_col(colCm, fm["CLM"])
        self.save_col(colCn, fm["CLN"])
        # Output
        return fm

    # Generate integrals and save
    def create_ll3x_comp_integrals(self, cols, **kw):
        r"""Integrate 3 or 6 line load columns

        :Call:
            >>> fm = db.create_ll3x_comp_integrals(cols, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *nPOD*: {``10``} | ``None`` | :class:`int` > 0
                Number of POD/SVD modes to use during optimization
            *mask*: {``None``} | :class:`np.ndarray`
                Mask or indices of which cases to include in POD
                calculation
            *method*: {``"trapz"``} | ``"left"`` | **callable**
                Integration method used to integrate columns
        :Outputs:
            *fm*: :class:`dict`\ [:class:`np.ndarray`]
                Integrated force/moment for each coefficient
            *fm["CA"]*: :class:`np.ndarray`
                Integrated *cols[0]*
            *fm["CY"]*: :class:`np.ndarray`
                Integrated *cols[1]*
            *fm["CN"]*: :class:`np.ndarray`
                Integrated *cols[2]*
            *fm["CLL"]*: :class:`np.ndarray`
                Integrated *cols[3]* or rolling moment induced by
                *CY* and *CN*
            *fm["CLM"]*: :class:`np.ndarray`
                Integrated *cols[4]* or pitching moment integrated
                from *cols[2]* plus what's induced by *CA*
            *fm["CLN"]*: :class:`np.ndarray`
                Integrated *cols54]* or pitching moment integrated
                from *cols[1]* plus what's induced by *CA*
        :Versions:
            * 2020-06-12 ``@ddalle``: First version
        """
        # Transfer options
        opts = _LL3XOpts(_section="integrals_comp_make", **kw)
        # Check the columns
        self._check_ll3x_cols(cols)
        # Options to integrator
        kw_int = opts.section_options("integrals_comp")
        # Get column names
        fmcols = self._getcols_ll3x_fmcomp(cols, **opts)
        # Unpack
        colCA = fmcols["CA"]
        colCY = fmcols["CY"]
        colCN = fmcols["CN"]
        colCl = fmcols["CLL"]
        colCm = fmcols["CLM"]
        colCn = fmcols["CLN"]
        # Generate the integrals
        fm = self.genr8_ll3x_comp_integrals(cols, **kw_int)
        # Save the integrated columns
        self.save_col(colCA, fm["CA"])
        self.save_col(colCY, fm["CY"])
        self.save_col(colCN, fm["CN"])
        self.save_col(colCl, fm["CLL"])
        self.save_col(colCm, fm["CLM"])
        self.save_col(colCn, fm["CLN"])
        # Output
        return fm

    # Generate integrals
    def genr8_ll3x_comp_integrals(self, cols, **kw):
        r"""Integrate 3 or 6 line load columns

        :Call:
            >>> fm = db.genr8_ll3x_comp_integrals(cols, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *mask*: {``None``} | :class:`np.ndarray`
                Mask or indices of which cases to include in POD
                calculation
            *method*: {``"trapz"``} | ``"left"`` | **callable**
                Integration method used to integrate columns
        :Outputs:
            *fm*: :class:`dict`\ [:class:`np.ndarray`]
                Integrated force/moment for each coefficient
            *fm["CA"]*: :class:`np.ndarray`
                Integrated *cols[0]*
            *fm["CY"]*: :class:`np.ndarray`
                Integrated *cols[1]*
            *fm["CN"]*: :class:`np.ndarray`
                Integrated *cols[2]*
            *fm["CLL"]*: :class:`np.ndarray`
                Integrated *cols[3]* or rolling moment induced by
                *CY* and *CN*
            *fm["CLM"]*: :class:`np.ndarray`
                Integrated *cols[4]* or pitching moment integrated
                from *cols[2]* plus what's induced by *CA*
            *fm["CLN"]*: :class:`np.ndarray`
                Integrated *cols54]* or pitching moment integrated
                from *cols[1]* plus what's induced by *CA*
        :Versions:
            * 2020-06-11 ``@ddalle``: First version
        """
        # Check options
        opts = _LL3XOpts(_section="integrals_comp", **kw)
        # Check column list
        self._check_ll3x_cols(cols)
        # Get line load columns
        llcols = self._getcols_ll_comp(cols, **opts)
        # Unpack
        colCA = llcols["CA"]
        colCY = llcols["CY"]
        colCN = llcols["CN"]
        colCLL = llcols["CLL"]
        colCLM = llcols["CLM"]
        colCLN = llcols["CLN"]
        # Get *xcols*
        xcolCA = self._getcol_ll_xcol(colCA, **opts)
        xcolCY = self._getcol_ll_xcol(colCA, **opts)
        xcolCN = self._getcol_ll_xcol(colCA, **opts)
        xcolCLL = self._getcol_ll_xcol(colCA, **opts)
        xcolCLM = self._getcol_ll_xcol(colCA, **opts)
        xcolCLN = self._getcol_ll_xcol(colCA, **opts)
        # Remove *xcol* from *opts* if specified
        opts.pop("xcol", None)
        # Reference length
        Lref = opts.get_option("Lref", self.__dict__.get("Lref", 1.0))
        # Get MRP
        xMRP = opts.get_option("xMRP", self.__dict__.get("xMRP", 0.0))
        yMRP = opts.get_option("yMRP", self.__dict__.get("yMRP", 0.0))
        zMRP = opts.get_option("zMRP", self.__dict__.get("zMRP", 0.0))
        # Get coordinates
        y = opts.get_option("y", yMRP)
        z = opts.get_option("z", zMRP)
        # Integrate forces
        CA = self.genr8_integral(colCA, xcol=xcolCA, **opts)
        CY = self.genr8_integral(colCY, xcol=xcolCY, **opts)
        CN = self.genr8_integral(colCN, xcol=xcolCN, **opts)
        # Check for moment loads
        if colCLL in self and colCLM in self and colCLN in self:
            # Integrate moment loads
            CLL = self.genr8_integral(colCLL, xcol=xcolCLL, **opts)
            CLM = self.genr8_integral(colCLM, xcol=xcolCLM, **opts)
            CLN = self.genr8_integral(colCLN, xcol=xcolCLN, **opts)
        else:
            # Create moment columns
            self.create_dclm(colCN, **opts)
            self.create_dcln(colCY, **opts)
            # Integrate moments
            CLL = np.zeros_like(CA)
            CLM = self.genr8_integral(colCLM, xcol=xcolCLM, **opts)
            CLN = self.genr8_integral(colCLN, xcol=xcolCLN, **opts)
            # Moments due to MRP offset from line load center
            CLM += CA * (z - zMRP) / Lref
            CLN += CA * (y - yMRP) / Lref
            CLL += CY * (z - zMRP) / Lref
            CLL -= CN * (y - yMRP) / Lref
        # Output
        return {
            "CA": CA,
            "CY": CY,
            "CN": CN,
            "CLL": CLL,
            "CLM": CLM,
            "CLN": CLN,
        }

   # --- Basis ---
    # Create basis for line load of one component by slice
    def make_ll3x_basis(self, comps, scol=None, **kw):
        r"""Get [and calculate] SVD-based basis for LL3X adjustments

        This is a highly customized function (hence the somewhat
        obscure name) that adjusts three line load force cols that
        are a function of *x*.  It adjusts for five scenarios:

            * Adjust the *dCA* load such that integrated *CA* is
              increased by ``1.0``
            * Adjust *dCY* such that *CY* increases ``1.0`` and *CLN*
              is unchanged
            * Adjust *dCY* such that *CY* is unchanged and *CLN*
              increases ``1.0``
            * Adjust *dCN* such that *CN* increases ``1.0`` and *CLM*
              is unchanged
            * Adjust *dCN* such that *CN* is unchanged and *CLM*
              increases ``1.0``

        :Call:
            >>> basis = db.make_ll3x_basis(comps, scol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3
                * (*col1*, *col2*, *col3*)

                List/tuple of column names for *CA*, *CY*, and *CN*
                line loads
            *scol*: {``None``} | :class:`str`
                Name of slice col; calculate basis for each value in
                *db.bkpts[scol]*
            *nPOD*: {``10``} | ``None`` | :class:`int` > 0
                Number of POD/SVD modes to use during optimization
            *mask*: {``None``} | :class:`np.ndarray`
                Mask or indices of which cases to include in POD
                calculation
            *method*: {``"trapz"``} | ``"left"`` | **callable**
                Integration method used to integrate columns
            *CompLLCols*: :class:`dict`\ [:class:`dict`]
                Line load column names to use for each *comp*
            *CompLLBasisCols*: :class:`dict`\ [:class:`dict`]
                Names to use when saving adjustment bases
        :Outputs:
            *basis*: :class:`dict`
                Basis adjustment loads for five scenarios
            *basis["s"]*: :class:`np.ndarray` | ``None``
                Unique values of *scol*
            *basis[comp]["dCA.CA"]*: :class:`np.ndarray`
                Delta *dCA* load to adjust *CA* by ``1.0``
            *basis[comp]["dCY.CY"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CY* by ``1.0``
            *basis[comp]["dCY.CLN"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CLN* by ``1.0``
            *basis[comp]["dCN.CN"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CN* by ``1.0``
            *basis[comp]["dCN.CLM"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CLM* by ``1.0``
        :Versions:
            * 2020-06-19 ``@ddalle``: Version 1.0
        """
        # Check the component list
        comps = self._check_ll3x_comps(comps)
        # Transfer options
        opts = _LL3XOpts(_section="basis_make", **kw)
        # Name of slice field
        ocol = opts.get_option("SliceColSave", "adjust.%s" % scol)
        # Initialize output if all cols present
        basis = {}
        # Check fir slice values
        if ocol in self:
            # Save it to *f*
            basis["s"] = self[ocol]
            # No reason to make
            qmake = False
        else:
            # Need to make if *ocol* expected
            qmake = scol is not None
        # Loop through the components
        for comp in comps:
            # Don't bother if *qmake* already triggered
            if qmake:
                break
            # Get column names
            bcols = self._getcols_llb_comp(None, comp=comp, **opts)
            # Unpack
            colCAA = bcols["dCA.CA"]
            colCYY = bcols["dCY.CY"]
            colCNN = bcols["dCN.CN"]
            colCNm = bcols["dCN.CLM"]
            colCYn = bcols["dCY.CLN"]
            # Check if all are present
            if colCAA not in self:
                break
            elif colCYY not in self:
                break
            elif colCNN not in self:
                break
            elif colCNm not in self:
                break
            elif colCYn not in self:
                break
            else:
                # Everything already present
                basis[comp] = {
                    "dCA.CA": self[colCAA],
                    "dCY.CY": self[colCYY],
                    "dCN.CN": self[colCNN],
                    "dCY.CLN": self[colCYn],
                    "dCN.CLM": self[colCNm],
                }
        else:
            # If no ``break`` encountered, return output
            return basis
        # Generate the weights and return them
        return self.create_ll3x_basis(comps, scol, **opts)

    # Create basis for line load of one component by slice
    def create_ll3x_basis(self, comps, scol=None, **kw):
        r"""Calculate and save SVD-based basis for LL3X adjustments

        This is a highly customized function (hence the somewhat
        obscure name) that adjusts three line load force cols that
        are a function of *x*.  It adjusts for five scenarios:

            * Adjust the *dCA* load such that integrated *CA* is
              increased by ``1.0``
            * Adjust *dCY* such that *CY* increases ``1.0`` and *CLN*
              is unchanged
            * Adjust *dCY* such that *CY* is unchanged and *CLN*
              increases ``1.0``
            * Adjust *dCN* such that *CN* increases ``1.0`` and *CLM*
              is unchanged
            * Adjust *dCN* such that *CN* is unchanged and *CLM*
              increases ``1.0``

        :Call:
            >>> basis = db.create_ll3x_basis(comps, scol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3
                * (*col1*, *col2*, *col3*)

                List/tuple of column names for *CA*, *CY*, and *CN*
                line loads
            *scol*: {``None``} | :class:`str`
                Name of slice col; calculate basis for each value in
                *db.bkpts[scol]*
            *nPOD*: {``10``} | ``None`` | :class:`int` > 0
                Number of POD/SVD modes to use during optimization
            *mask*: {``None``} | :class:`np.ndarray`
                Mask or indices of which cases to include in POD
                calculation
            *method*: {``"trapz"``} | ``"left"`` | **callable**
                Integration method used to integrate columns
            *CompLLCols*: :class:`dict`\ [:class:`dict`]
                Line load column names to use for each *comp*
            *CompLLBasisCols*: :class:`dict`\ [:class:`dict`]
                Names to use when saving adjustment bases
        :Outputs:
            *basis*: :class:`dict`
                Basis adjustment loads for five scenarios
            *basis["s"]*: :class:`np.ndarray` | ``None``
                Unique values of *scol*
            *basis[comp]["dCA.CA"]*: :class:`np.ndarray`
                Delta *dCA* load to adjust *CA* by ``1.0``
            *basis[comp]["dCY.CY"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CY* by ``1.0``
            *basis[comp]["dCY.CLN"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CLN* by ``1.0``
            *basis[comp]["dCN.CN"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CN* by ``1.0``
            *basis[comp]["dCN.CLM"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CLM* by ``1.0``
        :Versions:
            * 2020-06-19 ``@ddalle``: Version 1.0
        """
        # Check the component list
        comps = self._check_ll3x_comps(comps)
        # Transfer options
        opts = _LL3XOpts(_section="basis_make", **kw)
        # Options to integrator
        kw_b = opts.section_options("basis")
        # Generate the bases
        bases = self.genr8_ll3x_basis(comps, scol, **kw_b)
        # Name of slice field
        ocol = opts.get_option("SliceColSave", "adjust.%s" % scol)
        # Save the slice column
        if scol is not None:
            # Save values and definition
            self.save_col(ocol, bases.get("s"))
        # Loop through the components
        for comp in comps:
            # Get column names
            bcols = self._getcols_llb_comp(None, comp=comp, **opts)
            # Unpack
            colCAA = bcols["dCA.CA"]
            colCYY = bcols["dCY.CY"]
            colCNN = bcols["dCN.CN"]
            colCNm = bcols["dCN.CLM"]
            colCYn = bcols["dCY.CLN"]
            # Component data
            basis = bases[comp]
            # Save the basis cols
            self.save_col(colCAA, basis["dCA.CA"])
            self.save_col(colCYY, basis["dCY.CY"])
            self.save_col(colCNN, basis["dCN.CN"])
            self.save_col(colCNm, basis["dCN.CLM"])
            self.save_col(colCYn, basis["dCY.CLN"])
        # Output
        return bases

    # Create basis for line load of one component by slice
    def genr8_ll3x_basis(self, comps, scol=None, **kw):
        r"""Calculate SVD-based basis for adjusting line loads

        This is a highly customized function (hence the somewhat
        obscure name) that adjusts three line load force cols that
        are a function of *x*.  It adjusts for five scenarios:

            * Adjust the *dCA* load such that integrated *CA* is
              increased by ``1.0``
            * Adjust *dCY* such that *CY* increases ``1.0`` and *CLN*
              is unchanged
            * Adjust *dCY* such that *CY* is unchanged and *CLN*
              increases ``1.0``
            * Adjust *dCN* such that *CN* increases ``1.0`` and *CLM*
              is unchanged
            * Adjust *dCN* such that *CN* is unchanged and *CLM*
              increases ``1.0``

        :Call:
            >>> basis = db.genr8_ll3x_basis(comps, scol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3
                * (*col1*, *col2*, *col3*)

                List/tuple of column names for *CA*, *CY*, and *CN*
                line loads
            *scol*: {``None``} | :class:`str`
                Name of slice col; calculate basis for each value in
                *db.bkpts[scol]*
            *nPOD*: {``10``} | ``None`` | :class:`int` > 0
                Number of POD/SVD modes to use during optimization
            *mask*: {``None``} | :class:`np.ndarray`
                Mask or indices of which cases to include in POD
                calculation
            *method*: {``"trapz"``} | ``"left"`` | **callable**
                Integration method used to integrate columns
            *CompLLCols*: :class:`dict`\ [:class:`dict`]
                Line load column names to use for each *comp*
        :Outputs:
            *basis*: :class:`dict`
                Basis adjustment loads for five scenarios
            *basis["s"]*: :class:`np.ndarray` | ``None``
                Unique values of *scol*
            *basis[comp]["dCA.CA"]*: :class:`np.ndarray`
                Delta *dCA* load to adjust *CA* by ``1.0``
            *basis[comp]["dCY.CY"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CY* by ``1.0``
            *basis[comp]["dCY.CLN"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CLN* by ``1.0``
            *basis[comp]["dCN.CN"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CN* by ``1.0``
            *basis[comp]["dCN.CLM"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CLM* by ``1.0``
        :Versions:
            * 2020-06-04 ``@ddalle``: Version 1.0
            * 2020-06-19 ``@ddalle``: Version 2.0
        """
        # Check *comps*
        comps = self._check_ll3x_comps(comps)
        # Check options
        opts = _LL3XOpts(_section="basis", **kw)
        # Initialize output
        basis = {}
        # Get component cols
        compcols = opts.get_option("CompLLCols", {})
        # Coordinate shifts
        compy = opts.get_option("CompY", {})
        compz = opts.get_option("CompZ", {})
        # Loop through components
        for j, comp in enumerate(comps):
            # Get columns
            cols = compcols.get(comp)
            # Ensure list
            if (cols is not None) and not isinstance(cols, list):
                raise TypeError("LL cols for comp '%s' is not a list" % comp)
            # Get final component line load cols
            ccols = self._getcols_ll_comp(cols, comp, **opts)
            # Get the force line loads
            cols = [ccols["CA"], ccols["CY"], ccols["CN"]]
            # Generate basis loads
            basis[comp] = self.genr8_ll3x_comp_basis(cols, scol, **opts)
            # Save slice values
            if (j == 0) and (scol is not None):
                basis["s"] = basis[comp]["s"]
        # Output
        return basis

    # Create basis for line load of one component by slice
    def genr8_ll3x_comp_basis(self, cols, scol=None, **kw):
        r"""Calculate SVD-based basis for adjusting line loads

        This is a highly customized function (hence the somewhat
        obscure name) that adjusts three line load force cols that
        are a function of *x*.  It adjusts for five scenarios:

            * Adjust the *dCA* load such that integrated *CA* is
              increased by ``1.0``
            * Adjust *dCY* such that *CY* increases ``1.0`` and *CLN*
              is unchanged
            * Adjust *dCY* such that *CY* is unchanged and *CLN*
              increases ``1.0``
            * Adjust *dCN* such that *CN* increases ``1.0`` and *CLM*
              is unchanged
            * Adjust *dCN* such that *CN* is unchanged and *CLM*
              increases ``1.0``

        :Call:
            >>> basis = db.genr8_ll3x_comp_basis(cols, scol=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3
                * (*col1*, *col2*, *col3*)

                List/tuple of column names for *CA*, *CY*, and *CN*
                line loads
            *scol*: {``None``} | :class:`str`
                Name of slice col; calculate basis for each value in
                *db.bkpts[scol]*
            *nPOD*: {``10``} | ``None`` | :class:`int` > 0
                Number of POD/SVD modes to use during optimization
            *mask*: {``None``} | :class:`np.ndarray`
                Mask or indices of which cases to include in POD
                calculation
            *method*: {``"trapz"``} | ``"left"`` | **callable**
                Integration method used to integrate columns
        :Outputs:
            *basis*: :class:`dict`
                Basis adjustment loads for five scenarios
            *basis["s"]*: :class:`np.ndarray` | ``None``
                Unique values of *scol*
            *basis["dCA.CA"]*: :class:`np.ndarray`
                Delta *dCA* load to adjust *CA* by ``1.0``
            *basis["dCY.CY"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CY* by ``1.0``
            *basis["dCY.CLN"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CLN* by ``1.0``
            *basis["dCN.CN"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CN* by ``1.0``
            *basis["dCN.CLM"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CLM* by ``1.0``
        :Versions:
            * 2020-06-04 ``@ddalle``: Version 1.0
            * 2020-06-19 ``@ddalle``: Version 2.0
        """
       # --- Options ---
        # Check columns
        self._check_ll3x_cols(cols)
        # Convert options
        opts = _LL3XOpts(_section="basis", **kw)
        # Options for point genr8or
        kw_ = opts.section_options("basis_")
       # --- Special Case: No Slices ---
        # Slice
        if scol is None:
            # Simply calculate basis on one area
            basis = self._genr8_ll3x_basis(cols, **kw_)
            # Unpack cols (func call above ensures this will work)
            col1, col2, col3 = cols
            # Unpack basis loads and make 2D
            dCA_CA = np.ndarray([basis["dCA.CA"]])
            dCY_CY = np.ndarray([basis["dCY.CY"]])
            dCY_Cn = np.ndarray([basis["dCY.CLN"]])
            dCN_CN = np.ndarray([basis["dCN.CN"]])
            dCN_Cm = np.ndarray([basis["dCN.CLM"]])
            # Output
            return {
                "s": None,
                "dCA.CA": dCA_CA,
                "dCY.CY": dCY_CY,
                "dCY.CLN": dCY_Cn,
                "dCN.CN": dCN_CN,
                "dCN.CLM": dCN_Cm,
            }
       # --- Slices ---
        # Get mask for entire portion of run matrix to consider
        # (*mask* removed from *kw* because it's reused as kwarg below)
        mask = opts.get_option("mask")
        # Get indices
        mask_index = self.prep_mask(mask, col=scol)
        # Get slice values
        X_scol = self.get_values(scol, mask)
        # Cutoff for "equality" in slice col
        tol = 1e-5 * np.max(np.abs(X_scol))
        # Get slice values
        x_scol = self.genr8_bkpts(scol, tol=tol, nmin=5, mask=mask)
        # Number of slices
        nslice = x_scol.size
        # Mask of which slice values to keep
        mask_scol = np.ones(nslice, dtype="bool")
       # --- Basis Calculations ---
        # Loop through slices
        for j, xj in enumerate(x_scol):
            # Get subset in this slice
            maskx = np.where(np.abs(X_scol - xj) <= tol)[0]
            # Combine masks
            maskj = mask_index[maskx]
            # Calculate basis
            basisj = self._genr8_ll3x_basis(cols, mask=maskj, **kw_)
            # Initialize if first slice
            if j == 0:
                # Get number of cuts
                ncut = basisj["dCA.CA"].size
                # Get data type
                dtype = basisj["dCA.CA"].dtype.name
                # Initialize bases
                dCA_CA = np.zeros((ncut, nslice), dtype=dtype)
                dCY_CY = np.zeros((ncut, nslice), dtype=dtype)
                dCY_Cn = np.zeros((ncut, nslice), dtype=dtype)
                dCN_CN = np.zeros((ncut, nslice), dtype=dtype)
                dCN_Cm = np.zeros((ncut, nslice), dtype=dtype)
            # Save entries
            dCA_CA[:,j] = basisj["dCA.CA"]
            dCY_CY[:,j] = basisj["dCY.CY"]
            dCY_Cn[:,j] = basisj["dCY.CLN"]
            dCN_CN[:,j] = basisj["dCN.CN"]
            dCN_Cm[:,j] = basisj["dCN.CLM"]
       # --- Cleanup ---
        # Output
        return {
            "s": x_scol,
            "dCA.CA": dCA_CA,
            "dCY.CY": dCY_CY,
            "dCY.CLN": dCY_Cn,
            "dCN.CN": dCN_CN,
            "dCN.CLM": dCN_Cm,
        }

    # Create basis for line load of one component
    def _genr8_ll3x_basis(self, cols, nPOD=10, mask=None, **kw):
        r"""Calculate SVD-based basis for adjusting line loads

        This is a highly customized function (hence the somewhat
        obscure name) that adjusts three line load force cols that
        are a function of *x*.  It adjusts for five scenarios:

            * Adjust the *dCA* load such that integrated *CA* is
              increased by ``1.0``
            * Adjust *dCY* such that *CY* increases ``1.0`` and *CLN*
              is unchanged
            * Adjust *dCY* such that *CY* is unchanged and *CLN*
              increases ``1.0``
            * Adjust *dCN* such that *CN* increases ``1.0`` and *CLM*
              is unchanged
            * Adjust *dCN* such that *CN* is unchanged and *CLM*
              increases ``1.0``

        :Call:
            >>> basis = db._genr8_ll3x_basis(cols, nPOD=10, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3

                List/tuple of column names for *CA*, *CY*, and *CN*
                line loads
            *nPOD*: {``10``} | ``None`` | :class:`int` > 0
                Number of POD/SVD modes to use during optimization
            *mask*: {``None``} | :class:`np.ndarray`
                Mask or indices of which cases to include in POD
                calculation
            *method*: {``"trapz"``} | ``"left"`` | **callable**
                Integration method used to integrate columns
        :Outputs:
            *basis*: :class:`dict`
                Basis adjustment loads for five scenarios
            *basis["dCA.CA"]*: :class:`np.ndarray`
                Delta *dCA* load to adjust *CA* by ``1.0``
            *basis["dCY.CY"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CY* by ``1.0``
            *basis["dCY.CLN"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CLN* by ``1.0``
            *basis["dCN.CN"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CN* by ``1.0``
            *basis["dCN.CLM"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CLM* by ``1.0``
        :Versions:
            * 2020-06-04 ``@ddalle``: Version 1.0
        """
        # Check types
        if not isinstance(cols, (list, tuple)):
            # Wrong type
            raise TypeError("Adjusted col list must be list")
        elif len(cols) != 3:
            # Wrong size
            raise ValueError(
                "Adjusted col list has len=%i; must be 3" % len(cols))
        # Check membership of cols
        for j, col in enumerate(cols):
            # Check type
            if not typeutils.isstr(col):
                raise TypeError("Adjust col %i is not a 'str'" % j)
            elif col not in self:
                raise KeyError("Adjust col '%s' not in database" % col)
        # Other options
        method = kw.get("method", "trapz")
        # Get *xcol*
        xcol = kw.get("xcol")
        # Default *xcol*
        if (xcol is None):
            # Try to get the output *xarg*
            xargs = self.get_output_xargs(col)
            # Check if list
            if isinstance(xargs, (list, tuple)) and len(xargs) > 0:
                # Default
                xcol = xargs[0]
        # Confirm *xcol*
        if not typeutils.isstr(xcol):
            raise TypeError("Did not find a column for *x*-coords")
        # Get values
        x = self.get_all_values(xcol)
        # Check if present
        if x is None:
            # No coordinates found
            raise KeyError("No x-coords from col '%s'" % xcol)
        elif not isinstance(x, np.ndarray):
            # Bad type
            raise TypeError("X-coord col '%s' is not an array" % xcol)
        # Get dimension of *x*-coords
        ndimx = x.ndim
        # Confirm
        if ndimx == 0:
            # Cannot use scalar *x* here
            raise IndexError("Cannot use scalar *x* coords")
        elif ndimx == 2:
            # Apply *mask*
            x = self.get_values(xcol, mask)
        elif ndimx > 2:
            # Cannot use ND *x*
            raise IndexError("Cannot use %i-D *x* coords" % ndimx)
        # Unpack
        col1, col2, col3 = cols
        # Initialize output
        basis = {}
        # Get *CA* loads
        dCA = self.get_values(col1, mask)
        dCY = self.get_values(col2, mask)
        dCN = self.get_values(col3, mask)
        # Dimensions
        nx, ny = dCA.shape
        # Calculate SVD
        UCA, sCA, VCA = svd(dCA)
        UCY, sCY, VCY = svd(dCY)
        UCN, sCN, VCN = svd(dCN)
        # Downselect
        if nPOD is not None:
            # Select first *nPOD* mode shapes
            UCA = UCA[:, :nPOD]
            UCY = UCY[:, :nPOD]
            UCN = UCN[:, :nPOD]
            # Select the first *nPOD* singular values
            sCA = sCA[:nPOD]
            sCY = sCY[:nPOD]
            sCN = sCN[:nPOD]
        # Number of modes
        nmode = sCA.size
        # Calculate *L2* norm of each basis vector
        L2CA = np.sqrt(np.sum(UCA**2, axis=0))
        L2CY = np.sqrt(np.sum(UCY**2, axis=0))
        L2CN = np.sqrt(np.sum(UCN**2, axis=0))
        # Normalize basis functions
        psiCA = UCA / np.tile(L2CA, (nx, 1))
        psiCY = UCY / np.tile(L2CY, (nx, 1))
        psiCN = UCN / np.tile(L2CN, (nx, 1))
        # Max loads on any cut
        mxCA = np.max(np.abs(psiCA), axis=0)
        mxCY = np.max(np.abs(psiCY), axis=0)
        mxCN = np.max(np.abs(psiCN), axis=0)
        # Weights
        wCA = mxCA / sCA
        wCY = mxCY / sCY
        wCN = mxCN / sCN
        # Moments from normalized mode shapes
        psiCLM = self._genr8_ll_moment(psiCN, 2, 1, x=x)
        psiCLN = self._genr8_ll_moment(psiCY, 1, 2, x=x)
        # Integrate normalized mode shapes
        CAF = self._genr8_integral(psiCA, x, method)
        CYF = self._genr8_integral(psiCY, x, method)
        CNF = self._genr8_integral(psiCN, x, method)
        CLMF = self._genr8_integral(psiCLM, x, method)
        CLNF = self._genr8_integral(psiCLN, x, method)
        # Form matrix for linear system of constraints
        dCA = np.array([CAF])
        dCY = np.array([CYF, CLNF])
        dCN = np.array([CNF, CLMF])
        # First two equations: equality constraints on *CN* and *CLM*
        A1CA = np.hstack((dCA, np.zeros((1, 1))))
        A1CY = np.hstack((dCY, np.zeros((2, 2))))
        A1CN = np.hstack((dCN, np.zeros((2, 2))))
        # Last *n* equations: derivatives of the Lagrangian
        A2CA = np.hstack((np.diag(2*wCA), -dCA.T))
        A2CY = np.hstack((np.diag(2*wCY), -dCY.T))
        A2CN = np.hstack((np.diag(2*wCN), -dCN.T))
        # Assemble matrices
        ACA = np.vstack((A1CA, A2CA))
        ACY = np.vstack((A1CY, A2CY))
        ACN = np.vstack((A1CN, A2CN))
        # Right-hand sides of equations
        bCA = np.hstack(([1.0], np.zeros(nmode)))
        bCF = np.hstack(([1.0, 0.0], np.zeros(nmode)))
        bCM = np.hstack(([0.0, 1.0], np.zeros(nmode)))
        # Solve linear systems
        xCA = np.linalg.solve(ACA, bCA)
        xCY = np.linalg.solve(ACY, bCF)
        xCN = np.linalg.solve(ACN, bCF)
        xCm = np.linalg.solve(ACN, bCM)
        xCn = np.linalg.solve(ACY, bCM)
        # Calculate linear combination of SVD modes
        phiCA = np.dot(UCA, xCA[:nmode])
        phiCY = np.dot(UCY, xCY[:nmode])
        phiCN = np.dot(UCN, xCN[:nmode])
        phiCm = np.dot(UCN, xCm[:nmode])
        phiCn = np.dot(UCY, xCn[:nmode])
        # Basis
        return {
            "dCA.CA": phiCA,
            "dCY.CY": phiCY,
            "dCN.CN": phiCN,
            "dCN.CLM": phiCm,
            "dCY.CLN": phiCn,
        }

   # --- Checkers and Col Names ---
    # Check LL3X column list
    def _check_ll3x_cols(self, cols):
        r"""Check a list of three line load column names

        :Call:
            >>> db._check_ll3x_cols(cols)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
        :Versions:
            * 2020-06-12 ``@ddalle``: First version
        """
        # Check the columns
        if cols is None:
            # Ok; rely on defaults
            return
        elif not isinstance(cols, (list, tuple)):
            # Wrong type
            raise TypeError("LL3X cols must be list (got '%s')" % type(cols))
        elif len(cols) not in {3, 6}:
            # Wrong length
            raise IndexError(
                "LL3X cols must have length 3 or 6 (got %i)" % len(cols))
        # Check column presence
        for j, col in enumerate(cols):
            # Check type
            if not typeutils.isstr(col):
                # Wrong type
                raise TypeError("LL3X col %i is not a 'str'" % j)
            elif col not in self:
                # Not present
                raise KeyError("LL3X col %i '%s' not in database" % (j, col))
            # Get data type and dimension
            ndim = self.get_ndim(col)
            dtype = self.get_col_dtype(col)
            # Ensure type and dimension
            if not (dtype.startswith("float") or dtype.startswith("comp")):
                # Not data type
                raise TypeError(
                    ("Data type for col '%s' is '%s'; " % (col, dtype)) +
                    ("must be float or complex"))
            elif ndim != 2:
                # Line load must be 2D
                raise IndexError("Col '%s' is %iD, must be 2D" % (col, ndim))

    # Check component list
    def _check_ll3x_comps(self, comps):
        r"""Check types of a list of components

        :Call:
            >>> comps = db._check_ll3x_comps(comp)
            >>> comps = db._check_ll3x_comps(comps)
        :Inputs:
            *comp*: :class:`str`
                Single component name
            *comps*: :class:`list`\ [:class:`str`]
                List of component names
        :Versions:
            *comps*: :class:`list`\ [:class:`str`]
                List of component names
        :Outputs:
            * 2020-06-15 ``@ddalle``: First version
        """
        # Ensure list of components
        if typeutils.isstr(comps):
            # Single component; convert to list
            comps = [comps]
        elif not isinstance(comps, list):
            # Wrong type
            raise TypeError(
                "LL3X comps must be 'list' (got '%s')" % type(comps))
        # Check strings
        for j, comp in enumerate(comps):
            # Ensure string
            if not typeutils.isstr(comp):
                raise TypeError("LL3X comp %i is not a string" % j)
        # Return it in case it's converted
        return comps
        
    # Output column names for adjusted ll cols
    def _getcols_lla_comp(self, cols, comp=None, **kw):
        r"""Create :class:`dict` of line load cols

        :Call:
            >>> llacols = db._getcols_lla_comp(cols, comp, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *LLAdjustedCols*: :class:`dict`\ [:class:`str`]
                Columns to use for adjusted line load
            *CompLLAdjustedCols*: :class:`dict`\ [:class:`dict`]
                *LLAdjustedCols* for one or more components
        :Outputs:
            *llacols*: :class:`dict`\ [:class:`str`]
                Columns to use for six adjusted line loads
        :Versions:
            * 2020-06-15 ``@ddalle``: First version
        """
        # Line load column names
        llcols = self._getcols_ll_comp(cols, comp=comp, **kw)
        # Get column names
        colCA = "%s.adjusted" % llcols["CA"]
        colCY = "%s.adjusted" % llcols["CY"]
        colCN = "%s.adjusted" % llcols["CN"]
        colCl = "%s.adjusted" % llcols["CLL"]
        colCm = "%s.adjusted" % llcols["CLM"]
        colCn = "%s.adjusted" % llcols["CLN"]
        # Check for *comp*
        if comp is not None:
            # Get option
            acols = kw.get("CompLLAdjustedCols", {}).get(comp, {})
            # Check for overrides
            colCA = acols.get("CA", colCA)
            colCY = acols.get("CY", colCY)
            colCN = acols.get("CN", colCN)
            colCl = acols.get("CLL", colCl)
            colCm = acols.get("CLM", colCm)
            colCn = acols.get("CLN", colCn)
        # Check for manual names for this *comp*
        acols = kw.get("LLAdjustedCols", {})
        # Check for overrides
        colCA = acols.get("CA", colCA)
        colCY = acols.get("CY", colCY)
        colCN = acols.get("CN", colCN)
        colCl = acols.get("CLL", colCl)
        colCm = acols.get("CLM", colCm)
        colCn = acols.get("CLN", colCn)
        # Output
        return {
            "CA": colCA,
            "CY": colCY,
            "CN": colCN,
            "CLL": colCl,
            "CLM": colCm,
            "CLN": colCn,
        }

    # Output column names for ll adjustment basis cols
    def _getcols_llb_comp(self, cols, comp=None, **kw):
        r"""Create :class:`dict` of line load adjustment basis names

        :Call:
            >>> llbcols = db._getcols_llb_comp(cols, comp, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *LLBasisCols*: :class:`dict`\ [:class:`str`]
                Columns to use for adjusted line load
            *CompLLBasisCols*: :class:`dict`\ [:class:`dict`]
                *LLBasisCols* for one or more components
        :Outputs:
            *llbcols*: :class:`dict`\ [:class:`str`]
                Columns to use for LL adjustment basis
        :Versions:
            * 2020-06-19 ``@ddalle``: First version
        """
        # Line load column names
        llcols = self._getcols_ll_comp(cols, comp=comp, **kw)
        # Get column names
        colCAA = "%s.deltaCA" % llcols["CA"]
        colCYY = "%s.deltaCY" % llcols["CY"]
        colCNN = "%s.deltaCN" % llcols["CN"]
        colCNm = "%s.deltaCLM" % llcols["CN"]
        colCYn = "%s.deltaCLN" % llcols["CY"]
        # Check for *comp*
        if comp is not None:
            # Get option
            bcols = kw.get("CompLLBasisCols", {}).get(comp, {})
            # Check for overrides
            colCAA = bcols.get("dCA.CA", colCAA)
            colCYY = bcols.get("dCY.CY", colCYY)
            colCNN = bcols.get("dCN.CN", colCNN)
            colCNm = bcols.get("dCN.CLM", colCNm)
            colCYn = bcols.get("dCY.CLN", colCYn)
        # Check for manual names for this *comp*
        bcols = kw.get("LLBasisCols", {})
        # Check for overrides
        colCAA = bcols.get("dCA.CA", colCAA)
        colCYY = bcols.get("dCY.CY", colCYY)
        colCNN = bcols.get("dCN.CN", colCNN)
        colCNm = bcols.get("dCN.CLM", colCNm)
        colCYn = bcols.get("dCY.CLN", colCYn)
        # Output
        return {
            "dCA.CA": colCAA,
            "dCY.CY": colCYY,
            "dCN.CN": colCNN,
            "dCN.CLM": colCNm,
            "dCY.CLN": colCYn,
        }

    # Output column names for ll cols
    def _getcols_ll_comp(self, cols, comp=None, **kw):
        r"""Create :class:`dict` of line load cols

        :Call:
            >>> llcols = db._getcols_ll_comp(cols, comp, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *LLCols*: :class:`dict`\ [:class:`str`]
                Columns to use for each integral coefficient
            *CompLLCols*: :class:`dict`\ [:class:`dict`]
                *LLCols* for one or more components
        :Outputs:
            *llcols*: :class:`dict`\ [:class:`str`]
                Columns to use for six line loads
        :Versions:
            * 2020-06-15 ``@ddalle``: First version
        """
        # Get moment line load cols
        if cols is None:
            # Defaults
            if comp is None:
                # Just use base names
                colCA = "dCA"
                colCY = "dCY"
                colCN = "dCN"
                colCl = "dCLL"
                colCm = "dCLM"
                colCn = "dCLN"
            else:
                # Combine component name
                colCA = "%s.dCA" % comp
                colCY = "%s.dCY" % comp
                colCN = "%s.dCN" % comp
                colCl = "%s.dCLL" % comp
                colCm = "%s.dCLM" % comp
                colCn = "%s.dCLN" % comp
        else:
            # Get column names for forces
            colCA = cols[0]
            colCY = cols[1]
            colCN = cols[2]
            if len(cols) == 3:
                # Default conversions from force col names
                colCl = self._getcol_CLL_from_CN(colCN)
                colCm = self._getcol_CLM_from_CN(colCN)
                colCn = self._getcol_CLN_from_CY(colCY)
            else:
                # Directly specified
                colCl = cols[3]
                colCm = cols[4]
                colCn = cols[5]
        # Check for *comp*
        if comp is not None:
            # Get option
            llcols = kw.get("CompLLCols", {}).get(comp, {})
            # Check for overrides
            colCA = llcols.get("CA", colCA)
            colCY = llcols.get("CY", colCY)
            colCN = llcols.get("CN", colCN)
            colCl = llcols.get("CLL", colCl)
            colCm = llcols.get("CLM", colCm)
            colCn = llcols.get("CLN", colCn)
        # Check for manual names for this *comp*
        llcols = kw.get("LLCols", {})
        # Check for overrides
        colCA = llcols.get("CA", colCA)
        colCY = llcols.get("CY", colCY)
        colCN = llcols.get("CN", colCN)
        colCl = llcols.get("CLL", colCl)
        colCm = llcols.get("CLM", colCm)
        colCn = llcols.get("CLN", colCn)
        # Output
        return {
            "CA": colCA,
            "CY": colCY,
            "CN": colCN,
            "CLL": colCl,
            "CLM": colCm,
            "CLN": colCn,
        }

    # Output column names for ll cols
    def _getcols_ll3_comp(self, cols=None, comp=None, **kw):
        r"""Create :class:`dict` of line load cols

        :Call:
            >>> llcols = db._getcols_ll3_comp(cols, comp, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3

                List/tuple of column names for *CA*, *CY*, and *CN*
            *LLCols*: :class:`dict`\ [:class:`str`]
                Columns to use for each integral coefficient
            *CompLLCols*: :class:`dict`\ [:class:`dict`]
                *LLCols* for one or more components
        :Outputs:
            *llcols*: :class:`list`\ [:class:`str`]
                Columns to use for *CA*, *CY*, and *CN*
        :Versions:
            * 2020-06-25 ``@ddalle``: First version
        """
        # Get moment line load cols
        if cols is None:
            # Defaults
            if comp is None:
                # Just use base names
                colCA = "dCA"
                colCY = "dCY"
                colCN = "dCN"
            else:
                # Combine component name
                colCA = "%s.dCA" % comp
                colCY = "%s.dCY" % comp
                colCN = "%s.dCN" % comp
        else:
            # Get column names for forces
            colCA = cols[0]
            colCY = cols[1]
            colCN = cols[2]
        # Check for *comp*
        if comp is not None:
            # Get option
            llcols = kw.get("CompLLCols", {}).get(comp, {})
            # Check for overrides
            colCA = llcols.get("CA", colCA)
            colCY = llcols.get("CY", colCY)
            colCN = llcols.get("CN", colCN)
        # Check for manual names for this *comp*
        llcols = kw.get("LLCols", {})
        # Check for overrides
        colCA = llcols.get("CA", colCA)
        colCY = llcols.get("CY", colCY)
        colCN = llcols.get("CN", colCN)
        # Output
        return colCA, colCY, colCN

    # Output column names for integrals from ll cols
    def _getcols_ll3x_fmcomp(self, cols, comp=None, **kw):
        r"""Create :class:`dict` of FM cols based on line load *cols*

        :Call:
            >>> fmcols = db._getcols_ll3x_fmcomp(cols, comp=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *comp*: :class:`str`
                Name of component
            *FMCols*: :class:`dict`\ [:class:`str`]
                Columns to use for each integral coefficient
            *CompFMCols*: :class:`dict`\ [:class:`dict`]
                *FMCols* for one or more *comps*
        :Outputs:
            *fmcols*: :class:`dict`\ [:class:`str`]
                Columns to use for six integrated forces and moments
        :Versions:
            * 2020-06-15 ``@ddalle``: First version
        """
        # Line load column names
        llcols = self._getcols_ll_comp(cols, comp=comp, **kw)
        # Get column names
        colCA = self._getcol_CX_from_dCX(llcols["CA"])
        colCY = self._getcol_CX_from_dCX(llcols["CY"])
        colCN = self._getcol_CX_from_dCX(llcols["CN"])
        colCl = self._getcol_CX_from_dCX(llcols["CLL"])
        colCm = self._getcol_CX_from_dCX(llcols["CLM"])
        colCn = self._getcol_CX_from_dCX(llcols["CLN"])
        # Check for *comp*
        if comp is not None:
            # Get option
            fmcols = kw.get("CompFMCols", {}).get(comp, {})
            # Check for overrides
            colCA = fmcols.get("CA", colCA)
            colCY = fmcols.get("CY", colCY)
            colCN = fmcols.get("CN", colCN)
            colCl = fmcols.get("CLL", colCl)
            colCm = fmcols.get("CLM", colCm)
            colCn = fmcols.get("CLN", colCn)
        # Check for manual names for this *comp*
        fmcols = kw.get("FMCols", {})
        # Check for overrides
        colCA = fmcols.get("CA", colCA)
        colCY = fmcols.get("CY", colCY)
        colCN = fmcols.get("CN", colCN)
        colCl = fmcols.get("CLL", colCl)
        colCm = fmcols.get("CLM", colCm)
        colCn = fmcols.get("CLN", colCn)
        # Output
        return {
            "CA": colCA,
            "CY": colCY,
            "CN": colCN,
            "CLL": colCl,
            "CLM": colCm,
            "CLN": colCn,
        }

    # Output column names for integrals from ll cols
    def _getcol_ll_xcol(self, col, comp=None, **kw):
        r"""Create :class:`dict` of FM cols based on line load *cols*

        :Call:
            >>> xcol = db._getcol_ll_xcol(col, comp **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *comp*: {``None``} | :class:`str`
                Name of component
            *FMCols*: :class:`dict`\ [:class:`str`]
                Columns to use for each integral coefficient
            *CompFMCols*: :class:`dict`\ [:class:`dict`]
                *FMCols* for one or more *comps*
        :Outputs:
            *xcol*: :class:`str`
                Name of column to use for *x*-coordinates of *col*
        :Versions:
            * 2020-06-19 ``@ddalle``: First version
        """
        # Default
        if comp is None:
            # Just use "x"
            xcol = "x"
        else:
            # Combine component name
            xcol = "%s.x" % comp
        # Check for *col*
        if col is None:
            return xcol
        # Get *xargs* from line load response definition
        xargs = self.get_output_xargs(col)
        # If it's a list with one entry, use that
        if isinstance(xargs, list) and len(xargs) == 1:
            # This is the usual case
            xcol = xargs[0]
        # Get explicit option
        xcol = kw.get("xcol", xcol)
        # Check for component-wise options
        xcols = kw.get("CompXCol", {})
        # Check for this *comp*
        xcol = xcols.get(comp, xcol)
        # Output
        return xcol

    # Output column names for integral fractions
    def _getcols_fmfrac_comp(self, cols, comp=None, **kw):
        r"""Create :class:`dict` of FM adjustment fractions

        :Call:
            >>> fcols = db._getcols_fmfrac_comp(cols, comp=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *comp*: :class:`str`
                Name of component
            *CompFMFracCols*: :class:`dict`\ [:class:`dict`]
                Output column names for adjustment fractions
        :Outputs:
            *fcols*: :class:`dict`\ [:class:`str`]
                Columns to use for six integrated forces and moments
        :Versions:
            * 2020-06-16 ``@ddalle``: First version
        """
        # Line load column names
        fmcols = self._getcols_ll3x_fmcomp(cols, comp=comp, **kw)
        # Get column names
        colCA = "%s.fraction" % fmcols["CA"]
        colCY = "%s.fraction" % fmcols["CY"]
        colCN = "%s.fraction" % fmcols["CN"]
        colCl = "%s.fraction" % fmcols["CLL"]
        colCm = "%s.fraction" % fmcols["CLM"]
        colCn = "%s.fraction" % fmcols["CLN"]
        # Check for *comp*
        if comp is not None:
            # Get option
            fcols = kw.get("CompFMFracCols", {}).get(comp, {})
            # Check for overrides
            colCA = fcols.get("CA", colCA)
            colCY = fcols.get("CY", colCY)
            colCN = fcols.get("CN", colCN)
            colCl = fcols.get("CLL", colCl)
            colCm = fcols.get("CLM", colCm)
            colCn = fcols.get("CLN", colCn)
        # Check for manual names for this *comp*
        fmcols = kw.get("FMFracCols", {})
        # Check for overrides
        colCA = fmcols.get("CA", colCA)
        colCY = fmcols.get("CY", colCY)
        colCN = fmcols.get("CN", colCN)
        colCl = fmcols.get("CLL", colCl)
        colCm = fmcols.get("CLM", colCm)
        colCn = fmcols.get("CLN", colCn)
        # Output
        return {
            "CA": colCA,
            "CY": colCY,
            "CN": colCN,
            "CLL": colCl,
            "CLM": colCm,
            "CLN": colCn,
        }

    # Output column names for integral fractions
    def _getcols_fmweight_comp(self, cols, comp=None, **kw):
        r"""Create :class:`dict` of FM adjustment weights

        :Call:
            >>> wcols = db._getcols_fmweight_comp(cols, comp=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DataKit`
                Database with analysis tools
            *cols*: :class:`list`\ [:class:`str`]
                * *len*: 3 | 6

                List/tuple of column names for *CA*, *CY*, and *CN*
                [, *CLL*, *CLM*, *CLN*] line loads
            *comp*: :class:`str`
                Name of component
            *CompFMWeightCols*: :class:`dict`\ [:class:`dict`]
                Output column names for adjustment weights
        :Outputs:
            *wcols*: :class:`dict`\ [:class:`str`]
                Columns to use for six integrated forces and moments
        :Versions:
            * 2020-06-16 ``@ddalle``: First version
        """
        # Get column names
        colCA = "%s.wCA.CA"   % comp
        colCY = "%s.wCY.CY"   % comp
        colCN = "%s.wCN.CN"   % comp
        colC1 = "%s.wCY.CLL"  % comp
        colC2 = "%s.wCN.CLL"  % comp
        colCm = "%s.wCLM.CLM" % comp
        colCn = "%s.wCLN.CLN" % comp
        # Check for *comp*
        if comp is not None:
            # Get option
            fcols = kw.get("CompFMWeightCols", {}).get(comp, {})
            # Check for overrides
            colCA = fcols.get("wCA.CA", colCA)
            colCY = fcols.get("wCY.CY", colCY)
            colCN = fcols.get("wCN.CN", colCN)
            colC1 = fcols.get("wCY.CLL", colC1)
            colC2 = fcols.get("wCN.CLL", colC2)
            colCm = fcols.get("wCLM.CLM", colCm)
            colCn = fcols.get("wCLN.CLN", colCn)
        # Output
        return {
            "wCA.CA": colCA,
            "wCY.CY": colCY,
            "wCN.CN": colCN,
            "wCY.CLL": colC1,
            "wCN.CLL": colC2,
            "wCLM.CLM": colCm,
            "wCLN.CLN": colCn,
        }
        
  # >


# Combine options
kwutils._combine_val(DBLL._tagmap, dbfm.DBFM._tagmap)

# Invert the _tagmap
DBLL.create_tagcols()
