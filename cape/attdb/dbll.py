#!/us/bin/env python
# -*- coding: utf-8 -*-
r"""
-----------------------------------------------------------------
:mod:`cape.attdb.dbll`: Aero Task Team Line Load Databases
-----------------------------------------------------------------

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


# Options class for adjustments
class _LL3XOpts(kwutils.KwargHandler):
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Lists ---
    # All options
    _optlist = {
        "CompFMCols",
        "FMCols",
        "IntegralCols",
        "MomentCols",
        "TargetCols",
        "fm",
        "mask",
        "method",
        "nPOD",
        "ocols",
        "ocolCA",
        "ocolCLL",
        "ocolCLM",
        "ocolCLN",
        "ocolCN",
        "ocolCY",
        "x",
        "xMRP",
        "xcol",
        "y",
        "yMRP",
        "z",
        "zMRP"
    }

    # Alternate names
    _optmap = {
        "Method": "method",
        "NPOD": "nPOD",
        "XMRP": "xMRP",
        "YMRP": "yMRP",
        "ZMRP": "zMRP",
        "fmcols": "CompFMCols",
        "icols": "IntegralCols",
        "mcols": "MomentCols",
        "npod": "nPOD",
        "xmrp": "xMRP",
        "ymrp": "yMRP",
        "zmrp": "zMRP",
    }

    # Sections
    _optlists = {
        "integrals": {
            "Lref",
            "method",
            "xMRP",
            "yMRP",
            "y",
            "zMRP",
            "z"
        },
        "make_integrals": {
            "Lref",
            "method",
            "ocols",
            "ocolCA",
            "ocolCLL",
            "ocolCLM",
            "ocolCLN",
            "ocolCN",
            "ocolCY",
            "xMRP",
            "yMRP",
            "y",
            "zMRP",
            "z"
        }
    }

   # --- Type ---
    # Required types
    _optytpes = {
        "CompFMCols": dict,
        "IntegralCols": (list, tuple),
        "MomentCols": (list, tuple),
        "TargetCols": (list, dict),
        "mask": np.ndarray,
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
        return self.create_dclm(col, ocol=ocol, **kw)

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
        return self.create_dcln(col, ocol=ocol, **kw)
        
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
        # Save them
        self.save_col(ocol, v)
        # Create definition
        self.make_defn(ocol, v)
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
        # Save them
        self.save_col(ocol, v)
        # Create definition
        self.make_defn(ocol, v)
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
        # Save them
        self.save_col(ocol, v)
        # Create definition
        self.make_defn(ocol, v)
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
        # Save it
        self.save_col(col, v)
        # Update definition if necessary
        self.make_defn(col, v)
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
        # Save it
        self.save_col(col, v)
        # Update definition if necessary
        self.make_defn(col, v)
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
    def genr8_ll3x_adjust(self, comps, fm, scol=None, **kw):
       # --- Checks ---
        # Check and process component list
        comps = self._check_ll3x_comps(comps)
        # Form options
        opts = _LL3XOpts(**kw)
       # --- Targets ---
        # Basic coefficient names
        coeffs = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
        # Get target columns for *fm*
        fmcols = opts.get_option("TargetCols")
        # Check possible types
        if fmcols is None:
            # Default
            fmcols = {col: col for col in coeffs}
        elif isinstance(fmcols, list):
            # Must be six cols
            if len(fmcols) != 6:
                raise IndexError("'TargetCols' list must have length 6")
            # Unpack
            fmcols = {col: fmcols[j] for j, col in enumerate(coeffs)}
       # --- Division of Deltas ---
        # Get integral columns
        compcols = opts.get_option("CompFMCols", {})
        # Loop through columns
       # --- Adjustment Basis ---
        # Create basis
        basis = self.create_ll3x_basis(cols, scol, **kw)

   # --- Adjustment Fraction ---
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
        :Versions:
            * 2020-06-12 ``@ddalle``: First version
        """
       # --- Checks ---
        # Check and process component list
        comps = self._check_ll3x_comps(comps)
        # Form options
        opts = _LL3XOpts(**kw)
       # --- FM Cols ---
        # Get columns that determine deltas for each component
        fmcols = opts.get_option("CompFMCols", {})
        # Basic coefficient list
        coeffs = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
        # Loop through components
        for comp in comps:
            # Get current list
            cols = fmcols.get(comp, {})
            # Check for list
            if isinstance(cols, list):
                # Init dict
                dcols = {}
                # Loop through cols
                for j, col in enumerate(cols):
                    # Set it
                    dcols[coeffs[j]] = col
                # Handoff
                cols = dcols
            # Check cols
            if not isinstance(cols, dict):
                # Wrong type
                raise TypeError(
                    "Bad type for comp '%s' FM cols; must be 'dict', got '%s'"
                    % (comp, type(cols)))
            # Defaults
            for coeff in coeffs:
                # Check if present
                if coeff in cols:
                    continue
                # Prepend component name to coefficients
                cols = {coeff: "%s.%s" % (comp, coeff) for coeff in coeffs}
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
        else:
            # Get unique values
            s = np.unique(self.get_values(scol, mask))
            # Divide those values into slices
            masks, _ = self.find([scol], s, mapped=True)
       # --- Fractions ---
        # Initialize fractions
        f = {comp: {} for comp in comps}
        # Loop through coefficients
        for coeff in coeffs:
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
                fcc = np.zeros_like(self[col])
                # Slices
                for maskj in masks:
                    # Get mean
                    fj = np.mean(fcomp[maskj])
                    # Save it
                    fcc[maskj] = fj
                # Save fraction
                f[comp][coeff] = fcc
       # --- Output ---
        # Output
        return f

    # Check component list
    def _check_ll3x_comps(self, comps):
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

   # --- Integration ---
    # Generate integrals and save
    def make_ll3x_integrals(self, cols, **kw):
        r"""Integrate 3 or 6 line load columns

        :Call:
            >>> fm = db.make_ll3x_integrals(cols, **kw)
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
        opts = _LL3XOpts(_section="make_integrals", **kw)
        # Check the columns
        self._check_ll3x_cols(cols)
        # Options to integrator
        kw_int = opts.section_options("integrals")
        # Get column names
        ocols = self._getocols_integrals(cols, **opts)
        # Unpack
        colCA, colCY, colCN, colCl, colCm, colCn = ocols
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
        fm = self.genr8_ll3x_integrals(cols, **kw_int)
        # Save the integrated columns
        self.save_col(colCA, fm["CA"])
        self.save_col(colCY, fm["CY"])
        self.save_col(colCN, fm["CN"])
        self.save_col(colCl, fm["CLL"])
        self.save_col(colCm, fm["CLM"])
        self.save_col(colCn, fm["CLN"])
        # Make definitions
        self.make_defn(colCA, fm["CA"])
        self.make_defn(colCY, fm["CY"])
        self.make_defn(colCN, fm["CN"])
        self.make_defn(colCl, fm["CLL"])
        self.make_defn(colCm, fm["CLM"])
        self.make_defn(colCn, fm["CLN"])
        # Output
        return fm

    # Generate integrals and save
    def create_ll3x_integrals(self, cols, **kw):
        r"""Integrate 3 or 6 line load columns

        :Call:
            >>> fm = db.create_ll3x_integrals(cols, **kw)
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
        opts = _LL3XOpts(_section="make_integrals", **kw)
        # Check the columns
        self._check_ll3x_cols(cols)
        # Options to integrator
        kw_int = opts.section_options("integrals")
        # Get column names
        ocols = self._getocols_integrals(cols, **opts)
        # Unpack
        colCA, colCY, colCN, colCl, colCm, colCn = ocols
        # Generate the integrals
        fm = self.genr8_ll3x_integrals(cols, **kw_int)
        # Save the integrated columns
        self.save_col(colCA, fm["CA"])
        self.save_col(colCY, fm["CY"])
        self.save_col(colCN, fm["CN"])
        self.save_col(colCl, fm["CLL"])
        self.save_col(colCm, fm["CLM"])
        self.save_col(colCn, fm["CLN"])
        # Make definitions
        self.make_defn(colCA, fm["CA"])
        self.make_defn(colCY, fm["CY"])
        self.make_defn(colCN, fm["CN"])
        self.make_defn(colCl, fm["CLL"])
        self.make_defn(colCm, fm["CLM"])
        self.make_defn(colCn, fm["CLN"])
        # Output
        return fm
        
    # Generate integrals
    def genr8_ll3x_integrals(self, cols, **kw):
        r"""Integrate 3 or 6 line load columns

        :Call:
            >>> fm = db.genr8_ll3x_integrals(cols, **kw)
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
            * 2020-06-11 ``@ddalle``: First version
        """
        # Check options
        opts = _LL3XOpts(**kw)
        # Check column list
        self._check_ll3x_cols(cols)
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
        CA = self.genr8_integral(cols[0], **opts)
        CY = self.genr8_integral(cols[1], **opts)
        CN = self.genr8_integral(cols[2], **opts)
        # Check for moment loads
        if len(cols) == 6:
            # Integrate moment loads
            CLL = self.genr8_integral(cols[3], **opts)
            CLM = self.genr8_integral(cols[4], **opts)
            CLN = self.genr8_integral(cols[5], **opts)
        else:
            # Create moment columns
            self.create_dclm(cols[2], **opts)
            self.create_dcln(cols[1], **opts)
            # Column names used for those integrals
            colCLM = self._getcol_CLM_from_CN(cols[2])
            colCLN = self._getcol_CLN_from_CY(cols[1])
            # Integrate moments
            CLL = np.zeros_like(CA)
            CLM = self.genr8_integral(colCLM, **opts)
            CLN = self.genr8_integral(colCLN, **opts)
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
        
    # Output column names
    def _getocols_integrals(self, cols, **kw):
        # Get column names
        colCA = self._getcol_CX_from_dCX(cols[0])
        colCY = self._getcol_CX_from_dCX(cols[1])
        colCN = self._getcol_CX_from_dCX(cols[2])
        # Get moment line load cols
        if len(cols) == 3:
            # Default conversions from force col names
            coldCLL = self._getcol_CLL_from_CN(cols[2])
            coldCLM = self._getcol_CLM_from_CN(cols[2])
            coldCLN = self._getcol_CLN_from_CY(cols[1])
        else:
            # Directly specified
            coldCLL = cols[3]
            coldCLM = cols[4]
            coldCLN = cols[5]
        # Get output column names
        colCl = self._getcol_CX_from_dCX(coldCLL)
        colCm = self._getcol_CX_from_dCX(coldCLM)
        colCn = self._getcol_CX_from_dCX(coldCLN)
        # Check for *ocols*
        ocols = kw.get("ocols", {})
        # Check for overrides
        colCA = kw.get("ocolCA", ocols.get("CA", colCA))
        colCY = kw.get("ocolCY", ocols.get("CY", colCY))
        colCN = kw.get("ocolCN", ocols.get("CN", colCN))
        colCl = kw.get("ocolCLL", ocols.get("CLL", colCl))
        colCm = kw.get("ocolCLM", ocols.get("CLM", colCm))
        colCn = kw.get("ocolCLN", ocols.get("CLN", colCn))
        # Output
        return colCA, colCY, colCN, colCl, colCm, colCn

   # --- Checkers ---
    # Check LL3X column list
    def _check_ll3x_cols(self, cols):
        # Check the columns
        if not isinstance(cols, (list, tuple)):
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
        

   # --- Basis ---
    # Create basis for line load of one component by slice
    def create_ll3x_basis(self, cols, scol=None, **kw):
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
            >>> basis = db.create_ll3x_basis(cols, scol=None, **kw)
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
            *ocol*: {``None``} | :class:`str`
                Name of col to save values of *scol*
            *ocols*: {``None``} | :class:`list`\ [:class:`str`]
                Manual names of basis *cols*, defaults are
                ``col1 + ".deltaCA"``, ``col2 + ".deltaCY"``,
                ``col2 + ".deltaCLN"``, ``col3 + ".deltaCN"``, and
                ``col3 + ".deltaCLM"``
        :Outputs:
            *basis*: :class:`dict`
                Basis adjustment loads for five scenarios
            *basis["dCA_CA"]*: :class:`np.ndarray`
                Delta *dCA* load to adjust *CA* by ``1.0``
            *basis["dCY_CY"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CY* by ``1.0``
            *basis["dCY_CLN"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CLN* by ``1.0``
            *basis["dCN_CN"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CN* by ``1.0``
            *basis["dCN_CLM"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CLM* by ``1.0``
        :Versions:
            * 2020-06-04 ``@ddalle``: Version 1.0
        """
       # --- Special Case: No Slices ---
        # Slice
        if scol is None:
            # Simply calculate basis on one area
            basis = self.genr8_ll3x_basis(cols, nPOD=nPOD, **kw)
            # Unpack cols (func call above ensures this will work)
            col1, col2, col3 = cols
            # Unpack basis loads and make 2D
            dCA_CA = np.ndarray([basis["dCA_CA"]])
            dCY_CY = np.ndarray([basis["dCY_CY"]])
            dCY_Cn = np.ndarray([basis["dCY_CLN"]])
            dCN_CN = np.ndarray([basis["dCN_CN"]])
            dCN_Cm = np.ndarray([basis["dCN_CLM"]])
            # Output column names
            colAA = col1 + ".deltaCA"
            colYY = col2 + ".deltaCY"
            colYn = col2 + ".deltaCLN"
            colNN = col3 + ".deltaCN"
            colNm = col3 + ".deltaCLM"
            # Save columns
            self.save_col(colAA, dCA_CA)
            self.save_col(colYY, dCY_CY)
            self.save_col(colYn, dCY_Cn)
            self.save_col(colNN, dCN_CN)
            self.save_col(colNm, dCN_Cm)
            # Save definitions
            self.make_defn(colAA, dCA_CA)
            self.make_defn(colYY, dCY_CY)
            self.make_defn(colYn, dCY_Cn)
            self.make_defn(colNN, dCN_CN)
            self.make_defn(colNm, dCN_Cm)
            # Output
            return {
                "dCA_CA": dCA_CA,
                "dCY_CY": dCY_CY,
                "dCY_CLN": dCY_Cn,
                "dCN_CN": dCN_CN,
                "dCN_CLM": dCN_Cm,
            }
       # --- Slices ---
        # Get mask for entire portion of run matrix to consider
        # (*mask* removed from *kw* because it's reused as kwarg below)
        mask = kw.pop("mask", None)
        # Get indices
        mask_index = self.prep_mask(mask, col=scol)
        # Get slice values
        X_scol = self.get_values(scol, mask)
        # Get candidate slice coordinates
        x_scol = self.bkpts.get(scol)
        # Check if empty
        if x_scol is None:
            # Get all values
            x_scol = np.unique(X_scol)
        # Mask of which slice values to keep
        mask_scol = np.ones(x_scol.size, dtype="bool")
        # Cutoff for "equality" in slice col
        tol = 1e-5 * np.max(np.abs(x_scol))
        # Make sure all candidate values are in the candidate space
        for j, xj in enumerate(x_scol):
            # Count how many entries are in *X_scol*
            nj = np.count_nonzero(np.abs(xj - X_scol) <= tol)
            # Check if there are at least 5 entries
            if nj < 5:
                # Don't consider this slice; calculations will fail
                mask_scol[j] = False
        # Apply mask to candidate slice coordinates
        x_scol = x_scol[mask_scol]
        # Number of slices
        nslice = x_scol.size
       # --- Basis Calculations ---
        # Loop through slices
        for j, xj in enumerate(x_scol):
            # Get subset in this slice
            maskx = np.where(np.abs(X_scol - xj) <= tol)[0]
            # Combine masks
            maskj = mask_index[maskx]
            # Calculate basis
            basisj = self.genr8_ll3x_basis(cols, mask=maskj, **kw)
            # Initialize if first slice
            if j == 0:
                # Get number of cuts
                ncut = basisj["dCA_CA"].size
                # Get data type
                dtype = basisj["dCA_CA"].dtype.name
                # Initialize bases
                dCA_CA = np.zeros((ncut, nslice), dtype=dtype)
                dCY_CY = np.zeros((ncut, nslice), dtype=dtype)
                dCY_Cn = np.zeros((ncut, nslice), dtype=dtype)
                dCN_CN = np.zeros((ncut, nslice), dtype=dtype)
                dCN_Cm = np.zeros((ncut, nslice), dtype=dtype)
            # Save entries
            dCA_CA[:,j] = basisj["dCA_CA"]
            dCY_CY[:,j] = basisj["dCY_CY"]
            dCY_Cn[:,j] = basisj["dCY_CLN"]
            dCN_CN[:,j] = basisj["dCN_CN"]
            dCN_Cm[:,j] = basisj["dCN_CLM"]
       # --- Cleanup ---
        # Column names (ensured to work b/c of gnr8_ll3x_basis() call)
        col1, col2, col3 = cols
        # Check for manual names of output columns
        ocols = kw.get("ocols")
        # Ensure list
        if not isinstance(ocols, (list, tuple)):
            ocols = []
        # Number of manual columns
        n_ocol = len(ocols)
        # Output column names
        colAA = col1 + ".deltaCA"
        colYY = col2 + ".deltaCY"
        colYn = col2 + ".deltaCLN"
        colNN = col3 + ".deltaCN"
        colNm = col3 + ".deltaCLM"
        # Check for overrides
        if n_ocol > 0:
            colAA = ocols[0]
        if n_ocol > 1:
            colYY = ocols[1]
        if n_ocol > 2:
            colYn = ocols[2]
        if n_ocol > 3:
            colNN = ocols[3]
        if n_ocol > 4:
            colNm = ocols[4]
        # Save columns
        self.save_col(colAA, dCA_CA)
        self.save_col(colYY, dCY_CY)
        self.save_col(colYn, dCY_Cn)
        self.save_col(colNN, dCN_CN)
        self.save_col(colNm, dCN_Cm)
        # Save definitions
        self.make_defn(colAA, dCA_CA)
        self.make_defn(colYY, dCY_CY)
        self.make_defn(colYn, dCY_Cn)
        self.make_defn(colNN, dCN_CN)
        self.make_defn(colNm, dCN_Cm)
        # Check for output column
        ocol = kw.get("ocol")
        # Option to save it
        if ocol is not None:
            # Save the slice values
            self.save_col(ocol, x_scol)
            # Make a defintion
            self.make_defn(ocol, x_scol)
        # Output
        return {
            "dCA_CA": dCA_CA,
            "dCY_CY": dCY_CY,
            "dCY_CLN": dCY_Cn,
            "dCN_CN": dCN_CN,
            "dCN_CLM": dCN_Cm,
        }

    # Create basis for line load of one component
    def genr8_ll3x_basis(self, cols, nPOD=10, mask=None, **kw):
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
            >>> basis = db.genr8_ll3x_basis(cols, nPOD=10, **kw)
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
            *basis["dCA_CA"]*: :class:`np.ndarray`
                Delta *dCA* load to adjust *CA* by ``1.0``
            *basis["dCY_CY"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CY* by ``1.0``
            *basis["dCY_CLN"]*: :class:`np.ndarray`
                Delta *dCY* load to adjust *CLN* by ``1.0``
            *basis["dCN_CN"]*: :class:`np.ndarray`
                Delta *dCN* load to adjust *CN* by ``1.0``
            *basis["dCN_CLM"]*: :class:`np.ndarray`
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
        bCM = np.hstack(([1.0, 0.0], np.zeros(nmode)))
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
            "dCA_CA": phiCA,
            "dCY_CY": phiCY,
            "dCN_CN": phiCN,
            "dCN_CLM": phiCm,
            "dCY_CLN": phiCn,
        }
        
  # >


# Combine options
kwutils._combine_val(DBLL._tagmap, dbfm.DBFM._tagmap)

# Invert the _tagmap
DBLL.create_tagcols()
