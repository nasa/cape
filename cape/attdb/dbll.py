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
from . import dbfm


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
            ocol = self._get_CLM_from_CN(col)
        # Check if present
        if ocol in self:
            return self.get_values(ocol, mask=kw.get("mask"))
        # Calculate moment
        v = self.genr8_dclm(col, xcol, **kw)
        # Save them
        self.save_col(ocol, v)
        # Create definition
        self.make_defn(ocol, v)
        # Output
        return v

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
            ocol = self._get_CLM_from_CN(col)
        # Check if present
        if ocol in self:
            return self.get_values(ocol, mask=kw.get("mask"))
        # Calculate moment
        v = self.genr8_dclm(col, xcol, **kw)
        # Save them
        self.save_col(ocol, v)
        # Create definition
        self.make_defn(ocol, v)
        # Output
        return v
        
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
            ocol = self._get_CLM_from_CN(col)
        # Save them
        self.save_col(ocol, v)
        # Create definition
        self.make_defn(ocol, v)
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
        v = self.genr8_dclm(col, xcol, **kw)
        # Default column to save
        if ocol is None:
            # Try to convert name
            ocol = self._get_CLM_from_CN(col)
        # Save them
        self.save_col(ocol, v)
        # Create definition
        self.make_defn(ocol, v)
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
  # Adjustment
  # ==================
  # <
   # --- Basis ---
    # Create basis for line load of one component
    def genr8_ll3x_basis(self, cols, nPOD=10, mask=None, **kw):
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
