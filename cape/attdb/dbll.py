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

# CAPE modules
import cape.attdb.convert as convert
import cape.tnakit.kwutils as kwutils

# Local modules
from . import dbfm


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


# Combine options
kwutils._combine_val(DBLL._tagmap, dbfm.DBFM._tagmap)

# Invert the _tagmap
DBLL.create_tagcols()
