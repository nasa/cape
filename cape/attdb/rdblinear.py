#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.rdbline`: ATTDB with 1D output functions
===========================================================

This module provides the class :class:`DBResponseLinear` as a subclass
of :class:`DBResponseNull` that adds support for output variables that
have columns that are a function of one or more variables and whose
output is a function of one *other* variable.  For example
if *db* is an instance of :class:`DBResponseLinear` with an sectional
normal force coefficient line load *DCN* that is a function of *x*, and
also the entire *DCN(x)* line is a function of Mach number (``"mach"``)
and angle of attack (``"alpha"``), the database can allow for the
following syntaxes.

    .. code-block:: pycon

        >>> DCN = db("CN", mach=1.2, alpha=0.2)

Here *DCN* is a 1D array with the *DCN* sectional load at each *x*. In
addition, *x* can be looked up as a function of *mach* and *alpha*,
although in many cases it will be the same for every case.

    .. code-block:: pycon

        >>> x = db("x", mach=1.2, alpha=0.2)

Here *x* can be basically a constant lookup (same *x* for all
conditions) or it can vary, however it is mandatory that *x.shape* is
exactly the same as *DCN.shape*.

This is accomplished by implementing the special :func:`__call__`
method for this class.

This class inherits from :class:`cape.attdb.rdbscalar.DBResponseScalar`
so zero-dimensional outputs (e.g. integrated *CN* in the example above)
can also be part of the database.

"""

# Standard library modules
import os

# Third-party modules
import numpy as np

# Semi-optional third-party modules
try:
    import scipy.interpolate.rbf as scirbf
except ImportError:
    scirbf = None

# CAPE modules
import cape.tnakit.kwutils as kwutils
import cape.tnakit.typeutils as typeutils
import cape.tnakit.plot_mpl as pmpl

# Local modules, direct
from .rdbscalar import DBResponseScalar


# Declare base class
class DBResponseLinear(DBResponseScalar):
    r"""Basic database template with 1D output variables
    
    :Call:
        >>> db = DBResponseScalar(fname=None, **kw)
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
        *db*: :class:`cape.attdb.rdbscalar.DBResponseScalar`
            Database with scalar output functions
    :Versions:
        * 2019-12-31 ``@ddalle``: First version
    """
  # =====================
  # Class Attributes
  # =====================
  # <
   # --- Method Names ---
    # Method functions
    _method_funcs = dict(DBResponseScalar._method_funcs)
    _method_funcs[1] = {
        "exact": "eval_exact",
        "function": "eval_function",
        "multilinear": "eval_multilinear",
        "multilinear-schedule": "eval_multilinear_schedule",
        "nearest": "eval_nearest",
    }
  # >

  # ===================
  # Config
  # ===================
  # <
  # >

  # ===============
  # Eval/Call
  # ===============
  # <
   # --- Evaluation ---

   # --- Options: Get ---
    # Get xvars for output
    def get_output_xvars(self, col):
        r"""Get list of args to output for column *col*

        :Call:
            >>> xargs = db.get_output_xvars(col)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseLinear`
                Database with multidimensional output functions
            *col*: :class:`str`
                Name of column to evaluate
        :Outputs:
            *xargs*: {``[]``} | :class:`list`\ [:class:`str`]
                List of input args to one condition of *col*
        :Versions:
            * 2019-12-30 ``@ddalle``: First version
        """
        # Get column definition
        defn = self.get_defn(col)
        # Get dimensionality
        xargs = defn.get("OutputXVars")
        # De-None
        if xargs is None:
            xargs = []
        # Check type
        if not isinstance(xargs, list):
            raise TypeError(
                "OutputXVars for col '%s' must be list (got %s)"
                % (col, type(xargs)))
        # Output (copy)
        return list(xargs)

   # --- Options: Set ---
    # Set xvars for output
    def set_output_xvars(self, col, xargs):
        r"""Set list of args to output for column *col*

        :Call:
            >>> db.set_output_xvars(col, xargs)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseLinear`
                Database with multidimensional output functions
            *col*: :class:`str`
                Name of column to evaluate
                List of input args to one condition of *col*
        :Versions:
            * 2019-12-30 ``@ddalle``: First version
        """
        # Get column definition
        defn = self.get_defn(col)
        # De-None
        if xargs is None:
            xargs = []
        # Check type
        if not isinstance(xargs, list):
            raise TypeError(
                "OutputXVars for col '%s' must be list (got %s)"
                % (col, type(xargs)))
        # Check contents
        for (j, k) in enumerate(xargs):
            if not typeutils.isstr(k):
                raise TypeError(
                    "Output arg %i for col '%s' must be str (got %s)"
                    % (j, col, type(k)))
        # Set (copy)
        defn["OutputXVars"] = list(xargs)
  # >
