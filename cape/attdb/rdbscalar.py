#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.rdbscalar`: ATTDB with scalar output functions
================================================================

This module provides the class :class:`DBResponseScalar` as a subclass
of :class:`DBResponseNull` that adds support for output variables that
have scalars that are a function of one or more variables.  For example
if *db* is an instance of :class:`DBResponseScalar` with an axial force
coefficient *CA* that is a function of Mach number (``"mach"``) and
angle of attack (``"alpha"``), the database can allow for the following
syntax.

    .. code-block:: pycon

        >>> CA = db("CA", mach=1.2, alpha=0.2)

This is accomplished by implementing the special :func:`__call__`
method for this class.

This class also serves as a type test for non-null "response" databases
in the ATTDB framework.  Databases that can be evaluated in this manor
will pass the following test:

    .. code-block:: python

        isinstance(db, cape.attdb.rdbscalar.DBResponseScalar)

Because this class inherits from :class:`DBResponseNull`, it has
interfaces to several different file types.

"""

# Standard library modules
import os

# Third-party modules
import numpy as np

# CAPE modules
import cape.tnakit.typeutils as typeutils
import cape.tnakit.kwutils as kwutils

# Data types
import cape.attdb.ftypes as ftypes

# Local modules, direct
from .rdbnull import DBResponseNull


# Declare base class
class DBResponseScalar(DBResponseNull):
    r"""Basic database template with scalar output variables
    
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
            Generic database
    :Versions:
        * 2019-12-17 ``@ddalle``: First version
    """
  # =====================
  # Class Attributes
  # =====================
  # <
  # >

  # ===============
  # Eval/Call
  # ===============
  # <
   # --- Nearest/Exact ---
    # Find exact match
    def eval_exact(self, col, args, x, **kw):
        r"""Evaluate a coefficient by looking up exact matches

        :Call:
            >>> y = db.eval_exact(col, args, x, **kw)
            >>> Y = db.eval_exact(col, args, x, **kw)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of column to evaluate, with numeric data
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *tol*: {``1.0e-4``} | :class:`float` > 0
                Default tolerance for exact match
            *tols*: {``{}``} | :class:`dict`\ [:class:`float` > 0]
                Dictionary of key-specific tolerances
        :Outputs:
            *y*: ``None`` | :class:`float` | ``db[col].__class__``
                Value of ``db[col]`` exactly matching conditions *x*
            *Y*: :class:`np.ndarray`
                Multiple values matching exactly
        :Versions:
            * 2018-12-30 ``@ddalle``: First version
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Check for column
        if (col not in self.cols) or (col not in self):
            # Missing col
            raise KeyError("Col '%s' is not present" % col)
        # Get values
        V = self[col]
        # Create mask
        I = np.arange(len(V))
        # Tolerance dictionary
        tols = kw.get("tols", {})
        # Default tolerance
        tol = 1.0e-4
        # Loop through keys
        for (i, k) in enumerate(args):
            # Get value
            xi = x[i]
            # Get tolerance
            toli = tols.get(k, kw.get("tol", tol))
            # Apply test
            qi = np.abs(self[k][I] - xi) <= toli
            # Combine constraints
            I = I[qi]
            # Break if no matches
            if len(I) == 0:
                return None
        # Test number of outputs
        if len(I) == 1:
            # Single output
            return V[I[0]]
        else:
            # Multiple outputs
            return V[I]
  # >