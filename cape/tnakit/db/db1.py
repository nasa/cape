#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`tnakit.db.db1`: Aero Task Team force & moment database modules
======================================================================

This module provides several classes to interface with various *point* or *0D*
databases.  That is, each piece of data is a scalar in all of the columns.  A
common example of this is a force & moment database, for which the module is
named, where there are several scalar force or moment coefficients at each
flight condition.

An example of a database that does not fit into this paradigm would be a
surface pressure database, in which there is a vector of surface pressures for
each flight condition.  In the terminology of these database tools, that is an
example of a 1D database.

"""

# System module
import os
import sys
# Numerics
import numpy as np
# More powerful interpolation
import scipy.interpolate
# Mat interface
import scipy.io as sio
import scipy.io.matlab.mio5_params as siom

# Local modules
from .. import statutils as stats
from .. import convert
from .. import arrayutils
from .. import typeutils

# Relative local modules
from ..plotutils import mpl

# Statistics
try:
    from scipy.stats import norm
    from scipy.stats import t as student
except ImportError:
    pass

# Spreadsheet interfaces
import xlrd

# Plotting
if getattr(os.sys, 'frozen', False):
    # Running in a bundle
    plt = None
else:
    # Import plotting
    import matplotlib.pyplot as plt
# Sys check


# Accepted list for eval_method
RBF_METHODS = [
    "rbf", "rbf-map", "rbf-linear"
]
# RBF function types
RBF_FUNCS = [
    "multiquadric",
    "inverse_multiquadric",
    "gaussian",
    "linear",
    "cubic",
    "quintic",
    "thin_plate"
]

# Force and moment class
class DBCoeff(dict):
    """Generic coefficient database and interpolation class

    :Call:
        >>> DBc = DBCoeff(mat=None, xls=None, csv=None, **kw)
    :Inputs:
        *mat*: :class:`str`
            Name of Matlab file to read
        *xls*: :class:`str` | :class:`xlrd.book.Book`
            Name of spreadsheet or interface to open workbook
        *csv*: :class:`str`
            Name of CSV file to read
        *sheet*: {``None``} | :class:`str`
            Name of worksheet to read if reading from spreadsheet
    :Outputs:
        *DBc*: :class:`tnakit.db.db1.DBCoeff`
            Coefficient lookup database
        *DBc.coeffs*: :class:`list` (:class:`str` | :class:`unicode`)
            List of coefficients present in the database
        *DBc[coeff]*: :class:`np.ndarray` (:class:`float`)
            Data for coefficient named *coeff*
    :Versions:
        * 2018-06-08 ``@ddalle``: First version
    """
  # ==========
  # Config
  # ==========
  # <
   # --- Built-ins ---
    # Initialization method
    def __init__(self, mat=None, xls=None, csv=None, **kw):
        """Initialization method

        :Versions:
            * 2018-06-08 ``@ddalle``: First version
            * 2019-02-27 ``@ddalle``: Offloaded to :func:`read_db1_DBCoeff`
        """
        # Defer to class-specific reader
        self.read_db1_DBCoeff(mat=mat, xls=xls, csv=csv, **kw)

    # Representation method
    def __repr__(self):
        """Representation method

        :Versions:
            * 2018-06-08 ``@ddalle``: First version
        """
        return "<%s ncoeff=%s>" % (self.__class__.__name__, len(self.coeffs))

    # String method
    def __str__(self):
        """String method

        :Versions:
            * 2018-06-08 ``@ddalle``: First version
        """
        return "<%s ncoeff=%s>" % (self.__class__.__name__, len(self.coeffs))

   # --- Init ---
    # No-func

   # --- Read ---
    # Read
    def read_db1_DBCoeff(self, **kw):
        """Read scalar database data

        :Call:
            >>> DBc.read_db1_DBCoeff(**kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient lookup database
            *mat*: :class:`str`
                Name of Matlab file to read
            *xls*: :class:`str` | :class:`xlrd.book.Book`
                Name of spreadsheet or interface to open workbook
            *csv*: :class:`str`
                Name of CSV file to read
            *sheet*: {``None``} | :class:`str`
                Name of worksheet to read if reading from spreadsheet
        :Versions:
            * 2019-02-27 ``@ddalle``: Taken from previous :func:`__init__`
        """
        # Get possible file names
        mat = kw.pop("mat", None)
        xls = kw.pop("xls", None)
        csv = kw.pop("csv", None)
        # Check input type
        if mat is not None:
            # Read MATLAB file
            self.ReadMat(mat, **kw)
        elif xls is not None:
            # Read spreadsheet
            # Check for *sheet* keyword
            if "sheet" not in kw:
                raise ValueError(
                    "No worksheet name given using keyword 'sheet'")
            # Get sheet name
            sheet = kw.get("sheet")
            # Remove from *kw*
            del kw["sheet"]
            # Read spreadsheet
            self.ReadXLS(xls, sheet, **kw)
        elif csv is not None:
            # Read CSV file
            self.ReadCSV(csv, **kw)
        else:
            # Initialize coefficient list
            self.coeffs = []

   # --- Copy ---
    # Copy
    def copy(self):
        """Copy a coefficient lookup database

        :Call:
            >>> DBi = DBc.copy()
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient lookup database
        :Outputs:
            *DBi*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient lookup database
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
        """
        # Form a new database
        DBi = self.__class__()
        # Copy relevant parts
        self.copy_db1_DBCoeff(DBi)
        # Output
        return DBi

    # Copy to other databse
    def copy_db1_DBCoeff(self, DBi):
        """Copy a coefficient lookup database

        :Call:
            >>> DBc.copy_db1_DBCoeff(DBi)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient lookup database
            *DBi*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient lookup database
        :Versions:
            * 2019-02-21 ``@ddalle``: First version
        """
        # Copy the coefficient list
        DBi.coeffs = list(self.coeffs)
        # Loop through columns
        for coeff in self.coeffs:
            DBi[coeff] = self[coeff].copy()
        # Copy interpolants, break points, and other dictionaries
        self.copyattr(DBi, "rbfs")
        self.copyattr(DBi, "bkpts")
        # Copy evaluation attributes
        self.copyattr(DBi, "eval_args")
        self.copyattr(DBi, "eval_method")
        self.copyattr(DBi, "eval_kwargs")
        self.copyattr(DBi, "eval_arg_aliases")
        self.copyattr(DBi, "eval_arg_converters")
        self.copyattr(DBi, "eval_arg_defaults")
        self.copyattr(DBi, "eval_func")
        self.copyattr(DBi, "eval_func_self")
        # Copy UQ attributes
        self.copyattr(DBi, "uq_keys")
        self.copyattr(DBi, "uq_keys_extra")
        self.copyattr(DBi, "uq_keys_shift")
        self.copyattr(DBi, "uq_funcs_extra")
        self.copyattr(DBi, "uq_funcs_shift")
        # Output
        return DBi

    # Copy an attribute if present
    def copyattr(self, DBi, k, vdef={}):
        """Make a shallow copy of an attribute if present

        :Call:
            >>> DBc.copyattr(DBi, k, vdef={})
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient lookup database
            *DBi*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient lookup database
            *k*: :class:`str`
                Name of attribute to copy
            *vdef*: {``{}``} | :class:`any`
                Default value for output attribute if *DBc.(k)* is not present
        :Effects:
            *DBi.(k)*: *vdef* | *DBc.(k)*
                Shallow copy of attribute from *DBc* or *vdef* if necessary
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
        """
        # Get attribute (with default)
        v = getattr(self, k, vdef)
        # Type
        t = v.__class__.__name__
        # Check the type in order to make a copy
        if t == "dict":
            # Copy dictionary
            setattr(DBi, k, dict(v))
        elif t == "list":
            # Copy list
            setattr(DBi, k, list(v))
        elif t == "ndarray":
            # Copy NumPy array
            setattr(DBi, k, v.copy())
        else:
            # No copy
            setattr(DBi, k, v)
  # >

  # =============
  # Eval/CALL
  # =============
  # <
   # --- Evaluation ---
    # Evaluate interpolation
    def __call__(self, *a, **kw):
        """Generic evaluation function

        :Call:
            >>> v = DBc(*a, **kw)
            >>> v = DBc(coeff, x0, x1, ...)
            >>> V = DBc(coeff, x0, X1, ...)
            >>> v = DBc(coeff, k0=x0, k1=x1, ...)
            >>> V = DBc(coeff, k0=x0, k1=X1, ...)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str` | :class:`unicode`
                Name of coefficient to evaluate
            *x0*: :class:`float` | :class:`int`
                Numeric value for first argument to *coeff* evaluator
            *x1*: :class:`float` | :class:`int`
                Numeric value for second argument to *coeff* evaluator
            *X1*: :class:`np.ndarray` (:class:`float`)
                Array of *x1* values
            *k0*: :class:`str` | :class:`unicode`
                Name of first argument to *coeff* evaluator
            *k1*: :class:`str` | :class:`unicode`
                Name of second argument to *coeff* evaluator
        :Outputs:
            *v*: :class:`float` | :class:`int`
                Function output for scalar evaluation
            *V*: :class:`np.ndarray` (:class:`float`)
                Array of function outputs
        :Versions:
            * 2019-01-07 ``@ddalle``: First version
        """
       # --- Get coefficient name ---
        # Process coefficient
        coeff, a, kw = self._process_coeff(*a, **kw)
       # --- Get method ---
        # Attempt to get default evaluation method
        try:
            # Check for evaluation methods and "_" key is default
            method_def = self.eval_method["_"]
        except AttributeError:
            # No evaluation method at all
            raise AttributeError("Database has no evaluation methods")
        except KeyError:
            # No default
            method_def = "nearest"
        # Specific method
        method_coeff = self.eval_method.get(coeff, method_def)
        # Check for ``None``, which forbids lookup
        if method_coeff is None:
            raise ValueError("Coeff '%s' is not an evaluation coeff" % coeff)
       # --- Get argument list ---
        # Specific lookup arguments
        args_coeff = self.get_eval_arg_list(coeff)
       # --- Evaluation kwargs ---
        # Attempt to get default aliases
        try:
            # Check for attribute and "_" default
            kw_def = self.eval_kwargs["_"]
            # Use default as fallback
            kw_fn = self.eval_kwargs.get(coeff, kw_def)
        except AttributeError:
            # No kwargs to eval functions
            kw_fn = {}
        except KeyError:
            # No default
            kw_fn = self.eval_kwargs.get(coeff, {})
       # --- Aliases ---
        # Attempt to get default aliases
        try:
            # Check for attribute and "_" default
            alias_def = self.eval_arg_aliases["_"]
            # Use default as fallback
            arg_aliases = self.eval_arg_aliases.get(coeff, alias_def)
        except AttributeError:
            # No aliases
            arg_aliases = {}
        except KeyError:
            # No default
            arg_aliases = self.eval_arg_aliases.get(coeff, {})
        # Process aliases in *kw*
        for k in dict(kw):
            # Check if there's an alias for *k*
            if k not in arg_aliases: continue
            # Get alias for keyword *k*
            alias_k = arg_aliases[k]
            # Save the value under the alias as well
            kw[alias_k] = kw.pop(k)
       # --- Argument values ---
        # Initialize lookup point
        x = []
        # Loop through arguments
        for i, k in enumerate(args_coeff):
            # Get value
            xi = self.get_arg_value(i, k, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        X, dims = self.__class__.NormalizeArguments(x)
        # Maximum dimension
        nd = len(dims)
        # Data size
        nx = np.prod(dims)
       # --- Evaluation ---
        # Process the appropriate lookup function
        if method_coeff in ["nearest"]:
            # Evaluate nearest-neighbor lookup
            f = lambda x: self.eval_nearest(coeff, args_coeff, x, **kw_fn)
        elif method_coeff in ["linear", "multilinear"]:
            # Evaluate using multilinear interpolation
            f = lambda x: self.eval_multilinear(
                coeff, list(args_coeff), x, **kw_fn)
        elif method_coeff in ["linear-schedule", "multilinear-schedule"]:
            # Evaluate using scheduled (in 1D) multilinear interpolation
            f = lambda x: self.eval_multilinear_schedule(
                coeff, list(args_coeff), x, **kw_fn)
        elif method_coeff in ["rbf"]:
            # Evaluate global radial basis function
            f = lambda x: self.eval_rbf(coeff, args_coeff, x, **kw_fn)
        elif method_coeff in ["rbf-slice", "rbf-linear"]:
            # Evaluate linear interpolation of two RBFs
            f = lambda x: self.eval_rbf_linear(coeff, args_coeff, x, **kw_fn)
        elif method_coeff in ["rbf-map", "rbf-schedule"]:
            # Evaluate curvilinear interpolation of slice RBFs
            f = lambda x: self.eval_rbf_schedule(coeff, args_coeff, x, **kw_fn)
        elif method_coeff in ["function", "func", "fn"]:
            # Combine args
            kw_fk = dict(kw_fn, **kw)
            # Evaluate specific function
            f = lambda x: self.eval_function(coeff, args_coeff, x, **kw_fk)
        else:
            # Unknown method
            raise ValueError("Could not interpret evaluation method '%s'"
                % method_coeff)
        # Calls
        if nd == 0:
            # Scalar call
            v = f(x)
            # Output
            return v
        else:
            # Initialize output
            V = np.zeros(nx)
            # Loop through points
            for j in range(nx):
                # Construct inputs
                xj = [Xi[j] for Xi in X]
                # Call scalar function
                V[j] = f(xj)
            # Reshape
            V = V.reshape(dims)
            # Output
            return V

   # --- Alternative Evaluation ---
    # Evaluate only exact matches
    def EvalExact(self, *a, **kw):
        """Evaluate a coefficient but only at points with exact matches

        :Call:
            >>> V, I, J, X = DBc.EvalExact(*a, **kw)
            >>> V, I, J, X = DBc(coeff, x0, X1, ...)
            >>> V, I, J, X = DBc(coeff, k0=x0, k1=X1, ...)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str` | :class:`unicode`
                Name of coefficient to evaluate
            *x0*: :class:`float` | :class:`int`
                Numeric value for first argument to *coeff* evaluator
            *x1*: :class:`float` | :class:`int`
                Numeric value for second argument to *coeff* evaluator
            *X1*: :class:`np.ndarray` (:class:`float`)
                Array of *x1* values
            *k0*: :class:`str` | :class:`unicode`
                Name of first argument to *coeff* evaluator
            *k1*: :class:`str` | :class:`unicode`
                Name of second argument to *coeff* evaluator
        :Outputs:
            *V*: :class:`np.ndarray` (:class:`float`)
                Array of function outputs
            *I*: :class:`np.ndarray` (:class:`int`)
                Indices of cases matching inputs (see :func:`FindMatches`)
            *J*: :class:`np.ndarray` (:class:`int`)
                Indices of matches within input arrays
            *X*: :class:`tuple` (:class:`np.ndarray` (:class:`float`))
                Values of arguments at exact matches
        :Versions:
            * 2019-03-11 ``@ddalle``: First version
        """
       # --- Get coefficient name ---
        # Process coefficient name and remaining coeffs
        coeff, a, kw = self._process_coeff(*a, **kw)
       # --- Matching values
        # Get list of arguments for this coefficient
        args = self.get_eval_arg_list(coeff)
        # Possibility of fallback values
        arg_defaults = getattr(self, "eval_arg_defaults", {})
        # Find exact matches
        I, J = self.FindMatches(args, *a, **kw)
        # Initialize values
        x = []
        # Loop through coefficients
        for i, k in enumerate(args):
            # Get values
            V = self.get_all_values(k)
            # Check for mismatch
            if V is None:
                # Attempt to get value from inputs
                xi = self.get_arg_value(i, k, *a, **kw)
                # Check for scalar
                if xi is None:
                    raise ValueError(
                        ("Could not generate array of possible values ") +
                        ("for argument '%s'" % k))
                elif typeutils.isarray(xi):
                    raise ValueError(
                        ("Could not generate fixed scalar for test values ") +
                        ("of argument '%s'" % k))
                # Save the scalar value
                x.append(xi)
            else:
                # Save array of varying test values
                x.append(V[I])
        # Normalize
        X, dims = self.__class__.NormalizeArguments(x)
       # --- Evaluation ---
        # Evaluate coefficient at matching points
        if coeff in self:
            # Use direct indexing
            V = self[coeff][I]
        else:
            # Use evaluator (necessary for coeffs like *CLMX*)
            V = self.__call__(coeff, *X, **kw)
        # Output
        return V, I, J, X

    # Evaluate UQ from coefficient
    def EvalUQ(self, *a, **kw):
        """Evaluate specified UQ coefficients for a specified coefficient

        This function will evaluate the UQ coefficients savd for a given
        nominal coefficient by referencing the appropriate subset of
        *DBc.eval_args* for any UQ coefficients.  For example if *CN* is a
        function of ``"mach"``, ``"alpha"``, and ``"beta"``; and *UCN* is a
        function of ``"mach"`` only, this function passes only the Mach numbers
        to *UCN* for evaluation.

        :Call:
            >>> U = DBc.EvalUQ(*a, **kw)
            >>> U = DBc.EvalUQ(coeff, x0, X1, ...)
            >>> U = DBc.EvalUQ(coeff, k0=x0, k1=x1, ...)
            >>> U = DBc.EvalUQ(coeff, k0=x0, k1=X1, ...)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                Coefficient database interface
            *DBc.uq_coeffs*: :class:`dict` (:class:`str` | :class:`list`)
                Dictionary of UQ coefficient names for each coefficient
            *coeff*: :class:`str` | :class:`unicode`
                Name of coefficient to evaluate
            *x0*: :class:`float` | :class:`int`
                Numeric value for first argument to *coeff* evaluator
            *x1*: :class:`float` | :class:`int`
                Numeric value for second argument to *coeff* evaluator
            *X1*: :class:`np.ndarray` (:class:`float`)
                Array of *x1* values
            *k0*: :class:`str` | :class:`unicode`
                Name of first argument to *coeff* evaluator
            *k1*: :class:`str` | :class:`unicode`
                Name of second argument to *coeff* evaluator
        :Outputs:
            *U*: :class:`dict` (:class:`float` | :class:`np.ndarray`)
                Values of relevant UQ coefficients by name
        :Versions:
            * 2019-03-07 ``@ddalle``: First version
        """
       # --- Get coefficient name ---
        # Process coefficient name and remaining coeffs
        coeff, a, kw = self._process_coeff(*a, **kw)
       # --- Argument processing ---
        # Specific lookup arguments
        args_coeff = self.get_eval_arg_list(coeff)
        # Initialize lookup point
        x = []
        # Loop through arguments
        for i, k in enumerate(args_coeff):
            # Get value
            xi = self.get_arg_value(i, k, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        X, dims = self.__class__.NormalizeArguments(x)
        # Maximum dimension
        nd = len(dims)
       # --- UQ coeff ---
        # Dictionary of UQ coefficients
        uq_coeffs = getattr(self, "uq_coeffs", {})
        # Coefficients for this coefficient
        uq_coeff = uq_coeffs.get(coeff, [])
        # Check for list
        if typeutils.isarray(uq_coeff):
            # Save a flag for list of coeffs
            qscalar = False
            # Pass coefficient to list
            uq_coeff_list = list(uq_coeff)
        else:
            # Save a flag for scalar output
            qscalar = True
            # Make a list
            uq_coeff_list = [uq_coeff]
       # --- Evaluation ---
        # Initialize output
        U = {}
        # Loop through UQ coeffs
        for uk in uq_coeff_list:
            # Get evaluation args
            args_k = self.get_eval_arg_list(uk)
            # Initialize inputs to *uk*
            UX = []
            # Loop through eval args
            for ai in args_k:
                # Check for membership
                if ai not in args_coeff:
                    raise ValueError(
                        ("UQ coeff '%s' is a function of " % uk) +
                        ("'%s', but parent coeff '%s' is not" % (ai, coeff)))
                # Append value
                UX.append(X[args_coeff.index(ai)])
            # Evaluate
            U[uk] = self.__call__(uk, *UX, **kw)

       # --- Output ---
        # Check for scalar output
        if qscalar:
            # Return first value
            return U[uk]
        else:
            # Return list
            return U

    # Evaluate coefficient from arbitrary list of arguments
    def EvalFromArgList(self, coeff, args, *a, **kw):
        """Evaluate coefficient from arbitrary argument list

        This function is used to evaluate a coefficient when given the
        arguments to some other coefficient.

        :Call:
            >>> V = DBc.EvalFromArgList(coeff, args, *a, **kw)
            >>> V = DBc.EvalFromArgList(coeff, args, x0, X1, ...)
            >>> V = DBc.EvalFromArgList(coeff, args, k0=x0, k1=x1, ...)
            >>> V = DBc.EvalFromArgList(coeff, args, k0=x0, k1=X1, ...)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str` | :class:`unicode`
                Name of coefficient to evaluate
            *args*: :class:`list` (:class:`str`)
                List of arguments provided
            *x0*: :class:`float` | :class:`int`
                Numeric value for first argument to *coeff* evaluator
            *x1*: :class:`float` | :class:`int`
                Numeric value for second argument to *coeff* evaluator
            *X1*: :class:`np.ndarray` (:class:`float`)
                Array of *x1* values
            *k0*: :class:`str` | :class:`unicode`
                Name of first argument to *coeff* evaluator
            *k1*: :class:`str` | :class:`unicode`
                Name of second argument to *coeff* evaluator
        :Outputs:
            *V*: :class:`float` | :class:`np.ndarray`
                Values of *coeff* as appropriate
        :Versions:
            * 2019-03-13 ``@ddalle``: First version
        """
       # --- Argument processing ---
        # Specific lookup arguments for *coeff*
        args_coeff = self.get_eval_arg_list(coeff)
        # Initialize lookup point
        x = []
        # Loop through arguments asgiven
        for i, k in enumerate(args):
            # Get value
            xi = self.get_arg_value(i, k, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        X, dims = self.__class__.NormalizeArguments(x)
        # Maximum dimension
        nd = len(dims)
       # --- Evaluation ---
        # Initialize inputs to *coeff*
        A = []
        # Get aliases for this coeffiient
        aliases = getattr(self, "eval_arg_aliases", {})
        aliases = aliases.get(coeff, {})
        # Loop through eval args
        for ai in args_coeff:
            # Check for membership
            if ai in args:
                # Append value
                A.append(X[args.index(ai)])
                continue
            # Check aliases
            for k, v in aliases.items():
                # Check if we found the argument sought for
                if v != ai:
                    continue
                # Check if this alias is in the provided list
                if k in args:
                    # Replacement argument name
                    ai = k
            # Check for membership (second try)
            if ai in args:
                # Append value
                A.append(X[args.index(ai)])
                continue
            raise ValueError(
                ("Coeff '%s' is a function of " % coeff) +
                ("'%s', not provided in argument list" % ai))
        # Evaluate
        return self.__call__(coeff, *A, **kw)

    # Evaluate coefficient from arbitrary list of arguments
    def EvalFromIndex(self, coeff, I, **kw):
        """Evaluate coefficient from indices

        This function looks up the appropriate input variables and uses them to
        generate inputs to the database evaluation method.

        :Call:
            >>> V = DBc.EvalFromArgList(coeff, I, **kw)
            >>> v = DBc.EvalFromArgList(coeff, i, **kw)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str` | :class:`unicode`
                Name of coefficient to evaluate
            *I*: :class:`np.ndarray`
                Indices at which to evaluation function
            *i*: :class:`int`
                Single index at which to evaluate
        :Outputs:
            *V*: :class:`np.ndarray`
                Values of *coeff* as appropriate
            *v*: :class:`float`
                Scalar evaluation of *coeff*
        :Versions:
            * 2019-03-13 ``@ddalle``: First version
        """
       # --- Argument processing ---
        # Specific lookup arguments for *coeff*
        args_coeff = self.get_eval_arg_list(coeff)
       # --- Evaluation ---
        # Initialize inputs to *coeff*
        A = []
        # Loop through eval args
        for ai in args_coeff:
            # Append value
            A.append(self.GetXValues(ai, I, **kw))
        # Evaluate
        return self.__call__(coeff, *A, **kw)

   # --- Attributes ---
    # Get argument list
    def get_eval_arg_list(self, coeff):
        """Get list of evaluation arguments

        :Call:
            >>> args = DBc.get_eval_arg_list(coeff)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str` | :class:`unicode`
                Name of coefficient to evaluate
        :Outputs:
            *args*: :class:`list` (:class:`str`)
                List of parameters used to evaluate *coeff*
        :Versions:
            * 2019-03-11 ``@ddalle``: Forked from :func:`__call__`
        """
        # Attempt to get default
        try:
            # Check for attribute and "_" default
            args_def = self.eval_args["_"]
        except AttributeError:
            # No argument lists at all
            raise AttributeError("Database has no evaluation argument lists")
        except KeyError:
            # No default
            args_def = None
        # Specific lookup arguments
        args_coeff = self.eval_args.get(coeff, args_def)
        # Check for ``None``, which forbids lookup
        if args_coeff is None:
            raise ValueError("Coeff '%s' is not an evaluation coeff" % coeff)
        # Output a copy
        return list(args_coeff)

    # Get evaluation method
    def get_eval_method(self, coeff):
        """Get evaluation method (if any) for a coefficient

        :Call:
            >>> meth = DBc.get_eval_method(coeff)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str` | :class:`unicode`
                Name of coefficient to evaluate
        :Outputs:
            *meth*: ``None`` | :class:`str`
                Name of evaluation method, if any
        :Versions:
            * 2019-03-13 ``@ddalle``: First version
        """
        # Get attribute
        try:
            # Get dictionary
            eval_methods = self.eval_method
        except AttributeError:
            # Set default
            self.eval_method = {}
            # Default value
            eval_methods = self.eval_method
        # Get method
        return eval_methods.get(coeff)

    # Get evaluation argument converter
    def get_eval_arg_converter(self, k):
        """Get evaluation argument converter

        :Call:
            >>> f = DBc.get_eval_arg_converter(k)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                Coefficient database interface
            *k*: :class:`str` | :class:`unicode`
                Name of argument
        :Outputs:
            *f*: ``None`` | :class:`function`
                Callable converter
        :Versions:
            * 2019-03-13 ``@ddalle``: First version
        """
        # Get converter dictionary
        try:
            # Get dictionary of converters
            converters = self.eval_arg_converters
        except AttributeError:
            # Default values
            converters = {}
            # Save
            self.eval_arg_converters = {}
        # Get converter
        f = converters.get(k)
        # Output if None
        if f is None: return f
        # Check class
        if not hasattr(f, "__call__"):
            raise TypeError("Converter for '%s' is not callable" % k)
        # Output
        return f

    # Get UQ coefficient
    def get_uq_coeff(self, coeff):
        """Get name of UQ coefficient(s) for *coeff*

        :Call:
            >>> ucoeff = DBc.get_uq_coeff(coeff)
            >>> ucoeffs = DBc.get_uq_coeff(coeff)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str` | :class:`unicode`
                Name of coefficient to evaluate
        :Outputs:
            *ucoeff*: ``None`` | :class:`str`
                Name of UQ coefficient for *coeff*
            *ucoeffs*: :class:`list` (:class:`str`)
                List of UQ coefficients for *coeff*
        :Versions:
            * 2019-03-13 ``@ddalle``: First version
        """
        # Get attribute
        try:
            # Get dictionary
            uq_coeffs = self.uq_coeffs
        except AttributeError:
            # Set default
            self.uq_coeffs = {}
            # Get handle
            uq_coeffs = self.uq_coeffs
        # Get entry for this coefficient
        return uq_coeffs.get(coeff)

   # --- Arguments ---
    # Process coefficient name
    def _process_coeff(self, *a, **kw):
        """Process coefficient name from arbitrary inputs

        :Call:
            >>> coeff, a, kw = DBc._process_coeff(*a, **kw)
            >>> coeff, a, kw = DBc._process_coeff(coeff, *a, **kw)
            >>> coeff, a, kw = DBc._process_coeff(*a, coeff=c, **kw)
        :Inputs:
            *coeff*: :class:`str`
                Name of input coefficient
            *a*: :class:`tuple`
                Other sequential inputs
            *kw*: :class:`dict`
                Other keyword inputs
        :Outputs:
            *coeff*: :class:`str`
                Coefficient name processed from either args or kwargs
            *a*: :class:`tuple`
                Remaining inputs with coefficient name removed
            *kw*: :class:`dict`
                Keyword inputs with coefficient name removed
        :Versions:
            * 2019-03-12 ``@ddalle``: First version
        """
        # Check for keyword
        coeff = kw.pop("coeff", None)
        # Check for string
        if typeutils.isstr(coeff):
            # Output
            return coeff, a, kw
        # Number of direct inputs
        na = len(a)
        # Process *coeff* from *a* if possible
        if na > 0:
            # First argument is coefficient
            coeff = a[0]
            # Check for string
            if typeutils.isstr(coeff):
                # Remove first entry
                a = a[1:]
                # Output
                return coeff, a, kw
        # Must be string-like
        raise TypeError("Coefficient must be a string")

    # Get argument value
    def get_arg_value(self, i, k, *a, **kw):
        """Get the value of the *i*\ th argument to a function

        :Call:
            >>> v = DBc.get_arg_value(i, k, *a, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *i*: :class:`int`
                Argument index within *DBc.eval_args*
            *k*: :class:`str`
                Name of evaluation argument
            *a*: :class:`tuple`
                Arguments provided by :func:`__call__`
            *kw*: :class:`dict`
                Keyword arguments provided by :func:`__call__`
        :Outputs:
            *v*: :class:`float` | :class:`np.ndarray`
                Value of the argument, possibly converted
        :Versions:
            * 2019-02-28 ``@ddalle``: First version
        """
        # Number of direct arguments
        na = len(a)
        # Converters
        arg_converters = getattr(self, "eval_arg_converters", {})
        arg_defaults   = getattr(self, "eval_arg_defaults",   {})
        # Check for sufficient non-keyword inputs
        if na > i:
            # Directly specified
            xi = kw.get(k, a[i])
        else:
            # Get from keywords
            xi = kw.get(k)
        # In most cases, this is sufficient
        if xi is not None:
            return xi
        # Check for a converter
        fk = arg_converters.get(k)
        # Apply converter
        if fk:
            # Apply converters
            try:
                # Convert values
                try:
                    # Use args and kwargs
                    xi = fk(*x, **kw)
                except Exception:
                    # Use just kwargs
                    xi = fk(**kw)
                # Save it
                if xi is not None: return xi
            except Exception:
                # Function failure
                print("Eval argument converter for '%s' failed")
        # Get default
        xi = arg_defaults.get(k)
        # Final check
        if xi is None:
            # No value determined
            raise ValueError(
                "Could not determine value for argument '%s'" % k)
        else:
            # Final output
            return xi

    # Get dictionary of argument values
    def get_arg_value_dict(self, *a, **kw):
        """Return a dictionary of normalized argument variables

        Specifically, he dictionary contains a key for every argument used to
        evaluate the coefficient that is either the first argument or uses the
        keyword argument *coeff*.

        :Call:
            >>> X = DBc.get_arg_value_dict(*a, **kw)
            >>> X = DBc.get_arg_value_dict(coeff, x1, x2, ..., k3=x3)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient
            *x1*: :class:`float` | :class:`np.ndarray`
                Value(s) of first argument
            *x2*: :class:`float` | :class:`np.ndarray`
                Value(s) of second argument, if applicable
            *k3*: :class:`str`
                Name of third argument or optional variant
            *x3*: :class:`float` | :class:`np.ndarray`
                Value(s) of argument *k3*, if applicable
        :Outputs:
            *X*: :class:`dict` (:class:`np.ndarray`)
                Dictionary of values for each key used to evaluate *coeff*
                according to *DBc.eval_args[coeff]*; each entry of *X* will
                have the same size
        :Versions:
            * 2019-03-12 ``@ddalle``: First version
        """
       # --- Get coefficient name ---
        # Use normal situation
        coeff, a, kw = self._process_coeff(*a, **kw)
       # --- Argument processing ---
        # Specific lookup arguments
        args_coeff = self.get_eval_arg_list(coeff)
        # Initialize lookup point
        x = []
        # Loop through arguments
        for i, k in enumerate(args_coeff):
            # Get value
            xi = self.get_arg_value(i, k, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        xn, dims = self.__class__.NormalizeArguments(x)
       # --- Output ---
        # Initialize
        X = {}
        # Loop through args
        for i, k in enumerate(args_coeff):
            # Save value
            X[k] = xn[i]
        # Output
        return X


    # Attempt to get all values of an argument
    def get_all_values(self, k):
        """Attempt to get all values of a specified argument

        This will use *eval_arg_converters* if possible

        :Call:
            >>> V = DBc.get_all_values(k)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *k*: :class:`str`
                Name of evaluation argument
        :Outputs:
            *V*: ``None`` | :class:`np.ndarray` (:class:`float`)
                *DBc[k]* if available, otherwise an attempt to apply
                *DBc.eval_arg_converters[k]*
        :Versions:
            * 2019-03-11 ``@ddalle``: First version
        """
        # Check if present
        if k in self:
            # Get values
            return self[k]
        # Otherwise check for evaluation argument
        arg_converters = getattr(self, "eval_arg_converters", {})
        # Check if there's a converter
        if k not in arg_converters:
            return None
        # Get converter
        f = arg_converters[k]
        # Attempt to apply it
        try:
            # Call in keyword-only mode
            V = f(**self)
            # Return values
            return V
        except Exception:
            # Failed
            return None

    # Normalize arguments
    @staticmethod
    def NormalizeArguments(x, asarray=False):
        """Normalized mixed float and array arguments

        :Call:
            >>> X, dims = DBCoeff.NormalizeArguments(x, asarray=False)
        :Inputs:
            *x*: :class:`list` (:class:`float` | :class:`np.ndarray`)
                Values for arguments, either float or array
            *asarray*: ``True`` | {``False``}
                Force array output
        :Outputs:
            *X*: :class:`list` (:class:`np.ndarray`)
                Normalized arrays all with same size
            *dims*: :class:`tuple` (:class:`int`)
                Original dimensions of non-scalar input array
        :Versions:
            * 2019-03-11 ``@ddalle``: First version
            * 2019-03-14 ``@ddalle``: Added *asarray* input
        """
        # Input size by argument
        nxi = [xi.size for xi in x]
        ndi = [xi.ndim for xi in x]
        # Maximum size
        nx = np.max(nxi)
        nd = np.max(ndi)
        # Index of maximum size
        ix = nxi.index(nx)
        # Corresponding shape
        dims = x[ix].shape
        # Check for forced array output
        if asarray and (nd == 0):
            # Ensure 1D output
            nd = 1
            dims = (1,)
        # Initialize final arguments
        X = []
        # Loop through arguments again
        for i, xi in enumerate(x):
            # Check for trivial case
            if nd == 0:
                # Save scalar
                X.append(xi)
                continue
            # Get sizes
            nxk = nxi[i]
            ndk = ndi[i]
            # Check for expansion
            if ndk == 0:
                # Copy size
                X.append(xi*np.ones(nx))
            elif ndk != nd:
                # Inconsistent size
                raise ValueError(
                    "Cannot normalize %id and %id inputs" % (ndk, nd))
            elif nxk != nx:
                # Inconsistent size
                raise IndexError(
                    "Cannot normalize inputs with size %i and %i" % (nxk, nx))
            else:
                # Already array
                X.append(xi.flatten())
        # Output
        return X, dims

   # --- Options ---
    # Set a default value for an argument
    def set_arg_default(self, k, v):
        """Set a default value for an evaluation argument

        :Call:
            >>> DBc.set_arg_default(k, v)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *k*: :class:`str`
                Name of evaluation argument
            *v*: :class:`float`
                Default value of the argument to set
        :Versions:
            * 2019-02-28 ``@ddalle``: First version
        """
        # Ensure attribute exists
        try:
            self.eval_arg_defaults
        except AttributeError:
            self.eval_arg_defaults = {}
        # Save key/value
        self.eval_arg_defaults[k] = v

    # Set a conversion function for input variables
    def set_arg_converter(self, k, fn):
        """Set a function to evaluation argument for a specific argument

        :Call:
            >>> DBc.set_arg_converter(k, fn)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *k*: :class:`str`
                Name of evaluation argument
            *fn*: :class:`function`
                Conversion function
        :Versions:
            * 2019-02-28 ``@ddalle``: First version
        """
        # Ensure attribute exists
        try:
            self.eval_arg_converters
        except AttributeError:
            self.eval_arg_converters = {}
        # Save function
        self.eval_arg_converters[k] = fn

   # --- Declaration ---
    # Set evaluation methods
    def SetEvalMethod(self, coeffs=None, method=None, args=None, *a, **kw):
        """Set evaluation method for a one or more coefficients

        :Call:
            >>> DBc.SetEvalMethod(coeff, method=None, args=None, **kw)
            >>> DBc.SetEvalMethod(coeffs, method=None, args=None, **kw)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                Coefficient database interface
            *coeffs*: :class:`list` (:class:`str` | :class:`unicode`)
                List of coefficients to declare
            *coeff*: :class:`str` | :class:`unicode`
                Name of coefficient to evaluate
            *method*: ``"nearest"`` | ``"linear"`` | ``"rbf"`` | :class:`str`
                Interpolation/evaluation method
            *args*: :class:`list` (:class:`str`)
                List of input arguments
            *aliases*: {``{}``} | :class:`dict` (:class:`str`)
                Dictionary of alternate variable names during evaluation; if
                *aliases[k1]* is *k2*, that means *k1* is an alternate name for
                *k2*, and *k2* is in *args*
            *eval_kwargs*: {``{}``} | :class:`dict`
                Keyword arguments passed to functions
            *I*: {``None``} | :class:`np.ndarray`
                Indices of cases to include in RBF (default is all)
            *function*: {``"cubic"``} | :class:`str`
                Radial basis function type
            *smooth*: {``0.0``} | :class:`float` >= 0
                Smoothing factor, ``0.0`` for exact interpolation
        :Versions:
            * 2019-01-07 ``@ddalle``: First version
        """
        # Get coefficients type
        t = coeffs.__class__.__name__
        # Check for list
        if t not in ["list", "ndarray"]:
            # Singleton list
            coeffs = [coeffs]
        # Loop through coefficients
        for coeff in coeffs:
            self._set_method1(coeff, method, args, *a, **kw)

    # Save a method for one coefficient
    def _set_method1(self, coeff=None, method=None, args=None, *a, **kw):
        """Set evaluation method for a single coefficient

        :Call:
            >>> DBc._set_method1(coeff=None, method=None, args=None, **kw)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str` | :class:`unicode`
                Name of coefficient to evaluate
            *method*: ``"nearest"`` | ``"linear"`` | ``"rbf"`` | :class:`str`
                Interpolation/evaluation method
            *args*: :class:`list` (:class:`str`)
                List of input arguments
            *aliases*: {``{}``} | :class:`dict` (:class:`str`)
                Dictionary of alternate variable names during evaluation; if
                *aliases[k1]* is *k2*, that means *k1* is an alternate name for
                *k2*, and *k2* is in *args*
            *eval_kwargs*: {``{}``} | :class:`dict`
                Keyword arguments passed to functions
            *I*: {``None``} | :class:`np.ndarray`
                Indices of cases to include in RBF (default is all)
            *function*: {``"cubic"``} | :class:`str`
                Radial basis function type
            *smooth*: {``0.0``} | :class:`float` >= 0
                Smoothing factor, ``0.0`` for exact interpolation
        :Versions:
            * 2019-01-07 ``@ddalle``: First version
        """
       # --- Metadata checks ---
        # Ensure metadata is present
        try:
            # Access dictionary of evaluation method
            self.eval_method
        except AttributeError:
            # Create it
            self.eval_method = {}
        # Argument lists
        try:
            # Access dictionary of argument lists
            self.eval_args
        except AttributeError:
            # Create it
            self.eval_args = {}
        # Argument aliases
        try:
            # Access dictionary of argument alias names
            self.eval_arg_aliases
        except AttributeError:
            # Create it
            self.eval_arg_aliases = {}
        # Evaluation keyword arguments
        try:
            # Access dictionary of keywords to evaluators
            self.eval_kwargs
        except AttributeError:
            # Create it
            self.eval_kwargs = {}
       # --- Input checks ---
        # Check inputs
        if coeff is None:
            # Set the default
            coeff = "_"
        # Check for valid argument list
        if args is None:
            raise ValueError("Argument list (keyword 'args') is required")
        # Check for method
        if method is None:
            raise ValueError("Eval method (keyword 'method') is required")
        # Get alias option
        arg_aliases = kw.get("aliases", {})
        # Check for ``None``
        if (not arg_aliases):
            # Empty option is empty dictionary
            arg_aliases = {}
        # Save aliases
        self.eval_arg_aliases[coeff] = arg_aliases
        # Get alias option
        eval_kwargs = kw.get("eval_kwargs", {})
        # Check for ``None``
        if (not eval_kwargs):
            # Empty option is empty dictionary
            eval_kwargs = {}
        # Save keywords (new copy)
        self.eval_kwargs[coeff] = dict(eval_kwargs)
       # --- Method switch ---
        # Check for identifiable method
        if method in ["nearest"]:
            # Nearest-neighbor lookup
            self.eval_method[coeff] = "nearest"
        elif method in ["linear", "multilinear"]:
            # Linear/multilinear interpolation
            self.eval_method[coeff] = "multilinear"
        elif method in ["linear-schedule", "multilinear-schedule"]:
            # (N-1)D linear interp in last keys, 1D in first key
            self.eval_method[coeff] = "multilinear-schedule"
        elif method in ["rbf", "rbg-global", "rbf0"]:
            # Create global RBF
            self.CreateGlobalRBFs([coeff], args, **kw)
            # Metadata
            self.eval_method[coeff] = "rbf"
        elif method in ["lin-rbf", "rbf-linear", "linear-rbf"]:
            # Create RBFs on slices
            self.CreateSliceRBFs([coeff], args, **kw)
            # Metadata
            self.eval_method[coeff] = "rbf-linear"
        elif method in ["map-rbf", "rbf-schedule", "rbf-map", "rbf1"]:
            # Create RBFs on slices but scheduled
            self.CreateSliceRBFs([coeff], args, **kw)
            # Metadata
            self.eval_method[coeff] = "rbf-map"
        elif method in ["function", "fn", "func"]:
            # Create eval_func dictionary
            try:
                self.eval_func
            except AttributeError:
                self.eval_func = {}
            # Create eval_func dictionary
            try:
                self.eval_func_self
            except AttributeError:
                self.eval_func_self = {}
            # Get the function
            if len(a) > 0:
                # Function given as arg
                fn = a[0]
            else:
                # Function better be a keyword because there are no args
                fn = None

            # Save the function
            self.eval_func[coeff] = kw.get("function", kw.get("func", fn))
            self.eval_func_self[coeff] = kw.get("self", True)

            # Dedicated function
            self.eval_method[coeff] = "function"
        else:
            raise ValueError(
                "Did not recognize evaluation type '%s'" % method)
        # Argument list is the same for all methods
        self.eval_args[coeff] = args

   # --- Schedule Tools ---
    # Return break points for schedule
    def get_schedule(self, args, x, extrap=True):
        """Get lookup points for interpolation scheduled by master key

        This is a utility that is used for situations where the break
        points of some keys may vary as a schedule of another one.
        For example if the maximum angle of attack in the database is
        different at each Mach number.  This utility provides the
        appropriate point at which to interpolate the remaining keys
        at the value of the first key both above and below the input
        value.  The first argument, ``args[0]``, is the master key
        that controls the schedule.

        :Call:
            >>> i0, i1, f, x0, x1 = DBc.get_schedule(args, x, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *args*: :class:`list` (:class:`str`)
                List of input argument names (*args[0]* is master key)
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *extrap*: {``True``} | ``False``
                If ``False``, raise error when lookup value is outside
                break point range for any key at any slice
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is at upper bound
            *x0*: :class:`np.ndarray` (:class:`float`)
                Evaluation values for ``args[1:]`` at *i0*
            *x1*: :class:`np.ndarray` (:class:`float`)
                Evaluation values for ``args[1:]`` at *i1*
        :Versions:
            * 2019-04-19 ``@ddalle``: First version
            * 2019-07-26 ``@ddalle``: Vectorized
        """
        # Number of args
        narg = len(args)
        # Error check
        if narg < 2:
            raise ValueError("At least two args required for scheduled lookup")
        # Flag for array or scalar
        qvec = False
        # Number of points in arrays (if any)
        n = None
        # Loop through args
        for i, k in enumerate(args):
            # Get value
            V = x[i]
            # Check type
            if typeutils.isarray(V):
                # Turn on array flag
                qvec = True
                # Get size
                nk = len(V)
                # Check consistency
                if n is None:
                    # New size
                    n = nk
                elif nk != n:
                    # Inconsistent size
                    raise ValueError(
                        "Eval arg '%s' has size %i, expected %i" % (k, nk, n))
            elif not isinstance(V, (float, int, np.ndarray)):
                # Improper type
                raise TypeError(
                    "Eval arg '%s' has type '%s'" % (k, V.__class__.__name__))
        # Check for arrays
        if not qvec:
            # Call scalar version
            return self._get_schedule(args, x, extrap=extrap)
        # Initialize tuple of fixed-size lookup points
        X = tuple()
        # Loop through args again
        for i, k in enumerate(args):
            # Get value
            V = x[i]
            # Check type
            if isinstance(V, (float, int)):
                # Create constant-value array
                X += (V * np.ones(n),)
            else:
                # Copy array
                X += (V,)
        # Otherwise initialize arrays
        I0 = np.zeros(n, dtype="int")
        I1 = np.zeros(n, dtype="int")
        F  = np.zeros(n)
        # Initialize tuples of modified lookup points
        X0 = tuple([np.zeros(n) for i in range(narg-1)])
        X1 = tuple([np.zeros(n) for i in range(narg-1)])
        # Loop through points
        for j in range(n):
            # Get lookup points
            xj = tuple([X[i][j] for i in range(narg)])
            # Evaluations
            i0, i1, f, x0, x1 = self._get_schedule(
                list(args), xj, extrap=extrap)
            # Save indices
            I0[j] = i0
            I1[j] = i1
            # Save lookup fraction
            F[j] = f
            # Save modified lookup points
            for i in range(narg-1):
                X0[i][j] = x0[i]
                X1[i][j] = x1[i]
        # Output
        return I0, I1, F, X0, X1

    # Return break points for schedule
    def _get_schedule(self, args, x, extrap=True):
        """Get lookup points for interpolation scheduled by master key

        This is a utility that is used for situations where the break
        points of some keys may vary as a schedule of another one.
        For example if the maximum angle of attack in the database is
        different at each Mach number.  This utility provides the
        appropriate point at which to interpolate the remaining keys
        at the value of the first key both above and below the input
        value.  The first argument, ``args[0]``, is the master key
        that controls the schedule.

        :Call:
            >>> i0, i1, f, x0, x1 = DBc.get_schedule(args, x, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *args*: :class:`list` (:class:`str`)
                List of input argument names (*args[0]* is master key)
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *extrap*: {``True``} | ``False``
                If ``False``, raise error when lookup value is outside
                break point range for any key at any slice
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is at upper bound
            *x0*: :class:`np.ndarray` (:class:`float`)
                Evaluation values for ``args[1:]`` at *i0*
            *x1*: :class:`np.ndarray` (:class:`float`)
                Evaluation values for ``args[1:]`` at *i1*
        :Versions:
            * 2019-04-19 ``@ddalle``: First version
        """
        # Error check
        if len(args) < 2:
            raise ValueError("At least two args required for scheduled lookup")
        # Slice/scheduling key
        skey = args.pop(0)
        # Lookup value for first variable
        i0, i1, f = self.get_bkpt_index(skey, x[0])
        # Number of additional args
        narg = len(args)
        # Initialize lookup points at slice *i0* and slice *i1*
        x0 = np.zeros(narg)
        x1 = np.zeros(narg)
        # Loop through arguments
        for j, k in enumerate(args):
            # Get min and max values
            try:
                # Try the case of varying break points indexed to *skey*
                xmin0 = self.get_bkpt(k, i0, 0)
                xmin1 = self.get_bkpt(k, i1, 0)
                xmax0 = self.get_bkpt(k, i0, -1)
                xmax1 = self.get_bkpt(k, i1, -1)
            except TypeError:
                # Fixed break points (apparently)
                xmin0 = self.get_bkpt(k, 0)
                xmin1 = self.get_bkpt(k, 0)
                xmax0 = self.get_bkpt(k, -1)
                xmax1 = self.get_bkpt(k, -1)
            # Interpolate to current *skey* value
            xmin = (1-f)*xmin0 + f*xmin1
            xmax = (1-f)*xmax0 + f*xmax1
            # Get the progress fraction at current inter-slice *skey* value
            fj = (x[j+1] - xmin) / (xmax-xmin)
            # Check for extrapolation
            if not extrap and ((fj < -1e-3) or (fj - 1 > 1e-3)):
                # Raise extrapolation error
                raise ValueError(
                    ("Lookup value %.4e is outside " % x[j+1]) +
                    ("scheduled bounds [%.4e, %.4e]" % (xmin, xmax)))
            # Get lookup points at slices *i0* and *i1* using this prog frac
            x0[j] = (1-fj)*xmin0 + fj*xmax0
            x1[j] = (1-fj)*xmin1 + fj*xmax1
        # Output
        return i0, i1, f, x0, x1

   # --- Breakpoints ---
    # Function to get interpolation weights for uq
    def get_bkpt_index(self, k, v):
        """Get interpolation weights for 1D linear interpolation

        :Call:
            >>> i0, i1, f = DBc.get_bkpt_index(k, v)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *k*: :class:`str`
                Name of trajectory key in *FM.bkpts* for lookup
            *v*: :class:`float`
                Value at which to lookup
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is equal to upper bound
        :Versions:
            * 2018-12-30 ``@ddalle``: First version
        """
        # Extract values
        try:
            # Naive extractions
            V = np.asarray(self.bkpts[k])
        except AttributeError:
            # No break points
            raise AttributeError("No break point dictionary present")
        except KeyError:
            # Missing key
            raise KeyError(
                "Lookup key '%s' is not present in break point dict" % k)
        # Output
        return self._bkpt_index(V, v)

    # Function to get interpolation weights for uq
    def get_bkpt_index_schedule(self, k, v, j):
        """Get weights 1D interpolation of *k* at a slice of master key

        :Call:
            >>> i0, i1, f = DBc.get_bkpt_index_schedule(k, v, j)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *k*: :class:`str`
                Name of trajectory key in *FM.bkpts* for lookup
            *v*: :class:`float`
                Value at which to lookup
            *j*: :class:`int`
                Index of master "slice" key, if *k* has scheduled
                break points
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is equal to upper bound
        :Versions:
            * 2018-04-19 ``@ddalle``: First version
        """
        # Get potential values
        V = self._scheduled_bkpts(k, j)
        # Lookup within this vector
        return self._bkpt(V, v)

    # Get break point from vector
    def _bkpt_index(self, V, v):
        """Get interpolation weights for 1D interpolation

        :Call:
            >>> i0, i1, f = DBc._bkpt_index(V, v)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *V*: :class:`np.ndarray`\ [:class:`float`]
                1D array of data values
            *v*: :class:`float`
                Value at which to lookup
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is equal to upper bound
        :Versions:
            * 2018-12-30 ``@ddalle``: First version
        """
        # Get length
        n = V.size
        # Get min/max
        vmin = np.min(V)
        vmax = np.max(V)
        # Check for extrapolation cases
        if v < vmin - 1e-8*(vmax-vmin):
            # Extrapolation left
            return None, 0, 1.0
        elif v > vmax + 1e-8*(vmax-vmin):
            # Extrapolation right
            return n-1, None, 1.0
        # Otherwise, count up values below
        i0 = np.sum(V[:-1] <= v) - 1
        i1 = i0 + 1
        # Progress fraction
        f = (v - V[i0]) / (V[i1] - V[i0])
        # Output
        return i0, i1, f

    # Get a break point, with error checking
    def get_bkpt(self, k, *I):
        """Extract a breakpoint by index, with error checking

        :Call:
            >>> v = DBc.get_bkpt(k, *I)
            >>> v = DBc.get_bkpt(k)
            >>> v = DBc.get_bkpt(k, i)
            >>> v = DBc.get_bkpt(k, i, j)
            >>> v = DBc.get_bkpt(k, i, j, ...)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
            *I*: :class:`tuple`
                Tuple of lookup indices
            *i*: :class:`int`
                (Optional) first RBF list index
            *j*: :class:`int`
                (Optional) second RBF list index
        :Outputs:
            *v*: :class:`float` | :class:`np.ndarray` (:class:`float`)
                Break point or array of break points
        :Versions:
            * 2018-12-31 ``@ddalle``: First version
        """
        # Get the radial basis function
        try:
            v = self.bkpts[k]
        except AttributeError:
            # No radial basis functions at all
            raise AttributeError("No break points found")
        except KeyError:
            # No RBF for this coefficient
            raise KeyError("No break points for key '%s'" % k)
        # Number of indices given
        nd = len(I)
        # Loop through indices
        for n, i in enumerate(I):
            # Try to extract
            try:
                # Get the *ith* list entry
                v = v[i]
            except (IndexError, TypeError):
                # Reached scalar too soon
                raise TypeError(
                    ("Breakpoints for '%s':\n" % k) +
                    ("Expecting %i-dimensional " % nd) +
                    ("array but found %i-dim" % n))
        # Output
        return v

    # Get all break points
    def _scheduled_bkpts(self, k, j):
        """Get list of break points for key *k* at schedule *j*

        :Call:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *k*: :class:`str`
                Name of trajectory key in *FM.bkpts* for lookup
            *j*: :class:`int`
                Index of master "slice" key, if *k* has scheduled
                break points
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is equal to upper bound
        :Versions:
            * 2018-12-30 ``@ddalle``: First version
        """
        # Get the radial basis function
        try:
            V = self.bkpts[k]
        except AttributeError:
            # No radial basis functions at all
            raise AttributeError("No break points found")
        except KeyError:
            # No RBF for this coefficient
            raise KeyError("No break points for key '%s'" % str(k))
        # Get length
        n = len(V)
        # Size check
        if n == 0:
            raise ValueError("Found zero break points for key '%s'" % k)
        # Check first key for array
        if isinstance(V[0], (np.ndarray, list)):
            # Get break points for this slice
            V = V[j]
            # Reset size
            n = V.size
            # Recheck size
            if n == 0:
                raise ValueError("Found zero break points for key '%s'" % k)
        # Output
        return V

   # --- Linear ---
    # Multilinear lookup
    def eval_multilinear(self, coeff, args, x, **kw):
        """Perform linear interpolation in as many dimensions as necessary

        This assumes the database is ordered with the first entry of *args*
        varying the most slowly and that the data is perfectly regular.

        :Call:
            >>> y = DBc.eval_multilinear(coeff, args, x)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *bkpt*: ``True`` | {``False``}
                Whether or not to interpolate break points instead of data
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Interpolated value from ``DBc[coeff]``
        :Versions:
            * 2018-12-30 ``@ddalle``: First version
        """
        # Call root method without two of the options
        return self._eval_multilinear(coeff, args, x, **kw)

    # Evaluate multilinear interpolation with caveats
    def _eval_multilinear(self, coeff, args, x, I=None, j=None, **kw):
        """Perform linear interpolation in as many dimensions as necessary

        This assumes the database is ordered with the first entry of *args*
        varying the most slowly and that the data is perfectly regular.

        :Call:
            >>> y = DBc._eval_multilinear(coeff, args, x, I=None, j=None)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *I*: {``None``} | :class:`np.ndarray`\ [:class:`int`]
                Optional subset of database on which to perform
                interpolation
            *j*: {``None``} | :class:`int`
                Slice index, used by :func:`eval_multilinear_schedule`
            *bkpt*: ``True`` | {``False``}
                Whether or not to interpolate break points instead of data
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Interpolated value from ``DBc[coeff]``
        :Versions:
            * 2018-12-30 ``@ddalle``: First version
            * 2019-04-19 ``@ddalle``: Moved from :func:`eval_multilnear`
        """
        # Check for break-point evaluation flag
        bkpt = kw.get("bkpt", kw.get("breakpoint", False))
        # Possible values
        try:
            # Extract coefficient
            if bkpt:
                # Lookup from breakpoints
                V = self.bkpts[coeff]
            else:
                # Lookup from main data
                V = self[coeff]
        except KeyError:
            # Missing key
            raise KeyError("Coefficient '%s' is not present" % coeff)
        # Subset if appropriate
        if I is not None:
            # Attempt to subset
            try:
                # Select some indices
                V = V[I]
            except Exception:
                # Informative error
                raise ValueError(
                    "Failed to subset coeff '%s' using class '%s'"
                    % (coeff, I.__class__.__name__))
        # Number of keys
        nk = len(args)
        # Count
        n = len(V)
        # Get break points for this schedule
        bkpts = {}
        for k in args:
            bkpts[k] = self._scheduled_bkpts(k, j)
        # Lengths for each variable
        N = [len(bkpts[k]) for k in args]
        # Check consistency
        if np.prod(N) != n:
            raise ValueError(
                ("Coefficient '%s' has size %i, " % (coeff, n)),
                ("but total size of args %s is %i." % (args, np.prod(N))))
        # Initialize list of indices for each key
        I0 = []
        I1 = []
        F1 = []
        # Get lookup indices for each argument
        for i, k in enumerate(args):
            # Lookup value
            xi = x[i]
            # Values
            Vk = bkpts[k]
            # Get indices
            i0, i1, f = self._bkpt_index(Vk, xi)
            # Check for problems
            if i0 is None:
                # Below
                raise ValueError(
                    ("Value %s=%.4e " % (k, xi)) +
                    ("below lower bound (%.4e)" % Vk[0]))
            elif i1 is None:
                raise ValueError(
                    ("Value %s=%.4e " % (k, xi)) +
                    ("above upper bound (%.4e)" % Vk[-1]))
            # Save values
            I0.append(i0)
            I1.append(i1)
            F1.append(f)
        # Index of the lowest corner
        j0 = 0
        # Loop through the keys
        for i in range(nk):
            # Get value
            i0 = I0[i]
            # Overall ; multiply index by size of remaining block
            j0 += i0 * int(np.prod(N[i+1:]))
        # Initialize overall indices and weights
        J = j0 * np.ones(2**nk, dtype="int")
        F = np.ones(2**nk)
        # Counter from 0 to 2^nk-1
        E = np.arange(2**nk)
        # Loop through keys again
        for i in range(nk):
            # Exponent of two to use for this key
            e = nk - i
            # Up or down for each of the 2^nk individual lookup points
            jupdown = E % 2**e // 2**(e-1)
            # Size of remaining block
            subblock = int(np.prod(N[i+1:]))
            # Increment overall indices
            J += jupdown*subblock
            # Progress fraction for this variable
            fi = F1[i]
            # Convert up/down to either fi or 1-fi
            Fi = (1-fi)*(1-jupdown) + jupdown*fi
            # Apply weights
            F *= Fi
        # Perform interpolation
        return np.sum(F*V[J])

   # --- Multilinear-schedule ---
    # Multilinear lookup at each value of arg
    def eval_multilinear_schedule(self, coeff, args, x, **kw):
        """Perform linear interpolation in as many dimensions as necessary

        This assumes the database is ordered with the first entry of *args*
        varying the most slowly and that the data is perfectly regular.

        :Call:
            >>> y = DBc.eval_multilinear(coeff, args, x)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *tol*: {``1e-6``} | :class:`float` >= 0
                Tolerance for matching slice key
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Interpolated value from ``DBc[coeff]``
        :Versions:
            * 2019-04-19 ``@ddalle``: First version
        """
        # Slice tolerance
        tol = kw.get("tol", 1e-6)
        # Name of master (slice) key
        skey = args[0]
        # Get lookup points at both sides of scheduling key
        i0, i1, f, x0, x1 = self.get_schedule(args, x, extrap=False)
        # Get the values for the slice key
        x00 = self.get_bkpt(skey, i0)
        x01 = self.get_bkpt(skey, i1)
        # Find indices of the two slices
        I0 = np.where(np.abs(self[skey] - x00) <= tol)[0]
        I1 = np.where(np.abs(self[skey] - x01) <= tol)[0]
        # Perform interpolations
        y0 = self._eval_multilinear(coeff, args, x0, I=I0, j=i0)
        y1 = self._eval_multilinear(coeff, args, x1, I=I1, j=i1)
        # Linear interpolation in the schedule key
        return (1-f)*y0 + f*y1

   # --- Radial Basis Functions ---
    # Get an RBF
    def get_rbf(self, coeff, *I):
        """Extract a radial basis function, with error checking

        :Call:
            >>> f = DBc.get_rbf(coeff, *I)
            >>> f = DBc.get_rbf(coeff)
            >>> f = DBc.get_rbf(coeff, i)
            >>> f = DBc.get_rbf(coeff, i, j)
            >>> f = DBc.get_rbf(coeff, i, j, ...)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
            *I*: :class:`tuple`
                Tuple of lookup indices
            *i*: :class:`int`
                (Optional) first RBF list index
            *j*: :class:`int`
                (Optional) second RBF list index
        :Outputs:
            *f*: :class:`scipy.interpolate.rbf.Rbf`
                Callable radial basis function
        :Versions:
            * 2018-12-31 ``@ddalle``: First version
        """
        # Get the radial basis function
        try:
            fn = self.rbf[coeff]
        except AttributeError:
            # No radial basis functions at all
            raise AttributeError("No radial basis functions found")
        except KeyError:
            # No RBF for this coefficient
            raise KeyError("No radial basis function for coeff '%s'" % coeff)
        # Number of indices given
        nd = len(I)
        # Loop through indices
        for n, i in enumerate(I):
            # Try to extract
            try:
                # Get the *ith* list entry
                fn = fn[i]
            except TypeError:
                # Reached RBF too soon
                raise TypeError(
                    ("RBF for '%s':\n" % coeff) +
                    ("Expecting %i-dimensional " % nd) +
                    ("array but found %i-dim" % n))
        # Test type
        if fn.__class__.__name__ not in ["function", "Rbf"]:
            raise TypeError("RBF '%s' index %i is not callable" % (coeff, I))
        # Output
        return fn

    # RBF lookup
    def eval_rbf(self, coeff, args, x, **kw):
        """Evaluate a single radial basis function

        :Call:
            >>> y = DBc.eval_rbf(coeff, args, x)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Interpolated value from ``DBc[coeff]``
        :Versions:
            * 2018-12-31 ``@ddalle``: First version
        """
        # Get the radial basis function
        f = self.get_rbf(coeff)
        # Evaluate
        return f(*x)

   # --- Generic Function ---
    # Generic function
    def eval_function(self, coeff, args, x, **kw):
        """Evaluate a single user-saved function

        :Call:
            >>> y = DBc.eval_function(coeff, args, x)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Interpolated value from ``DBc[coeff]``
        :Versions:
            * 2018-12-31 ``@ddalle``: First version
        """
        # Get the function
        try:
            f = self.eval_func[coeff]
        except AttributeError:
            # No evaluation functions set
            raise AttributeError(
                "No evaluation functions present in database")
        except KeyError:
            # No keys
            raise KeyError(
                "No evaluation function for coeff '%s'" % coeff)
        # Evaluate
        if self.eval_func_self.get(coeff):
            # Use reference to *self*
            return f(self, *x, **kw)
        else:
            # Stand-alone function
            return f(*x, **kw)

   # --- RBF-linear ---
    # Multiple RBF lookup
    def eval_rbf_linear(self, coeff, args, x, **kw):
        """Evaluate two RBFs at slices of first *arg* and interpolate

        :Call:
            >>> y = DBc.eval_rbf_linear(coeff, args, x)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Interpolated value from ``DBc[coeff]``
        :Versions:
            * 2018-12-31 ``@ddalle``: First version
        """
        # Lookup value for first variable
        i0, i1, f = self.get_bkpt_index(args[0], x[0])
        # Get lookup functions for *i0* and *i1*
        f0 = self.get_rbf(coeff, i0)
        f1 = self.get_rbf(coeff, i1)
        # Evaluate both functions
        y0 = f0(*x[1:])
        y1 = f1(*x[1:])
        # Interpolate
        y = (1-f)*y0 + f*y1
        # Output
        return y

   # --- RBF-schedule ---
    # Multiple RBF lookup, curvilinear
    def eval_rbf_schedule(self, coeff, args, x, **kw):
        """Evaluate a single radial basis function

        :Call:
            >>> y = DBc.eval_rbf_schedule(coeff, args, x)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Interpolated value from ``DBc[coeff]``
        :Versions:
            * 2018-12-31 ``@ddalle``: First version
        """
        # Extrapolation option
        extrap = kw.get("extrap", False)
        # Get lookup points at both sides of scheduling key
        i0, i1, f, x0, x1 = self.get_schedule(list(args), x, extrap=extrap)
        # Get lookup functions for *i0* and *i1*
        f0 = self.get_rbf(coeff, i0)
        f1 = self.get_rbf(coeff, i1)
        # Evaluate the RBFs at both slices
        y0 = f0(*x0)
        y1 = f1(*x1)
        # Interpolate between the slices
        y = (1-f)*y0 + f*y1
        # Output
        return y

   # --- Nearest ---
    # Exact match
    def eval_exact(self, coeff, args, x, **kw):
        """Evaluate a coefficient by looking up exact matches

        :Call:
            >>> y = DBc.eval_exact(coeff, args, x, **kw)
            >>> Y = DBc.eval_exact(coeff, args, x, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *tol*: {``1.0e-4``} | :class:`float` > 0
                Default tolerance for exact match
            *tols*: {``{}``} | :class:`dict` (:class:`float` > 0)
                Dictionary of key-specific tolerances
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Value of ``DBc[coeff]`` exactly matching *x*
            *Y*: :class:`np.ndarray`
                Multiple values matching exactly
        :Versions:
            * 2018-12-30 ``@ddalle``: First version
        """
        # Possible values
        try:
            # Extract coefficient
            V = self[coeff]
        except KeyError:
            # Missing key
            raise KeyError("Coefficient '%s' is not present" % coeff)
        # Create mask
        I = np.arange(len(V))
        # Tolerance dictionary
        tols = kw.get("tols", {})
        # Default tolerance
        tol = 1.0e-4
        # Loop through keys
        for i, k in enumerate(args):
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

    # Lookup nearest value
    def eval_nearest(self, coeff, args, x, **kw):
        """Evaluate a coefficient by looking up nearest match

        :Call:
            >>> y = DBc.eval_nearest(coeff, args, x, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *weights*: {``{}``} | :class:`dict` (:class:`float` > 0)
                Dictionary of key-specific distance weights
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Value of ``DBc[coeff]`` exactly matching *x*
        :Versions:
            * 2018-12-30 ``@ddalle``: First version
        """
        # Possible values
        try:
            # Extract coefficient
            V = self[coeff]
        except KeyError:
            # Missing key
            raise KeyError("Coefficient '%s' is not present" % coeff)
        # Array length
        n = len(V)
        # Initialize distances
        d = np.zeros(V.shape, dtype="float")
        # Dictionary of distance weights
        W = kw.get("weights", {})
        # Loop through keys
        for i, k in enumerate(args):
            # Get value
            xi = x[i]
            # Get weight
            wi = W.get(k, 1.0)
            # Distance
            d += wi*(self[k] - xi)**2
        # Find minimum distance
        j = np.argmin(d)
        # Use that value
        return V[j]

   # --- RBF construction ---
    # Regularization
    def CreateGlobalRBFs(self, coeffs, args, I=None, **kw):
        """Create global radial basis functions for one or more coeffs

        :Call:
            >>> DBc.CreateGlobalRBFs(coeffs, args, I=None)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeffs*: :class:`list` (:class:`str`)
                List of coefficients to interpolate
            *args*: :class:`list` (:class:`str`)
                List of (ordered) input keys, default is from *DBc.bkpts*
            *I*: {``None``} | :class:`np.ndarray`
                Indices of cases to include in RBF (default is all)
            *function*: {``"cubic"``} | :class:`str`
                Radial basis function type
            *smooth*: {``0.0``} | :class:`float` >= 0
                Smoothing factor, ``0.0`` for exact interpolation
        :Outputs:
            *DBi*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface with regularized data
        :Versions:
            * 2019-01-01 ``@ddalle``: First version
        """
        # Create *rbf* attribute if needed
        try:
            # Test if present
            self.rbf
        except AttributeError:
            # Create the attribute
            self.rbf = {}
        # RBF options
        func   = kw.get("function", "cubic")
        smooth = kw.get("smooth", 0.0)
        # Default indices
        if I is None:
            # Size of database
            n = len(self[args[0]])
            # All the indices
            I = np.arange(n)
        # Create tuple of input points
        V = tuple(self[k][I] for k in args)
        # Loop through coefficients
        for coeff in coeffs:
            # Eval arguments for status update
            txt = str(tuple(args)).replace(" ", "")
            # Trim if too long
            if len(txt) > 50:
                txt = txt[:45] + "...)"
            # Status update line
            txt = "Creating RBF for %s%s" % (coeff, txt)
            sys.stdout.write("%-72s\r" % txt)
            sys.stdout.flush()
            # Append reference values to input tuple
            Z = V + (self[coeff][I],)
            # Create a single RBF
            f = scipy.interpolate.rbf.Rbf(*Z,
                function=func, smooth=smooth)
            # Save it
            self.rbf[coeff] = f
        # Clean up the prompt
        sys.stdout.write("%72s\r" % "")
        sys.stdout.flush()

    # Regularization
    def CreateSliceRBFs(self, coeffs, args, I=None, **kw):
        """Create global radial basis functions for each slice of first *arg*

        The first entry in *args* is interpreted as a "slice" key; RBFs will be
        constructed at constant values
        :Call:
            >>> DBc.CreateSliceRBFs(coeffs, args, I=None)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeffs*: :class:`list` (:class:`str`)
                List of coefficients to interpolate
            *args*: :class:`list` (:class:`str`)
                List of (ordered) input keys, default is from *DBc.bkpts*
            *I*: {``None``} | :class:`np.ndarray`
                Indices of cases to include in RBF (default is all)
            *function*: {``"cubic"``} | :class:`str`
                Radial basis function type
            *smooth*: {``0.0``} | :class:`float` >= 0
                Smoothing factor, ``0.0`` for exact interpolation
        :Outputs:
            *DBi*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface with regularized data
        :Versions:
            * 2019-01-01 ``@ddalle``: First version
        """
        # Create *rbf* attribute if needed
        try:
            # Test if present
            self.rbf
        except AttributeError:
            # Create the attribute
            self.rbf = {}
        # RBF options
        func  = kw.get("function", "cubic")
        smooth = kw.get("smooth", 0.0)
        # Name of slice key
        skey = args[0]
        # Tolerances
        tols = kw.get("tols", {})
        # Tolerance for slice key
        tol = kw.get("tol", tols.get(skey, 1e-6))
        # Default indices
        if I is None:
            # Size of database
            n = len(self[skey])
            # All the indices
            I = np.arange(n)
        # Get break points for slice key
        B = self.bkpts[skey]
        # Number of slices
        nslice = len(B)
        # Initialize the RBFs
        for coeff in coeffs:
            self.rbf[coeff] = []
        # Loop through slices
        for b in B:
            # Get slice constraints
            qj = np.abs(self[skey][I] - b) <= tol
            # Select slice and add to list
            J = I[qj]
            # Create tuple of input points
            V = tuple(self[k][J] for k in args[1:])
            # Create a string for slice coordinate and remaining args
            arg_string_list = ["%s=%g" % (skey,b)]
            arg_string_list += [str(k) for k in args[1:]]
            # Joint list with commas
            arg_string = "(" + (",".join(arg_string_list)) + ")"
            # Loop through coefficients
            for coeff in coeffs:
                # Status update
                txt = "Creating RBF for %s%s" % (coeff, arg_string)
                sys.stdout.write("%-72s\r" % txt[:72])
                sys.stdout.flush()
                # Append reference values to input tuple
                Z = V + (self[coeff][J],)
                # Create a single RBF
                f = scipy.interpolate.rbf.Rbf(*Z,
                    function=func, smooth=smooth)
                # Save it
                self.rbf[coeff].append(f)
        # Clean up the prompt
        sys.stdout.write("%72s\r" % "")
        sys.stdout.flush()
  # >

  # ============
  # Data
  # ============
  # <
   # --- Manual Addition ---
    # Add a field
    def AddCoeff(self, coeff, V, **kw):
        """Add a coefficient to the database, with checks as requested

        :Call:
            >>> DBc.AddCoeff(coeff, V, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of new coefficient
            *V*: :class:`np.ndarray`
                Array of values for this new coefficient
            *k*: {``None``} | :class:`str`
                Name of coefficient whose dimension must match *V*
        :Versions:
            * 2018-07-20 ``@ddalle``: First version
        """
        # Check for a key
        k = kw.get('k')
        # If given, match dimensions
        if k and (k in self.coeffs):
            # Get dimension of *V* and *self[k]*
            nk = self[k].size
            nV = len(V)
            # Check consistency
            if nk != nV:
                raise ValueError("Input array does not match dimension " +
                    ("of specified key '%s'" % k))
        # Set the data
        self[coeff] = np.asarray(V)
        # Add to coefficient list if needed
        if coeff not in self.coeffs:
            self.coeffs.append(coeff)

   # --- Independent Key Values ---
    # Get the value of an independent variable if possible
    def GetXValues(self, k, I=None, **kw):
        """Get values of specified coefficients, which may need conversion

        This function can be used to calculate independent variables that are
        derived from extant data columns.  For example if columns *alpha* and
        *beta* (for angle of attack and angle of sideslip, respectively) are
        present and the user wants to get the total angle of attack *aoap*,
        this function will attempt to use ``DBc.eval_arg_converters["aoap"]``
        to convert available *alpha* and *beta* data.

        :Call:
            >>> V = DBc.GetXValues(k, I=None, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *I*: ``None`` | :class:`np.ndarray` | :class:`int`
                Subset indices or index
            *kw*: :class:`dict`
                Dictionary of values to use instead of of columns of *DBc*
            *IndexKW*: ``True`` | {``False``}
                Option to use *kw[k][I]* instead of just *kw[k]*
        :Outputs:
            *V*: :class:`np.ndarray` | :class:`float`
                Array of values for variable *k* (scalar if *I* is scalar)
        :Versions:
            * 2019-03-12 ``@ddalle``: First version
        """
        # Option for processing keywrods
        qkw = kw.pop("IndexKW", False)
        # Check for direct membership
        if k in kw:
            # Get values from inputs
            V = kw[k]
            # Checking for quick output
            if not qkw: return V
        elif k in self:
            # Get all values from column
            V = self[k]
        else:
            # Get converter
            f = self.get_eval_arg_converter(k)
            # Check for converter
            if f is None:
                raise ValueError("No converter for key '%s'" % k)
            # Create a dictionary of values
            X = dict(self, **kw)
            # Attempt to convert
            try:
                # Use entire dictionary as inputs
                V = f(**X)
            except Exception:
                raise ValueError("Conversion function for '%s' failed" % k)
        # Check for indexing
        if I is None:
            # No subset
            return V
        elif typeutils.isarray(V):
            # Apply subset
            return V[I]
        else:
            # Not subsettable
            return V

    # Get independent variable from eval inputs
    def GetEvalXValues(self, k, *a, **kw):
        """Return values of a coefficient from inputs to :func:`__call__`

        For example, this can be used to derive the total angle of attack from
        inputs to an evaluation call to *CN* when it is a function of *mach*,
        *alpha*, and *beta*.  This function attempts to use
        :func:`DBc.eval_arg_converters`.

        :Call:
            >>> V = DBc.GetEvalXValues(k, *a, **kw)
            >>> V = DBc.GetEvalXValues(k, coeff, x1, x2, ..., k3=x3)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *k*: :class:`str`
                Name of key to calculate
            *coeff*: :class:`str`
                Name of coefficient
            *x1*: :class:`float` | :class:`np.ndarray`
                Value(s) of first argument
            *x2*: :class:`float` | :class:`np.ndarray`
                Value(s) of second argument, if applicable
            *k3*: :class:`str`
                Name of third argument or optional variant
            *x3*: :class:`float` | :class:`np.ndarray`
                Value(s) of argument *k3*, if applicable
        :Outputs:
            *V*: :class:`np.ndarray`
                Values of key *k* from conditions in *a* and *kw*
        :Versions:
            * 2019-03-12 ``@ddalle``: First version
        """
        # Process coefficient
        X = self.get_arg_value_dict(*a, **kw)
        # Check if key is present
        if k in X:
            # Return values
            return X[k]
        else:
            # Get dictionary of converters
            converters = getattr(self, "eval_arg_converters", {})
            # Check for membership
            if k not in converters:
                raise ValueError(
                    "Could not interpret independent variable '%s'" % k)
            # Get converter
            f = converters[k]
            # Check class
            if not hasattr(f, "__call__"):
                raise TypeError("Converter is not callable")
            # Attempt to convert
            try:
                # Use entire dictionary as inputs
                V = f(**X)
            except Exception:
                raise ValueError("Conversion function for '%s' failed" % k)
            # Output
            return V

   # --- Dependent Key Values ---
    # Get exact values
    def GetExactYValues(self, coeff, I=None, **kw):
        """Get exact values of a data coefficient

        :Call:
            >>> V = DBc.GetExactYValues(coeff, I=None, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of data coefficient to extract
        :Versions:
            * 2019-03-13 ``@ddalle``: First version
        """
        # Check for direct membership
        if coeff in self:
            # Get all values from column
            V = self[coeff]
            # Check for indexing
            if I is None:
                # No subset
                return V
            else:
                # Apply subset
                return V[I]
        else:
            # Get evaluation type
            meth = self.get_eval_method(coeff)
            # Only allow "function" type
            if meth != "function":
                raise ValueError(
                    ("Cannot evaluate exact values for '%s', " % coeff) +
                    ("which has method '%s'" % meth))
            # Get args
            args = self.get_eval_arg_list(coeff)
            # Create inputs
            a = tuple([self.GetXValues(k, I, **kw) for k in args])
            # Evaluate
            V = self.__call__(coeff, *a)
            # Output
            return V

   # --- Search ---
    # Find matches
    def FindMatches(self, args, *a, **kw):
        """Find cases that match a condition within a certain tolerance

        :Call:
            >>> I, J = DBc.FindMatches(args, *a, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *args*: :class:`list` (:class:`str`)
                List of argument names to match
            *a*: :class:`tuple` (:class:`float`)
                Values of the arguments
            *tol*: {``1e-4``} | :class:`float` >= 0
                Default tolerance for all *args*
            *tols*: {``{}``} | :class:`dict` (:class:`float` >= 0)
                Dictionary of tolerances specific to arguments
            *kw*: :class:`dict`
                Additional values to use during evaluation
        :Outputs:
            *I*: :class:`np.ndarray` (:class:`int`)
                Indices of cases in *DBc* that match a point in (*a*, *kw*)
            *J*: :class:`np.ndarray` (:class:`int`)
                Indices of entries in (*a*, *kw*) that have a match in *DBc*
        :Versions:
            * 2019-03-11 ``@ddalle``: First version
        """
       # --- Input Checks ---
        # Find a valid argument
        a0 = None
        for arg in args:
            # Check if it's present
            if arg in self:
                # Found at least one valid argument
                a0 = arg
                break
        # Check for error
        if a0 is None:
            raise ValueError(
                "Cannot find matches for argument list %s" % args)
        # Overall tolerance default
        tol = kw.pop("tol", 1e-4)
        # Specific tolerances
        tols = kw.pop("tols", {})
        # Number of values
        n = len(self[arg])
       # --- Argument values ---
        # Initialize lookup point
        x = []
        # Loop through arguments
        for i, k in enumerate(args):
            # Get value
            xi = self.get_arg_value(i, k, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        X, dims = self.__class__.NormalizeArguments(x, True)
        # Number of test points
        nx = np.prod(dims)
       # --- Checks ---
        # Initialize tests for database indices (set to ``False``)
        MI = np.arange(n) < 0
        # Initialize tests for input data indices (set to ``False``)
        MJ = np.arange(nx) < 0
        # Loop through entries
        for i in range(nx):
            # Initialize tests for this point (set to ``True``)
            Mi = np.arange(n) > -1
            # Loop through arguments
            for j, k in enumerate(args):
                # Get total set of values
                Xk = self.get_all_values(k)
                # Check if present
                if (k is None) or (Xk is None):
                    continue
                # Check size
                if (i == 0) and (len(Xk) != n):
                    raise ValueError(
                        ("Parameter '%s' has size %i, " % (k, len(Xk))) +
                        ("expecting %i" % n))
                # Get argument value
                xi = X[j][i]
                # Get tolerance for this key
                xtol = tols.get(k, tol)
                # Check tolerance
                Mi = np.logical_and(Mi, np.abs(Xk-xi) <= xtol)
            # Combine point constraints
            MI = np.logical_or(MI, Mi)
            # Check for any matches of this data point
            MJ[i] = np.any(Mi)
        # Convert masks to indices
        I = np.where(MI)[0]
        J = np.where(MJ)[0]
        # Return combined set of matches
        return I, J

    # Find matches
    def FindMatchesPair(self, DB2, args, args_test, *a, **kw):
        """Find cases that match conditions in two databases within tolerances

        :Call:
            >>> I1, I2, J = DBc.FindMatchesPair(DB2, args, args_test, *a, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *DB2*: :class:`tnakit.db.db1.DBCoeff`
                Second coefficient database interface
            *args*: :class:`list` (:class:`str`)
                List of argument names to match
            *args_test*: :class:`list` (:class:`str`)
                Additional arguments required for *DBc* and *DB2* match
            *a*: :class:`tuple` (:class:`float`)
                Values of the arguments
            *tol*: {``1e-4``} | :class:`float` >= 0
                Default tolerance for all *args*
            *tols*: {``{}``} | :class:`dict` (:class:`float` >= 0)
                Dictionary of tolerances specific to arguments
            *unique*: ``True`` | {``False``}
                Option to only allow indices to appear once in *I1* and *I2*
            *kw*: :class:`dict`
                Additional values to use during evaluation
        :Outputs:
            *I1*: :class:`np.ndarray` (:class:`int`)
                Indices of cases in *DBc* that match a point in (*a*, *kw*)
            *I2*: :class:`np.ndarray` (:class:`int`)
                Indices of cases in *DB2* that match a point in (*a*, *kw*)
            *J*: :class:`np.ndarray` (:class:`int`)
                Indices of entries in (*a*, *kw*) that have a match in *DBc*
        :Versions:
            * 2019-03-11 ``@ddalle``: First version
        """
       # --- Input Checks ---
        # Find a valid argument
        a0 = None
        for arg in args:
            # Check if it's present
            if arg in self:
                # Found at least one valid argument
                a0 = arg
                break
        # Check for error
        if a0 is None:
            raise ValueError(
                "Cannot find matches for argument list %s" % args)
        # Overall tolerance default
        tol = kw.pop("tol", 1e-4)
        # Specific tolerances
        tols = kw.pop("tols", {})
        # Uniqueness option
        unique = kw.pop("unique", False)
        # Number of values
        n1 = len(self[arg])
        n2 = len(DB2[arg])
       # --- Argument values ---
        # Initialize lookup point
        x = []
        # Loop through arguments
        for i, k in enumerate(args):
            # Get value
            xi = self.get_arg_value(i, k, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        X, dims = self.__class__.NormalizeArguments(x, True)
        # Number of test points
        nx = np.prod(dims)
       # --- Checks ---
        # Initialize tests for database indices (set to ``False``)
        I1 = []
        I2 = []
        # Initialize tests for input data indices (set to ``False``)
        M = np.arange(nx) < 0
        # Loop through entries
        for i in range(nx):
            # Initialize tests for this point (set to ``True``)
            M1 = np.arange(n1) > -1
            M2 = np.arange(n2) > -1
            # Loop through arguments
            for j, k in enumerate(args):
                # Get total set of values
                Xk1 = self.get_all_values(k)
                Xk2 = DB2.get_all_values(k)
                # Check if present
                if k is None:
                    # Null key, not sure why this would happen
                    continue
                elif (Xk1 is None) or (Xk2 is None):
                    # Failed to find in one or both databases
                    continue
                # Check size
                if (i == 0) and (len(Xk1) != n1):
                    raise ValueError(
                        ("Parameter '%s' has size %i, " % (k, len(Xk1))) +
                        ("expecting %i" % n1))
                elif (i == 0) and (len(Xk2) != n2):
                    raise ValueError(
                        ("Parameter '%s' has size %i, " % (k, len(Xk2))) +
                        ("expecting %i" % n2))
                # Get argument value
                xi = X[j][i]
                # Get tolerance for this key
                xtol = tols.get(k, tol)
                # Check tolerance
                M1 = np.logical_and(M1, np.abs(Xk1-xi) <= xtol)
                M2 = np.logical_and(M2, np.abs(Xk2-xi) <= xtol)
            # Convert to indices
            J1 = np.where(M1)[0]
            J2 = np.where(M2)[0]
            # Check empty
            if (J1.size == 0) or (J2.size == 0):
                continue
            # Loop through candidates from *DBc*
            for j1 in J1:
                # Initialize matches to *DB2*
                M3 = np.arange(J2.size) > -1
                # Loop through constraint arguments
                for k in args_test:
                    # Get value from *DBc* and *DB2*
                    xk1 = self.GetXValues(k, j1)
                    Xk2 = DB2.GetXValues(k, J2)
                    # Get tolerance for this key
                    xtol = tols.get(k, tol)
                    # Combine constraints
                    M3 = np.logical_and(M3, np.abs(Xk2-xk1) <= xtol)
                # Convert to indices
                I3 = np.where(M3)[0]
                # Number of matches
                m3 = I3.size
                # Check for matches
                if m3 == 0: continue
                # Save input data index: at least one pair match found
                M[i] = True
                # Save matches
                for j2 in J2[I3]:
                    # Save indices
                    I1.append(j1)
                    I2.append(j2)
                    # Check uniqueness flag
                    if unique:
                        # Remove *i2* from *M2*
                        M2[i2] = False
                        # Update *J2*
                        J2 = np.where(M2)[0]
                        break
        # Convert to arrays
        I1 = np.asarray(I1)
        I2 = np.asarray(I2)
        # Convert masks to indices
        J = np.where(M)[0]
        # Return combined set of matches
        return I1, I2, J

    # Find matches
    def FindMatchesPairIndex(self, DB2, args_test, I=None, **kw):
        """Find indices of cases that have matches in both databases

        :Call:
            >>> I1, I2, J = DBc.FindMatches(DB2, args_test, I=None, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *DB2*: :class:`tnakit.db.db1.DBCoeff`
                Second coefficient database interface
            *args_test*: :class:`list` (:class:`str`)
                Additional arguments required for *DBc* and *DB2* match
            *I*: {``None``} | :class:`np.ndarray` (:class:`int`)
                Indices of cases in *DBc* to consider
            *tol*: {``1e-4``} | :class:`float` >= 0
                Default tolerance for all *args*
            *tols*: {``{}``} | :class:`dict` (:class:`float` >= 0)
                Dictionary of tolerances specific to arguments
            *unique*: ``True`` | {``False``}
                Option to only allow indices to appear once in *I1* and *I2*
        :Outputs:
            *I1*: :class:`np.ndarray` (:class:`int`)
                Indices of cases in *DBc* from *I* that match a *DB2* case
            *I2*: :class:`np.ndarray` (:class:`int`)
                Indices of *I1* cases in *DB2*
            *J*: :class:`np.ndarray` (:class:`int`)
                Indices of *I1* entries in *I*
        :Versions:
            * 2019-03-21 ``@ddalle``: First version
        """
       # --- Input Checks ---
        # Check constraint count
        if not typeutils.isarray(args_test):
            # Not a list
            raise TypeError("List of test arguments must be list-like")
        elif len(args_test) == 0:
            # Empty list
            raise ValueError("At least one test argument must be given")
        # Pick test argument for sizing candidate arrays
        arg = args_test[0]
        # Overall tolerance default
        tol = kw.pop("tol", 1e-4)
        # Specific tolerances
        tols = kw.pop("tols", {})
        # Uniqueness option
        unique = kw.pop("unique", False)
        # Number of values
        n1 = len(self[arg])
        n2 = len(DB2[arg])
        # Default candidate list
        if I is None:
            I = np.arange(n1)
       # --- Checks ---
        # Initialize lists of finds
        I1 = []
        I2 = []
        # Initialize find indices
        J = []
        # Loop through entries
        for (j, i) in enumerate(I):
            # Initialize tests for this point (set to ``True``)
            M2 = np.arange(n2) > -1
            # Loop through arguments
            for k in args_test:
                # Don't test if not a data key
                if k not in self: continue
                # Get total set of values
                xk1 = self.GetXValues(k, i)
                Xk2 = DB2.get_all_values(k)
                # Check if present
                if k is None:
                    # Null key, not sure why this would happen
                    continue
                elif Xk2 is None:
                    # Failed to find in one or both databases
                    continue
                # Check size
                elif (i == 0) and (len(Xk2) != n2):
                    raise ValueError(
                        ("Parameter '%s' has size %i, " % (k, len(Xk2))) +
                        ("expecting %i" % n2))
                # Get tolerance for this key
                xtol = tols.get(k, tol)
                # Check tolerance
                M2 = np.logical_and(M2, np.abs(Xk2-xk1) <= xtol)
            # Convert to indices
            J2 = np.where(M2)[0]
            # Check empty
            if (J2.size == 0):
                continue
            # Loop through candidates from *DBc*
            for j2 in J2:
                # Save indices
                I1.append(i)
                I2.append(j2)
                # Save indices of indices
                J.append(j)
                # Check uniqueness flag
                if unique: break
        # Convert to arrays
        I1 = np.asarray(I1)
        I2 = np.asarray(I2)
        J = np.asarray(J)
        # Return combined set of matches
        return I1, I2, J
  # >

  # ===========
  # I/O
  # ===========
  # <
   # --- Spreadsheets ---
    # Read a spreadsheet
    def ReadXLS(self, fxls, sheet, prefix=None, **kw):
        """Read a simple table from one worksheet

        :Call:
            >>> DBc.ReadXLS(fxls, sheet, prefix=None, **kw)
            >>> DBc.ReadXLS(ws, sheet, prefix=None, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *fxls*: :class:`str`
                Name of worksheet to read
            *ws*: :class:`xlrd.book.Book`
                Open workbook/spreadsheet interface
            *sheet*: :class:`str`
                Name of worksheet to read
            *prefix*: {``None``} | :class:`str` | :class:`dict`
                Prefix to prepend to each coefficient name or dictionary of
                prefixes specific to column headers
            *suffix*: {``None``} | :class:`str` | :class:`dict`
                Prefix to append to each coefficient name or dictionary of
                prefixes specific to column headers
            *skiprows*: {``0``} | :class:`int` > 0
                Number of rows to skip before the table
            *skipcols*: {``0``} | :class:`int` > 0
                Number of columns to skip before the table
            *maxrows*: {``None``} | :class:`int` > 0
                Maximum row to read
            *maxcols*: {``None``} | :class:`int` > 0
                Maximum column number to read
            *translators*: {``{}``} | :class:`dict`
                Dictionary of coefficient translations, e.g. *CAF* -> *CA*
            *coeffs*: {``None``} | :class:`list` (:class:`str`)
                Optional list of coefficients (skip others)
            *subrows*: {``0``} | :class:`int` >= 0
                Number of rows to skip *after* header
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
            * 2018-07-20 ``@ddalle``: Added *coeffs*
            * 2018-07-24 ``@ddalle``: Added *subrows*
        """
        # Get suffix
        suffix = kw.get("suffix")
        # Check input types
        tw = fxls.__class__.__name__
        tp = prefix.__class__.__name__
        ts = suffix.__class__.__name__
        # Get workbook handle
        if tw == "Book":
            # Already a workbook
            wb = fxls
            # Save spreadsheet name
        else:
            # Open workbook
            wb = xlrd.open_workbook(fxls)
            # Save spreadsheet name
            self.fxls = fxls
        # Get the worksheet name
        ws = wb.sheet_by_name(sheet)
        # Row/column info
        skiprows = kw.get("skiprows")
        skipcols = kw.get("skipcols")
        maxrows = kw.get("maxrows")
        maxcols = kw.get("maxcols")
        subrows = kw.get("subrows", 0)
        # Dictionary of lookup translators
        translators = kw.get("translators", {})
        # List of coefficients
        coeff_list = kw.get("coeffs")
        # If no explicit row skip, try to guess
        if skiprows is None:
            # Read first row and column
            row0 = ws.row_values(0)
            col0 = ws.col_values(0)
            # Max dimensions
            nrow = len(col0)
            ncol = len(row0)
            # Initialize
            skiprows = 0
            # Loop through rows until we find a decent candidate
            for i in range(nrow):
                # Read the row
                rowi = ws.row_values(i)
                # Filter out empty cells
                V = [v for v in rowi if v != ""]
                # Check
                if len(V) > ncol/2:
                    # This is a good candidate for the header row
                    skiprows = i
                    break
        # If no explicit column skip, try to guess
        if skipcols is None:
            # Read header row
            rowi = ws.row_values(skiprows)
            # Initialize
            skipcols = 0
            # Loop through entries to find first non-empty entry
            for i in range(len(rowi)):
                # Check if empty
                if rowi[i] != "":
                    skipcols = i
                    break
        # Read the column headers
        cols = ws.row_values(skiprows, skipcols, end_colx=maxcols)
        # Initialize list of coefficients
        try:
            self.coeffs
        except AttributeError:
            self.coeffs = []
        # Loop through columns
        for i in range(len(cols)):
            # Get column name
            col = cols[i]
            # Translate if necessary
            col = translators.get(col, col)
            # Check if this is actually a column
            if not col.strip():
                continue
            # Check for prefix
            if tp == "dict":
                # Get column prefix from dictionary
                pre = prefix.get(col, prefix.get("def", ""))
            elif prefix:
                # Use single prefix
                pre = prefix
            else:
                # No prefix
                pre = ""
            # Check for suffix
            if ts == "dict":
                # Get column suffix from dictionary
                suf = suffix.get(col, suffix.get("def", ""))
            elif suffix:
                # Use single suffix
                suf = suffix
            else:
                # No suffix
                suf = ""
            # Total name of coefficient
            coeff = pre + col + suf
            # Check if not in pre-specified list
            if coeff_list and (coeff not in coeff_list):
                continue
            # Check for duplication
            if coeff in self.coeffs:
                raise ValueError(
                    "Coefficient '%s' is already present." % coeff)
            # Read the column
            V = ws.col_values(skipcols+i, skiprows+1+subrows,
                end_rowx=maxrows)
            # Ensure proper interpretation
            self[coeff] = np.array([v for v in V if v != ""])
            # Add to coefficient list
            self.coeffs.append(coeff)

   # --- CSV Files ---
    # Read a CSV file
    def ReadCSV(self, fcsv, prefix=None, delim=",", **kw):
        """Read comma-separated value file

        :Call:
            >>> DBc.ReadCSV(fcsv, prefix=None, delim=",", **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *fcsv*: :class:`str`
                Name of ASCII data file to read
            *prefix*: {``None``} | :class:`str` | :class:`dict`
                Prefix to prepend to each coefficient name or dictionary of
                prefixes specific to column headers
            *suffix*: {``None``} | :class:`str` | :class:`dict`
                Prefix to append to each coefficient name or dictionary of
                prefixes specific to column headers
            *delim*: {``","``} | :class:`str`
                Delimiter between data entries in each row
            *nskip*: {``None``} | :class:`int` >= 0
                Number of rows to skip
            *translators*: {``{}``} | :class:`dict`
                Dictionary of coefficient translations, e.g. *CAF* -> *CA*
            *txt*: ``True`` | {``False``}
                Whether or not to allow text values for coefficients
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
            * 2018-07-20 ``@ddalle``: Accepting text
        """
        # Get suffix
        suffix = kw.get("suffix")
        # Dictionary of translators
        translators = kw.get("translators", {})
        # Open the file
        f = open(fcsv, 'r')
        # Comment character
        comments = kw.get("comments", '#')
        # Row skip
        nskip = kw.get("nskip")
        if nskip is None:
            nskip = 0
        # Read first line
        line1 = f.readline().lstrip()
        line0 = line1
        # Read until line that does not start with column
        while line1.startswith(comments):
            # Transfer line
            line0 = line1
            # Skip a line
            nskip += 1
            # Read next line
            line1 = f.readline().lstrip()
        # Make sure we are skipping at least one line
        if nskip == 0: nskip = 1
        # Strip comments
        line = line0.lstrip(comments).lstrip()
        # Coefficients
        cols = [col.strip() for col in line.split(delim)]
        # Initialize list of coefficients
        try:
            self.coeffs
        except AttributeError:
            self.coeffs = []
        # Close the file
        f.close()
        # Loop through columns
        for (i, col) in enumerate(cols):
            # Apply translators
            col = translators.get(col, col)
            # Check for prefix
            if isinstance(prefix, dict):
                # Get column prefix from dictionary
                pre = prefix.get(col, prefix.get("def", ""))
            elif prefix:
                # Use single prefix
                pre = prefix
            else:
                # No prefix
                pre = ""
            # Check for suffix
            if isinstance(suffix, dict):
                # Get column suffix from dictionary
                suf = suffix.get(col, suffix.get("def", ""))
            elif suffix:
                # Use single suffix
                suf = suffix
            else:
                # No suffix
                suf = ""
            # Total name of coefficient
            coeff = pre + col + suf
            # Check for duplication
            if coeff in self.coeffs:
                raise ValueError(
                    "Coefficient '%s' is already present." % coeff)
            # Read the column
            try:
                # Read as float
                V = np.loadtxt(fcsv, delimiter=delim,
                    skiprows=nskip, comments=comments, usecols=(i,))
            except ValueError:
                # Check for text
                if kw.get("txt", False):
                    # Read as text
                    V = np.loadtxt(fcsv, delimiter=delim, dtype="str",
                        skiprows=nskip, comments=comments, usecols=(i,))
                else:
                    # Just skip
                    if kw.get("v", False):
                        print("  Skipping column '%s'" % col)
                    continue
            # Save the data
            self[coeff] = V
            # Add to coefficient list
            self.coeffs.append(coeff)

    # Write a CSV file
    def WriteCSV(self, fcsv, coeffs=None, fmt=None, **kw):
        """Write a comma-separated file of some of the coefficients

        :Call:
            >>> DBc.WriteCSV(fcsv, coeffs=None, fmt=None, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *fcsv*: :class:`str`
                Name of ASCII data file to write
            *coeffs*: {``None``} | :class:`list` (:class:`str`)
                List of coefficients to write, or write all coefficients
            *fmt*: {``None``} | :class:`str`
                Format string to be used for each row (optional)
            *fmts*: :class:`dict` | :class:`str`
                Dictionary of formats to use for each *coeff*
            *comments*: {``"#"``} | :class:`str`
                Comment character, used as first character of file
            *delim*: {``", "``} | :class:`str`
                Delimiter
            *translators*: {``{}``} | :class:`dict`
                Dictionary of coefficient translations, e.g. *CAF* -> *CA*
        :Versions:
            * 2018-06-11 ``@ddalle``: First versions
        """
        # Process coefficient list
        if coeffs is None:
            coeffs = list(self.coeffs)
        # Check for presence
        for coeff in coeffs:
            if coeff not in self:
                raise KeyError("No output coefficient '%s'" % coeff)
        # Get the count of the first key
        n = len(self[coeffs[0]])
        # Loop through the keys
        for i in range(len(coeffs)-1, 0, -1):
            # Coefficient
            coeff = coeffs[i]
            # Check length
            if len(self[coeff]) != n:
                # Print a warning
                sys.stderr.write("WARNING: skipping ")
                sys.stderr.write("coefficient '%s' " % coeff)
                sys.stderr.write("with mismatching length\n")
                sys.stderr.flush()
                # Delete it
                del coeffs[i]

        # Dictionary of translators
        translators = kw.get("translators", {})
        # Get comment character and delimiter
        cchar = kw.get("comments", "#")
        delim = kw.get("delim", ", ")

        # Default line format
        if fmt is None:
            # Set up printing format
            fmts = kw.get("fmts", {})
            # Options for default print flag
            prec = kw.get("prec", kw.get("precision", 6))
            emax = kw.get("emax", 4)
            emin = kw.get("emin", -2)
            echr = kw.get("echar", "e")
            # Specific
            precs = kw.get("precs", kw.get("precisions", {}))
            emaxs = kw.get("emaxs", {})
            emins = kw.get("emins", {})
            # Initialize final format
            fmt_list = []
            # Loop through keys to create default format
            for coeff in coeffs:
                # Options
                kwf = {
                    "prec": precs.get(coeff, prec),
                    "emax": emaxs.get(coeff, emax),
                    "emin": emins.get(coeff, emin),
                    "echar": echr,
                }
                # Make a default *fmt* for this coefficient
                fmti = arrayutils.get_printf_fmt(self[coeff], **kwf)
                # Get format, using above default
                fmti = fmts.get(coeff, fmti)
                # Save to list
                fmt_list.append(fmti)
            # Just use the delimiter
            fmt = delim.join(fmt_list)
        # Apply translators to the headers
        cols = [translators.get(coeff, coeff) for coeff in coeffs]

        # Create the file
        f = open(fcsv, 'w')
        # Write header
        f.write("%s " % cchar)
        f.write(delim.join(cols))
        f.write("\n")
        # Loop through entries
        for i in range(n):
            # Get values of coefficients
            V = tuple(self[coeff][i] for coeff in coeffs)
            # Use the format string
            f.write(fmt % V)
            # Newline
            f.write("\n")

        # Close the file
        f.close()

   # --- RBF CSV files ---
    # Write a CSV file for radial basis functions
    def WriteRBFCSV(self, fcsv, coeffs, **kw):
        """Write an ASCII file of radial basis func coefficients

        :Call:
            >>> DBc.WriteRBFCSV(fcsv, coeffs=None, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *fcsv*: :class:`str`
                Name of ASCII data file to write
            *coeffs*: :class:`list`\ [:class:`str`]
                List of output coefficients to write
            *fmts*: :class:`dict` | :class:`str`
                Dictionary of formats to use for each *coeff*
            *comments*: {``"#"``} | :class:`str`
                Comment character, used as first character of file
            *delim*: {``", "``} | :class:`str`
                Delimiter
            *translators*: {``{}``} | :class:`dict`
                Dictionary of coefficient translations, e.g.
                *CAF* -> *CA*
        :Versions:
            * 2019-07-24 ``@ddalle``: First versions
        """
        # Check input type
        if not isinstance(coeffs, (list, tuple)):
            raise TypeError("Coefficient list must be a list of strings")
        elif len(coeffs) < 1:
            raise ValueError("Coefficient list must have at least one entry")
        # Reference coefficient
        k = coeffs[0]
        # Check if present
        if k not in self:
            raise KeyError("No coeff '%s' in database" % k)
        # Get length
        n = len(self[k])
        # Check for this list of evaluation args to write
        try:
            # Get arg list for first *coeff*
            eval_args = self.eval_args[k]
        except AttributeError:
            # No definitions at all
            raise AttributeError(
                "No 'eval_args' have been set for this database")
        except KeyError:
            # No eval args for this coef
            raise KeyError("No 'eval_args' for coeff '%s'" % k)
        # Get evaluation type
        try:
            # Get type
            eval_meth = self.eval_method[k]
        except AttributeError:
            # No definitions at all
            raise AttributeError(
                "No 'eval_method' has been set for this database")
        except KeyError:
            # No eval method for reference coeff
            raise KeyError("No 'eval_method' for coeff '%s'" % k)
        # Check for RBFs attribute
        try:
            self.rbf[k]
        except AttributeError:
            # No definitions at all
            raise AttributeError("No RBFs defined for this database")
        except KeyError:
            # No RBFs saved for ref coeff
            raise KeyError("No RBFs for coeff '%s'" % k)
        # Check if the method is in our accepted list
        if eval_meth not in RBF_METHODS:
            raise ValueError(
                "Eval method '%s' is not consistent with RBFs" % eval_meth)
        # Loop through the keys
        for k in coeffs[1:]:
            # Check presence of required information
            if k not in self.coeffs:
                raise KeyError("No coeff '%s' in database" % k)
            elif k not in self.eval_args:
                raise KeyError("No 'eval_args' for coeff '%s'" % k)
            elif k not in self.eval_method:
                raise KeyError("No 'eval_method' for coeff '%s'" % k)
            elif k not in self.rbf:
                raise KeyError("No RBFs for coeff '%s'" % k)
            # Check length
            if len(self[k]) != n:
                raise ValueError(
                    "Coeff '%s' has length %i; expected %i"
                    % (k, len(self[k]), n))
            # Check values
            if self.eval_args[k] != eval_args:
                raise ValueError("Mismatching eval_args for coeff '%s'" % k)
            elif self.eval_method[k] != eval_meth:
                raise ValueError("Mismatching eval_method for coeff '%s" % k)

        # Number of arguments
        narg = len(eval_args)
        # Get method index
        imeth = RBF_METHODS.index(eval_meth)

        # Dictionary of translators
        translators = kw.get("translators", {})
        # Get comment character and delimiter
        cchar = kw.get("comments", "#")
        delim = kw.get("delim", ", ")

        # Set up printing format
        fmts = kw.get("fmts", {})
        # Options for default print flag
        prec = kw.get("prec", kw.get("precision", 6))
        emax = kw.get("emax", 4)
        emin = kw.get("emin", -2)
        echr = kw.get("echar", "e")
        # Specific
        precs = kw.get("precs", kw.get("precisions", {}))
        emaxs = kw.get("emaxs", {})
        emins = kw.get("emins", {})

        # Initialize final format
        fmt_list = []
        # Initialize final column list
        cols = []
        # Loop through independent variables
        for k in eval_args:
            # Check for alias
            col = translators.get(k, k)
            # Save column name
            cols.append(col)
            # Format options
            kwf = {
                "prec": precs.get(col, prec),
                "emax": emaxs.get(col, emax),
                "emin": emins.get(col, emin),
                "echar": echr,
            }
            # Make a default *fmt* for this coefficient
            fmti = arrayutils.get_printf_fmt(self[col], **kwf)
            # Get format, using above default
            fmti = fmts.get(col, fmti)
            # Save to list
            fmt_list.append(fmti)

        # Append type
        for k in ["eval_method"]:
            # Check for alias
            col = translators.get(k, k)
            # Save column name
            cols.append(col)
            # Format options
            kwf = {
                "prec": precs.get(col, prec),
                "emax": emaxs.get(col, emax),
                "emin": emins.get(col, emin),
                "echar": echr,
            }
            # Create dummy array
            x = np.array([imeth])
            # Make a default *fmt* for this coefficient
            fmti = arrayutils.get_printf_fmt(x, **kwf)
            # Get format, using above default
            fmti = fmts.get(col, fmti)
            # Save to list
            fmt_list.append(fmti)

        # Extra columns for each output coeff
        COEFF_COLS = ["", "rbf", "func", "eps", "smooth"]
        # Loop through coefficients
        for k in coeffs:
            # Get RBF
            if imeth == 0:
                # Get global rbf
                rbf = self.rbf[k]
                # No RBF list
                nrbf = 1
            else:
                # Get first mapped/linear rbf slice+
                rbfs = self.rbf[k]
                rbf = self.rbf[k][0]
                # Number of RBF slices
                nrbf = len(rbfs)
            # Loop through suffixes
            for suf in COEFF_COLS:
                # Append suffix
                col = ("%s_%s" % (k, suf)).rstrip("_")
                # Check for translation
                col = translators.get(col, col)
                # Add actual value at that point
                cols.append(col)
                # Get format, using above default
                fmti = fmts.get(col, "%%13.6%s" % echr)
                # Save to list
                fmt_list.append(fmti)

        # Just use the delimiter
        fmt = delim.join(fmt_list)
        # Number of columns
        ncol = len(cols)

        # Create the file
        f = open(fcsv, 'w')
        # Write header
        f.write("%s " % cchar)
        f.write(delim.join(cols))
        f.write("\n")
        # Loop through RBF slices
        for j in range(nrbf):
            # Reference coefficient
            k = coeffs[0]
            # Get RBF handle
            if imeth == 0:
                # Global RBF
                rbf = self.rbf[k]
            else:
                # Slice RBF
                rbf = self.rbf[k][j]
            # Number of points in slice
            nj = rbf.nodes.size
            # Initialize data array
            V = np.zeros((nj, ncol))
            # Save the coefficients
            if imeth == 0:
                # Save all the position variables
                V[:,:narg] = rbf.xi
            else:
                # Save the slice points
                V[:,0] = self.bkpts[eval_args[0]][j]
                # Save the other eval_arg values
                V[:,1:narg] = rbf.xi.T
            # Save the RBF type
            V[:,narg] = imeth
            # Loop through coefficients
            for (i, k) in enumerate(coeffs):
                # Get the RBF
                if imeth == 0:
                    # Global RBF
                    rbf = self.rbf[k]
                else:
                    # Slice RBF
                    rbf = self.rbf[k][j]
                # Save values
                V[:, narg + i*5 + 1] = np.asarray(rbf.di)
                V[:, narg + i*5 + 2] = np.asarray(rbf.nodes)
                V[:, narg + i*5 + 3] = RBF_FUNCS.index(rbf.function)
                V[:, narg + i*5 + 4] = rbf.epsilon
                V[:, narg + i*5 + 5] = rbf.smooth
            # Write values
            for i in range(nj):
                # Use the format string
                f.write(fmt % tuple(V[i]))
                # Newline
                f.write("\n")

        # Close the file
        f.close()

    # Read RBF function
    def ReadRBFCSV(self, fcsv, prefix=None, delim=",", **kw):
        """Read ASCII file of radial basis function coefficients

        :Call:
            >>> DBc.ReadRBFCSV(fcsv, prefix=None, delimiter=",", **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *fcsv*: :class:`str`
                Name of ASCII data file to read
            *prefix*: {``None``} | :class:`str` | :class:`dict`
                Prefix to prepend to each coefficient name or dictionary of
                prefixes specific to column headers
            *suffix*: {``None``} | :class:`str` | :class:`dict`
                Prefix to append to each coefficient name or dictionary of
                prefixes specific to column headers
            *delim*: {``","``} | :class:`str`
                Delimiter between data entries in each row
            *nskip*: {``None``} | :class:`int` >= 0
                Number of rows to skip
            *translators*: {``{}``} | :class:`dict`
                Dictionary of coefficient translations, e.g. *CAF* -> *CA*
        :Versions:
            * 2019-07-24 ``@ddalle``: First version
        """
       # --- Options and Column Names ---
        # Get suffix
        suffix = kw.get("suffix")
        # Dictionary of translators
        translators = kw.get("translators", {})
        # Open the file
        f = open(fcsv, 'r')
        # Comment character
        comments = kw.get("comments", '#')
        # Row skip
        nskip = kw.get("nskip")
        if nskip is None:
            nskip = 0
        # Read first line
        line1 = f.readline().lstrip()
        line0 = line1
        # Read until line that does not start with column
        while line1.startswith(comments):
            # Transfer line
            line0 = line1
            # Skip a line
            nskip += 1
            # Read next line
            line1 = f.readline().lstrip()
        # Make sure we are skipping at least one line
        if nskip == 0:
            nskip = 1
        # Strip comments
        line = line0.lstrip(comments).lstrip()
        # Coefficients
        cols = [col.strip() for col in line.split(delim)]
        # Initialize list of coefficients
        try:
            self.coeffs
        except AttributeError:
            self.coeffs = []
        # Close the file
        f.close()
        # Check for column called "eval_method"
        if "eval_method" not in cols:
            raise ValueError(
                "File '%s' does not have column called 'eval_method'" % fcsv)
       # --- Column Types ---
        # Number of args
        narg = cols.index("eval_method")
        # Read that column
        imeth = np.loadtxt(fcsv, delimiter=delim, skiprows=nskip,
            comments=comments, usecols=(narg,))
        # Check for valid method
        if np.any(imeth != imeth[0]):
            raise ValueError("Column 'eval_method' must have uniform values")
        # Get first entry
        imeth = int(imeth[0])
        # Check validity
        if (imeth < 0) or (imeth >= len(RBF_METHODS)):
            raise ValueError("Invalid 'eval_method' %s" % imeth)
        # Store first columns as evaluation args
        arg_cols = cols[:narg]
        # Store names of output coefficients
        coeff_cols = cols[narg+1::5]
        # Initialize prefixed/suffixed names
        args = []
        coeffs = []
        # Read the whole file as float
        V = np.loadtxt(
            fcsv, delimiter=delim,
            skiprows=nskip, comments=comments, ndmin=2)

       # --- Save values ---
        # Loop through arg columns
        for (j, col) in enumerate(arg_cols + coeff_cols):
            # True column number
            if j < narg:
                # First *narg* columns are lookup variables
                i = j
            else:
                # After *eval_method*,
                i = narg + 1 + (j-narg)*5
            # Apply translators
            col = translators.get(col, col)
            # Check for prefix
            if isinstance(prefix, dict):
                # Get column prefix from dictionary
                pre = prefix.get(col, prefix.get("def", ""))
            elif prefix:
                # Use single prefix
                pre = prefix
            else:
                # No prefix
                pre = ""
            # Check for suffix
            if isinstance(suffix, dict):
                # Get column suffix from dictionary
                suf = suffix.get(col, suffix.get("def", ""))
            elif suffix:
                # Use single suffix
                suf = suffix
            else:
                # No suffix
                suf = ""
            # Total name of coefficient
            coeff = pre + col + suf
            # Save name
            if j < narg:
                # Save independent variable name
                args.append(coeff)
            else:
                # Save dependent variable name
                coeffs.append(coeff)
            # Check for duplication
            if coeff in self.coeffs:
                raise ValueError(
                    "Coefficient '%s' is already present." % coeff)
            # Save the data
            self.AddCoeff(coeff, V[:,i])
       # --- Create RBFs ---
        # Create evaluation methods if needed
        try:
            self.eval_method
        except AttributeError:
            self.eval_method = {}
        # Create evaluation args if needed
        try:
            self.eval_args
        except AttributeError:
            self.eval_args = {}
        # Create rbf handle if needed
        try:
            self.rbf
        except AttributeError:
            self.rbf = {}
        # Dereference evaluation method list
        eval_meth = RBF_METHODS[imeth]
        # Get slice locations if appropriate
        if eval_meth == "rbf":
            # Create dummy inputs for blank RBF
            Z = ([0.0, 1.0],) * narg
        else:
            # Create dummy inputs for blank RBF
            Z = ([0.0, 1.0],) * (narg - 1)
            # Get unique values of first *arg*
            xs = np.unique(V[:, 0])


        # # Loop through coefficients
        for j, coeff in enumerate(coeffs):
            # Save evaluation arguments
            self.eval_args[coeff] = args
            # Save evaluation method
            self.eval_method[coeff] = eval_meth
            # Base column number
            j0 = narg + 1 + j*5
            # Cumulative row index
            i0 = 0
            # Check for slices
            if eval_meth == "rbf":
                # Global RBF
                # Read function type
                func = RBF_FUNCS[int(V[0, j0+2])]
                # Create RBF
                rbf = scipy.interpolate.rbf.Rbf(*Z, function=func)
                # Save not-too-important actual values
                rbf.di = V[:, j0]
                # Use all conditions
                rbf.xi = V[:,:narg].T
                # Get RBF weights
                rbf.nodes = V[:, j0+1]
                # Get function type
                rbf.function = func
                # Get scale factor
                rbf.epsilon = V[0, j0+3]
                # Smoothing parameter
                rbf.smooth = V[0, j0+4]
                # Save count
                rbf.N = V.shape[0]
                # Save RBF
                self.rbf[coeff] = rbf
                # Break points for each independent variable
                self.GetBreakPoints(args)
            else:
                # Initialize RBFs
                self.rbf[coeff] = []
                # Slice RBFs
                for i, xi in enumerate(xs):
                    # Get function type
                    func = RBF_FUNCS[int(V[i0, j0+2])]
                    # Initialize slice RBF
                    rbf = scipy.interpolate.rbf.Rbf(*Z, function=func)
                    # Number of points in this slice
                    ni = np.count_nonzero(V[i0:,0] == xi)
                    # End of range
                    i1 = i0 + ni
                    # Use subset of conditions
                    rbf.xi = V[i0:i1, 1:narg].T
                    # Save not-too-important original values
                    rbf.di = V[i0:i1, j0]
                    # Get RBF weights
                    rbf.nodes = V[i0:i1, j0+1]
                    # Get function type
                    rbf.function = func
                    # Get scale factor
                    rbf.epsilon = V[i0, j0+3]
                    # Smoothing parameter
                    rbf.smooth = V[i0, j0+4]
                    # Save count
                    rbf.N = ni
                    # Save RBF
                    self.rbf[coeff].append(rbf)
                    # Update indices
                    i0 = i1
                # Break points for mapping variable
                self.GetBreakPoints(args[0:1])
                # Get break points at each slice
                self.ScheduleBreakPoints(args[1:], args[0])


   # --- Matlab Files ---
    # Read a Matlab file
    def ReadMat(self, fmat, **kw):
        """Read a Matlab ``.mat`` file to import data

        :Call:
            >>> DBc.ReadMat(fmat, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *fmat*: :class:`str`
                Name of Matlab data file
        :Versions:
            * 2018-07-06 ``@ddalle``: First version
        """
        # Initialize coefficients
        try:
            # Check for existing
            self.coeffs
        except AttributeError:
            # Empty list
            self.coeffs = []

        # Read the MAT file using SciPy
        FM = sio.loadmat(fmat, struct_as_record=False, squeeze_me=True)

        # Get primary data interface
        DB = FM["DB"]
        # Loop through keys
        for k in DB._fieldnames:
            # Save data
            self[k] = getattr(DB, k)
            # Append coefficient name, if necessary
            if k not in self.coeffs:
                self.coeffs.append(k)

        # Initialize break points if needed
        try:
            # See if break points are present
            self.bkpts
        except Exception:
            # Empty dictionary
            self.bkpts = {}
        # Check for break points
        try:
            # Access the break points
            bkpts = FM["bkpts"]
            # Loop through break points
            for k in bkpts._fieldnames:
                # Save to present dictionary
                self.bkpts[k] = getattr(bkpts,k)
        except KeyError:
            # No break points
            pass

        # Save Matlab object
        self.mat = FM

    # Prepare a Matlab file output
    def prepmat_db1_DBCoeff(self, **kw):
        """Create a Matlab object for output

        :Call:
            >>> DBc.prepmat_db1_DBCoeff(**kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
        :Outputs:
            *FM*: :class:`dict` (:class:`mat_struct`)
                Partially constructed Matlab output object
        :Versions:
            * 2019-02-27 ``@ddalle``: First version
        """
        # Create a struct of variables to save
        FM = kw.get("FM", {})
        # Create struct for main data
        DB = siom.mat_struct()
        DB._fieldnames = []
        # Loop through keys
        for k in self.coeffs:
            # Save the attribute
            setattr(DB,k, self[k])
            # Append key
            DB._fieldnames.append(k)
        # Save to output dictionary
        FM["DB"] = DB
        # Check for break points
        try:
            # Call the instance to see if it exists
            self.bkpts
            # If so, create a struct
            bkpts = siom.mat_struct()
            # Loop through break point keys
            for k in self.bkpts:
                # Set the value to a copy of the break points
                setattr(bkpts,k, self.bkpts[k])
            # Save the key names: keys -> names
            bkpts._fieldnames = self.bkpts.keys()
            # Save to output dictionary
            FM["bkpts"] = bkpts
        except AttributeError:
            # That's fine; just don't write one
            pass
        # Output
        return FM

    # Prepare a Matlab file output
    def PrepareMat(self, **kw):
        """Create a Matlab object for output

        :Call:
            >>> FM = DBc.PrepareMat(**kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
        :Outputs:
            *FM*: :class:`dict` (:class:`mat_struct`)
                Partially constructed Matlab output object
        :Versions:
            * 2018-07-06 ``@ddalle``: First version
            * 2019-02-27 ``@ddalle``: Moved from :func:`WriteMat`
        """
        # Create a struct of variables to save
        FM = self.prepmat_db1_DBCoeff(**kw)
        # Output
        return FM

    # Write a generic Matlab file
    def WriteMat(self, fmat, **kw):
        """Write a generic Matlab ``.mat`` file as a struct

        :Call:
            >>> DBc.WriteMat(fmat, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *fmat*: :class:`str`
                Name of Matlab data file
            *FM*: {``{}``} | :class:`dict`
                Partially constructed output object; allows for the use of
                additional fields
        :Versions:
            * 2018-07-06 ``@ddalle``: First version
        """
        # Form the object for output
        FM = self.PrepareMat(**kw)
        # Output
        sio.savemat(fmat, FM, oned_as="column")
  # >

  # ===============
  # Interpolation
  # ===============
  # <
   # --- Break Points ---
    # Get automatic break points
    def GetBreakPoints(self, keys, nmin=5, tol=1e-12):
        """Create automatic list of break points for interpolation

        :Call:
            >>> DBc.GetBreakPoints(key, nmin=5, tol=1e-12)
            >>> DBc.GetBreakPoints(keys, nmin=5, tol=1e-12)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *key*: :class:`str` | :class:`unicode`
                Individual lookup variable
            *keys*: :class:`list` (:class:`str` | :class:`unicode`)
                List of lookup variables
            *nmin*: {``5``} | :class:`int` > 0
                Minimum number of data points at one value of a key
            *tol*: {``1e-12``} | :class:`float` >= 0
                Tolerance cutoff
        :Outputs:
            *DBc.bkpts*: :class:`dict`
                Dictionary of 1D unique lookup values
            *DBc.bkpts[key]*: :class:`np.ndarray` (:class:`float`)
                Unique values of *DBc[key]* with at least *nmin* entries
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
        """
        # Check for single key list
        if keys.__class__.__name__ not in ["list", "ndarray"]:
            # Make list
            keys = [keys]
        # Initialize break points
        try:
            self.bkpts
        except AttributeError:
            self.bkpts = {}
        # Loop through keys
        for k in keys:
            # Check if present
            if k not in self.coeffs:
                raise KeyError("Lookup key '%s' is not present" % k)
            # Get all values
            V = self[k]
            # Get unique values
            U = np.unique(V)
            # Initialize filtered value
            T = []
            # Loop through entries
            for v in U:
                # Check if too close to a previous entry
                if T and np.min(np.abs(v-T))<=tol:
                    continue
                # Count entries
                if np.sum(np.abs(V-v)<=tol) >= nmin:
                    # Save the value
                    T.append(v)
            # Save these break points
            self.bkpts[k] = np.array(T)

    # Map break points from other key
    def MapBreakPoints(self, keys, skey):
        """Create automatic list of break points for interpolation

        :Call:
            >>> DBc.MapBreakPoints(keys, skey)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *key*: :class:`str` | :class:`unicode`
                Individual lookup variable
            *skey*: :class:`str`
                Name of key to drive interpolation
        :Outputs:
            *DBc.bkpts*: :class:`dict`
                Dictionary of 1D unique lookup values
            *DBc.bkpts[key]*: :class:`np.ndarray` (:class:`float`)
                Unique values of *DBc[key]* with at least *nmin* entries
        :Versions:
            * 2018-06-29 ``@ddalle``: First version
        """
        # Loop through keys
        for k in keys:
            # Check if present
            if k not in self.coeffs:
                raise KeyError("Lookup key '%s' is not present" % k)
            # Initialize break points
            T = []
            # Loop through slice values
            for m in self.bkpts[skey]:
                # Find value of slice key matching that parameter
                i = np.where(self[skey] == m)[0][0]
                # Output value
                v = self[k][i]
                # Save break point
                T.append(v)
            # Save break points
            self.bkpts[k] = np.array(T)

    # Schedule break points at slices at other key
    def ScheduleBreakPoints(self, keys, skey, nmin=5, tol=1e-12):
        """Create automatic list of scheduled break points for interpolation

        :Call:
            >>> DBc.MapBreakPoints(keys, skey)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient database interface
            *key*: :class:`str` | :class:`unicode`
                Individual lookup variable
            *skey*: :class:`str`
                Name of key to drive interpolation
            *nmin*: {``5``} | :class:`int` > 0
                Minimum number of data points at one value of a key
            *tol*: {``1e-12``} | :class:`float` >= 0
                Tolerance cutoff
        :Outputs:
            *DBc.bkpts*: :class:`dict`
                Dictionary of 1D unique lookup values
            *DBc.bkpts[key]*: :class:`list` (:class:`np.ndarray`)
                Unique values of *DBc[key]* at each value of *skey*
        :Versions:
            * 2018-06-29 ``@ddalle``: First version
        """
        # Loop through keys
        for k in keys:
            # Check if present
            if k not in self.coeffs:
                raise KeyError("Lookup key '%s' is not present" % k)
            # Initialize scheduled break points
            X = []
            # Get all values for this key
            V = self[k]
            # Loop through slice values
            for m in self.bkpts[skey]:
                # Indices of points in this slice
                I = np.where(np.abs(self[skey] - m) <= tol)[0]
                # Check for broken break point
                if len(I) == 0:
                    raise ValueError("No points matching slice at " +
                        ("%s = %.2f" % (skey, m)))
                # First value
                i = I[0]
                # Get unique values on the slice
                U = np.unique(V[I])
                # Initialize filtered value
                T = []
                # Loop through entries
                for v in U:
                    # Check if too close to a previous entry
                    if T and np.min(np.abs(v-T))<=tol:
                        continue
                    # Count entries
                    if np.sum(np.abs(V-v)<=tol) >= nmin:
                        # Save the value
                        T.append(v)
                # Save break point
                X.append(np.asarray(T))
            # Save break points
            self.bkpts[k] = X


   # --- Key Lookup ---
    # Look up a generic key
    def get_key(self, k=None, defs=[], **kw):
        """Process a key name, using an ordered list of defaults

        :Call:
            >>> ka = DBc.get_key(k3=None)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBFM`
                Coefficient lookup database
            *k*: {``None``} | :class:`str`
                User-specified name of lookup key; if ``None``, automatic value
            *defs*: :class:`list`
                List of applicable default names for the key
            *title*: {``None``} | :class:`str`
                Key title to use in any error messages
            *error*: {``True``} | ``False``
                Whether or not to raise an exception if key is not found
        :Outputs:
            *ka*: *k* | *defs[0]* | *defs[1]* | ... | ``None``
                Name of lookup key in *DBc.coeffs*
        :Versions:
            * 2018-06-22 ``@ddalle``: First version
        """
        # Get default
        if (k is not None):
            # Check if it's present
            if k not in self.coeffs:
                # Error option
                if kw.get("error", True):
                    # Get title for error message, e.g. "angle of attack"
                    ttl = kw.get("title", "lookup")
                    # Error message
                    raise KeyError("No %s key found" % ttl)
                else:
                    # If no error, return ``None``
                    return defs[0]
            # Otherwise, done
            return k
        # Loop through defaults
        for ki in defs:
            # Check if it's present
            if ki in self.coeffs:
                return ki
        # If this point is reached, no default found
        if not kw.get("error", True):
            # No error, as specified by user
            return defs[0]
        # Get title for error message, e.g. "angle of attack"
        ttl = kw.get("title", "lookup")
        # Error message
        raise KeyError("No %s key found" % ttl)
  # >

  # =====
  # UQ
  # =====
  # <
   # --- Grouping ---
    # Get dictionary of test values
    def _get_test_values(self, arg_list, **kw):
        """Get test values for creating windows or comparing databases

        :Call:
            >>> vals, bkpts = DBc._get_test_values(arg_list, **kw)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                General coefficient database
            *arg_list*: :class:`list` (:class:`str`)
                List of arguments to use for windowing
        :Keyword Arguments:
            *test_values*: {*DBc*} | :class:`DBCoeff` | :class:`dict`
                Specify values of each parameter in *arg_list* that are the
                candidate points for the window; default is from *DBc*
            *test_bkpts*: {*DBc.bkpts*} | :class:`dict`
                Specify candidate window boundaries; must be ascending array of
                unique values for each key
        :Outputs:
            *vals*: :class:`dict` (:class:`np.ndarray`)
                Dictionary of lookup values for each key in *arg_list*
            *bkpts*: :class:`dict` (:class:`np.ndarray`)
                Dictionary of unique candidate values for each key
        :Versions:
            * 2019-02-13 ``@ddalle``: First version
        """
       # --- Lookup values ---
        # Values to use (default is direct from database)
        vals = {}
        # Dictionary from keyword args
        test_values = kw.get("test_values", {})
        # Loop through parameters
        for k in arg_list:
            # Get reference values for parameter *k*
            vals[k] = test_values.get(k, self[k])
       # --- Break points ---
        # Values to use for searching
        bkpts = {}
        # Primary default: *DBc.bkpts*
        self_bkpts = getattr(self, "bkpts", {})
        # Secondary default: *bkpts* attribute from secondary database
        test_bkpts = test_values.get("test_bkpts", {})
        # User-specified option
        test_bkpts = kw.get("test_bkpts", test_bkpts)
        # Primary default: *bkpts* from this database
        for k in arg_list:
            # Check for specified values
            if k in test_bkpts:
                # Use user-specified or secondary-db values
                bkpts[k] = test_bkpts[k]
            elif k in self_bkpts:
                # USe break points from this database
                bkpts[k] = self_bkpts[k]
            else:
                # Use unique test values
                bkpts[k] = np.unique(vals[k])
       # --- Output ---
        return vals, bkpts

    # Find *N* neighbors based on list of args
    def _get_uq_conditions(self, arg_list, **kw):
        """Get list of points at which to estimate UQ database

        :Call:
            >>> A = DBc.GetWindows_DB(arg_list, **kw)
            >>> [[a00, a01, ...], [a10, ...]] = DBc.GetWindows_DB(...)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                General coefficient database
            *arg_list*: :class:`list` (:class:`str`)
                List of arguments to use for windowing
        :Keyword Arguments:
            *test_values*: {*DBc*} | :class:`DBCoeff` | :class:`dict`
                Specify values of each parameter in *arg_list* that are the
                candidate points for the window; default is from *DBc*
            *test_bkpts*: {*DBc.bkpts*} | :class:`dict`
                Specify candidate window boundaries; must be ascending array of
                unique values for each key
        :Outputs:
            *A*: :class:`np.ndarray` (:class:`float`)
                List of conditions for each window
            *a00*: :class:`float`
                Value of *arg_list[0]* for first window
            *a01*: :class:`float`
                Value of *arg_list[1]* for first window
        :Versions:
            * 2019-02-16 ``@ddalle``: First version
        """
        # Get test values and test break points
        vals, bkpts = self._get_test_values(arg_list, **kw)
        # Number of args
        narg = len(arg_list)
        # Create tuple of all breakpoints
        bkpts1d = tuple(bkpts[k] for k in arg_list)
        # Create *n*-dimensional array of points (full factorial)
        vals_nd = np.meshgrid(*bkpts1d, indexing="ij")
        # Flatten each entry and make into a row vector
        V = tuple([v.flatten()] for v in vals_nd)
        # Combine into single vector
        A = np.vstack(V).T
        # Output
        return A

    # Find *N* neighbors based on list of args
    def GetWindow(self, n, arg_list, *a, **kw):
        """Get indices of neighboring

        :Call:
            >>> I = DBc.GetWindow(n, arg_list, *a, **kw)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                General coefficient database
            *n*: :class:`int`
                Minimum number of points in window
            *arg_list*: :class:`list` (:class:`str`)
                List of arguments to use for windowing
            *a[0]*: :class:`float`
                Value of the first argument
            *a[1]*: :class:`float`
                Value of the second argument
        :Keyword Arguments:
            *test_values*: {*DBc*} | :class:`DBCoeff` | :class:`dict`
                Specify values of each parameter in *arg_list* that are the
                candidate points for the window; default is from *DBc*
            *test_bkpts*: {*DBc.bkpts*} | :class:`dict`
                Specify candidate window boundaries; must be ascending array of
                unique values for each key
        :Outputs:
            *I*: :class:`np.ndarray`
                Indices of cases (relative to *test_values*) in window
        :Versions:
            * 2019-02-13 ``@ddalle``: First version
        """
       # --- Check bounds ---
        def check_bounds(vmin, vmax, vals):
            # Loop through parameters
            for i,k in enumerate(arg_list):
                # Get values
                Vk = vals[k]
                # Check bounds
                J = np.logical_and(Vk>=vmin[k], Vk<=vmax[k])
                # Combine constraints if i>0
                if i == 0:
                    # First run
                    I = J
                else:
                    # Combine constraints
                    I = np.logical_and(I, J)
            # Output
            return I
       # --- Init ---
        # Number of args
        narg = len(arg_list)
        # Check inputs
        if narg == 0:
            # No windowing arguments
            raise ValueError("At least one named argument required")
        elif len(a) != narg:
            # Not enough values
            raise ValueError("%i argument names provided but %i values"
                % (narg, len(a)))
        # Get the length of the database for the first argument
        nx = self[arg_list[0]].size
        # Initialize mask
        I = np.arange(nx) > -1
       # --- Lookup values ---
        # Get test values and test break points
        vals, bkpts = self._get_test_values(arg_list, **kw)
       # --- Tolerances/Bounds ---
        # Default tolerance
        tol = kw.get("tol", 1e-8)
        # Tolerance dictionary
        tols = {}
        # Initialize bounds
        vmin = {}
        vmax = {}
        # Loop through parameters
        for i,k in enumerate(arg_list):
            # Get value
            v = a[i]
            # Get tolerance
            tolk = kw.get("%stol"%k, tol)
            # Save it
            tols[k] = tolk
            # Bounds
            vmin[k] = v - tolk
            vmax[k] = v + tolk
       # --- Initial Check ---
        I = check_bounds(vmin, vmax, vals)
        # Initial window count
        m = np.count_nonzero(I)
       # --- Expansion ---
        # Maximum loop
        maxloops = kw.get("nmax", kw.get("maxloops", 10))
        # Loop until enough points are included
        for nloop in range(maxloops):
            # Check count
            if m >= n:
                break
            # Expand each argument in order
            for i,k in enumerate(arg_list):
                # Current bounds
                ak = vmin[k]
                bk = vmax[k]
                # Tolerance
                tolk = tols[k]
                # Break points
                Xk = bkpts[k]
                # Look for next point outside bound
                ja = Xk < ak - tolk
                jb = Xk > bk + tolk
                # Expand bounds if possible
                if np.any(ja): vmin[k] = np.max(Xk[ja])
                if np.any(jb): vmax[k] = np.min(Xk[jb])
                # Check new window
                I = check_bounds(vmin, vmax, vals)
                # Update count
                m = np.count_nonzero(I)
                # Check count
                if m >= n:
                    break
       # --- Output ---
        # Output
        return np.where(I)[0]




   # ---  Pairwise UQ ---
    # Single-point UQ estimate with indices determined
    def _estimate_UQ_point(self, FM2, coeff, ucoeff, I, **kw):
        """Quantify uncertainty interval for a single point or window

        :Call:
            >>> u, = FM1._estimateUQ_point(FM2, coeff, ucoeff, I, **kw)
            >>> U  = FM1._estimateUQ_point(FM2, coeff, ucoeff, I, **kw)
        :Inputs:
            *FM1*: :class:`DBCoeff`
                Original coefficient database
            *FM2*: :class:`DBCoeff`
                Comparison coefficient database
            *coeff*: :class:`str`
                Name of coefficient whose uncertainty is being estimated
            *ucoeff*: :class:`str`
                Name of uncertainty coefficient to estimate
        :Required Attributes:
            *FM1.eval_args[coeff]*: :class:`list` (:class:`str`)
                List of parameters of which *coeff* is a function
            *FM1.eval_args[ucoeff]*: :class:`list` (:class:`str`)
                List of parameters of which *ucoeff* is a function
            *FM1.uq_keys_extra*: {``{}``} | :class:`dict` (:class:`list`)
                List of any extra keys for each uncertainty coeff
            *FM1.uq_keys_shift*: {``{}``} | :class:`dict` (:class:`list`)
                List of keys whose deltas are used to shift *coeff* deltas
            *FM1.uq_funcs_extra*: {``{}``} | :class:`dict` (:class:`function`)
                Function to calculate any "extra" keys by name of key
            *FM1.uq_funcs_shift*: {``{}``} | :class:`dict` (:class:`function`)
                Function to use when "shifting" deltas in *coeff*
            *FM1.uq_keys_extra[ucoeff]*: :class:`list` (:class:`str`)
                List of extra coeffs for *ucoeff* {``[]``}
            *FM1.uq_keys_shift[ucoeff]*: :class:`list` (:class:`str`)
                List of coeffs to use while shifting *ucoeff* deltas  {``[]``}
            *FM1.uq_funcs_extra[ecoeff]*: :class:`function`
                Function to use to estimate extra key *ecoeff*
            *FM1.uq_funcs_shift[ucoeff]*: :class:`function`
                Function to use to shift/alter *ucoeff* deltas
        :Outputs:
            *u*: :class:`float`
                Single uncertainty estimate for generated window
            *U*: :class:`tuple` (:class:`float`)
                Value of *ucoeff* and any "extra" coefficients in
                ``FM1.uq_keys_extra[ucoeff]``
        :Versions:
            * 2019-02-15 ``@ddalle``: First version
        """
       # --- Inputs ---
        # Probability
        cov = kw.get("Coverage", kw.get("cov", 0.99865))
        cdf = kw.get("CoverageCDF", kw.get("cdf", cov))
        # Outlier cutoff
        osig_kw = kw.get('OutlierSigma', kw.get("osig"))
       # --- Attributes ---
        # Unpack useful attributes for additional functions
        uq_funcs_extra = getattr(self, "uq_funcs_extra", {})
        uq_funcs_shift = getattr(self, "uq_funcs_shift", {})
        # Additional information
        uq_keys_extra = getattr(self, "uq_keys_extra", {})
        uq_keys_shift = getattr(self, "uq_keys_shift", {})
        # Get eval arguments for input coeff and UQ coeff
        argsc = self.eval_args[coeff]
        argsu = self.eval_args[ucoeff]
       # --- Test Conditions ---
        # Get test values and test break points
        vals, bkpts = self._get_test_values(argsc, **kw)
        # --- Evaluation ---
        # Initialize input values for comparison evaluation
        A = tuple()
        # Get dictionary values
        for k in argsc:
            # Get indices
            A += (vals[k][I],)
        # Evaluate both databases
        V1 = self(coeff, *A)
        V2 = FM2(coeff, *A)
        # Deltas
        DV = V2 - V1
       # --- Shift coeffs ---
        # Get extra keys
        cokeys = uq_keys_shift.get(coeff, [])
        # Ensure list
        if cokeys.__class__.__name__ in ["str", "unicode"]:
            # Convert string to singleton
            cokeys = [cokeys]
        # Deltas of co-keys
        DV0_shift = tuple()
        # Loop through shift keys
        for k in cokeys:
            # Evaluate both databases
            V1 = self(k, *A)
            V2 = FM2(k, *A)
            # Append deltas
            DV0_shift += (V2-V1,)
       # --- Outliers ---
        # Degrees of freedom
        df = DV.size
        # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
        ksig = student.ppf(0.5+0.5*cdf, df)
        kcov = student.ppf(0.5+0.5*cov, df)
        # Outlier cutoff
        if osig_kw is None:
            # Default
            osig = 1.5*ksig
        else:
            # User-supplied value
            osig = osig_kw
        # Check outliers on main deltas
        J = stats.check_outliers(DV, cov, cdf=cdf, osig=osig)
        # Loop through shift keys; points must be non-outlier in all keys
        for DVk in DV0_shift:
            # Check outliers in these deltas
            Jk = stats.check_outliers(DV, cov, cdf=cdf, osig=osig)
            # Combine constraints
            J = np.logical_and(J, Jk)
        # Downselect original deltas
        DV = DV[J]
        # Initialize downselected correlated deltas
        DV_shift = tuple()
        # Downselect correlated deltas
        for DVk in DV0_shift:
            DV_shift += (DVk[J],)
        # New degrees of freedom
        df = DV.size
        # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
        ksig = student.ppf(0.5+0.5*cdf, df)
        kcov = student.ppf(0.5+0.5*cov, df)
        # Outlier cutoff
        if osig_kw is None:
            # Default
            osig = 1.5*ksig
        else:
            # User-supplied value
            osig = osig_kw
       # --- Extra coeffs ---
        # List of extra keys
        extrakeys = uq_keys_extra.get(ucoeff, [])
        # Ensure list
        if extrakeys.__class__.__name__ in ["str", "unicode"]:
            # Convert string to singleton
            extrakeys = [extrakeys]
        # Initialize tuple of extra key values
        a_extra = tuple()
        # Loop through extra keys
        for k in extrakeys:
            # Get function
            fn = uq_funcs_extra.get(k)
            # This function is required
            if fn is None:
                raise ValueError("No function for extra key '%s'" % k)
            # Evaluate
            a_extra += fn(self, DV, *DV_shift),
       # --- Delta shifting ---
        # Function to perform any shifts
        f_shift = uq_funcs_shift.get(ucoeff)
        # Perform shift
        if f_shift is not None:
            # Extra arguments for shift key
            a_shift = DV_shift + a_extra
            # Perform shift using appropriate function
            DV = f_shift(self, DV, *a_shift)
       # --- Statistics ---
        # Calculate coverage interval
        vmin, vmax = stats.get_cov_interval(DV, cov, cdf=cdf, osig=osig)
        # Max value
        u = max(abs(vmin), abs(vmax))
       # --- Output ---
        # Return all extra values
        return (u,) + a_extra

    # Single-point UQ estimate
    def EstimateUQ_point(self, FM2, coeff, ucoeff, *a, **kw):
        """Quantify uncertainty interval for a single point or window

        :Call:
            >>> u, = FM1.EstimateUQ_point(FM2, coeff, ucoeff, *a, **kw)
            >>> U  = FM1.EstimateUQ_point(FM2, coeff, ucoeff, *a, **kw)
        :Inputs:
            *FM1*: :class:`DBCoeff`
                Original coefficient database
            *FM2*: :class:`DBCoeff`
                Comparison coefficient database
            *coeff*: :class:`str`
                Name of coefficient whose uncertainty is being estimated
            *ucoeff*: :class:`str`
                Name of uncertainty coefficient
            *a*: :class:`tuple` (:class:`float`)
                Conditions at which to evaluate uncertainty
            *a[0]*: :class:`float`
                Value of *FM1.eval_args[ucoeff][0]*
        :Keyword Arguments:
            *nmin*: {``30``} | :class:`int` > 0
                Minimum number of points in window
            *cov*, *Coverage*: {``0.99865``} | 0 < :class:`float` < 1
                Fraction of data that must be covered by UQ term
            *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
                Coverage fraction assuming perfect distribution
            *test_values*: {``{}``} | :class:`dict` (:class:`np.ndarray`)
                Candidate values of each *eval_arg* for comparison
            *test_bkpts*: {``{}``} | :class:`dict` (:class:`np.ndarray`)
                Candidate break points (1D unique values) for *eval_args*
        :Required Attributes:
            *FM1.eval_args[coeff]*: :class:`list` (:class:`str`)
                List of parameters of which *coeff* is a function
            *FM1.eval_args[ucoeff]*: :class:`list` (:class:`str`)
                List of parameters of which *ucoeff* is a function
            *FM1.uq_keys_extra*: {``{}``} | :class:`dict` (:class:`list`)
                List of any extra keys for each uncertainty coeff
            *FM1.uq_keys_shift*: {``{}``} | :class:`dict` (:class:`list`)
                List of keys whose deltas are used to shift *coeff* deltas
            *FM1.uq_funcs_extra*: {``{}``} | :class:`dict` (:class:`function`)
                Function to calculate any "extra" keys by name of key
            *FM1.uq_funcs_shift*: {``{}``} | :class:`dict` (:class:`function`)
                Function to use when "shifting" deltas in *coeff*
            *FM1.uq_keys_extra[ucoeff]*: :class:`list` (:class:`str`)
                List of extra coeffs for *ucoeff* {``[]``}
            *FM1.uq_keys_shift[ucoeff]*: :class:`list` (:class:`str`)
                List of coeffs to use while shifting *ucoeff* deltas  {``[]``}
            *FM1.uq_funcs_extra[ecoeff]*: :class:`function`
                Function to use to estimate extra key *ecoeff*
            *FM1.uq_funcs_shift[ucoeff]*: :class:`function`
                Function to use to shift/alter *ucoeff* deltas
        :Outputs:
            *u*: :class:`float`
                Single uncertainty estimate for generated window
            *U*: :class:`tuple` (:class:`float`)
                Value of *ucoeff* and any "extra" coefficients in
                ``FM1.uq_keys_extra[ucoeff]``
        :Versions:
            * 2019-02-15 ``@ddalle``: First version
        """
       # --- Inputs ---
        # Get eval arguments for input coeff and UQ coeff
        argsu = self.eval_args[ucoeff]
        # Get minimum number of points in statistical window
        nmin = kw.get("nmin", 30)
       # --- Windowing ---
        # Get window
        I = self.GetWindow(nmin, argsu, *a, **kw)
       # --- Evaluation ---
        # Call stand-alone method
        U = self._estimate_UQ_point(FM2, coeff, ucoeff, I, **kw)
       # --- Output ---
        return U

    # Single-point UQ estimate
    def EstimateUQ_coeff(self, FM2, coeff, ucoeff, **kw):
        """Quantify uncertainty interval for all points of one UQ coefficient

        :Call:
            >>> A, U  = FM1.EstimateUQ_coeff(FM2, coeff, ucoeff, **kw)
        :Inputs:
            *FM1*: :class:`DBCoeff`
                Original coefficient database
            *FM2*: :class:`DBCoeff`
                Comparison coefficient database
            *coeff*: :class:`str`
                Name of coefficient whose uncertainty is being estimated
            *ucoeff*: :class:`str`
                Name of uncertainty coefficient
        :Keyword Arguments:
            *nmin*: {``30``} | :class:`int` > 0
                Minimum number of points in window
            *cov*, *Coverage*: {``0.99865``} | 0 < :class:`float` < 1
                Fraction of data that must be covered by UQ term
            *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
                Coverage fraction assuming perfect distribution
            *test_values*: {``{}``} | :class:`dict` (:class:`np.ndarray`)
                Candidate values of each *eval_arg* for comparison
            *test_bkpts*: {``{}``} | :class:`dict` (:class:`np.ndarray`)
                Candidate break points (1D unique values) for *eval_args*
        :Required Attributes:
            *FM1.eval_args[coeff]*: :class:`list` (:class:`str`)
                List of parameters of which *coeff* is a function
            *FM1.eval_args[ucoeff]*: :class:`list` (:class:`str`)
                List of parameters of which *ucoeff* is a function
            *FM1.uq_keys_extra*: {``{}``} | :class:`dict` (:class:`list`)
                List of any extra keys for each uncertainty coeff
            *FM1.uq_keys_shift*: {``{}``} | :class:`dict` (:class:`list`)
                List of keys whose deltas are used to shift *coeff* deltas
            *FM1.uq_funcs_extra*: {``{}``} | :class:`dict` (:class:`function`)
                Function to calculate any "extra" keys by name of key
            *FM1.uq_funcs_shift*: {``{}``} | :class:`dict` (:class:`function`)
                Function to use when "shifting" deltas in *coeff*
            *FM1.uq_keys_extra[ucoeff]*: :class:`list` (:class:`str`)
                List of extra coeffs for *ucoeff* {``[]``}
            *FM1.uq_keys_shift[ucoeff]*: :class:`list` (:class:`str`)
                List of coeffs to use while shifting *ucoeff* deltas  {``[]``}
            *FM1.uq_funcs_extra[ecoeff]*: :class:`function`
                Function to use to estimate extra key *ecoeff*
            *FM1.uq_funcs_shift[ucoeff]*: :class:`function`
                Function to use to shift/alter *ucoeff* deltas
        :Outputs:
            *A*: :class:`np.ndarray` (:class:`float`, size=(*nx*\ ,*na*\ ))
                List of conditions for each window, for *nx* windows, each
                with *na* values (length of *FM1.eval_args[ucoeff]*)
            *U*: :class:`np.ndarray` (:class:`float`, size=(*nx*\ ,*nu*\ +1)
                Values of *ucoeff* and any *nu* "extra" coefficients in
                ``FM1.uq_keys_extra[ucoeff]``
        :Versions:
            * 2019-02-15 ``@ddalle``: First version
        """
       # --- Inputs ---
        # Get eval arguments for input coeff and UQ coeff
        argsu = self.eval_args[ucoeff]
        # Get minimum number of points in statistical window
        nmin = kw.get("nmin", 30)
        # Additional information
        uq_keys_extra = getattr(self, "uq_keys_extra", {})
        # Additional keys
        keys_extra = uq_keys_extra.get(ucoeff, [])
        # Number of keys
        nu = 1 + len(keys_extra)
       # --- Windowing ---
        # Break up possible run matrix into window centers
        A = self._get_uq_conditions(argsu, **kw)
        # Number of windows
        nx = len(A)
        # Initialize output
        U = np.zeros((nx, nu))
       # --- Evaluation ---
        # Loop through conditions
        for (i,a) in enumerate(A):
            # Get window
            I = self.GetWindow(nmin, argsu, *a, **kw)
            # Estimate UQ for this window
            U[i] = self._estimate_UQ_point(FM2, coeff, ucoeff, I, **kw)
       # --- Output ---
        return A, U

    # Single-point UQ estimate
    def EstimateUQ_DB(self, FM2, **kw):
        """Quantify uncertainty for all coefficient pairings in a DB

        :Call:
            >>> FM1.EstimateUQ_DB(FM2, **kw)
        :Inputs:
            *FM1*: :class:`DBCoeff`
                Original coefficient database
            *FM2*: :class:`DBCoeff`
                Comparison coefficient database
        :Keyword Arguments:
            *nmin*: {``30``} | :class:`int` > 0
                Minimum number of points in window
            *cov*, *Coverage*: {``0.99865``} | 0 < :class:`float` < 1
                Fraction of data that must be covered by UQ term
            *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
                Coverage fraction assuming perfect distribution
            *test_values*: {``{}``} | :class:`dict` (:class:`np.ndarray`)
                Candidate values of each *eval_arg* for comparison
            *test_bkpts*: {``{}``} | :class:`dict` (:class:`np.ndarray`)
                Candidate break points (1D unique values) for *eval_args*
        :Required Attributes:
            *FM1.uq_coeffs*: :class:`dict` (:class:`list`)
                List of UQ coefficient names for each nominal coefficient
            *FM1.uq_coeffs[coeff]*: ``[ucoeff]``
                List of UQ coefficients for coefficient *coeff* {``[]``}
            *FM1.eval_args[coeff]*: :class:`list` (:class:`str`)
                List of parameters of which *coeff* is a function
            *FM1.eval_args[ucoeff]*: :class:`list` (:class:`str`)
                List of parameters of which *ucoeff* is a function
            *FM1.uq_keys_extra*: {``{}``} | :class:`dict` (:class:`list`)
                List of any extra keys for each uncertainty coeff
            *FM1.uq_keys_shift*: {``{}``} | :class:`dict` (:class:`list`)
                List of keys whose deltas are used to shift *coeff* deltas
            *FM1.uq_funcs_extra*: {``{}``} | :class:`dict` (:class:`function`)
                Function to calculate any "extra" keys by name of key
            *FM1.uq_funcs_shift*: {``{}``} | :class:`dict` (:class:`function`)
                Function to use when "shifting" deltas in *coeff*
            *FM1.uq_keys_extra[ucoeff]*: :class:`list` (:class:`str`)
                List of extra coeffs for *ucoeff* {``[]``}
            *FM1.uq_keys_shift[ucoeff]*: :class:`list` (:class:`str`)
                List of coeffs to use while shifting *ucoeff* deltas  {``[]``}
            *FM1.uq_funcs_extra[ecoeff]*: :class:`function`
                Function to use to estimate extra key *ecoeff*
            *FM1.uq_funcs_shift[ucoeff]*: :class:`function`
                Function to use to shift/alter *ucoeff* deltas
        :Versions:
            * 2019-02-15 ``@ddalle``: First version
        """
       # --- Inputs ---
        # Get minimum number of points in statistical window
        nmin = kw.get("nmin", 30)
        # Additional information
        uq_keys_extra = getattr(self, "uq_keys_extra", {})
        # List of UQ coefficients
        try:
            # Get the dictionary of coefficients
            uq_coeffs = self.uq_coeffs
        except AttributeError:
            # Required
            raise AttributeError(
                "Cannot process database UQ without 'uq_coeffs' attribute")
       # --- Coefficient Loop ---
        # Loop through data coefficients
        for coeff in uq_coeffs:
            # Get UQ coeff list
            ucoeffs = uq_coeffs[coeff]
            # Ensure list
            if type(ucoeffs) in [str, unicode]: ucoeffs = [ucoeffs]
            # Loop through them
            for ucoeff in ucoeffs:
                # Status update
                sys.stdout.write("%-60s\r" %
                    ("Estimating UQ: %s --> %s" % (coeff, ucoeff)))
                sys.stdout.flush()
                # Process "extra" keys
                keys_extra = uq_keys_extra.get(ucoeff, [])
                # Ensure list
                if type(keys_extra) in [str, unicode]:
                    keys_extra = [keys_extra]
                # Call particular method
                A, U = self.EstimateUQ_coeff(FM2, coeff, ucoeff, **kw)
                # Save primary key
                self[ucoeff] = U[:,0]
                # Save additional keys
                for (j,k) in enumerate(keys_extra):
                    # Save additional key values
                    self[k] = U[:,j+1]
        # Clean up prompt
        sys.stdout.write("%60s\r" % "")

  # >

  # =============
  # Increment
  # =============
  # <
   # --- Conditions and Slices ---
    # Generate slices for interpolation
    def _get_test_slices(self, arg_list, slice_args, **kw):
        """Get test values for creating windows or comparing databases

        :Call:
            >>> vals,bkpts,J = DBc._get_test_slices(arg_list,slice_args, **kw)
        :Inputs:
            *DBc*: :class:`DBCoeff`
                General coefficient database
            *arg_list*: :class:`list` (:class:`str`)
                List of arguments to use for windowing
        :Keyword Arguments:
            *test_values*: {*DBc*} | :class:`DBCoeff` | :class:`dict`
                Specify values of each parameter in *arg_list* that are the
                candidate points for the window; default is from *DBc*
            *test_bkpts*: {*DBc.bkpts*} | :class:`dict`
                Specify candidate window boundaries; must be ascending array of
                unique values for each key
            *tol*: {``1e-4``} | :class:`float` >= 0
                Default tolerance for all slice keys
            *tols*: {``{}``} | :class:`dict` (:class:`float` >= 0)
                Specific tolerance for particular slice keys
        :Outputs:
            *vals*: :class:`dict` (:class:`np.ndarray`)
                Dictionary of lookup values for each key in *arg_list*
            *bkpts*: :class:`dict` (:class:`np.ndarray`)
                Dictionary of unique candidate values for each key
            *J*: :class:`list` (:class:`np.ndarray` (:class:`int`))
                Indices of points (relative to *vals*) of points in each slice
        :Versions:
            * 2019-02-20 ``@ddalle``: First version
        """
       # --- Tolerances ---
        # Initialize tolerances
        tols = dict(kw.get("tols", {}))
        # Default global tolerance
        tol = kw.get("tol", 1e-4)
        # Loop through keys
        for k in slice_args:
            # Set default
            tols.setdefault(k, tol)
       # --- Test values ---
        # Get values
        vals, bkpts = self._get_test_values(arg_list, **kw)
        # Number of values
        nv = vals[arg_list[0]].size
       # --- Slice candidates ---
        # Create tuple of all breakpoint combinations for slice keys
        bkpts1d = tuple(bkpts[k] for k in slice_args)
        # Create *n*-dimensional array of points (full factorial)
        vals_nd = np.meshgrid(*bkpts1d, indexing="ij")
        # Create matrix of potential slice coordinates
        slice_vals = {}
        # Default number of candidate slices: 1 if no slice keys
        ns = 1
        # Loop through slice keys
        for (i,k) in enumerate(slice_args):
            # Save the flattened array
            slice_vals[k] = vals_nd[i].flatten()
            # Number of possible of values
            ns = slice_vals[k].size
       # --- Slice indices ---
        # Initialize slices
        J = []
        # Loop through potential slices
        for j in range(ns):
            # Initialize indices meeting all constraints
            M = np.arange(nv) > -1
            # Loop through slice arguments
            for k in slice_args:
                # Get key value
                v = slice_vals[k][j]
                # Get tolerance
                tolk = tols.get(k, tol)
                # Apply constraint
                M = np.logical_and(M, np.abs(vals[k] - v) <= tolk)
            # Convert *M* to indices
            I = np.where(M)[0]
            # Check for matches
            if I.size == 0: continue
            # Save slice
            J.append(I)
        # Output
        return vals, bkpts, J

   # --- Pairwise Deltas ---
    # Generate raw deltas
    def DiffDB(self, FM2, coeffs, skeys=[], **kw):
        """Create database of raw deltas between two databases

        :Call:
            >>> dFM = FM1.DiffDB(FM2, coeffs, skeys=[], **kw)
        :Inputs:
            *FM1*: :class:`DBCoeff` | :class:`DBFM`
                Original coefficient database
            *FM2*: :class:`DBCoeff` | :class:`DBFM`
                Comparison coefficient database
            *coeffs*: :class:`list` (:class:`str`)
                Coefficients to analyze
        :Keyword Arguments:
            *skeys*: {``[]``} | :class:`list` (:class:`str`)
                List of arguments to define slices on which to smooth
            *smooth*: {``0``} | :class:`float` >= 0
                Smoothing parameter for interpolation on slices
            *function*: {``"multiquadric"``} | :class:`str`
                RBF basis function type, see :func:`scipy.interpolate.rbf.Rbf`
            *test_values*: {``{}``} | :class:`dict` (:class:`np.ndarray`)
                Candidate values of each *eval_arg* for comparison
            *test_bkpts*: {``{}``} | :class:`dict` (:class:`np.ndarray`)
                Candidate break points (1D unique values) for *eval_args*
            *tol*: {``1e-4``} | :class:`float` > 0
                Default tolerance for matching slice constraints
            *tols*: {``{}``} | :class:`dict` (:class:`float` >= 0)
                Specific tolerance for particular slice keys
            *increment_slice_args*: {``None``} | :class:`dict` (:class:`list`)
                List of slice keys for particular coefficients (overrides attr)
            *increment_smooth*: {``None``} | :class:`dict` (:class:`float`)
                Separate smoothing for each parameter (overrides attribute)
            *increment_function*: {``None``} | :class:`dict` (:class:`str`)
                Separate RBF functions for each parameter (overrides attr)
            *global_slice_args*: {``None``} | :class:`list` (:class:`str`)
                Override list of arguments
        :Required Attributes:
            *FM1.increment_slice_args*: :class:`dict` (:class:`list`)
                List of slice keys for particular coefficients
            *FM1.increment_smooth*: :class:`dict` (:class:`float`)
                Separate smoothing for each parameter
            *FM1.increment_function*: :class:`dict` (:class:`str`)
                Separate RBF functions for each parameter
        :Outputs:
            *dFM*: *FM1.__class__*
                Copy of *FM1* with each coeff in *coeffs* replaced with delta
        :Versions:
            * 2019-02-20 ``@ddalle``: First version
        """
        # Create copy
        dFM = self.copy()
        # Increment parameters from attributes
        inc_slice_args = getattr(self, "increment_slice_args", {})
        inc_smooth     = getattr(self, "increment_smooth",     {})
        inc_funcs      = getattr(self, "increment_function",   {})
        # Increments
        kw_slice_args = kw.get("increment_slice_args", {})
        kw_smooth     = kw.get("increment_smooth",     {})
        kw_funcs      = kw.get("increment_function",   {})
        # Default smoothing
        smooth = kw.get("smooth", 0.0)
        func   = kw.get("function", "multiquadric")
        # Label format
        fmt = "Differencing %-40s\r" % ("%s slice %i/%i")
        # Loop through coefficients
        for coeff in coeffs:
            # Smoothing and function options
            sk = inc_smooth.get(coeff, smooth)
            fk = inc_funcs.get(coeff, func)
            # Overrides
            sk = kw_smooth.get(coeff, sk)
            fk = kw_funcs.get(coeff, fk)
            # Globals
            sk = kw.get("global_smooth", sk)
            fk = kw.get("global_function", fk)
            # Evaluation arguments
            arg_list = self.eval_args[coeff]
            # Number of arguments
            narg = len(arg_list)
            # Exit if no arguments
            if narg == 0: continue
            # Get slice arguments
            slice_args = inc_slice_args.get(coeff, skeys)
            # Check for specific override
            slice_args = kw_slice_args.get(coeff, slice_args)
            # Check for global override
            slice_args = kw.get("global_slice_args", slice_args)
            # Get test values and slices
            vals, bkpts, J = self._get_test_slices(arg_list, slice_args, **kw)
            # Initialize non-slice args
            interp_args = []
            # Save break points and lookup values
            for (i,k) in enumerate(arg_list):
                # Save values
                dFM[k] = vals[k]
                # Save break points
                dFM.bkpts[k] = bkpts[k]
                # Add to *interp_args* if not a slice argument
                if k not in slice_args: interp_args.append(k)
            # Number of test points
            nx = len(vals[arg_list[0]])
            # Number of interpolation args
            nargi = len(interp_args)
            # Number of slices
            ns = len(J)
            # Initialize deltas
            DV = np.zeros(nx)
            # Loop through slices
            for (i,I) in enumerate(J):
                # Status update
                sys.stdout.write(fmt % (coeff, i+1, ns))
                sys.stdout.flush()
                # Number of points in slice
                nxi = len(I)
                # Initialize matrix of evaluation points
                A = np.zeros((narg, nxi))
                # Initialize tuple of inputs to radial basis function
                X = tuple()
                # Loop through arguments in the list
                for (i,k) in enumerate(arg_list):
                    # Save data to column
                    A[i] = vals[k][I]
                    # Save to RBF inputs
                    if k not in slice_args:
                        X += (vals[k][I],)
                # Evaluate both coefficients
                V1 = self(coeff, *A)
                V2 = FM2(coeff, *A)
                # Deltas
                DVi = V2 - V1
                # Check for valid smoothing
                if (nargi > 0) and (sk > 0):
                    # Add deltas to RBF inputs
                    R = X + (DVi,)
                    # Create RBF
                    fn = scipy.interpolate.rbf.Rbf(*R, function=fk, smooth=sk)
                    # Save evaluated (smoothed deltas)
                    DV[I] = fn(*X)
                else:
                    # Raw deltas if no smoothing or all slices
                    DV[I] = DVi
            # Save differenced values to output database
            dFM[coeff] = DV
            # Reset evaluation method of increment databse
            dFM.eval_method[coeff] = "nearest"
        # Output
        return dFM

    # Create increment
    def CreateIncrementDB(self, FM2, coeffs, skeys=[], **kw):
        """Create database of raw deltas between two databases

        :Call:
            >>> dFM = FM1.CreateIncrementDB(FM2, coeffs, skeys=[], **kw)
        :Inputs:
            *FM1*: :class:`DBCoeff` | :class:`DBFM`
                Original coefficient database
            *FM2*: :class:`DBCoeff` | :class:`DBFM`
                Comparison coefficient database
            *coeffs*: :class:`list` (:class:`str`)
                Coefficients to analyze
        :Keyword Arguments:
            *skeys*: {``[]``} | :class:`list` (:class:`str`)
                List of arguments to define slices on which to smooth
            *test_values*: {``{}``} | :class:`dict` (:class:`np.ndarray`)
                Candidate values of each *eval_arg* for comparison
            *test_bkpts*: {``{}``} | :class:`dict` (:class:`np.ndarray`)
                Candidate break points (1D unique values) for *eval_args*
        :Required Attributes:
            *FM1.uq_coeffs*: :class:`dict` (:class:`list`)
                List of UQ coefficient names for each nominal coefficient
            *FM1.uq_coeffs[coeff]*: ``[ucoeff]``
                List of UQ coefficients for coefficient *coeff* {``[]``}
            *FM1.eval_args[coeff]*: :class:`list` (:class:`str`)
                List of parameters of which *coeff* is a function
            *FM1.eval_args[ucoeff]*: :class:`list` (:class:`str`)
                List of parameters of which *ucoeff* is a function
            *FM1.uq_keys_extra*: {``{}``} | :class:`dict` (:class:`list`)
                List of any extra keys for each uncertainty coeff
            *FM1.uq_keys_shift*: {``{}``} | :class:`dict` (:class:`list`)
                List of keys whose deltas are used to shift *coeff* deltas
            *FM1.uq_funcs_extra*: {``{}``} | :class:`dict` (:class:`function`)
                Function to calculate any "extra" keys by name of key
            *FM1.uq_funcs_shift*: {``{}``} | :class:`dict` (:class:`function`)
                Function to use when "shifting" deltas in *coeff*
            *FM1.uq_keys_extra[ucoeff]*: :class:`list` (:class:`str`)
                List of extra coeffs for *ucoeff* {``[]``}
            *FM1.uq_keys_shift[ucoeff]*: :class:`list` (:class:`str`)
                List of coeffs to use while shifting *ucoeff* deltas  {``[]``}
            *FM1.uq_funcs_extra[ecoeff]*: :class:`function`
                Function to use to estimate extra key *ecoeff*
            *FM1.uq_funcs_shift[ucoeff]*: :class:`function`
                Function to use to shift/alter *ucoeff* deltas
        :Outputs:
            *dFM*: *FM1.__class__*
                Copy of *FM1* with each coeff in *coeffs* replaced with delta
        :Versions:
            * 2019-02-20 ``@ddalle``: First version
        """
        # Create primary increment
        dFM1 = self.DiffDB(FM2, coeffs, skeys=skeys, **kw)
        # Set inputs to force function to generate raw deltas
        kw["global_slice_args"] = []
        kw["global_smooth"] = 0.0
        # Create increment without smoothing
        dFM2 = self.DiffDB(FM2, coeffs, **kw)
        # Remove extra keys from UQ coeff dictionary
        for k in dFM1.uq_coeffs.keys():
            # Check if it's in *coeffs*
            if k not in coeffs:
                # Delete it
                dFM1.uq_coeffs.pop(k)
        # Analyze UQ
        dFM1.EstimateUQ_DB(dFM2, **kw)
        # Output
        return dFM1
  # >

  # ===========
  # Plot
  # ===========
  # <
   # --- Preprocessors ---
    # Process arguments to PlotCoeff()
    def _process_plot_args1(self, *a, **kw):
        """Process arguments to :func:`PlotCoeff` and other plot methods

        :Call:
            >>> coeff, I, J, a, kw = DBc._process_plot_args1(*a, **kw)
            >>> coeff, I, J, a, kw = DBc._process_plot_args1(I, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBFM`
                Coefficient lookup databook
            *a*: :class:`tuple` (:class:`np.ndarray` | :class:`float`)
                Array of values for arguments to evaluator for *coeff*
            *I*: :class:`np.ndarray` (:class:`int`)
                Indices of exact entries to plot
            *kw*: :class:`dict`
                Keyword arguments to plot function and evaluation
        :Outputs:
            *coeff*: :class:`str`
                Coefficient to evaluate
            *I*: :class:`np.ndarray` (:class:`int`)
                Indices of exact entries to plot
            *J*: :class:`np.ndarray` (:class:`int`)
                Indices of matches within *a*
            *a*: :class:`tuple` (:class:`float` | :class:`np.ndarray`)
                Values for arguments for *coeff* evaluator
            *kw*: :class:`dict`
                Processed keyword arguments with defaults applied
        :Versions:
            * 2019-03-14 ``@ddalle``: First version
        """
       # --- Argument Types ---
        # Process coefficient name and remaining coeffs
        coeff, a, kw = self._process_coeff(*a, **kw)
        # Get list of arguments
        arg_list = self.get_eval_arg_list(coeff)
        # Get key for *x* axis
        xk = kw.setdefault("xk", arg_list[0])
        # Check for indices
        if len(a) == 0:
            raise ValueError("At least 3 inputs required; received 2")
        # Process first second arg as indices
        I = np.asarray(a[0])
        # Check for integer
        if (I.ndim > 0) and isinstance(I[0], int):
            # Request for exact values
            qexact  = True
            qinterp = False
            qmark   = False
            qindex  = True
            # Get values of arg list from *DBc* and *I*
            A = []
            # Loop through *eval_args*
            for k in arg_list:
                # Get values
                A.append(self.GetXValues(k, I, **kw))
            # Convert to tuple
            a = tuple(A)
            # Plot all points
            J = np.arange(I.size)
        else:
            # No request for exact values
            qexact  = False
            qinterp = True
            qmark   = True
            qindex  = False
            # Find matches from *a to database points
            I, J = self.FindMatches(arg_list, *a, **kw)
       # --- Options: What to plot ---
        # Plot exact values, interpolated (eval), and markers of actual data
        qexact  = kw.setdefault("PlotExact",  qexact)
        qinterp = kw.setdefault("PlotInterp", qinterp and (not qexact))
        qmark   = kw.setdefault("MarkExact",  qmark and (not qexact))
        # Default UQ coefficient
        uk_def = self.get_uq_coeff(coeff)
        # Check situation
        if typeutils.isarray(uk_def):
            # Get first entry
            uk_def = uk_def[0]
        # Get UQ coefficient
        uk  = kw.get("uk",  kw.get("ucoeff"))
        ukM = kw.get("ukM", kw.get("ucoeff_minus", uk))
        ukP = kw.get("ukP", kw.get("ucoeff_plus",  uk))
        # Turn on *PlotUQ* if UQ key specified
        if ukM or ukP:
            kw.setdefault("PlotUQ", True)
        # UQ flag
        quq = kw.get("PlotUncertainty", kw.get("PlotUQ", False))
        # Set default UQ keys if needed
        if quq:
            uk  = kw.setdefault("uk",  uk_def)
            ukM = kw.setdefault("ukM", uk)
            ukP = kw.setdefault("ukP", uk)
       # --- Default Labels ---
        # Default label starter
        try:
            # Name of component
            try:
                # In some cases *name* is more specific than
                dlbl = self.name
            except AttributeError:
                # Component is usually there
                dlbl = self.comp
        except AttributeError:
            # Backup default
            try:
                # Name of object
                dlbl = self.Name
            except AttributeError:
                # No default
                dlbl = coeff
        # Set default label
        kw.setdefault("Label", dlbl)
        # Default x-axis label is *xk*
        kw.setdefault("XLabel", xk)
        kw.setdefault("YLabel", coeff)
       # --- Cleanup ---
        # Output
        return coeff, I, J, a, kw

   # --- Base Plot Commands ---
    # Plot a sweep of one or more coefficients
    def PlotCoeff(self, *a, **kw):
        """Plot a sweep of one coefficient or quantity over several cases

        This is the base method upon which data book sweep plotting is built.
        Other methods may call this one with modifications to the default
        settings.  For example :func:`cape.cfdx.dataBook.DBTarget.PlotCoeff` changes
        the default *LineOptions* to show a red line instead of the standard
        black line.  All settings can still be overruled by explicit inputs to
        either this function or any of its children.

        :Call:
            >>> h = DBc.PlotCoeff(coeff, *a, **kw)
            >>> h = DBc.PlotCoeff(coeff, I, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBFM`
                Coefficient lookup databook
            *coeff*: :class:`str`
                Coefficient to evaluate
            *a*: :class:`tuple` (:class:`np.ndarray` | :class:`float`)
                Array of values for arguments to evaluator for *coeff*
            *I*: :class:`np.ndarray` (:class:`int`)
                Indices of exact entries to plot
        :Keyword Arguments:
            *xk*: {``None``} | :class:`str`
                Key name for *x* axis
            *PlotExact*: ``True`` | ``False``
                Plot exact values directly from database without interpolation
                Default is ``True`` if *I* is used
            *PlotInterp*: ``True`` | ``False``
                Plot values by using :func:`DBc.__call__`
            *MarkExact*: ``True`` | ``False``
                Mark interpolated curves with markers where actual data points
                are present
        :Plot Options:
            *Legend*: {``True``} | ``False``
                Whether or not to use a legend
            *LegendFontSize*: {``9``} | :class:`int` > 0 | :class:`float`
                Font size for use in legends
            *Grid*: {``None``} | ``True`` | ``False``
                Turn on/off major grid lines, or leave as is if ``None``
            *GridStyle*: {``{}``} | :class:`dict`
                Dictionary of major grid line line style options
            *MinorGrid*: {``None``} | ``True`` | ``False``
                Turn on/off minor grid lines, or leave as is if ``None``
            *MinorGridStyle*: {``{}``} | :class:`dict`
                Dictionary of minor grid line line style options
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
            * 2015-12-14 ``@ddalle``: Added error bars
        """
       # --- Process Args ---
        # Process coefficient name and remaining coeffs
        coeff, I, J, a, kw = self._process_plot_args1(*a, **kw)
        # Get list of arguments
        arg_list = self.get_eval_arg_list(coeff)
        # Get key for *x* axis
        xk = kw.pop("xk", arg_list[0])
       # --- Options: What to plot ---
        # Plot exact values, interpolated (eval), and markers of actual data
        qexact  = kw.pop("PlotExact",  False)
        qinterp = kw.pop("PlotInterp", True)
        qmark   = kw.pop("MarkExact",  True)
        # Uncertainty plot
        quq = kw.pop("PlotUncertainty", kw.pop("PlotUQ", False))
        # Default UQ coefficient
        uk_def = self.get_uq_coeff(coeff)
        # Ensure string
        if typeutils.isarray(uk_def): uk_def = uk_def[0]
        # Get UQ coefficient
        uk  = kw.pop("ucoeff",       kw.pop("uk", uk_def))
        ukM = kw.pop("ucoeff_minus", kw.pop("ukM", uk))
        ukP = kw.pop("ucoeff_plus",  kw.pop("ukP", uk))
       # --- Plot Values ---
        # Y-axis values: exact
        if qexact:
            # Get corresponding *x* values
            xe = self.GetXValues(xk, I, **kw)
            # Try to get values directly from database
            ye = self.GetExactYValues(coeff, I, **kw)
            # Evaluate UQ-minus
            if quq and ukM:
                # Get UQ value below
                uyeM = self.EvalFromIndex(ukM, I, **kw)
            elif quq:
                # Use zeros for negative error term
                uyeM = np.zeros_like(ye)
            # Evaluate UQ-pluts
            if quq and ukP and ukP==ukM:
                # Copy negative terms to positive
                uyeP = uyeM
            elif quq and ukP:
                # Evaluate separate UQ above
                uyeP = self.EvalFromIndex(ukP, I, **kw)
            elif quq:
                # Use zeros
                uyeP = np.zeros_like(ye)
        # Y-axis values: evaluated/interpolated
        if qmark or qinterp:
            # Get values for *x*-axis
            xv = self.GetEvalXValues(xk, coeff, *a, **kw)
            # Evaluate function
            yv = self.__call__(coeff, *a, **kw)
            # Evaluate UQ-minus
            if quq and ukM:
                # Get UQ value below
                uyM = self.EvalFromArgList(ukM, arg_list, *a, **kw)
            elif quq:
                # Use zeros for negative error term
                uyM = np.zeros_like(yv)
            # Evaluate UQ-pluts
            if quq and ukP and ukP==ukM:
                # Copy negative terms to positive
                uyP = uyM
            elif quq and ukP:
                # Evaluate separate UQ above
                uyP = self.EvalFromArgList(ukP, arg_list, *a, **kw)
            elif quq:
                # Use zeros
                uyP = np.zeros_like(yv)
       # --- Data Cleanup ---
        # Create input to *markevery*
        if qmark:
            # Check length
            if J.size == xv.size:
                # Mark all cases
                marke = None
            else:
                # Convert to list
                marke = list(J)
        # Remove extra keywords if possible
        for k in arg_list:
            kw.pop(k, None)
       # --- Primary Plot ---
        # Initialize plot options and get them locally
        kw_p = kw.setdefault("PlotOptions", {})
        # Make copies
        kw_p0 = dict(kw_p)
        # Existing uncertainty plot type
        t_uq = kw.get("PlotTypeUncertainty", kw.get("PlotTypeUQ"))
        # Initialize output
        h = {}
        # Marked and interpolated data
        if qinterp or qmark:
            # Default marker setting
            if qmark:
                kw_p.setdefault("marker", "^")
            else:
                kw_p.setdefault("marker", "")
            # Default line style
            if not qinterp:
                # Turn off lines
                kw.setdefault("ls", "")
                # Default UQ style is error bars
                if t_uq is None:
                    kw["PlotTypeUncertainty"] = "errorbar"
            # Check for uncertainty
            if quq:
                # Set values
                kw["yerr"] = np.array([uyM, uyP])
            # Call the main function
            hi = mpl.plot(xv, yv, **kw)
            # Apply markers
            if qmark:
                # Get line handle
                hl = hi["line"][0]
                # Apply which indices to mark
                hl.set_markevery(marke)
            # Combine
            h = dict(h, **hi)
       # --- Exact Plot ---
        # Plot exact values
        if qexact:
            # Turn on marker
            if "marker" not in kw_p0: kw_p["marker"] = "^"
            # Turn *lw* off
            if "ls" not in kw_p0: kw_p["ls"] = ""
            # Set UQ style to "errorbar"
            if t_uq is None:
                kw["PlotTypeUncertainty"] = "errorbar"
            # Check for uncertainty
            if quq:
                # Set values
                kw["yerr"] = np.array([uyeM, uyeP])
            # Plot exact data
            he = mpl.plot(xe, ye, **kw)
            # Combine
            h = dict(h, **he)
       # --- Cleanup ---
        # Output
        return h
       # ---

    # Plot a sweep of one or more coefficients
    def PlotCoeffDiff(self, DB2, *a, **kw):
        """Plot a sweep of one coefficient or quantity over several cases

        This is the base method upon which data book sweep plotting is built.
        Other methods may call this one with modifications to the default
        settings.  For example :func:`cape.cfdx.dataBook.DBTarget.PlotCoeff` changes
        the default *LineOptions* to show a red line instead of the standard
        black line.  All settings can still be overruled by explicit inputs to
        either this function or any of its children.

        :Call:
            >>> h = DBc.PlotCoeffDiff(DB2, coeff, *a, **kw)
            >>> h = DBc.PlotCoeffDiff(DB2, coeff, I, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBCoeff`
                Coefficient lookup databook
            *DB2*: :class:`tnakit.db.db1.DBCoeff`
                Target second database
            *coeff*: :class:`str`
                Coefficient to evaluate
            *a*: :class:`tuple` (:class:`np.ndarray` | :class:`float`)
                Array of values for arguments to evaluator for *coeff*
            *I*: :class:`np.ndarray` (:class:`int`)
                Indices of exact entries in *DBc* to plot
        :Keyword Arguments:
            *xk*: {``None``} | :class:`str`
                Key name for *x* axis
            *PlotExact*: ``True`` | ``False``
                Plot exact values directly from database without interpolation
                Default is ``True`` if *I* is used
            *PlotInterp*: ``True`` | ``False``
                Plot values by using :func:`DBc.__call__`
            *InterpDB2*: ``True`` | {``False``}
                Interpolate second database or use only exact matches
            *ReverseY*: ``True`` | {``False``}
                Plot values of first database minus second database
            *I2*: {``None``} | :class:`np.ndarray` (:class:`int`)
                Prespecified indices of cases in *DB2* to use; error checked
                but not corrected (can be used where *DB2* is *DBc*)
        :Plot Options:
            *Legend*: {``True``} | ``False``
                Whether or not to use a legend
            *LegendFontSize*: {``9``} | :class:`int` > 0 | :class:`float`
                Font size for use in legends
            *Grid*: {``None``} | ``True`` | ``False``
                Turn on/off major grid lines, or leave as is if ``None``
            *GridStyle*: {``{}``} | :class:`dict`
                Dictionary of major grid line line style options
            *MinorGrid*: {``None``} | ``True`` | ``False``
                Turn on/off minor grid lines, or leave as is if ``None``
            *MinorGridStyle*: {``{}``} | :class:`dict`
                Dictionary of minor grid line line style options
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
            * 2015-12-14 ``@ddalle``: Added error bars
        """
       # --- Process Args ---
        # Process coefficient name and remaining coeffs
        coeff, I1, J1, a, kw = self._process_plot_args1(*a, **kw)
        # Get list of arguments
        arg_list = self.get_eval_arg_list(coeff)
        # Get key for *x* axis
        xk = kw.pop("xk", arg_list[0])
        # Specified indices for *DB2*?
        IB = kw.pop("I2", None)
       # --- Options: What to plot ---
        # Plot exact values, interpolated (eval), and markers of actual data
        qexact  = kw.pop("PlotExact",  False)
        qinterp = kw.pop("PlotInterp", True)
        qmark   = kw.pop("MarkExact",  None)
        # Option for how to get points for *DB2*
        qidb2 = kw.pop("InterpDB2", False)
        # Option to reverse deltas
        revy = kw.pop("ReverseY", False)
       # --- Second Database ---
        # Check method for getting points from second database
        if qidb2:
            # Nothing to demand, so just pass *I* to *I1*
            pass
        elif IB is not None:
            # Manually specified
            I2 = np.asarray(IB)
            # Checking for exact matches but specified by user
            if len(I1) != len(I2):
                raise ValueError(
                    ("Specified %i indices for main database, " % len(I1)) +
                    ("but %i for target" % len(I2)))
        else:
            # Search for exact matches
            I1, I2, J = self.FindMatchesPairIndex(DB2, arg_list, I1, **kw)
            # Remove tolerances
            kw.pop("tol", None)
            kw.pop("tols", None)
       # --- Plot Values ---
        # Y-axis values: exact
        if qexact:
            # Get corresponding *x* values
            xe = self.GetXValues(xk, I1, **kw)
            # Try to get values directly from database
            ye1 = self.GetExactYValues(coeff, I1, **kw)
            # Get target values
            if qidb2:
                # Subset the arguments as well
                A = tuple([self.GetXValues(k, I1) for k in arg_list])
                # Interpolate second database
                ye2 = DB2(coeff, *A)
            else:
                # Exact values
                ye2 = DB2.GetExactYValues(coeff, I2, **kw)
            # Delta
            if revy:
                ye = ye1 - ye2
            else:
                ye = ye2 - ye1
        # Y-axis values: evaluated/interpolated
        if qinterp:
            # Get values for *x*-axis
            xv = self.GetEvalXValues(xk, coeff, *a, **kw)
            # Check interpolation issue
            if qidb2:
                # Evaluate first database
                yv1 = self(coeff, *a, **kw)
                # Evaluate second database
                yv2 = DB2(coeff, *a, **kw)
            else:
                # Subset argument values to matches
                A = tuple([self.GetXValues(k, I1) for k in arg_list])
                # Interpolate first database
                yv1 = self(coeff, *A)
                # Get values from second database
                yv2 = DB2.GetExactYValues(coeff, I2, **kw)
            # Delta
            if revy:
                yv = yv1 - yv2
            else:
                yv = yv2 - yv1
       # --- Data Cleanup ---
        # Remove extra keywords if possible
        for k in arg_list:
            kw.pop(k, None)
       # --- Primary Plot ---
        # Initialize plot options and get them locally
        kw_p = kw.setdefault("PlotOptions", {})
        # Turn off default line style
        kw_p.setdefault("ls", "")
        kw_p.setdefault("marker", "^")
        # Initialize output
        h = {}
        # Marked and interpolated data
        if qinterp:
            # Call the main function
            hi = mpl.plot(xv, yv, **kw)
            # Combine
            h = dict(h, **hi)
       # --- Exact Plot ---
        # Plot exact values
        if qexact:
            # Plot exact data
            he = mpl.plot(xe, ye, **kw)
            # Combine
            h = dict(h, **he)
       # --- Cleanup ---
        # Output
        return h
       # ---

    # Plot a sweep of one or more coefficients
    def PlotHistBase(self, V, **kw):
        """Plot a histogram of one coefficient over several cases

        :Call:
            >>> h = DBc.PlotHistBase(V, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBFM`
                Coefficient lookup databook
            *V*: :class:`numpy.ndarray` (:class:`float`)
                List of values for which to create histogram
        :Keyword Arguments:
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Target*: {``None``} | :class:`DBBase` | :class:`list`
                Target database or list thereof
            *TargetValue*: :class:`float` | :class:`list` (:class:`float`)
                Target or list of target values
            *TargetLabel*: :class:`str` | :class:`list` (:class:`str`)
                Legend label(s) for target(s)
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *HistOptions*: :class:`dict`
                Plot options for the primary histogram
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *MeanOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for mean line
            *TargetOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for target value lines
            *OutlierSigma*: {``7.0``} | :class:`float`
                Standard deviation multiplier for determining outliers
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
            *ShowError*: :class:`bool`
                Option to print value of sampling error
            *ShowDelta*: :class:`bool`
                Option to print reference value
            *ShowTarget*: :class:`bool`
                Option to show target value
            *MuFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the mean value
            *DeltaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the reference value *d*
            *SigmaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the iterative standard deviation
            *TargetFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the target value
            *XLabel*: :class:`str`
                Specified label for *x*-axis, default is ``Iteration Number``
            *YLabel*: :class:`str`
                Specified label for *y*-axis, default is *c*
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
            * 2015-12-14 ``@ddalle``: Added error bars
            * 2016-04-04 ``@ddalle``: Moved from point sensor to data book
        """
       # --- Preparation---
        # Figure dimensions
        fw = kw.get('FigWidth', 6)
        fh = kw.get('FigHeight', 4.5)
       # --- Statistics/Values ---
        # Filter out non-numeric entries
        V = V[np.logical_not(np.isnan(V))]
        # Calculate basic statistics
        vmu = np.mean(V)
        vstd = np.std(V)
        # Check for outliers ...
        ostd = kw.get('OutlierSigma', 3.6863)
        fstd = kw.get('FilterSigma', 5.0777)
        # Range
        cov = kw.get("Coverage", kw.get("cov", 0.99))
        cdf = kw.pop("CoverageCDF", kw.pop("cdf", cov))
        # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
        kcdf = student.ppf(0.5+0.5*cdf, V.size)
        # Check for outliers ...
        fstd = kw.get('FilterSigma', 2.0*kcdf)
        # Remove values from histogram
        if fstd:
            # Find indices of cases that are within outlier range
            J = np.abs(V-vmu)/vstd <= fstd
            # Filter values
            V = V[J]
        # Calculate interval
        acov, bcov = stats.get_cov_interval(V, cov, cdf=cdf, **kw)
       # --- More Options ---
        # Uncertainty options
        ksig = kw.get('StDev')
        # Reference delta
        dc = kw.get('Delta', 0.0)
        # Target values and labels
        vtarg = kw.get('TargetValue')
        ltarg = kw.get('TargetLabel')
        # Convert target values to list
        if vtarg in [None, False]:
            vtarg = []
        elif type(vtarg).__name__ not in ['list', 'tuple', 'ndarray']:
            vtarg = [vtarg]
        # Create appropriate target list for
        if type(ltarg).__name__ not in ['list', 'tuple', 'ndarray']:
            ltarg = [ltarg]
       # --- Histogram Plot ---
        # Initialize handles
        h = {}
        # Initialize plot options for histogram.
        kw_h = dict(facecolor='c', zorder=2, bins=20)
        # Check for global options that should be in here
        for k in ["normed", "density", "orientation"]:
            # Check if present
            if k not in kw:
                continue
            # Apply option
            kw_h.setdefault(k, kw[k])
        # Apply *Label* option if present
        lbl = kw.get("Label")
        if lbl:
            kw_h["label"] = lbl
        # Extract options from kwargs
        for k in denone(kw.get("HistOptions", {})):
            # Override the default option.
            if kw["HistOptions"][k] is not None:
                kw_h[k] = kw["HistOptions"][k]
        # Check for range based on standard deviation
        if kw.get("Range"):
            # Use this number of pair of numbers as multiples of *vstd*
            r = kw["Range"]
            # Check for single number or list
            if type(r).__name__ in ['ndarray', 'list', 'tuple']:
                # Separate lower and upper limits
                vmin = vmu - r[0]*vstd
                vmax = vmu + r[1]*vstd
            else:
                # Use as a single number
                vmin = vmu - r*vstd
                vmax = vmu + r*vstd
            # Overwrite any range option in *kw_h*
            kw_h['range'] = (vmin, vmax)
        # Plot the histogram
        h['hist'] = plt.hist(V, **kw_h)
       # --- Axes Handles ---
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        ax = h['ax']
        # Determine whether or not the distribution is normed
        q_normed = kw_h.get("normed", kw_h.get("density", False))
        # Determine whether or not the bars are vertical
        q_vert = kw_h.get("orientation", "vertical") == "vertical"
        # Get current axis limits
        if q_vert:
            xmin, xmax = ax.get_xlim()
            pmin, pmax = ax.get_ylim()
        else:
            xmin, xmax = ax.get_ylim()
            pmin, pmax = ax.get_xlim()
       # --- Plot Type Switches ---
        # Options to plot various types
        qmu   = kw.get("PlotMean", True)
        qgaus = q_normed and kw.get("PlotGaussian",True)
        qint  = (kw.get("PlotInterval",False) or
            (kw.get("PlotInterval",True) and ("Coverage" in kw)))
        qsig  = ("StDev" in kw) or kw.get("PlotSigma",False)
        qdelt = ("Delta" in kw) or kw.get("PlotDelta",False)
       # --- Mean Plot ---
        # Initialize options for mean plot
        kw_m = dict(color='k', lw=2, zorder=6)
        kw_m["label"] = "Mean value"
        # Extract options from kwargs
        for k in denone(kw.get("MeanOptions", {})):
            # Override the default option.
            if kw["MeanOptions"][k] is not None:
                kw_m[k] = kw["MeanOptions"][k]
        # Option whether or not to plot mean as vertical line.
        if qmu:
            # Check orientation
            if q_vert:
                # Plot a vertical line for the mean.
                h['mean'] = plt.plot([vmu,vmu], [pmin,pmax], **kw_m)
            else:
                # Plot a horizontal line for th emean.
                h['mean'] = plt.plot([pmin,pmax], [vmu,vmu], **kw_m)
       # --- Interval Plot ---
        # Initialize options for mean plot
        kw_i = dict(color='b', lw=0, zorder=1, alpha=0.15)
        kw_m["label"] = "%5.2f%% coverage interval" % (100*cdf)
        # Extract options from kwargs
        for k in denone(kw.get("IntervalOptions", {})):
            # Override the default option.
            if kw["IntervalOptions"][k] is not None:
                kw_i[k] = kw["IntervalOptions"][k]
        # Option whether or not to plot mean as vertical line.
        if qint:
            # Check orientation
            if q_vert:
                # Plot a vertical line for the mean.
                h['interval'] = plt.fill_between(
                    [acov,bcov], [pmax,pmax], **kw_i)
                # Reset vertical limits
                ax.set_ylim(pmin, pmax)
            else:
                # Plot a horizontal line for interval bounds
                h['interval'] = plt.fill_betweenx(
                    [acov,bcov], [pmax,pmax], **kw_i)
                # Reset horizontal limits
                ax.set_xlim(pmin, pmax)
       # --- Target Plot ---
        # Option whether or not to plot targets
        if vtarg is not None and len(vtarg)>0:
            # Initialize options for target plot
            kw_t = dict(color='k', lw=2, ls='--', zorder=8)
            # Set label
            if ltarg is not None:
                # User-specified list of labels
                kw_t["label"] = ltarg
            else:
                # Default label
                kw_t["label"] = "Target"
            # Extract options for target plot
            for k in denone(kw.get("TargetOptions", {})):
                # Override the default option.
                if kw["TargetOptions"][k] is not None:
                    kw_t[k] = kw["TargetOptions"][k]
            # Loop through target values
            for i in range(len(vtarg)):
                # Select the value
                vt = vtarg[i]
                # Check for NaN or None
                if np.isnan(vt) or vt in [None, False]: continue
                # Downselect options
                kw_ti = {}
                for k in kw_t:
                    kw_ti[k] = kw_t.get_key(k, i)
                # Initialize handles
                h['target'] = []
                # Check orientation
                if q_vert:
                    # Plot a vertical line for the target.
                    h['target'].append(
                        plt.plot([vt,vt], [pmin,pmax], **kw_ti))
                else:
                    # Plot a horizontal line for the target.
                    h['target'].append(
                        plt.plot([pmin,pmax], [vt,vt], **kw_ti))
       # --- Standard Deviation Plot ---
        # Initialize options for std plot
        kw_s = dict(color='navy', lw=2, zorder=5)
        # Extract options from kwargs
        for k in denone(kw.get("StDevOptions", {})):
            # Override the default option.
            if kw["StDevOptions"][k] is not None:
                kw_s[k] = kw["StDevOptions"][k]
        # Check whether or not to plot it
        if qsig:
            # Check for single number or list
            if type(ksig).__name__ in ['ndarray', 'list', 'tuple']:
                # Separate lower and upper limits
                vmin = vmu - ksig[0]*vstd
                vmax = vmu + ksig[1]*vstd
            else:
                # Use as a single number
                vmin = vmu - ksig*vstd
                vmax = vmu + ksig*vstd
            # Check orientation
            if q_vert:
                # Plot a vertical line for the min and max
                h['std'] = (
                    plt.plot([vmin,vmin], [pmin,pmax], **kw_s) +
                    plt.plot([vmax,vmax], [pmin,pmax], **kw_s))
            else:
                # Plot a horizontal line for the min and max
                h['std'] = (
                    plt.plot([pmin,pmax], [vmin,vmin], **kw_s) +
                    plt.plot([pmin,pmax], [vmax,vmax], **kw_s))
       # --- Delta Plot ---
        # Initialize options for delta plot
        kw_d = dict(color="r", ls="--", lw=1.0, zorder=3)
        # Extract options from kwargs
        for k in denone(kw.get("DeltaOptions", {})):
            # Override the default option.
            if kw["DeltaOptions"][k] is not None:
                kw_d[k] = kw["DeltaOptions"][k]
        # Check whether or not to plot it
        if qdelt:
            # Check for single number or list
            if type(dc).__name__ in ['ndarray', 'list', 'tuple']:
                # Separate lower and upper limits
                cmin = vmu - dc[0]
                cmax = vmu + dc[1]
            else:
                # Use as a single number
                cmin = vmu - dc
                cmax = vmu + dc
            # Check orientation
            if q_vert:
                # Plot vertical lines for the reference length
                h['delta'] = (
                    plt.plot([cmin,cmin], [pmin,pmax], **kw_d) +
                    plt.plot([cmax,cmax], [pmin,pmax], **kw_d))
            else:
                # Plot horizontal lines for reference length
                h['delta'] = (
                    plt.plot([pmin,pmax], [cmin,cmin], **kw_d) +
                    plt.plot([pmin,pmax], [cmax,cmax], **kw_d))
       # --- Gaussian Plot ---
        # Initialize options for guassian plot
        kw_g = dict(color='navy', lw=1.5, zorder=7)
        kw_g["label"] = "Normal distribution"
        # Extract options from kwargs
        for k in denone(kw.get("GaussianOptions", {})):
            # Override the default option.
            if kw["GaussianOptions"][k] is not None:
                kw_g[k] = kw["GaussianOptions"][k]
        # Check whether or not to plot it
        if qgaus:
            # Get current axis limits
            if q_vert:
                xmin, xmax = ax.get_xlim()
                pmin, pmax = ax.get_ylim()
            else:
                xmin, xmax = ax.get_ylim()
                pmin, pmax = ax.get_xlim()
            # Lookup probabilities
            xval = np.linspace(xmin, xmax, 151)
            # Compute Gaussian distribution
            yval = 1/(vstd*np.sqrt(2*np.pi))*np.exp(-0.5*((xval-vmu)/vstd)**2)
            # Check orientation
            if q_vert:
                # Plot a vertical line for the mean.
                h['mean'] = plt.plot(xval, yval, **kw_g)
            else:
                # Plot a horizontal line for th emean.
                h['mean'] = plt.plot(yval, xval, **kw_g)
       # --- Formatting ---
        # Coefficient name
        coeff = kw.get("coeff", "mu")
        # Is this a delta plot?
        targ = kw.get("target", False)
        # Default value-axis label
        if targ:
            # Error in coeff
            lx = u'\u0394%s' % coeff
        else:
            # Just the value
            lx = coeff
        # Default probability-axis label
        if q_normed:
            # Size of bars is probability
            ly = "Probability Density"
        else:
            # Size of bars is count
            ly = "Count"
        # Process axis labels
        xlbl = kw.get('XLabel')
        ylbl = kw.get('YLabel')
        # Apply defaults
        if xlbl is None: xlbl = lx
        if ylbl is None: ylbl = ly
        # Check for flipping
        if not q_vert:
            xlbl, ylbl = ylbl, xlbl
        # Labels.
        h['x'] = plt.xlabel(xlbl)
        h['y'] = plt.ylabel(ylbl)
        # Correct the font.
        try: h['x'].set_family("DejaVu Sans")
        except Exception: pass
        # Set figure dimensions
        if fh: h['fig'].set_figheight(fh)
        if fw: h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes.
        try: plt.tight_layout()
        except Exception: pass
       # --- Labels ---
        # y-coordinates of the current axes w.r.t. figure scale
        ya = h['ax'].get_position().get_points()
        ha = ya[1,1] - ya[0,1]
        # y-coordinates above and below the box
        yf = 2.5 / ha / h['fig'].get_figheight()
        yu = 1.0 + 0.065*yf
        yl = 1.0 - 0.04*yf
        # Make a label for the mean value.
        if qmu or kw.get("ShowMu", True):
            # printf-style format flag
            flbl = kw.get("MuFormat", "%.4f")
            # Check for deltas
            if targ:
                # Form: mu(DCA) = 0.0204
                klbl = (u'\u03bc(\u0394%s)' % coeff)
            else:
                # Form: CA = 0.0204
                klbl = (u'%s' % coeff)
            # Check for option
            olbl = kw.get("MuLabel", klbl)
            # Use non-default user-specified value
            if olbl is not None: klbl = olbl
            # Insert value
            lbl = ('%s = %s' % (klbl, flbl)) % vmu
            # Create the handle.
            h['mu'] = plt.text(0.99, yu, lbl, color=kw_m['color'],
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['mu'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the deviation.
        if qdelt and kw.get("ShowDelta", True):
            # printf-style flag
            flbl = kw.get("DeltaFormat", "%.4f")
            # Form: \DeltaCA = 0.0050
            lbl = (u'\u0394%s = %s' % (coeff, flbl)) % dc
            # Create the handle.
            h['d'] = plt.text(0.01, yl, lbl,
                color=kw_d.get('color','r'),
                horizontalalignment='left', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['d'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the standard deviation.
        if qsig and kw.get("ShowSigma", True):
            # Printf-style flag
            flbl = kw.get("SigmaFormat", "%.4f")
            # Check for deltas
            if targ:
                # Form: sigma(DCA) = 0.0204
                klbl = (u'\u03c3(\u0394%s)' % coeff)
            else:
                # Form: sigma(CA) = 0.0204
                klbl = (u'\u03c3(%s)' % coeff)
            # Check for option
            olbl = kw.get("SigmaLabel", klbl)
            # Use non-default user-specified value
            if olbl is not None: klbl = olbl
            # Insert value
            lbl = ('%s = %s' % (klbl, flbl)) % vstd
            # Create the handle.
            h['sig'] = plt.text(0.01, yu, lbl,
                color=kw_s.get('color','navy'),
                horizontalalignment='left', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['sig'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the Interval
        if qint and kw.get("ShowInterval", True):
            # Printf-style flag
            flbl = kw.get("IntervalFormat", "%.4f")
            # Form
            klbl = "I(%.1f%%%%)" % (100*cdf)
            # Check for option
            olbl = kw.get("IntervalLabel", klbl)
            # Use non-default user-specified value
            if olbl is not None: klbl = olbl
            # Insert value
            lbl = ('%s = [%s,%s]' % (klbl, flbl, flbl)) % (acov,bcov)
            # Create the handle.
            h['int'] = plt.text(0.99, yl, lbl,
                color=kw_i.get('color','navy'),
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['int'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the iterative uncertainty.
        if len(vtarg)>0 and kw.get("ShowTarget", True):
            # printf-style format flag
            flbl = kw.get("TargetFormat", "%.4f")
            # Form Target = 0.0032
            lbl = (u'%s = %s' % (ltarg[0], flbl)) % vtarg[0]
            # Create the handle.
            h['t'] = plt.text(0.99, yl, lbl,
                color=kw_t.get('color','b'),
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['t'].set_family("DejaVu Sans")
            except Exception: pass
        # Output.
        return h
       # ---

    # Plot a sweep of one or more coefficients
    def PlotRangeHistBase(self, V, **kw):
        """Plot a range histogram of one coefficient over several cases

        :Call:
            >>> h = DBc.PlotRangeHistBase(V, **kw)
        :Inputs:
            *DBc*: :class:`tnakit.db.db1.DBFM`
                Coefficient lookup databook
            *V*: :class:`numpy.ndarray` (:class:`float`)
                List of values for which to create range histogram
        :Keyword Arguments:
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Target*: {``None``} | :class:`DBBase` | :class:`list`
                Target database or list thereof
            *TargetValue*: :class:`float` | :class:`list` (:class:`float`)
                Target or list of target values
            *TargetLabel*: :class:`str` | :class:`list` (:class:`str`)
                Legend label(s) for target(s)
            *Coverage*: {``0.99``} | 0 < :class:`float` < 1
                Coverage fraction
            *CoverageFactor*: {``1.0``} | :class:`float`
                Manual multiplier on range
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *HistOptions*: :class:`dict`
                Plot options for the primary histogram
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *CoverageOptions*: :class:`dict`
                Dictionary of plot options for the coverage range plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *MeanOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for mean line
            *TargetOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for target value lines
            *OutlierSigma*: {``3.6863``} | :class:`float`
                Standard deviation multiplier for determining outliers
            *FilterSigma*: {``5.0777``} | :class:`float`
                Standard deviation multiplier for removing from plot
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
            *ShowError*: :class:`bool`
                Option to print value of sampling error
            *ShowDelta*: :class:`bool`
                Option to print reference value
            *ShowRange*: :class:`bool`
                Option to print coverage range
            *ShowTarget*: :class:`bool`
                Option to show target value
            *MuFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the mean value
            *DeltaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the reference value *d*
            *RangeFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the coverage range width
            *SigmaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the iterative standard deviation
            *TargetFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the target value
            *XLabel*: :class:`str`
                Specified label for *x*-axis, default is ``Iteration Number``
            *YLabel*: :class:`str`
                Specified label for *y*-axis, default is *c*
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
            * 2015-12-14 ``@ddalle``: Added error bars
            * 2016-04-04 ``@ddalle``: Moved from point sensor to data book
        """
       # --- Preparation ---
        # Figure dimensions
        fw = kw.get('FigWidth', 6)
        fh = kw.get('FigHeight', 4.5)
       # --- Statistics/Values ---
        # Filter out non-numeric entries
        V = V[np.logical_not(np.isnan(V))]
        # Get ranges (absolute values)
        R = np.abs(V)
        # Calculate basic statistics
        vmu = np.mean(R)
        vstd = 0.5*vmu*np.sqrt(np.pi)
        # Check for outliers ...
        ostd = kw.get('OutlierSigma', 3.6863)
        fstd = kw.get('FilterSigma', 5.0777)
        # Range
        cov = kw.get("Coverage", kw.get("cov", 0.99))
        rcov = kw.get("CoverageFactor", 1.0)
        cdf = kw.get("CoverageCDF", kw.get("cdf", cov))
        # Remove values from histogram
        if fstd:
            # Find indices of cases that are within outlier range
            J = np.abs(R)/vstd <= fstd
            # Filter values
            V = V[J]
            R = R[J]
        # Calculate range
        wrange = rcov*stats.get_range(R, cov, cdf=cdf, **kw)
       # --- More Options ---
        # Uncertainty options
        ksig = kw.get('StDev', 3.0)
        # Reference delta
        dc = kw.get('Delta', 0.0)
        # Target values and labels
        vtarg = kw.get('TargetValue')
        ltarg = kw.get('TargetLabel')
        # Convert target values to list
        if vtarg in [None, False]:
            vtarg = []
        elif type(vtarg).__name__ not in ['list', 'tuple', 'ndarray']:
            vtarg = [vtarg]
        # Create appropriate target list for
        if type(ltarg).__name__ not in ['list', 'tuple', 'ndarray']:
            ltarg = [ltarg]
       # --- Plotting ---
        # Initialize dictionary of handles.
        h = {}
       # --- Histogram Plot ---
        # Initialize plot options for histogram.
        kw_h = dict(facecolor='c', zorder=2, bins=20)
        # Check for global options that should be in here
        for k in ["normed", "density", "orientation"]:
            # Check if present
            if k not in kw:
                continue
            # Apply option
            kw_h.setdefault(k, kw[k])
        # Apply *Label* option if present
        lbl = kw.get("Label")
        if lbl:
            kw_h["label"] = lbl
        # Extract options from kwargs
        for k in denone(kw.get("HistOptions", {})):
            # Override the default option.
            if kw["HistOptions"][k] is not None:
                kw_h[k] = kw["HistOptions"][k]
        # Check for range based on standard deviation
        if kw.get("Range"):
            # Use this number of pair of numbers as multiples of *vstd*
            r = kw["Range"]
            # Check for single number or list
            if type(r).__name__ in ['ndarray', 'list', 'tuple']:
                # Separate lower and upper limits
                vmin = vmu - r[0]*vstd
                vmax = vmu + r[1]*vstd
            else:
                # Use as a single number
                vmin = 0
                vmax = r*vstd
            # Overwrite any range option in *kw_h*
            kw_h['range'] = (vmin, vmax)
        # Plot the histogram
        h['hist'] = plt.hist(R, **kw_h)
       # --- Axes Handles ---
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        ax = h['ax']
        # Determine whether or not the distribution is normed
        q_normed = kw_h.get("normed", kw_h.get("density", False))
        # Determine whether or not the bars are vertical
        q_vert = kw_h.get("orientation", "vertical") == "vertical"
        # Get current axis limits
        if q_vert:
            xmin, xmax = ax.get_xlim()
            pmin, pmax = ax.get_ylim()
        else:
            xmin, xmax = ax.get_ylim()
            pmin, pmax = ax.get_xlim()
       # --- Gaussian Plot ---
        # Initialize options for guassian plot
        kw_g = dict(color='navy', lw=1.5, zorder=7)
        kw_g["label"] = "Normal distribution"
        # Extract options from kwargs
        for k in denone(kw.get("GaussianOptions", {})):
            # Override the default option.
            if kw["GaussianOptions"][k] is not None:
                kw_g[k] = kw["GaussianOptions"][k]
        # Check whether or not to plot it
        if q_normed and kw.get("PlotGaussian"):
            # Lookup probabilities
            xval = np.linspace(0, xmax, 151)
            # Compute Gaussian distribution
            yval = 1/(vstd*np.sqrt(np.pi))*np.exp(-0.25*(xval/vstd)**2)
            # Check orientation
            if q_vert:
                # Plot a vertical line for the mean.
                h['mean'] = plt.plot(xval, yval, **kw_g)
            else:
                # Plot a horizontal line for th emean.
                h['mean'] = plt.plot(yval, xval, **kw_g)
       # --- Mean Plot ---
        # Initialize options for mean plot
        kw_m = dict(color='k', lw=2, zorder=6)
        kw_m["label"] = "Mean value"
        # Extract options from kwargs
        for k in denone(kw.get("MeanOptions", {})):
            # Override the default option.
            if kw["MeanOptions"][k] is not None:
                kw_m[k] = kw["MeanOptions"][k]
        # Option whether or not to plot mean as vertical line.
        if kw.get("PlotMean", True):
            # Check orientation
            if q_vert:
                # Plot a vertical line for the mean.
                h['mean'] = plt.plot([vmu,vmu], [pmin,pmax], **kw_m)
            else:
                # Plot a horizontal line for th emean.
                h['mean'] = plt.plot([pmin,pmax], [vmu,vmu], **kw_m)
       # --- Target Plot ---
        # Option whether or not to plot targets
        if vtarg is not None and len(vtarg)>0:
            # Initialize options for target plot
            kw_t = dict(color='k', lw=2, ls='--', zorder=8)
            # Set label
            if ltarg is not None:
                # User-specified list of labels
                kw_t["label"] = ltarg
            else:
                # Default label
                kw_t["label"] = "Target"
            # Extract options for target plot
            for k in denone(kw.get("TargetOptions", {})):
                # Override the default option.
                if kw["TargetOptions"][k] is not None:
                    kw_t[k] = kw["TargetOptions"][k]
            # Loop through target values
            for i in range(len(vtarg)):
                # Select the value
                vt = vtarg[i]
                # Check for NaN or None
                if np.isnan(vt) or vt in [None, False]: continue
                # Downselect options
                kw_ti = {}
                for k in kw_t:
                    kw_ti[k] = kw_t.get_key(k, i)
                # Initialize handles
                h['target'] = []
                # Check orientation
                if q_vert:
                    # Plot a vertical line for the target.
                    h['target'].append(
                        plt.plot([vt,vt], [pmin,pmax], **kw_ti))
                else:
                    # Plot a horizontal line for the target.
                    h['target'].append(
                        plt.plot([pmin,pmax], [vt,vt], **kw_ti))
       # --- Standard Deviation Plot ---
        # Initialize options for std plot
        kw_s = dict(color='navy', lw=1.5, zorder=5, dashes=[3,2])
        # Extract options from kwargs
        for k in denone(kw.get("StDevOptions", {})):
            # Override the default option.
            if kw["StDevOptions"][k] is not None:
                kw_s[k] = kw["StDevOptions"][k]
        # Check whether or not to plot it
        if ("StDev" in kw) or kw.get("PlotSigma",False):
            # Use a single number
            vmax = ksig*vstd
            # Check orientation
            if q_vert:
                # Plot a vertical line for the min and max
                h['std'] = plt.plot([vmax,vmax], [pmin,pmax], **kw_s)
            else:
                # Plot a horizontal line for the min and max
                h['std'] = plt.plot([pmin,pmax], [vmax,vmax], **kw_s)
       # --- Range Plot ---
        # Initialize options for std plot
        kw_r = dict(color='b', lw=2, zorder=5)
        # Extract options from kwargs
        for k in denone(kw.get("RangeOptions", {})):
            # Override the default option.
            if kw["RangeOptions"][k] is not None:
                kw_r[k] = kw["RangeOptions"][k]
        # Check whether or not to plot it
        if wrange and len(V)>2 and kw.get("PlotRange",True):
            # Check orientation
            if q_vert:
                # Plot a vertical line for the min and max
                h['std'] = plt.plot([wrange,wrange], [pmin,pmax], **kw_r)
            else:
                # Plot a horizontal line for the min and max
                h['std'] = plt.plot([pmin,pmax], [wrange,wrange], **kw_r)
       # --- Delta Plot ---
        # Initialize options for delta plot
        kw_d = dict(color="r", ls="--", lw=1.0, zorder=3)
        # Extract options from kwargs
        for k in denone(kw.get("DeltaOptions", {})):
            # Override the default option.
            if kw["DeltaOptions"][k] is not None:
                kw_d[k] = kw["DeltaOptions"][k]
        # Check whether or not to plot it
        if dc:
            # Check orientation
            if q_vert:
                # Plot vertical lines for the reference length
                h['delta'] = plt.plot([dc,dc], [pmin,pmax], **kw_d)
            else:
                # Plot horizontal lines for reference length
                h['delta'] = plt.plot([pmin,pmax], [dc,dc], **kw_d)
       # --- Formatting ---
        # Coefficient name
        coeff = kw.get("coeff", "mu")
        # Is this a delta plot?
        targ = kw.get("target", False)
        # Default value-axis label
        if targ:
            # Error in coeff
            lx = u'|\u0394%s|' % coeff
        else:
            # Just the value
            lx = u'|%s|' % coeff
        # Default probability-axis label
        if q_normed:
            # Size of bars is probability
            ly = "Probability Density"
        else:
            # Size of bars is count
            ly = "Count"
        # Process axis labels
        xlbl = kw.get('XLabel')
        ylbl = kw.get('YLabel')
        # Apply defaults
        if xlbl is None: xlbl = lx
        if ylbl is None: ylbl = ly
        # Check for flipping
        if not q_vert:
            xlbl, ylbl = ylbl, xlbl
        # Labels.
        h['x'] = plt.xlabel(xlbl)
        h['y'] = plt.ylabel(ylbl)
        # Correct the font.
        try: h['x'].set_family("DejaVu Sans")
        except Exception: pass
        # Set figure dimensions
        if fh: h['fig'].set_figheight(fh)
        if fw: h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes.
        try: plt.tight_layout()
        except Exception: pass
       # --- Labels ---
        # y-coordinates of the current axes w.r.t. figure scale
        ya = h['ax'].get_position().get_points()
        ha = ya[1,1] - ya[0,1]
        # y-coordinates above and below the box
        yf = 2.5 / ha / h['fig'].get_figheight()
        yu = 1.0 + 0.065*yf
        yl = 1.0 - 0.04*yf
        # Make a label for the mean value.
        if kw.get("ShowMu", True):
            # printf-style format flag
            flbl = kw.get("MuFormat", "%.4f")
            # Check for deltas
            if targ:
                # Form: mu(DCA) = 0.0204
                klbl = (u'\u03bc(\u0394%s)' % coeff)
            else:
                # Form: CA = 0.0204
                klbl = (u'%s' % coeff)
            # Check for option
            olbl = kw.get("MuLabel", klbl)
            # Use non-default user-specified value
            if olbl is not None: klbl = olbl
            # Insert value
            lbl = ('%s = %s' % (klbl, flbl)) % vmu
            # Create the handle.
            h['mu'] = plt.text(0.01, yu, lbl, color=kw_m['color'],
                horizontalalignment='left', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['mu'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the deviation.
        if dc and kw.get("ShowDelta", True):
            # printf-style flag
            flbl = kw.get("DeltaFormat", "%.4f")
            # Form: \DeltaCA = 0.0050
            lbl = (u'\u0394%s = %s' % (coeff, flbl)) % dc
            # Create the handle.
            h['d'] = plt.text(0.01, yl, lbl,
                color=kw_d.get('color','r'),
                horizontalalignment='left', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['d'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the standard deviation.
        if len(V)>2 and (ksig and kw.get("ShowSigma", True)):
            # Printf-style flag
            flbl = kw.get("SigmaFormat", "%.4f")
            # Check for deltas
            if targ:
                # Form: sigma(DCA) = 0.0204
                klbl = (u'\u03c3(\u0394%s)' % coeff)
            else:
                # Form: sigma(CA) = 0.0204
                klbl = (u'\u03c3(%s)' % coeff)
            # Check for option
            olbl = kw.get("SigmaLabel", klbl)
            # Use non-default user-specified value
            if olbl is not None: klbl = olbl
            # Insert value
            lbl = ('%.1f%s = %s' % (ksig, klbl, flbl)) % (ksig*vstd)
            # Create the handle.
            h['sig'] = plt.text(0.99, yu, lbl,
                color=kw_s.get('color','navy'),
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['sig'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the standard deviation.
        if len(V)>2 and ((cov and kw.get("ShowRange", True))
                or kw.get("ShowRange", False)):
            # Printf-style flag
            flbl = kw.get("RangeFormat", "%.4f")
            # Form: R99(DCA) = 0.0204
            klbl = (u'R%02i(\u0394%s)' % (100*cdf, coeff))
            # Check for option
            olbl = kw.get("RangeLabel", klbl)
            # Use non-default user-specified value
            if olbl is not None: klbl = olbl
            # Insert value
            lbl = ('%s = %s' % (klbl, flbl)) % wrange
            # Create the handle.
            h['sig'] = plt.text(0.99, yl, lbl,
                color=kw_r.get('color','b'),
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['sig'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the iterative uncertainty.
        if len(vtarg)>0 and kw.get("ShowTarget", True):
            # printf-style format flag
            flbl = kw.get("TargetFormat", "%.4f")
            # Form Target = 0.0032
            lbl = (u'%s = %s' % (ltarg[0], flbl)) % vtarg[0]
            # Create the handle.
            h['t'] = plt.text(0.99, yl, lbl,
                color=kw_t.get('color','b'),
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['t'].set_family("DejaVu Sans")
            except Exception: pass
        # Output.
        return h
       # ---
  # >
# class DBCoeff


# Function to fix "NoneType is not iterable" nonsense
def denone(x):
    """Replace ``None`` with ``[]`` to avoid iterative problems

    :Call:
        >>> y = denone(x)
    :Inputs:
        *x*: any
            Any variable
    :Outputs:
        *y*: any
            Same as *x* unless *x* is ``None``, then ``[]``
    :Versions:
        * 2015-03-09 ``@ddalle``: First version
        * 2018-06-22 ``@ddalle``: Copied from :mod:`cape.util`
    """
    if x is None:
        return []
    else:
        return x
# def denone

# Function to automatically get inclusive data limits.
def get_ylim(ha, pad=0.05):
    """Calculate appropriate *y*-limits to include all lines in a plot

    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` and
    :class:`matplotlib.collections.PolyCollection` are checked.

    :Call:
        >>> ymin, ymax = get_ylim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *ymin*: :class:`float`
            Minimum *y* coordinate including padding
        *ymax*: :class:`float`
            Maximum *y* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    # Initialize limits.
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Get the y data for this line
            ydata = h.get_ydata()
            # Check the min and max data
            if len(ydata) > 0:
                ymin = min(ymin, min(h.get_ydata()))
                ymax = max(ymax, max(h.get_ydata()))
        elif t in ['PathCollection', 'PolyCollection']:
            # Loop through paths
            for P in h.get_paths():
                # Get the coordinates
                ymin = min(ymin, min(P.vertices[:,1]))
                ymax = max(ymax, max(P.vertices[:,1]))
    # Check for identical values
    if ymax - ymin <= 0.1*pad:
        # Expand by manual amount,.
        ymax += pad*abs(ymax)
        ymin -= pad*abs(ymin)
    # Add padding.
    yminv = (1+pad)*ymin - pad*ymax
    ymaxv = (1+pad)*ymax - pad*ymin
    # Output
    return yminv, ymaxv

# Function to automatically get inclusive data limits.
def get_xlim(ha, pad=0.05):
    """Calculate appropriate *x*-limits to include all lines in a plot

    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` are
    checked.

    :Call:
        >>> xmin, xmax = get_xlim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *xmin*: :class:`float`
            Minimum *x* coordinate including padding
        *xmax*: :class:`float`
            Maximum *x* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Get data
            xdata = h.get_xdata()
            # Check the min and max data
            if len(xdata) > 0:
                xmin = min(xmin, min(h.get_xdata()))
                xmax = max(xmax, max(h.get_xdata()))
        elif t in ['PathCollection', 'PolyCollection']:
            # Loop through paths
            for P in h.get_paths():
                # Get the coordinates
                xmin = min(xmin, min(P.vertices[:,1]))
                xmax = max(xmax, max(P.vertices[:,1]))
    # Check for identical values
    if xmax - xmin <= 0.1*pad:
        # Expand by manual amount,.
        xmax += pad*abs(xmax)
        xmin -= pad*abs(xmin)
    # Add padding.
    xminv = (1+pad)*xmin - pad*xmax
    xmaxv = (1+pad)*xmax - pad*xmin
    # Output
    return xminv, xmaxv
# def get_xlim

