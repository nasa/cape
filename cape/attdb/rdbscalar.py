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

  # ===================
  # Config
  # ===================
  # <
   # --- Attributes ---
    # Get attribute with defaults
    def getattrdefault(self, attr, vdef):
        r"""Get attribute from instance

        :Call:
            >>> v = db.getattrdefault(attr, vdef)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *attr*: :class:`str`
                Name of attribute
            *vdef*: any
                Default value of attribute
        :Outputs:
            *v*: *db.__dict__[attr]* | *vdef*
                One of the two values specified above
        :Effects:
            *db.__dict__[attr]*: *v*
                Sets *db.(attr)* if necessary
        :Versions:
            * 2019-12-18 ``@ddalle``: First version
        """
        return self.__dict__.setdefault(attr, vdef)

    # Get an attribute with default to dictionary
    def getattrdict(self, attr):
        r"""Get attribute from instance

        :Call:
            >>> v = db.getattrdefault(attr, vdef)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *attr*: :class:`str`
                Name of attribute
            *vdef*: any
                Default value of attribute
        :Outputs:
            *v*: *db.__dict__[attr]* | *vdef*
                One of the two values specified above
        :Effects:
            *db.__dict__[attr]*: *v*
                Sets *db.__dict__[attr]* if necessary
        :Versions:
            * 2019-12-18 ``@ddalle``: First version
        """
        return self.__dict__.setdefault(attr, {})
  # >

  # ===============
  # Eval/Call
  # ===============
  # <
   # --- Evaluation ---
    # Evaluate interpolation
    def __call__(self, *a, **kw):
        """Generic evaluation function

        :Call:
            >>> v = db(*a, **kw)
            >>> v = db(col, x0, x1, ...)
            >>> V = db(col, x0, X1, ...)
            >>> v = db(col, k0=x0, k1=x1, ...)
            >>> V = db(col, k0=x0, k1=X1, ...)
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
        col, a, kw = self._get_colname(*a, **kw)
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
        method_col = self.eval_method.get(col, method_def)
        # Check for ``None``, which forbids lookup
        if method_col is None:
            raise ValueError("Col '%s' is not an evaluation coeff" % col)
       # --- Get argument list ---
        # Specific lookup arguments (and copy it)
        args_col = list(self.get_eval_arg_list(col))
       # --- Evaluation kwargs ---
        # Attempt to get default aliases
        try:
            # Check for attribute and "_" default
            kw_def = self.eval_kwargs["_"]
            # Use default as fallback
            kw_fn = self.eval_kwargs.get(col, kw_def)
        except AttributeError:
            # No kwargs to eval functions
            kw_fn = {}
        except KeyError:
            # No default
            kw_fn = self.eval_kwargs.get(col, {})
       # --- Aliases ---
        # Attempt to get default aliases
        try:
            # Check for attribute and "_" default
            alias_def = self.eval_arg_aliases["_"]
            # Use default as fallback
            arg_aliases = self.eval_arg_aliases.get(col, alias_def)
        except AttributeError:
            # No aliases
            arg_aliases = {}
        except KeyError:
            # No default
            arg_aliases = self.eval_arg_aliases.get(col, {})
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
        for i, k in enumerate(args_col):
            # Get value
            xi = self.get_arg_value(i, k, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        X, dims = self.normalize_args(x)
        # Maximum dimension
        nd = len(dims)
        # Data size
        nx = np.prod(dims)
       # --- Evaluation ---
        # Process the appropriate lookup function
        if method_col in ["nearest"]:
            # Evaluate nearest-neighbor lookup
            f = self.eval_nearest
        elif method_col in ["linear", "multilinear"]:
            # Evaluate using multilinear interpolation
            f = self.eval_multilinear
        elif method_col in ["linear-schedule", "multilinear-schedule"]:
            # Evaluate using scheduled (in 1D) multilinear interpolation
            f = self.eval_multilinear_schedule
        elif method_col in ["rbf"]:
            # Evaluate global radial basis function
            f = self.eval_rbf
        elif method_col in ["rbf-slice", "rbf-linear"]:
            # Evaluate linear interpolation of two RBFs
            f = self.eval_rbf_linear
        elif method_col in ["rbf-map", "rbf-schedule"]:
            # Evaluate curvilinear interpolation of slice RBFs
            f = self.eval_rbf_schedule
        elif method_col in ["function", "func", "fn"]:
            # Combine args
            kw_fn = dict(kw_fn, **kw)
            # Evaluate specific function
            f = self.eval_function
        else:
            # Unknown method
            raise ValueError(
                "Could not interpret evaluation method '%s'" % method_col)
        # Calls
        if nd == 0:
            # Scalar call
            v = f(col, args_col, x, **kw_fn)
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
                V[j] = f(col, args_col, xj, **kw_fn)
            # Reshape
            V = V.reshape(dims)
            # Output
            return V

   # --- Declaration ---
    # Set evaluation methods
    def SetEvalMethod(self, cols=None, method=None, args=None, *a, **kw):
        r"""Set evaluation method for a one or more columns

        :Call:
            >>> db.SetEvalMethod(col, method=None, args=None, **kw)
            >>> db.SetEvalMethod(cols, method=None, args=None, **kw)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *cols*: :class:`list`\ [:class:`str`]
                List of columns for which to declare evaluation rules
            *col*: :class:`str`
                Name of column for which to declare evaluation rules
            *method*: ``"nearest"`` | ``"linear"`` | :class:`str`
                Response (lookup/interpolation/evaluation) method name 
            *args*: :class:`list`\ [:class:`str`]
                List of input arguments
            *aliases*: {``{}``} | :class:`dict`\ [:class:`str`]
                Dictionary of alternate variable names during
                evaluation; if *aliases[k1]* is *k2*, that means *k1*
                is an alternate name for *k2*, and *k2* is in *args*
            *eval_kwargs*: {``{}``} | :class:`dict`
                Keyword arguments passed to functions
            *I*: {``None``} | :class:`np.ndarray`
                Indices of cases to include in response surface {all}
            *function*: {``"cubic"``} | :class:`str`
                Radial basis function type
            *smooth*: {``0.0``} | :class:`float` >= 0
                Smoothing factor for methods that allow inexact
                interpolation, ``0.0`` for exact interpolation
        :Versions:
            * 2019-01-07 ``@ddalle``: First version
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Check for list
        if isinstance(cols, typeutils.strlike):
            # Singleton list
            cols = [cols]
        elif not isinstance(cols, (list, tuple, set)):
            # Not a list
            raise TypeError(
                "Columns to specify evaluation for must be list; " +
                ("got '%s'" % type(cols)))
        # Loop through coefficients
        for col in cols:
            # Check type
            if not isinstance(col, typeutils.strlike):
                # Not a string
                raise TypeError("Eval col must be a string")
            # Specify individual col
            self._set_method1(col, method, args, *a, **kw)

    # Save a method for one coefficient
    def _set_method1(self, col=None, method=None, args=None, *a, **kw):
        r"""Set evaluation method for a single column

        :Call:
            >>> db._set_method1(col=None, method=None, args=None, **kw)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of column for which to declare evaluation rules
            *method*: ``"nearest"`` | ``"linear"`` | :class:`str`
                Response (lookup/interpolation/evaluation) method name 
            *args*: :class:`list`\ [:class:`str`]
                List of input arguments
            *aliases*: {``{}``} | :class:`dict`\ [:class:`str`]
                Dictionary of alternate variable names during
                evaluation; if *aliases[k1]* is *k2*, that means *k1*
                is an alternate name for *k2*, and *k2* is in *args*
            *eval_kwargs*: {``{}``} | :class:`dict`
                Keyword arguments passed to functions
            *I*: {``None``} | :class:`np.ndarray`
                Indices of cases to include in response surface {all}
            *function*: {``"cubic"``} | :class:`str`
                Radial basis function type
            *smooth*: {``0.0``} | :class:`float` >= 0
                Smoothing factor for methods that allow inexact
                interpolation, ``0.0`` for exact interpolation
        :Versions:
            * 2019-01-07 ``@ddalle``: First version
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
       # --- Metadata checks ---
        # Dictionary of methods
        eval_method = self.getattrdict("eval_method")
        # Argument lists
        eval_args = self.getattrdict("eval_args")
        # Argument aliases (i.e. alternative names)
        eval_arg_aliases = self.getattrdict("eval_arg_aliases")
        # Evaluation keyword arguments
        eval_kwargs = self.getattrdict("eval_kwargs")
       # --- Input checks ---
        # Check inputs
        if col is None:
            # Set the default
            col = "_"
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
        eval_arg_aliases[col] = arg_aliases
        # Get alias option
        eval_kwargs_kw = kw.get("eval_kwargs", {})
        # Check for ``None``
        if (not eval_kwargs_kw):
            # Empty option is empty dictionary
            eval_kwargs_kw = {}
        # Save keywords (new copy)
        eval_kwargs[col] = dict(eval_kwargs_kw)
       # --- Method switch ---
        # Check for identifiable method
        if method in ["nearest"]:
            # Nearest-neighbor lookup
            eval_method[col] = "nearest"
        elif method in ["linear", "multilinear"]:
            # Linear/multilinear interpolation
            eval_method[col] = "multilinear"
        elif method in ["linear-schedule", "multilinear-schedule"]:
            # (N-1)D linear interp in last keys, 1D in first key
            eval_method[col] = "multilinear-schedule"
        elif method in ["rbf", "rbg-global", "rbf0"]:
            # Create global RBF
            self.CreateGlobalRBFs([col], args, **kw)
            # Metadata
            eval_method[col] = "rbf"
        elif method in ["lin-rbf", "rbf-linear", "linear-rbf"]:
            # Create RBFs on slices
            self.CreateSliceRBFs([col], args, **kw)
            # Metadata
            eval_method[col] = "rbf-linear"
        elif method in ["map-rbf", "rbf-schedule", "rbf-map", "rbf1"]:
            # Create RBFs on slices but scheduled
            self.CreateSliceRBFs([col], args, **kw)
            # Metadata
            eval_method[col] = "rbf-map"
        elif method in ["function", "fn", "func"]:
            # Create eval_func dictionary
            eval_func = self.getattrdict("eval_func")
            # Create eval_func dictionary
            eval_func_self = self.getattrdict("eval_func_self")
            # Get the function
            if len(a) > 0:
                # Function given as arg
                fn = a[0]
            else:
                # Function better be a keyword because there are no args
                fn = None

            # Save the function
            eval_func[col] = kw.get("function", kw.get("func", fn))
            eval_func_self[col] = kw.get("self", True)

            # Dedicated function
            eval_method[col] = "function"
        else:
            raise ValueError(
                "Did not recognize evaluation type '%s'" % method)
        # Argument list is the same for all methods
        eval_args[col] = args

   # --- Attributes ---
    # Get argument list
    def get_eval_arg_list(self, col):
        r"""Get list of evaluation arguments

        :Call:
            >>> args = db.get_eval_arg_list(col)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of column to evaluate
        :Outputs:
            *args*: :class:`list`\ [:class:`str`]
                List of parameters used to evaluate *col*
        :Versions:
            * 2019-03-11 ``@ddalle``: Forked from :func:`__call__`
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
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
        args_col = self.eval_args.get(col, args_def)
        # Check for ``None``, which forbids lookup
        if args_col is None:
            raise ValueError("Column '%s' is not an evaluation cooeff" % col)
        # Output a copy
        return list(args_col)

    # Get evaluation method
    def get_eval_method(self, col):
        r"""Get evaluation method (if any) for a column

        :Call:
            >>> meth = db.get_eval_method(col)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of column to evaluate
        :Outputs:
            *meth*: ``None`` | :class:`str`
                Name of evaluation method, if any
        :Versions:
            * 2019-03-13 ``@ddalle``: First version
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get attribute
        eval_methods = self.__dict__.setdefault("eval_method", {})
        # Get method
        return eval_methods.get(col)

    # Get evaluation argument converter
    def get_eval_arg_converter(self, k):
        r"""Get evaluation argument converter

        :Call:
            >>> f = db.get_eval_arg_converter(k)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *k*: :class:`str` | :class:`unicode`
                Name of argument
        :Outputs:
            *f*: ``None`` | callable
                Callable converter
        :Versions:
            * 2019-03-13 ``@ddalle``: First version
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get converter dictionary
        converters = self.__dict__.setdefault("eval_arg_covnerters", {})
        # Get converter
        f = converters.get(k)
        # Output if None
        if f is None:
            return f
        # Check class
        if not callable(f):
            raise TypeError("Converter for '%s' is not callable" % k)
        # Output
        return f

    # Get UQ coefficient
    def get_uq_coeff(self, coeff):
        r"""Get name of UQ coefficient(s) for *coeff*

        :Call:
            >>> ucoeff = db.get_uq_coeff(coeff)
            >>> ucoeffs = db.get_uq_coeff(coeff)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *coeff*: :class:`str`
                Name of coefficient to evaluate
        :Outputs:
            *ucoeff*: ``None`` | :class:`str`
                Name of UQ coefficient for *coeff*
            *ucoeffs*: :class:`list`\ [:class:`str`]
                List of UQ coefficients for *coeff*
        :Versions:
            * 2019-03-13 ``@ddalle``: First version
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get dictionary of UQ coeffs
        uq_coeffs = self.__dict__.setdefault("uq_coeffs", {})
        # Get entry for this coefficient
        return uq_coeffs.get(coeff)

   # --- Options: Set ---
    # Set a default value for an argument
    def set_arg_default(self, k, v):
        r"""Set a default value for an evaluation argument

        :Call:
            >>> db.set_arg_default(k, v)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *k*: :class:`str`
                Name of evaluation argument
            *v*: :class:`float`
                Default value of the argument to set
        :Versions:
            * 2019-02-28 ``@ddalle``: First version
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get dictionary
        arg_defaults = self.__dict__.setdefault("eval_arg_defaults", {})
        # Save key/value
        arg_defaults[k] = v

    # Set a conversion function for input variables
    def set_arg_converter(self, k, fn):
        r"""Set a function to evaluation argument for a specific argument

        :Call:
            >>> db.set_arg_converter(k, fn)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *k*: :class:`str`
                Name of evaluation argument
            *fn*: :class:`function`
                Conversion function
        :Versions:
            * 2019-02-28 ``@ddalle``: First version
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Check input
        if not callable(fn):
            raise TypeError("Converter is not callable")
        # Get dictionary of converters
        arg_converters = self.__dict__.setdefault("eval_arg_converters", {})
        # Save function
        arg_converters[k] = fn

   # --- Arguments ---
    # Attempt to get all values of an argument
    def get_all_values(self, k):
        r"""Attempt to get all values of a specified argument

        This will use *db.eval_arg_converters* if possible.

        :Call:
            >>> V = db.get_all_values(k)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *k*: :class:`str`
                Name of evaluation argument
        :Outputs:
            *V*: ``None`` | :class:`np.ndarray`\ [:class:`float`]
                *db[k]* if available, otherwise an attempt to apply
                *db.eval_arg_converters[k]*
        :Versions:
            * 2019-03-11 ``@ddalle``: First version
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Check if present
        if k in self:
            # Get values
            return self[k]
        # Otherwise check for evaluation argument
        arg_converters = self.__dict__.get("eval_arg_converters", {})
        # Check if there's a converter
        if k not in arg_converters:
            return None
        # Get converter
        f = arg_converters.get(k)
        # Check if there's a converter
        if f is None:
            # No converter
            return
        elif not callable(f):
            # Not callable
            raise TypeError("Converter for col '%s' is not callable" % k)
        # Attempt to apply it
        try:
            # Call in keyword-only mode
            V = f(**self)
            # Return values
            return V
        except Exception:
            # Failed
            return None

    # Get argument value
    def get_arg_value(self, i, k, *a, **kw):
        r"""Get the value of the *i*\ th argument to a function

        :Call:
            >>> v = db.get_arg_value(i, k, *a, **kw)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *i*: :class:`int`
                Argument index within *db.eval_args*
            *k*: :class:`str`
                Name of evaluation argument
            *a*: :class:`tuple`
                Arguments to :func:`__call__`
            *kw*: :class:`dict`
                Keyword arguments to :func:`__call__`
        :Outputs:
            *v*: :class:`float` | :class:`np.ndarray`
                Value of the argument, possibly converted
        :Versions:
            * 2019-02-28 ``@ddalle``: First version
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Number of direct arguments
        na = len(a)
        # Converters
        arg_converters = self.__dict__.get("eval_arg_converters", {})
        arg_defaults   = self.__dict__.get("eval_arg_defaults",   {})
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
                print("Eval argument converter for '%s' failed" % k)
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
        r"""Return a dictionary of normalized argument variables

        Specifically, he dictionary contains a key for every argument used to
        evaluate the coefficient that is either the first argument or uses the
        keyword argument *coeff*.

        :Call:
            >>> X = db.get_arg_value_dict(*a, **kw)
            >>> X = db.get_arg_value_dict(coeff, x1, x2, ..., k3=x3)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
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
                according to *b.eval_args[coeff]*; each entry of *X* will
                have the same size
        :Versions:
            * 2019-03-12 ``@ddalle``: First version
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
       # --- Get coefficient name ---
        # Coeff name should be either a[0] or kw["coeff"]
        coeff, a, kw = self._get_colname(*a, **kw)
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
        xn, dims = self.normalize_args(x)
       # --- Output ---
        # Initialize
        X = {}
        # Loop through args
        for i, k in enumerate(args_coeff):
            # Save value
            X[k] = xn[i]
        # Output
        return X

    # Process coefficient name
    def _get_colname(self, *a, **kw):
        r"""Process coefficient name from arbitrary inputs

        :Call:
            >>> col, a, kw = db._get_colname(*a, **kw)
            >>> col, a, kw = db._get_colname(col, *a, **kw)
            >>> col, a, kw = db._get_colname(*a, col=c, **kw)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of evaluation col
            *a*: :class:`tuple`
                Other sequential inputs
            *kw*: :class:`dict`
                Other keyword inputs
        :Outputs:
            *col*: :class:`str`
                Name of column to evaluate, from *a[0]* or *kw["col"]*
            *a*: :class:`tuple`
                Remaining inputs with coefficient name removed
            *kw*: :class:`dict`
                Keyword inputs with coefficient name removed
        :Versions:
            * 2019-03-12 ``@ddalle``: First version
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
            * 2019-12-18 ``@ddalle``: From :func:`_process_coeff`
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

    # Normalize arguments
    def normalize_args(self, x, asarray=False):
        r"""Normalized mixed float and array arguments

        :Call:
            >>> X, dims = db.normalize_args(x, asarray=False)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *x*: :class:`list`\ [:class:`float` | :class:`np.ndarray`]
                Values for arguments, either float or array
            *asarray*: ``True`` | {``False``}
                Force array output (otherwise allow scalars)
        :Outputs:
            *X*: :class:`list`\ [:class:`float` | :class:`np.ndarray`]
                Normalized arrays/floats all with same size
            *dims*: :class:`tuple` (:class:`int`)
                Original dimensions of non-scalar input array
        :Versions:
            * 2019-03-11 ``@ddalle``: First version
            * 2019-03-14 ``@ddalle``: Added *asarray* input
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
            * 2019-12-18 ``@ddalle``: Removed ``@staticmethod``
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
        for (i, xi) in enumerate(x):
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
                # Scalar to array
                X.append(xi*np.ones(nx))
            elif ndk != nd:
                # Inconsistent size
                raise ValueError(
                    "Cannot normalize %iD and %iD inputs" % (ndk, nd))
            elif nxk != nx:
                # Inconsistent size
                raise IndexError(
                    "Cannot normalize inputs with size %i and %i" % (nxk, nx))
            else:
                # Already array
                X.append(xi.flatten())
        # Output
        return X, dims

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
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of explanatory col names (numeric)
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *tol*: {``1.0e-4``} | :class:`float` > 0
                Default tolerance for exact match
            *tols*: {``{}``} | :class:`dict`\ [:class:`float` > 0]
                Dictionary of key-specific tolerances
        :Outputs:
            *y*: ``None`` | :class:`float` | *db[col].__class__*
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

    # Lookup nearest value
    def eval_nearest(self, col, args, x, **kw):
        r"""Evaluate a coefficient by looking up nearest match

        :Call:
            >>> y = db.eval_nearest(col, args, x, **kw)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of (numeric) column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of explanatory col names (numeric)
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *weights*: {``{}``} | :class:`dict` (:class:`float` > 0)
                Dictionary of key-specific distance weights
        :Outputs:
            *y*: :class:`float` | *db[col].__class__*
                Value of *db[col]* at point closest to *x*
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
        # Array length
        n = len(V)
        # Initialize distances
        d = np.zeros(n, dtype="float")
        # Dictionary of distance weights
        W = kw.get("weights", {})
        # Loop through keys
        for (i, k) in enumerate(args):
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

   # --- Linear ---
    # Multilinear lookup
    def eval_multilinear(self, col, args, x, **kw):
        r"""Perform linear interpolation in *n* dimensions

        This assumes the database is ordered with the first entry of
        *args* varying the most slowly and that the data is perfectly
        regular.

        :Call:
            >>> y = db.eval_multilinear(col, args, x)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *bkpt*: ``True`` | {``False``}
                Flag to interpolate break points instead of data
        :Outputs:
            *y*: ``None`` | :class:`float` | ``db[col].__class__``
                Interpolated value from ``db[col]``
        :Versions:
            * 2018-12-30 ``@ddalle``: First version
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Call root method without two of the options
        return self._eval_multilinear(col, args, x, **kw)

    # Evaluate multilinear interpolation with caveats
    def _eval_multilinear(self, col, args, x, I=None, j=None, **kw):
        r"""Perform linear interpolation in *n* dimensions

        This assumes the database is ordered with the first entry of
        *args* varying the most slowly and that the data is perfectly
        regular.

        :Call:
            >>> y = db._eval_multilinear(col, args, x, I=None, j=None)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of column to evaluate
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
                Flag to interpolate break points instead of data
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Interpolated value from ``DBc[coeff]``
        :Versions:
            * 2018-12-30 ``@ddalle``: First version
            * 2019-04-19 ``@ddalle``: Moved from :func:`eval_multilnear`
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Check for break-point evaluation flag
        bkpt = kw.get("bkpt", kw.get("breakpoint", False))
        # Possible values
        try:
            # Extract coefficient
            if bkpt:
                # Lookup from breakpoints
                V = self.bkpts[col]
            else:
                # Lookup from main data
                V = self[col]
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
                    "Failed to subset col '%s' using class '%s'"
                    % (coeff, I.__class__))
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
                ("Column '%s' has size %i, " % (col, n)),
                ("but total size of args %s is %i." % (args, np.prod(N))))
        # Initialize list of indices for each key
        I0 = []
        I1 = []
        F1 = []
        # Get lookup indices for each argument
        for (i, k) in enumerate(args):
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
    def eval_multilinear_schedule(self, col, args, x, **kw):
        r"""Perform "scheduled" linear interpolation in *n* dimensions

        This assumes the database is ordered with the first entry of
        *args* varying the most slowly and that the data is perfectly
        regular.  However, each slice at a constant value of *args[0]*
        may have separate break points for all the other args.  For
        example, the matrix of angle of attack and angle of sideslip
        may be different at each Mach number.  In this case, *db.bkpts*
        will be a list of 1D arrays for *alpha* and *beta* and just a
        single 1D array for *mach*.

        :Call:
            >>> y = db.eval_multilinear(col, args, x)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *tol*: {``1e-6``} | :class:`float` >= 0
                Tolerance for matching slice key
        :Outputs:
            *y*: ``None`` | :class:`float` | ``db[col].__class__``
                Interpolated value from ``db[col]``
        :Versions:
            * 2019-04-19 ``@ddalle``: First version
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
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
        y0 = self._eval_multilinear(col, args, x0, I=I0, j=i0)
        y1 = self._eval_multilinear(col, args, x1, I=I1, j=i1)
        # Linear interpolation in the schedule key
        return (1-f)*y0 + f*y1

   # --- Radial Basis Functions ---
    # RBF lookup
    def eval_rbf(self, col, args, x, **kw):
        """Evaluate a single radial basis function

        :Call:
            >>> y = DBc.eval_rbf(col, args, x)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
        :Outputs:
            *y*: ``None`` | :class:`float` | ``db[col].__class__``
                Interpolated value from ``db[col]``
        :Versions:
            * 2018-12-31 ``@ddalle``: First version
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get the radial basis function
        f = self.get_rbf(col)
        # Evaluate
        return f(*x)

    # Get an RBF
    def get_rbf(self, col, *I):
        r"""Extract a radial basis function, with error checking

        :Call:
            >>> f = db.get_rbf(col, *I)
            >>> f = db.get_rbf(col)
            >>> f = db.get_rbf(col, i)
            >>> f = db.get_rbf(col, i, j)
            >>> f = db.get_rbf(col, i, j, ...)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of column to evaluate
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
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get the radial basis function
        try:
            fn = self.rbf[col]
        except AttributeError:
            # No radial basis functions at all
            raise AttributeError("No radial basis functions found")
        except KeyError:
            # No RBF for this coefficient
            raise KeyError("No radial basis function for col '%s'" % col)
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
                    ("RBF for '%s':\n" % col) +
                    ("Expecting %i-dimensional " % nd) +
                    ("array but found %i-dim" % n))
        # Test type
        if not callable(fn):
            raise TypeError("RBF '%s' index %i is not callable" % (col, I))
        # Output
        return fn

   # --- Generic Function ---
    # Generic function
    def eval_function(self, col, args, x, **kw):
        """Evaluate a single user-saved function

        :Call:
            >>> y = DBc.eval_function(col, args, x)
        :Inputs:
            *db*: :class:`attdb.rdbscalar.DBResponseScalar`
                Coefficient database interface
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Interpolated value from ``DBc[coeff]``
        :Versions:
            * 2018-12-31 ``@ddalle``: First version
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get the function
        try:
            f = self.eval_func[col]
        except AttributeError:
            # No evaluation functions set
            raise AttributeError(
                "No evaluation functions present in database")
        except KeyError:
            # No keys
            raise KeyError(
                "No evaluation function for col '%s'" % col)
        # Evaluate
        if self.eval_func_self.get(col):
            # Use reference to *self*
            return f(self, *x, **kw)
        else:
            # Stand-alone function
            return f(*x, **kw)
  # >