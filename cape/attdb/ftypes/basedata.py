#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.ftypes.basedata`: Common ATTDB data container
=====================================================================

This module provides the class :class:`BaseData` as a subclass of
:class:`dict` that contains methods common to each of the other
data-like file readers and writers.  It defines common attributes that
each other such data-file class will have and provides several other
common methods.

For example, it defines a :func:`__repr__` method that updates with the
name of any other classes subclassed to this one.

Finally, having this common template class provides a single point of
entry for testing if an object is based on a product of the
:mod:`cape.attdb.ftypes` module.  The following Python sample tests if
any Python object *db* is an instance of any class from this data-file
collection.

    .. code-block:: python

        isinstance(db, cape.attdb.ftypes.BaseData)
"""

# Standard library modules
import os
import warnings

# Third-party modules
import numpy as np

# CAPE modules
import cape.tnakit.kwutils as kwutils
import cape.tnakit.typeutils as typeutils

# Fixed parameter for size of new chunks
NUM_ARRAY_CHUNK = 5000


# Declare basic class
class BaseData(dict):
    r"""Generic class for storing data from a data-style file
    
    This class has no initialization method, and as such it is unlikely
    that there will be instances of this class in use.  It provides
    methods and structure to other classes.
    
    This class inherits from :class:`dict` and can be used in that
    matter in the unlikely event that it's useful.
    
    :Outputs:
        *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
            CSV file interface
        *db.cols*: :class:`list`\ [:class:`str`]
            List of columns read
        *db.opts*: :class:`dict`
            Options for this instance
        *db.opts["Definitions"]*: :class:`dict`
            Definitions for each column/coefficient
        *db[col]*: :class:`np.ndarray` | :class:`list`
            Numeric array or list of strings for each column
    :See also:
        * :class:`cape.attdb.ftypes.csv.CSVFile`
        * :class:`cape.attdb.ftypes.csv.CSVSimple`
        * :class:`cape.attdb.ftypes.textdata.TextDataFile`
    :Versions:
        * 2019-11-26 ``@ddalle``: First version
    """
    # Class attributes
    _classtypes = []
    # Recognized types and other defaults
    _DefaultOpts = {
        "ExpandScalars": True,
    }
    _DefaultDefn = {
        "Type": "float64",
        "Label": True,
        "LabelFormat": "%s",
        "WriteFormat": "%s",
    }
    _DefaultRoleDefns = {}
    _DTypeMap = {}
    _RoleMap = {}
    # Permitted keyword names
    _kw = [
        "cols",
        "ExpandScalars",
        "Definitions",
        "DefaultDefinition",
        "Values"
    ]
    _kw_map = {
        "ColumnNames": "cols",
        "Keys": "cols",
        "defns": "Definitions",
        "vals": "Values",
    }
    _kw_depends = {}
    _kw_types = {
        "cols": list,
        "Definitions": dict,
        "Values": dict,
        "ExpandScalars": (bool, int),
        "DefaultDefinition": dict,
        "DefaultType": typeutils.strlike
    }

  # ==========
  # Config
  # ==========
  # <
    # Representation method
    def __repr__(self):
        """Generic representation method

        :Versions:
            * 2019-11-08 ``@ddalle``: First version
            * 2019-12-31 ``@ddalle``: Safe attributes
        """
        # Module name
        modname = self.__class__.__module__
        clsname = self.__class__.__name__
        # Strip module name part
        modname = modname.rsplit(".", 1)[-1]
        # Start output
        lbl = "<%s.%s(" % (modname, clsname)
        # Get file name (safely)
        fname = self.__dict__.get("fname")
        # Append file name if appropriate
        if fname:
            lbl += "'%s', " % os.path.basename(fname)
        # Get columns (safely)
        cols = self.__dict__.get("cols", [])
        # Display columns
        if len(cols) <= 6:
            # Show all columns
            lbl += "cols=%s)>" % str(cols)
        else:
            # Just show number of columns
            lbl += "ncol=%i)>" % len(cols)
        # Output
        return lbl

    # String method
    def __str__(self):
        """Generic representation method

        :Versions:
            * 2019-11-08 ``@ddalle``: First version
            * 2019-12-04 ``@ddalle``: Only last part of module name
            * 2019-12-31 ``@ddalle``: Safe attributes
        """
        # Module name
        modname = self.__class__.__module__
        clsname = self.__class__.__name__
        # Strip module name part
        modname = modname.rsplit(".", 1)[-1]
        # Start output
        lbl = "<%s.%s(" % (modname, clsname)
        # Get file name (safely)
        fname = self.__dict__.get("fname")
        # Append file name if appropriate
        if fname:
            lbl += "'%s', " % os.path.basename(fname)
        # Get columns (safely)
        cols = self.__dict__.get("cols", [])
        # Display columns
        if len(cols) <= 5:
            # Show all columns
            lbl += "cols=%s)>" % str(cols)
        else:
            # Just show number of columns
            lbl += "ncol=%i)>" % len(cols)
        # Output
        return lbl
  # >
  
  # =================
  # Options
  # =================
  # <
   # --- Options ---
    # Get options handle
    def get_opts(self):
        r"""Get dictionary of options

        :Call:
            >>> opts = db.get_opts()
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
        :Outputs:
            *opts*: :class:`dict`
                Options for data container
        :Versions:
            * 2019-12-31 ``@ddalle``: First version
        """
        return self.__dict__.setdefault("opts", {})

    # Single option
    def get_opt(self, key, vdef=None):
        r"""Get an option, appealing to default if necessary

        :Call:
            >>> val = db.get_opt(key, vdef=None)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *key*: :class:`str`
                Name of option to access
            *vdef*: {``None``} | :class:`any`
                Default option for fallback
        :Outputs:
            *val*: *db.opts[key]* | *cls._DefaultOpts[key]* | *vdef*
                Value of option with fallback
        :Versions:
            * 2019-12-31 ``@ddalle``: First version
        """
        # Get options
        opts = self.get_opts()
        # Check for option
        if key in opts:
            # Directly specified
            return opts[key]
        # Otherwise use the class
        return self.__class__._DefaultOpts.get(key, vdef)
        
   # --- Definitions ---
    # Get definitions
    def get_defns(self):
        r"""Get dictionary of column definitions

        :Call:
            >>> defns = db.get_defns()
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
        :Outputs:
            *defns*: :class:`dict`\ [:class:`dict`]
                Definitions for each column
        :Versions:
            * 2019-12-31 ``@ddalle``: First version
        """
        # Get options
        opts = self.get_opts()
        # Get definitions
        return opts.get("Definitions", {})

   # --- Column Properties ---
    # Get generic property from column
    def get_col_prop(self, col, prop, vdef=None):
        """Get property for specific column
        
        :Call:
            >>> v = db.get_col_prop(col, prop, vdef=None)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column
            *prop*: :class:`str`
                Name of property
            *vdef*: {``None``} | :class:`any`
                Default value if not specified in *db.opts*
        :Outputs:
            *v*: :class:`any`
                Value of ``defns[col][prop]`` if possible; defaulting to
                ``defns["_"][prop]`` or *vdef*
        :Versions:
            * 2019-11-24 ``@ddalle``: First version
            * 2019-12-31 ``@ddalle``: Moved from :mod:`basefile`
        """
        # Check if column is present
        if col not in self.cols:
            # Allow default
            if col != "_":
                raise KeyError("No column '%s'" % col)
        # Get definitions
        defns = self.get_defns()
        # Get specific definition
        defn = defns.get(col, {})
        # Check if option available
        if prop in defn:
            # Return it
            return defn[prop]
        else:
            # Use default
            defn = defns.get("_", {})
            # Get property from default definition
            return defn.get(prop, vdef)

    # Get type
    def get_col_type(self, col):
        """Get data type for specific column
        
        :Call:
            >>> cls = db.get_col_type(col, prop)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column
        :Outputs:
            *cls*: ``"int"`` | ``"float"`` | ``"str"`` | :class:`str`
                Name of data type
        :Versions:
            * 2019-11-24 ``@ddalle``: First version
        """
        return self.get_col_prop(col, "Type", vdef="float64")

    # Get data type
    def get_col_dtype(self, col):
        """Get data type for specific column
        
        :Call:
            >>> cls = db.get_col_type(col, prop)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column
        :Outputs:
            *cls*: ``"int"`` | ``"float"`` | ``"str"`` | :class:`str`
                Name of data type
        :Versions:
            * 2019-11-24 ``@ddalle``: First version
        """
        # Get type
        coltype = self.get_col_type(col)
        # Apply mapping if needed
        return self.__class__._DTypeMap.get(coltype, coltype)
        
   # --- Keyword Values ---
    # Query keyword arguments for manual values
    def process_kw_values(self, **kw):
        r"""Process *Values* argument for manual column values
        
        :Call:
            >>> db.process_kw_values(**kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *Values*, *vals*: :class:`dict`
                Dictionary of values for some columns
            *ExpandScalars*: ``True`` | ``False``
                Option to expand scalars to match dimension of arrays
            *n*: {*db.n*} | :class:`int` > 0
                Target length for *ExpandScalars*
        :Versions:
            * 2019-11-12 ``@ddalle``: First version
            * 2019-12-31 ``@ddalle``: Removed :func:`pop` and output
        """
        # Get values
        vals1 = kw.get("Values", {})
        vals2 = kw.get("vals", {})
        # Get expansion option
        expand = kw.get("ExpandScalars", self.get_opt("ExpandScalars", True))
        # Get number for expansion
        n = max(1, self.__dict__.get("n", 1))
        # Check types
        if not isinstance(vals1, dict):
            raise TypeError(
                "'Values' keyword must be dict, found %s" % vals1.__class__)
        elif not isinstance(vals2, dict):
            raise TypeError(
                "'vals' keyword must be dict, found %s" % vals2.__class__)
        # Combine inputs
        vals = dict(vals2, **vals1)
        # Process values
        for (col, v) in vals.items():
            # Check for scalar
            if isinstance(v, (list, np.ndarray)):
                # Use array
                V = v
                # Update *n*
                n = max(n, len(V))
            elif expand:
                # Get type
                coltyp = self.get_col_type(col)
                # Convert if necessary
                colcls = self.__class__._DTypeMap.get(coltyp, coltyp)
                # Check for list-like
                if colcls == "str":
                    # Create expanded list
                    V = [v] * n
                else:
                    # Create array
                    V = v * np.ones(n, dtype=colcls)
            else:
                # Use as is
                V = v
            # Save values
            self.save_col(col, V)
            # Save length
            self.n = n

   # --- Keyword Checker ---
    # Check valid keyword names, with dependencies
    def map_kw(self, kwmap, **kw):
        r"""Map alternate keyword names with no checks

        :Call:
            >>> kwo = db.map_kw(**kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *kw*: :class:`dict`
                Any keyword arguments
        :Outputs:
            *kwo*: :class:`dict`
                Translated keywords and their values from *kw*
        :Versions:
            * 2019-12-13 ``@ddalle``: First version
        """
        # Get class
        cls = self.__class__
        # Call generic function
        return kwutils.map_kw(cls._kw_map, **kw)

    # Check valid keyword names, with dependencies
    def check_kw(self, mode, **kw):
        r"""Check and map valid keyword names

        :Call:
            >>> kwo = db.check_kw(mode, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *mode*: ``0`` | {``1``} | ``2``
                Flag for quiet (``0``), warn (``1``), or strict (``2``)
            *kw*: :class:`dict`
                Any keyword arguments
        :Outputs:
            *kwo*: :class:`dict`
                Valid keywords
        :Versions:
            * 2019-12-13 ``@ddalle``: First version
        """
        # Get class
        cls = self.__class__
        # Call generic function with specific attributes
        return kwutils.check_kw(
            cls._kw,
            cls._kw_map,
            cls._kw_depends,
            mode, **kw)

    # Check valid keyword names against specified list
    def check_kw_list(self, kwlist, mode, **kw):
        r"""Check and map valid keyword names

        :Call:
            >>> kwo = db.check_kw_list(kwlist, mode, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *kwlist*: :class:`list`\ [:class:`str`]
                List of acceptable keyword names
            *mode*: ``0`` | {``1``} | ``2``
                Flag for quiet (``0``), warn (``1``), or strict (``2``)
            *kw*: :class:`dict`
                Any keyword arguments
        :Outputs:
            *kwo*: :class:`dict`
                Valid keywords
        :Versions:
            * 2019-12-13 ``@ddalle``: First version
        """
        # Get class
        cls = self.__class__
        # Call generic function with specific attributes
        return kwutils.check_kw(
            kwlist,
            cls._kw_map,
            cls._kw_depends,
            mode, **kw)

    # Check valid keyword names, with dependencies
    def check_kw_types(self, mode, **kw):
        r"""Check and map valid keyword names and types

        :Call:
            >>> kwo = db.check_kw_types(mode, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *mode*: ``0`` | {``1``} | ``2``
                Flag for quiet (``0``), warn (``1``), or strict (``2``)
            *kw*: :class:`dict`
                Any keyword arguments
        :Outputs:
            *kwo*: :class:`dict`
                Valid keywords
        :Versions:
            * 2019-12-13 ``@ddalle``: First version
        """
        # Get class
        cls = self.__class__
        # Call generic function with specific attributes
        return kwutils.check_kw_types(
            cls._kw,
            cls._kw_map,
            cls._kw_types,
            cls._kw_depends,
            mode, **kw)
  # >

  # ===============
  # Data
  # ===============
  # <
   # --- Save Data ---
    # Save a column
    def save_col(self, col, v):
        r"""Save a column value, updating other metadata as needed
        
        :Call:
            >>> db.save_col(col, v)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column
            *v*: :class:`np.ndarray` | :class:`list` | :class:`scalar`
                Value(s) to save for specified column
        :Versions:
            * 2019-11-12 ``@ddalle``: Started
        """
        # Check if column is present
        if col not in self.cols:
            self.cols.append(col)
        # Check type
        if isinstance(v, np.ndarray):
            # Save as is
            self[col] = v
        elif isinstance(v, list):
            # Check first element
            if len(v) == 0:
                # Nothing to convert
                self[col] = v
            elif isinstance(v[0], (int, float, complex)):
                # Convert to array
                self[col] = np.asarray(v)
            else:
                # No conversion
                self[col] = v
        else:
            # Nonstandard value; don't convert
            self[col] = v
  # >
# class BaseData


# Append keywords for *DefaultDefinition* thing
def _append_kw_DefaultDefn(cls, attr="_kw"):
    # Get list of parameters
    _kw = getattr(cls, attr)
    # Loop through keys in *cls._DefaultDefn*
    for k in cls._DefaultDefn:
        # Derivative key name
        k1 = "Default" + k
        # Check if key is present
        if k1 not in _kw:
            # Append the parameter
            _kw.append(k1)
    # Loop through keys in *cls._DefaultDefn*
    for k in cls._DefaultDefn:
        # Derivative key name
        k1 = k + "s"
        # Check if key is present
        if k1 not in _kw:
            # Append the parameter
            _kw.append(k1)
        # Save the type as a dict
        cls._kw_types[k1] = dict


# Add parameters
_append_kw_DefaultDefn(BaseData)
