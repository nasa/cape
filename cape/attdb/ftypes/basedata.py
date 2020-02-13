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
import copy
import os

# Third-party modules
import numpy as np

# CAPE modules
import cape.tnakit.kwutils as kwutils
import cape.tnakit.typeutils as typeutils


# Options for BaseData
class BaseDataOpts(kwutils.KwargHandler):
  # ====================
  # Class Attributes
  # ====================
  # <
   # --- Global Options ---
    # List of options
    _optlist = {
        "Columns",
        "ExpandScalars",
        "Definitions",
        "DefaultDefinition",
        "Values"
    }

    # Alternate names
    _optmap = {
        "cols": "Columns",
        "defns": "Definitions",
        "vals": "Values",
    }

   # --- Types ---
    # Types
    _opttypes = {
        "Columns": list,
        "ExpandScalars": bool,
        "Definitions": dict,
        "DefaultDefinition": dict,
        "Values": dict,
    }

   # --- Defaults ---
    # Default values
    _rc = {
        "Definitions": {},
        "DefaultDefinition": {},
    }

   # --- Documentation ---
        
  # >

  # =================
  # Methods
  # =================
  # <
   # --- Definitions ---
    # Get definition for specified column
    def get_defn(self, col):
        r"""Get the processed definition, without applying defaults

        This method returns a definition-type instance that checks the
        *Definition* option and dict-like options like *Types*.
        Defaults, both from the definition class's *_rc* attribute and
        options like *DefaultType*, are not applied so that they can be
        automatically guessed from the data.

        :Call:
            >>> defn = opts.get_defn(col)
        :Inputs:
            *opts*: :class:`BaseDataOpts`
                Options interface for :mod:`cape.attdb.ftypes`
            *col*: :class:`str`
                Name of data column
        :Outputs:
            *defn*: :class:`BaseDataDefn` | *opts._defncls*
                Column definition for *col*
        :Versions:
            * 2020-01-31 ``@ddalle``: First version
        """
        # Get class
        cls = self.__class__
        # Get class for column definitions
        defncls = cls._defncls
        # Get directly-specified definitions
        defns = self.get_option("Definitions", {})
        # Get specific definition and set type
        defn = defncls(defns.get(col, {}))
        # Loop through dicts of definition parameters
        for opt in defncls._optlist:
            # Derived key name; e.g. "Definitions.Type" -> "Types"
            opt2 = opt + "s"
            # Get dict of definitions by col, e.g. "Types"
            val2 = self.get(opt2)
            # Skip if None
            if val2 is None:
                continue
            # Get value for this column
            val = val2.get(col, val2.get("_"))
            # Check for ``None``
            if val is None:
                continue
            # Try to set it
            defn._set_option(opt, val)
        # Output
        return defn

    # Apply defaults to a definition
    def apply_defaults_defn(self, defn):
        r"""Apply any defaults to a data column definition

        This first checks instance options like ``"DefaultType"`` and
        then the global defaults such as ``defn._rc["Type"]``.

        :Call:
            >>> opts.apply_defaults_defn(defn)
        :Inputs:
            *opts*: :class:`BaseDataOpts`
                Options interface for :mod:`cape.attdb.ftypes`
            *defn*: :class:`BaseDataDefn` | *opts._defncls*
                Data column definition
        :Versions:
            * 2020-01-31 ``@ddalle``: First version
        """
        # Get definition class
        defncls = defn.__class__
        # Loop through *default* keys
        for opt in defncls._optlist:
            # Check if already set
            if opt in defn:
                continue
            # Process option name
            opt1 = "Default" + opt
            # Get value
            val1 = self.get_option(opt1)
            # Check for trivial setting
            if val1 is None:
                continue
            # Set it
            defn[opt] = val1
        # Apply defaults internally
        defn.apply_defaults()

   # --- Class Methods ---
    # Add options from a definition
    @classmethod
    def set_defncls(cls, defncls):
        r"""Add all the "default" options from a definition class

        This loops through the options in *defncls._optlist* and adds
        two versions of them to *cls._optlist*:

            * Prefixed with "Default"
            * Suffixed with "s"

        For example if *defncls._optlist* has ``"Type"``, this adds
        ``"DefaultType"`` and ``"Types"`` to *cls._optlist*.

        :Call:
            >>> cls.set_defncls(cls, defncls)
        :Inputs:
            *cls*: :class:`BaseDataOpts`
                Parent options class
            *defncls*: :class:`BaseDataDefn`
                Definition options class
        :Versions:
            * 2020-01-30 ``@ddalle``: First version
        """
        # Get parameters
        _optlist = cls._getattr_class("_optlist")
        _opttypes = cls._getattr_class("_opttypes")
        # Save the definition class
        cls._defncls = defncls
        # Loop through options from definition
        for opt in defncls._optlist:
            # Derivative names
            opt1 = "Default" + opt
            opt2 = opt + "s"
            # Add to set
            _optlist.add(opt1)
            _optlist.add(opt2)
            # Second option must be a dict
            _opttypes.setdefault(opt2, dict)
        # Loop through types
        for (opt, val) in defncls._opttypes.items():
            # Derivative name
            opt1 = "Default" + opt
            # Copy type for "default"
            _opttypes.setdefault(opt1, val)
        
  # >


# Options for generic definition
class BaseDataDefn(kwutils.KwargHandler):
  # ====================
  # Class Attributes
  # ====================
  # <
   # --- Global Options ---
    # List of options
    _optlist = {
        "Label",
        "LabelFormat",
        "Type",
        "WriteFormat"
    }

    # Alternate names
    _optmap = {
        "Format": "LabelFormat"
    }

   # --- Types ---
    # Types
    _opttypes = {
        "Label": bool,
        "LabelFormat": typeutils.strlike,
        "Type": typeutils.strlike,
        "WriteFormat": typeutils.strlike,
    }

   # --- Defaults ---
    # Default values
    _rc = {
        "Label": True,
        "LabelFormat": "%s",
        "Type": "float64",
    }

   # --- Values ---
    # Alternate names for parameters
    _optvalmap = {
        "Type": {
            "double": "float64",
            "c": "complex128",
            "c128": "complex128",
            "c256": "complex256",
            "c64": "complex64",
            "char": "uint8",
            "f": "float64",
            "f16": "float16",
            "f128": "float128",
            "f32": "float32",
            "f64": "float64",
            "i": "int32",
            "i1": "bool",
            "i16": "int16",
            "i32": "int32",
            "i8": "int8",
            "int": "int32",
            "long": "int32",
            "long long": "int64",
            "short": "i16",
            "s": "str",
            "single": "float32",
            "ui": "uint32",
            "ui16": "uint16",
            "ui32": "uint32",
            "ui64": "uint64",
            "ui8": "uint8",
        }
    }

    # Allowed values
    _optvals = {
        "Type": {
            "bool",
            "complex64",
            "complex128",
            "complex256",
            "float16",
            "float32",
            "float64",
            "float128",
            "int8",
            "int16",
            "int32",
            "int64",
            "str",
            "uint8",
            "uint16",
            "uint32",
            "uint64"
        }
    }

   # --- DType ---
    # Map of data types based on *Type*
    _dtypemap = {}

   # --- Documentation ---
        
  # >

  # =================
  # Methods
  # =================
  # <
   # --- DType ---
    # Get data type (from *Type*)
    def get_dtype(self):
        r"""Get (and save) data type (*DType*) based on *Type*

        :Call:
            >>> dtype = defn.get_dtype()
        :Inputs:
            *defn*: :class:`BaseDataDefn`
                Data column definition
        :Outputs:
            *dtype*: :class:`str`
                Data type, looks up *Type* in *defn._dtypemap*
        :Versions:
            * 2020-02-01 ``@ddalle``: First version
        """
        # Check if already present
        if "DType" in self:
            # No need to map
            return self["DType"]
        # Otherwise, get *Type*
        typ = self.get_option("Type")
        # Get data type
        dtype = self.__class__._dtypemap.get(typ, typ)
        # Save it
        if dtype is not None:
            self["DType"] = dtype
        # Output
        return dtype

   # --- Defaults ---
    # Apply defaults from *rc*
    def apply_defaults(self):
        r"""Apply default values from *defn._rc*

        :Call:
            >>> defn.apply_defaults()
        :Inputs:
            *defn*: :class:`BaseDataDefn`
                Data column definition
        :Versions:
            * 2020-01-31 ``@ddalle``: First version
        """
        # Loop through _rc
        for (k, v) in self.__class__._rc.items():
            # Check if already set
            if k in self:
                continue
            # Otherwise; save a copy
            self[k] = copy.copy(v)
  # >


# Add definition support to option
BaseDataOpts.set_defncls(BaseDataDefn)


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
        *db.opts*: :class:`BaseDataOpts` | *db._optscls*
            Options for this instance
        *db.defns*: :class:`dict`\ [:class:`BaseDataDefn`]
            Definitions for each data column
        *db[col]*: :class:`np.ndarray` | :class:`list`
            Numeric array or list of strings for column *col*
    :See also:
        * :class:`cape.attdb.ftypes.csv.CSVFile`
        * :class:`cape.attdb.ftypes.csv.CSVSimple`
        * :class:`cape.attdb.ftypes.textdata.TextDataFile`
    :Versions:
        * 2019-11-26 ``@ddalle``: First version
        * 2020-02-02 ``@ddalle``: Second version
    """
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Options ---
    # Class for options
    _optscls = BaseDataOpts
  # >

  # ==========
  # Config
  # ==========
  # <
    # Template initialization method
    def __init__(self, **kw):
        r"""Initialization method

        :Versions:
            * 2020-02-02 ``@ddalle``: First version
        """
        # Initialize columns
        self.cols = []
        # Process options
        self.opts = self.process_kw(**kw)
        # Ensure definitions are present
        self.get_defns()
        # Apply defaults to definitions
        self.apply_defn_defaults()
        # Process values
        self.process_kw_values()

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
  # Inputs & Kwargs
  # =================
  # <
   # --- Options ---
    # Convert *kw* to options
    def process_kw(self, **kw):
        r"""Process options from keyword arguments

        :Call:
            >>> opts = db.process_kw(**kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *kw*: :class:`dict`
                Arbitrary keyword arguments
        :Outputs:
            *opts*: :class:`BaseDataOpts` | *db._optscls*
                Validated options from *kw*
        :Versions:
            * 2020-02-02 ``@ddalle``: First version
        """
        # Get class
        optscls = self.__class__._optscls
        # Convert kwargs to options; return it
        return optscls(**kw)

    # Query keyword arguments for manual values
    def process_kw_values(self):
        r"""Process *Values* argument for manual column values
        
        :Call:
            >>> db.process_kw_values()
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
        :Options:
            *Values*: :class:`dict`
                Dictionary of values for some columns
            *ExpandScalars*: ``True`` | ``False``
                Option to expand scalars to match dimension of arrays
        :Versions:
            * 2019-11-12 ``@ddalle``: First version
            * 2019-12-31 ``@ddalle``: Removed :func:`pop` and output
            * 2020-02-02 ``@ddalle``: Deleted *kw* as input
        """
        # Get values
        vals = self.get_option("Values", {})
        # Get expansion option
        expand = self.get_option("ExpandScalars", True)
        # Get number for expansion
        n = max(1, self.__dict__.get("n", 1))
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
                coldtype = self.get_col_dtype(col)
                # Check for list-like
                if coldtype == "str":
                    # Create expanded list
                    V = [v] * n
                else:
                    # Create array
                    V = v * np.ones(n, dtype=coldtype)
            else:
                # Use as is
                V = v
            # Save values
            self.save_col(col, V)
            # Save length
            self.n = n
  # >
  
  # =================
  # Options
  # =================
  # <
   # --- Options ---
    # Single option
    def get_option(self, key, vdef=None):
        r"""Get an option, appealing to default if necessary

        :Call:
            >>> val = db.get_option(key, vdef=None)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *key*: :class:`str`
                Name of option to access
            *vdef*: {``None``} | :class:`any`
                Default option for fallback
        :Outputs:
            *val*: *db.opts[key]* | *db.opts._rc[key]* | *vdef*
                Value of option with fallback
        :Versions:
            * 2019-12-31 ``@ddalle``: First version
            * 2020-02-01 ``@ddalle``: Using :class:`BaseDataOpts`
        """
        return self.opts.get_option(key, vdef)
        
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
            *defns*: :class:`dict`\ [:class:`BaseDataDefn`]
                Definitions for each column
        :Versions:
            * 2019-12-31 ``@ddalle``: First version
            * 2020-02-01 ``@ddalle``: Move from ``opts["Definitions"]``
        """
        # Definitions from "defns" attribute
        return self.__dict__.setdefault("defns", {})

    # Get definition for specific column
    def get_defn(self, col):
        r"""Get column definition for data column *col*
        
        :Call:
            >>> defn = db.get_col_prop(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column
        :Outputs:
            *defn*: :class:`BaseDataDefn`
                Definition for column *col*
        :Versions:
            * 2020-02-01 ``@ddalle``: First version
        """
        # Get definition dictionary
        defns = self.get_defns()
        # Check for definition
        if col in defns:
            # Already formulated
            return defns[col]
        # Otherwise, create from options
        defn = self.opts.get_defn(col)
        # Save it
        defns[col] = defn
        # Output
        return defn

    # Apply defaults to a definition
    def apply_defn_defaults(self, cols=None):
        r"""Apply any defaults to data column definitions

        This first checks instance options like ``"DefaultType"`` and
        then the global defaults such as ``defn._rc["Type"]``.

        :Call:
            >>> db.apply_defn_defaults(defn)
        :Inputs:
            *opts*: :class:`BaseDataOpts`
                Options interface for :mod:`cape.attdb.ftypes`
            *defn*: :class:`BaseDataDefn` | *opts._defncls*
                Data column definition
        :Versions:
            * 2020-02-03 ``@ddalle``: First version
        """
        # Default column list
        if cols is None:
            # Use all listed columns
            cols = self.cols
            # If empty, get from option
            if len(cols) == 0:
                # Get *Columns* option
                cols = self.opts.get_option("Columns")
        # Loop through those columns
        for col in cols:
            # Get definition
            defn = self.get_defn(col)
            # Apply defaults
            defn.apply_defaults()

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
            * 2020-02-01 ``@ddalle``: Using :class:`BaseDataDefn`
        """
        # Get specific definition
        defn = self.get_defn(col)
        # Get option from definition
        return defn.get_option(prop, vdef)

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
        # Get definition
        defn = self.get_defn(col)
        # Process *DType* from definition
        return defn.get_dtype()
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
