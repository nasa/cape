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
from ...tnakit import kwutils, typeutils


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
        "Definitions",
        "DefaultDefinition",
        "ExpandScalars",
        "Values",
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
        "Definitions": dict,
        "DefaultDefinition": dict,
        "ExpandScalars": bool,
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
            * 2020-01-31 ``@ddalle``: Version 1.0
        """
        # Get class
        cls = self.__class__
        # Get class for column definitions
        defncls = cls._defncls
        # Get directly-specified definitions
        defns = self.get_option("Definitions", {})
        # Get specific definition and set type
        defn = defns.get(col, {})
        # Remove "DType"
        defn.pop("DType", None)
        # Convert to class
        defn = defncls(defns.get(col, {}), _warnmode=self._warnmode)
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
    def finish_defn(self, defn):
        r"""Apply any defaults to a data column definition

        This first checks instance options like ``"DefaultType"`` and
        then the global defaults such as ``defn._rc["Type"]``.

        :Call:
            >>> opts.finish_defn(defn)
        :Inputs:
            *opts*: :class:`BaseDataOpts`
                Options interface for :mod:`cape.attdb.ftypes`
            *defn*: :class:`BaseDataDefn` | *opts._defncls*
                Data column definition
        :Versions:
            * 2020-01-31 ``@ddalle``: Version 1.0
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
        defn.finish()

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
            * 2020-01-30 ``@ddalle``: Version 1.0
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
        "Dimension",
        "Label",
        "LabelFormat",
        "LongName",
        "Shape",
        "Tag",
        "Type",
        "WriteFormat",
        "Units",
    }

    # Alternate names
    _optmap = {
        "Format": "LabelFormat",
        "dim": "Dimension",
        "longname": "LongName",
        "ndim": "Dimension",
        "shape": "Shape",
        "units": "Units",
    }

   # --- Types ---
    # Types
    _opttypes = {
        "Dimension": int,
        "Label": bool,
        "LabelFormat": typeutils.strlike,
        "LongName": typeutils.strlike,
        "Shape": tuple,
        "Type": typeutils.strlike,
        "WriteFormat": typeutils.strlike,
        "Units": typeutils.strlike,
    }

   # --- Defaults ---
    # Default values
    _rc = {
        "Label": True,
        "LabelFormat": "%s",
        "Type": "float64",
        "Tag": None,
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
            "o": "object",
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
            "object",
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
            * 2020-02-01 ``@ddalle``: Version 1.0
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
    def finish(self):
        r"""Apply default values from *defn._rc*

        :Call:
            >>> defn.finish()
        :Inputs:
            *defn*: :class:`BaseDataDefn`
                Data column definition
        :Versions:
            * 2020-01-31 ``@ddalle``: Version 1.0
            * 2020-03-06 ``@ddalle``: Rename :func:`apply_defaults`
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
        * 2019-11-26 ``@ddalle``: Version 1.0
        * 2020-02-02 ``@ddalle``: Second version
    """
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Tags ---
    # Map for "tag" based on column name
    _tagmap = {}
    # Extra columns expected as kwargs for some tags
    _tagsubcols = {}

   # --- Options ---
    # Class for options
    _optscls = BaseDataOpts
    # Definition class
    _defncls = BaseDataDefn

   # --- Class Functions ---
    # Invert the _tagmap
    @classmethod
    def create_tagcols(cls):
        r"""Invert *cls._tagmap* as *cls._tagcols*

        :Call:
            >>> cls.create_tagcols()
        :Inputs:
            *cls*: :class:`type`
                Data container class
        :Versions:
            * 2020-03-19 ``@ddalle``: Version 1.0
        """
        # Initialize tag -> set(col) map
        _tagcols = {}
        # Loop through _tagmap
        for col, tag in cls._tagmap.items():
            # Get set of cols for current *tag*
            cols = _tagcols.setdefault(tag, set())
            # Add this column to the set
            cols.add(col)
        # Set the attribute
        cls._tagcols = _tagcols
  # >

  # ==========
  # Config
  # ==========
  # <
    # Template initialization method
    def __init__(self, **kw):
        r"""Initialization method

        :Versions:
            * 2020-02-02 ``@ddalle``: Version 1.0
        """
        # Initialize columns
        self.cols = []
        # Process options
        self.opts = self.process_kw(**kw)
        # Ensure definitions are present
        self.get_defns()
        # Apply defaults to definitions
        self.finish_defns()
        # Process values
        self.process_kw_values()

    # Representation method
    def __repr__(self):
        """Generic representation method

        :Versions:
            * 2019-11-08 ``@ddalle``: Version 1.0
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
            * 2019-11-08 ``@ddalle``: Version 1.0
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
            * 2020-02-02 ``@ddalle``: Version 1.0
        """
        # Get class
        optscls = self.__class__._optscls
        # Convert kwargs to options
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
            * 2019-11-12 ``@ddalle``: Version 1.0
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
            * 2019-12-31 ``@ddalle``: Version 1.0
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
            * 2019-12-31 ``@ddalle``: Version 1.0
            * 2020-02-01 ``@ddalle``: Move from ``opts["Definitions"]``
        """
        # Definitions from "defns" attribute
        return self.__dict__.setdefault("defns", {})

    # Get definition for specific column
    def get_defn(self, col):
        r"""Get column definition for data column *col*
        
        :Call:
            >>> defn = db.get_defn(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column
        :Outputs:
            *defn*: :class:`BaseDataDefn`
                Definition for column *col*
        :Versions:
            * 2020-02-01 ``@ddalle``: Version 1.0
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

    # Form a new definition if needed for a given column
    def make_defn(self, col, V, **kw):
        r"""Access or create new definition based on values

        :Call:
            >>> defn = db.make_defn(col, V, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column (used for "Tag" option)
            *V*: :class:`list` | :class:`np.ndarray`
                Values for column *col*
            *kw*: :class:`dict`
                Optional overrides or additions to definition
        :Outputs:
            *defn*: *db._defncls*
                Definition based on values *V*
        :Effects:
            *db[col]*: *defn*
        :Versions:
            * 2020-03-19 ``@ddalle``: Version 1.0
            * 2020-06-24 ``@ddalle``: Version 1.1; merge defns
        """
        # Attempt to get definition
        defns = self.get_defns()
        # Generate a new definition based on values
        defn0 = self.genr8_defn(col, V, **kw)
        # Return it if any
        if col in defns:
            # Get the definition
            defn = defns[col]
            # Merge columns
            for k, v in defn0.items():
                # Apply but don't overwrite
                defn.setdefault(k, v)
        else:
            # Save it
            self.defns[col] = defn0
            # Transfer it
            defn = defn0
        # Output
        return defn

    # Form a new definition for a given column
    def create_defn(self, col, V, **kw):
        r"""Create and save a new definition based on values

        :Call:
            >>> defn = db.create_defn(col, V, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column (used for "Tag" option)
            *V*: :class:`list` | :class:`np.ndarray`
                Values for column *col*
            *kw*: :class:`dict`
                Optional overrides or additions to definition
        :Outputs:
            *defn*: *db._defncls*
                Definition based on values *V*
        :Effects:
            *db[col]*: *defn*
        :Versions:
            * 2020-03-19 ``@ddalle``: Version 1.0
        """
        # Create the definition
        defn = self.genr8_defn(col, V, **kw)
        # Save it
        self.defns[col] = defn
        # Output
        return defn

    # Create a definition for a given column
    def genr8_defn(self, col, V, **kw):
        r"""Generate a new definition based on values

        :Call:
            >>> defn = db.genr8_defn(col, V, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column (used for "Tag" option)
            *V*: :class:`list` | :class:`np.ndarray`
                Values for column *col*
            *kw*: :class:`dict`
                Optional overrides or additions to definition
        :Outputs:
            *defn*: *db._defncls*
                Definition based on values *V*
        :Versions:
            * 2020-03-19 ``@ddalle``: Version 1.0
        """
        # Get definition from values (and kwargs)
        defn = self._genr8_defn(V, **kw)
        # Check for tag
        if not defn.get("Tag"):
            # Get default tag
            tag = self._tagmap.get(col.split(".")[-1])
            # If valid tag, set it
            if tag:
                defn.set_option("Tag", tag)
        # Output
        return defn

    # Create a definition for a given column
    def _genr8_defn(self, V, **kw):
        r"""Generate a new definition based on values

        :Call:
            >>> defn = db._genr8_defn(V, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *V*: :class:`list` | :class:`np.ndarray`
                Values for column *col*
            *kw*: :class:`dict`
                Optional overrides or additions to definition
        :Outputs:
            *defn*: *db._defncls*
                Definition based on values *V*
        :Versions:
            * 2020-03-19 ``@ddalle``: Version 1.0
        """
        # Initialize definition
        defn = self._defncls()
        # Determine a default warning mode
        opts = self.__dict__.get("opts", {})
        _warnmode = getattr(opts, "_warnmode", 1)
        # Set the warningmode
        defn._warnmode = kw.get("_warnmode", _warnmode)
        # Check values
        if isinstance(V, list):
            # Assume string
            dtype = "str"
            # Save length and dimension
            defn.set_option("Dimension", 1)
            defn.set_option("Shape", (len(V), ))
        elif isinstance(V, np.ndarray):
            # Array; get data type from instance
            dtype = V.dtype.name
            # Dimensions
            defn.set_option("Dimension", V.ndim)
            defn.set_option("Shape", V.shape)
            # Check for strings (convert to list of strings)
            if dtype.startswith("str"):
                # Regular strings (Python 2 only)
                dtype = "str"
            elif dtype.startswith("unicode"):
                # Strings using Python 2 "unicode" or Python 3 "str"
                dtype = "str"
        elif V.__class__.__module__ == "numpy":
            # Scalar NumPy object
            dtype = V.__class__.__name__
        elif isinstance(V, float):
            # Float (64)
            dtype = "float64"
        elif isinstance(V, typeutils.intlike):
            # Integer (pretend we know the type)
            if typeutils.PY_MAJOR_VERSION > 2:
                # Long
                dtype = "int64"
            elif V.__class__.__name__ == "long":
                # Long (Python 2)
                dtype = "int64"
            else:
                # Regular int (Python 2)
                dtype = "int32"
        else:
            # Unrecognized
            raise TypeError(
                "Could not generate definition for type '%s'" % type(V))
        # Set type
        defn["Type"] = dtype
        
        # Loop through any kwargs
        for k, v in kw.items():
            # Attempt to set it (_warnmode is 1)
            defn.set_option(k, v)
        # Output
        return defn

    # Apply defaults to a definition
    def finish_defns(self, cols=None):
        r"""Apply any defaults to data column definitions

        This first checks instance options like ``"DefaultType"`` and
        then the global defaults such as ``defn._rc["Type"]``.

        :Call:
            >>> db.finish_defns(cols=None)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2020-02-03 ``@ddalle``: Version 1.0
            * 2020-03-31 ``@ddalle``: Handled *db.opts* properly
        """
        # Default column list
        if cols is None:
            # Use all listed columns
            cols = self.cols
            # If empty, get from option
            if len(cols) == 0:
                # Get *Columns* option
                cols = self.opts.get_option("Columns", [])
        # Loop through those columns
        for col in cols:
            # Individual-column function
            self.finish_defn(col)

    # Apply any defaults to one col's definition
    def finish_defn(self, col):
        r"""Apply any defaults to a data column definition

        This first checks instance options like ``"DefaultType"`` and
        then the global defaults such as ``defn._rc["Type"]``.

        :Call:
            >>> db.finish_defn(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Data column name
        :Versions:
            * 2020-03-31 ``@ddalle``: Split from :func:`finish_defns`
        """
        # Get definition
        defn = self.get_defn(col)
        # Apply values from *opts*
        self._finish_defn_opts(col, defn)
        # Apply defaults
        defn.finish()
        # Apply default tag
        self.apply_defn_tag(col)

    # Apply defaults from *self.opts* to a definition
    def _finish_defn_opts(self, col, defn):
        r"""Apply any defaults from *opts* to partial definition

        :Call:
            >>> db._finish_defn_opts(col, defn)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *cols*: :class:`str`
                Data column name
            *defn*: *db._optscls._defncls*
                Partial definition
        :Versions:
            * 2020-03-31 ``@ddalle``: Version 1.0
        """
        # Options and definition class
        optscls = self._optscls
        defncls = optscls._defncls
        # Option list from options
        optlist = optscls._optlist
        # Options handle
        opts = self.opts
        # Loop through definition parameters
        for opt in defncls._optlist:
            # Check if already set (in which case ignore)
            if opt in defn:
                continue
            # Combined option names
            opt1 = "Default" + opt
            opt2 = opt + "s"
            # Check for dictionary of *opt* (like "Types")
            if (opt2 in optlist) and (opt2 in opts):
                # Dictionary of *opt* for each *col*
                d2 = opts[opt2]
                # Get value for this *col*
                val2 = d2.get(col, d2.get("_"))
                # Check if valid
                if val2 is not None:
                    # Set it
                    defn[opt] = val2
            # Check for *Default* value
            if (opt1 in optlist) and (opt1 in opts):
                # Copy value
                defn[opt] = copy.copy(opts[opt1])

    # Apply all default tags
    def apply_defns_tag(self, cols=None):
        r"""Apply all default *Tag* properties based on col name

        :Call:
            >>> db.apply_defns_tag(cols=None)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2020-03-18 ``@ddalle``: Version 1.0
        """
        # Default column list
        if cols is None:
            # Use all listed columns
            cols = self.cols
            # If empty, get from option
            if len(cols) == 0:
                # Get *Columns* option
                cols = self.opts.get_option("Columns", [])
        # Loop through columns
        for col in cols:
            # Apply default tag
            self.apply_defn_tag(col)

    # Apply default tags
    def apply_defn_tag(self, col, tagdef=None):
        r"""Apply default *Tag* to each definition

        :Call:
            >>> db.apply_defn_tag(col, tagdef=None)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column for which to set default tag
            *tagdef*: {``None``} | :class:`str`
                Manually specified default tag
        :Versions:
            * 2020-03-18 ``@ddalle``: Version 1.0
        """
        # Check for specified tag
        if tagdef is None:
            # Check for a default tag
            tagdef = self._tagmap.get(col.split(".")[-1])
        # If no default, exit
        if not tagdef:
            return
        # Get definition
        defn = self.get_defn(col)
        # Get current tag
        tag = defn.get("Tag")
        # If there is a tag, do nothing
        if tag:
            return
        # Otherwise set tag
        defn["Tag"] = tagdef

    # Get columns by tag
    def get_col_by_tag(self, tag, coldef=None):
        r"""Return the first *col* with specified "Tag", if any

        :Call:
            >>> col = db.get_col_by_tag(tag)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *tag*: :class:`str`
                Target "Tag" from column definitions
            *coldef*: {``None``} | :class:`str`
                Default column name
        :Outputs:
            *col*: ``None`` | :class:`str`
                Name of column for which to set default tag
        :Versions:
            * 2020-03-18 ``@ddalle``: Version 1.0
        """
        # Loop through columns
        for col in self.cols:
            # Get definition
            defn = self.get_defn(col)
            # Get the tag
            coltag = defn.get("Tag")
            # Check match
            if coltag == tag:
                # Match
                return col
        # Otherwise return default column name
        return coldef

    # Get all columns by tag
    def get_cols_by_tag(self, tag):
        r"""Return all *col* with specified "Tag"

        :Call:
            >>> cols = db.get_cols_by_tag(tag)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *tag*: :class:`str`
                Target "Tag" from column definitions
        :Outputs:
            *cols*: :class:`list`\ [:class:`str`]
                Name of column for which to set default tag
        :Versions:
            * 2020-03-18 ``@ddalle``: Version 1.0
        """
        # Initialize list
        cols = []
        # Loop through columns
        for col in self:
            # Get definition
            defn = self.get_defn(col)
            # Get the tag
            coltag = defn.get("Tag")
            # Check match
            if coltag == tag:
                # Append to list
                cols.append(col)
        # Output
        return cols

   # --- Column Properties ---
    # Set generic property from column
    def set_col_prop(self, col, prop, v):
        r"""Set property for specific column

        :Call:
            >>> v = db.set_col_prop(col, prop, v, vdef=None)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column
            *prop*: :class:`str`
                Name of property
            *v*: :class:`any`
                Value to set to ``defns[col][prop]``
        :Versions:
            * 2021-01-22 ``@aburkhea``: Version 1.0
        """
        # Get definition
        defn = self.get_defn(col)
        # Set property
        defn.set_option(prop,v)

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
            * 2019-11-24 ``@ddalle``: Version 1.0
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
            * 2019-11-24 ``@ddalle``: Version 1.0
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
            * 2019-11-24 ``@ddalle``: Version 1.0
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
            * 2020-02-14 ``@ddalle``: Tweak rules for *cols* append
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
        # Basic definition
        self.make_defn(col, v)

   # --- Keep Only Some Data ---
    # Remove all columns outside specified list
    def keeponly_cols(self, cols):
        r"""Remove all columns outside specified list

        :Call:
            >>> db.keeponly_cols(cols)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *cols*: :class:`list` | :class:`set`
                Name of columns to **keep**
        :Versions:
            * 2020-12-21 ``@ddalle``: Version 1.0
        """
        # Ensure input types
        if not isinstance(cols, (list, tuple, set)):
            raise TypeError(
                "Input 'cols' has type '%s' (required 'list')" % type(cols))
        # Loop through existing cols
        for col in self.cols:
            # Check if it's worth keeping
            if col in cols:
                # Keep it
                continue
            # Otherwise, remove it and its settings/options
            self.burst_col(col)

   # --- Remove Data ---
    # Remove a column and its parameters
    def burst_col(self, col):
        r"""Remove a column and its definition is possible

        :Call:
            >>> V = db.burst_col(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basedata.BaseData`
                Data container
            *col*: :class:`str`
                Name of column
        :Outputs:
            *V*: :class:`np.ndarray` | :class:`list` | :class:`scalar`
                Value(s) to save for specified column
        :Versions:
            * 2020-03-19 ``@ddalle``: Version 1.0
        """
        # Check if column is present
        if col in self:
            # Get values
            V = self.pop(col)
        else:
            # No values
            V = None
        # Get definitions
        defns = self.get_defns()
        # Check if present
        if col in defns:
            # Remove it
            defns.pop(col)
        # Check if in list
        if col in self.cols:
            self.cols.remove(col)
        # Output
        return V

   # --- Name Translation ---
    # Rename a column
    def rename_col(self, col1, col2):
        r"""Rename a column *col1* to *col2*

        :Call:
            >>> db.rename_col(col1, col2)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col1*: :class:`str`
                Existing column name
            *col2*: :class:`str`
                New column name
        :Versions:
            * 2021-07-09 ``@ddalle``: Version 1.0
        """
        # Get definitions
        defns = self.get_defns()
        # Check if present
        if col1 in defns:
            # Remove it and save it with new name
            defns[col2] = defns.pop(col1)
        # Check if in list
        if col1 in self.cols:
            # Get index
            i = self.cols.index(col1)
            # Overwrite
            self.cols[i] = col2
        else:
            # Just add new column
            self.cols.append(col2)
        # Check if column is present
        if col1 in self:
            # Get values
            v = self.pop(col1)
            # Save the new column (creates defn if needed)
            self.save_col(col2, v)
            # Get new definition for customization
            defn = defns[col2]
            # Set value for *LongName* (but don't overwrite)
            defn.setdefault_option("LongName", col1)
        else:
            # No values
            v = None

    # Translate column names
    def _translate_colnames(self, cols, trans, prefix, suffix):
        r"""Translate column names

        :Call:
            >>> dbcols = db._translate_colnames(cols, |args|)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *cols*: :class:`list`\ [:class:`str`]
                List of "original" column names, e.g. from file
            *trans*: :class:`dict`\ [:class:`str`]
                Alternate names; *col* -> *trans[col]*
            *prefix*: :class:`str` | :class:`dict`
                Universal prefix or *col*-specific prefixes
            *suffix*: :class:`str` | :class:`dict`
                Universal suffix or *col*-specific suffixes
        :Outputs:
            *dbcols*: :class:`list`\ [:class:`str`]
                List of column names as stored in *db*
        :Versions:
            * 2019-12-04 ``@ddalle``: Version 1.0
            * 2020-02-22 ``@ddalle``: Stole content from main function

        .. |args| replace:: trans, prefix, suffix
        """
        # Translate each col name
        return [
            self._translate_colname(col, trans, prefix, suffix)
            for col in cols
        ]

    # Reverse translation of column names
    def _translate_colnames_reverse(self, dbcols, trans, prefix, suffix):
        r"""Reverse translation of column names

        :Call:
            >>> cols = db._translate_colnames_reverse(dbcols, |args|)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *dbcols*: :class:`list`\ [:class:`str`]
                List of column names as stored in *db*
            *trans*: :class:`dict`\ [:class:`str`]
                Alternate names; *col* -> *trans[col]*
            *prefix*: :class:`str` | :class:`dict`
                Universal prefix or *col*-specific prefixes
            *suffix*: :class:`str` | :class:`dict`
                Universal suffix or *col*-specific suffixes
        :Outputs:
            *cols*: :class:`list`\ [:class:`str`]
                List of "original" column names, e.g. from file
        :Versions:
            * 2019-12-04 ``@ddalle``: Version 1.0
            * 2019-12-11 ``@jmeeroff``: From :func:`translate_colnames`
            * 2020-02-22 ``@ddalle``: Moved code from main function

        .. |args| replace:: trans, prefix, suffix
        """
        # Initialize output
        cols = []
        # Reverse the translation dictionary
        transr = {dbcol: col for (col, dbcol) in trans.items()}
        # Prepare output
        for dbcol in dbcols:
            # First strip prefix
            # Get prefix
            if isinstance(prefix, dict):
                # Get specific prefix
                pre = prefix.get(dbcol, prefix.get("_", ""))
            elif prefix:
                # Universal prefix
                pre = prefix
            else:
                # No prefix (type-safe)
                pre = ""
            # Check if col starts with specified prefix
            if pre and dbcol.startswith(pre):
                # Get length of prefix
                lval = len(pre)
                # Strip of prefix
                dbcol = dbcol[lval:]
            # Now strip suffix
            # Get suffix
            if isinstance(suffix, dict):
                # Get specific suffix
                suf = suffix.get(dbcol, suffix.get("_", ""))
            elif suffix:
                # Universal suffix
                suf = suffix
            else:
                # No suffix (type-safe)
                suf = ""
            # Check if column name ends with suffix
            if suf and dbcol.endsiwth(suf):
                # Get length of suf
                lval = len(suf)
                # Strip of prefix
                dbcol = dbcol[:-lval]
            # Perform the translation
            col = transr.get(dbcol, dbcol)
            # Append to output
            cols.append(col)
        # Output
        return cols

    # Translate column names
    def _translate_colname(self, col, trans, prefix, suffix):
        r"""Translate column name

        :Call:
            >>> dbcol = db._translate_colname(col, |args|)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                "Original" column name, e.g. from file
            *trans*: :class:`dict`\ [:class:`str`]
                Alternate names; *col* -> *trans[col]*
            *prefix*: :class:`str` | :class:`dict`
                Universal prefix or *col*-specific prefixes
            *suffix*: :class:`str` | :class:`dict`
                Universal suffix or *col*-specific suffixes
        :Outputs:
            *dbcol*: :class:`str`
                Column names as stored in *db*
        :Versions:
            * 2019-12-04 ``@ddalle``: Version 1.0
            * 2020-02-22 ``@ddalle``: Single-column version

        .. |args| replace:: trans, prefix, suffix
        """
        # Get substitution (default is no substitution)
        dbcol = trans.get(col, col)
        # Get prefix
        if isinstance(prefix, dict):
            # Get specific prefix
            pre = prefix.get(col, prefix.get("_", ""))
        elif prefix:
            # Universal prefix
            pre = prefix
        else:
            # No prefix (type-safe)
            pre = ""
        # Get suffix
        if isinstance(suffix, dict):
            # Get specific suffix
            suf = suffix.get(col, suffix.get("_", ""))
        elif suffix:
            # Universal suffix
            suf = suffix
        else:
            # No suffix (type-safe)
            suf = ""
        # Combine fixes
        return pre + dbcol + suf

    # Reverse translation of column names
    def _translate_colname_reverse(self, dbcol, transr, prefix, suffix):
        r"""Reverse translation of column name

        :Call:
            >>> col = db._translate_colname_reverse(dbcol, |args|)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *dbcol*: :class:`str`
                Column names as stored in *db*
            *transr*: :class:`dict`\ [:class:`str`]
                Alternate names; *dbcol* -> *trans[dbcol]*
            *prefix*: :class:`str` | :class:`dict`
                Universal prefix or *col*-specific prefixes
            *suffix*: :class:`str` | :class:`dict`
                Universal suffix or *col*-specific suffixes
        :Outputs:
            *col*: :class:`str`
                "Original" column name, e.g. from file
        :Versions:
            * 2019-12-04 ``@ddalle``: Version 1.0
            * 2019-12-11 ``@jmeeroff``: From :func:`translate_colnames`
            * 2020-02-22 ``@ddalle``: Single-column version

        .. |args| replace:: transr, prefix, suffix
        """
        # First strip prefix
        # Get prefix
        if isinstance(prefix, dict):
            # Get specific prefix
            pre = prefix.get(dbcol, prefix.get("_", ""))
        elif prefix:
            # Universal prefix
            pre = prefix
        else:
            # No prefix (type-safe)
            pre = ""
        # Check if col starts with specified prefix
        if pre and dbcol.startswith(pre):
            # Get length of prefix
            lval = len(pre)
            # Strip of prefix
            dbcol = dbcol[lval:]
        # Now strip suffix
        # Get suffix
        if isinstance(suffix, dict):
            # Get specific suffix
            suf = suffix.get(dbcol, suffix.get("_", ""))
        elif suffix:
            # Universal suffix
            suf = suffix
        else:
            # No suffix (type-safe)
            suf = ""
        # Check if column name ends with suffix
        if suf and dbcol.endsiwth(suf):
            # Get length of suf
            lval = len(suf)
            # Strip of prefix
            dbcol = dbcol[:-lval]
        # Check for alternate name
        return transr.get(dbcol, dbcol)
  # >
# class BaseData
