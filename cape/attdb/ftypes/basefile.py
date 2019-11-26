#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.ftypes.basefile`: Common ATTDB file type attributes
=====================================================================

This module provides the class :class:`BaseFile` as a subclass of
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

        isinstance(db, cape.attdb.ftypes.BaseFile)
"""

# Standard library modules
import os

# Third-party modules
import numpy as np


# Fixed parameter for size of new chunks
NUM_ARRAY_CHUNK = 5000


# Declare basic class
class BaseFile(dict):
    # Common properties
    cols = []
    fname = ""
    lines = []
    linenos = []
    opts = {}
    n = 0

  # ==========
  # Config
  # ==========
  # <
    # Representation method
    def __repr__(self):
        """Generic representation method

        :Versions:
            * 2019-11-08 ``@ddalle``: First version
        """
        # Module name
        modname = self.__class__.__module__
        clsname = self.__class__.__name__
        # Start output
        lbl = "<%s.%s(" % (modname, clsname)
        # Append file name if appropriate
        if self.fname:
            lbl += "'%s', " % os.path.basename(self.fname)
        # Append count
        if self.n:
            lbl += "n=%i, " % self.n
        # Append columns
        if len(self.cols) <= 6:
            # Show all columns
            lbl += "cols=%s)" % str(self.cols)
        else:
            # Just show number of columns
            lbl += "ncol=%i)" % len(self.cols)
        # Output
        return lbl

    # String method
    def __str__(self):
        """Generic representation method

        :Versions:
            * 2019-11-08 ``@ddalle``: First version
        """
        # Module name
        modname = self.__class__.__module__
        clsname = self.__class__.__name__
        # Start output
        lbl = "<%s.%s(" % (modname, clsname)
        # Append file name if appropriate
        if self.fname:
            lbl += "'%s', " % os.path.basename(self.fname)
        # Append count
        if self.n:
            lbl += "n=%i, " % self.n
        # Append columns
        if len(self.cols) <= 6:
            # Show all columns
            lbl += "cols=%s)" % str(self.cols)
        else:
            # Just show number of columns
            lbl += "ncol=%i)" % len(self.cols)
        # Output
        return lbl
  # >
  
  # =================
  # Options
  # =================
  # <
   # --- Key Definitions ---
    # Process key definitions
    def process_col_types(self, **kw):
        r"""Process *Definitions* of column types
        
        :Call:
            >>> kwo = db.process_col_types(**kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of columns to process
            *Types*: {``{}``} | :class:`dict`
                Dictionary of just tye *Type* for one or more cols
            *Definitions*, *defns*: {``{}``} | :class:`dict`
                Dictionary of specific definitions for each *col*
            *DefaultType*: {``"float"``} | :class:`str`
                Name of default class
            *DefaultFormat*: {``None``} | :class:`str`
                Optional default format string
            *DefaultDefinition*: :class:`dict`
                :class:`dict` of default *Type*, *Format*
        :Outputs:
            *kwo*: :class:`dict`
                Options not used in this method
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
            * 2014-06-17 ``@ddalle``: Read from *defns* :class:`dict`
            * 2019-11-12 ``@ddalle``: Copied from :class:`RunMatrix`
        """
        # Get options for key definitions
        defns1 = kw.pop("Types", {})
        defns2 = kw.pop("Definitions", {})
        defns3 = kw.pop("defns", {})
        # Combine definitions
        defns = dict(defns3, **defns2)
        # Process current list of columns
        cols = getattr(self, "cols", [])
        # Check for default definition
        odefn = kw.pop("DefaultDefinition", {})
        # Process various defaults
        odefcls = kw.pop("DefaultType", "float")
        odeffmt = kw.pop("DefaultFormat", None)
        # Set defaults
        odefn.setdefault("Type",  odefcls)
        odefn.setdefault("Format", odeffmt)
        # Validate default definition
        self.validate_defn(odefn)
        # Ensure definitions exist
        opts = self.opts.setdefault("Definitions", {})
        # Save defaults
        self.opts["Definitions"]["_"] = odefn
        
        # Loop through columns mentioned in input
        for (col, kwdefn) in defns.items():
            # Get existing definition
            defn = opts.setdefault(col, {})
            # Apply keyword definitions
            for (key, opt) in kwdefn.items():
                # Kwargs override anything created automatically
                defn[key] = opt
            # Validate values
            self.validate_defn(defn)

        # Loop through specifically types
        for (col, cls) in defns1.items():
            # Get existing definition
            defn = opts.setdefault(col, {})
            # Apply the type
            defn["Type"] = self.validate_dtype(cls)

        # Loop through known columns
        for col in self.cols:
            # Get definition
            defn = opts.setdefault(col, {})
            # Loop through default keys
            for key, opt in odefn.items():
                # Apply default but don't override
                defn.setdefault(key, opt)
            # Validate the definition
            self.validate_defn(defn)
            
        # Return unused options
        return kw

   # --- Keyword Checkers ---
    # Validate a dictionary of options
    def validate_defn(self, defn):
        r"""Validate each key in a dictionary column definition
        
        :Call:
            >>> db.validate_defn(defn)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *defn*: :class:`dict`
                Name of column definition option to validate
        :Effects:
            *defn*: :class:`dict`
                Each item in *defn* is validated
        :See Also:
            * :func:`validate_defnopt`
        :Versions:
            * 2019-11-26 ``@ddalle``: First version
        """
        # Ensure input type
        if not isinstance(defn, dict):
            raise TypeError("Definition for validation must be 'dict'" +
                ("; got '%s'" % defn.__class__.__name__))
        # Loop through keys
        for (k, v) in defn.items():
            # Validate individual key
            defn[k] = self.validate_defnopt(k, v)
        
    # Validate any keyword argument
    def validate_defnopt(self, prop, val):
        r"""Translate any key definition into validated output
        
        :Call:
            >>> v = db.validate_defnopt(prop, val)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *prop*: :class:`str`
                Name of column definition option to validate
            *val*: :class:`any`
                Initial value for option (raw input)
        :Outputs:
            *v*: :class:`any`
                Validated version of *val*
        :Versions:
            * 2019-11-26 ``@ddalle``: First version
        """
        # Check property
        if prop == "Type":
            return self.validate_dtype(val)
        else:
            # Default is to accept any input
            return val

    # Convert *Type* to validated *Type*
    def validate_dtype(self, clsname):
        r"""Translate free-form type name into type code
        
        :Call:
            >>> dtype = db.validate_dtype(clsname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *clsname*: :class:`str`
                Name of column
        :Outputs:
            *dtype*: ``"f64"`` | ``"i32"`` | ``"str"`` | :class:`str`
                Name of data type
        :Versions:
            * 2019-11-24 ``@ddalle``: First version
        """
        # Force lower case
        clsname = clsname.lower()
        # Make some substitutions
        clsname = clsname.replace("float", "f")
        clsname = clsname.replace("int",  "i")
        clsname = clsname.replace("complex", "c")
        # Filter it
        if clsname in ["f", "f64", "double"]:
            # 64-bit float (default
            return "float64"
        elif clsname in ["i", "i32", "long"]:
            # 32-bit int
            return "int32"
        elif clsname in ["i16", "short"]:
            # 16-bit int
            return "int16"
        elif clsname in ["f32", "single"]:
            # 32-bit float
            return "float32"
        elif clsname in ["i64", "long long"]:
            # Double long integer
            return "int64"
        elif clsname in ["f128"]:
            # Double long float
            return "float128"
        elif clsname in ["f16"]:
            # Short float
            return "float16"
        elif clsname in ["i8"]:
            # Extra short integer
            return "int8"
        elif clsname in ["i1", "bool"]:
            # Boolean
            return "bool"
        elif clsname in ["c", "c128"]:
            # Complex (double)
            return "complex128"
        elif clsname in ["c", "c64"]:
            # Complex (double)
            return "complex64"
        elif clsname in ["c256"]:
            # Complex (single)
            return "complex256"
        elif clsname in ["str"]:
            # String
            return "str"
        else:
            # Unrecognized
            return TypeError("Unrecognized class/type '%s'" % clsname)
    
   # --- Column Properties ---
    # Get generic property from column
    def get_col_prop(self, col, prop):
        """Get property for specific column
        
        :Call:
            >>> val = db.get_col_prop(col, prop)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column
            *prop*: :class:`str`
                Name of property
        :Outputs:
            *val*: :class:`any`
                Value of ``db.opts["Definitions"][col][prop]`` if
                possible; defaulting to
                ``db.opts["Definitions"]["_"][prop]`` or ``None``
        :Versions:
            * 2019-11-24 ``@ddalle``: First version
        """
        # Check if column is present
        if col not in self.cols:
            # Allow default
            if col != "_":
                raise KeyError("No column '%s'" % col)
        # Get definitions
        defns = self.opts.get("Definitions", {})
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
            return defn.get(prop)

    # Get type
    def get_col_type(self, col):
        """Get data type for specific column
        
        :Call:
            >>> cls = db.get_col_type(col, prop)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column
        :Outputs:
            *cls*: ``"int"`` | ``"float"`` | ``"str"`` | :class:`str`
                Name of data type
        :Versions:
            * 2019-11-24 ``@ddalle``: First version
        """
        return self.get_col_prop(col, "Type")
        
    # Get array type
    

   # --- Keyword Values ---
    # Query keyword arguments for manual values
    def process_values(self, **kw):
        r"""Process *Values* argument for manual column values
        
        :Call:
            >>> kw = db.process_values(**kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *Values*, *vals*: :class:`dict`
                Dictionary of values for some columns
        :Outputs:
            *kwo*: :class:`dict`
                Options not used in this method
        :Versions:
            * 2019-11-12 ``@ddalle``: First version
        """
        # Get values
        vals1 = kw.pop("Values", {})
        vals2 = kw.pop("vals", {})
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
        for col, v in kw.items():
            # Save values
            self.save_column(col, v)
            
  # >
  
  
  # ===============
  # Data
  # ===============
  # <
   # --- Init ---
    # Initialize single column
    def init_col(self, col):
        r"""Initialize column
        
        :Call:
            >>> db.init_col(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column to initialize
        :Effects:
            *db[col]*: :class:`np.ndarray` | :class:`list`
                Initialized array with appropriate type
        :Versions:
            * 2019-11-23 ``@ddalle``: First version
        """
        # Check validity
        if col not in self.cols:
            raise KeyError("Unrecognized column '%s'" % col)
        # Get type
        clsname = self.get_col_type(col)
        # Make sure _nmax (array length) attribute is present
        if not hasattr(self, "_nmax"):
            self._nmax = {}
        # Make sure _n (current length) attribute is present
        if not hasattr(self, "_n"):
            self._n = {}
        # Check for string
        if clsname == "str":
            # Initialize strings in empty list
            self[col] = []
            # No max length
            self._n[col] = 0
            self._nmax[col] = None
        else:
            # Use existing dtype code
            self[col] = np.zeros(NUM_ARRAY_CHUNK, dtype=clsname)
            # Set max length
            self._n[col] = 0
            self._nmax[col] = NUM_ARRAY_CHUNK
        
    # Initialize list of columns
    def init_cols(self, cols):
        r"""Initialize list of columns
        
        :Call:
            >>> db.init_cols(cols)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column to initialize
        :See Also:
            * :func:`init_col`
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Loop through columns
        for col in cols:
            # Initialize column
            self.init_col(col)

   # --- Save Data ---
    # Save a column
    def save_column(self, col, v):
        r"""Save a column value, updating other metadata as needed
        
        :Call:
            >>> db.save_column(col, v)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
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
            self[k] = v
        elif isinstance(v, list):
            # Check first element
            if len(v) == 0:
                # Nothing to convert
                self[k] = v
            elif isinstance(v[0], (int, float, complex)):
                # Convert to array
                self[k] = np.asarray(v)
        else:
            # Nonstandard value; don't convert
            self[k] = v

    # Save next value to column's array
    def append_colval(self, col, v):
        """Save the next value to a column's array or list
        
        This will update counts and allocate a new chunk if necessary.
        
        :Call:
            >>> db.init_col(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column to which to save value
            *v*: ``db.get_col_type(col)``
                Value to save to array/list
        :Effects:
            *db[col]*: :class:`np.ndarray` | :class:`list`
                Column's array with extra new entry
            *db._n[col]*: :class:`int`
                Updated length of array/list
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Check NMAX attribute to check process
        nmax = self._nmax.get(col)
        # Get current count
        n = self._n[col]
        # Process options
        if nmax is None:
            # It's a list; just append
            self[col].append(v)
        elif n >= nmax:
            # Get dtype
            clsname = self.get_col_type(col)
            # Allocate new chunk
            self[col] = np.hstack(
                (self[col], np.zeros(NUM_ARRAY_CHUNK, dtype=clsname)))
            # Update maximum
            self._nmax[col] += NUM_ARRAY_CHUNK
            # Save new value
            self[col][n] = v
        else:
            # Save new value without new allocation
            self[col][n] = v
        # Update count
        self._n[col] = n + 1

    # Trim columns
    def trim_colarray(self, col):
        r"""Trim extra entries from data rows
        
        :Call:
            >>> db.trim_colarray(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column to which to save value
        :Effects:
            *db[col]*: :class:`np.ndarray` | :class:`list`
                Trimmed to length *db._n[col]* if an array
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Check NMAX attribute to check process
        nmax = self._nmax.get(col)
        # Get current count
        n = self._n.get(col)
        # Check for invalid length
        if not isinstance(n, int):
            raise TypeError("No valid length to trim column '%s'" % col)
        # Process options
        if nmax is None:
            # No trimming needed
            return
        else:
            # Trim the array
            self[col] = self[col][:n]
            # Trim *nmax*
            self._nmax[col] = n
        
            
  # >

  # ===============
  # Attributes
  # ===============
  # <
    # Save a value as an attribute (risky)
    def register_attribute(self, col):
        """Register a data field as an attribute
        
        For example, if *col* is ``"mach"``, this will create
        *db.mach*, which will be a reference to ``db["mach"]``.
        
        :Call:
            >>> db.register_attribute(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of existing column
        :Versions:
            * 2019-11-10 ``@ddalle``: First version
        """
        # Check if column is present
        if col not in self.cols:
            raise KeyError("Column '%s' not in database" % col)
        # Create pointer
        setattr(self, col, self[col])
  # >