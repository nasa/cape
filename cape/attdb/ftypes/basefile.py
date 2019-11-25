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
            *cols*: {*db.cols*} | :class:`list`\ [:lass:`str`]
                List of columns to process
            *Types*, *Definitions*, *defns*: {``{}``} | :class:`dict`
                Dictionary of specific types for each *col*
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
        defns = dict(dict(defns3, **defns2), **defns3)
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
        # Ensure definitions exist
        opts = self.opts.setdefault("Definitions", {})
        # Save defaults
        self.opts["Definitions"]["_"] = odefn
        # Loop through columns
        for col in cols:
            # Get definition
            defn = defns.get(col, odefn)
            # Loop through default keys
            for key, opt in odefn.items():
                # Apply default but don't override
                defn.setdefault(key, opt)
            # Set definition
            opts[col] = defn
        # Return unused options
        return kw

   # --- Type/Class Translators ---
    # Convert class names
    def translate_classname(self, clsname):
        """Convert class name into abbreviation code
        
        This function serves a similar purpose to the NumPy function
        :func:`np.dtype`.  The base types are ``int``, ``uint``, 
        ``float``, and ``str``.
        
        :Call:
            >>> typname = db.translate_classname(clsname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *clsname*: :class:`str`
                Name of class, for example ``"float"``, ``"f64"``,
                ``"uint16"``, etc.
        :Outputs:
            *typname*: :class:`str`
                Full descriptive name, ``"int32"``, ``"float64"``, etc
        :Versions:
            * 2019-11-24 ``@ddalle``: First version
        """
        pass
    
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
        retrun self.get_col_prop(col, "Type")
        
    # Get array type
    def get_col_dtype(self, col):
        """Get data type for arrays for specific column
        
        :Call:
            >>> dtype = db.get_col_dtype(col, prop)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column
        :Outputs:
            *dtype*: ``"f64"`` | ``"i32"`` | ``"str"`` | :class:`str`
                Name of data type
        :Versions:
            * 2019-11-24 ``@ddalle``: First version
        """
        # Get input type
        clsname = self.get_col_dtype(col)
        # Filter it
        if col in ["f", "f64", "float", "float64", "double"]:
            # 64-bit float (default
            return "f64"
        elif col in ["i", "i32"]:
            # 32-bit int
  # >

  # ==========
  # Values
  # ==========
  # <
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
            
  # >
  
  
  # ===============
  # Data
  # ===============
  # <
   # --- Init ---
    def initcol(self, col):
        """Initialize column
        
        :Call:
            >>> db.initcol(col)
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
        # Get type
        clsname = self.get_col_type(col)
        
            
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