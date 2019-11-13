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
            *DefaultClass*: {``"float"``} | :class:`str`
                Name of default class
            *DefaultFormat*: {``None``} | :class:`str`
                Optional default format string
            *DefaultType*: :class:`dict`
                :class:`dict` of default *Class*, *Format*
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
        odefn = kw.pop("DefaultType", {})
        # Process various defaults
        odefcls = kw.pop("DefaultClass", "float")
        odeffmt = kw.pop("DefaultFormat", None)
        # Set defaults
        odefn.setdefault("Class",  odefcls)
        odefn.setdefault("Format", odeffmt)
        # Ensure definitions exist
        opts = self.opts.setdefault("Types", {})
        # Save defaults
        self.opts["Types"]["_"] = odefn
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

   # --- Values ---
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