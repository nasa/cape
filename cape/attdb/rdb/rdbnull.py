#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.rdb.rdbnull`: Template ATTDB database
=====================================================================

This module provides the class :class:`DBResponseNull` as a subclass of
:class:`dict` that contains methods common to each of the other (mostly)
databases.  The :class:`DBResponseNull` class has most of the database
properties and methods but does not define "response surface"
capabilities that come from other classes that inherit from
:class:`DBResponseNull`.

Finally, having this common template class provides a single point of
entry for testing if an object is based on a product of the
:mod:`cape.attdb.rdb` module.  The following Python sample tests if
any Python object *db* is an instance of any class from this data-file
collection.

    .. code-block:: python

        isinstance(db, cape.attdb.rdb.DBResponseNull)

This class is the basic data container for ATTDB databases and has
interfaces to several different file types.

"""

# Standard library modules
import os
import copy
import warnings

# Third-party modules
import numpy as np

# CAPE modules
import cape.tnakit.typeutils as typeutils

# Data Interfaces
import cape.attdb.ftypes as ftypes



# Declare base class
class DBResponseNull(dict):
    r"""Basic database template without responses
    
    :Call:
        >>> db = DBResponseNull(fname=None, **kw)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            File name; extension is used to guess data format
        *csv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVFile` read
        *textdata*: {``None``} | :class:`str`
            Explicit file name for :class:`TextDataFile`
        *simplecsv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVSimple`
    :Outputs:
        *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
            Generic database
    :Versions:
        * 2019-12-04 ``@ddalle``: First version
    """
  # =============
  # Config
  # =============
  # <
   # --- Primary Methods ---
    # Initialization method
    def __init__(self, fname=None, **kw):
        """Initialization method
        
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Required attributes
        self.cols = []
        self.opts = {}
        self.n = 0

        # Check for null inputs
        if (fname is None) and (not kw):
            return

        # Get file name extension
        if typeutils.isstr(fname):
            # Get extension
            ext = fname.split(".")[-1]
        elif fname is not None:
            # Too confusing
            raise TypeError("Non-keyword input must be ``None`` or a string")
        else:
            # No file extension
            ext = None

        # Initialize file name handles for each type
        fcsv  = None
        fcsvs = None
        ftdat = None
        # Filter *ext*
        if ext == "csv":
            # Guess it's a mid-level CSV file
            fcsv = fname
        elif ext is not None:
            # Unable to guess
            raise ValueError(
                "Unable to guess file type of file name '%s'" % fname)

        # Last-check file names
        fcsv  = kw.pop("csv", fcsv)
        fcsvs = kw.pop("simplecsv", fcsvs)
        ftdat = kw.pop("textdata",  ftdat)

        # Read
        if fcsv is not None:
            # Read CSV file
            self.read_csv(fcsv, **kw)
        elif fcsvs is not None:
            # Read simple CSV file
            self.read_csvsimple(fcsvs, **kw)
        elif ftdat is not None:
            # Read generic textual data file
            self.read_textdata(ftdat, **kw)
        else:
            # If reaching this point, process values
            kw = ftypes.BaseFile.process_kw_values(**kw)
            # Warn extra keywords
            ftypes.BaseFile.warn_kwargs(kw)

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
        # Append count
        if self.__dict__.get("n"):
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
            * 2019-12-04 ``@ddalle``: Only last part of module name
        """
        # Module name
        modname = self.__class__.__module__.split(".")[-1]
        clsname = self.__class__.__name__
        # Start output
        lbl = "<%s.%s(" % (modname, clsname)
        # Append count
        if self.__dict__.get("n"):
            lbl += "n=%i, " % self.n
        # Append columns
        if len(self.cols) <= 5:
            # Show all columns
            lbl += "cols=%s)" % str(self.cols)
        else:
            # Just show number of columns
            lbl += "ncol=%i)" % len(self.cols)
        # Output
        return lbl

   # --- Constructors ---
    # Read data from a CSV instance
    @classmethod
    def from_csv(cls, fname, **kw):
        r"""Read a database from a CSV file

        :Call:
            >>> db = DBResponseNull.from_csv(fname, **kw)
            >>> db = DBResponseNull.from_csv(dbcsv, **kw)
            >>> db = DBResponseNull.from_csv(f, **kw)
        :Inputs:
            *fname*: :class:`str`
                Name of CSV file to read
            *dbcsv*: :class:`cape.attdb.ftypes.csv.CSVFile`
                Existing CSV file
            *f*: :class:`file`
                Open CSV file interface
            *save*, *SaveCSV*: ``True`` | {``False``}
                Option to save the CSV interface to *db._csv*
        :Outputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database
        :See Also:
            * :class:`cape.attdb.ftypes.csv.CSVFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Create a new instance
        self = cls()
        # Call reader method
        self.read_csv(fname, **kw)
        # Output
        return self

    # Read data from a simple CSV instance
    @classmethod
    def from_csvsimple(cls, fname, **kw):
        r"""Read a database from a CSV file

        :Call:
            >>> db = DBResponseNull.from_simplecsv(fname, **kw)
            >>> db = DBResponseNull.from_simplecsv(dbcsv, **kw)
            >>> db = DBResponseNull.from_simplecsv(f, **kw)
        :Inputs:
            *fname*: :class:`str`
                Name of CSV file to read
            *dbcsv*: :class:`cape.attdb.ftypes.csv.CSVSimple`
                Existing CSV file
            *f*: :class:`file`
                Open CSV file interface
            *save*, *SaveCSV*: ``True`` | {``False``}
                Option to save the CSV interface to *db._csv*
        :Outputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database
        :See Also:
            * :class:`cape.attdb.ftypes.csv.CSVFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Create a new instance
        self = cls()
        # Call reader method
        self.read_csvsimple(fname, **kw)
        # Output
        return self

    # Read data from an arbitrary-text data instance
    @classmethod
    def from_textdata(cls, fname, **kw):
        r"""Read a database from a generic text data file

        :Call:
            >>> db = DBResponseNull.from_textdata(fname, **kw)
            >>> db = DBResponseNull.from_textdata(dbf, **kw)
            >>> db = DBResponseNull.from_textdata(f, **kw)
        :Inputs:
            *fname*: :class:`str`
                Name of CSV file to read
            *dbf*: :class:`cape.attdb.ftypes.TextDataFile`
                Existing text data file interface
            *f*: :class:`file`
                Open CSV file interface
            *save*: {``True``} | ``False``
                Option to save the CSV interface to *db._csv*
        :Outputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database
        :See Also:
            * :class:`cape.attdb.ftypes.textdata.TextDataFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # New instance
        self = cls()
        # Call reader method
        self.read_textdata(fname, **kw)
        # Output
        return self

  # ==================
  # Readers
  # ==================
  # <
    # Read CSV file
    def read_csv(self, fname, **kw):
        r"""Read data from a CSV file
        
        :Call:
            >>> db.read_csv(fname, **kw)
            >>> db.read_csv(dbcsv, **kw)
            >>> db.read_csv(f, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database
            *fname*: :class:`str`
                Name of CSV file to read
            *dbcsv*: :class:`cape.attdb.ftypes.csv.CSVFile`
                Existing CSV file
            *f*: :class:`file`
                Open CSV file interface
            *save*, *SaveCSV*: ``True`` | {``False``}
                Option to save the CSV interface to *db._csv*
        :See Also:
            * :class:`cape.attdb.ftypes.csv.CSVFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Get option to save database
        savecsv = kw.pop("save", kw.pop("SaveCSV", False))
        # Check input type
        if isinstance(fname, ftypes.CSVFile):
            # Already a CSV database
            dbf = fname
        else:
            # Create an instance
            dbf = ftypes.CSVFile(fname, **kw)
        # Link the data
        self.link_data(dbf)
        # Copy the options
        self.copy_options(dbf.opts)
        # Save the file interface if needed
        if savecsv:
            self._csv = dbf

    # Read simple CSV file
    def read_csvsimple(self, fname, **kw):
        r"""Read data from a simple CSV file
        
        :Call:
            >>> db.read_csvsimple(fname, **kw)
            >>> db.read_csvsimple(dbcsv, **kw)
            >>> db.read_csvsimple(f, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database
            *fname*: :class:`str`
                Name of CSV file to read
            *dbcsv*: :class:`cape.attdb.ftypes.csv.CSVSimple`
                Existing CSV file
            *f*: :class:`file`
                Open CSV file interface
            *save*, *SaveCSV*: ``True`` | {``False``}
                Option to save the CSV interface to *db._csv*
        :See Also:
            * :class:`cape.attdb.ftypes.csv.CSVFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Get option to save database
        savecsv = kw.pop("save", kw.pop("SaveCSV", False))
        # Check input type
        if isinstance(fname, ftypes.CSVSimple):
            # Already a CSV database
            dbf = fname
        else:
            # Create an instance
            dbf = ftypes.CSVSimple(fname, **kw)
            import pdb
            pdb.set_trace()
        # Link the data
        self.link_data(dbf)
        # Copy the options
        self.copy_options(dbf.opts)
        # Save the file interface if needed
        if savecsv:
            self._csvsimple = dbf

    # Read text data fiel
    def read_textdata(self, fname, **kw):
        r"""Read data from a simple CSV file
        
        :Call:
            >>> db.read_textdata(fname, **kw)
            >>> db.read_textdata(dbcsv, **kw)
            >>> db.read_textdata(f, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database
            *fname*: :class:`str`
                Name of CSV file to read
            *dbcsv*: :class:`cape.attdb.ftypes.csv.CSVSimple`
                Existing CSV file
            *f*: :class:`file`
                Open CSV file interface
            *save*: {``True``} | ``False``
                Option to save the CSV interface to *db._csv*
        :See Also:
            * :class:`cape.attdb.ftypes.csv.CSVFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Get option to save database
        savedat = kw.pop("save", True)
        # Check input type
        if isinstance(fname, ftypes.TextDataFile):
            # Already a file itnerface
            dbf = fname
        else:
            # Create an insteance
            dbf = ftypes.TextDataFile(fname, **kw)
        # Linke the data
        self.link_data(dbf)
        # Copy the otpions
        self.copy_options(dbf.opts)
        # Save the file interface if needed
        if savedat:
            self._textdata = dbf
  # >
        
  # ==================
  # Options
  # ==================
  # <
   # --- Copy/Link ---
    # Link options
    def copy_options(self, opts):
        r"""Copy a database's options

        :Call:
            >>> db.copy_options(dbsrc)
        :Inputs:
            *opts*: :class:`dict`
                Options dictionary
        :Effects:
            *db.opts*: :class:`dict`
                Options merged with or copied from *opts*
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Check input
        if not isinstance(opts, dict):
            raise TypeError("Options input must be dict-type")
        # Get options
        dbopts = self.__dict__.setdefault("opts", {})
        # Merge
        self.opts = dict(dbopts, **opts)
  # >

  # ==================
  # Data
  # ==================
  # <
   # --- Save ---
    # Save a column
    def save_col(self, col, V):
        r"""Save a column to database
        
        :Call:
            >>> db.save_col(col, V)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Data container
            *col*: :class:`str`
                Name of column to save
            *V*: :class:`any`
                Value to save for "column"
        :Effects:
            *db.cols*: 
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Safely get columns list
        cols = self.__dict__.setdefault("cols", [])
        # Check if present
        if col not in cols:
            cols.append(col)
        # Save the data (don't copy it)
        self[col] = V

   # --- Copy/Link ---
    # Link data
    def link_data(self, dbsrc, cols=None):
        # Default columns
        if cols is None:
            cols = dbsrc.cols
        # Check input types
        if not isinstance(cols, list):
            # Column list must be a list
            raise TypeError("Column list must be a list, got '%s'"
                % cols.__class__.__name__)
        elif not isinstance(dbsrc, dict):
            # Source must be a dictionary
            raise TypeError("Source data must be a dict")
        # Loop through columns
        for col in cols:
            # Check type
            if not typeutils.isstr(col):
                raise TypeError("Column names must be strings")
            # Check if data is present
            if col not in dbsrc:
                raise KeyError("No column '%s'" % col)
            # Save the data
            self.save_col(col, dbsrc[col])
  # >

  # =================
  # Writers
  # =================
  # <
   # --- CSV ---
    # Create a CSV file
    def get_CSVFile(self, cols=None):
        r"""Turn data container into CSV file interface

        :Call:
            >>> dbcsv = db.get_CSVFile(cols=None)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Data container
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                List of columns to include (default *db.cols*)
        :Outputs:
            *dbcsv*: :class:`cape.attdb.ftypes.CSVFile`
                CSV file interface
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Default column list
        if cols is None:
            cols = self.cols
        # Get options interface
        opts = self.__dict__.get("opts", {})
        # Get relevant options
        kw = {}
        for (k, vdef) in CSVFile._DefaultOpts.items():
            # Set property
            kw[k] = opts.get(k, vdef)
        # Set values
        kw["Values"] = {col: self[col] for col in cols}
        # Turn off expansion
        kw["ExpandScalars"] = False
        # Explicit column list
        kw["cols"] = cols
        # Create instance
        return CSVFile(**kw)
        

    # Write dense CSV file
    def write_csv_dense(self, fname, cols=None):
        r""""Write dense CSV file

        If *db._csv* exists, the database will be written from that
        interface.  Otherwise, :func:`get_CSVFile` will be called.

        :Call:
            >>> db.write_csv_dense(fname, cols=None)
            >>> db.write_csv_dense(f, cols=None)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Data container
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
                CSV file interface
            *fname*: {*db.fname*} | :class:`str`
                Name of file to write
            *f*: :class:`file`
                File open for writing
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of columns to write
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Check for CSV handle
        if "_csv" in self.__dict__:
            # Already ready
            dbcsv = self._csv
        else:
            # Get a CSV file interface
            dbcsv = self.get_CSVFile(cols=cols)
        # Write it
        dbcsv.write_csv_dense(fname, cols=cols)
    
  # >

  # ============
  # Config
  # ============
  # <
   # --- Copy ---
    # Copy
    def copy(self):
        r"""Make a copy of a database class
        
        Each database class may need its own version of this class
        
        :Call:
            >>> dbcopy = db.copy()
        :Inputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database
        :Outputs:
            *dbcopy*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Copy of generic database
        :Versions:
            * 2019-12-04 ``@ddalle``: First version
        """
        # Form a new database
        dbcopy = self.__class__()
        # Copy relevant parts
        self.copy_DBResponseNull(dbcopy)
        # Output
        return dbcopy

    # Copy attributes and data known to DBResponseNull class
    def copy_DBResponseNull(self, dbcopy):
        r"""Copy attributes and data relevant to null-response DB
        
        :Call:
            >>> db.copy_DBResponseNull(dbcopy)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database
            *dbcopy*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Copy of generic database
        :Versions:
            * 2019-12-04 ``@ddalle``: First version
        """
        # Loop through columns
        for col in self.cols:
            dbcopy[col] = copy.copy(self[col])
        # Copy all attributes
        self.copy__dict__(dbcopy, skip=[])
            
            
            
    # Copy any remaining items
    def copy__dict__(self, dbtarg, skip=[]):
        r"""Copy all attributes except for specified list
        
        :Call:
            >>> db.copy__dict__(dbtarg, skip=[])
        :Inputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database
            *dbtarg*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database; target copy
            *skip*: :class:`list`\ [:class:`str`]
                List of attributes not to copy
        :Effects:
            ``getattr(dbtarg, k)``: ``getattr(db, k, vdef)``
                Shallow copy of attribute from *DBc* or *vdef* if necessary
        :Versions:
            * 2019-12-04 ``@ddalle``: First version
        """
        
        # Check *skip*
        if not isinstance(skip, (list, tuple)):
            raise TypeError("Attributes to skip during copy must be list")
        # Loop through dict
        for k in self.__dict__:
            # Check if it should be skipped
            if k in skip:
                continue
            # Get copy, if possible
            try:
                # Create the copy
                vcopy = self.copyitem(self.__dict__[k])
                # Set it
                dbtarg.__dict__[k] = vcopy
            except Exception:
                # No copy
                continue

    # Copy an attribute if present
    def copyattr(self, dbtarg, k, vdef={}):
        r"""Make an appropriate copy of an attribute if present

        :Call:
            >>> db.copyattr(dbtarg, k, vdef={})
        :Inputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database
            *dbtarg*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database; target copy
            *k*: :class:`str`
                Name of attribute to copy
            *vdef*: {``{}``} | :class:`any`
                Default value for output attribute if ``getattr(db,k)``
                does not exist
        :Effects:
            ``getattr(dbtarg, k)``: ``getattr(db, k, vdef)``
                Shallow copy of attribute from *DBc* or *vdef* if necessary
        :Versions:
            * 2018-06-08 ``@ddalle``: First version
            * 2019-12-04 ``@ddalle``: Copied from :class:`DBCoeff`
        """
        # Check for attribute
        if hasattr(self, k):
            # Get attribute
            v = getattr(self, k)
        else:
            # Use default
            v = vdef
        # Copy item
        vcopy = self.copyitem(v)
        # Save it
        setattr(dbtarg, k, vcopy)

    # Copy an item according to local rules
    def copyitem(self, v):
        r"""Return a copy of appropriate depth following class rules
        
        :Call:
            >>> vcopy = db.copyitem(v)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
                Generic database
            *v*: :class:`any`
                Variable to be copied
        :Outputs:
            *vcopy*: *v.__class__*
                Copy of *v* (shallow or deep)
        :Versions:
            * 2019-12-04 ``@ddalle``: First version
        """
        # Type
        t = v.__class__
        # Check the type in order to make a copy
        if v is None:
            # Not necessary to copy
            return
        elif t == dict:
            # Deep copy of dictionary
            try:
                return copy.deepcopy(v)
            except Exception:
                return dict(v)
        elif hasattr(v, "copy"):
            # Use class's already-build copy() method
            return v.copy()
        elif t == list:
            # Copy list
            return list(v)
        else:
            # Shallow copy
            return copy.copy(v)
  # >

