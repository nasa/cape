#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.rdbnull`: Template ATTDB database
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

# Third-party modules
import numpy as np

# CAPE modules
import cape.tnakit.typeutils as typeutils
import cape.tnakit.kwutils as kwutils

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
        *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
            Generic database
    :Versions:
        * 2019-12-04 ``@ddalle``: First version
    """
  # =====================
  # Class Attributes
  # =====================
  # <
    # Data types
    _DTypeMap = {}
    # Default definition
    _DefaultDefn = {
        "Type": "float64",
        "Label": True,
        "LabelFormat": "%s",
        "WriteFormat": "%s",
        "OutputDim": 0,
    }
    # Default evaluation parameters
    _DefaultRole = {
        "ResponseRole": "xvar",
        "EvalMethod": "nearest",
        "XVars": [],
        "kwargs": {},
        "xvar_aliases": {},
        "OutputDim": 0,
        "OutputXVars": [],
    }
    # Definitions based on names
    _DefaultDefnMap = {}
  # >

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
        self.bkpts = {}

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
        fxls  = None
        # Filter *ext*
        if ext == "csv":
            # Guess it's a mid-level CSV file
            fcsv = fname
        elif ext == "xls":
            # Guess it's a spreadsheet
            fxls = fname
        elif ext == "xlsx":
            # Guess it's a spreadsheet
            fxls = fname
        elif ext is not None:
            # Unable to guess
            raise ValueError(
                "Unable to guess file type of file name '%s'" % fname)

        # Last-check file names
        fcsv  = kw.pop("csv", fcsv)
        fxls  = kw.pop("xls", fxls)
        fcsvs = kw.pop("simplecsv", fcsvs)
        ftdat = kw.pop("textdata",  ftdat)

        # Read
        if fcsv is not None:
            # Read CSV file
            self.read_csv(fcsv, **kw)
        elif fxls is not None:
            # Read XLS file
            self.read_xls(fxls, **kw)
        elif fcsvs is not None:
            # Read simple CSV file
            self.read_csvsimple(fcsvs, **kw)
        elif ftdat is not None:
            # Read generic textual data file
            self.read_textdata(ftdat, **kw)
        else:
            # If reaching this point, process values
            kw = ftypes.BaseFile.process_kw_values(**kw)

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

   # --- Class Constructors ---
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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
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

    # Read data from an Excel file
    @classmethod
    def from_xls(cls, fname, **kw):
        r"""Read a database from a spreadsheet

        :Call:
            >>> db = DBResponseNull.from_xls(fname, **kw)
            >>> db = DBResponseNull.from_xls(dbf, **kw)
            >>> db = DBResponseNull.from_xls(wb, **kw)
            >>> db = DBResponseNull.from_xls(ws, **kw)
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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
                Generic database
        :See Also:
            * :class:`cape.attdb.ftypes.textdata.TextDataFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # New instance
        self = cls()
        # Call reader method
        self.read_xls(fname, **kw)
        # Output
        return self

   # --- Copy ---
    # Copy
    def copy(self):
        r"""Make a copy of a database class
        
        Each database class may need its own version of this class
        
        :Call:
            >>> dbcopy = db.copy()
        :Inputs:
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
                Generic database
        :Outputs:
            *dbcopy*: :class:`cape.attdb.rdbnull.DBResponseNull`
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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
                Generic database
            *dbcopy*: :class:`cape.attdb.rdbnull.DBResponseNull`
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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
                Generic database
            *dbtarg*: :class:`cape.attdb.rdbnull.DBResponseNull`
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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
                Generic database
            *dbtarg*: :class:`cape.attdb.rdbnull.DBResponseNull`
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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
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
            *db*:: :class:`
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
        dbopts = self.__dict__.get("opts", {})
        # Merge
        self.opts = dict(dbopts, **opts)

   # --- Column Properties ---
    # Get generic property from column
    def get_col_prop(self, col, prop, vdef=None):
        """Get property for specific column
        
        :Call:
            >>> v = db.get_col_prop(col, prop, vdef=None)
        :Inputs:
            *db*: :class:`cape.attdb.rdb.DBResponseNull`
                Data container
            *col*: :class:`str`
                Name of column
            *prop*: :class:`str`
                Name of property
            *vdef*: {``None``} | :class:`any`
                Default value if not specified in *db.opts*
        :Outputs:
            *v*: :class:`any`
                Value of ``db.opts["Definitions"][col][prop]`` if
                possible; defaulting to
                ``db.opts["Definitions"]["_"][prop]`` or *vdef*
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
            return defn.get(prop, vdef)

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
        return self.get_col_prop(col, "Type", vdef="float64")

    # Get data type
    def get_col_dtype(self, col):
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
        # Get type
        coltype = self.get_col_type(col)
        # Apply mapping if needed
        return self.__class__._DTypeMap.get(coltype, coltype)
  # >

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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
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

    # Read XLS file
    def read_xls(self, fname, **kw):
        r"""Read data from an ``.xls`` or ``.xlsx`` file
        
        :Call:
            >>> db.read_xls(fname, **kw)
            >>> db.read_xls(dbcsv, **kw)
            >>> db.read_xls(wb, **kw)
            >>> db.read_xls(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
                Generic database
            *fname*: :class:`str`
                Name of ``.xls`` or ``.xlsx`` file to read
            *sheet*: {``0``} | :class:`int` | :class:`str`
                Worksheet name or number
            *wb*: :class:`xlrd.book.Book`
                Open workbook (spreadsheet file)
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *skiprows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *subrows*: {``0``} | :class:`int` > 0
                Number of rows below header row to skip
            *skipcols*: {``None``} | :class:`int` >= 0
                Number of columns to skip before first data column
            *maxrows*: {``None``} | :class:`int` > *skiprows*
                Maximum row number of data
            *maxcols*: {``None``} | :class:`int` > *skipcols*
                Maximum column number of data
            *save*, *SaveXLS*: ``True`` | {``False``}
                Option to save the XLS interface to *db._xls*
        :See Also:
            * :class:`cape.attdb.ftypes.csv.CSVFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Get option to save database
        save = kw.pop("save", kw.pop("SaveXLS", False))
        # Check input type
        if isinstance(fname, ftypes.XLSFile):
            # Already a CSV database
            dbf = fname
        else:
            # Create an instance
            dbf = ftypes.XLSFile(fname, **kw)
        # Link the data
        self.link_data(dbf)
        # Copy the options
        self.copy_options(dbf.opts)
        # Save the file interface if needed
        if save:
            self._xls = dbf
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
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
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
        for (k, vdef) in ftypes.CSVFile._DefaultOpts.items():
            # Set property
            kw[k] = opts.get(k, vdef)
        # Set values
        kw["Values"] = {col: self[col] for col in cols}
        # Turn off expansion
        kw["ExpandScalars"] = False
        # Explicit column list
        kw["cols"] = cols
        # Create instance
        return ftypes.CSVFile(**kw)

    # Write dense CSV file
    def write_csv_dense(self, fname, cols=None):
        r""""Write dense CSV file

        If *db._csv* exists, the database will be written from that
        interface.  Otherwise, :func:`get_CSVFile` will be called.

        :Call:
            >>> db.write_csv_dense(fname, cols=None)
            >>> db.write_csv_dense(f, cols=None)
        :Inputs:
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
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

  # ==================
  # Data
  # ==================
  # <
   # --- Save/Add ---
    # Save a column
    def save_col(self, col, V):
        r"""Save a column to database
        
        :Call:
            >>> db.save_col(col, V)
        :Inputs:
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
                Data container
            *col*: :class:`str`
                Name of column to save
            *V*: :class:`any`
                Value to save for "column"
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                Appends *col* if not present already
            *db[col]*: *V*
                Value saved
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
            raise TypeError(
                "Column list must be a list, got '%s'"
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

  # ===================
  # Break Points
  # ===================
  # <
   # --- Creation ---
    # Get automatic break points
    def GetBreakPoints(self, cols, nmin=5, tol=1e-12):
        r"""Create automatic list of break points for interpolation

        :Call:
            >>> db.GetBreakPoints(col, nmin=5, tol=1e-12)
            >>> db.GetBreakPoints(cols, nmin=5, tol=1e-12)
        :Inputs:
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
                Data container
            *col*: :class:`str`
                Individual lookup variable
            *cols*: :class:`list`\ [:class:`str`]
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
            * 2019-12-16 ``@ddalle``: Updated for :mod:`rdbnull`
        """
        # Check for single key list
        if not isinstance(cols, (list, tuple)):
            # Make list
            cols = [cols]
        # Initialize break points
        bkpts = self.__dict__.setdefault("bkpts", {})
        # Loop through keys
        for col in cols:
            # Check type
            if not isinstance(col, typeutils.strlike):
                raise TypeError("Column name is not a string")
            # Check if present
            if col not in self.cols:
                raise KeyError("Lookup column '%s' is not present" % col)
            # Get all values
            V = self[col]
            # Get data type
            dtype = self.get_col_dtype(col)
            # Check dtype
            if dtype == "str":
                # Get unique values without converting to array
                B = list(set(V))
            elif dtype.startswith("int"):
                # No need to apply tolerance
                B = np.unique(V)
            else:
                # Get unique values of array
                U = np.unique(V)
                # Initialize filtered value
                B = np.zeros_like(U)
                n = 0
                # Loop through entries
                for v in U:
                    # Check if too close to a previous entry
                    if (n > 0) and np.min(np.abs(v - B[:n])) <= tol:
                        # Close to previous "unique" value
                        continue
                    # Count entries
                    if np.count_nonzero(np.abs(V - v) <= tol) >= nmin:
                        # Save the value
                        B[n] = v
                        # Increase count
                        n += 1
                # Trim
                B = B[:n]
            # Save these break points
            bkpts[col] = B

    # Map break points from other key
    def MapBreakPoints(self, cols, scol, tol=1e-12):
        r"""Map break points of one column to one or more others

        The most common purpose to use this method is to create
        non-ascending break points.  One common example is to keep track
        of the dynamic pressure values at each Mach number.  These
        dynamic pressures may be unique, but sorting them by dynamic
        pressure is different from the order in which they occur in
        flight.

        :Call:
            >>> db.MapBreakPoints(cols, scol, tol=1e-12)
        :Inputs:
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
                Data container
            *cols*: :class:`list`\ [:class:`str`]
                Individual lookup variable
            *scol*: :class:`str`
                Name of key to drive map/schedule
            *tol*: {``1e-12``} | :class:`float` >= 0
                Tolerance cutoff (used for *scol*)
        :Outputs:
            *DBc.bkpts*: :class:`dict`
                Dictionary of 1D unique lookup values
            *DBc.bkpts[key]*: :class:`np.ndarray` (:class:`float`)
                Unique values of *DBc[key]* with at least *nmin* entries
        :Versions:
            * 2018-06-29 ``@ddalle``: First version
            * 2019-12-16 ``@ddalle``: Ported to :mod:`rdbnull`
        """
        # Check inputs
        if not isinstance(cols, list):
            raise TypeError("Columns input must be a list")
        elif not isinstance(scol, typeutils.strlike):
            raise TypeError("Schedule key must be a string")
        # Get break points
        bkpts = self.__dict__.get("bkpts")
        # Check break points for *scol*
        if bkpts is None:
            raise AttributeError("No 'bkpts' attribute; call GetBreakPoints()")
        elif scol not in bkpts:
            raise AttributeError("No bkpts for col '%s'" % col)
        # Get data type of *scol*
        dtype = self.get_col_dtype(scol)
        # Check data type
        if not (dtype.startswith("float") or dtype.startswith("int")):
            raise TypeError(
                ("Schedule col '%s' must have either " % scol) +
                ("float or int type (got %s)" % dtype))
        # Get schedule break points and nominal values
        V0 = self[scol]
        U0 = self.bkpts[scol]
        # Loop through keys
        for col in cols:
            # Check type
            if not isinstance(col, typeutils.strlike):
                raise TypeError("Column name is not a string")
            # Check if present
            if col not in self.cols:
                raise KeyError("Lookup column '%s' is not present" % col)
            # Get data type
            dtype = self.get_col_dtype(col)
            # Check it
            if not (dtype.startswith("float") or dtype.startswith("int")):
                raise TypeError(
                    ("Break point col '%s' must have either " % col) +
                    ("float or int type (got %s)" % dtype))
            # Values for *col*
            V = self[col]
            # Initialize break point array for *col*
            U = np.zeros(U0.size, dtype=dtype)
            # Check shape
            if V.size != V0.size:
                raise ValueError(
                    ("Col '%s' (%i) has different size " % (col, V.size)) +
                    ("from schedule col '%s' (%i)" % (scol, V0.size)))
            # Loop through slice values
            for (j, v0) in enumerate(U0):
                # Find value of slice key matching that parameter
                i = np.where(np.abs(V0 - v0) <= tol)[0][0]
                # Save value of that index from col
                V[j] = V[i]
            # Save break points
            bkpts[col] = V

    # Schedule break points at slices at other key
    def ScheduleBreakPoints(self, cols, scol, nmin=5, tol=1e-12):
        r"""Create lists of unique values at each unique value of *scol*

        This function creates a break point list of the unique values of
        each *col* in *cols* at each unique value of a "scheduling"
        column *scol*.  For example, if a different run matrix of
        *alpha* and *beta* is used at each *mach* number, this function
        creates a list of the unique *alpha* and *beta* values for each
        Mach number in *db.bkpts["mach"]*.

        :Call:
            >>> db.ScheduleBreakPoints(cols, scol)
        :Inputs:
            *db*: :class:`cape.attdb.rdbnull.DBResponseNull`
                Data container
            *cols*: :class:`list`\ [:class:`str`]
                Individual lookup variable
            *scol*: :class:`str`
                Name of key to drive map/schedule
            *nmin*: {``5``} | :class:`int` > 0
                Minimum number of data points at one value of a key
            *tol*: {``1e-12``} | :class:`float` >= 0
                Tolerance cutoff
        :Outputs:
            *db.bkpts*: :class:`dict`
                Dictionary of unique lookup values
            *db.bkpts[col]*: :class:`list`\ [:class:`np.ndarray`]
                Unique values of *db[col]* at each value of *scol*
        :Versions:
            * 2018-06-29 ``@ddalle``: First version
            * 2019-12-16 ``@ddalle``: Ported to :mod:`rdbnull`
        """
        # Check inputs
        if not isinstance(cols, list):
            raise TypeError("Columns input must be a list")
        elif not isinstance(scol, typeutils.strlike):
            raise TypeError("Schedule key must be a string")
        # Get break points
        bkpts = self.__dict__.get("bkpts")
        # Check break points for *scol*
        if bkpts is None:
            raise AttributeError("No 'bkpts' attribute; call GetBreakPoints()")
        elif scol not in bkpts:
            raise AttributeError("No bkpts for col '%s'" % col)
        # Get data type of *scol*
        dtype = self.get_col_dtype(scol)
        # Check data type
        if not (dtype.startswith("float") or dtype.startswith("int")):
            raise TypeError(
                ("Schedule col '%s' must have either " % scol) +
                ("float or int type (got %s)" % dtype))
        # Get schedule break points and nominal values
        V0 = self[scol]
        U0 = self.bkpts[scol]
        # Loop through keys
        for col in cols:
            # Check type
            if not isinstance(col, typeutils.strlike):
                raise TypeError("Column name is not a string")
            # Check if present
            if col not in self.cols:
                raise KeyError("Lookup column '%s' is not present" % col)
            # Get data type
            dtype = self.get_col_dtype(col)
            # Check it
            if not (dtype.startswith("float") or dtype.startswith("int")):
                raise TypeError(
                    ("Break point col '%s' must have either " % col) +
                    ("float or int type (got %s)" % dtype))
            # Values for *col*
            V = self[col]
            # Initialize break point array for *col*
            U = np.zeros(U0.size, dtype=dtype)
            # Check shape
            if V.size != V0.size:
                raise ValueError(
                    ("Col '%s' (%i) has different size " % (col, V.size)) +
                    ("from schedule col '%s' (%i)" % (scol, V0.size)))
            # Initialize scheduled break points
            X = []
            # Get all values for this key
            V = self[col]
            # Loop through slice values
            for (j, v0) in enumerate(U0):
                # Indices of points in this slice
                I = np.where(np.abs(V0 - v0) <= tol)[0]
                # Check for broken break point
                if I.size == 0:
                    # This shouldn't happen
                    raise ValueError("No points matching slice at " +
                        ("%s = %.2e" % (scol, v0)))
                elif I.size < nmin:
                    # No hope of break points at this *scol* value
                    X.append(np.zeros(0, dtype=V.dtype))
                    continue
                # Get unique values on the slice
                U = np.unique(V[I])
                # Initialize filtered value
                B = np.zeros_like(U)
                n = 0
                # Loop through entries
                for v in U:
                    # Check if too close to a previous entry
                    if (n > 0) and np.min(np.abs(v - B[:n])) <= tol:
                        continue
                    # Count entries
                    if np.count_nonzero(np.abs(V[I] - v) <= tol) >= nmin:
                        # Save the value
                        B[n] = v
                        # Increment count
                        n += 1
                # Save break point
                X.append(B[:n])
            # Save break points
            bkpts[col] = X
  # >
