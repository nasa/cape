#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.ftypes.xls`: Excel spreadsheet data interface
===============================================================

This module provides a class :class:`XLSFile` for reading and writing
tabular data from Excel-like spreadsheets with the file extension
``.xls`` or ``.xlsx``.  It relies on two third-party libraries readily
available from the Python Package Index (PyPI):

    * :mod:`xlrd` for reading spreadsheets
    * :mod:`xlsxwriter` for writing them

These can be readily installed on any machine with both Python and
access to the internet (even without elevated privileges) using the
commands below:

    .. code-block:: console

        $ pip install --user xlrd
        $ pip install --user xlsxwriter

Because CAPE may also be used on machines without regular access to the
internet, this module does not raise an ``ImportError`` in the case that
these third-party modules are not available.  However, the module will
provide no functionality if these modules are not available.

"""

# Third-party modules
import numpy as np

# Quasi-optional third-party modules
try:
    import xlrd
except ImportError:
    xlrd = None
try:
    import xlsxwriter
except ImportError:
    xlsxwriter = None

# CAPE modules
import cape.tnakit.typeutils as typeutils

# Local modules
from .basefile import BaseFile


# Class for handling data from XLS files
class XLSFile(BaseFile):
    r"""Class for reading ``.xls`` and ``.xlsx`` files

    :Call:
        >>> db = XLSFile(fname, sheet=0, **kw)
        >>> db = XLSFile(wb, sheet=0, **kw)
        >>> db = XLSFile(ws, **kw)
    :Inputs:
        *fname*: :class:`str`
            Name of ``.xls`` or ``.xlsx`` file to read
        *sheet*: {``0``} | :class:`int` | :class:`str`
            Worksheet name or number
        *wb*: :class:`xlrd.book.Book`
            Open workbook (spreadsheet file)
        *ws*: :class:`xlrd.sheet.Sheet`
            Direct access to a worksheet
    :Outputs:
        *db*: :class:`cape.attdb.ftypes.xls.XLSFile`
            XLS file interface
        *db.cols*: :class:`list`\ [:class:`str`]
            List of columns read
        *db.opts*: :class:`dict`
            Options for this interface
        *db[col]*: :class:`np.ndarray` | :class:`list`
            Numeric array or list of strings for each column
    :Versions:
        * 2019-12-12 ``@ddalle``: First version
    """
  # =============
  # Config
  # =============
  # <
    # Initialization method
    def __init__(self, fname, sheet=0, **kw):
        """Initialization method

        :Versions:
            * 2019-12-12 ``@ddalle``: First version
        """
        # Initialize options
        self.opts = {}
        self.cols = []
        self.n = 0
        self.fname = None

        # Process options
        kw = self.process_opts_generic(**kw)

        # Read file if appropriate
        if fname:
            # Read file
            kw = self.read_xls(fname, sheet=sheet, **kw)
        else:
            # Process input column defs
            kw = self.process_col_defns(**kw)

        # Check for overrides of values
        kw = self.process_kw_values(**kw)
        # Warn about any unused inputs
        self.warn_kwargs(kw)
  # >

  # ================
  # Read
  # ================
  # <
   # --- Control ---
    # Reader
    def read_xls(self, fname, sheet=0, **kw):
        r"""Read an ``.xls`` or ``.xlsx`` file

        :Call:
            >>> db.read_xls(fname, sheet=0, **kw)
            >>> db.read_xls(wb, sheet=0, **kw)
            >>> db.read_xls(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xls.XLSFile`
                XLS file interface
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
        :Versions:
            * 2019-12-12 ``@ddalle``: First version
        """
        # Check module
        if xlrd is None:
            raise ImportError("No module 'xlrd'")
            
        # Check type
        if isinstance(fname, xlrd.sheet.Sheet):
            # Already a worksheet
            ws = fname
        else:
            # Get sheet based on workbook or file name
            if isinstance(fname, xlrd.Book):
                # Workbook
                wb = fname
            elif typeutils.isstr(fname):
                # Open workbook
                wb = xlrd.open_workbook(fname)
                # Save file name
                self.fname = fname
            # Get sheet
            if isinstance(sheet, int):
                # Get sheet number
                ws = wb.sheet_by_index(sheet)
            elif typeutils.isstr(sheet):
                # Get sheet by its name
                ws = wb.sheet_by_name(sheet)
            else:
                # Unknown type
                raise TypeError(
                    "Sheet (worksheet) must be index (int) or name (str)")

        # Read worksheet
        kw = self.read_xls_worksheet(ws, **kw)

        # Output remaining options
        return kw

    # Read a worksheet
    def read_xls_worksheet(self, ws, **kw):
        r"""Read one worksheet of an ``.xls`` or ``.xlsx`` file

        :Call:
            >>> db.read_xls_worksheet(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xls.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *skiprows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *subrows*: {``None``} | :class:`int` >= 0
                Number of rows below header row to skip
            *skipcols*: {``None``} | :class:`int` >= 0
                Number of columns to skip before first data column
            *maxrows*: {``None``} | :class:`int` > *skiprows*
                Maximum row number of data
            *maxcols*: {``None``} | :class:`int` > *skipcols*
                Maximum column number of data
        :Versions:
            * 2019-12-12 ``@ddalle``: First version
        """
        # Process options
        kwread = self.process_kw_xlsread(kw)
        # Read header
        cols = self.read_xls_header(ws, **kwread)
        # Process definitions
        kw = self.process_col_defns(**kw)
        # Read data
        self.read_xls_data(ws, cols)
        # Output remaining options
        return kw

   # --- Header ---
    # Read worksheet header
    def read_xls_header(self, ws, **kw):
        r"""Read header row from a worksheet

        :Call:
            >>> cols = db.read_xls_header(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xls.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *skiprows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *subrows*: {``None``} | :class:`int` >= 0
                Number of rows below header row to skip
            *skipcols*: {``None``} | :class:`int` >= 0
                Number of columns to skip before first data column
            *maxcols*: {``None``} | :class:`int` > *skipcols*
                Maximum column number of data
        :Outputs:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names if read
        :Versions:
            * 2019-12-12 ``@ddalle``: First version
        """
        # Get skip options
        skiprows = kw.get("skiprows")
        skipcols = kw.get("skipcols")
        # Get maximum option
        maxcols = kw.get("maxcols")
        # Get subrow types
        subrows = kw.get("subrows")
        # Check types
        if not ((maxcols is None) or isinstance(maxcols, int)):
            raise TypeError("'maxcols' arg must be None or int")
        # Find header row if needed
        if skiprows is None:
            # Loop until we have an empty row
            for skiprows in range(ws.nrows):
                # Read the row
                header = ws.row_values(skiprows, end_colx=maxcols)
                # Check if there's anything in it
                if any(header):
                    break
        elif isinstance(skiprows, int):
            # Read the specified header row
            header = ws.row_values(skiprows, end_colx=maxcols)
        else:
            raise TypeError("'skiprows' arg must be None or int")
        # Save *skiprows* options
        self.opts["SkipRows"] = skiprows
        # Find header column if needed
        if skipcols is None:
            # Find first nonempty entry
            for skipcols in range(len(header)):
                # Check for entry
                if header[skipcols]:
                    # Found something other than ""
                    break
        elif not isinstance(skipcols, int):
            raise TypeError("'skipcols' arg must be None or int")
        # Save *skipcols* option
        self.opts["SkipCols"] = skipcols
        # Check header and guess types
        for (j, col) in enumerate(header):
            # Check type
            if col == "":
                # Rename column "col1", "col2", etc.
                header[j] = "col%i" % (j + 1)
            elif not typeutils.isstr(col):
                # Column name invalid
                raise TypeError(
                    "Column %i header is not a string" % (j + 1 + skipcols))
        # Translate column names
        cols = self.translate_colnames(header)
        # Process types from first row of data
        self.read_xls_firstrowtypes(ws, cols)
        # Save column names
        for col in cols:
            # Check if present
            if col not in self.cols:
                # Append if not
                self.cols.append(col)
        # Output
        return cols

    # Guess data types from first row of data
    def read_xls_firstrowtypes(self, ws, cols, **kw):
        r"""Guess data types from first row of data

        :Call:
            >>> db.read_xls_firstrowtypes(ws, cols, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xls.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *cols*: :class:`list`\ [:class:`str`]
                List of column names to process
            *skiprows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *subrows*: {``None``} | :class:`int` >= 0
                Number of rows below header row to skip
            *skipcols*: {``None``} | :class:`int` >= 0
                Number of columns to skip before first data column
            *maxcols*: {``None``} | :class:`int` > *skipcols*
                Maximum column number of data
        :Effects:
            *db.opts[col]*: :class:`dict`
                *Type* is set for each *col* in *cols*
        :Versions:
            * 2019-12-12 ``@ddalle``: First version
        """
        # Get skip options
        skiprows = kw.get("skiprows", self.opts.get("SkipRows", 0))
        skipcols = kw.get("skipcols", self.opts.get("SkipCols", 0))
        # Maximum option
        maxcols = kw.get("maxcols", self.opts.get("MaxCols"))
        maxrows = kw.get("maxrows", self.opts.get("MaxRows"))
        # Sub-header row count
        subrows = kw.get("subrows", self.opts.get("SubRows"))
        # Check types
        if not ((maxcols is None) or isinstance(maxcols, int)):
            raise TypeError("'maxcols' arg must be None or int")
        if not ((maxrows is None) or isinstance(maxrows, int)):
            raise TypeError("'maxrows' arg must be None or int")
        if not ((subrows is None) or isinstance(subrows, int)):
            raise TypeError("'subrows' arg must be None or int")
        # Initialize types
        defns = self.opts.setdefault("Definitions", {})
        # Find first row if needed
        if subrows is None:
            # Maximum row count of worksheet
            nrow = ws.nrows - skiprows - 1
            # Check for lower row limit
            if maxrows is not None:
                # Use lower limit
                nrow = min(nrow, maxrows - skiprows - 1)
            # Loop until we have an entry with the right size
            for subrows in range(nrow):
                # Row count
                irow = skiprows + subrows + 1
                # Get the row
                row1 = ws.row_values(irow, skipcols, end_rowx=maxrows)
                # Check count
                if any(row1):
                    break
        else:
            # Overall row number
            irow = skiprows + subrows + 1
            # Read specified row
            row1 = ws.row_values(irow, skipcols, end_colx=maxcols)
        # Save sub-row count
        self.opts["SubRows"] = subrows
        # Check consistency
        if len(row1) != len(cols):
            raise ValueError(
                ("First data row and list of columns have different lengths") +
                ("(%i and %i)" % (len(row1), len(cols))))
        # Check types
        for (j, col) in enumerate(cols):
            # Create definitions if necessary
            defn = defns.setdefault(col, {})
            # Get value for this column
            v = row1[j]
            # Filter its type
            if isinstance(v, float):
                # Convert float type
                defn.setdefault("Type", "float64")
            else:
                # Only float or string
                defn.setdefault("Type", "str")

   # --- Data ---
    # Read data
    def read_xls_data(self, ws, cols, **kw):
        r"""Read column data from one ``.xls`` or ``.xlsx`` worksheet

        :Call:
            >>> db.read_xls_data(ws, cols, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xls.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *cols*: :class:`list`\ [:class:`str`]
                List of column names to process
            *skiprows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *subrows*: {``0``} | :class:`int` > 0
                Number of rows below header row to skip
            *skipcols*: {``None``} | :class:`int` >= 0
                Number of columns to skip before first data column
            *maxcols*: {``None``} | :class:`int` > *skipcols*
                Maximum column number of data
        :Effects:
            *db[col]*: :class:`list` | :class:`np.ndarray`
                Column of data read from each column
            *db._n[col]*: :class:`int`
                Length of each column
        :Versions:
            * 2019-12-12 ``@ddalle``: First version
        """
        # Get skip options
        skiprows = kw.get("skiprows", self.opts.get("SkipRows", 0))
        skipcols = kw.get("skipcols", self.opts.get("SkipCols", 0))
        # Maximum option
        maxcols = kw.get("maxcols", self.opts.get("MaxCols"))
        maxrows = kw.get("maxrows", self.opts.get("MaxRows"))
        # Sub-header row count
        subrows = kw.get("subrows", self.opts.get("SubRows", 0))
        # Check types
        if not ((maxcols is None) or isinstance(maxcols, int)):
            raise TypeError("'maxcols' arg must be None or int")
        if not ((maxrows is None) or isinstance(maxrows, int)):
            raise TypeError("'maxrows' arg must be None or int")
        if not ((subrows is None) or isinstance(subrows, int)):
            raise TypeError("'subrows' arg must be None or int")
        # Get counts
        _n = self.__dict__.setdefault("_n", {})
        # Initialize types
        defns = self.opts.get("Definitions", {})
        # First data row number
        irow = skiprows + subrows + 1
        # First data col number
        icol = skipcols
        # Loop through columns
        for (j, col) in enumerate(cols):
            # Get type
            defn = defns.get(col, {})
            # Get data type
            clsj = defn.get("Type", "float64")
            # Translate if necessary
            dtj = self._DTypeMap.get(clsj, clsj)
            # Read the whole column
            V = ws.col_values(icol + j, irow, end_rowx=maxrows)
            # Read data based on type
            if dtj == "str":
                # Read the whole column and allow empty strings
                pass
            elif dtj.startswith("float") or dtj.startswith("int"):
                # Check for empty strings
                if "" in V:
                    # Find index of first such one
                    iend = V.index("")
                    # Check for empty column
                    if iend == 0:
                        raise ValueError(
                            "Found no valid floats in col %i" % (icol + j))
                    # Strip trailing entries
                    V = V[:iend]
                # Convert to array
                V = np.array(V, dtype=dtj)
            # Save
            self.save_col(col, V)

   # --- Options ---
    # Process all read-related options
    def process_kw_xlsread(self, kw):
        r"""Process all options related to reading XLS worksheet

        The mentioned options will be removed from *kw*.

        :Call:
            >>> kwread = db.process_kw_xlsread(kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xls.XLSFile`
                XLS file interface
            *kw*: :class:`dict`
                Keyword arguments from parent function
        :Keys:
            *SkipRows*, *skiprows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *SubRows*, *subrows*: {``0``} | :class:`int` > 0
                Number of rows below header row to skip
            *SkipCols*, *skipcols*: {``None``} | :class:`int` >= 0
                Number of columns to skip before first data column
            *MaxRows*, *maxrows*: {``None``} | :class:`int` > *skiprows*
                Maximum row number of data
            *MaxCols*, *maxcols*: {``None``} | :class:`int` > *skipcols*
                Maximum column number of data
            *warn*: {``True``} | ``False``
                Whether or not to warn about unused keyword args
        :Outputs:
            *kwread*: :class:`dict`
                Dictionary of options mentioned above
        :Versions:
            * 2019-12-12 ``@ddalle``: First version
        """
        # Check inputs
        if not isinstance(kw, dict):
            raise TypeError("Keyword input must be dict")
        # Initialize options
        kwread = dict(
            maxcols=self.process_xls_opt(["MaxCols", "maxcols"],  kw, None),
            maxrows=self.process_xls_opt(["MaxRows", "maxrows"],  kw, None),
            subrows=self.process_xls_opt(["SubRows", "subrows"],  kw, 0),
            skipcols=self.process_xls_opt(["SkipCols", "skipcols"], kw, None),
            skiprows=self.process_xls_opt(["SkipRows", "skiprows"], kw, None))
        # Output
        return kwread

    # Process number of rows to skip or similar
    def process_xls_opt(self, K, kw, vdef=None):
        r"""Process an option for reading XLS files

        :Call:
            >>> v = db.process_xls_opt(K, kw, vdef=None)
            >>> v = db.process_xls_opt([k1, k2], kw, vdef=None)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xls.XLSFile`
                XLS file interface
            *K*: :class:`list`\ [:class:`str`]
                List of key names
            *kw*: :class:`dict`
                Keyword options to parent function call
            *vdef*: {``None``} | :class:`object`
                Default value if not found in *kw* or *db.opts*
        :Outputs:
            *v*: {*kw[k1]*} | *kw[k2]* | *db.opts[k1]* | *vdef*
                Appropriate option by cascading preference
        :Versions:
            * 2019-12-12 ``@ddalle``: First version
        """
        # Check type
        if not isinstance(K, list):
            raise TypeError("Key names must be specified as list")
        elif len(K) == 0:
            raise ValueError("Empty key name list")
        elif not isinstance(kw, dict):
            raise TypeError("Keyword input must be dict")
        # Primary key name
        kref = K[0]
        # Option value
        v = self.opts.get(kref, vdef)
        # Loop through kwargs in reverse order
        for k in reversed(K):
            # Pop option from *kw*
            v = kw.pop(k, v)
        # Save option
        self.opts[kref] = v
        # Output
        return v
  # >
