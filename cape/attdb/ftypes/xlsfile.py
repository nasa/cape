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

# Standard library
import copy

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
from .basefile import BaseFile, BaseFileDefn, BaseFileOpts


# Options
class XLSFileOpts(BaseFileOpts):
   # --- Global Options ---
    # Option list
    _optlist = {
        "ColSpec",
        "MaxCols",
        "MaxRows",
        "SkipCols",
        "SkipRows",
        "SkipSheets",
        "SubCols",
        "SubRows",
        "WorksheetOptions",
        "WorksheetPrefix",
        "sheet"
    }

    # Alternate names
    _optmap = {
        "WorksheetOpts": "WorksheetOptions",
        "colspec": "ColSpec",
        "maxcols": "MaxCols",
        "maxrows": "MaxRows",
        "sheetopts": "WorksheetOptions",
        "skipcols": "SkipCols",
        "skiprows": "SkipRows",
        "skipsheets": "SkipSheets",
        "subcols": "SubCols",
        "subrows": "SubRows",
        "wsprefix": "WorksheetPrefix",
        "wsprefixes": "WorksheetPrefixes",
        "worksheetprefix": "WorksheetPrefix",
        "worksheetprefixes": "WorksheetPrefixes",
    }

   # --- Types ---
    # Allowed types
    _opttypes = {
        "ColSpec": (list, tuple),
        "MaxCols": int,
        "MaxRows": int,
        "SubCols": int,
        "SubRows": int,
        "SkipCols": int,
        "SkipRows": int,
        "SkipSheets": (set, list),
        "WorksheetOptions": dict,
        "WorksheetPrefix": bool,
        "WorksheetPrefixes": dict,
    }


# Combine options with parent class
XLSFileOpts.combine_optdefs()


# Options
class XLSSheetOpts(XLSFileOpts):
   # --- Global Options ---
    # Disallowed options
    _optrm = {
        "ExpandScalars",
        "Definitions",
        "SkipSheets",
        "Values",
        "WorksheetOptions",
        "sheet"
    }

    # Reduce option list
    _optlist = set.difference(XLSFileOpts._optlist, _optrm)


# Definition
class XLSFileDefn(BaseFileDefn):
   # --- Global Options ---
    # Option list
    _optlist = set.union(BaseFileDefn._optlist,
        {
            "ColWidth"
        })

    # Alternate names
    _optmap = dict(BaseFileDefn._optmap,
        colwidth="ColWidth")

   # --- Types ---
    # Allowed types
    _opttypes = dict(BaseFileDefn._opttypes,
        ColWidth=int)


# Options for CSVFile.write_csv()
class _WriteXLSOpts(XLSSheetOpts):
  # ===================
  # Class Attributes
  # ===================
  # <
   # --- Global Options ---
    # List of options
    _optlist = {
        "ColMasks",
        "SheetCols",
        "SheetNames",
        "SheetPostWriters",
        "SheetPreWriters",
        "SheetWriters",
        "SheetWritersSelfArg",
        "SheetWritersWorksheetArg",
        "TransposeCols"
    }

    # Alternate names
    _optmap = {
        "colmasks": "ColMasks",
        "sheets": "SheetNames",
        "sheetcols": "SheetCols",
        "sheetpostwriters": "SheetPostWriters",
        "sheetprewriters": "SheetPreWriters",
        "sheetwriters": "SheetWriters",
        "sheetwriterspost": "SheetPostWriters",
        "sheetwriterspre": "SheetPreWriters",
        "sheetwritersself": "SheetWritersSelf",
        "sheetwritersworksheet": "SheetWritersWorksheetArg",
        "transposecols": "TransposeCols",
    }

   # --- Types ---
    # Types allowed
    _opttypes = {
        "ColMasks": dict,
        "SheetCols": dict,
        "SheetNames": list,
        "SheetWriters": dict,
        "SheetPreWriters": dict,
        "SheetPostWriters": dict,
        "SheetWritersSelfArg": (list, set),
        "SheetWritersWorksheetArg": (list, set),
        "TransposeCols": (list, set),
    }

   # --- Defaults ---
    _rc = {
        "SheetNames": ["Sheet1"],
        "TransposeCols": set(),
    }
  # >


# Combine options
_WriteXLSOpts.combine_optdefs()
# Add definition support to option
XLSFileOpts.set_defncls(XLSFileDefn)


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
        *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
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
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Options ---
    # Class for options
    _optscls = XLSFileOpts
  # >

  # =============
  # Config
  # =============
  # <
    # Initialization method
    def __init__(self, fname=None, sheet=None, **kw):
        r"""Initialization method

        :Versions:
            * 2019-12-12 ``@ddalle``: First version
            * 2019-12-26 ``@ddalle``: Follow TNA/S-24
        """
        # Initialize options
        self.cols = []
        self.n = 0
        self._n = {}
        self.fname = None

        # Options for each sheet
        self.opts_by_sheet = {}

        # Process keyword arguments
        self.opts = self.process_kw(sheet=sheet, **kw)

        # Reassess worksheet, in case it got lost in *kw*
        sheet = self.opts.get_option("sheet")

        # Read file if appropriate
        if fname:
            # Read file
            self.read_xls(fname, sheet=sheet)
        else:
            # Process input column defs
            self.finish_defns()

        # Check for overrides of values
        self.process_kw_values()
  # >

  # ================
  # Options
  # ================
  # <
   # --- Worksheet Options ---
    # Get options for a specific worksheet, combining *kw*
    def get_worksheet_opts(self, sheet, **kw):
        r"""Get or create specific options for each worksheet

        :Call:
            >>> opts = db.get_worksheet_opts(sheet, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *sheet*: :class:`str`
                Name of worksheet in question
            *kw*: :class:`dict`
                Keyword options valid to *db._optscls*
        :Outputs:
            *opts*: :class:`XLSFileOpts` | *db._optscls*
                Combined options from *db.opts* and *kw*
        :Effects:
            *db.opts_by_sheet[sheet]*: :class:`XLSFileOpts`
                Set to *opts*
        :Versions:
            * 2020-02-06 ``@ddalle``: First version
        """
        # Check if present
        opts = self.opts_by_sheet.get(sheet)
        # If present, exit
        if not isinstance(opts, XLSSheetOpts):
            # Create a copy of global options
            opts = copy.deepcopy(self.opts)
            # Check for "WorksheetOptions"
            wbopts = opts.pop("WorksheetOptions", {})
            # Get options from there
            wsopts = wbopts.get(sheet)
            # Eliminate other options
            for k in XLSSheetOpts._optrm:
                opts.pop(k, None)
            # Convert to worksheet options
            opts = XLSSheetOpts(**opts)
            # If present, apply them
            if wsopts and isinstance(wsopts, dict):
                # Apply WorksheetOptions with checks
                opts.update(**wsopts)
            # Save the options for future use
            self.opts_by_sheet[sheet] = opts
        # Apply *kw*
        opts.update(**kw)
        # Output
        return opts
  # >

  # ================
  # Read
  # ================
  # <
   # --- Control ---
    # Reader
    def read_xls(self, fname, sheet=None, **kw):
        r"""Read an ``.xls`` or ``.xlsx`` file

        :Call:
            >>> db.read_xls(fname, sheet, **kw)
            >>> db.read_xls(wb, sheet, **kw)
            >>> db.read_xls(wb, **kw)
            >>> db.read_xls(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *fname*: :class:`str`
                Name of ``.xls`` or ``.xlsx`` file to read
            *sheet*: {``0``} | :class:`int` | :class:`str`
                Worksheet name or number
            *wb*: :class:`xlrd.book.Book`
                Open workbook (spreadsheet file)
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
        :Versions:
            * 2019-12-12 ``@ddalle``: First version
            * 2019-12-26 ``@ddalle``: Support "array" worksheets
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Update options if needed
        self.opts.update(**kw)
        # Check module
        if xlrd is None:
            raise ImportError("No module 'xlrd'")
        # Initialize ws
        ws = None

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
            
        # Read worksheet if possible, else read workbook
        if ws:
            # Read directly-specified worksheet
            self.read_xls_worksheet(ws)
        else:
            # Read all worksheets
            self.read_xls_workbook(wb)

    # Read a worksheet
    def read_xls_workbook(self, wb, **kw):
        r"""Read ``.xls`` or ``.xlsx`` workbook with multiple worksheets

        :Call:
            >>> db.read_xls_workbook(wb, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *wb*: :class:`xlrd.Book`
                Direct access to a workbook
            *ndim*: {``0``} | ``1``
                Dimensionality of one row of data column(s) to read
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
            * 2020-01-08 ``@jmeeroff`` : First version
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Combine options if needed
        self.opts.update(**kw)
        # Get sheets to skip
        skipsheets = self.opts.get_option("SkipSheets", set())
        # Option for automatic prefix
        wsprefix = self.opts.get_option("WorksheetPrefix", True)
        wsprefixes = self.opts.get_option("WorksheetPrefixes", {})
        # Check number of sheets
        if wb.nsheets == 1:
            # Get the first (only) worksheet
            ws = wb.sheet_by_index(0)
            # Read it without prefixing column names
            self.read_xls_worksheet(ws)
        else:
            # Save original prefix
            _prefix = self.opts.get("Prefix")
            # Loop through worksheets
            for wsname in wb.sheet_names():
                # Check for explicit instruction to skip
                if wsname in skipsheets:
                    continue
                # Try to read worsheet; do nothing if fails
                try:
                    # Check for prefix option
                    if wsprefix:
                        # Use worksheet name as prefix to avoid name clash
                        prefix = wsname + "."
                    else:
                        # Default to no prefix
                        prefix = ""
                    # Check for explicit prefix
                    prefix = wsprefixes.get(wsname, prefix)
                    # Apply prefix
                    self.opts['Prefix'] = prefix
                    # Get the worksheet handle
                    ws = wb.sheet_by_name(wsname)
                    # Read it
                    self.read_xls_worksheet(ws)
                except Exception:
                    pass
            # Reset original prefix
            if _prefix is None:
                # Remove any that was added
                self.opts.pop("Prefix", None)
            else:
                # Reset original
                self.opts["Prefix"] = _prefix

    # Read a worksheet
    def read_xls_worksheet(self, ws, **kw):
        r"""Read one worksheet of an ``.xls`` or ``.xlsx`` file

        :Call:
            >>> db.read_xls_worksheet(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
        :Versions:
            * 2019-12-12 ``@ddalle``: First version
            * 2019-12-26 ``@ddalle``: Support "array" worksheets
            * 2020-01-10 ``@jmeeroff``: Array read as fallback
            * 2020-01-16 ``@ddalle``: Unified two worksheet methods
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Read header
        cols = self.read_xls_header(ws, **kw)
        # Process definitions
        self.finish_defns()
        # Read data
        self.read_xls_coldata(ws, cols)

   # --- Header ---
    # Read worksheet header
    def read_xls_header(self, ws, **kw):
        r"""Read header row from a worksheet

        :Call:
            >>> cols = db.read_xls_header(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *colspec*: {``None``} | :class:`list`
                List of column names *col* or ``(col, colwidth)``;
                using this option preempts reading column names
                from header row
            *skiprows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *subrows*: {``None``} | :class:`int` >= 0
                Number of rows below header row to skip
            *skipcols*: {``None``} | :class:`int` >= 0
                Number of columns to skip before first data column
            *maxcols*: {``None``} | :class:`int` > *skipcols*
                Maximum column number of data
        :Outputs:
            *cols*: :class:`list`\ [:class:`str`]
                List of column names if read
        :Versions:
            * 2019-12-12 ``@ddalle``: First version
            * 2020-01-16 ``@ddalle``: Full scalar/array support
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Process skip options
        opts = self.get_autoskip(ws, **kw)
        # Check for directly specified columns
        colspec = opts.get_option("ColSpec", {})
        # Unpack skip options
        skipcols = opts["SkipCols"]
        skiprows = opts["SkipRows"]
        subrows = opts["SubRows"]
        maxcols = opts["MaxCols"]
        maxrows = opts["MaxRows"]
        # Read the header row
        header = ws.row_values(
            skiprows, start_colx=skipcols, end_colx=maxcols)
        # Read the first data row
        row1 = ws.row_values(
            skiprows + subrows + 1, start_colx=skipcols, end_colx=maxcols)
        # Number of cols read
        nheader = len(header)
        # Process column specification
        if isinstance(colspec, (list, tuple)):
            # Initialize columns and widths
            cols = []
            dim2 = []
            # Column
            j = 0
            # Loop through columns specified by user
            for spec in colspec:
                # Check type
                if typeutils.isstr(spec):
                    # Save a scalar
                    col = spec
                    colwidth = 1
                elif isinstance(spec, (list, tuple)):
                    # Check length
                    if len(spec) != 2:
                        raise ValueError(
                            "Col specification must have length 2 (got %i)"
                            % len(spec))
                    # Unpack
                    col, colwidth = spec
                # Check length of row
                if j + colwidth > nheader:
                    raise ValueError(
                        ("Column specification for '%s' exceeds " % col) +
                        ("number of data columns in worksheet"))
                # Translate name
                col, = self.translate_colnames([col])
                # Get definition
                defn = self.get_defn(col)
                # Save
                cols.append(col)
                # Get value
                v = row1[j]
                # Check type
                if isinstance(v, float):
                    # Numeric types are all floats in Excel
                    defn.setdefault("Type", "float64")
                else:
                    # Otherwise string
                    defn.setdefault("Type", "str")
                # Save dimension
                defn["ColWidth"] = colwidth
                # Move to next data column
                j += colwidth
            # Terminate early
            return cols
        # Initialize list of columns
        cols = []
        # Number of spreadsheet columns in each field
        dim2 = []
        # List of data types
        dtypes = []
        # Flag for previous array
        array_last = False
        # Start with first column
        j = 0
        # Loop through raw header fields
        while (j < nheader):
            # Get value
            col = header[j]
            # Get value for this column
            v = row1[j]
            # True column number
            jcol = j + 1 + skipcols
            # Check type
            if (j > 0) and isinstance(col, float):
                # Check if it's an int (xls files have no ints)
                if col % 1.0 > 1e-8:
                    # Can't have 4.1 array length, etc.
                    raise TypeError(
                        "Column %i header is not a string or int" % jcol)
                # Convert to int
                col_nrow = int(col)
                # Check length (off by 1 since describing previous col)
                if skipcols + j + col_nrow > maxcols + 1:
                    raise ValueError(
                        "Array len in col %i exceeds worksheet width" % jcol)
                # Update the previous entry's width
                dim2[-1] = col_nrow
                # Increment column
                j += col_nrow - 1
                # Go to next column (allows one dedent below)
                continue
            elif typeutils.isstr(col):
                # Check for empty string
                if col.strip() == "":
                    # Get type
                    if isinstance(v, float):
                        # All numbers are floats in Excel
                        dtype = "float64"
                    else:
                        # Everything else is a string
                        dtype = "str"
                    # Check for same type as previous column
                    if (j > 0) and (dtype == dtypes[-1]):
                        # Continuation of previous column
                        dim2[-1] += 1
                        # Go to next column
                        j += 1
                        continue
                    # Not a continuation: use column number
                    col = "col%i" % (j+1)
                # Check if this column was previously used
                while col in cols:
                    col = col + "_"
            else:
                # Column name invalid
                raise TypeError("Column %i header is not a string" % jcol)
            # Translate name
            col, = self.translate_colnames([col])
            # Save column name
            cols.append(col)
            # Save it as a scalar for now
            dim2.append(1)
            # Get definition for new column
            defn = self.get_defn(col)
            # Filter its type
            if isinstance(v, float):
                # Convert float type
                dtype = defn.setdefault("Type", "float64")
            else:
                # Only float or string
                dtype = defn.setdefault("Type", "str")
            # Save data type
            dtypes.append(dtype)
            # Increment column
            j += 1
        # Save column names
        for (j, col) in enumerate(cols):
            # Check if present
            if col not in self.cols:
                # Append if not
                self.cols.append(col)
            # Get the definition
            defn = self.get_defn(col)
            # Set the definition
            defn["ColWidth"] = dim2[j]
        # Output
        return cols

    # Default *skiprows* and *skipcols*
    def get_autoskip(self, ws, **kw):
        r"""Automatically determine number of rows and columns to skip

        :Call:
            >>> wsopts = db.get_autoskip(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *skiprows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *subcols*: {``0``} | :class:`int` >= 0
                Number of cols to skip right of first col (for arrays)
            *subrows*: {``0``} | :class:`int` >= 0
                Number of rows below header row to skip
            *skipcols*: {``None``} | :class:`int` >= 0
                Number of columns to skip before first data column
            *maxcols*: {``None``} | :class:`int` > *skipcols*
                Maximum column number of data
        :Outputs:
            *wsopts*: :class:`XLSSheetOpts`
                Options from *db.opts_by_sheet[ws.name]*,
        :Versions:
            * 2019-12-26 ``@ddalle``: Split from :func:`read_xls_header`
            * 2020-01-14 ``@ddalle``: Moved everything to smaller funcs
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Same as :func:`_get_skip` but with options output
        self._get_skip(ws, **kw)
        # Output the options
        return self.get_worksheet_opts(ws.name)

    # Get all skip and max options
    def _get_skip(self, ws, **kw):
        r"""Determine number of rows and columns to skip

        This method saves a marginal amount of time by search for the
        various skip and max parameters in the optimal order.

        :Call:
            >>> skips = db._get_skip(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *skiprows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *subcols*: {``0``} | :class:`int` >= 0
                Number of cols to skip right of first col (for arrays)
            *subrows*: {``0``} | :class:`int` >= 0
                Number of rows below header row to skip
            *skipcols*: {``None``} | :class:`int` >= 0
                Number of columns to skip before first data column
            *maxcols*: {``None``} | :class:`int` > *skipcols*
                Maximum column number of data
            *maxrows*: {``None``} | :class:`int` > *skiprows*
                Maximum row number of data
        :Outputs:
            *skips*: :class:`dict`\ [:class:`int`]
                Dict of the following outputs
            *skiprows*: {``0``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *subrows*: {``0``} | :class:`int` >= 0
                Number of rows below header row to skip
            *maxrows*: {*ws.nrows*} | :class:`int` > *skiprows*
                Maximum row number of data
            *skipcols*: {``0``} | :class:`int` >= 0
                Number of columns to skip before first data column
            *subcols*: {``0``} | :class:`int` >= 0
                Number of cols to skip right of first col (for arrays)
            *maxcols*: {*ws.ncols*} | :class:`int` > *skipcols*
                Maximum column number of data
        :Versions:
            * 2020-01-14 ``@ddalle``: First version
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Get maximum extents
        maxcols = self._get_maxcols(ws, **kw)
        maxrows = self._get_maxrows(ws, **kw)
        # Get pre-header row count
        skiprows = self._get_skiprows(ws, **kw)
        # Get pre-data col count
        skipcols = self._get_skipcols(ws, **kw)
        # Get empty rows/cols between header and data
        subrows = self._get_subrows(ws, **kw)
        subcols = self._get_subcols(ws, **kw)
        # Output
        return {
            "skiprows": skiprows,
            "subrows": subrows,
            "maxrows": maxrows,
            "skipcols": skipcols,
            "subcols": subcols,
            "maxcols": maxcols
        }

    # Process *maxrows*
    def _get_maxrows(self, ws, **kw):
        r"""Automatically determine maximum number of cols

        :Call:
            >>> maxrows = db._get_maxrows(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *maxrows*, *MaxRows*: {``None``} | :class:`int` > 0
                Maximum number of rows to allow
        :Outputs:
            *maxrows*: {*ws.nrows*} | :class:`int` >= 0
                Maximum number of rows to allow
        :Versions:
            * 2019-12-26 ``@ddalle``: Split from :func:`read_xls_header`
            * 2020-01-13 ``@ddalle``: Split from :func:`get_autoskip`
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Get worksheet options
        opts = self.get_worksheet_opts(ws.name, **kw)
        # Check for explicit option
        maxrows = opts.get_option("MaxRows")
        # Find header row if needed
        if maxrows is None:
            # Use worksheet size
            maxrows = ws.nrows
            # Set it
            opts._set_option("MaxRows", maxrows)
        elif isinstance(maxrows, int):
            # Check value
            if maxrows < 1:
                # Negative skip?
                raise ValueError("Cannot have %i rows" % maxrows)
        else:
            raise TypeError("'maxrows' arg must be None or int")
        # Output (convenience)
        return maxrows

    # Process *maxcols*
    def _get_maxcols(self, ws, **kw):
        r"""Automatically determine maximum number of cols

        :Call:
            >>> maxcols = db._get_maxcols(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *maxcols*, *MaxCols*: {``None``} | :class:`int` > 0
                Maximum number of columns to allow
        :Outputs:
            *maxcols*: {*ws.ncols*} | :class:`int` >= 0
                Maximum number of columns to allow
        :Versions:
            * 2019-12-26 ``@ddalle``: Split from :func:`read_xls_header`
            * 2020-01-13 ``@ddalle``: Split from :func:`get_autoskip`
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Get worksheet options
        opts = self.get_worksheet_opts(ws.name, **kw)
        # Check for explicit option
        maxcols = opts.get_option("MaxCols")
        # Find header row if needed
        if maxcols is None:
            # Use worksheet size
            maxcols = ws.ncols
            # Set it
            opts._set_option("MaxCols", maxcols)
        elif isinstance(maxcols, int):
            # Check value
            if maxcols < 1:
                # Negative skip?
                raise ValueError("Cannot have %i columns" % maxcols)
        else:
            raise TypeError("'maxcols' arg must be None or int")
        # Output
        return maxcols

    # Process *skiprows*
    def _get_skiprows(self, ws, **kw):
        r"""Automatically determine number of rows to skip

        :Call:
            >>> skiprows = db._get_skiprows(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *skiprows*, *SkipRows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *maxcols*, *MaxCols*: {``None``} | :class:`int` > 0
                Maximum number of columns to allow
        :Outputs:
            *skiprows*: :class:`int` >= 0
                Number of rows to skip before reading data
        :Versions:
            * 2019-12-26 ``@ddalle``: Split from :func:`read_xls_header`
            * 2020-01-13 ``@ddalle``: Split from :func:`get_autoskip`
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Get worksheet options
        opts = self.get_worksheet_opts(ws.name, **kw)
        # Check for explicit option
        skiprows = opts.get_option("SkipRows")
        # Find header row if needed
        if skiprows is None:
            # Process maximum column count
            # (can be relevant in worksheets with two or more tables)
            maxcols = self._get_maxcols(ws, **kw)
            # Check for *skipcols* option, but don't recurse!
            skipcols = opts.get_option("SkipCols")
            # Loop until we have an empty row
            for skiprows in range(ws.nrows):
                # Read the row
                header = ws.row_values(
                    skiprows, start_colx=skipcols, end_colx=maxcols)
                # Check if there's anything in it
                if any(header):
                    break
            else:
                # This means an empty worksheet or only has "" and 0.0
                raise ValueError("No nonempty rows found")
            # Save option
            opts._set_option("SkipRows", skiprows)
        elif isinstance(skiprows, int):
            # Check value
            if skiprows < 0:
                # Negative skip?
                raise ValueError("Cannot skip %i rows" % skiprows)
            elif skiprows >= ws.nrows:
                # Skip the whole worksheet
                raise ValueError(
                    "Cannot skip %i rows in worksheet with %i rows"
                    % (skiprows, ws.nrows))
        else:
            raise TypeError("'skiprows' arg must be None or int")
        # Output
        return skiprows

    # Process *skipcols*
    def _get_skipcols(self, ws, **kw):
        r"""Automatically determine number of cols to skip

        :Call:
            >>> skipcols = db._get_skipcols(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *skiprows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *skipcols*: {``None``} | :class:`int` >= 0
                Number of cols to skip before reading data
        :Outputs:
            *skipcols*: :class:`int` >= 0
                Number of cols to skip before reading data
        :Versions:
            * 2019-12-26 ``@ddalle``: Split from :func:`read_xls_header`
            * 2020-01-13 ``@ddalle``: Split from :func:`get_autoskip`
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Get worksheet options
        opts = self.get_worksheet_opts(ws.name, **kw)
        # Get option
        skipcols = opts.get_option("SkipCols")
        # Find header column if needed
        if skipcols is None:
            # Get number of rows to skip
            skiprows = self._get_skiprows(ws, **kw)
            # Get last column
            maxcols = self._get_maxcols(ws, **kw)
            # Read header row
            header = ws.row_values(skiprows, end_colx=maxcols)
            # Find first nonempty entry
            for skipcols in range(len(header)):
                # Check for entry
                if header[skipcols] != "":
                    # Found something other than ""
                    break
            else:
                # Empty header plausible if *skiprows* is bad
                raise ValueError("No nonempty columns in row %i" % skiprows)
            # Save option
            opts._set_option("SkipCols", skipcols)
        elif isinstance(skipcols, int):
            # Check value
            if skipcols < 0:
                # Negative skip?
                raise ValueError("Cannot skip %i cols" % skipcols)
            elif skipcols >= ws.ncols:
                # Skip the whole worksheet
                raise ValueError(
                    "Cannot skip %i cols in worksheet with %i cols"
                    % (skipcols, ws.ncols))
        else:
            raise TypeError("'skipcols' arg must be None or int")
        # Output (for convenience)
        return skipcols

    # Process *subrows*
    def _get_subrows(self, ws, **kw):
        r"""Determine number of rows to skip *below* header row

        :Call:
            >>> subrows = db._get_subrows(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *subrows*, *SubRows*: {``None``} | :class:`int` >= 0
                Number of rows between header row and first data row
            *skiprows*, *SkipRows*: {``None``} | :class:`int` >= 0
                Number of rows to skip before reading data
            *maxcols*, *MaxCols*: {``None``} | :class:`int` > 0
                Maximum number of columns to allow
        :Outputs:
            *subrows*, *SubRows*: {``0``} | :class:`int` >= 0
                Number of rows between header row and first data row
        :Versions:
            * 2020-01-14 ``@ddalle``: First version
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Get worksheet options
        opts = self.get_worksheet_opts(ws.name, **kw)
        # Get option
        subrows = opts.get_option("SubRows")
        # Find rows between header and data if needed
        if subrows is None:
            # Get header skip rows
            skiprows = self._get_skiprows(ws, **kw)
            # Process maximum column count
            # (can be relevant in worksheets with two or more tables)
            maxcols = self._get_maxcols(ws, **kw)
            # Loop until we have an empty row
            for subrows in range(ws.nrows - skiprows):
                # Read the row
                row1 = ws.row_values(skiprows + subrows + 1, end_colx=maxcols)
                # Check if there's anything in it
                if any(row1):
                    break
            else:
                # This means the worksheet only has a header row
                subrows = 0
            # Save option
            opts._set_option("SubRows", subrows)
        elif isinstance(subrows, int):
            # Check value
            if subrows < 0:
                # Negative skip?
                raise ValueError("Cannot skip %i rows" % subrows)
            elif subrows >= ws.nrows:
                # Skip the whole worksheet
                raise ValueError(
                    "Cannot skip %i rows in worksheet with %i rows"
                    % (subrows, ws.nrows))
        else:
            raise TypeError("'subrows' arg must be None or int")
        # Output
        return subrows

    # Process *subcols*
    def _get_subcols(self, ws, **kw):
        r"""Determine number of cols to skip *after* first column

        :Call:
            >>> subcols = db._get_subcols(ws, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *ws*: :class:`xlrd.sheet.Sheet`
                Direct access to a worksheet
            *subcols*, *SubCols*: {``None``} | :class:`int` >= 0
                Number of cols between header col and first data col
        :Outputs:
            *subcols*, *SubCols*: {``None``} | :class:`int` >= 0
                Number of cols between header col and first data col
        :Versions:
            * 2020-01-14 ``@ddalle``: First version
            * 2020-02-07 ``@ddalle``: Using :class:`XLSSheetOpts`
        """
        # Get worksheet options
        opts = self.get_worksheet_opts(ws.name, **kw)
        # Get option
        subcols = opts.get_option("SubCols")
        # Find rows between header and data if needed
        if subcols is None:
            # Default
            subcols = 0
            # Save option
            opts._set_option("SubCols", subcols)
        elif isinstance(subcols, int):
            # Check value
            if subcols < 0:
                # Negative skip?
                raise ValueError("Cannot skip %i cols" % subcols)
            elif subcols >= ws.ncols:
                # Skip the whole worksheet
                raise ValueError(
                    "Cannot skip %i cols in worksheet with %i rows"
                    % (subcols, ws.ncols))
        else:
            raise TypeError("'subcols' arg must be None or int")
        # Output
        return subcols


   # --- Data ---
    # Read data
    def read_xls_coldata(self, ws, cols, **kw):
        r"""Read column data from one ``.xls`` or ``.xlsx`` worksheet

        :Call:
            >>> db.read_xls_coldata(ws, cols, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
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
        # Get worksheet options
        opts = self.get_worksheet_opts(ws.name)
        # Apply any updates
        opts.update(**kw)
        # Get skip options
        skiprows = opts.get_option("SkipRows")
        skipcols = opts.get_option("SkipCols")
        # Maximum option
        maxcols = opts.get_option("MaxCols")
        maxrows = opts.get_option("MaxRows")
        # Sub-header gaps
        subcols = opts.get_option("SubCols")
        subrows = opts.get_option("SubRows")
        # Get counts
        _n = self.__dict__.setdefault("_n", {})
        # First data row number
        irow = skiprows + subrows + 1
        # First data col number
        icol = skipcols + subcols
        # Loop through columns
        for col in cols:
            # Get data column definition
            defn = self.get_defn(col)
            # Get data class name
            clsname = defn.get("Type", "float64")
            # Get data type, if different
            dtype = defn.get("DType", clsname)
            # Get array length
            colwidth = defn.get("ColWidth", 1)
            # Read the whole column
            V0 = ws.col_values(icol, irow, end_rowx=maxrows)
            # Read data based on type
            if dtype == "str":
                # Read the whole column and allow empty strings
                V = V0
                # Save the full length
                _n[col] = len(V)
            elif dtype.startswith("float") or dtype.startswith("int"):
                # Check for empty strings
                if "" in V0:
                    # Find index of first such one
                    iend = V0.index("")
                    # Check for empty column
                    if iend == 0:
                        raise ValueError(
                            "Found no valid floats in col %i" % icol)
                    # Strip trailing entries
                    V0 = V0[:iend]
                # Current size
                m = len(V0)
                # Create array
                if colwidth == 1:
                    # Convert to array
                    V = np.array(V0, dtype=dtype)
                    # Save size
                    _n[col] = V.size
                else:
                    # Initialize array
                    V = np.zeros((m, colwidth), dtype=dtype)
                    # Save first column
                    V[:,0] = V0
                    # Loop through other columns
                    for jcol in range(1, colwidth):
                        # Read the values
                        Vj = ws.col_values(icol+jcol, irow, end_rowx=irow+m)
                        # Save column
                        V[:,jcol] = Vj
            # Go to next column
            icol += colwidth
            # Save
            self.save_col(col, V)
  # >

  # ================
  # Write
  # ================
  # <
   # --- Write Drivers ---
    # Write a workbook
    def write_xls(self, fname, cols=None, **kw):
        r"""Write data to one or more worksheets

        :Call:
            >>> db.write_xls(fname, cols=None, **kw)
            >>> db.write_xls(wb, cols=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *fname*: {*db.fname*} | :class:`str`
                Name of XLS file to create
            *wb*: :class:`xlsxwriter.workbook.Workbook`
                Workbook (write) interface
            *cols*: :class:`list`\ [:class:`str`]
                List of data columns to write
            *SheetNames* | *sheets*: :class:`list`\ [:class:`str`]
                List of worksheets to write
            *SheetCols* | *sheetcols*: :class:`dict`\ [:class:`list`]
                List of columns to write to each worksheet
            *SheetWriters*: :class:`dict`\ [**callable**]
                Handles to standalone writers for some worksheets
            *SheetPreWriters*: :class:`dict`\ [**callable**]
                Functions to write some things to worksheet before also
                writing data columns
            *SheetPostWriters*: :class:`dict`\ [**callable**]
                Functions to write to worksheet after writing data
            *SheetWritersSelfArg*: :class:`list` | :class:`set`
                List/set of sheets whose *Writers* take *db* as an arg
            *SheetWritersWorksheetArg*: :class:`list` | :class:`set`
                List/set of sheets whose *Writers* take worksheet
                instead of workbook as arg
            *SkipCols*: {``0``} | :class:`int` > 0
                Number of columns to skip before writing
            *SkipRows*: {``0``} | :class:`int` > 0
                Number of rows to skip before writing
            *ColMasks*: {``{}``} | :class:`dict`
                Masks (subset indices) for some *cols*
            *Translators*: {``{}``} | :class:`dict`
                Column name translators (applied in reverse)
            *Prefix*: {``None``} | :class:`dict` | :class:`str`
                Prefix (applied in reverse)
            *Suffix*: {``None``} | :class:`dict` | :class:`str`
                Suffix (applied in reverse)
        :Versions:
            * 2020-05-21 ``@ddalle``: First version
        """
        # Get file handle based on input type
        if fname is None:
            # Use *db.fname*
            wb = xlsxwriter.Workbook(self.fname)
            # Close it upon completion
            qclose = True
        elif typeutils.isstr(fname):
            # Open new workbook
            wb = xlsxwriter.Workbook(fname)
            # Close it upon completion
            qclose = True
        elif isinstance(fname, xlsxwriter.Workbook):
            # Already a workbook
            wb = fname
            # Don't close it upon completion
            qclose = False
        else:
            # Not a recognized type
            raise TypeError(
                "Unrecognized type %s for XLS output file" % type(fname))
        # Run the primary writer
        self._write_xls(wb, cols, **kw)
        # Close the workbook
        if qclose:
            wb.close()

    # Write a workbook
    def _write_xls(self, wb, cols=None, **kw):
        r"""Write data to one or more worksheets

        :Call:
            >>> db._write_xls(wb, cols=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *wb*: :class:`xlsxwriter.workbook.Workbook`
                Workbook (write) interface
            *cols*: :class:`list`\ [:class:`str`]
                List of data columns to write
            *SheetNames* | *sheets*: :class:`list`\ [:class:`str`]
                List of worksheets to write
            *SheetCols* | *sheetcols*: :class:`dict`\ [:class:`list`]
                List of columns to write to each worksheet
            *SheetWriters*: :class:`dict`\ [**callable**]
                Handles to standalone writers for some worksheets
            *SheetPreWriters*: :class:`dict`\ [**callable**]
                Functions to write some things to worksheet before also
                writing data columns
            *SheetPostWriters*: :class:`dict`\ [**callable**]
                Functions to write to worksheet after writing data
            *SheetWritersSelfArg*: :class:`list` | :class:`set`
                List/set of sheets whose *Writers* take *db* as an arg
            *SheetWritersWorksheetArg*: :class:`list` | :class:`set`
                List/set of sheets whose *Writers* take worksheet
                instead of workbook as arg
            *SkipCols*: {``0``} | :class:`int` > 0
                Number of columns to skip before writing
            *SkipRows*: {``0``} | :class:`int` > 0
                Number of rows to skip before writing
            *ColMasks*: {``{}``} | :class:`dict`
                Masks (subset indices) for some *cols*
            *Translators*: {``{}``} | :class:`dict`
                Column name translators (applied in reverse)
            *Prefix*: {``None``} | :class:`dict` | :class:`str`
                Prefix (applied in reverse)
            *Suffix*: {``None``} | :class:`dict` | :class:`str`
                Suffix (applied in reverse)
        :Versions:
            * 2020-05-21 ``@ddalle``: First version
        """
       # --- Special Functions ---
        def _writer(self, sheet, fn, wb, ws):
            # Write the special worksheet
            if sheet in sheetwritersself:
                # Include database as an argument
                if sheet in sheetwritersws:
                    # Use database and worksheet handle
                    fn(self, ws)
                else:
                    # Use database and workbook handle
                    fn(self, wb)
            else:
                # Include database as an argument
                if sheet in sheetwritersws:
                    # USe worksheet handle alone
                    fn(ws)
                else:
                    # Just give the worksheet handle
                    fn(wb)
       # --- Column List ---
        # Get default list of columns
        if cols is None:
            # Use all columns
            cols = self.cols
        # Check input type
        if isinstance(wb, xlsxwriter.worksheet.Worksheet):
            # Already a worksheet
            ws = wb
            wb = None
        elif isinstance(wb, xlsxwriter.Workbook):
            # Workbook given (normal case)
            ws = None
        else:
            # Bad type
            raise TypeError(
                "Unrecognized type %s for XLS output file" % type(wb))
       # --- Options ---
        # Options handle
        opts = self.opts
        # Process kwargs
        kw = _WriteXLSOpts(**kw)
        # Dictionary of translators
        trans = opts.get_option("Translators", {})
        kw.setdefault("Translators", trans)
        # Check for single worksheet
        if ws is not None:
            # Write that worksheet
            self._write_xls_worksheet(ws, cols, **kw)
            # Done
            return
        # Get worksheet list
        sheets = kw.get_option("SheetNames")
        # Additional worksheet writers
        sheetwriters = kw.get_option("SheetWriters", {})
        sheetwriterspre = kw.get_option("SheetPreWriters", {})
        sheetwriterspost = kw.get_option("SheetPostWriters", {})
        # Ensure strings
        for j, sheet in enumerate(sheets):
            if not typeutils.isstr(sheet):
                raise TypeError("Worksheet %i is not a str" % sheet)
        # Worksheet columns
        sheetcols = kw.get_option("SheetCols", {})
        # Additional worksheet writers
        sheetwritersself = kw.get_option("SheetWritersSelfArg", set())
        sheetwritersws = kw.get_option("SheetWritersWorksheetArg", set())
        # Loop through worksheets
        for sheet in sheets:
            # Check if the worksheet is already present
            ws = wb.sheetnames.get(sheet)
            # Create new worksheet if needed
            if ws is None:
                ws = wb.add_worksheet(sheet)
            # Check for writer
            if sheet in sheetwriters:
                # Get writer
                fn = sheetwriters[sheet]
                # Write stand-alone worksheet
                _writer(self, sheet, fn, wb, ws)
                # Move to next sheet
                continue
            elif sheet in sheetwriterspre:
                # Get writer
                fn = sheetwriterspre[sheet]
                # Prepare sheet but also write data
                _writer(self, sheet, fn, wb, ws)
            # Get columns
            wscols = sheetcols.get(sheet, cols)
            # Write those columns to this sheet
            self._write_xls_worksheet(ws, wscols, **kw)
            # Check for a posteriori writer
            if sheet in sheetwriterspost:
                # Get writer
                fn = sheetwriterspost[sheet]
                # Prepare sheet but also write data
                _writer(self, sheet, fn, wb, ws)

   # --- Worksheet Writers ---
    # Write worksheet
    def _write_xls_worksheet(self, ws, cols, **kw):
        r"""Write data to one worksheet

        :Call:
            >>> db._write_xls_worksheet(ws, cols, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.xlsfile.XLSFile`
                XLS file interface
            *ws*: :class:`xlsxwriter.worksheet.Worksheet`
                Worksheet interface
            *cols*: :class:`list`\ [:class:`str`]
                List of data columns to write
            *SkipCols*: {``0``} | :class:`int` > 0
                Number of columns to skip before writing
            *SkipRows*: {``0``} | :class:`int` > 0
                Number of rows to skip before writing
            *ColMasks*: {``{}``} | :class:`dict`
                Masks (subset indices) for some *cols*
            *Translators*: {``{}``} | :class:`dict`
                Column name translators (applied in reverse)
            *Prefix*: {``None``} | :class:`dict` | :class:`str`
                Prefix (applied in reverse)
            *Suffix*: {``None``} | :class:`dict` | :class:`str`
                Suffix (applied in reverse)
        :Versions:
            * 2020-05-21 ``@ddalle``: First version
        """
        # Process kwargs
        kw = _WriteXLSOpts(**kw)
        # Get column and row to start on
        startcol = kw.get("SkipCols", 0)
        skiprows = kw.get("SkipRows", 0)
        subrows = kw.get("SubRows", 0)
        # Current col and row to write at
        irow = skiprows + subrows + 1
        jcol = startcol
        # Other options
        colmasks = kw.get_option("ColMasks", {})
        transposecols = kw.get_option("TransposeCols")
        # Instance column name translation
        trans = self.opts.get_option("Translators")
        prefix = self.opts.get_option("Prefix")
        suffix = self.opts.get_option("Suffix")
        # Check for overrides
        trans = kw.get_option("Translators", trans)
        prefix = kw.get_option("Prefix", prefix)
        suffix = kw.get_option("Suffix", suffix)
        # Loop through columns
        for col in cols:
            # Get formatting options...
            
            # Translate column names
            ocol = self._translate_colname_reverse(col, trans, prefix, suffix)
            # Write header
            ws.write(skiprows, jcol, ocol)
            # Check for subset option
            mask = colmasks.get(col)
            # Get values
            v = self[col]
            # Check for a mask
            if mask:
                # Apply it
                v = v[mask]
            # Check type
            if isinstance(v, list):
                # Simply write column
                ws.write_column(irow, jcol, v)
                # Increment column counter
                jcol += 1
            elif v.ndim == 1:
                # Write array to column
                ws.write_column(irow, jcol, v)
                # Increment column counter
                jcol += 1
            elif v.ndim == 2:
                # Check for transpose option
                if col in transposecols:
                    # Write rows instead of columns
                    for i in range(v.shape[0]):
                        # Write row *i* as a column
                        ws.write_column(irow, jcol + i, v[i])
                    # Increment column counter
                    jcol += v.shape[0]
                else:
                    # Write columns as columns
                    for j in range(v.shape[1]):
                        # Write col *j* as a column
                        ws.write_column(irow, jcol + j, v[:,j])
                    # Increment column counter
                    jcol += v.shape[1]
  # >
