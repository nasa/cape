#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.ftypes.csv`: Comma-separated value read/write
===============================================================

This module contains a basic interface in the spirit of
:mod:`cape.attdb.ftypes` for standard comma-separated value files.  It
creates a class, :class:`CSVFile` that does not rely on the popular
:func:`numpy.loadtxt` function.

If possible, the column names (which become keys in the
:class:`dict`-like class) are read from the header row.  If the file
begins with multiple comment lines, the column names are read from the
final comment before the beginning of data.
"""

# Standard library
import re
import sys

# CAPE modules
import cape.tnakit.typeutils as typeutils
import cape.tnakit.arrayutils as arrayutils

# Local modules
from .basefile import BaseFile, BaseFileDefn, BaseFileOpts, TextInterpreter

# Local extension
try:
    from . import _ftypes
except ImportError:
    _ftypes = None


# Regular expressions
regex_numeric = re.compile(r"\d")
regex_alpha   = re.compile("[A-z_]")

# Options
class CSVFileOpts(BaseFileOpts):
    pass


# Definition
class CSVFileDefn(BaseFileDefn):
    pass


# Add definition support to option
CSVFileOpts.set_defncls(CSVFileDefn)


# Class for handling data from CSV files
class CSVFile(BaseFile, TextInterpreter):
    r"""Class for reading CSV files
    
    :Call:
        >>> db = CSVFile(fname, **kw)
        >>> db = CSVFile(f, **kw)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *f*: :class:`file`
            Open file handle
    :Outputs:
        *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
            CSV file interface
        *db.cols*: :class:`list`\ [:class:`str`]
            List of columns read
        *db.opts*: :class:`CSVFileOpts`
            Options for this instance
        *db.defns*: :class:`dict`\ [:class:`CSVFileDefn`]
            Definitions for each column
        *db[col]*: :class:`np.ndarray` | :class:`list`
            Numeric array or list of strings for each column
    :See also:
        * :class:`cape.attdb.ftypes.basefile.BaseFile`
        * :class:`cape.attdb.ftypes.basefile.TextInterpreter`
    :Versions:
        * 2019-11-12 ``@ddalle``: First version
        * 2019-11-26 ``@ddalle``: Generic version
    """
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Options ---
    # Class for options
    _optscls = CSVFileOpts
  # >

  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, fname=None, **kw):
        """Initialization method
        
        :Versions:
            * 2019-11-12 ``@ddalle``: First version
        """
        # Initialize common attributes
        self.cols = []
        self.n = 0
        self.fname = None

        # Process keyword arguments
        self.opts = self.process_kw(**kw)

        # Explicit definition declarations
        self.get_defns()

        # Read file if appropriate
        if fname:
            # Read valid file
            self.read_csv(fname)
        else:
            # Apply defaults to definitions
            self.finish_defns()

        # Check for overrides of values
        self.process_kw_values()
  # >

  # =============
  # Read
  # =============
  # <
   # --- Control ---
    # Reader
    def read_csv(self, fname):
        r"""Read a CSV file, including header

        Reads either entire file or from current location
        
        :Call:
            >>> db.read_csv(f)
            >>> db.read_csv(fname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                File open for reading
            *fname*: :class:`str`
                Name of file to read
        :See Also:
            * :func:`read_csv_header`
            * :func:`read_csv_data`
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Check type
        if typeutils.isfile(fname):
            # Safe file name
            self.fname = fname.name
            # Already a file
            self._read_csv(fname)
        else:
            # Save file name
            self.fname = fname
            # Open file
            with open(fname, 'r') as f:
                # Process file handle
                self._read_csv(f)

    # Read CSV file from file handle
    def _read_csv(self, f):
        r"""Read a CSV file from current position
        
        :Call:
            >>> db._read_csv(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                File open for reading
        :See Also:
            * :func:`read_csv_header`
            * :func:`read_csv_data`
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Process column names
        self.read_csv_header(f)
        # Process column types
        self.finish_defns()
        # Loop through lines
        self.read_csv_data(f)

    # Reader: C only
    def c_read_csv(self, fname, **kw):
        r"""Read an entire CSV file, including header using C
        
        :Call:
            >>> db.read_csv(fname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *fname*: :class:`str`
                Name of file to read
        :See Also:
            * :func:`read_csv_header`
            * :func:`read_csv_data`
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Open file
        with open(fname, 'r') as f:
            # Process column names
            self.read_csv_header(f)
            # Process column types
            self.finish_defns()
            # Loop through lines
            self.c_read_csv_data(f)

    # Reader: Python only
    def py_read_csv(self, fname):
        r"""Read an entire CSV file with pure Python
        
        :Call:
            >>> db.py_read_csv(fname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *fname*: :class:`str`
                Name of file to read
        :See Also:
            * :func:`read_csv_header`
            * :func:`read_csv_data`
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Open file
        with open(fname, 'r') as f:
            # Process column names
            self.read_csv_header(f)
            # Process column types
            self.finish_defns()
            # Loop through lines
            self.py_read_csv_data(f)
   
   # --- Header ---
    # Read initial comments
    def read_csv_header(self, f):
        r"""Read column names from beginning of open file
        
        :Call:
            >>> db.read_csv_header(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2019-11-12 ``@ddalle``: First version
        """
        # Set header flags
        self._csv_header_once = False
        self._csv_header_complete = False
        # Read until header_complete flag set
        while not self._csv_header_complete:
            self.read_csv_headerline(f)
        # Remove flags
        del self._csv_header_once
        del self._csv_header_complete
        # Get default column names if necessary
        self.read_csv_headerdefaultcols(f)
        # Get guesses as to types
        self.read_csv_firstrowtypes(f)

    # Read a line as if it were a header
    def read_csv_headerline(self, f):
        r"""Read line and process column names if possible
        
        :Call:
            >>> db.read_csv_headerline(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: ``None`` | :class:`list`\ [:class:`str`]
                List of column names if read
            *db._csv_header_once*: ``True`` | ``False``
                Set to ``True`` if column names are read at all
            *db._csv_header_complete*: ``True`` | ``False``
                Set to ``True`` if next line is expected to be data
        :Versions:
            * 2019-11-22 ``@ddalle``: First version
        """
        # Check if header has already been processed
        if self._csv_header_complete:
            return
        # Save current position
        pos = f.tell()
        # Read line
        line = f.readline()
        # Check if it starts with a comment
        if line == "":
            # End of file
            self._csv_header_complete = True
            return
        elif line.startswith("#"):
            # Remove comment
            line = line.lstrip("#")
            # Check for empty comment
            if line.strip() == "":
                # Don't process and don't set any flags
                return
            # Strip comment char and split line into columns
            cols = [col.strip() for col in line.split(",")]
            # Marker that header has been read
            self._csv_header_once = True
        elif not self._csv_header_once:
            # Check for empty line
            if line.strip() == "":
                # Return without setting any flags
                return
            # Split line into columns without strip
            cols = [col.strip() for col in line.split(",")]
            # Marker that header has been read
            self._csv_header_once = True
            # Check valid names of each column
            for col in cols:
                # If it begins with a number, it's probably a data row
                if not regex_alpha.match(col):
                    # Marker for no header
                    self._csv_header_complete = True
                    # Return file to previous position
                    f.seek(pos)
                    # Exit
                    return
        else:
            # If row is empty; allow this
            if line.strip() == "":
                return
            # Non-comment row following comment: data
            f.seek(pos)
            # Mark completion of header
            self._csv_header_complete = True
            # Exit
            return
        # Save column names if reaching this point
        self.cols = self.translate_colnames(cols)
        # Output column names for kicks
        return cols
        
    # Read header types from first data row
    def read_csv_firstrowtypes(self, f):
        r"""Get initial guess at data types from first data row
        
        If (and only if) the *DefaultType* input is an integer type,
        guessed types can be integers.  Otherwise the sequence of
        possibilities is :class:`float`, :class:`complex`,
        :class:`str`.
        
        :Call:
            >>> db.read_csv_firstrowtypes(f, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                Open file handle
            *DefaultType*: {``"float"``} | :class:`str`
                Name of default class
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Get integer option
        odefcls = self.get_option("DefaultType", "float64")
        # Save position
        pos = f.tell()
        # Read line
        line = f.readline()
        # Check for empty data
        if line == "":
            return
        # Return to original location so first data row can be read
        f.seek(pos)
        # Otherwise, split into data
        coltxts = [txt.strip() for txt in line.split(",")]
        # Attempt to convert columns to ints, then floats
        for (j, col) in enumerate(self.cols):
            # Create definitions if necessary
            defn = self.get_defn(col)
            # Check if type already set
            if "Type" in defn:
                continue
            # Get text from *j*th column
            txtj = coltxts[j]
            # Cascade through possible conversions
            if odefcls.startswith("int"):
                try:
                    # Try an integer first
                    int(txtj)
                    # If it works; save it
                    defn["Type"] = odefcls
                    continue
                except ValueError:
                    pass
            # Try a float next
            try:
                # Substitutions for "2.4D+00"
                txtj = txtj.replace("D", "e")
                txtj = txtj.replace("d", "e")
                # Try conversion
                float(txtj)
                # If it works; save type
                if odefcls.startswith("float"):
                    # Use specific version
                    defn["Type"] = odefcls
                else:
                    # Use global default
                    defn["Type"] = "float64"
                continue
            except Exception:
                pass
            # Try a complex number first
            try:
                # Substitutions for "1+2i"
                txtj = txtj.replace("I", "j")
                txtj = txtj.replace("i", "j")
                # Try conversion
                complex(txtj)
                # If it works; save type
                defn["Type"] = "complex128"
            except Exception:
                # Only option left is a string
                defn["Type"] = "str"
        
    # Read first data line to count columns if necessary
    def read_csv_headerdefaultcols(self, f):
        r"""Create column names "col1", "col2", etc. if needed
        
        :Call:
            >>> db.read_csv_headerdefaultcols(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                If not previously determined, this becomes
                ``["col1", "col2", ...]`` based on number of columns
                in the first data row
        :Versions:
            * 2019-11-27 ``@ddalle``: First version
        """
        # Check if columns already determined
        if self.cols:
            return
        # Save position
        pos = f.tell()
        # Read line
        line = f.readline()
        # Check for empty data
        if line == "":
            return
        # Return to original location so first data row can be read
        f.seek(pos)
        # Otherwise, split into data
        coltxts = [txt.strip() for txt in line.split(",")]
        # Count the number of columns
        ncol = len(coltxts)
        # Create default column names
        cols = ["col%i" % (i+1) for i in range(ncol)]
        # Apply translations
        self.cols = self.translate_colnames(cols)

   # --- Data ---
    # Read data
    def read_csv_data(self, f):
        r"""Read data portion of CSV file
        
        :Call:
            >>> db.read_csv_data(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
            * 2019-11-29 ``@ddalle``: Tries C versionfirst
        """
        try:
            self.c_read_csv_data(f)
        except Exception:
            self.py_read_csv_data(f)

    # Read data: C implementation
    def c_read_csv_data(self, f):
        r"""Read data portion of CSV file using C extension
        
        :Call:
            >>> db.c_read_csv_data(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Test module
        if _ftypes is None:
            raise ImportError("No _ftypes extension module")
        # Get data types
        self.create_c_dtypes()
        # Call C function
        _ftypes.CSVFileReadData(self, f)
        # Get lengths
        self._n = {k: len(self[k]) for k in self.cols}
        # Save overall length
        self.n = self._n[self.cols[0]]
        # Delete _c_dtypes
        del self._c_dtypes

    # Read data: Python implementation
    def py_read_csv_data(self, f):
        r"""Read data portion of CSV file using Python
        
        :Call:
            >>> db.py_read_csv_data(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Initialize columns
        self.init_cols(self.cols)
        # Initialize types
        self._types = [self.get_col_type(col) for col in self.cols]
        # Set count
        self.n = 0
        # Read data lines
        while True:
            # Process next line
            eof = self.read_csv_dataline(f)
            # Check for end of file
            if eof == -1:
                break
            # Increase count
            self.n += 1
        # Delete types
        del self._types
        # Trim each column
        for col in self.cols:
            self.trim_colarray(col)

    # Read data line
    def read_csv_dataline(self, f):
        r"""Read one data line of a CSV file
        
        :Call:
            >>> db.read_csv_dataline(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Read line
        line = f.readline()
        # Check for end of file
        if line == "":
            return -1
        # Check for comment
        if line.startswith("#"):
            return
        # Check for empty line
        if line.strip() == "":
            return
        # Split line
        coltxts = [txt.strip() for txt in line.split(",")]
        # List of types
        _types = self._types
        # Loop through columns
        for (j, col) in enumerate(self.cols):
            # Get type
            clsname = _types[j]
            # Convert text
            v = self.fromtext_val(coltxts[j], clsname)
            # Save data
            self.append_colval(col, v)
    
   # --- C Interface ---
    # Get data types for C input
    def create_c_dtypes(self):
        r"""Initialize *db._c_dtypes* for C text input
        
        :Call:
            >>> db.create_c_dtypes()
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
        :Effects:
            *db._c_dtypes*: :class:`list`\ [:class:`int`]
                List of integer codes for each data type
        :Versions:
            * 2019-11-29 ``@ddalle``: First version
        """
        # Check if module is present
        if _ftypes is None:
            return
        # Create list
        dtypes = []
        # Handle to list of supported names
        DTYPE_NAMES = _ftypes.capeDTYPE_NAMES
        # Loop through columns
        for col in self.cols:
            # Get name of data type
            dtype = self.get_col_type(col)
            # Check if it's ported to C
            if dtype not in DTYPE_NAMES:
                raise NotImplementedError(
                    "Data type '%s' not ported to C" % dtype)
            # Convert to integer
            dtypes.append(DTYPE_NAMES.index(dtype))
        # Save the data types
        self._c_dtypes = dtypes
  # >
  
  # =============
  # Write
  # =============
  # <
   # --- Write Drivers ---
    # Write a CSV file
    def write_csv(self, fname, cols=None, fmt=None, **kw):
        """Write a comma-separated file of some of the coefficients

        :Call:
            >>> db.write_csv(fcsv, coeffs=None, fmt=None, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *fname*: :class:`str`
                Name of ASCII data file to write
            *cols*: {``None``} | :class:`list` (:class:`str`)
                List of coefficients to write, or write all coefficients
            *fmt*: {``None``} | :class:`str`
                Format string to be used for each row (optional)
            *fmts*: :class:`dict` | :class:`str`
                Dictionary of formats to use for each *coeff*
            *comments*: {``"#"``} | :class:`str`
                Comment character, used as first character of file
            *delim*: {``", "``} | :class:`str`
                Delimiter
            *translators*: {``{}``} | :class:`dict`
                Dictionary of coefficient translations, e.g. *CAF* -> *CA*
        :Versions:
            * 2018-06-11 ``@ddalle``: First versions
            * 2020-01-15 ``@jmeeroff``: Copied From cape.attdb.db.db1
        """
        # Process coefficient list
        if cols is None:
            cols = list(self.cols)
        # Check for presence
        for col in cols:
            if col not in self:
                raise KeyError("No output coefficient '%s'" % col)
        # Get the count of the first key
        n = len(self[cols[0]])
        # Loop through the keys
        for i in range(len(cols)-1, 0, -1):
            # Coefficient
            col = cols[i]
            # Check length
            if len(self[col]) != n:
                # Print a warning
                sys.stderr.write("WARNING: skipping ")
                sys.stderr.write("coefficient '%s' " % col)
                sys.stderr.write("with mismatching length\n")
                sys.stderr.flush()
                # Delete it
                del cols[i]

        # Dictionary of translators
        translators = kw.get("translators", {})
        # Get comment character and delimiter
        cchar = kw.get("comments", "#")
        delim = kw.get("delim", ", ")

        # Default line format
        if fmt is None:
            # Set up printing format
            fmts = kw.get("fmts", {})
            # Options for default print flag
            prec = kw.get("prec", kw.get("precision", 6))
            emax = kw.get("emax", 4)
            emin = kw.get("emin", -2)
            echr = kw.get("echar", "e")
            # Specific
            precs = kw.get("precs", kw.get("precisions", {}))
            emaxs = kw.get("emaxs", {})
            emins = kw.get("emins", {})
            # Initialize final format
            fmt_list = []
            # Loop through keys to create default format
            for col in cols:
                # Options
                kwf = {
                    "prec": precs.get(col, prec),
                    "emax": emaxs.get(col, emax),
                    "emin": emins.get(col, emin),
                    "echar": echr,
                }
                # Make a default *fmt* for this coefficient
                fmti = arrayutils.get_printf_fmt(self[col], **kwf)
                # Get format, using above default
                fmti = fmts.get(col, fmti)
                # Save to list
                fmt_list.append(fmti)
            # Just use the delimiter
            fmt = delim.join(fmt_list)
        # Apply translators to the headers
        cols = [translators.get(col, col) for col in cols]

        # Create the file
        f = open(fname, 'w')
        # Write header
        f.write("%s " % cchar)
        f.write(delim.join(cols))
        f.write("\n")
        # Loop through entries
        for i in range(n):
            # Get values of coefficients
            V = tuple(self[col][i] for col in cols)
            # Use the format string
            f.write(fmt % V)
            # Newline
            f.write("\n")

        # Close the file
        f.close()

    # Write raw
    def write_csv_dense(self, fname=None, cols=None):
        r"""Write dense CSV file using *WriteFlag* for each column
        
        :Call:
            >>> db.write_csv_dense(f, cols=None)
            >>> db.write_csv_dense(fname=None, cols=None)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                File open for writing
            *fname*: {*db.fname*} | :class:`str`
                Name of file to write
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of columns to write
        :Versions:
            * 2019-12-05 ``@ddalle``: First version
        """
        # Get file handle
        if fname is None:
            # Use *db.fname*
            with open(self.fname, "w") as f:
                self._write_csv_dense(f, cols=cols)
        elif typeutils.isstr(fname):
            # Open file based in specified name
            with open(fname, "w") as f:
                self._write_csv_dense(f, cols=cols)
        else:
            # Already a file (maybe)
            self._write_csv_dense(fname, cols=cols)

    # Write raw CSV file given file handle
    def _write_csv_dense(self, f, cols=None):
        r"""Write dense CSV file using *WriteFlag* for each column
        
        :Call:
            >>> db._write_csv_dense(f, cols=None)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                File open for writing
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of columns to write
        :Versions:
            * 2019-12-05 ``@ddalle``: First version
        """
        # Default column list
        if cols is None:
            cols = self.cols
        # Number of columns
        ncol = len(cols)
        # Format flags
        wflags = [self.get_col_prop(col, "WriteFormat", "%s") for col in cols]
        # Get characters
        comnt = self.opts.get("Comment", "#")
        delim = self.opts.get("Delimiter", ",")
        # Strip delimiter if not whitespace
        if delim.strip():
            delim = delim.strip()
        # Write variable list
        f.write(comnt)
        f.write(" ")
        f.write(delim.join(cols))
        f.write("\n")
        # Loop through database rows
        for i in range(self.n):
            # Loop through columns
            for (j, col) in enumerate(cols):
                # Get value
                v = self[col][i]
                # Write according to appropriate flag
                f.write(wflags[j] % v)
                # Check for last column
                if (j + 1 == ncol):
                    # End of line
                    f.write("\n")
                else:
                    # Delimiter
                    f.write(delim)

   # --- Component Writers ---
    # Get write flag
    def get_autoformat(self, col, **kw):
        # Get type
        dtype = self.get_col_type(col)
        # Get current write flag
        fmt = self.get_col_prop(col, "CSVFormat")
        # Return it if explicit
        if fmt:
            return fmt
        # Check type
        if dtype.startswith("int") or dtype.startswith("float"):
            # Get the values
            V = self[col]
            # Process from inputs
            return arrayutils.get_printf_fmt(V, **kw)
        elif dtype == "str":
            # Get the values
            V = self[col]
            # Get maximum length
            nmax = max([len(v) for v in V])
            # Pad write string
            return "%%-%is" % nmax
        else:
            # Assume string
            return "%s"
        
  # >
# class CSVFile


# Simple CSV file
class CSVSimple(BaseFile):
    r"""Class to read CSV file with only :class:`float` data
    
    This class differs from :class:`CSVFile` in that it is less
    flexible, does not permit multirow or empty headers, has fixed
    delimiter and comment characters, and assumes all data is a
    :class:`float` with the system default length.
    
    :Call:
        >>> db = CSVSimple(fname, **kw)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *db*: :class:`cape.attdb.ftypes.csvfile.CSVSimple`
            CSV file interface
        *db.cols*: :class:`list`\ [:class:`str`]
            List of columns read
        *db.opts*: :class:`dict`
            Options for this instance
        *db.opts["Definitions"]*: :class:`dict`
            Definitions for each column
        *db[col]*: :class:`np.ndarray` | :class:`list`
            Numeric array or list of strings for each column
    :See also:
        * :class:`cape.attdb.ftypes.basefile.BaseFile`
    :Versions:
        * 2019-11-26 ``@ddalle``: Started
    """
  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, fname=None, **kw):
        """Initialization method
        
        :Versions:
            * 2019-11-12 ``@ddalle``: First version
        """
        # Initialize common attributes
        self.cols = []
        self.n = 0
        self.fname = None

        # Process keyword arguments
        self.opts = self.process_kw(**kw)

        # Explicit definition declarations
        self.get_defns()

        # Read file if appropriate
        if fname:
            # Read valid file
            self.read_csvsimple(fname)
        else:
            # Apply defaults to definitions
            self.finish_defns()

        # Check for overrides of values
        self.process_kw_values()
  # >

  # =============
  # Read
  # =============
  # <
   # --- Control ---
    # Reader
    def read_csvsimple(self, fname):
        r"""Read an entire CSV file, including header
        
        The CSV file requires exactly one header row, which is the
        first non-empty line, whether or not it begins with a comment
        character (which must be ``"#"``).  All entries, both in the
        header and in the data, must be separated by a ``,``.
        
        :Call:
            >>> db.read_csvsimple(fname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVSimple`
                CSV file interface
            *fname*: :class:`str`
                Name of file to read
        :See Also:
            * :func:`read_csvsimple_header`
            * :func:`read_csvsimple_data`
        :Versions:
            * 2019-11-27 ``@ddalle``: First version
        """
        # Open file
        with open(fname, 'r') as f:
            # Process column names
            self.read_csvsimple_header(f)
            # Initialize columns
            self.init_cols(self.cols)
            # Loop through lines
            self.read_csvsimple_data(f)
   
   # --- Header ---
    # Read initial comments
    def read_csvsimple_header(self, f):
        r"""Read column names from beginning of open file
        
        :Call:
            >>> db.read_csv_header(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVSimple`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2019-11-12 ``@ddalle``: First version
        """
        # Loop until a nonempty line is read
        while True:
            # Read the next line
            line = f.readline()
            # Check contents of line
            if line == "":
                raise ValueError("File '%s' has no header" % self.fname)
            # Strip comment and white space
            line = line.lstrip("#").strip()
            # Check for empty line
            if line == "":
                continue
            # Process header line, strip white space from each col
            cols = [col.strip() for col in line.split(",")]
            # Apply translations
            self.cols = self.translate_colnames(cols)
            # Once this is done, task completed
            return

   # --- Data ---
    # Rad data
    def read_csvsimple_data(self, f):
        r"""Read data portion of simple CSV file
        
        :Call:
            >>> db.read_csvsimple_data(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVSimple`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Set data count
        self.n = 0
        # Read data lines
        while True:
            # Process next line
            eof = self.read_csvsimple_dataline(f)
            # Check for end of file
            if eof == -1:
                break
            # Increase count
            self.n += 1
        # Trim each column
        for col in self.cols:
            self.trim_colarray(col)

    # Read data line
    def read_csvsimple_dataline(self, f):
        r"""Read one data line of a simple CSV file
        
        :Call:
            >>> db.read_csvsimple_dataline(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVSimple`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Versions:
            * 2019-11-27 ``@ddalle``: First version
        """
        # Read line
        line = f.readline()
        # Check for end of file
        if line == "":
            return -1
        # Check for comment
        if line.startswith("#"):
            return
        # Check for empty line
        if line.strip() == "":
            return
        # Split line
        coltxts = [txt.strip() for txt in line.split(",")]
        # Loop through columns
        for (j, col) in enumerate(self.cols):
            # Convert text
            v = self.translate_simplefloat(coltxts[j])
            # Save data
            self.append_colval(col, v)

    # Convert text to float
    def translate_simplefloat(self, txt):
        r"""Convert a string to default float
        
        This conversion allows for the format ``"2.40D+00"`` if the
        built-in :func:`float` converter fails.  Python expects the
        exponent character to be ``E`` or ``e``, but ``D`` and ``d``
        are allowed here.  Other exceptions are not handled.
        
        :Call:
            >>> v = db.translate_simplefloat(txt)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                CSV file interface
            *txt*: :class:`str`
                Text to be converted to :class:`float`
        :Outputs:
            *v*: :class:`float`
                Converted value, if possible
        :Versions:
            * 2019-11-27 ``@ddalle``: First version
        """
        # Attempt conversion
        try:
            # Basic conversion
            return float(txt)
        except ValueError:
            # Substitute "E" for "D" and "e" for "d"
            txt = txt.replace("D", "E")
            txt = txt.replace("d", "e")
        # Second attempt
        return float(txt)
  # >
# class CSVSimple
