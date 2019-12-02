#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.ftypes.textdata`: Generic textual data interface
=================================================================

This module contains a basic interface in the spirit of
:mod:`cape.attdb.ftypes` for standard text data files. It creates a
class, :class:`TextDataFile` that does not rely on the popular
:func:`numpy.loadtxt` function and supports a more capabilities than
the :mod:`cape.attdb.ftypes.csv.CSVFile` class.

For example, the :class:`TextDataFile` class supports a variety of
delimiters, whereas a :class:`CSVFile` instance must use ``','`` as the
delimiter.  The :class:`TextDataFile` class also remembers its text

If possible, the column names (which become keys in the
:class:`dict`-like class) are read from the header row.  If the file
begins with multiple comment lines, the column names are read from the
final comment before the beginning of data.
"""

# Standard library
import re

# Third-party modules
import numpy as np

# CAPE modules
import cape.tnakit.typeutils as typeutils

# Local modules
from .basefile import BaseFile, TextInterpreter

# Local extension
try:
    from . import _ftypes
except ImportError:
    _ftypes = None


# Class for generic text data
class TextDataFile(BaseFile, TextInterpreter):
    r"""Interface to generic data text files
    
    :Call:
        >>> db = TextDataFile(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *delim*, *Delimiter*: {``", "``} | :class:`str`
            Delimiter(s) option
    :Outputs:
        *db*: :class:`cape.attdb.ftypes.textdata.TextDatafile`
            Text data file interface
        *db.cols*: :class:`list`\ [:class:`str`]
            List of columns read
        *db.lines*: :class:`list`\ [:class:`str`]
            Lines of text from the file that was read
        *db.opts*: :class:`dict`
            Options for this instance
        *db.opts["Delimiter"]: {``" ,"``} | :class:`str`
            Delimiter(s) to allow
        *db.opts["Definitions"]*: :class:`dict`
            Definitions for each column
        *db[col]*: :class:`np.ndarray` | :class:`list`
            Numeric array or list of strings for each column
    :Versions:
        * 2019-12-02 ``@ddalle``: First version
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
        # Initialize options
        self.opts = {}

        # Save file name
        self.fname = fname
        
        # Initialize text
        self.lines = []

        # Process generic options
        kw = self.process_opts(**kw)

        # Read file if appropriate
        if fname and typeutils.isstr(fname):
            # Read valid file
            kw = self.read_textdata(fname, **kw)
        else:
            # Process inputs
            kw = self.process_col_defns(**kw)

        # Check for overrides of values
        kw = self.process_values(**kw)
        # Warn about any unused inputs
        self.warn_kwargs(kw)
  # >
  
  # ===========
  # Options
  # ===========
  # <
    # Process more options
    def process_opts(self, **kw):
        r"""Process all options for data text file instances
        
        :Call:
            >>> kwo = db.process_opts(**kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *cols*, *ColNames*, *Keys*: :class:`list`\ [:class:`str`]
                User-specified column names
            *delim*, *Delimiter*: {``", "``} | :class:`str`
                Delimiter(s) option
            *comment*, *Comment*: {``"#"``} | :class:`str`
                Character(s) used to demark a line comment
            *[Tt]ranslators*: :class:`dict`\ [:class:`str`]
                Dictionary of alternate names to store column names as;
                for example if the header has a column called
                ``"CAF"``, and ``translators["CAF"]`` is ``"CA"``, that
                column will be stored as ``db["CA"]`` instead of
                ``db["CAF"]``
        :Outputs:
            *kwo*: :class:`dict`
                Options not used in this method
        :See also:
            * :func:`BaseFile.process_opts_generic`
        :Versions:
            * 2019-11-27 ``@ddalle``: First version
        """
        # Generic options
        kw = self.process_opts_generic(**kw)
        # Check for local options
        delim = kw.pop("delim", kw.pop("Delimiter", ", "))
        # Comment character
        comment = kw.pop("comment", kw.pop("Comment", "#"))
        # Check type
        if not typeutils.isstr(delim):
            raise TypeError("Delimiter must be a string")
        if not typeutils.isstr(comment):
            raise TypeError("Comment character(s) must be a string")
        # Save the delimiter
        self.opts["Delimiter"] = delim
        # Save the comment character
        self.opts["Comment"] = comment
        # Return remaining options
        return kw
  # >
  
  # ============
  # Read
  # ============
  # <
   # --- Readers ---
    # Reader: Python only
    def read_textdata(self, fname, **kw):
        r"""Read an entire text data file
        
        :Call:
            >>> db.read_textdata(fname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
            *fname*: :class:`str`
                Name of file to read
        :See Also:
            * :func:`read_textdata_header`
            * :func:`read_textdata_data`
        :Versions:
            * 2019-12-02 ``@ddalle``: First version
        """
        # Initialize line number
        self._nline = 0
        # Reinitialize lines
        self.lines = []
        self.linenos = []
        # Process line splitting regular expression
        self.get_regex_linesplitter()
        # Open file
        with open(fname, 'r') as f:
            # Process column names
            self.read_textdata_header(f, **kw)
            # Process column types
            kw = self.process_col_defns(**kw)
            # Loop through lines
            self.read_textdata_data(f)
        # Cleanup
        del self._nline
        del self._delim
        del self._comment
        # Output remaining options
        return kw
   
   # --- Header ---
    # Read initial comments
    def read_textdata_header(self, f, **kw):
        r"""Read column names from beginning of open file
        
        :Call:
            >>> db.read_textdata_header(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2019-11-12 ``@ddalle``: First version
        """
        # Save special characters
        self._delim = self.opts.get("Delimiter", ", ")
        self._comment = self.opts.get("Comment", "#")
        # Set header flags
        self._textdata_header_once = False
        self._textdata_header_complete = False
        # Read until header_complete flag set
        while not self._textdata_header_complete:
            self.read_textdata_headerline(f)
        # Remove flags
        del self._textdata_header_once
        del self._textdata_header_complete
        # Get default column names if necessary
        self.read_textdata_headerdefaultcols(f)
        # Get guesses as to types
        self.read_textdata_firstrowtypes(f, **kw)

    # Read a line as if it were a header
    def read_textdata_headerline(self, f):
        r"""Read line and process column names if possible
        
        :Call:
            >>> db.read_textdata_headerline(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: ``None`` | :class:`list`\ [:class:`str`]
                List of column names if read
            *db._textdata_header_once*: ``True`` | ``False``
                Set to ``True`` if column names are read at all
            *db._textdata_header_complete*: ``True`` | ``False``
                Set to ``True`` if next line is expected to be data
        :Versions:
            * 2019-11-22 ``@ddalle``: First version
            * 2019-12-02 ``@ddalle``: Copied from :class:`CSVFile`
        """
        # Check if header has already been processed
        if self._textdata_header_complete:
            return
        # Save comment character and delimiter
        _comment = self._comment
        _delim   = self._delim
        # Save current position
        pos = f.tell()
        # Read line
        line = f.readline()
        # Check if it starts with a comment
        if line == "":
            # End of file
            self._textdata_header_complete = True
            return
        elif line.startswith(_comment):
            # Save the line
            self.lines.append(line)
            # Remove comment
            line = line.lstrip(_comment)
            # Check for empty comment
            if line.strip() == "":
                # Don't process and don't set any flags
                return
            # Strip comment char and split line into columns
            cols = self.split_textdata_line(line)
            # Marker that header has been read
            self._textdata_header_once = True
        elif not self._textdata_header_once:
            # Check for empty line
            if line.strip() == "":
                # Return without setting any flags
                return
            # Split line into columns without strip
            cols = self.split_textdata_line(line)
            # Marker that header has been read
            self._textdata_header_once = True
            # Check valid names of each column
            for col in cols:
                # If it begins with a number, it's probably a data row
                if not regex_alpha.match(col):
                    # Marker for no header
                    self._textdata_header_complete = True
                    # Return file to previous position
                    f.seek(pos)
                    # Exit
                    return
            # Save the line
            self.lines.append(line)
        else:
            # If row is empty; allow this
            if line.strip() == "":
                self.lines.append(line)
                return
            # Non-comment row following comment: data
            f.seek(pos)
            # Mark completion of header
            self._textdata_header_complete = True
            # Exit
            return
        # Save column names if reaching this point
        self.cols = cols
        # Output column names for kicks
        return cols
        
    # Read header types from first data row
    def read_textdata_firstrowtypes(self, f, **kw):
        r"""Get initial guess at data types from first data row
        
        If (and only if) the *DefaultType* input is an integer type,
        guessed types can be integers.  Otherwise the sequence of
        possibilities is :class:`float`, :class:`complex`,
        :class:`str`.
        
        :Call:
            >>> db.read_textdata_firstrowtypes(f, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
            *f*: :class:`file`
                Open file handle
            *DefaultType*: {``"float"``} | :class:`str`
                Name of default class
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
            * 2019-12-02 ``@ddalle``: Copied from :class:`CSVFile`
        """
        # Get integer option
        odefcls = kw.get("DefaultType", "float64")
        # Translate abbreviated codes
        odefcls = self.validate_dtype(odefcls)
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
        coltxts = self.split_textdata_line(line)
        # Initialize types
        defns = self.opts.setdefault("Definitions", {})
        # Attempt to convert columns to ints, then floats
        for (j, col) in enumerate(self.cols):
            # Create definitions if necessary
            defn = defns.setdefault(col, {})
            # Get text from *j*th column
            txtj = coltxts[j]
            # Cascade through possible conversions
            if odefcls.startswith("int"):
                try:
                    # Try an integer first
                    int(txtj)
                    # If it works; save it
                    defn.setdefault("Type", odefcls)
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
                    defn.setdefault("Type", odefcls)
                else:
                    # Use global default
                    defn.setdefault("Type", "float64")
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
                defn.setdefault("Type", "complex128")
            except Exception:
                # Only option left is a string
                defn.setdefault("Type", "str")
        
    # Read first data line to count columns if necessary
    def read_textdata_headerdefaultcols(self, f):
        r"""Create column names "col1", "col2", etc. if needed
        
        :Call:
            >>> db.read_textdata_headerdefaultcols(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                If not previously determined, this becomes
                ``["col1", "col2", ...]`` based on number of columns
                in the first data row
        :Versions:
            * 2019-11-27 ``@ddalle``: First version
            * 2019-12-02 ``@ddalle``: Copied from :class:`CSVFile`
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
        coltxts = self.split_textdata_line(line)
        # Count the number of columns
        ncol = len(coltxts)
        # Create default column names
        self.cols = ["col%i" % (i+1) for i in range(ncol)]
   
   # --- Data ---# Read data: Python implementation
    def read_textdata_data(self, f):
        r"""Read data portion of text data file
        
        :Call:
            >>> db.read_textdata_data(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
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
        # Read data lines
        while True:
            # Process next line
            eof = self.read_textdata_line(f)
            # Check for end of file
            if eof == -1:
                break
        # Delete types
        del self._types
        # Trim each column
        for col in self.cols:
            self.trim_colarray(col)

    # Read the next line
    def read_textdata_line(self, f):
        r"""Read a data row from a text data file
        
        :Call:
            >>> db.read_textdata_line(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
            *f*: :class:`file`
                Open file handle
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Read the next line
        line = f.readline()
        # Check for end of file
        if line == "":
            return -1
        # Line counter
        self._nline += 1
        # Save the line
        self.lines.append(line)
        # Check if line is a comment
        if line.startswith(self._comment):
            # Comment
            return
        elif line.strip() == "":
            # Empty line
            return
        # Save the line number for this data row
        self.linenos.append(self._nline)
        # Split line
        coltxts = self.split_textdata_line(line)
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

    # Split a data line into values
    def split_textdata_line(self, line):
        r"""Split a line into its parts
        
        Splits line of text by specified delimiter and strips
        whitespace and delimiter from each entry
        
        :Call:
            >>> parts = db.split_textdata_line(line)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
            *line*: :class:`str`
                Line of text to be split
        :Outputs:
            *parts*: :class:`list`\ [:class:`str`]
                List of strings
        :Versions:
            * 2019-12-02 ``@ddalle``: First version
        """
        # Split line
        coltxts = self.regex_linesplit.findall(line)
        # Strip white space and delimiters
        parts = [txt.strip(self._delim) for txt in coltxts]
        # Output
        return parts

   # --- Line Splitter ---
    # Get the regular expression for splitting a line into parts
    def get_regex_linesplitter(self):
        r"""Generate regular expression used to split a line
        
        :Call:
            >>> db.get_regex_linesplitter()
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
        :Effects:
            *db.regex_linesplit*: :class:`re.SRE_Pattern`
                Compiled regular expression object
        :Versions:
            * 2019-12-02 ``@ddalle``: First version
        """
        # Get the delimiter
        delim = self.opts.get("Delimiter", ", ")
        # Save it
        self._delim = delim
        # Check if white space is allowed
        if " " in delim:
            # Remove the space
            delim = delim.replace(" ", "")
            # Two-part regular expression
            regex = r"\s*[^\s%(delim)s]*\s*[%(delim)s]|\s*[^\s%(delim)s]+"
        else:
            # If not using white space, require a delimiter
            regex = r"\s*[^\s%(delim)s]*\s*[%(delim)s]"
        # Make substitutions
        regex = regex % {"delim": delim}
        # Compile
        self.regex_linesplit = re.compile(regex)
        
        
        
  # >
    
        
