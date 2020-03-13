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

# CAPE modules
import cape.tnakit.typeutils as typeutils

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
class TextDataOpts(BaseFileOpts):
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Global Options ---
    # List of options
    _optlist = {
        "Delimeter",
        "Comment",
        "FirstColBoolMap",
        "FirstColName"
    }

    # Alternate names
    _optmap = {
        "comments": "Comment",
        "delim": "Delimiter",
        "delimeter": "Delimiter",
    }

   # --- Types ---
    # Types allowed
    _opttypes = {
        "Comment": typeutils.strlike,
        "Delimiter": typeutils.strlike,
        "FirstColBoolMap": (bool, dict),
        "FirstColName": typeutils.strlike,
    }

   # --- Defaults ---
    _rc = {
        "Comment": "#",
        "Delimeter": ",",
        "FirstColBoolMap": False,
        "FirstColName": "_col1",
    }
  # >


# Definition
class TextDataDefn(BaseFileDefn):
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Global Options ---
    # List of options
    _optlist = {
        "Abbreviations",
        "Keys",
        "Map"
    }

    # Types
    _opttypes = {
        "Abbreviations": (set, list),
        "Keys": (set, list),
        "Map": (bool, dict),
    }

   # --- Values ---
    # Allowed values
    _optvals = {
        "Type": {
            "boolmap"
        }
    }

   # --- DType ---
    # Map of data types based on *Type*
    _dtypemap = {
        "boolmap": "str",
    }
  # >


# Combine options with parent class
TextDataOpts.combine_optdefs()
TextDataDefn.combine_optdefs()
TextDataDefn.combine_optdict("_dtypemap")

# Add definition support to option
TextDataOpts.set_defncls(TextDataDefn)


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
        *db.opts*: :class:`TextdataOpts`
            Options for this instance
        *db.defns*: :class:`dict`\ [:class:`TextDataDefn`
            Definitions for each column
        *db[col]*: :class:`np.ndarray` | :class:`list`
            Numeric array or list of strings for each column
    :Versions:
        * 2019-12-02 ``@ddalle``: First version
    """
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Options ---
    # Class for options
    _optscls = TextDataOpts
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
        self.lines = []

        # Process keyword arguments
        self.opts = self.process_kw(**kw)

        # Explicit definition declarations
        self.get_defns()

        # Read file if appropriate
        if fname and typeutils.isstr(fname):
            # Read valid file
            self.read_textdata(fname)
        else:
            # Process inputs
            self.finish_defns()

        # Check for overrides of values
        self.process_kw_values()
  # >
  
  # ===========
  # Options
  # ===========
  # <
   # --- Main ---
    # Process key definitions
    def finish_defns(self):
        r"""Process *Definitions* of column types
        
        :Call:
            >>> db.finish_defns(**kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Data file interface
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
            * 2014-06-17 ``@ddalle``: Read from *defns* :class:`dict`
            * 2019-11-12 ``@ddalle``: Forked from :class:`RunMatrix`
            * 2020-02-06 ``@ddalle``: Using *self.opts*
        """
        # Check for first-column boolean map
        col1bmap = self.opts.get_option("FirstColBoolMap", False)
        # Validate it if not False-like
        if col1bmap:
            # Name
            col0 = self.opts.get_option("FirstColName", "_col1")
            # Add to column lists
            if self.cols[0] != col0:
                # List of coefficients in data set
                self.cols.insert(0, col0)
                # List of columns printed to document
                self.textcols.insert(0, col0)
            # Process option
            self.process_defns_boolmap(col0, col1bmap)

        # Call parent method
        BaseFile.finish_defns(self)

    # Process boolean map definitions
    def process_defns_boolmap(self, col, bmap):
        r"""Process definitions for columns of type *BoolMap*

        :Call:
            >>> db.process_defns_boolmap(col, bmap)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Data file interface
            *col*: :class:`str`
                Name of column with type ``"BollMap"``
            *bmap*: :class:`dict`
                Map for abbreviations that set boolean columns
        :See Also:
            * :func:`validate_boolmap`
        :Versions:
            * 2019-12-03 ``@ddalle``: First version
        """
        # Validate
        boolmap = self.validate_boolmap(bmap)
        # Save definition for this key
        defn = self.get_defn(col)
        # Set properties
        defn["Map"] = boolmap
        defn["Type"] = "boolmap"
        # Possible values
        keys = []
        vals = []
        # Get list of boolean columns
        for (k, abbrevs) in boolmap.items():
            # Create a definition
            defnc = self.get_defn(k)
            # Set type (forced)
            defnc["Type"] = "bool"
            # Save to list of children
            keys.append(k)
            # Save column
            self.cols.append(k)
            # Append possible values
            for v in abbrevs:
                # Check if already present
                if v in vals:
                    raise ValueError(
                        ("Abbreviation '%s' used in previous key" % v) +
                        ("of column '%s' BoolMap" % col))
                # Save
                vals.append(v)
        # Append empty string to abbreviations
        if '' not in vals:
            vals.insert(0, '')
        # Save keys and values
        defn["Keys"] = keys
        defn["Abbreviations"] = vals

   # --- Keyword Checkers ---
    # Validate boolean flag columns
    def validate_boolmap(self, boolmap):
        r"""Translate free-form *Type* option into validated code
        
        :Call:
            >>> bmap = db.validate_boolmap(boolmap)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextData`
                Data file interface
            *boolmap*: :class:`str`\ [:class:`str` | :class:`list`]
                Initial boolean flag map; the keys are names of the
                boolean coefficients that are set, and the item values
                are the one or more abbreviations for each key
        :Outputs:
            *bmap*: :class:`str`\ [:class:`list`\ [:class:`str`]]
                Validated map
        :Versions:
            * 2019-12-03 ``@ddalle``: First version
        """
        # Check type
        if not isinstance(boolmap, dict):
            raise TypeError("'BoolMap' parameter must be a dict")
        # Create new list to ensure list types
        for (k, v) in boolmap.items():
            # Check if it's scalar
            if isinstance(v, list):
                # Already a list
                V = v
            elif isinstance(v, (tuple, set)):
                # Convert other array to list
                V = list(v)
            else:
                # Convert scalar to singleton
                V = [v]
            # Check entry types
            for vi in V:
                if not typeutils.isstr(vi):
                    raise TypeError(
                        "All map entries for '%s' must be strings" % k)
            # Save
            boolmap[k] = V
        # Output
        return boolmap
  # >
  
  # ============
  # Read
  # ============
  # <
   # --- Readers ---
    # Reader: Python only
    def read_textdata(self, fname):
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
        self.set_regex_linesplitter()
        # Open file
        with open(fname, 'r') as f:
            # Process column names
            self.read_textdata_header(f)
            # Process column types
            self.finish_defns()
            # Loop through lines
            self.read_textdata_data(f)
        # Cleanup
        del self._nline
        del self._delim
        del self._comment
   
   # --- Header ---
    # Read initial comments
    def read_textdata_header(self, f):
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
        self.read_textdata_firstrowtypes(f)
        # Save text columns
        self.textcols = list(self.cols)

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
        self.cols = self.translate_colnames(cols)
        # Output column names for kicks
        return cols
        
    # Read header types from first data row
    def read_textdata_firstrowtypes(self, f):
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
        odefcls = self.opts.get_option("DefaultType", "float64")
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
        # Attempt to convert columns to ints, then floats
        for (j, col) in enumerate(self.cols):
            # Create definitions if necessary
            defn = self.get_defn(col)
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
        cols = ["col%i" % (i+1) for i in range(ncol)]
        # Translate column names
        self.cols = self.translate_colnames(cols)
   
   # --- Data ---
    # Read data: Python implementation
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
        # Set data count
        self.n = 0
        # Read data lines
        while True:
            # Process next line
            eof = self.read_textdata_line(f)
            # Check for end of file
            if eof == -1:
                break
            # Increment count
            self.n += 1
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
        for (j, col) in enumerate(self.textcols):
            # Get type
            clsname = _types[j]
            # Convert text
            v = self.fromtext_val(coltxts[j], clsname, col)
            # Save data
            if isinstance(v, tuple):
                # Got text and a map
                self.append_colval(col, v[0])
                # Loop through map
                for (vk, vv) in v[1].items():
                    self.append_colval(vk, vv)
            else:
                # Save value directly
                self.append_colval(col, v)
        
   # --- Text Interpretation ---
    # Convert to text to appropriate class
    def fromtext_val(self, txt, clsname, col=None):
        r"""Convert a string to appropriate type
        
        :Call:
            >>> v = db.fromtext_val(txt, clsname, col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
            *txt*: :class:`str`
                Text to be converted to :class:`float`
            *clsname*: {``"float64"``} | ``"int32"`` | :class:`str`
                Valid data type name
            *col*: :class:`str`
                Name of flag column, for ``"boolmap"`` keys
        :Outputs:
            *v*: :class:`clsname`
                Text translated to requested type
        :Versions:
            * 2019-12-02 ``@ddalle``: First version
        """
        # Check type
        if clsname == "boolmap":
            # Convert flag to value
            return self.fromtext_boolmap(txt, col)
        else:
            # Fall back to main categories
            return self.fromtext_base(txt, clsname)

    # Convert a flag
    def fromtext_boolmap(self, txt, col):
        r"""Convert boolean flag text to dictionary
        
        :Call:
            >>> v, vmap = db.fromtext_boolmap(txt, col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
            *txt*: :class:`str`
                Text to be converted to :class:`float`
            *clsname*: {``"float64"``} | ``"int32"`` | :class:`str`
                Valid data type name
            *col*: :class:`str`
                Name of flag column, for ``"boolmap"`` keys
        :Outputs:
            *txt*: :class:`str`
                Text returned
            *vmap*: :class:`dict`\ [``True`` | ``False``]
                Flags for each flag in *col* definition
        :Versions:
            * 2019-12-02 ``@ddalle``: First version
        """
        # Get definition for column
        boolmap = self.get_col_prop(col, "Map", {})
        # Initialize map values
        vmap = {}
        # Check the text
        for (colname, vals) in boolmap.items():
            # Check text vs flags values
            if txt in vals:
                vmap[colname] = True
            else:
                vmap[colname] = False
        # Output the text and the map
        return txt, vmap

   # --- Line Splitter ---
    # Get the regular expression for splitting a line into parts
    def set_regex_linesplitter(self):
        r"""Generate regular expression used to split a line
        
        :Call:
            >>> db.set_regex_linesplitter()
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
        # Get types
        _types = getattr(self, "_types", None)
        # Check for declared types
        if _types is None:
            return parts
        # Loop through columns
        for (j, col) in enumerate(self.cols):
            # Check for columns with optional values
            if _types[j] == "boolmap":
                # Get text
                txtj = parts[j]
                # Get abbreviations
                abbrevs = self.get_col_prop(col, "Abbreviations", [""])
                # Check for a value
                if txtj not in abbrevs:
                    # Insert empty space to list
                    parts.insert(j, "")
        # Output
        return parts
        
  # >

  # =============
  # Write
  # =============
  # <
   # --- Writers ---
    # Write lines
    def write_textdata(self, fname=None):
        r"""Write text data file based on existing *db.lines*
        
        Checks are not performed that values in e.g. *db[col]* have
        been synchronized with the text in *db.lines*.  It is
        therefore possible to write a file that does not match the
        values in the database.  To avoid this, use :func:`set_colval`.
        
        :Call:
            >>> db.write_textdata()
            >>> db.write_textdata(fname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Text data file interface
            *fname*: {*db.fname*} | :class:`str`
                Name of file to write
        :Versions:
            * 2019-12-04 ``@ddalle``: First version
        """
        # Default file name
        if fname is None:
            fname = self.fname
        # Open the filw for writing
        with open(fname, 'w') as f:
            # Loop through lines
            for line in self.lines:
                # Write it
                f.write(line)
  # >
# class TextDataFile
