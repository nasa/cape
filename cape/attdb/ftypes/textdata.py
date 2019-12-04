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
        *db.opts["Delimiter"]*: {``" ,"``} | :class:`str`
            Delimiter(s) to allow
        *db.opts["Definitions"]*: :class:`dict`
            Definitions for each column
        *db[col]*: :class:`np.ndarray` | :class:`list`
            Numeric array or list of strings for each column
    :Versions:
        * 2019-12-02 ``@ddalle``: First version
    """
    # Class attributes
    _classtypes = ["boolmap"]

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
        self.cols = []
        self.n = 0

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
        kw = self.process_kw_values(**kw)
        # Warn about any unused inputs
        self.warn_kwargs(kw)
  # >
  
  # ===========
  # Options
  # ===========
  # <
   # --- Main ---
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
    
    # Process key definitions
    def process_col_defns(self, **kw):
        r"""Process *Definitions* of column types
        
        :Call:
            >>> kwo = db.process_col_defns(**kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Data file interface
            *FirstColBoolMap*: {``False``} | :class:`dict`
                Optional map for abbreviations that set boolean columns
            *FirstColName*: {``"_col1"``} | :class:`str`
                Name for special added first column
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
            * 2019-11-12 ``@ddalle``: Forked from :class:`RunMatrix`
        """
        # No special first column by default
        self.opts["OptionalFirstCol"] = False

        # Check for first-column boolean map
        col1bmap = kw.pop("FirstColBoolMap", False)
        # Validate it if not False-like
        if col1bmap:
            # Name
            col0 = kw.pop("FirstColName", "_col1")
            # Add to column lists
            if self.cols[0] != col0:
                # List of coefficients in data set
                self.cols.insert(0, col0)
                # List of columns printed to document
                self.textcols.insert(0, col0)
            # Process option
            self.process_defns_boolmap(col0, col1bmap)
            # Save option for special first column
            self.opts["OptionalFirstCol"] = True

        # Call parent method
        kw = BaseFile.process_col_defns(self, **kw)
        # Output
        return kw

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
        # Get definitions
        defns = self.opts.setdefault("Definitions", {})
        # Save definition for this key
        defn = defns.setdefault(col, {})
        # Set properties
        defn["Map"] = boolmap
        defn["Type"] = "boolmap"
        # Possible values
        keys = []
        vals = []
        # Get list of boolean columns
        for (k, abbrevs) in boolmap.items():
            # Create a definition
            defnc = defns.setdefault(k, {})
            # Set type
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
                        ("of BoolMap column '%s'" % col))
                # Save
                vals.append(v)
        # Append empty string to abbreviations
        if '' not in vals:
            vals.insert(0, '')
        # Save keys and values
        defn["Keys"] = keys
        defn["Abbreviations"] = vals

   # --- Keyword Checkers ---
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
            # Local type validator
            return self.validate_type(val)
        elif prop == "BoolMap":
            # Validate dictionary of boolean maps
            return self.validate_boolmap(val)
        else:
            # Default is to accept any input
            return val

    # Validate dtype
    def validate_type(self, clsname):
        r"""Translate free-form *Type* option into validated code
        
        :Call:
            >>> dtype = db.validate_dtype(clsname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *clsname*: :class:`str`
                Free-form *Type* option for a column
        :Outputs:
            *dtype*: ``"f64"`` | ``"i32"`` | ``"str"`` | :class:`str`
                Name of data type
        :Versions:
            * 2019-11-24 ``@ddalle``: First version
            * 2019-12-02 ``@ddalle``: Forked from :class:`BaseFile`
        """
        # Check type
        if not typeutils.isstr(clsname):
            raise TypeError("'Type' parameter must be a string")
        # Force lower case
        clsname = clsname.lower()
        # Filter
        if clsname in ["boolmap"]:
            # Valid
            return "boolmap"
        else:
            # Fallback
            return self.validate_dtype(clsname)
            
    # Validate boolean flag columns
    def validate_boolmap(self, boolmap):
        r"""Translate free-form *Type* option into validated code
        
        :Call:
            >>> bmap = db.validate_boolmap(boolmap)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
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
  # Data
  # ============
  # <
    # Class-specific class initializer
    def init_col_class(self, col, clsname):
        r"""Initialize a class-specific column
        
        This is used for special classes and should be overwritten in
        specific classes if that class has its own ``"Type"``
        definitions that are not generic.
        
        :Call:
            >>> db.init_col_class(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column to initialize
            *clsname*: :class:`str`
                Value of *Type* from *col* definition
        :Versions:
            * 2019-12-03 ``@ddalle``: First version
        """
        # Check class
        if clsname == "boolmap":
            # Initialize list of strings to remember text actually used
            self[col] = []
            # No max length
            self._n[col] = 0
            self._nmax[col] = None
        else:
            # Unreachable
            raise ValueError(
                "%s class has no special column types"
                % self.__class__.__name__)
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
        self.set_regex_linesplitter()
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
        self.cols = self.translate_colnames(cols)
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
            >>> v = db.fromtext_boolmap(txt, col)
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
            *v*: :class:`dict`\ [``True`` | ``False``]
                Flags for each flag in *col* definition
        :Versions:
            * 2019-12-02 ``@ddalle``: First version
        """
        # Get definition for column
        boolmap = self.get_col_prop(col, "Map", {})
        # Check the text
        for (colname, vals) in boolmap.items():
            # Check text vs flags values
            if txt in vals:
                self.append_colval(colname, True)
            else:
                self.append_colval(colname, False)

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

