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

# Third-party modules
import numpy as np

# CAPE modules
import cape.tnakit.typeutils as typeutils

# Local modules
from .basefile import BaseFile


# Regular expressions
regex_numeric = re.compile("\d")
regex_alpha   = re.compile("[A-z_]")

# Class for handling data from CSV files
class CSVFile(BaseFile):
    r"""Class for reading CSV files
    
    :Call:
        >>> db = CSVFile(fname, **kw)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
            CSV file interface
        *db.cols*: :class:`list`\ [:class:`str`]
            List of columns read
    :See also:
        * :class:`cape.attdb.ftypes.basefile.BaseFile`
    :Versions:
        * 2019-11-12 ``@ddalle``: First version
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
        # Save file name
        self.fname = fname
        # Process definitions
        kw = self.process_col_types(**kw)
        
        # Read file if appropriate
        if fname and typeutils.isstr(fname):
            # Read valid file
            self.read_csv(fname)
        
        
        
  # >
    
  # =============
  # Read
  # =============
  # <
   # --- Control ---
    # Reader
    def read_csv(self, fname):
        r"""Read an entire CSV file, including header
        
        :Call:
            >>> db.read_csv(fname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
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
            # Initialize columns
            self.init_cols(self.cols)
            # Loop through lines
            self.read_csv_data(f)
   
   # --- Header ---
    # Read initial comments
    def read_csv_header(self, f):
        r"""Read column names from beginning of open file
        
        :Call:
            >>> db.read_csv_header(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
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

    # Read a line as if it were a header
    def read_csv_headerline(self, f):
        r"""Read line and process column names if possible
        
        :Call:
            >>> db.read_csv_headerline(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
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
            # Non-comment row following comment: data
            f.seek(pos)
            # Mark completion of header
            self._csv_header_complete = True
            # Exit
            return
        # Save column names if reaching this point
        self.cols = cols
        # Output column names for kicks
        return cols

   # --- Data ---
    # Rad data
    def read_csv_data(self, f):
        r"""Read data portion of CSV file
        
        :Call:
            >>> db.read_csv_data(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Initialize types
        self._types = [self.get_col_type(col) for col in self.cols]
        # Read data lines
        while True:
            # Process next line
            eof = self.read_csv_dataline(f)
            # Check for end of file
            if eof == -1:
                break
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
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
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
            v = self.translate_text(coltxts[j], clsname)
            # Save data
            self.append_colval(col, v)

   # --- Translators ---
    # Convert to text to appropriate class
    def translate_text(self, txt, clsname):
        r"""Convert a string to appropriate type
        
        :Call:
            >>> v = db.translate_text(txt, clsname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
                CSV file interface
            *txt*: :class:`str`
                Text to be converted to :class:`float`
            *clsname*: {``"float64"``} | ``"int32"`` | :class:`str`
                Valid data type name
        :Outputs:
            *v*: :class:`clsname`
                Text translated to requested type
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Filter class name
        if clsname.startswith("float"):
            # Convert float
            return self.translate_float(txt, clsname)
        elif clsname.startswith("str"):
            # No conversion
            return txt
        elif clsname.startswith("int"):
            # Convert integer
            return self.translate_int(txt, clsname)
        elif clsname.startswith("uint"):
            # Convert unsigned integer
            return self.translate_int(txt, clsname)
        elif clsname.startswith("complex"):
            # Convert complex number
            return self.translate_complex(txt, clsname)
        else:
            # Invalid type
            raise TypeError("Invalid class name '%s'" % clsname)

    # Convert text to float
    def translate_float(self, txt, clsname=None):
        r"""Convert a string to float
        
        This conversion allows for the format ``"2.40D+00"`` if the
        built-in :func:`float` converter fails.  Python expects the
        exponent character to be ``E`` or ``e``, but ``D`` and ``d``
        are allowed here.  Other exceptions are not handled.
        
        Special processing of specific :class:`float` subtypes is
        handled if the *clsname* keyword is specified.  Specific types
        are handled by valid NumPy classes.
        
        :Call:
            >>> v = db.translate_float(txt)
            >>> v = db.translate_float(txt, clsname="float64")
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
                CSV file interface
            *txt*: :class:`str`
                Text to be converted to :class:`float`
            *clsname*: {``"float64"``} | ``"float32"`` | ``"float128"``
                Specific data type
        :Outputs:
            *v*: :class:`float`
                Converted value
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Filter name
        if clsname is None:
            # Standard Python type
            cls = float
        elif clsname == "float64":
            # Standard NumPy float
            cls = np.float64
        elif clsname == "float16":
            # Extra short NumPy float
            cls = np.float16
        elif clsname == "float32":
            # Single-precision
            cls = np.float32
        elif clsname == "float128":
            # Extra long
            cls = np.float128
        else:
            # Invalid
            raise ValueError("Invalid float subtype '%s'" % clsname)
        # Attempt conversion
        try:
            # Basic conversion
            return cls(txt)
        except ValueError as e:
            # Substitute "E" for "D" and "e" for "d"
            txt = txt.replace("D", "E")
            txt = txt.replace("d", "e")
        # Second attempt
        try:
            # Basic conversion after substitution
            return cls(txt)
        except Exception:
            # Use original message to avoid confusion
            raise ValueError(e.message)
    
    # Convert text to complex
    def translate_complex(self, clsname=None):
        r"""Convert a string to complex float
        
        This conversion allows for the format ``"2.40D+00 + 1.2I"``
        where ``I``, ``i``, and ``J`` are converted to ``j``; and
        ``D`` and ``d`` are converted to ``E`` if necessary.
        
        Special processing of specific :class:`complex` subtypes is
        handled if the *clsname* keyword is specified.  Specific types
        are handled by valid NumPy classes.
        
        :Call:
            >>> v = db.translate_complex(txt)
            >>> v = db.translate_complex(txt, clsname="complex128")
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
                CSV file interface
            *txt*: :class:`str`
                Text to be converted to :class:`float`
            *clsname*: {``"complex128"``} | ``"complex64"``
                Specific data type
        :Outputs:
            *v*: :class:`float`
                Converted value
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Filter name
        if clsname is None:
            # Standard Python type
            cls = complex
            clsf = float
        elif clsname == "complex128":
            # Standard NumPy float
            cls = np.complex128
            clsf = np.float64
        elif clsname == "complex64":
            # Single-precision
            cls = np.complex64
            clsf = np.float32
        elif clsname == "complex256":
            # Extra long
            cls = np.complex256
            clsf = np.float128
        else:
            # Invalid
            raise ValueError("Invalid complex number subtype '%s'" % clsname)
        # Initialize value
        v = cls(0.0)
        # Substitute "i" for "j"
        txt = txt.replace("i", "j")
        txt = txt.replace("I", "j")
        txt = txt.replace("J", "j")
        # Split text into real and imaginary parts
        txts = txt.split("+")
        # Loop through parts
        for txti in txts:
            # Check if it's complex
            if "j" in txti:
                # Get rid of it
                txti = txti.replace("j", "")
                # Convert imaginary part to float
                v += self.translate_float(txti, clsf) * 1j
            else:
                # Convert real part to float
                v += self.translate_float(txti, clsf)
        # Output
        return v

    # Convert text to int
    def translate_int(self, txt, clsname=None):
        r"""Convert a string to integer
        
        Special processing of specific :class:`int` and :class:`uint`
        subtypes is handled if the *clsname* keyword is specified.
        Specific types are handled by valid NumPy classes.
        
        :Call:
            >>> v = db.translate_float(txt)
            >>> v = db.translate_float(txt, clsname="int32")
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
                CSV file interface
            *txt*: :class:`str`
                Text to be converted to :class:`float`
            *clsname*: {``"int32"``} | ``"int64"`` | ``"uint64"``
                Specific data type
        :Outputs:
            *v*: :class:`float`
                Converted value
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Filter name
        if clsname is None:
            # Standard Python type
            cls = int
        elif clsname == "int32":
            # Standard NumPy float
            cls = np.int32
        elif clsname == "int64":
            # Extra short NumPy float
            cls = np.float64
        elif clsname == "int16":
            # Single-precision
            cls = np.int16
        elif clsname == "int8":
            # Extra long
            cls = np.int8
        elif clsname == "uint":
            # Long unsigned
            cls = np.uint32
        elif clsname == "uint32":
            # Long unsigned
            cls = np.uint32
        elif clsname == "uint64":
            # Extra long unsigned
            cls = np.uint64
        elif clsname == "uint8":
            # Extra short unsigned
            cls = np.uint8
        else:
            # Invalid
            raise ValueError("Invalid integer subtype '%s'" % clsname)
        # Attempt conversion
        return cls(txt)

