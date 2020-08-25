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
import warnings

# Third-party modules
import numpy as np

# CAPE modules
import cape.tnakit.kwutils as kwutils
import cape.tnakit.typeutils as typeutils

# Local modules
from .basedata import BaseData, BaseDataDefn, BaseDataOpts

# Fixed parameter for size of new chunks
NUM_ARRAY_CHUNK = 5000


# Options
class BaseFileOpts(BaseDataOpts):
  # ===================
  # Class Attributes
  # ===================
  # <
   # --- Global Options ---
    # List of options
    _optlist = {
        "Prefix",
        "Suffix",
        "Translators"
    }

    # Alternate names
    _optmap = {
        "prefix": "Prefix",
        "suffix": "Suffix",
        "translators": "Translators",
    }

   # --- Types ---
    # Types allowed
    _opttypes = {
        "Prefix": (typeutils.strlike, dict),
        "Suffix": (typeutils.strlike, dict),
        "Translators": dict,
    }
  # >


# Combine options with parent class
BaseFileOpts.combine_optdefs()


# Definition
class BaseFileDefn(BaseDataDefn):
    pass


# Add definition support to option
BaseFileOpts.set_defncls(BaseFileDefn)


# Declare basic class
class BaseFile(BaseData):
    r"""Generic class for storing data from a data-style file
    
    This class has no initialization method, and as such it is unlikely
    that there will be instances of this class in use.  It provides
    methods and structure to other classes.
    
    This class inherits from :class:`dict` and can be used in that
    matter in the unlikely event that it's useful.
    
    :Outputs:
        *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
            CSV file interface
        *db.cols*: :class:`list`\ [:class:`str`]
            List of columns read
        *db.opts*: :class:`dict`
            Options for this instance
        *db.defns*: :class:`dict`
            Definitions for each column/coefficient
        *db[col]*: :class:`np.ndarray` | :class:`list`
            Numeric array or list of strings for each column
    :See also:
        * :class:`cape.attdb.ftypes.csv.CSVFile`
        * :class:`cape.attdb.ftypes.csv.CSVSimple`
        * :class:`cape.attdb.ftypes.textdata.TextDataFile`
    :Versions:
        * 2019-11-26 ``@ddalle``: First version
    """
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Options ---
    # Class for options
    _optscls = BaseFileOpts
    # Definition class
    _defncls = BaseFileDefn
  # >

  # ==========
  # Config
  # ==========
  # <
    # Template initialization method
    def __init__(self, **kw):
        r"""Initialization method

        :Versions:
            * 2020-02-02 ``@ddalle``: First version
        """
        # Initialize columns
        self.cols = []
        # Process options
        self.opts = self.process_kw(**kw)
        # Ensure definitions are present
        self.get_defns()
        # Apply defaults to definitions
        self.finish_defns()
        # Process values
        self.process_kw_values()
  # >
  
  # =================
  # Data Columns
  # =================
  # <
   # --- Name Translation ---
    # Translate column names
    def translate_colnames(self, cols):
        r"""Translate column names

        This method utilizes the options *Translators*, *Prefix*, and
        *Suffix* from the *db.opts* dictionary. The *Translators* are
        applied before *Prefix* and *Suffix*.

        :Call:
            >>> dbcols = db.translate_colnames(cols)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *cols*: :class:`list`\ [:class:`str`]
                List of "original" column names, e.g. from file
        :Outputs:
            *dbcols*: :class:`list`\ [:class:`str`]
                List of column names as stored in *db*
        :Versions:
            * 2019-12-04 ``@ddalle``: First version
        """
        # Get options
        trans  = self.get_option("Translators", {})
        prefix = self.get_option("Prefix", "")
        suffix = self.get_option("Suffix", "")
        # Call private function
        return self._translate_colnames(cols, trans, prefix, suffix)

    # Reverse translation of column names
    def translate_colnames_reverse(self, dbcols):
        r"""Reverse translation of column names

        This method utilizes the options *Translators*, *Prefix*, and
        *Suffix* from the *db.opts* dictionary.*Prefix* and *Suffix*
        removed before reverse translation.

        :Call:
            >>> cols = db.translate_colnames_reverse(dbcols)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *dbcols*: :class:`list`\ [:class:`str`]
                List of raw column names as stored in *db*
        :Outputs:
            *cols*: :class:`list`\ [:class:`str`]
                List of "original" column names, e.g. from file
        :Versions:
            * 2019-12-04 ``@ddalle``: First version
            * 2019-12-11 ``@jmeeroff``: From :func:`translate_colnames`
        """
        # Get options
        trans  = self.get_option("Translators", {})
        prefix = self.get_option("Prefix", "")
        suffix = self.get_option("Suffix", "")
        # Call private function
        return self._translate_colnames_reverse(dbcols, trans, prefix, suffix)

   # --- Init ---
    # Initialize list of columns
    def init_cols(self, cols):
        r"""Initialize list of columns
        
        :Call:
            >>> db.init_cols(cols)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column to initialize
        :See Also:
            * :func:`init_col`
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Loop through columns
        for col in cols:
            # Initialize column
            self.init_col(col)

    # Initialize single column
    def init_col(self, col):
        r"""Initialize column
        
        :Call:
            >>> db.init_col(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column to initialize
        :Effects:
            *db[col]*: :class:`np.ndarray` | :class:`list`
                Initialized array with appropriate type
            *db._n[col]*: ``0``
                Number of entries saved to *db[col]*
            *db._nmax[col]*: ``None`` | :class:`int`
                Number of entries allocated, if appropriate
        :Versions:
            * 2019-11-23 ``@ddalle``: First version
            * 2019-12-03 ``@ddalle``: Added :func:`init_col_class`
        """
        # Check validity
        if col not in self.cols:
            raise KeyError("Unrecognized column '%s'" % col)
        # Convert special type to actual Python type, if necessary
        dtype = self.get_col_dtype(col)
        # Make sure _nmax (array length) attribute is present
        if not hasattr(self, "_nmax"):
            self._nmax = {}
        # Make sure _n (current length) attribute is present
        if not hasattr(self, "_n"):
            self._n = {}
        # Check for string
        if dtype == "str":
            # Initialize strings in empty list
            self[col] = []
            # No max length
            self._n[col] = 0
            self._nmax[col] = None
        else:
            # Use existing dtype code
            self[col] = np.zeros(NUM_ARRAY_CHUNK, dtype=dtype)
            # Set max length
            self._n[col] = 0
            self._nmax[col] = NUM_ARRAY_CHUNK

    # Class-specific class initializer
    def init_col_class(self, col):
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
        :Versions:
            * 2019-12-03 ``@ddalle``: First version
        """
        raise ValueError(
            "%s class has no special column types" % self.__class__.__name__)

   # --- Save Data ---
    # Save next value to column's array
    def append_colval(self, col, v):
        """Save the next value to a column's array or list
        
        This will update counts and allocate a new chunk if necessary.
        
        :Call:
            >>> db.init_col(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column to which to save value
            *v*: ``db.get_col_type(col)``
                Value to save to array/list
        :Effects:
            *db[col]*: :class:`np.ndarray` | :class:`list`
                Column's array with extra new entry
            *db._n[col]*: :class:`int`
                Updated length of array/list
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Check NMAX attribute to check process
        nmax = self._nmax.get(col)
        # Get current count
        n = self._n[col]
        # Process options
        if nmax is None:
            # It's a list; just append
            self[col].append(v)
        elif n >= nmax:
            # Get dtype
            dtype = self.get_col_dtype(col)
            # Allocate new chunk
            self[col] = np.hstack(
                (self[col], np.zeros(NUM_ARRAY_CHUNK, dtype=dtype)))
            # Update maximum
            self._nmax[col] += NUM_ARRAY_CHUNK
            # Save new value
            self[col][n] = v
        else:
            # Save new value without new allocation
            self[col][n] = v
        # Update count
        self._n[col] = n + 1

    # Trim columns
    def trim_colarray(self, col):
        r"""Trim extra entries from data rows
        
        :Call:
            >>> db.trim_colarray(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column to which to save value
        :Effects:
            *db[col]*: :class:`np.ndarray` | :class:`list`
                Trimmed to length *db._n[col]* if an array
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Check NMAX attribute to check process
        nmax = self._nmax.get(col)
        # Get current count
        n = self._n.get(col)
        # Check for invalid length
        if not isinstance(n, int):
            raise TypeError("No valid length to trim column '%s'" % col)
        # Process options
        if nmax is None:
            # No trimming needed
            return
        else:
            # Trim the array
            self[col] = self[col][:n]
            # Trim *nmax*
            self._nmax[col] = n
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


# Text interpretation classes
class TextInterpreter(object):
    r"""Class to contain methods for interpreting text
    
    The class is kept separate from :class:`BaseFile` because not all
    file-type interfaces need sophisticated rules for converting text
    to numeric or other values.
    
    This class provides several methods for inheritance, but the intent
    is that instances of this class are not useful and should not be
    used.
    
    :Versions:
        * 2019-11-26 ``@ddalle``: First version
        * 2019-12-02 ``@ddalle``: Changed from :class:`TextFile`
    """
    # Convert to text to appropriate class
    def fromtext_val(self, txt, clsname):
        r"""Convert a string to appropriate type
        
        :Call:
            >>> v = db.fromtext_val(txt, clsname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
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
        return self.fromtext_base(txt, clsname)
        
    # Convert to text to appropriate class
    def fromtext_base(self, txt, clsname):
        r"""Convert a string to appropriate numeric/string type
        
        :Call:
            >>> v = db.fromtext_num(txt, clsname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
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
            return self.fromtext_float(txt, clsname)
        elif clsname.startswith("str"):
            # No conversion
            return txt
        elif clsname.startswith("int"):
            # Convert integer
            return self.fromtext_int(txt, clsname)
        elif clsname.startswith("uint"):
            # Convert unsigned integer
            return self.fromtext_int(txt, clsname)
        elif clsname.startswith("complex"):
            # Convert complex number
            return self.fromtext_complex(txt, clsname)
        else:
            # Invalid type
            raise TypeError("Invalid class name '%s'" % clsname)

    # Convert text to float
    def fromtext_float(self, txt, clsname=None):
        r"""Convert a string to float
        
        This conversion allows for the format ``"2.40D+00"`` if the
        built-in :func:`float` converter fails.  Python expects the
        exponent character to be ``E`` or ``e``, but ``D`` and ``d``
        are allowed here.  Other exceptions are not handled.
        
        Special processing of specific :class:`float` subtypes is
        handled if the *clsname* keyword is specified.  Specific types
        are handled by valid NumPy classes.
        
        :Call:
            >>> v = db.fromtext_float(txt)
            >>> v = db.fromtext_float(txt, clsname="float64")
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
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
        # Check for NaN (empty text)
        if txt.strip() == "":
            return np.nan
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
        except ValueError:
            # Substitute "E" for "D" and "e" for "d"
            txt = txt.replace("D", "E")
            txt = txt.replace("d", "e")
        # Second attempt
        return cls(txt)
    
    # Convert text to complex
    def fromtext_complex(self, txt, clsname=None):
        r"""Convert a string to complex float
        
        This conversion allows for the format ``"2.40D+00 + 1.2I"``
        where ``I``, ``i``, and ``J`` are converted to ``j``; and
        ``D`` and ``d`` are converted to ``E`` if necessary.
        
        Special processing of specific :class:`complex` subtypes is
        handled if the *clsname* keyword is specified.  Specific types
        are handled by valid NumPy classes.
        
        :Call:
            >>> v = db.fromtext_complex(txt)
            >>> v = db.fromtext_complex(txt, clsname="complex128")
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
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
            clsf = "float64"
        elif clsname == "complex128":
            # Standard NumPy float
            cls = np.complex128
            clsf = "float64"
        elif clsname == "complex64":
            # Single-precision
            cls = np.complex64
            clsf = "float32"
        elif clsname == "complex256":
            # Extra long
            cls = np.complex256
            clsf = "float128"
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
                v += self.fromtext_float(txti, clsf) * 1j
            else:
                # Convert real part to float
                v += self.fromtext_float(txti, clsf)
        # Output
        return v

    # Convert text to int
    def fromtext_int(self, txt, clsname=None):
        r"""Convert a string to integer
        
        Special processing of specific :class:`int` and :class:`uint`
        subtypes is handled if the *clsname* keyword is specified.
        Specific types are handled by valid NumPy classes.
        
        :Call:
            >>> v = db.fromtext_float(txt)
            >>> v = db.fromtext_float(txt, clsname="int32")
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
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
        # Check for NaN (empty text)
        if txt.strip() == "":
            return 0
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
