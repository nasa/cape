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
    _optlist = set.union(BaseDataOpts._optlist,
        {
            "Prefix",
            "Suffix",
            "Translators"
        }
    )

    # Alternate names
    _optmap = dict(BaseDataOpts._optmap,
        prefix="Prefix",
        suffix="Suffix",
        translators="Translators")

   # --- Types ---
    # Types allowed
    _opttypes = dict(BaseDataOpts._opttypes,
        Prefix=(str, dict),
        Suffix=(str, dict),
        Translators=(dict))
  # >


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
        *db.opts["Definitions"]*: :class:`dict`
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
    _optsclass = BaseDataOpts
  # >

  # ==========
  # Config
  # ==========
  # <
  # >
  
  # =================
  # Options
  # =================
  # <
   # --- Key Definitions ---
    # Process generic options
    def process_opts_generic(self, **kw):
        r"""Process generic options from keyword arguments

        :Call:
            >>> kwo = db.process_opts_generic(**kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *cols*, *ColNames*, *Keys*: :class:`list`\ [:class:`str`]
                User-specified column names
            *ExpandScalars*: {``True``} | ``False``
                Option to expand any scalar inputs into arrays that
                match other columns (with *size* equal to *db.n*)
            *[Tt]ranslators*: :class:`dict`\ [:class:`str`]
                Dictionary of alternate names to store column names as;
                for example if the header has a column called
                ``"CAF"``, and ``translators["CAF"]`` is ``"CA"``, that
                column will be stored as ``db["CA"]`` instead of
                ``db["CAF"]``
            *[Pp]refix*: ``""`` | :class:`str` | :class:`dict`
                Universal column name prefix or dictionary thereof
            *[Ss]uffix*: ``""`` | :class:`str` | :class:`dict`
                Universal column name suffix or dictionary thereof
        :Outputs:
            *kwo*: :class:`dict`
                Options not used in this method
        :Versions:
            * 2019-11-27 ``@ddalle``: First version
        """
        # Process options
        kwo = self.check_kw_types(1, **kw)
        # Get columns
        cols = kwo.get("cols", None)
        # Save column list if appropriate
        if isinstance(cols, list):
            self.cols = cols
        # Save translators
        trans = kwo.get("Translators", {})
        # Save
        self.opts["Translators"] = trans
        # Prefix/suffix options
        prefix = kwo.get("Prefix", "")
        suffix = kwo.get("Suffix", "")
        # De-none
        if prefix is None:
            prefix = ""
        if suffix is None:
            suffix = ""
        # Save prefix and suffix
        self.opts["Prefix"] = prefix
        self.opts["Suffix"] = suffix
        # Values option
        if kwo.get("ExpandScalars", True):
            # Something true-like
            self.opts["ExpandScalars"] = True
        else:
            # Turned off
            self.opts["ExpandScalars"] = False
            
        # Return unused options
        return kwo

   # --- Columns ---
    # Process key definitions
    def process_col_defns(self, **kw):
        r"""Process *Definitions* of column types
        
        :Call:
            >>> db.process_col_defns(**kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
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
        # Get options for key definitions
        defns1 = kw.get("defns", {})
        defns2 = kw.get("Definitions", {})
        # Combine definitions
        defns = dict(defns1, **defns2)
        # Get default definition from class definition
        clsdefn = self.__class__._DefaultDefn
        # Check for default definition
        odefn = kw.get("DefaultDefinition", {})
        # Process each option from class definition
        for (k, opt) in clsdefn.items():
            # Prepend "Default" to the name
            key = "Default" + k
            # Check for option
            odefk = kw.get(key, opt)
            # Save it if appropriate
            odefn.setdefault(k, odefk)
        # Validate default definition
        self.validate_defn(odefn)
        # Ensure definitions exist
        opts = self.get_defns()
        # Save defaults
        opts["_"] = odefn

        # Loop through columns mentioned in input
        for (col, kwdefn) in defns.items():
            # Get existing definition
            defn = opts.setdefault(col, {})
            # Apply keyword definitions
            for (key, opt) in kwdefn.items():
                # Kwargs override anything created automatically
                defn[key] = opt
            # Validate values
            self.validate_defn(defn)

        # Loop through options specific to individual keys
        for k in clsdefn.keys():
            # Keyword argument name appends an "s"
            defnk = k + "s"
            # Check if present
            if defnk not in kw:
                continue
            # Check the type
            if not isinstance(kw[defnk], dict):
                continue
            # Get the option
            defnkw = kw.get(defnk)
            # Loop through cols affected by this dictionary
            for (col, opt) in defnkw.items():
                # Get existing definition
                defn = opts.setdefault(col, {})
                # Apply the type
                defn[k] = self.validate_type(opt)

        # Loop through known columns
        for col in self.cols:
            # Get definition
            defn = defns.setdefault(col, {})
            # Get default definition based on column name
            odefncol = self.get_col_defaultdefn(col)
            # Loop through default keys
            for key, opt in odefncol.items():
                # Apply default but don't override
                defn.setdefault(key, opt)
            # Validate the definition
            self.validate_defn(defn)

    # Get default definition based on column name
    def get_col_defaultdefn(self, col):
        r"""Get the default definition based on a column name
        
        :Call:
            >>> odefn = db.get_col_defaultdefn(col)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *col*: :class:`str`
                Name of column to process
        :Outputs:
            *odefn*: :class:`dict`
                Dictionary of default parameters that might be specific
                to the column name
        :Versions:
            * 2019-12-05 ``@ddalle``: First version
        """
        # Get class
        cls = self.__class__
        # Get class's default definitions for each family
        defndict = cls._DefaultRoleDefns
        # Get map from name to family
        rolemap = cls._RoleMap
        # Get global default
        odefn = cls._DefaultDefn
        # Loop through roles
        for role, names in rolemap.items():
            # Check type
            if not isinstance(names, list):
                # Create list
                names = [names]
            # Check for a match
            if col in names:
                # Get definition for that role
                coldefn = defndict.get(role, odefn)
                # Set role
                return dict(coldefn, Role=role)
        # If reaching this point, return global default
        return odefn
        
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
                List of raw column names
        :Outputs:
            *dbcols*: :class:`list`\ [:class:`str`]
                Prepended and appended column names with substitutions
                if appropriate
        :Versions:
            * 2019-12-04 ``@ddalle``: First version
        """
        # Get options
        trans  = self.opts.get("Translators", {})
        prefix = self.opts.get("Prefix", "")
        suffix = self.opts.get("Suffix", "")
        # Initialize output
        dbcols = []
        # Output
        for col in cols:
            # Get substitution (default is no substitution)
            col = trans.get(col, col)
            # Get prefix
            if isinstance(prefix, dict):
                # Get specific prefix
                pre = prefix.get(col, prefix.get("_", ""))
            elif prefix:
                # Universal prefix
                pre = prefix
            else:
                # No prefix (type-safe)
                pre = ""
            # Get suffix
            if isinstance(suffix, dict):
                # Get specific suffix
                suf = suffix.get(col, suffix.get("_", ""))
            elif suffix:
                # Universal suffix
                suf = suffix
            else:
                # No suffix (type-safe)
                suf = ""
            # Combine appendages
            dbcols.append(pre + col + suf)
        # Output
        return dbcols

    # Reverse translation of column names
    def translate_colnames_reverse(self, dbcols):
        r"""Reverse translation of column names

        This method utilizes the options *Translators*, *Prefix*, and
        *Suffix* from the *db.opts* dictionary.*Prefix* and *Suffix*
        removed before reverse translation.

        :Call:
            >>> dbcols = db.translate_colnames_reverse(cols)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *dbcols*: :class:`list`\ [:class:`str`]
                List of raw column names as stored in *db*
        :Outputs:
            *cols*: :class:`list`\ [:class:`str`]
                List of column names as written to file
        :Versions:
            * 2019-12-04 ``@ddalle``: First version
            * 2019-12-11 ``@jmeeroff``: From :func:`translate_colnames`
        """
        # Get options
        trans  = self.opts.get("Translators", {})
        prefix = self.opts.get("Prefix", "")
        suffix = self.opts.get("Suffix", "")
        # Initialize output
        cols = []
        # Reverse the translation dictionary
        transr = {dbcol: col for (col, dbcol) in trans.items()}
        # Prepare output
        for dbcol in dbcols:
            # First strip prefix
            # Get prefix
            if isinstance(prefix, dict):
                # Get specific prefix
                pre = prefix.get(dbcol, prefix.get("_", ""))
            elif prefix:
                # Universal prefix
                pre = prefix
            else:
                # No prefix (type-safe)
                pre = ""
            # Check if col starts with specified prefix
            if pre and dbcol.startswith(pre):
                # Get length of prefix
                lval = len(pre)
                # Strip of prefix
                dbcol = dbcol[lval:]
            # Now strip suffix
            # Get suffix
            if isinstance(suffix, dict):
                # Get specific suffix
                suf = suffix.get(dbcol, suffix.get("_", ""))
            elif suffix:
                # Universal suffix
                suf = suffix
            else:
                # No suffix (type-safe)
                suf = ""
            # Check if column name ends with suffix
            if suf and dbcol.endsiwth(suf):
                # Get length of suf
                lval = len(suf)
                # Strip of prefix
                dbcol = dbcol[:-lval]
            # Perform the translation
            col = transr.get(dbcol, dbcol)
            # Append to output
            cols.append(col)
        # Output
        return cols

   # --- Keyword Checkers ---
    # Validate a dictionary of options
    def validate_defn(self, defn):
        r"""Validate each key in a dictionary column definition
        
        :Call:
            >>> db.validate_defn(defn)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *defn*: :class:`dict`
                Name of column definition option to validate
        :Effects:
            *defn*: :class:`dict`
                Each item in *defn* is validated
        :See Also:
            * :func:`validate_defnopt`
        :Versions:
            * 2019-11-26 ``@ddalle``: First version
        """
        # Ensure input type
        if not isinstance(defn, dict):
            raise TypeError(
                ("Valid col definition must be 'dict'") +
                ("; got '%s'" % defn.__class__))
        # Loop through keys
        for (k, v) in defn.items():
            # Validate individual key
            defn[k] = self.validate_defnopt(k, v)
        
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
            return self.validate_type_base(val)
        else:
            # Default is to accept any input
            return val

    # Convert *Type* to validated *Type*
    def validate_type(self, clsname):
        r"""Translate free-form type name into type code
        
        :Call:
            >>> dtype = db.validate_type(clsname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *clsname*: :class:`str`
                Free-form *Type* option for a column
        :Outputs:
            *dtype*: ``"f64"`` | ``"i32"`` | ``"str"`` | :class:`str`
                Name of data type
        :Versions:
            * 2019-12-03 ``@ddalle``: First version
        """
        return self.validate_type_base(clsname)

    # Convert *Type* to validated *Type*
    def validate_type_base(self, clsname):
        r"""Translate free-form type name into type code
        
        :Call:
            >>> dtype = db.validate_type_base(clsname)
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
        """
        # Force lower case
        clsname = clsname.lower()
        # Make some substitutions
        clsname = clsname.replace("float", "f")
        clsname = clsname.replace("int",  "i")
        clsname = clsname.replace("complex", "c")
        # Filter it
        if clsname in ["f", "f64", "double"]:
            # 64-bit float (default
            return "float64"
        elif clsname in ["i", "i32", "long"]:
            # 32-bit int
            return "int32"
        elif clsname in ["i16", "short"]:
            # 16-bit int
            return "int16"
        elif clsname in ["f32", "single"]:
            # 32-bit float
            return "float32"
        elif clsname in ["i64", "long long"]:
            # Double long integer
            return "int64"
        elif clsname in ["f128"]:
            # Double long float
            return "float128"
        elif clsname in ["f16"]:
            # Short float
            return "float16"
        elif clsname in ["i8"]:
            # Extra short integer
            return "int8"
        elif clsname in ["ui8"]:
            # Extra short unsigned
            return "uint8"
        elif clsname in ["ui16"]:
            # Short unsigned
            return "uint16"
        elif clsname in ["ui32"]:
            # Long unsigned
            return "uint32"
        elif clsname in ["ui64"]:
            # Long long unsigned
            return "uint64"
        elif clsname in ["i1", "bool"]:
            # Boolean
            return "bool"
        elif clsname in ["c", "c128"]:
            # Complex (double)
            return "complex128"
        elif clsname in ["c", "c64"]:
            # Complex (double)
            return "complex64"
        elif clsname in ["c256"]:
            # Complex (single)
            return "complex256"
        elif clsname in ["str"]:
            # String
            return "str"
        else:
            # Unrecognized
            return TypeError("Unrecognized class/type '%s'" % clsname)
  # >

  # ===============
  # Data
  # ===============
  # <
   # --- Init ---
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
        # Get type
        coltyp = self.get_col_type(col)
        # Convert special type to actual Python type, if necessary
        colcls = self._DTypeMap.get(coltyp, coltyp)
        # Make sure _nmax (array length) attribute is present
        if not hasattr(self, "_nmax"):
            self._nmax = {}
        # Make sure _n (current length) attribute is present
        if not hasattr(self, "_n"):
            self._n = {}
        # Check for string
        if colcls == "str":
            # Initialize strings in empty list
            self[col] = []
            # No max length
            self._n[col] = 0
            self._nmax[col] = None
        else:
            # Use existing dtype code
            self[col] = np.zeros(NUM_ARRAY_CHUNK, dtype=colcls)
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
            clsname = self.get_col_type(col)
            # Allocate new chunk
            self[col] = np.hstack(
                (self[col], np.zeros(NUM_ARRAY_CHUNK, dtype=clsname)))
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
# class BaseFile


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
# class TextFile
