#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.ftypes.mat`: MATLAB data interface
===============================================================

This module provides a class :class:`MATFile` for reading and writing
data from files using version 5.0 of MATLAB's ``.mat`` format.  Later
versions of ``.mat`` files utilize HDF5 and are not supported here.
It relies on two third-party libraries readily available from the Python
Package Index (PyPI):

    * :mod:`scipy.io` for reading many files
    * :mod:`scipy.io.matlab.mio5_params` for MATLAB files

These can be readily installed on any machine with both Python and
access to the internet (even without elevated privileges) using the
commands below:

    .. code-block:: console

        $ pip install --user scipy

Because CAPE may also be used on machines without regular access to the
internet, this module does not raise an ``ImportError`` in the case that
these third-party modules are not available.  However, the module will
provide no functionality if these modules are not available.

"""

# Third-party modules
import numpy as np

# Quasi-optional third-party modules
try:
    import scipy.io as sio
except ImportError:
    sio = None
try:
    import scipy.io.matlab.mio5_params as siom
except ImportError:
    siom = None

# CAPE modules
import cape.tnakit.typeutils as typeutils

# Local modules
from .basefile import BaseFile


# Class for handling data from XLS files
class MATFile(BaseFile):
    r"""Class for reading ``.mat`` files (version 5)

    :Call:
        >>> db = MATFile(fname, sheet=0, **kw)
    :Inputs:
        *fname*: :class:`str`
            Name of ``.mat`` file to read
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
        * 2019-12-17 ``@ddalle``: First version
    """
    # Special class list
    _classtypes = ["boolmap"]
    # Recognized types and other defaults
    _DTypeMap = dict(BaseFile._DTypeMap, boolmap="str")

  # =============
  # Config
  # =============
  # <
    # Initialization method
    def __init__(self, fname=None, **kw):
        """Initialization method

        :Versions:
            * 2019-12-17 ``@ddalle``: First version
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
            # Read valid file
            kw = self.read_mat(fname, **kw)
        else:
            # Process inputs
            kw = self.process_col_defns(**kw)

        # Check for overrides of values
        kw = self.process_kw_values(**kw)
  # >

  # ===============
  # Read
  # ===============
  # <
    # Read MAT file
    def read_mat(self, fname, **kw):
        r"""Read a MATLAB ``.mat`` file

        The primary data is assumed to be in a variable called *DB*.

        :Call:
            >>> db.read_mat(f)
            >>> db.read_mat(fname)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.mat.MATFile`
                MAT file interface
            *f*: :class:`file`
                File open for reading (at position ``0``)
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2019-11-25 ``@ddalle``: First version
        """
        # Check modules
        _check_sio()
        # Check type
        if typeutils.isfile(fname):
            # Safe file name
            self.fname = fname.name
        else:
            # Save file name
            self.fname = fname

        # Read MAT file
        db = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
        # Get primary data interface
        DB = db.get("DB")
        # Check type
        if DB is None:
            # Nothing to do
            return
        elif not isinstance(DB, siom.mat_struct):
            # The "database" is not a struct
            raise TypeError("The 'DB' field must be a MATLAB struct")

        # Get options
        opts = self.__dict__.setdefault("opts", {})
        # Get lengths
        _n = self.__dict__.setdefault("_n", {})
        # Get definitions
        defns = opts.setdefault("Definitions", {})

        # Loop through database
        for col in DB._fieldnames:
            # Get value
            V = DB.__dict__[col]
            # Definition for this column
            defn = defns.setdefault(col, {})
            # Check type
            if not isinstance(V, (list, np.ndarray)):
                raise TypeError(
                    "Database field '%s' must be list or array" % col)
            # Process type
            if isinstance(V, list):
                # Assume string
                dtype = "str"
                # Save length
                defn["Shape"] = (len(V), )
            else:
                # Array; get data type from instance
                dtype = str(V.dtype)
                # Dimensions
                defn["Dimension"] = V.ndim
                defn["Shape"] = V.shape
            # Set type
            defn["Type"] = dtype
            # Save column
            self.save_col(col, V)

        # Save other stuff
        for (k, v) in db.items():
            # Check special names
            if k.startswith("_"):
                # Reserved for Python
                continue
            elif k == "DB":
                # Primary data
                continue
            # Otherwise save it
            self.__dict__[k] = from_matlab(v)

        # Process column definitions
        return self.process_col_defns(**kw)
  # >


  # ===============
  # Write
  # ===============
  # <
    # Write MAT file
    def write_mat(self, fname, **kw):
        r"""Write database to ``.mat`` file

        :Call:
            >>> db.write_mat(fname, **kw)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.mat.MATFile`
                MAT file interface
            *fname*: :class:`str`
                Name of ``.mat`` file to write
        :See Also:
            * :func:`create_mat`
        :Versions:
            * 2019-12-17 ``@ddalle``: First version
        """
        # Create database
        dbmat = self.create_mat(**kw)
        # Write it
        sio.savemat(fname, dbmat, oned_as="column")

    # Create MAT file
    def create_mat(self, **kw):
        r"""Create a :class:`dict` for output as ``.mat`` file

        :Call:
            >>> dbmat = db.create_mat(dbmat={})
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.mat.MATFile`
                MAT file interface
            *dbmat*: {``{}``} | :class:`dict`
                Dictionary to add contents of *db* to before writing
            *attrs*: {``None``} | :class:`list`\ [:class:`str`]
                List of additional attributes to save in *dbmat*
        :Outputs:
            *dbmat*: :class:`dict`
                Dict in format ready for :func:`sio.savemat`
            *dbmat["DB"]*: :class:`scipy.io.matlab.mat_struct`
                Struct containing primary data cols from *db*
        :Versions:
            * 2019-12-17 ``@ddalle``: First version
        """
        # Check modules
        _check_sio()
        # Check for existing interface
        dbmat = kw.get("dbmat")
        # Check type
        if (dbmat is not None) and (type(dbmat) != dict):
            # Must be a dict (exactly)
            raise TypeError("Keyword arg 'dbmat' must be a dict")
        # Create new instance if necessary
        if not dbmat:
            # Empty dict
            dbmat = {}
        # Columns to add
        cols = kw.get("cols")
        # Default list
        if cols is None:
            # Write all columns
            cols = self.cols
        # Check type
        if not isinstance(cols, (tuple, list)):
            raise TypeError("Column list 'cols' must be a list")
        # Get list of attributes
        attrs = kw.get("attrs")
        # Check type
        if attrs and not isinstance(attrs, (tuple, list)):
            # Should really be a list of strings
            raise TypeError("Extra attribute list 'attrs' must be a list")
        # Get data interface
        if "DB" in dbmat:
            # Get interface
            DB = dbmat["DB"]
            # Check type
            if not isintance(DB, siom.mat_struct):
                raise TypeError("Existing 'DB' field must be a mat_struct")
        else:
            # Create new instance
            DB = dbmat.setdefault("DB", siom.mat_struct())
            # Initialize fields
            DB._fieldnames = []
        # Loop through columns
        for col in cols:
            # Check column
            if col not in self.cols:
                raise KeyError("No data column '%s'" % col)
            # Save data to database
            DB.__dict__[col] = to_matlab(self[col])
            # Append to field list
            if col not in DB._fieldnames:
                DB._fieldnames.append(col)
        # Check for any extra attributes
        if attrs:
            # Loop through attributes
            for attr in attrs:
                # Check type and validity
                if not typeutils.isstr(attr):
                    raise TypeError("Extra attr '%s' must be a string" % attr)
                elif attr not in self.__dict__:
                    raise AttributeError("No attribute '%s' to copy" % attr)
                #Save the value
                dbmat[attr] = to_matlab(self.__dict__[attr])
        # Output
        return dbmat
  # >


# Generic converter
def from_matlab(x):
    r"""Convert a generic MATLAB object to Python

    This function recurses if necessary

        ====================  ====================
        MATLAB                Python
        ====================  ====================
        :class:`struct`       :class:`dict`
        ====================  ====================

    :Call:
        >>> v = from_matlab(x)
    :Inputs:
        *x*: :class:`any` (MATLAB)
            Item read from ``.mat`` file
    :Outputs:
        *v*: :class:`any` (Python)
            Python interpretation
    :Versions:
        * 2019-12-17 ``@ddalle``: First version
    """
    # Check modules
    _check_sio()
    # Check type
    if isinstance(x, siom.mat_struct):
        # Convert to dict
        return struct_to_dict(x)
    else:
        # No conversion
        return x


# Generic converter
def to_matlab(v):
    r"""Convert a generic MATLAB object to Python

    This function recurses if necessary

        ====================  ====================
        MATLAB                Python
        ====================  ====================
        :class:`struct`       :class:`dict`
        ====================  ====================

    :Call:
        >>> x = to_matlab(v)
    :Inputs:
        *v*: :class:`any` (Python)
            Python interpretation
    :Outputs:
        *x*: :class:`any` (MATLAB)
            Item ready for ``.mat`` file
    :Versions:
        * 2019-12-17 ``@ddalle``: First version
    """
    # Check modules
    _check_sio()
    # Check type
    if isinstance(v, dict):
        # Convert to dict
        return dict_to_struct(v)
    else:
        # No conversion
        return v


# Convert MATLAB struct to Python dict
def struct_to_dict(s):
    r"""Convert aMATLAB ``struct`` to a Python :class:`dict`

    This function is recursive if necessary.

    :Call:
        >>> d = struct_to_dict(s)
    :Inputs:
        *s*: :class:`scipy.io.matlab.mio5_params.mat_struct`
            Interface to MATLAB struct
    :Outputs:
        *d*: :class:`dict`
            Dict with keys from *s._fieldnames*
    :Versions:
        * 2019-12-17 ``@ddalle``: First version
    """
    # Check modules
    _check_sio()
    # Check type
    if not isinstance(s, siom.mat_struct):
        raise TypeError("Input must be a MATLAB struct interface")
    # Create dict
    d = {}
    # Loop through fields
    for k in s._fieldnames:
        # Get value
        V = s.__dict__[k]
        # Check type
        if isinstance(V, siom.mat_struct):
            # Recurse
            d[k] = struct_to_dict(V)
        else:
            # Save value
            d[k] = V
    # Output
    return d


# Convert Python dict to MATLAB struct
def dict_to_struct(d):
    r"""Convert a Python :class:`dict` to a MATLAB ``struct`` 

    This function is recursive if necessary.

    :Call:
        >>> s = dict_to_struct(d)
    :Inputs:
        *d*: :class:`dict`
            Dict with keys from *s._fieldnames*
    :Outputs:
        *s*: :class:`scipy.io.matlab.mio5_params.mat_struct`
            Interface to MATLAB struct
    :Versions:
        * 2019-12-17 ``@ddalle``: First version
    """
    # Check modules
    _check_sio()
    # Check type
    if not isinstance(d, dict):
        raise TypeError("Input must be a dict")
    # Create struct
    s = siom.mat_struct()
    s._fieldnames = []
    # Loop through keys
    for (k, v) in d.items():
        # Check type
        if isinstance(v, dict):
            # Recurse
            s.__dict__[k] = dict_to_struct(v)
        else:
            # Save value
            s.__dict__[k] = v
        # Append name
        if k not in s._fieldnames:
            s._fieldnames.append(k)
    # Output
    return s


# Merge two structs
def merge_structs(DB1, DB2):
    r"""Merge two MATLAB structs

    :Call:
        >>> merge_structs(DB1, DB2)
    :Inputs:
        *DB1*: :class:`scipy.io.matlab.mio5_params.mat_struct`
            Primary struct
        *DB2*: :class:`scipy.io.matlab.mio5_params.mat_struct`
            Second struct
    :Effects:
        *DB1*: :class:`scipy.io.matlab.mio5_params.mat_struct`
            Data from *DB2* added to *DB1*
    :Versions:
        * 2019-12-17 ``@ddalle``: First version
    """
    # Check modules
    _check_sio()
    # Check types
    if not isinstance(DB1, siom.mat_struct):
        raise TypeError("Input must be a MATLAB struct interface")
    if not isinstance(DB2, siom.mat_struct):
        raise TypeError("Input must be a MATLAB struct interface")
    # Loop through fields of *DB2*
    for col in DB2._fieldnames:
        # Overwrite data if needed
        DB1.__dict__[col] = DB2.__dict__[col]
        # Append to field list
        if col not in DB1._fieldnames:
            DB1._fieldnames.append(col)


# Check modules
def _check_sio():
    r"""Check if needed :mod:`scipy.io` modules are present

    :Call:
        >>> _check_sio()
    """
    # Check modules
    if sio is None:
        raise ImportError("No scipy.io module")
    if siom is None:
        raise ImportError("No scipy.io.matlab module")

